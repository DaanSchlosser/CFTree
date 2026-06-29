# Copyright (C) 2025 Noah Alting
# Licensed under the GNU General Public License v3.0
# See the LICENSE file for more details.

# scripts/bench_morphometrics.py
"""Benchmark and validate the morphometrics hot path (r50 + porosity).

This is the safety net for the GPU acceleration in
``src/reconstruction/gpu_metrics.py``. It does two things on a sample of tree
crowns:

1. times the CPU path (embree ``mesh.contains`` + scipy ``cKDTree``) against the
   GPU path (Warp winding number + cuPy brute-force), so you can see the real
   r50-vs-porosity split and the GPU speedup on the target machine;
2. diffs the GPU result against the CPU result per tree, so a numeric drift in
   r50 or porosity shows up before the GPU path is enabled in a real run.

Run it on the GPU machine (inside the cftree env or the container):

    # synthetic crowns, no pipeline data needed -- validates correctness + speed
    python -m scripts.bench_morphometrics --synthetic --n-trees 50

    # real cached crowns: a directory holding matching <stem>.ply (crown mesh)
    # and <stem>.xyz (vegetation points) pairs
    python -m scripts.bench_morphometrics --trees-dir data/<case>/... --limit 100

The CPU path is always run (it is the baseline). The GPU path is run only when
Warp + cuPy + a CUDA device are present; otherwise the harness reports the CPU
timings alone, which still gives the r50-vs-porosity split.

Note on the alpha-wrap fraction: this harness measures the two morphometrics,
which a prior study put at ~80% of per-tree time. The remaining alpha-wrap +
I/O fraction is most cheaply seen by running one real tile with and without
``--geometry-only`` and comparing the stage timings in the run log.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

# Allow `python scripts/bench_morphometrics.py` as well as `-m scripts....`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.reconstruction import gpu_metrics  # noqa: E402
from src.reconstruction.extract_tree_metrics import _compute_porosity, _compute_r50  # noqa: E402


def _synthetic_tree(rng: np.random.Generator, radius: float = 2.5) -> tuple[trimesh.Trimesh, np.ndarray]:
    """A watertight crown proxy and a plausible vegetation point cloud.

    The crown is an icosphere (watertight + 2-manifold, like an alpha-wrap), and
    the points are a noisy sample inside it, so r50 and porosity are well-defined.
    """
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    n = int(rng.integers(400, 2000))
    # Points roughly filling the sphere, with surface bias like real crowns.
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    r = radius * rng.uniform(0.2, 1.0, size=(n, 1)) ** 0.5
    pts = dirs * r + rng.normal(scale=0.05, size=(n, 3))
    return mesh, pts.astype(np.float64)


def _load_pair(ply: Path) -> tuple[trimesh.Trimesh, np.ndarray] | None:
    """Load a crown mesh and its matching point cloud (<stem>.xyz / .npy)."""
    mesh = trimesh.load(ply, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        return None
    for ext in (".xyz", ".npy"):
        pts_path = ply.with_suffix(ext)
        if pts_path.is_file():
            pts = np.load(pts_path) if ext == ".npy" else np.loadtxt(pts_path, usecols=(0, 1, 2))
            return mesh, np.asarray(pts, dtype=np.float64)
    return None


def _trees(args: argparse.Namespace):
    if args.synthetic:
        rng = np.random.default_rng(args.seed)
        for _ in range(args.n_trees):
            yield _synthetic_tree(rng)
        return
    plys = sorted(Path(args.trees_dir).rglob("*.ply"))[: args.limit]
    if not plys:
        raise SystemExit(f"No .ply crown meshes found under {args.trees_dir}")
    for ply in plys:
        pair = _load_pair(ply)
        if pair is not None:
            yield pair


def _pipeline_metrics(mesh, pts, *, use_gpu: bool) -> tuple[float, float, float, float]:
    """Return (r50, porosity, r50_seconds, porosity_seconds) in pipeline order."""
    t0 = time.perf_counter()
    if use_gpu:
        r50 = gpu_metrics.gpu_r50(mesh, pts)
        r50 = _compute_r50(mesh, pts) if r50 is None else r50
    else:
        r50 = _compute_r50(mesh, pts)
    t1 = time.perf_counter()
    voxel_size = r50 * 0.8 if np.isfinite(r50) and r50 > 0 else 0.25
    if use_gpu:
        por = gpu_metrics.gpu_porosity(mesh, pts, voxel_size=voxel_size)
        por = _compute_porosity(mesh, pts, voxel_size=voxel_size) if por is None else por
    else:
        por = _compute_porosity(mesh, pts, voxel_size=voxel_size)
    t2 = time.perf_counter()
    return float(r50), float(por), t1 - t0, t2 - t1


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark and validate r50 + porosity (CPU vs GPU).")
    ap.add_argument("--synthetic", action="store_true", help="Use synthetic crowns (no pipeline data needed).")
    ap.add_argument("--n-trees", type=int, default=50, help="Number of synthetic trees (with --synthetic).")
    ap.add_argument("--trees-dir", type=str, default=None, help="Directory of <stem>.ply + <stem>.xyz tree pairs.")
    ap.add_argument("--limit", type=int, default=200, help="Max real trees to load from --trees-dir.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for synthetic trees.")
    ap.add_argument("--r50-tol", type=float, default=0.02, help="Allowed |GPU-CPU| r50 difference in metres.")
    ap.add_argument("--porosity-tol", type=float, default=0.02, help="Allowed |GPU-CPU| porosity difference.")
    args = ap.parse_args()

    if not args.synthetic and not args.trees_dir:
        ap.error("pass --synthetic or --trees-dir")

    gpu_on = gpu_metrics.gpu_available()
    print(f"GPU available: {gpu_on}  (Warp+cuPy+CUDA device)")
    print(f"{'tree':>5} | {'r50_cpu':>8} {'r50_gpu':>8} {'dr50':>7} | "
          f"{'por_cpu':>8} {'por_gpu':>8} {'dpor':>7} | {'cpu_s':>7} {'gpu_s':>7}")

    cpu_r50_t = cpu_por_t = gpu_r50_t = gpu_por_t = 0.0
    n = r50_fail = por_fail = 0
    for idx, (mesh, pts) in enumerate(_trees(args)):
        r50_c, por_c, r50_ct, por_ct = _pipeline_metrics(mesh, pts, use_gpu=False)
        cpu_r50_t += r50_ct
        cpu_por_t += por_ct
        if gpu_on:
            r50_g, por_g, r50_gt, por_gt = _pipeline_metrics(mesh, pts, use_gpu=True)
            gpu_r50_t += r50_gt
            gpu_por_t += por_gt
            dr50 = abs(r50_g - r50_c) if np.isfinite(r50_g) and np.isfinite(r50_c) else float("nan")
            dpor = abs(por_g - por_c) if np.isfinite(por_g) and np.isfinite(por_c) else float("nan")
            if np.isfinite(dr50) and dr50 > args.r50_tol:
                r50_fail += 1
            if np.isfinite(dpor) and dpor > args.porosity_tol:
                por_fail += 1
            print(f"{idx:>5} | {r50_c:8.4f} {r50_g:8.4f} {dr50:7.4f} | "
                  f"{por_c:8.4f} {por_g:8.4f} {dpor:7.4f} | {r50_ct + por_ct:7.3f} {r50_gt + por_gt:7.3f}")
        else:
            print(f"{idx:>5} | {r50_c:8.4f} {'-':>8} {'-':>7} | {por_c:8.4f} {'-':>8} {'-':>7} | "
                  f"{r50_ct + por_ct:7.3f} {'-':>7}")
        n += 1

    if n == 0:
        raise SystemExit("No trees processed.")

    print("\n--- summary over", n, "trees ---")
    print(f"CPU: r50 {cpu_r50_t:.2f}s  porosity {cpu_por_t:.2f}s  "
          f"total {cpu_r50_t + cpu_por_t:.2f}s  (r50 {100 * cpu_r50_t / (cpu_r50_t + cpu_por_t):.0f}% / "
          f"porosity {100 * cpu_por_t / (cpu_r50_t + cpu_por_t):.0f}%)")
    if gpu_on:
        gpu_total = gpu_r50_t + gpu_por_t
        cpu_total = cpu_r50_t + cpu_por_t
        print(f"GPU: r50 {gpu_r50_t:.2f}s  porosity {gpu_por_t:.2f}s  total {gpu_total:.2f}s")
        if gpu_total > 0:
            print(f"Speedup: r50 {cpu_r50_t / gpu_r50_t:.1f}x  porosity {cpu_por_t / gpu_por_t:.1f}x  "
                  f"overall {cpu_total / gpu_total:.1f}x")
        print(f"Validation: r50 within {args.r50_tol} m on {n - r50_fail}/{n}; "
              f"porosity within {args.porosity_tol} on {n - por_fail}/{n}")
        if r50_fail or por_fail:
            print("FAIL: GPU output drifted beyond tolerance on some trees -- do not enable CFTREE_GPU_METRICS yet.")
            return 1
        print("PASS: GPU output matches CPU within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
