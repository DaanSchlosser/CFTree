#!/usr/bin/env python3
"""In-image smoke test for the CFTree Docker image.

Run inside the built image, where the entrypoint has activated the conda
environment:

    docker run --rm cftree:local python /opt/cftree/docker/smoke_test.py

The test covers what a Docker build adds over a plain source checkout, and it
does so with no network download and no committed point-cloud fixture, so it is
fast and cannot flake on an external server. It confirms that the baked conda
environment imports the geospatial and scientific stack, that both compiled C++
binaries resolve through ``CFTREE_BIN``, and that each binary runs to completion
on a tiny synthetic point cloud and writes well-formed output. The last point is
the one a unit test cannot reach: a binary that compiled but cannot load its
shared libraries at runtime (the activation and ``LD_LIBRARY_PATH`` wiring the
image relies on), or one the GCC/CGAL build miscompiled, fails here rather than
in a colleague's first real run.

Exit code is 0 only when every check passes; any failure exits 1 with a reason.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

# This script lives at <repo>/docker/smoke_test.py, so the repo root (where the
# `src` package is importable) is two levels up. It is prepended to sys.path
# because the script may be launched from any working directory (the image's
# WORKDIR is /work, an empty bind-mount point), and importing src.config below
# would otherwise fail.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402  (import after the sys.path fix above)

from src.config import resolve_native_binary  # noqa: E402

SEG_BINARY = "src/segmentation/TreeSeparation/build/segmentation"
AWRAP_BINARY = "src/reconstruction/AlphaWrap/build/awrap_points"


def _resolve_binary(default_relpath: str) -> Path:
    """Resolve a baked binary through the same code path the pipeline uses.

    ``resolve_native_binary`` honours ``CFTREE_BIN``, which the image sets to the
    fixed bake path outside the working tree, so this also checks that override
    rather than only the default ``build/`` location.
    """
    exe = resolve_native_binary(REPO_ROOT / default_relpath)
    if not exe.exists():
        raise FileNotFoundError(f"binary not found at {exe} (expected under CFTREE_BIN)")
    return exe


def check_imports() -> None:
    """The baked environment imports the geospatial and scientific stack."""
    import geopandas  # noqa: F401
    import laspy  # noqa: F401
    import pdal  # noqa: F401
    import rasterio  # noqa: F401
    import scipy  # noqa: F401
    import shapely  # noqa: F401


def check_pipeline_help() -> None:
    """The entrypoint activates the env and main.py runs far enough to parse args."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "main.py"), "--help"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"main.py --help exited {result.returncode}: {result.stderr.strip()[:400]}")
    if "CFTree" not in result.stdout:
        raise RuntimeError("main.py --help did not print the expected usage banner")


def _synthetic_tree(cx: float, cy: float, rng: np.random.Generator) -> np.ndarray:
    """A stem plus a crown shell at (cx, cy), shaped like a single tree."""
    z = np.linspace(0.0, 4.0, 60)
    stem = np.column_stack([cx + rng.normal(0.0, 0.05, z.size), cy + rng.normal(0.0, 0.05, z.size), z])
    shell = rng.normal(size=(400, 3))
    shell /= np.linalg.norm(shell, axis=1, keepdims=True)
    crown = shell * 2.0 + np.array([cx, cy, 6.0])
    return np.vstack([stem, crown])


def check_segmentation() -> None:
    """The TreeSeparation binary clusters a synthetic two-tree cloud."""
    exe = _resolve_binary(SEG_BINARY)
    rng = np.random.default_rng(0)
    pts = np.vstack([_synthetic_tree(0.0, 0.0, rng), _synthetic_tree(12.0, 12.0, rng)])
    with tempfile.TemporaryDirectory() as tmp:
        in_xyz = Path(tmp) / "vegetation.xyz"
        out_xyz = Path(tmp) / "segmentation.xyz"
        np.savetxt(in_xyz, pts, fmt="%.6f")
        result = subprocess.run(
            [str(exe), str(in_xyz), str(out_xyz), "2.5", "1.5", "3"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"exit {result.returncode}: {result.stderr.strip()[:400]}")
        if not out_xyz.exists() or out_xyz.stat().st_size == 0:
            raise RuntimeError("produced no segmentation output")
        seg = np.loadtxt(out_xyz, ndmin=2)
        if seg.shape[1] != 4:
            raise RuntimeError(f"expected 4-column 'tid x y z' output, got shape {seg.shape}")
        n_trees = int(np.unique(seg[:, 0]).size)
        if n_trees < 1:
            raise RuntimeError("segmentation found no trees in a two-tree cloud")
        print(f"      segmentation: {len(seg)} points in {n_trees} tree id(s)")


def _ply_counts(path: Path) -> tuple[int, int]:
    """Vertex and face counts from an ASCII PLY header."""
    n_vertices = n_faces = 0
    with open(path, "rb") as handle:
        for raw in handle:
            line = raw.decode("ascii", "replace").strip()
            if line.startswith("element vertex"):
                n_vertices = int(line.split()[-1])
            elif line.startswith("element face"):
                n_faces = int(line.split()[-1])
            elif line == "end_header":
                break
    return n_vertices, n_faces


def check_alpha_wrap() -> None:
    """The CGAL alpha-wrap binary wraps a synthetic crown into a closed mesh."""
    exe = _resolve_binary(AWRAP_BINARY)
    rng = np.random.default_rng(1)
    shell = rng.normal(size=(800, 3))
    shell /= np.linalg.norm(shell, axis=1, keepdims=True)
    pts = shell * 3.0 + np.array([5.0, 5.0, 8.0])
    with tempfile.TemporaryDirectory() as tmp:
        in_xyz = Path(tmp) / "crown.xyz"
        out_ply = Path(tmp) / "crown.ply"
        np.savetxt(in_xyz, pts, fmt="%.6f")
        result = subprocess.run(
            [str(exe), str(in_xyz), "15", "50", str(out_ply)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"exit {result.returncode}: {result.stderr.strip()[:400]}")
        if not out_ply.exists():
            raise RuntimeError("produced no PLY mesh")
        n_vertices, n_faces = _ply_counts(out_ply)
        if n_vertices <= 0 or n_faces <= 0:
            raise RuntimeError(f"degenerate mesh: {n_vertices} vertices, {n_faces} faces")
        print(f"      alpha-wrap: {n_vertices} vertices, {n_faces} faces")


CHECKS: tuple[tuple[str, Callable[[], None]], ...] = (
    ("conda env imports the geospatial stack", check_imports),
    ("main.py --help runs under the entrypoint", check_pipeline_help),
    ("segmentation binary runs on synthetic data", check_segmentation),
    ("alpha-wrap binary runs on synthetic data", check_alpha_wrap),
)


def main() -> int:
    failures = 0
    for label, check in CHECKS:
        try:
            check()
        except Exception as exc:  # noqa: BLE001  (report any failure, keep going)
            print(f"FAIL  {label}: {exc}")
            failures += 1
        else:
            print(f"ok    {label}")
    if failures:
        print(f"\nsmoke test FAILED ({failures} of {len(CHECKS)} checks)")
        return 1
    print(f"\nsmoke test passed ({len(CHECKS)} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
