# Copyright (C) 2025 Noah Alting
# Licensed under the GNU General Public License v3.0
# See the LICENSE file for more details.

# src/reconstruction/gpu_metrics.py
"""GPU implementations of the two dominant per-tree morphometrics.

A prior performance study found that ``r50`` and ``porosity`` account for the
large majority of per-tree reconstruction time, and that both reduce to two
primitives on a single small watertight crown mesh: an inside/outside test of a
voxel-center grid, and a nearest-neighbour distance from interior points to the
vegetation points. This module computes those two primitives on the GPU:

* the inside/outside test uses NVIDIA Warp's winding-number mesh query, which is
  valid here because the crown is a CGAL alpha-wrap and is therefore guaranteed
  watertight and 2-manifold. This replaces both the embree ``mesh.contains`` of
  the porosity grid and trimesh's pure-Python ``voxelized().fill()`` interior of
  r50, which the in-container benchmark showed were the parts worth moving;
* the r50 nearest-neighbour distance stays on scipy's ``cKDTree``. A k-d tree is
  O(S log N) for the small per-tree point counts here and beat a GPU brute-force
  (O(S*N) with large temporaries) decisively in the benchmark, so only the
  interior test is GPU-accelerated, not the query.

Only Warp is required (no cuPy): the query reuses the k-d tree the CPU path
already builds.

The functions mirror the semantics of :func:`extract_tree_metrics._compute_porosity`
and :func:`extract_tree_metrics._compute_r50` line for line, so a validation
harness can diff the two implementations on the same inputs. Anything that
cannot run on the GPU (no CUDA device, missing package, an unexpected error)
returns ``None`` so the caller can fall back to the CPU path rather than fail.

This is GPU-only acceleration. It does not change the algorithm or the intended
output; the only differences from the CPU path are floating-point ordering and
the inside-test backend (winding number versus embree ray parity), both of which
the harness quantifies before this path is enabled.
"""

from __future__ import annotations

import logging

import numpy as np
import trimesh

_LOG = logging.getLogger(__name__)

# Warp and cuPy are optional and only present in a GPU-enabled environment, so
# import lazily and record availability rather than making them hard deps.
try:  # pragma: no cover - import guard, exercised only where the wheels exist
    import warp as wp

    _HAS_WARP = True
except Exception:  # noqa: BLE001 - any import failure means "no GPU path"
    wp = None  # type: ignore[assignment]
    _HAS_WARP = False

_WARP_READY = False


def gpu_available() -> bool:
    """True when Warp imports and a CUDA device is present.

    Cheap to call repeatedly: the Warp runtime is initialised once on the first
    successful call. A machine without an NVIDIA GPU returns ``False`` here, so
    the caller keeps the CPU path with no error.
    """
    global _WARP_READY
    if not _HAS_WARP:
        return False
    try:
        if not _WARP_READY:
            wp.init()
            _WARP_READY = True
        return wp.get_cuda_device_count() > 0
    except Exception as exc:  # noqa: BLE001
        _LOG.debug("GPU metrics unavailable: %s", exc)
        return False


# ---------------------------------------------------------------------
# Warp inside/outside test (winding number on a watertight mesh)
# ---------------------------------------------------------------------
if _HAS_WARP:

    @wp.kernel
    def _inside_kernel(
        mesh: wp.uint64,
        points: wp.array(dtype=wp.vec3),
        max_dist: float,
        out: wp.array(dtype=wp.int32),
    ) -> None:
        tid = wp.tid()
        query = wp.mesh_query_point_sign_winding_number(mesh, points[tid], max_dist)
        # Warp's sign convention is negative inside, positive outside.
        if query.result and query.sign < 0.0:
            out[tid] = 1
        else:
            out[tid] = 0


def _inside_mask(verts: np.ndarray, faces: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Classify each query point inside (True) or outside the mesh, on the GPU.

    *verts* (V, 3) and *faces* (F, 3) describe a watertight triangle mesh;
    *query* (M, 3) are the points to classify. Returns a boolean array (M,).
    """
    device = "cuda"
    v = wp.array(np.ascontiguousarray(verts, dtype=np.float32), dtype=wp.vec3, device=device)
    f = wp.array(np.ascontiguousarray(faces, dtype=np.int32).reshape(-1), dtype=wp.int32, device=device)
    mesh = wp.Mesh(points=v, indices=f, support_winding_number=True)

    qp = wp.array(np.ascontiguousarray(query, dtype=np.float32), dtype=wp.vec3, device=device)
    out = wp.zeros(len(query), dtype=wp.int32, device=device)
    # A max query distance comfortably larger than the mesh keeps the winding
    # number well-defined for every point regardless of how far outside it sits.
    span = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    max_dist = span * 2.0 + 1.0
    wp.launch(_inside_kernel, dim=len(query), inputs=[mesh.id, qp, max_dist, out], device=device)
    wp.synchronize_device(device)
    return out.numpy().astype(bool)


# ---------------------------------------------------------------------
# Porosity (GPU mirror of extract_tree_metrics._compute_porosity)
# ---------------------------------------------------------------------
def gpu_porosity(mesh: trimesh.Trimesh, pts_xyz: np.ndarray, voxel_size: float = 0.25) -> float | None:
    """GPU porosity; ``None`` on any failure so the caller falls back to CPU.

    Identical to the CPU version except the voxel-center inside test runs as a
    Warp winding-number query instead of an embree ``mesh.contains``.
    """
    try:
        local_mesh = mesh.copy()
        local_mesh.apply_translation(-local_mesh.centroid)
        local_pts = pts_xyz - mesh.centroid

        bmin, bmax = local_mesh.bounds
        span = bmax - bmin
        if np.any(span > 100):
            return np.nan

        nx, ny, nz = np.ceil(span / voxel_size).astype(int)
        total_voxels = int(nx) * int(ny) * int(nz)
        if total_voxels > 2e7:
            return np.nan

        xs = np.arange(bmin[0], bmax[0], voxel_size)
        ys = np.arange(bmin[1], bmax[1], voxel_size)
        zs = np.arange(bmin[2], bmax[2], voxel_size)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="xy")
        centers = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
        if centers.shape[0] == 0:
            return np.nan

        inside_mask = _inside_mask(local_mesh.vertices, local_mesh.faces, centers)
        interior = centers[inside_mask]
        if interior.size == 0:
            return np.nan

        ijk = np.floor((local_pts - bmin) / voxel_size).astype(int)
        uniq, _ = np.unique(ijk, axis=0, return_index=True)
        i, j, k = uniq.T
        valid = (i >= 0) & (i < nx) & (j >= 0) & (j < ny) & (k >= 0) & (k < nz)
        occ = np.zeros((ny, nx, nz), dtype=bool)
        occ[j[valid], i[valid], k[valid]] = True

        ci = np.floor((interior - bmin) / voxel_size).astype(int)
        valid2 = (
            (ci[:, 0] >= 0) & (ci[:, 0] < nx) & (ci[:, 1] >= 0) & (ci[:, 1] < ny) & (ci[:, 2] >= 0) & (ci[:, 2] < nz)
        )
        ci = ci[valid2]
        occ_interior = np.sum(occ[ci[:, 1], ci[:, 0], ci[:, 2]])

        porosity = 1.0 - (occ_interior / len(interior))
        return float(np.clip(porosity, 0, 1))

    except Exception as exc:  # noqa: BLE001 - fall back to CPU
        _LOG.warning("GPU porosity failed (%s); falling back to CPU", exc)
        return None


# ---------------------------------------------------------------------
# r50 (GPU mirror of extract_tree_metrics._compute_r50)
# ---------------------------------------------------------------------
def gpu_r50(
    mesh: trimesh.Trimesh,
    pts_xyz: np.ndarray,
    nn_samples: int = 600_000,
    voxel_ds: float = 0.02,
    interior_pitch: float = 0.1,
) -> float | None:
    """GPU r50; ``None`` on any failure so the caller falls back to CPU.

    The interior voxel set is taken as the centers of a ``interior_pitch`` grid
    over the crown bounds that fall inside the mesh (the GPU equivalent of
    ``mesh.voxelized(0.1).fill().points``), and the nearest-neighbour distance to
    the downsampled vegetation points is the exact cuPy brute-force minimum.
    """
    try:
        bmin, bmax = mesh.bounds
        xs = np.arange(bmin[0], bmax[0], interior_pitch)
        ys = np.arange(bmin[1], bmax[1], interior_pitch)
        zs = np.arange(bmin[2], bmax[2], interior_pitch)
        if xs.size == 0 or ys.size == 0 or zs.size == 0:
            return np.nan
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="xy")
        centers = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
        if centers.shape[0] == 0 or centers.shape[0] > 2e7:
            return np.nan

        inside_mask = _inside_mask(mesh.vertices, mesh.faces, centers)
        interior_pts = centers[inside_mask]
        if interior_pts.shape[0] == 0:
            return np.nan

        if voxel_ds > 0:
            ijk = np.floor((pts_xyz - pts_xyz.min(axis=0)) / voxel_ds).astype(int)
            _, keep = np.unique(ijk, axis=0, return_index=True)
            pts_xyz = pts_xyz[np.sort(keep)]
        if len(pts_xyz) == 0:
            return np.nan

        n = min(nn_samples, len(interior_pts))
        sample = interior_pts[:n]
        # The nearest-neighbour query is left on scipy's cKDTree: for the small
        # per-tree point counts here a k-d tree is O(S log N) and beats a GPU
        # brute-force (O(S*N) with large temporaries), which lost badly in the
        # in-container benchmark. The GPU win for r50 is the interior test above
        # (replacing trimesh's pure-Python voxelized().fill()), not the KNN.
        from scipy.spatial import cKDTree

        tree = cKDTree(pts_xyz, compact_nodes=True)
        d, _ = tree.query(sample, k=1, workers=-1)
        return float(np.median(d))

    except Exception as exc:  # noqa: BLE001 - fall back to CPU
        _LOG.warning("GPU r50 failed (%s); falling back to CPU", exc)
        return None
