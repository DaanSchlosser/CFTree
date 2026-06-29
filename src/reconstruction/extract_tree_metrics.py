# Copyright (C) 2025 Noah Alting
# Licensed under the GNU General Public License v3.0
# See the LICENSE file for more details.

# src/reconstruction/extract_tree_metrics.py
"""Compute all per-tree geometric and allometric metrics.

Inputs:
    - crown_mesh : trimesh.Trimesh (alpha-wrapped tree crown)
    - pts_xyz    : np.ndarray (N,3) vegetation points of this tree
    - dtm_path   : Path to DTM GeoTIFF for the tile

Returns a `TreeMetrics` dataclass. Undefined sub-metrics are encoded as `NaN`
(or `(NaN, NaN, NaN)` for `trunk_base_xyz`) — callers downstream filter on
`np.isfinite(...)`. Hard failures raise `StageFailureError`.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import rasterio
import shapely
import trimesh
from rasterio import mask
from scipy.spatial import ConvexHull, cKDTree
from shapely.geometry import Polygon

from src.reconstruction import gpu_metrics
from src.stages import StageFailureError, TreeMetrics


def _gpu_metrics_enabled() -> bool:
    """True when CFTREE_GPU_METRICS is set and a usable CUDA device is present.

    The flag is opt-in (default off) so the embree / cKDTree path stays the
    baseline until the GPU path has been validated against it on the target
    machine via ``scripts/bench_morphometrics.py``. ``gpu_available`` caches the
    Warp runtime init, so calling this per tree is cheap.
    """
    val = os.environ.get("CFTREE_GPU_METRICS", "").strip().lower()
    if val not in {"1", "true", "yes", "on"}:
        return False
    return gpu_metrics.gpu_available()


# ---------------------------------------------------------------------
# Crown geometry metrics
# ---------------------------------------------------------------------
def _compute_crown_metrics(crown_mesh: trimesh.Trimesh) -> tuple[float, float]:
    """Return (CW_m, crown_median_z)."""
    try:
        xy = crown_mesh.vertices[:, :2]
        if xy.shape[0] < 3:
            return (np.nan, np.nan)

        # convex hull area → equivalent circular crown width
        hull = ConvexHull(xy)
        area = float(hull.volume)  # in 2D, 'volume' = area
        CW_m = 2.0 * np.sqrt(area / np.pi)
        crown_median_z = float(np.median(crown_mesh.vertices[:, 2]))
        return (CW_m, crown_median_z)

    except Exception as e:
        logging.warning(f"Crown metric computation failed: {e}")
        return (np.nan, np.nan)


# ---------------------------------------------------------------------
# Porosity
# ---------------------------------------------------------------------
def _compute_porosity(mesh: trimesh.Trimesh, pts_xyz: np.ndarray, voxel_size: float = 0.25) -> float:
    try:
        # Shift coordinates to local origin to avoid large RD offsets
        local_mesh = mesh.copy()
        local_mesh.apply_translation(-local_mesh.centroid)
        local_pts = pts_xyz - mesh.centroid

        bmin, bmax = local_mesh.bounds
        span = bmax - bmin

        # Safety guard: if extent is physically unrealistic (>100 m³ crown)
        if np.any(span > 100):
            logging.debug(f"Porosity skipped — mesh span too large ({span})")
            return np.nan

        nx, ny, nz = np.ceil(span / voxel_size).astype(int)
        total_voxels = nx * ny * nz
        if total_voxels > 2e7:  # ~160 MB upper bound
            logging.debug(f"Porosity skipped — grid too dense ({total_voxels:,} voxels)")
            return np.nan

        xs = np.arange(bmin[0], bmax[0], voxel_size)
        ys = np.arange(bmin[1], bmax[1], voxel_size)
        zs = np.arange(bmin[2], bmax[2], voxel_size)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="xy")
        centers = np.c_[X.ravel(), Y.ravel(), Z.ravel()]

        inside_mask = local_mesh.contains(centers)
        interior = centers[inside_mask]
        if interior.size == 0:
            return np.nan

        # Occupied voxels by vegetation points (local)
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

    except Exception as e:
        logging.warning(f"Porosity computation failed: {e}")
        return np.nan


# ---------------------------------------------------------------------
# r50 nearest-neighbor distance
# ---------------------------------------------------------------------
def _compute_r50(
    mesh: trimesh.Trimesh, pts_xyz: np.ndarray, nn_samples: int = 600_000, voxel_ds: float = 0.02
) -> float:
    """
    Compute median NN distance (r50) between interior voxels and vegetation points.
    """
    try:
        vg = mesh.voxelized(0.1).fill()
        interior_pts = vg.points
        if interior_pts.shape[0] == 0:
            return np.nan

        # Downsample vegetation points for speed
        if voxel_ds > 0:
            ijk = np.floor((pts_xyz - pts_xyz.min(axis=0)) / voxel_ds).astype(int)
            _, keep = np.unique(ijk, axis=0, return_index=True)
            pts_xyz = pts_xyz[np.sort(keep)]

        n = min(nn_samples, len(interior_pts))
        sample = interior_pts[:n]
        tree = cKDTree(pts_xyz, compact_nodes=True)
        d, _ = tree.query(sample, k=1, workers=1)
        return float(np.median(d))

    except Exception as e:
        logging.warning(f"r50 computation failed: {e}")
        return np.nan


# ---------------------------------------------------------------------
# Tree metrics extraction
# ---------------------------------------------------------------------
def compute_trunk_base_from_dtm(
    crown_mesh: trimesh.Trimesh,
    dtm: Path | rasterio.io.DatasetReader,
    offset: np.ndarray | list[float] | tuple[float, float, float],
) -> np.ndarray | None:
    """
    Estimate the trunk base position [bx, by, bz] using the DTM under the crown footprint.

    This version is offset-aware:
    - `crown_mesh` is assumed to be in local coordinates (centered near 0).
    - `offset` gives the translation vector [x0, y0, z0] back to global RD coordinates.

    The function converts the local crown footprint to global coordinates before
    raster sampling to ensure alignment with the DTM.

    Parameters
    ----------
    crown_mesh : trimesh.Trimesh
        Crown mesh in local coordinates.
    dtm : Path | rasterio.io.DatasetReader
        The clipped DTM raster (RD New CRS), as a path or an already-open dataset.
        The DTM is identical for every tree in a tile, so a worker can open it
        once and pass the dataset here to avoid re-opening per tree; results are
        identical either way. A path is opened and closed within this call.
    offset : np.ndarray | list | tuple
        Translation vector applied to localize the tree (same units as DTM CRS).

    Returns
    -------
    np.ndarray | None
        [bx, by, bz] in global coordinates, or None if no valid DTM cell found.
    """
    try:
        # --- convert offset to array
        offset = np.asarray(offset, dtype=float)

        # --- crown footprint in local coordinates
        pts_xy_local = crown_mesh.vertices[:, :2]
        if pts_xy_local.shape[0] < 3:
            return None

        # --- convex hull of the crown projection (local)
        hull = ConvexHull(pts_xy_local)
        poly_local = Polygon(pts_xy_local[hull.vertices, :])

        # --- translate to global coordinates for raster sampling (ignore z offset)
        poly_global = shapely.affinity.translate(poly_local, xoff=offset[0], yoff=offset[1])

        # --- DTM source: open a path here (and close it), or reuse a dataset the
        #     caller already opened (one open per worker rather than per tree).
        opened = rasterio.open(dtm) if isinstance(dtm, (str, Path)) else None
        src = opened if opened is not None else dtm
        try:
            out_img, out_transform = mask.mask(src, [poly_global], crop=True, filled=False)
            band = out_img[0]
            rows, cols = np.where(~band.mask)

            if rows.size == 0:
                logging.warning("No valid DTM pixels under crown footprint.")
                return None

            xs, ys = rasterio.transform.xy(out_transform, rows, cols)
            coords = np.column_stack([xs, ys])

            # --- find DTM cell closest to the projected crown centroid (global)
            center_global = crown_mesh.centroid[:2] + offset[:2]
            idx = np.argmin(np.linalg.norm(coords - center_global, axis=1))

            bx, by, bz = (
                coords[idx, 0],
                coords[idx, 1],
                float(band.data[rows[idx], cols[idx]]),
            )
            return np.array([bx, by, bz], dtype=np.float64)
        finally:
            if opened is not None:
                opened.close()

    except Exception as e:
        logging.warning(f"Trunk base extraction failed: {e}")
        return None


def estimate_trunk_dimensions(
    CW_m: float, crown_median_z: float, trunk_base_z: float, a=1.0, b=1.1, c=0.7
) -> tuple[float, float, float]:
    """
    Returns (H, DBH_m, r_trunk_m). Applies simple allometry; no slenderness clamp by default.
    """
    if not (np.isfinite(CW_m) and np.isfinite(crown_median_z) and np.isfinite(trunk_base_z)):
        return (np.nan, np.nan, np.nan)
    H = float(crown_median_z - trunk_base_z)
    if H <= 0:
        return (H, np.nan, np.nan)
    DBH_m = float(a * (CW_m**b) * (H**c) / 100.0)
    r_trunk = 0.5 * DBH_m if np.isfinite(DBH_m) and DBH_m > 0 else np.nan
    return (H, DBH_m, r_trunk)


# ---------------------------------------------------------------------
# Main interface
# ---------------------------------------------------------------------
def compute_tree_metrics(
    crown_mesh: trimesh.Trimesh,
    pts_xyz: np.ndarray,
    dtm: Path | rasterio.io.DatasetReader,
    offset: np.ndarray | list[float] | tuple[float, float, float],
    compute_semantics: bool = True,
) -> TreeMetrics:
    """Compute all metrics for a single reconstructed tree, in global RD CRS.

    Sub-metrics that cannot be defined for this tree (e.g. degenerate hull,
    no DTM under crown) are returned as NaN; callers decide how to react.

    When `compute_semantics` is False (geometry-only runs), the two expensive,
    purely-descriptive metrics — `r50_m` and `porosity` — are skipped and
    returned as NaN. They account for ~80% of per-tree runtime and feed no
    geometry: crown comes from the alpha-wrap mesh and the trunk from
    crown width / DTM / allometry, all of which are still computed here.

    Raises
    ------
    StageFailureError
        Top-level extraction crashed (numpy / rasterio / etc. blew up).
    """
    try:
        CW_local, crown_median_z_local = _compute_crown_metrics(crown_mesh)
        CW_m = CW_local
        crown_median_z = crown_median_z_local + offset[2]
        logging.debug(f"Crown metrics (global): CW={CW_m:.3f}, median_z={crown_median_z:.3f}")

        trunk_base = compute_trunk_base_from_dtm(crown_mesh, dtm, offset)
        logging.debug(f"Trunk base (global): {trunk_base}")

        H_m, DBH_m, r_trunk = estimate_trunk_dimensions(
            CW_m, crown_median_z, trunk_base[2] if trunk_base is not None else np.nan
        )
        logging.debug(f"Trunk dimensions: H={H_m:.3f}, DBH={DBH_m:.3f}, r_trunk={r_trunk:.3f}")

        if compute_semantics:
            use_gpu = _gpu_metrics_enabled()
            # r50 is computed first because porosity's voxel size derives from it.
            r50_m = gpu_metrics.gpu_r50(crown_mesh, pts_xyz) if use_gpu else None
            if r50_m is None:
                r50_m = _compute_r50(crown_mesh, pts_xyz)
            voxel_size = r50_m * 0.8 if np.isfinite(r50_m) and r50_m > 0 else 0.25
            porosity = gpu_metrics.gpu_porosity(crown_mesh, pts_xyz, voxel_size=voxel_size) if use_gpu else None
            if porosity is None:
                porosity = _compute_porosity(crown_mesh, pts_xyz, voxel_size=voxel_size)
            logging.debug(f"r50={r50_m:.3f}, porosity={porosity:.3f}")
        else:
            # Geometry-only: skip the expensive descriptive metrics entirely.
            r50_m = np.nan
            porosity = np.nan

        base_xyz = tuple(trunk_base.tolist()) if trunk_base is not None else (np.nan, np.nan, np.nan)
        return TreeMetrics(
            crown_width_m=float(CW_m),
            crown_median_z=float(crown_median_z),
            porosity=float(porosity),
            r50_m=float(r50_m),
            height_m=float(H_m),
            dbh_m=float(DBH_m),
            trunk_radius_m=float(r_trunk),
            trunk_base_xyz=base_xyz,
        )

    except Exception as e:
        raise StageFailureError(f"Tree metric extraction failed: {e}") from e
