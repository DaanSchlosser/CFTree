# Copyright (C) 2025 Noah Alting
# Licensed under the GNU General Public License v3.0
# See the LICENSE file for more details.

# src/reconstruction/construct_geometry.py

"""Construct LoD3 tree geometries (crown + trunk) in local coordinates.

Both crown and trunk are represented as CityJSON "Solid" geometries.
This module is pure: no file I/O or side effects beyond logging.

Inputs:
    - crown_mesh   : trimesh.Trimesh   # alpha-wrapped crown in local coords
    - metrics       : TreeMetrics
    - offset_global : list[float]      # translation back to RD New
    - gtid, tile_id : optional identifiers for logging

Returns `Lod3Result(components, attributes)` — `components` may be empty if
neither crown nor trunk could be constructed.
"""

from __future__ import annotations

import logging
from collections import OrderedDict

import numpy as np
import trimesh

from src.stages import Lod3Result, TreeMetrics

# ---------------------------------------------------------------------
# Canonical attribute order (extendable later)
# ---------------------------------------------------------------------
ATTR_KEYS = [
    "gtid",
    "tile_id",
    "crown_width_m",
    "crown_median_z",
    "crown_r50_m",
    "crown_porosity",
    "trunk_H_m",
    "trunk_DBH_m",
    "trunk_radius_m",
    "trunk_base_height_m",
]


# ---------------------------------------------------------------------
# Geometry builders
# ---------------------------------------------------------------------
def _build_crown_solid(mesh: trimesh.Trimesh, gtid: int | None = None) -> dict | None:
    """Return crown as a Solid (LoD3)."""
    if mesh.is_empty or mesh.vertices.size == 0 or mesh.faces.size == 0:
        logging.debug(f"[GTID {gtid}] Crown mesh empty — skipped.")
        return None

    vertices_local = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)

    logging.debug(
        f"[GTID {gtid}] Crown Solid: {len(vertices_local)} verts, {len(faces)} faces, "
        f"volume={mesh.volume:.3f}, watertight={mesh.is_watertight}"
    )

    return {
        "role": "crown",
        "lod": 3.0,
        "vertices_local": vertices_local,
        "faces": faces,
    }


def _build_trunk_solid(
    crown_mesh: trimesh.Trimesh,
    trunk_base: np.ndarray,
    r_trunk: float,
    crown_median_z: float,
    gtid: int | None = None,
) -> dict | None:
    """Return slanted trunk as a Solid (LoD3)."""
    try:
        if trunk_base is None or not np.all(np.isfinite(trunk_base)):
            logging.debug(f"[GTID {gtid}] Trunk base invalid — skipped.")
            return None
        if not np.isfinite(r_trunk) or r_trunk <= 0:
            logging.debug(f"[GTID {gtid}] Invalid trunk radius ({r_trunk}) — skipped.")
            return None

        # Axis from base → (crown centroid XY, crown_median_z)
        top = np.array([crown_mesh.centroid[0], crown_mesh.centroid[1], crown_median_z], dtype=float)
        base = np.asarray(trunk_base, dtype=float)
        axis = top - base
        length = np.linalg.norm(axis)
        if not np.isfinite(length) or length <= 0:
            logging.debug(f"[GTID {gtid}] Invalid trunk length ({length}) — skipped.")
            return None

        # Create cylinder aligned along +Z, bottom at (0, 0, 0)
        cyl = trimesh.creation.cylinder(radius=r_trunk, height=length, sections=24)
        cyl.apply_translation([0, 0, length / 2])

        # Align +Z axis with direction from base to top
        R = trimesh.geometry.align_vectors([0, 0, 1], axis / length)
        cyl.apply_transform(R)
        cyl.apply_translation(base)

        logging.debug(
            f"[GTID {gtid}] Trunk Solid: r={r_trunk:.3f} m, length={length:.3f} m, "
            f"{len(cyl.vertices)} verts, {len(cyl.faces)} faces"
        )

        return {
            "role": "trunk",
            "lod": 3.0,
            "vertices_local": np.asarray(cyl.vertices, dtype=float),
            "faces": np.asarray(cyl.faces, dtype=int),
        }

    except Exception as e:
        logging.debug(f"[GTID {gtid}] Trunk Solid construction failed: {e}")
        return None


# ---------------------------------------------------------------------
# Attribute normalization
# ---------------------------------------------------------------------
def _normalize_attributes(metrics: TreeMetrics, gtid: int, tile_id: str | None = None) -> OrderedDict:
    """Flatten metrics into the canonical CityJSON attribute order, with None for NaN."""
    bz = metrics.trunk_base_xyz[2]
    vals: dict[str, object] = {
        "gtid": gtid,
        "tile_id": tile_id,
        "crown_width_m": metrics.crown_width_m,
        "crown_median_z": metrics.crown_median_z,
        "crown_r50_m": metrics.r50_m,
        "crown_porosity": metrics.porosity,
        "trunk_H_m": metrics.height_m,
        "trunk_DBH_m": metrics.dbh_m,
        "trunk_radius_m": metrics.trunk_radius_m,
        "trunk_base_height_m": bz,
    }

    ordered = OrderedDict()
    for k in ATTR_KEYS:
        v = vals.get(k)
        if isinstance(v, float) and not np.isfinite(v):
            v = None
        ordered[k] = v

    # Compact, aligned one-line summary for debug readability
    vals_fmt = ", ".join(
        f"{k.split('.')[-1]}={v:.3f}" if isinstance(v, (float, int)) and v is not None else f"{k.split('.')[-1]}={v}"
        for k, v in ordered.items()
    )
    logging.debug(f"[GTID {gtid}] Normalized attributes: {vals_fmt}")

    return ordered


# ---------------------------------------------------------------------
# Main LoD3 constructor
# ---------------------------------------------------------------------
def construct_lod3(
    crown_mesh: trimesh.Trimesh,
    metrics: TreeMetrics,
    offset_global: list[float] | np.ndarray,
    gtid: int | None = None,
    tile_id: str | None = None,
) -> Lod3Result:
    """Construct LoD3 components (crown + trunk) for one tree in local coordinates.

    `Lod3Result.components` may be empty if neither crown nor trunk could be built;
    the caller decides whether to skip the tree.
    """
    gtid_str = f"GTID {gtid}" if gtid is not None else "GTID ?"
    logging.debug(f"[{tile_id}] [{gtid_str}] Constructing LoD3 geometry...")

    components: list[dict] = []

    crown_comp = _build_crown_solid(crown_mesh, gtid)
    if crown_comp is not None:
        components.append(crown_comp)
    else:
        logging.warning(f"[{tile_id}] [{gtid_str}] No valid crown component created.")

    offset_arr = np.asarray(offset_global, dtype=float)
    trunk_base_global = np.asarray(metrics.trunk_base_xyz, dtype=float)
    trunk_base_local = trunk_base_global - offset_arr if np.all(np.isfinite(trunk_base_global)) else None
    crown_median_z_local = metrics.crown_median_z - float(offset_arr[2])

    trunk_comp = _build_trunk_solid(
        crown_mesh=crown_mesh,
        trunk_base=trunk_base_local,
        r_trunk=metrics.trunk_radius_m,
        crown_median_z=crown_median_z_local,
        gtid=gtid,
    )
    if trunk_comp is not None:
        components.append(trunk_comp)
    else:
        logging.warning(f"[{tile_id}] [{gtid_str}] No valid trunk component created.")

    # Local coordinates shouldn't exceed ~5 000 m — guards against a global-coord leak.
    for comp in components:
        verts = comp.get("vertices_local")
        vmax = float(np.abs(np.asarray(verts)).max()) if verts is not None and len(verts) else 0.0
        if np.isfinite(vmax) and vmax > 5000:
            logging.warning(
                f"[Tile {tile_id} | GTID {gtid}] Suspicious local magnitude ({vmax:.1f}); "
                f"possible global coords leaked into local component."
            )

    attributes = _normalize_attributes(metrics, gtid or -1, tile_id)

    logging.info(
        f"[{tile_id}] [{gtid_str}] Constructed {len(components)} LoD3 components "
        f"(Crown={any(c['role'] == 'crown' for c in components)}, "
        f"Trunk={any(c['role'] == 'trunk' for c in components)})"
    )

    return Lod3Result(components=components, attributes=dict(attributes))
