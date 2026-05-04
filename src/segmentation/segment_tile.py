# Copyright (C) 2025 Noah Alting
# Licensed under the GNU General Public License v3.0
# See the LICENSE file for more details.

# src/segmentation/segment_tile.py

"""Wrapper for the C++ TreeSeparation segmentation binary.

Reads:
    data/<case>/tiles/<tile_id>/vegetation.xyz
Writes:
    data/<case>/tiles/<tile_id>/segmentation.xyz
    data/<case>/tiles/<tile_id>/tree_hulls.geojson
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPoint

from src.config import get_config
from src.stages import MissingPrerequisiteError, SegmentationResult, StageFailureError
from src.tile_layout import TileLayout

# Parameters passed to segmentation binary
SEG_PARAMS = {
    "radius": 2.5,
    "vres": 1.5,
    "min_pts": 3,
}
cfg = get_config()


def segment_tile(tile_dir: Path, overwrite: bool = False) -> SegmentationResult:
    """Run TreeSeparation C++ segmentation on one tile directory.

    Raises
    ------
    MissingPrerequisiteError
        Input vegetation.xyz or the C++ binary is missing.
    StageFailureError
        Segmentation binary failed or post-processing crashed.
    """
    tile = TileLayout(tile_dir)
    tile_id = tile.tile_id
    input_xyz = tile.vegetation_xyz
    output_xyz = tile.segmentation_xyz
    hulls_geojson = tile.tree_hulls

    exe = Path(__file__).parent / "TreeSeparation" / "build" / "segmentation"

    if not input_xyz.exists():
        raise MissingPrerequisiteError(f"[{tile_id}] Missing input vegetation.xyz at {input_xyz}")
    if not exe.exists():
        raise MissingPrerequisiteError(f"[{tile_id}] Missing C++ segmentation binary: {exe}")

    if output_xyz.exists() and hulls_geojson.exists() and not overwrite:
        logging.info(f"[{tile_id}] Segmentation already exists — skipping (use --overwrite to redo).")
        return SegmentationResult(segmentation_xyz=output_xyz, tree_hulls=hulls_geojson, did_work=False)

    cmd = [
        str(exe),
        str(input_xyz),
        str(output_xyz),
        str(SEG_PARAMS["radius"]),
        str(SEG_PARAMS["vres"]),
        str(SEG_PARAMS["min_pts"]),
    ]

    logging.info(f"[{tile_id}] Running segmentation binary...")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise StageFailureError(f"[{tile_id}] Segmentation failed: {e.stderr.strip()}") from e

    try:
        seg_df = pd.read_csv(output_xyz, sep=r"\s+", header=None, names=["tid", "x", "y", "z"])
        seg_gdf = gpd.GeoDataFrame(seg_df, geometry=gpd.points_from_xy(seg_df.x, seg_df.y), crs=cfg["crs"])

        hulls = []
        for tid, group in seg_gdf.groupby("tid"):
            if len(group) >= 3:
                hull_geom = MultiPoint(group.geometry.values).convex_hull
                hulls.append({"tid": tid, "geometry": hull_geom})
            else:
                logging.debug(f"[{tile_id}] Tree ID {tid} has <3 points — skipped.")

        if hulls:
            hulls_gdf = gpd.GeoDataFrame(hulls, crs=cfg["crs"])
            hulls_gdf.to_file(hulls_geojson, driver="GeoJSON")
        else:
            logging.warning(f"[{tile_id}] No valid hulls produced.")

        logging.info(f"[{tile_id}] Segmentation complete.")
        return SegmentationResult(segmentation_xyz=output_xyz, tree_hulls=hulls_geojson, did_work=True)

    except Exception as e:
        raise StageFailureError(f"[{tile_id}] Segmentation post-processing failed: {e}") from e
