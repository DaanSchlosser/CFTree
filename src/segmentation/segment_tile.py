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
import numpy as np
import pandas as pd
import shapely

from src.config import DEFAULT_CONFIG, resolve_native_binary
from src.stages import MissingPrerequisiteError, SegmentationResult, StageFailureError
from src.tile_layout import TileLayout

# Parameters passed to segmentation binary
SEG_PARAMS = {
    "radius": 2.5,
    "vres": 1.5,
    "min_pts": 3,
}

# Wall-clock bound for the C++ segmentation binary on one tile. Minutes at
# worst on a dense tile; an hour means it is wedged, and without a bound a
# bare (non-docker) run would stall forever with no per-tile failure reported.
_SEG_TIMEOUT_S = 3600


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

    exe = resolve_native_binary(Path(__file__).parent / "TreeSeparation" / "build" / "segmentation")

    if not input_xyz.exists():
        raise MissingPrerequisiteError(f"[{tile_id}] Missing input vegetation.xyz at {input_xyz}")
    if not exe.exists():
        raise MissingPrerequisiteError(f"[{tile_id}] Missing C++ segmentation binary: {exe}")

    if output_xyz.exists() and hulls_geojson.exists() and not overwrite:
        logging.info(f"[{tile_id}] Segmentation already exists — skipping (use --overwrite to redo).")
        return SegmentationResult(segmentation_xyz=output_xyz, tree_hulls=hulls_geojson, did_work=False)

    # The binary writes to a temp name that is renamed only after a clean exit,
    # with the (stale) hulls removed first. The skip gate above trusts the
    # existence of the xyz/hulls pair, so no kill point may leave a fresh xyz
    # beside hulls from an earlier run — the pair would silently disagree.
    tmp_xyz = output_xyz.with_name(output_xyz.name + ".part")
    cmd = [
        str(exe),
        str(input_xyz),
        str(tmp_xyz),
        str(SEG_PARAMS["radius"]),
        str(SEG_PARAMS["vres"]),
        str(SEG_PARAMS["min_pts"]),
    ]

    logging.info(f"[{tile_id}] Running segmentation binary...")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=_SEG_TIMEOUT_S)
    except subprocess.CalledProcessError as e:
        tmp_xyz.unlink(missing_ok=True)
        raise StageFailureError(f"[{tile_id}] Segmentation failed: {e.stderr.strip()}") from e
    except subprocess.TimeoutExpired as e:
        tmp_xyz.unlink(missing_ok=True)
        raise StageFailureError(f"[{tile_id}] Segmentation hung for {_SEG_TIMEOUT_S}s and was killed") from e
    hulls_geojson.unlink(missing_ok=True)
    tmp_xyz.replace(output_xyz)

    try:
        seg_df = pd.read_csv(output_xyz, sep=r"\s+", header=None, names=["tid", "x", "y", "z"])

        # Group points by tree id and hull each group in C (shapely 2), instead
        # of a Python loop building a MultiPoint per pandas groupby slice —
        # byte-identical hulls (same point multisets in the same order to the
        # same GEOS routine), an order of magnitude faster on a dense tile.
        codes, tids = pd.factorize(seg_df["tid"], sort=True)
        order = np.argsort(codes, kind="stable")  # multipoints() needs sorted indices
        points = shapely.points(seg_df["x"].to_numpy()[order], seg_df["y"].to_numpy()[order])
        multipoints = shapely.multipoints(points, indices=codes[order])
        counts = np.bincount(codes)
        hull_geoms = shapely.convex_hull(multipoints)

        enough = counts >= 3
        for tid in tids[~enough]:
            logging.debug(f"[{tile_id}] Tree ID {tid} has <3 points — skipped.")
        hulls = [{"tid": tid, "geometry": geom} for tid, geom in zip(tids[enough], hull_geoms[enough], strict=True)]

        if hulls:
            hulls_gdf = gpd.GeoDataFrame(hulls, crs=DEFAULT_CONFIG["crs"])
            # Temp name + rename: the skip gate must never see a half-written
            # hulls file (generalize_forest_ids reads every tile's hulls).
            tmp_hulls = hulls_geojson.with_name(hulls_geojson.name + ".part")
            hulls_gdf.to_file(tmp_hulls, driver="GeoJSON")
            tmp_hulls.replace(hulls_geojson)
        else:
            logging.warning(f"[{tile_id}] No valid hulls produced.")

        logging.info(f"[{tile_id}] Segmentation complete.")
        return SegmentationResult(segmentation_xyz=output_xyz, tree_hulls=hulls_geojson, did_work=True)

    except Exception as e:
        raise StageFailureError(f"[{tile_id}] Segmentation post-processing failed: {e}") from e
