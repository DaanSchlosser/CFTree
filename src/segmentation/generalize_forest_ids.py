# Copyright (C) 2025 Noah Alting
# Licensed under the GNU General Public License v3.0
# See the LICENSE file for more details.

# src/segmentation/generalize_forest_ids.py

"""
Assigns global tree IDs (GTIDs) to segmented trees across all tiles of a case,
removes trees outside the AOI, and enriches vegetation.laz with GTIDs.

Reads:
    cases/<case>/case_area.geojson
    data/<case>/tiles/<tile_id>/tree_hulls.geojson
    data/<case>/tiles/<tile_id>/segmentation.xyz
    data/<case>/tiles/<tile_id>/vegetation.laz

Writes:
    data/<case>/forest_hulls.geojson
    data/<case>/gtid_map.csv
    data/<case>/tiles/<tile_id>/forest.laz
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import cast

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd

from src.config import get_config
from src.get_data.tile_sources import from_version
from src.stages import GeneralizeForestIdsResult, MissingPrerequisiteError, StageFailureError
from src.tile_layout import CaseLayout, TileLayout


def _write_forest_laz(tile_dir: Path, tile_map: pd.DataFrame, overwrite: bool) -> tuple[str, int]:
    """Attach GTIDs to one tile's vegetation points and write `forest.laz`.

    Pure per-tile work (reads its own ``segmentation.xyz`` + ``vegetation.laz``,
    writes its own ``forest.laz``) with no shared mutable state, so tiles can run
    concurrently. `tile_map` is the slice of the global GTID map for this tile.

    Returns ``(tile_name, n_points_written)``; ``n_points_written == 0`` means the
    tile was skipped (missing inputs, already present, or no GTID matches).
    """
    tile = TileLayout(tile_dir)
    name = tile_dir.name
    veg_path = tile.vegetation_laz
    seg_path = tile.segmentation_xyz
    out_forest = tile.forest_laz

    if not veg_path.exists() or not seg_path.exists():
        logging.debug(f"[{name}] Missing vegetation or segmentation file — skipped.")
        return name, 0
    if out_forest.exists() and not overwrite:
        logging.debug(f"[{name}] Forest already exists — skipped.")
        return name, 0
    if tile_map.empty:
        logging.warning(f"[{name}] No GTIDs found for this tile — skipped.")
        return name, 0

    # segmentation.xyz: whitespace-separated "tid x y z" produced by the C++
    # binary; keep the regex separator to tolerate its exact spacing.
    try:
        seg_df = pd.read_csv(seg_path, sep=r"\s+", header=None, names=["tid", "x", "y", "z"])
    except Exception as e:
        logging.warning(f"[{name}] Failed reading segmentation.xyz: {e}")
        return name, 0

    # Attach gtid by tree id. tid is unique per tile, so this is a 1:1 lookup.
    seg_df = seg_df.merge(tile_map[["tid", "gtid"]], on="tid", how="inner")
    if seg_df.empty:
        logging.debug(f"[{name}] No matching GTIDs after merge — skipped.")
        return name, 0

    try:
        with laspy.open(veg_path) as src:
            las = src.read()
    except Exception as e:
        logging.warning(f"[{name}] Failed reading vegetation.laz: {e}")
        return name, 0
    if len(las.points) == 0:
        logging.debug(f"[{name}] LAS has no coordinate data — skipped.")
        return name, 0

    # The C++ segmentation output dropped the original point ordering, so the
    # gtid is recovered by matching exact (x, y, z) against the vegetation cloud.
    veg_df = pd.DataFrame({"x": np.asarray(las.x), "y": np.asarray(las.y), "z": np.asarray(las.z)})
    merged = pd.merge(veg_df, seg_df[["x", "y", "z", "gtid"]], on=["x", "y", "z"], how="inner")
    if merged.empty:
        logging.debug(f"[{name}] No coordinate matches — skipped.")
        return name, 0

    try:
        header = laspy.LasHeader(point_format=las.header.point_format, version=las.header.version)
        las_out = laspy.LasData(header)
        las_out.x = merged["x"].values
        las_out.y = merged["y"].values
        las_out.z = merged["z"].values
        # uint32 gtid extra dimension for downstream reconstruction.
        if "gtid" not in las_out.point_format.extra_dimension_names:
            las_out.add_extra_dim(laspy.ExtraBytesParams(name="gtid", type=np.uint32))
        las_out["gtid"] = merged["gtid"].astype(np.uint32).values
        las_out.write(out_forest)
    except Exception as e:
        logging.warning(f"[{name}] Failed writing forest.laz: {e}")
        return name, 0

    # Note: the happy-path "wrote N points" line is logged by the caller, not
    # here — spawned workers have no logging config, so an INFO emitted in the
    # worker would never reach the case log file in the parallel path.
    return name, len(las_out.points)


def generalize_forest_ids(case: str, overwrite: bool = False, n_cores: int = 1) -> GeneralizeForestIdsResult:
    """Create global tree IDs (GTIDs) and write forest.laz per tile.

    The per-tile `forest.laz` enrichment runs across `n_cores` worker processes
    (tiles are independent); pass ``n_cores=1`` for serial execution.

    Raises
    ------
    MissingPrerequisiteError
        AOI file is missing or empty.
    StageFailureError
        No tree hulls inside the AOI across any tile.
    """
    cfg = get_config(case_name=case)
    layout = CaseLayout.from_config(cfg)
    tiles_dir = layout.tiles_dir

    aoi_path = layout.aoi
    if not aoi_path.exists():
        raise MissingPrerequisiteError(f"AOI not found: {aoi_path}")

    aoi = gpd.read_file(aoi_path)
    if aoi.empty:
        raise MissingPrerequisiteError(f"AOI file is empty: {aoi_path}")

    aoi_geom = aoi.to_crs(cfg["crs"]).geometry.union_all()

    # Resolve the tile source (persisted by get_data) so each tile's
    # non-overlapping core cell is known. Each physical tree is kept by exactly
    # one tile — the one whose core cell contains the tree's centroid — which
    # removes cross-tile duplicates (overlapping AHN4/5 tiles) deterministically
    # and uniformly, with no per-version branch here. Fail loud rather than guess
    # the version: feeding an AHN4/5 sub-tile id to the AHN6 grid (or vice versa)
    # would mis-own or crash.
    manifest_path = layout.tile_source_manifest
    if not manifest_path.exists():
        raise MissingPrerequisiteError(
            f"Tile-source manifest not found: {manifest_path}. "
            "Re-run get_data to record the AHN version before segmentation."
        )
    ahn_version = int(json.loads(manifest_path.read_text())["ahn_version"])
    source = from_version(ahn_version, cfg["resources_dir"])

    # ------------------------------------------------------------------
    # Collect tree hulls inside the AOI, owned by their tile's core cell
    # ------------------------------------------------------------------
    hulls_all: list[gpd.GeoDataFrame] = []  # in-AOI hulls owned by their producing tile
    orphans: list[gpd.GeoDataFrame] = []  # in-AOI hulls whose centroid is owned by another tile
    present_tile_ids: list[str] = []
    for tile_dir in sorted(tiles_dir.iterdir()):
        logging.debug(f"Entering tile: {tile_dir}")

        hull_path = TileLayout(tile_dir).tree_hulls
        if not hull_path.exists():
            continue
        try:
            gdf = gpd.read_file(hull_path).to_crs(cfg["crs"])
            gdf["tile_id"] = tile_dir.name
            gdf["centroid"] = gdf.geometry.centroid
            present_tile_ids.append(tile_dir.name)

            cx = gdf["centroid"].x.to_numpy()
            cy = gdf["centroid"].y.to_numpy()
            # Outer gate: inside the study AOI. Inner gate: owned by THIS tile's
            # core cell (the half-open partition rule lives in TileSource), so a
            # tree present in two overlapping tiles is kept by exactly one.
            # owns_centroids → core_cell raises for a foreign/stale tile dir,
            # which the except below logs and skips.
            in_aoi = gdf["centroid"].within(aoi_geom).to_numpy()
            owned = source.owns_centroids(tile_dir.name, cx, cy)

            kept = gdf[in_aoi & owned]
            if not kept.empty:
                hulls_all.append(kept)
            orphan = gdf[in_aoi & ~owned]
            if not orphan.empty:
                orphans.append(orphan)
        except Exception as e:
            logging.warning(f"[{tile_dir.name}] Failed reading/owning hulls: {e}")

    if not hulls_all and not orphans:
        raise StageFailureError(f"No valid tree hulls found inside AOI for case {case}")

    # Coverage guard: a hull dropped by its producing tile (centroid in a
    # neighbour's core cell) is normally kept by that neighbour — that IS the
    # cross-tile dedup. But if NO processed tile owns its centroid (owner tile
    # absent / not downloaded), rescue it in its producing tile rather than lose
    # it entirely.
    keep_frames: list[gpd.GeoDataFrame] = list(hulls_all)
    n_deduped = 0
    if orphans:
        orphan_all = pd.concat(orphans, ignore_index=True)
        ocx = orphan_all["centroid"].x.to_numpy()
        ocy = orphan_all["centroid"].y.to_numpy()
        covered = np.zeros(len(orphan_all), dtype=bool)
        for tid in dict.fromkeys(present_tile_ids):
            try:
                owned_by_tid = source.owns_centroids(tid, ocx, ocy)
            except Exception:
                continue
            covered |= owned_by_tid
        n_deduped = int(covered.sum())
        rescued = orphan_all[~covered]
        if not rescued.empty:
            logging.warning(
                f"{len(rescued)} in-AOI tree(s) owned by an absent tile — kept in their "
                "producing tile to avoid loss (no processed tile owns their centroid)."
            )
            keep_frames.append(rescued)
    if n_deduped:
        logging.info(f"Ownership dedup: dropped {n_deduped} cross-tile duplicate/relocated hull(s).")

    if not keep_frames:
        raise StageFailureError(f"No valid tree hulls found inside AOI for case {case}")

    hulls = pd.concat(keep_frames, ignore_index=True)
    hulls = hulls.drop(columns="centroid")
    hulls = gpd.GeoDataFrame(hulls, crs=cfg["crs"])

    # ------------------------------------------------------------------
    # Assign GTIDs sequentially
    # ------------------------------------------------------------------
    hulls["gtid"] = np.arange(1, len(hulls) + 1, dtype=np.uint32)
    n_trees = len(hulls)
    logging.info(f"Assigned GTIDs for {n_trees} trees across {len(hulls_all)} tiles.")

    # ------------------------------------------------------------------
    # Write forest-level outputs
    # ------------------------------------------------------------------
    out_forest_hulls = layout.forest_hulls
    out_gtid_map = layout.gtid_map

    hulls[["tile_id", "tid", "gtid", "geometry"]].to_file(out_forest_hulls, driver="GeoJSON")
    hulls[["tile_id", "tid", "gtid"]].to_csv(out_gtid_map, index=False)
    logging.info(f"Wrote forest hulls: {out_forest_hulls}")
    logging.info(f"Wrote GTID map: {out_gtid_map}")

    # ------------------------------------------------------------------
    # Enrich vegetation.laz per tile with GTID
    # ------------------------------------------------------------------
    # Pre-slice the global GTID map per tile so each worker receives only its
    # own (small) rows; the per-tile work is otherwise fully independent.
    gtid_map = pd.read_csv(out_gtid_map)
    tasks = [
        (tile_dir, cast("pd.DataFrame", gtid_map[gtid_map["tile_id"] == tile_dir.name]))
        for tile_dir in sorted(tiles_dir.iterdir())
    ]

    if n_cores > 1 and len(tasks) > 1:
        logging.info(f"Enriching {len(tasks)} tiles with GTIDs across {n_cores} workers.")
        # Use a "spawn" context, not the default fork: this pool is created after
        # GDAL/GEOS work (union_all, to_file) has run in the parent, and forking
        # at that point inherits library locks in a locked state, which dead-
        # locks the children on their first laspy/pandas call. Spawn starts a
        # fresh interpreter and sidesteps it — matching scripts/reconstruction.py.
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_cores, mp_context=ctx) as pool:
            futures = {pool.submit(_write_forest_laz, td, tm, overwrite): td.name for td, tm in tasks}
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    tname, n_pts = fut.result()
                    if n_pts:
                        logging.info(f"[{tname}] Wrote forest.laz ({n_pts} points)")
                except Exception as e:
                    logging.warning(f"[{name}] forest.laz enrichment failed: {e}")
    else:
        for td, tm in tasks:
            tname, n_pts = _write_forest_laz(td, tm, overwrite)
            if n_pts:
                logging.info(f"[{tname}] Wrote forest.laz ({n_pts} points)")

    return GeneralizeForestIdsResult(
        n_trees=n_trees,
        forest_hulls=out_forest_hulls,
        gtid_map=out_gtid_map,
    )
