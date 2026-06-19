# Copyright (C) 2025 Noah Alting
# Licensed under the GNU General Public License v3.0
# See the LICENSE file for more details.

#!/usr/bin/env python3
"""
scripts/run_get_data.py

Full pipeline for downloading and clipping AHN geotiles for a case.

Example:
    nohup python -m scripts.run_get_data --n-cores 2 --case emmer_compascuum &
"""

import argparse
import json
import logging
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import laspy
from shapely.geometry import Polygon, box

from src.config import get_config, setup_logger
from src.get_data.clip_tile import clip_tile
from src.get_data.download_geotiles import download_tile
from src.get_data.extract_dtm import compute_tile_dtm
from src.get_data.tile_sources import TileSource, from_version
from src.stages import (
    MissingPrerequisiteError,
    RemoteUnavailableError,
    StageError,
    TileOutcome,
    failed_statuses,
)
from src.tile_layout import CaseLayout


# ---------------------------------------------------------------------
# Tile workers (must be top-level for multiprocessing)
# ---------------------------------------------------------------------
def download_one(
    tile_id: str,
    output_dir: Path,
    source: TileSource,
    overwrite: bool,
) -> TileOutcome:
    """Sweep 1: download one tile's raw point cloud.

    A separate sweep from clipping because a tile's clip now reads its
    neighbours' raw clouds (the halo); all downloads must finish before any clip
    starts, or which border points a clip sees would depend on download order.
    """
    try:
        dl = download_tile(tile_id, output_dir, source, overwrite=overwrite)
    except RemoteUnavailableError as e:
        logging.info(f"[{tile_id}] Skipped: {e}")
        return TileOutcome(tile_id=tile_id, status="not_in_coverage", detail=str(e))
    except StageError as e:
        logging.warning(f"[{tile_id}] Download failed: {e}")
        return TileOutcome(tile_id=tile_id, status="download_failed", detail=str(e))
    return TileOutcome(tile_id=tile_id, status="downloaded", paths={"raw": dl.laz})


def clip_and_dtm(
    tile_id: str,
    output_dir: Path,
    inputs: list[Path],
    overwrite: bool,
) -> TileOutcome:
    """Sweep 2: clip the tile's own + neighbour clouds to its halo region, then DTM.

    `inputs` is the owning tile's `raw.laz` plus neighbour `raw.laz` overlapping
    the tile's halo region (built in the parent); the clip region is written to
    `tile.clip_region` there.
    """
    tile = CaseLayout(data_dir=output_dir).tile(tile_id)
    try:
        clip = clip_tile(inputs, tile.clip_region, tile.clipped_laz, overwrite=overwrite)
    except MissingPrerequisiteError as e:
        logging.error(f"[{tile_id}] Clip prerequisite missing: {e}")
        return TileOutcome(tile_id=tile_id, status="clip_prereq_missing", detail=str(e))
    except StageError as e:
        logging.warning(f"[{tile_id}] Clip failed: {e}")
        return TileOutcome(tile_id=tile_id, status="clip_failed", detail=str(e))

    try:
        logging.info(f"[{tile_id}] Computing DTM from clipped tile...")
        # Recompute the DTM whenever the clip actually (re)ran: a clip
        # invalidated by a changed region/inputs would otherwise leave a stale
        # DTM derived from the old point support.
        dtm = compute_tile_dtm(clip.clipped, tile.dtm, ground_only=True, overwrite=overwrite or clip.did_work)
    except StageError as e:
        logging.warning(f"[{tile_id}] DTM generation failed: {e}")
        return TileOutcome(tile_id=tile_id, status="dtm_failed", paths={"clipped": clip.clipped}, detail=str(e))

    return TileOutcome(tile_id=tile_id, status="ok", paths={"clipped": clip.clipped, "dtm": dtm.dtm})


def run_sweep(
    worker: Callable[..., TileOutcome],
    tasks: list[tuple],
    n_cores: int,
) -> list[TileOutcome]:
    """Run `worker(*task)` over `tasks`, in parallel when n_cores > 1.

    `task[0]` must be the tile id (used for the per-tile log line). A worker that
    raises is recorded as an `exception` outcome rather than aborting the sweep.
    """
    outcomes: list[TileOutcome] = []
    if n_cores > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=n_cores) as pool:
            futures = {pool.submit(worker, *t): t[0] for t in tasks}
            for f in as_completed(futures):
                tid = futures[f]
                try:
                    res = f.result()
                    outcomes.append(res)
                    logging.info(f"[{tid}] {res.status.upper()}")
                except Exception as e:
                    logging.warning(f"[{tid}] Exception: {e}")
                    outcomes.append(TileOutcome(tile_id=str(tid), status="exception", detail=str(e)))
    else:
        for t in tasks:
            res = worker(*t)
            outcomes.append(res)
            logging.info(f"[{res.tile_id}] {res.status.upper()}")
    return outcomes


# ---------------------------------------------------------------------
# Runner main
# ---------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Run get_data pipeline for a case.")
    parser.add_argument("--case", type=str, help="Case name (default from config)")
    parser.add_argument("--n-cores", type=int, default=None, help="Number of parallel workers (default from config)")
    parser.add_argument("--overwrite", action="store_true", help="Re-download tiles if they exist")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING)")
    parser.add_argument("--dry-run", action="store_true", help="Only list tiles to be processed")
    parser.add_argument("--buffer", type=float, default=20.0, help="Buffer in meters around AOI")
    parser.add_argument(
        "--halo-margin",
        type=float,
        default=12.0,
        help="Per-tile overlap (m) into neighbouring tiles so a border tree is reconstructed "
        "whole in its owning tile (default: 12). Should exceed the largest crown radius and be "
        "<= --buffer.",
    )
    parser.add_argument(
        "--ahn-version",
        type=int,
        choices=(4, 5, 6),
        default=6,
        help="AHN release to download (default: 6). 4/5 are TU Delft GeoTiles; 6 is basisdata.nl COPC.",
    )
    args = parser.parse_args()

    # Load configuration
    cfg = get_config(case_name=args.case, n_cores=args.n_cores)
    case = cfg["case"]
    n_cores = cfg["default_cores"]

    setup_logger(case, "get_data", args.log_level)

    logging.info(f"Starting get_data for case: {case}")
    logging.info(f"Parallel workers: {n_cores} (from {'CLI' if args.n_cores else 'config'})")
    logging.info(f"Buffer distance: {args.buffer} m")
    logging.info(f"AHN release: AHN{args.ahn_version}")

    layout = CaseLayout.from_config(cfg)
    aoi_path = layout.aoi
    buffered_aoi_path = layout.buffered_aoi
    resources_dir = cfg["resources_dir"]
    output_dir = cfg["data_case_path"]

    # ------------------------------------------------------------------
    # Step 1: Load and buffer AOI
    # ------------------------------------------------------------------
    logging.info(f"Loading AOI from {aoi_path}")
    logging.info(f"CRS: {cfg['crs']}")
    aoi = gpd.read_file(aoi_path).to_crs(cfg["crs"])
    aoi["geometry"] = aoi.buffer(args.buffer)
    aoi.to_file(buffered_aoi_path, driver="GeoJSON")
    logging.info(f"Buffered AOI saved to {buffered_aoi_path}")

    # ------------------------------------------------------------------
    # Step 2: Resolve tile catalog and intersecting tiles
    # ------------------------------------------------------------------
    source = from_version(args.ahn_version, resources_dir)
    logging.info(f"Tile source: {source.name}, {source.attribution}")

    # Persist the AHN version/source so the segmentation stage can resolve each
    # tile's core cell for ownership-based dedup (the version is only a CLI flag,
    # not part of the config that downstream stages load).
    layout.data_dir.mkdir(parents=True, exist_ok=True)
    layout.tile_source_manifest.write_text(json.dumps({"ahn_version": args.ahn_version, "source": source.name}))

    tile_ids = source.tiles_for_aoi(aoi.union_all())

    if not tile_ids:
        logging.info("No intersecting tiles found.")
        return 0

    logging.info(
        f"Found {len(tile_ids)} intersecting tiles: {tile_ids if len(tile_ids) <= 10 else tile_ids[:10] + ['...']}"
    )

    if args.dry_run:
        logging.info("[DRY RUN] Exiting before downloads.")
        return 0

    # ------------------------------------------------------------------
    # Step 3: download all tiles (barrier), then clip + DTM with a halo
    # ------------------------------------------------------------------
    margin = args.halo_margin
    if args.buffer < margin:
        logging.warning(
            f"--buffer ({args.buffer} m) < --halo-margin ({margin} m): AOI-perimeter neighbour "
            "clouds may not be downloaded, leaving border crowns incomplete there. "
            "Increase --buffer to at least the halo margin."
        )

    # Sweep 1: download every tile. Must finish before any clip starts, because a
    # tile's clip now reads its neighbours' raw clouds (the halo) — otherwise which
    # border points a clip sees would depend on download order (non-deterministic).
    logging.info(f"Downloading {len(tile_ids)} tiles ({n_cores} workers)...")
    download_outcomes = run_sweep(
        download_one, [(tid, output_dir, source, args.overwrite) for tid in tile_ids], n_cores
    )
    download_failed = failed_statuses(o.status for o in download_outcomes)

    present = [tid for tid in tile_ids if layout.tile(tid).raw_laz.exists()]
    if not present:
        # No raw clouds on disk. A genuine download failure exits non-zero;
        # an AHN6 AOI entirely outside coverage (all "not_in_coverage") is a
        # graceful skip, so it exits 0 with nothing to clip.
        if download_failed:
            logging.error(f"No tiles downloaded; {len(download_failed)} download failure(s).")
            return 1
        logging.info("No tiles downloaded (none in coverage); nothing to clip.")
        return 0
    logging.info(f"{len(present)}/{len(tile_ids)} tiles downloaded; building halo clip regions.")

    # Build each tile's clip region (its core cell + halo margin, intersected with
    # the buffered AOI) and the neighbour raw clouds needed to fill it.
    buffered_geom = aoi.union_all()
    cells = {tid: source.core_cell(tid) for tid in present}
    # Actual extent of each tile's own raw cloud (from the LAZ header). This is what
    # makes the halo gather uniform across AHN versions WITHOUT a per-version branch
    # and without duplicating points: we only pull a neighbour for the part of the
    # region the tile's OWN raw does not already cover. Hard-partitioned AHN6 raws
    # stop at the cell edge, so the margin band is "missing" and the (disjoint)
    # neighbour cells are pulled to fill it; already-overlapping AHN4/5 raws cover
    # the whole region, so nothing is missing and the clip degenerates to a single
    # input. (This stays duplicate-point-free as long as the halo margin does not
    # exceed an overlapping source's inter-tile overlap — true for AHN4/5's ~20 m
    # vs the 12 m default; AHN6 has zero overlap so any margin is safe.)
    raw_bbox: dict[str, Polygon] = {}
    for tid in present:
        with laspy.open(layout.tile(tid).raw_laz) as f:
            h = f.header
            raw_bbox[tid] = box(h.x_min, h.y_min, h.x_max, h.y_max)

    clip_tasks: list[tuple] = []
    for tid in present:
        minx, miny, maxx, maxy = cells[tid].bounds
        region = box(minx - margin, miny - margin, maxx + margin, maxy + margin).intersection(buffered_geom)
        if region.is_empty:
            region = box(minx - margin, miny - margin, maxx + margin, maxy + margin)
        tile = layout.tile(tid)
        gpd.GeoDataFrame(geometry=[region], crs=cfg["crs"]).to_file(tile.clip_region, driver="GeoJSON")
        inputs = [tile.raw_laz]
        missing = region.difference(raw_bbox[tid])
        if missing.area > 1e-6:  # the tile's own raw does not cover the whole region
            inputs += [layout.tile(nbr).raw_laz for nbr in present if nbr != tid and raw_bbox[nbr].intersects(missing)]
        clip_tasks.append((tid, output_dir, inputs, args.overwrite))

    # Sweep 2: clip (with halo) + DTM, per tile.
    logging.info(f"Clipping + DTM for {len(clip_tasks)} tiles ({n_cores} workers)...")
    outcomes = run_sweep(clip_and_dtm, clip_tasks, n_cores)

    n_ok = sum(1 for o in outcomes if o.status == "ok")
    logging.info(f"Completed get_data for case: {case} — {n_ok}/{len(clip_tasks)} tiles clipped + DTM ok")

    # Exit non-zero on any genuine download or clip/DTM failure (graceful
    # "not_in_coverage" skips excluded), so the orchestrator aborts rather
    # than feeding a partial tile set to segmentation and reconstruction.
    failed = download_failed + failed_statuses(o.status for o in outcomes)
    if failed:
        logging.error(f"get_data failed: {len(failed)} stage failure(s): {failed}")
        return 1
    return 0


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
