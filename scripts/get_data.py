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
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd

from src.config import get_config, setup_logger
from src.get_data.clip_tile import clip_tile
from src.get_data.download_geotiles import download_tile
from src.get_data.extract_dtm import compute_tile_dtm
from src.get_data.tile_sources import TileSource, from_version
from src.stages import MissingPrerequisiteError, RemoteUnavailableError, StageError, TileOutcome
from src.tile_layout import CaseLayout


# ---------------------------------------------------------------------
# Tile worker (must be top-level for multiprocessing)
# ---------------------------------------------------------------------
def process_tile(
    tile_id: str,
    output_dir: Path,
    source: TileSource,
    overwrite: bool,
    aoi_path: Path,
) -> TileOutcome:
    """Download, clip, and compute DTM for one tile.

    Catches `StageError` subclasses to map them to a single `TileOutcome` row
    for the summary log; never re-raises (keeps the pool moving).
    """
    try:
        dl = download_tile(tile_id, output_dir, source, overwrite=overwrite)
    except RemoteUnavailableError as e:
        logging.info(f"[{tile_id}] Skipped: {e}")
        return TileOutcome(tile_id=tile_id, status="not_in_coverage", detail=str(e))
    except StageError as e:
        logging.warning(f"[{tile_id}] Download failed: {e}")
        return TileOutcome(tile_id=tile_id, status="download_failed", detail=str(e))

    try:
        clip = clip_tile(dl.laz, aoi_path, overwrite=overwrite)
    except MissingPrerequisiteError as e:
        logging.error(f"[{tile_id}] Clip prerequisite missing: {e}")
        return TileOutcome(tile_id=tile_id, status="clip_prereq_missing", detail=str(e))
    except StageError as e:
        logging.warning(f"[{tile_id}] Clip failed: {e}")
        return TileOutcome(tile_id=tile_id, status="clip_failed", detail=str(e))

    dtm_out = clip.clipped.parent / "clipped_dtm.tif"
    try:
        logging.info(f"[{tile_id}] Computing DTM from clipped tile...")
        dtm = compute_tile_dtm(clip.clipped, dtm_out, ground_only=True, overwrite=overwrite)
    except StageError as e:
        logging.warning(f"[{tile_id}] DTM generation failed: {e}")
        return TileOutcome(
            tile_id=tile_id,
            status="dtm_failed",
            paths={"raw": dl.laz, "clipped": clip.clipped},
            detail=str(e),
        )

    return TileOutcome(
        tile_id=tile_id,
        status="ok",
        paths={"raw": dl.laz, "clipped": clip.clipped, "dtm": dtm.dtm},
    )


# ---------------------------------------------------------------------
# Runner main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run get_data pipeline for a case.")
    parser.add_argument("--case", type=str, help="Case name (default from config)")
    parser.add_argument("--n-cores", type=int, default=None, help="Number of parallel workers (default from config)")
    parser.add_argument("--overwrite", action="store_true", help="Re-download tiles if they exist")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING)")
    parser.add_argument("--dry-run", action="store_true", help="Only list tiles to be processed")
    parser.add_argument("--buffer", type=float, default=20.0, help="Buffer in meters around AOI")
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

    tile_ids = source.tiles_for_aoi(aoi.union_all())

    if not tile_ids:
        logging.info("No intersecting tiles found.")
        return

    logging.info(
        f"Found {len(tile_ids)} intersecting tiles: {tile_ids if len(tile_ids) <= 10 else tile_ids[:10] + ['...']}"
    )

    if args.dry_run:
        logging.info("[DRY RUN] Exiting before downloads.")
        return

    # ------------------------------------------------------------------
    # Step 3: Per-tile pipeline (download → clip → DTM)
    # ------------------------------------------------------------------
    if n_cores > 1:
        logging.info(f"Running {len(tile_ids)} tiles in parallel using {n_cores} cores.")
        with ProcessPoolExecutor(max_workers=n_cores) as pool:
            futures = {
                pool.submit(process_tile, tid, output_dir, source, args.overwrite, buffered_aoi_path): tid
                for tid in tile_ids
            }
            for f in as_completed(futures):
                tid = futures[f]
                try:
                    result = f.result()
                    logging.info(f"[{tid}] {result.status.upper()}")
                except Exception as e:
                    logging.warning(f"[{tid}] Exception: {e}")
    else:
        logging.info("Running serial mode.")
        for tid in tile_ids:
            result = process_tile(tid, output_dir, source, args.overwrite, buffered_aoi_path)
            logging.info(f"[{tid}] {result.status.upper()}")

    logging.info(f"Completed get_data for case: {case}")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
