# Copyright (C) 2025 Noah Alting
# Licensed under the GNU General Public License v3.0
# See the LICENSE file for more details.

#!/usr/bin/env python3
import subprocess
import logging
import time
import argparse
from src.config import get_config, setup_logger
import os
import signal


def run_stage(name, cmd):
    """Run one pipeline stage and ensure cleanup of all workers if interrupted."""
    start = time.time()

    # Start subprocess in a new process group so we can kill all its children later
    process = subprocess.Popen(
        cmd,
        shell=True,
        preexec_fn=os.setsid,  # create new session (POSIX)
    )

    try:
        process.wait()
        if process.returncode != 0:
            logging.warning(f"{name} failed with exit code {process.returncode}")
    except KeyboardInterrupt:
        logging.warning(f"KeyboardInterrupt detected — terminating {name} and its workers...")
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        raise
    except Exception as e:
        logging.warning(f"{name} encountered error: {e}")
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        raise

    elapsed = (time.time() - start) / 60
    logging.info(f"Finished {name} in {elapsed:.2f} min")


def main():
    start = time.time()  # <-- define start time

    # -------------------------------
    # Parse CLI arguments
    # -------------------------------
    parser = argparse.ArgumentParser(description="Run full CFTree pipeline (get_data → segmentation → reconstruction).")
    parser.add_argument("--case", type=str, help="Case name (default from config)")
    parser.add_argument("--n-cores", type=int, help="Number of parallel workers (default from config)")
    parser.add_argument("--overwrite", action="store_true", help="Re-run even if outputs exist")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING)")
    parser.add_argument("--dry-run", action="store_true", help="Only list tiles to be processed")
    parser.add_argument("--buffer", type=float, default=20.0, help="Buffer distance around AOI (default 20m)")
    parser.add_argument("--max-trees", type=int, default=None, help="Limit number of trees per tile (for testing)")
    parser.add_argument(
        "--ahn-version",
        type=int,
        choices=(4, 5, 6),
        default=6,
        help="AHN release to download (default: 6). 4/5 are TU Delft GeoTiles; 6 is basisdata.nl COPC.",
    )
    args = parser.parse_args()

    # -------------------------------
    # Load configuration
    # -------------------------------
    cfg = get_config()
    
    case = args.case if args.case is not None else cfg["case"]
    n_cores = args.n_cores if args.n_cores is not None else cfg["default_cores"]

    # -------------------------------
    # Setup main logger
    # -------------------------------
    log_path = setup_logger(case, "main", level="INFO")
    logger = logging.getLogger()

    # -------------------------------
    # Base command builder
    # -------------------------------
    base_cmd = f"--case {case} --n-cores {n_cores}"
    if args.overwrite:
        base_cmd += " --overwrite"
    if args.dry_run:
        base_cmd += " --dry-run"
    base_cmd += f" --log-level {args.log_level}"

    # -------------------------------
    # Stage command definitions
    # -------------------------------
    cmd_get_data = f"python -m scripts.get_data {base_cmd} --buffer {args.buffer} --ahn-version {args.ahn_version}"
    cmd_segmentation = f"python -m scripts.segmentation {base_cmd}"
    cmd_reconstruction = f"python -m scripts.reconstruction {base_cmd}"
    if args.max_trees is not None:
        cmd_reconstruction += f" --max-trees {args.max_trees}" 

    # -------------------------------
    # Run stages sequentially
    # -------------------------------
    logger.info("\n" + "=" * 60 + "Starting full CFTree pipeline")
    logger.info("=" * 20 + " Stage 1: Data Acquisition")
    logger.info(f"buffer distance: {args.buffer} m, AHN version: {args.ahn_version}")
    run_stage("get_data", cmd_get_data)

    logger.info("=" * 20 + " Stage 2: Segmentation")
    run_stage("segmentation", cmd_segmentation)

    logger.info("=" * 20 + " Stage 3: Reconstruction")
    logger.info(f"max trees per tile: {args.max_trees if args.max_trees is not None else 'unlimited'}")
    run_stage("reconstruction", cmd_reconstruction)

    total_time = (time.time() - start) / 60
    logger.info("\n" + "=" * 60 + "Pipeline complete")
    logger.info(f"total elapsed time: {total_time:.2f} minutes")


if __name__ == "__main__":
    main()
