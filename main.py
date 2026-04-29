# Copyright (C) 2025 Noah Alting
# Licensed under the GNU General Public License v3.0
# See the LICENSE file for more details.

#!/usr/bin/env python3
import argparse
import logging
import os
import signal
import subprocess
import time

from src.config import get_config, setup_logger

_IS_POSIX = os.name == "posix"


def _terminate_process_group(process: subprocess.Popen) -> None:
    """Best-effort termination of a subprocess and its children.

    Uses POSIX process groups when available; falls back to terminate() on
    Windows / other platforms.
    """
    if _IS_POSIX:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)  # type: ignore[attr-defined]
    else:
        process.terminate()


def run_stage(name: str, cmd: str) -> None:
    """Run one pipeline stage and ensure cleanup of all workers if interrupted."""
    start = time.time()

    # On POSIX, start subprocess in a new session so we can kill the whole
    # process group on interrupt. On Windows, fall back to default behaviour.
    popen_kwargs: dict = {"shell": True}
    if _IS_POSIX:
        popen_kwargs["preexec_fn"] = os.setsid  # type: ignore[attr-defined]

    process = subprocess.Popen(cmd, **popen_kwargs)

    try:
        process.wait()
        if process.returncode != 0:
            logging.warning(f"{name} failed with exit code {process.returncode}")
    except KeyboardInterrupt:
        logging.warning(f"KeyboardInterrupt detected — terminating {name} and its workers...")
        _terminate_process_group(process)
        raise
    except Exception as e:
        logging.warning(f"{name} encountered error: {e}")
        _terminate_process_group(process)
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
    setup_logger(case, "main", level="INFO")
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
    banner_wide = "=" * 60
    banner_section = "=" * 20
    logger.info(f"\n{banner_wide}Starting full CFTree pipeline")
    logger.info(f"{banner_section} Stage 1: Data Acquisition")
    logger.info(f"buffer distance: {args.buffer} m, AHN version: {args.ahn_version}")
    run_stage("get_data", cmd_get_data)

    logger.info(f"{banner_section} Stage 2: Segmentation")
    run_stage("segmentation", cmd_segmentation)

    logger.info(f"{banner_section} Stage 3: Reconstruction")
    logger.info(f"max trees per tile: {args.max_trees if args.max_trees is not None else 'unlimited'}")
    run_stage("reconstruction", cmd_reconstruction)

    total_time = (time.time() - start) / 60
    logger.info(f"\n{banner_wide}Pipeline complete")
    logger.info(f"total elapsed time: {total_time:.2f} minutes")


if __name__ == "__main__":
    main()
