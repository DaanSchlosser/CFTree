# Copyright (C) 2025 Noah Alting
# Licensed under the GNU General Public License v3.0
# See the LICENSE file for more details.

# src/config.py

"""
Central configuration for the CFTree pipeline.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict


class DefaultConfig(TypedDict):
    case_root: Path
    data_root: Path
    resources_dir: Path
    case: str
    default_cores: int
    crs: str


class ResolvedConfig(TypedDict):
    case_root: Path
    data_root: Path
    resources_dir: Path
    case: str
    default_cores: int
    crs: str
    case_path: Path
    data_case_path: Path


# ---------------------------------------------------------------------
# Default case configurations
# ---------------------------------------------------------------------
DEFAULT_CONFIG: DefaultConfig = {
    "case_root": Path("cases"),  # user case input directory
    "data_root": Path("data"),  # data storage root (large files)
    "resources_dir": Path("resources"),  # shared resources
    "case": "wippolder",  # default case
    "default_cores": 2,  # global default for parallelization
    "crs": "EPSG:28992",  # Amersfoort / RD New
}


# ---------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------
def setup_logger(case: str, logfile_name: str, level: str = "INFO") -> Path:
    """
    Set up a logger that writes to cases/<case>/logs/<logfile_name>.log.

    - Creates directories automatically.
    - Logs to both console and file.
    - Uses UTC ISO-8601 timestamps.
    - Adds a NEW SESSION banner at start.
    """
    case_root = DEFAULT_CONFIG["case_root"]
    case_path = case_root / case
    log_dir = case_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{logfile_name}.log"

    # Reset any prior logging config
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)sZ [%(levelname)s] [%(processName)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, mode="a"),
            logging.StreamHandler(),
        ],
    )

    # UTC timestamps
    logging.Formatter.converter = lambda *_args: datetime.now(UTC).timetuple()

    banner = "\n" + "=" * 40 + f" NEW SESSION {datetime.now(UTC).isoformat()}Z" + "=" * 40
    logging.info(banner)
    logging.info(f"Logging to: {log_path}")
    return log_path


# ---------------------------------------------------------------------
# Config management
# ---------------------------------------------------------------------
def get_config(case_name: str | None = None, n_cores: int | None = None) -> ResolvedConfig:
    """
    Return resolved configuration with canonical paths and compute settings.

    Parameters
    ----------
    case_name : str, optional
        Case name to override default.
    n_cores : int, optional
        Number of cores to override default.

    If not provided, defaults to 'wippolder' and 2 cores.
    """
    case_name = case_name if case_name is not None else DEFAULT_CONFIG["case"]
    n_cores = n_cores if n_cores is not None else DEFAULT_CONFIG["default_cores"]

    case_root = Path(DEFAULT_CONFIG["case_root"]).expanduser().resolve()
    data_root = Path(DEFAULT_CONFIG["data_root"]).expanduser().resolve()
    resources_dir = Path(DEFAULT_CONFIG["resources_dir"]).expanduser().resolve()

    data_case_path = data_root / case_name
    data_case_path.mkdir(parents=True, exist_ok=True)

    return {
        "case_root": case_root,
        "data_root": data_root,
        "resources_dir": resources_dir,
        "case": case_name,
        "default_cores": int(n_cores),
        "crs": DEFAULT_CONFIG["crs"],
        "case_path": case_root / case_name,
        "data_case_path": data_case_path,
    }


# ---------------------------------------------------------------------
# CLI inspection
# ---------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = get_config()
    print("Active CFTree configuration:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
