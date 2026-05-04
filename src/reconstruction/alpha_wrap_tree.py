# Copyright (C) 2025 Noah Alting
# Licensed under the GNU General Public License v3.0
# See the LICENSE file for more details.

# src/reconstruction/alpha_wrap_tree.py

"""Python interface for per-tree alpha wrapping using the CGAL CLI binary.

Wraps: src/reconstruction/AlphaWrap/build/awrap_points

Reads:
    <cache_dir>/tree_<gtid>.xyz

Writes:
    <cache_dir>/tree_<gtid>.ply   # temporary geometry (deleted later)
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from src.stages import AlphaWrapResult, MissingPrerequisiteError, StageFailureError
from src.tile_layout import TileCacheLayout


def alpha_wrap_tree(
    tree_xyz: Path,
    cache_dir: Path,
    ralpha: float = 15.0,
    roffset: float = 50.0,
    binary_path: Path | None = None,
    overwrite: bool = False,
) -> AlphaWrapResult:
    """Run CGAL alpha wrapping on a single tree point cloud.

    Parameters
    ----------
    tree_xyz : Path
        Input .xyz file containing tree points.
    cache_dir : Path
        Directory for temporary files (_cache/).
    ralpha : float, default=15.0
        Alpha scaling factor relative to point cloud diagonal.
    roffset : float, default=50.0
        Offset scaling factor relative to alpha.
    binary_path : Path, optional
        Path to compiled awrap_points binary.
    overwrite : bool, default=False
        If True, re-run even if output already exists.

    Raises
    ------
    MissingPrerequisiteError
        Input .xyz or compiled binary not on disk.
    StageFailureError
        CGAL binary returned non-zero or crashed.
    """
    gtid = tree_xyz.stem.split("_")[-1]
    mesh_ply = TileCacheLayout(cache_dir).tree_ply(int(gtid))
    binary_path = binary_path or Path(__file__).parent / "AlphaWrap" / "build" / "awrap_points"

    if not tree_xyz.exists():
        raise MissingPrerequisiteError(f"[GTID {gtid}] Missing input file: {tree_xyz}")
    if not binary_path.exists():
        raise MissingPrerequisiteError(f"[GTID {gtid}] Missing alpha wrap binary: {binary_path}")

    if mesh_ply.exists() and not overwrite:
        logging.debug(f"[GTID {gtid}] Existing mesh found, skipping.")
        return AlphaWrapResult(mesh_ply=mesh_ply, did_work=False)

    cmd = [str(binary_path), str(tree_xyz), str(ralpha), str(roffset), str(mesh_ply)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise StageFailureError(f"[GTID {gtid}] Alpha wrapping failed: {e.stderr.decode(errors='ignore')}") from e
    except Exception as e:
        raise StageFailureError(f"[GTID {gtid}] Alpha wrapping unexpected error: {e}") from e

    logging.debug(f"[GTID {gtid}] Alpha wrap complete -> {mesh_ply.name}")
    return AlphaWrapResult(mesh_ply=mesh_ply, did_work=True)
