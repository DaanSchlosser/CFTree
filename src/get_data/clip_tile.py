# Copyright (C) 2025 Noah Alting
# Licensed under the GNU General Public License v3.0
# See the LICENSE file for more details.

# src/get_data/clip_tile.py

import logging
import subprocess
from pathlib import Path

from src.stages import ClipResult, MissingPrerequisiteError, StageFailureError
from src.tile_layout import TileLayout


def clip_tile(laz_path: Path, aoi_path: Path, output_dir: Path | None = None, overwrite: bool = False) -> ClipResult:
    """Clip a single LAZ file using PDAL through the robust bash script.

    Raises
    ------
    MissingPrerequisiteError
        Input LAZ, AOI file, or the bash script is not on disk.
    StageFailureError
        PDAL ran but did not produce the expected output.
    """
    script_path = Path(__file__).parent / "tiles_clipper_robust.sh"
    tile_id = laz_path.parent.name
    output_dir = output_dir or laz_path.parent
    clipped_path = TileLayout(output_dir).clipped_laz

    if not script_path.exists():
        raise MissingPrerequisiteError(f"[{tile_id}] Clipping script not found: {script_path}")
    if not laz_path.exists():
        raise MissingPrerequisiteError(f"[{tile_id}] Input LAZ not found: {laz_path}")
    if not aoi_path.exists():
        raise MissingPrerequisiteError(f"[{tile_id}] AOI file not found: {aoi_path}")

    if clipped_path.exists() and not overwrite:
        logging.info(f"[{tile_id}] Skipping existing clipped tile")
        return ClipResult(clipped=clipped_path, did_work=False)

    logging.info(f"[{tile_id}] Clipping raw tile → {clipped_path.name}")
    try:
        subprocess.run(
            ["bash", str(script_path), str(laz_path), str(aoi_path), str(clipped_path)],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="ignore").strip()
        raise StageFailureError(f"[{tile_id}] Clipping failed: {stderr}") from e

    if not clipped_path.exists():
        raise StageFailureError(f"[{tile_id}] Clipping completed but file missing: {clipped_path}")

    logging.info(f"[{tile_id}] Clipped successfully → {clipped_path}")
    return ClipResult(clipped=clipped_path, did_work=True)
