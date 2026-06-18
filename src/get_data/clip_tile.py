# Copyright (C) 2025 Noah Alting
# Licensed under the GNU General Public License v3.0
# See the LICENSE file for more details.

# src/get_data/clip_tile.py

import logging
import subprocess
from pathlib import Path

from src.stages import ClipResult, MissingPrerequisiteError, StageFailureError


def clip_tile(
    inputs: list[Path],
    region_path: Path,
    output_path: Path,
    overwrite: bool = False,
) -> ClipResult:
    """Clip one or more LAZ files to `region_path`, writing one `output_path`.

    `inputs` is the owning tile's `raw.laz` plus any neighbour `raw.laz` that
    overlap the tile's halo region; they are merged before the crop so a tree
    straddling a tile boundary is reconstructed from the combined cloud. With a
    single input this is the plain per-tile clip.

    The owning tile id is taken from `output_path`'s parent directory (the tile
    that owns the clipped result), not from any input path.

    Raises
    ------
    MissingPrerequisiteError
        An input LAZ, the region file, or the bash script is not on disk.
    StageFailureError
        PDAL ran but did not produce the expected output.
    """
    script_path = Path(__file__).parent / "tiles_clipper_robust.sh"
    tile_id = output_path.parent.name

    if not script_path.exists():
        raise MissingPrerequisiteError(f"[{tile_id}] Clipping script not found: {script_path}")
    if not inputs:
        raise MissingPrerequisiteError(f"[{tile_id}] No input LAZ files given to clip")
    for laz in inputs:
        if not laz.exists():
            raise MissingPrerequisiteError(f"[{tile_id}] Input LAZ not found: {laz}")
    if not region_path.exists():
        raise MissingPrerequisiteError(f"[{tile_id}] Clip region not found: {region_path}")

    if output_path.exists() and not overwrite:
        logging.info(f"[{tile_id}] Skipping existing clipped tile")
        return ClipResult(clipped=output_path, did_work=False)

    logging.info(f"[{tile_id}] Clipping {len(inputs)} input(s) → {output_path.name}")
    try:
        subprocess.run(
            ["bash", str(script_path), str(region_path), str(output_path), *[str(p) for p in inputs]],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="ignore").strip()
        raise StageFailureError(f"[{tile_id}] Clipping failed: {stderr}") from e

    if not output_path.exists():
        raise StageFailureError(f"[{tile_id}] Clipping completed but file missing: {output_path}")

    logging.info(f"[{tile_id}] Clipped successfully → {output_path}")
    return ClipResult(clipped=output_path, did_work=True)
