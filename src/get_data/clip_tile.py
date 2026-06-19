# Copyright (C) 2025 Noah Alting
# Licensed under the GNU General Public License v3.0
# See the LICENSE file for more details.

# src/get_data/clip_tile.py

import contextlib
import hashlib
import logging
import subprocess
from pathlib import Path

from src.stages import ClipResult, MissingPrerequisiteError, StageFailureError


def _clip_sig_path(output_path: Path) -> Path:
    """Sidecar recording the clip region + input set that produced *output_path*."""
    return output_path.with_name(output_path.name + ".clipsig")


def clip_signature(region_path: Path, inputs: list[Path]) -> str:
    """Stable signature of a clip: the region geometry plus the input set.

    Re-clip when either changes. A different ``--halo-margin`` / ``--buffer``
    rewrites ``clip_region.geojson`` (and can change which neighbour tiles are
    inputs), so a ``clipped.laz`` from the old region must not be reused — its
    point support no longer matches the current halo, which would break the
    ownership-based cross-tile dedup that assumes each tile saw its full halo.
    """
    h = hashlib.sha256()
    h.update(region_path.read_bytes())
    for name in sorted(p.name for p in inputs):
        h.update(b"\0")
        h.update(name.encode("utf-8"))
    return h.hexdigest()


def clip_cache_valid(sig_path: Path, current_sig: str) -> bool:
    """Whether an existing clip matches *current_sig*.

    A missing or unreadable sidecar (e.g. a clip from before this guard existed)
    is treated as a mismatch, so the stale output is conservatively re-clipped.
    """
    try:
        return sig_path.read_text().strip() == current_sig
    except OSError:
        return False


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

    sig_path = _clip_sig_path(output_path)
    current_sig = clip_signature(region_path, inputs)
    if output_path.exists() and not overwrite:
        if clip_cache_valid(sig_path, current_sig):
            logging.info(f"[{tile_id}] Skipping existing clipped tile")
            return ClipResult(clipped=output_path, did_work=False)
        logging.info(f"[{tile_id}] Clip region/inputs changed since last run — re-clipping")

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

    # Record the signature so a later run with a changed region/inputs re-clips
    # instead of reusing this file. A write failure here is non-fatal: a missing
    # sidecar simply forces a (safe) re-clip next time.
    with contextlib.suppress(OSError):
        sig_path.write_text(current_sig)

    logging.info(f"[{tile_id}] Clipped successfully → {output_path}")
    return ClipResult(clipped=output_path, did_work=True)
