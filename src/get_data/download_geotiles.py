# Copyright (C) 2025 Noah Alting
# Licensed under the GNU General Public License v3.0
# See the LICENSE file for more details.

# src/get_data/download_geotiles.py

import logging
from pathlib import Path

import requests

from src.get_data.tile_sources import TileSource
from src.stages import DownloadResult, RemoteUnavailableError, StageFailureError
from src.tile_layout import CaseLayout

_REQUEST_TIMEOUT = 30  # connect/read seconds
_CHUNK_SIZE = 1 << 20  # 1 MiB streaming chunks


def download_tile(
    tile_id: str,
    output_dir: Path,
    source: TileSource,
    overwrite: bool = False,
) -> DownloadResult:
    """Download the LAZ (and optional `.LAX`) for a single tile.

    `output_dir` is the per-case data root (i.e. `data/<case>/`); the tile
    sub-tree is resolved through `CaseLayout`.

    Raises
    ------
    RemoteUnavailableError
        Required LAZ returned 403/404 (AHN6 outside first-release coverage).
    StageFailureError
        Required LAZ download failed for any other reason.
    """
    tile = CaseLayout(data_dir=output_dir).tile(tile_id)
    tile.dir.mkdir(parents=True, exist_ok=True)

    laz_did_work = _ensure_file(
        url=source.laz_url(tile_id),
        dest=tile.raw_laz,
        tile_id=tile_id,
        label="LAZ",
        overwrite=overwrite,
        required=True,
    )

    lax_url = source.lax_url(tile_id)
    lax_final: Path | None = None
    lax_did_work = False
    if lax_url is not None:
        try:
            lax_did_work = _ensure_file(
                url=lax_url,
                dest=tile.raw_lax,
                tile_id=tile_id,
                label="LAX",
                overwrite=overwrite,
                required=False,
            )
            lax_final = tile.raw_lax
        except (RemoteUnavailableError, StageFailureError):
            # LAX is optional — log was emitted by `_ensure_file`; proceed without it.
            lax_final = None

    return DownloadResult(
        laz=tile.raw_laz,
        lax=lax_final,
        did_work=laz_did_work or lax_did_work,
    )


def _ensure_file(
    url: str,
    dest: Path,
    tile_id: str,
    label: str,
    overwrite: bool,
    required: bool,
) -> bool:
    """Stream `url` to `dest` unless it already exists. Returns True if work was done.

    Raises `RemoteUnavailableError` for HTTP 403/404, `StageFailureError` for other failures.
    For non-required files (e.g. AHN4/5 LAX), absence is logged at INFO; for
    required files at WARNING. Partial files are removed on failure so re-runs
    start clean.
    """
    if dest.exists() and not overwrite:
        logging.info(f"[{tile_id}] Skipping existing {label}")
        return False

    logging.info(f"[{tile_id}] Downloading {label} from {url}")
    try:
        _stream_download(url, dest)
        return True
    except requests.HTTPError as e:
        code = e.response.status_code if e.response is not None else None
        # basisdata.nl's Ceph backend can return 403 instead of 404 for missing
        # objects; treat both as "not in coverage" to keep the user-facing
        # message accurate without confusing it with auth/server errors.
        if code in (403, 404):
            log = logging.warning if required else logging.info
            log(f"[{tile_id}] {label} not available at {url} (HTTP {code})")
            dest.unlink(missing_ok=True)
            raise RemoteUnavailableError(f"{label} HTTP {code} at {url}") from e
        logging.warning(f"[{tile_id}] {label} download failed: {e}")
        dest.unlink(missing_ok=True)
        raise StageFailureError(f"{label} HTTP error: {e}") from e
    except (requests.RequestException, OSError) as e:
        logging.warning(f"[{tile_id}] {label} download failed: {e}")
        dest.unlink(missing_ok=True)
        raise StageFailureError(f"{label} download failed: {e}") from e


def _stream_download(url: str, dest: Path) -> None:
    """Download `url` to `dest` atomically: write to .part, then rename."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, allow_redirects=True, timeout=_REQUEST_TIMEOUT) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=_CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
    tmp.replace(dest)
