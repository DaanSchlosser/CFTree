# Copyright (C) 2025 Noah Alting
# Licensed under the GNU General Public License v3.0
# See the LICENSE file for more details.

# src/get_data/download_geotiles.py

import logging
import time
from pathlib import Path

import requests

from src.get_data.tile_sources import TileSource
from src.stages import DownloadResult, RemoteUnavailableError, StageFailureError
from src.tile_layout import CaseLayout

# A short connect timeout still fast-fails a genuinely unreachable host, but the
# read timeout has to be generous. The GeoTiles host can stall ~20 s staging a
# cold tile before the first byte, and the AHN LAZ sub-tiles are ~190 MB, so a
# tight read timeout drops large downloads under concurrent load (a single
# value would apply to both connect and read).
_CONNECT_TIMEOUT = 15  # seconds to establish the connection
_READ_TIMEOUT = 120  # seconds to wait for the next streamed chunk (or first byte)
_REQUEST_TIMEOUT = (_CONNECT_TIMEOUT, _READ_TIMEOUT)
_CHUNK_SIZE = 1 << 20  # 1 MiB streaming chunks
# A cold tile often stalls on the first hit and then serves from a warm cache,
# so a few linear-backoff retries recover a transient GeoTiles stall rather than
# failing the whole acquisition stage.
_DOWNLOAD_ATTEMPTS = 3
_RETRY_BACKOFF_S = 5  # multiplied by the attempt number


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
    """Download `url` to `dest` atomically: write to .part, then rename.

    A transient connection or timeout failure is retried a few times with linear
    backoff, because the GeoTiles host can stall staging a cold tile and the LAZ
    sub-tiles are large, so a single attempt drops mid-stream under load. HTTP
    status errors (403/404 out of coverage, other 4xx/5xx) are not transient, so
    they are not retried and propagate to `_ensure_file` on the first attempt.
    The partial `.part` file is removed between attempts, so each retry starts
    clean and a final failure leaves nothing behind.
    """
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
        try:
            with requests.get(
                url, stream=True, allow_redirects=True, timeout=_REQUEST_TIMEOUT
            ) as r:
                r.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=_CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
            tmp.replace(dest)
            return
        except requests.HTTPError:
            # Not transient (e.g. 403/404 out of coverage); let the caller decide.
            tmp.unlink(missing_ok=True)
            raise
        except (
            requests.ConnectionError,
            requests.Timeout,
            requests.ChunkedEncodingError,
        ) as e:
            tmp.unlink(missing_ok=True)
            if attempt == _DOWNLOAD_ATTEMPTS:
                raise
            wait = _RETRY_BACKOFF_S * attempt
            logging.info(
                f"Download stalled ({e.__class__.__name__}); "
                f"retrying {attempt + 1}/{_DOWNLOAD_ATTEMPTS} in {wait}s"
            )
            time.sleep(wait)
