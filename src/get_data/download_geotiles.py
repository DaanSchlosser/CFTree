# Copyright (C) 2025 Noah Alting
# Licensed under the GNU General Public License v3.0
# See the LICENSE file for more details.

# src/get_data/download_geotiles.py

import logging
import os
import shutil
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

# Default sub-directory (under the data root, a sibling of each case's data dir)
# for the shared tile cache. Overridable with CFTREE_AHN_CACHE for a fast-local
# or shared-network cache. See `_default_cache_dir`.
_CACHE_DIRNAME = ".ahn_cache"


def download_tile(
    tile_id: str,
    output_dir: Path,
    source: TileSource,
    overwrite: bool = False,
    cache_dir: Path | None = None,
) -> DownloadResult:
    """Provide the LAZ (and optional `.LAX`) for a single tile, via a shared cache.

    `output_dir` is the per-case data root (i.e. `data/<case>/`); the tile
    sub-tree is resolved through `CaseLayout`.

    The raw national tile is fetched into a cross-case cache keyed by
    source + tile id (see `_default_cache_dir`) and then placed at the case's
    `raw.laz` by a hardlink (falling back to a copy). The cache means a second
    AOI that overlaps an already-fetched tile pays no network cost, and it is
    deliberately independent of `overwrite`: the AHN bytes for a given version
    and tile are immutable, so re-downloading them on a forced case rebuild is
    pure waste. `overwrite` still refreshes the case-local `raw.laz` from the
    cache, so a rebuild always re-derives downstream artifacts.

    Raises
    ------
    RemoteUnavailableError
        Required LAZ returned 403/404 (AHN6 outside first-release coverage).
    StageFailureError
        Required LAZ download failed for any other reason.
    """
    tile = CaseLayout(data_dir=output_dir).tile(tile_id)
    tile.dir.mkdir(parents=True, exist_ok=True)
    cdir = cache_dir if cache_dir is not None else _default_cache_dir(output_dir, source)

    laz_cache = cdir / f"{tile_id}.LAZ"
    laz_downloaded = _ensure_cached(
        url=source.laz_url(tile_id),
        cache=laz_cache,
        tile_id=tile_id,
        label="LAZ",
        required=True,
    )
    laz_placed = _provision(laz_cache, tile.raw_laz, overwrite=overwrite)

    lax_url = source.lax_url(tile_id)
    lax_final: Path | None = None
    lax_downloaded = False
    if lax_url is not None:
        lax_cache = cdir / f"{tile_id}.LAX"
        try:
            lax_downloaded = _ensure_cached(
                url=lax_url,
                cache=lax_cache,
                tile_id=tile_id,
                label="LAX",
                required=False,
            )
            _provision(lax_cache, tile.raw_lax, overwrite=overwrite)
            lax_final = tile.raw_lax
        except (RemoteUnavailableError, StageFailureError):
            # LAX is optional — log was emitted by `_ensure_cached`; proceed without it.
            lax_final = None

    return DownloadResult(
        laz=tile.raw_laz,
        lax=lax_final,
        did_work=laz_downloaded or lax_downloaded or laz_placed,
    )


def _default_cache_dir(output_dir: Path, source: TileSource) -> Path:
    """Resolve the shared tile-cache directory for a source.

    Defaults to ``<data_root>/.ahn_cache/<source>`` (the data root is the parent
    of the per-case `output_dir`), so the cache sits beside the case data and
    persists for both the WSL mount and the Docker bind-mount. ``CFTREE_AHN_CACHE``
    overrides the base directory for a fast-local or shared cache.
    """
    env = os.environ.get("CFTREE_AHN_CACHE")
    base = Path(env).expanduser() if env else Path(output_dir).parent / _CACHE_DIRNAME
    return base / source.name


def _ensure_cached(
    url: str,
    cache: Path,
    tile_id: str,
    label: str,
    required: bool,
) -> bool:
    """Make sure the immutable tile file is in the shared cache. Returns True if downloaded.

    A cache hit skips the network entirely and is independent of any per-case
    `overwrite`, because the AHN bytes for a (version, tile) never change.

    Raises `RemoteUnavailableError` for HTTP 403/404, `StageFailureError` for
    other failures. For non-required files (e.g. AHN4/5 LAX), absence is logged
    at INFO; for required files at WARNING. Partial files are removed on failure
    so re-runs start clean.
    """
    if cache.exists():
        logging.info(f"[{tile_id}] Using cached {label}: {cache}")
        return False

    cache.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"[{tile_id}] Downloading {label} from {url}")
    try:
        _stream_download(url, cache)
        return True
    except requests.HTTPError as e:
        code = e.response.status_code if e.response is not None else None
        # basisdata.nl's Ceph backend can return 403 instead of 404 for missing
        # objects; treat both as "not in coverage" to keep the user-facing
        # message accurate without confusing it with auth/server errors.
        if code in (403, 404):
            log = logging.warning if required else logging.info
            log(f"[{tile_id}] {label} not available at {url} (HTTP {code})")
            cache.unlink(missing_ok=True)
            raise RemoteUnavailableError(f"{label} HTTP {code} at {url}") from e
        logging.warning(f"[{tile_id}] {label} download failed: {e}")
        cache.unlink(missing_ok=True)
        raise StageFailureError(f"{label} HTTP error: {e}") from e
    except (requests.RequestException, OSError) as e:
        logging.warning(f"[{tile_id}] {label} download failed: {e}")
        cache.unlink(missing_ok=True)
        raise StageFailureError(f"{label} download failed: {e}") from e


def _provision(cache: Path, dest: Path, overwrite: bool) -> bool:
    """Place the cached tile at the case path. Returns True if (re)placed.

    Uses a hardlink (instant, no extra space) when the cache and the case dir
    share a filesystem, falling back to a copy across filesystems. The link/copy
    is staged under a unique temp name and atomically renamed, so an interrupted
    run never leaves a partial `raw.laz`. An existing case file of the same size
    is trusted (the tile is immutable) and left in place unless `overwrite`.
    """
    if dest.exists() and not overwrite:
        try:
            if dest.stat().st_size == cache.stat().st_size:
                return False
        except OSError:
            pass  # fall through and re-provision from the cache

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + f".{os.getpid()}.linktmp")
    tmp.unlink(missing_ok=True)
    try:
        os.link(cache, tmp)
    except OSError:
        shutil.copy2(cache, tmp)
    tmp.replace(dest)
    return True


def _stream_download(url: str, dest: Path) -> None:
    """Download `url` to `dest` atomically: write to a unique .part, then rename.

    A transient connection or timeout failure is retried a few times with linear
    backoff, because the GeoTiles host can stall staging a cold tile and the LAZ
    sub-tiles are large, so a single attempt drops mid-stream under load. HTTP
    status errors (403/404 out of coverage, other 4xx/5xx) are not transient, so
    they are not retried and propagate to the caller on the first attempt. The
    `.part` name carries the writer's pid so two cases downloading the same tile
    into the shared cache concurrently never clobber each other's temp; the final
    rename is atomic and the bytes are identical either way. The partial file is
    removed between attempts, so each retry starts clean and a final failure
    leaves nothing behind.
    """
    tmp = dest.with_suffix(dest.suffix + f".{os.getpid()}.part")
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
