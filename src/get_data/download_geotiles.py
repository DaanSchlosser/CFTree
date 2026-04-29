# Copyright (C) 2025 Noah Alting
# Licensed under the GNU General Public License v3.0
# See the LICENSE file for more details.

# src/get_data/download_geotiles.py

import logging
from pathlib import Path

import requests

from src.get_data.tile_sources import TileSource

_REQUEST_TIMEOUT = 30  # connect/read seconds
_CHUNK_SIZE = 1 << 20  # 1 MiB streaming chunks


def download_tile(
    tile_id: str,
    output_dir: Path,
    source: TileSource,
    overwrite: bool = False,
) -> dict:
    """Download the LAZ (and optional `.LAX`) for a single tile."""
    tile_folder = output_dir / "tiles" / tile_id
    tile_folder.mkdir(parents=True, exist_ok=True)
    laz_path = tile_folder / "raw.laz"
    lax_path = tile_folder / "raw.lax"

    laz_status = _ensure_file(
        url=source.laz_url(tile_id), dest=laz_path,
        tile_id=tile_id, label="LAZ", overwrite=overwrite, required=True,
    )
    if laz_status != "ok":
        return {"tile_id": tile_id, "status": laz_status, "paths": {"laz": None, "lax": None}}

    lax_url = source.lax_url(tile_id)
    lax_final: Path | None = None
    if lax_url is not None:
        lax_status = _ensure_file(
            url=lax_url, dest=lax_path,
            tile_id=tile_id, label="LAX", overwrite=overwrite, required=False,
        )
        if lax_status == "ok":
            lax_final = lax_path

    return {"tile_id": tile_id, "status": "ok", "paths": {"laz": laz_path, "lax": lax_final}}


def _ensure_file(
    url: str, dest: Path, tile_id: str, label: str,
    overwrite: bool, required: bool,
) -> str:
    """Stream `url` to `dest` unless it already exists. Returns a status string.

    Maps HTTP 403/404 to `not_found_remote` and other failures to `download_failed`.
    For non-required files (e.g. AHN4/5 LAX), absence is logged at INFO; for required
    files at WARNING. Partial files are removed on failure so re-runs start clean.
    """
    if dest.exists() and not overwrite:
        logging.info(f"[{tile_id}] Skipping existing {label}")
        return "ok"

    logging.info(f"[{tile_id}] Downloading {label} from {url}")
    try:
        _stream_download(url, dest)
        return "ok"
    except requests.HTTPError as e:
        code = e.response.status_code if e.response is not None else None
        # basisdata.nl's Ceph backend can return 403 instead of 404 for missing
        # objects; treat both as "not in coverage" to keep the user-facing
        # message accurate without confusing it with auth/server errors.
        if code in (403, 404):
            log = logging.warning if required else logging.info
            log(f"[{tile_id}] {label} not available at {url} (HTTP {code})")
            dest.unlink(missing_ok=True)
            return "not_found_remote"
        logging.warning(f"[{tile_id}] {label} download failed: {e}")
        dest.unlink(missing_ok=True)
        return "download_failed"
    except (requests.RequestException, OSError) as e:
        logging.warning(f"[{tile_id}] {label} download failed: {e}")
        dest.unlink(missing_ok=True)
        return "download_failed"


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
