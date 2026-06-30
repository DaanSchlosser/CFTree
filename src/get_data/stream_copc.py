# Copyright (C) 2025 Noah Alting
# Licensed under the GNU General Public License v3.0
# See the LICENSE file for more details.

# src/get_data/stream_copc.py

"""Range-read a single tile's AOI region from a remote COPC.

A Cloud-Optimized Point Cloud (COPC) is an octree-ordered LAZ whose spatial
layout lets a reader fetch only the nodes overlapping a query region over HTTP
range requests. AHN6 ships COPC, so instead of downloading a whole 1 km cell
(tens of millions of points) just to clip a small AOI out of it, this module
reads only the AOI region directly from the remote file. Measured on an Emmen
AHN6 cell: a 1.2%-of-cell region read transferred and decoded in ~2.5 s versus
~74 s for the whole cell.

The result mirrors a `download_tile` for the GeoTiles path: a `raw.laz`
containing the tile's region (its own cell's share of it; the COPC holds only
that cell's points). The existing get_data clip sweep then merges each tile's
region with its neighbours' to fill the halo band exactly as before, so nothing
downstream changes for a streaming source.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pdal
import requests
from shapely.geometry.base import BaseGeometry

from src.get_data.tile_sources import TileSource
from src.stages import DownloadResult, RemoteUnavailableError, StageFailureError
from src.tile_layout import CaseLayout

# Connect/read timeout for the lightweight coverage probe (the heavy range reads
# are driven by PDAL's own HTTP client, not this).
_PROBE_TIMEOUT = (15, 30)


def stream_tile_region(
    tile_id: str,
    output_dir: Path,
    source: TileSource,
    region: BaseGeometry,
    overwrite: bool = False,
) -> DownloadResult:
    """Range-read *region* from *tile_id*'s remote COPC into the case `raw.laz`.

    `output_dir` is the per-case data root (`data/<case>/`). `region` is the
    tile's clip region (core cell + halo margin, intersected with the buffered
    AOI), the same polygon the clip sweep uses, so the streamed `raw.laz` carries
    exactly the points that region needs from this tile's cell.

    Raises
    ------
    RemoteUnavailableError
        The COPC is absent (AHN6 outside first-release coverage, 403/404), or the
        region holds no points for this cell (a graceful skip, not a failure).
    StageFailureError
        The probe or the PDAL range read failed for any other reason.
    """
    tile = CaseLayout(data_dir=output_dir).tile(tile_id)
    tile.dir.mkdir(parents=True, exist_ok=True)

    url = source.streaming_url(tile_id)
    if url is None:  # pragma: no cover - guarded by caller (source.is_streaming)
        raise StageFailureError(f"{source.name} is not a streaming (COPC) source")

    if tile.raw_laz.exists() and not overwrite:
        logging.info(f"[{tile_id}] Skipping existing region read")
        return DownloadResult(laz=tile.raw_laz, lax=None, did_work=False)

    # Coverage probe: AHN6's first release covers the northeast only, and
    # basisdata.nl's Ceph backend can answer 403 for a missing object. Follow
    # redirects (the host 307s to storage) and treat 403/404 as out-of-coverage.
    try:
        resp = requests.head(url, allow_redirects=True, timeout=_PROBE_TIMEOUT)
    except requests.RequestException as e:
        raise StageFailureError(f"COPC probe failed for {url}: {e}") from e
    if resp.status_code in (403, 404):
        raise RemoteUnavailableError(f"COPC HTTP {resp.status_code} at {url}")

    pipeline = [
        {"type": "readers.copc", "filename": url, "polygon": region.wkt},
        {
            "type": "writers.las",
            "filename": str(tile.raw_laz),
            "compression": True,
            "minor_version": 4,
            "dataformat_id": 8,
        },
    ]
    logging.info(f"[{tile_id}] Range-reading AOI region from COPC {url}")
    try:
        n = pdal.Pipeline(json.dumps(pipeline)).execute()
    except RuntimeError as e:
        tile.raw_laz.unlink(missing_ok=True)
        raise StageFailureError(f"COPC region read failed for {tile_id}: {e}") from e

    if n == 0:
        # No points for this cell inside the region (e.g. a tile the AOI only
        # nicks at a corner). Leave no raw.laz so the tile drops out of `present`,
        # matching how an out-of-coverage tile is simply absent downstream.
        tile.raw_laz.unlink(missing_ok=True)
        raise RemoteUnavailableError(f"no COPC points in AOI region for {tile_id}")

    logging.info(f"[{tile_id}] Streamed {n:,} points from COPC region → raw.laz")
    return DownloadResult(laz=tile.raw_laz, lax=None, did_work=True)
