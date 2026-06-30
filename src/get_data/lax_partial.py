# Copyright (C) 2025 Noah Alting
# Licensed under the GNU General Public License v3.0
# See the LICENSE file for more details.
#
# The LAX quadtree parser (`LaxIndex`) is a port of laxpy
# (https://github.com/brycefrank/laxpy, MIT License, (c) Bryce Frank). Only the
# index-reading and cell-geometry arithmetic are reused; the remote range read
# and the get_data integration are original.

# src/get_data/lax_partial.py

"""Partial range read of a GeoTiles AHN4/AHN5 tile, driven by its `.lax` index.

AHN6 ships Cloud-Optimized Point Clouds, so a small AOI is read straight from
the remote file (see `stream_copc.py`). AHN4 and AHN5 on the TU Delft GeoTiles
host are plain LAZ instead, with a sidecar `.lax` spatial index. A plain LAZ
still supports two things that together allow a partial read: the host honours
HTTP range requests, and the LAZ chunk table lets `laspy.seek` jump to the
compressed chunk holding a given point without reading the chunks before it. So
this module reads the small `.lax` (about 0.1 MB), turns the AOI into the set of
point-index intervals that cover it, and pulls only the chunks those intervals
touch over HTTP range requests.

How much this saves is bounded by the file, not by this code. A GeoTiles LAZ is
not spatially ordered the way a COPC is, so a region's points are scattered over
a sizeable fraction of the file's chunks: a 250 m AOI touches about a quarter of
a tile and a 400 m AOI about half. The read is therefore worth it for small
areas (a few times less data) and not for large ones, so `read_region` estimates
the fraction from the index first and downloads the whole tile (into the shared
cache, reused across areas) when a partial read would not pay. Either way the
exact crop happens later in the clip sweep, so the only thing the index decides
is which bytes to fetch; any index or read failure falls back to the whole-tile
download, and the result is always the same points a whole-tile clip would give.
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path

import laspy
import numpy as np
import requests
from shapely.geometry.base import BaseGeometry

from src.get_data.download_geotiles import _default_cache_dir, _ensure_cached, download_tile
from src.get_data.tile_sources import TileSource
from src.stages import DownloadResult, RemoteUnavailableError, StageFailureError
from src.tile_layout import CaseLayout

# Above this estimated fraction of a tile a partial read stops paying off (more,
# smaller range requests for little saved data), so the whole tile is downloaded
# into the shared cache instead, where it is also reused by overlapping areas.
_PARTIAL_FRACTION_MAX = 0.85

# A small region's points are scattered across the file, so the index returns many
# short point-index runs. Runs whose index gap is at most this many points are
# merged into one (their gap is read and then discarded by the region filter) so a
# cluster of nearby chunks becomes one sequential read instead of many requests.
# Kept near one LAZ chunk (~50k points): larger values bridge big holes and pull
# far more data than the region needs, which (with the per-run read-ahead window)
# is what actually dominates the time on a plain, non-spatially-ordered tile.
_RUN_MERGE_GAP = 60_000

# HTTP range-read tuning. The connect timeout fast-fails an unreachable host; the
# read timeout is generous because a cold GeoTiles tile can stall before the
# first byte. Transient failures get a few linear-backoff retries. A sequential
# read is served by a read-ahead block so a contiguous run is pulled in a few
# large requests rather than one per LAZ chunk; a non-sequential read (a seek to a
# new run) fetches only a small block, so scattered reads do not over-fetch.
_CONNECT_TIMEOUT = 15
_READ_TIMEOUT = 120
_RANGE_ATTEMPTS = 3
_RETRY_BACKOFF_S = 3
_READAHEAD = 8 << 20  # 8 MiB sequential read-ahead
_MIN_FETCH = 256 << 10  # 256 KiB minimum range request


# ---------------------------------------------------------------------------
# LAX index (ported from laxpy, MIT)
# ---------------------------------------------------------------------------
class LaxIndex:
    """A parsed GeoTiles `.lax` quadtree: cell -> point-index intervals.

    The `.lax` layout is LAStools' `LASindex` (a `LASquadtree` followed by a
    `LASinterval`). Following laxpy, the cell geometry is recovered purely from
    the file bounding box and each cell's position in the quadtree, so only the
    bounding box and the per-cell intervals are read; the quadtree's own level and
    cell-size fields are not needed.
    """

    def __init__(self, raw: bytes):
        words = np.frombuffer(raw, dtype="<u4")
        if words.size < 20 or raw[:4] != b"LASX":
            raise ValueError("not a LASX .lax index")
        self._words = words
        # Bounding box (min_x, max_x, min_y, max_y) sits at 32-bit words 9..12.
        self.bbox = tuple(float(v) for v in np.frombuffer(raw[36:52], dtype="<f4"))
        self.number_cells = int(words[15])
        self.cells = self._parse_cells()
        if not self.cells:
            raise ValueError("empty .lax index")
        self.max_index = max(self.cells)

    def _parse_cells(self) -> dict[int, np.ndarray]:
        """Read each cell's `[cell_id, n_intervals, n_points, (start,end)*]` record."""
        w = self._words
        cells: dict[int, np.ndarray] = {}
        start = 19
        n = int(w[17])  # interval count of the first cell
        for i in range(self.number_cells):
            intervals = np.asarray(w[start : start + n * 2], dtype=np.int64)
            cell_id = int(w[start - 3])
            cells[cell_id] = intervals
            start = start + n * 2 + 3
            if i != self.number_cells - 1:
                n = int(w[start - 2])
        return cells

    # --- quadtree cell geometry (laxpy LAXTree algorithm) ------------------
    def _level_sizes(self) -> list[int]:
        sizes, i = [], 0
        while True:
            m = 4**i
            sizes.append(m)
            if m > self.max_index:
                return sizes
            i += 1

    def _level_edges(self) -> dict[int, tuple[int, int]]:
        left = np.cumsum(self._level_sizes())
        right = left * 4
        return {k + 1: (int(left[k]), int(right[k])) for k in range(len(left))}

    @staticmethod
    def _parent(cell_index: int, edges: dict[int, tuple[int, int]]) -> int:
        for lvl, (lo, hi) in edges.items():
            if lo <= cell_index <= hi:
                offset = (cell_index - lo) + 1
                parent_offset = -(-offset // 4)  # ceil division
                if (lvl - 1) in edges:
                    return edges[lvl - 1][0] + parent_offset - 1
                return 0
        return 0

    def _cell_bbox(self, cell_index: int, edges: dict[int, tuple[int, int]]) -> tuple[float, float, float, float]:
        trace = []
        ci = cell_index
        while ci != 0:
            pos = ((ci - 1) % 4) + 1  # quadrant of this cell within its parent
            ci = self._parent(ci, edges)
            trace.append(pos)
        minx, maxx, miny, maxy = self.bbox
        xw = (maxx - minx) / 2
        yw = (maxy - miny) / 2
        for pos in reversed(trace):
            if pos == 2:
                minx += xw
            elif pos == 3:
                miny += yw
            elif pos == 4:
                minx, miny = minx + xw, miny + yw
            xw /= 2
            yw /= 2
        return (minx, miny, minx + xw * 2, miny + yw * 2)  # (minx, miny, maxx, maxy)

    # --- selection --------------------------------------------------------
    def select(self, region: BaseGeometry) -> tuple[list[tuple[int, int]], float]:
        """Return merged `(start, count)` point runs for *region*, and the fraction
        of the tile's points they cover (an estimate of the bytes a read pulls)."""
        from shapely.geometry import box

        edges = self._level_edges()
        minx, miny, maxx, maxy = region.bounds
        rbox = box(minx, miny, maxx, maxy)
        chosen: list[tuple[int, int]] = []
        total = 0
        selected = 0
        for cell_index, raw in self.cells.items():
            cell_pts = 0
            for j in range(0, len(raw), 2):
                cell_pts += int(raw[j + 1]) - int(raw[j]) + 1
            total += cell_pts
            cminx, cminy, cmaxx, cmaxy = self._cell_bbox(cell_index, edges)
            if box(cminx, cminy, cmaxx, cmaxy).intersects(rbox):
                selected += cell_pts
                for j in range(0, len(raw), 2):
                    chosen.append((int(raw[j]), int(raw[j + 1])))  # inclusive [start, end]
        fraction = (selected / total) if total else 1.0
        return _merge_runs(chosen, _RUN_MERGE_GAP), fraction


def _merge_runs(intervals: list[tuple[int, int]], gap: int = 0) -> list[tuple[int, int]]:
    """Merge inclusive `[start, end]` intervals into sorted `(start, count)` runs.

    Runs whose gap is at most *gap* points are merged into one, trading reading
    (and later discarding) the gap for one fewer HTTP request.
    """
    if not intervals:
        return []
    intervals.sort()
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1] + 1 + gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e - s + 1) for s, e in merged]


# ---------------------------------------------------------------------------
# Seekable HTTP file
# ---------------------------------------------------------------------------
class HttpRangeFile(io.RawIOBase):
    """A seekable, read-only file over HTTP range requests.

    `laspy.open` needs a seekable stream; given one it reads only the header, the
    chunk table, and the chunks a `seek` lands in, so wrapping the remote LAZ in
    this object turns `laspy.seek`/`read_points` into range reads of just those
    bytes. Transient failures are retried; redirects are followed (basisdata.nl
    and some mirrors 307 to storage).
    """

    def __init__(self, url: str, session: requests.Session | None = None):
        self._url = url
        self._pos = 0
        self._session = session or requests.Session()
        resp = self._session.head(url, timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT), allow_redirects=True)
        if resp.status_code in (403, 404):
            raise RemoteUnavailableError(f"LAZ HTTP {resp.status_code} at {url}")
        resp.raise_for_status()
        self._size = int(resp.headers["Content-Length"])
        self.bytes_fetched = 0
        self.n_requests = 0
        # Read-ahead cache: bytes for file range [_buf_start, _buf_start + len(_buf)).
        self._buf = b""
        self._buf_start = 0
        self._last_end = -1  # file offset just past the previous read (sequentiality)
        # Soft upper bound for read-ahead, set per run so a run's read-ahead never
        # spills far past the bytes that run needs. Never blocks a read: a read
        # always gets at least the bytes it asked for.
        self._window_end = self._size

    def set_window(self, end_offset: int) -> None:
        self._window_end = min(end_offset, self._size)

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self._size + offset
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self._size - self._pos
        if size <= 0 or self._pos >= self._size:
            return b""
        want_end = min(self._pos + size, self._size)
        if not (self._buf_start <= self._pos and want_end <= self._buf_start + len(self._buf)):
            # Cache miss: read ahead on a sequential continuation, fetch a small
            # block on a fresh seek (so scattered runs do not over-fetch). Cap the
            # read-ahead at the current run's window, but always serve the full
            # request (the window only trims speculative bytes, never needed ones).
            sequential = self._pos == self._last_end
            block = _READAHEAD if sequential else _MIN_FETCH
            fetch_end = min(self._pos + max(size, block), self._size)
            fetch_end = max(want_end, min(fetch_end, self._window_end))
            self._buf = self._fetch(self._pos, fetch_end)
            self._buf_start = self._pos
        off = self._pos - self._buf_start
        data = self._buf[off : off + size]
        self._pos += len(data)
        self._last_end = self._pos
        return data

    def _fetch(self, start: int, end: int) -> bytes:
        """Range-read bytes for `[start, end)` with retries; return them."""
        headers = {"Range": f"bytes={start}-{end - 1}"}
        last: Exception | None = None
        for attempt in range(1, _RANGE_ATTEMPTS + 1):
            try:
                r = self._session.get(
                    self._url, headers=headers, timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT), allow_redirects=True
                )
                r.raise_for_status()
                data = r.content
                self.bytes_fetched += len(data)
                self.n_requests += 1
                return data
            except (requests.ConnectionError, requests.Timeout, requests.ChunkedEncodingError) as e:
                last = e
                if attempt < _RANGE_ATTEMPTS:
                    time.sleep(_RETRY_BACKOFF_S * attempt)
        raise StageFailureError(f"range read failed at {self._url}: {last}")

    def readinto(self, b) -> int:
        data = self.read(len(b))
        n = len(data)
        b[:n] = data
        return n


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def read_region(
    tile_id: str,
    output_dir: Path,
    source: TileSource,
    region: BaseGeometry,
    overwrite: bool = False,
) -> DownloadResult:
    """Provide *region*'s points for *tile_id* as the case `raw.laz`, cheaply.

    Reads only the region from the remote LAZ via its `.lax` index when that pulls
    materially less than the whole tile; otherwise (or on any failure) downloads
    the whole tile into the shared cache and provisions it, exactly like the plain
    GeoTiles path. The exact crop to *region* still happens in the clip sweep, so
    the produced points are a superset bounded by `region.bounds` and the
    downstream result is identical to a whole-tile clip.

    Raises
    ------
    RemoteUnavailableError
        The tile is absent at the remote host, or holds no points in *region*.
    StageFailureError
        The whole-tile fallback download failed.
    """
    tile = CaseLayout(data_dir=output_dir).tile(tile_id)
    tile.dir.mkdir(parents=True, exist_ok=True)

    if tile.raw_laz.exists() and not overwrite:
        logging.info(f"[{tile_id}] Skipping existing region read")
        return DownloadResult(laz=tile.raw_laz, lax=None, did_work=False)

    # If the whole tile is already cached (a previous whole download), use it: a
    # local clip is faster than any range read and reuses the immutable bytes.
    cache_dir = _default_cache_dir(output_dir, source)
    if (cache_dir / f"{tile_id}.LAZ").exists():
        logging.info(f"[{tile_id}] Whole tile already cached; using it instead of a range read")
        return download_tile(tile_id, output_dir, source, overwrite=overwrite, cache_dir=cache_dir)

    try:
        return _partial_or_whole(tile_id, output_dir, source, region, overwrite, cache_dir, tile)
    except (RemoteUnavailableError, StageFailureError):
        raise
    except Exception as e:  # noqa: BLE001 - any partial-read fault falls back to the whole tile
        logging.warning(f"[{tile_id}] Partial read failed ({e.__class__.__name__}: {e}); downloading whole tile")
        return download_tile(tile_id, output_dir, source, overwrite=overwrite, cache_dir=cache_dir)


def _partial_or_whole(tile_id, output_dir, source, region, overwrite, cache_dir, tile) -> DownloadResult:
    lax_url = source.lax_url(tile_id)
    if lax_url is None:
        return download_tile(tile_id, output_dir, source, overwrite=overwrite, cache_dir=cache_dir)

    # Fetch (and cache) the small .lax, then size up the read.
    lax_cache = cache_dir / f"{tile_id}.LAX"
    _ensure_cached(url=lax_url, cache=lax_cache, tile_id=tile_id, label="LAX", required=True)
    index = LaxIndex(lax_cache.read_bytes())
    runs, fraction = index.select(region)

    if not runs:
        raise RemoteUnavailableError(f"no points in AOI region for {tile_id}")
    if fraction > _PARTIAL_FRACTION_MAX:
        logging.info(
            f"[{tile_id}] Region covers ~{fraction:.0%} of the tile; downloading whole tile instead of a range read"
        )
        return download_tile(tile_id, output_dir, source, overwrite=overwrite, cache_dir=cache_dir)

    logging.info(f"[{tile_id}] Range-reading ~{fraction:.0%} of the tile from its .lax index ({len(runs)} run(s))")
    n, mb, reqs = _read_runs_to_laz(source.laz_url(tile_id), runs, region, tile.raw_laz)
    if n == 0:
        tile.raw_laz.unlink(missing_ok=True)
        raise RemoteUnavailableError(f"no points in AOI region for {tile_id}")
    logging.info(f"[{tile_id}] Range-read {n:,} points ({mb:.0f} MB, {reqs} requests) → raw.laz")
    return DownloadResult(laz=tile.raw_laz, lax=None, did_work=True)


def _read_runs_to_laz(
    laz_url: str, runs: list[tuple[int, int]], region: BaseGeometry, dest: Path
) -> tuple[int, float, int]:
    """Range-read `runs` from the remote LAZ, keep points inside `region.bounds`, write `dest`.

    Returns the kept point count, the megabytes fetched, and the request count.
    """
    minx, miny, maxx, maxy = region.bounds
    http = HttpRangeFile(laz_url)
    reader = laspy.open(http)
    header = reader.header
    sx, sy = header.scales[0], header.scales[1]
    ox, oy = header.offsets[0], header.offsets[1]

    # Estimate the byte offset of a point index from the average compressed
    # density, so each run's read-ahead is capped near the bytes that run needs.
    # The estimate only trims speculative read-ahead, so its accuracy affects
    # speed, never correctness.
    data_start = header.offset_to_point_data
    bytes_per_point = max(1.0, (http._size - data_start) / max(1, header.point_count))

    def window_for(start: int, count: int) -> int:
        return data_start + int((start + count) * bytes_per_point) + (2 << 20)  # +2 MiB safety

    kept: list[np.ndarray] = []
    for start, count in runs:
        http.set_window(window_for(start, count))
        reader.seek(start)
        rec = reader.read_points(count)
        X = np.asarray(rec.X)
        Y = np.asarray(rec.Y)
        rx = X * sx + ox
        ry = Y * sy + oy
        mask = (rx >= minx) & (rx <= maxx) & (ry >= miny) & (ry <= maxy)
        if mask.any():
            kept.append(rec.array[mask])

    mb, reqs = http.bytes_fetched / 1e6, http.n_requests
    if not kept:
        return 0, mb, reqs
    merged = np.concatenate(kept)
    out = laspy.LasData(header)
    out.points = laspy.ScaleAwarePointRecord(merged, header.point_format, scales=header.scales, offsets=header.offsets)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.write(dest)
    return len(merged), mb, reqs
