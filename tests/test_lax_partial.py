# Copyright (C) 2025 Noah Alting
# Licensed under the GNU General Public License v3.0
# See the LICENSE file for more details.

"""Network-free tests for the AHN4/AHN5 partial range read (`lax_partial`).

The quadtree cell geometry, the run merging, and the HTTP range file's read-ahead
and per-run window are all exercised without touching the network: the index is
built by hand and the range file is fed by a fake session backed by a byte
buffer.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import box

import src.get_data.lax_partial as lp
from src.get_data.lax_partial import HttpRangeFile, LaxIndex, _merge_runs


def _index(cells: dict[int, list[int]]) -> LaxIndex:
    """A LaxIndex over the unit-square-scaled bbox (0..100) with given cells."""
    idx = object.__new__(LaxIndex)
    idx._words = None
    idx.bbox = (0.0, 100.0, 0.0, 100.0)  # (min_x, max_x, min_y, max_y)
    idx.cells = {k: np.asarray(v, dtype=np.int64) for k, v in cells.items()}
    idx.number_cells = len(cells)
    idx.max_index = max(cells)
    return idx


# ---------------------------------------------------------------------------
# run merging
# ---------------------------------------------------------------------------
def test_merge_runs_adjacent_and_gap():
    # inclusive [start, end] -> (start, count); adjacent intervals always merge
    assert _merge_runs([(0, 9), (10, 19)]) == [(0, 20)]
    # a gap larger than the tolerance stays split
    assert _merge_runs([(0, 9), (50, 59)], gap=10) == [(0, 10), (50, 10)]
    # a gap within the tolerance merges (reading the gap)
    assert _merge_runs([(0, 9), (15, 19)], gap=10) == [(0, 20)]
    assert _merge_runs([]) == []


# ---------------------------------------------------------------------------
# quadtree cell geometry + region selection
# ---------------------------------------------------------------------------
def test_cell_bbox_quadrants():
    idx = _index({1: [0, 0], 2: [0, 0], 3: [0, 0], 4: [0, 0]})
    edges = idx._level_edges()
    # level-1 children are the four quadrants (pos 1=SW, 2=SE, 3=NW, 4=NE)
    assert idx._cell_bbox(1, edges) == (0.0, 0.0, 50.0, 50.0)
    assert idx._cell_bbox(2, edges) == (50.0, 0.0, 100.0, 50.0)
    assert idx._cell_bbox(3, edges) == (0.0, 50.0, 50.0, 100.0)
    assert idx._cell_bbox(4, edges) == (50.0, 50.0, 100.0, 100.0)


def test_select_picks_intersecting_cells_and_fraction():
    # four quadrant cells, 100 points each (disjoint index ranges)
    idx = _index({1: [0, 99], 2: [100, 199], 3: [200, 299], 4: [300, 399]})

    # a box wholly inside the SW quadrant selects only cell 1
    runs, frac = idx.select(box(10, 10, 20, 20))
    assert runs == [(0, 100)]
    assert frac == 100 / 400

    # a box straddling the centre touches all four; runs merge to one span
    runs, frac = idx.select(box(40, 40, 60, 60))
    assert runs == [(0, 400)]
    assert frac == 1.0


# ---------------------------------------------------------------------------
# HTTP range file: correctness, read-ahead, window
# ---------------------------------------------------------------------------
class _FakeResp:
    def __init__(self, status, headers=None, content=b""):
        self.status_code = status
        self.headers = headers or {}
        self.content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f"HTTP {self.status_code}")


class _FakeSession:
    """Serves byte ranges from an in-memory buffer and counts GET requests."""

    def __init__(self, data: bytes):
        self.data = data
        self.gets: list[tuple[int, int]] = []

    def head(self, url, timeout=None, allow_redirects=True):
        return _FakeResp(200, {"Content-Length": str(len(self.data))})

    def get(self, url, headers=None, timeout=None, allow_redirects=True):
        rng = headers["Range"].removeprefix("bytes=")
        start, end = (int(x) for x in rng.split("-"))
        self.gets.append((start, end))
        return _FakeResp(200, content=self.data[start : end + 1])


def _make_file(data: bytes, monkeypatch, readahead=64, min_fetch=16):
    monkeypatch.setattr(lp, "_READAHEAD", readahead)
    monkeypatch.setattr(lp, "_MIN_FETCH", min_fetch)
    return HttpRangeFile("http://x/tile.laz", session=_FakeSession(data))


def test_range_file_reads_correct_bytes(monkeypatch):
    data = bytes(range(256)) * 8  # 2048 bytes
    f = _make_file(data, monkeypatch)
    f.seek(1000)
    assert f.read(50) == data[1000:1050]
    f.seek(0)
    assert f.read(10) == data[0:10]


def test_range_file_readahead_coalesces_sequential(monkeypatch):
    data = bytes(range(256)) * 8
    f = _make_file(data, monkeypatch, readahead=256)
    f.seek(0)
    chunks = b"".join(f.read(16) for _ in range(8))  # 128 bytes, sequential 16-byte reads
    assert chunks == data[0:128]
    # the first (post-seek) read fetches a small block, then one read-ahead covers
    # the rest: 8 reads collapse to at most 2 requests
    assert len(f._session.gets) <= 2


def test_range_file_window_caps_readahead(monkeypatch):
    data = bytes(range(256)) * 8
    f = _make_file(data, monkeypatch, readahead=1024, min_fetch=16)
    f.seek(0)
    f.read(16)  # fresh seek: small fetch (0, 15)
    f.set_window(40)  # cap read-ahead at offset 40
    f.read(16)  # sequential continuation: read-ahead would reach 1040 but is capped
    assert f._session.gets[1] == (16, 39)


def test_range_file_nonsequential_fetches_small(monkeypatch):
    data = bytes(range(256)) * 8
    f = _make_file(data, monkeypatch, readahead=1024, min_fetch=16)
    f.seek(500)
    f.read(4)  # a fresh seek: fetch only the small min block, not the big read-ahead
    start, end = f._session.gets[0]
    assert start == 500 and (end - start + 1) == 16
