# Copyright (C) 2025 Noah Alting
# Licensed under the GNU General Public License v3.0
# See the LICENSE file for more details.

"""Tests for the two seams the cross-tile dedup rests on.

``owns_centroids`` is the single definition of centroid ownership (a tree is
kept by exactly one tile), and ``_reconcile_cross_tile_duplicates`` is the
geometric backstop for crowns wider than the halo margin. Both decide which
physical trees exist in the output, so a flipped inequality or a geopandas
API drift must fail a test rather than silently double-count or drop trees.
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.get_data.tile_sources import AHN6KMSource  # noqa: E402
from src.segmentation.generalize_forest_ids import _reconcile_cross_tile_duplicates  # noqa: E402


# ---------------------------------------------------------------------
# Centroid ownership: the core cells partition the plane exactly
# ---------------------------------------------------------------------
def test_owns_centroids_partitions_the_plane_exactly() -> None:
    # A 2x2 block of AHN6 cells. Probe strictly-interior points, points on the
    # shared edges, and the shared corner: every probe must be owned by exactly
    # one tile (south/west edges inclusive, north/east exclusive). A flipped
    # `<`/`<=` would own boundary trees twice (a double count) or never (a
    # dropped tree) and no other test would notice.
    source = AHN6KMSource()
    tiles = ["012000_304000", "013000_304000", "012000_305000", "013000_305000"]
    cx = np.array([12500.0, 13500.0, 12500.0, 13500.0, 13000.0, 12500.0, 13000.0, 12000.0, 13999.9])
    cy = np.array([304500.0, 304500.0, 305500.0, 305500.0, 304500.0, 305000.0, 305000.0, 304000.0, 305999.9])

    owners = np.zeros(len(cx), dtype=int)
    for tid in tiles:
        owners += source.owns_centroids(tid, cx, cy).astype(int)
    assert owners.tolist() == [1] * len(cx)


def test_owns_centroids_accepts_scalars() -> None:
    source = AHN6KMSource()
    assert bool(source.owns_centroids("012000_304000", 12000.0, 304000.0))
    assert not bool(source.owns_centroids("012000_304000", 13000.0, 304500.0))


# ---------------------------------------------------------------------
# AHN6 grid maths: tile selection and cell round-trip
# ---------------------------------------------------------------------
def test_ahn6_tiles_for_aoi_and_core_cell_round_trip() -> None:
    source = AHN6KMSource()

    # Strictly inside one cell: exactly that cell.
    assert source.tiles_for_aoi(box(12100, 304100, 12900, 304900)) == ["012000_304000"]

    # Spanning a km line: both cells, and each id round-trips through
    # core_cell back to its own SW corner.
    ids = source.tiles_for_aoi(box(12500, 304100, 13500, 304900))
    assert ids == ["012000_304000", "013000_304000"]
    for tid in ids:
        minx, miny, maxx, maxy = source.core_cell(tid).bounds
        assert f"{int(minx):06d}_{int(miny):06d}" == tid
        assert (maxx - minx, maxy - miny) == (source.TILE_SIZE, source.TILE_SIZE)


# ---------------------------------------------------------------------
# IoU backstop: one crown straddling a boundary is counted once
# ---------------------------------------------------------------------
def _hulls(rows: list[tuple[str, int, object]]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "tile_id": [r[0] for r in rows],
            "tid": [r[1] for r in rows],
            "geometry": [r[2] for r in rows],
        },
        crs="EPSG:28992",
    )


def test_reconcile_drops_the_smaller_of_a_cross_tile_duplicate() -> None:
    # The same physical crown, truncated differently in two tiles: IoU 0.9.
    hulls = _hulls(
        [
            ("A", 1, box(0, 0, 10, 10)),
            ("B", 1, box(0, 0, 9, 10)),
        ]
    )
    out = _reconcile_cross_tile_duplicates(hulls)
    assert len(out) == 1
    assert out.iloc[0]["tile_id"] == "A"  # the larger crown is kept


def test_reconcile_keeps_distinct_neighbours_and_same_tile_overlaps() -> None:
    hulls = _hulls(
        [
            # Cross-tile pair barely touching: IoU well under 0.5, keep both.
            ("A", 1, box(20, 0, 30, 10)),
            ("B", 2, box(29, 0, 39, 10)),
            # Same-tile heavy overlap: tids are unique within a tile, keep both.
            ("A", 3, box(50, 0, 60, 10)),
            ("A", 4, box(50, 0, 59, 10)),
        ]
    )
    out = _reconcile_cross_tile_duplicates(hulls)
    assert len(out) == 4


def test_reconcile_passes_through_trivial_inputs() -> None:
    one = _hulls([("A", 1, box(0, 0, 1, 1))])
    assert _reconcile_cross_tile_duplicates(one).equals(one)
