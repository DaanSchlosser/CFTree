# Copyright (C) 2025 Noah Alting
# Licensed under the GNU General Public License v3.0
# See the LICENSE file for more details.

"""Tests for the get_data caches: the tile-index cache and the shared tile cache.

Both are network-free: the index test builds a tiny synthetic shapefile, and the
shared-cache test stubs the HTTP download, so they run in CI without touching
GeoTiles or basisdata.nl.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import geopandas as gpd
from shapely.geometry import box

import src.get_data.download_geotiles as dl
from src.get_data.download_geotiles import download_tile
from src.get_data.tile_sources import GeoTilesSource


def _write_synthetic_index(path: Path, n: int = 4) -> None:
    """Write a tiny grid of 1 km cells (ids T0..) as an EPSG:28992 shapefile."""
    coords = [(0, 0), (1000, 0), (0, 1000), (1000, 1000)][:n]
    gdf = gpd.GeoDataFrame(
        {"GT_AHNSUB": [f"T{i}" for i in range(n)]},
        geometry=[box(x, y, x + 1000, y + 1000) for x, y in coords],
        crs="EPSG:28992",
    )
    gdf.to_file(path)


# ---------------------------------------------------------------------------
# A: tile-index cache
# ---------------------------------------------------------------------------
def test_index_cache_builds_and_selection_matches_legacy(tmp_path):
    shp = tmp_path / "idx.shp"
    _write_synthetic_index(shp)
    cache = shp.with_name(shp.stem + ".bounds.npz")

    # Cold: builds the sidecar. Warm (fresh instance): reads it. Same answers.
    aoi = box(900, 900, 1100, 1100)  # straddles all four cells
    cold = sorted(GeoTilesSource(5, shp).tiles_for_aoi(aoi))
    assert cache.exists()
    warm = sorted(GeoTilesSource(5, shp).tiles_for_aoi(aoi))
    assert cold == warm == ["T0", "T1", "T2", "T3"]

    src = GeoTilesSource(5, shp)
    # A single-cell AOI selects exactly that cell; selection equals a legacy
    # full-read intersects on the same geometries.
    gdf = gpd.read_file(shp).to_crs("EPSG:28992")
    for probe in (box(100, 100, 200, 200), box(1200, 100, 1300, 200), box(400, 1400, 600, 1600)):
        legacy = sorted(str(t) for t in gdf[gdf.intersects(probe)]["GT_AHNSUB"])
        assert sorted(src.tiles_for_aoi(probe)) == legacy

    # core_cell round-trips through the cache as the exact cell polygon.
    assert tuple(src.core_cell("T3").bounds) == (1000.0, 1000.0, 2000.0, 2000.0)


def test_index_cache_rebuilds_when_shapefile_changes(tmp_path):
    shp = tmp_path / "idx.shp"
    big = box(-10, -10, 5000, 5000)

    _write_synthetic_index(shp, n=4)
    assert len(GeoTilesSource(5, shp).tiles_for_aoi(big)) == 4

    # Replace the index with fewer cells and bump the mtime: a new source must
    # rebuild from the shapefile rather than serve the stale 4-cell cache.
    _write_synthetic_index(shp, n=2)
    st = shp.stat()
    os.utime(shp, (st.st_atime, st.st_mtime + 1000))
    assert len(GeoTilesSource(5, shp).tiles_for_aoi(big)) == 2


# ---------------------------------------------------------------------------
# B: shared immutable tile cache
# ---------------------------------------------------------------------------
class _FakeSource:
    name = "AHNX"

    def laz_url(self, tile_id: str) -> str:
        return f"http://example/{tile_id}.LAZ"

    def lax_url(self, tile_id: str) -> None:
        return None  # exercise the LAZ-only path


def test_shared_cache_dedups_across_cases_and_ignores_overwrite(tmp_path):
    calls: list[str] = []

    def fake_stream(url: str, dest: Path) -> None:
        calls.append(url)
        dest.write_bytes(b"LAZ:" + url.encode())

    data_root = tmp_path / "data"
    case_a = data_root / "caseA"
    case_b = data_root / "caseB"
    src = _FakeSource()

    with mock.patch.object(dl, "_stream_download", fake_stream):
        download_tile("30FZ1_22", case_a, src, overwrite=False)  # miss -> downloads
        download_tile("30FZ1_22", case_b, src, overwrite=True)   # cache hit despite overwrite
        download_tile("30HN1_02", case_a, src, overwrite=False)  # different tile -> downloads

    # The shared tile was fetched exactly once across the two cases.
    assert calls.count("http://example/30FZ1_22.LAZ") == 1
    assert calls.count("http://example/30HN1_02.LAZ") == 1

    raw_a = case_a / "tiles" / "30FZ1_22" / "raw.laz"
    raw_b = case_b / "tiles" / "30FZ1_22" / "raw.laz"
    assert raw_a.read_bytes() == raw_b.read_bytes()
    assert (data_root / ".ahn_cache" / "AHNX" / "30FZ1_22.LAZ").exists()
