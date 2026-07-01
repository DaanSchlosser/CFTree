# Copyright (C) 2025 Noah Alting
# Licensed under the GNU General Public License v3.0
# See the LICENSE file for more details.

"""Tests for the AHN tile-source registry: from_version and the GeoTiles URLs.

Network-free: from_version only constructs a source (it does not read the index),
the URL checks build no geometry, and the shared-grid check uses a tiny synthetic
shapefile. AHN2 and AHN3 reuse the same GeoTiles host, index and .lax partial
reads as AHN4/AHN5; only the URL prefix differs.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box

from src.get_data.tile_sources import AHN6KMSource, GeoTilesSource, from_version


def _write_synthetic_index(path: Path, n: int = 4) -> None:
    """Write a tiny grid of 1 km cells (ids T0..) as an EPSG:28992 shapefile."""
    coords = [(0, 0), (1000, 0), (0, 1000), (1000, 1000)][:n]
    gpd.GeoDataFrame(
        {"GT_AHNSUB": [f"T{i}" for i in range(n)]},
        geometry=[box(x, y, x + 1000, y + 1000) for x, y in coords],
        crs="EPSG:28992",
    ).to_file(path)


@pytest.mark.parametrize("version", [2, 3, 4, 5])
def test_from_version_returns_geotiles_source(version, tmp_path):
    src = from_version(version, tmp_path)
    assert isinstance(src, GeoTilesSource)
    assert src.name == f"AHN{version}"


def test_from_version_ahn6_is_copc(tmp_path):
    assert isinstance(from_version(6, tmp_path), AHN6KMSource)


@pytest.mark.parametrize("version", [0, 1, 7])
def test_from_version_rejects_unsupported(version, tmp_path):
    with pytest.raises(ValueError, match="Unsupported AHN version"):
        from_version(version, tmp_path)


@pytest.mark.parametrize("version", [2, 3, 4, 5])
def test_geotiles_urls_use_version_prefix(version, tmp_path):
    src = GeoTilesSource(version, tmp_path / "idx.shp")
    tile = "30FZ1_22"
    base = f"https://geotiles.citg.tudelft.nl/AHN{version}_T/{tile}"
    assert src.laz_url(tile) == f"{base}.LAZ"
    assert src.lax_url(tile) == f"{base}.LAX"


def test_geotiles_grid_shared_across_versions(tmp_path):
    """AHN2-AHN5 select the same sub-tiles from one index; only the URL differs."""
    shp = tmp_path / "idx.shp"
    _write_synthetic_index(shp)
    aoi = box(900, 900, 1100, 1100)  # straddles all four cells
    selections = {v: sorted(GeoTilesSource(v, shp).tiles_for_aoi(aoi)) for v in (2, 3, 4, 5)}
    assert selections[2] == selections[3] == selections[4] == selections[5] == ["T0", "T1", "T2", "T3"]
