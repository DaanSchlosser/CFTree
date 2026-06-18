# Copyright (C) 2025 Noah Alting
# Licensed under the GNU General Public License v3.0
# See the LICENSE file for more details.

# src/get_data/tile_sources.py

"""AHN tile-source registry.

AHN4 and AHN5 share the TU Delft GeoTiles host and the shipped
`AHN_subunits_GeoTiles.shp` index, only the URL prefix differs.
AHN6 is a separate first-release product on basisdata.nl with a derived
1x1 km grid with Cloud-Optimized Point Clouds.

"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely.geometry as sg
from shapely.geometry.base import BaseGeometry


class TileSource(ABC):
    name: str
    attribution: str

    @abstractmethod
    def tiles_for_aoi(self, aoi_geom: BaseGeometry) -> list[str]:
        """Tile IDs intersecting `aoi_geom` (EPSG:28992)."""

    @abstractmethod
    def laz_url(self, tile_id: str) -> str:
        """Canonical LAZ URL for `tile_id`."""

    @abstractmethod
    def core_cell(self, tile_id: str) -> BaseGeometry:
        """Non-overlapping nominal cell this tile exclusively owns (EPSG:28992).

        The core cells of all tiles in a source partition the plane, so a tree is
        assigned to exactly one tile by testing which core cell its centroid falls
        in (see `owns_centroids`). This is the single place that knows a version's
        cell geometry; callers never branch on the AHN version.
        """

    def lax_url(self, tile_id: str) -> str | None:
        """Optional `.LAX` sidecar URL; `None` if this source does not serve one."""
        return None

    def owns_centroids(self, tile_id: str, cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
        """Boolean mask of which points `(cx[i], cy[i])` lie in this tile's core cell.

        Uses a half-open rule on the cell's bounding box — south/west edges
        inclusive, north/east exclusive — so a point on a shared cell edge or
        corner is owned by exactly one tile, making the partition exact and
        order-independent. `cx`/`cy` are parallel arrays of EPSG:28992
        coordinates (scalars are accepted and returned as a 0-d array). This is
        the single definition of the ownership rule; callers never re-implement
        the bounds test.
        """
        minx, miny, maxx, maxy = self.core_cell(tile_id).bounds
        cx = np.asarray(cx)
        cy = np.asarray(cy)
        mask: np.ndarray = (minx <= cx) & (cx < maxx) & (miny <= cy) & (cy < maxy)
        return mask


class GeoTilesSource(TileSource):
    """AHN4 / AHN5 sub-tiles served by the TU Delft GeoTiles host."""

    def __init__(self, version: int, index_shp: Path):
        self._version = version
        self._index_shp = Path(index_shp)
        self.name = f"AHN{version}"
        self.attribution = f"AHN{version} (c) Rijkswaterstaat / Waterschappen, CC0 1.0"
        self._base_url = f"https://geotiles.citg.tudelft.nl/AHN{version}_T"
        self._index_gdf: gpd.GeoDataFrame | None = None
        self._cell_by_id: dict[str, BaseGeometry] | None = None

    def _load_index(self) -> gpd.GeoDataFrame:
        """Lazily load and cache the GeoTiles sub-tile index (EPSG:28992).

        Cached on the instance because `core_cell` is queried per tile and the
        index has tens of thousands of rows; re-reading the shapefile per call
        would dominate runtime.
        """
        if self._index_gdf is None:
            if not self._index_shp.exists():
                raise FileNotFoundError(
                    f"AHN sub-tile index not found: {self._index_shp}. "
                    "Expected the shipped resource at resources/AHN_subunits_GeoTiles/."
                )
            self._index_gdf = gpd.read_file(self._index_shp).to_crs("EPSG:28992")
        return self._index_gdf

    def tiles_for_aoi(self, aoi_geom: BaseGeometry) -> list[str]:
        gdf = self._load_index()
        sel = gdf[gdf.intersects(aoi_geom)]
        return [str(tid) for tid in sel["GT_AHNSUB"].tolist()]

    def core_cell(self, tile_id: str) -> BaseGeometry:
        # Nominal sub-tile polygon from the GeoTiles index. The downloaded LAZ
        # tiles overlap, but the index subunits partition the plane; ownership
        # uses the polygon's axis-aligned bounds (see TileSource.owns_centroids).
        if self._cell_by_id is None:
            gdf = self._load_index()
            self._cell_by_id = {
                str(tid): geom for tid, geom in zip(gdf["GT_AHNSUB"].tolist(), gdf.geometry.tolist(), strict=True)
            }
        try:
            return self._cell_by_id[str(tile_id)]
        except KeyError as e:
            raise KeyError(f"Unknown {self.name} sub-tile id: {tile_id}") from e

    def laz_url(self, tile_id: str) -> str:
        return f"{self._base_url}/{tile_id}.LAZ"

    def lax_url(self, tile_id: str) -> str | None:
        return f"{self._base_url}/{tile_id}.LAX"


class AHN6KMSource(TileSource):
    """AHN6 1×1 km kaartbladen served by basisdata.nl (Cloud-Optimized Point Cloud)."""

    name = "AHN6"
    attribution = "AHN6 (c) Rijkswaterstaat / Waterschappen, CC BY 4.0"

    # Grid parameters extracted from the AHN bladwijzer's sheets-DoENGwi0.bin:
    # tile lower-left = (GRID_ORIGIN_X + col*TILE_SIZE, GRID_ORIGIN_Y + row*TILE_SIZE).
    GRID_ORIGIN_X = 12_000
    GRID_ORIGIN_Y = 304_000
    TILE_SIZE = 1_000
    _BASE_URL = "https://basisdata.nl/hwh-ahn/AHN6/01_LAZ"

    def tiles_for_aoi(self, aoi_geom: BaseGeometry) -> list[str]:
        minx, miny, maxx, maxy = aoi_geom.bounds
        x_start = self._floor_to_grid(minx, self.GRID_ORIGIN_X)
        y_start = self._floor_to_grid(miny, self.GRID_ORIGIN_Y)
        x_stop = self._floor_to_grid(maxx, self.GRID_ORIGIN_X) + self.TILE_SIZE
        y_stop = self._floor_to_grid(maxy, self.GRID_ORIGIN_Y) + self.TILE_SIZE

        ids: list[str] = []
        for x in range(x_start, x_stop, self.TILE_SIZE):
            for y in range(y_start, y_stop, self.TILE_SIZE):
                if sg.box(x, y, x + self.TILE_SIZE, y + self.TILE_SIZE).intersects(aoi_geom):
                    ids.append(f"{x:06d}_{y:06d}")
        return ids

    def laz_url(self, tile_id: str) -> str:
        return f"{self._BASE_URL}/AHN6_2025_C_{tile_id}.COPC.LAZ"

    def core_cell(self, tile_id: str) -> BaseGeometry:
        # tile_id "x_y" is the SW corner of the 1x1 km cell; reuses the same box
        # math and id format as tiles_for_aoi.
        x_str, y_str = tile_id.split("_")
        x, y = int(x_str), int(y_str)
        return sg.box(x, y, x + self.TILE_SIZE, y + self.TILE_SIZE)

    @classmethod
    def _floor_to_grid(cls, value: float, origin: int) -> int:
        return origin + int(math.floor((value - origin) / cls.TILE_SIZE)) * cls.TILE_SIZE


def from_version(version: int, resources_dir: Path) -> TileSource:
    """Return the `TileSource` for an AHN release version (4, 5, or 6)."""
    if version in (4, 5):
        return GeoTilesSource(
            version=version,
            index_shp=resources_dir / "AHN_subunits_GeoTiles" / "AHN_subunits_GeoTiles.shp",
        )
    if version == 6:
        logging.info(
            "AHN6 first release covers the northeast of the Netherlands only. "
            "AOIs outside that footprint will fail tile probes; "
            "use --ahn-version 4 or 5 to fall back."
        )
        return AHN6KMSource()
    raise ValueError(f"Unsupported AHN version: {version}. Supported: 4, 5, 6.")
