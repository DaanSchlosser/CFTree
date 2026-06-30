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
import shapely
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
    """AHN4 / AHN5 sub-tiles served by the TU Delft GeoTiles host.

    The shipped sub-tile index is a national shapefile of ~76k polygons (a
    ~49 MB ``.shp`` / ``.dbf`` pair). Reading and reprojecting all of it with
    GeoPandas on every run cost ~30-60 s on a virtualised mount and dominated
    ``get_data`` for a small AOI. The geometry is the only thing any caller
    needs (``tiles_for_aoi`` intersects it; ``core_cell`` / ``owns_centroids``
    read its bounds), so the first run distils the index to a slim
    ``<index>.bounds.npz`` sidecar holding just the sub-tile ids and their
    geometries (as WKB), keyed by the shapefile's size and mtime. Later runs
    load that in well under a second and rebuild it automatically when the
    shapefile changes. The sidecar is a pure derivation of the shipped
    shapefile, so it is safe to delete and is git-ignored.
    """

    _CACHE_SUFFIX = ".bounds.npz"

    def __init__(self, version: int, index_shp: Path):
        self._version = version
        self._index_shp = Path(index_shp)
        self.name = f"AHN{version}"
        self.attribution = f"AHN{version} (c) Rijkswaterstaat / Waterschappen, CC0 1.0"
        self._base_url = f"https://geotiles.citg.tudelft.nl/AHN{version}_T"
        self._ids: list[str] | None = None
        self._geoms: np.ndarray | None = None  # parallel array of shapely polygons (EPSG:28992)
        self._row_by_id: dict[str, int] | None = None
        self._tree: shapely.STRtree | None = None

    def _cache_path(self) -> Path:
        return self._index_shp.with_name(self._index_shp.stem + self._CACHE_SUFFIX)

    def _index_stamp(self) -> tuple[int, int]:
        st = self._index_shp.stat()
        return int(st.st_size), int(st.st_mtime)

    def _load_index(self) -> None:
        """Populate ``_ids`` / ``_geoms`` from the slim cache, building it once.

        Cached on the instance because ``core_cell`` is queried per tile and the
        index has tens of thousands of rows; re-reading per call would dominate
        runtime.
        """
        if self._ids is not None:
            return
        if not self._index_shp.exists():
            raise FileNotFoundError(
                f"AHN sub-tile index not found: {self._index_shp}. "
                "Expected the shipped resource at resources/AHN_subunits_GeoTiles/."
            )
        stamp = self._index_stamp()
        cached = self._read_cache(stamp)
        ids, geoms = cached if cached is not None else self._build_cache(stamp)
        self._ids = ids
        self._geoms = geoms
        self._row_by_id = {tid: i for i, tid in enumerate(ids)}

    def _read_cache(self, stamp: tuple[int, int]) -> tuple[list[str], np.ndarray] | None:
        """Read the slim sidecar if present and current; else ``None`` to rebuild."""
        cache = self._cache_path()
        if not cache.exists():
            return None
        try:
            with np.load(cache, allow_pickle=False) as data:
                if tuple(int(v) for v in data["stamp"]) != stamp:
                    return None
                ids = [str(t) for t in data["ids"].tolist()]
                wkb = data["wkb"]
                off = data["wkb_off"]
            geoms = shapely.from_wkb([wkb[off[i] : off[i + 1]].tobytes() for i in range(len(ids))])
            return ids, np.asarray(geoms, dtype=object)
        except Exception as exc:  # noqa: BLE001 - any cache problem falls back to a rebuild
            logging.debug("Ignoring unreadable index cache %s (%s); rebuilding", cache, exc)
            return None

    def _build_cache(self, stamp: tuple[int, int]) -> tuple[list[str], np.ndarray]:
        """Read the full shapefile once, distil it to the slim sidecar, and return it."""
        logging.info(
            "Building one-time AHN sub-tile index cache from %s (later runs reuse it)",
            self._index_shp.name,
        )
        gdf = gpd.read_file(self._index_shp, columns=["GT_AHNSUB"]).to_crs("EPSG:28992")
        ids = [str(t) for t in gdf["GT_AHNSUB"].tolist()]
        geoms = np.asarray(gdf.geometry.values, dtype=object)
        try:
            wkb_list = shapely.to_wkb(geoms)
            off = np.zeros(len(wkb_list) + 1, dtype=np.int64)
            for i, b in enumerate(wkb_list):
                off[i + 1] = off[i] + len(b)
            np.savez(
                self._cache_path(),
                stamp=np.asarray(stamp, dtype=np.int64),
                ids=np.asarray(ids, dtype="U16"),
                wkb=np.frombuffer(b"".join(wkb_list), dtype=np.uint8),
                wkb_off=off,
            )
        except OSError as exc:
            logging.debug("Could not persist AHN index cache (%s); continuing without it", exc)
        return ids, geoms

    def _strtree(self) -> shapely.STRtree:
        self._load_index()
        if self._tree is None:
            self._tree = shapely.STRtree(self._geoms)
        return self._tree

    def tiles_for_aoi(self, aoi_geom: BaseGeometry) -> list[str]:
        self._load_index()
        rows = self._strtree().query(aoi_geom, predicate="intersects")
        return [self._ids[r] for r in sorted(rows.tolist())]

    def core_cell(self, tile_id: str) -> BaseGeometry:
        # Nominal sub-tile polygon from the GeoTiles index. The downloaded LAZ
        # tiles overlap, but the index subunits partition the plane; ownership
        # uses the polygon's axis-aligned bounds (see TileSource.owns_centroids).
        self._load_index()
        try:
            return self._geoms[self._row_by_id[str(tile_id)]]
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
