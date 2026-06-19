# Copyright (C) 2025 Noah Alting
# Licensed under the GNU General Public License v3.0
# See the LICENSE file for more details.

# src/tile_layout.py

"""Filesystem layout for case and tile artifacts.

Each pipeline stage reads and writes a small set of conventional files
under `data/<case>/...`. This module is the single source of truth for
those names; stages call `TileLayout` / `CaseLayout` accessors instead
of stitching path literals.

Mirrors the shape of `tile_sources.py`: one place that knows the
conventions, N callers that don't.
"""

from __future__ import annotations

from pathlib import Path

from src.config import ResolvedConfig


class TileCacheLayout:
    """Per-tile reconstruction cache (`<tile_dir>/_cache/`).

    Holds the chunk-recycling state used by `scripts/reconstruction.py`:
    in-flight markers, the skipped-gtid list, per-tree intermediate
    point clouds and meshes, and finalized pickled tree results.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.trees_dir = root / "trees"
        self.in_flight = root / "in_flight.txt"
        self.skipped = root / "skipped.txt"

    def tree_xyz(self, gtid: int) -> Path:
        return self.root / f"tree_{gtid}.xyz"

    def tree_ply(self, gtid: int) -> Path:
        return self.root / f"tree_{gtid}.ply"

    def tree_pkl(self, gtid: int) -> Path:
        return self.trees_dir / f"{gtid}.pkl"


class TileLayout:
    """Filesystem artifacts for a single tile under `<case_data>/tiles/<tile_id>/`."""

    def __init__(self, tile_dir: Path) -> None:
        self.dir = tile_dir
        self.tile_id = tile_dir.name

    # --- data acquisition ---
    @property
    def raw_laz(self) -> Path:
        return self.dir / "raw.laz"

    @property
    def raw_lax(self) -> Path:
        return self.dir / "raw.lax"

    @property
    def clipped_laz(self) -> Path:
        return self.dir / "clipped.laz"

    @property
    def dtm(self) -> Path:
        return self.dir / "clipped_dtm.tif"

    @property
    def clip_region(self) -> Path:
        """Polygon this tile's cloud was clipped to: core cell + halo margin,
        intersected with the buffered AOI. Written by get_data, read by the clip."""
        return self.dir / "clip_region.geojson"

    # --- vegetation + segmentation ---
    @property
    def vegetation_laz(self) -> Path:
        return self.dir / "vegetation.laz"

    @property
    def vegetation_xyz(self) -> Path:
        return self.dir / "vegetation.xyz"

    @property
    def segmentation_xyz(self) -> Path:
        return self.dir / "segmentation.xyz"

    @property
    def tree_hulls(self) -> Path:
        return self.dir / "tree_hulls.geojson"

    # --- reconstruction ---
    @property
    def forest_laz(self) -> Path:
        return self.dir / "forest.laz"

    @property
    def cityjson(self) -> Path:
        return self.dir / "trees_lod3.city.json"

    @property
    def geometry_only_marker(self) -> Path:
        """Present iff ``cityjson`` was written in ``--geometry-only`` mode.

        Lets a later full run tell a geometry-only output (r50/porosity null)
        from a complete one and rebuild it, instead of reusing the nulls as if
        finished. Absent means a full run (or a legacy pre-marker output)."""
        return self.dir / "trees_lod3.geometry_only"

    @property
    def cache(self) -> TileCacheLayout:
        return TileCacheLayout(self.dir / "_cache")


class CaseLayout:
    """Filesystem layout for a case.

    User inputs (the AOI) live under `cases/<case>/` (`input_dir`); pipeline
    outputs live under `data/<case>/` (`data_dir`). The tile tree hangs
    off `data_dir/tiles/<tile_id>/`.
    """

    def __init__(self, data_dir: Path, input_dir: Path | None = None) -> None:
        self.data_dir = data_dir
        self._input_dir = input_dir

    @classmethod
    def from_config(cls, cfg: ResolvedConfig) -> CaseLayout:
        return cls(data_dir=cfg["data_case_path"], input_dir=cfg["case_path"])

    @property
    def input_dir(self) -> Path:
        if self._input_dir is None:
            raise ValueError("CaseLayout was constructed without input_dir; AOI accessors unavailable.")
        return self._input_dir

    # --- inputs ---
    @property
    def aoi(self) -> Path:
        return self.input_dir / "case_area.geojson"

    @property
    def buffered_aoi(self) -> Path:
        return self.input_dir / "case_area_buffered.geojson"

    # --- outputs ---
    @property
    def tiles_dir(self) -> Path:
        return self.data_dir / "tiles"

    @property
    def forest_hulls(self) -> Path:
        return self.data_dir / "forest_hulls.geojson"

    @property
    def gtid_map(self) -> Path:
        return self.data_dir / "gtid_map.csv"

    @property
    def tile_source_manifest(self) -> Path:
        """Records the AHN version/source used for this case (written by get_data).

        Read back by the segmentation stage so it can resolve each tile's core
        cell without re-passing --ahn-version; the AHN version is otherwise only
        a CLI flag and is not in the config.
        """
        return self.data_dir / "tile_source.json"

    # --- traversal ---
    def tile(self, tile_id: str) -> TileLayout:
        return TileLayout(self.tiles_dir / tile_id)

    def iter_tiles(self) -> list[TileLayout]:
        if not self.tiles_dir.exists():
            return []
        return [TileLayout(p) for p in sorted(self.tiles_dir.iterdir()) if p.is_dir()]
