# Scripts — CFTree Pipeline Execution

This folder contains all high-level **orchestration scripts** for running the CFD-ready urban tree reconstruction pipeline.  
Each script represents one stage of the workflow and can be executed independently or sequentially.

## Full Pipeline instructions:

### Input Requirements:
Each case must be initialized by case folder holding a polygon of the desired area to process in geojson format. 
For testing, an example neighbourhood of the city of Delft is provided: `/cases/wippolder/case_area.geojson`.

Global, case-specific configurations can be set in `/src/config.py` and is automatically propagated to all modules.
``` python
# ---------------------------------------------------------------------
# Case configurations used throughout the pipeline
# ---------------------------------------------------------------------
DEFAULT_CONFIG = {
    "case_root": Path("cases"),             # user case input directory
    "data_root": Path("data"),              # data storage root (large files)
    "resources_dir": Path("resources"),     # shared resources
    "case": "wippolder",                    # default case
    "default_cores": 2,                     # global default for parallelization
    "crs": "EPSG:28992",                    # Amersfoort / RD New
}
```

Before running it is necessary to build the cpp executables at `src/segmentation/TreeSeparation` and `src/reconstruction/AlphaWrap`.

> **NOTE:** The pipeline supports city-scale processing. Large cases may contain many tiles and will therefore produce large files.
It is recommended to make sure enough disk space is available at `data_root`.

### Execution Order:

1. **Data acquisition** → `get_data.py`  
2. **Tree segmentation** → `segmentation.py`  
3. **Geometry reconstruction** → `reconstruction.py`

Each stage reads and writes to `data/<case>/tiles/<tile_id>/`, ensuring reproducibility and modularity.
Logs for each run are written to:  `cases/<case_name>/logs/<step_name>.log`

### Example Full Pipeline:
``` bash
python -m scripts.get_data --case wippolder --n-cores 4
python -m scripts.segmentation --case wippolder --n-cores 8
python -m scripts.reconstruction --case wippolder --n-cores 16
```
All stages are independent, enabling partial reruns or debugging at any point.
Logs and outputs are structured per case for traceability.

### Shared CLI Parameters

All scripts accept the following common flags to overwrite the default settings:

| Flag | Type | Description |
|------|------|--------------|
| `--case` | `str` | Name of the case folder under `cases/` (default from config). |
| `--n-cores` | `int` | Number of CPU cores to use. Parallel execution via `ProcessPoolExecutor`. |
| `--overwrite` | `flag` | Force regeneration of existing outputs. |
| `--log-level` | `str` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`). |
| `--dry-run` | `flag` | List tasks without executing them. |



## 1. Data Acquisition — `get_data.py`

Downloads and prepares AHN tiles for the selected case.
Performs download, clipping, and DTM generation.

### Main steps:
1. Buffer the case polygon (`case_area.geojson`).
2. Resolve the AHN tile catalog (AHN6 by default; opt in to AHN4 or AHN5). For
   AHN4/AHN5 the national GeoTiles index is read once and cached to a slim
   `<index>.bounds.npz` sidecar, so later runs resolve tiles in under a second.
3. Acquire each tile's points. AHN6 Cloud-Optimized Point Clouds are range-read
   over HTTP for the AOI region only (PDAL `readers.copc`). AHN4/AHN5 sub-tiles
   from the [TU Delft GeoTiles server](https://geotiles.citg.tudelft.nl/) are plain
   LAZ; they are range-read for the AOI region through their `.lax` index when that
   pulls materially less than the whole tile, and otherwise downloaded whole into a
   shared cache (see below) and reused across areas.
4. Clip tiles to AOI and compute `clipped_dtm.tif`. Overlapping sources (AHN4/AHN5)
   clip each tile from its own cloud and fuse this step with acquisition per tile;
   a hard-partitioned source (AHN6) fills each tile's halo from neighbouring cells,
   so it acquires all tiles before clipping.

### Caches

- **Tile index** (`resources/AHN_subunits_GeoTiles/AHN_subunits_GeoTiles.bounds.npz`)
  is rebuilt on demand from the shipped shapefile and git-ignored.
- **Shared tiles** for AHN4/AHN5 land in `<data_root>/.ahn_cache/<source>/`
  (override with `CFTREE_AHN_CACHE`). A downloaded tile is part of an immutable
  national dataset, so it is fetched once and hardlinked into each case; the cache
  is not invalidated by `--overwrite`. The small `.lax` index used by the partial
  range reads is cached here too. AHN6 and AHN4/AHN5 partial range reads pull only
  the AOI region per area and are not cached as whole tiles.

### Tile sources

| Version | Grid           | Endpoint                                                              | License     |
|---------|----------------|-----------------------------------------------------------------------|-------------|
| AHN6    | 1x1 km KM      | `basisdata.nl/hwh-ahn/AHN6/01_LAZ/AHN6_2025_C_<minX>_<minY>.COPC.LAZ` | CC BY 4.0   |
| AHN5    | 1x1.25 km TOP  | `geotiles.citg.tudelft.nl/AHN5_T/<tile_id>.LAZ` (+ `.LAX`)            | CC0 1.0     |
| AHN4    | 1x1.25 km TOP  | `geotiles.citg.tudelft.nl/AHN4_T/<tile_id>.LAZ` (+ `.LAX`)            | CC0 1.0     |

The AHN6 first release covers only the northeast of the Netherlands. Outside that
footprint each tile logs `not_found_remote`; fall back to AHN4 or AHN5 for those AOIs.

### Outputs:
per tile:
``` bash
raw.laz                 # raw points: the AOI region (range read) or the whole tile (download)
raw.lax                 # spatial index, only when a whole AHN4/AHN5 tile is downloaded
clipped.laz             # point cloud clipped to case polygon
clipped_dtm.tif         # DTM raster of clipped tile
```


### Optional flags:
| Flag | Type | Description |
|------|------|-------------|
| `--buffer` | `float` | Buffer distance in meters around AOI (default 20m). |
| `--ahn-version` | `int` | AHN release: `6` (default), `5`, or `4`. |

### Example:
```bash
python -m scripts.get_data --case emmer_compascuum --n-cores 4 --buffer 10
python -m scripts.get_data --case wippolder --n-cores 4 --ahn-version 5   # AHN6 not yet available here
```


## 2. Tree Segmentation — `segmentation.py`
Applies vegetation filtering and tree segmentation, producing per-tree point clusters and harmonized IDs.

### Main steps:

1. Vegetation filtering using HOMED algorithm.
2. Tree segmentation via modified TreeSeparation (C++)
3. Forest ID generalization across all tiles.

### Outputs:
per tile:
``` bash
vegetation.laz          # filtered vegetation point cloud in LAS format
vegetation.xyz          # filtered vegetation point cloud in XYZ format used for segmentation
segmentation.xyz        # segmented tree clusters 
forest.laz              # segmented tree clusters with unified gtid attribute
```

per case:
``` bash
forest_hulls.geojson    # 2D projected convex hulls of tree clusters
gtid_map.csv            # case index registry
```

### Example:
``` bash
python -m scripts.tree_segmentation --case wippolder --n-cores 8
```

## 3. Geometry Reconstruction — `reconstruction.py`
Generates watertight 3D tree geometries (crown + trunk) for CFD analysis.

### Main steps:
1. Load `forest.laz` and `clipped_dtm.tif` per tile.
2. Compute morphological metrics per tree.
3. Reconstruct each tree geometry in `LoD3.B`
4. Export tree geometries and attributes to `CityJSON` per tile.

### Outputs:
per tile:
``` bash
trees_lod3.city.json    # final output file, ready for CFD-use
```

A temporary per-tile scratch cache (per-tree point clouds, meshes and pickled
results) is written to a fast local directory, not next to the data: `CFTREE_SCRATCH`
if set, otherwise the system temp directory. It is removed after a tile finishes
(keep it with `--keep-cache`). Keeping this fsync-heavy cache off a slow mount
(the WSL `/mnt/c` path or a Docker bind-mount) is a large reconstruction speedup.

### Optional flags:
| Flag           | Type   | Description                                   |
| -------------- | ------ | --------------------------------------------- |
| `--keep-cache` | `flag` | Keep intermediate per-tree cached files.      |
| `--max-trees`  | `int`  | Limit number of trees per tile (for testing). |

### Example:
```bash
python -m scripts.tree_reconstruction --case wippolder --n-cores 16 --keep-cache
```

