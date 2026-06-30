# CFTree

**Author:** Noah Alting (MSc Geomatics, TU Delft, 2025)  
**License:** [GPL-3.0](./LICENSE)

This repository contains the code accompanying the MSc thesis  
**“From Point Clouds to Porous Crowns: A Scalable Approach for CFD-Ready Urban Tree Reconstruction”**,  
submitted at the Delft University of Technology, Faculty of Architecture and the Built Environment.  

The project presents a fully automated, scalable pipeline that reconstructs CFD-ready 3D tree geometries at the city scale, directly from open-access airborne lidar data ([AHN](https://www.ahn.nl/)).


## Overview

Urban climate simulations increasingly rely on digital twins of cities, yet vegetation remains largely absent or oversimplified despite its strong influence on wind flow and heat exchange.  
This project introduces a complete end-to-end workflow that reconstructs detailed, watertight, and physically meaningful tree models from raw airborne laser scanning (ALS) point clouds.  

<p align="center">
  <img src="docs/img/input_street.png" alt="Input point cloud data" width="48%" style="margin-right:2%;">
  <img src="docs/img/output_street.png" alt="Reconstructed CFD-ready trees" width="48%">
</p>


## Pipeline Overview

The pipeline operates directly on unstructured lidar data and consists of three major components:

1. **Data acquisition** — Automatic download, clipping, and preprocessing of AHN tiles for a user-defined area.  
2. **Segmentation** — Novel *High-Order Multi-Echo Density (HOMED)* vegetation filtering and per-tree clustering via a modified [*TreeSeparation algorithm*](https://github.com/Jinhu-Wang/TreeSeparation) algorithm.  
3. **Reconstruction** — Generation of CFD-ready watertight geometries (crown and trunk) using CGAL-based α-wrapping and per-tree metric extraction.

The workflow has been applied to several major Dutch cities, including Amsterdam, Rotterdam, Utrecht, and Delft, reconstructing over 380,000 trees within practical runtimes (~13 hours for Amsterdam on 16 CPU cores).


## Repository Structure
``` bash
CFTree/
├── cases/      # Case definitions and logs per study area
├── data/       # Intermediate and final outputs per case
├── resources/  # Static reference datasets (e.g., AHN tile index)
├── scripts/    # Orchestration scripts for each pipeline step
└── src/        # Core modules (data, segmentation, reconstruction)
```

## Pipeline Workflow

### 1. Data Acquisition (`get_data`)
- Input: `cases/<case_name>/case_area.geojson`
- Downloads and clips AHN LiDAR tiles and digital terrain models (DTM) for the specified area.
- Defaults to **AHN6** (1x1 km Cloud-Optimized Point Cloud, served by `basisdata.nl`); pass
  `--ahn-version 4` or `--ahn-version 5` to use the TU Delft GeoTiles host instead.
- Note: the first AHN6 release covers only the northeast of the Netherlands. AHN6 outputs
  must be attributed (CC BY 4.0); AHN4 and AHN5 are CC0.
- Outputs stored under `data/<case_name>/tiles/<tile_id>/`.

### 2. Tree Segmentation (`segmentation`)
- Applies the **HOMED** vegetation filter to isolate vegetation points.
- Segments individual trees using a modified [**TreeSeparation algorithm**](https://github.com/Jinhu-Wang/TreeSeparation) (C++).
- Produces per-tree point clusters and a harmonized forest-level point cloud (`forest.laz`) with global tree identifiers (`gtid`).

### 3. Geometry Reconstruction (`reconstruction`)
- For each segmented tree:
  - Generates local coordinate systems for efficiency.
  - Reconstructs watertight meshes using the [**CGAL 3D Alpha Wrapping algorithm**](https://doc.cgal.org/latest/Alpha_wrap_3/index.html#Chapter_3D_Alpha_wrapping).
  - Derives morphological attributes (e.g., crown width, height, trunk dimensions).
  - Constructs LOD3-level crown and trunk geometries.
- Exports per-tile CityJSON files with full geometric and attribute information.

For detailed execution instructions, see `/scripts/README.md`.

## Installation

This repository uses a Conda environment for reproducibility.

```bash
conda env create -f environment.yml
conda activate cftree
```

> **NOTE:** Ensure sufficient storage capacity (≥100 GB recommended for full-city processing). The data path can be configured in `src/config.py`.

> **NOTE:** Creating the environment may take a while, don't worry!

## Run with Docker

The Docker image bakes the conda environment and the two compiled C++ binaries,
so a colleague runs CFTree without a manual WSL setup, a conda environment, or a
C++ toolchain. The only prerequisites are Docker and a clone of this repository.

Build the image once:

```bash
docker build -t cftree:local .
```

Then run a case by bind-mounting your checkout at `/work`, which keeps `cases/`
and `data/` on the host so the outputs land next to the source as usual:

```bash
docker run --rm -v "$PWD":/work cftree:local \
    python main.py --case wippolder --ahn-version 6 --n-cores 8 --buffer 20 --overwrite
```

The compiled binaries are baked at `/opt/cftree/bin` inside the image and are
found through the `CFTREE_BIN` environment variable, so the bind-mounted checkout
resolves them even though a clone carries no `build/` outputs. On Windows and
macOS the build context excludes `data/`, so a large local `data/` directory does
not slow the build.

The CityGML/Energy ADE creator drives this image directly through its docker
runner. Set `CFTREE_RUNNER=docker` and `CFTREE_IMAGE=cftree:local` in that
project's `.env`; the creator then bind-mounts the checkout and runs the image
for each area of interest.

The image embeds CGAL alpha-wrap (GPL-3.0) and the TreeSeparation binary, so the
image is a GPL-3.0 distribution, consistent with this repository's licence.

### Verifying the image

A smoke test ships with the image and confirms the build is healthy without a
network download or a sample dataset. It checks that the baked environment
imports the geospatial stack and that both compiled binaries run on tiny
synthetic point clouds, which catches a binary that compiled but cannot load its
shared libraries at runtime:

```bash
docker run --rm cftree:local python /opt/cftree/docker/smoke_test.py
```

A GitHub Actions workflow (`.github/workflows/docker-smoke.yml`) builds the image
and runs this same smoke test on every change to the Dockerfile, the environment,
or the source, so a broken build is caught in CI rather than on a colleague's
first run.

## Performance: data acquisition (get_data)

For a small area the data acquisition stage, not reconstruction, is the longest
part of the pipeline. On a Leiden 250 m area it took about 117 s, split roughly
evenly between resolving which tiles intersect the area (about 53 s) and
downloading them (about 51 s). The download is bandwidth-bound, so adding
connections does not help (a single connection already saturates the link); the
only levers are reading fewer bytes and not reading the same bytes twice. Three
changes address that.

**Cached tile index.** The shipped GeoTiles sub-tile index is a national
shapefile of about 76 000 polygons (a roughly 49 MB `.shp` and `.dbf` pair).
Reading and reprojecting all of it to find a handful of intersecting tiles cost
30 to 60 s on a virtualized mount, every run. The first run now distils the index
to a slim `<index>.bounds.npz` sidecar (sub-tile ids and geometries only), keyed
by the shapefile's size and mtime, and later runs load it in under a second and
rebuild it automatically if the shapefile changes. Tile selection is identical to
the full read. This sidecar is git-ignored and also speeds the segmentation
stage, which resolves the same index.

**Shared tile cache.** A downloaded AHN tile is part of an immutable national
dataset, so re-downloading it for a second area that overlaps the first is pure
waste. Raw tiles now land in a cross-area cache (under `<data_root>/.ahn_cache`,
or `CFTREE_AHN_CACHE` if set) and are hardlinked into each case, so an area that
reuses a tile pays no network cost. The cache is independent of `--overwrite`,
which still re-derives the per-area clip and downstream artifacts but never
re-fetches an immutable tile.

**COPC range reads for AHN6.** AHN6 ships Cloud-Optimized Point Clouds, which a
reader can range-read over HTTP. Instead of downloading a whole 1 km cell (tens
of millions of points) to clip a small area out of it, get_data now reads only
each tile's area region directly from the remote file with PDAL. On an Emmen AHN6
cell a 1.2 %-of-cell region read took about 2.5 s against about 74 s for the whole
cell, and the read returns the same point set the whole-download-then-clip path
produced.

**Partial range reads for AHN4/AHN5.** AHN4 and AHN5 from GeoTiles are plain LAZ
with no COPC, but the host still honours HTTP range requests and each tile ships a
small `.lax` spatial index. get_data reads that index, turns the area into the
point-index runs that cover it, and pulls only the chunks those runs touch, with a
read-ahead window that keeps a run to a few large requests rather than one per
chunk. How much this saves is set by the file, not the reader. A plain LAZ is not
spatially ordered the way a COPC is, so a small area's points are scattered across
a sizeable fraction of the chunks, about a quarter of a tile for a 250 m area and
about half for a 400 m area. The read is therefore worth it for small areas and
not for large ones, so get_data estimates the fraction from the index first and
downloads the whole tile (into the shared cache, reused across areas) when a
partial read would not pay, and falls back to the whole tile on any index or read
fault. The exact crop still happens in the clip step, so the points are identical
to a whole-tile clip.

**Overlapping fetch and clip.** Overlapping GeoTiles tiles carry each tile's halo
within that tile's own cloud, so a tile's clip needs no neighbour. get_data runs a
fused acquire, clip, and DTM per tile for such sources, with no barrier between
fetching and clipping, so one tile's clip and DTM (on the processor) overlap
another tile's range read (on the network). A hard-partitioned source (AHN6) fills
its halo from neighbouring cells and so keeps the two-sweep barrier.

Together, on a cache hit the Leiden 250 m get_data stage drops from about 117 s to
about 14 s with byte-identical clipped output. Forced fully cold, with no cached
tiles, the partial reads bring it to about 25 s for the 250 m area and about 40 s
for the 400 m area, against roughly 100 s for a whole-tile download.

## Performance: scratch cache and parallel reconstruction

Two changes dominate reconstruction wall time, and both apply to every run
including `--geometry-only` (where the descriptive metrics below are skipped).

The per-tree cache is fsync-heavy (an in-flight marker plus an atomic pickle per
tree) and ephemeral (it is deleted once a tile finishes). Writing it next to the
data put that churn on whatever filesystem holds the data root, which for the two
common runners is a slow virtualized mount, the WSL `/mnt/c` 9p share and the
Docker bind-mount of a Windows path. The cache now goes to a fast local directory
instead, `CFTREE_SCRATCH` if set and the system temp directory otherwise (ext4 or
tmpfs under WSL, the container's own overlay under Docker). On a Leiden 400 m area
this alone took reconstruction from about 96 s to about 14 s with byte-identical
output, and the container gets it for free because its temp directory is not the
bind-mount. Only the final CityJSON is written to the data directory.

Reconstruction also runs all tiles' trees through one global queue of work
batches, so up to `--n-cores` worker subprocesses stay busy regardless of how the
trees are distributed. A single tile with thousands of trees is split into many
batches that run in parallel rather than one sequential stream, and a handful of
small tiles no longer leave most cores idle. On a 1000 m area whose largest tile
holds about 2400 trees, reconstruction went from about 117 s to about 37 s at
`--n-cores 8` and about 29 s at `--n-cores 16`, again byte-identical. Each worker
holds up to `--chunk-size` trees in memory, so size `--n-cores` to RAM; a
geometry-only worker is lighter than a full-metric one.

## Performance: GPU morphometrics

This lever helps full-semantic runs only. It has no effect under
`--geometry-only`, which skips the two metrics described here.

The two descriptive metrics computed per tree, `r50` and `porosity`, account for
the large majority of reconstruction time. `r50` is a voxelization plus a
nearest-neighbour query, and `porosity` is an inside/outside test of a voxel grid
against the crown mesh. The expensive shared step is the watertight inside/outside
test, and that is what moves to the GPU through an NVIDIA Warp winding-number
query (valid because the alpha-wrapped crown is watertight). The r50
nearest-neighbour query stays on scipy's k-d tree, which is faster than a GPU
brute force at these small per-tree point counts, so Warp is the only GPU
dependency.

The GPU path is opt-in and off by default. Enable it with `CFTREE_GPU_METRICS=1`
on a machine with a CUDA-capable NVIDIA GPU; without one, or if any GPU step
fails, the run falls back to the existing CPU path with no change in output.

```bash
# inside the cftree env or the container, with a GPU present
CFTREE_GPU_METRICS=1 python main.py --case wippolder --ahn-version 6 --n-cores 8 --buffer 20 --overwrite
```

In the container, pass the GPU through and set the flag:

```bash
docker run --rm --gpus all -e CFTREE_GPU_METRICS=1 -v "$PWD":/work cftree:local \
    python main.py --case wippolder --ahn-version 6 --n-cores 8 --buffer 20 --overwrite
```

`--gpus all` needs an NVIDIA driver on the host and the nvidia-container-toolkit,
which Docker Desktop provides through its WSL2 backend.

Before enabling the GPU path on real data, validate it against the CPU baseline.
The benchmark harness times the two metrics and diffs the GPU result against the
CPU result per tree, failing if either drifts beyond tolerance:

```bash
# synthetic crowns, no pipeline data needed
python -m scripts.bench_morphometrics --synthetic --n-trees 50
```

A `PASS` line confirms the GPU output matches the CPU output within tolerance and
reports the measured speedup. The GPU path accelerates the descriptive metrics
only; the crown and trunk geometry is produced by the same CGAL alpha-wrap and is
unchanged.

On a laptop RTX 4070 the morphometrics ran about twice as fast end to end and
roughly three times as fast per tree once the one-time Warp kernel compile is
amortized, with the metric values matching the CPU baseline. The reconstruction
workers share one GPU, so on a small card a very high `--n-cores` can crowd GPU
memory; a moderate value is enough because the GPU already parallelizes the
per-tree work.

## Quickstart Example

### 1. Define your area of interest in:
```bash 
cases/<case_name>/case_area.geojson
```
### 2. Set case variables in `src/config.py`, defaults for example run are:
``` python
# ---------------------------------------------------------------------
# Default case configurations
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

### 3. Build the cpp files.
Detailed instructions see `src/segmentation/TreeSeparation/README.md` and `src/reconstruction/AlphaWrap/README.md`.

### 4. Run the pipeline steps individually:
``` bash
python -m scripts.get_data
python -m scripts.segmentation
python -m scripts.tree_reconstruction
```
If desired, each stage can have the settings defined in `src/config.py` be overwritten:
``` bash
--case <path>                       # path to case to run
--n-cores <number_of_cores>         # number of available cores for paralellisation
--log-level <[INFO, WARNING, DEBUG] # log detail level
--overwrite                         # overwrite existing files
--dry-run                           # only list tiles
```
Logs for each run are stored under: `cases/<case_name>/logs/`


### 5. (New) Run the entire pipeline with a single command

A new `main.py` script orchestrates all three stages — data acquisition, segmentation, and reconstruction — in sequence.  
It automatically loads configuration values from `src/config.py` and logs high-level progress to `cases/<case_name>/logs/main.log`.

```bash
python main.py --case wippolder --overwrite --n-cores 16
```

Optional arguments (same as for individual stages):

``` bash
--case <case_name>                 # case to process (default from config)
--n-cores <number>                 # number of CPU cores (default from config)
--overwrite                        # re-run even if outputs exist
--log-level <INFO|DEBUG|WARNING>   # logging verbosity
--dry-run                          # only list tiles to process
--buffer <distance>                # buffer distance around AOI (default 20 m)
--max-trees <number>               # limit number of trees per tile (for testing)
```

Logs summary and timing for each stage are written to `cases/<case_name>/logs/main.log`.

## Outputs
Each fully processed tile will contain:
``` bash
clipped.laz
clipped_dtm.tif
vegetation.laz
vegetation.xyz
segmentation.xyz
forest.laz
tree_hulls.geojson
trees_lod3.city.json
```
Case-level aggregated outputs:
``` bash
data/<case_name>/forest_hulls.geojson
data/<case_name>/gtid_map.csv
```

## Performance
| City      | # Trees | Runtime (16 cores) | Notes                       |
| --------- | ------- | ------------------ | --------------------------- |
| Amsterdam | ~380k   | ~13 h              | Full pipeline, AHN5 dataset |
| Rotterdam | ~210k   | ~7 h               |                             |
| Utrecht   | ~150k   | ~5 h               |                             |
| Delft     | ~90k    | ~3 h               |                             |


## Acknowledgements and Contact
This repository is part of my MSc thesis:
“From Point Clouds to Porous Crowns: A Scalable Approach for CFD-Ready Urban Tree Reconstruction.”
at the Delft University of Technology, Faculty of Architecture and the Built Environment.  
This thesis is available at the [TU Delft Repository](https://resolver.tudelft.nl/uuid:adc9c299-b8f7-4854-b004-dcf2393d06fb).

Supervised by: Dr. Hugo Ledoux and Dr. Clara García Sánchez

For questions or collaborations, feel free to contact me via:  
- Email: noahalting (at) gmail.com  
- [LinkedIn](https://www.linkedin.com/in/noah-alting-6b041916b)


## How to Cite

If you use this code or parts of it in your research, please cite the corresponding MSc thesis:

**Alting, N.** (2025). *From Point Clouds to Porous Crowns: A Scalable Approach for CFD-Ready Urban Tree Reconstruction.*  
MSc Thesis, Delft University of Technology, Faculty of Architecture and the Built Environment.  
To reference this document use: [https://resolver.tudelft.nl/uuid:adc9c299-b8f7-4854-b004-dcf2393d06fb](https://repository.tudelft.nl/)


## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0).

It includes and modifies components from:
- [TreeSeparation](https://github.com/Jinhu-Wang/TreeSeparation) (originally LGPL-3.0)
- [CGAL Alpha Wrap 3 example](https://doc.cgal.org/latest/Alpha_wrap_3/) (GPL-3.0)

Therefore, the entire project is distributed under the GNU General Public License v3.0.
See the [LICENSE](./LICENSE) file for details.
