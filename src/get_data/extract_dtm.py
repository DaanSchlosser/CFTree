# Copyright (C) 2025 Noah Alting
# Licensed under the GNU General Public License v3.0
# See the LICENSE file for more details.

# src/get_data/extract_dtm.py


import json
import logging
from pathlib import Path

import pdal

from src.config import get_config
from src.stages import DtmResult, MissingPrerequisiteError, StageFailureError
from src.tile_layout import TileLayout


def compute_tile_dtm(
    clipped_las: Path,
    dtm_out: Path,
    resolution: float = 0.5,
    rigidness: int = 3,
    iterations: int = 500,
    ground_only: bool = True,
    overwrite: bool = False,
) -> DtmResult:
    """Compute DTM from a clipped .laz file using PDAL CSF + GDAL writer.

    Raises
    ------
    MissingPrerequisiteError
        `clipped_las` is not on disk.
    StageFailureError
        PDAL pipeline failed at runtime.
    """
    if not clipped_las.exists():
        raise MissingPrerequisiteError(f"Input clipped LAS not found: {clipped_las}")

    if dtm_out.exists() and not overwrite:
        logging.debug(f"DTM already exists at {dtm_out} — skipped.")
        return DtmResult(dtm=dtm_out, did_work=False)

    cfg = get_config()
    crs = cfg["crs"]

    pipeline_def: list = [
        str(clipped_las),
        {
            "type": "filters.csf",
            "resolution": resolution,
            "rigidness": rigidness,
            "iterations": iterations,
        },
    ]

    if ground_only:
        pipeline_def.append({"type": "filters.range", "limits": "Classification[2:2]"})

    pipeline_def.append(
        {
            "type": "writers.gdal",
            "filename": str(dtm_out),
            "resolution": resolution,
            "output_type": "min",
            "nodata": -9999,
            "override_srs": crs,
        }
    )

    logging.debug(f"Running PDAL DTM pipeline on {clipped_las}")
    pipeline = pdal.Pipeline(json.dumps(pipeline_def))
    try:
        pipeline.execute()
    except RuntimeError as e:
        raise StageFailureError(f"PDAL pipeline failed for {clipped_las}: {e}") from e

    logging.info(f"DTM written to {dtm_out}")
    return DtmResult(dtm=dtm_out, did_work=True)


# For manual test:
if __name__ == "__main__":
    tile = TileLayout(Path("data/wippolder/tiles/37EN2_11"))
    compute_tile_dtm(tile.clipped_laz, tile.dtm)
