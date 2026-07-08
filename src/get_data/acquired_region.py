# Copyright (C) 2025 Noah Alting
# Licensed under the GNU General Public License v3.0
# See the LICENSE file for more details.

# src/get_data/acquired_region.py

"""Sidecar recording which region a range-read `raw.laz` was acquired for.

A streamed `raw.laz` (COPC region read, `.lax` partial read) holds only the
points of the region it was read with, unlike a whole-tile download. Reusing it
by bare existence is therefore unsafe: a rerun with a larger ``--buffer`` or
``--halo-margin`` rewrites the clip region and re-clips, but against a cloud
that never held the new band, silently shrinking the halo. The acquirers record
the read region in this sidecar, and their skip gates reuse the file only when
the recorded region covers the requested one (equal on a plain rerun; a shrunk
region is also covered, since the exact crop happens in the clip sweep).

A missing sidecar is never trusted: a whole-tile `raw.laz` re-provisions from
the shared cache at the cost of a hardlink, and a pre-sidecar streamed file is
re-acquired once.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

import shapely
from shapely.geometry.base import BaseGeometry


def _sidecar_path(raw_laz: Path) -> Path:
    return raw_laz.with_name(raw_laz.name + ".region")


def record_acquired_region(raw_laz: Path, region: BaseGeometry) -> None:
    """Record the region *raw_laz* was acquired for.

    A write failure is non-fatal: a missing sidecar only forces a (safe)
    re-acquisition on the next run.
    """
    with contextlib.suppress(OSError):
        _sidecar_path(raw_laz).write_text(shapely.to_wkt(region, rounding_precision=-1))


def acquired_region_covers(raw_laz: Path, region: BaseGeometry) -> bool:
    """Whether the region recorded for *raw_laz* covers the requested *region*.

    False when the sidecar is missing or unreadable, so an unknown acquisition
    is conservatively re-done rather than trusted.
    """
    try:
        recorded = shapely.from_wkt(_sidecar_path(raw_laz).read_text())
    except (OSError, shapely.errors.ShapelyError) as e:
        logging.debug(f"No usable acquisition-region sidecar for {raw_laz} ({e})")
        return False
    return bool(recorded.covers(region))
