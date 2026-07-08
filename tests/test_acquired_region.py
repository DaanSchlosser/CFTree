# Copyright (C) 2025 Noah Alting
# Licensed under the GNU General Public License v3.0
# See the LICENSE file for more details.

"""Tests for the acquisition-region sidecar guarding streamed raw.laz reuse.

A range-read raw.laz holds only the points of the region it was read with, so
its reuse gate must compare regions, not existence: a grown region (a larger
--buffer / --halo-margin on a rerun) must re-acquire, while an identical or
shrunk region may reuse. A missing or unreadable sidecar must never be
trusted.
"""

from __future__ import annotations

import sys
from pathlib import Path

from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.get_data.acquired_region import acquired_region_covers, record_acquired_region  # noqa: E402


def test_identical_region_is_covered(tmp_path: Path) -> None:
    raw = tmp_path / "raw.laz"
    region = box(1000.5, 2000.25, 1400.5, 2400.25)
    record_acquired_region(raw, region)
    assert acquired_region_covers(raw, region)


def test_grown_region_forces_reacquisition(tmp_path: Path) -> None:
    raw = tmp_path / "raw.laz"
    record_acquired_region(raw, box(1000, 2000, 1400, 2400))
    # A larger --buffer/--halo-margin grows the clip region by a band the old
    # read never fetched.
    assert not acquired_region_covers(raw, box(992, 1992, 1408, 2408))


def test_shrunk_region_reuses_the_acquisition(tmp_path: Path) -> None:
    raw = tmp_path / "raw.laz"
    record_acquired_region(raw, box(1000, 2000, 1400, 2400))
    # The exact crop happens in the clip sweep, so a smaller region is covered.
    assert acquired_region_covers(raw, box(1100, 2100, 1300, 2300))


def test_missing_or_garbage_sidecar_is_never_trusted(tmp_path: Path) -> None:
    raw = tmp_path / "raw.laz"
    region = box(0, 0, 1, 1)
    assert not acquired_region_covers(raw, region)  # no sidecar (whole tile or legacy)
    raw.with_name(raw.name + ".region").write_text("not wkt at all")
    assert not acquired_region_covers(raw, region)
