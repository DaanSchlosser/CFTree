# Copyright (C) 2025 Noah Alting
# Licensed under the GNU General Public License v3.0
# See the LICENSE file for more details.

"""Regression tests for pipeline failure propagation.

A failed stage must surface as a non-zero process exit so a downstream
consumer (the CityGML creator's on-demand runner) never caches a partial
run as complete. The bug these lock down: ``run_stage`` used to log a
warning and swallow a non-zero stage exit, and each stage's ``main()``
returned 0 even when its own tiles failed.

These exercise the two light, pure seams that carry the contract; they do
not need the heavy CGAL / PDAL reconstruction dependencies.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main as cftree_main  # noqa: E402
from src.stages import (  # noqa: E402
    FAILURE_STATUSES,
    FOREST_ENRICH_FAILED,
    FOREST_ENRICH_SKIPPED,
    FOREST_ENRICH_WRITTEN,
    StageError,
    failed_statuses,
    forest_enrichment_failures,
    missing_tiles_exit_code,
    should_reconstruct,
)


def test_failed_statuses_empty_for_success_and_graceful_skips() -> None:
    # Successes and graceful skips ("not_in_coverage" = AHN6 coverage gap,
    # "empty_tile" = a tile with no trees) are not failures.
    clean = ["ok", "downloaded", "complete", "chunk_done", "exists", "not_in_coverage", "empty_tile"]
    assert failed_statuses(clean) == []
    assert failed_statuses([]) == []


def test_failed_statuses_reports_genuine_failures() -> None:
    statuses = ["ok", "not_in_coverage", "seg_failed", "download_failed", "empty_tile"]
    assert failed_statuses(statuses) == ["seg_failed", "download_failed"]


def test_reconstruction_failure_statuses_are_classified() -> None:
    # The reconstruction stage's own failure set must be a subset of the
    # central classification, so a failed reconstruction exits non-zero.
    assert {"failed", "stalled", "failed_max_attempts"} <= FAILURE_STATUSES


def test_run_stage_raises_on_nonzero_exit() -> None:
    # A stage that exits non-zero must abort the pipeline, not warn and continue.
    with pytest.raises(StageError):
        cftree_main.run_stage("diagstage", "exit 7")


def test_run_stage_returns_on_clean_exit() -> None:
    # A clean stage must not raise.
    assert cftree_main.run_stage("diagstage", "exit 0") is None


# ---------------------------------------------------------------------
# forest.laz enrichment: a genuine failure must not look like a skip
# ---------------------------------------------------------------------
def test_forest_enrichment_failures_flags_only_genuine_failures() -> None:
    # A written tile and a legitimate skip are not failures; an I/O failure is.
    # This is the seam the bug crossed: a swallowed write error used to be
    # indistinguishable from "nothing to write" (both returned 0 points).
    outcomes = [
        ("t1", FOREST_ENRICH_WRITTEN),
        ("t2", FOREST_ENRICH_SKIPPED),
        ("t3", FOREST_ENRICH_FAILED),
    ]
    assert forest_enrichment_failures(outcomes) == ["t3"]


def test_forest_enrichment_failures_empty_for_writes_and_skips() -> None:
    outcomes = [("t1", FOREST_ENRICH_WRITTEN), ("t2", FOREST_ENRICH_SKIPPED)]
    assert forest_enrichment_failures(outcomes) == []
    assert forest_enrichment_failures([]) == []


# ---------------------------------------------------------------------
# --dry-run must not be aborted by an absent tiles directory
# ---------------------------------------------------------------------
def test_missing_tiles_is_fatal_only_for_a_real_run() -> None:
    # A real run needs tiles (exit 1); a dry-run has nothing to list (exit 0),
    # so `main.py --dry-run` on a not-yet-downloaded case does not abort.
    assert missing_tiles_exit_code(dry_run=False) == 1
    assert missing_tiles_exit_code(dry_run=True) == 0


# ---------------------------------------------------------------------
# geometry-only output must not be reused when full metrics are requested
# ---------------------------------------------------------------------
def test_should_reconstruct_rebuilds_geometry_only_for_full_metrics() -> None:
    # The bug: a geometry-only CityJSON (r50/porosity null) was reused as if
    # complete on a later full run. Only this combination must force a rebuild.
    assert should_reconstruct(
        output_exists=True, overwrite=False, existing_is_geometry_only=True, requested_geometry_only=False
    )


def test_should_reconstruct_reuses_when_safe() -> None:
    # A full output satisfies any request; a geometry-only output satisfies a
    # geometry-only request (identical geometry).
    assert not should_reconstruct(
        output_exists=True, overwrite=False, existing_is_geometry_only=False, requested_geometry_only=False
    )
    assert not should_reconstruct(
        output_exists=True, overwrite=False, existing_is_geometry_only=False, requested_geometry_only=True
    )
    assert not should_reconstruct(
        output_exists=True, overwrite=False, existing_is_geometry_only=True, requested_geometry_only=True
    )


def test_should_reconstruct_runs_when_absent_or_overwrite() -> None:
    assert should_reconstruct(
        output_exists=False, overwrite=False, existing_is_geometry_only=False, requested_geometry_only=False
    )
    assert should_reconstruct(
        output_exists=True, overwrite=True, existing_is_geometry_only=False, requested_geometry_only=False
    )
