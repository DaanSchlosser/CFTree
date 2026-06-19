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
from src.stages import FAILURE_STATUSES, StageError, failed_statuses  # noqa: E402


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
