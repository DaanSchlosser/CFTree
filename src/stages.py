# Copyright (C) 2025 Noah Alting
# Licensed under the GNU General Public License v3.0
# See the LICENSE file for more details.

# src/stages.py

"""Stage seam for the CFTree pipeline.

Every long-running pipeline step (`download_tile`, `clip_tile`, `filter_tile`, ...)
shares a single calling convention:

- On success, return a typed `*Result` dataclass carrying the artifact paths and
  a `did_work` flag (False if the stage detected its output already on disk and
  short-circuited).
- On a terminal failure, raise a `StageError` subclass. The orchestrator catches
  these and decides whether to abort the tile, downgrade to INFO (e.g. AHN6
  outside coverage), or keep going.

This replaces the older "stage returns a `dict` with a stringly-typed status"
convention. The seam is the same — one place callers learn the protocol — but
the interface is now typed end-to-end and failure modes are distinguishable.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------
class StageError(Exception):
    """Base for terminal stage failures."""


class MissingPrerequisiteError(StageError):
    """Required input or external binary not on disk.

    Distinct from `StageFailureError` because it indicates an upstream-stage
    gap or install-time misconfiguration, not a runtime failure inside the
    stage.
    """


class StageFailureError(StageError):
    """Stage executed but did not produce its expected output."""


class RemoteUnavailableError(StageError):
    """Resource not present at the remote URL.

    AHN6's first release covers only the northeast of the Netherlands; tile
    probes outside that footprint return 403/404. The orchestrator treats this
    as a graceful skip (INFO log) rather than a network error (WARNING).
    """


# ---------------------------------------------------------------------
# Per-stage result dataclasses
# ---------------------------------------------------------------------
# Each stage returns its own typed result rather than a generic dict so callers
# get type narrowing and explicit field names. `did_work=False` means the stage
# found its output already on disk and short-circuited; treat it as success.
@dataclass(frozen=True)
class DownloadResult:
    laz: Path
    lax: Path | None
    did_work: bool


@dataclass(frozen=True)
class ClipResult:
    clipped: Path
    did_work: bool


@dataclass(frozen=True)
class DtmResult:
    dtm: Path
    did_work: bool


@dataclass(frozen=True)
class VegetationResult:
    vegetation_laz: Path
    vegetation_xyz: Path
    did_work: bool


@dataclass(frozen=True)
class SegmentationResult:
    segmentation_xyz: Path
    tree_hulls: Path
    did_work: bool


@dataclass(frozen=True)
class AlphaWrapResult:
    mesh_ply: Path
    did_work: bool


@dataclass(frozen=True)
class TreeMetrics:
    """All per-tree geometric and allometric metrics in global RD CRS."""

    crown_width_m: float
    crown_median_z: float
    porosity: float
    r50_m: float
    height_m: float
    dbh_m: float
    trunk_radius_m: float
    trunk_base_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class Lod3Result:
    components: list[dict]
    attributes: dict


@dataclass(frozen=True)
class GeneralizeForestIdsResult:
    n_trees: int
    forest_hulls: Path
    gtid_map: Path


# ---------------------------------------------------------------------
# Tile-level summary for orchestrator logging
# ---------------------------------------------------------------------
# The per-stage results above are the seam. The TileOutcome below is *not* a
# stage result — it's how the orchestrator collapses a multi-stage tile run
# into one row for the summary log.
@dataclass(frozen=True)
class TileOutcome:
    tile_id: str
    status: str  # "ok" | "not_in_coverage" | "missing_input" | "failed" | ...
    paths: dict[str, Path] = field(default_factory=dict)
    detail: str | None = None


# ---------------------------------------------------------------------
# Failure classification (single source of truth for stage exit codes)
# ---------------------------------------------------------------------
# A stage's `main()` exits non-zero when its tile outcomes include any of
# these, so the orchestrator (`main.py::run_stage`) aborts the pipeline
# instead of letting a downstream consumer cache a partial run as complete.
# Successes ("ok", "downloaded", "complete", "chunk_done", "exists") and
# graceful skips are deliberately NOT failures: "not_in_coverage" is an AHN6
# coverage gap (an expected skip, not an error), and "empty_tile" is a tile
# with no trees (a valid result). Reconstruction's "missing_input" /
# "invalid_input" are likewise left non-fatal, matching that stage's own
# ok/failed/other split, so this set never reclassifies an existing outcome.
FAILURE_STATUSES: frozenset[str] = frozenset(
    {
        "download_failed",
        "clip_failed",
        "dtm_failed",
        "clip_prereq_missing",
        "veg_failed",
        "veg_prereq_missing",
        "seg_failed",
        "seg_prereq_missing",
        "exception",
        "failed",
        "stalled",
        "failed_max_attempts",
    }
)


def failed_statuses(statuses: Iterable[str]) -> list[str]:
    """Return the genuine-failure statuses present in *statuses*.

    Empty when every tile succeeded or was gracefully skipped, so a stage
    can ``sys.exit(1 if failed_statuses(...) else 0)`` and an AHN6 run with
    out-of-coverage tiles still exits 0. See :data:`FAILURE_STATUSES`.
    """
    return [s for s in statuses if s in FAILURE_STATUSES]


# ---------------------------------------------------------------------
# forest.laz enrichment outcomes (generalize_forest_ids)
# ---------------------------------------------------------------------
# A tile's forest.laz enrichment either WROTE its points, was legitimately
# SKIPPED (optional inputs absent, no GTID matches, nothing to do), or
# genuinely FAILED (an I/O error reading segmentation.xyz / vegetation.laz or
# writing forest.laz). The distinction is the point: a SKIP is a valid empty
# result, but a FAILURE means a tile that should have produced trees did not,
# so segmentation must abort rather than let reconstruction proceed on the gap
# and a partial run be cached as complete. Kept here, beside FAILURE_STATUSES,
# so the one-tile-one-status convention has a single home.
FOREST_ENRICH_WRITTEN = "written"
FOREST_ENRICH_SKIPPED = "skipped"
FOREST_ENRICH_FAILED = "failed"


def forest_enrichment_failures(outcomes: Iterable[tuple[str, str]]) -> list[str]:
    """Return tile ids whose forest.laz enrichment genuinely failed.

    *outcomes* is an iterable of ``(tile_id, status)`` pairs. Only
    :data:`FOREST_ENRICH_FAILED` counts; a written tile or a legitimate skip
    does not. An empty result means segmentation may exit 0.
    """
    return [tile_id for tile_id, status in outcomes if status == FOREST_ENRICH_FAILED]


def missing_tiles_exit_code(*, dry_run: bool) -> int:
    """Exit code for a stage whose tiles directory is absent.

    A real run needs tiles to process, so a missing directory is a failure
    (exit 1). A ``--dry-run`` only lists what *would* be processed; with no
    tiles there is simply nothing to list, which is not an error (exit 0). The
    full-pipeline dry-run relies on this: ``get_data --dry-run`` exits before
    downloading and never creates the tiles directory, so the following
    segmentation/reconstruction dry-runs must not abort on its absence.
    """
    return 0 if dry_run else 1


def should_reconstruct(
    *,
    output_exists: bool,
    overwrite: bool,
    existing_is_geometry_only: bool,
    requested_geometry_only: bool,
) -> bool:
    """Whether a tile's reconstruction must (re)run rather than reuse its output.

    Re-run when there is no output, on an explicit overwrite, or when the
    existing output was produced ``--geometry-only`` (its r50/porosity are
    null) but full descriptive metrics are now requested. Reuse otherwise:
    a full output satisfies any request, and a geometry-only output satisfies
    a geometry-only request (the geometry is identical to a full run).
    """
    if not output_exists or overwrite:
        return True
    return existing_is_geometry_only and not requested_geometry_only
