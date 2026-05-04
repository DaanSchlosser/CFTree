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
