# Copyright (C) 2025 Noah Alting
# Licensed under the GNU General Public License v3.0
# See the LICENSE file for more details.

#!/usr/bin/env python
"""
scripts/reconstruction.py

Step 3, 3D geometry reconstruction.

Architecture
------------
The per-tree pipeline (alpha-wrap → trimesh metrics → CGAL/embree-backed
geometry checks) accumulates a small amount of native memory per tree in
C-extension caches that Python's garbage collector cannot reclaim. With
the `embreex` ray-mesh accelerator installed (see environment.yml), the
leak is bounded enough that a worker can reasonably process several
hundred trees before its RSS becomes a concern.

Work is split into *batches* of up to `chunk_size` trees. Each batch runs in
its own fresh `spawn` subprocess and exits when done, so the OS reclaims the
accumulated C-extension memory between batches (the same bound that used to be
called "chunk recycling"). A pool of up to `n_cores` batch subprocesses runs
concurrently, drawn from a single global queue across ALL tiles, so the machine
stays saturated regardless of how the trees are distributed over tiles: a single
tile with thousands of trees is split into many batches that run in parallel,
not one sequential stream, and small tiles no longer leave most cores idle.

Crash isolation: each batch is a raw `multiprocessing.Process` (not a pool
worker), joined with a timeout and its exit code inspected, so a CGAL/embree
segfault takes down only that one batch and is retried, never the whole run.

State on disk per tile, in a fast local scratch cache (see
`TileLayout.cache`; on local temp by default, not under the data dir, so the
fsync-heavy churn stays off a slow bind/9p mount):
- `trees/<gtid>.pkl`       — completed tree result (atomic write via .tmp + rename)
- `in_flight.<tag>.txt`    — gtid a given batch is processing; cleared on success.
                             Per-batch so concurrent batches of one tile do not
                             clobber each other's crash marker.
- `skipped.txt`            — gtids that crashed a worker `_PATHOLOGY_THRESHOLD`
                             times in a row (only set in genuine pathology cases)

A worker crashing while no tree is in flight counts as a transient
failure and the batch is retried, with a small budget.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import logging
import multiprocessing as mp
import os
import pickle
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import laspy
import numpy as np
import rasterio
import trimesh
from scipy.spatial import cKDTree
from trimesh import load as load_mesh

from src.config import ResolvedConfig, get_config, setup_logger
from src.reconstruction.alpha_wrap_tree import AlphaWrapServer, AlphaWrapTimeoutError
from src.reconstruction.construct_geometry import construct_lod3
from src.reconstruction.extract_tree_metrics import compute_tree_metrics
from src.reconstruction.write_cityjson import add_tree, finalize_cityjson, init_cityjson
from src.stages import StageError, missing_tiles_exit_code, should_reconstruct
from src.tile_layout import CaseLayout, TileCacheLayout, TileLayout

# ---------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------
# Default trees per batch. With embreex available, per-tree native
# allocation is bounded (~10-50 MB) and frequently plateaus, so a worker
# can comfortably process a few hundred trees. Larger batches amortize
# subprocess startup (~3 s import + warmup + LAS load) over more trees;
# smaller batches expose more parallelism and bound peak RSS more tightly.
_DEFAULT_CHUNK_SIZE = 200

# Sanity ceiling per batch. With chunk_size=200 and ~1-3 s per tree,
# normal batch runtime is well under 15 min. A wall time longer than this
# means the worker is hung (degenerate alpha-wrap input, embree assertion
# stuck); terminate and treat the in-flight gtid as a crash.
_CHUNK_TIMEOUT_S = 30 * 60

# Minimum points required to attempt reconstruction for a single tree.
# Below this, the alpha-wrap output is too sparse to be meaningful.
_MIN_POINTS_PER_TREE = 50

# A tree is marked pathological (and permanently skipped) only after this
# many *consecutive* worker deaths blame the same gtid. Set above 1 so a
# transient external memory pressure event (other processes competing for
# RAM) cannot falsely flag a healthy tree.
_PATHOLOGY_THRESHOLD = 3

# If this many batch attempts in a row die without making any progress, abort
# the batch. Catches scenarios where every attempt crashes on a different gtid
# due to persistent external pressure or systemic environment failure.
_MAX_CONSECUTIVE_CRASHES = 20

# Grace period to drain the result queue after a worker exits cleanly.
# `Queue.put` from the child can race with the join() return on the parent
# side; a short timeout closes that window without slowing the happy path.
_QUEUE_DRAIN_TIMEOUT_S = 2.0

# Run a full `gc.collect()` once every this many completed trees rather than
# after every tree. The per-tree `del` already drops references so non-cyclic
# objects are freed immediately; gc only reclaims reference cycles (trimesh
# caches), and a full generational sweep per tree is pure overhead across
# hundreds of thousands of trees. Native (embree) memory is reclaimed by
# worker recycling, not by gc, so this does not affect the RSS ceiling.
_GC_INTERVAL = 32


# ---------------------------------------------------------------------
# C-extension warmup (avoids first-call latency in the hot loop)
# ---------------------------------------------------------------------
def _warmup_once() -> None:
    pts = np.random.rand(2000, 3).astype(np.float32)
    q = np.random.rand(200, 3).astype(np.float32)
    cKDTree(pts, compact_nodes=True, balanced_tree=True).query(q, k=1, workers=1)

    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=int)
    m = trimesh.Trimesh(verts, faces, process=False)
    _ = m.contains(np.array([[0.1, 0.1, 0.1], [2, 2, 2]], dtype=float))
    _ = m.voxelized(0.2).fill()


# ---------------------------------------------------------------------
# Per-tile state helpers
# ---------------------------------------------------------------------
def _read_int_set(path: Path) -> set[int]:
    if not path.exists():
        return set()
    try:
        return {int(line.strip()) for line in path.read_text().splitlines() if line.strip()}
    except Exception:
        return set()


def _write_atomic(path: Path, content: str) -> None:
    """Write text durably: write+fsync to .tmp, then atomic rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def _append_durable(path: Path, line: str) -> None:
    """Append a line and fsync so it survives a SIGKILL on the next instruction."""
    with open(path, "a") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def _in_flight_path(cache: TileCacheLayout, tag: str) -> Path:
    """Per-batch in-flight marker.

    Each batch gets its own marker so two batches of the same tile, running
    concurrently, never overwrite each other's "the gtid I died on" record.
    """
    return cache.root / f"in_flight.{tag}.txt"


def _save_tree_pkl(cache: TileCacheLayout, gtid: int, components, offset, attributes) -> None:
    final = cache.tree_pkl(gtid)
    tmp = final.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(
            {
                "gtid": gtid,
                "components": components,
                "offset": offset,
                "attributes": attributes,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    tmp.replace(final)


# ---------------------------------------------------------------------
# Work units
# ---------------------------------------------------------------------
@dataclass
class _Batch:
    """One unit of reconstruction work: a fixed set of gtids from one tile."""

    tile_dir: Path
    tile_id: str
    gtids: list[int]
    tag: str  # unique within the tile, e.g. "b0", "b1", ...


@dataclass
class _TilePlan:
    """Planning result for a tile: what to do and the batches to run."""

    tile_dir: Path
    tile_id: str
    status: str  # "needs_work" | "exists" | "missing_input" | "empty_tile" | "invalid_input"
    geometry_only: bool = False
    unique_gtids: list[int] = field(default_factory=list)
    n_total: int = 0
    batches: list[_Batch] = field(default_factory=list)


# ---------------------------------------------------------------------
# Worker: process one explicit batch of trees, then exit cleanly
# ---------------------------------------------------------------------
def process_batch(
    tile_dir: Path,
    cfg: dict,
    gtids: list[int],
    tag: str,
    geometry_only: bool,
    result_q: mp.Queue,
) -> None:
    """Reconstruct exactly the trees in `gtids` for one tile, then exit.

    `gtids` is this batch's fixed assignment (disjoint from every other batch of
    the tile). Trees already on disk (a completed `.pkl`) or retired (`skipped`)
    are re-skipped, so re-running the same batch after a crash resumes where it
    left off. This function never finalizes the CityJSON and never wipes the
    cache; the parent does both, once per tile.

    Communicates back via `result_q`:
      {"status": "batch_ok", "n_done": int}  — every assigned tree resolved
      {"status": "missing_input"}            — required tile inputs absent
      {"status": "invalid_input"}            — forest.laz lacks a gtid dimension

    On a worker crash mid-tree, the parent inspects `in_flight.<tag>.txt` to
    identify the offending gtid; this function writes that marker before each tree.
    """
    os.environ.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    _warmup_once()

    tile = TileLayout(tile_dir)
    tile_id = tile.tile_id
    cache = tile.cache

    if not tile.forest_laz.exists() or not tile.dtm.exists():
        result_q.put({"tile_id": tile_id, "tag": tag, "status": "missing_input"})
        return

    cache.trees_dir.mkdir(parents=True, exist_ok=True)

    completed = {int(p.stem) for p in cache.trees_dir.glob("*.pkl")}
    skipped = _read_int_set(cache.skipped)

    with laspy.open(tile.forest_laz) as lf:
        las = lf.read()
    if "gtid" not in las.point_format.dimension_names:
        result_q.put({"tile_id": tile_id, "tag": tag, "status": "invalid_input"})
        return

    # Only the trees this batch owns that are not already done or retired.
    pending = [int(g) for g in gtids if int(g) not in completed and int(g) not in skipped]

    # Build a gtid -> point-indices map for just this batch's trees. Scanning
    # `las["gtid"] == gid` per tree is O(N) per iteration; sorting once and
    # bucketing is O(N log N + N) total. Done per batch (not per tile) so each
    # batch only pays for the trees it owns.
    gtid_indices: dict[int, np.ndarray] = {}
    if pending:
        gtid_arr = las["gtid"]
        sort_order = np.argsort(gtid_arr, kind="stable")
        sorted_gtids = gtid_arr[sort_order]
        starts = np.searchsorted(sorted_gtids, np.array(pending, dtype=gtid_arr.dtype))
        ends = np.searchsorted(sorted_gtids, np.array(pending, dtype=gtid_arr.dtype), side="right")
        gtid_indices = {gid: sort_order[s:e] for gid, s, e in zip(pending, starts, ends, strict=True)}

    logging.info(
        f"[{tile_id}] Batch {tag} start: {len(pending)} pending of {len(gtids)} assigned "
        f"(already done={len(gtids) - len(pending)}{', geometry-only' if geometry_only else ''})"
    )

    in_flight = _in_flight_path(cache, tag)

    # Open the tile DTM once and reuse it for every tree's trunk-base lookup, and
    # launch one persistent alpha-wrap coprocess for the whole batch instead of
    # spawning the CGAL binary per tree (~10 ms launch each). Both are torn down
    # in the `finally` below on every exit path; the coprocess also exits on its
    # own when this short-lived worker dies, because its stdin pipe closes.
    dtm_src = rasterio.open(tile.dtm)
    awrap = AlphaWrapServer()

    try:
        n_done = 0
        for gid in pending:
            idxs = gtid_indices[gid]
            if idxs.size < _MIN_POINTS_PER_TREE:
                continue

            # Mark in-flight BEFORE doing the work, durable across SIGKILL.
            # If the worker dies after this point, the parent reads this file to
            # identify the gtid that caused the crash.
            _write_atomic(in_flight, str(gid))

            pts = np.c_[las.x[idxs], las.y[idxs], las.z[idxs]]
            offset = pts.mean(axis=0)
            local_pts = pts - offset

            xyz_path = cache.tree_xyz(gid)
            np.savetxt(xyz_path, local_pts, fmt="%.6f")

            try:
                res_alpha = awrap.wrap(xyz_path, cache.tree_ply(gid))
            except AlphaWrapTimeoutError as e:
                # A genuine CGAL hang: deterministic and expensive (it re-stalls
                # the full per-tree timeout on every resume). Retire the gtid
                # durably so a rerun skips it, rather than paying the timeout
                # again. The persistent coprocess died with the wrap, so it
                # respawns on the next tree.
                logging.warning(f"[{tile_id}] GTID {gid}: alpha wrap timed out — retiring permanently ({e})")
                _append_durable(cache.skipped, str(gid))
                skipped.add(gid)
                in_flight.unlink(missing_ok=True)
                continue
            except StageError as e:
                logging.warning(f"[{tile_id}] GTID {gid}: alpha wrap failed ({e})")
                in_flight.unlink(missing_ok=True)
                continue

            try:
                mesh = load_mesh(res_alpha.mesh_ply)
            except Exception as e:
                logging.warning(f"[{tile_id}] GTID {gid}: mesh load failed ({e})")
                in_flight.unlink(missing_ok=True)
                continue

            try:
                metrics = compute_tree_metrics(mesh, local_pts, dtm_src, offset, compute_semantics=not geometry_only)
            except StageError as e:
                logging.warning(f"[{tile_id}] GTID {gid}: metrics failed ({e})")
                del mesh, local_pts
                in_flight.unlink(missing_ok=True)
                continue

            tree_geom = construct_lod3(mesh, metrics, offset, gtid=gid, tile_id=tile_id)
            if not tree_geom.components:
                del mesh, local_pts, res_alpha
                in_flight.unlink(missing_ok=True)
                continue

            _save_tree_pkl(cache, gid, tree_geom.components, offset, tree_geom.attributes)
            completed.add(gid)
            n_done += 1

            # Tree fully done: clear the in-flight marker so a later crash
            # is not blamed on this gtid.
            in_flight.unlink(missing_ok=True)

            del mesh, local_pts, res_alpha, tree_geom
            if n_done % _GC_INTERVAL == 0:
                gc.collect()

        result_q.put({"tile_id": tile_id, "tag": tag, "status": "batch_ok", "n_done": n_done})
    finally:
        awrap.close()
        dtm_src.close()


# ---------------------------------------------------------------------
# Worker entry: configures logging then runs the batch
# ---------------------------------------------------------------------
def _worker_entry(tile_dir, cfg, gtids, tag, geometry_only, log_level, result_q):
    setup_logger(cfg["case"], "tree_reconstruction", level=log_level)
    for noisy in ["trimesh", "rasterio", "fiona", "shapely"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    process_batch(tile_dir, cfg, gtids, tag, geometry_only, result_q)


# ---------------------------------------------------------------------
# Per-batch driver: run a batch to completion, retrying worker crashes
# ---------------------------------------------------------------------
def _run_one_batch(
    batch: _Batch,
    cfg: ResolvedConfig,
    geometry_only: bool,
    log_level: str,
) -> dict:
    """Drive a single batch to a terminal state, respawning on crash.

    Runs in a thread of the global pool; each attempt is a fresh `spawn`
    subprocess so a CGAL/embree segfault is isolated and retried. Mirrors the
    crash-recovery the per-tile loop used to do, scoped to one batch: the
    per-batch in-flight marker identifies the offending gtid, a gtid that kills
    `_PATHOLOGY_THRESHOLD` attempts in a row is retired permanently, and a batch
    that makes no progress for `_MAX_CONSECUTIVE_CRASHES` attempts is abandoned.

    Returns one of:
      {"status": "batch_ok", ...}                  — all assigned trees resolved
      {"status": "missing_input" | "invalid_input"} — terminal input problem
      {"status": "failed", ...}                     — gave up after repeated crashes
    """
    tile = TileLayout(batch.tile_dir)
    cache = tile.cache
    ctx = mp.get_context("spawn")

    consecutive_crashes = 0
    crashes_per_gtid: dict[int, int] = {}
    attempts = 0

    while True:
        attempts += 1
        result_q: mp.Queue = ctx.Queue()
        proc = ctx.Process(
            target=_worker_entry,
            args=(batch.tile_dir, cfg, batch.gtids, batch.tag, geometry_only, log_level, result_q),
            name=f"recon-{batch.tile_id}-{batch.tag}-a{attempts}",
        )
        proc.start()
        proc.join(timeout=_CHUNK_TIMEOUT_S)

        timed_out = proc.is_alive()
        if timed_out:
            logging.warning(f"[{batch.tile_id}] Batch {batch.tag} hit {_CHUNK_TIMEOUT_S}s timeout — terminating")
            proc.terminate()
            proc.join(timeout=10)
            if proc.is_alive():
                proc.kill()
                proc.join()

        # Clean exits always put a result, but the put-side flush can race with
        # proc.join() returning, so use a short blocking timeout rather than
        # get_nowait(). A crash may put nothing; the timeout bounds that wait.
        result = None
        with contextlib.suppress(Exception):
            result = result_q.get(timeout=_QUEUE_DRAIN_TIMEOUT_S)

        if result is not None and result.get("status") in ("batch_ok", "missing_input", "invalid_input"):
            return {**result, "attempts": attempts}

        # Worker died. Inspect the per-batch in-flight marker for the culprit
        # gtid and update the crash counters; retire a gtid that crashes
        # `_PATHOLOGY_THRESHOLD` attempts in a row.
        in_flight = _in_flight_path(cache, batch.tag)
        marker = in_flight.read_text().strip() if in_flight.exists() else ""
        reason = "timeout" if timed_out else f"exitcode={proc.exitcode}"
        consecutive_crashes += 1

        if marker:
            try:
                bad_gtid = int(marker)
            except ValueError:
                bad_gtid = None
            if bad_gtid is not None:
                crashes_per_gtid[bad_gtid] = crashes_per_gtid.get(bad_gtid, 0) + 1
                count = crashes_per_gtid[bad_gtid]
                in_flight.unlink(missing_ok=True)
                if count >= _PATHOLOGY_THRESHOLD:
                    _append_durable(cache.skipped, str(bad_gtid))
                    del crashes_per_gtid[bad_gtid]
                    logging.warning(
                        f"[{batch.tile_id}] Batch {batch.tag} died ({reason}) on GTID {bad_gtid} "
                        f"({count}x consecutive) — marked pathological and skipped permanently"
                    )
                else:
                    logging.warning(
                        f"[{batch.tile_id}] Batch {batch.tag} died ({reason}) on GTID {bad_gtid} "
                        f"({count}/{_PATHOLOGY_THRESHOLD}) — retrying"
                    )
        else:
            logging.warning(
                f"[{batch.tile_id}] Batch {batch.tag} died ({reason}) with no in-flight gtid "
                f"(crash during startup) — retrying"
            )

        if consecutive_crashes >= _MAX_CONSECUTIVE_CRASHES:
            n_done = len({int(p.stem) for p in cache.trees_dir.glob("*.pkl")} & set(batch.gtids))
            logging.error(
                f"[{batch.tile_id}] Abandoning batch {batch.tag} after {consecutive_crashes} crashes "
                f"with no progress. Likely persistent environment issue."
            )
            return {"tile_id": batch.tile_id, "tag": batch.tag, "status": "failed", "n_done": n_done}


# ---------------------------------------------------------------------
# Per-tile finalize: assemble the CityJSON from the cached tree results
# ---------------------------------------------------------------------
def finalize_tile(
    tile_dir: Path,
    cfg: ResolvedConfig,
    geometry_only: bool,
    keep_cache: bool,
    unique_gtids: list[int],
) -> dict:
    """Assemble one tile's CityJSON from its cached per-tree pickles.

    Run once per tile after all its batches finish. Output is independent of the
    order trees were produced in: pickles are read sorted by gtid, so the bytes
    are identical no matter how the batches were scheduled.
    """
    tile = TileLayout(tile_dir)
    tile_id = tile.tile_id
    cache = tile.cache

    # Only emit trees still present in this tile's current forest.laz: an
    # ownership change upstream can REDUCE a tile's gtid set between runs, and a
    # stale <gtid>.pkl from a prior run must not re-introduce a tree this tile no
    # longer owns. (With --overwrite the cache is wiped up front; this guard
    # makes a plain resume safe too.)
    valid_gtids = {int(g) for g in unique_gtids}
    skipped = _read_int_set(cache.skipped)
    city = init_cityjson()
    if cache.trees_dir.exists():
        for pkl_path in sorted(cache.trees_dir.glob("*.pkl"), key=lambda p: int(p.stem)):
            if int(pkl_path.stem) not in valid_gtids:
                continue
            try:
                with open(pkl_path, "rb") as f:
                    t = pickle.load(f)
            except Exception as e:
                logging.warning(f"[{tile_id}] Failed to read {pkl_path.name}: {e}")
                continue
            add_tree(city, t["gtid"], t["components"], t["offset"], t["attributes"])

    n_objects = len(city["CityObjects"])
    if n_objects == 0:
        logging.warning(f"[{tile_id}] No trees reconstructed — skipping CityJSON write.")
        if not keep_cache:
            shutil.rmtree(cache.root, ignore_errors=True)
        return {"tile_id": tile_id, "status": "empty_tile"}

    city_final = finalize_cityjson(city)
    # Compact separators: CityJSON is machine-read, and a tile holds hundreds of
    # thousands of quantized-integer vertices. Dropping indentation whitespace
    # both shrinks the file substantially and speeds serialization; the content
    # is identical.
    with open(tile.cityjson, "w", encoding="utf-8") as f:
        json.dump(city_final, f, separators=(",", ":"))
    logging.info(f"[{tile_id}] CityJSON written: {tile.cityjson.name} ({n_objects} trees, {len(skipped)} skipped)")

    # Record which mode produced this CityJSON. A geometry-only output has null
    # r50/porosity; the marker lets a later full run tell it from a complete one
    # and rebuild rather than reuse the nulls (see _plan_tile).
    if geometry_only:
        _write_atomic(tile.geometry_only_marker, "1")
    else:
        tile.geometry_only_marker.unlink(missing_ok=True)

    if not keep_cache:
        shutil.rmtree(cache.root, ignore_errors=True)

    return {"tile_id": tile_id, "status": "complete", "n_trees": n_objects, "n_skipped": len(skipped)}


# ---------------------------------------------------------------------
# Per-tile planning: decide what to do and split the work into batches
# ---------------------------------------------------------------------
def _plan_tile(
    tile_dir: Path,
    cfg: ResolvedConfig,
    overwrite: bool,
    geometry_only: bool,
    chunk_size: int,
    max_trees: int | None,
) -> _TilePlan:
    """Validate a tile, apply the overwrite/upgrade rules, and list its batches.

    Reading the (small) forest.laz here lets the scheduler build one global batch
    queue across all tiles up front. Cache wipes and stale-output removal happen
    here, in the parent, exactly once per tile, so the concurrent batch workers
    never race on them.
    """
    tile = TileLayout(tile_dir)
    tile_id = tile.tile_id
    cache = tile.cache

    if not tile.forest_laz.exists() or not tile.dtm.exists():
        return _TilePlan(tile_dir, tile_id, status="missing_input")

    existing_geometry_only = tile.geometry_only_marker.exists()
    if not should_reconstruct(
        output_exists=tile.cityjson.exists(),
        overwrite=overwrite,
        existing_is_geometry_only=existing_geometry_only,
        requested_geometry_only=geometry_only,
    ):
        logging.info(f"[{tile_id}] CityJSON already exists — skipping")
        return _TilePlan(tile_dir, tile_id, status="exists")

    # Reaching here with an existing CityJSON and no explicit --overwrite means a
    # geometry-only output is being upgraded to full metrics. Force a rebuild so
    # the cached null-metric trees are recomputed rather than reused.
    if tile.cityjson.exists() and not overwrite:
        logging.info(f"[{tile_id}] Existing CityJSON is geometry-only — rebuilding for full metrics")
        overwrite = True

    # Drop the stale CityJSON and wipe the cache up front on an overwrite run, so
    # batches start from a clean slate and a later finalize sees only fresh trees.
    if overwrite and tile.cityjson.exists():
        tile.cityjson.unlink()
    if overwrite and cache.root.exists():
        shutil.rmtree(cache.root)
    cache.trees_dir.mkdir(parents=True, exist_ok=True)

    with laspy.open(tile.forest_laz) as lf:
        las = lf.read()
    if "gtid" not in las.point_format.dimension_names:
        return _TilePlan(tile_dir, tile_id, status="invalid_input")

    unique_gtids = np.unique(las["gtid"])
    if max_trees:
        unique_gtids = unique_gtids[:max_trees]
    if len(unique_gtids) == 0:
        return _TilePlan(tile_dir, tile_id, status="empty_tile")

    unique_list = [int(g) for g in unique_gtids]
    completed = {int(p.stem) for p in cache.trees_dir.glob("*.pkl")}
    skipped = _read_int_set(cache.skipped)
    pending = [g for g in unique_list if g not in completed and g not in skipped]

    batches = [
        _Batch(tile_dir, tile_id, pending[i : i + chunk_size], f"b{k}")
        for k, i in enumerate(range(0, len(pending), chunk_size))
    ]
    logging.info(
        f"[{tile_id}] Planned {len(pending)} pending of {len(unique_list)} trees "
        f"in {len(batches)} batch(es) (already done={len(completed)}, skipped={len(skipped)})"
    )
    return _TilePlan(
        tile_dir,
        tile_id,
        status="needs_work",
        geometry_only=geometry_only,
        unique_gtids=unique_list,
        n_total=len(unique_list),
        batches=batches,
    )


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Run 3D tree reconstruction (Step 3)")
    parser.add_argument("--case", type=str, help="Case name (default from config if omitted)")
    parser.add_argument(
        "--n-cores",
        type=int,
        help="Maximum reconstruction worker subprocesses to run at once (default from config). "
        "Workers are drawn from a global queue of tree batches across all tiles, so this is the "
        "real parallelism whether the trees sit in one big tile or many small ones. Each worker "
        "holds up to --chunk-size trees in memory, so size this to RAM (peak ~= n_cores x per-worker "
        "RSS); geometry-only workers are lighter than full-metric ones.",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--max-trees", type=int, default=None, help="Limit trees per tile (for testing)")
    parser.add_argument(
        "--geometry-only",
        action="store_true",
        help="Generate crown+trunk geometry only, skipping the expensive descriptive "
        "metrics (r50, porosity). ~5-6x faster reconstruction; the r50/porosity "
        "attributes are written as null. Geometry is identical to a full run.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=_DEFAULT_CHUNK_SIZE,
        help=f"Trees per batch subprocess (default: {_DEFAULT_CHUNK_SIZE}). Bigger batches "
        "amortize subprocess startup over more trees; smaller batches expose more "
        "parallelism and bound peak RSS more tightly. With embreex installed, per-worker "
        "RSS plateaus around 2-4 GB regardless of batch size.",
    )
    args = parser.parse_args()

    cfg = get_config(case_name=args.case, n_cores=args.n_cores)
    case = cfg["case"]
    n_cores = cfg["default_cores"]

    setup_logger(case, "tree_reconstruction", level=args.log_level)
    for noisy in ["trimesh", "rasterio", "fiona", "shapely"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.info(
        f"Reconstruction starting: case={case}, n_cores={n_cores}, chunk_size={args.chunk_size}"
        f"{', geometry-only (no r50/porosity)' if args.geometry_only else ''}"
    )

    tiles_root = CaseLayout.from_config(cfg).tiles_dir
    if not tiles_root.exists():
        # Missing tiles abort a real run, but a --dry-run has nothing to list
        # (its predecessors may not have produced tiles yet). See
        # missing_tiles_exit_code.
        if args.dry_run:
            logging.info(f"No tiles directory yet at {tiles_root} — nothing to list.")
        else:
            logging.error(f"No tiles found at {tiles_root}")
        return missing_tiles_exit_code(dry_run=args.dry_run)

    tile_dirs = [p for p in tiles_root.iterdir() if p.is_dir()]
    if args.dry_run:
        logging.info(f"Dry run — found {len(tile_dirs)} tiles")
        return 0

    # ------------------------------------------------------------------
    # Plan every tile, then run all their batches through one global pool.
    # ------------------------------------------------------------------
    plans = [
        _plan_tile(t, cfg, args.overwrite, args.geometry_only, args.chunk_size, args.max_trees) for t in tile_dirs
    ]

    results: list[dict] = []
    work_plans: list[_TilePlan] = []
    for p in plans:
        if p.status == "needs_work":
            work_plans.append(p)
        else:
            results.append({"tile_id": p.tile_id, "status": p.status})

    plan_by_tile = {p.tile_id: p for p in work_plans}
    all_batches = [b for p in work_plans for b in p.batches]
    # A tile with nothing pending (e.g. resume after every tree was cached, or an
    # overwrite that cleared a now-empty gtid set) still needs its CityJSON
    # assembled; its batch count is zero, so finalize it directly.
    remaining = {p.tile_id: len(p.batches) for p in work_plans}
    tile_failed: set[str] = set()

    n_batches = len(all_batches)
    logging.info(
        f"Scheduling {n_batches} batch(es) across {len(work_plans)} tile(s) on up to {n_cores} workers "
        f"({sum(len(b.gtids) for b in all_batches)} trees pending)"
    )

    def _finalize(plan: _TilePlan) -> None:
        # A tile with any abandoned batch is reported failed and left without a
        # CityJSON (the stage then exits non-zero), matching the old behaviour on
        # a give-up: a partial tree set is never published as if complete.
        if plan.tile_id in tile_failed:
            results.append({"tile_id": plan.tile_id, "status": "failed", "n_trees": 0})
            return
        results.append(finalize_tile(plan.tile_dir, cfg, plan.geometry_only, args.keep_cache, plan.unique_gtids))

    # Tiles with zero batches: finalize straight away.
    for p in work_plans:
        if not p.batches:
            _finalize(p)

    if all_batches:
        with ThreadPoolExecutor(max_workers=max(1, n_cores)) as ex:
            futs = {
                ex.submit(_run_one_batch, b, cfg, plan_by_tile[b.tile_id].geometry_only, args.log_level): b
                for b in all_batches
            }
            for fut in as_completed(futs):
                b = futs[fut]
                res = fut.result()
                if res.get("status") != "batch_ok":
                    tile_failed.add(b.tile_id)
                    logging.error(f"[{b.tile_id}] Batch {b.tag} did not complete: status={res.get('status')}")
                remaining[b.tile_id] -= 1
                if remaining[b.tile_id] == 0:
                    _finalize(plan_by_tile[b.tile_id])

    for res in results:
        logging.info(
            f"[{res['tile_id']}] DONE status={res['status']} "
            f"trees={res.get('n_trees', 0)}"
        )

    ok = [r for r in results if r["status"] in ("ok", "complete", "exists")]
    failed = [r for r in results if r["status"] in ("failed", "stalled", "failed_max_attempts")]
    other = [r for r in results if r not in ok and r not in failed]
    logging.info(f"Reconstruction summary: {len(ok)}/{len(results)} ok")
    if failed:
        failed_summary = ", ".join(f"{r['tile_id']}({r['status']})" for r in failed)
        logging.error(f"{len(failed)} tile(s) failed: {failed_summary}")
    if other:
        other_summary = ", ".join(f"{r['tile_id']}({r['status']})" for r in other)
        logging.info(f"{len(other)} tile(s) other: {other_summary}")

    # Exit non-zero when any tile failed reconstruction, so the orchestrator
    # aborts rather than letting a partial tree set be cached as complete.
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
