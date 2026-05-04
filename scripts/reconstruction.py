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

The orchestrator bounds the worker's lifetime by tree count anyway, as
defense-in-depth: each tile is processed in chunks of N trees per fresh
subprocess; after N trees the worker exits cleanly and the orchestrator
spawns the next one. M parallel tile workers stay safely within RAM as
long as M x N x per_tree_peak < system_RAM. The architecture also makes
the worst case predictable when running on memory-constrained hosts.

State on disk per tile (under `_cache/`):
- `trees/<gtid>.pkl` — completed tree result (atomic write via .tmp + rename)
- `in_flight.txt`    — gtid currently being processed; cleared on success
- `skipped.txt`      — gtids that crashed the worker `_PATHOLOGY_THRESHOLD`
                       times in a row (only set in genuine pathology cases)

A worker crashing while no tree is in flight counts as a transient
failure and the chunk is retried, with a small budget.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import laspy
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from trimesh import load as load_mesh

from src.config import ResolvedConfig, get_config, setup_logger
from src.reconstruction.alpha_wrap_tree import alpha_wrap_tree
from src.reconstruction.construct_geometry import construct_lod3
from src.reconstruction.extract_tree_metrics import compute_tree_metrics
from src.reconstruction.write_cityjson import add_tree, finalize_cityjson, init_cityjson
from src.stages import StageError
from src.tile_layout import CaseLayout, TileCacheLayout, TileLayout

# ---------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------
# Default trees per chunk. With embreex available, per-tree native
# allocation is bounded (~10-50 MB) and frequently plateaus, so a worker
# can comfortably process a few hundred trees. Larger chunks amortize
# subprocess startup (~7-10 s import + warmup + LAS load) over more trees.
_DEFAULT_CHUNK_SIZE = 200

# Sanity ceiling per chunk. With chunk_size=200 and ~1-3 s per tree,
# normal chunk runtime is well under 15 min. A wall time longer than this
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

# If this many chunks in a row die without making any progress, abort the
# tile. Catches scenarios where every chunk crashes on a different gtid
# due to persistent external pressure or systemic environment failure.
_MAX_CONSECUTIVE_CRASHES = 20

# Grace period to drain the result queue after a worker exits cleanly.
# `Queue.put` from the child can race with the join() return on the parent
# side; a short timeout closes that window without slowing the happy path.
_QUEUE_DRAIN_TIMEOUT_S = 2.0


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
# Worker: process up to `chunk_size` trees and exit cleanly
# ---------------------------------------------------------------------
def process_chunk(
    tile_dir: Path,
    cfg: dict,
    chunk_size: int,
    overwrite: bool,
    keep_cache: bool,
    max_trees: int | None,
    result_q: mp.Queue,
) -> None:
    """Process up to `chunk_size` pending trees from a tile, then exit.

    Communicates back via `result_q`:
      {"status": "complete", "n_trees": int}  — all trees done, CityJSON written
      {"status": "chunk_done", "n_chunk": int, "n_total": int}  — more chunks needed
      {"status": "exists"}        — CityJSON already on disk
      {"status": "missing_input"} — required tile inputs absent
      {"status": "empty_tile"}    — LAS has no GTIDs

    On a worker crash mid-tree, the orchestrator inspects `_cache/in_flight.txt`
    to identify the offending gtid; this function is then responsible for
    just writing the in-flight marker.
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
        result_q.put({"tile_id": tile_id, "status": "missing_input"})
        return

    if tile.cityjson.exists() and not overwrite:
        result_q.put({"tile_id": tile_id, "status": "exists"})
        return

    if overwrite and cache.root.exists():
        shutil.rmtree(cache.root)
    cache.trees_dir.mkdir(parents=True, exist_ok=True)

    completed = {int(p.stem) for p in cache.trees_dir.glob("*.pkl")}
    skipped = _read_int_set(cache.skipped)

    with laspy.open(tile.forest_laz) as lf:
        las = lf.read()
    if "gtid" not in las.point_format.dimension_names:
        result_q.put({"tile_id": tile_id, "status": "invalid_input"})
        return
    unique_gtids = np.unique(las["gtid"])
    if max_trees:
        unique_gtids = unique_gtids[:max_trees]
    if len(unique_gtids) == 0:
        result_q.put({"tile_id": tile_id, "status": "empty_tile"})
        return

    pending = [int(g) for g in unique_gtids if int(g) not in completed and int(g) not in skipped]
    n_total = len(unique_gtids)

    # Build a single gtid -> indices map up front. Scanning `las["gtid"] == gid`
    # per tree is O(N) per iteration; for tiles with thousands of trees and
    # tens of millions of points this dominates the chunk runtime. Sorting
    # once and bucketing is O(N log N + N) total.
    gtid_arr = las["gtid"]
    sort_order = np.argsort(gtid_arr, kind="stable")
    sorted_gtids = gtid_arr[sort_order]
    boundaries = np.searchsorted(sorted_gtids, np.array(pending, dtype=gtid_arr.dtype))
    boundaries_end = np.searchsorted(sorted_gtids, np.array(pending, dtype=gtid_arr.dtype), side="right")
    gtid_indices: dict[int, np.ndarray] = {
        gid: sort_order[start:end]
        for gid, start, end in zip(pending, boundaries, boundaries_end, strict=True)
    }

    logging.info(
        f"[{tile_id}] Chunk start: {len(pending)} pending of {n_total} trees "
        f"(completed={len(completed)}, skipped={len(skipped)}, chunk_size={chunk_size})"
    )

    n_processed_this_chunk = 0
    for gid in pending:
        if n_processed_this_chunk >= chunk_size:
            # Hand off to the next worker; this worker exits cleanly so the
            # OS reclaims its accumulated C-extension memory.
            logging.info(f"[{tile_id}] Chunk budget reached ({chunk_size} trees); recycling worker")
            result_q.put(
                {
                    "tile_id": tile_id,
                    "status": "chunk_done",
                    "n_chunk": n_processed_this_chunk,
                    "n_completed": len(completed),
                    "n_total": n_total,
                }
            )
            return

        idxs = gtid_indices[gid]
        if idxs.size < _MIN_POINTS_PER_TREE:
            continue

        # Mark in-flight BEFORE doing the work, durable across SIGKILL.
        # If the worker dies after this point, the orchestrator reads this
        # file to identify the gtid that caused the crash.
        _write_atomic(cache.in_flight, str(gid))

        pts = np.c_[las.x[idxs], las.y[idxs], las.z[idxs]]
        offset = pts.mean(axis=0)
        local_pts = pts - offset

        xyz_path = cache.tree_xyz(gid)
        np.savetxt(xyz_path, local_pts, fmt="%.6f")

        try:
            res_alpha = alpha_wrap_tree(xyz_path, cache.root, overwrite=False)
        except StageError as e:
            logging.warning(f"[{tile_id}] GTID {gid}: alpha wrap failed ({e})")
            cache.in_flight.unlink(missing_ok=True)
            continue

        try:
            mesh = load_mesh(res_alpha.mesh_ply)
        except Exception as e:
            logging.warning(f"[{tile_id}] GTID {gid}: mesh load failed ({e})")
            cache.in_flight.unlink(missing_ok=True)
            continue

        try:
            metrics = compute_tree_metrics(mesh, local_pts, tile.dtm, offset)
        except StageError as e:
            logging.warning(f"[{tile_id}] GTID {gid}: metrics failed ({e})")
            del mesh, local_pts
            cache.in_flight.unlink(missing_ok=True)
            continue

        tree_geom = construct_lod3(mesh, metrics, offset, gtid=gid, tile_id=tile_id)
        if not tree_geom.components:
            del mesh, local_pts, res_alpha
            cache.in_flight.unlink(missing_ok=True)
            continue

        _save_tree_pkl(cache, gid, tree_geom.components, offset, tree_geom.attributes)
        completed.add(gid)
        n_processed_this_chunk += 1

        # Tree fully done: clear the in-flight marker so a later crash
        # is not blamed on this gtid.
        cache.in_flight.unlink(missing_ok=True)

        del mesh, local_pts, res_alpha, tree_geom
        gc.collect()

    # Assemblage of the final CityJSON.
    city = init_cityjson()
    for pkl_path in sorted(cache.trees_dir.glob("*.pkl"), key=lambda p: int(p.stem)):
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
        result_q.put({"tile_id": tile_id, "status": "empty_tile"})
        return

    city_final = finalize_cityjson(city)
    with open(tile.cityjson, "w", encoding="utf-8") as f:
        json.dump(city_final, f, indent=2)
    logging.info(f"[{tile_id}] CityJSON written: {tile.cityjson.name} ({n_objects} trees, {len(skipped)} skipped)")

    if not keep_cache:
        shutil.rmtree(cache.root, ignore_errors=True)

    result_q.put(
        {
            "tile_id": tile_id,
            "status": "complete",
            "n_trees": n_objects,
            "n_skipped": len(skipped),
        }
    )


# ---------------------------------------------------------------------
# Worker entry: configures logging then runs the chunk
# ---------------------------------------------------------------------
def _worker_entry(tile_dir, cfg, chunk_size, overwrite, keep_cache, max_trees, log_level, result_q):
    setup_logger(cfg["case"], "tree_reconstruction", level=log_level)
    for noisy in ["trimesh", "rasterio", "fiona", "shapely"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    process_chunk(tile_dir, cfg, chunk_size, overwrite, keep_cache, max_trees, result_q)


# ---------------------------------------------------------------------
# Per-tile orchestrator: drives a tile to completion via chunk recycling
# ---------------------------------------------------------------------
def _run_tile(
    tile_dir: Path,
    cfg: ResolvedConfig,
    chunk_size: int,
    overwrite: bool,
    keep_cache: bool,
    max_trees: int | None,
    log_level: str,
) -> dict:
    tile = TileLayout(tile_dir)
    tile_id = tile.tile_id
    cache = tile.cache

    if tile.cityjson.exists() and not overwrite:
        logging.info(f"[{tile_id}] CityJSON already exists — skipping")
        return {"tile_id": tile_id, "status": "exists", "chunks": 0}

    ctx = mp.get_context("spawn")
    chunks_run = 0
    consecutive_crashes = 0  # crashes since last successful chunk
    crashes_per_gtid: dict[int, int] = {}  # blame counter per gtid

    while True:
        chunks_run += 1
        # `overwrite` is honored only on the first chunk.
        # Subsequent chunks must resume.
        chunk_overwrite = overwrite and chunks_run == 1

        result_q: mp.Queue = ctx.Queue()
        proc = ctx.Process(
            target=_worker_entry,
            args=(
                tile_dir,
                cfg,
                chunk_size,
                chunk_overwrite,
                keep_cache,
                max_trees,
                log_level,
                result_q,
            ),
            name=f"recon-{tile_id}-c{chunks_run}",
        )
        proc.start()
        proc.join(timeout=_CHUNK_TIMEOUT_S)

        timed_out = proc.is_alive()
        if timed_out:
            logging.warning(f"[{tile_id}] Chunk {chunks_run} hit {_CHUNK_TIMEOUT_S}s timeout — terminating")
            proc.terminate()
            proc.join(timeout=10)
            if proc.is_alive():
                proc.kill()
                proc.join()

        # Worker may or may not have put a result on the queue. Clean exits
        # always put one but the put-side flush can race with proc.join()
        # returning, so use a short blocking timeout instead of get_nowait().
        # Crashes may not put anything; the timeout ensures we don't wait
        # forever in that case.
        result = None
        with contextlib.suppress(Exception):
            result = result_q.get(timeout=_QUEUE_DRAIN_TIMEOUT_S)

        if result is not None and result.get("status") in (
            "complete",
            "exists",
            "missing_input",
            "empty_tile",
            "invalid_input",
        ):
            return {**result, "chunks": chunks_run}

        if result is not None and result["status"] == "chunk_done":
            consecutive_crashes = 0
            crashes_per_gtid.clear()  # any tree that ran ok is exonerated
            logging.info(
                f"[{tile_id}] Chunk {chunks_run} ok: +{result['n_chunk']} trees "
                f"(total {result['n_completed']}/{result['n_total']})"
            )
            continue

        # Worker died. Inspect the in-flight marker to identify the likely culprit gtid, and
        # update the crash counters. If the same gtid causes _PATHOLOGY_THRESHOLD consecutive
        # crashes, mark it pathological and skip it permanently.
        in_flight = cache.in_flight.read_text().strip() if cache.in_flight.exists() else ""
        n_completed = len(list(cache.trees_dir.glob("*.pkl"))) if cache.trees_dir.exists() else 0
        reason = "timeout" if timed_out else f"exitcode={proc.exitcode}"
        consecutive_crashes += 1

        if in_flight:
            try:
                bad_gtid = int(in_flight)
            except ValueError:
                bad_gtid = None
            if bad_gtid is not None:
                crashes_per_gtid[bad_gtid] = crashes_per_gtid.get(bad_gtid, 0) + 1
                count = crashes_per_gtid[bad_gtid]
                cache.in_flight.unlink(missing_ok=True)
                if count >= _PATHOLOGY_THRESHOLD:
                    _append_durable(cache.skipped, str(bad_gtid))
                    del crashes_per_gtid[bad_gtid]
                    logging.warning(
                        f"[{tile_id}] Chunk {chunks_run} died ({reason}) on GTID {bad_gtid} "
                        f"({count}x consecutive) — marked pathological and skipped permanently"
                    )
                else:
                    logging.warning(
                        f"[{tile_id}] Chunk {chunks_run} died ({reason}) on GTID {bad_gtid} "
                        f"({count}/{_PATHOLOGY_THRESHOLD}) — retrying"
                    )

        else:
            logging.warning(
                f"[{tile_id}] Chunk {chunks_run} died ({reason}) with no in-flight gtid "
                f"(crash during startup or finalize) — retrying"
            )

        if consecutive_crashes >= _MAX_CONSECUTIVE_CRASHES:
            logging.error(
                f"[{tile_id}] Aborting after {consecutive_crashes} consecutive crashes "
                f"(no clean chunk in between). Likely persistent environment issue."
            )
            return {
                "tile_id": tile_id,
                "status": "failed",
                "n_trees": n_completed,
                "chunks": chunks_run,
            }


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run 3D tree reconstruction (Step 3)")
    parser.add_argument("--case", type=str, help="Case name (default from config if omitted)")
    parser.add_argument(
        "--n-cores",
        type=int,
        help="Number of tiles to process in parallel (default from config)",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--max-trees", type=int, default=None, help="Limit trees per tile (for testing)")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=_DEFAULT_CHUNK_SIZE,
        help=f"Trees per worker invocation before recycling (default: {_DEFAULT_CHUNK_SIZE}). "
        "Bigger chunks amortize subprocess startup over more trees; smaller "
        "chunks bound peak RSS more tightly. With embreex installed, per-worker "
        "RSS plateaus around 2-4 GB regardless of chunk size; lower the value "
        "only on memory-constrained hosts.",
    )
    args = parser.parse_args()

    cfg = get_config(case_name=args.case, n_cores=args.n_cores)
    case = cfg["case"]
    n_cores = cfg["default_cores"]

    setup_logger(case, "tree_reconstruction", level=args.log_level)
    for noisy in ["trimesh", "rasterio", "fiona", "shapely"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.info(f"Reconstruction starting: case={case}, n_cores={n_cores}, chunk_size={args.chunk_size}")

    tiles_root = CaseLayout.from_config(cfg).tiles_dir
    if not tiles_root.exists():
        logging.error(f"No tiles found at {tiles_root}")
        return

    tile_dirs = [p for p in tiles_root.iterdir() if p.is_dir()]
    if args.dry_run:
        logging.info(f"Dry run — found {len(tile_dirs)} tiles")
        return

    results = []
    with ThreadPoolExecutor(max_workers=n_cores) as ex:
        futs = {
            ex.submit(
                _run_tile,
                t,
                cfg,
                args.chunk_size,
                args.overwrite,
                args.keep_cache,
                args.max_trees,
                args.log_level,
            ): t
            for t in tile_dirs
        }
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            logging.info(
                f"[{res['tile_id']}] DONE status={res['status']} "
                f"trees={res.get('n_trees', 0)} chunks={res.get('chunks', 0)}"
            )

    ok = [r for r in results if r["status"] in ("ok", "complete", "exists")]
    failed = [r for r in results if r["status"] in ("failed", "stalled", "failed_max_attempts")]
    other = [r for r in results if r not in ok and r not in failed]
    logging.info(f"Reconstruction summary: {len(ok)}/{len(results)} ok")
    if failed:
        failed_summary = ", ".join(f"{r['tile_id']}({r['status']})" for r in failed)
        logging.warning(f"{len(failed)} tile(s) failed: {failed_summary}")
    if other:
        other_summary = ", ".join(f"{r['tile_id']}({r['status']})" for r in other)
        logging.info(f"{len(other)} tile(s) other: {other_summary}")


if __name__ == "__main__":
    main()
