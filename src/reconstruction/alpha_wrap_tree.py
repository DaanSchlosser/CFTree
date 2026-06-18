# Copyright (C) 2025 Noah Alting
# Licensed under the GNU General Public License v3.0
# See the LICENSE file for more details.

# src/reconstruction/alpha_wrap_tree.py

"""Persistent alpha-wrap coprocess driving the CGAL CLI binary.

Wraps: src/reconstruction/AlphaWrap/build/awrap_points (its ``--server`` mode)

Each tree's point cloud is read from a ``.xyz`` file and the wrapped mesh is
written to a ``.ply``; the coprocess is fed one tree per request over a pipe.
For one-off / debugging / regression use, the same binary still offers a
single-shot CLI (`awrap_points <in.xyz> [ralpha] [roffset] <out.ply>`), which
produces byte-identical output to this server path.
"""

from __future__ import annotations

import contextlib
import os
import select
import subprocess
from pathlib import Path

from src.stages import AlphaWrapResult, MissingPrerequisiteError, StageFailureError

# A single alpha-wrap of a ~1000-point tree takes a few milliseconds; this
# ceiling only ever fires on a genuinely hung CGAL call (degenerate input).
# It bounds a hang to seconds instead of letting it ride the 30-min chunk
# timeout and orphan the coprocess.
_WRAP_TIMEOUT_S = 120.0


class AlphaWrapServerError(StageFailureError):
    """The persistent coprocess died, hung, or rejected a tree.

    Subclasses `StageFailureError` so the reconstruction worker's existing
    ``except StageError`` handler skips this tree and continues, just like any
    other alpha-wrap failure. The next ``wrap()`` call respawns the coprocess
    automatically.
    """


class AlphaWrapServer:
    """Persistent CGAL alpha-wrap coprocess (`awrap_points --server`).

    The single-shot path pays the CGAL binary's launch cost — process
    creation + dynamic-link of CGAL/GMP/MPFR/Boost + static init, ~10 ms on a
    native disk — for *every* tree, only to wrap ~1000 points in ~5 ms. This
    launches one long-lived process and feeds it many trees over a pipe,
    amortizing that fixed cost across a whole chunk. The wrapped PLY is
    byte-identical to the single-shot binary (the wrap goes through the same
    C++ ``wrap_one``), so geometry and every downstream metric are unchanged.

    Lifetime is meant to equal one reconstruction chunk: a worker opens it,
    feeds it up to ``chunk_size`` trees, and closes it, so the coprocess's
    native memory is reclaimed by the same chunk recycling that bounds the
    Python worker's RSS.

    Robustness:
      * Lazy spawn — the process starts on the first ``wrap()``, so a chunk
        with no pending trees (resume/finalize-only) costs nothing.
      * Lazy respawn — if the coprocess has died (a pathological tree crashed
        CGAL), the next ``wrap()`` starts a fresh one; the crashing tree is
        skipped like any other alpha-wrap failure.
      * Per-tree timeout — a hung wrap is killed in seconds rather than riding
        the 30-min chunk timeout and orphaning the process.
      * stderr is sent to /dev/null so CGAL chatter can never fill a pipe and
        deadlock the status channel.
    """

    def __init__(self, binary_path: Path | None = None, timeout: float = _WRAP_TIMEOUT_S) -> None:
        self._binary = binary_path or Path(__file__).parent / "AlphaWrap" / "build" / "awrap_points"
        self._timeout = timeout
        self._proc: subprocess.Popen[str] | None = None

    # -- process lifecycle -------------------------------------------------
    def _spawn(self) -> None:
        if not self._binary.exists():
            raise MissingPrerequisiteError(f"Missing alpha wrap binary: {self._binary}")
        # text mode + line buffering: the protocol is one text line per
        # direction. stderr -> DEVNULL so an undrained pipe can never wedge
        # the stdout status channel (see class docstring).
        self._proc = subprocess.Popen(
            [str(self._binary), "--server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def _reap(self) -> None:
        """Drop the current process handle, closing pipes and reaping it."""
        if self._proc is None:
            return
        for stream in (self._proc.stdin, self._proc.stdout):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass
        with contextlib.suppress(Exception):
            self._proc.wait(timeout=5)
        self._proc = None

    def _kill(self) -> None:
        if self._proc is not None:
            with contextlib.suppress(Exception):
                self._proc.kill()
        self._reap()

    def _ensure_alive(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            # First use, or the coprocess exited on the previous tree.
            self._reap()
            self._spawn()

    def _read_status(self) -> str | None:
        """Read one status line. Returns the line, "" on EOF, or None on timeout.

        On POSIX, ``select`` bounds the wait on the stdout pipe. The protocol is
        strict lockstep (one command -> one flushed status line), so at most one
        line is ever outstanding and there is no buffered-readahead to hide from
        ``select``. On non-POSIX (no pipe select) we fall back to a plain
        blocking readline; that path is unused because the pipeline runs on
        Linux.
        """
        assert self._proc is not None and self._proc.stdout is not None
        if os.name == "posix":
            try:
                ready, _, _ = select.select([self._proc.stdout], [], [], self._timeout)
            except (ValueError, OSError):
                return ""  # fd already closed -> treat as death
            if not ready:
                return None
        return self._proc.stdout.readline()

    # -- the hot path ------------------------------------------------------
    def wrap(
        self,
        tree_xyz: Path,
        mesh_ply: Path,
        ralpha: float = 15.0,
        roffset: float = 50.0,
    ) -> AlphaWrapResult:
        """Wrap one tree's point cloud, writing the mesh to ``mesh_ply``.

        Returns an ``AlphaWrapResult`` on success, raises a ``StageError``
        subclass on failure (which the caller skips). The output PLY is always
        regenerated: the worker already excludes completed trees from its
        pending list, so any stale ``.ply`` for a pending tree must be rewritten.
        """
        tree_xyz = Path(tree_xyz)
        mesh_ply = Path(mesh_ply)
        if not tree_xyz.exists():
            raise MissingPrerequisiteError(f"Missing input file: {tree_xyz}")
        # Tab/newline would corrupt the line-framed command. Cache paths never
        # contain these, but guard rather than silently mis-wrap.
        if any(c in f"{tree_xyz}{mesh_ply}" for c in ("\t", "\n", "\r")):
            raise StageFailureError(f"path contains a tab/newline, cannot frame command: {tree_xyz} / {mesh_ply}")

        self._ensure_alive()
        assert self._proc is not None and self._proc.stdin is not None
        cmd = f"{tree_xyz}\t{ralpha}\t{roffset}\t{mesh_ply}\n"
        try:
            self._proc.stdin.write(cmd)
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            self._kill()
            raise AlphaWrapServerError(f"coprocess write failed: {e}") from e

        status = self._read_status()
        if status is None:
            self._kill()
            raise AlphaWrapServerError(f"alpha wrap timed out after {self._timeout:.0f}s")
        if status == "":
            self._reap()
            raise AlphaWrapServerError("coprocess died during wrap")

        status = status.rstrip("\n")
        if status == "OK":
            return AlphaWrapResult(mesh_ply=mesh_ply, did_work=True)
        reason = status.split("\t", 1)[1] if "\t" in status else status
        raise StageFailureError(f"alpha wrap failed: {reason}")

    # -- teardown ----------------------------------------------------------
    def close(self) -> None:
        """Close stdin (EOF -> coprocess exits cleanly), then reap it."""
        if self._proc is not None and self._proc.poll() is None:
            try:
                if self._proc.stdin is not None:
                    self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=10)
            except Exception:
                self._kill()
                return
        self._reap()

    def __enter__(self) -> AlphaWrapServer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
