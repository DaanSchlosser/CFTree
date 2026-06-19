# Copyright (C) 2025 Noah Alting
# Licensed under the GNU General Public License v3.0
# See the LICENSE file for more details.

"""Regression tests for two reuse/retry correctness fixes.

These exercise light, pure seams (no GDAL/PDAL/CGAL): the clip-cache
signature that decides when a cached ``clipped.laz`` is stale, and the
alpha-wrap timeout error that lets the reconstruction worker durably retire a
deterministic CGAL hang instead of re-stalling on it every run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.get_data.clip_tile import clip_cache_valid, clip_signature  # noqa: E402
from src.reconstruction.alpha_wrap_tree import (  # noqa: E402
    AlphaWrapServer,
    AlphaWrapServerError,
    AlphaWrapTimeoutError,
)


# ---------------------------------------------------------------------
# Clip cache: a changed region / input set must invalidate clipped.laz
# ---------------------------------------------------------------------
def test_clip_signature_changes_when_region_changes(tmp_path: Path) -> None:
    region = tmp_path / "clip_region.geojson"
    inputs = [tmp_path / "raw.laz"]
    region.write_text('{"coordinates": [[0, 0]]}')
    before = clip_signature(region, inputs)
    # A different --halo-margin / --buffer rewrites the region geometry.
    region.write_text('{"coordinates": [[0, 99]]}')
    assert clip_signature(region, inputs) != before


def test_clip_signature_changes_when_input_set_changes(tmp_path: Path) -> None:
    region = tmp_path / "clip_region.geojson"
    region.write_text('{"coordinates": [[0, 0]]}')
    one = clip_signature(region, [tmp_path / "raw.laz"])
    # A wider halo pulls in a neighbour tile as an additional input.
    two = clip_signature(region, [tmp_path / "raw.laz", tmp_path / "neighbour.laz"])
    assert one != two


def test_clip_signature_is_order_independent(tmp_path: Path) -> None:
    region = tmp_path / "clip_region.geojson"
    region.write_text('{"coordinates": [[0, 0]]}')
    inputs = [tmp_path / "b.laz", tmp_path / "a.laz"]
    assert clip_signature(region, inputs) == clip_signature(region, list(reversed(inputs)))


def test_clip_cache_valid_only_on_exact_match(tmp_path: Path) -> None:
    sig_path = tmp_path / "clipped.laz.clipsig"
    # A missing sidecar (e.g. a clip from before this guard) is a mismatch.
    assert clip_cache_valid(sig_path, "abc") is False
    sig_path.write_text("abc")
    assert clip_cache_valid(sig_path, "abc") is True
    assert clip_cache_valid(sig_path, "xyz") is False


# ---------------------------------------------------------------------
# Alpha-wrap: a per-tree timeout is a distinct, retire-able failure
# ---------------------------------------------------------------------
class _FakePipe:
    def __init__(self) -> None:
        self.closed = False

    def write(self, _s: str) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class _FakeProc:
    """Minimal stand-in for a live coprocess (poll() -> None means alive)."""

    def __init__(self) -> None:
        self.stdin = _FakePipe()
        self.stdout = _FakePipe()
        self.killed = False

    def poll(self) -> None:
        return None

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout: float | None = None) -> int:
        return 0


def _server_with_fake_proc(monkeypatch: pytest.MonkeyPatch, status):
    server = AlphaWrapServer()
    server._proc = _FakeProc()  # type: ignore[assignment]
    # Drive the status channel directly so no real binary / select is needed.
    monkeypatch.setattr(server, "_read_status", lambda: status)
    return server


def test_wrap_timeout_raises_timeout_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    xyz = tmp_path / "tree.xyz"
    xyz.write_text("0 0 0\n")
    server = _server_with_fake_proc(monkeypatch, status=None)  # None == timeout
    with pytest.raises(AlphaWrapTimeoutError):
        server.wrap(xyz, tmp_path / "tree.ply")


def test_wrap_eof_is_not_a_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A coprocess death (EOF) is a generic failure, not the deterministic hang
    # that warrants permanent retirement.
    xyz = tmp_path / "tree.xyz"
    xyz.write_text("0 0 0\n")
    server = _server_with_fake_proc(monkeypatch, status="")  # "" == EOF
    with pytest.raises(AlphaWrapServerError) as excinfo:
        server.wrap(xyz, tmp_path / "tree.ply")
    assert not isinstance(excinfo.value, AlphaWrapTimeoutError)


def test_timeout_error_is_catchable_as_server_error() -> None:
    # The worker's generic ``except AlphaWrapServerError`` / ``except StageError``
    # must still catch a timeout (it is only handled more specifically first).
    assert issubclass(AlphaWrapTimeoutError, AlphaWrapServerError)
