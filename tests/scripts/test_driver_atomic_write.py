"""`_driver._write_bytes_atomically` under concurrent identical writers (ADR-128: two jobs that both see a
complete stage both reduce it and write the same bytes). On Windows a concurrent `os.replace` onto the same
target fails with `PermissionError` (WinError 5); POSIX renames never collide. Measured: the scaffolded
toy's two fanned jobs finished together and both reduces raced on `_summary.json`."""

from __future__ import annotations

import os
import threading

import pytest

from scripts import _driver


def test_identical_bytes_already_in_place_is_success(tmp_path, monkeypatch):
    target = tmp_path / "_summary.json"
    target.write_bytes(b"same")
    real = os.replace

    def busy(src, dst):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(os, "replace", busy)
    _driver._write_bytes_atomically(target, b"same")
    monkeypatch.setattr(os, "replace", real)
    assert target.read_bytes() == b"same"
    assert not list(tmp_path.glob(".*.part")), "the losing writer's temp file is removed"


def test_a_transient_sharing_violation_is_retried(tmp_path, monkeypatch):
    target = tmp_path / "x.json"
    target.write_bytes(b"old")
    real = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError(5, "Access is denied")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    monkeypatch.setattr(_driver, "_REPLACE_BACKOFF_S", 0.0)
    _driver._write_bytes_atomically(target, b"new")
    assert target.read_bytes() == b"new" and calls["n"] == 2


def test_a_persistent_failure_with_other_bytes_still_raises(tmp_path, monkeypatch):
    """Anti-vacuity: the acceptance is ONLY for identical bytes; anything else is never swallowed."""
    target = tmp_path / "x.json"
    target.write_bytes(b"old")

    def always(src, dst):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(os, "replace", always)
    monkeypatch.setattr(_driver, "_REPLACE_BACKOFF_S", 0.0)
    with pytest.raises(PermissionError):
        _driver._write_bytes_atomically(target, b"new")
    assert target.read_bytes() == b"old"


def test_concurrent_identical_writers_never_fail(tmp_path):
    """The real race: many threads write the same bytes to one target at once."""
    target = tmp_path / "_summary.json"
    errors: list[BaseException] = []

    def writer():
        try:
            for _ in range(40):
                _driver._write_bytes_atomically(target, b'{"n_units":4}')
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[:3]
    assert target.read_bytes() == b'{"n_units":4}'


def test_the_temp_name_never_starts_with_dot_underscore(tmp_path, monkeypatch):
    """Real HF (Cycle 1, 2026-10-19): the read-write bucket mount refuses to create a `._*` file
    (`PermissionError: [Errno 13] Permission denied: '/scratch/.../._profile.json.<uuid>.part'`), while
    `.<key>.parquet.<uuid>.part` and `.checksummed` were written. `._` is the macOS AppleDouble prefix. Every
    `_`-prefixed store file (`_summary.json`, `_inputs.json`, `_profile.json`) took that temp name."""
    seen = []
    real = os.replace

    def spy(src, dst):
        seen.append(os.path.basename(src))
        return real(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    for name in ("_summary.json", "_inputs.json", "_profile.json", "m001.parquet"):
        _driver._write_bytes_atomically(tmp_path / name, b"x")
    assert len(seen) == 4 and not [n for n in seen if n.startswith("._")], seen
    assert all(n.startswith(".") and n.endswith(".part") for n in seen), seen  # still hidden temp files
