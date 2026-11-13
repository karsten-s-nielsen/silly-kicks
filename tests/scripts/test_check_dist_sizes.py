"""The release size check (corpus widening spec 5.6, owner ruling r4; ADR-132): `publish.yml` fails before upload when
the wheel OR the sdist in `dist/` is 80 MB (80_000_000 bytes) or larger. Sparse files keep the stub `dist/` cheap."""

from __future__ import annotations

from scripts.check_dist_sizes import LIMIT_BYTES, check, main


def _file(path, size: int):
    with open(path, "wb") as f:
        f.truncate(size)
    return path


def _dist(tmp_path, wheel: int | None = 1000, sdist: int | None = 2000):
    d = tmp_path / "dist"
    d.mkdir(parents=True)
    if wheel is not None:
        _file(d / "silly_kicks-9.9.9-py3-none-any.whl", wheel)
    if sdist is not None:
        _file(d / "silly_kicks-9.9.9.tar.gz", sdist)
    return d


def test_limit_is_80_mb_decimal():
    assert LIMIT_BYTES == 80_000_000


def test_both_files_under_limit_pass(tmp_path, capsys):
    d = _dist(tmp_path)
    assert check(d) == []
    out = capsys.readouterr().out
    assert "silly_kicks-9.9.9-py3-none-any.whl" in out and "silly_kicks-9.9.9.tar.gz" in out  # every size printed
    assert main([str(d)]) == 0


def test_sdist_over_limit_fails(tmp_path):
    d = _dist(tmp_path, sdist=90_000_000)
    problems = check(d)
    assert len(problems) == 1 and "tar.gz" in problems[0]
    assert main([str(d)]) == 1


def test_wheel_over_limit_fails(tmp_path):
    d = _dist(tmp_path, wheel=90_000_000)
    problems = check(d)
    assert len(problems) == 1 and ".whl" in problems[0]
    assert main([str(d)]) == 1


def test_limit_boundary(tmp_path):
    assert check(_dist(tmp_path / "a", wheel=80_000_000))  # exactly 80 MB fails
    assert check(_dist(tmp_path / "b", wheel=79_999_999)) == []  # one byte under passes


def test_empty_dist_fails(tmp_path):
    assert check(_dist(tmp_path / "w", sdist=None))  # no sdist
    assert check(_dist(tmp_path / "s", wheel=None))  # no wheel
    assert check(_dist(tmp_path / "e", wheel=None, sdist=None))
    assert main([str(tmp_path / "missing")]) == 1
