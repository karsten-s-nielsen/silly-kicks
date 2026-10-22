"""S5.9: the lock records each stage's closure and data membership; CI recomputes and compares."""

from __future__ import annotations

import pytest

from scripts.hf_jobs import closures


def test_closure_file_bytes_normalizes_text_but_not_binary(tmp_path):
    # The closure token must be identical on a CRLF Windows checkout and the LF tree a Linux job builds from
    # (ADR-129; same class as the image-tag CRLF fix). Text files (no NUL) normalize CRLF->LF; a binary file
    # (a NUL byte) is left raw, since normalizing it would corrupt the content hash.
    from scripts._stages import _closure_file_bytes

    def _read(p):
        s = p.stat()
        return _closure_file_bytes(str(p), s.st_mtime_ns, s.st_size)

    lf = tmp_path / "a_lf.py"
    lf.write_bytes(b"x = 1\ny = 2\n")
    crlf = tmp_path / "a_crlf.py"
    crlf.write_bytes(b"x = 1\r\ny = 2\r\n")
    assert _read(lf) == _read(crlf)
    binf = tmp_path / "w.npz"
    binf.write_bytes(b"PK\x00\x03\r\ndata")
    assert _read(binf) == b"PK\x00\x03\r\ndata"  # NUL -> binary, unchanged


def test_the_lock_matches_the_computed_memberships():
    want = closures.record(write=False)
    have = closures.load_lock()
    assert have == want, "closure lock is stale: run `python -m scripts.hf_jobs closures record`"


def test_the_resolver_runs_on_the_real_tree_with_the_scripts_namespace_package():
    """r2 B R2-03: `scripts` has no __init__.py; the resolver must not crash and must still follow it."""
    import pathlib

    assert not pathlib.Path("scripts/__init__.py").exists()  # the real tree
    got = closures.module_closure("scripts._stages")
    assert got and all(p.startswith(("scripts/", "silly_kicks/")) for p in got)


def test_a_package_init_that_an_import_executes_is_in_the_closure():
    got = closures.module_closure("scripts._ghost_gk_stages")
    assert "silly_kicks/__init__.py" in got  # `import silly_kicks.tracking...` executes it (R1 measures its width)


def test_a_function_level_import_is_in_the_closure():
    got = closures.module_closure("scripts._stages")
    assert "scripts/_driver.py" in got  # module-level
    assert "scripts/hf_jobs/store.py" in got  # imported inside `open_store` (r1 B CCTP-SPEC-13)


@pytest.mark.parametrize(
    "module", sorted(__import__("tests.scripts.test_stage_modules", fromlist=["REUSABLE"]).REUSABLE)
)
def test_a_stage_module_closure_never_contains_its_trainer(module):
    import importlib

    from tests.scripts.test_stage_modules import REUSABLE, _callables

    pipe = importlib.import_module(module).PIPELINE
    home = next(iter({fn.__module__ for _, fn in _callables(pipe.stage(REUSABLE[module][0]))}))
    assert module.replace(".", "/") + ".py" not in closures.module_closure(home)


def test_the_census_bare_import_resolves_once():
    # The census bare `from _crossed_icc import ...` lives in the DRIVER (the bootstrap/report side, which
    # the reusable refs/fit_prepass/prefit/score stages never touch), so the resolver proves the bare form
    # (via the `scripts/` sys.path base) de-duplicates to ONE `scripts/_crossed_icc.py` on the driver closure.
    got = closures.module_closure("scripts.build_territory_ranking_census")
    assert got.count("scripts/_crossed_icc.py") == 1


def test_the_population_module_imports_nothing_from_tests():
    """S5.9: a JOB-side closure module must never import `tests/` (the population lives in production code)."""
    import subprocess
    import sys

    code = "import scripts._pipeline_population, sys; assert not any(m.startswith('tests') for m in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)  # noqa: S603
    assert out.returncode == 0, out.stderr


@pytest.mark.parametrize(
    "module", sorted(__import__("tests.scripts.test_stage_modules", fromlist=["REUSABLE"]).REUSABLE)
)
def test_a_reusable_stage_token_keys_on_its_stage_module_not_the_trainer(module):
    """S5.9 token tests: a reusable stage's closure (hashed into its clean-tree token) contains its stage module
    (a load/reduce edit moves the token) and NOT the trainer (a trainer-only edit leaves the token unchanged)."""
    import importlib

    from scripts.hf_jobs.closures import stage_module
    from tests.scripts.test_stage_modules import REUSABLE

    pipe = importlib.import_module(module).PIPELINE
    lock = closures.load_lock()
    trainer_path = module.replace(".", "/") + ".py"
    for name in REUSABLE[module]:
        entry = lock[f"{pipe.name}/{name}"]
        home_path = stage_module(pipe.stage(name)).replace(".", "/") + ".py"
        assert home_path in entry["modules"], (name, home_path)
        assert trainer_path not in entry["modules"], (name, trainer_path)


def test_a_data_members_change_moves_only_the_declaring_stages_token(tmp_path, monkeypatch):
    """S5.9 (adopt destinations): a change to a stage's DECLARED data file moves that stage's token and no
    other. Behavioural (not a membership tautology): a synthetic lock where stage 'a' declares a weight file
    and 'b' does not; rewriting the weight moves only 'a'. This gates the data-membership MECHANISM without a
    stage over-declaring a file it never opens (IMPL-20)."""
    from scripts import _stages as s

    (tmp_path / "w" / "default").mkdir(parents=True)
    weight = tmp_path / "w" / "default" / "model.json"
    weight.write_text("1")
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "a.py").write_text("A = 1\n")
    (tmp_path / "uv.lock").write_text("l\n")
    lock = {
        "t/a": {"modules": ["m/a.py"], "data": ["w/default/model.json"]},
        "t/b": {"modules": ["m/a.py"], "data": []},
    }
    monkeypatch.setattr(closures, "load_lock", lambda: lock)
    sa = s.Stage(name="a", units=lambda c: ["x"], work=lambda k, c: 1)
    sb = s.Stage(name="b", units=lambda c: ["x"], work=lambda k, c: 1)
    p = s.Pipeline(name="t", module="m", stages=(sa, sb))
    kw = dict(root=tmp_path, env_values={}, image="i", python="3.12")
    before = {st.name: s.stage_code_identity(p, st, **kw) for st in p.stages}
    weight.write_text("2")  # the adopt writes new weights into the declared prefix
    # a real adopt changes the file's mtime; bump it so `_closure_file_bytes` (cached by path+mtime+size) rereads
    # -- a same-size rewrite within one filesystem tick would otherwise keep the cached bytes (CCTP-IMPL-28).
    import os

    bumped = weight.stat().st_mtime + 10
    os.utime(weight, (bumped, bumped))
    after = {st.name: s.stage_code_identity(p, st, **kw) for st in p.stages}
    assert before["a"] != after["a"] and before["b"] == after["b"]
