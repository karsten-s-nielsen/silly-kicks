"""S5.9 runtime checks, in a FRESH interpreter (the worker), fail closed."""

from __future__ import annotations

import pathlib
import subprocess
import sys
import textwrap

_HARNESS = textwrap.dedent(
    """
    import sys, pathlib
    from scripts.hf_jobs.stage_checks import install_stage_checks, check_modules, ClosureViolationError
    entry = {{"modules": {mods!r}, "data": {data!r}}}
    install_stage_checks("t/a", root=pathlib.Path.cwd(), entry=entry, tracked={tracked!r})
    try:
        {body}
        check_modules("t/a")
        print("PASS")
    except ClosureViolationError as exc:
        print("FAIL", exc)
    """
)


def _run(body, mods=("scripts/_stages.py",), data=(), tracked=None, extra_env=None):
    import os

    code = _HARNESS.format(mods=list(mods), data=list(data), tracked=tracked, body=body)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", **(extra_env or {})}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, env=env)  # noqa: S603
    return (out.stdout or "") + (out.stderr or "")


def test_a_dynamic_import_outside_the_closure_fails():
    assert "FAIL" in _run("import importlib; importlib.import_module('scripts._toml')")


def test_a_lazy_import_of_a_closure_module_passes():
    assert "PASS" in _run("import scripts._toml", mods=("scripts/_stages.py", "scripts/_toml.py"))


def test_an_undeclared_tracked_data_file_fails_and_a_member_passes():
    tracked = ["docs/research/skillcorner_s1_event_validity/verdicts.json", "uv.lock"]
    body = "open('docs/research/skillcorner_s1_event_validity/verdicts.json','rb').read()"
    assert "FAIL" in _run(body, tracked=tracked)
    assert "PASS" in _run(body, data=tracked[:1], tracked=tracked)
    assert "PASS" in _run("open('uv.lock','rb').read()", tracked=tracked)


def test_a_third_party_read_in_the_venv_passes():
    """The local and CI venv is `<repo>/.venv`, under the repo root: neither check may flag it (r1 B PLAN-04)."""
    assert "PASS" in _run("import json, pandas; open(pandas.__file__,'rb').read()", tracked=["uv.lock"])


def test_a_module_preloaded_outside_the_closure_fails_at_the_first_check():
    import os

    code = "import scripts._toml\n" + _HARNESS.format(mods=["scripts/_stages.py"], data=[], tracked=None, body="pass")
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
    )
    assert "FAIL" in out.stdout or "ClosureViolationError" in out.stderr


def test_a_sourceless_pyc_of_a_non_closure_module_fails():
    import os

    code = textwrap.dedent(
        """
        import importlib.machinery, importlib.util, py_compile, pathlib, sys
        from scripts.hf_jobs.stage_checks import install_stage_checks, check_modules, ClosureViolationError
        entry = {"modules": ["scripts/_stages.py"], "data": []}
        install_stage_checks("t/a", root=pathlib.Path.cwd(), entry=entry, tracked=None)
        try:
            pyc = py_compile.compile("scripts/_toml.py", cfile="scripts/__pycache__/_sk_probe_toml.pyc", doraise=True)
            loader = importlib.machinery.SourcelessFileLoader("_sk_probe_toml", pyc)
            mod = importlib.util.module_from_spec(importlib.util.spec_from_loader("_sk_probe_toml", loader))
            loader.exec_module(mod); sys.modules["_sk_probe_toml"] = mod
            check_modules("t/a")
            print("PASS")
        except ClosureViolationError as exc:
            print("FAIL", exc)
        """
    )
    try:
        out = subprocess.run(  # noqa: S603
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "HF_HUB_OFFLINE": "1"},
        )
        assert "FAIL" in (out.stdout + out.stderr), out.stdout + out.stderr
    finally:
        pathlib.Path("scripts/__pycache__/_sk_probe_toml.pyc").unlink(missing_ok=True)


def test_a_toy_corpus_file_under_tests_passes_as_a_dataset_root():
    """A REAL tracked file (r2 B R2-05): it passes under its dataset root and fails without the root."""
    root = "tests/scripts/_fixtures"
    path = f"{root}/d2_calibration_golden.json"  # tracked
    body = f"open('{path}','rb').read()"
    assert "PASS" in _run(body, tracked=[path, "uv.lock"], extra_env={"SILLY_KICKS_DATA_ROOT": root})
    assert "FAIL" in _run(body, tracked=[path, "uv.lock"])


def test_an_undeclared_hub_download_fails_offline(tmp_path):
    import os

    code = (
        "from huggingface_hub import snapshot_download\n"
        "try:\n    snapshot_download('silly-kicks/ghost-gk', revision='main')\n    print('PASS')\n"
        "except Exception as exc:\n    print('FAIL', type(exc).__name__)"
    )
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_HOME": str(tmp_path)}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, env=env)  # noqa: S603
    assert out.stdout.startswith("FAIL")


def test_declared_hub_files_are_prefetched_and_key_the_token(monkeypatch):
    from scripts import _stages as s

    calls = []
    st = s.Stage(
        name="a",
        units=lambda c: ["x"],
        work=lambda k, c: 1,
        hub_files=(("silly-kicks/ghost-gk", "abc123", "weights/model.json"),),
    )
    s.prefetch_hub_files([st], fetch=lambda repo, rev, path: calls.append((repo, rev, path)))
    assert calls == [("silly-kicks/ghost-gk", "abc123", "weights/model.json")]
    p = s.Pipeline(name="t", module="m", stages=(st,))
    st2 = s.Stage(
        name="a",
        units=lambda c: ["x"],
        work=lambda k, c: 1,
        hub_files=(("silly-kicks/ghost-gk", "def456", "weights/model.json"),),
    )
    p2 = s.Pipeline(name="t", module="m", stages=(st2,))
    kw = dict(root=pathlib.Path("."), env_values={}, image="i", python="3.12")
    import scripts.hf_jobs.closures as cl

    monkeypatch.setattr(cl, "load_lock", lambda: {"t/a": {"modules": [], "data": []}})  # never leaks (r2 B R2-06)
    assert s.stage_code_identity(p, st, **kw) != s.stage_code_identity(p2, st2, **kw)


def test_a_stage_unit_that_downloads_undeclared_hub_files_fails_under_the_checks(tmp_path):
    """r2 B R2-09: not only the library -- a UNIT under `install_stage_checks` fails, and records the failure."""
    code = textwrap.dedent(
        f"""
        import pathlib
        from scripts import _stages as s
        from scripts.hf_jobs.stage_checks import install_stage_checks
        def work(k, ctx):
            from huggingface_hub import snapshot_download
            return {{"p": snapshot_download("silly-kicks/ghost-gk", revision="main")}}
        st = s.Stage(name="a", units=lambda c: ["only"], work=work)
        pipe = s.Pipeline(name="t", module="m", stages=(st,))
        prov = {{"commit": "c" * 40, "tree_state": "dirty", "code_identity": "dirty-x"}}
        ctx = s.make_context(pipe, {{}}, prov, store=s.LocalStore(pathlib.Path({str(tmp_path)!r})), data_root=None)
        from scripts.hf_jobs.closures import module_closure  # the REAL runner closure (r3 B R3-01)
        entry = {{"modules": list(module_closure("scripts._stages")), "data": []}}
        install_stage_checks("t/a", root=pathlib.Path.cwd(), entry=entry)
        print("OUTCOME", s.run_unit(ctx, st, "only", worker="w")[0])
        """
    )
    import os

    env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_HOME": str(tmp_path / "hf")}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180, env=env)  # noqa: S603
    assert "OUTCOME failed" in out.stdout, out.stdout + out.stderr


def test_a_stage_unit_that_imports_a_non_closure_module_fails_and_writes_no_result(tmp_path):
    """r2 B IMPL-16: a UNIT that imports a repo module OUTSIDE its closure fails -- the import audit fires DURING
    run_unit (before any result is written), so the unit is `failed` and no result file lands."""
    import os

    code = textwrap.dedent(
        f"""
        import os, pathlib
        from scripts import _stages as s
        from scripts.hf_jobs.closures import module_closure
        from scripts.hf_jobs.stage_checks import check_modules, install_stage_checks
        cl = set(module_closure("scripts._stages"))
        cands = ("scripts/build_corner_routines.py", "scripts/_loader_pining.py", "scripts/_loader_databricks.py")
        outside = next(m[:-3].replace("/", ".") for m in cands if m not in cl)
        def work(k, ctx):
            import importlib
            return {{"loaded": importlib.import_module(outside) is not None}}
        st = s.Stage(name="a", units=lambda c: ["only"], work=work)
        pipe = s.Pipeline(name="t", module="m", stages=(st,))
        prov = {{"commit": "c" * 40, "tree_state": "dirty", "code_identity": "dirty-x"}}
        store_root = pathlib.Path({str(tmp_path)!r})
        ctx = s.make_context(pipe, {{}}, prov, store=s.LocalStore(store_root), data_root=None)
        install_stage_checks("t/a", root=pathlib.Path.cwd(), entry={{"modules": sorted(cl), "data": []}})
        outcome = s.run_unit(ctx, st, "only", worker="w", post_work=lambda: check_modules("t/a"))[0]
        gen = store_root / ctx.generation("a")
        wrote_result = gen.is_dir() and any(
            p.name in ("only.parquet", "only.json", "only.fileset.json") for p in gen.iterdir()
        )
        print("OUTCOME", outcome, "RESULT", wrote_result)
        """
    )
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, env=env)  # noqa: S603
    assert "OUTCOME failed RESULT False" in out.stdout, out.stdout + out.stderr


def test_the_worker_and_the_reduce_load_a_reusable_stage_without_its_trainer(tmp_path):
    import importlib
    import json
    import os

    from scripts._stages import LocalStore, context_to_json, make_context, sorted_units
    from scripts._toy_corpora import TOY_PARAMS, build_toy_for
    from tests.scripts._stage_harness import PROV, params_from_args

    mod = "scripts.train_ghost_outfield"
    pipe = importlib.import_module(mod).PIPELINE
    data = build_toy_for(mod, tmp_path / "d")
    ctx = make_context(
        pipe, params_from_args(pipe, list(TOY_PARAMS[mod])), PROV, store=LocalStore(tmp_path / "s"), data_root=data
    )
    keys = sorted_units(ctx, pipe.stage("extract"))
    (tmp_path / "ctx.json").write_text(context_to_json(ctx), encoding="utf-8")
    (tmp_path / "pending.json").write_text(json.dumps(keys), encoding="utf-8")
    for d in ("claims", "outcomes"):
        (tmp_path / d).mkdir()
    # the executor starts workers with HF_HUB_OFFLINE=1 (Task 14); a test that starts them directly sets it too
    env = {**os.environ, "SILLY_KICKS_DEBUG_MODULES": "1", "SILLY_KICKS_STAGE_CHECKS": "1", "HF_HUB_OFFLINE": "1"}
    base = [sys.executable, "-m", "scripts.hf_jobs.executor"]
    worker = subprocess.run(  # noqa: S603
        [
            *base,
            "worker",
            "--ctx",
            str(tmp_path / "ctx.json"),
            "--stage",
            "extract",
            "--pending",
            str(tmp_path / "pending.json"),
            "--claims",
            str(tmp_path / "claims"),
            "--outcomes",
            str(tmp_path / "outcomes"),
            "--worker-id",
            "w0",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    reduce = subprocess.run(  # noqa: S603
        [*base, "reduce", "--ctx", str(tmp_path / "ctx.json"), "--stage", "extract"],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    for out in (worker, reduce):
        assert out.returncode == 0, out.stderr[-2000:]
        mods = next(line for line in out.stdout.splitlines() if line.startswith("MODULES:"))
        assert "scripts.train_ghost_outfield" not in mods.split()


def test_an_inner_task_that_opens_an_undeclared_file_fails():
    import os

    code = textwrap.dedent(
        """
        from scripts.hf_jobs.closures import module_closure
        from scripts.hf_jobs.executor import InnerPool
        from scripts.hf_jobs.stage_checks import ClosureViolationError
        from tests.scripts._toy_pipeline import open_undeclared
        # The task's whole import closure is in the entry (so the import audit allows unpickling it and its
        # imports); the UNDECLARED verdicts.json open inside the task is what must fail closed -- cleanly, as a
        # task exception, never a BrokenProcessPool (the import audit must not fire during deserialization).
        mods = sorted(set(module_closure("scripts._stages")) | {"tests/scripts/_toy_pipeline.py"})
        entry = {"modules": mods, "data": []}
        pool = InnerPool(2, stage_checks=("t/a", entry))
        try:
            pool.map_keyed(open_undeclared, {"k1": (0,), "k2": (1,)})
            print("PASS")
        except ClosureViolationError as exc:
            print("FAIL", exc)
        finally:
            pool.shutdown()
        """
    )
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}  # the pool initializer refuses a process without it (r2 B R2-02)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, env=env)  # noqa: S603
    assert "FAIL" in out.stdout and "BrokenProcessPool" not in out.stdout + out.stderr, out.stdout + out.stderr


def test_an_inner_task_that_loads_a_non_closure_module_fails():
    import os

    code = textwrap.dedent(
        """
        from scripts.hf_jobs.closures import module_closure
        from scripts.hf_jobs.executor import InnerPool
        from scripts.hf_jobs.stage_checks import ClosureViolationError
        from tests.scripts._toy_pipeline import load_undeclared
        # The task LOADS scripts/_cache.py by exec_module (NO import event, so the open/import hooks miss it);
        # only the after-task module check RUN IN THE CHILD catches it (CCTP-IMPL-16). _cache is outside the
        # runner closure, and the parent's own module check cannot see the child's load.
        mods = sorted(set(module_closure("scripts._stages")) | {"tests/scripts/_toy_pipeline.py"})
        entry = {"modules": mods, "data": []}
        pool = InnerPool(2, stage_checks=("t/a", entry))
        try:
            pool.map_keyed(load_undeclared, {"k1": (0,), "k2": (1,)})
            print("PASS")
        except ClosureViolationError as exc:
            print("FAIL", exc)
        finally:
            pool.shutdown()
        """
    )
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}  # the pool initializer refuses a process without it (r2 B R2-02)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, env=env)  # noqa: S603
    assert "FAIL" in out.stdout and "BrokenProcessPool" not in out.stdout + out.stderr, out.stdout + out.stderr
