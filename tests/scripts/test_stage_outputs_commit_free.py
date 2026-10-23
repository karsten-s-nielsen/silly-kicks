"""S5.10: no stage module reads ctx.prov; no toy output holds a run-dependent provenance key or SHA256SUMS."""

from __future__ import annotations

import ast
import importlib
import json

import pytest

from scripts.hf_jobs.identity import PROVENANCE_KEYS

RUN_DEPENDENT = tuple(k for k in PROVENANCE_KEYS if k != "generation")


def _stage_modules():
    from scripts.hf_jobs.closures import _population, stage_module

    for mod in _population():
        for st in importlib.import_module(mod).PIPELINE.stages:
            yield stage_module(st)


@pytest.mark.parametrize("module", sorted(set(_stage_modules())))
def test_no_stage_module_reads_ctx_prov(module):
    src = importlib.import_module(module).__file__
    assert src is not None
    tree = ast.parse(open(src, encoding="utf-8").read())
    # Only ``ctx.prov`` (the RUN provenance) is banned in a stage; ``ref.prov`` (a staged match's INPUT
    # provenance, a ref attribute) is a legitimate data read, not run state.
    hits = [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and n.attr == "prov" and isinstance(n.value, ast.Name) and n.value.id == "ctx"
    ]
    assert not hits, f"{module}: ctx.prov read at lines {hits}"


def test_the_ghost_gk_toy_output_is_commit_free(tmp_path):
    from scripts._stages import StageIncompleteError
    from tests.scripts._stage_harness import run_toy

    try:
        out = run_toy("scripts.train_ghost_gk", tmp_path)  # -> {filename: bytes} of the output FileSet
    except StageIncompleteError as e:
        pytest.skip(f"ghost_gk toy fit unavailable in this env (sklearn [train] guard; CI runs it): {e}")
    assert "SHA256SUMS" not in out
    for name in ("metadata.json", "metrics.json"):
        doc = json.loads(out[name])
        assert not (set(RUN_DEPENDENT) & set(doc)), name


def _one_stage_pipeline(files: dict[str, bytes]):
    from scripts._stages import FileSet, Pipeline, Stage

    st = Stage(
        name="assemble", units=lambda c: ["x"], work=lambda k, c: FileSet(dict(files)), reduce=lambda r, c: r["x"]
    )
    return Pipeline(name="toy_export", module=__name__, stages=(st,), output="assemble")


_LOCAL_PROV = {
    "commit": "c" * 40,
    "tree_state": "dirty",
    "dirty": True,
    "code_identity": "dirty-x",
    "platform": "Linux-x",
}


def test_a_local_export_is_stamped_and_loadable_while_the_store_stays_commit_free(tmp_path):
    """`--run-all --out` (owner ruling 2026-10-22): the exported BUNDLE carries THIS run's provenance and a valid
    SHA256SUMS, so `load()` accepts it; the store output it was copied from stays commit-free (S5.10)."""
    from scripts._stages import export_output
    from scripts.hf_jobs import identity
    from tests.scripts._stage_harness import run_in_process

    meta = json.dumps({"variant": "default", "reproducibility": "public"}).encode("utf-8")
    pipe = _one_stage_pipeline({"metadata.json": meta, "w.npz": b"\x00w"})
    ctx = run_in_process(pipe, [], store=tmp_path / "s", data_root=tmp_path, prov=_LOCAL_PROV)
    export_output(ctx, tmp_path / "out")
    doc = json.loads((tmp_path / "out" / "metadata.json").read_text(encoding="utf-8"))
    assert doc["training_commit"] == doc["run_commit"] == "c" * 40
    assert doc["run_tree_state"] == "dirty" and doc["run_tree_dirty"] is True  # a dirty local run says so
    assert doc["training_platform"] == "Linux-x"
    assert identity.verify_sums(tmp_path / "out")
    stored = ctx.upstream("assemble").files
    assert identity.SUMS not in stored
    assert not set(RUN_DEPENDENT) & set(json.loads(stored["metadata.json"]))


def test_a_non_bundle_export_gets_no_sums(tmp_path):
    """An output with no metadata.json / metrics.json (the census `census.json`) is exported as is."""
    from scripts._stages import export_output
    from scripts.hf_jobs import identity
    from tests.scripts._stage_harness import run_in_process

    pipe = _one_stage_pipeline({"census.json": b'{"n": 1}\n'})
    ctx = run_in_process(pipe, [], store=tmp_path / "s", data_root=tmp_path, prov=_LOCAL_PROV)
    export_output(ctx, tmp_path / "out")
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["census.json"]
    assert not (tmp_path / "out" / identity.SUMS).exists()
