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
