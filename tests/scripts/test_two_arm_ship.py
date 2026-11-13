"""The two-arm paired path, the per-provider reporting and the size gate (corpus widening spec 5.2, 5.4, 5.6, 5.7;
ADR-132). Both trainers where it applies."""

from __future__ import annotations

import importlib
import inspect
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts._paired import is_paired, two_arm_ship
from scripts._stages import FileSet
from scripts._study_shared import ARTIFACT_SIZE_LIMIT_BYTES
from silly_kicks.tracking._artifact_integrity import skill_gates

TRAINERS = ["scripts.train_xshot_occurrence", "scripts.train_xcross_attempt"]
ARMS = ["public", "sc_extended"]
CLEARS = [0.1, 0.2, -0.01, 0.1, 0.05]
FAILS = [0.1, -0.2, -0.01, 0.1, 0.05]


def test_two_arm_ship_clears():
    shipped, why = two_arm_ship(CLEARS)
    assert shipped == "sc_extended" and why


def test_two_arm_ship_does_not_clear():
    shipped, why = two_arm_ship(FAILS)
    assert shipped == "public" and why


def test_fixed_sequence_ship_is_gone():
    import scripts._paired as paired_mod

    assert not hasattr(paired_mod, "fixed_sequence_ship")


@pytest.mark.parametrize("module", TRAINERS)
def test_run_paired_needs_public_and_sc_extended(module):
    assert is_paired(["public", "sc_extended"]) and is_paired({"sc_extended": 1, "public": 2})
    assert not is_paired(["public"]) and not is_paired(["sc_extended"]) and not is_paired([])
    src = inspect.getsource(importlib.import_module(module)._prep_work)
    assert "is_paired(" in src and '"full" in arms' not in src  # the trainers single-source the predicate


def _selection_world(module, monkeypatch, deltas, *, by_provider=None):
    mod = importlib.import_module(module)
    full_rows = {a: {"n_estimators": 100 + i, "max_depth": 3} for i, a in enumerate(ARMS)}
    gf = {"n_estimators": 7, "max_depth": 2}
    studies = {a: {"params": full_rows[a]} for a in ARMS}
    studies.update({f"{a}_gf{k}": {"params": gf} for a in ARMS for k in range(5)})
    cv = {a: {"n_usable_folds": 5, "fold_params": [gf] * 5, "arm": a} for a in ARMS}
    n = 8
    masks = {"public": np.array([True] * 4 + [False] * 4), "sc_extended": np.ones(n, bool)}
    inp = SimpleNamespace(
        X=None,
        y=np.zeros(n),
        groups=np.array([str(i) for i in range(n)]),
        is_public=masks["public"],
        match_ids=np.arange(n),
        arms=masks,
        providers=np.array(["sportec"] * 2 + ["gradientsports"] * 2 + ["skillcorner"] * 4),
        config={
            "run_paired": True,
            "n_trials": 1,
            "objective_inputs": {},
            "run_prov": {},
            "negative_subsample": None,
            "seed": 42,
            "ship_variant": None,
            "expect_variant": None,
            "ship_order": ARMS,
        },
    )
    seen = {}

    def fake_paired(*a, **k):
        seen["providers"] = k.get("providers")
        out = {x: {"nested": list(deltas), "shared_params": list(deltas)} for x in k["candidates"]}
        if by_provider is not None:
            out["sc_extended"]["by_provider"] = by_provider
        return out

    monkeypatch.setattr(mod, "_inputs", lambda ctx: (None, inp))
    monkeypatch.setattr(mod, "_paired_data_effect", fake_paired)
    up = {"study": studies, "cv": cv}
    return mod, mod._select_work("only", SimpleNamespace(upstream=lambda name: up[name])), full_rows, seen


@pytest.mark.parametrize("module", TRAINERS)
@pytest.mark.parametrize(("deltas", "want"), [(CLEARS, "sc_extended"), (FAILS, "public")])
def test_paired_select_ships_the_rule_choice(module, monkeypatch, deltas, want):
    _mod, sel, full_rows, seen = _selection_world(module, monkeypatch, deltas)
    assert sel["shipped"] == want
    assert sel["candidates"][want]["params"] == full_rows[want]  # that arm's FULL-ROWS study
    assert sel["candidates"][want]["metrics"]["arm"] == want
    assert sel["candidates"]["paired"]["shipped"] == want
    assert "full_vs_sc" not in sel["candidates"]["paired"]
    assert set(sel["candidates"]["paired"]["nested"]) == set(ARMS)
    # the paired effect receives the provider IDENTITIES (sportec -> idsse) for the per-provider deltas
    assert sorted(set(seen["providers"].tolist())) == ["gradientsports", "idsse", "skillcorner"]


@pytest.mark.parametrize("module", TRAINERS)
def test_provider_deltas_reported_not_deciding(module, monkeypatch):
    by = {"idsse": [-0.3] * 5, "gradientsports": [0.2] * 5, "skillcorner": [float("nan")] * 5}
    _m, with_by, _f, _s = _selection_world(module, monkeypatch, CLEARS, by_provider=by)
    _m, without, _f, _s = _selection_world(module, monkeypatch, CLEARS)
    assert with_by["shipped"] == without["shipped"] == "sc_extended"
    assert with_by["why"] == without["why"]
    assert with_by["candidates"]["paired"]["by_provider"] == by
    assert without["candidates"]["paired"]["by_provider"] == {}


def _prep_ctx(arms_json: str):
    def boom(name):
        raise AssertionError(f"prep read upstream {name!r} before refusing")

    params = {"arms_json": arms_json, "probe_providers": "", "probe_comparison_providers": ""}
    return SimpleNamespace(params=params, upstream=boom)


@pytest.mark.parametrize("module", TRAINERS)
def test_a_full_arm_is_refused(module):
    payload = {
        "arms": {"public": ["a"], "sc_extended": ["a", "b"], "full": ["a", "b", "c"]},
        "ship_order": ["public", "sc_extended", "full"],
    }
    with pytest.raises(ValueError, match="full"):
        importlib.import_module(module)._prep_work("only", _prep_ctx(json.dumps(payload)))


def test_xcross_ship_variant_drops_full():
    mod = importlib.import_module("scripts.train_xcross_attempt")
    (param,) = [p for p in mod.PIPELINE.params if p.name == "ship_variant"]
    assert param.choices == ("public", "sc_extended")


def test_probe_held_out():
    trainer = importlib.import_module("scripts.train_xcross_attempt")
    probe = [["gradientsports", "10502"], ["gradientsports", "10503"]]
    in_training = {"probe_matches": probe, "in_training_folds": {"10502": True, "10503": True}}
    with pytest.raises(SystemExit, match="in-sample"):
        trainer._gated_probe_matches(in_training, True)  # a probe match as a training row: the record refuses
    held = {"probe_matches": probe, "in_training_folds": {"10502": False, "10503": False}}
    assert trainer._gated_probe_matches(held, True) == probe


def _passing_skill(**provider_overrides):
    by = {
        "idsse": {"brier_skill": 0.02, "log_loss_skill": 0.05},
        "skillcorner": {"brier_skill": 0.03, "log_loss_skill": 0.06},
    }
    by.update(provider_overrides)
    bootstrap = {"brier_gain_lb": 0.001, "log_loss_gain_lb": 0.01, "pr_auc_lift_lb": 0.02}
    return {"brier_skill": 0.04, "log_loss_skill": 0.1, "pr_auc_lift": 0.05, "bootstrap": bootstrap, "by_provider": by}


class _FakeModel:
    """Stands in for the booster: `save` writes one file of `size` bytes (the size gate reads the bundle total)."""

    size = 0

    def __init__(self, params=None, feature_set=None):
        pass

    def fit(self, *a, **k):
        return self

    def save(self, path):
        path.mkdir(parents=True)
        with open(path / "booster.json", "wb") as f:
            f.truncate(type(self).size)

    @classmethod
    def load(cls, path):
        return cls()

    def predict_proba(self, rows):
        return np.zeros(len(rows))


def _fit_world(module, monkeypatch, skill, *, size=0):
    mod = importlib.import_module(module)
    n = 6
    inp = SimpleNamespace(
        X=pd.DataFrame({"a": np.arange(n, dtype=float)}),
        y=np.array([0, 1] * 3),
        arms={"public": np.ones(n, bool), "sc_extended": np.ones(n, bool)},
        config={
            "negative_subsample": None,
            "seed": 42,
            "feature_set": "faithful",
            "horizon_seconds": 1.0,
            "run_prov": {"commit": "c"},
        },
    )
    sel = {
        "shipped": "public",
        "candidates": {
            "public": {"params": {}, "metrics": {"n_usable_folds": 5, "skill": skill}, "providers": ["idsse"]},
            "sc_extended": {"params": {}, "metrics": {"n_usable_folds": 5, "skill": _passing_skill()}, "providers": []},
        },
    }
    written = {}
    store = SimpleNamespace(write_bytes=lambda path, data: written.__setitem__(path, data))
    ctx = SimpleNamespace(upstream=lambda name: sel, store=store, generation=lambda stage: f"gen/{stage}")
    monkeypatch.setattr(mod, "_inputs", lambda ctx: (None, inp))
    model_mod = (
        "silly_kicks.tracking._xshot_occurrence" if "xshot" in module else "silly_kicks.tracking._xcross_attempt"
    )
    cls = "XShotOccurrenceModel" if "xshot" in module else "XCrossAttemptModel"
    fake = type("Fake", (_FakeModel,), {"size": size})
    monkeypatch.setattr(importlib.import_module(model_mod), cls, fake)
    return mod, ctx, written


@pytest.mark.parametrize("module", TRAINERS)
def test_provider_gate_fails_the_run(module, monkeypatch):
    skill = _passing_skill(idsse={"brier_skill": -0.01, "log_loss_skill": 0.05})
    mod, ctx, written = _fit_world(module, monkeypatch, skill)
    with pytest.raises(mod.AcceptanceGatesFailedError):
        mod._fit_work("only", ctx)
    (payload,) = [json.loads(v) for k, v in written.items() if k.endswith("only.metrics_FAILED.json")]
    assert payload["acceptance"]["provider_brier_skill_gt_0"] is False
    assert payload["shipped_variant"] == "public"  # no fallback to the other arm


@pytest.mark.parametrize("module", TRAINERS)
def test_size_gate_fails_closed(module, monkeypatch):
    assert ARTIFACT_SIZE_LIMIT_BYTES == 5_000_000
    mod, ctx, written = _fit_world(module, monkeypatch, _passing_skill(), size=ARTIFACT_SIZE_LIMIT_BYTES)
    with pytest.raises(mod.AcceptanceGatesFailedError):
        mod._fit_work("only", ctx)
    (payload,) = [json.loads(v) for k, v in written.items() if k.endswith("only.metrics_FAILED.json")]
    assert payload["acceptance"]["artifact_size_lt_5mb"] is False
    assert payload["artifact_size_bytes"] == ARTIFACT_SIZE_LIMIT_BYTES

    mod, ctx, written = _fit_world(module, monkeypatch, _passing_skill(), size=ARTIFACT_SIZE_LIMIT_BYTES - 1)
    fs = mod._fit_work("only", ctx)
    assert sum(len(b) for b in fs.files.values()) == ARTIFACT_SIZE_LIMIT_BYTES - 1
    assert not written


def test_assembled_acceptance_carries_the_size_gate(monkeypatch):
    mod = importlib.import_module("scripts.train_xshot_occurrence")
    n = 4
    inp = SimpleNamespace(
        X=pd.DataFrame({"a": np.zeros(n)}),
        y=np.array([0, 1, 0, 1]),
        providers=np.array(["skillcorner"] * n),
        match_ids=np.array(["1", "1", "2", "2"]),
        config={"run_prov": {"commit": "c" * 40, "dirty": False, "tree_state": "clean"}},
    )
    skill = _passing_skill()
    sel = {
        "shipped": "public",
        "candidates": {
            "public": {"params": {}, "metrics": {"n_usable_folds": 5, "skill": skill}, "providers": ["skillcorner"]}
        },
    }
    fit = FileSet({"booster.json": b"x" * 10})
    up = {"select": sel, "fit": fit}
    monkeypatch.setattr(mod, "_inputs", lambda ctx: (None, inp))
    out = mod._assemble_work("only", SimpleNamespace(upstream=lambda name: up[name]))
    metrics = json.loads(out.files["metrics.json"])
    assert metrics["acceptance"] == {
        **skill_gates(sel["candidates"]["public"]["metrics"]),
        "artifact_size_lt_5mb": True,
    }
    assert metrics["artifact_size_bytes"] == 10
