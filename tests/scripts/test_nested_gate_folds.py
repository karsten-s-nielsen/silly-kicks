"""Nested gate-fold studies (ADR-131; spec 4.1 and 5, Tests 12-14).

Each cv arm gets one study per gate fold, ``{arm}_gf{k}``, tuned on the arm's rows OUTSIDE gate fold k.
"""

from __future__ import annotations

import importlib
import inspect
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts._study_shared import study_tags

TRAINERS = ["scripts.train_xshot_occurrence", "scripts.train_xcross_attempt"]


def _gate_plan(mod):
    if mod.__name__.endswith("xshot_occurrence"):
        return mod._cv_metrics_plan
    from silly_kicks.tracking._xcross_eval import _cv_score_plan

    return _cv_score_plan


def _inputs(n_matches: int = 12, rows: int = 30, *, paired: bool) -> SimpleNamespace:
    rng = np.random.default_rng(0)
    groups = np.repeat([f"m{i:02d}" for i in range(n_matches)], rows).astype(str)
    y = (rng.random(len(groups)) < 0.2).astype(int)
    X = pd.DataFrame({"a": rng.normal(size=len(groups))})
    match = np.repeat(np.arange(n_matches), rows)
    is_public = match < n_matches // 2
    arms = {}
    if paired:
        arms = {"public": is_public, "sc_extended": np.ones(len(y), bool)}  # ADR-132: two arms
    return SimpleNamespace(X=X, y=y, groups=groups, is_public=is_public, arms=arms)


def test_study_tags_appends_gate_fold_tags():
    arms = ["public", "sc_extended"]
    cv_arms = {"public": 3, "sc_extended": 5}
    old = [f"{a}_f{f}" for f in range(3) for a in sorted(arms)] + sorted(arms)
    gf = [f"{a}_gf{k}" for a in sorted(cv_arms) for k in range(cv_arms[a])]
    assert study_tags(arms, 3, cv_arms, run_paired=True) == old + gf
    assert study_tags([], 0, {"single": 5}, run_paired=False) == ["single"] + [f"single_gf{k}" for k in range(5)]


@pytest.mark.parametrize("module", TRAINERS)
def test_prep_builds_its_tags_with_study_tags(module):
    """Both trainers' prep stage builds `study_tags` through the shared helper (one source for the tag set)."""
    src = inspect.getsource(importlib.import_module(module)._prep_work)
    assert "study_tags(" in src


@pytest.mark.parametrize("module", TRAINERS)
@pytest.mark.parametrize("arm", ["single", "public", "sc_extended"])
def test_gf_rows_exclude_their_fold(module, arm):
    mod = importlib.import_module(module)
    inp = _inputs(paired=arm != "single")
    arm_rows = np.arange(len(inp.y)) if arm == "single" else np.flatnonzero(inp.arms[arm])
    plan = _gate_plan(mod)(inp.y[arm_rows], inp.groups[arm_rows])
    for k, (_i, _tr, te) in enumerate(plan):
        rows = mod._study_rows(inp, f"{arm}_gf{k}")
        held = arm_rows[te]
        assert np.all(np.diff(rows) > 0)  # ascending
        assert np.isin(rows, arm_rows).all()
        assert not np.isin(rows, held).any()  # no row of gate fold k
        assert np.array_equal(np.sort(np.r_[rows, held]), arm_rows)


@pytest.mark.parametrize("module", TRAINERS)
def test_gf_tag_never_parsed_as_public_fold_tag(module):
    mod = importlib.import_module(module)
    inp = _inputs(paired=True)
    _, _, folds = mod._public_folds(inp.X, inp.y, inp.groups, inp.is_public)
    for name in ("public", "sc_extended"):  # today's rule, unchanged
        assert np.array_equal(mod._study_rows(inp, f"{name}_f1"), np.flatnonzero(inp.arms[name] & folds[1][1]))
    # On the sc_extended arm, the public fold (drops fold-1 PUBLIC games) and the gate fold (drops gate fold 1 of
    # that arm's own plan) are different row sets, so each tag must resolve through its own branch. (On the public arm
    # both plans are the same splitter on the same rows, so there the two sets coincide by construction.)
    assert not np.array_equal(mod._study_rows(inp, "sc_extended_gf1"), mod._study_rows(inp, "sc_extended_f1"))
    assert not np.array_equal(mod._study_rows(inp, "sc_extended_gf0"), mod._study_rows(inp, "sc_extended_gf1"))


# --- Task 4: the cv units fit at their own fold's nested params; the reduce adds `skill` --------------------------


def _cv_world(module, monkeypatch, fold_params=None):
    """A local single-arm cv world: the trainer's `_inputs` patched to local inputs, a fake ctx serving the study
    stage's `single_gf{k}` params."""
    mod = importlib.import_module(module)
    rng = np.random.default_rng(1)
    groups = np.repeat([f"m{i:02d}" for i in range(10)], 40).astype(str)
    X = pd.DataFrame(rng.normal(size=(len(groups), 3)), columns=list("abc"))
    y = (X["a"] + rng.normal(scale=0.7, size=len(groups)) > 1.0).astype(int).to_numpy()
    inp = SimpleNamespace(
        X=X,
        y=y,
        groups=groups,
        # ADR-132: the row-aligned converter provider (bare game-id groups; IDSSE rows carry `sportec`)
        providers=np.where(np.arange(len(groups)) // 40 % 2 == 0, "sportec", "skillcorner"),
        arms={},
        is_public=np.ones(len(y), bool),
        config={"negative_subsample": None, "seed": 42},
    )
    monkeypatch.setattr(mod, "_inputs", lambda ctx: (None, inp))
    n = len(_gate_plan(mod)(y, groups))
    fold_params = fold_params or [{"n_estimators": 20 + 10 * k, "max_depth": 2} for k in range(n)]
    studies = {f"single_gf{k}": {"params": fold_params[k]} for k in range(n)}
    ctx = SimpleNamespace(upstream=lambda name: studies if name == "study" else {"cv_arms": {"single": n}})
    keys = [f"single__{k:04d}" for k in range(n)]
    return mod, inp, ctx, keys, fold_params


def _direct_fold(mod, inp, k, params):
    """The fold fitted directly at ``params``: ``(metrics_tuple, p, train_rate)`` (never a skipped fold here)."""
    plan = _gate_plan(mod)(inp.y, inp.groups)
    _i, tr, te = plan[k]
    if mod.__name__.endswith("xshot_occurrence"):
        out = mod._cv_metrics_fold(inp.X, inp.y, k, tr, te, params, None, 42, return_oof=True)
    else:
        from silly_kicks.tracking._xcross_eval import _cv_score_fold

        out = _cv_score_fold(inp.X, inp.y, k, tr, te, params, seed=42, negative_subsample=None, return_oof=True)
    assert out is not None
    return out


@pytest.mark.parametrize("module", TRAINERS)
def test_cv_unit_uses_its_fold_study(module, monkeypatch):
    from scripts._study_shared import read_cv_unit

    mod, inp, ctx, keys, fold_params = _cv_world(
        module,
        monkeypatch,
        fold_params=[{"n_estimators": 5, "max_depth": 2}, {"n_estimators": 60, "max_depth": 4}]
        + [{"n_estimators": 20, "max_depth": 2}] * 3,
    )
    for k in (0, 1):
        unit = read_cv_unit(mod._cv_work(keys[k], ctx))
        _res, p_direct, _rate = _direct_fold(mod, inp, k, fold_params[k])
        assert np.array_equal(unit["p"], p_direct)  # fitted at ITS fold's params
        _res, p_other, _rate = _direct_fold(mod, inp, k, fold_params[1 - k])
        assert not np.array_equal(unit["p"], p_other)


@pytest.mark.parametrize("module", TRAINERS)
def test_reduce_order_independent_and_equals_cv_metrics(module, monkeypatch):
    mod, inp, ctx, keys, fold_params = _cv_world(module, monkeypatch)
    results = {k: mod._cv_work(k, ctx) for k in keys}
    out = mod._cv_reduce(results, ctx)["single"]
    shuffled = {k: results[k] for k in reversed(keys)}
    assert mod._cv_reduce(shuffled, ctx)["single"] == out
    assert out["skill"]["nested"] is True
    assert out["fold_params"] == fold_params
    assert out == mod._cv_metrics(
        inp.X, inp.y, inp.groups, fold_params, providers=inp.providers
    )  # staged == composition


@pytest.mark.parametrize("module", TRAINERS)
def test_reduce_with_a_skipped_fold(module, monkeypatch):
    from scripts._study_shared import cv_unit, cv_unit_fileset

    mod, inp, ctx, keys, fold_params = _cv_world(module, monkeypatch)
    results = {k: mod._cv_work(k, ctx) for k in keys}
    full = mod._cv_reduce(results, ctx)["single"]
    results[keys[0]] = cv_unit_fileset(
        cv_unit(None, float(inp.y.mean()), None, None, None, None, None)
    )  # a skipped fold
    out = mod._cv_reduce(results, ctx)["single"]
    plan = _gate_plan(mod)(inp.y, inp.groups)
    assert out["n_usable_folds"] == full["n_usable_folds"] - 1
    assert out["skill"]["n_oof_rows"] == len(inp.y) - len(plan[0][2])
    assert out["fold_params"] == fold_params  # every gate fold keeps its entry


# --- Task 5: the trainers gate on skill_gates; a FAILED payload carries the skill dict -----------------------------


@pytest.mark.parametrize("module", TRAINERS)
def test_failed_payload_carries_skill(module, monkeypatch):
    """A failed gate writes `only.metrics_FAILED.json` with the 7 ADR-131 keys and the candidate's full `skill`
    dict, so the refit session can apply the spec 6.1 finite-values rule from the payload alone."""
    import json

    from silly_kicks.tracking._artifact_integrity import skill_gates

    mod = importlib.import_module(module)
    skill = {
        "brier_skill": 0.009,
        "log_loss_skill": 0.03,
        "pr_auc_lift": 0.02,
        "bootstrap": {"brier_gain_lb": -1.4e-5, "log_loss_gain_lb": 0.0019, "pr_auc_lift_lb": 0.011},
    }
    metrics = {"n_usable_folds": 5, "skill": skill}
    sel = {"shipped": "public", "candidates": {"public": {"params": {}, "metrics": metrics, "providers": []}}}
    written = {}
    store = SimpleNamespace(write_bytes=lambda path, data: written.__setitem__(path, data))
    ctx = SimpleNamespace(upstream=lambda name: sel, store=store, generation=lambda stage: f"gen/{stage}")
    monkeypatch.setattr(mod, "_inputs", lambda ctx: (None, SimpleNamespace(config={})))
    with pytest.raises(mod.AcceptanceGatesFailedError):
        mod._fit_work("only", ctx)
    (payload,) = [json.loads(v) for k, v in written.items() if k.endswith("only.metrics_FAILED.json")]
    assert payload["acceptance"] == skill_gates(metrics)
    assert list(payload["acceptance"]) == [
        "enough_usable_folds",
        "brier_skill_gt_0",
        "log_loss_skill_gt_0",
        "pr_auc_lift_gt_0",
        "brier_gain_lb95_gt_0",
        "log_loss_gain_lb95_gt_0",
        "pr_auc_lift_lb95_gt_0",
        "provider_brier_skill_gt_0",
        "provider_log_loss_skill_gt_0",
    ]
    assert payload["acceptance"]["brier_gain_lb95_gt_0"] is False
    assert payload["candidates"]["public"]["metrics"]["skill"] == skill


@pytest.mark.parametrize("module", TRAINERS)
def test_the_old_gate_function_is_gone(module):
    assert not hasattr(importlib.import_module(module), "_gates")


def test_an_arm_name_containing_the_gate_fold_suffix_is_refused():
    """FSG-IB03: an arm named like ``x_gf1`` would collide with arm ``x``'s gate-fold study tag; refuse it loud."""
    with pytest.raises(ValueError, match="_gf"):
        study_tags(["x", "x_gf1"], 0, {"x": 2, "x_gf1": 2}, run_paired=True)
    with pytest.raises(ValueError, match="_gf"):
        study_tags([], 0, {"single_gf0": 2}, run_paired=False)


@pytest.mark.parametrize("module", TRAINERS)
def test_a_real_one_class_training_fold_is_skipped(module, monkeypatch):
    """FSG-IB05: one match holds every positive, so the gate fold that holds it trains on negatives only. The fold
    function skips it for real (no injected payload); the reduce drops its rows and keeps every fold_params entry."""
    mod, inp, ctx, keys, fold_params = _cv_world(module, monkeypatch)
    y = np.zeros(len(inp.y), dtype=int)
    hot = np.flatnonzero(inp.groups == "m03")
    y[hot[::4]] = 1  # positives in one match only
    inp.y = y
    plan = _gate_plan(mod)(inp.y, inp.groups)
    skipped = [k for k, (_i, tr, _te) in enumerate(plan) if len(np.unique(inp.y[tr])) < 2]
    assert len(skipped) == 1  # precondition: exactly one real one-class training fold
    out = mod._cv_reduce({k: mod._cv_work(k, ctx) for k in keys}, ctx)["single"]
    assert out["n_usable_folds"] == len(plan) - 1
    assert out["skill"]["n_oof_rows"] == len(inp.y) - len(plan[skipped[0]][2])
    assert out["fold_params"] == fold_params


# --- Implementation-review findings (A: FSG-IMPL-01/02/04/05) --------------------------------------------------------


@pytest.mark.parametrize("module", TRAINERS)
def test_reduce_skill_equals_an_independent_reference(module, monkeypatch):
    """FSG-IMPL-02/04: the cv `skill` uses each row's TRAINING-FOLD TRUE rate as the reference and pools the rows in
    ascending order. Built here independently of cv_skill (own rates, own ordering) and compared exactly."""
    from silly_kicks.tracking._artifact_integrity import forecast_skill, provider_skill

    mod, inp, ctx, keys, fold_params = _cv_world(module, monkeypatch)
    out = mod._cv_reduce({k: mod._cv_work(k, ctx) for k in keys}, ctx)["single"]
    plan = _gate_plan(mod)(inp.y, inp.groups)
    n = len(inp.y)
    p, r, fold = np.full(n, np.nan), np.full(n, np.nan), np.full(n, -1)
    for k, (_i, tr, te) in enumerate(plan):
        _res, p[te], _rate = _direct_fold(mod, inp, k, fold_params[k])
        r[te] = inp.y[tr].mean()  # the reference: the training folds' TRUE rate (never the arm's pooled rate)
        fold[te] = k
    want = forecast_skill(inp.y, p, r, fold, inp.groups.astype(str), nested=True)
    # ADR-132: each row's OWN provider's training-fold rate, keyed by identity (sportec -> idsse)
    ident = np.where(inp.providers == "sportec", "idsse", inp.providers)
    r_prov = np.full(n, np.nan)
    for _i, tr, te in plan:
        for prov in ("idsse", "skillcorner"):
            sel = te[ident[te] == prov]
            r_prov[sel] = inp.y[tr][ident[tr] == prov].mean()
    want["by_provider"] = provider_skill(inp.y, p, r_prov, ident, inp.groups.astype(str), fold)
    assert out["skill"] == want
    assert set(want["by_provider"]) == {"idsse", "skillcorner"}
    assert (
        out["skill"]["brier_ref"]
        != forecast_skill(  # non-vacuity: the pooled-rate reference differs
            inp.y, p, np.full(n, inp.y.mean()), fold, inp.groups.astype(str), nested=True
        )["brier_ref"]
    )


@pytest.mark.parametrize("module", TRAINERS)
def test_shipped_fit_uses_the_full_rows_study(module, monkeypatch):
    """FSG-IMPL-01 (spec Test 13, second half): the select stage ships the arm's FULL-ROWS study params ("single" on
    the non-paired path), never a gate-fold study's, while the candidate's metrics carry the nested fold params."""
    mod = importlib.import_module(module)
    full = {"n_estimators": 111, "max_depth": 3}
    gf = {"n_estimators": 7, "max_depth": 2}
    metrics = {"n_usable_folds": 5, "fold_params": [gf] * 5}
    upstream = {
        "study": {"single": {"params": full}, **{f"single_gf{k}": {"params": gf} for k in range(5)}},
        "cv": {"single": metrics},
    }
    inp = SimpleNamespace(
        config={"run_paired": False, "ship_order": ["public"], "expect_variant": None},
        providers=np.array(["skillcorner"] * 4),
    )
    monkeypatch.setattr(mod, "_inputs", lambda ctx: (None, inp))
    sel = mod._select_work("only", SimpleNamespace(upstream=lambda name: upstream[name]))
    shipped = sel["shipped"]
    assert sel["candidates"][shipped]["params"] == full
    assert sel["candidates"][shipped]["params"] != gf
    assert sel["candidates"][shipped]["metrics"]["fold_params"] == [gf] * 5


@pytest.mark.parametrize("module", TRAINERS)
def test_a_per_fold_params_list_of_the_wrong_length_is_refused(module, monkeypatch):
    """FSG-IMPL-05: a per-fold params sequence must have one entry per gate fold."""
    mod, inp, _ctx, _keys, fold_params = _cv_world(module, monkeypatch)
    with pytest.raises(ValueError, match="per-fold params"):
        mod._cv_metrics(inp.X, inp.y, inp.groups, fold_params[:-1])
    with pytest.raises(ValueError, match="per-fold params"):
        mod._cv_metrics(inp.X, inp.y, inp.groups, [*fold_params, fold_params[0]])


@pytest.mark.parametrize("module", TRAINERS)
def test_paired_shipped_fit_uses_the_shipped_arms_full_rows_study(module, monkeypatch):
    """FSG-IMPL-01 (paired branch): when the two-arm rule (ADR-132) ships an arm, the shipped params are that arm's
    FULL-ROWS study, never one of its gate-fold studies `{arm}_gf{k}`. The paired deltas and the ship verdict are
    stubbed; what is pinned is where the shipped params come from."""
    import scripts._paired as paired_mod

    mod = importlib.import_module(module)
    arms = ["public", "sc_extended"]
    full_rows = {a: {"n_estimators": 100 + i, "max_depth": 3} for i, a in enumerate(arms)}
    gf = {"n_estimators": 7, "max_depth": 2}
    studies = {a: {"params": full_rows[a]} for a in arms}
    studies.update({f"{a}_gf{k}": {"params": gf} for a in arms for k in range(5)})
    cv = {a: {"n_usable_folds": 5, "fold_params": [gf] * 5} for a in arms}
    n = 8
    masks = {"public": np.array([True] * 4 + [False] * 4), "sc_extended": np.ones(n, bool)}
    inp = SimpleNamespace(
        X=None,
        y=np.zeros(n),
        groups=np.array([f"m{i}" for i in range(n)]),
        is_public=masks["public"],
        match_ids=np.arange(n),
        arms=masks,
        providers=np.array(["skillcorner"] * n),
        config={
            "run_paired": True,
            "n_trials": 1,
            "objective_inputs": {},
            "run_prov": {},
            "negative_subsample": None,
            "seed": 42,
            "ship_variant": None,
            "expect_variant": None,
            "ship_order": ["public", "sc_extended"],
        },
    )
    monkeypatch.setattr(mod, "_inputs", lambda ctx: (None, inp))
    monkeypatch.setattr(
        mod, "_paired_data_effect", lambda *a, **k: {x: {"nested": [0.0], "shared_params": [0.0]} for x in arms}
    )
    monkeypatch.setattr(paired_mod, "two_arm_ship", lambda deltas: ("sc_extended", "stubbed verdict"))
    up = {"study": studies, "cv": cv}
    sel = mod._select_work("only", SimpleNamespace(upstream=lambda name: up[name]))
    assert sel["shipped"] == "sc_extended"
    assert sel["candidates"]["sc_extended"]["params"] == full_rows["sc_extended"]
    assert sel["candidates"]["sc_extended"]["params"] != gf
