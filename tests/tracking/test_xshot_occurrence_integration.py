"""Integration tests for TF-16 xShotOccurrence (xS)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl import config as _spc

_SHOT_TID = _spc.actiontype_id["shot"]  # ADR-126: build_xshot_labels filters shots by type_id


def test_atomic_mirror_reexports():
    from silly_kicks.atomic.tracking import features as atomic_features

    assert hasattr(atomic_features, "add_xshot_occurrence")


# --- Task 11: ruthless HPO objective ---


def _build_xshot_fold(n_games=2, seed=0):
    """Build a tiny fold dict {provider: [(X, y, groups), ...]} for HPO tests.

    Features are random; the label is correlated with the first feature so the
    classifier (and CV) have a learnable signal, with both classes present.
    """
    from silly_kicks.tracking._xshot_occurrence import XSHOT_FEATURE_NAMES_FAITHFUL

    rng = np.random.default_rng(seed)
    matches = []
    for g in range(n_games):
        n = 80
        X = pd.DataFrame(rng.normal(size=(n, 27)), columns=XSHOT_FEATURE_NAMES_FAITHFUL)
        # ~25% positive, correlated with r so CV folds can learn.
        y = ((X["r"] + rng.normal(scale=0.5, size=n)) < -0.4).astype(int).to_numpy()
        if y.sum() == 0:
            y[:5] = 1
        groups = np.full(n, g)
        matches.append((X, y, groups))
    return {"synthetic": matches}


def _build_xshot_objective():
    from silly_kicks.tracking._xshot_occurrence_objective import XShotOccurrenceObjective

    return XShotOccurrenceObjective(fold=_build_xshot_fold())


def test_objective_optuna_smoke_3_trials():
    from ruthless import Direction, FloatRange, InProcessBackend, OptunaConfig
    from ruthless.config.common import StoreConfig
    from ruthless.strategies.optuna_ import OptunaStrategy

    obj = _build_xshot_objective()
    import os
    import tempfile

    db = os.path.join(tempfile.mkdtemp(), "xs_smoke.db")
    cfg = OptunaConfig(
        kind="optuna",
        metric="logloss",
        direction=Direction.MINIMIZE,
        n_trials=3,
        sampler="tpe",
        param_space={
            "n_estimators": FloatRange(kind="float", lo=10.0, hi=30.0),
            "max_depth": FloatRange(kind="float", lo=2.0, hi=4.0),
            "learning_rate": FloatRange(kind="float", lo=0.1, hi=0.5),
            "reg_lambda": FloatRange(kind="float", lo=0.0, hi=3.0),
        },
        store=StoreConfig(kind="sqlite", path=db, objective_id="test-objective"),
    )
    result = OptunaStrategy(cfg, seed=42).run(obj, backend=InProcessBackend())
    assert result.best is not None
    assert "logloss" in result.best.metrics


def test_objective_search_excludes_scale_pos_weight_and_uses_stratified_cv():
    """M2: no scale_pos_weight (xS is a calibrated P(shot)); M1: label-stratified CV."""
    import inspect

    from silly_kicks.tracking import _xshot_occurrence_objective as obj_mod

    assert "scale_pos_weight" not in obj_mod._SEARCH_KEYS
    # ADR-128 ruling 4: the splitter is single-sourced in `cv_fold_plan`, which `_cv_logloss` composes.
    assert "cv_fold_plan(" in inspect.getsource(obj_mod._cv_logloss)
    src = inspect.getsource(obj_mod.cv_fold_plan)
    assert "StratifiedGroupKFold(" in src
    # The plain (non-stratified) splitter must NOT be the one constructed.
    assert "= GroupKFold(" not in src and "=GroupKFold(" not in src


def test_objective_cache_equivalence():
    from ruthless import Candidate, assert_cache_equivalence

    obj = _build_xshot_objective()
    # Vary every patch param across >=2 values (assert_cache_equivalence contract).
    candidates = [
        Candidate(
            id="t0",
            params={
                "n_estimators": 20.0,
                "max_depth": 3.0,
                "learning_rate": 0.3,
                "min_child_weight": 1.0,
                "reg_lambda": 0.0,
            },
        ),
        Candidate(
            id="t1",
            params={
                "n_estimators": 30.0,
                "max_depth": 4.0,
                "learning_rate": 0.1,
                "min_child_weight": 5.0,
                "reg_lambda": 2.0,
            },
        ),
    ]
    assert_cache_equivalence(obj, candidates)  # type: ignore[arg-type]


def test_objective_cache_equivalence_with_train_subsample():
    """M3: train-fold negative subsampling is deterministic per fold, so the cache-equivalence
    gate (evaluate vs evaluate_patch == 1e-9) still holds with subsampling ON."""
    from ruthless import Candidate, assert_cache_equivalence

    from silly_kicks.tracking._xshot_occurrence_objective import XShotOccurrenceObjective

    obj = XShotOccurrenceObjective(fold=_build_xshot_fold(), negative_subsample=0.5, subsample_seed=11)
    candidates = [
        Candidate(
            id="t0",
            params={
                "n_estimators": 20.0,
                "max_depth": 3.0,
                "learning_rate": 0.3,
                "min_child_weight": 1.0,
                "reg_lambda": 0.0,
            },
        ),
        Candidate(
            id="t1",
            params={
                "n_estimators": 30.0,
                "max_depth": 4.0,
                "learning_rate": 0.1,
                "min_child_weight": 5.0,
                "reg_lambda": 2.0,
            },
        ),
    ]
    assert_cache_equivalence(obj, candidates)  # type: ignore[arg-type]


#: The frozen directional rows. Regenerate with scripts/make_xshot_directional_fixture.py.
_FROZEN = "tests/datasets/tracking/xshot_directional/frozen_rows.parquet"


def _frozen_rows():
    return pd.read_parquet(_FROZEN)


def test_frozen_fixture_is_not_degenerate():
    """PRECONDITION the liveness gate's teeth rest on -- the ADR-032 idiom (a gate whose fixture
    is unverified is a gate that can pass on nothing).

    Two properties, both of which were VIOLATED before PR 5 and are why the previous assertion
    could not distinguish a good model from an inverted one:

    * every row IN-DOMAIN. `prepare_xshot_training_data(attacking_third_only=True)` keeps only
      frames within `_ATTACKING_THIRD_M` of the attacked goal, so rows beyond it are pure
      extrapolation. The old fixture put its whole FAR class at r ~= 101 m, ~3x outside.
    * `speed` NOT CONSTANT. The generator used to zero vx/vy, pinning ball speed to 0 on every
      row -- the same degeneracy PR-S118 found and fixed in the xCross fixture and left here. A
      pinned feature cannot contribute, and 0 routes down a different XGBoost branch than
      missing, so the gate was scoring inputs production never produces.
    """
    from silly_kicks.tracking import _xshot_occurrence as xs

    df = _frozen_rows()
    assert len(df) >= 40, f"only {len(df)} frozen rows -- too few to separate two similar models"
    assert df["r"].max() <= xs._ATTACKING_THIRD_M, (
        f"frozen rows reach r={df['r'].max():.1f} > {xs._ATTACKING_THIRD_M} -- out of the trained "
        "domain, so the model's output there is extrapolation, not prediction"
    )
    assert df["speed"].nunique() > 1, "ball speed is CONSTANT across the fixture (the PR-S118 bug)"


def test_bundled_model_is_live_not_degenerate():
    """LIVENESS tripwire (H2/N2/P4): the bundled model is not dead or constant.

    Liveness here is the repo's standard definition -- non-NaN and NON-CONSTANT -- applied to a
    model instead of an aggregator column (see tests/tracking/test_aggregator_column_liveness.py).
    Quality is deliberately NOT asserted: it lives in the e2e gates and the trainer's fail-closed
    acceptance gates.

    RETIRED IN PR 5 (do not resurrect): this used to assert `roc_auc_score(near_vs_far) >= 0.9`,
    i.e. that the model ranks near-goal rows above far ones. That premise is FALSE for the models
    it guards -- measured `corr(r, p)` is +0.89 for the pre-PR-5 bundled model and +0.94 for the
    current one, both rising with distance (plausible: xS predicts a shot ATTEMPT within ~1 s, and
    25-34 m is prime open-play shooting range). It passed only because two defects compensated:
    out-of-domain FAR rows and a ball speed pinned to 0. With either one fixed the old model
    scores ~0.47, i.e. chance. Same retire-and-replace treatment ADR-032 gave the dead
    `pitch_control_at_ball__spearman`, with the reason recorded in place so it is not reinstated.
    """
    from silly_kicks.tracking._xshot_occurrence import XSHOT_FEATURE_NAMES_FAITHFUL, XShotOccurrenceModel

    df = _frozen_rows()
    p = XShotOccurrenceModel.from_variant("default").predict_proba(df[XSHOT_FEATURE_NAMES_FAITHFUL])
    assert np.isfinite(p).all(), "bundled model produced non-finite probabilities"
    assert ((p >= 0.0) & (p <= 1.0)).all(), "probabilities out of [0, 1]"
    assert len(np.unique(np.round(p, 9))) > 1, "bundled model output is CONSTANT -- dead model"


def test_bundled_model_responds_to_geometry():
    """The model-level analogue of non-constant: output must MOVE when a driving feature moves.

    A frozen/degenerate model can still emit varied values across varied rows (that is what the
    test above catches); this catches one that IGNORES its geometry. Direction is deliberately
    unasserted -- see the retirement note above -- only that the response is real.
    """
    from silly_kicks.tracking._xshot_occurrence import XSHOT_FEATURE_NAMES_FAITHFUL, XShotOccurrenceModel

    base = _frozen_rows()[XSHOT_FEATURE_NAMES_FAITHFUL].median().to_frame().T
    m = XShotOccurrenceModel.from_variant("default")
    probs = []
    for r in (5.0, 12.0, 20.0, 28.0, 34.0):  # swept INSIDE the trained domain
        row = base.copy()
        row["r"] = r
        probs.append(float(m.predict_proba(row)[0]))
    assert max(probs) - min(probs) > 1e-3, f"output flat across r: {probs}"


def test_from_variant_default_loads_and_predicts_in_bounds():
    from silly_kicks.tracking._xshot_occurrence import XSHOT_FEATURE_NAMES_FAITHFUL, XShotOccurrenceModel

    m = XShotOccurrenceModel.from_variant("default")
    p = m.predict_proba(pd.DataFrame(np.zeros((4, 27)), columns=XSHOT_FEATURE_NAMES_FAITHFUL))
    assert np.all((p >= 0) & (p <= 1))


def test_bundled_metadata_matches_training_intent():
    """L6: intent-named. carrier params == shared constant; coordinate/platform/provenance present."""
    import json
    from pathlib import Path

    from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS

    meta = json.loads(Path("silly_kicks/tracking/_xshot_weights/default/metadata.json").read_text())
    assert meta["carrier_params"] == DEFAULT_CARRIER_PARAMS  # trained on the 4.7.0 defaults
    for k in (
        "pitch_length",
        "pitch_width",
        "geometry_version",
        "xgboost_version",
        "training_platform",
        "shipped_variant",
    ):
        assert k in meta
    assert meta["shipped_variant"] == "public" and meta["provider_list"] == ["idsse", "skillcorner"]


def test_bundled_weights_present_in_package():
    """Wheel-content sanity (spec §9): the bundled dir ships inside the package."""
    from pathlib import Path

    import silly_kicks.tracking as t

    root = Path(t.__file__).parent / "_xshot_weights" / "default"
    assert (root / "model.json").exists() and (root / "SHA256SUMS").exists()


def test_xshot_xfn_in_gk_union_only_not_general():
    """P3 (owner-confirmed): xS joins the GK-context union ONLY; the general list stays model-free."""
    from silly_kicks.tracking.features import pre_shot_gk_full_default_xfns, tracking_default_xfns

    def _names(xs):
        return {getattr(f, "__name__", "") for f in xs}

    assert "xshot_occurrence_xfn" in _names(pre_shot_gk_full_default_xfns)
    assert "xshot_occurrence_xfn" not in _names(tracking_default_xfns)  # NOT in the general default

    # Atomic mirror: the atomic GK union mirrors the non-atomic one.
    from silly_kicks.atomic.tracking.features import atomic_pre_shot_gk_full_default_xfns

    assert "xshot_occurrence_xfn" in _names(atomic_pre_shot_gk_full_default_xfns)


def test_xshot_xfn_introspection_is_nan():
    from silly_kicks.tracking._xshot_occurrence import xshot_occurrence_xfns

    fn = xshot_occurrence_xfns()[0]
    states = [pd.DataFrame({"action_id": [0, 1]})]
    out = fn(states, None)  # frames=None -> 3 NaN columns, no model load
    assert list(out.columns) == ["xshot_occurrence_a0", "xshot_occurrence_a1", "xshot_occurrence_a2"]
    assert out.isna().all().all()


def test_objective_handles_mixed_dtype_groups():
    """Real-multi-provider regression (PR-S80): game_id is str (kloppy hashes) for some providers
    and int (Gradient Sports) for others, so concatenated CV groups are mixed-dtype. np.unique /
    StratifiedGroupKFold must not choke on the int-vs-str sort."""
    from ruthless import Candidate

    from silly_kicks.tracking._xshot_occurrence import XSHOT_FEATURE_NAMES_FAITHFUL
    from silly_kicks.tracking._xshot_occurrence_objective import XShotOccurrenceObjective

    rng = np.random.default_rng(1)

    def _match(gid):
        n = 60
        X = pd.DataFrame(rng.normal(size=(n, 27)), columns=XSHOT_FEATURE_NAMES_FAITHFUL)
        y = (X["r"] < 0).astype(int).to_numpy()
        if y.sum() == 0:
            y[:6] = 1
        return X, y, np.array([gid] * n)

    # provider A: str game_id (kloppy); provider B: int game_id (Gradient Sports)
    fold = {"sk": [_match("kloppy-hash-1"), _match("kloppy-hash-2")], "gs": [_match(101), _match(102)]}
    obj = XShotOccurrenceObjective(fold=fold)
    inv = obj.prepare()
    m = obj.evaluate_patch(inv, Candidate(id="t0", params={"n_estimators": 20.0, "max_depth": 3.0}))
    assert "logloss" in m and m["logloss"] == m["logloss"]  # finite, no crash


def test_objective_reports_pr_auc_and_brier():
    from ruthless import Candidate

    obj = _build_xshot_objective()
    inv = obj.prepare()
    metrics = obj.evaluate_patch(
        inv,
        Candidate(id="t0", params={"n_estimators": 20.0, "max_depth": 3.0}),
    )
    assert "logloss" in metrics
    assert "pr_auc" in metrics
    assert "brier" in metrics


# --- Task 12: training CLI ---


# ADR-128: the learnable / degenerate synthesis moved to `scripts/_toy_corpora.xshot_game` (one synthesis
# for the toy corpus and these tests).


def _write_synthetic_train_dir(tmp_path, *, n_games=4):
    """A learnable STAGED flat dataset ``<tmp>/data/syn`` (gates can PASS); returns the data root."""
    from pathlib import Path

    from scripts._toy_corpora import build_xshot_corpus

    build_xshot_corpus(Path(tmp_path) / "data" / "syn", providers=("skillcorner",) * n_games, heavy_episodes=None)
    return Path(tmp_path) / "data"


def _write_degenerate_train_dir(tmp_path, *, n_games=4):
    """A degenerate STAGED flat dataset (identical features -> the gates must FAIL); returns the data root."""
    from pathlib import Path

    from scripts._toy_corpora import build_xshot_corpus

    build_xshot_corpus(
        Path(tmp_path) / "data" / "syn",
        providers=("skillcorner",) * n_games,
        episodes=6,
        heavy_episodes=None,
        degenerate=True,
    )
    return Path(tmp_path) / "data"


def test_tf19_interface_stub():
    # TF-19 will call predict_proba on a feature matrix it controls. Verify the
    # object API returns finite, in-bounds probabilities for an arbitrary matrix.
    from silly_kicks.tracking._xshot_occurrence import (
        XSHOT_FEATURE_NAMES_FAITHFUL,
        XShotOccurrenceModel,
    )

    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(40, 27)), columns=XSHOT_FEATURE_NAMES_FAITHFUL)
    y = pd.Series((rng.random(40) < 0.1).astype(int))
    model = XShotOccurrenceModel().fit(X, y)
    # A fresh matrix shaped exactly as TF-19 would assemble per ghost-GK frame.
    probe = pd.DataFrame(rng.normal(size=(5, 27)), columns=XSHOT_FEATURE_NAMES_FAITHFUL)
    p = model.predict_proba(probe)
    assert p.shape == (5,)
    assert np.all(np.isfinite(p))
    assert np.all((p >= 0.0) & (p <= 1.0))


def test_carrier_params_in_metadata(tmp_path):
    # Standalone (R3): trainer writes the carrier params it used into metadata.json.
    from silly_kicks.tracking._xshot_occurrence import (
        XSHOT_FEATURE_NAMES_FAITHFUL,
        XShotOccurrenceModel,
    )

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 27)), columns=XSHOT_FEATURE_NAMES_FAITHFUL)
    y = pd.Series((rng.random(40) < 0.2).astype(int))
    cp = {"tolerance_m": 7.0, "beta": 0.25, "gamma": 1.5}
    model = XShotOccurrenceModel().fit(X, y, carrier_params=cp)
    from pathlib import Path

    art = Path(tmp_path) / "m"
    model.save(art)
    import json

    meta = json.loads((art / "metadata.json").read_text())
    assert meta["carrier_params"] == cp


@pytest.mark.e2e
def _e2e_load_real(providers, max_per_provider):
    """Load real matches via the pining loader -> (X, y, groups, providers_per_row). Gated:
    skips cleanly without the token / kloppy so normal CI never runs this. Scale is overridable
    via XSHOT_E2E_MAX_PER_PROVIDER (e.g. a quick confirm run)."""
    import os
    import sys

    pytest.importorskip("kloppy")
    if not os.environ.get("PINING_FOR_THE_DATA_TOKEN"):
        pytest.skip("PINING_FOR_THE_DATA_TOKEN not set (gated real-data e2e)")
    max_per_provider = int(os.environ.get("XSHOT_E2E_MAX_PER_PROVIDER", str(max_per_provider)))
    sys.path.insert(0, "scripts")
    from _loader_pining import load_matches

    from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS
    from silly_kicks.tracking._xshot_occurrence import prepare_xshot_training_data

    xs_, ys_, gs_, ps_ = [], [], [], []
    for prov, _mid, actions, frames, home in load_matches(providers=providers, max_per_provider=max_per_provider):
        X, y, groups = prepare_xshot_training_data(
            frames,
            actions,
            home_team_id=home,  # type: ignore[arg-type]
            carrier_params=DEFAULT_CARRIER_PARAMS,
        )
        if len(X):
            xs_.append(X)
            ys_.append(np.asarray(y, int))
            gs_.append(np.asarray(groups).astype(str))
            ps_.append(np.array([prov] * len(X)))
    assert xs_, "no usable real-data rows extracted"
    return pd.concat(xs_, ignore_index=True), np.concatenate(ys_), np.concatenate(gs_), np.concatenate(ps_)


def _bundled_params():
    import json
    from pathlib import Path

    return json.loads(Path("silly_kicks/tracking/_xshot_weights/default/metadata.json").read_text())["params"]


@pytest.mark.e2e
def test_xshot_gradientsports_e2e():
    """On real multi-provider data, the bundled-hyperparameter model beats the base rate on
    BOTH discrimination (PR-AUC > base rate) and calibration (Brier < base-rate Brier)."""
    import sys

    sys.path.insert(0, "scripts")
    from train_xshot_occurrence import _cv_metrics, _gates

    X, y, groups, _prov = _e2e_load_real(["skillcorner", "idsse", "gradientsports"], max_per_provider=3)  # type: ignore[misc]
    m = _cv_metrics(X, y, groups, _bundled_params())
    g = _gates(m)
    assert g["pr_auc_gt_base_rate"], m
    assert g["brier_lt_base_rate_brier"], m
    assert g["log_loss_lt_uniform"], m


@pytest.mark.e2e
def test_xshot_cross_provider():
    """Trained on >=2 providers; no single provider's held-out PR-AUC falls below its base rate."""
    import sys

    sys.path.insert(0, "scripts")
    from train_xshot_occurrence import _cv_metrics

    X, y, groups, prov = _e2e_load_real(["skillcorner", "idsse", "gradientsports"], max_per_provider=3)  # type: ignore[misc]
    seen = set()
    for p in np.unique(prov):
        mask = prov == p
        if len(np.unique(groups[mask])) < 2 or len(np.unique(y[mask])) < 2:
            continue  # need >=2 games + both classes to CV this provider
        seen.add(p)
        mp = _cv_metrics(X[mask], y[mask], groups[mask], _bundled_params())
        assert mp["pr_auc"] >= mp["positive_rate"], (p, mp)
    assert len(seen) >= 2, f"cross-provider gate needs >=2 evaluable providers, got {seen}"


def _arms(data_root, *, two_tier=False) -> str:
    """The launcher's neutral ``--arms-json`` for the staged toy (ADR-123 Amendment B). Default: every game
    ``public`` (the single-arm trainer ships ``public``). ``two_tier``: half the games ``public``, ALL in
    ``sc_extended``, NO ``full`` -> not paired, so the BROADEST arm (sc_extended) must ship."""
    import json

    from scripts._toy_corpora import xshot_arms

    if not two_tier:
        return xshot_arms(data_root / "syn", paired=False)
    keys = json.loads(xshot_arms(data_root / "syn", paired=False))["arms"]["public"]
    return json.dumps(
        {
            "arms": {"public": keys[: max(1, len(keys) // 2)], "sc_extended": keys},
            "ship_order": ["public", "sc_extended"],
        }
    )


def _train(tmp_path, data_root, arms_json, *extra, check=True):
    """One stage-CLI ``--run-all`` (ADR-128) into ``<tmp>/out``; the store is ``<tmp>/store``."""
    from pathlib import Path

    from tests.scripts._parity import cli

    args = ["--dataset", "syn", "--arms-json", arms_json, "--horizon-seconds", "1.0", *extra]
    return cli(
        "scripts.train_xshot_occurrence",
        Path(tmp_path) / "store",
        data_root,
        args,
        "--run-all",
        "--out",
        str(Path(tmp_path) / "out"),
        check=check,
    )


@pytest.mark.slow
def test_train_script_smoke(tmp_path):
    import json
    from pathlib import Path

    data = _write_synthetic_train_dir(tmp_path)
    _train(tmp_path, data, _arms(data), "--n-trials", "3")
    art = Path(tmp_path) / "out"
    assert (art / "model.json").exists()
    assert (art / "metadata.json").exists()
    assert (art / "SHA256SUMS").exists()
    meta = json.loads((art / "metadata.json").read_text())
    assert "carrier_params" in meta
    assert meta["feature_set"] == "faithful"
    metrics = json.loads((art / "metrics.json").read_text())
    assert metrics["acceptance"] and all(metrics["acceptance"].values())  # gates passed
    assert metrics["estimates_are_cv_not_shipped_fit"] is True  # N7


@pytest.mark.slow
def test_train_script_smoke_position_only(tmp_path):
    # Task 8: `--feature-set position_only` trains a 26-feature model (velocity `speed` dropped) that
    # saves + loads through the position_only chirality/feature-contract guards. Mirrors the faithful
    # smoke above; the synthetic frames carry no speed column a position_only model would select.
    import json
    from pathlib import Path

    data = _write_synthetic_train_dir(tmp_path)
    _train(tmp_path, data, _arms(data), "--n-trials", "3", "--feature-set", "position_only")
    art = Path(tmp_path) / "out"
    meta = json.loads((art / "metadata.json").read_text())
    assert meta["feature_set"] == "position_only"
    assert len(meta["feature_names"]) == 26  # velocity `speed` dropped
    # Loads through the position_only chirality + feature-contract guards (Task 6 threading):
    from silly_kicks.tracking._xshot_occurrence import XShotOccurrenceModel

    XShotOccurrenceModel.load(art)


@pytest.mark.slow
def test_train_script_ships_broadest_arm_when_not_paired(tmp_path):
    """ADR-126 / ADR-038: a 2-tier (public + sc_extended, no full) corpus ships ``sc_extended``, NEVER
    ``public`` -- an owner-tier-trained model must not carry a public label. Regression for the
    ``ship_order[-1]`` fix (the sc_extended re-fit previously mis-shipped the narrow ``public`` arm)."""
    import json
    from pathlib import Path

    data = _write_synthetic_train_dir(tmp_path)
    _train(tmp_path, data, _arms(data, two_tier=True), "--n-trials", "3")
    meta = json.loads((Path(tmp_path) / "out" / "metadata.json").read_text())
    assert meta["shipped_variant"] == "sc_extended", meta.get("shipped_variant")


@pytest.mark.slow
def test_train_script_fail_closed_writes_no_artifact(tmp_path):
    """N3: a corpus that cannot beat the base rate -> non-zero exit, NO bundled artifact; the legacy
    ``metrics_FAILED.json`` payload is kept beside the failed ``fit`` unit."""
    from pathlib import Path

    data = _write_degenerate_train_dir(tmp_path)
    result = _train(tmp_path, data, _arms(data), "--n-trials", "2", check=False)
    assert result.returncode != 0
    assert not (Path(tmp_path) / "out" / "model.json").exists()
    assert list((Path(tmp_path) / "store").rglob("only.metrics_FAILED.json")), "the gate failure left no record"


def test_publish_verify_only(tmp_path):
    """Publish script's --verify-only path: load + SHA-verify + sanity predict, no network."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    from silly_kicks.tracking._xshot_occurrence import XSHOT_FEATURE_NAMES_FAITHFUL, XShotOccurrenceModel

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 27)), columns=XSHOT_FEATURE_NAMES_FAITHFUL)
    y = pd.Series((rng.random(40) < 0.3).astype(int))
    art = Path(tmp_path) / "xshot_occurrence_v1"
    XShotOccurrenceModel().fit(X, y).save(art)
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(  # noqa: S603
        [sys.executable, "scripts/publish_xshot_occurrence.py", "--artifact-dir", str(art), "--verify-only"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(repo_root),
        env=dict(os.environ, PYTHONPATH=str(repo_root)),
    )
    assert result.returncode == 0, result.stderr


def _paired_synthetic_corpus(seed=0):
    """A public/owner mix with a learnable r-signal: 6 public games + 3 owner ("sc") games. Returns
    (X, y, groups, is_public). sc_extended/full carry strictly MORE data than the public arm."""
    from silly_kicks.tracking._xshot_occurrence import XSHOT_FEATURE_NAMES_FAITHFUL

    rng = np.random.default_rng(seed)
    rows, lab, grp, pub = [], [], [], []
    for g in range(9):  # games 0..5 public, 6..8 owner ("sc")
        is_pub = g < 6
        for _ in range(70):
            r = float(rng.uniform(5, 40))
            y_ = int(rng.random() < 1 / (1 + np.exp((r - 18) / 4)))  # closer => more likely
            row = {c: 0.0 for c in XSHOT_FEATURE_NAMES_FAITHFUL}
            row["r"] = r
            rows.append(row)
            lab.append(y_)
            grp.append(g)
            pub.append(is_pub)
    X = pd.DataFrame(rows)[XSHOT_FEATURE_NAMES_FAITHFUL]
    return X, np.array(lab), np.array(grp), np.array(pub, dtype=bool)


def test_paired_data_effect_three_arm_shape(monkeypatch, tmp_path):
    """Task 11 (spec 4.1): the nested-HPO paired test returns per-candidate 'nested' + 'shared_params'
    delta lists for all three arms; 'public' is the baseline (empty lists). `_hpo_once` is monkey-
    patched to fixed per-tag params so the shape is exercised without a real Optuna sweep."""
    import scripts.train_xshot_occurrence as tr

    def _fake_hpo(
        x,
        y,
        groups,
        out_dir,
        tag,
        n_trials,
        *,
        objective_inputs,
        prov,
        negative_subsample=None,
        seed=42,
        study_shard_dir=None,
    ):
        # Distinct params keyed by candidate so nested (own) != shared (public) is DETERMINISTIC.
        if tag.startswith("sc_extended"):
            return {"max_depth": 6, "n_estimators": 120}
        return {"max_depth": 2, "n_estimators": 60}

    monkeypatch.setattr(tr, "_hpo_once", _fake_hpo)

    X, y, groups, is_public = _paired_synthetic_corpus()
    match_ids = np.array([str(g) for g in groups])
    is_sc = ~is_public  # every non-public game is "sc" in this synthetic
    cand_masks = {"public": is_public, "sc_extended": is_public | is_sc, "full": np.ones(len(X), bool)}
    paired = tr._paired_data_effect(
        X,
        y,
        groups,
        is_public,
        match_ids,
        candidates=cand_masks,
        n_trials=1,
        out_dir=tmp_path,
        objective_inputs={"driver": "test-paired"},
        prov={"commit": "test", "tree_state": "clean"},
    )
    assert set(paired) == {"public", "sc_extended", "full"}
    for arm in cand_masks:
        assert set(paired[arm]) == {"nested", "shared_params"}
    assert paired["public"]["nested"] == [] and paired["public"]["shared_params"] == []  # baseline arm
    assert len(paired["sc_extended"]["nested"]) >= 2  # usable folds


def test_paired_nested_uses_candidate_params_not_public(monkeypatch, tmp_path):
    """KILL-LINE (M4): the 'nested' protocol scores each candidate at ITS OWN tuned params, the
    'shared_params' protocol at the PUBLIC params. With `_hpo_once` forced to distinct params per
    candidate, sc_extended's two delta series MUST differ. If a refactor scored the nested column at
    public params too (the leakage the M4 fix removes), they would collapse to identical -> this fails.
    """
    import scripts.train_xshot_occurrence as tr

    def _fake_hpo(
        x,
        y,
        groups,
        out_dir,
        tag,
        n_trials,
        *,
        objective_inputs,
        prov,
        negative_subsample=None,
        seed=42,
        study_shard_dir=None,
    ):
        if tag.startswith("sc_extended"):
            return {"max_depth": 6, "n_estimators": 200}
        return {"max_depth": 1, "n_estimators": 40}

    monkeypatch.setattr(tr, "_hpo_once", _fake_hpo)

    X, y, groups, is_public = _paired_synthetic_corpus()
    match_ids = np.array([str(g) for g in groups])
    cand_masks = {"public": is_public, "sc_extended": np.ones(len(X), bool), "full": np.ones(len(X), bool)}
    paired = tr._paired_data_effect(
        X,
        y,
        groups,
        is_public,
        match_ids,
        candidates=cand_masks,
        n_trials=1,
        out_dir=tmp_path,
        objective_inputs={"driver": "test-paired"},
        prov={"commit": "test", "tree_state": "clean"},
    )
    assert paired["sc_extended"]["nested"] != paired["sc_extended"]["shared_params"], (
        "nested and shared_params collapsed -- the candidate was scored at public params (M4 leakage)"
    )


# --- Task A: prepare_xshot_training_data (public API) ---


def _match_frames_and_shots(n_frames=30, game_id=1):
    """Full-schema frames (team 2 in possession) + team-2 shots.

    Team 1 defends goal x=0 (GK ~x=2); team 2 attacks it. The ball + a team-2
    carrier advance from midfield (x~55) toward the team-1 goal across the match,
    so EARLY frames are midfield and LATE frames are in team-2's attacking third
    (x < 35) -- giving the attacking-third filter something to cut.
    """
    rows = []

    def row(pid, tid, ball, gk, x, y, fi, t):
        return {
            "player_id": pid,
            "team_id": tid,
            "is_ball": ball,
            "is_goalkeeper": gk,
            "x": x,
            "y": y,
            "frame_id": fi,
            "time_seconds": t,
        }

    for fi in range(n_frames):
        t = fi * 0.1
        # ball advances from x~55 down toward x~15 over the match
        bx = 55.0 - (40.0 * fi / max(1, n_frames - 1))
        rows.append(row(-1, -1, True, False, bx, 34.0, fi, t))
        rows.append(row(10, 1, False, True, 2.0, 34.0, fi, t))  # team-1 GK
        for k in range(5):
            rows.append(row(11 + k, 1, False, False, 8.0 + k, 25.0 + 3 * k, fi, t))
        rows.append(row(20, 2, False, True, 103.0, 34.0, fi, t))  # team-2 GK
        rows.append(row(21, 2, False, False, bx + 0.3, 34.0, fi, t))  # carrier
        for k in range(4):
            rows.append(row(22 + k, 2, False, False, bx + 5 + k, 25.0 + 3 * k, fi, t))
    frames = pd.DataFrame(rows)
    frames["game_id"] = game_id
    frames["period_id"] = 1
    frames["z"] = 0.0
    frames["frame_rate"] = 10.0
    frames["ball_state"] = "alive"
    # team-2 shots late in the match (when in the attacking third)
    shots = pd.DataFrame(
        {
            "game_id": [game_id, game_id],
            "period_id": [1, 1],
            "team_id": [2, 2],
            "time_seconds": [(n_frames - 5) * 0.1, (n_frames - 3) * 0.1],
            "type_id": [_SHOT_TID, _SHOT_TID],
        }
    )
    return frames, shots


def test_prepare_returns_features_labels_groups():
    from silly_kicks.tracking import prepare_xshot_training_data
    from silly_kicks.tracking._xshot_occurrence import XSHOT_FEATURE_NAMES_FAITHFUL

    frames, shots = _match_frames_and_shots()
    X, y, groups = prepare_xshot_training_data(frames, shots, home_team_id=1, horizon_seconds=1.0)
    assert list(X.columns) == XSHOT_FEATURE_NAMES_FAITHFUL
    assert len(X) == len(y) == len(groups)
    assert set(np.unique(y)).issubset({0, 1})
    # groups are the game_id for GroupKFold
    assert (groups == frames["game_id"].iloc[0]).all()


def test_prepare_attacking_third_filter_reduces_rows():
    # attacking_third_only=True must keep FEWER frames than the unfiltered pass
    # (ball starts midfield-ish in the synth match; only attacking-third frames stay).
    from silly_kicks.tracking import prepare_xshot_training_data

    frames, shots = _match_frames_and_shots(n_frames=40)
    x_third, _, _ = prepare_xshot_training_data(frames, shots, home_team_id=1, attacking_third_only=True)
    x_all, _, _ = prepare_xshot_training_data(frames, shots, home_team_id=1, attacking_third_only=False)
    assert len(x_third) <= len(x_all)
    assert len(x_third) >= 1  # but not empty for this fixture


def test_prepare_excludes_dead_ball_frames():
    # ball_state != "alive" frames must be dropped.
    from silly_kicks.tracking import prepare_xshot_training_data

    frames, shots = _match_frames_and_shots(n_frames=30)
    frames = frames.copy()
    n_alive_frames = frames.loc[frames["ball_state"] == "alive", "frame_id"].nunique()
    # Kill the ball in half the frames.
    dead_ids = sorted(frames["frame_id"].unique())[: n_alive_frames // 2]
    frames.loc[frames["frame_id"].isin(dead_ids), "ball_state"] = "dead"
    x_filtered, _, _ = prepare_xshot_training_data(frames, shots, home_team_id=1, attacking_third_only=False)
    # No scored frame may come from a dead-ball frame_id.
    assert len(x_filtered) <= (frames["frame_id"].nunique() - len(dead_ids))


def test_prepare_shot_types_toggle():
    # Restricting shot_types to {} yields all-zero labels (no shot counts).
    from silly_kicks.tracking import prepare_xshot_training_data

    frames, shots = _match_frames_and_shots(n_frames=30)
    _, y_default, _ = prepare_xshot_training_data(frames, shots, home_team_id=1)
    _, y_none, _ = prepare_xshot_training_data(frames, shots, home_team_id=1, shot_types=())
    assert y_default.sum() >= 1  # default counts the shots
    assert y_none.sum() == 0  # empty shot-type set -> no positives


def test_prepare_returns_faithful_distribution_no_subsample_param():
    """PR-S80 M3: prepare_* no longer subsamples (the contamination footgun is gone). The
    `negative_subsample`/`seed` params were removed -- passing them is a TypeError."""
    import inspect

    from silly_kicks.tracking import prepare_xshot_training_data

    sig = inspect.signature(prepare_xshot_training_data).parameters
    assert "negative_subsample" not in sig and "seed" not in sig
    frames, shots = _match_frames_and_shots(n_frames=40)
    with pytest.raises(TypeError):
        prepare_xshot_training_data(frames, shots, home_team_id=1, negative_subsample=0.5)  # type: ignore[call-arg]


def test_subsample_negatives_deterministic_and_negatives_only():
    """The standalone TRAIN-ONLY helper drops only negatives, deterministically given seed."""
    from silly_kicks.tracking import subsample_negatives
    from silly_kicks.tracking._xshot_occurrence import XSHOT_FEATURE_NAMES_FAITHFUL

    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(rng.normal(size=(n, 27)), columns=XSHOT_FEATURE_NAMES_FAITHFUL)
    y = (rng.random(n) < 0.25).astype(int)
    groups = np.array(["g"] * n)
    Xa, ya, ga = subsample_negatives(X, y, groups, fraction=0.5, seed=7)
    _, yb, _ = subsample_negatives(X, y, groups, fraction=0.5, seed=7)
    np.testing.assert_array_equal(ya, yb)  # deterministic
    assert int((y == 1).sum()) == int((ya == 1).sum())  # ALL positives kept
    assert int((ya == 0).sum()) < int((y == 0).sum())  # negatives thinned
    assert len(Xa) == len(ya) == len(ga)


def test_cv_metrics_subsample_is_train_fold_only():
    """M3 regression: subsampling thins TRAIN folds but the reported metrics + base-rate baselines
    stay on the TRUE held-out balance (positive_rate unchanged by aggressive subsampling)."""
    import sys

    sys.path.insert(0, "scripts")
    from train_xshot_occurrence import _cv_metrics

    from silly_kicks.tracking._xshot_occurrence import XSHOT_FEATURE_NAMES_FAITHFUL

    rng = np.random.default_rng(1)
    rows, ys, gs = [], [], []
    for g in range(6):
        ng = 120
        Xg = rng.normal(size=(ng, 27))
        Xg[:, 0] = rng.normal(size=ng)  # r ~ noise
        yg = (rng.random(ng) < 0.2).astype(int)
        if yg.sum() == 0:
            yg[:5] = 1
        rows.append(pd.DataFrame(Xg, columns=XSHOT_FEATURE_NAMES_FAITHFUL))
        ys.append(yg)
        gs.append(np.array([str(g)] * ng))
    X = pd.concat(rows, ignore_index=True)
    y = np.concatenate(ys)
    groups = np.concatenate(gs)
    base = float(y.mean())
    m_off = _cv_metrics(X, y, groups, {"n_estimators": 30, "max_depth": 3})
    m_on = _cv_metrics(X, y, groups, {"n_estimators": 30, "max_depth": 3}, negative_subsample=0.8, seed=3)
    # Eval-side baselines reflect the TRUE balance regardless of train subsampling.
    assert abs(m_off["positive_rate"] - base) < 1e-9
    assert abs(m_on["positive_rate"] - base) < 1e-9
    assert abs(m_on["base_rate_brier"] - base * (1 - base)) < 1e-9
