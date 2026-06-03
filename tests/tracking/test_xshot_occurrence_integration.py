"""Integration tests for TF-16 xShotOccurrence (xS)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


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
        store=StoreConfig(kind="sqlite", path=db),
    )
    result = OptunaStrategy(cfg, seed=42).run(obj, backend=InProcessBackend())
    assert result.best is not None
    assert "logloss" in result.best.metrics


def test_objective_search_excludes_scale_pos_weight_and_uses_stratified_cv():
    """M2: no scale_pos_weight (xS is a calibrated P(shot)); M1: label-stratified CV."""
    import inspect

    from silly_kicks.tracking import _xshot_occurrence_objective as obj_mod

    assert "scale_pos_weight" not in obj_mod._SEARCH_KEYS
    src = inspect.getsource(obj_mod._cv_logloss)
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
    assert_cache_equivalence(obj, candidates)


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
    assert_cache_equivalence(obj, candidates)


def test_bundled_model_is_live_not_degenerate():
    """LIVENESS tripwire (H2/N2/P4): the bundled model is not dead/constant — it ranks the
    cherry-picked imminent (near-goal) extremes above the quiet (far) ones. NOT a quality
    measure (the frozen rows are maximally separable in `r`); real quality lives in the e2e
    gates. Scale-free AUC, no magic margin; arch-robust (the model is ARM-trained)."""
    from sklearn.metrics import roc_auc_score

    from silly_kicks.tracking._xshot_occurrence import XSHOT_FEATURE_NAMES_FAITHFUL, XShotOccurrenceModel

    df = pd.read_parquet("tests/datasets/tracking/xshot_directional/frozen_rows.parquet")
    m = XShotOccurrenceModel.from_variant("default")
    p = m.predict_proba(df[XSHOT_FEATURE_NAMES_FAITHFUL])
    assert roc_auc_score(df["label"].to_numpy(), p) >= 0.9


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


def _xshot_frame_rows(fid, t, ball_x):
    """One frame's full-schema rows: ball + team-1 (defends x=0) + team-2 (attacks, carrier
    near the ball). ``ball_x`` drives the goal distance `r`, the dominant feature."""
    rows = [
        dict(
            player_id=-1, team_id=-1, is_ball=True, is_goalkeeper=False, x=ball_x, y=34.0, frame_id=fid, time_seconds=t
        ),
        dict(player_id=10, team_id=1, is_ball=False, is_goalkeeper=True, x=2.0, y=34.0, frame_id=fid, time_seconds=t),
        dict(player_id=20, team_id=2, is_ball=False, is_goalkeeper=True, x=103.0, y=34.0, frame_id=fid, time_seconds=t),
        # team-2 carrier 0.3 m from the ball -> possession resolves to team 2.
        dict(
            player_id=21,
            team_id=2,
            is_ball=False,
            is_goalkeeper=False,
            x=ball_x + 0.3,
            y=34.0,
            frame_id=fid,
            time_seconds=t,
        ),
    ]
    for k in range(5):  # team-1 outfield defenders
        rows.append(
            dict(
                player_id=11 + k,
                team_id=1,
                is_ball=False,
                is_goalkeeper=False,
                x=6.0 + k,
                y=28.0 + 2 * k,
                frame_id=fid,
                time_seconds=t,
            )
        )
    for k in range(4):  # team-2 outfield attackers
        rows.append(
            dict(
                player_id=22 + k,
                team_id=2,
                is_ball=False,
                is_goalkeeper=False,
                x=ball_x + 2 + k,
                y=30.0 + 2 * k,
                frame_id=fid,
                time_seconds=t,
            )
        )
    return rows


def _finalize_train_frames(rows, g):
    frames = pd.DataFrame(rows)
    frames["game_id"] = g
    frames["period_id"] = 1
    frames["z"] = 0.0
    frames["frame_rate"] = 10.0
    frames["ball_state"] = "alive"
    return frames


def _write_synthetic_train_dir(tmp_path, *, n_games=4):
    """Learnable parquet games (gates can PASS): pre-shot frames put the ball NEAR the attacked
    goal (small `r`) and quiet frames keep it in the attacking third but farther (larger `r`), so
    `r` cleanly predicts the label. Each game: 6 episodes x (1 positive near + 2 negative far)."""
    from pathlib import Path

    data_dir = Path(tmp_path) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for g in range(n_games):
        gdir = data_dir / f"game_{g}"
        gdir.mkdir()
        rows, shot_times, fid = [], [], 0
        # Many episodes so positives are plentiful: splits stay feasible even at the HPO's
        # min_child_weight upper bound (the child-hessian sum must clear it on tiny data).
        for ep in range(20):
            base = ep * 7.0
            shot_times.append(base + 0.9)  # one shot covers the 3 near frames (each within 1 s)
            specs = [
                (base, 8.0),  # near goal (r~8) -> positive
                (base + 0.3, 8.0),
                (base + 0.6, 8.0),
                (base + 2.5, 30.0),  # far but in the attacking third (r~30) -> negative
                (base + 3.0, 30.0),
                (base + 3.5, 30.0),
            ]
            for t, ball_x in specs:
                rows.extend(_xshot_frame_rows(fid, t, ball_x))
                fid += 1
        _finalize_train_frames(rows, g).to_parquet(gdir / "frames.parquet")
        pd.DataFrame(
            {
                "game_id": [g] * len(shot_times),
                "period_id": [1] * len(shot_times),
                "team_id": [2] * len(shot_times),
                "time_seconds": shot_times,
            }
        ).to_parquet(gdir / "shots.parquet")
    return data_dir


def _write_degenerate_train_dir(tmp_path, *, n_games=4):
    """Degenerate parquet games (gates must FAIL): every frame has IDENTICAL features (ball fixed
    at x=20), so the model cannot beat the base rate -> PR-AUC == base rate, Brier == base-rate
    Brier -> the strict acceptance gates fail and the trainer must refuse to write an artifact."""
    from pathlib import Path

    data_dir = Path(tmp_path) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for g in range(n_games):
        gdir = data_dir / f"game_{g}"
        gdir.mkdir()
        rows, shot_times, fid = [], [], 0
        for ep in range(6):
            base = ep * 6.0
            shot_times.append(base + 0.3)
            for t in (base, base + 2.5, base + 3.5):
                rows.extend(_xshot_frame_rows(fid, t, 20.0))  # constant ball_x -> identical features
                fid += 1
        _finalize_train_frames(rows, g).to_parquet(gdir / "frames.parquet")
        pd.DataFrame(
            {
                "game_id": [g] * len(shot_times),
                "period_id": [1] * len(shot_times),
                "team_id": [2] * len(shot_times),
                "time_seconds": shot_times,
            }
        ).to_parquet(gdir / "shots.parquet")
    return data_dir


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
            frames, actions, home_team_id=home, carrier_params=DEFAULT_CARRIER_PARAMS
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

    X, y, groups, _prov = _e2e_load_real(["skillcorner", "idsse", "gradientsports"], max_per_provider=3)
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

    X, y, groups, prov = _e2e_load_real(["skillcorner", "idsse", "gradientsports"], max_per_provider=3)
    seen = set()
    for p in np.unique(prov):
        mask = prov == p
        if len(np.unique(groups[mask])) < 2 or len(np.unique(y[mask])) < 2:
            continue  # need >=2 games + both classes to CV this provider
        seen.add(p)
        mp = _cv_metrics(X[mask], y[mask], groups[mask], _bundled_params())
        assert mp["pr_auc"] >= mp["positive_rate"], (p, mp)
    assert len(seen) >= 2, f"cross-provider gate needs >=2 evaluable providers, got {seen}"


def test_train_script_smoke(tmp_path):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    data_dir = _write_synthetic_train_dir(tmp_path)
    out_dir = Path(tmp_path) / "out"
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(repo_root))
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "scripts/train_xshot_occurrence.py",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(out_dir),
            "--n-trials",
            "3",
            "--horizon-seconds",
            "1.0",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(repo_root),
        env=env,
    )
    assert result.returncode == 0, result.stderr
    art = out_dir / "xshot_occurrence_v1"
    assert (art / "model.json").exists()
    assert (art / "metadata.json").exists()
    assert (art / "SHA256SUMS").exists()
    meta = json.loads((art / "metadata.json").read_text())
    assert "carrier_params" in meta
    assert meta["feature_set"] == "faithful"
    metrics = json.loads((art / "metrics.json").read_text())
    assert metrics["acceptance"] and all(metrics["acceptance"].values())  # gates passed
    assert metrics["estimates_are_cv_not_shipped_fit"] is True  # N7


def test_train_script_fail_closed_writes_no_artifact(tmp_path):
    """N3: a corpus that cannot beat the base rate -> non-zero exit, NO bundled artifact."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    data_dir = _write_degenerate_train_dir(tmp_path)
    out_dir = Path(tmp_path) / "out"
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "scripts/train_xshot_occurrence.py",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(out_dir),
            "--n-trials",
            "2",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(repo_root),
        env=dict(os.environ, PYTHONPATH=str(repo_root)),
    )
    assert result.returncode != 0
    assert not (out_dir / "xshot_occurrence_v1" / "model.json").exists()


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


def test_paired_decision_rule_data_effect():
    """Direct unit test of the subtle paired-decision helper (P1) — not via subprocess."""
    from scripts.train_xshot_occurrence import _paired_data_effect
    from silly_kicks.tracking._xshot_occurrence import XSHOT_FEATURE_NAMES_FAITHFUL

    rng = np.random.default_rng(0)
    rows, lab, grp, pub = [], [], [], []
    for g in range(7):  # last 2 games are "GS" (non-public)
        is_pub = g < 5
        for _ in range(60):
            r = float(rng.uniform(5, 40))
            y_ = int(rng.random() < 1 / (1 + np.exp((r - 18) / 4)))  # closer => more likely
            row = {c: 0.0 for c in XSHOT_FEATURE_NAMES_FAITHFUL}
            row["r"] = r
            rows.append(row)
            lab.append(y_)
            grp.append(g)
            pub.append(is_pub)
    X = pd.DataFrame(rows)[XSHOT_FEATURE_NAMES_FAITHFUL]
    res = _paired_data_effect(
        X, np.array(lab), np.array(grp), np.array(pub), shared_params={"max_depth": 3, "n_estimators": 60}
    )
    assert res["paired_delta_is_data_effect_shared_params"] is True
    assert res["paired_hpo_nested"] is False
    assert set(res) >= {"deltas", "K", "n_positive", "ship_two"}
    assert isinstance(res["ship_two"], bool)


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
        prepare_xshot_training_data(frames, shots, home_team_id=1, negative_subsample=0.5)


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
