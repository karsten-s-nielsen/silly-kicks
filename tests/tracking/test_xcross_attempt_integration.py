"""Integration tests for TF-17 xCrossAttempt (HPO objective, exports, atomic mirror, train CLI)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._xcross_attempt import XCROSS_FEATURE_NAMES_FAITHFUL


def _build_xcross_fold():
    """Two synthetic games, each with both label classes -> StratifiedGroupKFold(2) is well-posed."""
    rng = np.random.default_rng(7)
    frames = []
    for gi, gid in enumerate(["g1", "g2"]):
        n = 120
        X = pd.DataFrame(rng.normal(size=(n, 16)), columns=XCROSS_FEATURE_NAMES_FAITHFUL)
        # signal in gk_r so folds are learnable + both classes present
        y = (X["gk_r"] + rng.normal(scale=0.7, size=n) + 0.2 * gi > 0).astype(int).to_numpy()
        groups = np.array([gid] * n)
        frames.append((X, y, groups))
    return {"synthetic": frames}


def test_objective_cache_equivalence():
    from ruthless import Candidate, assert_cache_equivalence

    from silly_kicks.tracking._xcross_attempt_objective import XCrossAttemptObjective

    obj = XCrossAttemptObjective(fold=_build_xcross_fold())
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


# --- Task 11: training CLI ---


def _xcross_frame_rows(fid, t, ball_x):
    """One frame's rows in the WIDE corridor (y=8). Team-1 defends x=0 (GK x=2); team-2 attacks +
    carries the ball near the byline. ``ball_x`` (= carrier gr_x, goal at 0) drives dist_endline,
    the learnable signal (near byline -> cross imminent -> positive)."""

    def _r(pid, tid, x, y, *, is_ball=False, is_gk=False):
        return dict(
            player_id=pid,
            team_id=tid,
            is_ball=is_ball,
            is_goalkeeper=is_gk,
            x=x,
            y=y,
            frame_id=fid,
            time_seconds=t,
        )

    rows = [
        _r(-1, -1, ball_x, 8.0, is_ball=True),
        _r(10, 1, 2.0, 34.0, is_gk=True),
        _r(20, 2, 103.0, 34.0, is_gk=True),
        _r(21, 2, ball_x + 0.3, 8.0),  # team-2 carrier 0.3 m from the ball -> possession = team 2
    ]
    rows += [_r(11 + k, 1, 6.0 + k, 28.0 + 2 * k) for k in range(5)]  # team-1 defenders (box near x=0)
    rows += [_r(22 + k, 2, 8.0 + k, 30.0 + 2 * k) for k in range(4)]  # team-2 attackers (box)
    return rows


def _finalize_xcross_frames(rows, g):
    frames = pd.DataFrame(rows)
    frames["game_id"] = g
    frames["period_id"] = 1
    frames["z"] = 0.0
    frames["frame_rate"] = 10.0
    frames["ball_state"] = "alive"
    frames["source_provider"] = "synthetic"
    frames["vx"] = 0.0
    frames["vy"] = 0.0
    return frames


def _write_synthetic_train_dir(tmp_path, *, n_games=4, learnable=True):
    """Learnable parquet games (gates PASS when learnable=True): near-byline frames (small
    dist_endline) precede a cross -> positive; farther wide frames -> negative. learnable=False
    fixes the carrier position so the model cannot beat the base rate (gates FAIL)."""
    from pathlib import Path

    from silly_kicks.spadl import config as spc

    data_dir = Path(tmp_path) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for g in range(n_games):
        gdir = data_dir / f"game_{g}"
        gdir.mkdir()
        rows, cross_times, fid = [], [], 0
        for ep in range(20):
            base = ep * 7.0
            cross_times.append(base + 0.9)  # one cross covers the 3 near frames (each within 1 s)
            near_x = 8.0 if learnable else 20.0
            far_x = 30.0 if learnable else 20.0
            specs = [
                (base, near_x),
                (base + 0.3, near_x),
                (base + 0.6, near_x),
                (base + 2.5, far_x),
                (base + 3.0, far_x),
                (base + 3.5, far_x),
            ]
            for t, ball_x in specs:
                rows.extend(_xcross_frame_rows(fid, t, ball_x))
                fid += 1
        _finalize_xcross_frames(rows, g).to_parquet(gdir / "frames.parquet")
        pd.DataFrame(
            {
                "game_id": [g] * len(cross_times),
                "period_id": [1] * len(cross_times),
                "team_id": [2] * len(cross_times),
                "time_seconds": cross_times,
                "type_id": [spc.actiontype_id["cross"]] * len(cross_times),
                "result_id": [spc.result_id["success"]] * len(cross_times),
            }
        ).to_parquet(gdir / "actions.parquet")
    return data_dir


@pytest.mark.slow
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
            "scripts/train_xcross_attempt.py",
            # ADR-052: the trainer refuses a dirty tree (it writes bundled weights); a test
            # run is by definition a dev run. The artifact still records run_tree_dirty.
            "--allow-dirty",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(out_dir),
            "--n-trials",
            "3",
            "--horizon-seconds",
            "1.0",
            # M5: probe capture is provider-CONTROLLED (default = gradientsports, the gated
            # cohort); the synthetic fixture's provider must be opted in explicitly.
            "--probe-providers",
            "synthetic",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(repo_root),
        env=env,
    )
    assert result.returncode == 0, result.stderr
    art = out_dir / "xcross_attempt_v1"
    assert (art / "model.json").exists()
    assert (art / "metadata.json").exists()
    assert (art / "SHA256SUMS").exists()
    meta = json.loads((art / "metadata.json").read_text())
    assert "carrier_params" in meta
    assert meta["feature_set"] == "faithful"
    metrics = json.loads((art / "metrics.json").read_text())
    assert metrics["acceptance"] and all(metrics["acceptance"].values())  # gates passed
    assert metrics["estimates_are_cv_not_shipped_fit"] is True
    # PR-B headline validations land in metrics.json
    assert "delta_pr_auc" in metrics["gk_block_ablation"] and "delta_log_loss" in metrics["gk_block_ablation"]
    probe = metrics["gk_substitution_probe"]
    assert "tf19_ready" in probe and "gk_median_abs_delta" in probe
    assert "nearest_def_median_abs_delta" in probe and "random_band_median_abs_delta" in probe
    assert "score_differential_coverage" in metrics["permutation_importance"]
    assert metrics["permutation_importance"]["held_out"] is True
    assert metrics["score_differential_range_probe"]["abs_ge_12_count"] == 0  # clean synthetic
    assert "tf19_ready" in metrics
    # the substitution probe actually found eligible wide-area frames (the probe sample was saved + read)
    assert probe["n_frames_used"] >= 1
    # M5/M6 provenance: the probe sample carries its matches, the capture filter, and the
    # gate-recorded training-fold membership (single-candidate path: recorded, not refused).
    pm = json.loads((art / "_probe_sample" / "meta.json").read_text())
    assert pm["probe_providers"] == ["synthetic"]
    assert pm["probe_matches"] and all(p == "synthetic" for p, _m in pm["probe_matches"])
    assert "in_training_folds" in pm and set(pm["in_training_folds"]) == {m for _p, m in pm["probe_matches"]}
    assert metrics["probe_sample_matches"] == pm["probe_matches"]
    assert metrics["probe_gated_on_held_out"] is False  # single-candidate: no paired test ran


@pytest.mark.slow
def test_train_script_fail_closed_writes_no_artifact(tmp_path):
    """Fail-closed: a corpus that cannot beat the base rate -> non-zero exit, NO bundled artifact."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    data_dir = _write_synthetic_train_dir(tmp_path, learnable=False)
    out_dir = Path(tmp_path) / "out"
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(repo_root))
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "scripts/train_xcross_attempt.py",
            # ADR-052: the trainer refuses a dirty tree (it writes bundled weights); a test
            # run is by definition a dev run. The artifact still records run_tree_dirty.
            "--allow-dirty",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(out_dir),
            "--n-trials",
            "3",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(repo_root),
        env=env,
    )
    assert result.returncode != 0
    art = out_dir / "xcross_attempt_v1"
    assert not (art / "model.json").exists()  # fail-closed: no artifact written


# --- Task 12: exports + atomic mirror ---


def test_public_exports():
    import silly_kicks.tracking as t

    for name in [
        "XCrossFeatureSet",
        "XCrossAttemptModel",
        "add_xcross_attempt",
        "compute_xcross_attempt",
        "extract_xcross_features",
        "prepare_xcross_training_data",
        "xcross_attempt_xfns",
        "subsample_negatives",
    ]:
        assert hasattr(t, name), name


def test_atomic_mirror():
    from silly_kicks.atomic.tracking import features as af

    assert hasattr(af, "add_xcross_attempt") and hasattr(af, "xcross_attempt_xfns")


def test_import_silly_kicks_is_dependency_light():
    """import silly_kicks must NOT pull xgboost (inference gates on the [xgboost] extra, lazy)."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    code = "import sys, silly_kicks; assert 'xgboost' not in sys.modules, sorted(m for m in sys.modules if 'xgb' in m)"
    r = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=str(repo_root)),
    )
    assert r.returncode == 0, r.stderr


def test_objective_cache_equivalence_with_train_subsample():
    """Train-fold negative subsampling is deterministic per fold -> cache-equivalence still holds."""
    from ruthless import Candidate, assert_cache_equivalence

    from silly_kicks.tracking._xcross_attempt_objective import XCrossAttemptObjective

    obj = XCrossAttemptObjective(fold=_build_xcross_fold(), negative_subsample=0.5, subsample_seed=11)
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
                "n_estimators": 25.0,
                "max_depth": 4.0,
                "learning_rate": 0.2,
                "min_child_weight": 3.0,
                "reg_lambda": 1.0,
            },
        ),
    ]
    assert_cache_equivalence(obj, candidates)  # type: ignore[arg-type]


@pytest.mark.parametrize("seed,ns", [(42, None), (7, None), (7, 0.5)])
def test_cv_metrics_delegates_to_eval_cv_score(seed, ns):
    """M4 (closed by extraction): the acceptance gate (_cv_metrics) and the ablation share ONE
    _cv_score, so they use identical folds for ANY seed / negative_subsample. Exact equality (same
    call), exercised on the seed + ns axes the old parity test could not see."""
    import importlib.util
    from pathlib import Path

    from silly_kicks.tracking import _xcross_eval as ev

    spec = importlib.util.spec_from_file_location("_train_xcross", Path("scripts/train_xcross_attempt.py"))
    assert spec is not None and spec.loader is not None
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 16)), columns=XCROSS_FEATURE_NAMES_FAITHFUL)
    y = (rng.random(300) > 0.6).astype(int)
    groups = np.array((["g1"] * 150) + (["g2"] * 150))
    params = {"n_estimators": 40, "max_depth": 3, "learning_rate": 0.1, "min_child_weight": 1, "reg_lambda": 1.0}

    m = trainer._cv_metrics(X, y, groups, params, seed=seed, negative_subsample=ns)
    s = ev._cv_score(X, y, groups, params, seed=seed, negative_subsample=ns)
    assert m["pr_auc"] == s["pr_auc"]  # exact: _cv_metrics literally calls _cv_score
    assert m["log_loss"] == s["log_loss"]
    assert {"positive_rate", "base_rate_brier"} <= set(m)  # the gate keys still present


def test_from_hub_shape_mocked(monkeypatch, tmp_path):
    """Task 5: from_hub downloads then loads; mock snapshot_download to a local saved artifact."""
    import huggingface_hub

    from silly_kicks.tracking import _xcross_attempt as xc

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(120, 16)), columns=XCROSS_FEATURE_NAMES_FAITHFUL)
    m = xc.XCrossAttemptModel().fit(X, pd.Series((rng.random(120) > 0.6).astype(int)))
    d = tmp_path / "art"
    m.save(d)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo_id: str(d))
    back = xc.XCrossAttemptModel.from_hub("silly-kicks/xcross-attempt-v1")
    np.testing.assert_allclose(m.predict_proba(X), back.predict_proba(X), rtol=1e-9)


def test_from_variant_default_does_not_cascade_to_hub(monkeypatch):
    """B4: default loads bundled-or-raises; it must NOT call snapshot_download."""
    from silly_kicks.tracking import _xcross_attempt as xc

    xc._VARIANT_CACHE.clear()
    called = {"hub": False}
    monkeypatch.setattr(
        xc.XCrossAttemptModel,
        "from_hub",
        classmethod(lambda cls, repo_id=xc._HF_REPO_ID: called.__setitem__("hub", True)),
    )
    if not (xc._XCROSS_WEIGHTS_ROOT / "default" / "SHA256SUMS").exists():
        with pytest.raises(FileNotFoundError):
            xc.XCrossAttemptModel.from_variant("default")
        assert called["hub"] is False


def test_xcross_xfns_in_pre_shot_gk_full_default():
    from silly_kicks.tracking import features as tf

    names = {getattr(fn, "__name__", "") for fn in tf.pre_shot_gk_full_default_xfns}
    assert any("xcross" in n for n in names), names


def test_atomic_xcross_xfns_in_pre_shot_gk_full_default():
    from silly_kicks.atomic.tracking import features as af

    names = {getattr(fn, "__name__", "") for fn in af.atomic_pre_shot_gk_full_default_xfns}
    assert any("xcross" in n for n in names), names


# --- Task 7: directional fixture + bundled-model tripwire (bundled tests skip until weights land) ---
_XCROSS_DIRECTIONAL = "tests/datasets/tracking/xcross_directional/frozen_rows.parquet"
_NO_XCROSS_WEIGHTS = not __import__("pathlib").Path("silly_kicks/tracking/_xcross_weights/default/SHA256SUMS").exists()


def test_xcross_directional_fixture_schema():
    df = pd.read_parquet(_XCROSS_DIRECTIONAL)
    assert set(XCROSS_FEATURE_NAMES_FAITHFUL).issubset(df.columns)
    assert "label" in df.columns and df["label"].nunique() == 2
    assert df["label"].sum() >= 3 and (df["label"] == 0).sum() >= 3


# --- 4.74.0: `test_xcross_bundled_model_is_live_not_degenerate` RETIRED ---------------------------
#
# It asserted `roc_auc_score(...) >= 0.9` and measured 1.0000 on a fixture where SEVEN of sixteen
# features were constant -- the entire GK block among them, pinned at (2.0, 34.0) by the generator.
# Overwriting that block with 0, 99 or NaN each left AUC at 1.0000, so a model ignoring keeper
# position entirely was indistinguishable from the real one. Retired-and-replaced per ADR-032's
# treatment of the dead `pitch_control_at_ball__spearman`, with the reason recorded here so an AUC
# check is not reinstated by someone who notices its absence.
#
# Ranking is NOT retained as a substitute: measured, a live-but-GK-blind model scores AUC 1.0000 on
# the repaired fixture -- HIGHER than the real model -- while responding 0.00% to keeper position.
# Only the response gate below separates them.

#: Frozen literals, measured at authoring time on the 48-row fixture, one order of magnitude below
#: each feature's observed range. NOT "a fixed fraction of the observed range": computed from the
#: dataframe at test time that rule is degenerate (`range <= frac*range` is False for every frac < 1),
#: and at frac 0.001 or 0.01 it finds only 7 of the 9 inert columns on the OLD fixture -- missing
#: exactly the two the range clause exists for (`gk_dist_near_post`/`_far_post`, 3.0 ULP apart).
_MIN_RANGE = {
    "ball_r": 1.0, "ball_theta": 0.1, "ball_speed": 1.0, "score_differential": 0.1,
    "dist_nearest_def": 1.0, "space_controlled": 100.0, "dist_nearest_teammate": 1.0,
    "dist_endline": 1.0, "box_off_def_ratio": 0.1, "ten_minute_warning": 0.1,
    "gk_r": 1.0, "gk_theta": 0.1, "gk_lateral_offset": 1.0,
    "gk_dist_near_post": 0.1, "gk_dist_far_post": 0.1, "gk_carrier_side": 1.0,
}  # fmt: skip

#: Keeper probes pinned in GOAL-RELATIVE coordinates. `gk_x`/`gk_y` are absolute, so a pinned
#: absolute probe would be a keeper on his line at one goal end and 103 m upfield at the other --
#: the gate would still pass, failing quietly on a physically absurd state.
_GK_PROBES = ((2.0, 34.0), (2.0, 26.0), (2.0, 42.0), (6.0, 28.0), (10.0, 34.0), (12.0, 40.0), (4.0, 30.0))

#: Minimum mean(|dp|) as a fraction of the fixture's own base rate. Measured range across the seven
#: probes above: 26.04%-50.29%, so 0.15 carries ~1.7x headroom below the weakest probe. Expressed
#: RELATIVE to the base rate (mean p = 0.012232) because an absolute bar cannot survive a base-rate
#: shift: at ~1% any absolute threshold above ~0.008 is unreachable by construction.
_MIN_GK_RESPONSE_REL = 0.15

_GK_COLS = (
    "gk_r", "gk_theta", "gk_lateral_offset",
    "gk_dist_near_post", "gk_dist_far_post", "gk_carrier_side",
)  # fmt: skip


def _gk_block_at(gk_gr_x, gk_y, carrier_gr_y):
    """The extractor's GK block (`_xcross_attempt.py:222-235`), from raw geometry.

    A deliberate train/serve duplication, kept in ONE helper so the drift has a single site. Pinned
    by `test_xcross_gk_block_rederivation_is_faithful`, which fails the moment the extractor changes.
    """
    import math

    from silly_kicks.tracking import _geometry as _geo
    from silly_kicks.tracking._xcross_attempt import _GOAL_HALF_WIDTH_M

    side = np.sign(carrier_gr_y - _geo.GOAL_Y) or 1.0
    near_y = _geo.GOAL_Y + _GOAL_HALF_WIDTH_M * side
    far_y = _geo.GOAL_Y - _GOAL_HALF_WIDTH_M * side
    return {
        "gk_r": math.hypot(gk_gr_x, gk_y - _geo.GOAL_Y),
        "gk_theta": math.atan2(gk_y - _geo.GOAL_Y, gk_gr_x),
        "gk_lateral_offset": float(gk_y - _geo.GOAL_Y),
        "gk_dist_near_post": math.hypot(gk_gr_x, gk_y - near_y),
        "gk_dist_far_post": math.hypot(gk_gr_x, gk_y - far_y),
        "gk_carrier_side": float((gk_y - _geo.GOAL_Y) * side),
    }


def _with_keeper_at(df, gr_x, gr_y):
    """Copy of `df` with the GK block recomputed for a keeper at goal-relative (gr_x, gr_y)."""
    from silly_kicks.tracking import _geometry as _geo

    out = df.copy()
    carrier_y = df["carrier_y"].to_numpy(dtype=float)
    goal_x = df["goal_x"].to_numpy(dtype=float)
    for i in range(len(df)):
        blk = _gk_block_at(gr_x, gr_y, _geo.to_goal_relative_y(carrier_y[i], goal_x=goal_x[i]))
        for c in _GK_COLS:
            out.iloc[i, out.columns.get_loc(c)] = blk[c]
    return out


def test_xcross_fixture_is_not_degenerate():
    """PRECONDITION. The single assertion that would have caught the defect cycles earlier.

    Lands RED on the pre-4.74.0 fixture: 16 rows (< 40), nine inert features, and half its rows
    outside the model's own `wide_area_only` trained domain.
    """
    from silly_kicks.tracking._xcross_attempt import _ADVANCE_M, _in_wide_area

    df = pd.read_parquet(_XCROSS_DIRECTIONAL)
    assert len(df) >= 40, f"only {len(df)} rows"

    # NaN-EXPLICIT. A bare `(max - min) <= tol` is NaN-BLIND, since `nan <= tol` is False -- and
    # `score_differential` was all-NaN on the old fixture, the headline example of the blindness.
    # `nunique <= 1` alone is also insufficient: gk_dist_near_post/_far_post held two values 3.0 ULP
    # apart. NEITHER clause alone finds all nine.
    inert = [
        c
        for c in XCROSS_FEATURE_NAMES_FAITHFUL
        if not df[c].notna().any() or float(df[c].max() - df[c].min()) <= _MIN_RANGE[c]
    ]
    assert not inert, f"inert features cannot contribute to any ranking: {inert}"

    missing = {"ball_x", "ball_y", "goal_x", "gk_x", "gk_y", "carrier_y"} - set(df.columns)
    assert not missing, f"raw geometry absent, so the domain clause is unfalsifiable: {missing}"
    bx = df["ball_x"].to_numpy(dtype=float)
    by = df["ball_y"].to_numpy(dtype=float)
    gx = df["goal_x"].to_numpy(dtype=float)
    out_of_domain = [(bx[i], by[i]) for i in range(len(df)) if not _in_wide_area(bx[i], by[i], gx[i], _ADVANCE_M)]
    assert not out_of_domain, f"rows outside wide_area_only are extrapolation: {out_of_domain[:3]}"


@pytest.mark.skipif(_NO_XCROSS_WEIGHTS, reason="bundled xcross default weights land in Task 13")
def test_xcross_gk_block_rederivation_is_faithful():
    """`_gk_block_at` duplicates the extractor; this is what stops the copy drifting silently."""
    from silly_kicks.tracking import _geometry as _geo

    df = pd.read_parquet(_XCROSS_DIRECTIONAL)
    gk_x = df["gk_x"].to_numpy(dtype=float)
    gk_y = df["gk_y"].to_numpy(dtype=float)
    goal_x = df["goal_x"].to_numpy(dtype=float)
    carrier_y = df["carrier_y"].to_numpy(dtype=float)
    actual = {c: df[c].to_numpy(dtype=float) for c in _GK_COLS}
    worst = 0.0
    for i in range(len(df)):
        got = _gk_block_at(
            _geo.to_goal_relative_x(gk_x[i], goal_x=goal_x[i]),
            gk_y[i],
            _geo.to_goal_relative_y(carrier_y[i], goal_x=goal_x[i]),
        )
        for c in _GK_COLS:
            worst = max(worst, abs(got[c] - actual[c][i]))
    assert worst <= 1e-9, f"re-derivation drifted from the extractor by {worst:.3e}"


@pytest.mark.skipif(_NO_XCROSS_WEIGHTS, reason="bundled xcross default weights land in Task 13")
def test_xcross_bundled_model_predictions_are_live():
    """LIVENESS -- finite, in [0, 1], non-constant.

    Renamed from `test_xcross_bundled_model_is_live_not_degenerate`: reusing the retired gate's exact
    name with a different body would hide the change from git history and from name-anchored guards.
    """
    from silly_kicks.tracking._xcross_attempt import XCrossAttemptModel

    df = pd.read_parquet(_XCROSS_DIRECTIONAL)
    p = XCrossAttemptModel.from_variant("default").predict_proba(df[XCROSS_FEATURE_NAMES_FAITHFUL])
    assert np.all(np.isfinite(p)), "non-finite predictions"
    assert np.all((p >= 0.0) & (p <= 1.0)), f"out of [0,1]: {p.min()} .. {p.max()}"
    assert len(np.unique(p)) > 1, "constant predictions"


@pytest.mark.skipif(_NO_XCROSS_WEIGHTS, reason="bundled xcross default weights land in Task 13")
def test_xcross_bundled_model_reads_the_gk_block():
    """RESPONSE. Claims exactly what it tests: perturbing the GK BLOCK moves the prediction.

    NOT "responds to keeper position" -- moving a real keeper also shifts `dist_nearest_def` and
    `space_controlled`, which a block-only sweep holds fixed. The swept state is a probe of which
    columns the model reads, not a realizable frame.

    `mean(|dp|)`, never `|mean dp|`: measured across these seven probes the absolute form spans 1.93x
    and the signed form 6.83x, so a signed-mean threshold would fail a correct model purely on which
    probe the author happened to pick. Direction is deliberately NOT asserted -- `gk_r` and
    `gk_carrier_side` are single-split step functions and the response sign flips between probes.
    """
    from silly_kicks.tracking._xcross_attempt import XCrossAttemptModel

    df = pd.read_parquet(_XCROSS_DIRECTIONAL)
    model = XCrossAttemptModel.from_variant("default")
    base = model.predict_proba(df[XCROSS_FEATURE_NAMES_FAITHFUL])
    assert base.mean() > 0, "degenerate base rate"

    for gr_x, gr_y in _GK_PROBES:
        probe = _with_keeper_at(df, gr_x, gr_y)
        moved = np.abs(model.predict_proba(probe[XCROSS_FEATURE_NAMES_FAITHFUL]) - base).mean()
        rel = moved / base.mean()
        assert rel >= _MIN_GK_RESPONSE_REL, (
            f"keeper probe ({gr_x}, {gr_y}): mean|dp|/base = {rel:.4f} < {_MIN_GK_RESPONSE_REL}. "
            "A model that does not read the GK block is indistinguishable from one that does."
        )


@pytest.mark.skipif(_NO_XCROSS_WEIGHTS, reason="bundled xcross default weights land in Task 13")
def test_xcross_from_variant_default_in_bounds():
    from silly_kicks.tracking._xcross_attempt import XCrossAttemptModel

    m = XCrossAttemptModel.from_variant("default")
    p = m.predict_proba(pd.DataFrame(np.zeros((4, 16)), columns=XCROSS_FEATURE_NAMES_FAITHFUL))
    assert np.all((p >= 0) & (p <= 1))


@pytest.mark.skipif(_NO_XCROSS_WEIGHTS, reason="bundled xcross default weights land in Task 13")
def test_xcross_bundled_metadata_matches_training_intent():
    import json
    from pathlib import Path

    md = json.loads(Path("silly_kicks/tracking/_xcross_weights/default/metadata.json").read_text())
    assert md["carrier_params"] == {"tolerance_m": 3.0, "beta": 0.0, "gamma": 0.25}
    assert md["pitch_length"] == 105.0 and md["pitch_width"] == 68.0
    assert "geometry_version" in md and "xgboost_version" in md
