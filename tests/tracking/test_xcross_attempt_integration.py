"""Integration tests for TF-17 xCrossAttempt (HPO objective, exports, atomic mirror, train CLI)."""

from __future__ import annotations

import numpy as np
import pandas as pd

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
    assert_cache_equivalence(obj, candidates)


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
    assert_cache_equivalence(obj, candidates)
