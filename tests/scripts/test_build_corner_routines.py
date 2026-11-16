"""CI-safe unit slice for scripts/build_corner_routines.py (the pure fit kernel; corpus pass is e2e)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.build_corner_routines import _EMITTED_SHARD_COLUMNS, _SHARD_SCHEMA_VERSION, fit_models


def _snapshots(n_corners=12, players=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_corners):
        for p in range(players):
            # init near the penalty spot / 6-yard box; target inside the penalty area (x>88.5)
            rows.append(
                {
                    "game_id": "g",
                    "corner_id": c,
                    "player_id": f"p{p}",
                    "init_x": float(rng.normal(90.0, 1.5)),
                    "init_y": float(rng.normal(34.0, 6.0)),
                    "target_x": float(rng.normal(100.0, 1.5)),
                    "target_y": float(rng.normal(34.0, 8.0)),
                }
            )
    return pd.DataFrame(rows)


def test_emitted_shard_columns_declared():
    assert _EMITTED_SHARD_COLUMNS[0] == "game_id"
    assert "init_x" in _EMITTED_SHARD_COLUMNS and "target_y" in _EMITTED_SHARD_COLUMNS
    assert isinstance(_SHARD_SCHEMA_VERSION, str)


def test_fit_models_writes_loadable_bundles(tmp_path):
    from silly_kicks.corners import CornerRoutineModel, RunZoneModel

    summary = fit_models(_snapshots(), tmp_path)
    assert summary["n_corners"] == 12
    assert summary["n_runs"] >= 1
    # both params-only bundles are written and load (chirality + integrity pass round-trip)
    zone = RunZoneModel.load(tmp_path / "run_zone")
    routine = CornerRoutineModel.load(tmp_path / "routine")
    assert zone.n_initial_zones >= 1 and zone.n_target_zones >= 1
    assert routine.rank is not None and routine.rank >= 1


def test_fit_models_stamps_training_commit(tmp_path):
    # ADR-011 provenance: a bundled artifact must trace back to a commit. The corpus run passes the
    # clean-HEAD commit; here a sentinel confirms both bundles carry it in metadata.
    import json

    fit_models(_snapshots(), tmp_path, training_commit="deadbeef")
    for sub in ("run_zone", "routine"):
        meta = json.loads((tmp_path / sub / "metadata.json").read_text(encoding="utf-8"))
        assert meta["training_commit"] == "deadbeef"


def test_routines_select_tracking_direct():
    # ADR-133 decision 10: routine building keeps a corner that tracking decided "direct" and skips one that
    # tracking decided "short", whatever the SPADL corner type says. The actions carry the staged attempt.
    from scripts.build_corner_routines import _direct_corners
    from silly_kicks.spadl import config as spc

    actions = pd.DataFrame(
        {
            "game_id": ["g1", "g1"],
            "period_id": [1, 1],
            "time_seconds": [100.0, 200.0],
            "team_id": ["t1", "t1"],
            "start_x": [104.5, 104.5],
            "start_y": [0.2, 67.8],
            "type_id": [spc.actiontype_id["corner_short"], spc.actiontype_id["corner_crossed"]],
            # the tracking verdict: the short-typed corner reached the box, the crossed-typed one did not
            "cross_attempt": pd.array([True, False], dtype="boolean"),
        }
    )
    direct = _direct_corners(actions, pd.DataFrame(), "sportec")
    assert direct["t_kick"].tolist() == [100.0]
    assert direct["corner_kind_source"].astype(str).tolist() == ["tracking"]


def test_routines_compute_the_tracking_attempt_from_frames():
    # The compute path: unstaged actions + frames. The short-typed corner's ball enters the box and the
    # crossed-typed one stays wide, so tracking keeps only the first, against its SPADL type.
    from scripts.build_corner_routines import _direct_corners
    from tests.tracking.test_cross_delivery import _actions, _frames, _path

    ball = _path((1.0, 2.0, (104.5, 0.5), (95.0, 34.0))) + _path((11.0, 12.0, (104.5, 0.5), (100.0, 3.0)))
    actions = _actions([("corner_short", 1.0, 104.5, 0.5), ("corner_crossed", 11.0, 104.5, 0.5)])
    frames = _frames(ball, {1: ("A", 104.5, 0.5), 2: ("A", 96.0, 34.0), 3: ("B", 60.0, 40.0)})
    direct = _direct_corners(actions, frames, "sportec")
    assert direct["t_kick"].tolist() == [1.0]
    assert direct["corner_kind_source"].astype(str).tolist() == ["tracking"]
