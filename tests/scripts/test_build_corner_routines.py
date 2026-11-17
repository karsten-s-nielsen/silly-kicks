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


def test_snapshot_rows_give_no_row_for_a_corner_with_no_window(monkeypatch):
    # Supervisor C (bar A3): an unmeasured corner has no window; argmin of an all-NaN distance took the period's first
    # frame for both snapshots, so the corner gave rows with start == target into the corner-model build
    from types import SimpleNamespace

    from scripts.build_corner_routines import _snapshot_rows
    from tests.scripts._corner_no_window import frames, stub_corner_pipeline

    stub_corner_pipeline(monkeypatch)
    out = _snapshot_rows(SimpleNamespace(actions=pd.DataFrame(), frames=frames(), provider="skillcorner"))
    assert set(out["corner_id"]) == {0}  # the measured corner keeps its rows (non-vacuity); corner 1 gives none
    assert len(out) == 2


def test_snapshot_rows_give_no_row_for_a_measured_corner_with_no_side(monkeypatch):
    # a kick time but no corner_side: str(<NA>) is the text "None", which _reflect_y treated as a right-flag corner
    from types import SimpleNamespace

    from scripts.build_corner_routines import _snapshot_rows
    from tests.scripts._corner_no_window import frames, stub_corner_pipeline, windowed_corners

    table = windowed_corners()
    table["corner_side"] = pd.Series([pd.NA, pd.NA], dtype="category")
    stub_corner_pipeline(monkeypatch, table)
    out = _snapshot_rows(SimpleNamespace(actions=pd.DataFrame(), frames=frames(), provider="skillcorner"))
    assert len(out) == 0
