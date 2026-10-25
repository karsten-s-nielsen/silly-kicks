"""CI-safe unit slice for scripts/build_corner_routines.py (the pure fit kernel; corpus pass is e2e)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.build_corner_routines import (
    _EMITTED_SHARD_COLUMNS,
    _SHARD_SCHEMA_VERSION,
    fit_models,
    load_extra_tier,
    refit_without_course_decision,
)


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


def test_refit_without_course_decision_both_sides():
    # run (16) pre-registered rule: refit without the course tier only when BOTH models agree >= 0.95
    assert refit_without_course_decision(0.97, 0.96) is True
    assert refit_without_course_decision(0.97, 0.90) is False  # routine below the bar
    assert refit_without_course_decision(0.90, 0.99) is False  # zone below the bar


def test_load_extra_tier_requires_revision_and_refuses_parquet(tmp_path):
    with pytest.raises(ValueError, match="revision"):
        load_extra_tier(tmp_path, "")
    (tmp_path / "actions.parquet").write_bytes(b"not-really")  # a pre-converted actions dir is refused
    with pytest.raises(ValueError, match="parquet"):
        load_extra_tier(tmp_path, "rev1")


def test_load_extra_tier_converts_raw_files_through_the_library(tmp_path):
    # C-12 / C2-IMPL-01: each course-tier match directory converts IN-RUN (GI events + match.json + tracking)
    # through spadl.skillcorner + the native tracking builder; nothing pre-converted is accepted.
    import json
    import shutil

    src = Path("tests/datasets/skillcorner/opendata")
    mdir = tmp_path / "1886347"
    mdir.mkdir()
    shutil.copy(src / "1886347_match.json", mdir / "match.json")
    shutil.copy(src / "1886347_events_slim.csv", mdir / "dynamic_events.csv")
    meta = json.loads((mdir / "match.json").read_text(encoding="utf-8"))
    pids = [p["id"] for p in meta["players"]][:4]
    with (mdir / "tracking_extrapolated.jsonl").open("w", encoding="utf-8") as fh:
        for f in range(1, 11):
            fh.write(
                json.dumps(
                    {
                        "period": 1,
                        "frame": f,
                        "timestamp": f"00:00:{f / 10:04.1f}",
                        "player_data": [
                            {"player_id": pid, "x": -10.0 + i, "y": 1.0 * i, "is_detected": True}
                            for i, pid in enumerate(pids)
                        ],
                        "ball_data": {"x": 0.0, "y": 0.0, "z": 0.1, "is_detected": True},
                    }
                )
                + "\n"
            )
    items, sidecar = load_extra_tier(tmp_path, "rev1")
    assert len(items) == 1 and items[0].provider == "course" and items[0].match_id == "1886347"
    assert not items[0].actions.empty and items[0].frames is not None and not items[0].frames.empty
    assert (
        sidecar["n_matches"] == 1 and sidecar["extra_tier_revision"] == "rev1" and sidecar["data_profile"] == "course"
    )


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
