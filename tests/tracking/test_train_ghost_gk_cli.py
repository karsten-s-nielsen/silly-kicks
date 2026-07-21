"""Trainer CLI wires one carrier cp into prepare + fit (PR-S81 / N1 / NP3).

The recorded==used invariant is unit-tested at the model level (test_ghost_gk_r3)
and the prepare level; this guards the place it is actually WIRED -- the trainer
passing one cp to both prepare and fit -- end to end through the CLI.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS
from tests.tracking.test_ghost_gk import _make_ghost_gk_frames


def _tiny_cache(root: Path, n_games: int = 2) -> None:
    for g in range(n_games):
        gid = f"{100 + g}"
        gdir = root / "test" / gid
        gdir.mkdir(parents=True, exist_ok=True)
        frames = pd.concat(
            [_make_ghost_gk_frames(game_id=gid, frame_id=f, timestamp=float(f)) for f in range(1, 40)],
            ignore_index=True,
        )
        frames.to_parquet(gdir / "frames.parquet")
        (gdir / "meta.json").write_text(json.dumps({"home_team_id": 1}))


def _run(tmp_path: Path, *extra: str) -> dict:
    data = tmp_path / "cache"
    data.mkdir()
    _tiny_cache(data)
    out = tmp_path / "out"
    proc = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "scripts/train_ghost_gk.py",
            "--data-dir",
            str(data),
            "--output-dir",
            str(out),
            "--n-estimators",
            "10",
            "--cv-folds",
            "2",
            *extra,
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads((out / "ghost_gk_v1" / "metadata.json").read_text())


@pytest.mark.slow
def test_cli_records_supplied_carrier_params(tmp_path):
    meta = _run(tmp_path, "--carrier-beta", "0.9", "--carrier-gamma", "0.3", "--carrier-tolerance", "2.5")
    assert meta["carrier_params"] == {"tolerance_m": 2.5, "beta": 0.9, "gamma": 0.3}
    assert meta["version"] == "1.3.0"  # Option A artifact format (gk_y ensemble + baselines)
    assert meta["serve_estimator"] == "boosted_mean"


@pytest.mark.slow
def test_cli_omitted_records_library_default(tmp_path):
    meta = _run(tmp_path)
    assert meta["carrier_params"] == dict(DEFAULT_CARRIER_PARAMS)
