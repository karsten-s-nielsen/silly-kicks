"""The pining->cache writer produces a layout train_ghost_gk.py consumes (PR-S81 / NP1)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

from tests.tracking.test_ghost_gk import _make_ghost_gk_frames

_spec = importlib.util.spec_from_file_location("_loader_pining_to_cache", Path("scripts/_loader_pining_to_cache.py"))
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]


def test_write_cache_layout_is_trainer_consumable(tmp_path):
    frames = pd.concat(
        [_make_ghost_gk_frames(game_id="g1", frame_id=f, timestamp=float(f)) for f in range(1, 5)],
        ignore_index=True,
    )
    actions = pd.DataFrame({"action_id": [0], "game_id": ["g1"], "period_id": [1], "team_id": [1]})

    _mod.write_match_cache(tmp_path, provider="test", match_id="g1", frames=frames, actions=actions, home_team_id=1)

    fp = tmp_path / "test" / "g1" / "frames.parquet"
    mp = tmp_path / "test" / "g1" / "meta.json"
    assert fp.exists() and mp.exists()
    assert json.loads(mp.read_text())["home_team_id"] == 1
    cols = set(pd.read_parquet(fp).columns)
    assert {"vx", "vy", "game_id", "is_goalkeeper"} <= cols  # trainer-required schema
    # the trainer's discovery glob finds it:
    assert list(tmp_path.glob("**/frames.parquet")) == [fp]
