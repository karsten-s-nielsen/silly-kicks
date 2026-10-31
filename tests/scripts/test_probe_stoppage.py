"""TF-46 Task 5: the xcross_probe build route -- repackage a staged FLAT Gradient Sports probe corpus to the
per-game layout, carrying the ``events_v3`` stoppage through, and FAIL LOUD (O6) on a match with no restart/goal.

The real build materializes from pining (a Commit-2 owner op); these cases exercise the repackage + the O6
guard on a hand-built flat corpus, so no pining is needed."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.build_xcross_probe import ProbeBuildError, assert_probe_units_have_spans, build_probe_pergame
from silly_kicks.tracking import STOPPAGE_SOURCES

_EVENTS_V3 = STOPPAGE_SOURCES[1]
_GEN = "gen1"


def _frames(game: str, n: int = 6) -> pd.DataFrame:
    rows = []
    for fid in range(n):
        for is_ball, pid, team in ((True, pd.NA, pd.NA), (False, 1, "T1"), (False, 2, "T2")):
            rows.append(
                {
                    "game_id": game,
                    "period_id": 1,
                    "frame_id": 1000 + fid,
                    "time_seconds": fid * 0.1,
                    "player_id": pid,
                    "team_id": team,
                    "is_ball": is_ball,
                    "x": 50.0 + fid,
                    "y": 34.0,
                    "visibility": True,
                    # TF-46: the event-only staged shard carries the stoppage columns (tc3-frames-4)
                    "stoppage_state": "live",
                    "stoppage_source": _EVENTS_V3,
                }
            )
    return pd.DataFrame(rows)


def _actions(game: str, *, restart: bool) -> pd.DataFrame:
    # restart=True: one corner (a restart event -> a dead-ball span); restart=False: a lone pass (no span)
    type_name = "corner_crossed" if restart else "pass"
    return pd.DataFrame(
        {
            "game_id": [game],
            "period_id": [1],
            "action_id": [0],
            "type_name": [type_name],
            "result_name": ["success"],
            "time_seconds": [1.0],
            "team_id": ["T1"],
            "player_id": [1],
            "start_x": [50.0],
            "start_y": [34.0],
            "end_x": [52.0],
            "end_y": [34.0],
        }
    )


def _write_flat_unit(flat, game: str, *, restart: bool) -> None:
    stem = f"gradientsports__{game}"
    shards = flat / "shards" / _GEN
    shards.mkdir(parents=True, exist_ok=True)
    (flat / "_actions").mkdir(parents=True, exist_ok=True)
    _frames(game).to_parquet(shards / f"{stem}.parquet")
    _actions(game, restart=restart).to_parquet(flat / "_actions" / f"{stem}.parquet")


def test_probe_build_preserves_stoppage_columns(tmp_path):
    """The build REPACKAGES a staged flat corpus; it does NOT attach stoppage -- that is the loader's job,
    pinned in Task 1 (``test_loader_stoppage_attach``). This pins that the repackage PRESERVES the staged
    stoppage columns (B06: the old name ``attaches_events_v3`` overclaimed and could not fail for the
    behaviour it named). The flat shard carries ``stoppage_state``/``stoppage_source``; the per-game
    ``frames.parquet`` must carry them unchanged, never dropped."""
    flat, out = tmp_path / "flat", tmp_path / "out"
    _write_flat_unit(flat, "10502", restart=True)
    build_probe_pergame(flat, out)
    frames = pd.read_parquet(out / "gradientsports-10502" / "frames.parquet")
    assert "stoppage_state" in frames.columns and "stoppage_source" in frames.columns  # preserved, not dropped
    assert set(frames["stoppage_source"].astype(str).unique()) == {_EVENTS_V3}
    assert set(frames["stoppage_state"].astype(str).unique()) == {"live"}


def test_probe_units_have_actions_parquet(tmp_path):
    flat, out = tmp_path / "flat", tmp_path / "out"
    _write_flat_unit(flat, "10502", restart=True)
    _write_flat_unit(flat, "10503", restart=True)
    n = build_probe_pergame(flat, out)
    assert n == 2
    for game in ("10502", "10503"):
        unit = out / f"gradientsports-{game}"
        assert (unit / "frames.parquet").exists()
        assert (unit / "actions.parquet").exists()


def test_probe_stage_no_longer_raises(tmp_path):
    # the valid probe (restart present) passes the O6 guard end to end -- no raise, every unit kept
    flat, out = tmp_path / "flat", tmp_path / "out"
    _write_flat_unit(flat, "10502", restart=True)
    assert build_probe_pergame(flat, out) == 1
    assert_probe_units_have_spans(out)  # idempotent: still clean


def test_probe_gs_without_restart_goal_actions_fails_loud(tmp_path):
    # O6: a GS probe match with only a pass -> events_v3 has no span -> the build refuses, naming the unit
    flat, out = tmp_path / "flat", tmp_path / "out"
    _write_flat_unit(flat, "10502", restart=False)
    with pytest.raises(ProbeBuildError, match="no restart/goal"):
        build_probe_pergame(flat, out)
