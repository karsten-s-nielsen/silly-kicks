"""Structural perf guard + carrier-resolution parity for _pressure_at_entry (ADR-068).

The per-frame carrier lookup is built ONCE, not re-scanned per spell -- the O(n_spells * n_players)
the sibling `_defending_team_id` fixed. Isolates the changed loop with a stub `add_pressure` so the
guard does not depend on the heavy real pressure model."""

import pandas as pd

import silly_kicks.causal._confounders as C
from tests._perf_structural import call_counter


def _stub_add_pressure(synth, frames, methods):
    out = synth.copy()
    out["pressure_on_actor__bekkers_pi"] = 1.0
    return out


def _frames_and_spells(n_frames):
    """One (game, period); each entry frame carries a ball + two team-5 players + one team-6 player.

    Ball at x=50; team-5 player 100 at x=48 (nearest teammate), team-5 player 101 at x=40 (far),
    team-6 player 200 at x=49.5 (closest overall, but WRONG team -> must be filtered out)."""
    frows = []
    for fid in range(10, 10 + n_frames):
        frows += [
            dict(game_id=1, period_id=1, frame_id=fid, is_ball=True, x=50.0, y=34.0, player_id=pd.NA, team_id=pd.NA),
            dict(game_id=1, period_id=1, frame_id=fid, is_ball=False, x=48.0, y=34.0, player_id=100, team_id=5),
            dict(game_id=1, period_id=1, frame_id=fid, is_ball=False, x=40.0, y=34.0, player_id=101, team_id=5),
            dict(game_id=1, period_id=1, frame_id=fid, is_ball=False, x=49.5, y=34.0, player_id=200, team_id=6),
        ]
    frames = pd.DataFrame(frows)
    spells = pd.DataFrame(
        {
            "game_id": [1] * n_frames,
            "period_id": [1] * n_frames,
            "entry_frame_id": list(range(10, 10 + n_frames)),
            "possessing_team": [5] * n_frames,
            "entry_time": [float(i) for i in range(n_frames)],
        }
    )
    return frames, spells


def test_carrier_lookup_built_once_not_per_spell(monkeypatch):
    frames, spells = _frames_and_spells(n_frames=3)
    calls = call_counter(monkeypatch, C, "group_rows")
    out = C._pressure_at_entry(spells, frames, _stub_add_pressure)
    assert len(spells) >= 2  # non-vacuity: a per-spell rescan would call group_rows once PER spell
    assert calls["n"] == 1  # built ONCE, independent of spell count
    assert len(out) == len(spells)


def test_carrier_is_nearest_possessing_teammate_not_nearest_overall(monkeypatch):
    # Parity/behaviour: team filter (ids_match, dtype-safe) THEN nearest -- so the resolved carrier is
    # team-5's player 100 (x=48), NOT team-6's 200 (x=49.5, closest to the ball but the wrong team).
    frames, spells = _frames_and_spells(n_frames=1)
    seen = {}

    def _capture(synth, frames_, methods):
        seen["player_id"] = synth["player_id"].tolist()
        out = synth.copy()
        out["pressure_on_actor__bekkers_pi"] = 1.0
        return out

    C._pressure_at_entry(spells, frames, _capture)
    assert seen["player_id"] == [100]
