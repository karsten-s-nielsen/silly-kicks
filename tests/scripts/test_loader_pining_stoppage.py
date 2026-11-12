"""The loader adds the stoppage columns to every tracking provider's frames (scripts/_loader_pining.py; ADR-130)."""

from __future__ import annotations

import pandas as pd
import pytest

import scripts._loader_pining as lp
import silly_kicks.spadl.config as spc


def _frames(n=10):
    rows = []
    for f in range(n):
        base = {"game_id": 1, "period_id": 1, "frame_id": f, "time_seconds": f * 1.0, "ball_state": "alive"}
        rows += [{**base, "is_ball": True, "x": 50.0, "y": 30.0}, {**base, "is_ball": False, "x": 40.0, "y": 30.0}]
    return pd.DataFrame(rows)


def _actions():
    ids = [spc.actiontype_id[t] for t in ("pass", "throw_in", "pass")]
    return pd.DataFrame({"game_id": 1, "period_id": 1, "time_seconds": [1.0, 4.0, 8.0], "type_id": ids, "result_id": 1})


@pytest.mark.parametrize("provider", ["idsse", "skillcorner", "gradientsports"])
def test_build_match_adds_the_stoppage_columns(monkeypatch, provider):
    monkeypatch.setattr(lp, "_build_idsse", lambda *a, **k: (_actions(), _frames().assign(ball_state="dead"), 1))
    monkeypatch.setattr(lp, "_build_skillcorner", lambda *a, **k: (_actions(), _frames(), 1, None))
    monkeypatch.setattr(lp, "_build_gradientsports", lambda *a, **k: (_actions(), _frames(), 1))
    events = pd.DataFrame(
        {
            "event_type": ["player_possession"] * 2,
            "frame_start": [1, 6],
            "frame_end": [2, 8],
            "game_interruption_after": ["throw_in_against", None],
            "game_interruption_before": [None, "throw_in_for"],
        }
    )
    monkeypatch.setattr(lp, "_read_skillcorner_events", lambda paths: events)
    _actions_out, frames, *_ = lp._build_match(provider, "m", {}, None)
    state = frames[frames["is_ball"]].set_index("frame_id")["stoppage_state"]
    source = set(frames["stoppage_source"])
    if provider == "idsse":
        assert set(state) == {"dead"} and source == {"ball_state"}
    elif provider == "skillcorner":
        assert state[3] == "dead" and state[6] == "unknown" and state[9] == "dead"  # the marker pair; after the last
    else:
        assert state[2] == "dead" and state[5] == "live" and state[9] == "dead"  # [1 s, 4 s) dead by the restart
    assert set(frames["ball_state"]) == ({"dead"} if provider == "idsse" else {"alive"})
