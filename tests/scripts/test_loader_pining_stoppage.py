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
    # Every builder emits `time_source` (action-time repair spec 4.1); the loader's time step refuses a table without
    # it. "tracking" on every provider: with no "event" row the step runs no ELASTIC on these toy frames (this test
    # checks the stoppage columns, not the time step).
    ids = [spc.actiontype_id[t] for t in ("pass", "throw_in", "pass")]
    return pd.DataFrame(
        {"game_id": 1, "original_event_id": ["e0", "e1", "e2"], "action_id": [0, 1, 2], "period_id": 1,
         "time_seconds": [1.0, 4.0, 8.0], "team_id": 1, "player_id": 10, "start_x": 50.0, "start_y": 34.0,
         "end_x": 55.0, "end_y": 34.0, "type_id": ids, "result_id": 1, "bodypart_id": 0,
         "action_provenance": "native", "time_source": "tracking"}
    )  # fmt: skip


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
        # the marker pair is dead; R44: the first possession after it (b, from frame 6) is live; after the last
        assert state[3] == "dead" and state[6] == "live" and state[9] == "dead"
    else:
        assert state[2] == "dead" and state[5] == "live" and state[9] == "dead"  # [1 s, 4 s) dead by the restart
    assert set(frames["ball_state"]) == ({"dead"} if provider == "idsse" else {"alive"})


def test_skillcorner_build_passes_each_period_last_frame_to_the_spans(monkeypatch):
    # R71 / A07 wiring: the loader hands the spans each period's last frame (from the frames), so a period-end
    # penalty's span stops at its period's end; without it the span runs on to the next period's first possession
    import silly_kicks.tracking._stoppages as st

    frames = pd.concat([_frames(), _frames().assign(period_id=2, frame_id=lambda f: f["frame_id"] + 100)])
    monkeypatch.setattr(lp, "_build_skillcorner", lambda *a, **k: (_actions(), frames, 1, None))
    events = pd.DataFrame({"event_type": ["player_possession"], "frame_start": [1], "frame_end": [2], "period": [1],
                           "game_interruption_after": [None], "game_interruption_before": [None]})  # fmt: skip
    monkeypatch.setattr(lp, "_read_skillcorner_events", lambda paths: events)
    seen = {}
    real = st.skillcorner_stoppage_spans

    def spy(events, **kw):
        seen.update(kw)
        return real(events, **kw)

    monkeypatch.setattr(st, "skillcorner_stoppage_spans", spy)
    lp._build_match("skillcorner", "m", {}, None)
    assert seen.get("period_last_frame") == {1: 9, 2: 109}
