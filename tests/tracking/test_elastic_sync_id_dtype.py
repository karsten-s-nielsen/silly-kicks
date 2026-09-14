"""ELASTIC-NW alignment must resolve the actor-membership gate regardless of id dtype (ADR-019).

The actor-membership hard gate (``actor in candidate.players``) joins the action's ``player_id`` to
the frames' player ids. This is the exact seam of the historical constant-0.6 collapse: a FLOAT id
column -- which a concat produces whenever a frame set carries an NA id, i.e. every ball row --
rendered ``"10.0"`` on one side and ``"10"`` on the other, so every membership check missed. Under
the greedy algorithm that surfaced as a constant 0.6 confidence; under NW a total miss surfaces as
an EMPTY alignment (every score 0 < min_confidence). Both are "a plausible result from a computation
that did not happen".

These tests feed one physical scene under four id dtypes on each side and require BYTE-IDENTICAL,
non-empty output -- the property a raw ``==`` / ``astype(str)`` join cannot satisfy.
"""

from __future__ import annotations

import pandas as pd
import pytest

from silly_kicks.tracking._elastic_sync import align_events_to_frames

_DTYPES = ["python_int", "float", "Int64", "string"]


def _alignable():
    """A scene the NW aligns: four team-1 players (numeric ids) at fixed x; the ball rests at each
    in turn and hops between them; three passes at the kick frames."""
    xs = {10: 20.0, 11: 45.0, 12: 70.0, 13: 90.0}

    def ball_x(f: int) -> float:
        if f <= 10:
            return 20.0
        if f <= 20:
            return 20.0 + 2.5 * (f - 10)
        if f <= 30:
            return 45.0
        if f <= 40:
            return 45.0 + 2.5 * (f - 30)
        if f <= 50:
            return 70.0
        if f <= 60:
            return 70.0 + 2.0 * (f - 50)
        return 90.0

    rows = []
    for f in range(71):
        t = f / 25.0
        rows.append(
            {
                "game_id": 7,
                "period_id": 1,
                "frame_id": f,
                "time_seconds": t,
                "player_id": None,
                "team_id": None,
                "x": ball_x(f),
                "y": 34.0,
                "z": float("nan"),
                "is_ball": True,
            }
        )
        for pid, x in xs.items():
            rows.append(
                {
                    "game_id": 7,
                    "period_id": 1,
                    "frame_id": f,
                    "time_seconds": t,
                    "player_id": pid,
                    "team_id": 1,
                    "x": x,
                    "y": 34.0,
                    "z": float("nan"),
                    "is_ball": False,
                }
            )
    frames = pd.DataFrame(rows)
    actions = pd.DataFrame(
        {
            "action_id": [0, 1, 2],
            "game_id": [7, 7, 7],
            "period_id": [1, 1, 1],
            "time_seconds": [0.4, 1.2, 2.0],
            "player_id": [10, 11, 12],
            "team_id": [1, 1, 1],
            "type_id": [0, 0, 0],
            "type_name": ["pass"] * 3,
            "result_name": ["success"] * 3,
        }
    )
    return frames, actions


def _cast_ids(df: pd.DataFrame, cols, dtype: str) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if dtype == "float":
            out[col] = out[col].astype("float64")
        elif dtype == "Int64":
            out[col] = out[col].astype("Int64")
        elif dtype == "string":
            out[col] = out[col].map(lambda v: v if pd.isna(v) else str(int(v)))
        # "python_int": leave as-is (object column with python ints + None on ball rows)
    return out


def _reference() -> pd.DataFrame:
    frames, actions = _alignable()
    return align_events_to_frames(actions, frames).reset_index(drop=True)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_invariant_to_frame_player_id_dtype(dtype):
    """Vary the FRAMES' player/team id dtype; the alignment is byte-identical and non-empty."""
    frames, actions = _alignable()
    frames = _cast_ids(frames, ["player_id", "team_id"], dtype)
    result = align_events_to_frames(actions, frames).reset_index(drop=True)
    assert len(result) > 0, f"frame id dtype {dtype!r} produced NO matches -- the membership join missed"
    pd.testing.assert_frame_equal(result, _reference())


@pytest.mark.parametrize("dtype", _DTYPES)
def test_invariant_to_action_player_id_dtype(dtype):
    """Vary the ACTIONS' player/team id dtype; the alignment is byte-identical and non-empty."""
    frames, actions = _alignable()
    actions = _cast_ids(actions, ["player_id", "team_id"], dtype)
    result = align_events_to_frames(actions, frames).reset_index(drop=True)
    assert len(result) > 0, f"action id dtype {dtype!r} produced NO matches -- the membership join missed"
    pd.testing.assert_frame_equal(result, _reference())


def test_no_constant_confidence_collapse():
    """The behavioural anti-regression: real, varying confidences -- not a single fabricated value
    (the greedy 0.6) nor an all-NaN/empty result (the NW manifestation of a total membership miss)."""
    frames, actions = _alignable()
    # the exact historical trigger: float id columns (ball NA upcasts int -> float -> "10.0")
    frames = _cast_ids(frames, ["player_id", "team_id"], "float")
    result = align_events_to_frames(actions, frames)
    conf = result["elastic_confidence"].dropna()
    assert len(conf) > 0
    assert (conf > 0.0).all() and (conf <= 1.0).all()
