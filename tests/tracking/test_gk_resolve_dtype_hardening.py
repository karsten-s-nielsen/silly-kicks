"""Part C: GK-resolver team-join dtype hardening.

The raw `==` at the shared `_gk_from_frames_linked` team predicate is dtype-fragile. `acting_gk_from_frames`
is PROTECTED by its roster-identity fallback (a cross-dtype miss still resolves), so the observable defect is
in `defending_gk_from_frames` (pure linked path): on a cross-dtype team mismatch the opposing branch
(`~match_team` over an all-False raw compare) returns the ACTING team's OWN keeper instead of the opponent.
The fix is per-branch — `ids_equal` (acting) / `ids_differ` (defending) — dtype-safe (ADR-019), byte-identical
on matched/NA paths, and it makes cross-dtype defending pick the true opponent.
"""

import numpy as np
import pandas as pd

from silly_kicks.tracking import acting_gk_from_frames, defending_gk_from_frames

_PASS = 0


def _frow(pid, team, gk, t, *, x=50.0):
    return dict(
        game_id=1,
        period_id=1,
        frame_id=round(t * 25),
        time_seconds=t,
        frame_rate=25.0,
        player_id=pid,
        team_id=team,
        is_ball=False,
        is_goalkeeper=gk,
        x=float(x),
        y=34.0,
        z=0.0,
        speed=1.0,
        vx=0.0,
        vy=0.0,
        speed_source="native",
        ball_state="alive",
        team_attacking_direction="ltr",
        confidence=None,
        visibility=None,
        source_provider="gradientsports",
        is_goalkeeper_source="native",
    )


def _frames(rows):
    # GS SPADL id contract: Int64 team/player ids (the action-side dtype is what varies in these tests).
    f = pd.DataFrame(rows)
    f["player_id"] = f["player_id"].astype("Int64")
    f["team_id"] = f["team_id"].astype("Int64")
    return f


# GK team 5 = player 1, GK team 6 = player 2, detected at t~10 and t~20.
def _two_keeper_frames():
    rows = []
    for t in (9.9, 10.0, 19.9, 20.0):
        rows += [_frow(1, 5, True, t), _frow(2, 6, True, t, x=100.0)]
    return _frames(rows)


def _action(team_id, t, *, team_dtype=None):
    df = pd.DataFrame(
        [
            dict(
                game_id=1,
                action_id=0,
                period_id=1,
                time_seconds=t,
                team_id=team_id,
                player_id=99,
                type_id=_PASS,
                result_id=1,
            )
        ]
    )
    if team_dtype:
        df["team_id"] = df["team_id"].astype(team_dtype)
    return df


def test_byte_identity_matched_nan_and_int64na():
    """Anchor: matched-dtype + NA paths must be byte-identical pre/post the fix (non-vacuous — includes the
    ~match_team NaN branch that the fix touches). Golden pinned from current code (Step 2 confirmed)."""
    f = _two_keeper_frames()

    # matched dtype (int action team 5): defending -> opponent GK player 2; acting -> own GK player 1
    a_matched = _action(5, 10.0)
    assert defending_gk_from_frames(a_matched, f).tolist() == [2]
    assert acting_gk_from_frames(a_matched, f).tolist() == [1]

    # float-NaN team on the DEFENDING path -> unresolved -> NaN (unknown acting team => no opponent).
    # The fix must PRESERVE this NaN (ids_differ: NA -> not-differ -> False -> empty pick). NaN != NaN -> isna.
    assert defending_gk_from_frames(_action(np.nan, 20.0, team_dtype="float64"), f).isna().all()

    # nullable-Int64 NA team: no raise, unresolved -> NaN (both resolvers), byte-identical.
    a_na = _action(pd.NA, 10.0, team_dtype="Int64")
    assert defending_gk_from_frames(a_na, f).isna().all()
    assert acting_gk_from_frames(a_na, f).isna().all()


def test_mismatched_dtype_defending_now_returns_correct_opponent():
    """RED-first: current code returns [1] (the acting team's OWN keeper) on a str-vs-Int64 mismatch; the
    fix must return [2] (the true opponent, team 6)."""
    f = _two_keeper_frames()  # Int64 team ids
    a = _action("5", 10.0, team_dtype="object")  # string action team_id
    assert defending_gk_from_frames(a, f).tolist() == [2]


def test_mismatched_dtype_acting_unaffected():
    """Acting is fallback-protected: a str-vs-Int64 mismatch still resolves to the acting GK (player 1),
    before and after the fix (byte-identity on the acting path)."""
    f = _two_keeper_frames()
    a = _action("5", 10.0, team_dtype="object")
    assert acting_gk_from_frames(a, f).tolist() == [1]
