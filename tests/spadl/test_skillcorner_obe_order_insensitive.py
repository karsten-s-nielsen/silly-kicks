"""SkillCorner defensive-action OBE matching is order-insensitive (ADR-065).

`infer_defensive_actions` upgrades a defensive `player_possession` action to a `tackle` when a
same-team `direct_regain` `on_ball_engagement` (OBE) row falls within a +/-2s window, attributing the
tackle to the NEAREST such OBE. On a tie (two equidistant candidates) the pick must be a pure function
of content, NOT of the OBE frame's input row order -- a positional ``argmin()`` is first-on-tie and so
flips the attributed player/team/coords when the input is permuted. This guards the content-keyed
tiebreak (nearest time, then ``event_id``).
"""

from __future__ import annotations

import pandas as pd

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl._skillcorner_inference import infer_defensive_actions


def _pp() -> pd.DataFrame:
    # One defensive action (start_type "recovery") by team_home at t=30.0.
    return pd.DataFrame(
        [
            {
                "event_id": "pp_def",
                "period": 1,
                "time_seconds": 30.0,
                "team_id": "team_home",
                "player_id": "p_actor",
                "start_type": "recovery",
                "x_start": 40.0,
                "y_start": 34.0,
            }
        ]
    )


def _obe() -> pd.DataFrame:
    # Two same-team direct_regain OBE rows EQUIDISTANT from t=30.0 (29.0 and 31.0), distinct players
    # and event_ids -> a genuine tie the pick must resolve by content (event_id), not row order.
    return pd.DataFrame(
        [
            {
                "event_id": "obe_A",
                "period": 1,
                "time_seconds": 29.0,
                "team_id": "team_home",
                "player_id": "keeper_A",
                "end_type": "direct_regain",
                "x_start": 41.0,
                "y_start": 35.0,
            },
            {
                "event_id": "obe_B",
                "period": 1,
                "time_seconds": 31.0,
                "team_id": "team_home",
                "player_id": "keeper_B",
                "end_type": "direct_regain",
                "x_start": 43.0,
                "y_start": 37.0,
            },
        ]
    )


def _attribution(out: pd.DataFrame) -> tuple:
    tackle = spadlconfig.actiontype_id["tackle"]
    row = out[out["type_id"] == tackle].iloc[0]
    return (row["player_id"], row["team_id"], float(row["start_x"]), float(row["start_y"]))


def test_obe_tie_attribution_is_order_insensitive():
    pp = _pp()
    obe = _obe()
    native = infer_defensive_actions(pp, obe)
    reversed_ = infer_defensive_actions(pp, obe.iloc[::-1].reset_index(drop=True))
    # The tackle must be attributed identically regardless of OBE input row order.
    assert _attribution(native) == _attribution(reversed_), (
        f"OBE-tie attribution is order-dependent: native={_attribution(native)} reversed={_attribution(reversed_)}"
    )
    # And it resolves to the content tiebreak winner (lower event_id "obe_A" -> keeper_A).
    assert _attribution(native)[0] == "keeper_A"
