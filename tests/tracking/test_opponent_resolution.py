"""Shared opponents_within core (TF-51 v2, N6): the ONE nearest-opponent producer.

Returns the within-threshold opponents SORTED ascending by distance (serves all three of
resolve_responsible_defenders's modes + the press-commitment cue). GK exclusion is the caller's
choice, never baked into the core.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._opponent_resolution import opponents_within


def _frame(rows):
    common = dict(game_id="g1", period_id=1, frame_id=1, time_seconds=1.0, source_provider="test")
    return pd.DataFrame([{**common, **r} for r in rows])


def _opp(x, y, *, player_id, team_id=20, is_gk=False, is_ball=False):
    return dict(x=x, y=y, player_id=player_id, team_id=team_id, is_goalkeeper=is_gk, is_ball=is_ball)


def test_returns_within_threshold_sorted_by_distance():
    fr = _frame(
        [
            _opp(96.0, 34.0, player_id=901),  # 1 m
            _opp(97.0, 34.0, player_id=902),  # 2 m
            _opp(80.0, 34.0, player_id=903),  # 15 m (outside)
            dict(x=95.0, y=34.0, player_id=1, team_id=10, is_goalkeeper=False, is_ball=False),  # acting team
            dict(x=90.0, y=34.0, player_id=np.nan, team_id=np.nan, is_goalkeeper=False, is_ball=True),  # ball
        ]
    )
    out = opponents_within(fr, anchor_x=95.0, anchor_y=34.0, acting_team_id=10, threshold_m=4.5, flip=False)
    assert list(out["player_id"]) == [901, 902]  # sorted ascending; 903 dropped, acting/ball excluded
    assert list(out.columns) == ["player_id", "team_id", "distance_m"]
    assert out["distance_m"].iloc[0] == 1.0


def test_empty_when_none_within_threshold():
    fr = _frame([_opp(80.0, 34.0, player_id=901)])  # 15 m from (95,34)
    out = opponents_within(fr, anchor_x=95.0, anchor_y=34.0, acting_team_id=10, threshold_m=4.5, flip=False)
    assert out.empty


def test_exclude_goalkeeper_is_the_callers_choice():
    fr = _frame([_opp(96.0, 34.0, player_id=901, is_gk=True), _opp(97.0, 34.0, player_id=902)])
    kept = opponents_within(fr, anchor_x=95.0, anchor_y=34.0, acting_team_id=10, threshold_m=4.5, flip=False)
    assert list(kept["player_id"]) == [901, 902]  # GK kept by default (a keeper can press)
    dropped = opponents_within(
        fr, anchor_x=95.0, anchor_y=34.0, acting_team_id=10, threshold_m=4.5, flip=False, exclude_goalkeeper=True
    )
    assert list(dropped["player_id"]) == [902]  # GK excluded on request


def test_flip_reprojects_to_action_ltr():
    # an away action: frame coords reflect to action-LTR via (105-x, 68-y). A defender at frame (9,48)
    # reprojects to (96,20); with the anchor at action-LTR (95,20) it is 1 m away.
    fr = _frame([_opp(9.0, 48.0, player_id=901)])
    out = opponents_within(fr, anchor_x=95.0, anchor_y=20.0, acting_team_id=10, threshold_m=4.5, flip=True)
    assert list(out["player_id"]) == [901]
    assert out["distance_m"].iloc[0] == pytest.approx(1.0)
