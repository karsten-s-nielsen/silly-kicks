"""Unit tests for the canonical action-LTR re-projection helper (ADR-028)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._action_orientation import (
    FIELD_LENGTH,
    FIELD_WIDTH,
    acting_team_attacks_rtl,
    reproject_to_action_ltr,
)


def _frames(home_dir="ltr", away_dir="rtl"):
    # home team=1 (ltr), away team=2 (rtl), one period, one frame
    return pd.DataFrame(
        [
            dict(game_id=1, period_id=1, frame_id=10, team_id=1, is_ball=False, team_attacking_direction=home_dir),
            dict(game_id=1, period_id=1, frame_id=10, team_id=2, is_ball=False, team_attacking_direction=away_dir),
            dict(game_id=1, period_id=1, frame_id=10, team_id=np.nan, is_ball=True, team_attacking_direction=None),
        ]
    )


def test_acting_team_attacks_rtl_home_false_away_true():
    actions = pd.DataFrame(
        [
            dict(game_id=1, period_id=1, action_id=0, team_id=1),  # home → not rtl
            dict(game_id=1, period_id=1, action_id=1, team_id=2),  # away → rtl
        ]
    )
    flip = acting_team_attacks_rtl(actions, _frames())
    assert flip.tolist() == [False, True]


def test_acting_team_unknown_direction_is_NA_not_False():
    """An acting team absent from the frames resolves to ``<NA>``, never ``False``.

    Until 4.80.0 this returned ``False`` (no flip). ADR-028 D2 had already added the warning, but
    the VALUE was still indistinguishable from a healthy resolved left-to-right action, so a
    consumer that does not inspect warnings -- which is most of them -- could not tell a
    mis-keyed join from a correct one. ADR-051 D3 moved the signal into the value: ``<NA>`` means
    "not resolved", and the consumer has to decide. ``.fillna(False)`` is still a legitimate
    answer at many sites; it just has to be WRITTEN.

    The contrast is asserted in the SAME test on purpose. Checking only that the unknown team is
    ``<NA>`` would pass just as well if every action returned ``<NA>``, which would make the
    helper useless while looking strict.
    """
    from silly_kicks.tracking import OrientationUnresolvedWarning

    actions = pd.DataFrame([dict(game_id=1, period_id=1, action_id=0, team_id=999)])
    with pytest.warns(OrientationUnresolvedWarning):
        flip = acting_team_attacks_rtl(actions, _frames())
    assert str(flip.dtype) == "boolean", "must be the NULLABLE dtype, or <NA> cannot be carried"
    assert flip.isna().tolist() == [True]

    # ... and a RESOLVED left-to-right action is False, which is what <NA> must not collapse to.
    known = pd.DataFrame([dict(game_id=1, period_id=1, action_id=0, team_id=1)])
    resolved = acting_team_attacks_rtl(known, _frames())
    assert resolved.isna().tolist() == [False]
    assert resolved.tolist() == [False]


def test_reproject_flips_only_marked_rows_both_axes():
    df = pd.DataFrame({"x": [10.0, 10.0], "y": [20.0, 20.0]})
    flip = pd.Series([False, True])
    out = reproject_to_action_ltr(df, flip, x_cols=["x"], y_cols=["y"])
    assert out["x"].tolist() == [10.0, FIELD_LENGTH - 10.0]
    assert out["y"].tolist() == [20.0, FIELD_WIDTH - 20.0]


def test_reproject_preserves_nan():
    df = pd.DataFrame({"x": [np.nan], "y": [np.nan]})
    out = reproject_to_action_ltr(df, pd.Series([True]), x_cols=["x"], y_cols=["y"])
    assert out["x"].isna().all() and out["y"].isna().all()
