"""validate_gk_position_clamp: detect a provider that clamps the GK to a hard zone (e.g. Gradient
Sports pins the keeper at 27.5 m from goal). The SIGNATURE is a spike (pileup) at the ceiling, not the
ceiling value itself -- so a natural keeper that merely REACHES the same max (one sweep) must NOT fire,
and a clamped keeper (many frames pinned at the max) must. Both sides are asserted.
"""

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import GkClampDiagnosis, GoalkeeperClampWarning, validate_gk_position_clamp


def _frames(keeper_x, *, game="g1", team="t1"):
    """Frames with one keeper trajectory + a ball row and an outfielder pinned AT 27.5 m (which the
    detector must IGNORE -- the clamp is a keeper property, and the ball/outfield rows prove exclusion)."""
    keeper_x = np.asarray(keeper_x, dtype=float)
    n = len(keeper_x)
    keeper = pd.DataFrame(
        {
            "game_id": [game] * n,
            "period_id": [1] * n,
            "team_id": [team] * n,
            "player_id": [1] * n,
            "is_ball": [False] * n,
            "is_goalkeeper": [True] * n,
            "x": keeper_x,
            "y": [34.0] * n,
        }
    )
    noise = pd.DataFrame(
        {
            "game_id": [game, game],
            "period_id": [1, 1],
            "team_id": [pd.NA, team],
            "player_id": [pd.NA, 2],
            "is_ball": [True, False],
            "is_goalkeeper": [False, False],
            "x": [52.5, 27.5],
            "y": [34.0, 34.0],
        }
    )
    return pd.concat([keeper, noise], ignore_index=True)


# 250 frames on the line + 50 pinned at the 27.5 m ceiling == the Gradient Sports clamp signature.
_CLAMPED = np.concatenate([np.full(250, 10.0), np.full(50, 27.5)])
# Same MAX (a single 45 m sweep), no pileup == a natural keeper.
_NATURAL = np.concatenate([np.full(299, 10.0), [45.0]])


def test_detects_clamp_signature():
    d = validate_gk_position_clamp(_frames(_CLAMPED), warn=False)
    assert isinstance(d, GkClampDiagnosis)
    assert d.clamped and d.n_units == 1
    (unit,) = d.clamped_units
    assert unit == ("g1", "t1")  # canonical-id keys
    assert abs(d.ceiling_by_unit[unit] - 27.5) < 0.2
    assert d.pileup_by_unit[unit] > 0.1  # 50/300


def test_natural_keeper_reaching_the_same_max_does_not_fire():
    # The other side: a keeper that REACHES a high max with a single sweep is NOT clamped. This is the
    # non-vacuity guard -- a detector that fired on "max is high" would flag every sweeping keeper.
    d = validate_gk_position_clamp(_frames(_NATURAL), warn=False)
    assert not d.clamped and d.n_units == 1 and d.clamped_units == ()


def test_pileup_not_the_max_is_the_signal():
    # Same max on both legs (45 m). Only the pileup differs -> only the clamped leg fires.
    pile = np.concatenate([np.full(250, 10.0), np.full(50, 45.0)])
    assert not validate_gk_position_clamp(_frames(_NATURAL), warn=False).clamped
    assert validate_gk_position_clamp(_frames(pile), warn=False).clamped


def test_orientation_free_detects_clamp_at_the_far_goal():
    # min(x, 105-x): a keeper clamped near the x=105 goal is detected identically (no goal_map needed).
    d = validate_gk_position_clamp(_frames(105.0 - _CLAMPED), warn=False)
    assert d.clamped and abs(d.ceiling_by_unit[("g1", "t1")] - 27.5) < 0.2


def test_warns_by_default_on_clamp():
    with pytest.warns(GoalkeeperClampWarning, match="27.5"):
        validate_gk_position_clamp(_frames(_CLAMPED))


def test_no_warning_when_disabled(recwarn):
    validate_gk_position_clamp(_frames(_CLAMPED), warn=False)
    assert not any(isinstance(w.message, GoalkeeperClampWarning) for w in recwarn)


def test_no_warning_when_not_clamped(recwarn):
    validate_gk_position_clamp(_frames(_NATURAL))  # warn defaults True, but nothing is clamped
    assert not any(isinstance(w.message, GoalkeeperClampWarning) for w in recwarn)


def test_too_few_keeper_frames_are_skipped():
    d = validate_gk_position_clamp(_frames(np.full(50, 27.5)), warn=False)  # 50 < min_keeper_frames
    assert not d.clamped and d.n_units == 0


def test_empty_and_missing_columns_never_crash():
    assert not validate_gk_position_clamp(pd.DataFrame(), warn=False).clamped
    assert not validate_gk_position_clamp(pd.DataFrame({"x": [1.0]}), warn=False).clamped


def test_no_keeper_rows_returns_unclamped():
    frames = _frames(_CLAMPED)
    frames["is_goalkeeper"] = False
    d = validate_gk_position_clamp(frames, warn=False)
    assert not d.clamped and d.n_units == 0


def test_na_keeper_flag_is_false_not_a_crash():
    # is_goalkeeper carrying <NA> (nullable boolean) resolves to False, never a raise.
    frames = _frames(_CLAMPED)
    frames["is_goalkeeper"] = frames["is_goalkeeper"].astype("boolean")
    frames.loc[0, "is_goalkeeper"] = pd.NA
    d = validate_gk_position_clamp(frames, warn=False)
    assert d.clamped  # the remaining keeper rows still carry the clamp
