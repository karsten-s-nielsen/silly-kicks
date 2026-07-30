"""D2: the orientation seam must not fail silent (ADR-028 spec §4.2).

``acting_team_attacks_rtl`` returned an all-False flip on THREE distinct unresolvable inputs, with
no signal of any kind. An all-False flip means no ADR-028 re-projection is applied, so every
away-team action's geometry silently mixes coordinate conventions.

This is not hypothetical. Measured on one canonical y-asymmetric away action, labelled frames vs the
same frames with the direction column dropped: ``nearest_defender_distance`` 7.6158 -> 19.6977,
``receiver_zone_density`` 1 -> 0, ``defenders_in_triangle_to_goal`` 1 -> 0. And the pining loader
shipped SkillCorner frames with the column null on 100% of rows (RC4), so the whole provider's
action-coupled geometry was computed unoriented.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking._action_orientation import acting_team_attacks_rtl

PASS_ID = spadlconfig.actiontype_id["pass"]


def _frames() -> pd.DataFrame:
    base = dict(
        game_id=1,
        period_id=1,
        frame_id=10,
        time_seconds=1.0,
        frame_rate=25.0,
        z=0.0,
        speed=0.0,
        speed_source="native",
        ball_state="alive",
        confidence=None,
        visibility=None,
        source_provider="synthetic",
        is_goalkeeper_source="native",
        is_goalkeeper=False,
    )
    return pd.DataFrame(
        [
            {
                **base,
                "player_id": 1,
                "team_id": 1,
                "is_ball": False,
                "x": 30.0,
                "y": 20.0,
                "team_attacking_direction": "ltr",
            },
            {
                **base,
                "player_id": 2,
                "team_id": 2,
                "is_ball": False,
                "x": 70.0,
                "y": 50.0,
                "team_attacking_direction": "rtl",
            },
        ]
    )


def _actions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            dict(
                game_id=1,
                period_id=1,
                action_id=1,
                team_id=2,
                player_id=2.0,
                type_id=PASS_ID,
                result_id=1,
                start_x=35.0,
                start_y=18.0,
                end_x=50.0,
                end_y=30.0,
                time_seconds=1.0,
            )
        ]
    )


# ---------------------------------------------------------------------------
# Task 5 -- the warning category
# ---------------------------------------------------------------------------


def test_category_is_public_and_importable():
    from silly_kicks.tracking import OrientationUnresolvedWarning

    assert issubclass(OrientationUnresolvedWarning, UserWarning)


def test_category_is_not_a_subclass_of_the_other_categories():
    """Separate categories on purpose: silencing one signal must not silence another."""
    from silly_kicks.tracking import OrientationUnresolvedWarning, SyntheticEPVWarning

    assert not issubclass(OrientationUnresolvedWarning, SyntheticEPVWarning)
    assert not issubclass(SyntheticEPVWarning, OrientationUnresolvedWarning)


# ---------------------------------------------------------------------------
# Task 6 -- every silent branch warns
# ---------------------------------------------------------------------------


def test_warns_when_direction_column_is_absent():
    from silly_kicks.tracking import OrientationUnresolvedWarning

    f = _frames().drop(columns=["team_attacking_direction"])
    with pytest.warns(OrientationUnresolvedWarning):
        flip = acting_team_attacks_rtl(_actions(), f)
    assert not flip.any()


def test_warns_when_direction_column_is_all_null():
    """RC4's shape: SkillCorner sets the column to None, so it EXISTS and is entirely NA."""
    from silly_kicks.tracking import OrientationUnresolvedWarning

    f = _frames()
    f["team_attacking_direction"] = None
    with pytest.warns(OrientationUnresolvedWarning):
        flip = acting_team_attacks_rtl(_actions(), f)
    assert not flip.any()


def test_warns_when_join_keys_do_not_align():
    """The branch the first draft of the spec MISSED. Reachable, and silent before D2."""
    from silly_kicks.tracking import OrientationUnresolvedWarning

    f = _frames().drop(columns=["team_id"])
    with pytest.warns(OrientationUnresolvedWarning):
        flip = acting_team_attacks_rtl(_actions(), f)
    assert not flip.any()


def test_warns_when_frames_are_empty_but_actions_are_not():
    """'No frames but plenty of actions' is a caller error, not a no-op."""
    from silly_kicks.tracking import OrientationUnresolvedWarning

    with pytest.warns(OrientationUnresolvedWarning):
        acting_team_attacks_rtl(_actions(), _frames().iloc[0:0])


def test_silent_when_there_are_no_actions_to_flip():
    """The ONE carve-out, narrower than the original disjunction on purpose.

    When a rule is specified by OUTCOME, its single carve-out is where the rule leaks -- so the
    carve-out is 'there were no actions to flip', not 'either input was empty'.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        flip = acting_team_attacks_rtl(_actions().iloc[0:0], _frames())
    assert flip.empty


def test_healthy_frames_do_not_warn():
    """Non-vacuity for the whole group: the warning must not fire on the normal path.

    Without this, every test above would still pass if the seam warned unconditionally.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        flip = acting_team_attacks_rtl(_actions(), _frames())
    assert flip.to_numpy().tolist() == [True]


def test_warns_when_the_join_resolves_nothing():
    """The ESCAPE ROUTE: keys align, labels exist, but the merge matches no action.

    ``(keyed["_dir"] == "rtl").fillna(False)`` turns an all-NaN merge result into an all-False
    flip with no signal at all. Two independent fixture groups walked through this exact hole
    during the D2 sweep -- one whose frames contained only the defender's row (acting team simply
    absent), one whose actions carried ``game_id="idsse_J03WMX"`` against frames keyed
    ``"J03WMX"``. Both looked oriented and were not.

    This is the case the early-exit branches CANNOT catch, which is why D2 is specified by
    OUTCOME rather than by enumerated condition.
    """
    from silly_kicks.tracking import OrientationUnresolvedWarning

    f = _frames()
    a = _actions()
    a["team_id"] = 999  # acting team is absent from the frames -> merge matches nothing
    with pytest.warns(OrientationUnresolvedWarning):
        flip = acting_team_attacks_rtl(a, f)
    assert not flip.any()


def test_all_home_actions_do_not_warn():
    """Non-vacuity partner: 'resolved, and nothing needed flipping' must stay SILENT.

    Without this, the fix above could be implemented as 'warn whenever the flip is all-False',
    which would fire on every legitimately all-home action set. The signal is NOTHING RESOLVED,
    not NOTHING FLIPPED.
    """
    a = _actions()
    a["team_id"] = 1  # the ltr/home team -> resolves fine, flip is legitimately all-False
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        flip = acting_team_attacks_rtl(a, _frames())
    assert not flip.any()


def test_partial_resolution_does_not_warn():
    """A mix of resolvable and unresolvable actions is not a failure.

    ADR-027 NaN-team rows (GS null-actor duels/fouls) legitimately never resolve. Warning on a
    partial miss would fire on healthy Gradient Sports data every time.
    """
    a = pd.concat([_actions(), _actions().assign(action_id=2, team_id=999)], ignore_index=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        flip = acting_team_attacks_rtl(a, _frames())
    assert flip.to_numpy().tolist() == [True, False]


def test_period_five_shootout_is_exempt():
    """PSO orientation is undefined by design, so an unresolved direction there is not a defect.

    ``direction.py``'s ``_LTR_KNOWN_PERIODS = (1, 2, 3, 4)`` already excludes period 5, and
    ``test_off_ball_runs_orientation.py::test_period_five_is_exempt`` pins period-5 frames as an
    ACCEPTED unoriented shape. Warning there would be noise.
    """
    f = _frames()
    f["period_id"] = 5
    f["team_attacking_direction"] = None
    a = _actions()
    a["period_id"] = 5
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        flip = acting_team_attacks_rtl(a, f)
    assert not flip.any()


def test_mixed_period_five_and_real_play_still_warns():
    """The exemption requires EVERY action to be a shootout.

    A call mixing a PSO with real play still has resolvable rows, so suppressing the warning
    would hide a genuine defect behind one period-5 row.
    """
    from silly_kicks.tracking import OrientationUnresolvedWarning

    f = _frames()
    f["team_attacking_direction"] = None
    a = pd.concat([_actions(), _actions().assign(action_id=2, period_id=5)], ignore_index=True)
    with pytest.warns(OrientationUnresolvedWarning):
        acting_team_attacks_rtl(a, f)


def test_feature_column_names_probe_is_silent():
    """``feature_column_names`` is a NAME probe and must not warn.

    It builds dummy actions plus deliberately EMPTY frames and calls every frame-aware
    transformer purely to read back its output column names, discarding the values. Before this
    was scoped, a single ``VAEP.fit`` on a frame-aware xfn list emitted ~12
    OrientationUnresolvedWarnings -- which would make the CI escalation fail every fit.

    This also keeps the empty-frames introspection path COVERED. The D2 fixture sweep converted
    the one other test that exercised it into a real-scene test, so without this nothing would
    pin that the library still calls frame-aware transformers with empty frames.
    """
    from silly_kicks.tracking import OrientationUnresolvedWarning
    from silly_kicks.tracking.features import tracking_default_xfns
    from silly_kicks.vaep.features.core import feature_column_names

    # Escalate ONLY this category: the probe legitimately emits others (SyntheticEPVWarning among
    # them, since it runs the placeholder-EPV path), and a blanket "error" would assert something
    # this test is not about.
    with warnings.catch_warnings():
        warnings.simplefilter("error", OrientationUnresolvedWarning)
        names = feature_column_names(tracking_default_xfns, 3)
    assert names, "probe returned no column names -- it is no longer exercising the transformers"


def test_warning_names_the_remedy():
    """A warning that does not say what to do about it gets filtered, not fixed."""
    from silly_kicks.tracking import OrientationUnresolvedWarning

    f = _frames()
    f["team_attacking_direction"] = None
    with pytest.warns(OrientationUnresolvedWarning) as record:
        acting_team_attacks_rtl(_actions(), f)
    message = str(record[0].message)
    assert "orient" in message.lower()
    assert np.any([tok in message for tok in ("output_convention", "orient_frames_to_ltr")])
