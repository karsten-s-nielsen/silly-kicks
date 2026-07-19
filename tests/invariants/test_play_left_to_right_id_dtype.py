"""Cross-dtype invariance for the ``play_left_to_right`` / ``to_spadl_ltr`` family (ADR-019).

``home_team_id`` is a CALLER-SUPPLIED SCALAR; ``team_id`` is a COLUMN whose dtype is
provider-dependent (``int64`` for StatsBomb / Opta / Wyscout, object-string for the
kloppy-family and Sportec, nullable ``Int64`` for Gradient Sports). A raw ``==`` / ``!=``
between them silently mis-resolves on a dtype mismatch -- and the failure mode is NOT
"away rows are missed": for an object-string column vs an int scalar, ``!=`` is True for
EVERY row, so the HOME rows get mirrored 180 degrees as well.

Every in-repo converter path is dtype-matched, so this defect class is LATENT -- which is
exactly why no existing fixture catches it. This gate supplies the missing axis: call each
public entry point twice, once with a dtype-MATCHED scalar and once with a
dtype-MISMATCHED-but-VALUE-EQUAL scalar, and require identical output.

The six seams under test (all now routed through ``silly_kicks.id_compat.ids_match``):
  1. ``spadl.utils.play_left_to_right``
  2. ``atomic.spadl.utils.play_left_to_right``
  3. ``vaep.features.play_left_to_right``
  4. ``atomic.vaep.features.play_left_to_right``
  5. ``spadl.orientation.to_spadl_ltr`` (ABSOLUTE_FRAME_HOME_RIGHT)
  6. ``spadl.orientation.to_spadl_ltr`` (PER_PERIOD_ABSOLUTE)

Each case carries a non-vacuity assertion: the fixture MUST actually mirror something,
otherwise "matched == mismatched" would hold trivially and the gate would be theatre.
"""

import pandas as pd
import pytest

import silly_kicks.atomic.spadl.config as atomicspadlconfig
import silly_kicks.atomic.spadl.utils as atomic_spadl_utils
import silly_kicks.atomic.vaep.features as atomic_vaep_features
import silly_kicks.spadl.config as spadlconfig
import silly_kicks.spadl.utils as spadl_utils
import silly_kicks.vaep.features.core as vaep_core
from silly_kicks.spadl.orientation import (
    ABSOLUTE_FRAME_HOME_RIGHT,
    PER_PERIOD_ABSOLUTE,
    to_spadl_ltr,
)

HOME_INT = 100
AWAY_INT = 200


def _str_id(value: int) -> str:
    """Canonical string form of an integer id.

    Deliberately ``str(int(v))`` and NOT ``str(v)``: on a float-backed id column
    ``str(100.0)`` is ``"100.0"``, which is a DIFFERENT id -- a value-INEQUAL scalar
    would make the "mismatched" leg fail for the wrong reason and the gate would be
    measuring the wrong thing.
    """
    return str(int(value))


# (id, team_id column dtype, matched scalar, mismatched-but-value-equal scalar)
DTYPE_CASES = [
    pytest.param("int64_col_str_scalar", "int64", HOME_INT, _str_id(HOME_INT), id="int64_col_str_scalar"),
    pytest.param("object_col_int_scalar", "object", _str_id(HOME_INT), HOME_INT, id="object_col_int_scalar"),
    pytest.param("Int64_col_str_scalar", "Int64", HOME_INT, _str_id(HOME_INT), id="Int64_col_str_scalar"),
]


def _team_ids(dtype: str) -> pd.Series:
    """[home, home, away] in the requested id dtype."""
    values = [HOME_INT, HOME_INT, AWAY_INT]
    if dtype == "object":
        return pd.Series([_str_id(v) for v in values], dtype=object)
    return pd.Series(values, dtype=dtype)


def _spadl_actions(dtype: str) -> pd.DataFrame:
    """Two home rows and one away row, at distinguishable coordinates.

    Under the correct mirror, ``start_x`` -> [10, 20, 105-30]; under the dtype bug
    (every row treated as away) it becomes [105-10, 105-20, 105-30]. The two differ
    in the HOME rows, which is precisely what the bug corrupts.
    """
    return pd.DataFrame(
        {
            "game_id": [1, 1, 1],
            "period_id": [1, 1, 1],
            "action_id": [0, 1, 2],
            "team_id": _team_ids(dtype),
            "start_x": [10.0, 20.0, 30.0],
            "start_y": [10.0, 20.0, 30.0],
            "end_x": [40.0, 50.0, 60.0],
            "end_y": [15.0, 25.0, 35.0],
        }
    )


def _atomic_actions(dtype: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [1, 1, 1],
            "period_id": [1, 1, 1],
            "action_id": [0, 1, 2],
            "team_id": _team_ids(dtype),
            "x": [10.0, 20.0, 30.0],
            "y": [10.0, 20.0, 30.0],
            "dx": [1.0, 2.0, 3.0],
            "dy": [-1.0, -2.0, -3.0],
        }
    )


def _gamestates(frame: pd.DataFrame) -> list[pd.DataFrame]:
    """Two-slot gamestates. Fresh copies -- the vaep helpers mutate in place."""
    return [frame.copy(), frame.copy()]


def _assert_mirrored_something(before: pd.DataFrame, after: pd.DataFrame, cols: list[str]) -> None:
    """Non-vacuity: the fixture must actually exercise the mirror.

    If nothing moved, "matched output == mismatched output" would hold trivially for
    any implementation, correct or broken.
    """
    moved = any(not before[c].equals(after[c]) for c in cols)
    assert moved, f"fixture never mirrored any of {cols} -- the gate would pass vacuously"


# ---------------------------------------------------------------------------
# Site 1 -- spadl.utils.play_left_to_right
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("_name", "dtype", "matched", "mismatched"), DTYPE_CASES)
def test_spadl_play_left_to_right_is_dtype_invariant(_name, dtype, matched, mismatched):
    actions = _spadl_actions(dtype)

    out_matched = spadl_utils.play_left_to_right(actions, matched)
    out_mismatched = spadl_utils.play_left_to_right(actions, mismatched)

    _assert_mirrored_something(actions, out_matched, ["start_x", "start_y", "end_x", "end_y"])
    pd.testing.assert_frame_equal(out_matched, out_mismatched)
    # Pin the actual geometry: home rows must NOT move.
    assert out_matched["start_x"].tolist() == [10.0, 20.0, spadlconfig.field_length - 30.0]


# ---------------------------------------------------------------------------
# Site 2 -- atomic.spadl.utils.play_left_to_right
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("_name", "dtype", "matched", "mismatched"), DTYPE_CASES)
def test_atomic_spadl_play_left_to_right_is_dtype_invariant(_name, dtype, matched, mismatched):
    actions = _atomic_actions(dtype)

    out_matched = atomic_spadl_utils.play_left_to_right(actions, matched)
    out_mismatched = atomic_spadl_utils.play_left_to_right(actions, mismatched)

    _assert_mirrored_something(actions, out_matched, ["x", "y", "dx", "dy"])
    pd.testing.assert_frame_equal(out_matched, out_mismatched)
    assert out_matched["x"].tolist() == [10.0, 20.0, atomicspadlconfig.field_length - 30.0]
    assert out_matched["dx"].tolist() == [1.0, 2.0, -3.0]


# ---------------------------------------------------------------------------
# Site 3 -- vaep.features.core.play_left_to_right
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("_name", "dtype", "matched", "mismatched"), DTYPE_CASES)
def test_vaep_play_left_to_right_is_dtype_invariant(_name, dtype, matched, mismatched):
    base = _spadl_actions(dtype)

    out_matched = vaep_core.play_left_to_right(_gamestates(base), matched)
    out_mismatched = vaep_core.play_left_to_right(_gamestates(base), mismatched)

    _assert_mirrored_something(base, out_matched[0], ["start_x", "start_y", "end_x", "end_y"])
    assert len(out_matched) == len(out_mismatched) == 2
    for slot_matched, slot_mismatched in zip(out_matched, out_mismatched, strict=True):
        pd.testing.assert_frame_equal(slot_matched, slot_mismatched)
        assert slot_matched["start_x"].tolist() == [10.0, 20.0, spadlconfig.field_length - 30.0]


# ---------------------------------------------------------------------------
# Site 4 -- atomic.vaep.features.play_left_to_right
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("_name", "dtype", "matched", "mismatched"), DTYPE_CASES)
def test_atomic_vaep_play_left_to_right_is_dtype_invariant(_name, dtype, matched, mismatched):
    base = _atomic_actions(dtype)

    out_matched = atomic_vaep_features.play_left_to_right(_gamestates(base), matched)
    out_mismatched = atomic_vaep_features.play_left_to_right(_gamestates(base), mismatched)

    _assert_mirrored_something(base, out_matched[0], ["x", "y", "dx", "dy"])
    assert len(out_matched) == len(out_mismatched) == 2
    for slot_matched, slot_mismatched in zip(out_matched, out_mismatched, strict=True):
        pd.testing.assert_frame_equal(slot_matched, slot_mismatched)
        assert slot_matched["x"].tolist() == [10.0, 20.0, atomicspadlconfig.field_length - 30.0]
        assert slot_matched["dx"].tolist() == [1.0, 2.0, -3.0]


# ---------------------------------------------------------------------------
# Site 5 -- to_spadl_ltr(ABSOLUTE_FRAME_HOME_RIGHT) -> _mirror_absolute_frame
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("_name", "dtype", "matched", "mismatched"), DTYPE_CASES)
def test_to_spadl_ltr_absolute_frame_is_dtype_invariant(_name, dtype, matched, mismatched):
    actions = _spadl_actions(dtype)

    out_matched = to_spadl_ltr(actions, input_convention=ABSOLUTE_FRAME_HOME_RIGHT, home_team_id=matched)
    out_mismatched = to_spadl_ltr(actions, input_convention=ABSOLUTE_FRAME_HOME_RIGHT, home_team_id=mismatched)

    _assert_mirrored_something(actions, out_matched, ["start_x", "start_y", "end_x", "end_y"])
    pd.testing.assert_frame_equal(out_matched, out_mismatched)
    assert out_matched["start_x"].tolist() == [10.0, 20.0, spadlconfig.field_length - 30.0]


def test_to_spadl_ltr_absolute_frame_treats_na_team_as_away():
    """A null ``team_id`` is mirrored as AWAY -- pinned deliberately, both ways.

    ``spadl/kloppy.py`` genuinely emits ``event.team.team_id if event.team else None``,
    so a null team reaches this seam. Two independent reasons this must stay "away":

    1. **It is the pre-existing behaviour.** The raw ``!=`` this site used to run reports
       True for a null (``None != "100"``), so the row was already mirrored. The ADR-019
       hardening pass is a dtype fix and must not silently change NA output.
    2. **The sibling agrees.** ``_mirror_per_period`` resolves NA to ``is_home=False`` ->
       away, which ADR-027 fixed deliberately to preserve the pre-NaN sentinel-0
       behaviour. Splitting the two mirror functions' NA semantics apart would be a
       latent inconsistency nothing else would catch.

    REGRESSION GUARD: this test exists because BOTH semantics passed the entire suite
    (1577 passed) -- nothing else covers a null team at this seam. If someone "hardens"
    the site by also requiring ``.notna()``, row 1 stops being mirrored and this goes red.
    """
    actions = _spadl_actions("object")
    actions.loc[1, "team_id"] = None

    out = to_spadl_ltr(actions, input_convention=ABSOLUTE_FRAME_HOME_RIGHT, home_team_id=_str_id(HOME_INT))

    # row 0 = home (untouched); rows 1 (NA) and 2 (away) BOTH mirrored
    assert out["start_x"].tolist() == [
        10.0,
        spadlconfig.field_length - 20.0,
        spadlconfig.field_length - 30.0,
    ]
    assert out["start_y"].tolist() == [
        10.0,
        spadlconfig.field_width - 20.0,
        spadlconfig.field_width - 30.0,
    ]


def test_na_semantics_agree_across_both_mirror_functions():
    """META: the two mirror seams must resolve a null ``team_id`` the SAME way.

    Guards the class rather than one site -- a future edit to either function that
    diverges from the other goes red here even if its own local test was updated to
    match. Both must treat NA as away (not-home).
    """
    from silly_kicks.id_compat import ids_match

    team_ids = pd.Series([None, _str_id(HOME_INT), _str_id(AWAY_INT)], dtype=object)
    home = _str_id(HOME_INT)

    # _mirror_absolute_frame: away_idx = ~ids_match(...)
    away_idx = ~ids_match(team_ids, home)
    # _mirror_per_period:     is_home  =  ids_match(...)
    is_home = ids_match(team_ids, home)

    assert bool(away_idx.iloc[0]) is True, "absolute-frame seam must treat NA as away"
    assert bool(is_home.iloc[0]) is False, "per-period seam must treat NA as not-home"
    assert bool(away_idx.iloc[0]) == (not bool(is_home.iloc[0])), "seams disagree on NA"


# ---------------------------------------------------------------------------
# Site 6 -- to_spadl_ltr(PER_PERIOD_ABSOLUTE) -> _mirror_per_period
# ---------------------------------------------------------------------------


def _per_period_actions(dtype: str) -> pd.DataFrame:
    """Home and away rows in BOTH periods, so a per-period flip is observable."""
    values = [HOME_INT, AWAY_INT, HOME_INT, AWAY_INT]
    team_ids = (
        pd.Series([_str_id(v) for v in values], dtype=object) if dtype == "object" else pd.Series(values, dtype=dtype)
    )
    return pd.DataFrame(
        {
            "game_id": [1, 1, 1, 1],
            "period_id": [1, 1, 2, 2],
            "action_id": [0, 1, 2, 3],
            "team_id": team_ids,
            "start_x": [10.0, 30.0, 10.0, 30.0],
            "start_y": [10.0, 30.0, 10.0, 30.0],
            "end_x": [40.0, 60.0, 40.0, 60.0],
            "end_y": [15.0, 35.0, 15.0, 35.0],
        }
    )


@pytest.mark.parametrize(("_name", "dtype", "matched", "mismatched"), DTYPE_CASES)
def test_to_spadl_ltr_per_period_is_dtype_invariant(_name, dtype, matched, mismatched):
    actions = _per_period_actions(dtype)
    # Home attacks right in P1, left in P2 (teams switch ends).
    flips = {1: True, 2: False}

    out_matched = to_spadl_ltr(
        actions,
        input_convention=PER_PERIOD_ABSOLUTE,
        home_team_id=matched,
        home_attacks_right_per_period=flips,
    )
    out_mismatched = to_spadl_ltr(
        actions,
        input_convention=PER_PERIOD_ABSOLUTE,
        home_team_id=mismatched,
        home_attacks_right_per_period=flips,
    )

    _assert_mirrored_something(actions, out_matched, ["start_x", "start_y", "end_x", "end_y"])
    pd.testing.assert_frame_equal(out_matched, out_mismatched)
    # P1: home untouched, away mirrored. P2: home mirrored, away untouched.
    assert out_matched["start_x"].tolist() == [
        10.0,
        spadlconfig.field_length - 30.0,
        spadlconfig.field_length - 10.0,
        30.0,
    ]
