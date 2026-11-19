"""The library holds at its declared pandas floor, pandas 2.1.1.

The pandas-floor CI job (pandas 2.1.1, numpy<2) found two library failures that no job had run
before. Each test here is red on pandas 2.1.1 at the code of ``2901fa5c`` and green on 2.1.1 and
on a later pandas:

- C: pandas before 2.2.1 compares an object-dtype ``Series`` with a scalar through a Cython routine
  that needs a WRITEABLE buffer. ``PitchControlSurface`` keeps its id arrays read-only (an
  immutability contract), so every id lookup on a surface raised "buffer source array is
  read-only". The GK-influence family caught the error per frame and returned NaN.
- D: on pandas before 2.2, ``.to_numpy()`` of a nullable integer column holding NA gives an object
  array of ``pd.NA``, so ``team_id != prev_team`` feeding ``&`` raised "boolean value of NA is
  ambiguous" in ``add_possessions``. pandas 2.2 gives float64 with NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.id_compat import ids_equal, ids_isin, ids_match
from silly_kicks.tracking.pitch_control._surface import PitchControlSurface

_PANDAS_AT_LEAST_2_2 = np.lib.NumpyVersion(pd.__version__) >= "2.2.0"


def _surface_with_read_only_ids() -> PitchControlSurface:
    grid_x = np.linspace(0.0, 105.0, 4)
    grid_y = np.linspace(0.0, 68.0, 3)
    influence = np.stack([np.full((3, 4), float(k + 1)) for k in range(3)])
    return PitchControlSurface(
        grid_x=grid_x,
        grid_y=grid_y,
        surface=np.full((3, 4), 0.5),
        method="spearman",
        attacking_team_id="home",
        per_player_influence=influence,
        player_ids=np.array(["11", "22", "33"], dtype=object),
        player_team_ids=np.array(["home", "home", "away"], dtype=object),
    )


# --------------------------------------------------------------------------- C: read-only object ids


def test_surface_lookup_with_read_only_ids_works_at_pandas_floor_2_1_1():
    """Red on pandas 2.1.1 without the copy: ``player_surface`` raised "buffer source array is read-only"."""
    surface = _surface_with_read_only_ids()
    assert surface.player_ids is not None
    assert not surface.player_ids.flags.writeable  # the precondition: the fence makes the ids read-only

    np.testing.assert_array_equal(surface.player_surface("22"), np.full((3, 4), 2.0))
    assert surface.player_share("33") == pytest.approx(1.0)


def test_surface_fence_stays_read_only_after_a_lookup():
    """The copy is taken inside ``id_compat``; the surface's own id arrays stay read-only (the fence)."""
    surface = _surface_with_read_only_ids()
    surface.player_surface("11")
    assert surface.player_ids is not None
    assert surface.player_team_ids is not None
    assert not surface.player_ids.flags.writeable
    assert not surface.player_team_ids.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        surface.player_ids[0] = "99"


@pytest.mark.parametrize("compares_read_only", [True, False])
def test_id_compat_entries_accept_read_only_object_ids_on_both_branches(monkeypatch, compares_read_only):
    """Both branches of the floor gate give the same answers.

    The ``False`` branch (copy first) is the one pandas 2.1.1 takes; forcing it on a later pandas proves
    the copy changes no value. The ``True`` branch passes the array through untouched.
    """
    from silly_kicks import _pandas_compat

    if compares_read_only and not _pandas_compat.PANDAS_COMPARES_READ_ONLY_OBJECT:
        pytest.skip("before pandas 2.2.1 the no-copy branch raises by design; the floor takes the copy branch")
    monkeypatch.setattr(_pandas_compat, "PANDAS_COMPARES_READ_ONLY_OBJECT", compares_read_only)
    ids = np.array(["7", "8", "7"], dtype=object)
    ids.flags.writeable = False
    other = np.array(["7", "9", "7"], dtype=object)
    other.flags.writeable = False

    assert ids_match(ids, "7").tolist() == [True, False, True]
    assert ids_equal(pd.Series(ids), pd.Series(other)).tolist() == [True, False, True]
    assert ids_isin(ids, ["8"]).tolist() == [False, True, False]
    assert not ids.flags.writeable  # the caller's array is never made writeable or modified


def test_floor_gate_constant_matches_the_measured_first_fixed_pandas():
    """pandas 2.2.1 is the first release whose object ``Series == scalar`` accepts a read-only buffer.

    Measured 2026-11-19 (py3.11, numpy 1.26.4): 2.1.1, 2.1.4 and 2.2.0 raise; 2.2.1, 2.2.2, 2.2.3 and
    2.3.3 do not.
    """
    from silly_kicks._pandas_compat import PANDAS_COMPARES_READ_ONLY_OBJECT

    assert PANDAS_COMPARES_READ_ONLY_OBJECT == (np.lib.NumpyVersion(pd.__version__) >= "2.2.1")
    probe = np.array(["1", "2"], dtype=object)
    probe.flags.writeable = False
    if PANDAS_COMPARES_READ_ONLY_OBJECT:
        assert (pd.Series(probe) == "2").tolist() == [False, True]
    else:
        with pytest.raises(ValueError, match="read-only"):
            _ = pd.Series(probe) == "2"


# --------------------------------------------------------------------------- D: masked numeric to_numpy

_MASKED_CASES = [
    pd.Series([1, None, 3], dtype="Int64"),
    pd.Series([1, 2, 3], dtype="Int64"),
    pd.Series([1, None, 3], dtype="UInt8"),
    pd.Series([1.5, None], dtype="Float32"),
    pd.Series([1.5, 2.5], dtype="Float64"),
]
_UNCHANGED_CASES = [
    pd.Series([1, 2, 3], dtype="int64"),
    pd.Series([1.0, np.nan], dtype="float64"),
    pd.Series(["a", None], dtype=object),
    pd.Series([True, None], dtype="boolean"),
    pd.Series(["a", None], dtype="string"),
]


def _assert_same_array(got: np.ndarray, expected: np.ndarray) -> None:
    """Same dtype, same missing positions, same present values (NA-aware: ``pd.NA`` has no truth value)."""
    assert got.dtype == expected.dtype
    np.testing.assert_array_equal(pd.isna(got), pd.isna(expected))
    present = ~pd.isna(expected)
    assert got[present].tolist() == expected[present].tolist()


def test_masked_to_numpy_gives_the_pandas_2_2_result_at_the_floor():
    """The pandas 2.2 result, pinned: int with NA -> float64 NaN; without NA -> the numpy dtype; float stays."""
    from silly_kicks._pandas_compat import masked_to_numpy

    out = masked_to_numpy(pd.Series([1, None, 3], dtype="Int64"))
    assert out.dtype == np.float64
    np.testing.assert_array_equal(out, [1.0, np.nan, 3.0])
    assert masked_to_numpy(pd.Series([1, 2], dtype="Int64")).dtype == np.int64
    assert masked_to_numpy(pd.Series([1, None], dtype="UInt8")).dtype == np.float64
    out32 = masked_to_numpy(pd.Series([1.5, None], dtype="Float32"))
    assert out32.dtype == np.float32
    assert np.isnan(out32[1])


@pytest.mark.parametrize("s", _MASKED_CASES + _UNCHANGED_CASES, ids=lambda s: f"{s.dtype}-na{s.isna().any()}")
def test_masked_to_numpy_equals_to_numpy_on_a_later_pandas(s):
    """No change on a later pandas: from 2.2 the helper returns exactly what ``.to_numpy()`` returns."""
    from silly_kicks._pandas_compat import masked_to_numpy

    if not _PANDAS_AT_LEAST_2_2:
        pytest.skip("the identity holds from pandas 2.2; the floor result is pinned by the test above")
    _assert_same_array(masked_to_numpy(s), s.to_numpy())


@pytest.mark.parametrize("s", _UNCHANGED_CASES, ids=lambda s: str(s.dtype))
def test_masked_to_numpy_leaves_every_other_dtype_as_it_is(s):
    from silly_kicks._pandas_compat import masked_to_numpy

    _assert_same_array(masked_to_numpy(s), s.to_numpy())


# --------------------------------------------------------------------------- D: the proven sites


def _spadl_rows(team_ids: list[object]) -> pd.DataFrame:
    from silly_kicks.spadl import config as spadlconfig

    return pd.DataFrame(
        {
            "game_id": 1,
            "original_event_id": [str(i) for i in range(len(team_ids))],
            "action_id": range(len(team_ids)),
            "period_id": 1,
            "time_seconds": [float(i) for i in range(len(team_ids))],
            "team_id": team_ids,
            "player_id": 200,
            "start_x": 50.0,
            "start_y": 34.0,
            "end_x": 60.0,
            "end_y": 34.0,
            "type_id": spadlconfig.actiontype_id["pass"],
            "result_id": spadlconfig.result_id["success"],
            "bodypart_id": spadlconfig.bodypart_id["foot"],
        }
    )


def test_add_possessions_with_a_nullable_team_id_holding_na_at_pandas_floor_2_1_1():
    """Red on pandas 2.1.1: ``team_id != prev_team`` on ``pd.NA`` objects raised "boolean value of NA is ambiguous".

    The result equals the float64-NaN encoding of the same ids, which is what pandas 2.2+ itself compares.
    """
    from silly_kicks.spadl.utils import add_possessions

    ids = [100, None, 100, 200]
    nullable = add_possessions(_spadl_rows(ids).astype({"team_id": "Int64"}))
    floats = add_possessions(_spadl_rows(ids).astype({"team_id": "float64"}))
    assert nullable["possession_id"].tolist() == floats["possession_id"].tolist() == [0, 1, 2, 3]


def test_atomic_add_possessions_with_a_nullable_team_id_holding_na_at_pandas_floor_2_1_1():
    """Red on pandas 2.1.1 at ``atomic/spadl/utils.py`` (the same ``team_id != prev_team`` on ``pd.NA``)."""
    from silly_kicks.atomic.spadl.utils import add_possessions
    from tests.atomic._atomic_test_fixtures import _make_atomic_action

    def frame(dtype: str) -> pd.DataFrame:
        rows = [
            _make_atomic_action(action_id=i, time_seconds=float(i), team_id=t)  # type: ignore[arg-type]
            for i, t in enumerate([100, None, 100, 200])
        ]
        return pd.DataFrame(rows).astype({"team_id": dtype})

    nullable = add_possessions(frame("Int64"))
    floats = add_possessions(frame("float64"))
    assert nullable["possession_id"].tolist() == floats["possession_id"].tolist() == [0, 1, 2, 3]


# ---------------------------------------------------------------- D, the read compare sites: pre-shot GK context

_GK_CONTEXT_COLUMNS = ["gk_was_engaged", "gk_was_distributing", "gk_actions_in_possession", "defending_gk_player_id"]


def _pre_shot_rows(type_names: list[str]) -> pd.DataFrame:
    """A defending keeper action, then a Gradient Sports null-actor row (team and player NA), then a shot."""
    rows = _spadl_rows([1, 2, None, 1])
    from silly_kicks.spadl import config as spadlconfig

    rows["type_id"] = [spadlconfig.actiontype_id[t] for t in type_names]
    rows["player_id"] = [10, 90, None, 11]
    return rows


def _assert_gk_context_equal_and_engaged(nullable: pd.DataFrame, floats: pd.DataFrame) -> None:
    shot = nullable.index[-1]
    assert bool(nullable.loc[shot, "gk_was_engaged"])  # non-vacuity: the keeper is found through the NA row's window
    assert nullable["defending_gk_player_id"].iloc[-1] == 90.0
    for col in _GK_CONTEXT_COLUMNS:
        pd.testing.assert_series_equal(nullable[col], floats[col], check_names=True)


def test_add_pre_shot_gk_context_with_a_null_actor_row_in_the_window_at_pandas_floor_2_1_1():
    """Red on pandas 2.1.1: ``team_id[win] != shooter_team`` on ``pd.NA`` objects fed ``&`` (a read compare site).

    Gradient Sports emits ``team_id`` / ``player_id`` as ``Int64`` with NA on the null-actor duel and foul
    events (``GRADIENTSPORTS_SPADL_COLUMNS``). The result equals the float64-NaN encoding, which is what
    pandas 2.2+ itself compares.
    """
    from silly_kicks.spadl.utils import add_pre_shot_gk_context

    types = ["pass", "keeper_save", "foul", "shot"]
    nullable = add_pre_shot_gk_context(_pre_shot_rows(types).astype({"team_id": "Int64", "player_id": "Int64"}))
    floats = add_pre_shot_gk_context(_pre_shot_rows(types).astype({"team_id": "float64", "player_id": "float64"}))
    _assert_gk_context_equal_and_engaged(nullable, floats)


def test_atomic_add_pre_shot_gk_context_with_a_null_actor_row_in_the_window_at_pandas_floor_2_1_1():
    """Red on pandas 2.1.1 at ``atomic/spadl/utils.py`` (the same compare on ``pd.NA``; a read compare site)."""
    from silly_kicks.atomic.spadl import config as atomicconfig
    from silly_kicks.atomic.spadl.utils import add_pre_shot_gk_context
    from tests.atomic._atomic_test_fixtures import _make_atomic_action

    def frame(dtype: str) -> pd.DataFrame:
        spec = [(1, 10, "pass"), (2, 90, "keeper_save"), (None, None, "foul"), (1, 11, "shot")]
        rows = []
        for i, (team, player, name) in enumerate(spec):
            row = _make_atomic_action(action_id=i, time_seconds=float(i), team_id=team, player_id=player)  # type: ignore[arg-type]
            row["type_id"] = atomicconfig.actiontype_id[name]
            rows.append(row)
        return pd.DataFrame(rows).astype({"team_id": dtype, "player_id": dtype})

    _assert_gk_context_equal_and_engaged(
        add_pre_shot_gk_context(frame("Int64")), add_pre_shot_gk_context(frame("float64"))
    )
