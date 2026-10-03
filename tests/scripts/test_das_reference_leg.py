"""Offline unit tests for scripts/_das_reference_leg.py (no accessible-space, no network).

The accessible-space-calling paths (``reference_leg_arrays`` / ``__main__``) run only on the DGX under a
pandas-2 interpreter and are covered by the owner corpus smoke (spec 8); here we test the sk-free pure
helpers, the fail-loud env checker, the collision-free frame key, and the lazy-import contract.
"""

from __future__ import annotations

import builtins
import importlib
import warnings

import pandas as pd
import pytest

from scripts import _das_reference_leg as R


def test_module_imports_without_accessible_space(monkeypatch):
    # The lazy-import contract: importing the module and using its pure helpers must NOT require
    # accessible_space (absent in CI). Simulate absence and re-import.
    real_import = builtins.__import__

    def blocked(name, *a, **k):
        if name == "accessible_space" or name.startswith("accessible_space."):
            raise ImportError("accessible_space blocked for test")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blocked)
    mod = importlib.reload(R)
    assert callable(mod._reference_lib_frames)
    assert mod._REFERENCE_COMMON["frame_col"] == "frame_id"
    importlib.reload(R)  # restore with the real import


def test_reference_common_matches_golden_generator():
    # NEW gate (no such test existed): the recipe must equal the frozen golden generator's _COMMON,
    # modulo attacking_direction_col (the generator uses "dir"; the parity pins the shared dir column).
    gen = importlib.import_module("tests.tracking._fixtures.das_golden._generate")
    common = dict(R._REFERENCE_COMMON)
    expected = dict(gen._COMMON)
    common.pop("attacking_direction_col")
    expected.pop("attacking_direction_col", None)
    assert common == expected


def test_unique_frame_col_distinguishes_reused_frame_id_across_periods():
    lib = pd.DataFrame(
        {
            "game_id": [1, 1, 1, 1],
            "period_id": [1, 1, 2, 2],
            "frame_id": [5, 5, 5, 5],  # reused across periods
            "player_id": ["a", "b", "a", "b"],
        }
    )
    out = R._add_unique_frame_col(lib)
    # rows within one (game,period,frame) share a code; the two periods differ.
    assert out.loc[0, "_uframe"] == out.loc[1, "_uframe"]
    assert out.loc[2, "_uframe"] == out.loc[3, "_uframe"]
    assert out.loc[0, "_uframe"] != out.loc[2, "_uframe"]


@pytest.mark.parametrize(
    "pv,av,ok",
    [("2.3.3", "2.0.15", True), ("3.0.6", "2.0.15", False), ("2.3.3", "2.1.0", False)],
)
def test_check_reference_env(pv, av, ok):
    if ok:
        R._check_reference_env(pv, av)
    else:
        with pytest.raises(RuntimeError):
            R._check_reference_env(pv, av)


def test_offside_warning_is_an_error_under_the_filter():
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Offside not properly detectable")
        with pytest.raises(UserWarning):
            warnings.warn("Offside not properly detectable, maybe too few defenders. Ignoring offside.", stacklevel=2)


def test_best_of_returns_the_minimum_and_the_last_result():
    from scripts._das_reference_leg import best_of

    calls = []

    def fn():
        calls.append(1)
        return len(calls)

    result, seconds = best_of(fn, 3)
    assert result == 3 and len(calls) == 3 and seconds >= 0.0
    assert best_of(fn, 0)[0] == 4  # repeat < 1 still runs once


def test_inferred_direction_common_drops_the_supplied_direction():
    from scripts._das_reference_leg import _common

    c = _common(infer_direction=True)
    assert c["infer_attacking_direction"] is True and c["attacking_direction_col"] is None
    assert _common(infer_direction=False)["attacking_direction_col"] == "_das_parity_dir"


def _idsse_frames() -> pd.DataFrame:
    """One IDSSE-shaped frame: STRING player/team ids (``DFL-OBJ-*`` / ``DFL-CLU-*``) and a ball row with NA ids.

    The DGX smoke (combined-cycle Task 17) failed every IDSSE match on ``int('DFL-OBJ-0002DR')``.
    """
    return pd.DataFrame(
        {
            "game_id": ["DFL-MAT-X"] * 3,
            "period_id": [1, 1, 1],
            "frame_id": [10, 10, 10],
            "player_id": pd.Series(["DFL-OBJ-0002DR", "DFL-OBJ-00008K", pd.NA], dtype=object),
            "team_id": pd.Series(["DFL-CLU-000004", "DFL-CLU-00000P", pd.NA], dtype=object),
            "is_ball": [False, False, True],
            "x": [50.0, 60.0, 55.0],
            "y": [30.0, 40.0, 34.0],
            "vx": [0.0, 0.0, 0.0],
            "vy": [0.0, 0.0, 0.0],
            "team_in_possession": pd.Series(["DFL-CLU-000004"] * 3, dtype=object),
            "_das_parity_dir": [1.0, 1.0, 1.0],
        }
    )


def test_reference_lib_frames_passes_string_ids_through():
    lib = R._reference_lib_frames(_idsse_frames())
    assert list(lib["player_id"]) == ["pDFL-OBJ-0002DR", "pDFL-OBJ-00008K", "ball"]
    assert list(lib["team_id"]) == ["tDFL-CLU-000004", "tDFL-CLU-00000P", None]
    assert list(lib["team_in_possession"]) == ["tDFL-CLU-000004"] * 3


def test_reference_lib_frames_numeric_ids_keep_the_golden_tokens():
    # Numeric ids keep the frozen golden generator's ``str(int(v))`` tokens exactly (the recipe is
    # byte-identical for the providers whose ids are integers).
    from tests.tracking._das_golden import load_golden

    gen = importlib.import_module("tests.tracking._fixtures.das_golden._generate")
    frames = load_golden().frames_for("S01")
    ours, theirs = R._reference_lib_frames(frames), gen._lib_frames(frames)
    for c in ("player_id", "team_id", "team_in_possession"):
        a = [None if pd.isna(v) else v for v in ours[c].astype(object)]
        b = [None if pd.isna(v) else v for v in theirs[c].astype(object)]
        assert a == b, c


@pytest.mark.parametrize(
    "token,key",
    [("100", 100), ("DFL-OBJ-0002DR", "DFL-OBJ-0002DR"), ("GK-7", "GK-7")],
)
def test_id_key_is_an_int_for_a_digit_token_else_the_string(token, key):
    got = R._id_key(token)
    assert got == key and type(got) is type(key)


def test_reference_leg_arrays_parses_string_player_ids_back(monkeypatch):
    # The parse-back of the library's "p<id>" tokens (was ``int(str(pid)[1:])``) must survive string ids.
    # accessible_space is faked: only the keying around it is under test here.
    import sys
    import types

    import numpy as np

    frames = _idsse_frames()
    n = len(frames)  # every row carries team_in_possession, so the library returns one value per row
    fake = types.ModuleType("accessible_space")
    fake.get_dangerous_accessible_space = lambda df, **k: types.SimpleNamespace(  # type: ignore[attr-defined]
        acc_space=np.arange(n, dtype=float), das=np.arange(n, dtype=float)
    )
    fake.get_individual_dangerous_accessible_space = lambda df, **k: types.SimpleNamespace(  # type: ignore[attr-defined]
        player_acc_space=np.ones(n), player_das=np.ones(n)
    )
    monkeypatch.setitem(sys.modules, "accessible_space", fake)
    monkeypatch.setattr(R, "_check_reference_env", lambda *a: None)
    monkeypatch.setattr(R.importlib.metadata, "version", lambda name: "2.0.15")
    with warnings.catch_warnings():  # reference_leg_arrays installs an "error" filter; keep it scoped
        out = R.reference_leg_arrays(frames)
    assert sorted(out["player_keys"][:, 3]) == ["DFL-OBJ-00008K", "DFL-OBJ-0002DR"]


def _fake_library(monkeypatch, *, n_team: int, player_values, calls: list | None = None):
    """Install a fake accessible_space with the 2.0.15 result SHAPES measured on the DGX (combined-cycle
    Phase B, F5): team results cover only the rows WITH possession; player results cover EVERY row."""
    import sys
    import types

    import numpy as np

    def team(df, **k):
        if calls is not None:
            calls.append(("team", df.copy(), k))
        return types.SimpleNamespace(acc_space=np.full(n_team, 7.0), das=np.full(n_team, 7.0))

    def individual(df, **k):
        if calls is not None:
            calls.append(("individual", df.copy(), k))
        v = np.asarray(player_values, dtype=float)
        return types.SimpleNamespace(player_acc_space=v, player_das=v)

    fake = types.ModuleType("accessible_space")
    fake.get_dangerous_accessible_space = team  # type: ignore[attr-defined]
    fake.get_individual_dangerous_accessible_space = individual  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "accessible_space", fake)
    monkeypatch.setattr(R, "_check_reference_env", lambda *a: None)
    monkeypatch.setattr(R.importlib.metadata, "version", lambda name: "2.0.15")


def _two_frames(*, carrier: bool = False) -> pd.DataFrame:
    """Frame 10 has NO possession (a dead-ball frame) and comes first; frame 11 has possession."""
    rows = []
    for fid, poss in ((10, None), (11, 1)):
        for pid, tid, ball in ((100, 1, False), (200, 2, False), (None, None, True)):
            rows.append(
                {
                    "game_id": 1,
                    "period_id": 1,
                    "frame_id": fid,
                    "player_id": pid,
                    "team_id": tid,
                    "is_ball": ball,
                    "x": 50.0,
                    "y": 34.0,
                    "vx": 0.0,
                    "vy": 0.0,
                    "team_in_possession": poss,
                    "_das_parity_dir": 1.0,
                }
            )
    f = pd.DataFrame(rows)
    for c in ("player_id", "team_id", "team_in_possession"):
        f[c] = f[c].astype("Int64")
    if carrier:
        f["ball_carrier_player_id"] = pd.array([pd.NA, pd.NA, pd.NA, 100, 100, 100], dtype="Int64")
    return f


def test_player_values_are_read_by_row_not_by_possession_row(monkeypatch):
    # F5: the library returns player values for EVERY row, team values for possession rows only. Reading
    # player values with the possession-row counter shifted every value after a no-possession row onto
    # another player (on the DGX, 20022 of 20097 SkillCorner player rows were wrong).
    frames = _two_frames()
    _fake_library(monkeypatch, n_team=3, player_values=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    with warnings.catch_warnings():
        out = R.reference_leg_arrays(frames)
    got = {int(k[3]): float(v) for k, v in zip(out["player_keys"], out["player_das"], strict=True)}
    assert got == {100: 3.0, 200: 4.0}  # frame 11's players sit at rows 3 and 4


@pytest.mark.parametrize("n_team,n_player", [(2, 6), (3, 5)])
def test_a_result_length_off_the_measured_contract_fails_loud(monkeypatch, n_team, n_player):
    _fake_library(monkeypatch, n_team=n_team, player_values=list(range(n_player)))
    with warnings.catch_warnings(), pytest.raises(RuntimeError, match="accessible-space returned"):
        R.reference_leg_arrays(_two_frames())


def test_a_series_result_must_sit_on_the_input_rows_index():
    rows = pd.Index([0, 1])
    assert list(R._row_values(pd.Series([1.0, 2.0], index=rows), rows, "team")) == [1.0, 2.0]
    with pytest.raises(RuntimeError, match="index other than the input rows"):
        R._row_values(pd.Series([1.0, 2.0], index=[1, 0]), rows, "team")


def test_the_carrier_is_forwarded_when_the_frames_carry_it(monkeypatch):
    # F6: native DAS excludes the ball carrier from offside (``ball_carrier_player_id``); the reference must
    # be told the carrier too, or a carrier beyond the defensive line is marked offside by the library alone.
    calls: list = []
    _fake_library(monkeypatch, n_team=3, player_values=[0.0] * 6, calls=calls)
    with warnings.catch_warnings():
        R.reference_leg_arrays(_two_frames(carrier=True))
    assert [c[0] for c in calls] == ["team", "individual"]
    for _name, df, kwargs in calls:
        assert kwargs["player_in_possession_col"] == "ball_carrier_player_id"
        # the carrier carries the same "p<id>" token as the carrier's own player row (missing = None under
        # the pandas-2 reference interpreter; pandas 3 would store NaN, so compare the missing values as such)
        tokens = [None if pd.isna(v) else v for v in df["ball_carrier_player_id"]]
        assert tokens == [None, None, None, "p100", "p100", "p100"]
        assert "p100" in set(df["player_id"])


def test_no_carrier_column_means_no_carrier_kwarg(monkeypatch):
    # The golden frames carry no carrier: the recipe there stays the frozen generator's _COMMON exactly.
    calls: list = []
    _fake_library(monkeypatch, n_team=3, player_values=[0.0] * 6, calls=calls)
    with warnings.catch_warnings():
        R.reference_leg_arrays(_two_frames())
    assert all("player_in_possession_col" not in kwargs for _n, _df, kwargs in calls)


def test_the_inferred_direction_leg_forwards_the_carrier_too(monkeypatch):
    calls: list = []
    _fake_library(monkeypatch, n_team=3, player_values=[0.0] * 6, calls=calls)
    with warnings.catch_warnings():
        R.reference_leg_arrays(_two_frames(carrier=True), infer_direction=True)
    assert [c[0] for c in calls] == ["team"]
    assert calls[0][2]["player_in_possession_col"] == "ball_carrier_player_id"
    assert calls[0][2]["infer_attacking_direction"] is True


def test_the_carrier_column_name_is_the_native_default():
    from silly_kicks.tracking._das_pack import _DEFAULT_PLAYER_IN_POSSESSION_COL

    assert R._CARRIER_COL == _DEFAULT_PLAYER_IN_POSSESSION_COL
