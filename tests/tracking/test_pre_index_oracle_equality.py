"""The rewritten ``_pre_index_frames`` equals its oracle (HFU fit-units spec 4.3).

The oracle (`tests/tracking/_pre_index_oracle.py`) is the function as it was at 4.140.0, copied verbatim. Every
case below is a named frame table; the new function must return an equal dict (`assert_pre_index_equal`), or
raise the same exception type where the oracle raises. `test_the_equality_test_detects_a_fault` and
`test_assert_pre_index_equal_detects` prove that an inequality fails.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pandas as pd
import pytest

from scripts._toy_corpora import PROVIDER_ID_KIND, _game_frames, staged_frames
from silly_kicks.tracking import _ball_carrier
from silly_kicks.tracking._ball_carrier import _pre_index_frames, infer_ball_carrier
from tests.tracking._pre_index_oracle import assert_pre_index_equal, pre_index_frames_oracle
from tests.tracking.test_ball_carrier import _make_carrier_frame

# ---- frame builders ---------------------------------------------------------------------------------


def _players(n: int = 3, *, velocity: bool = True, first_pid: int = 10) -> list[dict]:
    out = []
    for i in range(n):
        p = {"pid": first_pid + i, "tid": 1 if i % 2 == 0 else 2, "x": 40.0 + 3.0 * i, "y": 30.0 + 2.0 * i}
        if velocity:
            p |= {"vx": 0.5 * i, "vy": -0.25 * i}
        out.append(p)
    return out


def _match(n_frames: int = 4, n_players: int = 3, *, velocity: bool = True) -> pd.DataFrame:
    return pd.concat(
        [
            _make_carrier_frame(frame_id=f, ball_x=45.0 + f, players=_players(n_players, velocity=velocity))
            for f in range(1, n_frames + 1)
        ],
        ignore_index=True,
    )


def _big_match(n_frames: int = 20_000, n_players: int = 22) -> pd.DataFrame:
    """A synthetic 25 Hz match built with numpy (one ball row and ``n_players`` player rows per frame)."""
    rng = np.random.default_rng(7)
    per = n_players + 1
    n = n_frames * per
    slot = np.tile(np.arange(per), n_frames)
    frame_id = np.repeat(np.arange(1, n_frames + 1), per)
    is_ball = slot == 0
    return pd.DataFrame(
        {
            "game_id": 1,
            "period_id": np.where(frame_id <= n_frames // 2, 1, 2),
            "frame_id": frame_id,
            "time_seconds": frame_id * 0.04,
            "frame_rate": 25.0,
            "player_id": np.where(is_ball, np.nan, 100.0 + slot),
            "team_id": np.where(is_ball, np.nan, np.where(slot <= n_players // 2, 1.0, 2.0)),
            "is_ball": is_ball,
            "is_goalkeeper": False,
            "x": rng.uniform(0.0, 105.0, n),
            "y": rng.uniform(0.0, 68.0, n),
            "vx": rng.normal(0.0, 2.0, n),
            "vy": rng.normal(0.0, 2.0, n),
            "ball_state": "alive",
            "source_provider": "sportec",
            "team_attacking_direction": "ltr",
        }
    )


def _toy(provider: str) -> pd.DataFrame:
    return staged_frames(_game_frames("7", provider, n_frames=6, seed=3), PROVIDER_ID_KIND[provider])


def _null_frame_key() -> pd.DataFrame:
    f = _match().astype({"frame_id": "float64"})
    f.loc[f.index[2], "frame_id"] = np.nan  # a player row of frame 1
    return f


def _pid_only_on_a_null_key_row() -> pd.DataFrame:
    f = _match().astype({"frame_id": "float64"})
    i = f.index[2]
    f.loc[i, "frame_id"] = np.nan
    f.loc[i, "player_id"] = 99.0
    return f


def _float_nan_pid() -> pd.DataFrame:
    f = _match()
    f.loc[f.index[1], "player_id"] = np.nan  # a player row
    return f


def _object_pid_with_np_nan() -> pd.DataFrame:
    f = _match()
    f["player_id"] = f["player_id"].astype(object)
    f.loc[f.index[1], "player_id"] = np.nan  # THE np.nan object: `dict.get` finds it by identity
    return f


def _object_pid_with_two_nan_objects() -> pd.DataFrame:
    f = _match()
    ids = f["player_id"].astype(object).tolist()
    ids[1], ids[5] = float("nan"), float("nan")  # two DISTINCT NaN objects on two player rows
    f["player_id"] = pd.Series(ids, dtype=object, index=f.index)
    assert ids[1] is not ids[5]
    return f


def _duplicate_pid_in_a_frame() -> pd.DataFrame:
    f = _match()
    f.loc[f.index[2], "player_id"] = f.loc[f.index[1], "player_id"]
    return f


def _two_ball_rows() -> pd.DataFrame:
    f = _match()
    extra = f.iloc[[0]].assign(x=60.0)
    return pd.concat([f.iloc[:1], extra, f.iloc[1:]], ignore_index=True)


def _frame_with_no_player() -> pd.DataFrame:
    f = _match()
    return f[~((f["frame_id"] == 2) & ~f["is_ball"])].reset_index(drop=True)


def _null_is_ball_object() -> pd.DataFrame:
    f = _match()
    f["is_ball"] = f["is_ball"].astype(object)
    f.loc[f.index[1], "is_ball"] = None
    return f


def _null_is_ball_boolean() -> pd.DataFrame:
    f = _match()
    f["is_ball"] = f["is_ball"].astype("boolean")
    f.loc[f.index[1], "is_ball"] = pd.NA
    return f


def _category_team_id() -> pd.DataFrame:
    f = _match()
    f["team_id"] = f["team_id"].astype("Int64").astype("string").astype("category")
    return f


def _category_frame_keys() -> pd.DataFrame:
    f = _match()
    f["game_id"] = f["game_id"].astype(str).astype("category")
    f["frame_id"] = f["frame_id"].astype("category")
    return f


def _category_frame_keys_unused() -> pd.DataFrame:
    f = _match()
    f["game_id"] = pd.Categorical(f["game_id"].astype(str), categories=["1", "2", "9"])  # "2", "9" unused
    f["frame_id"] = pd.Categorical(f["frame_id"], categories=[1, 2, 3, 4, 5, 6])  # 5, 6 unused
    return f


def _category_frame_keys_with_nulls() -> pd.DataFrame:
    """`category` frame keys with unused categories and a null key on a ball row and on a player row."""
    f = _match()
    f["game_id"] = pd.Categorical(f["game_id"].astype(str), categories=["1", "7"])
    f["frame_id"] = pd.Categorical(f["frame_id"], categories=[1, 2, 3, 4, 5])
    ball = f.index[f["is_ball"].to_numpy(dtype=bool)]
    f.loc[ball[1], "frame_id"] = np.nan  # the ball row of frame 2
    f.loc[f.index.difference(ball)[0], "frame_id"] = np.nan  # a player row of frame 1
    return f


def _float32_coords() -> pd.DataFrame:
    return _match().astype({"x": "float32", "y": "float32", "vx": "float32", "vy": "float32"})


def _unsorted_rows() -> pd.DataFrame:
    f = _match(n_frames=5, n_players=4)
    return f.sample(frac=1.0, random_state=11).reset_index(drop=True)


def _empty_table() -> pd.DataFrame:
    return _match().iloc[0:0]


def _int64_na_pid() -> pd.DataFrame:
    f = _match()
    f["player_id"] = f["player_id"].astype("Int64")
    f.loc[f.index[1], "player_id"] = pd.NA  # a player row
    return f


def _object_none_pid_beside_strings() -> pd.DataFrame:
    f = _match()
    f["player_id"] = f["player_id"].map(lambda v: None if pd.isna(v) else f"p{int(v)}").astype(object)
    f.loc[f.index[1], "player_id"] = None  # a player row
    return f


CASES_OK = [
    *[pytest.param(lambda p=p: _toy(p), id=f"toy_{p}") for p in sorted(PROVIDER_ID_KIND)],
    pytest.param(_match, id="plain"),
    pytest.param(_big_match, id="synthetic_20000_frames"),
    pytest.param(_null_frame_key, id="null_frame_key"),
    pytest.param(_pid_only_on_a_null_key_row, id="pid_only_on_a_null_key_row"),
    pytest.param(_float_nan_pid, id="float_nan_pid"),
    pytest.param(_object_pid_with_np_nan, id="object_pid_with_np_nan"),
    pytest.param(_object_pid_with_two_nan_objects, id="object_pid_with_two_nan_objects"),
    pytest.param(_duplicate_pid_in_a_frame, id="duplicate_pid_in_a_frame"),
    pytest.param(_two_ball_rows, id="two_ball_rows"),
    pytest.param(_frame_with_no_player, id="frame_with_no_player"),
    pytest.param(_null_is_ball_object, id="null_is_ball_object"),
    pytest.param(_null_is_ball_boolean, id="null_is_ball_boolean"),
    pytest.param(lambda: _match(velocity=False), id="no_velocity"),
    pytest.param(_category_team_id, id="category_team_id"),
    pytest.param(_category_frame_keys, id="category_frame_keys"),
    pytest.param(_category_frame_keys_unused, id="category_frame_keys_unused"),
    pytest.param(_category_frame_keys_with_nulls, id="category_frame_keys_with_nulls"),
    pytest.param(_float32_coords, id="float32_coords"),
    pytest.param(_unsorted_rows, id="unsorted_rows"),
    pytest.param(_empty_table, id="empty_table"),
]
CASES_RAISE = [
    pytest.param(_int64_na_pid, id="int64_na_pid"),
    pytest.param(_object_none_pid_beside_strings, id="object_none_pid_beside_strings"),
]


# ---- the equality rule's own tests ------------------------------------------------------------------


def _result() -> dict:
    return pre_index_frames_oracle(_object_pid_with_np_nan())


@pytest.mark.parametrize("fault", ["array_element", "negative_zero", "array_dtype", "list_entry", "key_set"])
def test_assert_pre_index_equal_detects(fault):
    old, new = _result(), _result()
    if fault == "array_element":
        new["px"] = new["px"].copy()
        new["px"][0, 0] += 1e-12
    elif fault == "negative_zero":  # the float rule is by bits: -0.0 is not 0.0
        old["px"], new["px"] = old["px"].copy(), new["px"].copy()
        old["px"][0, 0], new["px"][0, 0] = 0.0, -0.0
    elif fault == "array_dtype":
        new["n_valid"] = new["n_valid"].astype(np.int32)
    elif fault == "list_entry":
        new["slot_to_team_id"] = [*new["slot_to_team_id"][:-1], -7]
    else:
        del new["has_velocity"]
    with pytest.raises(AssertionError):
        assert_pre_index_equal(old, new)


def test_assert_pre_index_equal_accepts_nan_and_object_arrays():
    old, new = _result(), _result()  # two calls: each holds its own NaN in the slot maps
    assert any(isinstance(v, float) and v != v for v in old["slot_to_pid"])
    old["frame_meta_gid"] = old["frame_meta_gid"].astype(object)
    new["frame_meta_gid"] = new["frame_meta_gid"].astype(object)
    assert_pre_index_equal(old, new)


# ---- new against oracle -----------------------------------------------------------------------------


@pytest.mark.parametrize("build", CASES_OK)
def test_new_equals_oracle(build):
    frames = build()
    before = frames.copy(deep=True)
    assert_pre_index_equal(pre_index_frames_oracle(frames.copy(deep=True)), _pre_index_frames(frames))
    pd.testing.assert_frame_equal(frames, before, check_exact=True)  # the input is not changed


@pytest.mark.parametrize("build", CASES_RAISE)
def test_new_raises_what_the_oracle_raises(build):
    with pytest.raises(Exception) as want:  # the type is asserted on the next line
        pre_index_frames_oracle(build())
    with pytest.raises(type(want.value)):
        _pre_index_frames(build())


def _infer(frames: pd.DataFrame):
    try:
        return infer_ball_carrier(frames), None
    except Exception as exc:  # the same input must fail the same way on both paths
        return None, type(exc)


@pytest.mark.parametrize("build", CASES_OK)
def test_infer_ball_carrier_is_unchanged(build, monkeypatch):
    got, got_exc = _infer(build())
    monkeypatch.setattr(_ball_carrier, "_pre_index_frames", pre_index_frames_oracle)
    want, want_exc = _infer(build())
    assert got_exc is want_exc
    if want is not None:
        assert got is not None
        pd.testing.assert_frame_equal(got, want, check_exact=True, check_dtype=True)


@pytest.mark.parametrize("build", [_category_frame_keys, _category_frame_keys_unused, _category_frame_keys_with_nulls])
def test_category_frame_keys_give_no_future_warning(build):
    """Spec 4.5: `category` frame keys give no pandas `FutureWarning` (every `groupby` states `observed=`)."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        _pre_index_frames(build())
    if (
        int(pd.__version__.split(".")[0]) < 3
    ):  # non-vacuity: pandas 2 warns in the oracle (pandas 3 has no such warning)
        with pytest.warns(FutureWarning, match="observed"):
            pre_index_frames_oracle(build())


def test_the_equality_test_detects_a_fault():
    """A deliberate fault (two slots of one frame swapped) must fail the comparison."""
    frames = _match()
    faulty = _pre_index_frames(frames)
    faulty["px"] = faulty["px"].copy()
    faulty["px"][0, [0, 1]] = faulty["px"][0, [1, 0]]
    assert faulty["px"][0, 0] != faulty["px"][0, 1]
    with pytest.raises(AssertionError):
        assert_pre_index_equal(pre_index_frames_oracle(frames), faulty)


def test_the_cases_cover_the_semantics():
    """Non-vacuity: the named cases exercise B-S3, B-S5, B-S6 and the no-velocity path, and two cases raise."""
    builders = {p.id: cast("Callable[[], pd.DataFrame]", p.values[0]) for p in CASES_OK}
    out = {name: pre_index_frames_oracle(build()) for name, build in builders.items()}

    def valid(r: dict) -> np.ndarray:
        return np.arange(r["player_slots"].shape[1])[None, :] < r["n_valid"][:, None]

    assert any((r["player_slots"][valid(r)] == -1).any() for r in out.values())  # B-S5: -1 in a VALID slot
    dup = out["duplicate_pid_in_a_frame"]
    assert any(len(set(row[:n])) < n for row, n in zip(dup["player_slots"], dup["n_valid"], strict=True))  # B-S6
    assert (out["frame_with_no_player"]["n_valid"] == 0).any()
    assert (out["null_frame_key"]["n_valid"] == 0).any()  # B-S3
    assert out["no_velocity"]["has_velocity"] is False
    assert len(CASES_RAISE) == 2


@pytest.mark.usefixtures("shadow_pre_index_oracle")
def test_the_shadow_ran(shadow_pre_index_oracle):
    infer_ball_carrier(_match())
    assert shadow_pre_index_oracle["n"] > 0


# ---- structure (ADR-073) -------------------------------------------------------------------------------
# The growth guard is `tests/test_scale_guards.py::test__pre_index_frames_is_subquadratic` (`SCALE_GUARDED`).


def test_pre_index_builds_no_per_frame_object(monkeypatch):
    """The rewrite iterates no ``groupby`` and no frame-table row; the oracle trips both counters (detection)."""
    from pandas.core.groupby import DataFrameGroupBy

    from tests._perf_structural import call_counter, row_iteration_counter

    group_iters = call_counter(monkeypatch, DataFrameGroupBy, "__iter__")
    row_iters = row_iteration_counter(monkeypatch)
    _pre_index_frames(_match(n_frames=50))
    assert (group_iters["n"], row_iters["n"]) == (0, 0)
    pre_index_frames_oracle(_match(n_frames=50))
    assert group_iters["n"] >= 1 and row_iters["n"] >= 1
