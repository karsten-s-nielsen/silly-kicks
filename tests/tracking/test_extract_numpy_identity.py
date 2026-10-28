"""Spec r4 D20: the numpy extract is bit-identical to the per-frame pandas code it replaced.

The audit (HF job `6ae4bd0f0e0338d1bb7cbdf2`) found the extract stages spending most of their time on pandas object
work per frame: a DataFrame per frame, boolean masks, per-frame id comparisons. D20 turns the columns into arrays ONCE
per match and slices them per frame, keeping every per-frame computation and its argument types. The oracles are the
`2474b73` bodies kept verbatim (`tests/tracking/_extract_reference.py`); the inputs are the committed tracking frames
of every provider (with and without derived velocities) and randomized synthetic matches that hit the edge cases:
NaN positions, a missing ball, dead-ball frames, int and str ids, uneven and empty player sets, two periods, float32
storage, shuffled rows.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import _ball_carrier as BC
from silly_kicks.tracking import _ghost_gk as GG
from tests.tracking import _extract_reference as REF

PROVIDERS = ("gradientsports", "metrica", "skillcorner", "sportec")


def _provider_frames(provider: str, *, velocities: bool) -> pd.DataFrame:
    from tests.tracking._provider_inputs import load_provider_frames

    frames = load_provider_frames(provider)
    if velocities:
        from silly_kicks.tracking.preprocess import derive_velocities, smooth_frames

        frames = derive_velocities(smooth_frames(frames))
    return frames


def _idsse_frames() -> pd.DataFrame:
    import pathlib

    return pd.read_parquet(pathlib.Path(__file__).resolve().parents[1] / "datasets/tracking/idsse_half/frames.parquet")


def _synthetic(seed: int, *, ids: str, float32: bool, shuffle: bool, velocities: bool = True) -> pd.DataFrame:
    """A two-period match: uneven (sometimes empty) player sets, NaN positions, missing and dead balls."""
    rng = np.random.default_rng(seed)
    game = "g1" if ids == "str" else 7
    rows = []
    for period in (1, 2):
        for f in range(1, 31):
            if rng.random() > 0.15:  # some frames carry no ball row
                bx = np.nan if rng.random() < 0.1 else float(rng.uniform(0, 105))
                state = "dead" if rng.random() < 0.15 else "alive"
                rows.append(
                    dict(game_id=game, period_id=period, frame_id=f, player_id=None, team_id=None, is_ball=True,
                         x=bx, y=float(rng.uniform(0, 68)), vx=float(rng.normal()), vy=float(rng.normal()),
                         ball_state=state)
                )  # fmt: skip
            for j in range(int(rng.integers(0, 23))):
                pid = f"p{j}" if ids == "str" else 100 + j
                team = ("A" if j < 11 else "B") if ids == "str" else (1 if j < 11 else 2)
                x = np.nan if rng.random() < 0.05 else float(rng.uniform(0, 105))
                rows.append(
                    dict(game_id=game, period_id=period, frame_id=f, player_id=pid, team_id=team, is_ball=False,
                         x=x, y=float(rng.uniform(0, 68)), vx=float(rng.normal()), vy=float(rng.normal()),
                         ball_state="alive")
                )  # fmt: skip
    df = pd.DataFrame(rows)
    if not velocities:
        df = df.drop(columns=["vx", "vy"])
    if float32:
        for c in ("x", "y", "vx", "vy"):
            if c in df.columns:
                df[c] = df[c].astype(np.float32)
    if shuffle:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def _same_scalar(a, b) -> bool:
    if type(a) is not type(b):
        return False
    try:
        both_nan = bool(pd.isna(a)) and bool(pd.isna(b))
    except (TypeError, ValueError):
        both_nan = False
    return both_nan or bool(a == b)


def _assert_same(new: dict, old: dict) -> None:
    """Every key equal: arrays by dtype, shape and value (NaN equal to NaN), containers element by element."""
    assert list(new) == list(old)
    for k, a in old.items():
        b = new[k]
        if isinstance(a, np.ndarray):
            assert isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape, k
            if a.dtype.kind in "fc":
                assert np.array_equal(a, b, equal_nan=True), k
            else:
                assert all(_same_scalar(x, y) for x, y in zip(a.ravel().tolist(), b.ravel().tolist(), strict=True)), k
        elif isinstance(a, dict):
            assert len(a) == len(b), k
            assert all(
                _same_scalar(ka, kb) and _same_scalar(va, vb)
                for (ka, va), (kb, vb) in zip(a.items(), b.items(), strict=True)
            ), k
        elif isinstance(a, list):
            assert len(a) == len(b) and all(_same_scalar(x, y) for x, y in zip(a, b, strict=True)), k
        else:
            assert a == b, k


def _inputs():
    out = [pytest.param(lambda p=p, v=v: _provider_frames(p, velocities=v), id=f"{p}-{'vel' if v else 'novel'}")
           for p in PROVIDERS for v in (False, True)]  # fmt: skip
    out.append(pytest.param(_idsse_frames, id="idsse_half"))
    for seed in (1, 2):
        for ids in ("int", "str"):
            for f32 in (False, True):
                for shuffle in (False, True):
                    kind = f"{'f32' if f32 else 'f64'}-{'shuffled' if shuffle else 'sorted'}"
                    out.append(
                        pytest.param(
                            lambda s=seed, i=ids, f=f32, sh=shuffle: _synthetic(s, ids=i, float32=f, shuffle=sh),
                            id=f"synthetic-{seed}-{ids}-{kind}",
                        )
                    )
    out.append(
        pytest.param(lambda: _synthetic(3, ids="int", float32=False, shuffle=True, velocities=False), id="no-vel")
    )
    return out


def test_pre_index_frames_builds_no_per_frame_dataframe(monkeypatch):
    """Spec r4 D20: the per-frame groupby iteration -- one DataFrame per frame, the audited cost -- is gone. The
    identity tests below prove the arrays are unchanged; this one proves the work moved."""
    from pandas.core.groupby.generic import DataFrameGroupBy

    def _per_frame(self):
        raise AssertionError("a DataFrame per frame was built (groupby iteration)")

    monkeypatch.setattr(DataFrameGroupBy, "__iter__", _per_frame)
    pre = BC._pre_index_frames(_synthetic(1, ids="int", float32=False, shuffle=True))
    assert pre["n_valid"].sum() > 0  # non-vacuous: players were placed


@pytest.mark.parametrize("make", _inputs())
def test_pre_index_frames_equals_the_oracle(make):
    frames = make()
    _assert_same(BC._pre_index_frames(frames), REF._pre_index_frames(frames))


@pytest.mark.parametrize("make", _inputs())
def test_infer_ball_carrier_equals_the_oracle_path(make, monkeypatch):
    frames = make()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new = BC.infer_ball_carrier(frames)
        monkeypatch.setattr(BC, "_pre_index_frames", REF._pre_index_frames)
        old = BC.infer_ball_carrier(frames)
    assert list(new.columns) == list(old.columns) and dict(new.dtypes) == dict(old.dtypes)
    assert new.equals(old)


# --- R4-7: the ghost-GK extract --------------------------------------------------------------------------------------

_CARRIER_COLS = ["game_id", "period_id", "frame_id", "ball_carrier_team_id"]
_GHOST_FEATURE_SETS = ("faithful", "position_only")


def _ghost_synthetic(
    seed: int,
    *,
    ids: str,
    float32: bool,
    shuffle: bool,
    visibility: str = "bool",
    velocities: bool = True,
    time: bool = True,
) -> tuple[pd.DataFrame, Any]:
    """A two-period match for the ghost-GK extract, and its home team id.

    Team A defends x=0 in period 1 and x=105 in period 2. Keepers come and go (a frame without one, a second keeper
    of one team, a keeper with no team, whose end does not resolve); outfield sets are uneven and sometimes empty;
    some back lines are collinear (the hull's ``QhullError``); positions go NaN; the ball is sometimes missing or
    has no position. ``ids``: ``float`` (int ids that a missing value turned float), ``Int64``, ``category``,
    ``str``. A shuffle keeps the shuffled index labels, so only positional code passes.
    """
    rng = np.random.default_rng(seed)
    game = "g1" if ids == "str" else 7
    team = {"A": "A", "B": "B"} if ids == "str" else {"A": 1, "B": 2}
    rows = []
    for period in (1, 2):
        for f in range(1, 41):
            base = dict(game_id=game, period_id=period, frame_id=f, time_seconds=(period - 1) * 2700.0 + f / 25.0,
                        frame_rate=25.0, ball_state="dead" if rng.random() < 0.15 else "alive")  # fmt: skip
            if rng.random() > 0.15:
                rows.append(
                    dict(base, player_id=None, team_id=None, is_ball=True, is_goalkeeper=False, visibility=None,
                         x=np.nan if rng.random() < 0.1 else float(rng.uniform(0, 105)), y=float(rng.uniform(0, 68)),
                         vx=float(rng.normal()), vy=float(rng.normal()))
                )  # fmt: skip
            for side in ("A", "B"):
                goal = (0.0 if period == 1 else 105.0) if side == "A" else (105.0 if period == 1 else 0.0)
                n_keepers = 0 if rng.random() < 0.08 else (2 if rng.random() < 0.06 else 1)
                for k in range(n_keepers):
                    rows.append(
                        dict(base, player_id=f"{side}k{k}" if ids == "str" else (900 if side == "A" else 950) + k,
                             team_id=team[side], is_ball=False, is_goalkeeper=True,
                             visibility=bool(rng.random() < 0.9),
                             x=np.nan if rng.random() < 0.03 else abs(goal - float(rng.uniform(1, 12))),
                             y=float(rng.uniform(26, 42)), vx=float(rng.normal()), vy=float(rng.normal()))
                    )  # fmt: skip
                flat = rng.random() < 0.1  # a collinear back line: ConvexHull raises QhullError
                n_out = int(rng.integers(0, 11))
                for j in range(n_out):
                    # A NaN position only where no hull is built (2 or fewer outfield players): the `2474b73` body
                    # raises on a NaN hull point (`test_ghost_gk_nan_back_line_raises_as_before`).
                    rows.append(
                        dict(base, player_id=f"{side}{j}" if ids == "str" else (100 if side == "A" else 200) + j,
                             team_id=team[side], is_ball=False, is_goalkeeper=False, visibility=True,
                             x=np.nan if n_out <= 2 and rng.random() < 0.3 else float(rng.uniform(0, 105)),
                             y=20.0 if flat else float(rng.uniform(0, 68)), vx=float(rng.normal()),
                             vy=float(rng.normal()))
                    )  # fmt: skip
            if rng.random() < 0.05:
                rows.append(
                    dict(base, player_id="kx" if ids == "str" else 999, team_id=None, is_ball=False,
                         is_goalkeeper=True, visibility=True, x=float(rng.uniform(0, 105)),
                         y=float(rng.uniform(0, 68)), vx=0.0, vy=0.0)
                )  # fmt: skip
    df = pd.DataFrame(rows)
    home: Any = team["A"]
    if ids == "float":
        home = 1.0
    elif ids in ("Int64", "category"):
        df["team_id"] = df["team_id"].astype("Int64")
        df["player_id"] = df["player_id"].astype("Int64")
        if ids == "category":
            df["team_id"] = df["team_id"].astype("category")
    if visibility == "nullable":
        df["visibility"] = df["visibility"].astype("boolean")
    elif visibility == "absent":
        df = df.drop(columns=["visibility"])
    if not velocities:
        df = df.drop(columns=["vx", "vy"])
    if not time:
        df = df.drop(columns=["time_seconds"])
    if float32:
        for c in ("x", "y", "vx", "vy"):
            if c in df.columns:
                df[c] = df[c].astype(np.float32)
    if shuffle:
        df = df.sample(frac=1.0, random_state=seed)
    return df, home


_FRAME_KEYS = ["game_id", "period_id", "frame_id"]


def _first_frames(frames: pd.DataFrame, per_period: int) -> pd.DataFrame:
    """The rows of each period's first ``per_period`` frames, in their original order and with their index."""
    keys = frames[_FRAME_KEYS].drop_duplicates().sort_values(_FRAME_KEYS)
    head = keys.groupby(["game_id", "period_id"], sort=False).head(per_period)
    return frames[pd.MultiIndex.from_frame(frames[_FRAME_KEYS]).isin(pd.MultiIndex.from_frame(head))]


def _ghost_provider(provider: str, *, velocities: bool, window: bool = True) -> tuple[pd.DataFrame, Any]:
    """A provider's committed tracking fixture (its velocities derived on the whole fixture); by default only the
    first 75 frames of each period, so the oracle's per-frame pandas cost stays inside the main suite. The slow
    tests run the whole fixtures."""
    frames = _provider_frames(provider, velocities=velocities)
    if window:
        frames = _first_frames(frames, 75)
    return frames, frames["team_id"].dropna().iloc[0]


def _ghost_idsse() -> tuple[pd.DataFrame, Any]:
    frames = _idsse_frames()
    return frames, frames["team_id"].dropna().iloc[0]


def _ghost_full_inputs():
    """The whole committed fixtures, for the slow tests: every provider (with derived velocities) and the IDSSE
    half (15,000 frames)."""
    out = [pytest.param(lambda p=p: _ghost_provider(p, velocities=True, window=False), id=f"{p}-full")
           for p in PROVIDERS]  # fmt: skip
    out.append(pytest.param(_ghost_idsse, id="idsse_half"))
    return out


def _ghost_inputs(*, velocities_only: bool = False):
    out = [pytest.param(lambda p=p, v=v: _ghost_provider(p, velocities=v), id=f"{p}-{'vel' if v else 'novel'}")
           for p in PROVIDERS for v in ((True,) if velocities_only else (False, True))]  # fmt: skip
    variants = [
        ("float", False, False, "bool"),
        ("Int64", True, True, "nullable"),
        ("category", True, True, "bool"),
        ("str", False, True, "absent"),
        ("float", True, True, "nullable"),
    ]
    for seed in (1, 2):
        for ids, f32, shuffle, vis in variants:
            out.append(
                pytest.param(
                    lambda s=seed, i=ids, f=f32, sh=shuffle, v=vis: _ghost_synthetic(
                        s, ids=i, float32=f, shuffle=sh, visibility=v
                    ),
                    id=f"synthetic-{seed}-{ids}-{'f32' if f32 else 'f64'}-{'shuffled' if shuffle else 'sorted'}-{vis}",
                )
            )
    if not velocities_only:
        out.append(
            pytest.param(
                lambda: _ghost_synthetic(3, ids="Int64", float32=True, shuffle=True, velocities=False, time=False),
                id="synthetic-no-vel-no-time",
            )
        )
    return out


def _assert_frame_same(new: pd.DataFrame, old: pd.DataFrame) -> None:
    """The same columns, dtypes, index and cells: float cells bit for bit, every other cell by type and value."""
    assert list(new.columns) == list(old.columns)
    assert list(new.dtypes) == list(old.dtypes)
    assert new.index.equals(old.index)
    assert new.equals(old)
    for c in old.columns:
        a, b = old[c].to_numpy(), new[c].to_numpy()
        if a.dtype.kind == "f":
            bits = np.dtype(f"u{a.dtype.itemsize}")
            assert np.array_equal(a.view(bits), b.view(bits)), c
        else:
            assert all(_same_scalar(x, y) for x, y in zip(a.tolist(), b.tolist(), strict=True)), c


def _recording(log: list, fn):
    """A context callback that records each call's argument types and values, and its answer."""

    def call(game_id, time_s):
        out = fn(game_id, time_s)
        log.append((type(game_id), game_id, type(time_s), time_s, out))
        return out

    return call


def _link_subset(frames: pd.DataFrame) -> set[int]:
    return {int(f) for f in pd.unique(frames["frame_id"]) if int(f) % 3 == 0}


def _carrier(frames: pd.DataFrame) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return BC.infer_ball_carrier(frames)[_CARRIER_COLS]


def _ghost_actions(frames: pd.DataFrame) -> pd.DataFrame:
    """SPADL actions for the match: a goal, an own goal, a corner and a goal kick, at times inside the frames."""
    game = frames["game_id"].iloc[0]
    teams = list(pd.unique(frames["team_id"].dropna()))[:2]
    t = frames["time_seconds"].quantile([0.2, 0.5, 0.55, 0.8]).tolist()
    return pd.DataFrame(
        {
            "game_id": [game] * 4,
            "time_seconds": t,
            "team_id": [teams[0], teams[-1], teams[-1], teams[0]],
            "type_name": ["shot", "shot", "corner_crossed", "goalkick"],
            "result_name": ["success", "owngoal", "success", "success"],
        }
    )


def test_extract_all_ghost_gk_features_builds_no_per_frame_objects(monkeypatch):
    """Spec r4 D20: no DataFrame per frame (groupby iteration), no Series per keeper row (``iterrows``), no
    per-element ``Series.apply``. The identity tests below prove the output is unchanged; this one proves the work
    moved. A groupby walk of a handful of groups is allowed: `ordered_group_rows` iterates one small regroup to
    learn how this pandas types its keys (A review R4IMPL-C3); a per-frame walk yields one group per frame."""
    frames, home = _ghost_synthetic(1, ids="Int64", float32=True, shuffle=True)
    carrier = _carrier(frames)
    n_frames = len(frames[_FRAME_KEYS].drop_duplicates())

    def _refuse(what: str):
        def refuse(*args, **kwargs):
            raise AssertionError(f"per-frame pandas work: {what}")

        return refuse

    walked = _per_frame_walk_guard(monkeypatch)
    monkeypatch.setattr(pd.DataFrame, "iterrows", _refuse("a Series per keeper row (iterrows)"))
    monkeypatch.setattr(pd.Series, "apply", _refuse("a per-element Series.apply"))
    features, meta = GG._extract_all_ghost_gk_features(
        frames, home_team_id=home, carrier=carrier, score_at_time=lambda g, t: 1.0, phase_at_time=lambda g, t: 1
    )
    assert all(n < n_frames // 4 for n in walked), f"a per-frame groupby walk: {walked} groups, {n_frames} frames"
    assert len(features) == len(meta) > 0  # non-vacuous: keeper rows were extracted


@pytest.mark.parametrize(
    "frame",
    [
        pytest.param(lambda i=i, f=f, v=v: _ghost_synthetic(4, ids=i, float32=f, shuffle=True, visibility=v)[0],
                     id=f"{i}-{v}")
        for i, f, v in (("float", False, "bool"), ("Int64", True, "nullable"), ("category", True, "bool"),
                        ("str", False, "absent"))
    ]
    + [pytest.param(lambda: _ghost_provider("gradientsports", velocities=True)[0], id="gradientsports")],
)  # fmt: skip
def test_keeper_row_cells_are_the_iterrows_cells(frame):
    """The keeper-row cells the extract hands on (a dict key, a group key, a meta cell) are ``iterrows``' cells:
    ``DataFrame.values`` converts every column to the frame's common dtype first."""
    df = frame()
    rows = np.flatnonzero(df["is_goalkeeper"].astype(bool).to_numpy() & ~df["is_ball"].astype(bool).to_numpy())
    assert len(rows) > 0
    cells = GG._KeeperCells(df, rows)
    for column in ("team_id", "player_id", "x", "visibility"):
        if column not in df.columns:
            assert cells.get(column) is None
            continue
        expected = [r.get(column) for _, r in df.iloc[rows].iterrows()]
        got = cells.get(column)
        assert got is not None
        assert all(_same_scalar(a, b) for a, b in zip(got.tolist(), expected, strict=True)), column


@pytest.mark.parametrize("make", _ghost_inputs())
def test_extract_all_ghost_gk_features_equals_the_oracle(make):
    """Every feature set, linked or not, subsampled or not: the features, the meta and every context callback call
    (argument types included) are those of the `2474b73` body."""
    frames, home = make()
    carrier = _carrier(frames)
    link = _link_subset(frames)
    for feature_set in _GHOST_FEATURE_SETS:
        for subsample in (None, 5.0):
            for link_frame_ids in (None, link):
                logs: dict[str, list] = {"new": [], "old": []}
                out = {}
                for side, fn in (
                    ("new", GG._extract_all_ghost_gk_features),
                    ("old", REF._extract_all_ghost_gk_features),
                ):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        out[side] = fn(
                            frames,
                            home_team_id=home,
                            carrier=carrier,
                            score_at_time=_recording(logs[side], lambda g, t: float(int(t) % 3) - 1.0),
                            phase_at_time=_recording(logs[side], lambda g, t: int(t * 10) % 3),
                            subsample_fps=subsample,
                            link_frame_ids=link_frame_ids,
                            feature_set=feature_set,
                        )
                _assert_frame_same(out["new"][0], out["old"][0])
                _assert_frame_same(out["new"][1], out["old"][1])
                assert logs["new"] == logs["old"]
                assert len(out["old"][0]) > 0  # non-vacuous


@pytest.mark.parametrize("make", _ghost_inputs())
def test_prepare_ghost_gk_training_data_equals_the_oracle(make, monkeypatch):
    """The trainer's call (`prepare_ghost_gk_training_data`, `return_meta=True`) on both feature sets and both
    grids (the default 30 m and the sweeper's 52.5 m) equals the oracle path."""
    frames, home = make()
    actions = _ghost_actions(frames) if "time_seconds" in frames.columns else None
    grids = (GG.DEFAULT_GHOST_GRID, dataclasses.replace(GG.DEFAULT_GHOST_GRID, x_max=52.5))
    for feature_set in _GHOST_FEATURE_SETS:
        for grid in grids:
            kw: dict[str, Any] = dict(
                home_team_id=home,
                actions=actions,
                subsample_fps=5.0,
                feature_set=feature_set,
                grid_spec=grid,
                return_meta=True,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                new = GG.prepare_ghost_gk_training_data(frames, **kw)
                with monkeypatch.context() as m:
                    m.setattr(GG, "_extract_all_ghost_gk_features", REF._extract_all_ghost_gk_features)
                    old = GG.prepare_ghost_gk_training_data(frames, **kw)
            assert len(new) == len(old) == 3
            for a, b in zip(new, old, strict=True):
                _assert_frame_same(a, b)


def _ghost_model(feature_set: str):
    names = GG.GHOST_GK_FEATURE_NAMES_POSITION_ONLY if feature_set == "position_only" else GG.GHOST_GK_FEATURE_NAMES
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((100, len(names))), columns=list(names))
    labels = pd.DataFrame({"gk_x": rng.uniform(2, 20, 100), "gk_y": rng.uniform(25, 45, 100)})
    model = GG.GhostGkModel(n_estimators=5, feature_set=feature_set)  # type: ignore[arg-type]
    model.fit(X, labels)
    return model


@pytest.mark.parametrize("make", _ghost_inputs(velocities_only=True))
def test_ghost_gk_serving_equals_the_oracle(make, monkeypatch):
    """The serving paths (`compute_ghost_gk`, `serve_ghost_gk_positions`; their shared core restricts the extract
    to the linked frames) equal the oracle path on both feature sets."""
    frames, home = make()
    actions = _ghost_actions(frames)
    link = _link_subset(frames)
    for feature_set in _GHOST_FEATURE_SETS:
        model = _ghost_model(feature_set)
        for serve in (GG.compute_ghost_gk, GG.serve_ghost_gk_positions):
            kw: dict[str, Any] = dict(model=model, home_team_id=home, actions=actions, link_frame_ids=link)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                new = serve(frames, **kw)
                with monkeypatch.context() as m:
                    m.setattr(GG, "_extract_all_ghost_gk_features", REF._extract_all_ghost_gk_features)
                    old = serve(frames, **kw)
            _assert_frame_same(new, old)


@pytest.mark.parametrize("make", _ghost_inputs())
def test_extract_ghost_gk_features_equals_the_oracle(make):
    """The public per-frame function, now a wrapper over the array core, returns the `2474b73` row for every
    keeper team of the first frames, and for a team id of another dtype, a missing one and an unknown one."""
    frames, _home = make()
    rng = np.random.default_rng(0)
    groups = frames.groupby(["game_id", "period_id", "frame_id"], sort=True)
    for n, (_key, frame) in enumerate(groups):
        if n >= 40:
            break
        teams = list(pd.unique(frame["team_id"].dropna()))
        for team in [*teams, *(str(t) for t in teams[:1]), None, "nobody"]:
            kw: dict[str, Any] = dict(
                gk_team_id=team,
                goal_x=float(rng.choice([0.0, 105.0])),
                score_diff=float(rng.integers(-2, 3)),
                phase=int(rng.integers(0, 3)),
                ball_carrier_team_id=[team, None, "nobody"][int(rng.integers(0, 3))],
                prev_defensive_line_x=[None, float(rng.uniform(0, 40))][int(rng.integers(0, 2))],
                prev_defending_centroid_x=[None, float(rng.uniform(0, 60))][int(rng.integers(0, 2))],
                dt=float(rng.choice([0.04, 0.5])),
                feature_set=_GHOST_FEATURE_SETS[int(rng.integers(0, 2))],
            )
            _assert_frame_same(GG.extract_ghost_gk_features(frame, **kw), REF.extract_ghost_gk_features(frame, **kw))


@pytest.mark.slow
@pytest.mark.parametrize("make", _ghost_full_inputs())
def test_ghost_gk_extract_equals_the_oracle_on_whole_fixtures(make, monkeypatch):
    """Whole fixtures (slow: the oracle pays its per-frame pandas cost on every frame): the serving walk (every
    frame, the linked ones extracted) and the trainer's call (the faithful superset on the sweeper grid)."""
    frames, home = make()
    carrier = _carrier(frames)
    subsample = 5.0 if len(frames) > 100_000 else None  # the IDSSE half: 15,000 frames at 25 fps
    out = {}
    for side, fn in (("new", GG._extract_all_ghost_gk_features), ("old", REF._extract_all_ghost_gk_features)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[side] = fn(frames, home_team_id=home, carrier=carrier, subsample_fps=subsample,
                           link_frame_ids=_link_subset(frames))  # fmt: skip
    _assert_frame_same(out["new"][0], out["old"][0])
    _assert_frame_same(out["new"][1], out["old"][1])
    assert len(out["old"][0]) > 0
    kw: dict[str, Any] = dict(
        home_team_id=home,
        actions=_ghost_actions(frames),
        subsample_fps=5.0,
        feature_set="faithful",
        grid_spec=dataclasses.replace(GG.DEFAULT_GHOST_GRID, x_max=52.5),
        return_meta=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new = GG.prepare_ghost_gk_training_data(frames, **kw)
        with monkeypatch.context() as m:
            m.setattr(GG, "_extract_all_ghost_gk_features", REF._extract_all_ghost_gk_features)
            old = GG.prepare_ghost_gk_training_data(frames, **kw)
    for a, b in zip(new, old, strict=True):
        _assert_frame_same(a, b)


def test_ghost_gk_nan_back_line_raises_as_before():
    """A NaN position in a back line of three or more: the `2474b73` body raises, because ``ConvexHull`` refuses a
    NaN point with a ``ValueError``, not the ``QhullError`` it catches. The array core keeps that behaviour, and
    changing it is not D20's call."""
    rows = [
        dict(player_id=None, team_id=None, is_ball=True, is_goalkeeper=False, x=50.0, y=34.0),
        dict(player_id=1, team_id=1, is_ball=False, is_goalkeeper=True, x=5.0, y=34.0),
        dict(player_id=2, team_id=1, is_ball=False, is_goalkeeper=False, x=20.0, y=10.0),
        dict(player_id=3, team_id=1, is_ball=False, is_goalkeeper=False, x=np.nan, y=30.0),
        dict(player_id=4, team_id=1, is_ball=False, is_goalkeeper=False, x=25.0, y=50.0),
        dict(player_id=5, team_id=2, is_ball=False, is_goalkeeper=False, x=60.0, y=30.0),
        dict(player_id=6, team_id=2, is_ball=False, is_goalkeeper=True, x=100.0, y=34.0),
    ]
    frames = pd.DataFrame(
        [dict(r, game_id=7, period_id=1, frame_id=1, time_seconds=0.04, vx=0.0, vy=0.0) for r in rows]
    )
    frames["team_id"] = frames["team_id"].astype("Int64")
    for new, old in (
        (lambda: GG.extract_ghost_gk_features(frames, gk_team_id=1, goal_x=0.0),
         lambda: REF.extract_ghost_gk_features(frames, gk_team_id=1, goal_x=0.0)),
        (lambda: GG._extract_all_ghost_gk_features(frames, home_team_id=1),
         lambda: REF._extract_all_ghost_gk_features(frames, home_team_id=1)),
    ):  # fmt: skip
        with pytest.raises(ValueError) as raised_old:
            old()
        with pytest.raises(ValueError) as raised_new:
            new()
        assert str(raised_new.value) == str(raised_old.value)


def test_ids_match_answers_element_by_element():
    """The batch extract computes ``ids_match`` once per keeper team on the whole column and slices it per frame.
    That is the per-frame call only if ``ids_match`` answers each element alone. Its object-column content probe
    can pick another path for a subset (all strings in one frame, a boxed number elsewhere); the answers must not
    change."""
    from silly_kicks.id_compat import ids_match

    rng = np.random.default_rng(7)
    columns = [
        pd.Series([1, 2, None, 2, 1] * 8, dtype="Int64"),
        pd.Series([1.0, 2.0, np.nan, 2.0, 1.0] * 8),
        pd.Series(pd.Categorical([1, 2, None, 2, 1] * 8)),
        pd.Series(["A", "B", None, "B", "A"] * 8, dtype=object),
        pd.Series(["A", 2, None, 2.0, "A", "2", 1, np.nan] * 5, dtype=object),
    ]
    scalars = [1, 2, 2.0, np.int64(2), "2", "A", None, np.nan, "nobody"]
    for column in columns:
        for scalar in scalars:
            whole = ids_match(column, scalar).to_numpy()
            for _ in range(40):
                rows = np.sort(rng.choice(len(column), size=int(rng.integers(1, 9)), replace=False))
                part = ids_match(column.iloc[rows], scalar).to_numpy()
                assert np.array_equal(part, whole[rows]), (column.dtype, scalar, rows)


# --- R4-8: the xCross extract ----------------------------------------------------------------------------------------


def _with_carriers(frames: pd.DataFrame, seed: int) -> pd.DataFrame:
    """The ghost-GK synthetic match with one outfield player beside the ball in most frames, so the carrier, the
    possession and the carrier-anchored features resolve (random positions alone rarely put a player in range)."""
    rng = np.random.default_rng(seed)
    out = frames.copy()
    keys = out[_FRAME_KEYS].astype(str).agg("|".join, axis=1).to_numpy()
    is_ball = out["is_ball"].astype(bool).to_numpy()
    outfield = ~is_ball & ~out["is_goalkeeper"].astype(bool).to_numpy()
    ball_x, ball_y = out["x"].to_numpy()[is_ball], out["y"].to_numpy()[is_ball]
    ball_at = {k: (x, y) for k, x, y in zip(keys[is_ball], ball_x, ball_y, strict=True)}
    x, y = out["x"].to_numpy(dtype="float64").copy(), out["y"].to_numpy(dtype="float64").copy()
    placed: set[str] = set()
    for i in np.flatnonzero(outfield):
        k = keys[i]
        if k in placed or k not in ball_at or rng.random() < 0.2:
            continue
        bx, by = ball_at[k]
        if np.isnan(bx) or np.isnan(by):
            continue
        x[i], y[i] = bx + rng.uniform(-0.5, 0.5), by + rng.uniform(-0.5, 0.5)
        placed.add(k)
    out["x"] = x.astype(out["x"].dtype)
    out["y"] = y.astype(out["y"].dtype)
    return out


def _wide_ball(frames: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Most ball rows moved into a wide, advanced corridor, so the wide-area domain filter keeps frames."""
    rng = np.random.default_rng(seed)
    out = frames.copy()
    is_ball = out["is_ball"].astype(bool).to_numpy()
    x, y = out["x"].to_numpy(dtype="float64").copy(), out["y"].to_numpy(dtype="float64").copy()
    for i in np.flatnonzero(is_ball):
        if np.isnan(x[i]) or rng.random() < 0.2:
            continue
        x[i] = rng.choice([rng.uniform(75, 104), rng.uniform(1, 30)])
        y[i] = rng.choice([rng.uniform(1, 13), rng.uniform(55, 67)])
    out["x"] = x.astype(out["x"].dtype)
    out["y"] = y.astype(out["y"].dtype)
    return out


def _cross_synthetic(seed: int, **kw) -> tuple[pd.DataFrame, Any]:
    frames, home = _ghost_synthetic(seed, **kw)
    return _with_carriers(_wide_ball(frames, seed), seed), home


def _cross_inputs(*, velocities_only: bool = False):
    out = [pytest.param(lambda p=p, v=v: _ghost_provider(p, velocities=v), id=f"{p}-{'vel' if v else 'novel'}")
           for p in PROVIDERS for v in ((True,) if velocities_only else (False, True))]  # fmt: skip
    variants = [
        ("float", False, False, "bool"),
        ("Int64", True, True, "nullable"),
        ("category", True, True, "bool"),
        ("str", False, True, "absent"),
    ]
    for seed in (1, 2):
        for ids, f32, shuffle, vis in variants:
            kind = f"{'f32' if f32 else 'f64'}-{'shuffled' if shuffle else 'sorted'}"
            out.append(
                pytest.param(
                    lambda s=seed, i=ids, f=f32, sh=shuffle, v=vis: _cross_synthetic(
                        s, ids=i, float32=f, shuffle=sh, visibility=v
                    ),
                    id=f"synthetic-{seed}-{ids}-{kind}-{vis}",
                )
            )
    if not velocities_only:
        out.append(
            pytest.param(
                lambda: _cross_synthetic(3, ids="Int64", float32=True, shuffle=True, velocities=False),
                id="synthetic-no-vel",
            )
        )
    return out


def _spadl_actions(frames: pd.DataFrame) -> pd.DataFrame:
    """SPADL actions inside the frames: crosses and shots (a goal and an own goal) by both teams."""
    from silly_kicks.spadl import config as spadlconfig

    game = frames["game_id"].iloc[0]
    teams = list(pd.unique(frames["team_id"].dropna()))[:2]
    periods = sorted(pd.unique(frames["period_id"]))
    rows = []
    for k, q in enumerate((0.1, 0.25, 0.4, 0.55, 0.7, 0.85)):
        period = periods[k % len(periods)]
        t = float(frames["time_seconds"][frames["period_id"] == period].quantile(q))
        kind = ("cross", "shot", "cross", "shot", "cross", "cross")[k]
        result = ("success", "success", "fail", "owngoal", "success", "fail")[k]
        rows.append(
            dict(game_id=game, period_id=period, time_seconds=t, team_id=teams[k % len(teams)],
                 type_id=spadlconfig.actiontype_id[kind], type_name=kind,
                 result_id=spadlconfig.result_id[result], result_name=result)
        )  # fmt: skip
    return pd.DataFrame(rows)


def _per_frame_walk_guard(monkeypatch) -> list[int]:
    """Count the groups every ``DataFrameGroupBy`` walk yields. A per-frame walk yields one group per frame; the
    per-game walks of the score and label helpers yield a handful."""
    from pandas.core.groupby.generic import DataFrameGroupBy

    walked: list[int] = []
    original = DataFrameGroupBy.__iter__

    def counting(self):
        n = 0
        for item in original(self):
            n += 1
            yield item
        walked.append(n)

    monkeypatch.setattr(DataFrameGroupBy, "__iter__", counting)
    return walked


def test_xcross_batch_builds_no_per_frame_objects(monkeypatch):
    """Spec r4 D20: neither xCross batch walks its frames as one DataFrame per frame."""
    from silly_kicks.tracking import _xcross_attempt as XC

    frames, home = _cross_synthetic(1, ids="Int64", float32=True, shuffle=True)
    actions = _spadl_actions(frames)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X, y, _ = XC.prepare_xcross_training_data(frames, actions, home_team_id=home)
        model = XC.XCrossAttemptModel().fit(X, pd.Series(y))
        n_frames = len(frames[_FRAME_KEYS].drop_duplicates())
        walked = _per_frame_walk_guard(monkeypatch)
        features, labels, _groups = XC.prepare_xcross_training_data(frames, actions, home_team_id=home)
        out = XC.compute_xcross_attempt(frames, model=model, home_team_id=home, actions=actions)
    assert all(n < n_frames // 4 for n in walked), f"a per-frame groupby walk: {walked} groups, {n_frames} frames"
    assert len(features) == len(labels) > 0  # non-vacuous
    assert out["xcross_attempt"].notna().any()


def _outcome(fn, *args, **kwargs) -> tuple:
    """``fn``'s result, or the exception it raised (type and message): the `2474b73` bodies refuse some inputs
    (a missing team id, a float frame id under ``link_frame_ids``), and the array core must refuse them alike."""
    try:
        return ("ok", fn(*args, **kwargs))
    except Exception as exc:
        return ("raised", type(exc), str(exc))


def _assert_outcome_same(new: tuple, old: tuple, same) -> None:
    assert new[0] == old[0], (new[1:] if new[0] == "raised" else "ok", old[1:] if old[0] == "raised" else "ok")
    if old[0] == "raised":
        assert new[1:] == old[1:]
    else:
        same(new[1], old[1])


def _assert_array_same(new: np.ndarray, old: np.ndarray) -> None:
    assert isinstance(new, np.ndarray) and new.dtype == old.dtype and new.shape == old.shape
    if old.dtype.kind == "f":
        bits = np.dtype(f"u{old.dtype.itemsize}")
        assert np.array_equal(old.view(bits), new.view(bits))
    else:
        assert all(_same_scalar(a, b) for a, b in zip(old.tolist(), new.tolist(), strict=True))


def _assert_value_same(new, old) -> None:
    """A box-detail value: arrays by dtype and bits, scalars by type and value (NaN equal to NaN)."""
    if isinstance(old, np.ndarray):
        _assert_array_same(new, old)
    else:
        assert _same_scalar(new, old), (new, old)


def _assert_meta_same(new: pd.DataFrame, old: pd.DataFrame) -> None:
    """The xCross meta frame: the per-row behind-line arrays by dtype and bits, every other column as
    ``_assert_frame_same``."""
    arrays = "behind_line_box_gr_x"
    assert list(new.columns) == list(old.columns)
    if arrays in old.columns:
        for a, b in zip(new[arrays], old[arrays], strict=True):
            _assert_array_same(a, b)
        new, old = new.drop(columns=[arrays]), old.drop(columns=[arrays])
    _assert_frame_same(new, old)


def _assert_row_same(new: tuple, old: tuple) -> None:
    """``extract_xcross_features(..., return_box_detail=True)``: the row, and the box detail key by key."""
    _assert_frame_same(new[0], old[0])
    assert list(new[1]) == list(old[1])
    for key in old[1]:
        _assert_value_same(new[1][key], old[1][key])


def _xcross_model(feature_set: str):
    from silly_kicks.tracking import _xcross_attempt as XC

    position_only = feature_set == "position_only"
    names = XC.XCROSS_FEATURE_NAMES_POSITION_ONLY if position_only else XC.XCROSS_FEATURE_NAMES_FAITHFUL
    rng = np.random.default_rng(5)
    X = pd.DataFrame(rng.normal(size=(120, len(names))), columns=list(names))
    y = pd.Series((rng.random(120) > 0.6).astype(int))
    return XC.XCrossAttemptModel(feature_set=feature_set).fit(X, y)  # type: ignore[arg-type]


@pytest.mark.parametrize("make", _cross_inputs())
def test_prepare_xcross_training_data_equals_the_oracle(make):
    """Both feature sets, the wide-area domain on and off, with and without the meta frame: the features, labels,
    groups, meta and warnings are those of the `2474b73` body."""
    from silly_kicks.tracking import _xcross_attempt as XC

    frames, home = make()
    actions = _spadl_actions(frames)
    n_rows = 0
    for feature_set in ("faithful", "position_only"):
        for wide in (True, False):
            for return_meta in (False, True):
                kw: dict[str, Any] = dict(
                    home_team_id=home, feature_set=feature_set, wide_area_only=wide, return_meta=return_meta
                )
                out, caught = {}, {}
                for side, fn in (("new", XC.prepare_xcross_training_data), ("old", REF.prepare_xcross_training_data)):
                    with warnings.catch_warnings(record=True) as seen:
                        warnings.simplefilter("always")
                        out[side] = fn(frames, actions, **kw)
                    caught[side] = [(w.category, str(w.message)) for w in seen]
                assert caught["new"] == caught["old"]
                new, old = out["new"], out["old"]
                assert len(new) == len(old) == (4 if return_meta else 3)
                _assert_frame_same(new[0], old[0])
                _assert_array_same(new[1], old[1])
                _assert_array_same(new[2], old[2])
                if return_meta:
                    _assert_meta_same(new[3], old[3])
                n_rows += len(old[0])
    assert n_rows > 0  # non-vacuous


@pytest.mark.parametrize("make", _cross_inputs(velocities_only=True))
def test_compute_xcross_attempt_equals_the_oracle(make):
    """The serving path, both feature sets, every frame and the linked ones only."""
    from silly_kicks.tracking import _xcross_attempt as XC

    frames, home = make()
    actions = _spadl_actions(frames)
    scored = 0
    for feature_set in ("faithful", "position_only"):
        model = _xcross_model(feature_set)
        for link in (None, _link_subset(frames)):
            kw: dict[str, Any] = dict(model=model, home_team_id=home, actions=actions, link_frame_ids=link)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                new = _outcome(XC.compute_xcross_attempt, frames, **kw)
                old = _outcome(REF.compute_xcross_attempt, frames, **kw)
            _assert_outcome_same(new, old, _assert_frame_same)
            if old[0] == "ok":
                scored += int(old[1]["xcross_attempt"].notna().sum())
    assert scored > 0  # non-vacuous


@pytest.mark.parametrize("make", _cross_inputs())
def test_extract_xcross_features_equals_the_oracle(make):
    """The public per-frame function, now a wrapper over the array core, returns the `2474b73` row (and box
    detail) for the first frames, every team and carrier id, a missing and an unknown one."""
    from silly_kicks.tracking import _xcross_attempt as XC

    frames, _home = make()
    rng = np.random.default_rng(0)
    for n, (_key, frame) in enumerate(frames.groupby(_FRAME_KEYS, sort=True)):
        if n >= 40:
            break
        teams = list(pd.unique(frame["team_id"].dropna()))
        players = list(pd.unique(frame["player_id"].dropna()))[:3]
        for team in [*teams, *(str(t) for t in teams[:1]), None, "nobody"]:
            for carrier in [*players, None, np.nan, "nobody"]:
                kw: dict[str, Any] = dict(
                    gk_team_id=team,
                    goal_x=float(rng.choice([0.0, 105.0])),
                    carrier_player_id=carrier,
                    feature_set=("faithful", "position_only")[int(rng.integers(0, 2))],
                    score_differential=[np.nan, None, float(rng.integers(-2, 3))][int(rng.integers(0, 3))],
                    return_box_detail=bool(rng.integers(0, 2)),
                )
                new = _outcome(XC.extract_xcross_features, frame, **kw)
                old = _outcome(REF.extract_xcross_features, frame, **kw)
                _assert_outcome_same(new, old, _assert_row_same if kw["return_box_detail"] else _assert_frame_same)


@pytest.mark.slow
@pytest.mark.parametrize("make", _ghost_full_inputs())
def test_xcross_equals_the_oracle_on_whole_fixtures(make):
    """Whole fixtures (slow: the oracle pays its per-frame pandas cost on every frame): the trainer's call and
    the serving path."""
    from silly_kicks.tracking import _xcross_attempt as XC

    frames, home = make()
    actions = _spadl_actions(frames)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new = XC.prepare_xcross_training_data(frames, actions, home_team_id=home, return_meta=True)
        old = REF.prepare_xcross_training_data(frames, actions, home_team_id=home, return_meta=True)
        model = _xcross_model("faithful")
        link = _link_subset(frames)
        serve: dict[str, Any] = dict(model=model, home_team_id=home, actions=actions, link_frame_ids=link)
        new_serve = _outcome(XC.compute_xcross_attempt, frames, **serve)
        old_serve = _outcome(REF.compute_xcross_attempt, frames, **serve)
    _assert_frame_same(new[0], old[0])
    _assert_array_same(new[1], old[1])
    _assert_array_same(new[2], old[2])
    _assert_meta_same(new[3], old[3])
    _assert_outcome_same(new_serve, old_serve, _assert_frame_same)


# --- R4-9: the xShot extract -----------------------------------------------------------------------------------------


def _alive(frames: pd.DataFrame, home: Any) -> tuple[pd.DataFrame, Any]:
    """A missing ``ball_state`` read as ``alive``. The metrica and skillcorner fixtures carry no ball state, and
    xShot's training domain keeps only ``alive`` frames, so without this both paths would return nothing."""
    if "ball_state" in frames.columns and frames["ball_state"].isna().any():
        frames = frames.assign(
            ball_state=frames["ball_state"].astype(object).where(frames["ball_state"].notna(), "alive")
        )
    return frames, home


def _shot_inputs(*, velocities_only: bool = False):
    """The xCross inputs (their balls sit near either goal, so the attacking-third domain keeps frames), with a
    missing ``ball_state`` read as ``alive``."""
    out = []
    for param in _cross_inputs(velocities_only=velocities_only):
        make = cast("Callable[[], tuple[pd.DataFrame, Any]]", param.values[0])
        out.append(pytest.param(lambda m=make: _alive(*m()), id=param.id))
    return out


def _xshot_model(feature_set: str):
    from silly_kicks.tracking import _xshot_occurrence as XS

    position_only = feature_set == "position_only"
    names = XS.XSHOT_FEATURE_NAMES_POSITION_ONLY if position_only else XS.XSHOT_FEATURE_NAMES_FAITHFUL
    rng = np.random.default_rng(6)
    X = pd.DataFrame(rng.normal(size=(150, len(names))), columns=list(names))
    y = pd.Series((rng.random(150) > 0.7).astype(int))
    return XS.XShotOccurrenceModel(feature_set=feature_set, params={"random_state": 42}).fit(X, y)  # type: ignore[arg-type]


def test_xshot_batch_builds_no_per_frame_objects(monkeypatch):
    """Spec r4 D20: neither xShot batch walks its frames as one DataFrame per frame."""
    from silly_kicks.tracking import _xshot_occurrence as XS

    frames, home = _cross_synthetic(1, ids="Int64", float32=True, shuffle=True)
    shots = _spadl_actions(frames)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _xshot_model("faithful")
        n_frames = len(frames[_FRAME_KEYS].drop_duplicates())
        walked = _per_frame_walk_guard(monkeypatch)
        features, labels, _groups = XS.prepare_xshot_training_data(
            frames, shots, home_team_id=home, attacking_third_only=False
        )
        out = XS.compute_xshot_occurrence(frames, model=model, home_team_id=home)
    assert all(n < n_frames // 4 for n in walked), f"a per-frame groupby walk: {walked} groups, {n_frames} frames"
    assert len(features) == len(labels) > 0  # non-vacuous
    assert out["xshot_occurrence"].notna().any()


@pytest.mark.parametrize("make", _shot_inputs())
def test_prepare_xshot_training_data_equals_the_oracle(make):
    """Both feature sets, the attacking-third domain on and off, with and without the keys: the features,
    labels, groups, keys and warnings are those of the `2474b73` body."""
    from silly_kicks.tracking import _xshot_occurrence as XS

    frames, home = make()
    shots = _spadl_actions(frames)
    n_rows = 0
    for feature_set in ("faithful", "position_only"):
        for third in (True, False):
            for return_keys in (False, True):
                kw: dict[str, Any] = dict(
                    home_team_id=home, feature_set=feature_set, attacking_third_only=third, return_keys=return_keys
                )
                out, caught = {}, {}
                for side, fn in (("new", XS.prepare_xshot_training_data), ("old", REF.prepare_xshot_training_data)):
                    with warnings.catch_warnings(record=True) as seen:
                        warnings.simplefilter("always")
                        out[side] = fn(frames, shots, **kw)
                    caught[side] = [(w.category, str(w.message)) for w in seen]
                assert caught["new"] == caught["old"]
                new, old = out["new"], out["old"]
                assert len(new) == len(old) == (4 if return_keys else 3)
                _assert_frame_same(new[0], old[0])
                _assert_array_same(new[1], old[1])
                _assert_array_same(new[2], old[2])
                if return_keys:
                    _assert_frame_same(new[3], old[3])
                n_rows += len(old[0])
    assert n_rows > 0  # non-vacuous


@pytest.mark.parametrize("make", _shot_inputs(velocities_only=True))
def test_compute_xshot_occurrence_equals_the_oracle(make):
    """The serving path, both feature sets, every frame and the linked ones only, warnings included."""
    from silly_kicks.tracking import _xshot_occurrence as XS

    frames, home = make()
    scored = 0
    for feature_set in ("faithful", "position_only"):
        model = _xshot_model(feature_set)
        for link in (None, _link_subset(frames)):
            kw: dict[str, Any] = dict(model=model, home_team_id=home, link_frame_ids=link)
            out, caught = {}, {}
            for side, fn in (("new", XS.compute_xshot_occurrence), ("old", REF.compute_xshot_occurrence)):
                with warnings.catch_warnings(record=True) as seen:
                    warnings.simplefilter("always")
                    out[side] = _outcome(fn, frames, **kw)
                caught[side] = [(w.category, str(w.message)) for w in seen]
            assert caught["new"] == caught["old"]
            _assert_outcome_same(out["new"], out["old"], _assert_frame_same)
            if out["old"][0] == "ok":
                scored += int(out["old"][1]["xshot_occurrence"].notna().sum())
    assert scored > 0  # non-vacuous


@pytest.mark.parametrize("make", _shot_inputs())
def test_extract_xshot_features_equals_the_oracle(make):
    """The public per-frame function, now a wrapper over the array core, returns the `2474b73` row for the first
    frames, every team id, one of another dtype, a missing one and an unknown one."""
    from silly_kicks.tracking import _xshot_occurrence as XS

    frames, _home = make()
    rng = np.random.default_rng(0)
    for n, (_key, frame) in enumerate(frames.groupby(_FRAME_KEYS, sort=True)):
        if n >= 40:
            break
        teams = list(pd.unique(frame["team_id"].dropna()))
        for team in [*teams, *(str(t) for t in teams[:1]), None, "nobody"]:
            for feature_set in ("faithful", "position_only"):
                kw: dict[str, Any] = dict(
                    gk_team_id=team, goal_x=float(rng.choice([0.0, 105.0])), feature_set=feature_set
                )
                new = _outcome(XS.extract_xshot_features, frame, **kw)
                old = _outcome(REF.extract_xshot_features, frame, **kw)
                _assert_outcome_same(new, old, _assert_frame_same)


@pytest.mark.slow
@pytest.mark.parametrize("make", _ghost_full_inputs())
def test_xshot_equals_the_oracle_on_whole_fixtures(make):
    """Whole fixtures (slow: the oracle pays its per-frame pandas cost on every frame): the trainer's call and
    the serving path."""
    from silly_kicks.tracking import _xshot_occurrence as XS

    frames, home = make()
    shots = _spadl_actions(frames)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kw: dict[str, Any] = dict(home_team_id=home, return_keys=True)
        new = XS.prepare_xshot_training_data(frames, shots, **kw)
        old = REF.prepare_xshot_training_data(frames, shots, **kw)
        serve: dict[str, Any] = dict(model=_xshot_model("faithful"), home_team_id=home, link_frame_ids=None)
        new_serve = _outcome(XS.compute_xshot_occurrence, frames, **serve)
        old_serve = _outcome(REF.compute_xshot_occurrence, frames, **serve)
    for a, b in zip(new[:1] + new[3:], old[:1] + old[3:], strict=True):
        _assert_frame_same(a, b)
    _assert_array_same(new[1], old[1])
    _assert_array_same(new[2], old[2])
    _assert_outcome_same(new_serve, old_serve, _assert_frame_same)


def test_canonical_id_series_answers_element_by_element():
    """The xCross row reads ``canonical_id_series`` once on the whole column and slices it per frame. That is the
    per-frame read only if it answers each element alone: its object-column probe can pick another path for a
    subset (all strings in one frame, a boxed number elsewhere); the answers must not change."""
    from silly_kicks.id_compat import canonical_id_series

    rng = np.random.default_rng(8)
    columns = [
        pd.Series([1, 2, None, 2, 1] * 8, dtype="Int64"),
        pd.Series([1.0, 2.0, np.nan, 2.0, 1.0] * 8),
        pd.Series(pd.Categorical([1, 2, None, 2, 1] * 8)),
        pd.Series(["A", "B", None, "B", "A"] * 8, dtype=object),
        pd.Series(["A", 2, None, 2.0, "A", "2", 1, np.nan] * 5, dtype=object),
        pd.Series([366.0, 1e20, -0.0, 2.5, np.inf] * 8),
    ]
    for column in columns:
        whole = canonical_id_series(column).fillna("\x00").to_numpy()
        for _ in range(60):
            rows = np.sort(rng.choice(len(column), size=int(rng.integers(1, 9)), replace=False))
            part = canonical_id_series(column.iloc[rows]).fillna("\x00").to_numpy()
            assert all(_same_scalar(a, b) for a, b in zip(part.tolist(), whole[rows].tolist(), strict=True)), (
                column.dtype,
                rows,
            )
