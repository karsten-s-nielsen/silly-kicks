"""W5.2-W5.3: ``compute_shape_graphs``, the multi-frame shape-graph entry point, and the single source behind
``add_shape_graph`` / ``shape_graph_xfns`` (spec 2026-11-08 tracking-batch section 4.5; plan Task 11)."""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest

from scripts._bitwise import assert_bits_equal
from scripts._frame_reflection import reflect_frames
from silly_kicks.tracking import (
    SHAPE_GRAPH_SOURCE_VALUES,
    ShapeGraphBatch,
    add_shape_graph,
    compute_shape_graph,
    compute_shape_graphs,
    infer_positions,
)
from silly_kicks.tracking.features import shape_graph_xfns
from tests._prod_fixtures import make_production_frames, make_single_game_inputs

_KEYS = ["game_id", "period_id", "frame_id", "team_id"]
_SUMMARY = [*_KEYS, "n_players", "n_edges", "density", "mean_stability", "shape_graph_source"]
_PLAYERS = [*_KEYS, "player_id", "shape_graph_source", "vertical", "horizontal", "position_label"]
_LABELS = ["vertical", "horizontal", "position_label"]


def _frames(n_frames: int = 12) -> pd.DataFrame:
    return make_production_frames(n_frames=n_frames, n_periods=2)


def _outfield(frames: pd.DataFrame) -> pd.DataFrame:
    keep = (
        ~frames["is_ball"].to_numpy(dtype=bool)
        & ~frames["is_goalkeeper"].to_numpy(dtype=bool)
        & frames["x"].notna().to_numpy()
        & frames["y"].notna().to_numpy()
    )
    return frames[keep]


def _team_frames(frames: pd.DataFrame):
    """The reference selection, in first-appearance order: (key, rows) per team-frame."""
    return list(_outfield(frames).groupby(_KEYS, sort=False, observed=True))


def _missing_key_warnings(frames: pd.DataFrame) -> tuple[ShapeGraphBatch, list[str]]:
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        batch = compute_shape_graphs(frames)
    return batch, [str(w.message) for w in record if "missing" in str(w.message)]


def test_rows_with_a_missing_key_are_counted_in_a_warning():
    # Review A-03 (spec AG4): a row with a missing game_id, period_id, frame_id or team_id is in no team-frame. The
    # call never drops it silently: one warning counts those rows. Both sides of the rule are tested.
    frames = _frames()
    clean, messages = _missing_key_warnings(frames)
    assert messages == []  # no missing key, no warning
    first = (frames["period_id"] == frames["period_id"].iloc[0]) & (frames["frame_id"] == frames["frame_id"].iloc[0])
    holed = frames.astype({"game_id": object})
    holed.loc[first, "game_id"] = None
    n_rows = len(_outfield(holed[first]))
    n_team_frames = _outfield(holed[first])["team_id"].nunique()
    assert n_rows > 0  # non-vacuity
    batch, messages = _missing_key_warnings(holed)
    assert messages == [
        f"compute_shape_graphs: {n_rows} outfield row(s) have a missing game_id, period_id, frame_id or team_id, so "
        "they are in no team-frame."
    ]
    assert len(batch.frames_summary) == len(clean.frames_summary) - n_team_frames
    # every game_id missing: no team-frame, and every outfield row is counted
    none, messages = _missing_key_warnings(frames.assign(game_id=None))
    assert len(none.frames_summary) == 0
    assert messages == [
        f"compute_shape_graphs: {len(_outfield(frames))} outfield row(s) have a missing game_id, period_id, frame_id "
        "or team_id, so they are in no team-frame."
    ]


def _direction(label) -> float | None:
    return {"ltr": 1.0, "rtl": -1.0}.get(label)


# ---- the batch equals the per-frame public calls ----------------------------------------------------------------


def test_batch_equals_per_frame_public_calls():
    frames = _frames()
    batch = compute_shape_graphs(frames)
    assert isinstance(batch, ShapeGraphBatch)
    assert list(batch.frames_summary.columns) == _SUMMARY and list(batch.players.columns) == _PLAYERS
    groups = _team_frames(frames)
    assert len(batch.frames_summary) == len(groups) > 0
    player_row = 0
    for i, (key, rows) in enumerate(groups):
        xy = rows[["x", "y"]].to_numpy(dtype=np.float64)
        summary = batch.frames_summary.iloc[i]
        assert tuple(summary[_KEYS]) == key
        sg = compute_shape_graph(xy)
        n = len(xy)
        assert summary["n_players"] == n and summary["n_edges"] == len(sg.edges)
        assert summary["density"] == len(sg.edges) / (n * (n - 1) / 2)
        assert summary["mean_stability"] == float(np.mean(sg.stabilities))
        assert summary["shape_graph_source"] == "ok"
        direction = _direction(rows["team_attacking_direction"].iloc[0])
        assert direction is not None  # precondition: the fixture labels every team-frame
        labels = infer_positions(sg, xy, direction)
        got = batch.players.iloc[player_row : player_row + n]
        assert got["player_id"].tolist() == rows["player_id"].tolist()
        assert got["vertical"].tolist() == [lb.vertical for lb in labels]
        assert got["horizontal"].tolist() == [lb.horizontal for lb in labels]
        assert got["position_label"].tolist() == [lb.label for lb in labels]
        player_row += n
    assert player_row == len(batch.players)


def test_row_order_first_appearance_no_id_sort():
    # two games, the str game first in the rows, a mixed-dtype game_id column: no sort, no raise
    one = _frames(4)
    a = one.assign(game_id=pd.Series(["g-b"] * len(one), dtype=object))
    b = one.assign(game_id=pd.Series([7] * len(one), dtype=object))
    frames = pd.concat([a, b], ignore_index=True)
    batch = compute_shape_graphs(frames)
    expected = [key for key, _rows in _team_frames(frames)]
    assert [tuple(r) for r in batch.frames_summary[_KEYS].itertuples(index=False, name=None)] == expected
    assert batch.frames_summary["game_id"].iloc[0] == "g-b"


def test_direction_from_frame_label():
    frames = _frames(4)
    batch = compute_shape_graphs(frames)
    rtl = frames["team_attacking_direction"].eq("rtl").to_numpy()
    assert rtl.any()  # precondition: an rtl team exists
    key, rows = next(g for g in _team_frames(frames) if g[1]["team_attacking_direction"].iloc[0] == "rtl")
    xy = rows[["x", "y"]].to_numpy(dtype=np.float64)
    want = infer_positions(compute_shape_graph(xy), xy, -1.0)
    got = batch.players[batch.players[_KEYS].apply(tuple, axis=1) == key]
    assert got["position_label"].tolist() == [lb.label for lb in want]

    unknown = frames.copy()
    team = unknown["team_id"].astype(object) == key[3]
    unknown.loc[team.to_numpy(), "team_attacking_direction"] = "sideways"
    out = compute_shape_graphs(unknown)
    summary = out.frames_summary[out.frames_summary["team_id"].astype(object) == key[3]]
    assert set(summary["shape_graph_source"]) == {"direction_unresolved"}
    assert summary["n_edges"].gt(0).all()  # the graph metrics are still emitted
    labels = out.players[out.players["team_id"].astype(object) == key[3]][_LABELS]
    assert labels.isna().all().all()


def test_graph_accessor_equals_retained():
    frames = _frames(4)
    lean, kept = compute_shape_graphs(frames), compute_shape_graphs(frames, retain_graphs=True)
    for key, rows in _team_frames(frames):
        recomputed = lean.graph(*key)
        assert_bits_equal(recomputed, kept.graph(*key))
        assert_bits_equal(recomputed, compute_shape_graph(rows[["x", "y"]].to_numpy(dtype=np.float64)))
    with pytest.raises(KeyError):
        lean.graph(999, 1, 0, 1)


# ---- the direction contract (TB-SPEC-57, TB-SPEC-60, A7-SPEC-90) -------------------------------------------------


def _mixed(frames: pd.DataFrame, value) -> tuple[pd.DataFrame, tuple]:
    """``frames`` with ONE outfield row of the first team-frame given ``value`` as its direction."""
    key, rows = _team_frames(frames)[0]
    out = frames.copy()
    out.loc[rows.index[1], "team_attacking_direction"] = value
    return out, key


@pytest.mark.parametrize("value", ["rtl", None], ids=["ltr+rtl", "ltr+null"])
def test_mixed_direction_raises(value):
    frames, key = _mixed(_frames(4), value)
    with pytest.raises(ValueError, match="team_attacking_direction") as err:
        compute_shape_graphs(frames)
    assert all(str(part) in str(err.value) for part in key)  # it names the team-frame


def test_goalkeeper_and_ball_rows_are_not_read():
    frames = _frames(4)
    key, rows = _team_frames(frames)[0]
    opposite = "rtl" if rows["team_attacking_direction"].iloc[0] == "ltr" else "ltr"
    in_frame = (frames["period_id"] == key[1]) & (frames["frame_id"] == key[2])
    keeper = in_frame & frames["is_goalkeeper"].to_numpy(dtype=bool) & (frames["team_id"].astype(object) == key[3])
    ball = in_frame & frames["is_ball"].to_numpy(dtype=bool)
    assert keeper.any() and ball.any()  # precondition
    out = frames.copy()
    out.loc[keeper.to_numpy(), "team_attacking_direction"] = opposite
    out.loc[ball.to_numpy(), "team_attacking_direction"] = "sideways"
    compute_shape_graphs(out)  # does not raise


def test_direction_checked_only_on_label_path():
    actions, frames = make_single_game_inputs()
    mixed, _key = _mixed(frames, "rtl")
    add_shape_graph(actions, mixed)  # does not raise
    shape_graph_xfns()[0]([actions, actions, actions], mixed)  # does not raise
    batch = compute_shape_graphs(mixed, with_positions=False)
    assert "direction_unresolved" not in set(batch.frames_summary["shape_graph_source"])
    unresolved = frames.assign(team_attacking_direction=None)
    batch = compute_shape_graphs(unresolved, with_positions=False)
    assert "direction_unresolved" not in set(batch.frames_summary["shape_graph_source"])


# ---- the players table ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("with_positions", [True, False])
def test_players_source_filter(with_positions):
    frames = _frames(4)
    frames.loc[frames["period_id"] == 2, "team_attacking_direction"] = None  # unresolved in period 2
    batch = compute_shape_graphs(frames, with_positions=with_positions)
    summary = batch.frames_summary.set_index(_KEYS)["shape_graph_source"]
    keys = list(batch.players[_KEYS].itertuples(index=False, name=None))
    assert batch.players["shape_graph_source"].tolist() == [summary.loc[k] for k in keys]
    assert set(batch.frames_summary["shape_graph_source"]) <= SHAPE_GRAPH_SOURCE_VALUES
    if with_positions:
        ok = batch.players[batch.players["shape_graph_source"] == "ok"]
        assert len(ok) > 0 and ok[_LABELS].notna().all().all()
        assert {"ok", "direction_unresolved"} <= set(summary)
    else:
        assert batch.players[_LABELS].isna().all().all()


def test_all_or_nothing_labels():
    frames = _frames(6)
    frames.loc[frames["frame_id"] % 3 == 0, "team_attacking_direction"] = None
    batch = compute_shape_graphs(frames)
    set_per_frame = (
        batch.players[_LABELS].notna().all(axis=1).groupby([batch.players[k] for k in _KEYS], sort=False, observed=True)
    )
    assert set_per_frame.nunique().eq(1).all()
    assert set_per_frame.all().any() and (~set_per_frame.any()).any()  # both kinds present


def test_chunked_calls_concat_equals_whole():
    frames = _frames(10)
    whole = compute_shape_graphs(frames)
    frame_key = frames["period_id"].astype(str) + "/" + frames["frame_id"].astype(str)
    starts = np.flatnonzero(frame_key.ne(frame_key.shift()).to_numpy())
    cuts = [0, int(starts[len(starts) // 3]), int(starts[2 * len(starts) // 3]), len(frames)]
    parts = [compute_shape_graphs(frames.iloc[lo:hi]) for lo, hi in itertools.pairwise(cuts)]
    summary = pd.concat([p.frames_summary for p in parts], ignore_index=True)
    players = pd.concat([p.players for p in parts], ignore_index=True)
    assert_bits_equal(summary, whole.frames_summary.reset_index(drop=True))
    assert_bits_equal(players, whole.players.reset_index(drop=True))


def test_batch_mirror_invariance():
    frames = _frames(10)
    assert frames["x"].dtype == np.float32  # precondition: float32 storage (ADR-106)
    assert set(frames["team_attacking_direction"].dropna()) == {"ltr", "rtl"}  # both directions
    plain = compute_shape_graphs(frames).players
    mirrored = compute_shape_graphs(reflect_frames(frames)).players
    assert plain["shape_graph_source"].eq("ok").mean() > 0.9
    assert plain[_LABELS].equals(mirrored[_LABELS])


def test_coordinates_read_as_float64():
    frames = _frames(3).astype({"x": "float64", "y": "float64"})
    frames["x"] = frames["x"] + 1e-9  # not representable in float32
    batch = compute_shape_graphs(frames)
    for key, rows in _team_frames(frames):
        assert_bits_equal(batch.graph(*key).points, rows[["x", "y"]].to_numpy(dtype=np.float64))


# ---- review-focus edge cases ------------------------------------------------------------------------------------


def test_too_few_players_and_ball_only():
    frames = _frames(3)
    key, rows = _team_frames(frames)[0]
    sparse = frames.drop(index=rows.index[2:])  # two outfield players left in that team-frame
    ball_only = frames[frames["is_ball"].to_numpy(dtype=bool)].assign(frame_id=999)
    batch = compute_shape_graphs(pd.concat([sparse, ball_only], ignore_index=True))
    first = batch.frames_summary.iloc[0]
    assert tuple(first[_KEYS]) == key and first["shape_graph_source"] == "too_few_players"
    assert first["n_players"] == 2 and first["n_edges"] == 0
    assert np.isnan(first["density"]) and np.isnan(first["mean_stability"])
    assert batch.players.iloc[:2][_LABELS].isna().all().all()
    assert 999 not in set(batch.frames_summary["frame_id"])  # a ball-only frame has no team-frame


def test_empty_frames():
    batch = compute_shape_graphs(_frames(3).iloc[:0])
    assert batch.frames_summary.empty and list(batch.frames_summary.columns) == _SUMMARY
    assert batch.players.empty and list(batch.players.columns) == _PLAYERS


def test_categorical_team_id():
    frames = _frames(4)
    cat = frames.assign(team_id=frames["team_id"].astype("category"))
    plain, categorical = compute_shape_graphs(frames), compute_shape_graphs(cat)
    assert isinstance(categorical.frames_summary["team_id"].dtype, pd.CategoricalDtype)
    assert categorical.players[_LABELS].equals(plain.players[_LABELS])
    assert categorical.frames_summary["n_edges"].equals(plain.frames_summary["n_edges"])
