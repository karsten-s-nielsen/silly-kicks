"""The shared production-shape fixtures hold the shapes the bit-identity and W9 gates rely on (plan Task 1)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.tracking.schema import TRACKING_FRAMES_COLUMNS
from silly_kicks.tracking.utils import link_actions_to_frames
from tests._prod_fixtures import (
    ACTION_LEAD_S,
    CLOCK_OFFSET_S,
    GAME_A,
    GAME_B,
    make_gamestate_slots,
    make_production_frames,
    make_single_game_inputs,
    make_two_game_inputs,
    spec_4_9_reproduction_inputs,
)


def test_two_game_inputs_share_keys():
    actions, frames = make_two_game_inputs()
    g = actions.groupby("game_id")["action_id"].apply(list)
    assert g.iloc[0] == g.iloc[1]
    first = frames["game_id"].to_numpy() == GAME_A
    fids = frames["frame_id"].to_numpy()
    assert set(fids[first].tolist()) == set(fids[~first].tolist())


def test_two_game_clock_offset_and_jitter():
    actions, frames = make_two_game_inputs()
    a, b = (actions[actions.game_id == g].reset_index(drop=True) for g in (GAME_A, GAME_B))
    np.testing.assert_allclose(b["time_seconds"] - a["time_seconds"], CLOCK_OFFSET_S - ACTION_LEAD_S, rtol=0, atol=1e-9)
    fa, fb = (frames[frames.game_id == g].reset_index(drop=True) for g in (GAME_A, GAME_B))
    np.testing.assert_array_equal(fb["time_seconds"] - fa["time_seconds"], CLOCK_OFFSET_S)
    players = ~fa["is_ball"]
    assert (fa.loc[players, "x"] != fb.loc[players, "x"]).any()  # a cross-game link changes values
    assert set(a["team_id"]) == set(b["team_id"])


def test_two_game_action_ids_link_to_different_frames():
    """ACTION_LEAD_S: one action_id names a different frame_id in each game, while the frame ids stay shared, so a
    pointer re-attached by action_id reads the other game's frame and the W9.6 property sees it (2026-11-12)."""
    actions, frames = make_two_game_inputs()
    ptr = link_actions_to_frames(actions, frames, on_low_coverage="ignore")[0].sort_values("action_row")
    n = int((actions.game_id == GAME_A).sum())
    fid = ptr["frame_id"].to_numpy(dtype="float64", na_value=np.nan)  # Int64 <NA>: na_value for pandas 2.1
    fid_a, fid_b = fid[:n], fid[n:]
    both = ~np.isnan(fid_a) & ~np.isnan(fid_b)
    assert both.sum() >= 5  # non-vacuity: most actions link in both games
    assert (fid_a[both] != fid_b[both]).all()
    assert set(fid_b[both]) <= set(frames.loc[frames.game_id == GAME_A, "frame_id"])  # ids still shared


def test_two_game_inputs_seeded():
    a1, f1 = make_two_game_inputs(3)
    a2, f2 = make_two_game_inputs(3)
    pd.testing.assert_frame_equal(f1, f2)
    pd.testing.assert_frame_equal(a1, a2)


def test_single_game_is_game_a():
    actions, frames = make_single_game_inputs()
    assert set(actions["game_id"]) == {GAME_A} and set(frames["game_id"]) == {GAME_A}
    assert len(actions) == len(make_two_game_inputs()[0]) // 2


def test_production_frames_schema():
    f = make_production_frames(n_frames=200)
    assert list(f.columns) == list(TRACKING_FRAMES_COLUMNS)
    assert {c: str(t) for c, t in f.dtypes.items()} == TRACKING_FRAMES_COLUMNS
    per_frame = f.groupby(["period_id", "frame_id"]).size()
    assert per_frame.max() == 23 and per_frame.min() < 23  # 22 players + ball, with detection gaps
    assert f.loc[~f.is_ball, "x"].isna().any()
    assert f.loc[f.is_goalkeeper, "player_id"].nunique() == 2
    assert set(f["period_id"]) == {1, 2}
    dirs = f[~f.is_ball].groupby(["period_id", "team_id"], observed=True)["team_attacking_direction"].unique()
    assert all(len(v) == 1 for v in dirs)


def test_production_frames_mixed_ids():
    f = make_production_frames(n_frames=200, mixed_ids=True)
    ids = f.loc[~f.is_ball, "player_id"]
    assert ids.dtype == object
    assert {type(v) for v in ids} == {int, str}
    early = f.loc[(~f.is_ball) & (f.frame_id < 100), "player_id"]
    assert {type(v) for v in early} == {int}


def test_gamestate_slots_have_duplicate_action_ids():
    actions, _ = make_single_game_inputs()
    slots = make_gamestate_slots(actions)
    assert len(slots) == 3
    assert slots[1]["action_id"].duplicated().any()


def test_spec_4_9_reproduction_shape():
    actions, frames = spec_4_9_reproduction_inputs()
    assert frames.groupby("game_id")["frame_id"].nunique().tolist() == [100, 100]
    assert actions["time_seconds"].tolist() == [0.40, 1.20, 2.40, 3.20]
    g8 = frames[frames.game_id == GAME_B]
    assert g8["time_seconds"].iloc[10] == 10 / 25.0 + CLOCK_OFFSET_S
