"""W9.1 / W9.2: the game-aware linkers (spec 2026-11-08 tracking-batch §4.9, owner rulings R5, R7, R9).

A two-game call equals the one-game calls on each game, and input without ``game_id`` links as one game. The
warning and error texts are pinned as literals (Hyrum). Single-game bit identity to the pre-change linker was proven
against a frozen oracle, which ended with ruling C5 (spec 2026-11-08 tracking-batch, section 7); the existing linker
tests pin single-game behaviour.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from scripts._bitwise import assert_bits_equal
from silly_kicks.tracking._link_identity import action_fingerprints
from silly_kicks.tracking.utils import link_actions_to_frames, link_actions_to_frames_elastic, slice_around_event
from tests._prod_fixtures import (
    CLOCK_OFFSET_S,
    GAME_A,
    GAME_B,
    make_single_game_inputs,
    make_two_game_inputs,
    spec_4_9_reproduction_inputs,
)

_POINTER_COLUMNS = ["action_id", "frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]
_NEW_COLUMNS = ["action_row", "action_fingerprint"]


def test_two_game_reproduction_links_own_game():
    actions, frames = spec_4_9_reproduction_inputs()
    ptr, rep = link_actions_to_frames(actions, frames)
    assert ptr.sort_values("action_row")["frame_id"].tolist() == [10, 30, 10, 30]
    assert list(ptr.columns) == _POINTER_COLUMNS + _NEW_COLUMNS
    assert ptr["action_row"].dtype == np.int64 and ptr["action_fingerprint"].dtype == np.uint64
    assert rep.per_game_period_link_rate == {("7", 1): 1.0, ("8", 1): 1.0}
    # the game split of the linker itself: each game's pointers equal the linker on that game alone
    for game in (GAME_A, GAME_B):
        mask = (actions["game_id"] == game).to_numpy()
        alone, _ = link_actions_to_frames(
            actions[mask].reset_index(drop=True), frames[frames["game_id"] == game].reset_index(drop=True)
        )
        got = ptr.set_index("action_row").loc[np.flatnonzero(mask), _POINTER_COLUMNS].reset_index(drop=True)
        assert_bits_equal(got, alone.sort_values("action_row")[_POINTER_COLUMNS].reset_index(drop=True))


def test_slice_around_event_game_split():
    # Review A-02 (owner ruling 2026-11-23): with game_id on both frames, a window reads only its own game's frames,
    # so the two-game call equals each game alone (G7).
    actions, frames = make_two_game_inputs()
    both = slice_around_event(actions, frames, pre_seconds=0.5, post_seconds=0.5)
    for game in (GAME_A, GAME_B):
        mask = (actions["game_id"] == game).to_numpy()
        alone = slice_around_event(
            actions[mask].reset_index(drop=True),
            frames[frames["game_id"] == game].reset_index(drop=True),
            pre_seconds=0.5,
            post_seconds=0.5,
        )
        assert len(alone) > 0  # non-vacuity
        assert_bits_equal(both[(both["game_id"] == game).to_numpy()].reset_index(drop=True), alone)


def test_pointer_fingerprint_binds_the_indexed_action():
    actions, frames = make_two_game_inputs()
    ptr, _ = link_actions_to_frames(actions, frames, on_low_coverage="ignore")
    expected = action_fingerprints(actions)[ptr["action_row"].to_numpy()]
    assert ptr["action_fingerprint"].to_numpy().tolist() == expected.tolist()
    assert sorted(ptr["action_row"].tolist()) == list(range(len(actions)))


def test_row_order_is_period_time_then_game_rank():
    actions, frames = spec_4_9_reproduction_inputs()
    # both games at the same clock: ties in time; game 8 appears first in actions, so it ranks first
    acts = actions.copy()
    acts.loc[acts["game_id"] == GAME_B, "time_seconds"] = [0.40, 1.20]  # exact ties (2.40 - 2.0 is not 0.40)
    frs = frames.copy()
    frs.loc[frs["game_id"] == GAME_B, "time_seconds"] -= CLOCK_OFFSET_S
    acts = pd.concat([acts.iloc[2:], acts.iloc[:2]], ignore_index=True)  # rows: game 8 (0, 1), game 7 (2, 3)
    ptr, _ = link_actions_to_frames(acts, frs)
    assert ptr["action_row"].tolist() == [0, 2, 1, 3]


def test_actions_without_game_id_link_as_one_game():
    # the linker contract: "When either side has no game_id column, the call is treated as one game"
    actions, frames = make_single_game_inputs()
    no_game = actions.drop(columns="game_id")
    new, _ = link_actions_to_frames(no_game, frames, on_low_coverage="ignore")
    one_game, _ = link_actions_to_frames(actions, frames, on_low_coverage="ignore")
    assert_bits_equal(new[_POINTER_COLUMNS], one_game[_POINTER_COLUMNS])
    assert new["frame_id"].notna().any()  # non-vacuity
    assert new["action_fingerprint"].to_numpy().tolist() == action_fingerprints(no_game)[new["action_row"]].tolist()


def test_single_game_elastic_pointer_columns():
    actions, frames = make_single_game_inputs()
    new, _ = link_actions_to_frames_elastic(actions, frames, on_low_coverage="ignore")
    assert list(new.columns) == _POINTER_COLUMNS + _NEW_COLUMNS
    assert new["action_row"].tolist() == list(range(len(actions)))
    assert new["frame_id"].notna().any()  # non-vacuity


def _low_coverage_single_game():
    actions, frames = make_single_game_inputs()
    acts = actions.copy()
    acts["time_seconds"] = acts["time_seconds"] + 5000.0  # a time-base mismatch: nothing links, the hint fires
    return acts, frames


#: The single-game low-coverage text, as at 4.139 (Hyrum: callers match on it). Copied from the pre-change linker's
#: output when the oracles ended (ruling C5).
_LOW_COVERAGE_TEXT = (
    "link_actions_to_frames: period 1 link_rate 0.00 (7 actions, 7 unlinked) below min_link_rate 0.5. "
    "period 1: actions [5010, 5051.5] vs frames [8.6, 50] — near-disjoint (overlap 0.00); suspected "
    "period-relative/absolute time-base mismatch. See the time-base contract in the docstring."
)


def test_single_game_warning_text_pinned():
    acts, frames = _low_coverage_single_game()
    with warnings.catch_warnings(record=True) as new_w:
        warnings.simplefilter("always")
        link_actions_to_frames(acts, frames)
    assert [str(w.message) for w in new_w if issubclass(w.category, UserWarning)] == [_LOW_COVERAGE_TEXT]


def test_single_game_error_text_pinned():
    acts, frames = _low_coverage_single_game()
    with pytest.raises(ValueError) as new_e:
        link_actions_to_frames(acts, frames, on_low_coverage="raise")
    assert str(new_e.value) == _LOW_COVERAGE_TEXT


def test_multigame_floor_names_the_bad_game():
    actions, frames = spec_4_9_reproduction_inputs()
    acts = actions.copy()
    acts.loc[acts["game_id"] == GAME_B, "time_seconds"] += 500.0  # game 8 links nothing; game 7 links all
    _, rep = link_actions_to_frames(acts, frames, on_low_coverage="ignore")
    assert rep.per_period_link_rate == {1: 0.5}  # the pooled period rate keeps its meaning
    assert rep.per_game_period_link_rate == {("7", 1): 1.0, ("8", 1): 0.0}
    with pytest.raises(ValueError, match=r"game 8 period 1 link_rate 0\.00 \(2 actions, 2 unlinked\)"):
        link_actions_to_frames(acts, frames, on_low_coverage="raise")


def _with_na_game(actions, frames):
    acts, frs = actions.copy(), frames.copy()
    acts["game_id"] = pd.Series([GAME_A, GAME_A, None, None], dtype=object)
    frs["game_id"] = pd.Series(np.where(frs["game_id"] == GAME_B, None, frs["game_id"]), dtype=object)
    return acts, frs


def test_na_game_links_only_na_frames():  # Review Focus 2
    acts, frs = _with_na_game(*spec_4_9_reproduction_inputs())
    ptr, rep = link_actions_to_frames(acts, frs)
    assert ptr.sort_values("action_row")["frame_id"].tolist() == [10, 30, 10, 30]
    assert set(rep.per_game_period_link_rate) == {("7", 1), ("<NA>", 1)}
    # NA-game actions with no NA-game frames stay unlinked: never a link into a real game
    ptr2, _ = link_actions_to_frames(acts, frs[frs["game_id"].notna()], on_low_coverage="ignore")
    linked = ptr2.sort_values("action_row")["frame_id"]
    assert linked.iloc[:2].tolist() == [10, 30] and linked.iloc[2:].isna().all()


def test_period5_unlinked_no_cross_game():  # Review Focus 4
    actions, frames = spec_4_9_reproduction_inputs()
    frs = frames.copy()
    frs.loc[frs["game_id"] == GAME_B, "period_id"] = 5  # only game 8 has shootout frames
    acts = actions.copy()
    acts.loc[acts["game_id"] == GAME_A, "period_id"] = 5  # game 7's actions sit in period 5 with no game-7 frames
    acts.loc[acts["game_id"] == GAME_A, "time_seconds"] = [2.40, 3.20]  # on game 8's period-5 clock
    ptr, rep = link_actions_to_frames(acts, frs, on_low_coverage="ignore")
    by_row = ptr.set_index("action_row").sort_index()
    assert by_row.loc[[0, 1], "frame_id"].isna().all()
    assert (by_row.loc[[0, 1], "n_candidate_frames"] == 0).all()
    assert rep.per_game_period_link_rate[("7", 5)] == 0.0


@pytest.mark.parametrize("linker", [link_actions_to_frames, link_actions_to_frames_elastic])
def test_empty_actions_pointers_have_new_columns(linker):  # Review Focus 1
    actions, frames = make_single_game_inputs()
    ptr, rep = linker(actions.iloc[:0], frames)
    assert list(ptr.columns) == _POINTER_COLUMNS + _NEW_COLUMNS
    assert len(ptr) == 0
    assert ptr["action_row"].dtype == np.int64 and ptr["action_fingerprint"].dtype == np.uint64
    assert rep.per_game_period_link_rate == {}


def test_elastic_two_games_equal_per_game_calls():
    actions, frames = make_two_game_inputs()
    whole, rep = link_actions_to_frames_elastic(actions, frames, on_low_coverage="ignore")
    for game in (GAME_A, GAME_B):
        mask = (actions["game_id"] == game).to_numpy()
        part, _ = link_actions_to_frames_elastic(
            actions[mask].reset_index(drop=True), frames[frames["game_id"] == game], on_low_coverage="ignore"
        )
        rows = np.flatnonzero(mask)
        got = whole.set_index("action_row").loc[rows, _POINTER_COLUMNS].reset_index(drop=True)
        assert_bits_equal(got, part[_POINTER_COLUMNS])
    assert {g for g, _ in rep.per_game_period_link_rate} == {"7", "8"}
