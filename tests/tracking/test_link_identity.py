"""W9.2 / W9.5 pointer identity: the frozen action fingerprint, the positional binding, and the per-call cache.

Spec 2026-11-08 tracking-batch §4.9 W9.2 (R9) and W9.5. The record strings and the uint64 vector are the contract:
the vector was computed once with a direct ``hashlib.blake2b(record.encode(), digest_size=8)`` read little-endian
over the records, never with the code under test (plan Task 4 step 1).
"""

from __future__ import annotations

import gc

import numpy as np
import pandas as pd
import pytest

from scripts._bitwise import assert_bits_equal
from silly_kicks.id_compat import ids_match
from silly_kicks.tracking._link_identity import (
    LinkedFrameRows,
    action_fingerprints,
    append_columns_by_action_row,
    append_pointer_columns,
    assemble_by_game,
    fingerprint_records,
    frame_join_keys,
    iter_games,
    link_identity_scope,
    linked_frame_rows,
    pointers_by_action_row,
    pointers_with_action_columns,
    remap_links,
    require_links_bound,
    resolve_link_positions,
)
from silly_kicks.tracking.utils import link_actions_to_frames
from tests._prod_fixtures import GAME_A, GAME_B, make_single_game_inputs, make_two_game_inputs

FROZEN_FRAMES = {
    "object": pd.DataFrame(
        {
            "game_id": pd.Series([7, "7", "abc", None], dtype=object),
            "period_id": [1, 1, 1, 2],
            "action_id": pd.Series([0, "0", "x9", 5], dtype=object),
            "time_seconds": [12.04, 0.0, -0.0, np.nan],
        }
    ),
    "typed": pd.DataFrame(
        {
            "game_id": pd.array([7, pd.NA], dtype="Int64"),
            "period_id": pd.array([1, pd.NA], dtype="Int64"),
            "action_id": np.array([0.0, 3.0]),  # the real slot case (float64)
            "time_seconds": np.array([12.04, 1234.56], dtype=np.float32),
        }
    ),
    "nan_period": pd.DataFrame(
        {
            "game_id": np.array([8]),
            "period_id": np.array([np.nan]),
            "action_id": np.array([4]),
            "time_seconds": [1234.56],
        }
    ),
}
FROZEN_RECORDS = {
    "object": ["7|1|0|0x1.8147ae147ae14p+3", "7|1|0|0x0.0p+0", "abc|1|x9|-0x0.0p+0", "<NA>|2|5|<NA>"],
    "typed": ["7|1|0|0x1.8147ae0000000p+3", "<NA>|<NA>|3|0x1.34a3d80000000p+10"],
    "nan_period": ["8|<NA>|4|0x1.34a3d70a3d70ap+10"],
}
FROZEN_VECTOR = {
    "object": [16979687726410920508, 10644627275026789326, 6067231871704346529, 2089187131554254615],
    "typed": [4584638944281581430, 744339165980172688],
    "nan_period": [1524493086814904283],
}

_MISMATCH = "links must be built from exactly this actions frame, in this row order"


@pytest.mark.parametrize("name", sorted(FROZEN_FRAMES))
def test_fingerprint_records_and_vector(name):
    assert fingerprint_records(FROZEN_FRAMES[name]) == FROZEN_RECORDS[name]
    fp = action_fingerprints(FROZEN_FRAMES[name])
    assert fp.dtype == np.uint64
    assert fp.tolist() == FROZEN_VECTOR[name]


def test_fingerprint_float32_rounded_time_differs():
    a = pd.DataFrame(
        {"game_id": [7, 7], "period_id": [1, 1], "action_id": [0, 0], "time_seconds": [12.04, float(np.float32(12.04))]}
    )
    fp = action_fingerprints(a)
    assert fp[0] != fp[1]


def test_fingerprint_without_a_game_id_column_is_the_na_game():
    # the linker treats a frame without game_id as one game (key "<NA>"); the record follows the same rule
    actions, _ = make_single_game_inputs()
    no_game = actions.drop(columns="game_id")
    na_game = actions.assign(game_id=pd.array([pd.NA] * len(actions), dtype="Int64"))
    assert fingerprint_records(no_game) == fingerprint_records(na_game)
    assert all(record.startswith("<NA>|") for record in fingerprint_records(no_game))
    assert action_fingerprints(no_game).tolist() == action_fingerprints(na_game).tolist()


def test_fingerprint_empty_actions():  # Review Focus 1
    empty = FROZEN_FRAMES["object"].iloc[:0]
    assert fingerprint_records(empty) == []
    fp = action_fingerprints(empty)
    assert fp.dtype == np.uint64 and fp.shape == (0,)


def test_public_path():
    from silly_kicks.tracking.utils import FINGERPRINT_RECORD_FORMAT, action_fingerprints  # noqa: F401

    assert FINGERPRINT_RECORD_FORMAT == "{game}|{period}|{action}|{time}"


def test_resolve_link_positions_reads_action_row():
    actions, frames = make_two_game_inputs()
    ptr, _ = link_actions_to_frames(actions, frames, on_low_coverage="ignore")
    pos = resolve_link_positions(actions, ptr)
    assert pos.dtype == np.int64
    assert pos.tolist() == ptr["action_row"].tolist()


def test_resolve_link_positions_rejects_reordered_frame():
    actions, frames = make_two_game_inputs()
    ptr, _ = link_actions_to_frames(actions, frames, on_low_coverage="ignore")
    reordered = actions.iloc[::-1].reset_index(drop=True)  # same length: a range check alone passes it
    with pytest.raises(ValueError, match=_MISMATCH + "; first mismatch at links row"):
        resolve_link_positions(reordered, ptr)


def test_resolve_link_positions_rejects_out_of_range_row():
    actions, frames = make_single_game_inputs()
    ptr, _ = link_actions_to_frames(actions, frames, on_low_coverage="ignore")
    with pytest.raises(ValueError, match=_MISMATCH):
        resolve_link_positions(actions.iloc[:-1], ptr)


def test_resolve_link_positions_names_the_first_mismatched_row():
    actions, frames = make_single_game_inputs()
    ptr, _ = link_actions_to_frames(actions, frames, on_low_coverage="ignore")
    fingerprints = ptr["action_fingerprint"].to_numpy(dtype=np.uint64, copy=True)
    fingerprints[3] ^= np.uint64(1)
    bad = ptr.assign(action_fingerprint=fingerprints)
    with pytest.raises(ValueError, match=r"first mismatch at links row 3\b"):
        resolve_link_positions(actions, bad)


def _legacy(ptr: pd.DataFrame) -> pd.DataFrame:
    """A pointer frame as every release before W9 produced it: the five columns, no action_row."""
    return ptr.drop(columns=["action_row", "action_fingerprint"])


def test_legacy_links_single_game_unique_ok_multigame_raises():
    actions, frames = make_single_game_inputs()
    ptr, _ = link_actions_to_frames(actions, frames, on_low_coverage="ignore")
    legacy = _legacy(ptr)
    # single game, unique ids on both sides: joined on action_id, so the positions equal the new pointers' rows
    assert resolve_link_positions(actions, legacy).tolist() == ptr["action_row"].tolist()

    two_actions, two_frames = make_two_game_inputs()
    two_ptr, _ = link_actions_to_frames(two_actions, two_frames, on_low_coverage="ignore")
    with pytest.raises(ValueError, match="re-link"):
        resolve_link_positions(two_actions, _legacy(two_ptr))


def test_legacy_links_duplicate_ids_raise():
    actions, frames = make_single_game_inputs()
    ptr, _ = link_actions_to_frames(actions, frames, on_low_coverage="ignore")
    dup_links = pd.concat([_legacy(ptr), _legacy(ptr).iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="re-link"):
        resolve_link_positions(actions, dup_links)
    dup_actions = pd.concat([actions, actions.iloc[:1]], ignore_index=True)  # a gamestate-slot shape
    with pytest.raises(ValueError, match="re-link"):
        resolve_link_positions(dup_actions, _legacy(ptr))


def test_legacy_links_unknown_action_id_raises():
    actions, frames = make_single_game_inputs()
    ptr, _ = link_actions_to_frames(actions, frames, on_low_coverage="ignore")
    with pytest.raises(ValueError, match=_MISMATCH):
        resolve_link_positions(actions.iloc[1:].reset_index(drop=True), _legacy(ptr))


def test_pointers_by_action_row_gives_one_row_per_action():
    actions, frames = make_two_game_inputs()
    ptr, _ = link_actions_to_frames(actions, frames, on_low_coverage="ignore")
    assert not ptr["action_row"].is_monotonic_increasing  # non-vacuity: the linker order is not the action order
    table = pointers_by_action_row(actions, ptr)
    assert table.index.equals(pd.RangeIndex(len(actions)))
    assert table["action_row"].tolist() == list(range(len(actions)))
    assert table.dtypes.equals(ptr.dtypes)  # a complete pointer frame keeps every dtype
    assert_bits_equal(table.reset_index(drop=True), ptr.sort_values("action_row", kind="stable").reset_index(drop=True))


def test_pointers_by_action_row_action_without_pointer_is_na():
    actions, frames = make_single_game_inputs()
    ptr, _ = link_actions_to_frames(actions, frames, on_low_coverage="ignore")
    table = pointers_by_action_row(actions, ptr[ptr["action_row"] != 2].reset_index(drop=True))
    assert len(table) == len(actions)
    assert bool(table.loc[2].isna().all())
    assert table["frame_id"].drop(index=2).tolist() == ptr.sort_values("action_row")["frame_id"].drop(index=2).tolist()


def test_pointers_by_action_row_first_pointer_wins():
    actions, frames = make_single_game_inputs()
    ptr, _ = link_actions_to_frames(actions, frames, on_low_coverage="ignore")
    doubled = pd.concat([ptr, ptr.assign(time_offset_seconds=99.0)], ignore_index=True)
    assert_bits_equal(pointers_by_action_row(actions, doubled), pointers_by_action_row(actions, ptr))


def test_pointers_by_action_row_rejects_another_frame():
    actions, frames = make_single_game_inputs()
    ptr, _ = link_actions_to_frames(actions, frames, on_low_coverage="ignore")
    with pytest.raises(ValueError, match=_MISMATCH):
        pointers_by_action_row(actions.iloc[::-1].reset_index(drop=True), ptr)


def test_pointers_by_action_row_reads_legacy_links():
    actions, frames = make_single_game_inputs()
    ptr, _ = link_actions_to_frames(actions, frames, on_low_coverage="ignore")
    legacy = pointers_by_action_row(actions, _legacy(ptr))
    assert "action_row" not in legacy.columns
    assert_bits_equal(legacy, pointers_by_action_row(actions, ptr)[list(legacy.columns)])


def test_pointers_with_action_columns_reads_each_action_by_position():
    actions, frames = make_two_game_inputs()
    actions = actions.assign(team_id=actions["team_id"].astype("Int64"))
    ptr, _ = link_actions_to_frames(actions, frames, on_low_coverage="ignore")
    out = pointers_with_action_columns(actions, ptr, ["game_id", "period_id", "team_id"])
    assert list(out.columns) == [*ptr.columns, "game_id", "period_id", "team_id"]
    assert out.index.equals(pd.RangeIndex(len(ptr)))
    assert_bits_equal(out[list(ptr.columns)], ptr.reset_index(drop=True))  # pointer order and values are kept
    rows = ptr["action_row"].to_numpy()
    for col in ("game_id", "period_id", "team_id"):
        assert_bits_equal(out[col], actions[col].iloc[rows].reset_index(drop=True))  # the dtype is kept too
    assert out["game_id"].nunique() == 2  # non-vacuity: both games are present


def test_pointers_with_action_columns_legacy_links_gain_action_row():
    actions, frames = make_single_game_inputs()
    ptr, _ = link_actions_to_frames(actions, frames, on_low_coverage="ignore")
    out = pointers_with_action_columns(actions, _legacy(ptr), ["period_id"])
    assert list(out.columns) == [*_legacy(ptr).columns, "action_row", "period_id"]
    assert out["action_row"].tolist() == ptr["action_row"].tolist()


def test_pointers_with_action_columns_action_value_wins_a_name_clash():
    actions, frames = make_single_game_inputs()
    ptr, _ = link_actions_to_frames(actions, frames, on_low_coverage="ignore")
    out = pointers_with_action_columns(actions, ptr.assign(period_id=99), ["period_id"])
    assert out.columns.is_unique
    assert out["period_id"].tolist() == actions["period_id"].iloc[ptr["action_row"].to_numpy()].tolist()


def test_pointers_with_action_columns_rejects_another_frame():
    actions, frames = make_single_game_inputs()
    ptr, _ = link_actions_to_frames(actions, frames, on_low_coverage="ignore")
    with pytest.raises(ValueError, match=_MISMATCH):
        pointers_with_action_columns(actions.iloc[::-1].reset_index(drop=True), ptr, ["period_id"])


def test_iter_games_partitions_actions_and_frames_by_game():
    actions, frames = make_two_game_inputs()
    mixed = actions.iloc[[0, 7, 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13]].reset_index(drop=True)  # games interleaved
    parts = list(iter_games(mixed, frames.assign(game_id=frames["game_id"].astype(str))))  # ADR-019: int vs str
    assert len(parts) == 2
    assert sorted(np.concatenate([rows for rows, _, _ in parts]).tolist()) == list(range(len(mixed)))
    for rows, game_actions, game_frames in parts:
        assert_bits_equal(game_actions, mixed.iloc[rows])
        assert game_actions["game_id"].nunique() == 1
        assert set(game_frames["game_id"]) == {str(game_actions["game_id"].iat[0])}
        assert len(game_frames) == (frames["game_id"] == game_actions["game_id"].iat[0]).sum()
    assert [p[1]["game_id"].iat[0] for p in parts] == [GAME_A, GAME_B]  # first-appearance order


def test_iter_games_no_actions_yields_the_inputs_once():
    actions, frames = make_two_game_inputs()
    empty = actions.iloc[:0]
    ((rows, game_actions, game_frames),) = list(iter_games(empty, frames))
    assert len(rows) == 0 and game_actions is empty and game_frames is frames


def test_assemble_by_game_puts_the_rows_back_in_action_order():
    actions, frames = make_two_game_inputs()
    mixed = actions.iloc[[0, 7, 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13]].set_axis(np.arange(100, 114), axis=0)
    parts = [(rows, game_actions.assign(tag=1)) for rows, game_actions, _ in iter_games(mixed, frames)]
    assert len(parts) == 2
    assert_bits_equal(assemble_by_game(parts), mixed.assign(tag=1))  # the row order and the index come back
    (one,) = [(rows, game_actions) for rows, game_actions, _ in iter_games(actions.iloc[:7], frames)]
    assert assemble_by_game([one]) is one[1]  # one game: the part itself


def test_assemble_by_game_keeps_the_missing_value_of_an_object_column():
    # pandas concat turns an all-NA object column (pd.NA / None) into nan; the per-game result keeps its values
    # object Series, not a raw array: pandas 3 infers `str` for a column built from ["rdp", NA]
    a = pd.DataFrame({"src": pd.Series([pd.NA, None], index=[0, 2], dtype=object), "v": [1.0, 2.0]}, index=[0, 2])
    b = pd.DataFrame({"src": pd.Series(["rdp", pd.NA], index=[1, 3], dtype=object), "v": [3.0, 4.0]}, index=[1, 3])
    out = assemble_by_game([(np.array([0, 2]), a), (np.array([1, 3]), b)])
    assert out.index.tolist() == [0, 1, 2, 3]
    assert [repr(v) for v in out["src"]] == ["<NA>", "'rdp'", "None", "<NA>"]
    assert out["src"].dtype == object and out["v"].tolist() == [1.0, 3.0, 2.0, 4.0]


def test_iter_games_one_game_yields_the_inputs_themselves():
    actions, frames = make_single_game_inputs()
    for a, f in ((actions, frames), (actions.drop(columns="game_id"), frames)):
        ((rows, game_actions, game_frames),) = list(iter_games(a, f))
        assert rows.tolist() == list(range(len(a)))
        assert game_actions is a and game_frames is f  # no copy of the frames


def test_iter_games_drops_the_frames_of_a_game_with_no_action():
    actions, frames = make_two_game_inputs()
    only_a = actions[actions["game_id"] == GAME_A].reset_index(drop=True)
    ((rows, game_actions, game_frames),) = list(iter_games(only_a, frames))
    assert len(rows) == len(game_actions) == len(only_a) and set(game_frames["game_id"]) == {GAME_A}


def test_frame_join_keys_follow_the_linker_one_game_rule():
    actions, frames = make_single_game_inputs()
    assert frame_join_keys(actions, frames) == ["game_id", "period_id", "frame_id"]
    # either side without game_id: the call is one game, as the linker treats it
    assert frame_join_keys(actions.drop(columns="game_id"), frames) == ["period_id", "frame_id"]
    assert frame_join_keys(actions, frames.drop(columns="game_id")) == ["period_id", "frame_id"]
    # the same rule for the period: a minimal frame set without period_id joins on what both sides carry
    assert frame_join_keys(actions.drop(columns="period_id"), frames) == ["game_id", "frame_id"]
    assert frame_join_keys(actions.drop(columns=["game_id", "period_id"]), frames) == ["frame_id"]


def _frame_rows_by_filter(actions, frames, row, frame_id):
    mask = (
        ids_match(frames["game_id"], actions["game_id"].iat[row])
        & (frames["period_id"] == actions["period_id"].iat[row]).to_numpy()
        & (frames["frame_id"] == frame_id).to_numpy()
    )
    return frames[mask]


def test_linked_frame_rows_reads_the_frame_of_the_actions_own_game():
    actions, frames = make_two_game_inputs()
    # non-vacuity: one (period_id, frame_id) names a frame in BOTH games
    assert frames.groupby(["period_id", "frame_id"])["game_id"].nunique().max() == 2
    ptr, _ = link_actions_to_frames(actions, frames, on_low_coverage="ignore")
    frame_ids = pointers_by_action_row(actions, ptr)["frame_id"]
    lookup = LinkedFrameRows(actions, frames)
    str_lookup = LinkedFrameRows(actions.assign(game_id=actions["game_id"].astype(str)), frames)  # ADR-019
    linked = 0
    for row in range(len(actions)):
        got = lookup.get(row, frame_ids.iat[row])
        if pd.isna(frame_ids.iat[row]):
            assert got.empty and list(got.columns) == list(frames.columns)
            continue
        want = _frame_rows_by_filter(actions, frames, row, frame_ids.iat[row])
        assert not want.empty
        assert_bits_equal(got, want)
        assert_bits_equal(str_lookup.get(row, float(frame_ids.to_numpy(dtype="float64", na_value=np.nan)[row])), want)
        linked += 1
    assert linked > 0 and linked < len(actions)  # non-vacuity: linked and unlinked rows


def test_linked_frame_rows_gives_each_action_the_frame_of_its_own_game():
    actions, frames = make_two_game_inputs()
    ptr, _ = link_actions_to_frames(actions, frames, on_low_coverage="ignore")
    per_row = linked_frame_rows(actions, frames, ptr)
    assert len(per_row) == len(actions)
    frame_ids = pointers_by_action_row(actions, ptr)["frame_id"]
    n_linked = 0
    for row, fr in enumerate(per_row):
        if pd.isna(frame_ids.iat[row]):
            assert fr.empty and list(fr.columns) == list(frames.columns)
            continue
        n_linked += 1
        assert set(fr["game_id"]) == {actions["game_id"].iat[row]}  # never the other game's frame
        assert set(fr["frame_id"]) == {frame_ids.iat[row]}
    assert 0 < n_linked < len(actions)  # non-vacuity


def test_linked_frame_rows_at_reads_an_explicit_key():
    # a gamestate slot row is not a row of the frame the lookup was built on: it gives its own key
    actions, frames = make_two_game_inputs()
    ptr, _ = link_actions_to_frames(actions, frames, on_low_coverage="ignore")
    frame_ids = pointers_by_action_row(actions, ptr)["frame_id"]
    lookup = LinkedFrameRows(actions, frames)
    keys = set()
    for row in (int(r) for r in np.flatnonzero(frame_ids.notna().to_numpy())):
        game, period = actions["game_id"].iat[row], actions["period_id"].iat[row]
        fid = int(frame_ids.to_numpy(dtype="float64", na_value=np.nan)[row])
        assert_bits_equal(lookup.at(game, period, fid), lookup.get(row, fid))
        assert_bits_equal(lookup.at(str(game), period, float(fid)), lookup.get(row, fid))  # ADR-019
        assert lookup.key(game, period, fid) == lookup.key(str(game), period, fid)
        assert lookup.key(game, period, fid)[:-1] == lookup.group_key(game, period)
        keys.add(lookup.key(game, period, fid))
    assert len({k[0] for k in keys}) == 2  # non-vacuity: the key separates the two games
    assert lookup.at(GAME_A, 1, None).empty and lookup.at(GAME_A, 1, 10**9).empty
    one_game = LinkedFrameRows(actions.drop(columns="game_id"), frames)
    assert one_game.key(GAME_A, 1, 10) == one_game.key(GAME_B, 1, 10)  # no game key: the game is not in the key
    assert one_game.group_key(GAME_A, 1) == one_game.group_key(GAME_B, 1.0) == ("1",)


def test_linked_frame_rows_without_a_game_key_is_one_game():
    actions, frames = make_single_game_inputs()
    ptr, _ = link_actions_to_frames(actions, frames, on_low_coverage="ignore")
    frame_ids = pointers_by_action_row(actions, ptr)["frame_id"]
    row = int(np.flatnonzero(frame_ids.notna().to_numpy())[0])
    want = _frame_rows_by_filter(actions, frames, row, frame_ids.iat[row])
    for a, f in ((actions.drop(columns="game_id"), frames), (actions, frames.drop(columns="game_id"))):
        got = LinkedFrameRows(a, f).get(row, frame_ids.iat[row])
        assert_bits_equal(got, want[list(f.columns)])


def test_linked_frame_rows_without_a_period_on_one_side_reads_by_frame():
    # a caller with hand-built links and minimal actions (no period_id): the frame is found on what both sides carry
    actions, frames = make_single_game_inputs()
    ptr, _ = link_actions_to_frames(actions, frames, on_low_coverage="ignore")
    frame_ids = pointers_by_action_row(actions, ptr)["frame_id"]
    row = int(np.flatnonzero(frame_ids.notna().to_numpy())[0])
    want = frames[(frames["frame_id"] == frame_ids.iat[row]).to_numpy()]
    lookup = LinkedFrameRows(actions.drop(columns="period_id"), frames)
    assert_bits_equal(lookup.get(row, frame_ids.iat[row]), want)
    assert lookup.key(GAME_A, 1, 10) == lookup.key(GAME_A, 2, 10)  # the period is not part of the key
    assert lookup.group_key(GAME_A, 1) == lookup.group_key(GAME_A, 2)


def test_linked_frame_rows_na_game_reads_only_na_game_frames():  # the linker's rule (Review Focus 2)
    frames = pd.DataFrame(
        {
            "game_id": pd.array([7, pd.NA, pd.NA], dtype="Int64"),
            "period_id": [1, 1, 1],
            "frame_id": [10, 10, 11],
            "x": [1.0, 2.0, 3.0],
        }
    )
    actions = pd.DataFrame({"game_id": pd.array([pd.NA, 7], dtype="Int64"), "period_id": [1, 1]})
    lookup = LinkedFrameRows(actions, frames)
    assert lookup.get(0, 10)["x"].tolist() == [2.0]
    assert lookup.get(1, 10)["x"].tolist() == [1.0]
    assert lookup.get(1, 11).empty


_PROVENANCE = ["frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]


def test_append_pointer_columns_reads_each_action_by_position():
    actions, frames = make_two_game_inputs()
    ptr, _ = link_actions_to_frames(actions, frames, on_low_coverage="ignore")
    out = append_pointer_columns(actions.copy(), actions, ptr, _PROVENANCE)
    assert list(out.columns) == [*actions.columns, *_PROVENANCE]
    assert out.index.equals(actions.index)
    assert_bits_equal(
        out[_PROVENANCE].reset_index(drop=True),
        ptr.sort_values("action_row", kind="stable")[_PROVENANCE].reset_index(drop=True),
    )


def test_append_pointer_columns_equals_the_action_id_merge_on_one_game():
    # the join every add_* used before W9: on one game with unique ids, the positional join is bit-identical
    actions, frames = make_single_game_inputs()
    ptr, _ = link_actions_to_frames(actions, frames, on_low_coverage="ignore")
    old = actions.copy().merge(
        ptr.set_index("action_id")[_PROVENANCE], left_on="action_id", right_index=True, how="left"
    )
    assert_bits_equal(append_pointer_columns(actions.copy(), actions, ptr, _PROVENANCE), old)
    assert old["frame_id"].isna().any() and old["frame_id"].notna().any()  # non-vacuity: linked and unlinked rows


def test_append_pointer_columns_keeps_the_merge_suffixes_on_a_name_clash():
    # an add_* without the provenance skip guard merged twice: the positional join keeps that exact result
    actions, frames = make_single_game_inputs()
    ptr, _ = link_actions_to_frames(actions, frames, on_low_coverage="ignore")
    carried = actions.assign(frame_id=1)
    old = carried.merge(ptr.set_index("action_id")[_PROVENANCE], left_on="action_id", right_index=True, how="left")
    assert {"frame_id_x", "frame_id_y"} <= set(old.columns)  # non-vacuity: the clash happened
    assert_bits_equal(append_pointer_columns(carried, actions, ptr, _PROVENANCE), old)


def test_append_pointer_columns_does_not_write_into_its_input():
    actions, frames = make_single_game_inputs()
    ptr, _ = link_actions_to_frames(actions, frames, on_low_coverage="ignore")
    before = actions.copy()
    append_pointer_columns(actions, actions, ptr, _PROVENANCE)
    assert_bits_equal(actions, before)


def test_append_columns_by_action_row_equals_the_action_id_merge_on_one_game():
    # a per-action result keyed on action_row: the positional join is the old action_id merge on one game
    actions, _ = make_single_game_inputs()
    value = pd.Series([1.5, 2.5], index=pd.Index([4, 1]), name="value")  # two actions have a result
    old = actions.merge(
        value.set_axis(actions["action_id"].to_numpy()[[4, 1]]), left_on="action_id", right_index=True, how="left"
    )
    assert_bits_equal(append_columns_by_action_row(actions, value.to_frame()), old)
    assert old["value"].notna().sum() == 2 and old["value"].isna().any()  # non-vacuity
    # a column that `out` already has keeps the merge suffixes
    clash = append_columns_by_action_row(actions.assign(value=0.0), value.to_frame())
    assert {"value_x", "value_y"} <= set(clash.columns)


def test_append_pointer_columns_keeps_a_repeated_index():
    actions, frames = make_single_game_inputs()
    ptr, _ = link_actions_to_frames(actions, frames, on_low_coverage="ignore")
    repeated = actions.set_axis(np.zeros(len(actions), dtype=np.int64), axis=0)
    out = append_pointer_columns(repeated.copy(), repeated, ptr, _PROVENANCE)
    assert out.index.equals(repeated.index)
    assert_bits_equal(
        out[_PROVENANCE].reset_index(drop=True),
        append_pointer_columns(actions.copy(), actions, ptr, _PROVENANCE)[_PROVENANCE].reset_index(drop=True),
    )


def test_remap_links_follows_a_permutation_and_a_subset():
    actions, frames = make_two_game_inputs()
    ptr, _ = link_actions_to_frames(actions, frames, on_low_coverage="ignore")
    perm = np.arange(len(actions))[::-1][::2]  # reverse, then keep every other row
    new_actions = actions.iloc[perm].reset_index(drop=True)
    remapped = remap_links(ptr, perm, new_actions)
    assert remapped is not None
    pos = resolve_link_positions(new_actions, remapped)  # verifies every fingerprint on the new frame
    kept = np.isin(ptr["action_row"].to_numpy(), perm)
    assert len(remapped) == int(kept.sum())
    assert perm[pos].tolist() == ptr.loc[kept, "action_row"].tolist()  # each pointer still names its own action
    assert remapped["frame_id"].tolist() == ptr.loc[kept, "frame_id"].tolist()


def test_remap_links_verifies_the_source_frame_when_it_is_given():
    # links built on a LARGER frame: its first rows equal the caller's frame, so the remapped pointers verify,
    # and the pointers of the missing rows are dropped like the pointers of a filtered action. `actions=` closes it.
    actions, frames = make_two_game_inputs()
    ptr, _ = link_actions_to_frames(actions, frames, on_low_coverage="ignore")
    prefix = actions.iloc[:7].reset_index(drop=True)
    order = np.arange(len(prefix))[::-1].copy()
    resorted = prefix.iloc[order].reset_index(drop=True)
    hole = remap_links(ptr, order, resorted)
    assert hole is not None and len(hole) == len(prefix)  # the hole, without the source frame
    with pytest.raises(ValueError, match=_MISMATCH):
        remap_links(ptr, order, resorted, actions=prefix)
    own, _ = link_actions_to_frames(prefix, frames, on_low_coverage="ignore")
    assert_bits_equal(remap_links(own, order, resorted, actions=prefix), remap_links(own, order, resorted))
    assert remap_links(None, order, resorted, actions=prefix) is None


def test_remap_links_passes_none_through():
    # a re-sort site calls it for every caller: links=None (the member links for itself) stays None
    actions, _ = make_single_game_inputs()
    assert remap_links(None, np.arange(len(actions)), actions) is None


def test_remap_links_keeps_legacy_links():
    actions, frames = make_single_game_inputs()
    legacy = _legacy(link_actions_to_frames(actions, frames, on_low_coverage="ignore")[0])
    perm = np.arange(len(actions))[::-1]
    assert remap_links(legacy, perm, actions.iloc[perm].reset_index(drop=True)) is legacy


def test_require_links_bound_verifies_caller_links():
    actions, frames = make_single_game_inputs()
    ptr, _ = link_actions_to_frames(actions, frames, on_low_coverage="ignore")
    assert require_links_bound(actions, None) is None  # no caller links: nothing to bind
    assert require_links_bound(actions, ptr) is None
    with pytest.raises(ValueError, match=_MISMATCH):
        require_links_bound(actions.iloc[::-1].reset_index(drop=True), ptr)
    with pytest.raises(ValueError, match=_MISMATCH):  # links built on a LARGER frame: a prefix is not the frame
        require_links_bound(actions.iloc[:-2], ptr)


def test_require_links_bound_adds_no_frame_to_a_warning():
    """An acceptor calls it as a statement, so a ``stacklevel=2`` warning in the acceptor still names the
    caller (the reason the check is not a wrapper decorator)."""
    import warnings

    actions, frames = make_single_game_inputs()
    ptr, _ = link_actions_to_frames(actions, frames, on_low_coverage="ignore")

    def acceptor(actions, frames, *, links=None):
        require_links_bound(actions, links)
        warnings.warn("probe", UserWarning, stacklevel=2)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        acceptor(actions, frames, links=ptr)
    assert caught and caught[0].filename == __file__


def test_cache_never_serves_freed_frame():
    # Each frame is created, used, and dropped inside ONE scope. CPython reuses an object id after it is freed, so an
    # id()-keyed cache would serve the previous frame's fingerprints to the next one (spec W9.2 cost note).
    with link_identity_scope():
        for i in range(40):
            acts = pd.DataFrame(
                {"game_id": 7, "period_id": 1, "action_id": np.arange(5), "time_seconds": np.arange(5) + 0.5 * i}
            )
            links = pd.DataFrame(
                {"action_row": np.arange(5, dtype=np.int64), "action_fingerprint": action_fingerprints(acts)}
            )
            assert resolve_link_positions(acts, links).tolist() == list(range(5))
            if i:  # the previous frame's links never verify against this frame
                with pytest.raises(ValueError, match=_MISMATCH):
                    resolve_link_positions(acts, prev_links)  # noqa: F821
            prev_links = links  # noqa: F841
            del acts
            gc.collect()


def test_scope_is_reentrant_and_cleared_on_exit():
    from silly_kicks.tracking import _link_identity as li

    acts = FROZEN_FRAMES["object"]
    with link_identity_scope():
        with link_identity_scope():
            li._cached_fingerprints(acts)
        assert li._cache_size() == 1  # the inner exit keeps the outer call's cache
    assert li._cache_size() == 0


def test_add_elastic_sync_keeps_the_actions_index_on_two_games():
    # the alignment merge reset each game's index to 0..n-1: the assembled two-game result then had duplicate
    # labels, and the producer's index-aligned column assignment raised "cannot reindex ... duplicate labels"
    from silly_kicks.tracking.features import add_elastic_sync

    actions, frames = make_two_game_inputs()
    out = add_elastic_sync(actions, frames)
    assert out.index.equals(actions.index)
    shifted = actions.set_axis(actions.index + 100, axis=0)
    assert add_elastic_sync(shifted, frames).index.equals(shifted.index)
