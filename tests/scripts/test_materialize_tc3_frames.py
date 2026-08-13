"""The materialized corpus must contain everything the trainer needs, not just the frames.

Two halves. `assert_frames_parity` protects the WRITE path: if the parquet round-trip drifts a
schema, dtype or value, the trainer fits on different data than the established pipeline produces.
(It cannot protect the PARSE -- this driver and `_loader_pining_to_cache.py` share one
`load_matches`, so the design note claiming "the established input comes from a different pipeline"
was wrong.)

`collect_home_team_map` protects the INPUT SET: `train_ghost_gk.py` needs a home-team mapping and
per-game actions, neither of which a `for_each` generation carries -- flat shards have no `meta.json`
sibling. Missing them, the trainer either exits 1 or silently fits on a shorter corpus.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.materialize_tc3_frames import assert_frames_parity, collect_home_team_map


def _frame(**over) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "game_id": [1, 1],
            "period_id": [1, 1],
            "frame_id": [1, 2],
            "player_id": ["a", "b"],
            "x": [1.0, 2.0],
            "y": [3.0, 4.0],
            "team_id": ["H", "A"],
            "vx": [0.5, 0.6],
        }
    )
    for k, v in over.items():
        base[k] = v
    return base


def test_identical_frames_pass():
    assert_frames_parity(_frame(), _frame(), match_id="m1")


def test_missing_column_is_rejected():
    with pytest.raises(AssertionError, match="column"):
        assert_frames_parity(_frame().drop(columns=["team_id"]), _frame(), match_id="m1")


def test_row_count_mismatch_is_rejected():
    with pytest.raises(AssertionError, match="row count"):
        assert_frames_parity(_frame().iloc[:1], _frame(), match_id="m1")


def test_dtype_drift_is_rejected():
    with pytest.raises(AssertionError, match="dtype"):
        assert_frames_parity(_frame(x=[1, 2]), _frame(), match_id="m1")


def test_value_drift_is_rejected():
    """Compare CONTENT, not just shape -- a schema-equal frame with different coordinates is
    exactly the silent-skew case."""
    with pytest.raises(AssertionError, match="checksum"):
        assert_frames_parity(_frame(x=[1.5, 2.0]), _frame(), match_id="m1")


def test_NON_KEY_value_drift_is_also_rejected():
    """Measured defect in the first draft: hashing only the identity columns let `vx` drift from
    0.5 to 99.0 undetected. Ghost's extractor consumes velocity and so does `infer_ball_carrier`,
    so a positions-right / velocities-wrong parse is precisely the silent skew this gate names."""
    with pytest.raises(AssertionError, match="checksum"):
        assert_frames_parity(_frame(vx=[99.0, 0.6]), _frame(), match_id="m1")


def test_negative_zero_does_not_trip_a_spurious_failure():
    """`-0.0` and `0.0` hash differently unless normalised. Negative zero is reachable via the
    velocity NEGATION (`-vx` where `vx == 0.0`), and the corpus driver treats a parity failure as
    STOP -- so this would cost a corpus pass to diagnose."""
    assert_frames_parity(_frame(x=[-0.0, 2.0]), _frame(x=[0.0, 2.0]), match_id="m1")


def _sidecar(home_dir, stem: str, *, home: str, game_ids: list[str]) -> None:
    home_dir.mkdir(parents=True, exist_ok=True)
    (home_dir / f"{stem}.json").write_text(json.dumps({"home_team_id": home, "game_ids": game_ids}))


def test_home_map_is_keyed_by_game_id_not_match_id(tmp_path):
    """SkillCorner's `game_id` is a kloppy hash unrelated to its match id, and the trainer looks up
    `home_team_map.get(str(game_id))`. Keying by the match id would miss every SkillCorner game
    while looking populated."""
    home = tmp_path / "_home"
    _sidecar(home, "skillcorner__1886347", home="55", game_ids=["a1b2c3hash"])
    assert collect_home_team_map(home, [("skillcorner", "1886347")]) == {"a1b2c3hash": "55"}


def test_a_match_contributing_several_game_ids_maps_all_of_them(tmp_path):
    home = tmp_path / "_home"
    _sidecar(home, "gradientsports__10502", home="7", game_ids=["g1", "g2"])
    assert collect_home_team_map(home, [("gradientsports", "10502")]) == {"g1": "7", "g2": "7"}


def test_a_missing_sidecar_RAISES_rather_than_yielding_a_short_map(tmp_path):
    """The failure this prevents is quiet: `for_each` resumes by skipping items whose SHARD exists,
    so a generation written before the side artifacts has shards and no sidecars. Returning the
    partial map would let the trainer print `SKIP game <id>` per game and fit on a SHORTER corpus
    while reporting success."""
    home = tmp_path / "_home"
    _sidecar(home, "skillcorner__1", home="55", game_ids=["g1"])
    with pytest.raises(SystemExit, match="no home-team sidecar"):
        collect_home_team_map(home, [("skillcorner", "1"), ("skillcorner", "2")])


def test_the_missing_sidecar_guard_is_not_vacuous(tmp_path):
    """Companion to the test above: the same two keys pass once BOTH sidecars exist, so the raise
    is caused by absence and not by the shape of the call."""
    home = tmp_path / "_home"
    _sidecar(home, "skillcorner__1", home="55", game_ids=["g1"])
    _sidecar(home, "skillcorner__2", home="66", game_ids=["g2"])
    assert collect_home_team_map(home, [("skillcorner", "1"), ("skillcorner", "2")]) == {"g1": "55", "g2": "66"}


def test_duplicate_identity_rows_are_order_insensitive():
    """Two rows tying on every identity column but differing on `vx` must hash the same regardless
    of order. They did not, until the sort key was widened to ALL columns -- and GS duplicate frames
    make this reachable, at the same spurious-STOP cost as negative zero."""
    dup = _frame()
    dup["frame_id"] = [1, 1]
    dup["player_id"] = ["a", "a"]
    dup["x"] = [1.0, 1.0]
    dup["y"] = [3.0, 3.0]
    dup["team_id"] = ["H", "H"]
    dup["vx"] = [0.5, 9.9]
    assert_frames_parity(dup, dup.iloc[::-1].reset_index(drop=True), match_id="m1")


def test_the_missing_sidecar_message_reports_the_real_corpus_size(tmp_path):
    """Regression: `keys` was consumed by the loop and then re-listed for the message, so a
    GENERATOR reported "N of 0 shards" -- a diagnostic that misstates the corpus size at exactly the
    moment someone is working out what went missing."""
    home = tmp_path / "_home"
    _sidecar(home, "skillcorner__1", home="55", game_ids=["g1"])
    keys = iter([("skillcorner", "1"), ("skillcorner", "2"), ("skillcorner", "3")])
    with pytest.raises(SystemExit, match="2 of 3 shards"):
        collect_home_team_map(home, keys)
