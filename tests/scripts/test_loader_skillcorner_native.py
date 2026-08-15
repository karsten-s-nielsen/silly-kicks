"""The pining SkillCorner path must produce frames with REAL visibility and ball_z (spec 3.3).

The kloppy gateway hard-codes visibility=None and drops ball_z; the native builder carries both.
"""

import numpy as np
import pytest
from _loader_pining import _skillcorner_bronze


def test_bronze_carries_detection_and_ball_z_and_pitch_dims():
    meta = {
        "pitch_length": 104.0,
        "pitch_width": 68.0,
        "home_team": {"id": 1},
        "players": [
            {"id": 10, "team_id": 1, "player_role": {"acronym": "GK"}},
            {"id": 11, "team_id": 2, "player_role": {"acronym": "CB"}},
        ],
    }
    raw = [
        {
            "period": 1,
            "frame": 1,
            "timestamp": 0.0,
            "player_data": [
                {"player_id": 10, "x": -50.0, "y": 0.0, "is_detected": True},
                {"player_id": 11, "x": 5.0, "y": 3.0, "is_detected": False},
            ],
            "ball_data": {"x": 0.0, "y": 0.0, "z": 1.5},
        }
    ]
    bronze = _skillcorner_bronze(raw, meta, match_id="m1")

    assert set(bronze.columns) >= {
        "match_id",
        "period",
        "frame",
        "timestamp",
        "player_id",
        "team_id",
        "is_goalkeeper",
        "x",
        "y",
        "ball_x",
        "ball_y",
        "ball_z",
        "is_visible",
        "frame_rate",
        "pitch_length",
        "pitch_width",
    }
    assert bronze["is_visible"].tolist() == [True, False]  # is_detected survives
    assert bronze["ball_z"].iloc[0] == 1.5  # ball_z survives
    assert bronze["pitch_length"].iloc[0] == 104.0  # real dims reach the builder
    assert bronze.loc[bronze["player_id"] == "10", "is_goalkeeper"].iloc[0]  # roster GK


def test_bronze_parses_string_timestamp_to_continuous_seconds():
    """SkillCorner V3 `timestamp` is a broadcast-clock STRING ("MM:SS.s" / "H:MM:SS.s"), not seconds.

    It must reach the native builder as CONTINUOUS-clock seconds (the builder subtracts the nominal
    period offset itself); a raw string trips `str - float` inside `convert_to_frames`. This is the
    exact defect the float-only plan fixtures hid.
    """
    meta = {
        "pitch_length": 105.0,
        "pitch_width": 68.0,
        "home_team": {"id": 1},
        "players": [{"id": 10, "team_id": 1, "player_role": {"acronym": "GK"}}],
    }
    raw = [
        {
            "period": 2,
            "frame": 2,
            "timestamp": "47:30.0",  # continuous clock: 47 min 30 s into the match (2nd half)
            "player_data": [{"player_id": 10, "x": 0.0, "y": 0.0, "is_detected": True}],
            "ball_data": {"x": 0.0, "y": 0.0, "z": 0.0},
        }
    ]
    bronze = _skillcorner_bronze(raw, meta, match_id="m1")
    assert bronze["timestamp"].iloc[0] == pytest.approx(47 * 60 + 30.0)  # 2850.0 s, numeric


def test_game_id_is_the_REAL_match_id_not_a_path_fragment():
    """The silent killer. game_id is the StratifiedGroupKFold grouping key: if it is derived from
    a temp filename, all ten public matches collapse into ONE group called "tracking" and the
    public arm -- the arm that decides what ships -- drops from 17 groups to 8.

    KILL-LINE: replace `match_id=str(match_id)` with `match_id=str(paths["tracking"]).split("_")[-2]`
    and this MUST fail.
    """
    meta = {
        "pitch_length": 105.0,
        "pitch_width": 68.0,
        "home_team": {"id": 1},
        "players": [{"id": 10, "team_id": 1, "player_role": {"acronym": "GK"}}],
    }
    raw = [
        {
            "period": 1,
            "frame": 1,
            "timestamp": 0.0,
            "player_data": [{"player_id": 10, "x": 0.0, "y": 0.0, "is_detected": True}],
            "ball_data": {"x": 0.0, "y": 0.0, "z": 0.0},
        }
    ]
    bronze = _skillcorner_bronze(raw, meta, match_id="1886347")
    assert bronze["match_id"].unique().tolist() == ["1886347"]
    assert "tracking" not in bronze["match_id"].tolist()


@pytest.mark.e2e
def test_action_frame_colocation_on_a_non_105_pitch():
    """On a 104 m match, a same-player event and its linked tracking frame must agree.

    This exercises the native route + the pitch fix end-to-end: the co-location residual must land
    well under 2 m. It is the make-or-break gate -- a wrong pitch scale, a y-flip, or a broken
    centre-origin transform makes it large and cannot be resolved away by orientation below.

    ORIENTATION NOTE: SPADL actions are per-acting-team LTR (the SkillCorner events converter is
    POSSESSION_PERSPECTIVE); the loader's ``absolute_frame`` frames carry ONE orientation
    (``team_attacking_direction`` is null), so an AWAY-team action is a 180 deg point-reflection of
    the frame (and teams switch ends at half). A naive merge is therefore bimodal (~27 m median) --
    an orientation confound, NOT a geometry error. We resolve the attack direction per
    (period, acting team) -- an ADR-029-style GEOMETRIC orientation, a group-level decision, never
    per-row -- then measure the residual, which isolates coordinate/pitch-scale correctness (a real
    y-flip or scale bug leaves BOTH sides large and survives). Measured 1.20 m median / 3.23 m p90.

    Owner-gated: needs PINING_FOR_THE_DATA_TOKEN and network.
    """
    import sys

    sys.path.insert(0, "scripts")
    from _loader_pining import load_matches

    from silly_kicks.tracking import link_actions_to_frames

    _prov, _mid, actions, frames, _home = next(
        iter(load_matches(providers=["skillcorner"], match_ids={"skillcorner": ["1886347"]}))
    )
    assert frames["visibility"].notna().any(), "the whole point: detection must survive"
    assert frames.loc[frames["is_ball"].astype(bool), "z"].notna().any(), "ball_z must be recovered"

    # `link_actions_to_frames` returns (pointers, LinkReport); pointers carry action_id + frame_id
    # (Int64, NaN if unlinked) but NOT period_id -- pull period_id + acting team_id from the actions.
    links, _report = link_actions_to_frames(actions, frames)
    links = links.dropna(subset=["frame_id"]).copy()
    links["frame_id"] = links["frame_id"].astype("int64")
    merged = links.merge(actions[["action_id", "period_id", "team_id", "start_x", "start_y"]], on="action_id")
    ball = frames.loc[frames["is_ball"].astype(bool), ["frame_id", "period_id", "x", "y"]].copy()
    ball["frame_id"] = ball["frame_id"].astype("int64")
    j = merged.merge(ball, on=["frame_id", "period_id"], how="inner")

    d_direct = np.hypot(j["start_x"] - j["x"], j["start_y"] - j["y"])
    d_refl = np.hypot(j["start_x"] - (105.0 - j["x"]), j["start_y"] - (68.0 - j["y"]))
    j = j.assign(_d_direct=d_direct, _d_refl=d_refl)
    grp = j.groupby(["period_id", "team_id"])
    use_refl = grp["_d_refl"].transform("median") < grp["_d_direct"].transform("median")
    resid = np.where(use_refl.to_numpy(), j["_d_refl"].to_numpy(), j["_d_direct"].to_numpy())
    resid_median = float(np.median(resid))
    assert resid_median < 2.0, f"orientation-resolved action-frame co-location median {resid_median:.2f} m"


def test_unrostered_player_is_dropped_not_given_a_None_team():
    """A player_id absent from the roster (referee / tracking artifact) must be DROPPED, not
    stamped team_id="None" -- which would corrupt every team-based feature. Matches the old
    kloppy path, which filtered to the roster (PR-S115 review hardening)."""
    from _loader_pining import _skillcorner_bronze

    meta = {
        "pitch_length": 105.0,
        "pitch_width": 68.0,
        "home_team": {"id": 1},
        "players": [{"id": 10, "team_id": 1, "player_role": {"acronym": "GK"}}],
    }
    raw = [
        {
            "period": 1,
            "frame": 1,
            "timestamp": 0.0,
            "player_data": [
                {"player_id": 10, "x": 0.0, "y": 0.0, "is_detected": True},
                {"player_id": 999, "x": 5.0, "y": 5.0, "is_detected": True},  # unrostered
            ],
            "ball_data": {"x": 0.0, "y": 0.0, "z": 0.0},
        }
    ]
    bronze = _skillcorner_bronze(raw, meta, match_id="m1")
    assert bronze["player_id"].tolist() == ["10"]  # the unrostered 999 dropped
    assert "None" not in bronze["team_id"].tolist()


# --------------------------------------------------------------------------------------------
# tracking_limit must count records that CARRY WORK, not raw records.


def _rec(period, players):
    return {"period": period, "frame": 1, "timestamp": "00:00:00.0", "ball_data": {}, "player_data": players}


def test_tracking_limit_skips_the_empty_pre_match_prefix():
    """A SkillCorner feed opens with `period: null` records carrying an EMPTY `player_data`.

    Measured on the pining corpus: 19 such records on match 1886347 but **12,559** on 2011166 and
    **13,459** on 2013725. A raw head slice at the harness's `tracking_limit=50` therefore lands
    entirely inside the prefix on those two, `_skillcorner_bronze` emits ZERO rows, and
    `convert_to_frames` raises `bronze missing column(s)` listing EVERY column -- because
    `pd.DataFrame([])` has none. That reads as a corrupt download rather than an empty slice, and
    it aborted a Stage-2 run at the `for_each` consecutive-failure guard.
    """
    from scripts._loader_pining import _head_with_player_data

    raw = [_rec(None, []) for _ in range(200)] + [_rec(1, [{"player_id": 1, "x": 1.0, "y": 2.0}]) for _ in range(80)]
    out = _head_with_player_data(raw, 50)
    assert len(out) == 50, "the limit must be filled from records that carry player data"
    assert all(r["player_data"] for r in out), "an empty-player_data record survived the filter"


def test_tracking_limit_still_caps_when_there_is_no_prefix():
    """Non-vacuity: the filter must still BOUND work, not just skip prefixes. A function that
    returned everything would satisfy the test above."""
    from scripts._loader_pining import _head_with_player_data

    raw = [_rec(1, [{"player_id": 1, "x": 1.0, "y": 2.0}]) for _ in range(500)]
    assert len(_head_with_player_data(raw, 50)) == 50


def test_tracking_limit_returns_what_exists_when_the_feed_is_shorter():
    """A short feed must not raise or pad -- it yields what it has."""
    from scripts._loader_pining import _head_with_player_data

    raw = [_rec(1, [{"player_id": 1, "x": 1.0, "y": 2.0}]) for _ in range(7)]
    assert len(_head_with_player_data(raw, 50)) == 7


def test_an_empty_bronze_names_its_REAL_cause():
    """`pd.DataFrame([])` has no columns, so returning it made `convert_to_frames` report EVERY
    expected column as missing -- reading as a corrupt download rather than an empty slice. The
    prefix fix removed the most common route to this state; the diagnostic has to be right for the
    others (an off-roster-only feed, a match with no `x`)."""
    import pytest

    from scripts._loader_pining import _skillcorner_bronze

    meta = {
        "players": [{"id": 1, "team_id": "H", "player_role": {"acronym": "GK"}}],
        "pitch_length": 105.0,
        "pitch_width": 68.0,
    }
    with pytest.raises(ValueError, match="no player rows"):
        _skillcorner_bronze([_rec(None, [])], meta, match_id="X")


def test_the_empty_bronze_error_reports_the_counts_a_reader_needs():
    """A bare "no rows" does not say whether the feed was empty, the roster was, or the filter ate
    everything -- which is the distinction that cost a Stage-2 run."""
    import pytest

    from scripts._loader_pining import _skillcorner_bronze

    meta = {
        "players": [{"id": 1, "team_id": "H", "player_role": {"acronym": "GK"}}],
        "pitch_length": 105.0,
        "pitch_width": 68.0,
    }
    with pytest.raises(ValueError) as exc:
        _skillcorner_bronze([_rec(None, []), _rec(None, [])], meta, match_id="X")
    msg = str(exc.value)
    assert "2 raw record" in msg, "the raw record count must be reported"
    assert "roster=1" in msg, "the roster size must be reported"


def test_a_NON_empty_bronze_still_builds():
    """Non-vacuity: the guard must not reject a healthy feed."""
    from scripts._loader_pining import _skillcorner_bronze

    meta = {
        "players": [{"id": 1, "team_id": "H", "player_role": {"acronym": "GK"}}],
        "pitch_length": 105.0,
        "pitch_width": 68.0,
    }
    out = _skillcorner_bronze([_rec(1, [{"player_id": 1, "x": 1.0, "y": 2.0}])], meta, match_id="X")
    assert len(out) == 1 and out["team_id"].iloc[0] == "H" and bool(out["is_goalkeeper"].iloc[0])
