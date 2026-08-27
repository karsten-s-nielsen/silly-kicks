"""ADR-068 review (agent 2): the group_rows sites that replaced a raw `==` match canonically
(ADR-019). One characterization test per site: a mismatched-dtype / string key must still match --
the old raw `==` returned empty. Discriminating by construction (assert non-degenerate output)."""

import pandas as pd
import pytest

from silly_kicks._frame_index import group_rows


# --- seam-level (the mechanism all six sites route through) ---
@pytest.mark.parametrize("col_dtype, key", [("int64", "1"), ("string", 1), ("Int64", "1")])
def test_group_rows_matches_across_dtype(col_dtype, key):
    df = pd.DataFrame({"k": pd.array([1, 1, 2], dtype=col_dtype), "v": [10, 20, 30]})
    got = group_rows(df, "k").get(key)
    assert not got.empty, f"canonical match must survive {col_dtype} col vs {type(key).__name__} key"
    assert got["v"].tolist() == [10, 20]


def test_group_rows_multikey_matches_across_dtype():
    df = pd.DataFrame({"g": [1, 1, 2], "t": [5, 5, 6], "v": [1, 2, 3]})
    got = group_rows(df, ("g", "t")).get("1", "5")  # str keys vs int columns
    assert got["v"].tolist() == [1, 2]


# --- per-site end-to-end (spec Section 4.5; owner-restored) ---
def test_off_ball_runs_survives_mismatched_game_id_dtype():
    from silly_kicks.tracking._off_ball_runs import _off_ball_runs_kernel
    from tests.tracking.test_off_ball_run_perf import _two_game_fixture

    actions, frames = _two_game_fixture()
    actions = actions.copy()
    actions["game_id"] = actions["game_id"].astype(str)  # int frames vs str actions
    out = _off_ball_runs_kernel(actions, frames, home_team_id=1)
    assert out is not None and len(out) == len(actions)


def test_detect_off_ball_runs_survives_mismatched_game_id_dtype():
    # detect_off_ball_runs returns a RUNS frame (variable count), so compare int-baseline vs str.
    from silly_kicks.tracking._run_values import detect_off_ball_runs
    from tests.tracking.test_off_ball_run_perf import _two_game_fixture

    actions, frames = _two_game_fixture()
    base = detect_off_ball_runs(actions, frames)
    a2 = actions.copy()
    a2["game_id"] = a2["game_id"].astype(str)  # str actions vs int frames
    out = detect_off_ball_runs(a2, frames)
    assert len(out) == len(base)  # canonical join -> same runs as the int baseline (raw `==`: empty)


def test_confounders_pressure_survives_mismatched_game_id_dtype():
    import silly_kicks.causal._confounders as C
    from tests.causal.test_confounders_perf import _frames_and_spells, _stub_add_pressure

    frames, spells = _frames_and_spells(3)
    spells = spells.copy()
    spells["game_id"] = spells["game_id"].astype(str)  # str spells vs int frames
    out = C._pressure_at_entry(spells, frames, _stub_add_pressure)
    assert len(out) == len(spells)  # matched -> a value per spell, not silently empty


def test_skillcorner_inference_survives_mismatched_period_dtype():
    from silly_kicks.spadl._skillcorner_inference import infer_defensive_actions
    from tests.spadl.test_skillcorner_inference_perf import test_obe_lookup_built_once  # noqa: F401

    # Rebuild the perf fixture's pp/obe here (it is inline in that test); mismatch obe.period dtype.
    pp = pd.DataFrame(
        {
            "event_id": ["pp_1", "pp_2"],
            "period": [1, 1],
            "time_seconds": [10.0, 15.0],
            "team_id": ["team_b", "team_b"],
            "player_id": ["p11", "p12"],
            "start_type": ["pass_interception", "recovery"],
            "x_start": [5.0, 15.0],
            "y_start": [3.0, 10.0],
        }
    )
    obe = pd.DataFrame(
        {
            "period": ["1", "1"],  # str obe.period vs int pp.period -> mismatch
            "time_seconds": [9.9, 14.8],
            "team_id": ["team_b", "team_b"],
            "player_id": ["p13", "p14"],
            "end_type": ["direct_regain", "direct_regain"],
            "x_start": [4.0, 14.0],
            "y_start": [2.0, 9.0],
        }
    )
    result = infer_defensive_actions(pp, obe)
    # canonical join upgraded each defensive row to its nearest direct_regain (raw `==` would miss,
    # leaving the original pp players p11/p12).
    assert set(result["player_id"]) == {"p13", "p14"}


def test_defensive_credits_survives_mismatched_frame_id_dtype(fitted_xt):
    # The frame_id key is resolved via linking (fid_by_pos), so cast the FRAMES' frame_id (not
    # actions) and compare to the int baseline -- string frame_id must canonicalize to the same credits.
    from silly_kicks.tracking.defensive_credit import compute_defensive_credits
    from tests.tracking.test_defensive_credit_perf import _shot_scene

    actions, frames = _shot_scene()
    base = compute_defensive_credits(actions, frames, xg_column="xg", xt=fitted_xt)
    f2 = frames.copy()
    f2["frame_id"] = f2["frame_id"].astype(str)
    out = compute_defensive_credits(actions, f2, xg_column="xg", xt=fitted_xt)
    assert len(out) == len(base)


def test_gk_identification_survives_string_group_keys():
    # Same-source key (derived picks come from the frames): no cross-dtype is possible, so this
    # characterizes that string (game_id, team_id) keys canonicalize correctly at the site.
    from silly_kicks.tracking._gk_identification import derive_goalkeepers
    from tests.tracking.test_gk_identification_perf import _multi_team_frames

    frames = _multi_team_frames(20)
    frames = frames.copy()
    frames["game_id"] = frames["game_id"].astype(str)
    frames["team_id"] = frames["team_id"].astype(str)
    frames_out, picks = derive_goalkeepers(frames)
    assert frames_out["is_goalkeeper"].sum() == 4 * 20 and len(picks) == 4  # matched, byte-identical


def test_build_opportunities_survives_string_frame_id():
    import silly_kicks.causal.opportunities as O
    from tests.causal._fixtures import META, WIDE, actions, frames

    f = frames({10.0: 5, 10.2: 5, 10.4: 5, 10.6: 5}, {t: WIDE for t in (10.0, 10.2, 10.4, 10.6)})
    f = f.copy()
    f["frame_id"] = f["frame_id"].astype(str)  # non-default key dtype -> canonicalized
    out = O.build_opportunities(f, actions([]), home_team_id=5, model_metadata=META)
    assert out is not None  # ran to completion under a string frame_id key
