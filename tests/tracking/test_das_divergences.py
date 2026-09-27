"""Documented divergences from the accessible-space 2.0.15 reference (spec §6.8, ADR-107).

Each test asserts BOTH sides:

* the NATIVE behaviour (fail loud, degrade to NaN, or resolve correctly), and
* that the golden oracle RECORDED the reference defect -- either a recorded value in
  ``reference_das_team.csv`` (the library silently returned a wrong number) or an entry in
  ``reference_errors.json`` (the library raised).

The native side runs on the SAME golden scene frames the reference saw, so the two are directly
comparable. Where the divergence is algorithmic (not fail-loud), the native engine is run under
the REFERENCE quadrature so the difference isolates the algorithm change, not the quadrature
change (ADR-108, covered separately in ``test_das_quadrature.py``).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from silly_kicks.tracking._das_pack import pack_frames
from silly_kicks.tracking._das_params import DAS_PARAMS
from tests.tracking._das_golden import load_golden
from tests.tracking._das_helpers import engine_arrays, golden_frames, run_engine

REF = dataclasses.replace(DAS_PARAMS, quadrature="reference")


def _team_das(scene: str) -> np.ndarray:
    g = load_golden()
    sub = g.das_team[g.das_team["scene_id"] == scene].sort_values(["game_id", "period_id", "frame_id"], kind="stable")
    return sub["DAS"].to_numpy(dtype=float)


# --- fail-loud divergences: the reference silently mishandled; native RAISES -------------------


def test_d_dup_native_raises_reference_silently_dropped():
    with pytest.raises(ValueError, match="duplicate"):
        pack_frames(golden_frames("V-DUP"), attacking_direction_col="dir")
    # D-DUP: the reference kept the first duplicate and returned a value (recorded, not an error).
    assert "V-DUP" in set(load_golden().das_team["scene_id"]), "the reference recorded a (defective) V-DUP value"


def test_d_multiball_native_raises_reference_also_raised():
    with pytest.raises(ValueError, match="more than one ball"):
        pack_frames(golden_frames("V-MULTIBALL"), attacking_direction_col="dir")
    assert "V-MULTIBALL" in load_golden().errors, "the reference recorded a V-MULTIBALL exception"


def test_d_dirval_native_raises_reference_scaled_coords():
    # D-DIRVAL: a direction value of 0.5 scaled the coordinates in the reference (a recorded value);
    # native refuses any direction that is not exactly +1/-1.
    with pytest.raises(ValueError, match=r"\+1/-1"):
        pack_frames(golden_frames("V-DIRVAL"), attacking_direction_col="dir")
    assert "V-DIRVAL" in set(load_golden().das_team["scene_id"])


def test_d_possvar_native_raises_reference_took_first_row():
    with pytest.raises(ValueError, match="varies"):
        pack_frames(golden_frames("V-POSSVAR"), attacking_direction_col="dir")
    assert "V-POSSVAR" in set(load_golden().das_team["scene_id"])


# --- degrade-to-NaN divergences: the reference returned a fictional 0.0; native NaNs ----------


def test_d_ballnan_native_nan_reference_zero():
    got = engine_arrays("V-BALLNAN", DAS_PARAMS)
    assert np.isnan(got["team_das"]).all(), "native: a NaN ball makes the frame unscoreable (NaN)"
    assert (_team_das("V-BALLNAN") == 0.0).all(), "the reference recorded a fictional 0.0"


def test_d_possabsent_native_nan_reference_zero():
    got = engine_arrays("V-POSSABSENT", DAS_PARAMS)
    assert np.isnan(got["team_das"]).all(), "native: possession team absent -> unscoreable (NaN)"
    assert (_team_das("V-POSSABSENT") == 0.0).all(), "the reference recorded AS=0 (zero attackers)"


# --- key divergence: the reference conflated frames across periods/games -----------------------


def test_d_key_native_keeps_frames_distinct():
    packed = pack_frames(golden_frames("V-KEY"), attacking_direction_col="dir")
    assert packed.n_frames == 2, "native keys on (game, period, frame): the two periods stay distinct"
    # The reference keyed on frame_id alone -> ONE conflated frame, its value copied to both rows.
    ref = _team_das("V-KEY")
    assert len(ref) == 2 and np.isclose(ref[0], ref[1]), "the reference conflated the two into one value"


# --- algorithmic divergences (isolated under the reference quadrature) --------------------------


def test_d_off_native_does_not_apply_arbitrary_offside():
    # < 2 finite defenders: the reference still applied an offside line (arbitrary); native does not.
    got = engine_arrays("V-OFF", REF)
    exp = load_golden().reference_for("V-OFF")
    assert np.isfinite(got["team_das"]).any(), "native still scores the frame"
    assert not np.allclose(got["team_das"], exp["team_das"], equal_nan=True), (
        "native must diverge from the reference's arbitrary <2-defender offside line"
    )


def test_d_passer_native_aligns_by_frame_key_not_row_order():
    # D-PASSER: the reference aligned the passer array to rows by POSITION, so a reordered input
    # misaligns the exclusion (interface.py:908). Native identifies the passer by frame key / player
    # id (pack_frames sorts players by canonical id), so its DAS is INVARIANT to input row order --
    # the divergence is robustness, not a single-scene numeric gap. Shown on the V-PASSER scene
    # (passer beyond the offside line), whose reference value the golden recorded (order-sensitive).
    frames = golden_frames("V-PASSER")
    shuffled = frames.iloc[::-1].reset_index(drop=True)  # reverse the row order the reference saw

    a_team, _ = run_engine(frames, REF, engine="numpy")
    b_team, _ = run_engine(shuffled, REF, engine="numpy")
    key = ["game_id", "period_id", "frame_id"]
    a = a_team.sort_values(key, kind="stable")["team_das"].to_numpy()
    b = b_team.sort_values(key, kind="stable")["team_das"].to_numpy()
    assert np.isfinite(a).any(), "the scene must score (non-vacuous)"
    np.testing.assert_allclose(a, b, rtol=0, atol=0, equal_nan=True)  # native: order-invariant
    assert "V-PASSER" in set(load_golden().das_team["scene_id"]), "the reference recorded its row-order value"


# --- convention divergences (basis D2 / ADR-019): NOT golden-recorded. The reference took an explicit -
# --- `dir` column and clean ids, so it has no defect to record here; these contrast the NATIVE
# --- resolution with the OLD silly-kicks path it replaced (like D-PASSER, a behaviour assertion). -----


def _ddir_frames():
    import pandas as pd

    # Home builds from the back in its OWN half; its keeper sits deep by goal x=105.
    return pd.DataFrame(
        [
            dict(
                game_id=1,
                period_id=1,
                frame_id=1,
                player_id="ball",
                team_id=None,
                is_ball=True,
                is_goalkeeper=False,
                x=80.0,
                y=34.0,
                vx=0.0,
                vy=0.0,
                team_in_possession="Home",
            ),
            dict(
                game_id=1,
                period_id=1,
                frame_id=1,
                player_id="HGK",
                team_id="Home",
                is_ball=False,
                is_goalkeeper=True,
                x=101.0,
                y=34.0,
                vx=0.0,
                vy=0.0,
                team_in_possession="Home",
            ),
            dict(
                game_id=1,
                period_id=1,
                frame_id=1,
                player_id="H1",
                team_id="Home",
                is_ball=False,
                is_goalkeeper=False,
                x=78.0,
                y=30.0,
                vx=1.0,
                vy=0.0,
                team_in_possession="Home",
            ),
            dict(
                game_id=1,
                period_id=1,
                frame_id=1,
                player_id="H2",
                team_id="Home",
                is_ball=False,
                is_goalkeeper=False,
                x=85.0,
                y=40.0,
                vx=1.0,
                vy=0.0,
                team_in_possession="Home",
            ),
            dict(
                game_id=1,
                period_id=1,
                frame_id=1,
                player_id="AGK",
                team_id="Away",
                is_ball=False,
                is_goalkeeper=True,
                x=4.0,
                y=34.0,
                vx=0.0,
                vy=0.0,
                team_in_possession="Home",
            ),
            dict(
                game_id=1,
                period_id=1,
                frame_id=1,
                player_id="A1",
                team_id="Away",
                is_ball=False,
                is_goalkeeper=False,
                x=25.0,
                y=30.0,
                vx=-1.0,
                vy=0.0,
                team_in_possession="Home",
            ),
        ]
    )


def test_d_dir_direction_from_goalmap_not_mean_x():
    # D-DIR: the OLD silly-kicks _pin inferred attacking direction from the mean-x of the possession
    # team's players; native resolves it from the GoalMap (keeper geometry, ADR-055, allow_guess=True).
    # Scene: Home's mean-x is 88 (> 52.5, so mean-x would infer attacking x=105), but its keeper at
    # x=101 means Home DEFENDS 105 and attacks x=0. The two inferences DISAGREE, and native follows
    # the GoalMap -- direction is never derived from the possession centroid.
    from silly_kicks.tracking._gk_resolve import resolve_defended_goals

    frames = _ddir_frames()
    gm = resolve_defended_goals(frames)
    attacked = gm.attacked_goal(1, 1, "Home", allow_guess=True)
    assert attacked is not None, "the keeper geometry resolves the defended end"
    goalmap_dir = 1.0 if attacked >= 52.5 else -1.0

    home = frames[(~frames["is_ball"]) & (frames["team_id"] == "Home")]
    mean_x_dir = 1.0 if float(home["x"].mean()) > 52.5 else -1.0
    assert goalmap_dir != mean_x_dir, "scene is non-discriminating -- mean-x must disagree with the GoalMap"

    packed = pack_frames(frames)  # no dir column, no goal_map -> resolves via the GoalMap
    assert packed.direction[0] == goalmap_dir, "native takes direction from the GoalMap, not the centroid"


def _dids_frames(poss):
    import pandas as pd

    # dir supplied so the id-match is isolated from the D-DIR direction path (no is_goalkeeper needed).
    return pd.DataFrame(
        [
            dict(
                game_id=1,
                period_id=1,
                frame_id=1,
                player_id="ball",
                team_id=None,
                is_ball=True,
                is_goalkeeper=False,
                x=52.5,
                y=34.0,
                vx=0.0,
                vy=0.0,
                team_in_possession=poss,
                dir=1.0,
            ),
            dict(
                game_id=1,
                period_id=1,
                frame_id=1,
                player_id=10,
                team_id=1,
                is_ball=False,
                is_goalkeeper=False,
                x=60.0,
                y=34.0,
                vx=1.0,
                vy=0.0,
                team_in_possession=poss,
                dir=1.0,
            ),
            dict(
                game_id=1,
                period_id=1,
                frame_id=1,
                player_id=20,
                team_id=2,
                is_ball=False,
                is_goalkeeper=False,
                x=40.0,
                y=34.0,
                vx=-1.0,
                vy=0.0,
                team_in_possession=poss,
                dir=1.0,
            ),
        ]
    )


def test_d_ids_team_match_through_id_compat_not_raw_eq():
    # D-IDS: the reference compared object-cast ids with a raw ``==``, so an int team_id 1 never matched
    # a string possession "1" (``1 == "1"`` is False) -> ZERO attackers. Native canonicalises both
    # through id_compat (ADR-019), so the possession team's player IS found -- and the attacking mask is
    # invariant to the id dtype.
    str_poss = _dids_frames("1")
    packed = pack_frames(str_poss, attacking_direction_col="dir")
    assert int(packed.p_attacking.sum()) == 1, "native: id_compat matches str '1' to int team 1 (one attacker)"

    players = str_poss[~str_poss["is_ball"]]
    raw = players["team_id"].astype(object) == players["team_in_possession"].astype(object)
    assert not bool(raw.any()), "a raw object-cast == finds no attacker -- the reference's zero-attacker defect"

    int_packed = pack_frames(_dids_frames(1), attacking_direction_col="dir")
    assert np.array_equal(packed.p_attacking, int_packed.p_attacking), "native is invariant to the id dtype"


# --- xC divergences: the reference raised; native per-pass NaN + one aggregated warning ---------


def _xc_frames():
    import pandas as pd

    return pd.DataFrame(
        [
            dict(
                game_id=1,
                period_id=1,
                frame_id=2,
                player_id="ball",
                team_id=None,
                is_ball=True,
                is_goalkeeper=False,
                x=50.0,
                y=34.0,
                vx=0.0,
                vy=0.0,
                team_in_possession="Home",
            ),
            dict(
                game_id=1,
                period_id=1,
                frame_id=2,
                player_id="A",
                team_id="Home",
                is_ball=False,
                is_goalkeeper=True,
                x=5.0,
                y=30.0,
                vx=1.0,
                vy=0.0,
                team_in_possession="Home",
            ),
            dict(
                game_id=1,
                period_id=1,
                frame_id=2,
                player_id="B",
                team_id="Away",
                is_ball=False,
                is_goalkeeper=True,
                x=100.0,
                y=30.0,
                vx=-1.0,
                vy=0.0,
                team_in_possession="Home",
            ),
        ]
    )


def _xc_pass(*, frame_id=2, team="Home"):
    import pandas as pd

    return pd.DataFrame(
        [
            dict(
                action_id=1,
                game_id=1,
                period_id=1,
                frame_id=frame_id,
                player_id="A",
                team_id=team,
                start_x=40.0,
                start_y=30.0,
                end_x=60.0,
                end_y=30.0,
            )
        ]
    )


def test_d_xc_frame_native_nan_reference_raised():
    from silly_kicks.tracking._das import get_xc

    with pytest.warns(UserWarning):
        out = get_xc(_xc_pass(frame_id=99), _xc_frames())  # pass frame absent from tracking
    assert out["xC"].isna().all(), "native: a missing pass frame gives per-pass NaN, not a crash"
    assert "V-XC-FRAME" in load_golden().errors, "the reference raised on a missing pass frame"


def test_d_xc_team_native_nan_reference_raised():
    from silly_kicks.tracking._das import get_xc

    with pytest.warns(UserWarning):
        out = get_xc(_xc_pass(team="Nowhere"), _xc_frames())  # pass team absent from tracking
    assert out["xC"].isna().all(), "native: a pass team absent from tracking gives per-pass NaN"
    assert "V-XC-TEAM" in load_golden().errors, "the reference raised on an absent pass team"
