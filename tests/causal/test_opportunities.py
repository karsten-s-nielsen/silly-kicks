import numpy as np
import pandas as pd

from silly_kicks._causal import opportunities as O
from silly_kicks.spadl import config as _c
from tests.causal._fixtures import CENTRAL, META, NEAR0, WIDE, actions, frames, spell


def test_single_spell_one_row():
    f = frames({10.0: 5, 10.2: 5, 10.4: 5}, {10.0: WIDE, 10.2: WIDE, 10.4: NEAR0})
    opp = O.build_opportunities(f, actions([]), home_team_id=5, model_metadata=META)
    assert len(opp) == 1 and opp["possessing_team"].iloc[0] == 5
    assert opp["spell_duration_seconds"].iloc[0] >= 0


def test_reentry_after_turnover_is_new_spell():
    # team 6's low-x ball is NOT advanced toward team-6's attacked goal (105) -> out-of-domain;
    # so the turnover closes team 5's spell and team 5 re-entering opens a second.
    f = frames({10.0: 5, 10.2: 5, 10.4: 6, 10.6: 5, 10.8: 5}, {t: WIDE for t in (10.0, 10.2, 10.4, 10.6, 10.8)})
    opp = O.build_opportunities(f, actions([]), home_team_id=5, model_metadata=META)
    assert (opp["possessing_team"] == 5).sum() == 2


def test_reentry_after_domain_exit_is_new_spell():
    f = frames({10.0: 5, 10.2: 5, 10.4: 5, 10.6: 5}, {10.0: WIDE, 10.2: CENTRAL, 10.4: WIDE, 10.6: WIDE})
    opp = O.build_opportunities(f, actions([]), home_team_id=5, model_metadata=META)
    assert len(opp) == 2


def test_one_frame_domain_blip():  # R2-M4: single out-of-domain frame closes+reopens
    f = frames({10.0: 5, 10.2: 5, 10.4: 5}, {10.0: WIDE, 10.2: CENTRAL, 10.4: WIDE})
    opp = O.build_opportunities(f, actions([]), home_team_id=5, model_metadata=META)
    assert len(opp) == 2


def test_period_boundary_splits_spells():
    f1 = frames({10.0: 5, 10.2: 5}, {10.0: WIDE, 10.2: WIDE}, period=1)
    f2 = frames({1.0: 5, 1.2: 5}, {1.0: WIDE, 1.2: WIDE}, period=2)  # period-relative time reset
    opp = O.build_opportunities(
        pd.concat([f1, f2], ignore_index=True), actions([]), home_team_id=5, model_metadata=META
    )
    assert len(opp) == 2  # never merged across the period boundary


def test_treatment_capped_by_window_T():  # R3-M1/R2-H3: the fixed T cap, on a LONG continuous spell
    T = O.EXPOSURE_WINDOW_SECONDS
    f = spell(5, 10.0, 10.0 + T + 4.0)  # spell extends well past entry+T (and < MAX_SPELL_SECONDS)
    cross = _c.actiontype_id["cross"]
    a_in = actions([[1, 0, 1, 5, 10.0 + T - 1.0, cross, 1, 20, 8, 14, 6]])
    assert int(O.build_opportunities(f, a_in, home_team_id=5, model_metadata=META)["Z"].iloc[0]) == 1
    a_out = actions([[1, 0, 1, 5, 10.0 + T + 1.0, cross, 1, 20, 8, 14, 6]])  # past T, still within the spell
    assert int(O.build_opportunities(f, a_out, home_team_id=5, model_metadata=META)["Z"].iloc[0]) == 0


def test_treatment_capped_by_possession_end():  # R3-M1: clamp to spell_end kills cross-phase misattribution
    f = spell(5, 10.0, 10.4)  # short spell ends ~10.4, well within T
    cross = _c.actiontype_id["cross"]
    a_after = actions([[1, 0, 1, 5, 12.0, cross, 1, 20, 8, 14, 6]])  # cross AFTER possession ended
    assert int(O.build_opportunities(f, a_after, home_team_id=5, model_metadata=META)["Z"].iloc[0]) == 0
    a_in = actions([[1, 0, 1, 5, 10.3, cross, 1, 20, 8, 14, 6]])  # cross within the spell
    assert int(O.build_opportunities(f, a_in, home_team_id=5, model_metadata=META)["Z"].iloc[0]) == 1


def test_opponent_cross_is_negative():
    f = spell(5, 10.0, 10.4)
    acts = actions([[1, 0, 1, 6, 10.3, _c.actiontype_id["cross"], 1, 20, 8, 14, 6]])
    assert int(O.build_opportunities(f, acts, home_team_id=5, model_metadata=META)["Z"].iloc[0]) == 0


def test_outcome_strictly_post_cross():  # R2-M1
    f = spell(5, 10.0, 11.5)  # spell spans the cross at 11.0
    cross, shot = _c.actiontype_id["cross"], _c.actiontype_id["shot"]
    pre = actions(
        [[1, 0, 1, 5, 11.0, cross, 1, 20, 8, 14, 6], [1, 1, 1, 5, 10.5, shot, 1, 14, 6, 0, 34]]
    )  # shot precedes the cross
    o1 = O.build_opportunities(f, pre, home_team_id=5, model_metadata=META)
    assert int(o1["Z"].iloc[0]) == 1 and int(o1["Y"].iloc[0]) == 0
    post = actions(
        [[1, 0, 1, 5, 11.0, cross, 1, 20, 8, 14, 6], [1, 1, 1, 5, 11.5, shot, 1, 14, 6, 0, 34]]
    )  # shot after the cross, within W
    assert int(O.build_opportunities(f, post, home_team_id=5, model_metadata=META)["Y"].iloc[0]) == 1


def test_control_outcome_from_entry():
    f = spell(5, 10.0, 10.4)
    acts = actions([[1, 0, 1, 5, 11.0, _c.actiontype_id["shot"], 1, 14, 6, 0, 34]])  # no cross -> control
    opp = O.build_opportunities(f, acts, home_team_id=5, model_metadata=META)
    assert int(opp["Z"].iloc[0]) == 0 and int(opp["Y"].iloc[0]) == 1  # control Y from entry over W


def test_score_differential_populated():  # M1
    f = spell(5, 10.0, 10.4)
    acts = actions([[1, 0, 1, 5, 1.0, _c.actiontype_id["shot"], _c.result_id["success"], 14, 6, 0, 34]])
    opp = O.build_opportunities(f, acts, home_team_id=5, model_metadata=META)  # team 5 (home) scored at t=1
    assert not np.isnan(opp["score_differential"].iloc[0])


def test_confounder_set_is_seven_no_ball_features():  # M3 + R2-M2
    from silly_kicks.tracking._xcross_attempt import _CONFOUNDERS

    assert O.PAPER_CONFOUNDERS == list(_CONFOUNDERS)  # single source of truth, not re-literal'd
    f = spell(5, 10.0, 10.4)
    opp = O.build_opportunities(f, actions([]), home_team_id=5, model_metadata=META)
    for c in O.PAPER_CONFOUNDERS + O.GK_BLOCK:
        assert c in opp.columns
    for ball in ("ball_r", "ball_theta", "ball_speed"):
        assert ball not in O.PAPER_CONFOUNDERS


def test_carrier_handoff_midspell_stays_one_row():  # H4: genuine carrier flip, one continuous spell
    from silly_kicks.tracking._ball_carrier import infer_ball_carrier

    f = frames({10.0: 5, 10.2: 5}, {10.0: (12.0, 6.0), 10.2: (30.0, 6.0)})  # ball stays advanced+wide
    m2 = f["time_seconds"] == 10.2
    f.loc[m2 & (f["player_id"] == 10), ["x", "y"]] = [12.0, 6.0]  # p10 left behind
    f.loc[m2 & (f["player_id"] == 11), ["x", "y"]] = [30.0, 6.0]  # p11 now on the ball
    car = infer_ball_carrier(f, **META["carrier_params"])
    assert car["ball_carrier_player_id"].dropna().nunique() >= 2  # fixture genuinely flips the carrier
    opp = O.build_opportunities(f, actions([]), home_team_id=5, model_metadata=META)
    assert len(opp) == 1


def test_carrier_coverage_reported():
    f = spell(5, 10.0, 10.2)
    opp = O.build_opportunities(f, actions([]), home_team_id=5, model_metadata=META)
    assert "carrier_resolved" in opp.columns


def test_int64_id_frames_resolve_carrier_and_gk_features():
    # GradientSports frames carry nullable-Int64 player_id/team_id; ``.to_numpy()`` upcasts to
    # float64, so the old ``pid.astype(str)`` produced "366.0" and never matched the clean-int
    # carrier id "366" -> carrier_mask all-False -> every carrier-anchored + GK feature silently NaN
    # (~83% of the real corpus). The canonical-id fix (ADR-019) must resolve them. _fixtures.spell()
    # builds Int64-id frames (the GS shape), so this is the regression guard.
    f = spell(5, 10.0, 10.4)
    assert str(f["player_id"].dtype) == "Int64"  # precondition: GS-style nullable-int ids
    opp = O.build_opportunities(f, actions([]), home_team_id=5, model_metadata=META)
    must_resolve = [
        "dist_nearest_def",
        "dist_nearest_teammate",
        "dist_endline",
        "space_controlled",
        "box_off_def_ratio",
        *O.GK_BLOCK,
    ]
    for c in must_resolve:
        assert opp[c].notna().all(), f"{c} must resolve (not NaN) on Int64-id frames"
