"""TF-60 Task 7: Layer-3 deterrent arms (outfield + keeper)."""

import numpy as np
import pandas as pd
import pytest

from silly_kicks.restdefense import compute_rest_defense, summarize_rest_defense
from silly_kicks.restdefense._arms import merge_rest_defense, rest_defense_outfield_deterrent
from silly_kicks.restdefense._columns import (
    RD_ARM_SOURCE_VALUES,
    RD_OUTFIELD_DETER_SPACE,
    RD_OUTFIELD_DETER_THREAT,
    RD_OUTFIELD_SOURCE,
    RD_SAMPLE_KEYS,
)
from tests.restdefense._fixtures import make_fitted_xt, make_rest_defense_fixture
from tests.tracking.test_ghost_outfield_model import _fit_toy


def _outfield(actions, frames, **kw):
    model, _ = _fit_toy()
    return rest_defense_outfield_deterrent(
        actions, frames, xt=make_fitted_xt(), ghost_outfield_model=model, home_team_id=1, **kw
    )


def test_arm_table_shape_source_and_keys():
    actions, frames = make_rest_defense_fixture()
    arm, _rep = _outfield(actions, frames)
    assert list(arm.columns) == [*RD_SAMPLE_KEYS, RD_OUTFIELD_DETER_THREAT, RD_OUTFIELD_DETER_SPACE, RD_OUTFIELD_SOURCE]
    assert set(arm[RD_OUTFIELD_SOURCE]) <= set(RD_ARM_SOURCE_VALUES)
    assert arm[RD_OUTFIELD_DETER_THREAT].dtype == np.float64
    # arm keys are a subset of the compute_rest_defense sample keys (declared subset, spec 14).
    from silly_kicks.restdefense import compute_rest_defense

    samples, _ = compute_rest_defense(actions, frames, xt=make_fitted_xt())
    arm_keys = set(map(tuple, arm[RD_SAMPLE_KEYS].astype(str).to_numpy()))
    sample_keys = set(map(tuple, samples[RD_SAMPLE_KEYS].astype(str).to_numpy()))
    assert arm_keys <= sample_keys


@pytest.mark.filterwarnings("ignore")
def test_outfield_arm_honest_nan_on_unresolvable_goal_not_crash():
    """ADR-055 / ADR-089 FINDING-5: a keeperless frame set makes the attacked goal unresolvable, so
    ``compute_threat_pc`` REFUSES (``GoalEndUnresolvedError``). The arm must CATCH that and degrade the
    threat leg to honest-NaN -- never crash, exactly as ``compute_rest_defense`` and the gkdv/keeper
    siblings do. Before the fix ``delta_threat_suppression_batch`` propagated the raise and the whole
    arm crashed. Reproduced on the SB360 ``gk_absent`` roster -- the exact scene that raised. Real
    full-tracking data always resolves the goal (keepers present), so this never fires there."""
    from silly_kicks.tracking import GhostOutfieldModel
    from tests.sb360._fixture import HOME_TEAM_ID, build_leg_b

    a, frames, links = build_leg_b(roster="gk_absent")
    # The assertion IS that this returns rather than raising GoalEndUnresolvedError.
    arm, _rep = rest_defense_outfield_deterrent(
        a,
        frames,
        links=links,
        xt=make_fitted_xt(),
        ghost_outfield_model=GhostOutfieldModel.from_variant("default"),
        home_team_id=HOME_TEAM_ID,
    )
    assert len(arm) > 0, "gk_absent must still produce (all-NaN) arm rows, else the guard is vacuous"
    # Every unresolvable-goal frame NaNs the threat -- never a fabricated deterrent value.
    assert arm[RD_OUTFIELD_DETER_THREAT].isna().all()


def test_outfield_ghost_never_repositions_As_keeper():
    """The HONEST isolation property (exact + stable): the outfield counterfactual the arm builds
    repositions only A's rearguard and NEVER moves A's keeper.

    NOTE (FINDING-2, see ADR-089): the outfield arm is NOT exactly keeper-VALUE-invariant. The old
    `test_outfield_arm_is_keeper_agnostic` asserted moving the keeper leaves the arm byte-identical;
    that only held under the since-fixed B=A opponent bug. delta_threat_suppression carries lambda_gk
    (A's keeper as a TTI control agent) and pitch control is nonlinear, so a fixed keeper interacts
    differently with the DIFFERING rearguard across legs (measured: threat ~3.6% on the toy fixture).
    The real, exact isolation is STRUCTURAL: the ghost does not touch the keeper.
    """
    from silly_kicks.restdefense._arms import _sample_carrier
    from silly_kicks.restdefense._counterfactual import build_restdefense_ghost_frames
    from silly_kicks.restdefense._windows import select_rest_defense_samples
    from silly_kicks.tracking import resolve_defended_goals

    actions, frames = make_rest_defense_fixture()
    model, _ = _fit_toy()
    gm = resolve_defended_goals(frames)
    scored = select_rest_defense_samples(actions, frames, goal_map=gm)
    scored = scored[scored["gate_drop_reason"].isna()]
    cf, _prov, _rep = build_restdefense_ghost_frames(
        frames, which="rearguard", model=model, home_team_id=1, carrier=_sample_carrier(scored)
    )
    key = ["game_id", "period_id", "frame_id", "player_id"]
    keeper_key = frames[frames["is_goalkeeper"].astype(bool)][key]
    kk = cf.merge(keeper_key, on=key).merge(frames, on=key, suffixes=("_cf", ""))
    assert len(kk) > 0
    assert (kk["x_cf"].to_numpy(float) == kk["x"].to_numpy(float)).all()
    assert (kk["y_cf"].to_numpy(float) == kk["y"].to_numpy(float)).all()


def test_outfield_space_arm_is_near_keeper_invariant_but_threat_is_not():
    """FINDING-2 honesty check: the DAS/space arm (keeper-blind-generic) is ~invariant to a keeper
    move, while the threat arm (lambda_gk) is materially sensitive -- the two behave differently, which
    is exactly why the exact-equality claim was wrong. Loose, directional (not an exact pin)."""
    actions, frames = make_rest_defense_fixture()
    base, _ = _outfield(actions, frames)
    moved = frames.copy()
    gk = moved["is_goalkeeper"].astype(bool)
    moved.loc[gk, "x"] = (moved.loc[gk, "x"] + 4.0).clip(1.0, 104.0)
    moved.loc[gk, "y"] = (moved.loc[gk, "y"] - 3.0).clip(1.0, 67.0)
    out, _ = _outfield(actions, moved)
    m = base.merge(out, on=RD_SAMPLE_KEYS, suffixes=("_b", "_m"))
    threat_shift = np.nanmax(
        np.abs(m[f"{RD_OUTFIELD_DETER_THREAT}_b"].to_numpy(float) - m[f"{RD_OUTFIELD_DETER_THREAT}_m"].to_numpy(float))
    )
    space_shift = np.nanmax(
        np.abs(m[f"{RD_OUTFIELD_DETER_SPACE}_b"].to_numpy(float) - m[f"{RD_OUTFIELD_DETER_SPACE}_m"].to_numpy(float))
    )
    assert space_shift < 0.05  # DAS is keeper-blind-generic -> near-invariant
    assert threat_shift > space_shift  # lambda_gk makes the threat arm materially keeper-sensitive


def test_ghost_frames_measurably_differ_from_the_factual_twin():
    # The counterfactual must ACTUALLY move A's rearguard -- else the arm is vacuously 0 for any
    # metric ("every counterfactual needs a non-vacuity assertion", CLAUDE.md). The arm's THREAT/SPACE
    # values can legitimately be ~0 for a toy model + coarse xT grid (a metric property, not a bug),
    # so non-vacuity is asserted on the FRAMES, not the metric.
    from silly_kicks.restdefense._arms import _sample_carrier
    from silly_kicks.restdefense._counterfactual import build_restdefense_ghost_frames
    from silly_kicks.restdefense._windows import select_rest_defense_samples
    from silly_kicks.tracking import resolve_defended_goals

    actions, frames = make_rest_defense_fixture()
    model, _ = _fit_toy()
    gm = resolve_defended_goals(frames)
    samples = select_rest_defense_samples(actions, frames, goal_map=gm)
    scored = samples[samples["gate_drop_reason"].isna()]
    cf, prov, rep = build_restdefense_ghost_frames(
        frames, which="rearguard", model=model, home_team_id=1, carrier=_sample_carrier(scored)
    )
    assert rep.n_frames_scored >= 1
    key = ["game_id", "period_id", "frame_id", "player_id"]
    sc = prov[prov["drop_reason"].isna()][key]
    merged = cf.merge(frames, on=key, suffixes=("_cf", "")).merge(sc, on=key)
    dx = (merged["x_cf"] - merged["x"]).abs()
    dy = (merged["y_cf"] - merged["y"]).abs()
    assert bool((dx > 1e-6).any() or (dy > 1e-6).any())


def test_opponent_by_frame_selects_B_not_A():
    """_opponent_by_frame must return the OPPONENT B, never the in-possession team A.

    Regression (found at T10): a Series-vs-Series ids_match misuse (ids_match is Series-vs-SCALAR)
    yielded an all-False mask, so `~mask` was all-True and B silently resolved to the FIRST team per
    frame (== A). The arm then priced A's OWN accessible space, not the counter-attacker's.
    """
    from silly_kicks.restdefense._arms import _opponent_by_frame

    _actions, frames = make_rest_defense_fixture()
    scored_prov = pd.DataFrame(
        {"game_id": [1], "period_id": [1], "frame_id": [100], "team_id": pd.array([1], dtype="Int64")}
    )
    b = _opponent_by_frame(frames, scored_prov)
    assert int(b.loc[(1, 1, 100)]) == 2  # A is team 1 -> opponent B is team 2


def test_outfield_threat_arm_is_mirror_invariant():
    """The both-axes write-back guard (the T7 mirror test): a scene and its 180-degree point reflection
    score the SAME arm. Fixture frames 100 (a0, home attacks right) and 102 (a2, away attacks left) are
    exact point-reflections (x->105-x, y->68-y), so the outfield THREAT arm for a0 and a2 must match to
    numerical precision. An x-ONLY write-back would mislocate a2's away-team rearguard in y and break it.
    Non-vacuous: the two scenes are genuinely different (different acting team + positions) and the arm
    value is ~7, not ~0.
    """
    actions, frames = make_rest_defense_fixture()
    arm, _ = _outfield(actions, frames)
    a0 = float(arm[arm["action_id"] == 0][RD_OUTFIELD_DETER_THREAT].iloc[0])
    a2 = float(arm[arm["action_id"] == 2][RD_OUTFIELD_DETER_THREAT].iloc[0])
    assert abs(a0) > 1.0  # non-vacuity: the arm actually computed a non-trivial value
    assert a0 == pytest.approx(a2, abs=1e-9), (
        f"outfield threat arm not mirror-invariant: a0={a0} vs its 180-deg reflection a2={a2} "
        "(an x-only serve->frame write-back would break this)"
    )


def test_arms_do_not_mutate_their_inputs():
    """ADR-033: the arms are PURE -- neither the outfield nor the keeper arm mutates actions/frames."""
    from silly_kicks.restdefense._arms import rest_defense_gk_deterrent
    from tests.tracking.test_ghost_gk import _fitted_model

    actions, frames = make_rest_defense_fixture()
    a_before, f_before = actions.copy(deep=True), frames.copy(deep=True)
    rest_defense_outfield_deterrent(
        actions, frames, xt=make_fitted_xt(), ghost_outfield_model=_fit_toy()[0], home_team_id=1
    )
    rest_defense_gk_deterrent(actions, frames, xt=make_fitted_xt(), ghost_gk_model=_fitted_model()[0], home_team_id=1)
    pd.testing.assert_frame_equal(actions, a_before)
    pd.testing.assert_frame_equal(frames, f_before)


# --- Task 8: merge_rest_defense + summarize arm rollup -----------------------------------------


def test_merge_left_joins_all_samples():
    actions, frames = make_rest_defense_fixture()
    xt = make_fitted_xt()
    samples, _ = compute_rest_defense(actions, frames, xt=xt)
    arm, _ = _outfield(actions, frames)
    merged = merge_rest_defense(samples, arm)
    assert len(merged) == len(samples)  # left-join keeps every sample
    assert RD_OUTFIELD_DETER_THREAT in merged.columns
    assert RD_OUTFIELD_SOURCE in merged.columns


def test_merge_raises_on_arm_key_absent_from_samples():
    import pytest

    actions, frames = make_rest_defense_fixture()
    samples, _ = compute_rest_defense(actions, frames, xt=make_fitted_xt())
    bad = pd.DataFrame(
        [
            {
                "game_id": "ZZZ",
                "period_id": 9,
                "team_id": 99,
                "action_id": 999999,
                RD_OUTFIELD_DETER_THREAT: -0.1,
                RD_OUTFIELD_DETER_SPACE: -0.2,
                RD_OUTFIELD_SOURCE: "computed",
            }
        ]
    )
    with pytest.raises(ValueError, match="SUBSET"):
        merge_rest_defense(samples, bad)


def test_summarize_means_arm_columns_only_when_present():
    actions, frames = make_rest_defense_fixture()
    xt = make_fitted_xt()
    samples, _ = compute_rest_defense(actions, frames, xt=xt)
    # Layer-1/2-only samples: no arm column (byte-identical to PR1/PR2)
    base = summarize_rest_defense(samples, by="match")
    assert RD_OUTFIELD_DETER_THREAT not in base.columns
    # merged table: the arm column is meaned
    arm, _ = _outfield(actions, frames)
    merged = merge_rest_defense(samples, arm)
    per_match = summarize_rest_defense(merged, by="match")
    assert RD_OUTFIELD_DETER_THREAT in per_match.columns


def test_layer3_arm_columns_non_null_and_non_constant():
    """Experimental-arm liveness (relocated from the shipped-metric liveness gate when the arms were
    demoted to experimental, ADR-089): every emitted arm column is non-NaN and non-constant on the
    `computed` rows of the multi-domain fixture. Checked on the `computed`-source rows (a `ghost_missing`
    row is an honest NaN, not a liveness failure)."""
    from silly_kicks.restdefense._arms import rest_defense_gk_deterrent, rest_defense_outfield_deterrent
    from silly_kicks.restdefense._columns import (
        RD_GK_ARM_COLUMNS,
        RD_GK_SOURCE,
        RD_OUTFIELD_ARM_COLUMNS,
        RD_OUTFIELD_SOURCE,
    )
    from tests.tracking.test_ghost_gk import _fitted_model

    actions, frames = make_rest_defense_fixture()
    xt = make_fitted_xt()
    of_arm, _ = rest_defense_outfield_deterrent(
        actions, frames, xt=xt, ghost_outfield_model=_fit_toy()[0], home_team_id=1
    )
    gk_arm, _ = rest_defense_gk_deterrent(actions, frames, xt=xt, ghost_gk_model=_fitted_model()[0], home_team_id=1)
    for arm, cols, source in (
        (of_arm, RD_OUTFIELD_ARM_COLUMNS, RD_OUTFIELD_SOURCE),
        (gk_arm, RD_GK_ARM_COLUMNS, RD_GK_SOURCE),
    ):
        computed = arm[arm[source] == "computed"]
        assert len(computed) >= 2, f"fixture scored <2 computed rows for {source}"
        for c in cols:
            assert computed[c].notna().all(), f"arm column {c} has a NaN on a computed row"
            vals = computed[c].dropna()
            assert vals.nunique() > 1, f"arm column {c} is constant across computed samples"
