"""build_ghost_frames: domain, drop accounting, write-back, purity (spec §4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.gkdv import GkdvParams, build_ghost_frames
from silly_kicks.tracking import infer_ball_carrier
from tests.gkdv._fixtures import in_domain_frames, multi_frame_in_domain
from tests.tracking.test_ghost_gk import _fitted_model


def _build(frames, **kwargs):
    return build_ghost_frames(frames, model=_fitted_model()[0], home_team_id=1, **kwargs)


def _scored(prov: pd.DataFrame) -> pd.DataFrame:
    return prov[prov["drop_reason"].isna()]


# ---------------------------------------------------------------------------
# Non-vacuity anchor
# ---------------------------------------------------------------------------


def test_the_fixture_actually_reaches_the_domain():
    """META: every assertion below is over the SCORED rows, and an empty scored set makes
    them all pass vacuously. The anchor fixture parks the ball 50 m from the attacked goal
    (outside the 35 m domain), which is exactly how that vacuity arises -- so pin it here."""
    _cf, prov, report = _build(in_domain_frames())
    assert report.n_frames_scored == 1, report.drop_reasons
    assert len(_scored(prov)) == 2, "both teams' keepers are expected on a scored frame"


# ---------------------------------------------------------------------------
# Return contract
# ---------------------------------------------------------------------------


def test_returns_counterfactual_provenance_and_report():
    cf, prov, report = _build(in_domain_frames())
    assert isinstance(cf, pd.DataFrame) and isinstance(prov, pd.DataFrame)
    assert report.n_frames_in >= report.n_frames_scored
    assert set(prov.columns) >= {
        "game_id",
        "period_id",
        "frame_id",
        "gk_team_id",
        "player_id",
        "ghost_x",
        "ghost_y",
        "displacement_m",
        "ghost_clamped",
        "ghost_out_of_box",
        "drop_reason",
    }


def test_report_echoes_the_params_actually_used():
    """Registration without traceability is not registration."""
    params = GkdvParams(possession_stride=1, domain_ball_to_goal_m=40.0)
    _cf, _prov, report = _build(in_domain_frames(), params=params)
    assert report.params is params


# ---------------------------------------------------------------------------
# Purity (constraint: never mutate the caller's frames)
# ---------------------------------------------------------------------------


def test_input_frames_are_never_mutated():
    frames = in_domain_frames()
    before = frames.copy(deep=True)
    _build(frames)
    pd.testing.assert_frame_equal(frames, before)


def test_counterfactual_is_a_new_object():
    frames = in_domain_frames()
    cf, _prov, _r = _build(frames)
    assert cf is not frames


# ---------------------------------------------------------------------------
# Drop accounting: dropped-and-COUNTED, never scored as Delta = 0
# ---------------------------------------------------------------------------


def test_drop_reasons_conserve():
    _cf, prov, report = _build(multi_frame_in_domain(6))
    dropped = int(prov["drop_reason"].notna().sum())
    assert dropped > 0, "non-vacuity: the stride must actually drop frames here"
    assert report.n_frames_scored + dropped == report.n_frames_in
    assert sum(report.drop_reasons.values()) == dropped


def test_stride_skips_are_a_COUNTED_drop_reason_not_a_silent_discard():
    """A silently discarded frame is neither scored nor dropped -- the report would not
    conserve and the shortfall would be invisible."""
    _cf, _prov, report = _build(multi_frame_in_domain(6), params=GkdvParams(possession_stride=5))
    assert report.drop_reasons.get("stride_skipped") == 4
    assert report.n_frames_scored == 2


def test_out_of_domain_frame_is_dropped_and_left_UNTOUCHED_in_the_counterfactual():
    """The bias this domain exists to prevent: a frame we cannot score must not surface as
    a legitimate Delta = 0 (which reads as 'this keeper deterred nothing')."""
    frames = in_domain_frames()
    # Move the ball AND its carrier together, so possession still resolves and the ONLY
    # thing that changes is the distance to the attacked goal (x=0).
    frames.loc[frames["is_ball"].astype(bool), "x"] = 60.0
    frames.loc[frames["player_id"] == "a13", "x"] = 60.2
    cf, prov, report = _build(frames)
    assert report.n_frames_scored == 0
    assert report.drop_reasons == {"ball_far_from_attacked_goal": 1}
    assert len(_scored(prov)) == 0
    pd.testing.assert_frame_equal(cf, frames)


def test_missing_defending_gk_is_dropped_not_scored():
    frames = in_domain_frames()
    frames = frames[frames["player_id"] != "p1"].reset_index(drop=True)  # home (defending) GK
    _cf, prov, report = _build(frames)
    assert report.n_frames_scored == 0
    assert set(report.drop_reasons) == {"no_defending_gk"}
    assert len(_scored(prov)) == 0


def test_both_keepers_mapped_to_the_same_end_is_dropped_as_degenerate():
    """A realistic upstream `is_goalkeeper` mis-flag puts both teams' GK-mean-x at one end.
    Scoring such a frame would place the attacked goal behind the attackers."""
    frames = in_domain_frames()
    frames.loc[frames["player_id"] == "a1", "x"] = 10.0  # away "keeper" now at the home end
    _cf, _prov, report = _build(frames)
    assert report.n_frames_scored == 0
    assert report.drop_reasons == {"goal_map_degenerate": 1}


def test_dead_ball_is_dropped():
    frames = in_domain_frames()
    frames["ball_state"] = "dead"
    _cf, _prov, report = _build(frames)
    assert report.drop_reasons == {"ball_not_alive": 1}


# ---------------------------------------------------------------------------
# Finite-coordinate guard (pitch control SILENTLY DROPS NaN rows)
# ---------------------------------------------------------------------------


def test_scored_ghost_coordinates_are_finite():
    """A NaN ghost is SILENTLY DROPPED by pitch control -- never emit one."""
    _cf, prov, _r = _build(in_domain_frames())
    scored = _scored(prov)
    assert len(scored) > 0
    assert np.isfinite(scored["ghost_x"]).all()
    assert np.isfinite(scored["ghost_y"]).all()


def test_a_non_finite_ghost_RAISES_rather_than_making_the_keeper_vanish(monkeypatch):
    """The engine must fail loud. `_spearman.py` does `.dropna(subset=["x","y"])`, so a NaN
    ghost removes the keeper from the pitch-control surface instead of erroring -- the delta
    would then measure 'no keeper at all', not 'a league-average keeper'."""
    model = _fitted_model()[0]
    monkeypatch.setattr(
        type(model),
        "predict_mean",
        lambda self, feats: np.column_stack([np.full(len(feats), np.nan), np.full(len(feats), 34.0)]),
    )
    with pytest.raises(ValueError, match="non-finite ghost coordinate"):
        build_ghost_frames(in_domain_frames(), model=model, home_team_id=1)


# ---------------------------------------------------------------------------
# Write-back
# ---------------------------------------------------------------------------


def test_writeback_moves_the_DEFENDING_keeper_and_ONLY_that_keeper():
    """The plan's in-range assertion passes on UNMODIFIED frames; this asserts the actual
    substitution and its team selection."""
    frames = in_domain_frames()
    cf, prov, _r = _build(frames)
    scored = _scored(prov)
    defending = scored[scored["gk_team_id"] == scored["defending_team_id"]]
    assert len(defending) == 1

    gk = cf["is_goalkeeper"].astype(bool) & ~cf["is_ball"].astype(bool)
    moved = cf.loc[gk & (cf["player_id"] == "p1")]
    kept = cf.loc[gk & (cf["player_id"] == "a1")]
    assert float(moved["x"].iloc[0]) == pytest.approx(float(defending["ghost_x"].iloc[0]))
    assert float(moved["y"].iloc[0]) == pytest.approx(float(defending["ghost_y"].iloc[0]))
    assert float(moved["x"].iloc[0]) != pytest.approx(5.0), "the defending keeper did not move"
    # The attacking team's keeper is served (both appear in provenance) but must NOT be
    # substituted: doing so would contaminate the counterfactual with a second intervention.
    assert float(kept["x"].iloc[0]) == pytest.approx(100.0)
    assert float(kept["y"].iloc[0]) == pytest.approx(34.0)


def test_writeback_keeps_all_positions_on_the_pitch():
    cf, _prov, _r = _build(in_domain_frames())
    gk_rows = cf[cf["is_goalkeeper"].astype(bool) & ~cf["is_ball"].astype(bool)]
    assert len(gk_rows) == 2
    assert gk_rows["x"].between(0.0, 105.0).all()
    assert gk_rows["y"].between(0.0, 68.0).all()


def test_velocity_policy_zeroes_the_ghost_when_disabled():
    """The registered sensitivity variant (spec §4.5) must actually change something."""
    frames = in_domain_frames()
    frames.loc[frames["player_id"] == "p1", ["vx", "vy", "speed"]] = [3.0, 1.0, 3.2]
    keep, _p, _r = _build(frames, params=GkdvParams())
    zeroed, _p2, _r2 = _build(frames, params=GkdvParams(ghost_keeps_actual_velocity=False))
    gk = lambda d: d.loc[d["player_id"] == "p1"].iloc[0]  # noqa: E731
    assert float(gk(keep)["vx"]) == pytest.approx(3.0)
    assert float(gk(zeroed)["vx"]) == 0.0
    assert float(gk(zeroed)["speed"]) == 0.0


def test_out_of_box_flag_keys_on_GOAL_RELATIVE_x_and_survives_writeback(monkeypatch):
    """Plant a ghost 90 m goal-relative -- far outside the 30 m trained label hull.

    The AWAY keeper defends x=105, so write-back puts it at a frame x of 105-90 = 15, which
    is INSIDE 30. That asymmetry is the whole point: a flag recomputed on the post-flip
    frame coordinate would report False for that row, so the surviving True proves the flag
    keys on the goal-relative value. (The plan planted 45 m, whose flip lands at 60 -- still
    >30, so a post-flip recompute produced the same answer and the test could not fail.)
    """
    model = _fitted_model()[0]
    monkeypatch.setattr(
        type(model),
        "predict_mean",
        lambda self, feats: np.column_stack([np.full(len(feats), 90.0), np.full(len(feats), 34.0)]),
    )
    _cf, prov, _r = build_ghost_frames(in_domain_frames(), model=model, home_team_id=1)
    scored = _scored(prov)
    assert len(scored) == 2, "both keepers must be present or the flip leg is untested"
    assert set(np.round(scored["ghost_x"].to_numpy(), 3)) == {90.0, 15.0}
    flipped = scored[scored["ghost_x"] < 30.0]
    assert len(flipped) == 1, "non-vacuity: the discriminating row must exist"
    assert bool(flipped["ghost_out_of_box"].iloc[0]), "flag recomputed post-flip, not preserved"
    assert scored["ghost_out_of_box"].all(), "flag lost across write-back"
    assert (scored["ghost_x"] < 105.0).all() and (scored["ghost_x"] > 0.0).all()


# ---------------------------------------------------------------------------
# ADR-019 id safety
# ---------------------------------------------------------------------------


def test_dtype_mismatched_home_team_id_gives_IDENTICAL_output():
    """`home_team_id` is a caller-supplied scalar of uncontrolled dtype. A raw ==/!= against
    an id column is the most damaging bug shape in this codebase. The repo-wide guard is the
    enumerated registry (tests/invariants/test_public_id_scalar_registry.py), which reaches
    only PUBLIC entry points; `build_ghost_frames` is covered here, at the behavioural level,
    on the engine's own fixture."""
    frames = in_domain_frames()
    cf_int, prov_int, rep_int = build_ghost_frames(frames, model=_fitted_model()[0], home_team_id=1)
    cf_str, prov_str, rep_str = build_ghost_frames(frames, model=_fitted_model()[0], home_team_id="1")
    pd.testing.assert_frame_equal(cf_int, cf_str)
    pd.testing.assert_frame_equal(prov_int, prov_str)
    assert rep_int.n_frames_scored == rep_str.n_frames_scored


def test_object_boxed_carrier_team_id_still_resolves_the_defending_keeper():
    """A carrier frame carrying `ball_carrier_team_id` as an OBJECT column of floats must
    still resolve the defending keeper.

    HISTORY -- this guard's premise deliberately moved. `infer_ball_carrier` USED to emit
    exactly this shape, and `canonical_id_series` USED to render it "2.0" against the frames'
    "2", so `ids_equal` was False for EVERY row: both teams looked like defending candidates
    and the domain emptied. Both root causes are now fixed (the emitter restores its source
    dtype; the canonicalizer probes object CONTENT instead of trusting dtype).

    So this is now a DEFENSIVE guard, not a description of the emitter. `build_ghost_frames`
    accepts a CALLER-SUPPLIED `carrier=`, and nothing stops an external caller handing us a
    boxed-object column, so the tolerance must not regress. The emitter's own dtype contract
    is pinned separately in `tests/test_id_compat.py`.
    """
    frames = in_domain_frames()
    carrier = infer_ball_carrier(frames)

    # Rebuild the legacy boxed-object shape EXPLICITLY, so this tests tolerance of the input
    # rather than the emitter's (now-fixed) output dtype.
    boxed = carrier.copy()
    boxed["ball_carrier_team_id"] = boxed["ball_carrier_team_id"].astype(object)
    assert boxed["ball_carrier_team_id"].dtype == object
    assert isinstance(boxed["ball_carrier_team_id"].iloc[0], float), "fixture must hold boxed FLOATS"

    _cf, prov, report = build_ghost_frames(frames, model=_fitted_model()[0], home_team_id=1, carrier=boxed)
    assert report.n_frames_scored == 1, report.drop_reasons
    assert float(_scored(prov)["defending_team_id"].iloc[0]) == 1.0
