"""TF-19 A+2 Task 1: the physics-arm dose imposer.

Uses the real in-domain gkdv fixture + a fitted ghost model (the seam every gkdv engine test
uses), so `build_ghost_frames` produces genuine provenance -- the imposer sources the defending
keeper + actual position from there and from `frames`, NOT from `provenance_to_targets` (whose
7-col contract can't supply it; see plan P1).

Fixture geometry (`in_domain_frames`): home (team 1) defends x=0 and IS the defending team;
its keeper is `p1` at (5, 34); away keeper `a1` is at (100, 34); ball ~20 m from the x=0 goal.
"""

from __future__ import annotations

import pandas as pd

from tests.gkdv._fixtures import in_domain_frames
from tests.tracking.test_ghost_gk import _fitted_model


def _model():
    return _fitted_model()[0]  # (model, X, labels) -> model


def _defending_gk(frames):
    return frames[(frames["team_id"] == 1) & frames["is_goalkeeper"].astype(bool)]


def _away_gk(frames):
    return frames[(frames["team_id"] == 2) & frames["is_goalkeeper"].astype(bool)]


def _substitution_window(frames):
    """Add a SECOND defending-team (team 1) keeper row to every frame -- a real substitution window
    where the outgoing and incoming keeper are both on the pitch for the same ``frame_id`` (GS matches
    ship 22/23/24-player frames for exactly this reason). Before the ``_build_dose_targets`` dedup,
    this fanned the write-back merge out and raised ``ValueError: Length mismatch`` in
    ``_substitute_defending_keeper`` -- the real-data crash on WC2022 GS match 3828."""
    sub = _defending_gk(frames).drop_duplicates(subset=["game_id", "period_id", "frame_id"]).copy()
    sub["player_id"] = sub["player_id"].astype(str) + "_sub"
    sub["x"] = 8.0  # distinct from the starter (x=5), so a fan-out would move different rows
    sub["y"] = 30.0
    return pd.concat([frames, sub], ignore_index=True)


def test_saturating_goalline_puts_defending_keeper_on_the_defended_goal_line():
    from silly_kicks.gkdv._probe import impose_defending_keeper_dose

    frames = in_domain_frames()
    imposed, targets = impose_defending_keeper_dose(frames, home_team_id=1, dose="saturating_goalline", model=_model())
    assert len(targets), "fixture must produce at least one scored defending frame"
    gk = _defending_gk(imposed)
    assert float(gk["x"].iloc[0]) == 0.0  # defended goal line (x=0) centre
    assert float(gk["y"].iloc[0]) == 34.0
    assert float(_away_gk(imposed)["x"].iloc[0]) == 100.0  # away keeper unchanged
    assert float(_defending_gk(frames)["x"].iloc[0]) == 5.0  # input frame NOT mutated


def test_ladder_moves_defending_keeper_toward_the_defended_goal():
    from silly_kicks.gkdv._probe import impose_defending_keeper_dose

    frames = in_domain_frames()
    imposed, _ = impose_defending_keeper_dose(frames, home_team_id=1, dose="ladder", displacement=3.0, model=_model())
    # p1 at x=5 defending x=0 -> 3 m toward the goal -> x=2.0
    assert float(_defending_gk(imposed)["x"].iloc[0]) == 2.0


def test_saturating_x30_is_30m_from_the_defended_goal():
    from silly_kicks.gkdv._probe import impose_defending_keeper_dose

    frames = in_domain_frames()
    imposed, _ = impose_defending_keeper_dose(frames, home_team_id=1, dose="saturating_x30", model=_model())
    assert float(_defending_gk(imposed)["x"].iloc[0]) == 30.0  # goal-relative x=30 from x=0


def test_targets_carry_the_task3_contract_columns():
    from silly_kicks.gkdv._probe import impose_defending_keeper_dose

    frames = in_domain_frames()
    _, targets = impose_defending_keeper_dose(frames, home_team_id=1, dose="ladder", displacement=2.0, model=_model())
    # the Task 1 -> Task 3 producer/consumer contract (plan P1 ripple)
    for col in (
        "game_id",
        "period_id",
        "frame_id",
        "defending_team_id",
        "actual_x",
        "actual_y",
        "defended_goal_x",
        "imp_x",
        "imp_y",
    ):
        assert col in targets.columns, f"targets missing {col} (Task 1->Task 3 contract)"


def test_multi_keeper_frame_does_not_fan_out():
    """Regression: a substitution window (two defending keepers in one frame) must NOT crash.

    Without the ``_build_dose_targets`` keeper dedup, the write-back LEFT merge fans out and
    ``_substitute_defending_keeper``'s ``joined.index = gk_side.index`` raises ``Length mismatch``
    -- the exact real-data failure on WC2022 GS match 3828 (frames carry 22/23/24 players).
    """
    from silly_kicks.gkdv._probe import impose_defending_keeper_dose

    frames = _substitution_window(in_domain_frames())
    # (pre-fix this RAISES ValueError: Length mismatch)
    imposed, targets = impose_defending_keeper_dose(frames, home_team_id=1, dose="saturating_goalline", model=_model())

    assert len(targets), "domain must be non-empty (else the guard is vacuous)"
    # the dedup made targets one row per (frame, defending team) -- no fan-out
    key = ["game_id", "period_id", "frame_id", "defending_team_id"]
    assert not targets.duplicated(subset=key).any(), "targets fanned out on the multi-keeper frame"
    # substitution is in-place: no rows added or lost
    assert len(imposed) == len(frames)
    # non-vacuity: the defending keeper(s) actually moved to the defended goal line (x=0), factual was x!=0
    defending = imposed[(imposed["team_id"] == 1) & imposed["is_goalkeeper"].astype(bool)]
    assert (defending["x"] == 0.0).all(), "defending keeper rows not substituted to the dose position"
    assert (_defending_gk(frames)["x"] != 0.0).all(), "factual keeper already at the goal line -- vacuous"
    # the away keeper is untouched
    assert float(_away_gk(imposed)["x"].iloc[0]) == 100.0
