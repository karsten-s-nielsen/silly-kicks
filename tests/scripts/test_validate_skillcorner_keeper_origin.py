"""A1: SkillCorner keeper-origin validation driver (keeper-box detection-quality cycle)."""

from __future__ import annotations

import pathlib

import validate_skillcorner_keeper_origin as drv

EXPECTED_COLS = set(drv.EXPECTED_COLS)
_ORIGIN_SOURCES = {"tracking_gk", "goalkick_prior", "unresolved", "native"}


def test_measure_match_shape(slim_skillcorner_match):
    out = drv.measure_match(*slim_skillcorner_match)
    assert set(out.columns) == EXPECTED_COLS
    assert len(out) > 0
    assert out["xt_gk_origin_source"].isin(_ORIGIN_SOURCES).all()


def test_fixture_exercises_the_domain_non_vacuously(slim_skillcorner_match):
    """M3: the fixture must genuinely exercise the GK-distribution domain, or A1/A2 are vacuous."""
    out = drv.measure_match(*slim_skillcorner_match)
    assert out["xt_gk_origin_source"].nunique() >= 2  # a mix of resolved sources
    assert out["is_goalkick"].all()
    assert out["in_own_box"].all()  # shipped distrust resolves every goal-kick in-box
    # the RAW diagnostic still sees the broadcast artifact, or A1's whole reason to exist is moot
    assert drv.raw_native_goalkick_out_of_region_rate(out) > 0.0


def test_resolved_origin_sources_are_imputed_not_native(slim_skillcorner_match):
    """SkillCorner goal-kicks are distrusted, so the shipped resolution IMPUTES -- never keeps native."""
    out = drv.measure_match(*slim_skillcorner_match)
    got = set(out["xt_gk_origin_source"].unique())
    assert {"tracking_gk", "goalkick_prior"} <= got
    assert "native" not in got  # every raw broadcast-ball origin was distrusted + imputed


def test_shipped_resolution_meets_the_own_box_acceptance(slim_skillcorner_match):
    """ADR-024 acceptance on the shipped resolution: goal-kick origins ~=100% own-box, none behind
    the line, no gross off-pitch (the resolver fixed the raw broadcast artifact)."""
    out = drv.measure_match(*slim_skillcorner_match)
    assert out["in_own_box"].all()
    assert int(out["is_behind_line"].sum()) == 0
    assert drv.offpitch_rate(out) == 0.0


def test_gated_rate_clean_but_raw_diagnostic_sees_the_artifact(slim_skillcorner_match):
    """The GATED out-of-region rate is 0 (resolver handled SkillCorner) while the RAW diagnostic
    reports the broadcast-ball artifact it corrected -- the before/after split."""
    out = drv.measure_match(*slim_skillcorner_match)
    assert drv.out_of_region_goalkick_rate(out) == 0.0  # gated / resolved
    assert drv.raw_native_goalkick_out_of_region_rate(out) > 0.0  # diagnostic / raw


def test_away_team_goalkick_is_action_ltr_not_frame_mirrored(slim_skillcorner_match):
    """THE ADR-028 defect this probe exists to catch, and the reason a single-team fixture is vacuous.

    a5 is an AWAY-team goal-kick: team 2 DEFENDS x=105 in the frame. ``resolve_gk_geometry`` emits its
    origin in ACTION-LTR (own goal at x=0), so ``gr_x = origin_x`` is small and in-box. A driver that
    read ``gr_x`` off the frame goal map (``resolve_defended_goals``, home-attacks-right) would place
    it at ~105-origin_x and call it out-of-box -- so this asserts the fix AND that the buggy path would
    MEASURABLY differ (non-vacuity), the exact real-data failure (28.6% vs 100% own-box)."""
    from silly_kicks.tracking._geometry import in_penalty_area_goal_relative

    out = drv.measure_match(*slim_skillcorner_match)
    away = out[out["action_id"] == 5]
    assert len(away) == 1, "fixture must carry the away-team goal-kick a5"
    row = away.iloc[0]
    assert bool(row["in_own_box"]) is True  # the FIX: action-LTR gr_x = origin_x
    assert float(row["origin_x"]) < 16.5  # imputed near the own goal (action-LTR)
    # the buggy frame-goal_map path would mirror it to ~105-origin_x and flip in_box to False:
    frame_mirrored_grx = 105.0 - float(row["origin_x"])
    assert not bool(in_penalty_area_goal_relative(frame_mirrored_grx, float(row["origin_y"])))


def test_does_not_mutate_inputs(slim_skillcorner_match):
    provider, match_id, actions, frames, home = slim_skillcorner_match
    a_before = actions.copy()
    f_before = frames.copy()
    drv.measure_match(provider, match_id, actions, frames, home)
    import pandas as pd

    pd.testing.assert_frame_equal(actions, a_before)
    pd.testing.assert_frame_equal(frames, f_before)


# --- driver-convention guards (ASCII-only source, argparse --help) ---


def _source() -> str:
    return (pathlib.Path(drv.__file__)).read_text(encoding="utf-8")


def test_driver_source_is_ascii_only():
    src = _source()
    assert src.isascii(), "scripts/ drivers must be ASCII-only (the driver ASCII gate)"


def test_driver_help_parses(capsys):
    import pytest

    with pytest.raises(SystemExit) as exc:
        import sys

        old = sys.argv
        sys.argv = ["validate_skillcorner_keeper_origin.py", "--help"]
        try:
            drv.main()
        finally:
            sys.argv = old
    assert exc.value.code == 0
