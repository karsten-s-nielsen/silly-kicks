"""The ghost path must obey the speed_source contract in BOTH directions.

CLAUDE.md: "An UNMARKED or PARTIALLY-marked frame set missing vx/vy still RAISES: fail-loud
wins on a mixed frame set." Measured before this cycle, the ghost path fabricated in both.
"""

from __future__ import annotations

import pytest

import silly_kicks.tracking as T
from tests.sb360._fixture import build_leg_a


def _marked():
    """Freeze-frame leg: every row already carries speed_source='unavailable'."""
    return build_leg_a()


def _unmarked():
    actions, frames, links = build_leg_a()
    frames = frames.copy()
    frames["speed_source"] = None
    return actions, frames, links


def test_marked_frames_degrade_to_nan_not_a_coordinate():
    actions, frames, _ = _marked()
    out = T.add_ghost_gk(actions, frames, home_team_id=1)
    assert out["ghost_gk_x"].isna().all(), "marked frames must not produce a coordinate"
    assert out["ghost_gk_y"].isna().all()
    # Provenance (Task 7): the SB360 degrade records which variant auto-select chose. `position_only`
    # even when unbundled (resolver picks it, then None -> NaN). D2: assert the VALUE type, not the
    # dtype literal (object on pandas 2, StringDtype on pandas 3 -- ADR-057).
    assert (out["ghost_gk_variant"] == "position_only").all()
    assert all(isinstance(v, str) for v in out["ghost_gk_variant"])


def test_unmarked_velocity_less_frames_RAISE():
    """The failing side. Fail-loud wins on a mixed frame set."""
    actions, frames, _ = _unmarked()
    with pytest.raises(ValueError, match="speed_source"):
        T.add_ghost_gk(actions, frames, home_team_id=1)


def test_the_marked_case_is_not_vacuous():
    """Non-vacuity: the fixture must actually have rows to refuse on."""
    actions, frames, _ = _marked()
    assert len(actions) > 0 and len(frames) > 0


def test_precomputed_ghosts_UNMARKED_and_velocity_less_still_RAISE():
    """The hole a marker-only check leaves open.

    ``velocity_unavailable_by_design`` is an ALL-rows predicate, so an UNMARKED or
    PARTIALLY-marked frame set returns False. If the aggregator relied on that alone, the
    precompute short-circuit would be reached, ``compute_ghost_gk`` would never run, and the
    serving-seam guard would never fire. Measured before the fix: ``ghost_gk_x = 52.5``.

    ``build_leg_a`` cannot catch this -- it carries NO precomputed ghost columns, so
    ``notna().any()`` is False and the short-circuit is skipped.
    """
    from tests.test_add_star_purity import _frames_with_ghost, make_actions

    frames = _frames_with_ghost().copy()
    frames = frames.drop(columns=[c for c in ("vx", "vy") if c in frames.columns])
    frames["speed_source"] = None
    assert frames["ghost_gk_x"].notna().any(), "fixture must pre-populate ghosts or this is vacuous"

    with pytest.raises(ValueError, match="vx/vy"):
        T.add_ghost_gk(make_actions(), frames, home_team_id=5)
