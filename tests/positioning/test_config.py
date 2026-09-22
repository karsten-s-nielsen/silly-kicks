"""TF-56 positioning: frozen params with documented defaults (spec section 4, ADR-009)."""

from __future__ import annotations

import dataclasses

from silly_kicks.positioning import PositioningParams, ReachabilityParams, SAParams


def _is_frozen(obj) -> bool:
    # pyright does not synthesize __dataclass_params__ on the class; the attribute is real at runtime.
    return bool(type(obj).__dataclass_params__.frozen)  # type: ignore[attr-defined]


def test_params_are_frozen_with_documented_defaults():
    p = PositioningParams.default()
    assert dataclasses.is_dataclass(p) and _is_frozen(p)
    assert p.domain_ball_to_goal_m > 0 and p.sample_fps == 1.0

    r = ReachabilityParams.default()
    assert dataclasses.is_dataclass(r) and _is_frozen(r)
    assert (r.max_reach_seconds, r.reaction_time) == (0.7, 0.1)
    assert r.max_acceleration == 7.0  # SpearmanParams.max_acceleration

    s = SAParams.default()
    assert dataclasses.is_dataclass(s) and _is_frozen(s)
    assert (s.num_iterations, s.patience) == (2000, 200)
    # init_sigma_m is the spike-fixed, reachability-matched default (spec section 8 amendment,
    # owner-ratified 2026-09-22): at 2.0 the SA optimum is seed-invariant (the optimizer-stability
    # instrument); the retired 5.0 left the column a partial seed artifact.
    assert s.init_sigma_m == 2.0


def test_domain_matches_gkdv_danger_distance():
    """The domain boundary is shared with the GKDV sibling (spec section 6)."""
    from silly_kicks.gkdv._engine import _DOMAIN_BALL_TO_GOAL_M

    assert PositioningParams.default().domain_ball_to_goal_m == _DOMAIN_BALL_TO_GOAL_M


def test_nested_params_are_the_frozen_defaults():
    p = PositioningParams.default()
    assert p.reachability == ReachabilityParams.default()
    assert p.sa == SAParams.default()


def test_for_provider_is_empty_adr009():
    # No per-provider tuning ships; for_provider returns the frozen default for any provider.
    for cls in (PositioningParams, ReachabilityParams, SAParams):
        assert cls.for_provider("skillcorner") == cls.default()
        assert cls.for_provider("sportec") == cls.default()
