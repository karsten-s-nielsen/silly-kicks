"""Frozen configuration for the TF-56 positioning optimiser (spec section 4).

All three params are frozen dataclasses with ``for_provider`` returning the frozen default
for any provider (ADR-009: no per-provider tuning ships this cycle -- the metric is
intent-set, not calibrated). See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Public seam: SpearmanParams.max_acceleration (7.0 m/s^2, lakehouse-calibrated) is the ONE
# kinematic constant the reachability horizon shares with pitch control, so the reachability
# default reads it rather than re-inventing a second copy.
from silly_kicks.tracking.pitch_control import SpearmanParams


@dataclass(frozen=True)
class ReachabilityParams:
    """Time-to-intercept reachability horizon for a repositioning candidate.

    ``max_reach_seconds`` / ``reaction_time`` are INTENT-SET demo values (the "where should
    they have been in the next fraction of a second" horizon), deliberately tighter than
    databallpy's loose 1.0 s so the optimum stays realistically reachable. ``max_acceleration``
    is the calibrated Spearman constant.

    Examples
    --------
    >>> ReachabilityParams.default().max_reach_seconds
    0.7
    """

    max_reach_seconds: float = 0.7
    reaction_time: float = 0.1
    max_acceleration: float = SpearmanParams().max_acceleration  # 7.0 m/s^2

    @classmethod
    def default(cls) -> ReachabilityParams:
        """The frozen default reachability horizon.

        >>> ReachabilityParams.default().max_reach_seconds
        0.7
        """
        return cls()

    @classmethod
    def for_provider(cls, provider: str) -> ReachabilityParams:
        """The frozen default for any provider (ADR-009: no per-provider tune ships).

        >>> ReachabilityParams.for_provider("skillcorner") == ReachabilityParams.default()
        True
        """
        return cls()


@dataclass(frozen=True)
class SAParams:
    """Simulated-annealing schedule (Oonk & Shah, databallpy ``optimization``, MIT).

    Examples
    --------
    >>> SAParams.default().num_iterations
    2000
    """

    num_iterations: int = 2000
    patience: int = 200
    init_sigma_m: float = 2.0  # initial Gaussian perturbation stddev, metres -- matched to the
    # ~2 m reachable radius: spike-verified SEED-INVARIANT optimum (spec section 8 amendment,
    # owner-ratified 2026-09-22). 5.0 left the optimum seed-noisy (per-frame gap std 0.324 vs
    # 0.000 at 2.0), which would VOID the optimizer-stability instrument.
    cooling: float = 0.995  # geometric temperature / sigma decay per iteration
    init_temperature: float = 1.0  # Metropolis accept-worse temperature (threat-Delta scale ~O(1))

    @classmethod
    def default(cls) -> SAParams:
        """The frozen default SA schedule.

        >>> SAParams.default().num_iterations
        2000
        """
        return cls()

    @classmethod
    def for_provider(cls, provider: str) -> SAParams:
        """The frozen default for any provider (ADR-009: no per-provider tune ships).

        >>> SAParams.for_provider("sportec") == SAParams.default()
        True
        """
        return cls()


@dataclass(frozen=True)
class PositioningParams:
    """Domain + sampling knobs for :func:`compute_positioning_gap`.

    ``domain_ball_to_goal_m`` reuses GKDV's established danger-domain distance
    (``gkdv._engine._DOMAIN_BALL_TO_GOAL_M`` = 35.0), so the two counterfactual siblings
    share one domain boundary.

    Examples
    --------
    >>> PositioningParams.default().sample_fps
    1.0
    """

    domain_ball_to_goal_m: float = 35.0  # == gkdv._engine._DOMAIN_BALL_TO_GOAL_M
    sample_fps: float = 1.0
    reachability: ReachabilityParams = field(default_factory=ReachabilityParams)
    sa: SAParams = field(default_factory=SAParams)

    @classmethod
    def default(cls) -> PositioningParams:
        """The frozen default positioning config.

        >>> PositioningParams.default().sample_fps
        1.0
        """
        return cls()

    @classmethod
    def for_provider(cls, provider: str) -> PositioningParams:
        """The frozen default for any provider (ADR-009: no per-provider tune ships).

        >>> PositioningParams.for_provider("gradientsports") == PositioningParams.default()
        True
        """
        return cls()
