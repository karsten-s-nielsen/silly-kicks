"""Task 11: the cover-shadow sigma/lambda DISCRIMINATION re-tuning objective (M1).

This re-tunes the sigma/lambda SHAPE for discrimination -- it does NOT calibrate ``p_blocked`` MAGNITUDE
(M1: magnitude calibration needs counterfactual block ground truth we lack; stated in the manifest).

- **Primary** -- maximise the failed-vs-completed margin-AUC (``p_blocked - p_received``) with the
  failed-pass ``p_blocked`` computed at the de-leaked target, on the trajectory-VALIDATED intercepted
  subset + completed passes (R2: use-population = validation-population).
- **Constraint** -- the completed-pass FP rate (predicted-blocked | completed) must not exceed the
  incumbent's; a sigma/lambda that buys AUC by over-blocking is rejected (returns NaN).
- **Out-of-play failures are NOT in the objective** (Low-1: empty-space, low-``p_blocked`` by
  construction -- not a blocking phenomenon).

The lane-pressure ablation (H2/R3) and the ``AugmentedVaepBrier`` cross-check are corpus-level (Task 14);
this module owns the pure per-(sigma, lambda) score + the ablation-share arithmetic.

See NOTICE for full bibliographic citations (Cascioli et al. 2025).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from silly_kicks.tracking._cover_shadows import CoverShadowParams, lane_control
from silly_kicks.tracking._gk_resolve import GoalMap


@dataclass
class PreparedPass:
    """One pass prepared for the sweep: a frame + passer/target (the de-leaked target for a failure)."""

    frame: pd.DataFrame
    passer_xy: tuple[float, float]
    target_xy: tuple[float, float]
    attacking_team_id: int | str
    goal_map: GoalMap
    is_fail: bool
    is_completed: bool


class CoverShadowDiscriminationObjective:
    """Maximise failed-vs-completed margin-AUC over (sigma, lambda), subject to the FP-rate constraint."""

    def __init__(self, passes: list[PreparedPass], *, incumbent_fp: float | None = None) -> None:
        """``incumbent_fp=None`` DISABLES the FP-rate constraint -- needed ONLY to bootstrap the incumbent's
        own FP measurement. The SELECTION sweep MUST pass the measured incumbent FP, or an over-blocking
        sigma/lambda that buys AUC wins silently (the constraint is a MUST, not a nicety)."""
        for p in passes:
            if p.is_fail and p.is_completed:
                raise ValueError("PreparedPass cannot be both is_fail and is_completed")
        self._passes = passes
        self._incumbent_fp = incumbent_fp

    def _measure(self, sigma: float, lambda_ctrl: float):
        params = CoverShadowParams(sigma=sigma, lambda_ctrl=lambda_ctrl)
        margin, is_fail, is_completed, is_blocked = [], [], [], []
        for p in self._passes:
            r = lane_control(
                p.frame,
                p.passer_xy,
                p.target_xy,
                goal_map=p.goal_map,
                attacking_team_id=p.attacking_team_id,
                params=params,
            )
            pb = np.mean([r.p_blocked_center, r.p_blocked_left, r.p_blocked_right])
            pr = np.mean([r.p_received_center, r.p_received_left, r.p_received_right])
            margin.append(float(pb - pr))
            is_fail.append(bool(p.is_fail))
            is_completed.append(bool(p.is_completed))
            is_blocked.append(bool(r.is_blocked_majority))
        return (np.array(margin), np.array(is_fail), np.array(is_completed), np.array(is_blocked))

    def score(self, sigma: float, lambda_ctrl: float) -> float:
        """Margin-AUC (maximise). NaN when a single class, or when the FP constraint is violated."""
        from sklearn.metrics import roc_auc_score

        margin, is_fail, is_completed, is_blocked = self._measure(sigma, lambda_ctrl)
        if len(np.unique(is_fail)) < 2:
            return float("nan")
        if self._incumbent_fp is not None and is_completed.any():
            fp = float(is_blocked[is_completed].mean())  # predicted-blocked among completed passes
            if fp > self._incumbent_fp:
                return float("nan")  # reject: bought AUC by over-blocking
        return float(roc_auc_score(is_fail, margin))

    def argmax(self, sigma_grid, lambda_grid) -> tuple[float, float, float]:
        """Grid argmax over (sigma, lambda); NaN scores (single-class / FP-violated) are skipped.

        Returns ``(nan, nan, -inf)`` when NO point is admissible (every candidate single-class or
        FP-rejected). The caller MUST treat a non-finite result as *no admissible point* --
        ``apply_cover_shadow_retune.decide_apply`` routes non-finite candidate params to the safe null,
        so a degenerate sweep can never drive a spurious ``applied``.
        """
        best = (float("nan"), float("nan"), -np.inf)
        for s in sigma_grid:
            for lam in lambda_grid:
                v = self.score(s, lam)
                if np.isfinite(v) and v > best[2]:
                    best = (s, lam, v)
        return best


def lane_pressure_shift_share(
    argmax_with: tuple[float, float],
    argmax_without: tuple[float, float],
    *,
    sigma_range: float,
    lambda_range: float,
) -> float:
    """H2/R3 ablation: normalised distance between the sigma/lambda argmax WITH vs WITHOUT lane-pressure
    in the receiver features -- the share of the sigma/lambda shift attributable to the open-target bias.

    ``sigma_range`` / ``lambda_range`` MUST be the grid SPAN (so ``|delta| <= range``); the result is
    CLAMPED to ``[0, 1]`` regardless, so a caller passing an over-wide range cannot understate the bias
    below ``MAX_BIAS_SHARE`` and let a bias-driven shift through as ``applied``.
    """
    ds = abs(argmax_with[0] - argmax_without[0]) / sigma_range if sigma_range else 0.0
    dl = abs(argmax_with[1] - argmax_without[1]) / lambda_range if lambda_range else 0.0
    return float(min(1.0, np.hypot(ds, dl) / np.sqrt(2.0)))
