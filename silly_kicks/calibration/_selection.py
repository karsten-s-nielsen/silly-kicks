"""TF-24 Stage-1 recommendation selection (ADR-060).

Pure decision logic: given the CV-fold scores of the shipped incumbent and a set of candidate
points, decide whether to move the recommendation. `beta`/`gamma` are non-identifiable, so the rule
is prefer-incumbent under TWO bars -- a practical effect-size floor AND a paired-difference-SE
significance test -- both required to move. `tolerance_m` is NOT part of this decision (held constant;
see ADR-060). I/O lives in the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

from silly_kicks.calibration._cv import cv_standard_error
from silly_kicks.calibration._diagnostics import exceeds_noise_floor

#: Practical-significance floor on carrier accuracy, FROZEN at 0.005. The 179-match confirmation
#: (run_commit 2cecd2b, docs/research/tf24_stage1_confirmation/) showed the keep-incumbent
#: recommendation is invariant to this value across [0, 0.1]: the shipped default is the outright
#: highest-mean point, so no candidate clears any positive floor and the result does not hinge on it.
MIN_EFFECT_SIZE: float = 0.005


@dataclass(frozen=True)
class PointScore:
    """A parameter point's per-CV-fold carrier accuracy. `per_fold` is aligned by fold index across
    every point (all points are scored on the SAME folds), which is what makes the paired SE valid."""

    label: str
    params: dict
    per_fold: tuple[float, ...]
    mean: float


@dataclass(frozen=True)
class Selection:
    selected: PointScore
    incumbent: PointScore
    moved: bool
    reason: str
    best_candidate: PointScore | None
    effect_size: float | None
    paired_se: float | None


def select_recommended_point(
    *,
    incumbent: PointScore,
    candidates: list[PointScore],
    min_effect_size: float = MIN_EFFECT_SIZE,
    policy: str = "prefer_incumbent",
) -> Selection:
    """Prefer-incumbent selection. Returns the incumbent unless some candidate clears BOTH the
    effect-size floor (strict `gain > min_effect_size`) and the paired-difference-SE test.

    Examples
    --------
    >>> from silly_kicks.calibration._selection import PointScore, select_recommended_point
    >>> inc = PointScore("shipped", {"beta": 0.0, "gamma": 0.25}, (0.79, 0.80, 0.81), 0.80)
    >>> cand = PointScore("c", {"beta": 0.1, "gamma": 0.3}, (0.791, 0.801, 0.811), 0.801)
    >>> # the ~0.001 gain does not clear the 0.01 floor, so the incumbent is kept
    >>> select_recommended_point(incumbent=inc, candidates=[cand], min_effect_size=0.01).moved
    False
    """
    if policy != "prefer_incumbent":
        raise ValueError(f"unknown policy {policy!r}; only 'prefer_incumbent' is implemented")

    clearing: list[tuple[PointScore, float, float]] = []
    for c in candidates:
        if len(c.per_fold) != len(incumbent.per_fold):
            raise ValueError(
                "per_fold length mismatch: candidate and incumbent must be scored on the same folds "
                f"({len(c.per_fold)} vs {len(incumbent.per_fold)})"
            )
        gain = c.mean - incumbent.mean
        paired_se = cv_standard_error([a - b for a, b in zip(c.per_fold, incumbent.per_fold, strict=True)])
        if gain > min_effect_size and exceeds_noise_floor(gain, paired_se):
            clearing.append((c, gain, paired_se))

    if not clearing:
        return Selection(
            selected=incumbent,
            incumbent=incumbent,
            moved=False,
            reason="no candidate cleared both the effect-size floor and the paired-SE test",
            best_candidate=None,
            effect_size=None,
            paired_se=None,
        )
    # Deterministic tie-break: highest gain, then label, so equal gains do not depend on input order.
    best, gain, paired_se = max(clearing, key=lambda t: (t[1], t[0].label))
    return Selection(
        selected=best,
        incumbent=incumbent,
        moved=True,
        reason=f"candidate {best.label!r} cleared both bars (gain {gain:.6g} > δ, paired_se {paired_se:.6g})",
        best_candidate=best,
        effect_size=gain,
        paired_se=paired_se,
    )


def build_selection_artifact(selection: Selection, *, provenance: dict) -> dict:
    """The committed `carrier_selected.json` payload. PURE -- provenance is passed in, never read
    from git here, so the builder is unit-testable and the caller owns the I/O.

    Carries `{beta, gamma}` (never `tolerance_m` -- held constant, ADR-060) plus the run provenance
    the structural output gate (`tests/scripts/test_artifact_provenance_output.py`) requires.

    Examples
    --------
    >>> from silly_kicks.calibration._selection import (
    ...     PointScore,
    ...     Selection,
    ...     build_selection_artifact,
    ... )
    >>> pt = PointScore("shipped", {"beta": 0.0, "gamma": 0.25}, (0.80,), 0.80)
    >>> sel = Selection(
    ...     selected=pt,
    ...     incumbent=pt,
    ...     moved=False,
    ...     reason="kept",
    ...     best_candidate=None,
    ...     effect_size=None,
    ...     paired_se=None,
    ... )
    >>> sorted(build_selection_artifact(sel, provenance={"commit": "abc", "dirty": False}))
    ['beta', 'gamma', 'moved', 'reason', 'run_commit', 'run_tree_dirty']
    """
    return {
        "beta": float(selection.selected.params["beta"]),
        "gamma": float(selection.selected.params["gamma"]),
        "moved": selection.moved,
        "reason": selection.reason,
        "run_commit": provenance["commit"],
        "run_tree_dirty": provenance["dirty"],
    }
