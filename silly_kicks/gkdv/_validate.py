"""GKDV validation harness: spec §6.1-6.3 registered constants + §6.4 Layer 4.

Layers 0-3 and ``gkdv_discrimination_verdict`` are PR-3b, after owner sign-off -- they are
deliberately NOT built here. Layer 4 ships in PR-3 (spec §6.4 review round 2, N3(a))
because it GATES §6.1's ICC, which ships here: shipping the primary criterion without the
guard the spec says must precede its interpretation would hand anyone running §6.1 between
the two PRs a number the spec forbids interpreting, with nothing saying so.

Every constant below is REGISTERED -- locked in code before the owner run. Changing one
after a run has been executed invalidates the pre-registration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Pre-registered ICC anchor band (spec §1.3, measured 0.015-0.026 across cohorts).
#: A RANGE, not a point: SkillCorner's 0.0147 sits below a single 0.02 anchor, so a
#: power curve is reported at all three rather than at a midpoint.
#:
#: PRECONDITION DISCHARGED 2026-07-28 (``docs/research/tf19_signoff_power/``): §6.1 registers the
#: gate only if detection at the anchor is >= 0.8, and the plasmode curve returned **1.0 at all
#: three** on the 64-match GKDV arm-values corpus (123,430 scored frames, 41 keepers). The
#: null-calibration check is what makes that credible rather than suspicious:
#: ``mean_observed_icc_at_zero = -0.00034``, i.e. with NO injected effect the estimator returns
#: ~zero. 8 of the 41 keepers appear in a single match, for which the block permutation is a pure
#: relabelling -- reported, not hidden. Until this run the promise "a power curve is reported at
#: all three" was a docstring no code could keep (ADR-037 F2).
ICC_ANCHORS: tuple[float, float, float] = (0.015, 0.020, 0.026)

#: Row 5's ATT effect-size anchors: a RANGE, mirroring :data:`ICC_ANCHORS`, expressed as a RELATIVE
#: change in the outcome's base rate. The ICC anchor does NOT transfer -- it is a keeper-level
#: variance share, while row 5 gates a spell-level mean difference on a binary outcome, and no
#: mapping between them exists. Relative rather than absolute per spec §1.3: "scale-free relative
#: criteria + placebo bands are the honest idiom for small-probability quantities".
#: ``N_MIN_MATCHED`` is registered at the 0.15 anchor; the curve is reported at all three.
ATT_RELATIVE_ANCHORS: tuple[float, float, float] = (0.10, 0.15, 0.20)

#: Layer 3's headroom threshold as a fraction of ``openGoal``'s OBSERVED range.
#: COMMITTED BEFORE the measurement: measuring the range first and choosing the fraction afterwards
#: would make the threshold tunable to any desired Layer 3 outcome, which is exactly what the
#: derivation duty exists to prevent. ``openGoal`` is a dimensionless open-goal-mouth fraction
#: constructively bounded by [0, 1], so on a corpus spanning most of that interval 2 % of the
#: observed range lands near the 0.02 the spec originally guessed -- the duty is discharged by making
#: that number derived and interpretable, not by moving it.
LAYER3_HEADROOM_RANGE_FRACTION: float = 0.02

#: Derived by ``scripts/run_signoff_power.py`` on the locked corpus: the smallest matched-n bin at
#: which ATT power reaches 0.80 at the 0.15 relative anchor, taken as the MAXIMUM over the two
#: Layer 2 outcomes (``Y_close_attempt`` has the lower base rate, so an ``N_min`` derived on
#: ``Y_attempt`` alone would be anti-conservative for the outcome row 7 fires on).
#:
#: MEASURED 2026-07-28 (``docs/research/tf19_signoff_power/``, run_commit ``6b242cf``, clean tree):
#: still ``None`` -- but the meaning has CHANGED and the distinction is the point. It no longer
#: reads "the run has not happened"; it reads "the run happened and NO bin reaches 0.80". Max power
#: was **0.055** at n=8000, against a required 0.80, at every anchor and for both outcomes -- the
#: corpus carries only 151 treated spells (prevalence 0.0041). Crucially the degenerate-replicate
#: counts were **0/200 at n=4000 and n=8000**, so this is an estimable design with no power, NOT a
#: positivity failure masquerading as one. Per §6.1 the response is to adjust floors/sampling
#: FIRST; a row-5 threshold is not registered on a corpus that cannot support one, and the 16.5 m
#: Layer 2 treatment threshold is NOT retuned to raise prevalence (Law-defined, so the decider
#: stays untuned -- changing it is a re-registration decision).
#:
#: Contrast the ICC leg, which the SAME run discharged: power 1.0 at all three ``ICC_ANCHORS``.
#: The two estimands were conflated until ADR-037 F3 split them, and they answer OPPOSITELY.
N_MIN_MATCHED: int | None = None

#: Layer 4: minimum mean signed goal-relative depth separation between outer terciles,
#: in metres. Below this the arm is not tracking a behaviour keepers actually vary.
TERCILE_SEPARATION_M: float = 0.5

#: Per-arm expected direction (spec §5): both arms are ATTACKER-value, so a deterrent
#: keeper reads NEGATIVE on both. Registered here so the §6.2 sign panel cannot pick a
#: direction after seeing the data.
EXPECTED_DIRECTION: dict[str, str] = {"delta_das": "negative", "delta_threat": "negative"}

#: Arm OUTPUT column -> :data:`EXPECTED_DIRECTION` key. The threat arm's OUTPUT column is
#: ``delta_threat_suppression`` (``_arms.py``), but its registered direction key is ``delta_threat``;
#: this is the canonical bridge, so a new arm cannot silently skip its §6.2 sign check and an unmapped
#: arm raises rather than passing.
_ARM_DIRECTION_KEY: dict[str, str] = {
    "delta_das": "delta_das",
    "delta_threat_suppression": "delta_threat",
}


def expected_direction_for_arm(arm_column: str) -> str:
    """The expected sign for an arm's OUTPUT column (``"negative"`` == deterrent).

    The threat arm emits ``delta_threat_suppression`` while :data:`EXPECTED_DIRECTION` is keyed on
    ``delta_threat``; this bridges the arm column to its direction key so every arm column resolves.
    An arm column absent from :data:`_ARM_DIRECTION_KEY` raises ``KeyError`` (never a silent skip).

    Examples
    --------
    >>> from silly_kicks.gkdv import expected_direction_for_arm
    >>> expected_direction_for_arm("delta_das")
    'negative'
    >>> expected_direction_for_arm("delta_threat_suppression")
    'negative'
    """
    return EXPECTED_DIRECTION[_ARM_DIRECTION_KEY[arm_column]]


#: The two Layer 4 verdicts. ``uninterpretable`` is NOT a failure of the keepers -- it is a
#: statement that the ICC computed on this arm carries no information about them.
_ANCHORED = "anchored"
_UNINTERPRETABLE = "uninterpretable"


def behavioural_anchoring_verdict(per_keeper: pd.DataFrame, *, value_col: str, depth_col: str) -> str:
    """Layer 4: is the arm tracking a behaviour keepers actually VARY?

    Splits keepers into terciles by arm value; the top and bottom terciles must differ in
    mean signed goal-relative depth by at least :data:`TERCILE_SEPARATION_M`. If they do
    not, the arm's ICC is reported ``"uninterpretable"`` rather than as evidence.

    This is the guard the sibling possession-value metric's failure teaches: a metric can
    read flat because it rewards a behaviour good keepers do not perform, in which case an
    ICC near zero says nothing about keepers.

    **Rows with a missing value or depth are dropped before ranking**, and the mechanism is
    worth stating precisely because the obvious guess is wrong. A NaN does NOT poison the
    tercile mean -- pandas' ``mean`` skips it. What it does is worse, because it is silent:
    ``sort_values`` parks every NaN-VALUE row at the END, so those rows are ranked as though
    they were the highest-valued keepers and their real depths are averaged into the TOP
    tercile. They also inflate ``len(ranked)``, widening ``k`` and pulling additional
    genuine keepers into both outer terciles. Measured on a fixture whose 9 real keepers
    separate by 4.50 m: three NaN-value rows collapse it to 0.375 m -- flipping ``anchored``
    to ``uninterpretable`` with nothing in the output to say why.

    The verdict is a magnitude, not a direction: Layer 4 asks whether keepers vary, and §6.2
    (:data:`EXPECTED_DIRECTION`) is where direction is adjudicated.

    LIMITATION, stated rather than guarded: with fewer than three keepers the tercile
    reduces to the single extreme row on each side. The spec registers no minimum keeper
    count for Layer 4, so none is invented here -- the §6.1 clustering floors
    (``aggregate_by_keeper``'s ``min_nonzero``/``min_games``) are what thin the surface, and
    they run first.

    Parameters
    ----------
    per_keeper : pd.DataFrame
        One row per keeper, as returned by
        :func:`~silly_kicks.gkdv.aggregate_by_keeper` joined to a depth column.
        Never mutated.
    value_col : str
        Per-keeper arm value used for the tercile ranking.
    depth_col : str
        Per-keeper mean SIGNED goal-relative depth (delta-x), in metres.

    Returns
    -------
    str
        ``"anchored"`` or ``"uninterpretable"``.

    Examples
    --------
    >>> per_keeper = pd.DataFrame(
    ...     {"value": [-0.03, 0.0, 0.03], "signed_dx": [-2.0, 0.0, 2.0]}
    ... )
    >>> behavioural_anchoring_verdict(per_keeper, value_col="value", depth_col="signed_dx")
    'anchored'
    """
    ranked = per_keeper.dropna(subset=[value_col, depth_col]).sort_values(value_col)
    if not len(ranked):
        return _UNINTERPRETABLE
    k = max(1, len(ranked) // 3)
    lo = float(ranked.head(k)[depth_col].mean())
    hi = float(ranked.tail(k)[depth_col].mean())
    separation = abs(hi - lo)
    if not np.isfinite(separation):
        return _UNINTERPRETABLE
    return _ANCHORED if separation >= TERCILE_SEPARATION_M else _UNINTERPRETABLE
