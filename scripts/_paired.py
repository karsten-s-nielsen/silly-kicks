"""The registered decision rules (spec 4.1 and 4.3; the two-arm rule, ADR-132). Pure; table-tested; no I/O.

These live in one place BECAUSE they decide what ships. A rule re-derived inside a training loop
is a rule nobody can review.
"""

from __future__ import annotations

from collections.abc import Sequence


def clears_rule(deltas: Sequence[float]) -> bool:
    """The unchanged 4.9.0/4.18.0 rule: positive in >= K-1 of K folds AND a positive mean."""
    k = len(deltas)
    if k < 2:
        return False
    n_pos = sum(1 for d in deltas if d > 0.0)
    return n_pos >= k - 1 and (sum(deltas) / k) > 0.0


#: The two arms of the paired path (ADR-132; corpus widening spec 5.2). `full` is retired for new artifacts.
PAIRED_ARMS = ("public", "sc_extended")


def is_paired(arm_names) -> bool:
    """True iff a run gets both paired arms, so the two-arm rule decides what ships (spec 5.2)."""
    names = set(arm_names)
    return all(a in names for a in PAIRED_ARMS)


def refuse_retired_arms(arm_names) -> None:
    """Fail loud on a `full` arm (ADR-132 retired it; `compute_arms` never emits it). A hand-written legacy
    ``--arms-json`` must not train a third arm."""
    if "full" in set(arm_names):
        raise ValueError(
            "the --arms-json payload carries a `full` arm; ADR-132 retired it (public vs sc_extended only). "
            "Rebuild the payload with arm_policy.compute_arms."
        )


def two_arm_ship(sc_extended_deltas: Sequence[float]) -> tuple[str, str]:
    """The two-arm rule (ADR-132; corpus widening spec 5.2, owner ruling Q-B): ship ``sc_extended`` iff its per-fold
    nested PR-AUC deltas over ``public`` (on the common public held-out folds) clear :func:`clears_rule`; else ship
    ``public`` -- ties go to less data. One test, one candidate: no sequence, no alpha correction."""
    if clears_rule(sc_extended_deltas):
        return "sc_extended", "sc_extended clears the rule vs public"
    return "public", "sc_extended does not clear the rule vs public -- ties go to less data"


def ghost_admission(detected_only_deltas: Sequence[float]) -> bool:
    """Ghost-GK admission (spec 4.3). Deltas are MAE_expanded - MAE_baseline; NEGATIVE is better.

    Admit only on a DEMONSTRATED improvement under sign-consistency, measured on frames where the
    keeper was actually SEEN -- a wash leaves the 81-match status quo in place. (The rev-1 fixed
    0.05 m band was never costed: the gate's own tolerated fold noise is ~10x that band.)

    NaN folds (single-class, no usable score) DROP OUT -- they must not veto the run.
    """
    usable = [d for d in detected_only_deltas if d == d]  # NaN != NaN
    return clears_rule([-d for d in usable])


def ghost_admission_report(
    detected_only_deltas: Sequence[float], all_frames_deltas: Sequence[float] | None = None
) -> tuple[bool, str]:
    """The verdict PLUS a reason string. The reason is a DIAGNOSTIC and decides nothing (spec rev 5).

    The rev-2 'interpolator tell' REFUSAL was retired: admission already requires detected-only
    improvement, so the refusal branch was reachable only when the fall-through returned False
    anyway -- it could never change a verdict. And under rev 3's detected-keeper TRAINING rule the
    mechanism is gone: the model never sees an interpolated target.

    What survives is the ability to say WHY a candidate failed -- 'no improvement anywhere' reads
    very differently from 'improved only where the keeper was invented' -- and that distinction
    belongs in the record, not in the gate.
    """
    verdict = ghost_admission(detected_only_deltas)
    if verdict:
        return True, "improved on detected keepers under sign-consistency"
    if all_frames_deltas is not None and ghost_admission(all_frames_deltas):
        return False, (
            "no improvement on DETECTED keepers, but improved on all frames -- the gain sits on "
            "interpolated (invented) keeper positions. Diagnostic only; the verdict is unchanged."
        )
    return False, "no improvement on detected keepers"
