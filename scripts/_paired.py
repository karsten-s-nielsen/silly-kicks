"""The registered decision rules (spec 4.1 and 4.3). Pure; table-tested; no I/O.

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


def fixed_sequence_ship(
    *, sc_extended: Sequence[float], full: Sequence[float], full_vs_sc: Sequence[float]
) -> tuple[str, str]:
    """Fixed-sequence selection (spec 4.1). Order is pre-registered; stop at the first failure.

    Testing two shipping candidates independently would roughly double the noise-win rate (a
    single candidate clears the sign rule ~19% of the time under a symmetric null). A fixed
    sequence holds the error rate at the single-test level with no alpha correction.

    Registered cost: if `sc_extended` fails, `full` CANNOT ship on this registration, even if its
    own deltas clear. That outcome is recorded as a finding that triggers a NEW registration.
    """
    if not clears_rule(sc_extended):
        return "public", "sc_extended failed the rule; the sequence stops (full cannot ship here)"
    if clears_rule(full) and clears_rule(full_vs_sc):
        return "full", "full clears vs public AND dominates sc_extended fold-by-fold"
    return "sc_extended", "sc_extended clears; full does not dominate it -- ties go to less data"


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
