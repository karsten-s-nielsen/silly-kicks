"""Provider visibility taxonomy + the detection-aware visibility-usability rule.

Neutral home (the 4.53.0 ``_id_compat`` precedent, ``docs/PRIVATE_CONSUMERS.md``) for provider-DATA
facts that more than one consumer must obey: the ghost model (keeper detection,
:func:`silly_kicks.tracking._ghost_gk.keeper_detection_mask`), the tc3 materializer (a build-time
guard), and the ghost trainer (a consume-time pre-flight). Kept here rather than in ``_ghost_gk.py``
because "which providers carry a per-player detection flag" is a property of the provider's data, not
of the ghost model -- a general tc3 builder importing from a ghost-private module is the layering
smell this move removes. See ADR (detection-aware visibility guardrails) + the design spec.
"""

from __future__ import annotations

import pandas as pd

# Which providers' feeds carry a per-player detection flag (spec 4.3).
# A null `visibility` is AMBIGUOUS: for a fully-observed provider it means "no flag exists and
# none is needed"; for a detection-aware provider it means "the pipeline DISCARDED the flag"
# (the kloppy gateway hard-codes visibility=None). Reading the second as the first would train
# ghost-GK on interpolator output -- ~80% of SkillCorner keeper positions are extrapolated.
_DETECTION_AWARE_PROVIDERS = frozenset({"skillcorner"})
# metrica is full optical tracking (all players every frame, NO detection flag) -- fully observed
# like the native providers. Classifying it here keeps ghost-GK trainable on metrica (a pre-PR
# capability the always-run detected-only filter would otherwise crash); metrica's exclusion from
# the registered GKDV corpora is a separate corpus-composition decision (Tier-2 data quality).
_FULLY_OBSERVED_PROVIDERS = frozenset({"gradientsports", "sportec", "idsse", "metrica"})


def validate_provider(provider: str) -> None:
    """Raise unless ``provider`` is classified as detection-aware or fully observed.

    Single source for the membership rule: :func:`silly_kicks.tracking._ghost_gk.keeper_detection_mask`,
    the tc3 materializer build-time guard, and the ghost trainer's startup check all call this. Two
    copies of the set would drift the moment a provider is added -- and the failure mode of that drift
    is silent, because an unclassified provider only surfaces deep inside a training run, after the
    expensive extraction.

    Raises ``ValueError`` naming the two sets and their current members.

    Examples
    --------
    >>> validate_provider("gradientsports")
    """
    known = _DETECTION_AWARE_PROVIDERS | _FULLY_OBSERVED_PROVIDERS
    if provider not in known:
        raise ValueError(
            f"unclassified provider {provider!r}: add it to _DETECTION_AWARE_PROVIDERS or "
            f"_FULLY_OBSERVED_PROVIDERS -- an unknown provider is NOT assumed observed. "
            f"Known: {sorted(known)}"
        )


def _detection_discarded_message(provider: str) -> str:
    """The remedy message shared by every layer that catches a discarded detection flag.

    Single-sourced so the ghost mask, the materializer guard, and the trainer pre-flight name the
    SAME fix (rebuild via ``tracking.skillcorner``) -- deliberately provider-generic (no
    ``keeper_detection_mask:`` prefix), because it is now surfaced from three call sites where that
    prefix would misdescribe where the failure was caught.
    """
    return (
        f"provider {provider!r} carries a detection flag, but `visibility` is entirely null -- "
        "the pipeline discarded it (the kloppy gateway hard-codes visibility=None). Build these "
        "frames with tracking.skillcorner instead; training on undetected keepers means training "
        "on the interpolator (spec 4.3)."
    )


def assert_detection_aware_visibility(visibility: pd.Series, *, provider: str) -> None:
    """Raise iff ``provider`` is detection-aware and its ``visibility`` is entirely null.

    An empty series raises too (``pd.Series([]).isna().all()`` is ``True``), matching the original
    :func:`keeper_detection_mask` semantics exactly -- no ``len()`` guard. This is the shared rule;
    the returned MASK and the empty->raise trigger are preserved from the original, only the message
    is unified (see :func:`_detection_discarded_message`). Provider classification is the caller's
    responsibility (``keeper_detection_mask`` and the trainer pre-flight both run
    :func:`validate_provider` on their own), so an unknown provider is a no-op here, not a raise.
    """
    if provider in _DETECTION_AWARE_PROVIDERS and visibility.isna().all():
        raise ValueError(_detection_discarded_message(provider))
