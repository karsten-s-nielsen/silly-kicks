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

import numpy as np
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

# TF-58 D20: which providers' native feeds carry a RELIABLE dead-ball signal (`ball_state`). skillcorner and
# metrica hard-code ball_state="alive" (tracking/skillcorner.py, tracking/metrica.py), so a stoppage split
# reading only `ball_state` would silently do nothing on them -- they are unobserved BY CONSTRUCTION.
_DEAD_BALL_OBSERVED_PROVIDERS = frozenset({"sportec", "idsse", "gradientsports"})


def dead_ball_observed(provider: str) -> bool:
    """True iff ``provider``'s native feed carries a reliable dead-ball (`ball_state`) signal (TF-58 D20).

    Calls :func:`validate_provider` first, so an unclassified provider RAISES rather than being assumed
    observed (fail-closed). ``"snapshot"`` (freeze frames) is unclassified and therefore raises.

    Examples
    --------
    >>> dead_ball_observed("sportec")
    True
    >>> dead_ball_observed("skillcorner")
    False
    """
    validate_provider(provider)
    return provider in _DEAD_BALL_OBSERVED_PROVIDERS


def classified_provider(provider) -> str | None:
    """``provider`` when it is classified (detection-aware or fully observed), else ``None`` -- for a caller that
    must DECLARE how a frame's visibility reads but may hold an unclassified source (e.g. SB360 freeze frames):
    ``None`` makes the visibility-based outputs fail closed instead of raising.

    Examples
    --------
    >>> classified_provider("idsse"), classified_provider("statsbomb")
    ('idsse', None)
    """
    p = None if provider is None else str(provider)
    return p if p in _DETECTION_AWARE_PROVIDERS | _FULLY_OBSERVED_PROVIDERS else None


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


def detected_mask(visibility: pd.Series, *, provider: str, assume_observed: bool = False) -> np.ndarray:
    """Per-row boolean mask: which rows' player was ACTUALLY DETECTED. Fail-closed (ADR-109).

    The general, player-agnostic form of
    :func:`silly_kicks.tracking._ghost_gk.keeper_detection_mask` (which now delegates here) -- the
    caller passes the ``visibility`` of whatever rows it selected (keeper, outfield, ...). The shared
    seam for every detection consumer: ghost-GK training (via the wrapper), and, in their own cycles,
    TF-58's outfield collective variables and the GK detection-gate.

    Order (ADR-109; matches the historical ``keeper_detection_mask``):

    - :func:`validate_provider` runs FIRST -- **always**, even under ``assume_observed``. A typo'd or
      unclassified provider raises here; the opt-out bypasses only the detection verdict, never the
      provider taxonomy.
    - ``assume_observed=True`` -> all-``True`` (the general opt-out; logged/flagged by the caller,
      never the default, never keyed to a specific consumer).
    - ``provider`` in :data:`_FULLY_OBSERVED_PROVIDERS` -> all-``True`` (no detection flag exists).
    - ``provider`` in :data:`_DETECTION_AWARE_PROVIDERS` -> the entirely-null trap raises
      (:func:`assert_detection_aware_visibility`), then ``visibility.fillna(False)`` (detected
      ``True``; extrapolated / per-row-null ``False``).

    PUBLIC (exported via ``silly_kicks.tracking.__all__``): the shared detection primitive for every
    consumer -- the intra-``tracking`` :func:`keeper_detection_mask`, the TF-64 keeper-row wrappers,
    and cross-package vectorized consumers (gkdv's detected-keeper presence set). Exported rather than
    kept private so cross-package callers import a public seam (the GK detection-gate S3 decision)
    instead of a ``docs/PRIVATE_CONSUMERS.md`` allowlist row on a single-underscore module.
    """
    validate_provider(provider)
    if assume_observed:
        return np.ones(len(visibility), dtype=bool)
    if provider in _FULLY_OBSERVED_PROVIDERS:
        return np.ones(len(visibility), dtype=bool)
    assert_detection_aware_visibility(visibility, provider=provider)
    return visibility.fillna(False).astype(bool).to_numpy()


# --- TF-64 GK detection-gate: public keeper-row wrappers over detected_mask (ADR-109) -----------
# The keeper-metric wiring ADR-109 explicitly deferred ("the ~14 GK outputs ... a separate, later
# change"). These two wrappers are that seam: a keeper-position consumer passes the frame it already
# holds, and the wrapper returns the defending keeper's detected position (or NaN) / a per-keeper
# detection verdict, so an undetected SkillCorner keeper goes honest-NaN/drop instead of being read
# as an ~80%-extrapolated position (anti-pattern E7).


def detection_aware_provider(frames: pd.DataFrame) -> str | None:
    """The detection-aware provider of a frame slice, or ``None`` when the slice is not gated.

    The keeper detection-gate ACTIVATES only for a provider in :data:`_DETECTION_AWARE_PROVIDERS`
    (currently ``skillcorner``). Every other frame set -- fully observed, SB360 ``snapshot``, or a
    ``source_provider`` this taxonomy does not classify (synthetic test fixtures, a not-yet-wired
    feed) -- is read AS-OBSERVED, byte-identical to the pre-gate behaviour (spec §2 scope = SkillCorner;
    test 5 "fully-observed unaffected"). Returns the detection-aware provider name present in the
    slice (so a caller can hand it to :func:`detected_mask`), else ``None``.

    Deliberately NON-raising on an unclassified ``source_provider`` (unlike :func:`validate_provider`):
    a keeper-metric consumer must not crash on an arbitrary provider, and the gate only engages for
    the one detection-aware feed. A detection-aware provider ALWAYS stamps ``source_provider`` (the
    native builder / materializer contract), so a real SkillCorner slice is never silently un-gated;
    the discarded-flag build error stays fail-loud at the materializer guard and inside
    :func:`detected_mask` (the entirely-null trap), which the gate reaches only for ``skillcorner``.
    """
    if "source_provider" not in frames.columns:
        return None
    provs = {str(p).lower() for p in pd.unique(frames["source_provider"].dropna())}
    hit = sorted(provs & _DETECTION_AWARE_PROVIDERS)
    return hit[0] if hit else None


def _keeper_rows_verdict(
    keeper_rows: pd.DataFrame, *, provider: str | None, assume_keeper_observed: bool
) -> bool | None:
    """Shared verdict for the defending-keeper candidate rows of ONE frame (spec §3.1).

    ``keeper_rows`` is the already-filtered keeper candidate set (the caller selects ``is_goalkeeper &
    ~is_ball``, optionally for one team). ``provider`` is the detection-aware provider name (from
    :func:`detection_aware_provider`) or ``None`` (the slice is not gated). Returns ``True`` (detected
    / observed), ``False`` (detection-aware, keeper present but NOT detected -> extrapolated), or
    ``None`` (keeper row absent or not exactly one -> not-observed; **never** read as observed).

    - ``provider is None`` (not detection-aware) or ``assume_keeper_observed=True`` (the general
      opt-out; logged by the caller, never default) -> ``True`` iff exactly one keeper row, else
      ``None`` (no detection flag to consult).
    - detection-aware: exactly one keeper row required (else ``None``); the verdict is
      :func:`detected_mask` over that row's ``visibility`` (a present-but-entirely-null flag raises,
      surfacing a discarded detection flag; ADR-109).
    """
    if provider is None or assume_keeper_observed:
        return True if len(keeper_rows) == 1 else None
    if len(keeper_rows) != 1:
        return None
    return bool(detected_mask(keeper_rows["visibility"], provider=provider)[0])


def keeper_detected(frame_rows: pd.DataFrame, gk_id: object, *, assume_keeper_observed: bool = False) -> bool | None:
    """Was keeper ``gk_id`` actually DETECTED in ``frame_rows``? (spec §3.1, ADR-109).

    ``frame_rows`` is the frame (all players, carrying ``source_provider``); the keeper row is
    selected by ``gk_id`` (dtype-safe via :func:`silly_kicks.id_compat.ids_match`, never raw ``==``)
    and ``~is_ball``, the provider derived via :func:`detection_aware_provider`. Returns ``True`` /
    ``False`` / ``None``. **Callers must map BOTH ``False`` and ``None`` to honest-NaN/drop** --
    treating ``None`` ("not observed") as observed is the fail-OPEN E7 hole this forbids.
    """
    from silly_kicks.id_compat import ids_match

    provider = detection_aware_provider(frame_rows)
    kr = frame_rows[~frame_rows["is_ball"].astype(bool) & ids_match(frame_rows["player_id"], gk_id).to_numpy()]
    return _keeper_rows_verdict(kr, provider=provider, assume_keeper_observed=assume_keeper_observed)


def resolve_detected_keeper_xy(
    frame_rows: pd.DataFrame, *, assume_keeper_observed: bool = False
) -> tuple[float, float]:
    """The DETECTED defending-keeper position in ``frame_rows``, or ``(nan, nan)`` (spec §3.1).

    ``frame_rows`` is the keeper candidate set for one team (caller pre-filters ``is_goalkeeper``, as
    the existing ``_gk_x``/``defending_gk_rows`` sites already do; ``~is_ball`` is enforced here). The
    provider is derived from the slice via :func:`detection_aware_provider`. Returns the single
    detected keeper's ``(x, y)``; returns ``(nan, nan)`` when the keeper is absent, ambiguous (not
    exactly one), or present-but-undetected on a detection-aware provider -- so the dependent metric
    honest-NaNs rather than consuming an extrapolated position.
    """
    provider = detection_aware_provider(frame_rows)
    kr = frame_rows[frame_rows["is_goalkeeper"].astype(bool) & ~frame_rows["is_ball"].astype(bool)]
    if _keeper_rows_verdict(kr, provider=provider, assume_keeper_observed=assume_keeper_observed) is not True:
        return (float("nan"), float("nan"))
    row = kr.iloc[0]
    return (float(row["x"]), float(row["y"]))
