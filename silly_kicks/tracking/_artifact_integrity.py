"""Trained-artifact integrity predicates (ADR-126). Pure; no I/O.

The base-rate band is DERIVED physically per artifact (``n_events * (fps * horizon) / n_rows``
is the maximum possible per-row positive rate for an occurrence label), never a fixed constant:
the committed xShot defect sat at a 22% positive rate that cleared ``pr_auc > base_rate`` but is
13x the physical maximum. ``tol`` is the only slack.
"""

from __future__ import annotations

#: Per-family slack over the physical maximum positive rate (the only tunable). Justified in ADR-126:
#: 1.5x leaves headroom for the share of pre-event frames that fall inside the domain filter while
#: still rejecting a >=1.5x inflation.
MODEL_INTEGRITY_TOL: dict[str, float] = {"xshot_occurrence": 1.5, "xcross_attempt": 1.5}


def physical_upper_bound(n_events: float, horizon_s: float, fps: float, n_rows: float, *, tol: float = 1.5) -> float:
    """Maximum plausible per-row positive rate for an occurrence label, with ``tol`` slack.

    Physical maximum = ``n_events * (fps * horizon_s) / n_rows`` (the most positive frames an
    event set can flag, over the extracted rows). For a mixed-fps corpus pass
    ``fps = n_rows / total_duration_s`` so the bound reads ``tol * n_events * horizon_s /
    total_duration_s`` (fps-independent). Returns ``inf`` for a degenerate empty corpus.
    """
    if n_rows <= 0:
        return float("inf")
    return tol * n_events * horizon_s * fps / n_rows


def base_rate_plausible(
    positive_rate: float,
    *,
    n_events: float,
    horizon_s: float,
    fps: float,
    n_rows: float,
    lo: float = 0.0005,
    tol: float = 1.5,
) -> bool:
    """True iff ``lo <= positive_rate <= physical_upper_bound(...)`` (NaN-safe: NaN -> False)."""
    hi = physical_upper_bound(n_events, horizon_s, fps, n_rows, tol=tol)
    return bool(positive_rate == positive_rate and lo <= positive_rate <= hi)
