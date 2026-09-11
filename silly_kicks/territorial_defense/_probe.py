"""Construct-validity probe battery for territorial defense (TF-54b, spec §9).

Owner-run, reported-not-gated. Self-contained: it ADAPTS the gkdv / TF-19 probe pattern
(``gkdv/_probe.py``) rather than importing it, with its OWN uniquely-named ``TD_PROBE_RATIO`` (NOT
``PHYSICS_ARM_PROBE_RATIO`` / ``TF19_PROBE_RATIO`` / ``XS_PROBE_RATIO``, all model-specific).

**The verdicts are POOLED-corpus statistics computed in a REDUCE over ALL shards, NEVER per shard**
(a per-shard impl reads a thin per-match domain as ``arm_unscoreable``). Direction is POSITIVE
(attacker-value units: ``positive = threat suppressed``), inverted from gkdv's "negative = deterrent".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- registered constants (all NEW; not shared with sibling probes) ---------------------------
MIN_DOMAIN_FRAMES: int = 200
SATURATING_MULTIPLE: float = 5.0
TD_PROBE_RATIO: float = 2.0

LAYER0_VERDICTS: tuple[str, str, str] = ("instrument_valid", "instrument_void", "arm_unscoreable")
LAYER1_VERDICTS: tuple[str, str, str] = ("responsive", "not_responsive", "arm_unscoreable")

#: Attacker-value units -> a real defender's suppression is POSITIVE (inverted from gkdv).
EXPECTED_DIRECTION: dict[str, str] = {"a_threat_suppressed": "positive", "b_threat_suppressed": "positive"}


def expected_direction_for_arm(arm_column: str) -> str:
    """The expected sign for an arm's output column (``"positive"`` == threat suppressed).

    An arm column absent from :data:`EXPECTED_DIRECTION` raises ``KeyError`` (never a silent skip).

    Examples
    --------
    >>> from silly_kicks.territorial_defense._probe import expected_direction_for_arm
    >>> expected_direction_for_arm("a_threat_suppressed")
    'positive'
    """
    return EXPECTED_DIRECTION[arm_column]


def layer0_instrument_verdict(*, realistic_abs, saturating_abs, placebo_p95, n_domain) -> str:
    """Instrument-validity verdict over the POOLED corpus (mirrors gkdv ``layer0``).

    ``arm_unscoreable`` first (domain < MIN_DOMAIN_FRAMES, or any non-finite pooled stat); else void
    iff the saturating median clears NEITHER ``SATURATING_MULTIPLE x`` the realistic median NOR the
    placebo p95. The ``> 0`` guard on the multiple leg is load-bearing (a zero-dominated arm would
    vacuously pass ``5 * 0``; the placebo leg is the backstop).
    """
    real = np.asarray(realistic_abs, dtype=float)
    sat = np.asarray(saturating_abs, dtype=float)
    if n_domain < MIN_DOMAIN_FRAMES:
        return "arm_unscoreable"
    real_med = float(np.nanmedian(real)) if real.size and bool(np.isfinite(real).any()) else float("nan")
    sat_med = float(np.nanmedian(sat)) if sat.size and bool(np.isfinite(sat).any()) else float("nan")
    if not (np.isfinite(real_med) and np.isfinite(sat_med) and np.isfinite(float(placebo_p95))):
        return "arm_unscoreable"
    passes_multiple = real_med > 0.0 and sat_med >= SATURATING_MULTIPLE * real_med
    passes_placebo = sat_med > float(placebo_p95)
    return "instrument_void" if (not passes_multiple and not passes_placebo) else "instrument_valid"


def layer1_responsiveness_verdict(*, defender_med, nd_med, placebo_p95, n_domain) -> str:
    """Responsiveness verdict over the POOLED corpus (mirrors gkdv ``layer1``, ratio = TD_PROBE_RATIO).

    ``responsive`` iff the dosed-defender median clears ``TD_PROBE_RATIO x max(nearest-control median,
    placebo p95)``. No absolute floor (comparable-not-decisive).
    """
    if n_domain < MIN_DOMAIN_FRAMES:
        return "arm_unscoreable"
    if not (np.isfinite(float(defender_med)) and np.isfinite(float(nd_med)) and np.isfinite(float(placebo_p95))):
        return "arm_unscoreable"
    thresh = TD_PROBE_RATIO * max(float(nd_med), float(placebo_p95))
    return "responsive" if float(defender_med) >= thresh else "not_responsive"


def impose_defender_dose(frame: pd.DataFrame, *, defender_pos: int, dx: float, dy: float) -> pd.DataFrame:
    """Return a copy of ``frame`` with the row at ``defender_pos`` displaced by ``(dx, dy)``. PURE."""
    out = frame.copy()
    x = out["x"].to_numpy(dtype=float, copy=True)
    y = out["y"].to_numpy(dtype=float, copy=True)
    x[defender_pos] += float(dx)
    y[defender_pos] += float(dy)
    out["x"] = x
    out["y"] = y
    return out


def paired_vector_controls(
    frame: pd.DataFrame,
    *,
    defender_pos: int,
    defending_team_id,
    dx: float,
    dy: float,
    r: int,
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    """One OTHER defending outfielder displaced by the SAME ``(dx, dy)`` vector the dosed defender
    moved: the NEAREST to the dosed defender (``"nearest"``) + ``r`` random single-player placebos
    (``"placebo_k"``). Each value is a NEW frame (never mutates the input). Mirrors the gkdv idiom.
    """
    from silly_kicks.id_compat import ids_match

    is_ball = frame["is_ball"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    on_def = ids_match(frame["team_id"], defending_team_id).to_numpy(dtype=bool)
    cand = np.flatnonzero(on_def & ~is_ball & (np.arange(len(frame)) != defender_pos))
    out: dict[str, pd.DataFrame] = {}
    if cand.size == 0:
        return out
    xs = frame["x"].to_numpy(dtype=float)
    ys = frame["y"].to_numpy(dtype=float)
    d2 = (xs[cand] - xs[defender_pos]) ** 2 + (ys[cand] - ys[defender_pos]) ** 2
    nearest = int(cand[int(np.argmin(d2))])
    out["nearest"] = impose_defender_dose(frame, defender_pos=nearest, dx=dx, dy=dy)
    for k in range(r):
        pick = int(rng.choice(cand))
        out[f"placebo_{k}"] = impose_defender_dose(frame, defender_pos=pick, dx=dx, dy=dy)
    return out
