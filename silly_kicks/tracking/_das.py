"""Dangerous Accessible Space (TF-28) — NATIVE engine facade (ADR-107/108).

The public DAS surface: ``get_das`` / ``get_individual_das`` (team + per-player AS/DAS per frame),
``get_xc`` (expected pass completion), ``estimate_das_cost``, the degradation taxonomy
(``DasUnscoreableError`` / ``DAS_SOURCE_*``, re-exported from ``_das_taxonomy``), and the confined
private ``individual_das_paired`` gkdv consumes.

This module no longer wraps the external ``accessible-space`` package: the physics is reimplemented in
``_das_params`` / ``_das_pack`` / ``_das_engine`` / ``_das_numba``. The ball is identified by the
``is_ball`` mask (no ``player_id = "ball"`` sentinel — ADR-106). Direction comes from the ``GoalMap``
(ADR-055). The shipped quadrature is periodic (ADR-108); the reference quadrature is parity-only and is
refused on the public surface. See NOTICE for the Bischofberger & Baca (2026) attribution.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from silly_kicks.id_compat import canonical_id
from silly_kicks.tracking._das_engine import compute_das, compute_das_paired, compute_xc
from silly_kicks.tracking._das_pack import pack_frames, pack_paired
from silly_kicks.tracking._das_params import DAS_PARAMS, XC_PARAMS, PassSimParams
from silly_kicks.tracking._das_taxonomy import (
    DAS_SOURCE_COMPUTED,
    DAS_SOURCE_TEAM_UNRESOLVED,
    DAS_SOURCE_UNLINKED,
    DAS_SOURCE_UNSCOREABLE_CALL,
    DAS_SOURCE_UNSCOREABLE_FRAME,
    DAS_SOURCE_VALUES,
    DasUnscoreableError,
)
from silly_kicks.tracking._warnings import DasCostWarning

__all__ = [
    "DAS_SOURCE_COMPUTED",
    "DAS_SOURCE_TEAM_UNRESOLVED",
    "DAS_SOURCE_UNLINKED",
    "DAS_SOURCE_UNSCOREABLE_CALL",
    "DAS_SOURCE_UNSCOREABLE_FRAME",
    "DAS_SOURCE_VALUES",
    "DasUnscoreableError",
    "estimate_das_cost",
    "get_das",
    "get_individual_das",
    "get_xc",
]

_DEFAULT_PLAYER_IN_POSSESSION_COL = "ball_carrier_player_id"
_OFFSIDE_WARNED = False

# Advisory cost guardrail (ADR-107/108 §6.14): reported-not-gated; DAS values are unchanged by it.
# Per-engine constants from the ADR-107 chunk table (20 000 frames, best of 3), rounded
# conservatively; re-derived from the corpus performance.json at release.
#: numba serial kernel, seconds per distinct frame (1.17 ms measured, rounded up to 2 significant figures).
_DAS_SECONDS_PER_FRAME = 0.0012
#: numpy engine at the default chunk, seconds per distinct frame (6.06 ms measured, rounded up) --
#: used when numba is absent (numpy has no prange path).
_DAS_SECONDS_PER_FRAME_NUMPY = 0.0061
#: prange scaling efficiency (numba parallel): effective throughput ~= n_threads * this. Measured at
#: 16 threads (0.498, rounded down); < 1 because of the per-frame serial residue.
_PRANGE_EFFICIENCY = 0.49
#: seconds budget for DasCostWarning (replaces the old 5000-frame count = 5000 x 0.02 s = 100 s).
_DAS_COST_WARN_SECONDS = 100.0


def _resolve_params(params: PassSimParams | None, default: PassSimParams) -> PassSimParams:
    p = default if params is None else params
    if p.quadrature == "reference":
        raise ValueError(
            "quadrature='reference' is a parity-only mode and is refused on the public DAS surface; "
            "the shipped quadrature is 'periodic' (ADR-108)."
        )
    return p


def _resolve_player_in_possession_col(frames: pd.DataFrame, player_in_possession_col: str | None) -> str | None:
    """Validate the carrier column: an explicitly-named missing column is a caller error."""
    if player_in_possession_col is None:
        return None
    if player_in_possession_col in frames.columns:
        return player_in_possession_col
    if player_in_possession_col != _DEFAULT_PLAYER_IN_POSSESSION_COL:
        raise ValueError(f"player_in_possession_col={player_in_possession_col!r} not found in frames columns")
    return None


def _warn_no_carrier_once(frames: pd.DataFrame, carrier: str | None) -> None:
    global _OFFSIDE_WARNED
    if carrier is None and not _OFFSIDE_WARNED:
        _OFFSIDE_WARNED = True
        warnings.warn(
            "DAS respect_offside is on but no ball-carrier column was available to exclude the passer "
            "from the offside mask. Pass player_in_possession_col, or run derive_team_in_possession "
            "(which preserves ball_carrier_player_id). Proceeding without passer exclusion.",
            UserWarning,
            stacklevel=3,
        )


def _n_distinct_frames(frames: pd.DataFrame) -> int:
    keys = [c for c in ("game_id", "period_id", "frame_id") if c in frames.columns]
    if not keys:
        return len(frames)
    return int(frames[keys].drop_duplicates().shape[0])


def estimate_das_cost(frames: pd.DataFrame, *, n_threads: int | None = None) -> float:
    """Rough wall-time (seconds) to run DAS over ``frames`` — engine- and thread-aware (§6.14).

    Distinct scored frames x the per-frame constant of the engine that will run: the numpy
    constant when numba is absent (numpy has no ``prange`` path), else the numba serial constant
    divided by the effective thread count (``n_threads > 1`` selects the ``prange`` kernel, scaled by
    ``_PRANGE_EFFICIENCY``; ``None`` / 1 is the serial kernel). A pure, side-effect-free
    order-of-magnitude estimate (advisory, not a guarantee). Used by :func:`get_das` /
    :func:`get_individual_das` to emit a :class:`DasCostWarning` before a large run.

    Examples
    --------
    >>> import pandas as pd
    >>> from silly_kicks.tracking import estimate_das_cost
    >>> estimate_das_cost(pd.DataFrame({"game_id": 1, "period_id": 1, "frame_id": range(100)})) > 0
    True
    """
    from silly_kicks.tracking import _das_engine

    if not _das_engine._numba_available():
        return _n_distinct_frames(frames) * _DAS_SECONDS_PER_FRAME_NUMPY  # numpy has no prange path
    serial = _n_distinct_frames(frames) * _DAS_SECONDS_PER_FRAME
    if n_threads is not None and n_threads > 1:
        return serial / (n_threads * _PRANGE_EFFICIENCY)
    return serial


def _maybe_warn_das_cost(frames: pd.DataFrame, *, warn_cost: bool, n_threads: int | None = None) -> None:
    if not warn_cost:
        return
    est = estimate_das_cost(frames, n_threads=n_threads)
    if est > _DAS_COST_WARN_SECONDS:
        n = _n_distinct_frames(frames)
        warnings.warn(
            f"DAS over {n} distinct frames without sampling ~= {est:.0f} s of per-frame simulation. "
            "Sample frames, or pass warn_cost=False to silence. DAS values are unchanged.",
            DasCostWarning,
            stacklevel=3,
        )


def _frame_key_values(frames: pd.DataFrame) -> list[tuple]:
    cols = frames[["game_id", "period_id", "frame_id"]].to_numpy()
    return [(canonical_id(a), canonical_id(b), canonical_id(c)) for a, b, c in cols]


def _team_maps(packed, res):
    as_map: dict = {}
    das_map: dict = {}
    for krow, a, d in zip(packed.keys.to_numpy(), res.team_as, res.team_das, strict=True):
        k = (canonical_id(krow[0]), canonical_id(krow[1]), canonical_id(krow[2]))
        as_map[k] = a
        das_map[k] = d
    return as_map, das_map


def _run_team(
    frames, *, goal_map, attacking_direction_col, player_in_possession_col, params, chunk_size, n_threads, warn_cost
):
    _maybe_warn_das_cost(frames, warn_cost=warn_cost, n_threads=n_threads)
    p = _resolve_params(params, DAS_PARAMS)
    carrier = _resolve_player_in_possession_col(frames, player_in_possession_col)
    packed = pack_frames(
        frames, goal_map=goal_map, attacking_direction_col=attacking_direction_col, player_in_possession_col=carrier
    )
    if p.respect_offside:
        _warn_no_carrier_once(frames, carrier)
    res = compute_das(packed, p, chunk_size=chunk_size, n_threads=n_threads)
    return packed, res


def get_das(
    frames: pd.DataFrame,
    *,
    goal_map=None,
    attacking_direction_col: str | None = None,
    player_in_possession_col: str | None = _DEFAULT_PLAYER_IN_POSSESSION_COL,
    params: PassSimParams | None = None,
    chunk_size: int | None = None,
    n_threads: int | None = None,
    warn_cost: bool = True,
) -> pd.DataFrame:
    """Team-level Accessible Space and Dangerous Accessible Space per frame.

    Adds ``AS`` and ``DAS`` (the in-possession team's values, broadcast to every row of each frame;
    NaN on a non-scoreable frame). Direction is taken from ``goal_map`` (or built from ``frames``), or
    from ``attacking_direction_col`` (mutually exclusive). See NOTICE for citations.

    Examples
    --------
    Score a linked match's frames::

        from silly_kicks.tracking import get_das
        result = get_das(frames)
    """
    packed, res = _run_team(
        frames,
        goal_map=goal_map,
        attacking_direction_col=attacking_direction_col,
        player_in_possession_col=player_in_possession_col,
        params=params,
        chunk_size=chunk_size,
        n_threads=n_threads,
        warn_cost=warn_cost,
    )
    as_map, das_map = _team_maps(packed, res)
    keys = _frame_key_values(frames)
    out = frames.copy()
    out["AS"] = [as_map.get(k, np.nan) for k in keys]
    out["DAS"] = [das_map.get(k, np.nan) for k in keys]
    return out


def get_individual_das(
    frames: pd.DataFrame,
    *,
    goal_map=None,
    attacking_direction_col: str | None = None,
    player_in_possession_col: str | None = _DEFAULT_PLAYER_IN_POSSESSION_COL,
    params: PassSimParams | None = None,
    chunk_size: int | None = None,
    n_threads: int | None = None,
    warn_cost: bool = True,
) -> pd.DataFrame:
    """Per-player Accessible Space and Dangerous Accessible Space per frame.

    Adds ``AS`` and ``DAS`` per player row (ball rows and non-scoreable frames are NaN). Sum per team
    for a team-level value (the ``add_das`` contract). See NOTICE for citations.

    Examples
    --------
    Per-player DAS decomposition::

        from silly_kicks.tracking import get_individual_das
        result = get_individual_das(frames)
    """
    packed, res = _run_team(
        frames,
        goal_map=goal_map,
        attacking_direction_col=attacking_direction_col,
        player_in_possession_col=player_in_possession_col,
        params=params,
        chunk_size=chunk_size,
        n_threads=n_threads,
        warn_cost=warn_cost,
    )
    out = frames.copy()
    as_col = np.full(len(frames), np.nan)
    das_col = np.full(len(frames), np.nan)
    as_col[packed.p_input_pos] = res.player_as
    das_col[packed.p_input_pos] = res.player_das
    out["AS"] = as_col
    out["DAS"] = das_col
    return out


def get_xc(
    passes: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    params: PassSimParams | None = None,
    chunk_size: int | None = None,
    n_threads: int | None = None,
) -> pd.DataFrame:
    """Expected pass completion (xC) for each pass using tracking context.

    Adds an ``xC`` column (probability in [0,1]); a pass whose frame or team is absent from tracking is
    NaN with one aggregated warning. See NOTICE for citations.

    Examples
    --------
    Compute xC for all passes in a match::

        from silly_kicks.tracking import get_xc
        result = get_xc(pass_actions, frames)
    """
    p = _resolve_params(params, XC_PARAMS)
    xc = compute_xc(passes, frames, params=p, chunk_size=chunk_size, n_threads=n_threads)
    n_missing = int(np.isnan(xc).sum())
    if n_missing:
        warnings.warn(
            f"xC is NaN for {n_missing} pass(es) whose tracking frame or event team was absent "
            "(D-XC-FRAME / D-XC-TEAM); the rest are scored.",
            UserWarning,
            stacklevel=2,
        )
    out = passes.copy()
    out["xC"] = xc
    return out


def individual_das_paired(
    actual: pd.DataFrame,
    counterfactual: pd.DataFrame,
    *,
    goal_map=None,
    attacking_direction_col: str | None = None,
    player_in_possession_col: str | None = _DEFAULT_PLAYER_IN_POSSESSION_COL,
    params: PassSimParams | None = None,
    chunk_size: int | None = None,
    n_threads: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-player AS/DAS for a counterfactual pair, ADR-043-safe (spec 6.6; the confined gkdv seam).

    ``actual`` and ``counterfactual`` must differ ONLY in the moved rows' kinematics (validated by
    :func:`~silly_kicks.tracking._das_pack.pack_paired`). Each returned frame is its input plus per-player
    ``AS``/``DAS``, bit-identical to an independent :func:`get_individual_das` call.

    Examples
    --------
    Score a factual frame and a ghost-keeper counterfactual together (the gkdv Delta-DAS arm), then
    difference the attacking team's DAS::

        # `ghost` is `actual` with only the defending keeper's coordinates moved
        actual_scored, ghost_scored = individual_das_paired(actual, ghost, goal_map=goal_map)
        attackers = ~actual_scored["is_ball"] & (actual_scored["team_id"] == attacking_team_id)
        delta = actual_scored.loc[attackers, "DAS"].sum() - ghost_scored.loc[attackers, "DAS"].sum()
    """
    p = _resolve_params(params, DAS_PARAMS)
    carrier = _resolve_player_in_possession_col(actual, player_in_possession_col)
    pa, pc, moved = pack_paired(
        actual,
        counterfactual,
        goal_map=goal_map,
        attacking_direction_col=attacking_direction_col,
        player_in_possession_col=carrier,
    )
    ra, rc = compute_das_paired(pa, pc, moved, p, chunk_size=chunk_size, n_threads=n_threads)
    return _attach_player(actual, pa, ra), _attach_player(counterfactual, pc, rc)


def _attach_player(frames, packed, res):
    out = frames.copy()
    as_col = np.full(len(frames), np.nan)
    das_col = np.full(len(frames), np.nan)
    as_col[packed.p_input_pos] = res.player_as
    das_col[packed.p_input_pos] = res.player_das
    out["AS"] = as_col
    out["DAS"] = das_col
    return out
