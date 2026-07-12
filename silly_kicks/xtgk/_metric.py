"""xT-GK v2 metric assembler (ADR-036 §Part 4). Depends ONLY on the three ports.

xT-GK = rho*dV_position + rho*dV_pressure(=PEV) - (1-rho)*V(s) - (1-rho)*kappa*V_opp
The four terms sum to the metric exactly. Columns namespaced xt_gk_v2_* (v1's xt_gk_* are frozen).
PEV is 0 by construction when p'=p (base metric); it lights up only with receiver-pressure q.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from silly_kicks.xtgk._possession_value import State, finite_coord_mask, flat_zones
from silly_kicks.xtgk._pressure_levels import PressureLevels
from silly_kicks.xthreat._grid import M, N

_OUTPUT_COLS = [
    "xt_gk_v2_position",
    "xt_gk_v2_pev",
    "xt_gk_v2_retention_loss",
    "xt_gk_v2_dzv",
    "xt_gk_v2",
]


def _check_coordinate_coherence(actions: pd.DataFrame, retention_features: pd.DataFrame) -> None:
    """``actions`` and ``retention_features`` MUST describe the SAME coordinates (ADR-036 amendment).

    Compares COORDINATES, not provenance: recomputes the coordinate-derived retention features from
    ``actions`` and compares. Catches, symmetrically and with no case table:

    * resolved ``actions`` + rho-features built from the RAW frame (F1);
    * RAW ``actions`` + rho-features built from the resolved frame (F1's mirror);
    * two frames resolved against different mart vintages (equal stamps, different coordinates).

    A stamp-equality check would miss the third and would need a 4-row case table for the first two.
    """
    from silly_kicks.xtgk._retention_features import COORD_DERIVED_NAMES, _coord_derived

    if not set(COORD_DERIVED_NAMES).issubset(retention_features.columns):
        return  # not a retention-feature frame (e.g. a caller stub) -- nothing to attest
    expected = _coord_derived(actions)
    for col in COORD_DERIVED_NAMES:
        got = pd.to_numeric(retention_features[col], errors="coerce").to_numpy(dtype=float)
        exp = expected[col].to_numpy(dtype=float)
        if got.shape != exp.shape or not np.allclose(got, exp, atol=1e-6, rtol=0.0, equal_nan=True):
            raise ValueError(
                f"compute_xt_gk_v2: retention_features[{col!r}] does not match the coordinates in "
                "`actions`. The rho features were built from a DIFFERENT frame -- typically one "
                "side went through apply_resolved_gk_geometry and the other did not, so the grid "
                "zones and rho would disagree. Build retention_features from the SAME (resolved) "
                "frame you pass as `actions`. See ADR-036."
            )


def _warn_if_unattested(actions: pd.DataFrame, domain_column: str) -> None:
    """Warn when a GK-distribution domain is present but resolution was never attested.

    This is the one thing coordinates can never reveal: raw coordinates are perfectly
    self-consistent, so no numeric check can tell you that resolution was never *attempted*.
    """
    from silly_kicks.xtgk._resolved_geometry import GK_GEOMETRY_SOURCE_COLUMN

    if domain_column not in actions.columns:
        return
    domain = actions[domain_column].fillna(False).to_numpy(dtype=bool)
    if not domain.any():
        return
    n_unattested = int(domain.sum())
    if GK_GEOMETRY_SOURCE_COLUMN in actions.columns:
        stamps = actions.loc[domain, GK_GEOMETRY_SOURCE_COLUMN].to_numpy()
        # ANY unattested row, not ALL: a concatenated mixed-vintage frame (realistic for the
        # lakehouse) would otherwise score its unattested rows on RAW origins in silence.
        n_unattested = int((stamps == "unattested").sum())
        if n_unattested == 0:
            return
    warnings.warn(
        f"compute_xt_gk_v2: {n_unattested} of {int(domain.sum())} {domain_column} rows carry no "
        "attested resolved geometry. Their origins are RAW -- Gradient Sports goal-kicks are ~60% "
        "NaN and SkillCorner's are the broadcast BALL, not the keeper. Route actions through "
        "apply_resolved_gk_geometry first. See ADR-036.",
        stacklevel=2,
    )


def compute_xt_gk_v2(
    actions: pd.DataFrame,
    *,
    possession_value,
    retention,
    turnover_cost,
    kappa: float = 1.0,
    pressure_column: str = "pressure",
    domain_column: str = "is_gk_distribution",
    pressure_levels: PressureLevels | None = None,
    retention_features: pd.DataFrame | None = None,
    l: int = N,
    w: int = M,
) -> pd.DataFrame:
    """Per action, the v2 metric + four-term decomposition. Ports are injected (swappable).

    The metric's terciles MUST match the surfaces V was fit on: pass ``pressure_levels=`` (or a
    ``possession_value`` exposing ``.pressure_levels``) -- never refit a fresh one.
    ``retention_features`` must be built the same way rho was trained -- from the gold action marts
    via ``extract_retention_features`` (geometry + pressure; tracking-frames deprecated)."""
    pl = pressure_levels if pressure_levels is not None else getattr(possession_value, "pressure_levels", None)
    if pl is None:
        raise ValueError(
            "compute_xt_gk_v2 needs pressure_levels= (or a possession_value exposing .pressure_levels) "
            "so the metric's terciles match the surfaces V was fit on -- never refit."
        )
    if retention_features is None:
        raise ValueError(
            "compute_xt_gk_v2 needs retention_features= built the same way rho was trained "
            "(extract_retention_features over the gold action marts); never silently defaulted."
        )
    # Warn BEFORE raising, so a caller who made both mistakes still sees the actionable warning.
    _warn_if_unattested(actions, domain_column)
    _check_coordinate_coherence(actions, retention_features)
    # ADR-036 amendment: NEVER fabricate a grid zone from a NaN coordinate. flat_zones maps NaN ->
    # (0,0) -> zone 176 (the own-corner cell); that is safe for the FIT seams (they all dropna) but
    # here it would emit a real number at a location the action never had.
    finite = finite_coord_mask(actions)
    n = len(actions)
    n_bad = int((~finite).sum())
    if n_bad:
        warnings.warn(
            f"compute_xt_gk_v2: {n_bad} of {n} actions have non-finite coordinates; their "
            "xt_gk_v2_* are emitted as NaN (never fabricated to a grid zone). For the "
            "GK-distribution domain, route actions through apply_resolved_gk_geometry first -- "
            "unresolved rows are honest NaN, not zone 176. See ADR-036.",
            stacklevel=2,
        )

    zones_o = flat_zones(actions["start_x"], actions["start_y"], l, w)
    zones_d = flat_zones(actions["end_x"], actions["end_y"], l, w)
    zones_arg = zones_o if pl.mode == "zone_conditional" else None
    levels = pl.apply(actions[pressure_column], zones=zones_arg)  # p' = p (base metric)

    # NOTE (scale): the per-action Python loop below calling delta_v/value is correctness-first and
    # fine for the GK-distribution slice (a small fraction of actions); a batch path (vectorized grid
    # lookups over the pressure-stratified surfaces) is a follow-up if the lakehouse needs full-stream.
    position = np.full(n, np.nan)
    pev = np.full(n, np.nan)
    ret_loss = np.full(n, np.nan)
    dzv = np.full(n, np.nan)

    idx = np.flatnonzero(finite)
    if len(idx):
        # rho is scored ONLY on finite rows -- a NaN-coord row would otherwise be silently
        # mean-imputed by GkRetentionModel.predict_proba (_retention.py:81) and would then
        # multiply every term of the metric with a no-information value.
        rho = np.asarray(retention.predict_proba(retention_features.iloc[idx]), dtype=float)
        for k, i in enumerate(idx):
            p = int(levels[i])
            s = State(int(zones_o[i]), p)  # type: ignore[arg-type]
            s_next = State(int(zones_d[i]), p)  # type: ignore[arg-type]
            dv = possession_value.delta_v(s, s_next)
            v_s = float(possession_value.value(int(zones_o[i]), p))
            v_opp = float(turnover_cost.value(int(zones_o[i]), p))
            position[i] = rho[k] * dv.position_component
            pev[i] = rho[k] * dv.pressure_component
            ret_loss[i] = -(1.0 - rho[k]) * v_s
            dzv[i] = -(1.0 - rho[k]) * kappa * v_opp
    total = position + pev + ret_loss + dzv
    return pd.DataFrame(
        {
            "xt_gk_v2_position": position,
            "xt_gk_v2_pev": pev,
            "xt_gk_v2_retention_loss": ret_loss,
            "xt_gk_v2_dzv": dzv,
            "xt_gk_v2": total,
        },
        index=actions.index,
    )
