"""xT-GK v2 metric assembler (ADR-036 §Part 4). Depends ONLY on the three ports.

xT-GK = rho*dV_position + rho*dV_pressure(=PEV) - (1-rho)*V(s) - (1-rho)*kappa*V_opp
The four terms sum to the metric exactly. Columns namespaced xt_gk_v2_* (v1's xt_gk_* are frozen).
PEV is 0 by construction when p'=p (base metric); it lights up only with receiver-pressure q.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.xtgk._possession_value import State, flat_zones
from silly_kicks.xtgk._pressure_levels import PressureLevels
from silly_kicks.xthreat._grid import M, N

_OUTPUT_COLS = [
    "xt_gk_v2_position",
    "xt_gk_v2_pev",
    "xt_gk_v2_retention_loss",
    "xt_gk_v2_dzv",
    "xt_gk_v2",
]


def compute_xt_gk_v2(
    actions: pd.DataFrame,
    *,
    possession_value,
    retention,
    turnover_cost,
    kappa: float = 1.0,
    pressure_column: str = "pressure",
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
    zones_o = flat_zones(actions["start_x"], actions["start_y"], l, w)
    zones_d = flat_zones(actions["end_x"], actions["end_y"], l, w)
    zones_arg = zones_o if pl.mode == "zone_conditional" else None
    levels = pl.apply(actions[pressure_column], zones=zones_arg)  # p' = p (base metric)

    rho = np.asarray(retention.predict_proba(retention_features), dtype=float)

    # NOTE (scale): the per-action Python loop below calling delta_v/value is correctness-first and
    # fine for the GK-distribution slice (a small fraction of actions); a batch path (vectorized grid
    # lookups over the pressure-stratified surfaces) is a follow-up if the lakehouse needs full-stream.
    n = len(actions)
    position = np.zeros(n)
    pev = np.zeros(n)
    ret_loss = np.zeros(n)
    dzv = np.zeros(n)
    for i in range(n):
        p = int(levels[i])
        s = State(int(zones_o[i]), p)  # type: ignore[arg-type]
        s_next = State(int(zones_d[i]), p)  # type: ignore[arg-type]
        dv = possession_value.delta_v(s, s_next)
        v_s = float(possession_value.value(int(zones_o[i]), p))
        v_opp = float(turnover_cost.value(int(zones_o[i]), p))
        position[i] = rho[i] * dv.position_component
        pev[i] = rho[i] * dv.pressure_component
        ret_loss[i] = -(1.0 - rho[i]) * v_s
        dzv[i] = -(1.0 - rho[i]) * kappa * v_opp
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
