"""Pre-registered deep-zone go/no-go gate (ADR-036 §8, BLOCKING).

Numbers (effect_floor, n_min, min_occupied_cells, direction) are LOCKED by owner/Eyestone
before fitting (Q4); GateConfig carries them so the STRUCTURE is testable on synthetic data now.

Occupied-cell semantics: a keeper populates only a HANDFUL of deep cells, so the gate operates
on deep cells with >= n_min support in ALL THREE terciles (the effect check reads level 2 as
well, so every averaged tercile must be supported); it requires at least min_occupied_cells such
cells (else STOP — the gate cannot run) and computes effect / monotonicity over ONLY those cells.
Direction is configurable (Q2 still open) and always reported. Cross-check agreement is graded on
BUILD-UP cells, not deep cells.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from silly_kicks.xtgk._empirical import EmpiricalPossessionValue
from silly_kicks.xtgk._markov import MarkovPossessionValue

_L, _W = 16, 12
DEEP_ZONE_CELLS: tuple[int, ...] = tuple((_W - 1 - yj) * _L + xi for yj in range(_W) for xi in (0, 1))
BUILD_UP_CELLS: tuple[int, ...] = tuple((_W - 1 - yj) * _L + xi for yj in range(_W) for xi in range(2, 7))

Direction = Literal["decreasing", "increasing", "either"]


@dataclass(frozen=True)
class GateConfig:
    effect_floor: float
    n_min: int
    min_occupied_cells: int = 2
    crosscheck_rel_tol: float = 0.5
    expected_direction: Direction = "either"


@dataclass(frozen=True)
class DeepZoneGateReport:
    passed: bool
    effect_size: float
    observed_direction: str
    monotone_ok: bool
    crosscheck_agrees: bool
    n_occupied_cells: int
    stop_reason: str


def _occupied(mk: MarkovPossessionValue, cfg: GateConfig) -> list[int]:
    s = {p: mk.support(p).ravel() for p in (1, 2, 3)}  # all three terciles
    return [c for c in DEEP_ZONE_CELLS if all(s[p][c] >= cfg.n_min for p in (1, 2, 3))]


def _mean(fn, cells, p) -> float:
    return float(np.mean([fn(c, p) for c in cells])) if cells else 0.0


def run_deep_zone_gate(mk: MarkovPossessionValue, emp: EmpiricalPossessionValue, cfg: GateConfig) -> DeepZoneGateReport:
    occ = _occupied(mk, cfg)
    if len(occ) < cfg.min_occupied_cells:
        return DeepZoneGateReport(
            False,
            0.0,
            "n/a",
            False,
            False,
            len(occ),
            f"insufficient support: {len(occ)} occupied deep cells "
            f"(>= n_min in all terciles) < {cfg.min_occupied_cells}",
        )
    v1, v2, v3 = _mean(mk.value, occ, 1), _mean(mk.value, occ, 2), _mean(mk.value, occ, 3)
    effect = abs(v1 - v3)
    nonincreasing = v1 >= v2 >= v3
    nondecreasing = v1 <= v2 <= v3
    observed = "decreasing" if v1 > v3 else ("increasing" if v1 < v3 else "flat")
    if cfg.expected_direction == "decreasing":
        monotone_ok = nonincreasing
    elif cfg.expected_direction == "increasing":
        monotone_ok = nondecreasing
    else:
        monotone_ok = nonincreasing or nondecreasing
    mk_grad = _mean(mk.value, BUILD_UP_CELLS, 1) - _mean(mk.value, BUILD_UP_CELLS, 3)
    emp_grad = _mean(emp.value, BUILD_UP_CELLS, 1) - _mean(emp.value, BUILD_UP_CELLS, 3)
    same_sign = np.sign(mk_grad) == np.sign(emp_grad)
    rel_ok = abs(mk_grad - emp_grad) <= cfg.crosscheck_rel_tol * max(abs(mk_grad), abs(emp_grad), 1e-9)
    crosscheck = bool(same_sign and rel_ok)
    passed = bool(effect >= cfg.effect_floor and monotone_ok and crosscheck)
    reason = (
        ""
        if passed
        else "; ".join(
            s
            for s, ok in [
                (f"effect {effect:.4f}<{cfg.effect_floor}", effect >= cfg.effect_floor),
                (f"direction {observed}!={cfg.expected_direction}/non-monotone", monotone_ok),
                ("cross-check divergent", crosscheck),
            ]
            if not ok
        )
    )
    return DeepZoneGateReport(passed, effect, observed, monotone_ok, crosscheck, len(occ), reason)


# --- Pre-gate input-QC reports (owner-run, ADR-036 §6 Q3 + §5 G8) --------------


def ood_rate_by_source(
    shot_xg: pd.DataFrame, *, source_col: str = "data_source", ood_col: str = "ood_flag"
) -> dict[str, float]:
    """Per-cohort out-of-distribution rate of the injected xG reward (Q3).

    ``ood_flag`` rides on ``fct_shot_xg``; a high rate means the xG model is uncertified on that
    cohort's shots (RM/SkillCorner is 100% OOD live) -> the gate verdict for that cohort is
    provisional. Emitted pre-gate by the owner-run alongside the pressure-coverage report.
    """
    return {str(src): float(grp[ood_col].mean()) for src, grp in shot_xg.groupby(source_col, sort=False)}


def frame_present_null_pressure_count(
    actions: pd.DataFrame,
    *,
    pressure_col: str,
    frame_present_col: str,
    source_col: str = "data_source",
) -> dict[str, int]:
    """Per-cohort count of genuinely-unpressured restarts (frame present AND pressure null, §5 G8).

    These are signal, not loss: an unpressured goal-kick coalesces to the LOW tercile (kept), not a
    tracking gap. Reported so the operator sees how much of the deep-zone low tercile is restarts.
    """
    null_present = actions[frame_present_col].to_numpy(dtype=bool) & actions[pressure_col].isna().to_numpy()
    sub = actions[null_present]
    if source_col not in actions.columns:
        return {"_all": len(sub)}
    return {str(src): int(n) for src, n in sub.groupby(source_col, sort=False).size().items()}
