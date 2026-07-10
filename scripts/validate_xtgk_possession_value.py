"""Owner-run real-data gate for xT-GK v2 SP1 -- the make-or-break deep-zone gradient (ADR-036 §8).

DO NOT RUN until the GateConfig numbers are owner/Eyestone-LOCKED (deep-cell set, effect floor,
N_min, cross-check tolerance, expected direction). Q3 (the xG source) is RESOLVED: the injected
`xg_column` is the lakehouse gold mart `soccer_analytics.dev_gold.fct_shot_xg.xg`, joined to the
action stream on `(match_key, action_id)`.

PASS on real data (WC2022 / gradientsports) authorises xT-GK v2 sub-projects 2-5. FAIL => STOP and
escalate to owner + Eyestone; do not build V_opp / rho.

Cohort scope (2026-07-09, live):
  - WC2022 (gradientsports): certified in `fct_shot_xg` (context_aware, ood_flag=False) → the
    primary cohort.
  - SkillCorner (RM): `ood_flag=True` in `fct_shot_xg` (uncertified) → DROPPED per the
    `ood_flag => drop cohort` contract. The gate runs WC2022-only unless SkillCorner later certifies.

This is a wired-but-not-run script: it composes the pieces the gate needs and emits a JSON report.
The actual data loading (actions + tracking-derived pressure + the `fct_shot_xg` xg join) is done
by the owner-run harness (pining / Databricks); the placeholders below name the required inputs.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

from silly_kicks.xtgk import (
    EmpiricalPossessionValue,
    GateConfig,
    MarkovPossessionValue,
    PressureLevels,
    coalesce_frame_present_null_pressure,
    frame_present_null_pressure_count,
    ood_rate_by_source,
    run_deep_zone_gate,
)

# --- LOCKED GATE NUMBERS (Q4) -- placeholders until owner/Eyestone lock them. ------------------
# Do not treat these as final; they are illustrative so the harness is runnable end-to-end once
# real numbers land. The gate is meaningless with unpinned thresholds.
_GATE_CONFIG_PENDING = GateConfig(
    effect_floor=0.0,  # TODO(Q4): absolute |V(deep,lo) - V(deep,hi)| floor
    n_min=0,  # TODO(Q4): min support per gate cell in ALL THREE terciles
    min_occupied_cells=2,  # TODO(Q4)
    crosscheck_rel_tol=0.5,  # TODO(Q4)
    expected_direction="either",  # TODO(Q2/Eyestone): decreasing? confirm sign
)

_XG_COLUMN = "xg"  # fct_shot_xg.xg (calibrated pre-shot), joined on (match_key, action_id) (Q3)
_OOD_COLUMN = "ood_flag"  # fct_shot_xg per-shot certification flag (RM is 100% OOD live)
_CI_COLUMNS = ("xg_ci_low", "xg_ci_high")  # fct_shot_xg per-shot CI
_PRESSURE_COLUMN = "pressure_on_actor__andrienko_oval"  # tracking-derived; pin per §5 gradient check
_FRAME_PRESENT_COLUMN = "frame_present"  # frame-derived non-null indicator (e.g. team_shape_* IS NOT NULL)


def _gate_is_locked(cfg: GateConfig) -> bool:
    return cfg.effect_floor > 0.0 and cfg.n_min > 0


def prepare_cohort(actions, *, pressure_column: str, frame_present_column: str):
    """G8 (§5) frame-aware null-pressure data-prep, applied BEFORE fit.

    Coalesces null pressure -> 0 (LOW tercile) for frame-present rows (genuinely unpressured
    restarts — 595/595 GS goal-kicks live); leaves frame-absent nulls null so PressureLevels.apply
    fail-loud-drops the genuine tracking gaps. Returns a NEW frame; never mutates input.
    """
    out = actions.copy()
    out[pressure_column] = coalesce_frame_present_null_pressure(out[pressure_column], out[frame_present_column])
    return out


def reward_provenance_summary(shot_xg, *, ood_column: str, ci_columns) -> dict:
    """Q3 (§6): summarize the injected reward's certification for MarkovPossessionValue.provenance.

    silly-kicks records but never interprets ood_flag/CI semantics (ships no xG model)."""
    lo, hi = ci_columns
    return {
        "xg_source": "fct_shot_xg.xg",
        "ood_rate": float(shot_xg[ood_column].mean()),
        "xg_ci_mean_width": float((shot_xg[hi] - shot_xg[lo]).mean()),
        "n_shots": len(shot_xg),
    }


def run_gate_on_cohort(
    actions, *, xg_column: str, pressure_column: str, cfg: GateConfig, reward_provenance: dict | None = None
) -> dict:
    """Fit V + the cross-check on ONE attack-LTR, frame-aware-prepared cohort and run the gate.

    ``actions`` must be attack-LTR SPADL carrying ``xg_column`` (from fct_shot_xg) + ``pressure_column``,
    joined on (match_key, action_id), and already run through ``prepare_cohort`` (G8).
    """
    pl = PressureLevels().fit(actions[pressure_column])  # one tercile fit, shared by both estimators
    mk = MarkovPossessionValue().fit(
        actions,
        xg_column=xg_column,
        pressure_column=pressure_column,
        pressure_levels=pl,
        reward_provenance=reward_provenance,
    )
    emp = EmpiricalPossessionValue().fit(
        actions, xg_column=xg_column, pressure_column=pressure_column, pressure_levels=pl
    )
    report = run_deep_zone_gate(mk, emp, cfg)
    return {
        "gate": asdict(report),
        "cutpoints": list(pl.cutpoints) if pl.cutpoints else [],
        "support": {p: int(mk.support(p).sum()) for p in (1, 2, 3)},
        "reward_provenance": mk.provenance.get("reward_provenance"),
        "tercile_occupancy": pl.occupancy(actions[pressure_column]),
    }


def input_qc_reports(shot_xg, actions) -> dict:
    """Pre-gate input QC (Q3 OOD-rate + §5 G8 unpressured-restart count), emitted per cohort."""
    return {
        "ood_rate_by_source": ood_rate_by_source(shot_xg, ood_col=_OOD_COLUMN),
        "unpressured_restart_count": frame_present_null_pressure_count(
            actions, pressure_col=_PRESSURE_COLUMN, frame_present_col=_FRAME_PRESENT_COLUMN
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/research/xtgk_possession_value/gate.json"))
    parser.add_argument("--force-unlocked", action="store_true", help="run with placeholder GateConfig")
    args = parser.parse_args()

    if not _gate_is_locked(_GATE_CONFIG_PENDING) and not args.force_unlocked:
        print(
            "BLOCKED: GateConfig numbers are not locked (Q4). This gate is the make-or-break go/no-go "
            "for xT-GK v2 SP2-5 and is meaningless with unpinned thresholds. Lock effect_floor / "
            "n_min / direction with owner + Eyestone, then re-run (or pass --force-unlocked for a "
            "dry structural check).",
            file=sys.stderr,
        )
        return 2

    # The owner-run harness supplies WC2022 (gradientsports) attack-LTR actions with the
    # fct_shot_xg xg column and tracking-derived pressure joined on (match_key, action_id).
    # Left unwired here by design: pining/Databricks loading lives in the owner run.
    raise NotImplementedError(
        "Wire the WC2022 loader (actions + fct_shot_xg xg + pressure, both orientations) here for "
        "the owner run. run_gate_on_cohort() is the composed entry point; emit the report to --out."
    )


if __name__ == "__main__":
    raise SystemExit(main())
