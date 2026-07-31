"""Owner-run real-data gate for xT-GK v2 -- the make-or-break deep-zone gradient (ADR-036 §8).

GateConfig numbers are owner/Eyestone-LOCKED (Q4) and the loader is wired to Databricks gold
(`_loader_databricks.load_xtgk_cohort` = `bronze.spadl_actions ⋈ dim_matches ⋈ dev_gold.fct_action_context`
[pressure + frame-present] `⋈ dev_gold.fct_shot_xg` [calibrated xG], keyed on (match_key, action_id)).

RESULT (run 2026-07-10; see docs/research/xtgk_possession_value/GATE_FINDINGS.md): WC2022 STOP,
RM (100% OOD) FAIL-crosscheck -- root cause is a 52% pressure-exactly-0 mass degenerating the tercile
stratification. Per the owner build-ahead directive SP2-5 shipped regardless (the STOP quantifies the
degeneracy); escalated to Eyestone for a pressure-zero-stratum fix.

Cohort scope (owner decision):
  - WC2022 (gradientsports): certified in `fct_shot_xg` (ood_flag=False) → the AUTHORISING cohort.
  - SkillCorner (RM): `ood_flag=True` (uncertified) → INCLUDE as a PROVISIONAL second read (not dropped),
    reported separately, tagged provisional because 100% OOD.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

from silly_kicks.xtgk import (
    GateConfig,
    coalesce_frame_present_null_pressure,
)

# --- LOCKED GATE NUMBERS (Q4, owner-set 2026-07-09; Eyestone to confirm) -----------------------
_GATE_CONFIG_LOCKED = GateConfig(
    effect_floor=0.005,
    relative_effect_floor=0.25,  # primary acceptance criterion (B2, gate-enforced)
    n_min=30,
    min_occupied_cells=2,
    crosscheck_rel_tol=0.5,
    expected_direction="decreasing",
)

_XG_COLUMN = "xg"  # fct_shot_xg.xg (calibrated pre-shot), joined on (match_key, action_id) (Q3)
_OOD_COLUMN = "ood_flag"  # fct_shot_xg per-shot certification flag (RM is 100% OOD live)
_CI_COLUMNS = ("xg_ci_low", "xg_ci_high")  # fct_shot_xg per-shot CI
# Pinned to bekkers_pi (§5 Q3, resolved by the lakehouse 3-method audit 2026-07-10): andrienko_oval
# floors to exactly 0 for ~47% of actions (link_zones ~80%) -- a parameterization artifact that
# degenerates the pressure terciles; bekkers_pi has a non-degenerate tail (~5% zero) and is the
# trustworthy measure. See docs/research/xtgk_possession_value/LAKEHOUSE_HANDOFF.md (F2).
_PRESSURE_COLUMN = "pressure_on_actor__bekkers_pi"
_FRAME_PRESENT_COLUMN = "frame_present"  # frame-derived non-null indicator (e.g. team_shape_* IS NOT NULL)


def _gate_is_locked(cfg: GateConfig) -> bool:
    return cfg.effect_floor > 0.0 and cfg.n_min > 0


def prepare_cohort(actions, *, pressure_column: str, frame_present_column: str):
    """G8 (§5) frame-aware null-pressure data-prep, applied BEFORE fit.

    Coalesces null pressure -> 0 (LOW tercile) for frame-present rows (genuinely unpressured
    restarts), then DROPS the residual frame-absent nulls (genuine tracking gaps — the §5 backstop;
    ``PressureLevels.apply`` would otherwise fail loud on them). Returns a NEW frame; never mutates input.
    """
    out = actions.copy()
    out[pressure_column] = coalesce_frame_present_null_pressure(out[pressure_column], out[frame_present_column])
    return out[out[pressure_column].notna()].reset_index(drop=True)


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/research/xtgk_possession_value/gate.json"))
    parser.add_argument("--force-unlocked", action="store_true", help="run with placeholder GateConfig")
    parser.add_argument(
        "--cohort-cache",
        default=None,
        help=(
            "DIRECTORY for the per-cohort parquet caches; fetch each cohort once and reuse it. "
            "Absent = fetch every run (today's behaviour). A directory rather than a single path "
            "because this driver loads two cohorts and two frames each. Explicitly named because a "
            "mart re-materializes and a cached cohort has no token this can verify -- so reuse must "
            "be the operator's decision, never automatic."
        ),
    )
    args = parser.parse_args()

    if not _gate_is_locked(_GATE_CONFIG_LOCKED) and not args.force_unlocked:
        print(
            "BLOCKED: GateConfig numbers are not locked (Q4). This gate is the make-or-break go/no-go "
            "for xT-GK v2 SP2-5 and is meaningless with unpinned thresholds. Lock effect_floor / "
            "n_min / direction with owner + Eyestone, then re-run (or pass --force-unlocked for a "
            "dry structural check).",
            file=sys.stderr,
        )
        return 2

    import json

    # OWNER-RUN DATA ACCESS (Databricks read-only gold marts): bronze.spadl_actions bridged via
    # dim_matches to fct_action_context (pressure + frame-present) + fct_shot_xg (calibrated xG).
    # Returns attack-LTR SPADL with xg / pressure / frame_present / possession_id + per-shot OOD/CI.
    from scripts._loader_databricks import load_xtgk_cohort
    from silly_kicks.xtgk import run_gate_both_orientations
    from silly_kicks.xtgk._diagnostics import frame_present_null_pressure_count, ood_rate_by_source

    cohorts = {
        "wc2022": {"data_source": "gradientsports", "authorising": True},
        "rm": {"data_source": "skillcorner", "authorising": False},  # provisional (100% OOD)
    }
    from scripts._driver import cohort_cache

    # `load_xtgk_cohort` returns TWO frames from ONE query, and `cohort_cache` caches one parquet
    # per path -- so the two cached halves would otherwise cost two full queries. This memo makes
    # the second `cohort_cache` call reuse the first's result. On a cache HIT neither call reaches
    # `build`, so no query runs at all; on a MISS exactly one does.
    _query_memo: dict = {}

    def _query(src):
        if src not in _query_memo:
            _query_memo[src] = load_xtgk_cohort(src)
        return _query_memo[src]

    cache_dir = Path(args.cohort_cache) if getattr(args, "cohort_cache", None) else None

    report: dict = {}
    for name, spec in cohorts.items():
        src = spec["data_source"]
        raw = cohort_cache(
            cache_dir / f"{name}_actions.parquet" if cache_dir else None, build=lambda s=src: _query(s)[0]
        )
        shot_xg = cohort_cache(
            cache_dir / f"{name}_shot_xg.parquet" if cache_dir else None, build=lambda s=src: _query(s)[1]
        )
        # count the frame-present unpressured restarts on the RAW frame (prepare_cohort coalesces them
        # to 0, so the count must precede prep); then prep drops the frame-absent tracking gaps.
        unpressured = frame_present_null_pressure_count(
            raw, pressure_col=_PRESSURE_COLUMN, frame_present_col=_FRAME_PRESENT_COLUMN
        )
        actions = prepare_cohort(raw, pressure_column=_PRESSURE_COLUMN, frame_present_column=_FRAME_PRESENT_COLUMN)
        prov = reward_provenance_summary(shot_xg, ood_column=_OOD_COLUMN, ci_columns=_CI_COLUMNS)
        gate = run_gate_both_orientations(
            actions, xg_column=_XG_COLUMN, pressure_column=_PRESSURE_COLUMN, cfg=_GATE_CONFIG_LOCKED
        )
        report[name] = {
            "authorising": spec["authorising"],
            "n_actions_fit": len(actions),
            "n_actions_dropped_gap": len(raw) - len(actions),
            "reward_provenance": prov,
            "ood_rate_by_source": ood_rate_by_source(shot_xg, ood_col=_OOD_COLUMN),
            "unpressured_restart_count": unpressured,
            "fit_rung": gate["fit"]["rung"],
            "fit_gate": asdict(gate["fit"]["report"]),
            "mirror_y_gate": asdict(gate["mirror_y"]["report"]),
            "mirror_x_rejected": gate["mirror_x"]["orientation_rejected"],
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote gate report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
