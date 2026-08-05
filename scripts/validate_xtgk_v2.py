"""Owner-run validation suite for xT-GK v2 (ADR-036 SPart 5).

Construct validity is OUT-OF-SAMPLE (possession-parity split) and reported as LIFT over baselines --
V is (by construction) the expected first-shot xG, so absolute AUC vs a possession->shot target is
partly circular; the informative quantity is v2's margin over raw completion / destination-only V /
the v1 composite. The synthetic CI smoke uses a constant-rho stub (frames-free); the owner-run passes
the REAL calibrated rho with frames-derived retention_features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.spadl.utils import add_possessions
from silly_kicks.xtgk import (
    EmpiricalTurnoverValue,
    MarkovPossessionValue,
    MirroredTurnoverCost,
    PressureLevels,
    compute_xt_gk_v2,
)
from silly_kicks.xtgk._possession_value import flat_zones  # NaN-coord-safe (real cohorts carry NaN coords)

_SHOT = spadlconfig.actiontype_id["shot"]


class _ConstRho:
    """Frames-free stub for the CI smoke; the owner-run injects the real GkRetentionModel."""

    def predict_proba(self, features):
        return np.full(len(features), 0.75)


def _possession_reaches_shot(actions: pd.DataFrame) -> np.ndarray:
    a = actions
    out = np.zeros(len(a), dtype=int)
    typ = a["type_id"].to_numpy()
    poss = a["possession_id"].to_numpy()
    for i in range(len(a)):
        for j in range(i, len(a)):
            if poss[j] != poss[i]:
                break
            if typ[j] == _SHOT:
                out[i] = 1
                break
    return out


def _auc(y, s) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    ok = np.isfinite(s)
    if ok.sum() < 2 or len(np.unique(y[ok])) < 2:
        return float("nan")
    return float(roc_auc_score(y[ok], s[ok]))


def _destination_only_v(
    test: pd.DataFrame, v: MarkovPossessionValue, pl: PressureLevels, pressure_column: str
) -> np.ndarray:
    zd = flat_zones(test["end_x"], test["end_y"])
    zones_arg = flat_zones(test["start_x"], test["start_y"]) if pl.mode == "zone_conditional" else None
    lv = pl.apply(test[pressure_column], zones=zones_arg)
    return np.array([v.value(int(z), int(p)) for z, p in zip(zd, lv, strict=True)])  # type: ignore[arg-type]


def construct_validity_scores(
    actions: pd.DataFrame,
    *,
    xg_column: str,
    pressure_column: str,
    retention=None,
    turnover_cost=None,
    kappa: float = 1.0,
) -> dict:
    a = actions.reset_index(drop=True)
    if "possession_id" not in a.columns:
        a = add_possessions(a)
    train_mask = (a["possession_id"] % 2 == 0).to_numpy()  # out-of-sample by possession parity
    train, test = a[train_mask].copy(), a[~train_mask].copy()
    pl = PressureLevels().fit(train[pressure_column])
    v = MarkovPossessionValue().fit(train, xg_column=xg_column, pressure_column=pressure_column, pressure_levels=pl)
    # FAITHFUL V_opp (Jeff S2.3): observed post-turnover, possession-bound, bin-widened -- fit on TRAIN
    # (no leakage into the AUC). Default production; a caller may inject a different TurnoverCost.
    if turnover_cost is None:
        turnover_cost = EmpiricalTurnoverValue(min_support=30).fit(
            train, xg_column=xg_column, pressure_column=pressure_column, pressure_levels=pl
        )
    tc = turnover_cost

    # target on the FULL test (the forward scan needs the intact possession sequence), then restrict to the
    # GK-distribution domain (where rho / v2 are defined; compute_xt_gk_v2 does not self-gate).
    y_full = _possession_reaches_shot(test)
    if "is_gk_distribution" in test.columns:
        gk = test["is_gk_distribution"].fillna(False).to_numpy(dtype=bool)
    else:
        gk = np.ones(len(test), dtype=bool)  # synthetic CI fixtures: no domain column -> all-test
    test_gk = test[gk].reset_index(drop=True)
    y = y_full[gk]

    if retention is None:
        retention = _ConstRho()
        feats = pd.DataFrame(index=test_gk.index)  # _ConstRho ignores content
    else:
        from silly_kicks.xtgk._retention_features import extract_retention_features

        feats = extract_retention_features(test_gk, pressure_column=pressure_column)

    v2 = compute_xt_gk_v2(
        test_gk,
        possession_value=v,
        retention=retention,
        turnover_cost=tc,
        pressure_column=pressure_column,
        pressure_levels=pl,
        retention_features=feats,
        kappa=kappa,
    )
    raw_completion = (test_gk["result_id"] == spadlconfig.result_id["success"]).astype(int).to_numpy()
    dest = _destination_only_v(test_gk, v, pl, pressure_column)
    v1 = (
        pd.to_numeric(test_gk["xt_gk"], errors="coerce").to_numpy()
        if "xt_gk" in test_gk.columns
        else np.full(len(test_gk), np.nan)
    )

    v2_arr = v2["xt_gk_v2"].to_numpy()
    v2_auc, raw_auc, dest_auc = _auc(y, v2_arr), _auc(y, raw_completion), _auc(y, dest)
    v1_ok = np.isfinite(v1)
    v1_auc = _auc(y, v1)  # _auc filters non-finite scores (v1 nulls dropped from its denominator)
    # apples-to-apples v2-vs-v1: v2 restricted to the v1-covered rows (v1 covers ~89% GS / 100% SC), so the
    # "does v2 beat v1" number is on a matched denominator (the lift below uses full-test-gk baselines).
    v2_on_v1_rows = _auc(y[v1_ok], v2_arr[v1_ok])
    baselines = [b for b in (raw_auc, dest_auc, v1_auc) if np.isfinite(b)]
    lift = float(v2_auc - max(baselines)) if (np.isfinite(v2_auc) and baselines) else float("nan")

    # component decomposition (did the faithful V_opp un-swamp the value-added rho*dV term?)
    pos, pev, ret, dzv = (
        v2[c].to_numpy() for c in ("xt_gk_v2_position", "xt_gk_v2_pev", "xt_gk_v2_retention_loss", "xt_gk_v2_dzv")
    )
    mags = {
        k: float(np.nanmean(np.abs(x)))
        for k, x in (("position", pos), ("pev", pev), ("retention_loss", ret), ("dzv", dzv))
    }
    tot = sum(mags.values()) or 1.0
    decomposition = {
        "share": {k: v / tot for k, v in mags.items()},
        "auc_rho_dv": _auc(y, pos + pev),  # value-added alone
        "auc_partial": _auc(y, pos + pev + ret),  # + retention loss
        "auc_full": v2_auc,
    }

    return {
        "xt_gk_v2": {"auc": v2_auc},
        "raw_completion": {"auc": raw_auc},
        "destination_xt": {"auc": dest_auc},
        "v1_stored": {"auc": v1_auc, "n": int(v1_ok.sum())},
        "v2_on_v1_rows": {"auc": v2_on_v1_rows},
        "lift": lift,
        "decomposition": decomposition,
        "n_test_gk": len(test_gk),
        # fitted TRAIN artifacts for the owner-run report (R1 disentanglement / R4 resolution map); underscore
        # keys are report-only and skipped by _write_report's table.
        "_v": v,
        "_pl": pl,
        "_train": train,
        "_turnover_cost": tc,
        "_xg_column": xg_column,
        "_pressure_column": pressure_column,
        "_note": (
            "GK-distribution-domain eval (is_gk_distribution); V out-of-sample (possession-parity split), "
            "rho IN-SAMPLE (the production model serves its training population); V is ~expected first-shot "
            "xG so absolute AUC vs possession-reaches-shot is partly circular -- read LIFT over max(baselines). "
            "V_opp = faithful observed-post-turnover (possession-bound), TRAIN-fit; v1_stored from "
            "fct_action_context.xt_gk (no frames)."
        ),
    }


def _deep_cell_disentanglement(scores: dict) -> tuple[list[str], dict[int, int]]:
    """R1: per deep cell, mean across terciles of the production possession-bound V_opp, the mirror proxy,
    and the 10s-capped sensitivity, + native n + modal resolution level. Separates 'mirror over-stated'
    (pb << mirror at real support) from 'window shrinks V_opp' (10s << pb). R4: resolution-level census."""
    import numpy as np

    from silly_kicks.xtgk._diagnostics import DEEP_ZONE_CELLS
    from silly_kicks.xtgk._turnover import EmpiricalTurnoverValue

    v, pl, train = scores["_v"], scores["_pl"], scores["_train"]
    emp_pb = scores["_turnover_cost"]  # possession-bound production
    xg_col, p_col = scores["_xg_column"], scores["_pressure_column"]
    emp_10s = EmpiricalTurnoverValue(min_support=30, window_seconds=10.0).fit(
        train, xg_column=xg_col, pressure_column=p_col, pressure_levels=pl
    )
    mirror = MirroredTurnoverCost(v)
    rows = [
        "\n### R1 deep-cell disentanglement (V_opp, train-fit; mean over terciles)\n",
        "| zone | possession-bound (prod) | mirror (proxy) | 10s (sens.) | native n | level |\n"
        "|---|---|---|---|---|---|\n",
    ]
    level_census: dict[int, int] = {0: 0, 1: 0, 2: 0, -1: 0}
    for c in DEEP_ZONE_CELLS:
        pb = np.mean([emp_pb.value(c, p) for p in (1, 2, 3)])
        mi = np.mean([mirror.value(c, p) for p in (1, 2, 3)])
        te = np.mean([emp_10s.value(c, p) for p in (1, 2, 3)])
        nmin = int(min(emp_pb.support(p).ravel()[c] for p in (1, 2, 3)))
        lvl = int(np.bincount([emp_pb.resolution_level(p).ravel()[c] + 1 for p in (1, 2, 3)]).argmax()) - 1
        level_census[lvl] = level_census.get(lvl, 0) + 1
        rows.append(f"| {c} | {pb:.4f} | {mi:.4f} | {te:.4f} | {nmin} | {lvl} |\n")
    rows.append(
        f"\n> Read: **possession-bound << mirror at real support (level 0/1)** = the mirror over-stated deep "
        f"threat (the genuine finding). **10s << possession-bound** = window shrinkage (an artifact -- NOT the "
        f"finding). Level census over {len(DEEP_ZONE_CELLS)} deep cells: "
        f"native {level_census.get(0, 0)} / block {level_census.get(1, 0)} / "
        f"global {level_census.get(-1, 0) + level_census.get(2, 0)} "
        f"(a global-fallback deep cell is NOT a real estimate).\n"
    )
    return rows, level_census


def _write_report(provider: str, variant: str, scores: dict) -> str:
    from pathlib import Path

    d = scores["decomposition"]
    disentangle, _ = _deep_cell_disentanglement(scores)
    lines = [
        f"# xT-GK v2 construct-validity -- {provider} (FAITHFUL V_opp)\n",
        f"- rho variant: `{variant}` * GK-distribution test rows: **{scores['n_test_gk']}** * "
        f"V_opp = faithful observed-post-turnover, possession-bound, TRAIN-fit\n",
        "\n| metric | AUC | n |\n|---|---|---|\n",
        f"| **xt_gk_v2** | {scores['xt_gk_v2']['auc']:.4f} | {scores['n_test_gk']} |\n",
        f"| raw_completion | {scores['raw_completion']['auc']:.4f} | {scores['n_test_gk']} |\n",
        f"| destination_xt | {scores['destination_xt']['auc']:.4f} | {scores['n_test_gk']} |\n",
        f"| v1_stored (c.xt_gk) | {scores['v1_stored']['auc']:.4f} | {scores['v1_stored']['n']} |\n",
        f"| xt_gk_v2 (on v1-covered rows) | {scores['v2_on_v1_rows']['auc']:.4f} | {scores['v1_stored']['n']} |\n",
        f"\n**LIFT** (v2 - max baseline, full GK-test): **{scores['lift']:+.4f}**\n",
        f"\n**v2 vs v1 (matched rows):** v2 {scores['v2_on_v1_rows']['auc']:.4f} "
        f"vs v1 {scores['v1_stored']['auc']:.4f} "
        f"(d {scores['v2_on_v1_rows']['auc'] - scores['v1_stored']['auc']:+.4f})\n",
        "\n### Component decomposition (did the faithful V_opp un-swamp rho*dV?)\n",
        "| term | \\|mean\\| share |\n|---|---|\n" + "".join(f"| {k} | {v:.0%} |\n" for k, v in d["share"].items()),
        f"\nAUC (harness target): **rho*dV alone {d['auc_rho_dv']:.4f}** * +retention {d['auc_partial']:.4f} * "
        f"full {d['auc_full']:.4f}\n",
        *disentangle,
        f"\n> {scores['_note']}\n",
    ]
    out = Path(__file__).resolve().parent.parent / "docs" / "research" / "xtgk_v2_construct_validity" / f"{provider}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(lines), encoding="utf-8")
    return str(out)


def main() -> int:
    import argparse

    # Parse args BEFORE the heavy/connection imports so `--help` short-circuits connection-free.
    ap = argparse.ArgumentParser(description="xT-GK v2 construct-validity with the real bundled rho (SP5).")
    ap.add_argument("--provider", default="gradientsports")
    ap.add_argument(
        "--cohort-cache",
        default=None,
        help=(
            "parquet path; fetch the cohort once and reuse it. Absent = fetch every run (today's "
            "behaviour). Explicitly named because a mart re-materializes and a cached cohort has no "
            "token this can verify -- so reuse must be the operator's decision, never automatic."
        ),
    )
    ap.add_argument(
        "--retention-weights",
        default=None,
        help="Path to a rho artifact dir (model.json + SHA256SUMS). Overrides the provider variant. "
        "Used by the ADR-036 two-leg SP5 re-run: leg 1 = corrected coords + PRE-FIX rho; "
        "leg 2 = corrected coords + retrained rho.",
    )
    a = ap.parse_args()

    from _loader_databricks import load_xtgk_cohort, resolve_retention_model  # type: ignore[import-not-found]
    from validate_xtgk_possession_value import (  # type: ignore[import-not-found]
        _FRAME_PRESENT_COLUMN,
        _PRESSURE_COLUMN,
        _XG_COLUMN,
        prepare_cohort,
    )

    from scripts._driver import cohort_cache
    from silly_kicks.xtgk._retention import variant_key_for_provider

    raw = cohort_cache(a.cohort_cache, build=lambda: load_xtgk_cohort(a.provider)[0])
    actions = prepare_cohort(raw, pressure_column=_PRESSURE_COLUMN, frame_present_column=_FRAME_PRESENT_COLUMN)
    # ADR-036 two-leg SP5: the report MUST name which rho produced it, else leg 1 (pre-fix rho) and
    # leg 2 (retrained rho) are indistinguishable in their own artifacts.
    variant = a.retention_weights or f"variant:{variant_key_for_provider(a.provider)}"
    rho = resolve_retention_model(a.provider, a.retention_weights)
    scores = construct_validity_scores(actions, xg_column=_XG_COLUMN, pressure_column=_PRESSURE_COLUMN, retention=rho)
    print(f"provider={a.provider} rho={variant} n_test_gk={scores['n_test_gk']}")
    for k in ("xt_gk_v2", "raw_completion", "destination_xt", "v1_stored", "v2_on_v1_rows"):
        print(f"  {k}: AUC={scores[k]['auc']:.4f}" + (f" (n={scores[k]['n']})" if "n" in scores[k] else ""))
    print(f"  LIFT (v2 - max baseline) = {scores['lift']:+.4f}")
    print("wrote", _write_report(a.provider, variant, scores))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
