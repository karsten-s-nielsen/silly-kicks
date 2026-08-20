"""Cover-shadow RQ1 validation artifact from the persisted per-pass scores.

Consumer of ``build_rq_pass_scores``'s ``pass_scores.parquet`` (+ its ``manifest.json``). Reported,
NEVER gated: it computes the leakage-aware metric hierarchy and writes an auditable research
artifact; it does not pass or fail CI on its numbers.

Metric hierarchy (spec 2): the leakage-free PASS-ONLY completed-pass false-positive rate leads;
AUC / recall / slope read failed passes and are OPTIMISTIC; ECE is the recalibration baseline. This
measures OVER-PREDICTION, not DETECTION (recall needs the failed-pass class, deferred to the
Power-2017 receiver model).

Spec: docs/superpowers/specs/2026-08-19-cover-shadow-rq1-and-pass-risk-calibration-design.md
"""

from __future__ import annotations

import argparse
import json
import pathlib

import pandas as pd

from scripts import _rq_metrics as rqm
from scripts._corpus import artifact_label
from scripts._provenance import git_provenance, require_clean_tree

_CASCIOLI_MAJORITY_RECALL = 0.369
_CASCIOLI_MAJORITY_PRECISION = 0.220
_SCOPE_NOTE = (
    "This measures OVER-PREDICTION (specificity on completed passes), NOT DETECTION (recall) -- recall "
    "needs the failed-pass class, which is both leaked (outcome-selected end_xy target) and confounded, "
    "until the deferred Power-2017 expected-receiver model lands. The clean headline is not a full validation."
)


def _fp_rates(sub: pd.DataFrame) -> dict:
    out = {"majority": rqm.false_positive_rate(sub["is_blocked_majority"], sub["is_completed"])}
    for agg in ("center", "mean", "max"):
        out[agg] = rqm.false_positive_rate(sub[f"p_blocked_{agg}"] > 0.5, sub["is_completed"])
    return out


def compute_cover_shadow_metrics(df: pd.DataFrame) -> dict:
    """The full leakage-aware hierarchy (pure -- no I/O)."""
    pass_only = df[~df["is_cross"].astype(bool)]
    confusion = rqm.confusion(df["is_blocked_majority"], df["is_fail"])
    recall = confusion["recall"]
    # The model's decision compares p_blocked to p_received PER LANE, so the discriminating
    # continuous score is the margin the majority rule counts (n_blocked) / the mean margin --
    # NOT the absolute p_blocked intensity, whose AUC is ~0.5 (kept below as `abs_p_blocked`).
    p_received_mean = df[["p_received_center", "p_received_left", "p_received_right"]].mean(axis=1)
    margin_mean = df["p_blocked_mean"] - p_received_mean
    return {
        "headline_fp_rate": _fp_rates(pass_only),  # PASS-ONLY, leakage-free -- the trustworthy number
        "pass_plus_cross_secondary": _fp_rates(df),  # paper-comparable cut (crosses are aerial)
        "optimistic": {
            "leakage_inflated": True,
            "auc": {
                # the DISCRIMINATING score: the margin the majority rule thresholds
                "n_blocked": rqm.auc(df["is_fail"], df["n_blocked"]),
                "margin_mean": rqm.auc(df["is_fail"], margin_mean),
                # absolute p_blocked kept to SHOW the magnitude alone does not discriminate (~0.5)
                "abs_p_blocked": {
                    agg: rqm.auc(df["is_fail"], df[f"p_blocked_{agg}"]) for agg in ("center", "mean", "max")
                },
            },
            "confusion": confusion,
            "reliability_slope": rqm.reliability_slope(df["is_fail"], df["p_blocked_mean"]),
        },
        "recalibration_baseline": {
            "ece": rqm.ece(df["is_fail"], df["p_blocked_mean"]),
            "reliability_curve": rqm.reliability_curve(df["is_fail"], df["p_blocked_mean"]),
            "note": (
                "pre-recalibration baseline (p_blocked is P(screened), not P(fail)); the sigma/lambda "
                "cycle drives ECE down. Selection-biased to attempted passes -- see the TF-24 handoff."
            ),
        },
        "paper_reconciliation": {
            "cascioli_majority_recall": _CASCIOLI_MAJORITY_RECALL,
            "cascioli_majority_precision": _CASCIOLI_MAJORITY_PRECISION,
            "required_sentence": (
                f"our majority recall {recall:.3f} vs the paper's {_CASCIOLI_MAJORITY_RECALL} "
                "(recomputed from Cascioli Appendix B, not the handoff table)."
            ),
        },
        "scope_note": _SCOPE_NOTE,
        "n_passes": len(df),
        "n_pass_only": len(pass_only),
        "n_completed": int(df["is_completed"].astype(bool).sum()),
    }


def refuse_dirty_upstream(pass_scores: pathlib.Path, expect_commit: str | None) -> dict:
    """ADR-037: this artifact derives from build_rq_pass_scores -- refuse a dirty/missing/mismatched upstream."""
    man_path = pass_scores.parent / "manifest.json"
    if not man_path.exists():
        raise SystemExit(f"missing upstream manifest {man_path} -- unprovenanced counts as dirty (ADR-037)")
    man = json.loads(man_path.read_text(encoding="utf-8"))
    if man.get("run_tree_dirty", True):
        raise SystemExit(f"upstream {man_path} was produced on a DIRTY tree (ADR-037)")
    if expect_commit is not None and man.get("run_commit") != expect_commit:
        raise SystemExit(f"upstream run_commit {man.get('run_commit')!r} != expected {expect_commit!r}")
    return man


def _readme(metrics: dict, corpus_label: str) -> str:
    h = metrics["headline_fp_rate"]
    return (
        "# Cover-shadow RQ1 -- real-data validation\n\n"
        f"Corpus: **{corpus_label}** (owner-tier Gradient Sports WC2022). Raw per-pass positions are NOT "
        "committed (see the gitignored `pass_scores.parquet`); only these aggregate rates are.\n\n"
        f"**{_SCOPE_NOTE}**\n\n"
        "## Headline (leakage-free): completed-pass false-positive rate, PASS-ONLY\n"
        f"- majority rule: {h['majority']}\n- p_blocked>0.5 (center/mean/max): "
        f"{h['center']} / {h['mean']} / {h['max']}\n\n"
        "The pass-only cut leads because `lane_control` models GROUND-lane screening and crosses are aerial.\n\n"
        "## Optimistic (read failed passes -> leakage-inflated; not the headline)\n"
        f"- AUC, discriminating score (n_blocked / mean margin): {metrics['optimistic']['auc']['n_blocked']} / "
        f"{metrics['optimistic']['auc']['margin_mean']}\n"
        f"- AUC, absolute p_blocked magnitude (center/mean/max, ~0.5 -- the WRONG score): "
        f"{metrics['optimistic']['auc']['abs_p_blocked']}\n"
        "  The model compares `p_blocked` to `p_received` per lane, so the discriminating quantity is the\n"
        "  margin / `n_blocked` count the majority rule thresholds, not the absolute `p_blocked` intensity.\n"
        f"- reliability slope (on P(screened) = p_blocked_mean): {metrics['optimistic']['reliability_slope']}\n"
        f"- confusion (paper-comparable): {metrics['optimistic']['confusion']}\n\n"
        "## Paper reconciliation\n"
        f"{metrics['paper_reconciliation']['required_sentence']}\n\n"
        "## Limitations\n"
        "- Selection bias (unfixable): only attempted passes are observed; precision is a lower bound.\n"
        "- Failed-pass `end_xy` target is outcome-selected -> the failed-pass legs are optimistic bounds.\n"
        "- Screening != failure: `p_blocked` is P(lane screened); the reliability curve is a mapping.\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Cover-shadow RQ1 validation artifact from the persisted pass scores.")
    ap.add_argument("--pass-scores", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--expect-commit", default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    prov = git_provenance()
    require_clean_tree(prov, allow_dirty=args.allow_dirty)  # THIS tree
    refuse_dirty_upstream(args.pass_scores, args.expect_commit)  # the upstream it derives from

    df = pd.read_parquet(args.pass_scores)
    metrics = compute_cover_shadow_metrics(df)
    corpus_label = artifact_label(providers={"gradientsports"}, all_public=False)  # ship-mask (F5)
    metrics["corpus_visibility"] = corpus_label
    metrics["run_commit"] = prov["commit"]
    metrics["run_tree_dirty"] = prov["dirty"]

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (args.out / "README.md").write_text(_readme(metrics, corpus_label), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
