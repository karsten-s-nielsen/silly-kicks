"""Pass-risk calibration validation artifact from the persisted per-pass scores.

Consumer of ``build_rq_pass_scores``'s ``pass_scores.parquet``. Reported, NEVER gated. Validates
``pitch_control_at_target`` as a pass-success predictor: the completed-pass false-alarm rate leads
(leakage-free); AUC / ECE / slope read failed passes and are OPTIMISTIC (the same end_xy leakage as
Driver A); the low-control COMPLETION band is contaminated and kept distinct from the headline.

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
from scripts.validate_cover_shadow_rq1 import refuse_dirty_upstream  # single-source the ADR-037 refusal

_SCOPE_NOTE = (
    "This measures OVER-PREDICTION (specificity on completed passes), NOT DETECTION (recall) -- recall "
    "needs the failed-pass class, which is both leaked (outcome-selected end_xy target) and confounded, "
    "until the deferred Power-2017 expected-receiver model lands. The clean headline is not a full validation."
)


def compute_pass_risk_metrics(df: pd.DataFrame) -> dict:
    """The leakage-aware pass-risk hierarchy (pure -- no I/O). A completed pass IS a success (result_id)."""
    is_success = df["is_completed"].astype(bool)
    band = rqm.low_control_completion_band(df["control"], is_success)
    return {
        "headline_false_alarm_rate": {  # completed-only, leakage-free (mirror of Driver A's FP rate)
            str(tau): rqm.false_alarm_rate(df["control"], is_success, tau) for tau in (0.1, 0.2, 0.3)
        },
        "optimistic": {
            "leakage_inflated": True,
            "auc": rqm.auc(is_success, df["control"]),
            "ece": rqm.ece(is_success, df["control"]),
            "reliability_slope": rqm.reliability_slope(is_success, df["control"]),
            "reliability_curve": rqm.reliability_curve(is_success, df["control"]),
        },
        "low_control_completion_band": {
            "contaminated": True,
            "by_tau": {str(k): v for k, v in band.items()},
            "note": (
                "P(success | control<tau) over ALL passes -- the 'technically complete, functionally "
                "lost' read. Failed passes cluster at low control via the end_xy selection, so this is "
                "CONTAMINATED and kept distinct from the clean false-alarm headline; never conflated."
            ),
        },
        "scope_note": _SCOPE_NOTE,
        "n_passes": len(df),
        "n_completed": int(is_success.sum()),
    }


def _readme(metrics: dict, corpus_label: str) -> str:
    return (
        "# Pass-risk calibration -- real-data validation\n\n"
        f"Corpus: **{corpus_label}** (owner-tier Gradient Sports WC2022). Raw per-pass positions are NOT "
        "committed (see the gitignored `pass_scores.parquet`); only these aggregate rates are.\n\n"
        f"**{_SCOPE_NOTE}**\n\n"
        "## Headline (leakage-free): completed-pass false-alarm rate `P(control < tau | completed)`\n"
        f"{metrics['headline_false_alarm_rate']}\n\n"
        "## Optimistic (reads failed passes -> leakage-inflated; not the headline)\n"
        f"- AUC(is_success, control): {metrics['optimistic']['auc']}\n"
        f"- ECE: {metrics['optimistic']['ece']}\n- reliability slope: {metrics['optimistic']['reliability_slope']}\n\n"
        "## Low-control COMPLETION band (CONTAMINATED, not the headline)\n"
        f"{metrics['low_control_completion_band']['by_tau']}\n"
        f"{metrics['low_control_completion_band']['note']}\n\n"
        "## Limitations\n"
        "- Selection bias (unfixable): only attempted passes are observed.\n"
        "- Control != completion: pitch control is a positional model; the reliability curve is a mapping.\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Pass-risk calibration artifact from the persisted pass scores.")
    ap.add_argument("--pass-scores", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--expect-commit", default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    prov = git_provenance()
    require_clean_tree(prov, allow_dirty=args.allow_dirty)  # THIS tree
    refuse_dirty_upstream(args.pass_scores, args.expect_commit)  # the upstream it derives from

    df = pd.read_parquet(args.pass_scores)
    metrics = compute_pass_risk_metrics(df)
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
