"""W6 (4.45.0): secondary faithfulness audit report -- kappa sweep (reported for Jeff, NOT tuned) +
V-reward interpretation (deferred) + PEV dormant note. Owner-run; writes faithfulness_audit.md.

Per Jeff §3 / the honest-reporting guardrail: kappa=1 is the a-priori HEADLINE; the [1,2] sweep is
evidence for the Jeff kappa/turnover-weighting question -- never used to pick the headline.
"""

from pathlib import Path


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="xT-GK v2 faithfulness audit (W6): kappa sweep + notes.")
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
        help="Path to a rho artifact dir (model.json + SHA256SUMS). Overrides the provider variant "
        "(ADR-036 two-leg SP5 re-run).",
    )
    a = ap.parse_args()

    from _loader_databricks import load_xtgk_cohort, resolve_retention_model  # type: ignore[import-not-found]
    from validate_xtgk_possession_value import (  # type: ignore[import-not-found]
        _FRAME_PRESENT_COLUMN,
        _PRESSURE_COLUMN,
        _XG_COLUMN,
        prepare_cohort,
    )
    from validate_xtgk_v2 import construct_validity_scores  # type: ignore[import-not-found]

    from scripts._driver import cohort_cache
    from silly_kicks.xtgk._retention import variant_key_for_provider

    raw = cohort_cache(a.cohort_cache, build=lambda: load_xtgk_cohort(a.provider)[0])
    actions = prepare_cohort(raw, pressure_column=_PRESSURE_COLUMN, frame_present_column=_FRAME_PRESENT_COLUMN)
    rho = resolve_retention_model(a.provider, a.retention_weights)
    rho_label = a.retention_weights or f"variant:{variant_key_for_provider(a.provider)}"
    print(f"provider={a.provider} rho={rho_label}")

    sweep = {}
    for kappa in (1.0, 1.5, 2.0):
        s = construct_validity_scores(
            actions, xg_column=_XG_COLUMN, pressure_column=_PRESSURE_COLUMN, retention=rho, kappa=kappa
        )
        sweep[kappa] = (s["xt_gk_v2"]["auc"], s["lift"])
        print(f"  kappa={kappa}: v2_auc={s['xt_gk_v2']['auc']:.4f} lift={s['lift']:+.4f}")

    lines = [
        "# xT-GK v2 secondary faithfulness audit (W6)\n",
        "\n## kappa sweep (faithful V_opp, possession-bound; provider "
        f"`{a.provider}`) — REPORTED, not tuned (§3: kappa=1 is the a-priori headline)\n",
        "| kappa | xt_gk_v2 AUC | lift |\n|---|---|---|\n"
        + "".join(f"| {k} | {auc:.4f} | {lift:+.4f} |\n" for k, (auc, lift) in sweep.items()),
        "\n> kappa scales the turnover term `dzv = -(1-rho)*kappa*V_opp`. With the faithful (small) V_opp, "
        "raising kappa adds more of a term that (per W4) drags the metric below `rho*dV` alone, so a larger "
        "kappa does not help; **kappa=1 stays the headline** (not chosen to optimise this, it is the default). "
        "The kappa/turnover-weighting is a genuine question for Jeff, given the faithful V_opp shifts the balance.\n",
        "\n## V reward interpretation — DEFERRED (flagged for owner/Jeff, not re-implemented here)\n",
        '> V uses **`E[first-shot xG]`** (our Singh-spirit reading); Jeff §2.1 says *"expected threat over '
        'the remainder of the possession."* First-shot vs cumulative-remainder is a real interpretation '
        "fork that may relate to V's weak realized-xG out-of-sample correlation (Spearman 0.03-0.06). "
        "**Deferred**: re-implementing V is out of scope for this release (a separate decision if it "
        "matters) — surfaced for the Jeff conversation, not silently changed.\n",
        "\n## PEV dormant (note)\n",
        "> PEV is 0 (`p'=p`; receiver-pressure `q` deferred per Jeff §8-step-7), so the metric currently "
        "carries no pressure-value-added term — faithful to his sequencing, noted for completeness.\n",
    ]
    out = (
        Path(__file__).resolve().parent.parent
        / "docs"
        / "research"
        / "xtgk_v2_construct_validity"
        / "faithfulness_audit.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(lines), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
