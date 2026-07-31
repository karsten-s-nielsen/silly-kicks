"""W5 (4.45.0): keeper discrimination -- the real SP5 instrument (Jeff's Bravo/Navas reranking mode).

The program exists to fix v1's keeper NON-discrimination (near-constant per-keeper mean). Here we ask,
on the FAITHFUL metric: does xt_gk_v2 separate keepers where v1 was flat? Descriptive (V_opp fit on the
FULL cohort, R4). Discrimination = ICC on the ACTION-level values grouped by player_key (within-keeper
replication), NOT collapsed per-keeper means (that would be degenerate, R2). Per-keeper mean is used only
for the ranking. CV is reported secondary (unstable near zero mean). Honest-reporting: report whatever it
shows (§3) -- if v2 is still keeper-flat, that is the finding.
"""

from __future__ import annotations

import pandas as pd

# Delete-and-depend (TF-19 PR-3): the two statistics were lifted VERBATIM into the library
# so gkdv/ -- which cannot import from scripts/ -- shares one body. `keeper_spread` was
# renamed `group_spread` at lift time (nothing in it is keeper-specific).
from silly_kicks._group_metrics import DEFAULT_MIN_N, group_spread

_MIN_N = DEFAULT_MIN_N  # a-priori: min distributions per keeper for a stable within-keeper term


def _report(provider: str, variant: str, n_actions: int, v2: dict, v1: dict) -> str:
    from pathlib import Path

    def _rank(rows, top=8):
        return "".join(f"| {i + 1} | `{k}` | {mean:+.4f} | {n} |\n" for i, (k, mean, n) in enumerate(rows[:top]))

    lines = [
        f"# xT-GK v2 keeper discrimination — {provider} (FAITHFUL V_opp)\n",
        f"- rho variant: `{variant}` * GK-distribution actions: **{n_actions}** * min {_MIN_N} dist/keeper * "
        f"V_opp fit on FULL cohort (descriptive spread)\n",
        "\n| metric | ICC (action-level) | CV (means, unstable) | n keepers |\n|---|---|---|---|\n",
        f"| **xt_gk_v2** | **{v2['icc']:.4f}** | {v2['cv']:.3f} | {v2['n_keepers']} |\n",
        f"| v1 (c.xt_gk) | {v1['icc']:.4f} | {v1['cv']:.3f} | {v1['n_keepers']} |\n",
        "\n> ICC = between-keeper variance ÷ total (action-level; NOT per-keeper means). Higher = separates "
        "keepers more. CV = std/|mean| of per-keeper means (secondary; unstable when the metric mean ~ 0). "
        "**§3: report whatever it shows** — if v2's ICC ~ v1's, v2 is still keeper-flat on the faithful metric.\n",
        "\n### xt_gk_v2 top keepers (per-action mean; face validity — the owner's coaching eye)\n"
        "| # | player_key | v2 mean | n |\n|---|---|---|---|\n" + _rank(v2["ranking"]),
    ]
    out = (
        Path(__file__).resolve().parent.parent
        / "docs"
        / "research"
        / "xtgk_v2_construct_validity"
        / "keeper_discrimination.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    prior = out.read_text(encoding="utf-8") if out.exists() else ""
    out.write_text(prior + "".join(lines) + "\n---\n", encoding="utf-8")
    return str(out)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="xT-GK v2 keeper discrimination (W5).")
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

    from _loader_databricks import load_xtgk_cohort  # type: ignore[import-not-found]
    from validate_xtgk_possession_value import (  # type: ignore[import-not-found]
        _FRAME_PRESENT_COLUMN,
        _PRESSURE_COLUMN,
        _XG_COLUMN,
        prepare_cohort,
    )

    from scripts._driver import cohort_cache
    from silly_kicks.xtgk import EmpiricalTurnoverValue, MarkovPossessionValue, PressureLevels, compute_xt_gk_v2
    from silly_kicks.xtgk._retention import variant_key_for_provider
    from silly_kicks.xtgk._retention_features import extract_retention_features

    raw = cohort_cache(a.cohort_cache, build=lambda: load_xtgk_cohort(a.provider)[0])
    full = prepare_cohort(raw, pressure_column=_PRESSURE_COLUMN, frame_present_column=_FRAME_PRESENT_COLUMN)
    pl = PressureLevels().fit(full[_PRESSURE_COLUMN])
    v = MarkovPossessionValue().fit(full, xg_column=_XG_COLUMN, pressure_column=_PRESSURE_COLUMN, pressure_levels=pl)
    tc = EmpiricalTurnoverValue(min_support=30).fit(
        full, xg_column=_XG_COLUMN, pressure_column=_PRESSURE_COLUMN, pressure_levels=pl
    )
    from _loader_databricks import resolve_retention_model  # type: ignore[import-not-found]

    # ADR-036 two-leg SP5: every artifact must state WHICH rho produced it, else leg 1 (pre-fix rho)
    # and leg 2 (retrained rho) are indistinguishable in their own reports.
    variant = a.retention_weights or f"variant:{variant_key_for_provider(a.provider)}"
    rho = resolve_retention_model(a.provider, a.retention_weights)

    gk = full[full["is_gk_distribution"].fillna(False)].reset_index(drop=True)
    feats = extract_retention_features(gk, pressure_column=_PRESSURE_COLUMN)
    v2 = compute_xt_gk_v2(
        gk,
        possession_value=v,
        retention=rho,
        turnover_cost=tc,
        pressure_column=_PRESSURE_COLUMN,
        pressure_levels=pl,
        retention_features=feats,
    )["xt_gk_v2"].to_numpy()
    keys = gk["player_key"].to_numpy()
    v1 = pd.to_numeric(gk["xt_gk"], errors="coerce").to_numpy()

    # ADR-036 census (B5): the NaN-out count and the resolution provenance have no other scripted
    # source. NOTE the denominator: this cohort is POST-prepare_cohort (frame-absent null-pressure
    # rows are already dropped), so it is <= the spec's raw-domain figures.
    from silly_kicks.xtgk import finite_coord_mask
    from silly_kicks.xtgk._resolved_geometry import GK_GEOMETRY_SOURCE_COLUMN

    n_nan = int((~finite_coord_mask(gk)).sum())
    print(f"  census: {len(gk)} GK-distribution actions (POST-prepare_cohort); {n_nan} NaN-coord -> xt_gk_v2=NaN")
    if GK_GEOMETRY_SOURCE_COLUMN in gk.columns:
        print(f"  census: gk_geometry_source = {gk[GK_GEOMETRY_SOURCE_COLUMN].value_counts().to_dict()}")

    sv2, sv1 = group_spread(v2, keys), group_spread(v1, keys)
    print(f"provider={a.provider} n_actions={len(gk)} keepers(v2)={sv2['n_keepers']}")
    print(f"  v2 ICC={sv2['icc']:.4f} CV={sv2['cv']:.3f} | v1 ICC={sv1['icc']:.4f} CV={sv1['cv']:.3f}")
    print("wrote", _report(a.provider, variant, len(gk), sv2, sv1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
