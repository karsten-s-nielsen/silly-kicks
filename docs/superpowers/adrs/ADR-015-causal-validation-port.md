# ADR-015: Private causal-validation port for trained-model confounder testing

## Status
implemented (private port, 4.18.0) → PROMOTED to public silly_kicks/causal/ (PR-1, TF-19/ADR-037)

## Context
TF-17's novel claim is a goalkeeper-position confounder block on cross propensity. PR-B's shipped
validation (GK-block ablation + substitution probe) measures *surface movement*, not a causal
effect. Cao et al. (2025, arXiv:2505.11841) frame crossing causally via propensity-score matching
(R `Matching`). We need a paper-faithful causal test — ATT/ATNT on crosser-anchored opportunity
rows — without adding R or a new Python dependency, and without letting a research finding gate a
runtime feature.

## Decision
- A **private** `silly_kicks/causal/` package (private as `_causal/` until the TF-19/ADR-037
  promotion): pure numpy/sklearn matching estimators
  (`matching.py`) + a pure spell-based opportunity-row builder (`opportunities.py`). No public API;
  not imported by `silly_kicks/__init__`. Promote to public `silly_kicks/causal/` only when a second
  consumer (TF-19) lands.
- **1:1 NN propensity matching, with replacement, ties allowed, no caliper** (paper-faithful);
  logistic propensity on **standardized** covariates. **Abadie-Imbens (2006) matching SE**
  (Imbens & Rubin 2015, Ch. 19).
- **Two named approximations** (so a future production consumer knows what to revisit):
  1. `sigma^2(X)` via the J=1 within-treatment-group nearest neighbor.
  2. Matching is on the **estimated** propensity score; the fixed-matching-variable AI formula is
     **conservative** under estimated-PS matching (Abadie-Imbens 2016, *Econometrica*). Acceptable
     for a *reported* artifact.
- The treatment window is `(entry, min(entry+T, spell_end)]` (R3-M1): a **fixed `T` cap** keeps
  Z-exposure bounded (no spell-length confounding — and since `Y`'s window is already fixed, the
  `spell_end` clamp adds no duration->`Y` path), while the `spell_end` cap prevents misattributing a
  cross from a *later* re-possession phase. NOT the variable spell length (rejected — R2-H3
  confounder), NOT the surface model's 1 s frame horizon. The outcome is measured strictly **after**
  treatment (`(t_cross, t_cross+W]` treated; `(entry, entry+W]` control) to avoid reverse leakage
  (R2-M1); `Y` is not possession-clamped (treated/control windows are time-shifted — documented). The
  confounder set is the **7 paper confounders** (ball-geometry surface features excluded). GK
  missingness uses the **missing-indicator method**, not mean-fill. No causal claim is made when PS
  overlap (treated-within-control-PS-range; no density trimming) or post-match balance fails
  (`causal_claim_supported=False`).
- The GK-vs-placebo null is the **row-permuted GK block** (preserves GK marginals + within-block
  correlation). Note: row-permutation also breaks GK<->base-confounder correlation, so the null is
  *slightly conservative* vs a pure `Z`/`Y`-alignment null — standard for permutation nulls. The
  finding is **reported, never a ship/CI gate**; the only CI gates are the known-truth method tests
  (`tests/causal/`). A null causal finding is valid.

## Promotion (PR-1, TF-19/ADR-037, 4.47.0)

TF-19 is the second consumer the Decision anticipated, so the "one move" fires:
`silly_kicks/_causal/` becomes public `silly_kicks/causal/` (registered in the
`test_public_api_examples.py::_PUBLIC_MODULE_FILES` gate from day one). `matching.py`'s
estimators are unchanged (fit/match/ATT/ATNT/Abadie-Imbens SEs byte-identical);
`placebo_shift` gains the cluster-aware mode — `cluster_ids` + `_cluster_reassign`
whole-cluster reassignment-with-recycling (a permutation over clusters, not a cluster
bootstrap) with the `permutation_unit` reported. `opportunities.py` is parameterized with the **full builder surface enumerated
now** — a frozen `OpportunityConfig` carrying `treatment_type_names`,
`outcome_type_names`, `outcome_result_ids`, `outcome_window_seconds`,
`outcome_window_anchor_inclusive`, `exposure_window_seconds`, `max_spell_seconds`,
`confounders`, `gk_block`, `domain`, and `extractor`. The load-bearing widening is a
**result-conditioned, anchor-inclusive outcome axis**: `_label_outcome` gains
`result_id`-filtering and an anchor-inclusive window (`ts ≥ anchor`), which is what makes
the ADR-037 §3.3 shot arm expressible purely as builder arguments (`shot_arm_config`) —
the own-result-only form was structurally degenerate for controls (control `Y ≡ 0`) and is
banned. The xCross configuration is preserved as the `config=None` default-constants path,
so the `tests/causal/` known-truth gates stay green unmodified; a regression guard asserts
the parameterized builder reproduces the xCross default **byte-identically**. The clustering
fix (match-level / whole-cluster-reassignment placebo bands, not row-i.i.d. permutation) applies to
both causal legs.

## Consequences
- No new runtime dependency; `import silly_kicks` stays light.
- The harness is maintainer-re-runnable on any corpus; its report is bundled, not recomputed in CI.
- If TF-19 needs the estimator, the private port is promoted (one move), not rewritten — **done in PR-1** (above).
