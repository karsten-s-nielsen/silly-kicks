# ADR-009: TF-24 Optuna calibration harness

| Field | Value |
|---|---|
| **Date** | 2026-05-29 |
| **Status** | Accepted |
| **Deciders** | Karsten S. Nielsen; lakehouse review (multi-round); silly-kicks maintainers |

## Context

Three tracking defaults shipped as documented *engineering choices*, not empirically calibrated values: `infer_ball_carrier`'s `tolerance_m`/`beta`/`gamma`, `LinkParams.k3` (Link 2016 explicitly leaves `k1..k5` unpublished), and the off-ball-run `pre_seconds`/`min_displacement_m`. TF-24 calibrates them against real multi-provider tracking data. Constraints: the core library must stay pure (pandas-in/out, zero I/O) and dependency-light (`import silly_kicks` must not pull heavyweight optimisation deps); calibration data is gated (SkillCorner + IDSSE public on pining-for-the-data, Gradient Sports owner-tier, Databricks bronze for scale) and must never be committed; and the held-out objective must be an *honest* number (no leakage). The optimisation substrate (`ruthless-efficiency`, PyPI) provides `OptunaStrategy` + a `CachedObjective` invariant/patch pattern.

## Decision

Ship a new optional `silly_kicks.calibration` subpackage (pure objectives/CV/gates, behind a `[calibration]` extra, NOT imported by `silly_kicks/__init__`) plus a `scripts/` CLI + loaders that own all I/O. Stage 1 maximises carrier accuracy; Stage 2 minimises augmented-VAEP held-out Brier as a `ruthless` `CachedObjective`. xT is a **frozen exogenous artifact** fit once on a corpus disjoint from the calibration matches. The harness **recommends** values + emits an auditable manifest; it does **not** change the library default constants (that is a separate "apply" PR).

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Lakehouse `evolve` framework | Already in the lakehouse | Evolve is for code/structure evolution; single-scalar Bayesian opt is Optuna-shaped | Wrong tool — handoff rubric |
| B. Per-CV-fold xT refit (textbook "refit transform in each fold") | Leak-free per held-out fold | Refits the 2 most expensive feature families ×fold; injects fold-structure variance into the metric; calibrates under a regime that never occurs at serving | xT is a fixed *exogenous extractor*, not part of the model under calibration; train–serve consistency demands a frozen grid |
| C. Frozen exogenous xT on a disjoint corpus (chosen) | Train–serve-consistent, zero leak, cleaner TPE signal, simpler invariant | Needs a disjoint corpus + a fail-closed id-space check | — |

## Consequences

### Positive
- Tracking defaults become empirically defensible + reattributable (manifest records silly-kicks/ruthless/xgboost versions, per-provider match-IDs, frozen-xT sha256 + corpus identity).
- The `CachedObjective` invariant/patch split gives a ~7× per-trial speedup; a deterministic-XGBoost cache-equivalence test (`assert_cache_equivalence`, 1e-9) proves the fast path equals an independent monolithic recompute.
- `import silly_kicks` stays dependency-light (CI-guarded subprocess test); calibration is opt-in.

### Negative
- A new cross-cutting optional dependency (`ruthless-efficiency[optuna]`) + a bounded `xgboost>=2.0,<3.0` (the 1e-9 cache gate rides on `tree_method="hist"` determinism, which a future major could change).
- The pining loader reimplements per-provider parsing glue (kloppy for SkillCorner/Sportec; a bespoke flatten for Gradient Sports events) that overlaps the lakehouse's bronze ingestion.

### Neutral
- The `scripts/` loaders map a kloppy Sportec `TrackingDataset` to the frames schema *locally* (see CLAUDE.md amendment) rather than through the library gateway.

## CLAUDE.md Amendment

ADR-004 (tracking-namespace charter) routes Sportec tracking through `silly_kicks.tracking.sportec` and the library's tracking kloppy gateway explicitly **refuses** `Provider.SPORTEC`. The IDSSE pining loader serves *raw DFL/Sportec XML*, which the library `sportec` converter does not parse (it takes already-normalised DataFrames). The calibration **loader** (`scripts/_loader_pining._kloppy_tracking_to_frames`, I/O glue — NOT library core) therefore uses `kloppy.sportec.load_tracking` to parse the XML and maps the result to `TRACKING_FRAMES_COLUMNS` locally. This is a scoped exception confined to `scripts/`; the library's ADR-004 routing is unchanged.

## Related

- **Specs:** `docs/superpowers/specs/2026-05-29-tf24-optuna-calibration-harness-design.md`
- **Plans:** `docs/superpowers/plans/2026-05-29-tf24-optuna-calibration-harness.md`
- **ADRs:** relates to ADR-004 (tracking namespace), ADR-005 (tracking-aware features), ADR-008 (pitch control / pitch-control-cache reuse)
- **External references:** `ruthless-efficiency` (PyPI 0.2.0); pining-for-the-data mock provider API; Link, Lang & Seidenschwarz 2016 (k3); Decroos/Van Roy xT

---

## Amendment (2026-06-08, SK-xT-3): xT bandwidth/resolution NLL objective

Extends the recommends-an-auditable-manifest-never-mutates-defaults harness to xT
(`silly_kicks/calibration/_xt_bandwidth_objective.py` + `xt_bandwidth_config`, ADR-021). It sweeps
`KDEParams.bandwidth` x `GridSpec` resolution x `adaptive` over the held-out transition-NLL
substrate (`compute_holdout_nll`) and recommends a `KDEParams`+`GridSpec`; the library default
`KDEParams.bandwidth=1.0` is untouched.

**Deliberate divergence from this ADR's CachedObjective pattern.** Stage 1/2 ship as a
`ruthless.CachedObjective` guarded by `assert_cache_equivalence`. SK-xT-3 instead uses a plain
duck-typed object with a resolution-keyed lazy cache and a hand-written warm==cold equivalence test,
because the resolution axis means the invariant is keyed by `(grid, fold)` — which a single
`prepare()` does not model. This is by design, not oversight.

**Gaussian-core re-pin (Chesterton's Fence).** SK-xT-3 re-pins the gaussian KDE numerics to a
vectorized implementation (shared `_gaussian_transition_from_grouped` seam) for a large per-trial
speedup. `grep` confirms `kde_smoothed_transition_matrix` has exactly one caller (`_model.py:123`)
and `ExpectedThreat` defaults to `singh_counts` (`_model.py:78`), so no shipped artifact (VAEP,
bundled weights, lakehouse) depends on the KDE numerics — the re-pin is safe. The gaussian path now
also stays finite/correct in the small-bandwidth regime where the previous sklearn-wrapper
underflowed to the mean-row fallback.

- **Specs:** `docs/superpowers/specs/2026-06-08-xt-bandwidth-calibration-design.md`
- **Plans:** `docs/superpowers/plans/2026-06-08-xt-bandwidth-calibration.md`
