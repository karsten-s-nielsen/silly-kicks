# ADR-112: TF-58 coordination reduce-memory via categorical combine

| Field | Value |
|---|---|
| **Date** | 2026-10-06 |
| **Status** | Accepted |
| **Deciders** | owner (Karsten); implementing session |

## Context

The TF-58 coordination cycle (ADR-111) shipped D1/D2/D3 corpus drivers that were never run at full corpus scale (980 matches). The authoritative DGX run surfaced that the reduces OOM: a pass emits a per-window `match_tables` / `occlusion_metrics` melt, and its reduce `combine_workers`-concatenates every worker's whole-corpus share into one process before aggregating.

Measured, not estimated: D1 occlusion `occ_err` share = **465M rows @ 71 matches**, 33.9 GB object → OOM at 116 GiB (`RC=137`). D3 metrics melt = **38,914 (SK) / 65,102 (GS) rows/match × 980 ≈ 40M rows / ~88 GB object** (126 cols, 41 of them string) → OOM by WIDTH/dtype, NOT row count (the initial "~6B rows" estimate mis-read the 126 metric COLUMNS as a row multiplier). D2 `combine_levels` holds the same full-corpus melt ×11 levels. Separately, pass-b ran 166.6 s/match (budget ≤45 s): profiled to a gap-invariant ball-carrier re-inference, not inherent work.

`SCALE_GUARDED` (ADR-073) guards only in-process signal-building subquadratic-ness; it is structurally blind to reduce-memory.

## Decision

The `match_tables` reduces read each worker share with the string columns dictionary-encoded into a **shared, explicitly SORTED `CategoricalDtype`** (`_read_concat_categorical`, `categorical=True` on the D3 metrics/stoppage, D2 layer-a/b/confirm and D1 occlusion combines), and the categorical dtype PERSISTS through the reduce (combined table ~29 GB vs ~88 GB object; measured 3× metrics / 7× occlusion). The full melt is retained — no row projection, no schema change. Every reduce `groupby`/`pivot_table` pins `observed=True` (no-op on object; the categorical `observed=` default flips pd2 False / pd3 True). Byte-identity is proven, both pandas legs, by the D-5 harness (`build_report`, `reliability_over_folds`, and the D1 occlusion kernels object-vs-sorted-categorical). Two speed fixes ride along: the gap-invariant ball carrier is inferred once (F3, ~4× pass-b); `provider_bootstrap_se` resamples row-index arrays instead of 1000× `pd.concat` (F2). Both byte-identical.

One further, SEPARATELY-APPROVED reduce-statistic change also rides this cycle (owner decision 5, 2026-10-07): the occlusion per-bin CI bootstrap (`_bootstrap_weighted_median_ci`) is changed from a row-grain bootstrap to a MATCH-GRAIN cluster bootstrap (resample the matches in a bin, not its rows — `occ_err` rows within a match are not independent) plus a 20k-row seeded subsample cap. This **changes** `derivation.json`'s occlusion `ci_by_bin` values (the point estimate `err_by_bin` is UNCHANGED — still computed on the full bin). It is the fix for the D1-reduce hotspot: the old row-grain 400×-full-bin bootstrap ran > 1 h at corpus scale (465M-row `occ_err`), match-grain + capped is ~1 s per 1.5M-row bin. **For the occlusion CI, the D-5 harness therefore proves object == categorical of the NEW statistic, NOT new == old** — this is the one place this cycle is not byte-identical to the pre-change row-grain output, and it is here by explicit owner approval (decision 5), not silently.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. per-consumer map-side PROJECTION (emit only consumed grain) | large shrink | completeness is load-bearing + hard to prove; a dropped consumed cell fails SILENTLY; needed a `metrics.json` liveness schema change (drop `n_unique`) | superseded once measurement showed the OOM is string-WIDTH not row-count — categorical fixes it with the full melt retained |
| B. `union_categoricals` (append-order categories) | less code | append-order categories make `sort_values`/group order diverge from the object (lexicographic) path | not byte-identical; pinned an explicit `CategoricalDtype(categories=sorted(union))` instead |
| C. categorical combine + `observed=True` (chosen) | one mechanism fixes all 3 OOMs; full melt retained; byte-identical; no schema change | ~2× headroom at the metrics reduce (3× only) — a far larger corpus would want A | — |

## Consequences

### Positive
- All three corpus-scale OOMs fit 119 GiB with one mechanism; no metric-definition or artifact-schema change (liveness `n_unique` unchanged, computed on the full categorical table), and exactly ONE owner-approved reduce-statistic change (the occlusion per-bin CI bootstrap — decision 5; everything else byte-identical).
- Reduce-memory is now guarded (categorical-dtype + per-row-mem bound + the `observed=True` AST gate over a DERIVED coordination-script population), closing the `SCALE_GUARDED` blind spot.
- pass-b ~4× faster (F3); d1_reduce bootstrap no longer 1000× concats (F2).

### Negative / watch
- The metrics reduce shrinks only ~3× (29 GB base / ~58 GB concat peak, ~2× headroom). A far-larger corpus would re-approach the ceiling → the deferred projection (Option A) is the escape. Confirmed by the authoritative-run metrics-combine peak-RSS go/no-go.
- Occlusion per-bin CIs produced under decision 5 (match-grain + 20k cap) are NOT comparable to a row-grain CI from before this change (a single-match bin now yields NaN rather than a finite row-bootstrap CI). Any consumer comparing a new `derivation.json` occlusion `ci_by_bin` against a pre-change one must re-materialize first.

## Owner decisions recorded (2026-10-06)
1. **rev-3 pivot adopted**: categorical-only architecture; the rev-1/2 projection + liveness schema change are dropped.
2. **F3 budget superseded**: the parent TF-58 spec §5 goal-6 ≤45 s/match is superseded by the binding corpus **≤1 h / 16 workers** (parent spec l.272); pass-b ~45 s/match post-hoist reconciles against it (GS to be confirmed at the authoritative run).
3. **`_pre_index_frames` de-Arrow DEFERRED**: its Arrow-`__getitem__`/`slice_block_rows` residual is a shared-infra optimization (broad blast radius across ball-carrier/linkage consumers) → a separate future cycle.
4. **Metrics ~3× headroom accepted**: the projection (Option A) is deferred unless the corpus grows far larger; the authoritative-run metrics-peak RSS is an explicit go/no-go.
5. **Occlusion per-bin CI bootstrap: match-grain + capped (2026-10-07, VALUE-CHANGING, explicitly approved)**. During the authoritative DGX run the D1 `occ_err` reduce ran > 1 h single-thread — profiled to the per-bin CI bootstrap (`_bootstrap_weighted_median_ci`): 400 draws each resampling + weighted-quantile-sorting the whole million-row bin (O(400·n log n)). Owner approved **option A**: resample at MATCH grain (cluster bootstrap — the statistically-sounder unit, since `occ_err` rows within a match are not independent) and cap each bin's CI to a seeded 20k-row subsample (per-match-balanced so every draw's pool ≤ cap; point estimate untouched). This supersedes this cycle's "no reduce-statistic change" / "occlusion kernels byte-identical" claims FOR THE OCCLUSION per-bin CI ONLY (`derivation.json` `ci_by_bin` values change; `err_by_bin` and everything else unchanged). Measured ~1 s per 1.5M-row bin (occ reduce hours → minutes). Re-materialization of `derivation.json` under it is owed at the authoritative run.

Related: ADR-111 (TF-58), ADR-073 (`SCALE_GUARDED`), ADR-105 (vectorized batch), ADR-056 (anti-rot gate population). Spec: `docs/superpowers/specs/2026-10-06-tf58-reduce-memory-design.md`.
