# TF-58 D2 layer-a memory — per-variant share split (reduce-memory follow-up)

Follow-up to ADR-112. Surfaced by the authoritative DGX run at `fcca558`: D2 `--layer a --level baseline` OOMs (6 of 14 workers OOM-Killed at 119 GiB). ADR-112 fixed the REDUCE combine; this is a distinct, newly-exercised defect in the per-worker MAP share-write AND the baseline combine. D1 artifacts (`derivation.json`, `_provider_params_generated.py`) are intact and reused; **D1 does NOT re-run.**

## Problem (measured)

`_layer_a` baseline runs `match_tables(loaded, base, variants=_post_preparation_variants(base), …)` — one reused-signals pass that emits the full 126-column per-window melt for **~13 post-preparation variants** (`min_observed_fraction`, `welch_segment_s`, `vc_epsilon` sweeps; spec 8.4), stacked into one frame tagged by the `variant` column (`LEAD_COLUMNS` = provider, match_id, variant, table). `write_worker_partial` then `pd.concat`s that worker's whole stack (~500K rows/match × ~91 matches ≈ **~45–50M rows, object ≈ 56 GiB/worker**, confirmed by two survivors at 55–56 GiB RSS). 14 workers fanned concurrently → ≫ 119 GiB → OOM. Latent second hazard: the baseline COMBINE ingests the same ~13× stack corpus-wide (~490M rows); categorical alone (ADR-112) does not tame 13×.

Not caught earlier: ADR-112's audit scoped to `combine_workers`; the per-worker share-write was never flagged, and **D2 layer-a had never run at full corpus** (the first authoritative attempt died at d1_reduce).

## Every consumer of the layer-a combined table (derived, evidence both sides)

The full melt is REQUIRED — a projection to a column/row subset is impossible, because the confirm reads ~the whole melt:

- **reliability objective** — `CoordinationReliabilityObjective._read:296` (filters `variant`), `reliability_over_folds:245` / `_reliability_weighted_over_providers:220`: reads `provider`, `match_id`, `table` ∈ {pair, spectral, cluster_team}, the team key, and the 8 `_FAMILY_COLUMNS` metric columns, **per variant**.
- **confirm hypotheses** — `_confirm:672` runs `evaluate_hypotheses(split_tables(joint_frame))` UNCONDITIONALLY; H1–H7 (`_coordination_hypotheses.py`) read ~the full melt: tables `pair`, `pair_phase`, `spectral`, `cluster_team`, `rsi`, `windows`, `rsi_switch_times`, `possession_changes`; cols `level`, `window_kind`, `signal_a`, `signal`, `axis`, `phase_index`, `period_id`, `coord_rp_mean_deg`, `coord_xc_r_at_max`, `coord_xc_lag_s`, `coord_vc_pct_anti_phase`, `terminal_action`, `attacking_team_id`, `team_a_id`, `team_b_id`, … — for the ONE selected joint-point variant. The `_has(...)` guards (ADR-032) skip silently when columns are absent → a lossy table would flip `gate_cleared`/`fallback_reason` with every gate green (the rejected-A failure mode).
- **population** — `_scored_pairs:474` reads `[provider, match_id]`; `corpus_visibility:685` (ADR-038).
- **combine** — `combine_workers` canonical sort `_ORDER_COLUMNS` = provider, match_id; completeness is MANIFEST-based (row-count-independent).

(Reader set derived by grep over `calibrate_coordination.py` + the `_coordination_hypotheses` call graph; `_joint_source:720` / `_JointObjective:708` included. Credit: the spec-review caught that the first draft's hand-listed reader set omitted the confirm — the completeness floor.)

**Conclusion: the full melt must be retained per variant. The memory fix must shrink the WORKING SET (one variant at a time), not the data.**

## Decision — C: per-variant shares + per-variant combines

Keep the single reused-signals layer-a pass (it still builds signals once and emits all variants — the speed optimization is preserved), but **split the baseline worker's output into one share per `variant`, and combine per variant.** Nothing is dropped; the full melt is retained for every variant.

- **Share-write:** the baseline pass writes one share file per variant (`level_share_name(baseline, variant)`), each the full melt for that single variant (~39K rows/match × 91 ≈ **~3.5M rows ≈ ~4 GiB/worker**), materialised one variant at a time (read the per-match shards, emit per-variant; peak ≈ one variant, not 13×). `write_worker_partial` is invoked per variant (or a per-variant wrapper), never concatenating the 13× stack.
- **Combine:** `combine_levels` iterates the baseline's variants and calls `combine_workers(categorical=True)` per variant → each combine is one variant corpus-wide (~38M rows = D3-metrics scale, which the ADR-112 categorical combine already fits at ~2.72×). Peak = one variant, not ~490M rows. **The baseline level still contributes exactly ONE `summaries["baseline"]` entry** (the per-variant combines all share one shard generation and one population digest over the identical match set — asserted equal across variants, else refuse): so `summaries` stays keyed by LEVEL, and `generation_tokens:440` / `objective_id_for:430` (`objective_ids`) / `_population:478` (the `population` block) are UNCHANGED. The per-variant split is an internal combine/table-layout detail; it does not fan `summaries` out to 13 entries (D2-SPEC-05).
- **Read-path:** `level_combined_path` gains a variant dimension for the baseline variants (`level_combined_path(out, BASELINE_LEVEL, variant)`); the objective (`_frame_for`), `_joint_source`, `_confirm`, and `_scored_pairs` read the per-variant file instead of filtering `frame["variant"] == variant` on one stacked file. Each reads the FULL melt for its variant → correct.
- **Non-baseline preparation levels** are already single-variant (`base`, small, ~39K/match) — one share each, unchanged, and now consistent with the per-variant schema.
- **`_layer_joint`** emits a single joint variant → one share (not stacked); its combined file + read-path are made schema-consistent with the per-variant baseline files (the review's schema-split flag).

### Memory math (re-derived)

| Stage | Before (stacked) | After (per-variant) |
|---|---|---|
| worker share-write concat | 13× full melt ≈ 56 GiB/worker × 14 → OOM | 1 variant ≈ 4 GiB/worker × 14 ≈ 56 GiB total — still tight; materialise one variant at a time per worker so the per-worker peak is ~4 GiB, and (if needed) cap fan concurrency |
| baseline combine | ~490M rows (13×) → OOM | 1 variant ≈ 38M rows → ~29 GiB categorical (D3-metrics-proven) |

The authoritative re-run's per-variant worker-peak and per-variant baseline-combine peak-RSS are the explicit go/no-go (same discipline as ADR-112's metrics-combine go/no-go).

## Byte-identity

The full melt is retained, only its on-disk layout changes (one file per variant vs one stacked file). Every consumer reads the full melt for its variant → `derivation.json` (D1, reused), the reliability objective's per-fold values + selections, the confirm's `evaluate_hypotheses` result, and — because `summaries` stays LEVEL-keyed (baseline = one aggregated entry, above) so `population`/`objective_ids` are unchanged — the **WHOLE `calibration.json`** and `_provider_params_generated.py` are **byte-identical** (modulo the named volatiles `*timings*`/`stage_seconds`/`run_*`). The D-5 harness asserts the whole `calibration.json` minus those volatiles, not a named subset (so a `population`/`objective_ids` drift is caught, not just the decision fields — D2-SPEC-05). A-SPEC's silent-wrong risk cannot occur: nothing is dropped.

## Alternatives rejected
- **A. map-side projection to the reliability subset** — the confirm's `evaluate_hypotheses` needs ~the full melt (verified); projecting to a small P silently flips the calibration gate (all gates green), and a correct P ≈ the full melt (no memory win). Fatal both ways (spec-review BLOCKING).
- **B. stream the share-write only** — bounds the per-worker write but leaves the ~490M-row baseline COMBINE OOM.
- **Separate pass per variant** — loses the reused-signals speed (rebuilds signals per variant).

## Non-goals
- No change to `match_tables`, the reliability statistic, the CV scheme, `SWEEP`/`BASELINE`, or `calibration.json`'s schema.
- No projection / column- or row-dropping — the full melt is retained.
- No change to D1 (reused), D3 metrics, or numerics.

## Risks
- **A consumer reads the wrong per-variant file / a missed read-path site** → the per-variant read-path is derived + gated, and D-5 drives the FULL confirm (calibration.json byte-identity), which exercises every reader on the per-variant layout.
- **`_layer_joint` schema divergence** from the per-variant baseline files → a schema-consistency test across all per-variant combined files.
- **Per-variant worker peak still ~4 GiB × 14 ≈ 56 GiB** → materialise one variant at a time per worker (peak ~4 GiB/worker); confirm the fan peak at the authoritative re-run; cap concurrency if marginal.

## Validation / CI
- **D-5 (strengthened):** byte-identity of the FULL confirm — `calibration.json` (`hypotheses`, `gate_cleared`, `fallback_reason`, selections) AND the reliability objective — on a small synthetic corpus run through the stacked-vs-per-variant layouts; both pandas legs where runnable (D2 needs ruthless 0.7.0 → pd3/DGX; the split/read-path functions are pure-pandas, local). Plant: a read-path pointed at the wrong variant file → red.
- Per-variant read-path completeness (every `level_combined_path` reader uses the variant dimension) + `_layer_joint` schema-consistency, registered in the ADR-056 anti-rot meta-assertion.
- Per-variant worker-share + combine memory assertion (one variant « the stacked size). ruff CI-scope, bare pyright, green before the commit gate.
