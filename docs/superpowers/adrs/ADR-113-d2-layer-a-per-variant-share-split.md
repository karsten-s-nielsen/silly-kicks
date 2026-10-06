# ADR-113: TF-58 D2 layer-a per-variant share split

| Field | Value |
|---|---|
| **Date** | 2026-10-07 |
| **Status** | Accepted |
| **Deciders** | owner (Karsten); implementing session |

## Context

The reduce-memory cycle (ADR-112) fixed the corpus-reduce OOMs via the categorical combine. Its authoritative DGX re-run (at commit `fcca558`) then surfaced a DISTINCT, newly-exercised defect: D2 `calibrate_coordination --layer a --level baseline` OOM-Killed 6 of 14 workers at 119 GiB. D2 layer-a had never run at full corpus before (the prior attempt died at d1_reduce), so this was its first exposure.

Measured root cause: `_layer_a` baseline builds `match_tables(base, variants=_post_preparation_variants(base))` — one reused-signals pass that emits the full 126-column per-window melt for ~13 post-preparation variants (`min_observed_fraction`/`welch_segment_s`/`vc_epsilon` sweeps), stacked into one frame tagged by `variant`. `write_worker_partial` then `pd.concat`s that worker's whole stack (~45–50M rows, object ≈ 56 GiB/worker; two survivors measured 55–56 GiB RSS); 14 fanned concurrently → OOM. The baseline COMBINE ingests the same ~490M-row stack corpus-wide — categorical alone (ADR-112) does not tame 13×.

ADR-112's audit scoped to `combine_workers` (the reduce); this is the per-worker MAP share-write + the per-level combine.

## Decision

Per-variant share split (spec/plan `2026-10-07-tf58-d2-layer-a-memory*`). The baseline keeps its single reused-signals pass, but:

- **Share-write** (`write_worker_partial_by_variant`): one share per `variant`, materialised one variant at a time (re-read the per-match shards per variant, filter `df["variant"]==v`) → per-worker peak ≈ one variant (~4 GiB), not the 13× stack.
- **Combine** (`_combine_baseline_variants`): `combine_workers(categorical=True)` per variant → per-variant combined tables (~38M rows each = D3-metrics scale, ADR-112 categorical fits). Folded to ONE `summaries["baseline"]` (`_fold_baseline_summary`): the variants share one shard `generation` + one `population_digest` (asserted equal, else refuse — also fail-closes a cross-run corruption), `stage_seconds` summed across variants, `pass` reset to the canonical level name → `generation_tokens`/`objective_id`/`_population` UNCHANGED (D2-SPEC-05).
- **Read-path**: `level_combined_path(out, level, variant)`; `_frame_for`/`_joint_source`/`_scored_pairs` read the per-variant file.
- `_SHARD_SCHEMA_VERSION` → `tf58-d2-3`.

The FULL melt is retained per variant — nothing is dropped — so the confirm's `evaluate_hypotheses` reads correctly and the WHOLE `calibration.json` is byte-identical (proven by a golden captured from the pre-C `fcca558` stacked code + live differential reducer tests). D1 is reused (not re-run).

## Alternatives considered

| Option | Why rejected |
|---|---|
| **A. map-side projection** to the reliability-consumed column subset | FATAL (spec-review BLOCKING): the confirm's `evaluate_hypotheses` (`_confirm:672`, unconditional) reads ~the full melt; projecting to a reliability subset silently flips `calibration.json`'s `gate_cleared`/`fallback_reason` (the `_has` guards skip, no crash), and a CORRECT projection ≈ the full melt (no memory win). |
| **B. stream the share-write only** | bounds the per-worker write but leaves the ~490M-row baseline COMBINE OOM. |
| **separate pass per post-prep variant** | loses the reused-signals speed (rebuilds signals per variant). |

## Consequences

### Positive
- The D2 layer-a OOM is fixed with the full melt retained → no metric/statistic/schema change; whole `calibration.json` byte-identical; D1 artifacts reused.
- The agree-or-refuse fold fail-closes a genuine cross-run share corruption.

### Negative / watch
- Per-variant worker-share peak + per-variant baseline-combine peak-RSS are UNVALIDATED locally; the authoritative re-run's peaks are the explicit go/no-go (collective 14×~4 GiB ≈ 56 GiB ≈ 47% of 119 GiB — one-variant-at-a-time keeps the per-worker peak ~4 GiB; cap fan concurrency if marginal).
- Per-variant occlusion/confirm artifacts are not comparable to a pre-`fcca558` stacked layout without re-materialising (layout change).

## Owner decisions recorded
1. **Option C adopted; A (projection) rejected** — A silently corrupts the confirm's calibration (reviewer-BLOCKING) and defeats its own memory win.
2. **D2-SPEC-05 single-summary fold** — the per-variant combine folds to ONE `summaries["baseline"]` so `objective_ids`/`population` stay byte-identical; the D-5 harness asserts the WHOLE `calibration.json` minus named volatiles.
3. **IMPL-02: `_layer_joint` accept-via-test, NOT wrapped (2026-10-07)** — `_layer_joint` is single-variant by construction (`variants={"joint": …}`), so wrapping it in the per-variant writer is a functional no-op (one variant → one identical share, already bounded ~4 GiB — never part of the 56 GiB stack OOM). The plan's "wrap `_layer_joint.work` too" was over-broad; wrapping would be a spurious edit. Schema consistency across the per-variant baseline files, the prep-level files and the joint file is guaranteed BY CONSTRUCTION (all are `match_tables` output, one producer) and confirmed by `test_joint_variant_schema_matches_baseline`.

Related: ADR-112 (reduce-memory), ADR-111 (TF-58). Spec: `docs/superpowers/specs/2026-10-07-tf58-d2-layer-a-memory-design.md`.
