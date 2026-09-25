# ADR-105: Vectorized spearman kernel, scorer batching & bounded-memory streaming

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-09-25 |
| **Cycle** | PR-S199 (4.127.0) |
| **Spec** | `docs/superpowers/specs/2026-09-24-tracking-perf-vectorized-kernel-and-streaming-design.md` |
| **Plan** | `docs/superpowers/plans/2026-09-24-tracking-perf-vectorized-kernel-and-streaming.md` |
| **Extends** | ADR-008 (pitch-control cache), ADR-076 (numba bit-identical parity pattern), ADR-103 (frame-memory category + batched pitch-control seam), ADR-068/073 (`group_rows` + scale guard), ADR-057 (pandas-major span), ADR-043 (counterfactual-cache landmine) |
| **Retrain / re-materialize** | **NONE.** Every change is value-neutral or parity-gated byte-identical (`np.array_equal` / `assert_frame_equal`, max |Δ| exactly 0). C4-free. |

## Context

The lakehouse per-unit tracking-scorer OOM handoff (`_reviews/2026-09-24-sk-tracking-scorer-memory-perf-handoff.md`) profiled a GS half at ~3.4 GB peak with `rest_defense` at **361 s**. ADR-103 (4.125.0) shipped the frame-memory pass (F1a category dtypes, F5 cache LRU, F2 the *seam* `compute_pitch_control_batch` as a per-frame loop, F6). This cycle delivers the CPU + bounded-memory work ADR-103 deferred, all no-retrain (the owner-set "every optimization that does not require retraining" cycle; the retrain-carrying F1b/id→category is deferred to a later cycle).

## Decision

1. **A vectorized cross-frame spearman kernel** (`tracking/pitch_control/_spearman_batch.compute_spearman_batch`) sits behind the public `compute_pitch_control_batch`. The "after-TTI" combine is single-sourced (`_spearman._spearman_combine`, the ADR-102 idiom), and the win comes from ONE `compute_tti` over all frames' players CONCATENATED — TTI is per-`(player, target)` element-wise, so flattening frames is bit-identical and uses the same numba/numpy path. **Byte-identity precondition (load-bearing):** the influence `.sum(axis=0)` is order-dependent, so the kernel preserves each frame's own valid-player row order; real per-frame counts stay below numpy's 128-element pairwise-summation threshold, so the reduction is sequential and a masked-to-0.0 padding row is an exact no-op.

2. **The whole-unit scorers route their pitch control through the batch, in bounded chunks.** `rest_defense` layer-2 (its 4 surfaces/sample) — the 2 `compute_threat_pc` legs via a new `compute_threat_pc_batch` (over EXPLICIT frame slices, so the keeper-removed counterfactual frame is scored on its own content — the ADR-043 landmine that forbids the frame-keyed cache), and surf_a + gk_influence via a warmed per-chunk cache. `value_off_ball_runs` — its per-action canonical `cache.surface` via a warmed per-chunk cache. Both are **byte-identical** (goldens + the existing value tests) and **chunked from the start** (VKS-PLAN-07: never the ~1000-surface all-at-once regression ADR-103's F3-drop avoided).

3. **F4 is internal auto-batching, NOT a consumer-driven scorer primitive.** The action→frame link is global over the unit, so no consumer can score sub-unit; a per-batch scorer API could not reduce peak below the frame-hold and would be speculative debt (`feedback_speculative_api_surface_is_debt`). Each PC-consuming scorer (`compute_rest_defense`, `value_off_ball_runs`) gains a keyword-only `batch_size` (default bounded 64; `None` = whole-loop); output is byte-identical for any `batch_size` (invariance-gated — the merge prerequisite for the default flip, VKS-SPEC-03). `defensive_credit` + `gk_decision` are pitch-control-free (audited) — not routed, not given `batch_size`.

4. **Two ADR-103 shipped-code fixes.** `PitchControlCache.warm` groups the frames ONCE (`_dispatch.batch_from_groups` takes a pre-built `RowGroups`) instead of the double-`group_rows`. The `_key` per-call scan is RETAINED — building the key from the request's frame-key instead of the frame's raw values would risk a cross-dtype cache MISS; the scan is correctness, not just perf (a discovered constraint the plan's T5 Step 3 had not anticipated).

5. **`group_rows(observed=True)`** — defensive: ADR-103 introduced category frame columns, and on a categorical key the pandas-2 `observed=False` default injects phantom empty groups and is pandas-major-dependent (ADR-057). Byte-identical for the current non-categorical keys.

6. **A DAS cost guardrail** — `estimate_das_cost` + a one-time opt-out `DasCostWarning` before a large all-frame `get_das` run (~394 h/season, ADR-014). Advisory; DAS values unchanged. Independent of the kernel work (owner single-cycle scope; liftable).

## Consequences

- rest_defense pitch control now batches all 4 surfaces/sample (the F3 CPU win); off_ball batches its per-action surface; per-unit peak stays O(batch) surfaces (F4). Byte-identical output for every consumer/provider.
- No VAEP/tracking retrain, no re-materialize, C4-free (no new aggregator/backend/model; the kernel is internal behind the existing public seam).
- **Deferred to the retrain cycle:** F1b float32 coordinates + id→category (value-neutral through `id_compat` but a perf regression + a `group_rows` behavior change) + off-frame provenance.

## Alternatives

See spec §12. Chiefly: F4 (b)/(c) consumer-driven primitive rejected (global-link constraint); id→category deferred (spike negative); reviving F3 as warm-routing rejected (ADR-103 measured it a memory regression).
