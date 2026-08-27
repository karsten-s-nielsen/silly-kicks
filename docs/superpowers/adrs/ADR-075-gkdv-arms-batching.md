# ADR-075: Batched GKDV arms + once-per-unit direction pinning

| Field | Value |
|---|---|
| **Date** | 2026-08-27 |
| **Status** | Accepted |
| **Deciders** | Karsten S. Nielsen (owner); lakehouse session (consumer / reviewer) |

## Context

The two GKDV physics arms (`delta_das`, `delta_threat_suppression`) are documented and implemented as **one-scored-frame** operations. The lakehouse `gkdv_writer` calls them once per scored frame in a pure-Python loop over thousands of scored-and-defending frames per unit, across 374 `(match, period)` units. Measured **> 45 min/unit**, which exceeds the drain's **2700 s per-unit watchdog** → every unit is abandoned with zero output. gkdv has therefore **never** completed a production run and is gated off (`GKDV_ENABLED = False`).

`accessible-space` is already batch-native: `get_individual_dangerous_accessible_space` scores many frames in one call, amortizing a large fixed per-call setup (frameify → grid → direction inference → model init) across the whole batch. gkdv's single-frame arms collapse each call to a scalar and defeat that path one frame at a time. A sizing measurement (synthetic DAS-scoreable unit, `accessible_space==2.0.15`) put numbers on it: looped-per-frame vs one batched call is **9.2× / 23.4× / 39.7×** at N = 10 / 30 / 60 (fixed setup ≈ 0.6 s, true per-frame solve ≈ 6–8 ms), so the speedup *grows with N* — extrapolated to a ~2000-frame unit, ~2600 s → ~30 s (~90×). The same measurement showed `get_individual_das` is **~630×** the per-frame cost of `compute_threat_pc` (653 ms vs 1.03 ms), so the DAS arm alone dominates the wall.

`spearman` is the only GK-aware pitch-control method (`GkdvParams._GK_AWARE_METHODS`), so the win cannot come from a cheaper method — it must come from amortizing the per-frame setup, i.e. batching.

## Decision

Add batched arm entry points `delta_das_batch` / `delta_threat_suppression_batch` that make **one** `accessible-space` call per leg over a unit's scored frames; make the single-frame arms thin wrappers over them; pin attacking direction **once over the unit** (robust, and free because gkdv has no persisted output); and build **no vectorized spearman kernel** — the threat arm is 0.16 % of the cost, so batching the DAS arm is the whole win.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Lakehouse-side levers (raise `possession_stride`, add workers, raise the watchdog) | No silly-kicks change | A higher stride is a coarser sampling of a per-player evaluative model (a governance-reviewable methodology change); more workers/timeouts just throw compute at an inefficiency | Degrades the science or brute-forces the cost; does not fix the actual per-frame-setup waste |
| B. Batch DAS + a vectorized/fft spearman pitch-control kernel | Speeds up both arms | The threat arm is measured at 0.16 % of the wall (~2–4 s/unit); a batched kernel means ragged rosters + a new numba path + parity gates | Measured YAGNI — buys nothing that clears the watchdog, for real risk |
| C. **Batch DAS + batch-first threat API (thin loop), single-frame arms delegate, once-per-unit pin** | Fixes the actual bottleneck; uniform per-unit caller API; bit-exact amortization; no kernel | ~15 lines of genuine duplication removed via delegation (required reworking 3 library-free structural tests to the batch seam) | — |

## Consequences

### Positive

- gkdv becomes viable at production scale (~90× at unit scale; ~2600 s → ~30 s per ~2000-frame unit), so the lakehouse can re-enable `GKDV_ENABLED`.
- Once-per-unit direction pinning is strictly more robust than the historical per-frame pin (a single frame whose team mean-x argmin flips no longer flips the counterfactual's direction), at zero cost.
- The 0.0 → NaN reduce (`sum(min_count=1)`) fixes a latent fictional-zero on a non-simulatable frame, in the single-frame arm too.

### Negative / maintenance

- The mixed-batch NaN is a **version-pinned third-party contract** (a non-simulatable frame is NaN'd, not raised, inside a batch — verified at `accessible_space==2.0.15`, and already relied on by `add_das`'s single-pass call). Pinned by `tests/tracking/test_das.py::test_get_individual_das_mixed_batch_nans_bad_frame_not_crash`; if a future version changes it the fix belongs in `get_individual_das` (protecting `add_das` too), not a gkdv-local pre-filter.
- Delegating the single-frame arms to the batch changed which `_das_port` seam they call (`team_das` → `team_das_by_frame`), so three library-free structural tests were reworked to stub the batch seam.

### Neutral

- The two arms are **independently scoreable**: DAS structurally needs velocity + a possession-inferred direction (can be `DasUnscoreableError` → NaN), while `compute_threat_pc` takes `attacking_team_id` explicitly and reads no `team_in_possession` (always scores). No symmetric guard.
- The threat arm's ~1 ms/frame rests on the already-shipped 4.92.0 (ADR-068) pitch-control loop-invariant hoists.

### Known limits (stated, not discovered)

- The amortization oracle proves batch == loop **given the same direction**; it does not prove once-per-unit is "more correct" than per-frame — the pathological-frame test shows the pin is *stable*, not ground-truth.
- Batch-vs-loop bit-exactness is a property of `accessible_space==2.0.15`; a bump lands as a test delta (pinned, version-noted).

## Related

- **Specs:** `docs/superpowers/specs/2026-08-27-gkdv-arms-batching-design.md`
- **Plans:** `docs/superpowers/plans/2026-08-27-gkdv-arms-batching.md`
- **ADRs:** builds on ADR-043 (gkdv arms), ADR-063 (velocity-availability), ADR-068 (4.92.0 pitch-control hoists that make the threat arm cheap)
- **External references:** `accessible-space` 2.0.15 (`get_individual_dangerous_accessible_space`)

## Notes

**No retrain, no re-materialize:** gkdv has never produced output, so the once-per-unit pin and the 0.0 → NaN change alter no persisted value. Additive public API; C4-free (no new action-coupled aggregator).

**Lakehouse handoff-back (enacted at re-enable, not in this change):** filter both legs identically (scored-and-defending, same row order) so `_assert_legs_aligned` passes; treat a NaN delta as "exclude", never 0; build a complete `attacking_team_id_by_frame` (silly-kicks fails loud on a missing key); then drop the per-frame loop and call each batched arm once per unit.
