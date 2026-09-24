# ADR-102: `ExpectedThreat.fit_from_counts` — distributed counts-based fit + public zone-binning contract

| Field | Value |
|---|---|
| **Date** | 2026-09-22 |
| **Status** | Accepted |
| **Deciders** | Karsten S. Nielsen (owner), luxury-lakehouse session (consumer), silly-kicks session |

## Context

`ExpectedThreat.fit(actions: pd.DataFrame)` requires a materialised DataFrame. The luxury-lakehouse ExT-grid producer fits a per-competition + a global grid over ~9.5 M actions across 28 competitions; with only `fit(actions)` it must pull each competition to the Spark driver via `.toPandas()` and `pd.concat` ~1 GB for the global fit — driver-bound, single-threaded, timed out at 0/28 grids, and a scale cliff. Before adopting sk (lakehouse ADR-085) the in-repo v1 producer used single-pass distributed ZoneCounter accumulation; that had to be reverted because sk exposed no counts-based fit.

Every quantity `fit` derives is a zone-count sum (per-zone shot/goal/move counts; per-(from,to)-zone successful-move counts), computable in one distributed `groupBy` and **additive** across partitions/competitions. sk should fit from those counts natively.

## Decision

Add `ExpectedThreat.fit_from_counts(*, shot_counts, goal_counts, move_counts, transition_start_counts, transition_counts, params=None)` (instance method, mutates `self`, returns `self`) that builds the four probability matrices from raw integer zone counts and runs the identical `value_iteration`, plus public `zones_of` / `flat_indexes_of` + `MOVE_TYPE_NAMES` / `SHOT_TYPE_NAME` so a producer replicates sk's exact binning and filters. The count→matrix logic is **single-sourced**: `_scoring_prob` / `_action_prob` / `singh_transition_matrix` are refactored into a pure `*_from_counts` core + a thin from-actions wrapper, and both `fit` and `fit_from_counts` call the same core.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Keep only `fit(actions)`; the lakehouse pulls to the driver | no sk change | driver-bound `.toPandas()` + `pd.concat` ~1 GB; timed out at 0/28; scale cliff | the problem |
| B. A parallel counts-based reimplementation of the matrices | isolated | two sources of the smoothing/normalisation that a future edit desyncs | drift risk |
| C. **chosen** — `fit_from_counts` calling single-sourced `*_from_counts` cores extracted from the existing builders | one implementation; `fit` provably byte-identical (parity-gated); distributed + additive | five counts (the filter distinction below) is more surface than a naive one-count API | — |

## Consequences

### Positive
- The producer fits the global grid from one distributed `groupBy` pass (summed per-competition counts) — no driver `.toPandas()`, no `pd.concat`; cost no longer grows with the corpus.
- The count cores are single-sourced, so `fit(actions)` and `fit_from_counts(aggregates)` cannot diverge.
- `zones_of` / `flat_indexes_of` + the membership constants make sk's binning a documented, test-pinned contract instead of a reverse-engineered one.

### Negative
- `fit_from_counts` is a cross-repo consumer contract keyed on the exact zone-binning + action-type filters — a change to either is a coordinated bump.
- Five counts (not the handoff's naive four/one) — because the `_action_prob` move population and the Singh denominator are DIFFERENT (below); the extra `transition_start_counts` is essential, not incidental.

### Neutral
- Purely additive: `fit`/`rate`/`interpolator`/`values_at_points`/`destination_profiles`/`to_dict`/`from_dict` and the SK-xT-1 `singh_counts` frozen-oracle parity are byte-identical. No VAEP/tracking retrain, no re-materialize, C4-free. No new bundled artifact.
- Singh (count-based) transition only — a KDE `params` raises (KDE is a density smooth of raw rows, not a count aggregate).

## Notes

**No smoothing/prior** (correcting the handoff): sk's matrices are raw counts through `_grid._safe_divide` (`np.divide(a,b, where=b!=0)`, 0 elsewhere) — `fit_from_counts` takes RAW counts and must NOT be pre-smoothed.

**The correctness crux — the `_action_prob` move-count ≠ the Singh denominator.** `_action_prob`'s `movematrix` masks NaN-**start** only (`_grid.py`), so a move with a valid start but a NaN end IS counted. `singh_transition_matrix` `dropna(subset=[start_x,start_y,end_x,end_y])` FIRST (`_transitions.py`), so that same move is NOT in its denominator. Therefore the contract carries **three** move aggregates: `move_counts` (valid start, → `_action_prob`), `transition_start_counts` (valid start+end, → the Singh row denominator), `transition_counts` (successful, valid start+end, → the Singh numerator). A single `move_counts` for both would silently diverge on any valid-start/NaN-end move — guarded by a named boundary fixture in `tests/xthreat/test_fit_from_counts.py` (plus the `_safe_divide` 0-branch and the off-pitch-end clamp).

**y-inverted flat index (ADR-041).** `flat_indexes_of` = `(w-1 - zone_y)*l + zone_x` — the transition matrix's row/col ordering (row 0 = pitch top). A producer MUST index `transition_counts` with this exact formula.

## Amendment (corpus-driver load seam) — `fit_from_counts` backs an in-repo resumable xT count pass

The lakehouse Spark producer was the motivating consumer, but the same additive-counts property makes
an xT fit a first-class corpus pass for the `scripts/` drivers too (ADR-052 D15). `scripts/_xt_corpus`
adds `xt_count_pass` — an events-only `for_each` that writes one sparse `XtZoneCounts` shard per match
(`counts_to_frame` / `counts_from_frames` are the sparse round-trip) — and `fit_xt_from_count_pass`,
which sums the shards and calls `fit_from_counts`. Every xT-fitting driver
(`build_tf60_layer3_arm_values`, `measure_cover_shadow_argmax_agreement`, `_xtgk_comparability`,
`calibrate_xt_bandwidth`, `calibrate_tracking_defaults`) fits this way instead of materialising the
whole corpus's actions in memory: the fit resumes, records failures, and is byte-identical to a pooled
`fit(actions)` (the same SK-xT-1 oracle guarantee), so **no retrain**. No library change — this reuses
`fit_from_counts` unchanged.

## Related

- **Specs:** `docs/superpowers/specs/2026-09-22-expectedthreat-fit-from-counts-design.md`; `docs/superpowers/specs/2026-09-22-corpus-driver-load-seam-design.md` (the count-pass consumer)
- **Plans:** `docs/superpowers/plans/2026-09-22-expectedthreat-fit-from-counts-plan.md`; `docs/superpowers/plans/2026-09-23-corpus-driver-load-seam.md`
- **ADRs:** ADR-021 (xthreat package), ADR-100 (`to_dict`/`from_dict` seam — the counts round-trip reuses it), ADR-041 (raw-orientation / y-inverted storage), ADR-052 (the corpus-driver seam that consumes the count pass). Consumer: lakehouse ADR-085 (single-canonical xT surface).
