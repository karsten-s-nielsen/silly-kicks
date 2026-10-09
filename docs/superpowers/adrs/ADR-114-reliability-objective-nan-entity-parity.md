# ADR-114: D2 reliability objective excludes NA team entities (D3 parity)

| Field | Value |
|---|---|
| **Date** | 2026-10-08 |
| **Status** | Accepted |
| **Deciders** | owner (Karsten); implementing session |

## Context

The authoritative TF-58 DGX re-run (at commit `fe449af`) cleared both prior OOMs — D1 reduce (76.6 GiB) and all 11 D2 layer-a levels (`Killed:0`) — then crashed in single-process D2 **layer-b**:

```
scripts/calibrate_coordination.py:233  np.unique(groups[finite])
TypeError: '<' not supported between instances of 'float' and 'str'
```

Call chain: `_layer_b` -> `_run_oat` -> `GridSearchStrategy.run` -> `CoordinationReliabilityObjective.evaluate` -> `reliability_over_folds` -> `_reliability_weighted_over_providers`.

Root cause (measured on the preserved combined parquet `out/d2/layer_a__baseline__v__base.parquet`, not inferred): the **spectral** table's `team_id` column (categorical) carries string team ids mixed with float NaN. The objective grouped the spectral table by `team_id` for a team-discrimination ICC, masking `finite = np.isfinite(vals)` on the **value** column only and never on the **team-id group key**, so an NA-team row with a finite value left a NaN in the group array and `np.unique` could not sort `str` against `float`.

The NaN is **correct data, not a resolution failure.** `_compute._possession_spectral_row` writes `team_id = pd.NA` for the Moura-2013 `possession` signal (1 = team A in possession, 0 = team B) — a match-level alternation series that belongs to no single team. Full-corpus crosstab:

- 266,208 of 6,655,200 spectral rows (4.00%) are NA-team; **100% are `signal == "possession"`**; **zero** non-possession rows are NA-team.
- All 12 per-team signals (centroid_x, stretch_index, team_length, convex_hull_area, ...) are 100% team-resolved — **no team data is unresolved or lost.**
- Of the reliability population (3,666 finite-value spectral rows), 162 (4.42%) are NA-team, **all 162 the possession signal**; 3,504 team-signal rows (100%) retained.

The crash was latent because layer-b had never run at full multi-provider corpus (prior runs died upstream) and no CI fixture combined a finite spectral value with an NA team in one row.

The **D3** report already handles this correctly: `scripts/_coordination_reliability.py` does `dropna(subset=["value", "entity"])`, so the teamless possession spectrum becomes its own `signal`-keyed construct recorded with an honest `unmeasurable` verdict (no team entity to retest). The D2 selection objective and the D3 report are contracted (A-35) to use "the SAME per-construct estimator"; on NA-entity handling they diverged. D2's was the defect.

## Decision

In `_reliability_weighted_over_providers`, exclude NA team keys from the per-provider mask, matching D3's `dropna` mechanism:

```python
finite = np.isfinite(vals) & pd.notna(groups)
```

A teamless unit cannot enter a team-discrimination ICC; dropping it is correct by construction, not convenience. This restores the A-35 "same per-construct estimator" parity between the D2 objective and the D3 report. `icc1` / `circular_reliability` and the `>= 2 distinct teams` guard then receive an NA-free group array.

## Alternatives considered

| Option | Why rejected |
|---|---|
| **Semantic filter** (`signal in TEAM_SIGNALS`) in D2 only | Equivalent on today's data (possession is the only teamless signal) but a *different* mechanism than D3's `dropna(entity)` — it would re-diverge the two estimators the A-35 contract binds together. Structural NA-entity exclusion mirrors D3 exactly. |
| **Resolve a team id for the possession rows** | There is no team to resolve — the possession series is teamless by construction (Moura 2013). Inventing one would fabricate a team observation. |
| **Leave possession rows in the team ICC** | Undefined: grouping a teamless series "by team" is not a team-discrimination measurement; it is also the crash. |

## Consequences

### Positive
- D2 layer-b/confirm run clean at full multi-provider corpus; D2 and D3 now use the same NA-entity handling.
- No team data dropped (measured: 0 team-signal rows excluded); the possession spectrum keeps its honest `unmeasurable` verdict in D3.
- D1 is untouched -> `derivation.json` byte-identical; the fix is read-side only, so the preserved D1 + layer-a shards are reusable.

### Negative / watch
- Value-changing for the spectral family vs a hypothetical crash-free old run (none exists): spectral reliability is now computed on team signals only, so the OAT selection recorded in `calibration.json` MAY shift. This is correctness-restoring, not an arbitrary change.
- The possession spectrum's own reliability treatment (D3 `unmeasurable`) is pre-existing behaviour, unchanged here.

## Owner decisions recorded
1. **Gold-standard investigation required before the fix** (owner, 2026-10-08): establish WHY the team id is NA and the exact fraction dropped before excluding anything. Done — teamless-by-construction, 4.42% of the finite spectral population, all possession, zero team data lost.
2. **Lighter process path** (owner, 2026-10-08): a localized correctness/parity fix (one function) + TDD regression + dual impl-review + owner commit gate; a full spec/plan is not warranted.

## Impl-review resolution (2026-10-08)

Two independent reviews: A APPROVE, B REQUEST-CHANGES; both findings folded (artifact was frozen across the round):

- **ADR114-01 (B, SHOULD-FIX)** — re-baselining the golden updated `d2_calibration_golden.json`, the regenerate script and `__provenance__`, but left `test_d2_layer_a_share_split_identity.py`'s test-8 docstrings still claiming "golden captured from pre-C (fcca558)" — a stale, contradictory contract. Resolved: the section comment + test docstring now state the golden pins the CURRENT expected output, and layout-neutrality is carried by the live differential tests, not this change-detector.
- **D2-REL-01 / ADR114-02 (A + B, CONSIDER)** — removing the anti-circular regenerate guard left the golden freely re-baselineable. Resolved with a freshness guard: `test_whole_calibration_json_byte_identical_to_golden` now asserts `golden["input_contract"]["objective_version"] == OBJECTIVE_VERSION`, so a future version bump without a regenerate fails loud rather than reading as expected drift.

Related: ADR-111 (TF-58), ADR-112 (reduce-memory), ADR-113 (D2 layer-a per-variant share split). Estimator parity: `scripts/_coordination_reliability.py` (D3).
