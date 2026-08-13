# Ghost box-constant unification — measured delta (D2)

`metrics.json` in this directory. Produced by `scripts/measure_box_constant_delta.py` at
`run_commit 968f819`, `run_tree_dirty false`, over the full 179-match pining corpus
(108 skillcorner + 64 gradientsports + 7 idsse), materialized at `run_commit 8d37540` with parity
asserted against the established `_loader_pining_to_cache.py` output for match 1886347.

## The count, and the ship claim it selects

| quantity | rows | share |
|---|---:|---:|
| total player-frame rows | 401,908,642 | — |
| **flipped by the unification** | **14,548** | **0.00362%** |
| ...from the depth boundary (`<` → `<=`) | 11,328 | 77.9% of flips |
| ...from the 1 cm band (20.15 → 20.16) | 3,217 | 22.1% of flips |
| ...from both together | 3 | — |

`n_flipped > 0`, so the release carries **the band/boundary split plus a before/after weights
comparison** — not the "unification, measured no-op" claim.

The split conserves exactly (3,217 + 11,328 + 3 = 14,548), which is the property
`test_measure_box_constant_delta.py` pins: any partition satisfies a bare `sum == total`, so the
attribution is asserted per cause rather than in aggregate.

**The dominant cause is not the constant.** 77.9% of flips come from the strictness change
`gr_x < 16.5` → `gr_x <= 16.5`, i.e. players standing exactly on the 16.5 m line; only 22.1% come
from widening the half-width by 1 cm. A reader who assumes "unifying 40.3 → 40.32" explains this
number would be wrong about three quarters of it.

## Behind-the-line: DECISION — retain unbounded (D6 / Task 7 Step 6)

`in_penalty_area_goal_relative*` has no `0 <= gr_x` guard, so points beyond the goal line count as
in-box. Measured on the same corpus:

| quantity | rows | share |
|---|---:|---:|
| `gr_x < 0` | 340,261 | 0.08466% |
| **`gr_x < 0` AND inside the y band** — would change under a clamp | **233,359** | **0.05806%** |

The second row is the decision-relevant one; `n_behind_line` alone is an upper bound, because a
behind-the-line point outside the y band is out of the box either way.

**Decision: retain the unbounded predicate for this cycle.** Not because the population is
negligible — it is **16.0× larger than the unification this cycle is re-fitting for** — but for
attribution. Task 8 re-fits ghost against the canonical constant; a predicate-shape change riding
along would make the resulting weights differ for two reasons at once, and the whole purpose of the
number above is to say what the constant change did. Clamping is also not a one-model change: the
signed `gr_x` reaches both `_ghost_gk` (`attackers_in_box`) and `_xcross_attempt` (feature #6), both
trained, and xCross is not otherwise re-fit here.

**Revisit trigger.** Take this up as its own cycle, sized as a two-model re-fit and republish, if
either holds: (a) a keeper-domain analysis shows behind-the-line rows concentrating in the GK box
rather than spreading across the pitch, or (b) the population grows materially on a corpus with more
GS extra-time or higher-noise tracking. Note some of these rows are the same off-pitch detections
`_loader_pining` already warns about (`N row(s) off-pitch beyond the S1 tolerance`, deliberately not
clamped), so a clamp would partly be papering over an upstream data-quality issue rather than fixing
a geometry bug.

**ADR-050's contract will not enforce this either way, and that is measured** — a lower bound
declares no new constant and the probe frame carries no behind-the-line player, so
`_feature_contract_block()` is byte-identical with and without the clamp. The discipline is manual;
this note is the record.
