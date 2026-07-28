# TF-30 cover shadows — invariant repair, clamp verdict, per-defender identity — design

**Date:** 2026-07-27
**Target:** silly-kicks — **version / PR-S / ADR numbers are assigned at commit-prep, not here.**
(rev 1 named provisional numbers; part-deux had already taken them. Three collisions in two days, all
from the same hedged "provisional X — do not pre-claim" form. A hedge is still a claim.)
**Status:** design **rev 3 — review round 2 applied in full (1 blocker / 3 major / 3 minor + 3 answers).**
Every finding re-verified against code before adoption; all held.
**Source:** Cascioli et al. (2025) audit handoff, 2026-07-27
(`D:\[Karsten]\Dropbox\[Microsoft]\Desktop\silly-kicks TF-30 handoff\`). TODO row **TF-30 (a)**.
**Scope class:** test repair + documentation correction + ONE additive aggregator-only column →
**C4-free (count stays 32), no retrain, no change to any shipped column's values, NO API change**

---

## 1. Problem

TF-30 is a **faithful** port of the paper — constant-by-constant. Nothing here corrects the physics.

### 1.1 The monotonicity invariant cannot fail

`_cover_shadows.py:907` clamps `score = max(threat_unblocked - threat_orig, 0.0)`;
`tests/invariants/test_cover_shadow_invariants.py:31-34` then asserts `>= -1e-9` on that clamped
column. Green by construction. `test_zero_blocked_implies_low_score` (`:55-66`) is **misnamed** — it
promises "low score" and asserts only non-negative, as its own comment admits.

### 1.2 A SECOND clamp, not in the audit

`:1109` — `delta = np.maximum(new_recv - old_recv, 0.0)`. `max_single_defender_blocking_score` is
non-negative by **two** independent mechanisms.

### 1.3 Non-negativity is STRUCTURAL — for the methods argued

`_voronoi_threat` partitions over `attackers_outfield` only (`:698`); `frame_reduced` removes only
defenders (`:892`). Under defender removal the partition **and** the `dangerous` set are invariant; only
the attacking pitch-control grid changes (`:691`). Monotonicity reduces to *"is attacking pitch control
pointwise non-decreasing when a defender is removed?"*

⚠ `compute_blocking_score` accepts **three** methods (`pitch_control/_dispatch.py:49`): `spearman`,
`voronoi`, **`fernandez_bornn`**. The reduction is argued for the first two. See §3.6.

The consequence: substituting `xT × pitch control` for the paper's SoccerMap CNN means our score
**cannot express "this defender's positioning made things worse"** — a reading the paper attaches to
the sign. The glossary says nothing about this.

### 1.4 Per-defender identity is computed, then discarded

`:1040-1052` (exact) and `:1112` (cheap) both know which defender produced the max and drop it.

---

## 2. Decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | **Keep both clamps. Repair the tests, correct the docs.** Scope stated in **documentation only**. | Chesterton's Fence — the invariant is a declared decision (`2026-05-10-...:605`). §1.3 covers `spearman`/`voronoi`; `fernandez_bornn` is settled **empirically** in §3.6, not by extending the proof. |
| D2 | **No change to any shipped column's values.** | Shipped 4.11.0 (Hyrum). |
| D3 | **Identity column is AGGREGATOR-ONLY, never in `cover_shadow_xfns`.** | `features.py:3784` feeds `_CS_COL_NAMES` into the VAEP factory; a player-id column is non-numeric. `das_source` precedent (ADR-043), guard `test_das.py:1933`. |
| D4 | **Identity ships only behind a MEASURED cheap-vs-exact agreement — measured at a sample that can resolve the rule.** | ρ ≥ 0.7 is a *rank* guarantee, near-silent about the **top** of the ranking, which is all this column uses. Precedent: this cheap path shipped a silent ADR-041 orientation defect that *"reached `max_single_defender_blocking_score` on every action"* (`:1060-1067`), caught only by final review. See §5.1. |
| D5 | **NA whenever there is no attribution to make**, not only when `n_lb == 0`. | §5.2. |
| D6 | **No API change of any kind.** No method rejected, no signature altered, no return type widened. | MAJOR 1 / O1. Rev 2 proposed asserting D1's scope "in code", which would reject or warn on `method="fernandez_bornn"` — an API break on a method the entry point accepts and serves today, **strictly worse than the column changes D2 forbids**. Hyrum applies to parameters, not only to columns. |

---

## 3. The invariant repair

**3.1** Re-point `test_blocking_score_non_negative` at the unclamped `threat_unblocked` vs
`threat_original` (both on `BlockingScoreResult`), calling `compute_blocking_score` per action.

### 3.2 The plant — direction AND target both specified

rev 1's plant (defenders into the Voronoi partition) left the test green *and* contradicted §1.3.
rev 2 fixed the direction — drop **attackers** at `:892` — but left the **target** unspecified, which
decides whether it works:

- Dropping a **non-dangerous** attacker (behind the ball) can leave the counted set nearly unchanged —
  its cells reassign to neighbours, some dangerous, so cells can even be **added** — and the PC change
  may not clear `-TOL_INVARIANT`. Plant stays green, proves nothing.
- Dropping a **dangerous** attacker with non-zero `per_receiver` threat is guaranteed negative: no cell
  is ever added (the vacated region redistributes only among existing generators, and every other
  dangerous attacker's growth comes from *inside* the already-counted region), so the counted set weakly
  **shrinks**, *and* attacking PC weakly falls. Both push the difference negative.

**Specification:** the plant drops at least one attacker present in `dangerous` with non-zero
`per_receiver` threat, and asserts the observed difference clears `-TOL_INVARIANT` **by a stated
margin**, not merely that it is negative.

**Why this matters beyond pedantry:** an underspecified plant silently degrades into a no-op as
fixtures evolve — **the same failure class this whole spec was written to fix, recurring one level
down.** A plant that stops planting fails exactly like a guard that cannot fail.

**Fixture adequacy:** if no fixture action offers a qualifying dangerous attacker, that is a **fixture
inadequacy to fix**, not a plant to weaken or skip.

### 3.3 The second clamp — with an access mechanism (MAJOR 3)

rev 2 said "assert the per-blocker `new_recv - old_recv` before `np.maximum`" without saying how.
Those are locals in `_compute_cover_shadow_dict` (`:1099-1109`) and nothing returns them. Recomputing
them in the test would assert **the test's own arithmetic** — a vacuous fixture in this repo's own
vocabulary.

**Route, no API widening:** `_lane_received_batched` is private and already returns the raw pieces —
verified signature `(p_blocked_full, p_received_full, p_received_loo)` at `:463-467`. The test calls
`_lane_int_probs` + `_lane_received_batched` directly on fixture-derived inputs and asserts on the real
returned values.

⚠ **Assert on the SUM over the three lanes, not per lane.** Verified at `:1101-1109`: `old_recv` and
`new_recv` accumulate across center/left/right and the **total** is clamped. A per-lane assertion tests
a stronger property than the code relies on and could fail for reasons that are not defects.

**3.4** Repair `test_zero_blocked_implies_low_score` to assert what its name claims. If the property
does not hold, that is a finding.

**3.5** Record the measurement: *"measured no-op on N actions"* is a different artifact from *"assumed
hygiene"*.

### 3.6 `fernandez_bornn` — RUN it, do not prove it

rev 2 framed this as extending an analytical reduction. For a Gaussian-influence-field model with a
logistic normalisation that is a **research task, not a PR task** — and the asymmetry is the point: an
empirical check **falsifies cheaply**.

Run the existing counterfactual on a fixture with `method="fernandez_bornn"` and inspect the sign of
the raw difference.

- Holds → record *"verified empirically on N actions for fernandez_bornn; argued structurally for
  spearman/voronoi"* — an honest **mixed-provenance** statement, which is more informative than a
  uniform claim and cheaper than a proof.
- Fails → a genuine finding, and **D1 changes**: the clamp is masking real negatives on a supported
  configuration, not performing hygiene.

Either way the outcome is a **documentation** statement (D6). The repaired invariant test only
exercises whichever method the fixture uses, so this needs its own explicit per-method run.

### 3.7 Runtime — measured; stays in CI

| Quantity | Value |
|---|---|
| Fixture | 12,696 frame rows → **10 actions** |
| `add_cover_shadows` pass | ~0.45 s warm (2.35 s cold) |
| Whole test file today | **4.51 s** |

The aggregator already calls `compute_blocking_score` per action, so the repaired assertion costs
~**one extra pass (~0.5 s)**. The fixture is **function-scoped**, re-running for all five tests;
module/session-scoping saves ~1.8 s — more than the new assertion costs. ⇒ stays in CI, and the file
gets *faster*.

⚠ **Scope-widening introduces cross-test coupling** (MINOR 5) — a shared, non-copied DataFrame is a
classic order-dependence flake: any test that mutates it (`.loc` assignment, `inplace=True`, a new
column) silently contaminates the others. This risk is **introduced by this fix**, so it is a gate in
§6, not an assumption: confirm no current test mutates the fixture, and either return a per-test copy
or assert immutability.

---

## 4. Glossary correction

**FIVE** TF-30 entries, not four: `blocking_score` (`:1027`), `n_blocked_receivers` (`:1034`),
`max_single_defender_blocking_score` (`:1041`), `blocked_threat_fraction` (`:1048`),
**`n_potential_receivers` (`:1055`)** — the fifth falls outside rev 1's cited range, which is how it
was miscounted.

The non-negativity caveat applies to the **scoring** columns only (`blocking_score`,
`max_single_defender_blocking_score`, and `blocked_threat_fraction` insofar as it inherits). On the two
**counts** non-negativity is trivial and the caveat would be noise.

Add to the scoring entries: (1) non-negativity is **by construction**, with the §3.6 provenance split
stated; (2) the **divergence from the paper** — ours cannot say "this defender made things worse".

`higher_is_better` stays `None` on all five (direction flips by perspective) — recorded as decided.

---

## 5. The per-defender identity column

**New column:** `max_single_defender_player_id`.

### 5.1 The agreement measurement — sample must resolve the rule (BLOCKER)

rev 2 pre-registered a **0.9** cut but measured it on the **10-action fixture** — fewer once §5.2
makes zero-max rows NA. At n ≈ 5–10 the 95% interval on 9/10 runs roughly **[0.55, 1.00]**: the rule
cannot distinguish "ships" from "does not ship", and whichever side it lands on is an artifact of which
fixture actions happen to have ≥ 2 lane blockers. Pre-registering a cut **without** pre-registering a
minimum sample is ceremony, not discipline — the same shape as the TF-19 §6.4 finding.

**This is a ONE-OFF measurement, not a CI gate.** It therefore does not belong on the committed
fixtures at all. §3.7 established the CI budget is for the *invariant*.

**Protocol:**

- Run **owner-gated on provider / `@e2e` data** (pining GS WC2022), target **n in the hundreds**.
- **Minimum n = 100** qualifying actions (those with ≥ 2 lane blockers and `max_def > TOL_ATTRIB`).
  Below that the rule does not fire and the column does **not** ship on the cheap path.
- Report agreement **with an interval**, not a point.
- **Also record the value gap at disagreements.** Agreement rate alone is the wrong decision input:
  a disagreement when the top two defenders are near-tied is harmless; the same rate with large gaps is
  serious. The gap distribution tells you whether an 0.85 is benign.
- **Decision rule, pre-registered:** agreement ≥ **0.9** at n ≥ 100 → ship as specified with the number
  recorded. Below → do **not** ship silently: either gate the column to `detailed=True` (NA on the cheap
  path) or drop it, as an explicit owner decision. A column that confidently names the wrong defender is
  worse than no column.
- 0.9 is a **stated engineering threshold, not a derived one** — "a consumer reading `..._player_id`
  will assume it is usually right". Named before the number is seen.

**Sequencing — the measurement cannot quietly not happen.** It is a **prerequisite deliverable**: the
column does not merge until the number exists and is recorded in the docstring and glossary. If the
owner-run cannot be scheduled, the honest outcome is §5.1's fallback (gate to `detailed=True`, record
that the cheap-path number was not measurable), **not** shipping on the caveat alone — which is exactly
the weaker standard D4 was written to reject.

### 5.2 NA discipline (D5)

`score_per_blocker = np.zeros(n_lb)` (`:1078`) accumulates only clamped non-negative deltas. When no
lane is meaningfully affected **every entry stays 0.0**, `max_def = 0.0` (`:1112`), and `argmax()`
returns **index 0** — naming `lane_blocker_ids[0]`, a defender who did nothing. Verified:
`np.zeros(4).argmax() == 0`.

The exact path has the same hole: `max_def = 0.0` at `:1041`, and `max(max_def, ...)` at `:1052` never
fires when all scores are 0.0.

**Rule: identity is NA whenever `max_def <= TOL_ATTRIB`** — subsuming `n_lb == 0` and the degenerate
early returns at `:980`/`:1026`. The exact path initialises its argmax sentinel to **`None`**, never
index 0, and assigns only on strict improvement.

### 5.3 Two tolerances, named separately (MINOR 4)

They are conceptually different quantities and must not silently share a constant:

| Name | Meaning | Value |
|---|---|---|
| `TOL_INVARIANT` | "how negative may integration error make the raw difference" | `1e-9`, matching the existing invariant tolerance |
| `TOL_ATTRIB` | "how small is *not an attribution*" | a small positive floor, distinctly **larger** than float noise |

A strict `max_def <= 0` would still fabricate an attribution when `max_def` is numerical noise (1e-14),
which is why `TOL_ATTRIB` exists and is not simply zero.

### 5.4 Mechanics

- **Aggregator-only** (D3): `_CS_COL_NAMES` splits into the numeric set the factory consumes and the
  full set the aggregator emits.
- **Dtype:** source-dtype passthrough via `id_compat.restore_id_dtype` (ADR-019).
- **Tie-breaking:** first maximum (earliest in `lane_blocker_ids`, frame row order). Deterministic per
  frame; documented, not relied on as meaningful.
- **Provenance:** docstring states exactness under `detailed=True`, the cheap path's approximation
  under the default, and the §5.1 measured number.

---

## 6. Testing

| Gate | Asserts |
|---|---|
| Repaired invariant (§3.1) | unclamped `threat_unblocked - threat_original >= -TOL_INVARIANT` |
| **Plant** (§3.2) | RED when `frame_reduced` drops a **dangerous** attacker with non-zero `per_receiver` threat; difference clears `-TOL_INVARIANT` **by a stated margin** |
| Plant adequacy (§3.2) | a qualifying action exists; if not → **fix the fixture**, do not weaken the plant |
| Second clamp (§3.3) | summed-over-three-lanes `new_recv - old_recv >= -TOL_INVARIANT`, via `_lane_int_probs` + `_lane_received_batched` (**not** recomputed in the test) |
| Repaired low-score test (§3.4) | what its name claims |
| `fernandez_bornn` (§3.6) | run per-method; record the sign result and its provenance |
| **NA discipline** (§5.2) | identity NA **wherever `max_def <= TOL_ATTRIB`** — including `n_lb > 0` rows whose max is 0. *(rev 1 said NA "**exactly** where `n_lb == 0`", which would have asserted NA is absent elsewhere and thereby enforced the fabrication.)* |
| Identity ↔ value | identity is the argmax of the same array that produced the value |
| Dtype invariance | numeric-vs-string `player_id` frames (ADR-019) |
| **xfns numeric purity** | `cover_shadow_xfns` emits NO identity column (mirrors `test_das.py:1933`) |
| **Fixture immutability** (§3.7) | no test mutates the now-shared fixture; per-test copy or asserted immutability |
| Glossary / NOTICE | new column documented; coverage gate green; linkage holds |

**Non-vacuity (MINOR 6):** the identity test must name a *different* player than `lane_blocker_ids[0]`
on at least one action. At 10 actions — fewer once zero-max rows are NA — there may be none. **If no
fixture action produces a non-zero argmax, that is a fixture inadequacy to fix, not a test to skip.**

---

## 7. Out of scope

- **RQ1 real-data validation and σ/λ recalibration** — TODO row TF-30 (b).
- **RQ3.** Stays scoped out per `2026-05-10-...:30-32`. This PR **appends evidence strengthening** it:
  the RQ3 headline ("789 of 822") is circular — defenders are optimised *into* the Cone Corridor, then
  success is scored as "is a defender in the cone", using their **worst** model (7.3% recall); the real
  result is threat reduced in 75% of 63,037 snapshots, i.e. worse a quarter of the time, sign only, no
  magnitude/CI/placebo. **No new ADR** — an ADR records a change of decision.
- **Any API change** (D6) and **any shipped column value change** (D2).

---

## 8. Open questions

- **O1 — second clamp scope.** ✅ **RESOLVED: (a) test-only**, now implementable — §3.3 names the
  access mechanism. Widening a return type no consumer has asked for remains the wrong trade.
- **O2 — CI vs `@e2e` for the invariant.** ✅ **RESOLVED** by §3.7: stays in CI.
- **O3 — part-deux.** ✅ **RESOLVED, needs a decision.** Present at
  `../karstenskyt__silly-kicks_part-deux` (rev 1 checked the wrong directory name). Verified
  **byte-identical** on all four touched files. This PR **will** diverge the repos unless propagated.
  ⚠ part-deux is at **4.65.0** locally (`2670f2a`, not on PyPI) — coordinate before either side ships.
- **O4 — NEW: scheduling the §5.1 owner-run.** The identity column is blocked on a measurement that
  needs provider data and owner time. If that cannot be scheduled within this cycle, the column should
  be **split to its own follow-up** and the invariant/glossary work shipped alone — both halves stand
  independently, and shipping the column on an unmeasured caveat is the outcome D4 exists to prevent.

---

## 9. Attribution

No new citation. `NOTICE:511-523` already carries Cascioli et al. (2025) and Spearman et al. (2017).
Author spelling is **Cascioli** (the Hudl URL misspells it "Casciolio"); NOTICE is correct. The Substack
review by Marin Felices is **not** citable — it mis-attributes the paper's 11–15% balanced-accuracy
figure to false-positive reduction.
