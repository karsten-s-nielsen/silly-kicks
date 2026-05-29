# Cover Shadows — leave-one-out perf refactor (hoist redundant man-marking + vectorized masked re-scan)

- **Date:** 2026-05-28
- **PR:** PR-S65
- **Target release:** 3.25.1 patch (recommended) — **pure performance refactor; bit-identical within `rtol≈1e-10`; no API or value change** (minor 3.26.0 acceptable; see §9)
- **Status:** Approved design, pre-implementation
- **Area:** `silly_kicks/tracking/_cover_shadows.py` (TF-30)
- **Origin:** TODO.md "TF-7 cover_shadows nested-loop optimization (deferred, 3.25.0 review)"

## 1. Context

`max_single_defender_blocking_score` (the "lightweight" / `detailed=False` cover-shadow
feature) answers: *which single lane-blocking defender removes the most threat, measured
by the increase in receiver reception probability when that defender is removed?* It is
computed in `_compute_cover_shadow_dict` (`_cover_shadows.py:778-940`) and surfaced through
`add_cover_shadows` and `cover_shadow_xfns`. Atomic SPADL inherits it by pure delegation
(`atomic/tracking/features.py:750` re-calls the standard implementation; no duplicate code).

The lightweight value is an approximation of the full pitch-control counterfactual produced by
`detailed=True`; its contract is **Spearman ρ ≥ 0.7** against `detailed=True`
(`tests/tracking/test_cover_shadows.py::TestDetailedVsLightweightCorrelation`). **This PR does not
change the lightweight value** — it makes its computation dramatically cheaper while producing
bit-identical results (within float reduction-order tolerance). So ρ, and every consumed value, is
unchanged.

## 2. Problem

The lightweight branch (`_cover_shadows.py:895-932`) is `O(blockers × receivers)` full
`lane_control` calls:

```python
for d_pid in lane_blocker_ids:                 # O(blockers)  — lane-blockers only
    frame_without_d = frame_data[frame_data["player_id"] != d_pid]
    for recv_x, recv_y, recv_xt, old_recv in receiver_records:   # O(receivers)
        lc_new = lane_control(frame_without_d, passer_xy, (recv_x, recv_y), ...)  # FULL recompute
        ...
```

Each inner `lane_control` call (`_cover_shadows.py:438-559`) does two expensive things on the
reduced frame:

1. **Re-runs `_classify_man_markers`** (`:298-348`) — a global greedy nearest-first 1:1
   defender→attacker assignment — on `frame_without_d`.
2. **Re-runs `_compute_lane_probabilities`** (`:356-435`) for all 3 lanes — the per-player
   `player_tti` race plus a **pure-Python triple loop** (sample points × defenders × attackers)
   for the sequential survival integration.

### 2.1 The key insight — the per-`d` re-classification is redundant (no-ripple property)

**The exact algorithm (pinned, so the proof below is auditable against the code).**
`_classify_man_markers` (`_cover_shadows.py:298-348`) does precisely this:

1. For each attacker, a *behind-point* `= attacker_xy + man_mark_behind_offset · û_own_goal`.
2. Build candidate edges `(defender_idx, attacker_idx, dist)` for **every** (defender, behind-point)
   pair with `dist < man_mark_radius` (`dist` = Euclidean defender↔behind-point). A defender outside
   every behind-point's radius produces **no** edges.
3. `candidates.sort(key=lambda c: c[2])` — ascending by distance, Python's **stable** sort
   (insertion order is attacker-outer, defender-inner).
4. Greedy 1:1: iterate sorted edges; for `(di, ai, dist)`, `if di in assigned_defenders or ai in
   assigned_attackers: continue`, else assign `di→ai` and add `di` to `man_markers`.
5. Return `man_markers` (the set of defenders that won an edge). A **lane-blocker** is exactly a
   defender **not** in `man_markers`.

There is **no** capacity rule beyond 1:1, **no** global normalization, and **no**
defender-set-cardinality term — each edge's existence and weight (its distance) depend only on that
one (defender, attacker) pair, not on the roster size.

**Proof that removing a lane-blocker is a no-op for the matching.** Let `d` be a lane-blocker
(`d ∉ man_markers`, i.e. `d` won no edge). Because `d` is never assigned, `d` is *free* at every
moment, so for each of `d`'s edges `(d, A)`: had `A` been free when `(d, A)` was processed, the
greedy would have assigned `d→A`, making `d` a winner — contradiction. Hence **every** `d`-edge had
its attacker `A` already assigned when processed: every `d`-edge was a no-op. Now remove `d`:
- It deletes only `d`'s edges (per step 2, no other defender's edges exist *through* `d`).
- The remaining edges keep their distances (step 2 is pairwise) and therefore their sorted order
  (step 3; stable sort preserves relative order of equal-distance non-`d` edges).
- Each remaining edge meets the identical `assigned_*` state at its turn, because the deleted
  `d`-edges were no-ops that never mutated any `assigned_*` set.

So every other defender's assignment is byte-for-byte identical, and `d ∉ man_markers` anyway.
Therefore, for every lane-blocker `d`:

> **`man_markers(frame_without_d) == man_markers(full)`** — proven above against the pinned
> algorithm, and guarded by the §6.4 property-based test (random rosters) plus adversarial
> fixtures (two defenders contesting one behind-point, contested chains, a four-defender pile-up),
> lifted from the investigation probe.

Consequences:
- `lane_control(frame_without_d)` already uses racer set `= lane_blockers_full − {d}` (since the
  man-marker set is invariant and `d ∉ man_markers`). The current per-`d` re-classification
  recomputes the **identical** man-marker set every iteration — pure wasted work.
- So computing man-markers **once on the full frame** and reusing it is **bit-identical** to the
  current code. This is a perf refactor, not a semantics change.

This is also why a generic **corridor prune** was correctly rejected: a prune may skip *arbitrary*
defenders by geometry, possibly including man-markers, and removing a *man-marker* **can** ripple
(it frees its attacker, promoting another defender). The chosen leave-one-out is immune precisely
because it only ever removes lane-blockers.

### 2.2 Structural facts the refactor exploits

- `player_tti` (`:173-259`) and `ball_drag_time` (`:135-165`) are **per-player / per-lane-geometry**.
  A player's interception probability along a lane does not depend on which *other* players are
  present; the only cross-player coupling is the sequential survival sum.
- Lane sample points (center/left/right) and `t_ball` depend only on passer→receiver geometry,
  invariant across defender removals for a fixed receiver.
- The **main** `compute_blocking_score` (`:647-762`) already classifies man-markers **once** and
  removes all lane-blockers together; only the lightweight `max_single` branch re-classifies
  per-removal — redundantly, per §2.1.

## 3. Decision

Adopt **hoist the redundant man-marking classification + precompute `p_int` once per receiver +
vectorized masked re-scan** (chosen over a semantics-preserving micro-opt and over a corridor prune).

Classify man-markers **once on the full frame** (bit-identical per §2.1) and hold that racer set
fixed for the leave-one-out. With a fixed racer set, removing defender `d` affects receiver `r`
only through `d`'s own interception-probability contribution. Precompute the **clamp-independent**
quantities (lane geometry, `t_ball`, `p_ctrl`, per-player `p_int` matrices) once per receiver,
then compute the leave-one-out for *every* lane-blocker in a single vectorized pass that **re-runs
the clamped survival recurrence with that blocker's `p_int` row masked out**. **No corridor prune**
(rejected per §2.1; vectorization over `n_blockers ≤ ~10` subsumes any pruning benefit). **No numba**
(residual work is numpy-backed and cheap).

> **INV-1 (no subtraction under the clamp).** The survival recurrence applies a path-dependent
> nonlinearity, `min(p_anyone_prior + total, 1.0)`. Subtracting a defender's precomputed
> contribution from the full-set result is **NOT** equal to re-running the recurrence without that
> defender once the running sum saturates — subtraction and `min(·, 1.0)` do not commute. The
> leave-one-out MUST re-run the clamped recurrence with the defender's `p_int` row masked.
> "Precompute" refers only to clamp-independent quantities (`p_int`, `t_ball`, `p_ctrl`, lane
> geometry), never to the clamped accumulation. An implementer must not "optimize" the masked
> re-scan into a subtraction. The §6.1 exactness test is the guard.

No API change; no value change beyond `≤ rtol 1e-10` float reduction-order drift from vectorization.

## 4. Why this is bit-identical (no semantic change)

The refactor changes *how* the lightweight value is computed, not *what* it computes:

1. **Man-marking hoist is exact (§2.1).** Removing a lane-blocker provably never changes the
   man-marker set, so classifying once == re-classifying per-`d`. The §6.4 regression test pins
   this property as a permanent guard, since the entire optimization rests on it.
2. **Precompute is exact.** `player_tti`/`t_ball`/lane geometry are per-player/per-geometry; the
   `p_int` row for a retained defender is identical whether computed once on the full racer set or
   per-removal. Masking `d`'s row yields the same survival inputs as `lane_control(frame_without_d)`.
3. **Vectorization is exact up to float reduction order.** The only numeric difference is summing
   the small player/blocker contributions via numpy vs sequential `+=`; for ~10-element sums this
   stays well under `rtol 1e-10`. The `n_points` scan stays a loop (preserving the clamp, INV-1).

**Monotonicity (not an artifact removal — true in both old and new code).** Under the fixed cast,
removing a blocker can only *raise* a receiver's reception probability (`new_recv ≥ old_recv`),
because fewer blockers contribute to the survival product. So `max(new_recv − old_recv, 0)` is a
float-noise-only clamp. This already holds in the current code (no man-marking ripple to produce
`new_recv < old_recv`), so the refactor neither introduces nor removes negative-delta artifacts.

Man-marking defenders are **intentionally never `max_single` candidates** — the loop iterates
`lane_blocker_ids`, which excludes man-markers (a man-marker is modelled as marking an attacker,
not contesting the pass lane). Unchanged by this PR.

**Effect on outputs:** none beyond `≤ 1e-10` float drift. All five `_CS_COL_NAMES` columns and the
`detailed=True` path are unchanged. No consumer impact; no golden/model regeneration required.

## 5. Algorithm

Scope of edit: the `detailed=False` branch of `_compute_cover_shadow_dict` (`:895-932`) and a small
refactor of `_compute_lane_probabilities` to expose its precomputed `p_int` matrices.

1. **Classify once.** Reuse the already-computed `lane_blocker_ids` (`:847-859`) — non-man-marking,
   non-GK outfield defenders on the full frame. This is the fixed racer set.
2. **Per dangerous receiver `r`, precompute once:** the 3 lane geometries + `t_ball` + `p_ctrl`;
   `p_int_def` `(n_lane_blockers, n_points)` and `p_int_att` `(n_attackers, n_points)` per lane via
   `player_tti` + sigmoid; and the **baseline** `p_received_r` (= `old_recv`) from the survival scan
   over the full racer set. **Implementation choice (revised):** the existing first-loop `lc_orig`
   pass (`:833-843`) is **kept unchanged** to compute `n_blocked` via `lane_control`, so
   `n_blocked_receivers` stays provably bit-identical (it flows through untouched code). The new
   branch recomputes `old_recv` (variant 0 of the batched re-scan), accepting a cheap `O(receivers)`
   double-compute of the baseline rather than merging the pass — the dominant
   `O(blockers × receivers)` cost is still eliminated. Merging the baseline into the precompute is a
   deferred future cleanup; bit-stability of `n_blocked` is the higher priority here.
3. **Vectorized leave-one-out (single pass over the defender axis):** compute `new_recv` for *every*
   lane-blocker `d` by re-running the clamped survival recurrence with `d`'s `p_int_def` row masked,
   batched across blockers. Retains the `min(prior + total, 1.0)` clamp and the `dt_k <= 0` skip;
   only the player-axis and blocker (leave-one-out) axis vectorize. Python loop reduces to the
   `n_points (≈30)` scan steps per lane (not per defender).
4. **Score:** `score_d = Σ_r xT(r) · max(new_recv[d, r] − old_recv[r], 0.0)`;
   `max_single = max_d score_d`. `xT(r)` / `old_recv[r]` are receiver loop-invariants.

Degenerate-input guards (missing `vx`/`vy`, no ball, NaN coords, zero potential receivers, no
lane-blockers) and all `_CS_COL_NAMES` keys are preserved exactly.

### 5.1 Internal helper shape

- `_lane_int_probs(lane_targets, def_pos, def_vel, att_pos, att_vel, *, params) -> (p_int_def, p_int_att, t_ball, p_ctrl)`
  — per-lane precompute (extracted from the front of `_compute_lane_probabilities`).
- `_lane_received_survival(p_int_def, p_int_att, p_ctrl, *, exclude_def_mask=None) -> (p_blocked, p_received)`
  — the clamped survival scan; `exclude_def_mask` selects participating defender rows, enabling a
  batched `(n_blockers,)` leave-one-out.
- `_compute_lane_probabilities` is re-expressed as their composition, preserving its current
  internal return for existing callers.

These are private (`_`-prefixed); no public surface changes. **`lane_control` remains the public
per-(passer, receiver) primitive** and keeps its own tests; `_compute_cover_shadow_dict` may use the
helpers directly for both baseline and leave-one-out, so confirm `lane_control` retains a live
caller/coverage (it does — `TestLaneControl` and the baseline `lc_orig` path).

## 6. Exactness and validation

The anchor is **bit-identicality to current behavior** (`rtol 1e-10`), which equals the decoupled
definition because the man-marking hoist is exact (§2.1).

### 6.1 Independent exactness test (golden master)

Build an oracle that **shares none of the new helpers**: a **test-vendored frozen copy of the
pre-refactor `_compute_lane_probabilities`** (`tests/_vendored/` or a module-local
`_reference_lane_probabilities`), invoked per `(d, r)` with `defender_pos`/`defender_vel` = the
full-frame lane-blocker set minus `d` (attackers unchanged), then `Σ xT(r)·max(new−old,0)`, max
over `d`. Assert production `== reference` via `np.testing.assert_allclose(rtol=1e-10)` on a
multi-blocker, multi-receiver fixture. Because the man-marking hoist is exact, this reference also
equals current behavior; the test therefore validates the helper extraction **and** the
vectorization against independent code, and directly guards INV-1 (a subtraction-under-clamp impl
fails it). `rtol=1e-10` is achievable: only ~10-element player/blocker sums change reduction order;
the `n_points` scan is preserved as a loop in both paths.

### 6.2 Correlation contract — unchanged

`TestDetailedVsLightweightCorrelation` (lightweight vs `detailed=True`) passes with the **same ρ**
as today, since values are bit-identical within tolerance. No re-measurement risk and no threshold
change; the test serves as a guard that the refactor didn't perturb the value.

### 6.3 Real-match bit-identicality confirmation

On a **real match** (not only the synthetic fixture), confirm `max |Δ| < 1e-10` between pre- and
post-refactor `max_single_defender_blocking_score` over all actions — empirical evidence (beyond
the synthetic golden) that no consumer-visible drift occurs. Report it in the PR description.

### 6.4 Man-marker invariance — property-based guard (the load-bearing test)

The entire PR's correctness reduces to §2.1, so this guard must cover the invariant *broadly*, not
on a few hand-picked geometries — a fixture test can pass on its chosen configs while a future
`_classify_man_markers` change silently breaks no-ripple elsewhere, making the hoist non-identical.

Primary guard — a **Hypothesis property test**: generate random rosters (e.g. 2–11 defenders and
2–11 attackers at random pitch positions, random `man_mark_radius`/offset within sane bounds) and
assert, for **every** lane-blocker `d`,
`_classify_man_markers(defenders − {d}) == _classify_man_markers(defenders) − {d}`. This is the
exact invariant the optimization depends on, under broad random coverage — cheap and far stronger
than examples.

Retain `TestManMarkerInvariantUnderLaneBlockerRemoval` (the investigation probe) as **named
adversarial examples** alongside the property test (two defenders contesting one behind-point;
contested chains; a four-defender pile-up), plus the "guard-the-guard" assertion that at least one
fixture contains a *within-radius losing* lane-blocker so the check is non-vacuous. Together: the
proof (§2.1) is the guarantee, the property test is broad coverage, the fixtures are readable
witnesses.

### 6.5 Regression sweep + atomic parity

All `_cover_shadows` tests pass unchanged; assert all five `_CS_COL_NAMES` columns are bit-stable
(within `rtol 1e-10`) on existing fixtures. The `n_blocked_receivers` stability is **load-bearing**:
it flows through `lane_control → _compute_lane_probabilities`, so it proves the helper extraction is
bit-identical for the function's *other* consumers, not just the leave-one-out. Add a test asserting
`atomic.add_cover_shadows(...).max_single == standard.add_cover_shadows(...).max_single` on a shared
fixture, enforcing the pure-delegation claim against future atomic drift.

### 6.6 TDD sequencing — three independently-green steps

Each step proven `== reference` (§6.1) to isolate any break:

1. **Reference + naive hoisted path.** Land the vendored frozen oracle and a straightforward
   hoisted leave-one-out (per-`(d, r)` `_compute_lane_probabilities` on the fixed lane-blocker set
   minus `d`, no helper extraction, no vectorization). Prove `== reference`. Isolates the man-marking
   hoist.
2. **Extract helpers.** Factor `_lane_int_probs` / `_lane_received_survival`; prove still `== reference`.
3. **Vectorize** the leave-one-out over the blocker axis; prove still `== reference`.

## 7. Performance

Per the measure-before-optimize discipline:

1. Capture a **baseline** wall-clock on a representative multi-blocker frame (e.g. the 10v10 fixture
   in `test_rank_correlation_ge_07`) for `add_cover_shadows(detailed=False)` / raw
   `_compute_cover_shadow_dict`, **before** any change.
2. Report the post-change speedup in the PR description.
3. Add a flat CI perf budget (`_BUDGET = <worst observed CI timing> × 1.5`) in a `pytest` timing
   guard following the repo's existing pattern. The guard **lands in this same PR** (not deferred),
   budget derived from the step-1 baseline on un-changed code.

Expected win: eliminates `O(blockers × receivers)` full `lane_control` calls (`player_tti` +
man-marker classification per call) in favor of `O(receivers)` precomputes + one vectorized
leave-one-out pass.

## 8. Non-goals

- No public API / signature changes (`lane_control`, `compute_blocking_score`, `add_cover_shadows`,
  `cover_shadow_xfns`, `_CS_COL_NAMES`).
- **No semantic / value change** — bit-identical within `rtol 1e-10`.
- No change to the main `blocking_score`, `blocked_threat_fraction`, or the `detailed=True` path.
- No pitch-control / `PitchControlCache` interaction (this branch never touched it).
- No corridor prune (rejected per §2.1); no numba; no atomic-side edits (delegation inherits the win).

## 9. Version and documentation

- **3.25.1 patch** (recommended). Since the change is provably bit-identical (no value or API
  change), a patch is the SemVer-honest signal — a minor traditionally implies new
  behavior/features, which this adds none of. A minor (3.26.0) is acceptable if release-cadence
  consistency is preferred; confirm at ship. Either way the version-bump hard gate applies:
  `pyproject.toml`, `__init__.py`, `TODO.md`, `CHANGELOG.md` must all match, and the CHANGELOG's
  "no value or API change; no regeneration required" line is what actually prevents consumer
  confusion.
- **CHANGELOG.md:** "perf: cover_shadows `max_single_defender_blocking_score` computed via a single
  vectorized leave-one-out (hoisted redundant man-marking re-classification) — ~Nx faster,
  **bit-identical within `rtol 1e-10`**; no value or API change." Explicitly state no
  golden/model regeneration is required (it is *not* a behavioral change).
- **Docstrings:** update `_compute_cover_shadow_dict` and the new helpers to state the no-ripple
  property (§2.1), the precompute + masked re-scan approach, and INV-1.
- **TF-30 spec** (`docs/superpowers/specs/2026-05-10-tf30-cover-shadows-design.md`): short amendment
  note pointing here for the lightweight-path perf refactor.
- **TODO.md:** delete the "TF-7 cover_shadows nested-loop optimization (deferred)" row on ship
  (CHANGELOG is the record; do not strikethrough).

## 10. Relationship to the parked xfn-context (4.0.0) work

This change attacks **counterfactual** (player-removed) reception probabilities — the part the
shared `PitchControlCache` deliberately cannot serve (counterfactual safety) — so it is
non-overlapping with the parked xfn-context shared-cache refactor and order-independent. Ships first
as 3.26.0; xfn-context follows as the 4.0.0 contract change.

**Note — the two `max_single` definitions are a deliberate fast/exact pair, not cleanup-debt.**
`detailed=True` is the exact per-defender pitch-control counterfactual (remove defender, recompute
the PC surface, measure xT-weighted threat); `detailed=False` is the lightweight `lane_control`
survival leave-one-out this PR optimizes, bridged by ρ ≥ 0.7. They are **not** cheaply convergeable
via `PitchControlCache`: the cache serves only **canonical** surfaces, and `detailed=True`'s cost is
entirely the **O(blockers) per-defender counterfactual** surfaces, which counterfactual-safety
forbids caching (`_cover_shadows.py:697-699, 747-752`). So the lightweight path remains permanently
justified as the cheap default; neither mode is droppable. No follow-up required.

## 11. References

- Cascioli, Wang, Stradiotti, Van Roy, Robberechts, Wouters, Jaspers & Davis (2025). "Quantifying
  Off-Ball Defensive Impact through Cover Shadows" (Hudl / DTAI, KU Leuven).
- Spearman (2017) — ball drag / time-to-intercept model (see NOTICE).
- See `NOTICE` for full bibliographic citations.
