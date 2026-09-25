# Tracking performance — vectorized spearman kernel, scorer batching & bounded-memory streaming — Design Spec

| | |
|---|---|
| **Status** | Draft (awaiting other-session review r1/r2/r3) |
| **Date** | 2026-09-24 |
| **Decision** | ADR-105 (provisional — re-derive at commit-prep) |
| **Version** | next-free minor (assigned at commit-prep; do NOT hardcode) |
| **Extends** | ADR-008 (pitch-control cache), ADR-076 (numba bit-identical parity-gate pattern), ADR-103 (frame-memory category dtypes + batched pitch-control seam), ADR-068/073 (`group_rows` + sub-quadratic scale guard), ADR-063 (velocity tiering), ADR-019/057 (id_compat + pandas-major span) |
| **Retrain / re-materialize** | **NONE.** Every change is value-neutral or parity-gated byte-identical. Any observed value change is a bug to fix, not accept. |

---

## 1. Motivation

A lakehouse `compute_tracking_marts` drain OOM-killed (exit 137) every worker on a Gradient Sports half
(`D:\Development\_reviews\2026-09-24-sk-tracking-scorer-memory-perf-handoff.md`). Profiling one GS half
(match `10502` p1 — 1.91 M rows / 84 K frames / 16 players, sk 4.123.0) showed the per-unit scoring path
peaks **~3.4 GB** and `rest_defense` runs **361 s**. None is lakehouse-specific — it is inherent to the
silly-kicks tracking frame schema and pitch-control scorers, and every tracking consumer/provider pays it.

**ADR-103 (4.125.0) shipped the first pass** — F1a (static-column `category` dtypes + drop `confidence`,
~2× frame memory), F5 (`PitchControlCache(maxsize=)` bounded LRU), F2 (the *seam* `compute_pitch_control_batch`
+ `PitchControlCache.warm`, currently a per-frame loop), F6 (shared-cache contract). It DROPPED F3
(rest_defense layer-2 warm-routing — a measured memory regression, no CPU win) and DEFERRED F4
(bounded-memory streaming) + the vectorized kernel + F1b to a later cycle.

**This cycle is the owner-set "every optimization that does NOT require retraining" cycle.** It delivers the
CPU win F2's seam was built for (the vectorized kernel — which is what actually closes F3), the F4
bounded-memory path, and no-retrain fixes to ADR-103's own shipped code. F1b (float32 coordinates + id→category
+ off-frame provenance) is value-changing / retrain-carrying and is **deferred to the retrain cycle**.

## 2. Scope

**In scope (all no-retrain):**
1. A **vectorized cross-frame spearman kernel** — the payload behind the ADR-103 `compute_pitch_control_batch` seam.
2. **Routing the whole-unit scorers' pitch control through it** — the CPU win; closes F3 (rest_defense).
3. **F4 — internal auto-batching + bounded surface retention** across the whole-unit scorers.
4. **ADR-103 shipped-code fixes:** the `PitchControlCache.warm` double-group; the `_key` per-call scan.
5. **`group_rows(observed=True)` + a pandas-major guard** — defensive, now that category frame columns exist.
6. **A DAS all-frame cost guardrail** (minor).

**Out of scope — DEFERRED to the retrain cycle (owner-ruled 2026-09-24):**
- **#3 id→category** (`player_id`/`team_id`) — the spike (§9) found it value-neutral but a **perf regression**
  in `id_compat` + a **behavior change** through `group_rows` (phantom empty groups; pandas-2-vs-3 default).
  It travels with the `group_rows(observed=True)` prereq and F1b.
- **F1b float32 coordinates** + off-frame provenance-to-metadata — value-changing / retrain-carrying.

**Non-goals:** no algorithm change to any metric; no new metric; no C4 aggregator; no VAEP feature change.

## 3. The vectorized spearman kernel (Change 1)

### 3.1 Current shape (mapped)
- `tracking/pitch_control/_spearman.py::compute_spearman` (L122-273) reads per-frame `x`,`y`,`vx`,`vy`,
  `team_id`,`is_goalkeeper`,`player_id`,`is_ball`; **filters ball + NaN-position rows per frame** (L149-150),
  so **player count and row order VARY per frame**.
- `compute_tti` (L31-84) and `_compute_influence` (L87-119) are **already numpy-vectorized within one frame**
  (`disp (n_players, n_targets, 2)`; logistic sigmoid `1/(1+exp(-k·(opp_min−team_tti)))`); numba path via
  `_HAS_NUMBA`.
- The grid is **FIXED** — `_grids.py::pitch_grid` is `@lru_cache`, default `50×32 = 1600` targets, read-only.
- Combine: per-team min-TTI → logistic influence per side → optional ball-travel-time zeroing → GK
  `×lambda_gk` → ratio `att_sum/(att_sum+def_sum)`, default `0.5` where both zero.

### 3.2 Design
Add `compute_spearman_batch(frame_slices, requests, params) -> list[PitchControlSurface]` (internal), wired
under the existing public `compute_pitch_control_batch` (`_dispatch.py`). It:
- Stacks the N request frames into padded 3D arrays `pos (N, max_players, 2)`, `vel (N, max_players, 2)`,
  plus a per-row **validity mask** `(N, max_players)` and per-row `team`/`is_gk` arrays. Padding rows are
  masked out so they contribute exactly 0 to `att_sum`/`def_sum`.
- Runs TTI + influence **broadcast over the frame axis** (`(N, max_players, n_targets)`), reusing the SAME
  `compute_tti`/`_compute_influence` math (single-sourced cores; the per-frame path becomes a thin N=1 caller,
  so it stays byte-identical — the ADR-102 single-source idiom).
- Reduces per frame to `(N, ny, nx)` surfaces, applying the identical per-team min / logistic / GK /
  ball-zeroing / ratio steps.

**The per-frame filter is reproduced exactly** — a padded/masked player must contribute identically to being
absent, including the "both zero → 0.5" default and the ball-travel-time zeroing. `attacking_team_id`,
`ball_position` and `decompose` are per-request.

### 3.3 Parity — BYTE-IDENTICAL (the hard gate)
`compute_pitch_control_batch(frames, requests)` must be `np.array_equal` to calling `compute_pitch_control`
per request over the same per-frame slice — the ADR-076 precedent (max |Δ| exactly 0). Gated across both
methods × `decompose` T/F, over **ragged fixtures** (frames with different player counts, a NaN-position
player, a GK-only side, a both-empty target region → the 0.5 default). **The fixture MUST exercise padding
and order (VKS-SPEC-02): at least one frame has FEWER players than the batch's `max_players` (so padding rows
exist and must be masked to 0), and the players are UNSORTED / in a different order across frames** — because
byte-identity depends on the valid-player axis staying aligned to each request's own row order after the
pad/mask stack. The 4.125.0 `test_batch.py` parity test is extended from the loop-baseline to the vectorized
kernel; **two mutations must go red — dropping the mask, AND permuting the stacked valid-player rows.**

**Numba:** if a numba kernel is added it is the ADR-076 shape — a numpy PORT (reference + fallback) + a
`@njit` ADAPTER, `np.array_equal`, lazy import, `SILLY_KICKS_...FORCE_NUMPY` escape. Optional; the numpy
vectorized path is the baseline and already the win over the per-frame Python loop.

**No retrain:** byte-identical surfaces → every golden/chirality/feature-contract test unchanged.

## 4. Route the scorers through the kernel (Change 2 — closes F3)

The whole-unit scorers currently compute one surface at a time. Route their pitch-control through the
vectorized batch:
- **rest_defense layer-2** (`restdefense/_danger.py::layer2_metrics`, the 361 s path): **4 surfaces/sample**
  — 1 `spearman decompose=True @team_id` (`_danger.py:L112`), 1 `tracking/_gk_influence.py::compute_gk_influence`
  (def L232) whose `cache.surface @opponent_id` is at `tracking/_gk_influence.py:376`, 2
  `tracking/_cover_shadows.py::compute_threat_pc` (def L826) `@opponent_id` on `frame` and `frame_no_gk` — which
  **bypass the cache by design** (`tracking/_cover_shadows.py:848-851` docstring + `:899` direct compute). Batch
  these across the ~250 samples of a unit. `compute_threat_pc` gains a batched entry (it wraps
  `compute_pitch_control` internally) so its two legs batch too. **NB — the symbols live in `tracking/`, not
  `restdefense/`; `restdefense/_danger.py` only CALLS them** (VKS-SPEC-01).
- **off_ball_runs** (`_run_values.py`), **defensive_credit** (`defensive_credit/_orchestration.py`),
  **gk_decision reconstruction** (`gk_decision/_reconstruct.py`): route their per-item surface computes
  through the batch where the surfaces are canonical (cache-eligible); `gkdv`/`territorial` reuse the public
  seam unchanged.

**F3 target:** rest_defense on a ~250-sample GS half runs in **≲60 s** (from 361 s), output byte-identical.
Measured by a benchmark (structural op-count guard + a wall benchmark in the standalone `benchmark` job, per
the repo's no-`assert ms<budget` convention).

**Correctness landmine (carry from ADR-043):** counterfactual surfaces (player removed/moved — cover-shadow,
space-creation) must **NEVER** route through the cache (frame-identity key excludes player positions). The
batch kernel is only for **canonical** frames; the counterfactual path stays direct. Guarded.

## 5. F4 — internal auto-batching + bounded surface retention (Change 3)

**Decision: internal auto-batching, NOT a consumer-driven per-batch scorer primitive.** The action→frame link
is **global over the unit** (built once over all frames: `_run_values.py:530`, `_orchestration.py:96`,
restdefense engine-tables over full `rframes`), so **no consumer can score sub-unit** — the frames cannot be
split. A public per-batch scorer primitive therefore **cannot reduce peak below the unit's frame-hold**
(~630 MB post-F1a for a GS half) and would be speculative API debt (`feedback_speculative_api_surface_is_debt`).
The batched primitive a distributed/GPU consumer actually needs is the **public `compute_pitch_control_batch`**
(Change 1) — it drives *that*, not a scorer API.

The map confirms the structural precondition: each scorer does 1-2 **global whole-frame passes up front**
(groupby / `link_actions_to_frames` / engine-table build), then a **per-item-independent loop** — so chunking
the loop is output-preserving.

**Design:** each whole-unit scorer takes an optional keyword-only `batch_size: int | None`. It runs its
per-item loop in chunks of `batch_size`: compute the chunk's surfaces via the vectorized batch, emit the
chunk's rows, **release the chunk's surfaces before the next chunk** (compute-and-discard; the F5 `maxsize`
LRU bounds any retained cache). Peak = frames-hold + O(`batch_size`) surfaces + engine tables.

- **Default:** a bounded `batch_size` (e.g. 256) so **every consumer gets the bounded peak for free**;
  `batch_size=None` = whole-loop (the current behavior, for callers who want it).
- **Output byte-identical for any `batch_size`** — the loop bodies are per-item-independent, so chunk
  boundaries cannot move a value. Parity gate: `batch_size ∈ {None, 1, 7, len}` all byte-identical.
- **The default flip to a bounded `batch_size` changes the execution path for EVERY existing caller**
  (output-identical, but a real path change), so it is a **HARD MERGE PREREQUISITE (VKS-SPEC-03) that the
  `{None, 1, 7, len}` byte-identical invariance gate is GREEN for every routed scorer** — the flip does not
  land until each scorer's invariance test passes red-first-then-green. (A routed scorer whose invariance
  gate is not yet green keeps `batch_size=None` until it is.)

**F4 target:** a consumer scores a GS half's off_ball + defensive_credit + rest_defense under a documented
peak (≤~1 GB) with unchanged output. **Honest bound:** F4 caps the *surface/working-set* accumulation, NOT
the frame-hold — the link needs all the unit's frames. Post-F1a (~630 MB) + O(batch) fits ≤1 GB; the spec
does not claim O(batch)-total.

## 6. ADR-103 shipped-code fixes (Change 4 — "optimize the most recent release")

Both no-retrain, both confirmed by the map:
- **`PitchControlCache.warm` double-groups the frames.** `_cache.py:143` calls `compute_pitch_control_batch`
  (which builds `group_rows(frames, ...)` at `_dispatch.py:208`), then `_cache.py:144` builds
  `group_rows(frames, ...)` **again** with the identical key, purely to re-fetch each `frame` for `_key`. Fix:
  `compute_pitch_control_batch` returns (or `warm` reuses) the grouping / the per-request frame — one
  `group_rows` per `warm` call, not two. Output byte-identical (a structural call-count guard on `group_rows`).
- **`_key` re-scans per `.surface()` call.** `_cache.py:_key` runs `.dropna().unique()` over 3 columns on every
  call (incl. cache hits) to derive the key. For a single-frame slice this is cheap-but-repeated; tighten to a
  cheaper key derivation (the caller already knows `(game_id, period_id, frame_id)` in the batch/`warm` path —
  pass it through instead of re-deriving). Byte-identical key semantics (a parity test on the produced key).

## 7. `group_rows(observed=True)` + pandas-major guard (Change 5 — defensive)

`_frame_index.py::group_rows` calls `df.groupby(list(by), sort=False)` (L38) with **no `observed=`**. On a
categorical key this (a) emits a `FutureWarning`, (b) returns **phantom empty groups** for zero-row categories,
and (c) has a **pandas-2-vs-3 version-dependent default** (`observed` flips to True on pandas 3, ADR-057).
ADR-103 just introduced category frame columns (`source_provider`/`ball_state`/`is_goalkeeper_source`), so a
future `group_rows` on one is a live-adjacent landmine (and it is the prereq that would make deferred-#3
viable). **Fix:** pass `observed=True` explicitly (observed groups only, deterministic across pandas majors).
No current caller keys `group_rows` on a category column, so this is **byte-identical today** — a defensive
pin. A test keys `group_rows` on a wider-than-observed category and asserts no phantom groups + no warning,
across the pandas-major span (ADR-057).

## 8. DAS all-frame cost guardrail (Change 6 — minor)

`tracking/_das.py` all-frame DAS is ~394 h/season, documented-not-gated (ADR-014). Add a **lightweight cost
estimate + a one-time warning** when DAS is invoked over a full unit without frame sampling (opt-out), so a
consumer is told the cost before paying it. No behavior/output change (a warning + a pure cost function); NOT
a hard gate.

**Coupling note (VKS-SPEC-04):** this change is INDEPENDENT of the kernel/streaming work — it shares only the
"non-retrain optimization" theme, not code. It ships in this cycle **per the owner's single-cycle scope ruling
(2026-09-24: "all together in a single cycle that do not require retraining")**, and the cycle's ONE coherent
commit carries it. Scope is the owner's — it is trivially liftable to its own commit/cycle if the owner
prefers to split it out; the spec does not pre-decide that.

## 9. The deferred #3 spike (recorded, not implemented)

Measured (pandas 2.3.3) — id→category is value-neutral but:
1. **Perf regression in `id_compat`:** `infer_dtype(category)` returns `"categorical"`, never `"string"`, so
   `_all_genuine_strings` (`id_compat.py:165-177`) is always False → every category-id compare takes the SLOW
   `canonical_id_series` element-wise path, never the fast raw-`==`.
2. **`group_rows` behavior change:** phantom empty groups + the pandas-major default (§7).

→ **Deferred to the retrain cycle** with F1b float32 (owner-ruled). §7's `observed=True` fix is the prereq
that de-risks it there. Recorded so the retrain cycle inherits the finding rather than re-deriving it.

## 10. Testing (TDD, non-vacuous)

- **Kernel parity (Change 1):** `compute_pitch_control_batch` `np.array_equal` to the per-frame loop across
  both methods × `decompose` T/F, on **ragged** fixtures (varying player counts, NaN-position player, GK-only
  side, both-empty region → 0.5). The fixture MUST include padding (a frame with fewer players than the batch
  `max_players`) and unsorted/differently-ordered players (VKS-SPEC-02). **Two mutations go red: dropping the
  mask, AND permuting the stacked valid-player rows.** (Extends `test_batch.py`.)
- **Scorer routing (Change 2):** each routed scorer's output byte-identical pre/post routing (golden compare);
  a benchmark shows rest_defense ≲60 s and the batch < the per-frame loop (structural op-count guard + wall
  benchmark in the standalone `benchmark` job).
- **F4 batching (Change 3):** `batch_size ∈ {None, 1, 7, len}` all byte-identical per scorer; a documented
  peak-memory assertion (or a structural surface-retention count) shows peak is O(batch), not O(unit).
  **This per-scorer invariance being green is the HARD MERGE PREREQUISITE for flipping the default to a bounded
  `batch_size` (VKS-SPEC-03)** — a scorer without a green invariance gate stays `batch_size=None`.
- **Change 4:** a `group_rows` call-count guard on `warm` (1, not 2); a `_key` parity test (same key, fewer
  scans — structural).
- **Change 5:** category-key `group_rows` → no phantom groups, no `FutureWarning`, across the pandas-major span.
- **Change 6:** the cost warning fires once on a full-unit DAS call and is opt-out; the cost function is pure.
- **No-retrain proof:** every golden/chirality/feature-contract test unchanged; the SB360 audit re-derives
  unchanged (the kernel is byte-identical, batching is output-preserving).
- **ADR-073 scale guards:** any NEW `group_rows` caller registers a growth guard; the vectorized kernel and
  each batched scorer carry a sub-quadratic growth check where a rescan would regress.

## 11. ADR + rollout

- **ADR-105** (provisional): the vectorized spearman kernel + its byte-identical parity contract; the scorer
  routing that closes F3; F4 internal auto-batching (and WHY not a consumer-driven scorer primitive — the
  global-link constraint); the two ADR-103 shipped-code fixes; the `group_rows(observed=True)` pin; the DAS
  guardrail. Records the deferred #3/F1b as the retrain cycle's inheritance.
- **Version:** next-free minor, bumped in `silly_kicks/_version.py` (ADR-079 single source) **at commit-prep**.
- **Branch:** one feature branch `feat/tracking-perf-kernel-streaming`; single coherent commit at the end
  after the explicit owner commit gate (no per-step commits).
- **Consequences:** ~5-10× rest_defense CPU (F3); a bounded per-unit peak (F4); byte-identical output for
  every consumer/provider; **no VAEP/tracking retrain, no re-materialize, C4-free** (no new aggregator/backend/
  model; the batched entry is an internal kernel behind the existing public seam).

## 12. Alternatives considered

| Alternative | Rejected because |
|---|---|
| **F4 (b) public per-batch scorer primitive** / **(c) both** | The action→frame link is global over the unit; no consumer can score sub-unit, so the primitive can't reduce peak below the frame-hold — speculative API debt. The public `compute_pitch_control_batch` IS the consumer-drivable primitive. |
| **Include #3 id→category this cycle** | Spike (§9) = value-neutral but a perf regression + a behavior change; deferred to the retrain cycle (owner-ruled). |
| **Include F1b float32 coordinates** | Value-changing → retrain-carrying; excluded by the cycle's constraint. |
| **Revive F3 as layer-2 warm-routing** | ADR-103 measured it a memory regression with no CPU win; the vectorized kernel is the real F3 fix. |
| **Vectorize by requiring uniform player count** | Real frames are ragged (per-frame ball/NaN filter); padding+masking is mandatory, not optional. |

## 13. Open questions (for the reviewer)

1. `batch_size` default — 256 proposed; is there a measured sweet spot, or leave it tunable with a documented
   default?
2. Should `compute_threat_pc` grow a **public** batched entry (it currently bypasses the cache by design), or a
   private one consumed only by the routed scorers? (Leaning private — no external caller needs it.)
3. DAS guardrail — a warning (proposed) vs a returned cost estimate object vs both.
