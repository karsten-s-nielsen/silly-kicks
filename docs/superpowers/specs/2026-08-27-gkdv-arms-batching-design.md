# gkdv arms batching — amortize accessible-space's per-frame setup — design

**Status:** APPROVED — lakehouse reviews #1 (verified empirically) + #2 ("approve as revised") incorporated (2026-08-27). Ready for implementation planning.
**Reported by:** lakehouse gkdv writer work (`src/ingestion/gkdv_writer.py:195-211`); handoff
`scratchpad/HANDOFF-silly-kicks-gkdv-das-batching.md`.
**Repo target:** `silly_kicks/gkdv/_arms.py`, `silly_kicks/gkdv/_das_port.py`.
**Retrain / re-materialize:** NONE. gkdv has never produced production output
(`GKDV_ENABLED = False`), so there is no persisted value to change.

The two GKDV physics arms are single-scored-frame. The lakehouse loops them once per scored
frame, so accessible-space's fixed per-call setup (frameify → grid → direction inference →
per-player indexing → model init) is paid **per frame** instead of amortized. This defeats
accessible-space's batch-native API. This spec adds batched arm entry points that make **one**
accessible-space call per leg over all a unit's scored frames, and adopts once-per-unit
direction pinning (free here, more robust).

---

## 1. Problem

Both arms take the FULL factual and counterfactual frames for **ONE scored frame**
(`_arms.py:13`), each doing the work twice (factual leg + ghost leg):

- `delta_das` → `_das_port.team_das(actual_pinned)` + `team_das(ghost_pinned)` →
  `get_individual_das(one_frame)` → one `accessible_space.get_individual_dangerous_accessible_space`
  call **per leg per frame**.
- `delta_threat_suppression` → `tracking.compute_threat_pc(actual_frame)` + `(ghost_frame)` →
  one `compute_pitch_control` surface **per leg per frame**.

The lakehouse driver calls them once per scored frame in a pure-Python loop over thousands of
scored-and-defending frames per unit (`gkdv_writer.py:195-211`), across **374 units**
(`(match, period)`). Measured **> 45 min/unit**, which exceeds the drain's **2700 s per-unit
watchdog**, so every unit is abandoned with zero output. gkdv has therefore **never** completed a
production run, and the lakehouse gated it OFF pending this fix.

### 1.1 Sizing measurement (this spec's grounding)

Synthetic DAS-scoreable unit (the `tests/gkdv/test_arms.py` frame shape stacked to N frames),
`accessible_space==2.0.15`, `numpy 2.4.2 / pandas 2.3.3`. Read-only scratch
(`scratchpad/gkdv_das_sizing.py`):

**DAS arm — looped-per-frame vs one batched call:**

| N frames | looped | batched | speedup | loop/frame | batch/frame |
|---:|---:|---:|---:|---:|---:|
| 10 | 6.28 s | 0.69 s | 9.2× | 628 ms | 69 ms |
| 30 | 19.74 s | 0.84 s | 23.4× | 658 ms | 28 ms |
| 60 | 40.79 s | 1.03 s | 39.7× | 680 ms | 17 ms |

Batched total is nearly flat (0.69 → 1.03 s for 6× the frames): fixed setup ≈ **0.6 s**, true
per-frame solve ≈ **6–8 ms**. The speedup therefore **grows with N** — extrapolated to a
~2000-frame unit, looped ≈ 2600 s → batched ≈ **~30 s (~90×)**, decisively under the 2700 s
watchdog.

**Which arm dominates:** `get_individual_das(1 frame)` = **653 ms**;
`compute_pitch_control(1 frame, spearman)` = **1.03 ms** → the DAS arm is **~630×** the threat
arm. The threat arm is 0.16 % of the wall.

### 1.2 Not already fixed

No recent release touched the DAS arm: `gkdv/_arms.py` last changed at 4.88.0, `_das_port.py` at
4.53.0, `tracking/_das.py` at 4.61.0; the 4.92.0 rescan merge (`5c4e0dd`) contained no gkdv/`_das`
files, and 4.95.0/4.96.0 were tests + CI only. **The threat arm, however, WAS already helped by the
released work** — 4.92.0 (`886351e`, ADR-068) added pitch-control loop-invariant hoists
(`_grids.py` grid memoization + `_surface.py` interpolator cache, with `test_grid_memoization.py`
and `test_surface_interp_cache.py`). That prior work is part of why the threat arm now measures
~1 ms/frame (§1.1), which is in turn the foundation of the "no vectorized spearman kernel" non-goal
(§2/§3.2). (Separately, the *cross-referenced* `HANDOFF-silly-kicks-turnover-scan.md` **is** resolved
by the same 4.92.0/ADR-068 — `_turnover.py:193` sort + `:244` numba scan — with only the lakehouse
V_opp re-fit outstanding. That is a different finding.)

---

## 2. Goals / non-goals

**Goals**
- `delta_das_batch` — one accessible-space call per leg over a unit's scored frames; the entire
  performance win.
- `delta_threat_suppression_batch` — a batch-first API so the caller makes one uniform per-unit
  call per arm (internally a thin loop; the threat arm is already negligible).
- Single-frame `delta_das` / `delta_threat_suppression` become thin wrappers over the batch, so
  their output is unchanged (a one-frame unit pins direction from that frame) — except the one
  beneficial NaN fix in §4.3.
- **Once-per-unit direction pinning** for the DAS batch (owner decision; free here, more robust).
- Two distinct regression properties: **amortization bit-exactness** and **direction semantics**.
- **NaN, never 0.0**, for an unscoreable frame in a mixed batch.

**Non-goals**
- A vectorized/batched spearman pitch-control kernel — **measured YAGNI** (the threat arm is
  0.16 % of the cost; a batched threat loop at ~1 ms/frame is a few seconds/unit). Deliberately
  not built, even under "everything in scope".
- Any change to `GkdvParams`, the domain filter, `build_ghost_frames`, or the arms' units/sign.
- Lakehouse-side changes (drop the per-frame loop, re-enable `GKDV_ENABLED`) — a separate
  handoff-back, out of this repo.
- No VAEP/tracking retrain; no re-materialize.

---

## 3. Settled design decisions (with rationale)

1. **Once-per-unit direction pin (owner decision).** The current single-frame arm pins direction
   per frame (`pin_direction(one_frame)` infers from that frame's mean-x, which can flip on a
   pathological single frame). The batch pins **once over the whole unit** — `pin_direction` run on
   the full factual stack already infers per `(period, team_in_possession)` over all its rows, the
   robust estimate. Free because gkdv has no persisted output to change; strictly more robust.
2. **No vectorized spearman kernel.** §1.1 measured the threat arm at 0.16 % of the wall. Building
   the kernel (ragged rosters, a new numba path, parity gates) buys nothing that clears the
   watchdog. Best practice here is *not* building it.
3. **Batch-first API for both arms.** Uniform caller pattern; the threat batch is a thin loop
   sharing the per-unit `goal_map`.
4. **Single-frame arms delegate to the batch** (one-frame stack). DRY — one implementation — and
   output-preserving for scoreable frames.
5. **Unit-agnostic API, keyed by `(game_id, period_id, frame_id)`.** Direction is inferred per
   `(period, team)`, so the batch is correct across multiple units in one call; per-unit is the
   documented natural granularity (bounds memory, matches the per-unit watchdog).
6. **`attacking_team_id_by_frame` accepts a scalar OR a per-frame `Series`.** A unit whose
   attacking team is constant (the common case) passes a scalar; a mixed unit passes a Series keyed
   by `(game_id, period_id, frame_id)`.

---

## 4. Architecture

### §4.1 Public API (`silly_kicks/gkdv/_arms.py`)

```python
def delta_das_batch(
    actual_frames: pd.DataFrame,       # all scored factual frames, stacked; carries (game_id, period_id, frame_id)
    ghost_frames: pd.DataFrame,        # matching ghost-substituted frames, SAME identity + row order
    *,
    attacking_team_id_by_frame: int | str | pd.Series,   # scalar, or Series keyed by the frame tuple
    params: GkdvParams = _DEFAULT_PARAMS,
) -> pd.Series:                        # index = MultiIndex(game_id, period_id, frame_id); value = das(actual) - das(ghost)

def delta_threat_suppression_batch(
    actual_frames: pd.DataFrame,
    ghost_frames: pd.DataFrame,
    *,
    attacking_team_id_by_frame: int | str | pd.Series,
    xt,
    goal_map: GoalMap,
    params: GkdvParams = _DEFAULT_PARAMS,
) -> pd.Series:                        # same index/shape
```

Both return a `pd.Series` indexed by the frame key, one value per scored frame, in
attacker-value units (negative = deterrent), matching the single-frame arms' sign.

### §4.2 DAS batch mechanics (the win)

1. **Pin direction ONCE** over `actual_frames`: `direction = _das_port.pin_direction(actual_frames)`
   (a per-row Series; `_pin_attacking_direction` infers per `(period, team_in_possession)` over the
   whole stack). Attach the SAME column to both legs, positionally
   (`frames["attacking_direction"] = direction.to_numpy()`), because the factual and ghost stacks
   are row-aligned by construction (see §4.6).
2. **One** `get_individual_das(actual_pinned, attacking_direction_col="attacking_direction")` +
   **one** `get_individual_das(ghost_pinned, ...)`. This is where 653 ms/frame → ~7 ms/frame.
3. **Reduce** per `(game_id, period_id, frame_id)`: for each frame sum `DAS` over
   `~is_ball & ids_match(team_id, attacking_team_for_that_frame)` with `min_count=1` (§4.3), on both
   legs; `delta = actual_sum - ghost_sum`.

`attacking_team_id_by_frame` is broadcast to a per-frame value before the reduce; a scalar applies
to every frame, a Series is looked up by the frame key (through `ids_match`, the ADR-019 seam, so a
value-equal scalar of a different dtype resolves identically). **A Series missing a key for any
scored frame RAISES** (fail-loud), never silently NaNs that frame: an incomplete mapping is a caller
bug, and a silent NaN would hide it (lakehouse handoff-back requirement, §10).

### §4.3 NaN correctness — never a fictional 0.0

`team_das` sums `DAS.dropna()`, and the sum of an empty selection is **0.0** — the exact fictional
zero that `tests/gkdv/test_arms.py::test_das_arm_returns_a_LIVE_FINITE_delta` documents. The per-frame
reduce therefore uses pandas `sum(min_count=1)`, so a frame with **no finite** attacking-team DAS on
either leg yields **NaN**, and `NaN - x = NaN` propagates it to the delta — never a spurious 0.0.

**The mixed-batch behaviour is VERIFIED, not assumed (review response).** The single-frame path
short-circuits on `_has_simulatable_frame` and never calls accessible-space for a lone bad frame; a
batch with ≥1 simulatable frame passes the *whole* prepared set — including non-simulatable frames —
into `get_individual_dangerous_accessible_space`. Two facts settle this:

1. **This is already a production path, not a new one.** `tracking/features.py:2859`
   (`add_das`'s single-pass architecture) *already* calls `get_individual_das` ONCE on a full
   multi-frame match including dead-ball frames. accessible-space's mixed-batch NaN is an
   established, production-relied-upon contract at the pinned version.
2. **Measured (`scratchpad/gkdv_mixed_batch_probe.py`, accessible_space 2.0.15):** a stack of one
   simulatable frame + one non-simulatable (no-ball) frame scores the good frame (DAS 72.195) and
   **NaN-s the bad frame's player rows** — no raise, order-independent (GOOD+BAD and BAD+GOOD both
   behave identically). The bad frame's `sum(dropna)` is `0.000` *without* `min_count=1`, which is
   precisely the fictional zero the reduce converts to NaN.

**Decision — guard it, do not pre-filter.** The reviewer's fallback (a batch-level pre-filter
reproducing the single-frame short-circuit) is deliberately declined: the risk it guards (a future
accessible-space that raises rather than NaN-s a bad frame in a batch) is **system-wide and
version-pinned** — `add_das` has the identical exposure — so a gkdv-local pre-filter would be an
inconsistent defense, and re-implementing `_has_simulatable_frame`'s logic at the gkdv layer breaches
the `_das_port` seam and duplicates a guard whose disjoint-subset rationale (`_das.py:344-350`) is
Chesterton-fenced. Instead the property is pinned by a **mixed-batch regression test** (§5.8): a
non-simulatable frame in the batch must come back NaN (not 0.0, not a crash), version-noted, so a
future accessible-space that changes this fails loudly — which is also the right signal for `add_das`.
Moreover gkdv's batch is pre-filtered to scored-and-defending frames (alive ball, defending GK
present — the engine domain), so its non-simulatable exposure is a strict **subset** of `add_das`'s
full-match call: the production-stability argument holds *a fortiori* (review round 2).

Consequence for the single-frame wrapper: for a genuinely-unscoreable single frame the arm still
returns NaN (unchanged); for the *scoreable* case it is unchanged; only the previously-latent
"no simulatable frame → 0.0" edge changes to NaN — a **fix**, not a regression. Execution step 5
confirms no test/consumer pins the old 0.0.

**Whole-batch `DasUnscoreableError` (velocity-less or possession-less unit).** The single-frame
arm catches `DasUnscoreableError` (raised by `pin_direction`'s DAS-input validation on a declared
velocity-less frame, or an all-NaN-`team_in_possession` dead-ball frame) and returns NaN. In the
batch this is a **whole-batch** condition — `pin_direction(actual_frames)` validates the stack's
columns and raises only when the entire batch is unscoreable (a velocity-less unit, or one with no
possession anywhere). `delta_das_batch` therefore computes the frame-key index **first**, wraps the
pin+solve in `try/except DasUnscoreableError`, and on catch returns an **all-NaN Series over those
keys** — so a one-frame velocity-less stack (through the wrapper) still yields NaN, and a wholly
velocity-less unit yields all-NaN, both correct. A *per-frame* unscoreable frame inside an otherwise
scoreable batch does **not** raise (the column exists; accessible-space NaNs that frame's rows) and
is handled by the `min_count=1` reduce above. The amortization oracle (§5.1) exercises the same
exception path in its reference loop, so the two stay consistent.

### §4.4 Threat batch (thin loop, no kernel change)

`delta_threat_suppression_batch` loops `compute_threat_pc(actual_frame_i)` / `(ghost_frame_i)` over
the frame groups, sharing the single per-unit `goal_map` and the SpearmanParams(`lambda_gk`) the
single-frame arm builds. At ~1 ms/frame this is a few seconds/unit — negligible — so the loop is
kept and no vectorized kernel is introduced. Output is bit-exact with looping the single-frame arm
(`compute_threat_pc` is already per-frame-independent given `goal_map`).

### §4.5 Single-frame wrappers (back-compat)

`delta_das(actual_frame, ghost_frame, *, attacking_team_id, params)` becomes
`delta_das_batch(actual_frame, ghost_frame, attacking_team_id_by_frame=attacking_team_id,
params=params).iloc[0]` (a one-frame stack), likewise `delta_threat_suppression`. A one-frame unit
pins direction from that frame, so scoreable-frame output is identical to today; doctests and
existing callers are unchanged.

### §4.6 Invariants preserved

- Both legs come from the SAME `build_ghost_frames` call; the caller restricts to scored frames
  (`provenance["drop_reason"].isna()`) before calling — a dropped frame is byte-identical across
  legs and must not contribute a spurious 0.0.
- **Row/key alignment guard** (generalizing the single-frame `index.equals`): `delta_das_batch`
  raises if `actual_frames` and `ghost_frames` are not aligned on `(game_id, period_id, frame_id,
  player_id)` order — the pinned direction is applied positionally, so a misaligned ghost would be
  scored against another row's direction (a per-row sign flip invisible in the returned scalar).
  `build_ghost_frames` returns the full input with only the keeper's coordinates rewritten, so the
  order is preserved for a correct caller.
- `_das_port` stays the ONLY seam onto accessible-space; the batch adds no new tracking-internal
  import (`tests/gkdv/test_import_allowlist.py` unaffected).

---

## 5. Testing strategy (TDD, red-first)

All new tests land red before implementation. Every band/counterfactual asserts **both sides**
(CLAUDE.md): a mutation that SHOULD move the number out, and that the counterfactual measurably
differs from its factual twin.

1. **Amortization oracle (bit-exact).** On a multi-frame unit, `delta_das_batch` equals a reference
   that pins the SAME once-per-unit direction column and then calls `get_individual_das` **per
   frame** with it, reducing identically. This isolates the amortization (batch vs loop of identical
   math) from the direction change. Assert equality; if `accessible_space==2.0.15` introduces
   sub-1e-9 batch-vs-loop fp noise, pin a **documented** `atol` (measured, version-noted) rather
   than claim false bit-exactness. Requires `[das]` → `importorskip`.
2. **Direction semantics (deliberate change).** On the extreme 2v2 fixture where the per-frame pin
   flips (`_GHOST_GK_X`), assert the once-per-unit pin is stable and both legs share it — i.e. the
   batch differs from *looping the OLD per-frame arm* in exactly the pathological frame, and that
   difference is the intended robustness. Structural (stub `_das_port`), runs on every leg.
3. **Threat batch == looped single-frame** (bit-exact); shares one `goal_map`.
4. **Single-frame back-compat.** `delta_das(one_frame)` / `delta_threat_suppression(one_frame)`
   equal today's output on scoreable frames.
5. **NaN-not-0.0.** A mixed batch (scoreable + a non-simulatable / velocity-less frame) yields a
   finite delta for the good frames and **NaN** for the bad one — never 0.0. Non-vacuity: reverting
   `min_count=1` turns the bad frame's delta to 0.0 and fails. Also confirm no existing test pins the
   old single-frame 0.0.
6. **Purity.** The batch functions do not mutate `actual_frames` / `ghost_frames` (they `.copy()`
   before attaching `attacking_direction`); a snapshot-equality test guards it (the ADR-033 posture,
   though these are gkdv functions, not `add_*`).
7. **Scale (optional, ADR-073 idiom).** A `tests/_perf_structural` growth check that the batch's
   accessible-space **call count** is O(1) in the number of frames (one per leg), not O(frames) —
   the structural proof that the amortization is real, complementing the wall-clock measurement in
   §1.1 (which is scratch, not a CI test).
8. **Mixed-batch contract (review response).** A batch of `[simulatable, non-simulatable]` frames
   scores the good frame and returns **NaN** for the bad one — asserting all three: not a raise, not
   a `0.0`, and the good frame unaffected (order-independent). Pins the third-party mixed-batch NaN
   behaviour so a future accessible-space that changes it fails loudly. `importorskip` + a version
   note. (The scratch probe `gkdv_mixed_batch_probe.py` established the current behaviour; this is
   its durable CI form.)

---

## 6. Constraints

- **ADR-019 id_compat** for every `team_id` compare (already in `_das_port.team_das`; the per-frame
  `attacking_team_id_by_frame` lookup routes through `ids_match`).
- **`[das]` extra:** real-scoring tests `importorskip("accessible_space")`; the structural
  direction-pin test stays stub-based so it runs on every CI leg (the existing `_das_port` posture).
- **Bit-exactness is measured, not assumed** — the amortization oracle's tolerance is chosen at the
  scale it asserts (an m² area delta), and is pinned to the accessible-space version.
- **No commit without explicit owner approval.** Version / PR-S / ADR numbers are assigned at
  commit-prep after `git fetch && git merge origin/main` (NEXT-FREE at time of writing:
  4.97.0 / PR-S168 / ADR-075 — do not reserve).
- Lint at CI scope (`ruff check/format silly_kicks/ tests/ scripts/`), bare `pyright`, full suite
  `-m "not e2e"`.

---

## 7. Execution ordering (review-tractable; NOT commit boundaries)

1. **Mixed-batch behaviour verification — DONE.** The review's step-1 concern is resolved:
   `scratchpad/gkdv_mixed_batch_probe.py` shows accessible-space 2.0.15 NaN-s a non-simulatable
   frame in a batch (no raise, order-independent), and `features.py:2859` already relies on this in
   production. Land it as the CI regression test §5.8 first, so the property is guarded before the
   API is built. (Optionally also re-confirm the §1.1 timing on a handful of **real** pined scored
   frames — the synthetic result is already decisive and the §5.7 call-count test is the durable
   proof.)
2. `delta_das_batch` + amortization oracle (§5.1) + direction-semantics test (§5.2), red-first.
3. Per-frame reduce with `min_count=1` + NaN-not-0.0 test (§5.5).
4. Row/key-alignment guard + missing-key fail-loud (§4.2, §4.6) + their negative tests.
5. `delta_threat_suppression_batch` + oracle (§5.3).
6. Single-frame wrappers delegate to the batch (§4.5) + back-compat tests (§5.4); confirm no test
   pins the old 0.0 edge (§4.3).
7. Purity (§5.6) + optional scale guard (§5.7).
8. Docs: function docstrings (Examples per the public-API examples gate), `CHANGELOG.md`, the
   CLAUDE.md gkdv bullet (note the batch arms + once-per-unit pin), `NOTICE` unchanged (no new
   methodology). C4: gkdv arms already modeled, no new action-coupled aggregator — verify count
   unchanged, no re-render needed unless the C4 gate flags drift.

---

## 8. Known limits (stated, not discovered)

- The amortization oracle proves **batch == loop given the same direction**; it does NOT prove
  once-per-unit is "more correct" than per-frame. The pathological-fixture test shows the
  once-per-unit pin is *stable* where per-frame flips — stability, not ground truth. The robustness
  argument stands on the physics (a team's attacking direction is constant within a period), not on
  the test.
- Batch-vs-loop bit-exactness is a property of `accessible_space==2.0.15`; a version bump could
  shift the low bits. The pinned tolerance is documented and version-noted so a bump lands as a
  visible test delta, not a silent drift.
- The batch is memory-bound at very large N (many units in one call): `get_individual_das` holds the
  frameified wide arrays for all frames at once. At the confirmed per-`(match, period)` granularity
  (~2–4 k frames/unit) the wide arrays are tens of MB — **not a real risk** (lakehouse reviewer); the
  concern is theoretical and only bites a caller that batches many units into one call. The API does
  not cap N; per-unit is the documented granularity.
- **Mixed-batch NaN is a version-pinned third-party contract, not a silly-kicks guarantee.** The
  design relies on accessible-space NaN-ing (not raising on) a non-simulatable frame inside a batch —
  verified at 2.0.15, and already depended on by `add_das`. A batch-level pre-filter was declined
  (see §4.3); the §5.8 regression test converts the dependency into a loud failure if a future
  version changes it. If that ever fires, the correct-layer fix is a per-frame simulatable filter in
  `get_individual_das` (protecting `add_das` too), not a gkdv-local workaround.
- **Memory-capped context (forward-note, not a finding — review round 2).** gkdv scoring runs in the
  drain's 16 GB driver, where a per-unit batch (tens of MB) is comfortable, so `delta_das_batch`
  makes one un-chunked `get_individual_das` call per leg. The `add_das` precompute path
  (`features.py`, `chunk_size` param) already chunks frames before the DAS call for memory-capped
  contexts (e.g. a Databricks `applyInPandas` 1 GB group cap). If the drain topology ever moves gkdv
  scoring into such a context, `delta_das_batch` should adopt the same chunk-then-concat pattern —
  moot today, recorded so the lever is known.
- The threat arm remains O(frames) per unit at ~1 ms/frame — fine now. If a future provider makes
  the GK-aware surface materially heavier, revisit the vectorized-kernel lever then (out of scope
  now, and recorded so the deferral is a decision, not an omission).

---

## 9. Open questions — RESOLVED by the lakehouse review

- **Combined helper? → No — two composable functions.** The lakehouse joins the two Series on the
  shared `(game_id, period_id, frame_id)` MultiIndex in one line and attaches the keeper `player_id`
  from provenance; a combined DataFrame-returning call would couple the arms for no gain. Matches the
  single-frame pair.
- **Batch granularity? → per-`(match, period)` unit.** Exactly the drain's enumeration unit, matches
  the per-unit watchdog, and bounds the frameified working set (§8). Per-match is *correct* but
  needlessly widens it.
- **Tolerance? → empirical, version-pinned** ("bit-exact, else a measured documented `atol`",
  §5.1/§8). Agreed.

## 10. Lakehouse handoff-back requirements (record now; enacted at the eventual re-enable)

Lakehouse-side changes stay out of this repo (§2), but these are recorded so the re-enable is clean
and so this spec's API meets them:

- **Filter both legs identically** (scored-and-defending, same row order) so the §4.6 alignment guard
  passes — do not filter actual/ghost independently.
- **Treat a NaN delta as "exclude," never 0.** The 0.0 → NaN change (§4.3) is the exact null-bias the
  drop-reason exclusion already fights; `pool_keepers` / `build_keeper_observations` must honour it.
- **Build a complete `attacking_team_id_by_frame`** (one entry per scored frame). silly-kicks
  **fails loud on a missing key** (§4.2) rather than silently NaN-ing that frame — the API side of
  this requirement.
- On re-enable: drop the `gkdv_writer.py:195-211` per-frame loop, call each batched arm once per unit,
  set `GKDV_ENABLED = True`, run `preflight_tracking_marts --full`.
