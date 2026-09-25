# Tracking performance — vectorized spearman kernel, scorer batching & bounded-memory streaming — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make the tracking pitch-control path fast + bounded-memory for every consumer, with byte-identical output — a vectorized cross-frame spearman kernel, scorer routing that closes F3, and F4 internal auto-batching.

**Architecture:** A vectorized spearman kernel (3D pad/mask stack) sits behind the existing public `compute_pitch_control_batch` seam; the whole-unit scorers route their per-item pitch-control through it and stream their loops in bounded chunks. All value-neutral or parity-gated byte-identical.

**Tech Stack:** Python, numpy (broadcast reductions), pandas, optional numba (ADR-076 pattern). No new runtime dependency.

**Spec:** `docs/superpowers/specs/2026-09-24-tracking-perf-vectorized-kernel-and-streaming-design.md` (r2-APPROVED). Reviewer report `D:\Development\_reviews\2026-09-24-sk-tracking-perf-vectorized-kernel-streaming-spec-r2.md`.

## Global Constraints

- **No retrain / no re-materialize / C4-free.** Every change is value-neutral or parity-gated byte-identical (`np.array_equal`, max |Δ| exactly 0). Any observed value change is a bug to fix, not accept.
- **Commit discipline (owner rule, OVERRIDES the sub-skill's per-task cadence):** NO per-task commits. Build the whole coherent change with tests green throughout; a SINGLE commit at the very end, only after explicit owner approval at the commit gate (Task 8). Do NOT add `git commit` steps to Tasks 1–7.
- **One feature branch** off `main`: `feat/tracking-perf-kernel-streaming` (already created). No worktree.
- **No version number anywhere until commit-prep (Task 8).** ADR is provisional `ADR-105`; version is the next-free minor assigned at commit-prep (`git fetch && git merge origin/main` first — main is 4.126.0, TF-56 shipped since ADR-103).
- **Byte-identity is load-bearing and SUBTLE (Task 1):** the padded player-axis reduction is bit-identical to the per-frame sum ONLY because real per-frame valid-player counts stay below numpy's pairwise-summation threshold (128), so the reduction is SEQUENTIAL and a masked-to-0.0 padding row is an exact no-op. This is a correctness precondition, tested explicitly.
- **ADR-043 counterfactual-cache landmine:** counterfactual (player-removed/moved) surfaces MUST NEVER route through the batch/cache — only CANONICAL frames. The counterfactual paths stay direct.
- **Full local gate before the commit gate:** `python -m ruff check silly_kicks/ tests/ scripts/`, `python -m ruff format --check silly_kicks/ tests/ scripts/`, `python -m pyright`, `python -m pytest tests/ -m "not e2e"`. Run pytest/pyright backgrounded (>30 s) and poll. Mirror `ci.yml` exactly.
- **ADR-073 scale guards:** any NEW `group_rows` caller registers a growth guard in `tests/_scale_guarded.SCALE_GUARDED`.

---

### Task 1: Vectorized cross-frame spearman kernel (Change 1)

**Files:**
- Create: `silly_kicks/tracking/pitch_control/_spearman_batch.py` (the vectorized kernel)
- Modify: `silly_kicks/tracking/pitch_control/_dispatch.py` (`compute_pitch_control_batch` routes spearman requests to the kernel; keeps the dedup + non-spearman fallback to the per-frame loop)
- Modify: `silly_kicks/tracking/pitch_control/_spearman.py` (extract the within-frame TTI→influence→combine into a single-sourced core the per-frame path and the batch both call; the per-frame path becomes a thin N=1 caller so it stays byte-identical — ADR-102 idiom)
- Test: `tests/tracking/pitch_control/test_batch.py` (extend), `tests/tracking/pitch_control/test_spearman_batch.py` (new)

**Interfaces:**
- Consumes: `pitch_grid` (fixed grid), `compute_tti`/`_compute_influence` cores (`_spearman.py`), `SpearmanParams`.
- Produces: `_spearman_batch.compute_spearman_batch(frame_slices: list[pd.DataFrame], requests: Sequence[tuple[int|str, bool]], *, params: SpearmanParams, ball_positions: list[tuple[float,float]|None]) -> list[PitchControlSurface]` — one surface per request, byte-identical to `compute_spearman` per slice. `compute_pitch_control_batch` (public) uses it for `method="spearman"`.

- [ ] **Step 1: Write the failing byte-identity parity test (ragged + padding + order)** — `tests/tracking/pitch_control/test_spearman_batch.py`.

```python
import numpy as np, pandas as pd
from silly_kicks.tracking.pitch_control import compute_pitch_control, compute_pitch_control_batch

def _frame(fid, players, ball=(50.0, 34.0)):
    rows = [dict(game_id=1, period_id=1, frame_id=fid, is_ball=False, is_goalkeeper=g,
                 team_id=t, player_id=pid, x=x, y=y, vx=vx, vy=vy)
            for (pid, t, g, x, y, vx, vy) in players]
    rows.append(dict(game_id=1, period_id=1, frame_id=fid, is_ball=True, is_goalkeeper=False,
                     team_id=np.nan, player_id=np.nan, x=ball[0], y=ball[1], vx=0.0, vy=0.0))
    return pd.DataFrame(rows)

def test_batch_byte_identical_ragged_padded_unsorted():
    # Frame A: 5 players. Frame B: 3 players (padding vs A) in a DIFFERENT team/gk order (order sensitivity).
    A = _frame(10, [(1,1,True,8,34,0,0),(2,1,False,60,20,1,0),(3,2,True,45,40,0,1),
                    (4,2,False,70,34,0,0),(5,1,False,30,50,-1,0)])
    B = _frame(11, [(9,2,False,70,34,0,0),(8,1,True,8,34,0,0),(7,2,True,45,40,0,1)])  # unsorted
    frames = pd.concat([A, B], ignore_index=True)
    reqs = [((1,1,10), 1, dec) for dec in (False, True)] + [((1,1,11), 1, dec) for dec in (False, True)]
    got = compute_pitch_control_batch(frames, reqs, method="spearman")
    for (fk, team, dec), g in zip(reqs, got, strict=True):
        ref = compute_pitch_control(frames[frames["frame_id"] == fk[2]], team, method="spearman", decompose=dec)
        assert np.array_equal(g.surface, ref.surface), (fk, dec)   # max |Δ| exactly 0
        if dec:
            assert np.array_equal(g.per_player_influence, ref.per_player_influence)
```

- [ ] **Step 2: Run it — expect FAIL** (`compute_pitch_control_batch` still loops per frame, so this passes trivially IF the loop is already correct; the point of the batch kernel is speed — so this test PASSES on the loop baseline and MUST stay green after the kernel replaces the loop). Run: `pytest tests/tracking/pitch_control/test_spearman_batch.py -v`. Expected: PASS on baseline (it is the parity oracle the kernel must preserve).

- [ ] **Step 3: Implement `compute_spearman_batch`** — `_spearman_batch.py`. Per request: build padded `pos (N, P, 2)`, `vel (N, P, 2)`, `team (N, P)`, `is_gk (N, P)`, `valid_mask (N, P)` where `P = max valid players over the batch`; drop ball + NaN-position rows per frame (identical filter to `_spearman.py:149-150`), preserving each frame's own surviving row order. Run TTI + influence broadcast over the frame axis (reuse the `_spearman.py` cores). Reduce per team with masked (padding→0.0) rows, summing over the player axis in each frame's OWN order. Apply the identical min/logistic/GK×lambda/ball-zeroing/ratio (`att_sum/(att_sum+def_sum)`, 0.5 both-zero). **Load-bearing:** assert (comment + a test in Step 5) that per-frame valid `P < 128` so the reduction is sequential and zero-padding is exact.

- [ ] **Step 4: Route `compute_pitch_control_batch` through the kernel for spearman** — `_dispatch.py`: for `method == "spearman"`, dedup as today, then call `compute_spearman_batch` over the distinct requests; other methods keep the per-frame loop. Non-spearman + counterfactual callers unaffected.

- [ ] **Step 5: Add the pairwise-regime + mutation guards** — in `test_spearman_batch.py`: (a) a test that a frame with e.g. 40 players (still < 128) stays byte-identical; (b) two mutation guards — patch the kernel to skip the mask (padding leaks) → assert the parity test would FAIL; patch to permute the stacked valid rows → assert FAIL. (Use `monkeypatch`/a local mutated copy; the mutations prove the fixture is discriminating, VKS-SPEC-02.)

- [ ] **Step 6: Run the extended `test_batch.py` + the new file** — Run: `pytest tests/tracking/pitch_control/ -v`. Expected: PASS (byte-identical), both mutation guards RED-then-caught.

- [ ] **Step 7: Register the ADR-073 growth guard** — `compute_spearman_batch` scales the frame/request dimension. Add a `SCALE_GUARDED` entry + a `tests/test_scale_guards.py::test_compute_spearman_batch_is_subquadratic` (stub the within-frame math if needed; scale the distinct-frame dimension).

- [ ] **Step 8: Benchmark** — add a wall benchmark (standalone `benchmark` job / `@pytest.mark.slow`) showing the batch < the per-frame loop on a multi-hundred-frame batch (no `assert ms<budget`; a structural op-count guard is the CI gate, the wall number is reported).

---

### Task 2: `compute_threat_pc` batched entry + route rest_defense layer-2 (Change 2 — closes F3)

**Files:**
- Modify: `silly_kicks/tracking/_cover_shadows.py` (add a private `compute_threat_pc_batch` wrapping the batch kernel; `compute_threat_pc` L826 stays the single-frame public entry, byte-identical)
- Modify: `silly_kicks/restdefense/_danger.py` (`layer2_metrics` batches its 4 per-sample surfaces across the unit's samples)
- Modify: `silly_kicks/restdefense/_compute.py` (`_score_samples` drives the batched layer-2 over all samples)
- Test: `tests/restdefense/test_danger_batch_parity.py` (new), extend `tests/restdefense/` goldens

**Interfaces:**
- Consumes: `compute_spearman_batch` / `compute_pitch_control_batch` (Task 1).
- Produces: `compute_threat_pc_batch(frames_and_teams, *, xt, goal_map, field_weight) -> list[np.ndarray]` (private, consumed only by the routed scorers — Q2 resolved private). `compute_rest_defense` output byte-identical.

- [ ] **Step 1: Write the byte-identity golden test for rest_defense layer-2** — `tests/restdefense/test_danger_batch_parity.py`: compute `compute_rest_defense` output on a committed multi-sample fixture BEFORE routing (capture as the oracle), assert the routed version is `assert_frame_equal(check_dtype=False)` identical. Also assert the ADR-043 landmine: the counterfactual/GK-blind `frame_no_gk` leg is NOT served from the canonical cache.

- [ ] **Step 2: Run it — expect FAIL** (`compute_threat_pc_batch` not defined). Run: `pytest tests/restdefense/test_danger_batch_parity.py -v`. Expected: FAIL (ImportError / not routed).

- [ ] **Step 3: Implement `compute_threat_pc_batch`** — batch the two `compute_threat_pc` legs (`frame` + `frame_no_gk`) across samples via the Task-1 kernel; the GK-blind leg is a distinct frame slice (`frame[~ids_match(...)]`) — batch it as its own slice set, NEVER via the canonical cache.

- [ ] **Step 4: Route `layer2_metrics` + `_score_samples` — CHUNKED from the start (VKS-PLAN-07).** Do NOT compute all ~250 samples × ~4 surfaces (~1000 surfaces) in one batch — that is exactly the F3 warm-routing memory regression ADR-103 rejected. Stream the samples in bounded chunks (an internal chunk size; Task 4 promotes it to the public `batch_size` param), computing each chunk's surfaces via the Task-1 batch, running that chunk's per-sample derived ops (`player_surface`/`control_in_region`/`compute_gk_influence`/threat legs), emitting the chunk's rows, and **releasing the chunk's surfaces before the next chunk**. Peak = O(chunk) surfaces, never O(all samples). Preserve per-sample output order + values (byte-identical). **rest_defense is thus already chunked when it lands — T4 only adds the public param + the invariance gate + the default, so there is no transient all-at-once state between T2 and T4.**

- [ ] **Step 5: Run rest_defense parity + full restdefense suite** — Run: `pytest tests/restdefense/ -m "not e2e" -v`. Expected: PASS (byte-identical).

- [ ] **Step 6: Benchmark F3** — a `@pytest.mark.slow`/benchmark-job measurement that `compute_rest_defense` on a ~250-sample fixture is materially faster (target order: tens of seconds not minutes at real scale; reported, not asserted-in-CI). Structural op-count guard: pitch-control surface computes now happen in batches, not one-call-per-sample-per-leg (call-count spy on `compute_pitch_control`).

---

### Task 3: Route off_ball through the batch + audit the PC-free scorers (Change 2 cont.)

**Scope correction (VKS-PLAN-05):** `defensive_credit/` and `gk_decision/_reconstruct.py` are **PC-FREE** — verified zero `pitch_control`/`.surface(`/`compute_pitch`/`compute_threat_pc`/`compute_gk_influence` references (consistent with 4.125.0's F6, which established `add_defensive_credit` takes no `pitch_control_cache` because it is PC-free). The **only** pitch-control-consuming scorers are **off_ball (`value_off_ball_runs`, `_run_values.py`, uses `PitchControlCache`) + rest_defense (Task 2)**. So this task routes **off_ball ONLY**; routing the PC-free scorers would be dead edits + a vacuous call-count guard (the F6/D3-PLAN-11 class).

**Files:**
- Modify: `silly_kicks/tracking/_run_values.py` (`value_off_ball_runs` per-action canonical surfaces via the batch)
- Test: extend `tests/tracking/` off_ball golden/parity; add `tests/tracking/test_pc_free_scorers_audit.py` (the PC-free assertion)

**Interfaces:**
- Consumes: Task 1 batch.
- Produces: off_ball output byte-identical.

- [ ] **Step 1: Write/extend the off_ball byte-identity golden** capturing pre-routing output as the oracle. Run: `pytest <that test> -v`. Expected: PASS on baseline (oracle).
- [ ] **Step 2: Route off_ball's canonical (cache-eligible) surface computes through the batch** — the global pass (`frame_groups`/link) is unchanged; only the per-action surface fetch is batched. Any counterfactual surface stays direct (ADR-043).
- [ ] **Step 3: Run the off_ball suite** — Run: `pytest tests/tracking/ -m "not e2e" -k "off_ball or run_value" -v`. Expected: PASS byte-identical.
- [ ] **Step 4: Structural guard** — a `compute_pitch_control` call-count spy shows off_ball's per-action loop no longer calls it once-per-action (it batches).
- [ ] **Step 5: PC-free audit test** — `tests/tracking/test_pc_free_scorers_audit.py`: an AST/grep assertion that `defensive_credit/` + `gk_decision/_reconstruct.py` reference NO pitch-control symbol (pins the fact so a future edit that adds PC there fails CI, prompting a routing decision). Mirrors the corrected-F6 audit shape.

---

### Task 4: F4 — internal auto-batching + `batch_size` on the PC-consuming scorers (Change 3)

**Scope (VKS-PLAN-06):** `batch_size` bounds SURFACE retention, so it belongs on the **PC-consuming scorers ONLY — off_ball + rest_defense**. `defensive_credit` + `gk_decision` are PC-free (Task 3): they hold no surfaces, so their peak is the (shared, F1a-shrunk ~630 MB) frames-hold + a small per-item working set → **already under the ≤1 GB budget post-F1a without chunking**; adding `batch_size` there would bound nothing (documented, not a task). If a benchmark shows a PC-free scorer still exceeds budget, THAT is a new finding to surface, not a silent param add.

**Files:**
- Modify: `silly_kicks/tracking/_run_values.py`, `silly_kicks/restdefense/_compute.py` (each gains keyword-only `batch_size: int | None`)
- Test: `tests/tracking/test_scorer_batch_invariance.py` (new)

**Interfaces:**
- Consumes: Task 1–3 routing (rest_defense already chunks internally from Task 2 — see VKS-PLAN-07).
- Produces: off_ball + rest_defense accept `batch_size`; peak = frames-hold + O(batch) surfaces.

- [ ] **Step 1: Write the batch-size invariance test PER routed scorer** — `tests/tracking/test_scorer_batch_invariance.py`:

```python
import pandas as pd, pytest
# for each routed scorer + its committed fixture:
@pytest.mark.parametrize("batch_size", [None, 1, 7, "len"])
def test_scorer_output_invariant_to_batch_size(batch_size, scorer, fixture):
    bs = len(fixture.items) if batch_size == "len" else batch_size
    out = scorer(fixture.actions, fixture.frames, batch_size=bs, **fixture.kw)
    pd.testing.assert_frame_equal(out.reset_index(drop=True),
                                  fixture.oracle.reset_index(drop=True), check_dtype=False)
```

- [ ] **Step 2: Run it — expect FAIL** (`batch_size` param not present). Expected: TypeError.
- [ ] **Step 3: Implement chunked streaming per scorer** — after the global pass, iterate the per-item loop in chunks of `batch_size`: compute the chunk's surfaces (Task 1 batch), emit the chunk's rows, release the chunk's surfaces before the next chunk (compute-and-discard; F5 `maxsize` bounds any retained cache). `batch_size=None` = whole-loop (current behavior).
- [ ] **Step 4: Run the invariance suite** — Expected: PASS for `{None,1,7,len}` per scorer.
- [ ] **Step 5: Flip the default to a bounded `batch_size` (e.g. 256) — GATED** — only flip a scorer's default AFTER its Step-4 invariance is green (VKS-SPEC-03 merge prerequisite). A scorer whose invariance is not green keeps `batch_size=None`. Re-run the full scorer suites after the flip.
- [ ] **Step 6: Bounded-peak structural guard** — a surface-retention count (or a `tracemalloc`-based peak assertion in a `@slow` test) shows peak scales with `batch_size`, not unit size.

---

### Task 5: ADR-103 shipped-code fixes (Change 4)

**Files:**
- Modify: `silly_kicks/tracking/pitch_control/_dispatch.py` (`compute_pitch_control_batch` optionally returns/exposes the grouping) + `_cache.py` (`warm` reuses it; `_key` takes the known frame-key instead of re-deriving)
- Test: `tests/tracking/pitch_control/test_cache.py` (extend), `tests/tracking/test_scale_guards.py` (the `warm` group-count)

- [ ] **Step 1: Write a `group_rows` call-count guard for `warm`** — assert `warm` builds `group_rows` exactly ONCE (currently twice: `_cache.py:143` batch + `:144` re-group). Use a `call_counter` on `_frame_index.group_rows`. Expected: FAIL (count == 2).
- [ ] **Step 2: Implement** — have `compute_pitch_control_batch` return (or `warm` reuse) the per-request frame / grouping so `warm` does not re-`group_rows`. Byte-identical surfaces + keys.
- [ ] **Step 3: `_key` scan** — where the caller already knows `(game_id, period_id, frame_id)` (batch/`warm`), pass it through instead of `.dropna().unique()` re-deriving. Add a parity test: the produced key is identical to the re-derived one. Keep `.surface()`'s standalone path (single arbitrary frame) unchanged.
- [ ] **Step 4: Run** — `pytest tests/tracking/pitch_control/ -v`. Expected: PASS, `warm` group-count == 1.

---

### Task 6: `group_rows(observed=True)` + pandas-major guard (Change 5)

**Files:**
- Modify: `silly_kicks/_frame_index.py:38` (`groupby(list(by), sort=False, observed=True)`)
- Test: `tests/test_frame_index.py` (extend)

- [ ] **Step 1: Write the category-key test** — build a frame with a `category` key column whose declared categories are WIDER than observed; `group_rows` on it must yield ONLY observed groups (no phantom empty keys), and emit no `FutureWarning`. Run across the pandas-major span idiom (assert behaviour, not a version literal — ADR-057). Expected: FAIL (phantom empty group present / warning).
- [ ] **Step 2: Implement** — add `observed=True` to the `groupby` call. No current caller keys on a category column → byte-identical today.
- [ ] **Step 3: Run** — `pytest tests/test_frame_index.py -v`. Expected: PASS (observed-only, no warning). Full `group_rows`-consumer suites unchanged.

---

### Task 7: DAS all-frame cost guardrail (Change 6)

**Files:**
- Modify: `silly_kicks/tracking/_das.py` (a pure cost-estimate fn + a one-time opt-out warning on a full-unit call)
- Modify: `silly_kicks/tracking/_warnings.py` (a dedicated warning category, per the "separate categories" convention)
- Test: `tests/tracking/test_das_cost_guardrail.py` (new)

- [ ] **Step 1: Write the test** — a full-unit DAS call emits the cost warning ONCE (opt-out honored); the cost fn is pure (deterministic on inputs, no side effects). Expected: FAIL (no warning / fn absent).
- [ ] **Step 2: Implement** — a pure `estimate_das_cost(...)` + a `warnings.warn(..., DasCostWarning, stacklevel=2)` fired once when DAS runs over a full unit without sampling; opt-out kwarg/flag. No output/behavior change to DAS values.
- [ ] **Step 3: Run** — `pytest tests/tracking/test_das_cost_guardrail.py -v` + `tests/tracking/test_das.py`. Expected: PASS, DAS values unchanged.

---

### Task 8: ADR + version + docs + C4 check + full gate + commit gate

**Files:**
- Create: `docs/superpowers/adrs/ADR-105-...md` (number assigned at commit-prep)
- Modify: `silly_kicks/_version.py` (bump at commit-prep), `CHANGELOG.md`, `TODO.md`, `CLAUDE.md` (a durable-contract bullet: the vectorized kernel byte-identity precondition + F4 internal-batching + the `group_rows(observed=True)` pin)
- Verify: C4 (`docs/c4`) — this cycle adds NO aggregator/container (count 33 unchanged); confirm the completeness gate stays green (no regen unless it drifts)

- [ ] **Step 1: Re-derive numbering** — `git fetch && git merge origin/main`; assign the next-free minor + PR-S + ADR number (main is 4.126.0). Write the ADR (Accepted) from the spec's §11.
- [ ] **Step 2: Bump `silly_kicks/_version.py`** (single source, ADR-079) + add the `[<version>]` CHANGELOG entry + the TODO NOW-block + the CLAUDE.md bullet.
- [ ] **Step 3: Full local gate** — `ruff check` + `ruff format --check` (CI scope) + `pyright` + `pytest tests/ -m "not e2e"` (backgrounded, polled). ALL green. Fix any full-suite-only registry-gate failures (public-API examples for new symbols; ADR-073 for new `group_rows` callers; liveness/purity/mirror as applicable) — the ADR-103 lesson.
- [ ] **Step 4: STOP — commit gate.** Present the diff / file list to the owner and wait for explicit approval for THIS commit (no `git commit`/`push`/`gh pr` without separate explicit approval).
- [ ] **Step 5 (after approval): single commit** on `feat/tracking-perf-kernel-streaming` (message summarising the changes + no-retrain, ending with the session attribution line), then push + PR only on the owner's further explicit go, and watch CI to green.

---

## Self-Review

- **Spec coverage:** Change1→T1, Change2→T2 (rest_defense) + T3 (off_ball), Change3→T4, Change4→T5, Change5→T6, Change6→T7, ADR/version/docs→T8. All 6 spec changes + the deferred-#3 (documented in spec §9, not a task) mapped. No gaps.
- **r1 corrections folded in:** the PC-consuming scorer set is **off_ball + rest_defense ONLY** — `defensive_credit` + `gk_decision` are verified PC-free, so T3 routes off_ball only + audits the two PC-free (VKS-PLAN-05), T4's `batch_size` is scoped to the two PC-consumers (VKS-PLAN-06), and T2 chunks rest_defense from the start so there is no transient ~1000-surface regression before T4 (VKS-PLAN-07).
- **Placeholder scan:** `ADR-105` + "next-free minor" are intentional (numbering at commit-prep per the Global Constraint), not placeholders. Every code step names files + the concrete change; test steps carry real snippets or a precise assertion.
- **Type/name consistency:** `compute_spearman_batch` (T1) / `compute_threat_pc_batch` (T2, private) / `batch_size: int | None` keyword-only (T4) / `group_rows(..., observed=True)` (T6) / `estimate_das_cost` + `DasCostWarning` (T7) — consistent across tasks and matched to the spec.
- **Byte-identity precondition:** the pairwise-summation-regime (`P < 128` → sequential → zero-padding exact) is called out in the Global Constraints AND T1 Step 3/5 — the load-bearing correctness detail the reviewer flagged as ACHIEVABLE.
- **No-retrain:** every task is value-neutral (T5/T6/T7) or parity-gated byte-identical (T1–T4) — proven by the oracle/golden + `np.array_equal`/`assert_frame_equal` gates, and the SB360 audit re-derives unchanged.
