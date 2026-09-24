# Tracking-scorer Memory & Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut per-unit tracking-frame memory ~2× (static-column `category`) and ship a batched pitch-control API, for every tracking consumer, with **zero retrain / re-materialize** (all changes value-neutral or parity-gated byte-identical).

**Architecture (implemented, r3):** F1a: `category`-dtype only the **STATIC** low-card columns (`ball_state`, `source_provider`, `is_goalkeeper_source`, `_preprocessed_with`) + drop all-null `confidence`; the dynamic 3 (`team_attacking_direction`/`speed_source`/`visibility`) stay `object` (category is not setitem/fillna-transparent — see `feedback_category_dtype_only_for_static_columns`). F5: bound `PitchControlCache` with an LRU. F6: document + test the one-shared-cache pattern across the two PC scorers (off_ball + rest_defense; defensive_credit is PC-free). F2: a batched `compute_pitch_control_batch` (parity-gated byte-identical). **F3: DROPPED** (warm-routing = memory regression, no CPU win — real win is a deferred vectorized kernel).

**Tech Stack:** Python 3.10, pandas (2.x/3.x span, ADR-057), numpy, pytest, ruff, pyright.

**Spec:** `docs/superpowers/specs/2026-09-24-tracking-memory-perf-design.md` (approved r2). Read it alongside this plan.

## Global Constraints

- **Commit discipline (owner rule, OVERRIDES the sub-skill's per-task commit cadence):** NO per-task commits. Build the whole coherent change with tests green throughout; a SINGLE commit at the very end, only after explicit owner approval at the commit gate (Task 6). Do NOT add `git commit` steps to Tasks 1–5.
- **One feature branch** off `main` for the whole cycle: `feat/tracking-memory-perf`. No worktree.
- **No retrain / no re-materialize.** Every change is value-neutral (category) or parity-gated byte-identical (batched pitch control). Any observed value change is a bug to fix, not accept.
- **Value-neutrality is the acceptance bar.** `category` is transparent to `==`/`.dropna()`/`pd.unique`/presence but NOT to `value_counts`/category-keyed `groupby` (zero-count categories) — every such site on the **4 static category columns** must be guarded so output is byte-identical (the 3 dynamic columns stay `object`, so they cannot hit the zero-count trap).
- **Deferred (do NOT do here):** float32 positions, id→category, off-frame provenance (all F1b — retrain); bounded-memory streaming (F4 — next cycle).
- **Full local gate before the commit gate:** `python -m ruff check silly_kicks/ tests/ scripts/`, `python -m ruff format --check silly_kicks/ tests/ scripts/`, `python -m pyright`, `python -m pytest tests/ -m "not e2e"`. Run pytest/pyright backgrounded (>30 s) and poll.
- **The 4 STATIC category columns** (set-once, never mutated post-build): `ball_state`, `source_provider`, `is_goalkeeper_source`, `_preprocessed_with`. **The 3 DYNAMIC columns stay `object`** (mutated after build — `category` is not setitem/`fillna`-transparent, so category on these raises `TypeError: Cannot setitem on a Categorical with a new category`): `team_attacking_direction` (orientation flips it), `speed_source` (velocity derivation sets it), `visibility` (`_truthy_bool` `fillna("")`). **Drop:** `confidence`.

---

### Task 1: F1a — category dtypes + drop `confidence` + value_counts guards

**Files:**
- Modify: `silly_kicks/tracking/schema.py` (`TRACKING_FRAMES_COLUMNS`, `TRACKING_CATEGORICAL_DOMAINS`, variants)
- Modify: builders `silly_kicks/tracking/gradientsports.py`, `metrica.py`, `skillcorner.py`, `sportec.py`, `kloppy.py`, `_snapshot.py` (emit `category`; drop `confidence` stamps)
- Modify: `silly_kicks/tracking/utils.py:479` + `:833` (value_counts guards)
- Modify: `silly_kicks/vaep/features/core.py:65` (drop `confidence` from the empty-frame template)
- Modify: `silly_kicks/reflection.py:140` (drop `confidence` entry)
- Test: `tests/test_tracking_schema.py`, `tests/tracking/test_frame_dtype_memory.py` (new)

**Interfaces:**
- Consumes: nothing new.
- Produces: `TRACKING_FRAMES_COLUMNS` has 19 keys (no `confidence`); the **4 static** columns (`ball_state`, `source_provider`, `is_goalkeeper_source`, `_preprocessed_with`) are `"category"` and the **3 dynamic** (`team_attacking_direction`, `speed_source`, `visibility`) stay `"object"`; `LinkReport.per_provider_link_rate` and the `speed_source` velocity-counts dict are byte-identical to the object-dtype baseline.

- [ ] **Step 1: Write the failing memory + value-neutrality test** — `tests/tracking/test_frame_dtype_memory.py`. Build a realistic-cardinality synthetic frame set (2 teams, ~16 players, a ball row, ≥2 frames), assert dtypes + a memory drop + LinkReport neutrality.

```python
import numpy as np
import pandas as pd
from silly_kicks.tracking.schema import TRACKING_FRAMES_COLUMNS


_CATEGORY_COLS = (
    "ball_state", "team_attacking_direction", "speed_source",
    "visibility", "source_provider", "is_goalkeeper_source",
)


def test_confidence_dropped_and_low_card_cols_are_category():
    assert "confidence" not in TRACKING_FRAMES_COLUMNS
    assert len(TRACKING_FRAMES_COLUMNS) == 19
    for c in _CATEGORY_COLS:
        assert TRACKING_FRAMES_COLUMNS[c] == "category", c
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/tracking/test_frame_dtype_memory.py -v`
Expected: FAIL — `confidence` still present / dtypes still `object` / len 20.

- [ ] **Step 3: Update the schema** `silly_kicks/tracking/schema.py`. In `TRACKING_FRAMES_COLUMNS` (`:10`) remove the `"confidence": "object"` entry and set `"speed_source"`, `"ball_state"`, `"team_attacking_direction"`, `"visibility"`, `"source_provider"`, `"is_goalkeeper_source"` to `"category"`. Keep `player_id`/`team_id` as `Int64`, positions `float64` (F1b). The variants (`KLOPPY_/GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS`) inherit via `**TRACKING_FRAMES_COLUMNS` — confirm they only override the id dtypes. Add the category columns to `TRACKING_CATEGORICAL_DOMAINS` where a closed vocabulary exists (already declared for `ball_state`/`team_attacking_direction`/`speed_source`/`source_provider`).

- [ ] **Step 4: Emit category + stop stamping `confidence` in the builders.** In each of `gradientsports.py`, `metrica.py`, `skillcorner.py`, `sportec.py`, `kloppy.py`, `_snapshot.py`: delete the `out["confidence"] = None` (/ `df["confidence"] = None`) stamps; ensure the 6 category columns are cast `category` at the finalize step (add a shared `.astype({c: "category" for c in ...})` in the common finalizer if one exists, else per-builder). `_preprocessed_with` is stamped `category` in `preprocess/_smoothing.py:138` (`sorted_frames["_preprocessed_with"] = pd.Categorical([tag] * len(...))` or `.astype("category")`).

- [ ] **Step 5: Guard the two `value_counts` sites (the category zero-count trap).** `tracking/utils.py:479` — change `provider_col.value_counts()` to `provider_col.astype("object").value_counts()`. (Note: `utils.py:1028` `src.value_counts()` is on `visible_area_source`, a coverage-helper output NOT in the 7 frame columns → out of scope, no guard.) `tracking/utils.py:833` — change `frames["speed_source"].value_counts(dropna=False)` to `frames["speed_source"].astype("object").value_counts(dropna=False)`. (Cast-to-object restores only-observed keys → byte-identical dict.)

- [ ] **Step 6: Drop `confidence` from the two schema-contract refs.** `vaep/features/core.py:65` — remove `"confidence"` from the template column list. `reflection.py:140` — remove the `"confidence": "invariant"` entry. Confirm the ADR-045 reflection registry-completeness meta-assertion still passes (the registry must match the frame column set).

- [ ] **Step 7: Add the value-neutrality assertions** to `tests/tracking/test_frame_dtype_memory.py`: (a) `memory_usage(deep=True).sum()` on the category frame ≤ ⅓ of the object frame; (b) build a small action set + call `value_off_ball_runs`, `add_defensive_credit`, `compute_rest_defense`, `compute_gk_decision_value` on both the object-dtype and category-dtype frames → outputs `pd.testing.assert_frame_equal` (check_dtype=False); (c) `link_actions_to_frames(...)` `LinkReport.per_provider_link_rate` equal dicts on both; (d) the `speed_source` counts (utils.py:833 consumer) equal on both.

```python
def test_category_is_value_neutral_incl_linkreport(obj_frames, actions, xt):
    cat_frames = obj_frames.astype({c: "category" for c in _CATEGORY_COLS})
    from silly_kicks.tracking import link_actions_to_frames
    _, rep_o = link_actions_to_frames(actions, obj_frames)
    _, rep_c = link_actions_to_frames(actions, cat_frames)
    assert rep_o.per_provider_link_rate == rep_c.per_provider_link_rate  # zero-count trap guarded
```

- [ ] **Step 8: Update `tests/test_tracking_schema.py`** — `test_tracking_frames_columns_is_20_columns` → 19 (rename); the required-keys set drops `confidence`; add per-variant dtype asserts that the 6 columns are `"category"`. Confirm the id-dtype-invariance + schema gates still pass (category columns are not ids).

- [ ] **Step 9: Run the full tracking + schema + reflection suites to verify pass**

Run: `python -m pytest tests/tracking/ tests/test_tracking_schema.py tests/test_reflection.py -m "not e2e" -q`
Expected: PASS; the new memory test shows each static category column drops ≥5× per-column (total materially smaller, ≈2×); LinkReport dicts equal.

---

### Task 2: F5 — `PitchControlCache` bounded LRU

**Files:**
- Modify: `silly_kicks/tracking/pitch_control/_cache.py`
- Test: `tests/tracking/pitch_control/test_cache.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `PitchControlCache(maxsize: int | None = None)`; `maxsize=None` = today's unbounded behaviour (byte-identical); a bounded cache retains ≤ `maxsize` surfaces (LRU).

- [ ] **Step 1: Write the failing test**

```python
def test_cache_lru_evicts_beyond_maxsize(five_distinct_frames):
    from silly_kicks.tracking.pitch_control import PitchControlCache
    cache = PitchControlCache(maxsize=2)
    for f in five_distinct_frames:                    # 5 distinct (game,period,frame)
        cache.surface(f, attacking_team_id=1, method="spearman")
    assert len(cache) == 2                            # only the 2 most-recent retained


def test_cache_unbounded_default(five_distinct_frames):
    from silly_kicks.tracking.pitch_control import PitchControlCache
    cache = PitchControlCache()
    for f in five_distinct_frames:
        cache.surface(f, attacking_team_id=1, method="spearman")
    assert len(cache) == 5                            # maxsize=None unchanged
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/tracking/pitch_control/test_cache.py -k lru -v`
Expected: FAIL — `PitchControlCache()` takes no `maxsize` (TypeError).

- [ ] **Step 3: Implement the LRU** in `_cache.py`. Import `OrderedDict`; `__init__(self, maxsize: int | None = None)` sets `self._maxsize = maxsize` and `self._store: OrderedDict = OrderedDict()`. In `surface`, on a hit `self._store.move_to_end(key)`; after storing, `if self._maxsize is not None and len(self._store) > self._maxsize: self._store.popitem(last=False)`.

```python
    def __init__(self, maxsize: int | None = None) -> None:
        self._store: OrderedDict = OrderedDict()
        self._maxsize = maxsize
    ...
        if key is not None and key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        surface = compute_pitch_control(...)
        if key is not None:
            self._store[key] = surface
            if self._maxsize is not None and len(self._store) > self._maxsize:
                self._store.popitem(last=False)
        return surface
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/tracking/pitch_control/test_cache.py -v`
Expected: PASS (both LRU + unbounded-default).

---

### Task 3: F6 — document + demonstrate one shared `pitch_control_cache=` across the PC scorers

**Corrected after plan review (r1):** only `value_off_ball_runs` and `compute_rest_defense` use pitch
control and **both already accept `pitch_control_cache=`**. `defensive_credit` uses NO pitch control
(verified: zero `pitch_control`/`.surface`/`compute_pitch_control` in `tracking/defensive_credit/` or
`add_defensive_credit`); `compute_gk_decision_value` is lean (F7). So there is **nothing to thread** —
this task is documentation + a real-reuse test, no scorer signature change.

**Files:**
- Modify: `silly_kicks/tracking/pitch_control/_cache.py` (docstring: the shared-cache pattern) + `silly_kicks/tracking/pitch_control/__init__.py` (module docstring note)
- Test: `tests/tracking/test_shared_pitch_control_cache.py` (new)

**Interfaces:**
- Consumes: `PitchControlCache` (Task 2), `value_off_ball_runs` + `compute_rest_defense` (both already accept `pitch_control_cache=`).
- Produces: no signature change; a documented + tested one-cache-per-unit pattern.

- [ ] **Step 1: Confirm the PC-scorer surface (audit).** `grep -rn '\.surface(\|compute_pitch_control' silly_kicks/tracking/ silly_kicks/restdefense/` → the only public scorers are `value_off_ball_runs` (`_run_values.py:441`) + `compute_rest_defense` (`restdefense/_compute.py:195`), both with `pitch_control_cache=`. Assert (in the test below) that `add_defensive_credit` does NOT take `pitch_control_cache` (it uses no PC) — so the pattern spans exactly these two.

- [ ] **Step 2: Write the failing real-reuse test** (asserts reuse, not mere growth — D2-CONSIDER-12)

```python
def test_shared_cache_computes_overlapping_frames_once(actions, frames, xt):
    from silly_kicks.tracking.pitch_control import PitchControlCache
    # independent caches: each scorer computes its own frames from scratch
    c_off, c_rd = PitchControlCache(), PitchControlCache()
    value_off_ball_runs(actions, frames, xt, pitch_control_cache=c_off)
    compute_rest_defense(actions, frames, xt, pitch_control_cache=c_rd)
    independent_total = len(c_off) + len(c_rd)
    # one shared cache: frames used by BOTH scorers are computed once
    shared = PitchControlCache()
    value_off_ball_runs(actions, frames, xt, pitch_control_cache=shared)
    compute_rest_defense(actions, frames, xt, pitch_control_cache=shared)
    assert len(shared) < independent_total        # REAL reuse (overlap computed once), not just growth
    # and defensive_credit takes no cache (uses no pitch control):
    import inspect
    assert "pitch_control_cache" not in inspect.signature(add_defensive_credit).parameters
```

**Fixture precision (D3-CONSIDER-15):** the cache key is `((game,period,frame), attacking_team_id,
method, decompose)`, NOT the frame alone. off_ball queries `(frame, action.team_id,
params.pitch_control_method, decompose=True)`; rest_defense queries `(frame, ctx.team_id, "spearman",
decompose=True)`. For the shared cache to reuse, both must hit the **same full key** — build the fixture
so at least one action is evaluated by BOTH scorers at the same `(frame, team)` with `method="spearman"`
+ `decompose=True` (confirm `params.pitch_control_method` default is `"spearman"`). A frame-only overlap
with a different team/method is a cache MISS → `len(shared) < independent_total` would fail falsely. If a
genuine shared key is impractical on the fixture, instead assert key-level reuse directly: prime the
shared cache with one `cache.surface(frame, team, method="spearman", decompose=True)`, then assert a
second identical call returns the same object (`is`) and does not grow `len(cache)`.

- [ ] **Step 3: Run to verify it fails / passes**

Run: `python -m pytest tests/tracking/test_shared_pitch_control_cache.py -v`
Expected: PASS once the fixture has an overlapping frame (no code change needed — both scorers already
accept the cache); the `inspect` assertion documents that defensive_credit is PC-free.

- [ ] **Step 4: Document the pattern** — a docstring block on `PitchControlCache` + a note in the pitch_control `__init__` module docstring: "One `PitchControlCache(maxsize=…)` per unit, passed to every PC scorer (`value_off_ball_runs`, `compute_rest_defense`), so a unit's pitch control is computed once and the shared cache is bounded."

- [ ] **Step 5: Run to verify pass**

Run: `python -m pytest tests/tracking/test_shared_pitch_control_cache.py -v`
Expected: PASS.

---

### Task 4: F2 — batched pitch-control API (parity-gated)

**Files:**
- Modify: `silly_kicks/tracking/pitch_control/_dispatch.py` (add `compute_pitch_control_batch`)
- Modify: `silly_kicks/tracking/pitch_control/__init__.py` (export it)
- Modify: `silly_kicks/tracking/pitch_control/_cache.py` (`warm` batch path)
- Test: `tests/tracking/pitch_control/test_batch.py` (new)

**Interfaces:**
- Consumes: `compute_pitch_control`.
- Produces: `compute_pitch_control_batch(frames: pd.DataFrame, requests: list[PitchControlRequest], *, method="spearman", params=None) -> list[PitchControlSurface]`, where `PitchControlRequest = tuple[tuple, int|str, bool]` = `((game_id, period_id, frame_id), attacking_team_id, decompose)`. Byte-identical to per-request `compute_pitch_control`. `PitchControlCache.warm(frames, requests, *, method, params)` populates the cache in one call.

- [ ] **Step 1: Write the failing parity test** (the load-bearing gate — ADR-076 precedent)

```python
import numpy as np


def test_batch_is_byte_identical_to_loop(multi_frame_df):
    from silly_kicks.tracking.pitch_control import compute_pitch_control, compute_pitch_control_batch
    reqs = [((1, 1, fid), 1, dec) for fid in (10, 11, 12) for dec in (False, True)]
    for method in ("spearman", "fernandez_bornn"):
        batched = compute_pitch_control_batch(multi_frame_df, reqs, method=method)
        for (key, team, dec), got in zip(reqs, batched):
            frame = multi_frame_df[
                (multi_frame_df["game_id"] == key[0])
                & (multi_frame_df["period_id"] == key[1])
                & (multi_frame_df["frame_id"] == key[2])
            ]
            ref = compute_pitch_control(frame, team, method=method, decompose=dec)
            assert np.array_equal(got.grid, ref.grid)          # max |Δ| 0.0
            if dec:
                assert np.array_equal(got.per_player_influence, ref.per_player_influence)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/tracking/pitch_control/test_batch.py -v`
Expected: FAIL — `compute_pitch_control_batch` not defined.

- [ ] **Step 3: Implement the batched entry point** in `_dispatch.py`. Correctness-first reference: group `frames` once by `(game_id, period_id, frame_id)` (via `silly_kicks._frame_index.group_rows`), dedup the request set, compute each unique `(frame, team, decompose)` surface with the existing `compute_pitch_control`, and return surfaces aligned to `requests`. The win vs the per-action loop: frames grouped ONCE + duplicate requests computed once. (A vectorised spearman kernel across unique frames is an allowed later optimisation ONLY under the Step-1 parity gate.) Export from `__init__.py` (`:17` + `__all__` at `:38`).

- [ ] **Step 4: Add `PitchControlCache.warm`** — `warm(self, frames, requests, *, method="spearman", params=None)` calls `compute_pitch_control_batch` once and stores each surface under the same `_key`, honouring `maxsize`.

- [ ] **Step 5: Run to verify pass + add the benchmark**

Run: `python -m pytest tests/tracking/pitch_control/test_batch.py -v`
Expected: PASS (parity, both methods, decompose T/F). Add a structural benchmark (per the repo's `tests/_perf_structural.py` convention) asserting the batch path's `compute_pitch_control` call-count == unique-request-count (not per-request), proving dedup/amortisation.

---

### Task 5: F3 — rest_defense layer-2 batching — **DROPPED (owner-approved, r3)**

Not implemented. `_score_samples` already computes one decompose surface per sample and reuses it
(`_danger.py:113`); samples are distinct frames → warm-routing removes no redundant work AND holds all
~254 surfaces at once (~330 MB memory regression) with no CPU win. The real 361 s→60 s needs a vectorized
cross-sample spearman kernel (deferred to the next cycle with F4; F2's batch API is its seam).
`compute_rest_defense` / `_score_samples` / `_danger.py` are unchanged (byte-identical). No new test.

**Interfaces:**
- Consumes: `compute_pitch_control_batch` / `PitchControlCache.warm` (Task 4).
- Produces: `compute_rest_defense` output byte-identical; layer-2 surfaces come from one batched warm over all sample frames.

- [ ] **Step 1: Write the failing byte-identity + perf-structural test**

```python
def test_rest_defense_batched_is_byte_identical(actions, frames, xt):
    before = compute_rest_defense(actions, frames, xt)          # pre-change reference (captured)
    after = compute_rest_defense(actions, frames, xt)           # post-change
    pd.testing.assert_frame_equal(before, after)                # identical values

def test_rest_defense_warms_pitch_control_once(actions, frames, xt):
    # structural: layer-2 computes each distinct sample frame's surface ONCE (warmed), not per derived metric
    ...  # assert compute_pitch_control call-count == n_distinct_sample_frames
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/restdefense/test_compute_perf.py -v`
Expected: FAIL — surfaces still computed per-sample (call-count == n_samples, not n_distinct_frames warmed).

- [ ] **Step 3: Route layer-2 through the batch.** In `_score_samples` (`restdefense/_compute.py:131`): before the per-sample loop, collect the distinct `(game_id, period_id, frame_id)` sample frames + `ctx.team_id` requests and `pitch_control_cache.warm(frames, requests, method="spearman", ...)` (create a local `PitchControlCache` if none passed). `layer2_metrics` (`_danger.py:79`) already prefers `pitch_control_cache.surface(...)` when a cache is present (`_danger.py:113`) — with the cache warmed, every layer-2 surface is a hit. **`spearman` + `decompose=True` STAY** (ADR-081). No value change.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/restdefense/ -m "not e2e" -q`
Expected: PASS; byte-identical output; warm call-count == distinct sample frames.

---

### Task 6: ADR + version + TODO + full gate + commit gate

**Files:**
- Create: `docs/superpowers/adrs/ADR-103-tracking-frame-category-and-batched-pitch-control.md`
- Modify: `silly_kicks/_version.py` (4.124.0 → 4.125.0), `TODO.md`

- [ ] **Step 1: Write ADR-103** (extends ADR-008/058/063/069/076/081): the category-on-frame memory model + why off-frame provenance is NOT done (per-row contracts + xt_gk retrain → F1b); the value_counts zero-count trap + guard; `PitchControlCache(maxsize)`; `compute_pitch_control_batch` + the byte-identical parity contract; rest_defense batching. Record the F4 (next cycle, ownership "Both") + F1b (float32 + id→category + off-frame provenance) deferrals + the F1b retrain trigger.

- [ ] **Step 2: Bump version** `silly_kicks/_version.py` `__version__ = "4.125.0"`.

- [ ] **Step 3: Update `TODO.md`** — add the current-block entry (branch, ADR-103, the 5 findings, no-retrain, F4/F1b deferred); verify `git status TODO.md` shows the edit.

- [ ] **Step 4: Run the full local gate** (each long one backgrounded + polled):

```bash
python -m ruff check silly_kicks/ tests/ scripts/
python -m ruff format --check silly_kicks/ tests/ scripts/
python -m pyright
python -m pytest tests/ -m "not e2e"
```
Expected: all green. Fix any failure before proceeding.

- [ ] **Step 5: Plan self-review vs spec** — confirm every spec change maps to a task (F1a→T1, F5→T2, F6→T3, F2→T4, F3→T5 [DROPPED], ADR/version→T6); no placeholder; type/name consistency (`compute_pitch_control_batch`, `PitchControlCache(maxsize=)`/`.warm`, the 4 static category cols, `confidence` gone).

- [ ] **Step 6: STOP — commit gate.** Present the diff / file list to the owner and wait for explicit approval for THIS commit (CLAUDE.md: no `git commit`/`push`/`gh pr create`/`gh pr merge` without separate explicit approval). Do not commit before the yes.

- [ ] **Step 7 (after approval): single commit** on `feat/tracking-memory-perf` (message summarising the 5 changes + no-retrain, ending with the session's attribution line), then push + PR only on the owner's further explicit go, and watch CI to green.

## Self-Review

- **Spec coverage:** F1a→T1 (category on the 4 static cols + drop confidence + value_counts guards + template/reflection refs), F5→T2 (maxsize LRU), F6→T3 (document + real-reuse test; no threading — defensive_credit is PC-free), F2→T4 (batch API + parity), F3→T5 (rest_defense batching — **DROPPED, owner-approved r3**), ADR/version/TODO→T6. No gaps.
- **Placeholder scan:** the only deferred detail is F2's optional vectorised-kernel optimisation, explicitly bounded by the Step-1 parity gate (a reference grouped-loop impl is specified) — not a placeholder.
- **Type/name consistency:** `compute_pitch_control_batch(frames, requests, *, method, params)`, `PitchControlRequest = ((game,period,frame), team, decompose)`, `PitchControlCache(maxsize=None)` + `.warm(...)`, the 4 static category columns + `confidence` dropped — consistent across T1/T2/T4 (T5/F3 dropped).
- **No-retrain:** every change value-neutral (category, guarded value_counts) or parity-gated byte-identical (batched pitch control) — proven by T1 Step 7, T4 Step 1 (T5/F3 dropped, no rest_defense code change).
