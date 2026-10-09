# TF-65 + TF-64 Kinematics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix velocity-gap contamination, bake off the kinematics smoother (Butterworth vs Kalman/RTS vs a Savitzky-Golay + dense-grid floor) via a pre-registered in-cycle A/B, add acceleration + per-frame uncertainty, and wire the GK detection-gate — all against one batched retrain.

**Architecture:** Hexagonal, additive. All preprocess functions stay pure (pandas in/out). An internal dense-grid reindex makes smoothing/velocity respect `frame_id` spacing; the smoother is pluggable via `PreprocessConfig.smoothing_method`; acceleration + an always-run Kalman/RTS uncertainty pass ship unconditionally (decoupled from the point smoother); TF-64 wraps the shipped `detected_mask` primitive over the keeper-position readers.

**Tech Stack:** Python, numpy, scipy (`signal`, `sparse` not needed — spline arm dropped), pandas. No new runtime deps.

**Spec:** `docs/superpowers/specs/2026-10-09-tf65-tf64-kinematics-design.md` — the plan argues from the spec; executors read both.

## Global Constraints

- **COMMIT POLICY (overrides the writing-plans default — no exceptions).** NO per-task, per-step, or "frequent" commits. Each task ends at **"full CI-faithful suite green"**, not a commit. Commits happen ONLY at the phase boundaries in spec §10 (C1 / C2 / C3 / CN), each one a **coherent, fully-tested state** gated by the owner's **explicit per-commit approval**. Do not run `git commit`, `git push`, or `git tag` in any task. No micro-commits.
- **Branch:** work on `feat/tf65-tf64-kinematics` (already created, off `main`). No worktrees, no parallel checkouts.
- **CI-faithful verification (every task's "suite green"):** `python -m pytest tests/ -m "not e2e"` (full, never a package-scoped subset) **and** `python -m ruff check silly_kicks/ tests/ scripts/` **and** `python -m ruff format --check silly_kicks/ tests/ scripts/` **and** bare `pyright`. A task is done only when all four are green.
- **TDD, red first:** write the failing test, run it to see it fail for the stated reason, then the minimal implementation, then green.
- **Version:** do NOT write a version number until commit-prep (phase CN). `silly_kicks/_version.py` is the single source (ADR-079); bump that one file at CN only.
- **No local paths** in any committed file (spec, plan, docstrings, tests). Data is sourced via pining / the provider abstraction; private locations are owner-supplied variables.
- **Docs are not committed until the cycle ends** (owner standing rule) — the spec + this plan travel uncommitted until CN.
- **Subagent model routing (if dispatched):** reading/search → `haiku`; research → `sonnet`; any code write/edit/debug/architecture → `opus`.
- **All `warnings.warn(..., stacklevel=2)`; warning categories are separate (never one umbrella).** Converters/enrichers vectorize (`np.select`), never `apply(axis=1)`.
- **Float32 storage / float64 compute (ADR-106):** kinematic columns store float32; every kernel upcasts its slice to float64 at the boundary.
- **A/B pre-registration is frozen in spec §6 and the cycle ADR BEFORE the A/B runs** — no post-hoc metric selection; the floor is a legitimate outcome.

---

## Reconciliation (impl r1, owner-approved 2026-10-09) — supersedes the stale task text below

Two impl findings reconciled; this block is authoritative where it conflicts with Task 5/7/10 wording:

1. **Acceleration component columns are `accel_x`/`accel_y`, NOT `ax`/`ay`.** `ax`/`ay` already exist as
   `_kernels.py` triangle anchor-coordinate columns in `add_action_context`'s merge; a frame `ax`/`ay`
   collides (`KeyError: 'ax'`). Every `ax`/`ay` in the acceleration snippets below reads `accel_x`/`accel_y`;
   `accel`/`pos_var`/`vel_var`/`accel_var` unchanged. Contract in spec §7.
2. **Derived kinematic columns are NOT declared in `TRACKING_FRAMES_COLUMNS` / `metric_contracts` /
   `feature_glossary`** — they follow the `vx`/`vy` convention (preprocess-added; `schema.py` precedent;
   `vx`/`speed` absent from all three). **Task 5/7/10's schema + metric_contracts + glossary registration
   sub-tasks are STRUCK** (the 409-test contract gate confirms green without them). Reflection + NOTICE
   additions stand.
3. **Preprocess purity is gated by `tests/tracking/preprocess/test_preprocess_purity.py`, NOT
   `PURITY_ENTRIES`** — the latter is `add_*`-exact (ADR-056 `test_meta_registration_complete_per_package`
   rejects non-`add_*`). Task 10's "add to `PURITY_ENTRIES`" is struck in favour of the dedicated test.

---

## File Structure

**Preprocess (`silly_kicks/tracking/preprocess/`):**
- `_densify.py` — **new.** `densify_group(frames_group) -> (full_frame_ids, x, y, real_mask)` and `segments(valid_mask, max_gap_frames) -> list[(start,end)]` and `bridge_small(values, max_gap_frames) -> values`. The shared dense-grid machinery used by `_smoothing.py` and `_velocity.py`.
- `_kalman.py` — **new.** `kalman_ca(values, dt, cfg) -> KalmanOut(pos, vel, acc, var)` — constant-acceleration Kalman + RTS, native gap handling. Used both as the `"kalman"` point smoother and as the always-run uncertainty source.
- `_smoothing.py` — **modify.** Route per-segment on the dense grid; keep savgol/ema/butterworth per-segment.
- `_velocity.py` — **modify.** Emit `ax`/`ay`/`accel`; add the segment-edge-NaN rule; add `game_id` to the group key; always-run `_kalman` for `pos_var`/`vel_var`/`accel_var`; point vel/accel from the active smoother (or from `_kalman` when `smoothing_method="kalman"`).
- `_interpolation.py` — **modify.** Add `game_id` to the group key (§4.1a).
- `_config_dataclass.py` — **modify.** `SmoothingMethod` Literal += `"kalman"`; new fields `max_plausible_speed: float = 40.0`, `max_plausible_accel: float = 10.0`, Kalman noise params (`kalman_jerk_std`, `kalman_meas_noise_m`); default `smoothing_method` stays the floor for C1 (see Task 12).
- `_guard.py` — **new.** `apply_plausibility_guard(frames, cfg) -> (frames, GuardReport)` — soft NaN + count + `PlausibilityWarning`.
- `_butterworth.py` — unchanged (already ships `butterworth_lowpass` + `exact_prewarped_cutoff`).

**Schema / contracts / registries:**
- `silly_kicks/tracking/schema.py:10` — `TRACKING_FRAMES_COLUMNS` += `ax`,`ay`,`accel`,`pos_var`,`vel_var`,`accel_var` (float32).
- `silly_kicks/reflection.py` — register ReflectionKind: `ax`,`ay` vector; `accel`,`pos_var`,`vel_var`,`accel_var` magnitude.
- `silly_kicks/metric_contracts.py` — register the new frame columns.
- `silly_kicks/feature_glossary.py` — `FeatureColumn` entries for the 6 new columns.
- `NOTICE` — Savitzky-Golay (1964), Winter (2009), Kalman (1960) + RTS (1965).
- New warnings: `PlausibilityWarning` (its own category) in the preprocess warnings module.

**TF-64 (`silly_kicks/...`):** per its approved spec `2026-10-01-gk-detection-gate-design.md` — `restdefense/`, `tracking/_gk_influence.py`, `tracking/features.py` (`pre_shot_gk_*`), `gk_decision/`, `gkdv/`, `_provider_visibility.py`, glossary.

**Tests:** `tests/tracking/preprocess/` (new preprocess tests), `tests/test_add_star_purity.py` (`PURITY_ENTRIES`), the repo-wide gates (metric_contracts, reflection/mirror, call-convention, SB360 verdict, dtype-invariance, liveness).

---

## Phase C1 — correctness foundation + candidate arms + accel/uncertainty

> One coherent commit after ALL C1 tasks are green AND the owner approves. Default smoother stays the floor (SG + dense-grid) through C1; the A/B flips it in C2.

### Task 1: `game_id` key unification (§4.1a)

**Files:**
- Modify: `silly_kicks/tracking/preprocess/_velocity.py:16`
- Modify: `silly_kicks/tracking/preprocess/_interpolation.py:15`
- Test: `tests/tracking/preprocess/test_group_key_unification.py` (new)

**Interfaces:**
- Produces: no signature change; `_GROUP_KEYS = ["game_id", "period_id", "is_ball", "player_id"]` in both modules (sort order adds `game_id` first, mirroring `_smoothing.py:19-21`).

- [ ] **Step 1: Write failing tests**

```python
# tests/tracking/preprocess/test_group_key_unification.py
import numpy as np, pandas as pd
from silly_kicks.tracking.preprocess import PreprocessConfig, interpolate_frames, smooth_frames, derive_velocities

def _one_game(game_id, hz=25.0, n=60, seed=0):
    rng = np.random.default_rng(seed)
    f = pd.DataFrame({
        "game_id": game_id, "period_id": 1, "frame_id": np.arange(n),
        "time_seconds": np.arange(n) / hz, "frame_rate": hz,
        "player_id": "p1", "is_ball": False,
        "x": np.cumsum(rng.normal(0, 0.1, n)) + 50.0,
        "y": np.cumsum(rng.normal(0, 0.1, n)) + 30.0,
    })
    return f

def _set_three_key(monkeypatch):
    # force the OLD 3-key grouping (no game_id) in every preprocess module, to compare against
    import silly_kicks.tracking.preprocess._velocity as vel
    import silly_kicks.tracking.preprocess._interpolation as interp
    import silly_kicks.tracking.preprocess._smoothing as smo
    three = ["period_id", "is_ball", "player_id"]
    monkeypatch.setattr(vel, "_GROUP_KEYS", three)
    monkeypatch.setattr(interp, "_GROUP_KEYS", three)
    monkeypatch.setattr(smo, "_GROUP_KEYS", three)  # single-game: identical partition either way

def test_single_game_byte_identical_vs_three_key(monkeypatch):
    # single-game: adding game_id to the key must be a NO-OP (game_id constant -> identical partition)
    f = _one_game("g1")
    cfg = PreprocessConfig.default()
    new = derive_velocities(smooth_frames(f, config=cfg), config=cfg)          # 4-key (game_id)
    i_new = interpolate_frames(f.assign(x=f["x"].mask(f["frame_id"].eq(30))), config=cfg)
    _set_three_key(monkeypatch)
    old = derive_velocities(smooth_frames(f, config=cfg), config=cfg)          # 3-key
    i_old = interpolate_frames(f.assign(x=f["x"].mask(f["frame_id"].eq(30))), config=cfg)
    pd.testing.assert_frame_equal(new, old)                                    # byte-identical
    pd.testing.assert_frame_equal(i_new, i_old)

def test_two_game_no_cross_bridge_discriminating(monkeypatch):
    # two-game frame, 40 m jump across the seam. 4-key: no cross-game bridge. 3-key: WOULD bridge -> spike.
    # Asserting BOTH gives the test discriminating power (it can detect the fix's absence).
    g1 = _one_game("g1", n=40, seed=1)
    g2 = _one_game("g2", n=40, seed=2); g2["x"] += 40.0
    f = pd.concat([g1, g2], ignore_index=True)
    cfg = PreprocessConfig.default()
    v4 = derive_velocities(smooth_frames(f, config=cfg), config=cfg)
    seam4 = v4[(v4["game_id"] == "g2") & (v4["frame_id"] == 0)]["speed"]
    assert (seam4.dropna() < 40).all()                                         # 4-key: no cross-game spike
    _set_three_key(monkeypatch)
    v3 = derive_velocities(smooth_frames(f, config=cfg), config=cfg)
    seam3 = v3[(v3["game_id"] == "g2") & (v3["frame_id"] == 0)]["speed"]
    assert (seam3 > 40).any()                                                  # discriminating: 3-key bridges
```

- [ ] **Step 2: Run — expect FAIL** (two-game bridge produces a spike / NaN mismatch).
Run: `python -m pytest tests/tracking/preprocess/test_group_key_unification.py -v`

- [ ] **Step 3: Implement** — in both `_velocity.py` and `_interpolation.py` set
`_GROUP_KEYS = ["game_id", "period_id", "is_ball", "player_id"]` and add `game_id` as the first `sort_cols` entry (mirror `_smoothing.py`); copy the A-31 rationale comment.

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: Full CI-faithful suite green** (no commit — phase boundary only).

### Task 2: Dense-grid reindex core (§4.1)

**Files:**
- Create: `silly_kicks/tracking/preprocess/_densify.py`
- Test: `tests/tracking/preprocess/test_densify.py` (new)

**Interfaces:**
- Produces: `densify_group(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]` returning `(full_frame_ids, x_dense, y_dense, real_mask)`; `segments(valid: np.ndarray, max_gap_frames: int) -> list[tuple[int,int]]`; `bridge_small(values: np.ndarray, max_gap_frames: int) -> np.ndarray`. Consumed by Tasks 3/5/7 and `_smoothing.py`/`_velocity.py`.

- [ ] **Step 1: Write failing tests**

```python
# tests/tracking/preprocess/test_densify.py
import numpy as np, pandas as pd
from silly_kicks.tracking.preprocess._densify import densify_group, segments, bridge_small

def test_densify_inserts_missing_rows_as_nan():
    g = pd.DataFrame({"frame_id": [0, 1, 2, 10, 11], "x": [1.0, 1.1, 1.2, 5.0, 5.1], "y": [0.0]*5})
    full, x, y, real = densify_group(g)
    assert list(full) == list(range(0, 12))
    assert np.isnan(x[3:10]).all()          # frames 3..9 are missing -> NaN
    assert real.sum() == 5 and not real[3]

def test_densify_dedups_duplicate_frames():  # GS duplicate-frame guard
    g = pd.DataFrame({"frame_id": [0, 0, 1], "x": [1.0, 1.0, 1.1], "y": [0.0]*3})
    full, x, y, real = densify_group(g)
    assert list(full) == [0, 1]

def test_segments_split_at_big_gap():
    valid = np.array([1,1,1,0,0,0,0,1,1], dtype=bool)  # a 4-frame gap
    assert segments(valid, max_gap_frames=2) == [(0, 3), (7, 9)]   # split (gap 4 > 2)
    assert segments(valid, max_gap_frames=5) == [(0, 9)]           # not split (gap 4 <= 5)

def test_bridge_small_fills_short_leaves_long():
    v = np.array([0.0, np.nan, 2.0, np.nan, np.nan, np.nan, 6.0])
    out = bridge_small(v, max_gap_frames=1)
    assert out[1] == 1.0                      # 1-frame gap bridged
    assert np.isnan(out[3:6]).all()           # 3-frame gap left NaN
```

- [ ] **Step 2: Run — expect FAIL** (module absent).

- [ ] **Step 3: Implement `_densify.py`** — `densify_group` **de-duplicates `frame_id` keep-first** (owner-approved 2026-10-09; matches `_elastic_sync.py:118` / `features.py:6039`; DAS's `raise`-on-duplicate guards a different invariant), reindexes to the contiguous range, `real_mask = isin(full, original)`; `segments` splits at NaN runs > `max_gap_frames`, each segment bounded by valid rows; `bridge_small` linear-interpolates interior NaN runs ≤ `max_gap_frames`, leaves longer runs NaN. (Reference the validated scratch logic from the spike harness.)

- [ ] **Step 4: Run — expect PASS; full suite green.** (Task 2 ends at the pure helper; the wire-up is Task 2b.)

### Task 2b: Wire the dense grid into `smooth_frames` + `derive_velocities`

**Files:**
- Modify: `silly_kicks/tracking/preprocess/_smoothing.py`, `silly_kicks/tracking/preprocess/_velocity.py`
- Modify (docstrings, §4.1b): both modules — document the two-gap-fill-stage interaction (reindex = missing rows; `interpolate_frames` = NaN-position runs; shared `max_gap_seconds`; order `interpolate_frames` → smooth/derive; no double-fill)
- Test: append to `tests/tracking/preprocess/test_densify.py`

**Interfaces:**
- Consumes: `densify_group`/`segments`/`bridge_small` (Task 2). Produces: `smooth_frames`/`derive_velocities` run per `segments(...)` on the dense grid; output row-count unchanged.

- [ ] **Step 1: Write failing test**

```python
# append to test_densify.py
def test_missing_row_gap_yields_nan_velocity_not_spike():
    hz = 25.0
    pre = pd.DataFrame({"frame_id": np.arange(0, 20)})      # contiguous run A
    post = pd.DataFrame({"frame_id": np.arange(200, 220)})  # run B after a 180-frame (7.2 s) gap
    fid = np.concatenate([pre["frame_id"], post["frame_id"]])
    f = pd.DataFrame({"game_id": "g", "period_id": 1, "frame_id": fid,
                      "time_seconds": fid / hz, "frame_rate": hz, "player_id": "p1", "is_ball": True,
                      "x": np.r_[np.full(20, 10.0), np.full(20, 100.0)],  # 90 m jump across the gap
                      "y": 34.0})
    from silly_kicks.tracking.preprocess import PreprocessConfig, smooth_frames, derive_velocities
    cfg = PreprocessConfig.default()  # max_gap_seconds=0.5 -> 12 frames; 180 >> 12
    v = derive_velocities(smooth_frames(f, config=cfg), config=cfg)
    boundary = v[(v["frame_id"] == 19) | (v["frame_id"] == 200)]
    assert boundary["speed"].isna().all()   # no fabricated through-gap velocity

def test_idempotent_and_single_run_byte_identical():
    # a contiguous single-detection-run group: dense-grid reindex is a no-op (sportec control analogue)
    hz = 25.0; n = 60
    f = pd.DataFrame({"game_id": "g", "period_id": 1, "frame_id": np.arange(n), "time_seconds": np.arange(n)/hz,
                      "frame_rate": hz, "player_id": "p1", "is_ball": False, "x": 50.0 + 0.1*np.arange(n), "y": 30.0})
    from silly_kicks.tracking.preprocess import PreprocessConfig, smooth_frames, derive_velocities
    cfg = PreprocessConfig.default()
    once = derive_velocities(smooth_frames(f, config=cfg), config=cfg)
    twice = derive_velocities(smooth_frames(once, config=cfg), config=cfg)  # idempotent
    pd.testing.assert_series_equal(once["speed"], twice["speed"])
```

- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement** the per-group densify + per-segment smoother routing in `smooth_frames`/`derive_velocities`; return only original rows. Add the §4.1b docstring note to both modules.
- [ ] **Step 4: Run — expect PASS; full suite green.**

### Task 3: Segment-edge NaN (§4.2)

**Files:**
- Modify: `silly_kicks/tracking/preprocess/_velocity.py`
- Test: `tests/tracking/preprocess/test_segment_edge.py` (new)

**Interfaces:**
- Consumes: `segments` (Task 2). Produces: no signature change.

- [ ] **Step 1: Write failing test**

```python
# tests/tracking/preprocess/test_segment_edge.py
import numpy as np, pandas as pd
from silly_kicks.tracking.preprocess import PreprocessConfig, smooth_frames, derive_velocities

def test_no_edge_spike_on_short_pre_gap_segment():
    # a short (near-window-length) segment immediately before a big gap must not emit an edge-spike velocity
    hz = 29.97
    seg = np.arange(0, 6)                       # 6-frame segment (< a 11-frame window)
    post = np.arange(1000, 1020)
    fid = np.concatenate([seg, post])
    f = pd.DataFrame({"game_id": "g", "period_id": 1, "frame_id": fid,
                      "time_seconds": fid / hz, "frame_rate": hz, "player_id": "p1", "is_ball": True,
                      "x": np.r_[np.full(6, 113.0), np.full(20, 11.0)], "y": 34.0})
    cfg = PreprocessConfig.default()
    v = derive_velocities(smooth_frames(f, config=cfg), config=cfg)
    pre_gap = v[v["frame_id"] <= 5]
    assert (pre_gap["speed"].isna() | (pre_gap["speed"] < 40)).all()  # edge NaN'd, not a 300+ m/s spike
```

- [ ] **Step 2: Run — expect FAIL** (SG edge polynomial spikes → speed ≫ 40).

- [ ] **Step 3: Implement** — after per-segment derivation, NaN the derived `vx`/`vy`/`ax`/`ay` within the filter radius (`window_frames // 2`) of each segment boundary. Apply uniformly across smoother methods.

- [ ] **Step 4: Run — expect PASS; full suite green.**

### Task 4: Soft plausibility guard (§4.3)

**Files:**
- Create: `silly_kicks/tracking/preprocess/_guard.py`
- Modify: `silly_kicks/tracking/preprocess/_config_dataclass.py` (add `max_plausible_speed: float = 40.0`, `max_plausible_accel: float = 10.0`); new `PlausibilityWarning`
- Modify: `silly_kicks/tracking/preprocess/_velocity.py` (call the guard after derivation)
- Test: `tests/tracking/preprocess/test_plausibility_guard.py` (new)

**Interfaces:**
- Produces: `apply_plausibility_guard(frames: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame` — NaNs `vx`/`vy`/`speed` where `speed > cfg.max_plausible_speed` (and `ax`/`ay`/`accel` where `accel > cfg.max_plausible_accel`), counts them, emits `PlausibilityWarning`. NOT a raise.

- [ ] **Step 1: Write failing test**

```python
# tests/tracking/preprocess/test_plausibility_guard.py
import numpy as np, pandas as pd, pytest
from silly_kicks.tracking.preprocess import PreprocessConfig
from silly_kicks.tracking.preprocess._guard import apply_plausibility_guard, PlausibilityWarning

def _frames_with_speed(speeds):
    n = len(speeds)
    return pd.DataFrame({"vx": speeds, "vy": np.zeros(n), "speed": np.abs(speeds),
                         "ax": np.zeros(n), "ay": np.zeros(n), "accel": np.zeros(n)})

def test_guard_nans_implausible_soft_with_warning_not_raise():
    f = _frames_with_speed(np.array([5.0, 500.0, 10.0]))
    cfg = PreprocessConfig.default()
    with pytest.warns(PlausibilityWarning):
        out = apply_plausibility_guard(f, cfg)
    assert np.isnan(out.loc[1, "speed"]) and np.isnan(out.loc[1, "vx"])
    assert out.loc[0, "speed"] == 5.0 and out.loc[2, "speed"] == 10.0   # plausible untouched

def test_guard_does_not_raise_on_out_of_pitch_ball():
    f = _frames_with_speed(np.full(100, 60.0))  # ~6% out-of-pitch ball analogue, all > 40
    cfg = PreprocessConfig.default()
    out = apply_plausibility_guard(f, cfg)       # must return, not raise
    assert out["speed"].isna().all()
```

- [ ] **Step 2: Run — expect FAIL** (module absent).

- [ ] **Step 3: Implement `_guard.py`** + the `PlausibilityWarning` category (its own class, not subclassing another) + the config fields; call `apply_plausibility_guard` at the end of `derive_velocities`.

- [ ] **Step 4: Run — expect PASS; full suite green.**

### Task 4b: Real-fixture plausibility (in-CI) + named real repros (`@e2e`) — spec §4.5

> Spec §4.5 bullets (2) + (3). Owner-approved 2026-10-09: the public committed slices carry an always-run `≤40 m/s` guard; the owner-tier broadcast repros ride `@e2e` via pining (no synthetic substitution).

**Files:**
- Test (in-CI, always runs): extend `tests/invariants/test_invariant_velocity_physical_plausibility.py` over the committed public slices (`tests/datasets/tracking/action_context_slim/{sportec,skillcorner}_slim.parquet`, `tests/datasets/elastic_sync/j03wmx_slice/frames.parquet`).
- Test (`@pytest.mark.e2e`, skip-if-token-unset): `tests/tracking/preprocess/test_velocity_gap_repro_e2e.py` (new) — sources GS `10502` p1 + WC2022 `3851` via pining (`PINING_FOR_THE_DATA_TOKEN`), the named handoff repros.

- [ ] **Step 1: Write the in-CI plausibility test**

```python
# extend tests/invariants/test_invariant_velocity_physical_plausibility.py
import pandas as pd, pytest
from pathlib import Path
from silly_kicks.tracking.preprocess import PreprocessConfig, smooth_frames, derive_velocities

_DATA = Path(__file__).parent / "../datasets"
_SLICES = [_DATA / "tracking/action_context_slim/sportec_slim.parquet",
           _DATA / "tracking/action_context_slim/skillcorner_slim.parquet",
           _DATA / "elastic_sync/j03wmx_slice/frames.parquet"]

@pytest.mark.parametrize("path", _SLICES, ids=lambda p: p.stem)
def test_built_speed_within_physical_bound(path):
    f = pd.read_parquet(path)
    cfg = PreprocessConfig.default()
    v = derive_velocities(smooth_frames(f, config=cfg), config=cfg)
    assert (v["speed"].dropna() <= cfg.max_plausible_speed).all()   # post-guard: nothing implausible survives
```

- [ ] **Step 2: Run — expect FAIL** until the guard (Task 4) + dense-grid (Task 2b) land; then PASS.

- [ ] **Step 3: Write the `@e2e` named-repro test**

```python
# tests/tracking/preprocess/test_velocity_gap_repro_e2e.py
import os, pytest, numpy as np
pytestmark = pytest.mark.e2e

_SKIP = pytest.mark.skipif(not os.environ.get("PINING_FOR_THE_DATA_TOKEN"), reason="pining token unset")

@_SKIP
def test_gs_10502_ball_gap_no_spike():
    # THE named ball-gap repro: GS 10502 p1 ball, ~930-frame non-detection at 57169-58109.
    frames = _load_gs_frames_via_pining("10502")   # urllib two-step, per reference_pining_for_the_data_api
    from silly_kicks.tracking.preprocess import PreprocessConfig, smooth_frames, derive_velocities
    cfg = PreprocessConfig.for_provider("gradientsports")
    v = derive_velocities(smooth_frames(frames, config=cfg), config=cfg)
    ball = v[(v["is_ball"]) & (v["period_id"] == 1)]
    assert (ball["speed"].dropna() <= cfg.max_plausible_speed).all()   # the 500-850 m/s fabrication is gone

@_SKIP
def test_gs_3851_single_frame_player_nan():
    # NOT a ball gap — the WC2022 3851 case named in _velocity.py:71-73 is a single-frame PLAYER
    # (away #10, exactly 1 frame in p2): velocity is undefined -> NaN, not a value.
    frames = _load_gs_frames_via_pining("3851")
    from silly_kicks.tracking.preprocess import PreprocessConfig, smooth_frames, derive_velocities
    cfg = PreprocessConfig.for_provider("gradientsports")
    v = derive_velocities(smooth_frames(frames, config=cfg), config=cfg)
    single = v[(~v["is_ball"]) & (v["period_id"] == 2)].groupby("player_id").filter(lambda g: len(g) == 1)
    assert single["speed"].isna().all()   # single-frame player -> NaN velocity (no fabrication)
```

- [ ] **Step 4:** Implement `_load_gs_frames_via_pining` (inline urllib two-step; no committed GS raw, no local path). Run `@e2e` locally with the token (owner-run); CI skips without it.
- [ ] **Step 5: Full suite green** (in-CI plausibility passes; `@e2e` skips in CI, runs owner-side).

### Task 5: Acceleration columns (`ax`/`ay`/`accel`)

**Files:**
- Modify: `silly_kicks/tracking/preprocess/_velocity.py`
- Modify: `silly_kicks/tracking/schema.py:10` (TRACKING_FRAMES_COLUMNS += `ax`,`ay`,`accel` as float32)
- Modify: `silly_kicks/reflection.py` (register `ax`,`ay` vector; `accel` magnitude)
- Test: `tests/tracking/preprocess/test_acceleration.py` (new)

**Interfaces:**
- Produces: `derive_velocities` now adds `ax`,`ay`,`accel` (float32, m/s²) alongside `vx`,`vy`,`speed`. SG arm: `savgol_filter(..., deriv=2)`; butterworth arm: second finite difference of the smoothed position.

- [ ] **Step 1: Write failing test**

```python
# tests/tracking/preprocess/test_acceleration.py
import numpy as np, pandas as pd
from silly_kicks.tracking.preprocess import PreprocessConfig, smooth_frames, derive_velocities

def _const_accel_frames(a=2.0, hz=25.0, n=50):
    t = np.arange(n) / hz
    return pd.DataFrame({"game_id": "g", "period_id": 1, "frame_id": np.arange(n),
                         "time_seconds": t, "frame_rate": hz, "player_id": "p1", "is_ball": False,
                         "x": 0.5 * a * t**2, "y": 30.0})

def test_accel_columns_present_and_correct_sign_and_dtype():
    f = _const_accel_frames(a=2.0)
    cfg = PreprocessConfig.default()
    v = derive_velocities(smooth_frames(f, config=cfg), config=cfg)
    assert {"ax", "ay", "accel"} <= set(v.columns)
    assert v["ax"].dtype == np.float32 and v["accel"].dtype == np.float32
    mid = v.iloc[20:30]
    assert np.allclose(mid["ax"], 2.0, atol=0.2)     # recovers the constant acceleration
    assert (mid["accel"] >= 0).all()                 # magnitude non-negative
```

- [ ] **Step 2: Run — expect FAIL** (columns absent).

- [ ] **Step 3: Implement** acceleration emission in `derive_velocities` per smoother; add the three columns to `TRACKING_FRAMES_COLUMNS`; register reflection kinds (`ax`/`ay` vector, `accel` magnitude).

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: Mirror-invariance test** — add to the reflection/mirror gate: reflecting the frame flips `ax` sign, leaves `accel` invariant. Run the repo mirror gate green. Full suite green.

### Task 6: Kalman/RTS module (`_kalman.py`)

**Files:**
- Create: `silly_kicks/tracking/preprocess/_kalman.py`
- Modify: `silly_kicks/tracking/preprocess/_config_dataclass.py` (add `kalman_jerk_std: float`, `kalman_meas_noise_m: float`)
- Test: `tests/tracking/preprocess/test_kalman.py` (new)

**Interfaces:**
- Produces: `kalman_ca(values: np.ndarray, dt: float, cfg: PreprocessConfig) -> KalmanOut` where `KalmanOut` is a dataclass/namedtuple `(pos, vel, acc, var)` (each a float64 array; `var` = per-frame position-variance). Constant-acceleration state `[p, v, a]`, forward Kalman + RTS backward; missing frames (NaN) → predict-only (no update), so `var` grows through gaps.

- [ ] **Step 1: Write failing tests**

```python
# tests/tracking/preprocess/test_kalman.py
import numpy as np
from silly_kicks.tracking.preprocess import PreprocessConfig
from silly_kicks.tracking.preprocess._kalman import kalman_ca

def test_kalman_recovers_constant_velocity():
    hz = 25.0; dt = 1/hz; n = 100
    z = 2.0 * np.arange(n) * dt           # 2 m/s
    out = kalman_ca(z, dt, PreprocessConfig.default())
    assert np.allclose(out.vel[20:80], 2.0, atol=0.1)

def test_kalman_variance_grows_through_gap():
    hz = 25.0; dt = 1/hz; n = 120
    z = np.linspace(0, 10, n)
    z[50:80] = np.nan                     # a 30-frame occlusion
    out = kalman_ca(z, dt, PreprocessConfig.default())
    assert out.var[65] > out.var[20]      # mid-gap uncertainty exceeds a dense-interior frame
    assert np.isfinite(out.vel[65])       # still predicts through (not NaN) — honest-uncertainty, not fabrication
```

- [ ] **Step 2: Run — expect FAIL** (module absent).

- [ ] **Step 3: Implement `_kalman.py`** (CA Kalman + RTS; predict-only on NaN; `var` = smoothed position-variance). Reference the validated scratch implementation.

- [ ] **Step 4: Run — expect PASS; full suite green.**

### Task 7: Always-on uncertainty columns (`pos_var`/`vel_var`/`accel_var`)

**Files:**
- Modify: `silly_kicks/tracking/preprocess/_velocity.py` (always run `_kalman` per group on the densified raw positions; emit the three variance columns regardless of `smoothing_method`)
- Modify: `silly_kicks/tracking/schema.py:10` (+ `pos_var`,`vel_var`,`accel_var` float32)
- Modify: `silly_kicks/reflection.py` (magnitude-invariant)
- Test: `tests/tracking/preprocess/test_uncertainty.py` (new)

**Interfaces:**
- Produces: `derive_velocities` adds `pos_var`,`vel_var`,`accel_var` (float32) for every `smoothing_method`. Point `vx`/`vy`/`ax`/`ay` come from the active smoother, EXCEPT `smoothing_method="kalman"` (Task 9) where they come from the same `_kalman` pass (coherent).

- [ ] **Step 1: Write failing test (the honest-gap property — stronger than liveness)**

```python
# tests/tracking/preprocess/test_uncertainty.py
import numpy as np, pandas as pd
from silly_kicks.tracking.preprocess import PreprocessConfig, smooth_frames, derive_velocities

def _frames_with_gap(hz=25.0):
    pre = np.arange(0, 40); post = np.arange(60, 100)      # a 20-frame (0.8 s > max_gap) gap
    fid = np.concatenate([pre, post])
    return pd.DataFrame({"game_id": "g", "period_id": 1, "frame_id": fid, "time_seconds": fid/hz,
                         "frame_rate": hz, "player_id": "p1", "is_ball": False,
                         "x": 50.0 + 0.05*fid, "y": 30.0})

def test_variance_strictly_rises_toward_a_gap():
    cfg = PreprocessConfig.default()
    v = derive_velocities(smooth_frames(_frames_with_gap(), config=cfg), config=cfg)
    assert {"pos_var", "vel_var", "accel_var"} <= set(v.columns)
    for col in ("pos_var", "accel_var"):          # spec §11 names BOTH
        gap_adjacent = v[v["frame_id"] == 39][col].iloc[0]
        dense_interior = v[v["frame_id"] == 20][col].iloc[0]
        assert gap_adjacent > dense_interior       # honest-gap property, not merely non-constant

def test_uncertainty_present_for_every_smoother():
    cfg = PreprocessConfig.default()
    for method in ("savgol", "butterworth"):
        v = derive_velocities(smooth_frames(_frames_with_gap(), config=cfg, method=method), config=cfg)
        assert v["accel_var"].notna().any()        # decoupled from the point smoother
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement** the always-on `_kalman` pass + the three columns + reflection kinds (magnitude).

- [ ] **Step 4: Run — expect PASS; full suite green.**

### Task 8: Butterworth velocity/acceleration arm

**Files:**
- Modify: `silly_kicks/tracking/preprocess/_velocity.py`
- Test: `tests/tracking/preprocess/test_butterworth_arm.py` (new)

**Interfaces:**
- Consumes: `butterworth_lowpass` (`_butterworth.py`). Produces: with `smoothing_method="butterworth"`, `vx`/`vy`/`ax`/`ay` are finite differences of the BW-smoothed position per segment (edge-NaN per Task 3).

- [ ] **Step 1: Write failing test**

```python
# tests/tracking/preprocess/test_butterworth_arm.py
import numpy as np, pandas as pd
from silly_kicks.tracking.preprocess import PreprocessConfig, smooth_frames, derive_velocities

def test_butterworth_arm_velocity_bounded_across_gap():
    hz = 29.97; pre = np.arange(0, 20); post = np.arange(1000, 1020)
    fid = np.concatenate([pre, post])
    f = pd.DataFrame({"game_id": "g", "period_id": 1, "frame_id": fid, "time_seconds": fid/hz,
                      "frame_rate": hz, "player_id": "p1", "is_ball": True,
                      "x": np.r_[np.full(20, 113.0), np.full(20, 11.0)], "y": 34.0})
    cfg = PreprocessConfig.default()
    v = derive_velocities(smooth_frames(f, config=cfg, method="butterworth"), config=cfg)
    assert (v["speed"].dropna() < 40).all()   # no through-gap fabrication; tail clean (spike: BW best)
```

- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement** the BW derivative path in `derive_velocities`.
- [ ] **Step 4: Run — expect PASS; full suite green.**

### Task 9: Kalman as a point smoother (`smoothing_method="kalman"`)

**Files:**
- Modify: `silly_kicks/tracking/preprocess/_config_dataclass.py` (`SmoothingMethod` Literal += `"kalman"`)
- Modify: `silly_kicks/tracking/preprocess/_smoothing.py` (route `"kalman"` to `_kalman` for `x_smoothed`/`y_smoothed`)
- Modify: `silly_kicks/tracking/preprocess/_velocity.py` (when `"kalman"`, point vel/accel = the same `_kalman` state)
- Test: `tests/tracking/preprocess/test_kalman_arm.py` (new)

**Interfaces:**
- Consumes: `kalman_ca` (Task 6). Produces: `smoothing_method="kalman"` yields point pos/vel/accel from the Kalman state, and the Task-7 uncertainty is the SAME pass (coherent).

- [ ] **Step 1: Write failing test**

```python
# tests/tracking/preprocess/test_kalman_arm.py
import numpy as np, pandas as pd
from silly_kicks.tracking.preprocess import PreprocessConfig, smooth_frames, derive_velocities

def test_kalman_arm_produces_coherent_kinematics():
    hz = 25.0; n = 80
    f = pd.DataFrame({"game_id": "g", "period_id": 1, "frame_id": np.arange(n), "time_seconds": np.arange(n)/hz,
                      "frame_rate": hz, "player_id": "p1", "is_ball": False,
                      "x": 2.0*np.arange(n)/hz, "y": 30.0})
    cfg = PreprocessConfig.default()
    v = derive_velocities(smooth_frames(f, config=cfg, method="kalman"), config=cfg)
    assert np.allclose(v["vx"].iloc[20:60], 2.0, atol=0.1)
    assert {"pos_var", "accel"} <= set(v.columns)      # coherent: point + uncertainty from one estimator
```

- [ ] **Step 2–4:** FAIL → implement routing → PASS; full suite green.

### Task 10: Contracts, registries, glossary, NOTICE, purity

**Files:**
- Modify: `silly_kicks/metric_contracts.py`, `silly_kicks/feature_glossary.py`, `NOTICE`, `tests/test_add_star_purity.py` (`PURITY_ENTRIES`), plus the SB360-verdict / call-convention / dtype-invariance registries the new columns trip.
- Test: the existing repo-wide gate tests (run them; register until green).

- [ ] **Step 1:** Run the repo-wide gates and observe the expected failures:
`python -m pytest tests/test_metric_contracts.py tests/test_add_star_purity.py -v` (+ the reflection/mirror, call-convention, dtype-invariance, SB360-verdict, liveness gate tests). Expected: FAIL (new columns unregistered).
- [ ] **Step 2:** Register the 6 new frame columns in `metric_contracts.py`; add `FeatureColumn` glossary entries (with `describe_level`); add NOTICE entries for Savitzky-Golay (1964), Winter (2009), Kalman (1960)+RTS (1965); confirm `smooth_frames`/`derive_velocities` remain PURE (add/adjust `PURITY_ENTRIES`); add an SB360 freeze-frame verdict if any `add_*` surface changed; a discriminating-power precondition for every new liveness fixture.
- [ ] **Step 3: `*_xfns` leak-guard confirm.** `ax`/`ay`/`accel`/`*_var` are kinematics (no post-contact `result_id` read), so they are safe in default `*_xfns`. Run the leak-guard test and confirm no new column reads its own outcome; add the new columns to the guard's known-safe surface if the registry requires explicit enrollment. Expected: green.
- [ ] **Step 4:** Run all gates — expect PASS. Full suite green.

### Task 10b: C1 golden/snapshot blast-radius audit

**Files:**
- Audit (read): every committed parity oracle / snapshot in the default (`not e2e`) suite whose inputs flow through `smooth_frames`/`derive_velocities` — at least `tests/tracking/_das_golden.py` (+ `_fixtures/das_golden/`), the player-influence / pressure / empirical-AC snapshots, `tests/tracking/_fixtures/gk_geometry_golden_frames.parquet`.
- Test: a short audit note in `tests/tracking/preprocess/README.md` (new) recording the verdict per oracle.

**Interfaces:** none (verification task). This closes the gap that "full suite green" at the C1 boundary could silently rewrite a committed parity baseline.

- [ ] **Step 1:** For each committed golden/snapshot, determine whether the C1 changes move it:
  - **DAS unit golden** (`scenes_frames.csv`) — carries `vx`/`vy` as fixture inputs; `_das_pack.py:84` reads them directly; the test never calls the preprocess path ⇒ **byte-identical, no regen** (verified). Record this.
  - For every other oracle: run the default suite; if an oracle reddens, it is a value-change that needs **owner sign-off** before regenerating the baseline (a parity oracle is not silently rewritten). A contiguous/single-game/no-gap fixture should be byte-identical (reindex no-op, `game_id` no-op, guard no-op, edge-NaN no-op); the new additive columns must not break an exact-column-set assertion — adjust the snapshot's column handling if so.
- [ ] **Step 2:** If any oracle legitimately moves, STOP and present the diff to the owner for sign-off; otherwise record "byte-identical, why" per oracle in the audit note.
- [ ] **Step 3:** Full suite green (every committed oracle either byte-identical or owner-signed-off regenerated).

### Task 11: Consumer NaN-tolerance (ghost-GK + elastic-sync)

**Files:**
- Public `add_*` live in `silly_kicks/tracking/features.py` (`add_ghost_gk` at `features.py:5120`, `add_elastic_sync` at `features.py:6894`); the NaN-safe logic, if a fix is needed, is in `silly_kicks/tracking/_ghost_gk.py` / `silly_kicks/tracking/_elastic_sync.py`.
- Fixture: the committed `tests/datasets/elastic_sync/j03wmx_slice/{frames,actions}.parquet` (real slice) — NaN the ball `vx`/`vy`/`speed` on the linked frames to simulate the guard output. Ghost-GK model: load the bundled default ghost-GK artifact via the same loader `add_ghost_gk` uses in `tests/tracking/` (reuse the existing ghost-GK test fixture's model load — do not pass `model=...` as a placeholder).
- Test: `tests/tracking/test_consumer_nan_tolerance.py` (new).

**Interfaces:**
- Consumes: the guard-NaN'd `ball_vx`/`ball_vy`/`ball_speed`. Produces: `add_ghost_gk` + `add_elastic_sync` emit NaN-not-crash on those rows (per ADR-003 NaN-safe enrichment).

- [ ] **Step 1: Write failing test** — load the `j03wmx_slice` frames+actions; set the ball `vx`/`vy`/`speed` to NaN on the action-linked frames; assert `add_ghost_gk` / `add_elastic_sync` return with NaN (or dropped) outputs on the affected rows rather than raise.

```python
# tests/tracking/test_consumer_nan_tolerance.py
import pandas as pd, numpy as np
from pathlib import Path
from silly_kicks.tracking.features import add_ghost_gk, add_elastic_sync
from silly_kicks.tracking._ghost_gk import GhostGkModel  # import as tests/tracking/test_ghost_gk.py does

_SLICE = Path(__file__).parent / "../datasets/elastic_sync/j03wmx_slice"

def _nan_ball_velocity(frames):
    f = frames.copy()
    ball = f["is_ball"] == True
    f.loc[ball, ["vx", "vy", "speed"]] = np.nan
    return f

def test_elastic_sync_nan_tolerant():
    frames = _nan_ball_velocity(pd.read_parquet(_SLICE / "frames.parquet"))
    actions = pd.read_parquet(_SLICE / "actions.parquet")
    out = add_elastic_sync(actions, frames=frames)          # must not raise
    assert out is not None

def test_ghost_gk_nan_tolerant():
    frames = _nan_ball_velocity(pd.read_parquet(_SLICE / "frames.parquet"))
    actions = pd.read_parquet(_SLICE / "actions.parquet")
    model = GhostGkModel.from_variant("default")                        # real pattern, not a conftest fixture
    home_team_id = ...  # the j03wmx home team id — read from the slice (tests/datasets/.../j03wmx_slice/README.md)
    out = add_ghost_gk(actions, frames=frames, model=model, home_team_id=home_team_id)  # home_team_id is REQUIRED kw-only
    assert out is not None
```

- [ ] **Step 2–4:** FAIL (if any) → make the consumer NaN-safe (ADR-003) → PASS; full suite green.

### Task 12: Default-smoother + docstring

**Files:**
- Modify: `silly_kicks/tracking/preprocess/_config_dataclass.py` (confirm the C1 default `smoothing_method` = the SG + dense-grid floor)
- Modify: `silly_kicks/tracking/preprocess/_smoothing.py:106` (docstring: add `butterworth` to the `method` options)
- Test: `tests/tracking/preprocess/test_default_config.py` (new)

- [ ] **Step 1:** Test that `PreprocessConfig.default().smoothing_method == "savgol"` (the floor through C1) and that the `method` docstring lists `savgol`/`ema`/`butterworth`/`kalman`.
- [ ] **Step 2–4:** FAIL → fix docstring/default → PASS; full suite green.

> **C1 BOUNDARY:** full CI-faithful suite green + `/final-review` → **present the C1 diff/file-list to the owner; await explicit approval; then commit C1.**

---

## A/B milestone (scratch — between C1 and C2; NOT a commit)

- [ ] Build the A/B harness as a **scratch** run (not a committed driver), reusing the C1 smoothers. Per spec §6 (RE-amended 2026-10-09 — HYBRID): train ghost-GK (primary) + xshot/xcross (guardrail) under each of `savgol`(floor)/`butterworth`/`kalman`, match-level CV. **PRIMARY (decision, full 179-match corpus):** ghost-GK out-of-sample **Euclidean MAE** (`predict_mean`), one-sided 95% paired match-level bootstrap CI of `[floor − finalist]` per-match MAE **excludes 0** (B=2000, fixed seed). **CONFIRMATORY (fixed 10-match subset = 5 SK + 4 GS + 1 idsse, lexicographically-first per provider):** ghost-GK **density-NLL** (`predict_density`, `kde_backend="vectorized"` — exact numpy raw grid, no `"exact"` literal; the MAE winner must not regress it vs floor). Why hybrid: density-NLL is O(n_train)/eval-row (measured ~23 s/row at full-179 train) → infeasible full-corpus; MAE carries the decision, density guards calibration on a bounded subset. Guardrails: implausible-rate / coverage (≤0.5 pp drop) / accel plausibility (full corpus, free); xshot/xcross log-loss on a fixed 30-match subset (17 SK + 10 GS + 3 idsse, lexicographically-first, default params; full-179 would add ~18.6h); **DAS→danger AUC DEFERRED (owner-approved, multi-day; frame guardrails gate the velocity consumer)**. A/B materialize runs full C1 smooth+derive (no uncertainty-skip flag; ~99 s/match, irreducible). Decision → ADR-117 (pre-registration already recorded there).
- [ ] Apply the pre-registered §6.5 decision rule; **record the result + decision in the cycle ADR** (`docs/superpowers/adrs/ADR-NNN-*.md`, written here). Fallback = ship the floor.
- [ ] Owner go required to launch the A/B (DGX compute).

---

## Phase C2 — winner default + retrain

> One commit after the A/B decision + retrain, owner-approved.

### Task 13: Flip default smoother to the A/B winner

**Files:** `silly_kicks/tracking/preprocess/_config_dataclass.py` (+ per-provider promotion in `_provider_defaults_generated.py` / `_config.py` if the winner is per-provider).
- [ ] Test: `PreprocessConfig.default().smoothing_method == "<winner>"` and `for_provider(...)` promotes correctly. TDD red→green; full suite green.
- [ ] If the winner is the floor (no swap), this task sets nothing new beyond C1 and is a no-op recorded in the ADR.

### Task 14: Retrain artifacts + integrity

**Scope pinned 2026-10-10 (owner-approved).** Winner = **floor (SG + dense-grid)**; Task 13 = no-op. Retrain the velocity/position-dependent frame models on the corrected floor velocities. **Core 5 families:** `ghost_gk`, `ghost_outfield`, `xshot_occurrence`, `xcross_attempt`, `receiver`; **only the variants whose inputs moved** (dense-grid changed `vx/vy` AND `x_smoothed`): faithful/velocity variants (default, sweeper, provider-specific) for certain; `position_only` iff it reads `x_smoothed` (verify at impl — never re-publish a byte-identical variant). `gk_completion` (xT-GK v1) **EXCLUDED** (retired, Phase C4). xT-GK v2 unchanged (not velocity-dependent). Committed unit DAS golden byte-identical (Task 10b) — NOT regenerated.

**Files:** the 5-family bundled-weight dirs + `SHA256SUMS`; corpus DAS re-fit; T10 baseline; `scripts/materialize_tc3_frames.py` (`_SHARD_SCHEMA_VERSION` bump).
- [ ] **Driver prereq (a):** bump `materialize_tc3_frames.py` `_SHARD_SCHEMA_VERSION` `"tc3-frames-2"→"tc3-frames-3"` (6 new cols + moved vx/vy = shape+content change → new generation subdir, avoids stale-shard resume-skip). Test: `test_materialize_tc3_frames.py` green (checks content parity, not the version string → no pin to update). **DONE.**
- [ ] **Driver prereq (a2) — corpus roster:** add `--match-ids-json` to `materialize_tc3_frames.py` (mirrors `_loader_pining_to_cache.py`; loads `{provider:[ids]}` → `pining_source(match_ids=)`; records `roster_sha` in `token_inputs` so a constrained corpus gets its own generation). The retrain MUST use the **F1b 179 corpus** (GS 64 all · SK 108 GI · idsse 7 all), NOT the full ~980 pining catalog (the full catalog pulls non-GI + NDA SK → corpus-confound + licensing). Roster = the old tc3-cache's realized 179 shard ids, kept **DGX-local** (`~/tf65_roster_179.json`, owner data — not committed; keeps GI membership out of the public repo). Mechanism smoke-verified: `pining_source(match_ids=roster)` → exactly 179. **Flag added; TDD test owed before the C2-boundary commit** (focused: `--match-ids-json` → `match_ids` reaches `pining_source` + `roster_sha` in `token_inputs`; mock the source).
- [ ] **Driver prereq (b):** OMIT `--reference-parquet` for the re-materialize (old reference is pre-dense-grid; a check vs a freshly-made-from-975edc8 reference is vacuous). Snapshot a new-pipeline reference shard post-run for future passes. Run `--allow-dirty` (C2 bump uncommitted until the C2 force-amend; no unapproved WIP commit) → `run_tree_dirty=true` stamped (artifacts trained against WIP; C2 commits code+artifacts together, re-runnable).
- [ ] **GATE — explicit owner go to LAUNCH the retrain** (spec §10): expensive + value-changing; separate from the C2 commit approval. **GIVEN 2026-10-10.**
- [ ] Re-materialize corpus silver (`materialize_tc3_frames.py --cache-dir ~/pining-cache --out ~/tc3-kinematics --providers gradientsports skillcorner idsse`, DGX, 975edc8) → retrain the 5 families (owner/DGX); update each `SHA256SUMS`; `load()` runs `verify_chirality` + `_feature_contract` (declared constants compare first).
- [ ] Test: artifact load/integrity gates green; `.gitattributes binary` pin for any new weights dir. Full suite green.

> **C2 BOUNDARY:** suite green + `/final-review` → owner approval → commit C2.

---

## Phase C3 — TF-64 GK detection-gate

> Per the approved spec `2026-10-01-gk-detection-gate-design.md` §3–§4. One commit, owner-approved. Re-anchor that spec's file:line pins at the start of this phase (rebase `_provider_visibility.py` onto landed TF-58).

### Task 15: Keeper-row wrappers over `detected_mask`

**Files:** `silly_kicks/restdefense/` (8 GK cols), `silly_kicks/tracking/_gk_influence.py` (4), `silly_kicks/tracking/features.py` (`pre_shot_gk_*`), `silly_kicks/gkdv/`, `silly_kicks/gk_decision/` (Tier B).
- [ ] TDD per the approved spec's tests 1–7: non-vacuity over the named 14 + `pre_shot_gk_*` (a real `visibility=False` keeper → NaN/drop; `visibility=True` → numeric); missing/>1 keeper row → not-observed; fail-closed on null visibility; `assume_keeper_observed=True` byte-identical; fully-observed providers byte-identical; report conservation; Hyrum snapshot. Red→green each.

### Task 16: Mechanical population gate (test-8) + sweep-floor disposition

**Files:** the ADR-056 registry gate test; the 4 sweep-floor candidates (`tracking/_gk_geometry.py`, `shot_stopping/_compute.py`, `positioning/_compute.py`, `tracking/_cover_shadows.py`).
- [ ] Implement the mechanical test that DERIVES the keeper-position-reader population from code and asserts each member is detection-gated or in a reasoned out-of-scope allowlist (`gk_decision` Tier A native = caveat-only). Disposition the 4 candidates (gate or allowlist-with-reason). Red→green; full suite green.

> **C3 BOUNDARY:** suite green + `/final-review` → owner approval → commit C3.

---

## Phase C4 — retire xT-GK v1 (NEW, owner-approved 2026-10-10; own commit)

> v2 supersedes v1; retiring avoids a wasted v1 retrain. Distinct commit, own `/final-review` + 2 impl re-reviews + owner approval. Owner owns the cross-repo coordination (lakehouse v1 consumption + `exec_visibility.py:467-472` module-path pin) — stated not a concern.

### Task 18: Remove v1 modules + weights + consumers

**Files (remove):** `silly_kicks/tracking/_xt_gk.py`, `silly_kicks/tracking/_gk_completion.py`, `silly_kicks/tracking/_gk_completion_weights/{default,skillcorner}/`. **Files (edit):** `tracking/__init__.py` (drop re-exports), in-repo consumers `tracking/features.py` / `_fov_registry.py` / `_gk_geometry.py` / `_velocity_availability.py` (Chesterton's Fence — understand each use; redirect to v2 where a live need exists, else drop), `feature_glossary.py`, `NOTICE`, `docs/context/xt-gk.md`, `AGENTS.md` (v1 bullet), `docs/PRIVATE_CONSUMERS.md` (strike v1 rows).
- [ ] Enumerate every in-repo `compute_xt_gk`/`GkCompletionModel`/`_xt_gk`/`_gk_completion` reference (grep) + every test; decide redirect-to-v2 vs drop per consumer. TDD: red (removal breaks a stub) → green (consumer no longer needs v1).
- [ ] Remove modules + weights; update consumers + glossary + NOTICE + docs + PRIVATE_CONSUMERS. `.gitattributes` / registry gates that enumerate the weight dirs updated. Full CI-faithful suite green (the removal trips repo-wide gates — auto-enumerating surfaces, meta-registration, doctests).

> **C4 BOUNDARY:** suite green + `/final-review` → owner approval → commit C4.

---

## Phase CN — release (commit-prep)

### Task 17: Version, CHANGELOG, Hyrum notice, release

- [ ] Re-derive the next-free minor after `git fetch && git merge origin/main`; bump `silly_kicks/_version.py` (that one file only).
- [ ] `CHANGELOG.md` entry keyed `PR-Snnn`: velocity values move on broadcast providers; new `ax`/`ay`/`accel` + `pos_var`/`vel_var`/`accel_var` columns; TF-64 GK honest-NaN; the A/B winner.
- [ ] Hyrum notice (all observable output changes above). NOTICE finalized.
- [ ] Full CI-faithful suite + `/final-review` green.

> **CN BOUNDARY:** owner approval → commit CN → owner merges (non-squash admin), tags only after post-merge CI green (tag push = publish), publishes; session pushes HF model cards post-release; cross-repo lakehouse re-materialise handed to owner.

---

## Self-Review

**1. Spec coverage:** §4.1→Tasks 2 (pure helper) + 2b (wire-up); §4.1a→Task 1 (true byte-identity + discriminating two-game); §4.1b→Task 2b (docstring step); §4.2→Task 3; §4.3→Task 4; §4.4 (out of scope)→no task (correct); §4.5 bullet 1→Task 2b, bullets 2+3→**Task 4b** (in-CI plausibility on committed slices + `@e2e` named GS/WC2022 repros — the earlier self-review's "§11 covers §4.5" was an overclaim, now a real task); §5 arms→Tasks 8/9 (+ floor in C1); §6 A/B→A/B milestone; §7 accel+uncertainty→Tasks 5/6/7 (strict-rise on pos_var AND accel_var); §8 TF-64→Tasks 15/16; §9 retrain/release→Tasks 14 (owner-launch gate) /17, DAS-golden audit→Task 10b; §10 phasing→phase boundaries; §11 testing→Tasks 10/10b/11 + every task's TDD. No uncovered requirement.

**2. Placeholder scan:** TF-64 tasks reference the approved spec's enumerated tests rather than repeating code (acceptable — that spec is the authority and travels via §8); every C1 task carries real test code. ADR number is `ADR-NNN` pending allocation at write time (standard). No "TBD"/"add error handling"/"similar to Task N".

**3. Type consistency:** `densify_group`/`segments`/`bridge_small` (Task 2) consumed by Tasks 2b/3/5/7/9 with matching signatures; `kalman_ca -> KalmanOut(pos,vel,acc,var)` (Task 6) consumed by Tasks 7/9; the 6 new columns named identically across schema/reflection/metric_contracts/glossary/tests; `smoothing_method` values `savgol|ema|butterworth|kalman` consistent in config/smoothing/velocity.
