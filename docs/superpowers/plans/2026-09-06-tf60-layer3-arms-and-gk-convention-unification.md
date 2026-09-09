# TF-60 Layer-3 Arms + Ghost-GK Convention Unification — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the complete TF-60 Layer-3 counterfactual deterrent arms (keeper + outfield) as one cycle, unify the ghost-GK goal-relative convention to the correct both-axes point reflection (retrain all variants), move the goal-relative→frame conversion into the serve seams, and re-run the gkdv/TF-19 sign-off from the new weights.

**Architecture:** A gkdv-*sibling* counterfactual engine in `restdefense/` ghosts the in-possession team's rearguard (or keeper) to a league-average model in the same frame state, and prices the opponent's counter-danger via gkdv's *generic* delta seams (`negative = deterrent`). The ghost-GK feature extractor is switched from x-only to the full 180° point reflection (all signed-y features + the target flip); its goal-relative transforms are hoisted to module scope so both the extractor and the serve share one function. Both ghost serves gain frame-ready `ghost_x/_y` so no consumer re-derives orientation.

**Tech Stack:** Python 3.10–3.13, pandas (2.x + 3.x span, ADR-057), numpy, scikit-learn (HGBR, ghost models), scipy; xgboost/numba optional; hexagonal pure-function design.

**Spec:** `docs/superpowers/specs/2026-09-06-tf60-layer3-arms-and-gk-convention-unification-design.md` (read it — the plan argues from it).

## Global Constraints

- **Commit discipline (owner, absolute):** NEVER `git commit`/`push`/`tag`/`merge` without an explicit per-action owner go-ahead. **Phase A produces NO commits** (all tasks on one feature branch; each task ends with its tests green, not a commit). The ONLY commits are the two provenance-mandated Phase-B commits (§9 of the spec). No micro-commits, no worktrees — one feature branch off `main`.
- **TDD, always:** every task writes the failing test first, watches it fail, then the minimal implementation, then green. Full non-e2e suite green before proposing any commit.
- **Author does NOT run `/review-*` on own work** — the owner starts the independent `/review-plan` and `/review-impl` sessions.
- **restdefense import allowlist (ADR-037):** `restdefense/` imports `silly_kicks.tracking` / `silly_kicks.gkdv` **public** seams + `silly_kicks.id_compat` + `silly_kicks._frame_index` ONLY; never a private (`._foo`) tracking/gkdv submodule; `tracking` never imports `restdefense`. The probe reuses gkdv's **public verdict functions** only, never the private `_probe.py` threshold constants.
- **Orientation (ADR-055/051-D3):** direction NEVER from team identity; `resolve_defended_goals(frames) → GoalMap` built once per match and threaded; unresolved end → honest-NaN, never a confident guess.
- **id comparisons (ADR-019):** every id compare/join through `silly_kicks.id_compat` (`ids_equal`/`ids_match`/`same_id`/`align_join_keys`/`canonical_id`), never raw `==`; never `astype(str)` on an id used as a dict key/join token.
- **Nullable dtypes (ADR-027):** counts `Int64`, metrics `float64`, flags `boolean`; NA on unscoreable rows, never a sentinel 0.
- **`negative = deterrent`** for every arm (attacker-value units; gkdv sign convention).
- **Conservation (ADR-042):** `n_frames_scored + Σ drop_reasons == n_frames_in` — a CI gate, not a dataclass property.
- **Phase-A test policy (the transient-red set):** switching the ghost-GK extractor to both-axes (Task 1) invalidates the *bundled* ghost-GK weights' feature-contract + chirality fingerprints, so the ghost-GK **bundled-weight** tests (golden / chirality / feature-contract / integrity-on-load — `tests/tracking/test_ghost_gk*.py` cases that call `GhostGkModel.from_variant(...)` / load bundled npz) are **expected-red during Phase A** and are regenerated from the new weights in Phase B (Task 16/17). Every OTHER test — including all new tests, which use fresh toy-fit models — MUST be green in Phase A. The branch is green at its **tip** (Commit 2), which is all `--merge` requires (spec §9). List the exact expected-red tests in Task 1's notes and re-confirm them green after Task 17.
- **ASCII-only `scripts/` sources** (`test_driver_source_is_ascii`; no `§`/`≥`/`—` in driver files).
- **Lint/type at CI scope:** `python -m ruff check silly_kicks/ tests/ scripts/`, `python -m ruff format --check ...`, `python -m pyright` (bare). Not `.`.

---

## File structure

**Modify (tracking — ghost-GK convention + serves):**
- `silly_kicks/tracking/_ghost_gk.py` — hoist `to_gr_x/to_gr_y/to_gr_vx/to_gr_vy` to module scope; route signed-y features + target through both-axes; add `ghost_x/_y` to the serve output; strengthen the chirality probe to x+y.
- `silly_kicks/tracking/_chirality.py` — the `canonical_probe_frame` / verification y-leg (if the strengthening lives here).
- `silly_kicks/tracking/_ghost_outfield.py` — add `ghost_x/_y` to `serve_ghost_outfield_positions` (via the existing module-level `_gr_x/_gr_y`); extend `_SERVE_OUTPUT_COLS`.
- `silly_kicks/tracking/__init__.py` — no new export (serves already exported); confirm.

**Modify (gkdv — engine consumes frame coords):**
- `silly_kicks/gkdv/_engine.py` — `_build_provenance` reads the serve's `ghost_x/_y` (delete the x-only conversion); `build_ghost_frames` threads them.

**Create (restdefense — Layer 3):**
- `silly_kicks/restdefense/_counterfactual.py` — `build_restdefense_ghost_frames(which=…)`.
- `silly_kicks/restdefense/_arms.py` — `rest_defense_gk_deterrent`, `rest_defense_outfield_deterrent`, `merge_rest_defense`, the 4 arm columns.
- `silly_kicks/restdefense/_probe.py` — dose imposers + `EXPECTED_DIRECTION`; reuses gkdv public verdicts.
- `silly_kicks/restdefense/_ghost_report.py` — `RestDefenseGhostReport` (or add to `_report.py`).

**Modify (restdefense — surface + columns + rollup):**
- `silly_kicks/restdefense/_columns.py` — +4 arm columns + `RD_ARM_SOURCE_VALUES`.
- `silly_kicks/restdefense/_compute.py` — `summarize_rest_defense` arm-column rollup.
- `silly_kicks/restdefense/__init__.py` — export the new public surface.

**Create (scripts):**
- `scripts/build_tf60_layer3_arm_values.py` — corpus driver (ADR-052 shards + ADR-037 provenance).

**Create/modify (tests):** per task; plus registry edits in `tests/restdefense/`, `tests/sb360/_registry.py`, `tests/scripts/test_provenance_wiring.py`, `tests/scripts/_script_population.py` (ADR-056), the glossary/mirror/id-scalar registries.

**Docs (Commit 1):** `docs/superpowers/adrs/ADR-089-*.md`, `CHANGELOG.md`, `silly_kicks/_version.py`, `CLAUDE.md`, `NOTICE`, parent spec §17 arc, `silly_kicks/feature_glossary.py`.

---

## Pre-registered construct-validity anchor (LOCKED — owner-ratified 2026-09-06)

The confirmatory hypotheses, tests, thresholds, and keeper list below are **owner-ratified 2026-09-06 and locked before Phase B** (directions, tests, |ρ|≥0.10 / p<0.05 floors, and the {Alisson, Neuer} list all confirmed). T16 only **executes** them; it does not define them. A non-decisive result is reported as "unvalidatable where the arm would matter," never re-tuned; the anchor is stamped into `findings.md` verbatim. Reported-not-gated (repo convention).

**Outfield arm — LOCKED:**
- Hypothesis: possessions with a stronger Layer-1 rearguard score a more-negative (more deterrent) outfield arm.
- Test: one-sided Spearman rank correlation ρ, over the public corpus, between the per-possession outfield arm (`rd_outfield_deter_threat`, and separately `rd_outfield_deter_space`) and the per-possession Layer-1 `rd_num_superiority`; and separately against **negated** rearguard compactness (`-rd_compactness_x`, since more-compact = smaller x-range). Population: possessions with a scored arm value AND a resolved Layer-1 sample.
- Confirmatory criterion (pre-registered): **ρ < 0** (higher superiority → more deterrent), one-sided **p < 0.05**, AND **|ρ| ≥ 0.10** (minimal effect-size floor; the actual ρ + CI is reported regardless).

**Keeper arm — LOCKED:**
- Hypothesis: aggressive high-line/sweeper keepers score more deterrent than the corpus-median keeper.
- Named-keeper list (**LOCKED — owner-ratified 2026-09-06**, stamped into the artifact): **{Alisson, Neuer}** — identical to TF-19's already-owner-ratified `NAMED_KEEPER_PRIOR` (the same sweeper-keeper archetype). Not to change after the run.
- Test: one-sided Mann-Whitney rank test of the named set's per-keeper `rd_gk_deter_threat` vs the rest of the corpus's per-keeper values, plus an ADR-060 `exceeds_noise_floor`-style paired-difference SE report.
- Confirmatory criterion (pre-registered): the named-keeper arm is **below** the corpus median (more deterrent), one-sided **p < 0.05**.

**Instrument-validity (both arms, per §7):** the pooled Layer-0 (`layer0_instrument_verdict`) + Layer-1 (`layer1_responsiveness_verdict`) verdicts are reported per arm. Expectation (reported, not asserted): outfield ΔDAS + both threat arms `instrument_valid`; the keeper ΔDAS is likely `arm_unscoreable`/weak and is shipped with that honest verdict, not hidden.

---

## Phase A — code (one branch, NO commits; each task ends green)

### Task 1: Ghost-GK both-axes convention (hoist transforms + flip signed-y)

**Files:**
- Modify: `silly_kicks/tracking/_ghost_gk.py` (`extract_ghost_gk_features` ~`:734-856`; hoist `to_gr_x`/`to_gr_vx` from the closures `:737-741` to module scope; add `to_gr_y`/`to_gr_vy`).
- Test: `tests/tracking/test_ghost_gk_convention.py` (new).

**Interfaces:**
- Produces: module-level `_to_gr_x(x, flip)`, `_to_gr_y(y, flip)`, `_to_gr_vx(vx, flip)`, `_to_gr_vy(vy, flip)` in `_ghost_gk.py` (used by the extractor AND `serve_ghost_gk_positions` in Task 3). `flip = goal_x > 50.0`. `_to_gr_x`/`_to_gr_y` are self-inverse: `_to_gr_x(_to_gr_x(x, f), f) == x`.

- [ ] **Step 1: Write the failing test — the extractor is now y-chirality-invariant.**

```python
# tests/tracking/test_ghost_gk_convention.py
import numpy as np
import pandas as pd
from silly_kicks.tracking._ghost_gk import extract_ghost_gk_features, _to_gr_x, _to_gr_y

def _one_frame(*, goal_x):
    # A minimal single frame: ball + one GK + a few defenders/attackers, all finite.
    rows = [
        {"is_ball": True, "is_goalkeeper": False, "team_id": None, "player_id": None, "x": 60.0, "y": 20.0, "vx": 1.0, "vy": -2.0},
        {"is_ball": False, "is_goalkeeper": True, "team_id": 1, "player_id": 10, "x": (5.0 if goal_x < 50 else 100.0), "y": 30.0, "vx": 0.0, "vy": 0.0},
        {"is_ball": False, "is_goalkeeper": False, "team_id": 1, "player_id": 11, "x": (20.0 if goal_x < 50 else 85.0), "y": 15.0, "vx": 0.0, "vy": 0.0},
        {"is_ball": False, "is_goalkeeper": False, "team_id": 2, "player_id": 21, "x": 55.0, "y": 40.0, "vx": 0.0, "vy": 0.0},
    ]
    df = pd.DataFrame(rows)
    df["game_id"] = 1; df["period_id"] = 1; df["frame_id"] = 1; df["time_seconds"] = 0.0
    return df

def test_selfinverse_transforms():
    for f in (True, False):
        assert _to_gr_x(_to_gr_x(37.0, f), f) == 37.0
        assert _to_gr_y(_to_gr_y(21.0, f), f) == 21.0

def test_extractor_is_y_chirality_invariant():
    # A scene at goal_x=0 and its 180-deg point reflection at goal_x=105 must produce the SAME
    # goal-relative feature row. Under the OLD x-only convention the signed-y features (ball_y,
    # atk_cy, ball_to_goal_angle, ...) differ -> this test is RED before the both-axes change.
    left = _one_frame(goal_x=0.0)
    right = left.copy()
    right["x"] = 105.0 - right["x"]
    right["y"] = 68.0 - right["y"]
    right["vx"] = -right["vx"]; right["vy"] = -right["vy"]
    fl = extract_ghost_gk_features(left, gk_team_id=1, goal_x=0.0)
    fr = extract_ghost_gk_features(right, gk_team_id=1, goal_x=105.0)
    np.testing.assert_allclose(fl.to_numpy(dtype=float), fr.to_numpy(dtype=float), atol=1e-9, equal_nan=True)
```

- [ ] **Step 2: Run to verify it fails.** Run: `python -m pytest tests/tracking/test_ghost_gk_convention.py -v`. Expected: `test_extractor_is_y_chirality_invariant` FAILS (signed-y features differ); `_to_gr_y` import fails (`ImportError`) until Step 3.

- [ ] **Step 3: Hoist the transforms to module scope and route signed-y through them.** In `_ghost_gk.py`, add module-level (near the other module constants):

```python
def _to_gr_x(x: float, flip: bool) -> float:
    return (_FIELD_LENGTH - x) if flip else x

def _to_gr_y(y: float, flip: bool) -> float:
    return (_FIELD_WIDTH - y) if flip else y

def _to_gr_vx(vx: float, flip: bool) -> float:
    return -vx if flip else vx

def _to_gr_vy(vy: float, flip: bool) -> float:
    return -vy if flip else vy
```

Inside `extract_ghost_gk_features`, delete the local `to_gr_x`/`to_gr_vx` closures (`:737-741`); set `flip = goal_x > 50.0` and replace every `to_gr_x(...)` with `_to_gr_x(..., flip)` (and `to_gr_vx` → `_to_gr_vx(..., flip)`). Then route the signed-y features (spec §5.1) through the new helpers:
- `ball_y = _to_gr_y(by_raw, flip)` (was `by_raw`, `:754`)
- `ball_vy = _to_gr_vy(bvy_raw, flip)` (was `bvy_raw`, `:756`)
- `atk_cy = float(np.mean(_to_gr_y(np.asarray(attacking["y"].values), flip)))` (was raw mean, `:800`) — note `_to_gr_y` is scalar; apply the `(68 - y) if flip` vectorized form here.
- `ball_to_goal_angle` already reads `ball_y` (now goal-relative) → correct once `ball_y` flips (`:816`).
- The **target** `gr_y` (the keeper's predicted-position label built at fit time — find the target-y assignment and route it through `_to_gr_y(y, flip)`).

Leave the y-invariant features unchanged (`ball_dist`, `defensive_line_width`, box count, hull area, velocity-derivatives — spec §5.1).

- [ ] **Step 4: Run to verify green.** Run: `python -m pytest tests/tracking/test_ghost_gk_convention.py -v`. Expected: PASS.

- [ ] **Step 5: Record the transient-red set.** Run `python -m pytest tests/tracking/test_ghost_gk.py -v` and note which cases now fail on bundled-weight load (feature-contract/chirality mismatch). Write the list into this task's notes; these are the Phase-B-regenerated tests (do NOT try to fix them in Phase A). NO commit.

### Task 2: Strengthen the ghost-GK chirality probe to x+y

**Files:**
- Modify: `silly_kicks/tracking/_chirality.py` (`canonical_probe_frame` `:22`; the verification).
- Test: `tests/tracking/test_ghost_gk_convention.py` (extend).

**Interfaces:**
- Consumes: Task 1's both-axes extractor.
- Produces: a `verify_chirality` that exercises BOTH the x-goal-flip and a y-mirror, so a regression to x-only (or a half-flip) RAISES.

- [ ] **Step 1: Write the failing test.**

```python
def test_chirality_probe_catches_a_y_only_regression(monkeypatch):
    # If someone reverts atk_cy/ball_y to absolute, the chirality verification must RAISE.
    import silly_kicks.tracking._ghost_gk as g
    # Fit a tiny model on toy data, then monkeypatch _to_gr_y to the identity (x-only regression)
    # and assert verify_chirality raises.
    ...  # (fill with the concrete toy-fit + monkeypatch, mirroring tests/tracking/test_ghost_gk.py's fit helpers)
```

- [ ] **Step 2: Run — expect FAIL** (`verify_chirality` currently only checks the x-flip).
- [ ] **Step 3: Extend the probe.** In `_chirality.py`, add a y-mirror leg to `canonical_probe_frame` / the verification so a scene and its 180° point reflection must score consistently. Keep the existing x-leg.
- [ ] **Step 4: Run — PASS.** NO commit.

### Task 3: `serve_ghost_gk_positions` emits frame `ghost_x/_y`

**Files:**
- Modify: `silly_kicks/tracking/_ghost_gk.py` (`serve_ghost_gk_positions` `:2795`; the output-frame construction `:2900-2914` and the empty-frame branch `:2886-2899`).
- Test: `tests/tracking/test_ghost_gk_serve_framecoords.py` (new).

**Interfaces:**
- Consumes: Task 1's module-level `_to_gr_x`/`_to_gr_y`.
- Produces: `serve_ghost_gk_positions` output gains columns `ghost_x`, `ghost_y` (frame coords), keeping `ghost_gr_x`/`ghost_gr_y` as audit. Empty-frame branch adds `ghost_x`/`ghost_y` as `pd.Series(dtype=float)`.

- [ ] **Step 1: Write the failing test — frame coords match the model's own inverse, and are mirror-consistent.**

```python
# tests/tracking/test_ghost_gk_serve_framecoords.py
import numpy as np
from silly_kicks.tracking import serve_ghost_gk_positions
from silly_kicks.tracking._ghost_gk import _to_gr_x, _to_gr_y

def test_serve_emits_frame_coords_matching_the_inverse(toy_frames_with_gk, toy_gk_model):
    out = serve_ghost_gk_positions(toy_frames_with_gk, model=toy_gk_model, home_team_id=1)
    assert {"ghost_x", "ghost_y", "ghost_gr_x", "ghost_gr_y"} <= set(out.columns)
    # For each served row, frame == inverse of goal-relative under the row's defended-goal flip.
    # (The test derives flip from the row's gk_team_id via resolve_defended_goals.)
    ...
```

- [ ] **Step 2: Run — FAIL** (`ghost_x`/`ghost_y` absent).
- [ ] **Step 3: Implement.** In the serve, after building `positions`/`meta`, resolve each served keeper's defended-goal end (the serve already threads `goal_x` per keeper through `_serve_positions_core`; reuse it — do NOT re-derive orientation independently) and compute `ghost_x = _to_gr_x(ghost_gr_x, flip)`, `ghost_y = _to_gr_y(ghost_gr_y, flip)`. Add both columns to the populated output (`:2900-2914`) and the empty-frame branch (`:2886-2899`).
- [ ] **Step 4: Run — PASS.** NO commit.

### Task 4: `serve_ghost_outfield_positions` emits frame `ghost_x/_y`

**Files:**
- Modify: `silly_kicks/tracking/_ghost_outfield.py` (`serve_ghost_outfield_positions` `:1038`; `_SERVE_OUTPUT_COLS` `:938`).
- Test: `tests/tracking/test_ghost_outfield_model.py` (extend) or a new `test_ghost_outfield_serve_framecoords.py`.

**Interfaces:**
- Consumes: the existing module-level `_gr_x`/`_gr_y` (`:158/163`).
- Produces: serve output gains `ghost_x`, `ghost_y` (frame coords); `_SERVE_OUTPUT_COLS` extended.

- [ ] **Step 1: Write the failing test** (mirror of Task 3, keyed by `(team_id, slot_index)`; a `variant_unavailable`/`fov_cropped` row → `ghost_x/_y` NaN).
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement.** After `preds`, resolve each `(team_id)`'s defended-goal end (the extractor already builds a `GoalMap`; reuse the same `flip`) and set `ghost_x = _gr_x(ghost_gr_x, flip)`, `ghost_y = _gr_y(ghost_gr_y, flip)`; NaN ghosts stay NaN. Extend `_SERVE_OUTPUT_COLS`.
- [ ] **Step 4: Run — PASS.** NO commit.

### Task 5: gkdv engine consumes the serve's frame coords

**Files:**
- Modify: `silly_kicks/gkdv/_engine.py` (`_build_provenance` `:382-396`; `build_ghost_frames` `:554`).
- Test: `tests/gkdv/test_engine_framecoords.py` (new).

**Interfaces:**
- Consumes: Task 3's serve `ghost_x/_y`.
- Produces: `_build_provenance` reads `ghost_x`/`ghost_y` from the serve output (no local goal-relative→frame math); `provenance` shape unchanged (`_PROVENANCE_COLUMNS`).

- [ ] **Step 1: Write the failing test — gkdv uses the serve's frame coords, and away-team y is correct.**

```python
# tests/gkdv/test_engine_framecoords.py
def test_engine_uses_serve_frame_coords_and_is_mirror_correct(toy_frames, toy_gk_model):
    from silly_kicks.gkdv import build_ghost_frames
    cf, prov, rep = build_ghost_frames(toy_frames, model=toy_gk_model, home_team_id=1)
    scored = prov[prov["drop_reason"].isna()]
    # ghost_x/ghost_y equal the serve's frame coords (not a locally re-derived x-only value).
    # Mirror the frames (both axes) + swap home_team_id; the DEFENDING keeper's ghost position
    # must be the mirror of the un-mirrored one (this is RED if the engine still does x-only y).
    ...
```

- [ ] **Step 2: Run — FAIL** (the engine's x-only `ghost_y = ghost_gr_y` mislocates away-team y once the model is both-axes).
- [ ] **Step 3: Implement.** In `build_ghost_frames`, capture the serve output's `ghost_x`/`ghost_y`; in `_build_provenance`, replace the `np.where(defended==0.0, gr_x, 105-gr_x)` block (`:390-392`) with a join that carries the serve's `ghost_x`/`ghost_y` through. Keep `ghost_gr_x`/`ghost_gr_y` as audit. `_write_back` is unchanged (already writes `ghost_x/_y`).
- [ ] **Step 4: Run — PASS**, and run `python -m pytest tests/gkdv/ -v` to confirm no gkdv regression (note: gkdv tests that load the *bundled* ghost-GK model are in the Task-1 transient-red set). NO commit.

### Task 6: `restdefense/_counterfactual.py` — `build_restdefense_ghost_frames`

**Files:**
- Create: `silly_kicks/restdefense/_counterfactual.py`.
- Create: `silly_kicks/restdefense/_ghost_report.py` (`RestDefenseGhostReport`).
- Test: `tests/restdefense/test_counterfactual.py` (new).

**Interfaces:**
- Consumes: `silly_kicks.tracking.{serve_ghost_gk_positions, serve_ghost_outfield_positions, resolve_defended_goals, infer_ball_carrier}`; `silly_kicks.id_compat`; `silly_kicks._frame_index.group_rows`.
- Produces:
  ```python
  build_restdefense_ghost_frames(
      frames, *, which: Literal["keeper","rearguard"], model, home_team_id,
      carrier=None, params: RestDefenseParams = _DEFAULT,
  ) -> tuple[pd.DataFrame, pd.DataFrame, RestDefenseGhostReport]
  # returns (cf_frames, provenance, report). provenance keys (game_id, period_id, frame_id, team_id, player_id)
  # with drop_reason (NA on scored). report fields: params, n_frames_in, n_frames_scored, drop_reasons.
  ```

- [ ] **Step 1: Write failing tests — PURE, conservation, write-back moves ONLY A's rearguard, non-finite raises, missing→drop.**

```python
# tests/restdefense/test_counterfactual.py
def test_pure_and_conserves(rd_frames, outfield_model):
    before = rd_frames.copy(deep=True)
    cf, prov, rep = build_restdefense_ghost_frames(
        rd_frames, which="rearguard", model=outfield_model, home_team_id=1)
    pd.testing.assert_frame_equal(rd_frames, before)                     # PURE
    assert rep.n_frames_scored + sum(rep.drop_reasons.values()) == rep.n_frames_in  # conservation

def test_writeback_moves_only_As_rearguard(rd_frames, outfield_model):
    cf, prov, rep = build_restdefense_ghost_frames(
        rd_frames, which="rearguard", model=outfield_model, home_team_id=1)
    scored = prov[prov["drop_reason"].isna()]
    moved = cf.merge(rd_frames, on=["game_id","period_id","frame_id","player_id"], suffixes=("_cf",""))
    changed = moved[(moved["x_cf"] != moved["x"]) | (moved["y_cf"] != moved["y"])]
    # every changed row is a scored A-rearguard player; no ball/keeper/opponent/other-frame row moved.
    ...

def test_nonfinite_served_ghost_on_scored_frame_raises(...):
    # a model that serves NaN on a scored frame -> ValueError (mirrors gkdv _engine.py:574).
    ...

def test_missing_ghost_is_dropped_and_counted_not_zero(...):
    # a frame the serve returns nothing for is a drop_reason, never scored as delta=0.
    ...
```

- [ ] **Step 2: Run — FAIL** (module absent).
- [ ] **Step 3: Implement `build_restdefense_ghost_frames`** — model it on `gkdv/_engine.py` (domain filter → serve → provenance → write-back), but ghost A's OWN rearguard near A's OWN goal (committed-forward gate from `params.min_ball_advance_m`). `which="rearguard"` → `serve_ghost_outfield_positions(..., n_rearguard=params.n_rearguard)`, matched on `(game,period,frame,team_id,player_id)`; `which="keeper"` → `serve_ghost_gk_positions(...)` (the model must be a `sweeper` variant — validate/document). Write back the serve's **frame** `ghost_x/_y` (no reflection math). Non-finite served ghost on a scored frame → raise; missing/NaN/`variant_unavailable`/`fov_cropped` → dropped-and-counted. `RestDefenseGhostReport` in `_ghost_report.py`.
- [ ] **Step 4: Run — PASS.** Register a scoped `group_rows` guard in `tests/_scale_guarded.SCALE_GUARDED` + a sub-quadratic-growth fixture (ADR-073) if a new per-frame loop is introduced. NO commit.

### Task 7: `restdefense/_arms.py` — the four arm columns

**Files:**
- Create: `silly_kicks/restdefense/_arms.py`.
- Modify: `silly_kicks/restdefense/_columns.py` (+4 columns + `RD_ARM_SOURCE_VALUES`).
- Test: `tests/restdefense/test_arms.py` (new).

**Interfaces:**
- Consumes: `build_restdefense_ghost_frames` (Task 6); `silly_kicks.gkdv.{delta_das_batch, delta_threat_suppression_batch, GkdvParams}`; `restdefense._windows.select_rest_defense_samples` (the grain bridge — emits `frame_id` per scored in-possession sample).
- Produces:
  ```python
  rest_defense_gk_deterrent(actions, frames, *, xt, ghost_gk_model, home_team_id,
      goal_map=None, links=None, carrier=None, visible_area=None, params=_DEFAULT)
      -> tuple[pd.DataFrame, RestDefenseGhostReport]
      # columns: game_id, period_id, team_id, action_id, rd_gk_deter_threat, rd_gk_deter_space, rd_gk_source
  rest_defense_outfield_deterrent(actions, frames, *, xt, ghost_outfield_model, home_team_id, ...)
      -> tuple[pd.DataFrame, RestDefenseGhostReport]
      # columns: ..., rd_outfield_deter_threat, rd_outfield_deter_space, rd_outfield_source
  ```
  `RD_ARM_SOURCE_VALUES = ("computed","ghost_missing","unlinked","unresolved","fov_cropped")`.

- [ ] **Step 1: Write failing tests — negative=deterrent, keeper-agnostic outfield delta, non-vacuity/mirror-invariance, grain bridge, honest-NaN.**

```python
# tests/restdefense/test_arms.py
def test_arm_is_negative_for_a_better_than_average_rearguard(...):
    # a rearguard positioned better than the league-average ghost -> arm < 0.
    ...

def test_outfield_arm_is_keeper_agnostic(rd_frames, outfield_model, xt):
    # perturb A's KEEPER on both legs; the outfield arm value is unchanged (keeper cancels in the delta).
    base, _ = rest_defense_outfield_deterrent(actions, rd_frames, xt=xt,
        ghost_outfield_model=outfield_model, home_team_id=1)
    moved = rd_frames.copy(); # move A's keeper only
    ...
    pd.testing.assert_series_equal(base["rd_outfield_deter_threat"], moved_out["rd_outfield_deter_threat"])

def test_ghost_leg_measurably_differs(...):
    # mirrors tests/gkdv/test_arms.py:383 -- an unpinned/no-op implementation would differ.
    ...

def test_mirror_invariance_of_the_arm(rd_frames, outfield_model, xt):
    # mirror the frames (both axes) + swap home_team_id; the arm deltas are invariant.
    # This is the both-axes write-back guard: an x-only write-back would break it.
    ...

def test_arm_keys_are_a_subset_of_sample_keys(...):
    # every arm row keys onto a compute_rest_defense sample; drop-domain is a declared subset.
    ...
```

- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement.** Reuse `select_rest_defense_samples` for the sample set; build ghost frames (Task 6) for the linked frames; call `delta_threat_suppression_batch` / `delta_das_batch` with `attacking_team_id_by_frame = B` (per-frame opponent Series) + `GkdvParams()`; map the per-frame delta back to the per-action arm table via `frame_id`/`action_id`. One shared `rd_<arm>_source` (domain provenance). `xt` required + fail-closed. Add the 4 columns + vocab to `_columns.py`.
- [ ] **Step 4: Run — PASS.** NO commit.

### Task 8: `merge_rest_defense` + `summarize_rest_defense` arm rollup

**Files:**
- Modify: `silly_kicks/restdefense/_arms.py` (add `merge_rest_defense`) or `_compute.py`.
- Modify: `silly_kicks/restdefense/_compute.py` (`summarize_rest_defense` rollup).
- Test: `tests/restdefense/test_merge_and_rollup.py` (new).

**Interfaces:**
- Produces: `merge_rest_defense(samples, *arms) -> pd.DataFrame` (left-join, honest-NaN on arm-dropped rows, asserts arm keys ⊆ sample keys, reconciles conservation across reports); `summarize_rest_defense` means arm columns when present.

- [ ] **Step 1: Write failing tests** — left-join keeps all samples with NaN on arm-dropped rows; a raised assertion if an arm key is NOT a sample key; the per-match rollup means the arm columns; Layer-1/2 aggregation byte-identical with/without arm columns.
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement.** `merge_rest_defense` via `align_join_keys` + left-merge on `RD_SAMPLE_KEYS`; extend `summarize_rest_defense`'s `numeric` list to include any present arm columns (guard: only if present, so Layer-1/2-only callers are byte-identical).
- [ ] **Step 4: Run — PASS.** NO commit.

### Task 9: restdefense public surface

**Files:**
- Modify: `silly_kicks/restdefense/__init__.py` (export `build_restdefense_ghost_frames`, `rest_defense_gk_deterrent`, `rest_defense_outfield_deterrent`, `merge_rest_defense`, `RestDefenseGhostReport`, the 4 column constants + `RD_ARM_SOURCE_VALUES`).
- Test: `tests/restdefense/test_import_allowlist.py` (already asserts public seam; extend the surface assertion).

- [ ] **Step 1: Write the failing test** — the new names import from `silly_kicks.restdefense`; the import-allowlist gate still passes (no private tracking/gkdv import; probe reuses public verdicts).
- [ ] **Step 2: Run — FAIL** (names not exported).
- [ ] **Step 3: Implement** the `__all__` additions.
- [ ] **Step 4: Run — PASS**, and `python -m pytest tests/restdefense/test_import_allowlist.py -v`. NO commit.

### Task 10: `restdefense/_probe.py` — instrument-validity probe

**Files:**
- Create: `silly_kicks/restdefense/_probe.py`.
- Test: `tests/restdefense/test_probe.py` (new).

**Interfaces:**
- Consumes: `silly_kicks.gkdv.{layer0_instrument_verdict, layer1_responsiveness_verdict}` (public verdict FUNCTIONS only — they read the private thresholds internally); `build_restdefense_ghost_frames` (Task 6); `group_rows`.
- Produces: `impose_rearguard_dose(frames, *, which, home_team_id, dose, displacement=None, model=None, params=_DEFAULT) -> (imposed_frames, targets)`; `paired_vector_controls(frames, targets, *, r, rng)` (rearguard analog); `EXPECTED_DIRECTION` for the 4 rd arm columns; `expected_direction_for_arm(col)`.

- [ ] **Step 1: Write failing tests** — dose imposition moves only the targeted rearguard/keeper; paired controls are distinct single-player displacements; the pooled verdict functions are reused (a thin-domain pool → `arm_unscoreable`); a planted no-op dose → `not_responsive`.
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement**, mirroring `gkdv/_probe.py` (`_build_dose_targets` / `impose_defending_keeper_dose` / `paired_vector_controls` structure) but for the rearguard/keeper interventions and A's own outfielders as the placebo pool. Reuse the two gkdv public verdict functions verbatim.
- [ ] **Step 4: Run — PASS.** NO commit.

### Task 11: `scripts/build_tf60_layer3_arm_values.py` — corpus driver

**Files:**
- Create: `scripts/build_tf60_layer3_arm_values.py`.
- Test: `tests/scripts/test_build_tf60_layer3_arm_values.py` (new).

**Interfaces:**
- Consumes: `restdefense.rest_defense_{gk,outfield}_deterrent`; `scripts/_driver.for_each` (ADR-052); `scripts/_provenance.{git_provenance, require_clean_tree}` (ADR-037); the loader (`scripts/_loader_pining` / the corpus loaders).
- Produces: per-match shards → a per-`(team, match)` arm-values table; `_EMITTED_SHARD_COLUMNS` + `_SHARD_SCHEMA_VERSION` pinned together (4.77.1); ASCII-only source.

- [ ] **Step 1: Write failing tests** — `require_clean_tree` called from `main()`; `--allow-dirty` offered; no `rev-parse` shell-out; declared shard columns match the dict the work function builds; ASCII-only source; registered in `ARTIFACT_DRIVERS` + the ADR-056 script-population gate.
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement**, mirroring `scripts/build_gkdv_arm_values.py` + `scripts/build_tf19_instrument_responsiveness.py`.
- [ ] **Step 4: Run — PASS**, and `python -m pytest tests/scripts/test_provenance_wiring.py -v`. NO commit.

### Task 11b: `@e2e` method gate — both arms on a real linked-tracking match + a real SB360 match (spec §10)

**Files:**
- Create: `tests/restdefense/test_layer3_arms_e2e.py` (marked `@pytest.mark.e2e`; owner/fixture-gated, does NOT run in the normal `-m "not e2e"` suite).

**Interfaces:**
- Consumes: `rest_defense_gk_deterrent`, `rest_defense_outfield_deterrent`, `merge_rest_defense` (T7/T8); the shipped ghost models (`GhostGkModel.from_variant("sweeper")`, `GhostOutfieldModel.from_variant(...)`); `resolve_keeper_identities` / `apply_keeper_identities_to_frames` (ADR-078) for the SB360 leg; a fitted `xt`.

- [ ] **Step 1: Write the e2e test — assert the METHOD, never metric values (so it stays compatible with reported-not-gated).**

```python
# tests/restdefense/test_layer3_arms_e2e.py
import pytest

@pytest.mark.e2e
def test_layer3_arms_on_real_linked_tracking(real_tracking_match, fitted_xt):
    actions, frames = real_tracking_match          # native GK, full-coverage FOV
    gk_arm, gk_rep = rest_defense_gk_deterrent(actions, frames, xt=fitted_xt,
        ghost_gk_model=GhostGkModel.from_variant("sweeper"), home_team_id=real_tracking_match.home)
    of_arm, of_rep = rest_defense_outfield_deterrent(actions, frames, xt=fitted_xt,
        ghost_outfield_model=GhostOutfieldModel.from_variant("default"), home_team_id=real_tracking_match.home)
    assert len(gk_arm) > 0 and len(of_arm) > 0
    for rep in (gk_rep, of_rep):
        assert rep.n_frames_scored + sum(rep.drop_reasons.values()) == rep.n_frames_in   # conservation reconciles
    samples, _ = compute_rest_defense(actions, frames, xt=fitted_xt)
    merged = merge_rest_defense(samples, gk_arm, of_arm)
    assert set(merged.columns) >= {"rd_gk_deter_threat", "rd_outfield_deter_threat"}

@pytest.mark.e2e
def test_layer3_arms_on_real_sb360(real_sb360_match, sb360_roster, fitted_xt):
    # roster keeper identity + FOV companions with a <1.0 observed fraction on a cropped advanced-ball frame.
    actions, frames, visible_area = real_sb360_match
    keeper_map = resolve_keeper_identities(actions, frames, identity="roster", roster=sb360_roster)
    frames = apply_keeper_identities_to_frames(frames, keeper_map)
    of_arm, of_rep = rest_defense_outfield_deterrent(actions, frames, xt=fitted_xt,
        ghost_outfield_model=GhostOutfieldModel.from_variant("position_only"),
        home_team_id=real_sb360_match.home, visible_area=visible_area)
    assert of_rep.n_frames_scored + sum(of_rep.drop_reasons.values()) == of_rep.n_frames_in
    samples, _ = compute_rest_defense(actions, frames, xt=fitted_xt, visible_area=visible_area)
    # a genuinely cropped advanced-ball frame yields an observed fraction < 1.0 on a region companion.
    frac_cols = [c for c in samples.columns if c.endswith("_observed_fraction")]
    assert any((samples[c] < 1.0).any() for c in frac_cols)
```

- [ ] **Step 2: Run to confirm collection + skip.** Run: `python -m pytest tests/restdefense/test_layer3_arms_e2e.py -v` (no fixtures → skips/errors on the missing owner fixtures) and `python -m pytest tests/restdefense/test_layer3_arms_e2e.py -m e2e --collect-only` to confirm the markers. The normal `-m "not e2e"` suite does NOT run these.
- [ ] **Step 3: Wire the fixtures** to the owner-provided real-match fixtures (mirroring the existing `@e2e` precedents, e.g. the `worldcup-hdf5-e2e` fixture pattern). The test is a Phase-A deliverable (the code); it executes against owner fixtures at review time, not in CI. NO commit.

### Task 12: Register the new surface in every CI gate

**Files:**
- Modify: `silly_kicks/feature_glossary.py` (+4 arm columns; `emitting_module` = `_arms`).
- Modify: `tests/sb360/_registry.py` (+ `compute`-boundary verdicts for the arms; `verdict_provenance`).
- Modify: `tests/restdefense/` liveness + purity gates (+ arm columns; ≥2 variants for the `visible_area` branch).
- Modify: the D3/mirror registry + id-scalar registry (`tests/invariants/conftest_id_scalar.py`) if a new public id-scalar function was added (the arms take `home_team_id` scalar → likely a registry entry or a delegated/justified exemption).
- Modify: `tests/scripts/_script_population.py` + `tests/scripts/test_provenance_wiring.py` (the new driver).
- Test: the gates themselves.

- [ ] **Step 1: Run the gates to see them fail** on the un-registered new surface (glossary coverage, SB360 audit completeness, id-scalar meta-assertion, mirror registry). Run: `python -m pytest tests/sb360/ tests/restdefense/ tests/invariants/test_public_id_scalar_registry.py -v`.
- [ ] **Step 2: Register** each: glossary records, SB360 verdict + `verdict_provenance`, liveness/purity entries, id-scalar entry (with the matched/mismatched/float axes), mirror registry entry (or the D3 direction-invariance test for a one-team site), script population.
- [ ] **Step 3: Run — all gates PASS.** NO commit.

### Task 13: Docs (ride Commit 1) — ADR-089, CHANGELOG, version, CLAUDE.md, NOTICE, parent-arc, C4 verify

**Files:**
- Create: `docs/superpowers/adrs/ADR-089-tf60-layer3-arms-and-gk-convention-unification.md`.
- Modify: `CHANGELOG.md` (a keyed `PR-Snnn` entry; the ghost-GK convention retrain trigger + gkdv re-mat + TF-19 re-run; the Hyrum notes).
- Modify: `silly_kicks/_version.py` (one minor bump — the exact number claimed at release time, NOT now).
- Modify: `CLAUDE.md` (restdefense Layer-3 contract; the ghost-GK both-axes convention; the serve-frame-coords contract; the retrain triggers).
- Modify: `NOTICE` (Kim 2026 DEFCON, Le 2017, Bischofberger & Baca 2026).
- Modify: `docs/superpowers/specs/2026-08-30-tf60-rest-defense-structure-and-gk-design.md` §17 arc (fold PR4+PR6 + the convention-unification into this cycle).
- Verify: C4 completeness gate green (no new container/aggregator).

- [ ] **Step 1** Write ADR-089 (context = the two-convention inconsistency + the fold; decision = both-axes retrain + complete Layer 3 + serve-frame-coords + TF-19 in-cycle; consequences = retrain triggers, downstream re-runs).
- [ ] **Step 2** CHANGELOG + CLAUDE.md + NOTICE + parent-arc edits.
- [ ] **Step 3** Run `python -m pytest tests/ -m "not e2e"` (whole suite) — everything green EXCEPT the Task-1 transient-red bundled-weight ghost-GK tests. Run ruff + pyright at CI scope. Confirm the C4 gate is green.
- [ ] **Step 4** This is the **Commit-1 point.** STOP — present the full diff + the transient-red list to the owner for Commit-1 approval and the independent `/review-impl`. NO commit without go-ahead.

---

## Phase B — weights + re-runs (DGX; two owner-approved commits)

> Each step is a SEPARATE owner go-ahead. The author does not run `/review-*` on own work.

### Task 14: Commit 1 (owner-approved) — code, clean training SHA

- [ ] Owner approves the Phase-A diff. On go-ahead: `git commit` the code + docs (clean tree; the spec + plan ride here). This is the `training_commit`. Push only on a separate go-ahead. (Branch is transiently red on bundled ghost-GK loads until Commit 2 — expected; gates on the tip.)

### Task 15: DGX — re-fit all 5 ghost-GK variants (both-axes) from Commit-1 SHA

- [ ] On go-ahead: on the DGX (`sk-phaseb-venv`, sklearn [1.9,2)), check out the Commit-1 SHA, run `scripts/train_ghost_gk.py` for the 4 bundled variants (`default`/`position_only`/`sweeper`/`sweeper_position_only`) + `full`, on the public 179-match corpus, clean tree. Verify each: CV metrics, chirality (x+y), feature-contract, round-trip. Record the measured CV delta vs the old x-only weights (spec §5.4 magnitude note).

### Task 16: DGX — Layer-3 arm-values + probe + construct-validity; gkdv + TF-19 re-run

**Prerequisite (coordination):** the gkdv re-materialize + TF-19 sign-off re-run touch the parallel TF-19 workstream's artifacts (spec §5.4). Confirm with the owner that this cycle is authorized to re-produce those artifacts from the new weights before running them (per the owner ruling that another session's inflight work is not a constraint — but the re-produced artifacts must not collide silently).

- [ ] On go-ahead, from Commit-1 SHA + the new weights:
  - Run `scripts/build_tf60_layer3_arm_values.py` over the public corpus → the arm-values table; run the probe → pooled Layer-0/Layer-1 verdicts per arm.
  - **Execute the LOCKED pre-registered anchor** (the "Pre-registered construct-validity anchor" section above — do NOT re-define it): the outfield Spearman ρ tests (vs `rd_num_superiority` and `-rd_compactness_x`; ρ<0, one-sided p<0.05, |ρ|≥0.10) and the keeper named-set ({Alisson, Neuer} as ratified) Mann-Whitney test. Write `docs/research/tf60_layer3_construct_validity/findings.md`: the anchor stamped verbatim, the actual ρ/p/effect + CIs (reported regardless), the per-arm Layer-0/Layer-1 instrument verdicts (keeper ΔDAS honestly flagged), the honest construct-validity note.
  - Re-run `scripts/build_gkdv_arm_values.py` + `scripts/build_tf19_instrument_responsiveness.py` from the new weights; verify the ICC gate still fires (power 1.0) and `NAMED_KEEPER_PRIOR` still holds; write the robustness note (or surface any gate that moved — never silently re-baseline).

### Task 17: Regenerate goldens + un-red the transient set

- [ ] On go-ahead: regenerate the ghost-GK golden / chirality / feature-contract / `SHA256SUMS` from the new weights (the Task-1 transient-red list). Run the full suite — now green including the previously-red bundled-weight tests. Ruff + pyright clean.

### Task 18: Commit 2 (owner-approved) — weights + artifacts

- [ ] Owner approves. On go-ahead: `git commit` the 4 bundled re-fit weight files + regenerated goldens/chirality/feature-contract/`SHA256SUMS`, the HF cards, the construct-validity report, the gkdv/TF-19 re-run artifacts + robustness note, and any release-doc lines citing measured numbers. Full suite green at the tip.

### Task 19: Push / merge / tag / HF re-publish (each a separate go-ahead)

- [ ] Push the branch (go-ahead). Open the PR. **Wait for CI fully green (conclusion AND every job).**
- [ ] Merge with **`--merge`** (never squash — preserves the two-commit `training_commit`) (go-ahead).
- [ ] Tag the release (go-ahead) — one version, from the merged commit.
- [ ] HF re-publish all 5 GK variants through `scripts/_hub_publish.publish_model_with_card` (ADR-088 card-required) (go-ahead). Verify the Hub file count (~5 per repo, not hundreds).
- [ ] Update `TODO.md` (the TF-60 row → complete; the arc reshape).

---

## Self-review (author, before handing off)

**Spec coverage:** serve-frame-coords (T3/T4), gkdv simplification (T5), ghost-GK both-axes retrain (T1/T2 code + T15 weights), complete Layer-3 arms (T6/T7/T8/T9), probe (T10), corpus driver + construct-validity (T11/T16), the `@e2e` method gate (T11b, spec §10), gkdv+TF-19 re-run (T16), all CI gates (T12), docs + parent-arc (T13), two-commit provenance + `--merge` (T14–T19). **The pre-registered construct-validity anchor is now CONCRETE in the plan** (its own locked section — exact tests, thresholds, and the {Alisson, Neuer} keeper list), for owner ratification at `/review-plan`; T16 only executes it. The `@e2e` method gate has a named task + file (T11b), no longer merely excluded.

**Placeholder scan:** the `...` fills in the test bodies (T2 chirality monkeypatch, T3/T4 serve inverse, T6/T7 write-back/keeper-agnostic, T10 dose) are the sites the implementer completes against the existing fit/fixture helpers in `tests/tracking/test_ghost_gk.py` / `tests/restdefense/`; every such site names the exact assertion and the fixture it draws from. No "add error handling"/"TBD" steps.

**Type consistency:** `build_restdefense_ghost_frames` returns `(cf_frames, provenance, RestDefenseGhostReport)` everywhere; the arm functions return `(table, RestDefenseGhostReport)`; `RD_ARM_SOURCE_VALUES` is the single vocab; the 4 arm column names are fixed in T7 and reused in T8/T12/T13.
