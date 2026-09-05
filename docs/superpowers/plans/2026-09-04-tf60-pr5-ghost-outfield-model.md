# TF-60 PR5 — Ghost-Outfield Model — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `GhostOutfieldModel` + `serve_ghost_outfield_positions` — a league-average
rest-defense rearguard-positioning model mirroring `GhostGkModel`, with bundled weights and HF publish,
to feed PR6's outfield counterfactual arm.

**Architecture:** A new trained-model primitive in `silly_kicks/tracking/_ghost_outfield.py` that
mirrors `GhostGkModel` (HGBR boosted-mean x/y ensembles, pickle-free npz+JSON+SHA256 serialization,
chirality + feature-contract fail-closed load-guards, numba leaf walk). It predicts an individual
rearguard player's league-average goal-relative `(x, y)`, keyed by a deterministic lateral-rank slot
within the in-possession team's deepest-`n` defenders. No grid, no KDE density, no label-domain cap.
`default` (faithful) + `position_only` (velocity-dropped) variants, velocity-keyed at the serve seam.

**Tech Stack:** Python, numpy, pandas, scikit-learn (`HistGradientBoostingRegressor`, fit/extract only
— inference imports no sklearn), numba (optional `@njit` leaf kernels), huggingface_hub (publish).

**Spec:** `docs/superpowers/specs/2026-09-03-tf60-pr5-ghost-outfield-model-design.md`
(approved, round-2). The plan argues from the spec; executors read both.

## Global Constraints

Every task's requirements implicitly include this section. Values copied verbatim from the spec.

- **COMMIT CADENCE (overrides the writing-plans template's per-task `git commit`; follows the
  owner-approved PR3 TWO-COMMIT provenance structure).** Phase A makes **NO commit** — all Phase-A tasks
  accumulate on ONE feature branch. Phase B then lands **TWO commits**, because honest weight provenance
  *requires* it: a single commit is impossible — the weights must be trained **FROM an already-committed
  clean code state**, or `training_commit` stamps a dirty/dangling SHA (ADR-063 `_provenance`;
  `require_clean_tree` refuses a dirty tree). The structure, each step its own explicit owner go-ahead:
  - **Commit 1 — code (clean, CI-green):** all Phase-A code + toy-validated tests, **no weights**. This
    is the exact tree the DGX trains from, so `training_commit` = **commit 1's SHA** (honest, clean).
  - *(train on the DGX from commit 1)*
  - **Commit 2 — weights + release:** the real bundled weights + re-pointed tests + version bump + ADR +
    CHANGELOG + model cards + docs.
  - **Merge with `--merge` (MANDATORY, never squash):** a squash orphans commit 1, dangling the
    `training_commit` SHA the weights cite (the PR3 owner ruling — *"I approved the 2-commit strategy
    since it's needed to preserve provenance"*).
  - Then tag / publish / HF-publish, **each a separate explicit owner go-ahead**.
  **Never `git commit`/`git push`/merge/tag without Karsten's explicit approval for that specific
  action.** No half-tested micro-commits — **both** commits are complete, CI-green, coherent states
  (commit 1 is the fully-tested code; commit 2 is the fully-tested weighted release). No standalone doc
  commits: the **design docs** (this spec + plan) ride in **Commit 1** with the code they describe; the
  **release docs** (ADR / CHANGELOG / version bump / model cards) ride in **Commit 2** with the weights.
- **Branch, not worktree.** One feature branch for the cycle, off `main`.
- **Version / PR / ADR numbers are UNASSIGNED** until the Phase-B commit gate — taken from `main` then,
  never claimed earlier.
- **Additive:** no existing feature changes, **no VAEP retrain**, **no `add_*` aggregator** (+0 C4),
  in **no** default xfn list.
- **Trained-model discipline (ADR-011/016/040/044/050/076):** parameters-only, pickle-free, fail-closed
  load (chirality + feature-contract), `stores_training_data=false`. A full save→load→save metadata-SHA
  round-trip is **unachievable** (`save()` recomputes the feature-contract probe); prove byte-identity
  field-level.
- **Orientation:** `GoalMap`/`resolve_defended_goals` (ADR-055), **never team identity**; direction is a
  value not a default (ADR-051-D3). `id_compat` for every id compare/key (ADR-019); NA-never-a-0
  (ADR-027).
- **Leakage rule:** **no feature encodes A's rearguard coordinates or geometry** (the target set);
  `slot_index` (lateral rank) is exempt; **situational velocity only** (ball + opponent-mass), never the
  slot player's own velocity.
- **Feature vector (frozen):** 20 faithful / 16 position_only (§5 of the spec). `position_only` **drops**
  the 4 velocity features, never NaN-fills (ADR-067; the feature contract raises on non-finite).
- **`n_rearguard`** fixed (default 4, tunable serve parameter).
- **Weights provenance:** trained from a **clean, CI-green** commit (no `--allow-dirty` for shipped
  weights; ADR-063 `_provenance`).
- **Corpus visibility:** public-only corpus for bundled weights; fail-closed (`scripts/_corpus.py`).

---

## File Structure

| Path | Responsibility | Phase |
|---|---|---|
| `silly_kicks/tracking/_ghost_outfield.py` | `GhostOutfieldModel`, the `_ghost_outfield` feature extractor, feature-name constants, load-guard blocks, `serve_ghost_outfield_positions`, `from_variant`/`from_hub`. Mirrors `_ghost_gk.py`. | A |
| `silly_kicks/tracking/__init__.py` | Export `GhostOutfieldModel`, `serve_ghost_outfield_positions` (+ the `GhostOutfieldVariant` literal). | A |
| `scripts/train_ghost_outfield.py` | Trainer (mirror `scripts/train_ghost_gk.py`): corpus load, `for_each` sharding, fit, CV metrics, coherence metrics, save, provenance. | A (code) / B (real run) |
| `scripts/publish_ghost_outfield.py` | HF publisher (mirror `scripts/publish_ghost_gk.py`; `_hub_publish.upload_model_only` allowlist). | A (code) / B (real publish) |
| `silly_kicks/tracking/_ghost_outfield_weights/{default,position_only}/` | Bundled weights (npz + metadata.json + SHA256SUMS). | B |
| `docs/huggingface/model-cards/ghost-outfield-v1-model-card.md`, `…-position-only-v1-model-card.md` | Model cards (ship in the release commit). | B |
| `tests/tracking/test_ghost_outfield_model.py` | Extractor, leakage, fit/parity, save/load, load-guard negatives, slot, orientation, coherence. | A |
| `tests/tracking/test_ghost_outfield_variants.py` | Velocity-keyed variant resolution. | A |
| `tests/sb360/…` (registry + a cropped-rearguard test) | SB360 boundary-audit entry + cropped-rearguard honest-NaN. | A |
| `tests/…` registry accounting | public-API-examples + id-scalar registry entries. | A |
| `docs/superpowers/adrs/ADR-nnn-…md`, `CHANGELOG.md`, `CLAUDE.md`, `NOTICE`, `TODO.md`, parent spec §17 | Docs + the §1 recording obligation (reconcile §17 + swap TODO PR4↔PR5). | B |

**Interface anchors (verified in the current tree — re-verify at execution):**
- `select_back_line_players(frames, team_id, defends_x0, *, n=4, adaptive_max_n=5)` →
  `silly_kicks/tracking/_defensive_line.py:21`. Returns deepest-first player rows (all columns
  preserved). `defends_x0` is the **direction bool**, resolved by the caller from the goal map.
- `resolve_defended_goals(frames) -> GoalMap` (ADR-055); `goal_map.get(game_id, period_id, team, allow_guess=True)` returns the defended-goal x (`0.0`/`105.0`/`None`).
- `GhostGkModel` template → `_ghost_gk.py`: `__init__` :1842, `_feature_names` :1888, feature-name
  constants :458, `save` :2183, `load` :2284, `_chirality_block` :1753 / `_feature_contract_block` :1776,
  `serve_ghost_gk_positions` :2795, `from_variant` :2461, `from_hub` :2503, numba leaf via
  `_flatten_trees`/`_FlatTrees`. `IntegrityError` is the CLASS at :262; its raise sites (the tamper
  targets) are :2305 / :2357 / :2433. **Line numbers are approximate — re-grep at execution.**
- Serve contract to mirror: `serve_ghost_gk_positions` returns **goal-relative** `ghost_gr_x/ghost_gr_y`
  + provenance, one row per `(game_id, period_id, frame_id, team)`; **the caller writes back to frame
  coords.** `serve_ghost_outfield_positions` mirrors this (goal-relative, caller write-back), keyed by
  `(game_id, period_id, frame_id, team_id, slot_index)` — this pins spec §17.3.
- `variant_key_for_velocity(frames)` + the `_resolve_*_model_for_frames` resolver pattern (ADR-067).

---

## Phase A — local implementation (NO COMMIT)

All Phase-A tasks build on ONE feature branch and leave `main` untouched. **No `git commit` in Phase A.**

### Task 0: Feature branch + module skeleton + feature-name constants

**Files:**
- Create: `silly_kicks/tracking/_ghost_outfield.py`
- Modify: `silly_kicks/tracking/__init__.py`
- Test: `tests/tracking/test_ghost_outfield_model.py`

**Interfaces:**
- Produces: `GHOST_OUTFIELD_FEATURE_NAMES: list[str]` (20), `GHOST_OUTFIELD_FEATURE_NAMES_POSITION_ONLY`
  (16), `_GHOST_OUTFIELD_VELOCITY_FEATURES: tuple[str, ...]` (4), `GhostOutfieldFeatureSet =
  Literal["faithful", "position_only"]`.

- [ ] **Step 1: Create the feature branch off `main`.**

```bash
git switch main && git switch -c feat/tf60-pr5-ghost-outfield
```
(The untracked spec + plan files move onto the branch automatically.)

- [ ] **Step 2: Write the failing test — feature-name counts and membership.**

```python
# tests/tracking/test_ghost_outfield_model.py
from silly_kicks.tracking._ghost_outfield import (
    GHOST_OUTFIELD_FEATURE_NAMES,
    GHOST_OUTFIELD_FEATURE_NAMES_POSITION_ONLY,
    _GHOST_OUTFIELD_VELOCITY_FEATURES,
)

def test_feature_name_counts_and_partition():
    assert len(GHOST_OUTFIELD_FEATURE_NAMES) == 20
    assert len(_GHOST_OUTFIELD_VELOCITY_FEATURES) == 4
    assert len(GHOST_OUTFIELD_FEATURE_NAMES_POSITION_ONLY) == 16
    # position_only is exactly faithful minus the 4 velocity features, order-preserved.
    assert GHOST_OUTFIELD_FEATURE_NAMES_POSITION_ONLY == [
        f for f in GHOST_OUTFIELD_FEATURE_NAMES if f not in _GHOST_OUTFIELD_VELOCITY_FEATURES
    ]
    # No feature name references A's rearguard geometry (the leakage rule, at the name level).
    banned = ("rearguard", "back_line", "defensive_line", "deepest_defender", "def_compact")
    assert not [f for f in GHOST_OUTFIELD_FEATURE_NAMES if any(b in f for b in banned)]
    # slot_index is present (the one multi-agent feature).
    assert "slot_index" in GHOST_OUTFIELD_FEATURE_NAMES
```

- [ ] **Step 3: Run it — expect ImportError / collection failure.**

Run: `python -m pytest tests/tracking/test_ghost_outfield_model.py::test_feature_name_counts_and_partition -v`
Expected: FAIL (module or names missing).

- [ ] **Step 4: Implement the constants** in `_ghost_outfield.py` (mirror `_ghost_gk.py:458-492`).

```python
from typing import Literal

GHOST_OUTFIELD_FEATURE_NAMES: list[str] = [
    # Ball state (A's ball), goal-relative to A's defended goal
    "ball_x", "ball_y", "ball_vx", "ball_vy", "ball_speed",
    "ball_distance_to_own_goal", "ball_to_own_goal_angle", "ball_in_own_half",
    # Opponent (B) counter-threat geometry — leakage-safe (B is not the target set)
    "opp_in_def_third_count", "opp_deepest_x", "opp_forward_centroid_x",
    "opp_forward_centroid_y", "ball_to_deepest_opp_dist", "opp_forward_centroid_vx",
    # Game context
    "phase", "team_in_possession", "score_diff", "time_seconds", "period_id",
    # Slot (lateral rank 1..n) — the multi-agent feature
    "slot_index",
]
GhostOutfieldFeatureSet = Literal["faithful", "position_only"]
_GHOST_OUTFIELD_VELOCITY_FEATURES = ("ball_vx", "ball_vy", "ball_speed", "opp_forward_centroid_vx")
GHOST_OUTFIELD_FEATURE_NAMES_POSITION_ONLY = [
    f for f in GHOST_OUTFIELD_FEATURE_NAMES if f not in _GHOST_OUTFIELD_VELOCITY_FEATURES
]  # 16
```

- [ ] **Step 5: Add placeholder exports** to `silly_kicks/tracking/__init__.py` (the class + serve seam
  land in later tasks; export the names as they are implemented — do NOT export a name that does not
  exist yet, which would break `import silly_kicks.tracking`). For Task 0, export nothing new yet.

- [ ] **Step 6: Run the test — expect PASS.**

Run: `python -m pytest tests/tracking/test_ghost_outfield_model.py::test_feature_name_counts_and_partition -v`
Expected: PASS. **No commit.**

---

### Task 1: The `_ghost_outfield` feature extractor (with the leakage gate)

**Files:**
- Modify: `silly_kicks/tracking/_ghost_outfield.py`
- Test: `tests/tracking/test_ghost_outfield_model.py`

**Interfaces:**
- Produces: `_extract_all_ghost_outfield_features(frames, *, carrier=None, links=None, feature_set="faithful", goal_map=None, n_rearguard=4) -> pd.DataFrame` — one row per `(game_id, period_id, frame_id, team_id, slot_index)` with: the feature columns; the join/bookkeeping keys including **`player_id`** (the actual player occupying the slot — a KEY column, **NOT** in `GHOST_OUTFIELD_FEATURE_NAMES`, so it is never a model feature; PR6 needs it to match actual↔ghost per slot); and the target `(target_x, target_y)` (goal-relative) for training. Mirrors `_extract_all_ghost_gk_features` (`_ghost_gk.py:900`, the `_extract_all_*` shape) but the per-slot expansion and the leakage-safe feature computation are new.
- Consumes: `select_back_line_players`, `resolve_defended_goals`, `derive_team_in_possession` (TF-5) for the in-possession team A.

Design (record in a module docstring):
1. Resolve `goal_map = resolve_defended_goals(frames)` once (build-once contract, ADR-055).
2. Per frame: resolve the in-possession team `A` (via `derive_team_in_possession` / the frames'
   possession signal); `B` = the other non-ball team. Resolve `defends_x0` for A from `goal_map`.
3. `rearguard = select_back_line_players(frame, A, defends_x0, n=n_rearguard)` → deepest-first rows.
   Re-sort those rows by **lateral coordinate** (goal-relative `y`) → `slot_index = 1..len`.
4. Compute the 20 (or 16) features from **ball + B-forward geometry + context + slot** only. The target
   is slot-K's goal-relative `(x, y)`. **No feature reads any rearguard player's coordinates.**
5. Honest-NaN (never a fabricated 0/zone) when: unresolved geometry (`GoalEndUnresolvedError`),
   non-two-team frame, or `<3` outfield (ADR-027).

- [ ] **Step 1: Write the failing LEAKAGE test (the crux gate).**

```python
import numpy as np
import pandas as pd
from silly_kicks.id_compat import ids_match
from silly_kicks.tracking._ghost_outfield import (
    _extract_all_ghost_outfield_features,
    GHOST_OUTFIELD_FEATURE_NAMES,
)

def _toy_two_team_frame():
    """One frame, 2 teams x (1 GK + 10 outfield) + ball, oriented home-attacks-right,
    team 1 in possession, with team 1's deepest-4 WELL SEPARATED (>=2 m apart in both depth
    and lateral) so a small perturbation cannot change the depth/lateral rank. Reuse the
    ghost-GK fixture builder; set team_in_possession = team 1. Schema: game_id, period_id,
    frame_id, team_id, player_id, is_ball, is_goalkeeper, x, y, vx, vy, team_in_possession,
    team_attacking_direction, score_diff, time_seconds, phase."""
    ...

def test_features_do_not_leak_the_target_players_own_position():
    frame = _toy_two_team_frame()
    base = _extract_all_ghost_outfield_features(frame, feature_set="faithful")
    slot1 = base.sort_values("slot_index").iloc[0]
    target_pid = slot1["player_id"]  # player_id is a KEY column in the output (NOT a feature)
    # Perturb the target's own (x, y) by a SMALL amount -- large enough to move it, small enough
    # to preserve the deepest-4 membership and the lateral rank (the fixture is well-separated),
    # so `after`'s slot-1 is the SAME player and we compare like with like.
    moved = frame.copy()
    pmask = ids_match(moved["player_id"], target_pid) & (~moved["is_ball"].astype(bool))
    moved.loc[pmask, "x"] = moved.loc[pmask, "x"] + 0.5
    moved.loc[pmask, "y"] = moved.loc[pmask, "y"] - 0.5
    after = _extract_all_ghost_outfield_features(moved, feature_set="faithful")
    a_slot1 = after[ids_match(after["player_id"], target_pid)].sort_values("slot_index").iloc[0]
    # The same player still occupies slot 1 (like-with-like, not a re-ranked slot).
    assert int(a_slot1["slot_index"]) == int(slot1["slot_index"])
    # Its FEATURE columns are byte-identical: no feature encodes the target's own coordinates.
    pd.testing.assert_series_equal(
        a_slot1[GHOST_OUTFIELD_FEATURE_NAMES], slot1[GHOST_OUTFIELD_FEATURE_NAMES], check_names=False,
    )
    # ...and the TARGET moved (non-vacuity: the perturbation was real).
    assert not np.isclose(a_slot1["target_x"], slot1["target_x"])
```

- [ ] **Step 2: Run it — expect FAIL** (extractor missing).

Run: `python -m pytest tests/tracking/test_ghost_outfield_model.py::test_features_do_not_leak_the_target_players_own_position -v`
Expected: FAIL.

- [ ] **Step 3: Implement `_extract_all_ghost_outfield_features`** per the design above. Use
  `group_rows` (ADR-068) for any per-frame grouping — no rescan-in-loop. All id compares via `id_compat`.

- [ ] **Step 4: Run the leakage test + a positive extraction test — expect PASS.**

```python
def test_extractor_shape_and_slots():
    frame = _toy_two_team_frame()
    out = _extract_all_ghost_outfield_features(frame, feature_set="faithful", n_rearguard=4)
    assert len(out) == 4  # 4 slots for the one in-possession team
    assert set(out["slot_index"]) == {1, 2, 3, 4}
    assert out[GHOST_OUTFIELD_FEATURE_NAMES].notna().all().all()

def test_position_only_drops_velocity_columns():
    frame = _toy_two_team_frame()
    out = _extract_all_ghost_outfield_features(frame, feature_set="position_only")
    assert not set(_GHOST_OUTFIELD_VELOCITY_FEATURES) & set(out.columns)
```

Run: `python -m pytest tests/tracking/test_ghost_outfield_model.py -k "leak or shape or position_only" -v`
Expected: PASS. **No commit.**

- [ ] **Step 5: Honest-NaN test** — a single-team frame and an unresolvable-geometry frame yield an
  empty result (no fabricated rows), never a raise-through.

```python
def test_extractor_single_team_frame_yields_no_rows():
    frame = _toy_two_team_frame()
    one_team = frame[frame["team_id"].isin([1]) | frame["is_ball"].astype(bool)]
    out = _extract_all_ghost_outfield_features(one_team, feature_set="faithful")
    assert len(out) == 0
```

Run it; expect PASS. **No commit.**

---

### Task 2: `GhostOutfieldModel` — fit / predict_mean / save / load (parity)

**Files:**
- Modify: `silly_kicks/tracking/_ghost_outfield.py`
- Modify: `silly_kicks/tracking/__init__.py` (export `GhostOutfieldModel`, `GhostOutfieldVariant`)
- Test: `tests/tracking/test_ghost_outfield_model.py`

**Interfaces:**
- Produces: `class GhostOutfieldModel` with `__init__(*, n_estimators=500, max_depth=8, verbose=0, feature_set="faithful")` (NO `grid_spec` — mirror `_ghost_gk.py:1842` minus the grid), `_feature_names()`, `fit(frames, *, carrier=None) -> Self`, `predict_mean(features: pd.DataFrame) -> np.ndarray` (shape `(n, 2)`, exact pickle-free boosted mean), `save(dir)`, `classmethod load(dir)`, `from_variant(name)`, `from_hub(repo_id)`. Two ensembles (x, y) reconstructed as `baseline + Σ_trees leaf_value` via the numba leaf walk (reuse `_flatten_trees`/`_FlatTrees`/the `_leaf_values_numba` port from `_ghost_gk`/`_ghost_gk_numba` — do NOT fork the kernel).

- [ ] **Step 1: Write the failing parity test.**

```python
def test_predict_mean_matches_sklearn_within_1e_6():
    frames = _toy_training_frames(n_frames=200)  # many frames -> enough per-slot rows to fit
    model = GhostOutfieldModel(n_estimators=50, max_depth=4).fit(frames)
    feats = _extract_all_ghost_outfield_features(frames, feature_set="faithful")
    pred = model.predict_mean(feats)
    # sklearn reference (kept transiently after fit, like ghost-GK's _sk_reg_x/_y)
    X = feats[model._feature_names()].to_numpy(dtype=np.float64)
    ref = np.column_stack([model._sk_reg_x.predict(X), model._sk_reg_y.predict(X)])
    assert np.max(np.abs(pred - ref)) <= 1e-6
```

- [ ] **Step 2: Run — expect FAIL.** Then **Step 3: implement** `GhostOutfieldModel` mirroring
  `_ghost_gk.py` (fit trains the two HGBRs on `X = feats[_feature_names()]`, `y_x`/`y_y = target_x/_y`;
  store tree-node arrays + baselines; `predict_mean` reconstructs via the numba leaf walk). **Step 4:
  run — expect PASS.**

- [ ] **Step 5: save/load field-level round-trip** (the metadata-SHA full round-trip is unachievable —
  assert field-level, per the Global Constraints).

```python
def test_save_load_roundtrip_field_level(tmp_path):
    frames = _toy_training_frames(n_frames=200)
    model = GhostOutfieldModel(n_estimators=50, max_depth=4).fit(frames)
    model.save(tmp_path)
    loaded = GhostOutfieldModel.load(tmp_path)
    feats = _extract_all_ghost_outfield_features(frames, feature_set="faithful")
    np.testing.assert_array_equal(model.predict_mean(feats), loaded.predict_mean(feats))
    assert loaded.feature_set == "faithful"
    assert loaded._feature_names() == model._feature_names()
```

- [ ] **Step 6: position_only fit + width** — a `feature_set="position_only"` model fits on 16 columns
  and its `_feature_names()` is the 16-list; predicting with a 20-col frame selects the 16 by name.

Run the Task-2 tests; expect PASS. Export `GhostOutfieldModel` + `GhostOutfieldVariant` from
`__init__.py`. **No commit.**

---

### Task 3: Load-guards — chirality + feature-contract + SHA256SUMS (fires-on-tamper)

**Files:**
- Modify: `silly_kicks/tracking/_ghost_outfield.py`
- Test: `tests/tracking/test_ghost_outfield_model.py`

**Interfaces:**
- Produces: `_chirality_block(model) -> dict`, `_feature_contract_block(feature_set="faithful") -> dict`,
  and a module `IntegrityError`. `load()` runs both guards + the SHA256SUMS check and raises
  `IntegrityError` on any mismatch. Mirror `_ghost_gk.py:1753` (`_chirality_block`) / `:1776`
  (`_feature_contract_block`) and the `load()` guard block `:2427-2443`; the declared geometry constants
  are those the outfield extractor actually reads
  (pitch length/width; the `n_rearguard`; the defensive-third boundary if the extractor uses it) —
  **a module-level constant exists iff the module reads it** (ADR-050 enumerator).

- [ ] **Step 1: Write the failing non-vacuity NEGATIVE tests** — the guard must FIRE, not just pass.

```python
import pytest
from silly_kicks.tracking._ghost_outfield import GhostOutfieldModel, IntegrityError

def _bundled_or_toy(tmp_path):
    frames = _toy_training_frames(n_frames=200)
    m = GhostOutfieldModel(n_estimators=50, max_depth=4).fit(frames)
    m.save(tmp_path)
    return tmp_path

def test_load_raises_on_perturbed_weight(tmp_path):
    d = _bundled_or_toy(tmp_path)
    # perturb one npz array so the chirality re-run no longer matches the recorded probe
    import numpy as np
    npz = dict(np.load(d / "model.npz"))
    key = next(iter(npz))
    npz[key] = npz[key] + 1.0
    np.savez(d / "model.npz", **npz)
    with pytest.raises(IntegrityError):
        GhostOutfieldModel.load(d)

def test_load_raises_on_changed_geometry_constant(tmp_path):
    d = _bundled_or_toy(tmp_path)
    import json
    meta = json.loads((d / "metadata.json").read_text())
    meta["pitch_length"] = meta["pitch_length"] + 1.0  # a declared geometry constant
    (d / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(IntegrityError):
        GhostOutfieldModel.load(d)

def test_load_raises_on_tampered_sha256sums(tmp_path):
    d = _bundled_or_toy(tmp_path)
    sums = (d / "SHA256SUMS").read_text().splitlines()
    (d / "SHA256SUMS").write_text("\n".join(["0" * 64 + "  model.npz"] + sums[1:]))
    with pytest.raises(IntegrityError):
        GhostOutfieldModel.load(d)
```

- [ ] **Step 2: Run — expect FAIL** (guards not wired). **Step 3: implement** the two blocks + the
  SHA256SUMS verification in `load()`, mirroring ghost-GK. **Step 4: run — expect PASS.**

- [ ] **Step 5: Geometry-constant enumeration test** — extend
  `tests/tracking/test_geometry_constant_enumeration.py` (or its `_EXEMPT` list) so every geometry
  constant the outfield extractor reads is either in the feature-contract block or exempt-with-reason.
  Run; expect PASS. **No commit.**

---

### Task 4: `serve_ghost_outfield_positions` — orientation, velocity-keyed, honest-NaN

**Files:**
- Modify: `silly_kicks/tracking/_ghost_outfield.py`
- Modify: `silly_kicks/tracking/__init__.py` (export `serve_ghost_outfield_positions`)
- Test: `tests/tracking/test_ghost_outfield_model.py`, `tests/tracking/test_ghost_outfield_variants.py`

**Interfaces:**
- Produces: `serve_ghost_outfield_positions(frames, *, model=None, home_team_id, actions=None, carrier=None, links=None, n_rearguard=4) -> pd.DataFrame` — one row per `(game_id, period_id, frame_id, team_id, slot_index)` with **goal-relative** `ghost_gr_x`, `ghost_gr_y`, and a `ghost_outfield_source` provenance column (closed vocab `{computed, ghost_missing, unlinked, velocity_unavailable}` — the `das_source` string idiom). The **caller writes back** to frame coords (mirrors `serve_ghost_gk_positions`). Velocity-keyed variant auto-select via `_resolve_ghost_outfield_model_for_frames` (ADR-067).

- [ ] **Step 1: Write the failing per-slot + orientation tests.**

```python
def test_serve_yields_distinct_positions_per_slot():
    frames = _toy_training_frames(n_frames=200)
    model = GhostOutfieldModel(n_estimators=50, max_depth=4).fit(frames)
    out = serve_ghost_outfield_positions(frames, model=model, home_team_id=1, n_rearguard=4)
    one = out[(out["game_id"] == out["game_id"].iloc[0]) & (out["frame_id"] == out["frame_id"].iloc[0])]
    assert set(one["slot_index"]) == {1, 2, 3, 4}
    assert one[["ghost_gr_x", "ghost_gr_y"]].drop_duplicates().shape[0] >= 2  # not all identical

def test_serve_is_orientation_invariant_under_frame_mirror():
    # Mirror the FRAMES (point-reflect) and HOLD home_team_id constant; the goal-relative ghost is unchanged.
    frames = _toy_training_frames(n_frames=50)
    model = GhostOutfieldModel(n_estimators=50, max_depth=4).fit(frames)
    a = serve_ghost_outfield_positions(frames, model=model, home_team_id=1)
    mirrored = _point_reflect_frames(frames)  # reflection.reflect_columns, both axes
    b = serve_ghost_outfield_positions(mirrored, model=model, home_team_id=1)
    m = a.merge(b, on=["game_id", "period_id", "frame_id", "team_id", "slot_index"], suffixes=("_a", "_b"))
    assert np.allclose(m["ghost_gr_x_a"], m["ghost_gr_x_b"], atol=1e-6)
    assert np.allclose(m["ghost_gr_y_a"], m["ghost_gr_y_b"], atol=1e-6)
```

- [ ] **Step 2: Run — expect FAIL.** **Step 3: implement** `serve_ghost_outfield_positions`
  (resolve goal map, per frame select+rank rearguard, build features, `predict_mean`, emit goal-relative
  rows + provenance). Route missing velocity through `zero_velocity_if_unavailable` / the ADR-067
  resolver. **Step 4: run — expect PASS.**

- [ ] **Step 5: velocity-keyed variant tests** (`tests/tracking/test_ghost_outfield_variants.py`).

```python
def test_position_only_selected_on_declared_velocityless_frames(): ...   # faithful frames -> default; declared-velocityless -> position_only
def test_missing_position_only_returns_nan_not_default(): ...            # ADR-067 asymmetry
def test_mixed_velocity_availability_raises(): ...                       # velocity_availability_is_mixed
```

Run; expect PASS. **No commit.**

---

### Task 5: Coherence measurement + non-vacuity gate

**Files:**
- Modify: `silly_kicks/tracking/_ghost_outfield.py` (a `ghost_rearguard_coherence(served) -> dict`
  helper computing ordering-fraction + min-pairwise-distance)
- Test: `tests/tracking/test_ghost_outfield_model.py`

- [ ] **Step 1: Write the failing coherence + non-vacuity tests.**

```python
def test_coherence_metric_reports_ordering_and_min_distance():
    served = serve_ghost_outfield_positions(_toy_training_frames(200),
                                            model=GhostOutfieldModel(n_estimators=50, max_depth=4).fit(_toy_training_frames(200)),
                                            home_team_id=1)
    c = ghost_rearguard_coherence(served)
    assert 0.0 <= c["ordering_fraction"] <= 1.0
    assert c["min_pairwise_distance_m"] >= 0.0

def test_coherence_metric_is_not_vacuous():
    served = ...  # as above
    shuffled = _shuffle_slot_positions_within_frame(served)  # deliberately break lateral ordering
    assert ghost_rearguard_coherence(shuffled)["ordering_fraction"] < ghost_rearguard_coherence(served)["ordering_fraction"]
```

- [ ] **Step 2: Run — expect FAIL.** **Step 3: implement** `ghost_rearguard_coherence`. **Step 4: run —
  expect PASS.** The trainer (Task 8) records these in `metrics.json`; the **shape-constraint remedy is
  NOT built** unless the Phase-B measurement shows material incoherence (spec §9). **No commit.**

---

### Task 6: SB360 boundary-audit entry + cropped-rearguard honest-NaN

**Files:**
- Modify: `tests/sb360/_registry.py` (+ `_regenerate.py` round-trip; **back up `_entries/` first** —
  `_regenerate.py` is not idempotent)
- Test: `tests/sb360/…`, `tests/tracking/test_ghost_outfield_model.py`

- [ ] **Step 1: Write the failing cropped-rearguard test** — a rearguard slot whose player is outside
  the SB360 `visible_area` serves **NaN**, never a fabricated position; `ghost_outfield_source` marks it.

```python
def test_cropped_rearguard_slot_is_honest_nan():
    frames, visible_area = _sb360_frames_with_cropped_deep_defender()
    out = serve_ghost_outfield_positions(frames, model=GhostOutfieldModel.from_variant("position_only"), home_team_id=1)
    cropped = out[out["ghost_outfield_source"] != "computed"]
    assert cropped["ghost_gr_x"].isna().all()
```

- [ ] **Step 2: Run — expect FAIL** (or skip if `from_variant("position_only")` needs Phase-B weights;
  use a toy `position_only` model saved in the test in that case). **Step 3: implement** the FOV
  honest-NaN branch in the serve seam. **Step 4: register the SB360 verdict** (`differs_by_design` for
  velocity-invariant / `honest_nan` for cropped) via `_adjudicate.py`; regenerate; assert round-trip
  byte-identical. Run the SB360 gate; expect PASS. **No commit.**

---

### Task 7: Registry accounting (public-API-examples + id-scalar)

**Files:**
- Modify: `tests/…/conftest_id_scalar.py` (if `serve_ghost_outfield_positions` / `from_variant` take an
  id scalar — `home_team_id` is an id scalar), `tests/test_public_api_examples.py` (`_EXAMPLES_DEBT` or a
  real Examples block for `GhostOutfieldModel` + `serve_ghost_outfield_positions`)
- Test: the existing meta-gates.

- [ ] **Step 1: Run the meta-gates to see them fail** (a new public symbol is unaccounted).

Run: `python -m pytest tests/test_public_api_examples.py tests/invariants/test_public_id_scalar_registry.py -v`
Expected: FAIL (unregistered public surface).

- [ ] **Step 2: Register** `serve_ghost_outfield_positions` in the id-scalar registry (it takes
  `home_team_id`) — exercise it on a matched / mismatched-but-equal / float id, OR justify non-invariant
  with a reason; add a real Examples literal-block (or `_EXAMPLES_DEBT` entry) for the two public
  symbols. **Step 3: run — expect PASS.**

- [ ] **Step 4: Confirm NO other gate is triggered** — no `add_*` so no aggregator-liveness / purity /
  mirror-registry / glossary / C4-feature-count entry. Run the full `tests/tracking/` + `tests/sb360/` +
  `tests/invariants/` selection; expect green. **No commit.**

---

### Task 8: Trainer `scripts/train_ghost_outfield.py` (+ @slow smoke)

**Files:**
- Create: `scripts/train_ghost_outfield.py`
- Test: `tests/scripts/test_train_ghost_outfield_smoke.py` (`@pytest.mark.slow`)

**Interfaces:** mirror `scripts/train_ghost_gk.py` — argparse (`--variant {default,position_only}`,
`--data-dir` (toy/real frames source, the real ghost-GK smoke flag), `--out`, corpus selectors,
`--match-ids-json`/`--list-matches`), corpus load via `_loader_pining` /
`_driver.for_each` sharding (shard-token includes `feature_set` — the 4.77.1 stale-shard rule), fit,
CV + per-slot MAE + coherence metrics → `metrics.json`, `save()` with provenance
(`training_commit`/`training_platform`/`sklearn_version`/`corpus_provenance`). **The driver
`require_clean_tree` gate + provenance stamping (ADR-063/`_provenance`).** Uses `_corpus.assert_public_corpus`
for the bundled run.

- [ ] **Step 1: Write the failing `@slow` smoke** — `main()` on a tiny committed corpus produces a
  loadable toy artifact + a `metrics.json` carrying `cv_mae`, per-slot MAE, and the coherence block.

```python
import pytest
@pytest.mark.slow
def test_train_ghost_outfield_smoke(tmp_path):
    from scripts.train_ghost_outfield import main
    # Real ghost-GK smoke pattern (tests/scripts/test_train_ghost_gk_cli.py:41,51): write a tiny toy
    # frames.parquet and pass --data-dir. There is NO --tiny-fixture flag.
    data_dir = tmp_path / "data"; data_dir.mkdir()
    _write_toy_outfield_frames_parquet(data_dir / "frames.parquet")  # a few frames, 2 teams, in-possession set
    out = tmp_path / "out"
    main(["--variant", "default", "--data-dir", str(data_dir), "--out", str(out)])
    from silly_kicks.tracking import GhostOutfieldModel
    GhostOutfieldModel.load(out / "default")  # loads (guards pass)
    import json
    m = json.loads((out / "default" / "metrics.json").read_text())
    assert "cv_mae" in m and "coherence" in m
```

- [ ] **Step 2: Run — expect FAIL.** **Step 3: implement** the trainer (mirror ghost-GK; adapt to the
  outfield extractor + no grid + `feature_set`). **Step 4: run the smoke — expect PASS.**
  **⚠ `--help` executes `main()` on parserless scripts — this script MUST use argparse** (grep
  `add_argument` before ever running `--help`). **No commit.**

---

### Task 9: Publisher `scripts/publish_ghost_outfield.py`

**Files:**
- Create: `scripts/publish_ghost_outfield.py`
- Test: `tests/scripts/test_publish_ghost_outfield.py` (structure/dry-run, no network)

**Interfaces:** mirror `scripts/publish_ghost_gk.py` — `--artifact-dir`, `--repo-id`, uses
`_hub_publish.upload_model_only` (the allowlist leak-guard: uploads ONLY `model.npz` + `metadata.json`
+ `SHA256SUMS` + the card README, **never** the whole folder — the 4.94.0 raw-shard-leak lesson).

- [ ] **Step 1: Write the failing test** — the publisher's file allowlist is exactly the 4 model
  artifacts + README (assert the `upload_model_only` call is given the allowlist, not the folder).

- [ ] **Step 2: Run — expect FAIL.** **Step 3: implement** (mirror ghost-GK publisher). **Step 4: run —
  expect PASS.** **No commit.**

---

### Task 10: Phase-A gate — full suite + lint + scoped pyright (NO COMMIT)

- [ ] **Step 1: Run the full suite** (non-e2e), lint at CI scope, and the scoped pyright.

Run:
```bash
python -m pytest tests/ -m "not e2e" -q
python -m ruff check silly_kicks/ tests/ scripts/
python -m ruff format --check silly_kicks/ tests/ scripts/
python -m pyright   # bare; and confirm tests/tracking/test_pyright_clean_tracking_namespace passes
```
Expected: all green on the toy-fit model. **STOP. No commit.** Phase A ends here — hand off for the
independent plan review's Phase-A verification if requested, otherwise proceed to Phase B only on the
owner's go-ahead.

---

## Phase B — DGX weights + release (TWO owner-approved commits: code → weights)

Phase B follows the PR3 two-commit provenance structure (see Global Constraints). Nothing here happens
without a per-step owner go-ahead; the session never launches DGX training itself.

### Task 11: Commit 1 (code, clean) → train real weights on the DGX

- [ ] **Step 1: Phase-A green + clean tree.** Confirm Task 10 is green. **Present the full Commit-1 diff +
  file list** — all Phase-A code + tests + the trainer/publisher + the spec/plan (now tracked). **No
  weights, no model cards, no version bump / ADR / CHANGELOG that reference the weights** (those depend on
  the trained artifacts and belong to Commit 2). **STOP — wait for explicit owner approval of Commit 1.**
- [ ] **Step 2:** On approval → **Commit 1** (code only). **STOP.** On the next go-ahead → push; **watch
  CI green.** Commit 1's SHA is the `training_commit` the weights will cite.
- [ ] **Step 3: Owner runs the DGX train FROM commit 1's clean tree** (venv `sk-phaseb-venv`, corpus
  `sk_stageB_448/ghost_cache`, 179 matches). `train_ghost_outfield.py --variant default` and
  `--variant position_only`, full-match frames, public corpus. The trainer's `require_clean_tree` gate
  passes (tree is clean at commit 1) → `training_commit` = commit 1's SHA, `tree_dirty=false`. Record CV
  euclidean MAE (overall + per-provider + per-slot), the coherence block, parity vs sklearn (exact,
  ≤1e-6). **GS is fully usable** (the 27.5 m clamp is keeper-only).
- [ ] **Step 4:** Copy the two artifact dirs into
  `silly_kicks/tracking/_ghost_outfield_weights/{default,position_only}/`; add `.gitattributes` `** binary`.

### Task 12: Stage Commit 2 — bundle, re-point gates, cards, docs, C4 (full suite green)

- [ ] **Step 1: `from_variant("default")` / `from_variant("position_only")`** now load the bundled
  weights; re-point the toy-based tests at the real weights (direction-not-magnitude assertions;
  golden/chirality/feature-contract on the bundled artifacts). Regenerate any per-artifact goldens.
- [ ] **Step 2: Coherence decision** — read the recorded coherence block. If materially incoherent, build
  the spec §9 remedy (monotone re-sort / min-separation prior) and document it in the ADR; else record
  "coherent, no constraint."
- [ ] **Step 3: Model cards** — write `ghost-outfield-v1-model-card.md` +
  `ghost-outfield-position-only-v1-model-card.md` (mirror the ghost-GK sweeper cards: variants table,
  metrics, training data with the public-corpus + GS-usable note, usage, limitations, references).
  **These ship in Commit 2** (the PR3 dropped-cards lesson).
- [ ] **Step 4: Docs** — ADR-nnn (next-free), CHANGELOG (PR-Snnn, next-free), version bump
  (`silly_kicks/_version.py` single-source, ADR-079), CLAUDE.md contract bullet, `NOTICE` entry
  (Le et al. 2017 ghosting; DEFCON-GNN comparator). **The §1 recording obligation:** reconcile the parent
  spec §17 arc table to PR5-before-PR4 and swap the TODO TF-60 row's PR4↔PR5 entries (+ mark PR5 shipped).
  Update `TODO.md` per the release convention.
- [ ] **Step 5: C4** — verify the completeness gate; +0 aggregator, but touch the `tracking` container's
  model list if the DSL enumerates models; regenerate `architecture.html` with pinned `dot` if the DSL
  changed.
- [ ] **Step 6: Full suite + lint + scoped pyright green** on the bundled weights.

### Task 13: Commit 2 + release (each a SEPARATE owner go-ahead)

- [ ] **Step 1: Present the full Commit-2 diff + file list** (weights + re-pointed tests + version + ADR +
  CHANGELOG + cards + docs) and the CV/coherence numbers. **STOP — wait for explicit owner approval of
  Commit 2.**
- [ ] **Step 2:** On approval → **Commit 2** (weights + cards + docs). **STOP.**
- [ ] **Step 3:** On the next go-ahead → push; **watch CI until green (every job — conclusion AND each
  job).** **STOP** — never merge until CI green.
- [ ] **Step 4:** On the next go-ahead → **merge with `--merge` (MANDATORY — never squash: a squash
  orphans commit 1 and dangles the `training_commit` SHA the weights cite).** **STOP.**
- [ ] **Step 5:** On the next go-ahead → tag + push `v<version>`; verify the publish run + PyPI. **STOP.**
- [ ] **Step 6:** On the next go-ahead → HF publish (both repos + cards) via `publish_ghost_outfield.py`;
  verify the Hub file count (~5, not the whole folder). Round-trip verify.

---

## Self-Review

**1. Spec coverage.** Every spec section maps to a task: §2 artifact set → Tasks 0–13; §3 target/slot →
Tasks 1, 4; §3 fixed `n` → Task 1/4 (`n_rearguard`); §4 architecture → Task 2; §5 frozen feature vector
→ Tasks 0, 1 (+ leakage gate); §6 corpus → Tasks 8, 11; §7 variants → Tasks 2, 4; §8 SB360/FOV → Task 6;
§9 coherence → Task 5 (+ Phase-B decision, Task 12); §10 surface/C4 → Tasks 4, 7, 12; §11 bundling/HF →
Tasks 9, 11, 13; §12 discipline/attribution → Tasks 2, 3, 12; §13 testing → all Task test steps; §14
phasing → the Phase A/B split; §16 D0–D10 → the Global Constraints. §1 recording obligation → Task 12
Step 4. No spec requirement is unmapped.

**2. Placeholder scan.** The `...` markers in Task-1 `_toy_two_team_frame` and Task-4/5 helpers are
FIXTURE builders the executor writes against the canonical `tests/tracking/` frame schema — flagged as
"reuse the ghost-GK fixture builder," not silent TODOs. The machinery tasks (2, 3) deliberately say
"mirror `_ghost_gk.py:<lines>`" because the spec's core decision is to mirror that model; reproducing
~2000 lines here would risk transcription drift. Every behavioral gate has concrete test code.

**3. Type consistency.** `feature_set: Literal["faithful","position_only"]`, `n_rearguard: int`,
serve returns goal-relative `ghost_gr_x`/`ghost_gr_y` + `ghost_outfield_source` consistently across
Tasks 4/6/13. `GhostOutfieldModel.__init__` has NO `grid_spec` (Task 2) and no `predict_density` (spec
§4) — consistent everywhere. `select_back_line_players` is called with `defends_x0` (bool from GoalMap),
never `home_team_id` — consistent with the verified signature.

**Deviation from the writing-plans template, recorded:** the per-task `git commit` step is REMOVED
throughout and replaced by the Phase-A-no-commit / Phase-B **two-commit** cadence (Commit 1 = clean code
→ DGX train from it → Commit 2 = weights + release; merge `--merge` to preserve the `training_commit`
reference), per this repo's iron commit-discipline rules **and** the provenance requirement that bundled
weights be trained from an already-committed clean state (the owner-approved PR3 structure). This is
intentional and is the correct behaviour for this
codebase.
