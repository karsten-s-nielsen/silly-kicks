# Position-only model variants + velocity-keyed auto-select — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `XShotOccurrenceModel` / `XCrossAttemptModel` / `GhostGkModel` produce values on
velocity-less StatsBomb-360 freeze-frames by shipping position-only model variants, auto-selected at
the serve seam by the declared velocity marker, with a provenance column and the `model=` override
retained.

**Architecture:** Hexagonal — models stay pure scorers; variant selection is policy at the edge
(mirrors `variant_key_for_provider` + `_resolve_completion_for_frames`, keyed on velocity-availability
instead of provider). Position-only variants are the same models re-fit on the full-tracking corpus
with velocity features dropped (not NaN-filled). Design: `docs/superpowers/specs/2026-08-23-position-only-model-variants-and-velocity-auto-select-design.md`.

**Tech Stack:** Python, pandas/numpy, xgboost (xShot/xCross boosters), sklearn HGBR (ghost),
parameters-only npz/booster-JSON artifacts + JSON metadata + SHA256SUMS, pytest (TDD), DGX for
training/coverage.

## Global Constraints

- **Engineering bar:** SOLID, hexagonal, TDD (red-first), no shortcuts on type/security/tests.
- **No pickle; parameters-only; fail-closed load** (chirality + feature-contract + SHA256).
- **No default-xfn-list change** — none of the three models is in `tracking_default_xfns`; keep it so.
- **`extended` is a reserved slot** — extend the `feature_set` literal; never delete `"extended"`; its
  `NotImplementedError` stays intact.
- **Fallback direction:** a missing `position_only` variant falls back to **NaN, never to default**
  (default is invalid on velocity-less frames). Opposite of the completion template.
- **The ADR-054 raise stays in `compute_*`** (undeclared missing `vx`/`vy`), not in the resolver.
- **Six `feature_set` touch-points per model** must all be handled: the 3 guard sites
  (`extract_*_features`, `*Model.__init__`, `prepare_*_training_data`), the serve-time extract call,
  `_chirality_block(model)` (reads `model.feature_set`), `_feature_contract_block(feature_set=...)`
  (new parameter).
- **Provenance sequencing (spec §9), expressed as PHASE ORDERING, not commit steps:** Phase A
  (library) must be pushed before Phase B (weights) trains against it; Phase B weights must be
  committed before Phase C (coverage) runs against them. The three commits are the user's, made with
  explicit go-aheads — **this plan contains no commit steps.**
- **No version number is grabbed until Commit 3.** No version / ADR / PR-Snnn numbers appear in this
  plan; Commits 1 and 2 touch **none** of the five version sites and keep `CHANGELOG.md` under
  `## [Unreleased]`. The version (a minor bump) is chosen and applied **only in Commit 3**, per the
  standing rule that a number is decided when we are ready for the commit, not before. See the Commit
  plan below.
- **Verification net before any commit:** full suite (`-m "not e2e" --benchmark-skip`) on `.venv`
  (3.10) AND `.venv312` (pandas-3 span, ADR-057), ruff check+format at CI scope, whole-branch pyright.

---

## Commit plan & merge strategy

**Three commits on a single feature branch, merged with a NON-SQUASH merge commit.** The non-squash is
**required, not stylistic**: the weights record `training_commit` and the coverage records `run_commit`,
both of which must reference **live** SHAs. A squash collapses the branch into one commit and orphans
those references (dirty provenance — spec §9). A `--merge` preserves all three commits with their
original SHAs in main's history, so the references resolve. **Consequence: main's tip is a merge commit
with TWO parents** — a deliberate deviation from the usual squash-default "verify main tip has ONE
parent" check; do not flag it as an error.

| Commit | Phase / tasks | Provenance role | Must be pushed before |
|---|---|---|---|
| **1 — Library** | Phase A (Tasks 1–9) **+ the spec & plan & ADR docs themselves** | its SHA = the weights' `training_commit` | Phase B training |
| **2 — Weights** | Phase B (Task 10): bundled PO weights + comparability + weight-dependent tests | its SHA = the coverage's `run_commit` | Phase C coverage |
| **3 — Coverage + release** | Phase C (Task 11): refreshed coverage + narrative + **the version bump** | the release commit | the PR merge |

**Docs live in Commit 1 (the untracked-is-dirty rule).** `_provenance.py` counts untracked files as
dirty, so the spec + plan + ADR + all doc edits go in Commit 1 — otherwise Phase B's artifact drivers
(`require_clean_tree`) refuse on the still-untracked spec/plan.

**NO version number is grabbed until Commit 3.** Commits 1 and 2 leave ALL FIVE version sites
untouched (`pyproject.toml`, `silly_kicks/__init__.py`, `uv.lock`, `TODO.md`, `CHANGELOG.md`) and keep
the CHANGELOG entry under `## [Unreleased]`. The version number — a **minor** bump (new feature +
behavior change) — is chosen and applied **only in Commit 3**, when the release is assembled. The tag
`vX.Y.0` is applied to the **merge commit** (whose tree carries the Commit-3 version) after the
non-squash merge.

**Merge / tag:** one PR carrying all three commits → CI green on the final state →
`gh pr merge --merge --admin --delete-branch` (NON-squash) → tag `vX.Y.0` on the merge commit →
publish → verify PyPI. **Each commit takes a SEPARATE go-ahead**; Commits 2 and 3 run on the DGX with
owner data and are gated accordingly.

---

## File structure

**Phase A — library (Tasks 1–9):**
- `silly_kicks/tracking/_xshot_occurrence.py` — Literal + PO constants + PO extractor branch + 3 guard
  lifts + serve-seam restructure + `_resolve_xshot_model_for_frames` + load-block threading.
- `silly_kicks/tracking/_xcross_attempt.py` — same shape (single-column drop).
- `silly_kicks/tracking/_ghost_gk.py` — same, plus the asymmetric single-frame PO extractor +
  `GhostGkVariant` Literal extension.
- `silly_kicks/tracking/_velocity_availability.py` — `variant_key_for_velocity` (Layer A) +
  `velocity_availability_is_mixed` predicate.
- `silly_kicks/tracking/features.py` — `add_xshot_occurrence` / `add_xcross_attempt` / `add_ghost_gk`
  provenance columns.
- `silly_kicks/feature_glossary.py` — 3 provenance-column records.
- `scripts/train_xshot_occurrence.py`, `scripts/train_xcross_attempt.py`, `scripts/train_ghost_gk.py`
  — `--feature-set position_only`.
- `scripts/compare_position_only_variants.py` — new comparability script (delta velocity-vs-PO).
- `docs/superpowers/adrs/ADR-XXX-*.md` (number at commit time), `CHANGELOG.md`, `TODO.md`, `CLAUDE.md`,
  `docs/PRIVATE_CONSUMERS.md`.
- Tests: `tests/tracking/test_velocity_variant_resolution.py` (new),
  `tests/tracking/test_position_only_extractors.py` (new), plus additions to the three models'
  existing integration test files and the ADR-033 purity registry.

**Phase B — weights (Task 10):** `silly_kicks/tracking/_{xshot,xcross,ghost_gk}_weights/position_only/`
(model file + `metadata.json` + `SHA256SUMS` + `metrics.json`), `docs/research/position_only_variants/`
(comparability artifact), weight-dependent tests.

**Phase C — coverage (Task 11):** `docs/research/sb360_licensed_coverage/` (refreshed
parquet+md+manifest), final `CHANGELOG.md`/version-site bump.

---

## Phase A — Library

### Task 1: Position-only extractors + guard lifts — xShot & xCross

**Files:**
- Modify: `silly_kicks/tracking/_xshot_occurrence.py` (`:125` Literal, `:127-133` constants,
  `:154-182` extractor, `:413-414` init, `:734-735` prepare)
- Modify: `silly_kicks/tracking/_xcross_attempt.py` (`:39` Literal, `:49-68` constants, extractor,
  init, prepare — mirror sites)
- Test: `tests/tracking/test_position_only_extractors.py` (new)

**Interfaces:**
- Produces: `XSHOT_FEATURE_NAMES_POSITION_ONLY` (26), `XCROSS_FEATURE_NAMES_POSITION_ONLY` (15);
  `extract_xshot_features(..., feature_set="position_only")` and the xCross analogue return the
  reduced-length vector; `feature_set` literal now `Literal["faithful", "extended", "position_only"]`.

- [ ] **Step 1: Write the failing test** (`tests/tracking/test_position_only_extractors.py`)

```python
import numpy as np
from silly_kicks.tracking._xshot_occurrence import (
    XSHOT_FEATURE_NAMES_FAITHFUL, XSHOT_FEATURE_NAMES_POSITION_ONLY, extract_xshot_features,
)
from tests.tracking.test_xshot_occurrence import _synthetic_match_frames  # existing scoring fixture

def test_position_only_names_drop_only_speed():
    assert "speed" in XSHOT_FEATURE_NAMES_FAITHFUL
    assert "speed" not in XSHOT_FEATURE_NAMES_POSITION_ONLY
    assert XSHOT_FEATURE_NAMES_POSITION_ONLY == [f for f in XSHOT_FEATURE_NAMES_FAITHFUL if f != "speed"]
    assert len(XSHOT_FEATURE_NAMES_POSITION_ONLY) == 26

def test_position_only_extractor_shape_and_finite():
    frame = _synthetic_match_frames(n_frames=1)
    # pick one group as the extractor expects (a single frame's rows); reuse the module's grouping
    row = extract_xshot_features(frame, gk_team_id="B", goal_x=105.0, feature_set="position_only")
    assert row.shape == (1, 26)
    assert np.isfinite(row.iloc[0].to_numpy(dtype=float)).all()
```

- [ ] **Step 2: Run it to confirm it fails**
Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_position_only_extractors.py -q`
Expected: FAIL — `XSHOT_FEATURE_NAMES_POSITION_ONLY` undefined / `feature_set='position_only'` raises
`NotImplementedError`.

- [ ] **Step 3: Implement**
  - `_xshot_occurrence.py:125`: `XShotFeatureSet = Literal["faithful", "extended", "position_only"]`.
  - After `:132`: `XSHOT_FEATURE_NAMES_POSITION_ONLY = [f for f in XSHOT_FEATURE_NAMES_FAITHFUL if f != "speed"]`.
  - `extract_xshot_features` (`:177-182`): change the guard to raise only on `"extended"`; when
    `feature_set == "position_only"`, assemble the vector from `XSHOT_FEATURE_NAMES_POSITION_ONLY`
    (skip the `speed` element — do NOT compute/insert `hypot(bvx,bvy)`). Keep `"faithful"` byte-identical.
  - Mirror for xCross (`ball_speed`).

- [ ] **Step 4: Run to confirm pass** — same command, plus the xCross analogue tests.

- [ ] **Step 5: Guard-lift tests — `position_only` ACCEPTED (the red-first driver) + `extended` still raises (regression)**

```python
import pytest
from silly_kicks.tracking._xshot_occurrence import extract_xshot_features, XShotOccurrenceModel, prepare_xshot_training_data
from tests.tracking.test_xshot_occurrence import _synthetic_match_frames

def test_position_only_accepted_at_init_and_extract():
    # RED drivers for the init/extract lift: these currently raise NotImplementedError (!= "faithful",
    # :413 / :177) and MUST NOT after the lift.
    m = XShotOccurrenceModel(feature_set="position_only")   # must NOT raise
    assert m.feature_set == "position_only"
    frame = _synthetic_match_frames(n_frames=1)
    extract_xshot_features(frame, gk_team_id="B", goal_x=105.0, feature_set="position_only")  # no raise

def test_extended_still_raises_at_all_three_sites():
    # Regression guard (National Park Principle): the lift must reject ONLY "extended". This passes both
    # BEFORE and AFTER the lift ("extended" != "faithful" either way) -- its job is to ensure the lift
    # does not WIDEN to accept extended, NOT to be the red-first driver (that is the test above, V2).
    frame = _synthetic_match_frames(n_frames=1)
    with pytest.raises(NotImplementedError):
        extract_xshot_features(frame, gk_team_id="B", goal_x=105.0, feature_set="extended")
    with pytest.raises(NotImplementedError):
        XShotOccurrenceModel(feature_set="extended")  # init guard (NO `booster` kwarg; :412)
    with pytest.raises(NotImplementedError):
        # shots (positional) + home_team_id (kw) REQUIRED (:683-688); empty shots so BINDING succeeds
        # and the feature-set guard (:734, first stmt after docstring) is what raises (P2).
        prepare_xshot_training_data(frame, frame.iloc[:0], home_team_id=1, feature_set="extended")
```

- [ ] **Step 6: Run — `test_position_only_accepted_*` goes RED** (init/extract raise on
  `!= "faithful"`). Lift the init and prepare guards (and the extract guard from Step 3 if not already)
  to raise only on `"extended"`; re-run both tests to pass. **Do not** assert
  `prepare_*(..., feature_set="position_only")` "does not raise at all" here — an empty shots frame may
  raise a DIFFERENT downstream error; `prepare_*`'s `position_only` happy-path is covered by Task 8's
  slow train-smoke (V2).

### Task 2: Position-only ghost extractor — the asymmetric single-frame path

**Files:**
- Modify: `silly_kicks/tracking/_ghost_gk.py` (`:345-372` constants, `:2167` `GhostGkVariant`,
  extractor + `_extract_all_ghost_gk_features` `:773-1006`, init + prepare guard sites)
- Test: `tests/tracking/test_position_only_extractors.py` (add ghost cases)

**Interfaces:**
- Produces: `GHOST_GK_FEATURE_NAMES_POSITION_ONLY` (21); a single-frame-capable position-only ghost
  extraction path that needs no `prev_state`.

- [ ] **Step 1: Failing test — position-only ghost on a single freeze frame with NO predecessor**

```python
import numpy as np
from silly_kicks.tracking._ghost_gk import (
    GHOST_GK_FEATURE_NAMES, GHOST_GK_FEATURE_NAMES_POSITION_ONLY, extract_ghost_gk_features,
)
from tests.tracking.test_ghost_gk_integration import _single_freeze_frame  # build if absent (one frame, GK+outfield+ball, no prior)

_VELOCITY = {"ball_vx", "ball_vy", "ball_speed", "defensive_line_speed", "defending_centroid_vx"}

def test_ghost_position_only_names_drop_the_five_velocity_features():
    assert set(GHOST_GK_FEATURE_NAMES) - set(GHOST_GK_FEATURE_NAMES_POSITION_ONLY) == _VELOCITY
    assert len(GHOST_GK_FEATURE_NAMES_POSITION_ONLY) == 21

def test_ghost_position_only_extracts_on_a_lone_freeze_frame():
    frame = _single_freeze_frame()  # no prior frame exists
    row = extract_ghost_gk_features(
        frame, gk_team_id="B", goal_x=105.0, score_diff=0.0, phase=0, ball_carrier_team_id="A",
        feature_set="position_only",
    )
    vec = row.iloc[0].to_numpy(dtype=float)
    assert vec.shape == (21,)
    assert np.isfinite(vec).all()  # NO NaN from an absent predecessor (the single-frame obligation)
```

- [ ] **Step 2: Run, confirm fail.**
- [ ] **Step 3: Implement** — extend `GhostGkVariant` Literal to include `"position_only"`; add
  `GHOST_GK_FEATURE_NAMES_POSITION_ONLY`; add the `feature_set` param to `extract_ghost_gk_features`
  and a `position_only` branch that assembles the 21 positional features and **does not enter the
  `prev_state`/`prev_timestamps` cross-frame path** (the two temporal derivatives are simply absent);
  lift the init + prepare guards to raise only on `"extended"`.
- [ ] **Step 4: Run to pass.**
- [ ] **Step 5: Add the faithful-unchanged regression** — `extract_ghost_gk_features(..., feature_set="faithful")` byte-identical to pre-change on a two-frame fixture (with predecessor). Run to pass.

### Task 3: Layer A — `variant_key_for_velocity` + mixed-availability predicate

**Files:**
- Modify: `silly_kicks/tracking/_velocity_availability.py`
- Test: `tests/tracking/test_velocity_variant_resolution.py` (new)

**Interfaces:**
- Produces: `variant_key_for_velocity(frames) -> str` (`"position_only"` | `"default"`, pure, no
  IO/raise); `velocity_availability_is_mixed(frames) -> bool` (True iff `0 < n_marked < len(frames)`).

- [ ] **Step 1: Failing test**

```python
import pandas as pd
from silly_kicks.tracking._velocity_availability import variant_key_for_velocity, velocity_availability_is_mixed
from silly_kicks.tracking.schema import SPEED_SOURCE_UNAVAILABLE

def _frames(marks):  # marks: list of speed_source values
    return pd.DataFrame({"speed_source": marks, "vx": [0.0]*len(marks), "vy": [0.0]*len(marks)})

def test_key_two_way():
    assert variant_key_for_velocity(_frames([SPEED_SOURCE_UNAVAILABLE]*3)) == "position_only"
    assert variant_key_for_velocity(_frames(["native"]*3)) == "default"

def test_empty_is_default():
    assert variant_key_for_velocity(pd.DataFrame({"speed_source": []})) == "default"

def test_mixed_predicate():
    assert velocity_availability_is_mixed(_frames([SPEED_SOURCE_UNAVAILABLE, "native"])) is True
    assert velocity_availability_is_mixed(_frames([SPEED_SOURCE_UNAVAILABLE]*2)) is False
    assert velocity_availability_is_mixed(_frames(["native"]*2)) is False
    assert velocity_availability_is_mixed(pd.DataFrame({"speed_source": []})) is False
```

- [ ] **Step 2: Run, confirm fail.**
- [ ] **Step 3: Implement** both functions in `_velocity_availability.py` (single-sourced beside
  `velocity_unavailable_by_design`). `variant_key_for_velocity` = `"position_only" if
  velocity_unavailable_by_design(frames) else "default"`. `velocity_availability_is_mixed` counts the
  marker: `n = (frames["speed_source"] == SPEED_SOURCE_UNAVAILABLE).sum()`; True iff
  `0 < n < len(frames)` and `len(frames) > 0` (and the column exists).
- [ ] **Step 4: Run to pass.**

### Task 4: Layer B — per-model `_resolve_*_model_for_frames`

**Files:**
- Modify: `_xshot_occurrence.py`, `_xcross_attempt.py`, `_ghost_gk.py` (add the resolver near each
  `_resolve_model`)
- Test: `tests/tracking/test_velocity_variant_resolution.py` (add)

**Interfaces:**
- Consumes: `variant_key_for_velocity` (Task 3), `from_variant` (existing).
- Produces: `_resolve_xshot_model_for_frames(frames, model) -> (model_or_None, variant_key)` and the
  two analogues. `None` model signals the seam to emit NaN.

- [ ] **Step 1: Failing test** (uses a locally-fit tiny position-only model + monkeypatch of
  `from_variant`, so it does not depend on Phase-B bundled weights)

```python
import pandas as pd, pytest
import silly_kicks.tracking._xshot_occurrence as xs
from silly_kicks.tracking.schema import SPEED_SOURCE_UNAVAILABLE

def _declared(n=2): return pd.DataFrame({"speed_source": [SPEED_SOURCE_UNAVAILABLE]*n})
def _velocity(n=2): return pd.DataFrame({"speed_source": ["native"]*n, "vx":[1.0]*n, "vy":[1.0]*n})

def test_override_wins(tiny_position_only_xshot):
    # P3: use a REAL model instance, not object(). _resolve_model rejects non-(Model|str|None) by
    # design (:641), so the override must be a genuine instance; the key is derived from the model.
    m, key = xs._resolve_xshot_model_for_frames(_declared(), model=tiny_position_only_xshot)
    assert m is tiny_position_only_xshot
    assert key == "custom"  # V1: override ALWAYS maps to "custom" (closed set; never shipped_variant)

def test_declared_bundled_resolves_position_only(monkeypatch):
    stub = object()
    monkeypatch.setattr(xs.XShotOccurrenceModel, "from_variant", classmethod(lambda cls, v: stub))
    m, key = xs._resolve_xshot_model_for_frames(_declared(), model=None)
    assert key == "position_only" and m is stub

def test_declared_unbundled_falls_back_to_NaN_not_default(monkeypatch):
    def _boom(cls, v):
        if v == "position_only": raise FileNotFoundError
        raise AssertionError("must NOT fall back to default on velocity-less frames")
    monkeypatch.setattr(xs.XShotOccurrenceModel, "from_variant", classmethod(_boom))
    with pytest.warns(UserWarning):
        m, key = xs._resolve_xshot_model_for_frames(_declared(), model=None)
    assert m is None  # NaN sentinel; NEVER the default model

def test_velocity_bearing_resolves_default(monkeypatch):
    stub = object()
    monkeypatch.setattr(xs.XShotOccurrenceModel, "from_variant", classmethod(lambda cls, v: stub))
    m, key = xs._resolve_xshot_model_for_frames(_velocity(), model=None)
    assert key == "default" and m is stub
```

- [ ] **Step 2: Run, confirm fail.**
- [ ] **Step 3: Implement** `_resolve_xshot_model_for_frames(frames, model)`:
  1. `if model is not None: return _resolve_model(model), "custom"` — the override key is **always
     `"custom"`** (V1: the closed set is `{default, position_only, custom}`; do NOT read
     `shipped_variant`, which would leak open values like `"public"`). `_resolve_model` returns a Model
     instance as-is (`:637`) and maps a str → `from_variant` (`:639`); garbage input raises `TypeError`
     (`:641`), which is correct (P3).
  2. `key = variant_key_for_velocity(frames)`.
  3. `try: return XShotOccurrenceModel.from_variant(key), key` `except FileNotFoundError:`
     `if key == "position_only": warn(...); return None, key` `else: raise`.
  Mirror in xCross/ghost. **Do not** add a default fallback for the `position_only`-missing case (the
  NaN-not-default asymmetry — the default is invalid on velocity-less frames).
- [ ] **Step 4: Run to pass.**

### Task 5: Serve-seam restructure (the 3 seams)

**Files:**
- Modify: `_xshot_occurrence.py:854-912`, `_xcross_attempt.py` (compute seam),
  `_ghost_gk.py:2360-2484` (`_serve_positions_core`)
- Test: `tests/tracking/test_velocity_variant_resolution.py` (add behavioral)

**Interfaces:**
- Consumes: `_resolve_*_model_for_frames` (Task 4), `velocity_availability_is_mixed` (Task 3),
  the PO extractor (Tasks 1–2).

**Fixture note (D3):** `baseline_xshot_output` is the `xshot_occurrence` Series from the CURRENT
`compute_xshot_occurrence` on `_synthetic_match_frames(n_frames=5)`, captured/frozen **before** this
task's implement step (while the seam is still the 4.90.0 version), so the byte-identical regression
compares the velocity-bearing path against the true pre-change output. Add it to the fixtures list.

- [ ] **Step 1: Failing behavioral tests** (locally-fit tiny PO model via monkeypatch so no bundled
  artifact is needed yet)

```python
import numpy as np, pandas as pd, pytest
import silly_kicks.tracking._xshot_occurrence as xs
from silly_kicks.tracking.schema import SPEED_SOURCE_UNAVAILABLE
from tests.tracking.test_xshot_occurrence import _synthetic_match_frames

def _declare_unavailable(frames):
    out = frames.drop(columns=[c for c in ("vx","vy") if c in frames.columns]).copy()
    out["speed"] = np.nan; out["speed_source"] = SPEED_SOURCE_UNAVAILABLE
    return out

def test_declared_frame_scores_via_position_only(monkeypatch, tiny_position_only_xshot):
    monkeypatch.setattr(xs.XShotOccurrenceModel, "from_variant",
                        classmethod(lambda cls, v: tiny_position_only_xshot))
    frames = _declare_unavailable(_synthetic_match_frames(n_frames=5))
    out = xs.compute_xshot_occurrence(frames, model=None, home_team_id=1)
    assert out["xshot_occurrence"].notna().any()  # VALUE, not the 4.90.0 NaN

def test_mixed_availability_raises():
    frames = _synthetic_match_frames(n_frames=5)  # velocity-bearing
    frames.loc[frames.index[:2], "speed_source"] = SPEED_SOURCE_UNAVAILABLE  # partial-mark → mixed
    with pytest.raises(ValueError, match="mixed"):
        xs.compute_xshot_occurrence(frames, model=None, home_team_id=1)

def test_undeclared_missing_velocity_still_raises():
    frames = _synthetic_match_frames(n_frames=5).drop(columns=["vx","vy"])  # no marker
    with pytest.raises(ValueError, match="derive_velocities"):
        xs.compute_xshot_occurrence(frames, model=None, home_team_id=1)

def test_velocity_bearing_byte_identical(baseline_xshot_output):
    frames = _synthetic_match_frames(n_frames=5)
    out = xs.compute_xshot_occurrence(frames, model=None, home_team_id=1)
    pd.testing.assert_series_equal(out["xshot_occurrence"], baseline_xshot_output)  # unchanged
```

- [ ] **Step 2: Run, confirm fail** (mixed→currently no raise; declared→currently NaN).
- [ ] **Step 3: Implement the restructure at each seam**, in this order at `compute_xshot_occurrence`:
  1. `if velocity_availability_is_mixed(frames): raise ValueError("mixed velocity-availability ...")`
     — **before** both existing prongs.
  2. Keep the ADR-054 undeclared-missing raise (`:869`) verbatim.
  3. Replace the declared→`return out` (`:863-868`) with: `m, variant = _resolve_xshot_model_for_frames(frames, model)`;
     `if m is None: return out` (unbundled NaN fallback); else proceed to score with `m`.
  4. For the velocity-bearing path, also route through `_resolve_*_model_for_frames` (returns default).
  5. Thread the serve-time extract (`:912`): `extract_xshot_features(grp, ..., feature_set=m.feature_set)`.
  Mirror in xCross and in ghost's `_serve_positions_core` (there the current
  `_GhostVelocityUnavailableError` raise on the declared marker becomes a resolve-and-serve; the two
  entry-point catches degrade only on the None/unbundled sentinel).
- [ ] **Step 4: Run to pass.**
- [ ] **Step 5: gkdv non-vacuity** — `serve_ghost_gk_positions` on a declared freeze frame yields
  rows (was zero); add the rung, run to pass.

### Task 6: Load-guard `feature_set` threading

**Files:**
- Modify: `_xshot_occurrence.py:335-370` (`_chirality_block`, `_feature_contract_block`, `:504`
  `save()` call), `_xcross_attempt.py`, `_ghost_gk.py` (mirror blocks)
- Test: `tests/tracking/test_position_only_extractors.py` (add round-trip)

**Interfaces:**
- `_chirality_block(model)` reads `model.feature_set`; `_feature_contract_block(feature_set)` gains a
  parameter; `save()` passes `self.feature_set`.

- [ ] **Step 1: Failing test** — a locally-fit tiny position-only model round-trips `save()`+`load()`
  with chirality + feature-contract passing.

```python
def test_position_only_roundtrip_passes_guards(tmp_path, tiny_position_only_xshot):
    tiny_position_only_xshot.save(tmp_path)
    reloaded = tiny_position_only_xshot.load(tmp_path)  # must NOT raise chirality / contract
    assert reloaded.feature_set == "position_only"
```

- [ ] **Step 2: Run, confirm fail** (chirality/contract blocks run the faithful extractor → shape
  mismatch / wrong fingerprint).
- [ ] **Step 3: Implement** — `_chirality_block`: `extract_xshot_features(frame, ..., feature_set=model.feature_set)`.
  `_feature_contract_block(feature_set)`: add the param, `extract_xshot_features(contract_probe_frame(), ..., feature_set=feature_set)`;
  `save()` (`:504`) calls `_feature_contract_block(self.feature_set)`. Mirror across the three models.
- [ ] **Step 4: Run to pass; add a faithful-unchanged round-trip regression.**

### Task 7: Provenance columns + glossary + purity

**Files:**
- Modify: `silly_kicks/tracking/features.py` (`add_xshot_occurrence`, `add_xcross_attempt`,
  `add_ghost_gk`), `silly_kicks/feature_glossary.py`
- Test: `tests/tracking/test_velocity_variant_resolution.py` (add), `tests/test_add_star_purity.py`
  (register), the feature_glossary coverage gate (auto)

**Interfaces:**
- Produces columns `xshot_occurrence_variant` / `xcross_attempt_variant` / `ghost_gk_variant` over
  `{default, position_only, custom}`.

- [ ] **Step 1: Failing test** — `add_xshot_occurrence` on a declared frame emits the column
  `= "position_only"`; on a velocity frame `= "default"`; the `*_xfns` path does NOT emit it (and
  emits no internal column either).
- [ ] **Step 2: Run, confirm fail.**
- [ ] **Step 3: Implement — the STAMPED-COLUMN mechanism, NEVER a tuple return (D1).** `compute_*`
  returns a DataFrame consumed by THREE callers (`add_*`, the `*_xfns` path, direct callers), so
  changing its return signature is a Hyrum break. Instead: `compute_*` stamps an INTERNAL column (e.g.
  `__variant`) on its output; `add_*` PROMOTES it to the public `xshot_occurrence_variant`; the
  `*_xfns` path STRIPS the internal column (VAEP stays numeric, D5). Add "xfns drops the internal
  column" as an explicit sub-step and assert it.
- [ ] **Step 4: Verify the string column survives `@nan_safe_enrichment` (D2).** All three `add_*` are
  decorated (`_xshot_occurrence.py:943`); assert the `*_variant` string column passes through the
  decorator intact (value + dtype), not coerced to numeric or dropped.
- [ ] **Step 5: Add `feature_glossary` records** (3) + register the 3 columns in `PURITY_ENTRIES`
  with ≥2 variants each; run the purity + glossary-coverage gates to pass.

### Task 8: Trainer `--feature-set position_only` + comparability script

**Files:**
- Modify: `scripts/train_xshot_occurrence.py`, `scripts/train_xcross_attempt.py`,
  `scripts/train_ghost_gk.py`
- Create: `scripts/compare_position_only_variants.py`
- Test: additions to `tests/tracking/test_{xshot_occurrence,xcross_attempt,ghost_gk}_integration.py`
  (`@pytest.mark.slow`), `tests/scripts/test_compare_position_only.py`

**Interfaces:**
- Each trainer's CLI offers `--feature-set {faithful,position_only}` (default `faithful`) — **NOT
  `extended`** (D4): `extended` still raises in `prepare_*_training_data`, so exposing it as a CLI
  choice is a foot-gun that crashes on first reach. The library `feature_set` Literal keeps `extended`
  for forward-compat; the trainer CLI simply does not expose it until it is implemented. The chosen
  value is threaded to `prepare_*_training_data` + the extractor.
- `compare_position_only_variants.py`: pure delta computation (held-out AUC/Brier/MAE
  velocity-vs-PO), unit-testable; provenance-stamped (`require_clean_tree`, ADR-037).

- [ ] **Step 1: Failing unit test** for the comparability delta function (feed two fixed metric
  dicts, assert the delta table); failing slow-smoke asserting the `--feature-set position_only` CLI
  runs and writes acceptance-all-true.
- [ ] **Step 2: Run, confirm fail.**
- [ ] **Step 3: Implement** the flag wiring (mirror the existing `--data-dir`/`--providers` paths) and
  the comparability script (delta + `docs/research/position_only_variants/` writer with provenance).
- [ ] **Step 4: Run the unit test to pass; run the slow smoke on a tiny `--data-dir` fixture to pass.**

### Task 9: Docs (library-side)

**Files:**
- Create: `docs/superpowers/adrs/ADR-XXX-*.md` (number at commit time) — velocity-keyed variant
  auto-select + position-only family + the NaN-not-default asymmetry + mixed-set raise.
- Modify: `CHANGELOG.md` (`## [Unreleased]`), `TODO.md`, `CLAUDE.md` (durable contract),
  `docs/PRIVATE_CONSUMERS.md` (behavior change + new columns + numeric-path traceability rule).

- [ ] **Step 1:** Draft the ADR from the spec's Decision + the two review rounds' corrections.
- [ ] **Step 2:** Add the CHANGELOG `[Unreleased]` entry (retrain/Hyrum trigger; velocity frames
  byte-identical), the CLAUDE.md contract bullet, the PRIVATE_CONSUMERS note. No version number.
- [ ] **Step 3:** Run the full Phase-A verification net (see Global Constraints) and confirm green.

> **End of Phase A.** The library is complete and testable with locally-fit/stub models. The user
> commits + pushes Phase A (commit 1) on their explicit go-ahead before Phase B. **Commit 1 touches NO
> version site; CHANGELOG stays `[Unreleased]` — no version number is grabbed here.**

---

## Phase B — Weights (runs on the DGX against the pushed Phase-A tree)

### Task 10: Train, bundle, and ship the position-only weights (DECISION GATE)

**Files:**
- Create: `silly_kicks/tracking/_{xshot,xcross,ghost_gk}_weights/position_only/` (model file +
  `metadata.json` incl. `feature_set`, `feature_names`, chirality + PO feature-contract blocks,
  `shipped_variant`, `reproducibility ∈ {"public","restricted"}` + reason; `SHA256SUMS`; `metrics.json`)
- Create: `docs/research/position_only_variants/` (comparability artifact)
- Test: `tests/tracking/test_position_only_bundled.py` (load-guard + behavioral on the REAL artifacts;
  the metadata `reproducibility` gate)

**Interfaces:**
- Consumes: the pushed Phase-A library; the DGX pining corpus + owner token; `ruthless-efficiency>=0.4.0`.

- [ ] **Step 1:** On the DGX (clean 4.90.x checkout of the pushed Phase-A tree, non-calib venv,
  `set -a && . ~/.pining_owner.env && set +a`), run each trainer with `--feature-set position_only`
  `--providers idsse,skillcorner` (public arm). Verify each `metrics.json` acceptance passes.
- [ ] **Step 2:** Run `scripts/compare_position_only_variants.py` → `docs/research/position_only_variants/`.
- [ ] **Step 3 (the decision gate):** For each model, if the public PO variant clears its acceptance
  set → bundle it with `shipped_variant="public"`, `reproducibility="public"`. If (esp. ghost) it
  fails → train the full-corpus PO variant, bundle with `reproducibility="restricted"` + a reason.
  Copy artifacts into the `position_only/` dirs; regenerate `SHA256SUMS`.
- [ ] **Step 4: Bundled-artifact tests** — the real `position_only` artifacts pass chirality +
  feature-contract at `load()`; `add_*` on a declared frame emits a finite value + provenance
  `= "position_only"`; a `restricted` bundled default carries the documented caveat (machine-checkable).
- [ ] **Step 5:** Run the full verification net (both venvs, ruff, pyright) green.

> **End of Phase B.** User commits Phase B (commit 2) — the weights record `training_commit` = the
> live Phase-A commit — on explicit go-ahead before Phase C. **Commit 2 touches NO version site — still
> no version number.**

---

## Phase C — Coverage refresh (runs on the DGX against the committed Phase-B weights)

### Task 11: Refresh the licensed-corpus coverage + finalize

**Files:**
- Modify: `docs/research/sb360_licensed_coverage/` (parquet + md + manifest via the driver + render)
- Modify: `CHANGELOG.md` (finalize), the version sites (at commit time)

- [ ] **Step 1:** On the DGX (clean checkout of the committed Phase-B tree), run
  `scripts/validate_sb360_licensed_corpus.py --out docs/research/sb360_licensed_coverage --tag all`
  then `scripts/render_sb360_licensed_coverage.py`.
- [ ] **Step 2:** Confirm `add_xshot_occurrence` / `add_xcross_attempt` / `add_ghost_gk` move OUT of
  the fully-NaN set (they now populate via position_only), and the space_creation fix persists.
  Update the "40 → 31 fully-NaN lift" narrative to the new count.
- [ ] **Step 3:** Copy the refreshed parquet/md/manifest into the repo; verify `run_commit` = the
  Phase-B commit (clean provenance).
- [ ] **Step 4: Grab the version number — Commit 3 ONLY.** This is the FIRST point the number is
  chosen: the minor bump across `pyproject.toml`, `silly_kicks/__init__.py`, `uv.lock`, `TODO.md`, and
  `CHANGELOG.md` (`[Unreleased]` → `[X.Y.0]`). Commits 1–2 left all five untouched. Do not pre-select
  the number earlier.

> **End of Phase C.** Commit 3 is where the version number is **first grabbed** — the minor bump across
> all five sites + `CHANGELOG` `[Unreleased]` → `[X.Y.0]` — chosen at this moment, not before. User
> commits Phase C (commit 3) on explicit go-ahead, then one PR carrying all three commits → CI green →
> **`gh pr merge --merge` (NON-squash, preserves the live SHAs)** → tag `vX.Y.0` on the merge commit →
> publish → verify PyPI.

---

## Self-review (author)

- **Spec coverage:** D1→Tasks 1,2,6; D2/D3/M4→Task 10 (decision gate) + Task 6; D4 (Layer A/B, mixed
  raise, NaN-not-default, seam restructure)→Tasks 3,4,5; D5 (provenance)→Task 7; D6 (validation)→Tasks
  1–8,10; #4 coverage→Task 11; docs/ADR→Task 9. All spec sections map to a task.
- **Placeholder scan:** ADR number is intentionally `ADR-XXX` (assigned at commit time, per the
  standing rule) — the only deliberate placeholder; no TBD/"handle edge cases"/vague steps.
- **Type/name consistency:** `_resolve_*_model_for_frames` returns `(model_or_None, variant_key)`
  consistently across Tasks 4/5/7; `feature_set` literal is `["faithful","extended","position_only"]`
  everywhere; provenance vocabulary `{default,position_only,custom}` consistent.
- **No commit steps:** the three commits are the user's (phase-boundary notes only); provenance
  sequencing is expressed as Phase A→B→C ordering.
- **Fixtures to build:** `tiny_position_only_xshot` (a locally-fit small PO model instance — buildable
  after Task 1 lifts the `__init__` guard; used in Tasks 4–6), `_single_freeze_frame` (ghost
  single-frame, Task 2), and `baseline_xshot_output` (pre-change `xshot_occurrence` golden, frozen
  before Task 5's implement step, D3). Each is named where first used.
- **RED-snippet discipline:** every embedded `pytest.raises`/behavioral snippet must be run against the
  real signatures first and FAIL for the intended reason — a snippet that errors at call-binding
  (wrong kwargs) is not a valid red. P1/P2/P3 (round 1) were exactly this class; the executor validates
  each snippet's binding before implementing.

## Revision log

**Plan review round 1** (independent critic; ran the embedded snippets against the tree). All findings
verified and folded in:
- **P1** — Task 1 test: `XShotOccurrenceModel(feature_set="extended")` (no `booster` kwarg, `:412`).
- **P2** — Task 1 test: `prepare_xshot_training_data(frame, frame.iloc[:0], home_team_id=1, feature_set="extended")` (required `shots` + `home_team_id`, `:683-688`).
- **P3** — Task 4 `test_override_wins` uses a real `tiny_position_only_xshot` (not `object()`, which
  `_resolve_model` rejects at `:641`); impl reads the key from the RESOLVED model.
- **D1** — Task 7 commits to the stamped-internal-column mechanism (never a tuple return; 3 callers),
  with an explicit "xfns strips the internal column" sub-step.
- **D2** — Task 7 adds a step verifying the string `*_variant` column survives `@nan_safe_enrichment`.
- **D3** — Task 5 documents `baseline_xshot_output` (captured pre-change) + added to the fixtures list.
- **D4** — Task 8 CLI offers `{faithful, position_only}` only; `extended` is not exposed (it raises).

**Plan review round 2** (same critic; re-ran the revised snippets; verdict: structurally ready, two
residual). Both verified and folded in:
- **V1** (closed-vocabulary violation, touches spec D5/D4) — the override branch now maps to `"custom"`
  **unconditionally** and does NOT read `shipped_variant` (which would leak `"public"` outside the
  closed `{default, position_only, custom}` set). Fixed at Task 4 Step 3.1, `test_override_wins`
  (`assert key == "custom"`), and spec D5/D4-Layer-B.
- **V2** (red-first sequencing) — Task 1 Step 5 adds `test_position_only_accepted_at_init_and_extract`
  as the actual red-first driver for the init/extract lift; the `extended`-still-raises test is kept as
  a regression guard (it passes before and after the lift, so it is not the driver). Step 6 does not
  assert `prepare_*(position_only)` no-raise (empty shots may raise a different error; deferred to the
  Task 8 slow smoke).
