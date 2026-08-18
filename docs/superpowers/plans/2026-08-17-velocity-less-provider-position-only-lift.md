# Velocity-less-provider position-only lift — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the four velocity-requiring pitch-control aggregators produce a zero-velocity positional
model on frames that DECLARE velocity structurally unavailable (SB360), instead of raising/NaN — lifting
the **13 model-relative (Tier-1) columns** (fully-NaN count 40 → **~27 under the PREFERRED D1**, since
the 7 biased Tier-2 estimates stay NaN; → ~20 only if the owner opts Tier-2 in via ACCEPTABLE) — while
keeping the loud raise on a frame merely *missing* `vx`/`vy`.

**Architecture:** One shared edge helper (`zero_velocity_if_unavailable`) decides degrade-vs-raise from
the `speed_source` marker; the four aggregators + `pitch_control_at_target` call it at their pitch-control
seam. The `compute_pitch_control` dispatch stays a pure engine (policy at the edge). The 7 biased Tier-2
kinematic columns' fate is D1 — PREFERRED keeps them NaN, signaled by the existing frame-level
`VelocityRegimeDiagnosis` (no per-row token).

**Tech Stack:** Python, pandas/numpy, existing `silly_kicks.tracking` seams. No new runtime dependency.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-08-17-velocity-less-provider-position-only-lift-design.md`. Read
  it first; this plan implements it. Two decisions (§7 D1, D2) are OPEN — see Task 7 and the note below.
- **No retrain:** the change must be **byte-identical on velocity-bearing frames** (Task 8 proves it). It
  affects only frames missing `vx`/`vy`.
- **Breaking (intended):** `pitch_control_at_target` on a frame missing `vx`/`vy` that is NOT
  declared-unavailable now RAISES instead of silently zero-filling. Enumerate + accept.
- **Policy at the edge:** do NOT put degrade-vs-raise logic in `pitch_control/_dispatch.py`. It stays a
  pure engine that raises when it cannot compute.
- **Detection before fix:** every behavioural task lands its failing test FIRST and is observed failing.
- **No commits.** The user commits once, on their own approval. No commit/version/tag steps here.
- **D1 dependency (RESOLVED after re-review):** Task 7 implements the §7-D1 resolution. **PREFERRED
  (encoded): Tier-2 stays NaN, signaled by the existing frame-level `VelocityRegimeDiagnosis` — NO
  per-row token.** The rejected author-lean and a constant `velocity_unavailable` per-row token are the
  SAME redundant shape `schema.py:310-312` forbids (a per-row column "that would carry a constant");
  `das_source` is legitimate only because it VARIES per row (`computed`/`unlinked`/`unscoreable_frame`/…).
  **ACCEPTABLE (owner may pick): lift Tier-2 too, read-as-positional via the same frame-level
  diagnostic.** Task 7's steps encode PREFERRED; the ACCEPTABLE branch is noted inline.

---

## File structure

- `silly_kicks/tracking/_velocity_availability.py` — ADD `zero_velocity_if_unavailable`; export it.
- `silly_kicks/tracking/features.py` — `pitch_control_at_target` (`:2584`) uses the helper (replaces the
  ad-hoc block at `:2632-2638`).
- `silly_kicks/tracking/_cover_shadows.py` — `compute_threat_pc` (`:768`), `compute_blocking_score` (`:842`).
- `silly_kicks/tracking/_player_influence.py` — `compute_player_influence` (`:39`).
- `silly_kicks/tracking/_space_creation.py` — `compute_space_created` (`:75`).
- `silly_kicks/tracking/_gk_influence.py` — `compute_gk_influence` (`:231`).
- `silly_kicks/tracking/schema.py` — **NOT modified under the PREFERRED D1** (Tier-2 is NaN, no per-row
  token). Touched only under the ACCEPTABLE branch, or if a real varying `…_source` column is added.
- `tests/tracking/test_velocity_availability.py`, `tests/tracking/test_position_only_lift.py` (new).
- `tests/sb360/` — Task 9 audit re-adjudication.
- `docs/superpowers/adrs/ADR-063-*.md`, `CHANGELOG.md`, `CLAUDE.md`, `TODO.md` — Task 10.
- `scripts/validate_sb360_licensed_corpus.py` — Task 11 re-run (no code change).

---

### Task 1: The shared `zero_velocity_if_unavailable` helper

**Files:**
- Modify: `silly_kicks/tracking/_velocity_availability.py`
- Test: `tests/tracking/test_velocity_availability.py`

**Interfaces:**
- Consumes: `velocity_unavailable_by_design(frames)` (already in the module, `:15`).
- Produces: `zero_velocity_if_unavailable(frames: pd.DataFrame) -> pd.DataFrame`.

- [ ] **Step 1: Write the failing test (three cases).**

```python
import numpy as np, pandas as pd
from silly_kicks.tracking._velocity_availability import zero_velocity_if_unavailable
from silly_kicks.tracking.schema import SPEED_SOURCE_UNAVAILABLE

def _frame(speed_source, with_vel):
    d = {"player_id": [1, 2], "team_id": [1, 2], "is_ball": [False, False],
         "x": [10.0, 20.0], "y": [30.0, 40.0], "speed_source": [speed_source, speed_source]}
    if with_vel:
        d["vx"] = [1.0, 2.0]; d["vy"] = [3.0, 4.0]
    return pd.DataFrame(d)

def test_present_velocity_is_returned_unchanged_same_object():
    f = _frame("native", with_vel=True)
    assert zero_velocity_if_unavailable(f) is f  # no copy, no mutation

def test_declared_unavailable_gets_a_zero_velocity_copy():
    f = _frame(SPEED_SOURCE_UNAVAILABLE, with_vel=False)
    out = zero_velocity_if_unavailable(f)
    assert out is not f and "vx" not in f.columns          # input untouched
    assert (out["vx"] == 0.0).all() and (out["vy"] == 0.0).all()

def test_missing_but_NOT_declared_is_returned_unchanged_so_dispatch_raises():
    f = _frame("native", with_vel=False)  # forgot derive_velocities(): a MISTAKE, not a declaration
    out = zero_velocity_if_unavailable(f)
    assert out is f and "vx" not in out.columns  # unchanged -> the dispatch will raise loud
```

- [ ] **Step 2: Run, expect FAIL** (function absent). `python -m pytest tests/tracking/test_velocity_availability.py -q`.

- [ ] **Step 3: Implement.**

```python
def zero_velocity_if_unavailable(frames: pd.DataFrame) -> pd.DataFrame:
    """Prepare frames for a velocity-REQUIRING pitch-control call.

    Returns ``frames`` unchanged when ``vx``/``vy`` are present. When they are ABSENT and the frame set
    DECLARES velocity structurally unavailable (``velocity_unavailable_by_design``), returns a COPY with
    ``vx``=``vy``=0.0 so a velocity-requiring method computes the zero-velocity positional model. When the
    columns are absent and the marker is NOT set, returns ``frames`` unchanged so the dispatch's loud
    raise fires -- a forgotten ``derive_velocities()`` is a MISTAKE, not a declared-velocity-less
    provider. Policy lives here at the edge; the pitch-control engine stays pure.
    """
    if "vx" in frames.columns and "vy" in frames.columns:
        return frames
    if not velocity_unavailable_by_design(frames):
        return frames
    out = frames.copy()
    out["vx"] = 0.0
    out["vy"] = 0.0
    return out
```

Add `zero_velocity_if_unavailable` to the module's exports and to `tracking/__init__.py` beside
`velocity_unavailable_by_design` (if the latter is exported; match its visibility).

- [ ] **Step 4: Run, expect PASS.**

---

### Task 2: Rewire `pitch_control_at_target` (replace ad-hoc zero-fill; the intended tightening)

**Files:**
- Modify: `silly_kicks/tracking/features.py:2632-2638`
- Test: `tests/tracking/test_position_only_lift.py`

**Interfaces:**
- Consumes: `zero_velocity_if_unavailable` (Task 1).

- [ ] **Step 1: Write two failing tests** — behaviour preserved on declared-unavailable, tightened on a mistake.

```python
# On declared-unavailable frames, pitch_control_at_target STILL produces values (unchanged behaviour).
def test_pct_at_target_still_works_on_declared_unavailable(sb_like_actions, sb_like_frames_no_vel):
    s = pitch_control_at_target(sb_like_actions, sb_like_frames_no_vel, method="spearman")
    assert s.notna().mean() > 0.8

# On frames MISSING vx/vy that are NOT declared unavailable, it now RAISES (was silently zero-filling).
def test_pct_at_target_raises_on_forgotten_velocity(sb_like_actions, frames_native_no_vel):
    import pytest
    with pytest.raises(ValueError, match="requires velocity columns"):
        pitch_control_at_target(sb_like_actions, frames_native_no_vel, method="spearman")
```

(Fixtures: `sb_like_frames_no_vel` = frames with `speed_source=SPEED_SOURCE_UNAVAILABLE`, no `vx/vy`;
`frames_native_no_vel` = same positions but `speed_source="native"`, no `vx/vy`. Build both from a small
linked action set; reuse `tests/sb360/_fixture.py` builders if convenient.)

- [ ] **Step 2: Run, expect the tightening test to FAIL** (today it silently zero-fills, so no raise).

- [ ] **Step 3: Implement** — replace `features.py:2632-2638`:

```python
    # Declared-velocity-less providers get the zero-velocity positional model; a frame merely missing
    # vx/vy (forgotten derive_velocities()) still raises in the dispatch. Single-sourced at the edge.
    frames = zero_velocity_if_unavailable(frames)
```

- [ ] **Step 4: Run, expect PASS.**

**Caller sweep — FOLDED IN here, not deferred (review finding).** In-repo callers of `pitch_control_at_target`:
- `atomic/tracking/features.py:1099/1135/1158` — the atomic mirror (velocity handling inherited).
- `calibration/_features.py:221/350` — the calibration path, which **`derive_velocities` FIRST**
  (`_loader_pining.py:492`), so it never passes a velocity-less-undeclared frame.
- `features.py:2725/2751` — the internal `add_pitch_control` / `pitch_control_xfns`.
- `tests/tracking/pitch_control/*` — all velocity-bearing or `method="voronoi"`.

**In-repo blast radius is ZERO** — no caller passes a `vx`/`vy`-missing frame that is NOT declared-unavailable.
`add_pitch_control` is PUBLIC, so the **cross-repo lakehouse-d32 exposure** (a downstream caller passing
such a frame now raises — intended) is stated in the ADR (Task 10).

---

### Task 3: Wire `add_gk_influence` (`compute_gk_influence`)

**Files:**
- Modify: `silly_kicks/tracking/_gk_influence.py:231` (`compute_gk_influence`)
- Test: `tests/tracking/test_position_only_lift.py`

**Interfaces:**
- Consumes: `zero_velocity_if_unavailable`.
- Produces (behaviour): Tier-1 `gk_pitch_control_share_weighted` is finite on declared-unavailable
  frames. The Tier-2 columns (`gk_reachable_area_m2`, `gk_closing_time_*`) compute for free here but are
  **SUPPRESSED to NaN by Task 7** (PREFERRED D1) — so Task 3 asserts only Tier-1; the Tier-2-NaN assertion
  lives in Task 7 (asserting Tier-2 finite here would go RED the moment Task 7 runs).

- [ ] **Step 1: Write the failing both-sides test** (Tier-1 only — see Interfaces).

```python
def test_gk_influence_lifts_tier1_on_declared_unavailable(gk_frame_no_vel, gk_actions):
    out = add_gk_influence(gk_actions, gk_frame_no_vel)
    assert out["gk_pitch_control_share_weighted"].notna().any()   # Tier-1 lifted
    # non-vacuity: the value responds to a real change (move the GK -> share moves).
    moved = gk_frame_no_vel.copy(); moved.loc[moved["is_goalkeeper"], "x"] += 20.0
    assert (add_gk_influence(gk_actions, moved)["gk_pitch_control_share_weighted"]
            != out["gk_pitch_control_share_weighted"]).any()
    # Tier-2 (gk_reachable_area_m2, gk_closing_time_*) is asserted NaN in Task 7, NOT here.

def test_gk_influence_still_raises_on_forgotten_velocity(gk_frame_native_no_vel, gk_actions):
    import pytest
    with pytest.raises(ValueError, match="requires velocity columns"):
        add_gk_influence(gk_actions, gk_frame_native_no_vel)
```

- [ ] **Step 2: Run, expect FAIL** (today: NaN / no raise as specified).

- [ ] **Step 3: Implement** — at the top of `compute_gk_influence` (`:231`), immediately after it receives
  its `frame`/`frames` argument and before the pitch-control-share seam (`:363` `method=`):

```python
    frame = zero_velocity_if_unavailable(frame)   # declared-velocity-less -> zero-velocity model
```

(Name matches the local variable. **The `compute_tti` reachable-area/closing-time path has its OWN loose
`vx`/`vy`→0.0 defaults (`:206-207`, `:335-337`) — a SECOND zero-fill path NOT routed through the helper and
NOT `speed_source`-aware** (review finding). So the Tier-2 columns compute FOR FREE once the share path
stops raising — which under the PREFERRED D1 (Tier-2 stays NaN) is a problem: keeping them NaN needs
EXPLICIT suppression (Task 7), not just "don't lift." For a "single principled seam," either route those
`compute_tti` defaults through the helper or make them `speed_source`-aware; do not leave the second path
unexercised.)

- [ ] **Step 4: Run, expect PASS.**

---

### Task 4: Wire `add_cover_shadows` (`compute_threat_pc` + `compute_blocking_score`)

**Files:**
- Modify: `silly_kicks/tracking/_cover_shadows.py:768` (`compute_threat_pc`), `:842` (`compute_blocking_score`)
- Test: `tests/tracking/test_position_only_lift.py`

**Interfaces:** Consumes `zero_velocity_if_unavailable`. Produces: the 6 `cover_shadows` columns finite on
declared-unavailable frames; raises on forgotten velocity.

- [ ] **Step 1: Write the both-sides test** (mirror Task 3 for `blocking_score` / `n_potential_receivers`
  + the forgotten-velocity raise).
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** — insert `frame = zero_velocity_if_unavailable(frame)` at the top of BOTH
  `compute_threat_pc` and `compute_blocking_score`, before their `compute_pitch_control` calls (`:832`,
  `:948`).
- [ ] **Step 4: Run, expect PASS.**

---

### Task 5: Wire `add_player_influence` (`compute_player_influence`)

**Files:** Modify `silly_kicks/tracking/_player_influence.py:39` (`compute_player_influence`). Test as above.

**Interfaces:** Produces Tier-1 `off_ball_xt_*` (3) finite on declared-unavailable frames; the Tier-2
columns (`reachable_area_*`, `actor_reachable_area_m2`, 4) are **SUPPRESSED to NaN by Task 7** (PREFERRED
D1). Raises on forgotten velocity. Task 5 asserts only Tier-1 (the Tier-2-NaN assertion lives in Task 7).

- [ ] **Step 1: Failing both-sides test** (Tier-1 only — Tier-2 NaN is Task 7).
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** — `frame = zero_velocity_if_unavailable(frame)` at the top of
  `compute_player_influence`, before its `compute_pitch_control(decompose=True)` call (`:112`).
- [ ] **Step 4: Run, expect PASS.**

---

### Task 6: Wire `add_space_creation` (`compute_space_created`)

**Files:** Modify `silly_kicks/tracking/_space_creation.py:75` (`compute_space_created`). Test as above.

**Interfaces:** Produces `space_created_m2`, `space_denied_m2_opponent`, `obso_epv_source` finite on
declared-unavailable (this aggregator currently RAISES in the battery); raises on forgotten velocity.

- [ ] **Step 1: Failing both-sides test** — assert it no longer raises on declared-unavailable AND still
  raises on forgotten velocity.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** — `frame = zero_velocity_if_unavailable(frame)` at the top of
  `compute_space_created`, before its `compute_pitch_control` calls (`:455`, `:471`).
- [ ] **Step 4: Run, expect PASS.**

---

### Task 7: Tier-2 fate — SUPPRESS to NaN; NO per-row token (implements §7-D1, PREFERRED)

**REWRITTEN twice; a per-row token is REJECTED in BOTH forms (re-review R3).** The first author-lean
(`{measured, zero_velocity}`) and a constant `velocity_unavailable` token are the *same* shape
`schema.py:310-312` forbids — "a per-row provenance column that would carry a **constant**." `das_source`
is legitimate ONLY because it VARIES per row (`computed`/`unlinked`/`unscoreable_frame`/`team_unresolved`);
a token that only ever says `velocity_unavailable` conveys nothing the whole-frame-set
`VelocityRegimeDiagnosis` does not already carry. **So PREFERRED emits NO new per-row column:** Tier-2
columns are set to NaN, and the existing frame-level diagnostic (`validate_velocity_regime`, `utils.py:610`)
is the signal. This is the schema-consistent, simplest resolution.

**Files:** `silly_kicks/tracking/_gk_influence.py`, `_player_influence.py`;
`tests/tracking/test_position_only_lift.py`. No `schema.py` change.

**CRITICAL implementation subtlety (the whole reason this task exists):** `compute_tti` auto-fills
`vx`/`vy`=0 (`_gk_influence.py:206-207`, `335-337`), so once Task 3/5 make the share path stop raising,
the Tier-2 columns (`gk_reachable_area_m2`, `gk_closing_time_*`, `reachable_area_*`,
`actor_reachable_area_m2`) compute **FOR FREE**. Keeping them NaN is therefore an ACTIVE step — an
explicit suppression on the `velocity_unavailable_by_design(frames)` branch — not the absence of a lift.
Tasks 3/5's both-sides tests therefore assert Tier-1 finite but Tier-2 **NaN** on the declared-unavailable
fixture (adjust their assertions accordingly).

- [ ] **Step 1: Failing test** — on a declared-unavailable frame, Tier-1 columns are finite while the
  Tier-2 columns are NaN; on a velocity-bearing frame both are finite. **No token asserted.**

```python
def test_gk_influence_suppresses_tier2_but_keeps_tier1_on_declared_unavailable(gk_frame_no_vel, gk_actions):
    out = add_gk_influence(gk_actions, gk_frame_no_vel)
    assert out["gk_pitch_control_share_weighted"].notna().any()   # Tier-1 lifted
    assert not out["gk_reachable_area_m2"].notna().any()          # Tier-2 suppressed to NaN
    # The reason (velocity-unavailable) is a whole-frame-set fact -> validate_velocity_regime, not a column.
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** — on the `velocity_unavailable_by_design(frames)` branch, after computing, set
  the Tier-2 columns to NaN. Emit no per-row provenance column.
- [ ] **Step 4: Run, expect PASS.**

> **ACCEPTABLE branch (owner may pick): lift Tier-2 too.** Delete the suppression, let the Tier-2 columns
> compute, add NO per-row column, and note in the ADR/CHANGELOG that Tier-2 values are read-as-positional
> via `validate_velocity_regime`. Tasks 3/5 then assert Tier-2 finite. The fully-NaN count then reaches
> ~20 instead of ~27.
>
> **A per-row source column is defensible ONLY if it genuinely VARIES** — i.e. a real multi-reason
> `gk_reachable_area_source` in the `das_source` idiom distinguishing `velocity_unavailable` from
> `unlinked` / `no_gk_in_frame` (which also NaN it) — and then the test must exercise that VARIATION, not
> assert a constant. A constant token is rejected (R3). This is a strictly larger change; do not add it
> unless the owner asks for per-row NaN-reason disambiguation.

---

### Task 8: Regression fences — Tier-3 stays NaN AND velocity-bearing invariance

**Files:** Test: `tests/tracking/test_position_only_lift.py`.

**Interfaces:** none (pure guards).

- [ ] **Step 1: Tier-3 stays-NaN test.** On the declared-unavailable fixture, assert the constitutively-
  velocity/multi-frame columns REMAIN NaN (guards against an over-broad fix): `das_diff`, `ghost_gk_x`,
  `xcross_attempt`, `shot_speed`, `actor_speed`, `mean_off_ball_run_speed_pre_window`.
- [ ] **Step 2: Velocity-bearing invariance test (the no-retrain proof).** Build one velocity-BEARING
  fixture; assert each of the four aggregators' output is byte-identical before/after this change. Since
  the code is already applied, pin it by asserting the helper is a no-op on velocity-bearing frames
  (`zero_velocity_if_unavailable(f) is f`) AND run the aggregator once, asserting all target columns are
  finite and unchanged vs a recorded golden built on that same fixture. Both sides, per the codebase rule.
- [ ] **Step 2b: D2 boundary MEASUREMENT — reuse THIS fixture (review requirement, ADR-055).** With the
  same velocity-bearing fixture, compute each candidate column BOTH ways — velocity-aware (real `vx`/`vy`)
  and zero-velocity (`vx`/`vy`=0) — and record the per-column delta. This places the Tier-1/Tier-2 boundary
  EMPIRICALLY (small/none → model-relative Tier-1; large/directional → biased physical Tier-2), replacing
  the spec's asserted table (§7-D2). Land the measured deltas in the ADR (Task 10); if a boundary case
  disagrees with the table, MOVE the column and adjust Tasks 3–7 (and its Tier-2 suppression in Task 7).
- [ ] **Step 3: Run, expect PASS.**

---

### Task 9: ADR-053 SB360 audit re-adjudication

**Files:** `tests/sb360/` (regenerate via `_regenerate.py` + `_adjudicate.py`).

**Interfaces:** none.

- [ ] **Step 1:** Regenerate the audit observations for `add_gk_influence`, `add_cover_shadows`,
  `add_player_influence`, `add_space_creation`. The machine verdicts change **per column**: the **Tier-1**
  columns move from `raises`/`all_nan` toward `differs` (zero-velocity value differs from velocity-aware);
  the **Tier-2** columns stay all-NaN on the velocity-less leg (suppressed by Task 7), so they read
  `differs`/`all_nan`, NOT toward a populated `works`.
- [ ] **Step 2:** Re-adjudicate the human verdicts **per column** — NOT a blanket "toward works":
  **Tier-1 → `differs_by_design`** (the well-defined zero-velocity positional model); **Tier-2 →
  `honest_nan`** (the biased physical estimate is DELIBERATELY withheld on the velocity-less leg, PREFERRED
  D1). Write the per-column rationale citing this spec + ADR-063, and confirm the round-trip is
  byte-identical (`tests/sb360/_regenerate.py`). Re-run the sb360 suite.

---

### Task 10: ADR-063 + docs

**Files:** `docs/superpowers/adrs/ADR-063-velocity-less-position-only-lift.md`, `CHANGELOG.md`, `CLAUDE.md`
(Key conventions bullet), `TODO.md` (Release header).

- [ ] **Step 1:** Write ADR-063 recording: the decision framework, the three tiers, the edge-helper
  architecture (policy-at-edge, engine-pure), the `pitch_control_at_target` tightening (breaking, intended),
  the no-retrain reasoning (velocity-bearing byte-identical), and the D1 resolution the reviewer chose.
- [ ] **Step 2:** CHANGELOG `### Fixed`/`### Changed` entry; a CLAUDE.md durable bullet ("velocity-less
  providers get the zero-velocity model via the ONE `zero_velocity_if_unavailable` edge seam; the
  dispatch stays pure; Tier-3 constitutively-velocity columns stay honest-NaN"); TODO Release header.
- [ ] **Step 3:** Confirm C4 is unchanged (no new aggregator/backend/model — assert the C4 gates still pass).

---

### Task 11: Refresh the SB360 coverage artifact

**Files:** `scripts/validate_sb360_licensed_corpus.py` (no code change; owner-run).

- [ ] **Step 1:** From a CLEAN commit (ADR-037), clear the shard root and re-run the driver; confirm the
  fully-NaN battery-column count drops **40 → ~27 under the PREFERRED D1** (Tier-1's 13 populated; Tier-2's
  7 stay NaN; Tier-3's 20 stay NaN) — or **→ ~20 if the owner picked ACCEPTABLE** (Tier-2 also populated).
  The artifact re-stamps the new commit. This is the citation refresh, its own commit (ADR-052/ADR-037).

---

## Self-review

- **Spec coverage:** Tier 1 (Tasks 3–6 lift the 13), Tier 2 (Tasks 3/5 compute them, **Task 7 SUPPRESSES
  the 7 to NaN under PREFERRED** — no per-row token), Tier 3 (Task 8 fence keeps the 20 NaN); architecture
  (Tasks 1–2 the edge helper + tightening); audit (Task 9); ADR/docs (Task 10); artifact (Task 11). D1/D2
  routed to Task 7 + Task 8 Step 2b + the spec.
- **Placeholder scan:** none — helper code, rewire lines, and test bodies are concrete.
- **Type/name consistency:** `zero_velocity_if_unavailable(frames) -> pd.DataFrame` is used identically in
  Tasks 1–6; `velocity_unavailable_by_design` is the existing predicate; `validate_velocity_regime` is the
  existing public D1-alternative.
- **Known soft spot for the implementer:** the per-frame vs full-frames placement of the helper — the plan
  puts it at each `compute_*` entry (per-frame small copy). If a profiler shows the per-frame copy hurts on
  dense tracking, hoist it to the `add_*` entry once; the behaviour is identical either way.
