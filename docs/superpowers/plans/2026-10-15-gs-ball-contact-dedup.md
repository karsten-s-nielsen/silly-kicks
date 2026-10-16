# GS ball-contact trajectory-dedup — implementation plan

> **For agentic workers:** execute task-by-task. Steps use checkbox syntax. **COMMIT DISCIPLINE OVERRIDE:** ignore the generic "commit after every task" cadence — this plan has exactly **two** commits, both owner-gated (Tasks 6 and 7). Tasks 1–5 build one coherent change with NO commits.

**Goal:** fix GS ball-contact under-detection (the duplicate-frame velocity sawtooth) by deduping the raw ball trajectory before break detection, provider-agnostic, in a pure reusable helper.

**Architecture:** a pure `trajectory_dedup_mask` collapses consecutive-duplicate ball positions; `detect_ball_contacts` applies it per period before the RDP / velocity-break / |Δv| legs. Frame-count windows (`dvw`/`w`) become time-based so they span the intended seconds on the now-sparser sampling.

**Tech stack:** numpy, pandas; pytest. No new runtime deps. No model (pure primitive, no retrain).

**Spec:** `docs/superpowers/specs/2026-10-15-gs-ball-contact-dedup-design.md` (read it; this plan argues from it).

## Global constraints (from the spec — every task inherits these)

- **Provider-agnostic** dedup (unconditional); no provider branch.
- Helper is **PRIVATE** (`silly_kicks/tracking/preprocess/_dedup.py`, the ADR-120 gap-handling sibling; numpy-only leaf, no silly_kicks imports → no cycle). **No public `__all__` addition** → no mirror/c4/glossary/public-api gate changes.
- Contact **schema unchanged**; **byte-identical NOT claimed** (values move on every provider — GS massively, IDSSE/SkillCorner slightly). Intended correctness improvement.
- `detect_ball_contacts` stays **pure** (no input mutation; `PURITY_ENTRIES`).
- **No version number** claimed until commit-prep; **ADR-125 provisional** (confirm free at cycle end).
- **Two commits, both owner-gated**, non-squash. Spec/plan/ADR/docs stay uncommitted until the cycle-end commit.

---

### Task 1: `trajectory_dedup_mask` helper (pure)

**Files:**
- Create: `silly_kicks/tracking/preprocess/_dedup.py`
- Test: `tests/tracking/preprocess/test_dedup.py`

**Interfaces — Produces:**
```python
def trajectory_dedup_mask(x: np.ndarray, y: np.ndarray, *, tol_m: float = 1e-6) -> np.ndarray:
    """Keep-mask collapsing runs of consecutive-duplicate positions.

    Keep index 0; keep i>0 iff hypot(x[i]-x[i-1], y[i]-y[i-1]) > tol_m (compared to the
    IMMEDIATELY preceding point, so A->B->A is preserved). `tol_m` default 1e-6 m: coords are
    float32 storage (ADR-106), true duplicates differ by exactly 0; a sub-um tol removes only
    non-moving repeats, never real motion. Pure; returns a bool ndarray of len(x).
    """
```

- [ ] **Step 1: Failing tests** — `tests/tracking/preprocess/test_dedup.py`:

```python
import numpy as np
from silly_kicks.tracking.preprocess._dedup import trajectory_dedup_mask

def test_collapses_consecutive_runs():
    x = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 2.0]); y = np.zeros(6)
    assert trajectory_dedup_mask(x, y).tolist() == [True, False, False, True, False, True]

def test_preserves_a_b_a_oscillation():
    x = np.array([0.0, 1.0, 0.0]); y = np.zeros(3)
    assert trajectory_dedup_mask(x, y).tolist() == [True, True, True]

def test_all_duplicates_collapse_to_one():
    x = np.full(5, 3.0); y = np.full(5, 4.0)
    assert trajectory_dedup_mask(x, y).tolist() == [True, False, False, False, False]

def test_float32_exact_duplicates():
    x = np.array([1.1, 1.1, 1.2], dtype=np.float32).astype(np.float64); y = np.zeros(3)
    assert trajectory_dedup_mask(x, y).tolist() == [True, False, True]

def test_tol_boundary_keeps_motion_above_tol():
    x = np.array([0.0, 2e-6, 4e-6]); y = np.zeros(3)  # 2e-6 m steps > 1e-6 tol
    assert trajectory_dedup_mask(x, y).tolist() == [True, True, True]

def test_single_point():
    assert trajectory_dedup_mask(np.array([0.0]), np.array([0.0])).tolist() == [True]
```

- [ ] **Step 2: Run, verify fail** — `pytest tests/tracking/preprocess/test_dedup.py -v` → ImportError/fail.
- [ ] **Step 3: Implement** `_dedup.py`:

```python
"""Collapse consecutive-duplicate trajectory samples (ADR-120 gap-handling sibling: dups, not gaps)."""
from __future__ import annotations
import numpy as np

def trajectory_dedup_mask(x: np.ndarray, y: np.ndarray, *, tol_m: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    n = len(x)
    keep = np.ones(n, dtype=bool)
    if n < 2:
        return keep
    moved = np.hypot(np.diff(x), np.diff(y)) > tol_m
    keep[1:] = moved
    return keep
```

- [ ] **Step 4: Run, verify pass.**

---

### Task 2: time-based |Δv| and detection-quality windows (GSBC/GSD-SPEC-01)

**Why first (before wiring dedup):** dedup makes the per-period arrays sparse + non-uniform-dt. The current frame-count windows (`dvw = round(0.12/frame_period)`, `w = round(0.2/frame_period)`) then span the wrong seconds → silently corrupt |Δv| and detection-quality on every provider. Convert both to time-based (searchsorted on `tt`). `_velocity_breaks` needs NO change (it already derives velocity from `dt = diff(t)`).

**Files:**
- Modify: `silly_kicks/tracking/_ball_contact.py` (`_velocity_change_at`; the detection-quality window in `detect_ball_contacts`)
- Test: `tests/tracking/test_ball_contact.py` (new window-semantics test)

**Interfaces — Produces:**
```python
def _time_window(tt: np.ndarray, gp: int, span_s: float) -> tuple[int, int]:
    """[lo, hi) index range of samples within +/- span_s seconds of tt[gp] (searchsorted)."""

def _velocity_change_at(bx, by, tt, gp: int, span_s: float) -> float:  # window:int -> span_s:float
```

- [ ] **Step 1: Red-green failing tests** — assert the window spans the intended SECONDS on a sparse non-uniform-dt sampling, with teeth: the OLD frame-count semantics measurably VIOLATE the seconds-bound on the same fixture. Route the teeth through the WINDOW time-extent (not |Δv|, which is window-invariant for constant velocity). In `tests/tracking/test_ball_contact.py`:

```python
import numpy as np
from silly_kicks.tracking import _ball_contact as bc

# dense cluster around the touch, then a sparse tail (the post-dedup shape); 25 fps nominal
TT_SPARSE = np.array([0.00, 0.04, 0.08, 0.12, 0.72, 1.32, 1.92])

def test_time_window_bounds_each_side_within_span():
    # for every interior gp, the kept window extends no more than span_s on EITHER side
    span = 0.12
    for gp in range(1, len(TT_SPARSE) - 1):
        lo, hi = bc._time_window(TT_SPARSE, gp, span)
        assert TT_SPARSE[gp] - TT_SPARSE[lo] <= span + 1e-9
        assert TT_SPARSE[hi - 1] - TT_SPARSE[gp] <= span + 1e-9

def test_old_frame_count_window_VIOLATES_the_seconds_bound():  # RED-GREEN teeth (ADR-056)
    # the retired semantics: +/- round(span/frame_period) SAMPLES. At 25 fps, span 0.12 s -> +/-3
    # samples. At gp=3 (t=0.12) that reaches sample 6 (t=1.92) -> 1.80 s on the after side >> 0.12 s.
    span, frame_period = 0.12, 1.0 / 25.0
    k = max(1, round(span / frame_period))                     # = 3
    gp = 3
    i1_old = min(len(TT_SPARSE) - 1, gp + k)                   # old frame-count after-edge
    assert TT_SPARSE[i1_old] - TT_SPARSE[gp] > span            # OLD span is WRONG (1.80 s > 0.12 s)
    lo, hi = bc._time_window(TT_SPARSE, gp, span)              # NEW is correct
    assert TT_SPARSE[hi - 1] - TT_SPARSE[gp] <= span + 1e-9

def test_velocity_change_detects_turn_over_the_time_window():
    # +x 10 m/s before the touch, +y 10 m/s after: |dv| = hypot(10,10) ~= 14.14 over +/-0.12 s
    tt = np.array([0.00, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24])
    bx = np.where(tt <= 0.12, 10.0 * tt, 10.0 * 0.12); by = np.where(tt <= 0.12, 0.0, 10.0 * (tt - 0.12))
    dv = bc._velocity_change_at(bx, by, tt, 3, 0.12)
    assert 13.0 < dv < 15.0

def test_time_based_window_matches_old_on_dense_uniform_sampling():
    # regression: on uniform 25 fps the time-based window == the old frame-count window (good path unchanged)
    tt = np.arange(0, 1.0, 0.04); bx = 3.0 * tt; by = np.zeros_like(tt)
    lo, hi = bc._time_window(tt, 10, 0.12)
    assert (tt[10] - tt[lo]) <= 0.12 + 1e-9 and (tt[hi - 1] - tt[10]) <= 0.12 + 1e-9
    assert hi - lo == 2 * 3 + 1   # +/-3 samples at 25 fps, == the old frame-count window
```

- [ ] **Step 2: Run, verify fail** (old `_velocity_change_at(..., window:int)` signature / frame-count behavior).
- [ ] **Step 3: Implement** — add `_time_window`; change `_velocity_change_at` to take `span_s`:

```python
def _time_window(tt, gp, span_s):
    lo = int(np.searchsorted(tt, tt[gp] - span_s, side="left"))
    hi = int(np.searchsorted(tt, tt[gp] + span_s, side="right"))
    return lo, hi

def _velocity_change_at(bx, by, tt, gp, span_s):
    lo, hi = _time_window(tt, gp, span_s)
    i0, i1 = lo, hi - 1
    if gp - i0 < 1 or i1 - gp < 1:
        return float("nan")
    dt_b = tt[gp] - tt[i0]; dt_a = tt[i1] - tt[gp]
    if dt_b <= 0 or dt_a <= 0:
        return float("nan")
    vbx, vby = (bx[gp] - bx[i0]) / dt_b, (by[gp] - by[i0]) / dt_b
    vax, vay = (bx[i1] - bx[gp]) / dt_a, (by[i1] - by[gp]) / dt_a
    return float(np.hypot(vax - vbx, vay - vby))
```

In `detect_ball_contacts`: replace `dvw = max(1, round(0.12/frame_period))...` + `dv = _velocity_change_at(bx,by,tt,gp,dvw)` with `dv = _velocity_change_at(bx, by, tt, gp, 0.12)`; replace the detection-quality `w`/`a`/`b` block with `a, b = _time_window(tt, gp, 0.2); detection_quality = float(finite[a:b].mean())`.

- [ ] **Step 4: Run, verify pass** (new + the dense-uniform regression).

---

### Task 3: wire dedup into `detect_ball_contacts` + recovery test

**Files:**
- Modify: `silly_kicks/tracking/_ball_contact.py` (per-period, after sort)
- Test: `tests/tracking/test_ball_contact.py`

**Interfaces — Consumes:** `trajectory_dedup_mask` (Task 1), time-based windows (Task 2).

- [ ] **Step 1: Failing recovery test — two-sided IN THE COMMITTED test (GSD-PLAN-01) + concrete control bound (GSBC-PLAN-01).** The committed test must not lean on the transient red Step-2 state. Build the GS-like fixture as a sharp touch (player `P`, `T_TOUCH`) embedded in a trajectory with ~50% consecutive-duplicate ball frames (the sawtooth), and keep its **no-dup twin** (same touch, dups removed) in the same test. Assert BOTH, durably: (a) on the no-dup twin the touch IS detected (proves the fixture is detectable — the "both sides" the plan owes); (b) on the dup fixture `detect_ball_contacts` ALSO detects it after wiring. For the IDSSE-like control, pin a **concrete** bound — the dup fixture's resolved-contact set differs from its no-dup twin by **only the removed duplicate frames' indices** (exact), NOT a loose tolerance, so an over-collapsing dedup fails.

```python
def test_duplicate_frame_sawtooth_recovers_and_matches_nodup_twin():
    dup, nodup = _ball_touch_fixture(dup_fraction=0.5)       # concrete fixtures in this test module
    def touch_hit(frames):
        c = detect_ball_contacts(frames)
        r = c[c["contact_source"].astype(object) != "unresolved"]
        return _has_contact_near(r, player=P, t=T_TOUCH, tol=0.1)
    assert touch_hit(nodup)          # (a) the planted touch is detectable at all (two-sided floor)
    assert touch_hit(dup)            # (b) and survives the ~50% dup sawtooth after dedup wiring

def test_dedup_changes_only_the_duplicate_frames_on_a_dense_control():
    # IDSSE-like control: dedup must touch ONLY the (few) consecutive-dup frames, nothing else.
    dense = _dense_low_motion_ball_with_known_dups(n_dup=3)
    got = detect_ball_contacts(dense); twin = detect_ball_contacts(_strip_consecutive_dups(dense))
    # resolved contact (period, frame_id, player) sets are IDENTICAL -> dedup did not over-collapse
    assert _resolved_key_set(got) == _resolved_key_set(twin)
```

`_ball_touch_fixture`, `_dense_low_motion_ball_with_known_dups`, `_strip_consecutive_dups`, `_has_contact_near`, `_resolved_key_set` are concrete helpers written in the test module (not stubs). The REAL multi-match IDSSE-harmless evidence (spike: 0.437→0.435, −9 contacts) is the Task-7 966-match memo, not a unit assertion.

- [ ] **Step 2: Run, verify fail** (planted touch missed without the dedup).
- [ ] **Step 3: Implement** — in the per-period loop, after `seg_df = ... .sort_values("frame_id")` and the `fid/tt/bx/by/bz/fr` extraction, apply the keep-mask to ALL arrays:

```python
from silly_kicks.tracking.preprocess._dedup import trajectory_dedup_mask
...
keep = trajectory_dedup_mask(bx, by)
fid, tt, bx, by, bz = fid[keep], tt[keep], bx[keep], by[keep], bz[keep]
```

(`fr` / `frame_period` unchanged — cadence for the merge window comes from `frame_rate`; dt-based velocity and the Task-2 time windows handle the sparser sampling.)

- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Update existing `test_ball_contact.py` expected values** where dedup legitimately changes synthetic outputs; re-run the full `test_ball_contact.py` green.

---

### Task 4: documentation + ADR (uncommitted until Task 6)

**Files:**
- Modify: `docs/context/tracking-features.md` (GS caveat :71 — "blunted ball" → duplicate-frame velocity-sawtooth + dedup fix)
- Modify: `docs/superpowers/adrs/ADR-124-ball-contact-detection.md` (one-line forward note: GS "blunted ball" caveat superseded by ADR-125)
- Create: `docs/superpowers/adrs/ADR-125-ball-trajectory-dedup.md` (dedup decision; provider-agnostic; helper placement; option C ruled out — RDP position-geometric; corrected GS root cause; SkillCorner-circular-yardstick caveat)
- Modify: `CHANGELOG.md` (`[Unreleased]`/new entry — NO version number yet; states it corrects the 4.136.0 GS characterisation; does NOT rewrite the shipped 4.136.0 entry)

- [ ] Write each; verify `docs/context/tracking-features.md:71` no longer says "blunted ball"; ADR-125 provisional-number note present; `git grep "blunted" -- docs/context` returns nothing.

---

### Task 5: full CI-faithful gate sweep (shift-left, pre-commit)

- [ ] `python -m ruff check silly_kicks/ tests/ scripts/` + `python -m ruff format --check silly_kicks/ tests/ scripts/` (real exit, NO pipe).
- [ ] `python -m pyright` (bare).
- [ ] `python -m pytest tests/ -m "not e2e"` — full suite (the full-suite-only + committed-artifact gates). Confirm: `PURITY_ENTRIES` still green (detector still pure); NO mirror/c4/glossary/public-api count change (no public surface added); the new `preprocess/test_dedup.py` collected.
- [ ] Fix anything red; re-run until green.

---

### Task 6: COMMIT 1 (owner-gated)

- [ ] **STOP. Show owner the full diff + file list.** One coherent commit: `_dedup.py` + helper tests + `_ball_contact.py` (time windows + dedup wiring) + detector tests + docs (context, ADR-124 note, ADR-125) + CHANGELOG. **Claim the version number now** (derive next-free) + **confirm ADR-125 free**. Commit ONLY on explicit owner "yes". No micro-commits.

---

### Task 7: re-validation memo (owner-gated COMMIT 2)

- [ ] On the clean committed tree, re-run `scripts/measure_ball_contact_validation.py` over the owner-tier corpus on the canonical pining host (parallelised fan-out). Reported-not-gated.
- [ ] Verify the regenerated `docs/research/ball_contact_detection/metrics.json` carries top-level `run_commit` + `run_tree_dirty: false` (ADR-037 gate); run `test_artifact_provenance_output.py` + `test_research_artifacts_carry_no_ids.py` locally BEFORE committing.
- [ ] Expect (pre-registered, not gated): GS agree@0.5 ≈ 0.4 (from ~0.05); IDSSE/SkillCorner ~unchanged. Add the SkillCorner-circular-yardstick caveat to the report.
- [ ] **STOP. Show owner metrics.json + report.md.** Commit 2 (memo + CHANGELOG touch) ONLY on explicit owner "yes".
- [ ] Then: owner non-squash admin-merges + publishes.

---

## Self-review

- **Spec coverage:** dedup helper (T1), window-semantics incl. red-green teeth (T2, GSBC/GSD-SPEC-01), wiring + recovery (T3), doc corrections + ADR-125 (T4), gates (T5), 2-commit owner-gated + re-validation (T6/T7). All spec sections mapped.
- **Placeholders:** the load-bearing code is concrete (helper, time-window, wiring, all assertion logic). The T3 fixture BUILDERS (`_ball_touch_fixture`, `_dense_low_motion_ball_with_known_dups`, `_strip_consecutive_dups`, `_has_contact_near`, `_resolved_key_set`) are named with an explicit output CONTRACT, written in the test module at impl — standard fixture scaffolding, not elided logic. (GSD-PLAN-01: an earlier draft left `...` control bodies + a loose "within a small tolerance" and over-claimed "Both sides asserted"; the committed T3 test is now genuinely two-sided — no-dup twin detects the touch AND the dup fixture recovers it — and the control asserts an EXACT resolved-contact-set identity, GSBC-PLAN-01.)
- **Type consistency:** `_velocity_change_at` signature change (`window:int` → `span_s:float`) has ONE caller (`detect_ball_contacts`), updated in T2/T3. `_time_window` returns `[lo, hi)`.
- **Commit discipline:** two commits only, both owner-gated; no per-task commits (override noted in header).

---

## Amendment (2026-10-15, owner-ruled): z-provenance + `is_aerial` honest-NaN (spec §11)

Folds spec §11 into this cycle. **These tasks (Z1–Z5) execute AFTER Task 3 and BEFORE Task 6 (commit 1)** — they land in the same single code commit. Same TDD red-first discipline; no new commits.

### Task Z1: `z_source` frames column (provider declares)

**Files:** `silly_kicks/tracking/_schema.py` (or wherever `TRACKING_FRAMES_COLUMNS` + the GS alias + `_cast_to_declared_schema` live — pin by grep), the per-provider frame-build adapters (ENUMERATE via a caller sweep — the floor), a central classification constant. Tests: `tests/tracking/test_z_source_schema.py` + per-provider value tests.

**Interfaces — Produces:**
```python
# central ground-truth classification (provider adapter token, NOT inferred)
_Z_PROVENANCE: dict[str, str]  # {"sportec":"measured","idsse":"measured","gradientsports":"measured",
                               #  "skillcorner":"estimated","metrica":"measured","statsbomb":"absent", ...}
Z_SOURCE_VALUES: tuple[str, ...] = ("measured", "estimated", "absent")  # closed vocab, category dtype
```

- [ ] **Add `z_source` to BOTH schema bases (GSBC-PLAN-03):** `TRACKING_FRAMES_COLUMNS` (`schema.py:10`, 19→20) **and** `KLOPPY_TRACKING_FRAMES_COLUMNS` (`schema.py:64`). The aliases `SPORTEC`/`SKILLCORNER`/`METRICA` = `KLOPPY` (`:74/:78/:83`) and `GRADIENTSPORTS` = base (`:87`) inherit automatically. Plus `_cast_to_declared_schema` (category dtype; static per match). Omitting KLOPPY fails the ADR-058 KLOPPY-mirrors-base gate (`test_tracking_schema.py`) AND leaves SkillCorner (which builds from KLOPPY, `skillcorner.py:357`) with no `z_source`. Write the schema-contract test first (BOTH bases carry `z_source`; category; value ∈ `Z_SOURCE_VALUES`; the ADR-058 KLOPPY-mirror assertion stays green) → RED → implement.
- [ ] **Caller sweep (FLOOR):** enumerate every frame producer (`kloppy.py:213`, `skillcorner.py:357`, `sportec.py`, `metrica.py:216`, `gradientsports.py:146`, `snapshot_to_tracking_frames`) and set `z_source` from `_Z_PROVENANCE[provider]` (`absent` where no continuous ball z). **A SkillCorner-specific test asserts `z_source == "estimated"`** (the fix's TARGET reaches it) + IDSSE/Sportec `measured`, GS `measured`.
- [ ] `absent` wherever ball z is all-NaN; SB360/statsbomb `absent`.

### Task Z2: `is_aerial` honest-NaN unless `z_source == "measured"`

**Files:** `silly_kicks/tracking/_ball_contact.py`; `tests/tracking/test_ball_contact.py`.

- [ ] **Failing test:** a frames fixture tagged `z_source="estimated"` (+ one `"measured"`, one `"absent"`) with a ball at z=2.4 over a detected contact → assert `is_aerial` is `<NA>` for estimated/absent, `True` for measured. RED (detector ignores `z_source` today).
- [ ] **Implement:** in `detect_ball_contacts`, read the contact frame's `z_source` (carry it per period alongside `bz`); `is_aerial = pd.NA` unless `z_source == "measured"` and `z` finite, else `bool(z > aerial_z_m)`. **Fail-safe default = `"absent"` (GSBC-PLAN-02)** when the `z_source` column is ABSENT from input — NOT `"measured"` — so a missing declaration yields honest-NaN, never a false boolean (pin in the test: a no-`z_source` fixture → `is_aerial` NaN).
- [ ] **Tag the Task-3 + `_synthetic_match_frames` fixtures `z_source="measured"` explicitly** (they assert boolean `is_aerial`) so they pass under the fail-safe default; add one no-`z_source` fixture asserting NaN.

### Task Z3: `aerial_source` on the contacts table + `first_contact_aerial_source`

**Files:** `_ball_contact.py` (`_CONTACT_COLUMNS`, `detect_ball_contacts`, `add_ball_contacts`, `_empty_contacts`), `silly_kicks/feature_glossary.py`, `tests/sb360/_entries/_ball_contact.py` (verdict), `tests/tracking/test_ball_contact.py`.

- [ ] **Failing test:** `detect_ball_contacts` output has `aerial_source` (category, vocab `Z_SOURCE_VALUES`) = the contact frame's `z_source`; `add_ball_contacts` has `first_contact_aerial_source`. RED.
- [ ] **Implement:** add `aerial_source` to `_CONTACT_COLUMNS` + emit it; `add_ball_contacts` surfaces `first_contact_aerial_source`. Register in `feature_glossary` (2 columns) + the SB360 freeze-frame verdict entry. NOT geometric → NO `reflection.py` entry.

### Task Z4: `validate_z_provenance` diagnostic (declared-vs-signature guard)

**Files:** a tracking diagnostics module (beside `validate_gk_position_clamp`/`validate_velocity_regime` — pin by grep), public export; `tests/.../test_z_provenance.py`.

**Interfaces — Produces:** `validate_z_provenance(frames, *, warn=True) -> ZProvenanceDiagnosis` (+ `ZProvenanceWarning`). The flagged signature is **PERSISTENT ELEVATION (the extrapolation tell), NOT negatives (GSD-SPEC-02):** a `measured`-declared provider is flagged when its ball z (i) sits persistently off-ground — p50 above a small threshold AND off-ground fraction beyond ~0.7 (the SkillCorner signature: p50 0.20 m, 81–86 %) — OR (ii) is all-zero / all-NaN. **Ground-level negatives do NOT flag** (§11.3: GS negatives are below the 1.8 m boundary, legitimately measured).

- [ ] **Failing tests (THREE directions):** (a) `measured`-declared with the extrapolated signature (p50 0.2, 85 % off-ground) → flagged; (b) genuinely-measured (floored at 0, p50 ~0, 30 % off-ground) → clean; (c) **GS-like (p50 ~0, negatives to −3.2 m, 45 % off-ground) declared `measured` → CLEAN (must NOT flag)** — guards against Z4 contradicting §11.3. RED → implement. Thresholds fixed/pre-registered against the spike numbers (IDSSE p50 ~0 / 25–38 % AND GS p50 ~0 / 39–51 % both pass; SkillCorner p50 0.2 / 81–86 % flagged), never fit to the accuracy corpus.

### Task Z5: amended gate sweep

- [ ] Re-run the full CI-faithful suite + the NEW gates: `TRACKING_FRAMES_COLUMNS` schema/alias/cast tests, the category-dtype static-column gate (`z_source`/`aerial_source` category), `feature_glossary` (+2 columns), the SB360 verdict (`aerial_source`), `validate_z_provenance` tests, and confirm no mirror/c4 count change (no new public `add_*`). `detect_ball_contacts`/`add_ball_contacts` stay in `PURITY_ENTRIES`.

### Commit / memo (Tasks 6/7) — amended

- Task 6 commit now ALSO includes Z1–Z5 (one coherent commit: dedup + z-provenance). Task 7 re-validation memo additionally reports per-provider aerial shares keyed to `z_source` (trustworthy only where `measured`). Everything else (owner-gated, non-squash, version/ADR at commit-prep) unchanged.
