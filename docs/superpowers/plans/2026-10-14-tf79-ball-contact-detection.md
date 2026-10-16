# TF-79 Ball-Contact Detection — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan **inline, task-by-task in this session** (owner's standing inline-only rule, reaffirmed on sibling part-deux this cycle; and Tasks 1–5 accrete the SAME file in a strict serial chain — see the Sequencing note — so subagent fan-out gains nothing and risks drift). Do NOT use subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pure, tracking-derived ball-contact detection primitive (candidate touches + first-contact action enricher) that TF-46 and the deferred PathCRF item consume.

**Architecture:** One new pure module `silly_kicks/tracking/_ball_contact.py` (RDP ∪ velocity-break candidate detection, z-aware attribution, merge+representative, confidence, aerial), a per-contact table from `detect_ball_contacts(frames)`, an additive action enricher `add_ball_contacts(actions, frames)`, and a window query `first_contact_in_window`. Mirrors the `_ball_carrier.py` (TF-5) discovery-primitive shape. No trained model, no `*_xfns`, no retrain. Validated in-cycle by a `scripts/_driver.py`-seam driver, reported not gated.

**Tech Stack:** Python, numpy, pandas (pandas-in/out pure functions; no pandera). `silly_kicks.id_compat` for id dtype safety. `silly_kicks._frame_index.group_rows` for no-rescan grouping.

**Spec:** `docs/superpowers/specs/2026-10-14-tf79-ball-contact-detection-design.md` (read it — this plan argues from it).

> **As-built reconciliation (2026-10-15).** Where sketches below say `frame_idx`/`time`, the shipped contacts table uses **`frame_id`/`time_seconds`** (matching `TRACKING_FRAMES_COLUMNS`); the internal `_Candidate` field is `frame_pos`. Per the spec's 2026-10-15 Amendment the table also carries **`velocity_change_ms`** and a **`confidence` recalibrated monotone in |Δv|** (candidates ranked, never hard-dropped; `min_velocity_change_ms` opt-in floor defaults to 0), and §8 validation is the characterisation grid, not tight-window precision/recall. `first_contact_in_window` is half-open `[t0, t1)`. The code + spec are authoritative over any sketch here.

## Global Constraints

Copied verbatim from the spec and the repo's standing rules. Every task's requirements implicitly include this section.

- **Purity:** `detect_ball_contacts` and `add_ball_contacts` are PURE (pandas in, pandas out, no input mutation, no I/O, no global-state mutation). Gated by `PURITY_ENTRIES` (ADR-033).
- **NaN-safe:** `add_*` tolerates NaN ids (ADR-003). "Unresolved" is a kept NaN row (ADR-042), never a guess (E7).
- **Id dtype safety:** never raw `==`/`!=`/`astype(str)` on ids — use `silly_kicks.id_compat` (ADR-019). Tracking-frame `player_id` NULLABLE; ball row id is NA by construction (ADR-058).
- **Coords:** tracking-frame coords are float32 STORAGE / float64 COMPUTE — upcast the slice to float64 at each kernel boundary (ADR-106). SPADL action coords stay float64.
- **Call convention:** `add_ball_contacts(actions, frames, *, params)` — `frames` positional-or-keyword, `params` keyword-only (ADR-078). Canonical shape → no `_CALL_SHAPE_EXEMPT` entry.
- **Orientation:** contacts carry frame-LTR coords; this primitive does not reflect. Geometric columns register in `reflection.py` (ADR-045).
- **No-leakage (validation):** detection params are FIXED/pre-registered (derived from physics + TF-48 priors), checked only against the synthetic-recovery fixture — NEVER selected against the real accuracy corpus. The eval params (`match_tol_s`, match rule, §5.5 representative) are likewise pre-registered.
- **ADR number = ADR-124** (owner ruling 2026-10-15) — the next free number after the frozen, reviewed part-deux hf-jobs cycle (`6f6508a`, PR #284), which holds the two below it. Release versions independent (TF-79 v4.136.0; part-deux v4.137.0 after).
- **No version numbers** anywhere until commit prep (owner rule).
- **COMMIT POLICY (overrides the writing-plans skill's per-task "Commit" step):** NO micro-commits, NO per-step/per-task commits. Each task ends at **tests-green**, not a commit. ONE coherent, fully-tested commit for the code at the end (Task 12), shown to the owner as a diff/file-list and committed ONLY after explicit owner approval. Docs (spec, this plan, ADR, memo, AGENTS.md, context, CHANGELOG) stay UNCOMMITTED on the branch until cycle end, then landed per owner. `git commit`/`push` NEVER without the owner's explicit per-commit yes.
- **Branch:** `tf-79-ball-contact-detection` off `main`; no worktrees.
- **Lint/type at CI scope before done (Shift Left):** `python -m ruff check silly_kicks/ tests/ scripts/` + `--format --check`; `pyright` bare; `python -m pytest tests/ -m "not e2e"`.
- **Windows file-writing caveat:** a scripted `write_text` flips LF→CRLF; driver files pass a driver ASCII gate (no `§` etc.). Write via the native editor, keep scripts ASCII.

**Sequencing note (why the order matters):** several meta-gates assert `set(registry) == {public add_* in tracking.__all__}` (purity, mirror, liveness, SB360). The moment `add_ball_contacts` enters `tracking.__all__`, ALL of those gates must already have their entries or the full suite goes red. Therefore: build and unit-test the module via **direct module imports** (Tasks 1–6) BEFORE exporting it; then export + register every gate together (Tasks 7–9) so the suite never sits red.

---

### Task 1: `BallContactParams` + RDP kernel

**Files:**
- Create: `silly_kicks/tracking/_ball_contact.py`
- Test: `tests/tracking/test_ball_contact.py`

**Interfaces:**
- Produces: `BallContactParams` (frozen dataclass, fields + defaults per spec §3); `_rdp(xy: np.ndarray, epsilon_m: float) -> np.ndarray` (returns a boolean keep-mask over the input points, interior vertices True).

- [ ] **Step 1: Write the failing test**

```python
# tests/tracking/test_ball_contact.py
import numpy as np
import pandas as pd
import pytest
from silly_kicks.tracking import _ball_contact as bc


def test_rdp_keeps_true_vertices_drops_collinear():
    # An L-shaped path: straight run, one corner, straight run.
    xy = np.array([[0, 0], [1, 0], [2, 0], [2, 1], [2, 2]], dtype=float)
    keep = bc._rdp(xy, epsilon_m=0.3)
    # endpoints always kept; the corner (index 2) kept; collinear mids dropped.
    assert keep[0] and keep[-1]
    assert keep[2]
    assert not keep[1] and not keep[3]


def test_rdp_epsilon_monotone():
    xy = np.array([[0, 0], [1, 0.2], [2, 0], [3, 0.2], [4, 0]], dtype=float)
    coarse = bc._rdp(xy, epsilon_m=1.0).sum()
    fine = bc._rdp(xy, epsilon_m=0.05).sum()
    assert fine >= coarse  # smaller epsilon keeps at least as many points
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/tracking/test_ball_contact.py -v`
Expected: FAIL — `module 'silly_kicks.tracking._ball_contact' has no attribute '_rdp'` (and no `BallContactParams`).

- [ ] **Step 3: Write minimal implementation**

```python
# silly_kicks/tracking/_ball_contact.py
"""Ball-contact detection primitive (TF-79).

Candidate touches from a ball trajectory (RDP structural breaks UNION
velocity breaks), z-aware 3D proximity attribution to a player, confidence,
aerial flag, provenance. Pure (pandas in, pandas out). See the spec at
docs/superpowers/specs/2026-10-14-tf79-ball-contact-detection-design.md and
NOTICE for citations.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BallContactParams:
    rdp_epsilon_m: float = 0.3
    speed_drop_frac: float = 0.40
    heading_turn_deg: float = 35.0
    contact_radius_m: float = 2.0
    contact_exist_radius_m: float = 5.0
    contact_max_z_m: float = 2.6
    aerial_z_m: float = 1.8
    merge_window_s: float = 0.12
    min_candidate_speed_ms: float = 2.0

    def __post_init__(self) -> None:
        if self.contact_radius_m > self.contact_exist_radius_m:
            raise ValueError("contact_radius_m must be <= contact_exist_radius_m")
        if not (0.0 < self.speed_drop_frac <= 1.0):
            raise ValueError("speed_drop_frac must be in (0, 1]")


def _rdp(xy: np.ndarray, epsilon_m: float) -> np.ndarray:
    """Ramer-Douglas-Peucker. Returns a boolean keep-mask over xy rows.

    Classical algorithm (Ramer 1972; Douglas & Peucker 1973), implemented fresh.
    """
    n = len(xy)
    keep = np.zeros(n, dtype=bool)
    if n == 0:
        return keep
    keep[0] = keep[-1] = True
    if n <= 2:
        return keep
    stack = [(0, n - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        a, b = xy[i0], xy[i1]
        ab = b - a
        denom = float(np.hypot(*ab))
        seg = xy[i0 + 1 : i1] - a
        if denom == 0.0:
            dist = np.hypot(seg[:, 0], seg[:, 1])
        else:
            dist = np.abs(ab[0] * seg[:, 1] - ab[1] * seg[:, 0]) / denom
        k = int(np.argmax(dist))
        if dist[k] > epsilon_m:
            idx = i0 + 1 + k
            keep[idx] = True
            stack.append((i0, idx))
            stack.append((idx, i1))
    return keep
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/tracking/test_ball_contact.py -v`
Expected: PASS (both RDP tests).

- [ ] **Step 5: Checkpoint (NO commit)** — RDP + params green. Do not commit (see Global Constraints commit policy).

---

### Task 2: velocity-break kernel (frame-rate scaled)

**Files:**
- Modify: `silly_kicks/tracking/_ball_contact.py`
- Test: `tests/tracking/test_ball_contact.py`

**Interfaces:**
- Produces: `_velocity_breaks(xy, t, params) -> tuple[np.ndarray, np.ndarray]` — returns `(break_mask, break_magnitude)` over interior points; magnitude is `max(norm_speed_drop, norm_turn)` used later for the representative-frame rule and confidence.

- [ ] **Step 1: Write the failing test**

```python
def _line(n, dt, vx, vy, x0=0.0, y0=0.0):
    t = np.arange(n) * dt
    x = x0 + vx * t
    y = y0 + vy * t
    return np.column_stack([x, y]), t


def test_velocity_break_fires_on_speed_drop():
    xy1, t1 = _line(6, 0.1, 10.0, 0.0)                 # fast
    xy2, t2 = _line(6, 0.1, 1.0, 0.0, x0=xy1[-1, 0])   # slow, continues
    xy = np.vstack([xy1, xy2]); t = np.concatenate([t1, t1[-1] + 0.1 + t2])
    p = bc.BallContactParams()
    mask, mag = bc._velocity_breaks(xy, t, p)
    assert mask.any()


def test_velocity_break_fires_on_heading_turn():
    xy1, t1 = _line(6, 0.1, 5.0, 0.0)
    xy2, t2 = _line(6, 0.1, 0.0, 5.0, x0=xy1[-1, 0], y0=xy1[-1, 1])
    xy = np.vstack([xy1, xy2]); t = np.concatenate([t1, t1[-1] + 0.1 + t2])
    mask, mag = bc._velocity_breaks(xy, t, bc.BallContactParams())
    assert mask.any()


def test_velocity_break_silent_on_smooth_line():
    xy, t = _line(20, 0.1, 6.0, 0.0)
    mask, mag = bc._velocity_breaks(xy, t, bc.BallContactParams())
    assert not mask.any()


def test_velocity_break_frame_rate_invariant():
    # same physical event at 10 fps and 25 fps -> a break either way
    for dt in (0.1, 0.04):
        xy1, t1 = _line(8, dt, 8.0, 0.0)
        xy2, t2 = _line(8, dt, 0.5, 0.0, x0=xy1[-1, 0])
        xy = np.vstack([xy1, xy2]); t = np.concatenate([t1, t1[-1] + dt + t2])
        mask, _ = bc._velocity_breaks(xy, t, bc.BallContactParams())
        assert mask.any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/tracking/test_ball_contact.py -k velocity_break -v`
Expected: FAIL — `_velocity_breaks` not defined.

- [ ] **Step 3: Write minimal implementation**

```python
def _velocity_breaks(xy: np.ndarray, t: np.ndarray, params: BallContactParams):
    n = len(xy)
    mask = np.zeros(n, dtype=bool)
    mag = np.zeros(n, dtype=float)
    if n < 3:
        return mask, mag
    dt = np.diff(t)
    dt[dt <= 0] = np.nan
    vel = np.diff(xy, axis=0) / dt[:, None]          # per-second velocity (frame-rate scaled)
    speed = np.hypot(vel[:, 0], vel[:, 1])
    for i in range(1, n - 1):
        s0, s1 = speed[i - 1], speed[i]
        if not np.isfinite(s0) or not np.isfinite(s1):
            continue
        drop = (s0 - s1) / s0 if s0 > 0 else 0.0
        # heading change between successive per-second velocity vectors
        if s0 > 0 and s1 > 0:
            cos = np.clip(np.dot(vel[i - 1], vel[i]) / (s0 * s1), -1.0, 1.0)
            turn_deg = np.degrees(np.arccos(cos))
        else:
            turn_deg = 0.0
        fired_drop = drop >= params.speed_drop_frac and s0 >= params.min_candidate_speed_ms
        fired_turn = turn_deg >= params.heading_turn_deg and s0 >= params.min_candidate_speed_ms
        if fired_drop or fired_turn:
            mask[i] = True
            mag[i] = max(drop / params.speed_drop_frac, turn_deg / params.heading_turn_deg)
    return mask, mag
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/tracking/test_ball_contact.py -k velocity_break -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Checkpoint (NO commit)** — velocity-break green.

---

### Task 3: z-aware attribution + merge/representative rule

**Files:**
- Modify: `silly_kicks/tracking/_ball_contact.py`
- Test: `tests/tracking/test_ball_contact.py`

**Interfaces:**
- Produces: `_attribute(cand_frame, bx, by, bz, players, params) -> tuple[player_id|NA, float]` (nearest playable player within radius → `(player_id, distance)`, else `(NA, nan)`); `_merge_clusters(cands, mag, source, t, params) -> list[Candidate]` collapsing within `max(merge_window_s, 2*frame_period)`, representative = max-`mag` frame (earliest on tie; RDP-only cluster → the RDP vertex).

- [ ] **Step 1: Write the failing test**

```python
def test_attribution_rejects_overhead_ball():
    # player at (10,10); ball passing overhead at z=4m (> contact_max_z_m 2.6)
    players = pd.DataFrame({"player_id": [7], "x": [10.0], "y": [10.0]})
    pid, dist = bc._attribute(0, np.array([10.0]), np.array([10.0]), np.array([4.0]),
                              players, bc.BallContactParams())
    assert pd.isna(pid)


def test_attribution_nearest_playable():
    players = pd.DataFrame({"player_id": [7, 9], "x": [10.5, 20.0], "y": [10.0, 10.0]})
    pid, dist = bc._attribute(0, np.array([10.0]), np.array([10.0]), np.array([0.1]),
                              players, bc.BallContactParams())
    assert pid == 7 and dist < 1.0


def test_attribution_no_player_in_range_is_unresolved():
    players = pd.DataFrame({"player_id": [7], "x": [50.0], "y": [50.0]})
    pid, dist = bc._attribute(0, np.array([10.0]), np.array([10.0]), np.array([0.1]),
                              players, bc.BallContactParams())
    assert pd.isna(pid)


def test_merge_representative_is_max_magnitude_frame():
    # two candidate frames 0.04s apart (within merge window) -> one cluster;
    # representative = the higher-magnitude frame.
    cands = [1, 2]
    mag = {1: 0.5, 2: 1.3}
    src = {1: "velocity_break", 2: "velocity_break"}
    t = {1: 1.00, 2: 1.04}
    out = bc._merge_clusters(cands, mag, src, t, bc.BallContactParams())
    assert len(out) == 1 and out[0].frame_idx == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/tracking/test_ball_contact.py -k "attribution or merge" -v`
Expected: FAIL — `_attribute` / `_merge_clusters` not defined.

- [ ] **Step 3: Write minimal implementation**

Implement `_attribute` (vectorised distance to players, playable mask `z NaN | z <= contact_max_z_m`, prefer `contact_radius_m` then fall back to `contact_exist_radius_m`, return nearest `player_id` via `id_compat` dtype-preserving indexing, else `pd.NA`). Implement a small `Candidate` dataclass (`frame_idx, time, source`) and `_merge_clusters` (sort candidates by time; group while gap <= `max(merge_window_s, 2*median_frame_period)`; per cluster pick max-`mag` frame, ties→earliest; a cluster with no velocity-break member keeps its RDP vertex frame; `source` = `"both"` if the cluster mixes rdp+velocity_break else the single source).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/tracking/test_ball_contact.py -k "attribution or merge" -v`
Expected: PASS.

- [ ] **Step 5: Checkpoint (NO commit).**

---

### Task 4: `detect_ball_contacts` — assemble, gaps/E7, confidence, aerial, per-period `group_rows`

**Files:**
- Modify: `silly_kicks/tracking/_ball_contact.py`
- Test: `tests/tracking/test_ball_contact.py`
- Test (scale guard): `tests/test_scale_guards.py`
- Modify (scale registry): `tests/_scale_guarded.py:19`

**Interfaces:**
- Consumes: `BallContactParams`, `_rdp`, `_velocity_breaks`, `_attribute`, `_merge_clusters`, `silly_kicks._frame_index.group_rows`.
- Produces: `detect_ball_contacts(frames, *, params=BallContactParams()) -> pd.DataFrame` with columns `period_id, frame_idx, time, player_id, contact_x, contact_y, confidence, is_aerial, contact_source`; `_confidence(break_mag, proximity, detection_quality) -> float`.

- [ ] **Step 1: Write the failing test** (synthetic match with planted contacts; grounds the liveness fixture later)

```python
def _synthetic_match_frames():
    """Two periods, 25 fps; ball struck then received by a different player;
    one aerial touch (z>1.8) and one ground touch; a detection gap in P2."""
    # ... build long-form frames: is_ball rows + player rows, team_id category,
    # player_id Int64, x/y/z float32, period_id, frame_idx, time.
    # (Executor: construct ~2 planted contacts per period with the ball
    #  decelerating/turning at a player within 2 m; one with z=2.4, one z=0.1;
    #  insert a 3-frame NaN ball run in P2.)
    ...


def test_detect_contacts_recovers_planted_touches():
    frames = _synthetic_match_frames()
    out = bc.detect_ball_contacts(frames)
    assert set(out.columns) == {
        "period_id", "frame_idx", "time", "player_id",
        "contact_x", "contact_y", "confidence", "is_aerial", "contact_source",
    }
    resolved = out[out.contact_source != "unresolved"]
    assert resolved.player_id.notna().all()
    assert out.contact_source.nunique() >= 2          # non-vacuity: >1 source value
    assert resolved.player_id.nunique() >= 2          # non-vacuity: >1 player


def test_detect_contacts_is_aerial_nan_when_z_nan():
    frames = _synthetic_match_frames()
    out = bc.detect_ball_contacts(frames)
    # aerial is nullable boolean; where z was NaN at contact, is_aerial is <NA>
    assert out["is_aerial"].dtype.name == "boolean"


def test_detect_contacts_no_contact_across_gap():
    frames = _synthetic_match_frames()
    out = bc.detect_ball_contacts(frames)
    # no contact frame lands inside the known P2 NaN-ball gap window
    ...


def test_detect_contacts_confidence_bounded():
    out = bc.detect_ball_contacts(_synthetic_match_frames())
    c = out["confidence"].dropna()
    assert ((c >= 0) & (c <= 1)).all()


def test_detect_contacts_pure():
    frames = _synthetic_match_frames()
    before = frames.copy(deep=True)
    _ = bc.detect_ball_contacts(frames)
    pd.testing.assert_frame_equal(frames, before)


def test_detect_contacts_reflection_mirror_equal():
    frames = _synthetic_match_frames()
    base = bc.detect_ball_contacts(frames)
    mirrored = frames.copy()
    mirrored["x"] = 105.0 - mirrored["x"]     # physical x-reflection
    mout = bc.detect_ball_contacts(mirrored)
    # contact_x of the mirror equals 105 - base contact_x (same touches)
    np.testing.assert_allclose(
        sorted(mout["contact_x"].dropna()),
        sorted(105.0 - base["contact_x"].dropna()), atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/tracking/test_ball_contact.py -k detect_contacts -v`
Expected: FAIL — `detect_ball_contacts` not defined.

- [ ] **Step 3: Write minimal implementation**

Implement `detect_ball_contacts`:
- `group_rows(frames, by=("period_id",))` to iterate periods without rescanning (register the scale guard below).
- Per period: extract `is_ball` rows ordered by `frame_idx`; upcast `x/y/z` slices to float64 (ADR-106). Split into segments at NaN-coordinate runs (E7 — no detection across gaps).
- Per segment: `_rdp` keep-mask → RDP candidate indices; `_velocity_breaks` → velocity candidates + magnitudes; union; `_merge_clusters`.
- Per merged candidate: `_attribute` against that frame's player rows; `is_aerial = z > aerial_z_m` as nullable boolean, `<NA>` where z is NaN; `confidence = _confidence(break_mag, proximity, detection_quality)`; `contact_source` from the cluster (unresolved rows kept with `contact_source="unresolved"`, `player_id=<NA>`, `confidence=NaN`).
- Assemble the table; `contact_source` as pandas `category`; `player_id` via `id_compat.restore_id_dtype`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/tracking/test_ball_contact.py -k detect_contacts -v`
Expected: PASS.

- [ ] **Step 5: Register the scale guard**

Add to `tests/_scale_guarded.py` `SCALE_GUARDED` (line ~19):
```python
"silly_kicks.tracking._ball_contact.detect_ball_contacts": "test_detect_ball_contacts_is_subquadratic",
```
Add to `tests/test_scale_guards.py`:
```python
def test_detect_ball_contacts_is_subquadratic():
    from silly_kicks.tracking import _ball_contact as bc
    from tests._perf_structural import assert_subquadratic_growth, rows_scanned_counter

    def measure(n):
        frames = _ball_frames_scaling(n)   # n ball frames across 1 period, growing
        with rows_scanned_counter() as c:
            bc.detect_ball_contacts(frames)
        return c["n"]
    assert_subquadratic_growth(measure, sizes=(256, 1024, 4096), label="detect_ball_contacts")
```

- [ ] **Step 6: Run guards green**

Run: `python -m pytest tests/test_scale_guards.py -k detect_ball_contacts tests/test_scale_guard_registry.py -v`
Expected: PASS (`group_rows_callers() - set(SCALE_GUARDED)` empty; growth subquadratic).

- [ ] **Step 7: Checkpoint (NO commit).**

---

### Task 5: `add_ball_contacts` action enricher + `first_contact_in_window`

**Files:**
- Modify: `silly_kicks/tracking/_ball_contact.py`
- Test: `tests/tracking/test_ball_contact.py`

**Interfaces:**
- Consumes: `detect_ball_contacts`.
- Produces:
  - `first_contact_in_window(contacts, period_id, t0, t1) -> pd.Series | None` (earliest resolved contact with `t0 <= time_seconds < t1` — HALF-OPEN, so adjacent action windows never double-count the boundary contact; else None).
  - `add_ball_contacts(actions, frames, *, params=BallContactParams()) -> pd.DataFrame` — `actions` copy with added `first_contact_time, first_contact_player_id, first_contact_x, first_contact_y, first_contact_confidence, first_contact_is_aerial, first_contact_source` (all NaN/`<NA>` where no contact resolves in the action's window). Window = `[action.start_time, next_action.start_time)` within the same period (uses `start_time`; see spec — GS carries it; other providers derive from `time_seconds`).

- [ ] **Step 1: Write the failing test**

```python
def test_first_contact_in_window_picks_earliest():
    contacts = pd.DataFrame({
        "period_id": [1, 1], "frame_idx": [10, 20], "time": [1.0, 2.0],
        "player_id": pd.array([7, 9], dtype="Int64"),
        "contact_x": [30.0, 40.0], "contact_y": [20.0, 25.0],
        "confidence": [0.8, 0.9], "is_aerial": pd.array([False, True], dtype="boolean"),
        "contact_source": pd.Categorical(["both", "rdp"]),
    })
    s = bc.first_contact_in_window(contacts, 1, 0.5, 1.5)
    assert s is not None and s["player_id"] == 7


def test_add_ball_contacts_attaches_first_contact_and_is_pure():
    actions, frames = _actions_and_frames()     # a corner action followed by a window with a planted contact
    before = actions.copy(deep=True)
    out = bc.add_ball_contacts(actions, frames)
    assert {"first_contact_time", "first_contact_player_id", "first_contact_x",
            "first_contact_y", "first_contact_confidence", "first_contact_is_aerial",
            "first_contact_source"} <= set(out.columns)
    assert out.loc[0, "first_contact_player_id"] == _expected_pid
    pd.testing.assert_frame_equal(actions, before)   # purity


def test_add_ball_contacts_nan_when_no_contact():
    actions, frames = _actions_and_frames_no_contact()
    out = bc.add_ball_contacts(actions, frames)
    assert out["first_contact_player_id"].isna().all()
    assert out["first_contact_source"].isna().all() or (out["first_contact_source"] == "unresolved").all()


def test_add_ball_contacts_raises_without_time_base():
    # actions carrying neither start_time nor time_seconds -> fail loud, never silent-NaN
    actions = pd.DataFrame({"action_id": [0], "period_id": [1]})
    frames = _synthetic_match_frames()
    with pytest.raises(ValueError):
        bc.add_ball_contacts(actions, frames)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/tracking/test_ball_contact.py -k "first_contact or add_ball_contacts" -v`
Expected: FAIL — not defined.

- [ ] **Step 3: Write minimal implementation**

Implement `first_contact_in_window` (filter period + resolved + time_seconds in half-open `[t0, t1)`, sort by time, take first) and `add_ball_contacts` (compute `contacts = detect_ball_contacts(frames, params=params)`; derive per-action windows per period using `start_time`; for each action, `first_contact_in_window`; assemble additive columns NaN-safe; copy input, never mutate; `id_compat` for `first_contact_player_id`; `first_contact_x/y` float64).

**Fail loud on a missing time base (loud-raise-for-required-input-columns):** `add_ball_contacts` must `raise ValueError` if `actions` carries neither `start_time` nor a derivable time base (`time_seconds`) — it must NOT silently skip, return all-NaN, or guess a window. The window derivation is `[action.start_time, next_action.start_time)` within a period when `start_time` is present; derive `start_time` from `time_seconds` when only that is present; otherwise raise naming the missing columns.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/tracking/test_ball_contact.py -k "first_contact or add_ball_contacts" -v`
Expected: PASS.

- [ ] **Step 5: Checkpoint (NO commit).**

---

### Task 6: reflection registry + contact_source categorical home (verify spec §6)

**Files:**
- Modify: `silly_kicks/reflection.py` (`_TRACKING_REFLECTION_KINDS:108`, `_SPADL_REFLECTION_KINDS:154`)
- Modify: `silly_kicks/tracking/schema.py:149` OR the correct categorical-domain home (verify first)
- Test: `tests/tracking/test_ball_contact.py`, `tests/test_tracking_schema.py` (read the gate body)

**Interfaces:**
- Produces: reflection declarations so `contact_x/contact_y` (contacts table) and `first_contact_x/first_contact_y` (actions) reflect as `point_x`/`point_y`; `contact_source`/`first_contact_source` categorical vocab `{rdp, velocity_break, both, unresolved}`.

- [ ] **Step 1: Verify the categorical-domain gate body (spec §6 correction)**

Read `tests/test_tracking_schema.py::test_tracking_categorical_domains_keys_subset_of_columns`. Confirm whether keys must be columns of `TRACKING_FRAMES_COLUMNS` (expected: yes). Since `contact_source` is a contacts-table/actions column, NOT a frame column, it must NOT go into `TRACKING_CATEGORICAL_DOMAINS`.

**Decision + spec amendment:** route `contact_source` as a pandas `category` dtype on the contacts table with the fixed vocab, validated by a dedicated test in `tests/tracking/test_ball_contact.py` (below). Record a one-line **Amendment** in the spec (§6) that `contact_source` lives as a category dtype on the contacts table, not in `TRACKING_CATEGORICAL_DOMAINS` (the frame-column domain registry) — per the "record owner waiver / amend the source-of-truth doc" rule. Surface to owner.

- [ ] **Step 2: Write the failing tests**

```python
def test_contact_source_vocab_closed():
    out = bc.detect_ball_contacts(_synthetic_match_frames())
    assert set(out["contact_source"].cat.categories) <= {"rdp", "velocity_break", "both", "unresolved"}


def test_contact_columns_registered_for_reflection():
    from silly_kicks.reflection import TRACKING_REFLECTION_KINDS, SPADL_REFLECTION_KINDS
    assert TRACKING_REFLECTION_KINDS["contact_x"] == "point_x"
    assert TRACKING_REFLECTION_KINDS["contact_y"] == "point_y"
    assert SPADL_REFLECTION_KINDS["first_contact_x"] == "point_x"
    assert SPADL_REFLECTION_KINDS["first_contact_y"] == "point_y"
```

- [ ] **Step 3: Run to verify it fails**

Run: `python -m pytest tests/tracking/test_ball_contact.py -k "contact_source_vocab or reflection" -v`
Expected: FAIL (KeyError in the reflection dicts).

- [ ] **Step 4: Register**

- `silly_kicks/reflection.py:108` `_TRACKING_REFLECTION_KINDS`: add `"contact_x": "point_x", "contact_y": "point_y"`.
- `silly_kicks/reflection.py:154` `_SPADL_REFLECTION_KINDS`: add `"first_contact_x": "point_x", "first_contact_y": "point_y"` (mirror the `enriched_start_x` ADR-025 pattern). The other `first_contact_*`/`contact_source` columns are invariant (not geometric) → no entry needed; confirm `GEOMETRIC_NAME` does not warn on them (they don't match `.*_[xy]`).

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/tracking/test_ball_contact.py -k "contact_source_vocab or reflection" tests/test_tracking_schema.py -v`
Expected: PASS.

- [ ] **Step 6: Checkpoint (NO commit).**

---

### Task 7: Export + purity + call-convention + mirror registry

**Files:**
- Modify: `silly_kicks/tracking/_ball_contact.py` (no `__all__` inside; exports curated centrally)
- Modify: `silly_kicks/tracking/__init__.py` (`__all__` + import line ~329)
- Modify: `silly_kicks/tracking/features.py` (re-export `add_ball_contacts` — mirror the `add_das` export path; both `tracking.__all__` and `features.__all__`)
- Modify: `tests/test_add_star_purity.py` (`PURITY_ENTRIES`)
- Modify: `tests/tracking/_mirror_registry.py` (`MIRROR_ENTRIES` + size pin)

**Interfaces:**
- Produces: `silly_kicks.tracking.{detect_ball_contacts, add_ball_contacts, first_contact_in_window, BallContactParams}` public.

- [ ] **Step 1: Export** — add the four public names to `tracking/__init__.py` `__all__` (alphabetical) + `from ._ball_contact import BallContactParams, add_ball_contacts, detect_ball_contacts, first_contact_in_window` (~line 329). Re-export `add_ball_contacts` through `features.py` exactly as `add_das` is exported (verify add_das's path and mirror it) so `silly_kicks.tracking.features.add_ball_contacts` resolves and `features.__all__` includes it.

- [ ] **Step 2: Run the meta-surface gates to see them fail** (they now demand registration)

Run: `python -m pytest tests/test_add_star_purity.py tests/tracking/test_call_convention_registry.py tests/tracking/test_mirror_registry.py -v`
Expected: FAIL — purity `test_meta_registration_complete_per_package`; mirror `test_every_public_add_is_registered` + size pin (32 ≠ 33). Call-convention should PASS (canonical shape auto-discovered).

- [ ] **Step 3: Register purity** — `tests/test_add_star_purity.py` `PURITY_ENTRIES`:
```python
"tracking:add_ball_contacts": _one(
    lambda: [_make_actions_with_start_time(), _fresh_frames_with_contacts()],
    lambda i: F.add_ball_contacts(i[0], i[1]),
),
```
(Use a single variant — `add_ball_contacts` does not branch on input shape. If the executor adds a column-presence branch, provide ≥2 variants per the AST heuristic.)

- [ ] **Step 4: Register mirror** — `tests/tracking/_mirror_registry.py` `MIRROR_ENTRIES`:
```python
_entry(
    "add_ball_contacts",
    call=lambda actions, frames: T.add_ball_contacts(actions, frames),
    columns={
        "first_contact_x": "point_x",
        "first_contact_y": "point_y",
        "first_contact_time": "invariant",
        "first_contact_player_id": "invariant",
        "first_contact_confidence": "invariant",
        "first_contact_is_aerial": "invariant",
        "first_contact_source": "invariant",
    },
    tolerance=1e-6,
    home_team_id_role=...,          # match the add_das/other add_* role
    non_vacuity=...,                # assert ≥1 resolved contact in the mirror scene
),
```
Bump `test_registry_surface_is_the_expected_size` expected count `32 -> 33`.

- [ ] **Step 5: Run to verify green**

Run: `python -m pytest tests/test_add_star_purity.py tests/tracking/test_call_convention_registry.py tests/tracking/test_mirror_registry.py -v`
Expected: PASS.

- [ ] **Step 6: Checkpoint (NO commit).**

---

### Task 8: liveness gate + ADR-056 precondition test

**Files:**
- Modify: `tests/tracking/test_aggregator_column_liveness.py` (`ENTRIES:617` + a domain runner + a precondition test)

- [ ] **Step 1: Run to see the meta-surface gate fail**

Run: `python -m pytest tests/tracking/test_aggregator_column_liveness.py -v`
Expected: FAIL — `test_meta_surface_complete` (`add_ball_contacts` missing from `ENTRIES`).

- [ ] **Step 2: Add the domain runner + entry**
```python
def _run_ball_contacts():
    actions, frames = _ball_contact_scene()   # own scene: a real trajectory
                                               # crossing >=2 players' reach -> >=2 resolved contacts
    return actions, F.add_ball_contacts(actions, frames)

ENTRIES["add_ball_contacts"] = _run_ball_contacts
```

- [ ] **Step 3: Add the ADR-056 precondition test** (mirror `test_pitch_control_at_target_fixture_has_offball_destinations:689`)
```python
def test_ball_contact_fixture_has_resolvable_contacts():
    actions, frames = _ball_contact_scene()
    contacts = __import__("silly_kicks.tracking._ball_contact", fromlist=["detect_ball_contacts"]).detect_ball_contacts(frames)
    resolved = contacts[contacts.contact_source != "unresolved"]
    assert resolved.player_id.nunique() >= 2   # the liveness non-constant check has teeth
```

- [ ] **Step 4: Run to verify green**

Run: `python -m pytest tests/tracking/test_aggregator_column_liveness.py -v`
Expected: PASS (`add_ball_contacts` columns non-null + non-constant; precondition holds).

- [ ] **Step 5: Checkpoint (NO commit).**

---

### Task 9: SB360 verdict + feature_glossary + NOTICE

**Files:**
- Create: `tests/sb360/_entries/_ball_contact.py`
- Modify: `silly_kicks/feature_glossary.py`
- Modify: `NOTICE`

- [ ] **Step 1: Run SB360 + glossary + notice-linkage gates to see them fail**

Run: `python -m pytest tests/sb360/test_registry_surface.py tests/test_feature_glossary_coverage.py -v`
Expected: FAIL — SB360 `test_every_public_add_star_is_registered`; glossary `test_no_undocumented_columns`.

- [ ] **Step 2: SB360 honest-NaN verdict** — create `tests/sb360/_entries/_ball_contact.py`, mirroring `_context.py:140-166` (`_ELASTIC_HONEST_NAN_RATIONALE`/`_el_hn`):
```python
_BALL_CONTACT_HONEST_NAN_RATIONALE = (
    "Ball-contact detection is continuous-only: candidate detection needs a dense ball "
    "trajectory. A single SB360 freeze frame has no trajectory, so add_ball_contacts resolves "
    "no contact (all-NaN) by design -- it declines to fabricate a touch from one snapshot. The "
    "leg difference is a data-regime boundary, not a library defect."
)
def _bc_hn() -> AxisVerdict:
    return AxisVerdict("all_nan", "honest_nan", rationale=_BALL_CONTACT_HONEST_NAN_RATIONALE)

_BALL_CONTACT_COLS = ("first_contact_time", "first_contact_player_id", "first_contact_x",
                      "first_contact_y", "first_contact_confidence", "first_contact_is_aerial",
                      "first_contact_source")
_entry(
    "add_ball_contacts",
    C.generic(T.add_ball_contacts),
    columns=_BALL_CONTACT_COLS,
    velocity={c: _bc_hn() for c in _BALL_CONTACT_COLS},
    visibility={roster: {c: _bc_hn() for c in _BALL_CONTACT_COLS} for roster in VISIBILITY_ROSTERS},
)
```
(Executor: match the exact `_entry`/`C.generic`/`VISIBILITY_ROSTERS` imports used by `_context.py`.)

- [ ] **Step 3: feature_glossary** — add `_M_BALL_CONTACT = "silly_kicks.tracking._ball_contact"` + `_A_RDP`/`_A_PATHCRF` attribution tokens (verbatim-matching NOTICE), and a `FeatureColumn(...)` per emitted column (`first_contact_time`, `first_contact_player_id`, `first_contact_x`, `first_contact_y`, `first_contact_confidence`, `first_contact_is_aerial`, `first_contact_source`) inside the `_register(...)` call. `emitting_module=_M_BALL_CONTACT`; `higher_is_better` None (provenance/positional, not a quality score).

- [ ] **Step 4: NOTICE** — add an RDP citation entry (Ramer 1972; Douglas & Peucker 1973; `Used by: silly_kicks.tracking._ball_contact (TF-79)`; classical, no faithfulness caveat) and a PathCRF "INSPIRED BY, not lifted" entry mirroring the Cao/xCrossAttempt template (cite PathCRF, state no CRF code ported, MPL-2.0 → reimplemented; list adapted concept = velocity-break/RDP candidate segmentation). Any `_A_*` token used in `feature_glossary.py` must appear verbatim here.

- [ ] **Step 5: Run to verify green**

Run: `python -m pytest tests/sb360/ tests/test_feature_glossary_coverage.py tests/test_feature_glossary_notice_linkage.py -v`
Expected: PASS.

- [ ] **Step 6: Checkpoint (NO commit).**

---

### Task 10: Validation driver (in-cycle, reported-not-gated)

**Files:**
- Create: `scripts/measure_ball_contact_validation.py`
- Test: `tests/scripts/test_measure_ball_contact_validation.py` (light; `run()` over a tiny fixture, no network)

**Interfaces:**
- Consumes: `scripts/_driver.for_each`, `scripts/_provenance.{git_provenance, require_clean_tree}`, `scripts/_partition.write_table_atomically`, `scripts/_loader_pining.list_match_refs`, `silly_kicks.tracking.detect_ball_contacts`.
- Produces: a driver computing precision / recall / timing-error vs event-recorded touches (match rule: same-player + `|Δt| <= match_tol_s`, `match_tol_s=0.20` FIXED/pre-registered), honest unresolved-rate per provider; memo at `docs/research/ball_contact_detection/{metrics.json,report.md}`.

- [ ] **Step 1: Write the failing test** (driver `run()` on a synthetic ref, asserts metrics keys + that `match_tol_s` is fixed, not fit)

```python
def test_driver_run_reports_precision_recall_timing(tmp_path):
    import scripts.measure_ball_contact_validation as d
    m = d.run(tmp_path, match_ids=None, _fixture_refs=_tiny_fixture_refs(), provenance={"commit": "test", "dirty": False})
    assert {"precision", "recall", "timing_error_s", "unresolved_rate", "match_tol_s"} <= set(m["overall"])
    assert m["overall"]["match_tol_s"] == 0.20            # pre-registered, not fit
    assert "corpus_bound" in m                            # corpus bound recorded inside the artifact
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/scripts/test_measure_ball_contact_validation.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement the driver** — mirror `scripts/validate_xs_probe.py` skeleton exactly: module docstring + CLI example; `run(out, *, match_ids=None, provenance=None, cache_dir=None, _fixture_refs=None)` building `refs, load` via `list_match_refs` (or the injected `_fixture_refs` for the test), a `_work(ref)` closure computing per-match contacts and matching them to event touches, `for_each(refs, key=..., load=load, work=_work, counters=..., shard_root=Path(out)/"shards", token_inputs={... every content-affecting value incl. match_tol_s, params ...}, tag="ball_contact", label="match")`, combine via `res.shard_keys`+`shard_path`, compute metrics (dense-optical providers reported first; SkillCorner → unresolved-rate), `_write(out, metrics)` → `metrics.json`+`report.md`. `main()` with argparse (`--out type=Path`, `--match-ids-json`, `--providers`, `--allow-dirty` store_true) and `provenance=require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)`. ASCII only (driver ASCII gate). Merge `prov` dirty/commit into `metrics.json`.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/scripts/test_measure_ball_contact_validation.py -v`
Expected: PASS.

- [ ] **Step 5: Checkpoint (NO commit).** NOTE: the real-corpus run (owner-tier pining, dense-optical-first; heavy — media-PC delegation) and the memo numbers land during cycle execution, with the §8.1 z-provenance spike result. Reported, not gated.

---

### Task 11: z-provenance spike + full-suite + lint + type (Shift Left)

**Files:** none new (investigation + verification)

- [ ] **Step 1: z-provenance spike (§8.1)** — determine per provider whether ball `z` is measured or estimated: inspect kloppy's SkillCorner `z` handling + the SkillCorner tracking schema; measure `z` variance/quantisation in an owner-tier pining sample (media-PC delegation). Record the verdict in the validation memo + the ADR. If estimated for a provider, confirm `is_aerial` is honest-NaN there. Report the finding to the owner.
- [ ] **Step 2: full fast suite** — `python -m pytest tests/ -m "not e2e"`. Expected: PASS (several TF-79 gates are full-suite-only).
- [ ] **Step 3: lint** — `python -m ruff check silly_kicks/ tests/ scripts/` then `python -m ruff format --check silly_kicks/ tests/ scripts/`. Fix, re-run the whole lint.
- [ ] **Step 4: type** — `pyright`. Fix any TF-79 findings at full-package scope.
- [ ] **Step 5: Checkpoint (NO commit).** Report green status to the owner.

---

### Task 12: Cycle-end docs + commit prep (OWNER-GATED)

**Files:**
- Modify: `AGENTS.md` (TF table — TF-79 row: module + public surface)
- Modify: `docs/context/tracking-features.md` (narrative)
- Create: `docs/superpowers/adrs/ADR-NNN-ball-contact-detection.md` (**re-confirm the free ADR number first** — contended; see Global Constraints)
- Modify: `CHANGELOG.md` (keyed entry — NO version number until the owner assigns it at release)
- Modify: `TODO.md` (move TF-79 off On Deck)
- Modify: `docs/c4` (render via the pinned Graphviz `dot` pipeline — never Smetana)
- The spec, this plan, the memo: committed at cycle end.

- [ ] **Step 1: Re-confirm the free ADR number** — `ls docs/superpowers/adrs/ | grep -oE 'ADR-[0-9]+' | sort -t- -k2 -n | tail -3` AND check unmerged branches (`git log --all --oneline -- 'docs/superpowers/adrs/*'`). Pick the lowest free number not taken by part-deux. Write the ADR with the six decisions from spec §12.
- [ ] **Step 2: Write the docs** — AGENTS.md TF-79 row (module `_ball_contact.py`; public `detect_ball_contacts`/`add_ball_contacts`/`first_contact_in_window`/`BallContactParams`); tracking-features.md narrative; CHANGELOG entry (no version); TODO.md move.
- [ ] **Step 3: Render C4** — regenerate `docs/c4`/`architecture.html` with the pinned `dot` (per the user's C4 rule), confirm the assembler ran (non-placeholder SVG).
- [ ] **Step 4: Re-run the full fast suite + lint + pyright once more** (docs touch no code, but C4/glossary/contract gates can trip). Expected: PASS.
- [ ] **Step 5: Spec amendment** — the dated §6 `contact_source` Amendment was LANDED at plan-review r1 (A `TF79-PLAN-01` / B `TF79-SPEC-03`); here just confirm it is present and that the owner has ratified it.
- [ ] **Step 6: COMMIT PREP — STOP.** Show the owner: `git status`, `git diff --stat`, and the full diff. Propose ONE coherent commit for the code (and state whether docs land in the same commit or a trailing docs commit, per owner). Draft the commit message:
```
feat(tracking): TF-79 ball-contact detection primitive (ADR-NNN)

<body: candidate-touch detection (RDP ∪ velocity-break, z-aware attribution),
detect_ball_contacts table + add_ball_contacts enricher + first_contact_in_window;
pure, no retrain, no *_xfns; in-cycle validation driver; honest-NaN on estimated z.>

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```
**Do NOT run `git commit` until the owner replies with explicit approval for this specific commit.** Do NOT `git push`.

---

## Self-Review

**1. Spec coverage** — every spec section maps to a task:
- §3 surface → Tasks 1,5,7. §4 contacts table → Task 4. §5 algorithm → Tasks 1–4 (incl. §5.5 representative → Task 3). §6 confidence/aerial/provenance → Tasks 4,6 (contact_source home corrected). §7 reflection → Task 6. §8 validation (incl. §8.1 spike, §8.2 match_tol_s, no-leakage) → Tasks 10,11. §9 CI gates (8) → Tasks 6,7,8,9 + scale guard Task 4. §10 tests → every task (TDD). §11 file inventory → Tasks 1,6,7,9,10,12. §12 ADR → Task 12. §13 workflow/commit gate → Global Constraints + Task 12. §14 open items → Task 6 (contact_source), Task 11 (spike), fixed params Tasks 1–4.
- Gap check: none. The `metric_contracts` non-registration (spec §9) correctly yields NO task (confirmed: enricher, not mart).

**2. Placeholder scan** — the `...` in test bodies (synthetic-fixture builders `_synthetic_match_frames`, `_actions_and_frames`, `_ball_contact_scene`, `_ball_frames_scaling`) are fixture-construction stubs the executor fills against the described contract (planted contacts, known answers). They are the ONE deliberate exception, each with an explicit construction spec in prose; every registry edit, signature, and gate command is concrete. No "TBD/handle edge cases/add validation" placeholders.

**3. Type consistency** — `detect_ball_contacts(frames, *, params)`; `add_ball_contacts(actions, frames, *, params)`; `first_contact_in_window(contacts, period_id, t0, t1)`; `BallContactParams` field names identical across Tasks 1–12. Column names (`contact_*`, `first_contact_*`) consistent between detect (Task 4), enricher (Task 5), reflection (Task 6), mirror (Task 7), liveness (Task 8), SB360/glossary (Task 9).
