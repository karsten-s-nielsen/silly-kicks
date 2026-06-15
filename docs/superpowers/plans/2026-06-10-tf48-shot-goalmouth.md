# TF-48 `add_shot_goalmouth` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline — this
> repo's policy forbids subagents). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Post-shot goalmouth crossing geometry (y, z) + shot kinematics from tracking-frame ball
trajectories, per the approved spec `docs/superpowers/specs/2026-06-10-shot-goalmouth-psxg-design.md`.

**Architecture:** Pure per-shot fit kernel (`_fit_one_shot`) → pure batched engine
(`compute_shot_goalmouth`) → `@nan_safe_enrichment add_shot_goalmouth` edge in `features.py`
(ADR-025 engine/edge split). Goal ends resolved orientation-agnostically from the GK map
(extracted `defended_goal_x`); output canonicalized to attacked-goal-at-x=105. Atomic mirror is a
thin delegation (engine consumes no action coordinates). No VAEP xfns (leakage; guard test).

**Tech Stack:** pandas/numpy only (no new deps). `statsbombpy` (optional, importorskip) for the
owner-gated SB validation e2e.

**REPO POLICY OVERRIDES of the generic skill template (these win):**
- **ONE commit per branch, only after explicit user approval and after `/final-review`.** Every
  "Commit" step in the generic template is replaced by "run the relevant tests". Do NOT commit
  per task.
- **No worktrees.** Branch `pr-s93-shot-goalmouth` off `main`.
- **Session start:** `pip install -e ".[test]"`.
- Version: **4.23.0** (verify nothing shipped meanwhile: `git log origin/main -1`). PR-S93.
- Shift Left: `ruff format --check .` AND `ruff check .` AND `pyright silly_kicks/` (FULL package)
  AND `python -m pytest tests/ -m "not e2e" --tb=short` all green BEFORE declaring done.

**Spec is the contract.** Where this plan and the spec disagree, the spec wins; flag the conflict
instead of improvising.

---

## File map

| File | Action | Responsibility |
|---|---|---|
| `silly_kicks/tracking/_gk_resolve.py` | Modify | gains the extracted `defended_goal_x` (from `_xshot_occurrence.py`, byte-identical) |
| `silly_kicks/tracking/_xshot_occurrence.py` | Modify | `_defended_goal_x` becomes a re-import shim |
| `silly_kicks/tracking/_shot_goalmouth.py` | Create | `ShotGoalmouthParams`, `_fit_one_shot` + sub-kernels, `compute_shot_goalmouth`, `ShotGoalmouthReport` |
| `silly_kicks/tracking/features.py` | Modify | `add_shot_goalmouth` aggregator + per-Series wrappers + `__all__` |
| `silly_kicks/tracking/__init__.py` | Modify | export `add_shot_goalmouth`, `compute_shot_goalmouth`, `ShotGoalmouthParams`, `ShotGoalmouthReport` |
| `silly_kicks/atomic/tracking/features.py` | Modify | atomic mirror `add_shot_goalmouth` + per-Series wrappers |
| `tests/tracking/test_shot_goalmouth.py` | Create | kernel + engine + aggregator unit/behavioral suite |
| `tests/tracking/test_shot_goalmouth_no_xfns_guard.py` | Create | no-default-xfns leakage guard |
| `tests/atomic/tracking/test_shot_goalmouth_atomic.py` | Create | atomic mirror + parity |
| `tests/tracking/conftest_id_dtype.py` | Modify | register the new aggregator in `AGGREGATORS` |
| `tests/tracking/test_gk_resolve_goal_map.py` | Create | extraction byte-identity + N1-fallback tests |
| `scripts/validate_shot_goalmouth_sb.py` | Create | owner-run GS↔SB WC2022 validation harness (pilot + held-out) |
| `tests/tracking/test_shot_goalmouth_sb_e2e.py` | Create | `@pytest.mark.e2e` wrapper invoking the harness protocol |
| `docs/superpowers/adrs/ADR-030-*.md` | Create | decision record (number re-verified at PR time) |
| `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, `TODO.md`, `CLAUDE.md` | Modify | version-bump hard gate (all four version sites must match) + docs |

---

### Task 0: Branch + environment

- [ ] **Step 0.1:** `git checkout main; git pull; git checkout -b pr-s93-shot-goalmouth`
- [ ] **Step 0.2:** `pip install -e ".[test]"`
- [ ] **Step 0.3:** Baseline sanity: `python -m pytest tests/ -m "not e2e and not slow" -x -q --tb=short`
      Expected: PASS (record runtime for later comparison).

### Task 1: Extract `defended_goal_x` into `_gk_resolve.py` (byte-identical, National-Park)

**Files:** Modify `silly_kicks/tracking/_gk_resolve.py`, `silly_kicks/tracking/_xshot_occurrence.py`;
Create `tests/tracking/test_gk_resolve_goal_map.py`.

- [ ] **Step 1.1: Write the failing tests**

```python
"""Goal-end map extraction (TF-48 prerequisite; spec section 5.1)."""

import pandas as pd

from silly_kicks.tracking._gk_resolve import defended_goal_x


def _frames(gk_x_a=5.0, gk_x_b=100.0, with_gk=True):
    rows = []
    for pid, team, gk, x in [(1, "A", True, gk_x_a), (2, "A", False, 40.0),
                             (3, "B", True, gk_x_b), (4, "B", False, 60.0)]:
        if not with_gk and gk:
            continue
        rows.append(dict(game_id=1, period_id=1, frame_id=0, time_seconds=0.0,
                         player_id=pid, team_id=team, is_ball=False, is_goalkeeper=gk,
                         x=x, y=34.0, z=0.0))
    rows.append(dict(game_id=1, period_id=1, frame_id=0, time_seconds=0.0, player_id=None,
                     team_id=None, is_ball=True, is_goalkeeper=False, x=50.0, y=34.0, z=0.0))
    return pd.DataFrame(rows)


def test_gk_based_resolution():
    m = defended_goal_x(_frames())
    assert m[(1, 1, "A")] == 0.0 and m[(1, 1, "B")] == 105.0


def test_outfield_fallback_when_no_gk():
    m = defended_goal_x(_frames(with_gk=False))
    assert m[(1, 1, "A")] == 0.0 and m[(1, 1, "B")] == 105.0  # N1 fallback


def test_xs_shim_is_same_object():
    from silly_kicks.tracking._xshot_occurrence import _defended_goal_x
    assert _defended_goal_x is defended_goal_x
```

- [ ] **Step 1.2:** `python -m pytest tests/tracking/test_gk_resolve_goal_map.py -v` → FAIL
      (ImportError: no `defended_goal_x` in `_gk_resolve`).
- [ ] **Step 1.3: Move the function.** Cut `_defended_goal_x` from `_xshot_occurrence.py:553-567`
      VERBATIM (docstring, `.astype(bool)` calls and all — spec §5.1: do NOT "fix" the
      `.astype(bool)`; schema-real bools, byte-identical extraction) into `_gk_resolve.py`,
      renamed public-within-package `defended_goal_x` (module-level, after
      `defending_gk_from_frames`). Add to the docstring:
      `Extracted from _xshot_occurrence (TF-48, spec 2026-06-10-shot-goalmouth-psxg) — byte-identical.`
      In `_xshot_occurrence.py` replace the function body with a shim:

```python
from silly_kicks.tracking._gk_resolve import defended_goal_x as _defended_goal_x  # noqa: F401
```

      (placed in the import block; delete the old def; all internal call sites
      `_defended_goal_x(...)` keep working unchanged).
- [ ] **Step 1.4:** `python -m pytest tests/tracking/test_gk_resolve_goal_map.py tests/tracking/ -k "xshot" -q` → PASS
      (the existing xS suite is the byte-identity gate — it pins behavior through the shim).

### Task 2: `ShotGoalmouthParams`

**Files:** Create `silly_kicks/tracking/_shot_goalmouth.py`; Create `tests/tracking/test_shot_goalmouth.py`.

- [ ] **Step 2.1: Write the failing tests** (start `tests/tracking/test_shot_goalmouth.py`):

```python
"""TF-48 post-shot goalmouth geometry (spec 2026-06-10-shot-goalmouth-psxg-design)."""

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._shot_goalmouth import ShotGoalmouthParams


class TestParams:
    def test_defaults(self):
        p = ShotGoalmouthParams()
        assert p.post_window_seconds == 2.0 and p.min_fit_frames == 3
        assert p.rolling_z_max_m == 0.3 and p.bounce_min_dz_m == 0.25

    @pytest.mark.parametrize("kw", [
        {"post_window_seconds": 0.0}, {"post_window_seconds": -1.0},
        {"min_fit_frames": 1}, {"break_residual_m": 0.0},
        {"break_speed_drop_frac": 1.5}, {"max_time_to_plane_seconds": 0.0},
        {"rolling_z_max_m": -0.1}, {"bounce_min_dz_m": 0.0},
        {"on_target_tolerance_m": -0.01},
    ])
    def test_post_init_rejects(self, kw):
        with pytest.raises(ValueError):
            ShotGoalmouthParams(**kw)

    def test_frozen(self):
        with pytest.raises(Exception):
            ShotGoalmouthParams().min_fit_frames = 5  # type: ignore[misc]
```

- [ ] **Step 2.2:** Run → FAIL (module missing).
- [ ] **Step 2.3: Implement** module header + params in `_shot_goalmouth.py`:

```python
"""Post-shot goalmouth crossing geometry (TF-48, ADR-030).

Fits the post-contact ball trajectory from tracking frames for shot actions and
derives the goal-plane crossing (y, z), kinematics, and provenance. Pure geometry,
no model. Engine is orientation-agnostic (goal ends from the GK map); output is
canonicalized to attacked-goal-at-x=105 (full point reflection x->105-x, y->68-y).
NOT for VAEP features (post-contact outcome leakage; see ADR-030 + guard test).

See NOTICE for full bibliographic citations (Anzer & Bauer 2021 — xGOT lineage).
Spec: docs/superpowers/specs/2026-06-10-shot-goalmouth-psxg-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from silly_kicks.spadl import config as spadlconfig

_G = 9.81
_FIELD_LENGTH = spadlconfig.field_length  # 105.0
_FIELD_WIDTH = spadlconfig.field_width  # 68.0
_GOAL_Y_C = _FIELD_WIDTH / 2.0  # 34.0
_GOAL_HALF_MOUTH = 7.32 / 2.0  # 3.66
_BAR_Z = 2.44
_REFINE_WINDOW_S = 0.3  # contact-refinement search half-window (spec section 6)
_REFINE_SPEED_JUMP_MS = 3.0  # provisional noise floor for a "qualifying" speed increase
_PRE_SECONDS = 0.3  # pre-window pulled ONLY for contact refinement
# PILOT NOTE: _REFINE_SPEED_JUMP_MS, the reversal floor (s1 > 1.0 in _grow_segment) and the
# engine's truncation slack (0.5 s) are deliberately MODULE constants, not params (speculative
# API surface is debt) — but they ARE on the pilot's sensitivity checklist (Task 8.3): at
# SkillCorner's 10 fps (~3 m/frame at 30 m/s) they mean different things than at 25 fps.
# Promote to ShotGoalmouthParams ONLY if the pilot shows per-corpus tuning is needed.

STANDARD_SHOT_TYPE_IDS = frozenset(
    spadlconfig.actiontype_id[n] for n in ("shot", "shot_freekick", "shot_penalty")
)


@dataclass(frozen=True)
class ShotGoalmouthParams:
    """Tuning surface for the post-shot trajectory fit. Defaults are PROVISIONAL
    pending the SB-WC2022 pilot (spec section 10.4), incl. a per-frame-rate
    sensitivity row (SkillCorner 10 fps).

    Examples
    --------
    >>> ShotGoalmouthParams(post_window_seconds=1.5).post_window_seconds
    1.5
    """

    post_window_seconds: float = 2.0
    min_fit_frames: int = 3
    break_residual_m: float = 0.75
    break_speed_drop_frac: float = 0.5
    max_time_to_plane_seconds: float = 3.0
    # ONE ground band, deliberately shared: "rolling" classification AND the bounce
    # detector's z-at-flip ceiling (a bounce is by definition a near-ground event).
    rolling_z_max_m: float = 0.3
    bounce_min_dz_m: float = 0.25  # hysteresis: min drop-then-rise around a vz flip
    on_target_tolerance_m: float = 0.11  # ball radius; post/bar width folded in BY DECISION
    contact_refinement: bool = True

    def __post_init__(self) -> None:
        for name in (
            "post_window_seconds", "break_residual_m", "max_time_to_plane_seconds",
            "bounce_min_dz_m",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"ShotGoalmouthParams.{name} must be > 0")
        for name in ("rolling_z_max_m", "on_target_tolerance_m"):
            if getattr(self, name) < 0:
                raise ValueError(f"ShotGoalmouthParams.{name} must be >= 0")
        if not 0.0 < self.break_speed_drop_frac < 1.0:
            raise ValueError("ShotGoalmouthParams.break_speed_drop_frac must be in (0, 1)")
        if self.min_fit_frames < 2:
            raise ValueError("ShotGoalmouthParams.min_fit_frames must be >= 2")
```

- [ ] **Step 2.4:** `python -m pytest tests/tracking/test_shot_goalmouth.py -v` → PASS.

### Task 3: Per-shot fit kernel `_fit_one_shot`

**Files:** Modify `silly_kicks/tracking/_shot_goalmouth.py`; Modify `tests/tracking/test_shot_goalmouth.py`.

The kernel is a pure function over one shot's ball samples **in frame coordinates**:

```python
_fit_one_shot(t, x, y, z, goal_x, params) -> dict
```

`t` = seconds relative to the EVENT time (window includes `[-_PRE_SECONDS, post_window_seconds]`),
arrays sorted by `t`; `goal_x` ∈ {0.0, 105.0} = attacked goal in frame coords; `z` may be all-NaN.
Returns a dict with FRAME-coordinate results (canonicalization is the engine's job):
`crossing_y`, `crossing_z`, `speed`, `time_to_goal_line`, `source`, `end_reason`, `z_profile`,
`n_fit_frames`, `fit_rmse` (NaN/None where not applicable).

- [ ] **Step 3.1: Write the synthetic-trajectory factory + first failing tests** (append to the
      test file). The factory builds closed-form trajectories so every expected value is exact:

```python
from silly_kicks.tracking._shot_goalmouth import _fit_one_shot, ShotGoalmouthParams

P = ShotGoalmouthParams()


def traj(vx=25.0, vy=0.0, vz=5.0, x0=85.0, y0=34.0, z0=0.0, fps=25.0, n=12,
         t0=0.0, gravity=True):
    """Ballistic samples: x linear, y linear, z = z0 + vz t - 4.905 t^2 (>=0)."""
    t = t0 + np.arange(n) / fps
    g = 9.81 if gravity else 0.0
    z = np.maximum(z0 + vz * t - 0.5 * g * t**2, 0.0)
    return t, x0 + vx * t, y0 + vy * t, z


class TestFitOneShot:
    def test_straight_drive_extrapolated(self):
        # 25 m/s from x=85 toward 105: crossing at t*=0.8 s; samples cover 0..0.44 s only.
        t, x, y, z = traj(vx=25.0, vy=2.0, vz=4.0, n=12, fps=25.0)
        r = _fit_one_shot(t, x, y, z, goal_x=105.0, params=P)
        assert r["source"] == "extrapolated"
        assert r["crossing_y"] == pytest.approx(34.0 + 2.0 * 0.8, abs=1e-6)
        assert r["crossing_z"] == pytest.approx(max(4.0 * 0.8 - 4.905 * 0.64, 0.0), abs=1e-3)
        assert r["time_to_goal_line"] == pytest.approx(0.8, abs=1e-6)
        assert r["speed"] == pytest.approx(np.hypot(np.hypot(25.0, 2.0), 4.0), rel=0.05)
        assert r["z_profile"] == "airborne" and r["end_reason"] == "window_cap"

    def test_observed_crossing_interpolated(self):
        # samples STRADDLE the plane -> source observed, crossing from interpolation
        t, x, y, z = traj(vx=30.0, x0=95.0, n=12)  # crosses 105 at t=1/3 s (within samples)
        r = _fit_one_shot(t, x, y, z, goal_x=105.0, params=P)
        assert r["source"] == "observed" and r["end_reason"] == "plane_crossed"
        assert r["crossing_y"] == pytest.approx(34.0, abs=1e-6)

    def test_wide_miss_away_from_plane_is_no_crossing(self):
        t, x, y, z = traj(vx=-20.0, x0=85.0)  # moving AWAY from goal_x=105
        r = _fit_one_shot(t, x, y, z, goal_x=105.0, params=P)
        assert r["source"] == "no_crossing"
        assert np.isnan(r["crossing_y"]) and np.isnan(r["crossing_z"])

    def test_too_slow_crossing_is_no_crossing(self):
        t, x, y, z = traj(vx=2.0, x0=85.0)  # t* = 10 s > max_time_to_plane_seconds
        r = _fit_one_shot(t, x, y, z, goal_x=105.0, params=P)
        assert r["source"] == "no_crossing"

    def test_insufficient_frames(self):
        t, x, y, z = traj(n=2)
        r = _fit_one_shot(t, x, y, z, goal_x=105.0, params=P)
        assert r["source"] == "insufficient_frames"
```

- [ ] **Step 3.2:** Run `python -m pytest tests/tracking/test_shot_goalmouth.py::TestFitOneShot -v`
      → FAIL (`_fit_one_shot` undefined).
- [ ] **Step 3.3: Implement the kernel skeleton + sub-kernels** in `_shot_goalmouth.py`:

```python
def _ls_linear(t: np.ndarray, v: np.ndarray) -> tuple[float, float]:
    """Least-squares v = a + b*t. Requires len >= 2."""
    A = np.vstack([np.ones_like(t), t]).T
    (a, b), *_ = np.linalg.lstsq(A, v, rcond=None)
    return float(a), float(b)


def _ls_ballistic_z(t: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """Least-squares z = z0 + vz*t - 0.5*g*t^2 (fixed g). Requires len >= 2."""
    zz = z + 0.5 * _G * t**2
    return _ls_linear(t, zz)


def _grow_segment(t, x, y, goal_x, params):
    """Incremental fit from the first post-contact sample. Returns
    (end_idx_exclusive, end_reason, observed_crossing | None).

    observed_crossing = (cy_frame, cz_idx_pair) when consecutive samples straddle the
    goal plane: the pair (i-1, i) is recorded so the caller can interpolate y AND z
    at the exact plane time.
    """
    n = len(t)
    sign0 = np.sign(x[1] - x[0]) if n >= 2 else 0.0  # the shot's INITIAL x direction
    for i in range(1, n):
        # observed plane straddle between i-1 and i?
        if (x[i - 1] - goal_x) * (x[i] - goal_x) <= 0 and x[i] != x[i - 1]:
            return i + 1, "plane_crossed", (i - 1, i)
        if i >= 2:
            # residual of the newest sample vs fit on [0, i)
            x0, vx = _ls_linear(t[:i], x[:i])
            y0, vy = _ls_linear(t[:i], y[:i])
            px, py = x0 + vx * t[i], y0 + vy * t[i]
            if np.hypot(x[i] - px, y[i] - py) > params.break_residual_m:
                return i, "trajectory_break", None
            # consecutive horizontal-speed drop / direction reversal
            dt1 = t[i] - t[i - 1]
            dt0 = t[i - 1] - t[i - 2]
            s1 = np.hypot(x[i] - x[i - 1], y[i] - y[i - 1]) / dt1
            s0 = np.hypot(x[i - 1] - x[i - 2], y[i - 1] - y[i - 2]) / dt0
            vx1 = (x[i] - x[i - 1]) / dt1
            if s0 > 0 and s1 < s0 * (1.0 - params.break_speed_drop_frac):
                return i, "trajectory_break", None
            # reversal vs the shot's OWN initial direction (NOT vs the goal: a shot that
            # starts away from the attacked plane — own goal / mishit — must not break
            # here; it falls through to the fit and resolves as no_crossing)
            if sign0 != 0 and vx1 * sign0 < 0 and s1 > 1.0:
                return i, "trajectory_break", None
    return n, "window_cap", None
```

      (`"window_cap"` doubles for data-end: the caller maps it to `"data_end"` when the window
      was cut short by the slice rather than by `post_window_seconds` — see Step 3.7.)
- [ ] **Step 3.4: Implement `_refine_contact`** (spec §6 — FIRST qualifying discontinuity):

```python
def _refine_contact(t, x, y, goal_x, params) -> float:
    """First SHOT-CONSISTENT kinematic discontinuity in [-w, +w]: a horizontal speed
    INCREASE >= _REFINE_SPEED_JUMP_MS whose post-discontinuity vx points toward the
    attacked goal. Largest-discontinuity selection is REJECTED by spec (a close-range
    save inside the window would win). Returns the refined t0 (0.0 if none qualifies).
    """
    if not params.contact_refinement or len(t) < 3:
        return 0.0
    toward = 1.0 if goal_x > 50.0 else -1.0
    in_win = (t >= -_REFINE_WINDOW_S) & (t <= _REFINE_WINDOW_S)
    idx = np.where(in_win)[0]
    for k in range(1, len(idx) - 1):
        i = idx[k]
        dt0, dt1 = t[i] - t[i - 1], t[i + 1] - t[i]
        if dt0 <= 0 or dt1 <= 0:
            continue
        s_before = np.hypot(x[i] - x[i - 1], y[i] - y[i - 1]) / dt0
        s_after = np.hypot(x[i + 1] - x[i], y[i + 1] - y[i]) / dt1
        vx_after = (x[i + 1] - x[i]) / dt1
        if s_after - s_before >= _REFINE_SPEED_JUMP_MS and vx_after * toward > 0:
            return float(t[i])
    return 0.0
```

- [ ] **Step 3.5: Implement `_classify_z`** (spec §6 — rolling / airborne / bounced + hysteresis):

```python
def _classify_z(t, z, params):
    """-> (profile, z_seg_start_idx). profile in {"rolling","airborne","bounced"} or None
    (z unusable: all-NaN or < 2 finite samples). z_seg_start_idx = first index of the
    LATEST z-sub-segment (0 unless bounced).

    Bounce = finite-difference vz sign flip (- -> +) at sample k where
    (i) z[k] <= rolling_z_max_m (near ground) AND
    (ii) drop >= bounce_min_dz_m before k AND rise >= bounce_min_dz_m after k (hysteresis)
    — a noisy airborne trajectory whose vz flips at height stays "airborne".
    """
    ok = np.isfinite(z)
    if ok.sum() < 2:
        return None, 0
    if np.nanmax(z) <= params.rolling_z_max_m:
        return "rolling", 0
    vz = np.diff(z) / np.diff(t)
    start = 0
    bounced = False
    for k in range(1, len(vz)):
        if not (vz[k - 1] < 0 <= vz[k]):
            continue
        if z[k] > params.rolling_z_max_m:
            continue  # flip at height = noise, not a bounce
        drop = np.nanmax(z[start:k + 1]) - z[k]
        rise = np.nanmax(z[k:]) - z[k]
        if drop >= params.bounce_min_dz_m and rise >= params.bounce_min_dz_m:
            bounced, start = True, k  # recurse to the LATEST bounce
    if bounced:
        sub = z[start:]
        if np.nanmax(sub) <= params.rolling_z_max_m:
            return "rolling", start  # degenerated to rolling
        return "bounced", start
    return "airborne", 0
```

- [ ] **Step 3.6: Implement `_fit_one_shot`** assembling the above. Behavior contract (spec §6/§7
      — the M-1 per-column segment provenance is load-bearing):

```python
def _fit_one_shot(t, x, y, z, *, goal_x, params, window_truncated=False) -> dict:
    out = dict(crossing_y=np.nan, crossing_z=np.nan, speed=np.nan,
               time_to_goal_line=np.nan, source="no_ball_frames", end_reason=None,
               z_profile=None, n_fit_frames=0, fit_rmse=np.nan)
    if len(t) == 0:
        return out
    t0 = _refine_contact(t, x, y, goal_x, params)
    post = t >= t0
    t, x, y, z = t[post] - t0, x[post], y[post], z[post]
    if len(t) < 2:
        out["source"] = "insufficient_frames"
        return out
    end, reason, straddle = _grow_segment(t, x, y, goal_x, params)
    if reason == "window_cap" and window_truncated:
        reason = "data_end"
    out["end_reason"] = reason
    ts, xs, ys, zs = t[:end], x[:end], y[:end], z[:end]

    observed = straddle is not None
    if not observed and len(ts) < params.min_fit_frames:
        out["source"] = "insufficient_frames"
        return out

    # --- speed: ALWAYS the EARLIEST (contact) sub-segment (M-1) ---
    profile, zstart = _classify_z(ts, zs, params)
    out["z_profile"] = profile
    # pre-bounce sub-segment when it can support a 2-param fit; else the full segment
    # (graceful degradation, documented: a 1-sample pre-bounce cannot anchor a speed fit)
    contact_end = zstart if (zstart >= 2) else len(ts)
    cts, cxs, cys = ts[:contact_end], xs[:contact_end], ys[:contact_end]
    if len(cts) >= 2:
        _, vx0 = _ls_linear(cts, cxs)
        _, vy0 = _ls_linear(cts, cys)
        speed_h = float(np.hypot(vx0, vy0))
        czs = zs[:contact_end]
        if profile is not None and np.isfinite(czs).sum() >= 2:
            f = np.isfinite(czs)
            _, vz0 = _ls_ballistic_z(cts[f], czs[f])
            out["speed"] = float(np.hypot(speed_h, vz0))
        else:
            out["speed"] = speed_h  # 2D fallback (documented degradation)

    # --- crossing: the segment that PRODUCES it ---
    if observed:
        i, j = straddle
        frac = (goal_x - xs[i]) / (xs[j] - xs[i])
        out["crossing_y"] = float(ys[i] + frac * (ys[j] - ys[i]))
        t_star = float(ts[i] + frac * (ts[j] - ts[i]))
        if np.isfinite(zs[i]) and np.isfinite(zs[j]):
            out["crossing_z"] = max(float(zs[i] + frac * (zs[j] - zs[i])), 0.0)
        out["time_to_goal_line"] = t_star
        out["source"] = "observed"
        out["n_fit_frames"] = int(end)
        out["fit_rmse"] = _rmse_xy(ts, xs, ys) if end >= 2 else np.nan
        return out

    # extrapolated: pick the producing segment (post-bounce supersession, 3 branches)
    seg = slice(0, end)
    if profile == "bounced":
        post_n = end - zstart
        if post_n >= params.min_fit_frames:
            seg = slice(zstart, end)  # full x/y+z supersession
        # elif post_n >= 2: z-only refit (handled below); x/y stays full-segment
    fts, fxs, fys = ts[seg], xs[seg], ys[seg]
    x0, vx = _ls_linear(fts, fxs)
    y0, vy = _ls_linear(fts, fys)
    # a fit RAN — populate the diagnostics even when no crossing results (R4b: consumers
    # must not read NA as "no fit ran"; NA stays reserved for the truly-unfitted sources)
    out["n_fit_frames"] = int(seg.stop - seg.start)
    out["fit_rmse"] = _rmse_xy(fts, fxs, fys)
    toward = 1.0 if goal_x > 50.0 else -1.0
    if vx * toward <= 0:
        out["source"] = "no_crossing"
        return out
    t_star = (goal_x - x0) / vx
    if t_star <= float(fts[0]) or t_star > params.max_time_to_plane_seconds:
        out["source"] = "no_crossing"
        return out
    out["crossing_y"] = float(y0 + vy * t_star)
    out["time_to_goal_line"] = float(t_star)
    # crossing z by profile
    if profile == "rolling":
        out["crossing_z"] = float(np.nanmean(zs))
    elif profile in ("airborne", "bounced"):
        zseg = slice(zstart, end) if profile == "bounced" else seg
        zt, zv = ts[zseg], zs[zseg]
        f = np.isfinite(zv)
        if f.sum() >= 2:
            z0c, vz = _ls_ballistic_z(zt[f], zv[f])
            out["crossing_z"] = max(float(z0c + vz * t_star - 0.5 * _G * t_star**2), 0.0)
        # else: < 2 z samples in the producing z-sub-segment -> crossing_z stays NaN
    out["source"] = "extrapolated"
    return out


def _rmse_xy(t, x, y) -> float:
    x0, vx = _ls_linear(t, x)
    y0, vy = _ls_linear(t, y)
    return float(np.sqrt(np.mean((x - (x0 + vx * t)) ** 2 + (y - (y0 + vy * t)) ** 2)))
```

- [ ] **Step 3.7:** Run Step 3.1's tests → PASS. Fix discrepancies in the kernel, not the tests
      (expected values are closed-form).
- [ ] **Step 3.8: Add the break/deflection + bounce + refinement test classes** (each is
      red-first against the already-implemented kernel — they pin the M-1/M-2/H-3 contracts):

```python
class TestBreakDetection:
    def test_deflection_ends_segment(self):
        t, x, y, z = traj(vx=25.0, vy=0.0, n=14)
        x[8:], y[8:] = x[7] + np.arange(1, 7) * 0.2, y[7] + np.arange(1, 7) * 1.6  # deflected
        r = _fit_one_shot(t, x, y, z, goal_x=105.0, params=P)
        assert r["end_reason"] == "trajectory_break"
        assert r["source"] == "extrapolated"  # pre-break segment still extrapolates
        assert r["n_fit_frames"] <= 9

    def test_block_short_segment_insufficient(self):
        t, x, y, z = traj(vx=25.0, n=14)
        x[2:] = x[1]  # blocked dead after 2 samples
        r = _fit_one_shot(t, x, y, z, goal_x=105.0, params=ShotGoalmouthParams(contact_refinement=False))
        assert r["source"] == "insufficient_frames"


class TestZProfiles:
    def test_rolling_daisy_cutter(self):
        t, x, y, _ = traj(vx=20.0, n=12)
        z = np.abs(np.sin(np.arange(12))) * 0.1  # never above 0.3
        r = _fit_one_shot(t, x, y, z, goal_x=105.0, params=P)
        assert r["z_profile"] == "rolling"
        assert r["crossing_z"] == pytest.approx(float(np.nanmean(z)), abs=1e-9)

    def test_bounced_full_supersession(self):
        # pre-bounce 5 samples descending; bounce at k=5 (z=0.05); post-bounce 7 ballistic
        fps = 25.0
        t = np.arange(12) / fps
        x, y = 80.0 + 22.0 * t, 34.0 + 0.0 * t
        z = np.concatenate([np.linspace(1.0, 0.05, 6), 0.05 + 3.0 * (t[6:] - t[5]) - 4.905 * (t[6:] - t[5]) ** 2])
        r = _fit_one_shot(t, x, y, z, goal_x=105.0, params=ShotGoalmouthParams(contact_refinement=False))
        assert r["z_profile"] == "bounced"
        assert r["n_fit_frames"] <= 7  # producing segment = post-bounce (M-1)

    def test_noisy_airborne_stays_airborne(self):
        # vz sign flips AT HEIGHT (z ~ 1.5 m) from noise -> must NOT classify bounced (M-2)
        t, x, y, _ = traj(vx=22.0, n=12)
        z = 1.5 + np.array([0.0, 0.1, -0.08, 0.12, -0.1, 0.05, -0.07, 0.1, -0.05, 0.02, -0.04, 0.0])
        r = _fit_one_shot(t, x, y, z, goal_x=105.0, params=ShotGoalmouthParams(contact_refinement=False))
        assert r["z_profile"] == "airborne"

    def test_z_all_nan_degrades_visibly(self):
        t, x, y, _ = traj(vx=25.0, vy=1.0, n=10)
        z = np.full(10, np.nan)
        r = _fit_one_shot(t, x, y, z, goal_x=105.0, params=P)
        assert r["source"] == "extrapolated" and np.isfinite(r["crossing_y"])
        assert np.isnan(r["crossing_z"]) and r["z_profile"] is None
        assert np.isfinite(r["speed"])  # 2D fallback

    def test_bounced_z_only_refit_branch(self):
        # post-bounce sub-segment has exactly 2 samples (>= 2 but < min_fit_frames=3):
        # z refit on the sub-segment, x/y from the FULL segment (spec L-2 branch 2).
        # detector check: flip at k=6 (vz[5]<0<=vz[6]); z[6]=0.05<=0.3; drop=1.15>=0.25;
        # rise=0.45-0.05=0.40>=0.25; sub z=[0.05,0.45], max 0.45>0.3 -> bounced (not rolling)
        fps = 25.0
        t = np.arange(8) / fps
        x, y = 80.0 + 22.0 * t, np.full(8, 34.0)
        z = np.concatenate([np.linspace(1.2, 0.05, 7), [0.45]])
        r = _fit_one_shot(t, x, y, z, goal_x=105.0,
                          params=ShotGoalmouthParams(contact_refinement=False))
        assert r["z_profile"] == "bounced"
        assert r["n_fit_frames"] == 8  # x/y producing segment = FULL segment (no supersession)
        assert np.isfinite(r["crossing_z"])  # z refit on the 2-sample sub-segment

    # NOTE (spec section 6 branch 3, < 2 post-bounce samples -> crossing_z NaN): under this
    # detector the branch is UNREACHABLE BY CONSTRUCTION -- a vz sign flip at k requires
    # z[k-1], z[k], z[k+1] all finite (np.diff over NaN yields NaN, and NaN comparisons are
    # False), so a detected bounce always leaves >= 2 finite samples in the sub-segment.
    # The `f.sum() >= 2` guard in _fit_one_shot is retained DEFENSIVELY (it also covers the
    # airborne-profile path with sparse finite z). Record this proof in ADR-030; the spec's
    # three-branch contract is satisfied as: branch 1 + branch 2 tested, branch 3 documented
    # unreachable with the guard in place.

    def test_dipping_ballistic_closed_form(self):
        # z DESCENDS through the window (past apex): fixed-g fit must recover the
        # closed-form crossing z. vz=8, z0=0 -> apex at 0.815 s; samples 0.4..0.84 s.
        t, x, y, z = traj(vx=20.0, x0=80.0, vz=8.0, n=12, fps=25.0, t0=0.4)
        t = t - 0.4  # window-relative, post-contact-at--0.4 simulation: shift so t0=0
        # crossing at x=105: t* (absolute flight) solves 80+20*(t*+0.4)=105 -> t*=0.85
        r = _fit_one_shot(t, x, y, z, goal_x=105.0,
                          params=ShotGoalmouthParams(contact_refinement=False))
        z_true = 8.0 * 1.25 - 4.905 * 1.25**2  # absolute flight time 0.85+0.4=1.25 s
        assert r["source"] == "extrapolated"
        assert r["crossing_z"] == pytest.approx(max(z_true, 0.0), abs=0.05)

    def test_occlusion_gap_data_end(self):
        # samples stop mid-window (occlusion): window_truncated -> end_reason data_end
        t, x, y, z = traj(vx=20.0, x0=80.0, n=6)
        r = _fit_one_shot(t, x, y, z, goal_x=105.0, params=P, window_truncated=True)
        assert r["end_reason"] == "data_end"
        assert r["source"] == "extrapolated"


class TestContactRefinement:
    def test_refinement_skips_pre_contact_drift(self):
        # 0.2 s of slow drift before contact, then the shot
        fps = 25.0
        pre_t = np.arange(-5, 0) / fps
        shot_t, shot_x, shot_y, shot_z = traj(vx=25.0, n=10, fps=fps)
        t = np.concatenate([pre_t, shot_t])
        x = np.concatenate([85.0 + 0.5 * pre_t, shot_x])
        y = np.concatenate([np.full(5, 34.0), shot_y])
        z = np.concatenate([np.zeros(5), shot_z])
        r = _fit_one_shot(t, x, y, z, goal_x=105.0, params=P)
        assert r["source"] == "extrapolated"
        assert r["speed"] == pytest.approx(np.hypot(25.0, 4.0), rel=0.1)

    def test_refinement_does_not_lock_onto_save(self):
        # shot at t=0 (speed jump +25), SAVE at t=0.2 (huge reversal, inside the window):
        # refinement must pick the FIRST shot-consistent discontinuity (H-3)
        fps = 25.0
        t = np.arange(-2, 8) / fps
        x = np.where(t < 0, 95.0, 95.0 + 25.0 * t)
        x = np.where(t > 0.2, x[np.searchsorted(t, 0.2)] - 10.0 * (t - 0.2), x)
        y, z = np.full_like(t, 34.0), np.zeros_like(t)
        r = _fit_one_shot(t, x, y, z, goal_x=105.0, params=P)
        # the fit segment is the shot, broken at the save; never the post-save track
        assert r["end_reason"] in ("trajectory_break", "plane_crossed")
```

- [ ] **Step 3.9:** `python -m pytest tests/tracking/test_shot_goalmouth.py -v` → PASS. Iterate on
      kernel constants ONLY if a closed-form expectation fails (then re-check against spec §6).

### Task 4: Batched engine `compute_shot_goalmouth`

**Files:** Modify `silly_kicks/tracking/_shot_goalmouth.py`; Modify `tests/tracking/test_shot_goalmouth.py`.

- [ ] **Step 4.1: Write the failing engine tests** — a small synthetic match builder (2 teams,
      GKs at both ends, ball samples implementing a known trajectory around each shot):

```python
from silly_kicks.tracking._shot_goalmouth import compute_shot_goalmouth


def make_match(shot_team="A", period=1, goal_a_x=5.0, goal_b_x=100.0, flip=False,
               vy=2.0, game_id=1):
    """One shot at t=600.0 by `shot_team` toward the opponent's goal. Frames:
    GKs anchored at each end + ball trajectory (25 m/s, +vy, ballistic z).
    flip=True mirrors the WHOLE frame set (x->105-x, y->68-y) — same physical match
    in the opposite global convention (M-2 invariance)."""
    rows = []
    for off in np.arange(-0.4, 1.21, 0.04):  # 25 fps around the shot
        tt = 600.0 + off
        rows.append(dict(game_id=game_id, period_id=period, frame_id=int(tt * 25),
                         time_seconds=tt, frame_rate=25.0, player_id=1, team_id="A",
                         is_ball=False, is_goalkeeper=True, x=goal_a_x, y=34.0, z=0.0))
        rows.append(dict(game_id=game_id, period_id=period, frame_id=int(tt * 25),
                         time_seconds=tt, frame_rate=25.0, player_id=2, team_id="B",
                         is_ball=False, is_goalkeeper=True, x=goal_b_x, y=34.0, z=0.0))
        bt = max(off, 0.0)  # ball waits at the spot, then flies
        bx, by = 85.0 + 25.0 * bt, 30.0 + vy * bt
        bz = max(4.0 * bt - 4.905 * bt**2, 0.0)
        rows.append(dict(game_id=game_id, period_id=period, frame_id=int(tt * 25),
                         time_seconds=tt, frame_rate=25.0, player_id=None, team_id=None,
                         is_ball=True, is_goalkeeper=False, x=bx, y=by, z=bz))
    frames = pd.DataFrame(rows)
    if flip:
        frames["x"], frames["y"] = 105.0 - frames["x"], 68.0 - frames["y"]
    from silly_kicks.spadl import config as spadlconfig
    actions = pd.DataFrame({
        "game_id": [game_id], "action_id": [0], "period_id": [period],
        "time_seconds": [600.0], "team_id": [shot_team], "player_id": [10],
        "start_x": [85.0], "start_y": [30.0], "end_x": [105.0], "end_y": [34.0],
        "type_id": [spadlconfig.actiontype_id["shot"]],  # from config, never a literal
        "result_id": [0], "bodypart_id": [0],
    })
    return actions, frames


class TestComputeEngine:
    def test_home_shot_canonical_outputs(self):
        actions, frames = make_match()
        out = compute_shot_goalmouth(actions, frames)
        r = out.iloc[0]
        # make_match's ball reaches x=115 by window end -> the plane crossing is OBSERVED
        # (straddled samples, exact interpolation) — extrapolation is the KERNEL tests' domain
        assert r["shot_crossing_source"] == "observed"
        # crossing at t*=0.8: y = 30 + 2*0.8 = 31.6 (already attacked-goal-at-105)
        assert r["shot_crossing_y"] == pytest.approx(31.6, abs=0.05)

    def test_orientation_invariance_byte_identical(self):
        a1, f1 = make_match(flip=False)
        a2, f2 = make_match(flip=True)
        # SPADL actions are per-action LTR — identical in both cases by construction
        o1 = compute_shot_goalmouth(a1, f1)
        o2 = compute_shot_goalmouth(a2, f2)
        pd.testing.assert_frame_equal(o1, o2)  # M-2: engine assumes nothing about orientation

    def test_away_team_attacks_low_x_goal(self):
        # team B shoots: attacked goal is A's (x≈0 end). Ball flies toward LOW x.
        actions, frames = make_match(shot_team="B")
        frames.loc[frames["is_ball"], "x"] = 105.0 - frames.loc[frames["is_ball"], "x"]
        frames.loc[frames["is_ball"], "y"] = 68.0 - frames.loc[frames["is_ball"], "y"]
        actions["start_x"], actions["start_y"] = 85.0, 38.0  # action coords stay per-action-LTR
        out = compute_shot_goalmouth(actions, frames)
        r = out.iloc[0]
        assert r["shot_crossing_source"] == "observed"
        # the away shot IS the point reflection of the home shot -> canonical output identical
        assert r["shot_crossing_y"] == pytest.approx(31.6, abs=0.05)

    def test_non_shot_rows_all_nan(self):
        actions, frames = make_match()
        actions["type_id"] = 0  # pass
        out = compute_shot_goalmouth(actions, frames)
        assert out["shot_crossing_y"].isna().all() and out["shot_crossing_source"].isna().all()

    def test_no_ball_frames(self):
        actions, frames = make_match()
        out = compute_shot_goalmouth(actions, frames[~frames["is_ball"].astype(bool)])
        assert out.iloc[0]["shot_crossing_source"] == "no_ball_frames"

    def test_duplicate_frames_deduped(self):
        actions, frames = make_match()
        out_ref = compute_shot_goalmouth(actions, frames)
        dup = pd.concat([frames, frames], ignore_index=True)  # GS dup-frame pathology
        out_dup = compute_shot_goalmouth(actions, dup)
        pd.testing.assert_frame_equal(out_ref, out_dup)

    def test_unresolved_when_goal_map_degenerate(self):
        actions, frames = make_match(goal_a_x=50.0, goal_b_x=50.0)  # both GKs mid-pitch
        frames = frames[~frames["is_ball"].astype(bool)]  # and no ball fallback either
        out = compute_shot_goalmouth(actions, frames)
        assert out.iloc[0]["shot_crossing_source"] in ("unresolved", "no_ball_frames")

    def test_own_goal_is_no_crossing(self):
        # ball flies toward the shooter's OWN goal -> intentional exclusion (spec section 8)
        actions, frames = make_match()
        b = frames["is_ball"].astype(bool)
        frames.loc[b, "x"] = 105.0 - frames.loc[b, "x"]  # reverse ball direction only
        out = compute_shot_goalmouth(actions, frames)
        assert out.iloc[0]["shot_crossing_source"] == "no_crossing"

    def test_nan_team_id_unresolved_no_crash(self):
        actions, frames = make_match()
        actions["team_id"] = pd.array([pd.NA], dtype="object")
        out = compute_shot_goalmouth(actions, frames)  # ADR-003: never crash
        # pinned by same_id's contract (_id_compat.py:146 — "False if either is NA"):
        # NaN action team -> neither goal-map team matches -> 2 candidate ends -> unresolved
        assert out.iloc[0]["shot_crossing_source"] == "unresolved"

    def test_pso_degenerate_positive_path(self):
        # degenerate GK map (both teams classified to the SAME end) + ball present:
        # the ball-mean fallback resolves the attacked end -> a real crossing (spec 5.5)
        actions, frames = make_match(goal_a_x=95.0, goal_b_x=100.0)  # both ends -> 105.0
        out = compute_shot_goalmouth(actions, frames)
        r = out.iloc[0]
        assert r["shot_crossing_source"] == "observed"
        assert r["shot_crossing_y"] == pytest.approx(31.6, abs=0.05)  # ball flies toward 105

    def test_period2_flip_resolved_per_period(self):
        # period 2: GK anchors swap ends; ball trajectory mirrored (the physical second-half
        # shot). The GK map resolves per (game, period) -> canonical output identical to p1.
        actions, frames = make_match(period=2, goal_a_x=100.0, goal_b_x=5.0, flip=False)
        b = frames["is_ball"].astype(bool)
        frames.loc[b, "x"] = 105.0 - frames.loc[b, "x"]
        frames.loc[b, "y"] = 68.0 - frames.loc[b, "y"]
        out = compute_shot_goalmouth(actions, frames)
        r = out.iloc[0]
        assert r["shot_crossing_source"] == "observed"
        assert r["shot_crossing_y"] == pytest.approx(31.6, abs=0.05)


class TestOnTargetDerived:
    """Boundary tests for shot_on_target_derived (the lakehouse PSxG on-target gate).
    Drive the crossing point via the ball trajectory's vy/vz; tolerance = 0.11 m."""

    def _shot_with(self, vy, vz, gravity=True):
        actions, frames = make_match(vy=vy)
        b = frames["is_ball"].astype(bool)
        if not gravity:  # replace z with a linear rise to place crossing z precisely
            bt = np.maximum(frames.loc[b, "time_seconds"].to_numpy() - 600.0, 0.0)
            frames.loc[b, "z"] = vz * bt
        return compute_shot_goalmouth(actions, frames).iloc[0]

    def test_inside_mouth_true(self):
        # crossing y = 30 + 5*0.8 = 34.0 (centre), z linear 1.0*0.8 = 0.8 -> True
        r = self._shot_with(vy=5.0, vz=1.0, gravity=False)
        assert r["shot_on_target_derived"] == True  # noqa: E712

    def test_just_outside_post_false(self):
        # crossing y = 30 + 9.8*0.8 = 37.84 -> 3.84 from centre > 3.66 + 0.11 -> False
        r = self._shot_with(vy=9.8, vz=1.0, gravity=False)
        assert r["shot_on_target_derived"] == False  # noqa: E712

    def test_within_post_tolerance_true(self):
        # crossing y = 30 + 9.6*0.8 = 37.68 -> 3.68 from centre <= 3.66 + 0.11 -> True
        r = self._shot_with(vy=9.6, vz=1.0, gravity=False)
        assert r["shot_on_target_derived"] == True  # noqa: E712

    def test_lob_over_bar_false_with_valid_yz(self):
        # crossing z = 3.5*0.8 = 2.8 > 2.44 + 0.11 -> False, with VALID y and z (spec 11)
        r = self._shot_with(vy=0.0, vz=3.5, gravity=False)
        assert r["shot_on_target_derived"] == False  # noqa: E712
        assert np.isfinite(r["shot_crossing_y"]) and np.isfinite(r["shot_crossing_z"])

    def test_na_when_z_unavailable(self):
        actions, frames = make_match()
        frames.loc[frames["is_ball"].astype(bool), "z"] = np.nan
        r = compute_shot_goalmouth(actions, frames).iloc[0]
        assert pd.isna(r["shot_on_target_derived"])  # bar unknowable -> NA (spec 7)

    def test_na_when_not_resolved(self):
        actions, frames = make_match()
        r = compute_shot_goalmouth(actions, frames[~frames["is_ball"].astype(bool)]).iloc[0]
        assert pd.isna(r["shot_on_target_derived"])  # source no_ball_frames -> NA
```

- [ ] **Step 4.2:** Run → FAIL (`compute_shot_goalmouth` undefined).
- [ ] **Step 4.3: Implement the engine.** Key structure (full implementation; engine is PURE —
      no warnings, never mutates `actions`):

```python
def compute_shot_goalmouth(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    links: pd.DataFrame | None = None,
    params: ShotGoalmouthParams | None = None,
    shot_type_ids: frozenset[int] = STANDARD_SHOT_TYPE_IDS,
) -> pd.DataFrame:
    """PURE engine ... (docstring per spec sections 4-8, with Examples block). ``links``
    accepted for signature parity ONLY -- the ENGINE never reads it (the window is
    time-sliced via slice_around_event, link-independent); the add_* edge uses ``links``
    solely for its provenance-column merge."""
    from silly_kicks.tracking._gk_geometry import _truthy_bool
    from silly_kicks.tracking._gk_resolve import defended_goal_x
    from silly_kicks.tracking._id_compat import same_id
    from silly_kicks.tracking.utils import slice_around_event

    params = params or ShotGoalmouthParams()
    n = len(actions)
    out = pd.DataFrame(index=actions.index)
    fcols = ("shot_crossing_y", "shot_crossing_z", "shot_speed",
             "shot_time_to_goal_line", "shot_fit_rmse")
    for c in fcols:
        out[c] = np.full(n, np.nan)
    out["shot_on_target_derived"] = pd.array([pd.NA] * n, dtype="boolean")
    for c in ("shot_crossing_source", "shot_fit_end_reason", "shot_z_profile"):
        out[c] = pd.array([pd.NA] * n, dtype="object")
    out["shot_crossing_confidence"] = np.full(n, np.nan)
    out["shot_fit_n_frames"] = pd.array([pd.NA] * n, dtype="Int64")
    if "type_id" not in actions.columns or n == 0:
        return out
    is_shot = actions["type_id"].isin(shot_type_ids).to_numpy()
    if not is_shot.any():
        return out

    is_ball = _truthy_bool(frames["is_ball"])
    ball = frames[is_ball].drop_duplicates(["game_id", "period_id", "frame_id"], keep="first")
    goal_map = defended_goal_x(frames)

    shots = actions[is_shot]
    sl = slice_around_event(shots, ball, pre_seconds=_PRE_SECONDS,
                            post_seconds=params.post_window_seconds)
    by_action = dict(iter(sl.groupby("action_id"))) if len(sl) else {}

    for ridx, row in shots.iterrows():
        key_gp = (row["game_id"], row["period_id"])
        # attacked goal = the goal defended by the OTHER team in this (game, period).
        # Three mutually exclusive resolution states:
        #   resolved   — exactly 2 teams, action team identified (same_id), the two ends differ
        #   degenerate — exactly 2 teams but the GK map puts both at the SAME end (PSO case;
        #                spec section 5.5) -> fallback to the end nearer the window's mean ball x
        #   unresolved — anything else (NaN action team -> same_id never matches -> 2 candidate
        #                "opponent" ends; or a malformed (game, period) group)
        ends = {k[2]: v for k, v in goal_map.items() if (k[0], k[1]) == key_gp}
        opp = [v for tid, v in ends.items() if not same_id(tid, row["team_id"])]
        degenerate = len(ends) == 2 and len(set(ends.values())) == 1
        resolved = len(ends) == 2 and len(opp) == 1 and not degenerate
        if not resolved and not degenerate:
            out.loc[ridx, "shot_crossing_source"] = "unresolved"
            out.loc[ridx, "shot_crossing_confidence"] = 0.0
            continue
        g = by_action.get(row["action_id"])
        if g is None or g.empty:
            out.loc[ridx, "shot_crossing_source"] = "no_ball_frames"
            out.loc[ridx, "shot_crossing_confidence"] = 0.0
            continue
        g = g.sort_values("time_offset_seconds")
        t = g["time_offset_seconds"].to_numpy(float)
        xv, yv, zv = (g[c].to_numpy(float) for c in ("x", "y", "z"))
        if resolved:
            goal_x = opp[0]
        else:  # degenerate -> PSO fallback (spec section 5.5)
            goal_x = 0.0 if float(np.nanmean(xv)) < 52.5 else 105.0
        truncated = (t[-1] - max(t[0], 0.0)) < params.post_window_seconds - 0.5 if len(t) else True
        r = _fit_one_shot(t, xv, yv, zv, goal_x=goal_x, params=params,
                          window_truncated=truncated)
        # canonicalize to attacked-goal-at-105 (full point reflection)
        cy = r["crossing_y"] if goal_x > 50.0 else (_FIELD_WIDTH - r["crossing_y"])
        out.loc[ridx, "shot_crossing_y"] = cy
        out.loc[ridx, "shot_crossing_z"] = r["crossing_z"]
        out.loc[ridx, "shot_speed"] = r["speed"]
        out.loc[ridx, "shot_time_to_goal_line"] = r["time_to_goal_line"]
        out.loc[ridx, "shot_crossing_source"] = r["source"]
        out.loc[ridx, "shot_fit_end_reason"] = r["end_reason"]
        out.loc[ridx, "shot_z_profile"] = r["z_profile"]
        out.loc[ridx, "shot_fit_n_frames"] = r["n_fit_frames"] or pd.NA
        out.loc[ridx, "shot_fit_rmse"] = r["fit_rmse"]
        out.loc[ridx, "shot_crossing_confidence"] = _confidence(r, params)
        if r["source"] in ("observed", "extrapolated") and np.isfinite(cy):
            if np.isfinite(r["crossing_z"]):
                tol = params.on_target_tolerance_m
                on = (abs(cy - _GOAL_Y_C) <= _GOAL_HALF_MOUTH + tol
                      and r["crossing_z"] <= _BAR_Z + tol)
                out.loc[ridx, "shot_on_target_derived"] = bool(on)
            # crossing_z NaN -> on-target stays NA (bar unknowable; spec section 7)
    return out


def _confidence(r: dict, params: ShotGoalmouthParams) -> float:
    """PROVISIONAL map (ADR-025 style; calibrated at the SB pilot — spec sections 7/10).
    Inputs include z_profile + producing-segment size because a 2-sample z refit is
    exactly determined (RMSE == 0 would out-score an honest 5-point fit; spec L-1)."""
    if r["source"] == "observed":
        return 1.0
    if r["source"] != "extrapolated":
        return 0.0
    n_term = min(r["n_fit_frames"] / 8.0, 1.0)
    rmse_term = 1.0 / (1.0 + (r["fit_rmse"] / params.break_residual_m if np.isfinite(r["fit_rmse"]) else 1.0))
    z_term = 1.0 if r["z_profile"] in ("airborne", "rolling") else (0.7 if r["z_profile"] == "bounced" else 0.5)
    return float(np.clip(0.9 * n_term * rmse_term * z_term, 0.0, 0.9))
```

- [ ] **Step 4.4:** `python -m pytest tests/tracking/test_shot_goalmouth.py -v` → ALL PASS.
      Where an engine test exposes a kernel gap, fix the kernel and re-run Task 3's tests too.
- [ ] **Step 4.5:** Period-2 flip — already covered by `test_period2_flip_resolved_per_period`
      in Step 4.1 (asserts source `observed` + canonical y identical to period 1). No action.

### Task 5: `ShotGoalmouthReport`

**Files:** Modify `silly_kicks/tracking/_shot_goalmouth.py`; Modify `tests/tracking/test_shot_goalmouth.py`.

- [ ] **Step 5.1: Failing test:**

```python
from silly_kicks.tracking._shot_goalmouth import ShotGoalmouthReport


def test_report_counts():
    actions, frames = make_match()
    out = compute_shot_goalmouth(actions, frames)
    rep = ShotGoalmouthReport.from_frame(out)
    assert rep.n_shots == 1
    assert rep.source_counts == {"observed": 1}
    assert rep.z_profile_counts.get("airborne", 0) == 1
```

- [ ] **Step 5.2:** Implement (mirrors `RestartCoordinateReport`, `tracking/_restart_report.py`):

```python
@dataclass(frozen=True)
class ShotGoalmouthReport:
    """Aggregate provenance QA for shot-goalmouth output (convenience over value_counts;
    z_profile_counts is the corpus-scale bounce-misclassification detector — spec L-3).

    Examples
    --------
    >>> rep = ShotGoalmouthReport.from_frame(enriched)  # doctest: +SKIP
    """

    n_shots: int
    source_counts: dict[str, int]
    end_reason_counts: dict[str, int]
    z_profile_counts: dict[str, int]
    n_on_target_derived: int

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> "ShotGoalmouthReport":
        """Build from a compute_/add_shot_goalmouth frame.

        Examples
        --------
        >>> rep = ShotGoalmouthReport.from_frame(enriched)  # doctest: +SKIP
        """
        def _counts(col: str) -> dict[str, int]:
            return {str(k): int(v) for k, v in df[col].value_counts(dropna=True).items()}

        return cls(
            n_shots=int(df["shot_crossing_source"].notna().sum()),
            source_counts=_counts("shot_crossing_source"),
            end_reason_counts=_counts("shot_fit_end_reason"),
            z_profile_counts=_counts("shot_z_profile"),
            n_on_target_derived=int((df["shot_on_target_derived"] == True).sum()),  # noqa: E712
        )
```

- [ ] **Step 5.3:** Run → PASS.

### Task 6: `add_shot_goalmouth` aggregator + per-Series wrappers + exports

**Files:** Modify `silly_kicks/tracking/features.py`, `silly_kicks/tracking/__init__.py`,
`tests/tracking/conftest_id_dtype.py`; Modify `tests/tracking/test_shot_goalmouth.py`;
Create `tests/tracking/test_shot_goalmouth_no_xfns_guard.py`.

- [ ] **Step 6.1: Failing tests:**

```python
class TestAddAggregator:
    def test_columns_and_passthrough(self):
        actions, frames = make_match()
        from silly_kicks.tracking.features import add_shot_goalmouth
        out = add_shot_goalmouth(actions, frames)
        for c in ("shot_crossing_y", "shot_crossing_z", "shot_speed", "shot_time_to_goal_line",
                  "shot_on_target_derived", "shot_crossing_source", "shot_crossing_confidence",
                  "shot_fit_n_frames", "shot_fit_rmse", "shot_fit_end_reason", "shot_z_profile"):
            assert c in out.columns
        assert len(out) == len(actions)
        pd.testing.assert_frame_equal(out[actions.columns], actions)  # input never mutated

    def test_provenance_idempotent_skip(self):
        actions, frames = make_match()
        from silly_kicks.tracking.features import add_shot_goalmouth
        actions2 = actions.assign(frame_id=1, time_offset_seconds=0.0,
                                  n_candidate_frames=1, link_quality_score=1.0)
        out = add_shot_goalmouth(actions2, frames)
        assert not any(c.endswith(("_x", "_y")) and c.startswith("frame_id") for c in out.columns)

    def test_per_series_wrapper(self):
        actions, frames = make_match()
        from silly_kicks.tracking.features import shot_crossing_y
        s = shot_crossing_y(actions, frames)
        assert isinstance(s, pd.Series) and s.name == "shot_crossing_y"
```

And `tests/tracking/test_shot_goalmouth_no_xfns_guard.py`:

```python
"""TF-48 leakage guard: post-contact outcome descriptors must NEVER enter VAEP
default xfn lists (HybridVAEP leakage class — ADR-030, owner-decided 2026-06-10)."""


def test_no_shot_goalmouth_in_any_default_xfn_list():
    import silly_kicks.atomic.tracking.features as atf
    import silly_kicks.tracking.features as tf
    import silly_kicks.vaep.features as vf

    lists = {
        "tracking_default_xfns": tf.tracking_default_xfns,
        "pre_shot_gk_default_xfns": tf.pre_shot_gk_default_xfns,
        "pre_shot_gk_angle_default_xfns": tf.pre_shot_gk_angle_default_xfns,
        "pre_shot_gk_full_default_xfns": tf.pre_shot_gk_full_default_xfns,
        "atomic_pre_shot_gk_full_default_xfns": atf.pre_shot_gk_full_default_xfns,
        "vaep_xfns_default": vf.xfns_default,
        "vaep_hybrid_xfns_default": vf.hybrid_xfns_default,
    }
    for name, lst in lists.items():
        for fn in lst:
            assert "shot_goalmouth" not in fn.__name__ and "shot_crossing" not in fn.__name__, (
                f"{name} contains a TF-48 function ({fn.__name__}) — post-shot outcome leakage"
            )


def test_no_xfns_factory_exists():
    import silly_kicks.tracking.features as tf
    assert not hasattr(tf, "shot_goalmouth_xfns"), "TF-48 must not ship a VAEP xfns factory"
```

(Before running: verify each default-list name exists with `python -c "import silly_kicks.vaep.features as vf; print(hasattr(vf,'xfns_default'), hasattr(vf,'hybrid_xfns_default'))"` —
adjust the guard to the REAL list names found; the test must enumerate every default list that
exists, per the spec's "absence from every default xfn list".)

- [ ] **Step 6.2:** Run → FAIL.
- [ ] **Step 6.3: Implement in `features.py`** (place next to `add_xt_gk`; import
      `ShotGoalmouthParams`, `compute_shot_goalmouth` from `._shot_goalmouth` in the module's
      import block; `same_id` is already imported at `features.py:57`).
      **Recorded design rejection (cross-session review M1, R2):** the pointer resolution
      deliberately uses the `add_xt_gk`-verbatim
      `links if links is not None else link_actions_to_frames(actions, frames)[0]` path and NOT
      `_resolve_action_frame_context` — the context helper additionally builds actor/opponent
      row joins TF-48 never consumes (wasted per-call work for no benefit). Record this sentence
      in ADR-030 too.

```python
@nan_safe_enrichment
def add_shot_goalmouth(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    links: pd.DataFrame | None = None,
    params: ShotGoalmouthParams | None = None,
) -> pd.DataFrame:
    """Add TF-48 post-shot goalmouth crossing columns (shot_crossing_y/z, shot_speed,
    shot_time_to_goal_line, shot_on_target_derived + provenance shot_crossing_source/
    confidence/fit_n_frames/fit_rmse/fit_end_reason/z_profile) per shot action; NaN/NA
    out-of-scope. Pure geometry over the ball trajectory — NOT a VAEP feature
    (post-contact outcome leakage; ADR-030 guard). NaN identifiers resolve to
    "unresolved" rows (ADR-003 — implemented here, the decorator is marker-only).

    Examples
    --------
    >>> from silly_kicks.tracking.features import add_shot_goalmouth
    >>> enriched = add_shot_goalmouth(actions, frames)
    >>> enriched[["shot_crossing_y", "shot_crossing_z", "shot_crossing_source"]].head()

    See NOTICE for full bibliographic citations (Anzer & Bauer 2021).
    """
    out = actions.copy()
    comp = compute_shot_goalmouth(actions, frames, links=links, params=params)
    for c in comp.columns:
        out[c] = comp[c].to_numpy() if comp[c].dtype != "boolean" else comp[c].array
    # EDGE policy (spec section 4 -- warnings live here, never in the pure engine):
    # a mostly-unresolvable shot set is a data-quality signal (mirrors on_low_coverage).
    src = comp["shot_crossing_source"].dropna()
    if len(src) > 0:
        bad = src.isin(["no_ball_frames", "unresolved"]).mean()
        if bad > 0.5:
            _warnings.warn(  # features.py's existing alias (`import warnings as _warnings`, line ~1962)
                f"add_shot_goalmouth: {bad:.0%} of shot rows could not be resolved "
                "(no ball frames / goal-end unresolved) -- check frames coverage and the "
                "GK map for this match.",
                stacklevel=2,
            )
    provenance_cols = ["frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]
    if not any(c in out.columns for c in provenance_cols):
        pointers = links if links is not None else link_actions_to_frames(actions, frames)[0]
        if len(pointers) > 0:
            ptr_cols = pointers.set_index("action_id")[provenance_cols]
            out = out.merge(ptr_cols, left_on="action_id", right_index=True, how="left")
    return out
```

      Add to `TestAddAggregator` the edge-warning test (M3 — the spec's "warnings on anomalies"
      contract, consciously implemented rather than silently dropped):

```python
    def test_warns_when_mostly_unresolvable(self):
        actions, frames = make_match()
        no_ball = frames[~frames["is_ball"].astype(bool)]
        from silly_kicks.tracking.features import add_shot_goalmouth
        with pytest.warns(UserWarning, match="could not be resolved"):
            add_shot_goalmouth(actions, no_ball)

    def test_no_warning_on_healthy_match(self):
        import warnings as _w
        actions, frames = make_match()
        from silly_kicks.tracking.features import add_shot_goalmouth
        with _w.catch_warnings():
            # scoped to UserWarning (OUR edge contract) — an unscoped "error" filter would
            # also trip on pandas Future/DeprecationWarnings on a future pandas bump
            _w.simplefilter("error", UserWarning)
            add_shot_goalmouth(actions, frames)
```

      Per-Series wrappers (one per §7 value column, `pre_shot_gk_x` style — each with the same
      docstring skeleton + Examples):

```python
def shot_crossing_y(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Goal-plane crossing y (m, canonical attacked-goal-at-x=105). NaN out-of-scope.

    Examples
    --------
    >>> from silly_kicks.tracking.features import shot_crossing_y
    >>> shot_crossing_y(actions, frames).head()

    See NOTICE for full bibliographic citations (Anzer & Bauer 2021).
    """
    return compute_shot_goalmouth(actions, frames)["shot_crossing_y"].rename("shot_crossing_y")
```

      Repeat the same 4-line body pattern for `shot_crossing_z`, `shot_speed`,
      `shot_time_to_goal_line`, `shot_on_target_derived` (each selects/renames its column).
- [ ] **Step 6.4: Exports.** `features.py` `__all__`: add `add_shot_goalmouth`,
      `shot_crossing_y`, `shot_crossing_z`, `shot_speed`, `shot_time_to_goal_line`,
      `shot_on_target_derived`, `ShotGoalmouthParams` (alphabetical slots).
      `tracking/__init__.py`: add `add_shot_goalmouth`, `compute_shot_goalmouth`,
      `ShotGoalmouthParams`, `ShotGoalmouthReport` to the import block + `__all__`
      (match the `add_xt_gk` lines at `__init__.py:97/298`).
- [ ] **Step 6.5: Register in the id-dtype gate.** `tests/tracking/conftest_id_dtype.py`
      `AGGREGATORS` list, alphabetical slot:

```python
    _a(F.add_shot_goalmouth, "add_shot_goalmouth"),
```

- [ ] **Step 6.6: Run the auto-gates** (memory: auto-enumerating gates — run them, don't assume):

```
python -m pytest tests/tracking/test_shot_goalmouth.py tests/tracking/test_shot_goalmouth_no_xfns_guard.py -v
python -m pytest tests/tracking/test_id_dtype_invariance.py -v
python -m pytest tests/test_enrichment_nan_safety.py -v
python -m pytest tests/test_public_api_examples.py -v
python -m pytest tests/tracking/test_id_compat_lint.py -v
```

      Expected: ALL PASS. The id-dtype gate runs the new aggregator on numeric-actions ×
      string-frames and the reverse — if it fails, the fix is `_id_compat` routing in the engine
      (`same_id` at the team match — VERIFIED real: `_id_compat.py:146`, scalar↔scalar, "False if
      either is NA"), never test relaxation. The nan-safety fuzz auto-discovers the decorator.
      The Examples gate auto-discovers the new public defs and is PRESENCE-ONLY (verified:
      `tests/test_public_api_examples.py:124` — "no doctest verification is required"), so the
      `add_xt_gk`-style Examples blocks need no doctest setup/+SKIP. The AST lint checks the
      engine compares ids only through `_id_compat`.

### Task 7: Atomic mirror

**Files:** Modify `silly_kicks/atomic/tracking/features.py`;
Create `tests/atomic/tracking/test_shot_goalmouth_atomic.py`.

- [ ] **Step 7.1: Failing tests:**

```python
"""TF-48 atomic mirror: thin delegation (engine consumes NO action coordinates);
atomic shot domain is {shot, shot_penalty} — shot_freekick is a `freekick` atom
(atomic/spadl/base.py:274-278, existing pre-shot-GK precedent)."""

import pandas as pd
import pandas.testing as pdt

from silly_kicks.atomic.spadl import config as atomicconfig
from silly_kicks.atomic.tracking.features import add_shot_goalmouth as atomic_add
from silly_kicks.tracking.features import add_shot_goalmouth as std_add
from tests.tracking.test_shot_goalmouth import make_match


def _to_atomic(actions: pd.DataFrame) -> pd.DataFrame:
    a = actions.copy()
    a["x"], a["y"] = a.pop("start_x"), a.pop("start_y")
    a["dx"] = actions["end_x"] - actions["start_x"]
    a["dy"] = actions["end_y"] - actions["start_y"]
    a = a.drop(columns=["end_x", "end_y", "result_id"])
    a["type_id"] = atomicconfig.actiontype_id["shot"]
    return a


def test_parity_with_standard_on_shot_rows():
    actions, frames = make_match()
    std = std_add(actions, frames)
    atm = atomic_add(_to_atomic(actions), frames)
    cols = [c for c in std.columns if c.startswith("shot_")]
    pdt.assert_frame_equal(std[cols].reset_index(drop=True),
                           atm[cols].reset_index(drop=True))


def test_atomic_domain_excludes_freekick_atoms():
    actions, frames = make_match()
    a = _to_atomic(actions)
    a["type_id"] = atomicconfig.actiontype_id["freekick"]  # direct FK shot in atomic space
    out = atomic_add(a, frames)
    assert out["shot_crossing_source"].isna().all()
```

- [ ] **Step 7.2:** Run → FAIL.
- [ ] **Step 7.3: Implement** in `atomic/tracking/features.py` (next to the other mirrors; uses
      the module's existing `_ATOMIC_SHOT_TYPE_IDS` at line 52; import `ShotGoalmouthParams` +
      `compute_shot_goalmouth` from `silly_kicks.tracking._shot_goalmouth`):

```python
@nan_safe_enrichment
def add_shot_goalmouth(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    links: pd.DataFrame | None = None,
    params: ShotGoalmouthParams | None = None,
) -> pd.DataFrame:
    """Atomic-SPADL mirror of tracking.features.add_shot_goalmouth (TF-48). NO coordinate
    synthesis: the engine consumes only action_id/game_id/period_id/time_seconds/team_id/
    type_id (trajectory from frames, goal end from the GK map). Atomic shot domain is
    {shot, shot_penalty} (shot_freekick is a `freekick` atom — intentional, existing
    pre-shot-GK precedent).

    Examples
    --------
    >>> from silly_kicks.atomic.tracking.features import add_shot_goalmouth
    >>> enriched = add_shot_goalmouth(atomic_actions, frames)
    >>> enriched[["shot_crossing_y", "shot_crossing_source"]].head()

    See NOTICE for full bibliographic citations (Anzer & Bauer 2021).
    """
    out = actions.copy()
    comp = compute_shot_goalmouth(actions, frames, links=links, params=params,
                                  shot_type_ids=_ATOMIC_SHOT_TYPE_IDS)
    for c in comp.columns:
        out[c] = comp[c].to_numpy() if comp[c].dtype != "boolean" else comp[c].array
    provenance_cols = ["frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]
    if not any(c in out.columns for c in provenance_cols):
        pointers = links if links is not None else link_actions_to_frames(actions, frames)[0]
        if len(pointers) > 0:
            ptr_cols = pointers.set_index("action_id")[provenance_cols]
            out = out.merge(ptr_cols, left_on="action_id", right_index=True, how="left")
    return out
```

      Mirror the 5 per-Series wrappers the same way (each calls `compute_shot_goalmouth(...,
      shot_type_ids=_ATOMIC_SHOT_TYPE_IDS)` and selects its column). Add all names to the atomic
      module's `__all__`.
- [ ] **Step 7.4:** `python -m pytest tests/atomic/tracking/test_shot_goalmouth_atomic.py tests/test_public_api_examples.py tests/test_enrichment_nan_safety.py -v` → PASS.

### Task 8: SB validation harness (owner-gated e2e) + DGX pilot

**Files:** Create `scripts/validate_shot_goalmouth_sb.py`, `tests/tracking/test_shot_goalmouth_sb_e2e.py`.

- [ ] **Step 8.1: Harness script** `scripts/validate_shot_goalmouth_sb.py` — owner-run (precedent:
      `scripts/_xtgk_comparability.py`). Structure (full protocol from spec §10; CLI:
      `--matches pilot|holdout|all --provider gs --out report.json`):
      1. Load GS WC2022 events→SPADL + tracking→frames via the pining loaders
         (`scripts/_loader_pining.py` — DGX-canonical; never local).
      2. `add_shot_goalmouth` per match; collect `ShotGoalmouthReport` per provider.
      3. Pull SB open data per match via `statsbombpy` (`sb.matches(competition_id=43,
         season_id=106)`; map GS↔SB matches by team names + kickoff date).
      4. **Outcome-literal runtime assert** (spec L-4): `vocab = set(sb_shots["shot_outcome"])`;
         the on-target literal set used MUST be a subset, else `raise AssertionError(vocab)` —
         fail loud, never zero-match.
      5. Shot matching GS↔SB: per (match, period), nearest game-clock within ±10 s, same team;
         tie-breaker ordering documented in the module docstring: (1) clock distance, (2) same
         player if resolvable, (3) UNMATCHED (ambiguous → unmatched report, never best-effort).
      6. **Handedness settlement BEFORE floors** (spec §7/§10.2): on goals, regress sign of
         `(crossing_y − 34)` vs sign of SB `(end_location_y − 40)`; pick the sign with majority
         agreement; ASSERT agreement ≥ 0.9 after picking, else abort with a report.
      7. **Stratified comparison** (spec §10.3): Δy/Δz distributions for GOALS and SAVES
         separately (meters, after the §7 transform with the settled sign); coverage by source;
         z_profile counts; per-frame-rate sensitivity row (re-run the fit with
         `break_residual_m` ∈ {0.5, 0.75, 1.0, 1.5} on a SkillCorner-downsampled copy of the GS
         samples — 10 fps simulation by taking every 3rd frame); raw-z vs smoothed-z comparison
         (GS `ballsSmoothed` — load both ball variants and diff the fits).
      8. Emit `report.json` + a printed summary table.
- [ ] **Step 8.2: e2e test** `tests/tracking/test_shot_goalmouth_sb_e2e.py` — thin
      `@pytest.mark.e2e` wrapper: `statsbombpy = pytest.importorskip("statsbombpy")`; skips
      without the pining env token (follow `tests/spadl/test_gradientsports_scoreline_e2e.py`
      gating pattern); asserts the harness's pilot-subset run completes and the
      PRE-REGISTERED floors (read from the ADR-recorded constants once set) hold on goals.
      Until floors are registered (Step 8.4) the floor assert is
      `pytest.skip("floors not yet pre-registered — pilot pending")` — an explicit skip with
      reason, never a silent pass.
- [ ] **Step 8.3: DGX pilot run** (coordination step — DGX-canonical compute policy):
      `ssh karsten@192.168.68.73`, sync branch, run
      `python scripts/validate_shot_goalmouth_sb.py --matches pilot --out pilot.json` on a
      STRATIFIED 16-match pilot subset: 12 group-stage + 4 knockout matches (knockout is where
      ET periods 3/4 and tired-leg pathologies live), lowest match ids within each stratum —
      deterministic, documented in the script. Review the error distributions; calibrate
      `ShotGoalmouthParams` defaults + the `_confidence` map per spec §10.4. The sensitivity
      checklist EXPLICITLY includes the module constants invisible to the params surface:
      `_REFINE_SPEED_JUMP_MS`, the `_grow_segment` reversal floor (`s1 > 1.0`), and the engine
      truncation slack (0.5 s) — each swept on the 10 fps-downsampled copy. **Surface the raw-z
      vs smoothed-z conclusion to the user for cross-session relay (spec §2 conditional
      lakehouse item — M-3).**
- [ ] **Step 8.4: Pre-register floors.** Write the chosen numbers into ADR-030 (Task 9) AND into
      the e2e's floor constants. THEN run `--matches holdout` on the DGX. Floors hold → done;
      floors fail → STOP, report to user (no silent threshold adjustment — validation-rigor
      policy).

### Task 9: Docs + bookkeeping

**Files:** Create `docs/superpowers/adrs/ADR-030-shot-goalmouth-trajectory-geometry.md`;
Modify `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, `TODO.md`, `CLAUDE.md`.

- [ ] **Step 9.1: ADR-030** (verify 026 is still next: `ls docs/superpowers/adrs/`). Lakehouse-style
      (memory: ADR pattern). Contents per spec §12: window-end policy; z-profile taxonomy +
      latest-sub-segment rationale + per-column segment provenance (M-1); bounce hysteresis
      (M-2); fixed-g ballistic; first-qualifying contact refinement (H-3); no-xfns leakage
      decision + guard; canonical orientation + engine orientation-agnostic invariant; intentional
      own-goal exclusion; SB goals-vs-saves semantics + lakehouse train/serve note (H-4b);
      pre-registered floors (filled at Step 8.4); attribution note (Anzer & Bauer 2021 contextual,
      no new NOTICE entry); the ONE non-provisional `_confidence` design choice: observed (1.0)
      STRICTLY dominates any extrapolated score (capped 0.9) — the weights inside the cap are
      provisional, the dominance ordering is not; the branch-3 unreachability proof (a vz flip
      requires 3 finite z samples → a detected bounce always leaves ≥2 finite sub-segment
      samples; the `f.sum() >= 2` guard is defensive).
- [ ] **Step 9.2: Version bump hard gate** — ALL FOUR must match `4.23.0`:
      `pyproject.toml` `version`, `silly_kicks/__init__.py` `__version__`, `TODO.md` header
      "Current release", `CHANGELOG.md` new section
      `## 4.23.0 — TF-48 post-shot goalmouth crossing geometry (add_shot_goalmouth, ADR-030)`
      with the feature summary + the M-1/M-2 contracts + "C4 aggregator count 27→28" +
      "no VAEP retrain (not in any default xfn list)".
- [ ] **Step 9.3: TODO.md** — no TF-48 row exists (it arrived via handoff); add nothing (grooming
      policy: CHANGELOG is the record). Update the `TODO.md` header line for 4.23.0 only.
- [ ] **Step 9.4: CLAUDE.md** — append the one-line TF-48 summary to the Tracking section
      (PR-S93 style, matching the existing PR-S## entries' density).
- [ ] **Step 9.5: Spec/plan commit inclusion** — `docs/superpowers/specs/2026-06-10-shot-goalmouth-psxg-design.md`
      and this plan ship in the same single commit.

### Task 10: Shift-Left gate + final review + single commit

- [ ] **Step 10.1:** `ruff format --check .` → clean (run `ruff format .` first if needed).
- [ ] **Step 10.2:** `ruff check .` → clean.
- [ ] **Step 10.3:** `pyright silly_kicks/` (FULL package, never just changed files) → 0 errors.
- [ ] **Step 10.4:** `python -m pytest tests/ -m "not e2e" --tb=short` → ALL PASS (benchmark-skip
      flags per CI parity if locally configured).
- [ ] **Step 10.5:** Run `/final-review` (mandatory gate; regenerates the C4 diagram — verify
      aggregator count shows 28).
- [ ] **Step 10.6:** Present the diff summary to the user; **request explicit commit approval**.
      On approval: ONE commit
      `feat(tracking): TF-48 post-shot goalmouth crossing geometry (add_shot_goalmouth) -- silly-kicks 4.23.0 (ADR-030)`
      then PR per repo convention (squash-only org default).

---

## Self-review (run after writing, fixed inline)

- **Spec coverage:** §4 surface → Tasks 2/5/6/7; §5 orientation → Tasks 1/4; §6 fit/window →
  Task 3; §7 columns/provenance → Tasks 3–6; §8 edge cases → fixtures in Tasks 3/4; §9 perf →
  no structural guard added (slice is batched once — assert via Task 4's single-slice design;
  spy optional, deferred); §10 validation → Task 8; §11 gates → Tasks 6.5/6.6; §12 docs →
  Task 9; §13 open items → resolved in Task 8.3 (params/confidence pilot), 8.1.7 (SkillCorner
  probe deferred to the per-provider coverage run — note: the SkillCorner kloppy-z probe runs
  with the Task 8 coverage report).
- **Known judgment points for the implementer:** kernel constants (`_REFINE_SPEED_JUMP_MS`,
  confidence weights) are provisional BY SPEC — pilot calibrates; do not hand-tune beyond
  making the closed-form tests pass.
- **Type consistency:** column names match spec §7 everywhere (`shot_crossing_source` etc.);
  `_fit_one_shot` returns plain dict consumed only by the engine; `shot_type_ids` kwarg is the
  atomic seam.
