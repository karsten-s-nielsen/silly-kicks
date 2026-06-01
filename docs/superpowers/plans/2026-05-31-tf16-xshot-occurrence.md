# TF-16 xShotOccurrence (xS) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the code for a per-frame shot-occurrence probability model (xS = P(a shot is attempted within ~1 s of a tracking frame), GKDV Layer 2 / TF-16), including its feature extractor, pinned-XGBoost model class, ruthless-HPO training pipeline, and action-coupled / VAEP surfaces — **untrained** (no bundled weights this PR), with full TDD coverage including real-provider extraction tests.

**Architecture:** A new `silly_kicks/tracking/_xshot_occurrence.py` holds a shared train/serve feature extractor (`extract_xshot_features`, paper's 27 "faithful" features in goal-relative coords via a new shared `_geometry.py` helper), an `XShotOccurrenceModel` (deterministic XGBoost classifier, pickle-free booster-JSON serialization), and the ADR-005 surfaces (`compute_xshot_occurrence` / `add_xshot_occurrence` / `xshot_occurrence_xfns`). A separate `_xshot_occurrence_objective.py` wraps a ruthless `CachedObjective` for hyperparameter search, driven by `scripts/train_xshot_occurrence.py`. Inference resolves possession with carrier params **read from model metadata** (not live defaults) to prevent train/serve skew. No weights ship; `from_variant`/`from_hub` are wired but inert.

**Tech Stack:** Python 3.10 (venv), pandas/numpy (core), xgboost (inference, `[xgboost]` extra), `ruthless-efficiency[optuna]` + xgboost (training, new `[train]` extra), pytest + hypothesis (test), ruff 0.15.7 / pyright 1.1.409 (lint/type).

**Spec:** `docs/superpowers/specs/2026-05-31-tf16-xshot-occurrence-design.md`

**Conventions reused (read before starting):** ADR-003 (`@nan_safe_enrichment`), ADR-005 (`_frame_aware` xfns, NOTICE attribution), ADR-008 (`links`/`pitch_control_cache` kwargs), ADR-009 (ruthless `CachedObjective`), ADR-011 (NEW — trained-model lifecycle, written in this plan). Template feature: `silly_kicks/tracking/_ghost_gk.py` (packaging surface only — NOT its sklearn/CV choices).

**Environment / commands (this folder):**
- Python: `.venv\Scripts\python.exe` (CPython 3.10.19). Run tests with `SILLY_KICKS_ASSERT_INVARIANTS=1`.
- Tests (fast): `.venv\Scripts\python.exe -m pytest tests/tracking/test_xshot_occurrence.py -v --tb=short`
- Lint: `.venv\Scripts\python.exe -m ruff check silly_kicks/ tests/ scripts/` AND `.venv\Scripts\python.exe -m ruff format --check silly_kicks/ tests/ scripts/`
- Types: `.venv\Scripts\python.exe -m pyright silly_kicks/`
- Any command that may exceed ~30s MUST run in the background (a hook blocks long foreground commands).
- **Commit policy: NO `git commit` until the very end.** This project uses ONE squashable commit per branch, sentinel-gated, after `/final-review` + explicit user approval. The "Commit" steps below are written for the skill's TDD rhythm but are **NOT executed** here — instead, check the box and run the gates. A single commit is made at the end (Task 14). Branch already exists: `pr-s75-tf16-xshot-occurrence`.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `silly_kicks/tracking/_geometry.py` | Shared goal-relative coordinate transform (extracted; C5) | Create |
| `silly_kicks/tracking/_xshot_occurrence.py` | Feature extractor, `openGoal` helper, `XShotOccurrenceModel`, `compute_*`/`add_*`/`*_xfns`, `prepare_*_training_data` | Create |
| `silly_kicks/tracking/_xshot_occurrence_objective.py` | ruthless `CachedObjective` for HPO | Create |
| `silly_kicks/tracking/features.py` | (no new code — surfaces live in `_xshot_occurrence.py`; `__init__` re-exports) | — |
| `silly_kicks/tracking/__init__.py` | Public exports (`__all__` + imports) | Modify |
| `silly_kicks/atomic/tracking/features.py` | Atomic mirror re-export of `add_xshot_occurrence` | Modify |
| `scripts/train_xshot_occurrence.py` | Training CLI (I/O + `OptunaStrategy` driver) | Create |
| `pyproject.toml` | New `[train]` extra; xS inference relies on existing `[xgboost]` | Modify |
| `NOTICE` | Pipping/Feng/Sabin 2026 attribution | Modify |
| `docs/superpowers/adrs/ADR-011-trained-model-feature-lifecycle.md` | New ADR | Create |
| `tests/tracking/test_xshot_occurrence.py` | Unit tests | Create |
| `tests/tracking/test_xshot_occurrence_integration.py` | Integration tests | Create |
| `tests/tracking/test_xshot_occurrence_real_data.py` | Real-provider extraction tests (regular suite) | Create |
| `tests/invariants/test_xshot_occurrence_bounds.py` | `xshot_occurrence ∈ [0,1]` invariant | Create |
| `CHANGELOG.md`, `TODO.md`, `silly_kicks/__init__.py`, `pyproject.toml` | Version bump (provisional — §13/C1) | Modify (Task 14) |

**Task dependency order:** 1 (geometry) → 2 (openGoal) → 3 (extractor) → 4 (label) → 5 (model) → 6 (compute) → 7 (add_) → 8 (xfns) → 9 (atomic) → 10 (real-data tests) → 11 (objective) → 12 (train CLI) → 13 (NOTICE+ADR+exports+extras) → 14 (version bump + final-review + commit).

---

## Task 1: Shared goal-relative geometry helper

**Files:**
- Create: `silly_kicks/tracking/_geometry.py`
- Test: `tests/tracking/test_xshot_occurrence.py` (new file; first tests live here)

The goal-relative transform maps pitch coords so the *defended* goal sits at a canonical origin (x=0), making LTR/RTL frames identical. ghost-gk has this logic as private closures inside `extract_ghost_gk_features` (`_ghost_gk.py:453`, not importable). Extract a clean, pure version here for xS to consume (ghost-gk refactor deferred to a follow-up — R6).

- [ ] **Step 1: Write the failing test**

```python
# tests/tracking/test_xshot_occurrence.py
"""Unit tests for TF-16 xShotOccurrence (xS)."""
from __future__ import annotations

import numpy as np
import pytest

from silly_kicks.tracking import _geometry as geo

_FIELD_LENGTH = 105.0


def test_to_goal_relative_x_home_goal_no_flip():
    # Defending goal at x=0 (home GK end): coords pass through unchanged.
    assert geo.to_goal_relative_x(30.0, goal_x=0.0) == pytest.approx(30.0)
    assert geo.to_goal_relative_vx(2.0, goal_x=0.0) == pytest.approx(2.0)


def test_to_goal_relative_x_away_goal_flips():
    # Defending goal at x=105 (away GK end): x -> 105 - x, vx -> -vx.
    assert geo.to_goal_relative_x(30.0, goal_x=105.0) == pytest.approx(_FIELD_LENGTH - 30.0)
    assert geo.to_goal_relative_vx(2.0, goal_x=105.0) == pytest.approx(-2.0)


def test_to_goal_relative_nan_propagates():
    assert np.isnan(geo.to_goal_relative_x(np.nan, goal_x=0.0))
    assert np.isnan(geo.to_goal_relative_vx(np.nan, goal_x=105.0))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/tracking/test_xshot_occurrence.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'silly_kicks.tracking._geometry'`

- [ ] **Step 3: Write minimal implementation**

```python
# silly_kicks/tracking/_geometry.py
"""Shared goal-relative coordinate transforms for tracking features.

A frame is "goal-relative" when the *defended* goal sits at x=0, so that
LTR and RTL frames map to identical feature values (doubling effective data
and removing direction asymmetry). ``goal_x`` is the absolute x of the
defended goal: 0.0 for the goal at the low-x end, 105.0 for the high-x end.

See NOTICE for full bibliographic citations.
"""
from __future__ import annotations

import math

FIELD_LENGTH = 105.0
GOAL_Y = 34.0  # pitch half-width (68 / 2) — goal centre y


def _flip(goal_x: float) -> bool:
    return goal_x > 50.0


def to_goal_relative_x(x: float, *, goal_x: float) -> float:
    """Map absolute pitch x to goal-relative x (defended goal at 0).

    Examples
    --------
    >>> to_goal_relative_x(30.0, goal_x=0.0)
    30.0
    >>> to_goal_relative_x(30.0, goal_x=105.0)
    75.0
    """
    if math.isnan(x):
        return x
    return (FIELD_LENGTH - x) if _flip(goal_x) else x


def to_goal_relative_vx(vx: float, *, goal_x: float) -> float:
    """Map absolute x-velocity to goal-relative x-velocity (negated when flipped).

    Examples
    --------
    >>> to_goal_relative_vx(2.0, goal_x=0.0)
    2.0
    >>> to_goal_relative_vx(2.0, goal_x=105.0)
    -2.0
    """
    if math.isnan(vx):
        return vx
    return -vx if _flip(goal_x) else vx
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/tracking/test_xshot_occurrence.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Lint/type, then mark done (NO commit — see header)**

Run: `.venv\Scripts\python.exe -m ruff check silly_kicks/tracking/_geometry.py` and `.venv\Scripts\python.exe -m pyright silly_kicks/`
Expected: clean. Check the box; do not `git commit`.

---

## Task 2: `openGoal` obstruction helper

**Files:**
- Modify: `silly_kicks/tracking/_xshot_occurrence.py` (create the file with this helper)
- Test: `tests/tracking/test_xshot_occurrence.py`

`openGoal` = unobstructed share of the goal mouth (paper Appendix A). Defenders between ball and goal are 75 cm circles; each casts a "shadow" on the goal line via ball→defender tangent lines; shadows are **UNIONed** (not summed). GK excluded. Goal mouth = 7.32 m wide, centred at y=34 (so [30.34, 37.66]).

- [ ] **Step 1: Write the failing tests** (append to `tests/tracking/test_xshot_occurrence.py`)

```python
from silly_kicks.tracking import _xshot_occurrence as xs

# Goal mouth: y in [30.34, 37.66] at x=0 (defended goal). Ball is goal-relative.

def test_open_goal_no_defenders_is_one():
    # No defenders between ball and goal -> fully open.
    val = xs._open_goal_fraction(ball=(20.0, 34.0), defenders=np.empty((0, 2)))
    assert val == pytest.approx(1.0)


def test_open_goal_defender_behind_ball_no_shadow():
    # Defender farther from goal than the ball (x > ball_x) casts no shadow.
    defenders = np.array([[25.0, 34.0]])  # behind ball (ball at x=20)
    val = xs._open_goal_fraction(ball=(20.0, 34.0), defenders=defenders)
    assert val == pytest.approx(1.0)


def test_open_goal_defender_past_goal_line_no_shadow():
    # Defender beyond the goal line (x < 0) casts no shadow.
    defenders = np.array([[-1.0, 34.0]])
    val = xs._open_goal_fraction(ball=(20.0, 34.0), defenders=defenders)
    assert val == pytest.approx(1.0)


def test_open_goal_central_wall_reduces():
    # A defender on the ball->goal-centre line obstructs a central chunk: 0 < open < 1.
    defenders = np.array([[10.0, 34.0]])
    val = xs._open_goal_fraction(ball=(20.0, 34.0), defenders=defenders)
    assert 0.0 < val < 1.0


def test_open_goal_overlapping_shadows_unioned():
    # Two defenders whose shadows overlap must UNION (not sum) -> open >= the
    # open fraction from either one alone (summing would over-count and could
    # drive open below the single-defender value).
    d1 = np.array([[10.0, 34.0]])
    d2 = np.array([[10.0, 34.3]])  # almost same spot -> heavily overlapping shadow
    both = np.vstack([d1, d2])
    open_one = xs._open_goal_fraction(ball=(20.0, 34.0), defenders=d1)
    open_both = xs._open_goal_fraction(ball=(20.0, 34.0), defenders=both)
    # Unioned: adding a near-duplicate barely changes coverage.
    assert open_both == pytest.approx(open_one, abs=0.05)


def test_open_goal_bounds_property():
    # openGoal in [0,1] for random configs.
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(0, 6))
        defs = np.column_stack([rng.uniform(-5, 25, n), rng.uniform(20, 48, n)])
        val = xs._open_goal_fraction(ball=(rng.uniform(5, 30), rng.uniform(25, 43)), defenders=defs)
        assert 0.0 <= val <= 1.0 or np.isnan(val)


def test_open_goal_golden_master_single_defender():
    # FIRST-PRINCIPLES reference (R5), not copied from implementation output:
    # Goal mouth y in [30.34, 37.66] (width 7.32) at x=0. Ball at (20, 34).
    # One defender (radius 0.375 m) centred at (10, 34) — exactly between ball
    # and goal centre. Tangent lines from the ball graze the circle at angular
    # half-width asin(r / d_bd) about the ball->defender bearing, where
    # d_bd = 10. The shadow on the goal line (x=0) spans where those two tangent
    # rays cross x=0. Ball->defender points straight along -x (bearing pi). The
    # two tangents make angle +/- asin(0.375/10) = +/-0.0375 rad with that axis.
    # Ball is 20 m from goal line; each tangent hits x=0 at
    # y = 34 -/+ 20 * tan(0.0375) = 34 -/+ 0.7505 -> shadow ~ [33.25, 34.75],
    # width ~ 1.501 m. Open fraction = 1 - 1.501/7.32 = 0.7950.
    defenders = np.array([[10.0, 34.0]])
    val = xs._open_goal_fraction(ball=(20.0, 34.0), defenders=defenders)
    assert val == pytest.approx(0.7950, abs=0.01)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/tracking/test_xshot_occurrence.py -k open_goal -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'silly_kicks.tracking._xshot_occurrence'`

- [ ] **Step 3: Write minimal implementation** (create the module with constants + the helper)

```python
# silly_kicks/tracking/_xshot_occurrence.py
"""TF-16 xShotOccurrence (xS): P(a shot is attempted within ~1 s of a frame).

Implements the xS sub-model of Pipping, Feng & Sabin (2026), arXiv:2512.00203
("Beyond Expected Goals: A Probabilistic Framework for Shot Occurrences in
Soccer"). Only xS is implemented; the paper's xG and xG+ are out of scope
(silly-kicks values goals/threat via VAEP and xthreat). See NOTICE.

Ships UNTRAINED in PR-S75 (code + synthetic CI fixture + real-provider
extraction tests); maintainer training run + bundled/Hub weights follow.
"""
from __future__ import annotations

import math

import numpy as np

# Goal geometry (goal-relative coords: defended goal at x=0, centre y=34).
GOAL_WIDTH = 7.32
GOAL_Y_CENTRE = 34.0
GOAL_Y_MIN = GOAL_Y_CENTRE - GOAL_WIDTH / 2.0  # 30.34
GOAL_Y_MAX = GOAL_Y_CENTRE + GOAL_WIDTH / 2.0  # 37.66
DEFENDER_RADIUS = 0.375  # 75 cm diameter (paper Appendix A)


def _open_goal_fraction(ball: tuple[float, float], defenders: np.ndarray) -> float:
    """Unobstructed share of the goal mouth from the ball (paper Appendix A).

    Each defender between the ball and the goal line is a circle of radius
    ``DEFENDER_RADIUS``; the ball->defender tangent pair projects an obstructed
    interval onto the goal line. Intervals are UNIONed (overlaps not double
    counted). The GK is not passed in (excluded as an occluder).

    Parameters
    ----------
    ball : (x, y)
        Ball position in goal-relative coords (goal line at x=0).
    defenders : np.ndarray
        Shape (n, 2) of non-GK defender (x, y) in goal-relative coords.

    Returns
    -------
    float
        Open fraction in [0, 1]; NaN if the ball position is NaN.

    Examples
    --------
    >>> import numpy as np
    >>> _open_goal_fraction((20.0, 34.0), np.empty((0, 2)))
    1.0
    """
    bx, by = ball
    if math.isnan(bx) or math.isnan(by):
        return float("nan")
    if bx <= 0:
        return float("nan")  # ball on/behind goal line — undefined

    intervals: list[tuple[float, float]] = []
    for dx, dy in defenders:
        if math.isnan(dx) or math.isnan(dy):
            continue
        # Only defenders strictly between ball and goal line cast a shadow.
        if dx >= bx or dx <= 0:
            continue
        d_bd = math.hypot(dx - bx, dy - by)
        if d_bd <= DEFENDER_RADIUS:
            # Ball essentially on the defender — full obstruction of the mouth.
            intervals.append((GOAL_Y_MIN, GOAL_Y_MAX))
            continue
        half = math.asin(DEFENDER_RADIUS / d_bd)
        base = math.atan2(dy - by, dx - bx)  # ball -> defender bearing
        ys: list[float] = []
        for ang in (base - half, base + half):
            cos_a = math.cos(ang)
            if abs(cos_a) < 1e-12:
                continue
            t = (0.0 - bx) / cos_a  # param to reach x=0 along the tangent ray
            if t <= 0:
                continue
            ys.append(by + t * math.sin(ang))
        if len(ys) < 2:
            continue
        lo, hi = sorted(ys)
        lo = max(lo, GOAL_Y_MIN)
        hi = min(hi, GOAL_Y_MAX)
        if hi > lo:
            intervals.append((lo, hi))

    if not intervals:
        return 1.0
    intervals.sort()
    merged_len = 0.0
    cur_lo, cur_hi = intervals[0]
    for lo, hi in intervals[1:]:
        if lo <= cur_hi:
            cur_hi = max(cur_hi, hi)
        else:
            merged_len += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
    merged_len += cur_hi - cur_lo
    return max(0.0, 1.0 - merged_len / GOAL_WIDTH)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/tracking/test_xshot_occurrence.py -k open_goal -v`
Expected: PASS (7 passed). If the golden-master is off, the bug is in the geometry — fix the helper, not the test's first-principles number.

- [ ] **Step 5: Lint/type, mark done (no commit).**

---

## Task 3: Faithful feature extractor

**Files:**
- Modify: `silly_kicks/tracking/_xshot_occurrence.py`
- Test: `tests/tracking/test_xshot_occurrence.py`

`extract_xshot_features(frame_data, *, gk_team_id, goal_x, feature_set="faithful")` → 1-row DataFrame, 27 columns (the shared train/serve extractor). Mirrors `extract_ghost_gk_features` shape: one frame group in, one row out, goal-relative.

Feature names (exact order): `r, theta, z, speed, openGoal, GK_r, GK_theta, DefDist_0..4, DefAngle_0..4, OffDist_0..4, OffAngle_0..4`.

- [ ] **Step 1: Write the failing tests**

```python
def _one_frame():
    """Minimal single-frame DataFrame: ball + GK + 2 defenders + 2 attackers."""
    import pandas as pd
    rows = [
        # is_ball, is_goalkeeper, team_id, x, y, vx, vy
        dict(is_ball=True, is_goalkeeper=False, team_id=-1, x=20.0, y=34.0, vx=3.0, vy=0.0),
        dict(is_ball=False, is_goalkeeper=True, team_id=1, x=2.0, y=34.0, vx=0.0, vy=0.0),   # defending GK
        dict(is_ball=False, is_goalkeeper=False, team_id=1, x=10.0, y=30.0, vx=0.0, vy=0.0), # defender
        dict(is_ball=False, is_goalkeeper=False, team_id=1, x=12.0, y=38.0, vx=0.0, vy=0.0), # defender
        dict(is_ball=False, is_goalkeeper=False, team_id=2, x=18.0, y=33.0, vx=1.0, vy=0.0), # attacker
        dict(is_ball=False, is_goalkeeper=False, team_id=2, x=22.0, y=36.0, vx=1.0, vy=0.0), # attacker
    ]
    df = pd.DataFrame(rows)
    df["game_id"] = 1
    df["period_id"] = 1
    df["frame_id"] = 100
    return df


def test_extract_features_faithful_shape():
    out = xs.extract_xshot_features(_one_frame(), gk_team_id=1, goal_x=0.0)
    assert list(out.columns) == xs.XSHOT_FEATURE_NAMES_FAITHFUL
    assert len(out) == 1
    assert len(xs.XSHOT_FEATURE_NAMES_FAITHFUL) == 27


def test_extended_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="extended"):
        xs.extract_xshot_features(_one_frame(), gk_team_id=1, goal_x=0.0, feature_set="extended")


def test_extract_features_goal_relative_symmetry():
    import pandas as pd
    # Same scene mirrored to the other end must yield identical features.
    f0 = _one_frame()
    f1 = f0.copy()
    f1["x"] = 105.0 - f1["x"]
    f1["vx"] = -f1["vx"]
    a = xs.extract_xshot_features(f0, gk_team_id=1, goal_x=0.0)
    b = xs.extract_xshot_features(f1, gk_team_id=1, goal_x=105.0)
    pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-9)


def test_fewer_than_5_players_nan_slots():
    out = xs.extract_xshot_features(_one_frame(), gk_team_id=1, goal_x=0.0)
    # Only 2 defenders present -> DefDist_2..4 are NaN.
    assert np.isnan(out["DefDist_2"].iloc[0])
    assert np.isnan(out["DefAngle_4"].iloc[0])
    assert not np.isnan(out["DefDist_0"].iloc[0])
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/tracking/test_xshot_occurrence.py -k "extract_features or extended_raises or fewer_than" -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'extract_xshot_features'`

- [ ] **Step 3: Write minimal implementation** (append to `_xshot_occurrence.py`)

```python
from typing import Literal

import pandas as pd

from silly_kicks.tracking import _geometry as _geo

XShotFeatureSet = Literal["faithful", "extended"]

_BALL_FEATURES = ["r", "theta", "z", "speed", "openGoal"]
_GK_FEATURES = ["GK_r", "GK_theta"]
_DEF_FEATURES = [f"DefDist_{k}" for k in range(5)] + [f"DefAngle_{k}" for k in range(5)]
_OFF_FEATURES = [f"OffDist_{k}" for k in range(5)] + [f"OffAngle_{k}" for k in range(5)]
# Interleave Dist/Angle per the data dictionary order: Dist_0,Angle_0,Dist_1,...
_DEF_INTERLEAVED = [c for k in range(5) for c in (f"DefDist_{k}", f"DefAngle_{k}")]
_OFF_INTERLEAVED = [c for k in range(5) for c in (f"OffDist_{k}", f"OffAngle_{k}")]
XSHOT_FEATURE_NAMES_FAITHFUL = (
    _BALL_FEATURES + _GK_FEATURES + _DEF_INTERLEAVED + _OFF_INTERLEAVED
)  # 5 + 2 + 10 + 10 = 27


def _nearest_k(ball_xy: tuple[float, float], pts: np.ndarray, k: int = 5):
    """Return (dist[k], bearing[k]) of the k nearest pts to ball, NaN-padded.

    Bearing is the angle of (point - ball) in goal-relative coords.
    """
    dist = np.full(k, np.nan)
    ang = np.full(k, np.nan)
    if len(pts) == 0:
        return dist, ang
    bx, by = ball_xy
    d = np.hypot(pts[:, 0] - bx, pts[:, 1] - by)
    order = np.argsort(d)[:k]
    for i, j in enumerate(order):
        dist[i] = d[j]
        ang[i] = math.atan2(pts[j, 1] - by, pts[j, 0] - bx)
    return dist, ang


def extract_xshot_features(
    frame_data: pd.DataFrame,
    *,
    gk_team_id: int | str,
    goal_x: float,
    feature_set: XShotFeatureSet = "faithful",
) -> pd.DataFrame:
    """Extract xS features from one frame (goal-relative). Returns a 1-row frame.

    The defending team is ``gk_team_id`` (the team whose goal is being attacked);
    ``goal_x`` is the absolute x of that defended goal (0.0 or 105.0).

    ``feature_set="extended"`` is not implemented in PR-S75 (raises
    NotImplementedError) — see the spec; only ``"faithful"`` (the paper's 27
    features) ships here.

    Examples
    --------
    >>> # row = extract_xshot_features(frame, gk_team_id=1, goal_x=0.0)
    >>> # row.shape == (1, 27)

    See NOTICE for full bibliographic citations.
    """
    if feature_set != "faithful":
        raise NotImplementedError(
            "xShotOccurrence feature_set='extended' is not implemented in this "
            "release; only 'faithful' (paper Appendix A) is available. See the "
            "TF-16 weights/TF-19 follow-up."
        )

    gx = lambda x: _geo.to_goal_relative_x(float(x), goal_x=goal_x)  # noqa: E731

    is_ball = frame_data["is_ball"].astype(bool)
    is_gk = frame_data["is_goalkeeper"].astype(bool)
    ball = frame_data[is_ball]
    players = frame_data[~is_ball]

    if len(ball) > 0:
        bx_raw = float(ball["x"].iloc[0])
        by = float(ball["y"].iloc[0])
        bvx = float(ball["vx"].iloc[0]) if "vx" in ball.columns else np.nan
        bvy = float(ball["vy"].iloc[0]) if "vy" in ball.columns else np.nan
        bz = float(ball["z"].iloc[0]) if "z" in ball.columns else np.nan
    else:
        bx_raw = by = bvx = bvy = bz = np.nan
    bx = gx(bx_raw)

    r = math.hypot(bx, by - _geo.GOAL_Y) if not math.isnan(bx) else np.nan
    theta = math.atan2(by - _geo.GOAL_Y, bx) if not math.isnan(bx) else np.nan
    speed = math.hypot(bvx, bvy) if not math.isnan(bvx) else np.nan

    defending = players[(players["team_id"] == gk_team_id) & (~is_gk[players.index])]
    attacking = players[players["team_id"] != gk_team_id]
    gk_rows = players[(players["team_id"] == gk_team_id) & is_gk[players.index]]

    def_xy = np.column_stack([
        defending["x"].map(gx).to_numpy(dtype=float),
        defending["y"].to_numpy(dtype=float),
    ]) if len(defending) else np.empty((0, 2))
    atk_xy = np.column_stack([
        attacking["x"].map(gx).to_numpy(dtype=float),
        attacking["y"].to_numpy(dtype=float),
    ]) if len(attacking) else np.empty((0, 2))

    open_goal = _open_goal_fraction((bx, by), def_xy)

    if len(gk_rows) > 0:
        gkx = gx(gk_rows["x"].iloc[0])
        gky = float(gk_rows["y"].iloc[0])
        gk_r = math.hypot(gkx, gky - _geo.GOAL_Y)
        gk_theta = math.atan2(gky - _geo.GOAL_Y, gkx)
    else:
        gk_r = gk_theta = np.nan

    ddist, dang = _nearest_k((bx, by), def_xy)
    odist, oang = _nearest_k((bx, by), atk_xy)

    values: dict[str, float] = {
        "r": r, "theta": theta, "z": bz, "speed": speed, "openGoal": open_goal,
        "GK_r": gk_r, "GK_theta": gk_theta,
    }
    for k in range(5):
        values[f"DefDist_{k}"] = ddist[k]
        values[f"DefAngle_{k}"] = dang[k]
        values[f"OffDist_{k}"] = odist[k]
        values[f"OffAngle_{k}"] = oang[k]

    return pd.DataFrame([[values[c] for c in XSHOT_FEATURE_NAMES_FAITHFUL]],
                        columns=XSHOT_FEATURE_NAMES_FAITHFUL)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/tracking/test_xshot_occurrence.py -k "extract_features or extended_raises or fewer_than" -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Lint/type, mark done (no commit).**

---

## Task 4: Label builder (`time_seconds` window, no linkage — R2/B1)

**Files:**
- Modify: `silly_kicks/tracking/_xshot_occurrence.py`
- Test: `tests/tracking/test_xshot_occurrence.py`

`build_xshot_labels(frames_index, shots, *, horizon_seconds=1.0)` → per-frame-row 0/1 Series. A frame `(game_id, period_id, team_in_possession, time_seconds=t)` is positive iff a same-team shot's `time_seconds ∈ [t, t+horizon]` within the same period. **No `link_actions_to_frames`** — compares the shot action's own `time_seconds` directly (R2). Must be robust to non-contiguous `frame_id` (B1).

- [ ] **Step 1: Write the failing tests**

```python
def test_label_horizon_via_time_seconds():
    import pandas as pd
    # 3 in-possession frames (team 2) at t=0.0, 0.5, 1.0 in period 1; one shot at t=1.2.
    fidx = pd.DataFrame({
        "game_id": [1, 1, 1], "period_id": [1, 1, 1],
        "time_seconds": [0.0, 0.5, 1.0], "team_in_possession": [2, 2, 2],
    })
    shots = pd.DataFrame({"game_id": [1], "period_id": [1], "team_id": [2], "time_seconds": [1.2]})
    y = xs.build_xshot_labels(fidx, shots, horizon_seconds=1.0)
    # t=0.0: shot at 1.2 is >1.0 ahead -> 0; t=0.5: 1.2-0.5=0.7 <=1 ->1; t=1.0: 0.2 ->1
    assert list(y) == [0, 1, 1]


def test_label_robust_to_noncontiguous_frame_id():
    import pandas as pd
    # Same times as above but frame_id has a huge gap — label must be identical
    # (proves no frame_id arithmetic).
    fidx = pd.DataFrame({
        "game_id": [1, 1, 1], "period_id": [1, 1, 1], "frame_id": [10, 9999, 10000],
        "time_seconds": [0.0, 0.5, 1.0], "team_in_possession": [2, 2, 2],
    })
    shots = pd.DataFrame({"game_id": [1], "period_id": [1], "team_id": [2], "time_seconds": [1.2]})
    y = xs.build_xshot_labels(fidx, shots, horizon_seconds=1.0)
    assert list(y) == [0, 1, 1]


def test_label_no_period_bleed():
    import pandas as pd
    # Frame at end of P1; shot just after at start of P2 — must NOT label P1 frame positive.
    fidx = pd.DataFrame({
        "game_id": [1], "period_id": [1], "time_seconds": [45.0], "team_in_possession": [2],
    })
    shots = pd.DataFrame({"game_id": [1], "period_id": [2], "team_id": [2], "time_seconds": [45.2]})
    y = xs.build_xshot_labels(fidx, shots, horizon_seconds=1.0)
    assert list(y) == [0]


def test_label_inclusive_of_t():
    import pandas as pd
    fidx = pd.DataFrame({"game_id": [1], "period_id": [1], "time_seconds": [10.0], "team_in_possession": [2]})
    shots = pd.DataFrame({"game_id": [1], "period_id": [1], "team_id": [2], "time_seconds": [10.0]})
    y = xs.build_xshot_labels(fidx, shots, horizon_seconds=1.0)
    assert list(y) == [1]


def test_label_turnover_opponent_shot_is_negative():
    import pandas as pd
    # Frame: team 2 in possession at t=5.0. Opponent (team 1) shoots at t=5.5.
    fidx = pd.DataFrame({"game_id": [1], "period_id": [1], "time_seconds": [5.0], "team_in_possession": [2]})
    shots = pd.DataFrame({"game_id": [1], "period_id": [1], "team_id": [1], "time_seconds": [5.5]})
    y = xs.build_xshot_labels(fidx, shots, horizon_seconds=1.0)
    assert list(y) == [0]
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/tracking/test_xshot_occurrence.py -k label -v`
Expected: FAIL — `AttributeError: ... 'build_xshot_labels'`

- [ ] **Step 3: Write minimal implementation** (append to `_xshot_occurrence.py`)

```python
def build_xshot_labels(
    frames_index: pd.DataFrame,
    shots: pd.DataFrame,
    *,
    horizon_seconds: float = 1.0,
) -> pd.Series:
    """Per-row xS label: 1 iff a same-team shot occurs in [t, t+horizon] same period.

    Compares the shot action's own ``time_seconds`` directly against each frame
    row's window — NO linkage step (avoids the +/-tolerance link smear at the 1 s
    horizon). Robust to non-contiguous ``frame_id`` (uses time, not frame index).

    Parameters
    ----------
    frames_index : pd.DataFrame
        One row per scored frame slot; columns ``game_id``, ``period_id``,
        ``time_seconds``, ``team_in_possession``.
    shots : pd.DataFrame
        Shot actions; columns ``game_id``, ``period_id``, ``team_id``,
        ``time_seconds``.

    Returns
    -------
    pd.Series
        int (0/1), aligned to ``frames_index.index``.

    Examples
    --------
    >>> # y = build_xshot_labels(fidx, shots, horizon_seconds=1.0)
    """
    y = np.zeros(len(frames_index), dtype=int)
    if len(shots) == 0:
        return pd.Series(y, index=frames_index.index)
    # Group shots by (game, period, team) -> sorted time array.
    shot_groups: dict[tuple, np.ndarray] = {}
    for key, grp in shots.groupby(["game_id", "period_id", "team_id"], dropna=False):
        shot_groups[key] = np.sort(grp["time_seconds"].to_numpy(dtype=float))
    gids = frames_index["game_id"].to_numpy()
    pids = frames_index["period_id"].to_numpy()
    tpos = frames_index["team_in_possession"].to_numpy()
    ts = frames_index["time_seconds"].to_numpy(dtype=float)
    for i in range(len(frames_index)):
        key = (gids[i], pids[i], tpos[i])
        arr = shot_groups.get(key)
        if arr is None:
            continue
        lo = float(ts[i])
        hi = lo + horizon_seconds
        # any shot time in [lo, hi]?
        left = np.searchsorted(arr, lo, side="left")
        if left < len(arr) and arr[left] <= hi:
            y[i] = 1
    return pd.Series(y, index=frames_index.index)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/tracking/test_xshot_occurrence.py -k label -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Lint/type, mark done (no commit).**

---

## Task 5: `XShotOccurrenceModel` (pinned XGBoost, pickle-free serialization)

**Files:**
- Modify: `silly_kicks/tracking/_xshot_occurrence.py`
- Test: `tests/tracking/test_xshot_occurrence.py`

Deterministic XGBoost classifier matching `_xgb_classifier` (`tree_method="hist", n_jobs=1, subsample=1.0, colsample_bytree=1.0, random_state=seed, eval_metric="logloss"`). Save = booster JSON + `metadata.json` (feature_names, feature_set, horizon_seconds, shot_types, **carrier_params**, hyperparams, versions) + `SHA256SUMS`. Load verifies SHA-256, requires xgboost.

- [ ] **Step 1: Write the failing tests**

```python
def _toy_xy(n=400, seed=0):
    import pandas as pd
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 27)), columns=xs.XSHOT_FEATURE_NAMES_FAITHFUL)
    # Label correlated with first feature so the model can learn something.
    y = (X["r"] + rng.normal(scale=0.5, size=n) < 0).astype(int)
    return X, pd.Series(y)


def test_model_fit_predict_proba():
    X, y = _toy_xy()
    m = xs.XShotOccurrenceModel().fit(X, y)
    p = m.predict_proba(X)
    assert p.shape == (len(X),)
    assert np.all((p >= 0) & (p <= 1))


def test_model_deterministic():
    X, y = _toy_xy()
    p1 = xs.XShotOccurrenceModel(params={"random_state": 42}).fit(X, y).predict_proba(X)
    p2 = xs.XShotOccurrenceModel(params={"random_state": 42}).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(p1, p2)


def test_model_save_load_roundtrip(tmp_path):
    X, y = _toy_xy()
    m = xs.XShotOccurrenceModel().fit(X, y, carrier_params={"tolerance_m": 3.0, "beta": 0.5, "gamma": 1.0})
    m.save(tmp_path / "xs_v1")
    loaded = xs.XShotOccurrenceModel.load(tmp_path / "xs_v1")
    np.testing.assert_allclose(loaded.predict_proba(X), m.predict_proba(X), rtol=1e-9)
    assert loaded.carrier_params == {"tolerance_m": 3.0, "beta": 0.5, "gamma": 1.0}
    assert loaded.feature_set == "faithful"


def test_model_sha256_verification(tmp_path):
    X, y = _toy_xy()
    xs.XShotOccurrenceModel().fit(X, y).save(tmp_path / "xs_v1")
    (tmp_path / "xs_v1" / "model.json").write_text("tampered")
    with pytest.raises(xs.IntegrityError):
        xs.XShotOccurrenceModel.load(tmp_path / "xs_v1")


def test_model_carrier_params_default_when_unset():
    X, y = _toy_xy()
    m = xs.XShotOccurrenceModel().fit(X, y)
    assert m.carrier_params == {"tolerance_m": 3.0, "beta": 0.5, "gamma": 1.0}
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/tracking/test_xshot_occurrence.py -k model -v`
Expected: FAIL — `AttributeError: ... 'XShotOccurrenceModel'`

- [ ] **Step 3: Write minimal implementation** (append; `Path`/`json`/`hashlib` imports at top of file)

```python
# add to imports at top of _xshot_occurrence.py:
#   import hashlib, json
#   from pathlib import Path

_DEFAULT_CARRIER_PARAMS = {"tolerance_m": 3.0, "beta": 0.5, "gamma": 1.0}
_HF_REPO_ID = "silly-kicks/xshot-occurrence-v1"
_MODEL_VERSION = "1.0.0"


class IntegrityError(Exception):
    """Raised when a model artifact fails SHA-256 verification."""


_INT_PARAMS = ("n_estimators", "max_depth", "min_child_weight")


def _pinned_params(overrides: dict | None) -> dict:
    base = {
        "n_estimators": 100, "max_depth": 4, "learning_rate": 0.3,
        "tree_method": "hist", "n_jobs": 1, "subsample": 1.0,
        "colsample_bytree": 1.0, "random_state": 42, "eval_metric": "logloss",
        "verbosity": 0,
    }
    if overrides:
        base.update(overrides)
    # Optuna FloatRange feeds floats; XGBoost wants ints for these — round.
    for k in _INT_PARAMS:
        if k in base and base[k] is not None:
            base[k] = int(round(float(base[k])))
    return base


class XShotOccurrenceModel:
    """xS classifier: pinned-deterministic XGBoost over snapshot frame features.

    Serialization is pickle-free (xgboost native booster JSON + metadata.json +
    SHA256SUMS). ``carrier_params`` are recorded so inference can resolve
    possession identically to training (R3). See NOTICE.

    Examples
    --------
    >>> # m = XShotOccurrenceModel().fit(X, y)
    >>> # p = m.predict_proba(X)
    """

    def __init__(self, *, feature_set: XShotFeatureSet = "faithful", params: dict | None = None) -> None:
        if feature_set != "faithful":
            raise NotImplementedError("Only feature_set='faithful' is implemented.")
        self.feature_set: XShotFeatureSet = feature_set
        self._params = _pinned_params(params)
        self._booster = None  # xgboost.Booster after fit/load
        self.carrier_params: dict = dict(_DEFAULT_CARRIER_PARAMS)
        self.horizon_seconds: float = 1.0
        self.shot_types: list[str] = ["shot", "shot_penalty", "shot_freekick"]

    def fit(self, features: pd.DataFrame, labels: pd.Series, *,
            carrier_params: dict | None = None, horizon_seconds: float = 1.0) -> "XShotOccurrenceModel":
        import xgboost as xgb
        self.carrier_params = dict(carrier_params) if carrier_params else dict(_DEFAULT_CARRIER_PARAMS)
        self.horizon_seconds = horizon_seconds
        clf = xgb.XGBClassifier(**self._params)
        clf.fit(features.to_numpy(dtype=float), labels.to_numpy(dtype=int))
        self._booster = clf.get_booster()
        self._booster.feature_names = list(features.columns)
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if self._booster is None:
            raise RuntimeError("Model not fitted/loaded.")
        import xgboost as xgb
        dm = xgb.DMatrix(features.to_numpy(dtype=float), feature_names=list(features.columns))
        return np.asarray(self._booster.predict(dm), dtype=float)

    def save(self, path: Path) -> None:
        if self._booster is None:
            raise RuntimeError("Model not fitted.")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._booster.save_model(str(path / "model.json"))
        metadata = {
            "feature_names": XSHOT_FEATURE_NAMES_FAITHFUL,
            "feature_set": self.feature_set,
            "horizon_seconds": self.horizon_seconds,
            "shot_types": self.shot_types,
            "carrier_params": self.carrier_params,
            "params": self._params,
            "version": _MODEL_VERSION,
        }
        (path / "metadata.json").write_text(json.dumps(metadata, indent=2), newline="\n")
        with open(path / "SHA256SUMS", "w", newline="\n") as f:
            for fname in ["model.json", "metadata.json"]:
                raw = (path / fname).read_bytes()
                if fname.endswith(".json"):
                    raw = raw.replace(b"\r\n", b"\n")
                f.write(f"{hashlib.sha256(raw).hexdigest()}  {fname}\n")

    @classmethod
    def load(cls, path: Path) -> "XShotOccurrenceModel":
        import xgboost as xgb
        path = Path(path)
        sums = path / "SHA256SUMS"
        if not sums.exists():
            raise IntegrityError(f"SHA256SUMS not found in {path}")
        for line in sums.read_text().splitlines():
            if not line.strip():
                continue
            expected, fname = line.split("  ", 1)
            raw = (path / fname).read_bytes()
            if fname.endswith(".json"):
                raw = raw.replace(b"\r\n", b"\n")
            if hashlib.sha256(raw).hexdigest() != expected:
                raise IntegrityError(f"Integrity check failed for {fname}")
        meta = json.loads((path / "metadata.json").read_text())
        model = cls(feature_set=meta.get("feature_set", "faithful"), params=meta.get("params"))
        model.carrier_params = meta.get("carrier_params", dict(_DEFAULT_CARRIER_PARAMS))
        model.horizon_seconds = meta.get("horizon_seconds", 1.0)
        model.shot_types = meta.get("shot_types", model.shot_types)
        booster = xgb.Booster()
        booster.load_model(str(path / "model.json"))
        model._booster = booster
        return model

    @classmethod
    def from_variant(cls, variant: str = "default") -> "XShotOccurrenceModel":
        raise FileNotFoundError(
            "xShotOccurrence weights are not yet bundled. Train via "
            "scripts/train_xshot_occurrence.py, or await the TF-16 weights follow-up."
        )

    @classmethod
    def from_hub(cls, repo_id: str = _HF_REPO_ID) -> "XShotOccurrenceModel":
        from huggingface_hub import snapshot_download  # noqa: F401
        raise FileNotFoundError(
            f"No published xShotOccurrence weights at {repo_id} yet (weights follow-up)."
        )
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/tracking/test_xshot_occurrence.py -k model -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Lint/type, mark done (no commit).**

---

## Task 6: `compute_xshot_occurrence` (per-frame primitive; consumes metadata carrier params — R3)

**Files:**
- Modify: `silly_kicks/tracking/_xshot_occurrence.py`
- Test: `tests/tracking/test_xshot_occurrence.py`

Adds `xshot_occurrence` column to in-possession non-ball frame rows. Resolves possession + defended-goal via `infer_ball_carrier(frames, **model.carrier_params)` → `derive_team_in_possession`. Goal per (game, period, defending team) from mean GK x (period-flip-safe). NaN where undefined.

- [ ] **Step 1: Write the failing tests**

```python
def test_compute_xshot_no_model_errors():
    import pandas as pd
    frames = _one_frame()
    # needs full schema columns; reuse the real-data path in integration instead.
    with pytest.raises((FileNotFoundError, RuntimeError)):
        xs.compute_xshot_occurrence(frames, model=None, home_team_id=1)


def test_inference_uses_metadata_carrier_params(monkeypatch):
    # A model carrying non-default carrier params must drive infer_ball_carrier
    # with THOSE params, not the library defaults (R3).
    import pandas as pd
    from silly_kicks.tracking import _xshot_occurrence as xsmod

    captured = {}

    def fake_infer(frames, *, tolerance_m, beta, gamma):
        captured["params"] = (tolerance_m, beta, gamma)
        # minimal valid carrier output
        return pd.DataFrame({
            "game_id": [1], "period_id": [1], "frame_id": [100],
            "ball_carrier_player_id": [pd.NA], "ball_carrier_distance_m": [np.nan],
            "ball_carrier_team_id": [2],
        })

    monkeypatch.setattr(xsmod, "infer_ball_carrier", fake_infer)
    X, y = _toy_xy()
    model = xsmod.XShotOccurrenceModel().fit(
        X, y, carrier_params={"tolerance_m": 9.0, "beta": 0.1, "gamma": 0.2})
    frames = _frame_with_full_schema()  # helper defined in integration; see Step 3 note
    xsmod.compute_xshot_occurrence(frames, model=model, home_team_id=1)
    assert captured["params"] == (9.0, 0.1, 0.2)
```

Note: `_frame_with_full_schema()` builds a frames DataFrame with the full `TRACKING_FRAMES_COLUMNS` (add `time_seconds`, `frame_rate`, `ball_state`, etc. to `_one_frame()`); define it near `_one_frame` in the test file.

- [ ] **Step 2: Run to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/tracking/test_xshot_occurrence.py -k compute_xshot -v`
Expected: FAIL — `AttributeError: ... 'compute_xshot_occurrence'`

- [ ] **Step 3: Write minimal implementation** (append; import `infer_ball_carrier`/`derive_team_in_possession` at module level so monkeypatch works)

```python
# add at top of _xshot_occurrence.py (module-level, so tests can monkeypatch):
from silly_kicks.tracking._ball_carrier import derive_team_in_possession, infer_ball_carrier


def _resolve_model(model):
    if isinstance(model, XShotOccurrenceModel):
        return model
    if model is None or isinstance(model, str):
        return XShotOccurrenceModel.from_variant(model or "default")  # raises until weights ship
    raise TypeError(f"Unsupported model type: {type(model)!r}")


def _defended_goal_x(frames: pd.DataFrame) -> dict:
    """(game_id, period_id, team_id) -> defended goal_x (0 or 105).

    N1: GK identification quality is provider-variable (Metrica/SkillCorner were
    21-50% pre-fix). Prefer mean GK x; fall back to the team's mean outfield x
    when a (game, period, team) has no GK rows, so a mis-/missing-GK does not
    silently drop the team from the goal map. The defended goal is the end the
    team's own players sit nearer (LTR-normalized: defending team clusters in its
    own half).
    """
    players = frames[~frames["is_ball"].astype(bool)]
    is_gk = players["is_goalkeeper"].astype(bool)
    out: dict = {}
    for key, grp in players.groupby(["game_id", "period_id", "team_id"], dropna=False):
        gk_rows = grp[grp["is_goalkeeper"].astype(bool)]
        ref = gk_rows if len(gk_rows) else grp  # fallback: whole team mean-x
        out[key] = 0.0 if float(ref["x"].mean()) < 52.5 else 105.0
    return out


def compute_xshot_occurrence(
    frames: pd.DataFrame,
    *,
    model: "XShotOccurrenceModel | str | None" = None,
    home_team_id: int | str,
    pitch_control_cache=None,  # reserved for 'extended' (not used by 'faithful')
    link_frame_ids: set[int] | None = None,
) -> pd.DataFrame:
    """Add an ``xshot_occurrence`` column (P(shot within ~1 s)) per in-possession frame.

    Possession + defended goal are resolved with the carrier params stored in the
    model's metadata (R3), so serve-time selection matches training. Rows whose
    state is undefined get NaN. ``pitch_control_cache`` is accepted for forward
    compat with the (deferred) 'extended' variant but is valid only for canonical
    frames — counterfactual callers must omit it. See NOTICE / spec §7.

    Examples
    --------
    >>> # out = compute_xshot_occurrence(frames, model=m, home_team_id=1)
    """
    m = _resolve_model(model)
    out = frames.copy()
    out["xshot_occurrence"] = np.nan

    # N-A: carrier inference + possession MUST run on the FULL contiguous frames.
    # infer_ball_carrier has a CROSS-FRAME dependency (gamma hysteresis carries the
    # incumbent carrier across consecutive frames within a (game, period) segment).
    # Running it on the scattered link_frame_ids subset would yield a different
    # carrier -> different team_in_possession, AND diverge from the trainer (which
    # runs it on full frames) -> train/serve skew (R3). This is the documented
    # frame-restriction cross-frame-dependency trap. So: carrier on full frames;
    # restrict ONLY the per-frame extract + batched predict (the expensive part).
    carrier = infer_ball_carrier(frames, **m.carrier_params)
    poss = derive_team_in_possession(frames, carrier)
    goal_map = _defended_goal_x(frames)

    # Pass 1: build ONE feature row per target frame (extraction stays per-frame;
    # the XGBoost predict must NOT be — P1). Restrict to link_frame_ids HERE.
    feat_rows: list[pd.DataFrame] = []
    keys: list[tuple] = []  # (gid, pid, frame_id, team_in_possession)
    n_groups = 0
    n_skipped_goal = 0  # N1: count frames dropped for a missing goal map entry
    for (gid, pid, frame_id), grp in poss.groupby(["game_id", "period_id", "frame_id"], dropna=False):
        if link_frame_ids is not None and int(frame_id) not in link_frame_ids:
            continue
        n_groups += 1
        tip = grp["team_in_possession"].iloc[0]
        if pd.isna(tip):
            continue
        teams = [t for t in grp["team_id"].dropna().unique() if t != tip]
        if not teams:
            continue
        def_team = teams[0]
        goal_x = goal_map.get((gid, pid, def_team))
        if goal_x is None:
            n_skipped_goal += 1
            continue
        feat_rows.append(extract_xshot_features(grp, gk_team_id=def_team, goal_x=goal_x))
        keys.append((gid, pid, frame_id, tip))

    # N1: surface coverage loss rather than dropping silently.
    if n_skipped_goal and n_groups and n_skipped_goal / n_groups > 0.05:
        import warnings
        warnings.warn(
            f"xshot_occurrence: {n_skipped_goal}/{n_groups} frame-groups skipped "
            f"(no defended-goal resolution); possible GK-identification gap.",
            stacklevel=2,
        )

    if not feat_rows:
        return out

    # Pass 2: ONE batched predict over the stacked matrix, scatter back.
    feature_matrix = pd.concat(feat_rows, ignore_index=True)
    probs = m.predict_proba(feature_matrix)
    key_df = pd.DataFrame(keys, columns=["game_id", "period_id", "frame_id", "team_id"])
    key_df["__p"] = probs
    # N-B: join on TEMPORARY string keys so we never mutate out["game_id"]/["team_id"]
    # dtypes (preserve the TRACKING_FRAMES schema — "add one column, change nothing else").
    out["__gid"] = out["game_id"].astype(str)
    out["__tid"] = out["team_id"].astype(str)
    key_df["__gid"] = key_df["game_id"].astype(str)
    key_df["__tid"] = key_df["team_id"].astype(str)
    key_df = key_df.drop(columns=["game_id", "team_id"])
    out = out.merge(key_df, on=["__gid", "period_id", "frame_id", "__tid"], how="left")
    out["xshot_occurrence"] = out["__p"]
    return out.drop(columns=["__p", "__gid", "__tid"])
```

- [ ] **Step 4: Add a compute-level perf benchmark (P1)** — `tests/tracking/test_xshot_occurrence.py`

```python
def test_compute_xshot_benchmark(benchmark):
    # P1: budget the FULL compute (carrier + extraction + batched predict) over a
    # realistic frame count, not just per-frame extraction. Flat budget from worst
    # observed CI timing x1.5 (fill in after first green run; start generous).
    import pandas as pd
    frames = _synthetic_match_frames(n_frames=300)  # ball+22 players x 300 frames
    X, y = _toy_xy()
    model = xs.XShotOccurrenceModel().fit(X, y)
    home = frames["team_id"].dropna().iloc[0]
    result = benchmark(lambda: xs.compute_xshot_occurrence(frames, model=model, home_team_id=home))
    assert "xshot_occurrence" in result.columns
```

`_synthetic_match_frames(n_frames)` helper: full-schema frames, ball + 11v11 across `n_frames` frames in one period with `ball_state="alive"`, `time_seconds` increasing, `vx/vy` present. Set the `pytest-benchmark` budget (`@pytest.mark.benchmark` or a flat assert on `result` timing) after the first green run, per the house CI-perf-budget convention.

- [ ] **Step 5: Add N-A (full-frames carrier) + N-B (schema-preserving) regression tests**

```python
def test_compute_carrier_runs_on_full_frames(monkeypatch):
    # N-A: with link_frame_ids set, infer_ball_carrier must STILL receive the full
    # frame set (cross-frame hysteresis correctness + train/serve parity), and only
    # extraction/predict are restricted.
    from silly_kicks.tracking import _xshot_occurrence as xsmod
    seen = {}
    real = xsmod.infer_ball_carrier
    def spy(frames, **kw):
        seen["n_rows"] = len(frames)
        return real(frames, **kw)
    monkeypatch.setattr(xsmod, "infer_ball_carrier", spy)
    frames = _synthetic_match_frames(n_frames=40)
    total_rows = len(frames)
    X, y = _toy_xy()
    model = xsmod.XShotOccurrenceModel().fit(X, y)
    link_ids = set(frames["frame_id"].astype(int).unique()[:3])  # only 3 of 40 frames
    xsmod.compute_xshot_occurrence(
        frames, model=model, home_team_id=frames["team_id"].dropna().iloc[0],
        link_frame_ids=link_ids)
    # Carrier saw ALL rows, not just the 3 linked frames' rows.
    assert seen["n_rows"] == total_rows


def test_compute_preserves_id_dtypes():
    # N-B: returned frames keep original game_id/team_id dtypes (no schema mutation).
    frames = _synthetic_match_frames(n_frames=20)
    gid_dtype, tid_dtype = frames["game_id"].dtype, frames["team_id"].dtype
    X, y = _toy_xy()
    model = xs.XShotOccurrenceModel().fit(X, y)
    out = xs.compute_xshot_occurrence(frames, model=model, home_team_id=frames["team_id"].dropna().iloc[0])
    assert out["game_id"].dtype == gid_dtype
    assert out["team_id"].dtype == tid_dtype
    assert "xshot_occurrence" in out.columns
```

- [ ] **Step 6: Run to verify it passes**

Run (background): `.venv\Scripts\python.exe -m pytest tests/tracking/test_xshot_occurrence.py -k "compute_xshot or benchmark or carrier_runs or preserves_id" -v`
Expected: PASS. The batched predict means a 300-frame match is a single `predict_proba`, not 300; the carrier still runs on the full match.

- [ ] **Step 7: Lint/type, mark done (no commit).**

---

## Task 7: `add_xshot_occurrence` (action-coupled, `@nan_safe_enrichment`)

**Files:**
- Modify: `silly_kicks/tracking/_xshot_occurrence.py`
- Test: `tests/tracking/test_xshot_occurrence.py`

Adds one `xshot_occurrence` column to actions (xS at each action's linked frame). Uses `link_actions_to_frames` (this IS a serve-time mapping — linkage allowed here, unlike the label). `@nan_safe_enrichment`, `links` kwarg, provenance-skip guard.

**Semantic (S1, intentional + tested):** xS is the *possessing team's* shot probability — `compute_xshot_occurrence` writes it only on `team_id == team_in_possession` rows, and `add_xshot_occurrence` joins on `team_id`. So a possessing-team action (pass/dribble/shot) gets a value; a defensive action at the same frame by the non-possessing team (tackle/interception) gets **NaN by design**. This is documented in the docstring and covered by `test_add_xshot_defensive_action_is_nan`.

**Dtype trap (P2):** the join on `["game_id","period_id","frame_id","team_id"]` must align **both** `game_id` AND `team_id` dtypes — int64 (Gradient Sports) vs object (kloppy) `team_id` will silently miss the join → all-NaN. Align both (this is the exact provider asymmetry that bit TF-24).

- [ ] **Step 1: Write the failing tests**

```python
def test_add_xshot_nan_safe_marker():
    from silly_kicks._nan_safety import is_nan_safe_enrichment
    assert is_nan_safe_enrichment(xs.add_xshot_occurrence)


def test_add_xshot_adds_column():
    # Use the integration fixture; here assert the column exists + dtype float.
    import pandas as pd
    actions, frames = _actions_and_frames_for_add()  # defined in test file
    X, y = _toy_xy()
    model = xs.XShotOccurrenceModel().fit(X, y)
    out = xs.add_xshot_occurrence(actions, frames, model=model, home_team_id=1)
    assert "xshot_occurrence" in out.columns
    assert out["xshot_occurrence"].dtype.kind == "f"
    assert len(out) == len(actions)


def test_add_xshot_dtype_mismatch():
    # P2 / spec §10.2: int64 actions.team_id + object frames.team_id must not
    # silently miss the join. Build actions with int64 ids and frames with str ids
    # for the SAME logical teams; the possessing-team action must still get a value.
    import pandas as pd
    actions, frames = _actions_and_frames_for_add()
    frames = frames.copy()
    frames["team_id"] = frames["team_id"].astype(str)
    frames["game_id"] = frames["game_id"].astype(str)
    X, y = _toy_xy()
    model = xs.XShotOccurrenceModel().fit(X, y)
    out = xs.add_xshot_occurrence(actions, frames, model=model, home_team_id=str(1))
    # At least one possessing-team action gets a (non-NaN) value despite the dtype skew.
    assert out["xshot_occurrence"].notna().any()


def test_add_xshot_defensive_action_is_nan():
    # S1: an action by the NON-possessing team at a scored frame gets NaN by design.
    import pandas as pd
    actions, frames = _actions_and_frames_for_add()
    # flip one action to the defending team (the team NOT in possession at its frame)
    actions = actions.copy()
    # _actions_and_frames_for_add documents which team is in possession; pick the other.
    defending_team = _other_team(actions, frames)  # helper in test file
    actions.loc[actions.index[0], "team_id"] = defending_team
    X, y = _toy_xy()
    model = xs.XShotOccurrenceModel().fit(X, y)
    out = xs.add_xshot_occurrence(actions, frames, model=model, home_team_id=1)
    assert pd.isna(out.loc[out.index[0], "xshot_occurrence"])
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/tracking/test_xshot_occurrence.py -k add_xshot -v`
Expected: FAIL — `AttributeError: ... 'add_xshot_occurrence'`

- [ ] **Step 3: Write minimal implementation** (append; import `link_actions_to_frames` + `nan_safe_enrichment`)

```python
# imports: from silly_kicks._nan_safety import nan_safe_enrichment
#          from silly_kicks.tracking.utils import link_actions_to_frames

@nan_safe_enrichment
def add_xshot_occurrence(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    model: "XShotOccurrenceModel | str | None" = None,
    links: pd.DataFrame | None = None,
    home_team_id: int | str,
    pitch_control_cache=None,
) -> pd.DataFrame:
    """Enrich SPADL actions with an ``xshot_occurrence`` column (xS at the linked frame).

    xS is the **possessing team's** shot probability: an action by the team in
    possession at its linked frame receives a value; a defensive action by the
    non-possessing team at the same frame receives NaN by design (S1). NaN
    identifiers route to NaN output (ADR-003). ``links`` skips internal linking.

    Examples
    --------
    >>> # out = add_xshot_occurrence(actions, frames, model=m, home_team_id=1)

    See NOTICE for full bibliographic citations.
    """
    m = _resolve_model(model)
    out = actions.copy()
    pointers = links if links is not None else link_actions_to_frames(actions, frames)[0]

    link_frame_ids = None
    if "frame_id" in pointers.columns:
        link_frame_ids = set(pointers["frame_id"].dropna().astype(int).tolist())

    if "xshot_occurrence" in frames.columns and frames["xshot_occurrence"].notna().any():
        scored = frames
    else:
        scored = compute_xshot_occurrence(
            frames, model=m, home_team_id=home_team_id, link_frame_ids=link_frame_ids)

    # Map each action to the xS at its linked frame + its own team.
    xcol = scored[scored["xshot_occurrence"].notna()][
        ["game_id", "period_id", "frame_id", "team_id", "xshot_occurrence"]].copy()
    linked = pointers.merge(
        actions[["action_id", "game_id", "period_id", "team_id"]], on="action_id", how="left")
    # P2: align BOTH game_id AND team_id dtypes (provider asymmetry: int64 vs object).
    for col_name in ("game_id", "team_id"):
        if len(linked) and len(xcol) and linked[col_name].dtype != xcol[col_name].dtype:
            linked[col_name] = linked[col_name].astype(str)
            xcol[col_name] = xcol[col_name].astype(str)
    merged = linked.merge(xcol, on=["game_id", "period_id", "frame_id", "team_id"], how="left")
    deduped = merged.drop_duplicates(subset=["action_id"], keep="first")
    col = deduped.set_index("action_id")["xshot_occurrence"]
    out = out.merge(col.rename("xshot_occurrence"), left_on="action_id", right_index=True, how="left")
    return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/tracking/test_xshot_occurrence.py -k add_xshot -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Lint/type, mark done (no commit).**

---

## Task 8: `xshot_occurrence_xfns` (VAEP `_frame_aware` factory)

**Files:**
- Modify: `silly_kicks/tracking/_xshot_occurrence.py`
- Test: `tests/tracking/test_xshot_occurrence.py`

Returns a single `_frame_aware` transformer emitting `xshot_occurrence_a0/_a1/_a2`. On `frames=None` returns the 3 columns NaN (ADR-005 introspection).

- [ ] **Step 1: Write the failing tests**

```python
def test_xshot_xfns_frame_aware_marker():
    fns = xs.xshot_occurrence_xfns(model=None, home_team_id=1)
    assert len(fns) == 1
    assert getattr(fns[0], "_frame_aware", False) is True


def test_xshot_xfns_introspection_nan():
    import pandas as pd
    fns = xs.xshot_occurrence_xfns(model=None, home_team_id=1)
    states = [pd.DataFrame({"action_id": [1, 2]}) for _ in range(3)]
    out = fns[0](states, None)
    assert list(out.columns) == ["xshot_occurrence_a0", "xshot_occurrence_a1", "xshot_occurrence_a2"]
    assert out.isna().all().all()
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/tracking/test_xshot_occurrence.py -k xfns -v`
Expected: FAIL — `AttributeError: ... 'xshot_occurrence_xfns'`

- [ ] **Step 3: Write minimal implementation** (append)

```python
def xshot_occurrence_xfns(*, model=None, home_team_id: int | str, pitch_control_cache=None) -> list:
    """Factory returning a FrameAwareTransformer emitting xshot_occurrence_a0/_a1/_a2.

    NOT added to any default/union xfn list until weights ship (spec §9 / D1).

    Examples
    --------
    >>> # xfns = xshot_occurrence_xfns(home_team_id=1)

    See NOTICE for full bibliographic citations.
    """
    cols = ["xshot_occurrence_a0", "xshot_occurrence_a1", "xshot_occurrence_a2"]

    def _transformer(states, frames):
        out = pd.DataFrame(index=states[0].index)
        if frames is None:
            for c in cols:
                out[c] = np.nan
            return out
        m = _resolve_model(model)
        slot_pointers = []
        link_frame_ids: set[int] = set()
        for slot in states[:3]:
            ptr = link_actions_to_frames(slot, frames)[0]
            slot_pointers.append(ptr)
            if "frame_id" in ptr.columns:
                link_frame_ids |= set(ptr["frame_id"].dropna().astype(int).tolist())
        scored = compute_xshot_occurrence(
            frames, model=m, home_team_id=home_team_id, link_frame_ids=link_frame_ids)
        for i, (slot, ptr) in enumerate(zip(states[:3], slot_pointers, strict=False)):
            enriched = add_xshot_occurrence(slot, scored, model=m, home_team_id=home_team_id, links=ptr)
            out[cols[i]] = enriched["xshot_occurrence"].to_numpy() if "xshot_occurrence" in enriched else np.nan
        return out

    _transformer._frame_aware = True  # type: ignore[attr-defined]
    _transformer.__name__ = "xshot_occurrence_xfn"
    return [_transformer]
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/tracking/test_xshot_occurrence.py -k xfns -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Lint/type, mark done (no commit).**

---

## Task 9: Atomic mirror + invariant test

**Files:**
- Modify: `silly_kicks/atomic/tracking/features.py`
- Create: `tests/invariants/test_xshot_occurrence_bounds.py`
- Test: `tests/tracking/test_xshot_occurrence_integration.py` (new file)

- [ ] **Step 1: Write the failing tests**

```python
# tests/tracking/test_xshot_occurrence_integration.py
from __future__ import annotations
import numpy as np
import pandas as pd

def test_atomic_mirror_reexports():
    from silly_kicks.atomic.tracking import features as atomic_features
    assert hasattr(atomic_features, "add_xshot_occurrence")
```

```python
# tests/invariants/test_xshot_occurrence_bounds.py
import numpy as np
from silly_kicks.tracking import _xshot_occurrence as xs

def test_predict_proba_in_unit_interval():
    rng = np.random.default_rng(1)
    import pandas as pd
    X = pd.DataFrame(rng.normal(size=(50, 27)), columns=xs.XSHOT_FEATURE_NAMES_FAITHFUL)
    y = pd.Series((rng.random(50) < 0.1).astype(int))
    p = xs.XShotOccurrenceModel().fit(X, y).predict_proba(X)
    assert np.all((p >= 0.0) & (p <= 1.0))
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/tracking/test_xshot_occurrence_integration.py tests/invariants/test_xshot_occurrence_bounds.py -v`
Expected: FAIL — atomic re-export missing (bounds test may already pass).

- [ ] **Step 3: Add the atomic re-export**

In `silly_kicks/atomic/tracking/features.py`, add alongside the existing tracking re-exports (follow the existing `add_ghost_gk` mirror pattern in that file):

```python
from silly_kicks.tracking._xshot_occurrence import add_xshot_occurrence  # noqa: F401
```

If that file uses an `__all__`, append `"add_xshot_occurrence"` to it.

- [ ] **Step 4: Run to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/tracking/test_xshot_occurrence_integration.py tests/invariants/test_xshot_occurrence_bounds.py -v`
Expected: PASS.

- [ ] **Step 5: Lint/type, mark done (no commit).**

---

## Task 10: Real-provider extraction tests (regular suite — B3)

**Files:**
- Create: `tests/tracking/test_xshot_occurrence_real_data.py`
- Test: itself

The most important coverage. Uses committed slim fixtures `tests/datasets/tracking/action_context_slim/{sportec,metrica,skillcorner,pff}_slim.parquet` via the loader idiom from `test_action_context_cross_provider.py`. No trained weights — fit a tiny model on the extracted features themselves to exercise `compute_*`.

**Velocity gap (S3):** the slim `_KEEP` set omits `vx`/`vy`, so the tests set `vx=vy=0` → `speed` is degenerate in the real-data path. Extraction shape + bounds + dtype-asymmetry (the important parts) are still validated. To exercise velocity on real data too, optionally run `derive_velocities(frames)` (from `silly_kicks.tracking.preprocess`) on the slim frames first if cheap; otherwise note the gap explicitly in a test comment so it is not mistaken for full velocity coverage.

- [ ] **Step 1: Write the tests** (these are the implementation — extraction code already exists)

```python
# tests/tracking/test_xshot_occurrence_real_data.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import _xshot_occurrence as xs

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SLIM = REPO_ROOT / "tests" / "datasets" / "tracking" / "action_context_slim"
PROVIDERS = ["sportec", "metrica", "skillcorner", "pff"]

_KEEP = {
    "game_id", "period_id", "frame_id", "time_seconds", "frame_rate", "player_id",
    "team_id", "is_ball", "is_goalkeeper", "x", "y", "z", "speed", "speed_source",
    "ball_state", "team_attacking_direction", "confidence", "visibility", "source_provider",
}

def _load(provider: str) -> pd.DataFrame:
    p = SLIM / f"{provider}_slim.parquet"
    if not p.exists():
        pytest.skip(f"{p} not committed")
    df = pd.read_parquet(p)
    frames = df[df["__kind"] == "frame"].drop(columns=["__kind"]).reset_index(drop=True)
    return frames[[c for c in frames.columns if c in _KEEP]].copy()

@pytest.mark.parametrize("provider", PROVIDERS)
def test_extract_features_real_provider(provider):
    frames = _load(provider)
    # pick one frame group with a ball + at least one of each side
    grp_key = frames.drop_duplicates(["game_id", "period_id", "frame_id"]).iloc[0]
    g = frames[(frames["game_id"] == grp_key["game_id"]) &
               (frames["period_id"] == grp_key["period_id"]) &
               (frames["frame_id"] == grp_key["frame_id"])].copy()
    g["vx"] = 0.0
    g["vy"] = 0.0
    teams = [t for t in g["team_id"].dropna().unique()]
    if len(teams) < 2:
        pytest.skip("frame lacks two teams")
    row = xs.extract_xshot_features(g, gk_team_id=teams[0], goal_x=0.0)
    assert list(row.columns) == xs.XSHOT_FEATURE_NAMES_FAITHFUL
    og = row["openGoal"].iloc[0]
    assert np.isnan(og) or (0.0 <= og <= 1.0)
    r = row["r"].iloc[0]
    assert np.isnan(r) or r >= 0.0

@pytest.mark.parametrize("provider", PROVIDERS)
def test_compute_xshot_real_provider_in_bounds(provider):
    frames = _load(provider)
    frames = frames.copy()
    frames["vx"] = 0.0
    frames["vy"] = 0.0
    # tiny in-test model trained on this provider's own extracted features
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(60, 27)),
                     columns=xs.XSHOT_FEATURE_NAMES_FAITHFUL)
    y = pd.Series((np.random.default_rng(1).random(60) < 0.1).astype(int))
    model = xs.XShotOccurrenceModel().fit(X, y)
    out = xs.compute_xshot_occurrence(frames, model=model, home_team_id=frames["team_id"].dropna().iloc[0])
    vals = out["xshot_occurrence"].dropna()
    assert vals.between(0.0, 1.0).all()

def test_real_provider_dtype_asymmetry():
    # gradientsports(pff)=Int64, kloppy providers=object — both must not crash.
    for provider in ["pff", "skillcorner"]:
        frames = _load(provider)
        frames["vx"] = 0.0; frames["vy"] = 0.0
        grp = frames.drop_duplicates(["game_id", "period_id", "frame_id"]).iloc[0]
        g = frames[(frames["frame_id"] == grp["frame_id"]) &
                   (frames["period_id"] == grp["period_id"])].copy()
        teams = [t for t in g["team_id"].dropna().unique()]
        if len(teams) >= 1:
            xs.extract_xshot_features(g, gk_team_id=teams[0], goal_x=0.0)  # no crash
```

- [ ] **Step 2: Run** — these should pass immediately (exercise existing code on real data)

Run (background — may approach 30s across 4 providers): `.venv\Scripts\python.exe -m pytest tests/tracking/test_xshot_occurrence_real_data.py -v`
Expected: PASS (or `skip` if a fixture/team is absent). If any FAILS, it has found a real provider-shape bug — fix the extractor/compute, not the test. Record what broke.

- [ ] **Step 3: Lint/type, mark done (no commit).**

---

## Task 11: ruthless HPO objective + cache-equivalence

**Files:**
- Create: `silly_kicks/tracking/_xshot_occurrence_objective.py`
- Test: `tests/tracking/test_xshot_occurrence_integration.py`

`XShotOccurrenceObjective` = ruthless `CachedObjective`: `prepare()` builds (features, labels, groups) once; `evaluate_patch()` fits XGBoost with candidate params + `GroupKFold(game_id)` CV, returns log-loss (+ PR-AUC/Brier diagnostics); `evaluate()` is the independent monolith; `assert_cache_equivalence` to 1e-9.

**Honest note (S2):** unlike TF-24 (where feature extraction is the per-trial cost the cache eliminates), here feature extraction happens UPSTREAM in the trainer (Task 12), so `prepare()` only concats pre-built X and the expensive work (XGB fit + CV) runs per-trial in BOTH paths. The `CachedObjective` shape buys little speedup here — it is kept for **consistency with the house pattern + the `assert_cache_equivalence` correctness gate**, not performance. (A plain `Objective` would be functionally equivalent; if the reviewer/maintainer prefers simplicity over consistency, swapping to `Objective` and dropping the equivalence test is acceptable — flagged for the executor.)

- [ ] **Step 1: Write the failing tests**

**Verified ruthless 0.2.1 API (from `silly_kicks/calibration/` + `tests/calibration/`):** `from ruthless import Candidate, Direction, FloatRange, OptunaConfig, assert_cache_equivalence, InProcessBackend`; `from ruthless.config.common import StoreConfig`; `from ruthless.strategies.optuna_ import OptunaStrategy`. `Candidate(id="t0", params={...})` (id REQUIRED). `assert_cache_equivalence(obj, [candidate, ...])` — takes a **list**, **no `tol=` kwarg** (1e-9 is internal). Result API: `result.best.params`, `result.best.metrics["logloss"]`. **`IntRange` is NOT used in this codebase — use `FloatRange` for all params** (round ints inside `_pinned_params`). The contract requires each patch param to vary across ≥2 values in the candidate list.

```python
def test_objective_optuna_smoke_3_trials():
    obj, _ = _build_xshot_objective()  # helper in test file: builds a 2-game synthetic fold
    from ruthless import OptunaConfig, FloatRange, Direction, InProcessBackend
    from ruthless.config.common import StoreConfig
    from ruthless.strategies.optuna_ import OptunaStrategy
    import tempfile, os
    db = os.path.join(tempfile.mkdtemp(), "xs_smoke.db")
    cfg = OptunaConfig(
        kind="optuna", metric="logloss", direction=Direction.MINIMIZE, n_trials=3, sampler="tpe",
        param_space={
            "n_estimators": FloatRange(kind="float", lo=10.0, hi=30.0),
            "max_depth": FloatRange(kind="float", lo=2.0, hi=4.0),
            "learning_rate": FloatRange(kind="float", lo=0.1, hi=0.5),
            "scale_pos_weight": FloatRange(kind="float", lo=1.0, hi=50.0),
        },
        store=StoreConfig(kind="sqlite", path=db),
    )
    result = OptunaStrategy(cfg, seed=42).run(obj, backend=InProcessBackend())
    assert result.best is not None
    assert "logloss" in result.best.metrics


def test_objective_cache_equivalence():
    from ruthless import assert_cache_equivalence
    obj, candidates = _build_xshot_objective()  # candidates: list varying each patch param >=2 values
    assert_cache_equivalence(obj, candidates)
```

`_build_xshot_objective()` helper: synthesize 2 games of frames+shots (reuse `_one_frame`-style rows across two `game_id`s with a few positive labels), build the fold dict `{provider: [(features, labels, groups), ...]}`, return `(XShotOccurrenceObjective(fold=...), candidates)` where `candidates = [Candidate(id="t0", params={...lo...}), Candidate(id="t1", params={...hi...})]` varying every `_SEARCH_KEYS` entry across the two. Keep it tiny (≤200 rows) so 3 trials run in seconds.

- [ ] **Step 2: Run to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/tracking/test_xshot_occurrence_integration.py -k objective -v`
Expected: FAIL — module/objective missing.

- [ ] **Step 3: Write minimal implementation**

```python
# silly_kicks/tracking/_xshot_occurrence_objective.py
"""ruthless CachedObjective for xShotOccurrence HPO (ADR-009 pattern).

prepare(): build the trial-invariant (X, y, groups) per provider once.
evaluate_patch(): fit XGBoost with the candidate hyperparams + GroupKFold CV,
return held-out log-loss (+ PR-AUC/Brier diagnostics).
evaluate(): independent monolith (recompute), so assert_cache_equivalence is
non-tautological to 1e-9.

NOT imported by silly_kicks/__init__ or by the inference path. Requires the
[train] extra (ruthless-efficiency[optuna] + xgboost).

See NOTICE for full bibliographic citations.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
from ruthless import Direction, penalty_metrics  # noqa: F401
from ruthless.result import Candidate, Metrics
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold

from silly_kicks.tracking._xshot_occurrence import _pinned_params

_SEARCH_KEYS = ("n_estimators", "max_depth", "learning_rate",
                "min_child_weight", "scale_pos_weight", "reg_lambda")


@dataclasses.dataclass
class _Invariant:
    X: pd.DataFrame
    y: np.ndarray
    groups: np.ndarray


def _cv_logloss(X, y, groups, params) -> tuple[float, float, float]:
    import xgboost as xgb
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2:
        n_splits = 2
    gkf = GroupKFold(n_splits=n_splits)
    lls, prs, brs = [], [], []
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = xgb.XGBClassifier(**_pinned_params(params))
        clf.fit(X.iloc[tr].to_numpy(float), y[tr])
        p = clf.predict_proba(X.iloc[te].to_numpy(float))[:, 1]
        lls.append(log_loss(y[te], p, labels=[0, 1]))
        if len(np.unique(y[te])) == 2:
            prs.append(average_precision_score(y[te], p))
        brs.append(brier_score_loss(y[te], p))
    if not lls:
        return float("inf"), float("nan"), float("nan")
    return float(np.mean(lls)), float(np.mean(prs)) if prs else float("nan"), float(np.mean(brs))


class XShotOccurrenceObjective:
    """CachedObjective: minimize held-out log-loss over XGBoost hyperparameters."""

    patch_params = frozenset(_SEARCH_KEYS)

    def __init__(self, *, fold: dict[str, list[tuple]]) -> None:
        self._fold = fold

    def prepare(self) -> _Invariant:
        Xs, ys, gs = [], [], []
        for matches in self._fold.values():
            for X, y, groups in matches:
                Xs.append(X)
                ys.append(np.asarray(y, dtype=int))
                gs.append(np.asarray(groups))
        return _Invariant(pd.concat(Xs, ignore_index=True),
                          np.concatenate(ys), np.concatenate(gs))

    def _params(self, candidate: Candidate) -> dict:
        return {k: candidate.params[k] for k in _SEARCH_KEYS if k in candidate.params}

    def evaluate_patch(self, invariant: _Invariant, candidate: Candidate) -> Metrics:
        ll, pr, br = _cv_logloss(invariant.X, invariant.y, invariant.groups, self._params(candidate))
        return {"logloss": ll, "pr_auc": pr, "brier": br}

    def evaluate(self, candidate: Candidate) -> Metrics:
        inv = self.prepare()  # independent recompute (H1)
        ll, pr, br = _cv_logloss(inv.X, inv.y, inv.groups, self._params(candidate))
        return {"logloss": ll, "pr_auc": pr, "brier": br}
```

- [ ] **Step 4: Run to verify it passes**

Run (background): `.venv\Scripts\python.exe -m pytest tests/tracking/test_xshot_occurrence_integration.py -k objective -v`
Expected: PASS (2 passed). If `assert_cache_equivalence` signature differs from TF-24's usage, mirror exactly how `tests/calibration/test_vaep_brier_objective.py` calls it.

- [ ] **Step 5: Lint/type, mark done (no commit).**

---

## Task 12: Training CLI

**Files:**
- Create: `scripts/train_xshot_occurrence.py`
- Test: `tests/tracking/test_xshot_occurrence_integration.py`

CLI reads a parquet data-dir of frames + shots, builds labels (`build_xshot_labels`) + features (`extract_xshot_features`), runs `OptunaStrategy`, fits the final model with the best params, writes the artifact (`model.json`/`metadata.json`/`SHA256SUMS`). Smoke test runs it as a **module/subprocess with cwd+PYTHONPATH set** (avoid the ghost-gk subprocess-import trap).

- [ ] **Step 1: Write the failing smoke test**

```python
def test_train_script_smoke(tmp_path):
    import subprocess, sys, os, json
    # Build a tiny synthetic data-dir: frames.parquet + shots.parquet for 2 games.
    data_dir = _write_synthetic_train_dir(tmp_path)  # helper in test file
    out_dir = tmp_path / "out"
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    result = subprocess.run(
        [sys.executable, "scripts/train_xshot_occurrence.py",
         "--data-dir", str(data_dir), "--output-dir", str(out_dir),
         "--n-trials", "3", "--horizon-seconds", "1.0"],
        capture_output=True, text=True, timeout=180,
        cwd=str(Path(__file__).resolve().parents[2]), env=env)
    assert result.returncode == 0, result.stderr
    art = out_dir / "xshot_occurrence_v1"
    assert (art / "model.json").exists()
    assert (art / "metadata.json").exists()
    assert (art / "SHA256SUMS").exists()
    meta = json.loads((art / "metadata.json").read_text())
    assert "carrier_params" in meta
    assert meta["feature_set"] == "faithful"
```

- [ ] **Step 2: Run to verify it fails**

Run (background): `.venv\Scripts\python.exe -m pytest tests/tracking/test_xshot_occurrence_integration.py -k train_script -v`
Expected: FAIL — script missing.

- [ ] **Step 3: Write the CLI** (`scripts/train_xshot_occurrence.py`)

```python
#!/usr/bin/env python
"""Train the xShotOccurrence (xS) model (TF-16).

Reads {data-dir}/*/frames.parquet + shots.parquet, builds time-windowed labels
+ faithful features, runs a ruthless Optuna study over XGBoost hyperparameters,
fits the final model on the best params, and writes a pickle-free artifact.

Requires: silly-kicks[train].  See the TF-16 spec.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--horizon-seconds", type=float, default=1.0)
    ap.add_argument("--tolerance-m", type=float, default=3.0)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=1.0)
    args = ap.parse_args()

    from ruthless import FloatRange, Direction, InProcessBackend, OptunaConfig
    from ruthless.config.common import StoreConfig
    from ruthless.strategies.optuna_ import OptunaStrategy

    from silly_kicks.tracking._xshot_occurrence import (
        XSHOT_FEATURE_NAMES_FAITHFUL, XShotOccurrenceModel, build_xshot_labels,
        extract_xshot_features, _defended_goal_x,
    )
    from silly_kicks.tracking._ball_carrier import derive_team_in_possession, infer_ball_carrier
    from silly_kicks.tracking._xshot_occurrence_objective import XShotOccurrenceObjective

    carrier_params = {"tolerance_m": args.tolerance_m, "beta": args.beta, "gamma": args.gamma}
    data_dir = Path(args.data_dir)
    fold: dict[str, list[tuple]] = {"synthetic": []}
    for game_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        frames = pd.read_parquet(game_dir / "frames.parquet")
        shots = pd.read_parquet(game_dir / "shots.parquet")
        if "vx" not in frames.columns:
            frames["vx"] = 0.0
            frames["vy"] = 0.0
        carrier = infer_ball_carrier(frames, **carrier_params)
        poss = derive_team_in_possession(frames, carrier)
        goal_map = _defended_goal_x(frames)
        rows, labels_idx = [], []
        for (gid, pid, fid), g in poss.groupby(["game_id", "period_id", "frame_id"], dropna=False):
            tip = g["team_in_possession"].iloc[0]
            if pd.isna(tip):
                continue
            others = [t for t in g["team_id"].dropna().unique() if t != tip]
            if not others:
                continue
            goal_x = goal_map.get((gid, pid, others[0]))
            if goal_x is None:
                continue
            rows.append(extract_xshot_features(g, gk_team_id=others[0], goal_x=goal_x))
            labels_idx.append({"game_id": gid, "period_id": pid,
                               "time_seconds": float(g["time_seconds"].iloc[0]), "team_in_possession": tip})
        if not rows:
            continue
        X = pd.concat(rows, ignore_index=True)[XSHOT_FEATURE_NAMES_FAITHFUL]
        fidx = pd.DataFrame(labels_idx)
        y = build_xshot_labels(fidx, shots, horizon_seconds=args.horizon_seconds).to_numpy()
        groups = fidx["game_id"].to_numpy()
        fold["synthetic"].append((X, y, groups))

    obj = XShotOccurrenceObjective(fold=fold)
    cfg = OptunaConfig(
        kind="optuna", metric="logloss", direction=Direction.MINIMIZE,
        n_trials=args.n_trials, sampler="tpe",
        param_space={
            "n_estimators": FloatRange(kind="float", lo=50.0, hi=400.0),
            "max_depth": FloatRange(kind="float", lo=2.0, hi=6.0),
            "learning_rate": FloatRange(kind="float", lo=0.02, hi=0.4, log=True),
            "min_child_weight": FloatRange(kind="float", lo=1.0, hi=20.0),
            "scale_pos_weight": FloatRange(kind="float", lo=1.0, hi=200.0, log=True),
            "reg_lambda": FloatRange(kind="float", lo=0.0, hi=5.0),
        },
        store=StoreConfig(kind="sqlite", path=str(Path(args.output_dir) / "study.db")),
    )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    result = OptunaStrategy(cfg, seed=42).run(obj, backend=InProcessBackend())
    best = dict(result.best.params)  # ruthless result API: result.best.params / .metrics
    print(f"Best params: {best}")

    inv = obj.prepare()
    model = XShotOccurrenceModel(params=best).fit(
        inv.X, pd.Series(inv.y), carrier_params=carrier_params, horizon_seconds=args.horizon_seconds)
    model.save(Path(args.output_dir) / "xshot_occurrence_v1")
    print("Wrote artifact.")


if __name__ == "__main__":
    main()
```

`_write_synthetic_train_dir(tmp_path)` test helper: create 2 `game_*/` dirs each with `frames.parquet` (full tracking schema incl. `time_seconds`, `frame_rate`, `ball_state="alive"`, a ball + GK + outfielders across ~30 frames) and `shots.parquet` (a couple of shots with `game_id/period_id/team_id/time_seconds`).

**N3 (deferred-cost note):** the trainer extracts features in a per-frame Python loop over the whole corpus. Fine as one-time training cost on the tiny smoke fixture; on the real multi-provider corpus (the maintainer run, deferred to the weights follow-up) this loop is the dominant cost and may warrant batching then. Not a this-PR concern — the smoke fixture is small.

- [ ] **Step 4: Run to verify it passes**

Run (background): `.venv\Scripts\python.exe -m pytest tests/tracking/test_xshot_occurrence_integration.py -k train_script -v`
Expected: PASS. (`result.best.params` is the verified API — see `tests/calibration/test_calibration_e2e.py:43` `result.best`.)

- [ ] **Step 5: Lint/type, mark done (no commit).**

---

## Task 13: Public exports, NOTICE, ADR-011, `[train]` extra

**Files:**
- Modify: `silly_kicks/tracking/__init__.py`
- Modify: `NOTICE`
- Create: `docs/superpowers/adrs/ADR-011-trained-model-feature-lifecycle.md`
- Modify: `pyproject.toml`
- Test: `tests/tracking/test_xshot_occurrence.py`

- [ ] **Step 1: Write the failing test**

```python
def test_public_exports():
    import silly_kicks.tracking as t
    for name in ["compute_xshot_occurrence", "add_xshot_occurrence",
                 "xshot_occurrence_xfns", "XShotOccurrenceModel", "XShotFeatureSet",
                 "extract_xshot_features"]:
        assert hasattr(t, name), name


def test_import_silly_kicks_no_xgboost():
    # P3: dependency-light import must not pull xgboost at top level. Use a FRESH
    # SUBPROCESS (the established idiom — see tests/calibration/test_import_isolation.py),
    # NOT in-process importlib.reload (submodules stay cached; sys.modules fiddling is
    # flagged unreliable by this project).
    import subprocess
    import sys
    code = (
        "import sys; import silly_kicks; "
        "import silly_kicks.tracking; "  # the new module must not eagerly import xgboost either
        "bad=[m for m in ('xgboost',) if m in sys.modules]; "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)  # noqa: S603
    assert proc.returncode == 0, f"import leaked xgboost: {proc.stdout.strip()}"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/tracking/test_xshot_occurrence.py -k "public_exports or no_xgboost" -v`
Expected: FAIL — names not exported (`test_public_exports`). Note: `test_import_silly_kicks_no_xgboost` only passes if `_xshot_occurrence.py` keeps `import xgboost` **inside** functions (lazy) — verify no module-level `import xgboost` slipped in (Task 5/6/7 all use function-local imports).

- [ ] **Step 3: Implement exports + NOTICE + ADR + extra**

In `silly_kicks/tracking/__init__.py`: add to `__all__` (alphabetical) `"XShotFeatureSet"`, `"XShotOccurrenceModel"`, `"add_xshot_occurrence"`, `"compute_xshot_occurrence"`, `"extract_xshot_features"`, `"xshot_occurrence_xfns"`; and add the import block:

```python
from ._xshot_occurrence import (
    XShotFeatureSet,
    XShotOccurrenceModel,
    add_xshot_occurrence,
    compute_xshot_occurrence,
    extract_xshot_features,
    xshot_occurrence_xfns,
)
```

In `NOTICE`, append under "Mathematical / Methodological References" (verbatim from spec §12):

```
The xShotOccurrence model (silly_kicks/tracking/_xshot_occurrence.py, TF-16)
implements the shot-occurrence (xS) component of: Pipping, J., Feng, T., &
Sabin, P. (2026). "Beyond Expected Goals: A Probabilistic Framework for Shot
Occurrences in Soccer." arXiv:2512.00203. Only the xS sub-model (probability a
shot is attempted within ~1 s of a frame) is implemented; the paper's xG and
xG+ composition are out of scope (silly-kicks values goals/threat via VAEP and
xthreat). The openGoal goal-mouth-obstruction feature follows the paper's
Appendix A construction.
```

In `pyproject.toml`, add a new extra:

```toml
train = [
    "ruthless-efficiency[optuna]>=0.2.1",
    "xgboost>=2.0,<3.0",
]
```

Create `docs/superpowers/adrs/ADR-011-trained-model-feature-lifecycle.md` using the ADR-009 table-header format. Content: Context (trained-model features — ghost-gk 1st, xS 2nd — need a uniform staging pattern); Decision (code → training pipeline → bundled/Hub weights staged across PRs; pickle-free booster/npz + SHA256SUMS; `from_variant`/`from_hub`; `[train]` for HPO deps, inference gated on the model's native runtime extra; trained-model xfns stay out of default xfn lists until weights ship); Consequences; Related (ADR-005, ADR-009, TF-18, TF-16).

- [ ] **Step 4: Run to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/tracking/test_xshot_occurrence.py -k "public_exports or no_xgboost" -v`
Expected: PASS.

- [ ] **Step 5: Full suite + lint + types (no commit)**

Run (background): `SILLY_KICKS_ASSERT_INVARIANTS=1` + `.venv\Scripts\python.exe -m pytest tests/ -m "not e2e" -q`
Then: `ruff check silly_kicks/ tests/ scripts/`, `ruff format --check silly_kicks/ tests/ scripts/`, `pyright silly_kicks/`.
Expected: all green, 0 failures. (N2: confirm the pre-existing pass count by re-running this on `main`@4.0.3 BEFORE Task 1 if you want a delta; the part-deux clone's baseline on the 3.10.19 venv was measured at session start — use that measured number, do not assume a literal "3180".)

---

## Task 14: Version bump, /final-review, single commit

**Files:**
- Modify: `pyproject.toml`, `silly_kicks/__init__.py`, `TODO.md`, `CHANGELOG.md`

- [ ] **Step 1: Bump version in all four (PROVISIONAL — C1)**

Set version to the next free minor (draft: `4.1.0`) in `pyproject.toml` (`version = `), `silly_kicks/__init__.py` (`__version__ = `), `TODO.md` ("Current release: silly-kicks X.Y.Z"), and a new dated `CHANGELOG.md` section. **Before finalizing the number, check whether the TF-24 apply-PR has merged** — if it took 4.1.0, use 4.2.0. The CHANGELOG `### Added` entry notes the model ships **untrained** (code + synthetic fixture + real-provider extraction tests) and that a future TF-24 carrier-default change is an xS retrain trigger.

- [ ] **Step 2: Update TODO.md** — move the TF-16 row out of "On Deck" to reflect code-shipped / weights-pending; update the GKDV program note (TF-16 Layer-2 code landed).

- [ ] **Step 3: Run /final-review (mad-scientist-skills:final-review) — MANDATORY**

Apply the skill. Fix any documentation drift (README/CLAUDE.md provider/feature lists, CHANGELOG, C4 diagram if architecture.dsl enumerates modules). Re-run gates after fixes.

- [ ] **Step 4: Verify version sync + full gates green**

Run: `grep '^version' pyproject.toml` and `grep '__version__' silly_kicks/__init__.py` (must match the CHANGELOG). Re-run pytest (not e2e) + ruff (both) + pyright. All green.

- [ ] **Step 5: Present diff + single commit (sentinel-gated)**

Show the user `git status --short` + `git diff --stat` + the proposed commit message. Ask the user to drop the sentinel (`!touch ~/.claude-git-approval`). Only then:

```bash
git add -A
git commit
```

Commit message ends with the trailer:
`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

## Self-review notes (author)

- **Spec coverage:** §2 module structure → Tasks 1,3,5,11,12,13; §3 surfaces → Tasks 6,7,8; §4 label → Task 4 (R2/B1 covered by `test_label_robust_to_noncontiguous_frame_id`); §5 features+geometry → Tasks 1,2,3 (C4 golden-master/union/behind-ball/past-line all in Task 2; C5 helper Task 1); §6 model+HPO+carrier-consume → Tasks 5,11 + R3 `test_inference_uses_metadata_carrier_params` (Task 6); §8 serialization → Task 5; §9 staged shipping (from_variant inert, xfns not in default lists) → Tasks 5,8; §10 tests → every task + Task 10 real-data; §12 NOTICE / §11 extras / ADR-011 → Task 13; §13 version → Task 14.
- **R-items:** R1 (no extended-cache test) — not present, only `test_extended_raises_not_implemented` (Task 3) ✓. R2 (no linkage for label) — Task 4 ✓. R3 (consume metadata params) — Task 6 ✓. R4 (PR-AUC) — diagnostics in Task 11 objective ✓. R5 (first-principles golden master) — Task 2 ✓. R6 (geometry xS-only, `[train]` name) — Task 1 / Task 13 ✓. R7 (typo) — n/a to plan.
- **Type consistency:** `XSHOT_FEATURE_NAMES_FAITHFUL`, `extract_xshot_features(...)`, `build_xshot_labels(...)`, `XShotOccurrenceModel(.fit/.predict_proba/.save/.load/.carrier_params)`, `_pinned_params`, `_defended_goal_x`, `compute_xshot_occurrence`, `add_xshot_occurrence`, `xshot_occurrence_xfns`, `IntegrityError`, `XShotOccurrenceObjective(prepare/evaluate_patch/evaluate/patch_params)` — names consistent across tasks 3–13.
- **Plan-review round 2 (TF-24 session) folded in:** P1 batched predict + compute-level benchmark (Task 6); P2 align BOTH `game_id`+`team_id` dtypes + `test_add_xshot_dtype_mismatch` (Task 7); P3 subprocess import-isolation test, not `importlib.reload` (Task 13); S1 possession semantics intentional + `test_add_xshot_defensive_action_is_nan` (Task 7); S2 honest "CachedObjective buys little speedup here" note (Task 11); S3 real-data velocity-gap note (Task 10); N1 `_defended_goal_x` outfield fallback + coverage-loss warning (Task 6); N2 don't assume literal baseline count — measured baseline = **3221 tests / 0 failures** on the branch (Task 13); N3 trainer per-frame-loop deferred-cost note (Task 12).
- **Plan-review round 3 (TF-24 session) folded in:** N-A — `infer_ball_carrier` runs on FULL frames (cross-frame `gamma` hysteresis correctness + train/serve parity); ONLY per-frame extract+predict are restricted to `link_frame_ids` (Task 6) + `test_compute_carrier_runs_on_full_frames` regression guard. N-B — compute joins on TEMPORARY string keys (`__gid`/`__tid`), never mutates the returned `game_id`/`team_id` dtypes (schema preserved) + `test_compute_preserves_id_dtypes`. Both localized to Task 6.
- **ruthless-API (verified against `silly_kicks/calibration/` + `tests/calibration/`, ruthless 0.2.1):** top-level `from ruthless import Candidate, Direction, FloatRange, OptunaConfig, assert_cache_equivalence, InProcessBackend`; `StoreConfig` from `ruthless.config.common`; `OptunaStrategy` from `ruthless.strategies.optuna_`. `Candidate(id=..., params=...)` (id required). `assert_cache_equivalence(obj, [candidates])` — list arg, no `tol` kwarg. Result: `result.best.params` / `result.best.metrics[...]`. **`IntRange` does not exist in this stack** — all `param_space` entries are `FloatRange`; the objective rounds int params inside `_pinned_params` (`_INT_PARAMS`). These are now baked into Tasks 11/12 (not left as a risk).
- **Verification before coding (one-time):** in the venv, `from ruthless import Candidate, assert_cache_equivalence, FloatRange, OptunaConfig` to confirm the symbols resolve; if any import name differs in the installed 0.2.1, grep `tests/calibration/` for the exact form and adjust. This is the only external-API surface; everything else is internal silly-kicks.
