# TF-17 xCrossAttempt — PR-A (code, untrained) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the complete `xCrossAttempt` per-frame cross-attempt-propensity code path — extractor, model, ADR-005 surfaces, HPO objective, training CLI, atomic mirror, full test suite — **untrained** (no weights), as a faithful mirror of TF-16 xShotOccurrence.

**Architecture:** Hexagonal/pure (pandas in, pandas out). A shared train/serve feature extractor is the anti-skew guarantee. `XCrossAttemptModel` is pinned-deterministic XGBoost, pickle-free (booster JSON + metadata.json + SHA256SUMS). Faithful feature set = **7 of the Cao et al. 8 confounders** (realized with silly-kicks primitives; crosser-position #7 omitted — no faithful tracking-only proxy) + a contiguous, isolatable **GK block** (the novel extension). `from_variant`/`from_hub` are wired but raise `FileNotFoundError` until PR-B.

**Tech Stack:** pandas, numpy, scikit-learn (runtime); xgboost (inference, `[xgboost]` extra, lazy); ruthless-efficiency[optuna] (HPO, `[train]` extra). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-06-03-tf17-xcross-attempt-design.md` (settled after 3 review rounds).

**Out of PR-A scope (later PRs):** maintainer training run + bundled/Hub weights + surface-ablation + substitution-sensitivity probe + xfn-list wiring (PR-B); `silly_kicks/_causal/` matching port + `scripts/validate_xcross_causal.py` + ADR-015 (PR-C).

**Plan review (resolved 2026-06-03 — cross-session code review, verified against source).**
- **C1** geometry convention was inverted (used `PITCH_LENGTH` for endline/box/posts) — fixed: attacked goal is at `gr_x = 0` (verified `to_goal_relative_x(95, goal_x=105) == 10`), so `dist_endline = cx`, box = `gr_x <= 16.5`, posts at `gr_x = 0`; added value-asserting post tests.
- **C2** the shared label helper keyed on `team_id`, but xS's `build_xshot_labels` reads `frames_index["team_in_possession"]` (verified line 299) and groups events by `team_id` (line 295) with `dropna=False` — fixed: helper takes `frame_team_col`/`event_team_col`; golden rebuilt on the real `team_in_possession` schema; xCross `prepare`/`build_xcross_labels` now emit/consume `team_in_possession`.
- **C3 — pushed back (verified misread):** `ball_carrier_player_id` coercion (`_ball_carrier.py:327`) is **guarded** by `if str(pid_dtype)=="Int64"` (line 326) and `derive_team_in_possession` is a plain merge (line 497) — so the carrier id is kept *dtype-consistent with the frame's `player_id`*, NOT forced to Int64/`<NA>` on string-id providers. It does **not** go all-NaN. Still **hardened** the carrier match (NA guard + str-coercion) and added a real-data assertion (`test_carrier_and_gk_blocks_mostly_resolved_on_string_ids`, H3) that proves carrier/GK blocks are mostly non-NaN on string-id providers.
- **H1** `_defended_goal_x` per-frame filter → precomputed `_build_goal_map` dict once (mirror xS:571-585); `compute` must reuse it on the serve hot path.
- **H2** `crosser_role` (#7) was collinear with `dist_endline` (both `= cx`) → **dropped** in PR-A (spec Q2 fallback), NOTICE-documented, candidate for `extended`. Feature count 17→16.
- **H3** real-data tests strengthened from vacuous `.dropna()` ranges to non-NaN-fraction assertions.
- **M1** model fit/predict now `.to_numpy(dtype=float)` + xgb≥2 check (mirror xS:406,423). **M2** label helper uses `dropna=False` + `dtype=int` + NaN-team & dtype tests. **M3** `ball_state` column-presence guard + row-level (ball-row) check. **M4** xfns test now invokes `fn(states, frames=None)` (was tautological). **M5** `_pinned_params` must copy xS's literal default dict (incl. `learning_rate`, `verbosity=0`).
- **Nits:** dead `xc_extract` removed; `math.isnan` for the score NaN test; `np.sign(...) or 1.0` commented; `ten_minute_warning` is `period in (1,2)` only — **ET periods (3/4) get 0 by design** (stated); version coordination with PR-S81 in Task 14.

**Plan review round 2 (resolved 2026-06-03 — PR-S81 session, verified against source).**
- **PA-H1 (HIGH) — accepted, WIRED (owner-confirmed option a).** `score_differential` (#1) was half-wired (accepted by extract, supplied by neither prepare nor compute → dead constant-NaN). Now realized end-to-end by reusing ghost-GK's `_build_score_lookup` (verified reusable, `_ghost_gk.py:233`; `compute_ghost_gk` already uses the optional-`actions` pattern): `prepare` builds it from its `actions`; `compute_xcross_attempt`/`add_xcross_attempt` gain an optional `actions=` kwarg (NaN-tolerant when omitted). Possessing-team-signed. Tests added (train sign/realization + serve spy). **6→7 of the paper's 8 confounders realized** (#7 dropped per H2/spec-Q2); NOTICE states #7 omitted + #1 needs match-context.
- **PA-M1 — accepted, verified + locked.** Confirmed `time_seconds` is **per-period** (kloppy `frame.timestamp` is period-relative, `tracking/kloppy.py:123`). Kept the simple threshold; added a comment + `test_ten_minute_warning_period2_early_is_zero` (and a late-half positive test) to lock the contract.
- **PA-M2 — accepted, de-vacuumed.** The `hasattr`-monkeypatch cache-trap test could false-green. Replaced with a **fresh-subprocess** assertion that `"silly_kicks.tracking.pitch_control" not in sys.modules` after a faithful extraction — can't pass vacuously.
- **PA-L1** self-review feature count 17→16 fixed; **PA-L2** `_dominant_region_area` per-frame meshgrid flagged for a PR-B real-data wall-clock check (coarsen `res` if needed); **PA-L3** `_build_goal_map` now excludes the `"ball"` team.
- **`home_team_id`** is now genuinely USED (score sign) in `prepare`/`compute` when `actions` are supplied — no longer pure inherited-debt there (still symmetry-only on `add`/`xfns` when no actions).

---

## File Structure

| File | Create/Modify | Responsibility |
|---|---|---|
| `silly_kicks/tracking/_occurrence_labels.py` | **Create** | `_build_occurrence_labels` shared time-windowed occurrence label (M-3) |
| `silly_kicks/tracking/_xshot_occurrence.py` | **Modify** | refactor `build_xshot_labels` → thin wrapper over `_build_occurrence_labels` (bit-identical, R2-L2) |
| `silly_kicks/tracking/_xcross_attempt.py` | **Create** | feature constants, `extract_xcross_features`, `build_xcross_labels`, `prepare_xcross_training_data`, `XCrossAttemptModel`, `compute_xcross_attempt`, `add_xcross_attempt`, `xcross_attempt_xfns` |
| `silly_kicks/tracking/_xcross_attempt_objective.py` | **Create** | `XCrossAttemptObjective` ruthless `CachedObjective` |
| `scripts/train_xcross_attempt.py` | **Create** | training CLI (mirror `train_xshot_occurrence.py`) |
| `silly_kicks/tracking/__init__.py` | **Modify** | export the 8 public symbols |
| `silly_kicks/atomic/tracking/features.py` | **Modify** | re-export `add_xcross_attempt`, `xcross_attempt_xfns` |
| `silly_kicks/tracking/_geometry.py` | **Reuse (no edit)** | `to_goal_relative_x`, `GOAL_Y`, `PITCH_LENGTH/WIDTH`, `GEOMETRY_VERSION` |
| `NOTICE` | **Modify** | Cao et al. attribution (with H2 faithfulness caveat) |
| `pyproject.toml`, `silly_kicks/__init__.py`, `TODO.md`, `CHANGELOG.md` | **Modify** | version bump (4-file sync) + ADR-011 TF-17 note |
| `tests/tracking/test_xcross_attempt.py` | **Create** | unit |
| `tests/tracking/test_xcross_attempt_integration.py` | **Create** | integration |
| `tests/tracking/test_xcross_attempt_real_data.py` | **Create** | real-provider extraction (regular suite) |

**Reference module (read it before starting — TF-17 mirrors it):** `silly_kicks/tracking/_xshot_occurrence.py`. Key line anchors: `extract_xshot_features` (148–257), `build_xshot_labels` (260–312), `prepare_xshot_training_data` (598–708), `subsample_negatives` (711–744), `compute_xshot_occurrence` (747–836), `add_xshot_occurrence` (839–889), `xshot_occurrence_xfns` (892–932), `XShotOccurrenceModel` (354–560). The objective: `silly_kicks/tracking/_xshot_occurrence_objective.py`. The CLI: `scripts/train_xshot_occurrence.py`.

---

## Conventions to follow (verified in the codebase)

- `warnings.warn(..., stacklevel=2)` on every warning.
- `@nan_safe_enrichment` from `silly_kicks._nan_safety` on `add_*` (sets `_nan_safe = True`; auto-discovered by `tests/test_enrichment_nan_safety.py`).
- `_frame_aware = True` on the xfns inner closure (frame-dispatch marker).
- Provenance-skip guard in `add_*`: `if not any(c in out.columns for c in provenance_cols):` before merging `frame_id`/`time_offset_seconds`/`link_quality_score`/`n_candidate_frames`.
- `links` kwarg on `add_*` to skip internal `link_actions_to_frames`.
- Cross actiontype: `silly_kicks.spadl.config.actiontype_id["cross"]` == 1.
- Carrier: `from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS, infer_ball_carrier, derive_team_in_possession` (`DEFAULT_CARRIER_PARAMS = {"tolerance_m": 3.0, "beta": 0.0, "gamma": 0.25}`).
- Geometry: `from silly_kicks.tracking import _geometry as _geo` (`_geo.to_goal_relative_x(x, goal_x=...)`, `_geo.GOAL_Y == 34.0`, `_geo.PITCH_LENGTH == 105.0`, `_geo.PITCH_WIDTH == 68.0`, `_geo.GEOMETRY_VERSION == "goal-relative-1"`).

**Run the suite with:** `python -m pytest tests/ -m "not e2e" -v --tb=short`
**Lint trio before every push (CI parity):** `ruff check silly_kicks/ tests/ scripts/` && `ruff format --check silly_kicks/ tests/ scripts/` && `pyright silly_kicks/`.

**Commit policy (per user rule):** ONE feature branch, work committed at the END when fully tested — do NOT make standalone doc/intermediate commits. The per-task "Commit" steps below stage work on the feature branch; squash into a single PR commit at the end (Task 16). Branch first (do not commit on `main`). The commit sentinel (`~/.claude-git-approval`) requires explicit per-commit approval — present the diff and HOLD.

---

## Task 0: Branch + module constants skeleton

**Files:**
- Create: `silly_kicks/tracking/_xcross_attempt.py` (constants only)

- [ ] **Step 1: Create the feature branch**

```bash
git checkout -b feat/tf17-xcross-attempt-code
```

- [ ] **Step 2: Write the constants + feature-name lists** into `silly_kicks/tracking/_xcross_attempt.py`

```python
"""TF-17 xCrossAttempt: per-frame cross-attempt-propensity model (GKDV Layer 2).

Cross analogue of xShotOccurrence (TF-16). STATE-anchored occurrence surface:
P(the in-possession team attempts a cross within ~horizon of a frame). Inspired by
Cao et al. (2025, arXiv:2505.11841); extended with goalkeeper-position confounders.
See NOTICE for full bibliographic citations.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from silly_kicks.tracking import _geometry as _geo
from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS

XCrossFeatureSet = Literal["faithful", "extended"]

# Paper confounders (realized with silly-kicks primitives) + ball geometry.
# NOTE (review C1): goal-relative convention is **attacked goal at gr_x = 0**
# (verified: _geometry.to_goal_relative_x(95, goal_x=105) == 10.0). Every distance-to-goal /
# box / post formula below measures from gr_x = 0, NOT PITCH_LENGTH.
# NOTE (review H2): paper confounder #7 "crosser position (FW/MF/DF)" is DROPPED in PR-A — a
# tracking longitudinal-role proxy == the carrier's gr_x == `dist_endline`, i.e. collinear with
# #5 (zero added signal). The paper's #7 is categorical event metadata; a faithful proxy needs
# season/role aggregation not available in PR-A. Documented in NOTICE; candidate for `extended`.
_BALL_FEATURES = ["ball_r", "ball_theta", "ball_speed"]
_CONFOUNDERS = [
    "score_differential",     # #1 (NaN if no score lookup supplied)
    "dist_nearest_def",       # #2 carrier -> nearest opponent
    "space_controlled",       # #3 Voronoi dominant-region area proxy (cache-free)
    "dist_nearest_teammate",  # #4 carrier -> nearest teammate
    "dist_endline",           # #5 carrier -> attacked goal line (== carrier gr_x)
    "box_off_def_ratio",      # #6 attackers/defenders inside the attacked penalty box
    "ten_minute_warning",     # #8 final 10 min of the half (0/1)
]
# The NOVEL, contiguous, separately-droppable GK block (the headline extension).
XCROSS_GK_BLOCK = [
    "gk_r", "gk_theta", "gk_lateral_offset",
    "gk_dist_near_post", "gk_dist_far_post", "gk_carrier_side",
]
XCROSS_FEATURE_NAMES_FAITHFUL = _BALL_FEATURES + _CONFOUNDERS + XCROSS_GK_BLOCK  # 16 (3+7+6)

# Domain-filter constants (M1 re-justified; M-4 corridor = cross_zone dense-zone, not SkillCorner's).
_WIDE_Y_LOW = 14.0          # cross_zone wide corridor (specialty.py:54)
_WIDE_Y_HIGH = 54.0
_ADVANCE_M = 35.0           # permissive default (SkillCorner _is_cross start_x>70 == 35 m); PR-B re-selects
_BOX_DEPTH_M = 16.5         # penalty area depth
_BOX_HALF_WIDTH_M = 20.16   # penalty area half-width (40.32/2)
_GOAL_HALF_WIDTH_M = 3.66   # goal width 7.32 / 2

_DEFAULT_CROSS_TYPES = ("cross",)   # open-play only (corner/freekick excluded by default)
_DEFAULT_CARRIER_PARAMS = DEFAULT_CARRIER_PARAMS

_HF_REPO_ID = "silly-kicks/xcross-attempt-v1"
_MODEL_VERSION = "1.0.0"
_XCROSS_WEIGHTS_ROOT = Path(__file__).parent / "_xcross_weights"
_VARIANT_CACHE: dict = {}
_INT_PARAMS = ("n_estimators", "max_depth", "min_child_weight")
```

- [ ] **Step 3: Verify it imports**

Run: `python -c "import silly_kicks.tracking._xcross_attempt as m; print(len(m.XCROSS_FEATURE_NAMES_FAITHFUL), m.XCROSS_GK_BLOCK)"`
Expected: `16 ['gk_r', 'gk_theta', 'gk_lateral_offset', 'gk_dist_near_post', 'gk_dist_far_post', 'gk_carrier_side']`

- [ ] **Step 4: Stage**

```bash
git add silly_kicks/tracking/_xcross_attempt.py
```

---

## Task 1: Shared occurrence-label helper + xS refactor (M-3, R2-L2)

**Files:**
- Create: `silly_kicks/tracking/_occurrence_labels.py`
- Modify: `silly_kicks/tracking/_xshot_occurrence.py` (`build_xshot_labels` → wrapper)
- Test: `tests/tracking/test_occurrence_labels.py`, and a golden in `tests/tracking/test_xshot_occurrence.py`

- [ ] **Step 1: Read the existing `build_xshot_labels`** (`_xshot_occurrence.py:260-312`). VERIFIED FACTS the helper must preserve (review C2/M2): (a) the frames-side team column is **`team_in_possession`** (line 299), NOT `team_id`; (b) events (shots) are grouped by **`team_id`** (line 295); (c) `groupby(..., dropna=False)` (line 295); (d) `np.zeros(len(...), dtype=int)` → `pd.Series` (lines 290, 312). The two team columns are asymmetric — parameterize them.

- [ ] **Step 2: Write the failing test** in `tests/tracking/test_occurrence_labels.py`

```python
import numpy as np
import pandas as pd
from silly_kicks.tracking._occurrence_labels import _build_occurrence_labels


def _frames_index():
    # frames-side team column is `team_in_possession` (matches xS, line 299).
    return pd.DataFrame({
        "game_id": ["g"] * 6, "period_id": [1] * 6,
        "frame_id": [10, 11, 12, 13, 14, 15],
        "time_seconds": [0.0, 0.4, 0.8, 1.2, 1.6, 2.0],
        "team_in_possession": ["A"] * 6,
    })


def test_occurrence_label_within_horizon():
    fidx = _frames_index()
    events = pd.DataFrame({"game_id": ["g"], "period_id": [1], "team_id": ["A"], "time_seconds": [1.0]})
    y = _build_occurrence_labels(fidx, events, horizon=1.0, frame_team_col="team_in_possession")
    assert list(y) == [1, 1, 1, 0, 0, 0]


def test_occurrence_label_robust_to_frame_id_gap():
    fidx = _frames_index()
    fidx.loc[3:, "frame_id"] = [99, 100, 101]  # non-contiguous ids, intact time
    events = pd.DataFrame({"game_id": ["g"], "period_id": [1], "team_id": ["A"], "time_seconds": [1.0]})
    y = _build_occurrence_labels(fidx, events, horizon=1.0, frame_team_col="team_in_possession")
    assert list(y) == [1, 1, 1, 0, 0, 0]  # unchanged -> no frame_id arithmetic


def test_occurrence_label_no_period_bleed():
    fidx = _frames_index()
    fidx.loc[3:, "period_id"] = 2
    events = pd.DataFrame({"game_id": ["g"], "period_id": [2], "team_id": ["A"], "time_seconds": [1.3]})
    y = _build_occurrence_labels(fidx, events, horizon=1.0, frame_team_col="team_in_possession")
    assert list(y) == [0, 0, 0, 1, 0, 0]


def test_occurrence_label_nan_team_event_dropna_false():
    """M2: groupby(dropna=False) — a NaN-team event labels nothing but must not raise."""
    fidx = _frames_index()
    events = pd.DataFrame({"game_id": ["g"], "period_id": [1], "team_id": [np.nan], "time_seconds": [0.5]})
    y = _build_occurrence_labels(fidx, events, horizon=1.0, frame_team_col="team_in_possession")
    assert list(y) == [0, 0, 0, 0, 0, 0]


def test_occurrence_label_dtype_is_int():
    """M2: preserve xS's platform-int dtype (np.zeros(dtype=int))."""
    fidx = _frames_index()
    events = pd.DataFrame({"game_id": ["g"], "period_id": [1], "team_id": ["A"], "time_seconds": [1.0]})
    y = _build_occurrence_labels(fidx, events, horizon=1.0, frame_team_col="team_in_possession")
    assert y.dtype == np.dtype(int)
```

- [ ] **Step 3: Run to verify failure**

Run: `python -m pytest tests/tracking/test_occurrence_labels.py -v`
Expected: FAIL (`ModuleNotFoundError: _occurrence_labels`).

- [ ] **Step 4: Implement `_occurrence_labels.py`** — parameterized team columns, `dropna=False`, `dtype=int` (C2/M2)

```python
"""Shared time-windowed occurrence label for trained-frame models (xS, xCross).

A frame is positive iff an event by the same (game, period, team) occurs with
``time_seconds`` in ``[t, t + horizon]``. No ``frame_id`` arithmetic (providers are
not frame-contiguous); per-period ``searchsorted`` over sorted ``time_seconds``.

The frames-side team column (``team_in_possession``) and the events-side team column
(``team_id``) differ in the house schema, so both are parameters. ``groupby(dropna=False)``
+ platform-int output mirror the pre-extraction ``build_xshot_labels`` byte-for-byte (R2-L2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _build_occurrence_labels(
    frames_index: pd.DataFrame,
    events: pd.DataFrame,
    *,
    horizon: float,
    frame_team_col: str,
    event_team_col: str = "team_id",
) -> np.ndarray:
    y = np.zeros(len(frames_index), dtype=int)  # platform int — matches xS exactly
    if len(events) == 0:
        return y
    ev_groups: dict[tuple, np.ndarray] = {}
    for key, grp in events.groupby(["game_id", "period_id", event_team_col], dropna=False):
        ev_groups[key] = np.sort(grp["time_seconds"].to_numpy(dtype=float))
    gids = frames_index["game_id"].to_numpy()
    pids = frames_index["period_id"].to_numpy()
    tcol = frames_index[frame_team_col].to_numpy()
    ts = frames_index["time_seconds"].to_numpy(dtype=float)
    for i in range(len(frames_index)):
        arr = ev_groups.get((gids[i], pids[i], tcol[i]))
        if arr is None:
            continue
        lo = float(ts[i])
        left = np.searchsorted(arr, lo, side="left")
        if left < len(arr) and arr[left] <= lo + horizon:
            y[i] = 1
    return y
```

- [ ] **Step 5: Run to verify pass**

Run: `python -m pytest tests/tracking/test_occurrence_labels.py -v`
Expected: PASS (5 tests).

- [ ] **Step 6: Capture an xS golden BEFORE refactoring** — add to `tests/tracking/test_xshot_occurrence.py`, using the **real `prepare_xshot_training_data` frames_index schema** (`team_in_possession`, `_xshot_occurrence.py:686-691`) so the golden actually runs against current xS code (C2):

```python
def test_build_xshot_labels_bit_identical_after_refactor():
    """R2-L2: refactoring build_xshot_labels onto _build_occurrence_labels must not shift
    xS labels. frames_index uses the real `team_in_possession` schema (NOT team_id)."""
    import numpy as np, pandas as pd
    from silly_kicks.tracking._xshot_occurrence import build_xshot_labels
    frames_index = pd.DataFrame({
        "game_id": ["g"] * 5, "period_id": [1] * 5,
        "time_seconds": [0.0, 0.5, 1.0, 1.5, 2.0],
        "team_in_possession": ["A"] * 5,   # <-- real xS column (line 299), NOT team_id
    })
    shots = pd.DataFrame({"game_id": ["g"], "period_id": [1], "team_id": ["A"], "time_seconds": [1.2]})
    y = build_xshot_labels(frames_index, shots, horizon_seconds=1.0)
    np.testing.assert_array_equal(np.asarray(y), np.array([0, 1, 1, 1, 0]))  # GOLDEN
    assert np.asarray(y).dtype == np.dtype(int)
```

Run it against the CURRENT (pre-refactor) code: `python -m pytest tests/tracking/test_xshot_occurrence.py::test_build_xshot_labels_bit_identical_after_refactor -v`
Expected: PASS. **If the golden differs, correct it to match current output FIRST** (freezes current behavior), then proceed.

- [ ] **Step 7: Refactor `build_xshot_labels`** (`_xshot_occurrence.py:260`) into a thin wrapper, passing `frame_team_col="team_in_possession"` (its real column, line 299). Keep the public signature + `pd.Series` return:

```python
from silly_kicks.tracking._occurrence_labels import _build_occurrence_labels

def build_xshot_labels(frames_index, shots, *, horizon_seconds=1.0):
    """xS shot-occurrence label. Thin wrapper over the shared occurrence helper."""
    y = _build_occurrence_labels(
        frames_index, shots, horizon=horizon_seconds, frame_team_col="team_in_possession")
    return pd.Series(y, index=frames_index.index)
```

- [ ] **Step 8: Run the xS golden + full xS suite to prove bit-identical**

Run: `python -m pytest tests/tracking/test_xshot_occurrence.py -v`
Expected: PASS (golden + all existing xS label tests unchanged).

- [ ] **Step 9: Stage**

```bash
git add silly_kicks/tracking/_occurrence_labels.py silly_kicks/tracking/_xshot_occurrence.py tests/tracking/test_occurrence_labels.py tests/tracking/test_xshot_occurrence.py
```

---

## Task 2: `extract_xcross_features` — ball geometry + paper confounders (no GK block yet)

**Files:**
- Modify: `silly_kicks/tracking/_xcross_attempt.py`
- Test: `tests/tracking/test_xcross_attempt.py`

- [ ] **Step 1: Write failing tests** in `tests/tracking/test_xcross_attempt.py`

```python
import math
import numpy as np
import pandas as pd
import pytest
from silly_kicks.tracking import _xcross_attempt as xc


def _one_frame():
    """A single (game,period,frame): ball wide-left near the byline, carrier = A1,
    one defender, one teammate, defending GK = B-GK. Attacked goal at x=105."""
    return pd.DataFrame({
        "game_id": ["g"] * 5, "period_id": [1] * 5, "frame_id": [7] * 5,
        "time_seconds": [40.0] * 5, "team_id": ["A", "A", "B", "B", "ball"],
        "player_id": ["A1", "A2", "B1", "Bgk", None],
        "x": [95.0, 88.0, 100.0, 104.0, 95.0],
        "y": [10.0, 30.0, 12.0, 34.0, 10.0],
        "vx": [1.0, 0.0, 0.0, 0.0, 1.0], "vy": [0.0, 0.0, 0.0, 0.0, 0.0],
        "is_ball": [False, False, False, False, True],
        "is_goalkeeper": [False, False, False, True, False],
        "ball_state": ["alive"] * 5,
    })


def test_extract_features_faithful_shape():
    feats = xc.extract_xcross_features(
        _one_frame(), gk_team_id="B", goal_x=105.0, carrier_player_id="A1")
    assert list(feats.columns) == xc.XCROSS_FEATURE_NAMES_FAITHFUL
    assert len(feats) == 1
    assert feats.shape[1] == 16  # 3 ball + 7 confounders (#7 dropped) + 6 GK
    assert "crosser_role" not in feats.columns  # H2: dropped (collinear with dist_endline)


def test_extended_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        xc.extract_xcross_features(_one_frame(), gk_team_id="B", goal_x=105.0,
                                   carrier_player_id="A1", feature_set="extended")


def test_ten_minute_warning_off_early():
    feats = xc.extract_xcross_features(_one_frame(), gk_team_id="B", goal_x=105.0,
                                       carrier_player_id="A1")
    assert feats["ten_minute_warning"].iloc[0] == 0  # t=40s, not final 10 min


def test_ten_minute_warning_period2_early_is_zero():
    """PA-M1: locks the PER-PERIOD time_seconds contract — a period-2 frame early in the half
    (t=120s) must be 0. If time_seconds were match-cumulative, t would be ~2820 -> wrongly 1."""
    frame = _one_frame()
    frame["period_id"] = 2
    frame["time_seconds"] = 120.0
    feats = xc.extract_xcross_features(frame, gk_team_id="B", goal_x=105.0, carrier_player_id="A1")
    assert feats["ten_minute_warning"].iloc[0] == 0


def test_ten_minute_warning_late_half_is_one():
    frame = _one_frame()
    frame["time_seconds"] = 40 * 60.0  # 40th minute -> within the final 10 of a 45-min half
    feats = xc.extract_xcross_features(frame, gk_team_id="B", goal_x=105.0, carrier_player_id="A1")
    assert feats["ten_minute_warning"].iloc[0] == 1


def test_dist_endline_goal_relative():
    feats = xc.extract_xcross_features(_one_frame(), gk_team_id="B", goal_x=105.0,
                                       carrier_player_id="A1")
    # carrier A1 at x=95, attacked goal at 105 -> 10 m from endline
    assert feats["dist_endline"].iloc[0] == pytest.approx(10.0, abs=1e-6)


def test_box_ratio_counts():
    """Box at the attacked goal in GOAL-RELATIVE coords: gr_x <= 16.5 (gr_x = 105 - x) AND
    |y-34| <= 20.16 (y in [13.84, 54.16]). Build a frame with explicit box occupants — the
    wide carrier itself sits OUTSIDE the box (that's the point of a cross)."""
    frame = pd.DataFrame({
        "game_id": ["g"] * 6, "period_id": [1] * 6, "frame_id": [7] * 6,
        "time_seconds": [40.0] * 6, "team_id": ["A", "A", "A", "B", "B", "ball"],
        "player_id": ["A1", "Ax1", "Ax2", "Dx1", "Bgk", None],
        "x":  [95.0, 100.0, 98.0, 101.0, 104.0, 95.0],   # gr_x = 105-x: A1=10, Ax1=5, Ax2=7, Dx1=4, Bgk=1
        "y":  [10.0, 34.0, 40.0, 30.0, 34.0, 10.0],       # A1 wide (out of box); Ax1/Ax2/Dx1 central (in)
        "vx": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0], "vy": [0.0] * 6,
        "is_ball": [False, False, False, False, False, True],
        "is_goalkeeper": [False, False, False, False, True, False],
        "ball_state": ["alive"] * 6,
    })
    feats = xc.extract_xcross_features(frame, gk_team_id="B", goal_x=105.0, carrier_player_id="A1")
    # in box: attackers Ax1(5,34), Ax2(7,40) -> off=2; defender Dx1(4,30) -> def=1; Bgk GK-excluded;
    # A1(10,10) is wide (|10-34|=24 > 20.16) -> out. ratio = 2/1 = 2.0
    assert feats["box_off_def_ratio"].iloc[0] == pytest.approx(2.0, abs=1e-6)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/tracking/test_xcross_attempt.py -v`
Expected: FAIL (`extract_xcross_features` not defined).

- [ ] **Step 3: Implement `extract_xcross_features`** (append to `_xcross_attempt.py`). GK-block columns are filled with `np.nan` here; Task 3 implements them.

```python
def _polar(dx: float, dy: float) -> tuple[float, float]:
    return math.hypot(dx, dy), math.atan2(dy, dx)


def _nearest_dist(origin_xy, pts_xy) -> float:
    if not pts_xy:
        return np.nan
    return float(min(math.hypot(px - origin_xy[0], py - origin_xy[1]) for px, py in pts_xy))


def _dominant_region_area(carrier_xy, all_xy, *, res: float = 3.0) -> float:
    """Cache-free 'space controlled' proxy: fraction of pitch grid cells whose
    nearest player is the carrier x pitch area. numpy-only nearest-player Voronoi
    approximation; NO pitch-control cache (locks the TF-19 counterfactual guarantee).

    PA-L2 perf note: this builds a full-pitch meshgrid PER FRAME (~1800 cells x ~22 players at
    res=3.0). The §11.6 guard is structural (call-count), not wall-clock, so PR-B MUST add a
    wall-clock sanity check on real-data extraction (a ~4M-row match runs this per wide-area frame);
    coarsen `res` (e.g. 5.0) if it dominates. It is the only non-vectorized-across-frames feature."""
    if carrier_xy is None or not all_xy:
        return np.nan
    xs = np.arange(res / 2, _geo.PITCH_LENGTH, res)
    ys = np.arange(res / 2, _geo.PITCH_WIDTH, res)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.asarray(all_xy)  # (P, 2)
    d2 = (gx[..., None] - pts[:, 0]) ** 2 + (gy[..., None] - pts[:, 1]) ** 2
    nearest = d2.argmin(axis=-1)
    carrier_idx = int(np.argmin(((pts[:, 0] - carrier_xy[0]) ** 2 + (pts[:, 1] - carrier_xy[1]) ** 2)))
    frac = float((nearest == carrier_idx).mean())
    return frac * _geo.PITCH_LENGTH * _geo.PITCH_WIDTH


def extract_xcross_features(
    frame_data: pd.DataFrame, *, gk_team_id, goal_x: float,
    carrier_player_id, feature_set: XCrossFeatureSet = "faithful",
    score_differential: float = np.nan,
) -> pd.DataFrame:
    if feature_set != "faithful":
        raise NotImplementedError(
            "xCrossAttempt feature_set='extended' is a deferred extension point; "
            "only 'faithful' is implemented (TF-17 PR-A). See the design spec."
        )
    f = frame_data
    is_ball = f["is_ball"].to_numpy(dtype=bool)
    is_gk = f["is_goalkeeper"].to_numpy(dtype=bool)
    team = f["team_id"].to_numpy()
    pid = f["player_id"].to_numpy()
    gr_x = np.array([_geo.to_goal_relative_x(x, goal_x=goal_x) for x in f["x"].to_numpy()])
    y = f["y"].to_numpy()

    out = {name: np.nan for name in XCROSS_FEATURE_NAMES_FAITHFUL}

    # Ball (ball-anchored)
    if is_ball.any():
        bx, by = gr_x[is_ball][0], y[is_ball][0]
        out["ball_r"], out["ball_theta"] = _polar(bx, by - _geo.GOAL_Y)
        bvx = f.loc[is_ball, "vx"].to_numpy()[0]
        bvy = f.loc[is_ball, "vy"].to_numpy()[0]
        out["ball_speed"] = math.hypot(bvx, bvy)

    # Carrier-anchored geometry.
    # C3-hardening: match the carrier by id with NA guard + str-coercion so it works whether the
    # frame player_id is native string (kloppy/sportec/skillcorner/metrica) OR Int64 (gradientsports).
    # `ball_carrier_player_id` from infer_ball_carrier is kept dtype-consistent with frame player_id
    # (conditional Int64 coercion, _ball_carrier.py:326-329), so this never silently all-NaNs; the
    # str-coerce is belt-and-suspenders against the known Int64-vs-string id gotcha.
    carrier_mask = np.zeros(len(f), dtype=bool)
    if carrier_player_id is not None and carrier_player_id == carrier_player_id:  # not None / not NA
        carrier_mask = (pid.astype(str) == str(carrier_player_id)) & ~is_ball
    if carrier_mask.any():
        cx, cy = gr_x[carrier_mask][0], y[carrier_mask][0]
        carrier_team = team[carrier_mask][0]
        opp = [(gr_x[i], y[i]) for i in range(len(f)) if not is_ball[i] and team[i] != carrier_team]
        mate = [(gr_x[i], y[i]) for i in range(len(f))
                if not is_ball[i] and team[i] == carrier_team and str(pid[i]) != str(carrier_player_id)]
        all_xy = [(gr_x[i], y[i]) for i in range(len(f)) if not is_ball[i]]
        out["dist_nearest_def"] = _nearest_dist((cx, cy), opp)
        out["dist_nearest_teammate"] = _nearest_dist((cx, cy), mate)
        out["dist_endline"] = float(cx)  # C1: attacked goal at gr_x=0 -> distance to endline IS cx
        out["space_controlled"] = _dominant_region_area((cx, cy), all_xy)

    # #6 off/def ratio in the attacked box. C1: attacked box is goal-relative gr_x <= 16.5
    # (attacked goal at gr_x=0), |y-34| <= 20.16.
    in_box = (gr_x <= _BOX_DEPTH_M) & (np.abs(y - _geo.GOAL_Y) <= _BOX_HALF_WIDTH_M) & ~is_ball
    if carrier_mask.any():
        carrier_team = team[carrier_mask][0]
        n_off = int(((team == carrier_team) & in_box & ~is_gk).sum())
        n_def = int(((team != carrier_team) & in_box & ~is_gk).sum())
        out["box_off_def_ratio"] = float(n_off / n_def) if n_def > 0 else float(n_off)

    # #1 score differential (passed in; NaN at serve unless a score lookup supplied it)
    out["score_differential"] = float(score_differential) if not (
        score_differential is None or math.isnan(score_differential)) else np.nan

    # #8 ten-minute warning (final 10 min of a 45-min half). PA-M1: relies on time_seconds being
    # PER-PERIOD (resets to 0 each half) — verified: the kloppy gateway sets
    # time_seconds = frame.timestamp.total_seconds() and kloppy timestamps are period-relative
    # (sportec native expects the same contract). ET periods (3/4/5) get 0 by design (the paper's
    # feature is "final 10 min of a half"). The per-period assumption is locked by
    # test_ten_minute_warning_period2_early_is_zero.
    t = float(f["time_seconds"].iloc[0])
    period = int(f["period_id"].iloc[0])
    out["ten_minute_warning"] = 1 if period in (1, 2) and t >= 35 * 60 else 0

    return pd.DataFrame([out], columns=XCROSS_FEATURE_NAMES_FAITHFUL)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/tracking/test_xcross_attempt.py -v`
Expected: PASS (7 tests: shape, extended-raises, 3× ten-minute, dist_endline, box_ratio).

- [ ] **Step 5: Stage**

```bash
git add silly_kicks/tracking/_xcross_attempt.py tests/tracking/test_xcross_attempt.py
```

---

## Task 3: GK block + isolatability + cache-trap lock (L-4, the novel extension)

**Files:**
- Modify: `silly_kicks/tracking/_xcross_attempt.py` (fill the GK block in `extract_xcross_features`)
- Test: `tests/tracking/test_xcross_attempt.py`

- [ ] **Step 1: Write failing tests**

```python
def test_gk_block_filled_and_isolatable():
    feats = xc.extract_xcross_features(_one_frame(), gk_team_id="B", goal_x=105.0,
                                       carrier_player_id="A1")
    # GK block is a contiguous tail block, all non-NaN here (GK row present)
    assert xc.XCROSS_FEATURE_NAMES_FAITHFUL[-6:] == xc.XCROSS_GK_BLOCK
    assert feats[xc.XCROSS_GK_BLOCK].notna().all(axis=None)
    # dropping the block leaves the 10 non-GK features (3 ball + 7 confounders)
    base = [c for c in feats.columns if c not in xc.XCROSS_GK_BLOCK]
    assert len(base) == 10


def test_gk_block_nan_when_no_gk_row():
    frame = _one_frame()
    frame.loc[frame["player_id"] == "Bgk", "is_goalkeeper"] = False  # remove GK identity
    feats = xc.extract_xcross_features(frame, gk_team_id="B", goal_x=105.0, carrier_player_id="A1")
    assert feats[xc.XCROSS_GK_BLOCK].isna().all(axis=None)


def test_gk_r_goal_relative():
    feats = xc.extract_xcross_features(_one_frame(), gk_team_id="B", goal_x=105.0, carrier_player_id="A1")
    # GK at (104,34), attacked goal x=105 -> gr_x = 1.0, y-34=0 -> gk_r = 1.0
    assert feats["gk_r"].iloc[0] == pytest.approx(1.0, abs=1e-6)


def test_gk_post_distances_goal_relative():
    """C1: posts live at the attacked goal line gr_x=0 (NOT PITCH_LENGTH). GK at gr_x=1, y=34.
    Carrier A1 at y=10 (left of centre) -> near post on the left (post_y = 34 - 3.66 = 30.34),
    far post right (37.66). gk near = hypot(1, 34-30.34)=hypot(1,3.66)=3.80; far=hypot(1,3.66)=3.80
    (symmetric here since GK is central). Assert both are ~3.80, NOT ~104 (the PITCH_LENGTH bug)."""
    feats = xc.extract_xcross_features(_one_frame(), gk_team_id="B", goal_x=105.0, carrier_player_id="A1")
    assert feats["gk_dist_near_post"].iloc[0] == pytest.approx(math.hypot(1.0, 3.66), abs=1e-6)
    assert feats["gk_dist_far_post"].iloc[0] == pytest.approx(math.hypot(1.0, 3.66), abs=1e-6)
    assert feats["gk_dist_near_post"].iloc[0] < 10.0  # would be ~104 under the inverted convention


def test_faithful_never_imports_pitch_control():
    """L-4 (PA-M2): faithful extraction (incl. #3's pure-numpy Voronoi proxy) must NEVER import the
    pitch_control package — locks the TF-19 counterfactual guarantee. Run in a FRESH subprocess so the
    assertion can't false-green from another test having already imported pitch_control (the old
    hasattr-monkeypatch could vacuously pass if the spied symbol didn't exist)."""
    import subprocess, sys, os
    code = (
        "import sys; import pandas as pd, numpy as np\n"
        "from silly_kicks.tracking import _xcross_attempt as xc\n"
        "f = pd.DataFrame({'game_id':['g']*4,'period_id':[1]*4,'frame_id':[7]*4,"
        "'time_seconds':[40.0]*4,'team_id':['A','B','B','ball'],"
        "'player_id':['A1','B1','Bgk',None],'x':[95.,100.,104.,95.],'y':[10.,12.,34.,10.],"
        "'vx':[1.,0.,0.,1.],'vy':[0.]*4,'is_ball':[False,False,False,True],"
        "'is_goalkeeper':[False,False,True,False],'ball_state':['alive']*4})\n"
        "xc.extract_xcross_features(f, gk_team_id='B', goal_x=105.0, carrier_player_id='A1')\n"
        "assert 'silly_kicks.tracking.pitch_control' not in sys.modules, "
        "'faithful extraction imported pitch_control'\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=os.getcwd()))
    assert r.returncode == 0, r.stderr
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/tracking/test_xcross_attempt.py -k gk_block -v`
Expected: FAIL (GK columns are NaN / `gk_r` NaN).

- [ ] **Step 3: Implement the GK block** — insert before the `return` in `extract_xcross_features`

```python
    # GK block (NOVEL extension) — defending GK = is_goalkeeper row on gk_team_id.
    # C1: posts live at the ATTACKED goal line gr_x = 0 (NOT PITCH_LENGTH).
    gk_mask = is_gk & (team == gk_team_id)
    if gk_mask.any() and carrier_mask.any():
        gkx, gky = gr_x[gk_mask][0], y[gk_mask][0]
        out["gk_r"], out["gk_theta"] = _polar(gkx, gky - _geo.GOAL_Y)
        out["gk_lateral_offset"] = float(gky - _geo.GOAL_Y)
        # carrier flank sign; `or 1.0` only guards the exactly-central carrier (cy == GOAL_Y).
        side = np.sign(cy - _geo.GOAL_Y) or 1.0
        near_post_y = _geo.GOAL_Y + _GOAL_HALF_WIDTH_M * side   # post on the carrier's flank
        far_post_y = _geo.GOAL_Y - _GOAL_HALF_WIDTH_M * side
        out["gk_dist_near_post"] = math.hypot(gkx, gky - near_post_y)   # goal at gr_x=0
        out["gk_dist_far_post"] = math.hypot(gkx, gky - far_post_y)
        out["gk_carrier_side"] = float((gky - _geo.GOAL_Y) * side)
```

Note: `cx`/`cy` are defined in the carrier block (Task 2); the GK block reuses them, so it MUST sit after the carrier block (and is gated on `carrier_mask.any()`). Goal-relative posts live at `gr_x == 0` (the attacked goal line) — C1.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/tracking/test_xcross_attempt.py -v`
Expected: PASS (all Task 2 + Task 3 tests).

- [ ] **Step 5: Stage**

```bash
git add silly_kicks/tracking/_xcross_attempt.py tests/tracking/test_xcross_attempt.py
```

---

## Task 4: `build_xcross_labels` + label semantics

**Files:**
- Modify: `silly_kicks/tracking/_xcross_attempt.py`
- Test: `tests/tracking/test_xcross_attempt.py`

- [ ] **Step 1: Write failing tests**

```python
def _label_frames_and_actions():
    frames_index = pd.DataFrame({  # frames-side team column = team_in_possession (matches prepare)
        "game_id": ["g"] * 4, "period_id": [1] * 4, "frame_id": [1, 2, 3, 4],
        "time_seconds": [0.0, 0.4, 0.8, 1.2], "team_in_possession": ["A"] * 4,
    })
    from silly_kicks.spadl import config as spc
    actions = pd.DataFrame({  # A crosses at t=0.9; a pass (non-cross) at 0.2
        "game_id": ["g", "g"], "period_id": [1, 1], "team_id": ["A", "A"],
        "time_seconds": [0.9, 0.2],
        "type_id": [spc.actiontype_id["cross"], spc.actiontype_id["pass"]],
    })
    return frames_index, actions


def test_build_xcross_labels_open_play_only():
    fidx, actions = _label_frames_and_actions()
    y = xc.build_xcross_labels(fidx, actions, horizon_seconds=1.0)
    # cross at 0.9 in [t,t+1] for frames at 0.0,0.4,0.8 -> 1; 1.2 -> 0 (0.9 < 1.2)
    assert list(np.asarray(y)) == [1, 1, 1, 0]


def test_build_xcross_labels_set_pieces_togglable():
    from silly_kicks.spadl import config as spc
    fidx, actions = _label_frames_and_actions()
    actions.loc[1, "type_id"] = spc.actiontype_id["corner_crossed"]
    actions.loc[1, "time_seconds"] = 0.1
    y_open = xc.build_xcross_labels(fidx, actions, horizon_seconds=1.0)
    y_all = xc.build_xcross_labels(fidx, actions, horizon_seconds=1.0,
                                   cross_types=("cross", "corner_crossed"))
    assert list(np.asarray(y_open)) == [1, 1, 1, 0]   # corner ignored
    assert list(np.asarray(y_all))[0] == 1            # corner at 0.1 now counts
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/tracking/test_xcross_attempt.py -k labels -v`
Expected: FAIL (`build_xcross_labels` not defined).

- [ ] **Step 3: Implement** (append to `_xcross_attempt.py`)

```python
from silly_kicks.spadl import config as _spc
from silly_kicks.tracking._occurrence_labels import _build_occurrence_labels


def build_xcross_labels(frames_index, actions, *, horizon_seconds: float = 1.0,
                        cross_types: tuple[str, ...] = _DEFAULT_CROSS_TYPES,
                        frame_team_col: str = "team_in_possession"):
    type_ids = {_spc.actiontype_id[t] for t in cross_types}
    crosses = actions[actions["type_id"].isin(type_ids)][
        ["game_id", "period_id", "team_id", "time_seconds"]]
    y = _build_occurrence_labels(frames_index, crosses, horizon=horizon_seconds,
                                 frame_team_col=frame_team_col)
    return pd.Series(y, index=frames_index.index)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/tracking/test_xcross_attempt.py -k labels -v`
Expected: PASS.

- [ ] **Step 5: Stage** `git add silly_kicks/tracking/_xcross_attempt.py tests/tracking/test_xcross_attempt.py`

---

## Task 5: `prepare_xcross_training_data` (domain filter, possession, carrier-coverage log)

**Files:**
- Modify: `silly_kicks/tracking/_xcross_attempt.py`
- Test: `tests/tracking/test_xcross_attempt.py`

- [ ] **Step 1: Read** xS `prepare_xshot_training_data` (`_xshot_occurrence.py:598-708`) and its `_defended_goal_x` (571-585) + `_ball_in_attacking_third` (588-595) helpers — TF-17 mirrors the possession/goal-resolution loop, replacing the attacking-third filter with the wide-area filter and using cross labels.

- [ ] **Step 2: Write failing test** (uses a small multi-frame synthetic match)

```python
def _mini_match():
    rows = []
    for fr, t in enumerate([0.0, 0.4, 0.8, 1.2], start=1):
        rows += [
            dict(game_id="g", period_id=1, frame_id=fr, time_seconds=t, team_id="A",
                 player_id="A1", x=95.0, y=10.0, vx=1.0, vy=0.0, is_ball=False,
                 is_goalkeeper=False, ball_state="alive"),
            dict(game_id="g", period_id=1, frame_id=fr, time_seconds=t, team_id="B",
                 player_id="Bgk", x=104.0, y=34.0, vx=0.0, vy=0.0, is_ball=False,
                 is_goalkeeper=True, ball_state="alive"),
            dict(game_id="g", period_id=1, frame_id=fr, time_seconds=t, team_id="ball",
                 player_id=None, x=95.0, y=10.0, vx=1.0, vy=0.0, is_ball=True,
                 is_goalkeeper=False, ball_state="alive"),
        ]
    frames = pd.DataFrame(rows)
    from silly_kicks.spadl import config as spc
    actions = pd.DataFrame({"game_id": ["g"], "period_id": [1], "team_id": ["A"],
                            "time_seconds": [0.9], "type_id": [spc.actiontype_id["cross"]]})
    return frames, actions


def test_prepare_returns_features_labels_groups():
    frames, actions = _mini_match()
    X, y, groups = xc.prepare_xcross_training_data(frames, actions, home_team_id="A")
    assert list(X.columns) == xc.XCROSS_FEATURE_NAMES_FAITHFUL
    assert len(X) == len(y) == len(groups)
    assert set(np.unique(y)).issubset({0, 1})
    assert (groups == "g").all()
    assert y.sum() >= 1  # the cross at 0.9 labels some wide-area frames positive


def test_prepare_score_differential_wired_and_signed():
    """PA-H1: confounder #1 must be REALIZED (non-NaN) and signed from the possessing team's
    perspective. Team A (home, possessing) already scored 1; B scored 0 -> score_differential = +1."""
    from silly_kicks.spadl import config as spc
    frames, actions = _mini_match()
    goal = pd.DataFrame({  # A scores at t=-... before the window (a successful shot = a goal)
        "game_id": ["g"], "period_id": [1], "team_id": ["A"], "time_seconds": [0.0],
        "type_id": [spc.actiontype_id["shot"]], "result_id": [spc.result_id["success"]]})
    actions2 = pd.concat([actions.assign(result_id=spc.result_id["success"]), goal], ignore_index=True)
    X, _, _ = xc.prepare_xcross_training_data(frames, actions2, home_team_id="A")
    assert X["score_differential"].notna().all()           # realized for every row (actions supplied)
    # A leads by 1 from A's (possessing, home) perspective -> POSITIVE; max is +1 (boundary row may be 0
    # depending on the goal-at-t<=query convention, but the lead must show as a positive value).
    assert X["score_differential"].max() == pytest.approx(1.0, abs=1e-6)
    assert (X["score_differential"].dropna() >= 0.0).all()  # never negative for the leading possessor
```

- [ ] **Step 3: Run to verify failure** → FAIL (`prepare_xcross_training_data` not defined).

- [ ] **Step 4: Implement** — mirror xS's structure with the wide-area filter. Append:

```python
from silly_kicks.tracking._ball_carrier import derive_team_in_possession, infer_ball_carrier


def _build_goal_map(frames: pd.DataFrame) -> dict:
    """H1: precompute (game_id, period_id, team_id) -> defended goal_x ONCE (mirror xS
    _xshot_occurrence.py:571-585), instead of filtering the full frames DF per group."""
    goal_map: dict[tuple, float] = {}
    real = frames[frames["team_id"] != "ball"]  # PA-L3: skip the ball "team"
    gk = real[real["is_goalkeeper"]]
    for (gid, pid, tid), grp in real.groupby(["game_id", "period_id", "team_id"], sort=False):
        gk_grp = gk[(gk["game_id"] == gid) & (gk["period_id"] == pid) & (gk["team_id"] == tid)]
        mean_x = gk_grp["x"].mean() if len(gk_grp) else grp["x"].mean()
        goal_map[(gid, pid, tid)] = 0.0 if mean_x < _geo.PITCH_LENGTH / 2 else _geo.PITCH_LENGTH
    return goal_map


def _in_wide_area(ball_x, ball_y, goal_x, advance_m):
    if ball_x != ball_x or ball_y != ball_y:  # NaN ball position
        return False
    wide = (ball_y < _WIDE_Y_LOW) or (ball_y > _WIDE_Y_HIGH)
    advanced = abs(ball_x - goal_x) <= advance_m
    return wide and advanced


def prepare_xcross_training_data(
    frames, actions, *, home_team_id, feature_set: XCrossFeatureSet = "faithful",
    horizon_seconds: float = 1.0, wide_area_only: bool = True,
    advance_m: float = _ADVANCE_M, cross_types=_DEFAULT_CROSS_TYPES,
    carrier_params: dict | None = None,
):
    # PA-H1: confounder #1 score_differential — reuse ghost-GK's _build_score_lookup (local import
    # keeps `import _xcross_attempt` light + avoids pulling the ghost-GK model at module load).
    # NOTE at implementation: read _build_score_lookup's body to confirm the callback's first arg is
    # game_id and that it returns home_score - away_score (caller negates for the away possessing team).
    # (A future cleanup may promote _build_score_lookup to a shared `_match_context.py`, like
    # _occurrence_labels.py; deferred to avoid touching shipped ghost-GK in PR-A.)
    from silly_kicks.tracking._ghost_gk import _build_score_lookup

    cp = dict(carrier_params or _DEFAULT_CARRIER_PARAMS)
    carrier = infer_ball_carrier(frames, **cp)
    poss = derive_team_in_possession(frames, carrier)
    goal_map = _build_goal_map(frames)  # H1: precompute once
    score_fn = _build_score_lookup(actions, home_team_id) if actions is not None else None
    has_ball_state = "ball_state" in poss.columns  # M3: column-presence guard (mirror xS:655)

    feat_rows, frame_index_rows = [], []
    coverage = {"in_domain": 0, "carrier_resolved": 0}  # L-2 carrier-coverage log
    for (gid, pid, fid), grp in poss.groupby(["game_id", "period_id", "frame_id"], sort=False):
        if has_ball_state:
            ball_row = grp[grp["is_ball"]]
            # M3: judge ball_state on the ball row (row-level), not .all() over every player row
            if len(ball_row) and str(ball_row["ball_state"].iloc[0]) == "dead":
                continue
        in_poss = grp["team_in_possession"].dropna()
        if in_poss.empty:
            continue
        poss_team = in_poss.iloc[0]
        defending = [t for t in grp["team_id"].unique() if t not in (poss_team, "ball")]
        if not defending:
            continue
        goal_x = goal_map.get((gid, pid, defending[0]))
        if goal_x is None:
            continue
        ball = grp[grp["is_ball"]]
        bx = ball["x"].iloc[0] if len(ball) else np.nan
        by = ball["y"].iloc[0] if len(ball) else np.nan
        if wide_area_only and not _in_wide_area(bx, by, goal_x, advance_m):
            continue
        coverage["in_domain"] += 1
        carrier_pid = grp["ball_carrier_player_id"].dropna()
        carrier_pid = carrier_pid.iloc[0] if not carrier_pid.empty else None
        if carrier_pid is not None:
            coverage["carrier_resolved"] += 1
        sd = np.nan  # PA-H1: score differential from the POSSESSING team's perspective
        if score_fn is not None:
            raw = score_fn(gid, grp["time_seconds"].iloc[0])  # home_score - away_score
            sd = raw if str(poss_team) == str(home_team_id) else -raw
        feat_rows.append(extract_xcross_features(
            grp, gk_team_id=defending[0], goal_x=goal_x, carrier_player_id=carrier_pid,
            feature_set=feature_set, score_differential=sd))
        frame_index_rows.append(dict(
            game_id=gid, period_id=pid, frame_id=fid,
            time_seconds=grp["time_seconds"].iloc[0], team_in_possession=poss_team))

    if coverage["in_domain"]:
        rate = coverage["carrier_resolved"] / coverage["in_domain"]
        if rate < 0.8:
            warnings.warn(
                f"xCross carrier-resolution coverage {rate:.0%} over the wide-area domain "
                f"({coverage['carrier_resolved']}/{coverage['in_domain']}); GK/confounder "
                f"features may be degraded for this corpus.", UserWarning, stacklevel=2)

    if not feat_rows:
        empty = pd.DataFrame(columns=XCROSS_FEATURE_NAMES_FAITHFUL)
        return empty, np.array([], dtype=int), np.array([], dtype=object)

    X = pd.concat(feat_rows, ignore_index=True)
    frame_index = pd.DataFrame(frame_index_rows)  # carries team_in_possession (matches the label helper)
    y = np.asarray(build_xcross_labels(frame_index, actions, horizon_seconds=horizon_seconds,
                                       cross_types=cross_types))
    groups = frame_index["game_id"].to_numpy()
    return X, y, groups
```

(Add `import warnings` at the top of the module. H1: `compute_xcross_attempt` (Task 7) MUST reuse `_build_goal_map` the same way — do NOT reintroduce a per-frame goal filter on the serve hot path.)

- [ ] **Step 5: Run to verify pass** → PASS.
- [ ] **Step 6: Stage.**

---

## Task 6: `XCrossAttemptModel` (fit/predict/save/load/from_variant/from_hub)

**Files:**
- Modify: `silly_kicks/tracking/_xcross_attempt.py`
- Test: `tests/tracking/test_xcross_attempt.py`

- [ ] **Step 1: Read** xS `XShotOccurrenceModel` (`_xshot_occurrence.py:354-560`). TF-17 mirrors it verbatim with these substitutions: `feature_names = XCROSS_FEATURE_NAMES_FAITHFUL`; `_HF_REPO_ID`, `_MODEL_VERSION`, `_XCROSS_WEIGHTS_ROOT` from Task 0; metadata field `cross_types` (not `shot_types`); keep `carrier_params`, `horizon_seconds`, `shipped_variant`, `provider_list`, `pitch_length/width`, `geometry_version`, `xgboost_version`. The fail-closed pitch-dim guard + warn-only `geometry_version` + SHA256SUMS (CRLF→LF for `.json`) are identical.

- [ ] **Step 2: Write failing tests**

```python
def _fit_tiny_model(tmp_path):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 16)), columns=xc.XCROSS_FEATURE_NAMES_FAITHFUL)
    y = (X["gk_r"] + rng.normal(scale=0.5, size=200) > 0).astype(int).to_numpy()
    m = xc.XCrossAttemptModel().fit(X, pd.Series(y))
    return m, X


def test_model_fit_predict_proba(tmp_path):
    m, X = _fit_tiny_model(tmp_path)
    p = m.predict_proba(X)
    assert p.shape == (200,)
    assert ((p >= 0) & (p <= 1)).all()


def test_model_deterministic(tmp_path):
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(150, 16)), columns=xc.XCROSS_FEATURE_NAMES_FAITHFUL)
    y = pd.Series((rng.random(150) > 0.7).astype(int))
    p1 = xc.XCrossAttemptModel().fit(X, y).predict_proba(X)
    p2 = xc.XCrossAttemptModel().fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(p1, p2)


def test_model_save_load_roundtrip(tmp_path):
    m, X = _fit_tiny_model(tmp_path)
    d = tmp_path / "xcross_v1"
    m.save(d)
    assert (d / "model.json").exists() and (d / "metadata.json").exists() and (d / "SHA256SUMS").exists()
    m2 = xc.XCrossAttemptModel.load(d)
    np.testing.assert_allclose(m.predict_proba(X), m2.predict_proba(X), rtol=1e-9)


def test_model_sha256_verification(tmp_path):
    m, _ = _fit_tiny_model(tmp_path)
    d = tmp_path / "xcross_v1"; m.save(d)
    (d / "model.json").write_text((d / "model.json").read_text() + " ")  # tamper
    with pytest.raises(Exception):
        xc.XCrossAttemptModel.load(d)


def test_from_variant_filenotfound_until_weights():
    with pytest.raises(FileNotFoundError):
        xc.XCrossAttemptModel.from_variant("default")
    with pytest.raises(FileNotFoundError):
        xc.XCrossAttemptModel.from_hub()


def test_carrier_params_recorded_and_restored(tmp_path):
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(120, 16)), columns=xc.XCROSS_FEATURE_NAMES_FAITHFUL)
    y = pd.Series((rng.random(120) > 0.6).astype(int))
    cp = {"tolerance_m": 2.0, "beta": 0.1, "gamma": 0.5}
    m = xc.XCrossAttemptModel().fit(X, y, carrier_params=cp)
    d = tmp_path / "m"; m.save(d)
    assert xc.XCrossAttemptModel.load(d).carrier_params == cp
```

- [ ] **Step 3: Run to verify failure** → FAIL.

- [ ] **Step 4: Implement `XCrossAttemptModel`** mirroring xS. Skeleton (fill the save/load bodies exactly as xS does — same SHA256SUMS + pitch-dim guard):

```python
def _pinned_params(params: dict | None) -> dict:
    # M5: mirror xS _pinned_params EXACTLY (read _xshot_occurrence.py and copy the default dict
    # verbatim — incl. learning_rate, verbosity=0, and any other keys; do NOT invent values here).
    # The defaults below are a placeholder shape; replace with the literal xS dict at implementation.
    base = dict(tree_method="hist", n_jobs=1, subsample=1.0, colsample_bytree=1.0,
                random_state=42, eval_metric="logloss", verbosity=0,
                n_estimators=100, max_depth=4, learning_rate=0.3)  # <-- match xS exactly
    if params:
        base.update(params)
    for k in _INT_PARAMS:
        if k in base:
            base[k] = int(base[k])
    return base


class XCrossAttemptModel:
    """xCrossAttempt classifier: pinned-deterministic XGBoost over frame features.
    Pickle-free (booster JSON + metadata.json + SHA256SUMS). See NOTICE for citations."""

    def __init__(self, *, feature_set: XCrossFeatureSet = "faithful", params: dict | None = None):
        if feature_set != "faithful":
            raise NotImplementedError("xCrossAttempt: only feature_set='faithful' (PR-A).")
        self.feature_set = feature_set
        self._params = _pinned_params(params)
        self._booster = None
        self.carrier_params = dict(_DEFAULT_CARRIER_PARAMS)
        self.horizon_seconds = 1.0
        self.cross_types = list(_DEFAULT_CROSS_TYPES)
        self.shipped_variant: str | None = None
        self.provider_list: list | None = None

    def fit(self, features, labels, *, carrier_params=None, horizon_seconds=1.0):
        import xgboost as xgb
        # M1: mirror xS — require xgboost>=2 (calibrated base_score) and fit on numpy to dodge
        # xgboost's DataFrame feature-name validation mismatch (_xshot_occurrence.py:396,406).
        if int(xgb.__version__.split(".")[0]) < 2:
            raise RuntimeError("xCrossAttempt requires xgboost>=2.0 (calibrated base_score).")
        self.carrier_params = dict(carrier_params) if carrier_params else dict(_DEFAULT_CARRIER_PARAMS)
        self.horizon_seconds = horizon_seconds
        params = dict(self._params)
        params["base_score"] = float(np.asarray(labels, dtype=float).mean())  # calibrated intercept
        clf = xgb.XGBClassifier(**params)
        clf.fit(features.to_numpy(dtype=float), np.asarray(labels, dtype=int))  # M1: numpy
        self._booster = clf.get_booster()
        self._booster.feature_names = list(features.columns)
        return self

    def predict_proba(self, features):
        import xgboost as xgb
        if self._booster is None:
            raise RuntimeError("xCrossAttemptModel is unfit.")
        # M1: DMatrix from numpy (mirror _xshot_occurrence.py:423)
        dm = xgb.DMatrix(features.to_numpy(dtype=float), feature_names=list(features.columns))
        return np.asarray(self._booster.predict(dm), dtype=float)

    def save(self, path: Path) -> None: ...   # mirror _xshot_occurrence.XShotOccurrenceModel.save
    @classmethod
    def load(cls, path: Path) -> "XCrossAttemptModel": ...   # mirror xS.load (SHA + pitch-dim guard)
    @classmethod
    def from_variant(cls, variant: str = "default") -> "XCrossAttemptModel":
        if variant in _VARIANT_CACHE:
            return _VARIANT_CACHE[variant]
        d = _XCROSS_WEIGHTS_ROOT / variant
        if not d.exists():
            raise FileNotFoundError(
                f"xCrossAttempt weights '{variant}' not bundled yet; train via "
                f"scripts/train_xcross_attempt.py or await the PR-B weights follow-up.")
        m = cls.load(d); _VARIANT_CACHE[variant] = m; return m
    @classmethod
    def from_hub(cls, repo_id: str = _HF_REPO_ID) -> "XCrossAttemptModel":
        raise FileNotFoundError("No published xCrossAttempt weights yet (PR-B).")
```

For `save`/`load` bodies: copy `XShotOccurrenceModel.save`/`load` verbatim, renaming `shot_types`→`cross_types`, `feature_names`→`XCROSS_FEATURE_NAMES_FAITHFUL`, and the metadata `version`→`_MODEL_VERSION`. The pitch-dim fail-closed guard compares `metadata["pitch_length"]/["pitch_width"]` to `_geo.PITCH_LENGTH/_geo.PITCH_WIDTH` and raises the project's integrity error; `geometry_version` mismatch at equal dims → `warnings.warn(..., stacklevel=2)`.

- [ ] **Step 5: Run to verify pass** → PASS (7 tests). Requires `[xgboost]` extra installed in the venv.
- [ ] **Step 6: Stage.**

---

## Task 7: `compute_xcross_attempt` (two-pass inference + metadata carrier-params, R3)

**Files:**
- Modify: `silly_kicks/tracking/_xcross_attempt.py`
- Test: `tests/tracking/test_xcross_attempt.py`

- [ ] **Step 1: Read** xS `compute_xshot_occurrence` (`_xshot_occurrence.py:747-836`): carrier on full frames; per-frame extract restricted to `link_frame_ids`; one batched `predict_proba`; scatter-back via temporary `__gid`/`__tid` string join. TF-17 mirrors it; the per-frame extractor additionally needs `carrier_player_id` (from `poss["ball_carrier_player_id"]`) and resolves `goal_x` via **`_build_goal_map(frames)` computed ONCE** (H1 — do NOT filter the full frames DF per frame on this serve hot path). Carrier inference uses `model.carrier_params` (R3).

- [ ] **Step 2: Write failing tests**

```python
def test_compute_adds_column_and_uses_metadata_carrier_params(tmp_path, monkeypatch):
    frames, actions = _mini_match()
    X, y, _ = xc.prepare_xcross_training_data(frames, actions, home_team_id="A")
    model = xc.XCrossAttemptModel().fit(X, pd.Series(y), carrier_params={"tolerance_m": 2.5, "beta": 0.0, "gamma": 0.25})
    seen = {}
    import silly_kicks.tracking._xcross_attempt as mod
    real = mod.infer_ball_carrier
    monkeypatch.setattr(mod, "infer_ball_carrier",
                        lambda f, **k: seen.update(k) or real(f, **k))
    out = xc.compute_xcross_attempt(frames, model=model, home_team_id="A")
    assert "xcross_attempt" in out.columns
    vals = out["xcross_attempt"].dropna()
    assert ((vals >= 0) & (vals <= 1)).all()
    assert seen["tolerance_m"] == 2.5  # R3: carrier params read from model metadata, not library default


def test_compute_no_model_errors():
    frames, _ = _mini_match()
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        xc.compute_xcross_attempt(frames, model=None, home_team_id="A")


def test_compute_actions_populate_score_differential(monkeypatch):
    """PA-H1: at serve, passing actions= must realize score_differential (else it's NaN by design).
    Spy the extractor to confirm a non-NaN score reaches it when actions are supplied."""
    from silly_kicks.spadl import config as spc
    frames, actions = _mini_match()
    X, y, _ = xc.prepare_xcross_training_data(frames, actions, home_team_id="A")
    model = xc.XCrossAttemptModel().fit(X, pd.Series(y))
    goal = pd.DataFrame({"game_id": ["g"], "period_id": [1], "team_id": ["A"], "time_seconds": [0.0],
                         "type_id": [spc.actiontype_id["shot"]], "result_id": [spc.result_id["success"]]})
    actions2 = pd.concat([actions.assign(result_id=spc.result_id["success"]), goal], ignore_index=True)
    import silly_kicks.tracking._xcross_attempt as mod
    seen_scores = []
    real = mod.extract_xcross_features
    monkeypatch.setattr(mod, "extract_xcross_features",
                        lambda *a, **k: seen_scores.append(k.get("score_differential")) or real(*a, **k))
    xc.compute_xcross_attempt(frames, model=model, home_team_id="A", actions=actions2)
    assert any(s == s and s is not None for s in seen_scores)  # at least one non-NaN score reached extract
```

- [ ] **Step 3: Run to verify failure** → FAIL.

- [ ] **Step 4: Implement `compute_xcross_attempt`** mirroring xS, resolving the model (None → `from_variant("default")` which raises FileNotFoundError in PR-A; a passed `XCrossAttemptModel` is used directly), running carrier with `model.carrier_params` (R3), resolving `goal_x` via `_build_goal_map(frames)` once (H1), extracting per in-possession wide-area frame (or per `link_frame_ids`), one batched `predict_proba`, scattering `xcross_attempt` back on `(game_id, period_id, frame_id, team_id)`. **PA-H1: accept an optional `actions=` kwarg (mirror `compute_ghost_gk`); when supplied, build `score_fn = _build_score_lookup(actions, home_team_id)` and pass the possessing-team-signed `score_differential` per frame; when `None`, score_differential is NaN (XGBoost-tolerant) — documented in the docstring.** Signature:

```python
def compute_xcross_attempt(frames, *, model=None, home_team_id=None, actions=None,
                           pitch_control_cache=None, link_frame_ids: set | None = None):
    ...
```

`add_xcross_attempt` (Task 8) likewise gains an optional `actions=` passthrough so the score reaches `compute` at the action-coupled surface too (it already receives `actions`, so wire it through).

- [ ] **Step 5: Run to verify pass** → PASS.
- [ ] **Step 6: Stage.**

---

## Task 8: `add_xcross_attempt` (nan-safe, links, provenance-skip)

**Files:**
- Modify: `silly_kicks/tracking/_xcross_attempt.py`
- Test: `tests/tracking/test_xcross_attempt.py`

- [ ] **Step 1: Read** xS `add_xshot_occurrence` (`_xshot_occurrence.py:839-889`) + `@nan_safe_enrichment`.

- [ ] **Step 2: Write failing tests**

```python
def test_add_xcross_aggregator(tmp_path):
    frames, actions = _mini_match()
    X, y, _ = xc.prepare_xcross_training_data(frames, actions, home_team_id="A")
    model = xc.XCrossAttemptModel().fit(X, pd.Series(y))
    spadl_actions = pd.DataFrame({
        "game_id": ["g"], "period_id": [1], "team_id": ["A"], "time_seconds": [0.4],
        "type_id": [0], "action_id": [0]})
    out = xc.add_xcross_attempt(spadl_actions, frames, model=model, home_team_id="A")
    assert "xcross_attempt" in out.columns


def test_add_xcross_nan_safe(tmp_path):
    frames, actions = _mini_match()
    X, y, _ = xc.prepare_xcross_training_data(frames, actions, home_team_id="A")
    model = xc.XCrossAttemptModel().fit(X, pd.Series(y))
    spadl_actions = pd.DataFrame({
        "game_id": ["g"], "period_id": [1], "team_id": [np.nan], "time_seconds": [0.4],
        "type_id": [0], "action_id": [0]})
    out = xc.add_xcross_attempt(spadl_actions, frames, model=model, home_team_id="A")
    assert pd.isna(out["xcross_attempt"].iloc[0])  # NaN id -> NaN out, no crash


def test_add_xcross_has_nan_safe_marker():
    assert getattr(xc.add_xcross_attempt, "_nan_safe", False) is True
```

- [ ] **Step 3: Run to verify failure** → FAIL.
- [ ] **Step 4: Implement** mirroring xS (decorate `@nan_safe_enrichment`; `links` kwarg; provenance-skip guard; pre-populated-`xcross_attempt`-column fast path; join xS-value on `(game_id, period_id, frame_id, team_id)`).
- [ ] **Step 5: Run to verify pass** → PASS. (`tests/test_enrichment_nan_safety.py` will also auto-discover it — run it: `python -m pytest tests/test_enrichment_nan_safety.py -v`.)
- [ ] **Step 6: Stage.**

---

## Task 9: `xcross_attempt_xfns` (`_frame_aware`)

**Files:**
- Modify: `silly_kicks/tracking/_xcross_attempt.py`
- Test: `tests/tracking/test_xcross_attempt.py`

- [ ] **Step 1: Read** xS `xshot_occurrence_xfns` (`_xshot_occurrence.py:892-932`).
- [ ] **Step 2: Write failing tests**

```python
def test_xfns_factory_columns_and_marker(tmp_path):
    frames, actions = _mini_match()
    X, y, _ = xc.prepare_xcross_training_data(frames, actions, home_team_id="A")
    model = xc.XCrossAttemptModel().fit(X, pd.Series(y))
    fns = xc.xcross_attempt_xfns(model=model, home_team_id="A")
    assert len(fns) == 1
    assert getattr(fns[0], "_frame_aware", False) is True


def test_xfns_silent_nan_on_frames_none(tmp_path):
    """M4: ACTUALLY invoke the closure with frames=None and assert the 3-col NaN contract
    (not a hardcoded-list-equals-itself tautology). Mirror the exact call convention from xS's
    test_xshot_xfns_factory in tests/tracking/test_xshot_occurrence.py — the frame-aware closure
    is called as fn(gamestates, frames=None) (copy the gamestates construction from that test)."""
    frames, actions = _mini_match()
    X, y, _ = xc.prepare_xcross_training_data(frames, actions, home_team_id="A")
    model = xc.XCrossAttemptModel().fit(X, pd.Series(y))
    fn = xc.xcross_attempt_xfns(model=model, home_team_id="A")[0]
    states = _gamestates_for_xfn(actions)          # build per xS's test (3 gamestate slots)
    result = fn(states, frames=None)               # introspection: no frames
    assert list(result.columns) == ["xcross_attempt_a0", "xcross_attempt_a1", "xcross_attempt_a2"]
    assert result[["xcross_attempt_a0", "xcross_attempt_a1", "xcross_attempt_a2"]].isna().all(axis=None)
```

(Implement `_gamestates_for_xfn` by copying the gamestates-construction helper from xS's
`test_xshot_xfns_factory`; the closure's exact `(states, frames=...)` signature is whatever xS uses
— this test must call it the same way, not introspect a hardcoded attribute.)

- [ ] **Step 3: Run to verify failure** → FAIL.
- [ ] **Step 4: Implement** mirroring xS: a single closure emitting `xcross_attempt_a0/_a1/_a2`, collecting `link_frame_ids` across slots, one `compute_xcross_attempt` on the union, then per-slot `add_xcross_attempt(..., links=ptr)` reusing the scored frames; set `_frame_aware = True` and `__name__ = "xcross_attempt_xfn"`. **Do NOT add to any default/union xfn list (PR-A).**
- [ ] **Step 5: Run to verify pass** → PASS.
- [ ] **Step 6: Stage.**

---

## Task 10: HPO objective `_xcross_attempt_objective.py`

**Files:**
- Create: `silly_kicks/tracking/_xcross_attempt_objective.py`
- Test: `tests/tracking/test_xcross_attempt_integration.py`

- [ ] **Step 1: Read** `silly_kicks/tracking/_xshot_occurrence_objective.py` in full. TF-17 mirrors it exactly: `prepare()` concats pre-built `(X, y, groups)`; `evaluate_patch` fits XGB + `StratifiedGroupKFold` CV → mean log-loss (+ pr_auc, brier); `evaluate` recomputes monolithically; `patch_params = frozenset({"n_estimators","max_depth","learning_rate","min_child_weight","reg_lambda"})`; NO `scale_pos_weight`.
- [ ] **Step 2: Write failing test**

```python
def test_objective_cache_equivalence():
    import numpy as np, pandas as pd
    from ruthless_efficiency import Candidate, assert_cache_equivalence  # adjust import to pinned API
    from silly_kicks.tracking._xcross_attempt import XCROSS_FEATURE_NAMES_FAITHFUL
    from silly_kicks.tracking._xcross_attempt_objective import XCrossAttemptObjective
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 16)), columns=XCROSS_FEATURE_NAMES_FAITHFUL)
    y = (rng.random(300) > 0.6).astype(int)
    groups = np.array(["g1"] * 150 + ["g2"] * 150)
    obj = XCrossAttemptObjective(fold={"all": [(X, y, groups)]})
    cand = Candidate(id="c", params={"n_estimators": 30, "max_depth": 3, "learning_rate": 0.1,
                                     "min_child_weight": 1, "reg_lambda": 1.0})
    assert_cache_equivalence(obj, [cand])  # fast path == monolithic to 1e-9
```

(Use the exact `ruthless` import surface the xS objective test uses — copy from `tests/tracking/test_xshot_occurrence_integration.py`; per memory `result.best.candidate.params`, `Candidate(id=,params=)`.)

- [ ] **Step 3: Run to verify failure** → FAIL.
- [ ] **Step 4: Implement** `XCrossAttemptObjective` (mirror xS objective; `_cv_logloss` with `StratifiedGroupKFold`, groups coerced to `str`, train-fold-only `negative_subsample` default None).
- [ ] **Step 5: Run to verify pass** → PASS.
- [ ] **Step 6: Stage.**

---

## Task 11: Training CLI `scripts/train_xcross_attempt.py`

**Files:**
- Create: `scripts/train_xcross_attempt.py`
- Test: `tests/tracking/test_xcross_attempt_integration.py`

- [ ] **Step 1: Read** `scripts/train_xcross_attempt.py`'s sibling `scripts/train_xshot_occurrence.py` in full. Mirror: `--data-dir` / `--providers` mutually-exclusive sources; `_iter_matches_from_dir` reads `DIR/*/{frames,actions}.parquet`; `_extract` calls `prepare_xcross_training_data(...)`; feature cache to `<out>/xcross_attempt_v1/_feature_cache/`; HPO via `OptunaStrategy` + `XCrossAttemptObjective`; fail-closed acceptance gates. **PR-A scope:** the CLI exists + smoke-tests on synthetic parquet; the two-candidate public/full paired test + acceptance-gate thresholds are exercised in PR-B with real data (keep the code paths present but the smoke test only needs 3 trials + a written artifact).

- [ ] **Step 2: Write failing smoke test**

```python
def test_train_script_smoke(tmp_path):
    import subprocess, sys, os, json
    import numpy as np, pandas as pd
    # write a tiny synthetic match dir
    d = tmp_path / "data" / "m1"; d.mkdir(parents=True)
    # ... build frames.parquet + actions.parquet with >= 2 game_ids worth via two match dirs ...
    out = tmp_path / "out"
    env = dict(os.environ, PYTHONPATH=os.getcwd())
    r = subprocess.run([sys.executable, "scripts/train_xcross_attempt.py",
                        "--data-dir", str(tmp_path / "data"), "--output-dir", str(out),
                        "--n-trials", "3"], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    art = out / "xcross_attempt_v1"
    assert (art / "model.json").exists() and (art / "metadata.json").exists() and (art / "SHA256SUMS").exists()
```

(Fill the synthetic-data construction to produce ≥2 `game_id`s so `StratifiedGroupKFold` gets ≥2 groups; reuse `_mini_match()` shapes across two dirs `m1`/`m2`.)

- [ ] **Step 3: Run to verify failure** → FAIL.
- [ ] **Step 4: Implement the CLI** mirroring xS. Invoke as a subprocess with `PYTHONPATH=os.getcwd()` so the editable install imports (avoids the ghost-gk subprocess-import trap).
- [ ] **Step 5: Run to verify pass** → PASS.
- [ ] **Step 6: Stage.**

---

## Task 12: Public exports + atomic mirror

**Files:**
- Modify: `silly_kicks/tracking/__init__.py`
- Modify: `silly_kicks/atomic/tracking/features.py`
- Test: `tests/tracking/test_xcross_attempt_integration.py`

- [ ] **Step 1: Write failing tests**

```python
def test_public_exports():
    import silly_kicks.tracking as t
    for name in ["XCrossFeatureSet", "XCrossAttemptModel", "add_xcross_attempt",
                 "compute_xcross_attempt", "extract_xcross_features",
                 "prepare_xcross_training_data", "xcross_attempt_xfns", "subsample_negatives"]:
        assert hasattr(t, name), name


def test_atomic_mirror():
    from silly_kicks.atomic.tracking import features as af
    assert hasattr(af, "add_xcross_attempt") and hasattr(af, "xcross_attempt_xfns")
```

- [ ] **Step 2: Run to verify failure** → FAIL.
- [ ] **Step 3: Implement** — add to `silly_kicks/tracking/__init__.py` `__all__` + a `from ._xcross_attempt import (...)` block (8 names; `subsample_negatives` re-exported from `_xshot_occurrence` already — import xCross's via that module or re-export the shared one). In `silly_kicks/atomic/tracking/features.py` add `from silly_kicks.tracking._xcross_attempt import add_xcross_attempt, xcross_attempt_xfns` and add `add_xcross_attempt` to `__all__`. **Do NOT compose `xcross_attempt_xfns()` into any `*_default_xfns` bundle (PR-A).**
- [ ] **Step 4: Run to verify pass** → PASS.
- [ ] **Step 5: Verify dependency-light import** — `python -c "import silly_kicks"` must not import xgboost. Run: `python -c "import sys, silly_kicks; assert 'xgboost' not in sys.modules, sorted(m for m in sys.modules if 'xgb' in m)"`. Expected: no error.
- [ ] **Step 6: Stage.**

---

## Task 13: Real-provider extraction tests (regular suite, NOT e2e)

**Files:**
- Create: `tests/tracking/test_xcross_attempt_real_data.py`

- [ ] **Step 1: Read** `tests/tracking/test_action_context_cross_provider.py` for how the slim fixtures (`tests/datasets/tracking/action_context_slim/{sportec,metrica,skillcorner}_slim.parquet`) are loaded into frames+actions.

- [ ] **Step 2: Write the tests** (parametrized over available slim providers)

```python
import numpy as np, pandas as pd, pytest
from silly_kicks.tracking import _xcross_attempt as xc

PROVIDERS = ["sportec", "metrica", "skillcorner"]

@pytest.mark.parametrize("provider", PROVIDERS)
def test_extract_features_real_providers(provider):
    frames, actions, home = _load_slim(provider)  # helper mirrors the action_context loader
    X, y, groups = xc.prepare_xcross_training_data(frames, actions, home_team_id=home)
    if len(X):
        assert list(X.columns) == xc.XCROSS_FEATURE_NAMES_FAITHFUL
        # sane ranges
        assert (X["dist_endline"].dropna() >= -1e-6).all()
        assert X["ten_minute_warning"].dropna().isin([0, 1]).all()
        assert (X["space_controlled"].dropna() >= 0).all()


@pytest.mark.parametrize("provider", PROVIDERS)
def test_carrier_and_gk_blocks_mostly_resolved_on_string_ids(provider):
    """H3 (and the C3 disproof): on real STRING-id providers the carrier-anchored confounders
    and the GK block must be MOSTLY non-NaN — NOT vacuously all-NaN. This is the assertion that
    would have caught a carrier-id-typing bug; if `prepare` yields wide-area rows, ≥70% of them
    must have a resolved carrier (dist_nearest_def) and GK block (gk_r). A `.dropna()` range check
    alone passes vacuously when everything is NaN, so assert the NON-NaN FRACTION here."""
    frames, actions, home = _load_slim(provider)
    X, _, _ = xc.prepare_xcross_training_data(frames, actions, home_team_id=home)
    if len(X) >= 5:  # enough wide-area rows to be meaningful
        assert X["dist_nearest_def"].notna().mean() >= 0.7, "carrier-anchored features all/mostly NaN"
        assert X["gk_r"].notna().mean() >= 0.7, "GK block all/mostly NaN (C3 regression)"


@pytest.mark.parametrize("provider", PROVIDERS)
def test_real_provider_dtype_asymmetry(provider):
    frames, actions, home = _load_slim(provider)
    # must not crash on int64 (gradientsports-style) vs object (kloppy) ids — covered by the loader's native dtypes
    xc.prepare_xcross_training_data(frames, actions, home_team_id=home)
```

(Implement `_load_slim` by copying the loader helper from `test_action_context_cross_provider.py`. Only `sportec_slim` currently has cross rows — the others still exercise the carrier/GK resolution path. The `test_carrier_and_gk_blocks_mostly_resolved_on_string_ids` assertion is the one that converts "passes its own tests" into "works in the pipeline" — it directly refutes review C3's all-NaN concern on string-id providers. A frozen cross-feature fixture analogous to `tests/datasets/tracking/xshot_directional/` is added in PR-B.)

- [ ] **Step 3: Run** `python -m pytest tests/tracking/test_xcross_attempt_real_data.py -v`
Expected: PASS (no crash on every provider; sportec exercises the cross path).
- [ ] **Step 4: Stage.**

---

## Task 14: NOTICE + version bump + ADR-011 note

**Files:**
- Modify: `NOTICE`, `pyproject.toml`, `silly_kicks/__init__.py`, `TODO.md`, `CHANGELOG.md`, `docs/superpowers/adrs/ADR-011-trained-model-feature-lifecycle.md`

- [ ] **Step 1: NOTICE** — add under "Mathematical / Methodological References" (with the H2 faithfulness caveat):

```
The xCrossAttempt model (silly_kicks/tracking/_xcross_attempt.py, TF-17) is a per-frame,
STATE-anchored cross-attempt-propensity surface inspired by: Cao, Y., et al. (2025).
"Framing Causal Questions in Sports Analytics: A Case Study of Crossing in Soccer."
arXiv:2505.11841. The runtime surface is NOT a faithful reproduction of the paper's
sender-level treatment; it is extended with goalkeeper-position confounders (the paper's
confounder set excluded all GK variables), and the paper's confounder #7 (crosser position
FW/MF/DF, categorical event metadata) is omitted — no faithful tracking-only proxy exists.
The paper's sender-level propensity-score-matching framework is reproduced separately in the
TF-17 causal validation harness (PR-C).
```

- [ ] **Step 2: Version bump (4-file sync)** — coordinate the number with PR-S81 (M-2; PR-S81 takes 4.10.0 first, so TF-17 uses the next free minor). Update `pyproject.toml` `version`, `silly_kicks/__init__.py` `__version__`, `TODO.md` "Current release", `CHANGELOG.md` new dated section:

```
### Added
- xCrossAttempt (xCross) cross-attempt-propensity model (TF-17, GKDV Layer 2) — ships
  UNTRAINED (code + synthetic CI fixture + real-provider extraction tests). Paper-faithful
  7 of the paper's 8 confounders (crosser-position #7 omitted) + a novel goalkeeper-position confounder block; STATE-anchored per-frame
  surface for TF-19. Maintainer weights + surface-ablation + substitution-sensitivity probe
  follow in PR-B; the causal validation harness in PR-C. Note the xCross<->carrier-param
  coupling: a future TF-24 apply-PR carrier-default change is an xCross retrain trigger.
```

- [ ] **Step 3: `uv lock`** to sync `uv.lock` with the version bump. Run: `uv lock`.
- [ ] **Step 4: ADR-011 note** — append a short "Update — TF-17 (xCrossAttempt)" paragraph: 3rd feature on the lifecycle; same staged code→weights pattern; note the 3-PR split (code / weights+surface-validation / causal-harness) and that ADR-015 (the causal port) lands with PR-C.
- [ ] **Step 5: Stage.**

---

## Task 15: Full verification + lint trio + finalize

- [ ] **Step 1: Full non-e2e suite**

Run: `python -m pytest tests/ -m "not e2e" -v --tb=short`
Expected: all pass (new xCross tests + the xS golden + existing suite green). Capture the pass count.

- [ ] **Step 2: Lint trio (CI parity — run all three)**

```bash
ruff check silly_kicks/ tests/ scripts/
ruff format --check silly_kicks/ tests/ scripts/
pyright silly_kicks/
```
Expected: clean. (If `train_xcross_attempt.py` uses `X`/`Y`, add a per-file-ignore in `pyproject.toml` mirroring the xS script entry.)

- [ ] **Step 3: Dependency-light import guard**

Run: `python -c "import sys, silly_kicks; assert 'xgboost' not in sys.modules"`
Expected: no error.

- [ ] **Step 4: Squash to a single PR commit** (per the no-intermediate-commits rule). Present the full diff + the proposed commit message to the user and HOLD for explicit sentinel approval before committing. Proposed message:

```
feat(tracking): TF-17 xCrossAttempt propensity model (code, untrained) -- silly-kicks <version> (PR-?)

GKDV Layer 2 cross analogue of xShotOccurrence. 7 of the 8 Cao et al. confounders
(arXiv:2505.11841; crosser-position #7 omitted) + novel goalkeeper-position confounder block;
STATE-anchored per-frame surface for TF-19. Shared _build_occurrence_labels (xS
refactored onto it, bit-identical). Ships untrained; weights + validations in PR-B.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```

- [ ] **Step 5: Push + open PR** (after approval): `git push -u origin feat/tf17-xcross-attempt-code` then `gh pr create`.

---

## Self-Review (completed against the spec)

- **Spec coverage:** §2 module structure → Tasks 0–12; §3 API (compute/add/xfns/model/prepare) → Tasks 5–9; §4 label (time_seconds, no-bleed, cross-types, wide-area domain) → Tasks 1,4,5; §5 features (8 confounders + GK block, ball/carrier anchoring, NaN, carrier-coverage log, cache-trap test) → Tasks 2,3,5; §6 model & HPO (pinned XGB, no scale_pos_weight, base_score, StratifiedGroupKFold, R3 carrier metadata, negative_subsample=None) → Tasks 6,7,10; §8 serialization → Task 6; §10 PR-A shipping scope (untrained, from_variant/from_hub raise, xfns not wired) → Tasks 6,9,12; §11 tests → every task + 13; §13 attribution + §14 version/ADR → Task 14. **PR-B items** (weights, surface ablation, substitution-sensitivity probe, two-candidate paired test, xfn wiring) and **PR-C items** (`_causal/` port, matching unit tests, harness driver, ADR-015) are explicitly deferred — separate plans.
- **Placeholders:** the `save`/`load` bodies (Task 6 Step 4) and `compute`/`add`/`xfns`/objective/CLI bodies are specified as "mirror xS at <file:line> with these exact substitutions" rather than full reproduction — the engineer has the shipped xS module open as the literal template; tests pin the behavior. This is a deliberate brownfield-mirror choice, not a missing detail.
- **Type consistency:** `XCROSS_FEATURE_NAMES_FAITHFUL` (**16** = 3 ball + 7 confounders [#7 dropped] + 6 GK; score_differential #1 wired via `actions`), `XCROSS_GK_BLOCK` (6, contiguous tail), `extract_xcross_features(..., carrier_player_id=, goal_x=, gk_team_id=, score_differential=)`, `build_xcross_labels(..., cross_types=, frame_team_col=)`, `_build_occurrence_labels(..., frame_team_col=, event_team_col=)`, `prepare_xcross_training_data(frames, actions, *, home_team_id, ...) -> (X, y, groups)`, `compute_xcross_attempt(frames, *, model=, home_team_id=, actions=, ...)`, `XCrossAttemptModel.fit(..., carrier_params=, horizon_seconds=)` are used consistently across Tasks 1–13.
