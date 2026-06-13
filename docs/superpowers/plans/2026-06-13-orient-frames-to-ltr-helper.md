# `orient_frames_to_ltr` Helper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one public `silly_kicks.tracking.orient_frames_to_ltr` helper that orients *unlabeled* absolute tracking frames into the canonical home-attacks-right (LTR) frame, so consumers that build frames from non-kloppy sources (lakehouse metrica/skillcorner) single-source the orientation contract instead of re-implementing it.

**Architecture:** Pure composition of existing, tested primitives — `require_et_direction` + ADR-019 `ids_match` zero-match guard + `compute_attacking_direction` + `play_left_to_right` — with fail-loud preconditions. No new orientation math. Additive: no existing provider behaviour changes, no silly-kicks retrain.

**Tech Stack:** Python 3.10+, pandas, numpy. Lives in `silly_kicks/tracking/utils.py` next to `play_left_to_right`.

**Spec:** `docs/superpowers/specs/2026-06-13-orient-frames-to-ltr-helper-design.md`

**Commit policy (overrides skill default):** Per the maintainer's standing rules, this work lands as a **single bundled commit** (code + tests + ADR + spec + version bump + C4), created **only after explicit in-session approval**. Individual tasks below end with "verify green," NOT a commit. The one commit happens in the final task.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `silly_kicks/tracking/utils.py` | Add `orient_frames_to_ltr` + `_ORIENT_REQUIRED_COLUMNS` next to `play_left_to_right` | Modify |
| `silly_kicks/tracking/__init__.py` | Export `orient_frames_to_ltr` (only — `compute_attacking_direction` stays private) | Modify |
| `tests/tracking/test_orient_frames_to_ltr.py` | Unit tests (orientation, guards, mirror-invariance, adapter-equivalence, double-call lock) | Create |
| `tests/tracking/test_orient_frames_to_ltr_integration.py` | Integration test (defending GK at attacked goal, + un-oriented control) | Create |
| `tests/tracking/test_adapter_extra_time_orientation.py` | Positive ET orientation regression guard for gradientsports/sportec `convert_to_frames` (+ wrong-flag control) | Create |
| `docs/superpowers/adrs/ADR-029-orient-frames-to-ltr.md` | Decision record | Create |
| `docs/c4/architecture.dsl` | Add ADR-029 to the tracking container consumer-contracts clause | Modify |
| `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, `TODO.md`, `uv.lock` | Version bump to 4.27.0 | Modify |

---

### Task 0: Feature branch

**Files:** none (git only)

- [ ] **Step 1: Create the feature branch in the main checkout (no worktree)**

Run:
```bash
git switch -c feat/orient-frames-to-ltr
```
Expected: `Switched to a new branch 'feat/orient-frames-to-ltr'`

---

### Task 1: Unit tests for `orient_frames_to_ltr` (red-first)

**Files:**
- Test: `tests/tracking/test_orient_frames_to_ltr.py` (create)

- [ ] **Step 1: Write the failing unit-test file**

Create `tests/tracking/test_orient_frames_to_ltr.py`:

```python
"""Unit tests for tracking.orient_frames_to_ltr.

The helper takes UNLABELED absolute frames (team_attacking_direction all-null) and
returns canonical home-attacks-right (LTR) frames: home attacks high x in every
period, away attacks low x. It is the unlabeled-input sibling of play_left_to_right
(which serves already-labeled frames). Decision: ADR-029.

Frame-convention reminder: "home attacks LTR / high x" => home's own goal at x=0,
so the HOME GK sits near x=0 and the AWAY GK near x=105, in EVERY period.
"""

import numpy as np
import pandas as pd
import pytest

# Import from the defining module so Task 1 is self-contained (the package-level
# export is added in Task 2, validated by test_exported_from_tracking_namespace).
from silly_kicks.tracking.utils import orient_frames_to_ltr


def _make_frame(
    period_id: int,
    frame_id: int,
    *,
    home_team_id=100,
    away_team_id=200,
    home_gk_x: float = 5.0,
    away_gk_x: float = 100.0,
    ball_x: float = 52.5,
    ball_y: float = 34.0,
    direction=None,  # UNLABELED by default (absolute frames)
) -> pd.DataFrame:
    """One frame: home GK, away GK, ball. Absolute (unlabeled) by default."""
    base = {
        "game_id": 1,
        "period_id": period_id,
        "frame_id": frame_id,
        "time_seconds": frame_id / 25.0,
        "frame_rate": 25.0,
        "z": float("nan"),
        "speed": 0.0,
        "speed_source": "native",
        "ball_state": "alive",
        "confidence": None,
        "visibility": None,
        "source_provider": "metrica",
    }
    rows = [
        {**base, "player_id": "HOME-GK", "team_id": home_team_id, "is_ball": False,
         "is_goalkeeper": True, "x": home_gk_x, "y": 34.0, "team_attacking_direction": direction},
        {**base, "player_id": "AWAY-GK", "team_id": away_team_id, "is_ball": False,
         "is_goalkeeper": True, "x": away_gk_x, "y": 34.0, "team_attacking_direction": direction},
        {**base, "player_id": None, "team_id": None, "is_ball": True,
         "is_goalkeeper": False, "x": ball_x, "y": ball_y, "team_attacking_direction": None},
    ]
    return pd.DataFrame(rows)


def _two_period_absolute():
    """Home physically attacks right in P1 (home GK at x=5), left in P2 (home GK at x=100).

    home_team_start_left = True (home's own goal on the left in P1).
    """
    p1 = _make_frame(1, 0, home_gk_x=5.0, away_gk_x=100.0)
    p2 = _make_frame(2, 100, home_gk_x=100.0, away_gk_x=5.0)
    return pd.concat([p1, p2], ignore_index=True)


# --- per-period orientation (spec test #2) ---

def test_home_gk_low_x_both_periods():
    out = orient_frames_to_ltr(_two_period_absolute(), home_team_id=100, home_team_start_left=True)
    for pid in (1, 2):
        home_gk = out[(out["period_id"] == pid) & (out["player_id"] == "HOME-GK")].iloc[0]
        away_gk = out[(out["period_id"] == pid) & (out["player_id"] == "AWAY-GK")].iloc[0]
        assert abs(home_gk["x"] - 5.0) < 0.01, f"P{pid} home GK x={home_gk['x']}"
        assert abs(away_gk["x"] - 100.0) < 0.01, f"P{pid} away GK x={away_gk['x']}"


def test_direction_labels_after_orient():
    out = orient_frames_to_ltr(_two_period_absolute(), home_team_id=100, home_team_start_left=True)
    home = out[out["player_id"] == "HOME-GK"]
    away = out[out["player_id"] == "AWAY-GK"]
    assert set(home["team_attacking_direction"].unique()) == {"ltr"}
    assert set(away["team_attacking_direction"].unique()) == {"rtl"}


def test_ball_player_distance_preserved_per_period():
    frames = _two_period_absolute()
    out = orient_frames_to_ltr(frames, home_team_id=100, home_team_start_left=True)
    for pid in (1, 2):
        for pl in ("HOME-GK", "AWAY-GK"):
            raw = frames[(frames["period_id"] == pid)]
            o = out[(out["period_id"] == pid)]
            rp = raw[raw["player_id"] == pl].iloc[0]
            rb = raw[raw["is_ball"]].iloc[0]
            op = o[o["player_id"] == pl].iloc[0]
            ob = o[o["is_ball"]].iloc[0]
            raw_d = np.hypot(rp["x"] - rb["x"], rp["y"] - rb["y"])
            out_d = np.hypot(op["x"] - ob["x"], op["y"] - ob["y"])
            assert abs(raw_d - out_d) < 0.01


def test_extra_time_positive_orientation():  # review concern A
    """P3/P4 orient correctly when home_team_start_left_extratime IS supplied.

    With home_team_start_left=True, home_team_start_left_extratime=False, the
    per-period home-attacks-right flags are P1=True, P2=False, P3=False, P4=True
    (home_attacks_right_per_period). So in ABSOLUTE frames the home GK (own goal
    behind the attack) sits at x=5 in P1/P4 and x=100 in P2/P3. After orient, the
    home GK must land at x≈5 (and away GK x≈100) in ALL FOUR periods. This also
    de-risks the shared ET chain that the sportec/GS adapters use.
    """
    home_gk_abs = {1: 5.0, 2: 100.0, 3: 100.0, 4: 5.0}
    away_gk_abs = {1: 100.0, 2: 5.0, 3: 5.0, 4: 100.0}
    parts = [
        _make_frame(p, p * 100, home_gk_x=home_gk_abs[p], away_gk_x=away_gk_abs[p])
        for p in (1, 2, 3, 4)
    ]
    frames = pd.concat(parts, ignore_index=True)
    out = orient_frames_to_ltr(
        frames, home_team_id=100, home_team_start_left=True, home_team_start_left_extratime=False
    )
    for pid in (1, 2, 3, 4):
        home_gk = out[(out["period_id"] == pid) & (out["player_id"] == "HOME-GK")].iloc[0]
        away_gk = out[(out["period_id"] == pid) & (out["player_id"] == "AWAY-GK")].iloc[0]
        assert abs(home_gk["x"] - 5.0) < 0.01, f"P{pid} home GK x={home_gk['x']}"
        assert abs(away_gk["x"] - 100.0) < 0.01, f"P{pid} away GK x={away_gk['x']}"


# --- guards ---

def test_zero_match_raises():  # spec test #4 (C1)
    frames = _make_frame(1, 0)
    with pytest.raises(ValueError, match="ZERO player rows"):
        orient_frames_to_ltr(frames, home_team_id=999, home_team_start_left=True)


def test_missing_column_raises():  # spec test #4b (C5)
    frames = _make_frame(1, 0).drop(columns=["team_id"])
    with pytest.raises(ValueError, match="missing required columns"):
        orient_frames_to_ltr(frames, home_team_id=100, home_team_start_left=True)


def test_et_without_flag_raises():  # spec test #3
    frames = _make_frame(3, 0)
    with pytest.raises(ValueError, match="home_team_start_left_extratime"):
        orient_frames_to_ltr(frames, home_team_id=100, home_team_start_left=True)


def test_already_labeled_raises():  # spec test #6 (C2)
    frames = _make_frame(1, 0, direction="ltr")  # labeled on entry
    with pytest.raises(ValueError, match="play_left_to_right"):
        orient_frames_to_ltr(frames, home_team_id=100, home_team_start_left=True)


def test_double_call_raises():  # spec test #6 (C2): second call is labeled
    out = orient_frames_to_ltr(_two_period_absolute(), home_team_id=100, home_team_start_left=True)
    with pytest.raises(ValueError, match="play_left_to_right"):
        orient_frames_to_ltr(out, home_team_id=100, home_team_start_left=True)


def test_empty_frame_returns_copy():
    empty = _make_frame(1, 0).iloc[0:0]
    out = orient_frames_to_ltr(empty, home_team_id=100, home_team_start_left=True)
    assert len(out) == 0


# --- mirror-invariance (spec test #1, C6) ---

def test_mirror_invariance():
    """orient(F, flag) == orient(mirror(F), not flag)."""
    f = _two_period_absolute()
    mirror = f.copy()
    is_player_or_ball = mirror["x"].notna()
    mirror.loc[is_player_or_ball, "x"] = 105.0 - mirror.loc[is_player_or_ball, "x"]
    mirror.loc[is_player_or_ball, "y"] = 68.0 - mirror.loc[is_player_or_ball, "y"]

    a = orient_frames_to_ltr(f, home_team_id=100, home_team_start_left=True)
    b = orient_frames_to_ltr(mirror, home_team_id=100, home_team_start_left=False)

    a_sorted = a.sort_values(["period_id", "frame_id", "player_id"]).reset_index(drop=True)
    b_sorted = b.sort_values(["period_id", "frame_id", "player_id"]).reset_index(drop=True)
    pd.testing.assert_series_equal(a_sorted["x"], b_sorted["x"], check_names=False)
    pd.testing.assert_series_equal(a_sorted["y"], b_sorted["y"], check_names=False)
    pd.testing.assert_series_equal(
        a_sorted["team_attacking_direction"], b_sorted["team_attacking_direction"], check_names=False
    )


# --- equivalence to the native sportec adapter (spec test #5) ---

def test_equivalence_to_sportec_adapter():
    """sportec.convert_to_frames(ltr) == abs-frames(from same raw) + orient_frames_to_ltr."""
    from silly_kicks.tracking import sportec

    base = {
        "game_id": 1, "frame_rate": 25.0, "z": float("nan"),
        "speed_native": float("nan"), "ball_state": "alive",
    }
    # Raw sportec input (x_centered, y_centered), 2 periods. Home attacks right in P1,
    # left in P2 (physical), encoded in centered coords: P1 home GK at left (x_centered=-47.5),
    # P2 home GK at right (x_centered=+47.5). home_team_start_left=True.
    def raw_row(period, frame, pid, tid, isball, isgk, xc, yc):
        return {**base, "period_id": period, "frame_id": frame, "time_seconds": frame / 25.0,
                "player_id": pid, "team_id": tid, "is_ball": isball, "is_goalkeeper": isgk,
                "x_centered": xc, "y_centered": yc}
    raw = pd.DataFrame([
        raw_row(1, 0, "HOME-GK", "H", False, True, -47.5, 0.0),
        raw_row(1, 0, "AWAY-GK", "A", False, True, 47.5, 0.0),
        raw_row(1, 0, None, None, True, False, 0.0, 0.0),
        raw_row(2, 100, "HOME-GK", "H", False, True, 47.5, 0.0),
        raw_row(2, 100, "AWAY-GK", "A", False, True, -47.5, 0.0),
        raw_row(2, 100, None, None, True, False, 0.0, 0.0),
    ])
    adapter_out, _ = sportec.convert_to_frames(
        raw, home_team_id="H", home_team_start_left=True, output_convention="ltr",
    )

    # Build the equivalent UNLABELED absolute frames (x = x_centered + 52.5, y = y_centered + 34),
    # matching the sportec adapter's pre-flip absolute coordinates.
    abs_frames = raw.copy()
    abs_frames["x"] = abs_frames["x_centered"] + 52.5
    abs_frames["y"] = abs_frames["y_centered"] + 34.0
    abs_frames["speed"] = abs_frames["speed_native"]
    abs_frames["speed_source"] = None
    abs_frames["team_attacking_direction"] = None
    abs_frames["confidence"] = None
    abs_frames["visibility"] = None
    abs_frames["source_provider"] = "sportec"
    abs_frames = abs_frames.drop(columns=["x_centered", "y_centered", "speed_native"])

    helper_out = orient_frames_to_ltr(abs_frames, home_team_id="H", home_team_start_left=True)

    a = adapter_out.sort_values(["period_id", "frame_id", "is_ball", "player_id"]).reset_index(drop=True)
    h = helper_out.sort_values(["period_id", "frame_id", "is_ball", "player_id"]).reset_index(drop=True)
    pd.testing.assert_series_equal(a["x"], h["x"], check_names=False)
    pd.testing.assert_series_equal(a["y"], h["y"], check_names=False)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/tracking/test_orient_frames_to_ltr.py -v`
Expected: FAIL — `ImportError: cannot import name 'orient_frames_to_ltr'` (the symbol does not exist yet).

- [ ] **Step 3: Implement `orient_frames_to_ltr`**

In `silly_kicks/tracking/utils.py`, immediately AFTER the `play_left_to_right` function (ends ~line 181), add:

```python
_ORIENT_REQUIRED_COLUMNS = ("x", "y", "team_id", "period_id", "is_ball", "team_attacking_direction")


def orient_frames_to_ltr(
    frames: pd.DataFrame,
    *,
    home_team_id,
    home_team_start_left: bool,
    home_team_start_left_extratime: bool | None = None,
) -> pd.DataFrame:
    """Orient *unlabeled* absolute-orientation tracking frames into the canonical
    home-attacks-right (LTR) frame, per period.

    This is the unlabeled-input sibling of :func:`play_left_to_right`. It populates
    ``team_attacking_direction`` from ``home_team_start_left`` (the physical pre-flip
    direction) and then per-period flips so the home team attacks x=105 in every
    period and the away team attacks x=0 -- byte-identical to
    ``convert_to_frames(output_convention="ltr")`` and exactly the convention the
    per-action geometry layer (ADR-028) expects.

    Intended for consumers that build frames from a non-kloppy source (e.g. the
    lakehouse metrica/skillcorner bronze builders) in absolute orientation. For frames
    that ALREADY carry a populated ``team_attacking_direction`` (labeled, e.g.
    ``kloppy.convert_to_frames(output_convention="absolute_frame")`` output), use
    :func:`play_left_to_right` directly -- this helper raises on labeled input.

    Parameters
    ----------
    frames : pd.DataFrame
        Unlabeled absolute tracking frames. Required columns: ``x``, ``y``,
        ``team_id``, ``period_id``, ``is_ball``, ``team_attacking_direction`` (which
        must be all-null on entry). ``team_id`` may be any dtype -- comparisons route
        through the ADR-019 dtype-safe ``ids_match``.
    home_team_id : int | str
        Identifies the home team in ``team_id``. The caller derives this; silly-kicks
        does not infer it.
    home_team_start_left : bool
        True iff the home team's own goal is on the left (x=0) in period 1, i.e. it
        attacks toward x=105 in period 1. Source of truth for the orientation; the
        helper is only as correct as this flag (validate it per game -- see ADR-029).
    home_team_start_left_extratime : bool | None, default None
        Required only when ET periods (3/4) are present.

    Returns
    -------
    pd.DataFrame
        A new DataFrame in home-attacks-right convention. Not idempotent -- a second
        call raises (the first populated ``team_attacking_direction``).

    Raises
    ------
    ValueError
        Missing required columns; ``team_attacking_direction`` non-null on entry (use
        ``play_left_to_right``); ``home_team_id`` matches zero player rows; ET periods
        present without ``home_team_start_left_extratime``.

    See ADR-029 for the single-source-of-truth orientation contract.
    """
    out = frames.copy()
    if len(out) == 0:
        return out

    missing = [c for c in _ORIENT_REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"orient_frames_to_ltr: frames missing required columns: {missing}")

    is_ball = out["is_ball"].astype(bool)
    players = out[~is_ball]
    if players.empty:
        return out

    # C2: labeled-input guard. Unlabeled absolute frames carry an all-null direction;
    # any non-null means the frames are already labeled -> route to play_left_to_right.
    if out["team_attacking_direction"].notna().any():
        raise ValueError(
            "orient_frames_to_ltr: frames already carry a populated "
            "team_attacking_direction (labeled). This helper is for UNLABELED absolute "
            "frames; use silly_kicks.tracking.play_left_to_right for labeled frames."
        )

    # C1: zero-match guard (ADR-019 dtype-safe compare). Zero home-player match means
    # play_left_to_right cannot identify flip periods -> definitely-wrong output.
    is_home = ids_match(players["team_id"], home_team_id).fillna(False)
    if not bool(is_home.any()):
        raise ValueError(
            f"orient_frames_to_ltr: home_team_id={home_team_id!r} matched ZERO player rows "
            "(id dtype mismatch vs frame team_id?) -- orientation would be wrong."
        )

    from .direction import compute_attacking_direction, require_et_direction

    require_et_direction(out["period_id"], home_team_start_left_extratime, source="orient_frames_to_ltr")

    out["team_attacking_direction"] = compute_attacking_direction(
        team_id=out["team_id"],
        period_id=out["period_id"],
        is_ball=out["is_ball"],
        home_team_id=home_team_id,
        home_team_start_left=home_team_start_left,
        home_team_start_left_extratime=home_team_start_left_extratime,
    )
    return play_left_to_right(out, home_team_id)
```

Note: `ids_match` is already imported at the top of `utils.py` (used by `play_left_to_right`); `compute_attacking_direction` / `require_et_direction` are imported function-locally to match the lazy-import idiom and avoid any import-order coupling.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/tracking/test_orient_frames_to_ltr.py -v`
Expected: PASS (all tests).

---

### Task 2: Public export

**Files:**
- Modify: `silly_kicks/tracking/__init__.py` (`__all__` ~line 173; `from .utils import (...)` ~line 390)

- [ ] **Step 1: Write the failing export test**

Append to `tests/tracking/test_orient_frames_to_ltr.py`:

```python
def test_exported_from_tracking_namespace():
    import silly_kicks.tracking as t
    assert "orient_frames_to_ltr" in t.__all__
    assert callable(t.orient_frames_to_ltr)
    # compute_attacking_direction stays private (C4)
    assert "compute_attacking_direction" not in t.__all__
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/tracking/test_orient_frames_to_ltr.py::test_exported_from_tracking_namespace -v`
Expected: FAIL — `assert "orient_frames_to_ltr" in t.__all__` fails (the symbol is defined in `utils.py` but not yet re-exported from the package `__init__`).

- [ ] **Step 3: Add the export**

In `silly_kicks/tracking/__init__.py`:

a) In the `__all__` list, add `"orient_frames_to_ltr"` in alphabetical position (between `"obso_xfns"`/`"off_ball_..."` block and `"pausa_xfns"` — i.e. right before `"pausa_xfns"` at ~line 168, keeping alpha order: `orient_frames_to_ltr` sorts after `off_ball_xt_team` and before `pausa_xfns`):

```python
    "off_ball_xt_team",
    "orient_frames_to_ltr",
    "pausa_xfns",
```

b) In the `from .utils import (...)` block (~line 390), add the import in alpha order:

```python
from .utils import (
    add_sync_score,
    filter_extratime_frames,
    link_actions_to_frames,
    orient_frames_to_ltr,
    play_left_to_right,
    slice_around_event,
    sync_score,
    validate_id_dtypes,
    validate_time_base,
)
```

- [ ] **Step 4: Run the export test (and the full file) to verify pass**

Run: `python -m pytest tests/tracking/test_orient_frames_to_ltr.py -v`
Expected: PASS (all tests including the export test).

---

### Task 3: Integration test (defending GK at the attacked goal)

**Files:**
- Test: `tests/tracking/test_orient_frames_to_ltr_integration.py` (create)

This proves the fix end-to-end: chaining `orient_frames_to_ltr` → `add_pre_shot_gk_position` (which applies the ADR-028 per-action reprojection) makes the defending GK land near the attacked goal for BOTH home and away shots; the un-oriented control reproduces the bimodality.

Pure synthetic — no committed dataset fixtures — so it is **NOT** marked `@pytest.mark.e2e` (per CLAUDE.md, e2e = needs uncommitted dataset fixtures) and runs in the regular suite (review concern D). It also deliberately **shortcuts GK resolution**: `_actions()` hardcodes `defending_gk_player_id`, so the chain is orient → `add_pre_shot_gk_position`, NOT orient → `add_pre_shot_gk_context` resolution → position. The bug under test is the position reprojection; GK-resolution-under-orientation is out of scope here (review concern B).

- [ ] **Step 1: Write the integration test**

Create `tests/tracking/test_orient_frames_to_ltr_integration.py`:

```python
"""Integration: orient_frames_to_ltr closes the metrica/skillcorner GK bimodality.

Builds absolute frames + a home-team shot (P1) and an away-team shot (P2). After
orient_frames_to_ltr + the ADR-028 reprojection in add_pre_shot_gk_position, the
defending GK lands near the attacked goal (x>=95) for BOTH shots. The control on the
un-oriented absolute frames reproduces the bimodality (one GK ~near goal, one ~far).

Scope note (review concern B): defending_gk_player_id is hardcoded on the actions, so
this covers position reprojection under orientation, NOT add_pre_shot_gk_context GK
resolution. Pure synthetic -> intentionally NOT marked @pytest.mark.e2e (concern D).
"""

import numpy as np
import pandas as pd

from silly_kicks import spadlconfig
from silly_kicks.tracking import orient_frames_to_ltr
from silly_kicks.tracking.features import add_pre_shot_gk_position

SHOT = spadlconfig.actiontype_id["shot"]


def _frame_row(period, frame, pid, tid, isball, isgk, x, y):
    return {
        "game_id": 1, "period_id": period, "frame_id": frame, "time_seconds": frame / 25.0,
        "frame_rate": 25.0, "player_id": pid, "team_id": tid, "is_ball": isball,
        "is_goalkeeper": isgk, "x": x, "y": y, "z": float("nan"), "speed": 0.0,
        "speed_source": "native", "ball_state": "alive", "team_attacking_direction": None,
        "confidence": None, "visibility": None, "source_provider": "metrica",
    }


def _abs_frames():
    """Absolute frames. P1 home attacks right (away GK defends x=105 -> x=100).
    P2 home attacks left (home GK defends x=105 -> x=100). home_team_start_left=True."""
    rows = []
    # P1, frame 0: home shooter near x=90, away GK (defender) at x=100, home GK at x=5
    rows += [
        _frame_row(1, 0, "HOME-ATT", 100, False, False, 90.0, 34.0),
        _frame_row(1, 0, "HOME-GK", 100, False, True, 5.0, 34.0),
        _frame_row(1, 0, "AWAY-GK", 200, False, True, 100.0, 34.0),
        _frame_row(1, 0, None, None, True, False, 90.0, 34.0),
    ]
    # P2, frame 100: away shooter near x=15 (attacks left), home GK (defender) at x=5,
    # away GK at x=100  (absolute: away attacks toward x=0 in P2)
    rows += [
        _frame_row(2, 100, "AWAY-ATT", 200, False, False, 15.0, 34.0),
        _frame_row(2, 100, "HOME-GK", 100, False, True, 5.0, 34.0),
        _frame_row(2, 100, "AWAY-GK", 200, False, True, 100.0, 34.0),
        _frame_row(2, 100, None, None, True, False, 15.0, 34.0),
    ]
    return pd.DataFrame(rows)


def _actions():
    """A home shot in P1 and an away shot in P2, in per-acting-team LTR action coords
    (start_x ~ 90 for both; the acting team attacks x=105). defending_gk_player_id is
    the opposing team's GK."""
    return pd.DataFrame([
        {"action_id": 1, "game_id": 1, "period_id": 1, "time_seconds": 0.0,
         "team_id": 100, "player_id": "HOME-ATT", "type_id": SHOT,
         "start_x": 90.0, "start_y": 34.0, "end_x": 105.0, "end_y": 34.0,
         "defending_gk_player_id": "AWAY-GK"},
        {"action_id": 2, "game_id": 1, "period_id": 2, "time_seconds": 100 / 25.0,
         "team_id": 200, "player_id": "AWAY-ATT", "type_id": SHOT,
         "start_x": 90.0, "start_y": 34.0, "end_x": 105.0, "end_y": 34.0,
         "defending_gk_player_id": "HOME-GK"},
    ])


def test_oriented_gk_clusters_at_attacked_goal():
    oriented = orient_frames_to_ltr(_abs_frames(), home_team_id=100, home_team_start_left=True)
    enriched = add_pre_shot_gk_position(_actions(), oriented)
    gk_x = enriched.set_index("action_id")["pre_shot_gk_x"]
    assert gk_x[1] >= 95.0, f"home shot defending GK x={gk_x[1]}"
    assert gk_x[2] >= 95.0, f"away shot defending GK x={gk_x[2]}"


def test_unoriented_control_is_bimodal():
    """Sanity: WITHOUT orient, the away-team shot's GK is at the wrong end."""
    enriched = add_pre_shot_gk_position(_actions(), _abs_frames())
    gk_x = enriched.set_index("action_id")["pre_shot_gk_x"]
    # One near the attacked goal, one far -> bimodal (the bug).
    assert max(gk_x[1], gk_x[2]) >= 95.0
    assert min(gk_x[1], gk_x[2]) <= 20.0
```

- [ ] **Step 2: Run the integration test**

Run: `python -m pytest tests/tracking/test_orient_frames_to_ltr_integration.py -v`
Expected: PASS. If `test_oriented_gk_clusters_at_attacked_goal` fails because the linker/GK-resolution needs an extra action column, inspect the failure and add the missing column to `_actions()` — the test must fail for a *fixture* reason, never because orientation is wrong. (The control test passing while the oriented test fails would indicate a real orientation bug to debug, not a fixture gap.)

---

### Task 3b: Positive extra-time orientation regression guard for the native adapters

**Files:**
- Test: `tests/tracking/test_adapter_extra_time_orientation.py` (create)

Locks the POSITIVE P3/P4 orientation of `gradientsports.convert_to_frames` and
`sportec.convert_to_frames` (the shared `home_attacks_right_per_period` → per-period
flip chain). Born from the live GS-ET flip (2026-06-13): that was a *consumer* bug (a
wrong `home_team_start_left_extratime` placeholder passed in), and the adapters orient
ET correctly GIVEN a correct flag — but ET was only negatively tested (the
`require_et_direction` guard). This test would pass today and prevents a silent ET
regression. It includes a **wrong-flag control** that reproduces the live reversal, so
the guard demonstrably discriminates.

- [ ] **Step 1: Write the adapter ET test**

Create `tests/tracking/test_adapter_extra_time_orientation.py`:

```python
"""Positive extra-time (P3/P4) orientation regression guard for the native adapters.

gradientsports/sportec.convert_to_frames take RAW physical centered coords (x_centered)
+ the start-left flags and flip per period internally. With home_team_start_left=True,
home_team_start_left_extratime=False the per-period home-attacks-right flags are
{1:T, 2:F, 3:F, 4:T} (home_attacks_right_per_period); in physical coords the home GK
(own goal behind the attack) is at x_centered=-47.5 in P1/P4 and +47.5 in P2/P3.
Correct orientation -> home GK lands at x=5 in ALL four periods, away GK at x=100.

Context: the live GS-ET flip (2026-06-13) was a consumer bug (wrong
home_team_start_left_extratime placeholder), NOT a silly_kicks bug. This locks the
positive ET path; the wrong-flag control reproduces the live reversal so the guard
discriminates. sportec team_id/player_id are object strings; gradientsports are
nullable Int64 -> id values are parametrized per provider.
"""

import pandas as pd
import pytest

from silly_kicks.tracking import gradientsports, sportec


def _raw_row(period, frame, pid, tid, isball, isgk, xc, yc):
    return {
        "game_id": 1, "period_id": period, "frame_id": frame, "time_seconds": frame / 25.0,
        "frame_rate": 25.0, "player_id": pid, "team_id": tid, "is_ball": isball,
        "is_goalkeeper": isgk, "x_centered": xc, "y_centered": yc, "z": float("nan"),
        "speed_native": float("nan"), "ball_state": "alive",
    }


def _raw_4period(home_id, away_id, home_gk, away_gk):
    """Physical raw frames: home attacks right in P1/P4, left in P2/P3."""
    home_gk_xc = {1: -47.5, 2: 47.5, 3: 47.5, 4: -47.5}
    away_gk_xc = {1: 47.5, 2: -47.5, 3: -47.5, 4: 47.5}
    rows = []
    for p in (1, 2, 3, 4):
        f = p * 100
        rows += [
            _raw_row(p, f, home_gk, home_id, False, True, home_gk_xc[p], 0.0),
            _raw_row(p, f, away_gk, away_id, False, True, away_gk_xc[p], 0.0),
            _raw_row(p, f, None, None, True, False, 0.0, 0.0),
        ]
    return pd.DataFrame(rows)


_ADAPTERS = [
    pytest.param(sportec, "H", "A", "HOME-GK", "AWAY-GK", id="sportec"),
    pytest.param(gradientsports, 57, 99, 1, 2, id="gradientsports"),
]


@pytest.mark.parametrize("adapter, home_id, away_id, home_gk, away_gk", _ADAPTERS)
def test_positive_extra_time_orientation(adapter, home_id, away_id, home_gk, away_gk):
    """Correct ET flag -> home GK at x=5, away GK at x=100, in ALL FOUR periods."""
    raw = _raw_4period(home_id, away_id, home_gk, away_gk)
    out, _ = adapter.convert_to_frames(
        raw, home_team_id=home_id, home_team_start_left=True,
        home_team_start_left_extratime=False, output_convention="ltr",
    )
    for p in (1, 2, 3, 4):
        hg = out[(out["period_id"] == p) & (out["player_id"] == home_gk)].iloc[0]
        ag = out[(out["period_id"] == p) & (out["player_id"] == away_gk)].iloc[0]
        assert abs(hg["x"] - 5.0) < 0.01, f"{adapter.__name__} P{p} home GK x={hg['x']}"
        assert abs(ag["x"] - 100.0) < 0.01, f"{adapter.__name__} P{p} away GK x={ag['x']}"


@pytest.mark.parametrize("adapter, home_id, away_id, home_gk, away_gk", _ADAPTERS)
def test_wrong_extra_time_flag_reverses_p3_p4(adapter, home_id, away_id, home_gk, away_gk):
    """Control: a WRONG ET flag reverses P3/P4 (the live GS pattern) -> guard discriminates."""
    raw = _raw_4period(home_id, away_id, home_gk, away_gk)
    out, _ = adapter.convert_to_frames(
        raw, home_team_id=home_id, home_team_start_left=True,
        home_team_start_left_extratime=True,  # WRONG for this physical setup
        output_convention="ltr",
    )
    for p in (1, 2):  # regulation still correct
        hg = out[(out["period_id"] == p) & (out["player_id"] == home_gk)].iloc[0]
        assert abs(hg["x"] - 5.0) < 0.01, f"{adapter.__name__} P{p} home GK x={hg['x']}"
    for p in (3, 4):  # ET reversed by the wrong flag (the live bug signature)
        hg = out[(out["period_id"] == p) & (out["player_id"] == home_gk)].iloc[0]
        assert abs(hg["x"] - 100.0) < 0.01, f"{adapter.__name__} P{p} home GK should reverse, x={hg['x']}"
```

- [ ] **Step 2: Run the adapter ET test**

Run: `python -m pytest tests/tracking/test_adapter_extra_time_orientation.py -v`
Expected: PASS (all 4 cases — both providers × positive + control). It exercises the *current* adapter behaviour, so it passes immediately; its value is regression-locking.

---

### Task 4: ADR-029

**Files:**
- Create: `docs/superpowers/adrs/ADR-029-orient-frames-to-ltr.md`

- [ ] **Step 1: Write the ADR**

Create `docs/superpowers/adrs/ADR-029-orient-frames-to-ltr.md`:

```markdown
# ADR-029: Frame-LTR orientation is single-sourced via `orient_frames_to_ltr`

| Field | Value |
|---|---|
| **Date** | 2026-06-13 |
| **Status** | Accepted |
| **Deciders** | Karsten Nielsen, Claude (luxury-lakehouse report `silly_kicks_metrica_skillcorner_ltr_frame_20260613`) |

## Context

ADR-028 (4.26.0) re-projects per-action tracking geometry into the per-acting-team
LTR frame, but only for frames already in the canonical home-attacks-right
convention. silly-kicks owns that convention for providers it converts via
`convert_to_frames(output_convention="ltr")` (sportec, gradientsports) and the kloppy
gateway (metrica, skillcorner). But a consumer holding **bronze DataFrames** (not a
kloppy `TrackingDataset`) cannot use the gateway and must build frames itself; the
lakehouse does exactly this for metrica/skillcorner, in **absolute** orientation
(`team_attacking_direction = None`, no per-period flip). ADR-028's reprojection
filters to `team_attacking_direction.notna()` rows, so on all-null frames it no-ops
and ~50% of action rows carry mirror-wrong geometry. Empirically (post-4.26.0 local
recompute): idsse/GS `pre_shot_gk_x` ~101 (clean); metrica 60.6, skillcorner 53.5
(bimodal).

The orientation logic already existed as two shared primitives
(`compute_attacking_direction` + `play_left_to_right`) but only `play_left_to_right`
(labeled-input) was public; nothing served the unlabeled-absolute case, so the
consumer re-implemented orientation incompletely.

## Decision

Add one public `silly_kicks.tracking.orient_frames_to_ltr(frames, *, home_team_id,
home_team_start_left, home_team_start_left_extratime=None)` that composes the existing
primitives (no new orientation math) with fail-loud preconditions:

- required-schema guard (raises on missing columns);
- already-labeled guard (raises if `team_attacking_direction` non-null -> use
  `play_left_to_right`), which also makes the helper non-idempotent-but-guarded;
- zero-match guard (raises if `home_team_id` matches no player row -- ADR-019);
- ET guard (raises on ET periods without the ET flag).

Two public entry points, by input state: **labeled** absolute frames ->
`play_left_to_right`; **unlabeled** absolute frames -> `orient_frames_to_ltr`. The
lower-level `compute_attacking_direction` stays private.

**Consumer contract:** any consumer building tracking frames from a non-kloppy source
MUST orient them (via `orient_frames_to_ltr` for unlabeled frames) into the
home-attacks-right convention before the per-action geometry layer (ADR-028). The
helper is only as correct as the caller-derived `home_team_start_left`; consumers MUST
validate that flag per game (e.g. assert each game's defending GK lands near the
attacked goal post-orient).

## Consequences

- Additive: no existing provider behaviour changes; no silly-kicks model retrain. The
  sportec/GS adapters and kloppy gateway are NOT refactored through the helper
  (primitives already shared; refactor risks goldens/retrain for zero gain).
- The lakehouse adopts the helper in its metrica/skillcorner bronze builders and
  re-materializes those providers (its consequence, not a bundled-model retrain).
- Decided against a native `metrica`/`skillcorner.convert_to_frames` (option a): it
  would duplicate the kloppy gateway and still cannot consume bronze DataFrames; TF-23
  already retired the metrica native loader.
```

- [ ] **Step 2: Verify the ADR renders (no broken tables)**

Run: `python -c "p='docs/superpowers/adrs/ADR-029-orient-frames-to-ltr.md'; s=open(p,encoding='utf-8').read(); assert '| **Status** | Accepted |' in s; print('ok')"`
Expected: `ok`

---

### Task 5: Version bump + CHANGELOG + TODO + C4

**Files:**
- Modify: `pyproject.toml:7`, `silly_kicks/__init__.py:7`, `CHANGELOG.md`, `TODO.md` (header), `docs/c4/architecture.dsl`, `uv.lock`

- [ ] **Step 1: Bump `pyproject.toml`**

Change line 7 from `version = "4.26.0"` to:
```toml
version = "4.27.0"
```

- [ ] **Step 2: Bump `silly_kicks/__init__.py`**

Change line 7 from `__version__ = "4.26.0"` to:
```python
__version__ = "4.27.0"
```

- [ ] **Step 3: Add CHANGELOG entry**

In `CHANGELOG.md`, insert directly above `## [4.26.0] — 2026-06-12`:

```markdown
## [4.27.0] — 2026-06-13

### Added
- `silly_kicks.tracking.orient_frames_to_ltr(frames, *, home_team_id, home_team_start_left, home_team_start_left_extratime=None)` — orients *unlabeled* absolute tracking frames into the canonical home-attacks-right (LTR) frame, single-sourcing the orientation contract for consumers that build frames from a non-kloppy source (bronze DataFrames). Pure composition of existing primitives (`compute_attacking_direction` + `play_left_to_right`) with fail-loud guards (missing-schema, already-labeled → use `play_left_to_right`, zero home-match, ET-without-flag). Companion to ADR-028: ADR-028's per-action reprojection no-ops on absolute frames (`team_attacking_direction = None`), so consumers must orient first. Decision: ADR-029.

### Notes
- **Additive — no model retrain.** Existing providers (sportec/gradientsports/kloppy) are byte-unchanged; the helper is new and not called internally. **Consumer impact:** adopting `orient_frames_to_ltr` in the lakehouse metrica/skillcorner bronze builders fixes their previously-bimodal tracking action geometry (`pre_shot_gk_x`, `defensive_line_x`, `nearest_defender_distance`, `pressure_on_actor__*`, etc.); those providers must be re-materialized lakehouse-side. The helper is only as correct as the caller-derived `home_team_start_left` — validate it per game.
```

- [ ] **Step 4: Update TODO.md header**

In `TODO.md`, update the `**Last updated**` / `**Current release**` line (line 5) to lead with 4.27.0:

```markdown
**Last updated**: 2026-06-13. **Current release**: silly-kicks 4.27.0 (`orient_frames_to_ltr` — single-sources the frame-LTR orientation contract for consumers building frames from non-kloppy sources; companion to ADR-028; additive, no retrain; ADR-029). Prior: 4.26.0 (tracking geometry emitted in the per-action SPADL LTR frame; ADR-028). Per-version history lives in [CHANGELOG.md](CHANGELOG.md).
```

- [ ] **Step 5: Add ADR-029 to the C4 consumer-contracts clause**

In `docs/c4/architecture.dsl`, find the `tracking` container description's consumer-contracts enumeration (lists ADR-017/019/020/028) and append `ADR-029`. Run to locate it:
```bash
grep -n "ADR-028\|ADR-020\|consumer-contract" docs/c4/architecture.dsl
```
Edit the matched description string to include `ADR-029` in the same comma-separated list (e.g. `... ADR-020, ADR-028, ADR-029`). No aggregator/backend/model count changes (count stays 27).

- [ ] **Step 6: Re-lock `uv.lock`**

Run: `uv lock`
Expected: `uv.lock` updates the silly-kicks version to 4.27.0 (no dependency changes).

- [ ] **Step 7: Regenerate the C4 HTML (REQUIRED — do not skip)**

Regenerate `docs/c4/architecture.html` from the edited `.dsl` via the `mad-scientist-skills:c4` skill (Java 21 + structurizr.war + plantuml.jar in `~/.claude/tools/`; Context + Container only). This is **required**, not conditional (review concern C): the `.dsl` carries ADR-029 in the tracking container description, and a skipped regen leaves `.dsl`/`.html` diverged. If the toolchain genuinely fails, that is a **blocker to surface to the maintainer**, not a step to skip — the committed `.html` must reflect the committed `.dsl`.

Verify the regen actually picked up ADR-029:
```bash
grep -c "ADR-029" docs/c4/architecture.html
```
Expected: `>= 1`.

---

### Task 6: Full local quality gate (shift-left)

**Files:** none (verification only)

- [ ] **Step 1: Lint**

Run: `ruff check silly_kicks/ tests/ scripts/`
Expected: `All checks passed!`

- [ ] **Step 2: Format check**

Run: `ruff format --check silly_kicks/ tests/ scripts/`
Expected: no files would be reformatted (if any, run `ruff format` on them and re-verify).

- [ ] **Step 3: Type check (full tree — local-green ≠ CI-green, so check the whole tree)**

Run: `pyright silly_kicks/ tests/ scripts/`
Expected: 0 errors. (`home_team_id` is intentionally untyped — matches `play_left_to_right`'s signature; no `# type: ignore` needed.)

- [ ] **Step 4: Targeted tests**

Run: `python -m pytest tests/tracking/test_orient_frames_to_ltr.py tests/tracking/test_orient_frames_to_ltr_integration.py tests/tracking/test_adapter_extra_time_orientation.py -v`
Expected: all PASS.

- [ ] **Step 5: Regression — the existing orientation + id-lint guards still pass**

Run: `python -m pytest tests/tracking/test_play_left_to_right_ball_flip.py tests/tracking/test_id_compat_lint.py tests/tracking/test_action_ltr_mirror_invariance.py -v`
Expected: all PASS. (`test_id_compat_lint.py` enforces the boundary lint — the helper uses `ids_match`, so it must not trip the raw-`==` AST lint.)

- [ ] **Step 6: Broad suite (no e2e)**

Run: `python -m pytest tests/ -m "not e2e" -q`
Expected: all PASS, no new failures vs baseline.

---

### Task 7: Bundle and commit (AWAIT EXPLICIT APPROVAL)

**Files:** all of the above.

- [ ] **Step 1: Present the diff and request approval**

Run: `git status && git --no-pager diff --stat`
Then present the full command + diff summary to the maintainer and **wait for explicit in-session approval** (the commit sentinel / chat approval). Do NOT run `git commit` before approval.

- [ ] **Step 2: Commit (only after approval)**

Single bundled commit:
```bash
git add silly_kicks/tracking/utils.py silly_kicks/tracking/__init__.py \
  tests/tracking/test_orient_frames_to_ltr.py tests/tracking/test_orient_frames_to_ltr_integration.py \
  tests/tracking/test_adapter_extra_time_orientation.py \
  docs/superpowers/adrs/ADR-029-orient-frames-to-ltr.md \
  docs/superpowers/specs/2026-06-13-orient-frames-to-ltr-helper-design.md \
  docs/superpowers/plans/2026-06-13-orient-frames-to-ltr-helper.md \
  docs/c4/architecture.dsl docs/c4/architecture.html \
  pyproject.toml silly_kicks/__init__.py CHANGELOG.md TODO.md uv.lock
git commit -F <commit-message-file>
```
Use a commit-message file (not `-m`) with subject:
`feat(tracking): orient_frames_to_ltr single-sources the frame-LTR contract -- silly-kicks 4.27.0 (ADR-029)`
and a body summarizing the helper, the metrica/skillcorner consumer fix, and the additive/no-retrain note. End with the `Co-Authored-By` trailer.

- [ ] **Step 3: Push / PR / tag / publish (separate explicit approvals)**

Per the maintainer's release mechanics, push the branch, open the PR, squash-merge with `--admin`, then tag `v4.27.0` (→ publish.yml → PyPI). Each of push, merge, and tag is a separate chat-approved step run as a bare command.

---

## Notes for the implementer

- **No worktree** (maintainer rule): work on `feat/orient-frames-to-ltr` in the main checkout.
- **`ids_match` import:** already present at the top of `utils.py` (used by `play_left_to_right`). Do not re-import.
- **Why function-local imports of `direction`:** matches the established lazy-import idiom (`sportec.py` / `kloppy.py` both do `from .utils import play_left_to_right` locally); avoids any import-order surprises even though `direction.py` has no cycle with `utils.py`.
- **The mirror-invariance test** (`test_mirror_invariance`) is the durable cross-provider guard; it must assert the flag flips WITH the mirror (`orient(F, flag) == orient(mirror(F), not flag)`).
