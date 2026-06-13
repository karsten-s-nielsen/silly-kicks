# Tracking geometry action-LTR re-projection — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every emitted per-action tracking-geometry **position** column be expressed in the same SPADL **action-LTR** frame as the action it annotates (acting team attacks x=105), fixing a systemic orientation bug that corrupts ~50% of tracking-provider action rows (those where the acting team is the away team).

**Architecture:** Frames are home-attacks-right (`convert_to_frames` default; home attacks x=105 every period). Actions are per-acting-team-LTR (`to_spadl_ltr`; the *acting* team attacks x=105). The two agree for home-team actions and are a 180° point-reflection (`x→105−x, y→68−y`) apart for away-team actions. Today the geometry layer samples frame positions but never re-projects, so for away-team actions: (a) absolute-position outputs land at the wrong end (visible bimodality), and (b) features that *mix* an action-LTR anchor with frame-coord positions produce numerically wrong distances/counts (silent). Fix: one canonical re-projection (`_action_orientation.py`) applied at exactly three seams — the shared `ActionFrameContext` (fixes 8 context kernels at once), the `_defensive_line_at_actions` coupling, and the `add_team_shape`/`_team_shape_at_actions` coupling — plus an emit-time transform for ghost-GK (goal-relative → action-LTR). Guarded by a mirror-symmetry property test.

**Tech Stack:** Python 3.10 (`.venv`), pandas 2.3 / numpy 2.2, pytest. No new runtime deps.

**Version / ADR:** silly-kicks **4.26.0** (breaking value change, `fix(tracking)!`), **ADR-028**. VAEP/tracking-retrain trigger; lakehouse re-materializes all tracking action-context.

---

## Background facts (verified in-source during investigation)

- `to_spadl_ltr` / `_mirror_absolute_frame` mirror away rows by **both** `field_length−x` **and** `field_width−y` (180° point reflection). `silly_kicks/spadl/orientation.py:201-211`.
- `convert_to_frames` default emits home-attacks-right; `team_attacking_direction ∈ {"ltr","rtl"}` is populated by every adapter (home="ltr", away="rtl", None for ball). `gradientsports.py:139-153`, `sportec.py:151-164`, `kloppy.py:153/179`, `_snapshot.py:92/118` (snapshot sets BOTH teams "ltr" — synthetic; see Task 1 note).
- Shared context builder `_resolve_action_frame_context` (utils.py:631-742) feeds exactly the buggy kernel family: `_nearest_defender_distance`, `_receiver_zone_density`, `_defenders_in_triangle_to_goal`, `_pre_shot_gk_position`, `_pre_shot_gk_angle`, `_pressure_andrienko`, `_pressure_link`, `_pressure_bekkers` (all in `_kernels.py`). `_actor_speed_from_ctx` uses `speed` (magnitude) only → flip-invariant.
- After the frame rows are re-projected into action-LTR, the kernels' hardcoded `_GOAL_X=105`/`_GOAL_Y_CENTER=34` become **correct** (the acting team genuinely attacks x=105; defended goal at 105).
- Self-reconciling features (`structural_pass`, `gk_influence`, `player_influence`, `cover_shadows`, `shape_graph`, `obso`, `space_creation`, `das`, `pitch_control`, `pausa`, `xt_gk`) already handle orientation via their own `home_team_id`/`goal_x` flips OR are direction-invariant. They do **not** use `ActionFrameContext`. **Do not touch them.** (Double-flip risk.)
- `ghost_gk_x/y` are goal-relative (defended goal at x=0) by design; the model trains/serves goal-relative and stays that way. Only the emit transforms.

## Per-column re-projection classification (the authoritative table)

For each affected emitted column, the re-projection when the action is **away** (acting team attacks RTL in the frame):

| Column(s) | Kind | Transform when away | Seam |
|---|---|---|---|
| context frame rows: `x` | x-position | `105−x` | Task 2 (`ActionFrameContext`) |
| context frame rows: `y` | y-position | `68−y` | Task 2 |
| `defensive_line_x`, `back_line_high_x` | x-position | `105−x` | Task 3 |
| `compactness_x`, `lateral_width`, `max_lateral_gap`, `back_n_count` | invariant | — | Task 3 |
| `team_shape_centroid_x_{attacking,defending}` | x-position | `105−x` | Task 4 |
| `team_shape_centroid_y_{attacking,defending}` | y-position | `68−y` | Task 4 |
| `team_shape_defensive_line_height_{attacking,defending}` | x-position | `105−x` | Task 4 |
| `team_shape_{convex_hull_area,team_length,team_width,stretch_index,inter_line_gap_1,inter_line_gap_2,n_outfield_players}_*` | invariant | — | Task 4 |
| `ghost_gk_x` (and `ghost_gk_mode_x`, `ghost_gk_mean_x` if present) | goal-rel x → action-LTR x | `105 − gr_x` **(uniform — NOT per-action; see Task 5)** | Task 5 |
| `ghost_gk_y` (and `*_mode_y`, `*_mean_y`) | goal-rel y → action-LTR y | `68 − gr_y` when away, `gr_y` when home | Task 5 |
| `ghost_gk_density_spread` | invariant | — | Task 5 |

`pre_shot_gk_x/y/distance_to_goal/distance_to_shot` and `pre_shot_gk_angle_*` need **no** column-kind handling — they are fixed transitively by Task 2 (the frame rows they read become action-LTR; the hardcoded goal 105 becomes correct).

---

## Task 1: Canonical re-projection module

**Files:**
- Create: `silly_kicks/tracking/_action_orientation.py`
- Test: `tests/tracking/test_action_orientation.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/tracking/test_action_orientation.py
"""Unit tests for the canonical action-LTR re-projection helper (ADR-028)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.tracking._action_orientation import (
    FIELD_LENGTH,
    FIELD_WIDTH,
    acting_team_attacks_rtl,
    reproject_to_action_ltr,
)


def _frames(home_dir="ltr", away_dir="rtl"):
    # home team=1 (ltr), away team=2 (rtl), one period, one frame
    return pd.DataFrame(
        [
            dict(game_id=1, period_id=1, frame_id=10, team_id=1, is_ball=False, team_attacking_direction=home_dir),
            dict(game_id=1, period_id=1, frame_id=10, team_id=2, is_ball=False, team_attacking_direction=away_dir),
            dict(game_id=1, period_id=1, frame_id=10, team_id=np.nan, is_ball=True, team_attacking_direction=None),
        ]
    )


def test_acting_team_attacks_rtl_home_false_away_true():
    actions = pd.DataFrame(
        [
            dict(game_id=1, period_id=1, action_id=0, team_id=1),  # home → not rtl
            dict(game_id=1, period_id=1, action_id=1, team_id=2),  # away → rtl
        ]
    )
    flip = acting_team_attacks_rtl(actions, _frames())
    assert flip.tolist() == [False, True]


def test_acting_team_unknown_direction_defaults_false():
    actions = pd.DataFrame([dict(game_id=1, period_id=1, action_id=0, team_id=999)])
    flip = acting_team_attacks_rtl(actions, _frames())
    assert flip.tolist() == [False]


def test_reproject_flips_only_marked_rows_both_axes():
    df = pd.DataFrame({"x": [10.0, 10.0], "y": [20.0, 20.0]})
    flip = pd.Series([False, True])
    out = reproject_to_action_ltr(df, flip, x_cols=["x"], y_cols=["y"])
    assert out["x"].tolist() == [10.0, FIELD_LENGTH - 10.0]
    assert out["y"].tolist() == [20.0, FIELD_WIDTH - 20.0]


def test_reproject_preserves_nan():
    df = pd.DataFrame({"x": [np.nan], "y": [np.nan]})
    out = reproject_to_action_ltr(df, pd.Series([True]), x_cols=["x"], y_cols=["y"])
    assert out["x"].isna().all() and out["y"].isna().all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_action_orientation.py -v`
Expected: FAIL — `ModuleNotFoundError: silly_kicks.tracking._action_orientation`.

- [ ] **Step 3: Write the module**

```python
# silly_kicks/tracking/_action_orientation.py
"""Canonical per-action re-projection of frame-sampled positions into SPADL action-LTR.

`convert_to_frames` emits home-attacks-right coordinates; `to_spadl_ltr` emits
per-acting-team-LTR action coordinates. They agree for home-team actions and are
a 180-degree point reflection (x->105-x, y->68-y) apart for away-team actions.

Every emitted per-action tracking-geometry POSITION column must be expressed in
the action-LTR frame of the action it annotates. This module is the single
source of truth for that re-projection. Decision: ADR-028.

The per-action flip is derived from the frame's `team_attacking_direction`
(ground truth of "which way does this team attack in these coordinates"), so the
helper is robust to ANY frame orientation and needs no `home_team_id`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FIELD_LENGTH = 105.0
FIELD_WIDTH = 68.0

__all__ = ["FIELD_LENGTH", "FIELD_WIDTH", "acting_team_attacks_rtl", "reproject_to_action_ltr"]


def acting_team_attacks_rtl(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
) -> pd.Series:
    """Per-action boolean: True iff the acting team attacks RIGHT-TO-LEFT in the frames.

    A True row means the action's LTR frame is the 180-degree mirror of the frame
    coordinate system, so frame-sampled positions for that action must be flipped
    (x->105-x, y->68-y) to land in the action-LTR frame.

    Derivation: build a (game_id, period_id, team_id) -> attacking_direction lookup
    from non-ball frame rows, then map each action's (game_id, period_id, team_id).
    Actions whose acting team has no resolvable direction (absent from the frame,
    or NaN/None direction) default to False (no flip); such actions produce NaN
    geometry anyway because they cannot link to a usable position.

    Returned Series is index-aligned to `actions`.
    """
    flip = pd.Series(False, index=actions.index)
    if len(actions) == 0 or len(frames) == 0:
        return flip
    if "team_attacking_direction" not in frames.columns:
        return flip

    players = frames[~frames["is_ball"].astype(bool)]
    players = players[players["team_attacking_direction"].notna()]
    if players.empty:
        return flip

    # One direction per (game, period, team): first non-null (constant within a period).
    lookup = (
        players.groupby(["game_id", "period_id", "team_id"])["team_attacking_direction"]
        .first()
        .reset_index()
        .rename(columns={"team_attacking_direction": "_dir"})
    )

    keyed = actions[["game_id", "period_id", "team_id"]].merge(
        lookup, on=["game_id", "period_id", "team_id"], how="left"
    )
    keyed.index = actions.index
    return (keyed["_dir"] == "rtl").fillna(False)


def reproject_to_action_ltr(
    df: pd.DataFrame,
    flip_mask: pd.Series,
    *,
    x_cols: list[str],
    y_cols: list[str],
) -> pd.DataFrame:
    """Return a copy of `df` with `x_cols`/`y_cols` mirrored on rows where `flip_mask`.

    x -> 105 - x and y -> 68 - y on flipped rows; NaN preserved (NaN arithmetic).
    `flip_mask` must be index-aligned to `df`.
    """
    out = df.copy()
    mask = flip_mask.reindex(out.index, fill_value=False).to_numpy(dtype=bool)
    if not mask.any():
        return out
    for col in x_cols:
        if col in out.columns:
            vals = out[col].to_numpy(dtype="float64")
            out.loc[mask, col] = FIELD_LENGTH - vals[mask]
    for col in y_cols:
        if col in out.columns:
            vals = out[col].to_numpy(dtype="float64")
            out.loc[mask, col] = FIELD_WIDTH - vals[mask]
    return out
```

**Snapshot note:** `_snapshot.py` sets both teams' `team_attacking_direction="ltr"`, so `acting_team_attacks_rtl` returns all-False for snapshot frames → no flip → snapshot is treated as already action-LTR, which is the synthetic snapshot contract. No special-casing needed.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_action_orientation.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add silly_kicks/tracking/_action_orientation.py tests/tracking/test_action_orientation.py
git commit -F .git/COMMIT_ao.txt   # message: "feat(tracking): canonical action-LTR re-projection helper (ADR-028)"
```

---

## Task 2: Re-project the shared ActionFrameContext (fixes 8 context kernels)

**Files:**
- Modify: `silly_kicks/tracking/utils.py` — `_resolve_action_frame_context` (ends ~742)
- Test: `tests/tracking/test_pre_shot_gk_orientation.py` (new)

- [ ] **Step 1: Write the failing test** — the away-team-shot reproduction, asserting the FIXED contract.

```python
# tests/tracking/test_pre_shot_gk_orientation.py
"""ADR-028: pre-shot GK + pressure are emitted in the action-LTR frame."""
from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl.utils import add_pre_shot_gk_context

HOME, AWAY = 1, 2
SHOT = spadlconfig.actiontype_id["shot"]
GOALKICK = spadlconfig.actiontype_id["goalkick"]


def _frame_rows(shooter_is_away: bool):
    # Home-attacks-right frame: home GK defends x=0 (~3), away GK defends x=105 (~102).
    base = dict(game_id=1, period_id=1, frame_id=250, time_seconds=10.0, frame_rate=25.0,
                z=0.0, speed=0.0, speed_source="native", ball_state="alive",
                confidence=None, visibility=None, source_provider="synthetic",
                is_goalkeeper_source="native")
    rows = [
        dict(player_id=1, team_id=HOME, is_ball=False, is_goalkeeper=True, x=3.0, y=34.0, team_attacking_direction="ltr"),
        dict(player_id=50, team_id=AWAY, is_ball=False, is_goalkeeper=True, x=102.0, y=34.0, team_attacking_direction="rtl"),
        dict(player_id=11, team_id=HOME, is_ball=False, is_goalkeeper=False, x=40.0, y=20.0, team_attacking_direction="ltr"),
        dict(player_id=61, team_id=AWAY, is_ball=False, is_goalkeeper=False, x=65.0, y=40.0, team_attacking_direction="rtl"),
        dict(player_id=np.nan, team_id=np.nan, is_ball=True, is_goalkeeper=False, x=13.0, y=34.0, team_attacking_direction=None),
    ]
    return pd.DataFrame([{**base, **r} for r in rows])


def test_away_shot_gk_reprojected_to_attacked_goal():
    frames = _frame_rows(shooter_is_away=True)
    actions = pd.DataFrame([
        dict(game_id=1, period_id=1, action_id=0, team_id=HOME, player_id=1.0, type_id=GOALKICK,
             result_id=1, start_x=5.0, start_y=34.0, end_x=40.0, end_y=34.0, time_seconds=9.6),
        dict(game_id=1, period_id=1, action_id=1, team_id=AWAY, player_id=99.0, type_id=SHOT,
             result_id=1, start_x=92.0, start_y=34.0, end_x=105.0, end_y=34.0, time_seconds=10.0),
    ])
    enriched = add_pre_shot_gk_context(actions, frames=frames)
    shot = enriched[enriched["type_id"] == SHOT].iloc[0]
    # Defending GK (home, frame x=3) re-projected to action-LTR: 105-3 = 102, near attacked goal.
    assert shot["pre_shot_gk_x"] == 102.0
    assert shot["pre_shot_gk_y"] == 34.0
    assert shot["pre_shot_gk_distance_to_goal"] == 3.0          # |105-102|
    assert abs(shot["pre_shot_gk_distance_to_shot"] - 10.0) < 1e-9  # |102-92|
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_pre_shot_gk_orientation.py -v`
Expected: FAIL — `pre_shot_gk_x == 3.0`, `distance_to_goal == 102.0` (the bug).

- [ ] **Step 3: Implement — re-project the three sampled-row frames in `_resolve_action_frame_context`.**

At the top of `silly_kicks/tracking/utils.py`, add the import (near the other tracking imports):

```python
from ._action_orientation import acting_team_attacks_rtl, reproject_to_action_ltr
```

In `_resolve_action_frame_context`, immediately **before** the `return ActionFrameContext(...)` (currently utils.py:736), insert:

```python
    # ADR-028: re-project the sampled frame positions into each action's LTR frame.
    # Frames are home-attacks-right; actions are per-acting-team-LTR. They are a
    # 180-degree mirror apart for away-team actions. After this, the kernels'
    # hardcoded goal at (105, 34) is correct because the acting team attacks x=105.
    flip = acting_team_attacks_rtl(actions, frames)  # index: actions.index
    flip_by_action = pd.Series(flip.to_numpy(), index=actions["action_id"].to_numpy())

    def _reproject_rows(rows: pd.DataFrame) -> pd.DataFrame:
        if rows.empty or "action_id" not in rows.columns:
            return rows
        row_flip = rows["action_id"].map(flip_by_action).fillna(False)
        row_flip.index = rows.index
        return reproject_to_action_ltr(rows, row_flip, x_cols=["x"], y_cols=["y"])

    actor_rows = _reproject_rows(actor_rows)
    opposite = _reproject_rows(opposite)
    defending_gk_rows = _reproject_rows(defending_gk_rows)
```

(Note: `flip_by_action` keyed on `action_id` is safe here — within one `_resolve_action_frame_context` call the actions are the caller's unique action rows; the gamestates-duplicate concern lives only in the `_*_at_actions` positional path, not here.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_pre_shot_gk_orientation.py tests/tracking/test_action_orientation.py -v`
Expected: PASS.

- [ ] **Step 5: Run the broader context-feature suites to catch fixtures that encoded the bug**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_features_standard.py tests/tracking/test_kernels.py tests/tracking/test_pressure*.py tests/tracking/test_action_context_expected_output.py -v --tb=short`
Expected: PASS. If any fails asserting an OLD away-action value, it encoded the bug — fix the expectation to the re-projected value and note it in the commit. If a fixture used all-"ltr" frames (no flip) it is unchanged.

- [ ] **Step 6: Commit**

```bash
git add silly_kicks/tracking/utils.py tests/tracking/test_pre_shot_gk_orientation.py
git commit -F .git/COMMIT_ctx.txt   # "fix(tracking)!: re-project ActionFrameContext positions to action-LTR (ADR-028)"
```

---

## Task 3: Re-project defensive-line coupling

**Files:**
- Modify: `silly_kicks/tracking/_kernels.py` — `_defensive_line_at_actions` (783-872)
- Test: `tests/tracking/test_defensive_line_orientation.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/tracking/test_defensive_line_orientation.py
"""ADR-028: defensive_line_x / back_line_high_x emitted in action-LTR frame."""
from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking.features import add_defensive_line

HOME, AWAY = 1, 2
PASS = spadlconfig.actiontype_id["pass"]


def _frames():
    base = dict(game_id=1, period_id=1, frame_id=100, time_seconds=4.0, frame_rate=25.0,
                z=0.0, speed=0.0, speed_source="native", ball_state="alive",
                confidence=None, visibility=None, source_provider="synthetic",
                is_goalkeeper_source="native", is_goalkeeper=False)
    rows = []
    # Home back line near x=20 (defends x=0). Away back line near x=85 (defends x=105).
    for i, x in enumerate((18.0, 20.0, 22.0, 24.0)):
        rows.append(dict(player_id=10 + i, team_id=HOME, is_ball=False, x=x, y=20.0 + i * 8, team_attacking_direction="ltr"))
    for i, x in enumerate((81.0, 83.0, 85.0, 87.0)):
        rows.append(dict(player_id=60 + i, team_id=AWAY, is_ball=False, x=x, y=20.0 + i * 8, team_attacking_direction="rtl"))
    rows.append(dict(player_id=np.nan, team_id=np.nan, is_ball=True, x=50.0, y=34.0, team_attacking_direction=None))
    return pd.DataFrame([{**base, **r} for r in rows])


def test_away_action_defending_line_reprojected():
    frames = _frames()
    # Away team passes (LTR-normalized: away attacks x=105). Defending team = HOME (near x=20).
    actions = pd.DataFrame([
        dict(game_id=1, period_id=1, action_id=0, team_id=AWAY, player_id=99.0, type_id=PASS,
             result_id=1, start_x=70.0, start_y=34.0, end_x=80.0, end_y=40.0, time_seconds=4.0),
    ])
    out = add_defensive_line(actions, frames, home_team_id=HOME)
    # Home defenders at mean x=21 in frame; re-projected to action-LTR (away attacks 105): 105-21 = 84.
    assert abs(out["defensive_line_x"].iloc[0] - 84.0) < 1e-9
    # compactness_x (a span) is invariant: max(24)-min(18)=6 in frame, unchanged.
    assert abs(out["compactness_x"].iloc[0] - 6.0) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_defensive_line_orientation.py -v`
Expected: FAIL — `defensive_line_x == 21.0` (frame coords, not re-projected).

- [ ] **Step 3: Implement.** In `_kernels.py`, add to the imports at top of the module:

```python
from ._action_orientation import acting_team_attacks_rtl, reproject_to_action_ltr
```

In `_defensive_line_at_actions`, replace the final mapping block (currently lines 863-872, from `out = empty.copy()` through `return out`) with a version that re-projects per `_row_idx`:

```python
    # Map back to actions index by position
    out = empty.copy()
    for _, row in opposing.iterrows():
        pos = int(row["_row_idx"])
        idx = actions.index[pos]
        for col in feature_cols:
            out.at[idx, col] = row[col]

    out["back_n_count"] = out["back_n_count"].astype("Int64")

    # ADR-028: re-project the two x-positions into each action's LTR frame.
    # compactness_x / lateral_width / max_lateral_gap / back_n_count are spans/counts
    # (flip-invariant) and are left unchanged.
    flip = acting_team_attacks_rtl(actions, frames)
    out = reproject_to_action_ltr(out, flip, x_cols=["defensive_line_x", "back_line_high_x"], y_cols=[])
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_defensive_line_orientation.py tests/tracking/test_gk_influence_action_coupled.py -k "defensive or line" -v --tb=short`
Expected: PASS. (Also run the full `test_*defensive*` / `test_off_ball*` suites; fix any bug-encoding fixtures.)

- [ ] **Step 5: Commit**

```bash
git add silly_kicks/tracking/_kernels.py tests/tracking/test_defensive_line_orientation.py
git commit -F .git/COMMIT_dl.txt   # "fix(tracking)!: re-project defensive-line positions to action-LTR (ADR-028)"
```

---

## Task 4: Re-project team-shape coupling (both add_ and xfns paths)

**Files:**
- Modify: `silly_kicks/tracking/features.py` — `add_team_shape` (1491+) and `_team_shape_at_actions` (1679+)
- Test: `tests/tracking/test_team_shape_orientation.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/tracking/test_team_shape_orientation.py
"""ADR-028: team-shape centroids / line-height emitted in action-LTR frame."""
from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking.features import add_team_shape

HOME, AWAY = 1, 2
PASS = spadlconfig.actiontype_id["pass"]


def _frames():
    base = dict(game_id=1, period_id=1, frame_id=100, time_seconds=4.0, frame_rate=25.0,
                z=0.0, speed=0.0, speed_source="native", ball_state="alive",
                confidence=None, visibility=None, source_provider="synthetic",
                is_goalkeeper_source="native", is_goalkeeper=False)
    rows = []
    for i in range(10):
        rows.append(dict(player_id=10 + i, team_id=HOME, is_ball=False, x=20.0 + i, y=10.0 + i * 5, team_attacking_direction="ltr"))
    for i in range(10):
        rows.append(dict(player_id=60 + i, team_id=AWAY, is_ball=False, x=70.0 + i, y=10.0 + i * 5, team_attacking_direction="rtl"))
    rows.append(dict(player_id=np.nan, team_id=np.nan, is_ball=True, x=50.0, y=34.0, team_attacking_direction=None))
    return pd.DataFrame([{**base, **r} for r in rows])


def test_away_action_centroids_reprojected_both_axes():
    frames = _frames()
    actions = pd.DataFrame([
        dict(game_id=1, period_id=1, action_id=0, team_id=AWAY, player_id=99.0, type_id=PASS,
             result_id=1, start_x=70.0, start_y=34.0, end_x=80.0, end_y=40.0, time_seconds=4.0),
    ])
    out = add_team_shape(actions, frames, home_team_id=HOME).iloc[0]
    # Attacking team = AWAY. Frame centroid_x = mean(70..79)=74.5 -> action-LTR 105-74.5 = 30.5.
    assert abs(out["team_shape_centroid_x_attacking"] - 30.5) < 1e-9
    # Frame centroid_y = mean(10,15,...,55)=32.5 -> action-LTR 68-32.5 = 35.5.
    assert abs(out["team_shape_centroid_y_attacking"] - 35.5) < 1e-9
    # team_length is a span (max-min x) -> invariant: 9.0.
    assert abs(out["team_shape_team_length_attacking"] - 9.0) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_team_shape_orientation.py -v`
Expected: FAIL — `centroid_x_attacking == 74.5`.

- [ ] **Step 3: Implement.** Add a shared private helper in `features.py` (near the team-shape functions) and call it from both seams. First add the import at top of `features.py`:

```python
from ._action_orientation import acting_team_attacks_rtl, reproject_to_action_ltr
```

Add the column lists + helper (module-level, above `add_team_shape`):

```python
# ADR-028: team-shape position columns to re-project into action-LTR (both teams).
_TEAM_SHAPE_X_COLS = [
    "team_shape_centroid_x_attacking", "team_shape_centroid_x_defending",
    "team_shape_defensive_line_height_attacking", "team_shape_defensive_line_height_defending",
]
_TEAM_SHAPE_Y_COLS = [
    "team_shape_centroid_y_attacking", "team_shape_centroid_y_defending",
]


def _reproject_team_shape(out: pd.DataFrame, actions: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """Mirror team-shape centroid/line-height positions into each action's LTR frame (ADR-028)."""
    flip = acting_team_attacks_rtl(actions, frames)
    return reproject_to_action_ltr(out, flip, x_cols=_TEAM_SHAPE_X_COLS, y_cols=_TEAM_SHAPE_Y_COLS)
```

In `add_team_shape`, immediately before its `return out`, replace `return out` with:

```python
    out = _reproject_team_shape(out, actions, frames)
    return out
```

In `_team_shape_at_actions`, find where it assembles the per-slot output DataFrame and returns it; apply the same `_reproject_team_shape(result, actions, frames)` before returning. (Read the function body 1679+; the position columns it emits are the same `_TEAM_SHAPE_X_COLS`/`_TEAM_SHAPE_Y_COLS`. If `_team_shape_at_actions` operates on a per-slot `actions` argument, pass that same slot to `_reproject_team_shape` so the flip mask aligns.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_team_shape_orientation.py tests/tracking/test_team_shape*.py -v --tb=short`
Expected: PASS. Fix any bug-encoding fixtures.

- [ ] **Step 5: Commit**

```bash
git add silly_kicks/tracking/features.py tests/tracking/test_team_shape_orientation.py
git commit -F .git/COMMIT_ts.txt   # "fix(tracking)!: re-project team-shape positions to action-LTR (ADR-028)"
```

---

## Task 5: Emit ghost-GK in the action-LTR frame

**Files:**
- Modify: `silly_kicks/tracking/_ghost_gk.py` — the emit seam in `compute_ghost_gk` / `add_ghost_gk` (where `ghost_gk_x`/`ghost_gk_y` are written; ~1857+)
- Modify: `silly_kicks/atomic/tracking/features.py` — ghost-GK atomic mirror
- Test: `tests/tracking/test_ghost_gk_orientation.py` (new)

**Transform (verified):** the model predicts goal-relative `(gr_x, gr_y)` with the defended goal at x=0. In action-LTR the defended goal is at x=105, so:
- `ghost_gk_x_action_ltr = 105 - gr_x` — **uniform** (no per-action flip; `gr_x` already measures from the defended goal).
- `ghost_gk_y_action_ltr = gr_y` for home-team actions, `68 - gr_y` for away-team actions — **per-action flip**.

Apply identically to any `ghost_gk_mode_{x,y}` / `ghost_gk_mean_{x,y}` columns present. `ghost_gk_density_spread` is invariant.

- [ ] **Step 1: Write the failing test**

```python
# tests/tracking/test_ghost_gk_orientation.py
"""ADR-028: ghost_gk_x/y emitted in action-LTR frame (defended goal at x=105)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking.features import add_ghost_gk

HOME, AWAY = 1, 2
SHOT = spadlconfig.actiontype_id["shot"]


def _frames_two_shots():
    base = dict(game_id=1, period_id=1, frame_rate=25.0, z=0.0, speed=0.0, speed_source="native",
                ball_state="alive", confidence=None, visibility=None, source_provider="synthetic",
                is_goalkeeper_source="native")
    rows = []
    for fid, t in ((100, 4.0), (200, 8.0)):
        rows += [
            dict(frame_id=fid, time_seconds=t, player_id=1, team_id=HOME, is_ball=False, is_goalkeeper=True, x=4.0, y=34.0, team_attacking_direction="ltr"),
            dict(frame_id=fid, time_seconds=t, player_id=50, team_id=AWAY, is_ball=False, is_goalkeeper=True, x=101.0, y=34.0, team_attacking_direction="rtl"),
            dict(frame_id=fid, time_seconds=t, player_id=11, team_id=HOME, is_ball=False, is_goalkeeper=False, x=40.0, y=30.0, team_attacking_direction="ltr"),
            dict(frame_id=fid, time_seconds=t, player_id=61, team_id=AWAY, is_ball=False, is_goalkeeper=False, x=65.0, y=38.0, team_attacking_direction="rtl"),
            dict(frame_id=fid, time_seconds=t, player_id=np.nan, team_id=np.nan, is_ball=True, is_goalkeeper=False, x=50.0, y=34.0, team_attacking_direction=None),
        ]
    return pd.DataFrame([{**base, **r} for r in rows])


def test_ghost_gk_x_is_action_ltr_near_attacked_goal():
    frames = _frames_two_shots()
    actions = pd.DataFrame([
        dict(game_id=1, period_id=1, action_id=0, team_id=HOME, player_id=11.0, type_id=SHOT,
             result_id=1, start_x=90.0, start_y=34.0, end_x=105.0, end_y=34.0, time_seconds=4.0),
        dict(game_id=1, period_id=1, action_id=1, team_id=AWAY, player_id=61.0, type_id=SHOT,
             result_id=1, start_x=90.0, start_y=34.0, end_x=105.0, end_y=34.0, time_seconds=8.0),
    ])
    out = add_ghost_gk(actions, frames, home_team_id=HOME)
    gx = out["ghost_gk_x"].to_numpy()
    # In action-LTR the defended goal is x=105; a ghost defending GK sits near it (x >> 50)
    # for BOTH home and away shots (no own-goal-end bimodality).
    assert np.all(gx[np.isfinite(gx)] > 70.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_gk_orientation.py -v`
Expected: FAIL — `ghost_gk_x` ≈ 13 (goal-relative own-goal end).

- [ ] **Step 3: Implement.** Read the emit seam in `_ghost_gk.py` where goal-relative `ghost_gk_x`/`ghost_gk_y` (and any mode/mean variants) are assigned onto the per-action output. Add the import:

```python
from ._action_orientation import FIELD_LENGTH, FIELD_WIDTH, acting_team_attacks_rtl
```

After the goal-relative values are computed and aligned to `actions`, transform (let `gx_col`/`gy_col` iterate over the emitted x/y pairs, e.g. `[("ghost_gk_x","ghost_gk_y")]` plus any mode/mean pairs that exist):

```python
    # ADR-028: emit in action-LTR. gr_x measures from the defended goal (x=0); in
    # action-LTR the defended goal is x=105, so x -> 105 - gr_x uniformly. The
    # goal-relative transform left y in absolute-frame terms, so y mirrors only
    # for away-team actions (per-action 180-degree reflection).
    flip = acting_team_attacks_rtl(actions, frames).to_numpy(dtype=bool)  # aligned to actions order
    for gx_col, gy_col in _GHOST_GK_XY_PAIRS:  # define near the emit; only pairs that exist
        if gx_col in out.columns:
            x_gr = out[gx_col].to_numpy(dtype="float64")
            out[gx_col] = FIELD_LENGTH - x_gr
        if gy_col in out.columns:
            y_gr = out[gy_col].to_numpy(dtype="float64")
            out[gy_col] = np.where(flip, FIELD_WIDTH - y_gr, y_gr)
```

Ensure `flip` is aligned to the same row order as `out` (use `.to_numpy()` only if `out` is in `actions` order; otherwise key by action via `acting_team_attacks_rtl(...).reindex(out.index)`). Mirror the identical transform into the atomic `add_ghost_gk` in `silly_kicks/atomic/tracking/features.py` (it synthesizes `end=x+dx`; the GK-emit transform is the same).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_gk_orientation.py tests/tracking/test_ghost_gk*.py -v --tb=short`
Expected: PASS. Several existing ghost-GK tests assert goal-relative `ghost_gk_x≈[0,30]` — update those expectations to action-LTR (`105 − gr_x`) and note in commit (this is the intended breaking change).

- [ ] **Step 5: Commit**

```bash
git add silly_kicks/tracking/_ghost_gk.py silly_kicks/atomic/tracking/features.py tests/tracking/test_ghost_gk_orientation.py tests/tracking/test_ghost_gk*.py
git commit -F .git/COMMIT_ghost.txt   # "fix(tracking)!: emit ghost-GK position in action-LTR frame (ADR-028)"
```

---

## Task 6: Mirror-symmetry property test (the durable guard)

**Files:**
- Test: `tests/tracking/test_action_ltr_mirror_invariance.py` (new)

The invariant: for the *same physical situation*, the emitted action-LTR values must be identical whether the acting team attacks left or right in the frame. Construct one fixture, then build its physical mirror (flip every frame x/y AND swap which team is home / the action's `start/end` mirrored AND swap `team_attacking_direction`), and assert the emitted geometry columns match within tolerance.

- [ ] **Step 1: Write the test**

```python
# tests/tracking/test_action_ltr_mirror_invariance.py
"""ADR-028 durable guard: emitted action-LTR geometry is invariant under a physical
left/right mirror of the frame + action. A row that physically happens near the
absolute-left goal must yield the SAME action-LTR feature values as its mirror
near the absolute-right goal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl.utils import add_pre_shot_gk_context
from silly_kicks.tracking.features import add_defensive_line, add_team_shape

HOME, AWAY = 1, 2
SHOT = spadlconfig.actiontype_id["shot"]
GOALKICK = spadlconfig.actiontype_id["goalkick"]
PASS = spadlconfig.actiontype_id["pass"]
FL, FW = 105.0, 68.0


def _scenario():
    """Home team shoots toward x=105 (home action). Returns (actions, frames)."""
    base = dict(game_id=1, period_id=1, frame_id=100, time_seconds=4.0, frame_rate=25.0, z=0.0,
                speed=0.0, speed_source="native", ball_state="alive", confidence=None,
                visibility=None, source_provider="synthetic", is_goalkeeper_source="native")
    rows = [
        dict(player_id=1, team_id=HOME, is_ball=False, is_goalkeeper=True, x=4.0, y=34.0, team_attacking_direction="ltr"),
        dict(player_id=50, team_id=AWAY, is_ball=False, is_goalkeeper=True, x=101.0, y=34.0, team_attacking_direction="rtl"),
        dict(player_id=11, team_id=HOME, is_ball=False, is_goalkeeper=False, x=70.0, y=30.0, team_attacking_direction="ltr"),
        dict(player_id=12, team_id=HOME, is_ball=False, is_goalkeeper=False, x=60.0, y=44.0, team_attacking_direction="ltr"),
        dict(player_id=61, team_id=AWAY, is_ball=False, is_goalkeeper=False, x=86.0, y=30.0, team_attacking_direction="rtl"),
        dict(player_id=62, team_id=AWAY, is_ball=False, is_goalkeeper=False, x=84.0, y=40.0, team_attacking_direction="rtl"),
        dict(player_id=np.nan, team_id=np.nan, is_ball=True, is_goalkeeper=False, x=88.0, y=34.0, team_attacking_direction=None),
    ]
    frames = pd.DataFrame([{**base, **r} for r in rows])
    actions = pd.DataFrame([
        dict(game_id=1, period_id=1, action_id=0, team_id=AWAY, player_id=50.0, type_id=GOALKICK,
             result_id=1, start_x=5.0, start_y=34.0, end_x=40.0, end_y=34.0, time_seconds=3.6),
        dict(game_id=1, period_id=1, action_id=1, team_id=HOME, player_id=11.0, type_id=SHOT,
             result_id=1, start_x=88.0, start_y=34.0, end_x=105.0, end_y=34.0, time_seconds=4.0),
    ])
    return actions, frames


def _mirror(actions, frames):
    """Physical left/right mirror: flip all frame x/y, swap directions, mirror action coords.
    The home team now attacks left; the SAME physical situation, opposite absolute orientation.
    """
    f = frames.copy()
    f["x"] = FL - f["x"]
    f["y"] = FW - f["y"]
    f["team_attacking_direction"] = f["team_attacking_direction"].map({"ltr": "rtl", "rtl": "ltr"})
    a = actions.copy()
    for c in ("start_x", "end_x"):
        a[c] = FL - a[c]
    for c in ("start_y", "end_y"):
        a[c] = FW - a[c]
    return a, f


_GEOMETRY_COLS = [
    "pre_shot_gk_x", "pre_shot_gk_y", "pre_shot_gk_distance_to_goal", "pre_shot_gk_distance_to_shot",
]


def _run_pre_shot(actions, frames):
    return add_pre_shot_gk_context(actions, frames=frames)


def test_pre_shot_gk_mirror_invariant():
    a, f = _scenario()
    am, fm = _mirror(a, f)
    base = _run_pre_shot(a, f).set_index("action_id")
    mir = _run_pre_shot(am, fm).set_index("action_id")
    shot_aid = 1
    for col in _GEOMETRY_COLS:
        b, m = base.loc[shot_aid, col], mir.loc[shot_aid, col]
        assert (pd.isna(b) and pd.isna(m)) or abs(b - m) < 1e-6, f"{col}: {b} vs {m}"


def test_defensive_line_mirror_invariant():
    a, f = _scenario()
    am, fm = _mirror(a, f)
    base = add_defensive_line(a, f, home_team_id=HOME).set_index("action_id")
    # After mirror, home attacks left; the home_team_id is still HOME but its direction flipped.
    mir = add_defensive_line(am, fm, home_team_id=HOME).set_index("action_id")
    for col in ("defensive_line_x", "back_line_high_x", "compactness_x"):
        b, m = base.loc[1, col], mir.loc[1, col]
        assert (pd.isna(b) and pd.isna(m)) or abs(b - m) < 1e-6, f"{col}: {b} vs {m}"


def test_team_shape_mirror_invariant():
    a, f = _scenario()
    am, fm = _mirror(a, f)
    base = add_team_shape(a, f, home_team_id=HOME).set_index("action_id")
    mir = add_team_shape(am, fm, home_team_id=HOME).set_index("action_id")
    for col in ("team_shape_centroid_x_attacking", "team_shape_centroid_y_attacking",
                "team_shape_centroid_x_defending", "team_shape_team_length_attacking"):
        b, m = base.loc[1, col], mir.loc[1, col]
        assert (pd.isna(b) and pd.isna(m)) or abs(b - m) < 1e-6, f"{col}: {b} vs {m}"
```

- [ ] **Step 2: Run — all three must pass** (they exercise the full pipeline against the durable invariant).

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_action_ltr_mirror_invariance.py -v`
Expected: PASS (3 tests). If any fails, a seam still leaks frame orientation — fix that seam, do not weaken the test.

- [ ] **Step 3: Commit**

```bash
git add tests/tracking/test_action_ltr_mirror_invariance.py
git commit -F .git/COMMIT_mirror.txt   # "test(tracking): mirror-symmetry guard for action-LTR geometry (ADR-028)"
```

---

## Task 7: Full suite, docs, version bump, ADR

**Files:**
- Modify: `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, `TODO.md`, `uv.lock` (via `uv lock`)
- Create: `docs/superpowers/adrs/ADR-028-tracking-action-ltr-reprojection.md`
- Modify: `CLAUDE.md` (Tracking section), `NOTICE` (contract note — no new methodology)

- [ ] **Step 1: Full local suite (the gate)**

Run (background, poll): `.venv/Scripts/python.exe -m pytest tests/ -m "not e2e" -q --tb=short`
Expected: PASS. Triage failures into (a) bug-encoding fixtures → update to re-projected values; (b) real regressions → fix the seam. Pay attention to `test_aggregator_column_liveness.py` (still passes — non-null/non-constant unaffected) and `test_id_dtype_invariance.py` (the helper uses `team_attacking_direction`, not id `==`, so no new id seam; but the action↔frame `merge` in `acting_team_attacks_rtl` joins on `team_id` — run `align_join_keys` if a string-id leg fails, mirroring the existing pattern).

- [ ] **Step 2: Lint + types (replicate the full CI lint job)**

Run: `.venv/Scripts/python.exe -m ruff check silly_kicks/ tests/ scripts/` and `ruff format --check` and `.venv312`-pinned `pyright` over the whole tree. Expected: clean. (`# type: ignore` only where the codebase idiom requires.)

- [ ] **Step 3: Write ADR-028** — `docs/superpowers/adrs/ADR-028-tracking-action-ltr-reprojection.md`. Sections: Context (the two conventions + 180° mirror + why a single frame can't be pre-oriented for both teams), Decision (canonical contract: every emitted per-action geometry position column is action-LTR; centralized at 3 seams via `_action_orientation`; direction from `team_attacking_direction`; ghost-GK emit transform), Scope (the full A/B/C table; what is intentionally untouched and why — self-reconciling B features, invariant C), Consequences (VAEP/tracking retrain trigger; lakehouse re-materialize; mirror-symmetry guard; the report's "re-project outputs" rejected because it misses the mixed-frame Type-2 scalars).

- [ ] **Step 4: Version bump to 4.26.0** — `pyproject.toml` `version`, `silly_kicks/__init__.py` `__version__`, `CHANGELOG.md` top entry (breaking, list every changed column family + retrain note), `TODO.md` if it tracks this, then `uv lock`.

- [ ] **Step 5: CLAUDE.md Tracking section** — add a one-line ADR-028 note: every emitted per-action geometry POSITION column is in the action-LTR frame (acting team attacks x=105), re-projected centrally via `tracking/_action_orientation.py` from `team_attacking_direction`; ghost-GK now action-LTR (was goal-relative); self-reconciling B features untouched; mirror-symmetry CI guard. VAEP/tracking-retrain trigger.

- [ ] **Step 6: Commit docs + bump**

```bash
git add -A
git commit -F .git/COMMIT_docs.txt   # "docs(tracking): ADR-028 + 4.26.0 bump for action-LTR geometry re-projection"
```

---

## Self-review checklist (run before opening the PR)

- **Spec coverage:** every (A) column from the audit is fixed by Task 2/3/4; ghost-GK by Task 5; the Type-2 mixed-frame scalars (nearest_defender_distance, receiver_zone_density, defenders_in_triangle, all pressure flavors, pre_shot_gk distances/angles) are fixed transitively by Task 2. Invariant (C) columns untouched. Self-reconciling (B) features untouched.
- **No double-flip:** confirm none of the B features (structural_pass, gk_influence, player_influence, cover_shadows, shape_graph) consume `ActionFrameContext` (grep clean — only `_kernels.py` context functions do).
- **Mirror guard green** on pre_shot_gk + defensive_line + team_shape.
- **Liveness + id-dtype-invariance + dup-action_id gates green.**
- **C4:** no new aggregator / KDE backend / trained model → aggregator count unchanged → C4-free (confirm tokens/count unchanged).
- **Retrain trigger** stated in CHANGELOG + ADR + CLAUDE.md.
```
