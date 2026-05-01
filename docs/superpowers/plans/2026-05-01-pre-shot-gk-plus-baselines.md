# Pre-shot GK position + baselines backfill — Implementation Plan (PR-S21)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship silly-kicks 2.9.0 with TF-1 (`pre_shot_gk_position_*` 4-feature catalog refining `add_pre_shot_gk_context`) and TF-11 (baselines backfill + bit-exact per-row regression gate) bundled into a single PR, plus 3 National Park TODO additions (TF-12 / TF-13 / TF-14).

**Architecture:** Tracking-namespace canonical compute (`silly_kicks.tracking.features.pre_shot_gk_*`) shared by both standard SPADL and atomic SPADL via private `_kernels._pre_shot_gk_position`. Events-side `add_pre_shot_gk_context` gains optional `frames=None` kwarg that lazy-imports the canonical compute. Bit-exact regression gate via `*_expected.parquet` per provider; JSON baselines retained as documentation. Ships entirely within ADR-005's 7 cross-cutting decisions — no new ADR.

**Tech Stack:** Python 3.10+, pandas (long-form joins + pd.testing.assert_frame_equal), numpy (vectorized geometry), scikit-learn (HybridVAEP gradient-boosting), pytest (TDD), pyright (strict typing), ruff (lint), uv (env mgmt).

**Spec:** `docs/superpowers/specs/2026-05-01-pre-shot-gk-plus-baselines-design.md`

**ADRs:** ADR-003 (NaN safety), ADR-004 (tracking namespace charter), ADR-005 (tracking-aware feature integration contract).

**Memory references applied throughout:** `feedback_commit_policy` (one squash commit), `feedback_final_review_gate` (mandatory `/final-review`), `feedback_no_silent_skips_on_required_testing` (loud-skip), `feedback_inline_execution_default` (no subagents unless explicit), `feedback_lakehouse_consumer_not_source` (raw provider values), `reference_pff_data_local` (PFF tracking root via env var; never committed), `feedback_ci_cross_version` (pin pyright + pandas-stubs).

---

## Loop 0 — Bootstrap & shot-count probe

**Files:** none modified; one ad-hoc probe script written and discarded.

- [ ] **Step 0.1: Create the feature branch**

```bash
git checkout main
git pull --ff-only origin main
git checkout -b feat/pre-shot-gk-plus-baselines
```

Expected: `Switched to a new branch 'feat/pre-shot-gk-plus-baselines'`.

- [ ] **Step 0.2: Probe shot-class action counts in committed slim parquets**

Write a one-shot probe (do NOT commit):

```python
# scripts/_probe_slim_shots.py — temporary; deleted at end of Loop 0
from pathlib import Path
import pandas as pd
from silly_kicks.spadl import config as spadlconfig

shot_type_names = {"shot", "shot_freekick", "shot_penalty"}
shot_ids = {spadlconfig.actiontype_id[n] for n in shot_type_names}

base = Path("tests/datasets/tracking/action_context_slim")
for prov in ("sportec", "metrica", "skillcorner"):
    df = pd.read_parquet(base / f"{prov}_slim.parquet")
    actions = df[df["__row_kind"] == "action"] if "__row_kind" in df.columns else df  # accommodate either layout
    # If slim parquet stores actions and frames in separate files:
    actions_path = base / f"{prov}_slim_actions.parquet"
    if actions_path.exists():
        actions = pd.read_parquet(actions_path)
    n_shots = int(actions["type_id"].isin(shot_ids).sum()) if "type_id" in actions.columns else -1
    n_actions = len(actions)
    print(f"{prov}: {n_shots}/{n_actions} actions are shots")

pff_path = Path("tests/datasets/tracking/pff/medium_halftime.parquet")
df = pd.read_parquet(pff_path)
print(f"pff medium_halftime: {len(df)} rows; columns={list(df.columns)[:8]}...")
```

Run:

```bash
uv run python scripts/_probe_slim_shots.py
```

Expected output (exact format depends on slim-parquet layout — discovery step):
```
sportec: <N>/10 actions are shots
metrica: <N>/10 actions are shots
skillcorner: <N>/<M> actions are shots
pff medium_halftime: <K> rows; columns=[...]
```

- [ ] **Step 0.3: Decide R1 mitigation per provider**

For each provider:
- If `n_shots >= 1` → no augmentation needed; per-row gate verifies real GK rows.
- If `n_shots == 0` → record decision in execution log: per-row gate for that provider verifies the all-NaN path only (still a valid regression check).

For PFF: if `medium_halftime.parquet` lacks shots, prepare to use a stub `medium_halftime_with_shot.parquet` companion fixture in Loop 7 (do not regenerate `medium_halftime.parquet`).

- [ ] **Step 0.4: Delete the probe script**

```bash
git restore --staged scripts/_probe_slim_shots.py 2>$null
rm scripts/_probe_slim_shots.py
```

The probe was throwaway. Findings carry forward as notes (write them in this plan checkbox below).

**Findings notes (fill at execution time):**
- sportec shot count: ___
- metrica shot count: ___
- skillcorner shot count: ___
- pff medium_halftime shot count: ___
- decision: ___

---

## Loop 1 — `ActionFrameContext.defending_gk_rows` field + `_resolve_action_frame_context` extension

**Files:**
- Modify: `silly_kicks/tracking/feature_framework.py` (add field to dataclass)
- Modify: `silly_kicks/tracking/utils.py:245-308` (`_resolve_action_frame_context` extension)
- Test: `tests/tracking/test_feature_framework.py` (extend)
- Test: `tests/tracking/test_utils.py` (extend or create) — verify `defending_gk_rows` populated correctly

- [ ] **Step 1.1: Write failing test for the new field**

Add to `tests/tracking/test_feature_framework.py`:

```python
def test_action_frame_context_has_defending_gk_rows_field():
    """ActionFrameContext exposes a defending_gk_rows: pd.DataFrame field per ADR-005 § 3 (kernel extraction)."""
    import dataclasses
    from silly_kicks.tracking.feature_framework import ActionFrameContext

    fields = {f.name for f in dataclasses.fields(ActionFrameContext)}
    assert "defending_gk_rows" in fields
    field_type = next(f for f in dataclasses.fields(ActionFrameContext) if f.name == "defending_gk_rows").type
    # type may be a string under PEP 563 (from __future__ import annotations) — accept both
    assert "DataFrame" in str(field_type)
```

- [ ] **Step 1.2: Run test, verify RED**

```bash
uv run pytest tests/tracking/test_feature_framework.py::test_action_frame_context_has_defending_gk_rows_field -v
```

Expected: FAIL with `assert 'defending_gk_rows' in fields`.

- [ ] **Step 1.3: Add the field**

Edit `silly_kicks/tracking/feature_framework.py`:

```python
@dataclasses.dataclass(frozen=True)
class ActionFrameContext:
    """Linkage + actor/opposite-team/defending-GK frame slices, computed once per call.
    ... (existing docstring extended with one bullet) ...

    Attributes
    ----------
    actions : pd.DataFrame
    pointers : pd.DataFrame
    actor_rows : pd.DataFrame
    opposite_rows_per_action : pd.DataFrame
    defending_gk_rows : pd.DataFrame
        Long-form: one row per (linked action × frame row where
        frame.player_id == action.defending_gk_player_id AND not is_ball).
        Empty when defending_gk_player_id is absent / NaN, action is unlinked,
        or the GK player_id is absent from the linked frame (substitution case).
        Direct construction of ActionFrameContext is not part of the public API;
        always build via silly_kicks.tracking.utils._resolve_action_frame_context.
    """
    actions: pd.DataFrame
    pointers: pd.DataFrame
    actor_rows: pd.DataFrame
    opposite_rows_per_action: pd.DataFrame
    defending_gk_rows: pd.DataFrame
```

- [ ] **Step 1.4: Run test, verify GREEN**

```bash
uv run pytest tests/tracking/test_feature_framework.py::test_action_frame_context_has_defending_gk_rows_field -v
```

Expected: PASS.

- [ ] **Step 1.5: Write failing tests for `_resolve_action_frame_context` GK-rows population**

Add to `tests/tracking/test_utils.py` (or `test_feature_framework.py` if utils tests live there — verify):

```python
import pandas as pd
import numpy as np
import pytest
from silly_kicks.tracking.utils import _resolve_action_frame_context


def _make_minimal_actions_and_frames():
    """Two actions (one with defending_gk_player_id=99, one with NaN);
    one frame row matching player_id=99 on the defending team.
    """
    actions = pd.DataFrame({
        "action_id": [1, 2],
        "period_id": [1, 1],
        "team_id": [10, 10],
        "player_id": [11, 11],
        "time_seconds": [10.0, 12.0],
        "defending_gk_player_id": [99.0, float("nan")],
    })
    frames = pd.DataFrame({
        "period_id": [1, 1, 1, 1],
        "frame_id": [100, 100, 120, 120],
        "time_seconds": [10.0, 10.0, 12.0, 12.0],
        "team_id": [20, 20, 20, 20],          # opposite team
        "player_id": [99, 50, 99, 50],
        "x": [104.0, 50.0, 103.0, 51.0],
        "y": [34.0, 30.0, 34.5, 30.5],
        "is_ball": [False, False, False, False],
        "speed": [0.5, 4.0, 0.6, 5.0],
        "speed_source": ["native", "native", "native", "native"],
        "frame_rate": [25.0, 25.0, 25.0, 25.0],
        "team_attacking_direction": ["ltr", "ltr", "ltr", "ltr"],
        "source_provider": ["sportec"] * 4,
        # other TRACKING_FRAMES_COLUMNS columns set to defaults — fill from schema.TRACKING_FRAMES_COLUMNS
    })
    return actions, frames


def test_resolve_action_frame_context_populates_defending_gk_rows_when_player_id_matches():
    actions, frames = _make_minimal_actions_and_frames()
    ctx = _resolve_action_frame_context(actions, frames)
    # Action 1 has defending_gk_player_id=99 and the linked frame has a player_id=99 row → populated.
    gk_for_action_1 = ctx.defending_gk_rows[ctx.defending_gk_rows["action_id"] == 1]
    assert len(gk_for_action_1) == 1
    assert float(gk_for_action_1["x"].iloc[0]) == pytest.approx(104.0)
    assert float(gk_for_action_1["y"].iloc[0]) == pytest.approx(34.0)


def test_resolve_action_frame_context_excludes_action_with_nan_defending_gk_player_id():
    actions, frames = _make_minimal_actions_and_frames()
    ctx = _resolve_action_frame_context(actions, frames)
    # Action 2 has NaN defending_gk_player_id → no row in defending_gk_rows.
    gk_for_action_2 = ctx.defending_gk_rows[ctx.defending_gk_rows["action_id"] == 2]
    assert len(gk_for_action_2) == 0


def test_resolve_action_frame_context_defending_gk_rows_empty_when_column_absent():
    """Backward-compat: PR-S20 callers passing actions without defending_gk_player_id get empty."""
    actions, frames = _make_minimal_actions_and_frames()
    actions = actions.drop(columns=["defending_gk_player_id"])
    ctx = _resolve_action_frame_context(actions, frames)
    assert len(ctx.defending_gk_rows) == 0


def test_resolve_action_frame_context_excludes_ball_rows_from_defending_gk_rows():
    actions, frames = _make_minimal_actions_and_frames()
    ball_frame = pd.DataFrame({
        **{c: [frames.iloc[0][c]] for c in frames.columns if c not in ("player_id", "is_ball", "x", "y")},
        "player_id": [pd.NA],
        "is_ball": [True],
        "x": [50.0],
        "y": [34.0],
    })
    frames = pd.concat([frames, ball_frame], ignore_index=True)
    ctx = _resolve_action_frame_context(actions, frames)
    # Even if ball frame's player_id were 99 (it isn't), is_ball=True excludes it.
    assert (~ctx.defending_gk_rows["is_ball"]).all() if len(ctx.defending_gk_rows) else True
```

(Note: `_make_minimal_actions_and_frames` will need to be filled out with all `TRACKING_FRAMES_COLUMNS`; check `silly_kicks.tracking.schema.TRACKING_FRAMES_COLUMNS` for the full list and add missing columns with defaults.)

- [ ] **Step 1.6: Run tests, verify RED**

```bash
uv run pytest tests/tracking/test_utils.py -k "defending_gk_rows or resolve_action_frame_context" -v
```

Expected: FAIL — `defending_gk_rows` field exists (Step 1.3) but is not populated (still empty by old code path).

- [ ] **Step 1.7: Extend `_resolve_action_frame_context`**

Edit `silly_kicks/tracking/utils.py:245-308`. Two changes:

1. Widen `actions_with_period` projection to include `defending_gk_player_id` if present:

```python
# Was:
actions_with_period = actions[["action_id", "period_id", "team_id", "player_id"]]
# Replace with:
projection_cols = ["action_id", "period_id", "team_id", "player_id"]
if "defending_gk_player_id" in actions.columns:
    projection_cols.append("defending_gk_player_id")
actions_with_period = actions[projection_cols]
```

2. After existing `opposite` slice, compute `defending_gk_rows`:

```python
# After existing opposite_mask / opposite computation:
if "defending_gk_player_id" in long.columns and "player_id_frame" in long.columns:
    # Cast both sides to float64 for safe NaN comparison (R3).
    gk_id = long["defending_gk_player_id"].astype("float64")
    pid_frame = long["player_id_frame"].astype("float64")
    gk_mask = (
        (pid_frame == gk_id)
        & gk_id.notna()
        & (~long["is_ball"])
    )
    defending_gk_rows = long.loc[gk_mask].copy()
else:
    defending_gk_rows = long.iloc[0:0].copy()
```

3. Pass through to the constructor:

```python
return ActionFrameContext(
    actions=actions,
    pointers=pointers,
    actor_rows=actor_rows,
    opposite_rows_per_action=opposite,
    defending_gk_rows=defending_gk_rows,
)
```

- [ ] **Step 1.8: Run tests, verify GREEN**

```bash
uv run pytest tests/tracking/test_feature_framework.py tests/tracking/test_utils.py -v
```

Expected: ALL PASS, including PR-S20's existing `_resolve_action_frame_context` tests (no regression).

- [ ] **Step 1.9: Run the full PR-S20 tracking test suite — regression check**

```bash
uv run pytest tests/tracking/ tests/atomic/tracking/ -m "not e2e" -v
```

Expected: ALL PASS. The new field is additive; PR-S20's `add_action_context` doesn't read it.

---

## Loop 2 — `_pre_shot_gk_position` kernel

**Files:**
- Modify: `silly_kicks/tracking/_kernels.py` (add private kernel)
- Test: `tests/tracking/test_kernels.py` (extend)

- [ ] **Step 2.1: Write 6 analytical kernel tests (RED)**

Add to `tests/tracking/test_kernels.py`:

```python
import pandas as pd
import numpy as np
import pytest
from silly_kicks.tracking import _kernels
from silly_kicks.tracking.feature_framework import ActionFrameContext
from silly_kicks.spadl import config as spadlconfig


SHOT_IDS_STANDARD = frozenset(
    spadlconfig.actiontype_id[n] for n in ("shot", "shot_freekick", "shot_penalty")
)


def _stub_ctx(actions, defending_gk_rows):
    """Helper to construct a minimal ActionFrameContext for kernel tests."""
    return ActionFrameContext(
        actions=actions,
        pointers=pd.DataFrame({"action_id": actions["action_id"]}),
        actor_rows=pd.DataFrame({"action_id": actions["action_id"]}),
        opposite_rows_per_action=pd.DataFrame(),
        defending_gk_rows=defending_gk_rows,
    )


def test_pre_shot_gk_position_shot_with_gk_in_frame_emits_exact_values():
    actions = pd.DataFrame({
        "action_id": [1],
        "type_id": [spadlconfig.actiontype_id["shot"]],
        "start_x": [90.0],
        "start_y": [34.0],
        "defending_gk_player_id": [99.0],
    })
    gk_rows = pd.DataFrame({"action_id": [1], "x": [104.0], "y": [34.0]})
    ctx = _stub_ctx(actions, gk_rows)
    out = _kernels._pre_shot_gk_position(
        actions["start_x"], actions["start_y"], ctx, shot_type_ids=SHOT_IDS_STANDARD,
    )
    assert float(out["pre_shot_gk_x"].iloc[0]) == pytest.approx(104.0)
    assert float(out["pre_shot_gk_y"].iloc[0]) == pytest.approx(34.0)
    assert float(out["pre_shot_gk_distance_to_goal"].iloc[0]) == pytest.approx(1.0)
    # distance to (90, 34) → sqrt((104-90)^2 + 0) = 14.0
    assert float(out["pre_shot_gk_distance_to_shot"].iloc[0]) == pytest.approx(14.0)


def test_pre_shot_gk_position_non_shot_row_emits_all_nan():
    actions = pd.DataFrame({
        "action_id": [1],
        "type_id": [spadlconfig.actiontype_id["pass"]],   # NOT a shot
        "start_x": [50.0],
        "start_y": [34.0],
        "defending_gk_player_id": [99.0],
    })
    gk_rows = pd.DataFrame({"action_id": [1], "x": [104.0], "y": [34.0]})
    ctx = _stub_ctx(actions, gk_rows)
    out = _kernels._pre_shot_gk_position(
        actions["start_x"], actions["start_y"], ctx, shot_type_ids=SHOT_IDS_STANDARD,
    )
    assert pd.isna(out["pre_shot_gk_x"].iloc[0])
    assert pd.isna(out["pre_shot_gk_y"].iloc[0])
    assert pd.isna(out["pre_shot_gk_distance_to_goal"].iloc[0])
    assert pd.isna(out["pre_shot_gk_distance_to_shot"].iloc[0])


def test_pre_shot_gk_position_shot_with_nan_defending_gk_player_id_emits_all_nan():
    actions = pd.DataFrame({
        "action_id": [1],
        "type_id": [spadlconfig.actiontype_id["shot"]],
        "start_x": [90.0],
        "start_y": [34.0],
        "defending_gk_player_id": [float("nan")],
    })
    gk_rows = pd.DataFrame({"action_id": [], "x": [], "y": []})  # empty — no GK row
    ctx = _stub_ctx(actions, gk_rows)
    out = _kernels._pre_shot_gk_position(
        actions["start_x"], actions["start_y"], ctx, shot_type_ids=SHOT_IDS_STANDARD,
    )
    assert pd.isna(out["pre_shot_gk_x"].iloc[0])


def test_pre_shot_gk_position_shot_with_gk_absent_from_frame_emits_all_nan():
    """Substitution case: defending_gk_player_id is set but GK is no longer in the frame."""
    actions = pd.DataFrame({
        "action_id": [1],
        "type_id": [spadlconfig.actiontype_id["shot"]],
        "start_x": [90.0],
        "start_y": [34.0],
        "defending_gk_player_id": [99.0],
    })
    gk_rows = pd.DataFrame({"action_id": [], "x": [], "y": []})
    ctx = _stub_ctx(actions, gk_rows)
    out = _kernels._pre_shot_gk_position(
        actions["start_x"], actions["start_y"], ctx, shot_type_ids=SHOT_IDS_STANDARD,
    )
    assert pd.isna(out["pre_shot_gk_x"].iloc[0])


def test_pre_shot_gk_position_unlinked_action_emits_all_nan():
    """When ctx.defending_gk_rows is empty for an action_id (the linkage missed), output is NaN."""
    actions = pd.DataFrame({
        "action_id": [1, 2],
        "type_id": [spadlconfig.actiontype_id["shot"]] * 2,
        "start_x": [90.0, 95.0],
        "start_y": [34.0, 34.0],
        "defending_gk_player_id": [99.0, 99.0],
    })
    # GK row only for action 1; action 2 unlinked.
    gk_rows = pd.DataFrame({"action_id": [1], "x": [104.0], "y": [34.0]})
    ctx = _stub_ctx(actions, gk_rows)
    out = _kernels._pre_shot_gk_position(
        actions["start_x"], actions["start_y"], ctx, shot_type_ids=SHOT_IDS_STANDARD,
    )
    assert not pd.isna(out["pre_shot_gk_x"].iloc[0])
    assert pd.isna(out["pre_shot_gk_x"].iloc[1])


def test_pre_shot_gk_position_off_pitch_gk_passes_through_no_clamping():
    """Memory: feedback_lakehouse_consumer_not_source — raw provider values, no clamping."""
    actions = pd.DataFrame({
        "action_id": [1],
        "type_id": [spadlconfig.actiontype_id["shot"]],
        "start_x": [95.0],
        "start_y": [34.0],
        "defending_gk_player_id": [99.0],
    })
    # GK at x=107 (off pitch by 2m); y=-1 (off pitch by 1m).
    gk_rows = pd.DataFrame({"action_id": [1], "x": [107.0], "y": [-1.0]})
    ctx = _stub_ctx(actions, gk_rows)
    out = _kernels._pre_shot_gk_position(
        actions["start_x"], actions["start_y"], ctx, shot_type_ids=SHOT_IDS_STANDARD,
    )
    assert float(out["pre_shot_gk_x"].iloc[0]) == pytest.approx(107.0)   # NOT clamped
    assert float(out["pre_shot_gk_y"].iloc[0]) == pytest.approx(-1.0)    # NOT clamped
```

- [ ] **Step 2.2: Run tests, verify RED**

```bash
uv run pytest tests/tracking/test_kernels.py -k "pre_shot_gk_position" -v
```

Expected: 6 FAILS — `AttributeError: module ... has no attribute '_pre_shot_gk_position'`.

- [ ] **Step 2.3: Implement the kernel**

Add to `silly_kicks/tracking/_kernels.py`:

```python
def _pre_shot_gk_position(
    anchor_x: pd.Series,
    anchor_y: pd.Series,
    ctx: "ActionFrameContext",   # forward-ref to avoid circular import
    *,
    shot_type_ids: frozenset[int],
) -> pd.DataFrame:
    """Compute 4 GK-position features per action: x, y, distance-to-goal, distance-to-shot.

    Schema-agnostic per ADR-005 § 3 — caller supplies anchor columns and shot type ids.

    Returns
    -------
    pd.DataFrame
        Indexed identically to ctx.actions (preserves order).
        Columns: pre_shot_gk_x, pre_shot_gk_y,
                 pre_shot_gk_distance_to_goal, pre_shot_gk_distance_to_shot.
        All NaN for: non-shot rows / unlinked rows / pre-engagement rows
                     (defending_gk_player_id NaN) / GK-absent-from-frame rows.

    See NOTICE for full bibliographic citations (Anzer & Bauer 2021).
    """
    actions = ctx.actions
    out = pd.DataFrame(
        {
            "pre_shot_gk_x": np.nan,
            "pre_shot_gk_y": np.nan,
            "pre_shot_gk_distance_to_goal": np.nan,
            "pre_shot_gk_distance_to_shot": np.nan,
        },
        index=actions.index,
    )
    if "type_id" not in actions.columns:
        return out
    is_shot = actions["type_id"].isin(shot_type_ids).to_numpy()
    if not is_shot.any():
        return out

    # Left-join GK x/y on action_id; non-shots / pre-engagement / GK-absent rows have NaN x/y.
    if len(ctx.defending_gk_rows) > 0:
        gk = ctx.defending_gk_rows[["action_id", "x", "y"]].rename(
            columns={"x": "_gk_x", "y": "_gk_y"},
        )
        # First match per action_id (guard against accidental duplicates).
        gk = gk.drop_duplicates("action_id", keep="first")
        per_action = actions[["action_id"]].merge(gk, on="action_id", how="left")
    else:
        per_action = actions[["action_id"]].assign(_gk_x=np.nan, _gk_y=np.nan)
    per_action = per_action.set_index(actions.index)

    # Mask: shot AND GK present in linked frame (gk_x not NaN).
    shot_mask = pd.Series(is_shot, index=actions.index)
    gk_present_mask = per_action["_gk_x"].notna()
    valid = shot_mask & gk_present_mask

    out.loc[valid, "pre_shot_gk_x"] = per_action.loc[valid, "_gk_x"].astype("float64")
    out.loc[valid, "pre_shot_gk_y"] = per_action.loc[valid, "_gk_y"].astype("float64")
    out.loc[valid, "pre_shot_gk_distance_to_goal"] = np.sqrt(
        (105.0 - per_action.loc[valid, "_gk_x"]) ** 2
        + (34.0 - per_action.loc[valid, "_gk_y"]) ** 2
    ).astype("float64")
    out.loc[valid, "pre_shot_gk_distance_to_shot"] = np.sqrt(
        (anchor_x.loc[valid].astype("float64") - per_action.loc[valid, "_gk_x"]) ** 2
        + (anchor_y.loc[valid].astype("float64") - per_action.loc[valid, "_gk_y"]) ** 2
    ).astype("float64")
    return out
```

- [ ] **Step 2.4: Run tests, verify GREEN**

```bash
uv run pytest tests/tracking/test_kernels.py -k "pre_shot_gk_position" -v
```

Expected: 6 PASS.

- [ ] **Step 2.5: Run full kernels test suite — regression check**

```bash
uv run pytest tests/tracking/test_kernels.py -v
```

Expected: ALL PASS (PR-S20 kernel tests + 6 new GK kernel tests).

---

## Loop 3 — Tracking-namespace standard SPADL public surface

**Files:**
- Modify: `silly_kicks/tracking/features.py` (4 Series helpers + aggregator + xfn list)
- Test: `tests/tracking/test_features_standard.py` (extend)
- Test: `tests/tracking/test_add_pre_shot_gk_position.py` (new)

- [ ] **Step 3.1: Write failing tests for 4 Series helpers**

Add to `tests/tracking/test_features_standard.py`:

```python
import pytest
import pandas as pd
from silly_kicks.tracking import features as track_features
from tests.tracking._fixtures_action_context import (   # PR-S20 helper, reuse
    minimal_shot_actions_with_frames_and_gk,
)


@pytest.mark.parametrize("helper_name", [
    "pre_shot_gk_x",
    "pre_shot_gk_y",
    "pre_shot_gk_distance_to_goal",
    "pre_shot_gk_distance_to_shot",
])
def test_pre_shot_gk_helpers_return_named_series(helper_name):
    actions, frames = minimal_shot_actions_with_frames_and_gk()
    helper = getattr(track_features, helper_name)
    s = helper(actions, frames)
    assert isinstance(s, pd.Series)
    assert s.name == helper_name
    assert len(s) == len(actions)


def test_pre_shot_gk_x_finite_for_shot_with_linked_gk():
    actions, frames = minimal_shot_actions_with_frames_and_gk()
    s = track_features.pre_shot_gk_x(actions, frames)
    # First action is a shot with linked GK at known x — assert finite.
    assert s.iloc[0] == pytest.approx(actions["_expected_gk_x"].iloc[0])


def test_pre_shot_gk_distance_to_shot_uses_start_anchor_for_standard_spadl():
    actions, frames = minimal_shot_actions_with_frames_and_gk()
    expected = (
        (actions["start_x"].iloc[0] - actions["_expected_gk_x"].iloc[0]) ** 2
        + (actions["start_y"].iloc[0] - actions["_expected_gk_y"].iloc[0]) ** 2
    ) ** 0.5
    s = track_features.pre_shot_gk_distance_to_shot(actions, frames)
    assert float(s.iloc[0]) == pytest.approx(expected)
```

`tests/tracking/_fixtures_action_context.py` is a PR-S20 helper module (verify exists; if not, create it). Add `minimal_shot_actions_with_frames_and_gk` factory:

```python
# tests/tracking/_fixtures_action_context.py — extend
def minimal_shot_actions_with_frames_and_gk():
    """Two shot rows; one linked frame with the defending GK at known x/y.
    Returns (actions, frames) plus _expected_gk_{x,y} columns on actions for assertion convenience.
    """
    # ... fill in (use existing PR-S20 fixture as template, add defending_gk_player_id col + matching frame row)
```

- [ ] **Step 3.2: Run tests, verify RED**

```bash
uv run pytest tests/tracking/test_features_standard.py -k "pre_shot_gk" -v
```

Expected: FAIL — `AttributeError: module ... has no attribute 'pre_shot_gk_x'`.

- [ ] **Step 3.3: Implement 4 Series helpers**

Edit `silly_kicks/tracking/features.py`. Add after PR-S20's existing helpers, before `tracking_default_xfns`:

```python
def pre_shot_gk_x(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Defending GK's x at the linked frame, m, LTR-normalized.

    NaN for non-shot rows, unlinked actions, pre-engagement (NaN
    defending_gk_player_id), or GK-absent-from-frame (substitution) cases.

    REQUIRES the actions DataFrame to have a `defending_gk_player_id` column
    (run silly_kicks.spadl.utils.add_pre_shot_gk_context first).

    See NOTICE for full bibliographic citations.

    Examples
    --------
    Compute defending-GK x for a SPADL action stream after engagement-state enrichment::

        from silly_kicks.spadl.utils import add_pre_shot_gk_context
        from silly_kicks.tracking.features import pre_shot_gk_x
        actions = add_pre_shot_gk_context(actions)
        gk_x = pre_shot_gk_x(actions, frames)

    References
    ----------
    Anzer, G., & Bauer, P. (2021). "A goal scoring probability model for shots based on
        synchronized positional and event data in football and futsal." Frontiers in
        Sports and Active Living, 3, 624475. (defending-GK-position as xG feature)
    """
    ctx = _resolve_action_frame_context(actions, frames)
    df = _kernels._pre_shot_gk_position(
        actions["start_x"], actions["start_y"], ctx, shot_type_ids=_STANDARD_SHOT_TYPE_IDS,
    )
    return df["pre_shot_gk_x"].rename("pre_shot_gk_x")


def pre_shot_gk_y(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Defending GK's y at the linked frame, m, LTR-normalized.

    [Same NaN/REQUIRES contract as pre_shot_gk_x.]

    See NOTICE for full bibliographic citations.

    Examples
    --------
    ::

        from silly_kicks.spadl.utils import add_pre_shot_gk_context
        from silly_kicks.tracking.features import pre_shot_gk_y
        actions = add_pre_shot_gk_context(actions)
        gk_y = pre_shot_gk_y(actions, frames)

    References
    ----------
    Anzer, G., & Bauer, P. (2021). "A goal scoring probability model for shots based on
        synchronized positional and event data in football and futsal." Frontiers in
        Sports and Active Living, 3, 624475.
    """
    ctx = _resolve_action_frame_context(actions, frames)
    df = _kernels._pre_shot_gk_position(
        actions["start_x"], actions["start_y"], ctx, shot_type_ids=_STANDARD_SHOT_TYPE_IDS,
    )
    return df["pre_shot_gk_y"].rename("pre_shot_gk_y")


def pre_shot_gk_distance_to_goal(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Euclidean distance (m) from defending GK to goal-mouth center (105, 34).

    [Same NaN/REQUIRES contract as pre_shot_gk_x.]

    See NOTICE for full bibliographic citations.

    Examples
    --------
    ::

        from silly_kicks.spadl.utils import add_pre_shot_gk_context
        from silly_kicks.tracking.features import pre_shot_gk_distance_to_goal
        actions = add_pre_shot_gk_context(actions)
        d = pre_shot_gk_distance_to_goal(actions, frames)

    References
    ----------
    Anzer, G., & Bauer, P. (2021). (defending-GK-position as xG feature)
    """
    ctx = _resolve_action_frame_context(actions, frames)
    df = _kernels._pre_shot_gk_position(
        actions["start_x"], actions["start_y"], ctx, shot_type_ids=_STANDARD_SHOT_TYPE_IDS,
    )
    return df["pre_shot_gk_distance_to_goal"].rename("pre_shot_gk_distance_to_goal")


def pre_shot_gk_distance_to_shot(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Euclidean distance (m) from defending GK to shot anchor (action.start_x, action.start_y).

    [Same NaN/REQUIRES contract as pre_shot_gk_x.]

    See NOTICE for full bibliographic citations.

    Examples
    --------
    ::

        from silly_kicks.spadl.utils import add_pre_shot_gk_context
        from silly_kicks.tracking.features import pre_shot_gk_distance_to_shot
        actions = add_pre_shot_gk_context(actions)
        d = pre_shot_gk_distance_to_shot(actions, frames)

    References
    ----------
    Anzer, G., & Bauer, P. (2021).
    """
    ctx = _resolve_action_frame_context(actions, frames)
    df = _kernels._pre_shot_gk_position(
        actions["start_x"], actions["start_y"], ctx, shot_type_ids=_STANDARD_SHOT_TYPE_IDS,
    )
    return df["pre_shot_gk_distance_to_shot"].rename("pre_shot_gk_distance_to_shot")
```

Add the shot-type-id constant near the top of `features.py`:

```python
from silly_kicks.spadl import config as spadlconfig

_STANDARD_SHOT_TYPE_IDS = frozenset(
    spadlconfig.actiontype_id[n] for n in ("shot", "shot_freekick", "shot_penalty")
)
```

Update `__all__` to include the 4 new helpers.

- [ ] **Step 3.4: Run Series tests, verify GREEN**

```bash
uv run pytest tests/tracking/test_features_standard.py -k "pre_shot_gk" -v
```

Expected: PASS.

- [ ] **Step 3.5: Write failing tests for `add_pre_shot_gk_position` aggregator**

Create `tests/tracking/test_add_pre_shot_gk_position.py`:

```python
import pytest
import pandas as pd
from silly_kicks.tracking.features import add_pre_shot_gk_position
from tests.tracking._fixtures_action_context import minimal_shot_actions_with_frames_and_gk


def test_add_pre_shot_gk_position_emits_4_features_plus_4_provenance():
    actions, frames = minimal_shot_actions_with_frames_and_gk()
    out = add_pre_shot_gk_position(actions, frames)
    expected_new_cols = {
        "pre_shot_gk_x",
        "pre_shot_gk_y",
        "pre_shot_gk_distance_to_goal",
        "pre_shot_gk_distance_to_shot",
        "frame_id",
        "time_offset_seconds",
        "n_candidate_frames",
        "link_quality_score",
    }
    assert expected_new_cols.issubset(set(out.columns))


def test_add_pre_shot_gk_position_raises_on_missing_defending_gk_player_id_column():
    actions, frames = minimal_shot_actions_with_frames_and_gk()
    actions = actions.drop(columns=["defending_gk_player_id"])
    with pytest.raises(ValueError, match="defending_gk_player_id"):
        add_pre_shot_gk_position(actions, frames)


def test_add_pre_shot_gk_position_idempotent_when_called_twice():
    actions, frames = minimal_shot_actions_with_frames_and_gk()
    out1 = add_pre_shot_gk_position(actions, frames)
    # Drop the new columns from out1 to recreate input shape, then re-call.
    new_cols = [
        "pre_shot_gk_x", "pre_shot_gk_y",
        "pre_shot_gk_distance_to_goal", "pre_shot_gk_distance_to_shot",
        "frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score",
    ]
    out2 = add_pre_shot_gk_position(out1.drop(columns=new_cols), frames)
    pd.testing.assert_frame_equal(out1, out2)


def test_add_pre_shot_gk_position_provenance_columns_match_link_actions_to_frames():
    from silly_kicks.tracking.utils import link_actions_to_frames
    actions, frames = minimal_shot_actions_with_frames_and_gk()
    out = add_pre_shot_gk_position(actions, frames)
    pointers, _ = link_actions_to_frames(actions, frames)
    pd.testing.assert_series_equal(
        out["frame_id"].reset_index(drop=True),
        pointers.set_index("action_id").loc[out["action_id"], "frame_id"].reset_index(drop=True),
        check_names=False,
    )
```

The NaN-safe-per-ADR-003 test is auto-covered via the `@nan_safe_enrichment` decorator + `tests/test_enrichment_nan_safety.py` auto-discovery (verified in Step 3.7).

- [ ] **Step 3.6: Run tests, verify RED**

```bash
uv run pytest tests/tracking/test_add_pre_shot_gk_position.py -v
```

Expected: FAIL — `ImportError: cannot import name 'add_pre_shot_gk_position'`.

- [ ] **Step 3.7: Implement `add_pre_shot_gk_position` aggregator + lifted xfns**

Edit `silly_kicks/tracking/features.py`. Add after the 4 Series helpers:

```python
@nan_safe_enrichment
def add_pre_shot_gk_position(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
) -> pd.DataFrame:
    """Enrich actions with 4 GK-position columns + 4 linkage-provenance columns.

    REQUIRES the actions DataFrame to have a `defending_gk_player_id` column
    (run silly_kicks.spadl.utils.add_pre_shot_gk_context first).

    Returns
    -------
    pd.DataFrame
        Input actions with the columns:
        - pre_shot_gk_x (float64, m)
        - pre_shot_gk_y (float64, m)
        - pre_shot_gk_distance_to_goal (float64, m)
        - pre_shot_gk_distance_to_shot (float64, m)
        - frame_id (Int64; NaN if unlinked)
        - time_offset_seconds (float64; NaN if unlinked)
        - link_quality_score (float64; NaN if unlinked)
        - n_candidate_frames (int64)

    All 4 GK columns are NaN for non-shot / unlinked / pre-engagement /
    GK-absent-from-frame rows.

    Raises
    ------
    ValueError
        If `defending_gk_player_id` column is absent from actions.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    Tag pre-shot GK position via tracking-namespace canonical compute::

        from silly_kicks.spadl.utils import add_pre_shot_gk_context
        from silly_kicks.tracking.features import add_pre_shot_gk_position
        actions = add_pre_shot_gk_context(actions)            # populates defending_gk_player_id
        enriched = add_pre_shot_gk_position(actions, frames)  # adds 4 GK + 4 provenance columns
    """
    if "defending_gk_player_id" not in actions.columns:
        raise ValueError(
            "add_pre_shot_gk_position: actions missing required column "
            "'defending_gk_player_id'. Run silly_kicks.spadl.utils.add_pre_shot_gk_context "
            "first to populate it."
        )
    ctx = _resolve_action_frame_context(actions, frames)
    df = _kernels._pre_shot_gk_position(
        actions["start_x"], actions["start_y"], ctx, shot_type_ids=_STANDARD_SHOT_TYPE_IDS,
    )
    out = actions.copy()
    for col in df.columns:
        out[col] = df[col]
    pointer_cols = ctx.pointers.set_index("action_id")[
        ["frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]
    ]
    out = out.merge(pointer_cols, left_on="action_id", right_index=True, how="left")
    return out


pre_shot_gk_default_xfns = [
    lift_to_states(pre_shot_gk_x),
    lift_to_states(pre_shot_gk_y),
    lift_to_states(pre_shot_gk_distance_to_goal),
    lift_to_states(pre_shot_gk_distance_to_shot),
]
```

Update `__all__` to include `add_pre_shot_gk_position` and `pre_shot_gk_default_xfns`.

- [ ] **Step 3.8: Run tests, verify GREEN**

```bash
uv run pytest tests/tracking/test_add_pre_shot_gk_position.py tests/tracking/test_features_standard.py -v
```

Expected: ALL PASS.

- [ ] **Step 3.9: Verify auto-discovery of NaN-safety test**

```bash
uv run pytest tests/test_enrichment_nan_safety.py -v
```

Expected: PASS — `add_pre_shot_gk_position` auto-discovered via `@nan_safe_enrichment` registry; the existing test parametrizes over all decorated helpers.

---

## Loop 4 — Atomic-namespace mirror

**Files:**
- Create: `silly_kicks/atomic/tracking/features.py` (extend if exists, else create — verify in plan)
- Test: `tests/atomic/tracking/test_features_atomic.py` (extend)
- Test: `tests/atomic/tracking/test_add_atomic_pre_shot_gk_position.py` (new)

- [ ] **Step 4.1: Verify atomic tracking module already exists from PR-S20**

```bash
uv run python -c "from silly_kicks.atomic.tracking import features; print(dir(features))"
```

Expected: lists existing PR-S20 atomic surface (nearest_defender_distance, etc.). If module missing, create it following PR-S20 atomic mirror pattern.

- [ ] **Step 4.2: Write failing parametrized parity tests**

Add to `tests/atomic/tracking/test_features_atomic.py`:

```python
import pytest
import pandas as pd

from silly_kicks.atomic.tracking import features as atomic_track_features
from silly_kicks.tracking import features as track_features


@pytest.mark.parametrize("helper_name", [
    "pre_shot_gk_x",
    "pre_shot_gk_y",
    "pre_shot_gk_distance_to_goal",
    "pre_shot_gk_distance_to_shot",
])
def test_atomic_pre_shot_gk_helpers_exist_and_return_named_series(helper_name):
    helper = getattr(atomic_track_features, helper_name)
    # We won't run on a fixture here — just verify it's a callable with right signature.
    assert callable(helper)


def test_atomic_pre_shot_gk_distance_to_shot_anchors_on_atomic_x_y():
    """Atomic anchors on action.x/y; standard anchors on action.start_x/start_y.
    Same kernel; different schema wrappers.
    """
    from tests.atomic.tracking._fixtures import minimal_atomic_shot_with_gk_frame
    actions, frames = minimal_atomic_shot_with_gk_frame()
    s = atomic_track_features.pre_shot_gk_distance_to_shot(actions, frames)
    expected = (
        (actions["x"].iloc[0] - actions["_expected_gk_x"].iloc[0]) ** 2
        + (actions["y"].iloc[0] - actions["_expected_gk_y"].iloc[0]) ** 2
    ) ** 0.5
    assert float(s.iloc[0]) == pytest.approx(expected)
```

(Create `tests/atomic/tracking/_fixtures.py` if absent; mirror the standard-side fixture but with atomic columns `x`, `y`, `dx`, `dy` instead of `start_x`, `start_y`, `end_x`, `end_y`.)

- [ ] **Step 4.3: Run tests, verify RED**

```bash
uv run pytest tests/atomic/tracking/test_features_atomic.py -k "pre_shot_gk" -v
```

Expected: FAIL — `AttributeError`.

- [ ] **Step 4.4: Implement atomic mirror (4 Series helpers + aggregator + xfn list)**

Edit `silly_kicks/atomic/tracking/features.py`. Pattern is **identical to standard SPADL** (Loop 3) with TWO exceptions:

1. Anchor columns: use `actions["x"]` and `actions["y"]` instead of `actions["start_x"]` / `actions["start_y"]`.
2. Shot type ids: atomic recognizes only `{"shot", "shot_penalty"}` (not `shot_freekick`).

Concretely, define the constant:

```python
_ATOMIC_SHOT_TYPE_IDS = frozenset(
    spadlconfig.actiontype_id[n] for n in ("shot", "shot_penalty")
)
```

Then every `pre_shot_gk_*` helper and `add_pre_shot_gk_position` mirrors the standard version with `actions["x"], actions["y"]` and `_ATOMIC_SHOT_TYPE_IDS` substituted. Each docstring's Examples block uses the atomic import path. The aggregator's `ValueError` message is identical (caller still runs `silly_kicks.atomic.spadl.utils.add_pre_shot_gk_context` first to populate `defending_gk_player_id` on atomic actions).

Add to `__all__`: `pre_shot_gk_x`, `pre_shot_gk_y`, `pre_shot_gk_distance_to_goal`, `pre_shot_gk_distance_to_shot`, `add_pre_shot_gk_position`, `pre_shot_gk_default_xfns`.

- [ ] **Step 4.5: Write failing tests for atomic aggregator**

Create `tests/atomic/tracking/test_add_atomic_pre_shot_gk_position.py`:

```python
import pytest
import pandas as pd
from silly_kicks.atomic.tracking.features import add_pre_shot_gk_position
from tests.atomic.tracking._fixtures import minimal_atomic_shot_with_gk_frame


def test_atomic_add_pre_shot_gk_position_emits_4_features_plus_4_provenance():
    actions, frames = minimal_atomic_shot_with_gk_frame()
    out = add_pre_shot_gk_position(actions, frames)
    expected = {
        "pre_shot_gk_x", "pre_shot_gk_y",
        "pre_shot_gk_distance_to_goal", "pre_shot_gk_distance_to_shot",
        "frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score",
    }
    assert expected.issubset(set(out.columns))


def test_atomic_add_pre_shot_gk_position_raises_on_missing_defending_gk_player_id_column():
    actions, frames = minimal_atomic_shot_with_gk_frame()
    actions = actions.drop(columns=["defending_gk_player_id"])
    with pytest.raises(ValueError, match="defending_gk_player_id"):
        add_pre_shot_gk_position(actions, frames)
```

- [ ] **Step 4.6: Run atomic tests, verify GREEN**

```bash
uv run pytest tests/atomic/tracking/ -k "pre_shot_gk" -v
```

Expected: ALL PASS.

- [ ] **Step 4.7: Run full atomic regression check**

```bash
uv run pytest tests/atomic/ -m "not e2e" -v
```

Expected: ALL PASS — no regression on PR-S20 atomic features.

---

## Loop 5 — Events-side wrapper extension (standard + atomic)

**Files:**
- Modify: `silly_kicks/spadl/utils.py:486-654` (`add_pre_shot_gk_context`)
- Modify: `silly_kicks/atomic/spadl/utils.py:761-916` (atomic `add_pre_shot_gk_context`)
- Test: `tests/spadl/test_add_pre_shot_gk_context.py` (extend)
- Test: `tests/atomic/test_atomic_add_pre_shot_gk_context.py` (extend)
- Create: `tests/spadl/_golden_pre_shot_gk_context_v280.parquet` (golden fixture for backward-compat)

- [ ] **Step 5.1: Generate the v2.8.0 golden fixture (BEFORE editing the helper)**

Important: this MUST run on un-modified `add_pre_shot_gk_context` to capture v2.8.0 behavior.

```python
# scripts/_generate_golden_v280_pre_shot_gk_context.py — temporary; deleted after generation
import pandas as pd
from silly_kicks.spadl.utils import add_pre_shot_gk_context

# Build a small but representative actions DataFrame: 8 rows including 1 shot,
# 1 keeper_save, several passes, mixed teams, mixed periods.
# (Use existing tests/spadl/_gk_test_fixtures.py helpers if available.)
from tests.spadl._gk_test_fixtures import build_minimal_pre_shot_actions
actions = build_minimal_pre_shot_actions()
out = add_pre_shot_gk_context(actions)
out.to_parquet("tests/spadl/_golden_pre_shot_gk_context_v280.parquet", index=False)
print(f"Golden fixture written: {len(out)} rows, {len(out.columns)} cols")
```

```bash
uv run python scripts/_generate_golden_v280_pre_shot_gk_context.py
git add tests/spadl/_golden_pre_shot_gk_context_v280.parquet
rm scripts/_generate_golden_v280_pre_shot_gk_context.py
```

Same procedure for atomic:

```python
# scripts/_generate_golden_v280_atomic_pre_shot_gk_context.py — temporary
import pandas as pd
from silly_kicks.atomic.spadl.utils import add_pre_shot_gk_context
from tests.atomic._atomic_test_fixtures import build_minimal_atomic_pre_shot_actions
actions = build_minimal_atomic_pre_shot_actions()
out = add_pre_shot_gk_context(actions)
out.to_parquet("tests/atomic/_golden_atomic_pre_shot_gk_context_v280.parquet", index=False)
```

```bash
uv run python scripts/_generate_golden_v280_atomic_pre_shot_gk_context.py
git add tests/atomic/_golden_atomic_pre_shot_gk_context_v280.parquet
rm scripts/_generate_golden_v280_atomic_pre_shot_gk_context.py
```

If `build_minimal_pre_shot_actions` / `build_minimal_atomic_pre_shot_actions` don't exist in the existing fixture files, create them inline before the generator (matching the existing PR-S20-era test fixtures' shape).

- [ ] **Step 5.2: Write failing tests for the extended helper**

Add to `tests/spadl/test_add_pre_shot_gk_context.py`:

```python
import pytest
import pandas as pd
from pathlib import Path
from silly_kicks.spadl.utils import add_pre_shot_gk_context
from tests.spadl._gk_test_fixtures import build_minimal_pre_shot_actions


def test_add_pre_shot_gk_context_frames_none_bit_identical_to_v280():
    """Backward-compat: silly-kicks 2.8.0 behavior pinned by golden fixture."""
    actions = build_minimal_pre_shot_actions()
    actual = add_pre_shot_gk_context(actions)
    expected = pd.read_parquet(
        Path(__file__).parent / "_golden_pre_shot_gk_context_v280.parquet"
    )
    pd.testing.assert_frame_equal(actual, expected, check_dtype=True)


def test_add_pre_shot_gk_context_frames_supplied_emits_4_extra_columns_plus_provenance():
    from tests.tracking._fixtures_action_context import minimal_shot_actions_with_frames_and_gk
    actions_pre, frames = minimal_shot_actions_with_frames_and_gk()
    # Drop the pre-baked defending_gk_player_id; add_pre_shot_gk_context will set it.
    actions = actions_pre.drop(columns=["defending_gk_player_id"], errors="ignore")
    out = add_pre_shot_gk_context(actions, frames=frames)
    expected_extra = {
        "pre_shot_gk_x", "pre_shot_gk_y",
        "pre_shot_gk_distance_to_goal", "pre_shot_gk_distance_to_shot",
        "frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score",
    }
    assert expected_extra.issubset(set(out.columns))


def test_add_pre_shot_gk_context_frames_supplied_no_module_import_cycle():
    """Importing silly_kicks.spadl.utils alone must NOT eagerly import silly_kicks.tracking.*"""
    import sys
    # Wipe tracking modules from sys.modules to test fresh import path.
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("silly_kicks.tracking"):
            del sys.modules[mod_name]
    if "silly_kicks.spadl.utils" in sys.modules:
        del sys.modules["silly_kicks.spadl.utils"]

    import silly_kicks.spadl.utils  # noqa: F401

    # No silly_kicks.tracking submodule should be loaded by the spadl.utils import.
    tracking_loaded = [m for m in sys.modules if m.startswith("silly_kicks.tracking")]
    assert tracking_loaded == [], f"Lazy-import contract broken: {tracking_loaded}"


def test_add_pre_shot_gk_context_with_frames_handles_nan_defending_gk_player_id():
    """ADR-003: NaN defending_gk_player_id (no defending-keeper engagement in window)
    → NaN GK-position columns; helper does not crash."""
    from tests.tracking._fixtures_action_context import minimal_shot_actions_with_frames_and_gk
    actions_pre, frames = minimal_shot_actions_with_frames_and_gk()
    # Force NaN defending_gk_player_id by ensuring no recent defending-keeper actions.
    actions = actions_pre.drop(columns=["defending_gk_player_id"], errors="ignore")
    # Replace any keeper_* type_ids with passes so engagement-state never identifies a GK.
    from silly_kicks.spadl import config as spadlconfig
    pass_id = spadlconfig.actiontype_id["pass"]
    keeper_ids = {spadlconfig.actiontype_id[n] for n in ("keeper_save", "keeper_claim", "keeper_punch", "keeper_pick_up")}
    actions = actions.copy()
    actions.loc[actions["type_id"].isin(keeper_ids), "type_id"] = pass_id

    out = add_pre_shot_gk_context(actions, frames=frames)
    # All defending_gk_player_id NaN → all 4 GK-position cols NaN where there are shots.
    shot_mask = out["type_id"].isin({spadlconfig.actiontype_id[n] for n in ("shot", "shot_freekick", "shot_penalty")})
    if shot_mask.any():
        assert out.loc[shot_mask, "pre_shot_gk_x"].isna().all()
```

(Atomic mirror in `tests/atomic/test_atomic_add_pre_shot_gk_context.py` — same shape with atomic imports.)

- [ ] **Step 5.3: Run tests, verify RED**

```bash
uv run pytest tests/spadl/test_add_pre_shot_gk_context.py tests/atomic/test_atomic_add_pre_shot_gk_context.py -v
```

Expected: FAIL — golden-fixture test already passes (helper unchanged), but `frames=` tests fail because the kwarg doesn't exist yet.

- [ ] **Step 5.4: Extend the standard helper signature + body**

Edit `silly_kicks/spadl/utils.py:486-654`. Two changes:

1. Add `frames: pd.DataFrame | None = None` to keyword-only args.
2. After the existing assignment block (`sorted_actions["gk_was_distributing"] = ...` etc.), if `frames is not None`, lazy-import and merge.

```python
@nan_safe_enrichment
def add_pre_shot_gk_context(
    actions: pd.DataFrame,
    *,
    frames: pd.DataFrame | None = None,    # NEW
    lookback_seconds: float = 10.0,
    lookback_actions: int = 5,
) -> pd.DataFrame:
    """Tag each shot with the defending goalkeeper's recent activity ...

    [extend existing docstring with one paragraph:]

    When ``frames`` is supplied, additionally emits 4 GK-position columns
    (``pre_shot_gk_x``, ``pre_shot_gk_y``, ``pre_shot_gk_distance_to_goal``,
    ``pre_shot_gk_distance_to_shot``) plus 4 linkage-provenance columns
    (``frame_id``, ``time_offset_seconds``, ``link_quality_score``,
    ``n_candidate_frames``) via the ``silly_kicks.tracking.features`` canonical
    compute. When ``frames=None`` (default), behavior is bit-identical to
    silly-kicks 2.8.0 — no frames-related columns appear in the output.

    Parameters
    ----------
    ... (existing) ...
    frames : pd.DataFrame | None, default None
        Long-form tracking frames matching ``TRACKING_FRAMES_COLUMNS``.
        When supplied, enables 4 GK-position + 4 provenance output columns.

    Examples
    --------
    Events-only path (silly-kicks 2.8.0 backward-compat)::

        actions, _ = statsbomb.convert_to_actions(events, home_team_id=100)
        actions = add_pre_shot_gk_context(actions, lookback_seconds=10.0)
        engaged_shots = actions[actions["gk_was_engaged"]]

    Events + tracking path (silly-kicks 2.9.0+)::

        from silly_kicks.tracking import sportec
        frames, _ = sportec.convert_to_frames(raw, home_team_id="DFL-CLU-A", home_team_start_left=True)
        actions = add_pre_shot_gk_context(actions, frames=frames)
        # Now also has pre_shot_gk_x/_y/_distance_to_{goal,shot} columns.

    References
    ----------
    Related work:

    - Butcher et al. (2025), "An Expected Goals On Target (xGOT) Model" (MDPI) — focuses on
      the shot moment; does not surface pre-shot GK engagement state.
    - Anzer, G., & Bauer, P. (2021), "A goal scoring probability model for shots based on
      synchronized positional and event data in football and futsal." Frontiers in Sports
      and Active Living, 3, 624475 — defending-GK position as xG feature; basis of the
      4 GK-position columns when ``frames`` is supplied.
    """
    # ... (existing body, unchanged through `sorted_actions["defending_gk_player_id"] = defending_gk_player_id`) ...

    if frames is not None:
        # Lazy import per ADR-005 § 5 — preserves no-cycle invariant.
        from silly_kicks.tracking.features import add_pre_shot_gk_position
        sorted_actions = add_pre_shot_gk_position(sorted_actions, frames)

    return sorted_actions
```

Add a `TYPE_CHECKING` import at the top of `silly_kicks/spadl/utils.py` for type-checker fidelity:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from silly_kicks.tracking.features import add_pre_shot_gk_position  # noqa: F401  type-only
```

- [ ] **Step 5.5: Mirror the change to atomic**

Edit `silly_kicks/atomic/spadl/utils.py:761-916`. Identical extension shape:

1. Add `frames: pd.DataFrame | None = None` to keyword-only args.
2. After existing body, lazy-import from `silly_kicks.atomic.tracking.features.add_pre_shot_gk_position` and merge.
3. Update docstring.

- [ ] **Step 5.6: Run tests, verify GREEN**

```bash
uv run pytest tests/spadl/test_add_pre_shot_gk_context.py tests/atomic/test_atomic_add_pre_shot_gk_context.py -v
```

Expected: ALL PASS, including the golden-fixture bit-identity test (the frames=None code path is unchanged).

- [ ] **Step 5.7: Run full spadl + atomic regression suite**

```bash
uv run pytest tests/spadl/ tests/atomic/ -m "not e2e" -v
```

Expected: ALL PASS.

---

## Loop 6 — VAEP integration extension

**Files:**
- Test: `tests/vaep/test_compute_features_frames_kwarg.py` (extend)
- Test: `tests/vaep/test_hybrid_with_tracking.py` (extend)

No production-code changes — `pre_shot_gk_default_xfns` plugs into the existing PR-S20 frame-aware xfn dispatch via `lift_to_states`.

- [ ] **Step 6.1: Write failing test for `pre_shot_gk_default_xfns` dispatch**

Add to `tests/vaep/test_compute_features_frames_kwarg.py`:

```python
def test_compute_features_dispatches_pre_shot_gk_default_xfns():
    """xfns=pre_shot_gk_default_xfns + actions with defending_gk_player_id + frames →
    4 columns × nb_states emitted."""
    from silly_kicks.vaep.base import VAEP
    from silly_kicks.tracking.features import pre_shot_gk_default_xfns
    from tests.tracking._fixtures_action_context import minimal_vaep_states_with_gk_and_frames

    games, gamestates, frames = minimal_vaep_states_with_gk_and_frames()
    v = VAEP(xfns=pre_shot_gk_default_xfns)
    feats = v.compute_features(games, gamestates, frames=frames)
    # 4 features × nb_states (default 3) = 12 columns
    expected_cols = {
        f"pre_shot_gk_{name}_a{i}"
        for name in ("x", "y", "distance_to_goal", "distance_to_shot")
        for i in range(3)
    }
    assert expected_cols.issubset(set(feats.columns))


def test_compute_features_pre_shot_gk_xfn_raises_on_missing_defending_gk_player_id():
    from silly_kicks.vaep.base import VAEP
    from silly_kicks.tracking.features import pre_shot_gk_default_xfns
    from tests.tracking._fixtures_action_context import minimal_vaep_states_with_gk_and_frames

    games, gamestates, frames = minimal_vaep_states_with_gk_and_frames()
    # Strip defending_gk_player_id from each state slot.
    gamestates = [s.drop(columns=["defending_gk_player_id"]) for s in gamestates]
    v = VAEP(xfns=pre_shot_gk_default_xfns)
    with pytest.raises(ValueError, match="defending_gk_player_id"):
        v.compute_features(games, gamestates, frames=frames)
```

Add a fixture helper in `tests/tracking/_fixtures_action_context.py` for the VAEP states form:

```python
def minimal_vaep_states_with_gk_and_frames():
    """Returns (games, gamestates, frames) suitable for VAEP.compute_features.
    `gamestates` is a list of nb_states DataFrames (a0, a1, a2). All have
    defending_gk_player_id populated.
    """
    # Build off minimal_shot_actions_with_frames_and_gk — wrap into nb_states slots.
```

- [ ] **Step 6.2: Run tests, verify RED**

```bash
uv run pytest tests/vaep/test_compute_features_frames_kwarg.py -k "pre_shot_gk" -v
```

Expected: FAIL — fixture missing or behavior not yet wired.

- [ ] **Step 6.3: Implement fixtures and verify GREEN**

(No production-code change needed; extension is fixture-only since the dispatch already works via PR-S20 framework.)

```bash
uv run pytest tests/vaep/test_compute_features_frames_kwarg.py -v
```

Expected: ALL PASS.

- [ ] **Step 6.4: Write AUC-uplift integration test (HybridVAEP)**

Add to `tests/vaep/test_hybrid_with_tracking.py`:

```python
def test_hybrid_vaep_auc_uplift_with_gk_features():
    """Train HybridVAEP three times on the same synthetic data:
       - baseline: hybrid_xfns_default
       - with_action_context: + tracking_default_xfns
       - with_action_context_and_gk: + tracking_default_xfns + pre_shot_gk_default_xfns
       Assert AUC ranks: with_gk >= with_action_context >= baseline (within tolerance).
       Tolerance for the GK uplift: epsilon=0.005 (synthetic data has limited GK signal).
    """
    from silly_kicks.vaep.hybrid import HybridVAEP, hybrid_xfns_default
    from silly_kicks.tracking.features import tracking_default_xfns, pre_shot_gk_default_xfns
    from tests.vaep._synthetic_dataset import build_synthetic_dataset_with_gk

    games, gamestates, labels, frames = build_synthetic_dataset_with_gk()

    auc_baseline = _train_and_eval(HybridVAEP(xfns=hybrid_xfns_default), games, gamestates, labels)
    auc_w_ac = _train_and_eval(
        HybridVAEP(xfns=hybrid_xfns_default + tracking_default_xfns), games, gamestates, labels, frames=frames,
    )
    auc_w_gk = _train_and_eval(
        HybridVAEP(xfns=hybrid_xfns_default + tracking_default_xfns + pre_shot_gk_default_xfns),
        games, gamestates, labels, frames=frames,
    )
    assert auc_w_ac >= auc_baseline - 0.001
    assert auc_w_gk >= auc_w_ac - 0.005   # GK adds shot-only signal; tolerance accounts for sample noise


def _train_and_eval(model, games, gamestates, labels, *, frames=None):
    """Helper: split-fit-score; return AUC on held-out split."""
    # ... fill in (use sklearn StratifiedKFold or simple train/test split)
```

Build helper `tests/vaep/_synthetic_dataset.py::build_synthetic_dataset_with_gk` — extend the existing PR-S20 synthetic dataset to include defending_gk_player_id + matching frame rows for shot actions.

- [ ] **Step 6.5: Run tests, verify GREEN**

```bash
uv run pytest tests/vaep/test_hybrid_with_tracking.py -v
```

Expected: ALL PASS. AtomicVAEP smoke test (no AUC assertion) gets the same parametric extension if the test file already covers atomic; otherwise add an `@pytest.mark.parametrize("vaep_class", [HybridVAEP, AtomicVAEP])` wrapper.

---

## Loop 7 — TF-11 regenerator script

**Files:**
- Create: `scripts/regenerate_action_context_baselines.py`
- Modify: `tests/datasets/tracking/empirical_action_context_baselines.json` (overwritten by script)
- Create: `tests/datasets/tracking/action_context_slim/{provider}_expected.parquet` × 4

- [ ] **Step 7.1: Write the regenerator script**

Create `scripts/regenerate_action_context_baselines.py`:

```python
"""Regenerate slim parquet expected outputs + JSON distribution baselines.

For each provider:
  1. Load committed slim parquet (input actions + frames).
  2. Run add_pre_shot_gk_context(actions) (events-only) → defending_gk_player_id.
  3. Run add_action_context(actions, frames) → 4 PR-S20 features + provenance.
  4. Run add_pre_shot_gk_position(actions_with_gk_id, frames) → 4 GK features + provenance.
  5. Project to expected schema; write *_expected.parquet (overwrites).
  6. Compute p25/p50/p75/p99 per feature; populate JSON null slots.

Run: uv run python scripts/regenerate_action_context_baselines.py

Idempotent. Not part of CI. Reviewer-visible parquet diff via git diff.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from silly_kicks.spadl.utils import add_pre_shot_gk_context
from silly_kicks.tracking.features import add_action_context, add_pre_shot_gk_position


SLIM_DIR = Path("tests/datasets/tracking/action_context_slim")
JSON_PATH = Path("tests/datasets/tracking/empirical_action_context_baselines.json")
PFF_DIR = Path("tests/datasets/tracking/pff")

EXPECTED_COLUMNS = [
    "action_id",
    "nearest_defender_distance",
    "actor_speed",
    "receiver_zone_density",
    "defenders_in_triangle_to_goal",
    "pre_shot_gk_x",
    "pre_shot_gk_y",
    "pre_shot_gk_distance_to_goal",
    "pre_shot_gk_distance_to_shot",
    "frame_id",
    "time_offset_seconds",
    "link_quality_score",
    "n_candidate_frames",
]
FEATURE_COLS_FOR_BASELINES = [
    "nearest_defender_distance",
    "actor_speed",
    "receiver_zone_density",
    "defenders_in_triangle_to_goal",
    "pre_shot_gk_x",
    "pre_shot_gk_y",
    "pre_shot_gk_distance_to_goal",
    "pre_shot_gk_distance_to_shot",
]
PERCENTILES = [25, 50, 75, 99]


def _load_provider_inputs(provider: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load actions + frames for a provider. Layout depends on slim-parquet format."""
    if provider == "pff":
        # Synthetic; sourced from the PFF medium_halftime parquet (or a shot-augmented variant).
        ...   # fill at execution time based on Loop 0 findings
        return actions, frames
    actions_path = SLIM_DIR / f"{provider}_slim_actions.parquet"
    frames_path = SLIM_DIR / f"{provider}_slim_frames.parquet"
    if actions_path.exists() and frames_path.exists():
        return pd.read_parquet(actions_path), pd.read_parquet(frames_path)
    # Fallback: combined parquet with __row_kind discriminator.
    df = pd.read_parquet(SLIM_DIR / f"{provider}_slim.parquet")
    return df[df["__row_kind"] == "action"].drop(columns=["__row_kind"]), df[df["__row_kind"] == "frame"].drop(columns=["__row_kind"])


def _compute_expected(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    actions = add_pre_shot_gk_context(actions)
    out_ac = add_action_context(actions, frames)
    out_gk = add_pre_shot_gk_position(actions, frames)
    # Merge — both have action_id; we keep one set of provenance columns (they should match).
    out = out_ac.copy()
    for col in ("pre_shot_gk_x", "pre_shot_gk_y", "pre_shot_gk_distance_to_goal", "pre_shot_gk_distance_to_shot"):
        out[col] = out_gk[col].values
    return out[EXPECTED_COLUMNS]


def _compute_percentiles(expected: pd.DataFrame) -> dict:
    out = {}
    for col in FEATURE_COLS_FOR_BASELINES:
        series = expected[col].dropna()
        out_col = {}
        for p in PERCENTILES:
            key = f"{col}_p{p}"
            if len(series) == 0:
                out_col[key] = None   # documented coverage gap; assertion still flags this
            else:
                out_col[key] = float(np.percentile(series, p))
        out.update(out_col)
    return out


def main() -> None:
    json_state = json.loads(JSON_PATH.read_text())
    for provider in ("sportec", "metrica", "skillcorner", "pff"):
        actions, frames = _load_provider_inputs(provider)
        expected = _compute_expected(actions, frames)
        expected_path = SLIM_DIR / f"{provider}_expected.parquet"
        expected.to_parquet(expected_path, index=False)
        print(f"{provider}: wrote {len(expected)} rows to {expected_path}")

        percentiles = _compute_percentiles(expected)
        for k, v in percentiles.items():
            json_state["providers"][provider][k] = v

    JSON_PATH.write_text(json.dumps(json_state, indent=2))
    print(f"Wrote updated baselines JSON: {JSON_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7.2: Run the regenerator**

```bash
uv run python scripts/regenerate_action_context_baselines.py
```

Expected output: 4 lines `{provider}: wrote N rows to ...` + `Wrote updated baselines JSON`. Confirms expected parquets and JSON populated.

- [ ] **Step 7.3: Manual sanity-eyeball the parquets and JSON**

```bash
git diff --stat tests/datasets/tracking/
```

Verify:
- 4 new `*_expected.parquet` files appeared.
- `empirical_action_context_baselines.json` lost all `null` slots.
- No regression on `*_slim.parquet` files (input parquets unchanged).

- [ ] **Step 7.4: If a provider has 0 shots → GK-feature columns are all-NaN in that expected parquet**

This is the R1 mitigation per Loop 0 findings. Document in commit message; the per-row gate (Loop 8) will verify the all-NaN path for that provider.

---

## Loop 8 — TF-11 per-row regression gate (load-bearing CI test)

**Files:**
- Create: `tests/tracking/test_action_context_expected_output.py`

- [ ] **Step 8.1: Write the per-row regression gate**

Create `tests/tracking/test_action_context_expected_output.py`:

```python
"""Bit-exact regression gate for add_action_context + add_pre_shot_gk_position.

Per ADR-005-style hybrid validation: per-row gate is load-bearing; JSON
baselines are documentation. This file is the per-row gate.

Failure mode: 'row 5: pre_shot_gk_distance_to_shot expected 4.32, got 4.51' —
fully debuggable. No statistical noise tolerance.
"""
from pathlib import Path

import pandas as pd
import pytest

from silly_kicks.spadl.utils import add_pre_shot_gk_context
from silly_kicks.tracking.features import add_action_context, add_pre_shot_gk_position


SLIM_DIR = Path(__file__).parent.parent / "datasets/tracking/action_context_slim"
EXPECTED_COLUMNS = [
    "action_id",
    "nearest_defender_distance",
    "actor_speed",
    "receiver_zone_density",
    "defenders_in_triangle_to_goal",
    "pre_shot_gk_x",
    "pre_shot_gk_y",
    "pre_shot_gk_distance_to_goal",
    "pre_shot_gk_distance_to_shot",
    "frame_id",
    "time_offset_seconds",
    "link_quality_score",
    "n_candidate_frames",
]


def _load_provider_inputs(provider: str):
    """Mirror of regenerator's loader — must stay in sync with scripts/regenerate_action_context_baselines.py"""
    if provider == "pff":
        ...   # fill at execution time per Loop 0 / Loop 7 decision
        return actions, frames
    actions_path = SLIM_DIR / f"{provider}_slim_actions.parquet"
    frames_path = SLIM_DIR / f"{provider}_slim_frames.parquet"
    if actions_path.exists() and frames_path.exists():
        return pd.read_parquet(actions_path), pd.read_parquet(frames_path)
    df = pd.read_parquet(SLIM_DIR / f"{provider}_slim.parquet")
    return df[df["__row_kind"] == "action"].drop(columns=["__row_kind"]), df[df["__row_kind"] == "frame"].drop(columns=["__row_kind"])


@pytest.mark.parametrize("provider", ["sportec", "metrica", "skillcorner", "pff"])
def test_add_action_context_per_row_regression(provider):
    actions, frames = _load_provider_inputs(provider)
    actions = add_pre_shot_gk_context(actions)
    out_ac = add_action_context(actions, frames)
    out_gk = add_pre_shot_gk_position(actions, frames)
    actual = out_ac.copy()
    for col in ("pre_shot_gk_x", "pre_shot_gk_y", "pre_shot_gk_distance_to_goal", "pre_shot_gk_distance_to_shot"):
        actual[col] = out_gk[col].values
    actual = actual[EXPECTED_COLUMNS]

    expected = pd.read_parquet(SLIM_DIR / f"{provider}_expected.parquet")
    pd.testing.assert_frame_equal(actual, expected, atol=1e-9, rtol=0, check_dtype=True)
```

- [ ] **Step 8.2: Run gate, verify GREEN (output already regenerated in Loop 7)**

```bash
uv run pytest tests/tracking/test_action_context_expected_output.py -v
```

Expected: 4 PASS — bit-exact match on each provider's expected output.

- [ ] **Step 8.3: Refactor — extract `_load_provider_inputs` helper into a shared module**

The regenerator script (`scripts/regenerate_action_context_baselines.py`) and the per-row gate test both reload provider inputs the same way. To avoid divergence (Hyrum's Law: if either drifts, false-pass / false-fail), extract:

Create `tests/tracking/_provider_inputs.py`:

```python
"""Shared loader for slim provider inputs. Used by the regression test (CI) and
the regenerator script (manual). Keeping one source of truth ensures expected
parquets and the per-row gate read identical inputs.
"""
from pathlib import Path
import pandas as pd

SLIM_DIR = Path(__file__).parent.parent / "datasets/tracking/action_context_slim"


def load_provider_inputs(provider: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if provider == "pff":
        ...   # filled at execution time per Loop 0 / Loop 7 decision
        return actions, frames
    actions_path = SLIM_DIR / f"{provider}_slim_actions.parquet"
    frames_path = SLIM_DIR / f"{provider}_slim_frames.parquet"
    if actions_path.exists() and frames_path.exists():
        return pd.read_parquet(actions_path), pd.read_parquet(frames_path)
    df = pd.read_parquet(SLIM_DIR / f"{provider}_slim.parquet")
    return (
        df[df["__row_kind"] == "action"].drop(columns=["__row_kind"]),
        df[df["__row_kind"] == "frame"].drop(columns=["__row_kind"]),
    )
```

Update both the regenerator script and the test to import this helper (delete the local copies).

- [ ] **Step 8.4: Re-run gate to confirm refactor is green**

```bash
uv run pytest tests/tracking/test_action_context_expected_output.py -v
```

Expected: 4 PASS.

---

## Loop 9 — TF-11 JSON shape gate + cross-provider parity bounds extension

**Files:**
- Create: `tests/tracking/test_empirical_action_context_baselines.py`
- Modify: `tests/tracking/test_action_context_cross_provider.py` (extend with GK bounds)

- [ ] **Step 9.1: Write JSON shape gate test (RED)**

Create `tests/tracking/test_empirical_action_context_baselines.py`:

```python
"""JSON-shape sanity for the empirical baselines file (PR-S21 promoted from advisory)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

JSON_PATH = Path(__file__).parent.parent / "datasets/tracking/empirical_action_context_baselines.json"
SLIM_DIR = Path(__file__).parent.parent / "datasets/tracking/action_context_slim"

FEATURE_COLS = [
    "nearest_defender_distance",
    "actor_speed",
    "receiver_zone_density",
    "defenders_in_triangle_to_goal",
    "pre_shot_gk_x",
    "pre_shot_gk_y",
    "pre_shot_gk_distance_to_goal",
    "pre_shot_gk_distance_to_shot",
]
PERCENTILES = [25, 50, 75, 99]
PROVIDERS = ["sportec", "metrica", "skillcorner", "pff"]


def test_baselines_json_is_complete():
    """4 features × 4 percentiles × 4 providers all populated.
    Slots are allowed to be None ONLY where the data has 0 valid samples
    for that feature in that provider's slim slice (documented coverage gap)."""
    state = json.loads(JSON_PATH.read_text())
    for prov in PROVIDERS:
        for feat in FEATURE_COLS:
            for p in PERCENTILES:
                key = f"{feat}_p{p}"
                assert key in state["providers"][prov], f"Missing key {prov}/{key}"
                value = state["providers"][prov][key]
                # null is allowed only when the *_expected.parquet has all-NaN for that feature.
                if value is None:
                    expected = pd.read_parquet(SLIM_DIR / f"{prov}_expected.parquet")
                    assert expected[feat].isna().all(), (
                        f"JSON has null for {prov}/{key} but {prov}_expected.parquet has non-NaN values."
                    )


def test_baselines_json_matches_expected_parquet_distribution():
    """Sanity: JSON percentiles match those re-computed from *_expected.parquet (strict).
    Catches accidental hand-edits drifting the JSON away from the data."""
    state = json.loads(JSON_PATH.read_text())
    for prov in PROVIDERS:
        expected = pd.read_parquet(SLIM_DIR / f"{prov}_expected.parquet")
        for feat in FEATURE_COLS:
            series = expected[feat].dropna()
            for p in PERCENTILES:
                key = f"{feat}_p{p}"
                json_v = state["providers"][prov][key]
                if len(series) == 0:
                    assert json_v is None, f"Empty data but JSON has non-null for {prov}/{key}"
                    continue
                computed = float(np.percentile(series, p))
                assert json_v == pytest.approx(computed, abs=1e-6, rel=0), (
                    f"{prov}/{key}: JSON={json_v}, parquet-computed={computed}"
                )
```

- [ ] **Step 9.2: Run JSON tests, verify GREEN (regenerator already populated the JSON)**

```bash
uv run pytest tests/tracking/test_empirical_action_context_baselines.py -v
```

Expected: PASS — JSON populated by Loop 7; matches *_expected.parquet computed percentiles.

- [ ] **Step 9.3: Extend cross-provider parity test with GK bounds**

Add to `tests/tracking/test_action_context_cross_provider.py`:

```python
@pytest.mark.parametrize("provider", ["pff", "sportec", "metrica", "skillcorner"])
def test_pre_shot_gk_position_bounds(provider):
    """Where shots exist, GK position columns satisfy bound constraints.
    Off-pitch tolerance: x in [-5, 110], y in [-5, 73] — acknowledges per-provider
    asymmetry (memory: reference_lakehouse_tracking_traps).
    """
    expected = pd.read_parquet(
        Path(__file__).parent.parent / f"datasets/tracking/action_context_slim/{provider}_expected.parquet"
    )
    has_gk_rows = expected["pre_shot_gk_x"].notna()
    if not has_gk_rows.any():
        pytest.skip(f"{provider} has no shot rows with linked GK in slim slice; bounds check trivially passes")
    gk = expected[has_gk_rows]
    assert (gk["pre_shot_gk_x"].between(-5, 110)).all()
    assert (gk["pre_shot_gk_y"].between(-5, 73)).all()
    assert (gk["pre_shot_gk_distance_to_goal"].between(0, 130)).all()
    assert (gk["pre_shot_gk_distance_to_shot"].between(0, 130)).all()
```

- [ ] **Step 9.4: Run extended cross-provider tests**

```bash
uv run pytest tests/tracking/test_action_context_cross_provider.py -v
```

Expected: ALL PASS.

---

## Loop 10 — e2e real-data sweep extension + Public-API Examples coverage

**Files:**
- Modify: `tests/tracking/test_action_context_real_data_sweep.py` (extend with GK aggregator)
- Test: `tests/test_public_api_examples.py` — auto-discovers; verify

- [ ] **Step 10.1: Extend the existing parametrized e2e test**

Add a new parametrized test in `tests/tracking/test_action_context_real_data_sweep.py` that runs `add_pre_shot_gk_position` on full match data per provider:

```python
@pytest.mark.e2e
@pytest.mark.parametrize("provider,env_var", [
    ("pff",         "PFF_TRACKING_DIR"),
    ("sportec",     "IDSSE_TRACKING_DIR"),
    ("metrica",     "METRICA_TRACKING_DIR"),
    ("skillcorner", "SKILLCORNER_TRACKING_DIR"),
])
def test_add_pre_shot_gk_position_real_data_sweep(provider, env_var):
    """Per-provider full-match sweep:
       - Compute add_pre_shot_gk_context (events-only) → defending_gk_player_id
       - Compute add_pre_shot_gk_position(actions, frames)
       - Bounds audit (GK positions in expected ranges; non-shot rows are NaN)
    """
    path = os.environ.get(env_var)
    if not path:
        pytest.skip(
            f"{env_var} not set; cannot run {provider} e2e sweep "
            f"(memory: feedback_no_silent_skips_on_required_testing — surface loud)."
        )
    actions, frames = _load_full_match(provider, path)
    actions = add_pre_shot_gk_context(actions)
    enriched = add_pre_shot_gk_position(actions, frames)

    # Where defending_gk_player_id is set AND action is a shot AND linked, GK position is non-NaN.
    shot_ids = {spadlconfig.actiontype_id[n] for n in ("shot", "shot_freekick", "shot_penalty")}
    is_shot = enriched["type_id"].isin(shot_ids)
    has_gk_id = enriched["defending_gk_player_id"].notna()
    is_linked = enriched["frame_id"].notna()
    eligible = is_shot & has_gk_id & is_linked
    if eligible.sum() == 0:
        pytest.skip(f"{provider} match has no eligible shot rows; cannot validate GK columns.")

    eligible_rows = enriched[eligible]
    # Bounds (broader than slim — this is full-match real data).
    assert (eligible_rows["pre_shot_gk_x"].between(-5, 110)).all()
    assert (eligible_rows["pre_shot_gk_y"].between(-5, 73)).all()
    # Non-shot rows have NaN GK columns.
    assert enriched.loc[~is_shot, "pre_shot_gk_x"].isna().all()
```

`_load_full_match(provider, path)` is the existing PR-S20 helper — verify and reuse.

- [ ] **Step 10.2: Run e2e suite locally with all 4 env vars set**

Set env vars per memory references at runtime (NOT committed in spec/code):
- `$env:PFF_TRACKING_DIR = "<from memory reference_pff_data_local>"`
- `$env:IDSSE_TRACKING_DIR = "<from memory; lakehouse Sportec tracking root>"`
- `$env:METRICA_TRACKING_DIR = "<from memory>"`
- `$env:SKILLCORNER_TRACKING_DIR = "<from memory>"`

```bash
uv run pytest tests/tracking/test_action_context_real_data_sweep.py -m e2e -v
```

Expected: ALL PASS across 4 providers (PR-S20 4 features + PR-S21 GK aggregator).

If any provider skips for env-var-unset reason, surface inline in execution log BEFORE final-review (memory: `feedback_no_silent_skips_on_required_testing`). Resolve by setting the env var from runtime memory resolution.

- [ ] **Step 10.3: Verify Public-API Examples auto-discovery**

```bash
uv run pytest tests/test_public_api_examples.py -v
```

Expected: PASS — auto-discovers new public defs in `silly_kicks.tracking.features` and `silly_kicks.atomic.tracking.features` (4 helpers + aggregator each = 10 new public defs). Each has an `Examples` block per Loop 3 / Loop 4 implementation.

If any example block fails parsing, fix the docstring inline (typically `>>> ` indentation or `# noqa: B006` markers).

---

## Loop 11 — Documentation: NOTICE, TODO.md, CHANGELOG

**Files:**
- Modify: `NOTICE` (extend Anzer & Bauer entry)
- Modify: `TODO.md` (mark TF-1 + TF-11 SHIPPED; archive Active Cycle; add TF-12/13/14)
- Modify: `CHANGELOG.md` (add 2.9.0 entry)

- [ ] **Step 11.1: Extend NOTICE entry for Anzer & Bauer**

Edit `NOTICE`. Replace the existing one-line description:

```diff
 - Anzer, G., & Bauer, P. (2021). "A goal scoring probability model for shots
   based on synchronized positional and event data in football and futsal."
   Frontiers in Sports and Active Living, 3, 624475.
-  (player_speed and distance-to-defender as xG features)
+  (player_speed, distance-to-defender, and defending-GK-position as xG features)
```

Single-line change. No new bullets.

- [ ] **Step 11.2: Update TODO.md — mark SHIPPED + archive Active Cycle + add TF-12/13/14**

Edit `TODO.md`:

1. Header date update: `**Last updated**: 2026-05-01 (PR-S21 cycle in flight — feat/pre-shot-gk-plus-baselines)`. Update the (A) line to reflect 2.8.0 status (assuming PR-S20 has shipped to PyPI by now; otherwise leave 2.7.0 SHIPPED line).

2. Remove the TF-1 row from the On-Deck table.

3. Remove the TF-11 row from the On-Deck table.

4. Add 3 new rows (TF-12, TF-13, TF-14) per spec § 7.2:

```markdown
| TF-12 | `pre_shot_gk_angle_*` (signed angle from goal-line normal, off-line angular displacement, etc.) | Dunkin' | Anzer & Bauer 2021; PR-S21 deferral | Library ships positions + 2 distances in PR-S21; angle conventions deferred. Multiple competing definitions (relative to shot trajectory? to goal-line normal? signed vs unsigned?) — pick one canonical convention with downstream reviewer input before landing. ~30-50 LOC. |
| TF-13 | Frame-based defending-GK identification (fallback when events-based `defending_gk_player_id` is NaN) | Wicked | Bauer & Anzer 2021 (Section 3 carrier-ID heuristic, similar shape); Bekkers 2024 (DEFCON GK identification) | Heuristic: defender closest to own goal at the linked frame, possibly conditional on jersey/role data when supplied. Composes with PR-S21's strict events-only ID (callers opt into fallback). ~80-120 LOC + ADR if chosen heuristic is contentious. |
| TF-14 | Defensive-line features (line height, line compactness, line break detection) | Wicked | Power et al. 2017 (line break in OBSO); Spearman 2018; Anzer & Bauer 2021 | Per-frame defending team's outfield line geometry (median y of back-4, std dev, max gap). Could replace ad-hoc "defenders behind the ball" features in xG. ~150 LOC. |
```

5. Replace the Active Cycle section:

```markdown
## Active Cycle

PR-S21 — TF-1 (`pre_shot_gk_position_*`) + TF-11 (baselines backfill)
(target silly-kicks 2.9.0).

Branch: `feat/pre-shot-gk-plus-baselines`. Spec + plan: [docs/superpowers/specs/2026-05-01-pre-shot-gk-plus-baselines-design.md](docs/superpowers/specs/2026-05-01-pre-shot-gk-plus-baselines-design.md), [docs/superpowers/plans/2026-05-01-pre-shot-gk-plus-baselines.md](docs/superpowers/plans/2026-05-01-pre-shot-gk-plus-baselines.md).

After ship, this section gets archived; PR-S21 is expected to ship within
the existing ADR-005 envelope — no new ADR.
```

- [ ] **Step 11.3: Add CHANGELOG entry**

Edit `CHANGELOG.md`. Add new section at top:

```markdown
## 2.9.0 — 2026-05-XX (PR-S21)

### Added
- `silly_kicks.tracking.features.pre_shot_gk_x` / `pre_shot_gk_y` / `pre_shot_gk_distance_to_goal` / `pre_shot_gk_distance_to_shot` — 4 GK-position features at the linked frame.
- `silly_kicks.tracking.features.add_pre_shot_gk_position` — aggregator emitting the 4 GK columns + 4 linkage-provenance columns. NaN-safe per ADR-003.
- `silly_kicks.atomic.tracking.features` mirrors of all of the above (atomic SPADL parity).
- `silly_kicks.tracking.features.pre_shot_gk_default_xfns` — composable into HybridVAEP / AtomicVAEP via xfn-list append.
- `silly_kicks.spadl.utils.add_pre_shot_gk_context(*, frames=None)` — additive optional kwarg; emits 4 GK-position columns + 4 provenance columns when `frames` is supplied. Bit-identical when `frames=None` (backward-compat preserved). Atomic mirror in `silly_kicks.atomic.spadl.utils`.
- `tests/datasets/tracking/action_context_slim/{provider}_expected.parquet` — per-provider expected output for the bit-exact per-row regression gate (TF-11).
- `scripts/regenerate_action_context_baselines.py` — one-shot regenerator for the JSON baselines + `*_expected.parquet`.
- `tests/tracking/test_action_context_expected_output.py` — bit-exact per-row regression CI gate.
- `tests/tracking/test_empirical_action_context_baselines.py` — JSON shape + JSON-vs-parquet consistency CI gate.

### Changed
- `silly_kicks.tracking.feature_framework.ActionFrameContext` gains `defending_gk_rows: pd.DataFrame` field.
- `tests/datasets/tracking/empirical_action_context_baselines.json` — 64 null percentile slots backfilled (4 percentiles × 8 features × 4 providers, where shot data exists).
- `NOTICE` — Anzer & Bauer (2021) entry description expanded to include defending-GK-position.

### Notes
- No breaking changes. PR-S21 ships entirely within ADR-005's locked architecture; no new ADR.
- Ships under National Park Principle: TF-12 (`pre_shot_gk_angle_*`), TF-13 (frame-based GK identification), TF-14 (defensive-line features) added to TODO.md On-Deck.
```

- [ ] **Step 11.4: Run full test suite — final smoke**

```bash
uv run pytest tests/ -m "not e2e" -v
```

Expected: ALL PASS.

```bash
uv run pytest tests/ -m e2e -v
```

Expected: ALL PASS (e2e env vars set per Loop 10).

---

## Loop 12 — Final review + single squash commit

**Files:** none modified.

- [ ] **Step 12.1: Run full local pre-commit gates (Shift Left)**

Independent commands (run in parallel):

```bash
uv run ruff check silly_kicks/ tests/ scripts/
uv run pyright silly_kicks/ tests/
uv run pytest tests/ -m "not e2e" --tb=short -v
```

All three must be green. Pin pyright + pandas-stubs to CI versions before relying on local pyright (memory: `feedback_ci_cross_version`):

```bash
uv run python --version            # check version
uv pip list | grep -E "(pyright|pandas-stubs)"   # check pinned versions match CI
```

- [ ] **Step 12.2: Loud-skip surface check**

If any e2e test was pytest-skipped due to missing env vars in Loop 10, resolve them inline NOW (before final-review) using runtime memory resolution. Do not bury the skip in summary — surface inline (memory: `feedback_no_silent_skips_on_required_testing`).

```bash
uv run pytest tests/ -m e2e --tb=short -v 2>&1 | grep -E "SKIP|FAIL"
```

Expected: zero SKIP, zero FAIL.

- [ ] **Step 12.3: Run `/final-review` (mandatory pre-commit gate)**

Memory: `feedback_final_review_gate` — `/final-review` (mad-scientist-skills:final-review) is mandatory before the single squash commit. Triggers consistency / docs / C4 diagram pass.

Address any findings inline. Do not skip.

- [ ] **Step 12.4: Stage all changes + ask user for explicit commit approval**

```bash
git status
git diff --stat
```

Surface the diff for user review. Do NOT commit yet — memory: `feedback_commit_policy` requires **explicit approval before that one commit**.

- [ ] **Step 12.5: Single squash commit (after approval)**

Once user approves:

```bash
git add silly_kicks/ tests/ scripts/ docs/ NOTICE TODO.md CHANGELOG.md
git commit -m "$(cat <<'EOF'
feat(tracking): pre_shot_gk_position_* + baselines backfill -- silly-kicks 2.9.0 (PR-S21)

TF-1 — 4-feature pre_shot_gk_position catalog (x, y, distance_to_goal,
distance_to_shot) refining add_pre_shot_gk_context. Tracking-namespace
canonical compute + atomic mirror; events-side helper extended with
optional frames=None kwarg (lazy-imports tracking module per ADR-005 § 5).
Bit-identical when frames=None.

TF-11 — *_expected.parquet per provider (load-bearing bit-exact CI gate);
empirical_action_context_baselines.json populated as documentation +
shape gate. scripts/regenerate_action_context_baselines.py for
deliberate-invocation regeneration.

National Park bundle: TF-12 (`pre_shot_gk_angle_*`), TF-13 (frame-based
GK identification fallback), TF-14 (defensive-line features) added to
TODO.md On-Deck.

ADR-005 envelope respected; no new ADR. ActionFrameContext extension is
non-breaking field addition. Anzer & Bauer (2021) NOTICE entry expanded.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
git status
```

Expected: one commit on `feat/pre-shot-gk-plus-baselines`; clean working tree.

- [ ] **Step 12.6: Push branch + open PR**

```bash
git push -u origin feat/pre-shot-gk-plus-baselines
gh pr create --title "feat(tracking): pre_shot_gk_position_* + baselines backfill — silly-kicks 2.9.0 (PR-S21)" --body "$(cat <<'EOF'
## Summary

- TF-1: 4-feature `pre_shot_gk_position_*` catalog refining `add_pre_shot_gk_context` (Anzer & Bauer 2021). Both tracking-namespace canonical surface AND events-side helper extension via optional `frames=None` lazy-import. Atomic SPADL parity.
- TF-11: bit-exact per-row regression gate via `*_expected.parquet` per provider; JSON baselines backfilled as documentation. Regenerator script committed.
- National Park bundle: TF-12 / TF-13 / TF-14 added to TODO.md.
- Backward-compat preserved: golden-fixture test pins `add_pre_shot_gk_context(actions)` (no frames) bit-identical to silly-kicks 2.8.0.

Spec: [docs/superpowers/specs/2026-05-01-pre-shot-gk-plus-baselines-design.md](docs/superpowers/specs/2026-05-01-pre-shot-gk-plus-baselines-design.md).
Plan: [docs/superpowers/plans/2026-05-01-pre-shot-gk-plus-baselines.md](docs/superpowers/plans/2026-05-01-pre-shot-gk-plus-baselines.md).

## Test plan

- [x] Tier 1 analytical kernel tests (6 GK kernel tests)
- [x] Tier 2 schema-wrapper parity (standard + atomic)
- [x] Tier 3 per-row regression gate on slim parquets (4 providers)
- [x] e2e real-data sweep (4 providers; full match)
- [x] Backward-compat golden-fixture test (`add_pre_shot_gk_context` bit-identical when `frames=None`)
- [x] HybridVAEP AUC uplift test (synthetic data)
- [x] No-module-import-cycle test (`spadl.utils` does not eagerly import `tracking.*`)
- [x] Public-API Examples auto-discovery
- [x] NaN-safety auto-discovery (ADR-003)
- [x] `/final-review` clean

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Return the PR URL.

- [ ] **Step 12.7: After merge — tag + PyPI publish**

(User-driven; not part of plan execution. Memory: `project_release_state` pattern.)

```bash
# After PR is merged on GitHub:
git fetch origin
git checkout main
git pull --ff-only origin main
git tag v2.9.0 <merge-sha>
git push origin v2.9.0
# PyPI publish via GitHub Actions release workflow.
```

---

## Self-Review

**Spec coverage check** (each spec section/requirement → task that implements it):

| Spec section | Implementing task(s) |
|---|---|
| § 2.1 4-feature catalog | Loop 2 (kernel) + Loop 3 (standard) + Loop 4 (atomic) |
| § 2.2 Two public surfaces (Fork 1) | Loop 3 (tracking-namespace) + Loop 5 (events-side wrapper) |
| § 2.3 Atomic-SPADL parity | Loop 4 + Loop 5.5 (atomic mirror) |
| § 2.4 VAEP integration via composition (`pre_shot_gk_default_xfns` separate list) | Loop 3.7 + Loop 6 |
| § 2.5 TF-11 hybrid validation (Fork 2) | Loop 7 (regenerator) + Loop 8 (per-row gate) + Loop 9 (JSON gate) |
| § 2.6 TDD-first | Every Loop has RED→GREEN steps |
| § 2.7 NOTICE attribution (Fork 3 = extend) | Loop 11.1 |
| § 2.8 Backward-compat-bit-identity | Loop 5.1 (golden fixture) + Loop 5.6 (verify) |
| § 2.9 National Park bundle | Loop 11.2 (add TF-12/13/14) |
| § 4.3 Kernel | Loop 2 |
| § 4.5 Per-feature degradation policy | Loop 2 (kernel test cases 1-6) + Loop 5.4 (NaN-safety test) |
| § 4.5.1 Slim parquet expected-output companion | Loop 7 (write parquets) + Loop 8 (read parquets) |
| § 4.5.2 Regenerator script | Loop 7 |
| § 4.5.3 Per-row regression gate | Loop 8 |
| § 4.5.4 JSON shape gate + JSON-vs-parquet consistency | Loop 9.1 + 9.2 |
| § 4.5.5 Coverage gap acknowledgment | Loop 0 (probe) + Loop 7.4 (record decision) |
| § 5 Test files | Loops 1-9 (every test file referenced gets a corresponding Loop) |
| § 6.1 R1-R6 risks | R1: Loop 0; R2: documentation in Loop 1.3 docstring; R3: Loop 1.7 dtype-cast; R4: Loop 12.1 pin pyright; R5: Loop 5.1 golden fixture; R6: Loop 3.7 ValueError |
| § 6.2 OI-1, OI-2 | Loop 0; OI-3 resolved in spec |
| § 6.5 Execution-session expectations | Loop 12 (one commit, final-review, env vars from memory) |
| § 6.6 Sign-off criteria | Loop 12 (final review + golden fixture + AUC test + Examples + NOTICE/TODO/CHANGELOG) |
| § 7.1 NOTICE update | Loop 11.1 |
| § 7.2 TODO.md update | Loop 11.2 |
| § 7.3 CHANGELOG entry | Loop 11.3 |

All spec sections have implementing tasks. No gaps.

**Placeholder scan:** the plan contains 4 intentional `...   # fill at execution time` markers in Loops 7.1, 8.1, and 8.3 for the PFF input-loading branch. These are tied to OI-2 (PFF medium_halftime shot count probe); the execution session resolves them in Loop 0 + Loop 7. Not a placeholder failure — they're conditional-on-probe logic with clear instructions. All other "..." cases are docstring continuations marked explicitly.

**Type consistency:** Method/property names verified across loops:
- `defending_gk_player_id` consistent across spec, kernel, helpers, aggregator, events-side wrapper.
- `pre_shot_gk_x` / `pre_shot_gk_y` / `pre_shot_gk_distance_to_goal` / `pre_shot_gk_distance_to_shot` consistent across kernel, helpers, aggregator, default-xfn list, JSON baselines schema, expected parquet schema, CHANGELOG.
- `frame_id`, `time_offset_seconds`, `link_quality_score`, `n_candidate_frames` consistent (matches PR-S20 ADR-005 § 7).
- `pre_shot_gk_default_xfns` consistent across helper module, VAEP test, CHANGELOG.
- `_pre_shot_gk_position` (kernel name) consistent across `_kernels.py`, helpers, aggregator.
- `add_pre_shot_gk_position` (aggregator name) consistent across tracking module, atomic mirror, events-side wrapper lazy-import, regenerator script, per-row gate test.
- `add_pre_shot_gk_context` (events-side helper name) unchanged from PR-S20.

No type/name drift.

---

**Plan complete and saved to `docs/superpowers/plans/2026-05-01-pre-shot-gk-plus-baselines.md`.**
