# TF-60 PR2 — Layer-2 Danger-Behind-Line Valuation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the five additive Tier-1 Layer-2 rest-defense columns (`rd_attacker_space_control`, `rd_danger_behind_line`, `rd_danger_behind_line_gk`, `rd_gk_coverage_behind_line`, `rd_gk_reachable_coverage_m2`) to the `compute_rest_defense` samples table, via a new `silly_kicks/restdefense/_danger.py`, reusing existing tracking seams extended additively.

**Architecture:** Layer 2 is danger-behind-the-line valuation. It reuses TF-7 pitch control (`compute_pitch_control`, `PitchControlSurface.control_in_region`, `compute_threat_pc`) and TF-15 GK reachable area (`compute_gk_influence`), oriented entirely via `GoalMap` (ADR-055). The one threat engine (`compute_threat_pc`) and the one reachable-area engine (`compute_gk_influence`) are **extended additively** (default-`None` byte-identical) with, respectively, an optional per-cell `field_weight` hook (powers the OBPV `w_field` opt-in) and an optional `region` restriction (powers `∩ Z`); `zero_velocity_if_unavailable` is exported so restdefense can prepare velocity-less frames for its own `compute_pitch_control` calls. Everything is additive — no existing column changes, no VAEP retrain, in no default xfn list, C4 unchanged.

**Tech Stack:** Python, pandas (nullable dtypes), numpy, scipy (via TF-7). No new runtime dependency.

**Spec:** `docs/superpowers/specs/2026-08-30-tf60-rest-defense-structure-and-gk-design.md` (§7.2 metric catalog, §11 velocity/FOV, §13 params, §14 schema, §16.2 CI gates, §17 cycle table, §18 attribution). The plan argues from the spec; executors read both.

## Global Constraints

Copied verbatim from the spec / user standing rules — every task's requirements implicitly include this section.

- **Additive only.** No existing column changes value; primary columns are byte-identical with and without `visible_area` and with `danger_field_weight=False`. No VAEP retrain. In no default xfn list. C4 unchanged (no new action-coupled `add_*` aggregator, no new container).
- **Option 2 (owner-approved 2026-08-30):** the danger engine is `compute_threat_pc`, extended with an optional `field_weight` hook; `compute_gk_influence` gains an optional `region`; `zero_velocity_if_unavailable` is exported. All three are additive with default-`None`/re-export byte-identity → tracking behaviour unchanged, gates re-run green.
- **Orientation via `GoalMap` only (ADR-055):** `G_A = goal_map.get(...)`, `G_B = goal_map.attacked_goal(...)` — never `105 − G_A`, never team identity. An unresolvable end → honest-NaN (`GoalEndUnresolvedError` caught at the `compute_*` edge).
- **ids via `id_compat` (ADR-019):** `ids_match` / `ids_equal` / `canonical_id` — never raw `==`, never `astype(str)` on an id used as a key.
- **Nullable dtypes (ADR-027):** the five Layer-2 columns are all `float64`; NA (never a sentinel 0) on unscoreable rows. `pd.NA`/`NaN` is never a fabricated value.
- **Velocity tiers (ADR-063):** `#1`–`#4` are Tier-1 lifts (positional / zero-velocity model on velocity-less frames); `rd_gk_reachable_coverage_m2` is **Tier-2 → honest-NaN** on velocity-**declared-absent** frames (inherited from `compute_gk_influence`'s existing suppression). An **undeclared**-missing-velocity full-tracking frame **raises** (fail-loud).
- **FOV companions opt-in (ADR-062/077):** region columns get `<col>_observed_fraction` / `_observed_source` only when `visible_area=` is supplied; the ROI is a **fixed action-LTR zone** keyed on the column's role, never a `goal_map`. Companions are glossary-exempt and SB360-audit-exempt.
- **`xt` is fail-closed AND the Layer-2 gate (4.62.0 + P2-02, owner-approved):** the **entire** Layer-2 family is gated on a fitted `ExpectedThreat` — `xt=None` → all five columns NaN before any pitch-control call, so Layer-1-only callers are byte-identical to PR1 (no pitch-control cost, no velocity precondition). With an `xt`, an **unfitted** one → `compute_threat_pc` **raises** (fail-closed). `#1`/`#4` do not *use* `xt`'s values but still require one to be produced — the accepted cost of one opt-in signal.
- **Calibratable defaults ship un-tuned (ADR-009/066):** `WFieldParams` and `danger_field_weight` ship with documented defaults and an empty `for_provider` override map; `danger_field_weight` defaults `False` (so `rd_danger_behind_line` is byte-identical to `compute_threat_pc`). Calibration is a separate gated apply PR, never this cycle.
- **No rescan-in-loop (ADR-068) / sub-quadratic guard (ADR-073):** Layer 2 scores inside the orchestrator's existing single `group_rows` pass — it adds no new per-item loop over the full frames. (No new `group_rows` caller → no new `SCALE_GUARDED` entry required; confirm in Task 9.)
- **COMMIT DISCIPLINE (user standing rule — overrides the writing-plans per-task "Commit" steps):** this whole PR is **ONE commit**. No micro-commits, no per-task commits. Each task ends with its tests green; the **final task** runs the full suite + lint + pyright, then **STOPS and presents the diff for explicit commit approval**. Do not `git commit` or `git push` without Karsten's explicit go-ahead for that specific commit.
- **Branch:** do the work on one feature branch **`tf60-pr2-layer2-danger`** off `main` (one branch per cycle; never a worktree).
- **Version:** bump `silly_kicks/_version.py` `4.102.0` → `4.103.0` (PR-S174), then `uv lock`. Do not hand-edit `uv.lock`.
- **Lint at CI scope:** `python -m ruff check silly_kicks/ tests/ scripts/` and `python -m ruff format --check silly_kicks/ tests/ scripts/`; `python -m pyright` bare. Run the suite with `python -m pytest tests/ -m "not e2e" -v --tb=short`. `tests/tracking/` needs `--benchmark-skip` (it hangs otherwise).

---

## File structure

**New files**
- `silly_kicks/restdefense/_danger.py` — the five Layer-2 metric functions + `layer2_metrics(frame_rows, ctx, *, xt, goal_map, params)`.
- `silly_kicks/restdefense/_wfield.py` — `WFieldParams` (nested frozen) + `build_w_field(own_goal_x, params)` OBPV closure.
- `docs/superpowers/adrs/ADR-081-rest-defense-layer2-danger.md` — the PR2 decision record.
- `tests/restdefense/test_danger.py` — unit tests for the five metrics + keeper-absent / xt-None NaN behaviour.
- `tests/restdefense/test_non_vacuity.py` — GK-blind ≠ GK-included and `field_weight` moves the value (two-sided band).
- `tests/tracking/test_compute_threat_pc_field_weight.py` — the `field_weight` hook (default byte-identical + effect).
- `tests/tracking/test_gk_influence_region.py` — the `region` param (default byte-identical + restriction).

**Modified files**
- `silly_kicks/tracking/_cover_shadows.py` — `_voronoi_threat` + `compute_threat_pc` gain `field_weight`.
- `silly_kicks/tracking/_gk_influence.py` — `compute_gk_influence` gains `region`.
- `silly_kicks/tracking/__init__.py` — export `zero_velocity_if_unavailable` and `compute_gk_influence` (P2-01: the latter was not public).
- `silly_kicks/restdefense/_columns.py` — 5 name constants + `RD_LAYER2_COLUMNS` + `RD_METRIC_COLUMNS`.
- `silly_kicks/restdefense/_config.py` — `w_field_params: WFieldParams` field on `RestDefenseParams`.
- `silly_kicks/restdefense/_compute.py` — thread `xt` + `goal_map`, score Layer 2, emit `RD_METRIC_COLUMNS`.
- `silly_kicks/restdefense/_fov.py` — Z-ROI companions for the Layer-2 region columns.
- `silly_kicks/restdefense/__init__.py` — export the new column lists + `WFieldParams`.
- `silly_kicks/feature_glossary.py` — 5 `FeatureColumn` entries + `_A_NOVILLO_2025` constant.
- `NOTICE` — Ogawa 2025 + Novillo 2025 references.
- `tests/restdefense/_fixtures.py` — `vx`/`vy` on frames, `make_fitted_xt()`, `make_keeper_sensitive_fixture()`.
- `tests/restdefense/test_liveness.py`, `test_orientation.py`, `test_id_dtype_invariance.py`, `test_compute.py`, `test_fov_completeness.py`, `test_purity.py`, `test_e2e_method.py` — extend for the Layer-2 columns.
- `tests/sb360/_entries/_boundary.py` — pass `xt`, enumerate the Layer-2 columns with velocity-leg verdicts.
- `silly_kicks/_version.py`, `uv.lock`, `TODO.md`, `CHANGELOG.md`.

**Interface summary (names later tasks rely on)**
- `_danger.layer2_metrics(frame_rows: pd.DataFrame, ctx: SampleContext, *, xt, goal_map, params: RestDefenseParams, pitch_control_cache=None) -> dict[str, float]` — keys = the 5 `RD_LAYER2_COLUMNS`, values `float` (`NaN` on unscoreable).
- `_wfield.WFieldParams` (frozen); `_wfield.build_w_field(own_goal_x: float, params: WFieldParams) -> Callable[[np.ndarray, np.ndarray], np.ndarray]`.
- `_columns.RD_LAYER2_COLUMNS: list[str]`, `_columns.RD_METRIC_COLUMNS = [*RD_LAYER1_COLUMNS, *RD_LAYER2_COLUMNS]`.
- `compute_threat_pc(..., field_weight: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None)`.
- `compute_gk_influence(..., region: tuple[float, float, float, float] | None = None)`.
- `tests/restdefense/_fixtures.make_fitted_xt() -> ExpectedThreat`, `make_keeper_sensitive_fixture() -> tuple[actions, frames, xt]`.

---

## Task 1: `compute_threat_pc` gains an additive per-cell `field_weight` hook

**Files:**
- Modify: `silly_kicks/tracking/_cover_shadows.py` (`_voronoi_threat` def at line 725; threat-grid at lines 786–792; `compute_threat_pc` def at line 814, call to `_voronoi_threat` at line 883)
- Test: `tests/tracking/test_compute_threat_pc_field_weight.py`

**Interfaces:**
- Produces: `compute_threat_pc(..., field_weight=None)` and `_voronoi_threat(..., field_weight=None)`. `field_weight`, when supplied, is a callable `w(grid_x, grid_y) -> (ny, nx)` ndarray in **absolute** pitch coords (row `i` ↔ `grid_y[i]`, col `j` ↔ `grid_x[j]`), multiplied into the oriented threat grid before the Voronoi partition-sum. Default `None` → byte-identical to today.

- [ ] **Step 1: Write the failing tests**

```python
# tests/tracking/test_compute_threat_pc_field_weight.py
"""compute_threat_pc field_weight hook (TF-60 PR2): additive, default byte-identical."""

import numpy as np

from silly_kicks.tracking import compute_threat_pc
from tests.tracking.test_compute_threat_pc import HOME_GOAL_MAP, _fitted_xt, _frame


def test_field_weight_none_is_byte_identical_to_default():
    frame = _frame()
    a = compute_threat_pc(frame, attacking_team_id=2, xt=_fitted_xt(), goal_map=HOME_GOAL_MAP)
    b = compute_threat_pc(frame, attacking_team_id=2, xt=_fitted_xt(), goal_map=HOME_GOAL_MAP, field_weight=None)
    assert a == b  # exact


def test_uniform_half_weight_halves_the_threat():
    frame = _frame()
    base = compute_threat_pc(frame, attacking_team_id=2, xt=_fitted_xt(), goal_map=HOME_GOAL_MAP)
    halved = compute_threat_pc(
        frame, attacking_team_id=2, xt=_fitted_xt(), goal_map=HOME_GOAL_MAP,
        field_weight=lambda gx, gy: np.full((len(gy), len(gx)), 0.5),
    )
    assert base > 0.0
    assert halved == np.float64(base * 0.5) or abs(halved - base * 0.5) < 1e-12


def test_zero_weight_zeroes_the_threat():
    frame = _frame()
    zeroed = compute_threat_pc(
        frame, attacking_team_id=2, xt=_fitted_xt(), goal_map=HOME_GOAL_MAP,
        field_weight=lambda gx, gy: np.zeros((len(gy), len(gx))),
    )
    assert zeroed == 0.0
```

- [ ] **Step 2: Run — expect FAIL** (`TypeError: unexpected keyword 'field_weight'`)

Run: `python -m pytest tests/tracking/test_compute_threat_pc_field_weight.py -v --benchmark-skip`

- [ ] **Step 3: Implement**

In `_voronoi_threat` (line 725), add `field_weight: "Callable[[np.ndarray, np.ndarray], np.ndarray] | None" = None` as a keyword-only parameter (after `goal_map`). Immediately **after** the oriented `threat_grid` is built (after line 792, before the Voronoi partition at line 794), insert:

```python
        if field_weight is not None:
            threat_grid = threat_grid * field_weight(x_coords, y_coords)
```

(`x_coords = surface.grid_x`, `y_coords = surface.grid_y` are already bound at lines 775–776.) In `compute_threat_pc` (line 814), add the same `field_weight=None` keyword-only parameter and pass it through at the `_voronoi_threat` call (line 883): `..., goal_map=goal_map, field_weight=field_weight)`. Add `from collections.abc import Callable` to the imports if not present (check the file header; use `typing.Callable` only if that's the file's existing idiom). Document `field_weight` in `compute_threat_pc`'s docstring: "optional per-cell weight `w(grid_x, grid_y) -> (ny, nx)` in absolute pitch coords, multiplied into the oriented threat grid before the Voronoi partition (default None = unweighted)."

- [ ] **Step 4: Run — expect PASS** (`python -m pytest tests/tracking/test_compute_threat_pc_field_weight.py -v --benchmark-skip`)

- [ ] **Step 5: Regression — the existing threat-pc + gkdv suites must stay green** (default path unchanged)

Run: `python -m pytest tests/tracking/test_compute_threat_pc.py tests/tracking/test_cover_shadows.py tests/gkdv/ -v --benchmark-skip`
Expected: PASS (no numeric change on the default path).

---

## Task 2: `compute_gk_influence` gains an additive `region` restriction

**Files:**
- Modify: `silly_kicks/tracking/_gk_influence.py` (`compute_gk_influence` def at line 232; reachable-cell sum at line 482; `targets` built at line 442)
- Test: `tests/tracking/test_gk_influence_region.py`

**Interfaces:**
- Produces: `compute_gk_influence(..., region: tuple[float,float,float,float] | None = None)`. When supplied `(x_min, x_max, y_min, y_max)`, `reachable_area_m2` counts only reachable cells whose centre lies in the region. Default `None` → whole pitch, byte-identical. The velocity-suppression (`NaN` on `velocity_unavailable_by_design`) is unchanged and applies before the region restriction.

- [ ] **Step 1: Write the failing tests**

```python
# tests/tracking/test_gk_influence_region.py
"""compute_gk_influence region restriction (TF-60 PR2): additive, default byte-identical."""

from silly_kicks.tracking._gk_influence import compute_gk_influence
from tests.tracking.test_compute_threat_pc import HOME_GOAL_MAP, _fitted_xt, _frame


def _reach(region=None):
    # Home (team 1) keeper defends x=0; team 2 attacks toward x=0. gk_player_id resolved from _frame().
    frame = _frame()
    gk_id = frame[(frame["team_id"] == 1) & frame["is_goalkeeper"].astype(bool)]["player_id"].iloc[0]
    return compute_gk_influence(
        frame, attacking_team_id=2, gk_player_id=gk_id, xt=_fitted_xt(),
        goal_map=HOME_GOAL_MAP, region=region,
    ).reachable_area_m2


def test_region_none_is_whole_pitch():
    whole = _reach(region=None)
    also_whole = _reach(region=(0.0, 105.0, 0.0, 68.0))
    assert whole == also_whole


def test_region_restriction_never_exceeds_whole_pitch():
    whole = _reach(region=None)
    near_goal = _reach(region=(0.0, 20.0, 0.0, 68.0))
    assert 0.0 <= near_goal <= whole + 1e-9


def test_disjoint_region_is_zero():
    # A keeper defending x=0 has no reachable cells in the far attacking third.
    assert _reach(region=(90.0, 105.0, 0.0, 68.0)) == 0.0
```

- [ ] **Step 2: Run — expect FAIL** (unexpected keyword `region`)

Run: `python -m pytest tests/tracking/test_gk_influence_region.py -v --benchmark-skip`

- [ ] **Step 3: Implement**

In `compute_gk_influence` (line 232), add `region: tuple[float, float, float, float] | None = None` as a keyword-only parameter. Replace line 482:

```python
    reachable_area_m2 = float("nan") if _velocity_less else float(unique_cells.sum() * cell_area)
```

with a region-aware form (`targets` from line 442 is the `(n_targets, 2)` cell-centre array aligned to `unique_cells`):

```python
    if region is not None:
        x_min, x_max, y_min, y_max = region
        in_region = (
            (targets[:, 0] >= x_min) & (targets[:, 0] <= x_max)
            & (targets[:, 1] >= y_min) & (targets[:, 1] <= y_max)
        )
        cells = unique_cells & in_region
    else:
        cells = unique_cells
    reachable_area_m2 = float("nan") if _velocity_less else float(cells.sum() * cell_area)
```

Document `region` in the docstring: "optional `(x_min, x_max, y_min, y_max)`; restricts the reachable-area sum to cells in the region (default None = whole pitch). Velocity-suppression to NaN is applied first (ADR-063)."

- [ ] **Step 4: Run — expect PASS** (`python -m pytest tests/tracking/test_gk_influence_region.py -v --benchmark-skip`)

- [ ] **Step 5: Regression** — `python -m pytest tests/tracking/test_gk_influence.py -v --benchmark-skip` (default path unchanged).

---

## Task 3: Export `zero_velocity_if_unavailable` + `compute_gk_influence` from `tracking`

**Files:**
- Modify: `silly_kicks/tracking/__init__.py` (add both to `__all__` and the import block; `zero_velocity_if_unavailable` lives in `silly_kicks/tracking/_velocity_availability.py`, `compute_gk_influence` in `silly_kicks/tracking/_gk_influence.py`)
- Test: `tests/tracking/test_velocity_export.py`

**Interfaces:**
- Produces: `from silly_kicks.tracking import zero_velocity_if_unavailable`. Pure re-export; behaviour unchanged. Signature `zero_velocity_if_unavailable(frames, *, method="spearman") -> pd.DataFrame`: `vx`/`vy` present → same object; absent + declared velocity-less → a zero-velocity copy; absent + undeclared + a velocity-requiring method → raises.
- Produces: `from silly_kicks.tracking import compute_gk_influence`. **`compute_gk_influence` is currently NOT in `tracking.__all__`** (only `add_gk_influence` / `gk_influence_xfns` are; verified — importing it from `silly_kicks.tracking` raises `ImportError`). `_danger.py` (Task 8) needs it, and the restdefense import-allowlist forbids reaching into the private `_gk_influence` path — so it must be a public re-export. Pure additive export; behaviour unchanged (its Task-2 `region` param is already in place).

- [ ] **Step 1: Write the failing test**

```python
# tests/tracking/test_velocity_export.py
"""zero_velocity_if_unavailable is a public tracking seam (TF-60 PR2, ADR-063)."""

import pandas as pd
import pytest

import silly_kicks.tracking as tk


def test_is_exported():
    assert "zero_velocity_if_unavailable" in tk.__all__
    assert callable(tk.zero_velocity_if_unavailable)


def test_compute_gk_influence_is_exported():
    assert "compute_gk_influence" in tk.__all__
    assert callable(tk.compute_gk_influence)
    from silly_kicks.tracking import compute_gk_influence  # must not raise ImportError


def test_present_velocity_is_a_no_op_same_object():
    frame = pd.DataFrame(
        {"is_ball": [False], "team_id": [1], "player_id": [1], "x": [10.0], "y": [34.0],
         "vx": [0.0], "vy": [0.0], "is_goalkeeper": [False]}
    )
    assert tk.zero_velocity_if_unavailable(frame, method="spearman") is frame


def test_undeclared_missing_velocity_raises():
    frame = pd.DataFrame(
        {"is_ball": [False], "team_id": [1], "player_id": [1], "x": [10.0], "y": [34.0],
         "is_goalkeeper": [False]}  # no vx/vy, no velocity-unavailable marker
    )
    with pytest.raises((ValueError, KeyError)):
        tk.zero_velocity_if_unavailable(frame, method="spearman")
```

- [ ] **Step 2: Run — expect FAIL** (not in `__all__`)

Run: `python -m pytest tests/tracking/test_velocity_export.py -v --benchmark-skip`

- [ ] **Step 3: Implement** — in `silly_kicks/tracking/__init__.py`: (a) add `"zero_velocity_if_unavailable"` to `__all__` (near `validate_velocity_regime`) and `from ._velocity_availability import zero_velocity_if_unavailable` to the import section; (b) add `"compute_gk_influence"` to `__all__` (near the existing `"compute_pitch_control"` / `"compute_threat_pc"` entries) and `compute_gk_influence` to the `from ._gk_influence import add_gk_influence, ...` line (it's the same module `add_gk_influence` already imports from — just add the name). Match the existing per-module import grouping. Verify the exact source symbol name in `silly_kicks/tracking/_velocity_availability.py` (`zero_velocity_if_unavailable`) and that `_gk_influence.py` defines `compute_gk_influence`.

- [ ] **Step 4: Run — expect PASS.** Also confirm `test_undeclared_missing_velocity_raises` matches the seam's real exception type; adjust the `pytest.raises` tuple to the actual type if needed (read `_velocity_availability.py` to confirm).

---

## Task 4: Layer-2 column constants

**Files:**
- Modify: `silly_kicks/restdefense/_columns.py`, `silly_kicks/restdefense/__init__.py`
- Test: `tests/restdefense/test_columns.py` (new; or add to `test_compute.py`)

**Interfaces:**
- Produces: `RD_ATTACKER_SPACE_CONTROL`, `RD_DANGER_BEHIND_LINE`, `RD_DANGER_BEHIND_LINE_GK`, `RD_GK_COVERAGE_BEHIND_LINE`, `RD_GK_REACHABLE_COVERAGE_M2` name constants; `RD_LAYER2_COLUMNS` (list of the 5); `RD_METRIC_COLUMNS = [*RD_LAYER1_COLUMNS, *RD_LAYER2_COLUMNS]`. All exported from the package.

- [ ] **Step 1: Write the failing test**

```python
# tests/restdefense/test_columns.py
from silly_kicks.restdefense import RD_LAYER1_COLUMNS, RD_LAYER2_COLUMNS, RD_METRIC_COLUMNS


def test_layer2_columns():
    assert RD_LAYER2_COLUMNS == [
        "rd_attacker_space_control",
        "rd_danger_behind_line",
        "rd_danger_behind_line_gk",
        "rd_gk_coverage_behind_line",
        "rd_gk_reachable_coverage_m2",
    ]


def test_metric_columns_is_layer1_then_layer2_disjoint():
    assert RD_METRIC_COLUMNS == [*RD_LAYER1_COLUMNS, *RD_LAYER2_COLUMNS]
    assert set(RD_LAYER1_COLUMNS).isdisjoint(RD_LAYER2_COLUMNS)
```

- [ ] **Step 2: Run — expect FAIL** (ImportError)

- [ ] **Step 3: Implement** — in `_columns.py`, after the Layer-1 block add:

```python
# Layer-2 metric column bases (one per spec §7.2 row; TF-60 PR2).
RD_ATTACKER_SPACE_CONTROL = "rd_attacker_space_control"
RD_DANGER_BEHIND_LINE = "rd_danger_behind_line"
RD_DANGER_BEHIND_LINE_GK = "rd_danger_behind_line_gk"
RD_GK_COVERAGE_BEHIND_LINE = "rd_gk_coverage_behind_line"
RD_GK_REACHABLE_COVERAGE_M2 = "rd_gk_reachable_coverage_m2"

RD_LAYER2_COLUMNS = [
    RD_ATTACKER_SPACE_CONTROL,
    RD_DANGER_BEHIND_LINE,
    RD_DANGER_BEHIND_LINE_GK,
    RD_GK_COVERAGE_BEHIND_LINE,
    RD_GK_REACHABLE_COVERAGE_M2,
]

#: All emitted metric columns (Layer 1 + Layer 2); the canonical list every gate iterates.
RD_METRIC_COLUMNS = [*RD_LAYER1_COLUMNS, *RD_LAYER2_COLUMNS]
```

In `__init__.py`, add `RD_LAYER2_COLUMNS` and `RD_METRIC_COLUMNS` to the `from ._columns import ...` line and to `__all__`.

- [ ] **Step 4: Run — expect PASS.**

---

## Task 5: `WFieldParams` on `RestDefenseParams`

**Files:**
- Create: `silly_kicks/restdefense/_wfield.py` (the `WFieldParams` dataclass only in this task; `build_w_field` in Task 6)
- Modify: `silly_kicks/restdefense/_config.py` (add `w_field_params` field), `silly_kicks/restdefense/__init__.py` (export `WFieldParams`)
- Test: `tests/restdefense/test_config.py` (extend)

**Interfaces:**
- Produces: `WFieldParams` (frozen; fields `x_midpoint_m=30.0`, `x_steepness_m=8.0`, `y_center_m=34.0`, `y_sigma_m=20.0`); `RestDefenseParams.w_field_params: WFieldParams = WFieldParams()`.

- [ ] **Step 1: Write the failing test** (extend `tests/restdefense/test_config.py`)

```python
def test_w_field_params_defaults_and_frozen():
    from silly_kicks.restdefense import RestDefenseParams, WFieldParams

    p = RestDefenseParams()
    assert isinstance(p.w_field_params, WFieldParams)
    wp = p.w_field_params
    assert (wp.x_midpoint_m, wp.x_steepness_m, wp.y_center_m, wp.y_sigma_m) == (30.0, 8.0, 34.0, 20.0)
    import dataclasses
    import pytest
    with pytest.raises(dataclasses.FrozenInstanceError):
        wp.x_midpoint_m = 1.0


def test_adding_w_field_params_does_not_break_is_default_or_for_provider():
    from silly_kicks.restdefense import RestDefenseParams

    assert RestDefenseParams.default().is_default() is True
    assert RestDefenseParams().is_default() is False
    assert RestDefenseParams.for_provider("skillcorner") == RestDefenseParams()
```

- [ ] **Step 2: Run — expect FAIL** (ImportError / no attribute)

- [ ] **Step 3: Implement**

Create `silly_kicks/restdefense/_wfield.py`:

```python
"""OBPV field-value weighting for the deep-zone threat (TF-60 PR2, ADR-081).

Ogawa/Fujii et al. (2025) ("Space evaluation at the starting point of soccer transitions", OBPV)
weight space value as a LONGITUDINAL SIGMOID (in distance from the attacked goal) times a LATERAL
GAUSSIAN (in y), because a pure goal-proximity weighting misbehaves in the transition zone. This
module ships that FORM as an opt-in re-weighting of ``rd_danger_behind_line`` (gated by
``RestDefenseParams.danger_field_weight``). Defaults are un-tuned spec-time values (ADR-009); a
per-provider tune is a separate gated apply PR. See NOTICE (Ogawa 2025).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WFieldParams:
    """OBPV field-value weighting parameters (longitudinal sigmoid x lateral Gaussian)."""

    x_midpoint_m: float = 30.0   # sigmoid midpoint: distance from the defended goal G_A (m)
    x_steepness_m: float = 8.0   # sigmoid width (m)
    y_center_m: float = 34.0     # lateral Gaussian centre (pitch middle)
    y_sigma_m: float = 20.0      # lateral Gaussian width (m)
```

In `_config.py`: `from ._wfield import WFieldParams` and add the field to `RestDefenseParams` (after `danger_field_weight`, before `possession_stride` — note field ordering with defaults is fine since all have defaults):

```python
    w_field_params: WFieldParams = WFieldParams()
```

Because `WFieldParams` is frozen and hashable and `compare=True` by default, `for_provider`/`is_default`/`==` semantics are preserved (the existing doctests still hold). In `__init__.py`, export `WFieldParams` (add to the `from ._config import ...` line — re-export `WFieldParams` from `_config` — and `__all__`).

- [ ] **Step 4: Run — expect PASS.** Also run the existing config doctests: `python -m pytest --doctest-modules silly_kicks/restdefense/_config.py -q`.

---

## Task 6: `build_w_field` OBPV closure

**Files:**
- Modify: `silly_kicks/restdefense/_wfield.py` (add `build_w_field`)
- Test: `tests/restdefense/test_wfield.py`

**Interfaces:**
- Produces: `build_w_field(own_goal_x: float, params: WFieldParams) -> Callable[[np.ndarray, np.ndarray], np.ndarray]`. The returned `w(grid_x, grid_y)` yields an `(len(grid_y), len(grid_x))` array of weights in `(0, 1]`, in **absolute** pitch coords, oriented so weight is highest near `own_goal_x` (the defended goal `G_A`) and central in y.

- [ ] **Step 1: Write the failing tests**

```python
# tests/restdefense/test_wfield.py
"""OBPV w_field closure (TF-60 PR2): oriented toward G_A, sigmoid x Gaussian."""

import numpy as np

from silly_kicks.restdefense import WFieldParams
from silly_kicks.restdefense._wfield import build_w_field

_GX = np.linspace(0.0, 105.0, 50)
_GY = np.linspace(0.0, 68.0, 32)


def test_shape_and_range():
    w = build_w_field(own_goal_x=0.0, params=WFieldParams())(_GX, _GY)
    assert w.shape == (len(_GY), len(_GX))
    assert np.all((w > 0.0) & (w <= 1.0))


def test_high_near_defended_goal_low_far():
    w = build_w_field(own_goal_x=0.0, params=WFieldParams())(_GX, _GY)
    mid_y = len(_GY) // 2
    assert w[mid_y, 0] > w[mid_y, -1]  # near G_A=0 weighted above the far end


def test_orientation_flips_with_goal_end():
    lo = build_w_field(own_goal_x=0.0, params=WFieldParams())(_GX, _GY)
    hi = build_w_field(own_goal_x=105.0, params=WFieldParams())(_GX, _GY)
    mid_y = len(_GY) // 2
    assert lo[mid_y, 0] > lo[mid_y, -1]
    assert hi[mid_y, -1] > hi[mid_y, 0]  # mirror


def test_central_channel_weighted_above_wings():
    w = build_w_field(own_goal_x=0.0, params=WFieldParams())(_GX, _GY)
    assert w[len(_GY) // 2, 0] > w[0, 0]  # centre-y > touchline-y at the same x
```

- [ ] **Step 2: Run — expect FAIL** (no `build_w_field`)

- [ ] **Step 3: Implement** — append to `_wfield.py`:

```python
from collections.abc import Callable

import numpy as np


def build_w_field(own_goal_x: float, params: WFieldParams) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return ``w(grid_x, grid_y) -> (ny, nx)`` OBPV weights, oriented toward the defended goal G_A.

    Absolute pitch coords: row ``i`` <-> ``grid_y[i]``, col ``j`` <-> ``grid_x[j]``. Highest near
    ``own_goal_x`` (deep zone), central-channel-weighted in y. Values in ``(0, 1]``.
    """

    def w(grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
        d = np.abs(np.asarray(grid_x, dtype=float)[None, :] - own_goal_x)  # (1, nx) distance from G_A
        longitudinal = 1.0 / (1.0 + np.exp((d - params.x_midpoint_m) / params.x_steepness_m))  # (1, nx)
        y = np.asarray(grid_y, dtype=float)[:, None]  # (ny, 1)
        lateral = np.exp(-((y - params.y_center_m) ** 2) / (2.0 * params.y_sigma_m**2))  # (ny, 1)
        return lateral * longitudinal  # (ny, nx), broadcast

    return w
```

- [ ] **Step 4: Run — expect PASS.**

---

## Task 7: Layer-2 fixtures (velocity, fitted xt, keeper-sensitive)

**Files:**
- Modify: `tests/restdefense/_fixtures.py`
- Test: `tests/restdefense/test_fixtures_layer2.py` (a small self-check)

**Interfaces:**
- Produces: `make_fitted_xt() -> ExpectedThreat`; `make_keeper_sensitive_fixture() -> tuple[pd.DataFrame, pd.DataFrame, ExpectedThreat]` (actions, frames, xt) where team A (in possession) has a **rearguard line + a keeper alone in the deep zone**, and team B has a **counter-receiver broken in behind the rearguard** (so removing A's keeper measurably raises B's deep threat — the non-vacuity anchor). `make_rest_defense_fixture()`'s frames now carry `vx`/`vy`.

**Rationale (load-bearing — do not simplify):** `test_compute_threat_pc.py` documents that the keeper registers in the threat integral **only when it is the nearest defender to cells inside a dangerous receiver's Voronoi region** — a keeper screened by a nearer defender changes the threat by exactly 0.0 however far it moves. Rest defense normally *has* a rearguard screening the keeper, so `rd_danger_behind_line_gk ≈ rd_danger_behind_line` on a generic frame. The dedicated keeper-sensitive fixture must therefore place a B counter-receiver **behind** A's rearguard, in the zone only A's keeper covers.

- [ ] **Step 1: Write the failing self-check test**

```python
# tests/restdefense/test_fixtures_layer2.py
import numpy as np

from tests.restdefense._fixtures import (
    make_fitted_xt,
    make_keeper_sensitive_fixture,
    make_rest_defense_fixture,
)


def test_frames_carry_velocity():
    _actions, frames = make_rest_defense_fixture()
    assert {"vx", "vy"} <= set(frames.columns)


def test_fitted_xt_is_fitted():
    from silly_kicks.xthreat import require_fitted_xt

    require_fitted_xt(make_fitted_xt(), caller="test")  # does not raise


def test_keeper_sensitive_fixture_has_receiver_behind_A_rearguard():
    actions, frames, _xt = make_keeper_sensitive_fixture()
    f = frames[frames["frame_id"] == frames["frame_id"].iloc[0]]
    a_line = f[(f["team_id"] == 1) & ~f["is_goalkeeper"].astype(bool) & ~f["is_ball"].astype(bool)]["x"].nsmallest(4).max()
    a_gk = f[(f["team_id"] == 1) & f["is_goalkeeper"].astype(bool)]["x"].iloc[0]
    b_deep = f[(f["team_id"] == 2) & ~f["is_ball"].astype(bool)]["x"].min()
    assert a_gk < b_deep < a_line  # B receiver between A's keeper and A's back line
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement** — in `_fixtures.py`:

  1. Add `vx=0.0, vy=0.0` to the dicts returned by `_player(...)` and `_ball(...)`. Zero velocity is point-reflection-invariant (`-0.0 == 0.0`), so `make_rest_defense_fixture`'s a0/a2 stay exact mirrors; with `vx`/`vy` present the frames are **not** velocity-declared-absent, so `compute_pitch_control` runs (zero-velocity positional model) and `rd_gk_reachable_coverage_m2` computes a real number. After building `frames`, add `frames["vx"] = frames["vx"].astype("float64")` and likewise `vy` (they are already float via the dict).
  2. Add `make_fitted_xt()` (identical to the `fitted_xt` conftest fixture pattern):

```python
def make_fitted_xt():
    """A fitted ExpectedThreat (x-increasing grid), matching the tracking `fitted_xt` fixture."""
    import numpy as np

    from silly_kicks.xthreat import ExpectedThreat

    xt = ExpectedThreat(l=16, w=12)
    xt.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))
    return xt
```

  3. Add `make_keeper_sensitive_fixture()`. Team 1 (home, in possession, defends x=0, attacking right): back-4 at x≈[26,30] (a moderate line), keeper alone deep at x≈4, NO other team-1 player between the keeper and the back-4. Team 2 (counter): a receiver **broken in behind** at x≈14 (between the keeper x=4 and the back-4 x=26, y central ≈34), plus the rest of team 2 upfield. Ball with team 1 at x≈60 (committed forward). One frame, one in-possession action for team 1. Zero velocities. Return `(actions, frames, make_fitted_xt())`. Model the layout on `tests/tracking/test_compute_threat_pc.py`'s `_HOME_OUTFIELD`/`_AWAY_OUTFIELD` (deep zone left to the keeper), adapted so **team 1 (not team 2) is in possession** and the deep B receiver sits in team 1's keeper-only zone.

- [ ] **Step 4: Run — expect PASS.** Sanity: also assert `make_keeper_sensitive_fixture` frames carry `vx`/`vy`.

---

## Task 8: `_danger.py` — the five Layer-2 metrics

**Files:**
- Create: `silly_kicks/restdefense/_danger.py`
- Test: `tests/restdefense/test_danger.py`

**Interfaces:**
- Consumes: `SampleContext` (from `_structure.py`), `danger_zone_bounds` (from `_geometry.py`), `WFieldParams`/`build_w_field` (from `_wfield.py`), `compute_pitch_control`/`compute_threat_pc`/`compute_gk_influence`/`zero_velocity_if_unavailable`/`GoalEndUnresolvedError` (from `silly_kicks.tracking`), `ids_match` (from `id_compat`).
- Produces: `layer2_metrics(frame_rows, ctx, *, xt, goal_map, params, pitch_control_cache=None) -> dict[str, float]` (keys = `RD_LAYER2_COLUMNS`). Individual helpers `_gk_share_in_zone`, `_resolve_a_keeper_id`. `pitch_control_cache` (a `tracking.PitchControlCache`) is used for the **canonical** `#1`/`#4` surface (attacking=A, unmodified frame) via `cache.surface(...)` and passed through to `#5`'s `compute_gk_influence`; it is **never** used for `#2`/`#3` (they route through `compute_threat_pc`, which deliberately bypasses the cache — the ADR-043 moved-player trap, which is exactly what protects the keeper-removed base leg).

**Realizations (spec §7.2):**
- `rd_attacker_space_control` = `1 − surface(attacking=A).control_in_region(Z_lo, Z_hi, 0, 68)` (B's mean control-share of Z; keeper included). Z bounds from `danger_zone_bounds(ctx.defensive_line_x, ctx.own_goal_x, zone_depth_m)`.
- `rd_danger_behind_line` (**GK-blind base**) = `compute_threat_pc(frame_without_A_keeper, attacking=B, xt, goal_map, field_weight=w if danger_field_weight else None)`.
- `rd_danger_behind_line_gk` (**GK-included**) = `compute_threat_pc(frame_full, attacking=B, xt, goal_map, field_weight=…)`. (GK contribution `= base − gk` is derivable by the consumer, not a separate emitted column.)
- `rd_gk_coverage_behind_line` = the keeper's mean share of A's control over Z, from `compute_pitch_control(attacking=A, decompose=True)` (`share_grid = gk_influence / team_influence`, meaned over Z cells; the exact `_gk_influence.py:396` share_grid formula, restricted to Z). No `xt` (but Layer-2 gating still requires an xt to be *reached*).
- `rd_gk_reachable_coverage_m2` = `compute_gk_influence(frame, attacking=B, gk=A_keeper_id, xt, goal_map, region=(Z_lo, Z_hi, 0, 68)).reachable_area_m2`. Tier-2 → NaN inherited on velocity-declared-absent.

**Policies:** **`xt is None` → return all five NaN IMMEDIATELY, before any pitch-control call (P2-02, owner-approved).** Layer 2 IS the danger valuation; without a fitted `xt` it is not computed at all, so a Layer-1-only caller pays **zero** pitch-control cost and hits **no** velocity precondition (the `zero_velocity_if_unavailable` raise is only reached *with* an `xt` — i.e. only for a caller who opted into Layer 2 and forgot `derive_velocities()`; on SB360 the frames are declared velocity-less so it never raises). This also means `#1`/`#4` (which do not *use* `xt`) require an `xt` to be produced — the accepted cost of one opt-in signal, mitigated by the 2-line `make_fitted_xt`. — After the gate: route `frame` through `zero_velocity_if_unavailable(frame_rows, method="spearman")` once (for the `#1`/`#4` `compute_pitch_control` calls; `#2`/`#3`/`#5` seams re-apply it idempotently). Resolve `A_keeper_id` from `frame_rows` (`team_id==A ∧ is_goalkeeper`). **Keeper absent/unresolved → `#3`/`#4`/`#5` NaN** (`#2` still computes — it removes a keeper that isn't there). **Unresolvable Z (NaN `defensive_line_x`/`own_goal_x`) → all five NaN.** `GoalEndUnresolvedError` from a seam → the metric NaN (caught locally). `compute_threat_pc` on an unfitted `xt` raises — propagate (fail-closed). **`#2`/`#3` GK-inclusion note (CONSIDER):** GK-blind = drop A's keeper row; GK-included = keep it, which under the default `SpearmanParams(lambda_gk=3.0)` is *exactly* the spec's `SpearmanParams(lambda_gk)` framing (the surface scales the kept keeper by `lambda_gk`) — document this equivalence in ADR-081 (Task 18).

- [ ] **Step 1: Write the failing tests** (use the keeper-sensitive + generic fixtures)

```python
# tests/restdefense/test_danger.py
"""Layer-2 danger metrics (TF-60 PR2, ADR-081)."""

import math

import numpy as np
import pandas as pd

from silly_kicks.restdefense import RD_LAYER2_COLUMNS, RestDefenseParams
from silly_kicks.restdefense._danger import layer2_metrics
from silly_kicks.restdefense._structure import SampleContext
from silly_kicks.tracking import resolve_defended_goals
from tests.restdefense._fixtures import make_fitted_xt, make_keeper_sensitive_fixture


def _ctx_and_frame():
    actions, frames, xt = make_keeper_sensitive_fixture()
    gm = resolve_defended_goals(frames)
    fid = frames["frame_id"].iloc[0]
    frame_rows = frames[frames["frame_id"] == fid]
    ctx = SampleContext(
        team_id=1, opponent_id=2, ball_x=float(frame_rows[frame_rows["is_ball"]]["x"].iloc[0]),
        own_goal_x=0.0, attacked_goal_x=105.0, defensive_line_x=28.0,
        compactness_x=math.nan, lateral_width=math.nan, team_length=math.nan,
    )
    return frame_rows, ctx, gm, xt


def test_all_five_columns_finite_with_fitted_xt():
    frame_rows, ctx, gm, xt = _ctx_and_frame()
    m = layer2_metrics(frame_rows, ctx, xt=xt, goal_map=gm, params=RestDefenseParams())
    assert set(m) == set(RD_LAYER2_COLUMNS)
    for c in RD_LAYER2_COLUMNS:
        assert np.isfinite(m[c]), f"{c} is not finite"


def test_space_control_is_a_fraction():
    frame_rows, ctx, gm, xt = _ctx_and_frame()
    m = layer2_metrics(frame_rows, ctx, xt=xt, goal_map=gm, params=RestDefenseParams())
    assert 0.0 <= m["rd_attacker_space_control"] <= 1.0
    assert 0.0 <= m["rd_gk_coverage_behind_line"] <= 1.0


def test_all_columns_nan_without_xt():
    """P2-02: Layer 2 is gated ENTIRELY on xt -- no xt -> all five NaN, before any pitch-control call
    (a Layer-1-only caller pays nothing and hits no velocity precondition)."""
    frame_rows, ctx, gm, _xt = _ctx_and_frame()
    m = layer2_metrics(frame_rows, ctx, xt=None, goal_map=gm, params=RestDefenseParams())
    for c in RD_LAYER2_COLUMNS:
        assert math.isnan(m[c]), f"{c} should be NaN without xt (Layer-2 gate)"


def test_gk_dependent_columns_nan_when_A_keeper_absent():
    frame_rows, ctx, gm, xt = _ctx_and_frame()
    no_gk = frame_rows[~((frame_rows["team_id"] == 1) & frame_rows["is_goalkeeper"].astype(bool))]
    m = layer2_metrics(no_gk, ctx, xt=xt, goal_map=gm, params=RestDefenseParams())
    assert math.isnan(m["rd_danger_behind_line_gk"])
    assert math.isnan(m["rd_gk_coverage_behind_line"])
    assert math.isnan(m["rd_gk_reachable_coverage_m2"])
    assert np.isfinite(m["rd_danger_behind_line"])  # GK-blind base still computes


def test_unresolvable_zone_yields_all_nan():
    frame_rows, ctx, gm, xt = _ctx_and_frame()
    bad_ctx = SampleContext(
        team_id=1, opponent_id=2, ball_x=ctx.ball_x, own_goal_x=float("nan"),
        attacked_goal_x=105.0, defensive_line_x=float("nan"),
        compactness_x=float("nan"), lateral_width=float("nan"), team_length=float("nan"),
    )
    m = layer2_metrics(frame_rows, bad_ctx, xt=xt, goal_map=gm, params=RestDefenseParams())
    for c in RD_LAYER2_COLUMNS:
        assert math.isnan(m[c])
```

- [ ] **Step 2: Run — expect FAIL** (no `_danger`)

- [ ] **Step 3: Implement `_danger.py`** — key structure:

```python
"""Layer-2 danger-behind-line valuation (TF-60 PR2, ADR-081, spec §7.2)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from silly_kicks.id_compat import ids_match
from silly_kicks.tracking import (
    GoalEndUnresolvedError,
    compute_gk_influence,
    compute_pitch_control,
    compute_threat_pc,
    zero_velocity_if_unavailable,
)

from ._columns import (
    RD_ATTACKER_SPACE_CONTROL,
    RD_DANGER_BEHIND_LINE,
    RD_DANGER_BEHIND_LINE_GK,
    RD_GK_COVERAGE_BEHIND_LINE,
    RD_GK_REACHABLE_COVERAGE_M2,
)
from ._counting import bool_flag
from ._geometry import danger_zone_bounds
from ._wfield import build_w_field

_NAN = float("nan")
_SPEARMAN = "spearman"


def _zone(ctx, params):
    if not (math.isfinite(ctx.defensive_line_x) and math.isfinite(ctx.own_goal_x)):
        return None
    lo, hi = danger_zone_bounds(ctx.defensive_line_x, ctx.own_goal_x, zone_depth_m=params.zone_depth_m)
    return lo, hi


def _resolve_a_keeper_id(frame_rows, team_id):
    if "is_goalkeeper" not in frame_rows.columns:
        return None
    mask = ids_match(frame_rows["team_id"], team_id).to_numpy() & bool_flag(frame_rows["is_goalkeeper"])
    ids = frame_rows.loc[mask, "player_id"].dropna().unique()
    return ids[0] if len(ids) == 1 else None  # exactly one keeper, else unresolved


def layer2_metrics(frame_rows, ctx, *, xt, goal_map, params, pitch_control_cache=None):
    out = dict.fromkeys(
        (RD_ATTACKER_SPACE_CONTROL, RD_DANGER_BEHIND_LINE, RD_DANGER_BEHIND_LINE_GK,
         RD_GK_COVERAGE_BEHIND_LINE, RD_GK_REACHABLE_COVERAGE_M2),
        _NAN,
    )
    # P2-02 (owner-approved): Layer 2 is the danger valuation -- gated ENTIRELY on a fitted xt. Without
    # one, return all-NaN before any pitch-control call, so a Layer-1-only caller pays no cost and hits
    # no velocity precondition. Everything below is reached ONLY with an xt.
    if xt is None:
        return out
    zone = _zone(ctx, params)
    if zone is None or pd.isna(ctx.opponent_id) or pd.isna(ctx.team_id):
        return out  # unresolvable geometry / no opponent -> all NaN
    lo, hi = zone
    frame = zero_velocity_if_unavailable(frame_rows, method=_SPEARMAN)
    a_keeper = _resolve_a_keeper_id(frame, ctx.team_id)

    # #1 space control + #4 gk coverage share (ONE canonical decompose surface for team A).
    # Cacheable: the frame is UNMODIFIED and attacking=A is canonical, so cache.surface (keyed on
    # frame_id) is correct here -- unlike the keeper-removed #2 leg, which must never touch the cache.
    try:
        surf_a = (
            pitch_control_cache.surface(frame, ctx.team_id, method=_SPEARMAN, decompose=True)
            if pitch_control_cache is not None
            else compute_pitch_control(frame, ctx.team_id, method=_SPEARMAN, decompose=True)
        )
        out[RD_ATTACKER_SPACE_CONTROL] = 1.0 - surf_a.control_in_region(lo, hi, 0.0, 68.0)
        if a_keeper is not None:
            out[RD_GK_COVERAGE_BEHIND_LINE] = _gk_share_in_zone(surf_a, a_keeper, ctx.team_id, lo, hi)
    except (GoalEndUnresolvedError, ValueError):
        pass

    # #2 GK-blind + #3 GK-included danger
    w = build_w_field(ctx.own_goal_x, params.w_field_params) if params.danger_field_weight else None
    try:
        out[RD_DANGER_BEHIND_LINE_GK] = compute_threat_pc(
            frame, attacking_team_id=ctx.opponent_id, xt=xt, goal_map=goal_map, field_weight=w
        ) if a_keeper is not None else _NAN
        frame_no_gk = frame[~ids_match(frame["player_id"], a_keeper).to_numpy()] if a_keeper is not None else frame
        out[RD_DANGER_BEHIND_LINE] = compute_threat_pc(
            frame_no_gk, attacking_team_id=ctx.opponent_id, xt=xt, goal_map=goal_map, field_weight=w
        )
    except GoalEndUnresolvedError:
        pass
    # #5 reachable ∩ Z (needs xt for the compute_gk_influence seam; the reachable value itself ignores xt)
    if a_keeper is not None:
        try:
            out[RD_GK_REACHABLE_COVERAGE_M2] = compute_gk_influence(
                frame, attacking_team_id=ctx.opponent_id, gk_player_id=a_keeper,
                xt=xt, goal_map=goal_map, region=(lo, hi, 0.0, 68.0),
                pitch_control_cache=pitch_control_cache,
            ).reachable_area_m2
        except (GoalEndUnresolvedError, ValueError):
            pass
    return out


def _gk_share_in_zone(surface, gk_id, team_id, lo, hi):
    """Keeper's mean share of team A's per-cell influence over the Z x-band -- MIRRORS the TF-15
    ``compute_gk_influence`` share_grid (``_gk_influence.py:396``), restricted to Z. NaN if the
    decompose fields are missing. DRIFT NOTE: this duplicates that share formula; if TF-15's
    per-cell share changes, update here too (the reused-region alternative -- a region-share field on
    compute_gk_influence -- was declined to keep the tracking blast radius to the `region` param)."""
    if surface.per_player_influence is None or surface.player_ids is None or surface.player_team_ids is None:
        return _NAN
    gk_surface = surface.player_surface(gk_id)  # (ny, nx)
    # ADR-019: team_id is ACTION-sourced (ctx.team_id) while player_team_ids is FRAME-sourced -> a
    # CROSS-source compare (the numeric-actions x string-frames case Task 11 tests). Route via ids_match.
    team_mask = ids_match(surface.player_team_ids, team_id).to_numpy()
    team_surface = surface.per_player_influence[np.flatnonzero(team_mask)].sum(axis=0)
    safe = np.where(team_surface < 1e-8, np.inf, team_surface)
    share = np.where(team_surface < 1e-8, 0.0, gk_surface / safe)  # (ny, nx)
    xmask = (surface.grid_x >= lo) & (surface.grid_x <= hi)
    region = share[:, xmask]
    return float(region.mean()) if region.size else _NAN
```

Notes for the implementer:
- `compute_pitch_control(frame, ctx.team_id, ...)` — `attacking_team_id` is positional-2 (see its signature); pass the team id, not a keyword.
- **P2-03:** `_gk_share_in_zone` compares `surface.player_team_ids` (FRAME-sourced) against `ctx.team_id` (ACTION-sourced) — a **cross-source** id compare (unlike `_gk_influence.py:396`, whose scalar `defending_team_id` is drawn from the frame's own `gk_row`, so that one is genuinely same-source and stays raw). Route restdefense's compare through `ids_match` (already in the code above), or Task 11's numeric-actions × string-frames invariance fails.
- `field_weight=w` is `None` unless `danger_field_weight` — so with the default params, `#2`/`#3` are byte-identical to a plain `compute_threat_pc`.
- Wrap the seam calls so a `GoalEndUnresolvedError` (unresolvable end inside a seam) degrades that metric to NaN rather than crashing the whole sample (ADR-055 policy-at-the-edge). Let an **unfitted-xt** `IntegrityError`/`ValueError` from `require_fitted_xt` **propagate** (fail-closed) — do NOT catch it (only catch `GoalEndUnresolvedError` for the danger calls; catch `ValueError` only around `compute_gk_influence`'s "gk not found"/`compute_pitch_control` velocity paths where NaN is the honest answer — verify the exact exception types by reading the seams, and keep the catch as narrow as possible).

- [ ] **Step 4: Run — expect PASS** (`python -m pytest tests/restdefense/test_danger.py -v`).

---

## Task 9: Wire Layer 2 into `compute_rest_defense`

**Files:**
- Modify: `silly_kicks/restdefense/_compute.py`
- Test: `tests/restdefense/test_compute.py` (extend)

**Interfaces:**
- Consumes: `layer2_metrics` (Task 8).
- Produces: `compute_rest_defense(actions, frames, *, xt=None, goal_map=None, links=None, pitch_control_cache=None, visible_area=None, params=…)` — samples table now carries `RD_METRIC_COLUMNS` (Layer 1 + Layer 2) between `is_possession_loss` and `rd_geometry_source`. `xt` and `pitch_control_cache` keyword-only, default None (the spec §12 / ADR-078 pipeline-reuse convention).

- [ ] **Step 1: Write the failing tests** (extend `test_compute.py`)

```python
def test_layer2_columns_present_and_typed():
    from silly_kicks.restdefense import RD_LAYER2_COLUMNS
    from tests.restdefense._fixtures import make_fitted_xt

    actions, frames = make_rest_defense_fixture()
    samples, _ = compute_rest_defense(actions, frames, xt=make_fitted_xt())
    for c in RD_LAYER2_COLUMNS:
        assert c in samples.columns
        assert str(samples[c].dtype) == "float64"
    resolved = samples[samples["rd_geometry_source"] == "resolved"]
    for c in RD_LAYER2_COLUMNS:
        assert resolved[c].notna().any(), f"{c} all-NaN on the fixture with a fitted xt"


def test_layer2_all_nan_without_xt_layer1_preserved():
    from silly_kicks.restdefense import RD_LAYER2_COLUMNS

    actions, frames = make_rest_defense_fixture()
    samples, _ = compute_rest_defense(actions, frames)  # no xt -> Layer 2 gated off entirely (P2-02)
    resolved = samples[samples["rd_geometry_source"] == "resolved"]
    for c in RD_LAYER2_COLUMNS:
        assert resolved[c].isna().all(), f"{c} should be all-NaN without xt (Layer-2 gate)"
    assert resolved["rd_num_superiority"].notna().any()  # Layer 1 unchanged from PR1


def test_output_schema_includes_layer2():
    from silly_kicks.restdefense import RD_METRIC_COLUMNS
    from silly_kicks.restdefense._columns import RD_GEOMETRY_SOURCE, RD_SAMPLE_KEYS

    actions, frames = make_rest_defense_fixture()
    samples, _ = compute_rest_defense(actions, frames, xt=None)
    expected = [*RD_SAMPLE_KEYS, "possession_id", "is_possession_loss", *RD_METRIC_COLUMNS, RD_GEOMETRY_SOURCE]
    assert list(samples.columns) == expected
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement** — in `_compute.py`:
  - Add `xt=None` and `pitch_control_cache=None` to `compute_rest_defense`'s keyword-only params (`xt` first per spec §12 ordering; both keyword-only either way).
  - Change `_FLOAT_METRIC_COLS`/`_OUTPUT_COLS` to build over `RD_METRIC_COLUMNS` (import it): `_OUTPUT_COLS = [*RD_SAMPLE_KEYS, *_SAMPLE_META, *RD_METRIC_COLUMNS, RD_GEOMETRY_SOURCE]`; `_FLOAT_METRIC_COLS = [c for c in RD_METRIC_COLUMNS if c not in _COUNT_COLS and c != RD_SHAPE_STAGGER]` (the 5 Layer-2 columns are all floats → they join `_FLOAT_METRIC_COLS`, so the existing `pd.to_numeric(...).astype("float64")` loop types them).
  - In `_score_samples`, thread `xt` and `goal_map` through, and in the scored branch merge Layer 2:

```python
    m = layer1_metrics(frame_rows, ctx, params=params)
    m.update(layer2_metrics(frame_rows, ctx, xt=xt, goal_map=goal_map, params=params,
                            pitch_control_cache=pitch_control_cache))
    m[RD_GEOMETRY_SOURCE] = gsrc.get(key, "resolved")
```

    and in the unresolved branch, initialise the Layer-2 keys to `pd.NA` too: `m = {c: pd.NA for c in RD_METRIC_COLUMNS}`.
  - Pass `xt=xt, goal_map=goal_map, pitch_control_cache=pitch_control_cache` from `compute_rest_defense` into `_score_samples` (add them to its signature). `_score_samples` already builds `groups` once; the cache is a fresh local `PitchControlCache()` only if a pipeline caller does not supply one — but do **not** create a default cache inside `layer2_metrics` (its `#2` keeper-removed frame must never be cache-keyed); leave `pitch_control_cache=None` as "no caching" and let a pipeline caller opt in.
  - `layer2_metrics` needs the full frame slice per sample — it already has `frame_rows = groups.get(...)`. **No new `group_rows` pass** → no new ADR-073 `SCALE_GUARDED` entry (confirm: `_score_samples` still builds `groups` once; Layer 2 reuses the same `frame_rows`).

- [ ] **Step 4: Run — expect PASS** (`python -m pytest tests/restdefense/test_compute.py -v`).

- [ ] **Step 5: Update the existing "fully populated" test.** `test_scored_rows_are_fully_populated_and_pinned` iterates `RD_LAYER1_COLUMNS` and asserts non-null — it must stay on `RD_LAYER1_COLUMNS` (Layer-2 needs xt, which that test does not pass) OR pass a fitted xt and switch to `RD_METRIC_COLUMNS`. Choose the latter for full coverage: add `xt=make_fitted_xt()` to its `compute_rest_defense` call and switch its loop to `RD_METRIC_COLUMNS`. Keep the hard-coded Layer-1 value assertions (`a0["rd_num_superiority"] == 4`, etc.) unchanged.

---

## Task 10: FOV companions for the Layer-2 region columns

**Files:**
- Modify: `silly_kicks/restdefense/_fov.py`
- Test: `tests/restdefense/test_fov_completeness.py` (passes automatically once the partition is extended), `tests/restdefense/test_fov_companions.py` (extend if present)

**Interfaces:**
- Produces: `<col>_observed_fraction` / `_observed_source` for the Layer-2 **region** columns, all keyed on the **danger-zone (Z) ROI** (`gr_line`-based, action-LTR), matching the existing `_ZONE_COLUMNS` machinery.

**Decision (spec §11.2):** `rd_attacker_space_control`, `rd_gk_coverage_behind_line`, `rd_gk_reachable_coverage_m2` are literally region-`Z` metrics → **companion them with the zone ROI**. `rd_danger_behind_line` / `rd_danger_behind_line_gk` are whole-pitch xT-concentrated integrals whose FOV-sensitivity is B-receiver cropping in the deep zone → **also companion them with the zone ROI** (the honest "was the counter-danger zone observed" flag; they are more FOV-sensitive than the exempt position metrics). So **all five Layer-2 columns are FOV-sensitive, zone-ROI**.

- [ ] **Step 1: Extend the completeness test's expectation** — `test_partition_is_exact_and_disjoint` already asserts `FOV_SENSITIVE_COLUMNS ∪ _OBSERVABILITY_EXEMPT == set(RD_LAYER1_COLUMNS)`. Change its RHS to `RD_METRIC_COLUMNS` (import it) so the partition covers Layer 2. Run — expect FAIL (Layer-2 columns uncovered).

- [ ] **Step 2: Implement** — in `_fov.py`:
  - Import the 5 Layer-2 name constants and `RD_METRIC_COLUMNS`.
  - Extend `_ZONE_COLUMNS` to include all five Layer-2 columns: `_ZONE_COLUMNS = (RD_ZONE_OCCUPANCY, RD_ATTACKER_SPACE_CONTROL, RD_DANGER_BEHIND_LINE, RD_DANGER_BEHIND_LINE_GK, RD_GK_COVERAGE_BEHIND_LINE, RD_GK_REACHABLE_COVERAGE_M2)`. `FOV_SENSITIVE_COLUMNS = (*_BAND_COLUMNS, *_ZONE_COLUMNS)` picks them up; `append_fov_companions` already maps `_ZONE_COLUMNS → zone_far`, so no per-column code change is needed.
  - The completeness gate now needs `FOV_SENSITIVE_COLUMNS ∪ _OBSERVABILITY_EXEMPT == set(RD_METRIC_COLUMNS)`: the 5 Layer-2 columns are all in `FOV_SENSITIVE_COLUMNS`, and `_OBSERVABILITY_EXEMPT` is unchanged (Layer-1 position/shape) → the partition holds.

- [ ] **Step 3: Run — expect PASS** (`python -m pytest tests/restdefense/test_fov_completeness.py -v`). Also confirm `test_every_sensitive_column_is_companioned_when_visible_area_supplied` covers the new columns (it iterates `FOV_SENSITIVE_COLUMNS`).

---

## Task 11: Extend liveness / orientation / id-dtype gates to `RD_METRIC_COLUMNS`

**Files:**
- Modify: `tests/restdefense/test_liveness.py`, `tests/restdefense/test_orientation.py`, `tests/restdefense/test_id_dtype_invariance.py`
- Test: those same files.

- [ ] **Step 1: liveness** — in `test_liveness.py`: switch imports/iteration from `RD_LAYER1_COLUMNS` to `RD_METRIC_COLUMNS`; pass `xt=make_fitted_xt()` in `_resolved()` (so Layer-2 columns are non-null). The 5 Layer-2 columns are floats → they enter `_FLOAT_METRIC_COLS` (non-constant check). With the extended fixture (a0/a2 mirror, a1 variant) they vary → non-constant passes. `rd_gk_reachable_coverage_m2` is non-null because the fixture carries `vx=vy=0` (present → not velocity-declared-absent → real zero-velocity reachable area). Run — expect PASS.

- [ ] **Step 2: orientation** — in `test_orientation.py`: switch `_NUMERIC` to derive from `RD_METRIC_COLUMNS`; pass `xt=make_fitted_xt()` in both `compute_rest_defense` calls of both tests; add the direction-dependent Layer-2 columns to `_GATE_C_MUST_MOVE` (add `rd_attacker_space_control` and `rd_gk_coverage_behind_line` — both flip when the map swaps because the zone Z flips ends). The point-reflection invariance test then covers all numeric Layer-2 columns automatically (a0/a2 mirror, zero velocity is reflection-invariant). Run — expect PASS. **If a Layer-2 column does not move under the swap** (e.g. a symmetric-fixture coincidence), pick a different direction-dependent column for `_GATE_C_MUST_MOVE` and record why in a comment — do not weaken to a bare `moved > 0` (CLAUDE.md Gate-C completeness rule).

- [ ] **Step 3: id-dtype** — in `test_id_dtype_invariance.py`: switch `_NUMERIC` to derive from `RD_METRIC_COLUMNS`; pass `xt=make_fitted_xt()` in the three `compute_rest_defense` calls. The Layer-2 pitch-control metrics must be dtype-invariant (they route ids through `id_compat` in the seams). Run — expect PASS. **If a Layer-2 column differs across the numeric/string id axes**, the cause is a raw id compare in `_danger.py` — fix it there (route through `ids_match`), never relax the test.

---

## Task 12: Non-vacuity gates (GK-blind ≠ GK-included; `field_weight` moves the value)

**Files:**
- Create: `tests/restdefense/test_non_vacuity.py`
- Test: same.

**Interfaces:** consumes `make_keeper_sensitive_fixture` (Task 7), `layer2_metrics` (Task 8).

- [ ] **Step 1: Write the tests**

```python
# tests/restdefense/test_non_vacuity.py
"""Non-vacuity: the GK-inclusion and w_field re-weighting measurably move the danger (CLAUDE.md)."""

import numpy as np

from silly_kicks.restdefense import RestDefenseParams
from silly_kicks.restdefense._danger import layer2_metrics
from silly_kicks.restdefense._structure import SampleContext
from silly_kicks.tracking import resolve_defended_goals
from tests.restdefense._fixtures import make_keeper_sensitive_fixture


def _ctx(frame_rows):
    return SampleContext(
        team_id=1, opponent_id=2, ball_x=float(frame_rows[frame_rows["is_ball"]]["x"].iloc[0]),
        own_goal_x=0.0, attacked_goal_x=105.0, defensive_line_x=28.0,
        compactness_x=float("nan"), lateral_width=float("nan"), team_length=float("nan"),
    )


def test_gk_inclusion_measurably_reduces_deep_danger():
    """On a keeper-sensitive frame (a B receiver in behind A's rearguard), including A's keeper as a
    control agent LOWERS B's deep danger: base (GK-blind) > gk (GK-included), and by a real margin."""
    actions, frames, xt = make_keeper_sensitive_fixture()
    fr = frames[frames["frame_id"] == frames["frame_id"].iloc[0]]
    m = layer2_metrics(fr, _ctx(fr), xt=xt, goal_map=resolve_defended_goals(frames), params=RestDefenseParams())
    base, gk = m["rd_danger_behind_line"], m["rd_danger_behind_line_gk"]
    assert np.isfinite(base) and np.isfinite(gk)
    assert base > gk, f"keeper did not deter: base={base} gk={gk}"
    assert (base - gk) > 1e-6, "GK contribution is vacuously ~0 -- keeper is screened; fix the fixture"


def test_w_field_measurably_changes_the_danger():
    actions, frames, xt = make_keeper_sensitive_fixture()
    fr = frames[frames["frame_id"] == frames["frame_id"].iloc[0]]
    ctx, gm = _ctx(fr), resolve_defended_goals(frames)
    off = layer2_metrics(fr, ctx, xt=xt, goal_map=gm, params=RestDefenseParams(danger_field_weight=False))
    on = layer2_metrics(fr, ctx, xt=xt, goal_map=gm, params=RestDefenseParams(danger_field_weight=True))
    assert off["rd_danger_behind_line"] != on["rd_danger_behind_line"]
    # w_field weights in (0,1] -> it can only down-weight; the deep zone stays a real fraction of it.
    assert 0.0 < on["rd_danger_behind_line"] <= off["rd_danger_behind_line"] + 1e-9
```

- [ ] **Step 2: Run — expect FAIL first** (import), then after Task 8 exists, run and confirm `base > gk` by a real margin. **If `base == gk`**, the keeper is screened → strengthen `make_keeper_sensitive_fixture` (move the B receiver deeper, into the keeper's exclusive Voronoi zone; remove any A outfielder between the keeper and the receiver), per the `test_compute_threat_pc.py` geometry lesson. Iterate the fixture until the margin is real — this is the anti-vacuity work, not a test to relax.

- [ ] **Step 3: Run — expect PASS.**

---

## Task 13: Purity — the `danger_field_weight` and `xt` branches

**Files:**
- Modify: `tests/restdefense/test_purity.py`
- Test: same.

- [ ] **Step 1: Add tests** — the ADR-033 "≥2 variants for a value-changing/conditional param" rule: add a variant that calls `compute_rest_defense` with `danger_field_weight=True` and one with a fitted `xt`, asserting neither mutates `actions`/`frames`/`visible_area` and the polygon ndarray is untouched. Mirror the existing `test_compute_is_pure_with_visible_area_and_polygon_arrays` pattern:

```python
def test_compute_is_pure_with_xt_and_field_weight():
    from silly_kicks.restdefense import RestDefenseParams
    from tests.restdefense._fixtures import make_fitted_xt

    actions, frames = make_rest_defense_fixture()
    a_before, f_before = actions.copy(), frames.copy()
    compute_rest_defense(actions, frames, xt=make_fitted_xt(), params=RestDefenseParams(danger_field_weight=True))
    pd.testing.assert_frame_equal(actions, a_before)
    pd.testing.assert_frame_equal(frames, f_before)
```

- [ ] **Step 2: Run — expect PASS** (the implementation copies before mutating; `layer2_metrics` builds `frame_no_gk` via boolean indexing which returns a new frame, and `zero_velocity_if_unavailable` returns the same object only when it does not mutate). **If it fails**, the culprit is `zero_velocity_if_unavailable` returning a view that a downstream seam mutates, or a `frame[...]` slice being written to — fix in `_danger.py` (never mutate `frame_rows`), not in the test.

---

## Task 14: SB360 boundary audit for the Layer-2 columns

**Files:**
- Modify: `tests/sb360/_entries/_boundary.py` (`_call_rest_defense` + the `restdefense.compute_rest_defense` `_entry`)
- Test: the SB360 audit suite (`python -m pytest tests/sb360/ -v`)

**Interfaces:** the boundary entry is **hand-maintained** (the regenerate/adjudicate scripts loop `tracking.__all__` and never touch it). Observations are **transcribed from execution**; adjudications + rationales are author-asserted.

- [ ] **Step 1: Thread a fitted `xt` into `_call_rest_defense`** so the Layer-2 threat columns are actually exercised (without `xt`, `#2`/`#3`/`#5` are all-NaN → the audit says nothing about their velocity behaviour). Build a fitted xt the same way as the tracking `fitted_xt` fixture (`ExpectedThreat(l=16, w=12); xt.xT = np.tile(np.linspace(0,1,16),(12,1))`), or reuse whatever fitted xt the gkdv/xtgk boundary entries in this same file already construct (check `_call_xt_gk_v2` for a shared helper). Change the select list from `RD_LAYER1_COLUMNS` to `RD_METRIC_COLUMNS`:

```python
    from silly_kicks.restdefense import RD_METRIC_COLUMNS, compute_rest_defense
    samples, _ = compute_rest_defense(actions, frames, links=links, xt=_fitted_xt())
    cols = ["action_id", *RD_METRIC_COLUMNS, "rd_geometry_source"]
```

- [ ] **Step 2: Enumerate the 5 Layer-2 columns in the `_entry`** with explicit `AxisVerdict`s. The Layer-1 columns keep `AxisVerdict("identical", _WORKS)`. For the Layer-2 columns, the **decision rule** (adjudication is author-asserted; the observation is measured):
  - `rd_attacker_space_control`, `rd_danger_behind_line`, `rd_danger_behind_line_gk`, `rd_gk_coverage_behind_line` are **pitch-control positional metrics**: on the velocity-less Leg A they use the zero-velocity model, on the velocity-bearing Leg B the full model → they **differ by design** (ADR-063 Tier-1 lift — a valid positional model, not a fabrication). Adjudication `differs_by_design` (rationale ALWAYS required). *Unless* Leg B's velocities are also zero at these positions, in which case the observation is `identical` → `works`. **Transcribe the measured observation** and pick the matching adjudication.
  - `rd_gk_reachable_coverage_m2` is **Tier-2**: NaN on velocity-less Leg A, a value on Leg B → observation `partial_nan` (or `all_nan` if Leg A only) → adjudication `honest_nan` (rationale required if `partial_nan`).

  Write the `velocity=` and `visibility=` dicts so each Layer-2 column carries its verdict. Because `_RD_ALL = tuple(RD_LAYER1_COLUMNS)` currently drives the comprehensions, either (a) redefine `_RD_ALL = tuple(RD_METRIC_COLUMNS)` and add Layer-2 columns to the special-case sets, or (b) keep the Layer-1 comprehension and **add explicit per-column entries** for the 5 Layer-2 columns merged into the `velocity`/`visibility` dicts. Prefer (b) for clarity (Layer-2 verdicts differ from the Layer-1 `works` default). Provide rationale constants like the existing `_RD_PROVENANCE_RATIONALE`.

- [ ] **Step 3: Set `verdict_provenance`.** The composite already declares `verdict_provenance="structural"`. Per the ADR-053 amendment (4.88.0), the frame-blind `works` half is `structural`, the observationally-ambiguous `honest_nan` half author-asserts. Keep `verdict_provenance="structural"` for the entry (it is one entry); ensure each `differs_by_design`/`honest_nan` slot carries its mandatory rationale.

- [ ] **Step 4: Run the audit, transcribe, correct.** Run `python -m pytest tests/sb360/ -v`. The audit re-derives each column's observation from execution and asserts it matches the `AxisVerdict.observation` you wrote. **Where your guessed observation is wrong, the test prints the measured value — transcribe it verbatim into the `AxisVerdict`** (this is the SB360 discipline: observations are measured, not asserted). Keep iterating until green. If any Layer-2 column lands `no_signal`/`not_exercised` on additional rosters, raise `NOT_EXERCISED_BUDGET` in `_registry.py` with a note in the running-commentary style (as its lines 225–231 already do for the GK columns).

- [ ] **Step 5: Confirm the audited-surface gate** still passes (`restdefense.compute_rest_defense` is admitted via `BOUNDARY_ENTRY_POINTS`; the Layer-2 columns ride the same entry — no `audited_surface` exemption change needed). Run `python -m pytest tests/sb360/test_boundary_adapters.py tests/sb360/test_harness.py -v`.

---

## Task 15: Feature glossary entries

**Files:**
- Modify: `silly_kicks/feature_glossary.py`
- Test: the glossary coverage gate (part of the suite) + a targeted check.

**Interfaces:** each Layer-2 column gets a `FeatureColumn(name, definition, unit, emitting_module=_M_RESTDEFENSE, attribution=…, higher_is_better=…)` in the `_register(...)` call; a new `_A_NOVILLO_2025 = "Novillo et al. (2025)"` attribution constant.

- [ ] **Step 1: Add `_A_NOVILLO_2025`** near the other `_A_*` constants (lines 142–160): `_A_NOVILLO_2025 = "Novillo et al. (2025)"  # λ_GK-included control behind the line (TF-60 PR2)`.

- [ ] **Step 2: Add 5 `FeatureColumn` entries** immediately after the Layer-1 rd_* block (after line 1762), inside the `_register(...)` arg list, in `RD_LAYER2_COLUMNS` order. Attribution/`higher_is_better` per spec §7.2:

```python
    FeatureColumn(
        name="rd_attacker_space_control",
        definition=(
            "Opponent (counter-attacker) team's pitch-control share of the danger-behind-the-line "
            "zone Z (between the in-possession team's rearguard line and its own goal)."
        ),
        unit="ratio",  # a [0,1] pitch-control share (P2-04)
        emitting_module=_M_RESTDEFENSE,
        attribution=_A_FORCHER_2023,
        higher_is_better=False,  # more opponent control of the rest-defense zone = worse for the defender
    ),
    FeatureColumn(
        name="rd_danger_behind_line",
        definition=(
            "Threat-weighted counter-danger of zone Z: the xT-toward-own-goal-weighted pitch-control "
            "threat of the opponent's dangerous receivers, with the in-possession keeper EXCLUDED "
            "from control (GK-blind)."
        ),
        unit="dimensionless",
        emitting_module=_M_RESTDEFENSE,
        attribution=_A_NOVILLO_2025,
        higher_is_better=False,
    ),
    FeatureColumn(
        name="rd_danger_behind_line_gk",
        definition=(
            "As rd_danger_behind_line but with the in-possession keeper INCLUDED as a control agent "
            "(lambda_gk); the keeper's deterrent contribution is rd_danger_behind_line minus this."
        ),
        unit="dimensionless",
        emitting_module=_M_RESTDEFENSE,
        attribution=_A_NOVILLO_2025,
        higher_is_better=False,
    ),
    FeatureColumn(
        name="rd_gk_coverage_behind_line",
        definition="In-possession keeper's mean share of its team's pitch control over the danger zone Z.",
        unit="ratio",  # a [0,1] share (P2-04)
        emitting_module=_M_RESTDEFENSE,
        higher_is_better=True,
    ),
    FeatureColumn(
        name="rd_gk_reachable_coverage_m2",
        definition=(
            "Area (m^2) of zone Z the in-possession keeper can reach before any defender can "
            "(TF-15 reachable-area form); honest-NaN on velocity-less providers."
        ),
        unit="m^2",  # the Unit Literal's area token (verified feature_glossary.py:25; P2-04)
        emitting_module=_M_RESTDEFENSE,
        higher_is_better=True,
    ),
```

- [ ] **Step 3: Unit tokens (verified against the `Unit` Literal, `feature_glossary.py:23–36`).** Valid tokens include `"m^2"`, `"ratio"`, `"dimensionless"`, `"metres"`, `"count"`. Use: `#5` → `"m^2"` (`"square_metres"` is NOT in the Literal → would fail); `#1`/`#4` → `"ratio"` (genuine [0,1] shares); `#2`/`#3` → `"dimensionless"` (they are `compute_threat_pc` integrals — a sum of xT×control over many cells, unbounded, NOT a [0,1] fraction, so `ratio` would be wrong for them). This is the P2-04 fix, refined: only the two true fractions become `ratio`.

- [ ] **Step 4: Run the coverage gate** — `python -m pytest tests/ -k "glossary" -v` (find the exact test name). Expected: PASS (all `add_*`/`compute_*` emitted columns documented). The notice-linkage gate (ADR-005) also runs here: `_A_NOVILLO_2025 = "Novillo et al. (2025)"` must appear verbatim in `NOTICE` (Task 16) or this gate fails — order Task 16 before re-running.

---

## Task 16: NOTICE references (Ogawa 2025, Novillo 2025)

**Files:**
- Modify: `NOTICE`
- Test: the notice-linkage gate (runs with the glossary gate).

- [ ] **Step 1: Add two entries** to the "Mathematical / Methodological References" section, in the existing bulleted style (`- Author (year). "Title." Venue.` + `Used by:` line + prose). Place them near the other TF-60 refs (lines ~250–308):

```
- Novillo et al. (2025). "Offside Control." Chaos, Solitons & Fractals
  197:116445.
  Used by: silly_kicks.restdefense._danger (TF-60 PR2).
  Differentiates the goalkeeper inside a pitch-control danger model via a
  higher GK control rate (lambda_GK ~ 3x outfield) and measures control behind
  the opponent's offside/last line -- the anchor for rd_danger_behind_line and
  its GK-inclusive variant rd_danger_behind_line_gk.

- Ogawa et al. (2025). "Space evaluation at the starting point of soccer
  transitions." arXiv:2505.14711 (OBPV).
  Used by: silly_kicks.restdefense._wfield (TF-60 PR2).
  A transition-zone field-value weighting (longitudinal sigmoid x lateral
  Gaussian) adopted as the opt-in w_field re-weighting of the deep-zone threat
  (RestDefenseParams.danger_field_weight; off by default, un-tuned per ADR-009).
```

  The `Novillo et al. (2025)` string must match `_A_NOVILLO_2025` verbatim (the notice-linkage gate). Ogawa is cited for the `_wfield` module (no column attribution token needed — NOTICE may carry extra references).

- [ ] **Step 2: (optional) extend Shaw & Sudarshan's `Used by:`** to mention `restdefense` (`lambda_gk`) — only if it reads cleanly; not required by any gate.

- [ ] **Step 3: Run the notice-linkage + glossary gates** — expect PASS.

---

## Task 17: e2e method gate for Layer 2

**Files:**
- Modify: `tests/restdefense/test_e2e_method.py`
- Test: `python -m pytest tests/restdefense/test_e2e_method.py -m e2e -v` (owner/fixture-gated; self-skips without data).

- [ ] **Step 1: Thread a fitted xt + assert Layer-2 method.** Add `xt=make_fitted_xt()` (import from `_fixtures`) to both legs' `compute_rest_defense` calls. In `_assert_method`, iterate `RD_METRIC_COLUMNS` for presence, and add a Layer-2 must-populate check on the tracking leg: at least one of the pitch-control Layer-2 columns (`rd_attacker_space_control`) is non-NaN on resolved rows (proves the Layer-2 path ran on a real linked match). Do **not** assert `rd_gk_reachable_coverage_m2` non-NaN on the SB360 leg (it is Tier-2 → NaN there by design). Keep it method-only (no value assertions).

```python
    for c in RD_METRIC_COLUMNS:
        assert c in samples.columns
    assert resolved["rd_attacker_space_control"].notna().any(), "Layer-2 space control all-NaN on a real match"
```

- [ ] **Step 2: (SB360 leg) assert the Tier-2 honest-NaN** — on the velocity-less SB360 leg, `rd_gk_reachable_coverage_m2` should be all-NaN on resolved rows (Tier-2 suppression), while `rd_attacker_space_control` computes (Tier-1 lift). Add:

```python
    # SB360 leg only: reachable-area is Tier-2 (velocity-constitutive) -> honest-NaN; space control lifts.
    assert resolved["rd_gk_reachable_coverage_m2"].isna().all()
```

- [ ] **Step 3: Run (if data present) or confirm it self-skips.** This gate is `@e2e` — it is not in the CI "not e2e" run; verify it at least imports and skips cleanly: `python -m pytest tests/restdefense/test_e2e_method.py -v` (should skip, not error).

---

## Task 18: ADR-081, version bump, docs, changelog (housekeeping — folded into the single commit)

**Files:**
- Create: `docs/superpowers/adrs/ADR-081-rest-defense-layer2-danger.md`
- Modify: `silly_kicks/_version.py`, `uv.lock`, `TODO.md`, `CHANGELOG.md`, (optional) the spec header cross-ref.

- [ ] **Step 1: Write ADR-081** — mirror ADR-080's structure (Context / Decision / Alternatives / Consequences / Related / Notes). Record:
  - **(Decision)** Layer-2 danger-behind-line valuation ships in `restdefense/_danger.py` reusing `compute_threat_pc`/`compute_gk_influence`/`control_in_region`, oriented via `GoalMap`, additive, no retrain.
  - **(Alternatives)** the "extend tracking vs restdefense-local vs reimplement" fork, with **Option 2 chosen** (additive `field_weight` hook + `region` param + two additive public re-exports `zero_velocity_if_unavailable` and `compute_gk_influence` — one threat/reachable engine, default byte-identical) and why (semantic correctness of `compute_threat_pc` for counter-danger, DRY, spec-fidelity; owner-approved 2026-08-30).
  - **(Consequences)** GK-blind base = keeper-row-dropped; GK-included = keeper kept, which under the default `SpearmanParams(lambda_gk=3.0)` is **exactly equivalent** to the spec's literal `SpearmanParams(lambda_gk)` framing (the surface scales the kept keeper by `lambda_gk`; restdefense never constructs a `SpearmanParams` itself); GK contribution = base − gk (derivable, not a 6th column); `w_field` opt-in off-by-default un-tuned; `rd_gk_reachable_coverage_m2` Tier-2 honest-NaN; `xt` fail-closed.
  - **(Consequences — P2-02, owner-approved)** the whole Layer-2 family is **gated on a fitted `xt`**: without one, all five columns are NaN before any pitch-control call, so Layer-1-only callers are byte-identical to PR1 (no pitch-control cost, no velocity precondition). The cost — `#1`/`#4` need an `xt` despite not using its values — is the accepted price of a single opt-in signal and no caller break; SB360 loses nothing provider-specific (velocity-declared-absent frames never hit the raise; `#5` is Tier-2 NaN regardless).
  - **(Related)** spec §7.2/§13/§14, ADR-080, ADR-063/077/055/019/009/066/005; the independent plan review (`D:\Development\_reviews\2026-08-30-tf60-rest-defense-pr2-layer2-plan.md`, findings P2-01..P2-04).

- [ ] **Step 2: Bump the version** — `silly_kicks/_version.py` line 12: `__version__ = "4.102.0"` → `"4.103.0"`. Then `uv lock` (do not hand-edit `uv.lock`). Confirm `python -c "import silly_kicks; print(silly_kicks.__version__)"` prints `4.103.0`.

- [ ] **Step 3: CHANGELOG.md** — add a `4.103.0 (PR-S174, ADR-081)` entry keyed like the others: the 5 Layer-2 columns, the additive tracking seams (default byte-identical), no retrain, C4 unchanged.

- [ ] **Step 4: TODO.md** — update per the release convention (owner 2026-08-30): move TF-60 to "PARTLY SHIPPED" listing PR1+PR2 done and PR3–PR5 remaining; update the top summary to the current release.

- [ ] **Step 5: (optional) spec cross-ref** — add `ADR-081 (PR2)` beside `ADR-080` in the spec's §17 PR2 row or §8. Low-risk; skip if it muddies the committed spec.

---

## Task 19: Full-suite green, lint, pyright — then STOP for commit approval

**Files:** none (verification only).

- [ ] **Step 1: Full suite** — `python -m pytest tests/ -m "not e2e" --tb=short` (add `--benchmark-skip` if `tests/tracking/` hangs). Expected: all green. Capture the full `FAILED` list if any (never `tail`); diagnose root causes (do not patch tests to pass).

- [ ] **Step 2: Lint** — `python -m ruff check silly_kicks/ tests/ scripts/` and `python -m ruff format --check silly_kicks/ tests/ scripts/`. Fix any findings.

- [ ] **Step 3: Types** — `python -m pyright` (bare). Fix any new errors introduced by this PR.

- [ ] **Step 4: Doctests** — `python -m pytest --doctest-modules silly_kicks/restdefense/ -q` (the public `_config`/`__init__` doctests; the single-underscore modules are skipped in CI but keep their examples correct).

- [ ] **Step 5: STOP.** Do **not** commit. Present the full diff / file list to Karsten and wait for an explicit "commit" go-ahead (user standing rule: one fully-tested, coherent commit; approval is a separate gate). After approval, the commit / push / PR / squash-merge / tag `v4.103.0` + publish are **each** a separate go-ahead — do each, then stop; never merge until CI is green.

---

## Self-review (author checklist — run before handing off)

- **Spec §7.2 coverage:** all five columns → Tasks 4/8/9 (constants/metrics/wiring). `w_field` opt-in → Tasks 5/6/8. ✓
- **Spec §11 velocity/FOV:** Tier-1 lifts + Tier-2 NaN → Task 8 (inherited from `compute_gk_influence`) + Task 17 (e2e assertions). FOV companions → Task 10. Keeper identity (ADR-078) is the caller's job upstream; the fixtures/e2e resolve it. ✓
- **Spec §13 params:** `WFieldParams` + `danger_field_weight` un-tuned, empty `for_provider` → Task 5. ✓
- **Spec §14 schema:** nullable floats, honest-NaN, provenance unchanged → Tasks 8/9. (No new provenance column — Layer-2 NaN is honest via the existing `rd_geometry_source` + xt/keeper policy.) ✓
- **Spec §16.2 gates:** liveness/purity/id-dtype/orientation-D3/FOV-completeness/non-vacuity/SB360-audit/glossary/import-allowlist/e2e → Tasks 10–17. Import allowlist stays satisfied because restdefense imports only tracking **public** seams — but this required Task 3 to newly export `zero_velocity_if_unavailable` AND `compute_gk_influence` (the latter was NOT public; P2-01). `compute_pitch_control`/`compute_threat_pc` were already public; the `field_weight`/`region` extensions are on those already-public functions. ✓
- **Placeholder scan:** no TBD/TODO in the code steps; every code step has real content. ✓
- **Type consistency:** `layer2_metrics(frame_rows, ctx, *, xt, goal_map, params)` used identically in Tasks 8/9/12; `field_weight`/`region` signatures match across Tasks 1/2/8; `RD_METRIC_COLUMNS` used consistently in Tasks 4/9/10/11/14/15/17. ✓
- **Commit discipline:** one commit, no per-task commits, stop-at-gate — Task 19 + Global Constraints. ✓
