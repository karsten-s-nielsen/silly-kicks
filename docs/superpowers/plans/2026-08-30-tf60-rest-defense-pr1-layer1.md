# TF-60 Rest-Defense — PR1 (package + Layer-1 structure KPIs) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the `silly_kicks/restdefense/` package skeleton plus the Layer-1 *descriptive* rest-defense structure KPIs (`compute_rest_defense` / `summarize_rest_defense`), working on all tracking providers and best-effort on SB360, with the full restdefense CI-gate suite and ADR-080.

**Architecture:** A new hexagonal package (mirroring `gkdv/` / `xtgk/`) that consumes only `silly_kicks.tracking` public seams. PR1 samples rest-defense structure at the in-possession team's on-ball **action grid**, reduces to per-`(team, possession)` and per-`(team, match)` rollups plus a moment-of-loss snapshot, and composes existing engines (`resolve_defended_goals`/`GoalMap`, `compute_defensive_line`, `compute_team_shape`, `derive_team_in_possession`, `add_possessions`) plus one new `count_goalside` primitive. (`PitchControlSurface.control_in_region` and the threat/counterfactual engines belong to Layers 2–3, PR2/PR3 — NOT PR1.) Layers 2–3 (danger valuation + counterfactual arms) and the ghost-outfield model are later cycles.

**Tech Stack:** Python ≥3.10, pandas (2.1.1+, spanning majors — no upper bound), numpy ≥2.0, `silly_kicks.tracking`, `silly_kicks.id_compat`, `silly_kicks._frame_index`. No new runtime dependency.

**Spec:** `docs/superpowers/specs/2026-08-30-tf60-rest-defense-structure-and-gk-design.md` (read it alongside this plan; the plan argues from the spec, §-references point into it).

## Global Constraints

Copied verbatim from the spec / repo conventions; every task's requirements implicitly include these.

- **COMMIT DISCIPLINE (overrides the writing-plans template).** No per-task commits. Each task ends **test-green**, NOT committed. PR1 is **one** fully-tested coherent commit made **only** at Task 13 after the full suite is green **and after Karsten's explicit approval of that specific commit**. Never `git commit`/`git push` without it. No micro-commits, no "commit often."
- **Branch:** one feature branch off `main` for this cycle (e.g. `tf60-restdefense-pr1`). No worktrees, no parallel checkouts.
- **Docs land in the first (only) commit** — the spec, ADR-080, NOTICE, glossary, CHANGELOG, and C4 are committed **with** the code, never as a standalone doc commit.
- **Orientation (ADR-055 / ADR-051 D3):** direction NEVER from team identity. Build the `GoalMap` ONCE per match from the FULL frames via `resolve_defended_goals(frames)` and thread it in. Use `GoalMap.get(game, period, team)` for the team's OWN defended end and `.attacked_goal(...)` for the opponent's end (a real lookup, never `105 - get`). Keys are `canonical_id` strings. An unresolvable end → honest-NaN row (catch `GoalEndUnresolvedError` at the `compute_*` edge), never a guess. **Input frames MUST be LTR-normalized** (home-attacks-right; `play_left_to_right` / `orient_frames_to_ltr` applied) — the same precondition `compute_defensive_line` / `compute_team_shape` require (ADR-029). restdefense does NOT orient frames itself and does NOT use the private `tracking._action_orientation` helpers (`acting_team_attacks_rtl` is private to tracking and off-limits); orientation comes only from the `GoalMap`.
- **id_compat (ADR-019):** no raw `==`/`!=` on ids. `ids_match(series, scalar)`, `ids_equal`/`ids_differ` (column-vs-column), `same_id(a, b)` (scalar↔scalar), `align_join_keys` before an id merge, `canonical_id`/`canonical_id_series` for dict/hash keys. Never `astype(str)`/`astype(bool)` on an id/qualifier column (the NA-ball-row float-upcast trap; `astype(bool)` on object `"false"` is `True`).
- **No-rescan-in-loop (ADR-068) + sub-quadratic guard (ADR-073):** any per-frame/per-action loop builds the grouping ONCE with `silly_kicks._frame_index.group_rows(df, by)` then `.get(...)` per item. A new `group_rows` caller MUST register a scoped counter in `tests/_scale_guarded.py::SCALE_GUARDED` and pass `tests/_perf_structural.assert_subquadratic_growth` with a fixture that scales the **loop-iteration (group) dimension**.
- **Nullable dtypes (ADR-027/058):** counts `Int64`, fractions/metrics `float64`, flags `boolean`; NA on unscoreable rows — never a sentinel 0.
- **Purity (ADR-033):** every public `compute_*` / `summarize_*` returns a NEW object and mutates no caller input.
- **Layering (ADR-037-style):** `restdefense/` imports `silly_kicks.tracking` public seams only; **nothing imports `restdefense/`**, and `tracking` must never import it.
- **Version (ADR-079):** bump `silly_kicks/_version.py` ONLY (one literal); `pyproject` is dynamic; run `uv lock` (never hand-edit `uv.lock`).
- **Testing scope:** `python -m pytest tests/ -m "not e2e"`; lint `python -m ruff check silly_kicks/ tests/ scripts/` + `python -m ruff format --check silly_kicks/ tests/ scripts/`; `python -m pyright` (bare). Tools are `python -m` only (not on PATH). CI spans pandas 2 (ubuntu-3.10) and pandas 3 (others) — assert behaviour, never a dtype literal; a `.to_numpy()` you mutate needs `copy=True` (pandas-3 CoW read-only view).
- **No `__init__.py` in `tests/restdefense/`** (namespace-package shadowing); import as `from silly_kicks.restdefense import ...`.
- **Attribution (ADR-005):** every published-methodology metric gets a `NOTICE` entry; every emitted column gets a `feature_glossary.py` record.

---

## File Structure

**Package (`silly_kicks/restdefense/`):**
- `__init__.py` — public surface (`__all__`): `RestDefenseParams`, `compute_rest_defense`, `summarize_rest_defense`, `RestDefenseReport`, and the Layer-1 column-name constants.
- `_config.py` — `RestDefenseParams` (frozen dataclass, `for_provider`, flag-based `is_default`).
- `_geometry.py` — `danger_zone_bounds(...)` (the danger-behind-line rectangle; the rearguard line itself comes from `compute_defensive_line`, consumed in Task 7).
- `_counting.py` — `count_goalside(...)` (the new "behind the ball" primitive; `group_rows`-safe when looped).
- `_windows.py` — `select_rest_defense_samples(...)` (action-grid sampling: in-possession on-ball actions, committed-forward gate, loss-instant flag).
- `_structure.py` — the nine Layer-1 per-sample metric functions.
- `_compute.py` — `compute_rest_defense(...)` orchestrator + `summarize_rest_defense(...)` rollups.
- `_report.py` — `RestDefenseReport` (frozen; `n_frames_in`/`n_frames_scored`/`drop_reasons`).
- `_columns.py` — the Layer-1 output column-name constants + `RD_SAMPLE_KEYS`.

**Tests (`tests/restdefense/`, no `__init__.py`):**
- `_fixtures.py` — `make_rest_defense_fixture()` (multi-domain: 2 teams, a keeper each, advanced ball, both a home-possession and an away-possession action so orientation is exercised; LTR-normalized frames + action-LTR actions; loss action present).
- `test_config.py`, `test_geometry.py`, `test_counting.py`, `test_windows.py`, `test_structure.py`, `test_compute.py`.
- `test_import_allowlist.py`, `test_liveness.py`, `test_purity.py`, `test_id_dtype_invariance.py`, `test_orientation.py`.
- `tests/_scale_guarded.py` — register the `count_goalside`/window `group_rows` caller (MODIFY).
- `tests/restdefense/test_counting_perf_budget.py` — `assert_subquadratic_growth` for the counting loop.
- `tests/sb360/_registry.py` — register `compute_rest_defense` as a boundary entry (MODIFY).

**Docs/config:**
- `docs/superpowers/adrs/ADR-080-rest-defense-structure.md` (Create).
- `NOTICE`, `silly_kicks/feature_glossary.py`, `CHANGELOG.md`, `docs/PRIVATE_CONSUMERS.md`, `docs/c4/architecture.dsl`, `silly_kicks/_version.py`, `uv.lock` (Modify).

---

### Task 1: Package skeleton + import-allowlist gate

**Files:**
- Create: `silly_kicks/restdefense/__init__.py`, `silly_kicks/restdefense/_columns.py`
- Create: `tests/restdefense/test_import_allowlist.py`

**Interfaces:**
- Produces: the `silly_kicks.restdefense` package importable; `RD_SAMPLE_KEYS = ["game_id", "period_id", "team_id", "action_id"]` and the Layer-1 column-name constants in `_columns.py`.

- [ ] **Step 1: Write the failing import-allowlist test** (mirror `tests/gkdv/test_import_allowlist.py`).

```python
# tests/restdefense/test_import_allowlist.py
import ast, pathlib
import silly_kicks.restdefense  # must import cleanly

TRACKING = pathlib.Path(silly_kicks.__file__).parent / "tracking"

def test_tracking_never_imports_restdefense():
    for py in TRACKING.rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("silly_kicks.restdefense"):
                raise AssertionError(f"{py} imports restdefense (forbidden layering)")
            if isinstance(node, ast.Import):
                for a in node.names:
                    assert not a.name.startswith("silly_kicks.restdefense"), f"{py} imports restdefense"

def test_restdefense_public_surface_exists():
    from silly_kicks.restdefense import RD_SAMPLE_KEYS
    assert RD_SAMPLE_KEYS == ["game_id", "period_id", "team_id", "action_id"]

# Direction B (the half that catches a private-seam layering break, mirroring gkdv's 2-direction test):
# restdefense may import PUBLIC silly_kicks.tracking / silly_kicks.gkdv ONLY -- never their private
# (`._foo`) submodules. Empty allowlist: the Layer-1 impl uses only public seams
# (classify_region_observation, REGION_OBSERVATION_SOURCE_VALUES, compute_defensive_line, ...).
# group_rows is silly_kicks._frame_index (a package-level util in PRIVATE_CONSUMERS.md), NOT a
# tracking/gkdv private, so it is not caught here.
_PRIVATE_IMPORT_ALLOWLIST: set[tuple[str, str]] = set()

def test_restdefense_imports_only_public_tracking_and_gkdv_seams():
    rd_dir = pathlib.Path(silly_kicks.restdefense.__file__).parent
    for py in rd_dir.rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        mods = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                mods.append(node.module)
            elif isinstance(node, ast.Import):
                mods.extend(a.name for a in node.names)
        for m in mods:
            for pkg in ("silly_kicks.tracking.", "silly_kicks.gkdv."):
                if m.startswith(pkg) and m.rsplit(".", 1)[-1].startswith("_"):
                    assert (py.stem, m) in _PRIVATE_IMPORT_ALLOWLIST, (
                        f"{py.name} imports PRIVATE seam {m!r}; restdefense uses public seams only"
                    )
```

- [ ] **Step 2: Run — expect FAIL** (`ModuleNotFoundError: silly_kicks.restdefense`).

Run: `python -m pytest tests/restdefense/test_import_allowlist.py -v`

- [ ] **Step 3: Create `_columns.py` and `__init__.py`.**

```python
# silly_kicks/restdefense/_columns.py
"""Layer-1 output column names + sample keys (single source)."""
from __future__ import annotations

RD_SAMPLE_KEYS = ["game_id", "period_id", "team_id", "action_id"]

# Layer-1 metric column bases (one per spec §7.1 row).
RD_NUM_SUPERIORITY = "rd_num_superiority"
RD_NUM_SUPERIORITY_GK = "rd_num_superiority_gk"
RD_ZONE_OCCUPANCY = "rd_zone_occupancy"
RD_LINE_HEIGHT = "rd_line_height"
RD_LINE_HEIGHT_RELATIVE = "rd_line_height_relative"
RD_COMPACTNESS_X = "rd_compactness_x"
RD_WIDTH = "rd_width"
RD_DEPTH = "rd_depth"
RD_SHAPE_STAGGER = "rd_shape_2_3_vs_3_2"
RD_GK_LINE_HEIGHT = "rd_gk_line_height"
RD_GK_TO_LINE_DISTANCE = "rd_gk_to_line_distance"

RD_LAYER1_COLUMNS = [
    RD_NUM_SUPERIORITY, RD_NUM_SUPERIORITY_GK, RD_ZONE_OCCUPANCY, RD_LINE_HEIGHT,
    RD_LINE_HEIGHT_RELATIVE, RD_COMPACTNESS_X, RD_WIDTH, RD_DEPTH, RD_SHAPE_STAGGER,
    RD_GK_LINE_HEIGHT, RD_GK_TO_LINE_DISTANCE,
]
RD_GEOMETRY_SOURCE = "rd_geometry_source"  # {"resolved","unresolved"}
```

```python
# silly_kicks/restdefense/__init__.py
"""silly-kicks rest-defense structure metrics (TF-60). Hexagonal; consumes tracking public seams only."""
from __future__ import annotations

from ._columns import RD_LAYER1_COLUMNS, RD_SAMPLE_KEYS
from ._config import RestDefenseParams
from ._compute import compute_rest_defense, summarize_rest_defense
from ._report import RestDefenseReport

__all__ = [
    "RestDefenseParams",
    "compute_rest_defense",
    "summarize_rest_defense",
    "RestDefenseReport",
    "RD_LAYER1_COLUMNS",
    "RD_SAMPLE_KEYS",
]
```

(The `_config`/`_compute`/`_report` imports resolve as later tasks create those modules. For Step 4 to pass, create minimal stubs now: `_config.py` with a bare `RestDefenseParams` dataclass, `_report.py` with a bare `RestDefenseReport`, `_compute.py` with `def compute_rest_defense(*a, **k): raise NotImplementedError` and `summarize_rest_defense` likewise. Later tasks fill them.)

- [ ] **Step 4: Run — expect PASS.**

Run: `python -m pytest tests/restdefense/test_import_allowlist.py -v`
Expected: PASS (both tests).

---

### Task 2: `RestDefenseParams`

**Files:**
- Modify: `silly_kicks/restdefense/_config.py`
- Create: `tests/restdefense/test_config.py`

**Interfaces:**
- Produces: `RestDefenseParams(n_rearguard=4, min_ball_advance_m=52.5, zone_depth_m=None, danger_field_weight=False, possession_stride=1)` with `.default(*, force_universal=False)`, `.for_provider(provider) -> base for unlisted`, `.is_default() -> bool` (flag-based).

- [ ] **Step 1: Write the failing test.**

```python
# tests/restdefense/test_config.py
import dataclasses
from silly_kicks.restdefense import RestDefenseParams

def test_defaults():
    p = RestDefenseParams()
    assert p.n_rearguard == 4 and p.min_ball_advance_m == 52.5
    assert p.zone_depth_m is None and p.danger_field_weight is False and p.possession_stride == 1

def test_is_default_is_flag_based_not_value_equality():
    assert RestDefenseParams.default().is_default() is True
    assert RestDefenseParams().is_default() is False          # same field values, different provenance
    assert RestDefenseParams() == RestDefenseParams.default() # __eq__ ignores the provenance flag

def test_default_force_universal_disables_the_flag():
    assert RestDefenseParams.default(force_universal=True).is_default() is False
    assert RestDefenseParams.default(force_universal=False).is_default() is True

def test_for_provider_returns_base_for_unlisted():
    assert RestDefenseParams.for_provider("skillcorner") == RestDefenseParams()

def test_frozen():
    p = RestDefenseParams()
    try:
        p.n_rearguard = 5  # type: ignore[misc]
        raise AssertionError("expected FrozenInstanceError")
    except dataclasses.FrozenInstanceError:
        pass
```

- [ ] **Step 2: Run — expect FAIL.**  Run: `python -m pytest tests/restdefense/test_config.py -v`

- [ ] **Step 3: Implement `_config.py`** (mirror `CoverShadowParams.for_provider` + `PreprocessConfig.is_default`).

```python
# silly_kicks/restdefense/_config.py
from __future__ import annotations
import dataclasses
from dataclasses import dataclass, field

@dataclass(frozen=True)
class RestDefenseParams:
    n_rearguard: int = 4
    min_ball_advance_m: float = 52.5
    zone_depth_m: float | None = None
    danger_field_weight: bool = False
    possession_stride: int = 1
    _is_universal_default: bool = field(default=False, compare=False, repr=False)

    @classmethod
    def default(cls, *, force_universal: bool = False) -> "RestDefenseParams":
        # Mirror PreprocessConfig.default: force_universal=True returns a config whose is_default()
        # is False, so a (future) provider-aware caller does NOT auto-promote it.
        return cls(_is_universal_default=not force_universal)

    @classmethod
    def for_provider(cls, provider: str) -> "RestDefenseParams":
        return dataclasses.replace(cls(), **_PROVIDER_REST_DEFENSE_PARAMS.get(provider, {}))

    def is_default(self) -> bool:
        return self._is_universal_default

_PROVIDER_REST_DEFENSE_PARAMS: dict[str, dict] = {}  # EMPTY until an ADR-066-style apply-gate clears
```

- [ ] **Step 4: Run — expect PASS.**  Run: `python -m pytest tests/restdefense/test_config.py -v`

---

### Task 3: Geometry — rearguard line + danger zone (GoalMap-oriented)

**Files:**
- Create: `silly_kicks/restdefense/_geometry.py`
- Create: `tests/restdefense/test_geometry.py`

**Interfaces:**
- Consumes: `silly_kicks.tracking.resolve_defended_goals` → `GoalMap`; `GoalMap.get`, `GoalEndUnresolvedError`.
- Produces:
  - `danger_zone_bounds(line_x, own_goal_x, *, zone_depth_m=None) -> tuple[float, float]` — `(x_min, x_max)` of the danger strip between the rearguard line and the own goal (y is the full width 0–68), oriented by which end the own goal is at.
- **NO bespoke rearguard-line computation.** The rearguard line is `compute_defensive_line(frames, *, goal_map, n=params.n_rearguard).defensive_line_x` (TF-14 — GK-excluded, adaptive-n capable), computed ONCE per match in the orchestrator (Task 7) and read for team A at each sample's frame. Spec §6 mandates this single source; a second implementation would let the danger-zone boundary and `rd_line_height` disagree (the SHOULD-FIX that motivated this).

- [ ] **Step 1: Write the failing test.**

```python
# tests/restdefense/test_geometry.py
from silly_kicks.restdefense._geometry import danger_zone_bounds

def test_danger_zone_orientation_own_goal_low_x():
    assert danger_zone_bounds(20.0, 0.0) == (0.0, 20.0)      # own goal x=0, line x=20 -> [0, 20]

def test_danger_zone_orientation_own_goal_high_x():
    assert danger_zone_bounds(85.0, 105.0) == (85.0, 105.0)  # own goal x=105, line x=85 -> [85, 105]

def test_danger_zone_capped_depth():
    assert danger_zone_bounds(20.0, 0.0, zone_depth_m=10.0) == (0.0, 10.0)
    assert danger_zone_bounds(85.0, 105.0, zone_depth_m=10.0) == (95.0, 105.0)
```

- [ ] **Step 2: Run — expect FAIL.**  Run: `python -m pytest tests/restdefense/test_geometry.py -v`

- [ ] **Step 3: Implement `_geometry.py`.**

```python
# silly_kicks/restdefense/_geometry.py
from __future__ import annotations

def danger_zone_bounds(line_x: float, own_goal_x: float, *, zone_depth_m: float | None = None) -> tuple[float, float]:
    """(x_min, x_max) of the danger strip between the rearguard line and the OWN goal (y = full 0-68)."""
    lo, hi = (own_goal_x, line_x) if own_goal_x <= line_x else (line_x, own_goal_x)
    if zone_depth_m is not None:
        if own_goal_x <= line_x:          # own goal at low x: keep the strip nearest the goal
            hi = min(hi, own_goal_x + zone_depth_m)
        else:                             # own goal at high x
            lo = max(lo, own_goal_x - zone_depth_m)
    return (float(lo), float(hi))
```

- [ ] **Step 4: Run — expect PASS.**  Run: `python -m pytest tests/restdefense/test_geometry.py -v`

Note: `GoalEndUnresolvedError` / `resolve_defended_goals` / `GoalMap` are re-exported from `silly_kicks.tracking` (`tracking/__init__.py`). GK exclusion + adaptive-n for the rearguard line are `compute_defensive_line`'s responsibility (TF-14), consumed in Task 7 — not reimplemented here. The `pd`/`np`/`ids_match` imports the deleted `rearguard_line_x` needed are gone; `_geometry.py` is now pure-stdlib.

---

### Task 4: Counting primitive `count_goalside` + scale guard

**Files:**
- Create: `silly_kicks/restdefense/_counting.py`
- Create: `tests/restdefense/test_counting.py`, `tests/restdefense/test_counting_perf_budget.py`
- Modify: `tests/_scale_guarded.py` (register the caller); `docs/PRIVATE_CONSUMERS.md` (restdefense consumes `_frame_index.group_rows`).

**Interfaces:**
- Consumes: `silly_kicks.id_compat.ids_match`; `silly_kicks._frame_index.group_rows`.
- Produces:
  - `count_goalside(frame_rows, *, team_id, ball_x, goal_x, include_gk=True) -> int` — count of `team_id`'s players between `ball_x` and `goal_x` (goal-side of the ball toward `goal_x`). `goal_x` is the **reference goal to count TOWARD**, NOT necessarily the team's own goal: `rd_num_superiority` counts BOTH teams toward **A's** defended goal G_A (spec §7.1).
  - `count_goalside_by_sample(samples_index, frames, *, ...) -> pd.Series` — the `group_rows`-based batched form used by the orchestrator (registered scale-guarded caller).

- [ ] **Step 1: Write the failing test.**

```python
# tests/restdefense/test_counting.py
import pandas as pd
from silly_kicks.restdefense._counting import count_goalside

def _rows(team_id, xs, gk_flags=None):
    gk_flags = gk_flags or [False] * len(xs)
    return pd.DataFrame({"team_id": team_id, "player_id": range(len(xs)), "x": xs, "y": 34.0,
                         "is_ball": False, "is_goalkeeper": gk_flags})

def test_counts_goalside_reference_goal_low_x():
    # reference goal G at 0, ball at 40; goal-side = players with x in [0, 40]
    rows = _rows(3, [5.0, 20.0, 45.0])   # two in [0,40], one beyond the ball
    assert count_goalside(rows, team_id=3, ball_x=40.0, goal_x=0.0) == 2

def test_counts_goalside_reference_goal_high_x():
    rows = _rows(3, [100.0, 85.0, 60.0])  # G=105, ball 65 -> x in [65,105] => 100,85
    assert count_goalside(rows, team_id=3, ball_x=65.0, goal_x=105.0) == 2

def test_include_gk_flag():
    rows = _rows(3, [5.0, 2.0], gk_flags=[False, True])  # both goal-side; GK is one of them
    assert count_goalside(rows, team_id=3, ball_x=40.0, goal_x=0.0, include_gk=True) == 2
    assert count_goalside(rows, team_id=3, ball_x=40.0, goal_x=0.0, include_gk=False) == 1

def test_string_team_id_matches_numeric_column():  # ADR-019
    rows = _rows(3, [5.0]); rows["team_id"] = rows["team_id"].astype("Int64")
    assert count_goalside(rows, team_id="3", ball_x=40.0, goal_x=0.0) == 1
```

- [ ] **Step 2: Run — expect FAIL.**  Run: `python -m pytest tests/restdefense/test_counting.py -v`

- [ ] **Step 3: Implement `_counting.py`.**

```python
# silly_kicks/restdefense/_counting.py
from __future__ import annotations
import numpy as np, pandas as pd
from silly_kicks.id_compat import ids_match

def count_goalside(frame_rows, *, team_id, ball_x: float, goal_x: float, include_gk: bool = True) -> int:
    """Players of `team_id` between the ball and the REFERENCE goal `goal_x` (goal-side of the ball).

    `goal_x` is the goal to count TOWARD, not necessarily the team's own goal -- `rd_num_superiority`
    passes A's defended goal G_A for BOTH teams (spec §7.1).
    """
    own = frame_rows[ids_match(frame_rows["team_id"], team_id)]   # ball rows carry NA team -> excluded (ADR-058)
    if not include_gk and "is_goalkeeper" in own.columns:
        gk = own["is_goalkeeper"].fillna(False).to_numpy(dtype=bool)  # boolean-safe, NOT astype(bool)
        own = own.loc[~gk]
    xs = own["x"].to_numpy(dtype=float)
    xs = xs[np.isfinite(xs)]
    lo, hi = (goal_x, ball_x) if goal_x <= ball_x else (ball_x, goal_x)
    return int(np.count_nonzero((xs >= lo) & (xs <= hi)))
```

- [ ] **Step 4: Run — expect PASS.**  Run: `python -m pytest tests/restdefense/test_counting.py -v`

- [ ] **Step 5: Add the batched `group_rows` form + register the scale guard.** The orchestrator (Task 7) resolves one frame per sample; do it with `group_rows` ONCE, never a per-sample `frames[frames.frame_id==fid]`. Add `count_goalside_by_sample(...)` that builds `group_rows(frames, ("game_id","period_id","frame_id"))` once and `.get()` per sample. Then register in `tests/_scale_guarded.py::SCALE_GUARDED` (the meta-assertion `test_scale_guard_registry` fails otherwise) and write `tests/restdefense/test_counting_perf_budget.py`:

```python
# tests/restdefense/test_counting_perf_budget.py
from tests._perf_structural import assert_subquadratic_growth, rows_scanned_counter  # both live here

def test_count_goalside_by_sample_is_subquadratic():
    # fixture MUST scale the GROUP dimension (number of frames/samples), not within-group size.
    def measure_work(n_frames):
        from tests.restdefense._fixtures import make_scaling_fixture
        frames, samples = make_scaling_fixture(n_frames=n_frames)  # n distinct frames, small each
        with rows_scanned_counter() as counter:
            from silly_kicks.restdefense._counting import count_goalside_by_sample
            count_goalside_by_sample(samples, frames, team_col="team_id", ball_x_col="ball_x", goal_x_col="own_goal_x")
        return counter.count
    assert_subquadratic_growth(measure_work, sizes=(256, 1024, 4096), max_exponent=1.5)
```

Register in `tests/_scale_guarded.py` (add the `count_goalside_by_sample` qualname to `SCALE_GUARDED` with the `rows_scanned_counter` guard) and add the `restdefense` consumer line to `docs/PRIVATE_CONSUMERS.md` under `_frame_index.group_rows`.

- [ ] **Step 6: Run — expect PASS** (`python -m pytest tests/restdefense/test_counting_perf_budget.py tests/test_scale_guard_registry.py -v`).

---

### Task 5: Windows — action-grid sampling + committed-forward gate + loss instant

**Files:**
- Create: `silly_kicks/restdefense/_windows.py`
- Create: `tests/restdefense/_fixtures.py`, `tests/restdefense/test_windows.py`

**Interfaces:**
- Consumes: `silly_kicks.spadl.add_possessions` (→ `possession_id`); `silly_kicks.tracking.derive_team_in_possession`/`infer_ball_carrier`; `silly_kicks.tracking.link_actions_to_frames` → `(pointers, LinkReport)`; `GoalMap`.
- Produces: `select_rest_defense_samples(actions, frames, *, goal_map, params, links=None) -> pd.DataFrame` — one row per in-possession on-ball action that PASSES the committed-forward gate, with columns `[*RD_SAMPLE_KEYS, "possession_id", "frame_id", "ball_x", "own_goal_x", "attacked_goal_x", "is_possession_loss", "gate_drop_reason"]` (dropped actions carry a non-null `gate_drop_reason`, so the caller can conserve).

- [ ] **Step 1: Build the shared fixture** `make_rest_defense_fixture()` in `tests/restdefense/_fixtures.py`: 1 game, 1 period, two teams (`1` home, `2` away), a keeper + 5 outfield each, LTR-normalized frames (`team_attacking_direction`), a small `actions` frame with (a) a home in-possession advanced pass, (b) an away in-possession advanced pass (exercises orientation), (c) a possession-ending loss action, and (d) a non-advanced action that the committed-forward gate should drop. Include a `make_scaling_fixture(n_frames)` that emits `n` distinct frames for the perf guard.

- [ ] **Step 2: Write the failing test.**

```python
# tests/restdefense/test_windows.py
from silly_kicks.restdefense._config import RestDefenseParams
from silly_kicks.restdefense._windows import select_rest_defense_samples
from silly_kicks.tracking import resolve_defended_goals
from tests.restdefense._fixtures import make_rest_defense_fixture

def test_selects_advanced_in_possession_actions_and_flags_loss():
    actions, frames = make_rest_defense_fixture()
    gm = resolve_defended_goals(frames)
    s = select_rest_defense_samples(actions, frames, goal_map=gm, params=RestDefenseParams())
    scored = s[s["gate_drop_reason"].isna()]
    assert len(scored) >= 2                      # home + away advanced actions
    assert scored["is_possession_loss"].any()    # the terminal loss is flagged
    # the non-advanced action is dropped, not scored:
    assert (s["gate_drop_reason"] == "not_committed_forward").any()
    # conservation: every input in-possession on-ball action is either scored or dropped
    assert len(s) == len(s.drop_duplicates(subset=["game_id","period_id","team_id","action_id"]))
```

- [ ] **Step 3: Implement `_windows.py`.** Steps inside: add `possession_id` via `add_possessions`; derive team-in-possession per action (carrier at action / possession owner); link each action to its frame via `link_actions_to_frames` (unpack `(pointers, _report)`); resolve `ball_x`, `own_goal_x = goal_map.get(...)`, `attacked_goal_x = goal_map.attacked_goal(...)`; apply the committed-forward gate `abs(ball_x - own_goal_x) >= params.min_ball_advance_m` (drop-and-count with `gate_drop_reason ∈ {"dead_ball","not_in_possession","unlinked","not_committed_forward","goal_end_unresolved"}`); flag `is_possession_loss` on each possession's terminal action. Use `align_join_keys` before the action↔frame merge (ADR-019).

- [ ] **Step 4: Run — expect PASS.**  Run: `python -m pytest tests/restdefense/test_windows.py -v`

---

### Task 6: Layer-1 structure metrics (`_structure.py`)

**Files:**
- Create: `silly_kicks/restdefense/_structure.py`
- Create: `tests/restdefense/test_structure.py`

**Interfaces:**
- Consumes: `count_goalside` (Task 4); `danger_zone_bounds` (Task 3); `silly_kicks.tracking.compute_defensive_line` (→ `defensive_line_x` — the single rearguard-line source —, `compactness_x`, `lateral_width`); `silly_kicks.tracking.compute_team_shape` (→ `team_length`, `team_width`, `stretch_index`). The `rd_shape_2_3_vs_3_2` stagger is the largest-gap split of the behind-ball unit (below), NOT `compute_team_shape` clustering.
- Produces one function per spec-§7.1 metric, each `f(frame_rows, sample_ctx, *, params) -> float | int`, plus `layer1_metrics(frame_rows, sample_ctx, *, params) -> dict[str, float]` returning all Layer-1 columns for one sample. `sample_ctx` carries `team_id` (in-possession A), `opponent_id` (B), `ball_x`, `own_goal_x`, `attacked_goal_x`, `defending_line_x` (A's rearguard line).

Metric definitions (each a TDD sub-test with a hand-computed expected value on the fixture; formulas from spec §7.1):

- `rd_num_superiority` = `count_goalside(A, goal_x=G_A, include_gk=False) − count_goalside(B, goal_x=G_A, include_gk=False)`. **Both teams count toward G_A** (A's defended goal) — A's rearguard vs. B's players already goal-side of the ball (the potential counter receivers). Pass `goal_x=G_A` for BOTH; passing B's own goal G_B would invert the band.
- `rd_num_superiority_gk` = same, with `include_gk=True` for A's count (adds A's keeper when it is goal-side of the ball).
- `rd_zone_occupancy` = count of A's players inside `danger_zone_bounds(line_x, own_goal_x)` (x within the strip; `Int64`).
- `rd_line_height` = `abs(A.defensive_line_x − own_goal_x)` (from `compute_defensive_line`, filtered to A's row).
- `rd_line_height_relative` = `A.defensive_line_x − ball_x` in goal-relative orientation.
- `rd_compactness_x` / `rd_width` / `rd_depth` = A's `compactness_x` + `lateral_width` (both from `compute_defensive_line`, the rearguard back line, GK-excluded) + whole-team `team_length` (`compute_team_shape`). **Owner-ratified "Option B" (2026-08-30, ADR-080) — supersedes the original "`team_width` … restricted to A's rearguard subset" wording flagged by the `/review-impl` gate:** `rd_width` is the rearguard lateral width (not whole-team `team_width`); `rd_depth` stays whole-team `team_length` because a back-line depth would merely duplicate `rd_compactness_x` (a flat line has ~no independent depth).
- `rd_shape_2_3_vs_3_2` = a deterministic 2-line split of A's **behind-the-ball unit** (A's players goal-side of the ball toward G_A): take their goal-relative x, sort, split at the single **largest gap** into a deeper group (nearer G_A) and a shallower group, label `"{n_deeper}-{n_shallower}"` (e.g. `"3-2"`). This is NOT available from `compute_team_shape` (which emits inter-line gaps, not per-line member counts). A unit size ≠ 5 → the generic `"n-m"` (Task 7); a unit < 2 → NA.
- `rd_gk_line_height` = `abs(gk_x − own_goal_x)` (A's keeper, goal-relative).
- `rd_gk_to_line_distance` = `gk_x − A.defensive_line_x` (goal-relative; the FIFA coupled-unit gap).

- [ ] **Step 1: Write failing tests** — one per metric, each asserting a **hand-computed value** on `make_rest_defense_fixture()` for BOTH the home-possession and the away-possession sample (orientation exercised: geometrically-mirrored-but-semantically-equal). For `rd_num_superiority` the symmetry check is necessary but NOT sufficient — an all-zero or wrong-band impl also passes `home == away` — so ALSO pin a hand-computed **A≠B** value:

```python
def test_num_superiority_hand_computed_and_orientation_symmetric():
    # fixture: A keeps 4 goal-side of the ball toward G_A; B has 2 forwards goal-side of the ball.
    home = compute_one_sample(fixture, acting="home")                 # A = home
    away = compute_one_sample(point_reflect(fixture), acting="away")  # A = away, positions mirrored
    assert home["rd_num_superiority"] == 2                            # value pinned (4 - 2), not just symmetric
    assert away["rd_num_superiority"] == 2                            # AND orientation-invariant
```

- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement each metric** in `_structure.py`, composing the real engine calls with the goal-relative orientation from `sample_ctx` (never team identity). Compute `compute_defensive_line(frames, goal_map=gm)` and `compute_team_shape(frames, team_id=A)` ONCE per match in the orchestrator (Task 7) and pass the per-frame slice in, not per metric (no rescans).
- [ ] **Step 4: Run — expect PASS.**

---

### Task 7: Orchestrator — `compute_rest_defense` + `summarize_rest_defense` + report

**Files:**
- Modify: `silly_kicks/restdefense/_compute.py`, `silly_kicks/restdefense/_report.py`
- Create: `tests/restdefense/test_compute.py`

**Interfaces:**
- Produces:
  - `compute_rest_defense(actions, frames, *, goal_map=None, links=None, params=RestDefenseParams()) -> tuple[pd.DataFrame, RestDefenseReport]` — the per-sample table (`RD_SAMPLE_KEYS` + `possession_id` + `is_possession_loss` + `RD_LAYER1_COLUMNS` + `RD_GEOMETRY_SOURCE`) and the conserving report. Builds `GoalMap` once (or accepts one), pre-links once, computes `compute_defensive_line`/`compute_team_shape` once, loops samples via `group_rows` (no rescans), catches `GoalEndUnresolvedError` per sample → honest-NaN row with `rd_geometry_source="unresolved"`.
  - `summarize_rest_defense(samples, *, by: Literal["possession","match"]) -> pd.DataFrame` — pure reductions (mean of float metrics, sum-conserving of counts) per `(team, possession_id)` or `(team, game_id)`; a sample-less group is honest-NaN (never a fabricated 0).
  - `RestDefenseReport(params, n_frames_in, n_frames_scored, drop_reasons)` frozen; conservation asserted by a CI gate (not a property).

- [ ] **Step 1: Write the failing test** — `compute_rest_defense` on the fixture yields one row per scored sample with all `RD_LAYER1_COLUMNS` non-null on resolvable rows; conservation `report.n_frames_scored + sum(report.drop_reasons.values()) == report.n_frames_in`; an unresolvable-goal fixture yields `rd_geometry_source=="unresolved"` and NaN metrics (not a crash, not a 0). `summarize_rest_defense(..., by="match")` returns one row per `(team, game_id)`.

- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement `_compute.py`** — the full orchestration with the disciplines above (build-once, thread-in, honest-NaN, drop-conservation). Resolve the ≠5-player stagger edge here (per spec §20.7: label `"n-m"` generically when the behind-ball unit ≠ 5). `summarize_rest_defense` uses `groupby(...).agg` with `min_count=1` on counts. **Single-source the rearguard line:** the `line_x` fed to `danger_zone_bounds` AND the value behind `rd_line_height` are BOTH team A's `defensive_line_x` from the single per-match `compute_defensive_line(frames, goal_map=gm, n=params.n_rearguard)` call (spec §6) — never a bespoke recomputation.
- [ ] **Step 4: Run — expect PASS.**

---

### Task 8: FOV observability companions (ADR-077, opt-in)

**Files:**
- Modify: `silly_kicks/restdefense/_compute.py` (add `visible_area=None` kwarg)
- Create: `tests/restdefense/test_fov_companions.py`

**Interfaces:**
- Consumes: the **PUBLIC** ADR-062/077 seams `classify_region_observation(polygon, region) -> (fraction, source)` and `REGION_OBSERVATION_SOURCE_VALUES` (both in `tracking.__all__` — verified `tracking/__init__.py:165,46`). **No private-registry import:** the one-engine `append_observability_companions` / `OBSERVABILITY_REGISTRY` is private, so restdefense assembles companions itself in the CANONICAL naming (`<col>_observed_fraction` / `<col>_observed_source`, `source ∈ REGION_OBSERVATION_SOURCE_VALUES`) — format-identical to every other aggregator. The shared tracking FOV completeness gate structurally does NOT cover restdefense (it excludes non-`add_*`/boundary surfaces, verified `test_fov_completeness_gate.py`), so a restdefense-LOCAL gate does (Task 9, Step 2b).
- Produces: when `visible_area` is supplied, `compute_rest_defense` emits `<col>_observed_fraction` / `<col>_observed_source` for each region/count column (`rd_num_superiority*`, `rd_zone_occupancy`); primary columns byte-identical with/without `visible_area`.

- [ ] **Step 1: Failing test** — companions absent without `visible_area`; present + populated with it; primary columns byte-identical between the two calls (a full-coverage polygon → `observed_fraction==1.0`, `observed_source=="observed"`; a cropped polygon over the danger strip → `<1.0`).
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** — for each FOV-sensitive column build its fixed **action-LTR** ROI (the danger-zone rectangle for `rd_zone_occupancy`; the goal-side band for `rd_num_superiority*` — a fixed role-keyed region, NEVER a `goal_map`, per ADR-077 S1), call `classify_region_observation(visible_area_polygon, roi)`, and write `<col>_observed_fraction` / `<col>_observed_source`; overlay `unlinked` from the action↔frame link. Primary columns must be byte-identical with/without `visible_area` (2nd purity variant, Task 9).
- [ ] **Step 4: Run — PASS.**

---

### Task 9: The restdefense CI-gate suite

**Files:**
- Create: `tests/restdefense/test_liveness.py`, `test_purity.py`, `test_id_dtype_invariance.py`, `test_orientation.py`
- Modify: `tests/sb360/_registry.py` (boundary entry)

**Interfaces:** none new (gates over the public surface built in Tasks 1–8).

- [ ] **Step 1: Liveness** (mirror `tests/tracking/test_aggregator_column_liveness.py`): on the multi-domain fixture, every `RD_LAYER1_COLUMNS` column is non-null on resolvable rows and every float metric with ≥2 observations is non-constant. Write it RED first (before Task 6/7 fully land) so it is observed failing.
- [ ] **Step 2: Purity** (ADR-033): `compute_rest_defense` / `summarize_rest_defense` mutate no caller input and return new objects; register a 2nd variant for the `visible_area` present/absent branch.
- [ ] **Step 2b: FOV completeness** (ADR-077, restdefense-LOCAL — the shared tracking gate excludes non-`add_*`/boundary surfaces, verified `test_fov_completeness_gate.py`): `tests/restdefense/test_fov_completeness.py` asserts every FOV-sensitive restdefense column (`rd_num_superiority`, `rd_num_superiority_gk`, `rd_zone_occupancy` — the count/region columns) has BOTH `_observed_fraction` and `_observed_source` companions when `visible_area` is supplied, or is listed in a restdefense `_OBSERVABILITY_EXEMPT` with a stated reason. A meta-assertion pins the sensitive-column list against `RD_LAYER1_COLUMNS` so a new region/count column must companion-or-exempt (the anti-rot property). Assert every `_observed_source` value is in `REGION_OBSERVATION_SOURCE_VALUES`.
- [ ] **Step 3: id-dtype invariance** (ADR-019): numeric-actions × string-frames and the reverse yield identical `compute_rest_defense` output; `home_team_id`-independent (there is no `home_team_id` param — direction comes from `GoalMap`).
- [ ] **Step 4: Orientation / D3** (ADR-051): (a) **Gate C** — hold frames fixed, swap the `GoalMap`, require the geometry columns (`rd_line_height`, `rd_gk_to_line_distance`, `rd_num_superiority`) to move; (b) **direction-invariance** — mirror the frames (point-reflect) AND swap the goal map consistently, require the action-LTR metrics unchanged; assert direction is never taken from team identity (there is no such code path).
- [ ] **Step 5: SB360 boundary audit** (ADR-053): register `compute_rest_defense` in `tests/sb360/_registry.py` with a per-column verdict + `verdict_provenance`; Layer-1 positional columns are `works`/`structural` on the paired full-coverage fixture (velocity-invariant), the count columns adjudicated with a rationale. Run `tests/sb360/` and reconcile.
- [ ] **Step 6: Run the whole restdefense suite green.**  Run: `python -m pytest tests/restdefense tests/sb360 -m "not e2e" -v`

---

### Task 10: e2e method gate (`@e2e`, spec §16)

**Files:**
- Create: `tests/restdefense/test_e2e_method.py` (marked `@pytest.mark.e2e`)

- [ ] **Step 1: Write the `@e2e` method gate** — on ≥1 real linked tracking match (fixture-gated, `worldcup-hdf5-e2e` precedent) and ≥1 real SB360 match. **SB360 preparation (REQUIRED, spec §11.3/§16.2):** before `compute_rest_defense`, resolve keeper identity and bridge it onto the anonymous freeze-frames —
  ```python
  keeper_map, _rep = resolve_keeper_identities(actions, frames, identity="roster", roster=roster)
  frames = apply_keeper_identities_to_frames(frames, keeper_map)
  ```
  otherwise the `rd_gk_*` columns are all-NaN (the ADR-078 anonymity gap) and the GK assertion is vacuous. The continuous match uses the native `is_goalkeeper` path (no bridge). Then run `compute_rest_defense` / `summarize_rest_defense` and assert non-empty output, `report` conservation reconciles, the `rd_gk_line_height` / `rd_gk_to_line_distance` columns are populated (not all-NaN) on BOTH matches, and (SB360) FOV companions populate with a `<1.0` observed fraction on a cropped advanced-ball frame. Assert **method**, not metric values.
- [ ] **Step 2: Confirm it is `@e2e`** (skips in the default `-m "not e2e"` run; the wiring test `tests/test_ci_*` must still pass). Do NOT run it in the standard suite.

---

### Task 11: Attribution + glossary + ADR-080

**Files:**
- Create: `docs/superpowers/adrs/ADR-080-rest-defense-structure.md`
- Modify: `NOTICE`, `silly_kicks/feature_glossary.py`, `docs/PRIVATE_CONSUMERS.md`

- [ ] **Step 1: ADR-080** — record the decision (new `restdefense` package; action-grid sampling; GoalMap orientation; the descriptive Layer-1 scope; the deferral of Layers 2–3/model to later cycles; the honest SB360 ceiling). Follow `ADR-TEMPLATE.md`.
- [ ] **Step 2: NOTICE** — add the Layer-1 methodology entries (Forcher 2023; Peters 2025; Memmert 2022; Dash 2025; FIFA EFI + StatsBomb/Wyscout practitioner GK-depth), peer-reviewed vs practitioner labelled. **Also fix the pre-existing mis-cited arXiv IDs** for the TF-14 defensive-line block (`2511.06191` under *Herold 2022* and `2511.00121` under *Forcher 2022* — verify the intended IDs and correct; spec §18).
- [ ] **Step 3: feature_glossary** — add a `FeatureColumn` record for every `RD_LAYER1_COLUMNS` column (`emitting_module="_structure"`, `higher_is_better` per metric); FOV companions are glossary-exempt (ADR-062/077). Run the glossary coverage gate.
- [ ] **Step 4: Run** `python -m pytest tests/ -m "not e2e" -k "glossary or notice or adr"` (and any ADR-existence / provenance wiring tests) — expect PASS.

---

### Task 12: C4 + version bump

**Files:**
- Modify: `docs/c4/architecture.dsl` (+ regenerate `architecture.html`), `silly_kicks/_version.py`, `CHANGELOG.md`, `uv.lock`

- [ ] **Step 1: C4** — add the `restdefense` container to `architecture.dsl` (relationships: `restdefense -> tracking [public seams]`; consumed-by nothing). Regenerate via the `mad-scientist-skills:c4` pipeline with the pinned Graphviz `dot` (`-graphvizdot "C:/Users/Karsten/.claude/tools/graphviz/dot.exe"`). Run the C4 completeness gate.
- [ ] **Step 2: Version** — bump `silly_kicks/_version.py` to the next minor per the per-PR convention (take the next free number from `main`; NUMBER ≠ TAG). Run `uv lock`. Run `tests/test_version_single_source.py`.
- [ ] **Step 3: CHANGELOG** — add the `PR-Snnn` entry (Layer-1 rest-defense; additive; no retrain; new container). Keep it factual.
- [ ] **Step 4: Run the C4 + version + shard-wiring gates** — expect PASS.

---

### Task 13: Full-suite green + single commit gate (STOP for approval)

**Files:** none (verification + the one commit).

- [ ] **Step 1: Full suite** — `python -m pytest tests/ -m "not e2e"` → all green (capture any `FAILED` in full, never `tail`).
- [ ] **Step 2: Lint + types at CI scope** — `python -m ruff check silly_kicks/ tests/ scripts/`; `python -m ruff format --check silly_kicks/ tests/ scripts/`; `python -m pyright`. All clean.
- [ ] **Step 3: Slow/primary-leg sanity** — confirm no version-sensitive test was mis-marked `slow`; the new restdefense golden/liveness/id-dtype gates run on all legs.
- [ ] **Step 4: Show the diff and STOP.** Present `git status` + the staged file list + a one-paragraph summary of what PR1 delivers. **Do NOT commit.** Wait for Karsten's explicit approval of this specific commit.
- [ ] **Step 5: On explicit approval only** — `git add` the whole PR1 change set (code + spec + ADR-080 + NOTICE + glossary + C4 + CHANGELOG + version + uv.lock) and make **one** commit on the `tf60-restdefense-pr1` branch with the standard trailers. Then STOP (opening the PR / merge / tag are each separate go-aheads).

---

## Self-Review

**1. Spec coverage (PR1 scope only):** §4.1 package layout → Task 1; §5 measurement model / action grid / windows / loss instant → Task 5; §6 geometry → Task 3; §7.1 Layer-1 metrics → Task 6; §8 counting primitive → Task 4; §11.2 FOV companions → Task 8; §11.3 keeper identity → used via `frames` (native path in PR1 fixtures; the roster bridge is exercised only by the SB360 e2e, Task 10); §12 API (Layer-1 subset) → Tasks 2/7; §13 params → Task 2; §14 output schema (samples + report; arm tables are PR3) → Task 7; §16 gates → Tasks 9/10; §17 PR1 row → whole plan; §18 attribution + NOTICE fix → Task 11; C4 → Task 12. Layers 2–3, the arms, `merge_rest_defense`, and the ghost-outfield model are explicitly OUT of PR1 (later cycles) — not gaps.

**2. Placeholder scan:** metric bodies in Task 6 are specified by formula + the exact composed engine call (concrete, not "add logic"); each has a hand-computed test. No "TBD"/"handle edge cases" left. The ≠5 stagger edge is resolved in Task 7 with a stated rule.

**3. Type consistency:** `RD_SAMPLE_KEYS` (Task 1) is reused verbatim in Tasks 5/7/9; `count_goalside`(Task 4) signature matches its call in Task 6; `select_rest_defense_samples` output columns (Task 5) match the `compute_rest_defense` consumer (Task 7); `GoalMap.get`/`.attacked_goal` used consistently; `RestDefenseReport` fields (`n_frames_in`/`n_frames_scored`/`drop_reasons`) match the spec §14 correction and the conservation gate (Task 7/9).

Known follow-ups deferred by design (in the spec's §20, not plan gaps): the committed-forward gate default calibration, the OBPV `w_field` option (Layer 2), and the ghost-GK in-possession validity gate (PR3). **Plan-review R2 note (PLAN-12):** restdefense hand-assembles the FOV companions from the public low-level `classify_region_observation` because the one-engine appender (`append_observability_companions`) is private — a 2nd assembly path that is gated by the restdefense-local completeness gate but could drift; the durable later-cycle fix is a PUBLIC companion-assembly seam in tracking. Ship PR1 as-is.
