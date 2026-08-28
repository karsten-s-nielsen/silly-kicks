# SB360 FOV-Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make FOV-observability a first-class, single-sourced concern — a public `validate_fov`
diagnostic, one declarative region/area observability registry emitting opt-in `_observed_fraction` /
`_observed_source` companions across the region/aggregate metric family, a real-signal retirement of
the `space_creation` velocity FOV-proxy, and an anti-rot completeness gate.

**Architecture:** All FOV primitives already live in the neutral `tracking/_visibility.py` /
`silly_kicks/_polygon.py`. This cycle adds (1) a frame-set `validate_fov`/`FovDiagnosis` mirroring the
`validate_velocity_regime` family, (2) a `tracking/_fov_registry.py` declaring each metric's convex
region and one engine that maps `visible_area × region → (fraction, source)` via the existing
`classify_region_observation`, retiring ADR-062's hand-coded helper into it byte-identically, and (3)
seven aggregators gaining an opt-in `visible_area` kwarg. **One model only** — observed-*area* fraction
of a region (a per-contributor "roster" fraction is identically ≈1.0 on a freeze-frame provider, S1).

**Tech Stack:** Python, pandas, numpy; `pytest -m "not e2e"`; `ruff` + `pyright` (via `python -m`);
the codebase's registry+completeness-gate idiom.

## Global Constraints

- **Additive / opt-in.** Every primary feature column is **byte-identical** with and without
  `visible_area`. Companions appear only when `visible_area is not None`. **No VAEP retrain.**
- **One model.** Observed-*area* fraction of a **convex** region via `classify_region_observation`.
  There is **no per-contributor / `point_observed` path** in the companion engine (S1).
- **Frame-independence invariant (N2).** A region is a function of pitch/goal geometry only — a tight
  ROI keyed on the action anchor, or an aggregate zone keyed on `goal_map`. It is **never** computed
  from the frame's observed player coordinates (or S1 recurs).
- **Vocabulary reuse.** Sources come from `REGION_OBSERVATION_SOURCE_VALUES` (`observed`,
  `no_polygon`, `degenerate_polygon`, `degenerate_region`) + the caller-overlaid `unlinked`. No new
  token.
- **Neutral layering.** `_fov_registry.py` imports only `_visibility`, `_geometry`, `_kernels` —
  never `pitch_control` / `_das`.
- **No number is claimed until commit-ready.** The version (FIVE sites incl. `uv.lock`), the
  `PR-Sxxx`, and the `ADR-0xx` number are **placeholders throughout this plan** and are assigned only
  at the single commit at the end — another session may take the next number first. `ADR-077` in the
  spec is *provisional*; confirm the next free ADR + version + PR number against `main` at commit time.
  C4 aggregator count unchanged; `feature_glossary` count grows.
- **Single feature branch, single commit, single PR.** Branch `sb360-fov-observability` already exists.
  There are **no commit steps** in this plan; the user commits once, on explicit approval, at the end.
  The spec + this plan + the ADR go in that one commit (docs in the first/only commit — the
  `_provenance.py` untracked-is-dirty rule).
- **TDD / Hexagonal / e2e.** Every task is failing-test-first. Hexagonal: `_fov_registry.py` is a pure,
  neutral engine (`visible_area × region → (fraction, source)`); policy (opt-in kwarg, `space_creation`
  soften-vs-raise) lives at the aggregator/edge, never in the engine. e2e: unit tests use synthetic /
  committed fixtures (regular suite); real licensed-SB360-corpus validation is `@pytest.mark.e2e`
  (not run in CI — fixtures not committed), mirroring the ADR-062 licensed-corpus driver.
- Lint at CI scope only: `python -m ruff check silly_kicks/ tests/ scripts/`; `python -m pyright`.

---

## File Structure

- **Create** `silly_kicks/tracking/_fov_registry.py` — `ObservabilityEntry`, the convex region
  builders, `OBSERVABILITY_REGISTRY`, `_OBSERVABILITY_EXEMPT`, and `append_observability_companions()`
  (the one engine). Neutral imports only.
- **Modify** `silly_kicks/tracking/schema.py` — add `FovDiagnosis` (frozen dataclass) + the
  `FOV_REGIME_*` constants + `FOV_REGIME_VALUES`.
- **Modify** `silly_kicks/tracking/utils.py` — add `validate_fov(...)`.
- **Modify** `silly_kicks/tracking/__init__.py` — export `validate_fov`, `FovDiagnosis`,
  `FOV_REGIME_VALUES`.
- **Modify** `silly_kicks/tracking/features.py` — retire `_append_visibility_companions` to a thin
  call into the engine; add `visible_area=None` to `add_pressure_on_actor` (1148), `add_packing`
  (1524), `add_defensive_line` (1368), `add_team_shape` (2265), `add_player_influence` (4665),
  `add_xt_gk` (6773), `add_defensive_credit` (7040).
- **Modify** `silly_kicks/tracking/defensive_credit/_orchestration.py` — expose per-credit resolution
  region + mode so the engine can roll up (T1).
- **Modify** `silly_kicks/tracking/_space_creation.py` — retire the velocity FOV-proxy (line ~192).
- **Modify** `silly_kicks/feature_glossary.py` — the new companion columns.
- **Create** `tests/tracking/test_fov_diagnostic.py`, `tests/tracking/test_fov_registry.py`,
  `tests/tracking/test_fov_companions.py`, `tests/tracking/test_fov_completeness_gate.py`,
  `tests/tracking/test_space_creation_fov_migration.py`.
- **Modify** `tests/sb360/_registry.py::audited_surface` note; the ADR-033 `PURITY_ENTRIES`, the
  liveness/id-dtype/mirror registries auto-discover the newly-kwarg'd aggregators.

---

## Task 1: `validate_fov` / `FovDiagnosis` (Component 1)

**Files:**
- Modify: `silly_kicks/tracking/schema.py` (beside `VelocityRegimeDiagnosis` at :300)
- Modify: `silly_kicks/tracking/utils.py` (beside `validate_velocity_regime` at :610)
- Modify: `silly_kicks/tracking/__init__.py` (exports)
- Test: `tests/tracking/test_fov_diagnostic.py`

**Interfaces:**
- Produces:
  - `FOV_REGIME_FULL = "full_coverage"`, `FOV_REGIME_CROPPED = "fov_cropped"`,
    `FOV_REGIME_ABSENT = "absent"`, `FOV_REGIME_MIXED = "mixed"`, `FOV_REGIME_EMPTY = "empty"`;
    `FOV_REGIME_VALUES: tuple[str, ...]`.
  - `@dataclass(frozen=True) FovDiagnosis(regime: str, observed_pitch_fraction: dict[Any, float],
    source_counts: dict[str, int], n_actions: int, message: str)`.
  - `validate_fov(visible_area: pd.DataFrame, *, links: pd.DataFrame | None = None, full_coverage_floor:
    float = 0.98, on_mismatch: Literal["warn","raise","ignore"] = "raise") -> FovDiagnosis`.

- [ ] **Step 1: Write the failing tests** in `tests/tracking/test_fov_diagnostic.py`

```python
import numpy as np
import pandas as pd
import pytest
from silly_kicks.tracking import validate_fov, FovDiagnosis, FOV_REGIME_VALUES

_PITCH = np.array([[0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0]])
_LEFT_HALF = np.array([[0.0, 0.0], [52.5, 0.0], [52.5, 68.0], [0.0, 68.0]])

def _va(rows):  # rows: list[(action_id, polygon|None)]
    return pd.DataFrame({"action_id": [r[0] for r in rows], "polygon": [r[1] for r in rows]})

def test_full_coverage_regime():
    d = validate_fov(_va([(1, _PITCH), (2, _PITCH)]))
    assert d.regime == "full_coverage"
    assert d.n_actions == 2
    assert all(f >= 0.98 for f in d.observed_pitch_fraction.values())

def test_cropped_regime():
    d = validate_fov(_va([(1, _LEFT_HALF), (2, _LEFT_HALF)]))
    assert d.regime == "fov_cropped"
    assert round(d.observed_pitch_fraction[1], 3) == 0.5

def test_absent_regime():
    d = validate_fov(_va([(1, None), (2, np.zeros((2, 2)))]))  # None + degenerate
    assert d.regime == "absent"

def test_mixed_raises_by_default():
    with pytest.raises(ValueError):
        validate_fov(_va([(1, _PITCH), (2, _LEFT_HALF)]))

def test_mixed_warns_under_warn():
    with pytest.warns(UserWarning):
        d = validate_fov(_va([(1, _PITCH), (2, _LEFT_HALF)]), on_mismatch="warn")
    assert d.regime == "mixed"

def test_empty_never_raises():
    d = validate_fov(_va([]))
    assert d.regime == "empty" and d.n_actions == 0

def test_regime_in_vocabulary():
    assert validate_fov(_va([(1, _PITCH)])).regime in FOV_REGIME_VALUES
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/tracking/test_fov_diagnostic.py -x -q`
Expected: FAIL — `ImportError: cannot import name 'validate_fov'`.

- [ ] **Step 3: Add the dataclass + constants to `schema.py`**

Insert after `VelocityRegimeDiagnosis` (:324). Mirror its frozen-dataclass shape:

```python
FOV_REGIME_FULL = "full_coverage"
FOV_REGIME_CROPPED = "fov_cropped"
FOV_REGIME_ABSENT = "absent"
FOV_REGIME_MIXED = "mixed"
FOV_REGIME_EMPTY = "empty"
FOV_REGIME_VALUES: tuple[str, ...] = (
    FOV_REGIME_FULL, FOV_REGIME_CROPPED, FOV_REGIME_ABSENT, FOV_REGIME_MIXED, FOV_REGIME_EMPTY,
)


@dataclasses.dataclass(frozen=True)
class FovDiagnosis:
    """Whether a frame set's per-action visible_area is full / cropped / absent, before scoring.

    Fourth member of the validate_time_base / validate_velocity_regime / validate_id_dtypes family.
    FOV regime is a property of the WHOLE action set (like VelocityRegimeDiagnosis), so it is a
    diagnostic, not a per-row column; the per-action observed fractions that DO vary row-to-row are
    the observability companions, not this.

    Attributes:
        regime: one of FOV_REGIME_VALUES.
        observed_pitch_fraction: action_id -> observed pitch fraction (only for 'observed' actions).
        source_counts: visible_area_source token -> action count.
        n_actions: number of actions considered.
        message: human-readable summary, and the text of the raise when one occurs.
    """

    regime: str
    observed_pitch_fraction: dict
    source_counts: dict
    n_actions: int
    message: str
```

- [ ] **Step 4: Add `validate_fov` to `utils.py`**

Reuse `add_visible_area_coverage`'s per-action classification (it already emits `visible_area_fraction`
+ `visible_area_source`, clipping to the pitch — `_visibility.py:286-331`). `validate_fov` consumes
ONLY `visible_area` (+ optional `links` for the `unlinked` overlay) — no dead `frames`/`actions` param.

```python
def validate_fov(
    visible_area: pd.DataFrame,
    *,
    links: pd.DataFrame | None = None,
    full_coverage_floor: float = 0.98,
    on_mismatch: Literal["warn", "raise", "ignore"] = "raise",
) -> FovDiagnosis:
    """Report the field-of-view regime of a per-action visible_area table, before scoring.

    Empty input never raises; a `mixed` set (some actions full-coverage, others cropped/absent)
    raises (default) / warns / is silent, exactly like validate_velocity_regime's `mixed`.
    """
    from ._visibility import add_visible_area_coverage  # local: avoid import cycle

    n = 0 if visible_area is None else len(visible_area)
    if n == 0:
        return FovDiagnosis(FOV_REGIME_EMPTY, {}, {}, 0, "fov regime: empty visible_area.")
    actions = pd.DataFrame({"action_id": visible_area["action_id"].to_numpy()})
    cov = add_visible_area_coverage(actions, visible_area=visible_area, links=links)
    src = cov["visible_area_source"]
    frac = cov["visible_area_fraction"]
    source_counts = {str(k): int(v) for k, v in src.value_counts(dropna=False).items()}
    observed = src == VISIBLE_AREA_OBSERVED
    obs_frac = {aid: float(f) for aid, f, o in zip(cov["action_id"], frac, observed) if o}
    n_obs = int(observed.sum())
    n_full = int((observed & (frac >= full_coverage_floor)).sum())
    # Discriminator = full-COEXISTS-with-not-full (P1). `mixed` is the fail-loud case the spec defines
    # as "full-coverage actions mixed with cropped/absent ones", reachable ONLY by 0 < n_full < n.
    if n_obs == 0:
        regime = FOV_REGIME_ABSENT
    elif n_full == n:
        regime = FOV_REGIME_FULL
    elif n_full == 0:
        regime = FOV_REGIME_CROPPED   # no full action; partial polygons (+ any absent gaps of a partial provider)
    else:  # 0 < n_full < n: full-coverage actions coexist with cropped/absent ones
        regime = FOV_REGIME_MIXED
    message = f"fov regime: {regime} ({n_obs}/{n} observed, {n_full} full >= {full_coverage_floor})."
    if regime == FOV_REGIME_MIXED:
        message += " Some actions are full-coverage and others are cropped/absent; scoring the set as one FOV is incoherent."
        if on_mismatch == "raise":
            raise ValueError(message)
        if on_mismatch == "warn":
            warnings.warn(message, stacklevel=2)
    return FovDiagnosis(regime, obs_frac, source_counts, n, message)
```

Import `VISIBLE_AREA_OBSERVED` at the top of `utils.py` from `._visibility` if not already present.

- [ ] **Step 5: Export in `__init__.py`**

Add `validate_fov`, `FovDiagnosis`, `FOV_REGIME_VALUES` to the imports from `.utils`/`.schema` and to
`__all__`.

- [ ] **Step 6: Run tests to verify pass + full guard**

Run: `python -m pytest tests/tracking/test_fov_diagnostic.py -q`
Expected: PASS (7 tests).
Run: `python -m pytest --doctest-modules silly_kicks/tracking/utils.py -q` (no new doctest needed; a
literal-block Example is fine).

---

## Task 2: the observability registry + engine; retire ADR-062 helper byte-identically (Component 2)

**Files:**
- Create: `silly_kicks/tracking/_fov_registry.py`
- Modify: `silly_kicks/tracking/features.py` (`_append_visibility_companions` → thin call; :455-524)
- Test: `tests/tracking/test_fov_registry.py`

**Interfaces:**
- Produces:
  - `_NO_REGION = object()` — identity sentinel for "no measurable region" (P2), tested with `is`.
  - `@dataclass(frozen=True) ObservabilityEntry(column: str, region: Callable[[int, "RegionCtx"],
    np.ndarray | object], covers: tuple[str, ...] = ())` — `region(i, ctx)` returns a convex `(M,2)`
    polygon or `_NO_REGION` (→ `degenerate_region`), never a literal `None` (P7); `covers` is the RAW
    metric columns the companion annotates (default `(column,)`), consumed by the gate (R1).
  - `companioned_columns() -> set[str]` — every RAW column with a companion: each entry's `covers`
    (default `(column,)`) UNION `_CUSTOM_COMPANION_COVERS`. The gate's coverage source (Task 8).
  - `@dataclass RegionCtx(sx, sy, ex, ey, game_id, period_id, team_id, nearest_dist, goal_map, extras)`
    — per-action arrays the builders index with `i`; carries `game_id`/`period_id`/`team_id` for
    `GoalMap.get(game_id, period_id, team_id)` (P3).
  - `OBSERVABILITY_REGISTRY: dict[str, tuple[ObservabilityEntry, ...]]` keyed by aggregator name.
  - `_OBSERVABILITY_EXEMPT: dict[str, str]` — column → reason.
  - `append_observability_companions(out, actions, *, entries, visible_area, linked_ids, ctx:
    RegionCtx) -> pd.DataFrame` — the one engine; emits `<column>_observed_fraction` /
    `_observed_source` for each entry.
  - Convex builders `triangle_to_goal`, `receiver_disk` (reads `ctx.extras["receiver_radius"]`),
    `nearest_defender_disk`, and (Task 4) `pitch_zone(goal_map, game_id, period_id, team_id, which:
    Literal["own_half","attacking_half","defended_third"])`. Per-call params arrive via `ctx.extras`;
    entries are STATIC in `OBSERVABILITY_REGISTRY` (so the gate reads every column from one source).

- [ ] **Step 1: Write the parity test FIRST (freeze ADR-062 behaviour)** in `tests/tracking/test_fov_registry.py`

Use the committed SB360 paired fixture (`tests/sb360/_fixture.py`) or a small synthetic
`actions`+`frames`+`visible_area`. Assert the three ADR-062 companion columns are byte-identical
before/after the refactor:

```python
import numpy as np, pandas as pd
from silly_kicks.tracking import add_action_context
from tests.tracking._fov_fixtures import tiny_actions, tiny_frames, tiny_visible_area  # small helper

def test_adr062_companions_byte_identical_after_refactor():
    a, f, va = tiny_actions(), tiny_frames(), tiny_visible_area()
    out = add_action_context(a, f, visible_area=va)
    for col in ("nearest_defender_distance", "receiver_zone_density", "defenders_in_triangle_to_goal"):
        assert f"{col}_observed_fraction" in out.columns
        assert f"{col}_observed_source" in out.columns
    # golden values captured from the CURRENT implementation before refactor:
    golden = {  # fill from a pre-refactor run (Step 2)
        "defenders_in_triangle_to_goal_observed_fraction": [...],
        "receiver_zone_density_observed_fraction": [...],
        "nearest_defender_distance_observed_source": [...],
    }
    for col, vals in golden.items():
        np.testing.assert_array_equal(out[col].to_numpy(), np.array(vals, dtype=out[col].dtype))
```

- [ ] **Step 2: Capture the golden values from the CURRENT (pre-refactor) code**

Run a one-off in the venv: build `tiny_*`, call `add_action_context(..., visible_area=va)` on the
UNMODIFIED code, print the three columns, paste them into `golden`. Run the test → PASS on current
code (this freezes behaviour before restructuring — the fixture-generator discipline).

- [ ] **Step 3: Write `_fov_registry.py` — the engine + region builders + the three ADR-062 entries**

```python
"""Single-sourced FOV-observability: one convex region per metric, one engine (ADR-077).

Neutral: imports only _visibility, _geometry, _kernels. No pitch_control / _das.
"""
from __future__ import annotations
import dataclasses
from typing import Callable
import numpy as np
import pandas as pd
from silly_kicks.id_compat import canonical_id
from ._visibility import (
    classify_region_observation, _polygons_by_action,
    REGION_OBSERVATION_DEGENERATE_REGION, VISIBLE_AREA_UNLINKED,
)
from . import _kernels

#: Identity sentinel: a builder returns this when NO measurable region exists for this action
#: (NaN nearest-distance -> no radius; an unsupported/velocity pressure method -> no convex ROI;
#: an anchor_actor credit -> event-resolved; an unresolved defended end). Tested with `is`, NEVER
#: `==` -- `np.zeros((16, 2)) == "degenerate"` is an ARRAY and `if <array>:` RAISES (P2, executed).
_NO_REGION = object()


@dataclasses.dataclass
class RegionCtx:
    sx: np.ndarray; sy: np.ndarray; ex: np.ndarray; ey: np.ndarray
    game_id: np.ndarray; period_id: np.ndarray; team_id: np.ndarray   # all three needed for GoalMap.get (P3)
    nearest_dist: np.ndarray | None
    goal_map: object | None                                            # a GoalMap, or None
    extras: dict


@dataclasses.dataclass(frozen=True)
class ObservabilityEntry:
    column: str                    # companion KEY -> emits `<column>_observed_fraction` / `_source`
    region: Callable               # (i, ctx) -> (M,2) convex ndarray | _NO_REGION  (never None, P7)
    covers: tuple[str, ...] = ()   # RAW metric columns this companion annotates; () -> (column,).
                                   # Lets one companion cover several columns (team_shape x/y per role;
                                   # a rollup) so the completeness gate maps raw columns correctly (R1).


def triangle_to_goal(i, ctx):
    return np.array([[ctx.sx[i], ctx.sy[i]],
                     [_kernels._GOAL_X, _kernels._GOAL_LEFT_POST_Y],
                     [_kernels._GOAL_X, _kernels._GOAL_RIGHT_POST_Y]])

def receiver_disk(i, ctx):
    return _kernels._inscribed_disk(ctx.ex[i], ctx.ey[i], ctx.extras["receiver_radius"])

def nearest_defender_disk(i, ctx):
    d = ctx.nearest_dist[i]
    if not np.isfinite(d):
        return _NO_REGION                       # no radius -> no region (P2 sentinel)
    return _kernels._inscribed_disk(ctx.sx[i], ctx.sy[i], d)


def append_observability_companions(out, actions, *, entries, visible_area, linked_ids, ctx):
    polygons = _polygons_by_action(visible_area)
    n = len(actions)
    for e in entries:
        fracs = np.full(n, np.nan)
        sources: list[str] = []
        for i, aid in enumerate(actions["action_id"]):
            key = canonical_id(aid)
            if linked_ids is not None and key not in linked_ids:
                sources.append(VISIBLE_AREA_UNLINKED); continue
            region = e.region(i, ctx)
            if region is _NO_REGION:                                    # identity, NEVER == (P2)
                sources.append(REGION_OBSERVATION_DEGENERATE_REGION); continue
            frac, s = classify_region_observation(polygons.get(key), region)  # polygon None -> no_polygon
            fracs[i] = frac; sources.append(s)
        out[f"{e.column}_observed_fraction"] = fracs
        out[f"{e.column}_observed_source"] = sources
    return out


OBSERVABILITY_REGISTRY: dict = {
    # Every metric's region is STATIC here; per-call params (radius, pressure_method) arrive via
    # ctx.extras, so the completeness gate (Task 8) reads every companioned column from ONE source.
    # Later tasks assign the other aggregator keys.
    "add_action_context": (
        ObservabilityEntry("defenders_in_triangle_to_goal", triangle_to_goal),
        ObservabilityEntry("receiver_zone_density", receiver_disk),          # reads ctx.extras["receiver_radius"]
        ObservabilityEntry("nearest_defender_distance", nearest_defender_disk),
    ),
}
_OBSERVABILITY_EXEMPT: dict[str, str] = {}  # filled in Task 5 (xt_gk composite) + Task 8 (ghost_gk)
_CUSTOM_COMPANION_COVERS: set[str] = set()  # raw columns companioned by a NON-engine path (Task 6)


def companioned_columns() -> set[str]:
    """Every RAW metric column that receives a companion (engine or custom) -- the gate's coverage set."""
    cols = {c for ents in OBSERVABILITY_REGISTRY.values() for e in ents for c in (e.covers or (e.column,))}
    return cols | _CUSTOM_COMPANION_COVERS
```

(Note: params that vary per call — `receiver_radius`, later `pressure_method` — arrive via
`ctx.extras`, so the registry entries stay STATIC and the completeness gate (Task 8) reads every
companioned column from the single `OBSERVABILITY_REGISTRY`.)

- [ ] **Step 4: Retire `_append_visibility_companions` in `features.py` to a thin call**

Replace the body (:477-524) with construction of a `RegionCtx` (carrying `game_id`/`period_id`/
`team_id` arrays + `extras={"receiver_radius": receiver_zone_radius}`) + `append_observability_companions(
..., entries=OBSERVABILITY_REGISTRY["add_action_context"], ...)`. Keep the signature and the `linked`
set build (`ctx.pointers`). The emitted columns/values must be unchanged.

- [ ] **Step 5: Run the parity test + the whole add_action_context suite**

Run: `python -m pytest tests/tracking/test_fov_registry.py tests/tracking/test_action_context*.py -q`
Expected: PASS (byte-identical companions).

- [ ] **Step 6: id-dtype + purity re-check for add_action_context** (it already has both branches
  registered from ADR-062; confirm still green)

Run: `python -m pytest tests/test_add_star_purity.py tests/tracking/test_id_dtype_invariance.py -q -k action_context`
Expected: PASS.

---

## Task 3: region-count companions — `add_pressure_on_actor`, `add_packing`

**Files:**
- Modify: `silly_kicks/tracking/features.py` (`add_pressure_on_actor` :1148; `add_packing` :1524)
- Modify: `silly_kicks/tracking/_fov_registry.py` (register their entries)
- Modify: `silly_kicks/tracking/_pressure`? — the Andrienko oval region builder (see Step 1)
- Test: `tests/tracking/test_fov_companions.py`

**Interfaces:**
- Consumes: `append_observability_companions`, `ObservabilityEntry`, `RegionCtx` (Task 2).
- Produces: `andrienko_oval_region(i, ctx)` and `packing_zone_region(i, ctx)` convex builders;
  registry entries under `"add_pressure_on_actor"` and `"add_packing"`.

- [ ] **Step 1: Grounding read — obtain the two ROIs**

Read `_kernels._pressure_andrienko` (`_kernels.py:393`) to extract the oval geometry (centre = actor,
semi-axes from `AndrienkoParams`); read `compute_packing_metrics` (`_packing.py:119`) to obtain the
packing zone (the ball→goal corridor polygon). Record the exact parameters so the region builders
reproduce them, and record **`_PACKING_REGION_COUNT_COLUMNS`** — the subset of `add_packing`'s emitted
columns that are region-counts (only these get companions; P9). Confirm both regions are convex.

- [ ] **Step 2: Write failing both-sides tests** in `tests/tracking/test_fov_companions.py`

```python
def test_pressure_companion_half_crop():
    a, f, va_full, va_left = _wide_action_fixture()  # actor mid-pitch; oval spans the halfway line
    full = add_pressure_on_actor(a, f, method="andrienko_oval", visible_area=va_full)
    left = add_pressure_on_actor(a, f, method="andrienko_oval", visible_area=va_left)
    base = add_pressure_on_actor(a, f, method="andrienko_oval")  # no visible_area
    # primary byte-identical:
    np.testing.assert_array_equal(
        full["pressure_on_actor__andrienko_oval"].to_numpy(),
        base["pressure_on_actor__andrienko_oval"].to_numpy())
    # companion moves out of band under crop, ==1 under full:
    assert full["pressure_on_actor__andrienko_oval_observed_fraction"].iloc[0] == 1.0
    assert left["pressure_on_actor__andrienko_oval_observed_fraction"].iloc[0] < 1.0
```

(Analogous `test_packing_companion_half_crop`.)

- [ ] **Step 3: Run → FAIL** (`KeyError` on the companion column).

- [ ] **Step 4: Add the region builders + registry entries in `_fov_registry.py`**

```python
def _convex_ellipse(cx, cy, ax, ay, k=16):   # small new helper: convex k-gon approximating an ellipse
    t = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)
    return np.column_stack([cx + ax * np.cos(t), cy + ay * np.sin(t)])

def andrienko_oval_region(i, ctx):
    return _convex_ellipse(ctx.sx[i], ctx.sy[i], ctx.extras["oval_ax"], ctx.extras["oval_ay"])

def packing_zone_region(i, ctx):
    return ctx.extras["packing_zone"][i]  # precomputed convex ball->goal corridor per action (Step 1)

OBSERVABILITY_REGISTRY["add_pressure_on_actor"] = (
    ObservabilityEntry("pressure_on_actor__andrienko_oval", andrienko_oval_region),   # region_support
)
# P9: `add_packing` emits several columns; ONLY the region-count members get companions. The exact
# names are pinned in Step 1's read of compute_packing_metrics (_packing.py:119) as
# _PACKING_REGION_COUNT_COLUMNS -- do NOT guess "packing_made". These are VOLUNTARY companions
# (packing is not in the region_support set), which the Task 8 gate permits (registered >= region_support).
OBSERVABILITY_REGISTRY["add_packing"] = tuple(
    ObservabilityEntry(col, packing_zone_region) for col in _PACKING_REGION_COUNT_COLUMNS
)
```

- [ ] **Step 5: Thread the opt-in kwarg into both aggregators**

Add `visible_area: pd.DataFrame | None = None` to `add_pressure_on_actor` / `add_packing`; after
computing the primary columns, `if visible_area is not None:` build `RegionCtx` + call
`append_observability_companions(..., entries=OBSERVABILITY_REGISTRY["add_pressure_on_actor"], ...)`.
For `bekkers_pi` / non-andrienko methods, do NOT register a companion (the pressure entry is
method-specific; see Task 5's dispatch note — for `add_pressure_on_actor` the companion is emitted only
when `method == "andrienko_oval"`).

- [ ] **Step 6: Run → PASS**; then purity (two branches) + liveness + id-dtype for both aggregators.

Run: `python -m pytest tests/tracking/test_fov_companions.py tests/test_add_star_purity.py -q -k "pressure or packing"`

- [ ] **Step 7: Register the ADR-033 two-branch purity variants** for both aggregators in
  `PURITY_ENTRIES` (companion-present + companion-absent). Run `tests/test_add_star_purity.py` → PASS.

---

## Task 4: aggregate-position companions — `add_defensive_line`, `add_team_shape`, `add_player_influence` (+ N2)

> **⚠️ SUPERSEDED — this task's `goal_map` / `pitch_zone` mechanism was REJECTED during execution and REPLACED by ADR-077 "Design (A)": FIXED action-LTR pitch bands keyed on the column ROLE (no `goal_map`).** A `goal_map` lookup returns *frame-coordinate* ends, mis-orienting every away-possession action against the action-LTR `visible_area` polygon — the S1 silent-failure this cycle exists to prevent (in action-LTR the acting team always attacks x=105, so the defended end is FIXED per role). **None** of the `pitch_zone(goal_map, …)` / `GoalMap.get` / `allow_guess` machinery, the `RegionCtx.{game_id, period_id, team_id, goal_map}` fields, or the tests `test_zone_sits_on_the_correct_end` / `test_unresolved_end_yields_no_region` described below exist in the code. The IMPLEMENTED design is the fixed-zone builders (`defended_third_region` / `attacking_own_half_region` / `defending_own_half_region` / `attacking_half_region`) in `_fov_registry.py`, gated by `test_zones_sit_on_the_correct_action_ltr_ends` + the `test_*_invariant_to_outfield_shift` frame-independence tests. Authoritative record: **[ADR-077 §"Design (A)"](../adrs/ADR-077-fov-observability.md)** (records `goal_map` as rejected option C). The text below is retained as the historical design record only.

**Files:**
- Modify: `silly_kicks/tracking/features.py` (:1368, :2265, :4665)
- Modify: `silly_kicks/tracking/_fov_registry.py` (`pitch_zone` builder + entries + the N2 docstring)
- Test: `tests/tracking/test_fov_companions.py`

**Interfaces:**
- Produces: `pitch_zone(goal_map, team_id_i, which)` — a convex, **goal-keyed, frame-independent**
  rectangle (`defended_third` / `own_half` / `attacking_half`); registry entries for the three
  aggregators.

- [ ] **Step 1: Grounding read (P6/R2/M-guess — team + defended end + allow_guess per metric).** For
  each aggregator, record from its `compute_*`: **(a) the exact team(s)** whose players form the
  estimate and the `ctx.extras` team-id array to key on — `defensive_line_x` = the **DEFENDING** team
  (`defending_team_id`), its defended third (NOT the actor's); `add_team_shape` emits **FOUR** columns
  split by ROLE on OPPOSITE ends (`team_shape_centroid_{x,y}_{attacking,defending}`,
  `features.py:2236-2243`) → **two** companions, `attacking` keyed on `attacking_team_id` and
  `defending` on `defending_team_id` (there is NO `team_shape_centroid` column); `off_ball_xt_team` =
  the **attacking** team (`compute_player_influence(attacking_team_id=...)`), attacking half. **(b)** how
  each resolves its team's defended end — each consumes a `GoalMap` from `resolve_defended_goals(frames)`
  for its primary compute (`compute_defensive_line(frames, goal_map=, n=)`, `_kernels.py:863`); **reuse
  that same `GoalMap`** (P3) and record its **`allow_guess`** so `pitch_zone` matches (M-guess). Populate
  `ctx.extras` with `attacking_team_id` / `defending_team_id` arrays + `allow_guess`. N2 note: the
  `GoalMap` only picks the END (robust per-`(game,period,team)` mean-x); the zone extent stays fixed
  pitch-thirds, so N2 holds.

- [ ] **Step 2: Write the S1-regression both-sides test + the N2 frame-independence test**

```python
def test_defensive_line_area_companion_and_bias():
    # 4-man line across the width; one defender in the far half is CROPPED OUT (absent from the frame),
    # biasing the mean. visible_area = the defended half.
    a, f_full, f_cropframe, va = _line_fixture()
    full = add_defensive_line(a, f_full, visible_area=_full_pitch_va(a))
    crop = add_defensive_line(a, f_cropframe, visible_area=va)  # frame lacks the cropped defender
    assert crop["defensive_line_x"].iloc[0] != full["defensive_line_x"].iloc[0]  # bias demonstrated
    assert crop["defensive_line_x_observed_fraction"].iloc[0] < 1.0               # honest, < 1
    assert full["defensive_line_x_observed_fraction"].iloc[0] == 1.0

def test_companion_invariant_to_outfield_permutation():   # P4: the REAL frame-independence guard
    # Keep the GK rows fixed (so the frame-derived GoalMap is unchanged); permute the OUTFIELD player
    # rows. The companion fraction must be unchanged EVEN THOUGH the primary defensive_line_x moves --
    # proving the zone is GoalMap-keyed, not drawn around the observed outfielders (the S1 regression
    # the tautological pitch_zone()-twice test misses).
    a, f, va = _line_fixture_gk_fixed()
    base = add_defensive_line(a, f, visible_area=va)
    perm = add_defensive_line(a, _permute_outfield_rows(f), visible_area=va)
    np.testing.assert_array_equal(
        base["defensive_line_x_observed_fraction"].to_numpy(),
        perm["defensive_line_x_observed_fraction"].to_numpy())

def test_zone_sits_on_the_correct_end():   # P6: a wrong-team/wrong-third choice passes frame-independence
    from silly_kicks.tracking._fov_registry import pitch_zone
    gm = _goal_map_fixture()   # team 1 defends the x=0 end
    z = pitch_zone(gm, game_id=1, period_id=1, team_id=1, which="defended_third")
    assert z[:, 0].min() == 0.0 and z[:, 0].max() <= 35.0   # defended third sits at the x=0 end

def test_unresolved_end_yields_no_region():   # P3: GoalMap.get -> None must not fabricate a zone
    from silly_kicks.tracking._fov_registry import pitch_zone, _NO_REGION
    gm = _goal_map_unresolved_for_team(3)
    assert pitch_zone(gm, game_id=1, period_id=1, team_id=3, which="own_half") is _NO_REGION

def test_team_shape_two_role_companions():   # R2: four real columns -> two role-keyed companions
    out = add_team_shape(a, f, visible_area=va)
    assert "team_shape_centroid_attacking_observed_fraction" in out.columns   # annotates x/y_attacking
    assert "team_shape_centroid_defending_observed_fraction" in out.columns   # annotates x/y_defending
    assert "team_shape_centroid_observed_fraction" not in out.columns          # the nonexistent-column bug
    # the two roles' own halves are on OPPOSITE ends, so under a one-sided crop the companions differ:
    assert out["team_shape_centroid_attacking_observed_fraction"].iloc[0] != \
           out["team_shape_centroid_defending_observed_fraction"].iloc[0]
```

- [ ] **Step 3: Run → FAIL.**

- [ ] **Step 4: Add `pitch_zone` (convex, goal-keyed) + entries**

```python
def pitch_zone(goal_map, game_id, period_id, team_id, which, *, allow_guess=False):
    """Convex, FRAME-INDEPENDENT rectangle keyed on the DEFENDED goal (N2/P3). NEVER from player coords.

    Uses the aggregator's OWN GoalMap (resolve_defended_goals(frames)); only the L/R mirror comes from
    its robust per-(game,period,team) mean-x binary, never from the in-FOV subset. `allow_guess` MUST
    match whatever the primary metric passes (M-guess) -- else the companion is degenerate exactly where
    the metric is populated. An unresolved end (GoalMap.get -> None) returns _NO_REGION.
    """
    defended = goal_map.get(game_id, period_id, team_id, allow_guess=allow_guess)  # 0.0 | 105.0 | None (P3)
    if defended is None:
        return _NO_REGION
    low_end = defended == 0.0
    if which == "defended_third":
        lo, hi = (0.0, 35.0) if low_end else (70.0, 105.0)
    elif which == "own_half":
        lo, hi = (0.0, 52.5) if low_end else (52.5, 105.0)
    else:  # attacking_half
        lo, hi = (52.5, 105.0) if low_end else (0.0, 52.5)
    return np.array([[lo, 0.0], [hi, 0.0], [hi, 68.0], [lo, 68.0]])

def _zone(which, team_key):
    # team_key selects the RIGHT team per metric from ctx.extras (R2: NOT always the acting team).
    return lambda i, ctx: pitch_zone(
        ctx.goal_map, ctx.game_id[i], ctx.period_id[i], ctx.extras[team_key][i], which,
        allow_guess=ctx.extras.get("allow_guess", False))

# defensive_line = the DEFENDING team's line (P6) -> its defended third.
OBSERVABILITY_REGISTRY["add_defensive_line"] = (
    ObservabilityEntry("defensive_line_x", _zone("defended_third", "defending_team_id")),)
# R2: add_team_shape emits FOUR centroid columns split by team ROLE on OPPOSITE ends -> TWO companions,
# one per role, each keyed to that role's team. P8 collapses x/y WITHIN a role, not across roles. There
# is NO `team_shape_centroid` column. (Zone per role is grounding-refinable, N1 residual.)
OBSERVABILITY_REGISTRY["add_team_shape"] = (
    ObservabilityEntry("team_shape_centroid_attacking", _zone("own_half", "attacking_team_id"),
        covers=("team_shape_centroid_x_attacking", "team_shape_centroid_y_attacking")),
    ObservabilityEntry("team_shape_centroid_defending", _zone("own_half", "defending_team_id"),
        covers=("team_shape_centroid_x_defending", "team_shape_centroid_y_defending")),)
# off_ball_xt_team = the ATTACKING team (compute_player_influence(attacking_team_id=...)) -> attacking half.
OBSERVABILITY_REGISTRY["add_player_influence"] = (
    ObservabilityEntry("off_ball_xt_team", _zone("attacking_half", "attacking_team_id")),)
```

- [ ] **Step 5: Thread the opt-in kwarg** into the three aggregators (build `RegionCtx` with
  `goal_map` + per-action `team_id`; call the engine). Primary columns unchanged.

- [ ] **Step 6: Run → PASS**; register two-branch purity variants; run purity + liveness + id-dtype +
  the mirror registry (all auto-discover the newly-kwarg'd aggregators).

---

## Task 5: `add_xt_gk` companions — method-dispatched ROI, effective-support bound, composite exemption (T2/N3/M1)

**Files:**
- Modify: `silly_kicks/tracking/features.py` (`add_xt_gk` :6773)
- Modify: `silly_kicks/tracking/_fov_registry.py` (method-dispatch region + `link_zones` bound + exempt)
- Test: `tests/tracking/test_fov_companions.py`

**Interfaces:**
- Produces: `xt_gk_pressure_region(pressure_method, link_params)` — dispatches to
  `andrienko_oval_region` / `link_zones_support_region` / returns `None` (absent) for unsupported/
  velocity methods; registry entries for `xt_gk_pressure`, `xt_gk_pev`; `_OBSERVABILITY_EXEMPT["xt_gk"]`.

- [ ] **Step 1: Grounding read** — `_xt_gk.py:526` (pressure via `p.pressure_method`), `_pressure_link`
  (`_kernels.py:476`) + `LinkParams` (`pressure.py:36`) for the **effective-support radius** (largest
  non-negligible zone radius / cutoff). Confirm the convex outer bound (disk/oval) that contains the
  aggregation's support.

- [ ] **Step 2: Write tests** — companion present/populated under `andrienko_oval` AND `link_zones` on
  a cropped fixture (N3: NOT silently absent); `bekkers_pi` → no companion (honest-NaN pressure);
  no `xt_gk_observed_fraction` composite column (M1); `"xt_gk" in _OBSERVABILITY_EXEMPT`.

```python
def test_xt_gk_link_zones_companion_present_and_populated():
    out = add_xt_gk(a, f, xt=xt_grid, pressure_method="link_zones", visible_area=va_cropped)
    frac = out["xt_gk_pressure_observed_fraction"]
    assert frac.notna().any() and (frac.dropna() < 1.0).any()   # present + populated, not absent
    assert "xt_gk_observed_fraction" not in out.columns          # M1: no composite fraction
```

- [ ] **Step 3: Run → FAIL.**

- [ ] **Step 4: Implement the method-dispatch region + `link_zones` effective-support bound**

```python
def link_zones_support_region(i, ctx):
    r = ctx.extras["link_effective_radius"]        # from LinkParams (Step 1); convex disk
    return _kernels._inscribed_disk(ctx.sx[i], ctx.sy[i], r)

def xt_gk_pressure_region(i, ctx):
    method = ctx.extras["pressure_method"]          # dispatch PER ACTION from ctx.extras (static entry, P7)
    if method == "andrienko_oval":
        return andrienko_oval_region(i, ctx)
    if method == "link_zones":
        return link_zones_support_region(i, ctx)
    return _NO_REGION                               # bekkers_pi / unsupported -> degenerate_region (P2)

OBSERVABILITY_REGISTRY["add_xt_gk"] = (
    ObservabilityEntry("xt_gk_pressure", xt_gk_pressure_region),
    ObservabilityEntry("xt_gk_pev", xt_gk_pressure_region),
)
_OBSERVABILITY_EXEMPT["xt_gk"] = (
    "composite of a region-dependent gamma*pev term and GK-geometry base/rav/dzv; "
    "region-dependent part is covered by xt_gk_pev_observed_fraction; no honest single fraction (M1)."
)
```

`add_xt_gk` builds `ctx.extras = {"pressure_method": p.pressure_method, "link_effective_radius": ...,
"oval_ax": ..., "oval_ay": ...}` and calls the engine with `entries=OBSERVABILITY_REGISTRY["add_xt_gk"]`.
Emit `xt_gk_pressure_observed_fraction` + `xt_gk_pev_observed_fraction` ONLY (no composite, M1).

- [ ] **Step 5: Thread the opt-in kwarg into `add_xt_gk`** (frozen model untouched — only companion
  columns appended). Run → PASS; register purity variants; run the invariant gates.

---

## Task 6: `add_defensive_credit` — resolution-mode-aware corridor rollup (T1)

**Files:**
- Modify: `silly_kicks/tracking/defensive_credit/_orchestration.py` (expose per-credit region+mode)
- Modify: `silly_kicks/tracking/features.py` (`add_defensive_credit` :7040 — opt-in kwarg + rollup)
- Modify: `silly_kicks/tracking/_fov_registry.py` (the per-mode region + per-action rollup)
- Test: `tests/tracking/test_fov_companions.py`

**Interfaces:**
- Consumes: the six `RESOLUTION_*` tokens (`_params.py:59-72`).
- Produces: `defensive_credit_region_for_mode(mode, credit) -> region | None`; a per-action rollup
  that emits ONE `defensive_credit_observed_fraction` + `_observed_source` for the credit family.

- [ ] **Step 1: Grounding read** — `_resolution.py` for how each mode resolves its defender(s): `lane`
  / `all_within` / `all_within_beyond_nearest` → a corridor/region; `nearest` / `nearest_fallback` →
  the nearest defender at the resolved origin distance (→ inscribed disk, reuse
  `nearest_defender_disk`); `anchor_actor` → event-resolved (no region). Record the corridor geometry
  and where the resolved origin/distance is available per credit.

- [ ] **Step 2: Write the mixed-mode both-sides test**

```python
def test_defensive_credit_mode_aware_rollup():
    out = add_defensive_credit(a, f, visible_area=va_cropped)   # actions carrying lane + nearest + anchor_actor credits
    frac = out["defensive_credit_observed_fraction"]
    assert (frac.dropna() < 1.0).any()          # a lane/nearest credit's region is partly outside FOV
    assert np.isnan(frac.iloc[_anchor_only_row])  # anchor_actor-only action -> N/A, never a spurious 1.0

def test_rollup_is_magnitude_weighted():   # P5: pin the aggregation formula
    # ONE action, two region-bearing credits: magnitude 3.0 at observed_fraction 0.4, magnitude 1.0 at 1.0
    # -> weighted (3*0.4 + 1*1.0)/(3+1) = 0.55; anchor_actor credits excluded from BOTH sums.
    out = add_defensive_credit(a2, f2, visible_area=va2)
    assert round(float(out["defensive_credit_observed_fraction"].iloc[0]), 4) == 0.55
```

- [ ] **Step 3: Run → FAIL.**

- [ ] **Step 4: Implement.** Expose per-credit `(mode, resolved_origin, distance, corridor, magnitude)`
  from `_orchestration.py`. Map each credit → region by mode: `lane` / `all_within` /
  `all_within_beyond_nearest` → the corridor region; `nearest` / `nearest_fallback` →
  `_kernels._inscribed_disk` at the resolved origin/distance; `anchor_actor` → **`_NO_REGION`** (P2 —
  identity sentinel, event-resolved). Roll up per action as a **credit-magnitude-weighted mean** (P5):
  `sum(mag_c * frac_c) / sum(mag_c)` over credits whose region is not `_NO_REGION` (`anchor_actor`
  excluded from BOTH sums). An action with only `anchor_actor` credits (empty weight sum) → NaN
  fraction + `degenerate_region` source. `magnitude` = the credit's contribution to `net`/`minus`
  (confirm the exact field in Step 1). Emit ONE `defensive_credit_observed_fraction` + `_observed_source`
  for the credit family. Because this rollup is a CUSTOM path (not the generic engine), declare its
  coverage at **MODULE LOAD** in `_fov_registry.py` — `_CUSTOM_COMPANION_COVERS |=
  {"defensive_credit_net", "defensive_credit_minus", "n_defensive_credits"}` as a **module-level
  statement, NOT inside `add_defensive_credit`** (else the gate, which imports `_fov_registry`, reads an
  empty set until the aggregator first runs) — so the completeness gate (Task 8) counts these three
  `region_support` columns as covered.

- [ ] **Step 5: Run → PASS**; purity variants; invariant gates. `add_defensive_credit` keeps its
  "no xfns" status (a companion is not an xfns factory).

---

## Task 7: retire the `space_creation` velocity FOV-proxy (Component 4, M4)

**Files:**
- Modify: `silly_kicks/tracking/_space_creation.py` (:187-194, :335-338)
- Test: `tests/tracking/test_space_creation_fov_migration.py`

**Interfaces:**
- Consumes: `validate_fov` / per-action observed fraction (Task 1); the per-action polygon.

- [ ] **Step 1: Grounding read** — `_space_creation.py:180-200` + `:290-340` to see exactly how the
  `_mode = "nan" if velocity_unavailable_by_design(frame) else "raise"` gate and the
  `space_opponent_source` label are wired, and how `visible_area` can reach this function (add an
  opt-in kwarg, or accept the per-action FOV signal).

- [ ] **Step 2: Write the M4 synthetic both-sides tests** (fixtures decouple velocity ⊥ FOV)

```python
def test_velocity_less_but_full_coverage_now_raises():
    # velocity-unavailable frame BUT full-pitch visible_area -> not an FOV crop -> must RAISE
    with pytest.raises(ValueError):
        add_space_creation(a, f_velless, visible_area=_full_pitch_va(a))

def test_velocity_bearing_but_cropped_now_softens():
    # velocity-bearing frame BUT half-pitch visible_area -> real FOV crop -> soften, not raise
    out = add_space_creation(a, f_velbearing, visible_area=_left_half_va(a))
    assert (out["space_opponent_source"] == "unresolved_one_team").any()

def test_behaviour_flips_versus_old_velocity_proxy():
    # the same two inputs under the OLD proxy would behave oppositely; assert the NEW result differs
    ...
```

- [ ] **Step 3: Run → FAIL** (old proxy raises/softens on the wrong axis).

- [ ] **Step 4: Implement** — replace the `velocity_unavailable_by_design(frame)` gate with a decision
  driven by the real per-action FOV signal (polygon present ∧ `fov_cropped` for that action's region →
  soften; else `raise`). Remove the velocity-proxy import if now unused. Keep `space_opponent_source`
  values `{resolved, unresolved_one_team}` unchanged.

- [ ] **Step 5: Run → PASS**; re-run the existing `_space_creation` suite for no regression on
  full-tracking (velocity-bearing, full-coverage) inputs.

---

## Task 8: the completeness gate (Component 5, M2, M3) + ghost_gk exemption

**Files:**
- Create: `tests/tracking/test_fov_completeness_gate.py`
- Modify: `silly_kicks/tracking/_fov_registry.py` (`_OBSERVABILITY_EXEMPT["ghost_gk_x"/"ghost_gk_y"]`)

**Interfaces:**
- Consumes: the SB360 audit's `region_support` column set (`tests/sb360/_entries` +
  `_vocabulary.py:115`); `OBSERVABILITY_REGISTRY`; `_OBSERVABILITY_EXEMPT`.

- [ ] **Step 1: Declare the ghost_gk exemptions + the aggregate-FOV-sensitive bucket (R1)** in `_fov_registry.py`:

```python
_OBSERVABILITY_EXEMPT["ghost_gk_x"] = _OBSERVABILITY_EXEMPT["ghost_gk_y"] = (
    "learned model; FOV dependence is its whole-frame receptive field, no single clean ROI; "
    "a whole-pitch fraction would over-simplify. Bespoke ghost-observability model is a later cycle."
)

# R1: the SB360 `region_support` tag is a SINGLE-PLAYER-PERTURBATION axis -- a mean-over-many is robust
# to it (tagged `no_support`), a DIFFERENT axis from FOV-crop bias (S1: remove a player by cropping). So
# the perturbation probe CANNOT reach the aggregate metrics this cycle exists for (MEASURED: defensive_
# line_x / team_shape_* / packing_* are `no_support`; only off_ball_xt_team is `region_support`). This
# HAND-CURATED bucket is the FOV-crop axis, gate-enforced INDEPENDENTLY. It is a manual-discipline
# surface (like ADR-054's `_GUARD_EXEMPT`): a NEW aggregate/region metric must be added here to be
# gate-forced -- the M3 plant proves the enforcement fires.
_AGGREGATE_FOV_SENSITIVE: frozenset[str] = frozenset({
    "defensive_line_x",
    "team_shape_centroid_x_attacking", "team_shape_centroid_y_attacking",
    "team_shape_centroid_x_defending", "team_shape_centroid_y_defending",
    "off_ball_xt_team",
    *_PACKING_REGION_COUNT_COLUMNS,   # Task 3 Step 1
})
```

- [ ] **Step 2: Write the gate — landed RED first**

```python
def _region_support_columns():
    # derive structurally from tests/sb360/_entries (applicability == "region_support")
    ...

def _required_columns():   # R1: BOTH axes -- perturbation-sensitive (audit) AND FOV-crop-sensitive (curated)
    from silly_kicks.tracking._fov_registry import _AGGREGATE_FOV_SENSITIVE
    return set(_region_support_columns()) | set(_AGGREGATE_FOV_SENSITIVE)

def test_every_required_column_registered_or_exempt():
    from silly_kicks.tracking._fov_registry import companioned_columns, _OBSERVABILITY_EXEMPT
    covered = companioned_columns() | set(_OBSERVABILITY_EXEMPT)   # maps RAW columns via `covers` (R1)
    missing = _required_columns() - covered
    assert not missing, f"FOV-sensitive columns with no companion or exemption: {missing}"

def test_no_stale_exemption():
    # exempt MUST be a subset of REQUIRED (region_support u aggregate) -- no exemption for an unflagged
    # column. NOTE: companioned_columns() MAY EXCEED required -- extra tight-ROI companions
    # (receiver_zone_density / defenders_in_triangle_to_goal) are allowed, unforced.
    from silly_kicks.tracking._fov_registry import _OBSERVABILITY_EXEMPT
    stale = set(_OBSERVABILITY_EXEMPT) - _required_columns()
    assert not stale, f"exemptions for non-required columns (stale): {stale}"

def test_gate_scope_justification_present():  # M2
    # support_data_defined columns (actor_*_pre_window, elastic_confidence) are temporal, not area;
    # assert the gate module documents this exclusion (a constant / docstring the test reads).
    ...

def test_non_vacuity_plant():  # M3 -- the detector fires for a NEW required column
    # a synthetic column spelled to appear in NO committed entry: prove that IF it were required it would
    # be flagged missing, so the gate cannot pass by merely knowing today's population.
    from silly_kicks.tracking._fov_registry import companioned_columns, _OBSERVABILITY_EXEMPT
    synthetic = "zzz_synthetic_fov_probe_col"
    covered = companioned_columns() | set(_OBSERVABILITY_EXEMPT)
    assert synthetic not in covered
    assert synthetic in ({synthetic} | _required_columns()) - covered   # would be RED if required
```

- [ ] **Step 3: Run → RED** (some region_support columns not yet covered); then fill any real gaps
  (all should be covered by Tasks 3–6 + the two ghost_gk exemptions + the xt_gk-composite exemption) →
  GREEN. Observe the M3 plant fail if you temporarily add the synthetic column to the population.

---

## Task 9: glossary, SB360-audit note, invariant registrations, docs, version bump

**Files:**
- Modify: `silly_kicks/feature_glossary.py` (new companion columns)
- Modify: `tests/sb360/_registry.py::audited_surface` (companions are opt-in → outside the default audit)
- Modify: `PURITY_ENTRIES`, mirror/liveness/id-dtype registries (if not already done per-task)
- Create: `docs/superpowers/adrs/ADR-077-fov-observability.md`
- Modify: `CHANGELOG.md`, `CLAUDE.md` (Key-conventions line), C4 glossary count in `docs/c4/`
- Modify: the FIVE version sites incl. `uv.lock`

- [ ] **Step 1: Glossary entries** — one `FeatureColumn` per new companion column
  (`<metric>_observed_fraction` unit `fraction`, `<metric>_observed_source` categorical). Run the
  `feature_glossary` coverage gate → PASS.

- [ ] **Step 2: SB360 audit** — record the companions at `tests/sb360/_registry.py::audited_surface` as
  opt-in (outside the default-config audit, like ADR-062). Run `tests/sb360/` → PASS.

- [ ] **Step 3: Write ADR-077** — the decision, the S1 single-model rationale, the two round-2 items
  (N1 goal-keyed zones, N2 frame-independence), the exemptions.

- [ ] **Step 4: CHANGELOG + CLAUDE.md** — a Key-conventions line summarising the observability
  registry + `validate_fov` + the single-model rule + the completeness gate.

- [ ] **Step 5: C4** — regen only if the glossary count line changed (no new aggregator → C4 diagram
  count unchanged; the glossary count string updates).

- [ ] **Step 6: e2e validation on the real licensed SB360 corpus** (`@pytest.mark.e2e`, not in CI)

Add an `@e2e` test (mirroring the ADR-062 `scripts/validate_sb360_licensed_corpus.py` pattern) that,
on the real 30-match licensed corpus, asserts the companions are **present and sanely distributed** on
`fov_cropped` matches: `*_observed_fraction ∈ [0,1]` where `*_observed_source == "observed"`, NaN
elsewhere, and at least one cropped action yields a fraction `< 1` (proving the signal is live on real
FOV crops, not a synthetic-only artifact). Owner-run; self-skips without the licensed fixtures.

- [ ] **Step 7: Version + numbers at commit-readiness** — ONLY now, confirm the next free version
  (FIVE sites incl. `uv.lock`), `PR-Sxxx`, and `ADR-0xx` against `main`, and stamp them into the
  bumped files + the ADR filename + CHANGELOG. (Nothing above this step hardcodes a number.)

- [ ] **Step 8: Full local gate**

Run: `python -m pytest tests/ -m "not e2e" -q --tb=short`
Run: `python -m ruff check silly_kicks/ tests/ scripts/` ; `python -m ruff format --check silly_kicks/ tests/ scripts/`
Run: `python -m pyright`
Expected: all green. (Verify on `.venv312` for the pandas-3 leg; capture ALL `FAILED`, never `tail`.)

---

## Self-Review

- **Spec coverage:** Component 1 → Task 1; Component 2 → Task 2; Component 3 (7 aggregators) → Tasks
  2–6; Component 4 → Task 7; Component 5 → Task 8; glossary/audit/docs/version → Task 9. S1 (single
  model) is Global Constraints + Task 2; T1 → Task 6; T2 → Task 5; M1 → Task 5; M2/M3 → Task 8; M4 →
  Task 7; N1 → Task 4; N2 → Tasks 2 & 4; N3 → Task 5. All covered.
- **Placeholders:** the `...` markers in test bodies are golden-value / fixture-derivation steps with
  an explicit preceding grounding step (Task 2 Step 2, Task 3/4/5/6/7 Step 1) that says exactly what to
  read and extract — not "figure it out later". Every code step names its function and file:line.
- **Type consistency:** `ObservabilityEntry(column, region, covers=())`, `RegionCtx(sx, sy, ex, ey,
  game_id, period_id, team_id, nearest_dist, goal_map, extras)`, the `_NO_REGION` identity sentinel,
  `companioned_columns()`, and `append_observability_companions(out, actions, *, entries, visible_area,
  linked_ids, ctx)` are used identically across Tasks 2–6. Per-call params (`receiver_radius`, `pressure_method`, oval/link radii)
  flow through `ctx.extras`; every companioned column is a STATIC entry in `OBSERVABILITY_REGISTRY`.
  `_OBSERVABILITY_EXEMPT` is a `dict[str, str]` in Tasks 5 & 8.
- **Plan-review (executed) resolutions folded in:** P1 `validate_fov` `mixed` reachable only at
  `0 < n_full < n` (Task 1); P2 array-safe `_NO_REGION` identity sentinel, tested with `is` (Task 2, +
  cascade in Tasks 4/5/6); P3 `GoalMap.get(game_id, period_id, team_id) -> float | None`, `_NO_REGION`
  on unresolved (Task 4); P4 real permutation-invariance test (Task 4); P5 magnitude-weighted rollup +
  hand-computed assertion (Task 6); P6 team/end grounding + end-correctness test (Task 4); P7 static
  registry entries via `ctx.extras`, no literal `region=None`, Interfaces line says `entries=` (Task 2);
  P8 two role companions collapse x/y within a role (Task 4); P9 `_PACKING_REGION_COUNT_COLUMNS` from
  grounding, not a guessed name (Task 3).
- **Plan-review round-2 (executed) resolutions:** R1 the completeness gate enforces TWO axes —
  `region_support` ∪ hand-curated `_AGGREGATE_FOV_SENSITIVE` — via `companioned_columns()` mapping RAW
  columns through `ObservabilityEntry.covers` (the perturbation probe can't see FOV-crop bias; spec
  Component 5 reconciled) (Task 8); R2 `add_team_shape` emits FOUR role-split columns
  (`team_shape_centroid_{x,y}_{attacking,defending}`) → TWO team-keyed companions (no
  `team_shape_centroid` column), each zone keyed on the right team via `ctx.extras` (Task 4); M-msg the
  `mixed` message reworded to the new semantics (Task 1); M-guess `pitch_zone` threads `allow_guess` to
  match the primary metric (Task 4). Gate contract: `required = region_support ∪ _AGGREGATE_FOV_SENSITIVE`;
  `companioned_columns() ∪ exempt ⊇ required`; `exempt ⊆ required`; companioned MAY exceed required
  (unforced tight-ROI companions).
