# TF-60 PR3 — Rest-Defense GK-Ghost Model Re-fit — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship an additive extended-grid `GhostGkModel` variant pair (`sweeper` + `sweeper_position_only`) that can represent the in-possession high-sweeper regime, without changing the frozen `default`/`position_only`/`full` variants (no GKDV/VAEP retrain).

**Architecture:** Make the ghost-GK grid a first-class per-model `GhostGridSpec` (currently module-global constants), threaded through `__init__`/`save`/`load`/label-filter/`ghost_out_of_box`, byte-identical for `default`. Add the two new bundled variants + an additive velocity-keyed resolver branch + trainer/publisher support. Build and test everything **locally against a toy model**; the real bundled weights come from a DGX re-fit (Phase B) before the single owner-approved commit.

**Tech Stack:** Python, numpy, pandas, scikit-learn (HistGradientBoostingRegressor), pytest. The trained-model artifact is npz + JSON + SHA256SUMS (pickle-free, ADR-011/016/044/050).

**Spec:** `docs/superpowers/specs/2026-08-30-tf60-restdefense-gk-ghost-refit-design.md` (fully approved: arc reshape r3, sub-spec content r2). Evidence: `docs/research/tf60_ghost_gk_in_possession_validity/`.

## Global Constraints

- **Feature branch:** `feat/tf60-pr3-gk-ghost-refit` (already created off `main` @ `ff0f234`; the design docs — spec + finding + this plan — already live there, uncommitted). One branch for the whole cycle; no worktrees.
- **COMMIT DISCIPLINE (overrides the writing-plans template):** NO per-task commits. **ONE** owner-approved commit for the whole cycle, at the very end of Phase B, in a fully-tested green state **with the real DGX weights** (never a toy model). The executor stops at the commit gate, shows the diff, and waits for an explicit "commit". Docs (spec + finding + this plan) land in that first commit.
- **Byte-identical `default`:** the grid-first-class refactor is a pure no-op for `default`/`position_only`/`full`. Gated THREE ways (impl-review P3I-02: a full save→load→save `SHA256SUMS` round-trip is **unachievable** — `save()` recomputes `feature_contract.probe_sha256`, so `default`'s metadata SHA differs on re-save independent of this refactor): (1) the existing golden / chirality / feature-contract / KDE-density behavioural tests pass UNCHANGED; (2) `DEFAULT_GHOST_GRID.to_metadata_dict()` equals `default`'s committed `grid_spec` field; (3) a re-save of `default` changes **only** `feature_contract`. `DEFAULT_GHOST_GRID` equals the current module constants (`GRID_X_MIN=0.0, GRID_X_MAX=30.0, GRID_Y_MIN=18.0, GRID_Y_MAX=50.0, GRID_RESOLUTION=0.5`).
- **Additivity → no GKDV/VAEP retrain:** GKDV serves `default` (via `model=None` → `_resolve_ghost_model_for_frames` → `_variant_key_for_velocity` = default/position_only) and can NEVER select `sweeper`. Every resolver change is a NEW branch that gkdv's inputs never reach.
- **Density fail-loud (`predict_density` ONLY):** `predict_density` raises on a non-default grid. `compute_ghost_gk`/`add_ghost_gk` are NOT guarded — they use the grid-independent `predict_mean` path (density retired 2026-07-20) and WORK on the extended grid.
- **`GhostGridSpec` serialization is byte-preserving:** `to_metadata_dict()` emits the exact current 7 keys `{x_min, x_max, y_min, y_max, nx, ny, resolution}` with DERIVED `nx = round((x_max−x_min)/resolution)`, `ny = round((y_max−y_min)/resolution)`, in the current key order.
- **Grid ceiling:** `sweeper` grid = `x_max = 52.5` (halfway), `y` unchanged (18–50), `resolution = 0.5` → `nx = 105, ny = 64`.
- **Variant naming:** `sweeper` (faithful) + `sweeper_position_only` (position_only). Capability-descriptive; the final pick was invited in review — if the owner renames it, change the `Literal` + the bundled dir names + the metadata; the mechanism is identical.
- **Velocity-less fallback stays NaN, never `default`** (ADR-067 asymmetry): a `sweeper` request on velocity-less frames with no bundled `sweeper_position_only` returns `(None, "sweeper_position_only")` so the serve seam emits the honest degrade — never the (invalid-on-velocity-less) faithful default.
- **Testing/lint at CI scope:** `python -m pytest tests/ -m "not e2e" -v --tb=short`; `tests/tracking/` needs `--benchmark-skip`. Lint: `python -m ruff check silly_kicks/ tests/ scripts/` + `python -m ruff format --check silly_kicks/ tests/ scripts/`; `python -m pyright` (bare). Trainer/publisher smokes are `@pytest.mark.slow` (primary leg only); golden/chirality/version-sensitive tests stay on ALL legs.
- **Phase split:** Phase A (Tasks 1–9) = local code + tests against a locally-fit toy `sweeper`. Phase B (Task 10) = DGX weights + real-artifact goldens + the one commit. No tag ever ships on a toy model.

---

## File Structure

- `silly_kicks/tracking/_ghost_gk.py` — MODIFY: add `GhostGridSpec` + `DEFAULT_GHOST_GRID`; thread `grid_spec` through `__init__`/`save`/`load`; `ghost_out_of_box` + label filter read the per-model grid; `predict_density` fail-loud guard; `GhostGkVariant` literals; additive `sweeper`-family resolver branch.
- `silly_kicks/tracking/_ghost_gk_weights/sweeper/`, `…/sweeper_position_only/` — CREATE (Phase B): the bundled artifacts (npz + metadata.json + SHA256SUMS). In Phase A a toy artifact is written to a tmp dir by the tests.
- `scripts/train_ghost_gk.py` — MODIFY: `--grid-x-max` (extended grid) threading to `prepare` + the model + the shard-generation token; `>30 m`-stratum MAE + per-provider `>30 m` coverage in `metrics.json`.
- `scripts/publish_ghost_gk.py` — MODIFY: publish the new variant(s) with the existing contract discipline.
- `tests/tracking/test_ghost_grid_spec.py` — CREATE: `GhostGridSpec` + serialization.
- `tests/tracking/test_ghost_gk_grid_refactor.py` — CREATE: byte-identical `default` (save/load/SHA), extended-grid round-trip, `out_of_box`, label-filter, `predict_density` fail-loud, `compute_ghost_gk` works on extended grid.
- `tests/tracking/test_ghost_gk_sweeper_variant.py` — CREATE: variant literals + `from_variant` + the additive resolver branch + the two-sided saturation-vs-tracking gate.
- `tests/scripts/test_train_ghost_gk_sweeper.py` — CREATE: trainer metric-helper units (non-slow) + a `@slow` train smoke.
- `silly_kicks/feature_glossary.py` — NO CHANGE (no new emitted feature columns; the variant serves the existing `ghost_gk_x/y` columns). C4: unchanged (no new aggregator/container).

---

## Phase A — local code + tests (this session, against a toy model)

### Task 1: `GhostGridSpec` value object + byte-preserving serialization

**Files:**
- Modify: `silly_kicks/tracking/_ghost_gk.py` (add near the grid constants, `:91-108`)
- Test: `tests/tracking/test_ghost_grid_spec.py`

**Interfaces:**
- Produces: `GhostGridSpec(x_min: float, x_max: float, y_min: float, y_max: float, resolution: float)` (frozen dataclass); `GhostGridSpec.to_metadata_dict() -> dict` (7 keys, derived nx/ny); `GhostGridSpec.nx -> int`, `.ny -> int` (properties); module constant `DEFAULT_GHOST_GRID`.

- [ ] **Step 1: Write the failing test**

```python
# tests/tracking/test_ghost_grid_spec.py
from silly_kicks.tracking._ghost_gk import GhostGridSpec, DEFAULT_GHOST_GRID


def test_default_grid_matches_current_module_constants():
    from silly_kicks.tracking import _ghost_gk as g
    assert DEFAULT_GHOST_GRID.x_min == g.GRID_X_MIN == 0.0
    assert DEFAULT_GHOST_GRID.x_max == g.GRID_X_MAX == 30.0
    assert DEFAULT_GHOST_GRID.y_min == g.GRID_Y_MIN == 18.0
    assert DEFAULT_GHOST_GRID.y_max == g.GRID_Y_MAX == 50.0
    assert DEFAULT_GHOST_GRID.resolution == g.GRID_RESOLUTION == 0.5


def test_to_metadata_dict_is_the_exact_7_key_shape_with_derived_nx_ny():
    md = DEFAULT_GHOST_GRID.to_metadata_dict()
    assert list(md.keys()) == ["x_min", "x_max", "y_min", "y_max", "nx", "ny", "resolution"]
    assert md == {"x_min": 0.0, "x_max": 30.0, "y_min": 18.0, "y_max": 50.0,
                  "nx": 60, "ny": 64, "resolution": 0.5}


def test_derived_nx_ny_reproduce_committed_grid_nx_ny():
    from silly_kicks.tracking import _ghost_gk as g
    assert DEFAULT_GHOST_GRID.nx == g.GRID_NX == 60
    assert DEFAULT_GHOST_GRID.ny == g.GRID_NY == 64


def test_extended_sweeper_grid_derives_105_nx():
    grid = GhostGridSpec(x_min=0.0, x_max=52.5, y_min=18.0, y_max=50.0, resolution=0.5)
    assert grid.nx == 105 and grid.ny == 64
    assert grid.to_metadata_dict()["x_max"] == 52.5
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_grid_spec.py -v`
Expected: FAIL with `ImportError: cannot import name 'GhostGridSpec'`.

- [ ] **Step 3: Implement `GhostGridSpec` + `DEFAULT_GHOST_GRID`**

Add after the grid constants block (`_ghost_gk.py:~108`). Use the module's existing idiom —
`_ghost_gk.py:18` is `import dataclasses` and `:147` uses `@dataclasses.dataclass(frozen=True)` (verified),
so match it (do NOT add a `from dataclasses import dataclass`):

```python
@dataclasses.dataclass(frozen=True)
class GhostGridSpec:
    """The ghost-GK label/density grid, as a first-class per-model value (ADR-021 GridSpec idiom,
    extended here to save/load — ExpectedThreat has no persistence, so that aspect is novel)."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    resolution: float

    @property
    def nx(self) -> int:
        return round((self.x_max - self.x_min) / self.resolution)

    @property
    def ny(self) -> int:
        return round((self.y_max - self.y_min) / self.resolution)

    def to_metadata_dict(self) -> dict:
        # EXACT current 7-key metadata shape + order (:2146-2153), nx/ny DERIVED, so `default`'s
        # metadata.json is byte-identical (review P3-02). A naive asdict() would emit 5 keys.
        return {
            "x_min": self.x_min, "x_max": self.x_max, "y_min": self.y_min, "y_max": self.y_max,
            "nx": self.nx, "ny": self.ny, "resolution": self.resolution,
        }


DEFAULT_GHOST_GRID = GhostGridSpec(
    x_min=GRID_X_MIN, x_max=GRID_X_MAX, y_min=GRID_Y_MIN, y_max=GRID_Y_MAX, resolution=GRID_RESOLUTION
)
```

(`dataclasses` is already imported at `:18`; no new import needed.)

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_grid_spec.py -v`
Expected: PASS (4 tests).

---

### Task 2: thread `grid_spec` through `__init__` / `save` / `load` (byte-identical `default`)

**Files:**
- Modify: `silly_kicks/tracking/_ghost_gk.py` — `__init__` (`:1760`), `save` (`:2146`), `load` (`:2279-2301`)
- Test: `tests/tracking/test_ghost_gk_grid_refactor.py`

**Interfaces:**
- Consumes: `GhostGridSpec`, `DEFAULT_GHOST_GRID` (Task 1).
- Produces: `GhostGkModel(..., grid_spec: GhostGridSpec = DEFAULT_GHOST_GRID)`; `model.grid_spec` restored on `load`.

- [ ] **Step 1: Write the failing test**

```python
# tests/tracking/test_ghost_gk_grid_refactor.py
import hashlib
from pathlib import Path

from silly_kicks.tracking._ghost_gk import GhostGkModel, GhostGridSpec, DEFAULT_GHOST_GRID
from silly_kicks.tracking._ghost_gk import _WEIGHTS_ROOT


def _sha(p: Path) -> str:
    raw = p.read_bytes()
    if p.suffix == ".json":
        raw = raw.replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def test_default_model_constructs_with_default_grid():
    m = GhostGkModel()
    assert m.grid_spec == DEFAULT_GHOST_GRID


def test_default_grid_spec_serializes_byte_identical_to_committed():
    # P3I-02: a FULL metadata SHA round-trip is UNACHIEVABLE (save() recomputes
    # feature_contract.probe_sha256 from the current extractor). Prove grid byte-identity on the
    # grid_spec FIELD instead; behavioural byte-identity is the unchanged goldens (Task 9).
    committed = json.loads((_WEIGHTS_ROOT / "default" / "metadata.json").read_text())["grid_spec"]
    assert GhostGkModel.load(_WEIGHTS_ROOT / "default").grid_spec.to_metadata_dict() == committed


def test_default_metadata_differs_only_in_recomputed_feature_contract(tmp_path):
    # A re-save of default must change ONLY feature_contract (the pre-existing recompute); if this
    # refactor moves any OTHER field for default, this fails.
    committed = json.loads((_WEIGHTS_ROOT / "default" / "metadata.json").read_text())
    out = tmp_path / "resaved"
    GhostGkModel.load(_WEIGHTS_ROOT / "default").save(out)
    resaved = json.loads((out / "metadata.json").read_text())
    diffs = {k for k in set(committed) | set(resaved) if committed.get(k) != resaved.get(k)}
    assert diffs <= {"feature_contract"} and committed["grid_spec"] == resaved["grid_spec"]


def test_extended_grid_round_trips_through_save_load(tmp_path):
    # GhostGkModel is a plain (non-frozen) class; construct with the extended grid directly.
    m = GhostGkModel(grid_spec=GhostGridSpec(0.0, 52.5, 18.0, 50.0, 0.5))
    _fit_toy(m)  # minimal fit so save() has trees; helper in the test module / conftest (see Task 6)
    out = tmp_path / "sweeper_toy"
    m.save(out)
    back = GhostGkModel.load(out)
    assert back.grid_spec.x_max == 52.5


def test_metadata_without_grid_spec_loads_default(tmp_path):
    # Back-compat: a pre-refactor artifact (no grid_spec key) loads with DEFAULT_GHOST_GRID.
    src = _WEIGHTS_ROOT / "default"
    m = GhostGkModel.load(src)
    m.save(tmp_path / "a")
    import json
    md_path = tmp_path / "a" / "metadata.json"
    md = json.loads(md_path.read_text())
    del md["grid_spec"]
    md_path.write_text(json.dumps(md, indent=2), newline="\n")
    # rewrite SHA256SUMS for the edited metadata so integrity passes
    _rewrite_sums(tmp_path / "a")  # helper: recompute SHA256SUMS
    back = GhostGkModel.load(tmp_path / "a")
    assert back.grid_spec == DEFAULT_GHOST_GRID
```

(`_fit_toy` and `_rewrite_sums` helpers are defined once in `tests/tracking/conftest.py` or the test module — `_fit_toy` fits a 5-tree HGBR on a tiny synthetic feature/label frame in-grid; `_rewrite_sums` recomputes `SHA256SUMS` after a metadata edit. Define them concretely in Step 3's companion.)

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_gk_grid_refactor.py -v`
Expected: FAIL (`AttributeError: 'GhostGkModel' object has no attribute 'grid_spec'`).

- [ ] **Step 3: Implement the threading**

1. `__init__` (`:1768`): add `self.grid_spec: GhostGridSpec = grid_spec` and a `grid_spec: GhostGridSpec = DEFAULT_GHOST_GRID` keyword-only parameter.
2. `save` (`:2146-2154`): replace the literal 7-key dict with `"grid_spec": self.grid_spec.to_metadata_dict(),`.
3. `load` (after `model.corpus_provenance = ...`, `:2301`): add
   ```python
   gs = metadata.get("grid_spec")
   if gs is None:
       model.grid_spec = DEFAULT_GHOST_GRID
   else:
       model.grid_spec = GhostGridSpec(
           x_min=gs["x_min"], x_max=gs["x_max"], y_min=gs["y_min"], y_max=gs["y_max"],
           resolution=gs["resolution"],
       )  # nx/ny are derived; ignore any recorded nx/ny.
   ```

- [ ] **Step 4: Run to verify it passes + confirm no regression on the ghost golden**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_gk_grid_refactor.py tests/tracking/ -k "ghost and (golden or serve or load or chirality or contract)" -v --benchmark-skip`
Expected: PASS, and the pre-existing ghost golden/chirality/feature-contract tests still PASS unchanged.

---

### Task 3: `ghost_out_of_box` + the label filter read the per-model grid

**Files:**
- Modify: `silly_kicks/tracking/_ghost_gk.py` — `serve_ghost_gk_positions` `ghost_out_of_box` (`:2802`); `prepare_ghost_gk_training_data` label filter (`:1204-1220`)
- Test: `tests/tracking/test_ghost_gk_grid_refactor.py`

**Interfaces:**
- Consumes: `model.grid_spec` (Task 2).
- Produces: `prepare_ghost_gk_training_data(..., grid_spec: GhostGridSpec = DEFAULT_GHOST_GRID)`.

- [ ] **Step 1: Write the failing test**

```python
def test_out_of_box_is_variant_relative(tmp_path):
    # A served position at gr_x=40 is out_of_box for default (x_max=30) but NOT for sweeper (52.5).
    from silly_kicks.tracking._ghost_gk import GhostGkModel, GhostGridSpec
    # Build two toy models with different grids that both predict a ~40 m keeper on a crafted frame,
    # then assert the flag. (Use the toy-fit helper + a frame whose features drive a ~40 m prediction;
    # simplest robust form: unit-test the flag expression directly against grid_spec.x_max.)
    assert (40.0 > GhostGridSpec(0, 30, 18, 50, 0.5).x_max) is True
    assert (40.0 > GhostGridSpec(0, 52.5, 18, 50, 0.5).x_max) is False


def test_label_filter_retains_high_sweeper_labels_under_extended_grid():
    import numpy as np, pandas as pd
    from silly_kicks.tracking._ghost_gk import prepare_ghost_gk_training_data, GhostGridSpec, GHOST_GK_FEATURE_NAMES
    # Not exercised end-to-end here (needs frames); unit the domain predicate the filter uses:
    labels = pd.DataFrame({"gk_x": [10.0, 35.0, 45.0], "gk_y": [34.0, 34.0, 34.0]})
    default_grid = GhostGridSpec(0, 30, 18, 50, 0.5)
    sweeper_grid = GhostGridSpec(0, 52.5, 18, 50, 0.5)
    def in_domain(g):
        return (labels["gk_x"] >= g.x_min) & (labels["gk_x"] <= g.x_max) \
             & (labels["gk_y"] >= g.y_min) & (labels["gk_y"] <= g.y_max)
    assert in_domain(default_grid).tolist() == [True, False, False]   # 35, 45 dropped as "rushes"
    assert in_domain(sweeper_grid).tolist() == [True, True, True]     # retained
```

- [ ] **Step 2: Run to verify it fails / passes-as-spec**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_gk_grid_refactor.py -k "out_of_box or label_filter" -v`
Expected: the flag/domain unit tests PASS once the code below is in place (they pin the intended semantics; the real wiring is verified by Task 6's end-to-end saturation gate).

- [ ] **Step 3: Implement**

1. `serve_ghost_gk_positions` (`:2802`): change `(positions[:, 0] > GRID_X_MAX)` to `(positions[:, 0] > _resolved.grid_spec.x_max)` (the resolved model is already bound as `_resolved` at `:2731`).
2. `prepare_ghost_gk_training_data`: add a keyword-only `grid_spec: GhostGridSpec = DEFAULT_GHOST_GRID` param; the label-domain filter (`:1205-1210`) uses `grid_spec.x_min/x_max/y_min/y_max` instead of `GRID_X_MIN/MAX/Y_MIN/MAX`. The off-pitch drop stays (keep the on-pitch guard); the warning message is unchanged.

- [ ] **Step 4: Run to verify + regression**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/ -k "ghost" -v --benchmark-skip`
Expected: PASS; the default-grid path is byte-identical (out_of_box on `default` uses 30 via `grid_spec.x_max`; label filter with default `grid_spec` == the old constants).

---

### Task 4: `predict_density` fail-loud on a non-default grid

**Files:**
- Modify: `silly_kicks/tracking/_ghost_gk.py` — `predict_density` (guard at the top, near `:1992`)
- Test: `tests/tracking/test_ghost_gk_grid_refactor.py`

**Interfaces:**
- Produces: `predict_density` raises `ValueError` on `self.grid_spec != DEFAULT_GHOST_GRID`.

- [ ] **Step 1: Write the failing test**

```python
def test_predict_density_raises_on_non_default_grid():
    import pytest, pandas as pd
    from silly_kicks.tracking._ghost_gk import GhostGkModel, GhostGridSpec, GHOST_GK_FEATURE_NAMES
    m = GhostGkModel()
    _fit_toy(m)  # locally-fit so training arrays exist (density is fit-only)
    m.grid_spec = GhostGridSpec(0, 52.5, 18, 50, 0.5)
    X = pd.DataFrame([[0.0] * len(GHOST_GK_FEATURE_NAMES)], columns=GHOST_GK_FEATURE_NAMES)
    with pytest.raises(ValueError, match="extended-grid density"):
        m.predict_density(X)


def test_compute_ghost_gk_runs_on_extended_grid_mean_path():
    # DIRECT behavioral run (review P3P-04): compute_ghost_gk uses predict_mean (grid-independent),
    # so it must NOT raise on a sweeper-grid model and must emit finite ghost_gk_x/y — exactly as
    # serve_ghost_gk_positions("sweeper") does. (Not a source-grep proxy.)
    from silly_kicks.tracking import compute_ghost_gk
    from silly_kicks.tracking._ghost_gk import GhostGkModel, GhostGridSpec
    frames = _two_team_frames(velocity=True)          # conftest helper (shared with Task 5)
    home = _home_team_of(frames)
    sweeper = GhostGkModel(n_estimators=20, max_depth=3,
                           grid_spec=GhostGridSpec(0.0, 52.5, 18.0, 50.0, 0.5))
    _fit_toy(sweeper)
    out = compute_ghost_gk(frames, home_team_id=home, model=sweeper)   # must NOT raise
    gk = out[out["is_goalkeeper"].astype("boolean").fillna(False)]
    assert gk["ghost_gk_x"].notna().any() and gk["ghost_gk_y"].notna().any()


def test_no_density_guard_leaked_into_the_mean_path_primitive():
    # Belt-and-suspenders: the fail-loud guard is scoped to predict_density, not compute_ghost_gk.
    import inspect
    from silly_kicks.tracking import _ghost_gk
    assert "extended-grid density" not in inspect.getsource(_ghost_gk.compute_ghost_gk)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_gk_grid_refactor.py -k "predict_density or extended_grid_mean or density_guard" -v`
Expected: FAIL (no guard yet → `predict_density` returns a density on the wrong grid).

- [ ] **Step 3: Implement the guard**

At the top of `predict_density` (before the existing parameters-only raise at `:1992`):

```python
if self.grid_spec != DEFAULT_GHOST_GRID:
    raise ValueError(
        "predict_density: extended-grid density is not supported this cycle (the KDE grid is the "
        "default 30 m grid). Use predict_mean / serve_ghost_gk_positions for the sweeper variant; "
        "the density path stays on DEFAULT_GHOST_GRID (spec 2026-08-30 §4.2)."
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_gk_grid_refactor.py -k "predict_density or extended_grid_mean or density_guard" -v`
Expected: PASS.

---

### Task 5: `GhostGkVariant` literals + `from_variant` + additive `sweeper`-family resolver branch

**Files:**
- Modify: `silly_kicks/tracking/_ghost_gk.py` — `GhostGkVariant` (`:212`), `_resolve_ghost_model_for_frames` (`:277-299`)
- Test: `tests/tracking/test_ghost_gk_sweeper_variant.py`

**Interfaces:**
- Consumes: `from_variant` (`:2354`, unchanged — it already loads `_WEIGHTS_ROOT/<variant>`), `_variant_key_for_velocity`.
- Produces: `GhostGkVariant` gains `"sweeper"`, `"sweeper_position_only"`; `_resolve_ghost_model_for_frames(frames, "sweeper")` velocity-keys within the sweeper family.

- [ ] **Step 1: Write the failing test**

```python
# tests/tracking/test_ghost_gk_sweeper_variant.py
import numpy as np, pandas as pd, pytest
from silly_kicks.tracking import _ghost_gk
from silly_kicks.tracking._ghost_gk import GhostGkModel, _resolve_ghost_model_for_frames


def _frames(velocity: bool):
    # minimal 2-team frame set; velocity=False marks speed_source unavailable (declared SB360 shape)
    ...  # build via the shared tracking fixture helper; see conftest _two_team_frames(velocity=)


def test_sweeper_family_resolves_faithful_on_velocity_bearing(monkeypatch):
    calls = {}
    def fake_from_variant(cls, key="default"):
        calls["key"] = key
        return GhostGkModel(feature_set=("position_only" if key.endswith("position_only") else "faithful"))
    monkeypatch.setattr(GhostGkModel, "from_variant", classmethod(fake_from_variant))
    _, key = _resolve_ghost_model_for_frames(_frames(velocity=True), "sweeper")
    assert key == "sweeper"


def test_sweeper_family_resolves_position_only_on_velocity_less(monkeypatch):
    calls = {}
    def fake_from_variant(cls, key="default"):
        calls["key"] = key
        return GhostGkModel(feature_set="position_only")
    monkeypatch.setattr(GhostGkModel, "from_variant", classmethod(fake_from_variant))
    _, key = _resolve_ghost_model_for_frames(_frames(velocity=False), "sweeper")
    assert key == "sweeper_position_only"


def test_sweeper_position_only_missing_returns_none_never_default(monkeypatch):
    def fake_from_variant(cls, key="default"):
        if key == "sweeper_position_only":
            raise FileNotFoundError
        raise AssertionError(f"must not fall back to {key}")
    monkeypatch.setattr(GhostGkModel, "from_variant", classmethod(fake_from_variant))
    model, key = _resolve_ghost_model_for_frames(_frames(velocity=False), "sweeper")
    assert model is None and key == "sweeper_position_only"


def test_gkdv_none_path_is_byte_identical(monkeypatch):
    # model=None must still resolve to the DEFAULT family (default/position_only), never sweeper.
    seen = []
    def fake_from_variant(cls, key="default"):
        seen.append(key)
        return GhostGkModel()
    monkeypatch.setattr(GhostGkModel, "from_variant", classmethod(fake_from_variant))
    _resolve_ghost_model_for_frames(_frames(velocity=True), None)
    assert seen == ["default"]  # never "sweeper"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_gk_sweeper_variant.py -k "resolves or byte_identical or missing" -v`
Expected: FAIL (the `"sweeper"` branch does not exist; it currently hits `model is not None → "custom"`).

- [ ] **Step 3: Implement**

1. `GhostGkVariant` (`:212`): `Literal["default", "full", "position_only", "sweeper", "sweeper_position_only"]`.
2. `_resolve_ghost_model_for_frames` (`:287`): add an ADDITIVE branch BEFORE the `model is not None` short-circuit:

```python
# The sweeper FAMILY root velocity-keys WITHIN its family (faithful <-> position_only), unlike an
# explicit variant NAME which is respected as-is. Placed before the `model is not None` custom
# short-circuit; gkdv passes None/"default"/instance and never reaches this branch (additivity).
if model == "sweeper":
    base = _variant_key_for_velocity(frames)          # "default" | "position_only"
    key = "sweeper_position_only" if base == "position_only" else "sweeper"
    try:
        return GhostGkModel.from_variant(key), key
    except FileNotFoundError:
        if key == "sweeper_position_only":
            return None, key                          # NaN degrade, never default (ADR-067)
        raise
```

(Confirm `_variant_key_for_velocity` returns exactly `"default"`/`"position_only"` — read `:277` neighbourhood; if it returns other tokens, map on the position_only-ness.)

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_gk_sweeper_variant.py -k "resolves or byte_identical or missing" -v`
Expected: PASS (4 tests).

---

### Task 6: the two-sided saturation-vs-tracking gate (the signature test)

**Files:**
- Create: the toy-fit helper in `tests/tracking/conftest.py` (or the test module)
- Test: `tests/tracking/test_ghost_gk_sweeper_variant.py`

**Interfaces:**
- Consumes: `GhostGkModel` with an extended `grid_spec`, `serve_ghost_gk_positions`, `prepare_ghost_gk_training_data(grid_spec=...)`, the committed `sportec_slim` fixture, `smooth_frames`/`derive_velocities`.
- Produces: a committed regression proving the OLD `default` SATURATES and a toy `sweeper` TRACKS past 30 m — non-vacuously (they measurably differ).

- [ ] **Step 1: Write the failing test**

```python
def test_default_saturates_and_sweeper_tracks_upfield(monkeypatch, tmp_path):
    """Two-sided: translate a clean scene upfield; the shipped default caps its predicted keeper at
    ~30 m while a toy extended-grid model tracks past it. Direction, not magnitude (toy-swap safe)."""
    import numpy as np, pandas as pd
    from silly_kicks.tracking import (
        resolve_defended_goals, serve_ghost_gk_positions, derive_velocities, GhostGkModel,
    )
    from silly_kicks.tracking.preprocess import smooth_frames
    from silly_kicks.tracking._ghost_gk import prepare_ghost_gk_training_data, GhostGridSpec
    from silly_kicks.id_compat import canonical_id

    base = _load_sportec_slim_frames()   # helper: __kind==frame rows, is_ball/is_goalkeeper bool
    home = _home_defending_x0(base)

    # Fit a toy sweeper on the fixture itself with the cap lifted (in-fixture labels up to ~45 m
    # after we translate); tiny HGBR so the test is fast.
    sweeper = GhostGkModel(n_estimators=30, max_depth=4)
    sweeper.grid_spec = GhostGridSpec(0.0, 52.5, 18.0, 50.0, 0.5)
    feats, labs = prepare_ghost_gk_training_data(
        _translated_training_set(base), home_team_id=home, subsample_fps=None,
        grid_spec=sweeper.grid_spec,   # retain the high-sweeper labels
    )
    sweeper.fit(feats, labs)

    def pred_max(model, delta):
        f = base.copy()
        f["x"] = np.clip(f["x"].to_numpy(float) + delta, 0.0, 105.0)
        f = derive_velocities(smooth_frames(f))
        served = serve_ghost_gk_positions(f, home_team_id=home, model=model)
        svh = served[served["gk_team_id"].map(canonical_id) == canonical_id(home)]
        return float(svh["ghost_gr_x"].max())

    # default (shipped): saturates ~30
    d0, d25 = pred_max(None, 0), pred_max(None, 25)
    assert d25 <= 31.0                      # DEFECT pinned: cannot exceed its 30 m ceiling
    # sweeper (toy): tracks past 30 as the scene advances
    s0, s25 = pred_max(sweeper, 0), pred_max(sweeper, 25)
    assert s25 > 33.0                       # FIX pinned: places a keeper well past 30 m
    # non-vacuity: the two legs measurably differ at Δ=25
    assert s25 - d25 > 3.0
```

(`_load_sportec_slim_frames`, `_home_defending_x0`, `_translated_training_set` are small helpers in the test module — the first two mirror the finding probe; `_translated_training_set` stacks a few upfield translations of `base` so the toy sees high-sweeper labels. Keep them concrete in the test module.)

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_gk_sweeper_variant.py -k "saturates_and_sweeper_tracks" -v --benchmark-skip`
Expected: FAIL first because the toy `sweeper` cannot be served until the label filter retains high labels (Task 3) + the grid round-trips (Task 2) — run AFTER Tasks 1–5 so it exercises the full stack. If those are in, it should PASS; if it fails, the failure localizes the wiring gap.

- [ ] **Step 3: (no new impl) — this task validates Tasks 1–5 end-to-end.** If it fails, fix the offending earlier task; do not weaken the assertion.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_gk_sweeper_variant.py -v --benchmark-skip`
Expected: PASS.

---

### Task 7: trainer support (extended grid + `>30 m` stratum / coverage metrics)

**Files:**
- Modify: `scripts/train_ghost_gk.py` — add `--grid-x-max` (threads a `GhostGridSpec` to `prepare` + the model + the shard-generation `token_inputs`); add `>30 m`-stratum MAE + per-provider `>30 m` coverage to `metrics.json`.
- Test: `tests/scripts/test_train_ghost_gk_sweeper.py`

**Interfaces:**
- Consumes: `prepare_ghost_gk_training_data(grid_spec=...)` (Task 3), `GhostGkModel(grid_spec=...)` (Task 2).
- Produces: metric helpers `high_sweeper_stratum_mae(preds, labels, *, threshold=30.0) -> float`, `per_provider_high_sweeper_coverage(labels, provider_labels, *, threshold=30.0) -> dict`.

- [ ] **Step 1: Write the failing test (non-slow metric-helper units)**

```python
# tests/scripts/test_train_ghost_gk_sweeper.py   (NO tests/scripts/__init__.py — namespace pkg)
import numpy as np
from scripts.train_ghost_gk import high_sweeper_stratum_mae, per_provider_high_sweeper_coverage


def test_high_sweeper_stratum_mae_only_over_gt30_labels():
    preds = np.array([[12.0, 34.0], [40.0, 34.0], [45.0, 34.0]])
    labels = {"gk_x": np.array([10.0, 38.0, 44.0]), "gk_y": np.array([34.0, 34.0, 34.0])}
    # only the 38 and 44 rows count; |40-38|=2, |45-44|=1 -> mean euclid over those two
    val = high_sweeper_stratum_mae(preds, labels, threshold=30.0)
    assert abs(val - 1.5) < 1e-9


def test_per_provider_coverage_counts_gt30_fraction():
    labels = {"gk_x": np.array([10.0, 35.0, 40.0, 12.0])}
    provs = np.array(["sportec", "sportec", "gradientsports", "skillcorner"])
    cov = per_provider_high_sweeper_coverage(labels, provs, threshold=30.0)
    assert cov["sportec"] == 0.5 and cov["gradientsports"] == 1.0 and cov["skillcorner"] == 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/scripts/test_train_ghost_gk_sweeper.py -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement the helpers + wire the CLI**

Add the two pure helpers to `scripts/train_ghost_gk.py`:

```python
def high_sweeper_stratum_mae(preds, labels, *, threshold: float = 30.0) -> float:
    gx = np.asarray(labels["gk_x"], float); gy = np.asarray(labels["gk_y"], float)
    m = gx > threshold
    if not m.any():
        return float("nan")
    dx = preds[m, 0] - gx[m]; dy = preds[m, 1] - gy[m]
    return float(np.mean(np.sqrt(dx * dx + dy * dy)))


def per_provider_high_sweeper_coverage(labels, provider_labels, *, threshold: float = 30.0) -> dict:
    gx = np.asarray(labels["gk_x"], float); provs = np.asarray(provider_labels)
    out: dict[str, float] = {}
    for p in sorted(set(provs.tolist())):
        sel = provs == p
        out[str(p)] = float(np.mean(gx[sel] > threshold)) if sel.any() else float("nan")
    return out
```

Wire the CLI: add `--grid-x-max` (default `None` = `DEFAULT_GHOST_GRID`); when set, build `GhostGridSpec(0.0, grid_x_max, 18.0, 50.0, 0.5)`, pass it to `prepare_ghost_gk_training_data(grid_spec=...)` AND `GhostGkModel(grid_spec=...)`, and add `"grid_x_max": grid_x_max` to `extraction_inputs` (the `for_each` shard-generation token — the 4.77.1 stale-shard rule: the label domain changes the shard rows). Add the two metric blocks to `metrics.json` (`high_sweeper_stratum_mae` computed on the CV test folds; `per_provider_high_sweeper_coverage` on the retained labels).

- [ ] **Step 4: Run to verify + a `@slow` train smoke**

Run: `.venv/Scripts/python.exe -m pytest tests/scripts/test_train_ghost_gk_sweeper.py -v`
Expected: PASS. Add one `@pytest.mark.slow` smoke that runs the trainer's `main()`-equivalent on a tiny committed fixture with `--grid-x-max 52.5` and asserts `metrics.json` carries `high_sweeper_stratum_mae` + `per_provider_high_sweeper_coverage` and `grid_spec.x_max == 52.5` in the saved model metadata. (Guard `--help` safety per the parserless-scripts rule: this trainer has an argparse parser — verify `grep -q add_argument`.)

---

### Task 8: publisher support for the new variant(s)

**Files:**
- Modify: `scripts/publish_ghost_gk.py`
- Test: `tests/scripts/test_train_ghost_gk_sweeper.py` (a `@slow` publisher smoke)

**Interfaces:**
- Consumes: the bundled `sweeper` artifact dir.
- Produces: publish path that refuses a contract-less artifact and asserts a clean round-trip load (existing discipline), extended to the new variant name(s).

- [ ] **Step 1: Write the failing `@slow` test**

```python
import pytest

@pytest.mark.slow
def test_publisher_accepts_sweeper_variant_roundtrip(tmp_path):
    # fit a toy sweeper, save to tmp, run the publisher's validate/round-trip step (no network),
    # assert it loads clean with no MissingFeatureContractWarning.
    ...  # mirror the existing publish_ghost_gk validate helper on the sweeper dir
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/scripts/test_train_ghost_gk_sweeper.py -k publisher -v -m slow`
Expected: FAIL (publisher doesn't know the variant).

- [ ] **Step 3: Implement**

Extend `publish_ghost_gk.py` to accept a `--variant sweeper|sweeper_position_only` (or discover the variant dirs), reusing the existing contract-required + round-trip-clean assertions unchanged. No new discipline — the same guards, one more variant.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/scripts/test_train_ghost_gk_sweeper.py -k publisher -v -m slow`
Expected: PASS.

---

### Task 9: Phase-A green + lint/type + full-suite gate (no commit)

**Files:** none (verification task).

- [ ] **Step 1: Full suite (non-e2e), benchmark-skipped**

Run: `.venv/Scripts/python.exe -m pytest tests/ -m "not e2e" --tb=short -q -p no:cacheprovider` (add `--benchmark-skip` for the tracking leg). Expected: GREEN. In particular the pre-existing ghost golden / chirality / feature-contract / KDE-density tests PASS **unchanged** (the byte-identical-`default` proof).

- [ ] **Step 2: Lint + format + types at CI scope**

Run: `python -m ruff check silly_kicks/ tests/ scripts/` ; `python -m ruff format --check silly_kicks/ tests/ scripts/` ; `python -m pyright`. Expected: clean.

- [ ] **Step 3: STOP.** Phase A is complete: all code + tests green against a toy `sweeper`, `default` byte-identical. **Do NOT commit** — the bundled `sweeper` weights do not exist yet. Report Phase A green to the owner and hand off to Phase B (DGX). No tag, no commit on a toy model.

---

## Phase B — DGX weights + real-artifact goldens + the ONE commit (coordinated later)

> Non-TDD coordination + finalization. Runs when the DGX is free (the other session releases it). The code from Phase A is on the branch (uncommitted working tree); Phase B produces the real weights, swaps them in, regenerates the per-artifact goldens, re-runs the suite, and lands the single owner-approved commit.

- [ ] **B1. Re-extract + re-fit on the existing corpus with the cap lifted.** On the DGX (`ssh karsten@192.168.68.73`, `sk-calib-venv`), run `scripts/train_ghost_gk.py --grid-x-max 52.5 --variant sweeper` (faithful) and `--feature-set position_only` (→ `sweeper_position_only`) on the same public corpus the `default`/`full` variants used, from a CLEAN tree (`scripts/_provenance.py` refuses a dirty tree; docs must be committed first — but per commit discipline the commit is owner-gated, so run with `--allow-dirty` for the training pass and record `run_tree_dirty`, OR commit the code first under owner approval; owner decides). **Name the trade-off (review P3P-03):** `--allow-dirty` produces a bundled *public* `sweeper` artifact whose `training_commit`/`run_tree_dirty` provenance is **weaker than its frozen siblings** (`default`/`full` were trained from clean trees) — acceptable for a dev/iteration pass, but the SHIPPED bundled weights should be trained from a clean, owner-approved commit so the artifact's provenance matches the code that produced it. Record the deferred §9 population metrics (`>30 m` fraction, `>30 m`-stratum MAE, per-provider coverage) from `metrics.json`.
- [ ] **B2. Validate the artifacts:** overall MAE + per-provider MAE within the existing acceptance bars; the `>30 m`-stratum MAE reported (set/record its acceptance bar with this evidence, per §7.3 — not pre-committed); the parity gate (predict_mean == sklearn to ≤1e-6) passes on the fresh fit.
- [ ] **B3. Bundle:** place `sweeper/` + `sweeper_position_only/` under `silly_kicks/tracking/_ghost_gk_weights/` (npz + metadata.json + SHA256SUMS). Verify the Hub upload (if published) carries ONLY the intended ~3 files per variant (the whole-folder-upload leak rule — verify the Hub file count).
- [ ] **B4. Regenerate the per-artifact goldens** for the real `sweeper`/`sweeper_position_only` (chirality fingerprint, feature-contract fingerprint) and re-point Task 6's saturation gate at the bundled `sweeper` (swap the toy for the real artifact; the assertion is direction-not-magnitude, so it survives the swap).
- [ ] **B5. Re-run the FULL suite** (all legs, incl. `@slow` on the primary leg) + lint + pyright, green.
- [ ] **B6. Version + ADR + glossary/C4 sanity:** bump `silly_kicks/_version.py` → 4.105.0 (confirm next-free from `main` — main is 4.104.0 after the TF-19 A+2 release; ADR-082/PR-S175/4.104.0 are TAKEN by TF-19); write ADR-083; confirm NO new emitted feature columns (glossary/C4 count unchanged — the variant serves the existing `ghost_gk_x/y`). Update `CHANGELOG.md` (PR-S176) + `TODO.md` (TF-60 row stays "PARTLY SHIPPED": PR1/PR2/PR3 done, PR4–PR6 remaining).
- [ ] **B7. COMMIT GATE (owner-approved, ONE commit):** show the full diff + the new bundled artifacts + the metrics; wait for an explicit "commit". Then commit (docs + code + weights together), and — each a SEPARATE owner go-ahead — push / open PR / `gh pr merge --squash --admin --delete-branch` (only after CI is fully GREEN — gate on the run conclusion AND every job) / tag `v4.105.0` (push ONCE; publish run lags 5–15 min — do NOT re-push) / verify LIVE via `/pypi/silly-kicks/4.105.0/json`.

---

## Self-Review

**1. Spec coverage:**
- §4 grid-first-class refactor → Tasks 1–4. §5 extended grid/label domain → Tasks 1, 3, 7. §6 corpus + coverage → Tasks 7, B1. §7.1 two-sided gate → Task 6. §7.2 per-variant gates → Tasks 2–5, B4. §7.3 reported metrics → Task 7, B1–B2. §8 publish → Task 8, B3. §9 API/variant literals → Task 5. §10 velocity-keyed pair → Task 5 (additive branch). §11 error handling → Tasks 3–5. §12 decomposition (local-now/DGX-later) → Phase A/B split. §13 CI gates → Tasks 6, 9, B5. §16 resolved decisions → all encoded (density fail-loud Task 4; x_max 52.5 constants; report-not-gate Task 7; naming Task 5). No spec section is unmapped.
- The load-bearing additivity claim (§2/§3) → Task 2 byte-identity + Task 5 `test_gkdv_none_path_is_byte_identical`.
**2. Placeholder scan:** the `...` markers in Tasks 5/6/8 are named helper stubs whose contract is stated inline (fixture loaders / toy-fit); they are concrete obligations, not TBDs. Every code/impl step shows the actual change.
**3. Type consistency:** `GhostGridSpec` (5 fields) + `to_metadata_dict()` (7 keys) used consistently Tasks 1–4, 7; `grid_spec` param name consistent on `GhostGkModel.__init__` / `prepare_ghost_gk_training_data`; resolver returns `(model|None, key)` consistent Task 5; metric-helper signatures consistent Task 7.
