# TF-61 — xSuccess + VAEP_adjusted Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Ship two event-only primitives — a bundled, calibrated, **end-blind** action-completion model (`xsuccess`) and an opt-in outcome-bias-free VAEP rating (`VAEP.rate_adjusted`) that reuses a caller-fitted standard VAEP with no retrain.

**Architecture:** New event-only package `silly_kicks/xsuccess/` (XGBoost `P(success|context)`, start-anchored features only, pickle-free/fail-closed/xgboost-served — the `XShotOccurrenceModel` pattern). `silly_kicks/vaep/adjusted.py` + a `VAEP.rate_adjusted` method apply the paper's eqs 8–12 via a **surgical result-feature counterfactual** feeding the existing `formula.value`. Feature representation discovered by OpenEvolve, HPs by Optuna (both dev/train-time, ruthless-efficiency).

**Tech Stack:** Python, pandas, numpy, xgboost (fit + serve, `[xgboost]` extra), sklearn (isotonic calibration, fit-only), ruthless-efficiency[optuna] (`[train]` extra), OpenEvolve (dev-time), StatsBomb Open Data via statsbombpy.

**Spec:** `docs/superpowers/specs/2026-09-14-tf61-xsuccess-vaep-adjusted-design.md` (rev 4, APPROVED + owner-ratified 2026-09-15). Executors read both.

## Global Constraints

- **COMMIT POLICY (overrides the writing-plans per-task-commit template).** NO per-step or per-task commits. NO micro-commits. Tasks end at **"tests green,"** never at a commit. There are exactly **two human-gated commit checkpoints** (see "Commit & provenance sequencing"). Each commit is a fully-tested coherent state and lands only after explicit approval of its exact diff. **No worktrees** — one feature branch `feat/tf61-xsuccess-vaep-adjusted` off `main`.
- **The `/review-impl` gate runs in an independent session, never the author's.** HF publish, merge, and tag are each separate explicit go-aheads.
- **End-blind is a correctness invariant.** `xsuccess` features read only `{start_x, start_y, type_id, bodypart_id, time_seconds, period_id}` + **start**-derived geometry. NEVER `end_x`/`end_y` (target leakage) NEVER `result_id`/`result_name` (label). Enforced by the end-location invariance guard (Task 2).
- **Hexagonal boundary:** `xsuccess/` imports only `silly_kicks.spadl` (+ `spadl.config`), `silly_kicks.id_compat`, numpy, pandas; `xgboost`/`sklearn` **function-local** only. NEVER `silly_kicks.tracking`. Nothing imports `xsuccess` except `vaep`.
- **Fail-closed trained artifact (ADR-011/016/040/050):** pickle-free (booster JSON + metadata.json + SHA256SUMS); `load()` verifies SHA → chirality → feature-contract; inference imports no sklearn.
- **Corpus:** full redistributable StatsBomb Open Data via `load_open_data_matches` (open-data-only = redistributable by construction, the `train_pass_completion` convention — TF61-IMPL-01; `assert_public_corpus` is a pining-corpus visibility check, N/A for open data); `GroupKFold`/`StratifiedGroupKFold` by `game_id` (stringified). silly-kicks ships no xG; StatsBomb `shot_statsbomb_xg` is injected for validation only.
- **Additive:** no existing model/feature change, **no default VAEP retrain**, no `*_xfns`, no `feature_glossary` growth. One new C4 container `xsuccess`; `add_*` aggregator count stays 33.
- **Provenance (ADR-052/056):** every artifact driver calls `require_clean_tree(git_provenance())` in `main()` + `declare_inputs`, stamps `run_commit`/`run_tree_dirty`, registers in `ARTIFACT_DRIVERS`.
- **Version:** single per-PR minor bump in `silly_kicks/_version.py`, **claimed in Commit 2, NOT Commit 1** (owner decision — a concurrent silly-kicks session may release first). `_version.py` stays at `main`'s version through the code commit; provisional ~4.115.0 / ADR-094 / PR-S186, reconciled after `git fetch && git merge origin/main` at commit-prep.
- **Test/lint commands (TF61-PLAN-04 — the environment is specific).** Run tests with **`./.venv/Scripts/python.exe -m pytest ...`** and env **`SILLY_KICKS_ASSERT_INVARIANTS=1`** — NOT `uv run pytest` (re-resolves deps and fails: sklearn needs py≥3.11 while `.venv` is py3.10) and NOT bare `python`. **Never `pip install` into `.venv`.** Lint/type on system Python: `python -m ruff check silly_kicks/ tests/ scripts/`, `python -m ruff format --check silly_kicks/ tests/ scripts/`, `python -m pyright`. Every `Run:` line below uses `./.venv/Scripts/python.exe`.

## Commit & provenance sequencing (READ BEFORE STARTING)

The trained artifacts need a clean-tree run for honest provenance, which forces code-before-weights. This is spec §11 **rev 3 (owner-ratified 2026-09-15)**, refining rev-2's feature-split — recorded here so plan and spec agree:

- **Commit 1 (code):** all library code (xsuccess package incl. the *evolved* `_features.py`, `vaep/adjusted.py` + `rate_adjusted`), scripts, all unit tests, registry/docs; CHANGELOG carries an `[Unreleased]` heading and `_version.py` is **NOT** bumped (the number is claimed in Commit 2). Weights-dependent tests (`bundled()`, applied validation) are **skipped-pending-weights**. Full suite green, clean tree. (Tasks 1–11 + 9.)
- **Between the commits (execution, mine, DGX):** run the OpenEvolve discovery (dev; may be `--allow-dirty` — it is a discovery tool, output committed + validated), fold the evolved `_features.py` into Commit 1 *before* committing; then run the **final** train on the clean Commit-1 tree → clean-provenance weights + validation. (Tasks 12–13.)
- **Commit 2 (artifacts):** bundled `weights/` + `MODEL_CARD.md` + validation reports/provenance under `docs/research/xsuccess_vaep_adjusted/`; un-skip the weights-dependent tests; finalize CHANGELOG/TODO. Full suite green, clean provenance.
- HF publish (Task 12 driver) is a **separate go-ahead** after Commit 2 is approved.

---

## File structure

**Create:**
- `silly_kicks/xsuccess/__init__.py` — public `XSuccessModel`, `XSuccessIntegrityError`.
- `silly_kicks/xsuccess/_features.py` — `xsuccess_features(actions) -> np.ndarray`; `FEATURE_NAMES`; `feature_contract_block()`. End-blind.
- `silly_kicks/xsuccess/_model.py` — `XSuccessModel`, `XSuccessIntegrityError`.
- `silly_kicks/xsuccess/_objective.py` — `XSuccessObjective` (ruthless `CachedObjective`; `[train]`).
- `silly_kicks/xsuccess/weights/{model.json,metadata.json,SHA256SUMS,MODEL_CARD.md}` — bundled (Task 12).
- `silly_kicks/vaep/adjusted.py` — `adjusted_value(...)`.
- `scripts/evolve_xsuccess_features.py`, `scripts/train_xsuccess.py`, `scripts/publish_xsuccess.py`.
- `tests/xsuccess/{__init__.py?no,test_import_allowlist.py,test_features.py,test_leakage_guard.py,test_model_fit_predict.py,test_model_serialization.py,test_objective.py}`.
- `tests/vaep/test_adjusted.py`.
- `tests/scripts/` additions for the three drivers' provenance wiring.
- `docs/research/xsuccess_vaep_adjusted/{README.md,provenance.json}` (Task 12/13).

**Modify:**
- `silly_kicks/vaep/base.py` — add `VAEP.rate_adjusted`.
- `NOTICE`, `docs/c4/architecture.dsl` (+regen `.html`), `pyproject.toml` (ruff per-file-ignore for `X`), `CHANGELOG.md`, `TODO.md`, `silly_kicks/_version.py`.
- `tests/scripts/test_provenance_wiring.py` + `tests/scripts/_script_population.py` — register the 3 drivers (TF61-PLAN-03: the ADR-056 population lives in `_script_population.py`; there is **no** `test_artifact_driver_registry.py`).

> **Note on `tests/xsuccess/__init__.py`:** create an **empty** one (IMPL-03: `tests/expected_passing/__init__.py` exists and is empty — the earlier "has none" claim was wrong). It is REQUIRED so `test_import_allowlist.py` is package-qualified (`tests.xsuccess.test_import_allowlist`), avoiding the basename clash with `tests/restdefense/test_import_allowlist.py` under pytest prepend-import, and so cross-test imports (`from tests.xsuccess.test_model_fit_predict import _corpus`) resolve. The memory rule against `__init__.py` applies to a test dir mirroring a top-level NAMESPACE (`tests/scripts/`), not `tests/xsuccess/`.

---

### Task 1: Package skeleton + import-allowlist boundary

**Files:**
- Create: `silly_kicks/xsuccess/__init__.py`, `silly_kicks/xsuccess/_features.py` (stub), `silly_kicks/xsuccess/_model.py` (stub)
- Test: `tests/xsuccess/test_import_allowlist.py`

**Interfaces:**
- Produces: the `silly_kicks.xsuccess` package importing only the allowlist; `XSuccessModel`, `XSuccessIntegrityError` names (stubs fleshed out later).

- [ ] **Step 1: Write the failing allowlist test** (copy the shape of `tests/expected_passing/test_import_allowlist.py`, adapting the package name and the forbidden set).

```python
# tests/xsuccess/test_import_allowlist.py
import ast
import pathlib

_PKG = pathlib.Path(__file__).parents[2] / "silly_kicks" / "xsuccess"
_ALLOWED_TOP = {"silly_kicks.spadl", "silly_kicks.id_compat", "numpy", "pandas"}
_FORBIDDEN_SUBSTR = ("silly_kicks.tracking",)

def _module_level_imports(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods: set[str] = set()
    for node in ast.walk(tree):
        # module-level only: skip imports nested inside a function/class body
        if isinstance(node, (ast.Import, ast.ImportFrom)) and _is_module_level(tree, node):
            if isinstance(node, ast.Import):
                mods |= {a.name for a in node.names}
            elif node.module is not None and node.level == 0:
                mods.add(node.module)
    return mods

def _is_module_level(tree, node) -> bool:
    return any(node is child for child in ast.iter_child_nodes(tree))

def test_xsuccess_imports_only_allowlist():
    for py in _PKG.glob("*.py"):
        for mod in _module_level_imports(py):
            assert not any(s in mod for s in _FORBIDDEN_SUBSTR), f"{py.name} imports {mod}"
            top = mod.split(".")[0]
            if mod.startswith("silly_kicks"):
                assert any(mod == a or mod.startswith(a + ".") for a in _ALLOWED_TOP), f"{py.name}: {mod}"
            else:
                assert top in {"numpy", "pandas", "__future__", "importlib", "hashlib", "json", "warnings", "pathlib", "dataclasses"}, f"{py.name}: {mod}"
```

- [ ] **Step 2: Run — expect FAIL** (`ImportError`, package missing).
Run: `./.venv/Scripts/python.exe -m pytest tests/xsuccess/test_import_allowlist.py -v`

- [ ] **Step 3: Create the package.** `_features.py`/`_model.py` with module-level imports limited to the allowlist (xgboost/sklearn stay function-local). `__init__.py`:

```python
# silly_kicks/xsuccess/__init__.py
"""silly-kicks xSuccess (TF-61): event-only, end-blind action-completion P(success | context)
over all on-ball action types. See NOTICE for full bibliographic citations."""
from __future__ import annotations
from ._model import XSuccessIntegrityError, XSuccessModel

__all__ = ["XSuccessIntegrityError", "XSuccessModel"]
```

- [ ] **Step 4: Run — expect PASS.**

---

### Task 2: `_features.py` — end-blind feature builder + leakage guard

**Files:**
- Modify: `silly_kicks/xsuccess/_features.py`
- Test: `tests/xsuccess/test_features.py`, `tests/xsuccess/test_leakage_guard.py`

**Interfaces:**
- Produces:
  - `FEATURE_NAMES: list[str]` — seed order: `["seconds","start_x","start_y","distance_to_goal","angle_to_goal","period_id"]` + one-hot `type_<name>` (23) + one-hot `bodypart_<name>` (6). (Task 12 may replace the builder with the evolved version; `FEATURE_NAMES` reflects whatever ships.)
  - `xsuccess_features(actions: pd.DataFrame) -> np.ndarray` — `(n, len(FEATURE_NAMES))` float64; NaN-in (non-finite `start_x/y`) → all-NaN row.
  - `feature_contract_block() -> dict` — `{feature_names, geometry: {field_length, field_width, ...}, probe: {input, feature_vector}}`.

- [ ] **Step 1: Write feature tests.**

```python
# tests/xsuccess/test_features.py
import numpy as np, pandas as pd
import silly_kicks.spadl.config as cfg
from silly_kicks.xsuccess._features import FEATURE_NAMES, xsuccess_features

def _actions(**over):
    base = dict(type_id=cfg.actiontype_id["pass"], bodypart_id=cfg.bodypart_id["foot"],
                start_x=[20.0], start_y=[34.0], end_x=[60.0], end_y=[40.0],
                result_id=[cfg.result_id["success"]], time_seconds=[12.0], period_id=[1])
    base.update(over)
    return pd.DataFrame({k: (v if isinstance(v, list) else [v]) for k, v in base.items()})

def test_shape_and_names():
    X = xsuccess_features(_actions())
    assert X.shape == (1, len(FEATURE_NAMES))

def test_distance_and_angle_are_start_anchored():
    X = xsuccess_features(_actions(start_x=[20.0], start_y=[34.0]))
    di = FEATURE_NAMES.index("distance_to_goal")
    # goal centre (105, 34): distance from START = 85.0, independent of end
    assert round(float(X[0, di]), 1) == 85.0

def test_nan_start_gives_all_nan_row():
    X = xsuccess_features(_actions(start_x=[np.nan]))
    assert np.isnan(X[0]).all()

def test_onehot_type_and_bodypart_present():
    assert any(n.startswith("type_") for n in FEATURE_NAMES)
    assert any(n.startswith("bodypart_") for n in FEATURE_NAMES)
```

- [ ] **Step 2: Write the END-LOCATION leakage guard** (the ReceiverModel bar — this is the load-bearing novel guard).

```python
# tests/xsuccess/test_leakage_guard.py
import numpy as np, pandas as pd
import silly_kicks.spadl.config as cfg
from silly_kicks.xsuccess._features import xsuccess_features

def _multi():
    return pd.DataFrame(dict(
        type_id=[cfg.actiontype_id["pass"], cfg.actiontype_id["take_on"]],
        bodypart_id=[cfg.bodypart_id["foot"], cfg.bodypart_id["foot"]],
        start_x=[20.0, 55.0], start_y=[34.0, 20.0], end_x=[60.0, 70.0], end_y=[40.0, 25.0],
        result_id=[cfg.result_id["success"], cfg.result_id["fail"]],
        time_seconds=[12.0, 40.0], period_id=[1, 1]))

def test_features_invariant_to_end_and_result():
    a = _multi()
    base = xsuccess_features(a)
    b = a.copy(); b["end_x"] = b["end_x"] + 30.0; b["end_y"] = 5.0; b["result_id"] = cfg.result_id["success"]
    perturbed = xsuccess_features(b)
    assert np.array_equal(base, perturbed, equal_nan=True), "xSuccess features must be END-BLIND (and result-blind)"
```

- [ ] **Step 3: Run both — expect FAIL** (`xsuccess_features` not implemented).
Run: `./.venv/Scripts/python.exe -m pytest tests/xsuccess/test_features.py tests/xsuccess/test_leakage_guard.py -v`

- [ ] **Step 4: Implement `_features.py`** (start-anchored only; broadcast + NaN-in→NaN-out modeled on `expected_passing/_features.py`).

```python
# silly_kicks/xsuccess/_features.py  (seed; Task 12 may evolve the derived block)
from __future__ import annotations
import numpy as np
import silly_kicks.spadl.config as cfg

_GOAL = (float(cfg.field_length), float(cfg.field_width) / 2.0)
_TYPES = list(cfg.actiontypes); _BODY = list(cfg.bodyparts)
FEATURE_NAMES = (
    ["seconds", "start_x", "start_y", "distance_to_goal", "angle_to_goal", "period_id"]
    + [f"type_{t}" for t in _TYPES] + [f"bodypart_{b}" for b in _BODY]
)

def xsuccess_features(actions) -> np.ndarray:
    sx = np.asarray(actions["start_x"], float); sy = np.asarray(actions["start_y"], float)
    secs = np.asarray(actions["time_seconds"], float)
    period = np.asarray(actions["period_id"], float)
    tid = np.asarray(actions["type_id"]); bid = np.asarray(actions["bodypart_id"])
    dx = _GOAL[0] - sx; dy = _GOAL[1] - sy
    dist = np.hypot(dx, dy); ang = np.arctan2(dy, dx)  # from START to goal centre
    cols = [secs, sx, sy, dist, ang, period]
    for t in range(len(_TYPES)):
        cols.append((tid == t).astype(float))
    for b in range(len(_BODY)):
        cols.append((bid == b).astype(float))
    X = np.column_stack(cols)
    bad = ~np.isfinite(np.column_stack([sx, sy, secs, period])).all(axis=1)
    X[bad] = np.nan   # NOTE: to_numpy views are read-only under pandas-3 CoW; column_stack allocates, so this is writable
    return X

def feature_contract_block() -> dict:
    probe_x = xsuccess_features(_probe_actions())
    return {"feature_names": list(FEATURE_NAMES),
            "geometry": {"field_length": float(cfg.field_length), "field_width": float(cfg.field_width)},
            "probe": {"feature_vector": probe_x.tolist()}}
# _probe_actions(): a fixed 2-row DataFrame at opposite ends (define inline).
```

- [ ] **Step 5: Run both — expect PASS.**

---

### Task 3: `XSuccessModel` — fit + predict_success (+ calibration)

**Files:**
- Modify: `silly_kicks/xsuccess/_model.py`
- Test: `tests/xsuccess/test_model_fit_predict.py`

**Interfaces:**
- Produces: `XSuccessModel().fit(actions, *, calibrate=True) -> XSuccessModel`; `.predict_success(actions) -> np.ndarray` (∈[0,1], NaN-in→NaN); `.is_fitted`; `.feature_set` (`"xgboost"` | `"per_type_logistic"`); `XSuccessIntegrityError(RuntimeError)`.

- [ ] **Step 1: Write tests.**

```python
# tests/xsuccess/test_model_fit_predict.py
import numpy as np, pandas as pd
import silly_kicks.spadl.config as cfg
from silly_kicks.xsuccess import XSuccessModel, XSuccessIntegrityError

def _corpus(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 105, n)
    # deterministic-ish label from START context only (end-blind is safe to test):
    p = 1/(1+np.exp(-(3 - 0.04*(105-x))))
    y = (rng.uniform(size=n) < p).astype(int)
    return pd.DataFrame(dict(type_id=cfg.actiontype_id["pass"], bodypart_id=cfg.bodypart_id["foot"],
        start_x=x, start_y=rng.uniform(0,68,n), end_x=x, end_y=rng.uniform(0,68,n),
        result_id=np.where(y==1, cfg.result_id["success"], cfg.result_id["fail"]),
        time_seconds=rng.uniform(0,3000,n), period_id=1))

def test_unfitted_refuses():
    try:
        XSuccessModel().predict_success(_corpus(1))
    except XSuccessIntegrityError:
        return
    raise AssertionError("expected XSuccessIntegrityError")

def test_fit_predict_range():
    m = XSuccessModel().fit(_corpus())
    assert m.is_fitted
    p = m.predict_success(_corpus(50, seed=1))
    assert p.shape == (50,) and np.nanmin(p) >= 0.0 and np.nanmax(p) <= 1.0

def test_nan_feature_predicts_nan():
    m = XSuccessModel().fit(_corpus())
    a = _corpus(1); a.loc[a.index[0], "start_x"] = np.nan
    assert np.isnan(m.predict_success(a)[0])
```

- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement fit/predict** (xgboost + optional isotonic, both function-local; label `result_id == success`; drop `non_action`; drop non-finite rows before fit). Fit builds `X = xsuccess_features(actions)`, `y = (result_id == success)`; trains `xgb.XGBClassifier(**params)`; if `calibrate`, fit `sklearn.isotonic.IsotonicRegression` on out-of-fold probs and store its `X_thresholds_`/`y_thresholds_` for pure-numpy `np.interp` at serve. `predict_success`: `xsuccess_features` → booster `predict_proba`[:,1] → `np.interp` isotonic → set NaN rows to NaN.
- [ ] **Step 4: Run — expect PASS.**

---

### Task 4: `XSuccessModel` — pickle-free serialization + fail-closed load

**Files:**
- Modify: `silly_kicks/xsuccess/_model.py`
- Test: `tests/xsuccess/test_model_serialization.py`

**Interfaces:**
- Produces: `.save(path)`, `XSuccessModel.load(path, *, legacy_override=False)`, `.to_dict()`/`from_dict()`, `XSuccessModel.bundled()`. Artifact = `model.json` (booster) + `metadata.json` (feature_names, one-hot vocab, feature_contract, chirality probe, isotonic params, feature_set, training_commit, corpus) + `SHA256SUMS`.

- [ ] **Step 1: Write tests.**

```python
# tests/xsuccess/test_model_serialization.py
import json, numpy as np, pandas as pd, pytest
from silly_kicks.xsuccess import XSuccessModel, XSuccessIntegrityError
# reuse _corpus from test_model_fit_predict (import it)
from tests.xsuccess.test_model_fit_predict import _corpus

def test_roundtrip(tmp_path):
    m = XSuccessModel().fit(_corpus()); m.save(tmp_path)
    m2 = XSuccessModel.load(tmp_path)
    a = _corpus(30, seed=2)
    assert np.allclose(m.predict_success(a), m2.predict_success(a), atol=1e-6, equal_nan=True)

def test_sha_tamper_raises(tmp_path):
    XSuccessModel().fit(_corpus()).save(tmp_path)
    (tmp_path / "model.json").write_text((tmp_path / "model.json").read_text() + " ")
    with pytest.raises(XSuccessIntegrityError):
        XSuccessModel.load(tmp_path)

def test_feature_contract_drift_raises(tmp_path):
    XSuccessModel().fit(_corpus()).save(tmp_path)
    d = json.loads((tmp_path / "metadata.json").read_text())
    d["feature_contract"]["geometry"]["field_length"] = 100.0
    (tmp_path / "metadata.json").write_text(json.dumps(d))
    with pytest.raises(XSuccessIntegrityError):
        XSuccessModel.load(tmp_path)

def test_chirality_mismatch_raises(tmp_path):
    # TF61-PLAN-02: chirality must be tested DIRECTLY (not only via bundled()).
    # Precondition (a design constraint this pins): SHA256SUMS covers model.json (the booster),
    # NOT metadata.json — expected_passing precedent — so a tampered chirality fingerprint trips
    # the CHIRALITY check, not the SHA check.
    XSuccessModel().fit(_corpus()).save(tmp_path)
    d = json.loads((tmp_path / "metadata.json").read_text())
    d["chirality"]["probe_prediction"] = [min(1.0, v + 0.5) for v in d["chirality"]["probe_prediction"]]
    (tmp_path / "metadata.json").write_text(json.dumps(d))
    with pytest.raises(XSuccessIntegrityError):
        XSuccessModel.load(tmp_path)

def test_bundled_loads_or_skips():
    try:
        XSuccessModel.bundled()
    except FileNotFoundError:
        pytest.skip("weights bundled in Commit 2 (Task 12)")
```

- [ ] **Step 2: Run — expect FAIL (or the bundled test SKIP).**
- [ ] **Step 3: Implement serialization** mirroring `tracking/_xshot_occurrence.py` (`load_xgb_booster_base_score_safe` for the 2.x/3.x `base_score`) and `expected_passing/_model.py` (chirality + feature-contract verify, warn-on-missing / raise-on-drift). `bundled()` loads `importlib.resources.files("silly_kicks.xsuccess")/"weights"`.
- [ ] **Step 4: Run — expect PASS (bundled SKIP).**

---

### Task 5: `XSuccessObjective` — Optuna HPO (ruthless CachedObjective)

**Files:**
- Create: `silly_kicks/xsuccess/_objective.py`
- Test: `tests/xsuccess/test_objective.py`

**Interfaces:**
- Produces: `XSuccessObjective(fold={...})` with `patch_params`, `prepare()`, `evaluate_patch(inv, candidate)`, `evaluate(candidate)` returning `{"logloss","brier","pr_auc"}`; search keys `{n_estimators,max_depth,learning_rate,min_child_weight,reg_lambda,reg_alpha,subsample,colsample_bytree}`.

- [ ] **Step 1: Write the cache-equivalence test** (the non-tautological correctness gate).

```python
# tests/xsuccess/test_objective.py
import numpy as np
from ruthless.result import Candidate
from silly_kicks.xsuccess._objective import XSuccessObjective
from tests.xsuccess.test_model_fit_predict import _corpus
from silly_kicks.xsuccess._features import xsuccess_features
import silly_kicks.spadl.config as cfg

def _fold():
    a = _corpus(300); X = xsuccess_features(a)
    y = (a["result_id"].to_numpy() == cfg.result_id["success"]).astype(int)
    g = np.arange(len(a)) % 6  # 6 pseudo-matches
    return {"synthetic": [(X, y, g)]}

def test_cache_equivalence():
    obj = XSuccessObjective(fold=_fold())
    c = Candidate(params=dict(n_estimators=40, max_depth=3, learning_rate=0.1,
                              min_child_weight=1, reg_lambda=1.0, reg_alpha=0.0,
                              subsample=1.0, colsample_bytree=1.0))
    inv = obj.prepare()
    assert abs(obj.evaluate_patch(inv, c)["logloss"] - obj.evaluate(c)["logloss"]) < 1e-9
```

- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement** by copying `silly_kicks/tracking/_xshot_occurrence_objective.py` almost verbatim: `prepare()` concatenates the fold `(X,y,groups)`; `_cv_logloss` uses `StratifiedGroupKFold` (stringify groups) + `xgb.XGBClassifier`; drop the `subsample_negatives` / `_pinned_params`-from-xshot coupling and use the local search keys. `[train]` extra; not imported by `__init__`.
- [ ] **Step 4: Run — expect PASS** (needs `[train]` + `[xgboost]`; if unavailable in the leg, mark the module import `importorskip`).

---

### Task 6: `scripts/evolve_xsuccess_features.py` — OpenEvolve driver

**Files:**
- Create: `scripts/evolve_xsuccess_features.py`
- Test: `tests/scripts/test_evolve_xsuccess_smoke.py`

**Interfaces:**
- Produces: a CLI that (a) `require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)` in `main()`, (b) loads the corpus (public), (c) runs the OpenEvolve search over the feature-builder body within the end-free allowlist, fitness = held-out StratifiedGroupKFold calibrated log-loss + calibration-in-the-large + per-type reliability + regression penalties, (d) writes the winning `xsuccess_features` body candidate + `docs/research/xsuccess_vaep_adjusted/provenance.json` (config, run_commit, tree_dirty, metrics, LOO/CV).

- [ ] **Step 1: Smoke test** — the script imports, exposes `--help` safely (has an argparse parser — memory: parserless scripts execute `main()` on `--help`), and `require_clean_tree` is called from `main()`.

```python
# tests/scripts/test_evolve_xsuccess_smoke.py
import ast, pathlib
P = pathlib.Path(__file__).parents[2] / "scripts" / "evolve_xsuccess_features.py"
def test_has_parser_and_provenance():
    src = P.read_text(encoding="utf-8")
    assert "add_argument" in src            # parser exists → --help is safe
    assert "require_clean_tree" in src
    assert "rev-parse" not in src           # ADR-052/056: never shell out to rev-parse
```

- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement** the driver (argparse; `--allow-dirty`, `--corpus-*`, `--subsample-matches`, `--out`). The OpenEvolve integration + LLM proposers run at execution time (Task 12); the committed driver encodes config, fitness, allowlist enforcement, provenance stamping. Restrict the candidate program's readable columns to the end-free allowlist (§6.4).
- [ ] **Step 4: Run — expect PASS.**

---

### Task 7: `scripts/train_xsuccess.py` — HPO + calibrate + train + save

**Files:**
- Create: `scripts/train_xsuccess.py`
- Test: `tests/scripts/test_train_xsuccess_smoke.py`

**Interfaces:**
- Produces: a CLI that loads the public corpus via `load_open_data_matches` (open-data-only = the redistributability gate, TF61-IMPL-01), runs `XSuccessObjective` under Optuna (ruthless), fits isotonic if it improves held-out per-type reliability, trains the final `XSuccessModel`, `save()`s to `silly_kicks/xsuccess/weights/`, writes the validation battery (§9.1) + provenance; `require_clean_tree` in `main()`.

- [ ] **Step 1: Smoke test** (parser + provenance + `load_open_data_matches` present / no `_loader_pining` — the TF61-IMPL-01 redistributability gate).

```python
# tests/scripts/test_train_xsuccess_smoke.py
import pathlib
P = pathlib.Path(__file__).parents[2] / "scripts" / "train_xsuccess.py"
def test_train_xsuccess_wiring():
    src = P.read_text(encoding="utf-8")
    for needle in ("add_argument", "require_clean_tree", "load_open_data_matches"):  # redistributability gate
        assert needle in src, needle
    assert "_loader_pining" not in src  # never the non-redistributable pining loader (TF61-IMPL-01)
    assert "rev-parse" not in src
```

- [ ] **Step 2: Run — expect FAIL. Step 3: Implement** (model routing to xgboost family; per-type-logistic fallback path behind `--family per_type_logistic`; `--subsample-fps`/n/a here; GroupKFold-by-match; writes `weights/` + `MODEL_CARD.md` + validation JSON). **Step 4: Run — expect PASS.**

---

### Task 8: `scripts/publish_xsuccess.py` — HF publish (card-required seam)

**Files:**
- Create: `scripts/publish_xsuccess.py`
- Test: `tests/scripts/test_publish_xsuccess_guard.py`

**Interfaces:**
- Produces: a CLI delegating to `scripts/_hub_publish.publish_model_with_card` (ADR-088) with `--model-card` required, model-only allowlist (`model.json`, `metadata.json`, `SHA256SUMS`), `create_repo(exist_ok=True)`.

- [ ] **Step 1: Guard test** (fake `HfApi`, mirror `tests/scripts/test_hub_publish_guard.py`): a card-less publish refuses before any network; the card uploads as `README.md`.
- [ ] **Step 2: Run — FAIL. Step 3: Implement** (delegate to the seam; never call `upload_model_only` directly). **Step 4: Run — PASS.**

---

### Task 9: Registry & docs wiring (Commit-1 gates)

**Files:**
- Modify: `NOTICE`, `docs/c4/architecture.dsl` (+regen `.html`), `pyproject.toml`, `silly_kicks/_version.py`, `CHANGELOG.md`, `TODO.md`, `tests/scripts/_script_population.py` + `test_provenance_wiring.py` (register the 3 drivers).

- [ ] **Step 1: C4 gate first (red).** Add the `xsuccess` container to `architecture.dsl`; run the C4 completeness gate — it fails until `.html` is regenerated.
Run: `./.venv/Scripts/python.exe -m pytest tests/ -k "c4 or architecture" -v`
- [ ] **Step 2: Regenerate `architecture.html`** via the `mad-scientist-skills:c4` pipeline with Graphviz `dot` (never Smetana) — `-graphvizdot "C:/Users/Karsten/.claude/tools/graphviz/dot.exe"`. Confirm `add_*` count stays 33.
- [ ] **Step 3: NOTICE** — add Paul/Klemp/Memmert 2025; Anzer–Bauer Expected Passes; von Neumann–Morgenstern / Bernoulli; OpenEvolve/AlphaEvolve. Add per-feature docstring `See NOTICE …`.
- [ ] **Step 4: pyproject** — add `silly_kicks/xsuccess/*` (and `tests/xsuccess/*`) to the ruff per-file-ignore for `X`/`Y` uppercase naming (mirror the vaep/xthreat entries). Confirm `[xgboost]` + `[train]` extras already cover xsuccess (they do).
- [ ] **Step 5: ARTIFACT_DRIVERS + population gate** — register `evolve_xsuccess_features.py`, `train_xsuccess.py` (publish is `_NOT_A_DRIVER` — it uploads, doesn't produce a research artifact, but its output repo is policed; classify per the existing buckets). Run the ADR-056 population gate to green.
- [ ] **Step 6: CHANGELOG + TODO (NO version bump).** CHANGELOG gets an `[Unreleased]` TF-61 entry (version / PR-S / ADR left as `<nnn>` placeholders); TODO gets a brief in-progress note. **Do NOT bump `silly_kicks/_version.py`** — the number is claimed in Commit 2 at commit-prep (deferred so a concurrent session can release first).
- [ ] **Step 7: Run the full non-e2e suite** (`./.venv/Scripts/python.exe -m pytest tests/ -m "not e2e"` with `SILLY_KICKS_ASSERT_INVARIANTS=1`) + `python -m ruff check silly_kicks/ tests/ scripts/` + `python -m ruff format --check silly_kicks/ tests/ scripts/` + `python -m pyright`. Expect green with the `bundled()`/applied-validation tests SKIPPED.
- [ ] **Step 8: Doctests (public surface, TF61-PLAN-05 / spec §10).** Give the PUBLIC surface — `XSuccessModel` (+ `bundled`/`predict_success` as re-exported), `VAEP.rate_adjusted` (`vaep/base.py`), `adjusted_value` (`vaep/adjusted.py`) — doctest-safe docstrings: ≥4-space-indented RST literal blocks for any example needing a real `actions`/fitted model (mirror `PassCompletionModel`), executable `>>>` only where self-contained. Private `_model.py`/`_features.py` examples stay correct but are not CI-run (single-underscore modules are `--ignore-glob`'d). Run: `./.venv/Scripts/python.exe -m pytest --doctest-modules silly_kicks/ --ignore-glob="*/_[!_]*.py" -v` (the CI public-surface sweep, CLAUDE.md).

> **No `feature_glossary` entry and no `*_xfns`:** xSuccess is a model (reads `result` only as a label) and VAEP_adjusted is a rating method — confirm the leaky-`*_xfns`-absence guard and glossary-coverage gate stay green with no additions (they key on `add_*`/`*_xfns` names, which we don't create).

---

### Task 10: `vaep/adjusted.py` — `adjusted_value` (eqs 8–12)

**Files:**
- Create: `silly_kicks/vaep/adjusted.py`
- Test: `tests/vaep/test_adjusted.py` (part 1)

**Interfaces:**
- Consumes: `silly_kicks.vaep.formula.value`.
- Produces: `adjusted_value(actions, p_scores_success, p_concedes_fail, xsuccess) -> pd.DataFrame` with columns `offensive_value/defensive_value/vaep_value`; `p_scores_adj = xsuccess * p_scores_success`, `p_concedes_adj = (1 - xsuccess) * p_concedes_fail`, then `formula.value(actions, p_scores_adj, p_concedes_adj)`.

- [ ] **Step 1: Write tests** — eq 8–12 arithmetic + the `xs[i-1]` prev-weighting pin (TF61-SPEC-04).

```python
# tests/vaep/test_adjusted.py  (part 1)
import numpy as np, pandas as pd
from silly_kicks.spadl.utils import add_names
from silly_kicks.vaep import formula
from silly_kicks.vaep.adjusted import adjusted_value

def _acts(team):  # minimal frame formula.value needs: team_id, time_seconds, type_name, result_name
    n = len(team)
    df = pd.DataFrame(dict(team_id=team, time_seconds=np.arange(n, dtype=float),
        type_id=1, result_id=1, bodypart_id=0, start_x=50.0, start_y=34.0, end_x=60.0, end_y=34.0))
    return add_names(df)

def test_weighting_matches_formula():
    a = _acts([1,1,1])
    pss = pd.Series([0.2,0.5,0.7]); pcf = pd.Series([0.1,0.1,0.2]); xs = pd.Series([0.9,0.4,0.8])
    got = adjusted_value(a, pss, pcf, xs)
    want = formula.value(a, xs.values*pss.values, (1-xs).values*pcf.values)
    assert np.allclose(got["vaep_value"], want["vaep_value"])

def test_prev_state_uses_xs_of_previous_action():
    # team switch at index 2 → the delta for row 2 must draw on row 1's ADJUSTED prob (xs[1]), not xs[2].
    a = _acts([1,1,2,2])
    pss = pd.Series([0.3,0.6,0.2,0.5]); pcf = pd.Series([0.1,0.2,0.1,0.3]); xs = pd.Series([0.5,0.9,0.5,0.7])
    got = adjusted_value(a, pss, pcf, xs)
    # reference: build p_scores_adj/p_concedes_adj explicitly and run formula.value
    ref = formula.value(a, (xs*pss).values, ((1-xs)*pcf).values)
    assert np.allclose(got["vaep_value"], ref["vaep_value"])   # pins that prev() sees per-action xs
```

- [ ] **Step 2: Run — FAIL. Step 3: Implement `adjusted_value`.** **Step 4: Run — PASS.**

---

### Task 11: `VAEP.rate_adjusted` — surgical counterfactual

**Files:**
- Modify: `silly_kicks/vaep/base.py`
- Test: `tests/vaep/test_adjusted.py` (part 2)

**Interfaces:**
- Consumes: `self.compute_features`, `self._estimate_probabilities`, `xsuccess_model.predict_success`, `adjusted_value`.
- Produces: `VAEP.rate_adjusted(game, game_actions, xsuccess_model, *, frames=None, return_components=False) -> pd.DataFrame`.

Mechanism: compute `X = self.compute_features(game, actions, frames=frames)`; build `X_succ`/`X_fail` by **overriding only the current-action (a0) result-encoding columns** (`result_onehot`/`actiontype_result_onehot` for the a0 slot) to success/fail — do NOT rebuild from a mutated action table (avoids the goalscore/predecessor artifacts). Non-vacuity: assert `X_succ` differs from `X_fail` on ≥1 row → else raise (HybridVAEP no-op). Re-score `self._estimate_probabilities` on each; take the P_scores column from the succ scoring and the P_concedes column from the fail scoring; `xs = xsuccess_model.predict_success(actions)`; return `adjusted_value(...)`.

- [ ] **Step 1: Write tests.**

```python
# tests/vaep/test_adjusted.py  (part 2)
import numpy as np, pandas as pd, pytest
from silly_kicks.vaep import VAEP
from silly_kicks.vaep.hybrid import HybridVAEP
# a tiny fitted VAEP + a tiny fitted XSuccessModel via the _corpus helpers; keep synthetic + fast.

def test_hybrid_raises_non_vacuity(fitted_hybrid, fitted_xs, game, actions):
    with pytest.raises(ValueError, match="result-bearing|no-op|standard"):
        fitted_hybrid.rate_adjusted(game, actions, fitted_xs)

def test_rate_adjusted_shapes(fitted_vaep, fitted_xs, game, actions):
    out = fitted_vaep.rate_adjusted(game, actions, fitted_xs)
    assert list(out.columns) == ["offensive_value", "defensive_value", "vaep_value"]
    assert len(out) == len(actions)

def test_nan_xsuccess_propagates(fitted_vaep, fitted_xs, game, actions):
    a = actions.copy(); a.loc[a.index[0], "start_x"] = np.nan
    assert np.isnan(fitted_vaep.rate_adjusted(game, a, fitted_xs)["vaep_value"].iloc[0])

def test_purity_no_mutation(fitted_vaep, fitted_xs, game, actions):
    snap = actions.copy(deep=True)
    fitted_vaep.rate_adjusted(game, actions, fitted_xs)
    pd.testing.assert_frame_equal(actions, snap)

def test_rate_adjusted_requires_fitted(fitted_xs, game, actions):  # TF61-PLAN-02 (spec §7.3)
    from sklearn.exceptions import NotFittedError
    with pytest.raises(NotFittedError):
        VAEP().rate_adjusted(game, actions, fitted_xs)

def test_frames_threaded_to_both_feature_builds(fitted_vaep_frame_aware, fitted_xs, game, actions, frames):
    # TF61-PLAN-02 (spec §7.3): a frame-aware VAEP must receive `frames` for BOTH counterfactual
    # feature builds; supplying them works, omitting them raises (mirrors compute_features' own guard).
    out = fitted_vaep_frame_aware.rate_adjusted(game, actions, fitted_xs, frames=frames)
    assert len(out) == len(actions)
    with pytest.raises(ValueError, match="frames"):
        fitted_vaep_frame_aware.rate_adjusted(game, actions, fitted_xs, frames=None)

def test_surgical_flip_leaves_goalscore_and_locations(fitted_vaep, fitted_xs, game, actions):
    # assert the a0 result columns change between X_succ/X_fail and the goalscore/endlocation columns do not
    ...  # implement via a hook or by comparing the two feature matrices the method builds
```

- [ ] **Step 2: Run — FAIL. Step 3: Implement `rate_adjusted`** on `VAEP` (uses `self._VAEP__models` via the existing `_estimate_probabilities`; column-name detection for the a0 result one-hots from `self._feature_columns()`). **Step 4: Run — PASS.**
- [ ] **Step 5: Run the full non-e2e suite + ruff + pyright.** Expect green (weights-dependent SKIPs remain).

> **── CODE COMPLETE (SEED features) ──** All of Tasks 1–11 + 9 green with the SEED `_features.py`. Do **not** commit yet — the evolved representation (Task 12 Step 1) folds into this *same* code commit, so Commit Checkpoint 1 comes **after** the evolve, never here (no amend).

---

### Task 12: [EXECUTION — mine, DGX] Evolve → HPO → train → bundle xSuccess

Not a CI unit task — this runs the pipeline and produces committed artifacts.

- [ ] **Step 1:** Run `scripts/evolve_xsuccess_features.py` on a stratified corpus subsample (dev; `--allow-dirty` acceptable — discovery tool). Review the winning `xsuccess_features` body for interpretability + leakage (it must pass Task 2's end-location guard). Fold it into `silly_kicks/xsuccess/_features.py`; **if it does not beat the seed by the pre-registered margin, keep the seed** (§6.1). Re-run the full unit suite (features/leakage/model tests) green.

> **── COMMIT CHECKPOINT 1 (code) ──** All of Tasks 1–11 + 9 green **with the final (evolved-or-seed) `_features.py` folded in**. Present the exact diff for approval. On approval: commit `feat/tf61: xSuccess + VAEP_adjusted (code)` on `feat/tf61-xsuccess-vaep-adjusted`. No push/merge yet. This is the clean tree the final train stamps its `training_commit` against.

- [ ] **Step 2:** On that clean Commit-1 tree, run `scripts/train_xsuccess.py` on the full public corpus → `weights/{model.json,metadata.json,SHA256SUMS,MODEL_CARD.md}` with clean `training_commit`. Confirm the family gate: XGBoost unless per-type calibration fails → per-type-logistic fallback; record the decision.
- [ ] **Step 3:** Un-skip `test_bundled_loads_or_skips` (now loads) and the applied-validation tests; run them green.
- [ ] **Step 4:** Write `docs/research/xsuccess_vaep_adjusted/README.md` + `provenance.json` with the §9.1 battery: per-type reliability curves + Brier, calibration-in-the-large, OOF ROC-AUC (with/without one-sided types), evolved-representation LOO/CV, and the **outcome-conditional negative control** (end-using control vs the shipped end-blind model — the guard bites).

---

### Task 13: [EXECUTION — mine] VAEP_adjusted validation (§9.2)

- [ ] **Step 1:** On a fitted standard VAEP (public corpus) + the bundled xSuccess, compute `rate_adjusted` and record: aggregate `Σ P(scores|success)_adj / k` vs total StatsBomb xG (account for penalty/corner fixed odds, `formula.py:71-77`); outcome-bias-reduction plots; the Dortmund-shaped case study (shot correction clean; risky-pass milder, as documented); and the **literal-whole-dataset-flip cross-check** (once, to quantify the artifact).
- [ ] **Step 2:** Append results to `docs/research/xsuccess_vaep_adjusted/README.md`; finalize `CHANGELOG.md` + `TODO.md`.

> **── COMMIT CHECKPOINT 2 (artifacts) ──** Weights + validation green, clean provenance. Present the diff for approval. On approval: commit `feat/tf61: xSuccess bundled weights + validation`. Then, as **separate** go-aheads: HF publish (`scripts/publish_xsuccess.py`), PR/merge, tag. The `/review-impl` gate runs in an independent session before merge.

---

## Self-Review

**Spec coverage:** §4 architecture → Tasks 1,10,11,9(C4). §5 model (label/domain/geometry/features/learner/calibration/artifact/serve) → Tasks 2,3,4. §6 optimization (evolve/HPO/calibrate/leakage) → Tasks 2(guard),5,6,7,12. §7 VAEP_adjusted (surgical flip/API/guards) → Tasks 10,11. §8 corpus → Tasks 7,12. §9 validation → Tasks 12,13. §10 testing/registries → Tasks 1,2,4,8,9,10,11. §11 delivery → Commit checkpoints + Global Constraints. No spec section is unmapped.

**Placeholder scan:** the only intentionally-deferred content is the *evolved* `_features.py` body (discovered in Task 12, gated by the Task-2 guard + seed-fallback) and the trained `weights/` (Task 12) — both are execution outputs, not plan placeholders; the seed builder is fully specified. Task 11's `test_surgical_flip_leaves_goalscore_and_locations` and the pytest fixtures (`fitted_vaep`, `fitted_vaep_frame_aware`, `fitted_xs`, `game`, `actions`, `frames`) are named but their bodies must be written from the `_corpus` helpers — call out to the implementer: build these as a `conftest.py` in `tests/vaep/` from a synthetic 2-team, multi-action frame (and, for `fitted_vaep_frame_aware`/`frames`, one frame-aware xfn + a matching `TRACKING_FRAMES_COLUMNS` frame set).

**Type consistency:** `xsuccess_features` returns `np.ndarray` everywhere; `predict_success` returns `np.ndarray`; `adjusted_value` and `rate_adjusted` return the 3-column `offensive/defensive/vaep_value` DataFrame consistently; `XSuccessObjective` metrics dict keys `{logloss,brier,pr_auc}` match the xShot precedent.

**Known gap to close during execution:** confirm the exact a0 result one-hot column names emitted by `result_onehot`/`actiontype_result_onehot` (from `vaep/features/result.py` + `actiontype.py`) so Task 11's surgical override targets the right columns; write that detection against `self._feature_columns()`, not a hard-coded name.
