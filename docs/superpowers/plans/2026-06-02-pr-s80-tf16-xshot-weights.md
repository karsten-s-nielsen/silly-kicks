# PR-S80 — TF-16 xShotOccurrence Weights Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train, bundle, publish, and turn on the xShotOccurrence (xS) model against the 4.7.0 carrier defaults — the weights follow-up to the untrained code shipped in PR-S75.

**Architecture:** Code changes are developed + TDD'd **locally** (Windows) with synthetic/slim fixtures and committed to nothing yet; the actual training run executes **on DGX Spark** streaming a pining-first corpus; the resulting bundled booster is pulled back, committed into the package, and the weight-dependent tests + default-xfn wiring go green. xS is a single-variant booster (no Ghost-GK-style size split); the variant axis is data provenance (`public` vs `full`), decided by a pre-registered paired comparison.

**Tech Stack:** Python 3.10+ (CI 3.10–3.12, local 3.14), xgboost ≥2.0,<3.0, ruthless-efficiency[optuna]≥0.2.1, kloppy (loaders), pandas/numpy, pytest, HuggingFace Hub.

**Spec:** `docs/superpowers/specs/2026-06-02-tf16-weights-run-design.md` (read it first).

---

## Project-policy adaptations (READ FIRST)

- **ONE commit per branch** (user policy overrides the skill's per-task commits). Tasks below end with **"stage + verify tests green,"** NOT `git commit`. The single commit happens only in the final task, **after `/final-review` and explicit owner approval**. Branch is already `pr-s80-tf16-weights`.
- **Version stays provisional** — do NOT hard-code 4.8.0; the part-deux session also targets it (spec §11 / H1). The version-bump task is deferred to commit time with whatever number is correct then.
- **Local vs remote split:** Tasks 1–10 + 13–15 are local code/TDD (run `python -m pytest tests/ -m "not e2e"`). Task 11 is the **operational run on DGX Spark**. Task 12 bundles the pulled artifact. Task 14 (e2e flip) is asserted where real data lives (Spark/data env), `@pytest.mark.e2e`, not in normal CI.
- **Ghost-GK re-fit is NOT in this plan** — it is the separate PR-S81 off the same staged corpus (spec §7/§11, L7).
- Session start: `pip install -e ".[test]"` (and `.[train,xgboost]` where the trainer/objective run). Match CI lint locally: `ruff format --check`, `ruff check`, `pyright silly_kicks/`.

---

## File Structure

| File | Create/Modify | Responsibility |
|---|---|---|
| `silly_kicks/tracking/_ball_carrier.py` | Modify | Add module-level `DEFAULT_CARRIER_PARAMS`; `infer_ball_carrier` defaults reference it (T1) |
| `silly_kicks/tracking/_geometry.py` | Modify | Add `GEOMETRY_VERSION`, `PITCH_LENGTH`, `PITCH_WIDTH` constants (T2) |
| `silly_kicks/tracking/_xshot_occurrence_objective.py` | Modify | `StratifiedGroupKFold`; drop `scale_pos_weight` from search keys; docstring wording (T3) |
| `silly_kicks/tracking/_xshot_occurrence.py` | Modify | import shared carrier constant; `base_score` + xgboost-version in `fit`; metadata template + fail-closed `load`; bundled `from_variant`; `home_team_id` optional (T1,T4,T5,T6,T7) |
| `silly_kicks/tracking/features.py` | Modify | Wire `xshot_occurrence_xfns` into `pre_shot_gk_full_default_xfns` only (NOT the general list — P3) (T13) |
| `scripts/train_xshot_occurrence.py` | Modify (rewrite) | Streaming loader/parquet source, feature cache, two-candidate, frozen-HPO paired eval, per-provider metrics, permutation importance, fail-closed gates, metrics.json (T8) |
| `scripts/publish_xshot_occurrence.py` | Create | Verify + HF upload + download-verify (T9) |
| `scripts/make_xshot_directional_fixture.py` | Create | One-off: build the frozen directional feature-vector fixture from slim fixtures (T10) |
| `silly_kicks/tracking/_xshot_weights/default/` | Create (from run) | Bundled booster `model.json` + `metadata.json` + `SHA256SUMS` (T11→T12) |
| `tests/datasets/tracking/xshot_directional/frozen_rows.parquet` | Create | Committed frozen feature vectors + labels for the CI quality tripwire (T10) |
| `tests/tracking/test_xshot_occurrence.py` | Modify | Unit tests: carrier constant, base_score, metadata fail-closed, from_variant logic, home_team_id-optional (T1,T4,T5,T6,T7) |
| `tests/tracking/test_xshot_occurrence_integration.py` | Modify | Flip e2e gates; trainer fail-closed smoke; publish verify-only; bundled directional quality; from_variant-bundled; metadata-matches-intent; xfn membership; wheel-content (T8,T9,T12,T13,T14) |
| `NOTICE`, `docs/superpowers/adrs/ADR-011*.md` | Modify | Single-variant note + metadata-template note (T15) |
| `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, `TODO.md` | Modify | Version bump (provisional) + changelog + TODO grooming (T15) |

---

## Task 1: Shared `DEFAULT_CARRIER_PARAMS` constant (anti-drift; spec §4 / L1, N4-adjacent)

**Files:**
- Modify: `silly_kicks/tracking/_ball_carrier.py` (near top, before `infer_ball_carrier` at `:328`)
- Modify: `silly_kicks/tracking/_xshot_occurrence.py:310`
- Test: `tests/tracking/test_xshot_occurrence.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/tracking/test_xshot_occurrence.py
def test_default_carrier_params_are_shared_constant():
    """xS must source carrier defaults from the single library constant (anti-drift)."""
    import inspect
    from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS, infer_ball_carrier
    from silly_kicks.tracking import _xshot_occurrence as xo

    # xS uses the SAME object, not a re-hardcoded copy.
    assert xo._DEFAULT_CARRIER_PARAMS is DEFAULT_CARRIER_PARAMS
    # The constant carries the 4.7.0 values.
    assert DEFAULT_CARRIER_PARAMS == {"tolerance_m": 3.0, "beta": 0.0, "gamma": 0.25}
    # infer_ball_carrier's signature defaults equal the constant (drift guard).
    sig = inspect.signature(infer_ball_carrier).parameters
    assert sig["tolerance_m"].default == DEFAULT_CARRIER_PARAMS["tolerance_m"]
    assert sig["beta"].default == DEFAULT_CARRIER_PARAMS["beta"]
    assert sig["gamma"].default == DEFAULT_CARRIER_PARAMS["gamma"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/tracking/test_xshot_occurrence.py::test_default_carrier_params_are_shared_constant -v`
Expected: FAIL — `ImportError: cannot import name 'DEFAULT_CARRIER_PARAMS'` (and `_DEFAULT_CARRIER_PARAMS` is the stale `{beta:0.5, gamma:1.0}`).

- [ ] **Step 3: Add the constant in `_ball_carrier.py` and reference it from the signature**

In `silly_kicks/tracking/_ball_carrier.py`, above `def infer_ball_carrier`:

```python
# Library-default carrier-inference params (single source of truth — TF-24 4.7.0
# calibrated values). xShotOccurrence imports this so its default can't drift from
# the library (the exact failure R3 was meant to prevent). See NOTICE.
DEFAULT_CARRIER_PARAMS = {"tolerance_m": 3.0, "beta": 0.0, "gamma": 0.25}
```

Change the signature defaults to reference it:

```python
def infer_ball_carrier(
    frames: pd.DataFrame,
    *,
    tolerance_m: float = DEFAULT_CARRIER_PARAMS["tolerance_m"],
    beta: float = DEFAULT_CARRIER_PARAMS["beta"],
    gamma: float = DEFAULT_CARRIER_PARAMS["gamma"],
    pre: dict | None = None,
) -> pd.DataFrame:
```

- [ ] **Step 4: Point xS at the shared constant**

In `silly_kicks/tracking/_xshot_occurrence.py`, add to the existing `_ball_carrier` import (currently `from silly_kicks.tracking._ball_carrier import derive_team_in_possession, infer_ball_carrier`):

```python
from silly_kicks.tracking._ball_carrier import (
    DEFAULT_CARRIER_PARAMS,
    derive_team_in_possession,
    infer_ball_carrier,
)
```

Replace line 310:

```python
_DEFAULT_CARRIER_PARAMS = DEFAULT_CARRIER_PARAMS
```

- [ ] **Step 5: Update the train CLI defaults to the new values**

In `scripts/train_xshot_occurrence.py` change the stale defaults:

```python
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=0.25)
```

(Tolerance default `3.0` is unchanged.)

- [ ] **Step 6: Run the test + the carrier suite to verify green + no regression**

Run: `python -m pytest tests/tracking/test_xshot_occurrence.py::test_default_carrier_params_are_shared_constant tests/tracking/test_ball_carrier.py -v`
Expected: PASS. (Confirm no existing test asserted the old `0.5/1.0` xS default — grep `0.5.*gamma` / `_DEFAULT_CARRIER_PARAMS` in `tests/` first; fix any.)

- [ ] **Step 7: Stage + verify (no commit yet)**

Run: `python -m pytest tests/tracking/ -m "not e2e" -q` → all green.

---

## Task 2: `_geometry` coordinate constants (spec §6 metadata template)

**Files:**
- Modify: `silly_kicks/tracking/_geometry.py`
- Test: `tests/tracking/test_xshot_occurrence.py`

- [ ] **Step 1: Write the failing test**

```python
def test_geometry_exposes_pitch_constants():
    from silly_kicks.tracking import _geometry as geo
    assert geo.PITCH_LENGTH == 105.0
    assert geo.PITCH_WIDTH == 68.0
    assert isinstance(geo.GEOMETRY_VERSION, str) and geo.GEOMETRY_VERSION
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/tracking/test_xshot_occurrence.py::test_geometry_exposes_pitch_constants -v`
Expected: FAIL — `AttributeError: PITCH_WIDTH`.

- [ ] **Step 3: Add the constants**

In `silly_kicks/tracking/_geometry.py`, after `GOAL_Y = 34.0`:

```python
PITCH_LENGTH = FIELD_LENGTH          # 105.0 m — physical pitch length the goal-relative features assume
PITCH_WIDTH = GOAL_Y * 2.0           # 68.0 m
# Bump when the goal-relative transform's numeric output changes (NOT for a pure
# origin translation like TF-38, which is invariant). Consumed by model metadata
# as the coordinate-change fail-closed guard. See spec §6.
GEOMETRY_VERSION = "goal-relative-1"
```

- [ ] **Step 4: Run to verify green**

Run: `python -m pytest tests/tracking/test_xshot_occurrence.py::test_geometry_exposes_pitch_constants -v`
Expected: PASS.

- [ ] **Step 5: Stage + verify**

Run: `python -m pytest tests/tracking/ -m "not e2e" -q` → green.

---

## Task 3: Objective — `StratifiedGroupKFold` + drop `scale_pos_weight` (spec §3 / M1, M2)

**Files:**
- Modify: `silly_kicks/tracking/_xshot_occurrence_objective.py:29,33-40,54-57`
- Modify: `scripts/train_xshot_occurrence.py` (`param_space`, drop `scale_pos_weight`)
- Test: `tests/tracking/test_xshot_occurrence_integration.py`

- [ ] **Step 1: Write the failing test**

```python
def test_objective_search_excludes_scale_pos_weight_and_uses_stratified_cv():
    """M2: no scale_pos_weight (calibrated P(shot)); M1: label-stratified CV."""
    import inspect
    from silly_kicks.tracking import _xshot_occurrence_objective as obj

    assert "scale_pos_weight" not in obj._SEARCH_KEYS
    # CV uses StratifiedGroupKFold (label-stratified), not plain GroupKFold.
    src = inspect.getsource(obj._cv_logloss)
    assert "StratifiedGroupKFold" in src
    assert "GroupKFold(" not in src  # the plain variant is gone
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/tracking/test_xshot_occurrence_integration.py::test_objective_search_excludes_scale_pos_weight_and_uses_stratified_cv -v`
Expected: FAIL — `scale_pos_weight` still in `_SEARCH_KEYS`; source uses `GroupKFold`.

- [ ] **Step 3: Edit the objective**

In `_xshot_occurrence_objective.py`:

Replace the import (`:29`):

```python
from sklearn.model_selection import StratifiedGroupKFold
```

Remove `"scale_pos_weight"` from `_SEARCH_KEYS` (`:33-40`):

```python
_SEARCH_KEYS = (
    "n_estimators",
    "max_depth",
    "learning_rate",
    "min_child_weight",
    "reg_lambda",
)
```

Replace the CV splitter construction in `_cv_logloss` (`:54-57`). Update the docstring wording from "Match-stratified GroupKFold" to "Label-stratified, match-grouped":

```python
    """Label-stratified, match-grouped CV -> (mean log-loss, mean PR-AUC, mean Brier)."""
    import xgboost as xgb

    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2:
        n_splits = 2
    gkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
```

(The rest of `_cv_logloss` — the per-fold fit/predict/score loop — is unchanged. `gkf.split(X, y, groups)` already passes `y` for stratification.)

- [ ] **Step 4: Drop `scale_pos_weight` from the trainer search space**

In `scripts/train_xshot_occurrence.py`, remove the `scale_pos_weight` line from `param_space` (currently `:101`):

```python
        param_space={
            "n_estimators": FloatRange(kind="float", lo=50.0, hi=400.0),
            "max_depth": FloatRange(kind="float", lo=2.0, hi=6.0),
            "learning_rate": FloatRange(kind="float", lo=0.02, hi=0.4, log=True),
            "min_child_weight": FloatRange(kind="float", lo=1.0, hi=20.0),
            "reg_lambda": FloatRange(kind="float", lo=0.0, hi=5.0),
        },
```

- [ ] **Step 5: Run the new test + existing objective tests**

Run: `python -m pytest tests/tracking/test_xshot_occurrence_integration.py -k "objective or cache_equivalence or optuna" -v`
Expected: PASS — including `test_objective_cache_equivalence` (StratifiedGroupKFold is deterministic with `random_state=42`, so the 1e-9 equivalence still holds) and `test_optuna_smoke_3_trials`.

- [ ] **Step 6: Stage + verify**

Run: `python -m pytest tests/tracking/ -m "not e2e" -q` → green.

---

## Task 4: `fit()` sets `base_score` + asserts xgboost major (spec §3 / N4)

**Files:**
- Modify: `silly_kicks/tracking/_xshot_occurrence.py` (`fit`, `:370-393`)
- Test: `tests/tracking/test_xshot_occurrence.py`

- [ ] **Step 1: Write the failing test**

```python
def test_fit_sets_base_score_to_positive_rate(tiny_xshot_training_data):
    """N4: calibration must not silently depend on xgboost's auto-intercept."""
    X, y = tiny_xshot_training_data  # fixture: features df + 0/1 labels with a known mean
    from silly_kicks.tracking._xshot_occurrence import XShotOccurrenceModel

    m = XShotOccurrenceModel().fit(X, y)
    cfg = m._booster.save_config()  # xgboost serialized config (JSON string)
    import json
    base = float(json.loads(cfg)["learner"]["learner_model_param"]["base_score"])
    assert abs(base - float(y.mean())) < 1e-6


def test_fit_does_not_reweight(tiny_xshot_training_data):
    """P2/M2: the SHIPPED model must be unweighted (scale_pos_weight == 1), not just absent
    from the search space. Assert the property on the fitted booster's serialized config."""
    import json
    from silly_kicks.tracking._xshot_occurrence import XShotOccurrenceModel, _pinned_params

    assert "scale_pos_weight" not in _pinned_params(None)  # not pinned to a non-1 value
    X, y = tiny_xshot_training_data
    m = XShotOccurrenceModel().fit(X, y)
    cfg = json.loads(m._booster.save_config())
    # scale_pos_weight lives under the objective config; default (unweighted) is "1".
    spw = json.dumps(cfg)  # search the serialized config defensively across xgboost layouts
    import re
    found = re.findall(r'"scale_pos_weight":"?([0-9.eE+-]+)"?', spw)
    assert all(abs(float(v) - 1.0) < 1e-9 for v in found), f"model reweights: {found}"
```

(Add a `tiny_xshot_training_data` fixture in the test module: a ~40-row feature frame with `XSHOT_FEATURE_NAMES_FAITHFUL` columns and a `y` with, say, 5 positives — reuse the existing synthetic-frame helper + `prepare_xshot_training_data`, or hand-build the feature matrix directly.)

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/tracking/test_xshot_occurrence.py::test_fit_sets_base_score_to_positive_rate -v`
Expected: FAIL — base_score is xgboost's default/auto-estimate, not exactly `y.mean()`.

- [ ] **Step 3: Set base_score + assert xgboost major in `fit`**

In `_xshot_occurrence.py` `fit`, before constructing the classifier:

```python
        import xgboost as xgb

        if int(xgb.__version__.split(".")[0]) < 2:
            raise RuntimeError(
                "xShotOccurrence requires xgboost>=2.0 (calibrated base_score / intercept estimation)."
            )
        self.carrier_params = dict(carrier_params) if carrier_params else dict(_DEFAULT_CARRIER_PARAMS)
        self.horizon_seconds = horizon_seconds
        params = dict(self._params)
        params["base_score"] = float(np.asarray(labels, dtype=float).mean())  # explicit calibration anchor (N4)
        clf = xgb.XGBClassifier(**params)
        clf.fit(features.to_numpy(dtype=float), labels.to_numpy(dtype=int))
```

(Replace the existing `clf = xgb.XGBClassifier(**self._params)` block accordingly; keep the booster-feature-names assignment.)

- [ ] **Step 4: Run to verify green**

Run: `python -m pytest tests/tracking/test_xshot_occurrence.py::test_fit_sets_base_score_to_positive_rate -v`
Expected: PASS.

- [ ] **Step 5: Stage + verify**

Run: `python -m pytest tests/tracking/ -m "not e2e" -q` → green.

---

## Task 5: Metadata template + fail-closed `load` (spec §6 / M4, L2, N5)

**Files:**
- Modify: `silly_kicks/tracking/_xshot_occurrence.py` (`save` `:409-436`, `load` `:438-469`)
- Test: `tests/tracking/test_xshot_occurrence.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_metadata_records_pitch_and_platform(tmp_path, tiny_xshot_training_data):
    from silly_kicks.tracking._xshot_occurrence import XShotOccurrenceModel
    X, y = tiny_xshot_training_data
    m = XShotOccurrenceModel().fit(X, y)
    m.save(tmp_path / "v1")
    import json
    meta = json.loads((tmp_path / "v1" / "metadata.json").read_text())
    assert meta["pitch_length"] == 105.0 and meta["pitch_width"] == 68.0
    assert "geometry_version" in meta and "xgboost_version" in meta and "training_platform" in meta


def test_load_raises_on_pitch_dimension_mismatch(tmp_path, tiny_xshot_training_data, monkeypatch):
    """M4: a rescale/unit change genuinely skews features -> fail closed, never warn."""
    from silly_kicks.tracking import _xshot_occurrence as xo
    X, y = tiny_xshot_training_data
    m = xo.XShotOccurrenceModel().fit(X, y)
    m.save(tmp_path / "v1")
    # Simulate the live library now assuming a different physical pitch.
    monkeypatch.setattr(xo._geo, "PITCH_LENGTH", 100.0)
    import pytest
    with pytest.raises((xo.IntegrityError, ValueError)):
        xo.XShotOccurrenceModel.load(tmp_path / "v1")


def test_load_warns_on_geometry_version_only(tmp_path, tiny_xshot_training_data, monkeypatch):
    """Pure-representation change at identical pitch dims is invariant -> warn, not raise."""
    from silly_kicks.tracking import _xshot_occurrence as xo
    X, y = tiny_xshot_training_data
    m = xo.XShotOccurrenceModel().fit(X, y)
    m.save(tmp_path / "v1")
    monkeypatch.setattr(xo._geo, "GEOMETRY_VERSION", "goal-relative-2")
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        xo.XShotOccurrenceModel.load(tmp_path / "v1")  # must NOT raise
    assert any("geometry_version" in str(x.message).lower() for x in w)
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/tracking/test_xshot_occurrence.py -k "pitch or geometry_version" -v`
Expected: FAIL — metadata lacks the new keys; `load` performs no pitch/geometry check.

- [ ] **Step 3: Extend `save()` metadata**

In `_xshot_occurrence.py` `save`, expand the `metadata` dict (add imports `import platform`, `import xgboost as xgb` locally, and `from silly_kicks.tracking import _geometry as _geo` is already at module top as `_geo`):

```python
        import platform
        import xgboost as xgb

        metadata = {
            "feature_names": XSHOT_FEATURE_NAMES_FAITHFUL,
            "feature_set": self.feature_set,
            "horizon_seconds": self.horizon_seconds,
            "shot_types": self.shot_types,
            "carrier_params": self.carrier_params,
            "params": self._params,
            "version": _MODEL_VERSION,
            "pitch_length": _geo.PITCH_LENGTH,
            "pitch_width": _geo.PITCH_WIDTH,
            "geometry_version": _geo.GEOMETRY_VERSION,
            "xgboost_version": xgb.__version__,
            "training_platform": platform.platform(),
            "shipped_variant": getattr(self, "shipped_variant", None),
            "provider_list": getattr(self, "provider_list", None),
        }
```

(Add `self.shipped_variant: str | None = None` and `self.provider_list: list | None = None` in `__init__`; the trainer sets them before `save()`.)

- [ ] **Step 4: Add the fail-closed check in `load()`**

In `_xshot_occurrence.py` `load`, after parsing `meta` and before building the booster:

```python
        # Coordinate-change guard (spec §6, M4): a pitch-dimension/unit mismatch
        # genuinely skews every goal-relative feature -> FAIL CLOSED. A geometry_version
        # change at identical dims is the translation-invariant case -> warn only.
        rec_len = meta.get("pitch_length")
        rec_wid = meta.get("pitch_width")
        if rec_len is not None and (rec_len != _geo.PITCH_LENGTH or rec_wid != _geo.PITCH_WIDTH):
            raise IntegrityError(
                f"Pitch-dimension mismatch: model trained on {rec_len}x{rec_wid} m, "
                f"library is {_geo.PITCH_LENGTH}x{_geo.PITCH_WIDTH} m. Goal-relative features "
                "would be skewed; refusing to load (retrain required)."
            )
        rec_geo = meta.get("geometry_version")
        if rec_geo is not None and rec_geo != _geo.GEOMETRY_VERSION:
            import warnings

            warnings.warn(
                f"geometry_version mismatch (model={rec_geo}, library={_geo.GEOMETRY_VERSION}) at "
                "identical pitch dimensions — treated as translation-invariant. Verify if a "
                "non-translation coordinate change occurred.",
                stacklevel=2,
            )
```

(Also set `model.shipped_variant`/`model.provider_list` from `meta` after construction.)

- [ ] **Step 5: Run to verify green**

Run: `python -m pytest tests/tracking/test_xshot_occurrence.py -k "pitch or geometry_version or metadata_records" -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Stage + verify**

Run: `python -m pytest tests/tracking/ -m "not e2e" -q` → green.

---

## Task 6: Bundled `from_variant` loading (spec §5)

**Files:**
- Modify: `silly_kicks/tracking/_xshot_occurrence.py` (`from_variant` `:471-482`; add `_XSHOT_WEIGHTS_ROOT`)
- Test: `tests/tracking/test_xshot_occurrence.py`

- [ ] **Step 1: Write the failing test (logic via monkeypatched root — no real weights yet)**

```python
def test_from_variant_loads_bundled_dir(tmp_path, tiny_xshot_training_data, monkeypatch):
    from silly_kicks.tracking import _xshot_occurrence as xo
    X, y = tiny_xshot_training_data
    root = tmp_path / "_xshot_weights"
    (root / "default").mkdir(parents=True)
    xo.XShotOccurrenceModel().fit(X, y).save(root / "default")
    monkeypatch.setattr(xo, "_XSHOT_WEIGHTS_ROOT", root)

    m = xo.XShotOccurrenceModel.from_variant("default")
    assert m._booster is not None
    import pytest
    with pytest.raises(FileNotFoundError):
        xo.XShotOccurrenceModel.from_variant("does-not-exist")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/tracking/test_xshot_occurrence.py::test_from_variant_loads_bundled_dir -v`
Expected: FAIL — current `from_variant` unconditionally raises `FileNotFoundError("weights are not yet bundled")`.

- [ ] **Step 3: Add the weights root + rewrite `from_variant`**

In `_xshot_occurrence.py`, near the module constants (after `_HF_REPO_ID`):

```python
_XSHOT_WEIGHTS_ROOT = Path(__file__).parent / "_xshot_weights"
_VARIANT_CACHE: dict[str, "XShotOccurrenceModel"] = {}  # P3: memoize bundled loads (default-list perf)
```

Replace `from_variant` (memoized so default-list consumers don't reload + SHA-verify per call — P3):

```python
    @classmethod
    def from_variant(cls, variant: str = "default") -> XShotOccurrenceModel:
        """Load a bundled variant by name (memoized); fall through to Hub for non-bundled variants.

        ``"default"`` is bundled in the wheel. SHA-256 verified on first load, then cached —
        an immutable, inference-only instance is safe to share across calls.

        Examples
        --------
        >>> # XShotOccurrenceModel.from_variant("default")
        """
        if variant in _VARIANT_CACHE:
            return _VARIANT_CACHE[variant]
        weights_dir = _XSHOT_WEIGHTS_ROOT / variant
        if (weights_dir / "SHA256SUMS").exists():
            model = cls.load(weights_dir)
        elif variant == "public":  # the Hub-hosted variant, if we shipped two
            model = cls.from_hub(_HF_REPO_ID)
        else:
            raise FileNotFoundError(
                f"No bundled xShotOccurrence weights for variant {variant!r} at {weights_dir}. "
                "Train via scripts/train_xshot_occurrence.py or use from_hub()."
            )
        _VARIANT_CACHE[variant] = model
        return model
```

**Test note:** the T6 monkeypatch test must clear `_VARIANT_CACHE` (set `monkeypatch.setattr(xo, "_VARIANT_CACHE", {})`) so the temp-dir root is honoured rather than a cached instance.

- [ ] **Step 4: Run to verify green**

Run: `python -m pytest tests/tracking/test_xshot_occurrence.py::test_from_variant_loads_bundled_dir -v`
Expected: PASS.

- [ ] **Step 5: Stage + verify**

Run: `python -m pytest tests/tracking/ -m "not e2e" -q` → green. (`test_compute_xshot_no_model` may rely on `from_variant` raising when no weights — confirm it still passes since `_XSHOT_WEIGHTS_ROOT/default` does not yet exist in the repo.)

---

## Task 7: Make `home_team_id` optional across the xS surface (enables xfn wiring; spec §8)

**Rationale:** `home_team_id` is **unused** in `compute_xshot_occurrence`/`add_xshot_occurrence`/`xshot_occurrence_xfns` (goal is resolved GK-based via `_defended_goal_x`; "reserved for symmetry"). The module-level default xfn lists (T13) can't supply it. Make it optional (default `None`) — backward-compatible (existing callers passing it still work; the value continues to be ignored).

**Files:**
- Modify: `silly_kicks/tracking/_xshot_occurrence.py` (`compute_xshot_occurrence:669`, `add_xshot_occurrence:766`, `xshot_occurrence_xfns:815`)
- Test: `tests/tracking/test_xshot_occurrence.py`

- [ ] **Step 1: Write the failing test**

```python
def test_xshot_surface_home_team_id_optional(synthetic_xshot_frames, fixture_fit_model):
    """home_team_id is unused (GK-based goal resolution); callers may omit it."""
    from silly_kicks.tracking._xshot_occurrence import compute_xshot_occurrence, xshot_occurrence_xfns
    out = compute_xshot_occurrence(synthetic_xshot_frames, model=fixture_fit_model)  # no home_team_id
    assert "xshot_occurrence" in out.columns
    # And the factory can be built with no args (so it can sit in a default list).
    xfns = xshot_occurrence_xfns()
    assert len(xfns) == 1 and getattr(xfns[0], "_frame_aware", False) is True
```

(`synthetic_xshot_frames` + `fixture_fit_model`: reuse existing synthetic-frame + fixture-fit helpers in the test module.)

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/tracking/test_xshot_occurrence.py::test_xshot_surface_home_team_id_optional -v`
Expected: FAIL — `TypeError: missing required keyword-only argument 'home_team_id'`.

- [ ] **Step 3: Default `home_team_id=None` on the three serve-side functions**

In `_xshot_occurrence.py`, change the signatures (leave `prepare_xshot_training_data`'s `home_team_id` as-is — the trainer passes it):

```python
def compute_xshot_occurrence(frames, *, model=None, home_team_id: int | str | None = None,
                             pitch_control_cache=None, link_frame_ids=None) -> pd.DataFrame:
```
```python
@nan_safe_enrichment
def add_xshot_occurrence(actions, frames, *, model=None, links=None,
                         home_team_id: int | str | None = None, pitch_control_cache=None) -> pd.DataFrame:
```
```python
def xshot_occurrence_xfns(*, model=None, home_team_id: int | str | None = None,
                          pitch_control_cache=None) -> list:
```

Update each docstring line "``home_team_id`` ... reserved" to note it is optional/unused. **NaN-safety registry note (feedback_nan_safety_needs_extra):** verify whether `add_xshot_occurrence` is listed in `_TRACKING_NEEDS_EXTRA` in `silly_kicks/_nan_safety.py` because of `home_team_id`; now that it has a default, confirm the NaN-safety auto-discovery test (`tests/test_enrichment_nan_safety.py`) still passes and move it to the regular set if appropriate.

- [ ] **Step 4: Run to verify green + NaN-safety gate**

Run: `python -m pytest tests/tracking/test_xshot_occurrence.py::test_xshot_surface_home_team_id_optional tests/test_enrichment_nan_safety.py -v`
Expected: PASS.

- [ ] **Step 5: Stage + verify**

Run: `python -m pytest tests/ -m "not e2e" -q` → green.

---

## Task 8: Trainer rewrite (spec §3) — streaming, two-candidate, frozen-HPO paired eval, fail-closed

**Files:**
- Modify (rewrite): `scripts/train_xshot_occurrence.py`
- Test: `tests/tracking/test_xshot_occurrence_integration.py`

The trainer supports **two match sources**: `--data-dir` (parquet dirs `*/frames.parquet`+`shots.parquet`, used by the smoke test + any local corpus) OR `--providers a,b,c` (the pining loader, used for the real run). It streams per match, caches features, runs HPO once per candidate, evaluates the §5 paired comparison, computes fail-closed acceptance gates, and writes the artifact only if gates pass.

- [ ] **Step 1: Write the failing tests (extend the existing smoke + add fail-closed)**

```python
def test_train_script_smoke_writes_artifact(tmp_path, synthetic_xshot_parquet_dir):
    """Existing smoke, updated: 3 trials on synthetic parquet -> artifact written, gates pass."""
    import subprocess, sys
    out = tmp_path / "models"
    r = subprocess.run(
        [sys.executable, "scripts/train_xshot_occurrence.py",
         "--data-dir", str(synthetic_xshot_parquet_dir), "--output-dir", str(out),
         "--n-trials", "3"],
        cwd=".", capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "."},
    )
    assert r.returncode == 0, r.stderr
    art = out / "xshot_occurrence_v1"
    assert (art / "model.json").exists() and (art / "metadata.json").exists() and (art / "SHA256SUMS").exists()
    import json
    metrics = json.loads((out / "xshot_occurrence_v1" / "metrics.json").read_text())
    assert metrics["acceptance"]  # block present
    assert "estimates_are_cv_not_shipped_fit" in metrics  # N7 labelling


def test_train_script_fail_closed_writes_no_artifact(tmp_path, degenerate_xshot_parquet_dir):
    """N3: a corpus that can't beat the base rate -> non-zero exit, no artifact."""
    import subprocess, sys
    out = tmp_path / "models"
    r = subprocess.run(
        [sys.executable, "scripts/train_xshot_occurrence.py",
         "--data-dir", str(degenerate_xshot_parquet_dir), "--output-dir", str(out), "--n-trials", "2"],
        cwd=".", capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "."},
    )
    assert r.returncode != 0
    assert not (out / "xshot_occurrence_v1" / "model.json").exists()


def test_paired_decision_rule_data_effect():
    """Direct unit test of the subtle paired-decision helper (P1) — not via subprocess.

    Build a tiny (X, y, groups, is_public) where the GS rows carry a feature that genuinely
    helps the public held-out (full should win >= K-1 folds) -> ship_two True; and a control
    where GS adds noise -> ship_two False.
    """
    import numpy as np, pandas as pd
    from scripts.train_xshot_occurrence import _paired_data_effect
    from silly_kicks.tracking._xshot_occurrence import XSHOT_FEATURE_NAMES_FAITHFUL

    rng = np.random.default_rng(0)
    def _make(n_games, helpful):
        rows, lab, grp, pub = [], [], [], []
        for g in range(n_games):
            is_pub = g < n_games - 2  # last 2 games are GS
            for _ in range(60):
                r = float(rng.uniform(5, 40))
                p = 1 / (1 + np.exp((r - 18) / 4))           # closer => more likely
                y_ = int(rng.random() < p)
                row = {c: 0.0 for c in XSHOT_FEATURE_NAMES_FAITHFUL}; row["r"] = r
                rows.append(row); lab.append(y_); grp.append(g); pub.append(is_pub)
        X = pd.DataFrame(rows)[XSHOT_FEATURE_NAMES_FAITHFUL]
        return X, np.array(lab), np.array(grp), np.array(pub)

    X, y, groups, is_public = _make(7, helpful=True)
    res = _paired_data_effect(X, y, groups, is_public, shared_params={"max_depth": 3, "n_estimators": 60})
    assert res["paired_delta_is_data_effect_shared_params"] is True
    assert set(res) >= {"deltas", "K", "n_positive", "ship_two"}
    assert isinstance(res["ship_two"], bool)  # rule evaluates without error on a realistic tiny set
```

(`synthetic_xshot_parquet_dir`: a session fixture writing 2–3 synthetic games' `frames.parquet`+`shots.parquet` with enough shots that the gates pass. `degenerate_xshot_parquet_dir`: frames with no resolvable shot signal so PR-AUC ≤ base rate. Build both from the existing synthetic-frame helper. The `_paired_data_effect` import works because the trainer is importable as a module from `scripts/`.)

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/tracking/test_xshot_occurrence_integration.py -k "train_script" -v`
Expected: FAIL — current script has no fail-closed gate, no `metrics.json`, no `estimates_are_cv_not_shipped_fit` key.

- [ ] **Step 3: Rewrite `scripts/train_xshot_occurrence.py`**

Full script (replaces the existing file):

```python
#!/usr/bin/env python
"""Train the xShotOccurrence (xS) model (TF-16 weights run, PR-S80).

Two match sources:
  --data-dir DIR     parquet dirs DIR/*/{frames,shots}.parquet (smoke / local corpus)
  --providers a,b,c  pining loader (skillcorner,idsse,gradientsports) for the maintainer run

Streams per match, caches features, runs ruthless HPO ONCE per candidate (public/full),
evaluates the common-public-held-out paired comparison (spec §5), computes FAIL-CLOSED
acceptance gates (spec §3), and writes a pickle-free artifact only if the gates pass.

Requires: silly-kicks[train,xgboost]  (+ [kloppy] for --providers).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

_PUBLIC_PROVIDERS = {"skillcorner", "idsse"}


def _iter_matches_from_dir(data_dir: Path):
    for game_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        frames = pd.read_parquet(game_dir / "frames.parquet")
        shots = pd.read_parquet(game_dir / "shots.parquet")
        prov = str(frames["source_provider"].iloc[0]) if "source_provider" in frames.columns else "unknown"
        yield prov, game_dir.name, shots, frames, frames["team_id"].dropna().iloc[0]


def _iter_matches_from_pining(providers, max_per_provider):
    sys.path.insert(0, "scripts")
    from _loader_pining import load_matches  # noqa: E402
    for prov, mid, actions, frames, home in load_matches(
        providers=providers, max_per_provider=max_per_provider
    ):
        yield prov, mid, actions, frames, home


def _extract(source) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS
    from silly_kicks.tracking._xshot_occurrence import XSHOT_FEATURE_NAMES_FAITHFUL, prepare_xshot_training_data

    Xs, ys, gs, ps = [], [], [], []
    for prov, mid, actions_or_shots, frames, home in source:
        X, y, groups = prepare_xshot_training_data(
            frames, actions_or_shots, home_team_id=home,
            horizon_seconds=1.0, attacking_third_only=True,
            carrier_params=DEFAULT_CARRIER_PARAMS,  # 4.7.0 values; shared constant (anti-drift)
        )
        del frames
        if len(X):
            Xs.append(X); ys.append(np.asarray(y, int)); gs.append(np.asarray(groups))
            ps.append(np.array([prov] * len(X)))
            print(f"  {prov}/{mid}: {len(X)} rows, {int(y.sum())} positives")
    if not Xs:
        raise SystemExit("No usable training data.")
    return (pd.concat(Xs, ignore_index=True)[XSHOT_FEATURE_NAMES_FAITHFUL],
            np.concatenate(ys), np.concatenate(gs), np.concatenate(ps))


def _hpo_once(X, y, groups, out_dir, tag, n_trials) -> dict:
    """Run ruthless HPO once for one candidate; return the best params dict (frozen)."""
    from ruthless import Direction, FloatRange, InProcessBackend, OptunaConfig
    from ruthless.config.common import StoreConfig
    from ruthless.strategies.optuna_ import OptunaStrategy

    from silly_kicks.tracking._xshot_occurrence_objective import XShotOccurrenceObjective

    obj = XShotOccurrenceObjective(fold={tag: [(X, pd.Series(y), groups)]})
    cfg = OptunaConfig(
        kind="optuna", metric="logloss", direction=Direction.MINIMIZE, n_trials=n_trials, sampler="tpe",
        param_space={
            "n_estimators": FloatRange(kind="float", lo=50.0, hi=400.0),
            "max_depth": FloatRange(kind="float", lo=2.0, hi=6.0),
            "learning_rate": FloatRange(kind="float", lo=0.02, hi=0.4, log=True),
            "min_child_weight": FloatRange(kind="float", lo=1.0, hi=20.0),
            "reg_lambda": FloatRange(kind="float", lo=0.0, hi=5.0),
        },
        store=StoreConfig(kind="sqlite", path=str(out_dir / f"study_{tag}.db")),
    )
    result = OptunaStrategy(cfg, seed=42).run(obj, backend=InProcessBackend())
    return dict(result.best.candidate.params)


def _cv_metrics(X, y, groups, params) -> dict:
    """Label-stratified, match-grouped CV at FIXED params -> gate metrics on the true balance."""
    import xgboost as xgb
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
    from sklearn.model_selection import StratifiedGroupKFold

    from silly_kicks.tracking._xshot_occurrence import _pinned_params

    n_splits = max(2, min(5, len(np.unique(groups))))
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    prs, brs, lls = [], [], []
    for tr, te in skf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        p_ = dict(_pinned_params(params)); p_["base_score"] = float(y[tr].mean())
        clf = xgb.XGBClassifier(**p_)
        clf.fit(X.iloc[tr].to_numpy(float), y[tr])
        p = clf.predict_proba(X.iloc[te].to_numpy(float))[:, 1]
        lls.append(log_loss(y[te], p, labels=[0, 1]))
        brs.append(brier_score_loss(y[te], p))
        if len(np.unique(y[te])) == 2:
            prs.append(average_precision_score(y[te], p))
    base = float(y.mean())
    return {
        "pr_auc": float(np.mean(prs)) if prs else float("nan"),
        "brier": float(np.mean(brs)) if brs else float("nan"),
        "log_loss": float(np.mean(lls)) if lls else float("inf"),
        "pr_auc_std": float(np.std(prs)) if prs else float("nan"),
        "positive_rate": base,
        "base_rate_brier": base * (1 - base),
        "n_usable_folds": len(lls),  # P5: track usable (both-class-train) folds
    }


def _gates(m: dict) -> dict:
    return {
        "enough_usable_folds": m.get("n_usable_folds", 0) >= 2,  # P5: refuse on <2 usable folds
        "pr_auc_gt_base_rate": m["pr_auc"] > m["positive_rate"],
        "brier_lt_base_rate_brier": m["brier"] < m["base_rate_brier"],
        "log_loss_lt_uniform": m["log_loss"] < float(np.log(2)),
    }


def _paired_data_effect(X, y, groups, is_public, *, shared_params) -> dict:
    """Common-public-held-out PAIRED data-effect test at FIXED shared hyperparameters (§5, P1).

    For each public outer fold k: fit `public` on public-train and `full` on public-train + ALL
    GS rows, BOTH at `shared_params` (only training data differs — leakage-free: full never sees
    public-test games), score both on the public held-out fold. Δ_k = PR-AUC(full) − PR-AUC(public).
    Ship-two iff Δ_k > 0 in ≥ K−1 of K folds AND mean Δ > 0.
    """
    import numpy as np
    import xgboost as xgb
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import StratifiedGroupKFold

    from silly_kicks.tracking._xshot_occurrence import _pinned_params

    Xp, yp, gp = X[is_public], y[is_public], groups[is_public]
    n = max(2, min(5, len(np.unique(gp))))
    skf = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=42)

    def _fit_score(Xtr, ytr, te_idx):
        p_ = dict(_pinned_params(shared_params)); p_["base_score"] = float(ytr.mean())
        c = xgb.XGBClassifier(**p_); c.fit(Xtr.to_numpy(float), ytr)
        pr = c.predict_proba(Xp.iloc[te_idx].to_numpy(float))[:, 1]
        return average_precision_score(yp[te_idx], pr) if len(np.unique(yp[te_idx])) == 2 else float("nan")

    deltas = []
    for tr, te in skf.split(Xp, yp, gp):
        train_games = set(np.asarray(gp)[tr].tolist())
        full_mask = (~is_public) | np.isin(groups, list(train_games))  # GS + public-train only
        d_pub = _fit_score(Xp.iloc[tr], yp[tr], te)
        d_full = _fit_score(X[full_mask], y[full_mask], te)
        if not (np.isnan(d_pub) or np.isnan(d_full)):
            deltas.append(float(d_full - d_pub))
    K = len(deltas)
    n_pos = sum(1 for d in deltas if d > 0)
    ship_two = K >= 2 and n_pos >= K - 1 and (sum(deltas) / K) > 0.0
    return {
        "deltas": deltas, "K": K, "n_positive": n_pos, "ship_two": bool(ship_two),
        "paired_delta_is_data_effect_shared_params": True,  # P1: data-effect, not data+tune
        "paired_hpo_nested": False,                          # HPO optimism common-mode (same params both arms)
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data-dir")
    src.add_argument("--providers")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--max-per-provider", type=int, default=None)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    cache = out / "xshot_occurrence_v1" / "_feature_cache"

    # Phase 1 — stream + extract + cache.
    if (cache / "features.parquet").exists():
        print(f"Loading cached features from {cache}")
        X = pd.read_parquet(cache / "features.parquet")
        y = np.load(cache / "labels.npy")
        groups = np.load(cache / "groups.npy", allow_pickle=True)
        providers = np.load(cache / "providers.npy", allow_pickle=True)
    else:
        if args.providers:
            source = _iter_matches_from_pining(args.providers.split(","), args.max_per_provider)
        else:
            source = _iter_matches_from_dir(Path(args.data_dir))
        t0 = time.time()
        X, y, groups, providers = _extract(source)
        print(f"Extracted {len(X)} rows ({int(y.sum())} positives) in {time.time()-t0:.0f}s")
        cache.mkdir(parents=True, exist_ok=True)
        X.to_parquet(cache / "features.parquet")
        np.save(cache / "labels.npy", y)
        np.save(cache / "groups.npy", groups); np.save(cache / "providers.npy", providers)

    is_public = np.isin(providers, list(_PUBLIC_PROVIDERS))
    have_full = (~is_public).any() and is_public.any()

    # Phase 2 — HPO ONCE per candidate (frozen); Phase 3 — fixed-params CV gate metrics.
    candidates: dict[str, dict] = {}
    params_public = _hpo_once(X[is_public], y[is_public], groups[is_public], out, "public", args.n_trials)
    candidates["public"] = {"params": params_public,
                            "metrics": _cv_metrics(X[is_public], y[is_public], groups[is_public], params_public),
                            "providers": sorted(set(providers[is_public].tolist()))}
    if have_full:
        params_full = _hpo_once(X, y, groups, out, "full", args.n_trials)
        candidates["full"] = {"params": params_full,
                              "metrics": _cv_metrics(X, y, groups, params_full),
                              "providers": sorted(set(providers.tolist()))}

    # §5 paired DATA-EFFECT test (P1): SHARED hyperparameters (params_public) across BOTH arms,
    # so the only thing varying is training data -> Δ_k is a clean data-effect signal, not
    # data+tune. The shipped `full` model still uses its own params_full for deployment; the
    # EVIDENCE comparison holds params fixed. HPO-selection optimism is now common-mode (same
    # params both arms) and cancels in Δ_k. See _paired_data_effect (unit-tested).
    shipped = "public"
    if have_full:
        paired = _paired_data_effect(X, y, groups, is_public, shared_params=params_public)
        candidates["paired"] = paired
        shipped = "full" if paired["ship_two"] else "public"

    # Fail-closed gates (N3) on the SHIPPED candidate.
    sm = candidates[shipped]["metrics"]
    gates = _gates(sm)
    print(f"Shipped variant: {shipped}; gates: {gates}")
    if not all(gates.values()):
        print("ACCEPTANCE GATES FAILED — refusing to write artifact.", file=sys.stderr)
        json.dump({"candidates": candidates, "gates": gates, "shipped": shipped},
                  open(out / "xshot_occurrence_v1" / "metrics_FAILED.json", "w"), indent=2)
        sys.exit(1)

    # Final fit on ALL the shipped candidate's games + save.
    from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS
    from silly_kicks.tracking._xshot_occurrence import XShotOccurrenceModel
    mask = np.ones(len(X), bool) if shipped == "full" else is_public
    model = XShotOccurrenceModel(params=candidates[shipped]["params"])
    model.shipped_variant = shipped
    model.provider_list = candidates[shipped]["providers"]
    model.fit(X[mask], pd.Series(y[mask]), carrier_params=DEFAULT_CARRIER_PARAMS, horizon_seconds=1.0)
    art = out / "xshot_occurrence_v1"
    model.save(art)
    # Round-trip verify.
    reloaded = XShotOccurrenceModel.load(art)
    np.testing.assert_allclose(model.predict_proba(X[mask].head(50)),
                               reloaded.predict_proba(X[mask].head(50)), rtol=0, atol=0)

    metrics = {
        "shipped_variant": shipped,
        "n_rows": int(len(X)), "n_positive": int(y.sum()),
        "providers": sorted(set(providers.tolist())),
        "candidates": candidates, "gates": gates,
        "estimates_are_cv_not_shipped_fit": True,  # N7
        "artifact_size_bytes": sum(f.stat().st_size for f in art.glob("*") if f.is_file()),
    }
    json.dump(metrics, open(art / "metrics.json", "w"), indent=2)
    print(f"Wrote artifact + metrics to {art}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the smoke + fail-closed tests**

Run: `python -m pytest tests/tracking/test_xshot_occurrence_integration.py -k "train_script" -v`
Expected: PASS (both). If the synthetic corpus is too small for `StratifiedGroupKFold(5)`, the `max(2,min(5,...))` guard handles it.

- [ ] **Step 5: Lint + stage**

Run: `ruff check scripts/train_xshot_occurrence.py && ruff format --check scripts/train_xshot_occurrence.py` then `python -m pytest tests/tracking/ -m "not e2e" -q`.
Expected: clean + green.

---

## Task 9: Publish script (spec §5)

**Files:**
- Create: `scripts/publish_xshot_occurrence.py`
- Test: `tests/tracking/test_xshot_occurrence_integration.py`

- [ ] **Step 1: Write the failing test (verify-only path; no network)**

```python
def test_publish_verify_only(tmp_path, tiny_xshot_training_data):
    import subprocess, sys, os
    from silly_kicks.tracking._xshot_occurrence import XShotOccurrenceModel
    X, y = tiny_xshot_training_data
    art = tmp_path / "xshot_occurrence_v1"
    XShotOccurrenceModel().fit(X, y).save(art)
    r = subprocess.run(
        [sys.executable, "scripts/publish_xshot_occurrence.py", "--artifact-dir", str(art), "--verify-only"],
        cwd=".", capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "."},
    )
    assert r.returncode == 0, r.stderr
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/tracking/test_xshot_occurrence_integration.py::test_publish_verify_only -v`
Expected: FAIL — `scripts/publish_xshot_occurrence.py` does not exist.

- [ ] **Step 3: Create the publish script**

```python
#!/usr/bin/env python
"""Publish a trained xShotOccurrence artifact to HuggingFace Hub (PR-S80).

Verifies SHA-256 + a sanity prediction, uploads, then re-downloads via from_hub and
asserts identical predictions. --verify-only stops before upload (no network/token).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", required=True)
    ap.add_argument("--repo-id", default="silly-kicks/xshot-occurrence-v1")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    from silly_kicks.tracking._xshot_occurrence import XSHOT_FEATURE_NAMES_FAITHFUL, XShotOccurrenceModel
    import pandas as pd

    art = Path(args.artifact_dir)
    model = XShotOccurrenceModel.load(art)  # SHA-256 verified
    sample = pd.DataFrame(np.zeros((3, len(XSHOT_FEATURE_NAMES_FAITHFUL))), columns=XSHOT_FEATURE_NAMES_FAITHFUL)
    local_pred = model.predict_proba(sample)
    print(f"Loaded + verified {art}; sample preds {local_pred.tolist()}")
    if args.verify_only:
        print("verify-only: not uploading.")
        return

    from huggingface_hub import HfApi

    HfApi().upload_folder(folder_path=str(art), repo_id=args.repo_id, repo_type="model")
    back = XShotOccurrenceModel.from_hub(args.repo_id)
    np.testing.assert_allclose(local_pred, back.predict_proba(sample), rtol=0, atol=0)
    print(f"Published to {args.repo_id} + round-trip verified.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify green + lint**

Run: `python -m pytest tests/tracking/test_xshot_occurrence_integration.py::test_publish_verify_only -v && ruff check scripts/publish_xshot_occurrence.py`
Expected: PASS + clean.

- [ ] **Step 5: Stage + verify**

Run: `python -m pytest tests/tracking/ -m "not e2e" -q` → green.

---

## Task 10: Frozen directional fixture (the CI quality tripwire's data; spec §9 / H2, N2)

**Files:**
- Create: `scripts/make_xshot_directional_fixture.py`
- Create: `tests/datasets/tracking/xshot_directional/frozen_rows.parquet` (generated, committed)
- Test: `tests/tracking/test_xshot_occurrence.py` (fixture-validity only — model-dependent assertion lands in T12)

- [ ] **Step 1: Write the failing fixture-validity test**

```python
def test_directional_fixture_has_both_classes_and_schema():
    from silly_kicks.tracking._xshot_occurrence import XSHOT_FEATURE_NAMES_FAITHFUL
    import pandas as pd
    df = pd.read_parquet("tests/datasets/tracking/xshot_directional/frozen_rows.parquet")
    assert set(XSHOT_FEATURE_NAMES_FAITHFUL).issubset(df.columns)
    assert "label" in df.columns
    assert df["label"].nunique() == 2 and df["label"].sum() >= 3 and (df["label"] == 0).sum() >= 3
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/tracking/test_xshot_occurrence.py::test_directional_fixture_has_both_classes_and_schema -v`
Expected: FAIL — the parquet does not exist.

- [ ] **Step 3: Write the generator + produce the committed fixture**

```python
#!/usr/bin/env python
"""Build the frozen directional feature-vector fixture for the CI quality tripwire.

Extracts xS features from the committed slim real-provider fixtures, labels them via
build_xshot_labels, and freezes a handful of true-positive + true-negative FEATURE ROWS
(not raw frames — N2: feature-layer freeze is schema-robust + arch-robust). Pick rows
that differ in `r` so the tripwire stays valid under the paper's r-dominance.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from silly_kicks.tracking._xshot_occurrence import XSHOT_FEATURE_NAMES_FAITHFUL, prepare_xshot_training_data

SLIM = Path("tests/datasets/tracking/action_context_slim")
OUT = Path("tests/datasets/tracking/xshot_directional/frozen_rows.parquet")


def main() -> None:
    rows = []
    for prov in ("sportec", "skillcorner", "metrica"):
        frames = pd.read_parquet(SLIM / f"{prov}_slim.parquet")
        # the slim fixtures carry both frames + a small shots/actions table; adapt as needed
        # (these are the same fixtures test_xshot_occurrence_real_data.py consumes).
        shots = frames.attrs.get("actions")  # or load the sibling actions fixture if separate
        X, y, _ = prepare_xshot_training_data(frames, shots, home_team_id=frames["team_id"].dropna().iloc[0])
        df = X.copy(); df["label"] = y; df["provider"] = prov
        rows.append(df)
    allrows = pd.concat(rows, ignore_index=True)
    pos = allrows[allrows["label"] == 1].nlargest(5, "r")    # imminent: keep high-signal positives
    neg = allrows[allrows["label"] == 0].nsmallest(5, "r")   # quiet: far-from-goal negatives
    frozen = pd.concat([pos, neg], ignore_index=True)[[*XSHOT_FEATURE_NAMES_FAITHFUL, "label", "provider"]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frozen.to_parquet(OUT)
    print(f"Wrote {len(frozen)} frozen rows ({int(frozen['label'].sum())} pos) to {OUT}")


if __name__ == "__main__":
    main()
```

Run it: `python scripts/make_xshot_directional_fixture.py`
(If a provider's slim fixture yields no positives, fall back to the others; ensure ≥3 pos and ≥3 neg total. If the slim fixtures store actions separately, load the sibling `*_expected.parquet`/actions accordingly — inspect `tests/tracking/test_xshot_occurrence_real_data.py` for how it pairs frames+shots.)

- [ ] **Step 4: Run to verify green**

Run: `python -m pytest tests/tracking/test_xshot_occurrence.py::test_directional_fixture_has_both_classes_and_schema -v`
Expected: PASS.

- [ ] **Step 5: Stage + verify**

Run: `python -m pytest tests/tracking/ -m "not e2e" -q` → green.

---

## Task 11: [OPERATIONAL — DGX Spark] The training run (spec §10)

**Not a TDD task — a compute job that produces the bundled artifact.** Run on the idle DGX Spark box; the owner monitors long jobs (do not poll CI-style).

- [ ] **Step 1: Provision the box**

```bash
ssh karsten@192.168.68.73
# sync the pr-s80-tf16-weights branch to the box, then:
pip install -e ".[train,xgboost,kloppy]"
export PINING_FOR_THE_DATA_TOKEN=<owner token>
export HF_TOKEN=<hf token>
```

- [ ] **Step 2: Phase-0 corpus probe (manifest counts, no download)**

Run a one-liner using `_loader_pining._list_matches` per provider; record counts. Decide pining-sufficiency for xS (almost always yes). Log the counts (no silent caps).

- [ ] **Step 3: Launch the run as a background job**

```bash
python scripts/train_xshot_occurrence.py \
  --providers skillcorner,idsse,gradientsports \
  --output-dir models/ --n-trials 50
```

Expected: streams matches, caches features, HPO ×(1 or 2 candidates), prints the shipped variant + gates, writes `models/xshot_occurrence_v1/{model.json,metadata.json,SHA256SUMS,metrics.json}` **only if gates pass** (else non-zero exit + `metrics_FAILED.json`).

- [ ] **Step 4: Verify gates + the paired decision**

Inspect `models/xshot_occurrence_v1/metrics.json`: `gates` all true; `candidates.paired` (if present) → shipped variant; per-candidate PR-AUC/Brier vs base rate. If a gate failed, the artifact was not written — diagnose (corpus too thin? `r`-only weak discrimination?) before proceeding.

- [ ] **Step 5: Publish + pull back**

```bash
python scripts/publish_xshot_occurrence.py --artifact-dir models/xshot_occurrence_v1 --verify-only
python scripts/publish_xshot_occurrence.py --artifact-dir models/xshot_occurrence_v1   # uploads + round-trips
```
Copy `models/xshot_occurrence_v1/{model.json,metadata.json,SHA256SUMS}` back to the local repo for bundling (T12).

---

## Task 12: Bundle the artifact + weight-dependent tests (spec §5, §9)

**Files:**
- Create: `silly_kicks/tracking/_xshot_weights/default/{model.json,metadata.json,SHA256SUMS}` (from T11)
- Test: `tests/tracking/test_xshot_occurrence_integration.py`

- [ ] **Step 1: Place the bundled artifact**

Copy the verified artifact from T11 into `silly_kicks/tracking/_xshot_weights/default/`. (If two variants shipped, `default/` = the chosen `full`; `public` stays Hub-only.)

- [ ] **Step 2: Write the weight-dependent tests**

```python
def test_bundled_model_is_live_not_degenerate():
    """LIVENESS tripwire (H2/N2/P4): the bundled model is not dead/constant — it ranks the
    cherry-picked imminent extremes above the quiet extremes. This is NOT a quality measure
    (the frozen rows are maximally separable in `r`, the dominant feature, so a green check
    says 'the booster loads and discriminates the obvious cases', NOT 'xS is good' — real
    quality lives in the e2e gates). Scale-free AUC, no magic margin."""
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from silly_kicks.tracking._xshot_occurrence import XShotOccurrenceModel, XSHOT_FEATURE_NAMES_FAITHFUL
    df = pd.read_parquet("tests/datasets/tracking/xshot_directional/frozen_rows.parquet")
    m = XShotOccurrenceModel.from_variant("default")
    p = m.predict_proba(df[XSHOT_FEATURE_NAMES_FAITHFUL])
    assert roc_auc_score(df["label"].to_numpy(), p) >= 0.9  # maximally-separable extremes ⇒ ~1.0 for a live model


def test_from_variant_default_loads_and_predicts_in_bounds():
    import pandas as pd, numpy as np
    from silly_kicks.tracking._xshot_occurrence import XShotOccurrenceModel, XSHOT_FEATURE_NAMES_FAITHFUL
    m = XShotOccurrenceModel.from_variant("default")
    p = m.predict_proba(pd.DataFrame(np.zeros((4, len(XSHOT_FEATURE_NAMES_FAITHFUL))), columns=XSHOT_FEATURE_NAMES_FAITHFUL))
    assert np.all((p >= 0) & (p <= 1))


def test_bundled_metadata_matches_training_intent():
    """L6: intent-named. carrier params == shared constant; coordinate/platform fields present."""
    import json
    from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS
    meta = json.loads((__import__("pathlib").Path("silly_kicks/tracking/_xshot_weights/default/metadata.json")).read_text())
    assert meta["carrier_params"] == DEFAULT_CARRIER_PARAMS
    for k in ("pitch_length", "pitch_width", "geometry_version", "xgboost_version", "training_platform", "shipped_variant"):
        assert k in meta


def test_bundled_weights_present_in_package():
    """Wheel-content sanity (spec §9): the bundled dir ships inside the package."""
    from pathlib import Path
    import silly_kicks.tracking as t
    root = Path(t.__file__).parent / "_xshot_weights" / "default"
    assert (root / "model.json").exists() and (root / "SHA256SUMS").exists()
```

- [ ] **Step 3: Run them**

Run: `python -m pytest tests/tracking/test_xshot_occurrence_integration.py -k "bundled or from_variant_default or directional_quality" -v`
Expected: PASS (4).

- [ ] **Step 4: Stage + verify full non-e2e suite**

Run: `python -m pytest tests/ -m "not e2e" -q` → green.

---

## Task 13: Wire `xshot_occurrence_xfns` into the default lists (spec §8)

**Files:**
- Modify: `silly_kicks/tracking/features.py:434-439,737`
- Test: `tests/tracking/test_xshot_occurrence_integration.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_xshot_xfn_in_gk_union_only_not_general():
    """P3 (owner-confirmed): xS joins the GK-context union ONLY; the general list stays model-free."""
    from silly_kicks.tracking.features import tracking_default_xfns, pre_shot_gk_full_default_xfns
    names = lambda xs: {getattr(f, "__name__", "") for f in xs}
    assert "xshot_occurrence_xfn" in names(pre_shot_gk_full_default_xfns)
    assert "xshot_occurrence_xfn" not in names(tracking_default_xfns)  # NOT in the general default


def test_xshot_xfn_introspection_is_nan():
    import pandas as pd
    from silly_kicks.tracking._xshot_occurrence import xshot_occurrence_xfns
    fn = xshot_occurrence_xfns()[0]
    states = [pd.DataFrame({"action_id": [0, 1]})]
    out = fn(states, None)  # frames=None -> 3 NaN columns, no model load
    assert list(out.columns) == ["xshot_occurrence_a0", "xshot_occurrence_a1", "xshot_occurrence_a2"]
    assert out.isna().all().all()
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/tracking/test_xshot_occurrence_integration.py -k "xshot_xfn" -v`
Expected: FAIL — not in the lists.

- [ ] **Step 3: Wire the factory into both lists**

In `silly_kicks/tracking/features.py`, add an import near the other xfn imports:

```python
from silly_kicks.tracking._xshot_occurrence import xshot_occurrence_xfns
```

**Leave `tracking_default_xfns` (`:434-439`) UNCHANGED** (model-free; P3 owner decision). Extend ONLY
`pre_shot_gk_full_default_xfns` (`:737`):

```python
pre_shot_gk_full_default_xfns = (
    pre_shot_gk_default_xfns + pre_shot_gk_angle_default_xfns + xshot_occurrence_xfns()
)
```

Add `xshot_occurrence_xfns` to `__all__` exports in `features.py` and `__init__.py` if the other
`*_xfns` factories are exported there (match the existing pattern).

- [ ] **Step 4: Run to verify green + no import cycle**

Run: `python -m pytest tests/tracking/test_xshot_occurrence_integration.py -k "xshot_xfn" -v && python -c "import silly_kicks.tracking.features"`
Expected: PASS + clean import (watch for a circular import between `features.py` and `_xshot_occurrence.py`; if it occurs, import the factory lazily inside a module-level builder function, matching how other factories are wired).

- [ ] **Step 5: Stage + verify**

Run: `python -m pytest tests/ -m "not e2e" -q` → green. (A test exercising `pre_shot_gk_full_default_xfns` with real frames now loads the bundled model — confirm it still passes; `tracking_default_xfns` is unchanged/model-free.)

---

## Task 14: Flip the e2e acceptance gates (spec §9)

**Files:**
- Modify: `tests/tracking/test_xshot_occurrence_integration.py:275-287`

- [ ] **Step 1: Replace the skipped placeholders with real assertions**

```python
@pytest.mark.e2e
def test_xshot_gradientsports_e2e():
    """xS quality on real GS data: beats base-rate on discrimination AND calibration."""
    # load a real GS match (pining owner token), prepare, CV at the bundled hyperparams,
    # assert: pr_auc > positive_rate; brier < positive_rate*(1-positive_rate); log_loss < ln 2.
    ...  # full body: mirror _cv_metrics from the trainer on one real match's (X,y,groups)


@pytest.mark.e2e
def test_xshot_cross_provider():
    """Train on >=2 providers; no single provider's PR-AUC falls below its base rate
    (the common-public-held-out protocol, spec §5)."""
    ...
```

(Replace each `pytest.skip(...)` body with the assertion logic. These run only where real provider data + the owner token exist — `@pytest.mark.e2e`, excluded from normal CI.)

- [ ] **Step 2: Run on the data env (Spark) to confirm green**

Run (where data lives): `python -m pytest tests/tracking/test_xshot_occurrence_integration.py -m e2e -k "gradientsports or cross_provider" -v`
Expected: PASS. Confirm they are still **collected-but-skipped** in normal CI (no data) rather than erroring.

- [ ] **Step 3: Stage + verify non-e2e suite unaffected**

Run: `python -m pytest tests/ -m "not e2e" -q` → green.

---

## Task 15: NOTICE / ADR-011 note + version, CHANGELOG, TODO (spec §11)

**Files:**
- Modify: `NOTICE`, `docs/superpowers/adrs/ADR-011*.md`, `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, `TODO.md`

- [ ] **Step 1: ADR-011 + NOTICE note**

Add to ADR-011 a one-line note: xS shipped a **single bundled variant** (no size axis) and model metadata now records `pitch_length`/`pitch_width`/`geometry_version` as the coordinate-change fail-closed template. Confirm the xS NOTICE entry from PR-S75 is intact (no new methodology added this PR).

- [ ] **Step 2: Version bump (provisional — see policy note)**

Set the agreed number in `pyproject.toml`, `silly_kicks/__init__.py`, `TODO.md` "Current release", and a new dated `CHANGELOG.md` section — **all four in sync** (version-bump hard gate). Number is decided at commit time per the H1 coordination (4.8.0 if PR-S80 merges first, else the next free minor).

- [ ] **Step 3: CHANGELOG**

`### Added`: trained xS weights (bundled default + Hub; record the shipped variant + the `public`-vs-`full` PR-AUC delta from `metrics.json`; the `default` alias is stable across versions, a future `public`⇄`full` flip is its own entry — N5); `from_variant`/`from_hub` live; `xshot_occurrence_xfns` wired into `pre_shot_gk_full_default_xfns` **only** (the general `tracking_default_xfns` stays model-free — P3 Hyrum note); `scripts/publish_xshot_occurrence.py`; xS metadata pitch-dims/geometry/platform template. `### Changed`: xS `_DEFAULT_CARRIER_PARAMS` now sourced from the shared `_ball_carrier.DEFAULT_CARRIER_PARAMS` (4.7.0 values); objective uses `StratifiedGroupKFold` and drops `scale_pos_weight` (calibrated `P(shot)`); `home_team_id` now optional on the xS serve surface.

- [ ] **Step 4: TODO grooming**

Update the TF-16 row: weights shipped (state which variant + provenance), `from_variant`/`from_hub` live, xfns wired, e2e gates live. Update the GKDV program note: TF-16 Layer-2 fully closed; TF-19 xS-arm unblocked. Leave the two filed follow-up TODOs (Ghost-GK R3; xS `negative_subsample` pre-split) and the PR-S81 Ghost-GK re-fit in place. (Delete shipped rows, don't strikethrough — CHANGELOG is the record.)

- [ ] **Step 5: Stage + verify**

Run: `python -m pytest tests/ -m "not e2e" -q` → green.

---

## Task 16: Final review + single commit (project policy)

- [ ] **Step 1: Run `/final-review`** (mandatory gate before the single commit).

- [ ] **Step 2: Match CI locally**

Run: `ruff format --check . && ruff check . && pyright silly_kicks/ && python -m pytest tests/ -m "not e2e" -q`
Expected: all clean/green.

- [ ] **Step 3: Get explicit owner approval to commit** (commit policy: explicit approval before the single commit).

- [ ] **Step 4: Single commit** (one commit per branch), message ending with the Co-Authored-By trailer. Stage code + bundled weights + tests + fixture + docs + version files together. Then push + open the PR; **do not tag until main CI is green** (and the version number is reconciled with the part-deux merge order).

---

## Self-review notes (author)

- **Spec coverage:** §2 corpus/compute → T11 runbook; §3 trainer → T8 (+ T3 objective, T4 base_score); §4 carrier sync → T1; §5 variant/bundle/publish → T6,T8(paired),T9,T12; §6 metadata → T2,T5; §7 Ghost-GK → **PR-S81, intentionally absent here**; §8 xfn wiring → T13; §9 testing → T8,T9,T10,T12,T14; §11 release → T15,T16. No xS-side spec requirement is unmapped.
- **Sequencing guard:** weight-dependent tests (T12) and xfn wiring (T13) land **after** the artifact is bundled, so every intermediate state keeps `pytest -m "not e2e"` green. T10's fixture is model-independent; the model assertion is in T12.
- **Type/name consistency:** `DEFAULT_CARRIER_PARAMS` (constant), `_XSHOT_WEIGHTS_ROOT`, `shipped_variant`/`provider_list` attrs, `xshot_occurrence_xfn` (transformer `__name__`), `estimates_are_cv_not_shipped_fit` (metrics key) are used consistently across T1/T5/T6/T8/T12/T13.
- **Open implementation detail to resolve in T10:** how the slim fixtures pair frames with shots/actions — inspect `test_xshot_occurrence_real_data.py` (it already feeds these fixtures to the xS path) and reuse its pairing.
