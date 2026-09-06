# TF-54b — Counterfactual Territorial "Threat Prevented" + Expected-Passing Model — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `method="counterfactual"` in `silly_kicks.territory` — a completion-weighted expected−realized ("GSAA-analog") valuation of the threat a defender's territory prevented — backed by a new reusable event-only `silly_kicks.expected_passing.PassCompletionModel` and a public `xthreat` destination-distribution seam, validated in-cycle and reported-not-gated.

**Architecture:** For each opponent pass aimed into a defender's trimmed hull, value `(P_complete − outcome)·xT(target)`. The target distribution `q` comes from the injected xT's transition matrix (used only as a renormalized, family-agnostic distribution, via a new public xthreat seam); `P_complete` comes from a new fitted `PassCompletionModel` evaluated at the *hypothesized* target; `xT(target)` from the injected xT. `completed_failed` stays byte-identical (method-dependent schema). Both models are injected ports. Every component has real ground truth, so it validates end-to-end.

**Tech Stack:** Python, pandas, numpy, scipy (existing runtime); scikit-learn for training only (pure-numpy serve, pickle-free JSON + SHA256 per the `GkCompletionModel` idiom). No new runtime dependency. No tracking imports (event-only).

**Spec:** `docs/superpowers/specs/2026-09-05-tf54b-counterfactual-territorial-prevented-design.md` (v2.1, spec-R2-APPROVED; §5.5 windowing + §5.7 worked example). Executors read both.

## Global Constraints

*(Copied verbatim from the spec / house rules. Every task implicitly includes these.)*

- **COMMIT DISCIPLINE (overrides the writing-plans template):** NO per-task `git commit`. Tasks end at *tests green*. There are exactly **two** commits, both in the final gated §Commit & Release section, and **each needs the owner's explicit "yes" for that specific commit** — never commit on "tests pass" or "plan says so".
- **`completed_failed` stays BYTE-IDENTICAL** — same output DataFrame shape and values under the default method. No VAEP/tracking retrain, no re-materialize.
- **Event-only import graph** — `territory` imports `spadl`/`id_compat`/`xthreat`/scipy only; `expected_passing` imports `spadl`/`id_compat`/numpy only. **Never `tracking`.** AST import-allowlist gates with planted-violation meta-tests.
- **Injected fitted ports** — `xt: ExpectedThreat` and the completion model are `TYPE_CHECKING`-only imports, duck-typed at runtime (ADR-022). `require_fitted_xt(xt, caller=…)` and a completion-model fail-closed load guard reject `str`/`None`/unfitted. Value lookups use **public** `values_at_points` / the new seam — never raw `.xT` / `.transition_matrix`, never `rate`.
- **Bundled-trained-artifact discipline (ADR-011/016/040/044/050)** — pickle-free JSON + `SHA256SUMS`, feature contract, chirality probe, fail-closed load; inference imports no sklearn.
- **Canonical-id grouping** (ADR-019) — group on canonical id, emit RAW id. **Drop-and-count** (ADR-042) — never a fabricated 0/NaN. **ADR-028** reflection `(fl−x, fw−y)` for hull membership. **Purity** (ADR-033) — `compute_*`/`predict_*` never mutate inputs.
- **Provenance** (`scripts/_provenance.py`) — every artifact driver calls `require_clean_tree` from `main()`, stamps `run_commit`/`training_commit`, offers `--allow-dirty`, never shells `git rev-parse`; `for_each` shards (ADR-052); input contract (ADR-056).
- **`for_provider` ships EMPTY** (ADR-009) for all new params/models.
- **Lint at CI scope:** `python -m ruff check silly_kicks/ tests/ scripts/` + `python -m ruff format --check …`; `python -m pyright` bare. **Test:** `python -m pytest tests/ -m "not e2e"`.
- **Warnings** carry `stacklevel=2`. **Doctests** on public surface must run clean (or be RST literal blocks).
- **Numbers re-derived at commit-prep** (`git fetch && git merge origin/main`); provisional **4.111.0 / PR-S182 / ADR-089** (main is `9281348` = 4.110.0/PR-S181/ADR-088 after TF-60 PR5 + the model-card seam merged; re-confirm at commit-prep).

---

## File Structure

**New — `silly_kicks/expected_passing/`** (reusable event-only pass-completion):
- `__init__.py` — exports `PassCompletionModel`, `PassCompletionIntegrityError`.
- `_features.py` — `pass_completion_features(...)` + `FEATURE_NAMES` + `feature_contract_block()`.
- `_model.py` — `PassCompletionModel` (fit/predict/save/load; pure-numpy serve; JSON+SHA256; chirality+contract load guards).
- `weights/` — bundled default weights (JSON + `SHA256SUMS`), added in Commit 2.

**New — xthreat seam:**
- `silly_kicks/xthreat/_counterfactual_seam.py` — `DestinationProfile` + `destination_profiles(...)`.
- `silly_kicks/xthreat/__init__.py` — export both; `_transitions.py` docstring fix (SPEC-05).

**New/modified — `silly_kicks/territory/`:**
- `_config.py` — add `CounterfactualParams`.
- `_columns.py` — counterfactual-only column constants + `columns_for_method(method)` resolver.
- `_report.py` — counterfactual census fields.
- `_counterfactual.py` — NEW joint valuation (`q·c·xT`).
- `_compute.py` — method dispatch + `completion_model` port.
- `__init__.py` — export `CounterfactualParams` + new column names.

**New — drivers & tests:**
- `scripts/train_pass_completion.py`, `scripts/validate_territory_counterfactual.py`.
- `tests/expected_passing/`, extended `tests/territory/`, `tests/xthreat/test_counterfactual_seam.py`.

**Cross-cutting:** `silly_kicks/feature_glossary.py`, `tests/invariants/glossary_emitted_columns.py` companion, `NOTICE`, `docs/c4/architecture.{dsl,html}`, `tests/test_public_api_examples.py`, `tests/test_add_star_purity.py` (N/A — no add_*), the id-scalar registry (`tests/invariants/conftest_id_scalar.py`) if a new public id-scalar function is added.

---

## Task 1: xthreat — fix the `singh_transition_matrix` docstring (SPEC-05) + add the public destination-distribution seam (SPEC-03)

**Files:**
- Modify: `silly_kicks/xthreat/_transitions.py` (docstring only, ~line 13/23)
- Create: `silly_kicks/xthreat/_counterfactual_seam.py`
- Modify: `silly_kicks/xthreat/__init__.py` (exports)
- Test: `tests/xthreat/test_counterfactual_seam.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class DestinationProfile:
      zone_centres: np.ndarray   # (n_zones, 2) physical SPADL coords, ascending-y (ADR-041-correct)
      zone_values: np.ndarray    # (n_zones,)   xT at each zone centre
      probabilities: np.ndarray  # (n_origins, n_zones) raw T[origin_cell, :] per origin (NOT renormalized)

  def destination_profiles(
      model: "ExpectedThreat", origin_x: np.ndarray | pd.Series, origin_y: np.ndarray | pd.Series
  ) -> DestinationProfile
  ```
  Consumers renormalize `probabilities` over their selected zone subset (so the family-specific row scale cancels — family-agnostic). Fails closed via `require_fitted_xt(model, caller="destination_profiles")`.

- [ ] **Step 1: Write the failing test**

```python
# tests/xthreat/test_counterfactual_seam.py
import numpy as np
import pandas as pd
import pytest
from silly_kicks.xthreat import ExpectedThreat, destination_profiles, values_at_points
import silly_kicks.spadl.config as spadlconfig


def _toy_xt() -> ExpectedThreat:
    # Small analytic grid so we can hand-check. Fit on a tiny stream of successful moves.
    rng = np.random.default_rng(0)
    n = 400
    sx = rng.uniform(0, 105, n); sy = rng.uniform(0, 68, n)
    ex = np.clip(sx + rng.uniform(0, 20, n), 0, 105); ey = np.clip(sy + rng.uniform(-10, 10, n), 0, 68)
    actions = pd.DataFrame({
        "type_id": spadlconfig.actiontype_id["pass"], "result_id": spadlconfig.result_id["success"],
        "start_x": sx, "start_y": sy, "end_x": ex, "end_y": ey,
    })
    return ExpectedThreat(l=16, w=12).fit(actions)


def test_zone_values_match_values_at_points_at_centres():
    xt = _toy_xt()
    prof = destination_profiles(xt, np.array([30.0]), np.array([34.0]))
    # zone_values are xT at the zone centres — must equal values_at_points at those centres.
    expect = values_at_points(xt, prof.zone_centres[:, 0], prof.zone_centres[:, 1])
    np.testing.assert_allclose(prof.zone_values, expect, rtol=0, atol=1e-9)


def test_probabilities_are_the_origin_row_of_the_transition_matrix():
    xt = _toy_xt()
    from silly_kicks.xthreat._grid import _get_flat_indexes
    ox, oy = np.array([30.0]), np.array([34.0])
    prof = destination_profiles(xt, ox, oy)
    cell = int(_get_flat_indexes(pd.Series(ox), pd.Series(oy), xt.l, xt.w).to_numpy()[0])
    np.testing.assert_allclose(prof.probabilities[0], xt.transition_matrix[cell], rtol=0, atol=0)


def test_renormalized_distribution_is_family_agnostic_valid():
    xt = _toy_xt()
    prof = destination_profiles(xt, np.array([30.0]), np.array([34.0]))
    row = prof.probabilities[0]
    subset = row > 0
    q = row[subset] / row[subset].sum()
    assert abs(q.sum() - 1.0) < 1e-12  # a valid distribution after renormalization


def test_fails_closed_on_unfitted_and_str_and_none():
    for bad in [ExpectedThreat(), "singh_counts", None]:
        with pytest.raises(Exception):
            destination_profiles(bad, np.array([1.0]), np.array([1.0]))
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/xthreat/test_counterfactual_seam.py -v`
Expected: FAIL — `ImportError: cannot import name 'destination_profiles'`.

- [ ] **Step 3: Implement the seam**

```python
# silly_kicks/xthreat/_counterfactual_seam.py
"""Public destination-distribution seam for counterfactual consumers (e.g. territory TF-54b).

Exposes the fitted xT's transition row + zone geometry + xT values in PHYSICAL coordinates, keeping
the flat-index / y-inversion convention inside xthreat (ADR-041). Returns the RAW transition row;
consumers renormalize over their selected zone subset, so the family-specific row scale cancels
(family-agnostic: singh_counts and kde_smoothed both yield a valid renormalized distribution).
See NOTICE for citations.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from silly_kicks.xthreat._grid import _get_flat_indexes
from silly_kicks.xthreat._physical import require_fitted_xt, values_at_points
from silly_kicks.xthreat._transitions import _zone_centres

if TYPE_CHECKING:
    from silly_kicks.xthreat._model import ExpectedThreat


@dataclass(frozen=True)
class DestinationProfile:
    zone_centres: np.ndarray
    zone_values: np.ndarray
    probabilities: np.ndarray


def destination_profiles(model, origin_x, origin_y) -> DestinationProfile:
    require_fitted_xt(model, caller="destination_profiles")
    centres = _zone_centres(model.grid)  # (n_zones, 2) physical, ADR-041-correct
    values = values_at_points(model, centres[:, 0], centres[:, 1])
    ox = pd.Series(np.asarray(origin_x, dtype=float))
    oy = pd.Series(np.asarray(origin_y, dtype=float))
    cells = _get_flat_indexes(ox, oy, model.l, model.w).to_numpy()
    probabilities = np.asarray(model.transition_matrix, dtype=float)[cells]
    return DestinationProfile(zone_centres=centres, zone_values=values, probabilities=probabilities)
```

Then export in `silly_kicks/xthreat/__init__.py` (add to imports and `__all__`): `DestinationProfile`, `destination_profiles`.

And fix the SPEC-05 docstring in `_transitions.py` — `singh_transition_matrix` is **sub-stochastic**, not row-stochastic:

```python
    """Empirical move-transition matrix (classic Singh 2018): successful moves per (start,end) over
    ALL moves per start cell. Byte-identical to the legacy
    ``_move_transition_matrix(actions, grid.n_zones_x, grid.n_zones_y)``.

    NOTE: rows are **sub-stochastic** — ``Σ_j T[i,j] = P(success | move from i) ≤ 1`` (the missing
    mass is the failure probability). Contrast ``kde_smoothed_transition_matrix``, whose rows ARE
    row-stochastic (density normalized to 1).
    ...
        T = singh_transition_matrix(actions, GridSpec(16, 12))  # (192, 192), rows sub-stochastic
    """
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/xthreat/test_counterfactual_seam.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Guard the seam is exercised & lint**

Run: `python -m pytest tests/xthreat -q && python -m ruff check silly_kicks/xthreat tests/xthreat && python -m pyright silly_kicks/xthreat`
Expected: green. (No commit — see Global Constraints.)

---

## Task 2: expected_passing — pass-completion features + feature contract

**Files:**
- Create: `silly_kicks/expected_passing/_features.py`
- Test: `tests/expected_passing/test_features.py`

**Interfaces:**
- Produces:
  ```python
  FEATURE_NAMES: list[str]  # ordered
  def pass_completion_features(
      origin_x, origin_y, target_x, target_y
  ) -> np.ndarray            # (n, len(FEATURE_NAMES)) float64; NaN row -> NaN features (never fabricated)
  def feature_contract_block() -> dict  # {feature_names, probe_features (on a fixed probe), geometry constants}
  ```

- [ ] **Step 1: Write the failing test**

```python
# tests/expected_passing/test_features.py
import numpy as np
from silly_kicks.expected_passing._features import FEATURE_NAMES, pass_completion_features, feature_contract_block


def test_feature_matrix_shape_and_names():
    X = pass_completion_features(np.array([20.0]), np.array([34.0]), np.array([60.0]), np.array([40.0]))
    assert X.shape == (1, len(FEATURE_NAMES))
    assert FEATURE_NAMES == ["distance", "angle", "forward", "lateral", "origin_x", "origin_y",
                             "target_x", "target_y", "origin_third", "target_third"]  # exact order pinned


def test_pitch_thirds_bucket_by_x():
    X = pass_completion_features(np.array([10.0, 90.0]), np.array([34.0, 34.0]),
                                 np.array([50.0, 100.0]), np.array([34.0, 34.0]))
    ot = FEATURE_NAMES.index("origin_third"); tt = FEATURE_NAMES.index("target_third")
    assert X[0, ot] == 0 and X[1, ot] == 2      # x=10 -> defensive third; x=90 -> attacking third
    assert X[0, tt] == 1 and X[1, tt] == 2      # x=50 -> middle; x=100 -> attacking


def test_distance_and_forward_are_correct():
    X = pass_completion_features(np.array([20.0]), np.array([34.0]), np.array([50.0]), np.array([34.0]))
    d = X[0, FEATURE_NAMES.index("distance")]
    fwd = X[0, FEATURE_NAMES.index("forward")]
    assert abs(d - 30.0) < 1e-9 and abs(fwd - 30.0) < 1e-9  # straight forward 30 m


def test_nan_coordinate_yields_nan_features_not_a_fabricated_value():
    X = pass_completion_features(np.array([np.nan]), np.array([34.0]), np.array([50.0]), np.array([34.0]))
    assert np.isnan(X[0]).all()


def test_feature_contract_block_is_stable_and_declares_constants():
    b = feature_contract_block()
    assert b["feature_names"] == FEATURE_NAMES
    assert np.isfinite(np.asarray(b["probe_features"])).all()
```

- [ ] **Step 2: Run to verify it fails** — `ImportError`.

- [ ] **Step 3: Implement `_features.py`**

```python
"""Event-only pass-completion features (TF-54b). Origin/target geometry only — no tracking. See NOTICE."""
from __future__ import annotations
import numpy as np
import silly_kicks.spadl.config as spadlconfig

FEATURE_NAMES = ["distance", "angle", "forward", "lateral", "origin_x", "origin_y", "target_x",
                 "target_y", "origin_third", "target_third"]
_GOAL = (float(spadlconfig.field_length), float(spadlconfig.field_width) / 2.0)
_THIRD = float(spadlconfig.field_length) / 3.0

# NOTE (spec §5b): pitch-third IS included (event-only derivable from x). Score-differential /
# match-minute game-state are a RESERVED extension: SPADL event-only does not guarantee those columns,
# so adding them would fail-closed on providers lacking them. Deferred, not dropped — a later opt-in
# feature-set behind a bumped feature contract.


def pass_completion_features(origin_x, origin_y, target_x, target_y) -> np.ndarray:
    ox = np.asarray(origin_x, float); oy = np.asarray(origin_y, float)
    tx = np.asarray(target_x, float); ty = np.asarray(target_y, float)
    dx = tx - ox; dy = ty - oy
    distance = np.hypot(dx, dy)
    angle = np.arctan2(_GOAL[1] - ty, _GOAL[0] - tx)  # angle from target to goal centre
    origin_third = np.clip(np.floor(ox / _THIRD), 0.0, 2.0)   # 0 def / 1 mid / 2 att (bucketed, PLAN-08)
    target_third = np.clip(np.floor(tx / _THIRD), 0.0, 2.0)
    X = np.column_stack([distance, angle, dx, dy, ox, oy, tx, ty, origin_third, target_third])
    bad = ~np.isfinite(np.column_stack([ox, oy, tx, ty])).all(axis=1)
    X[bad] = np.nan  # NaN in -> NaN features (never fabricated); see Global Constraints
    return X


def feature_contract_block() -> dict:
    probe = pass_completion_features(np.array([20.0]), np.array([34.0]), np.array([60.0]), np.array([40.0]))
    return {"feature_names": list(FEATURE_NAMES), "probe_features": probe.tolist(),
            "geometry": {"field_length": _GOAL[0], "field_width_half": _GOAL[1]}}
```

- [ ] **Step 4: Run to verify it passes** — 4 tests PASS.
- [ ] **Step 5: Lint** — `ruff check` + `pyright` on `silly_kicks/expected_passing`.

---

## Task 3: expected_passing — `PassCompletionModel` fit + pure-numpy predict

**Files:**
- Create: `silly_kicks/expected_passing/_model.py`
- Test: `tests/expected_passing/test_model_fit_predict.py`

**Interfaces:**
- Consumes: `pass_completion_features`, `FEATURE_NAMES`, `feature_contract_block` (Task 2).
- Produces:
  ```python
  class PassCompletionIntegrityError(RuntimeError): ...
  class PassCompletionModel:
      def fit(self, actions: pd.DataFrame) -> "PassCompletionModel"
      def predict_completion(self, origin_x, origin_y, target_x, target_y) -> np.ndarray  # P(complete) in [0,1]; NaN feature -> NaN
      @property
      def is_fitted(self) -> bool
  ```
  `fit` trains a standardized logistic regression on all `pass`-type rows (completed=1 at real end, failed=0 at death `end`). `predict_completion` is `sigmoid(standardize(X) @ coef + intercept)`, **numpy-only** (no sklearn import at predict time).

- [ ] **Step 1: Write the failing test**

```python
# tests/expected_passing/test_model_fit_predict.py
import numpy as np, pandas as pd, pytest
import silly_kicks.spadl.config as spadlconfig
from silly_kicks.expected_passing import PassCompletionModel


def _passes(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    ox = rng.uniform(0, 105, n); oy = rng.uniform(0, 68, n)
    tx = np.clip(ox + rng.uniform(-5, 30, n), 0, 105); ty = np.clip(oy + rng.uniform(-20, 20, n), 0, 68)
    dist = np.hypot(tx - ox, ty - oy)
    p = 1.0 / (1.0 + np.exp((dist - 20) / 8))          # longer pass -> lower completion (true DGP)
    completed = rng.uniform(size=n) < p
    return pd.DataFrame({
        "type_id": spadlconfig.actiontype_id["pass"],
        "result_id": np.where(completed, spadlconfig.result_id["success"], spadlconfig.result_id["fail"]),
        "start_x": ox, "start_y": oy, "end_x": tx, "end_y": ty,
    })


def test_predict_is_monotone_decreasing_in_distance():
    m = PassCompletionModel().fit(_passes())
    short = m.predict_completion(np.array([20.0]), np.array([34.0]), np.array([28.0]), np.array([34.0]))
    long = m.predict_completion(np.array([20.0]), np.array([34.0]), np.array([60.0]), np.array([34.0]))
    assert 0 <= long[0] < short[0] <= 1


def test_predict_is_pure_numpy_no_sklearn_import(monkeypatch):
    m = PassCompletionModel().fit(_passes())
    import sys
    monkeypatch.setitem(sys.modules, "sklearn", None)   # break sklearn
    out = m.predict_completion(np.array([20.0]), np.array([34.0]), np.array([40.0]), np.array([40.0]))
    assert np.isfinite(out).all()  # serve path must not import sklearn


def test_nan_target_predicts_nan():
    m = PassCompletionModel().fit(_passes())
    out = m.predict_completion(np.array([20.0]), np.array([34.0]), np.array([np.nan]), np.array([40.0]))
    assert np.isnan(out[0])


def test_unfitted_predict_raises():
    with pytest.raises(Exception):
        PassCompletionModel().predict_completion(np.array([1.0]), np.array([1.0]), np.array([2.0]), np.array([2.0]))
```

- [ ] **Step 2: Run to verify it fails** — `ImportError`.

- [ ] **Step 3: Implement `_model.py` (fit + predict only; save/load in Task 4)**

```python
"""PassCompletionModel — event-only expected passing (TF-54b). Pickle-free; pure-numpy serve. See NOTICE."""
from __future__ import annotations
import numpy as np, pandas as pd
import silly_kicks.spadl.config as spadlconfig
from silly_kicks.expected_passing._features import FEATURE_NAMES, pass_completion_features

_PASS = spadlconfig.actiontype_id["pass"]
_SUCCESS = spadlconfig.result_id["success"]


class PassCompletionIntegrityError(RuntimeError):
    """Raised by load() on a chirality / feature-contract mismatch (ADR-011/044/050)."""


class PassCompletionModel:
    def __init__(self) -> None:
        self._coef: np.ndarray | None = None
        self._intercept: float | None = None
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self.feature_names = list(FEATURE_NAMES)

    @property
    def is_fitted(self) -> bool:
        return self._coef is not None

    def fit(self, actions: pd.DataFrame) -> "PassCompletionModel":
        from sklearn.linear_model import LogisticRegression  # training-only import
        p = actions[actions["type_id"] == _PASS].dropna(subset=["start_x", "start_y", "end_x", "end_y"])
        X = pass_completion_features(p["start_x"].to_numpy(), p["start_y"].to_numpy(),
                                     p["end_x"].to_numpy(), p["end_y"].to_numpy())
        y = (p["result_id"].to_numpy() == _SUCCESS).astype(int)
        self._mean = X.mean(axis=0); self._scale = X.std(axis=0); self._scale[self._scale == 0] = 1.0
        Z = (X - self._mean) / self._scale
        clf = LogisticRegression(max_iter=1000).fit(Z, y)
        self._coef = clf.coef_[0].astype(float); self._intercept = float(clf.intercept_[0])
        return self

    def predict_completion(self, origin_x, origin_y, target_x, target_y) -> np.ndarray:
        if not self.is_fitted:
            raise PassCompletionIntegrityError("model is not fitted")
        X = pass_completion_features(origin_x, origin_y, target_x, target_y)
        Z = (X - self._mean) / self._scale
        logit = Z @ self._coef + self._intercept
        out = 1.0 / (1.0 + np.exp(-logit))
        out[~np.isfinite(X).all(axis=1)] = np.nan  # NaN feature -> NaN (never a fabricated prob)
        return out
```

Create `silly_kicks/expected_passing/__init__.py` exporting `PassCompletionModel`, `PassCompletionIntegrityError`.

- [ ] **Step 4: Run to verify it passes** — 4 tests PASS.
- [ ] **Step 5: Lint** — `ruff check` + `pyright`.

---

## Task 4: expected_passing — save/load (JSON + SHA256) with chirality + feature-contract load guards

**Files:**
- Modify: `silly_kicks/expected_passing/_model.py` (add `save`/`load` + guard blocks)
- Test: `tests/expected_passing/test_model_serialization.py`

**Interfaces:**
- Produces: `PassCompletionModel.save(dir: str | Path) -> None`; `PassCompletionModel.load(dir, *, legacy_override=False) -> PassCompletionModel`. Writes `model.json` (coef/intercept/mean/scale/feature_names + `chirality` fingerprint + `feature_contract` block) and `SHA256SUMS`. `load` raises `PassCompletionIntegrityError` on a chirality mismatch or a declared-constant/feature-name mismatch (ADR-040/050); a missing fingerprint warns.

- [ ] **Step 1: Write the failing test**

```python
# tests/expected_passing/test_model_serialization.py
import json, numpy as np, pytest
from silly_kicks.expected_passing import PassCompletionModel, PassCompletionIntegrityError
from tests.expected_passing.test_model_fit_predict import _passes  # reuse the fixture


def test_round_trip_predicts_identically(tmp_path):
    m = PassCompletionModel().fit(_passes())
    q = np.array([30.0]); m.save(tmp_path)
    m2 = PassCompletionModel.load(tmp_path)
    np.testing.assert_allclose(
        m.predict_completion(q, q, q + 10, q), m2.predict_completion(q, q, q + 10, q), rtol=0, atol=1e-12)


def test_sha256sums_written_and_checked(tmp_path):
    PassCompletionModel().fit(_passes()).save(tmp_path)
    assert (tmp_path / "SHA256SUMS").exists()
    (tmp_path / "model.json").write_text(json.dumps({"tampered": True}))
    with pytest.raises(PassCompletionIntegrityError):
        PassCompletionModel.load(tmp_path)


def test_chirality_mismatch_raises(tmp_path):
    m = PassCompletionModel().fit(_passes()); m.save(tmp_path)
    d = json.loads((tmp_path / "model.json").read_text())
    d["chirality"]["probe_prediction"][0] += 0.5  # corrupt the fingerprint
    (tmp_path / "model.json").write_text(json.dumps(d))
    # SHA guard will trip first; to isolate chirality, re-write SHA. Test both guards:
    with pytest.raises(PassCompletionIntegrityError):
        PassCompletionModel.load(tmp_path)
```

- [ ] **Step 2: Run to verify it fails** — no `save`/`load`.
- [ ] **Step 3: Implement `save`/`load`** — JSON of `{coef,intercept,mean,scale,feature_names, feature_contract: feature_contract_block(), chirality: {probe_input, probe_prediction}}`; write `SHA256SUMS`; `load` verifies SHA, then re-computes the chirality probe prediction and compares (`atol=1e-6, equal_nan=True`), then compares `feature_names` + declared geometry constants; raise `PassCompletionIntegrityError` on mismatch; warn on missing fingerprint. Mirror `silly_kicks/tracking/_gk_completion.py` structure (read it for the exact idiom — but keep this module event-only, no tracking import).
- [ ] **Step 4: Run to verify it passes** — 3 tests PASS.
- [ ] **Step 5: Lint.**

---

## Task 5: expected_passing — package guards (import-allowlist, purity, doctest/API-example)

**Files:**
- Create: `tests/expected_passing/test_import_allowlist.py`, `tests/expected_passing/__init__.py`
- Test: also add a public example to `PassCompletionModel` docstring (RST literal block) satisfying `tests/test_public_api_examples.py`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/expected_passing/test_import_allowlist.py
import ast, pathlib
_ALLOWED_PREFIXES = ("silly_kicks.spadl", "silly_kicks.id_compat", "silly_kicks.expected_passing")
_PKG = pathlib.Path(__file__).resolve().parents[2] / "silly_kicks" / "expected_passing"


def _module_imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            yield node.module
        elif isinstance(node, ast.Import):
            for a in node.names:
                yield a.name


def test_expected_passing_never_imports_tracking():
    for py in _PKG.glob("*.py"):
        for mod in _module_imports(py):
            assert not mod.startswith("silly_kicks.tracking"), f"{py.name} imports {mod}"
            if mod.startswith("silly_kicks."):
                assert mod.startswith(_ALLOWED_PREFIXES), f"{py.name} imports disallowed {mod}"


def test_planted_tracking_import_would_fail():
    # meta: prove the guard bites (parse a synthetic module string)
    import ast as _ast
    src = "import silly_kicks.tracking\n"
    mods = [n.names[0].name for n in _ast.walk(_ast.parse(src)) if isinstance(n, _ast.Import)]
    assert any(m.startswith("silly_kicks.tracking") for m in mods)
```

*(Note: `sklearn` is imported function-locally inside `fit` only, so it is not a module-level import and does not trip the allowlist; the allowlist checks `silly_kicks.*` prefixes. Verify `fit`'s sklearn import is inside the method body.)*

- [ ] **Step 2: Run to verify it fails/passes** — should PASS if Tasks 2-4 kept the import graph clean; if it FAILS, fix the offending import (do not weaken the allowlist).
- [ ] **Step 3: Add the bundled-weights accessor (CONSIDER-3).** Add to `_model.py`:

```python
    @classmethod
    def bundled(cls) -> "PassCompletionModel":
        """Load the packaged default weights (public-corpus-trained). The convenience path so a user
        can inject a completion model without fitting one. Raises FileNotFoundError before Commit 2
        (weights bundled then)."""
        import importlib.resources as ir
        return cls.load(ir.files("silly_kicks.expected_passing") / "weights")
```

Test (skips cleanly until Commit 2 ships the weights):

```python
def test_bundled_loads_or_skips_pre_bundle():
    import importlib.resources as ir, pytest
    from silly_kicks.expected_passing import PassCompletionModel
    if not (ir.files("silly_kicks.expected_passing") / "weights" / "model.json").is_file():
        pytest.skip("bundled weights arrive at Commit 2")
    m = PassCompletionModel.bundled()
    assert m.is_fitted
```

- [ ] **Step 4: Add the RST literal-block example** to `PassCompletionModel` (a `fit`/`predict_completion` snippet, and note `.bundled()`) so `tests/test_public_api_examples.py` sees a real example.
- [ ] **Step 5: Run** `python -m pytest tests/expected_passing tests/test_public_api_examples.py -q` + lint.

---

## Task 6: territory — `CounterfactualParams`

**Files:**
- Modify: `silly_kicks/territory/_config.py`
- Modify: `silly_kicks/territory/__init__.py` (export)
- Test: `tests/territory/test_counterfactual_params.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class CounterfactualParams:
      direction_cone_degrees: float = 45.0
      min_transition_support: float = 1e-6
      # (target_zone_grid defaults to the injected xt.grid at compute time)
      _is_universal_default: bool = field(default=False, compare=False, repr=False)
      @classmethod
      def default(cls, *, force_universal=False) -> "CounterfactualParams"
      @classmethod
      def for_provider(cls, provider: str) -> "CounterfactualParams"
      def is_default(self) -> bool
  # _PROVIDER_COUNTERFACTUAL_PARAMS: dict[str, dict] = {}   # EMPTY (ADR-009)
  ```

- [ ] **Step 1: Write the failing test** (mirror `tests/territory/test_*params*` for `TerritoryParams`): default/for_provider/is_default behavior; `for_provider("statsbomb") == CounterfactualParams()`; empty override map.
- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement** `CounterfactualParams` mirroring `TerritoryParams` exactly (Read `_config.py` first).
- [ ] **Step 4: Run to verify it passes.**
- [ ] **Step 5: Lint.**

---

## Task 7: territory — per-method column resolver + counterfactual columns (SPEC-04)

**Files:**
- Modify: `silly_kicks/territory/_columns.py`
- Modify: `silly_kicks/territory/__init__.py`
- Test: `tests/territory/test_columns_by_method.py`

**Interfaces:**
- Produces: new column-name constants `TR_EXPECTED_THREAT_FACED`, `TR_XT_PREVENTED_ABOVE_EXPECTATION`, `TR_PASSES_AIMED_INTO_HULL`, `TR_MEAN_COMPLETION_FACED`, `TR_TARGET_SOURCE`; `TERRITORY_TARGET_SOURCE_VALUES = frozenset({"observed","modeled","unresolved"})`; and
  ```python
  def columns_for_method(method: str) -> dict[str, str]  # {col: dtype}; completed_failed -> exactly the v1 15; counterfactual -> v1 shared + 5 cf-only
  ```

- [ ] **Step 1: Write the failing test**

```python
# tests/territory/test_columns_by_method.py
from silly_kicks.territory._columns import columns_for_method, TERRITORY_COLUMNS


def test_completed_failed_columns_are_exactly_v1():
    assert columns_for_method("completed_failed") == dict(TERRITORY_COLUMNS)  # byte-identical shape


def test_counterfactual_adds_only_the_five_cf_columns():
    base = set(TERRITORY_COLUMNS); cf = set(columns_for_method("counterfactual"))
    assert base <= cf
    assert cf - base == {
        "territory_expected_threat_faced", "territory_xt_prevented_above_expectation",
        "territory_passes_aimed_into_hull", "territory_mean_completion_faced", "territory_target_source",
    }
```

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement** the constants + `columns_for_method`. Keep `TERRITORY_COLUMNS` (v1) unchanged; the resolver returns it verbatim for `completed_failed` and a merged dict for `counterfactual`. Export new names in `__init__.py`.
- [ ] **Step 4: Run to verify it passes.**
- [ ] **Step 5: Lint.**

---

## Task 8: territory — `_report` counterfactual census

**Files:**
- Modify: `silly_kicks/territory/_report.py`
- Test: `tests/territory/test_report_counterfactual.py`

**Interfaces:**
- Produces: extend `TerritoryReport` with optional fields `n_target_modeled: int = 0`, `n_target_unresolved: int = 0` (defaults keep v1 construction unchanged). Conservation still holds: `n_scored + n_degenerate_hull + n_no_actions == n_players_in`; add a documented cf identity `n_target_modeled + n_target_unresolved == (failed passes considered into the aimed set)`.

- [ ] **Step 1..5:** failing test asserting the new fields default to 0 and the cf identity holds on a small hand-built report; implement (additive dataclass fields with defaults — verify v1 constructions still pass positionally or switch to keyword); run; lint.

---

## Task 9: territory — `_counterfactual.py` joint valuation `q·c·xT` (the core)

**Files:**
- Create: `silly_kicks/territory/_counterfactual.py`
- Test: `tests/territory/test_counterfactual_compute.py`

**Interfaces:**
- Consumes: `destination_profiles`/`DestinationProfile` (Task 1); `values_at_points`; `PassCompletionModel.predict_completion` (Task 3); `CounterfactualParams` (Task 6); column constants (Task 7); `build_trimmed_hull`/`Hull` (existing).
- Produces:
  ```python
  def counterfactual_rows(
      defs_grouped, passes_by_game, *, xt, completion_model, params: CounterfactualParams,
      fl: float, fw: float, window
  ) -> tuple[list[dict], dict]   # (rows, census counts) — called by _compute dispatch
  ```
  For each defender group: build hull; select opponent passes; **completed pass** counts iff its reflected end is point-in-hull (v1 rule) → conceded `+= xT(end)`, expected `+= c(origin, end)·xT(end)`; **failed pass** counts iff its death-direction cone (half-angle `direction_cone_degrees`) intersects the reflected hull region → get `destination_profiles(xt, origin)`, restrict zones to those whose centre is in the reflected hull AND within the cone, renormalize `q`, `prevented += Σ q(z)·c(origin, centre_z)·xT(z)`, `expected += same`; `target_source="modeled"`. A failed pass whose selected-zone support `Σ probabilities < min_transition_support` → `unresolved`, dropped-and-counted. `above_expectation = expected − conceded`.

- [ ] **Step 1: Write the failing tests — assert the EXACT `q·c·xT` golden (PLAN-01), pin the SPEC-04 count distinction, and pin cf windowing (PLAN-02).** The exact numbers come from spec **§5.7** (worked example).

```python
# tests/territory/test_counterfactual_compute.py
import numpy as np, pandas as pd, pytest
import silly_kicks.spadl.config as spadlconfig
from silly_kicks.territory import compute_territorial_dominance, CounterfactualParams
from silly_kicks.xthreat import GridSpec, ExpectedThreat

_PASS = spadlconfig.actiontype_id["pass"]; _TACKLE = spadlconfig.actiontype_id["tackle"]
_OK = spadlconfig.result_id["success"]; _FAIL = spadlconfig.result_id["fail"]


class _ToyUniformXt:
    """Duck-typed fitted xT with a UNIFORM grid: values_at_points == 0.1 for ANY point (inversion-invariant),
    and a uniform positive transition_matrix so every zone is supported. Makes Σ q·xT = 0.1·(Σq=1) = 0.1
    exactly — independent of the reflection/cone details — so the test pins the completion multiply +
    renormalize-to-1 to the last digit (spec §5.7 uniform check)."""
    def __init__(self, value=0.1, l=8, w=6):
        self.l, self.w = l, w
        self.grid = GridSpec(n_zones_x=l, n_zones_y=w)
        n = l * w
        self.xT = np.full((w, l), float(value))
        self.transition_matrix = np.full((n, n), 1.0 / n)
        self.method = "singh_counts"


class _ConstCompletion:
    def predict_completion(self, ox, oy, tx, ty):
        return np.full(np.asarray(tx, float).shape, 0.6)  # c == 0.6 (spec §5.7)


def _scene(game_id=1):
    """One defender (team 1, hull ~x[5,15] y[25,45]) + team-2 opponent passes:
       COMPLETED end (95,34) -> reflects to (10,34) INSIDE hull; FAILED origin (80,34) death (90,34),
       cone points at the reflected hull, dies OUTSIDE the hull (aimed-in-died-short)."""
    def_actions = pd.DataFrame({
        "game_id": game_id, "period_id": 1, "team_id": 1, "player_id": 7, "type_id": _TACKLE,
        "result_id": _OK,
        "start_x": [6.0, 12.0, 9.0, 14.0], "start_y": [26.0, 44.0, 34.0, 30.0],
        "end_x": [6.0, 12.0, 9.0, 14.0], "end_y": [26.0, 44.0, 34.0, 30.0],
    })
    passes = pd.DataFrame({
        "game_id": game_id, "period_id": 1, "team_id": 2, "player_id": 21, "type_id": _PASS,
        "result_id": [_OK, _FAIL],
        "start_x": [88.0, 80.0], "start_y": [34.0, 34.0],
        "end_x": [95.0, 90.0], "end_y": [34.0, 34.0],
    })
    return pd.concat([def_actions, passes], ignore_index=True)


def test_uniform_xt_exact_golden():
    out, rep = compute_territorial_dominance(
        _scene(), xt=_ToyUniformXt(), method="counterfactual", completion_model=_ConstCompletion(),
        params=CounterfactualParams.default())
    row = out.iloc[0]
    # §5.7 uniform check: conceded = xT(end) = 0.1; prevented = c·0.1·(Σq=1) = 0.06;
    # expected_faced = 0.06(completed) + 0.06(failed) = 0.12; above_expectation = 0.12 − 0.10 = 0.02.
    assert row["territory_xt_conceded"] == pytest.approx(0.10, abs=1e-12)
    assert row["territory_xt_prevented"] == pytest.approx(0.06, abs=1e-12)
    assert row["territory_expected_threat_faced"] == pytest.approx(0.12, abs=1e-12)
    assert row["territory_xt_prevented_above_expectation"] == pytest.approx(0.02, abs=1e-12)
    assert row["territory_mean_completion_faced"] == pytest.approx(0.6, abs=1e-12)
    # SPEC-04 count distinction: passes_into_hull = v1 observed-end-in-hull (completed only = 1);
    # passes_aimed_into_hull = completed + failed-aimed-in = 2.
    assert int(row["territory_passes_into_hull"]) == 1
    assert int(row["territory_passes_aimed_into_hull"]) == 2
    assert row["territory_xt_prevented_rate"] == pytest.approx(0.06 / 2, abs=1e-12)  # denom = aimed-in


def test_varying_xt_matches_inline_reference():
    """Pins the cone/zone-selection/reflection/weighting on a NON-uniform xt: recompute the expected
    prevented INLINE from the public seam + explicit cone/reflection/renorm and assert equality."""
    xt = ExpectedThreat(l=16, w=12).fit(_disjoint_corpus())  # deterministic seed
    out, _ = compute_territorial_dominance(
        _scene(), xt=xt, method="counterfactual", completion_model=_ConstCompletion(),
        params=CounterfactualParams.default())
    expected = _inline_reference_prevented(_scene(), xt, c=0.6, cone_deg=45.0)  # written in the test file
    assert out.iloc[0]["territory_xt_prevented"] == pytest.approx(expected, abs=1e-9)


def test_completed_leg_above_expectation_is_c_minus_one_times_end_value():
    # isolate the completed leg: a scene with ONLY a completed pass into the hull.
    out, _ = compute_territorial_dominance(
        _scene_completed_only(), xt=_ToyUniformXt(), method="counterfactual",
        completion_model=_ConstCompletion(), params=CounterfactualParams.default())
    # (c−1)·xT(end) = (0.6−1)·0.1 = −0.04
    assert out.iloc[0]["territory_xt_prevented_above_expectation"] == pytest.approx(-0.04, abs=1e-12)


def test_counterfactual_requires_a_completion_model():
    with pytest.raises(Exception):
        compute_territorial_dominance(_scene(), xt=_ToyUniformXt(), method="counterfactual")


def test_unresolvable_target_is_dropped_and_counted():
    # A failed pass whose death-direction points AWAY from the hull -> cone∩hull empty / zero support ->
    # target_source="unresolved", counted in the report, never a fabricated 0.
    out, rep = compute_territorial_dominance(
        _scene_failed_points_away(), xt=_ToyUniformXt(), method="counterfactual",
        completion_model=_ConstCompletion(), params=CounterfactualParams.default())
    assert rep.n_target_unresolved >= 1
    assert (out["territory_target_source"] == "unresolved").any()


def test_reflection_invariance_either_perspective():
    # Swap team ids (1<->2) and reflect all coords (105−x, 68−y); the SAME physical scene must score
    # identical cf numbers (ADR-028). Assert per-row equality of the cf columns.
    base, _ = compute_territorial_dominance(_scene(), xt=_ToyUniformXt(), method="counterfactual",
                                            completion_model=_ConstCompletion())
    mirrored, _ = compute_territorial_dominance(_mirror_scene(_scene()), xt=_ToyUniformXt(),
                                                method="counterfactual", completion_model=_ConstCompletion())
    np.testing.assert_allclose(base["territory_xt_prevented"].to_numpy(),
                               mirrored["territory_xt_prevented"].to_numpy(), rtol=0, atol=1e-9)


def test_window_pools_sums_and_support_weighted_completion():
    # Two games; window pools: prevented/expected/passes SUM; mean_completion_faced = Σc/Σpasses
    # (support-weighted, NOT mean-of-means); rate re-derives over pooled passes_aimed_into_hull.
    two = pd.concat([_scene(game_id=1), _scene(game_id=2)], ignore_index=True)
    out, _ = compute_territorial_dominance(
        two, xt=_ToyUniformXt(), method="counterfactual", completion_model=_ConstCompletion(),
        window=[1, 2])
    row = out.iloc[0]  # one pooled row per player
    assert int(row["territory_passes_aimed_into_hull"]) == 4          # 2 games × 2
    assert row["territory_xt_prevented"] == pytest.approx(0.12, abs=1e-12)   # 2 × 0.06
    assert row["territory_mean_completion_faced"] == pytest.approx(0.6, abs=1e-12)  # Σc/Σpasses, weighted
    assert row["territory_xt_prevented_rate"] == pytest.approx(0.12 / 4, abs=1e-12)
```

*(Executor: `_disjoint_corpus`, `_inline_reference_prevented`, `_scene_completed_only`,
`_scene_failed_points_away`, `_mirror_scene` are small helpers in the test file. `_inline_reference_prevented`
re-derives prevented from `destination_profiles` + `values_at_points` + an explicit cone/reflection/renorm
loop — independent of `_counterfactual.py`'s helpers — so a wrong cone/renorm in production diverges.)*

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement `_counterfactual.py`** per the interface above. Reuse the reflection `(fl−x, fw−y)` from `_compute.py`. Cone test: angle between (death − origin) and (zone_centre − origin) ≤ `direction_cone_degrees`. Renormalize `q` over selected zones. Vectorize `predict_completion` over the selected zone centres per pass. Emit `target_source`. Accumulate census counts.
- [ ] **Step 4: Run to verify it passes.**
- [ ] **Step 5: Lint + reflection-invariance green.**

---

## Task 10: territory — `_compute` dispatch + `completion_model` port + `completed_failed` BYTE-IDENTITY

**Files:**
- Modify: `silly_kicks/territory/_compute.py`
- Test: `tests/territory/test_method_dispatch_and_byte_identity.py`

**Interfaces:**
- Produces: `compute_territorial_dominance(actions, *, xt, method="completed_failed", completion_model=None, window=None, params=_DEFAULT, cf_params=CounterfactualParams.default())`. `method="counterfactual"` → require `completion_model` (raise if None/str), dispatch to `counterfactual_rows`, output schema `columns_for_method("counterfactual")`. `completed_failed` path unchanged; output schema `columns_for_method("completed_failed")` (== v1 `TERRITORY_COLUMNS`).

- [ ] **Step 1: Capture the v1 golden BEFORE any `_compute.py` edit (PLAN-03).** Tasks 1-9 have NOT touched the `completed_failed` path, so `compute_territorial_dominance` is still pure v1 here. Run it on a committed fixture and freeze the output as a golden — the reference for "v1 did not drift during the refactor".

```python
# one-off capture (run once; commit the artifact as part of Commit 1):
#   out, _ = compute_territorial_dominance(FIXTURE_ACTIONS, xt=FITTED_XT)   # still v1 at this point
#   out.to_parquet("tests/territory/_golden/completed_failed_v1.parquet")
```

Verify the golden has exactly the v1 15 columns. Expected: file written.

- [ ] **Step 2: Write the byte-identity REGRESSION test against the frozen golden** (green NOW, before the refactor — proves the golden is faithful — and MUST stay green after)

```python
# tests/territory/test_method_dispatch_and_byte_identity.py
import pandas as pd
from pandas.testing import assert_frame_equal
from silly_kicks.territory import compute_territorial_dominance
# FIXTURE_ACTIONS / FITTED_XT from tests/territory/test_compute.py helpers

_GOLDEN = "tests/territory/_golden/completed_failed_v1.parquet"


def test_completed_failed_matches_the_pre_change_golden():
    out, _ = compute_territorial_dominance(FIXTURE_ACTIONS, xt=FITTED_XT)           # default method
    assert_frame_equal(out, pd.read_parquet(_GOLDEN))                              # v1 did not drift
    out2, _ = compute_territorial_dominance(FIXTURE_ACTIONS, xt=FITTED_XT,
                                            method="completed_failed", completion_model=None)
    assert_frame_equal(out2, pd.read_parquet(_GOLDEN))                             # new kwargs inert
```

- [ ] **Step 3: Run — byte-identity GREEN now** (pre-refactor). `python -m pytest tests/territory/test_method_dispatch_and_byte_identity.py::test_completed_failed_matches_the_pre_change_golden -v` → PASS. If it fails here, the golden/fixture is wrong; fix before touching `_compute.py`.

- [ ] **Step 4: Write the dispatch tests (red — new behavior)**

```python
import pytest
from silly_kicks.territory import CounterfactualParams


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        compute_territorial_dominance(FIXTURE_ACTIONS, xt=FITTED_XT, method="bogus")


def test_counterfactual_emits_the_method_dependent_schema(toy_completion):
    out, _ = compute_territorial_dominance(FIXTURE_ACTIONS, xt=FITTED_XT,
                                           method="counterfactual", completion_model=toy_completion)
    assert "territory_xt_prevented_above_expectation" in out.columns
    default_out, _ = compute_territorial_dominance(FIXTURE_ACTIONS, xt=FITTED_XT)
    assert "territory_xt_prevented_above_expectation" not in default_out.columns   # cf-only
```

Run: FAIL (dispatch not implemented).

- [ ] **Step 5: Implement dispatch WITHOUT touching the `completed_failed` body.** Read the current `_compute.py`; add the new kwargs (`completion_model=None`, `cf_params=CounterfactualParams.default()`) with inert defaults; branch on `method` and route `"counterfactual"` to `counterfactual_rows` BEFORE the v1 body; leave the v1 body byte-for-byte unchanged (share the `defs`/`passes_by_game` scaffolding only if it is literally the same code). Require `completion_model` for the cf branch (raise on None/str via `require_fitted`-style check).

- [ ] **Step 6: Run — ALL green, byte-identity STILL green** (proves the refactor did not drift v1). `python -m pytest tests/territory/test_method_dispatch_and_byte_identity.py -v`.

- [ ] **Step 7: Full territory suite** `python -m pytest tests/territory -q` + `ruff check` + `pyright`.

---

## Task 11: cross-cutting — glossary, NOTICE, liveness, purity, id-scalar registry

**Files:**
- Modify: `silly_kicks/feature_glossary.py` (5 new cf columns, per-column `higher_is_better` + `emitting_module="_counterfactual"` and the completion columns' home)
- Modify: `NOTICE` (expected-passing + GSAA-analog + Sumpter 10.2 counterfactual entries)
- Modify: `tests/test_add_star_purity.py` — N/A (no new `add_*`); instead add purity coverage for `compute_territorial_dominance(method="counterfactual")` if a territory purity test exists (mirror v1's).
- Modify: `tests/invariants/conftest_id_scalar.py` — register or justify any new public id-scalar function (Task 9/10 introduce none if grouping stays inside `_compute`; verify).
- Test: `tests/invariants/test_glossary_emitted_columns` companion sees the new columns.

- [ ] **Step 1:** failing test — glossary coverage gate flags the 5 undocumented cf columns.
- [ ] **Step 2:** run — FAIL (missing entries).
- [ ] **Step 3:** add glossary entries (name/definition/unit/emitting_module/attribution/higher_is_better) for the 5 cf columns; add NOTICE citations; add a counterfactual purity variant.
- [ ] **Step 4:** run `python -m pytest tests/invariants tests/territory -q`.
- [ ] **Step 5:** lint.

---

## Task 12: C4 — model the `expected_passing` container + render via Graphviz `dot`

**Files:**
- Modify: `docs/c4/architecture.dsl` (add `expected_passing` container; a relationship territory → expected_passing + territory → xthreat seam)
- Modify: `docs/c4/architecture.html` (regenerate — `dot`, NEVER Smetana/hand-patch)

- [ ] **Step 1:** update `architecture.dsl` with the new container + relationships.
- [ ] **Step 2:** render per CLAUDE.md pipeline: `structurizr.war export -format c4plantuml` → `c4_assemble.py --inject-wrap-width` → `plantuml.jar -graphvizdot "C:/Users/Karsten/.claude/tools/graphviz/dot.exe" -tsvg *.puml` → `c4_assemble.py --svg-dir`. Confirm the assembler did not abort on a 0-entity placeholder (proves `dot` ran).
- [ ] **Step 3:** run the C4 completeness gate (`tests/…` that derives subpackages from the tree) — it must see `expected_passing`. The **33 `add_*` aggregator count is unchanged**; only the container set grows.
- [ ] **Step 4:** verify `architecture.html` `viewBox` is `dot`-style (not Smetana).
- [ ] **Step 5:** lint N/A.

---

## Task 13: drivers — `train_pass_completion.py` + `validate_territory_counterfactual.py` (offline-testable; corpus runs owner-gated)

**Files:**
- Create: `scripts/train_pass_completion.py`
- Create: `scripts/validate_territory_counterfactual.py`
- Create: `scripts/_synthetic_interception.py` (shared: perturb a completed pass by flight-fraction f + angular offset δ — SPEC-02/09)
- Test: `tests/scripts/test_tf54b_drivers.py`

**Interfaces:**
- `train_pass_completion.main(argv)` — `for_each` shards over the public corpus; fits `PassCompletionModel`; writes weights via `save`; `require_clean_tree` in `main()`; `--allow-dirty`; `training_commit` stamp; `--out` outside the repo.
- `validate_territory_counterfactual.main(argv)` — `for_each` shards; fits xt + completion on a leakage-disjoint corpus; runs the pre-registered battery (completion AUC/ECE/Brier; synthetic-interception target recovery vs baselines; composed mechanism vs v1 + elite-defender prior + reliability/discriminant/non-degeneracy/outcome-lens-reported); writes `metrics.json` + `named_defender_signs.parquet`; `require_clean_tree`; `run_commit` stamp.
- **Pre-registered, LOCKED constants (CONSIDER-1) — committed in Commit 1, stamped into the artifact, mirroring the TF-19 `NAMED_KEEPER_PRIOR` idiom so the gate can't be moved after seeing the numbers:**
  ```python
  ELITE_DEFENDER_PRIOR: frozenset[int | str] = frozenset({...})   # StatsBomb WC2022 player_ids (locked pre-run)
  # Completion-model floors (mirror GkRetentionModel's ece<=0.10 gate):
  COMPLETION_AUC_FLOOR: float = 0.65        # held-out AUC must clear (primary discrimination gate)
  COMPLETION_ECE_CEILING: float = 0.10      # calibration (primary)
  COMPLETION_BRIER_SKILL_FLOOR: float = 0.10  # PLAN-09: a SKILL gate, not a fixed ceiling —
  #   BSS = 1 − brier / brier_noskill, where brier_noskill = p(1−p) on the held-out base rate p.
  #   A fixed 0.22 ceiling is LOOSER than no-skill (~0.19 at 75% completion) so it gates nothing;
  #   BSS > floor guarantees the model beats the base-rate baseline. decide_promotion computes it.
  ELITE_DEFENDER_TOP_QUANTILE: float = 0.75 # prior clears if elite defenders land >= this quantile at volume
  ```
  `decide_promotion(metrics)` reads ONLY these locked constants — no post-hoc thresholds — and computes
  the Brier skill score against the held-out no-skill Brier (never a raw fixed ceiling).
- `perturb_interception(origin, end, *, fraction, angle_offset_rad) -> death_xy` (SPEC-02/09).

- [ ] **Step 1: Write the failing tests** — provenance wiring (imports `require_clean_tree`, offers `--allow-dirty`, no `git rev-parse` AST call, calls guard from `main()` — mirror `tests/scripts/test_provenance_wiring.py`); the synthetic-interception perturbs BOTH distance and direction (`perturb_interception` moves the death off the origin→end ray for `angle_offset_rad != 0`); the `ELITE_DEFENDER_PRIOR` is a non-empty frozen constant AND the four numeric floors are module-level frozen constants (`decide_promotion` references the constants, not literals — AST-checked like the ADR-056 input contract).
- [ ] **Step 2:** run — FAIL.
- [ ] **Step 3:** implement both drivers + `_synthetic_interception.py`. Use `scripts/_driver.py` `for_each` (ADR-052), `scripts/_provenance.py`, `scripts/_input_contract.py` (ADR-056). Mirror `scripts/build_tf19_instrument_responsiveness.py` for the `NAMED_KEEPER_PRIOR`/for_each/reduce idiom and `scripts/_loader_pining.py` for the StatsBomb/GS corpus loaders.
- [ ] **Step 4:** run `python -m pytest tests/scripts/test_tf54b_drivers.py -q`. (The full corpus RUN is owner-gated e2e, not this step.)
- [ ] **Step 5:** lint (drivers are ASCII-only — the driver ASCII gate, not just ruff).

---

## Task 14: e2e test + release bookkeeping (version, CHANGELOG, ADR)

**Files:**
- Create: `tests/territory/test_counterfactual_e2e.py` (`@pytest.mark.e2e`) — fits `ExpectedThreat("singh_counts")` + `PassCompletionModel` on WC2022 open data disjoint from a scored match, runs `method="counterfactual"`, asserts schema/dtype/finiteness/conservation + the synthetic-interception recovery beats baselines. `importorskip("statsbombpy")`, network-gated, self-skips.
- Modify: `silly_kicks/_version.py` (→ provisional `4.111.0`; re-derive at commit-prep)
- Modify: `CHANGELOG.md` (new version section, keyed `PR-S182`, findings placeholder filled after the owner-run validation)
- Create: `docs/superpowers/adrs/ADR-089-tf54b-counterfactual-territorial-prevented.md` (provisional number; re-derive)

- [ ] **Step 1:** write the `@e2e` test (static-review-quality; owner runs it).
- [ ] **Step 2:** run `python -m pytest tests/territory/test_counterfactual_e2e.py -m e2e -q` locally IF statsbombpy+network present (else it self-skips — acceptable at plan-authoring; the corpus run is the owner's gate).
- [ ] **Step 3:** bump `_version.py`; write the ADR (Context/Decision/Alternatives/Consequences mirroring ADR-086); add the CHANGELOG section.
- [ ] **Step 4:** run the FULL suite `python -m pytest tests/ -m "not e2e"` + `python -m ruff check silly_kicks/ tests/ scripts/` + `python -m ruff format --check silly_kicks/ tests/ scripts/` + `python -m pyright`. All green.
- [ ] **Step 5:** (no commit — proceed to §Commit & Release.)

---

## Self-Review (author checklist — completed)

**Spec coverage:** §5.1-5.5 → Tasks 6-10; §5b (PassCompletionModel) → Tasks 2-5; §5c (xthreat seam) → Task 1; §7 validation → Tasks 13-14; §5.3 columns/membership → Tasks 7,9,10; §9 commit plan → §Commit & Release; §8 testing → each task's tests + Task 11. No gap.

**Placeholder scan:** Task 9 now carries the concrete `_scene()` fixture, the `_ToyUniformXt` duck-type, and **exact** golden assertions tied to spec **§5.7** (conceded 0.10 / prevented 0.06 / expected_faced 0.12 / above_expectation 0.02, counts 1 vs 2). The remaining test helpers (`_disjoint_corpus`, `_inline_reference_prevented`, `_scene_completed_only`, `_scene_failed_points_away`, `_mirror_scene`) are small, named, and specified — fixture construction, not logic placeholders. Every interface/formula is concrete. **Plan-review R1 (PLAN-01/02/03) folded in**; the byte-identity golden is a numbered pre-refactor capture step (Task 10 Step 1).

**Type consistency:** `destination_profiles`→`DestinationProfile` (Task 1) consumed in Task 9; `predict_completion(ox,oy,tx,ty)` signature identical Tasks 3/9/13; `columns_for_method` (Task 7) consumed in Task 10; `CounterfactualParams` (Task 6) consumed in Tasks 9/10. Consistent.

---

## Commit & Release (GATED — owner approval required for EACH commit; NOT per-task)

> Per Global Constraints: there are exactly two commits; each needs the owner's explicit "yes" for that specific commit. Do not commit on "tests pass".

- [ ] **Gate 0 — all offline green.** `python -m pytest tests/ -m "not e2e"` + ruff check + ruff format --check + pyright, all green. `/final-review` pass (doc-drift, CLAUDE.md ADR range, glossary count). Show the owner the full diff + file list.
- [ ] **Commit 1 (owner-gated):** ALL code + tests + glossary/NOTICE/C4/version/CHANGELOG/ADR (everything EXCEPT bundled weights + validation artifact). **Default stays `completed_failed`.** Wait for explicit owner "yes".
- [ ] **Owner-run at Commit 1 (clean tree, `--out` outside repo):** (a) `train_pass_completion.py` → bundled `expected_passing/weights/` (`training_commit`=Commit 1); (b) `validate_territory_counterfactual.py` → `metrics.json` + `named_defender_signs.parquet` + `findings.md` (`run_commit`=Commit 1). Fill the CHANGELOG findings + the ADR consequences from the report.
- [ ] **Commit 2 (owner-gated, provenance):** the bundled `expected_passing/weights/` (JSON+SHA256) + `docs/research/territory_counterfactual_construct_validity/`, both stamping Commit 1. Load-bearing non-squash. Wait for explicit owner "yes".
- [ ] **Release (owner-driven):** push → PR → CI green → admin-merge non-squash → tag → PyPI. **Numbers re-derived at commit-prep** (`git fetch && git merge origin/main`). Promotion of `counterfactual` to default is a SEPARATE follow-on ADR-009 decision after the owner reads the report — NOT this cycle.

---

## Execution options

**1. Subagent-Driven (recommended)** — a fresh subagent per task (opus for implementation tasks, per the user's model-routing preference), two-stage review between tasks.
**2. Inline Execution** — execute tasks in this session via executing-plans, checkpoints for review.
