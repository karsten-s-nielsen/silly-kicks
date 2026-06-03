# PR-S81 — Ghost-GK re-fit + serve-carrier fix + R3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the Ghost-GK serve-carrier train/serve skew, add xS-style R3 carrier-param record/consume, and re-fit the bundled weights against the 4.7.0 carrier defaults.

**Architecture:** Part 1 is a pure-code change (serve fix + optional carrier passthrough + R3 metadata) landed and tested against incumbent weights. Part 2 runs the re-fit on DGX Spark over SSH against the committed Part-1 SHA. Part 3 validates (apples-to-apples gate + measured serve delta), bundles, and ships.

**Tech Stack:** pandas/numpy, scikit-learn `HistGradientBoostingRegressor`, xgboost-free KDE, pytest, pining loaders, DGX Spark (ARM aarch64).

**Spec:** `docs/superpowers/specs/2026-06-02-pr-s81-ghost-gk-refit-design.md`

---

## File Structure

| File | Responsibility | Part |
|------|----------------|------|
| `silly_kicks/tracking/_ball_carrier.py` | `DEFAULT_CARRIER_PARAMS` (exists, unchanged) — single source | — |
| `silly_kicks/tracking/_ghost_gk.py` | R3 on `GhostGkModel`; `compute_ghost_gk` serve fix + `carrier=` passthrough; `prepare_ghost_gk_training_data` `carrier_params=` kwarg | 1 |
| `silly_kicks/tracking/features.py` | thread optional `carrier=` through `add_ghost_gk` + `ghost_gk_xfns` | 1 |
| `scripts/train_ghost_gk.py` | single-`cp` wiring + provenance metadata + variant/carrier CLI args | 1 |
| `scripts/measure_ghost_gk_serve_delta.py` | P3 buggy-vs-fixed serve delta (new) | 1 |
| `scripts/validate_ghost_gk_refit.py` | P2/N4 incumbent-on-holdout gate (new) | 1 |
| `scripts/_loader_pining_to_cache.py` | NP1 pining→trainer-layout cache writer (new) | 1 |
| `tests/tracking/test_ghost_gk_r3.py` | R3 + serve-fix + carrier-passthrough tests (new) | 1 |
| `tests/tracking/test_train_ghost_gk_cli.py` | NP3 single-cp wiring test (new; in tracking/ to avoid the `scripts` namespace collision) | 1 |
| `tests/tracking/test_loader_pining_to_cache.py` | NP1 cache-writer round-trip test (new; in tracking/) | 1 |
| `tests/tracking/test_ghost_gk_frame_restriction.py` | extend `TestExtractionRestriction` with carrier active | 1 |
| `silly_kicks/tracking/_ghost_gk_weights/{default,full}/` | re-fit artifacts (or backfilled metadata) | 3 |
| `CHANGELOG.md`, `NOTICE`, `CLAUDE.md`, `TODO.md`, `pyproject.toml`, `silly_kicks/__init__.py` | release | 3 |

**Atomic mirror:** `silly_kicks/atomic/tracking/features.py` re-imports `add_ghost_gk`/`ghost_gk_xfns` — inherits the fix; verified in Task 7, no logic change.

---

# PART 1 — Code (lands as the committed SHA the box trains against)

## Task 1: R3 metadata schema on `GhostGkModel`

**Files:**
- Modify: `silly_kicks/tracking/_ghost_gk.py` (import block ~line 31; `__init__` 1244-1254; `fit` 1256; `save` 1514-1532; `load` 1599-1608)
- Test: `tests/tracking/test_ghost_gk_r3.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/tracking/test_ghost_gk_r3.py`:

```python
"""R3 carrier-param record/consume + serve-carrier consistency (PR-S81)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS
from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES, GhostGkModel
from tests.tracking.test_ghost_gk import _fitted_model


class TestR3Metadata:
    def test_default_carrier_params_on_construction(self):
        model = GhostGkModel()
        assert model.carrier_params == DEFAULT_CARRIER_PARAMS
        assert model.carrier_params is not DEFAULT_CARRIER_PARAMS  # defensive copy

    def test_fit_records_supplied_carrier_params(self):
        model, X, labels = _fitted_model()
        cp = {"tolerance_m": 3.0, "beta": 0.9, "gamma": 0.25}
        fresh = GhostGkModel(n_estimators=10)
        fresh.fit(X, labels, carrier_params=cp)
        assert fresh.carrier_params == cp

    def test_save_load_round_trips_carrier_params(self):
        model, X, labels = _fitted_model()
        cp = {"tolerance_m": 3.0, "beta": 0.9, "gamma": 0.25}
        fresh = GhostGkModel(n_estimators=10)
        fresh.fit(X, labels, carrier_params=cp)
        with tempfile.TemporaryDirectory() as d:
            fresh.save(Path(d))
            meta = json.loads((Path(d) / "metadata.json").read_text())
            assert meta["carrier_params"] == cp
            assert meta["version"] == "1.1.0"
            assert "sklearn_version" in meta
            reloaded = GhostGkModel.load(Path(d))
            assert reloaded.carrier_params == cp

    def test_load_backcompat_v1_0_0_falls_back_to_default(self):
        # The bundled "default" variant is a v1.0.0 artifact without carrier_params.
        model = GhostGkModel.from_variant("default")
        assert model.carrier_params == DEFAULT_CARRIER_PARAMS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/tracking/test_ghost_gk_r3.py::TestR3Metadata -v`
Expected: FAIL — `AttributeError: 'GhostGkModel' object has no attribute 'carrier_params'`.

- [ ] **Step 3: Add the import**

In `silly_kicks/tracking/_ghost_gk.py`, after line 31 (`from silly_kicks.spadl import config as spadlconfig`), add:

```python
from ._ball_carrier import DEFAULT_CARRIER_PARAMS
```

(Safe — `_ball_carrier` does not import `_ghost_gk`; no cycle. The constant is a plain dict, no heavy import cost.)

- [ ] **Step 4: Add the model attributes**

In `GhostGkModel.__init__` (after line 1254, `self._regressor_y = None`), add:

```python
        # R3 (PR-S81): carrier params used to compute the training team_in_possession,
        # recorded in metadata so serve resolves possession identically. Provenance
        # fields are populated by the trainer before save().
        self.carrier_params: dict = dict(DEFAULT_CARRIER_PARAMS)
        self.training_commit: str | None = None
        self.training_platform: str | None = None
```

- [ ] **Step 5: Record carrier_params in `fit`**

Change the `fit` signature (line 1256) to:

```python
    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        *,
        carrier_params: dict | None = None,
    ) -> GhostGkModel:
```

Immediately after the docstring (before `from sklearn.ensemble import HistGradientBoostingRegressor`), add:

```python
        self.carrier_params = dict(carrier_params) if carrier_params else dict(DEFAULT_CARRIER_PARAMS)
```

- [ ] **Step 6: Write the full metadata block in `save`**

Replace the `metadata = {...}` dict (lines 1515-1529) with:

```python
        import sklearn

        metadata = {
            "feature_names": GHOST_GK_FEATURE_NAMES,
            "grid_spec": {
                "x_min": GRID_X_MIN,
                "x_max": GRID_X_MAX,
                "y_min": GRID_Y_MIN,
                "y_max": GRID_Y_MAX,
                "nx": GRID_NX,
                "ny": GRID_NY,
                "resolution": GRID_RESOLUTION,
            },
            "n_estimators": self._n_estimators,
            "max_depth": self._max_depth,
            "carrier_params": self.carrier_params,
            "sklearn_version": sklearn.__version__,
            "training_commit": self.training_commit,
            "training_platform": self.training_platform,
            "version": "1.1.0",
        }
```

- [ ] **Step 7: Read provenance back in `load`**

After line 1606 (`model._training_leaves = training_leaves`), before `return model`, add:

```python
        model.carrier_params = metadata.get("carrier_params", dict(DEFAULT_CARRIER_PARAMS))
        model.training_commit = metadata.get("training_commit")
        model.training_platform = metadata.get("training_platform")
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `python -m pytest tests/tracking/test_ghost_gk_r3.py::TestR3Metadata -v`
Expected: PASS (4 tests).

- [ ] **Step 9: Sweep existing tests for the old version string / metadata shape**

Run: `python -m pytest tests/tracking/test_ghost_gk.py tests/tracking/test_ghost_gk_integration.py -v`
Expected: PASS. If any test asserts `version == "1.0.0"` or an exact metadata key set, update it to `"1.1.0"` / include `carrier_params` (grep first: `Grep "1.0.0" tests/tracking`). Fix inline, re-run.

- [ ] **Step 10: Commit**

```bash
git add silly_kicks/tracking/_ghost_gk.py tests/tracking/test_ghost_gk_r3.py
git commit -m "feat(ghost-gk): R3 carrier-param record/consume on GhostGkModel (PR-S81)"
```

---

## Task 2: `prepare_ghost_gk_training_data` additive `carrier_params` kwarg

**Files:**
- Modify: `silly_kicks/tracking/_ghost_gk.py` (`prepare_ghost_gk_training_data` signature ~810; carrier call 853-860)
- Test: `tests/tracking/test_ghost_gk_r3.py` (`TestPrepareCarrierParams`)

- [ ] **Step 1: Write the failing test**

Append to `tests/tracking/test_ghost_gk_r3.py`:

```python
from silly_kicks.tracking import prepare_ghost_gk_training_data
from silly_kicks.tracking._ball_carrier import infer_ball_carrier
from tests.tracking.test_ghost_gk import _make_ghost_gk_frames


def _frames_with_velocities(n_frames: int = 30) -> pd.DataFrame:
    """A short multi-frame sequence so prepare has enough rows to extract."""
    parts = []
    for fid in range(1, n_frames + 1):
        parts.append(_make_ghost_gk_frames(frame_id=fid, timestamp=float(fid)))
    return pd.concat(parts, ignore_index=True)


class TestPrepareCarrierParams:
    def test_prepare_accepts_carrier_params_and_stays_2_tuple(self):
        frames = _frames_with_velocities()
        result = prepare_ghost_gk_training_data(
            frames, home_team_id=1, carrier_params={"tolerance_m": 3.0, "beta": 0.0, "gamma": 0.25}
        )
        assert isinstance(result, tuple) and len(result) == 2  # N1: no public break
        features, labels = result
        assert list(features.columns) == list(__import__("silly_kicks.tracking._ghost_gk", fromlist=["GHOST_GK_FEATURE_NAMES"]).GHOST_GK_FEATURE_NAMES)

    def test_prepare_none_is_unchanged_from_bare_default(self):
        frames = _frames_with_velocities()
        f_none, _ = prepare_ghost_gk_training_data(frames, home_team_id=1, carrier_params=None)
        f_default, _ = prepare_ghost_gk_training_data(
            frames, home_team_id=1, carrier_params=dict(DEFAULT_CARRIER_PARAMS)
        )
        pd.testing.assert_frame_equal(f_none, f_default)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/tracking/test_ghost_gk_r3.py::TestPrepareCarrierParams -v`
Expected: FAIL — `TypeError: prepare_ghost_gk_training_data() got an unexpected keyword argument 'carrier_params'`.

- [ ] **Step 3: Add the kwarg and thread it**

Find the `prepare_ghost_gk_training_data` signature (~line 810). Add `carrier_params: dict | None = None` to its keyword-only args. Then change the carrier computation (currently lines 853-860):

```python
    from ._ball_carrier import infer_ball_carrier

    # Build context callbacks
    score_fn = _build_score_lookup(actions, home_team_id) if actions is not None else None
    phase_fn = _build_phase_lookup(actions) if actions is not None else None

    # Carrier (always computed --- only needs frames). carrier_params=None is
    # byte-identical to the historical bare call because DEFAULT_CARRIER_PARAMS
    # equals infer_ball_carrier's signature defaults (PR-S81 / R3 single source).
    cp = dict(carrier_params) if carrier_params else dict(DEFAULT_CARRIER_PARAMS)
    carrier_raw = infer_ball_carrier(frames, **cp)
```

(Keep `DEFAULT_CARRIER_PARAMS` referenced from the module-level import added in Task 1.)

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/tracking/test_ghost_gk_r3.py::TestPrepareCarrierParams -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add silly_kicks/tracking/_ghost_gk.py tests/tracking/test_ghost_gk_r3.py
git commit -m "feat(ghost-gk): prepare_ghost_gk_training_data accepts carrier_params (PR-S81/N1)"
```

---

## Task 3: `compute_ghost_gk` serve fix + `carrier=` passthrough

**Files:**
- Modify: `silly_kicks/tracking/_ghost_gk.py` (`compute_ghost_gk` signature 1660-1668; body 1716-1738)
- Test: `tests/tracking/test_ghost_gk_r3.py` (`TestServeCarrier`)

- [ ] **Step 1: Write the failing tests**

Append to `tests/tracking/test_ghost_gk_r3.py`:

```python
from unittest.mock import patch

from silly_kicks.tracking._ghost_gk import (
    _extract_all_ghost_gk_features,
    _resolve_model,
    compute_ghost_gk,
)


def _team_in_poss_column(frames, *, carrier):
    """Extract the internal feature matrix and return the team_in_possession Series."""
    feats, meta = _extract_all_ghost_gk_features(frames, home_team_id=1, carrier=carrier)
    return feats["team_in_possession"], meta


class TestServeCarrier:
    def test_serve_feature_matrix_has_real_team_in_poss(self):
        # P7: assert on the internal feature matrix, not just ghost_gk_x/y.
        # Ball at (50,34) coincides with away attacker a13 -> away team carries.
        frames = _make_ghost_gk_frames()
        model = GhostGkModel.from_variant("default")
        carrier = infer_ball_carrier(frames, **model.carrier_params)[
            ["game_id", "period_id", "frame_id", "ball_carrier_team_id"]
        ]
        tip_fixed, meta = _team_in_poss_column(frames, carrier=carrier)
        tip_bug, _ = _team_in_poss_column(frames, carrier=None)  # simulates the old serve bug

        away_gk = meta["gk_team_id"] == 2
        assert (tip_bug == 0.0).all()  # RED-equivalent: bug path is all-zero
        assert (tip_fixed[away_gk.values] == 1.0).all()  # away GK: its team is in possession

    def test_compute_ghost_gk_uses_carrier_by_default(self):
        # Post-fix: compute_ghost_gk internally computes the carrier (no carrier= passed).
        frames = _make_ghost_gk_frames()
        with patch("silly_kicks.tracking._ghost_gk.infer_ball_carrier", wraps=infer_ball_carrier) as spy:
            compute_ghost_gk(frames, model="default", home_team_id=1)
        assert spy.called  # serve no longer skips carrier inference

    def test_supplied_carrier_skips_internal_inference(self):
        # N5: a caller-supplied carrier bypasses the internal infer_ball_carrier call.
        frames = _make_ghost_gk_frames()
        model = GhostGkModel.from_variant("default")
        carrier = infer_ball_carrier(frames, **model.carrier_params)[
            ["game_id", "period_id", "frame_id", "ball_carrier_team_id"]
        ]
        with patch("silly_kicks.tracking._ghost_gk.infer_ball_carrier") as spy:
            compute_ghost_gk(frames, model=model, home_team_id=1, carrier=carrier)
        assert not spy.called  # passthrough avoids recomputation
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/tracking/test_ghost_gk_r3.py::TestServeCarrier -v`
Expected: FAIL — `test_compute_ghost_gk_uses_carrier_by_default` fails (serve never calls `infer_ball_carrier`); `test_supplied_carrier_skips_internal_inference` fails on `TypeError: unexpected keyword argument 'carrier'`. (Note: the test patches `silly_kicks.tracking._ghost_gk.infer_ball_carrier`, which requires a module-level name — created in Step 3.)

- [ ] **Step 3: Add the module-level import + signature + body**

The current `compute_ghost_gk` imports `infer_ball_carrier` nowhere; the spy patches a module attribute, so import it at the **top of `compute_ghost_gk`** as a module-visible name is not enough for `patch`. Instead add a module-level lazy binding: at the top of `_ghost_gk.py` import block (after the Task-1 `DEFAULT_CARRIER_PARAMS` import), add:

```python
from ._ball_carrier import infer_ball_carrier
```

(P9 note: `infer_ball_carrier` pulls the numba try/except in `_ball_carrier`, already imported transitively by the `DEFAULT_CARRIER_PARAMS` import in Task 1, so this adds no new heavy top-level cost.)

Change the `compute_ghost_gk` signature (1660-1668) to add `carrier`:

```python
def compute_ghost_gk(
    frames: pd.DataFrame,
    *,
    model: GhostGkModel | GhostGkVariant | None = None,
    home_team_id: int | str,
    actions: pd.DataFrame | None = None,
    carrier: pd.DataFrame | None = None,
    link_frame_ids: set[int] | None = None,
    kde_backend: str = "vectorized",
) -> pd.DataFrame:
```

Replace the body region (current lines 1716-1738, from `resolved = _resolve_model(model)` through the `_extract_all_ghost_gk_features(...)` call) with:

```python
    resolved = _resolve_model(model)
    out = frames.copy()
    out["ghost_gk_x"] = np.nan
    out["ghost_gk_y"] = np.nan
    out["ghost_gk_spread"] = np.nan

    # Build context callbacks from actions
    score_fn = _build_score_lookup(actions, home_team_id) if actions is not None else None
    phase_fn = _build_phase_lookup(actions) if actions is not None else None

    # PR-S81 serve-carrier consistency: compute the carrier on FULL frames with the
    # model's recorded carrier_params (R3) so team_in_possession matches training.
    # The carrier lookup in _extract_one_frame is per-(game,period,frame) independent,
    # so restricting extraction to link_frame_ids stays byte-identical for kept frames
    # (mirrors xS _xshot_occurrence.py:776-781). A caller may supply a precomputed
    # `carrier` (computed on full frames with this model's carrier_params) to skip the
    # internal inference (N5 cache convention; mirrors `links`).
    if carrier is None:
        carrier_raw = infer_ball_carrier(frames, **resolved.carrier_params)
        carrier = carrier_raw[["game_id", "period_id", "frame_id", "ball_carrier_team_id"]]

    batch_features, meta = _extract_all_ghost_gk_features(
        frames,
        home_team_id=home_team_id,
        carrier=carrier,
        score_at_time=score_fn,
        phase_at_time=phase_fn,
        link_frame_ids=link_frame_ids,
    )
```

- [ ] **Step 4: Document the new kwarg**

Add to the `compute_ghost_gk` docstring Parameters (after `actions`):

```
    carrier : pd.DataFrame | None, default None
        Optional precomputed carrier — the
        ``["game_id","period_id","frame_id","ball_carrier_team_id"]`` projection of
        ``infer_ball_carrier(frames, **model.carrier_params)`` on the FULL frames. When
        None, computed internally. Supply it to avoid recomputation across repeated
        ghost-GK calls or across families that also resolve possession (mirrors ``links``).
        Must be computed on full frames with the model's carrier_params for the
        byte-identical frame-restriction invariant to hold.
```

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/tracking/test_ghost_gk_r3.py::TestServeCarrier -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Add the train==serve feature-parity test (P7 test #2)**

Append to `TestServeCarrier`:

```python
    def test_train_serve_feature_parity(self):
        # P7 test #2: team_in_possession extracted via prepare (train) == via
        # compute_ghost_gk's internal extraction (serve) on the same frames + params.
        frames = _frames_with_velocities()
        cp = dict(DEFAULT_CARRIER_PARAMS)
        train_feats, _ = prepare_ghost_gk_training_data(frames, home_team_id=1, carrier_params=cp)
        carrier = infer_ball_carrier(frames, **cp)[
            ["game_id", "period_id", "frame_id", "ball_carrier_team_id"]
        ]
        serve_feats, _ = _extract_all_ghost_gk_features(frames, home_team_id=1, carrier=carrier)
        # prepare drops NaN-label rows; align on the shared frames by comparing the
        # distribution of team_in_possession (both must be non-trivially non-zero).
        assert serve_feats["team_in_possession"].sum() > 0
        assert train_feats["team_in_possession"].sum() > 0
```

Run: `python -m pytest tests/tracking/test_ghost_gk_r3.py::TestServeCarrier::test_train_serve_feature_parity -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add silly_kicks/tracking/_ghost_gk.py tests/tracking/test_ghost_gk_r3.py
git commit -m "fix(ghost-gk): serve computes carrier (team_in_poss no longer 0) + carrier passthrough (PR-S81)"
```

---

## Task 4: thread `carrier` through `add_ghost_gk` + `ghost_gk_xfns`

**Files:**
- Modify: `silly_kicks/tracking/features.py` (`add_ghost_gk` 3671-3752; `ghost_gk_xfns` 3791-3847)
- Test: `tests/tracking/test_ghost_gk_r3.py` (`TestAggregatorCarrierPassthrough`)

- [ ] **Step 1: Write the failing test**

Append to `tests/tracking/test_ghost_gk_r3.py`:

```python
from silly_kicks.tracking.features import add_ghost_gk


class TestAggregatorCarrierPassthrough:
    def test_add_ghost_gk_accepts_and_forwards_carrier(self):
        frames = _make_ghost_gk_frames()
        actions = pd.DataFrame(
            {
                "action_id": [0],
                "game_id": ["100"],
                "period_id": [1],
                "team_id": [1],  # home acts -> defending GK is away
                "start_x": [50.0],
                "start_y": [34.0],
                "time_seconds": [1.0],
            }
        )
        model = GhostGkModel.from_variant("default")
        carrier = infer_ball_carrier(frames, **model.carrier_params)[
            ["game_id", "period_id", "frame_id", "ball_carrier_team_id"]
        ]
        with patch("silly_kicks.tracking._ghost_gk.infer_ball_carrier") as spy:
            add_ghost_gk(actions, frames, model=model, home_team_id=1, carrier=carrier)
        assert not spy.called  # carrier forwarded to compute_ghost_gk
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/tracking/test_ghost_gk_r3.py::TestAggregatorCarrierPassthrough -v`
Expected: FAIL — `TypeError: add_ghost_gk() got an unexpected keyword argument 'carrier'`.

- [ ] **Step 3: Add `carrier` to `add_ghost_gk`**

Add `carrier: pd.DataFrame | None = None` to the `add_ghost_gk` keyword-only signature (after `links`, line 3676). In its `compute_ghost_gk(...)` call (lines 3745-3752), add `carrier=carrier,`:

```python
        ghost_frames = compute_ghost_gk(
            frames,
            model=resolved_model,
            home_team_id=home_team_id,
            actions=actions_for_context,
            carrier=carrier,
            link_frame_ids=link_frame_ids,
            kde_backend=kde_backend,
        )
```

- [ ] **Step 4: Add `carrier` to `ghost_gk_xfns`**

Add `carrier: pd.DataFrame | None = None` to the `ghost_gk_xfns` signature (line 3791). In the inner `compute_ghost_gk(...)` call (lines 3841-3847), add `carrier=carrier,`:

```python
        ghost_frames = compute_ghost_gk(
            frames,
            model=resolved,
            home_team_id=home_team_id,
            carrier=carrier,
            link_frame_ids=link_frame_ids,
            kde_backend=kde_backend,
        )
```

Add a one-line Parameters doc note to both: `carrier : pd.DataFrame | None — optional precomputed carrier forwarded to compute_ghost_gk (see its docstring).`

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/tracking/test_ghost_gk_r3.py::TestAggregatorCarrierPassthrough -v`
Expected: PASS.

- [ ] **Step 6: Public-API examples gate**

Run: `python -m pytest tests/test_public_api_examples.py -v`
Expected: PASS (signatures changed but Examples already present; no new public symbols).

- [ ] **Step 7: Commit**

```bash
git add silly_kicks/tracking/features.py tests/tracking/test_ghost_gk_r3.py
git commit -m "feat(ghost-gk): carrier passthrough on add_ghost_gk + ghost_gk_xfns (PR-S81/N5)"
```

---

## Task 5: frame-restriction byte-identical (carrier active) + N2 recorded==used-and-flips test

**Files:**
- Modify: `tests/tracking/test_ghost_gk_frame_restriction.py` (extend `TestExtractionRestriction`)
- Test: `tests/tracking/test_ghost_gk_r3.py` (`TestRecordedEqualsUsed`)

- [ ] **Step 1: Write the byte-identical-with-carrier test**

In `tests/tracking/test_ghost_gk_frame_restriction.py`, add to `TestExtractionRestriction` a case that runs full vs `link_frame_ids`-restricted `compute_ghost_gk` (now with the carrier active by default) and asserts the kept-frame predictions are identical:

```python
    def test_restriction_byte_identical_with_carrier_active(self):
        frames = _multi_frame_fixture()  # reuse the module's existing builder
        full = compute_ghost_gk(frames, model="default", home_team_id=1)
        kept = {int(frames["frame_id"].iloc[0])}
        restricted = compute_ghost_gk(
            frames, model="default", home_team_id=1, link_frame_ids=kept
        )
        gk = full["is_goalkeeper"].astype(bool) & ~full["is_ball"].astype(bool)
        m = gk & full["frame_id"].isin(kept) & full["ghost_gk_x"].notna()
        pd.testing.assert_series_equal(
            full.loc[m, "ghost_gk_x"], restricted.loc[m, "ghost_gk_x"], check_names=False
        )
```

(If `_multi_frame_fixture` is not the existing helper name, reuse whatever the file already uses to build multi-frame input — read the file's top to confirm the builder.)

- [ ] **Step 2: Run it (should already pass — invariant holds)**

Run: `python -m pytest tests/tracking/test_ghost_gk_frame_restriction.py -v`
Expected: PASS. The carrier is computed on full frames in BOTH calls, so kept-frame features are identical. If it FAILS, the carrier is being computed on the restricted subset somewhere — re-check Task 3 (`carrier` must use full `frames`).

- [ ] **Step 3: Write the N2 "recorded == used AND the flip bites" test**

Append to `tests/tracking/test_ghost_gk_r3.py`. Construct a near-tie frame: a home and an away outfield player equidistant from the ball, the away one moving toward it, so `beta=0.9` flips the carrier to away vs the distance-only `beta=0.0` pick.

```python
class TestRecordedEqualsUsed:
    @staticmethod
    def _near_tie_frames() -> pd.DataFrame:
        base = {
            "game_id": "100", "period_id": 1, "frame_id": 1, "timestamp": 1.0,
            "ball_state": "alive", "time_seconds": 1.0, "source_provider": "test",
        }
        def row(pid, team, x, y, vx, vy, ball=False, gk=False):
            return {**base, "player_id": pid, "team_id": team, "x": x, "y": y,
                    "vx": vx, "vy": vy, "speed": (vx**2 + vy**2) ** 0.5,
                    "is_ball": ball, "is_goalkeeper": gk}
        rows = [
            row("ball", None, 50.0, 34.0, 0.0, 0.0, ball=True),
            # home player: 1.9m left of ball, stationary -> UNAMBIGUOUSLY closer (NP4:
            # removes tiebreak dependence so beta=0.0 deterministically picks home).
            row("h1", 1, 48.1, 34.0, 0.0, 0.0),
            # away player: 2.0m right of ball (slightly farther), moving LEFT toward it
            # -> beta=0.9 flips the pick to away purely via velocity, not distance.
            row("a1", 2, 52.0, 34.0, -3.0, 0.0),
            row("h_gk", 1, 5.0, 34.0, 0.0, 0.0, gk=True),
            row("a_gk", 2, 100.0, 34.0, 0.0, 0.0, gk=True),
        ]
        return pd.DataFrame(rows)

    def test_carrier_params_flip_the_carrier_and_are_recorded(self):
        frames = self._near_tie_frames()
        c_dist = infer_ball_carrier(frames, tolerance_m=3.0, beta=0.0, gamma=0.25)
        c_vel = infer_ball_carrier(frames, tolerance_m=3.0, beta=0.9, gamma=0.25)
        tid = lambda c: c["ball_carrier_team_id"].iloc[0]
        assert tid(c_dist) != tid(c_vel)  # the fixture genuinely makes beta bite (N2)

        # recorded == used: prepare uses beta=0.9, model records beta=0.9
        cp = {"tolerance_m": 3.0, "beta": 0.9, "gamma": 0.25}
        seq = pd.concat([self._near_tie_frames().assign(frame_id=i, time_seconds=float(i))
                         for i in range(1, 25)], ignore_index=True)
        feats, labels = prepare_ghost_gk_training_data(seq, home_team_id=1, carrier_params=cp)
        m = GhostGkModel(n_estimators=10)
        m.fit(feats, labels, carrier_params=cp)
        assert m.carrier_params == cp
```

- [ ] **Step 4: Run to verify**

Run: `python -m pytest tests/tracking/test_ghost_gk_r3.py::TestRecordedEqualsUsed -v`
Expected: PASS. If `assert tid(c_dist) != tid(c_vel)` fails, the fixture is not a genuine near-tie — adjust the distances so both players are equidistant (the assertion guards against a vacuous test per N2).

- [ ] **Step 5: Full ghost-GK suite + lint**

Run: `python -m pytest tests/tracking/ -k ghost_gk -v` then `ruff check silly_kicks/ && ruff format --check silly_kicks/ && pyright silly_kicks/`
Expected: all PASS / clean.

- [ ] **Step 6: Commit**

```bash
git add tests/tracking/test_ghost_gk_frame_restriction.py tests/tracking/test_ghost_gk_r3.py
git commit -m "test(ghost-gk): frame-restriction byte-identical with carrier + N2 flip-bites (PR-S81)"
```

---

## Task 6: trainer single-`cp` wiring + provenance + variant CLI

**Files:**
- Modify: `scripts/train_ghost_gk.py`

- [ ] **Step 1: Add CLI args**

In `parse_args` (after line 51, `--cv-folds`), add:

```python
    parser.add_argument("--carrier-beta", type=float, default=None, help="Carrier velocity weight (default: library DEFAULT_CARRIER_PARAMS)")
    parser.add_argument("--carrier-gamma", type=float, default=None, help="Carrier hysteresis (default: library)")
    parser.add_argument("--carrier-tolerance", type=float, default=None, help="Carrier radius m (default: library)")
    parser.add_argument("--variant", choices=["default", "full"], default="full", help="Which variant this run produces (affects --subsample-cap)")
    parser.add_argument("--subsample-cap", type=int, default=None, help="Cap total training samples (default: None=all; ~36000 for the bundled 'default')")
    parser.add_argument("--training-platform", type=str, default=None, help="Recorded in metadata (e.g. 'dgx-spark-aarch64')")
```

- [ ] **Step 2: Resolve one `cp` and record provenance source**

In `main`, after `args = parse_args()` (line 59), add:

```python
    import subprocess

    from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS

    cp = dict(DEFAULT_CARRIER_PARAMS)
    if args.carrier_tolerance is not None:
        cp["tolerance_m"] = args.carrier_tolerance
    if args.carrier_beta is not None:
        cp["beta"] = args.carrier_beta
    if args.carrier_gamma is not None:
        cp["gamma"] = args.carrier_gamma
    print(f"Carrier params (single source, recorded + used): {cp}")

    try:
        training_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        training_commit = None
    print(f"training_commit={training_commit}, training_platform={args.training_platform}")
```

- [ ] **Step 3: Pass `cp` into `prepare`**

In the per-game extraction loop, change the `prepare_ghost_gk_training_data(...)` call (lines 191-196) to add `carrier_params=cp`:

```python
                feats, labs = prepare_ghost_gk_training_data(
                    game_frames,
                    home_team_id=home,
                    actions=game_actions,
                    subsample_fps=args.subsample_fps,
                    carrier_params=cp,
                )
```

- [ ] **Step 4: Apply the subsample cap (variant axis)**

After `features = pd.concat(all_features, ...)` / cache-load (after line 233, both branches converge), add:

```python
    if args.subsample_cap is not None and len(features) > args.subsample_cap:
        rng = np.random.default_rng(42)
        keep = rng.choice(len(features), size=args.subsample_cap, replace=False)
        keep.sort()
        features = features.iloc[keep].reset_index(drop=True)
        labels = labels.iloc[keep].reset_index(drop=True)
        groups = groups[keep]
        provider_labels = provider_labels[keep]
        print(f"Subsampled to {len(features)} samples (variant={args.variant}, cap={args.subsample_cap})")
```

- [ ] **Step 5: Pass `cp` into every `fit` + set provenance on the final model**

Change the CV-fold `model.fit(X_train, y_train)` (line 260) to `model.fit(X_train, y_train, carrier_params=cp)`. Change the final model fit (line 326) to `final_model.fit(features, labels, carrier_params=cp)`. Immediately before `final_model.save(artifact_dir)` (line 364), add:

```python
    final_model.training_commit = training_commit
    final_model.training_platform = args.training_platform
```

- [ ] **Step 6: Extend the round-trip verify + metrics to cover provenance**

After the round-trip loop (line 378), add:

```python
    assert loaded.carrier_params == cp, f"carrier_params drift: {loaded.carrier_params} != {cp}"
    print(f"R3 round-trip: carrier_params={loaded.carrier_params} OK")
```

In the `metrics` dict (line 392), add `"carrier_params": cp, "variant": args.variant, "training_commit": training_commit`.

- [ ] **Step 7: Smoke-run the trainer on a tiny synthetic dir**

Run (build a 2-game tiny parquet dir first, or reuse `tests/datasets/tracking/`):
```bash
python scripts/train_ghost_gk.py --data-dir tests/datasets/tracking --output-dir /tmp/ggk_smoke --n-estimators 10 --cv-folds 2 --carrier-beta 0.0 --carrier-gamma 0.25 --training-platform smoke 2>&1 | tail -20
```
Expected: prints `Carrier params ... {'tolerance_m': 3.0, 'beta': 0.0, 'gamma': 0.25}`, `R3 round-trip ... OK`, and writes `metadata.json` with `carrier_params` + `version: 1.1.0`. (If the test dataset layout differs, point `--data-dir` at any dir with ≥2 valid frames parquets + meta.json.)

- [ ] **Step 8: Write a committed CLI test for the single-cp seam (NP3)**

The N1 invariant lives where the trainer wires `cp` to both `prepare` and `fit` — unit-test it, don't
only smoke it. Create `tests/scripts/test_train_ghost_gk_cli.py`:

```python
"""Trainer CLI wires one carrier cp into prepare + fit (PR-S81 / N1 / NP3)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS
from tests.tracking.test_ghost_gk import _make_ghost_gk_frames


def _tiny_cache(root: Path, n_games: int = 2) -> None:
    for g in range(n_games):
        gid = f"{100 + g}"
        gdir = root / "test" / gid
        gdir.mkdir(parents=True, exist_ok=True)
        frames = pd.concat(
            [_make_ghost_gk_frames(game_id=gid, frame_id=f, timestamp=float(f)) for f in range(1, 40)],
            ignore_index=True,
        )
        frames.to_parquet(gdir / "frames.parquet")
        (gdir / "meta.json").write_text(json.dumps({"home_team_id": 1}))


def _run(tmp_path: Path, *extra: str) -> dict:
    data = tmp_path / "cache"
    data.mkdir()
    _tiny_cache(data)
    out = tmp_path / "out"
    subprocess.run(
        [sys.executable, "scripts/train_ghost_gk.py", "--data-dir", str(data),
         "--output-dir", str(out), "--n-estimators", "10", "--cv-folds", "2", *extra],
        check=True, capture_output=True, text=True,
    )
    return json.loads((out / "ghost_gk_v1" / "metadata.json").read_text())


def test_cli_records_supplied_carrier_params(tmp_path):
    meta = _run(tmp_path, "--carrier-beta", "0.9", "--carrier-gamma", "0.3", "--carrier-tolerance", "2.5")
    assert meta["carrier_params"] == {"tolerance_m": 2.5, "beta": 0.9, "gamma": 0.3}
    assert meta["version"] == "1.1.0"


def test_cli_omitted_records_library_default(tmp_path):
    meta = _run(tmp_path)
    assert meta["carrier_params"] == dict(DEFAULT_CARRIER_PARAMS)
```

Run: `python -m pytest tests/scripts/test_train_ghost_gk_cli.py -v`
Expected: PASS (2 tests). (Create `tests/scripts/__init__.py` if the package marker is missing.)

- [ ] **Step 9: Commit**

```bash
git add scripts/train_ghost_gk.py tests/scripts/test_train_ghost_gk_cli.py
git commit -m "feat(ghost-gk): trainer single-cp carrier wiring + provenance + variant cap + CLI test (PR-S81)"
```

---

## Task 7: P3 measurement script + P2/N4 gate script + atomic verification

**Files:**
- Create: `scripts/measure_ghost_gk_serve_delta.py`, `scripts/validate_ghost_gk_refit.py`
- Test: `tests/tracking/test_ghost_gk_r3.py` (atomic mirror)

- [ ] **Step 1: Write the serve-delta measurement script (P3)**

Create `scripts/measure_ghost_gk_serve_delta.py`:

```python
#!/usr/bin/env python
"""Measure the buggy-vs-fixed Ghost-GK serve delta (PR-S81 / P3).

The "buggy" serve passed no carrier -> team_in_possession == 0 everywhere.
The "fixed" serve computes the carrier on full frames. This script runs the
internal extractor both ways on a real match, predicts both, and reports the
max/median |ghost_gk_x/y| delta in metres -- the real number for the CHANGELOG
and the lakehouse heads-up (NOT the word "small").

Usage: python scripts/measure_ghost_gk_serve_delta.py --frames match.parquet --home-team-id 1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from silly_kicks.tracking._ball_carrier import infer_ball_carrier
from silly_kicks.tracking._ghost_gk import (
    _extract_all_ghost_gk_features,
    _resolve_model,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--home-team-id", required=True)
    ap.add_argument("--variant", default="default")
    args = ap.parse_args()

    frames = pd.read_parquet(args.frames)
    model = _resolve_model(args.variant)
    carrier = infer_ball_carrier(frames, **model.carrier_params)[
        ["game_id", "period_id", "frame_id", "ball_carrier_team_id"]
    ]
    feats_fixed, meta = _extract_all_ghost_gk_features(frames, home_team_id=args.home_team_id, carrier=carrier)
    feats_bug, _ = _extract_all_ghost_gk_features(frames, home_team_id=args.home_team_id, carrier=None)

    pred_fixed = model.predict(feats_fixed)
    pred_bug = model.predict(feats_bug)
    dx = np.abs(pred_fixed[:, 0] - pred_bug[:, 0])
    dy = np.abs(pred_fixed[:, 1] - pred_bug[:, 1])
    d = np.sqrt(dx**2 + dy**2)
    n_changed = int((d > 1e-9).sum())
    print(f"samples={len(d)}  changed={n_changed} ({100*n_changed/len(d):.1f}%)")
    print(f"euclidean delta (m): max={d.max():.3f}  median={np.median(d):.3f}  mean={d.mean():.3f}")
    print(f"x delta (m): max={dx.max():.3f}  median={np.median(dx):.3f}")
    print(f"y delta (m): max={dy.max():.3f}  median={np.median(dy):.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the apples-to-apples gate script (P2/N4)**

Create `scripts/validate_ghost_gk_refit.py`:

```python
#!/usr/bin/env python
"""Apples-to-apples non-regression gate for the Ghost-GK re-fit (PR-S81 / P2/N4).

Evaluates BOTH the re-fit model and the incumbent model on the SAME held-out
folds (the re-fit's CV split), so the MAE delta isolates "different model" from
"different data". Prints per-fold + aggregate MAE for both, and the gate verdict.

N4: keeping the incumbent is AVAILABILITY-safe, not correctness-safe. If the gate
rejects the re-fit, this script prints a REJECT verdict; the operator must record
why and file a bronze-scale-refresh follow-up rather than declaring staleness fixed.

Usage:
    python scripts/validate_ghost_gk_refit.py \
        --features /path/_feature_cache/features.parquet \
        --labels   /path/_feature_cache/labels.parquet \
        --groups   /path/_feature_cache/groups.npy \
        --providers /path/_feature_cache/providers.npy \
        --refit-dir models/ghost_gk_v1 \
        --incumbent-variant default \
        --epsilon 0.02 --cv-folds 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from silly_kicks.tracking._ghost_gk import GhostGkModel


def _euclid_mae(model: GhostGkModel, X: pd.DataFrame, y: pd.DataFrame) -> float:
    pred = model.predict(X)  # KDE mode; load()ed models lack predict_mean
    return float(np.mean(np.sqrt((pred[:, 0] - y["gk_x"].values) ** 2 + (pred[:, 1] - y["gk_y"].values) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--groups", type=Path, required=True)
    ap.add_argument("--providers", type=Path, required=True)
    ap.add_argument("--refit-dir", type=Path, required=True)
    ap.add_argument("--incumbent-variant", default="default")
    ap.add_argument("--epsilon", type=float, default=0.02)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--max-depth", type=int, default=8)
    args = ap.parse_args()

    features = pd.read_parquet(args.features)
    labels = pd.read_parquet(args.labels)
    groups = np.load(args.groups, allow_pickle=True)
    provs = np.load(args.providers, allow_pickle=True)
    incumbent = GhostGkModel.from_variant(args.incumbent_variant)

    cv = StratifiedGroupKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    refit_maes, inc_maes = [], []
    for fold, (tr, te) in enumerate(cv.split(features, provs, groups)):
        Xte, yte = features.iloc[te], labels.iloc[te]
        # re-fit model for this fold (mirrors trainer CV exactly)
        rf = GhostGkModel(n_estimators=args.n_estimators, max_depth=args.max_depth)
        rf.fit(features.iloc[tr], labels.iloc[tr])
        rf_mae = _euclid_mae(rf, Xte, yte)
        inc_mae = _euclid_mae(incumbent, Xte, yte)  # SAME held-out -> apples-to-apples
        refit_maes.append(rf_mae)
        inc_maes.append(inc_mae)
        print(f"fold {fold+1}: refit={rf_mae:.3f}m  incumbent={inc_mae:.3f}m  delta={rf_mae-inc_mae:+.3f}m")

    rf_mean, inc_mean = float(np.mean(refit_maes)), float(np.mean(inc_maes))
    verdict = "SHIP REFIT" if rf_mean <= inc_mean + args.epsilon else "KEEP INCUMBENT"
    print(f"\nAGGREGATE: refit={rf_mean:.3f}m  incumbent={inc_mean:.3f}m  eps={args.epsilon}")
    print(f"VERDICT: {verdict}")
    print("L3 caveat: if held-out overlaps the incumbent's TRAINING corpus, incumbent MAE is optimistic.")
    if verdict == "KEEP INCUMBENT":
        print("N4: availability-safe only. Record why + file a bronze-scale refresh follow-up; do NOT declare staleness resolved.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write the atomic-mirror test**

Append to `tests/tracking/test_ghost_gk_r3.py`:

```python
class TestAtomicMirror:
    def test_atomic_add_ghost_gk_inherits_carrier_kwarg(self):
        import inspect

        from silly_kicks.atomic.tracking.features import add_ghost_gk as atomic_add
        from silly_kicks.tracking.features import add_ghost_gk as std_add

        assert atomic_add is std_add  # re-export, not a copy
        assert "carrier" in inspect.signature(atomic_add).parameters
```

- [ ] **Step 4: Run the atomic test + full ghost-GK suite**

Run: `python -m pytest tests/tracking/test_ghost_gk_r3.py tests/tracking/ -k ghost_gk -v`
Expected: PASS.

- [ ] **Step 5: Lint + typecheck the scripts**

Run: `ruff check scripts/ silly_kicks/ && ruff format --check scripts/ silly_kicks/ && pyright silly_kicks/`
Expected: clean. (Scripts are not in `pyright silly_kicks/` scope but must pass ruff.)

- [ ] **Step 6: Commit**

```bash
git add scripts/measure_ghost_gk_serve_delta.py scripts/validate_ghost_gk_refit.py tests/tracking/test_ghost_gk_r3.py
git commit -m "feat(ghost-gk): serve-delta measurement + apples-to-apples gate scripts (PR-S81)"
```

---

## Task 8: Build the pining→cache writer + verify the trainer layout contract (NP1 — prereq before the long job)

**Files:**
- Create: `scripts/_loader_pining_to_cache.py`
- Test: `tests/scripts/test_loader_pining_to_cache.py`

**Why in Part 1:** the expensive ~81-match SSH re-fit (Task 9) depends on this writer producing exactly
the layout `train_ghost_gk.py --data-dir` consumes. A mismatch must surface here (cheap, synthetic),
not after the multi-hour pull. Verified contract (from `scripts/train_ghost_gk.py:70` + `:123-134`):
the trainer reads `**/frames.parquet`, takes `home_team_id` from a sibling `meta.json`, and keys it by
both the parent dir name AND the `game_id` column. `scripts/_loader_pining.load_matches` yields
`(provider, match_id, actions, frames, home)` with `frames` already `smooth_frames`+`derive_velocities`
(so `vx`/`vy` present — the trainer's hard requirement at `:97`).

- [ ] **Step 1: Write the failing round-trip test**

Create `tests/scripts/test_loader_pining_to_cache.py`:

```python
"""The pining->cache writer produces a layout train_ghost_gk.py consumes (PR-S81 / NP1)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

from tests.tracking.test_ghost_gk import _make_ghost_gk_frames

_spec = importlib.util.spec_from_file_location(
    "_loader_pining_to_cache", Path("scripts/_loader_pining_to_cache.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]


def test_write_cache_layout_is_trainer_consumable(tmp_path):
    frames = pd.concat(
        [_make_ghost_gk_frames(game_id="g1", frame_id=f, timestamp=float(f)) for f in range(1, 5)],
        ignore_index=True,
    )
    actions = pd.DataFrame({"action_id": [0], "game_id": ["g1"], "period_id": [1], "team_id": [1]})

    _mod.write_match_cache(tmp_path, provider="test", match_id="g1", frames=frames, actions=actions, home_team_id=1)

    fp = tmp_path / "test" / "g1" / "frames.parquet"
    mp = tmp_path / "test" / "g1" / "meta.json"
    assert fp.exists() and mp.exists()
    assert json.loads(mp.read_text())["home_team_id"] == 1
    cols = set(pd.read_parquet(fp).columns)
    assert {"vx", "vy", "game_id", "is_goalkeeper"} <= cols  # trainer-required schema
    # the trainer's discovery glob finds it:
    assert list(tmp_path.glob("**/frames.parquet")) == [fp]
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/scripts/test_loader_pining_to_cache.py -v`
Expected: FAIL — `FileNotFoundError`/`ModuleNotFoundError` (script does not exist yet).

- [ ] **Step 3: Write the cache writer**

Create `scripts/_loader_pining_to_cache.py`:

```python
#!/usr/bin/env python
"""Stream pining matches -> per-match tc3-layout cache for train_ghost_gk.py (PR-S81).

Writes {out}/{provider}/{match_id}/frames.parquet + meta.json (home_team_id), and
optional actions to {out}/_actions/{match_id}.parquet. Frames carry vx/vy because
_loader_pining yields smooth_frames+derive_velocities output.

Usage:
    set -a; source ~/.pining_env; set +a
    python scripts/_loader_pining_to_cache.py --providers skillcorner idsse gradientsports \
        --out ~/Development/ghost_gk_refit/cache
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def write_match_cache(
    out: Path, *, provider: str, match_id: str, frames: pd.DataFrame,
    actions: pd.DataFrame | None, home_team_id: object,
) -> None:
    gdir = out / provider / str(match_id)
    gdir.mkdir(parents=True, exist_ok=True)
    frames.to_parquet(gdir / "frames.parquet")
    (gdir / "meta.json").write_text(json.dumps({"home_team_id": home_team_id}))
    if actions is not None and len(actions) > 0:
        adir = out / "_actions"
        adir.mkdir(parents=True, exist_ok=True)
        actions.to_parquet(adir / f"{match_id}.parquet")


def main() -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    from _loader_pining import load_matches  # noqa: E402

    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-per-provider", type=int, default=None)
    ap.add_argument("--tracking-limit", type=int, default=None)
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

    n = 0
    for provider, match_id, actions, frames, home in load_matches(
        providers=args.providers,
        max_per_provider=args.max_per_provider,
        tracking_limit=args.tracking_limit,
    ):
        if "vx" not in frames.columns or "vy" not in frames.columns:
            print(f"  SKIP {provider}/{match_id}: no vx/vy", file=sys.stderr)
            continue
        write_match_cache(args.out, provider=provider, match_id=match_id,
                          frames=frames, actions=actions, home_team_id=home)
        n += 1
        print(f"  [{n}] cached {provider}/{match_id}: {len(frames)} rows")
    print(f"Done: cached {n} matches to {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/scripts/test_loader_pining_to_cache.py -v`
Expected: PASS.

- [ ] **Step 5: Lint**

Run: `ruff check scripts/ && ruff format --check scripts/`
Expected: clean.

- [ ] **Step 6: Commit — THIS is the SHA the box trains against (P6/N3)**

```bash
git add scripts/_loader_pining_to_cache.py tests/scripts/test_loader_pining_to_cache.py
git commit -m "feat(ghost-gk): pining->cache writer + trainer-layout round-trip test (PR-S81/NP1)"
git rev-parse HEAD   # record this SHA -> training_commit for Part 2
```

Record the printed SHA — it is the **pre-squash PR-S81 branch SHA** the re-fit trains against and embeds as `training_commit` (N3: it will not be an ancestor of main after the final squash).

---

# PART 2 — Re-fit execution (operational; maintainer drives DGX Spark over SSH)

## Task 9: Re-pull corpus + run the re-fit (full + default variants)

> This task runs on `karsten@192.168.68.73` (DGX Spark, ARM aarch64, 119 GiB). See [[reference_dgx_spark]] + the PR-S80 deploy recipe. Long-running — launch under `nohup`, poll `tail` (per the background-task-polling policy; do NOT block).

- [ ] **Step 1: Sync the box repo to the Part-1 SHA**

```bash
ssh karsten@192.168.68.73
cd ~/Development/silly-kicks
git fetch origin
git checkout pr-s81-ghost-gk-refit   # or fetch the branch ref; must be the Task-7 SHA
git rev-parse HEAD                    # MUST equal the recorded training_commit
```

- [ ] **Step 2: Ensure the venv has the stack**

```bash
source ~/sk-phaseb-venv/bin/activate
pip install -e ".[train,xgboost,kloppy,ghost-gk]" pyarrow
python -c "import sklearn, silly_kicks; print(sklearn.__version__, silly_kicks.__version__)"
```

- [ ] **Step 3: Re-pull ~81 matches via pining (owner token) using the Task-8 writer**

```bash
set -a; source ~/.pining_env; set +a   # PINING_FOR_THE_DATA_TOKEN
nohup python scripts/_loader_pining_to_cache.py --providers skillcorner idsse gradientsports \
    --out ~/Development/ghost_gk_refit/cache > ~/Development/ghost_gk_refit/pull.log 2>&1 &
```

`_loader_pining_to_cache.py` (built + round-trip-tested in Task 8) writes the exact
`{provider}/{match_id}/frames.parquet` + `meta.json` layout the trainer consumes, with `vx`/`vy`
already present (frames come from `_loader_pining`'s `smooth_frames`+`derive_velocities`). Actions land
in `{out}/cache/_actions/` — pass `--actions-dir ~/Development/ghost_gk_refit/cache/_actions` to the
trainer in Steps 4-5 so score/phase context is available.

Poll: `tail -f ~/Development/ghost_gk_refit/pull.log` until all matches cached.

- [ ] **Step 4: Run the `full` variant re-fit**

```bash
cd ~/Development/silly-kicks
nohup python scripts/train_ghost_gk.py \
    --data-dir ~/Development/ghost_gk_refit/cache \
    --actions-dir ~/Development/ghost_gk_refit/cache/_actions \
    --output-dir ~/Development/ghost_gk_refit/full \
    --variant full --subsample-fps 1.0 --n-estimators 500 --max-depth 8 --cv-folds 5 \
    --carrier-beta 0.0 --carrier-gamma 0.25 --carrier-tolerance 3.0 \
    --training-platform dgx-spark-aarch64 \
    > ~/Development/ghost_gk_refit/full.log 2>&1 &
```

Poll `tail -f ~/Development/ghost_gk_refit/full.log`. Verify the printed `Carrier params ... {'tolerance_m': 3.0, 'beta': 0.0, 'gamma': 0.25}`, `R3 round-trip OK`, and the CV MAE summary.

- [ ] **Step 5: Run the `default` (bundled) variant re-fit**

Same command with `--output-dir .../default --variant default --subsample-cap 36000` (matches the incumbent default's ~36k samples / ~9 MB wheel budget). Verify the artifact size with `du -sh ~/Development/ghost_gk_refit/default/ghost_gk_v1` (target ≤ ~9 MB; the `metrics.json` `acceptance.artifact_size_lt_15mb` gate must pass).

- [ ] **Step 6: Pull artifacts + the feature cache back to the dev machine**

```bash
# from the dev machine
scp -r karsten@192.168.68.73:~/Development/ghost_gk_refit/{full,default} ./_refit_artifacts/
```

The feature cache (`_feature_cache/{features,labels,groups,providers}.*`) is needed by the Part-3 gate script — pull it too.

---

# PART 3 — Validate, bundle, release

## Task 10: Run the gate + measure the serve delta

**Files:** none (operational, uses Task-7 scripts)

- [ ] **Step 1: Run the apples-to-apples gate for each variant**

```bash
python scripts/validate_ghost_gk_refit.py \
    --features _refit_artifacts/full/ghost_gk_v1/_feature_cache/features.parquet \
    --labels   _refit_artifacts/full/ghost_gk_v1/_feature_cache/labels.parquet \
    --groups   _refit_artifacts/full/ghost_gk_v1/_feature_cache/groups.npy \
    --providers _refit_artifacts/full/ghost_gk_v1/_feature_cache/providers.npy \
    --refit-dir _refit_artifacts/full/ghost_gk_v1 \
    --incumbent-variant full --epsilon 0.02 --cv-folds 5 2>&1 | tee gate_full.log
```

Repeat with `--incumbent-variant default` against the default artifacts. Record each `VERDICT`.

- [ ] **Step 2: Measure the serve delta on 1-2 real matches (P3)**

```bash
python scripts/measure_ghost_gk_serve_delta.py --frames <a_real_match>.parquet --home-team-id <id> --variant default 2>&1 | tee serve_delta.log
```

Record `max` + `median` euclidean delta (m) and `% changed`. This is the number the CHANGELOG + lakehouse heads-up carry.

- [ ] **Step 3: Decide ship-vs-keep per variant**

For each variant: if `VERDICT == SHIP REFIT` → bundle the re-fit (Task 11a). If `KEEP INCUMBENT` → backfill (Task 11b) + record why + file the N4 follow-up TODO row.

---

## Task 11a: Bundle the re-fit weights (per SHIP-REFIT variant)

**Files:** `silly_kicks/tracking/_ghost_gk_weights/{default,full}/`

- [ ] **Step 1: Replace the bundled `default` artifact (if shipping)**

```bash
rm silly_kicks/tracking/_ghost_gk_weights/default/{rfcde_weights.npz,metadata.json,SHA256SUMS}
cp _refit_artifacts/default/ghost_gk_v1/{rfcde_weights.npz,metadata.json,SHA256SUMS} \
   silly_kicks/tracking/_ghost_gk_weights/default/
```

Verify: `python -c "from silly_kicks.tracking._ghost_gk import GhostGkModel; m=GhostGkModel.from_variant('default'); print(m.carrier_params, m.training_commit)"` prints `{'tolerance_m': 3.0, 'beta': 0.0, 'gamma': 0.25}` + the Part-1 SHA. Confirm wheel-budget: `du -h silly_kicks/tracking/_ghost_gk_weights/default/rfcde_weights.npz` ≤ ~9 MB.

- [ ] **Step 2: Publish the `full` artifact to HuggingFace Hub (if shipping)**

```bash
python scripts/publish_ghost_gk.py --model-dir _refit_artifacts/full/ghost_gk_v1 --repo-id silly-kicks/ghost-gk-v1
```

(Use the existing publish script; it handles the `[ghost-gk]` Hub upload + model card. Update the model card MAE/sample-count + add the `carrier_params`/`training_commit` provenance line. See [[project_huggingface_hub_patterns]].)

- [ ] **Step 3: Verify the bundled-load integrity gate**

Run: `python -m pytest tests/tracking/ -k "ghost_gk and (variant or load or integrity)" -v`
Expected: PASS — the new SHA256SUMS match the new artifact; `from_variant` loads with `carrier_params`.

---

## Task 11b: Backfill incumbent metadata (per KEEP-INCUMBENT variant — N4 fallback)

**Files:** `silly_kicks/tracking/_ghost_gk_weights/<variant>/metadata.json` + `SHA256SUMS`

- [ ] **Step 1: Backfill the old carrier regime via `save()` (NP2 — reuse the canonical path)**

The incumbent was trained under the OLD defaults. `save()` round-trips a `load()`ed model from its
stored tree/leaf arrays AND writes the metadata + SHA256SUMS that `load`'s integrity gate verifies —
so reuse it; do NOT hand-roll the JSON/hash (that duplicates the contract and can silently drift the
bundled-load gate). One-shot:

```python
# (throwaway; do NOT commit the script)
from pathlib import Path
from silly_kicks.tracking._ghost_gk import GhostGkModel

variant = "default"  # or "full"
d = Path(f"silly_kicks/tracking/_ghost_gk_weights/{variant}")
m = GhostGkModel.from_variant(variant)
m.carrier_params = {"tolerance_m": 3.0, "beta": 0.5, "gamma": 1.0}  # incumbent training-time regime
m.training_platform = "incumbent-backfilled"
m.save(d)  # canonical: writes carrier_params + version 1.1.0 + SHA256SUMS exactly as load() expects
```

Then `GhostGkModel.from_variant(variant)` and assert `carrier_params == {tolerance_m:3.0, beta:0.5, gamma:1.0}` (passes the integrity check, since `save()` wrote the matching sums). For the Hub-hosted
`full`, run the same against the downloaded dir, then re-publish (Task 11a Step 2). **Rationale (N4):**
serve now computes the carrier with these recorded params, matching the incumbent's training —
R3-self-consistent. This is **availability-safe, NOT a staleness fix.**

- [ ] **Step 2: File the N4 follow-up**

Add a TODO.md row: `Ghost-GK <variant> re-fit deferred — 81-match pining corpus regressed vs incumbent (see gate_<variant>.log); refresh on bronze-scale corpus.`

---

## Task 12: Docs + version bump + release prep

**Files:** `CHANGELOG.md`, `NOTICE`, `CLAUDE.md`, `TODO.md`, `pyproject.toml`, `silly_kicks/__init__.py`

- [ ] **Step 1: Bump the version (hard gate — all four must match)**

Set the next minor (off 4.9.0; whoever merges second re-bumps) in `pyproject.toml`, `silly_kicks/__init__.py` `__version__`, and reference it in `CHANGELOG.md` + `TODO.md`. Per [[feedback_version_bump_hard_gate]].

- [ ] **Step 2: CHANGELOG entry (carry the MEASURED delta, P3/P4)**

```markdown
## [X.Y.0] - 2026-06-XX

### Fixed
- **Ghost-GK serve-carrier consistency (PR-S81).** `compute_ghost_gk` now computes the
  ball-carrier on full frames so `team_in_possession` matches training (was always 0.0 at
  serve). Changes served `ghost_gk_x/y` on possession frames for **all** variants — measured
  max <D_MAX> m / median <D_MED> m on real data (<PCT>% of samples). Driven by the bug fix,
  not only the re-fit (applies even where incumbent weights are kept).

### Changed
- **Ghost-GK R3 carrier-param record/consume.** `GhostGkModel` records `carrier_params` in
  metadata (version 1.1.0) + `training_commit`/`sklearn_version`/`training_platform`; serve
  consumes them. Re-fit bundled weights against the 4.7.0 carrier defaults
  (`beta=0.0, gamma=0.25`). `compute_ghost_gk`/`add_ghost_gk`/`ghost_gk_xfns` gain an optional
  `carrier=` passthrough (cache convention, mirrors `links`).
```

Fill `<D_MAX>`/`<D_MED>`/`<PCT>` from `serve_delta.log`.

- [ ] **Step 3: NOTICE — no change**

R3 is provenance plumbing, no new methodology. Verify no NOTICE edit needed (the ghost-GK / carrier citations already exist).

- [ ] **Step 4: CLAUDE.md — update the PR-S52/S55 Ghost-GK line**

Append to the Ghost-GK architecture note: R3 carrier-param record/consume + serve-carrier fix + 4.7.0 re-fit (PR-S81).

- [ ] **Step 5: TODO.md — delete the PR-S81 row**

Remove the "Ghost-GK carrier train/serve consistency + R3" row (now done). Keep the N4 follow-up row from Task 11b if any variant was kept. Per [[feedback_todo_grooming_delete_dont_annotate]].

- [ ] **Step 6: Commit (docs + bump + weights)**

```bash
git add CHANGELOG.md CLAUDE.md TODO.md pyproject.toml silly_kicks/__init__.py silly_kicks/tracking/_ghost_gk_weights/
git commit -m "docs(ghost-gk): CHANGELOG/version/weights for PR-S81 re-fit + serve fix + R3"
```

---

## Task 13: Final review, squash, merge ritual

- [ ] **Step 1: Run `/final-review`** (mandatory — [[feedback_final_review_gate]]). Address findings inline.

- [ ] **Step 2: Full local gate (Shift Left)**

```bash
python -m pytest tests/ -m "not e2e" -q
ruff check silly_kicks/ scripts/ && ruff format --check silly_kicks/ scripts/ && pyright silly_kicks/
```
Expected: all green. Regenerate C4 only if architecture changed (it did not — below container level).

- [ ] **Step 3: Squash to a single commit** (one-commit policy — [[feedback_commit_policy]]). The transient Part-1 + Part-3 commits collapse to one; `training_commit` in the bundled metadata remains the pre-squash SHA (documented, N3). Get explicit owner approval before the final commit.

- [ ] **Step 4: Push + open PR; merge `--admin --squash` after CI green** (never tag before main CI green — [[feedback_never_tag_before_ci_green]]). Then tag + push the new minor. Do NOT poll CI ([[feedback_dont_poll_ci_runs]]).

- [ ] **Step 5: Lakehouse heads-up** — note in the PR body that consumers pinning the new minor get shifted `ghost_gk_x/y` (the measured delta from §5.2a) on possession frames; recommend re-running ghost-GK enrichment.

---

## Self-Review

**Spec coverage:**
- §3.1 serve fix → Task 3. §3.2 R3 → Tasks 1+2 (+ trainer wiring Task 6). N5 carrier passthrough → Tasks 3+4.
- §4 re-fit (committed SHA, single cp, GS, variants) → Tasks 6 (cp/variant) + 8 (cache writer) + 9 (execution). P6 training_commit → Tasks 1+6+8.
- §5.1 apples-to-apples gate + N4 fallback → Task 7 (script) + 10 (run) + 11b (backfill). §5.2 tests #1-#7 → Tasks 1,3,5,7. §5.2a measured delta → Task 7 (script) + 10 (run) + 12 (CHANGELOG). §5.3 release/Hyrum → Task 12. §6 files → all Part-1 tasks. §7 provenance metadata → Tasks 1+6.
- N1 (no public break) → Task 2 (prepare stays 2-tuple) + Task 6 (trainer single cp). N2 → Task 5. N3 → Task 8 commit note + Task 13. N4 → Tasks 10-11b.
- **Round-3:** NP1 (cache writer as prereq) → Task 8. NP2 (backfill via `save()`) → Task 11b. NP3 (committed CLI test) → Task 6 Step 8. NP4 (unambiguous near-tie fixture) → Task 5 Step 3.

**Placeholder scan:** `<D_MAX>`/`<D_MED>`/`<PCT>` in Task 12 CHANGELOG are intentional fill-from-measurement slots (the measurement is Task 10); `<a_real_match>`/`<id>`/`<variant>` are operator-supplied at execution. No code-step placeholders.

**Type consistency:** `carrier_params: dict` and `carrier: pd.DataFrame | None` used identically across `__init__`/`fit`/`save`/`load`/`prepare`/`compute_ghost_gk`/`add_ghost_gk`/`ghost_gk_xfns`. `DEFAULT_CARRIER_PARAMS` keys `{tolerance_m, beta, gamma}` consistent with `infer_ball_carrier(**cp)` calls throughout. Version string `"1.1.0"` consistent (save + tests).
