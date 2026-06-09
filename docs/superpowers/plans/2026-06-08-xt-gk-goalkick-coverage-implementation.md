# xT-GK goal-kick coverage — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make xT-GK's RAV / composite well-defined for ~all real goal-kicks by (A) deriving the missing goal-kick origin (conditional, provenance-tagged, never mutating shared `actions`) and (B) replacing the open-play `accessible-space` xC with a fitted **logistic** GK-distribution completion model — folding both into the in-flight 4.21.0 release.

**Architecture:** Hexagonal. `_resolve_gk_geometry` is a pure helper that supplies derived coords + provenance to `compute_xt_gk` *before* the coord gate. `GkCompletionModel` is a logistic-regression port mirroring the xS/xCross trained-light lifecycle (pure-numpy `sigmoid(Xβ)` serve, tiny JSON coefficient artifact + SHA256, `from_variant("default")` + caller-overridable). RAV consumes the model; `[das]`/`get_xc` are removed from the RAV path. All design claims are owner-data-measured (spec §1/§9).

**Tech Stack:** Python, pandas, numpy, scikit-learn (fit only; pure-numpy serve). No xgboost, no new runtime dep.

**Spec:** `docs/superpowers/specs/2026-06-08-xt-gk-goalkick-coverage-design.md` (approved; all D-/R- decisions confirmed + measured).

---

## ⚠️ Repo-convention notes (read first)
- **NO per-task commits.** This folds into the **single** 4.21.0 commit (`feat/xt-gk-eyestone`) at the very end, sentinel-gated. Each task ends in a **test-green checkpoint**.
- Run with the uv `.venv` (CPython 3.10.19): `.venv\Scripts\python.exe -m pytest …`. Owner-data steps use `_loader_pining` **read-only** (never modify `scripts/_loader_*`).
- This work amends ADR-024 (RAV P(success): get_xc → fitted model; `[das]` optional).

## File structure

| File | C/M | Responsibility |
|---|---|---|
| `silly_kicks/tracking/_gk_geometry.py` | **Create** | `resolve_gk_geometry` (conditional-origin + dest derivation + provenance/confidence). Pure, promotable (the general-enrichment follow-up lifts this). |
| `silly_kicks/tracking/_gk_completion.py` | **Create** | `GkCompletionModel` (logistic), `extract_gk_completion_features`, `GK_COMPLETION_FEATURE_NAMES`, `prepare_gk_completion_training_data`, `compute_gk_completion`, `add_gk_completion` |
| `silly_kicks/tracking/_gk_completion_weights/default/` | **Create** | bundled `model.json` (coef+intercept+standardization) + `metadata.json` + `SHA256SUMS` (tiny) |
| `silly_kicks/tracking/_xt_gk.py` | Modify | wire `resolve_gk_geometry` + completion model into `compute_xt_gk`; remove `_require_das`/`get_xc`/`_xc_for_passes`; emit provenance cols; `completion=` kwarg |
| `silly_kicks/tracking/features.py` | Modify | export `add_gk_completion`/`gk_completion_xfns`? (No xfns needed — completion is internal to RAV; export `add_gk_completion` for standalone use) |
| `silly_kicks/tracking/__init__.py` | Modify | export `GkCompletionModel`, `compute_gk_completion`, `add_gk_completion`, `resolve_gk_geometry` |
| `scripts/train_gk_completion.py` | **Create** | train-only; CV + gates + writes the bundled artifact |
| `tests/tracking/test_gk_geometry.py` | **Create** | geometry helper tests |
| `tests/tracking/test_gk_completion.py` | **Create** | model + features + train==serve + degenerate-label + serialization |
| `tests/tracking/test_xt_gk.py` | Modify | RAV-via-model, [das]-absent self-sufficiency, provenance, split coverage |
| `docs/superpowers/adrs/ADR-024-xt-gk.md` | Modify | amendment paragraph |
| `NOTICE`, `CLAUDE.md`, `docs/c4/architecture.dsl`+regen, `CHANGELOG.md` | Modify | attribution + arch line + count + changelog |

---

# PHASE A — goal-kick coordinate derivation

## Task A1: `resolve_gk_geometry` (conditional origin + dest + provenance)

**Files:** Create `silly_kicks/tracking/_gk_geometry.py`; Test `tests/tracking/test_gk_geometry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/tracking/test_gk_geometry.py
import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._gk_geometry import resolve_gk_geometry, _GOAL_AREA_DEPTH

_GK = 22  # goalkick type_id


def _actions(**over):
    base = dict(
        game_id=[9, 9], action_id=[0, 1], team_id=[1, 1], player_id=[10, 10],
        period_id=[1, 1], time_seconds=[5.0, 50.0], type_id=[_GK, _GK],
        start_x=[5.0, np.nan], start_y=[34.0, np.nan],
        end_x=[55.0, 60.0], end_y=[34.0, 30.0],
    )
    base.update(over)
    return pd.DataFrame(base)


class TestResolveGkGeometry:
    def test_native_origin_kept(self):
        a = _actions()
        g = resolve_gk_geometry(a, frames=None)
        assert g.loc[0, "origin_x"] == 5.0
        assert g.loc[0, "origin_source"] == "native"
        assert g.loc[0, "origin_confidence"] == pytest.approx(1.0)

    def test_nan_origin_falls_to_rule_point_when_no_frames(self):
        a = _actions()  # row 1 has NaN start, no frames -> rule point
        g = resolve_gk_geometry(a, frames=None)
        assert g.loc[1, "origin_x"] == pytest.approx(5.5)  # 6-yard-box centre
        assert g.loc[1, "origin_y"] == pytest.approx(34.0)
        assert g.loc[1, "origin_source"] == "goalkick_prior"
        assert g.loc[1, "origin_confidence"] < 0.7

    def test_tracking_gk_used_only_when_in_goal_area(self):
        a = _actions()
        # frame for action 1 (time 50): GK in goal area (x=4) -> tier 2
        frames = pd.DataFrame({
            "game_id": [9], "period_id": [1], "frame_id": [1250], "time_seconds": [50.0],
            "team_id": [1], "player_id": [10], "is_goalkeeper": [True], "is_ball": [False],
            "x": [4.0], "y": [33.0],
        })
        g = resolve_gk_geometry(a, frames=frames)
        assert g.loc[1, "origin_source"] == "tracking_gk"
        assert g.loc[1, "origin_x"] == pytest.approx(4.0)
        assert 0.6 <= g.loc[1, "origin_confidence"] < 1.0

    def test_tracking_gk_offposition_clamped_to_prior(self):
        a = _actions()
        frames = pd.DataFrame({  # GK at x=40 (off position) -> NOT used, fall to prior
            "game_id": [9], "period_id": [1], "frame_id": [1250], "time_seconds": [50.0],
            "team_id": [1], "player_id": [10], "is_goalkeeper": [True], "is_ball": [False],
            "x": [40.0], "y": [33.0],
        })
        g = resolve_gk_geometry(a, frames=frames)
        assert g.loc[1, "origin_source"] == "goalkick_prior"
        assert g.loc[1, "origin_x"] == pytest.approx(5.5)

    def test_dest_native_kept_and_nan_dest_unresolved(self):
        a = _actions(end_x=[55.0, np.nan])
        g = resolve_gk_geometry(a, frames=None)
        assert g.loc[0, "dest_source"] == "native"
        assert np.isnan(g.loc[1, "dest_x"])
        assert g.loc[1, "dest_source"] == "unresolved"

    def test_does_not_mutate_input(self):
        a = _actions()
        before = a["start_x"].copy()
        resolve_gk_geometry(a, frames=None)
        pd.testing.assert_series_equal(a["start_x"], before)
```

- [ ] **Step 2: Run → fail** (`ModuleNotFoundError`):
`.venv\Scripts\python.exe -m pytest tests/tracking/test_gk_geometry.py -q`

- [ ] **Step 3: Implement** `silly_kicks/tracking/_gk_geometry.py`

```python
"""Goal-kick geometry resolution for xT-GK (scoped; promotable — see TODO general-enrichment).

Derives a goal-kick's origin/destination when the SPADL event omits them (real GS data:
~67% NaN origin), WITHOUT mutating the shared `actions` frame. Conditional origin in
confidence order (native -> in-area tracking-GK -> empirical median -> rule point);
destination native -> next-event -> tracking-ball -> unresolved. Emits per-row source +
continuous confidence (review R3/R7; all tiers measured on owner data, spec §2/§9).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.spadl import config as spadlconfig

from ._id_compat import canonical_id_series, ids_match

_GOALKICK = spadlconfig.actiontype_id["goalkick"]
_GOAL_AREA_DEPTH = 16.5  # m from own goal line; tracking-GK beyond this is "off position" (R3: 48%)
_RULE_POINT = (5.5, 34.0)  # 6-yard-box centre, LTR (D-A2)
# NOTE (review m2): the spec listed an "empirical native-start median" tier, but median (8.8,32)
# and rule-point (5.5,34) are NOT distinguishable by available data (both fire when there is no
# native start and no in-area tracking-GK), so there is ONE positional-prior fallback = the
# rule point (D-A2 confirmed it). Effective tiers: native / in-area-tracking-GK / rule-point.
_CONF = {"native": 1.0, "tracking_gk": 0.7, "goalkick_prior": 0.2, "unresolved": 0.0}


def resolve_gk_geometry(actions: pd.DataFrame, *, frames: pd.DataFrame | None, links: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a frame indexed like ``actions`` with columns origin_x/origin_y/origin_source/
    origin_confidence + dest_x/dest_y/dest_source. Only goal-kicks (type 22) get origin
    imputation; other rows pass native coords through. ``actions`` is never mutated."""
    n = len(actions)
    out = pd.DataFrame(index=actions.index)
    sx = actions["start_x"].to_numpy(float)
    sy = actions["start_y"].to_numpy(float)
    ex = actions["end_x"].to_numpy(float)
    ey = actions["end_y"].to_numpy(float)
    type_id = actions["type_id"].to_numpy()
    is_goalkick = type_id == _GOALKICK

    origin_x = sx.copy()
    origin_y = sy.copy()
    source = np.where(np.isfinite(sx) & np.isfinite(sy), "native", "unresolved").astype(object)

    # tier 2: in-area tracking-GK (goal-kicks with NaN native origin only)
    need = is_goalkick & (source == "unresolved")
    if need.any() and frames is not None:
        gk_xy = _tracking_gk_xy(actions, frames, links)  # (n,2) float, NaN where unavailable/off-area
        use = need & np.isfinite(gk_xy[:, 0])
        origin_x[use] = gk_xy[use, 0]
        origin_y[use] = gk_xy[use, 1]
        source[use] = "tracking_gk"

    # tier 3: empirical median; tier 4: rule point (goal-kicks still unresolved)
    still = is_goalkick & (source == "unresolved")
    origin_x[still], origin_y[still] = _RULE_POINT  # terminal fallback
    source[still] = "goalkick_prior"

    out["origin_x"] = origin_x
    out["origin_y"] = origin_y
    out["origin_source"] = source
    out["origin_confidence"] = np.array([_CONF[s] for s in source], dtype=float)

    # destination: native -> next-event -> unresolved
    dest_x = ex.copy()
    dest_y = ey.copy()
    dsource = np.where(np.isfinite(ex) & np.isfinite(ey), "native", "unresolved").astype(object)
    nan_dest = is_goalkick & (dsource == "unresolved")
    if nan_dest.any():
        nx, ny = _next_event_start(actions)
        use = nan_dest & np.isfinite(nx)
        dest_x[use], dest_y[use], dsource[use] = nx[use], ny[use], "next_event"
    out["dest_x"] = dest_x
    out["dest_y"] = dest_y
    out["dest_source"] = dsource
    return out


def _next_event_start(actions: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Next row's start coords (the receiver location), positionally. NaN'd across a
    (game_id, period_id) boundary (review C-b: a period-/match-final goalkick must NOT take the
    next period's first action as its destination -> falls to dest_source == 'unresolved')."""
    nx = actions["start_x"].shift(-1).to_numpy(float)
    ny = actions["start_y"].shift(-1).to_numpy(float)
    same = np.ones(len(actions), dtype=bool)
    for col in ("game_id", "period_id"):
        if col in actions.columns:
            same &= actions[col].to_numpy() == actions[col].shift(-1).to_numpy()
    nx = np.where(same, nx, np.nan)
    ny = np.where(same, ny, np.nan)
    return nx, ny


def _tracking_gk_xy(actions: pd.DataFrame, frames: pd.DataFrame, links: pd.DataFrame | None) -> np.ndarray:
    """Acting-team GK position at each goal-kick's linked frame, CLAMPED to the goal area
    (x <= _GOAL_AREA_DEPTH in LTR own-half coords); NaN where unavailable or off-position."""
    from ._kernels import resolve_frame_ids_by_position

    n = len(actions)
    res = np.full((n, 2), np.nan, dtype=float)
    fid = resolve_frame_ids_by_position(actions, frames, links=links)
    fg = frames.groupby("frame_id")
    for i in range(n):
        if not np.isfinite(fid[i]):
            continue
        try:
            fr = fg.get_group(int(fid[i]))
        except KeyError:
            continue
        gk = fr[fr["is_goalkeeper"].astype(bool) & (~fr["is_ball"].astype(bool))
                & ids_match(fr["team_id"], actions["team_id"].iloc[i])]
        if gk.empty:
            continue
        gx, gy = float(gk.iloc[0]["x"]), float(gk.iloc[0]["y"])
        if gx <= _GOAL_AREA_DEPTH:  # R3 clamp
            res[i] = (gx, gy)
    return res
```

> Effective tiers are **3** (native / in-area-tracking-GK / rule-point): the spec's "empirical median" and "rule point" are not data-distinguishable (both fire on no-native-no-tracking), so they collapse to one positional prior = the rule point (D-A2). The dead `_EMPIRICAL_MEDIAN` constant + `"empirical_median"` confidence entry were removed (review m2).

- [ ] **Step 4: Run → pass.** **Step 5: Checkpoint** (geometry helper green).

## Task A2: wire `resolve_gk_geometry` into `compute_xt_gk`

*(Deferred to Phase C — the wiring lands together with the RAV/model rewrite to avoid a half-rewired intermediate. A2's behavior is covered by Phase C tests.)*

---

# PHASE B — GK-distribution completion model

## Task B1: `extract_gk_completion_features` + feature names

**Files:** Create `silly_kicks/tracking/_gk_completion.py`; Test `tests/tracking/test_gk_completion.py`

- [ ] **Step 1: failing test**

```python
# tests/tracking/test_gk_completion.py
import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._gk_completion import (
    GK_COMPLETION_FEATURE_NAMES,
    extract_gk_completion_features,
)


def _geom():
    # output of resolve_gk_geometry for 2 goal-kicks (origin imputed for row 1)
    return pd.DataFrame({
        "origin_x": [5.0, 5.5], "origin_y": [34.0, 34.0],
        "dest_x": [55.0, 60.0], "dest_y": [34.0, 30.0],
        "type_id": [22, 22],
    })


class TestFeatures:
    def test_feature_names_and_shape(self):
        X = extract_gk_completion_features(_geom(), defender_density=pd.Series([3.0, 5.0]))
        assert list(X.columns) == GK_COMPLETION_FEATURE_NAMES
        assert len(X) == 2

    def test_length_is_origin_to_dest(self):
        X = extract_gk_completion_features(_geom(), defender_density=pd.Series([np.nan, np.nan]))
        assert X.loc[0, "length"] == pytest.approx(50.0)  # |55-5|, dy 0

    def test_missing_density_left_nan_for_model_to_impute(self):
        # P3: extract does NOT sentinel-fill; the MODEL mean-imputes density NaN (neutral).
        X = extract_gk_completion_features(_geom(), defender_density=pd.Series([np.nan, 2.0]))
        assert np.isnan(X.loc[0, "dest_defender_density"])
        assert X.loc[1, "dest_defender_density"] == 2.0
```

- [ ] **Step 2: run → fail.**
- [ ] **Step 3: implement** (top of `_gk_completion.py`):

```python
"""GK-distribution pass-completion model for xT-GK RAV (Eyestone). Logistic regression,
pure-numpy serve, tiny JSON coefficient artifact. Replaces the open-play accessible-space
xC (OOD on goal-kicks: ~31% coverage, spec §1). See NOTICE / ADR-024."""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from silly_kicks.spadl import config as spadlconfig

_GOALKICK = spadlconfig.actiontype_id["goalkick"]
_THROW_IN = spadlconfig.actiontype_id["throw_in"]
_WEIGHTS_ROOT = Path(__file__).parent / "_gk_completion_weights"
_VARIANT_CACHE: dict = {}
_GEOM_FEATURES = ("length", "forwardness", "dest_x", "dest_y_off")  # geometry-unscoreable iff any NaN

GK_COMPLETION_FEATURE_NAMES = [
    "length", "forwardness", "dy_abs", "dest_x", "dest_y_off",
    "dest_defender_density", "is_goalkick", "is_throw_in",
]


def extract_gk_completion_features(
    geom: pd.DataFrame, *, defender_density: pd.Series | None = None
) -> pd.DataFrame:
    """Feature rows from resolved geometry (origin_x/y, dest_x/y, type_id). The SINGLE code
    path used at BOTH train and serve (review M2/R6 parity)."""
    ox = geom["origin_x"].to_numpy(float)
    oy = geom["origin_y"].to_numpy(float)
    dx = geom["dest_x"].to_numpy(float) - ox
    dy = geom["dest_y"].to_numpy(float) - oy
    length = np.hypot(dx, dy)
    dens = (
        defender_density.to_numpy(float)
        if defender_density is not None
        else np.full(len(geom), np.nan)
    )
    # P3: density NaN is LEFT NaN for the model to MEAN-impute (neutral after standardization).
    # A finite sentinel (e.g. -1) would be an OOD extrapolation in a linear model AND would
    # bypass the per-type base-rate fallback (which only triggers on geometry-unscoreable rows).
    tid = geom["type_id"].to_numpy()
    return pd.DataFrame({
        "length": length,
        "forwardness": np.divide(dx, length, out=np.zeros_like(dx), where=length > 0),
        "dy_abs": np.abs(dy),
        "dest_x": geom["dest_x"].to_numpy(float),
        "dest_y_off": np.abs(geom["dest_y"].to_numpy(float) - spadlconfig.field_width / 2),
        "dest_defender_density": dens,
        "is_goalkick": (tid == _GOALKICK).astype(float),
        "is_throw_in": (tid == _THROW_IN).astype(float),
    }, index=geom.index)
```

- [ ] **Step 4: run → pass. Step 5: checkpoint.**

## Task B2: `GkCompletionModel` (logistic, pure-numpy serve, per-type fallback, JSON envelope)

**Files:** Modify `_gk_completion.py`; Test `tests/tracking/test_gk_completion.py`

- [ ] **Step 1: failing test**

```python
# tests/tracking/test_gk_completion.py  (append)
from silly_kicks.tracking._gk_completion import GkCompletionModel


class TestModel:
    def _Xy(self, n=300):
        rng = np.random.default_rng(0)
        length = rng.uniform(5, 70, n)
        X = pd.DataFrame({
            "length": length, "forwardness": rng.uniform(-1, 1, n), "dy_abs": rng.uniform(0, 30, n),
            "dest_x": rng.uniform(20, 100, n), "dest_y_off": rng.uniform(0, 34, n),
            "dest_defender_density": rng.uniform(0, 6, n),
            "is_goalkick": (rng.random(n) > 0.5).astype(float), "is_throw_in": np.zeros(n),
        })
        y = (rng.random(n) < 1 / (1 + np.exp((length - 35) / 12))).astype(int)  # longer -> lower
        return X, pd.Series(y)

    def test_fit_predict_in_unit_interval(self):
        X, y = self._Xy()
        m = GkCompletionModel().fit(X, y)
        p = m.predict_proba(X)
        assert p.shape == (len(X),)
        assert (p >= 0).all() and (p <= 1).all()

    def test_pure_numpy_serve_matches_sklearn(self):
        X, y = self._Xy()
        m = GkCompletionModel().fit(X, y)
        # round-trip via JSON; reloaded predicts identically (pure-numpy, no sklearn)
        d = m.to_dict()
        m2 = GkCompletionModel.from_dict(d)
        np.testing.assert_allclose(m.predict_proba(X), m2.predict_proba(X), atol=1e-9)

    def test_per_type_base_rate_fallback(self):
        # C2: a GEOMETRY-unscoreable row (NaN geometry) -> the recorded per-type base rate.
        X, y = self._Xy()
        m = GkCompletionModel().fit(X, y)
        m._base_rates["goalkick"] = 0.55
        m._base_rates["throw_in"] = 0.95
        gk = pd.DataFrame([{c: np.nan for c in X.columns}]); gk["is_goalkick"] = 1.0; gk["is_throw_in"] = 0.0
        ti = pd.DataFrame([{c: np.nan for c in X.columns}]); ti["is_goalkick"] = 0.0; ti["is_throw_in"] = 1.0
        assert m.predict_proba(gk)[0] == pytest.approx(0.55)
        assert m.predict_proba(ti)[0] == pytest.approx(0.95)
        assert m._base_rate_for_type(1.0, 0.0) != m._base_rate_for_type(0.0, 1.0)

    def test_density_nan_is_mean_imputed_not_base_rate(self):
        # P3: only density missing (geometry fine) -> scored on geometry with density mean-imputed,
        # NOT routed to the base rate. NaN density == providing the training mean.
        X, y = self._Xy()
        m = GkCompletionModel().fit(X, y)
        di = m.feature_names.index("dest_defender_density")
        r_nan = X.iloc[[0]].copy(); r_nan["dest_defender_density"] = np.nan
        r_mean = X.iloc[[0]].copy(); r_mean["dest_defender_density"] = m._mean[di]
        np.testing.assert_allclose(m.predict_proba(r_nan), m.predict_proba(r_mean), atol=1e-12)

    def test_save_load_roundtrip_sha(self, tmp_path):
        X, y = self._Xy()
        m = GkCompletionModel().fit(X, y)
        m.save(tmp_path)
        assert (tmp_path / "model.json").exists() and (tmp_path / "SHA256SUMS").exists()
        r = GkCompletionModel.load(tmp_path)
        np.testing.assert_allclose(m.predict_proba(X), r.predict_proba(X), atol=1e-9)
```

- [ ] **Step 2: run → fail. Step 3: implement** (append to `_gk_completion.py`):

```python
class GkCompletionModel:
    """Logistic P(success) for GK distributions. sklearn at fit; pure-numpy at serve."""

    VERSION = "1.0.0"

    def __init__(self) -> None:
        self._coef: np.ndarray | None = None        # (n_features,)
        self._intercept: float = 0.0
        self._mean: np.ndarray | None = None         # standardization
        self._std: np.ndarray | None = None
        self.feature_names: list[str] = list(GK_COMPLETION_FEATURE_NAMES)
        self._base_rates: dict[str, float] = {}      # per-type fallback (R5)
        self.shipped_variant: str | None = None
        self.provider_list: list | None = None

    # ---- fit ----
    def fit(self, features: pd.DataFrame, labels: pd.Series) -> "GkCompletionModel":
        from sklearn.linear_model import LogisticRegression

        X_raw = features[self.feature_names].to_numpy(float)  # geometry finite (P2); density may be NaN
        y = np.asarray(labels, dtype=int)
        # m-a: standardization stats over PRESENT values only (nanmean/nanstd) so mean-imputed
        # density rows do NOT compress _std as the NaN fraction grows.
        self._mean = np.nanmean(X_raw, axis=0)
        std = np.nanstd(X_raw, axis=0)
        self._std = np.where(std > 1e-9, std, 1.0)
        # P3: density NaN -> training mean (neutral after standardization); stored _mean makes serve
        # impute identically. (Geometry is finite -- prepare dropped geometry-unscoreable rows, P2.)
        X = np.where(np.isfinite(X_raw), X_raw, self._mean[None, :])
        if not np.isfinite(X).all():
            raise ValueError("fit received non-finite GEOMETRY features; prepare must drop "
                             "geometry-unscoreable rows (review P2).")
        Xs = (X - self._mean) / self._std
        clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs").fit(Xs, y)
        self._coef = clf.coef_[0].astype(float)
        self._intercept = float(clf.intercept_[0])
        # per-type base rates for the missing-feature fallback (R5)
        gk = features["is_goalkick"].to_numpy() == 1.0
        ti = features["is_throw_in"].to_numpy() == 1.0
        self._base_rates = {
            "goalkick": float(y[gk].mean()) if gk.any() else float(y.mean()),
            "throw_in": float(y[ti].mean()) if ti.any() else float(y.mean()),
            "other": float(y[~(gk | ti)].mean()) if (~(gk | ti)).any() else float(y.mean()),
            "global": float(y.mean()),
        }
        return self

    # ---- serve (pure numpy) ----
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if self._coef is None:
            raise RuntimeError("GkCompletionModel not fitted/loaded.")
        X = features[self.feature_names].to_numpy(float)
        geom_idx = [self.feature_names.index(c) for c in _GEOM_FEATURES]
        geom_bad = ~np.isfinite(X[:, geom_idx]).all(axis=1)  # no resolved origin+dest
        # per-FEATURE missing (e.g. density NaN, ~8% of rows) -> training mean (neutral); the row is
        # still scored on its geometry (P3 -- NOT a sentinel, NOT whole-row base-rate).
        Xf = np.where(np.isfinite(X), X, self._mean[None, :])
        Xs = (Xf - self._mean) / self._std
        p = 1.0 / (1.0 + np.exp(-(Xs @ self._coef + self._intercept)))
        # whole-row geometry-unscoreable -> per-type base rate (R5/M4).
        if geom_bad.any():
            gk = features["is_goalkick"].to_numpy()
            ti = features["is_throw_in"].to_numpy()
            for i in np.flatnonzero(geom_bad):
                p[i] = self._base_rate_for_type(gk[i], ti[i])
        return p

    def _base_rate_for_type(self, is_gk: float, is_ti: float) -> float:
        if is_gk == 1.0:
            return self._base_rates.get("goalkick", self._base_rates.get("global", 0.5))
        if is_ti == 1.0:
            return self._base_rates.get("throw_in", self._base_rates.get("global", 0.5))
        return self._base_rates.get("other", self._base_rates.get("global", 0.5))

    # ---- serialization (pickle-free JSON envelope; mirror xS save/load) ----
    def to_dict(self) -> dict:
        import sklearn
        return {
            "version": self.VERSION,
            "feature_names": self.feature_names,  # explicit order (R6/m9)
            "coef": self._coef.tolist(),
            "intercept": self._intercept,
            "mean": self._mean.tolist(),
            "std": self._std.tolist(),
            "base_rates": self._base_rates,
            "sklearn_version": sklearn.__version__,
            "shipped_variant": self.shipped_variant,
            "provider_list": self.provider_list,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GkCompletionModel":
        m = cls()
        m.feature_names = list(d["feature_names"])
        m._coef = np.asarray(d["coef"], dtype=float)
        m._intercept = float(d["intercept"])
        m._mean = np.asarray(d["mean"], dtype=float)
        m._std = np.asarray(d["std"], dtype=float)
        m._base_rates = dict(d["base_rates"])
        m.shipped_variant = d.get("shipped_variant")
        m.provider_list = d.get("provider_list")
        return m

    def save(self, path: Path | str) -> None:
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        (path / "model.json").write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        sha = hashlib.sha256((path / "model.json").read_text(encoding="utf-8").replace("\r\n", "\n").encode()).hexdigest()
        (path / "SHA256SUMS").write_text(f"{sha}  model.json\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "GkCompletionModel":
        path = Path(path)
        want = (path / "SHA256SUMS").read_text(encoding="utf-8").split()[0]
        got = hashlib.sha256((path / "model.json").read_text(encoding="utf-8").replace("\r\n", "\n").encode()).hexdigest()
        if want != got:
            raise ValueError(f"GkCompletionModel integrity check failed at {path}")
        d = json.loads((path / "model.json").read_text(encoding="utf-8"))
        if d.get("sklearn_version") and d["sklearn_version"].split(".")[0] != __import__("sklearn").__version__.split(".")[0]:
            warnings.warn("GkCompletionModel: sklearn major version differs from training; serve is numpy-only so OK.", stacklevel=2)
        return cls.from_dict(d)

    @classmethod
    def from_variant(cls, variant: str = "default") -> "GkCompletionModel":
        # NOTE (review m3): _VARIANT_CACHE returns a SHARED instance — loaded models are treated
        # IMMUTABLE post-load (predict-only). A future mutator must clone, not mutate in place.
        if variant in _VARIANT_CACHE:
            return _VARIANT_CACHE[variant]
        wdir = _WEIGHTS_ROOT / variant
        if not (wdir / "SHA256SUMS").exists():
            raise FileNotFoundError(
                f"No bundled Gk-completion weights for {variant!r} at {wdir}. "
                "Train via scripts/train_gk_completion.py."
            )
        m = cls.load(wdir)
        _VARIANT_CACHE[variant] = m
        return m
```

- [ ] **Step 4: run → pass. Step 5: checkpoint.**

## Task B3: `prepare_gk_completion_training_data` + degenerate-label guard

**Files:** Modify `_gk_completion.py`; Test `tests/tracking/test_gk_completion.py`

- [ ] **Step 1: failing test**

```python
# tests/tracking/test_gk_completion.py (append)
from silly_kicks.tracking._gk_completion import prepare_gk_completion_training_data


class TestPrepare:
    def _actions(self, results):
        n = len(results)
        return pd.DataFrame({
            "game_id": [9] * n, "action_id": list(range(n)), "team_id": [1] * n,
            "player_id": [10] * n, "period_id": [1] * n, "time_seconds": np.arange(n) * 10.0,
            "type_id": [22] * n, "result_id": results,
            "start_x": [5.0] * n, "start_y": [34.0] * n,
            "end_x": np.linspace(40, 90, n), "end_y": [34.0] * n,
        })

    def test_returns_X_y_groups(self):
        a = self._actions([1, 0, 1, 1, 0, 1])
        X, y, groups = prepare_gk_completion_training_data(a, frames=None)
        assert len(X) == len(y) == len(groups)
        assert set(np.unique(y)) <= {0, 1}

    def test_degenerate_label_raises(self):
        a = self._actions([1, 1, 1, 1, 1, 1])  # all success -> degenerate
        with pytest.raises(ValueError, match="degenerate"):
            prepare_gk_completion_training_data(a, frames=None)
```

- [ ] **Step 2: run → fail. Step 3: implement** (append):

```python
def prepare_gk_completion_training_data(
    actions: pd.DataFrame, *, frames: pd.DataFrame | None, links: pd.DataFrame | None = None,
    min_class_fraction: float = 0.02,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build (features, labels, groups) for the completion model. Train==serve by construction:
    the SAME serve-domain predicate (review P1), the SAME resolve_gk_geometry, the SAME shared
    density helper (review C1), and the SAME extract_gk_completion_features (M2). Label =
    result_id==success. Drops geometry-unscoreable rows (review P2). Fails loud on a degenerate
    label distribution (m6)."""
    from ._gk_geometry import resolve_gk_geometry
    from ._xt_gk import _gk_distribution_mask

    SUCCESS = spadlconfig.result_id["success"]
    # P1: the EXACT serve domain -- goalkicks + GK-actor open-play passes/throw-ins. Frames are
    # needed for GK identity; without frames (unit tests) fall back to goalkicks only.
    if frames is not None:
        mask = _gk_distribution_mask(actions, frames)
    else:
        mask = actions["type_id"].to_numpy() == _GOALKICK
    # P1-residual (review rev2): resolve geometry on the FULL action list, THEN mask -- mirroring
    # serve (compute_xt_gk does resolve_gk_geometry(actions).loc[mask]). _next_event_start's
    # positional shift(-1) is frame-SIZE-dependent: resolving on the pre-masked domain would make
    # a NaN-dest goalkick's "next event" the next GK-DISTRIBUTION row (wrong receiver) instead of
    # the next actual action -> train/serve skew on dest_source=="next_event" rows.
    geom_full = resolve_gk_geometry(actions, frames=frames, links=links)
    domain = actions.loc[mask].copy()
    geom = geom_full.loc[mask]
    dens = _gk_completion_density(domain, frames, geom, links)  # shared with serve (C1)
    X = extract_gk_completion_features(geom.assign(type_id=domain["type_id"].to_numpy()), defender_density=dens)
    X["origin_source"] = geom["origin_source"].to_numpy()  # metadata (NOT a feature) for the native gate (#1)
    y = (domain["result_id"].to_numpy() == SUCCESS).astype(int)
    groups = domain["game_id"].to_numpy() if "game_id" in domain.columns else np.zeros(len(domain))
    # P2: drop GEOMETRY-unscoreable rows (no resolved origin+dest -> NaN length); serve base-rates
    # exactly those (serve tolerates, fit crashes on NaN). #2: also drop NaN-id rows -- serve's
    # id_ok gate would route them to default, so for STRICT domain parity train drops them too.
    geom_ok = np.isfinite(X["length"].to_numpy()) & np.isfinite(X["dest_x"].to_numpy())
    id_ok = domain["player_id"].notna().to_numpy() & domain["team_id"].notna().to_numpy()
    keep = geom_ok & id_ok
    X = X.loc[keep].reset_index(drop=True)
    y, groups = y[keep], groups[keep]
    frac = float(y.mean())
    if min(frac, 1 - frac) < min_class_fraction or len(np.unique(y)) < 2:
        raise ValueError(f"degenerate label distribution (success rate={frac:.3f}); "
                         "check provider result_id semantics (m6/D5) before training.")
    return X, y, groups


def _gk_completion_density(actions, frames, geom, links=None):
    """receiver_zone_density at the RESOLVED destination -- the SINGLE shared producer used by
    BOTH prepare and the serve _completion path (review C1; the divergence-prone step). NaN where
    unlinked -> the model mean-imputes (review P3)."""
    if frames is None:
        return pd.Series(np.nan, index=actions.index)
    from .features import receiver_zone_density

    a = actions.copy()
    a["end_x"] = geom["dest_x"].to_numpy()
    a["end_y"] = geom["dest_y"].to_numpy()
    return receiver_zone_density(a, frames)
```

- [ ] **Step 4: run → pass. Step 5: checkpoint.**

## Task B4: `compute_gk_completion` / `add_gk_completion` (ADR-005 surfaces)

**Files:** Modify `_gk_completion.py`, `features.py`, `__init__.py`; Test `tests/tracking/test_gk_completion.py`

- [ ] Mirror `compute_xshot_occurrence` / `add_xshot_occurrence` (xS lines 729/821): `compute_gk_completion(actions, frames, *, model=None, links=None) -> Series` (resolve geometry → density → features → `model.predict_proba`; `model=None` → `GkCompletionModel.from_variant("default")`); `@nan_safe_enrichment add_gk_completion(...) -> DataFrame` adding column `gk_completion`. Export both + `GkCompletionModel`, `resolve_gk_geometry` from `__init__.py` (+ `__all__`). Tests: columns present, in [0,1], NaN-id safe; **and (review #3) a geometry-unscoreable row** (NaN end, no next-event → unresolvable destination) **returns the per-type base rate** through `compute_gk_completion` — this standalone surface is the **only live path** for the R5 fallback (RAV NaNs such rows, C-a), so exercise it end-to-end here so a future refactor can't silently drop it. **No `gk_completion_xfns`** (completion is internal to RAV, not a VAEP feature). Checkpoint.

> **(review #4, cosmetic):** `is_throw_in` will be **near-inert on the GS default** (goalkeepers essentially never take throw-ins → the column ≈ 0 throughout GS training → ~0 coefficient). Kept for forward-compatibility with providers/situations where GK throw-ins occur. Note this in the ADR/metrics so the ~0 coefficient isn't mistaken for a bug.

## Task B5: train script + bundled `default` artifact (GS-trained, R1)

**Files:** Create `scripts/train_gk_completion.py`; Create `_gk_completion_weights/default/`

- [ ] **Step 1:** `scripts/train_gk_completion.py` — skeleton mirrors `scripts/train_xshot_occurrence.py`: `--providers gradientsports` (R1 GS-only default) / `--data-dir`; stream matches via `_loader_pining`; `prepare_gk_completion_training_data` per match → concat (X carries the `origin_source` metadata column). **GroupKFold(5)** producing **out-of-fold predictions** (each row scored by the fold where it was held out — features built through the model's own fit/serve path).
  - **Fail-closed GREEN GATE (review #1, M5) = native-origin calibration, POOLED out-of-fold:** compute AUC/Brier on the pooled `X["origin_source"] == "native"` rows (one estimate over ALL native rows ≈ hundreds, NOT a per-fold mean over tens — the underpowered version). **Sample-size-aware:** require the **lower bound of a bootstrap AUC CI > base-rate AUC (0.5)** (so a few-dozen-sample fluke can't pass), and report `n_native`, AUC, the CI, and Brier-vs-base-rate. Abort (`sys.exit(1)`) if the gate fails or `n_native` is below a floor (e.g. < 100 — surfaced, not silently passed).
  - Final `GkCompletionModel().fit(X[feature_names], y)` on ALL kept rows (native + imputed — train==serve), `shipped_variant="default"`, `provider_list=["gradientsports"]`; `.save(_gk_completion_weights/default)`; round-trip assert `allclose(atol=1e-9)`; write `metrics.json` (n, n_native, base_rate, native_auc, native_auc_ci, brier, density_finite_rate, label_split, providers).
- [ ] **Step 2: run it** (owner data): `.venv\Scripts\python.exe scripts/train_gk_completion.py --providers gradientsports --max-per-provider 60`. Confirm gate passes + artifact written (tiny JSON). **Record in the ADR (review m-b):** held-out AUC/Brier, the **GS goal-kick label split** (measured: ~31 fail / 62 success per 6 matches = success rate **0.667**, non-degenerate — eyeball the trainer's `value_counts` to confirm it isn't degenerate-by-construction), and the **serve-time `receiver_zone_density` finite-rate** (measured **~94%** — so density carries real signal, not mostly-inert mean-imputation). This is the bundled `default`.
- [ ] **Step 3: checkpoint** — `GkCompletionModel.from_variant("default")` loads + predicts.

---

# PHASE C — RAV integration, [das] removal, provenance

## Task C1: rewire `compute_xt_gk` (geometry + completion model; remove get_xc)

**Files:** Modify `silly_kicks/tracking/_xt_gk.py`; Test `tests/tracking/test_xt_gk.py`

- [ ] **Step 1: failing tests** (RAV via model; [das]-absent self-sufficiency; provenance cols; derived-coords-feed-compute)

```python
# tests/tracking/test_xt_gk.py (append to TestComputeXtGk)
    def test_imputed_origin_goalkick_is_scored_and_tagged(self):
        # NaN-origin goalkick -> derived origin FEEDS compute -> non-NaN composite + tag (m7/m8)
        actions = _gk_actions()
        actions.loc[0, "start_x"] = np.nan  # row 0 goalkick: origin NaN
        frames = _frames_for(actions)
        out = compute_xt_gk(actions, frames, xt=_fitted_xt())
        assert not np.isnan(out.loc[0, "xt_gk"])            # scored (was NaN before)
        assert out.loc[0, "xt_gk_origin_source"] == "goalkick_prior"
        assert out.loc[0, "xt_gk_origin_confidence"] < 0.7

    def test_rav_uses_completion_model_not_das(self):
        # with accessible-space monkeypatched ABSENT, RAV/composite still computed (M4/R8)
        import builtins, importlib
        real_import = builtins.__import__
        def no_as(name, *a, **k):
            if name == "accessible_space" or name.startswith("accessible_space."):
                raise ImportError("accessible_space disabled for test")
            return real_import(name, *a, **k)
        actions = _gk_actions(); frames = _frames_for(actions)
        import silly_kicks.tracking._xt_gk as mod
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(builtins, "__import__", no_as)
            out = compute_xt_gk(actions, frames, xt=_fitted_xt())
        assert not out["xt_gk_rav"].isna().all()  # RAV computed without [das]
```

- [ ] **Step 2: run → fail.**
- [ ] **Step 3: implement** — in `compute_xt_gk`:
  1. **Remove** `_require_das()`, `from ._das import get_xc`, the `_xc_for_passes` helper, and the `frame_id`/`get_xc` block.
  2. After the early-return guards, call `geom = resolve_gk_geometry(actions, frames=frames, links=links)` and use `geom["origin_x"/"origin_y"]` as `sx/sy` and `geom["dest_x"/"dest_y"]` as `ex/ey` for the in-scope rows (derived coords FEED compute — m7/m8). Re-derive the coords gate from the RESOLVED coords (`coords_ok` = resolved origin AND dest finite).
  3. `p = _completion_p(actions, frames, geom, mask, links, completion)` then `rav = _rav(p, dest_star, _counter_value(xt_star, ex, ey), params.delta)`. **`_completion_p` is spelled out (review C3) and uses the SHARED density helper + extract path (train==serve parity, C1):**

```python
def _completion_p(actions, frames, geom, mask, links, completion):
    """RAV P(success) for in-scope rows via the GK-completion model. Default = bundled GS model;
    caller may inject a fitted GkCompletionModel. Builds features through the SAME shared density
    helper + extract used at train (review C1/C3)."""
    from ._gk_completion import (
        GkCompletionModel,
        _gk_completion_density,
        extract_gk_completion_features,
    )

    model = completion if isinstance(completion, GkCompletionModel) else GkCompletionModel.from_variant("default")
    sub = actions.loc[mask]
    sub_geom = geom.loc[mask]
    dens = _gk_completion_density(sub, frames, sub_geom, links)  # the one shared producer
    X = extract_gk_completion_features(sub_geom.assign(type_id=sub["type_id"].to_numpy()), defender_density=dens)
    return model.predict_proba(X)
```
  4. Add signature kwarg `completion: GkCompletionModel | None = None`.
  - **Unresolvable-destination decision (review C-a):** `coords_ok` requires a resolved origin AND destination. Goal-kicks always get a resolved origin (rule-point), so a row fails `coords_ok` only when its **destination** can't be resolved (no native end, no in-period next-event) — those are **excluded from `mask`** and the composite is **NaN (honest — no z' ⇒ no RAV/xT★(z'))**, NOT base-rated. So in the RAV path `geom_bad` never fires; the per-type base-rate fallback in `predict_proba` exists **only** for the standalone `compute_gk_completion` (a completion probability without geometry). **Qualify the coverage claim accordingly: "~100% of goal-kicks *with a resolvable destination*"** — the tiny unresolvable-dest slice is honestly NaN.
  5. Emit `xt_gk_origin_source`, `xt_gk_dest_source`, `xt_gk_origin_confidence` columns from `geom` (full-length, NaN/`"unresolved"` off-scope).
  6. Keep the M2 NaN-warn for any in-scope row that still can't be scored.

- [ ] **Step 4: run → pass.** Also update the existing `TestComputeXtGk` tests that asserted `[das]`/`get_xc` behaviour (the `_HAS_DAS` skip + the `get_xc`-frame-linking tests are now obsolete — delete/rewrite them; RAV no longer needs `[das]`). **Step 5: checkpoint** — full `tests/tracking/test_xt_gk.py` green.

## Task C2: provenance report + add_xt_gk/atomic columns

- [ ] `add_xt_gk` passes through the 3 new provenance columns (+ existing linkage provenance); atomic mirror carries them. Optional `XtGkReport` (counts per `*_source`) from a `compute_xt_gk(..., report=True)` or a small helper. Update `_XT_GK_COLS`-consuming tests + the atomic parity test. Checkpoint.

---

# PHASE D — gates, docs, suite

## Task D1: split coverage gate + parity/guard tests (review M5/M2/m6/R8/C1/C4)
- [ ] Add to `tests/tracking/test_gk_completion.py`:
  - **train==serve parity (C1 + P1-residual):** assert end-to-end equality for the same rows — `np.allclose(atol=1e-9)`, not "byte-identical" (R6). MUST cover **three** production stages, not just two:
    - **(geom)** `resolve_gk_geometry(full_actions).loc[mask]` (serve order) == the train path — specifically a **NaN-destination goalkick's `next_event` dest is identical** whether geometry is resolved on the full action list or the pre-masked subset (the P1-residual blind spot: build a fixture where masking changes which row is "next" and assert the dests match);
    - **(density)** the shared `_gk_completion_density` on an **unlinked-destination row (density → NaN)**;
    - **(features)** `extract_gk_completion_features` final assembly + the model's density mean-impute.
  - **degenerate-label guard** (done in B3) · **per-type fallback** (fixed in B2, C2) · **density mean-impute** (B2 P3).
- [ ] **Owner-gated e2e** (extend `test_xt_gk_e2e.py`):
  - **green criterion (M5 + review #1):** native-origin **pooled out-of-fold calibration** — AUC over ALL native rows (≈ hundreds), **lower-CI > 0.5**, `n_native` reported (floor-guarded). NOT a per-fold mean over tens of samples (statistically underpowered). This is the only correctness gate.
  - **informational only (review C4):** report the imputed-origin coverage fraction + the §1 "composite finite for X% of coord-resolved goalkicks (was 31%)" as **coverage-tracking with a drift alarm** — NOT a pass/green assertion (finiteness ≠ correctness, the M5 anti-pattern).

## Task D2: docs
- [ ] **ADR-024 amendment:** RAV P(success): get_xc → fitted `GkCompletionModel`; `[das]` optional; the GS-trained default + held-out AUC/Brier; coordinate derivation (conditional origin, scoped, provenance). **NOTICE:** add the completion-model + coordinate-derivation note under the Eyestone xT-GK entry. **CLAUDE.md:** extend the PR-S88 line. **C4:** `tracking` container — `GkCompletionModel` is a new trained(-light) model → add to the model token list + regen (count unchanged — no new `add_*` aggregator unless `add_gk_completion` is added; if so bump 26→27 and verify the invariant). **CHANGELOG 4.21.0:** extend the xT-GK entry with the goal-kick-coverage work.

## Task D3: full suite + lint + final checkpoint
- [ ] `.venv\Scripts\python.exe -m pytest tests/ -m "not e2e" -q --benchmark-skip` → all green. `ruff check` + `ruff format` + `pyright silly_kicks/` clean. Confirm the C4 aggregator-count invariant. Then the work folds into the **single sentinel-gated 4.21.0 commit** (P1-13 of the parent xT-GK plan) — HOLD for explicit approval.

---

## Plan review round 1 (parallel critic, 2026-06-08) — resolutions

The reviewer found the M2 train==serve discipline was applied to geometry + feature-assembly but **not** to three other axes:
- **P1 (train/serve DOMAIN mismatch) — FIXED.** `prepare` now uses the EXACT serve predicate `_gk_distribution_mask` (goalkicks + GK-actor pass/throw), not `goalkick|throw_in`; the dead overwritten domain line is gone (B3).
- **P2 (fit crashes on NaN geometry; prepare only filtered dest) — FIXED.** `prepare` drops geometry-unscoreable rows (NaN resolved origin+dest); serve base-rates exactly those (the asymmetry: serve tolerates, fit must drop) (B3, B2 guard).
- **P3 (density `-1` sentinel = OOD input + bypasses fallback) — FIXED.** No sentinel; density NaN → **training-mean impute** (neutral after standardization), and the per-type base rate is reserved for whole-row geometry-unscoreable rows (B1, B2). A mostly-NaN density is harmless (mean-imputed → inert), so this is correct whether or not density is usually available.
- **C1 (parity tested feature-assembly, not density production) — FIXED.** One shared `_gk_completion_density` producer used by both `prepare` and the serve `_completion_p`; parity test includes an unlinked→NaN-density row (D1).
- **C2 (vacuous `or True` test) — FIXED** (B2). **C3 (`_completion` prose-only) — SPELLED OUT** (C1). **C4 (≥95%-finite reintroduced finiteness-as-pass) — FIXED:** informational/drift only; native-origin calibration is the sole green gate (D1).
- **m2 (dead empirical-median constants) — REMOVED** (3 effective tiers, A1). **m3 (cache immutability) — NOTED** (B2). **m1/m6 — owner-data-measured** (GS goal-kick label split 31 fail / 62 success = 0.667 non-degenerate; density 94% finite; recorded in the ADR at B5).

### Plan review round 2 (parallel critic, 2026-06-08) — resolutions

- **P1-residual (geom produced on different-sized frames train vs serve → next_event dest skew) — FIXED.** `prepare` resolves geometry on the **FULL** action list then `.loc[mask]`, mirroring serve; `_next_event_start`'s positional `shift(-1)` is frame-size-dependent (B3). The **D1 parity test now covers geom production**, not just density+extract (the blind spot).
- **C-a (per-type base-rate dead in RAV; ~100% claim hid the unresolvable-dest slice) — DECIDED + STATED.** RAV **NaNs** unresolvable-destination goal-kicks (no z' ⇒ no RAV); the base-rate fallback serves only standalone `compute_gk_completion`; coverage claim qualified to "with a resolvable destination" (C1).
- **C-b (`_next_event_start` no boundary guard) — FIXED:** same-`(game_id, period_id)` guard NaNs cross-boundary shifts → honest `unresolved` (A1).
- **m-a (density `_std` compressed by mean-imputed rows) — FIXED:** standardize with `nanmean`/`nanstd` over present values, then impute (B2). **m-b — record the 94% density rate in the ADR** (B5).

### Plan review round 3 (parallel critic, 2026-06-08) — verdict ship-quality; residuals folded

- **#1 (sole green gate may be underpowered — native is the 33% minority; per-fold ≈ tens) — FIXED:** the native-origin calibration gate is now **pooled out-of-fold** (one AUC over ALL native rows ≈ hundreds) with a **bootstrap CI lower-bound > 0.5** + reported `n_native` (floor-guarded) — sample-size-aware, not a per-fold mean (B5, D1).
- **#2 (domain parity missing the id_ok axis) — FIXED:** `prepare` also drops NaN-id rows (strict parity with serve's `id_ok`) (B3).
- **#3 (R5 fallback integration-tested nowhere) — FIXED:** a geometry-unscoreable row through `compute_gk_completion` asserts the per-type base rate (its only live path, since RAV NaNs such rows) (B4).
- **#4 (`is_throw_in` near-inert on GS) — NOTED:** kept for forward-compat; the ~0 coefficient documented so it's not read as a bug (B4).
- Reviewer verdict: *"Converged and implementable… after #1–#3 are folded in, this is ready for the single 4.21.0 commit."*

## Self-review notes
- **Spec coverage:** Part A → A1/C1; Part B model → B1–B5; D2 logistic → B2; provenance → C1/C2/§4; M2 parity → B1/D1; M3 conditional origin → A1; M4 [das] removal → C1; M5 split gate → D1; m6 degenerate guard → B3; R1 GS default → B5; R2 density-availability → B1 (fill) + D1 (parity); R3 tier-2 clamp → A1; R5 per-type fallback → B2; R6 atol parity → B2/D1; R7 confidence col → A1/C1; R8 [das]-absent test → C1; R9 → D2 note.
- **Open verification items (runtime, not placeholders):** confirm `receiver_zone_density` signature on the resolved-dest path; confirm `resolve_frame_ids_by_position` import; the train run's actual AUC/Brier (record in ADR); whether to add `add_gk_completion` to `__all__` (C4 count impact).
- **Deferred:** multi-provider default (R1 follow-up), general converter-level coordinate enrichment (TODO), XGBoost upgrade (D2 reserve).
