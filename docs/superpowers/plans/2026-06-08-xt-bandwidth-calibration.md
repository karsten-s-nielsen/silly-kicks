# SK-xT-3: xT bandwidth/resolution HPO sweep — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `ruthless`/Optuna sweep over xT `KDEParams.bandwidth` × `GridSpec` resolution × `adaptive` that, per corpus, minimizes K-fold held-out transition-NLL and recommends a `KDEParams`+`GridSpec` via an auditable manifest — changing no library default (ADR-009).

**Architecture:** Refactor `xthreat/_transitions.py` so the gaussian KDE is a *shared, vectorized* seam (`_gaussian_transition_from_grouped`) that both the library and a new calibration objective call; the objective caches the small param-invariant *grouped destinations* per `(grid, fold)` and re-runs only the cheap seam per trial. A CLI in `scripts/` owns I/O + manifest + a reported downstream xT-quality cross-check.

**Tech Stack:** numpy, pandas, scikit-learn (sklearn KDE retained only for non-gaussian kernels), `ruthless-efficiency` (Optuna strategy), scipy (`spearmanr`, already a dep via xthreat interpolator).

**Spec:** `docs/superpowers/specs/2026-06-08-xt-bandwidth-calibration-design.md`

---

## ⚠️ Repository policy notes (override the generic skill defaults)

- **ONE commit per branch, after `/final-review`, with explicit owner approval.** Do **NOT** commit per task. Each task ends by *staging* (`git add`) only. The single commit is Task 8, gated on approval.
- Branch already exists: `pr-s87-xt-bandwidth-calibration`.
- After each task run the relevant tests; before the final commit run the full Shift-Left gate (`ruff format --check`, `ruff check`, `pyright silly_kicks/`, `pytest -m "not e2e"`).

## File structure

| File | Create/Modify | Responsibility |
|---|---|---|
| `silly_kicks/xthreat/_transitions.py` | Modify | Extract `_bin_destinations_by_source` (vectorized grouping) + `_gaussian_transition_from_grouped` (shared vectorized gaussian seam) + `_kde_transition_from_grouped` (dispatch); `kde_smoothed_transition_matrix` composes them |
| `silly_kicks/calibration/_spaces.py` | Modify | Add `_GRIDS`, `grid_from_str`, `xt_bandwidth_config` |
| `silly_kicks/calibration/_xt_bandwidth_objective.py` | Create | `XtBandwidthObjective` — duck-typed `evaluate`, lazy per-`(grid,fold)` grouped cache, CV NLL + Singh baseline |
| `silly_kicks/calibration/__init__.py` | Modify | Export `XtBandwidthObjective`, `xt_bandwidth_config`, `grid_from_str` |
| `scripts/calibrate_xt_bandwidth.py` | Create | CLI: corpus loaders, `run_xt_bandwidth` seam, `build_manifest`, `xt_quality_cross_check`, report |
| `tests/test_xthreat_kde_vectorized.py` | Create | sklearn-parity (non-underflow) + small-`h` finiteness + library-composes-seam + scalar-NLL characterization pin |
| `tests/calibration/test_spaces.py` | Modify | `xt_bandwidth_config` shape + every `_GRIDS` parses |
| `tests/calibration/test_xt_bandwidth_objective.py` | Create | NLL determinism, CV, Singh baseline, empty-fold, dtype, cache-equivalence, structural perf guard, round-trip constructibility |
| `tests/calibration/test_calibrate_xt_bandwidth_cli.py` | Create | `run_xt_bandwidth` smoke + `build_manifest` shape + `xt_quality_cross_check` finite |
| `docs/superpowers/adrs/ADR-009-calibration-harness.md` | Modify | Amend: xT-NLL objective; CachedObjective divergence (M4) + re-pin Chesterton evidence |
| `CHANGELOG.md`, `pyproject.toml`, `silly_kicks/__init__.py`, `TODO.md` | Modify | 4.20.0 bump + grooming |

---

## Task 1: Refactor `_transitions.py` — shared vectorized gaussian seam

**Files:**
- Modify: `silly_kicks/xthreat/_transitions.py`
- Test: `tests/test_xthreat_kde_vectorized.py` (create)

### Background the implementer needs

- `kde_smoothed_transition_matrix(actions, grid, params)` currently (lines 79–131): per source zone, masks `end_xy[start_cell == s]` (O(n_zones·n_actions) loop, review M1), fits sklearn `KernelDensity`, scores at `_zone_centres(grid)`, normalizes, and fills unpopulated zones with the populated mean row.
- It has **exactly one caller** (`silly_kicks/xthreat/_model.py:123`) and `ExpectedThreat` defaults to `singh_counts` — so re-pinning the gaussian numerics breaks no shipped artifact (Chesterton-verified, spec §0).
- Existing helpers in this module: `silverman_2d(n, sigma)`, `_zone_centres(grid)`. From `_grid`: `_get_flat_indexes`, `_get_successful_move_actions`.
- **Numerical key point (review M5):** for the gaussian kernel, sklearn's density divided by its row sum equals `softmax(-D²/2h²)` over destination centres (the `1/(n·2πh²)` constant cancels). The vectorized path must subtract a **scalar global max** of the logits per source zone before `exp` (cancels in the row-normalization, prevents small-`h` underflow). It must **not** subtract a per-centre max — that would rescale centres unequally and corrupt the distribution.
- At extreme small `h` (e.g. `adaptive=False, bandwidth=0.1`), the *old* sklearn-wrapper underflowed (`np.exp(score_samples)` → 0 → mean-row fallback). The vectorized path instead returns the true peaky distribution — **strictly more correct** and what the optimizer needs (a tiny bandwidth must score as genuinely bad NLL, not be masked by a mean-row artifact). So sklearn-parity is asserted only in the non-underflow regime; the underflow regime gets a finiteness/row-stochastic assertion.

- [ ] **Step 1: Write the failing parity + finiteness tests**

Create `tests/test_xthreat_kde_vectorized.py`:

```python
"""SK-xT-3 Task 1: the vectorized gaussian KDE seam == sklearn gaussian (row-normalized) in the
non-underflow regime, and is finite/stable in the small-h underflow regime where the old
sklearn-wrapper degenerated to the mean-row fallback. See the spec §0 (review M5)."""

import numpy as np
import pytest
from sklearn.neighbors import KernelDensity

from silly_kicks.xthreat import KDEParams
from silly_kicks.xthreat._eval import compute_holdout_nll, holdout_split
from silly_kicks.xthreat._params import GridSpec
from silly_kicks.xthreat._transitions import (
    _bin_destinations_by_source,
    _gaussian_transition_from_grouped,
    kde_smoothed_transition_matrix,
)
from tests._xthreat_helpers import _moves, _sparse_overfit_corpus


def _sklearn_reference(grouped, centres, grid, params):
    """The old per-zone sklearn gaussian path, row-normalized — the parity reference."""
    n = grid.n_zones_x * grid.n_zones_y
    from silly_kicks.xthreat._transitions import silverman_2d

    T = np.zeros((n, n))
    populated = []
    for s, pts in grouped.items():
        if pts.shape[0] == 0:
            continue
        if params.adaptive:
            sigma = float(np.sqrt((pts[:, 0].var() + pts[:, 1].var()) / 2.0)) or 1e-6
            h = params.bandwidth * silverman_2d(pts.shape[0], sigma)
        else:
            h = params.bandwidth
        dens = np.exp(KernelDensity(kernel="gaussian", bandwidth=h).fit(pts).score_samples(centres))
        if dens.sum() > 0:
            T[s] = dens / dens.sum()
            populated.append(s)
    if populated:
        mean_row = T[populated].mean(axis=0)
        mean_row = mean_row / mean_row.sum() if mean_row.sum() > 0 else np.full(n, 1.0 / n)
        for s in range(n):
            if s not in populated:
                T[s] = mean_row
    return T


@pytest.mark.parametrize("adaptive,bandwidth", [(True, 0.5), (True, 1.0), (True, 2.0), (False, 2.0), (False, 5.0)])
def test_vectorized_gaussian_matches_sklearn_non_underflow(adaptive, bandwidth):
    grid = GridSpec(6, 4)
    grouped, centres = _bin_destinations_by_source(_moves(n_per_zone=120), grid)
    params = KDEParams(bandwidth=bandwidth, adaptive=adaptive)
    vec = _gaussian_transition_from_grouped(grouped, centres, grid, params)
    ref = _sklearn_reference(grouped, centres, grid, params)
    np.testing.assert_allclose(vec, ref, rtol=0, atol=1e-9)


def test_vectorized_gaussian_finite_in_underflow_regime():
    # adaptive=False, bandwidth=0.1 (raw metres) -> the old sklearn wrapper underflowed to 0 -> mean
    # row. The vectorized path must stay FINITE and row-stochastic (strictly more correct).
    grid = GridSpec(6, 4)
    grouped, centres = _bin_destinations_by_source(_moves(n_per_zone=120), grid)
    T = _gaussian_transition_from_grouped(grouped, centres, grid, KDEParams(bandwidth=0.1, adaptive=False))
    assert np.all(np.isfinite(T))
    np.testing.assert_allclose(T.sum(axis=1), np.ones(grid.n_zones), atol=1e-9)


def test_library_composes_the_shared_seam():
    # M6: kde_smoothed_transition_matrix bottoms out in the same seam — definitional, not a gate.
    grid = GridSpec(6, 4)
    df = _moves(n_per_zone=120)
    params = KDEParams(bandwidth=1.5, adaptive=True)
    grouped, centres = _bin_destinations_by_source(df, grid)
    np.testing.assert_array_equal(
        kde_smoothed_transition_matrix(df, grid, params),
        _gaussian_transition_from_grouped(grouped, centres, grid, params),
    )


def test_kde_holdout_nll_characterization_pin():
    # Scalar golden: pins the re-pinned gaussian numerics against accidental future drift, tolerant
    # to numpy micro-version noise. Value generated from the committed implementation (Step 6).
    df = _sparse_overfit_corpus(seed=3, n_games=20)
    train, holdout = holdout_split(df, holdout_fraction=0.25)
    grid = GridSpec(16, 12)
    T = kde_smoothed_transition_matrix(train, grid, KDEParams(bandwidth=1.0, adaptive=True))
    nll = compute_holdout_nll(T, holdout, grid=grid)
    assert nll == pytest.approx(_EXPECTED_NLL, abs=1e-4)


_EXPECTED_NLL = 0.0  # FILLED IN at Step 6 from the committed implementation
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_xthreat_kde_vectorized.py -x -q`
Expected: FAIL — `ImportError: cannot import name '_bin_destinations_by_source'`.

- [ ] **Step 3: Implement the refactor**

In `silly_kicks/xthreat/_transitions.py`, **replace** the body of `kde_smoothed_transition_matrix` (currently lines ~79–131) with the three helpers below + a thin composing wrapper. Keep `singh_transition_matrix`, `silverman_2d`, `_zone_centres` as-is. Add `from sklearn.neighbors import KernelDensity` only inside the non-gaussian branch (keep the top-level import lazy as today).

```python
def _bin_destinations_by_source(
    actions: pd.DataFrame,
    grid: GridSpec,
    *,
    max_points_per_zone: int | None = None,
    rng_seed: int | None = None,
) -> tuple[dict[int, npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
    """Group successful-move destinations by source zone in a SINGLE vectorized pass.

    Returns ``(grouped, centres)`` where ``grouped[s]`` is the ``(n_s, 2)`` destination coords of
    moves starting in flat zone ``s`` and ``centres`` is ``(n_zones, 2)``. ``grouped`` is the small
    param-invariant artifact the calibration objective caches (NOT pairwise D², which is
    ``n_s x n_zones`` and OOMs at scale — spec C2'). ``argsort + split`` replaces the legacy
    ``for s: end_xy[start_cell == s]`` mask-in-loop (review M1). Optional deterministic per-zone
    subsample bounds per-trial cdist FLOPs / pathological-zone memory (review C2'/N6); default
    ``(None, None)`` keeps every row (byte-identical grouping to the legacy binning).

    Examples
    --------
    Group a small SPADL corpus by source zone::

        from silly_kicks.xthreat import GridSpec
        from silly_kicks.xthreat._transitions import _bin_destinations_by_source

        grouped, centres = _bin_destinations_by_source(actions, GridSpec(16, 12))
    """
    l, w = grid.n_zones_x, grid.n_zones_y
    centres = _zone_centres(grid)
    move = _get_successful_move_actions(actions).dropna(subset=["start_x", "start_y", "end_x", "end_y"])
    if len(move) == 0:
        return {}, centres
    start_cell = _get_flat_indexes(move.start_x, move.start_y, l, w).to_numpy()
    end_xy = move[["end_x", "end_y"]].to_numpy(dtype=np.float64)
    order = np.argsort(start_cell, kind="stable")
    sc_sorted = start_cell[order]
    end_sorted = end_xy[order]
    boundaries = np.flatnonzero(np.diff(sc_sorted)) + 1
    zone_per_group = sc_sorted[np.concatenate(([0], boundaries))]
    groups = np.split(end_sorted, boundaries)
    rng = np.random.default_rng(rng_seed)
    grouped: dict[int, npt.NDArray[np.float64]] = {}
    for s, pts in zip(zone_per_group, groups):
        if max_points_per_zone is not None and len(pts) > max_points_per_zone:
            pts = pts[rng.choice(len(pts), size=max_points_per_zone, replace=False)]
        grouped[int(s)] = pts
    return grouped, centres


def _gaussian_transition_from_grouped(
    grouped: dict[int, npt.NDArray[np.float64]],
    centres: npt.NDArray[np.float64],
    grid: GridSpec,
    params: KDEParams,
) -> npt.NDArray[np.float64]:
    """SHARED vectorized gaussian KDE seam — called by both the library core and the calibration
    objective (review M6: equivalence is definitional, one function).

    Per source zone with destinations ``pts``: ``logits = -D2 / (2h^2)`` where ``D2`` is the
    ``(n_zones, n_s)`` pairwise squared distance from centres to pts; subtract the SCALAR global max
    of ``logits`` (softmax stabilization — cancels in the row-normalization, prevents small-h
    underflow; a per-centre max would corrupt the distribution); ``dens = exp(stabilized).sum``
    over pts; row-normalize. Unpopulated zones get the populated mean row (matches the legacy
    sklearn path's ``if total > 0 else mean-row`` branch).

    Examples
    --------
    Build a gaussian transition matrix from pre-grouped destinations::

        from silly_kicks.xthreat import GridSpec, KDEParams
        from silly_kicks.xthreat._transitions import _bin_destinations_by_source, _gaussian_transition_from_grouped

        grouped, centres = _bin_destinations_by_source(actions, GridSpec(16, 12))
        T = _gaussian_transition_from_grouped(grouped, centres, GridSpec(16, 12), KDEParams())
    """
    n = grid.n_zones_x * grid.n_zones_y
    T = np.zeros((n, n), dtype=np.float64)
    populated: list[int] = []
    for s, pts in grouped.items():
        if pts.shape[0] == 0:
            continue
        if params.adaptive:
            sigma = float(np.sqrt((pts[:, 0].var() + pts[:, 1].var()) / 2.0))
            if sigma == 0.0:
                sigma = 1e-6
            h = params.bandwidth * silverman_2d(pts.shape[0], sigma)
        else:
            h = params.bandwidth
        d2 = ((centres[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)  # (n_zones, n_s)
        logits = -d2 / (2.0 * h * h)
        logits = logits - logits.max()  # SCALAR global max — stabilize, cancels in normalization
        dens = np.exp(logits).sum(axis=1)  # (n_zones,)
        total = dens.sum()
        if total > 0:
            T[s] = dens / total
            populated.append(s)
    if populated:
        mean_row = T[populated].mean(axis=0)
        s_mean = mean_row.sum()
        mean_row = mean_row / s_mean if s_mean > 0 else np.full(n, 1.0 / n)
        for s in range(n):
            if s not in populated:
                T[s] = mean_row
    else:
        T[:] = 1.0 / n
    return T


def _kde_transition_from_grouped(
    grouped: dict[int, npt.NDArray[np.float64]],
    centres: npt.NDArray[np.float64],
    grid: GridSpec,
    params: KDEParams,
) -> npt.NDArray[np.float64]:
    """Dispatch the KDE core on ``params.kernel``: ``"gaussian"`` -> the vectorized shared seam;
    any other kernel -> the sklearn ``KernelDensity`` fallback (unchanged generality).

    Examples
    --------
    ::

        from silly_kicks.xthreat import GridSpec, KDEParams
        from silly_kicks.xthreat._transitions import _bin_destinations_by_source, _kde_transition_from_grouped

        grouped, centres = _bin_destinations_by_source(actions, GridSpec(16, 12))
        T = _kde_transition_from_grouped(grouped, centres, GridSpec(16, 12), KDEParams())
    """
    if params.kernel == "gaussian":
        return _gaussian_transition_from_grouped(grouped, centres, grid, params)
    from sklearn.neighbors import KernelDensity

    n = grid.n_zones_x * grid.n_zones_y
    T = np.zeros((n, n), dtype=np.float64)
    populated: list[int] = []
    for s, pts in grouped.items():
        if pts.shape[0] == 0:
            continue
        if params.adaptive:
            sigma = float(np.sqrt((pts[:, 0].var() + pts[:, 1].var()) / 2.0))
            if sigma == 0.0:
                sigma = 1e-6
            h = params.bandwidth * silverman_2d(pts.shape[0], sigma)
        else:
            h = params.bandwidth
        dens = np.exp(KernelDensity(kernel=params.kernel, bandwidth=h).fit(pts).score_samples(centres))
        total = dens.sum()
        if total > 0:
            T[s] = dens / total
            populated.append(s)
    if populated:
        mean_row = T[populated].mean(axis=0)
        s_mean = mean_row.sum()
        mean_row = mean_row / s_mean if s_mean > 0 else np.full(n, 1.0 / n)
        for s in range(n):
            if s not in populated:
                T[s] = mean_row
    else:
        T[:] = 1.0 / n
    return T
```

Then replace the public function body with the composition (keep its existing docstring/signature):

```python
def kde_smoothed_transition_matrix(actions: pd.DataFrame, grid: GridSpec, params: KDEParams) -> npt.NDArray[np.float64]:
    # ... keep the existing docstring ...
    grouped, centres = _bin_destinations_by_source(actions, grid)
    return _kde_transition_from_grouped(grouped, centres, grid, params)
```

- [ ] **Step 4: Run the parity + finiteness + composition tests**

Run: `python -m pytest tests/test_xthreat_kde_vectorized.py -q -k "not characterization"`
Expected: PASS (parity, finiteness, composition).

- [ ] **Step 5: Run the existing xthreat suite (refactor regression net)**

Run: `python -m pytest tests/test_xthreat.py tests/test_xthreat_kde.py tests/test_xthreat_kde_beats_singh.py tests/test_xthreat_resolution.py tests/test_xthreat_eval.py -q`
Expected: PASS — behavioral KDE tests (row-stochastic, entropy ordering, zero-event fallback) still hold; the re-pin only changes values within 1e-9 in the moderate-`h` regime they exercise.

- [ ] **Step 6: Fill the characterization pin**

Run once to read the value:
`python -c "import numpy as np; from tests._xthreat_helpers import _sparse_overfit_corpus; from silly_kicks.xthreat import KDEParams; from silly_kicks.xthreat._params import GridSpec; from silly_kicks.xthreat._transitions import kde_smoothed_transition_matrix; from silly_kicks.xthreat._eval import holdout_split, compute_holdout_nll; df=_sparse_overfit_corpus(seed=3,n_games=20); tr,ho=holdout_split(df,holdout_fraction=0.25); g=GridSpec(16,12); print(repr(compute_holdout_nll(kde_smoothed_transition_matrix(tr,g,KDEParams(bandwidth=1.0,adaptive=True)),ho,grid=g)))"`
Set `_EXPECTED_NLL` in the test to the printed float. Then:
Run: `python -m pytest tests/test_xthreat_kde_vectorized.py -q`
Expected: PASS (all, incl. characterization).

- [ ] **Step 7: Stage (no commit — policy)**

```bash
git add silly_kicks/xthreat/_transitions.py tests/test_xthreat_kde_vectorized.py
```

---

## Task 2: `_spaces.py` — grid parsing + `xt_bandwidth_config`

**Files:**
- Modify: `silly_kicks/calibration/_spaces.py`
- Test: `tests/calibration/test_spaces.py`

### Background
- `ruthless` verified signatures: `Choice(kind="choice", choices=<tuple>)`, `FloatRange(kind="float", lo, hi, log=False)`, `IntRange(kind="int", lo, hi)`. **Note `kind="choice"`, not `"categorical"`.**
- `OptunaConfig` validates `warm_start ⊆ param_space`; warm-start must use a real `_GRIDS` member.

- [ ] **Step 1: Write the failing tests** — append to `tests/calibration/test_spaces.py`:

```python
def test_xt_bandwidth_config_minimizes_nll_over_three_axes():
    from ruthless import Direction
    from silly_kicks.calibration._spaces import xt_bandwidth_config

    cfg = xt_bandwidth_config(n_trials=10, store_path="xt.db")
    assert cfg.metric == "xt_holdout_nll"
    assert cfg.direction is Direction.MINIMIZE
    assert set(cfg.param_space) == {"bandwidth", "adaptive", "grid"}
    assert cfg.param_space["bandwidth"].log is True
    assert set(cfg.warm_start) == {"bandwidth", "adaptive", "grid"}
    assert cfg.warm_start["grid"] == "16x12"


def test_every_grid_member_parses_to_valid_gridspec():
    from silly_kicks.calibration._spaces import _GRIDS, grid_from_str
    from silly_kicks.xthreat import GridSpec

    for s in _GRIDS:
        g = grid_from_str(s)
        assert isinstance(g, GridSpec)
        assert g.n_zones_x >= 1 and g.n_zones_y >= 1
    # the default/warm-start grid is a member
    assert "16x12" in _GRIDS
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/calibration/test_spaces.py -q -k "xt_bandwidth or grid_member"`
Expected: FAIL — `ImportError: cannot import name 'xt_bandwidth_config'`.

- [ ] **Step 3: Implement** — add to `silly_kicks/calibration/_spaces.py` (extend the imports at the top: `from ruthless import Choice, Direction, FloatRange, OptunaConfig` and `from silly_kicks.xthreat import GridSpec`):

```python
# Aspect-sane grids near the pitch's ~1.54 ratio (105x68). Resolution is SWEPT (spec M2) over this
# curated discrete set rather than two independent IntRanges (~475 cells, admits non-physical 32x6).
_GRIDS: tuple[str, ...] = ("12x8", "16x12", "20x14", "24x16", "28x18", "32x20")


def grid_from_str(s: str) -> GridSpec:
    """Parse a ``"<nx>x<ny>"`` grid string into a ``GridSpec`` (e.g. ``"16x12"`` -> 16x12).

    Examples
    --------
    >>> from silly_kicks.calibration._spaces import grid_from_str
    >>> grid_from_str("16x12").n_zones
    192
    """
    nx, ny = s.lower().split("x")
    return GridSpec(n_zones_x=int(nx), n_zones_y=int(ny))


def xt_bandwidth_config(*, n_trials: int, store_path: str, sampler: Literal["tpe", "random"] = "tpe") -> OptunaConfig:
    """SK-xT-3 — held-out xT transition-NLL sweep (minimize): bandwidth x adaptive x grid.

    Examples
    --------
    >>> from silly_kicks.calibration._spaces import xt_bandwidth_config
    >>> xt_bandwidth_config(n_trials=10, store_path="/tmp/xt.db").metric
    'xt_holdout_nll'
    """
    return OptunaConfig(
        kind="optuna",
        metric="xt_holdout_nll",
        direction=Direction.MINIMIZE,
        n_trials=n_trials,
        sampler=sampler,
        param_space={
            "bandwidth": FloatRange(kind="float", lo=0.1, hi=20.0, log=True),
            "adaptive": Choice(kind="choice", choices=(True, False)),
            "grid": Choice(kind="choice", choices=_GRIDS),
        },
        warm_start={"bandwidth": 1.0, "adaptive": True, "grid": "16x12"},
        store=StoreConfig(kind="sqlite", path=store_path),
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/calibration/test_spaces.py -q`
Expected: PASS (existing + new).

- [ ] **Step 5: Stage**

```bash
git add silly_kicks/calibration/_spaces.py tests/calibration/test_spaces.py
```

---

## Task 3: `XtBandwidthObjective` core + unit tests

**Files:**
- Create: `silly_kicks/calibration/_xt_bandwidth_objective.py`
- Test: `tests/calibration/test_xt_bandwidth_objective.py` (create)

### Background
- Mirror `CarrierAccuracyObjective`: plain object, `evaluate(candidate) -> Metrics`, internal lazy cache (NOT a `ruthless.CachedObjective` — resolution-keyed invariant).
- CV via `match_cv_splits(game_ids)` from `silly_kicks.calibration._cv`; SE via `cv_standard_error` (returns `nan` for <2 folds — N3).
- NLL eps `1e-10` matches `compute_holdout_nll`.
- The KDE seam + binning come from `_transitions`; the Singh baseline from `singh_transition_matrix` (different action filter — `_get_move_actions` — so it can't share the KDE grouped cache; N1).

- [ ] **Step 1: Write the failing unit tests**

Create `tests/calibration/test_xt_bandwidth_objective.py`:

```python
"""SK-xT-3 Task 3: XtBandwidthObjective unit behavior."""

import math

import numpy as np
import pytest
from ruthless import Candidate

from silly_kicks.calibration._xt_bandwidth_objective import XtBandwidthObjective
from tests._xthreat_helpers import _sparse_overfit_corpus


def _cand(bandwidth=1.0, adaptive=True, grid="16x12"):
    return Candidate(id="t0", params={"bandwidth": bandwidth, "adaptive": adaptive, "grid": grid})


def test_evaluate_returns_finite_nll_and_singh_baseline():
    obj = XtBandwidthObjective(_sparse_overfit_corpus(seed=1, n_games=20), seed=42)
    m = obj.evaluate(_cand())
    assert math.isfinite(m["xt_holdout_nll"])
    assert math.isfinite(m["singh_holdout_nll"])
    assert m["n_folds"] >= 2
    assert m["n_holdout_moves"] > 0


def test_evaluate_is_deterministic():
    obj = XtBandwidthObjective(_sparse_overfit_corpus(seed=1, n_games=20), seed=42)
    assert obj.evaluate(_cand())["xt_holdout_nll"] == obj.evaluate(_cand())["xt_holdout_nll"]


def test_grid_axis_changes_the_score():
    obj = XtBandwidthObjective(_sparse_overfit_corpus(seed=1, n_games=20), seed=42)
    a = obj.evaluate(_cand(grid="12x8"))["xt_holdout_nll"]
    b = obj.evaluate(_cand(grid="24x16"))["xt_holdout_nll"]
    assert a != b


def test_string_and_int_game_id_both_work():
    # provider-asymmetric game_id dtype must not crash CV grouping / NLL (spec NaN/dtype).
    df = _sparse_overfit_corpus(seed=2, n_games=20)
    int_score = XtBandwidthObjective(df, seed=42).evaluate(_cand())["xt_holdout_nll"]
    df_str = df.copy()
    df_str["game_id"] = df_str["game_id"].astype(str)
    str_score = XtBandwidthObjective(df_str, seed=42).evaluate(_cand())["xt_holdout_nll"]
    assert math.isfinite(int_score) and math.isfinite(str_score)


def test_no_signal_corpus_scores_inf_not_crash():
    # A corpus with no eligible holdout MOVES competes honestly as the worst score, never crashes.
    # NOTE: needs >=2 games — match_cv_splits uses LeaveOneGroupOut for <=7 games, which RAISES on a
    # single group. Two shot-only games => every fold's holdout has 0 moves => all excluded => inf.
    import pandas as pd

    import silly_kicks.spadl.config as cfg

    cols = ["game_id", "action_id", "period_id", "time_seconds", "team_id", "player_id",
            "bodypart_id", "type_id", "result_id", "start_x", "start_y", "end_x", "end_y"]
    shot, fail = cfg.actiontype_id["shot"], cfg.result_id["fail"]
    rows = [
        [1, 0, 1, 0.0, 1, 1, 0, shot, fail, 95.0, 34.0, 105.0, 34.0],
        [2, 1, 1, 0.0, 1, 1, 0, shot, fail, 95.0, 34.0, 105.0, 34.0],
    ]
    m = XtBandwidthObjective(pd.DataFrame(rows, columns=cols), seed=42).evaluate(_cand())
    assert m["xt_holdout_nll"] == float("inf")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/calibration/test_xt_bandwidth_objective.py -x -q`
Expected: FAIL — `ModuleNotFoundError: silly_kicks.calibration._xt_bandwidth_objective`.

- [ ] **Step 3: Implement the objective**

Create `silly_kicks/calibration/_xt_bandwidth_objective.py`:

```python
"""SK-xT-3 xT bandwidth/resolution HPO objective — held-out transition NLL (MINIMIZE).

Plain duck-typed object (NOT a ruthless.CachedObjective — the resolution axis means the invariant
is keyed by grid, which a single prepare() does not model; spec M4). Caches per-(grid, fold) the
small grouped destinations (NOT D²; spec C2') and re-runs only the shared vectorized gaussian seam
per trial — the same seam the library bottoms out in (definitional equivalence; spec M6).

See docs/superpowers/specs/2026-06-08-xt-bandwidth-calibration-design.md.

Examples
--------
>>> from silly_kicks.calibration._xt_bandwidth_objective import XtBandwidthObjective
>>> from ruthless import Candidate
>>> # obj = XtBandwidthObjective(actions, seed=42)  # actions: SPADL + game_id
>>> # obj.evaluate(Candidate(id="t0", params={"bandwidth": 1.0, "adaptive": True, "grid": "16x12"}))
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from ruthless.result import Candidate, Metrics

from silly_kicks.calibration._cv import cv_standard_error, match_cv_splits
from silly_kicks.calibration._spaces import grid_from_str
from silly_kicks.xthreat._eval import compute_holdout_nll
from silly_kicks.xthreat._grid import _get_flat_indexes, _get_successful_move_actions
from silly_kicks.xthreat._params import KDEParams
from silly_kicks.xthreat._transitions import (
    _bin_destinations_by_source,
    _gaussian_transition_from_grouped,
    singh_transition_matrix,
)

_EPS = 1e-10  # matches compute_holdout_nll


@dataclass
class _PreparedFold:
    """Param-invariant per-(grid, fold) state: cached grouped train destinations + centres, the
    held-out (src, dst) flat-zone indexes, and the param-free Singh held-out NLL."""

    grouped: dict[int, npt.NDArray[np.float64]]
    centres: npt.NDArray[np.float64]
    holdout_src: npt.NDArray[np.int_]
    holdout_dst: npt.NDArray[np.int_]
    singh_nll: float
    n_holdout_moves: int


class XtBandwidthObjective:
    """ruthless-compatible objective (MINIMIZE ``xt_holdout_nll``).

    Examples
    --------
    >>> from silly_kicks.calibration._xt_bandwidth_objective import XtBandwidthObjective
    >>> from ruthless import Candidate
    >>> # obj = XtBandwidthObjective(actions, seed=42)
    >>> # obj.evaluate(Candidate(id="t0", params={"bandwidth": 1.0, "adaptive": True, "grid": "16x12"}))
    """

    def __init__(self, actions: pd.DataFrame, *, seed: int = 42, max_points_per_zone: int | None = None) -> None:
        self._actions = actions.reset_index(drop=True)
        self._seed = seed
        self._cap = max_points_per_zone
        self.diagnostics: dict = {}
        self._prepared: dict[tuple[str, int], _PreparedFold] = {}
        # CV folds over game_id — computed once (invariant across all trials).
        self._game_ids = self._actions["game_id"].to_numpy()
        self._folds = match_cv_splits(self._game_ids)

    def _prepare(self, grid_str: str, train_idx, test_idx) -> _PreparedFold:
        grid = grid_from_str(grid_str)
        train = self._actions.iloc[train_idx]
        test = self._actions.iloc[test_idx]
        grouped, centres = _bin_destinations_by_source(
            train, grid, max_points_per_zone=self._cap, rng_seed=self._seed
        )
        move = _get_successful_move_actions(test).dropna(subset=["start_x", "start_y", "end_x", "end_y"])
        l, w = grid.n_zones_x, grid.n_zones_y
        src = _get_flat_indexes(move.start_x, move.start_y, l, w).to_numpy()
        dst = _get_flat_indexes(move.end_x, move.end_y, l, w).to_numpy()
        # Singh baseline (param-free) on the SAME split — N1: its filter (_get_move_actions) differs
        # from the KDE grouped cache, so it is computed via the library function, not the cache.
        singh_nll = compute_holdout_nll(singh_transition_matrix(train, grid), test, grid=grid)
        return _PreparedFold(grouped, centres, src, dst, float(singh_nll), int(len(move)))

    def evaluate(self, candidate: Candidate) -> Metrics:
        """K-fold held-out transition NLL for a (bandwidth, adaptive, grid) candidate."""
        p = candidate.params
        bandwidth, adaptive, grid_str = float(p["bandwidth"]), bool(p["adaptive"]), str(p["grid"])
        grid = grid_from_str(grid_str)
        params = KDEParams(bandwidth=bandwidth, adaptive=adaptive)  # kernel defaults to gaussian
        kde_nlls: list[float] = []
        singh_nlls: list[float] = []
        n_moves = 0
        for fi, (tr, te) in enumerate(self._folds):
            key = (grid_str, fi)
            prep = self._prepared.get(key)
            if prep is None:
                prep = self._prepare(grid_str, tr, te)
                self._prepared[key] = prep
            if prep.n_holdout_moves == 0:
                continue  # empty holdout fold excluded from the mean (spec N3)
            T = _gaussian_transition_from_grouped(prep.grouped, prep.centres, grid, params)
            probs = T[prep.holdout_src, prep.holdout_dst]
            kde_nlls.append(float(-np.mean(np.log(np.maximum(probs, _EPS)))))
            singh_nlls.append(prep.singh_nll)
            n_moves += prep.n_holdout_moves
        if not kde_nlls:
            return {"xt_holdout_nll": float("inf")}  # no-signal: worst score, competes honestly
        return {
            "xt_holdout_nll": float(np.mean(kde_nlls)),
            "xt_holdout_nll_se": cv_standard_error(kde_nlls),
            "singh_holdout_nll": float(np.mean(singh_nlls)),
            "n_folds": float(len(kde_nlls)),
            "n_holdout_moves": float(n_moves),
        }
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/calibration/test_xt_bandwidth_objective.py -q`
Expected: PASS.

- [ ] **Step 5: Stage**

```bash
git add silly_kicks/calibration/_xt_bandwidth_objective.py tests/calibration/test_xt_bandwidth_objective.py
```

---

## Task 4: Objective guards — cache-equivalence, structural perf, round-trip

**Files:**
- Test: `tests/calibration/test_xt_bandwidth_objective.py` (extend)

### Background
- `tests/_perf_structural.call_counter(monkeypatch, module, name)` patches `module.name` with a pass-through spy returning `{"n": count}`. Patch the symbol **in the objective module's namespace** (it imports the helpers by name), i.e. `silly_kicks.calibration._xt_bandwidth_objective`.
- Cache-equivalence here = warm-cache `evaluate` (second call, and a fresh object) gives identical metrics; the per-trial seam is deterministic, so the cache cannot alter results.

- [ ] **Step 1: Write the failing guard tests** — append to `tests/calibration/test_xt_bandwidth_objective.py`:

```python
def test_cache_equivalence_warm_equals_cold():
    df = _sparse_overfit_corpus(seed=5, n_games=20)
    cold = XtBandwidthObjective(df, seed=42).evaluate(_cand(bandwidth=2.0, adaptive=False, grid="20x14"))
    warm_obj = XtBandwidthObjective(df, seed=42)
    warm_obj.evaluate(_cand(bandwidth=0.5, adaptive=True, grid="20x14"))  # populate cache at this grid
    warm = warm_obj.evaluate(_cand(bandwidth=2.0, adaptive=False, grid="20x14"))  # reuse cache
    assert warm["xt_holdout_nll"] == pytest.approx(cold["xt_holdout_nll"], abs=1e-12)


def test_binning_cached_once_per_grid_fold_seam_runs_per_trial(monkeypatch):
    # Structural perf guard (spec C2): over N trials at ONE grid, the (expensive) binning runs once
    # per fold, while the (cheap) gaussian seam runs once per (trial, fold). Patch in the objective
    # module's namespace (it imports the helpers by name).
    import silly_kicks.calibration._xt_bandwidth_objective as mod
    from tests._perf_structural import call_counter

    bin_calls = call_counter(monkeypatch, mod, "_bin_destinations_by_source")
    seam_calls = call_counter(monkeypatch, mod, "_gaussian_transition_from_grouped")
    obj = XtBandwidthObjective(_sparse_overfit_corpus(seed=6, n_games=20), seed=42)
    n_folds = len(obj._folds)
    n_trials = 4
    for bw in (0.5, 1.0, 2.0, 4.0):
        obj.evaluate(_cand(bandwidth=bw, grid="16x12"))
    assert bin_calls["n"] == n_folds            # binning NOT re-run per trial (cache works)
    assert seam_calls["n"] == n_trials * n_folds  # seam re-runs per (trial, fold)


def test_recommendation_round_trips_into_expected_threat():
    # M3: the recommended config must construct a usable fitted xT (not just emit numbers).
    from silly_kicks.xthreat import ExpectedThreat, KDEParams
    from tests._xthreat_helpers import _corpus_with_shots

    grid = "20x14"
    nx, ny = (int(v) for v in grid.split("x"))
    xt = ExpectedThreat(l=nx, w=ny, method="kde_smoothed", params=KDEParams(bandwidth=1.5, adaptive=True))
    xt.fit(_corpus_with_shots(n_per_zone=40))
    assert np.any(xt.xT > 0)
    assert np.all(np.isfinite(xt.xT))
```

- [ ] **Step 2: Run to verify it fails (then passes after no code change — these guard existing behavior)**

Run: `python -m pytest tests/calibration/test_xt_bandwidth_objective.py -q -k "cache_equivalence or binning_cached or round_trips"`
Expected: PASS (the objective from Task 3 already satisfies these). If `binning_cached` FAILS, the cache key or the per-fold prepare is wrong — fix the objective until it passes (the guard is the spec's budget).

- [ ] **Step 3: Stage**

```bash
git add tests/calibration/test_xt_bandwidth_objective.py
```

---

## Task 5: CLI `scripts/calibrate_xt_bandwidth.py`

**Files:**
- Create: `scripts/calibrate_xt_bandwidth.py`
- Test: `tests/calibration/test_calibrate_xt_bandwidth_cli.py` (create)

### Background
- Mirror `scripts/calibrate_tracking_defaults.py`: a `run_*` seam (no I/O) + `build_manifest` + `main()` argparse. The loaders (`scripts._loader_pining`, `scripts._loader_databricks`) yield `(provider, mid, actions, frames, home)`; for xT we keep only `actions` and load pining with `tracking_limit=1` (NOT `0` — `_loader_pining.py:512` gates `if tracking_limit:`, so `0` loads all frames; spec N8).
- The cross-check (spec M7, review C1) is pure (takes `actions` + recommendation); reported in the manifest, not gated.

- [ ] **Step 1: Write the failing CLI tests**

Create `tests/calibration/test_calibrate_xt_bandwidth_cli.py`:

```python
"""SK-xT-3 Task 5: CLI seam smoke + manifest shape + reported cross-check."""

import math

import silly_kicks
from scripts.calibrate_xt_bandwidth import build_manifest, run_xt_bandwidth, xt_quality_cross_check
from tests._xthreat_helpers import _corpus_with_shots, _sparse_overfit_corpus


def _multi_game_shots(n_games=6):
    import pandas as pd

    parts = []
    for g in range(n_games):
        d = _corpus_with_shots(n_per_zone=20, seed=g)
        d["game_id"] = g
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def test_run_xt_bandwidth_smoke_returns_finite_best(tmp_path):
    result, objective = run_xt_bandwidth(
        actions=_sparse_overfit_corpus(seed=7, n_games=20),
        n_trials=3,
        seed=42,
        store_path=str(tmp_path / "xt.db"),
    )
    assert result.best is not None
    assert math.isfinite(result.best.metrics["xt_holdout_nll"])


def test_build_manifest_scopes_recommendation_and_versions(tmp_path):
    result, _obj = run_xt_bandwidth(
        actions=_sparse_overfit_corpus(seed=8, n_games=20),
        n_trials=3,
        seed=42,
        store_path=str(tmp_path / "xt.db"),
    )
    manifest = build_manifest(
        source="pining", seed=42, n_trials=3, max_points_per_zone=None,
        match_ids={"pining": ["m1"]}, result=result, cross_check=None,
    )
    assert manifest["stage"] == "xt_bandwidth"
    assert manifest["applies_to_library_default"] is False
    assert "unverified" in manifest["recommendation_scope"]
    assert manifest["silly_kicks_version"] == silly_kicks.__version__
    assert manifest["recommendation"]["method"] == "kde_smoothed"
    assert set(manifest["recommendation"]["grid"]) == {"n_zones_x", "n_zones_y"}


def test_cross_check_returns_finite_rho_for_both_grids():
    cc = xt_quality_cross_check(
        _multi_game_shots(), recommendation={"bandwidth": 1.5, "adaptive": True, "grid": {"n_zones_x": 16, "n_zones_y": 12}}, k=10, seed=42
    )
    assert math.isfinite(cc["rho_recommended"])
    assert math.isfinite(cc["rho_singh"])


def test_scores_per_game_does_not_leak_goal_across_game_boundary():
    # P1 regression: the LAST action of game A must NOT be labelled "scored" by game B's early goal.
    import pandas as pd

    import silly_kicks.spadl.config as cfg
    from scripts.calibrate_xt_bandwidth import _scores_per_game

    cols = ["game_id", "action_id", "period_id", "time_seconds", "team_id", "player_id",
            "bodypart_id", "type_id", "result_id", "start_x", "start_y", "end_x", "end_y"]
    pas, shot = cfg.actiontype_id["pass"], cfg.actiontype_id["shot"]
    succ, fail = cfg.result_id["success"], cfg.result_id["fail"]
    rows = [
        # game A: a single trailing pass (no goal in A) -> label MUST be 0
        [1, 0, 1, 0.0, 7, 7, 0, pas, succ, 50.0, 34.0, 60.0, 34.0],
        # game B: an immediate goal by the same team within k actions of A's last row
        [2, 1, 1, 0.0, 7, 7, 0, shot, succ, 100.0, 34.0, 105.0, 34.0],
    ]
    df = pd.DataFrame(rows, columns=cols)
    y = _scores_per_game(df, k=10)
    assert y[0] == 0  # game A's pass is NOT credited with game B's goal (would be 1 if leaked)


def test_load_corpus_pining_requests_minimal_tracking(monkeypatch):
    # N8: the pining corpus load must pass tracking_limit=1 (NOT 0 — 0 is falsy and loads all frames).
    import pandas as pd

    import scripts._loader_pining as loader
    from scripts.calibrate_xt_bandwidth import _load_corpus

    captured = {}
    cols = ["game_id", "start_x", "start_y", "end_x", "end_y", "type_id", "result_id"]

    def _fake_load_matches(*, providers, tracking_limit=None, max_per_provider=None):
        captured["tracking_limit"] = tracking_limit
        yield "skillcorner", "m1", pd.DataFrame([[1, 50.0, 34.0, 60.0, 34.0, 0, 1]], columns=cols), None, None

    monkeypatch.setattr(loader, "load_matches", _fake_load_matches)
    args = type("A", (), {"source": "pining", "providers": ["skillcorner"], "max_matches_per_provider": None})()
    actions, ids = _load_corpus(args)
    assert captured["tracking_limit"] == 1
    assert "skillcorner" in ids
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/calibration/test_calibrate_xt_bandwidth_cli.py -x -q`
Expected: FAIL — `ModuleNotFoundError: scripts.calibrate_xt_bandwidth`.

- [ ] **Step 3: Implement the CLI**

Create `scripts/calibrate_xt_bandwidth.py`:

```python
"""SK-xT-3 xT bandwidth/resolution calibration CLI — held-out transition-NLL sweep.

Pure objective lives in silly_kicks.calibration._xt_bandwidth_objective; this script owns I/O
(corpus loaders), the Optuna run, the manifest, and the reported downstream xT-quality cross-check.
Recommends a KDEParams+GridSpec; does NOT change any library default (ADR-009).

Usage:
    python scripts/calibrate_xt_bandwidth.py --source pining \
        --providers skillcorner idsse --n-trials 100 --store xt.db \
        --max-points-per-zone 5000 --report-out xt_report
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from importlib.metadata import version

import numpy as np
import pandas as pd
from ruthless import InProcessBackend, render_json, render_summary_md
from ruthless.strategies.optuna_ import OptunaStrategy

import silly_kicks
from silly_kicks.calibration._spaces import grid_from_str, xt_bandwidth_config
from silly_kicks.calibration._xt_bandwidth_objective import XtBandwidthObjective

_XT_COLS = ["game_id", "start_x", "start_y", "end_x", "end_y", "type_id", "result_id"]


def run_xt_bandwidth(*, actions, n_trials, seed, store_path, max_points_per_zone=None):
    """Run the Optuna study on an already-loaded corpus (the testable seam — no I/O)."""
    objective = XtBandwidthObjective(actions, seed=seed, max_points_per_zone=max_points_per_zone)
    config = xt_bandwidth_config(n_trials=n_trials, store_path=store_path)
    result = OptunaStrategy(config, seed=seed).run(objective, backend=InProcessBackend())
    return result, objective


def _finite_or_none(x):
    """Non-finite (nan/inf) -> None for strict-JSON consumers (review minor): xt_holdout_nll_se is
    nan on a 1-fold corpus, and bare NaN is invalid JSON for non-Python readers."""
    return float(x) if x is not None and np.isfinite(x) else None


def build_manifest(*, source, seed, n_trials, max_points_per_zone, match_ids, result, cross_check=None):
    """Auditable manifest (recommends; never applies — ADR-009)."""
    rec = None
    best = result.best
    if best is not None:
        p = best.candidate.params
        grid = grid_from_str(str(p["grid"]))
        rec = {
            "method": "kde_smoothed",
            "bandwidth": float(p["bandwidth"]),
            "adaptive": bool(p["adaptive"]),
            "kernel": "gaussian",
            "grid": {"n_zones_x": grid.n_zones_x, "n_zones_y": grid.n_zones_y},
            "xt_holdout_nll": _finite_or_none(best.metrics.get("xt_holdout_nll")),
            "xt_holdout_nll_se": _finite_or_none(best.metrics.get("xt_holdout_nll_se")),
            "singh_holdout_nll": _finite_or_none(best.metrics.get("singh_holdout_nll")),
        }
    return {
        "stage": "xt_bandwidth",
        "source": source,
        "seed": seed,
        "n_trials": n_trials,
        "max_points_per_zone": max_points_per_zone,
        "match_ids": match_ids,
        "recommendation": rec,
        "recommendation_scope": "optimal for held-out destination likelihood; xT-quality impact unverified",
        "applies_to_library_default": False,
        "bandwidth_dual_meaning": "adaptive=True -> Silverman multiplier; adaptive=False -> raw SPADL metres",
        "downstream_xt_quality_cross_check": cross_check,
        "silly_kicks_version": silly_kicks.__version__,
        "ruthless_version": version("ruthless-efficiency"),
        "xgboost_version": version("xgboost"),
        "generated_date": _dt.date.today().isoformat(),
    }


def _scores_per_game(actions, *, k):
    """vaep ``scores`` labels computed PER GAME, reassembled in row order (review P1).

    ``vaep.labels.scores`` uses a raw ``shift(-k)`` with NO ``game_id`` grouping (``labels.py``
    ~138-142), so on a multi-game frame it leaks goal-lookahead across game boundaries. Grouping by
    ``game_id`` first makes the label boundary-safe. ``rate()`` has no lookahead, so only the label
    needs this; the result stays row-aligned to ``actions``.
    """
    from silly_kicks.spadl import add_names
    from silly_kicks.vaep.labels import scores as scores_label

    return (
        actions.reset_index(drop=True)
        .groupby("game_id", group_keys=False, sort=False)
        .apply(lambda g: scores_label(add_names(g.reset_index(drop=True)), nr_actions=k)["scores"])
        .to_numpy()
    )


def xt_quality_cross_check(actions, recommendation, *, k=10, seed=42):
    """Reported (not gated) downstream xT-quality signal (spec M7 / review C1).

    Spearman rho between Delta-rate = rate(end) - rate(start) and "the in-possession team scored
    within K actions", on the held-out CV folds, for the recommended grid vs the Singh grid. A
    single number per grid; if NLL-best does NOT also win rho, that is a finding worth surfacing.
    """
    from scipy.stats import spearmanr

    from silly_kicks.calibration._cv import match_cv_splits
    from silly_kicks.xthreat import ExpectedThreat, KDEParams

    actions = actions.reset_index(drop=True)
    g = recommendation["grid"]
    nx, ny = int(g["n_zones_x"]), int(g["n_zones_y"])
    params = KDEParams(bandwidth=float(recommendation["bandwidth"]), adaptive=bool(recommendation["adaptive"]))
    game_ids = actions["game_id"].to_numpy()

    def _rho(method, kde_params):
        deltas, labels = [], []
        for tr, te in match_cv_splits(game_ids):
            train, test = actions.iloc[tr], actions.iloc[te]
            xt = (
                ExpectedThreat(l=nx, w=ny, method="kde_smoothed", params=kde_params).fit(train)
                if method == "kde"
                else ExpectedThreat(l=nx, w=ny, method="singh_counts").fit(train)
            )
            if not np.any(xt.xT):
                continue
            d = xt.rate(test)  # NaN on non-move rows; rate has no lookahead -> boundary-safe
            y = _scores_per_game(test, k=k)  # per-game label (no cross-game goal leak — P1)
            mask = np.isfinite(d)
            deltas.append(d[mask])
            labels.append(y[mask])
        if not deltas:
            return float("nan")
        dd, yy = np.concatenate(deltas), np.concatenate(labels)
        if len(dd) < 3 or len(np.unique(yy)) < 2:
            return float("nan")
        return float(spearmanr(dd, yy).statistic)

    return {"rho_recommended": _rho("kde", params), "rho_singh": _rho("singh", None), "k": k}


def _load_corpus(args) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Load SPADL actions only (minimal tracking footprint) into one corpus DataFrame + match ids."""
    if args.source == "pining":
        import scripts._loader_pining as loader

        gen = loader.load_matches(providers=args.providers, tracking_limit=1, max_per_provider=args.max_matches_per_provider)
    else:
        import scripts._loader_databricks as loader

        conn = loader._connect()
        try:
            cur = conn.cursor()
            cols = ", ".join(_XT_COLS)
            df = loader._query_param(cur, f"SELECT {cols} FROM soccer_analytics.bronze.spadl_actions")  # noqa: S608
        finally:
            conn.close()
        return df, {"databricks": ["bronze.spadl_actions"]}

    parts, ids = [], {}
    for provider, mid, actions, _frames, _home in gen:
        parts.append(actions)
        ids.setdefault(provider, []).append(str(mid))
    return pd.concat(parts, ignore_index=True), ids


def main() -> None:
    ap = argparse.ArgumentParser(description="SK-xT-3 xT bandwidth/resolution calibration")
    ap.add_argument("--source", choices=["pining", "databricks"], default="pining")
    ap.add_argument("--providers", nargs="+", default=["skillcorner", "idsse"])
    ap.add_argument("--n-trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--store", required=True)
    ap.add_argument("--max-points-per-zone", type=int, default=None)
    ap.add_argument("--max-matches-per-provider", type=int, default=None)
    ap.add_argument("--report-out", default="xt_bandwidth_report")
    ap.add_argument("--cross-check", action="store_true", help="run the reported downstream xT-quality cross-check")
    args = ap.parse_args()

    actions, match_ids = _load_corpus(args)
    result, _obj = run_xt_bandwidth(
        actions=actions, n_trials=args.n_trials, seed=args.seed,
        store_path=args.store, max_points_per_zone=args.max_points_per_zone,
    )
    cross_check = None
    if args.cross_check and result.best is not None:
        p = result.best.candidate.params
        grid = grid_from_str(str(p["grid"]))
        cross_check = xt_quality_cross_check(
            actions,
            {"bandwidth": float(p["bandwidth"]), "adaptive": bool(p["adaptive"]),
             "grid": {"n_zones_x": grid.n_zones_x, "n_zones_y": grid.n_zones_y}},
            seed=args.seed,
        )
    manifest = build_manifest(
        source=args.source, seed=args.seed, n_trials=args.n_trials,
        max_points_per_zone=args.max_points_per_zone, match_ids=match_ids,
        result=result, cross_check=cross_check,
    )
    report = {"ruthless": json.loads(render_json(result)), "calibration_manifest": manifest}
    with open(f"{args.report_out}.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    with open(f"{args.report_out}.md", "w", encoding="utf-8") as fh:
        fh.write(render_summary_md(result))
        fh.write("\n\n## Calibration manifest\n\n```json\n")
        fh.write(json.dumps(manifest, indent=2))
        fh.write("\n```\n")
    print(render_summary_md(result))
    print(f"Best: {result.best.metrics if result.best else None}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/calibration/test_calibrate_xt_bandwidth_cli.py -q`
Expected: PASS.

- [ ] **Step 5: Stage**

```bash
git add scripts/calibrate_xt_bandwidth.py tests/calibration/test_calibrate_xt_bandwidth_cli.py
```

---

## Task 6: Exports + ADR-009 amendment

**Files:**
- Modify: `silly_kicks/calibration/__init__.py`
- Modify: `docs/superpowers/adrs/ADR-009-calibration-harness.md`

- [ ] **Step 1: Write the failing export test** — append to `tests/calibration/test_spaces.py`:

```python
def test_xt_bandwidth_public_exports():
    from silly_kicks.calibration import XtBandwidthObjective, grid_from_str, xt_bandwidth_config

    assert callable(xt_bandwidth_config)
    assert callable(grid_from_str)
    assert XtBandwidthObjective.__name__ == "XtBandwidthObjective"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/calibration/test_spaces.py::test_xt_bandwidth_public_exports -q`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Add exports** to `silly_kicks/calibration/__init__.py` — add imports and `__all__` entries (alphabetical):

```python
from silly_kicks.calibration._spaces import grid_from_str, stage1_config, stage2_config, xt_bandwidth_config
from silly_kicks.calibration._xt_bandwidth_objective import XtBandwidthObjective
```

Add to `__all__`: `"XtBandwidthObjective"`, `"grid_from_str"`, `"xt_bandwidth_config"` (keep the list sorted).

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/calibration/test_spaces.py -q`
Expected: PASS.

- [ ] **Step 5: Amend ADR-009** — append a dated amendment section to `docs/superpowers/adrs/ADR-009-calibration-harness.md`:

```markdown
## Amendment (2026-06-08, SK-xT-3): xT bandwidth/resolution NLL objective

Extends the recommends-an-auditable-manifest-never-mutates-defaults harness to xT
(`silly_kicks/calibration/_xt_bandwidth_objective.py` + `xt_bandwidth_config`, ADR-021). It sweeps
`KDEParams.bandwidth` x `GridSpec` resolution x `adaptive` over the held-out transition-NLL
substrate (`compute_holdout_nll`) and recommends a `KDEParams`+`GridSpec`; the library default
`KDEParams.bandwidth=1.0` is untouched.

**Deliberate divergence from this ADR's CachedObjective pattern.** Stage 1/2 ship as a
`ruthless.CachedObjective` guarded by `assert_cache_equivalence`. SK-xT-3 instead uses a plain
duck-typed object with a resolution-keyed lazy cache and a hand-written warm==cold equivalence test,
because the resolution axis means the invariant is keyed by `(grid, fold)` — which a single
`prepare()` does not model. This is by design, not oversight.

**Gaussian-core re-pin (Chesterton's Fence).** Task 1 re-pins the gaussian KDE numerics to a
vectorized implementation (shared `_gaussian_transition_from_grouped` seam) for a ~1000x per-trial
speedup. `grep` confirms `kde_smoothed_transition_matrix` has exactly one caller (`_model.py:123`)
and `ExpectedThreat` defaults to `singh_counts` (`_model.py:78`), so no shipped artifact (VAEP,
bundled weights, lakehouse) depends on the KDE numerics — the re-pin is safe.
```

- [ ] **Step 6: Stage**

```bash
git add silly_kicks/calibration/__init__.py docs/superpowers/adrs/ADR-009-calibration-harness.md tests/calibration/test_spaces.py
```

---

## Task 7: Housekeeping — version bump, CHANGELOG, TODO

**Files:**
- Modify: `pyproject.toml:7`, `silly_kicks/__init__.py` (`__version__`), `CHANGELOG.md`, `TODO.md`

- [ ] **Step 1: Bump version to 4.20.0**

In `pyproject.toml` line 7: `version = "4.20.0"`. In `silly_kicks/__init__.py`, set `__version__ = "4.20.0"` (match exactly — the version-bump hard gate requires pyproject + __init__ + CHANGELOG aligned). NOTE: main was at **4.19.2** after the CI slow-test-gating merge (ADR-023, #115), so 4.20.0 is the next minor.

- [ ] **Step 2: CHANGELOG entry** — insert below the header, above `## [4.19.2]`:

```markdown
## [4.20.0] — 2026-06-08

### Added — SK-xT-3 calibration-integrated xT bandwidth/resolution sweep (ADR-009, ADR-021)

`silly_kicks.calibration.xt_bandwidth_config` + `XtBandwidthObjective` — a `ruthless`/Optuna sweep
over xT `KDEParams.bandwidth` × `GridSpec` resolution × `adaptive` minimizing K-fold held-out
transition-NLL, with the Singh no-smoothing baseline reported alongside. Recommends a
`KDEParams`+`GridSpec` via an auditable manifest (`scripts/calibrate_xt_bandwidth.py`); **changes no
library default** (ADR-009). The recommendation is scoped to held-out *destination likelihood*
(xT-quality impact reported, not asserted) and a downstream Spearman cross-check vs realised goals
is emitted.

### Changed — vectorized gaussian xT KDE core (internal; no public-API change)

`kde_smoothed_transition_matrix` now factors a shared, vectorized gaussian seam
(`_gaussian_transition_from_grouped`) — softmax-stabilized, ~1000× faster per call, sklearn retained
only for non-gaussian kernels. The gaussian numerics are re-pinned (Chesterton-verified: one caller,
`singh_counts` default) and now stay finite/correct in the small-bandwidth regime where the previous
sklearn-wrapper underflowed to the mean-row fallback.
```

- [ ] **Step 3: Groom TODO.md** — delete the On-Deck row "`calibration`-integrated xT bandwidth/HPO sweep (TF-24 pattern)" (the whole bullet under "SK-xT-1 follow-ups"); update the "**Last updated**"/"**Current release**" line to 4.20.0. Do NOT strikethrough — delete (the CHANGELOG is the record).

- [ ] **Step 4: Verify the version gate aligns**

Run: `python -c "import silly_kicks; print(silly_kicks.__version__)"` → `4.20.0`
Run: `grep -n "4.20.0" pyproject.toml CHANGELOG.md`
Expected: both files show 4.20.0.

- [ ] **Step 5: Stage**

```bash
git add pyproject.toml silly_kicks/__init__.py CHANGELOG.md TODO.md
```

---

## Task 8: Full verification + `/final-review` + single commit

**Files:** none (verification + commit)

- [ ] **Step 1: Shift-Left gate — formatting + lint**

Run: `ruff format --check . ; ruff check .`
Expected: both clean. (If `ruff format --check` reports diffs, run `ruff format .`, re-stage, re-check.)

- [ ] **Step 2: Type check (full package scope)**

Run: `pyright silly_kicks/`
Expected: 0 errors. (Common fixes: annotate `grouped: dict[int, npt.NDArray[np.float64]]`; ensure `match_cv_splits` receives `npt.NDArray`.)

- [ ] **Step 3: Full non-e2e suite**

Run: `python -m pytest tests/ -m "not e2e" -q`
Expected: all pass. Pay attention to `tests/test_xthreat_kde*.py`, `tests/calibration/`, and any Examples-doctest gate (every new public def has an Examples block).

- [ ] **Step 4: Run `/final-review`** (mandatory pre-commit gate per repo policy). Address any findings, re-stage, re-run Steps 1–3 if code changed.

- [ ] **Step 5: Request explicit owner approval to commit** (repo policy: explicit approval before the single commit). Do not proceed without it.

- [ ] **Step 6: Single commit**

```bash
git add -A
git commit -m "feat(calibration): SK-xT-3 xT bandwidth/resolution NLL sweep + vectorized gaussian KDE core -- silly-kicks 4.20.0 (ADR-009, ADR-021)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Fact-check (post-merge, owner-run on Databricks/DGX — NOT part of the committed suite)

Run the harness as a job (not inline) against `soccer_analytics.bronze.spadl_actions` with
`--max-points-per-zone 5000 --cross-check` to (1) confirm the NLL-optimal bandwidth is materially
larger on the 8.9M-action mart than on a 64-match sample and KDE beats Singh held-out, and (2)
record the Spearman cross-check (recommended vs Singh). This validates the motivating premise and
the manifest scope; it stays out of CI (the committed owner-gated mart NLL e2e is a separate TODO row).

**Cost note (review minor):** `--cross-check` refits a full `ExpectedThreat` (incl. value iteration)
`5 folds × 2 methods = 10×` on the corpus, on top of the sweep. Trivial in tests, non-trivial on the
8.9M mart — budget for it in the job (it is owner-run, off CI). The sweep itself is cheap per trial
(the vectorized seam off the cached `grouped`); the cross-check is the heavier one-shot tail.

---

## Self-review checklist (run before handing off to execution)

- [ ] Spec coverage: §0 refactor (Task 1), search space (Task 2), objective + cache + CV + Singh + N1/N2/N3 (Tasks 3–4), CLI + manifest + cross-check (Task 5), exports + ADR M4/re-pin (Task 6), housekeeping (Task 7), Shift-Left + final-review + one commit (Task 8). Fact-check documented.
- [ ] Type consistency: `_bin_destinations_by_source` → `(grouped, centres)`; `_gaussian_transition_from_grouped(grouped, centres, grid, params)`; `grid_from_str` everywhere; objective metric key `xt_holdout_nll` matches `xt_bandwidth_config.metric`; `run_xt_bandwidth`/`build_manifest`/`xt_quality_cross_check` signatures match their tests.
- [ ] No per-task commits (policy); single commit in Task 8 gated on approval.
```

---

## Validation addendum (2026-06-08, DGX full run — additions beyond the original plan)

Full-size validation ran 100% on the DGX (pining source #1; no Databricks; no local pulls). It
surfaced work that was added during execution and is part of the shipped PR:

**Download/parse caching (owner-requested mid-flight).** `_loader_pining.load_matches` gained an
opt-in `cache_dir` (persistent, atomic-write artifact cache + cache-hit skip); the CLI gained
`--cache-dir`, `--corpus-cache` (assembled-corpus parquet — skips download+parse on re-runs), and
`--subsample-games` (corpus-size contrast off the cache). `_canonicalize_corpus` projects to the
standard SPADL columns + string-casts the provider-asymmetric id columns so the multi-provider
parquet serialises; a fast-fail guard requires a parquet engine (pyarrow) up front. 4 new tests.

**4 real multi-provider bugs the synthetic (homogeneous) fixtures missed — each fixed + regression-
tested:** (1) `_scores_per_game` `groupby.apply` collapse; (2) `build_manifest` spurious
`version("xgboost")`; (3) mixed-dtype `game_id` CV `np.unique` sort crash; (4) `original_event_id`
heterogeneous-object-column parquet `ArrowTypeError`.

**Results (full 81-match pining corpus + 10/20/40 contrasts; all adaptive=False, grid 12×8):**
optimal bandwidth is corpus-size-dependent — 7.82 (10 games) → 6.74 (20) → 5.70 (40) → **5.07 (81)**;
KDE beats Singh at every size (full: KDE NLL 3.41 vs Singh 3.99). The reported xT-quality cross-check
shows Singh correlating better with goals at **every** size (ρ_singh ≈ 0.05 > ρ_kde ≈ 0.02) — a robust,
honest caveat that NLL-best ≠ xT-optimal (the manifest scope is correct). The harness does not change
any library default.
