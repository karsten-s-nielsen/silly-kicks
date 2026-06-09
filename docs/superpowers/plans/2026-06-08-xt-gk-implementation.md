# xT-GK (Eyestone) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Jeffrey Eyestone's **xT-GK** (Expected Threat for Goalkeepers) as a public, fully-attributed, frame-aware tracking feature in silly-kicks — a pure *parametric compute* feature (no trained weights) that re-values GK distribution actions by composing the xT grid with GK-specific terms (pressure-escape, risk-adjusted, defensive-zone) under a frozen, team-tunable parameter set.

**Architecture:** Mirrors the GK-suite sibling `gk_influence` (the closest precedent — it already takes an injected `xt: ExpectedThreat`). New production module `silly_kicks/tracking/_xt_gk.py` with pure helpers + a batch `compute_xt_gk`; a `@nan_safe_enrichment`-decorated `add_xt_gk` aggregator in `tracking/features.py`; a VAEP `xt_gk_xfns` factory carrying the injected `xt`; an atomic mirror with endpoint synthesis. Consumes shipped surfaces only: `ExpectedThreat` (baseline grid, ADR-021), `pressure_on_actor` (ρ), `get_xc` (RAV's P(success), `[das]` extra), `scipy.ndimage` (grid convolution).

**Tech Stack:** Python, pandas, numpy, scipy (`ndimage.gaussian_filter`), accessible-space (`[das]`, via `get_xc`). No xgboost, no new runtime dep, no calibration-harness wiring.

**Source spec:** `docs/superpowers/specs/2026-06-07-xt-gk-design.md` (final, Jeffrey-reviewed; formula gate Q1 cleared by his spec-read, preset values delegated by his prior "ship presets, whatever is easy").

---

## ⚠️ Repo-convention deviations from the writing-plans skill (read first)

This repo overrides two skill defaults (instruction priority: user conventions > skill):

1. **NO per-task commits.** This feature ships as **one feature branch → one commit → one PR at the end** ([no-standalone-doc-commits], [no per-PR doc commits]). Each task below ends with a **test-green checkpoint**, *not* a commit. The spec, ADR, NOTICE, CLAUDE.md line, and C4 regen all bundle into the **single final commit** (Task P1-13).
2. **Commits are sentinel-gated.** The `git_commit_guard.py` hook blocks any `git commit` without an explicit per-commit approval token. The final commit task documents this — **do not attempt to create the sentinel yourself**; present the command + diff and hold for the user's explicit "yes" ([never-create-sentinel-without-approval]).

Run work on a dedicated branch/worktree off `main`. Verify the full suite with:
```bash
.venv\Scripts\python.exe -m pytest tests/ -m "not e2e" -q --tb=short
```
(Use the uv-managed `.venv` CPython 3.10.19 — never `pip install` one-offs into it.)

---

## File structure

| File | Create/Modify | Responsibility |
|---|---|---|
| `silly_kicks/tracking/_xt_gk.py` | **Create** | `XtGkParams` + presets; pure helpers (`_convolve_grid`, `_grid_value`, `_counter_value`, `_normalize_pressure`, `_possession_depth`); component fns (`_base/_pev/_rav/_dzv/_temporal/_composite`); domain filter `_gk_distribution_mask`; batch `compute_xt_gk` |
| `silly_kicks/tracking/features.py` | Modify | `add_xt_gk` aggregator (`@nan_safe_enrichment`) + `xt_gk_xfns` factory |
| `silly_kicks/tracking/__init__.py` | Modify | export `compute_xt_gk`, `add_xt_gk`, `xt_gk_xfns`, `XtGkParams` (+ column-name string symbols) in imports + `__all__` |
| `silly_kicks/atomic/tracking/features.py` | Modify | mirror `add_xt_gk` + `xt_gk_xfns` with `x,y,dx,dy → start/end` endpoint synthesis |
| `tests/tracking/test_xt_gk.py` | **Create** | oracle tests, construct-invariant gates, leakage contract, domain filter, atomic parity |
| `tests/tracking/conftest_id_dtype.py` | Modify | register `add_xt_gk` in `AGGREGATORS` (id-dtype gate, ADR-019) |
| `tests/tracking/test_frame_aware_xfns_dup_action_id.py` | Modify | bump meta floor `>= 20` → `>= 21` (auto-discovers `xt_gk_xfns`) |
| `tests/test_enrichment_nan_safety.py` | Modify | add `add_xt_gk` to `_TRACKING_NEEDS_EXTRA` + an extra-kwargs construction branch |
| `NOTICE` | Modify | Eyestone xT-GK bibliographic entry + consent-trail note |
| `docs/superpowers/adrs/ADR-NNN-xt-gk.md` | **Create** | decision record (formula interpretation, tracking-required, no-calibration phasing, consent provenance) |
| `docs/c4/architecture.dsl` + `architecture.html` | Modify + regen | aggregator count 25 → 26 |
| `CLAUDE.md` | Modify | one `PR-S## ships … xT-GK …` architecture line |

---

## The metric (reference — implemented across Tasks P1-2/3)

Notation: action *a* moves the ball from origin `z=(start_x,start_y)` to destination `z'=(end_x,end_y)`. `xT★` = spatially-convolved baseline grid; `xT` = raw grid; `p` = xC; ρ ∈ [0,1] = normalized pressure.

```
progress(a) = xT★(z') − xT★(z)                              # forward move value; feeds PEV only
base(a)     = − xT★(z)                                       # Option B (Jeffrey 2026-06-08): origin-only
PEV(a)      = ρ · max(0, progress(a))
RAV(a)      = p · xT★(z') − δ · (1 − p) · xT★_counter(z')    # xT★_counter = 180° point-reflection; SOLE owner of z'
DZV(a)      = 1[start_x ≤ defensive_third_boundary] · (V_def − xT(z))   # raw xT(z), not xT★
T(a)        = η ** k(a)                                      # k = possession depth (≈0 for GK distributions)
xT-GK(a)    = T(a) · ( base(a) + γ·PEV(a) + RAV(a) ) + φ · DZV(a)   # T scales threat terms only, NOT DZV
```

**Option B resolution (Jeffrey 2026-06-08, closes spec §11 Q1):** the destination value `xT★(z')`
enters the composite's main value path **once** — via RAV, completion-weighted — not also via a
full-weight base term. `base` is **origin-only** (`−xT★(z)`). PEV keeps a separate `progress`
(`xT★(z')−xT★(z)`) because PEV is a forward-progress signal; collapsing PEV onto the origin-only
base would zero it (`max(0, −xT★(z)) ≡ 0`). Presets ship provisional (Jeffrey approved). Recorded
in the ADR consent trail.

Emitted columns (per in-scope GK-distribution action; NaN otherwise): `xt_gk_base` (base = −xT★(z)), `xt_gk_pev` (ρ·max(0,progress)), `xt_gk_rav` (RAV), `xt_gk_dzv` (DZV), `xt_gk_pressure` (ρ), `xt_gk` (composite). Components are raw (un-weighted); the composite applies γ/φ/T — so they recombine to `xt_gk`.

**Verified codebase facts the code depends on** (from source, 2026-06-08):
- `ExpectedThreat(l=16, w=12, ...)`; fitted grid is attribute **`.xT`**, shape `(w, l) = (12, 16)`, **y-major with inverted row**: cell for `(x,y)` is `xT[(w-1) - yj, xi]` where `xi = clip(int(x/105*l), 0, l-1)`, `yj = clip(int(y/68*w), 0, w-1)`. Fit-only (no `from_grid`). `spadlconfig.field_length = 105`, `field_width = 68`.
- `get_xc(passes, frames, *, use_progress_bar=False, **kwargs) -> DataFrame` → input copy + `"xC"` column (NaN on degenerate). Lazy-imports accessible-space via `silly_kicks.tracking._das._import_accessible_space()` which raises `ImportError("accessible-space is required ... pip install 'silly-kicks[das]'")`.
- `pressure_on_actor(actions, frames, *, method="andrienko_oval", params=None, links=None) -> pd.Series` named `pressure_on_actor__<method>`; never negative; NaN where unlinked. Methods: `andrienko_oval` (unbounded ≥0), `link_zones`/`bekkers_pi` (∈[0,1]).
- `link_actions_to_frames(actions, frames, ...) -> (pointers, LinkReport)`; pointers columns: `action_id, frame_id, time_offset_seconds, n_candidate_frames, link_quality_score`.
- `_kernels.resolve_frame_ids_by_position(actions, frames, *, links=None) -> np.ndarray[float64]` (positional, dup-action_id-safe).
- `_id_compat`: `ids_equal/ids_differ/ids_match/same_id/align_join_keys` + `canonical_id/canonical_id_series`.
- SPADL `spadlconfig.actiontype_id`: `pass=0`, `throw_in=2`, `goalkick=22` (canonical name is `"goalkick"`, **not** `"goal_kick"`).

---

# PHASE 1 — xT-GK core feature *(first release)*

## Task P1-1: `XtGkParams` dataclass + `for_philosophy` presets

**Files:**
- Create: `silly_kicks/tracking/_xt_gk.py`
- Test: `tests/tracking/test_xt_gk.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/tracking/test_xt_gk.py
import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._xt_gk import XtGkParams

_PHILOSOPHIES = ["possession", "counter", "direct", "high_press", "low_block"]


class TestXtGkParams:
    def test_default_is_frozen_and_in_range(self):
        p = XtGkParams()
        with pytest.raises(Exception):
            p.gamma = 0.5  # frozen
        assert 0.1 <= p.gamma <= 0.4
        assert 0.3 <= p.delta <= 0.8
        assert 0.8 <= p.eta <= 0.9
        assert p.phi > 0.0
        assert p.v_def > 0.0
        assert p.defensive_third_boundary > 0.0
        assert p.pressure_scale > 0.0
        assert p.convolution_sigma >= 0.0
        assert p.pressure_method == "andrienko_oval"

    @pytest.mark.parametrize("name", _PHILOSOPHIES)
    def test_for_philosophy_in_range(self, name):
        p = XtGkParams.for_philosophy(name)
        assert 0.1 <= p.gamma <= 0.4
        assert 0.3 <= p.delta <= 0.8
        assert 0.8 <= p.eta <= 0.9

    def test_for_philosophy_are_distinct(self):
        sigs = {
            (p.gamma, p.delta, p.phi, p.eta)
            for p in (XtGkParams.for_philosophy(n) for n in _PHILOSOPHIES)
        }
        assert len(sigs) == len(_PHILOSOPHIES)  # all five distinct

    def test_for_philosophy_rejects_unknown(self):
        with pytest.raises(ValueError, match="unknown"):
            XtGkParams.for_philosophy("tiki_taka")
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/tracking/test_xt_gk.py::TestXtGkParams -q`
Expected: FAIL — `ModuleNotFoundError: silly_kicks.tracking._xt_gk`.

- [ ] **Step 3: Write minimal implementation**

```python
# silly_kicks/tracking/_xt_gk.py
"""xT-GK — Expected Threat for Goalkeepers (Eyestone).

A pure parametric compute feature (NOT a trained model): re-values GK distribution
actions (goal-kicks, keeper passes/throws) by composing the xT grid with GK-specific
terms under a frozen, team-tunable parameter set.

Attribution: Jeffrey Eyestone, *Expected Threat for Goalkeepers (xT-GK)*, winner of
Pitch to the Pros 1 (May 2025). Contributed publicly with attribution by Jeffrey's
explicit permission (email 2026-06-06). The functional forms here are the silly-kicks
formulation of Eyestone's xT-GK (the deck gives components + parameter ranges, not
closed-form equations). See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

_PressureMethod = Literal["andrienko_oval", "link_zones", "bekkers_pi"]

# Deck parameter ranges: gamma 0.1-0.4, delta 0.3-0.8, eta 0.8-0.9.
# Point values are PROVISIONAL (in-range), per Jeffrey's "ship presets, whatever is
# easy" delegation; exact table is open Q2 in the spec. v_def / defensive_third_boundary
# / pressure_scale are normative intent-set constants (never calibrated).


@dataclass(frozen=True)
class XtGkParams:
    # --- interpretive / intent-set (NOT VAEP-calibrated) ---
    gamma: float = 0.25          # PEV pressure-escape sensitivity   (range 0.1-0.4)
    delta: float = 0.55          # RAV risk-aversion                 (range 0.3-0.8)
    phi: float = 1.0             # DZV defensive-zone weight
    eta: float = 0.85            # temporal-sequence discount        (range 0.8-0.9)
    v_def: float = 0.02          # NORMATIVE: back-pass-penalty-fix baseline value
    defensive_third_boundary: float = 35.0   # NORMATIVE: x (m) where own defensive third ends (105/3)
    pressure_scale: float = 50.0             # ρ squash scale (§4.4); intent-set
    # --- structural smoothing (hand-set via one-off Phase-1 scan, Task P1-12) ---
    convolution_sigma: float = 0.8
    # --- method selector ---
    pressure_method: _PressureMethod = "andrienko_oval"

    @classmethod
    def for_philosophy(cls, name: str) -> "XtGkParams":
        """Return the deck's five team-philosophy presets (provisional point values
        within the deck ranges; exact values are open Q2)."""
        base = cls()
        presets: dict[str, dict[str, float]] = {
            "possession": dict(gamma=0.30, delta=0.45, phi=1.2, eta=0.88),
            "counter": dict(gamma=0.15, delta=0.70, phi=0.8, eta=0.82),
            "direct": dict(gamma=0.20, delta=0.60, phi=0.9, eta=0.80),
            "high_press": dict(gamma=0.35, delta=0.50, phi=1.1, eta=0.86),
            "low_block": dict(gamma=0.12, delta=0.75, phi=1.3, eta=0.90),
        }
        if name not in presets:
            raise ValueError(f"unknown xT-GK philosophy preset: {name!r}")
        return replace(base, **presets[name])
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/tracking/test_xt_gk.py::TestXtGkParams -q`
Expected: PASS.

- [ ] **Step 5: Checkpoint** — params + presets green. (No commit — see deviations note.)

---

## Task P1-2: Pure helpers (convolution, grid lookup, counter, pressure squash, possession depth)

**Files:**
- Modify: `silly_kicks/tracking/_xt_gk.py`
- Test: `tests/tracking/test_xt_gk.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/tracking/test_xt_gk.py  (append)
from silly_kicks.tracking._xt_gk import (
    _convolve_grid,
    _grid_value,
    _counter_value,
    _normalize_pressure,
    _possession_depth,
)
from silly_kicks.spadl import config as spadlconfig


class TestPureHelpers:
    def _ramp_grid(self):
        # shape (w, l) = (12, 16); value increases along x (columns) like xT toward goal
        return np.tile(np.linspace(0.0, 1.0, 16), (12, 1))

    def test_convolve_sigma_zero_is_identity(self):
        g = self._ramp_grid()
        out = _convolve_grid(g, 0.0)
        np.testing.assert_array_equal(out, g)

    def test_convolve_preserves_shape_and_smooths(self):
        g = self._ramp_grid()
        out = _convolve_grid(g, 1.0)
        assert out.shape == g.shape
        assert not np.array_equal(out, g)

    def test_grid_value_matches_inverted_row_convention(self):
        g = self._ramp_grid()
        # near goal (x≈105) -> high column index -> high value
        v_far = _grid_value(g, np.array([100.0]), np.array([34.0]))[0]
        v_near = _grid_value(g, np.array([5.0]), np.array([34.0]))[0]
        assert v_far > v_near
        # exact cell: x=5 -> xi=int(5/105*16)=0 ; y=34 -> yj=int(34/68*12)=6 ; row=(12-1)-6=5
        assert v_near == pytest.approx(g[5, 0])

    def test_counter_value_is_point_reflection(self):
        g = self._ramp_grid()
        x, y = np.array([10.0]), np.array([20.0])
        L, W = spadlconfig.field_length, spadlconfig.field_width
        expected = _grid_value(g, np.array([L - 10.0]), np.array([W - 20.0]))[0]
        assert _counter_value(g, x, y)[0] == pytest.approx(expected)

    def test_normalize_pressure_exp_cdf(self):
        # ρ = 1 - exp(-max(0,raw)/s); monotone, in [0,1), clamps negatives to 0
        assert _normalize_pressure(np.array([0.0]), 50.0)[0] == pytest.approx(0.0)
        assert _normalize_pressure(np.array([-5.0]), 50.0)[0] == pytest.approx(0.0)
        mid = _normalize_pressure(np.array([50.0]), 50.0)[0]
        assert mid == pytest.approx(1 - np.exp(-1.0))
        assert 0.0 <= mid < 1.0
        assert _normalize_pressure(np.array([1e6]), 50.0)[0] < 1.0

    def test_possession_depth_counts_within_team_run(self):
        actions = pd.DataFrame(
            {
                "team_id": [1, 1, 1, 2, 2, 1],
                "period_id": [1, 1, 1, 1, 1, 1],
            }
        )
        k = _possession_depth(actions)
        np.testing.assert_array_equal(k, [0, 1, 2, 0, 1, 0])

    def test_possession_depth_resets_on_period(self):
        actions = pd.DataFrame({"team_id": [1, 1, 1], "period_id": [1, 1, 2]})
        np.testing.assert_array_equal(_possession_depth(actions), [0, 1, 0])

    def test_grid_value_pinned_to_expected_threat_rate(self):
        """H1 anti-circularity: _grid_value's convention must equal ExpectedThreat.rate's,
        not merely match _xt_gk's own arithmetic. A successful pass's rate() value is
        xT(z') - xT(z); with sigma=0, _base must reproduce it exactly."""
        from silly_kicks.xthreat import ExpectedThreat
        from silly_kicks.tracking._xt_gk import _progress
        from silly_kicks.spadl import config as spadlconfig

        xt = ExpectedThreat(l=16, w=12)
        xt.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))
        a = pd.DataFrame(
            {
                "type_id": [spadlconfig.actiontype_id["pass"]],
                "result_id": [spadlconfig.result_id["success"]],
                "start_x": [20.0],
                "start_y": [30.0],
                "end_x": [80.0],
                "end_y": [40.0],
            }
        )
        # rate() returns the move value xT(z') - xT(z) == _progress; pin the two conventions.
        rate_val = xt.rate(a)[0]
        prog = _progress(
            _grid_value(xt.xT, a["end_x"].to_numpy(), a["end_y"].to_numpy()),
            _grid_value(xt.xT, a["start_x"].to_numpy(), a["start_y"].to_numpy()),
        )[0]
        assert prog == pytest.approx(rate_val)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/tracking/test_xt_gk.py::TestPureHelpers -q`
Expected: FAIL — helpers not importable.

- [ ] **Step 3: Write minimal implementation**

```python
# silly_kicks/tracking/_xt_gk.py  (append; add imports at top)
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.ndimage import gaussian_filter

from silly_kicks.spadl import config as spadlconfig
from ._id_compat import ids_equal


def _convolve_grid(xt_grid: npt.NDArray[np.float64], sigma: float) -> npt.NDArray[np.float64]:
    """Separable Gaussian smoothing of the xT grid (the public-app spatial-convolution
    term). sigma <= 0 returns the raw grid unchanged (xT★ ≡ xT)."""
    if sigma <= 0:
        return xt_grid
    return gaussian_filter(xt_grid, sigma=sigma, mode="nearest")


def _grid_value(
    grid: npt.NDArray[np.float64], x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Vectorized grid lookup at SPADL coords (LTR-normalized; team attacks +x).

    DRY (review H1): reuse xthreat's frozen cell-indexer (ADR-021) instead of
    reimplementing the (x,y)->cell math — this is xthreat's port, not xT-GK's to own.
    Apply the same row inversion ExpectedThreat.rate uses: row = (w-1) - yj, col = xi
    (row 0 is the top of the pitch). Pinned to .rate by a cross-check test (P1-2)."""
    from silly_kicks.xthreat._grid import _get_cell_indexes

    w, l = grid.shape
    xi, yj = _get_cell_indexes(pd.Series(np.asarray(x, float)), pd.Series(np.asarray(y, float)), l, w)
    return grid[(w - 1) - yj.to_numpy(), xi.to_numpy()]


def _counter_value(
    grid: npt.NDArray[np.float64], x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Opponent threat from the (intended) loss zone: full 180° point-reflection
    xT★(L - x, W - y) (both axes), per spec §4.2."""
    return _grid_value(
        grid,
        spadlconfig.field_length - np.asarray(x, dtype=float),
        spadlconfig.field_width - np.asarray(y, dtype=float),
    )


def _normalize_pressure(raw: npt.NDArray[np.float64], scale: float) -> npt.NDArray[np.float64]:
    """Saturating exponential-CDF squash ρ = 1 - exp(-max(0, raw)/s) -> [0, 1).
    max(0, ·) guards any method returning a negative raw (none currently do)."""
    clamped = np.maximum(0.0, np.asarray(raw, dtype=float))
    return 1.0 - np.exp(-clamped / scale)


def _possession_depth(actions: pd.DataFrame) -> npt.NDArray[np.intp]:
    """k(a): action's positional depth within its possession run. A run breaks when
    team_id changes or period_id changes. GK distributions are possession-starters so
    k ≈ 0 (the temporal term is near-inert here by construction; spec §4.5)."""
    team = actions["team_id"]
    period = actions["period_id"]
    # same-column self-shift; ids_equal is dtype/NaN-safe (first row -> not-equal -> new run)
    team_changed = ~ids_equal(team, team.shift())
    period_changed = period.ne(period.shift())
    run_id = (team_changed | period_changed.to_numpy()).cumsum()
    return actions.groupby(run_id, sort=False).cumcount().to_numpy()
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/tracking/test_xt_gk.py::TestPureHelpers -q`
Expected: PASS.

- [ ] **Step 5: Checkpoint** — pure helpers green.

---

## Task P1-3: Component functions (B / PEV / RAV / DZV / T / composite)

**Files:**
- Modify: `silly_kicks/tracking/_xt_gk.py`
- Test: `tests/tracking/test_xt_gk.py`

- [ ] **Step 1: Write the failing tests** (closed-form oracles; `np.isclose` ~1e-9)

```python
# tests/tracking/test_xt_gk.py  (append)
from silly_kicks.tracking._xt_gk import (
    _base,
    _progress,
    _pev,
    _rav,
    _dzv,
    _temporal,
    _composite,
)


class TestComponents:
    def test_progress_is_destination_minus_origin(self):
        assert _progress(np.array([0.5]), np.array([0.1]))[0] == pytest.approx(0.4)

    def test_base_is_negative_origin(self):
        # Option B: base owns only the origin cost; RAV owns the destination value.
        assert _base(np.array([0.1]))[0] == pytest.approx(-0.1)

    def test_pev_rewards_forward_escape_under_pressure(self):
        progress = np.array([0.4, 0.4, -0.4])
        rho = np.array([0.0, 0.8, 0.8])
        pev = _pev(rho, progress)
        assert pev[0] == pytest.approx(0.0)          # no pressure -> no reward
        assert pev[1] == pytest.approx(0.8 * 0.4)    # pressure + forward -> reward
        assert pev[2] == pytest.approx(0.0)          # negative progress clamped

    def test_rav_completion_weighted_minus_risk(self):
        rav = _rav(
            p=np.array([0.7]),
            xt_star_dest=np.array([0.5]),
            xt_star_counter=np.array([0.3]),
            delta=0.5,
        )
        assert rav[0] == pytest.approx(0.7 * 0.5 - 0.5 * (1 - 0.7) * 0.3)

    def test_dzv_only_in_defensive_third_and_raises_value(self):
        dzv = _dzv(
            start_x=np.array([10.0, 60.0]),
            xt_raw_origin=np.array([0.001, 0.001]),
            v_def=0.02,
            boundary=35.0,
        )
        assert dzv[0] == pytest.approx(0.02 - 0.001)  # in def third -> positive correction
        assert dzv[1] == pytest.approx(0.0)           # outside def third -> 0

    def test_temporal_is_eta_to_the_k(self):
        np.testing.assert_allclose(_temporal(np.array([0, 1, 2]), 0.85), [1.0, 0.85, 0.85**2])

    def test_composite_discounts_threat_terms_only_not_dzv(self):
        # xT-GK = T·(B + γ·PEV + RAV) + φ·DZV
        out = _composite(
            t=np.array([0.5]),
            base=np.array([-0.1]),
            pev=np.array([0.2]),
            rav=np.array([0.1]),
            dzv=np.array([0.03]),
            gamma=0.25,
            phi=1.0,
        )
        expected = 0.5 * (-0.1 + 0.25 * 0.2 + 0.1) + 1.0 * 0.03
        assert out[0] == pytest.approx(expected)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/tracking/test_xt_gk.py::TestComponents -q`
Expected: FAIL — component fns not defined.

- [ ] **Step 3: Write minimal implementation**

```python
# silly_kicks/tracking/_xt_gk.py  (append)
def _progress(xt_star_dest, xt_star_origin):
    """Forward move value xT★(z') − xT★(z). Feeds PEV ONLY — Option B keeps the destination
    out of the composite's base term (RAV owns z')."""
    return np.asarray(xt_star_dest, float) - np.asarray(xt_star_origin, float)


def _base(xt_star_origin):
    """Composite base term = − xT★(z) (Option B, Jeffrey 2026-06-08): the threat given up by
    leaving the origin. The destination value is owned solely by RAV (no double-count)."""
    return -np.asarray(xt_star_origin, float)


def _pev(rho, progress):
    return np.asarray(rho, float) * np.maximum(0.0, np.asarray(progress, float))


def _rav(p, xt_star_dest, xt_star_counter, delta):
    p = np.asarray(p, float)
    return p * np.asarray(xt_star_dest, float) - delta * (1.0 - p) * np.asarray(xt_star_counter, float)


def _dzv(start_x, xt_raw_origin, v_def, boundary):
    in_def_third = np.asarray(start_x, float) <= boundary
    return np.where(in_def_third, v_def - np.asarray(xt_raw_origin, float), 0.0)


def _temporal(k, eta):
    return np.power(eta, np.asarray(k, float))


def _composite(t, base, pev, rav, dzv, gamma, phi):
    # T scales the threat-bearing terms only; the corrective DZV is undiscounted (spec §4.6).
    return np.asarray(t, float) * (np.asarray(base, float) + gamma * np.asarray(pev, float) + np.asarray(rav, float)) + phi * np.asarray(dzv, float)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/tracking/test_xt_gk.py::TestComponents -q`
Expected: PASS.

- [ ] **Step 5: Checkpoint** — components green.

---

## Task P1-4: Domain filter `_gk_distribution_mask`

**Files:**
- Modify: `silly_kicks/tracking/_xt_gk.py`
- Test: `tests/tracking/test_xt_gk.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/tracking/test_xt_gk.py  (append)
from silly_kicks.tracking._xt_gk import _gk_distribution_mask


class TestDomainFilter:
    def _frames(self):
        # GK of team 1 is player 10; GK of team 2 is player 20
        return pd.DataFrame(
            {
                "game_id": [9, 9, 9, 9],
                "team_id": [1, 2, 1, 2],
                "player_id": [10, 20, 11, 21],
                "is_goalkeeper": [True, True, False, False],
                "is_ball": [False, False, False, False],
            }
        )

    def test_goalkick_always_in_scope(self):
        actions = pd.DataFrame(
            {"game_id": [9], "team_id": [1], "player_id": [11], "type_id": [22]}  # goalkick by outfielder
        )
        assert _gk_distribution_mask(actions, self._frames()).tolist() == [True]

    def test_open_pass_in_scope_only_if_actor_is_gk(self):
        actions = pd.DataFrame(
            {
                "game_id": [9, 9],
                "team_id": [1, 1],
                "player_id": [10, 11],   # GK, outfielder
                "type_id": [0, 0],       # pass, pass
            }
        )
        assert _gk_distribution_mask(actions, self._frames()).tolist() == [True, False]

    def test_throw_in_by_gk_in_scope(self):
        actions = pd.DataFrame(
            {"game_id": [9], "team_id": [2], "player_id": [20], "type_id": [2]}  # throw_in by GK
        )
        assert _gk_distribution_mask(actions, self._frames()).tolist() == [True]

    def test_shot_never_in_scope(self):
        actions = pd.DataFrame(
            {"game_id": [9], "team_id": [1], "player_id": [10], "type_id": [11]}  # shot by GK
        )
        assert _gk_distribution_mask(actions, self._frames()).tolist() == [False]

    def test_id_dtype_mismatch_string_frames(self):
        # ADR-019: numeric action ids vs string frame ids must still resolve actor-is-GK
        frames = self._frames().astype({"team_id": str, "player_id": str})
        actions = pd.DataFrame(
            {"game_id": [9], "team_id": [1], "player_id": [10], "type_id": [0]}
        )
        assert _gk_distribution_mask(actions, frames).tolist() == [True]
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/tracking/test_xt_gk.py::TestDomainFilter -q`
Expected: FAIL — `_gk_distribution_mask` not defined.

- [ ] **Step 3: Write minimal implementation**

```python
# silly_kicks/tracking/_xt_gk.py  (append; add this import at top with the others)
from ._id_compat import canonical_id_series

_GOALKICK = spadlconfig.actiontype_id["goalkick"]   # 22
_PASS = spadlconfig.actiontype_id["pass"]           # 0
_THROW_IN = spadlconfig.actiontype_id["throw_in"]   # 2


def _gk_distribution_mask(actions: pd.DataFrame, frames: pd.DataFrame) -> npt.NDArray[np.bool_]:
    """True for in-scope GK distributions: any goalkick, OR a pass/throw_in whose actor
    is the acting team's goalkeeper (resolved from frames' is_goalkeeper flag, which
    derived-GK populates for Metrica/SkillCorner per PR-S26). dtype-safe id matching
    (ADR-019). Non-GK-distribution rows -> False (pass through unchanged downstream)."""
    type_id = actions["type_id"].to_numpy()
    is_goalkick = type_id == _GOALKICK
    is_open = np.isin(type_id, (_PASS, _THROW_IN))

    gk = frames[frames["is_goalkeeper"].astype(bool) & (~frames["is_ball"].astype(bool))]
    keyed_by_game = "game_id" in actions.columns and "game_id" in frames.columns

    # Build a canonical-id set of (game?, team, player) goalkeeper identities.
    gk_team = canonical_id_series(gk["team_id"])
    gk_player = canonical_id_series(gk["player_id"])
    act_team = canonical_id_series(actions["team_id"])
    act_player = canonical_id_series(actions["player_id"])
    if keyed_by_game:
        gk_game = canonical_id_series(gk["game_id"])
        act_game = canonical_id_series(actions["game_id"])
        gk_set = set(zip(gk_game, gk_team, gk_player))
        actor_is_gk = np.array(
            [(g, t, p) in gk_set for g, t, p in zip(act_game, act_team, act_player)]
        )
    else:
        gk_set = set(zip(gk_team, gk_player))
        actor_is_gk = np.array([(t, p) in gk_set for t, p in zip(act_team, act_player)])

    return is_goalkick | (is_open & actor_is_gk)
```

> If `canonical_id_series` returns a pandas Series, iterate via `.to_numpy()`; adjust the `zip(...)` sources to numpy arrays if the executor hits a dtype/iteration issue. Verify `canonical_id_series` exists and returns hashable scalars (it canonicalizes integral-float, leaves genuine strings).

- [ ] **Step 4: Run to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/tracking/test_xt_gk.py::TestDomainFilter -q`
Expected: PASS.

- [ ] **Step 5: Checkpoint** — domain filter green (including the ADR-019 string-frame case).

---

## Task P1-5: `compute_xt_gk` batch + `[das]` fail-loud guard + construct-invariant gates

**Files:**
- Modify: `silly_kicks/tracking/_xt_gk.py`
- Test: `tests/tracking/test_xt_gk.py`

- [ ] **Step 1: Write the failing tests** (fixture + the spec §7 construct-invariant gates)

```python
# tests/tracking/test_xt_gk.py  (append)
from silly_kicks.tracking._xt_gk import compute_xt_gk
from silly_kicks.xthreat import ExpectedThreat

pytest.importorskip("accessible_space")  # compute_xt_gk hard-requires [das]

_XT_GK_COLS = ["xt_gk_base", "xt_gk_pev", "xt_gk_rav", "xt_gk_dzv", "xt_gk_pressure", "xt_gk"]


def _fitted_xt():
    xt = ExpectedThreat(l=16, w=12)
    xt.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))  # value rises toward goal (+x)
    return xt


def _gk_realistic_xt():
    """GK-zone-REALISTIC grid: own third xT ~0.001-0.01 rising toward goal — REQUIRED by the
    DZV gate. The flat ramp (_fitted_xt) puts the defensive third at ~0.2, two orders of
    magnitude above real GK-zone xT (~0.001-0.005, spec §1) and >> v_def (0.02), which makes
    the back-pass correction (v_def - xT(z)) go NEGATIVE and the construct gate falsely fail.
    Do NOT simplify this back to the ramp. (Cube concentrates value toward goal; at the
    back-pass origin x=25 -> xi=3 -> (3/15)**3 = 0.008 < v_def.)"""
    xt = ExpectedThreat(l=16, w=12)
    xt.xT = np.tile(np.linspace(0.0, 1.0, 16) ** 3, (12, 1))
    return xt


def _gk_actions():
    # Two GK distributions by GK (player 10, team 1): a forward goalkick + a back-pass-ish
    return pd.DataFrame(
        {
            "game_id": [9, 9],
            "action_id": [0, 1],
            "team_id": [1, 1],
            "player_id": [10, 10],
            "period_id": [1, 1],
            "time_seconds": [5.0, 50.0],
            "type_id": [22, 0],
            "start_x": [5.0, 25.0],
            "start_y": [34.0, 34.0],
            "end_x": [55.0, 10.0],
            "end_y": [34.0, 34.0],
        }
    )


def _frames_for(actions):
    """DAS-valid frames (review B1): get_xc → _prepare_frames → _validate_das_inputs
    (_das.py:123) HARD-RAISES on missing vx/vy/team_in_possession. The frames MUST carry
    them, with non-degenerate velocities and a set possession so accessible-space yields a
    real (non-NaN) xC. PREFER importing a known-good frames factory from the existing
    get_xc / add_das suites (e.g. the `_frames()` helper in tests/tracking/test_das.py,
    used at test_das.py:1443+) — you inherit its DAS-contract correctness. The
    self-contained version below is the fallback; keep it in sync with the DAS contract."""
    rows = []
    for fid, (t, period) in enumerate([(5.0, 1), (50.0, 1)]):
        for pid, team, gk, x, y, vx, vy in [
            (10, 1, True, 5.0, 34.0, 0.5, 0.0),
            (11, 1, False, 30.0, 30.0, 1.0, 0.2),
            (12, 1, False, 45.0, 40.0, 1.2, -0.1),
            (20, 2, True, 100.0, 34.0, -0.3, 0.0),
            (21, 2, False, 40.0, 40.0, -1.0, 0.1),
            (22, 2, False, 55.0, 28.0, -0.8, 0.0),
            (-1, -1, False, 6.0, 34.0, 0.5, 0.0),  # ball
        ]:
            rows.append(
                dict(game_id=9, period_id=period, frame_id=fid, time_seconds=t,
                     frame_rate=25.0, team_id=team, player_id=pid, is_goalkeeper=gk,
                     is_ball=(pid == -1), x=x, y=y, vx=vx, vy=vy,
                     team_in_possession=1,  # team 1 (the GK's team) has the ball
                     source_provider="sportec", team_attacking_direction="ltr")
            )
    return pd.DataFrame(rows)


class TestComputeXtGk:
    def test_emits_all_columns_for_in_scope_only(self):
        actions = _gk_actions()
        actions.loc[2] = dict(  # add an outfield pass -> out of scope
            game_id=9, action_id=2, team_id=1, player_id=11, period_id=1,
            time_seconds=5.0, type_id=0, start_x=40.0, start_y=34.0, end_x=60.0, end_y=34.0,
        )
        frames = _frames_for(actions)
        out = compute_xt_gk(actions, frames, xt=_fitted_xt())
        assert list(out.columns) == _XT_GK_COLS
        assert len(out) == len(actions)
        assert out.loc[2, "xt_gk"] != out.loc[2, "xt_gk"]  # NaN for out-of-scope (NaN != NaN)
        assert out.loc[0, "xt_gk_base"] == out.loc[0, "xt_gk_base"]  # not NaN

    def test_uses_injected_grid_not_self_fit(self):
        # Option B: base = -xT★(origin); with sigma=0, xT★ == injected grid -> deterministic.
        # Proves the injected grid is consumed (origin lookup), no self-fit.
        from silly_kicks.tracking._xt_gk import XtGkParams, _grid_value
        xt = _fitted_xt()
        actions = _gk_actions().iloc[[0]].copy()  # goalkick from start_x=5, start_y=34
        frames = _frames_for(actions)
        params = XtGkParams(convolution_sigma=0.0)
        out = compute_xt_gk(actions, frames, xt=xt, params=params)
        expected_base = -_grid_value(xt.xT, np.array([5.0]), np.array([34.0]))[0]
        assert out.loc[0, "xt_gk_base"] == pytest.approx(expected_base)

    def test_rejects_unfitted_grid(self):
        # M1 — leakage contract, "garbage-in rejected" half: an all-zero (unfitted) grid raises.
        actions = _gk_actions().iloc[[0]].copy()
        frames = _frames_for(actions)
        unfitted = ExpectedThreat(l=16, w=12)  # xT.all() == 0
        with pytest.raises(ValueError, match="FITTED"):
            compute_xt_gk(actions, frames, xt=unfitted)

    # ---- spec §7 construct-invariant gates (Phase-1 gates) ----
    def test_backpass_penalty_corrected_upward(self):
        # composite WITH dzv strictly exceeds composite with dzv disabled (φ·DZV > 0).
        # MUST use the GK-zone-realistic grid (own third < v_def) — the flat ramp inverts
        # the correction (v_def - xT(z) < 0). See _gk_realistic_xt docstring.
        from silly_kicks.tracking._xt_gk import XtGkParams
        actions = _gk_actions().iloc[[1]].copy()  # start_x=25 -> defensive third
        frames = _frames_for(actions)
        out = compute_xt_gk(actions, frames, xt=_gk_realistic_xt(), params=XtGkParams(phi=1.0))
        assert out.loc[1, "xt_gk_dzv"] > 0.0
        # NOTE (review #3): we deliberately do NOT also assert `with_dzv.xt_gk >
        # without_dzv.xt_gk`. That would route the claim through the composite, whose threat
        # part carries RAV's xC — which can be NaN on a synthetic fixture (the get_xc OOD
        # note), giving a `NaN > NaN` false-fail unrelated to DZV. "DZV raises the composite"
        # is already proven xC-free by assertion 1 above + the P1-3 _composite unit oracle
        # (composite = T·(base + γ·PEV + RAV) + φ·DZV, so +φ·DZV strictly raises it for DZV>0).

    def test_higher_pressure_gives_higher_pev(self):
        # Same forward pass, more defenders crowding the actor -> higher ρ -> higher PEV.
        actions = _gk_actions().iloc[[0]].copy()
        low = _frames_for(actions)
        high = low.copy()
        # add two opponents right next to the actor at the linked frame (more pressure)
        extra = pd.DataFrame([
            dict(game_id=9, period_id=1, frame_id=0, time_seconds=5.0, team_id=2,
                 player_id=22, is_goalkeeper=False, is_ball=False, x=6.0, y=35.0),
            dict(game_id=9, period_id=1, frame_id=0, time_seconds=5.0, team_id=2,
                 player_id=23, is_goalkeeper=False, is_ball=False, x=7.0, y=33.0),
        ])
        high = pd.concat([high, extra], ignore_index=True)
        out_low = compute_xt_gk(actions, low, xt=_fitted_xt())
        out_high = compute_xt_gk(actions, high, xt=_fitted_xt())
        assert out_high.loc[0, "xt_gk_pressure"] > out_low.loc[0, "xt_gk_pressure"]
        assert out_high.loc[0, "xt_gk_pev"] >= out_low.loc[0, "xt_gk_pev"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/tracking/test_xt_gk.py::TestComputeXtGk -q`
Expected: FAIL — `compute_xt_gk` not defined.

- [ ] **Step 3: Write minimal implementation**

```python
# silly_kicks/tracking/_xt_gk.py  (append; add TYPE_CHECKING import of ExpectedThreat)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from silly_kicks.xthreat import ExpectedThreat

_OUTPUT_COLS = ["xt_gk_base", "xt_gk_pev", "xt_gk_rav", "xt_gk_dzv", "xt_gk_pressure", "xt_gk"]


def _require_das() -> None:
    """Fail loud if accessible-space is missing — xT-GK's RAV always needs xC (spec §6).
    Never silently ship a RAV-less composite."""
    from ._das import _import_accessible_space
    try:
        _import_accessible_space()
    except ImportError as e:
        raise ImportError("xT-GK requires the [das] extra: pip install silly-kicks[das]") from e


def compute_xt_gk(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    xt: "ExpectedThreat",
    params: XtGkParams | None = None,
    links: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Batch xT-GK over GK-distribution actions. Returns a DataFrame indexed like
    ``actions`` with the six xt_gk_* columns; out-of-scope rows are NaN.

    ``xt`` MUST be a pre-fitted ExpectedThreat fitted on a corpus DISJOINT from the
    scored matches (the OBSO/frozen pattern). This function NEVER fits xT internally
    (no in-sample leakage)."""
    _require_das()
    from ._das import get_xc
    from .features import pressure_on_actor  # avoid import cycle at module load

    p = params or XtGkParams()
    # M1 — leakage/garbage-in guard: xt MUST be fitted (no self-fit). compute reads
    # xt.xT directly and would not otherwise raise (only .rate() raises NotFittedError).
    if not np.asarray(xt.xT).any():
        raise ValueError("xT-GK requires a FITTED ExpectedThreat (xt.xT is all-zero). "
                         "Fit xt on a corpus disjoint from the scored matches; xT-GK never self-fits.")

    out = pd.DataFrame(
        {c: np.full(len(actions), np.nan, dtype=float) for c in _OUTPUT_COLS},
        index=actions.index,
    )

    # B2 — NaN-safety contract (ADR-003) implemented in the BODY (the @nan_safe_enrichment
    # marker confers no behavior). Route in-scope rows with a NaN identifier to the NaN
    # default; never hand a NaN id to pressure_on_actor / get_xc.
    in_scope = _gk_distribution_mask(actions, frames)
    id_ok = actions["player_id"].notna().to_numpy() & actions["team_id"].notna().to_numpy()
    mask = in_scope & id_ok
    if not mask.any():
        return out
    sub = actions.loc[mask]

    # Convolved + raw grids.
    xt_star = _convolve_grid(xt.xT, p.convolution_sigma)
    sx = sub["start_x"].to_numpy(float)
    sy = sub["start_y"].to_numpy(float)
    ex = sub["end_x"].to_numpy(float)
    ey = sub["end_y"].to_numpy(float)

    dest_star = _grid_value(xt_star, ex, ey)             # M5 — hoist xT★(z'), used by progress and RAV
    origin_star = _grid_value(xt_star, sx, sy)
    progress = _progress(dest_star, origin_star)         # forward move value (feeds PEV)
    base = _base(origin_star)                            # Option B: origin-only; RAV owns the destination

    # Pressure ρ (continuous; spec §4.4).
    rho_raw = pressure_on_actor(sub, frames, method=p.pressure_method, links=links).to_numpy(float)
    rho = _normalize_pressure(rho_raw, p.pressure_scale)
    pev = _pev(rho, progress)

    # RAV completion probability via the tracking xC model (spec §4.3); RAV solely owns z'.
    xc = get_xc(sub, frames)["xC"].to_numpy(float)
    rav = _rav(xc, dest_star, _counter_value(xt_star, ex, ey), p.delta)

    # DZV uses the RAW grid at origin (spec §4.2).
    dzv = _dzv(sx, _grid_value(xt.xT, sx, sy), p.v_def, p.defensive_third_boundary)

    # Temporal discount (near-inert for possession-starters).
    k = _possession_depth(actions)[mask]
    t = _temporal(k, p.eta)

    composite = _composite(t, base, pev, rav, dzv, p.gamma, p.phi)

    # M2 — surface (don't silently swallow) in-scope rows that produced a NaN composite
    # because pressure/xC failed to link. The per-row detail is already exposed via the
    # provenance link_quality_score column (NaN where unlinked); this warn makes it loud.
    # V2 (review #2): use a FIXED message (no count interpolation) so Python's per-(message,
    # location) dedup collapses it — xfns calls this 3×/slot/match, and a varying count would
    # emit a distinct warning every time and flood real feature builds.
    if bool(np.isnan(composite).any()):
        import warnings
        warnings.warn(
            "xT-GK: one or more in-scope GK distributions produced NaN xt_gk "
            "(pressure/xC could not link to a frame); see the link_quality_score column.",
            stacklevel=2,
        )

    out.loc[mask, "xt_gk_base"] = base
    out.loc[mask, "xt_gk_pev"] = pev
    out.loc[mask, "xt_gk_rav"] = rav
    out.loc[mask, "xt_gk_dzv"] = dzv
    out.loc[mask, "xt_gk_pressure"] = rho
    out.loc[mask, "xt_gk"] = composite
    return out
```

> `_gk_distribution_mask` and `id_ok` are plain `np.ndarray[bool]`, so `mask` is too and `_possession_depth(actions)[mask]` indexes positionally. This NaN-id routing is what makes `test_nan_identifier_routes_to_default_no_crash` (P1-6) pass *by the body*, not the decorator.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/tracking/test_xt_gk.py::TestComputeXtGk -q`
Expected: PASS.

> **If `get_xc` returns all-NaN xC on the goalkick fixture** (the review-#2.1 OOD risk for long-aerial goal-kicks/throws): that is the documented impl-time verification point. For the synthetic fixture, ensure frames carry enough players for accessible-space to simulate; if xC is NaN, RAV becomes NaN and the composite NaN — acceptable for the unit fixture (assert components other than RAV), but record the OOD finding in the ADR and consider a geometry-based completion prior for goal-kicks as a follow-up. Do NOT mask the OOD behind the fixture.

- [ ] **Step 5: Checkpoint** — `compute_xt_gk` + all three construct-invariant gates green.

---

## Task P1-6: `add_xt_gk` aggregator (nan-safe, idempotent provenance, `links`)

**Files:**
- Modify: `silly_kicks/tracking/features.py`
- Test: `tests/tracking/test_xt_gk.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/tracking/test_xt_gk.py  (append)
from silly_kicks.tracking.features import add_xt_gk


class TestAddXtGk:
    def test_merges_columns_and_provenance(self):
        actions = _gk_actions()
        frames = _frames_for(actions)
        out = add_xt_gk(actions, frames, _fitted_xt(), home_team_id=1)
        for c in _XT_GK_COLS:
            assert c in out.columns
        # provenance columns merged once
        assert "frame_id" in out.columns
        assert len(out) == len(actions)

    def test_idempotent_provenance_on_chained_calls(self):
        actions = _gk_actions()
        frames = _frames_for(actions)
        once = add_xt_gk(actions, frames, _fitted_xt(), home_team_id=1)
        twice = add_xt_gk(once, frames, _fitted_xt(), home_team_id=1)
        assert "frame_id_x" not in twice.columns
        assert "frame_id_y" not in twice.columns

    def test_nan_identifier_routes_to_default_no_crash(self):
        # ADR-003 contract is implemented in the BODY (not the marker). Row 0 is a goalkick
        # (type 22 -> in-scope), so a NaN player_id is an in-scope NaN-id action: it must
        # route to the NaN default WITHOUT ever reaching get_xc/pressure with a NaN id.
        actions = _gk_actions()
        actions.loc[0, "player_id"] = np.nan
        frames = _frames_for(actions)
        out = add_xt_gk(actions, frames, _fitted_xt(), home_team_id=1)
        assert len(out) == len(actions)
        assert np.isnan(out.loc[0, "xt_gk"])  # NaN-id row -> default, did not crash
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/tracking/test_xt_gk.py::TestAddXtGk -q`
Expected: FAIL — `add_xt_gk` not importable.

- [ ] **Step 3: Write minimal implementation**

Add to `silly_kicks/tracking/features.py` (near the `gk_influence` family; ensure imports `from ._xt_gk import XtGkParams, compute_xt_gk` and `from .utils import link_actions_to_frames` exist):

```python
@nan_safe_enrichment
def add_xt_gk(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    xt: ExpectedThreat,
    *,
    links: pd.DataFrame | None = None,
    home_team_id: int | str,
    params: XtGkParams | None = None,
) -> pd.DataFrame:
    """Add xT-GK columns (xt_gk_base/pev/rav/dzv/pressure + composite xt_gk) per
    GK-distribution action. ``xt`` is a REQUIRED pre-fitted ExpectedThreat (no self-fit
    — leakage contract, spec §3). ``home_team_id`` is accepted for GK-feature-family
    signature parity (and CI-gate construction); the xT-GK math operates on
    LTR-normalized SPADL action coordinates and does not consume it. Requires the [das]
    extra (RAV uses get_xc).

    See NOTICE for full bibliographic citations (Eyestone xT-GK)."""
    out = actions.copy()
    pointers = links if links is not None else link_actions_to_frames(actions, frames)[0]

    comp = compute_xt_gk(actions, frames, xt=xt, params=params, links=pointers)
    for c in comp.columns:
        out[c] = comp[c].to_numpy()

    # Idempotent provenance merge (skip if any provenance column already present).
    provenance_cols = ["frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]
    existing = [c for c in provenance_cols if c in out.columns]
    if not existing and len(pointers) > 0:
        ptr_cols = pointers.set_index("action_id")[provenance_cols]
        out = out.merge(ptr_cols, left_on="action_id", right_index=True, how="left")
    return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/tracking/test_xt_gk.py::TestAddXtGk -q`
Expected: PASS.

- [ ] **Step 5: Checkpoint** — aggregator green.

---

## Task P1-7: `xt_gk_xfns` VAEP factory (dup-action_id-safe)

**Files:**
- Modify: `silly_kicks/tracking/features.py`
- Test: `tests/tracking/test_xt_gk.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/tracking/test_xt_gk.py  (append)
from silly_kicks.tracking.features import xt_gk_xfns


class TestXtGkXfns:
    def test_factory_returns_frame_aware_transformer(self):
        fns = xt_gk_xfns(_fitted_xt(), home_team_id=1)
        assert len(fns) == 1
        assert getattr(fns[0], "_frame_aware", False) is True

    def test_produces_values_on_duplicate_action_ids(self):
        # VAEP shifted gamestate slots repeat the boundary action -> non-unique action_id.
        # The xfn must produce values (not crash / mis-resolve) — ADR-020.
        base = _gk_actions()
        frames = _frames_for(base)
        slot = pd.concat([base, base.iloc[[0]]], ignore_index=True)  # dup action_id=0
        states = [slot, slot, slot]
        fn = xt_gk_xfns(_fitted_xt(), home_team_id=1)[0]
        res = fn(states, frames)
        assert "xt_gk_a0" in res.columns
        assert len(res) == len(slot)
        # the two rows sharing action_id=0 get the same (time-linked) value
        assert res["xt_gk_base_a0"].iloc[0] == pytest.approx(res["xt_gk_base_a0"].iloc[-1], nan_ok=True)

    def test_none_frames_yields_nan_columns(self):
        base = _gk_actions()
        fn = xt_gk_xfns(_fitted_xt(), home_team_id=1)[0]
        res = fn([base, base, base], None)
        assert res["xt_gk_a0"].isna().all()
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/tracking/test_xt_gk.py::TestXtGkXfns -q`
Expected: FAIL — `xt_gk_xfns` not importable.

- [ ] **Step 3: Write minimal implementation**

**H2 — why the per-slot action_id rekey (not `resolve_frame_ids_by_position` alone):**
`gk_influence_xfns` can rely on `resolve_frame_ids_by_position` because it computes
**per-frame** (`compute_gk_influence(frame_data)`) and never calls an action_id-keyed
sub-linker. xT-GK *must* call `pressure_on_actor` and `get_xc`, which are **batch,
action_id-keyed** APIs (`pressure_on_actor` → `_resolve_action_frame_context` merges on
`action_id`, `utils.py:672/717`). On a shifted gamestate slot with non-unique `action_id`
those merges fan out / mis-resolve. Re-keying `action_id` to a unique positional surrogate
per slot is **exactly what `resolve_frame_ids_by_position` does internally** ("re-keys to a
unique positional surrogate before linking", `_kernels.py:875`) — same mechanism, applied
one layer up because the sub-calls own their linking. Frame linkage is by time, so values
are unchanged; `action_id` is ephemeral here (only the value columns are returned). The
behavioral gate (`test_frame_aware_xfns_dup_action_id.py`) auto-enrolls and exercises this.

Add to `silly_kicks/tracking/features.py`:

```python
def xt_gk_xfns(
    xt: ExpectedThreat,
    *,
    home_team_id: int | str,
    params: XtGkParams | None = None,
) -> list:
    """Factory returning one frame-aware VAEP transformer for xT-GK, closing over the
    caller-fitted ``xt`` (no self-fit — leakage contract, spec §6). Emits xt_gk_*_a{i}
    per gamestate slot. ``home_team_id`` accepted for family parity + CI-gate
    construction (see add_xt_gk)."""
    from ._xt_gk import _OUTPUT_COLS, compute_xt_gk

    def _xt_gk_transformer(states, frames):
        out = pd.DataFrame(index=states[0].index)
        if frames is None:
            for i in range(3):
                for col in _OUTPUT_COLS:
                    out[f"{col}_a{i}"] = np.nan
            return out
        for i, slot in enumerate(states[:3]):
            # Re-key action_id to a unique positional surrogate so action_id-keyed
            # linking (pressure / xC) is correct on shifted slots that repeat the
            # boundary action (ADR-020). Frame linkage is by time, so values are
            # unchanged; action_id is ephemeral here (we return only the value cols).
            safe = slot.copy()
            safe["action_id"] = np.arange(len(safe))
            comp = compute_xt_gk(safe, frames, xt=xt, params=params)
            comp.index = slot.index
            for col in _OUTPUT_COLS:
                out[f"{col}_a{i}"] = comp[col].to_numpy()
        return out

    _xt_gk_transformer._frame_aware = True
    _xt_gk_transformer.__name__ = "xt_gk"
    return [_xt_gk_transformer]
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/tracking/test_xt_gk.py::TestXtGkXfns -q`
Expected: PASS.

- [ ] **Step 5: Checkpoint** — xfns green.

---

## Task P1-8: Public exports

**Files:**
- Modify: `silly_kicks/tracking/__init__.py`
- Test: `tests/tracking/test_xt_gk.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/tracking/test_xt_gk.py  (append)
class TestExports:
    def test_public_surface_importable_from_tracking(self):
        import silly_kicks.tracking as T
        assert hasattr(T, "compute_xt_gk")
        assert hasattr(T, "add_xt_gk")
        assert hasattr(T, "xt_gk_xfns")
        assert hasattr(T, "XtGkParams")
        for name in ("compute_xt_gk", "add_xt_gk", "xt_gk_xfns", "XtGkParams"):
            assert name in T.__all__
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/tracking/test_xt_gk.py::TestExports -q`
Expected: FAIL — names absent from `silly_kicks.tracking`.

- [ ] **Step 3: Implement**

In `silly_kicks/tracking/__init__.py`:
- Add `add_xt_gk` and `xt_gk_xfns` to the `from .features import (...)` block (alphabetical-ish position).
- Add `from ._xt_gk import XtGkParams, compute_xt_gk` (these live in `_xt_gk.py`, not `features.py`).
- Add `"add_xt_gk"`, `"xt_gk_xfns"`, `"XtGkParams"`, `"compute_xt_gk"` to `__all__`.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/tracking/test_xt_gk.py::TestExports -q`
Expected: PASS.

- [ ] **Step 5: Checkpoint** — exports green.

---

## Task P1-9: Atomic mirror (endpoint synthesis)

**Files:**
- Modify: `silly_kicks/atomic/tracking/features.py`
- Test: `tests/tracking/test_xt_gk.py`

**Why synthesis (unlike `gk_influence`):** xT-GK reads action endpoint coords (`start_x/start_y/end_x/end_y` for z and z'). Atomic SPADL uses `x, y, dx, dy`. So the atomic mirror must synthesize `start_x=x, start_y=y, end_x=x+dx, end_y=y+dy` (the `_structural_pass_atomic_endpoints` pattern) before delegating.

- [ ] **Step 1: Write the failing test**

```python
# tests/tracking/test_xt_gk.py  (append)
class TestAtomicMirror:
    def test_atomic_add_xt_gk_matches_standard_via_synthesis(self):
        from silly_kicks.atomic.tracking.features import add_xt_gk as atomic_add_xt_gk
        std = _gk_actions()
        frames = _frames_for(std)
        std_out = add_xt_gk(std, frames, _fitted_xt(), home_team_id=1)

        # atomic representation of the same actions
        atom = std.rename(columns={"start_x": "x", "start_y": "y"}).copy()
        atom["dx"] = std["end_x"].to_numpy() - std["start_x"].to_numpy()
        atom["dy"] = std["end_y"].to_numpy() - std["start_y"].to_numpy()
        atom = atom.drop(columns=["end_x", "end_y"])
        atom_out = atomic_add_xt_gk(atom, frames, _fitted_xt(), home_team_id=1)

        np.testing.assert_allclose(
            atom_out["xt_gk_base"].to_numpy(), std_out["xt_gk_base"].to_numpy(),
            equal_nan=True, rtol=1e-9,
        )

    def test_atomic_exports(self):
        import silly_kicks.atomic.tracking.features as AF
        assert "add_xt_gk" in AF.__all__
        assert "xt_gk_xfns" in AF.__all__
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/tracking/test_xt_gk.py::TestAtomicMirror -q`
Expected: FAIL — atomic `add_xt_gk` absent.

- [ ] **Step 3: Implement** in `silly_kicks/atomic/tracking/features.py`

```python
# add near _structural_pass_atomic_endpoints
from silly_kicks.tracking.features import xt_gk_xfns as _std_xt_gk_xfns

def _xt_gk_atomic_endpoints(actions: pd.DataFrame) -> pd.DataFrame:
    adapted = actions.copy()
    adapted["start_x"] = adapted["x"]
    adapted["start_y"] = adapted["y"]
    adapted["end_x"] = adapted["x"] + adapted["dx"]
    adapted["end_y"] = adapted["y"] + adapted["dy"]
    return adapted


def add_xt_gk(actions, frames, xt, *, links=None, home_team_id, params=None):
    """Atomic mirror of tracking.add_xt_gk — synthesizes start/end from x,y,dx,dy then
    delegates. See silly_kicks.tracking.add_xt_gk + NOTICE."""
    from silly_kicks.tracking.features import add_xt_gk as _std_add_xt_gk
    adapted = _xt_gk_atomic_endpoints(actions)
    out = _std_add_xt_gk(adapted, frames, xt, links=links, home_team_id=home_team_id, params=params)
    return out.drop(columns=["start_x", "start_y", "end_x", "end_y"])


def xt_gk_xfns(xt, *, home_team_id, params=None):
    """Atomic mirror of tracking.xt_gk_xfns. The transformer synthesizes endpoints per
    slot before delegating to the standard kernel."""
    std = _std_xt_gk_xfns(xt, home_team_id=home_team_id, params=params)[0]

    def _atomic_xt_gk_transformer(states, frames):
        adapted_states = [_xt_gk_atomic_endpoints(s) for s in states]
        return std(adapted_states, frames)

    _atomic_xt_gk_transformer._frame_aware = True
    _atomic_xt_gk_transformer.__name__ = "xt_gk"
    return [_atomic_xt_gk_transformer]
```
Add `"add_xt_gk"` and `"xt_gk_xfns"` to the atomic features `__all__`.

> If the atomic features module instead re-exports unchanged for `start/end`-free features, follow the `structural_pass` precedent in that file exactly — it is the only existing endpoint-synthesizing mirror.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/tracking/test_xt_gk.py::TestAtomicMirror -q`
Expected: PASS.

- [ ] **Step 5: Checkpoint** — atomic mirror green.

---

## Task P1-9b: Real-provider OOD smoke for `get_xc` on goal-kicks (e2e, owner-gated)

**Files:**
- Test: `tests/tracking/test_xt_gk_e2e.py` (new, `@pytest.mark.e2e`)

**Why (review H3):** accessible-space's xC is validated on **open-play passes**, but xT-GK
scores **long aerial goal-kicks** — a different completion regime (spec review #2.1). The
synthetic construct-invariant gates (P1-5) are deliberately provider-independent and
*cannot* detect this. This is **not** a CI gate (e2e fixtures aren't committed; e2e tests
self-skip per CLAUDE.md, and spec §7 keeps real-data checks "corroborating, not gating") —
it is an **owner-run verification** with a hard escalation rule.

- [ ] **Step 1: Write the e2e smoke** — load real GK distributions (IDSSE or SkillCorner;
  reuse the existing e2e loaders), run `compute_xt_gk`, and assert: (a) `xt_gk_rav` is
  **finite** for goal-kicks specifically (not all-NaN); (b) the implied xC band is sane
  (e.g. goal-kick completion mostly in [0.2, 0.95], not pinned at 0/1/NaN).

```python
import pytest
pytestmark = pytest.mark.e2e

def test_get_xc_sane_on_real_goalkicks(idsse_or_skillcorner_match, frozen_xt_train_actions):
    # V4 (review #2): xt is fit on a CONCRETE corpus DISJOINT from the scored match — reuse
    # the held-out training corpus the OBSO / frozen-xt e2e already loads (e.g. the
    # competition/season excluded from `idsse_or_skillcorner_match`). Do not fit on the
    # scored match (leakage). If no such fixture exists yet, add a small held-out loader.
    actions, frames = idsse_or_skillcorner_match            # existing e2e loader
    from silly_kicks.tracking import compute_xt_gk
    from silly_kicks.xthreat import ExpectedThreat
    xt = ExpectedThreat(); xt.fit(frozen_xt_train_actions)  # DISJOINT corpus fixture
    out = compute_xt_gk(actions, frames, xt=xt)
    gk = out.loc[actions["type_id"] == 22]                   # goalkicks
    assert gk["xt_gk_rav"].notna().mean() > 0.5              # xC resolves for most goalkicks
```

- [ ] **Step 2: Escalation rule (binding).** If goal-kick xC is garbage (mostly NaN, or
  pinned at 0/1, or otherwise implausible for aerials), the **geometry-based completion
  prior** the spec hypothesizes (§4.3) becomes **Phase-1 scope** — add a goal-kick-regime
  completion fallback to `compute_xt_gk` and a unit test — **not** a post-hoc follow-up.
  Record the finding (sane / escalated) in the ADR (Task P1-12) with the measured band.

- [ ] **Step 3: Checkpoint** — e2e smoke runs on owner hardware; outcome recorded in the ADR.

---

## Task P1-10: CI-gate registration (auto-discovery surfaces)

**Files:**
- Modify: `tests/tracking/conftest_id_dtype.py` (ADR-019 id-dtype gate)
- Modify: `tests/tracking/test_frame_aware_xfns_dup_action_id.py` (ADR-020 meta floor)
- Modify: `tests/test_enrichment_nan_safety.py`

These gates auto-discover by name; registration is mandatory or the meta-assertions fail.

- [ ] **Step 1: id-dtype gate** — add to the `AGGREGATORS` list in `tests/tracking/conftest_id_dtype.py` (use `_axh` — positional `xt` + keyword `home_team_id`, same shape as `add_gk_influence`):

```python
    _axh(F.add_xt_gk, "add_xt_gk"),
```

- [ ] **Step 2: dup-action_id meta floor** — in `tests/tracking/test_frame_aware_xfns_dup_action_id.py`, bump the floor (the `_build` probe already handles `(xt, home_team_id=1)`):

```python
    assert len(_XFNS_NAMES) >= 21  # was >= 20 (added xt_gk_xfns)
```

- [ ] **Step 3: nan-safety extra-kwargs** — in `tests/test_enrichment_nan_safety.py`:
  - add `"add_xt_gk"` to `_TRACKING_NEEDS_EXTRA`;
  - extend the `gk_influence` construction branch in `test_tracking_helper_extra_kwargs_nan_safe` to include `add_xt_gk`:

```python
    elif name == "add_xt_gk":
        import numpy as np
        from silly_kicks.xthreat import ExpectedThreat
        from silly_kicks.tracking._xt_gk import _gk_distribution_mask
        # V1 (review #2): VERIFY (don't assume) that this shared fixture has zero in-scope GK
        # distributions, so get_xc is never reached. If the fixture ever gains a goalkick or a
        # GK-actor pass, get_xc runs and the frame ball+player co-occurrence contract applies
        # (a hard-crash path the defensive columns below do NOT cover) — fail loudly here with
        # a clear message instead of a confusing downstream crash.
        assert _gk_distribution_mask(actions, frames).sum() == 0, (
            "add_xt_gk nan-safety branch assumes no in-scope GK distributions in the shared "
            "fixture; it changed — give this branch frames satisfying the full get_xc contract."
        )
        xt = ExpectedThreat(l=16, w=12)
        xt.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))
        frames = frames.copy()  # defensive DAS columns (belt-and-suspenders; mask is empty)
        frames["vx"] = 0.0
        frames["vy"] = 0.0
        frames["team_in_possession"] = 1
        out = helper(actions, frames, xt, home_team_id=1)
    elif name in ("add_gk_influence", "add_cover_shadows", "add_player_influence"):
        import numpy as np
        from silly_kicks.xthreat import ExpectedThreat
        xt = ExpectedThreat(l=16, w=12)
        xt.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))
        out = helper(actions, frames, xt, home_team_id=1)
```

- [ ] **Step 4: Run the gates**

```bash
.venv\Scripts\python.exe -m pytest tests/tracking/test_id_dtype_invariance.py tests/tracking/test_frame_aware_xfns_dup_action_id.py tests/test_enrichment_nan_safety.py tests/tracking/test_provenance_skip_guard.py -q
```
Expected: PASS, including the meta-assertions (`test_enumerated_surface_equals_registered`, `test_meta_gate_covers_every_xfns_factory`).

> **Provenance-skip gate:** `test_provenance_skip_guard.py` uses a *static* chain of specific aggregators. Since `add_xt_gk` merges provenance, add it to the chain in `test_chained_enrichments_no_duplicate_provenance` (import it + append a call) so the chain exercises it. If the executor finds the chain is closed/parametrized, follow that file's existing pattern.

- [ ] **Step 5: Checkpoint** — all auto-discovery gates green.

---

## Task P1-11: Full-suite green + `convolution_sigma` one-off scan

**Files:**
- Modify: `silly_kicks/tracking/_xt_gk.py` (bake the scanned `convolution_sigma` default)
- One-off script (not committed): `scripts/_scan_xt_gk_sigma.py` *(or an inline session)*

- [ ] **Step 1: Run the one-off sensitivity scan** (spec §9 option (c) — fixes the lone smoothing constant; NOT a calibration phase). On a real tracking fixture (IDSSE/SkillCorner/GS), sweep `convolution_sigma ∈ {0.0, 0.5, 0.8, 1.0, 1.5, 2.0}`, compute xT-GK, and pick the smallest σ where the convolved grid is visibly smoothed without washing out the goal-ward gradient (eyeball + check `xt_gk_base` distribution stability). Record the chosen value + rationale in the ADR (Task P1-13).

- [ ] **Step 2: Bake the default** — set `XtGkParams.convolution_sigma` to the scanned value (default `0.8` is a placeholder; confirm or replace).

- [ ] **Step 3: Run the whole xT-GK suite + the broad tracking suite**

```bash
.venv\Scripts\python.exe -m pytest tests/tracking/test_xt_gk.py -v
.venv\Scripts\python.exe -m pytest tests/ -m "not e2e" -q --tb=short
```
Expected: all green. Read the actual summary line (`N passed`), not a piped tail.

- [ ] **Step 4: Lint + type the whole package** (replicate CI before push — see [replicate CI lint locally]):

```bash
.venv\Scripts\python.exe -m ruff check silly_kicks/ tests/ scripts/
.venv\Scripts\python.exe -m ruff format --check silly_kicks/ tests/ scripts/
.venv\Scripts\python.exe -m pyright silly_kicks/
```
Fix N806 / I001 / E402 / format issues. (`X`/`Y`/`Pscores` ML-naming ignores are vaep/xthreat-only; `_xt_gk.py` uses lowercase.)

- [ ] **Step 5: Checkpoint** — full suite + lint + types green.

---

## Task P1-12: Docs — NOTICE, ADR, CLAUDE.md, C4

**Files:**
- Modify: `NOTICE`
- Create: `docs/superpowers/adrs/ADR-NNN-xt-gk.md`
- Modify: `CLAUDE.md`
- Modify: `docs/c4/architecture.dsl` + regen `architecture.html`

- [ ] **Step 1: NOTICE** — add under "Mathematical / Methodological References" (mirror the GK-influence block format):

```
The xT-GK feature in silly_kicks/tracking/_xt_gk.py (PR-S##, Eyestone xT-GK)
implements an analytical Expected-Threat extension for goalkeeper distributions:

- Eyestone, J. (2025). "Expected Threat for Goalkeepers (xT-GK)." Winner, Pitch
  to the Pros 1.
  (GK-distribution value: pressure-escape, risk-adjusted, defensive-zone terms;
  team-philosophy parameter presets)

- Singh, K. (2018). "Introducing Expected Threat (xT)." karun.in/blog/expected-threat
  (baseline grid the GK terms compose over)

The functional forms are the silly-kicks formulation of Eyestone's xT-GK (the source
deck specifies components and parameter ranges, not closed-form equations).
Contributed publicly with attribution by Jeffrey Eyestone's explicit permission
(email 2026-06-06).
```

- [ ] **Step 2: ADR** — create `docs/superpowers/adrs/ADR-NNN-xt-gk.md` (reconcile the number against `origin/main` at release — next free after the current max; **verify the ADR-015 gap is genuinely free** before claiming it). Capture: pure-parametric-compute (not ADR-011 trained model); tracking-required (PEV needs pressure, no pressure survives SPADL); `get_xc` for P(success) + `[das]` hard requirement; required injected pre-fitted `xt` (no self-fit / leakage); normative params never calibrated; no separate calibration phase (option (c)); consent provenance (2026-06-06 public-with-attribution email). **Plus, from review:**
  - **M3:** one sentence stating xT-GK consumes **LTR-normalized** SPADL coordinates, so `home_team_id` is **parity-only** (CI-gate/family construction) and intentionally unused by the math — so a future reader does not "fix" the unused arg.
  - **M4:** record the **actual per-σ `xt_gk_base` distribution evidence** from the P1-11 scan (not just "picked 0.8"), noting the substrate is partly-suspect (spec §7).
  - **M2:** document that unlinked in-scope rows surface as a NaN composite **plus** a `warnings.warn` count + NaN `link_quality_score` provenance (never a silent substitution).
  - **H3:** record the goal-kick xC OOD smoke outcome (sane band, or escalated to a geometry prior).
  - **V3 (review #2):** one line stating xT-GK depends on **xthreat's private cell-index convention** (`xthreat._grid._get_cell_indexes`), pinned by `test_grid_value_pinned_to_expected_threat_rate`; **promote to a public `xthreat` API** (e.g. `ExpectedThreat.grid_value`) if xthreat refactors — the coupling is intentional, not an accidental reach-through.
  - **B3 (consent trail — audit-grade, review #2):** record Jeffrey's **verbatim words + date + channel** for BOTH consent points, to the same standard as the 2026-06-06 email: the 2026-06-08 reply *"1 B, 2 OK to go with provisional values"* (in response to the two-option question on the `(1+p)` destination weighting + preset publication) = **Option B** (destination counted once: base origin-only `−xT★(z)`, RAV solely owns `xT★(z')`) **and** provisional preset values approved. Note the engineering corollary (PEV retains a separate `progress` term so the origin-only base doesn't zero it). **State explicitly:** the composite `xt_gk` form is **no longer provisional** (Q1 resolved), but the **preset point-values remain provisional** pending an exact Q2 table — so §4.7's "provisional" labeling claims neither more nor less than Jeffrey signed off.

- [ ] **Step 3: CLAUDE.md** — add one architecture line in the tracking section:

```
PR-S## ships xT-GK (Eyestone, pure parametric GK-distribution value, tracking-required, [das]): `_xt_gk.py` `compute_xt_gk`/`add_xt_gk`/`xt_gk_xfns` + `XtGkParams.for_philosophy` presets; required injected pre-fitted `ExpectedThreat` (no self-fit); RAV via `get_xc`; emits xt_gk_base/pev/rav/dzv/pressure + composite. Additive (no default-xfn-list wiring, no retrain trigger). Decision: ADR-NNN.
```

- [ ] **Step 4: C4** — in `docs/c4/architecture.dsl`, bump the `tracking` container description count **`25 action-coupled aggregators` → `26`**. Verify the invariant: `len([n for n in silly_kicks.tracking.__all__ if n.startswith('add_')]) - 1 == 26` (the `-1` excludes `add_gradientsports_player_ids`). xT-GK is not a trained model / KDE backend, so no token-list change. Regen via the `mad-scientist-skills:c4` pipeline (structurizr.war + plantuml.jar, Java 21).

- [ ] **Step 5: Checkpoint** — docs updated; C4 regen produces a clean `architecture.html`.

---

## Task P1-13: Single final commit (sentinel-gated, approval required)

- [ ] **Step 1: Stage everything** — the feature code, tests, gate registrations, NOTICE, ADR, CLAUDE.md, spec (if not already committed), plan, and C4 artifacts go in **one** commit.

- [ ] **Step 2: Reconcile the version** — bump to the next free version against `origin/main` (currently 4.19.2 → likely **4.20.0**, a feature/minor) across the 5 sites (pyproject, `__init__`, CHANGELOG, TODO, `uv.lock` via `uv lock`) per the version-bump checklist.

- [ ] **Step 3: HOLD for explicit per-commit approval.** Present the exact `git commit` command + the full diff. **Do not create the sentinel** (`~/.claude-git-approval`) yourself ([never-create-sentinel-without-approval]). The user approves the sentinel and the commit explicitly. Write the multiline commit message to a temp file and use `git commit -F <file>` (never an inline `-m` with apostrophes — git-bash quoting trap). End the message with the `Co-Authored-By: Claude Opus 4.8 (1M context)` trailer.

- [ ] **Step 4: PR** — after the commit, bare `git push origin <branch>` (no pipes/chaining — the guard trips on compound push commands), then `gh pr create --body-file <file>`. Squash-merge with `--admin` (solo-maintainer review-required), annotated `v4.20.0` tag triggers publish.yml → PyPI. Confirm PyPI (cache-bust; CDN can lag ~1 min).

---

# PHASE 2 — Team/dataset parameter calibration *(subsequent release)*

**Status: gated on Jeffrey's spec Q7** (his preferred empirical estimation recipe, or ours to design). The API surface and a defensible default are below; the exact estimator is finalized once Q7 lands. This is **empirical per-team estimation of a tactical signature — NOT a VAEP-loss fit** (interpretability preserved); presets remain the default.

## Task P2-1: `XtGkParams.for_team(actions)` opt-in estimator

**Files:**
- Modify: `silly_kicks/tracking/_xt_gk.py`
- Test: `tests/tracking/test_xt_gk.py`

- [ ] **Step 1: Write the failing test** (behavioral — estimated params land in-range and shift with observed behaviour)

```python
class TestForTeam:
    def test_for_team_returns_in_range_params(self):
        from silly_kicks.tracking._xt_gk import XtGkParams
        actions = _gk_actions()  # extend to a realistic multi-action team fixture
        p = XtGkParams.for_team(actions, frames=_frames_for(actions), team_id=1)
        assert isinstance(p, XtGkParams)
        assert 0.1 <= p.gamma <= 0.4
        assert 0.3 <= p.delta <= 0.8

    def test_for_team_higher_risk_behaviour_raises_delta(self):
        # A team that loses possession more often on GK distributions -> higher δ (risk aversion).
        # Construct two corpora differing only in observed turnover rate; assert δ ordering.
        ...  # finalize once Q7 estimation recipe is confirmed
```

- [ ] **Step 2–4:** Implement `for_team` as an empirical estimator (defensible default pending Q7):
  - **δ (risk-aversion)** from the team's observed GK-distribution turnover rate (higher loss rate → higher δ);
  - **γ (pressure-escape)** from the team's observed propensity to progress forward under high pressure (more forward-under-pressure → higher γ);
  - **η / φ / v_def** left at preset/intent values unless Q7 says otherwise;
  - clamp all estimates into the deck ranges; fall back to `for_philosophy("possession")`-style neutral defaults on thin data.
  - Opt-in only — presets remain the library default; `for_team` is never auto-invoked.

- [ ] **Step 5: Checkpoint** — `for_team` green; document the estimator + its Q7 provenance in the ADR (amendment) and ship in the subsequent release (its own version bump + PR).

---

## Parallel review #3 (d32, 2026-06-08) — converged, incorporated

- **DZV gate assertion 2 was xC-fragile** (`with_dzv.xt_gk > without_dzv.xt_gk` routes through RAV's xC → `NaN > NaN` false-fail if get_xc is NaN on the synthetic fixture): **dropped** it. "DZV raises the composite" is already proven xC-free by assertion 1 (`xt_gk_dzv > 0`) + the P1-3 `_composite` oracle (P1-5).
- **Nit**: synced the stale cross-reference `test_nan_identifier_does_not_crash` → `test_nan_identifier_routes_to_default_no_crash` (P1-5 note).
- d32 verdict: **ship-quality / ready to execute**; hexagonal (pure core + SPADL/VAEP/atomic adapters; single outward `xthreat._grid` coupling acknowledged+pinned+ADR'd), TDD-disciplined, e2e appropriately scoped. No further holds.

## Parallel review #2 (d32, 2026-06-08) — incorporated

- **🔴 DZV gate fails on the synthetic ramp** (B1's fix unmasked it: ramp puts the defensive third at ~0.2 >> v_def=0.02 → `v_def − xT(z) < 0`): fixed test-side — dedicated `_gk_realistic_xt()` (own third < v_def, cube-concentrated toward goal) for the DZV gate, with a pinning comment; v_def untouched (normative), shared `_gk_actions` untouched (P1-5).
- **V1** (nan-safety "filtered out" was assumed): now **asserted** — the branch checks `_gk_distribution_mask(...).sum() == 0` with a clear failure message (P1-10).
- **V2** (count-interpolated warn defeats dedup → feature-build spam): fixed-message warn; per-row detail stays in `link_quality_score` (P1-5).
- **V3** (cross-package private import): kept (correct vs copy-paste, pinned by the H1 test); added an ADR line to promote `xthreat` cell-indexing to public API if it refactors (P1-12).
- **V4** (e2e referenced an undefined corpus): snippet now fits on a concrete disjoint corpus fixture (reuse the OBSO/frozen-xt held-out loader) (P1-9b).
- **Consent trail**: ADR now requires Jeffrey's **verbatim** 2026-06-08 reply + date + channel, and an explicit "composite final / presets provisional" statement (P1-12).
- **Verified-good (untouched):** H2 justification (source-confirmed: `_kernels.py:903` does the same `np.arange` rekey), B2 body routing, M1 guard, Option B math.

## Parallel review (d32, 2026-06-08) — incorporated

- **B1** (get_xc hard-raises without `vx`/`vy`/`team_in_possession`): fixed — DAS-valid `_frames_for` + reuse-existing-factory recommendation (P1-5).
- **B2** (`@nan_safe_enrichment` is marker-only): fixed — NaN-identifier routing implemented in the `compute_xt_gk` body; test asserts the routing (P1-5/P1-6); gate branch adds DAS cols defensively (P1-10).
- **B3** (Q1 `(1+p)` double-count + Q2 presets): **RESOLVED — Jeffrey 2026-06-08 chose Option B** (destination counted once; base origin-only, RAV owns z') + provisional presets approved. Composite updated to `T·(−xT★(z) + γ·PEV + RAV) + φ·DZV`; PEV keeps a separate `progress` term so it isn't zeroed by the origin-only base. Recorded in the ADR consent trail (P1-12).
- **H1** (circular grid-convention test): fixed — `_grid_value` reuses `xthreat._grid._get_cell_indexes` + a pin test vs `ExpectedThreat.rate` (P1-2).
- **H2** (xfns rekey vs `resolve_frame_ids_by_position`): kept with verified justification (sub-calls are action_id-keyed; rekey == the precedent's internal mechanism) — documented (P1-7).
- **H3** (no real-data / OOD deferred): added owner-gated e2e smoke + binding escalation to a geometry completion prior (P1-9b).
- **M1–M5**: unfitted-grid guard + test (M1); unlinked-rows warn + provenance (M2); `home_team_id` ADR sentence (M3); per-σ evidence in ADR (M4); hoisted `xT★(z′)` (M5).

## Self-review notes (author checklist — completed)

- **Spec coverage:** §3 module structure → P1-1/5/6/7/8; §4 metric → P1-2/3/5; §4.3 get_xc + [das] fail-loud → P1-5/6; §4.7 columns → P1-5; §5 params/presets → P1-1; §6 obligations (nan-safe, provenance, frame-id-by-position, id-dtype, atomic, fail-loud) → P1-6/7/9/10; §7 construct-invariant gates + leakage contract + oracle tests → P1-3/5/6; §8 C4/NOTICE/ADR/CLAUDE.md → P1-12; §9 phasing + convolution_sigma scan → P1-11 + Phase 2; §11 open Qs (Q1/Q2 cleared, Q3 temporal near-inert handled by `_possession_depth`+composite, Q4 turnover-location = RAV simplification documented in ADR, Q5 get_xc confirmed, Q7 = Phase-2 gate). All covered.
- **Type consistency:** `_OUTPUT_COLS` shared between `compute_xt_gk`, `add_xt_gk`, `xt_gk_xfns`. `XtGkParams` field names consistent across params/components/compute. `home_team_id` keyword on `add_xt_gk`/`xt_gk_xfns` (both surfaces) matches the `_axh`/`_build` gate adapters.
- **Open verification items flagged for the executor (not placeholders — runtime facts to confirm):** (a) `canonical_id_series` return type/iteration; (b) `get_xc` xC sanity on goal-kicks/throws (review #2.1 OOD — document, don't mask); (c) atomic features module's exact mirror idiom; (d) provenance-skip gate's chain shape; (e) ADR number reconciliation.
