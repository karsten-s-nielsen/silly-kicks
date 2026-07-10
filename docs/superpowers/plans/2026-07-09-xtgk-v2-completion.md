# xT-GK v2 Completion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete xT-GK v2 in one release — wire the make-or-break deep-zone gate loader (+ real zone-conditional terciles + gate-enforced relative effect), build `V_opp` (turnover cost) and `ρ` (retention) behind ports, assemble `compute_xt_gk_v2` with the four-term decomposition, and ship the owner-run validation suite; v1 frozen alongside.

**Architecture:** Hexagonal. `silly_kicks/xtgk/` gains three ports — `PossessionValue` (SP1, shipped), `RetentionModel` (SP3), `TurnoverCost` (SP2) — with their adapters, and a `_metric.py` assembler depending only on the ports (V/ρ/V_opp each swappable via injection, mirroring `compute_xt_gk`'s `xt=`/`completion=`). No `xthreat` edits; no v1 `tracking/_xt_gk.py` edits (byte-stability gated).

**Tech Stack:** Python, pandas, numpy, scikit-learn (fit-only; pure-numpy serve), pytest. Reuses `xthreat.value_iteration`, `spadl.add_possessions`, `tracking._gk_geometry.resolve_gk_geometry`, `tracking.features.receiver_zone_density`/`nearest_defender_distance`.

**Commit policy (owner decision 2026-07-09):** SINGLE feature / commit / PR on branch `pr-s109-xtgk-v2-completion` off `main` (no worktree). The task order below is the internal build sequence, **not** per-task commits — there is exactly ONE commit, in Task E5, after `/final-review`. Per-task "checkpoint" steps run the test suite instead of committing.

**Column naming (H1):** v2 decomposition columns are namespaced `xt_gk_v2_*` and MUST NOT reuse v1's frozen `xt_gk_pev/rav/dzv` (lakehouse-materialized, read by the GK-Analytics UI — Hyrum's Law).

**Phases:** A (gate loader + zone-conditional + relative-effect) ‖ B (V_opp) ‖ C (ρ) are independent; D (metric) needs A/B/C's ports; E (validation + release) needs D.

**Spec:** `docs/superpowers/specs/2026-07-09-xtgk-v2-completion-handoff.md` + `docs/superpowers/specs/2026-07-05-xtgk-v2-possession-value-design.md` (rev 4); ADR-036.

**Revision log — review round 1 (analysis session, 2026-07-09):** fixed 3 blocking + 3 high/medium + 4 minor. Blocking: E1 smoke used an all-one-class target (now a mixed shot/shotless cohort + out-of-sample split + all 4 baselines, Task E1); A4 never exercised the zone-conditional fallback (now a genuinely-deep-low-pressure fixture asserting `rung=="zone_conditional"`, Task A4); `compute_xt_gk_v2` broke under a zone-conditional `pl` and refit its own terciles (now reuses V's `pressure_levels` + threads zones, Task D1). High: `EmpiricalTurnoverValue` scan bounded to the post-turnover possession + time window (B4); metric requires frames-derived `retention_features` — no silent `frames=None` (D1); vectorized `_get_flat_indexes` in D1. Minor: `retains` truncated-window→NaN (C3/C6), v2/v1 column-disjointness guard (D2), `PressureLevels.apply` fitted-check-before-isna order preserved (A2).

---

## Task 0: Branch setup

**Files:** none (git only)

- [ ] **Step 1: Confirm clean tree and current branch**

Run: `git status --short && git branch --show-current`
Expected: on `main`; only the untracked specs/plan under `docs/superpowers/`.

- [ ] **Step 2: Create the feature branch**

```bash
git checkout -b pr-s109-xtgk-v2-completion
```

- [ ] **Step 3: Verify the test baseline is green before any change**

Run: `python -m pytest tests/xtgk/ -m "not e2e" -q`
Expected: PASS (SP1 suite green).

---

# Phase A — Gate loader + zone-conditional terciles + relative-effect

### Task A1: Gate-enforce the relative-effect floor (B2)

**Files:**
- Modify: `silly_kicks/xtgk/_diagnostics.py`
- Test: `tests/xtgk/test_diagnostics.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/xtgk/test_diagnostics.py`:

```python
from silly_kicks.xtgk._diagnostics import GateConfig, DeepZoneGateReport, run_deep_zone_gate


class _StubValue:
    """Minimal PossessionValue stub: constant per-tercile deep value + full support."""
    def __init__(self, per_level: dict[int, float], support: int = 100):
        self._v = per_level
        self._support = support

    def value(self, zone, p):
        return self._v[p]

    def support(self, p):
        import numpy as np
        return np.full((12, 16), self._support, dtype=int)


def test_relative_effect_floor_fails_a_trivial_gradient():
    # absolute effect passes (0.02 >= 0.005) but relative is tiny: 0.02 / mean(1.00,0.98)=~0.02 < 0.25
    mk = _StubValue({1: 1.00, 2: 0.99, 3: 0.98})
    cfg = GateConfig(effect_floor=0.005, relative_effect_floor=0.25, n_min=10,
                     min_occupied_cells=2, expected_direction="decreasing")
    report = run_deep_zone_gate(mk, mk, cfg)
    assert report.relative_effect < 0.25
    assert report.passed is False
    assert "relative" in report.stop_reason


def test_relative_effect_floor_passes_a_real_gradient():
    mk = _StubValue({1: 0.030, 2: 0.020, 3: 0.010})  # rel = 0.02/0.02 = 1.0 >= 0.25
    cfg = GateConfig(effect_floor=0.005, relative_effect_floor=0.25, n_min=10,
                     min_occupied_cells=2, expected_direction="decreasing")
    report = run_deep_zone_gate(mk, mk, cfg)
    assert report.relative_effect >= 0.25
    assert report.passed is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/xtgk/test_diagnostics.py::test_relative_effect_floor_fails_a_trivial_gradient -q`
Expected: FAIL (`GateConfig` has no `relative_effect_floor`; report has no `relative_effect`).

- [ ] **Step 3: Add the field + report attribute + enforcement**

In `silly_kicks/xtgk/_diagnostics.py`, extend `GateConfig` (insert `relative_effect_floor` with a back-compatible default of `0.0` so existing synthetic tests that omit it keep passing — `0.0` means "not enforced"):

```python
@dataclass(frozen=True)
class GateConfig:
    effect_floor: float
    n_min: int
    relative_effect_floor: float = 0.0
    min_occupied_cells: int = 2
    crosscheck_rel_tol: float = 0.5
    expected_direction: Direction = "either"
```

Extend `DeepZoneGateReport` with `relative_effect: float` (add as the last field with a default so both construction sites stay valid):

```python
@dataclass(frozen=True)
class DeepZoneGateReport:
    passed: bool
    effect_size: float
    observed_direction: str
    monotone_ok: bool
    crosscheck_agrees: bool
    n_occupied_cells: int
    stop_reason: str
    relative_effect: float = 0.0
```

In `run_deep_zone_gate`, after `effect = abs(v1 - v3)`, add the relative computation and fold it into `passed`/`reason`:

```python
    effect = abs(v1 - v3)
    rel_effect = abs(v1 - v3) / max(abs(0.5 * (v1 + v3)), 1e-9)
    rel_ok = rel_effect >= cfg.relative_effect_floor
    nonincreasing = v1 >= v2 >= v3
```

Update `passed` and `reason`:

```python
    passed = bool(effect >= cfg.effect_floor and rel_ok and monotone_ok and crosscheck)
    reason = (
        ""
        if passed
        else "; ".join(
            s
            for s, ok in [
                (f"effect {effect:.4f}<{cfg.effect_floor}", effect >= cfg.effect_floor),
                (f"relative {rel_effect:.3f}<{cfg.relative_effect_floor}", rel_ok),
                (f"direction {observed}!={cfg.expected_direction}/non-monotone", monotone_ok),
                ("cross-check divergent", crosscheck),
            ]
            if not ok
        )
    )
    return DeepZoneGateReport(passed, effect, observed, monotone_ok, crosscheck, len(occ), reason, rel_effect)
```

Also add `relative_effect=0.0` to the STOP early-return's `DeepZoneGateReport(...)` (it already passes positionally through `stop_reason`; append `0.0` as the final arg).

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/xtgk/test_diagnostics.py -q`
Expected: PASS (new tests + existing).

- [ ] **Step 5: Checkpoint**

Run: `python -m pytest tests/xtgk/ -m "not e2e" -q`
Expected: PASS.

---

### Task A2: Zone-band helper + zone-conditional `PressureLevels.fit`/`apply` (§1c)

**Files:**
- Modify: `silly_kicks/xtgk/_pressure_levels.py`
- Test: `tests/xtgk/test_pressure_levels.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/xtgk/test_pressure_levels.py`:

```python
import numpy as np
import pandas as pd
import pytest

from silly_kicks.xtgk._pressure_levels import PressureLevels, band_of_zone
from silly_kicks.xthreat._grid import N  # grid length (16)


def test_band_of_zone_deep_is_columns_0_and_1():
    # flat = (w-1-yj)*l + xi ; deep band = xi in {0,1}
    assert band_of_zone(0, N) == 0        # xi=0
    assert band_of_zone(1, N) == 0        # xi=1
    assert band_of_zone(2, N) == 1        # xi=2
    assert band_of_zone(N + 5, N) == 1    # xi=5 on the next row


def test_global_mode_unchanged_byte_for_byte():
    p = pd.Series(np.linspace(0.0, 1.0, 300))
    pl = PressureLevels(mode="global").fit(p)
    lv = pl.apply(p)
    assert set(np.unique(lv)) == {1, 2, 3}
    # counts ~ thirds
    assert abs((lv == 1).sum() - 100) <= 2


def test_zone_conditional_terciles_are_within_band():
    # deep band globally LOW pressure (0..0.2); rest band HIGH (0.6..1.0).
    # global terciles would push ALL deep actions into level 1; zone-conditional must give
    # each band its own ~1/3-1/3-1/3.
    deep_p = np.linspace(0.0, 0.2, 150)
    rest_p = np.linspace(0.6, 1.0, 150)
    pressure = pd.Series(np.concatenate([deep_p, rest_p]))
    zones = np.concatenate([np.zeros(150, dtype=int), np.full(150, 5, dtype=int)])  # deep vs rest
    pl = PressureLevels(mode="zone_conditional").fit(pressure, zones=zones)
    lv = pl.apply(pressure, zones=zones)
    deep_lv, rest_lv = lv[:150], lv[150:]
    for sub in (deep_lv, rest_lv):
        assert set(np.unique(sub)) == {1, 2, 3}
        assert abs((sub == 3).sum() - 50) <= 3  # ~1/3 within band


def test_zone_conditional_apply_requires_zones():
    pressure = pd.Series(np.linspace(0.0, 1.0, 30))
    zones = np.zeros(30, dtype=int)
    pl = PressureLevels(mode="zone_conditional").fit(pressure, zones=zones)
    with pytest.raises(ValueError, match="zones"):
        pl.apply(pressure)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/xtgk/test_pressure_levels.py::test_band_of_zone_deep_is_columns_0_and_1 -q`
Expected: FAIL (`band_of_zone` undefined).

- [ ] **Step 3: Implement zone-band mapping + zone-conditional fit/apply/occupancy**

Rewrite `silly_kicks/xtgk/_pressure_levels.py` (keep `coalesce_frame_present_null_pressure` unchanged; the global path stays byte-identical):

```python
"""Continuous pressure -> {1,2,3} tercile quantizer (ADR-036 §5, §1c).

fit() learns cutpoints on the fit cohort; apply() maps new actions; cutpoints persist with the
surface. mode="global" is the default and byte-identical to SP1. mode="zone_conditional" learns
per-BAND terciles (deep band = grid columns xi in {0,1} vs the rest) so a systematically
low-pressure deep zone still populates all three deep terciles (the M3 fix / gate fallback rung).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from silly_kicks.xthreat._grid import N

Mode = Literal["global", "zone_conditional"]

_DEEP_MAX_XI = 1  # deep band = grid columns xi in {0,1} (matches _diagnostics.DEEP_ZONE_CELLS)


def band_of_zone(zone: int, l: int = N) -> int:
    """0 for the deep band (xi in {0,1}), 1 otherwise. Flat index layout: xi = zone % l."""
    return 0 if (int(zone) % l) <= _DEEP_MAX_XI else 1


def _bands(zones: np.ndarray, l: int) -> np.ndarray:
    return np.where((np.asarray(zones).astype(int) % l) <= _DEEP_MAX_XI, 0, 1)


def coalesce_frame_present_null_pressure(
    pressure: pd.Series[float], frame_present: pd.Series[bool]
) -> pd.Series[float]:
    """Frame-aware null-pressure rule (ADR-036 §5, G8). Pure: returns a new Series."""
    out = pressure.copy()
    fill_mask = frame_present.to_numpy(dtype=bool) & out.isna().to_numpy()
    out[fill_mask] = 0.0
    return out


class PressureLevels:
    def __init__(self, *, mode: Mode = "global", l: int = N) -> None:
        self.mode: Mode = mode
        self.l = l
        self.cutpoints: tuple[float, float] | None = None                  # global
        self.band_cutpoints: dict[int, tuple[float, float]] | None = None  # zone_conditional

    def fit(self, pressure: pd.Series[float], *, zones: np.ndarray | None = None) -> PressureLevels:
        p_all = pressure.to_numpy(dtype=float)
        valid = ~np.isnan(p_all)
        if not valid.any():
            raise ValueError("cannot fit pressure terciles on empty/all-NaN pressure")
        if self.mode == "global":
            lo, hi = np.quantile(p_all[valid], [1 / 3, 2 / 3])
            self.cutpoints = (float(lo), float(hi))
            return self
        if zones is None:
            raise ValueError("zone_conditional fit requires zones= (each action's flat grid cell)")
        bands = _bands(zones, self.l)
        bc: dict[int, tuple[float, float]] = {}
        for b in (0, 1):
            sel = valid & (bands == b)
            if not sel.any():
                raise ValueError(f"zone band {b} has no non-NaN pressure at fit (check deep-zone coverage)")
            lo, hi = np.quantile(p_all[sel], [1 / 3, 2 / 3])
            bc[b] = (float(lo), float(hi))
        self.band_cutpoints = bc
        return self

    @classmethod
    def from_cutpoints(cls, cutpoints: tuple[float, float], *, mode: Mode = "global") -> PressureLevels:
        obj = cls(mode=mode)
        obj.cutpoints = (float(cutpoints[0]), float(cutpoints[1]))
        return obj

    @classmethod
    def from_band_cutpoints(cls, band_cutpoints: dict, *, l: int = N) -> PressureLevels:
        obj = cls(mode="zone_conditional", l=l)
        obj.band_cutpoints = {int(k): (float(v[0]), float(v[1])) for k, v in band_cutpoints.items()}
        return obj

    def apply(self, pressure: pd.Series[float], *, zones: np.ndarray | None = None) -> np.ndarray:
        # fitted-check BEFORE the isna-check (preserves SP1 apply ordering)
        if self.mode == "global":
            if self.cutpoints is None:
                raise ValueError("PressureLevels not fitted")
        elif self.band_cutpoints is None:
            raise ValueError("PressureLevels not fitted")
        if pressure.isna().any():
            raise ValueError("missing pressure value(s); never default a level (ADR-036 §5)")
        p = pressure.to_numpy(dtype=float)
        if self.mode == "global":
            lo, hi = self.cutpoints
            return np.where(p <= lo, 1, np.where(p <= hi, 2, 3)).astype(int)
        if zones is None:
            raise ValueError("zone_conditional apply requires zones= (each action's flat grid cell)")
        bands = _bands(zones, self.l)
        los = np.array([self.band_cutpoints[int(b)][0] for b in bands])
        his = np.array([self.band_cutpoints[int(b)][1] for b in bands])
        return np.where(p <= los, 1, np.where(p <= his, 2, 3)).astype(int)

    def occupancy(self, pressure: pd.Series[float], *, zones: np.ndarray | None = None) -> dict[int, int]:
        lv = self.apply(pressure, zones=zones)
        return {k: int((lv == k).sum()) for k in (1, 2, 3)}

    def to_meta(self) -> dict:
        """Serialize state. Global form is byte-identical to SP1 (`{"cutpoints": [lo, hi]}`)."""
        if self.mode == "global":
            if self.cutpoints is None:
                raise ValueError("cannot serialize an unfitted PressureLevels")
            return {"cutpoints": list(self.cutpoints)}
        if self.band_cutpoints is None:
            raise ValueError("cannot serialize an unfitted PressureLevels")
        return {
            "pressure_mode": "zone_conditional",
            "band_cutpoints": {str(b): list(c) for b, c in self.band_cutpoints.items()},
        }

    @classmethod
    def from_meta(cls, meta: dict, *, l: int = N) -> PressureLevels:
        """Reconstruct. Absent `pressure_mode` => global (back-compat with SP1 artifacts)."""
        if meta.get("pressure_mode") == "zone_conditional":
            return cls.from_band_cutpoints({int(b): tuple(c) for b, c in meta["band_cutpoints"].items()}, l=l)
        cut = meta["cutpoints"]
        return cls.from_cutpoints((float(cut[0]), float(cut[1])))
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/xtgk/test_pressure_levels.py -q`
Expected: PASS.

- [ ] **Step 5: Checkpoint (global byte-identity)**

Run: `python -m pytest tests/xtgk/ -m "not e2e" -q`
Expected: PASS (SP1 fixtures use `mode="global"` — unchanged).

---

### Task A3: Thread zones through the estimators + serialize `mode`/`band_cutpoints`

**Files:**
- Modify: `silly_kicks/xtgk/_markov.py`
- Modify: `silly_kicks/xtgk/_empirical.py`
- Test: `tests/xtgk/test_markov.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/xtgk/test_markov.py`:

```python
import numpy as np

from silly_kicks.xtgk import MarkovPossessionValue, PressureLevels
from tests.xtgk.conftest import three_band_cohort


def test_markov_fits_under_zone_conditional_and_roundtrips(tmp_path):
    actions = three_band_cohort()
    pl = PressureLevels(mode="zone_conditional")
    # zone-conditional pl must be fit with zones; the estimator derives zones internally, but the
    # externally-fit pl here needs them too — use the same grid binning.
    from silly_kicks.xthreat._grid import M, N, _get_flat_indexes
    zones = _get_flat_indexes(actions.start_x, actions.start_y, N, M).to_numpy()
    pl.fit(actions["pressure"], zones=zones)
    mk = MarkovPossessionValue().fit(actions, xg_column="xg", pressure_column="pressure", pressure_levels=pl)
    v_lo = mk.value(0, 1)  # deep cell, low tercile
    assert np.isfinite(v_lo)

    mk.save(tmp_path / "surf")
    reloaded = MarkovPossessionValue.load(tmp_path / "surf")
    assert reloaded.pressure_levels.mode == "zone_conditional"
    assert np.isclose(reloaded.value(0, 1), v_lo)


def test_markov_global_metadata_byte_identical(tmp_path):
    actions = three_band_cohort()
    mk = MarkovPossessionValue().fit(actions, xg_column="xg", pressure_column="pressure")
    mk.save(tmp_path / "surf")
    meta = (tmp_path / "surf" / "metadata.json").read_text()
    assert "pressure_mode" not in meta   # global form must NOT gain the zone-conditional key
    assert '"cutpoints"' in meta
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/xtgk/test_markov.py::test_markov_fits_under_zone_conditional_and_roundtrips -q`
Expected: FAIL (`fit` calls `pl.apply(...)` without zones → ValueError for zone_conditional; `load` uses `from_cutpoints`, not `from_meta`).

- [ ] **Step 3: Implement zone-threading + `to_meta`/`from_meta` serialization**

In `silly_kicks/xtgk/_markov.py`, change the `fit` body where it applies levels (currently `levels = pl.apply(actions[pressure_column])`):

```python
        pl = pressure_levels or PressureLevels().fit(actions[pressure_column])
        zones = None
        if pl.mode == "zone_conditional":
            from silly_kicks.xthreat._grid import _get_flat_indexes

            zones = _get_flat_indexes(actions["start_x"], actions["start_y"], self.l, self.w).to_numpy()
        levels = pl.apply(actions[pressure_column], zones=zones)
```

In the provenance dict, keep `"cutpoints": pl.cutpoints` (informational; `None` under zone_conditional is fine).

Replace the `save` cutpoints handling. Change:

```python
        pl = cast("PressureLevels", self.pressure_levels)  # _check() guarantees fitted
        cut = cast("tuple[float, float]", pl.cutpoints)
        meta = dict(self.provenance)
        meta["cutpoints"] = list(cut)
        save_surface(directory, surfaces=self._surfaces, support=self._support, metadata=meta)
```

to:

```python
        pl = cast("PressureLevels", self.pressure_levels)  # _check() guarantees fitted
        meta = dict(self.provenance)
        meta.update(pl.to_meta())  # global: {"cutpoints":[lo,hi]} (byte-identical); zone_cond adds band_cutpoints
        save_surface(directory, surfaces=self._surfaces, support=self._support, metadata=meta)
```

Replace the `load` pressure-level reconstruction. Change:

```python
        cut = meta["cutpoints"]
        obj.pressure_levels = PressureLevels.from_cutpoints((float(cut[0]), float(cut[1])))
```

to:

```python
        obj.pressure_levels = PressureLevels.from_meta(meta, l=int(l))
```

In `silly_kicks/xtgk/_empirical.py`, change the `fit` apply site (currently `a["_p_level"] = pl.apply(a[pressure_column])`):

```python
        zones = None
        if pl.mode == "zone_conditional":
            zones = _get_flat_indexes(a.start_x, a.start_y, self.l, self.w).to_numpy()
        a["_p_level"] = pl.apply(a[pressure_column], zones=zones)
```

(`_get_flat_indexes` is already imported in `_empirical.py`.)

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/xtgk/test_markov.py -q`
Expected: PASS.

- [ ] **Step 5: Checkpoint**

Run: `python -m pytest tests/xtgk/ -m "not e2e" -q`
Expected: PASS.

---

### Task A4: `_occupied` STOP + three-rung ladder helper

**Files:**
- Modify: `silly_kicks/xtgk/_diagnostics.py`
- Test: `tests/xtgk/test_diagnostics.py`

- [ ] **Step 1a: Add a genuinely-deep-low-pressure fixture to `tests/xtgk/conftest.py`**

The existing `three_band_cohort` sets one pressure per possession (deep goal-kick shares its chain's
band), so global terciles already populate the deep high tercile and the fallback never fires. This
fixture makes the deep band globally LOW while the outfield carries the HIGH-pressure mass, so global
terciles starve the deep high tercile (STOP) but zone-conditional deep-band terciles populate it:

```python
def deep_low_rest_high_cohort(n_per_cell=20) -> pd.DataFrame:
    """Deep zone globally LOW pressure (goal-kicks ~0.02..0.18); outfield build-up/shots HIGH
    (~0.6..0.95). Under GLOBAL terciles the deep high tercile is starved -> the ladder STOPs on
    rung 1; under ZONE-CONDITIONAL terciles the deep band's own 0.02..0.18 spread splits into thirds
    -> the two deep cells (deep_y in {24,44}, both xi=0) populate all three terciles -> rung 2 fires."""
    rows: list[dict] = []
    pid = 0
    for dy in (24.0, 44.0):
        for k in range(n_per_cell):
            base = 1000 * pid
            low_p = 0.02 + 0.16 * (k / max(n_per_cell - 1, 1))   # deep-band spread 0.02..0.18
            hi_p = 0.6 + 0.35 * ((k % 3) / 2)                     # outfield 0.60/0.775/0.95
            xg = 0.4 - 0.3 * (k / max(n_per_cell - 1, 1))         # xg falls as deep pressure rises
            rows += [
                _row(base + 0, GOALKICK, SUCCESS, 3.0, dy, 30.0, 34.0,
                     possession_id=pid, pressure=low_p, time_seconds=base + 0.0),
                _row(base + 1, PASS, SUCCESS, 30.0, 34.0, 55.0, 34.0,
                     possession_id=pid, pressure=hi_p, time_seconds=base + 1.0),
                _row(base + 2, PASS, SUCCESS, 55.0, 34.0, 80.0, 34.0,
                     possession_id=pid, pressure=hi_p, time_seconds=base + 2.0),
                _row(base + 3, PASS, SUCCESS, 80.0, 34.0, 100.0, 34.0,
                     possession_id=pid, pressure=hi_p, time_seconds=base + 3.0),
                _row(base + 4, SHOT, FAIL, 100.0, 34.0, 105.0, 34.0,
                     possession_id=pid, pressure=hi_p, xg=xg, time_seconds=base + 4.0),
            ]
            pid += 1
    return make_cohort(rows)
```

- [ ] **Step 1b: Write the failing test (asserts the fallback actually fires)**

Add to `tests/xtgk/test_diagnostics.py`:

```python
from silly_kicks.xtgk._diagnostics import run_gate_with_ladder, _fit_pair, _occupied, GateConfig
from tests.xtgk.conftest import deep_low_rest_high_cohort


def test_ladder_falls_to_zone_conditional_when_global_starves_deep_high_tercile():
    actions = deep_low_rest_high_cohort()
    cfg = GateConfig(effect_floor=0.0, relative_effect_floor=0.0, n_min=5,
                     min_occupied_cells=2, expected_direction="either")
    # rung 1 (global) alone starves the deep high tercile -> < 2 occupied deep cells (would STOP)
    _pl_g, mk_g, _emp_g = _fit_pair(actions, xg_column="xg", pressure_column="pressure", mode="global")
    assert len(_occupied(mk_g, cfg)) < cfg.min_occupied_cells
    # the ladder therefore falls through to zone-conditional and RUNS there
    result = run_gate_with_ladder(actions, xg_column="xg", pressure_column="pressure", cfg=cfg)
    assert result["rung"] == "zone_conditional"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/xtgk/test_diagnostics.py::test_ladder_falls_to_zone_conditional_when_global_starves_deep_high_tercile -q`
Expected: FAIL (`run_gate_with_ladder`/`_fit_pair` undefined).

- [ ] **Step 3: Implement the ladder**

Add to `silly_kicks/xtgk/_diagnostics.py`:

```python
def _fit_pair(actions, *, xg_column, pressure_column, mode):
    """Fit (Markov, Empirical) sharing one PressureLevels in the given mode."""
    from silly_kicks.xtgk._pressure_levels import PressureLevels
    from silly_kicks.xthreat._grid import M, N, _get_flat_indexes

    pl = PressureLevels(mode=mode)
    if mode == "zone_conditional":
        zones = _get_flat_indexes(actions["start_x"], actions["start_y"], N, M).to_numpy()
        pl.fit(actions[pressure_column], zones=zones)
    else:
        pl.fit(actions[pressure_column])
    mk = MarkovPossessionValue().fit(actions, xg_column=xg_column, pressure_column=pressure_column, pressure_levels=pl)
    emp = EmpiricalPossessionValue().fit(actions, xg_column=xg_column, pressure_column=pressure_column, pressure_levels=pl)
    return pl, mk, emp


def run_gate_with_ladder(actions, *, xg_column: str, pressure_column: str, cfg: GateConfig) -> dict:
    """Pre-registered three-rung ladder (§1c): global -> zone_conditional -> STOP.

    Rung 1 = global terciles. If <min_occupied_cells deep cells clear n_min in all three terciles,
    rung 2 refits zone-conditional terciles (deep-relative). If still short, STOP (inconclusive) --
    do NOT lower n_min. The winning rung is reported so a rung-2 pass is read as 'deep-relative'."""
    for rung in ("global", "zone_conditional"):
        pl, mk, emp = _fit_pair(actions, xg_column=xg_column, pressure_column=pressure_column, mode=rung)
        if len(_occupied(mk, cfg)) >= cfg.min_occupied_cells:
            report = run_deep_zone_gate(mk, emp, cfg)
            return {"rung": rung, "report": report, "cutpoints": _cutpoints_of(pl)}
    report = run_deep_zone_gate(mk, emp, cfg)  # zone_conditional mk/emp, still short -> STOP verdict
    return {"rung": "stop", "report": report, "cutpoints": _cutpoints_of(pl)}


def _cutpoints_of(pl) -> dict:
    return pl.to_meta()
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/xtgk/test_diagnostics.py -q`
Expected: PASS.

- [ ] **Step 5: Checkpoint**

Run: `python -m pytest tests/xtgk/ -m "not e2e" -q`
Expected: PASS.

---

### Task A5: Both-orientations gate wrapper (mirror_y equivariance / mirror_x rejection)

**Files:**
- Modify: `silly_kicks/xtgk/_diagnostics.py`
- Test: `tests/xtgk/test_diagnostics.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/xtgk/test_diagnostics.py`:

```python
from silly_kicks.xtgk._diagnostics import run_gate_both_orientations, GateConfig
from tests.xtgk.conftest import three_band_cohort, mirror_x, mirror_y


def test_both_orientations_equivariant_under_mirror_y_and_flags_mirror_x():
    actions = three_band_cohort()
    cfg = GateConfig(effect_floor=0.0, relative_effect_floor=0.0, n_min=5,
                     min_occupied_cells=2, expected_direction="either")
    out = run_gate_both_orientations(actions, xg_column="xg", pressure_column="pressure", cfg=cfg)
    # mirror_y (attack direction preserved) must reproduce the fit effect within tolerance
    assert abs(out["fit"]["report"].effect_size - out["mirror_y"]["report"].effect_size) < 1e-6
    # mirror_x reverses attack direction -> the fit validator rejects it (not attack-LTR)
    assert out["mirror_x"]["orientation_rejected"] is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/xtgk/test_diagnostics.py::test_both_orientations_equivariant_under_mirror_y_and_flags_mirror_x -q`
Expected: FAIL (`run_gate_both_orientations` undefined).

- [ ] **Step 3: Implement the both-orientations wrapper**

Add to `silly_kicks/xtgk/_diagnostics.py`:

```python
def run_gate_both_orientations(actions, *, xg_column: str, pressure_column: str, cfg: GateConfig) -> dict:
    """Run the ladder on the fit orientation + mirror_y (equivariance) + mirror_x (rejection check).

    mirror_y preserves attack direction (still attack-LTR) -> the effect must reproduce. mirror_x
    reverses it -> the fit input validator must reject it (deep/final-third inverted, §M4)."""
    import silly_kicks.spadl.config as spadlconfig
    from silly_kicks.xtgk._validate import validate_possession_value_input

    def _mirror_y(a):
        out = a.copy()
        out["start_y"] = spadlconfig.field_width - a["start_y"]
        out["end_y"] = spadlconfig.field_width - a["end_y"]
        return out

    def _mirror_x(a):
        out = a.copy()
        out["start_x"] = spadlconfig.field_length - a["start_x"]
        out["end_x"] = spadlconfig.field_length - a["end_x"]
        return out

    result = {"fit": run_gate_with_ladder(actions, xg_column=xg_column, pressure_column=pressure_column, cfg=cfg)}
    result["mirror_y"] = run_gate_with_ladder(
        _mirror_y(actions), xg_column=xg_column, pressure_column=pressure_column, cfg=cfg
    )
    mx = _mirror_x(actions)
    diag = validate_possession_value_input(mx, xg_column=xg_column, pressure_column=pressure_column)
    result["mirror_x"] = {"orientation_rejected": not diag.ok, "problems": list(diag.problems)}
    return result
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/xtgk/test_diagnostics.py -q`
Expected: PASS.

- [ ] **Step 5: Export the new gate entry points**

In `silly_kicks/xtgk/__init__.py`, add to the `_diagnostics` import block and `__all__`: `run_gate_with_ladder`, `run_gate_both_orientations`.

- [ ] **Step 6: Checkpoint**

Run: `python -m pytest tests/xtgk/ -m "not e2e" -q`
Expected: PASS.

---

### Task A6: Wire the owner-run loader in `validate_xtgk_possession_value.py`

**Files:**
- Modify: `scripts/validate_xtgk_possession_value.py`
- Test: none (owner-run, `NotImplementedError` replaced by a documented loader seam; structural dry-run covered by `--force-unlocked`)

- [ ] **Step 1: Update the locked GateConfig + docstring + cohort scope**

Replace `_GATE_CONFIG_PENDING` with the locked values and correct the cohort docstring (RM include-provisional, not dropped):

```python
# --- LOCKED GATE NUMBERS (Q4, owner-set 2026-07-09; Eyestone to confirm) --------------------
_GATE_CONFIG_LOCKED = GateConfig(
    effect_floor=0.005,
    relative_effect_floor=0.25,
    n_min=30,
    min_occupied_cells=2,
    crosscheck_rel_tol=0.5,
    expected_direction="decreasing",
)
```

Update `_gate_is_locked` references and `main()` to use `_GATE_CONFIG_LOCKED`. Change the module docstring cohort block from "SkillCorner (RM) ... DROPPED" to: "SkillCorner (RM): `ood_flag=True` (uncertified) → INCLUDE as a PROVISIONAL second read (owner decision 2026-07-09); its verdict is reported separately and tagged provisional. WC2022 (gradientsports) is the authorising verdict."

- [ ] **Step 2: Replace `NotImplementedError` with the real loader seam**

Replace the `main()` body after the lock check with the loader + both-orientations ladder over both cohorts. The pining/Databricks data access is the owner-run's responsibility; the script names the required inputs and composes the gate:

```python
    from silly_kicks.xtgk import run_gate_both_orientations
    from silly_kicks.xtgk._diagnostics import ood_rate_by_source, frame_present_null_pressure_count
    import json

    # OWNER-RUN DATA ACCESS (pining / Databricks). Each cohort loader must return attack-LTR SPADL
    # actions carrying: the fct_shot_xg xg column (joined on (match_key, action_id)), ood_flag,
    # xg_ci_low/high, the pressure column (_PRESSURE_COLUMN), and _FRAME_PRESENT_COLUMN.
    from scripts._loader_pining import load_xtgk_cohort  # owner-run loader (returns actions, shot_xg)

    cohorts = {
        "wc2022": {"data_source": "gradientsports", "authorising": True},
        "rm": {"data_source": "skillcorner", "authorising": False},  # provisional (100% OOD)
    }
    report: dict = {}
    for name, spec in cohorts.items():
        actions, shot_xg = load_xtgk_cohort(spec["data_source"])
        actions = prepare_cohort(actions, pressure_column=_PRESSURE_COLUMN, frame_present_column=_FRAME_PRESENT_COLUMN)
        prov = reward_provenance_summary(shot_xg, ood_column=_OOD_COLUMN, ci_columns=_CI_COLUMNS)
        gate = run_gate_both_orientations(
            actions, xg_column=_XG_COLUMN, pressure_column=_PRESSURE_COLUMN, cfg=_GATE_CONFIG_LOCKED
        )
        report[name] = {
            "authorising": spec["authorising"],
            "reward_provenance": prov,
            "ood_rate_by_source": ood_rate_by_source(shot_xg, ood_col=_OOD_COLUMN),
            "unpressured_restart_count": frame_present_null_pressure_count(
                actions, pressure_col=_PRESSURE_COLUMN, frame_present_col=_FRAME_PRESENT_COLUMN
            ),
            "fit_rung": gate["fit"]["rung"],
            "fit_gate": asdict(gate["fit"]["report"]),
            "mirror_y_gate": asdict(gate["mirror_y"]["report"]),
            "mirror_x_rejected": gate["mirror_x"]["orientation_rejected"],
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote gate report -> {args.out}")
    return 0
```

Note in a comment that `scripts/_loader_pining.load_xtgk_cohort` is the owner-run loader boundary (pining #1, Databricks read-only) and must emit the columns named above; it is intentionally not implemented in CI.

- [ ] **Step 3: Structural dry-run**

Run: `python scripts/validate_xtgk_possession_value.py --force-unlocked` (expect it to reach the loader import and fail there only if `_loader_pining.load_xtgk_cohort` is absent — that is the owner-run seam). With the lock in place, `python scripts/validate_xtgk_possession_value.py` must NOT print the BLOCKED message.

- [ ] **Step 4: Checkpoint**

Run: `python -m pytest tests/xtgk/ -m "not e2e" -q && ruff check scripts/validate_xtgk_possession_value.py`
Expected: PASS + clean.

---

# Phase B — SP2: `V_opp` turnover cost

### Task B1: `mirror_zone` on the flat grid index

**Files:**
- Modify: `silly_kicks/xtgk/_possession_value.py`
- Test: `tests/xtgk/test_turnover.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/xtgk/test_turnover.py`:

```python
import numpy as np

from silly_kicks.xtgk._possession_value import mirror_zone, zone_of
from silly_kicks.xthreat._grid import M, N  # M=12 (w), N=16 (l)


def test_mirror_zone_is_an_involution():
    for z in range(M * N):
        assert mirror_zone(mirror_zone(z, N, M), N, M) == z


def test_mirror_zone_reflects_both_axes():
    # deep-left-bottom cell -> attacking-right-top cell (180 deg point reflection)
    z_deep = zone_of(3.0, 4.0, N, M)      # xi~0, low y
    z_far = mirror_zone(z_deep, N, M)
    xi, yj = z_far % N, z_far // N
    assert xi == N - 1 - (z_deep % N)     # column reversed
    assert yj == M - 1 - (z_deep // N)    # row reversed
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/xtgk/test_turnover.py::test_mirror_zone_is_an_involution -q`
Expected: FAIL (`mirror_zone` undefined).

- [ ] **Step 3: Implement `mirror_zone`**

Add to `silly_kicks/xtgk/_possession_value.py` (after `zone_of`):

```python
def mirror_zone(zone: int, l: int = N, w: int = M) -> int:
    """180-degree point reflection of a flat grid index (column reversal xi->l-1-xi AND row
    reversal yj->w-1-yj). Maps a losing team's origin zone (attack-LTR) to the winning team's
    zone in its OWN attack-LTR frame -- the V_opp mirror (ADR-036 §Part 2)."""
    z = int(zone)
    xi, yj = z % l, z // l
    return (w - 1 - yj) * l + (l - 1 - xi)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/xtgk/test_turnover.py -q`
Expected: PASS.

- [ ] **Step 5: Checkpoint**

Run: `python -m pytest tests/xtgk/ -m "not e2e" -q`
Expected: PASS.

---

### Task B2: `TurnoverCost` port + `MirroredTurnoverCost` adapter

**Files:**
- Create: `silly_kicks/xtgk/_turnover.py`
- Test: `tests/xtgk/test_turnover.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/xtgk/test_turnover.py`:

```python
from silly_kicks.xtgk._turnover import MirroredTurnoverCost, TurnoverCost
from silly_kicks.xtgk import MarkovPossessionValue, PressureLevels
from silly_kicks.xtgk._possession_value import mirror_zone
from tests.xtgk.conftest import three_band_cohort


def _fit_v():
    actions = three_band_cohort()
    pl = PressureLevels().fit(actions["pressure"])
    return MarkovPossessionValue().fit(actions, xg_column="xg", pressure_column="pressure", pressure_levels=pl)


def test_mirrored_turnover_equals_v_at_the_mirror_zone():
    v = _fit_v()
    tc = MirroredTurnoverCost(v)
    assert isinstance(tc, TurnoverCost)
    for z in (0, 1, 20, 100):
        assert tc.value(z, 1) == v.value(mirror_zone(z), 1)


def test_pressure_transfer_policy_is_injectable():
    v = _fit_v()
    tc = MirroredTurnoverCost(v, pressure_policy=lambda p: 1)  # opponent always low pressure
    assert tc.value(0, 3) == v.value(mirror_zone(0), 1)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/xtgk/test_turnover.py::test_mirrored_turnover_equals_v_at_the_mirror_zone -q`
Expected: FAIL (`_turnover` module missing).

- [ ] **Step 3: Implement the port + adapter**

Create `silly_kicks/xtgk/_turnover.py`:

```python
"""TurnoverCost port + MirroredTurnoverCost adapter (ADR-036 §Part 2).

V(z,p) is team-agnostic (pooled attack-LTR), so the opponent's threat after winning the ball at
zone z is V at the 180-degree mirror zone. Zero new fitting -- wraps an already-fit PossessionValue.
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from silly_kicks.xtgk._possession_value import M, N, PossessionValue, PressureLevel, mirror_zone


@runtime_checkable
class TurnoverCost(Protocol):
    def value(self, zone: int, p: PressureLevel) -> float: ...
    def surface(self, p: PressureLevel) -> npt.NDArray[np.float64]: ...
    def support(self, p: PressureLevel) -> npt.NDArray[np.int_]: ...


class MirroredTurnoverCost:
    """E[opp threat | turnover at (zone, p)] = V(mirror_zone(zone), policy(p)). Zero new fitting."""

    def __init__(
        self,
        possession_value: PossessionValue,
        *,
        pressure_policy: Callable[[PressureLevel], PressureLevel] | None = None,
        l: int = N,
        w: int = M,
    ) -> None:
        self._v = possession_value
        self._policy = pressure_policy or (lambda p: p)  # default p_opp = p
        self.l, self.w = l, w

    def value(self, zone: int, p: PressureLevel) -> float:
        return float(self._v.value(mirror_zone(zone, self.l, self.w), self._policy(p)))

    def surface(self, p: PressureLevel) -> npt.NDArray[np.float64]:
        # point-reflect the whole V(policy(p)) surface (row + column reversal)
        base = np.asarray(self._v.surface(self._policy(p)))
        return base[::-1, ::-1].copy()

    def support(self, p: PressureLevel) -> npt.NDArray[np.int_]:
        # support of the mirrored cell = mirrored V-support; sparsity is load-bearing (expose it)
        base = np.asarray(self._v.support(self._policy(p)))
        return base[::-1, ::-1].copy()
```

Note: `_possession_value.py` currently imports `M, N` from `xthreat._grid`; re-export them so `_turnover` can import from `_possession_value`. Add to the top of `_possession_value.py`: `from silly_kicks.xthreat._grid import M, N, _get_flat_indexes` (M, N already available — ensure they are named exports by referencing them, no code change needed if already imported; if only `_get_flat_indexes` was imported, extend the import to include `M, N`).

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/xtgk/test_turnover.py -q`
Expected: PASS.

- [ ] **Step 5: Checkpoint**

Run: `python -m pytest tests/xtgk/ -m "not e2e" -q`
Expected: PASS.

---

### Task B3: `_is_turnover` helper

**Files:**
- Modify: `silly_kicks/xtgk/_moves.py`
- Test: `tests/xtgk/test_moves.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/xtgk/test_moves.py`:

```python
import numpy as np

from silly_kicks.xtgk._moves import _is_turnover, MOVE_TYPE_IDS
import silly_kicks.spadl.config as spadlconfig


def test_is_turnover_is_failed_move_only():
    import pandas as pd
    SUCCESS = spadlconfig.result_id["success"]
    FAIL = spadlconfig.result_id["fail"]
    PASS = spadlconfig.actiontype_id["pass"]
    SHOT = spadlconfig.actiontype_id["shot"]
    df = pd.DataFrame({
        "type_id": [PASS, PASS, SHOT],
        "result_id": [SUCCESS, FAIL, FAIL],
    })
    out = _is_turnover(df)
    assert list(out) == [False, True, False]  # failed pass=turnover; failed shot is NOT a move-set turnover
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/xtgk/test_moves.py::test_is_turnover_is_failed_move_only -q`
Expected: FAIL (`_is_turnover` undefined).

- [ ] **Step 3: Implement**

Add to `silly_kicks/xtgk/_moves.py`:

```python
def _is_turnover(actions: pd.DataFrame) -> npt.NDArray[np.bool_]:
    """A move-set action (pass/dribble/cross/goalkick/throw_in) that did NOT succeed.
    Single-predicate house pattern (cf. vaep.labels._is_owngoal)."""
    is_move = actions["type_id"].isin(MOVE_TYPE_IDS).to_numpy()
    failed = (actions["result_id"] != _SUCCESS).to_numpy()
    return is_move & failed
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/xtgk/test_moves.py -q`
Expected: PASS.

- [ ] **Step 5: Checkpoint**

Run: `python -m pytest tests/xtgk/ -m "not e2e" -q`
Expected: PASS.

---

### Task B4: `EmpiricalTurnoverValue` cross-check

**Files:**
- Modify: `silly_kicks/xtgk/_turnover.py`
- Test: `tests/xtgk/test_turnover.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/xtgk/test_turnover.py`:

```python
import numpy as np
import pandas as pd

from silly_kicks.xtgk._turnover import EmpiricalTurnoverValue
import silly_kicks.spadl.config as spadlconfig


def test_empirical_turnover_credits_the_opponents_post_turnover_shot():
    PASS = spadlconfig.actiontype_id["pass"]
    SHOT = spadlconfig.actiontype_id["shot"]
    SUCCESS = spadlconfig.result_id["success"]
    FAIL = spadlconfig.result_id["fail"]
    # team 10 loses the ball deep (failed pass), team 20 wins and shoots (xg 0.4)
    rows = [
        dict(game_id=1, period_id=1, action_id=0, time_seconds=0.0, team_id=10, player_id=1,
             type_id=PASS, result_id=FAIL, bodypart_id=0, start_x=5.0, start_y=34.0,
             end_x=20.0, end_y=34.0, possession_id=0, xg=np.nan, pressure=0.1),
        dict(game_id=1, period_id=1, action_id=1, time_seconds=1.0, team_id=20, player_id=2,
             type_id=PASS, result_id=SUCCESS, bodypart_id=0, start_x=85.0, start_y=34.0,
             end_x=100.0, end_y=34.0, possession_id=1, xg=np.nan, pressure=0.1),
        dict(game_id=1, period_id=1, action_id=2, time_seconds=2.0, team_id=20, player_id=2,
             type_id=SHOT, result_id=FAIL, bodypart_id=0, start_x=100.0, start_y=34.0,
             end_x=105.0, end_y=34.0, possession_id=1, xg=0.4, pressure=0.1),
    ]
    actions = pd.DataFrame(rows)
    etv = EmpiricalTurnoverValue().fit(actions, xg_column="xg", pressure_column="pressure")
    from silly_kicks.xtgk._possession_value import zone_of
    z_loss = zone_of(5.0, 34.0)
    assert etv.value(z_loss, 1) > 0.0  # opponent's post-turnover shot xg credited to the loss zone
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/xtgk/test_turnover.py::test_empirical_turnover_credits_the_opponents_post_turnover_shot -q`
Expected: FAIL (`EmpiricalTurnoverValue` undefined).

- [ ] **Step 3: Implement (reverse-scan clone retargeted to opponent post-turnover shots)**

Add to `silly_kicks/xtgk/_turnover.py`:

```python
import pandas as pd

from silly_kicks.spadl.utils import add_possessions
from silly_kicks.xtgk._moves import _is_turnover
from silly_kicks.xtgk._possession_value import PressureLevel as _PL
from silly_kicks.xthreat._grid import _get_flat_indexes


class EmpiricalTurnoverValue:
    """Model-free cross-check for the mirror assumption (ADR-036 §Part 2.5). NOT shipped.

    For each turnover action, credit the xG of the FIRST shot the OPPONENT takes in the
    possession(s) after the turnover, binned to the loss zone/tercile. More sparse than V --
    apply the support gate before trusting a cell."""

    def __init__(self, *, l: int = N, w: int = M, window_seconds: float = 10.0) -> None:
        self.l, self.w = l, w
        self.window_seconds = window_seconds
        self._surfaces: dict[int, np.ndarray] = {}
        self._support: dict[int, np.ndarray] = {}
        self._fitted = False

    def fit(self, actions: pd.DataFrame, *, xg_column: str, pressure_column: str, pressure_levels=None):
        from silly_kicks.xtgk._pressure_levels import PressureLevels

        a = actions.reset_index(drop=True).copy()
        if "possession_id" not in a.columns:
            a = add_possessions(a)
        pl = pressure_levels or PressureLevels().fit(a[pressure_column])
        zones = _get_flat_indexes(a.start_x, a.start_y, self.l, self.w).to_numpy() if pl.mode == "zone_conditional" else None
        a["_p_level"] = pl.apply(a[pressure_column], zones=zones)
        a["_turnover"] = _is_turnover(a)
        a["_opp_shot_xg"] = self._opp_first_shot_after_turnover(a, xg_column, window_seconds=self.window_seconds)
        turnovers = a[a["_turnover"]].dropna(subset=["start_x", "start_y"])
        for p in (1, 2, 3):
            sub = turnovers[turnovers["_p_level"] == p]
            flat = _get_flat_indexes(sub.start_x, sub.start_y, self.l, self.w).to_numpy()
            num = np.zeros(self.w * self.l)
            den = np.zeros(self.w * self.l)
            np.add.at(num, flat, sub["_opp_shot_xg"].to_numpy(dtype=float))
            np.add.at(den, flat, 1.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                surf = np.where(den > 0, num / den, 0.0)
            self._surfaces[p] = surf.reshape((self.w, self.l))
            self._support[p] = den.reshape((self.w, self.l)).astype(int)
        self._fitted = True
        return self

    def _opp_first_shot_after_turnover(self, a: pd.DataFrame, xg_column: str, *, window_seconds: float) -> np.ndarray:
        """Per turnover action, the xG of the OPPONENT's first shot in the BOUNDED post-turnover
        window: same game, within window_seconds, and before the ball returns to the loser's team.
        A minute-10 turnover must NOT be charged an unrelated minute-40 opponent shot (the scan
        that validates the mirror V_opp cannot itself be noisy)."""
        SHOT = _SHOT
        out = np.zeros(len(a), dtype=float)
        team = a["team_id"].to_numpy()
        typ = a["type_id"].to_numpy()
        xg = a[xg_column].fillna(0.0).to_numpy(dtype=float)
        game = a["game_id"].to_numpy() if "game_id" in a.columns else np.zeros(len(a))
        poss = a["possession_id"].to_numpy()
        t = a["time_seconds"].to_numpy(dtype=float)
        turn = a["_turnover"].to_numpy()
        n = len(a)
        for i in range(n):
            if not turn[i]:
                continue
            for j in range(i + 1, n):
                if game[j] != game[i] or (t[j] - t[i]) > window_seconds:
                    break  # out of the bounded window
                if poss[j] == poss[i]:
                    continue  # still the loser's own (briefly interrupted) possession
                if team[j] == team[i]:
                    break  # ball back with the loser -> no opponent-threat credit
                if typ[j] == SHOT:
                    out[i] = xg[j]
                    break
        return out

    def _check(self):
        if not self._fitted:
            from sklearn.exceptions import NotFittedError

            raise NotFittedError("EmpiricalTurnoverValue.fit not called")

    def surface(self, p: _PL) -> npt.NDArray[np.float64]:
        self._check()
        return self._surfaces[p]

    def value(self, zone: int, p: _PL) -> float:
        self._check()
        return float(self._surfaces[p].ravel()[zone])

    def support(self, p: _PL) -> npt.NDArray[np.int_]:
        self._check()
        return self._support[p]
```

Add `from silly_kicks.xtgk._moves import _SHOT` at the top of `_turnover.py` (or reference `spadlconfig.actiontype_id["shot"]` directly).

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/xtgk/test_turnover.py -q`
Expected: PASS.

- [ ] **Step 5: Checkpoint**

Run: `python -m pytest tests/xtgk/ -m "not e2e" -q`
Expected: PASS.

---

### Task B5: Deep-loss ⇒ high-cost property + empirical-vs-mirror agreement

**Files:**
- Test: `tests/xtgk/test_turnover.py`

- [ ] **Step 1: Write the property test**

Add to `tests/xtgk/test_turnover.py`:

```python
def test_deep_loss_has_higher_cost_than_final_third_loss():
    # On the honest cohort, losing in your deep zone (mirror = opponent near your goal) should
    # cost more than losing in the final third (mirror = opponent in its own deep zone).
    v = _fit_v()
    tc = MirroredTurnoverCost(v)
    from silly_kicks.xtgk._possession_value import zone_of
    z_deep = zone_of(3.0, 34.0)
    z_final = zone_of(100.0, 34.0)
    assert tc.value(z_deep, 1) >= tc.value(z_final, 1)
```

- [ ] **Step 2: Run to verify it passes (implementation already exists)**

Run: `python -m pytest tests/xtgk/test_turnover.py::test_deep_loss_has_higher_cost_than_final_third_loss -q`
Expected: PASS (the cohort routes deep → build-up → final-third shots, so `V(mirror(deep))` > `V(mirror(final))`). If it fails, the failure is diagnostic of the fixture, not the adapter — extend `three_band_cohort` shot placement before weakening the assertion.

- [ ] **Step 3: Checkpoint**

Run: `python -m pytest tests/xtgk/ -m "not e2e" -q`
Expected: PASS.

---

# Phase C — SP3: `ρ` retention classifier

### Task C1: Extract `_ece` / `_reliability_slope` to a shared, extra-free module (H3)

**Files:**
- Create: `silly_kicks/calibration/_metrics.py`
- Modify: `scripts/train_gk_completion.py` (import the shared helpers instead of local defs)
- Test: `tests/calibration/test_metrics.py` (new)

> Note: `silly_kicks/calibration/__init__.py` is lazy/optuna-gated. Put the pure-numpy metrics in a NEW `_metrics.py` module that does NOT import optuna, and import it directly (`from silly_kicks.calibration._metrics import ece, reliability_slope`) so SP3/SP5 never pull the `[calibration]` extra.

- [ ] **Step 1: Write the failing test**

Create `tests/calibration/test_metrics.py`:

```python
import numpy as np

from silly_kicks.calibration._metrics import ece, reliability_slope


def test_ece_zero_for_perfectly_calibrated():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 5000)
    y = (rng.uniform(0, 1, 5000) < p).astype(int)
    assert ece(y, p) < 0.05


def test_reliability_slope_near_one_for_calibrated():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 5000)
    y = (rng.uniform(0, 1, 5000) < p).astype(int)
    assert 0.75 <= reliability_slope(y, p) <= 1.25
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/calibration/test_metrics.py -q`
Expected: FAIL (module missing). (Create `tests/calibration/__init__.py` if the package dir doesn't exist.)

- [ ] **Step 3: Create the shared module (copy the exact bodies from the trainer)**

Create `silly_kicks/calibration/_metrics.py`:

```python
"""Pure-numpy calibration metrics (ECE + reliability slope). No optuna, no [calibration] extra --
importable by the xtgk retention model + the v2 validation suite. Bodies lifted verbatim from
scripts/train_gk_completion.py (single-sourced here; the trainer now imports them)."""

from __future__ import annotations

import numpy as np


def ece(y, p, n_bins: int = 10) -> float:
    """Expected calibration error (binned |mean_pred - mean_obs|, weighted by bin mass)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += abs(p[m].mean() - y[m].mean()) * (m.mean())
    return float(e)


def reliability_slope(y, p, n_bins: int = 10) -> float:
    """Slope of binned mean-observed on binned mean-predicted (weighted by sqrt bin mass).
    ~1 = calibrated; <1 over-confident; >1 under-confident. NaN if <2 occupied bins."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    mp, mo, w = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            mp.append(p[m].mean())
            mo.append(y[m].mean())
            w.append(m.sum())
    if len(mp) < 2 or np.ptp(mp) < 1e-9:
        return float("nan")
    coef = np.polyfit(np.asarray(mp), np.asarray(mo), 1, w=np.sqrt(np.asarray(w, float)))
    return float(coef[0])
```

- [ ] **Step 4: Re-point the trainer at the shared module**

In `scripts/train_gk_completion.py`, delete the local `_ece` and `_reliability_slope` defs and add at the top: `from silly_kicks.calibration._metrics import ece as _ece, reliability_slope as _reliability_slope`. (Keep the `_ece`/`_reliability_slope` names so the rest of the script is unchanged.)

- [ ] **Step 5: Run to verify it passes + trainer still imports**

Run: `python -m pytest tests/calibration/test_metrics.py -q && python -c "import scripts.train_gk_completion"`
Expected: PASS + clean import.

- [ ] **Step 6: Checkpoint**

Run: `python -m pytest tests/ -m "not e2e and not slow" -q -k "calibration or gk_completion"`
Expected: PASS.

---

### Task C2: `RetentionModel` port

**Files:**
- Create: `silly_kicks/xtgk/_retention.py`
- Test: `tests/xtgk/test_retention.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/xtgk/test_retention.py`:

```python
import numpy as np
import pandas as pd

from silly_kicks.xtgk._retention import RetentionModel


def test_a_stub_satisfies_the_port():
    class _Stub:
        def predict_proba(self, features):
            return np.full(len(features), 0.7)

    stub = _Stub()
    assert isinstance(stub, RetentionModel)
    assert list(stub.predict_proba(pd.DataFrame({"x": [1, 2]}))) == [0.7, 0.7]
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/xtgk/test_retention.py::test_a_stub_satisfies_the_port -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement the port**

Create `silly_kicks/xtgk/_retention.py`:

```python
"""RetentionModel port + GkRetentionModel adapter (ADR-036 §Part 3).

P(retain | s,a) for GK distributions. Injected into the v2 metric (same discipline as
compute_xt_gk's completion=). Jeffrey's xR-GK later = a second adapter satisfying this port.
Logistic, sklearn at fit, pure-numpy serve, pickle-free JSON+SHA256 -- mirrors GkCompletionModel.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import pandas as pd


@runtime_checkable
class RetentionModel(Protocol):
    def predict_proba(self, features: pd.DataFrame) -> npt.NDArray[np.float64]: ...
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/xtgk/test_retention.py -q`
Expected: PASS.

- [ ] **Step 5: Checkpoint**

Run: `python -m pytest tests/xtgk/ -m "not e2e" -q`
Expected: PASS.

---

### Task C3: `retains` label

**Files:**
- Create: `silly_kicks/xtgk/_retention_labels.py`
- Test: `tests/xtgk/test_retention_labels.py` (new)

- [ ] **Step 1: Write the failing test (truth table)**

Create `tests/xtgk/test_retention_labels.py`:

```python
import numpy as np
import pandas as pd

from silly_kicks.xtgk._retention_labels import retains
import silly_kicks.spadl.config as spadlconfig

PASS = spadlconfig.actiontype_id["pass"]
SHOT = spadlconfig.actiontype_id["shot"]
GOALKICK = spadlconfig.actiontype_id["goalkick"]
SUCCESS = spadlconfig.result_id["success"]
FAIL = spadlconfig.result_id["fail"]


def _row(aid, t, team, typ, res, pid):
    return dict(game_id=1, period_id=1, action_id=aid, time_seconds=t, team_id=team, player_id=1,
                type_id=typ, result_id=res, possession_id=pid, start_x=5.0, start_y=34.0,
                end_x=20.0, end_y=34.0)


def test_retained_when_team_keeps_ball_through_window():
    # window 1.5s is fully covered by the 2s of data -> observed retention -> 1.0 (not NaN)
    a = pd.DataFrame([
        _row(0, 0.0, 10, GOALKICK, SUCCESS, 0),
        _row(1, 1.0, 10, PASS, SUCCESS, 0),
        _row(2, 2.0, 10, PASS, SUCCESS, 0),
    ])
    out = retains(a, window_seconds=1.5)
    assert out.iloc[0] == 1.0


def test_lost_when_opponent_takes_over_in_window():
    a = pd.DataFrame([
        _row(0, 0.0, 10, GOALKICK, SUCCESS, 0),
        _row(1, 1.0, 20, PASS, SUCCESS, 1),  # opponent possession
    ])
    out = retains(a, window_seconds=10.0)
    assert out.iloc[0] == 0.0


def test_retained_when_team_shoots_in_window():
    a = pd.DataFrame([
        _row(0, 0.0, 10, GOALKICK, SUCCESS, 0),
        _row(1, 1.0, 10, SHOT, FAIL, 0),
    ])
    out = retains(a, window_seconds=10.0)  # decisive shot -> 1.0 regardless of truncation
    assert out.iloc[0] == 1.0


def test_truncated_window_with_no_decisive_event_is_nan():
    # a lone goal-kick near a period end: the 10s window is truncated to 0s of observable data and
    # nothing decisive happens -> we did NOT observe retention -> NaN (excluded from training).
    a = pd.DataFrame([_row(0, 2699.0, 10, GOALKICK, SUCCESS, 0)])
    out = retains(a, window_seconds=10.0)
    assert np.isnan(out.iloc[0])
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/xtgk/test_retention_labels.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `retains` (searchsorted window + possession-boundary OR-condition)**

Create `silly_kicks/xtgk/_retention_labels.py`:

```python
"""retains() label for the ρ classifier (ADR-036 §Part 3, GENUINELY NEW -- not a copy).

Per action, within window_seconds either (a) the actor's team still holds the ball at window end
(no opponent possession boundary intervenes) OR (b) the actor's team takes a shot -> label 1.0;
if the opponent takes over before either -> label 0.0. A window TRUNCATED by end-of-period data
with no decisive event -> NaN (retention was NOT observed; excluded from training) rather than a
falsely-optimistic 1.0. Returns FLOAT (1.0/0.0/NaN). Searchsorted boundary skeleton borrowed from
vaep.labels._scores_time; the retain/loss payload + add_possessions coupling + truncation-NaN are new.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.spadl.utils import add_possessions

_SHOT = spadlconfig.actiontype_id["shot"]


def retains(actions: pd.DataFrame, *, window_seconds: float = 10.0) -> pd.Series:
    a = actions
    if "possession_id" not in a.columns:
        a = add_possessions(a)
    team = np.asarray(a["team_id"].values)
    typ = a["type_id"].to_numpy()
    poss = a["possession_id"].to_numpy()
    time_s = np.asarray(a["time_seconds"].values, dtype=np.float64)
    result = np.full(len(a), np.nan, dtype=float)

    group_keys = [k for k in ("game_id", "period_id") if k in a.columns]
    groups = a.groupby(group_keys) if group_keys else [(None, a)]
    for _key, grp in groups:
        idx = np.asarray(grp.index)
        t = time_s[idx]
        if len(t) > 1 and not (np.diff(t) >= -1e-9).all():
            raise ValueError("time_seconds must be non-decreasing within each (game_id, period_id) group")
        boundaries = np.searchsorted(t, t + window_seconds, side="left")
        t_last = t[-1] if len(t) else 0.0
        for li in range(len(idx)):
            gi = idx[li]
            end = min(boundaries[li], len(idx))
            label = None
            for lj in range(li + 1, end):
                gj = idx[lj]
                if typ[gj] == _SHOT and team[gj] == team[gi]:
                    label = 1.0  # (b) actor's team shoots -> decisive retain
                    break
                if team[gj] != team[gi] and poss[gj] != poss[gi]:
                    label = 0.0  # opponent possession boundary intervened -> decisive loss
                    break
            if label is None:
                # No decisive event observed. If the FULL window was observable (>= window_seconds
                # of subsequent data), the team retained -> 1.0. If the window was truncated by the
                # end of the (game, period) data, retention was NOT observed -> NaN.
                label = 1.0 if (t_last - t[li]) >= window_seconds - 1e-9 else np.nan
            result[gi] = label
    return pd.Series(result, index=a.index, name="retains")
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/xtgk/test_retention_labels.py -q`
Expected: PASS.

- [ ] **Step 5: Checkpoint**

Run: `python -m pytest tests/xtgk/ -m "not e2e" -q`
Expected: PASS.

---

### Task C4: `extract_retention_features` (train==serve parity)

**Files:**
- Create: `silly_kicks/xtgk/_retention_features.py`
- Test: `tests/xtgk/test_retention_features.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/xtgk/test_retention_features.py`:

```python
import numpy as np

from silly_kicks.xtgk._retention_features import RETENTION_FEATURE_NAMES, extract_retention_features
from tests.xtgk.conftest import three_band_cohort


def test_features_have_expected_columns_and_length():
    actions = three_band_cohort()
    X = extract_retention_features(actions, frames=None)  # frames=None -> density NaN, geometry native
    assert list(X.columns) == RETENTION_FEATURE_NAMES
    assert len(X) == len(actions)
    assert np.isfinite(X["length"].to_numpy()).all()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/xtgk/test_retention_features.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement the shared extractor (mirror `extract_gk_completion_features`)**

Create `silly_kicks/xtgk/_retention_features.py`:

```python
"""Shared train==serve feature extractor for the ρ retention model (ADR-036 §Part 3).

Reuses resolve_gk_geometry (origin/dest + provenance) + receiver_zone_density; mirrors
tracking._gk_completion.extract_gk_completion_features. ONE code path at train and serve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.tracking._gk_geometry import resolve_gk_geometry

_GOALKICK = spadlconfig.actiontype_id["goalkick"]
_THROW_IN = spadlconfig.actiontype_id["throw_in"]

RETENTION_FEATURE_NAMES = [
    "length",
    "forwardness",
    "dy_abs",
    "dest_x",
    "dest_y_off",
    "dest_defender_density",
    "release_pressure",
    "is_goalkick",
    "is_throw_in",
]


def _density(actions: pd.DataFrame, frames: pd.DataFrame | None, geom: pd.DataFrame, links) -> pd.Series:
    if frames is None:
        return pd.Series(np.nan, index=actions.index)
    from silly_kicks.tracking.features import receiver_zone_density

    a = actions.copy()
    a["end_x"] = geom["dest_x"].to_numpy()
    a["end_y"] = geom["dest_y"].to_numpy()
    return receiver_zone_density(a, frames)


def extract_retention_features(
    actions: pd.DataFrame,
    *,
    frames: pd.DataFrame | None,
    links: pd.DataFrame | None = None,
    pressure_column: str = "pressure",
) -> pd.DataFrame:
    geom = resolve_gk_geometry(actions, frames=frames, links=links)
    ox = geom["origin_x"].to_numpy(float)
    oy = geom["origin_y"].to_numpy(float)
    dx = geom["dest_x"].to_numpy(float) - ox
    dy = geom["dest_y"].to_numpy(float) - oy
    length = np.hypot(dx, dy)
    dens = _density(actions, frames, geom, links).to_numpy(float)
    tid = actions["type_id"].to_numpy()
    release = (
        actions[pressure_column].to_numpy(float)
        if pressure_column in actions.columns
        else np.full(len(actions), np.nan)
    )
    return pd.DataFrame(
        {
            "length": length,
            "forwardness": np.divide(dx, length, out=np.zeros_like(dx), where=length > 0),
            "dy_abs": np.abs(dy),
            "dest_x": geom["dest_x"].to_numpy(float),
            "dest_y_off": np.abs(geom["dest_y"].to_numpy(float) - spadlconfig.field_width / 2),
            "dest_defender_density": dens,
            "release_pressure": release,
            "is_goalkick": (tid == _GOALKICK).astype(float),
            "is_throw_in": (tid == _THROW_IN).astype(float),
        },
        index=actions.index,
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/xtgk/test_retention_features.py -q`
Expected: PASS.

- [ ] **Step 5: Checkpoint**

Run: `python -m pytest tests/xtgk/ -m "not e2e" -q`
Expected: PASS.

---

### Task C5: `GkRetentionModel` (logistic, pure-numpy serve, JSON+SHA256)

**Files:**
- Modify: `silly_kicks/xtgk/_retention.py`
- Test: `tests/xtgk/test_retention.py`

> This is a deliberate structural copy of `tracking/_gk_completion.py::GkCompletionModel` with these exact deltas: (1) `feature_names = RETENTION_FEATURE_NAMES`; (2) NO per-type serve gate (`_type_serve_mode`/`serve_mode_from_lcb`) — ρ serves the model everywhere it is calibrated; (3) weights dir `_retention_weights`; (4) `from_variant` provider map identical shape. Copy the class body verbatim, then apply the deltas — do not invent a new serialization format (byte-for-byte per `GkCompletionModel.save()/load()`).

- [ ] **Step 1: Write the failing test**

Add to `tests/xtgk/test_retention.py`:

```python
import numpy as np
import pandas as pd

from silly_kicks.xtgk._retention import GkRetentionModel
from silly_kicks.xtgk._retention_features import RETENTION_FEATURE_NAMES


def _fake_training():
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({c: rng.normal(size=n) for c in RETENTION_FEATURE_NAMES})
    y = (X["forwardness"] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return X, y


def test_fit_serve_roundtrip_pure_numpy(tmp_path):
    X, y = _fake_training()
    m = GkRetentionModel().fit(X, pd.Series(y))
    p = m.predict_proba(X)
    assert p.shape == (len(X),)
    assert ((p >= 0) & (p <= 1)).all()
    m.save(tmp_path / "ret")
    reloaded = GkRetentionModel.load(tmp_path / "ret")
    assert np.allclose(reloaded.predict_proba(X), p)


def test_load_detects_tamper(tmp_path):
    X, y = _fake_training()
    GkRetentionModel().fit(X, pd.Series(y)).save(tmp_path / "ret")
    (tmp_path / "ret" / "model.json").write_text('{"version":"9"}')
    import pytest
    with pytest.raises(ValueError, match="integrity"):
        GkRetentionModel.load(tmp_path / "ret")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/xtgk/test_retention.py::test_fit_serve_roundtrip_pure_numpy -q`
Expected: FAIL (`GkRetentionModel` undefined).

- [ ] **Step 3: Implement (copy `GkCompletionModel` with the deltas above)**

Append to `silly_kicks/xtgk/_retention.py` a `GkRetentionModel` class copied from `tracking/_gk_completion.py::GkCompletionModel` with the deltas. Key structure to reproduce exactly (fit standardizes with `nanmean`/`nanstd`, mean-imputes NaN, `LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")`; serve is `1/(1+exp(-(Xs@coef+intercept)))`; `to_dict`/`from_dict`/`save`/`load`/`_sha`/`from_variant` byte-for-byte). Minimal shape:

```python
import hashlib
import json
import warnings
from pathlib import Path

from silly_kicks.xtgk._retention_features import RETENTION_FEATURE_NAMES

_WEIGHTS_ROOT = Path(__file__).parent / "_retention_weights"
_VARIANT_CACHE: dict = {}
_VARIANT_KEY_ALIASES = {"gs": "default"}
_PROVIDER_VARIANT = {"skillcorner": "skillcorner"}


def variant_key_for_provider(source_provider: str | None) -> str:
    return _PROVIDER_VARIANT.get(str(source_provider).lower() if source_provider is not None else "", "gs")


class GkRetentionModel:
    """Logistic P(retain) for GK distributions. sklearn at fit; pure-numpy at serve."""

    VERSION = "1.0.0"

    def __init__(self) -> None:
        self._coef: np.ndarray | None = None
        self._intercept: float = 0.0
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self.feature_names: list[str] = list(RETENTION_FEATURE_NAMES)
        self.shipped_variant: str | None = None
        self.provider_list: list | None = None

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> "GkRetentionModel":
        from sklearn.linear_model import LogisticRegression

        X_raw = features[self.feature_names].to_numpy(float)
        y = np.asarray(labels, dtype=int)
        mean = np.nanmean(X_raw, axis=0)
        std_raw = np.nanstd(X_raw, axis=0)
        std = np.where(std_raw > 1e-9, std_raw, 1.0)
        self._mean, self._std = mean, std
        X = np.where(np.isfinite(X_raw), X_raw, mean[None, :])
        Xs = (X - mean) / std
        clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs").fit(Xs, y)
        self._coef = clf.coef_[0].astype(float)
        self._intercept = float(clf.intercept_[0])
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        coef, mean, std = self._coef, self._mean, self._std
        if coef is None or mean is None or std is None:
            raise RuntimeError("GkRetentionModel not fitted/loaded.")
        X = features[self.feature_names].to_numpy(float)
        Xf = np.where(np.isfinite(X), X, mean[None, :])
        Xs = (Xf - mean) / std
        return 1.0 / (1.0 + np.exp(-(Xs @ coef + self._intercept)))

    def to_dict(self) -> dict:
        import sklearn

        if self._coef is None or self._mean is None or self._std is None:
            raise RuntimeError("GkRetentionModel not fitted/loaded; nothing to serialize.")
        return {
            "version": self.VERSION,
            "feature_names": self.feature_names,
            "coef": self._coef.tolist(),
            "intercept": self._intercept,
            "mean": self._mean.tolist(),
            "std": self._std.tolist(),
            "sklearn_version": sklearn.__version__,
            "shipped_variant": self.shipped_variant,
            "provider_list": self.provider_list,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GkRetentionModel":
        m = cls()
        m.feature_names = list(d["feature_names"])
        m._coef = np.asarray(d["coef"], dtype=float)
        m._intercept = float(d["intercept"])
        m._mean = np.asarray(d["mean"], dtype=float)
        m._std = np.asarray(d["std"], dtype=float)
        m.shipped_variant = d.get("shipped_variant")
        m.provider_list = d.get("provider_list")
        return m

    @staticmethod
    def _sha(path: Path) -> str:
        text = (path / "model.json").read_text(encoding="utf-8").replace("\r\n", "\n")
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "model.json").write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        (path / "SHA256SUMS").write_text(f"{self._sha(path)}  model.json\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "GkRetentionModel":
        path = Path(path)
        want = (path / "SHA256SUMS").read_text(encoding="utf-8").split()[0]
        if want != cls._sha(path):
            raise ValueError(f"GkRetentionModel integrity check failed at {path}")
        return cls.from_dict(json.loads((path / "model.json").read_text(encoding="utf-8")))

    @classmethod
    def from_variant(cls, variant: str = "default") -> "GkRetentionModel":
        variant = _VARIANT_KEY_ALIASES.get(variant, variant)
        if variant in _VARIANT_CACHE:
            return _VARIANT_CACHE[variant]
        wdir = _WEIGHTS_ROOT / variant
        if not (wdir / "SHA256SUMS").exists():
            raise FileNotFoundError(
                f"No bundled retention weights for {variant!r} at {wdir}. Train via scripts/train_gk_retention.py."
            )
        m = cls.load(wdir)
        _VARIANT_CACHE[variant] = m
        return m
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/xtgk/test_retention.py -q`
Expected: PASS.

- [ ] **Step 5: Checkpoint**

Run: `python -m pytest tests/xtgk/ -m "not e2e" -q`
Expected: PASS.

---

### Task C6: Retention training script + calibration gate

**Files:**
- Create: `scripts/train_gk_retention.py`
- Test: `tests/xtgk/test_train_gk_retention_smoke.py` (new, `@pytest.mark.slow`)

> Structural mirror of `scripts/train_gk_completion.py`: build `(features, labels, groups)` via `_gk_distribution_mask` + `extract_retention_features` + `retains` labels, GroupKFold(game_id) OOF preds, then the calibration gate `ece(y, oof) <= 0.10 AND |reliability_slope(y, oof) - 1| <= 0.25` applied to EVERY shipped variant.

- [ ] **Step 1: Write the failing smoke test**

Create `tests/xtgk/test_train_gk_retention_smoke.py`:

```python
import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.slow


def test_prepare_retention_training_data_builds_labels_and_features():
    from scripts.train_gk_retention import prepare_retention_training_data
    from tests.xtgk.conftest import three_band_cohort

    actions = three_band_cohort()
    X, y, groups = prepare_retention_training_data(actions, frames=None)
    assert len(X) == len(y) == len(groups)
    assert set(np.unique(y)) <= {0, 1}
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/xtgk/test_train_gk_retention_smoke.py -q`
Expected: FAIL (script missing).

- [ ] **Step 3: Implement the trainer**

Create `scripts/train_gk_retention.py` with:

```python
"""Train the GK-distribution retention model (ρ) for xT-GK v2 (ADR-036 §Part 3).

Mirror of train_gk_completion.py, but the label is retains() (NOT completion) and EVERY shipped
variant is calibration-gated (ece<=0.10 AND |reliability_slope-1|<=0.25). GS(WC2022) 'default' +
SkillCorner variant via the same GS-transfer-or-bundle decision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.calibration._metrics import ece, reliability_slope
from silly_kicks.xtgk._retention import GkRetentionModel
from silly_kicks.xtgk._retention_features import extract_retention_features
from silly_kicks.xtgk._retention_labels import retains

_ECE_MAX = 0.10
_SLOPE_TOL = 0.25


def prepare_retention_training_data(
    actions: pd.DataFrame, *, frames: pd.DataFrame | None, links: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build (features, labels, groups). Serve domain = _gk_distribution_mask (frames) or goalkicks
    (no frames). resolve-on-full-then-mask parity. Drops geometry-unscoreable rows."""
    import silly_kicks.spadl.config as spadlconfig
    from silly_kicks.tracking._xt_gk import _gk_distribution_mask

    _GOALKICK = spadlconfig.actiontype_id["goalkick"]
    if frames is not None:
        mask = _gk_distribution_mask(actions, frames)
    else:
        mask = actions["type_id"].to_numpy() == _GOALKICK
    X_full = extract_retention_features(actions, frames=frames, links=links)
    y_full = retains(actions).to_numpy(dtype=float)  # 1.0 / 0.0 / NaN (truncated windows)
    domain = np.asarray(mask, dtype=bool)
    X = X_full.loc[domain].reset_index(drop=True)
    y = y_full[domain]
    groups = (actions["game_id"].to_numpy() if "game_id" in actions.columns else np.zeros(len(actions)))[domain]
    # drop geometry-unscoreable rows AND truncated-window (NaN-label) rows -> observed labels only
    keep = (
        np.isfinite(X["length"].to_numpy())
        & np.isfinite(X["dest_x"].to_numpy())
        & np.isfinite(y)
    )
    return X.loc[keep].reset_index(drop=True), y[keep].astype(int), groups[keep]


def calibration_gate(y: np.ndarray, oof: np.ndarray) -> tuple[bool, dict]:
    e = ece(y, oof)
    s = reliability_slope(y, oof)
    ok = (e <= _ECE_MAX) and (np.isfinite(s) and abs(s - 1.0) <= _SLOPE_TOL)
    return bool(ok), {"ece": e, "reliability_slope": s, "ece_max": _ECE_MAX, "slope_tol": _SLOPE_TOL}


def cross_val_oof(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    from sklearn.model_selection import GroupKFold

    n_splits = min(5, len(np.unique(groups)))
    oof = np.full(len(y), np.nan)
    if n_splits < 2:
        return GkRetentionModel().fit(X, pd.Series(y)).predict_proba(X)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = GkRetentionModel().fit(X.iloc[tr], pd.Series(y[tr]))
        oof[te] = m.predict_proba(X.iloc[te])
    return oof


# NOTE: the __main__ owner-run (real pining/Databricks load, per-variant fit, calibration_gate,
# save under silly_kicks/xtgk/_retention_weights/{default,skillcorner}/) mirrors
# train_gk_completion.py's CLI; not exercised in CI (owner-run).
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/xtgk/test_train_gk_retention_smoke.py -q`
Expected: PASS.

- [ ] **Step 5: Checkpoint**

Run: `python -m pytest tests/xtgk/ -m "not e2e" -q`
Expected: PASS.

---

### Task C7: Bundle retention weights (owner-run) + a bundled-load test guard

**Files:**
- Create (owner-run): `silly_kicks/xtgk/_retention_weights/default/{model.json,SHA256SUMS,metrics.json,MODEL_CARD.md}`
- Create (owner-run): `silly_kicks/xtgk/_retention_weights/skillcorner/{...}`
- Test: `tests/xtgk/test_retention.py`

- [ ] **Step 1: Write the guard test (skips gracefully until weights land)**

Add to `tests/xtgk/test_retention.py`:

```python
import pytest

from silly_kicks.xtgk._retention import GkRetentionModel


def test_bundled_default_variant_loads_if_present():
    try:
        m = GkRetentionModel.from_variant("default")
    except FileNotFoundError:
        pytest.skip("retention weights not yet bundled (owner-run training)")
    import pandas as pd
    from silly_kicks.xtgk._retention_features import RETENTION_FEATURE_NAMES
    X = pd.DataFrame({c: [0.0] for c in RETENTION_FEATURE_NAMES})
    p = m.predict_proba(X)
    assert 0.0 <= float(p[0]) <= 1.0
```

- [ ] **Step 2: Run**

Run: `python -m pytest tests/xtgk/test_retention.py::test_bundled_default_variant_loads_if_present -q`
Expected: SKIP (weights not yet bundled) — becomes PASS after the owner-run training in Step 3.

- [ ] **Step 3: Owner-run training (documented, not CI)**

Run the owner-run: fit `default` (GS/WC2022) + `skillcorner` variants via `scripts/train_gk_retention.py` on the pining corpus, apply `calibration_gate` to EACH variant, and `save()` to `silly_kicks/xtgk/_retention_weights/{default,skillcorner}/`. Record ECE/slope in `metrics.json` + a `MODEL_CARD.md`. If a variant fails ECE with plain logistic, escalate to `sklearn.calibration.CalibratedClassifierCV` (documented in the ADR) before bundling.

- [ ] **Step 4: Checkpoint**

Run: `python -m pytest tests/xtgk/ -m "not e2e" -q`
Expected: PASS (guard skips until weights present).

---

### Task C8: Export the retention surface

**Files:**
- Modify: `silly_kicks/xtgk/__init__.py`

- [ ] **Step 1: Add exports**

Add `RetentionModel`, `GkRetentionModel` (from `_retention`), `retains` (from `_retention_labels`), `extract_retention_features` (from `_retention_features`) to imports + `__all__`.

- [ ] **Step 2: Checkpoint**

Run: `python -c "from silly_kicks.xtgk import RetentionModel, GkRetentionModel, retains, extract_retention_features" && python -m pytest tests/xtgk/ -m "not e2e" -q`
Expected: PASS.

---

# Phase D — SP4: metric assembly + decomposition

### Task D1: `compute_xt_gk_v2` assembler + four-term decomposition (on stub ports)

**Files:**
- Create: `silly_kicks/xtgk/_metric.py`
- Test: `tests/xtgk/test_metric.py` (new)

- [ ] **Step 1: Write the failing test (formula in isolation via stub ports)**

Create `tests/xtgk/test_metric.py`:

```python
import numpy as np
import pandas as pd

from silly_kicks.xtgk._metric import compute_xt_gk_v2
from silly_kicks.xtgk._possession_value import DeltaV, State
import silly_kicks.spadl.config as spadlconfig

GOALKICK = spadlconfig.actiontype_id["goalkick"]


class _StubV:
    def __init__(self, surface_value=0.02):
        self._val = surface_value

    def value(self, zone, p):
        return self._val

    def surface(self, p):
        return np.full((12, 16), self._val)

    def delta_v(self, s, s_next):
        # position-only ΔV (p'=p) of +0.03; pressure component 0
        return DeltaV(delta=0.03, pressure_component=0.0, position_component=0.03)


class _StubRho:
    def __init__(self, rho):
        self._rho = rho

    def predict_proba(self, features):
        return np.full(len(features), self._rho)


class _StubTurnover:
    def __init__(self, v_opp=0.05):
        self._v = v_opp

    def value(self, zone, p):
        return self._v

    def surface(self, p):
        return np.full((12, 16), self._v)

    def support(self, p):
        return np.full((12, 16), 100, dtype=int)


def _one_goalkick():
    return pd.DataFrame([dict(
        game_id=1, period_id=1, action_id=0, time_seconds=0.0, team_id=10, player_id=1,
        type_id=GOALKICK, result_id=spadlconfig.result_id["success"], bodypart_id=0,
        start_x=5.0, start_y=34.0, end_x=40.0, end_y=34.0, pressure=0.1,
    )])


def _pl_and_feats(actions):
    from silly_kicks.xtgk import PressureLevels
    pl = PressureLevels().fit(actions["pressure"])
    feats = pd.DataFrame(index=actions.index)  # stub ρ ignores content, uses len(features)
    return pl, feats


def test_four_terms_sum_to_metric_and_pev_zero_when_pprime_equals_p():
    actions = _one_goalkick()
    pl, feats = _pl_and_feats(actions)
    out = compute_xt_gk_v2(
        actions, possession_value=_StubV(), retention=_StubRho(0.8),
        turnover_cost=_StubTurnover(), kappa=1.0, pressure_column="pressure",
        pressure_levels=pl, retention_features=feats,
    )
    row = out.iloc[0]
    # terms: (1) 0.8*0.03  (2) 0.8*0.0=PEV  (3) -0.2*0.02  (4) -0.2*1.0*0.05
    assert np.isclose(row["xt_gk_v2_position"], 0.8 * 0.03)
    assert np.isclose(row["xt_gk_v2_pev"], 0.0)
    assert np.isclose(row["xt_gk_v2_retention_loss"], -0.2 * 0.02)
    assert np.isclose(row["xt_gk_v2_dzv"], -0.2 * 1.0 * 0.05)
    total = (row["xt_gk_v2_position"] + row["xt_gk_v2_pev"]
             + row["xt_gk_v2_retention_loss"] + row["xt_gk_v2_dzv"])
    assert np.isclose(row["xt_gk_v2"], total)


def test_kappa_scales_only_the_dzv_term():
    actions = _one_goalkick()
    pl, feats = _pl_and_feats(actions)
    out1 = compute_xt_gk_v2(actions, possession_value=_StubV(), retention=_StubRho(0.8),
                            turnover_cost=_StubTurnover(), kappa=1.0, pressure_column="pressure",
                            pressure_levels=pl, retention_features=feats)
    out2 = compute_xt_gk_v2(actions, possession_value=_StubV(), retention=_StubRho(0.8),
                            turnover_cost=_StubTurnover(), kappa=2.0, pressure_column="pressure",
                            pressure_levels=pl, retention_features=feats)
    assert np.isclose(out2.iloc[0]["xt_gk_v2_dzv"], 2 * out1.iloc[0]["xt_gk_v2_dzv"])
    assert np.isclose(out2.iloc[0]["xt_gk_v2_position"], out1.iloc[0]["xt_gk_v2_position"])


def test_requires_pressure_levels_and_features():
    import pytest
    actions = _one_goalkick()
    _pl, feats = _pl_and_feats(actions)
    with pytest.raises(ValueError, match="pressure_levels"):
        compute_xt_gk_v2(actions, possession_value=_StubV(), retention=_StubRho(0.8),
                         turnover_cost=_StubTurnover(), retention_features=feats)  # no pl, stub has none
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/xtgk/test_metric.py::test_four_terms_sum_to_metric_and_pev_zero_when_pprime_equals_p -q`
Expected: FAIL (`_metric` module missing).

- [ ] **Step 3: Implement the assembler**

Create `silly_kicks/xtgk/_metric.py`:

```python
"""xT-GK v2 metric assembler (ADR-036 §Part 4). Depends ONLY on the three ports.

xT-GK = rho*ΔV_position + rho*ΔV_pressure(=PEV) - (1-rho)*V(s) - (1-rho)*kappa*V_opp
The four terms sum to the metric exactly. Columns namespaced xt_gk_v2_* (v1's xt_gk_* are frozen).
PEV is 0 by construction when p'=p (base metric); it lights up only with receiver-pressure q.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.xtgk._possession_value import State
from silly_kicks.xtgk._pressure_levels import PressureLevels
from silly_kicks.xthreat._grid import M, N, _get_flat_indexes

_OUTPUT_COLS = [
    "xt_gk_v2_position",
    "xt_gk_v2_pev",
    "xt_gk_v2_retention_loss",
    "xt_gk_v2_dzv",
    "xt_gk_v2",
]


def compute_xt_gk_v2(
    actions: pd.DataFrame,
    *,
    possession_value,
    retention,
    turnover_cost,
    kappa: float = 1.0,
    pressure_column: str = "pressure",
    pressure_levels: PressureLevels | None = None,
    retention_features: pd.DataFrame | None = None,
    l: int = N,
    w: int = M,
) -> pd.DataFrame:
    """Per action, the v2 metric + four-term decomposition. Ports are injected (swappable).

    The metric's terciles MUST match the surfaces V was fit on: pass `pressure_levels=` (or a
    `possession_value` exposing `.pressure_levels`) -- never refit a fresh one. `retention_features`
    must be built WITH frames (matching ρ's training); `frames=None` is a stub-test-only path."""
    pl = pressure_levels if pressure_levels is not None else getattr(possession_value, "pressure_levels", None)
    if pl is None:
        raise ValueError(
            "compute_xt_gk_v2 needs pressure_levels= (or a possession_value exposing .pressure_levels) "
            "so the metric's terciles match the surfaces V was fit on -- never refit."
        )
    if retention_features is None:
        raise ValueError(
            "compute_xt_gk_v2 needs retention_features= built WITH frames (matching ρ's training); "
            "silently defaulting to frames=None yields NaN density -> train/serve skew."
        )
    zones_o = _get_flat_indexes(actions["start_x"], actions["start_y"], l, w).to_numpy()
    zones_d = _get_flat_indexes(actions["end_x"], actions["end_y"], l, w).to_numpy()
    zones_arg = zones_o if pl.mode == "zone_conditional" else None
    levels = pl.apply(actions[pressure_column], zones=zones_arg)  # p' = p (base metric)

    rho = np.asarray(retention.predict_proba(retention_features), dtype=float)

    # NOTE (scale): the per-action Python loop below calling delta_v/value is correctness-first and
    # fine for the GK-distribution slice (a small fraction of actions); a batch path (vectorized grid
    # lookups over the pressure-stratified surfaces) is a follow-up if the lakehouse needs full-stream.
    n = len(actions)
    position = np.zeros(n)
    pev = np.zeros(n)
    ret_loss = np.zeros(n)
    dzv = np.zeros(n)
    for i in range(n):
        p = int(levels[i])
        s = State(int(zones_o[i]), p)  # type: ignore[arg-type]
        s_next = State(int(zones_d[i]), p)  # type: ignore[arg-type]
        dv = possession_value.delta_v(s, s_next)
        v_s = float(possession_value.value(int(zones_o[i]), p))
        v_opp = float(turnover_cost.value(int(zones_o[i]), p))
        position[i] = rho[i] * dv.position_component
        pev[i] = rho[i] * dv.pressure_component
        ret_loss[i] = -(1.0 - rho[i]) * v_s
        dzv[i] = -(1.0 - rho[i]) * kappa * v_opp
    total = position + pev + ret_loss + dzv
    return pd.DataFrame(
        {
            "xt_gk_v2_position": position,
            "xt_gk_v2_pev": pev,
            "xt_gk_v2_retention_loss": ret_loss,
            "xt_gk_v2_dzv": dzv,
            "xt_gk_v2": total,
        },
        index=actions.index,
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/xtgk/test_metric.py -q`
Expected: PASS.

- [ ] **Step 5: Checkpoint**

Run: `python -m pytest tests/xtgk/ -m "not e2e" -q`
Expected: PASS.

---

### Task D2: Export the assembler + regression boundary (xthreat + v1 byte-unchanged)

**Files:**
- Modify: `silly_kicks/xtgk/__init__.py`
- Modify: `tests/xtgk/test_regression_boundary.py`

- [ ] **Step 1: Add exports**

Add `compute_xt_gk_v2` (from `_metric`), `TurnoverCost`, `MirroredTurnoverCost`, `EmpiricalTurnoverValue` (from `_turnover`), `mirror_zone` (from `_possession_value`) to imports + `__all__`.

- [ ] **Step 2: Write the v1 byte-stability test**

Add to `tests/xtgk/test_regression_boundary.py`:

```python
def test_v1_xt_gk_module_not_imported_by_v2():
    import silly_kicks.xtgk as v2
    import inspect
    src = inspect.getsource(v2._metric)
    assert "tracking._xt_gk" not in src and "tracking/_xt_gk" not in src


def test_v1_compute_xt_gk_still_produces_v1_columns():
    # Guard: v2 landing must not change v1 output column names (lakehouse/UI Hyrum).
    from silly_kicks.tracking._xt_gk import _OUTPUT_COLS as v1_cols
    assert v1_cols == ["xt_gk_base", "xt_gk_pev", "xt_gk_rav", "xt_gk_dzv", "xt_gk_pressure", "xt_gk"]


def test_v2_columns_disjoint_from_frozen_v1_columns():
    # H1: v2 must NOT reuse v1's xt_gk_pev/rav/dzv (frozen, lakehouse/UI-read).
    from silly_kicks.tracking._xt_gk import _OUTPUT_COLS as v1_cols
    from silly_kicks.xtgk._metric import _OUTPUT_COLS as v2_cols
    assert set(v1_cols).isdisjoint(set(v2_cols))
```

- [ ] **Step 3: Run**

Run: `python -m pytest tests/xtgk/test_regression_boundary.py -q`
Expected: PASS.

- [ ] **Step 4: Checkpoint (full xtgk + xthreat)**

Run: `python -m pytest tests/xtgk/ tests/xthreat_legacy_reference.py -m "not e2e" -q`
Expected: PASS (xthreat parity gate untouched).

---

# Phase E — SP5: validation suite + release

### Task E1: Owner-run validation script (construct validity / transfer / calibration / repeatability)

**Files:**
- Create: `scripts/validate_xtgk_v2.py`
- Test: `tests/xtgk/test_validate_v2_smoke.py` (new, `@pytest.mark.slow`)

- [ ] **Step 1a: Add a mixed (both-class) cohort to `tests/xtgk/conftest.py`**

The construct-validity target ("possession reaches a shot") needs BOTH classes or `roc_auc_score`
is undefined. `three_band_cohort` is all shot-reaching (all 1s); combine it with the shotless
`flat_no_shot_cohort` (all 0s), with distinct possession/action ids so they don't collide:

```python
def mixed_shot_and_shotless_cohort(n_per_band=40) -> pd.DataFrame:
    """Both classes for the construct-validity target: shot-reaching possessions (three_band, y=1)
    AND shotless possessions (flat_no_shot, y=0), in one game with distinct possession/action ids."""
    shot = three_band_cohort(n_per_band=n_per_band)
    noshot = flat_no_shot_cohort(n_per_band=n_per_band).copy()
    noshot["possession_id"] = noshot["possession_id"] + 100_000
    noshot["action_id"] = noshot["action_id"] + 10_000_000
    return make_cohort(pd.concat([shot, noshot], ignore_index=True))
```

- [ ] **Step 1b: Write the failing smoke test**

Create `tests/xtgk/test_validate_v2_smoke.py`:

```python
import numpy as np
import pytest

pytestmark = pytest.mark.slow


def test_construct_validity_reports_all_baselines_with_finite_v2_auc():
    from scripts.validate_xtgk_v2 import construct_validity_scores
    from tests.xtgk.conftest import mixed_shot_and_shotless_cohort

    actions = mixed_shot_and_shotless_cohort()
    scores = construct_validity_scores(actions, xg_column="xg", pressure_column="pressure")
    for key in ("xt_gk_v2", "raw_completion", "destination_xt", "v1_composite"):
        assert key in scores, f"missing baseline {key}"
    # both classes present in the out-of-sample test split -> finite AUC for the frame-free scorers
    assert np.isfinite(scores["xt_gk_v2"]["auc"])
    assert np.isfinite(scores["destination_xt"]["auc"])
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/xtgk/test_validate_v2_smoke.py -q`
Expected: FAIL (script missing).

- [ ] **Step 3: Implement the validation harness**

Create `scripts/validate_xtgk_v2.py` with a `construct_validity_scores(actions, *, xg_column, pressure_column)` that does an **out-of-sample** possession-parity split (fit V on even possessions, score odd), computes `xt_gk_v2`, builds the possession→shot target, and reports AUC for **all four baselines** the prose promises: v2, raw completion, destination-only V, and the v1 composite (frame-guarded → NaN in the frames-free synthetic path; real in the owner-run). A `_note` records the V∝first-shot-xG circularity so absolute AUC isn't over-read — the *lift over baselines* is the finding. The WC2018/Neuer motivating repro is a documented stub + TODO. The `__main__` (real pining/Databricks load, WC2022-authorising + RM-provisional with the REAL ρ + frames-derived features, JSON/markdown report under `docs/research/xtgk_possession_value/`) is owner-run, not CI. Exported core:

```python
"""Owner-run validation suite for xT-GK v2 (ADR-036 §Part 5).

Construct validity is OUT-OF-SAMPLE (possession-parity split) and reported as LIFT over baselines --
V is (by construction) the expected first-shot xG, so absolute AUC vs a possession->shot target is
partly circular; the informative quantity is v2's margin over raw completion / destination-only V /
the v1 composite. The synthetic CI smoke uses a constant-ρ stub (frames-free); the owner-run passes
the REAL calibrated ρ with frames-derived retention_features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.xtgk import MarkovPossessionValue, MirroredTurnoverCost, PressureLevels, compute_xt_gk_v2
import silly_kicks.spadl.config as spadlconfig
from silly_kicks.spadl.utils import add_possessions
from silly_kicks.xthreat._grid import M, N, _get_flat_indexes

_SHOT = spadlconfig.actiontype_id["shot"]


class _ConstRho:
    """Frames-free stub for the CI smoke; the owner-run injects the real GkRetentionModel."""

    def predict_proba(self, features):
        return np.full(len(features), 0.75)


def _possession_reaches_shot(actions: pd.DataFrame) -> np.ndarray:
    a = actions
    out = np.zeros(len(a), dtype=int)
    typ = a["type_id"].to_numpy()
    poss = a["possession_id"].to_numpy()
    for i in range(len(a)):
        for j in range(i, len(a)):
            if poss[j] != poss[i]:
                break
            if typ[j] == _SHOT:
                out[i] = 1
                break
    return out


def _auc(y, s) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    ok = np.isfinite(s)
    if len(np.unique(y[ok])) < 2 or ok.sum() < 2:
        return float("nan")
    return float(roc_auc_score(y[ok], s[ok]))


def _destination_only_v(test: pd.DataFrame, v: MarkovPossessionValue, pl: PressureLevels, pressure_column: str) -> np.ndarray:
    zd = _get_flat_indexes(test["end_x"], test["end_y"], N, M).to_numpy()
    zones_arg = _get_flat_indexes(test["start_x"], test["start_y"], N, M).to_numpy() if pl.mode == "zone_conditional" else None
    lv = pl.apply(test[pressure_column], zones=zones_arg)
    return np.array([v.value(int(z), int(p)) for z, p in zip(zd, lv)])


def _v1_composite(test: pd.DataFrame, frames: pd.DataFrame | None) -> np.ndarray:
    if frames is None:
        return np.full(len(test), np.nan)  # v1 is tracking-based; frames-free CI cannot score it
    from silly_kicks.tracking._xt_gk import add_xt_gk  # owner-run path (frames present)

    return add_xt_gk(test, frames)["xt_gk"].to_numpy()


def construct_validity_scores(
    actions: pd.DataFrame, *, xg_column: str, pressure_column: str, frames: pd.DataFrame | None = None
) -> dict:
    a = actions.reset_index(drop=True)
    if "possession_id" not in a.columns:
        a = add_possessions(a)
    train_mask = (a["possession_id"] % 2 == 0).to_numpy()  # out-of-sample by possession parity
    train, test = a[train_mask].copy(), a[~train_mask].copy()
    pl = PressureLevels().fit(train[pressure_column])
    v = MarkovPossessionValue().fit(train, xg_column=xg_column, pressure_column=pressure_column, pressure_levels=pl)
    tc = MirroredTurnoverCost(v)
    feats = pd.DataFrame(index=test.index)  # _ConstRho ignores content; owner-run supplies real features
    v2 = compute_xt_gk_v2(test, possession_value=v, retention=_ConstRho(), turnover_cost=tc,
                          pressure_column=pressure_column, pressure_levels=pl, retention_features=feats)
    y = _possession_reaches_shot(test)
    raw_completion = (test["result_id"] == spadlconfig.result_id["success"]).astype(int).to_numpy()
    return {
        "xt_gk_v2": {"auc": _auc(y, v2["xt_gk_v2"].to_numpy())},
        "raw_completion": {"auc": _auc(y, raw_completion)},
        "destination_xt": {"auc": _auc(y, _destination_only_v(test, v, pl, pressure_column))},
        "v1_composite": {"auc": _auc(y, _v1_composite(test, frames))},
        "_note": "V == expected first-shot xG; target == possession-reaches-shot -> partial "
                 "circularity. Read LIFT over baselines, not absolute AUC. Out-of-sample by "
                 "possession-parity split. WC2018/Neuer repro: TODO (needs Jeff's old data).",
    }
```

Verify `tracking._xt_gk.add_xt_gk` is the correct v1 aggregator name during implementation (it is
the frame-aware public aggregator); if the owner-run prefers `compute_xt_gk`, adjust the `_v1_composite`
call — this branch is owner-run only (frames present) and does not affect the CI smoke.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/xtgk/test_validate_v2_smoke.py -q`
Expected: PASS.

- [ ] **Step 5: Checkpoint**

Run: `python -m pytest tests/xtgk/ -m "not e2e" -q`
Expected: PASS.

---

### Task E2: ADR-036 amendment + NOTICE attribution

**Files:**
- Modify: `docs/superpowers/adrs/ADR-036-xtgk-possession-value-surface.md`
- Modify: `NOTICE`

- [ ] **Step 1: Amend ADR-036**

Append a "v2 completion (2026-07-09)" section documenting: the `RetentionModel` + `TurnoverCost` ports; the `retains` label construct + the ECE≤0.10 / |slope−1|≤0.25 calibration gate (noting ADR-011 does NOT govern this trained-light class per ADR-024); the four-term decomposition + `xt_gk_v2_*` namespacing vs frozen v1; the three-rung gate ladder + gate-enforced `relative_effect_floor`; the v1-freeze end-state (M5).

- [ ] **Step 2: NOTICE attribution**

Add the Eyestone xT-GK v2 methodology + Singh 2018 xT lineage entries under "Mathematical / Methodological References" (mirror the SP1 entry).

- [ ] **Step 3: Checkpoint (docs example gate if present)**

Run: `python -m pytest tests/ -m "not e2e and not slow" -q -k "notice or attribution or examples"`
Expected: PASS (or no tests collected — acceptable).

---

### Task E3: CLAUDE.md + M5 freeze note

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add the v2 bullet**

Extend the `xtgk/` architecture bullet with the SP2–SP5 surfaces (V_opp/ρ/metric/validation), the `xt_gk_v2_*` column set, the three-rung gate ladder, and the v1-frozen-alongside-v2 M5 end-state (v1 removed no earlier than one release after the lakehouse migrates). Note the two "xt-gk" homes: v1 `tracking/_xt_gk.py` (frozen) and v2 `silly_kicks/xtgk/`.

- [ ] **Step 2: Checkpoint**

Run: `python -m pytest tests/ -m "not e2e and not slow" -q -k "claude_md or c4"`
Expected: PASS (or none collected).

---

### Task E4: Full suite + lint + typecheck + `/final-review`

**Files:** none (verification)

- [ ] **Step 1: Full test suite (no e2e)**

Run: `python -m pytest tests/ -m "not e2e" -q`
Expected: PASS.

- [ ] **Step 2: Lint + format check**

Run: `ruff check silly_kicks/ scripts/ tests/ && ruff format --check silly_kicks/ scripts/ tests/`
Expected: clean (fix any findings, re-run).

- [ ] **Step 3: Typecheck (whole repo, per project rule)**

Run: `pyright`
Expected: 0 errors. (Fix per the pyright playbook — Literal propagation for `PressureLevel`, `float(Scalar)`, etc.)

- [ ] **Step 4: Run `/final-review` (mandatory, incl. C4 Phase 4)**

Invoke the `mad-scientist-skills:final-review` skill. Address findings. C4 count stays 28 (no new action-coupled aggregator — the v2 metric is an injected assembler, not an `add_*`).

---

### Task E5: Version bump + single commit + PR

**Files:**
- Modify: `pyproject.toml`, `silly_kicks/__init__.py`, `TODO.md`, `CHANGELOG` (+ `uv.lock` if present)

- [ ] **Step 1: Bump the version in lockstep**

Set the new version (e.g. `4.42.0`) in `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG`, and remove the shipped TF/SP items from `TODO.md` (delete, don't strikethrough). Verify all match.

Run: `python -c "import silly_kicks; print(silly_kicks.__version__)"` and `grep -n 'version' pyproject.toml`
Expected: identical versions.

- [ ] **Step 2: Stage + single commit**

```bash
git add -A
git commit -m "feat(xtgk): xT-GK v2 completion -- gate loader + V_opp + ρ + metric + validation -- silly-kicks 4.42.0 (ADR-036, PR-S109)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 3: Push + open the PR (only on explicit owner approval)**

```bash
git push -u origin pr-s109-xtgk-v2-completion
gh pr create --fill --base main
```

- [ ] **Step 4: Do NOT tag before main CI is green.** After merge + green CI, tag per the release policy.

---

## Self-Review (run after writing; issues fixed inline)

**Spec coverage:**
- Part 1 gate: A1 (relative-effect B2), A2–A3 (zone-conditional §1c), A4 (three-rung ladder), A5 (both orientations), A6 (loader + RM-provisional H2). ✅
- Part 2 V_opp: B1 (mirror_zone), B2 (port + MirroredTurnoverCost), B3 (_is_turnover), B4 (EmpiricalTurnoverValue), B5 (deep-loss property). ✅
- Part 3 ρ: C1 (H3 extract ece/slope), C2 (port), C3 (retains label), C4 (features), C5 (model), C6 (trainer + calibration gate), C7 (weights), C8 (exports). ✅
- Part 4 metric: D1 (assembler + four-term partition + κ), D2 (exports + v1/xthreat byte-stability). ✅
- Part 5 validation: E1 (construct validity + transfer + calibration; Neuer stub), E2 (ADR/NOTICE), E3 (CLAUDE.md M5), E4 (final-review), E5 (single commit + version bump). ✅

**Placeholder scan:** owner-run training (C7 Step 3) and the validate `__main__` (E1) are intentionally owner-run seams (real restricted-cohort data, uncommittable) with the CI-testable core fully specified — not placeholders. No "TBD"/"handle edge cases".

**Type consistency:** `PressureLevels.apply(pressure, *, zones=None)` used identically in `_markov`, `_empirical`, `_diagnostics._fit_pair`, `compute_xt_gk_v2`. `mirror_zone(zone, l, w)` signature consistent across `_turnover` + tests. `GkRetentionModel.predict_proba(features)->ndarray` matches the `RetentionModel` port. `compute_xt_gk_v2` column names match `_OUTPUT_COLS` and the DoD.

**One open sub-question surfaced for Eyestone (non-blocking, spec §1c/§4c):** whether zone-conditional should be the *primary* deep-gate mode (default: fallback rung) and confirmation of the PEV/DZV/RAV↔four-term acronym mapping. Both are flagged in the ADR amendment (E2); neither blocks the release.

**Review round 1 (2026-07-09) — resolved:** blocking E1 (mixed-class cohort + out-of-sample split + 4 baselines + circularity note), A4 (deep-low-pressure fixture asserting `rung=="zone_conditional"`), D1 (reuse V's `pressure_levels`, thread zones, require frames-derived `retention_features`); high B4 (bounded post-turnover scan), D1 vectorized zones; minor `retains` truncation→NaN + C6 NaN-drop, D2 v2/v1 column-disjointness, A2 `apply` fitted-before-isna order. Verified `resolve_gk_geometry(frames=None)` is accepted (signature `frames: pd.DataFrame | None`; the completion trainer already calls it with `None`), so `extract_retention_features(frames=None)` and the C4/C6 smokes hold.
