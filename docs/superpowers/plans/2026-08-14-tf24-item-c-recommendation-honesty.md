# TF-24 item C — recommendation honesty — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make TF-24's Stage-1 output honest — emit the indistinguishable `beta`/`gamma` set under a prefer-incumbent rule, and turn `tolerance_m` into a held constant with zero swept/recommended/consumed representation.

**Architecture:** Pure decision logic (`select_recommended_point`, `exceeds_noise_floor`, `build_selection_artifact`) lives in `silly_kicks/calibration/`; the confirmation script (`check_stage1_argmax.py`) and the calibration CLI (`calibrate_tracking_defaults.py`) keep all I/O, streaming, and provenance. `tolerance_m` is removed from the Stage-1 sweep, excluded from every recommendation artifact, and sourced from `DEFAULT_CARRIER_PARAMS` by Stage 2.

**Tech Stack:** Python 3.10–3.14, numpy, pandas, pytest; ruthless-efficiency (Optuna `OptunaConfig`/`Candidate`); the TF-24 calibration harness (`silly_kicks.calibration`).

## Global Constraints

- **TF-24 recommends only; never changes library constants** (ADR-009). `DEFAULT_CARRIER_PARAMS` (`silly_kicks/tracking/_ball_carrier.py:32`, `{"tolerance_m": 3.0, "beta": 0.0, "gamma": 0.25}`) is untouched — it is the single source of truth for the held radius.
- **No fresh Optuna sweep.** The confirmation reuses the existing Stage-1 store's `beta`/`gamma` trials.
- **Both bars are required to move** the recommendation: `gain > min_effect_size` (strict) AND `exceeds_noise_floor(gain, paired_se)`.
- **`min_effect_size` (δ) is frozen** — derived once (Task 8), recorded with corpus + rationale, and the landed keep-incumbent result asserted invariant to δ across `[δ_lo, δ_hi]`.
- **Provenance:** `carrier_selected.json` and `metrics.json` land under `docs/research/tf24_stage1_confirmation/` and MUST carry top-level `run_commit` + `run_tree_dirty: False` (the structural output gate `tests/scripts/test_artifact_provenance_output.py` globs `docs/research/**/*.json`).
- **Lint/type/test at CI scope:** `python -m ruff check silly_kicks/ tests/ scripts/`, `python -m ruff format --check ...`, `python -m pyright` (bare), `python -m pytest tests/ -m "not e2e" --benchmark-skip`. Tools are not on PATH — use `python -m`.
- **ADR-060 stays `Proposed`** until Task 8's two pre-land items (store reconciliation, δ derivation + robustness) resolve.
- **No breaking-change avoidance** — the owner has confirmed breaking changes are acceptable (the Stage-2 carrier-file contract `{tolerance_m, beta, gamma}` → `{beta, gamma}` is a deliberate break).

---

## File Structure

- `silly_kicks/calibration/_diagnostics.py` (modify) — add `exceeds_noise_floor`; refactor `tf25_gate_fires` onto it.
- `silly_kicks/calibration/_selection.py` (create) — `PointScore`, `Selection`, `select_recommended_point`, `build_selection_artifact`, `MIN_EFFECT_SIZE`.
- `silly_kicks/calibration/__init__.py` (modify) — export the new public symbols.
- `silly_kicks/calibration/_spaces.py` (modify) — drop `tolerance_m` from `stage1_config`.
- `silly_kicks/calibration/_carrier_objective.py` (modify:186) — default `tolerance_m` from the constant.
- `scripts/calibrate_tracking_defaults.py` (modify) — writer `{beta, gamma}`; Stage-2 `{beta, gamma}` + provenance validation.
- `scripts/check_stage1_argmax.py` (modify) — the confirmation redesign.
- Tests: `tests/calibration/test_diagnostics.py`, `tests/calibration/test_selection.py` (create), `tests/calibration/test_spaces.py`, `tests/calibration/test_carrier_objective.py`, `tests/calibration/test_calibrate_cli.py`, `tests/scripts/test_check_stage1_argmax.py`.

---

## Task 1: `exceeds_noise_floor` primitive + refactor `tf25_gate_fires`

**Files:**
- Modify: `silly_kicks/calibration/_diagnostics.py`
- Modify: `silly_kicks/calibration/__init__.py`
- Test: `tests/calibration/test_diagnostics.py`

**Interfaces:**
- Produces: `exceeds_noise_floor(gain: float, se: float) -> bool` — True iff `se` is finite and `gain > se`; `None`/`nan`/`inf` SE → False.

- [ ] **Step 1: Write the failing tests**

```python
# tests/calibration/test_diagnostics.py  (add)
import math
import pytest
from silly_kicks.calibration._diagnostics import exceeds_noise_floor, tf25_gate_fires

@pytest.mark.parametrize("se", [None, math.nan, math.inf])
def test_exceeds_noise_floor_non_finite_se_never_clears(se):
    assert exceeds_noise_floor(1.0, se) is False

def test_exceeds_noise_floor_strict_boundary():
    assert exceeds_noise_floor(0.06, 0.05) is True
    assert exceeds_noise_floor(0.05, 0.05) is False  # strict >

@pytest.mark.parametrize("se", [0.005, math.nan, math.inf])
def test_tf25_gate_verdict_unchanged_across_finite_nan_inf(se):
    # RED-first: pin the observable verdict is identical after the refactor.
    # gap = 0.06 - 0.05 = 0.01; finite se 0.005 -> True; nan/inf -> False.
    expected = (se == 0.005)
    assert tf25_gate_fires(global_brier=0.06, provider_best_brier=0.05, provider_se=se) is expected
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/calibration/test_diagnostics.py -k "noise_floor or verdict_unchanged" -v`
Expected: FAIL — `exceeds_noise_floor` not defined.

- [ ] **Step 3: Implement `exceeds_noise_floor` and refactor `tf25_gate_fires`**

```python
# silly_kicks/calibration/_diagnostics.py  (replace the body of tf25_gate_fires; add exceeds_noise_floor)
import math

def exceeds_noise_floor(gain: float, se: float) -> bool:
    """True iff `se` is finite and `gain` strictly exceeds it.

    A `None`/`nan`/`inf` SE (single fold, or an undefined spread) never clears the floor: a gain
    "beats the noise" only when the noise is a finite, measured quantity.
    """
    if se is None or not math.isfinite(se):
        return False
    return gain > se


def tf25_gate_fires(*, global_brier: float, provider_best_brier: float, provider_se: float) -> bool:
    """True if the (global - provider-best) Brier gap exceeds the provider's CV SE.

    Examples
    --------
    >>> from silly_kicks.calibration._diagnostics import tf25_gate_fires
    >>> tf25_gate_fires(global_brier=0.052, provider_best_brier=0.050, provider_se=0.005)
    False
    """
    return exceeds_noise_floor(global_brier - provider_best_brier, provider_se)
```

Note the inf-handling unification: the old body guarded with `math.isnan` (which permitted `inf` and then returned `gap > inf == False`); `exceeds_noise_floor` rejects `inf` up front. The verdict is identical — the Step-1 test pins it.

- [ ] **Step 4: Export the symbol**

```python
# silly_kicks/calibration/__init__.py  — add to the import line and __all__
from silly_kicks.calibration._diagnostics import exceeds_noise_floor, tf25_gate_fires  # exceeds_noise_floor is new
# ... and add "exceeds_noise_floor" to __all__
```

- [ ] **Step 5: Run to verify pass + no regression**

Run: `python -m pytest tests/calibration/test_diagnostics.py -v`
Expected: PASS (including any pre-existing `tf25_gate_fires` tests).

---

## Task 2: `_selection.py` — `PointScore`, `Selection`, `select_recommended_point`

**Files:**
- Create: `silly_kicks/calibration/_selection.py`
- Modify: `silly_kicks/calibration/__init__.py`
- Test: `tests/calibration/test_selection.py` (create)

**Interfaces:**
- Consumes: `exceeds_noise_floor` (Task 1); `cv_standard_error(list[float]) -> float` (`silly_kicks/calibration/_cv.py:65`).
- Produces:
  - `PointScore(label: str, params: dict, per_fold: tuple[float, ...], mean: float)` — frozen dataclass.
  - `Selection(selected, incumbent, moved: bool, reason: str, best_candidate, effect_size, paired_se)` — frozen dataclass.
  - `select_recommended_point(*, incumbent: PointScore, candidates: list[PointScore], min_effect_size: float, policy: str = "prefer_incumbent") -> Selection`.
  - `MIN_EFFECT_SIZE: float` — provisional δ (frozen value derived in Task 8).

- [ ] **Step 1: Write the failing tests**

```python
# tests/calibration/test_selection.py  (create)
import pytest
from silly_kicks.calibration._selection import PointScore, select_recommended_point

def _pt(label, per_fold, beta=0.0, gamma=0.25):
    per_fold = tuple(per_fold)
    return PointScore(label=label, params={"beta": beta, "gamma": gamma},
                      per_fold=per_fold, mean=sum(per_fold) / len(per_fold))

_INC = _pt("shipped", [0.79, 0.80, 0.81])  # mean 0.80

def test_incumbent_kept_when_no_candidate_clears():
    cand = _pt("c", [0.8001, 0.8001, 0.8001])  # gain 1e-4 < δ
    sel = select_recommended_point(incumbent=_INC, candidates=[cand], min_effect_size=0.01)
    assert sel.moved is False and sel.selected is _INC

def test_incumbent_replaced_when_candidate_clears_both():
    cand = _pt("c", [0.84, 0.85, 0.86])  # gain 0.05 > δ; consistent diff -> clears paired SE
    sel = select_recommended_point(incumbent=_INC, candidates=[cand], min_effect_size=0.01)
    assert sel.moved is True and sel.selected is cand
    assert sel.effect_size == pytest.approx(0.05)

def test_gain_exactly_at_delta_is_kept_strict():
    # gain lands EXACTLY on δ via exact binary fractions -> strict '>' keeps; a regression to '>=' MOVES.
    # (0.79/0.80/0.81 literals compute gain 0.00999...898 < δ, so they never exercise the boundary.)
    inc = _pt("shipped", [0.5, 0.5, 0.5])   # mean 0.5
    cand = _pt("c", [0.75, 0.75, 0.75])     # mean 0.75; gain exactly 0.25 == δ
    sel = select_recommended_point(incumbent=inc, candidates=[cand], min_effect_size=0.25)
    assert sel.moved is False

def test_effect_size_floor_is_load_bearing_both_sides():
    # SAME candidate, two δ: high δ keeps (δ blocks); low δ moves. The MOVE proves the paired-SE bar
    # cleared, so δ was the sole blocker -- the real from-both-sides "floor is load-bearing" assertion.
    cand = _pt("c", [0.792, 0.803, 0.815])  # gain ~0.00333, tiny consistent diff -> clears paired SE
    assert select_recommended_point(incumbent=_INC, candidates=[cand], min_effect_size=0.01).moved is False
    moved = select_recommended_point(incumbent=_INC, candidates=[cand], min_effect_size=0.001)
    assert moved.moved is True and moved.selected is cand

def test_paired_se_exactly_zero_is_recorded_on_the_moved_branch():
    # exactly-representable folds -> per-fold diff is exactly 0.25 -> paired_se == 0.0 (no float noise).
    inc = _pt("shipped", [0.5, 0.625, 0.75])   # mean 0.625
    cand = _pt("c", [0.75, 0.875, 1.0])        # mean 0.875; diffs all exactly 0.25
    sel = select_recommended_point(incumbent=inc, candidates=[cand], min_effect_size=0.1)  # gain 0.25 > δ
    assert sel.moved is True
    assert sel.paired_se == 0.0  # recorded on the moved branch -> the exact-zero SE path is exercised

def test_two_candidates_both_clear_best_gain_wins():
    c1 = _pt("c1", [0.83, 0.84, 0.85])  # gain 0.04
    c2 = _pt("c2", [0.87, 0.88, 0.89])  # gain 0.08
    sel = select_recommended_point(incumbent=_INC, candidates=[c1, c2], min_effect_size=0.01)
    assert sel.moved is True and sel.selected is c2

def test_recorded_optimum_is_a_candidate_distinct_from_incumbent():
    recorded = _pt("recorded_optimum", [0.84, 0.85, 0.86], beta=1.9e-4, gamma=0.221)
    neighbour = _pt("nb0", [0.805, 0.81, 0.815], beta=0.1, gamma=0.3)
    sel = select_recommended_point(incumbent=_INC, candidates=[recorded, neighbour], min_effect_size=0.01)
    assert sel.moved is True and sel.selected is recorded

def test_fold_length_mismatch_raises():
    cand = _pt("c", [0.85, 0.85])  # 2 folds vs incumbent's 3
    with pytest.raises(ValueError, match="per_fold length"):
        select_recommended_point(incumbent=_INC, candidates=[cand], min_effect_size=0.01)

def test_unknown_policy_raises():
    with pytest.raises(ValueError, match="unknown policy"):
        select_recommended_point(incumbent=_INC, candidates=[], min_effect_size=0.01, policy="argmax")
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/calibration/test_selection.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `_selection.py` (selection logic only)**

```python
# silly_kicks/calibration/_selection.py  (create)
"""TF-24 Stage-1 recommendation selection (ADR-060).

Pure decision logic: given the CV-fold scores of the shipped incumbent and a set of candidate
points, decide whether to move the recommendation. `beta`/`gamma` are non-identifiable, so the rule
is prefer-incumbent under TWO bars -- a practical effect-size floor AND a paired-difference-SE
significance test -- both required to move. `tolerance_m` is NOT part of this decision (held constant;
see ADR-060). I/O lives in the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

from silly_kicks.calibration._cv import cv_standard_error
from silly_kicks.calibration._diagnostics import exceeds_noise_floor

#: Provisional practical-significance floor on carrier accuracy. The FROZEN value is derived from
#: Stage-2 Brier sensitivity in the plan's Task 8, and the landed keep-incumbent result is asserted
#: invariant to it across a plausible range -- so the result does not hinge on this exact number.
MIN_EFFECT_SIZE: float = 0.005


@dataclass(frozen=True)
class PointScore:
    """A parameter point's per-CV-fold carrier accuracy. `per_fold` is aligned by fold index across
    every point (all points are scored on the SAME folds), which is what makes the paired SE valid."""

    label: str
    params: dict
    per_fold: tuple[float, ...]
    mean: float


@dataclass(frozen=True)
class Selection:
    selected: PointScore
    incumbent: PointScore
    moved: bool
    reason: str
    best_candidate: PointScore | None
    effect_size: float | None
    paired_se: float | None


def select_recommended_point(
    *,
    incumbent: PointScore,
    candidates: list[PointScore],
    min_effect_size: float = MIN_EFFECT_SIZE,
    policy: str = "prefer_incumbent",
) -> Selection:
    """Prefer-incumbent selection. Returns the incumbent unless some candidate clears BOTH the
    effect-size floor (strict `gain > min_effect_size`) and the paired-difference-SE test."""
    if policy != "prefer_incumbent":
        raise ValueError(f"unknown policy {policy!r}; only 'prefer_incumbent' is implemented")

    clearing: list[tuple[PointScore, float, float]] = []
    for c in candidates:
        if len(c.per_fold) != len(incumbent.per_fold):
            raise ValueError(
                "per_fold length mismatch: candidate and incumbent must be scored on the same folds "
                f"({len(c.per_fold)} vs {len(incumbent.per_fold)})"
            )
        gain = c.mean - incumbent.mean
        paired_se = cv_standard_error([a - b for a, b in zip(c.per_fold, incumbent.per_fold)])
        if gain > min_effect_size and exceeds_noise_floor(gain, paired_se):
            clearing.append((c, gain, paired_se))

    if not clearing:
        return Selection(
            selected=incumbent, incumbent=incumbent, moved=False,
            reason="no candidate cleared both the effect-size floor and the paired-SE test",
            best_candidate=None, effect_size=None, paired_se=None,
        )
    # Deterministic tie-break: highest gain, then label, so equal gains do not depend on input order.
    best, gain, paired_se = max(clearing, key=lambda t: (t[1], t[0].label))
    return Selection(
        selected=best, incumbent=incumbent, moved=True,
        reason=f"candidate {best.label!r} cleared both bars (gain {gain:.6g} > δ, paired_se {paired_se:.6g})",
        best_candidate=best, effect_size=gain, paired_se=paired_se,
    )
```

- [ ] **Step 4: Export the public symbols**

```python
# silly_kicks/calibration/__init__.py  — add
from silly_kicks.calibration._selection import (
    MIN_EFFECT_SIZE, PointScore, Selection, select_recommended_point,
)
# add "MIN_EFFECT_SIZE", "PointScore", "Selection", "select_recommended_point" to __all__
```

- [ ] **Step 5: Run to verify pass**

Run: `python -m pytest tests/calibration/test_selection.py -v`
Expected: PASS (all nine).

---

## Task 3: `build_selection_artifact` (pure; provenance passed in)

**Files:**
- Modify: `silly_kicks/calibration/_selection.py`
- Modify: `silly_kicks/calibration/__init__.py`
- Test: `tests/calibration/test_selection.py`

**Interfaces:**
- Produces: `build_selection_artifact(selection: Selection, *, provenance: dict) -> dict` — a JSON-ready dict carrying `{beta, gamma, moved, reason, run_commit, run_tree_dirty}` and **no `tolerance_m`**. Pure: `provenance` is an argument, never read from git internally.

- [ ] **Step 1: Write the failing tests**

```python
# tests/calibration/test_selection.py  (add)
from silly_kicks.calibration._selection import Selection, build_selection_artifact

def _sel(beta=0.0, gamma=0.25, extra_params=None):
    params = {"beta": beta, "gamma": gamma}
    if extra_params:
        params.update(extra_params)
    pt = PointScore(label="shipped", params=params, per_fold=(0.8, 0.8, 0.8), mean=0.8)
    return Selection(selected=pt, incumbent=pt, moved=False, reason="kept",
                     best_candidate=None, effect_size=None, paired_se=None)

def test_artifact_carries_beta_gamma_and_provenance_no_tolerance_m():
    # the selected point carries a stray tolerance_m; the artifact must NOT surface it
    art = build_selection_artifact(_sel(extra_params={"tolerance_m": 8.0}),
                                   provenance={"commit": "abc123", "dirty": False})
    assert art["beta"] == 0.0 and art["gamma"] == 0.25
    assert "tolerance_m" not in art
    assert art["run_commit"] == "abc123"
    assert art["run_tree_dirty"] is False

def test_artifact_from_a_dirty_tree_carries_true_and_would_fail_the_output_gate():
    # other side: the structural gate asserts `run_tree_dirty is False`, so a dirty artifact is rejected
    art = build_selection_artifact(_sel(), provenance={"commit": "abc123", "dirty": True})
    assert art["run_tree_dirty"] is True
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/calibration/test_selection.py -k artifact -v`
Expected: FAIL — `build_selection_artifact` not defined.

- [ ] **Step 3: Implement**

```python
# silly_kicks/calibration/_selection.py  (add)
def build_selection_artifact(selection: Selection, *, provenance: dict) -> dict:
    """The committed `carrier_selected.json` payload. PURE -- provenance is passed in, never read
    from git here, so the builder is unit-testable and the caller owns the I/O.

    Carries `{beta, gamma}` (never `tolerance_m` -- held constant, ADR-060) plus the run provenance
    the structural output gate (`tests/scripts/test_artifact_provenance_output.py`) requires.
    """
    return {
        "beta": float(selection.selected.params["beta"]),
        "gamma": float(selection.selected.params["gamma"]),
        "moved": selection.moved,
        "reason": selection.reason,
        "run_commit": provenance["commit"],
        "run_tree_dirty": provenance["dirty"],
    }
```

- [ ] **Step 4: Export**

```python
# silly_kicks/calibration/__init__.py  — add build_selection_artifact to the _selection import and __all__
```

- [ ] **Step 5: Run to verify pass**

Run: `python -m pytest tests/calibration/test_selection.py -v`
Expected: PASS.

---

## Task 4: Remove `tolerance_m` from the Stage-1 sweep source

**Files:**
- Modify: `silly_kicks/calibration/_spaces.py` (the `stage1_config` `param_space` + `warm_start`)
- Modify: `silly_kicks/calibration/_carrier_objective.py:186`
- Modify: `scripts/calibrate_tracking_defaults.py:345`
- Test: `tests/calibration/test_spaces.py`, `tests/calibration/test_carrier_objective.py`

**Interfaces:**
- Consumes: `DEFAULT_CARRIER_PARAMS` (`silly_kicks/tracking/_ball_carrier.py:32`).
- Produces: `stage1_config(...).param_space == {"beta", "gamma"}`; `CarrierAccuracyObjective.evaluate` accepts a params dict without `tolerance_m` and defaults it to `3.0`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/calibration/test_spaces.py  (add)
from silly_kicks.calibration._spaces import stage1_config

def test_stage1_config_does_not_sweep_tolerance_m():
    cfg = stage1_config(n_trials=1, store_path=":memory:")
    assert set(cfg.param_space) == {"beta", "gamma"}
    assert "tolerance_m" not in cfg.warm_start
```

```python
# tests/calibration/test_carrier_objective.py  (add)
from ruthless import Candidate
from silly_kicks.calibration._carrier_objective import CarrierAccuracyObjective

def test_evaluate_defaults_tolerance_m_when_absent(synth_unreachable_actor):
    """No tolerance_m in params -> the objective uses DEFAULT_CARRIER_PARAMS (3.0), so the 5 m
    'unreachable actor' still misses and the match scores 0.5 -- and it must NOT KeyError."""
    obj = CarrierAccuracyObjective(synth_unreachable_actor)
    m = obj.evaluate(Candidate(id="t", params={"beta": 0.0, "gamma": 0.25}))
    assert m["carrier_accuracy"] == 0.5
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/calibration/test_spaces.py -k tolerance_m tests/calibration/test_carrier_objective.py -k defaults -v`
Expected: FAIL — config still lists `tolerance_m`; `evaluate` KeyErrors on `p["tolerance_m"]`.

- [ ] **Step 3: Edit `stage1_config`**

In `silly_kicks/calibration/_spaces.py`, remove the `tolerance_m` entries:

```python
        param_space={
            "beta": FloatRange(kind="float", lo=0.0, hi=2.0),
            "gamma": FloatRange(kind="float", lo=0.0, hi=3.0),
        },
        warm_start={"beta": 0.5, "gamma": 1.0},
```

Update the `stage1_config` docstring line "carrier accuracy (maximize): tolerance_m, beta, gamma" → "carrier accuracy (maximize): beta, gamma (tolerance_m is held at DEFAULT_CARRIER_PARAMS — under-determined by this objective, ADR-060)".

- [ ] **Step 4: Default `tolerance_m` in the objective**

In `silly_kicks/calibration/_carrier_objective.py`, add the import and change line 186:

```python
from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS
# ...
        tolerance_m = float(p.get("tolerance_m", DEFAULT_CARRIER_PARAMS["tolerance_m"]))
        beta, gamma = float(p["beta"]), float(p["gamma"])
```

- [ ] **Step 5: Edit the `carrier_best.json` writer**

In `scripts/calibrate_tracking_defaults.py:345`, drop `tolerance_m` so the writer does not `KeyError` once the winning trial has no radius:

```python
        best_carrier = {k: result.best.candidate.params[k] for k in ("beta", "gamma")}
```

- [ ] **Step 6: Run to verify pass**

Run: `python -m pytest tests/calibration/test_spaces.py tests/calibration/test_carrier_objective.py -v`
Expected: PASS. Also run the existing `test_carrier_objective.py` suite to confirm the `_P` fixtures that DO pass `tolerance_m` still work (the `.get` default is only used when it is absent).

---

## Task 5: Stage 2 consumes `{beta, gamma}`, sources `tolerance_m`, validates provenance

**Files:**
- Modify: `scripts/calibrate_tracking_defaults.py` (the `--stage 2` block, `:325–331`)
- Test: `tests/calibration/test_calibrate_cli.py`

**Interfaces:**
- Consumes: `build_selection_artifact` output shape (`{beta, gamma, run_commit, run_tree_dirty}`); `DEFAULT_CARRIER_PARAMS`.
- Produces: Stage 2's `carrier_params == {"tolerance_m": 3.0, "beta": ..., "gamma": ...}`; a fail-closed refusal of a carrier file that is missing provenance or was produced from a dirty tree.

- [ ] **Step 1: Write the failing tests**

```python
# tests/calibration/test_calibrate_cli.py  (add) -- unit-level, exercises the loader helper only
import json
import pytest
from scripts.calibrate_tracking_defaults import _load_carrier_selection  # new helper (Step 3)

def test_carrier_selection_sources_tolerance_m_and_validates(tmp_path):
    f = tmp_path / "carrier_selected.json"
    f.write_text(json.dumps({"beta": 0.0, "gamma": 0.25, "run_commit": "abc", "run_tree_dirty": False}))
    params = _load_carrier_selection(f)
    assert params == {"tolerance_m": 3.0, "beta": 0.0, "gamma": 0.25}

def test_carrier_selection_refuses_missing_provenance(tmp_path):
    f = tmp_path / "carrier_selected.json"
    f.write_text(json.dumps({"beta": 0.0, "gamma": 0.25}))  # no run_commit
    with pytest.raises(ValueError, match="provenance"):
        _load_carrier_selection(f)

def test_carrier_selection_refuses_dirty_upstream(tmp_path):
    f = tmp_path / "carrier_selected.json"
    f.write_text(json.dumps({"beta": 0.0, "gamma": 0.25, "run_commit": "abc", "run_tree_dirty": True}))
    with pytest.raises(ValueError, match="dirty"):
        _load_carrier_selection(f)
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/calibration/test_calibrate_cli.py -k carrier_selection -v`
Expected: FAIL — `_load_carrier_selection` not defined.

- [ ] **Step 3: Extract the helper and rewire the Stage-2 block**

Add the pure-ish loader (I/O is the file read; logic is testable) to `scripts/calibrate_tracking_defaults.py`:

```python
def _load_carrier_selection(path) -> dict:
    """Read the Stage-1 selection artifact and build Stage-2's carrier params.

    `tolerance_m` is NOT read from the file -- it is a held constant sourced from
    DEFAULT_CARRIER_PARAMS (ADR-060). The upstream artifact must carry clean provenance, or Stage 2
    refuses it (missing manifest == dirty; the corpus-driver family rule).
    """
    from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS

    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    missing = {"beta", "gamma"} - set(data)
    if missing:
        raise ValueError(f"{path}: carrier selection missing keys {sorted(missing)} -- run Stage 1 first")
    if not data.get("run_commit"):
        raise ValueError(f"{path}: no run_commit provenance -- an unprovenanced upstream is treated as dirty")
    if data.get("run_tree_dirty") is not False:
        raise ValueError(f"{path}: upstream run_tree_dirty={data.get('run_tree_dirty')!r} -- refusing a dirty selection")
    return {"tolerance_m": DEFAULT_CARRIER_PARAMS["tolerance_m"], "beta": float(data["beta"]), "gamma": float(data["gamma"])}
```

Then replace the Stage-2 read in `main()` (`:327–331`):

```python
    if args.stage == "2":
        xt = _resolve_xt(args, fold, used_ids)
        carrier_params = _load_carrier_selection(args.carrier_best)
```

Also update the `--carrier-best` argparse help (`:276`) to name the SELECTION artifact, so an operator does not pass the raw `carrier_best.json` out of habit: `help="the Stage-1 SELECTION artifact carrier_selected.json (NOT the raw sweep best)"`.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/calibration/test_calibrate_cli.py -k carrier_selection -v`
Expected: PASS.

---

## Task 6: Redesign `check_stage1_argmax.py`

**Files:**
- Modify: `scripts/check_stage1_argmax.py`
- Test: `tests/scripts/test_check_stage1_argmax.py`

**Interfaces:**
- Consumes: `PointScore`, `select_recommended_point`, `build_selection_artifact`, `exceeds_noise_floor`, `MIN_EFFECT_SIZE` (Tasks 1–3).
- Produces: `augment_metrics(out: dict, *, provenance: dict, min_effect_size: float) -> tuple[dict, dict]` — AUGMENTS the existing confirmation `out` dict with `selection` + `fold_stability` blocks (never removes keys) and returns `(augmented_out, carrier_selected_dict)`; `carrier_selected.json` = `{beta, gamma}` + provenance, no `tolerance_m`.

- [ ] **Step 1: Write the failing test (synthetic `summary`, no frames)**

```python
# tests/scripts/test_check_stage1_argmax.py  (add)
from scripts.check_stage1_argmax import augment_metrics  # new non-lossy augmenter (Step 3)

def _summary():
    # shipped incumbent + two candidates, all within noise -> keep incumbent
    return {
        "shipped_point":    {"mean": 0.80, "se": 0.01, "per_fold": [0.79, 0.80, 0.81], "params": {"beta": 0.0, "gamma": 0.25}},
        "recorded_optimum": {"mean": 0.8001, "se": 0.01, "per_fold": [0.7901, 0.8001, 0.8101], "params": {"beta": 1.9e-4, "gamma": 0.221}},
        "nb0":              {"mean": 0.8002, "se": 0.01, "per_fold": [0.7902, 0.8002, 0.8102], "params": {"beta": 0.1, "gamma": 0.3}},
    }

def _base_out(summary):
    # a realistic confirmation `out`, carrying the Prong-1 invariance result + metadata that MUST survive
    return {
        "invariance": {"shipped_point": {"invariance_fraction": 0.9999, "verdict": "stands"}},
        "invariance_threshold": 0.999,
        "cv_scheme": "GroupKFold(5)",
        "objective": "CarrierAccuracyObjective.carrier_accuracy",
        "argmax_moved": False,
        "points": summary,
        "run_commit": "abc123",
        "run_tree_dirty": False,
    }

def test_augment_metrics_is_non_lossy_and_adds_selection_no_tolerance_m():
    base = _base_out(_summary())
    out, selected = augment_metrics(base, provenance={"commit": "abc123", "dirty": False}, min_effect_size=0.01)
    # NON-LOSSY (F1 regression guard): the invariance prong + all metadata survive
    assert out["invariance"] == base["invariance"]
    assert out["cv_scheme"] == "GroupKFold(5)" and out["objective"] == base["objective"]
    assert out["argmax_moved"] is False and out["run_commit"] == "abc123"
    # ADDED blocks
    assert out["selection"]["moved"] is False
    assert out["fold_stability"]["verdict"] == "no_discriminating_evidence"
    assert "fold_winners" in out["fold_stability"]             # §3.4 per-fold ranks
    assert "fold_to_point_var_ratio" in out["fold_stability"]  # §3.4 variance ratio
    # selection artifact: {beta, gamma} + provenance, NO tolerance_m
    assert selected["beta"] == 0.0 and selected["gamma"] == 0.25 and "tolerance_m" not in selected
    assert selected["run_commit"] == "abc123" and selected["run_tree_dirty"] is False

def test_fold_stability_verdict_flips_on_a_discriminating_fold_set():
    # non-vacuity (spec §6): a wide-separation, low-SE candidate MOVES the verdict off the incumbent
    summary = _summary()
    summary["nb0"] = {"mean": 0.91, "se": 0.005, "per_fold": [0.90, 0.91, 0.92], "params": {"beta": 0.1, "gamma": 0.3}}
    out, selected = augment_metrics(_base_out(summary), provenance={"commit": "abc", "dirty": False}, min_effect_size=0.01)
    assert out["selection"]["moved"] is True
    assert out["fold_stability"]["verdict"] == "moved"
    assert selected["beta"] == 0.1 and selected["gamma"] == 0.3  # the artifact follows the move
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/scripts/test_check_stage1_argmax.py -k "augment_metrics or verdict_flips" -v`
Expected: FAIL — `augment_metrics` not defined.

- [ ] **Step 3: Add the fold-stability diagnostic + the NON-LOSSY augmenter, and refactor `moved_beyond_noise`**

Hoist `import statistics` into the top-of-file import block (with `argparse`/`json`/`pathlib` — a mid-file import trips ruff **E402**), then add:

```python
def _fold_stability(summary: dict, selection) -> dict:
    """Per-fold winners (ranks) + between-fold vs between-point variance ratio + a verdict tied to the
    SAME selection, so the diagnostic and the recommendation cannot disagree (spec 3.4). The huge
    ratio on real data is the finding: fold noise dwarfs point differences."""
    labels = list(summary)
    per_fold = {name: list(summary[name]["per_fold"]) for name in labels}
    n_folds = len(per_fold[labels[0]])
    fold_winners = [max(labels, key=lambda name: per_fold[name][f]) for f in range(n_folds)]
    point_means = [summary[name]["mean"] for name in labels]
    fold_means = [statistics.fmean(per_fold[name][f] for name in labels) for f in range(n_folds)]
    between_point_var = statistics.pvariance(point_means) if len(point_means) > 1 else 0.0
    between_fold_var = statistics.pvariance(fold_means) if n_folds > 1 else 0.0
    return {
        "per_point_mean": {name: summary[name]["mean"] for name in labels},
        "n_folds": n_folds,
        "fold_winners": fold_winners,
        "n_distinct_fold_winners": len(set(fold_winners)),
        "between_fold_var": between_fold_var,
        "between_point_var": between_point_var,
        "fold_to_point_var_ratio": (between_fold_var / between_point_var) if between_point_var > 0 else None,
        "verdict": "moved" if selection.moved else "no_discriminating_evidence",
        "selection_reason": selection.reason,
    }


def augment_metrics(out: dict, *, provenance: dict, min_effect_size: float) -> tuple[dict, dict]:
    """AUGMENT the confirmation `out` dict with the selection + fold-stability blocks, and build the
    carrier_selected.json payload. NEVER replaces `out` -- the Prong-1 invariance result and every run
    metadatum survive (F1). Pure (provenance passed in). Returns (augmented_out, selected)."""
    from silly_kicks.calibration import PointScore, build_selection_artifact, select_recommended_point

    summary = out["points"]
    if "shipped_point" not in summary:
        raise KeyError("augment_metrics expects the shipped incumbent under 'shipped_point'")

    def _score(name: str) -> PointScore:
        s = summary[name]
        return PointScore(label=name, params=s["params"], per_fold=tuple(s["per_fold"]), mean=s["mean"])

    incumbent = _score("shipped_point")
    candidates = [_score(name) for name in summary if name != "shipped_point"]
    selection = select_recommended_point(incumbent=incumbent, candidates=candidates, min_effect_size=min_effect_size)
    selected = build_selection_artifact(selection, provenance=provenance)

    augmented = dict(out)  # shallow copy; ADD keys only, never remove
    augmented["selection"] = {
        "moved": selection.moved,
        "reason": selection.reason,
        "selected": {"beta": selected["beta"], "gamma": selected["gamma"]},
    }
    augmented["fold_stability"] = _fold_stability(summary, selection)
    return augmented, selected
```

Refactor `moved_beyond_noise` (`:93-108`) through the shared floor so 3.2's "one definition" is complete (`argmax_moved` stays a reported diagnostic; behaviour-preserving):

```python
def moved_beyond_noise(*, recorded, best_alternative, se, maximize=True):
    from silly_kicks.calibration import exceeds_noise_floor
    gain = (best_alternative - recorded) if maximize else (recorded - best_alternative)
    return exceeds_noise_floor(gain, se)
```

In `main()`, replace the metrics WRITE (`:445-447`) so the existing `out` dict (invariance + metadata intact) is AUGMENTED, not replaced, and `carrier_selected.json` is written alongside. Add `from silly_kicks.calibration import MIN_EFFECT_SIZE` at the top:

```python
    out, selected = augment_metrics(out, provenance=prov, min_effect_size=MIN_EFFECT_SIZE)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    (args.out / "carrier_selected.json").write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    print(f"selection moved={selected['moved']}; argmax_moved={out['argmax_moved']}")
```

- [ ] **Step 4: Correct the module docstring**

Replace the "`tolerance_m` … whether it SHOULD be swept is a live question" paragraph (`:29–32`) with the under-determination fence, cross-referencing `_ball_carrier.py`:

```python
# `tolerance_m` is HELD at 3.0 and is out of the recommendation BY CONSTRUCTION, not un-tuned: the
# carrier-actor objective has no loose-ball negatives, so it presses the radius to the search upper
# bound -- an artifact, not a value to apply (see `_ball_carrier.py` docstring + ADR-060). It is
# removed from the Stage-1 search space; this confirmation scores `beta`/`gamma` only.
```

- [ ] **Step 5: Run to verify pass**

Run: `python -m pytest tests/scripts/test_check_stage1_argmax.py -v`
Expected: PASS. The existing invariance/`for_each` tests are untouched — `main()` still builds the same `out` base and now AUGMENTS it, so the invariance prong survives (asserted by the non-lossy test in Step 1).

---

## Task 7: Local gates + 2-match smoke (pre-commit verification)

**Files:** none (verification only).

- [ ] **Step 1: Full lint + type at CI scope**

Run:
```
python -m ruff check silly_kicks/ tests/ scripts/
python -m ruff format --check silly_kicks/ tests/ scripts/
python -m pyright
```
Expected: clean; 0 errors.

- [ ] **Step 2: Full non-e2e suite**

Run: `python -m pytest tests/ -m "not e2e" --benchmark-skip -q`
Expected: PASS (prior baseline 7327 passed, plus the new tests; 0 failed).

- [ ] **Step 3: 2-match confirmation smoke (local or cached corpus)**

Run `check_stage1_argmax.py` with `--max-matches 2` against a cached 2-match store (a 1-match smoke is invalid — `match_cv_splits` does LOMO ≤7 and raises on a single group). Confirm it writes `metrics.json` + `carrier_selected.json`, both carrying `run_commit` + `run_tree_dirty`.

Expected: both files written; `carrier_selected.json` has no `tolerance_m` key.

---

## Task 8: DGX pre-land items + clean confirmation run + land artifacts

> **Owner-driven, runs on a COMMITTED clean tree** (the provenance drivers refuse a dirty tree; the artifacts must stamp `run_tree_dirty: false`). These steps are operational, not code-TDD. ADR-060 stays `Proposed` until Steps 1–2 resolve.

- [ ] **Step 1: Store reconciliation.** On the DGX, inspect the existing Stage-1 store's completed trials: confirm whether `tolerance_m` was swept or held, so the reused `beta`/`gamma` neighbours and the ADR's headline evidence (`argmax_moved=False`, the 1/40-SE spread) are interpreted against a known store. Record the finding in the confirmation artifact's notes.

- [ ] **Step 2: Derive, freeze, and de-risk δ.** Measure Stage-2 held-out Brier sensitivity to a carrier-accuracy shift; set δ (`MIN_EFFECT_SIZE`) to the smallest carrier-accuracy difference that produces a detectable Brier change, conservatively. Record the corpus + rationale beside the constant. Then **assert the keep-incumbent result is invariant to δ across `[δ_lo, δ_hi]`** (e.g. re-run `augment_metrics` on the real `out`/summary for a grid of δ and confirm `moved is False` throughout). If the frozen δ differs from the provisional `0.005`, that is a one-line constant edit (which the owner commits before the authoritative run).

- [ ] **Step 3: Authoritative confirmation run.** On the clean committed tree, run `check_stage1_argmax.py` over the held corpus. Verify `metrics.json` + `carrier_selected.json` carry `run_commit` + `run_tree_dirty: false`.

- [ ] **Step 4: Land the artifacts** under `docs/research/tf24_stage1_confirmation/`. The structural output gate (`tests/scripts/test_artifact_provenance_output.py`) will police them automatically (top-level `run_commit`, `run_tree_dirty is False`).

- [ ] **Step 5: Move ADR-060 to `Accepted`** once Steps 1–2 are recorded.

> After Task 8, the cycle's downstream phases follow (spec §5.6–5.7, out of this plan's scope): Stage 2 on `carrier_selected.json`, full gates, release.

---

## Self-Review

**Spec coverage:**
- §2/§3.1 selection (prefer-incumbent, two bars, incumbent-out-of-pool) → Task 2. §3.1 fold-length guard → Task 2. §3.2 `exceeds_noise_floor` + `tf25_gate_fires` inf-unification → Task 1; `moved_beyond_noise` routed through the same primitive → Task 6 Step 3 (completes "one definition"). §3.3 orchestration + `build_selection_artifact(+provenance)` + docstring → Tasks 3, 6. §3.4 fold-stability (per-fold ranks + between-fold/between-point variance ratio + verdict) → Task 6 `_fold_stability`, with the verdict-**flip** non-vacuity case tested in Task 6 Step 1. §3.5 not-swept/not-in-artifact/sourced-from-constant → Tasks 4 (source), 3 (artifact), 5 (consumer). §4 provenance + structural gate → Tasks 3, 5, 6; the metrics.json **non-lossiness** (invariance prong survives) is a Task 6 regression guard. §5 sequencing → Tasks 7, 8. §6 tests → Tasks 1–6, including BOTH sides of the effect-size floor (δ-parametrized) and the exact-zero paired-SE branch (Task 2). §7 store reconciliation + δ derivation/freeze/invariance → Task 8.
- No spec requirement is left without a task.

**Placeholder scan:** `MIN_EFFECT_SIZE = 0.005` is a real provisional value the code runs with, refined+frozen in Task 8 (not a TBD). δ is passed explicitly in every test, so tests are deterministic regardless of the frozen value. No "TODO/fill-in" steps.

**Type consistency:** `select_recommended_point` / `PointScore` / `Selection` / `build_selection_artifact(selection, *, provenance)` / `MIN_EFFECT_SIZE` are named identically in their defining task (2/3) and every consuming task (5, 6). `_load_carrier_selection` (Task 5) and `augment_metrics(out, *, provenance, min_effect_size)` (Task 6) are defined and consumed within their own tasks. `exceeds_noise_floor(gain, se)` signature matches across Tasks 1, 2, and 6.
