# xT-GK per-type base-rate serve switch — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a completion variant's per-type held-out AUC can't beat chance with confidence (lower-confidence-bound ≤ 0.5, or degenerate/too-small sample), serve the per-type calibrated base rate for that type (tagged `xt_gk_completion_source = "base_rate"`) instead of the geometric model — primarily fixing goal-kicks (skillcorner goal-kick AUC 0.433).

**Architecture:** A pure gate-decision function (`serve_mode_from_lcb`) + a per-type serve-mode field on `GkCompletionModel` (computed at train time from held-out CV, fail-open on load) + a serve switch in `compute_xt_gk` that reads the gate, overrides the RAV `p` with the base rate for gated types, and tags the source. `predict_proba` stays a pure scorer; the atomic mirror delegates. An owner-run re-bundle attaches the gate to the committed `default`/`skillcorner` artifacts with a corpus-identity guard (coefficients provably byte-unchanged).

**Tech Stack:** Python, numpy, pandas, scikit-learn (fit only), pytest. `silly_kicks.tracking._gk_completion` / `._xt_gk` / `scripts/train_gk_completion.py`.

**Spec:** `docs/superpowers/specs/2026-06-09-xt-gk-pertype-baserate-switch-design.md` (v3).

---

## Commit policy (read first)

One commit at the end (Task 9), gated on **explicit user approval + the git-commit sentinel**. Tasks 0–8 stage + test but do NOT commit. All work on `pr-s91-xt-gk-pertype-baserate` (created in Task 0). Use `.venv/Scripts/python.exe` / `.venv/Scripts/ruff.exe` / `.venv/Scripts/pyright.exe` for ALL commands — the bash-tool `python` is box 3.14, not the venv.

## File Structure

- **Modify** `silly_kicks/tracking/_gk_completion.py` — module constants `_GATE_LCB_FLOOR`/`_GATE_N_MIN`; pure `serve_mode_from_lcb`; `GkCompletionModel` gains `_type_serve_mode`/`_type_gate_metrics` + `serve_mode_for_types`/`base_rate_for_types`; `VERSION` bump; `to_dict`/`from_dict` round-trip the fields (fail-open).
- **Modify** `silly_kicks/tracking/_xt_gk.py` — the serve switch in `compute_xt_gk` (replace the hard-wired `"model"` at `:451-452`); `XtGkReport.completion_source_counts`.
- **Modify** `scripts/train_gk_completion.py` — shared 3-bucket per-type gate measurement (`_per_type_gate_from_oof`) wired into **both** `_train_skillcorner` and `main()`; corpus-identity guard before save.
- **Create** `tests/tracking/test_gk_completion_pertype_gate.py` — pure-function + model + save/load + real-artifact lock tests.
- **Create** `tests/tracking/test_xt_gk_pertype_baserate.py` — compute_xt_gk switch + report + atomic-parity + byte-identical regression lock.
- **Modify** `silly_kicks/tracking/_gk_completion_weights/default/` + `…/skillcorner/` — re-serialized with the gate (Task 5, owner-run).
- **Modify** `TODO.md`, version sites, `CHANGELOG.md`, `uv.lock`, `docs/superpowers/adrs/ADR-024-*.md`.

---

## Task 0: Branch + green baseline

**Files:** none modified.

- [ ] **Step 1: Create the branch**

Run: `git switch -c pr-s91-xt-gk-pertype-baserate`
Expected: switched, off `main` (HEAD `7887639` / 4.21.3).

- [ ] **Step 2: Capture the baseline for the touched areas**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/ -k "xt_gk or gk_completion" -q 2>&1 | tail -5`
Expected: all pass (record the count).

---

## Task 1: Pure gate-decision function `serve_mode_from_lcb`

**Files:**
- Modify: `silly_kicks/tracking/_gk_completion.py`
- Test: `tests/tracking/test_gk_completion_pertype_gate.py` (create)

- [ ] **Step 1: Write the failing boundary tests**

Create `tests/tracking/test_gk_completion_pertype_gate.py`:

```python
"""Per-type serve-gate: pure decision fn + model fields + save/load + real-artifact lock."""

import math

import numpy as np
import pandas as pd

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking._gk_completion import (
    GkCompletionModel,
    serve_mode_from_lcb,
    _GATE_LCB_FLOOR,
    _GATE_N_MIN,
)

_GOALKICK = spadlconfig.actiontype_id["goalkick"]
_THROW_IN = spadlconfig.actiontype_id["throw_in"]
_PASS = spadlconfig.actiontype_id["pass"]


def test_serve_mode_lcb_above_floor_is_model():
    assert serve_mode_from_lcb(0.55, n=200) == "model"


def test_serve_mode_lcb_at_or_below_floor_is_base_rate():
    assert serve_mode_from_lcb(_GATE_LCB_FLOOR, n=200) == "base_rate"  # 0.5 is NOT > 0.5
    assert serve_mode_from_lcb(0.42, n=200) == "base_rate"


def test_serve_mode_none_or_nan_lcb_is_base_rate():
    assert serve_mode_from_lcb(None, n=200) == "base_rate"
    assert serve_mode_from_lcb(float("nan"), n=200) == "base_rate"


def test_serve_mode_too_few_samples_is_base_rate():
    assert serve_mode_from_lcb(0.99, n=_GATE_N_MIN - 1) == "base_rate"
    assert serve_mode_from_lcb(0.55, n=_GATE_N_MIN) == "model"  # exactly n_min is enough
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_gk_completion_pertype_gate.py -q -k serve_mode`
Expected: FAIL — `ImportError: cannot import name 'serve_mode_from_lcb'`.

- [ ] **Step 3: Implement the function + constants**

In `silly_kicks/tracking/_gk_completion.py`, after the `_THROW_IN`/`_WEIGHTS_ROOT` constants (~line 19), add:

```python
_GATE_LCB_FLOOR = 0.5  # serve the model only if a type's held-out AUC LCB strictly exceeds chance
_GATE_N_MIN = 50  # below this per-type sample a bootstrap LCB is too unstable to trust -> base_rate


def serve_mode_from_lcb(lcb: float | None, n: int, *, lcb_floor: float = _GATE_LCB_FLOOR, n_min: int = _GATE_N_MIN) -> str:
    """Per-type serve-gate decision (the ONE place the rule lives; unit-tested at the boundaries).

    Returns ``"model"`` iff the type's held-out AUC lower-confidence-bound strictly exceeds ``lcb_floor``
    on a large-enough sample; else ``"base_rate"``. A ``None``/NaN ``lcb`` (undefined/degenerate AUC --
    e.g. a near-empty positive class like GK throw-ins) or ``n < n_min`` -> ``"base_rate"``. See spec
    Decision 2: serve uses the conservative LCB while *bundling* uses the point estimate -- different
    questions ("beats chance with confidence for THIS type" vs "good enough to ship the variant")."""
    if lcb is None or not math.isfinite(lcb) or n < n_min:
        return "base_rate"
    return "model" if lcb > lcb_floor else "base_rate"
```

Add `import math` to the top-of-file imports if absent.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_gk_completion_pertype_gate.py -q -k serve_mode`
Expected: 4 passed.

- [ ] **Step 5: Lint**

Run: `.venv/Scripts/ruff.exe check silly_kicks/tracking/_gk_completion.py tests/tracking/test_gk_completion_pertype_gate.py && .venv/Scripts/ruff.exe format --check silly_kicks/tracking/_gk_completion.py tests/tracking/test_gk_completion_pertype_gate.py`
Expected: no errors.

---

## Task 2: `GkCompletionModel` per-type gate fields + methods + serialization

**Files:**
- Modify: `silly_kicks/tracking/_gk_completion.py`
- Test: `tests/tracking/test_gk_completion_pertype_gate.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/tracking/test_gk_completion_pertype_gate.py`:

```python
def _fitted_model_with_gate(serve_mode):
    # A minimally-fitted model with explicit per-type gate + base rates (no real corpus needed).
    rng = np.random.default_rng(0)
    n = 200
    feats = pd.DataFrame(
        {
            "length": rng.normal(20, 5, n),
            "dx": rng.normal(10, 3, n),
            "dy_abs": np.abs(rng.normal(0, 5, n)),
            "forwardness": rng.normal(0, 1, n),
            "dest_x": rng.uniform(0, 105, n),
            "dest_y": rng.uniform(0, 68, n),
            "dest_y_off": np.abs(rng.uniform(0, 34, n)),
            "dest_defender_density": rng.uniform(0, 1, n),
            "is_goalkick": (np.arange(n) % 3 == 0).astype(float),
            "is_throw_in": (np.arange(n) % 3 == 1).astype(float),
        }
    )
    y = pd.Series((rng.random(n) < 0.6).astype(int))
    m = GkCompletionModel().fit(feats, y)
    m._type_serve_mode = dict(serve_mode)
    m._type_gate_metrics = {k: {"auc": 0.5, "lcb": 0.49, "n": 80} for k in serve_mode}
    return m


def test_serve_mode_for_types_maps_per_gate():
    m = _fitted_model_with_gate({"goalkick": "base_rate", "throw_in": "base_rate", "other": "model"})
    tids = np.array([_GOALKICK, _THROW_IN, _PASS])
    assert list(m.serve_mode_for_types(tids)) == ["base_rate", "base_rate", "model"]


def test_serve_mode_for_types_absent_type_defaults_model():
    m = _fitted_model_with_gate({})  # no gate -> fail-open
    assert list(m.serve_mode_for_types(np.array([_GOALKICK, _PASS]))) == ["model", "model"]


def test_base_rate_for_types_returns_per_type_rate():
    m = _fitted_model_with_gate({"goalkick": "base_rate"})
    br = m.base_rate_for_types(np.array([_GOALKICK, _THROW_IN, _PASS]))
    assert math.isclose(br[0], m._base_rates["goalkick"])
    assert math.isclose(br[1], m._base_rates["throw_in"])
    assert math.isclose(br[2], m._base_rates["other"])


def test_save_load_roundtrips_gate(tmp_path):
    m = _fitted_model_with_gate({"goalkick": "base_rate", "throw_in": "base_rate", "other": "model"})
    m.save(tmp_path)
    back = GkCompletionModel.load(tmp_path)
    assert back._type_serve_mode == m._type_serve_mode
    assert back._type_gate_metrics["goalkick"]["n"] == 80


def test_load_fail_open_when_gate_absent(tmp_path):
    import json

    m = _fitted_model_with_gate({"goalkick": "base_rate"})
    m.save(tmp_path)
    d = json.loads((tmp_path / "model.json").read_text(encoding="utf-8"))
    del d["type_serve_mode"]  # simulate a pre-gate (4.21.0) artifact
    (tmp_path / "model.json").write_text(json.dumps(d, indent=2), encoding="utf-8")
    (tmp_path / "SHA256SUMS").write_text(
        f"{GkCompletionModel._sha(tmp_path)}  model.json\n", encoding="utf-8"
    )
    back = GkCompletionModel.load(tmp_path)
    assert back._type_serve_mode == {}  # absent -> empty
    assert list(back.serve_mode_for_types(np.array([_GOALKICK]))) == ["model"]  # fail-open
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_gk_completion_pertype_gate.py -q -k "for_types or roundtrips or fail_open"`
Expected: FAIL — `AttributeError: 'GkCompletionModel' object has no attribute 'serve_mode_for_types'` (and `_type_serve_mode`).

- [ ] **Step 3: Implement the fields + methods + serialization**

In `silly_kicks/tracking/_gk_completion.py`:

(a) Bump `VERSION = "1.0.0"` → `VERSION = "1.1.0"` (line 65).

(b) In `__init__` (after `self._base_rates = {}`, line 73), add:

```python
        self._type_serve_mode: dict[str, str] = {}  # {goalkick|throw_in|other: "model"|"base_rate"}; empty -> fail-open all-"model"
        self._type_gate_metrics: dict[str, dict] = {}  # {type: {auc, lcb, n}} -- transparency (model card), not read at serve
```

(c) After `_base_rate_for_type` (line 137), add the two pure helpers:

```python
    @staticmethod
    def _type_key(type_id: int) -> str:
        if type_id == _GOALKICK:
            return "goalkick"
        if type_id == _THROW_IN:
            return "throw_in"
        return "other"

    def serve_mode_for_types(self, type_ids: np.ndarray) -> np.ndarray:
        """Per-row ``"model"``/``"base_rate"`` from the stored per-type gate; absent type -> ``"model"``
        (fail-open). Pure; the gate is computed at train time (held-out CV)."""
        return np.array(
            [self._type_serve_mode.get(self._type_key(int(t)), "model") for t in type_ids], dtype=object
        )

    def base_rate_for_types(self, type_ids: np.ndarray) -> np.ndarray:
        """Vectorized per-type calibrated base rate (reuses ``_base_rate_for_type``)."""
        return np.array(
            [self._base_rate_for_type(float(t == _GOALKICK), float(t == _THROW_IN)) for t in type_ids],
            dtype=float,
        )
```

(d) In `to_dict` (line 145-156 return dict), add two keys before the closing brace:

```python
            "type_serve_mode": self._type_serve_mode,
            "type_gate_metrics": self._type_gate_metrics,
```

(e) In `from_dict` (after `m._base_rates = dict(d["base_rates"])`, line 166), add (fail-open via `.get`):

```python
        m._type_serve_mode = dict(d.get("type_serve_mode", {}))
        m._type_gate_metrics = dict(d.get("type_gate_metrics", {}))
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_gk_completion_pertype_gate.py -q`
Expected: all pass (the Task-1 + Task-2 cases).

- [ ] **Step 5: Lint**

Run: `.venv/Scripts/ruff.exe check silly_kicks/tracking/_gk_completion.py tests/tracking/test_gk_completion_pertype_gate.py && .venv/Scripts/ruff.exe format --check silly_kicks/tracking/_gk_completion.py tests/tracking/test_gk_completion_pertype_gate.py`
Expected: no errors.

---

## Task 3: The `compute_xt_gk` serve switch + report + byte-identical lock

**Files:**
- Modify: `silly_kicks/tracking/_xt_gk.py`
- Test: `tests/tracking/test_xt_gk_pertype_baserate.py` (create)

> The switch site is `_xt_gk.py:448-452`: `pc = _completion_p(...)` then the hard-wired
> `out.loc[mask, "xt_gk_completion_source"] = "model"`. `completion_model` + `completion_key` are
> resolved at `:380` (`_resolve_completion_for_frames`). In this RAV path only geometry-resolved rows
> are in `mask`, so the only base-rate trigger is the per-type gate.

- [ ] **Step 1: Write the failing switch test**

Create `tests/tracking/test_xt_gk_pertype_baserate.py`. Reuse the **real** existing builders from
`test_xt_gk.py` — `_gk_actions()` (a goal-kick `type_id=22` + a GK back-pass `type_id=0`, i.e. the
`other` bucket), `_frames_for(actions)`, `_gk_realistic_xt()` (the DZV-valid grid) — and the Task-2
`_fitted_model_with_gate` (importable from the Task-2 test file). The `completion=` injection wins
over auto-selection (precedence 1), so the frames' `source_provider` is irrelevant.

```python
"""compute_xt_gk per-type base-rate serve switch (xt_gk_completion_source)."""

import numpy as np
import pandas as pd

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking._xt_gk import compute_xt_gk
from tests.tracking.test_xt_gk import _gk_actions, _frames_for, _gk_realistic_xt
from tests.tracking.test_gk_completion_pertype_gate import _fitted_model_with_gate as _gate_model

_GOALKICK = spadlconfig.actiontype_id["goalkick"]
_THROW_IN = spadlconfig.actiontype_id["throw_in"]


def test_goalkick_gated_to_base_rate_is_tagged_and_differs_from_model():
    actions, xt = _gk_actions(), _gk_realistic_xt()
    frames = _frames_for(actions)
    gated = _gate_model({"goalkick": "base_rate", "throw_in": "base_rate", "other": "model"})
    model = _gate_model({"goalkick": "model", "throw_in": "model", "other": "model"})
    out_b = compute_xt_gk(actions, frames, xt=xt, completion=gated)
    out_m = compute_xt_gk(actions, frames, xt=xt, completion=model)
    gk_b = out_b[out_b["type_id"] == _GOALKICK]
    assert (gk_b["xt_gk_completion_source"] == "base_rate").all()  # tagged
    # the back-pass (other) stays "model" in both
    other = out_b[(out_b["type_id"] != _GOALKICK) & out_b["xt_gk_completion_source"].notna()]
    assert (other["xt_gk_completion_source"] == "model").all()
    # base-rate p flows into RAV -> the goal-kick xt_gk differs from the model-scored value
    gk_m = out_m[out_m["type_id"] == _GOALKICK]
    assert not np.allclose(gk_b["xt_gk"].to_numpy(), gk_m["xt_gk"].to_numpy(), equal_nan=True)


def test_switch_is_noop_when_no_type_gated():
    # Regression lock (review L2): when no type is gated "base_rate", the switch never overrides pc,
    # so it is a provable no-op. (Compares an explicit all-"model" gate vs the fail-open empty gate --
    # both new-code paths; the pre-switch code is gone and can't be run, so this asserts the no-op
    # property, not literal new-vs-old byte-identity.)
    actions, xt = _gk_actions(), _gk_realistic_xt()
    frames = _frames_for(actions)
    gated = _gate_model({"goalkick": "model", "throw_in": "model", "other": "model"})
    ungated = _gate_model({})  # fail-open -> all model
    a = compute_xt_gk(actions, frames, xt=xt, completion=gated)
    b = compute_xt_gk(actions, frames, xt=xt, completion=ungated)
    np.testing.assert_allclose(a["xt_gk"].to_numpy(), b["xt_gk"].to_numpy(), atol=1e-12, equal_nan=True)
    assert (a["xt_gk_completion_source"].dropna() == "model").all()


def test_throw_in_gated_to_base_rate_is_tagged():
    # review M1: a throw-in (degenerate positive class -> base_rate gate) must also be tagged base_rate
    # at the switch, not just goal-kicks. Build a GK throw-in at an EXISTING frame time so it links
    # (review-3 L-A): _frames_for emits frames only at 5.0/50.0, so the throw-in uses t=50.0 (the GK,
    # player 10, sits at (5,34) in both frames) and the unmodified _frames_for(_gk_actions()).
    base = _gk_actions().iloc[[0]].copy()  # the goalkick row's shape (GK actor, finite coords)
    base["action_id"] = [2]
    base["type_id"] = [_THROW_IN]
    base["time_seconds"] = [50.0]  # an existing frame time -> the throw-in links + resolves geometry
    actions = pd.concat([_gk_actions(), base], ignore_index=True)
    frames = _frames_for(_gk_actions())  # frames at 5.0 + 50.0 (hard-coded; independent of action times)
    gated = _gate_model({"goalkick": "model", "throw_in": "base_rate", "other": "model"})
    out = compute_xt_gk(actions, frames, xt=_gk_realistic_xt(), completion=gated)
    ti = out[out["type_id"] == _THROW_IN]
    assert len(ti) == 1 and (ti["xt_gk_completion_source"] == "base_rate").all()
```

> Note: `_frames_for` (`test_xt_gk.py:87`) hard-codes frame times `[(5.0, 1), (50.0, 1)]` (independent
> of action times), so every in-scope action must sit at 5.0 or 50.0 to link. The goal-kick + no-op
> tests use the unmodified `_gk_actions()` (5.0/50.0); the throw-in row is placed at 50.0 above.

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_xt_gk_pertype_baserate.py -q`
Expected: FAIL — goal-kick rows are tagged `"model"` (current hard-wire), so the `"base_rate"` assertion fails.

- [ ] **Step 3: Implement the switch**

In `silly_kicks/tracking/_xt_gk.py`, replace lines ~449-452:

```python
    # Task 8 provenance: the variant that scored each row + the source. In the RAV path a row is
    # scored only when geometry resolves (else NaN, not base-rated -- m2), so scored rows are "model".
    out.loc[mask, "xt_gk_completion_variant"] = completion_key
    out.loc[mask, "xt_gk_completion_source"] = "model"
```

with the per-type serve switch:

```python
    # Per-type base-rate serve switch (spec 2026-06-09 §2.3/m3): a type whose held-out AUC can't beat
    # chance with confidence serves the calibrated per-type base rate (tagged "base_rate") instead of
    # the geometric p. Geometry-missing rows are already excluded from `mask` (m2), so the per-type
    # gate is the only base-rate trigger here.
    tids = actions.loc[mask, "type_id"].to_numpy()
    serve_mode = completion_model.serve_mode_for_types(tids)
    is_base = serve_mode == "base_rate"
    if is_base.any():
        pc[is_base] = completion_model.base_rate_for_types(tids[is_base])
    out.loc[mask, "xt_gk_completion_variant"] = completion_key
    out.loc[mask, "xt_gk_completion_source"] = np.where(is_base, "base_rate", "model")
```

(`completion_model` is the resolved model from `:380`; `pc` is the writable array from `_completion_p`.)

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_xt_gk_pertype_baserate.py -q`
Expected: 3 passed (goal-kick → base_rate tag + differs from model; no-op when no type gated; throw-in → base_rate).

- [ ] **Step 5: Add `completion_source_counts` to `XtGkReport` + a test**

In `_xt_gk.py`, in the `XtGkReport` dataclass add `completion_source_counts: dict[str, int]`, and in `from_frame` populate it from `df["xt_gk_completion_source"].value_counts(dropna=True)` (mirror the existing `completion_variant_counts` block at `:79-90`). Append a test to `test_xt_gk_pertype_baserate.py`:

```python
def test_report_completion_source_counts():
    from silly_kicks.tracking._xt_gk import XtGkReport

    actions, xt = _gk_actions(), _gk_realistic_xt()
    frames = _frames_for(actions)
    model = _gate_model({"goalkick": "base_rate", "throw_in": "base_rate", "other": "model"})
    out = compute_xt_gk(actions, frames, xt=xt, completion=model)
    rep = XtGkReport.from_frame(out)
    vc = out["xt_gk_completion_source"].value_counts(dropna=True)
    assert rep.completion_source_counts == {str(k): int(v) for k, v in vc.items()}
```

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_xt_gk_pertype_baserate.py -q` → all pass.

- [ ] **Step 6: Atomic-mirror parity test (monkeypatched gate)**

Atomic `add_xt_gk` delegates to `tracking.features.add_xt_gk` → `compute_xt_gk`, so the switch is
inherited. **But `add_xt_gk` does NOT thread `completion=`** (it calls `compute_xt_gk(..., xt=xt,
params=params, links=pointers)` at `features.py:5177`), so the model is auto-resolved from
`frames["source_provider"]` — a synthetic gate can't be injected via the `completion=` kwarg here.
Inject it by **monkeypatching `from_variant`** to return a gated synthetic model, then run the atomic
path. Build the atomic input with the **column-rename scaffold** the existing
`test_atomic_add_xt_gk_matches_standard_via_synthesis` uses (`test_xt_gk.py:608-612`) — **NOT
`convert_to_atomic`**, which remaps `type_id` into the atomic enumeration and would break the
`== _GOALKICK` (22) filter. The rename only touches coordinate columns, so `type_id == _GOALKICK`
holds. The atomic `add_xt_gk(actions, frames, xt, *, home_team_id, …)` DOES take `home_team_id`
(unlike `compute_xt_gk`):

```python
def test_atomic_path_inherits_base_rate_switch(monkeypatch):
    from silly_kicks.atomic.tracking.features import add_xt_gk as atomic_add_xt_gk
    from silly_kicks.tracking.features import add_xt_gk as std_add_xt_gk
    import silly_kicks.tracking._gk_completion as gkc

    gated = _gate_model({"goalkick": "base_rate", "throw_in": "model", "other": "model"})
    monkeypatch.setattr(gkc.GkCompletionModel, "from_variant", classmethod(lambda cls, variant="default": gated))
    std = _gk_actions()
    frames = _frames_for(std)
    atom = std.rename(columns={"start_x": "x", "start_y": "y"}).copy()
    atom["dx"] = std["end_x"].to_numpy() - std["start_x"].to_numpy()
    atom["dy"] = std["end_y"].to_numpy() - std["start_y"].to_numpy()
    atom = atom.drop(columns=["end_x", "end_y"])
    atom_out = atomic_add_xt_gk(atom, frames, _gk_realistic_xt(), home_team_id=1)
    std_out = std_add_xt_gk(std, frames, _gk_realistic_xt(), home_team_id=1)
    gk = atom_out[atom_out["type_id"] == _GOALKICK]
    assert len(gk) >= 1 and (gk["xt_gk_completion_source"] == "base_rate").all()  # switch inherited
    # parity: atomic mirror tags identically to the standard path (mirrors test_xt_gk.py:629-632)
    assert (
        atom_out["xt_gk_completion_source"].to_numpy().tolist()
        == std_out["xt_gk_completion_source"].to_numpy().tolist()
    )
```

Run the file → pass.

- [ ] **Step 7: Lint**

Run: `.venv/Scripts/ruff.exe check silly_kicks/tracking/_xt_gk.py tests/tracking/test_xt_gk_pertype_baserate.py && .venv/Scripts/ruff.exe format --check silly_kicks/tracking/_xt_gk.py tests/tracking/test_xt_gk_pertype_baserate.py`
Expected: no errors.

---

## Task 4: Train script — shared 3-bucket gate measurement + corpus-identity guard

**Files:**
- Modify: `scripts/train_gk_completion.py`

> H1: `_report` (per-type AUC+LCB) lives only in `_train_skillcorner` over `{goalkick, gk_pass=~goalkick}`; `main()` (gs) has no per-type AUC; neither matches the model's 3-way `{goalkick, throw_in, other}`. This task hoists a shared 3-bucket measurement and wires it into both fits, plus the corpus-identity guard.

- [ ] **Step 1: Add the shared 3-bucket gate measurement**

Add a module-level helper to `scripts/train_gk_completion.py` (reuses `_bootstrap_auc_ci` + `serve_mode_from_lcb`):

```python
def _per_type_gate_from_oof(oof: np.ndarray, y_all: np.ndarray, X_all) -> tuple[dict, dict]:
    """Per-type serve gate over the model's 3-way {goalkick, throw_in, other} partition (matches
    GkCompletionModel._base_rates). Returns (type_serve_mode, type_gate_metrics) from held-out OOF.
    A degenerate/insufficient bucket (AUC undefined or n < _GATE_N_MIN) -> base_rate via the shared
    serve_mode_from_lcb. Bucket masks use the feature columns is_goalkick / is_throw_in."""
    from sklearn.metrics import roc_auc_score

    from silly_kicks.tracking._gk_completion import serve_mode_from_lcb

    ok = np.isfinite(oof)
    is_gk = X_all["is_goalkick"].to_numpy() == 1.0
    is_ti = X_all["is_throw_in"].to_numpy() == 1.0
    buckets = {"goalkick": is_gk, "throw_in": is_ti, "other": ~(is_gk | is_ti)}
    serve_mode: dict[str, str] = {}
    metrics: dict[str, dict] = {}
    for name, b in buckets.items():
        m = b & ok
        n = int(m.sum())
        if n < 2 or len(np.unique(y_all[m])) < 2:
            auc = lcb = None  # degenerate (e.g. near-empty GK throw-in positive class)
        else:
            auc = float(roc_auc_score(y_all[m], oof[m]))
            lcb = float(_bootstrap_auc_ci(y_all[m], oof[m])[0])
        serve_mode[name] = serve_mode_from_lcb(lcb, n)
        metrics[name] = {"auc": auc, "lcb": lcb, "n": n}
        print(f"  [gate {name}] n={n} auc={auc} lcb={lcb} -> {serve_mode[name]}", flush=True)
    return serve_mode, metrics
```

- [ ] **Step 2: Replace the `_train_skillcorner` bundle block (review H2 — save the committed model + gate, NOT the re-fit)**

The current block (`:222-227`) **re-fits then saves** the fresh model:

```python
    if decision.startswith("bundle_skillcorner"):
        model = GkCompletionModel().fit(X_all, pd.Series(y_all))
        model.shipped_variant = "skillcorner"
        model.provider_list = ["skillcorner"]
        model.save(_SKILLCORNER_WEIGHTS_DIR)
        reloaded = GkCompletionModel.load(_SKILLCORNER_WEIGHTS_DIR)
        np.testing.assert_allclose(model.predict_proba(X_all), reloaded.predict_proba(X_all), atol=1e-9)
        bundled = True
```

Saving the re-fit `model` would ship freshly-re-fit coefficients (only `assert_allclose`'d, not byte-equal) — contradicting spec v3 §5 + the Task-5 byte-unchanged check. **Replace** it so the fresh fit is only the corpus-identity *probe* and the SERVED coefficients are the committed bytes:

```python
    if decision.startswith("bundle_skillcorner"):
        # `model` = fresh full-data fit, used as the CORPUS-IDENTITY PROBE only. The SERVED artifact is
        # the committed model (its bytes), so the OOF gate provably describes the served model AND
        # coef stay byte-identical (spec v3 §5 + Task-5 check). Re-fit is NEVER persisted on a re-bundle.
        model = GkCompletionModel().fit(X_all, pd.Series(y_all))
        sm, gm = _per_type_gate_from_oof(oof, y_all, X_all)
        try:
            served = GkCompletionModel.load(_SKILLCORNER_WEIGHTS_DIR)  # committed coef = the served bytes
            np.testing.assert_allclose(model._coef, served._coef, atol=1e-9)
            np.testing.assert_allclose([model._intercept], [served._intercept], atol=1e-9)
            np.testing.assert_allclose(model._mean, served._mean, atol=1e-9)
            np.testing.assert_allclose(model._std, served._std, atol=1e-9)
        except FileNotFoundError:
            served = model  # first-ever bundle: nothing committed to preserve -> ship the fresh fit
        served.shipped_variant = "skillcorner"
        served.provider_list = ["skillcorner"]
        served._type_serve_mode, served._type_gate_metrics = sm, gm
        served.save(_SKILLCORNER_WEIGHTS_DIR)
        reloaded = GkCompletionModel.load(_SKILLCORNER_WEIGHTS_DIR)
        np.testing.assert_allclose(served.predict_proba(X_all), reloaded.predict_proba(X_all), atol=1e-9)
        bundled = True
```

> **STOP note (review-3 L-C):** this gate-write only runs when `decision.startswith("bundle_skillcorner")`.
> On a **re-bundle** the decision MUST reproduce `bundle_skillcorner` (it will if the corpus is stable
> — the same data that produced the committed SC weights). If it flips to `alias_gs`, the block is
> skipped → the SC artifact is silently NOT re-gated AND the corpus-identity guard never runs. A
> decision flip is itself a corpus-drift signal — **abort and investigate**, do not proceed. Task 5
> Step 2 asserts the decision reproduced `bundle_skillcorner`.

- [ ] **Step 3: Replace the `main()` (GS default) bundle block the same way**

`main()` already builds an OOF array in its CV loop (`:285-291`) — reuse it (review L1: do **not** re-allocate; it exists). The current final block (`:324-328`) is:

```python
    model = GkCompletionModel().fit(X_all, pd.Series(y_all))
    model.shipped_variant = "default"
    model.provider_list = list(args.providers)
    model.save(_WEIGHTS_DIR)
    reloaded = GkCompletionModel.load(_WEIGHTS_DIR)
    np.testing.assert_allclose(model.predict_proba(X_all), reloaded.predict_proba(X_all), atol=1e-9)
```

Replace with the same probe-vs-served pattern (against `_WEIGHTS_DIR`, `shipped_variant="default"`, `provider_list = list(args.providers)`):

```python
    model = GkCompletionModel().fit(X_all, pd.Series(y_all))  # corpus-identity probe
    sm, gm = _per_type_gate_from_oof(oof, y_all, X_all)
    try:
        served = GkCompletionModel.load(_WEIGHTS_DIR)
        np.testing.assert_allclose(model._coef, served._coef, atol=1e-9)
        np.testing.assert_allclose([model._intercept], [served._intercept], atol=1e-9)
        np.testing.assert_allclose(model._mean, served._mean, atol=1e-9)
        np.testing.assert_allclose(model._std, served._std, atol=1e-9)
    except FileNotFoundError:
        served = model
    served.shipped_variant = "default"
    served.provider_list = list(args.providers)
    served._type_serve_mode, served._type_gate_metrics = sm, gm
    served.save(_WEIGHTS_DIR)
    reloaded = GkCompletionModel.load(_WEIGHTS_DIR)
    np.testing.assert_allclose(served.predict_proba(X_all), reloaded.predict_proba(X_all), atol=1e-9)
```

- [ ] **Step 4: Lint + a CI does-it-run smoke**

Run: `.venv/Scripts/ruff.exe check scripts/train_gk_completion.py && .venv/Scripts/ruff.exe format --check scripts/train_gk_completion.py`
Expected: no errors.

Add a CI smoke to `tests/tracking/test_gk_completion_pertype_gate.py` exercising the gate helper on a synthetic OOF (no owner data) so the wiring is locked even though the real AUC is owner-run:

```python
def test_per_type_gate_from_oof_smoke():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import importlib

    tg = importlib.import_module("train_gk_completion")
    n = 300
    rng = np.random.default_rng(1)
    X = pd.DataFrame(
        {"is_goalkick": (np.arange(n) % 3 == 0).astype(float), "is_throw_in": (np.arange(n) % 3 == 1).astype(float)}
    )
    y = (rng.random(n) < 0.5).astype(int)
    oof = rng.random(n)  # random scores -> AUC ~ 0.5 -> LCB <= 0.5 -> base_rate
    sm, gm = tg._per_type_gate_from_oof(oof, y, X)
    assert set(sm) == {"goalkick", "throw_in", "other"}
    assert all(v in ("model", "base_rate") for v in sm.values())
    assert all(set(gm[k]) == {"auc", "lcb", "n"} for k in gm)
```

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_gk_completion_pertype_gate.py -q -k gate_from_oof` → pass.

---

## Task 5: Owner-run re-measure + re-bundle (produces the gated artifacts)

**Files:**
- Modify: `silly_kicks/tracking/_gk_completion_weights/default/{model.json,SHA256SUMS}`, `…/skillcorner/{…}`

> Owner-run — needs the GS WC2022 + SkillCorner pining corpora via `_loader_*` (read-only) + the
> isolated env if a connector is involved. The corpus-identity guard (Task 4) asserts coef byte-identity;
> a mismatch aborts. Report numbers back to the user before finalizing.

- [ ] **Step 1: Re-bundle the GS `default`**

Run the GS path of the train script (the same invocation that produced the bundled `default` — see its module docstring / argparse) under `.venv/Scripts/python.exe`. Confirm the corpus-identity guard passes (no abort), capture the printed `[gate goalkick] … -> {model|base_rate}` + the `[gate other]` STOP-check + per-type `n`.

- [ ] **Step 2: Re-bundle the `skillcorner` variant**

Run `scripts/train_gk_completion.py --variant skillcorner` (per spec §3.3). **First confirm the printed `DECISION:` reproduced `bundle_skillcorner`** (review-3 L-C — a flip to `alias_gs` means corpus drift: abort + investigate, the SC artifact would otherwise be silently un-gated). Then confirm `[gate goalkick] -> base_rate` (expected: AUC ~0.433) and `[gate other] -> model` (GK-pass; STOP if it flips). Capture per-type `n` + the metrics.

- [ ] **Step 3: Verify the diff is additive-only (review L1)**

Run: `git --no-pager diff silly_kicks/tracking/_gk_completion_weights/`
Confirm the ONLY changed keys are `version` (→ 1.1.0), `type_serve_mode`, `type_gate_metrics` (+ the SHA256SUMS). `coef`/`intercept`/`mean`/`std`/`base_rates` must be **byte-unchanged**. If a coefficient moved, STOP — the corpus drifted (hidden retrain).

- [ ] **Step 4: Report to the user**

Report: GS goal-kick `auc/lcb/n` + decision; skillcorner goal-kick + other decisions; per-type counts → the CHANGELOG blast-radius figure (Task 7). **Hold for the user to confirm the GS decision before finalizing** (it sets the GS golden-lock value in Task 6).

---

## Task 6: Real-artifact gate-lock tests (CI everywhere)

**Files:**
- Test: `tests/tracking/test_gk_completion_pertype_gate.py`

- [ ] **Step 1: Add the bundled-artifact locks**

Append (the skillcorner lock is known a-priori; the GS value is set from the Task-5 report):

```python
def test_bundled_skillcorner_goalkick_is_base_rate():
    # Real-artifact lock (review M3): the committed skillcorner gate routes goal-kicks to base_rate
    # (goal-kick AUC ~0.433 < chance). Stronger than an owner e2e -- the variable is committed.
    m = GkCompletionModel.from_variant("skillcorner")
    assert m._type_serve_mode.get("goalkick") == "base_rate"


def test_bundled_gs_default_goalkick_mode_is_locked():
    # Measured-value golden (review-2 L-A): the GS goal-kick mode is set from the Task-5 owner-run
    # report, then a permanent regression lock. Replace <GS_GOALKICK_MODE> with the reported value.
    m = GkCompletionModel.from_variant("default")
    assert m._type_serve_mode.get("goalkick") == "<GS_GOALKICK_MODE>"
```

- [ ] **Step 2: Set the GS golden + run**

Replace `<GS_GOALKICK_MODE>` with the Task-5 reported value (`"model"` or `"base_rate"`).

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_gk_completion_pertype_gate.py -q`
Expected: all pass (incl. both real-artifact locks).

---

## Task 7: Housekeeping — TODO + version + CHANGELOG + ADR-024

**Files:** `TODO.md`, `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, `uv.lock`, `docs/superpowers/adrs/ADR-024-*.md`

- [ ] **Step 1: Remove the TODO item**

Delete the "xT-GK per-type base-rate serve switch for goal-kicks (4.21.0 follow-up)" bullet from `TODO.md`. Update the `**Current release**:` header line to 4.21.4 (the version-bump 5th site).

- [ ] **Step 2: Bump the version 4.21.3 → 4.21.4**

Run: `git fetch --tags && git tag --list 'v4.21.4'` (expect empty). Set `pyproject.toml` + `silly_kicks/__init__.py` (`__version__`) to `4.21.4`.

- [ ] **Step 3: CHANGELOG entry (with the blast radius, review-2 L-B)**

Add above `## [4.21.3] — 2026-06-09`:

```markdown
## [4.21.4] — 2026-06-09

### Changed — xT-GK per-type base-rate serve switch (goal-kick completion honesty)

`compute_xt_gk` now serves the **per-type calibrated base rate** (tagged `xt_gk_completion_source =
"base_rate"`) instead of the geometric model for any completion-variant sub-domain whose held-out AUC
lower-confidence-bound ≤ 0.5 (or degenerate/too-small) — fixing goal-kicks, where geometry is
near-chance (skillcorner goal-kick AUC 0.433). The gate is a `serve_mode_from_lcb(lcb, n)` decision
baked into the `GkCompletionModel` artifact (`_type_serve_mode` + `_type_gate_metrics`, version
1.1.0); `load()` fail-opens (a pre-gate artifact serves all types `"model"` = prior behavior).
Coefficients are byte-unchanged (corpus-identity-guarded re-bundle). Not a VAEP retrain (xt_gk is
opt-in) — but an `xt_gk` serve-output change for the flipped types: **lakehouse re-materializes
`xt_gk` for ~<N>% of rows (goal-kicks + degenerate throw-ins for variants <X>)**. Hub `full`
fail-opens to model-scored goal-kicks until re-uploaded (follow-up). ADR-024 amendment. (TODO 4.21.0
§2.3/m3 follow-up.)
```

Fill `<N>` / `<X>` from the Task-5 counts.

- [ ] **Step 4: ADR-024 amendment + re-lock**

Add a brief amendment paragraph to `docs/superpowers/adrs/ADR-024-*.md` (the xT-GK ADR) recording the per-type serve gate (refines completion serving; no new methodology). Then:

Run: `uv lock && grep -rn "4\.21\.4" pyproject.toml silly_kicks/__init__.py CHANGELOG.md && .venv/Scripts/python.exe -c "import silly_kicks; print(silly_kicks.__version__)"`
Expected: `4.21.4` everywhere; import prints `4.21.4`; `uv.lock` updates the silly-kicks pin only.

---

## Task 8: Full-suite verification + final-review

**Files:** none.

- [ ] **Step 1: Full non-e2e suite**

Run: `.venv/Scripts/python.exe -m pytest tests/ -m "not e2e and not slow" -q 2>&1 | tail -6; echo "EXIT: ${PIPESTATUS[0]}"`
Expected: baseline + the new tests; 0 failures; EXIT 0. (Read the real summary line.)

- [ ] **Step 2: Whole-tree lint + types**

Run: `.venv/Scripts/ruff.exe check silly_kicks/ tests/ scripts/ && .venv/Scripts/ruff.exe format --check silly_kicks/ tests/ scripts/ && .venv/Scripts/pyright.exe silly_kicks/`
Expected: all clean.

- [ ] **Step 3: `/final-review`**

Invoke `/final-review`. Confirm C4-free (a serve gate on an existing model — no new container/model/aggregator/backend); the re-serialized weight artifacts are a reviewed diff (additive-only).

---

## Task 9: Single commit + PR (gated on explicit approval)

- [ ] **Step 1: Present the diff + commit command, HOLD for approval**

**Placeholder gate (review L4):** the commit is BLOCKED until the two owner-run-measured values are
filled from the Task-5 report — `grep -rn "<GS_GOALKICK_MODE>\|<N>\|<X>" tests/ CHANGELOG.md` must
return **nothing** (a literal placeholder in the test or CHANGELOG must not reach the commit).

Run: `git status && git --no-pager diff --stat`. Present the staged set + commit command. **Do not create the sentinel or commit without an explicit "yes."**

- [ ] **Step 2: Commit (write the message to a temp file, `git commit -F`)**

```bash
git add silly_kicks/tracking/_gk_completion.py silly_kicks/tracking/_xt_gk.py scripts/train_gk_completion.py \
  silly_kicks/tracking/_gk_completion_weights/default/ silly_kicks/tracking/_gk_completion_weights/skillcorner/ \
  tests/tracking/test_gk_completion_pertype_gate.py tests/tracking/test_xt_gk_pertype_baserate.py \
  TODO.md pyproject.toml silly_kicks/__init__.py CHANGELOG.md uv.lock docs/superpowers/adrs/ADR-024-*.md \
  docs/superpowers/specs/2026-06-09-xt-gk-pertype-baserate-switch-design.md \
  docs/superpowers/plans/2026-06-09-xt-gk-pertype-baserate-switch.md
git commit -F .git/COMMIT_XTGK_GATE.txt
```

Subject: `feat(tracking): SK-91 xT-GK per-type base-rate serve switch (goal-kicks) -- silly-kicks 4.21.4 (ADR-024)`. Body: the gate (LCB>0.5, fail-open), the switch, the corpus-guarded re-bundle, the lakehouse blast radius, review rounds addressed. End with the `Co-Authored-By` trailer.

- [ ] **Step 3: Push + PR + merge + tag** (per `reference_version_bump_checklist`, bare commands, after CI green + user signal)

`git push -u origin pr-s91-xt-gk-pertype-baserate`; `gh pr create --base main --body-file <file>`; after CI green → squash `--admin` merge + delete branch; tag `v4.21.4` (annotated) + push (triggers publish). Confirm PyPI 4.21.4.

---

## Self-Review notes

- **Spec coverage:** Decision 1 gate-in-artifact (Task 2, 5). Decision 2 LCB>0.5 + n-guard + STOP-safeguard (Task 1 fn, Task 5 owner STOP-check). Decision 3 fail-open (Task 2 from_dict + test). Decision 4 uniform 3-bucket gate (Task 4 helper). Decision 5 switch-in-compute_xt_gk, predict_proba pure (Task 3). Components 1-5 (Tasks 1-5). Testing: pure-fn boundaries (T1), model+save/load+fail-open (T2), switch+byte-identical+report+atomic (T3), real-artifact lock incl. GS golden (T6), train smoke (T4). Corpus-identity guard (T4 Step 2/3, T5 Step 3). Hyrum/housekeeping (T7). Verification + final-review (T8).
- **Type/name consistency:** `serve_mode_from_lcb` / `_GATE_LCB_FLOOR` / `_GATE_N_MIN` / `_type_serve_mode` / `_type_gate_metrics` / `serve_mode_for_types` / `base_rate_for_types` / `_type_key` / `_per_type_gate_from_oof` used identically across tasks. The switch reads `completion_model` (`:380`) + `completion_key`; `pc` from `_completion_p` (`:448`).
- **No placeholders:** every code/command step is concrete except the two deliberately owner-set values — the GS golden `<GS_GOALKICK_MODE>` (Task 6, set from Task 5) and the CHANGELOG `<N>`/`<X>` blast radius (Task 7, from Task 5 counts) — both explicitly flagged as owner-run-measured, not guesses.
```
