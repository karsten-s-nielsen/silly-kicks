# Feature-Column Glossary + `describe_level` + TF-7 xfns Pitch-Control Cache — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a machine-readable, CI-gated glossary of every derived feature column + a direction-aware `describe_level` z-bucketer, and extend the TF-7 `PitchControlCache` to the VAEP `*_xfns` path via caller-injection.

**Architecture:** Two independent sub-designs (spec: `docs/superpowers/specs/2026-07-24-feature-glossary-and-xfns-pc-cache-design.md`). #1 = a pure Python registry (`feature_glossary.py`) + a generic `reporting.py::describe_level`, made complete-by-construction by an inspection-based coverage gate and an attribution↔NOTICE gate. #2 = add `pitch_control_cache=` to the pitch-control-consuming `*_xfns` factories, threaded into their aggregator calls; `None` default is byte-identical to today.

**Tech Stack:** pandas / numpy; `dataclasses` + `typing.Literal`; pytest; the existing `PitchControlCache` (`silly_kicks/tracking/pitch_control/_cache.py`), the id-scalar discovery idiom (`tests/invariants/conftest_id_scalar.py::_public_names`), the liveness fixture (`tests/tracking/test_aggregator_column_liveness.py`), and the structural perf harness (`tests/_perf_structural.py::call_counter`).

**Commit policy (project override of the skill's per-task commit step):** do NOT commit per task. Execute all tasks, then the commit-prep group (Task 16) runs `/final-review` + version/C4/ADR/CHANGELOG/TODO and **stops before the commit for owner approval** — one squash-merged commit per the owner's one-commit rule. Test commands assume the repo's `pytest` and `ruff`/`pyright` invocations.

**Execution order:** Sub-feature #2 first (contained, low-risk), then #1.

---

## SUB-FEATURE #2 — TF-7 xfns pitch-control cache

### Task 1: Value-identity — shared cache == per-family cache (single family)

**Files:**
- Test: `tests/tracking/test_pitch_control_xfns_cache.py` (create)
- Modify: `silly_kicks/tracking/features.py` (the pitch-control-consuming `*_xfns` factories)

- [ ] **Step 1: Write the failing test** — one PC-consuming family, cache vs no-cache byte-identical.

```python
# tests/tracking/test_pitch_control_xfns_cache.py
import inspect
import numpy as np
import pandas as pd
import pytest
from silly_kicks.tracking.pitch_control import PitchControlCache
from silly_kicks.tracking import features as F
# Reuse the existing PC-computable fixture (22-player frames + linkable actions):
from tests.tracking.test_pitch_control_cache import _pc_frames, _pc_actions

# EVERY PC-consuming family (not just one) -- a threading bug in any single factory must be caught.
_PC_FAMILIES = [
    ("pitch_control_xfns", dict(method="voronoi")),
    ("obso_xfns", dict(method="voronoi")),
    ("cover_shadow_xfns", dict()),
    ("gk_influence_xfns", dict()),
    ("player_influence_xfns", dict()),
    ("space_creation_xfns", dict()),
    ("pausa_xfns", dict()),
]  # + xshot_occurrence_xfns iff it builds a PitchControlSurface (Task 1 Step 3)

@pytest.mark.parametrize("factory_name,kwargs", _PC_FAMILIES)
def test_xfns_shared_cache_byte_identical_to_none(factory_name, kwargs, fitted_xt):
    frames, actions = _pc_frames(), _pc_actions()
    gs = [actions]
    factory = getattr(F, factory_name)
    # auto-supply xt to families whose factory takes it (value-identity holds for any fixed xt: both sides share it)
    if "xt" in inspect.signature(factory).parameters:
        kwargs = {**kwargs, "xt": fitted_xt}
    none_out = factory(**kwargs)[0](gs, frames)
    cached_out = factory(**{**kwargs, "pitch_control_cache": PitchControlCache()})[0](gs, frames)
    pd.testing.assert_frame_equal(none_out, cached_out)
```

`fitted_xt` is the `tests/tracking/conftest.py` fixture. If a heavier family cannot run on `_pc_actions` (needs a
pass+receiver or a shot the fixture lacks), EXTEND `_pc_actions`/`_pc_frames` with the missing row rather than
dropping the family from `_PC_FAMILIES` — every PC-consuming family must be exercised. A family whose factory
requires ANOTHER kwarg beyond `method`/`xt` (e.g. `home_team_id`) will `TypeError` on `factory(**kwargs)` — add
that kwarg to the family's `_PC_FAMILIES` entry; that is a param/fixture gap, NOT a threading bug.

- [ ] **Step 2: Run — expect FAIL** (`TypeError: ...() got an unexpected keyword argument 'pitch_control_cache'` for any factory that lacks the param; some params fail before others as you add them).

Run: `python -m pytest tests/tracking/test_pitch_control_xfns_cache.py::test_xfns_shared_cache_byte_identical_to_none -q`

- [ ] **Step 3: Add `pitch_control_cache=None` to each PC-consuming factory + thread it.** For EACH of these factories (inspect each current signature first; add the param only if absent — `xcross_attempt_xfns` already has it):
  `pitch_control_xfns` (features.py:2514), `gk_influence_xfns` (:3414), `cover_shadow_xfns` (:3718), `player_influence_xfns` (:4240), `obso_xfns` (:5406), `space_creation_xfns` (:5705), `pausa_xfns` (:5879), and `xshot_occurrence_xfns` (`_xshot_occurrence.py:929`) — the last two ONLY if they build/consume a `PitchControlSurface` (grep each body for `PitchControlCache`/`pitch_control_at_target`/`compute_pitch_control`; skip if none).

  Pattern (obso shown; identical shape for the others — add the kwarg, thread it into the aggregator call):

```python
def obso_xfns(method="spearman", *, xt=None, pitch_control_cache=None):
    def _helper(actions, frames):
        return add_obso(actions, frames, method=method, xt=xt,
                        pitch_control_cache=pitch_control_cache)   # was: no pitch_control_cache
    _helper.__name__ = f"obso__{method}"
    return [lift_to_states(_helper)]
```

- [ ] **Step 4: Run — expect PASS.** Same command as Step 2.

### Task 2: Mis-keying value-identity — ≥2 families with divergent PC params share one cache

**Files:** Test: `tests/tracking/test_pitch_control_xfns_cache.py` (extend)

- [ ] **Step 1: Write the failing test** — the failure mode a same-params test can't see.

```python
def test_two_families_divergent_params_share_one_cache_exactly(fitted_xt):
    frames, actions = _pc_frames(), _pc_actions()
    from silly_kicks.tracking.features import pitch_control_xfns, obso_xfns
    gs = [actions]
    # TWO DIFFERENT families with DIVERGENT PC params (voronoi vs spearman) sharing ONE cache instance --
    # the real mis-keying mode (two methods of one family, or same params, would pass trivially). Each family
    # must get its OWN surface; a key that omits method/family serves a wrong surface to the second consumer.
    cache = PitchControlCache()
    pc_shared = pitch_control_xfns("voronoi", pitch_control_cache=cache)[0](gs, frames)
    ob_shared = obso_xfns("spearman", xt=fitted_xt, pitch_control_cache=cache)[0](gs, frames)
    pc_solo = pitch_control_xfns("voronoi")[0](gs, frames)
    ob_solo = obso_xfns("spearman", xt=fitted_xt)[0](gs, frames)
    for shared, solo in ((pc_shared, pc_solo), (ob_shared, ob_solo)):
        for col in shared.columns:
            assert np.array_equal(shared[col].to_numpy(), solo[col].to_numpy(), equal_nan=True), col
```

- [ ] **Step 2: Run — expect PASS if Task 1's threading is correct** (the cache keys on frame identity + method/params, so pitch_control's voronoi surface and obso's spearman surface don't collide). If it FAILS, the cache key omits `method`/`params` — fix the key in `pitch_control/_cache.py`, not the test.

Run: `python -m pytest tests/tracking/test_pitch_control_xfns_cache.py::test_two_families_divergent_params_share_one_cache_exactly -q`

### Task 3: Structural perf guard — shared cache ⇒ PC primitive once per unique frame

**Files:** Test: `tests/tracking/test_pitch_control_xfns_cache_perf_budget.py` (create)

- [ ] **Step 1: Write the failing test** — spy the dominant PC primitive; with a shared cache across N families it runs once per unique frame, not N×.

```python
# tests/tracking/test_pitch_control_xfns_cache_perf_budget.py
from tests._perf_structural import call_counter
from silly_kicks.tracking.pitch_control import PitchControlCache
from silly_kicks.tracking.pitch_control import _cache  # where cache.surface() resolves compute_pitch_control
from silly_kicks.tracking.features import pitch_control_xfns, obso_xfns

def test_second_FAMILY_over_shared_cache_recomputes_nothing(monkeypatch, fitted_xt):
    from tests.tracking.test_pitch_control_cache import _pc_frames, _pc_actions
    frames, actions = _pc_frames(), _pc_actions()
    gs = [actions]
    # Spy compute_pitch_control -- the surface primitive cache.surface() calls on a MISS. Confirmed by the
    # cache-reuse test in test_pitch_control_cache.py, and by both families routing through cache.surface.
    # Patch it where the cache module RESOLVES it (per call_counter's docstring).
    calls = call_counter(monkeypatch, _cache, "compute_pitch_control")
    # (a) FAMILY 2 (obso) with its OWN fresh cache MUST hit the primitive (>0) -- proves it uses cache.surface,
    #     so the zero-additional assertion below cannot pass vacuously via a future bypass.
    obso_xfns("voronoi", xt=fitted_xt, pitch_control_cache=PitchControlCache())[0](gs, frames)
    assert calls["n"] > 0, "obso must go through compute_pitch_control (else the cache-hit test is vacuous)"
    # (b) FAMILY 1 pre-populates a SHARED cache; FAMILY 2 over the SAME cache computes ZERO additional.
    shared = PitchControlCache()
    pitch_control_xfns("voronoi", pitch_control_cache=shared)[0](gs, frames)
    baseline = calls["n"]
    obso_xfns("voronoi", xt=fitted_xt, pitch_control_cache=shared)[0](gs, frames)
    assert calls["n"] == baseline, "family 2 recomputed surfaces instead of hitting the shared cache"
```

- [ ] **Step 2: Run — expect PASS** (Task 1 wired the shared cache; obso and pitch_control both request the voronoi surface via `cache.surface(frame, team, method)` on the same frames, so family 2 reuses family 1's cached surfaces). Confirm `compute_pitch_control` is the exact symbol `_cache` resolves (cross-check `tests/tracking/test_pitch_control_cache.py`'s spy). Leg (a)'s `> 0` makes the cache-hit provable, not inferred.

Run: `python -m pytest tests/tracking/test_pitch_control_xfns_cache_perf_budget.py -x -q`

### Task 4: Completeness gate — every PC-Surface-consuming xfns factory accepts the cache

**Files:** Test: `tests/tracking/test_pitch_control_xfns_cache_wiring.py` (create)

- [ ] **Step 1: Write the failing test** — enumerate PC-consuming families; assert each `*_xfns` accepts + threads `pitch_control_cache`.

```python
# tests/tracking/test_pitch_control_xfns_cache_wiring.py
import inspect
from silly_kicks.tracking import features as F

# Families whose AGGREGATOR (add_*) accepts pitch_control_cache -> their xfns factory MUST too.
_PC_CONSUMING_XFNS = [
    "pitch_control_xfns", "obso_xfns", "cover_shadow_xfns", "gk_influence_xfns",
    "player_influence_xfns", "space_creation_xfns", "pausa_xfns",
]  # + xshot_occurrence_xfns / xcross_attempt_xfns iff they build a PitchControlSurface (see Task 1 Step 3)

def test_every_pc_consuming_xfns_factory_accepts_the_cache():
    missing = []
    for name in _PC_CONSUMING_XFNS:
        fn = getattr(F, name)
        if "pitch_control_cache" not in inspect.signature(fn).parameters:
            missing.append(name)
    assert not missing, f"PC-consuming xfns factories missing pitch_control_cache=: {missing}"

def test_pc_consuming_set_matches_aggregators_that_accept_the_cache():
    # META: the list above must equal the add_* aggregators that accept pitch_control_cache,
    # so a NEW PC-consuming family can't be added without appearing here (anti-rot).
    agg_with_cache = {
        n[len("add_"):] + "_xfns"
        for n in dir(F)
        if n.startswith("add_")
        and "pitch_control_cache" in inspect.signature(getattr(F, n)).parameters
    }
    xfns_present = {n for n in _PC_CONSUMING_XFNS if hasattr(F, n)}
    # Every aggregator-with-cache has a correspondingly-named xfns in the wired set (naming may differ;
    # adjust the mapping here if a family's xfns name is not add_<x>->_<x>_xfns).
    assert agg_with_cache.issubset(xfns_present | {"pitch_control_at_target_xfns"}), (
        sorted(agg_with_cache - xfns_present)
    )
```

- [ ] **Step 2: Run — expect PASS after Task 1.** If the meta-test's name mapping (`add_<x>` → `<x>_xfns`) doesn't hold for a family (e.g. `add_pitch_control`→`pitch_control_xfns`, `add_cover_shadows`→`cover_shadow_xfns`), record the explicit mapping in the test rather than loosening the assertion.

Run: `python -m pytest tests/tracking/test_pitch_control_xfns_cache_wiring.py -x -q`

- [ ] **Step 3: Multi-family e2e** (`tests/tracking/test_pitch_control_xfns_cache.py`, extend) — one cache across a realistic multi-family xfn list, asserting **byte-identity** of every family's output vs its own-cache baseline. Scope: OUTPUT correctness across the whole list; the **compute-once perf invariant is owned by Task 3's cross-family test** (a per-family count here would carry the same surface-overlap subtlety Task 3 already resolves) — this e2e is deliberately byte-identity-only.

```python
def test_multi_family_xfn_list_one_cache_byte_identical(fitted_xt):
    from tests.tracking.test_pitch_control_cache import _pc_frames, _pc_actions
    frames, actions = _pc_frames(), _pc_actions()
    gs = [actions]
    def _list(cache):
        kw = {"pitch_control_cache": cache} if cache is not None else {}
        return [F.pitch_control_xfns("voronoi", **kw),
                F.obso_xfns("voronoi", xt=fitted_xt, **kw),
                F.cover_shadow_xfns(**kw)]
    solo = [fam[0](gs, frames) for fam in _list(None)]           # per-family caches (baseline)
    shared = [fam[0](gs, frames) for fam in _list(PitchControlCache())]  # ONE shared cache across the list
    for s, b in zip(shared, solo):
        pd.testing.assert_frame_equal(s, b)
```
Run: `python -m pytest tests/tracking/test_pitch_control_xfns_cache.py::test_multi_family_xfn_list_one_cache_byte_identical -q` → PASS.

### Task 5: Atomic mirrors

**Files:** Modify: `silly_kicks/atomic/tracking/features.py` (`atomic_pitch_control_xfns` + any atomic PC-consumers); Test: extend `tests/tracking/pitch_control/test_atomic_pitch_control.py`

- [ ] **Step 1: Write the failing tests** — atomic PC-consuming xfns accepts the cache AND is byte-identical cache-vs-none (threading correctness, not just signature — mirrors Task 1).

```python
import inspect
import pandas as pd
from silly_kicks.tracking.pitch_control import PitchControlCache
from silly_kicks.atomic.tracking.features import atomic_pitch_control_xfns

def test_atomic_pitch_control_xfns_accepts_cache():
    assert "pitch_control_cache" in inspect.signature(atomic_pitch_control_xfns).parameters

def test_atomic_pc_xfns_shared_cache_byte_identical_to_none():
    # Reuse the atomic PC fixture from tests/tracking/pitch_control/test_atomic_pitch_control.py
    # (atomic actions + frames); import or replicate its builder rather than inventing one.
    from tests.tracking.pitch_control.test_atomic_pitch_control import _atomic_actions, _atomic_frames  # adapt to real names
    actions, frames = _atomic_actions(), _atomic_frames()
    gs = [actions]
    none_out = atomic_pitch_control_xfns("voronoi")[0](gs, frames)
    cached_out = atomic_pitch_control_xfns("voronoi", pitch_control_cache=PitchControlCache())[0](gs, frames)
    pd.testing.assert_frame_equal(none_out, cached_out)
```
(Read `tests/tracking/pitch_control/test_atomic_pitch_control.py` for the actual atomic actions/frames builder names.)

- [ ] **Step 2: Run — expect FAIL** (param absent).
Run: `python -m pytest tests/tracking/pitch_control/test_atomic_pitch_control.py::test_atomic_pitch_control_xfns_accepts_cache -x -q`
- [ ] **Step 3: Add `pitch_control_cache=None` to the atomic PC-consuming factories** + thread into their aggregator calls (same pattern as Task 1).
- [ ] **Step 4: Run — expect PASS.**

### Task 6: Confirm ADR-020 dup-`action_id` retrofit still holds

**Files:** Run the existing gate.

- [ ] **Step 1: Run** `tests/tracking/test_frame_aware_xfns_dup_action_id.py` unchanged.
Run: `python -m pytest tests/tracking/test_frame_aware_xfns_dup_action_id.py -q`
Expected: PASS (the cache param default `None` preserves the existing per-slot resolution).

---

## SUB-FEATURE #1 — feature-column glossary + `describe_level`

### Task 7: `describe_level` (`silly_kicks/reporting.py`)

**Files:** Create `silly_kicks/reporting.py`; Test `tests/test_reporting_describe_level.py`

- [ ] **Step 1: Write the failing tests** — bands both sides, direction flip, NaN, scalar/array/Series.

```python
# tests/test_reporting_describe_level.py
import numpy as np
import pandas as pd
import pytest
from silly_kicks.reporting import describe_level

@pytest.mark.parametrize("z,label", [
    (2.0, "outstanding"), (1.5, "outstanding"), (1.49, "excellent"), (1.0, "excellent"),
    (0.99, "good"), (0.5, "good"), (0.49, "average"), (-0.5, "average"),
    (-0.51, "below average"), (-1.0, "below average"), (-1.01, "poor"), (-5.0, "poor"),
])
def test_bands_higher_is_better(z, label):
    assert describe_level(z) == label

def test_direction_flip():
    # lower-is-better: a high z is BAD.
    assert describe_level(2.0, higher_is_better=False) == "poor"
    assert describe_level(-2.0, higher_is_better=False) == "outstanding"

def test_nan_is_unknown():
    assert describe_level(float("nan")) == "unknown"

def test_vectorised_array_and_series():
    out = describe_level(np.array([2.0, 0.0, np.nan]))
    assert list(out) == ["outstanding", "average", "unknown"]
    s = pd.Series([2.0, -2.0], index=["a", "b"])
    r = describe_level(s)
    assert isinstance(r, pd.Series) and list(r.index) == ["a", "b"]
    assert list(r) == ["outstanding", "poor"]

def test_scalar_returns_str_not_array():
    assert isinstance(describe_level(1.0), str)
```

- [ ] **Step 2: Run — expect FAIL** (`ModuleNotFoundError: silly_kicks.reporting`).
Run: `python -m pytest tests/test_reporting_describe_level.py -x -q`

- [ ] **Step 3: Implement `silly_kicks/reporting.py`.**

```python
"""Reporting / wordalisation helpers. Seed module: describe_level (generic z-score -> verbal band).

Deliberately separate from feature_glossary.py: this is a generic transform, not feature metadata.
See NOTICE for full bibliographic citations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Owner-specified coach-facing descriptive bands (relative-to-cohort, NOT absolute quality;
# z>=0.5 -> "good" is ~69th percentile). Provisional / adjustable.
_BANDS: list[tuple[float, str]] = [
    (1.5, "outstanding"), (1.0, "excellent"), (0.5, "good"),
    (-0.5, "average"), (-1.0, "below average"),
]
_FLOOR = "poor"
_UNKNOWN = "unknown"


def describe_level(z, *, higher_is_better: bool = True):
    """Map a z-score (or array/Series of them) to a verbal band, direction-aware and NaN-safe.

    Parameters
    ----------
    z : float | numpy.ndarray | pandas.Series
    higher_is_better : bool, default True
        When False the sign is flipped internally, so a lower-is-better metric (turnovers, times
        beaten) at high z is correctly "poor", not "outstanding".

    Examples
    --------
    >>> describe_level(1.6)
    'outstanding'
    >>> describe_level(1.6, higher_is_better=False)
    'poor'
    >>> describe_level(float("nan"))
    'unknown'
    """
    is_series = isinstance(z, pd.Series)
    index = z.index if is_series else None
    arr = np.asarray(z, dtype=float)
    scalar = arr.ndim == 0
    flat = arr.reshape(-1)
    score = flat if higher_is_better else -flat
    out = np.full(flat.shape, _FLOOR, dtype=object)
    for thr, label in reversed(_BANDS):          # ascending thresholds; highest satisfied wins
        out[score >= thr] = label
    out[np.isnan(flat)] = _UNKNOWN               # applied last: NaN never mislabels
    if scalar:
        return str(out[0])
    result = out.reshape(arr.shape)
    return pd.Series(result, index=index) if is_series else result
```

- [ ] **Step 4: Run — expect PASS.** Same command as Step 2.

### Task 8: `FeatureColumn` + registry skeleton + accessors (`silly_kicks/feature_glossary.py`)

**Files:** Create `silly_kicks/feature_glossary.py`; Test `tests/test_feature_glossary_registry.py`

- [ ] **Step 1: Write the failing tests** — dataclass shape, accessors, pure JSON w/ schema_version, thin writer.

```python
# tests/test_feature_glossary_registry.py
import json
from silly_kicks.feature_glossary import (
    FeatureColumn, FEATURE_GLOSSARY, GLOSSARY_SCHEMA_VERSION,
    glossary_entry, undocumented_columns, glossary_to_json, dump_glossary,
)

def test_entry_shape_and_lookup():
    # At least one real entry exists post-authoring; here assert the API on a known key added in Task 13.
    assert isinstance(FEATURE_GLOSSARY, dict)
    fc = FeatureColumn(name="x", definition="d", unit="metres", emitting_module="silly_kicks.tracking._packing")
    assert fc.higher_is_better is None and fc.attribution is None

def test_undocumented_columns():
    assert undocumented_columns(["definitely_not_a_real_column"]) == {"definitely_not_a_real_column"}

def test_json_is_pure_and_versioned():
    payload = json.loads(glossary_to_json())
    assert payload["schema_version"] == GLOSSARY_SCHEMA_VERSION
    assert "columns" in payload

def test_dump_glossary_writes(tmp_path):
    p = tmp_path / "g.json"
    dump_glossary(p)
    assert json.loads(p.read_text())["schema_version"] == GLOSSARY_SCHEMA_VERSION
```

- [ ] **Step 2: Run — expect FAIL** (`ModuleNotFoundError`).
Run: `python -m pytest tests/test_feature_glossary_registry.py -x -q`

- [ ] **Step 3: Implement `silly_kicks/feature_glossary.py`** (registry + accessors; `FEATURE_GLOSSARY` starts empty and is filled in Task 13).

```python
"""Machine-readable glossary of every derived feature column silly-kicks emits.

Pure data registry: FeatureColumn records keyed by exact base column name. describe_level lives in
reporting.py (generic transform, not metadata). See NOTICE for citations; attribution tokens are
gate-verified against NOTICE (tests/test_feature_glossary_notice_linkage.py).
"""
from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

Unit = Literal[
    "metres", "m^2", "m/s", "seconds", "degrees",
    "probability", "count", "xT", "xG", "ratio", "dimensionless",
]

GLOSSARY_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class FeatureColumn:
    name: str
    definition: str
    unit: Unit
    emitting_module: str
    attribution: str | None = None
    higher_is_better: bool | None = None


def _register(*entries: FeatureColumn) -> dict[str, FeatureColumn]:
    out: dict[str, FeatureColumn] = {}
    for e in entries:
        if e.name in out:
            raise ValueError(f"duplicate glossary entry: {e.name}")
        out[e.name] = e
    return out


# Filled in Task 13 (authored in per-module batches). Starts empty.
FEATURE_GLOSSARY: dict[str, FeatureColumn] = _register(
    # <entries added in Task 13>
)


def glossary_entry(name: str) -> FeatureColumn:
    return FEATURE_GLOSSARY[name]


def undocumented_columns(cols: Iterable[str]) -> set[str]:
    return {c for c in cols if c not in FEATURE_GLOSSARY}


def glossary_to_json() -> str:
    """Pure: no I/O. {schema_version, columns:{name: {...}}}."""
    columns = {
        fc.name: {
            "definition": fc.definition, "unit": fc.unit,
            "emitting_module": fc.emitting_module, "attribution": fc.attribution,
            "higher_is_better": fc.higher_is_better,
        }
        for fc in FEATURE_GLOSSARY.values()
    }
    return json.dumps({"schema_version": GLOSSARY_SCHEMA_VERSION, "columns": columns}, indent=2, sort_keys=True)


def dump_glossary(path) -> None:
    """Thin writer over glossary_to_json() (the only impure symbol here)."""
    Path(path).write_text(glossary_to_json(), encoding="utf-8")


def emitting_module_is_importable(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False
```

- [ ] **Step 4: Run — expect PASS** (Step 1's `test_entry_shape_and_lookup` asserts only API + a constructed record; the empty registry is fine until Task 13).
Run: `python -m pytest tests/test_feature_glossary_registry.py -x -q`

### Task 9: Producer discovery (inspection + `__all__`-less fallback)

**Files:** Create `tests/invariants/glossary_discovery.py` (shared discovery helper for the gate); Test `tests/invariants/test_glossary_discovery.py`

- [ ] **Step 1: Write the failing test** — discovery finds `add_*`/`*_xfns` across packages, incl. an `__all__`-less planted module.

```python
# tests/invariants/test_glossary_discovery.py
import types
from tests.invariants import glossary_discovery as G

def test_finds_known_producers():
    prods = G.discover_public_column_producers()
    names = {q.rsplit(".", 1)[1] for q in prods}
    assert "add_obso" in names and "obso_xfns" in names and "add_packing" in names

def test_no_unexpected_import_failures():
    # A module that fails to import in CI silently drops ALL its columns -> coverage hole. Surface it.
    G.discover_public_column_producers()  # walks all packages, populating _import_failures
    bad = G.unexpected_import_failures()
    assert not bad, f"modules failing to import (columns silently dropped), not in the optional-extra allowlist: {bad}"

def test_all_less_module_is_discovered(monkeypatch):
    mod = types.ModuleType("silly_kicks.tracking._planted_glossary_probe")
    def add_planted(actions, frames):  # add_* name shape, no __all__ on the module
        return actions
    add_planted.__module__ = mod.__name__
    mod.add_planted = add_planted
    monkeypatch.setitem(__import__("sys").modules, mod.__name__, mod)
    monkeypatch.setattr(G, "_extra_probe_modules", [mod], raising=False)
    found = G.discover_public_column_producers(extra_modules=[mod])
    assert any(q.endswith("._planted_glossary_probe.add_planted") for q in found)
```

- [ ] **Step 2: Run — expect FAIL** (`ModuleNotFoundError: tests.invariants.glossary_discovery`).
Run: `python -m pytest tests/invariants/test_glossary_discovery.py -x -q`

- [ ] **Step 3: Implement discovery**, reusing the id-scalar `_public_names` idiom.

```python
# tests/invariants/glossary_discovery.py
"""Discover public derived-column producers (add_*/*_xfns) by inspection, __all__-less-safe.

Mirrors tests/invariants/conftest_id_scalar.py::_public_names (surface = __all__ if declared, else
public callables defined in-module). The __all__-less fallback is defensive Chesterton's-fence; the
load-bearing dependency is the add_*/*_xfns NAME SHAPE (vaep fs.* are the recorded exception, enumerated
by the default-list run-and-diff leg, not here).
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil

PACKAGES = ["silly_kicks.tracking", "silly_kicks.atomic.tracking",
            "silly_kicks.spadl", "silly_kicks.atomic.spadl"]  # vaep handled by list-invocation leg


def _is_producer_name(name: str) -> bool:
    return name.startswith("add_") or name.endswith("_xfns")


# NOTE: this is a deliberate COPY of tests/invariants/conftest_id_scalar.py::_public_names (a conftest is
# awkward to import cross-directory). Keep the two in sync -- both encode the __all__-else-vars rule.
def _public_names(mod) -> list[str]:
    declared = getattr(mod, "__all__", None)
    if declared:
        return list(declared)
    return [
        n for n, o in vars(mod).items()
        if not n.startswith("_") and (inspect.isfunction(o) or inspect.isclass(o))
        and getattr(o, "__module__", None) == mod.__name__
    ]


# Modules that legitimately require an optional extra to import (accessible-space, xgboost, ...).
# Populate from the ACTUAL failures the first run surfaces -- do NOT pre-guess; an over-broad allowlist
# re-hides the coverage hole. A failing module drops ALL its columns from the gate, so this must be tight.
_OPTIONAL_IMPORT_MODULES: set[str] = set()
_import_failures: dict[str, str] = {}


def _iter_modules(pkg_name):
    pkg = importlib.import_module(pkg_name)
    yield pkg
    if hasattr(pkg, "__path__"):
        for info in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
            try:
                yield importlib.import_module(info.name)
            except Exception as exc:  # RECORD, don't silently drop -- a failing module drops its columns
                _import_failures[info.name] = repr(exc)


def unexpected_import_failures() -> dict[str, str]:
    """Import failures NOT explained by a recorded optional extra -- each silently drops that module's columns."""
    return {m: e for m, e in _import_failures.items() if m not in _OPTIONAL_IMPORT_MODULES}


def discover_public_column_producers(*, extra_modules=None) -> dict[str, str]:
    """{defining module.qualname: function-name} for public add_*/*_xfns across PACKAGES."""
    found: dict[str, str] = {}
    mods = [m for pkg in PACKAGES for m in _iter_modules(pkg)]
    if extra_modules:
        mods += list(extra_modules)
    for mod in mods:
        for name in _public_names(mod):
            obj = getattr(mod, name, None)
            if not inspect.isfunction(obj) or not _is_producer_name(name):
                continue
            found[f"{obj.__module__}.{obj.__qualname__}"] = name
    return found
```

- [ ] **Step 4: Run — expect PASS.** Same command as Step 2.

### Task 10: Run-and-diff harnesses (emitted-column collection, 5 legs)

**Files:** Create `tests/invariants/glossary_emitted_columns.py`; Test `tests/invariants/test_glossary_emitted_columns.py`

- [ ] **Step 1: Write the failing test** — the harness returns a non-empty base-column set per surface.

```python
# tests/invariants/test_glossary_emitted_columns.py
from tests.invariants import glossary_emitted_columns as E

def test_union_has_known_tracking_columns():
    cols = E.emitted_columns()  # union across all 5 legs, base-normalised
    assert "packing_made" in cols and "defensive_credit_net" in cols
    assert all(isinstance(c, str) for c in cols)

def test_each_leg_is_non_vacuous():
    # THE anti-lie guard: a stubbed leg (return set()) would silently under-cover with green CI. Each leg must
    # be non-empty with a known-column anchor. LIMITATION (honest): an anchor proves non-empty + contains-that-
    # column; a leg returning {anchor} + only HALF its real columns STILL passes -- partial-leg holes are
    # uncatchable without a second independent enumeration. Read "non-vacuous", NOT "complete".
    # The per-leg functions return RAW slotted names (base-normalisation happens at the emitted_columns() union),
    # so normalise here before matching base-name anchors.
    from tests.invariants.glossary_emitted_columns import _base
    assert "packing_made" in {_base(c) for c in E._tracking_add_star_columns()}
    assert "pitch_control_at_target__spearman" in {_base(c) for c in E._xfns_columns()}
    assert "start_coord_source" in {_base(c) for c in E._spadl_enricher_columns()}   # add_restart_coordinates (verify)
    assert E._vaep_columns(), "vaep leg empty (stubbed?) -- anchor with a real xfns_default column"
```

- [ ] **Step 2: Run — expect FAIL** (`ModuleNotFoundError`).
Run: `python -m pytest tests/invariants/test_glossary_emitted_columns.py -x -q`

- [ ] **Step 3: Implement the harness.** Five legs, each running its producers at default config on a fixture and diffing added columns. Reuse the liveness scene for tracking `add_*`; build gamestates for `*_xfns`; a SPADL/atomic fixture for enrichers; the vaep default lists for the vaep leg. Base-normalise per-slot xfn columns (strip the gamestate-slot marker — inspect an actual emitted xfns column first to learn the marker; if xfns emit the base name unchanged, normalisation is a no-op).

```python
# tests/invariants/glossary_emitted_columns.py
"""Union of derived columns emitted by every default-config producer (run-and-diff), base-normalised.

COMPLETENESS CEILING (honest): the coverage gate is only as complete as this harness. The per-leg
non-vacuity anchors (test_each_leg_is_non_vacuous) catch a fully-STUBBED leg, but NOT a PARTIAL one --
a leg that collects some of its real columns and drops the rest passes, and the dropped columns silently
never get required. Catching that would need a second independent enumeration (out of scope). So every
leg MUST genuinely run all its default-config producers and diff, not just enough to clear the anchor.
"""
from __future__ import annotations

import re

_SLOT_SUFFIX = re.compile(r"_a\d+$")  # gamestate-slot marker; VERIFY against a real xfns output in Step 3


def _base(col: str) -> str:
    return _SLOT_SUFFIX.sub("", col)


def _tracking_add_star_columns() -> set[str]:
    # The liveness gate runs each tracking add_* on a purpose-built scene: ENTRIES is a
    # dict[str, <runner>] and each value (e.g. _run_defensive_credit, _run_das) builds its own scene,
    # runs one aggregator, and returns the enriched frame. The liveness gate already computes the columns
    # each aggregator ADDS (for its non-null check) -- reuse that logic: for each ENTRIES value, run it and
    # diff enriched-columns minus input-columns. READ tests/tracking/test_aggregator_column_liveness.py for
    # the exact ENTRIES value shape + its added-column diff, and factor it into `added_columns_for(entry)`.
    from tests.tracking.test_aggregator_column_liveness import ENTRIES  # dict[str, runner]
    cols: set[str] = set()
    for _name, entry in ENTRIES.items():
        cols |= added_columns_for(entry)  # adapt to the actual runner shape (see note above)
    return cols


def _xfns_columns() -> set[str]:
    # Build gamestates once; run each default xfn list / default *_xfns(); diff.
    ...  # implement using tests.tracking fixtures + the default_xfns module constants


def _spadl_enricher_columns() -> set[str]:
    ...  # run add_* enrichers in spadl/ + atomic.spadl on a SPADL/atomic fixture


def _vaep_columns() -> set[str]:
    from silly_kicks.vaep.base import xfns_default
    from silly_kicks.vaep.hybrid import hybrid_xfns_default
    ...  # run the lists on a gamestates fixture; collect columns


def emitted_columns() -> set[str]:
    raw = _tracking_add_star_columns() | _xfns_columns() | _spadl_enricher_columns() | _vaep_columns()
    return {_base(c) for c in raw}
```

  NOTE (harness effort, per spec §1.3): the `_xfns`, `_spadl_enricher`, and `_vaep` legs are NEW run-and-diff harnesses with distinct fixtures — budget real time here; only the tracking `add_*` leg reuses liveness. Fill each `...` with concrete fixture wiring modelled on the liveness gate.

- [ ] **Step 4: Run — expect PASS** once the legs are wired.
Run: `python -m pytest tests/invariants/test_glossary_emitted_columns.py -x -q`

### Task 11: Coverage gate (assertions 1–4 + name-shape exception + `__all__`-less meta-test)

**Files:** Test `tests/test_feature_glossary_coverage.py`

- [ ] **Step 1: Write the gate** (initially expected to FAIL on assertion 1 until Task 13 authors entries).

```python
# tests/test_feature_glossary_coverage.py
from silly_kicks.feature_glossary import FEATURE_GLOSSARY, emitting_module_is_importable
from tests.invariants.glossary_emitted_columns import emitted_columns
from tests.invariants.glossary_discovery import discover_public_column_producers

_NON_CONFORMING_PRODUCERS = {  # recorded exception: vaep fs.* aren't add_*/*_xfns (enumerated by list-invocation)
    "silly_kicks.vaep.base.xfns_default", "silly_kicks.vaep.hybrid.hybrid_xfns_default",
}
# Metrics genuinely computed inline in features.py (NO separate _compute module) — add CONSCIOUSLY, never lazily.
_FEATURES_HOMED_ALLOWLIST: set[str] = set()

def test_no_undocumented_columns():
    missing = emitted_columns() - set(FEATURE_GLOSSARY)
    assert not missing, f"emitted columns with no glossary entry: {sorted(missing)}"

def test_no_stale_entries():
    stale = set(FEATURE_GLOSSARY) - emitted_columns()
    assert not stale, f"glossary entries for non-emitted columns: {sorted(stale)}"

def test_emitting_module_importable_and_not_lazily_features():
    bad_import = [fc.name for fc in FEATURE_GLOSSARY.values()
                  if not emitting_module_is_importable(fc.emitting_module)]
    assert not bad_import, f"non-importable emitting_module: {bad_import}"
    # Enforce the home-module convention (spec §1.1): don't lazily point the catalogue at the features.py
    # monolith (importable but zero provenance). Genuinely-features-homed metrics go in the allowlist.
    lazy = [fc.name for fc in FEATURE_GLOSSARY.values()
            if fc.emitting_module.endswith(".features") and fc.name not in _FEATURES_HOMED_ALLOWLIST]
    assert not lazy, (
        "emitting_module points at the features.py monolith (no provenance). Use the metric's home/compute "
        f"module (_packing/_obso/...), or add to _FEATURES_HOMED_ALLOWLIST if it has none: {lazy}"
    )
    # NOTE (honest): beyond importable + non-features, emitting_module is DOCUMENTATION, not gate-verified --
    # it can still name the WRONG home module and pass. The monolithic features.py (every producer's
    # __module__ == ...features) makes run-and-diff attribution impossible (spec §1.1 / rev-2 review).

def test_name_shape_completeness_is_a_documented_limitation():
    # HONEST LIMITATION (not anti-rot): discovery finds producers only by the add_*/*_xfns NAME SHAPE
    # (Task 9). A public function emitting derived columns but named otherwise AND not in the exception set
    # is invisible to the gate -- detecting it needs running every public function on a fixture (out of scope).
    # This PINS the known exception set so KNOWN non-conforming producers stay tracked; a NEW one is an
    # accepted blind spot, documented here rather than dressed up as a guard that catches nothing.
    assert _NON_CONFORMING_PRODUCERS
    for q in _NON_CONFORMING_PRODUCERS:
        assert q.count(".") >= 2, q  # real dotted qualnames, not typos
```

- [ ] **Step 2: Run — expect FAIL on `test_no_undocumented_columns`** (registry empty).
Run: `python -m pytest tests/test_feature_glossary_coverage.py -q`

- [ ] **Step 3: Add the `__all__`-less discovery meta-test** (proves the fallback is live — mirrors `test_discovery_sees_a_module_that_declares_no___all__`). Already covered by `tests/invariants/test_glossary_discovery.py::test_all_less_module_is_discovered` (Task 9); reference it here in a comment so the coverage gate's completeness story is traceable.

- [ ] **Step 4: (gate goes green in Task 13)** — leave failing for now; Task 13 authors entries until all four pass.

### Task 12: `attribution` ↔ NOTICE linkage gate

**Files:** Test `tests/test_feature_glossary_notice_linkage.py`

- [ ] **Step 1: Write the test.**

```python
# tests/test_feature_glossary_notice_linkage.py
from pathlib import Path
from silly_kicks.feature_glossary import FEATURE_GLOSSARY

def test_every_attribution_token_is_in_notice():
    notice = Path("NOTICE").read_text(encoding="utf-8")
    missing = sorted({fc.attribution for fc in FEATURE_GLOSSARY.values()
                      if fc.attribution is not None and fc.attribution not in notice})
    assert not missing, f"attribution tokens absent from NOTICE (add the citation): {missing}"
```

- [ ] **Step 2: Run — expect PASS** now (empty registry ⇒ no tokens); it becomes load-bearing in Task 13 as attributed entries are authored.
Run: `python -m pytest tests/test_feature_glossary_notice_linkage.py -q`

### Task 13: Author the `FEATURE_GLOSSARY` entries (gate-driven, per-module batches)

**Files:** Modify `silly_kicks/feature_glossary.py` (`FEATURE_GLOSSARY`); Modify `NOTICE` (fill any citation gaps the linkage gate surfaces)

- [ ] **Step 1: Run the coverage gate to list the work.**
Run: `python -m pytest tests/test_feature_glossary_coverage.py::test_no_undocumented_columns -q`
The failure message lists every undocumented column — that is the authoring worklist.

- [ ] **Step 2: Author entries in per-module batches** following this pattern (one real example each of a house-original and an attributed column):

```python
FEATURE_GLOSSARY: dict[str, FeatureColumn] = _register(
    FeatureColumn(
        name="packing_made",
        definition="Number of defenders the completed pass/carry plays past (bypasses) toward goal.",
        unit="count", emitting_module="silly_kicks.tracking._packing",
        attribution="arXiv:2603.28916", higher_is_better=True,
    ),
    FeatureColumn(
        name="defensive_credit_net",
        definition="Net signed defensive credit attributed to the defending team on this action (plus minus minus).",
        unit="xT", emitting_module="silly_kicks.tracking.defensive_credit._orchestration",
        attribution="arXiv:2606.19931", higher_is_better=True,
    ),
    # ... one FeatureColumn per undocumented column from Step 1, per-module batch ...
)
```
  - `unit` MUST be a value from the `Unit` `Literal` (extend the `Literal` in Task 8 only if a genuinely new unit is needed).
  - `emitting_module` = the metric's home/computation module (`_packing`, `_obso`, `defensive_credit._orchestration`, …), NOT `features.py`.
  - `attribution` = the citation token already in the feature's NOTICE entry / docstring, or `None` for house-original.
  - `higher_is_better` = `True`/`False` where unambiguous, else `None` (perspective-dependent).

- [ ] **Step 3: Re-run the gate; iterate until all four coverage assertions + the NOTICE-linkage gate are green.**
Run: `python -m pytest tests/test_feature_glossary_coverage.py tests/test_feature_glossary_notice_linkage.py -q`
Expected: PASS. If NOTICE-linkage fails, add the missing citation to `NOTICE` (ADR-005 hygiene) — do not blank the attribution.

### Task 14: Roundtrip e2e (both directions)

**Files:** Test `tests/test_feature_glossary_roundtrip.py`

- [ ] **Step 1: Write the test** — dump → reload → describe_level over one True + one False real column.

```python
# tests/test_feature_glossary_roundtrip.py
import json
from silly_kicks.feature_glossary import dump_glossary, GLOSSARY_SCHEMA_VERSION, FEATURE_GLOSSARY
from silly_kicks.reporting import describe_level

def test_dump_reload_and_direction_flip(tmp_path):
    p = tmp_path / "g.json"
    dump_glossary(p)
    payload = json.loads(p.read_text())
    assert payload["schema_version"] == GLOSSARY_SCHEMA_VERSION
    assert set(payload["columns"]) == set(FEATURE_GLOSSARY)  # every entry survives
    higher = next(c for c, v in payload["columns"].items() if v["higher_is_better"] is True)
    lower = next(c for c, v in payload["columns"].items() if v["higher_is_better"] is False)
    # A strongly-positive z reads as top band for higher-is-better, bottom band for lower-is-better.
    assert describe_level(2.0, higher_is_better=payload["columns"][higher]["higher_is_better"]) == "outstanding"
    assert describe_level(2.0, higher_is_better=payload["columns"][lower]["higher_is_better"]) == "poor"
```

- [ ] **Step 2: Run — expect PASS** (requires at least one `True` and one `False` entry authored in Task 13; if none exists, that itself is a finding — set direction on at least one clearly-lower-is-better metric, e.g. a turnovers/conceded column).
Run: `python -m pytest tests/test_feature_glossary_roundtrip.py -x -q`

### Task 15: Public exports + `Examples` gate

**Files:** Modify `silly_kicks/__init__.py` (or confirm module-level public access); Test `tests/test_public_api_examples.py` (register the 3 new modules/functions)

- [ ] **Step 1:** Add `reporting.py` + `feature_glossary.py` to the public-API-examples surface (`_PUBLIC_MODULE_FILES` in `tests/test_public_api_examples.py`) so `describe_level`, `glossary_entry`, `dump_glossary` are gated for an `Examples` section.
- [ ] **Step 2: Run** `python -m pytest tests/test_public_api_examples.py -q` — expect FAIL for any public function lacking an `Examples` section.
- [ ] **Step 3:** `describe_level` already has doctest `Examples` (Task 7). Add short `Examples` (literal blocks or `>>>`) to `glossary_entry`/`dump_glossary`/`glossary_to_json` as needed.
- [ ] **Step 4: Run — expect PASS.**

---

## Task 16: Commit-prep (STOP before commit — owner approval)

**Files:** `CHANGELOG.md`, `CLAUDE.md`, `TODO.md`, `pyproject.toml`, `silly_kicks/__init__.py`, `uv.lock`, `docs/c4/architecture.{dsl,html}`, new ADRs.

- [ ] **Step 1: Full non-e2e suite green.**
Run: `python -m pytest tests/ -m "not e2e" -q`
Expected: all pass (incl. the new coverage/linkage/roundtrip/cache gates).
- [ ] **Step 2: `ruff check` + `ruff format --check` + full-package `pyright`** — all clean (recall the lint job runs `ruff format --check`; new hand-written test scenes are the usual miss).
- [ ] **Step 3: Version bump** to the next-free MINOR across the 5 canonical spots (assigned NOW at commit-prep, not before): `pyproject.toml`, `silly_kicks/__init__.py`, `uv.lock`, `CHANGELOG.md` header, `TODO.md` current-release line. Groom TODO: remove the two shipped On-Deck rows (glossary+describe_level; TF-7 xfns cache).
- [ ] **Step 4: Draft ADRs** — a new ADR for the glossary governance convention (inspection-enumerated, NOTICE-linked, typed unit) + an **ADR-008 amendment** for the xfns cache. Add NOTICE entries only if Task 13 surfaced gaps.
- [ ] **Step 5: Regenerate C4** (`docs/c4/architecture.{dsl,html}`) — new `feature_glossary`/`reporting` components; **aggregator count stays 31**; respect the 200-char box-description cap.
- [ ] **Step 6: Run `/final-review`** (mandatory), then **STOP** and present `git status` + the proposed single commit message for **explicit owner approval**. Do NOT commit, push, tag, or PR without it.

---

## Self-review notes (author)

- **Plan-review-1 (analysis session) folded in:** (1) every run-and-diff leg now has a per-leg
  non-vacuity anchor (Task 10 `test_each_leg_is_non_vacuous`) so a stubbed leg can't pass silently;
  (2) #2 is tested cross-family — value-identity parametrized over ALL PC families (Task 1),
  mis-keying is two different families with divergent methods (Task 2), perf is a second *family*
  reusing the cache (Task 3), plus a multi-family e2e (Task 4 Step 3); (3) `emitting_module` gains a
  `!endswith(".features")` guard + conscious allowlist (Task 11) and is honestly labelled
  documentation-beyond-that; (4) the vacuous name-shape "anti-rot" test is downgraded to an honest
  documented limitation (Task 11); (5) discovery collects import failures against an optional-extra
  allowlist instead of swallowing them (Task 9); (6) atomic gains a value-identity test (Task 5);
  (7) test-name/run-command typos reconciled; (8) `_public_names` copy cross-linked to conftest_id_scalar.
- **Spec coverage:** #2 §2.1–2.3 → Tasks 1–6; #1 §1.1 registry → Task 8; §1.2 describe_level → Task 7; §1.3 coverage gate (discovery/harness/assertions/meta) → Tasks 9–11,13; §1.4 NOTICE gate → Task 12; §1.5 authoring → Task 13; §1.6 seam is named (no task — deferred by design); roundtrip e2e → Task 14; C4/version/ADR/retrain → Task 16.
- **Known open items resolved at implementation** (spec "Open items"): the exact `_SLOT_SUFFIX` marker (Task 10 Step 3 verifies against real output), whether `pausa_xfns`/`xshot_occurrence_xfns` build a `PitchControlSurface` (Task 1 Step 3 greps each body), and the `Unit` vocabulary (extended in Task 13 only as needed).
- **Plan-review-2 (analysis session) folded in — verdict "execute it":** (1) Task 3's cache-hit is now
  PROVABLE not inferred (family 2 on its own cache must compute `>0`, then `0` additional over the shared
  cache); the spy target is resolved to `compute_pitch_control` (the reviewer traced obso + pitch_control both
  route through `cache.surface → compute_pitch_control`); (2) Task 4 Step 3 is byte-identity-only, with Task 3
  explicitly owning the compute-once perf invariant; (3) Task 10 anchors `_base`-normalise (raw legs emit
  slotted names) + carry an honest "non-vacuous ≠ complete (partial-leg holes uncatchable)" note in the harness
  docstring; (4) Task 1 notes families needing an extra required kwarg.
- **Flagged-not-fabricated:** two deliberate "resolve-against-real-code" markers remain, each with a named
  source and a non-vacuity guard, not hand-wave: (a) Task 10's `added_columns_for(entry)` + the three `...`
  harness legs (`_xfns`/`_spadl_enricher`/`_vaep`) — the real run-and-diff work the spec §1.3 calls out as five
  distinct harnesses, wired against the actual liveness `ENTRIES` dict + `tests.tracking` fixtures; (b) Task 13's
  per-column authoring loop — a data task defined by the coverage gate (author until green). Both are the honest
  plan/execution boundary, verified against the real APIs (`_pc_frames`/`_pc_actions`,
  `call_counter(monkeypatch, module, name)`, `_cache.compute_pitch_control`, `ENTRIES: dict`).
