# NaN-safety contract for enrichment helpers + `goalkeeper_ids` feature Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Codify a NaN-safety contract for silly-kicks's 10 public enrichment helpers (5 standard + 5 atomic), enforced via auto-discovered CI gate; fix the 2 confirmed crash sites + 2 latent crash sites; add an opt-in `goalkeeper_ids` parameter on `add_gk_role` (std + atomic) closing the lakehouse coverage gap; ship as silly-kicks 2.5.0.

**Architecture:** New private module `silly_kicks/_nan_safety.py` provides `@nan_safe_enrichment` decorator that sets `fn._nan_safe = True`. Two new test files (`test_enrichment_nan_safety.py` synthetic-fuzz + `test_enrichment_provider_e2e.py` cross-provider e2e) auto-discover decorated helpers via `inspect.getmembers(module, inspect.isfunction)` filtered on the marker. Registry-floor sanity assertions catch silent discovery breakage. ADR-003 captures the contract; CLAUDE.md amendment makes it a project-wide rule.

**Tech Stack:** Python 3.10+, pandas, numpy, pytest, ruff 0.15.7, pyright 1.1.395, pandas-stubs 2.3.3.260113.

**Spec:** `docs/superpowers/specs/2026-04-30-nan-safety-enrichment-helpers-design.md`

**Commit policy:** Per project CLAUDE.md memory — *literally ONE commit per branch; explicit user approval before that one commit*. This plan therefore does NOT have intermediate `git commit` steps. All changes accumulate in the working tree; a single commit at the very end (Task 21) gathers everything after user approval.

**Test count target:** ~928 passing (884 baseline + ~44 net delta: ~14 fuzz + ~20 e2e + ~10 goalkeeper_ids).

---

## Phase 1 — Setup

### Task 1: Create feature branch

**Files:**
- (none — git operation only)

- [ ] **Step 1.1: Verify clean working tree**

Run:
```bash
git status --short
```
Expected: only the two pre-existing untracked items (`README.md.backup`, `uv.lock`) plus the spec file written this session (`docs/superpowers/specs/2026-04-30-nan-safety-enrichment-helpers-design.md`). No modified tracked files.

- [ ] **Step 1.2: Create and switch to the feature branch from main**

Run:
```bash
git checkout main
git pull
git checkout -b feat/nan-safety-enrichment-helpers
```
Expected: `Switched to a new branch 'feat/nan-safety-enrichment-helpers'`. Branch is on top of latest main (which includes the just-merged 2.4.0 commit `198b85f`).

- [ ] **Step 1.3: Verify branch position**

Run:
```bash
git rev-parse --abbrev-ref HEAD && git log -1 --oneline
```
Expected: branch `feat/nan-safety-enrichment-helpers`; last commit is the 2.4.0 merge.

### Task 2: Capture pytest baseline

**Files:**
- (none — verification only)

- [ ] **Step 2.1: Run baseline pytest count**

Run:
```bash
uv run pytest tests/ -m "not e2e" --tb=short -q
```
Expected: ends with `884 passed, 4 deselected`. Note the exact number — subsequent expectations adjust by `+~44`.

---

## Phase 2 — Decorator module

### Task 3: Create `silly_kicks/_nan_safety.py`

**Files:**
- Create: `silly_kicks/_nan_safety.py`

- [ ] **Step 3.1: Write the decorator module**

Create `silly_kicks/_nan_safety.py` with this exact content:

```python
"""NaN-safety contract decorator for enrichment helpers (ADR-003).

Decorated functions claim that they tolerate NaN in caller-supplied
identifier columns: NaN identifiers route to the documented per-row
default rather than crashing.

The CI gates at ``tests/test_enrichment_nan_safety.py`` and
``tests/test_enrichment_provider_e2e.py`` auto-discover decorated
helpers via the ``_nan_safe`` attribute set by this decorator.
"""

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T", bound=Callable)


def nan_safe_enrichment(fn: T) -> T:
    """Marker decorator declaring fn satisfies the NaN-safety contract.

    See ADR-003 for the contract definition (caller-supplied NaN
    identifiers route to per-row default; helper does not crash).

    Examples
    --------
    Mark an enrichment helper as NaN-safe::

        from silly_kicks._nan_safety import nan_safe_enrichment

        @nan_safe_enrichment
        def my_enrichment(actions):
            return enriched_actions
    """
    fn._nan_safe = True  # type: ignore[attr-defined]
    return fn
```

- [ ] **Step 3.2: Smoke-import the decorator**

Run:
```bash
uv run python -c "
from silly_kicks._nan_safety import nan_safe_enrichment

@nan_safe_enrichment
def f(x): return x

assert getattr(f, '_nan_safe', False) is True
print('decorator OK')
"
```
Expected output: `decorator OK`.

---

## Phase 3 — TDD setup (RED tests written first)

### Task 4: Create `tests/test_enrichment_nan_safety.py` (synthetic NaN fuzz)

**Files:**
- Create: `tests/test_enrichment_nan_safety.py`

- [ ] **Step 4.1: Write the test file with auto-discovery + fixtures + sanity**

Create `tests/test_enrichment_nan_safety.py` with this exact content:

```python
"""NaN-safety contract enforcement (ADR-003).

Auto-discovers every helper decorated with @nan_safe_enrichment and runs
it against a synthetic NaN-laced fixture. Fails fast if a helper crashes
on NaN-input rows in caller-supplied identifier columns.

Catches: future contributor adds a public enrichment helper without writing
a NaN-safety test. Auto-discovery here covers them automatically when they
opt in via the @nan_safe_enrichment decorator.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import silly_kicks.atomic.spadl.config as atomic_spadlcfg
import silly_kicks.atomic.spadl.utils as atomic_utils
import silly_kicks.spadl.config as spadlcfg
import silly_kicks.spadl.utils as std_utils


def _discover(module) -> tuple:
    """Return all functions in `module` whose `_nan_safe` attribute is True."""
    return tuple(
        fn
        for _, fn in inspect.getmembers(module, inspect.isfunction)
        if getattr(fn, "_nan_safe", False)
    )


STD_ENRICHMENTS = _discover(std_utils)
ATOMIC_ENRICHMENTS = _discover(atomic_utils)


# ---------------------------------------------------------------------------
# Registry-floor sanity — bulletproofs the auto-discovery mechanism itself.
# If a future refactor accidentally renames `_nan_safe` or breaks the
# decoration on every helper at once, these tests fail explicitly rather
# than silently running zero parametrize cases.
# ---------------------------------------------------------------------------


def test_registry_nonempty_std() -> None:
    """At least 5 @nan_safe_enrichment helpers in silly_kicks.spadl.utils."""
    assert len(STD_ENRICHMENTS) >= 5, (
        f"Expected ≥5 @nan_safe_enrichment helpers in silly_kicks.spadl.utils; "
        f"found {len(STD_ENRICHMENTS)}: {[fn.__name__ for fn in STD_ENRICHMENTS]}. "
        f"Did the marker name change or a helper lose its decoration?"
    )


def test_registry_nonempty_atomic() -> None:
    """At least 5 @nan_safe_enrichment helpers in silly_kicks.atomic.spadl.utils."""
    assert len(ATOMIC_ENRICHMENTS) >= 5, (
        f"Expected ≥5 @nan_safe_enrichment helpers in silly_kicks.atomic.spadl.utils; "
        f"found {len(ATOMIC_ENRICHMENTS)}: {[fn.__name__ for fn in ATOMIC_ENRICHMENTS]}. "
        f"Did the marker name change or a helper lose its decoration?"
    )


# ---------------------------------------------------------------------------
# Fixtures: synthetic NaN-laced SPADL frames covering boundary cases:
# - First/middle/last row NaN player_id (positional boundaries)
# - The IDSSE-failure pattern: keeper-action with NaN player_id preceding
#   a shot by the other team.
# - A distribution-eligible row with NaN coordinates (latent crash pattern
#   in add_gk_distribution_metrics).
# ---------------------------------------------------------------------------


@pytest.fixture
def std_nan_laced_actions() -> pd.DataFrame:
    """10-row synthetic standard-SPADL fixture with strategic NaN placements."""
    pass_id = spadlcfg.actiontype_id["pass"]
    keeper_save_id = spadlcfg.actiontype_id["keeper_save"]
    shot_id = spadlcfg.actiontype_id["shot"]
    success_id = spadlcfg.result_id["success"]
    foot_id = spadlcfg.bodypart_id["foot"]

    return pd.DataFrame(
        {
            "game_id": [1] * 10,
            "period_id": [1] * 10,
            "action_id": list(range(10)),
            "team_id": [10, 20, 20, 10, 20, 10, 10, 20, 20, 10],
            "player_id": pd.array(
                [
                    np.nan,  # row 0: NaN at first position (boundary)
                    201.0,
                    np.nan,  # row 2: KEEPER ACTION with NaN player_id (IDSSE pattern)
                    101.0,
                    202.0,
                    np.nan,  # row 5: NaN mid-stream
                    102.0,  # row 6: SHOT (preceded by NaN-keeper at row 2)
                    201.0,
                    202.0,
                    np.nan,  # row 9: NaN at last position (boundary)
                ],
                dtype="float64",
            ),
            "type_id": [
                pass_id,
                pass_id,
                keeper_save_id,  # row 2: defending keeper
                pass_id,
                pass_id,
                pass_id,
                shot_id,  # row 6: shot — triggers add_pre_shot_gk_context
                pass_id,
                pass_id,
                pass_id,
            ],
            "result_id": [success_id] * 10,
            "result_name": ["success"] * 10,
            "bodypart_id": [foot_id] * 10,
            "bodypart_name": ["foot"] * 10,
            "type_name": ["pass"] * 10,  # placeholder (overridden in some helpers)
            "time_seconds": [float(i) for i in range(10)],
            "start_x": [
                10.0,
                20.0,
                30.0,
                40.0,
                50.0,
                np.nan,  # row 5: NaN coordinate (latent crash pattern)
                60.0,
                70.0,
                80.0,
                90.0,
            ],
            "start_y": [10.0] * 10,
            "end_x": [20.0, 30.0, 40.0, 50.0, 60.0, np.nan, 70.0, 80.0, 90.0, 100.0],
            "end_y": [10.0] * 10,
        }
    )


@pytest.fixture
def atomic_nan_laced_actions() -> pd.DataFrame:
    """10-row synthetic atomic-SPADL fixture (same NaN positions; atomic schema)."""
    pass_id = atomic_spadlcfg.actiontype_id["pass"]
    keeper_save_id = atomic_spadlcfg.actiontype_id["keeper_save"]
    shot_id = atomic_spadlcfg.actiontype_id["shot"]

    return pd.DataFrame(
        {
            "game_id": [1] * 10,
            "period_id": [1] * 10,
            "action_id": list(range(10)),
            "team_id": [10, 20, 20, 10, 20, 10, 10, 20, 20, 10],
            "player_id": pd.array(
                [np.nan, 201.0, np.nan, 101.0, 202.0, np.nan, 102.0, 201.0, 202.0, np.nan],
                dtype="float64",
            ),
            "type_id": [
                pass_id,
                pass_id,
                keeper_save_id,
                pass_id,
                pass_id,
                pass_id,
                shot_id,
                pass_id,
                pass_id,
                pass_id,
            ],
            "type_name": ["pass"] * 10,
            "bodypart_id": [0] * 10,
            "bodypart_name": ["foot"] * 10,
            "time_seconds": [float(i) for i in range(10)],
            # Atomic uses x/y/dx/dy, no start_/end_ prefix.
            "x": [10.0, 20.0, 30.0, 40.0, 50.0, np.nan, 60.0, 70.0, 80.0, 90.0],
            "y": [10.0] * 10,
            "dx": [10.0] * 10,
            "dy": [0.0] * 10,
        }
    )


# ---------------------------------------------------------------------------
# Auto-discovered fuzz: every decorated helper × NaN-laced fixture.
# Failure mode: any decorated helper that crashes on NaN-laced input fails
# its parametrized case here. Adding a new @nan_safe_enrichment-decorated
# helper auto-extends this test (no test-author work needed).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("helper", STD_ENRICHMENTS, ids=lambda h: h.__name__)
def test_standard_helper_nan_safe(helper, std_nan_laced_actions) -> None:
    """Every @nan_safe_enrichment standard helper survives NaN-laced input
    with default kwargs.
    """
    out = helper(std_nan_laced_actions)
    assert isinstance(out, pd.DataFrame), (
        f"{helper.__name__} returned {type(out).__name__}, expected pd.DataFrame"
    )
    assert len(out) == len(std_nan_laced_actions), (
        f"{helper.__name__} changed row count on NaN-laced input "
        f"({len(std_nan_laced_actions)} -> {len(out)})"
    )


@pytest.mark.parametrize("helper", ATOMIC_ENRICHMENTS, ids=lambda h: h.__name__)
def test_atomic_helper_nan_safe(helper, atomic_nan_laced_actions) -> None:
    """Every @nan_safe_enrichment atomic helper survives NaN-laced input."""
    out = helper(atomic_nan_laced_actions)
    assert isinstance(out, pd.DataFrame), (
        f"{helper.__name__} returned {type(out).__name__}, expected pd.DataFrame"
    )
    assert len(out) == len(atomic_nan_laced_actions), (
        f"{helper.__name__} changed row count on NaN-laced input "
        f"({len(atomic_nan_laced_actions)} -> {len(out)})"
    )


# ---------------------------------------------------------------------------
# Per-helper specific assertions — exact behavior on the bug-triggering rows.
# ---------------------------------------------------------------------------


def test_pre_shot_gk_context_preserves_nan_on_unidentifiable_shot(std_nan_laced_actions) -> None:
    """When the most-recent defending keeper-action has NaN player_id, the
    shot's defending_gk_player_id is NaN — not raises, not 0, not a sentinel.
    """
    out = std_utils.add_pre_shot_gk_context(std_nan_laced_actions)
    # Row 6 is the shot (type_id=shot, team=10); row 2 is the defending team's
    # keeper_save with NaN player_id. The helper must skip the int(NaN) cast.
    shot_row = out[out["action_id"] == 6].iloc[0]
    assert pd.isna(shot_row["defending_gk_player_id"]), (
        f"Expected NaN defending_gk_player_id for shot following NaN-keeper; "
        f"got {shot_row['defending_gk_player_id']!r}"
    )


def test_atomic_pre_shot_gk_context_preserves_nan(atomic_nan_laced_actions) -> None:
    """Atomic counterpart of test_pre_shot_gk_context_preserves_nan."""
    out = atomic_utils.add_pre_shot_gk_context(atomic_nan_laced_actions)
    shot_row = out[out["action_id"] == 6].iloc[0]
    assert pd.isna(shot_row["defending_gk_player_id"])


def test_gk_distribution_metrics_excludes_nan_coords(std_nan_laced_actions) -> None:
    """When a distribution-eligible row has NaN coords, gk_xt_delta is NaN
    for that row (not raises, not arbitrary integer from int(NaN)).
    """
    # Build a 12x8 xt grid for the test (zeros — value doesn't matter; we
    # just need xt_grid to be non-None to exercise the zone-binning path).
    xt_grid = np.zeros((12, 8), dtype=np.float64)
    out = std_utils.add_gk_distribution_metrics(std_nan_laced_actions, xt_grid=xt_grid)
    # Whatever is at row 5 (NaN start_x) — gk_xt_delta there must be NaN
    # (regardless of whether the row is even tagged as distribution).
    nan_coord_row = out.iloc[5]
    assert pd.isna(nan_coord_row["gk_xt_delta"]) or nan_coord_row["gk_xt_delta"] == 0.0, (
        f"Expected NaN/0.0 gk_xt_delta on NaN-coord row; got {nan_coord_row['gk_xt_delta']!r}"
    )
```

- [ ] **Step 4.2: Run T-fuzz — verify RED state**

Run:
```bash
uv run pytest tests/test_enrichment_nan_safety.py -v --tb=short
```
Expected initial state (BEFORE any decoration): both `test_registry_nonempty_*` tests FAIL (registry is empty because no helpers are decorated yet); the parametrized fuzz tests collect zero cases (no parametrize entries = pytest reports as PASSED-with-warning, not failed). The two registry-floor sanity tests are the load-bearing RED indicators.

After Phase 4 (decoration), parametrized cases populate; after Phase 5 (fixes + feature), all turn GREEN.

### Task 5: Create `tests/test_enrichment_provider_e2e.py` (cross-provider e2e)

**Files:**
- Create: `tests/test_enrichment_provider_e2e.py`

- [ ] **Step 5.1: Write the e2e test file**

Create `tests/test_enrichment_provider_e2e.py` with this exact content:

```python
"""Cross-provider e2e regression for the NaN-safety contract (ADR-003).

Runs every @nan_safe_enrichment helper against vendored production-shape
fixtures from each supported provider. Catches: helper crashes on real
production data shape from a provider whose data shape differs from the
ones used during helper development.

Production-shape fixtures used:
- StatsBomb: tests/datasets/statsbomb/spadl-WorldCup-2018.h5 (1 match)
- IDSSE (DFL Sportec via sportec converter): tests/datasets/idsse/sample_match.parquet
- Metrica (via metrica converter): tests/datasets/metrica/sample_match.parquet

For atomic helpers we currently exercise StatsBomb only — atomic conversion
of IDSSE / Metrica fixtures requires verifying the atomic converter pipeline
against those providers, which is out of scope for this PR.
"""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

import pandas as pd
import pytest

import silly_kicks.atomic.spadl.utils as atomic_utils
import silly_kicks.spadl.utils as std_utils

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _discover(module) -> tuple:
    return tuple(
        fn for _, fn in inspect.getmembers(module, inspect.isfunction)
        if getattr(fn, "_nan_safe", False)
    )


STD_ENRICHMENTS = _discover(std_utils)
ATOMIC_ENRICHMENTS = _discover(atomic_utils)


# ---------------------------------------------------------------------------
# Provider fixture loaders — each returns a SPADL DataFrame ready for
# enrichment helpers. Adapted from tests/spadl/test_cross_provider_parity.py
# patterns.
# ---------------------------------------------------------------------------


def _load_statsbomb_one_match() -> pd.DataFrame:
    """First match from the vendored WorldCup-2018 SPADL HDF5 fixture."""
    h5_path = _REPO_ROOT / "tests" / "datasets" / "statsbomb" / "spadl-WorldCup-2018.h5"
    if not h5_path.exists():
        pytest.skip(f"StatsBomb HDF5 fixture not found at {h5_path}")
    games = pd.read_hdf(h5_path, key="games")
    if isinstance(games, pd.Series):
        games = games.to_frame()
    first_game_id = games.iloc[0]["game_id"]
    actions = pd.read_hdf(h5_path, key=f"actions/game_{first_game_id}")
    return actions


def _load_idsse_via_sportec() -> pd.DataFrame:
    """IDSSE bronze events → sportec converter → SPADL."""
    from silly_kicks.spadl import sportec

    parquet_path = _REPO_ROOT / "tests" / "datasets" / "idsse" / "sample_match.parquet"
    if not parquet_path.exists():
        pytest.skip(f"IDSSE fixture not found at {parquet_path}")
    events = pd.read_parquet(parquet_path)
    home_team = events["team"].dropna().iloc[0]
    gk_ids: set[str] | None = None
    if "play_goal_keeper_action" in events.columns:
        gk_ids = set(
            events.loc[events["play_goal_keeper_action"].notna(), "player_id"]
            .dropna()
            .astype(str)
            .tolist()
        )
    actions, _ = sportec.convert_to_actions(
        events, home_team_id=str(home_team), goalkeeper_ids=gk_ids
    )
    return actions


def _load_metrica_via_metrica() -> pd.DataFrame:
    """Metrica events → metrica converter → SPADL."""
    from silly_kicks.spadl import metrica

    parquet_path = _REPO_ROOT / "tests" / "datasets" / "metrica" / "sample_match.parquet"
    if not parquet_path.exists():
        pytest.skip(f"Metrica fixture not found at {parquet_path}")
    events = pd.read_parquet(parquet_path)
    home_team = events["team"].dropna().iloc[0]
    home_passes = events[
        (events["type"] == "PASS") & (events["team"] == home_team) & events["player"].notna()
    ]
    if home_passes.empty:
        pytest.skip("Metrica fixture lacks any PASS-by-home-team event with a known player_id")
    gk_id = str(home_passes["player"].iloc[0])
    actions, _ = metrica.convert_to_actions(
        events, home_team_id=str(home_team), goalkeeper_ids={gk_id}
    )
    return actions


@pytest.fixture(params=["statsbomb", "idsse", "metrica"])
def std_provider_actions(request) -> pd.DataFrame:
    """SPADL DataFrame from one production-shape provider fixture (parametrized)."""
    if request.param == "statsbomb":
        return _load_statsbomb_one_match()
    if request.param == "idsse":
        return _load_idsse_via_sportec()
    return _load_metrica_via_metrica()


# ---------------------------------------------------------------------------
# Auto-discovered cross-provider e2e: every decorated standard helper
# × every provider. Failure mode: helper crashes on real production data
# shape from any provider.
# ---------------------------------------------------------------------------


def test_provider_registry_nonempty() -> None:
    """Bulletproof: at least 5 standard helpers + 5 atomic helpers in registry."""
    assert len(STD_ENRICHMENTS) >= 5, [fn.__name__ for fn in STD_ENRICHMENTS]
    assert len(ATOMIC_ENRICHMENTS) >= 5, [fn.__name__ for fn in ATOMIC_ENRICHMENTS]


@pytest.mark.parametrize("helper", STD_ENRICHMENTS, ids=lambda h: h.__name__)
def test_standard_helper_provider_e2e(helper, std_provider_actions) -> None:
    """Every @nan_safe_enrichment standard helper produces a DataFrame
    on production-shape input from each supported provider.
    """
    out = helper(std_provider_actions)
    assert isinstance(out, pd.DataFrame), (
        f"{helper.__name__} returned {type(out).__name__} on provider data"
    )
    assert len(out) == len(std_provider_actions), (
        f"{helper.__name__} changed row count on provider data "
        f"({len(std_provider_actions)} -> {len(out)})"
    )


@pytest.fixture
def atomic_statsbomb_actions() -> pd.DataFrame:
    """Atomic-SPADL DataFrame from a StatsBomb one-match conversion.

    Loads the standard SPADL HDF5 fixture and runs it through the atomic
    converter to get an atomic-SPADL frame for atomic helper e2e.
    """
    from silly_kicks.atomic.spadl import convert_to_atomic

    std_actions = _load_statsbomb_one_match()
    return convert_to_atomic(std_actions)


@pytest.mark.parametrize("helper", ATOMIC_ENRICHMENTS, ids=lambda h: h.__name__)
def test_atomic_helper_provider_e2e(helper, atomic_statsbomb_actions) -> None:
    """Every @nan_safe_enrichment atomic helper produces a DataFrame on
    StatsBomb-derived atomic-SPADL data.
    """
    out = helper(atomic_statsbomb_actions)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(atomic_statsbomb_actions)
```

- [ ] **Step 5.2: Verify the e2e test collects**

Run:
```bash
uv run pytest tests/test_enrichment_provider_e2e.py --collect-only -q 2>&1 | tail -10
```
Expected (BEFORE Phase 4 decoration): only the registry-nonempty test collects; the parametrized cases collect zero entries (empty registry). After Phase 4, parametrized cases populate.

### Task 6: Create `tests/test_gk_role_goalkeeper_ids.py` (feature tests)

**Files:**
- Create: `tests/test_gk_role_goalkeeper_ids.py`

- [ ] **Step 6.1: Write the feature test file**

Create `tests/test_gk_role_goalkeeper_ids.py` with this exact content:

```python
"""add_gk_role goalkeeper_ids parameter — backward-compat + new behavior.

Covers the 2.5.0 opt-in coverage feature:
- Backward-compat: goalkeeper_ids=None (default) preserves byte-for-byte
  behavior of the pre-2.5.0 add_gk_role.
- Rule (a): goalkeeper_ids provided + clean player_ids → distribution
  detection extends to known-GK rows whose preceding action was keeper.
- Rule (b): goalkeeper_ids provided + NaN player_ids → coarser team-based
  fallback tags non-keeper rows after a keeper action by the same team.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import silly_kicks.atomic.spadl.config as atomic_spadlcfg
import silly_kicks.atomic.spadl.utils as atomic_utils
import silly_kicks.spadl.config as spadlcfg
import silly_kicks.spadl.utils as std_utils


def _make_std_actions(player_ids, type_ids, team_ids, *, n_rows: int) -> pd.DataFrame:
    """Build a minimal valid standard-SPADL frame with explicit per-row inputs."""
    foot_id = spadlcfg.bodypart_id["foot"]
    success_id = spadlcfg.result_id["success"]
    return pd.DataFrame(
        {
            "game_id": [1] * n_rows,
            "period_id": [1] * n_rows,
            "action_id": list(range(n_rows)),
            "team_id": team_ids,
            "player_id": pd.array(player_ids, dtype="float64"),
            "type_id": type_ids,
            "result_id": [success_id] * n_rows,
            "result_name": ["success"] * n_rows,
            "bodypart_id": [foot_id] * n_rows,
            "bodypart_name": ["foot"] * n_rows,
            "type_name": ["pass"] * n_rows,
            "time_seconds": [float(i) for i in range(n_rows)],
            "start_x": [10.0 + i for i in range(n_rows)],
            "start_y": [10.0] * n_rows,
            "end_x": [20.0 + i for i in range(n_rows)],
            "end_y": [10.0] * n_rows,
        }
    )


# ---------------------------------------------------------------------------
# Backward-compat: goalkeeper_ids=None preserves pre-2.5.0 behavior.
# ---------------------------------------------------------------------------


def test_std_default_none_preserves_existing_behavior() -> None:
    """add_gk_role() with no goalkeeper_ids argument == add_gk_role(goalkeeper_ids=None)."""
    pass_id = spadlcfg.actiontype_id["pass"]
    save_id = spadlcfg.actiontype_id["keeper_save"]
    actions = _make_std_actions(
        player_ids=[100.0, 100.0, 200.0, 200.0, 100.0],  # GK 100 saves then passes
        type_ids=[pass_id, save_id, pass_id, pass_id, pass_id],
        team_ids=[10, 10, 20, 20, 10],
        n_rows=5,
    )
    out_default = std_utils.add_gk_role(actions)
    out_explicit_none = std_utils.add_gk_role(actions, goalkeeper_ids=None)
    pd.testing.assert_series_equal(
        out_default["gk_role"].astype(object),
        out_explicit_none["gk_role"].astype(object),
        check_names=False,
    )


def test_atomic_default_none_preserves_existing_behavior() -> None:
    """Atomic counterpart: goalkeeper_ids=None == no goalkeeper_ids passed."""
    pass_id = atomic_spadlcfg.actiontype_id["pass"]
    save_id = atomic_spadlcfg.actiontype_id["keeper_save"]
    actions = pd.DataFrame(
        {
            "game_id": [1] * 5,
            "period_id": [1] * 5,
            "action_id": list(range(5)),
            "team_id": [10, 10, 20, 20, 10],
            "player_id": pd.array([100.0, 100.0, 200.0, 200.0, 100.0], dtype="float64"),
            "type_id": [pass_id, save_id, pass_id, pass_id, pass_id],
            "type_name": ["pass"] * 5,
            "bodypart_id": [0] * 5,
            "bodypart_name": ["foot"] * 5,
            "time_seconds": [float(i) for i in range(5)],
            "x": [10.0 + i for i in range(5)],
            "y": [10.0] * 5,
            "dx": [10.0] * 5,
            "dy": [0.0] * 5,
        }
    )
    out_default = atomic_utils.add_gk_role(actions)
    out_explicit_none = atomic_utils.add_gk_role(actions, goalkeeper_ids=None)
    pd.testing.assert_series_equal(
        out_default["gk_role"].astype(object),
        out_explicit_none["gk_role"].astype(object),
        check_names=False,
    )


# ---------------------------------------------------------------------------
# Rule (a) — known-GK match: clean player_id, GK in goalkeeper_ids set.
# ---------------------------------------------------------------------------


def test_std_rule_a_known_gk_match_extends_distribution() -> None:
    """When current row's player_id ∈ goalkeeper_ids AND prev action was keeper
    same-team, tag as distribution even if same_player matching would have
    succeeded (this is a redundant-but-additive rule on clean data).

    Setup: GK 100 makes a keeper_save (row 1), then GK 100 makes a pass (row 2).
    Both with/without goalkeeper_ids should tag row 2 as distribution
    (same_player rule already catches it). This test asserts adding
    goalkeeper_ids does not REGRESS the existing detection.
    """
    pass_id = spadlcfg.actiontype_id["pass"]
    save_id = spadlcfg.actiontype_id["keeper_save"]
    actions = _make_std_actions(
        player_ids=[200.0, 100.0, 100.0, 200.0],  # team 20 pass, GK 100 save, GK 100 pass, team 20 pass
        type_ids=[pass_id, save_id, pass_id, pass_id],
        team_ids=[20, 10, 10, 20],
        n_rows=4,
    )
    out = std_utils.add_gk_role(actions, goalkeeper_ids={100.0})
    # Row 2: GK pass after GK save → distribution.
    assert out.iloc[2]["gk_role"] == "distribution"


def test_std_rule_a_extends_to_gk_pass_with_no_strict_same_player_match() -> None:
    """Rule (a) is the load-bearing extension: keeper save by GK 100, then a
    pass by some-other-GK (101) on the same team — strict same_player would
    not match (different player_id), but goalkeeper_ids includes both 100 and
    101, AND prev was keeper, AND same team → tag as distribution.

    Use case: GK substitution mid-game; the new GK's first action is a pass
    that should tag as distribution because the team's GK role transferred.
    """
    pass_id = spadlcfg.actiontype_id["pass"]
    save_id = spadlcfg.actiontype_id["keeper_save"]
    actions = _make_std_actions(
        player_ids=[100.0, 101.0, 200.0],  # GK 100 saves, GK 101 (sub) passes, opponent passes
        type_ids=[save_id, pass_id, pass_id],
        team_ids=[10, 10, 20],
        n_rows=3,
    )
    # Without goalkeeper_ids: row 1 NOT tagged as distribution (player_ids differ).
    out_no_gks = std_utils.add_gk_role(actions)
    assert out_no_gks.iloc[1]["gk_role"] is None
    # With goalkeeper_ids: row 1 IS tagged as distribution.
    out_with_gks = std_utils.add_gk_role(actions, goalkeeper_ids={100.0, 101.0})
    assert out_with_gks.iloc[1]["gk_role"] == "distribution"


# ---------------------------------------------------------------------------
# Rule (b) — NaN-team fallback: both NaN player_ids, same team_id.
# ---------------------------------------------------------------------------


def test_std_rule_b_nan_team_fallback_tags_distribution() -> None:
    """When both current and shifted player_ids are NaN AND same team AND
    prev was keeper, tag as distribution (lakehouse coverage gap fix).
    """
    pass_id = spadlcfg.actiontype_id["pass"]
    save_id = spadlcfg.actiontype_id["keeper_save"]
    actions = _make_std_actions(
        player_ids=[np.nan, np.nan, 200.0],  # NaN-keeper, NaN-pass (likely same GK), opponent
        type_ids=[save_id, pass_id, pass_id],
        team_ids=[10, 10, 20],
        n_rows=3,
    )
    # Without goalkeeper_ids: row 1 NOT tagged (NaN==NaN is False).
    out_no_gks = std_utils.add_gk_role(actions)
    assert out_no_gks.iloc[1]["gk_role"] is None
    # With goalkeeper_ids (any non-empty set signals opt-in): row 1 IS tagged.
    out_with_gks = std_utils.add_gk_role(actions, goalkeeper_ids={100.0})
    assert out_with_gks.iloc[1]["gk_role"] == "distribution"


def test_std_rule_b_nan_team_fallback_respects_team_boundary() -> None:
    """Rule (b) requires same team — a NaN-pass by team 20 after NaN-keeper
    by team 10 must NOT be tagged as distribution.
    """
    pass_id = spadlcfg.actiontype_id["pass"]
    save_id = spadlcfg.actiontype_id["keeper_save"]
    actions = _make_std_actions(
        player_ids=[np.nan, np.nan],  # NaN-keeper team 10, NaN-pass team 20
        type_ids=[save_id, pass_id],
        team_ids=[10, 20],
        n_rows=2,
    )
    out = std_utils.add_gk_role(actions, goalkeeper_ids={100.0})
    assert out.iloc[1]["gk_role"] is None  # different team — no fallback


def test_atomic_rule_b_nan_team_fallback_tags_distribution() -> None:
    """Atomic counterpart of test_std_rule_b_nan_team_fallback_tags_distribution."""
    pass_id = atomic_spadlcfg.actiontype_id["pass"]
    save_id = atomic_spadlcfg.actiontype_id["keeper_save"]
    actions = pd.DataFrame(
        {
            "game_id": [1] * 3,
            "period_id": [1] * 3,
            "action_id": [0, 1, 2],
            "team_id": [10, 10, 20],
            "player_id": pd.array([np.nan, np.nan, 200.0], dtype="float64"),
            "type_id": [save_id, pass_id, pass_id],
            "type_name": ["pass"] * 3,
            "bodypart_id": [0] * 3,
            "bodypart_name": ["foot"] * 3,
            "time_seconds": [0.0, 1.0, 2.0],
            "x": [10.0, 20.0, 30.0],
            "y": [10.0, 10.0, 10.0],
            "dx": [10.0, 10.0, 10.0],
            "dy": [0.0, 0.0, 0.0],
        }
    )
    out_no_gks = atomic_utils.add_gk_role(actions)
    assert out_no_gks.iloc[1]["gk_role"] is None
    out_with_gks = atomic_utils.add_gk_role(actions, goalkeeper_ids={100.0})
    assert out_with_gks.iloc[1]["gk_role"] == "distribution"


# ---------------------------------------------------------------------------
# Empty set edge case — passing an empty goalkeeper_ids should be valid.
# ---------------------------------------------------------------------------


def test_std_empty_goalkeeper_ids_set() -> None:
    """Passing an empty set should opt into the NaN-team fallback (rule b)
    but NOT match anyone via rule (a).
    """
    pass_id = spadlcfg.actiontype_id["pass"]
    save_id = spadlcfg.actiontype_id["keeper_save"]
    actions = _make_std_actions(
        player_ids=[np.nan, np.nan],
        type_ids=[save_id, pass_id],
        team_ids=[10, 10],
        n_rows=2,
    )
    # Empty set still triggers the fallback rule (caller signaled opt-in).
    out = std_utils.add_gk_role(actions, goalkeeper_ids=set())
    assert out.iloc[1]["gk_role"] == "distribution"
```

- [ ] **Step 6.2: Verify file collects**

Run:
```bash
uv run pytest tests/test_gk_role_goalkeeper_ids.py --collect-only -q 2>&1 | tail -10
```
Expected: 9 tests collected (5 std + 3 atomic + 1 edge case = 9). All currently fail because `goalkeeper_ids` parameter doesn't exist yet → RED state confirmed.

### Task 7: Phase 3 RED-state verification

**Files:**
- (none — verification only)

- [ ] **Step 7.1: Run all three new test files**

Run:
```bash
uv run pytest tests/test_enrichment_nan_safety.py tests/test_enrichment_provider_e2e.py tests/test_gk_role_goalkeeper_ids.py --tb=short -q 2>&1 | tail -10
```
Expected RED state:
- `test_registry_nonempty_std/atomic` (×3 across the two files) → fail with empty registry
- Parametrized fuzz/e2e tests → 0 cases collected (empty registry)
- `test_gk_role_goalkeeper_ids.py` → 9 tests fail with `TypeError: add_gk_role() got an unexpected keyword argument 'goalkeeper_ids'`

This confirms the tests fail for the expected reasons. Proceed to implementation.

---

## Phase 4 — Audit and decorate the 5 standard + 5 atomic helpers

### Task 8: Decorate `add_gk_role` (std + atomic) — known NaN-safe

**Files:**
- Modify: `silly_kicks/spadl/utils.py` (function at line 47)
- Modify: `silly_kicks/atomic/spadl/utils.py` (function at line 54)

- [ ] **Step 8.1: Decorate standard `add_gk_role`**

In `silly_kicks/spadl/utils.py`, add the import at the top (alongside other absolute first-party imports):

```python
from silly_kicks._nan_safety import nan_safe_enrichment
```

Then add the decorator immediately above the `def add_gk_role(` definition at line 47:

```diff
+@nan_safe_enrichment
 def add_gk_role(
     actions: pd.DataFrame,
     *,
     penalty_area_x_threshold: float = 16.5,
     distribution_lookback_actions: int = 1,
 ) -> pd.DataFrame:
```

(Note: the goalkeeper_ids parameter is added in Task 13, not this task.)

- [ ] **Step 8.2: Decorate atomic `add_gk_role`**

In `silly_kicks/atomic/spadl/utils.py`, add the same import + decorator above the `def add_gk_role(` definition at line 54.

- [ ] **Step 8.3: Smoke-import to verify decoration**

Run:
```bash
uv run python -c "
from silly_kicks.spadl.utils import add_gk_role as std_add_gk_role
from silly_kicks.atomic.spadl.utils import add_gk_role as atomic_add_gk_role
assert getattr(std_add_gk_role, '_nan_safe', False) is True
assert getattr(atomic_add_gk_role, '_nan_safe', False) is True
print('add_gk_role × 2 decorated')
"
```
Expected output: `add_gk_role × 2 decorated`.

### Task 9: Decorate `add_possessions` (std + atomic)

**Files:**
- Modify: `silly_kicks/spadl/utils.py` (find `def add_possessions(`)
- Modify: `silly_kicks/atomic/spadl/utils.py` (function at line 186)

- [ ] **Step 9.1: Locate and decorate standard `add_possessions`**

Find the function via:
```bash
grep -n "^def add_possessions" silly_kicks/spadl/utils.py
```
Add `@nan_safe_enrichment` immediately above the `def add_possessions(...)` line. (Import is already in place from Task 8.)

- [ ] **Step 9.2: Decorate atomic `add_possessions`**

In `silly_kicks/atomic/spadl/utils.py`, add `@nan_safe_enrichment` immediately above the `def add_possessions(` at line 186.

- [ ] **Step 9.3: Verify decoration**

Run:
```bash
uv run python -c "
from silly_kicks.spadl.utils import add_possessions as std
from silly_kicks.atomic.spadl.utils import add_possessions as atomic
assert getattr(std, '_nan_safe', False) is True
assert getattr(atomic, '_nan_safe', False) is True
print('add_possessions × 2 decorated')
"
```
Expected: `add_possessions × 2 decorated`.

### Task 10: Decorate `add_names` (std + atomic) — NaN-safe via merge

**Files:**
- Modify: `silly_kicks/spadl/utils.py` (function at line 1083)
- Modify: `silly_kicks/atomic/spadl/utils.py` (function at line 844)

- [ ] **Step 10.1: Decorate standard `add_names`**

Add `@nan_safe_enrichment` above `def add_names(` at line 1083.

- [ ] **Step 10.2: Decorate atomic `add_names`**

Add `@nan_safe_enrichment` above `def add_names(` at line 844 in atomic utils.

- [ ] **Step 10.3: Verify decoration**

Run:
```bash
uv run python -c "
from silly_kicks.spadl.utils import add_names as std
from silly_kicks.atomic.spadl.utils import add_names as atomic
assert getattr(std, '_nan_safe', False) is True
assert getattr(atomic, '_nan_safe', False) is True
print('add_names × 2 decorated')
"
```

### Task 11: Fix + decorate `add_gk_distribution_metrics` (std + atomic) — latent coord-NaN risk

**Files:**
- Modify: `silly_kicks/spadl/utils.py` (function at line 228; fix at 374-377)
- Modify: `silly_kicks/atomic/spadl/utils.py` (function at line 505; fix at 665-668)

- [ ] **Step 11.1: Apply NaN-coord guard to standard helper**

In `silly_kicks/spadl/utils.py`, find lines 372-377 (the eligible-mask + zone-binning block). Replace the existing block:

```python
        success_id = spadlconfig.result_id["success"]
        result_id_arr = actions["result_id"].to_numpy()
        eligible = is_distribution & (result_id_arr == success_id)
        if eligible.any():
            zone_x_start = np.clip((start_x[eligible] / (_PITCH_LENGTH_M / 12.0)).astype(int), 0, 11)
            zone_y_start = np.clip((start_y[eligible] / (_PITCH_WIDTH_M / 8.0)).astype(int), 0, 7)
            zone_x_end = np.clip((end_x[eligible] / (_PITCH_LENGTH_M / 12.0)).astype(int), 0, 11)
            zone_y_end = np.clip((end_y[eligible] / (_PITCH_WIDTH_M / 8.0)).astype(int), 0, 7)
            xt_delta[eligible] = xt_grid[zone_x_end, zone_y_end] - xt_grid[zone_x_start, zone_y_start]
```

with:

```python
        success_id = spadlconfig.result_id["success"]
        result_id_arr = actions["result_id"].to_numpy()
        # NaN coordinates would crash the .astype(int) zone-binning below.
        # Filter to rows where all four coords are finite (guards against
        # caller data with sparse spatial information). Non-finite rows
        # leave xt_delta at NaN (default initialization).
        coords_finite = (
            np.isfinite(start_x)
            & np.isfinite(start_y)
            & np.isfinite(end_x)
            & np.isfinite(end_y)
        )
        eligible = is_distribution & (result_id_arr == success_id) & coords_finite
        if eligible.any():
            zone_x_start = np.clip((start_x[eligible] / (_PITCH_LENGTH_M / 12.0)).astype(int), 0, 11)
            zone_y_start = np.clip((start_y[eligible] / (_PITCH_WIDTH_M / 8.0)).astype(int), 0, 7)
            zone_x_end = np.clip((end_x[eligible] / (_PITCH_LENGTH_M / 12.0)).astype(int), 0, 11)
            zone_y_end = np.clip((end_y[eligible] / (_PITCH_WIDTH_M / 8.0)).astype(int), 0, 7)
            xt_delta[eligible] = xt_grid[zone_x_end, zone_y_end] - xt_grid[zone_x_start, zone_y_start]
```

- [ ] **Step 11.2: Add the `@nan_safe_enrichment` decorator above standard `add_gk_distribution_metrics`**

```diff
+@nan_safe_enrichment
 def add_gk_distribution_metrics(
     actions: pd.DataFrame,
     *,
     xt_grid: np.ndarray | None = None,
     short_threshold: float = 32.0,
     long_threshold: float = 60.0,
     require_gk_role: bool = True,
 ) -> pd.DataFrame:
```

- [ ] **Step 11.3: Apply the same NaN-coord guard to atomic helper**

In `silly_kicks/atomic/spadl/utils.py`, find the analogous block at lines 663-668. Atomic uses `x` / `y` / `end_x` / `end_y` (no `start_` prefix) instead of `start_x` / `start_y`. Apply the same `coords_finite` filter pattern:

```python
        # NaN coordinates would crash the .astype(int) zone-binning below.
        # Filter to rows where all four coords are finite. Non-finite rows
        # leave xt_delta at NaN (default initialization).
        coords_finite = (
            np.isfinite(x)
            & np.isfinite(y)
            & np.isfinite(end_x)
            & np.isfinite(end_y)
        )
        eligible = is_distribution & (result_id_arr == success_id) & coords_finite
```

(Read the file at lines 660-680 first to confirm the exact local variable names — `x` and `y` instead of `start_x` and `start_y` — before editing.)

- [ ] **Step 11.4: Add `@nan_safe_enrichment` to atomic `add_gk_distribution_metrics`**

Add the decorator above `def add_gk_distribution_metrics(` at line 505 in atomic utils.

- [ ] **Step 11.5: Run targeted test — verify fix takes effect**

Run:
```bash
uv run pytest tests/test_enrichment_nan_safety.py::test_gk_distribution_metrics_excludes_nan_coords -v --tb=short
```
Expected: PASS.

- [ ] **Step 11.6: Run existing distribution-metrics tests — verify no regression**

Run:
```bash
uv run pytest tests/ -k "distribution" --tb=short -q
```
Expected: all existing distribution-metrics tests pass (the fix only changes behavior on NaN-coord rows, which existing tests don't exercise).

### Task 12: Fix + decorate `add_pre_shot_gk_context` (std + atomic) — primary crash

**Files:**
- Modify: `silly_kicks/spadl/utils.py` (function at line 403; fix at 543)
- Modify: `silly_kicks/atomic/spadl/utils.py` (function at line 699; fix at 826)

- [ ] **Step 12.1: Apply NaN-player_id guard to standard helper**

In `silly_kicks/spadl/utils.py`, replace line 543:

```python
        gk_id = int(player_id[window_start + relative_indices[-1]])
```

with:

```python
        gk_id_raw = player_id[window_start + relative_indices[-1]]
        if pd.isna(gk_id_raw):
            # Defending keeper action is identified (in window), but its
            # player_id is NaN — caller's data does not provide enough
            # information to identify the defending GK. Leave defaults
            # (gk_was_engaged stays False, defending_gk_player_id stays NaN).
            continue
        gk_id = int(gk_id_raw)
```

- [ ] **Step 12.2: Add `@nan_safe_enrichment` decorator above standard `add_pre_shot_gk_context`**

```diff
+@nan_safe_enrichment
 def add_pre_shot_gk_context(
     actions: pd.DataFrame,
     *,
     lookback_seconds: float = 10.0,
     lookback_actions: int = 5,
 ) -> pd.DataFrame:
```

- [ ] **Step 12.3: Apply the same NaN guard to atomic helper at line 826**

In `silly_kicks/atomic/spadl/utils.py`, replace line 826 with the same pattern:

```python
        gk_id_raw = player_id[window_start + relative_indices[-1]]
        if pd.isna(gk_id_raw):
            continue
        gk_id = int(gk_id_raw)
```

- [ ] **Step 12.4: Add `@nan_safe_enrichment` to atomic `add_pre_shot_gk_context`**

Add the decorator above `def add_pre_shot_gk_context(` at line 699 in atomic utils.

- [ ] **Step 12.5: Run targeted test — verify primary fix**

Run:
```bash
uv run pytest tests/test_enrichment_nan_safety.py::test_pre_shot_gk_context_preserves_nan_on_unidentifiable_shot tests/test_enrichment_nan_safety.py::test_atomic_pre_shot_gk_context_preserves_nan -v --tb=short
```
Expected: 2 PASS.

- [ ] **Step 12.6: Run smoke test from production-failure shape**

Run:
```bash
uv run python -c "
import pandas as pd
import numpy as np
from silly_kicks.spadl.utils import add_pre_shot_gk_context

actions = pd.DataFrame({
    'game_id': [1, 1, 1, 1],
    'period_id': [1, 1, 1, 1],
    'action_id': [0, 1, 2, 3],
    'team_id': [10, 20, 20, 10],
    'player_id': pd.array([100.0, np.nan, 200.0, 100.0], dtype='float64'),
    'type_id': [0, 14, 0, 13],
    'time_seconds': [0.0, 5.0, 8.0, 9.0],
})
out = add_pre_shot_gk_context(actions)
assert pd.isna(out.iloc[3]['defending_gk_player_id']), 'NaN preserved'
print('production-failure smoke OK')
"
```
Expected: `production-failure smoke OK`.

- [ ] **Step 12.7: Run the parametrized fuzz across all 5 standard + 5 atomic helpers**

Run:
```bash
uv run pytest tests/test_enrichment_nan_safety.py -v --tb=short
```
Expected: all 14 cases PASS (registry nonempty × 2 + standard fuzz × 5 + atomic fuzz × 5 + per-helper assertions × 2).

---

## Phase 5 — Add `goalkeeper_ids` parameter to `add_gk_role`

### Task 13: Implement `goalkeeper_ids` on `add_gk_role` (std + atomic)

**Files:**
- Modify: `silly_kicks/spadl/utils.py` (function at line 47)
- Modify: `silly_kicks/atomic/spadl/utils.py` (function at line 54)

- [ ] **Step 13.1: Update standard `add_gk_role` signature**

In `silly_kicks/spadl/utils.py`, change the function signature from:

```python
def add_gk_role(
    actions: pd.DataFrame,
    *,
    penalty_area_x_threshold: float = 16.5,
    distribution_lookback_actions: int = 1,
) -> pd.DataFrame:
```

to:

```python
def add_gk_role(
    actions: pd.DataFrame,
    *,
    penalty_area_x_threshold: float = 16.5,
    distribution_lookback_actions: int = 1,
    goalkeeper_ids: set | None = None,
) -> pd.DataFrame:
```

- [ ] **Step 13.2: Update standard `add_gk_role` distribution-detection loop**

Find the distribution-detection block (the `for k in range(1, distribution_lookback_actions + 1):` loop). Replace it with:

```python
    prev_keeper_within_k = np.zeros(n, dtype=bool)
    is_keeper_series = pd.Series(is_keeper)
    for k in range(1, distribution_lookback_actions + 1):
        shifted_keeper = is_keeper_series.shift(k, fill_value=False).to_numpy(dtype=bool)
        shifted_player = player_id.shift(k).to_numpy()
        shifted_game = game_id.shift(k).to_numpy()
        cur_player_arr = player_id.to_numpy()
        cur_game_arr = game_id.to_numpy()
        same_player = cur_player_arr == shifted_player
        same_game = cur_game_arr == shifted_game

        match = same_player

        if goalkeeper_ids is not None:
            shifted_team = sorted_actions["team_id"].shift(k).to_numpy()
            cur_team_arr = sorted_actions["team_id"].to_numpy()
            same_team = cur_team_arr == shifted_team

            # Rule (a) — known-GK match: caller declared a GK player_id set.
            cur_is_known_gk = pd.Series(cur_player_arr).isin(goalkeeper_ids).to_numpy()
            match = match | (cur_is_known_gk & same_team)

            # Rule (b) — NaN-team fallback: both player_ids unidentifiable
            # but same team + prev was keeper. Coarse heuristic; caller
            # opts in by passing the goalkeeper_ids parameter (which signals
            # willingness to accept the over-counting risk on dense NaN data).
            cur_player_na = pd.isna(cur_player_arr)
            shifted_player_na = pd.isna(shifted_player)
            match = match | (cur_player_na & shifted_player_na & same_team)

        prev_keeper_within_k |= shifted_keeper & match & same_game
```

(The variable `team_id` is referenced in the new block — confirm by re-reading the function: in the existing code, `team_id` may be a numpy array or a pandas Series local. The block above uses `sorted_actions["team_id"]` to be unambiguous; adapt if the existing local variable is already a pandas Series with `.shift` available — in that case prefer the local name for consistency.)

- [ ] **Step 13.3: Update standard `add_gk_role` docstring**

Locate the Parameters section in the docstring. Add the new `goalkeeper_ids` parameter docs immediately after `distribution_lookback_actions`:

```rst
    goalkeeper_ids : set, optional
        When provided, distribution-detection extends beyond strict
        ``same_player`` matching to also tag rows where:

        - The current row's ``player_id`` is in ``goalkeeper_ids`` AND the
          preceding action (within ``distribution_lookback_actions`` steps,
          same ``team_id`` and ``game_id``) was a keeper-type action.
          (Use case: caller knows the GK player_ids; clean-attribution data.)
        - Both the current row's and the preceding action's ``player_id``
          are NaN AND the ``team_id`` matches AND the preceding action was
          keeper-type. (Use case: caller's data has NaN player_id but the
          team/sequence implies the GK distributed the ball.)

        Opting in via this parameter signals that the caller accepts the
        coarser heuristic (the second rule may over-tag if multiple
        NaN-player_id non-keeper actions follow a keeper action by the
        same team within the lookback window).

        When ``None`` (default), only strict ``same_player`` matching applies
        — byte-for-byte compatible with pre-2.5.0 behavior.
```

- [ ] **Step 13.4: Apply the same changes to atomic `add_gk_role`**

In `silly_kicks/atomic/spadl/utils.py`, apply the same three changes (signature, loop body, docstring) to `add_gk_role` at line 54. The atomic version should use atomic-SPADL's column references (no schema column-name differences for `team_id`, `player_id`, `game_id`).

- [ ] **Step 13.5: Run goalkeeper_ids feature tests — verify GREEN**

Run:
```bash
uv run pytest tests/test_gk_role_goalkeeper_ids.py -v --tb=short
```
Expected: 9 PASS.

- [ ] **Step 13.6: Run existing add_gk_role tests — verify no regression**

Run:
```bash
uv run pytest tests/ -k "gk_role" --tb=short -q
```
Expected: all existing add_gk_role tests pass (default `goalkeeper_ids=None` preserves byte-for-byte behavior).

---

## Phase 6 — Bonus defensive fix (coverage_metrics)

### Task 14: Defensive NaN guard in `coverage_metrics` (std + atomic)

**Files:**
- Modify: `silly_kicks/spadl/utils.py` (line 1074)
- Modify: `silly_kicks/atomic/spadl/utils.py` (line 1036)

NOT covered by ADR-003 (returns TypedDict, not DataFrame), but same `int(NaN)` crash class. Fix while we're here per "nothing deferred" principle.

- [ ] **Step 14.1: Apply NaN guard to standard `coverage_metrics`**

In `silly_kicks/spadl/utils.py`, replace line 1074 from:

```python
        for tid in type_id_arr:
            name = id_to_name.get(int(tid), "unknown")
            counts[name] = counts.get(name, 0) + 1
```

to:

```python
        for tid in type_id_arr:
            if pd.isna(tid):
                name = "unknown"
            else:
                name = id_to_name.get(int(tid), "unknown")
            counts[name] = counts.get(name, 0) + 1
```

- [ ] **Step 14.2: Apply the same fix to atomic `coverage_metrics`**

In `silly_kicks/atomic/spadl/utils.py`, replace the analogous block at line 1036 with the same `if pd.isna(tid): ... else: ...` pattern.

- [ ] **Step 14.3: Run existing coverage_metrics tests — verify no regression**

Run:
```bash
uv run pytest tests/ -k "coverage_metrics" --tb=short -q
```
Expected: all existing tests pass.

---

## Phase 7 — Verification gates

### Task 15: Run all verification gates

**Files:**
- (none — verification only)

- [ ] **Step 15.1: ruff check**

Run:
```bash
uv run ruff check silly_kicks/ tests/ 2>&1 | tail -10
```
Expected: `All checks passed!` (or empty stdout, exit 0). If failures: fix in the reported file, re-run.

- [ ] **Step 15.2: ruff format check**

Run:
```bash
uv run ruff format --check silly_kicks/ tests/ 2>&1 | tail -5
```
Expected: `XX files already formatted`, exit 0. If failures: run `uv run ruff format silly_kicks/ tests/` to apply.

- [ ] **Step 15.3: pyright**

Run (background, takes ~30-60s):
```bash
uv run pyright silly_kicks/
```
Expected: `0 errors, 0 warnings, 0 informations`. The new `goalkeeper_ids: set | None = None` parameter and `cur_is_known_gk` etc. should pyright-clean.

- [ ] **Step 15.4: Smoke test — full discovery + cross-package imports**

Run:
```bash
uv run python -c "
import inspect
from silly_kicks._nan_safety import nan_safe_enrichment
import silly_kicks.spadl.utils as su
import silly_kicks.atomic.spadl.utils as au
discovered_std = sorted(n for n, fn in inspect.getmembers(su, inspect.isfunction) if getattr(fn, '_nan_safe', False))
discovered_atomic = sorted(n for n, fn in inspect.getmembers(au, inspect.isfunction) if getattr(fn, '_nan_safe', False))
expected = ['add_gk_distribution_metrics', 'add_gk_role', 'add_names', 'add_possessions', 'add_pre_shot_gk_context']
assert discovered_std == expected, discovered_std
assert discovered_atomic == expected, discovered_atomic
print('discovery OK:', discovered_std)
"
```
Expected output: `discovery OK: ['add_gk_distribution_metrics', 'add_gk_role', 'add_names', 'add_possessions', 'add_pre_shot_gk_context']`.

- [ ] **Step 15.5: Smoke — production-failure pattern**

Run:
```bash
uv run python -c "
import pandas as pd
import numpy as np
from silly_kicks.spadl.utils import add_pre_shot_gk_context, add_gk_distribution_metrics, add_gk_role

# IDSSE failure shape — NaN in keeper action's player_id.
df = pd.DataFrame({
    'game_id': [1]*4, 'period_id': [1]*4, 'action_id': list(range(4)),
    'team_id': [10, 20, 20, 10],
    'player_id': pd.array([100.0, np.nan, 200.0, 100.0], dtype='float64'),
    'type_id': [0, 14, 0, 13],
    'result_id': [1]*4, 'result_name': ['success']*4,
    'bodypart_id': [0]*4, 'bodypart_name': ['foot']*4, 'type_name': ['pass']*4,
    'time_seconds': [0.0, 5.0, 8.0, 9.0],
    'start_x': [10.0, 20.0, 30.0, 40.0], 'start_y': [10.0]*4,
    'end_x': [20.0, 30.0, 40.0, 50.0], 'end_y': [10.0]*4,
})
# All three helpers should not crash on this input.
add_pre_shot_gk_context(df)
add_gk_role(df)
add_gk_role(df, goalkeeper_ids={100.0})
print('NaN-safe smokes OK')
"
```
Expected: `NaN-safe smokes OK`.

- [ ] **Step 15.6: Full pytest suite (background — likely 35-50s)**

Run in background:
```bash
uv run pytest tests/ -m "not e2e" --tb=short -q
```
Expected: ends with `~928 passed, 4 deselected` (884 baseline + ~44 net delta). Exact count depends on test parametrization; the +/-2 variance is acceptable.

If significantly off:
- Fewer than expected: investigate which test isn't running (check fixture skips).
- More than expected: a parametrize cardinality assumption was off; recount.
- Any failure: fix root cause, do not skip the test.

---

## Phase 8 — Documentation

### Task 16: Write ADR-003

**Files:**
- Create: `docs/superpowers/adrs/ADR-003-nan-safety-enrichment-helpers.md`

- [ ] **Step 16.1: Write the ADR**

Create the file with this exact content:

```markdown
# ADR-003: NaN-safety contract for enrichment helpers

| Field | Value |
|---|---|
| **Date** | 2026-04-30 |
| **Status** | Accepted |
| **Deciders** | Karsten S. Nielsen, Claude Opus 4.7 (1M) |

## Context

silly-kicks's post-conversion enrichment helpers (`add_possessions`,
`add_names`, `add_gk_role`, `add_gk_distribution_metrics`,
`add_pre_shot_gk_context`, plus atomic counterparts — 10 functions total)
operate on caller-supplied SPADL DataFrames. ADR-001 (silly-kicks 2.0.0)
commits silly-kicks to caller's identifier conventions: converters never
override the caller's `team_id` / `player_id`. That sacred-pass-through
implies enrichment helpers must be robust to NaN identifiers — caller
data shapes vary by provider, and StatsBomb-style dense attribution is
not the universal case.

On 2026-04-30, the luxury-lakehouse daily ingestion job's first
end-to-end run on real IDSSE bronze data hit
`ValueError: cannot convert float NaN to integer` at
`silly_kicks/spadl/utils.py:543` in `add_pre_shot_gk_context`:

```python
gk_id = int(player_id[window_start + relative_indices[-1]])
```

When the most-recent defending-keeper-action's row has NaN `player_id`,
`int(NaN)` raises. A symmetric latent crash exists in
`add_gk_distribution_metrics` at `silly_kicks/spadl/utils.py:374-377`
(zone-binning `.astype(int)` on possibly-NaN coordinates). The atomic
package mirrors both bugs at `silly_kicks/atomic/spadl/utils.py:826`
and `665-668`.

The deeper problem: silly-kicks did not have an explicit, codified,
CI-enforced NaN-safety contract for enrichment helpers. Each helper
made ad-hoc decisions about NaN handling; some were accidentally
NaN-safe (`add_gk_role` uses `==` comparison, which is NaN-safe),
some were accidentally NaN-unsafe. Without a contract + a
self-enforcing perimeter, the next helper a contributor adds is just
as likely to repeat the bug.

## Decision

silly-kicks codifies a **NaN-safety contract for public enrichment
helpers**. A NaN-safe enrichment helper is a public function
`add_*(actions: pd.DataFrame, ...) -> pd.DataFrame` that satisfies all of:

1. **No crash on NaN identifiers.** For every column in the input that
   is a caller-supplied identifier (`player_id`, `team_id`, `game_id`,
   `period_id`, `action_id`), NaN values do not raise. Internal logic
   that relies on the value detects NaN and routes to the per-row
   default.
2. **No crash on NaN numerics.** For every numeric column the helper
   internally casts to integer (e.g. coordinate columns flowing into
   zone-binning), NaN values do not raise. Affected rows are excluded
   from the cast-dependent computation; their output column receives
   NaN/default.
3. **NaN preservation in identifier outputs.** When the helper outputs
   an identifier-derived column (e.g. `defending_gk_player_id`), NaN
   inputs that prevent identification produce NaN outputs at those rows.
4. **Documented NaN-input semantics.** The helper's docstring contains
   an explicit sentence describing what happens when an input row has
   NaN in an identifier column.
5. **`@nan_safe_enrichment` decorator applied.** The decorator from
   `silly_kicks._nan_safety` sets `fn._nan_safe = True`; CI gates
   auto-discover decorated helpers via this attribute.

Enforcement is two-pronged:

- **Auto-discovered fuzz** (`tests/test_enrichment_nan_safety.py`)
  parametrizes over every decorated helper × a synthetic NaN-laced
  fixture; asserts no crash + sensible defaults.
- **Cross-provider e2e** (`tests/test_enrichment_provider_e2e.py`)
  parametrizes over every decorated helper × vendored production-shape
  fixtures from each supported provider (StatsBomb / IDSSE / Metrica);
  asserts no crash on real production data shapes.

Both gates include **registry-floor sanity assertions** that fail if the
auto-discovery silently regresses (e.g. marker name typo, decoration
removed). The fence has bulletproof posts.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Documentation-only — write a CONVENTIONS doc; rely on PR review | minimal change | no automated enforcement; future contributors miss the convention | rejected: lakehouse failure today shows convention-without-enforcement is insufficient |
| B. AST-based gate forbidding `int(...)` / `.astype(int)` patterns on caller-DataFrame values | catches the pattern at PR time | high false-positive rate (legitimate uses); hard to specify the "caller-data" predicate precisely; complex to maintain | rejected: auto-discovered fuzz catches the same class with much simpler implementation |
| C. Full nullable-Int64 dtype migration (player_id / team_id → pandas Int64) | type-level NaN-safety enforced by pyright | massive blast radius — every converter, every helper, every downstream consumer (lakehouse) needs to handle nullable Int64; multi-week migration | rejected for now: future direction; this PR addresses immediate pain |
| D. Per-helper hand-written NaN tests | explicit, easy to understand | requires test author to remember on every new helper; future helpers without NaN tests slip through | rejected: auto-discovered fuzz is the same effort with self-enforcing perimeter |
| E (chosen). Contract + decorator + auto-discovered fuzz + cross-provider e2e + registry-floor sanity | self-enforcing perimeter; decoration is explicit opt-in; future helpers auto-covered when decorated | requires opt-in via decoration; helpers that forget the decorator aren't tested by ADR-003 (mitigated by code review against CLAUDE.md rule + registry-floor sanity catching mass-decoration breakage) | — |

## Consequences

### Positive

- The 2 confirmed crash sites (`add_pre_shot_gk_context` × 2) and
  2 latent crash sites (`add_gk_distribution_metrics` × 2) are fixed.
- The contract is explicit, codified, and CI-enforced.
- New enrichment helpers added in future PRs auto-extend the fuzz
  coverage when decorated. ADR-003 + the decorator together create a
  self-enforcing perimeter: contributors must explicitly opt in (which
  forces them to think about NaN-safety) but the test infrastructure
  is automatic.
- The cross-provider e2e gate catches the "first time helper X meets
  provider Y data" bug class — exactly the class that surfaced today.
- Every existing call site (`goalkeeper_ids=None` on `add_gk_role`,
  unchanged signatures elsewhere) preserves byte-for-byte behavior.

### Negative

- Decoration is opt-in. A new helper that's NaN-unsafe AND undecorated
  AND has no dedicated NaN test would satisfy CI but fail ADR-003 in
  spirit. Mitigation: CLAUDE.md "Key conventions" amendment makes
  decoration a project-wide rule reviewers check during PR review.
- Adding `goalkeeper_ids: set | None = None` parameter to `add_gk_role`
  expands the function's `inspect.signature(add_gk_role)` surface.
  Consumers introspecting the signature (rare) would see the new
  keyword-only param. Documented in CHANGELOG.
- Rule (b) NaN-team fallback in `add_gk_role` may over-count distribution
  rows on data with multiple NaN-player_id non-keeper actions following
  a keeper action by the same team. Caller's opt-in via `goalkeeper_ids`
  is the explicit signal that they accept the coarser heuristic.

### Neutral

- `gamestates.__module__` (and `simple.__module__`) flips noted in
  ADR-002 (silly-kicks 2.4.0) is a similar Hyrum's-Law class of change;
  ADR-003's `_nan_safe` attribute on helpers is a parallel additive
  surface.
- ADR-003 follows the ADR-001/ADR-002 precedent. The vendored
  `ADR-TEMPLATE.md` is the canonical structural source.

## CLAUDE.md Amendment

Adds one rule to the project's `CLAUDE.md` "Key conventions" section:

> Public enrichment helpers (post-conversion `add_*` family) tolerate NaN
> in caller-supplied identifier columns. NaN identifiers route to the
> documented per-row default (typically NaN-output / False / 0); helpers
> never crash on NaN input. Decoration with `@nan_safe_enrichment` from
> `silly_kicks._nan_safety` is the formal opt-in. Decision: ADR-003.

Scope: every public `add_*` enrichment helper in `silly_kicks.spadl.utils`,
`silly_kicks.atomic.spadl.utils`, and any future post-conversion enrichment
module.

## Related

- **Spec:** `docs/superpowers/specs/2026-04-30-nan-safety-enrichment-helpers-design.md`
- **Plan:** `docs/superpowers/plans/2026-04-30-nan-safety-enrichment-helpers.md`
- **ADRs:** ADR-001 (caller-pass-through identifier convention — implies the NaN-safety requirement at the helper layer).
- **Issues / PRs:** silly-kicks PR-S17 (this PR — primary failure surfaced 2026-04-30 by luxury-lakehouse `compute_spadl_vaep` task on IDSSE bronze data; lakehouse memo references PR-LL3-era observation now superseded).

## Notes

### `_nan_safe` attribute name and discoverability

The marker `fn._nan_safe = True` is set as a function attribute by the
`@nan_safe_enrichment` decorator. Tests discover via
`inspect.getmembers(module, inspect.isfunction)` filtered on
`getattr(fn, "_nan_safe", False)`. Trade-off vs. an explicit registry
list:

- **Attribute-based** (chosen): no central list to maintain; marker
  travels with the function across refactors / module moves; failure
  mode is "helper not in registry" which is benign (helper just not
  fuzz-tested) and caught by registry-floor sanity if it happens
  en-masse.
- **Central registry list**: explicit and auditable, but creates a
  second maintenance point and an import-order subtlety (helpers must
  be imported for the registry to populate).

The attribute approach matches the lakehouse's own ADR-008 marker-on-
function pattern.

### Future direction

Long-term, migrating `player_id` / `team_id` columns to pandas nullable
`Int64` dtype throughout SPADL would push NaN-safety to the type
system. That's a multi-week migration with major Hyrum's Law surface;
ADR-003 explicitly defers it. When the migration happens, the
`@nan_safe_enrichment` decorator may be retired (or its semantic
shifted to "helper supports nullable-Int64 identifier columns").
```

### Task 17: CLAUDE.md amendment

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 17.1: Read existing "Key conventions" section**

Run:
```bash
grep -n "## Key conventions\|^- " CLAUDE.md | head -20
```
Note the structure of existing bullets (most are one paragraph each).

- [ ] **Step 17.2: Add the new bullet**

In `CLAUDE.md`, locate the "## Key conventions" section. Add this bullet at the end of the section (preserving the existing bullets verbatim):

```markdown
- **Public enrichment helpers (post-conversion `add_*` family) tolerate NaN in caller-supplied identifier columns.** NaN identifiers route to the documented per-row default (typically NaN-output / False / 0); helpers never crash on NaN input. Decoration with `@nan_safe_enrichment` from `silly_kicks._nan_safety` is the formal opt-in. Decision: ADR-003.
```

- [ ] **Step 17.3: Verify CLAUDE.md still readable**

Run:
```bash
uv run python -c "
from pathlib import Path
text = Path('CLAUDE.md').read_text(encoding='utf-8')
assert 'ADR-003' in text
assert '@nan_safe_enrichment' in text
print('CLAUDE.md OK')
"
```
Expected: `CLAUDE.md OK`.

### Task 18: Update CHANGELOG `[2.5.0]`

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 18.1: Insert the 2.5.0 entry**

In `CHANGELOG.md`, insert this block immediately above the existing `## [2.4.0]` heading:

```markdown
## [2.5.0] — 2026-04-30

### Added

- **`silly_kicks._nan_safety.nan_safe_enrichment`** — marker decorator
  declaring an enrichment helper satisfies the NaN-safety contract
  (ADR-003). Sets `fn._nan_safe = True`; CI gates auto-discover decorated
  helpers via this attribute.
- **`goalkeeper_ids: set | None = None`** keyword-only parameter on
  `silly_kicks.spadl.utils.add_gk_role` and
  `silly_kicks.atomic.spadl.utils.add_gk_role`. When provided,
  distribution-detection extends with two additional matching rules:
  (a) `current player_id ∈ goalkeeper_ids` AND prev keeper-type same-team;
  (b) NaN-team fallback — both player_ids NaN AND same team_id AND prev
  keeper-type. Closes the lakehouse coverage gap on IDSSE/Metrica data
  with sparse player attribution. When `None` (default), behavior is
  byte-for-byte unchanged.
- **`tests/test_enrichment_nan_safety.py`** — auto-discovered NaN-fuzz
  test (~14 cases). Parametrizes over every `@nan_safe_enrichment`
  helper × synthetic NaN-laced SPADL fixture; asserts no crash +
  sensible defaults. Includes registry-floor sanity assertions that
  catch silent discovery breakage.
- **`tests/test_enrichment_provider_e2e.py`** — auto-discovered
  cross-provider e2e regression (~20 cases). Parametrizes over every
  `@nan_safe_enrichment` standard helper × vendored fixtures from
  StatsBomb / IDSSE / Metrica; atomic helpers run on the
  StatsBomb-derived atomic-SPADL fixture.
- **`tests/test_gk_role_goalkeeper_ids.py`** — feature tests for the new
  `goalkeeper_ids` parameter (~9 cases): backward-compat, rule (a)
  known-GK match, rule (b) NaN-team fallback, edge cases (atomic, empty
  set, team-boundary respect).
- **`docs/superpowers/adrs/ADR-003-nan-safety-enrichment-helpers.md`** —
  formalizes the NaN-safety contract for public enrichment helpers,
  alternatives considered, and the registry-floor sanity assertion as
  the bulletproof for the auto-discovery mechanism.
- **CLAUDE.md "Key conventions" amendment** pointing to ADR-003.

### Fixed

- **`silly_kicks.spadl.utils.add_pre_shot_gk_context`** —
  `ValueError: cannot convert float NaN to integer` at line 543 when
  the most-recent defending-keeper-action's `player_id` is NaN
  (e.g. IDSSE bronze data with sparse player attribution). Surfaced
  2026-04-30 by the luxury-lakehouse `compute_spadl_vaep` task. Fix:
  detect NaN before the `int(...)` cast; `continue` to next shot
  (defending_gk_player_id stays NaN per the function's documented
  contract). Symmetric fix at
  `silly_kicks.atomic.spadl.utils.add_pre_shot_gk_context` line 826.
- **`silly_kicks.spadl.utils.add_gk_distribution_metrics`** — latent
  `ValueError: cannot convert float NaN to integer` at lines 374-377
  on `.astype(int)` zone-binning when a distribution-eligible row has
  NaN coordinates. Fix: filter `eligible` mask by `np.isfinite(...)`
  on all four coords. Symmetric fix at
  `silly_kicks.atomic.spadl.utils.add_gk_distribution_metrics`
  lines 665-668.
- **`silly_kicks.spadl.utils.coverage_metrics`** (defensive) — same
  `int(NaN)` crash class on `int(tid)` at line 1074 if input has NaN
  `type_id`. Fix: NaN guard before the cast; NaN type_ids tally as
  "unknown". Symmetric fix at
  `silly_kicks.atomic.spadl.utils.coverage_metrics` line 1036. Not
  under ADR-003 (TypedDict-returning, not enrichment helper) — fixed
  while we're here.

### Changed

- 10 public enrichment helpers (5 standard + 5 atomic) decorated with
  `@nan_safe_enrichment`: `add_possessions`, `add_names`, `add_gk_role`,
  `add_gk_distribution_metrics`, `add_pre_shot_gk_context` × 2 packages.
  Each helper's docstring gains an explicit "NaN-input semantics"
  sentence per ADR-003.

### Notes

- **Hyrum's Law surface:** `add_gk_role.__signature__` gains the new
  `goalkeeper_ids` keyword-only parameter. Consumers using
  `inspect.signature(add_gk_role)` would see the addition. Documented
  in ADR-003 as accepted exposure.
- **Test count:** 884 → ~928 passing, 4 deselected (+~44 net delta:
  ~14 fuzz + ~20 e2e + ~10 goalkeeper_ids + sanity). Pyright clean
  (0 errors / 0 warnings / 0 informations).
- Future direction: nullable-Int64 dtype migration for `player_id` /
  `team_id` columns is the long-term answer to type-level NaN-safety;
  out of scope for this PR (ADR-003 § Notes / Future direction).
```

### Task 19: Update per-helper docstrings

**Files:**
- Modify: `silly_kicks/spadl/utils.py` (5 docstrings)
- Modify: `silly_kicks/atomic/spadl/utils.py` (5 docstrings)

- [ ] **Step 19.1: Standard helpers — add NaN-input semantics sentence**

For each of the 5 standard enrichment helpers, locate the docstring and add this sentence at the end of the main description (before the Parameters section):

> NaN values in caller-supplied identifier columns (e.g. ``player_id``)
> are treated as "not identifiable" for that row's enrichment lookup;
> downstream rows behave as if no identifier were present. See ADR-003.

Apply to:
- `add_possessions` (find via `grep -n "^def add_possessions" silly_kicks/spadl/utils.py`)
- `add_names` (line 1083)
- `add_gk_role` (line 47) — also re-check the goalkeeper_ids docstring from Task 13
- `add_gk_distribution_metrics` (line 228) — additionally note: "Rows with NaN coordinates are excluded from xT-delta zone-binning; their `gk_xt_delta` is NaN."
- `add_pre_shot_gk_context` (line 403) — already documents NaN behavior implicitly; add the explicit sentence anyway for consistency.

- [ ] **Step 19.2: Atomic helpers — same sentence**

Apply the same sentence to each of the 5 atomic enrichment helper docstrings in `silly_kicks/atomic/spadl/utils.py`:
- `add_gk_role` (line 54)
- `add_possessions` (line 186)
- `add_gk_distribution_metrics` (line 505)
- `add_pre_shot_gk_context` (line 699)
- `add_names` (line 844)

- [ ] **Step 19.3: Verify Examples gate still passes**

Run:
```bash
uv run pytest tests/test_public_api_examples.py -q --tb=short
```
Expected: 27 passed (the gate count is unchanged — we modified existing helpers' docstrings, didn't add new public modules).

---

## Phase 9 — Version + memory

### Task 20: Bump version

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 20.1: Apply the version bump**

Edit `pyproject.toml`:

```diff
-version = "2.4.0"
+version = "2.5.0"
```

- [ ] **Step 20.2: Verify**

Run:
```bash
grep '^version' pyproject.toml
```
Expected: `version = "2.5.0"`.

### Task 21: Update memory files

**Files:**
- Modify: `C:\Users\Karsten\.claude\projects\D--Development-karstenskyt--silly-kicks\memory\project_release_state.md`
- Modify: `C:\Users\Karsten\.claude\projects\D--Development-karstenskyt--silly-kicks\memory\project_followup_prs.md`
- Modify: `C:\Users\Karsten\.claude\projects\D--Development-karstenskyt--silly-kicks\memory\MEMORY.md`

- [ ] **Step 21.1: Update `project_release_state.md`**

Bump current version line to 2.5.0; add a 2.5.0 trajectory entry summarizing PR-S17 (NaN-safety contract + decorator + 4 fixes + goalkeeper_ids feature + 2 new test files + ADR-003 + CLAUDE.md amendment).

Suggested entry text (single row in the trajectory table):

```markdown
| 2.5.0 | 2026-04-30 | **PR-S17:** NaN-safety contract for enrichment helpers (ADR-003) — `@nan_safe_enrichment` decorator + auto-discovered CI gate (registry-floor sanity bulletproof) + cross-provider e2e on StatsBomb/IDSSE/Metrica fixtures. **Fixed:** `add_pre_shot_gk_context` ValueError on NaN keeper-action player_id (IDSSE production crash, surfaced by lakehouse `compute_spadl_vaep` 2026-04-30); `add_gk_distribution_metrics` latent crash on NaN coords; `coverage_metrics` defensive int(NaN) guard. **Added:** `goalkeeper_ids: set \| None = None` opt-in parameter on `add_gk_role` (std + atomic) with two extension rules — known-GK match (rule a) + NaN-team fallback (rule b) — closes lakehouse coverage gap on IDSSE/Metrica. 10 public enrichment helpers decorated. CLAUDE.md "Key conventions" gains the rule. Test count 884 → ~928 (+~44 net). Pyright clean. |
```

Also bump the "Current" line: `## Current: silly-kicks 2.5.0 (tag pushed 2026-04-30)`.

- [ ] **Step 21.2: Update `project_followup_prs.md`**

Update header to PR-S9..S17. Add a brief PR-S17 SHIPPED note. Note that "TODO.md is empty" still holds (this PR was lakehouse-driven, not from queued TODOs).

- [ ] **Step 21.3: Update `MEMORY.md` index**

Update the [Release state] one-liner to reflect 2.5.0 + PR-S17. Update the [Follow-up PRs] one-liner. Update the [Public-API Examples discipline] one-liner if Examples-gate file count changed (it didn't this PR — still 27).

Optionally add a new memory file pointer if useful — e.g. a new `feedback_nan_safety_contract.md` capturing the lesson "auto-discovered marker-based contract enforcement > hand-maintained test lists" so future similar contracts can re-use the pattern. Keep it short (one paragraph + how-to-apply).

---

## Phase 10 — Final review + commit

### Task 22: /final-review

**Files:**
- (none — review pass)

- [ ] **Step 22.1: Run /final-review**

Invoke the `mad-scientist-skills:final-review` skill via the Skill tool. Address any findings inline. The architecture diagram in `docs/c4/architecture.html` describes container-level structure (`silly_kicks.spadl`, `silly_kicks.vaep`, `silly_kicks.atomic`); ADR-003 is a contract / convention, not a container-level architectural change. The C4 diagram is structurally unchanged. No regeneration needed.

- [ ] **Step 22.2: Final verification gate sequence**

Run:
```bash
uv run ruff check silly_kicks/ tests/ && \
uv run ruff format --check silly_kicks/ tests/ && \
uv run pyright silly_kicks/ && \
uv run pytest tests/ -m "not e2e" --tb=short -q
```
Expected: all green; pytest ~928 passing.

### Task 23: Single-commit gate (USER APPROVAL REQUIRED)

**Files:**
- (none — git operation)

- [ ] **Step 23.1: Show user the diff scope**

Run:
```bash
git status --short
git diff --stat
```

Expected file list (**12 staged for commit**):

**New (7):**
- `silly_kicks/_nan_safety.py`
- `tests/test_enrichment_nan_safety.py`
- `tests/test_enrichment_provider_e2e.py`
- `tests/test_gk_role_goalkeeper_ids.py`
- `docs/superpowers/specs/2026-04-30-nan-safety-enrichment-helpers-design.md`
- `docs/superpowers/plans/2026-04-30-nan-safety-enrichment-helpers.md`
- `docs/superpowers/adrs/ADR-003-nan-safety-enrichment-helpers.md`

**Modified (5):**
- `silly_kicks/spadl/utils.py` (5 helpers decorated + 2 fixes + coverage_metrics defensive + 5 docstring updates)
- `silly_kicks/atomic/spadl/utils.py` (mirror — 5 decorated + 2 fixes + coverage_metrics + 5 docstrings + goalkeeper_ids feature)
- `CLAUDE.md`
- `CHANGELOG.md`
- `pyproject.toml`

(Memory files outside the repo — track via memory tooling, not git.)

Wait for explicit user approval before continuing.

- [ ] **Step 23.2: Stage the files and commit**

Once approved:
```bash
git add silly_kicks/_nan_safety.py
git add tests/test_enrichment_nan_safety.py
git add tests/test_enrichment_provider_e2e.py
git add tests/test_gk_role_goalkeeper_ids.py
git add docs/superpowers/specs/2026-04-30-nan-safety-enrichment-helpers-design.md
git add docs/superpowers/plans/2026-04-30-nan-safety-enrichment-helpers.md
git add docs/superpowers/adrs/ADR-003-nan-safety-enrichment-helpers.md
git add silly_kicks/spadl/utils.py
git add silly_kicks/atomic/spadl/utils.py
git add CLAUDE.md
git add CHANGELOG.md
git add pyproject.toml
```

Then commit:
```bash
git commit -m "$(cat <<'EOF'
feat(spadl)!: NaN-safety contract for enrichment helpers + goalkeeper_ids feature — silly-kicks 2.5.0 (PR-S17)

Codifies a NaN-safety contract for silly-kicks's 10 public enrichment
helpers (5 standard + 5 atomic) via ADR-003, enforced by an
auto-discovered CI gate. Fixes 2 confirmed crash sites + 2 latent
crash sites. Adds opt-in goalkeeper_ids parameter on add_gk_role
closing the lakehouse coverage gap on IDSSE/Metrica.

Triggered by lakehouse compute_spadl_vaep task crashing on first
end-to-end run on real IDSSE bronze data (2026-04-30):
ValueError: cannot convert float NaN to integer in
add_pre_shot_gk_context when the most-recent defending-keeper-action's
player_id is NaN. ADR-001 (caller-pass-through identifier convention)
implies enrichment helpers must tolerate NaN identifiers; this PR
makes that contract explicit and CI-enforced.

Architecture:
- New private module silly_kicks/_nan_safety.py with
  @nan_safe_enrichment decorator. Sets fn._nan_safe = True.
- Two new test files auto-discover decorated helpers via
  inspect.getmembers + the marker attribute. Registry-floor sanity
  assertions catch silent discovery breakage. ADR-003 + decorator +
  auto-discovery = self-enforcing contract perimeter.
- 10 enrichment helpers audited + decorated. Each gains a NaN-input
  semantics sentence in its docstring.

Fixes:
- add_pre_shot_gk_context (std + atomic) — primary bug; NaN-guard
  before int(...) cast; skip path matches the existing "no defending
  keeper in window" path so the documented contract holds.
- add_gk_distribution_metrics (std + atomic) — latent risk; filter
  eligible mask by np.isfinite() on all four coords.
- coverage_metrics (std + atomic) — defensive, not under ADR-003
  scope (returns TypedDict, not enrichment helper) but same int(NaN)
  crash class. Fix while we're here per "nothing deferred" principle.

Feature:
- add_gk_role gains goalkeeper_ids: set | None = None keyword-only
  parameter (std + atomic). When provided, distribution-detection
  extends with rule (a) known-GK match and rule (b) NaN-team
  fallback. Closes the lakehouse coverage gap. Default None
  preserves byte-for-byte pre-2.5.0 behavior.

Tests added:
- tests/test_enrichment_nan_safety.py — synthetic NaN fuzz +
  registry-floor sanity (~14 cases).
- tests/test_enrichment_provider_e2e.py — cross-provider e2e on
  StatsBomb/IDSSE/Metrica + atomic StatsBomb-derived (~20 cases).
- tests/test_gk_role_goalkeeper_ids.py — feature tests covering
  backward-compat, rule (a), rule (b), edge cases (~10 cases).

Test count: 884 → ~928 (+~44 net). Pyright 0/0/0.

Documentation:
- ADR-003 captures the contract + alternatives + future-direction
  toward nullable-Int64.
- CLAUDE.md "Key conventions" amendment makes the NaN-safety rule
  project-wide.
- Per-helper docstrings gain explicit NaN-input semantics sentences.

Spec: docs/superpowers/specs/2026-04-30-nan-safety-enrichment-helpers-design.md
ADR:  docs/superpowers/adrs/ADR-003-nan-safety-enrichment-helpers.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 23.3: Verify the commit landed**

Run:
```bash
git log -1 --stat
git status --short
```
Expected: commit shows the 12 file changes; working tree clean except pre-existing untracked `README.md.backup` and `uv.lock`.

### Task 24: Push, PR, CI watch, merge, tag, verify (user-driven)

**Files:**
- (none — git/GitHub operations; require user approval per CLAUDE.md)

- [ ] **Step 24.1: Push the branch**

Run (with user approval):
```bash
git push -u origin feat/nan-safety-enrichment-helpers
```

- [ ] **Step 24.2: Open the PR**

```bash
gh pr create --title "feat(spadl)!: NaN-safety contract for enrichment helpers + goalkeeper_ids — silly-kicks 2.5.0 (PR-S17)" --body "$(cat <<'EOF'
## Summary

- Codifies NaN-safety contract for 10 public enrichment helpers (5 std + 5 atomic) via ADR-003.
- `@nan_safe_enrichment` decorator + auto-discovered CI gate (registry-floor sanity bulletproof).
- Fixes 2 confirmed + 2 latent + 2 defensive crash sites.
- Adds opt-in `goalkeeper_ids` parameter on `add_gk_role` closing lakehouse IDSSE/Metrica coverage gap.

## Trigger

Lakehouse `compute_spadl_vaep` task failed on first end-to-end run on real IDSSE bronze data (2026-04-30): `ValueError: cannot convert float NaN to integer` at `silly_kicks/spadl/utils.py:543`. ADR-001 (caller-pass-through identifier convention) implies enrichment helpers must tolerate NaN identifiers; this PR makes that contract explicit and CI-enforced.

## Test plan

- [ ] T-fuzz auto-discovered: ~14 cases (5 std × NaN-laced fixture + 5 atomic × NaN-laced fixture + 2 sanity + 2 per-helper specific)
- [ ] T-e2e cross-provider: ~20 cases (5 std × {StatsBomb, IDSSE, Metrica} + 5 atomic × StatsBomb-derived)
- [ ] T-feature goalkeeper_ids: ~10 cases (backward-compat, rule a, rule b, edge cases, atomic counterparts)
- [ ] Total: 884 → ~928 passing (+~44 net delta)
- [ ] ruff / ruff format / pyright: clean

## Hyrum's Law notes

- `add_gk_role.__signature__` gains `goalkeeper_ids` keyword-only param. `goalkeeper_ids=None` default preserves byte-for-byte pre-2.5.0 behavior.
- Decoration is opt-in. CLAUDE.md "Key conventions" amendment makes decoration a project-wide rule reviewers check during PR review.

## Refs

- Spec: `docs/superpowers/specs/2026-04-30-nan-safety-enrichment-helpers-design.md`
- ADR:  `docs/superpowers/adrs/ADR-003-nan-safety-enrichment-helpers.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 24.3: Watch CI**

```bash
gh pr checks --watch
```
Expected: all matrix jobs green (lint + ubuntu 3.10/3.11/3.12 + windows).

- [ ] **Step 24.4: Squash-merge with --admin**

Once CI green:
```bash
gh pr merge --squash --admin --delete-branch
```

- [ ] **Step 24.5: Tag and push the release**

```bash
git checkout main
git pull
git tag -a v2.5.0 -m "silly-kicks 2.5.0 — NaN-safety contract for enrichment helpers + goalkeeper_ids feature (PR-S17)"
git push origin v2.5.0
```

- [ ] **Step 24.6: Watch publish workflow**

```bash
gh run watch <publish-run-id>
```

- [ ] **Step 24.7: Verify PyPI**

```bash
uv run python -c "
import urllib.request, json
with urllib.request.urlopen('https://pypi.org/pypi/silly-kicks/json') as r:
    data = json.load(r)
print('Latest PyPI version:', data['info']['version'])
print('2.5.0 release present:', '2.5.0' in data['releases'])
"
```
Expected: `Latest PyPI version: 2.5.0`.

---

## Self-review checklist (applied during writing)

- [x] **Spec coverage:** Every section of the spec has a corresponding task. § 1-3 (problem/goals/non-goals) inform the plan but don't need standalone tasks; § 4.1 (contract) → ADR-003 (Task 16); § 4.2 (decorator module) → Task 3; § 4.3 (audit + decoration) → Tasks 8-12; § 4.4 (add_pre_shot_gk_context fix) → Task 12; § 4.5 (add_gk_distribution_metrics fix) → Task 11; § 4.6 (goalkeeper_ids feature) → Task 13; § 4.7 (test infrastructure) → Tasks 4-7; § 4.8 (documentation) → Tasks 16-19; § 5 (test plan) → covered across Phase 3 (RED) and verified throughout; § 6 (verification gates) → Task 15; § 7 (risks) → captured in CHANGELOG; § 8 (acceptance criteria) → all 17 items touched.
- [x] **Bonus coverage_metrics fix:** Added Task 14 explicitly — same int(NaN) class, "nothing deferred" principle.
- [x] **Placeholder scan:** Plan-level "TODO" / "TBD" check passes (the only TODO references are intentional references to ADR-003 / TODO.md from prior PRs).
- [x] **Type consistency:** `set | None = None` parameter type used consistently across `add_gk_role` (std + atomic), tests, ADR, CHANGELOG.
- [x] **Commit policy:** No intermediate commits — single commit gated on user approval at Task 23.
