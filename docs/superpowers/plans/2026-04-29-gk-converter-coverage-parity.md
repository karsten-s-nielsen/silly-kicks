# GK Converter Coverage Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close 3 distinct production bugs in silly-kicks's sportec + metrica converters that left IDSSE bronze SPADL with zero pass-class actions and zero keeper actions, and zero keeper actions on Metrica. Ship as silly-kicks 1.10.0 with a public `coverage_metrics` utility, production-shape vendored fixtures, and a cross-provider parity regression gate.

**Architecture:** Hexagonal — pure-function converters, zero I/O, zero global state mutation. New `coverage_metrics` utility lives next to `boundary_metrics` in `silly_kicks/spadl/utils.py`. Production-shape fixtures vendored under `tests/datasets/{idsse,metrica}/`. Cross-provider parity meta-test at `tests/spadl/test_cross_provider_parity.py` parametrizes over all 5 DataFrame converters.

**Tech Stack:** Python 3.10+, pandas 2.x (with pandas-stubs 2.3.3.260113 pinned), numpy 2.x, pytest, ruff 0.15.7, pyright 1.1.395. No new runtime deps.

**Convention deviation from writing-plans defaults:** silly-kicks ships ONE commit per branch (per `feedback_commit_policy` memory). Tasks below stage changes (`git add`) but DO NOT commit. The single commit happens in Task 12 after `/final-review` and explicit user approval (sentinel-gated by hook).

**Spec alignment note:** The spec at `docs/superpowers/specs/2026-04-29-gk-converter-coverage-parity-design.md` § 4.3 describes Bug 1 as a "single-line replacement at sportec.py:495". Code analysis (this session) shows the single-line change is insufficient: `is_play_no_action` at sportec.py:595-596 would still overwrite non-GK Play events back to `non_action` after the pass-class assignment. The plan below implements the *correct* multi-line restructure (Task 3) and supersedes the spec's single-line framing. Spec amendment to follow.

---

## Branch + Worktree

**Branch:** `feat/gk-converter-coverage-parity`
**Worktree:** None — single-commit ritual operates directly on the main folder. (The other Claude session is in luxury-lakehouse, not silly-kicks; no concurrent risk in this repo.)

## File Structure

| Path | Action | Δ Lines | Purpose |
|---|---|---|---|
| `silly_kicks/spadl/utils.py` | MOD | +60 / 0 | New `CoverageMetrics` TypedDict + `coverage_metrics()` public utility (mirrors `BoundaryMetrics` / `boundary_metrics` shape from PR-S8) |
| `silly_kicks/spadl/__init__.py` | MOD | +3 / 0 | Re-export `coverage_metrics` + `CoverageMetrics` |
| `silly_kicks/spadl/sportec.py` | MOD | +110 / -10 | Bug 1 (Pass→Play restructure), Bug 2 (throwOut/punt 2-action synthesis), goalkeeper_ids supplementary signal, vocabulary tables in docstring |
| `silly_kicks/spadl/metrica.py` | MOD | +90 / -5 | goalkeeper_ids primary mechanism (PASS→synth, RECOVERY→pickup, AERIAL-WON→claim), docstring limitation note |
| `silly_kicks/spadl/statsbomb.py` | MOD | +6 / 0 | Accept `goalkeeper_ids: set[int] \| None` as no-op (API symmetry); document |
| `silly_kicks/spadl/opta.py` | MOD | +6 / 0 | Accept `goalkeeper_ids: set[int] \| None` as no-op (API symmetry); document |
| `silly_kicks/spadl/wyscout.py` | NO CHANGE | 0 | Already has `goalkeeper_ids` from 1.0.0 |
| `tests/datasets/idsse/sample_match.parquet` | NEW | ~30 KB | Production-shape IDSSE bronze subset (~200-400 rows including throwOut/punt rows, sourced via Databricks pull from `bronze.idsse_events` match J03WMX) |
| `tests/datasets/idsse/README.md` | NEW | ~25 lines | DFL DataHub attribution, source match_id, license note, regeneration command |
| `tests/datasets/metrica/sample_match.parquet` | NEW | ~30 KB | Production-shape Metrica bronze subset (~300 rows, sourced from kloppy's vendored metrica_events.json subset, converted to bronze shape) |
| `tests/datasets/metrica/README.md` | NEW | ~25 lines | Metrica open-data attribution, source, license note (CC-BY-NC), regeneration command |
| `tests/spadl/test_sportec.py` | MOD | +200 / 0 | TestSportecPlayEventRecognition + TestSportecGKQualifierSynthesis + TestSportecGoalkeeperIdsSupplementary + TestSportecCoverageOnProductionFixture |
| `tests/spadl/test_metrica.py` | MOD | +130 / 0 | TestMetricaGoalkeeperIdsRouting + TestMetricaCoverageOnProductionFixture |
| `tests/spadl/test_statsbomb.py` | MOD | +25 / 0 | TestStatsBombGoalkeeperIdsNoOp |
| `tests/spadl/test_opta.py` | MOD | +25 / 0 | TestOptaGoalkeeperIdsNoOp |
| `tests/spadl/test_coverage_metrics.py` | NEW | ~140 lines | TestCoverageMetricsContract / Correctness / Degenerate (mirrors TestBoundaryMetricsXxx in test_add_possessions.py) |
| `tests/spadl/test_cross_provider_parity.py` | NEW | ~90 lines | Cross-provider parametrized parity meta-test |
| `scripts/extract_provider_fixtures.py` | NEW | ~180 lines | Build script: Databricks pull for IDSSE (Option A), kloppy fixture subset for Metrica (Option B) |
| `pyproject.toml` | MOD | +1 / -1 | Version `1.9.0` → `1.10.0` |
| `CHANGELOG.md` | MOD | +90 lines | `## [1.10.0]` entry |
| `TODO.md` | MOD | ~+5 / -3 | PR-S10 → shipped; PR-S11 (was PR-S10 add_possessions improvement) bumped in queue; atomic coverage_metrics tech debt |
| `docs/c4/architecture.dsl` | MOD | +1 / -1 | Add `coverage_metrics` to spadl container public-helper enumeration |
| `docs/c4/architecture.html` | REGEN | (script) | Regenerated from .dsl |
| `docs/superpowers/specs/2026-04-29-gk-converter-coverage-parity-design.md` | TRACK | (spec) | Already exists; bundled into the single commit |
| `docs/superpowers/plans/2026-04-29-gk-converter-coverage-parity.md` | THIS FILE | (plan) | Bundled into the single commit |

---

## Task 0: Pre-flight verification + branch creation

**Files:** none (shell-only)

- [ ] **Step 0.1: Verify clean working tree at v1.9.0**

Run:
```bash
git status
git log --oneline -1
grep "^version" pyproject.toml
```

Expected: `On branch main`, `Your branch is up to date with 'origin/main'`, only the spec + uv.lock + README.md.backup as untracked files; HEAD is `fb12b33 ... silly-kicks 1.9.0 (#14)`; pyproject.toml shows `version = "1.9.0"`.

If anything else is dirty, STOP and ask the user before proceeding.

- [ ] **Step 0.2: Create + checkout the feature branch**

Run:
```bash
git checkout -b feat/gk-converter-coverage-parity
```

Expected: `Switched to a new branch 'feat/gk-converter-coverage-parity'`.

- [ ] **Step 0.3: Verify CI pin install**

Run (with explicit timeout, may take ~20-30s):
```bash
uv pip install --upgrade ruff==0.15.7 pyright==1.1.395 pandas-stubs==2.3.3.260113
```

Expected: install succeeds with no errors. (If `uv` is not available locally, fall back to `pip install --upgrade ...`.)

- [ ] **Step 0.4: Baseline test pass + lint clean**

Run:
```bash
uv run pytest tests/ -m "not e2e" -v --tb=short -q 2>&1 | tail -20
uv run ruff check silly_kicks/ tests/ scripts/ 2>&1 | tail -5
uv run ruff format --check silly_kicks/ tests/ scripts/ 2>&1 | tail -5
uv run pyright silly_kicks/ 2>&1 | tail -5
```

Expected: all green; pytest ~560 passed; ruff/format/pyright zero errors. This is the baseline against which we'll measure later.

If anything fails, STOP and investigate before adding new code on top.

---

## Task 1: `coverage_metrics` public utility (TDD)

**Files:**
- Create: `tests/spadl/test_coverage_metrics.py`
- Modify: `silly_kicks/spadl/utils.py:716` (add new `CoverageMetrics` + `coverage_metrics` after the existing `BoundaryMetrics` + `boundary_metrics` block)
- Modify: `silly_kicks/spadl/__init__.py` (re-export)

This task mirrors the PR-S8 `BoundaryMetrics` / `boundary_metrics` discipline (see `tests/spadl/test_add_possessions.py::TestBoundaryMetricsContract` etc.). Lifecycle: write failing tests → implement → re-export → verify.

- [ ] **Step 1.1: Create the failing test file**

Create `tests/spadl/test_coverage_metrics.py`:

```python
"""Tests for ``silly_kicks.spadl.coverage_metrics`` (added in 1.10.0).

Mirrors the PR-S8 ``boundary_metrics`` test discipline. Coverage is measured
on a SPADL action stream, resolving ``type_id`` to action-type name via
``spadlconfig.actiontypes_df`` and reporting per-type counts plus any
``expected_action_types`` that produced zero rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl.utils import CoverageMetrics, coverage_metrics

_ACT = spadlconfig.actiontype_id


def _df_one(action_type: str = "pass") -> pd.DataFrame:
    """Single-row SPADL-shaped DataFrame with one action of the given type."""
    return pd.DataFrame({"type_id": [_ACT[action_type]]})


def _df_many(action_types: list[str]) -> pd.DataFrame:
    """SPADL-shaped DataFrame with one row per supplied action type name."""
    return pd.DataFrame({"type_id": [_ACT[t] for t in action_types]})


class TestCoverageMetricsContract:
    def test_returns_dict_with_required_keys(self):
        m = coverage_metrics(actions=_df_one("pass"))
        assert set(m.keys()) == {"counts", "missing", "total_actions"}

    def test_counts_value_type_is_int(self):
        m = coverage_metrics(actions=_df_many(["pass", "pass", "shot"]))
        for v in m["counts"].values():
            assert isinstance(v, int)

    def test_total_actions_is_int(self):
        m = coverage_metrics(actions=_df_many(["pass", "shot"]))
        assert isinstance(m["total_actions"], int)
        assert m["total_actions"] == 2

    def test_keyword_only_args_required(self):
        with pytest.raises(TypeError):
            coverage_metrics(_df_one("pass"))  # type: ignore[misc]

    def test_missing_type_id_column_raises_value_error(self):
        actions = pd.DataFrame({"team_id": [1, 2]})
        with pytest.raises(ValueError, match=r"type_id"):
            coverage_metrics(actions=actions)

    def test_returns_typeddict_shape(self):
        m: CoverageMetrics = coverage_metrics(actions=_df_one("pass"))
        # TypedDict access pattern
        assert m["counts"]["pass"] == 1


class TestCoverageMetricsCorrectness:
    def test_single_type_dataframe_counts_correctly(self):
        m = coverage_metrics(actions=_df_many(["pass", "pass", "pass"]))
        assert m["counts"] == {"pass": 3}
        assert m["total_actions"] == 3

    def test_multi_type_dataframe_counts_correctly(self):
        m = coverage_metrics(actions=_df_many(["pass", "pass", "shot", "tackle"]))
        assert m["counts"] == {"pass": 2, "shot": 1, "tackle": 1}
        assert m["total_actions"] == 4

    def test_expected_fully_present_missing_empty(self):
        m = coverage_metrics(
            actions=_df_many(["pass", "shot", "tackle"]),
            expected_action_types={"pass", "shot", "tackle"},
        )
        assert m["missing"] == []

    def test_expected_partially_absent_missing_sorted(self):
        m = coverage_metrics(
            actions=_df_many(["pass", "shot"]),
            expected_action_types={"pass", "shot", "tackle", "keeper_save"},
        )
        assert m["missing"] == ["keeper_save", "tackle"]

    def test_unknown_type_id_reported_as_unknown(self):
        # type_id = 999 is not in the spadlconfig.actiontype_id reverse map.
        actions = pd.DataFrame({"type_id": [_ACT["pass"], 999, 999]})
        m = coverage_metrics(actions=actions)
        assert m["counts"].get("pass") == 1
        assert m["counts"].get("unknown") == 2

    def test_counts_dict_is_ordered_by_action_type_alphabetical_or_first_seen(self):
        # Either alphabetical or first-seen ordering is acceptable, as long
        # as it is deterministic. Verify determinism by repeating.
        df = _df_many(["shot", "pass", "tackle"])
        m1 = coverage_metrics(actions=df)
        m2 = coverage_metrics(actions=df)
        assert list(m1["counts"].keys()) == list(m2["counts"].keys())


class TestCoverageMetricsDegenerate:
    def test_empty_dataframe_returns_zeros(self):
        actions = pd.DataFrame({"type_id": pd.Series([], dtype=np.int64)})
        m = coverage_metrics(actions=actions)
        assert m["counts"] == {}
        assert m["missing"] == []
        assert m["total_actions"] == 0

    def test_empty_with_expected_returns_all_missing_sorted(self):
        actions = pd.DataFrame({"type_id": pd.Series([], dtype=np.int64)})
        m = coverage_metrics(
            actions=actions,
            expected_action_types={"pass", "shot", "keeper_save"},
        )
        assert m["counts"] == {}
        assert m["missing"] == ["keeper_save", "pass", "shot"]
        assert m["total_actions"] == 0

    def test_expected_action_types_none_returns_empty_missing(self):
        m = coverage_metrics(
            actions=_df_many(["pass", "shot"]),
            expected_action_types=None,
        )
        assert m["missing"] == []

    def test_does_not_mutate_input(self):
        actions = _df_many(["pass", "shot"])
        cols_before = list(actions.columns)
        len_before = len(actions)
        coverage_metrics(actions=actions, expected_action_types={"pass"})
        assert list(actions.columns) == cols_before
        assert len(actions) == len_before
```

- [ ] **Step 1.2: Run the test file and verify it fails with ImportError**

Run:
```bash
uv run pytest tests/spadl/test_coverage_metrics.py -v --tb=short
```

Expected: collection error or test failures because `coverage_metrics` and `CoverageMetrics` are not yet exported from `silly_kicks.spadl.utils`. Errors like `ImportError: cannot import name 'coverage_metrics'`.

- [ ] **Step 1.3: Implement `CoverageMetrics` + `coverage_metrics` in utils.py**

In `silly_kicks/spadl/utils.py`, after the existing `boundary_metrics` function (which currently ends at line 818), insert the new TypedDict + function:

```python
class CoverageMetrics(TypedDict):
    """Per-action-type coverage statistics for a SPADL action stream.

    Returned by :func:`coverage_metrics` for downstream coverage validation
    and silly-kicks's own cross-provider parity regression gate.

    Attributes
    ----------
    counts : dict[str, int]
        Maps action-type name to row count. Action-type names are resolved
        from ``type_id`` via :func:`silly_kicks.spadl.config.actiontypes_df`.
        ``type_id`` values not found in the canonical spadlconfig vocabulary
        are reported under the name ``"unknown"`` (no exception raised).
    missing : list[str]
        Action types that the caller passed via ``expected_action_types`` but
        which produced zero rows in ``actions``. Returned sorted (so test
        assertions are stable). Empty list when ``expected_action_types`` is
        ``None`` or every expected type was present.
    total_actions : int
        Row count of the input ``actions`` DataFrame.
    """

    counts: dict[str, int]
    missing: list[str]
    total_actions: int


def coverage_metrics(
    *,
    actions: pd.DataFrame,
    expected_action_types: set[str] | None = None,
) -> CoverageMetrics:
    """Compute SPADL action-type coverage for an action DataFrame.

    Resolves ``type_id`` to action-type name via
    :func:`silly_kicks.spadl.config.actiontypes_df` and counts each action
    type present. When ``expected_action_types`` is provided, returns any
    of those types with zero rows under ``missing``.

    Use cases:

    1. Test discipline — assert converter X emits action types Y on a
       fixture (used by silly-kicks 1.10.0's cross-provider parity
       regression gate).
    2. Downstream validation — consumers calling silly-kicks-converted
       bronze data can verify expected coverage before downstream
       aggregation.

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL action stream. Must contain ``type_id``.
    expected_action_types : set[str] or None, default ``None``
        Action type names expected to be present. Returned (sorted) as
        ``missing`` if absent. ``None`` skips the expectation check; an
        empty list is then returned for ``missing``.

    Returns
    -------
    CoverageMetrics
        ``{"counts": {...}, "missing": [...], "total_actions": N}``.

    Raises
    ------
    ValueError
        If the ``type_id`` column is missing.

    Examples
    --------
    Validate IDSSE bronze→SPADL output covers all expected action types::

        from silly_kicks.spadl import sportec, coverage_metrics
        actions, _ = sportec.convert_to_actions(
            events, home_team_id="HOME", goalkeeper_ids={"DFL-OBJ-..."}
        )
        m = coverage_metrics(
            actions=actions,
            expected_action_types={"pass", "shot", "tackle", "keeper_pick_up"},
        )
        assert not m["missing"], f"Missing action types: {m['missing']}"
    """
    if "type_id" not in actions.columns:
        raise ValueError(
            f"coverage_metrics: actions missing required 'type_id' column. "
            f"Got: {sorted(actions.columns)}"
        )

    n = len(actions)
    counts: dict[str, int] = {}
    if n > 0:
        # Reverse map: id -> name. Built from spadlconfig.actiontypes (single
        # source of truth). Out-of-vocab ids report as "unknown".
        id_to_name = {i: name for i, name in enumerate(spadlconfig.actiontypes)}
        type_id_arr = actions["type_id"].to_numpy()
        # Tally with deterministic insertion order (numpy iteration order on
        # the input array). Equivalent to pd.value_counts but preserves
        # first-seen ordering for stable user-facing output.
        for tid in type_id_arr:
            name = id_to_name.get(int(tid), "unknown")
            counts[name] = counts.get(name, 0) + 1

    expected = set(expected_action_types) if expected_action_types else set()
    missing = sorted(expected - set(counts.keys())) if expected else []

    return CoverageMetrics(counts=counts, missing=missing, total_actions=n)
```

- [ ] **Step 1.4: Re-export from `silly_kicks/spadl/__init__.py`**

Edit `silly_kicks/spadl/__init__.py`:

```python
__all__ = [
    "SPADL_COLUMNS",
    "BoundaryMetrics",
    "ConversionReport",
    "CoverageMetrics",
    "actiontypes_df",
    "add_gk_distribution_metrics",
    "add_gk_role",
    "add_names",
    "add_possessions",
    "add_pre_shot_gk_context",
    "bodyparts_df",
    "boundary_metrics",
    "config",
    "coverage_metrics",
    "kloppy",
    "opta",
    "play_left_to_right",
    "results_df",
    "statsbomb",
    "validate_spadl",
    "wyscout",
]
```

And update the `from .utils import` block:

```python
from .utils import (
    BoundaryMetrics,
    CoverageMetrics,
    add_gk_distribution_metrics,
    add_gk_role,
    add_names,
    add_possessions,
    add_pre_shot_gk_context,
    boundary_metrics,
    coverage_metrics,
    play_left_to_right,
    validate_spadl,
)
```

- [ ] **Step 1.5: Run the test file and verify all PASS**

Run:
```bash
uv run pytest tests/spadl/test_coverage_metrics.py -v --tb=short
```

Expected: all ~14 tests pass.

- [ ] **Step 1.6: Stage the changes (DO NOT commit)**

Run:
```bash
git add silly_kicks/spadl/utils.py silly_kicks/spadl/__init__.py tests/spadl/test_coverage_metrics.py
git status
```

Expected: 3 files staged, no commit yet.

---

## Task 2: Vendor production-shape fixtures via build script

**Files:**
- Create: `scripts/extract_provider_fixtures.py`
- Create: `tests/datasets/idsse/sample_match.parquet` (~30 KB binary, generated)
- Create: `tests/datasets/idsse/README.md`
- Create: `tests/datasets/metrica/sample_match.parquet` (~30 KB binary, generated)
- Create: `tests/datasets/metrica/README.md`

**Source decision (resolved per spec § 5):**
- **IDSSE:** Option A (Databricks pull). The kloppy-vendored `tests/datasets/kloppy/sportec_events.xml` (29 events) has zero `throwOut`/`punt` qualifiers (verified this session). Pull a subset of `bronze.idsse_events` for match `J03WMX` (which has 49 throwOut + 25 punt rows known from prior probes) using env-var auth (`DATABRICKS_HOST`, `DATABRICKS_TOKEN`, `DATABRICKS_HTTP_PATH`).
- **Metrica:** Option B (kloppy fixture subset). The existing vendored `tests/datasets/kloppy/metrica_events.json` (3,620 events, public Metrica Sample Game 2) has no native GK markers — exactly the empirical situation Bug 3 fixes. Subset to ~300 rows + convert to bronze shape.

- [ ] **Step 2.1: Create `scripts/extract_provider_fixtures.py`**

Create `scripts/extract_provider_fixtures.py`:

```python
"""Build production-shape fixtures for sportec + metrica converter tests.

Two extractors with two source strategies:

1. IDSSE (Sportec/DFL): pulls a subset of ``bronze.idsse_events`` from the
   Databricks lakehouse via ``databricks-sql-connector`` using env-var auth.
   Source match: ``idsse_J03WMX`` (known to contain throwOut + punt
   qualifiers). Subset: ~200-400 representative rows including all
   keeper-relevant rows.

2. Metrica: subsets the already-vendored kloppy fixture
   ``tests/datasets/kloppy/metrica_events.json`` (Metrica Sample Game 2,
   3,620 events) and converts it to the bronze shape expected by
   ``silly_kicks.spadl.metrica.convert_to_actions``. No network required.

Both extractors write their output to ``tests/datasets/{provider}/sample_match.parquet``
as small (~30 KB compressed) files small enough to commit.

Usage::

    # Both extractors (default).
    python scripts/extract_provider_fixtures.py

    # Single provider.
    python scripts/extract_provider_fixtures.py --provider idsse
    python scripts/extract_provider_fixtures.py --provider metrica

    # Skip IDSSE if no Databricks creds in env.
    python scripts/extract_provider_fixtures.py --provider metrica

Env vars required for the IDSSE extractor:

- ``DATABRICKS_HOST``
- ``DATABRICKS_TOKEN``
- ``DATABRICKS_HTTP_PATH``

Without them, the IDSSE path raises and exits non-zero. The Metrica path
runs offline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
_IDSSE_OUT = _REPO_ROOT / "tests" / "datasets" / "idsse" / "sample_match.parquet"
_METRICA_OUT = _REPO_ROOT / "tests" / "datasets" / "metrica" / "sample_match.parquet"
_METRICA_KLOPPY_SOURCE = _REPO_ROOT / "tests" / "datasets" / "kloppy" / "metrica_events.json"

# IDSSE source match — known to contain throwOut + punt qualifiers
# (verified via direct probe against bronze.idsse_events 2026-04-29).
_IDSSE_SOURCE_MATCH_ID = "idsse_J03WMX"

# Maximum rows in the committed fixture. Small enough to commit; large
# enough to exercise pass-class events + GK qualifier rows + multiple
# event types.
_IDSSE_MAX_ROWS = 400


def _extract_idsse(out_path: Path) -> None:
    """Pull a subset of bronze.idsse_events for the source match.

    Bronze schema is preserved verbatim (the silly-kicks sportec converter
    consumes bronze shape directly). Subset prioritises:

    1. ALL Play events with non-null play_goal_keeper_action (the GK rows
       PR-S10 fixes).
    2. A representative sample of other event types (Play default, Shot,
       Tackle, Foul, FreeKick, Corner, ThrowIn, GoalKick).

    The result is written as a parquet (snappy compression) file.
    """
    try:
        from databricks import sql as dbsql
    except ImportError:
        print(
            "ERROR: databricks-sql-connector not installed. "
            "Install with: uv pip install databricks-sql-connector",
            file=sys.stderr,
        )
        sys.exit(1)

    host = os.environ.get("DATABRICKS_HOST")
    token = os.environ.get("DATABRICKS_TOKEN")
    http_path = os.environ.get("DATABRICKS_HTTP_PATH")
    missing = [
        n for n, v in (("DATABRICKS_HOST", host), ("DATABRICKS_TOKEN", token), ("DATABRICKS_HTTP_PATH", http_path)) if not v
    ]
    if missing:
        print(f"ERROR: missing env vars: {missing}", file=sys.stderr)
        sys.exit(1)

    # Query: keep all GK-relevant Play rows + a stratified sample of other rows.
    # The UNION ALL preserves a deterministic shape; row order is then made
    # deterministic by ORDER BY on (period, timestamp_seconds, event_id).
    query = f"""
    WITH source AS (
      SELECT * FROM bronze.idsse_events WHERE match_id = '{_IDSSE_SOURCE_MATCH_ID}'
    ),
    gk_rows AS (
      SELECT * FROM source
      WHERE event_type = 'Play' AND play_goal_keeper_action IS NOT NULL
    ),
    other_rows AS (
      SELECT * FROM source
      WHERE NOT (event_type = 'Play' AND play_goal_keeper_action IS NOT NULL)
      ORDER BY rand(42)
      LIMIT {_IDSSE_MAX_ROWS - 100}
    )
    SELECT * FROM gk_rows
    UNION ALL
    SELECT * FROM other_rows
    ORDER BY period, timestamp_seconds, event_id
    """

    print(f"Connecting to Databricks at {host}...")
    with dbsql.connect(server_hostname=host, http_path=http_path, access_token=token) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            df = cur.fetchall_arrow().to_pandas()

    print(f"Pulled {len(df)} rows from {_IDSSE_SOURCE_MATCH_ID}")
    print(f"event_type counts: {df['event_type'].value_counts().to_dict()}")
    if "play_goal_keeper_action" in df.columns:
        gk_counts = df.loc[df["play_goal_keeper_action"].notna(), "play_goal_keeper_action"].value_counts().to_dict()
        print(f"play_goal_keeper_action counts: {gk_counts}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, compression="snappy", index=False)
    size_kb = out_path.stat().st_size / 1024
    print(f"Wrote {out_path} ({size_kb:.1f} KB, {len(df)} rows)")


def _extract_metrica(out_path: Path) -> None:
    """Subset the kloppy-vendored Metrica fixture and convert to bronze shape.

    Source: ``tests/datasets/kloppy/metrica_events.json`` (Metrica Sample
    Game 2, public open-data, CC-BY-NC). The kloppy file is the kloppy-
    parsed JSON shape; we subset to ~300 representative events and convert
    to the bronze shape expected by ``silly_kicks.spadl.metrica`` (columns:
    ``match_id``, ``event_id``, ``type``, ``subtype``, ``period``,
    ``start_time_s``, ``end_time_s``, ``player``, ``team``, ``start_x``,
    ``start_y``, ``end_x``, ``end_y``).

    Goalkeeper player_ids are identified post-extraction by examining the
    raw data and writing the known GK ids into the README. No network.
    """
    if not _METRICA_KLOPPY_SOURCE.exists():
        print(f"ERROR: kloppy source not found at {_METRICA_KLOPPY_SOURCE}", file=sys.stderr)
        sys.exit(1)

    with open(_METRICA_KLOPPY_SOURCE, encoding="utf-8") as f:
        raw = json.load(f)

    # The kloppy metrica_events.json shape: {"data": [...]} with each event
    # having keys including type, subtypes, period, start.time, end.time,
    # player.player_id, team.id, start.x, start.y, end.x, end.y.
    events_raw = raw.get("data", raw if isinstance(raw, list) else [])

    # Subset: keep first ~300 events with non-null player + type. This
    # preserves event ordering and gives a representative slice.
    subset = []
    for ev in events_raw:
        if len(subset) >= 300:
            break
        # Defensive: only keep events with the keys we need.
        if not isinstance(ev, dict):
            continue
        subset.append(ev)

    bronze_rows = []
    for i, ev in enumerate(subset):
        typ = (ev.get("type") or {}).get("name") if isinstance(ev.get("type"), dict) else ev.get("type")
        sub = ev.get("subtypes") or ev.get("subtype")
        # Metrica's "subtype" can be a list — flatten to the first if so.
        if isinstance(sub, list):
            sub = sub[0].get("name") if (sub and isinstance(sub[0], dict)) else (sub[0] if sub else None)
        elif isinstance(sub, dict):
            sub = sub.get("name")

        period = (ev.get("period") or {}).get("id") if isinstance(ev.get("period"), dict) else ev.get("period") or 1
        start_time = ((ev.get("start") or {}).get("time") if isinstance(ev.get("start"), dict) else None) or 0.0
        end_time = ((ev.get("end") or {}).get("time") if isinstance(ev.get("end"), dict) else None) or start_time

        player_obj = ev.get("player") or {}
        player = player_obj.get("player_id") if isinstance(player_obj, dict) else player_obj

        team_obj = ev.get("team") or {}
        team = team_obj.get("id") if isinstance(team_obj, dict) else team_obj

        start_obj = ev.get("start") or {}
        end_obj = ev.get("end") or {}
        start_x = start_obj.get("x", 50.0) if isinstance(start_obj, dict) else 50.0
        start_y = start_obj.get("y", 34.0) if isinstance(start_obj, dict) else 34.0
        end_x = end_obj.get("x", start_x) if isinstance(end_obj, dict) else start_x
        end_y = end_obj.get("y", start_y) if isinstance(end_obj, dict) else start_y

        bronze_rows.append(
            {
                "match_id": "metrica_sample_game_2",
                "event_id": i,
                "type": str(typ).upper() if typ else "GENERIC",
                "subtype": str(sub).upper() if sub else None,
                "period": int(period),
                "start_time_s": float(start_time),
                "end_time_s": float(end_time),
                "player": str(player) if player else None,
                "team": str(team) if team else None,
                "start_x": float(start_x),
                "start_y": float(start_y),
                "end_x": float(end_x),
                "end_y": float(end_y),
            }
        )

    df = pd.DataFrame(bronze_rows)
    print(f"Subsetted {len(df)} Metrica events from kloppy fixture")
    print(f"type counts: {df['type'].value_counts().to_dict()}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, compression="snappy", index=False)
    size_kb = out_path.stat().st_size / 1024
    print(f"Wrote {out_path} ({size_kb:.1f} KB, {len(df)} rows)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=["idsse", "metrica", "all"],
        default="all",
        help="Which provider's fixture to (re)build.",
    )
    args = parser.parse_args()

    if args.provider in ("idsse", "all"):
        _extract_idsse(_IDSSE_OUT)
    if args.provider in ("metrica", "all"):
        _extract_metrica(_METRICA_OUT)

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2.2: Run the metrica extractor (offline, no network needed)**

Run (with explicit timeout):
```bash
uv run python scripts/extract_provider_fixtures.py --provider metrica
```

Expected: `Wrote .../metrica/sample_match.parquet (~25-30 KB, ~300 rows)`. Type counts shown should include PASS, CARRY, RECOVERY, SHOT, etc.

- [ ] **Step 2.3: Run the IDSSE extractor (requires Databricks env vars)**

Verify env vars are set:
```bash
echo "HOST: ${DATABRICKS_HOST:+SET}, TOKEN: ${DATABRICKS_TOKEN:+SET}, PATH: ${DATABRICKS_HTTP_PATH:+SET}"
```

Expected: all three say `SET`. If any are unset, ask the user to set them (the brief notes they were set in the prior session).

Then run:
```bash
uv run python scripts/extract_provider_fixtures.py --provider idsse
```

Expected: `Pulled ~400 rows from idsse_J03WMX`, `play_goal_keeper_action counts: {'throwOut': 49, 'punt': 25}` (or similar — exact counts depend on the GK + sample mix, but throwOut and punt should both appear with positive counts), `Wrote .../idsse/sample_match.parquet (~25-35 KB, ~400 rows)`.

If the `databricks-sql-connector` package is missing, install it:
```bash
uv pip install databricks-sql-connector
```

- [ ] **Step 2.4: Create `tests/datasets/idsse/README.md`**

Create:

```markdown
# IDSSE (Sportec/DFL) production-shape fixture

`sample_match.parquet` is a subset of one DFL DataHub match (`idsse_J03WMX`)
sourced from luxury-lakehouse production `bronze.idsse_events`, retained at
the bronze schema level (no aggregation, no enrichment).

## Provenance

- Source: DFL DataHub free-sample data, parsed by the luxury-lakehouse
  ingestion pipeline into `bronze.idsse_events`.
- Source match_id: `idsse_J03WMX` (Bundesliga 2024/25 season; full match_id
  is a public DFL competition identifier — no PII).
- Subset rule: every Play event with non-null `play_goal_keeper_action` plus
  a stratified sample of other event types, capped at ~400 rows. Designed to
  exercise the throwOut + punt qualifier paths fixed in silly-kicks 1.10.0
  (PR-S10 Bug 2).
- Extracted: 2026-04-29 via `scripts/extract_provider_fixtures.py --provider idsse`.

## License

DFL DataHub free-sample license permits non-commercial redistribution.
This fixture is included **only for testing** the silly-kicks open-source
library. Test fixtures are excluded from the published `silly-kicks` wheel
via `[tool.hatch.build.targets.wheel] packages = ["silly_kicks"]` in
`pyproject.toml`.

## Regenerating

```bash
# Requires DATABRICKS_HOST / DATABRICKS_TOKEN / DATABRICKS_HTTP_PATH env vars
uv run python scripts/extract_provider_fixtures.py --provider idsse
```
```

- [ ] **Step 2.5: Create `tests/datasets/metrica/README.md`**

Create:

```markdown
# Metrica production-shape fixture

`sample_match.parquet` is a ~300-row subset of Metrica Sample Game 2,
converted to the bronze shape expected by `silly_kicks.spadl.metrica`.

## Provenance

- Source: [`metrica-sports/sample-data`](https://github.com/metrica-sports/sample-data)
  Sample Game 2, vendored upstream into kloppy's test corpus and then into
  silly-kicks at `tests/datasets/kloppy/metrica_events.json`.
- Subset: first ~300 events with deterministic ordering, converted to
  bronze schema (`match_id`, `event_id`, `type`, `subtype`, `period`,
  `start_time_s`, `end_time_s`, `player`, `team`, `start_x`, `start_y`,
  `end_x`, `end_y`).
- No native GK markers exist anywhere in the Metrica event taxonomy —
  this is exactly the empirical situation that PR-S10 Bug 3 fixes by
  introducing the `goalkeeper_ids: set[str] | None` parameter on
  `silly_kicks.spadl.metrica.convert_to_actions`. Tests using this
  fixture supply the known goalkeeper player_ids from the source match.
- Extracted: 2026-04-29 via `scripts/extract_provider_fixtures.py --provider metrica`.

## License

Metrica Sample Game 2 is published under **CC-BY-NC-4.0**. This fixture is
included **only for testing** the silly-kicks open-source library
(non-commercial use). Test fixtures are excluded from the published
`silly-kicks` wheel via `[tool.hatch.build.targets.wheel] packages = ["silly_kicks"]`
in `pyproject.toml`.

## Regenerating

```bash
uv run python scripts/extract_provider_fixtures.py --provider metrica
```

(Reads `tests/datasets/kloppy/metrica_events.json` directly — no network.)
```

- [ ] **Step 2.6: Verify both fixtures exist and are committable**

Run:
```bash
ls -la tests/datasets/idsse/sample_match.parquet tests/datasets/metrica/sample_match.parquet
```

Expected: both files exist, each ~25-35 KB.

- [ ] **Step 2.7: Stage the fixtures + script + READMEs (DO NOT commit)**

Run:
```bash
git add scripts/extract_provider_fixtures.py \
        tests/datasets/idsse/sample_match.parquet tests/datasets/idsse/README.md \
        tests/datasets/metrica/sample_match.parquet tests/datasets/metrica/README.md
git status
```

Expected: 5 new files staged.

---

## Task 3: sportec.py Bug 1 — recognize DFL `Play` event_type for pass-class events (TDD)

**Files:**
- Modify: `silly_kicks/spadl/sportec.py:495` (and the `is_play_no_action` block at 595-596 — the spec's "single-line" framing was insufficient; see plan header note)
- Modify: `tests/spadl/test_sportec.py` (add `TestSportecPlayEventRecognition`)

**Behaviour change summary:**
- Currently: `is_pass = et == "Pass"` (sportec.py:495). DFL bronze never emits `"Pass"`; the actual event_type is `"Play"`. So all pass-class events drop to `non_action`.
- After fix: Restructure the Play branch so a Play event with NO recognized GK qualifier maps to `pass` / `cross` (with optional head bodypart), and a Play event with a recognized GK qualifier maps to `keeper_*` (existing behaviour preserved).
- The `is_play_no_action` filter is refined: only Play events with an UNRECOGNIZED non-empty GK qualifier go to `non_action` (defensive). Play events with empty/null GK qualifier fall through to pass-class.

- [ ] **Step 3.1: Write the failing tests for Play event recognition**

Append to `tests/spadl/test_sportec.py` (after `TestSportecActionMappingShotsTacklesFoulsGK`):

```python
# ---------------------------------------------------------------------------
# Bug 1 — DFL "Play" event_type maps to pass-class actions
# (1.10.0; supersedes pre-1.10.0 dispatch where `is_pass = et == "Pass"`
# silently dropped all DFL Play events to non_action)
# ---------------------------------------------------------------------------


def _df_play_default() -> pd.DataFrame:
    """A bare DFL Play event with no GK qualifier — should map to "pass"."""
    return pd.DataFrame(
        {
            "match_id": ["M1"],
            "event_id": ["e1"],
            "event_type": ["Play"],
            "period": [1],
            "timestamp_seconds": [10.0],
            "player_id": ["P1"],
            "team": ["T-HOME"],
            "x": [50.0],
            "y": [34.0],
        }
    )


def _df_play_cross() -> pd.DataFrame:
    df = _df_play_default()
    df["play_height"] = ["cross"]
    return df


def _df_play_head() -> pd.DataFrame:
    df = _df_play_default()
    df["play_height"] = ["head"]
    return df


class TestSportecPlayEventRecognition:
    """DFL "Play" event_type is the pass-class event in the DFL vocabulary
    (verified empirically against bronze.idsse_events). The pre-1.10.0
    converter checked ``et == "Pass"`` — silently dropping ALL pass-class
    events to non_action across all IDSSE matches in production.
    """

    def test_play_default_maps_to_pass(self):
        actions, _ = sportec_mod.convert_to_actions(_df_play_default(), home_team_id="T-HOME")
        assert len(actions) == 1, "DFL Play events without GK qualifier should produce a SPADL action"
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["pass"]

    def test_play_cross_maps_to_cross(self):
        actions, _ = sportec_mod.convert_to_actions(_df_play_cross(), home_team_id="T-HOME")
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["cross"]

    def test_play_head_uses_head_bodypart(self):
        actions, _ = sportec_mod.convert_to_actions(_df_play_head(), home_team_id="T-HOME")
        assert actions["bodypart_id"].iloc[0] == spadlconfig.bodypart_id["head"]
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["pass"]

    def test_play_with_recognized_gk_qualifier_keeps_keeper_action_mapping(self):
        # Pre-1.10.0 behavior preserved: Play+save → keeper_save.
        df = _df_play_default()
        df["play_goal_keeper_action"] = ["save"]
        actions, _ = sportec_mod.convert_to_actions(df, home_team_id="T-HOME")
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["keeper_save"]

    def test_play_with_unrecognized_gk_qualifier_drops_to_non_action(self):
        # Defensive: a Play row with a non-empty but unknown GK qualifier
        # remains conservatively non_action (not silently mapped to pass).
        df = _df_play_default()
        df["play_goal_keeper_action"] = ["someUnknownValue"]
        actions, _ = sportec_mod.convert_to_actions(df, home_team_id="T-HOME")
        assert len(actions) == 0

    def test_play_with_empty_gk_qualifier_falls_through_to_pass(self):
        # Empty string in the qualifier column ≡ no qualifier ≡ pass-class.
        df = _df_play_default()
        df["play_goal_keeper_action"] = [""]
        actions, _ = sportec_mod.convert_to_actions(df, home_team_id="T-HOME")
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["pass"]

    def test_legacy_Pass_event_type_no_longer_recognized(self):
        # Hyrum's Law check: zero current consumers passed event_type="Pass".
        # We assert it now drops to non_action (it's not in _MAPPED_EVENT_TYPES
        # if we remove "Pass" — but for safety we keep "Pass" in the mapped
        # set as a no-op, so this test checks dispatch shape only).
        # Actually: for the cleanest fix, we keep "Pass" in _MAPPED_EVENT_TYPES
        # and simply switch the dispatch to "Play". A "Pass" event would then
        # fall through to non_action. Test asserts that shape:
        events = pd.DataFrame(
            {
                "match_id": ["M1"],
                "event_id": ["e1"],
                "event_type": ["Pass"],
                "period": [1],
                "timestamp_seconds": [10.0],
                "player_id": ["P1"],
                "team": ["T-HOME"],
                "x": [50.0],
                "y": [34.0],
            }
        )
        actions, _ = sportec_mod.convert_to_actions(events, home_team_id="T-HOME")
        # event_type "Pass" no longer triggers the pass-class branch
        # post-1.10.0 (DFL never emits "Pass" — it was an implementation-time
        # mental model never reached by real data). Either drops to
        # non_action or produces zero rows depending on whether "Pass"
        # remains in _MAPPED_EVENT_TYPES.
        assert len(actions) == 0 or actions["type_id"].iloc[0] == spadlconfig.actiontype_id["non_action"]
```

- [ ] **Step 3.2: Update existing tests that assume `et == "Pass"` works**

Several existing tests in `test_sportec.py` use `event_type=["Pass"]` to test pass-class behaviour. These need to switch to `event_type=["Play"]` to remain valid:

Find and replace in `tests/spadl/test_sportec.py`:

| Function | Change |
|---|---|
| `_df_pass_default()` | `event_type` value `"Pass"` → `"Play"` |
| `_df_pass_cross()` | (inherits via `_df_pass_default`) |
| `_df_pass_flat_cross()` | (inherits via `_df_pass_default`) |
| `_df_pass_head()` | (inherits via `_df_pass_default`) |
| Inline `event_type` lists in `TestSportecDirectionOfPlay::test_away_team_x_flipped`, `TestSportecActionId::test_action_id_is_zero_indexed_range`, `TestSportecAddDribbles::test_dribble_inserted_between_distant_same_team_passes` | All `"Pass"` → `"Play"` |
| `_df_minimal_pass()` (used in `TestSportecContract` + `TestSportecPreserveNative`) | `event_type` value `"Pass"` → `"Play"` |

Use `Edit` with `replace_all=True` only AFTER confirming all instances are pass-class semantically (they are — the file has no `event_type="Pass"` outside the pass-class context, EXCEPT the new `test_legacy_Pass_event_type_no_longer_recognized` test which intentionally keeps `"Pass"`).

Concretely: edit `_df_minimal_pass` (around line 27-41 in current file), `_df_pass_default` (around 93-106), and the inline `event_type` lists in the 3 named test methods.

- [ ] **Step 3.3: Run the new + updated tests; verify they fail correctly**

Run:
```bash
uv run pytest tests/spadl/test_sportec.py::TestSportecPlayEventRecognition tests/spadl/test_sportec.py::TestSportecContract tests/spadl/test_sportec.py::TestSportecActionMappingPassAndSetPieces -v --tb=short
```

Expected: `TestSportecPlayEventRecognition` tests FAIL (because `is_pass = et == "Pass"` doesn't match `"Play"`). The pre-existing `TestSportecContract` / pass-mapping tests now also FAIL after the find-and-replace (they're checking `Play` events route to pass).

- [ ] **Step 3.4: Apply the Bug 1 fix in sportec.py**

Edit `silly_kicks/spadl/sportec.py:494-507` — replace the `# --- Pass / Cross detection ---` block AND the `# --- Play with goalkeeper_action ---` block (lines 583-596) with the restructured logic.

Replace:

```python
    # --- Pass / Cross detection ---
    is_pass = et == "Pass"
    is_cross_by_height = is_pass & _opt("play_height", None).eq("cross").to_numpy()
    is_cross_by_flag = is_pass & _opt("play_flat_cross", False).fillna(False).astype(bool).to_numpy()
    is_cross = is_cross_by_height | is_cross_by_flag
    is_pass_plain = is_pass & ~is_cross
    type_ids[is_pass_plain] = spadlconfig.actiontype_id["pass"]
    type_ids[is_cross] = spadlconfig.actiontype_id["cross"]
    result_ids[is_pass] = spadlconfig.result_id["success"]

    # Pass head bodypart
    is_head = is_pass & _opt("play_height", None).eq("head").to_numpy()
    bodypart_ids[is_head] = spadlconfig.bodypart_id["head"]
```

With:

```python
    # --- DFL "Play" events: pass-class by default; GK action when qualifier present ---
    # DFL bronze never emits "Pass" — the event_type for pass-class events is "Play".
    # Pre-1.10.0 the dispatch checked et == "Pass", silently dropping every IDSSE
    # pass to non_action. Post-1.10.0: Play events without a recognized
    # ``play_goal_keeper_action`` qualifier are pass-class; with a recognized
    # qualifier they are GK actions (handled in the GK block below).
    is_play = et == "Play"
    play_gk = _opt("play_goal_keeper_action", "").fillna("").astype(str).to_numpy()
    play_gk_save = is_play & (play_gk == "save")
    play_gk_claim = is_play & (play_gk == "claim")
    play_gk_punch = is_play & (play_gk == "punch")
    play_gk_pickup = is_play & (play_gk == "pickUp")
    is_play_known_gk = play_gk_save | play_gk_claim | play_gk_punch | play_gk_pickup

    # Pass-class: Play with empty / no qualifier (recognized-GK-qualifier rows
    # get keeper_* assignment in the GK block; unrecognized non-empty
    # qualifier rows fall to non_action below).
    is_pass = is_play & (play_gk == "") & ~is_play_known_gk
    is_cross_by_height = is_pass & _opt("play_height", None).eq("cross").to_numpy()
    is_cross_by_flag = is_pass & _opt("play_flat_cross", False).fillna(False).astype(bool).to_numpy()
    is_cross = is_cross_by_height | is_cross_by_flag
    is_pass_plain = is_pass & ~is_cross
    type_ids[is_pass_plain] = spadlconfig.actiontype_id["pass"]
    type_ids[is_cross] = spadlconfig.actiontype_id["cross"]
    result_ids[is_pass] = spadlconfig.result_id["success"]

    # Pass head bodypart
    is_head = is_pass & _opt("play_height", None).eq("head").to_numpy()
    bodypart_ids[is_head] = spadlconfig.bodypart_id["head"]
```

Then DELETE the now-redundant `# --- Play with goalkeeper_action ---` block at lines 582-596 and replace it with:

```python
    # --- GK action assignment (Play events with recognized qualifier) ---
    # is_play / play_gk / play_gk_* masks are all computed above in the
    # consolidated Play dispatch block.
    type_ids[play_gk_save] = spadlconfig.actiontype_id["keeper_save"]
    type_ids[play_gk_claim] = spadlconfig.actiontype_id["keeper_claim"]
    type_ids[play_gk_punch] = spadlconfig.actiontype_id["keeper_punch"]
    type_ids[play_gk_pickup] = spadlconfig.actiontype_id["keeper_pick_up"]
    result_ids[is_play_known_gk] = spadlconfig.result_id["success"]

    # Play events with UNRECOGNIZED non-empty qualifier value → non_action
    # (defensive: avoids silently emitting pass for future qualifier values
    # we haven't analyzed). Empty-qualifier rows are pass-class above.
    is_play_unrecognized_gk = is_play & (play_gk != "") & ~is_play_known_gk
    type_ids[is_play_unrecognized_gk] = spadlconfig.actiontype_id["non_action"]
```

- [ ] **Step 3.5: Run the failing tests; verify they now PASS**

Run:
```bash
uv run pytest tests/spadl/test_sportec.py -v --tb=short -q 2>&1 | tail -40
```

Expected: all sportec tests pass (including new `TestSportecPlayEventRecognition` and updated existing tests).

- [ ] **Step 3.6: Run the full test suite to confirm no regressions elsewhere**

Run:
```bash
uv run pytest tests/ -m "not e2e" -v --tb=short -q 2>&1 | tail -10
```

Expected: ~570+ passed (560 baseline + ~14 new coverage_metrics + ~7 new sportec tests). Zero failures. The cross-path consistency test in `TestSportecCrossPathConsistency` should still pass (kloppy bridge test data uses `event_type="Play"` for CARRY/GOALKEEPER/etc. — already correct).

- [ ] **Step 3.7: Stage Bug 1 changes (DO NOT commit)**

Run:
```bash
git add silly_kicks/spadl/sportec.py tests/spadl/test_sportec.py
git status
```

Expected: 2 files modified, staged.

---

## Task 4: sportec.py Bug 2 — `throwOut` / `punt` 2-action synthesis (TDD)

**Files:**
- Modify: `silly_kicks/spadl/sportec.py` (extend the GK qualifier vocabulary; add `_synthesize_gk_distribution_actions` helper; integrate synthesis into `_build_raw_actions`)
- Modify: `tests/spadl/test_sportec.py` (add `TestSportecGKQualifierSynthesis`)

**Behaviour:** A Play event with `play_goal_keeper_action="throwOut"` or `"punt"` produces TWO SPADL actions:

| Source qualifier | Action 1 | Action 2 |
|---|---|---|
| `throwOut` | `keeper_pick_up` (bodypart=other) | `pass` (bodypart=other) |
| `punt` | `keeper_pick_up` (bodypart=other) | `goalkick` (bodypart=foot) |

Both actions inherit the source row's `(player_id, team_id, period, time_seconds, x, y)`. The synthetic action is inserted IMMEDIATELY AFTER the source row in output ordering. `action_id` is renumbered to dense `range(len)` after insertion. `preserve_native` columns propagate to BOTH rows.

- [ ] **Step 4.1: Write failing tests for throwOut/punt synthesis**

Append to `tests/spadl/test_sportec.py`:

```python
# ---------------------------------------------------------------------------
# Bug 2 — DFL play_goal_keeper_action throwOut / punt → 2-action synthesis
# (1.10.0; supersedes pre-1.10.0 dispatch where throwOut + punt were silent
# non_action drops despite being legitimate GK distribution events)
# ---------------------------------------------------------------------------


def _df_play_gk(qualifier: str, **overrides) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "match_id": ["M1"],
            "event_id": ["e1"],
            "event_type": ["Play"],
            "period": [1],
            "timestamp_seconds": [60.0],
            "player_id": ["P-GK"],
            "team": ["T-AWAY"],
            "x": [3.0],
            "y": [34.0],
            "play_goal_keeper_action": [qualifier],
        }
    )
    for k, v in overrides.items():
        base[k] = [v]
    return base


class TestSportecGKQualifierSynthesis:
    """DFL distribution qualifiers (throwOut, punt) emit TWO SPADL actions:
    keeper_pick_up + pass for throwOut, keeper_pick_up + goalkick for punt.
    Each action inherits the source's (player_id, team, period, time, x, y).
    """

    def test_throwout_synthesizes_two_actions(self):
        actions, _ = sportec_mod.convert_to_actions(_df_play_gk("throwOut"), home_team_id="T-HOME")
        assert len(actions) == 2

    def test_throwout_first_action_is_keeper_pick_up(self):
        actions, _ = sportec_mod.convert_to_actions(_df_play_gk("throwOut"), home_team_id="T-HOME")
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["keeper_pick_up"]

    def test_throwout_second_action_is_pass_with_other_bodypart(self):
        actions, _ = sportec_mod.convert_to_actions(_df_play_gk("throwOut"), home_team_id="T-HOME")
        assert actions["type_id"].iloc[1] == spadlconfig.actiontype_id["pass"]
        assert actions["bodypart_id"].iloc[1] == spadlconfig.bodypart_id["other"]

    def test_punt_first_action_is_keeper_pick_up(self):
        actions, _ = sportec_mod.convert_to_actions(_df_play_gk("punt"), home_team_id="T-HOME")
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["keeper_pick_up"]

    def test_punt_second_action_is_goalkick_with_foot_bodypart(self):
        actions, _ = sportec_mod.convert_to_actions(_df_play_gk("punt"), home_team_id="T-HOME")
        assert actions["type_id"].iloc[1] == spadlconfig.actiontype_id["goalkick"]
        assert actions["bodypart_id"].iloc[1] == spadlconfig.bodypart_id["foot"]

    def test_both_synthesized_actions_share_source_player_team_time(self):
        df = _df_play_gk("throwOut")
        actions, _ = sportec_mod.convert_to_actions(df, home_team_id="T-HOME")
        assert actions["player_id"].iloc[0] == actions["player_id"].iloc[1] == "P-GK"
        # Note: away-team team_id is unchanged (team_id is the team string,
        # not flipped by direction-of-play).
        assert actions["team_id"].iloc[0] == actions["team_id"].iloc[1] == "T-AWAY"
        assert actions["period_id"].iloc[0] == actions["period_id"].iloc[1] == 1
        assert actions["time_seconds"].iloc[0] == actions["time_seconds"].iloc[1] == 60.0

    def test_action_ids_renumbered_dense_zero_indexed(self):
        # Single throwOut → 2 synthesized rows → action_id [0, 1].
        actions, _ = sportec_mod.convert_to_actions(_df_play_gk("throwOut"), home_team_id="T-HOME")
        assert actions["action_id"].tolist() == [0, 1]

    def test_multiple_throwouts_all_synthesized(self):
        # Three throwOut events → six SPADL actions (3 keeper_pick_up + 3 pass).
        events = pd.DataFrame(
            {
                "match_id": ["M1"] * 3,
                "event_id": ["e1", "e2", "e3"],
                "event_type": ["Play"] * 3,
                "period": [1, 1, 1],
                "timestamp_seconds": [10.0, 20.0, 30.0],
                "player_id": ["P-GK"] * 3,
                "team": ["T-AWAY"] * 3,
                "x": [3.0, 3.0, 3.0],
                "y": [34.0, 34.0, 34.0],
                "play_goal_keeper_action": ["throwOut"] * 3,
            }
        )
        actions, _ = sportec_mod.convert_to_actions(events, home_team_id="T-HOME")
        assert len(actions) == 6
        kp_count = (actions["type_id"] == spadlconfig.actiontype_id["keeper_pick_up"]).sum()
        pass_count = (actions["type_id"] == spadlconfig.actiontype_id["pass"]).sum()
        assert kp_count == 3
        assert pass_count == 3

    def test_mixed_distribution_and_shot_stopping_qualifiers(self):
        events = pd.DataFrame(
            {
                "match_id": ["M1"] * 3,
                "event_id": ["e1", "e2", "e3"],
                "event_type": ["Play"] * 3,
                "period": [1, 1, 1],
                "timestamp_seconds": [10.0, 20.0, 30.0],
                "player_id": ["P-GK"] * 3,
                "team": ["T-AWAY"] * 3,
                "x": [3.0] * 3,
                "y": [34.0] * 3,
                "play_goal_keeper_action": ["save", "throwOut", "punt"],
            }
        )
        actions, _ = sportec_mod.convert_to_actions(events, home_team_id="T-HOME")
        # save → 1 keeper_save; throwOut → 2 (keeper_pick_up + pass); punt → 2 (keeper_pick_up + goalkick)
        assert len(actions) == 5
        type_set = list(actions["type_id"])
        assert type_set.count(spadlconfig.actiontype_id["keeper_save"]) == 1
        assert type_set.count(spadlconfig.actiontype_id["keeper_pick_up"]) == 2
        assert type_set.count(spadlconfig.actiontype_id["pass"]) == 1
        assert type_set.count(spadlconfig.actiontype_id["goalkick"]) == 1

    def test_preserve_native_propagates_to_both_synthetic_actions(self):
        df = _df_play_gk("throwOut")
        df["my_extra"] = ["custom_val"]
        actions, _ = sportec_mod.convert_to_actions(df, home_team_id="T-HOME", preserve_native=["my_extra"])
        # Both rows should carry the source's preserved column value.
        assert actions["my_extra"].iloc[0] == "custom_val"
        assert actions["my_extra"].iloc[1] == "custom_val"
```

- [ ] **Step 4.2: Run new tests, verify they fail**

Run:
```bash
uv run pytest tests/spadl/test_sportec.py::TestSportecGKQualifierSynthesis -v --tb=short
```

Expected: all ~10 tests fail (throwOut/punt currently route to `is_play_unrecognized_gk` → non_action after Task 3).

- [ ] **Step 4.3: Add the synthesis helper to sportec.py**

Add a new helper `_synthesize_gk_distribution_actions` near the bottom of `silly_kicks/spadl/sportec.py` (after `_build_raw_actions`):

```python
def _synthesize_gk_distribution_actions(
    rows: pd.DataFrame,
    play_gk_throwout: np.ndarray,
    play_gk_punt: np.ndarray,
    play_gk_supplementary: np.ndarray,
    preserve_native: list[str] | None,
) -> pd.DataFrame:
    """Build the synthetic second-action DataFrame for GK distribution events.

    For each row in ``rows`` matching one of the distribution masks, emit a
    synthetic SPADL action immediately after the corresponding keeper_pick_up
    row (which is assigned in the main ``_build_raw_actions`` dispatch).

    Routing:

    - ``throwOut`` → ``pass`` (bodypart=other; GK distributed by hand)
    - ``punt`` → ``goalkick`` (bodypart=foot; GK distributed by foot)
    - ``goalkeeper_ids`` supplementary path (no native qualifier present
       but player_id is a known GK): → ``pass`` (bodypart=other; default
       to throwOut shape — consumers wanting punt-shape pass through the
       qualifier explicitly)

    All synthetic rows inherit the source row's ``(match_id, event_id,
    period, timestamp_seconds, player_id, team, x, y)``. The
    ``original_event_id`` of the synthetic row is the source's
    ``event_id`` suffixed with ``"_synth_pass"`` / ``"_synth_goalkick"``
    to keep IDs unique.

    Returned DataFrame has the same columns as the main raw actions
    DataFrame in ``_build_raw_actions``, plus a ``_row_order`` integer
    column used by the caller to interleave synthetic rows after their
    source rows.

    Parameters
    ----------
    rows : pd.DataFrame
        The post-mask Play-eligible source DataFrame.
    play_gk_throwout, play_gk_punt, play_gk_supplementary : np.ndarray of bool
        Aligned with ``rows`` (length n). Must be mutually exclusive (the
        caller ensures this — supplementary fires only when neither qualifier is set).
    preserve_native : list[str] or None
        Caller-attached columns to copy through.

    Returns
    -------
    pd.DataFrame
        Synthetic actions with ``_row_order`` column.
    """
    is_distribution = play_gk_throwout | play_gk_punt | play_gk_supplementary
    if not is_distribution.any():
        return pd.DataFrame()

    src_indices = np.where(is_distribution)[0]
    src = rows.iloc[src_indices].copy().reset_index(drop=True)

    n_synth = len(src)
    type_ids_synth = np.full(n_synth, spadlconfig.actiontype_id["pass"], dtype=np.int64)
    bodypart_ids_synth = np.full(n_synth, spadlconfig.bodypart_id["other"], dtype=np.int64)
    suffix = np.full(n_synth, "_synth_pass", dtype=object)

    # punt → goalkick + foot
    is_punt_synth = play_gk_punt[src_indices]
    type_ids_synth[is_punt_synth] = spadlconfig.actiontype_id["goalkick"]
    bodypart_ids_synth[is_punt_synth] = spadlconfig.bodypart_id["foot"]
    suffix[is_punt_synth] = "_synth_goalkick"

    synth = pd.DataFrame(
        {
            "game_id": src["match_id"].astype("object"),
            "original_event_id": (src["event_id"].astype(str) + pd.Series(suffix, index=src.index)).astype("object"),
            "period_id": src["period"].astype(np.int64),
            "time_seconds": src["timestamp_seconds"].astype(np.float64),
            "team_id": src["team"].astype("object"),
            "player_id": src["player_id"].astype("object"),
            "start_x": src["x"].astype(np.float64),
            "start_y": src["y"].astype(np.float64),
            "end_x": src["x"].astype(np.float64),
            "end_y": src["y"].astype(np.float64),
            "type_id": type_ids_synth,
            "result_id": np.full(n_synth, spadlconfig.result_id["success"], dtype=np.int64),
            "bodypart_id": bodypart_ids_synth,
        }
    )

    # _row_order: sort key for interleaving. Source rows get index*2; synth
    # rows get the source's index*2 + 1, so synth sorts immediately after.
    synth["_row_order"] = src_indices.astype(np.int64) * 2 + 1

    if preserve_native:
        for col in preserve_native:
            if col in src.columns:
                synth[col] = src[col].to_numpy()
            else:
                synth[col] = np.nan

    return synth
```

- [ ] **Step 4.4: Integrate synthesis into `_build_raw_actions`**

Modify `_build_raw_actions` in `silly_kicks/spadl/sportec.py`. The integration changes are:

1. Add throwOut + punt qualifier masks alongside save/claim/punch/pickUp.
2. Map `play_gk_throwout | play_gk_punt` to `keeper_pick_up` (the first half of the 2-action synthesis).
3. After building the main `actions` DataFrame, build synthetic rows and interleave.

Replace the GK action assignment block (the one we just added in Task 3.4) with:

```python
    # --- GK action assignment (Play events with recognized qualifier) ---
    play_gk_throwout = is_play & (play_gk == "throwOut")
    play_gk_punt = is_play & (play_gk == "punt")
    play_gk_distribution = play_gk_throwout | play_gk_punt
    is_play_known_gk_full = is_play_known_gk | play_gk_distribution

    type_ids[play_gk_save] = spadlconfig.actiontype_id["keeper_save"]
    type_ids[play_gk_claim] = spadlconfig.actiontype_id["keeper_claim"]
    type_ids[play_gk_punch] = spadlconfig.actiontype_id["keeper_punch"]
    type_ids[play_gk_pickup] = spadlconfig.actiontype_id["keeper_pick_up"]
    type_ids[play_gk_distribution] = spadlconfig.actiontype_id["keeper_pick_up"]
    bodypart_ids[is_play_known_gk_full] = spadlconfig.bodypart_id["other"]
    result_ids[is_play_known_gk_full] = spadlconfig.result_id["success"]

    # Play events with UNRECOGNIZED non-empty qualifier → non_action.
    is_play_unrecognized_gk = is_play & (play_gk != "") & ~is_play_known_gk_full
    type_ids[is_play_unrecognized_gk] = spadlconfig.actiontype_id["non_action"]
```

Then replace the lines that build the actions DataFrame and apply non_action filtering. Find the existing block (around lines 599-633 in the current file — which after our edits becomes lines after the GK assignment block):

Replace:

```python
    # Assemble actions DataFrame.
    actions = pd.DataFrame(
        {
            "game_id": rows["match_id"].astype("object"),
            "original_event_id": rows["event_id"].astype("object"),
            "period_id": rows["period"].astype(np.int64),
            "time_seconds": rows["timestamp_seconds"].astype(np.float64),
            "team_id": rows["team"].astype("object"),
            "player_id": rows["player_id"].astype("object"),
            "start_x": rows["x"].astype(np.float64),
            "start_y": rows["y"].astype(np.float64),
            "end_x": rows["x"].astype(np.float64),
            "end_y": rows["y"].astype(np.float64),
            "type_id": type_ids,
            "result_id": result_ids,
            "bodypart_id": bodypart_ids,
        }
    )

    # Drop non_action rows.
    actions = actions[actions["type_id"] != spadlconfig.actiontype_id["non_action"]].reset_index(drop=True)

    # If post-filter all rows dropped, return canonical empty schema so downstream
    # _finalize_output can assemble action_id and other schema-required columns.
    if len(actions) == 0:
        return _empty_raw_actions(preserve_native, events)

    # Carry preserve_native columns through.
    if preserve_native:
        for col in preserve_native:
            if col in rows.columns:
                actions[col] = rows.loc[actions.index, col].to_numpy()
            else:
                actions[col] = np.nan

    return actions
```

With:

```python
    # Assemble actions DataFrame (1:1 with rows).
    actions = pd.DataFrame(
        {
            "game_id": rows["match_id"].astype("object"),
            "original_event_id": rows["event_id"].astype("object"),
            "period_id": rows["period"].astype(np.int64),
            "time_seconds": rows["timestamp_seconds"].astype(np.float64),
            "team_id": rows["team"].astype("object"),
            "player_id": rows["player_id"].astype("object"),
            "start_x": rows["x"].astype(np.float64),
            "start_y": rows["y"].astype(np.float64),
            "end_x": rows["x"].astype(np.float64),
            "end_y": rows["y"].astype(np.float64),
            "type_id": type_ids,
            "result_id": result_ids,
            "bodypart_id": bodypart_ids,
        }
    )

    # Carry preserve_native columns onto the main DataFrame BEFORE synthesis
    # so synthetic rows can pick them up via the _synthesize helper.
    if preserve_native:
        for col in preserve_native:
            if col in rows.columns:
                actions[col] = rows[col].to_numpy()
            else:
                actions[col] = np.nan

    # Synthesize keeper-distribution second-action rows (throwOut / punt /
    # goalkeeper_ids supplementary path). The supplementary mask is computed
    # in Task 5; here we pass an all-False sentinel.
    play_gk_supplementary = np.zeros(n, dtype=bool)
    synth = _synthesize_gk_distribution_actions(
        rows,
        play_gk_throwout=play_gk_throwout,
        play_gk_punt=play_gk_punt,
        play_gk_supplementary=play_gk_supplementary,
        preserve_native=preserve_native,
    )

    if len(synth) > 0:
        actions["_row_order"] = np.arange(len(actions), dtype=np.int64) * 2
        combined = pd.concat([actions, synth], ignore_index=True, sort=False)
        combined = combined.sort_values("_row_order", kind="mergesort").drop(columns=["_row_order"]).reset_index(drop=True)
        actions = combined

    # Drop non_action rows.
    actions = actions[actions["type_id"] != spadlconfig.actiontype_id["non_action"]].reset_index(drop=True)

    if len(actions) == 0:
        return _empty_raw_actions(preserve_native, events)

    return actions
```

- [ ] **Step 4.5: Run synthesis tests, verify they pass**

Run:
```bash
uv run pytest tests/spadl/test_sportec.py::TestSportecGKQualifierSynthesis -v --tb=short
```

Expected: all ~10 tests pass.

- [ ] **Step 4.6: Run full sportec test suite, confirm no regressions**

Run:
```bash
uv run pytest tests/spadl/test_sportec.py -v --tb=short -q 2>&1 | tail -20
```

Expected: all sportec tests pass.

- [ ] **Step 4.7: Stage changes (DO NOT commit)**

```bash
git add silly_kicks/spadl/sportec.py tests/spadl/test_sportec.py
git status
```

---

## Task 5: sportec.py — `goalkeeper_ids` supplementary signal (TDD)

**Files:**
- Modify: `silly_kicks/spadl/sportec.py` (add `goalkeeper_ids: set[str] | None = None` parameter; route in `_build_raw_actions`)
- Modify: `tests/spadl/test_sportec.py` (add `TestSportecGoalkeeperIdsSupplementary`)

**Behaviour:** When `goalkeeper_ids` is provided AND a Play event has `player_id ∈ goalkeeper_ids` AND has NO explicit `play_goal_keeper_action` qualifier (empty/null) → route to keeper_pick_up + synthesized pass (matching throwOut shape: bodypart=other for the synthesized pass).

This catches GK distribution that DFL didn't annotate with a qualifier — a supplementary signal. The qualifier-driven mapping remains the primary contract.

- [ ] **Step 5.1: Write failing tests**

Append to `tests/spadl/test_sportec.py`:

```python
# ---------------------------------------------------------------------------
# goalkeeper_ids supplementary signal — fires only when a Play event has
# player_id in the supplied set AND no native GK qualifier
# ---------------------------------------------------------------------------


class TestSportecGoalkeeperIdsSupplementary:
    """When DFL doesn't annotate a Play event with play_goal_keeper_action
    but the player_id is known to be a goalkeeper, route the event to the
    keeper_pick_up + pass synthesis (treats it as a throwOut equivalent).
    """

    def test_no_goalkeeper_ids_keeps_default_pass_class_behavior(self):
        # Without goalkeeper_ids, a Play event with no qualifier is pass-class.
        df = _df_play_default()
        actions, _ = sportec_mod.convert_to_actions(df, home_team_id="T-HOME")
        assert len(actions) == 1
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["pass"]

    def test_player_in_goalkeeper_ids_with_no_qualifier_synthesizes(self):
        # Same Play event, but player_id is in goalkeeper_ids → synth.
        df = _df_play_default()
        df["player_id"] = ["P-GK"]
        actions, _ = sportec_mod.convert_to_actions(df, home_team_id="T-HOME", goalkeeper_ids={"P-GK"})
        assert len(actions) == 2
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["keeper_pick_up"]
        assert actions["type_id"].iloc[1] == spadlconfig.actiontype_id["pass"]

    def test_player_not_in_goalkeeper_ids_unchanged(self):
        df = _df_play_default()
        df["player_id"] = ["P-NOT-GK"]
        actions, _ = sportec_mod.convert_to_actions(df, home_team_id="T-HOME", goalkeeper_ids={"P-GK"})
        assert len(actions) == 1
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["pass"]

    def test_player_in_goalkeeper_ids_with_throwout_qualifier_uses_qualifier_path(self):
        # Both signals present — qualifier path wins (it's the primary contract).
        # Net result is identical to the qualifier-only path: 2 actions.
        df = _df_play_gk("throwOut")
        actions, _ = sportec_mod.convert_to_actions(df, home_team_id="T-HOME", goalkeeper_ids={"P-GK"})
        assert len(actions) == 2
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["keeper_pick_up"]
        assert actions["type_id"].iloc[1] == spadlconfig.actiontype_id["pass"]

    def test_player_in_goalkeeper_ids_with_save_qualifier_keeps_save(self):
        # Recognized non-distribution qualifier → keeper_save (single action),
        # supplementary path does not fire.
        df = _df_play_gk("save")
        actions, _ = sportec_mod.convert_to_actions(df, home_team_id="T-HOME", goalkeeper_ids={"P-GK"})
        assert len(actions) == 1
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["keeper_save"]

    def test_empty_goalkeeper_ids_set_equivalent_to_none(self):
        df = _df_play_default()
        actions_empty, _ = sportec_mod.convert_to_actions(df, home_team_id="T-HOME", goalkeeper_ids=set())
        actions_none, _ = sportec_mod.convert_to_actions(df, home_team_id="T-HOME", goalkeeper_ids=None)
        # Both should produce identical output (1 pass action).
        assert len(actions_empty) == len(actions_none) == 1
        assert actions_empty["type_id"].iloc[0] == actions_none["type_id"].iloc[0]
```

- [ ] **Step 5.2: Run failing tests**

Run:
```bash
uv run pytest tests/spadl/test_sportec.py::TestSportecGoalkeeperIdsSupplementary -v --tb=short
```

Expected: tests fail because `goalkeeper_ids` parameter doesn't exist yet on sportec's `convert_to_actions` (TypeError on unexpected keyword argument).

- [ ] **Step 5.3: Add `goalkeeper_ids` parameter to sportec's `convert_to_actions`**

Edit `silly_kicks/spadl/sportec.py`:

Find the signature:

```python
def convert_to_actions(
    events: pd.DataFrame,
    home_team_id: str,
    *,
    preserve_native: list[str] | None = None,
) -> tuple[pd.DataFrame, ConversionReport]:
```

Replace with:

```python
def convert_to_actions(
    events: pd.DataFrame,
    home_team_id: str,
    *,
    preserve_native: list[str] | None = None,
    goalkeeper_ids: set[str] | None = None,
) -> tuple[pd.DataFrame, ConversionReport]:
```

Update the docstring (Parameters section) to add:

```python
    goalkeeper_ids : set[str] or None, default ``None``
        Optional set of DFL player_ids known to be goalkeepers. When
        provided, Play events whose ``player_id`` is in this set and which
        have NO explicit ``play_goal_keeper_action`` qualifier are routed
        to the keeper_pick_up + pass 2-action synthesis (matching the
        ``throwOut`` qualifier shape). Use this to surface GK distribution
        events that DFL didn't natively annotate. The qualifier-driven
        mapping (save / claim / punch / pickUp / throwOut / punt) remains
        the primary contract and takes precedence when both signals are
        present. An empty set is equivalent to ``None``.
```

- [ ] **Step 5.4: Thread `goalkeeper_ids` through to `_build_raw_actions`**

Find the call site in `convert_to_actions`:

```python
    raw_actions = _build_raw_actions(events, preserve_native)
```

Replace with:

```python
    raw_actions = _build_raw_actions(events, preserve_native, goalkeeper_ids=goalkeeper_ids)
```

Update `_build_raw_actions` signature:

```python
def _build_raw_actions(
    events: pd.DataFrame,
    preserve_native: list[str] | None,
    *,
    goalkeeper_ids: set[str] | None = None,
) -> pd.DataFrame:
```

In `_build_raw_actions`, just before the GK action assignment block (where `play_gk_throwout` and `play_gk_punt` are defined), compute the supplementary mask AFTER the existing GK qualifier masks:

Replace:

```python
    # --- GK action assignment (Play events with recognized qualifier) ---
    play_gk_throwout = is_play & (play_gk == "throwOut")
    play_gk_punt = is_play & (play_gk == "punt")
    play_gk_distribution = play_gk_throwout | play_gk_punt
    is_play_known_gk_full = is_play_known_gk | play_gk_distribution
```

With:

```python
    # --- GK action assignment (Play events with recognized qualifier) ---
    play_gk_throwout = is_play & (play_gk == "throwOut")
    play_gk_punt = is_play & (play_gk == "punt")
    play_gk_distribution = play_gk_throwout | play_gk_punt
    is_play_known_gk_full = is_play_known_gk | play_gk_distribution

    # Supplementary signal: when goalkeeper_ids is provided, Play events
    # with player_id in the set AND no native GK qualifier route to the
    # keeper_pick_up + pass synthesis. The qualifier-driven path takes
    # precedence (no overlap by construction — supplementary fires only
    # when play_gk == "").
    play_gk_supplementary = np.zeros(n, dtype=bool)
    if goalkeeper_ids:
        gk_set = set(goalkeeper_ids)
        is_known_gk_player = rows["player_id"].astype(str).isin({str(g) for g in gk_set}).to_numpy()
        play_gk_supplementary = is_play & (play_gk == "") & is_known_gk_player

    is_play_known_gk_full = is_play_known_gk_full | play_gk_supplementary
```

Update the `type_ids[play_gk_distribution] = ...` line to include the supplementary mask. Find:

```python
    type_ids[play_gk_distribution] = spadlconfig.actiontype_id["keeper_pick_up"]
    bodypart_ids[is_play_known_gk_full] = spadlconfig.bodypart_id["other"]
    result_ids[is_play_known_gk_full] = spadlconfig.result_id["success"]
```

Replace with:

```python
    type_ids[play_gk_distribution] = spadlconfig.actiontype_id["keeper_pick_up"]
    type_ids[play_gk_supplementary] = spadlconfig.actiontype_id["keeper_pick_up"]
    bodypart_ids[is_play_known_gk_full] = spadlconfig.bodypart_id["other"]
    result_ids[is_play_known_gk_full] = spadlconfig.result_id["success"]
```

Also: Task 3's pass-class mask `is_pass = is_play & (play_gk == "") & ~is_play_known_gk` needs updating to exclude supplementary rows (they're no longer pass-class):

Find:

```python
    # Pass-class: Play with empty / no qualifier (recognized-GK-qualifier rows
    # get keeper_* assignment in the GK block; unrecognized non-empty
    # qualifier rows fall to non_action below).
    is_pass = is_play & (play_gk == "") & ~is_play_known_gk
```

Note that `play_gk_supplementary` isn't computed yet at this point — it's computed later. We need to either move the supplementary computation up, or recompute the pass mask down. Cleanest: move supplementary computation BEFORE the pass-class block.

Actual final structure (replacing the entire dispatch block we built in Task 3.4 + Task 4.4):

```python
    # --- DFL "Play" events: pass-class by default; GK action when qualifier present ---
    is_play = et == "Play"
    play_gk = _opt("play_goal_keeper_action", "").fillna("").astype(str).to_numpy()

    # GK qualifier masks (canonical DFL vocabulary)
    play_gk_save = is_play & (play_gk == "save")
    play_gk_claim = is_play & (play_gk == "claim")
    play_gk_punch = is_play & (play_gk == "punch")
    play_gk_pickup = is_play & (play_gk == "pickUp")
    play_gk_throwout = is_play & (play_gk == "throwOut")
    play_gk_punt = is_play & (play_gk == "punt")
    play_gk_distribution = play_gk_throwout | play_gk_punt
    is_play_known_gk = play_gk_save | play_gk_claim | play_gk_punch | play_gk_pickup
    is_play_known_qualifier = is_play_known_gk | play_gk_distribution

    # Supplementary signal: known GK player + no native qualifier → synth.
    play_gk_supplementary = np.zeros(n, dtype=bool)
    if goalkeeper_ids:
        gk_str_set = {str(g) for g in goalkeeper_ids}
        is_known_gk_player = rows["player_id"].astype(str).isin(gk_str_set).to_numpy()
        play_gk_supplementary = is_play & (play_gk == "") & is_known_gk_player

    is_play_gk_any = is_play_known_qualifier | play_gk_supplementary

    # Pass-class: Play with empty qualifier AND not in supplementary path.
    is_pass = is_play & (play_gk == "") & ~play_gk_supplementary
    is_cross_by_height = is_pass & _opt("play_height", None).eq("cross").to_numpy()
    is_cross_by_flag = is_pass & _opt("play_flat_cross", False).fillna(False).astype(bool).to_numpy()
    is_cross = is_cross_by_height | is_cross_by_flag
    is_pass_plain = is_pass & ~is_cross
    type_ids[is_pass_plain] = spadlconfig.actiontype_id["pass"]
    type_ids[is_cross] = spadlconfig.actiontype_id["cross"]
    result_ids[is_pass] = spadlconfig.result_id["success"]
    is_head = is_pass & _opt("play_height", None).eq("head").to_numpy()
    bodypart_ids[is_head] = spadlconfig.bodypart_id["head"]
```

And the GK assignment block:

```python
    # --- GK action assignment ---
    type_ids[play_gk_save] = spadlconfig.actiontype_id["keeper_save"]
    type_ids[play_gk_claim] = spadlconfig.actiontype_id["keeper_claim"]
    type_ids[play_gk_punch] = spadlconfig.actiontype_id["keeper_punch"]
    type_ids[play_gk_pickup] = spadlconfig.actiontype_id["keeper_pick_up"]
    type_ids[play_gk_distribution] = spadlconfig.actiontype_id["keeper_pick_up"]
    type_ids[play_gk_supplementary] = spadlconfig.actiontype_id["keeper_pick_up"]
    bodypart_ids[is_play_gk_any] = spadlconfig.bodypart_id["other"]
    result_ids[is_play_gk_any] = spadlconfig.result_id["success"]

    # Play events with UNRECOGNIZED non-empty qualifier → non_action.
    is_play_unrecognized_gk = is_play & (play_gk != "") & ~is_play_known_qualifier
    type_ids[is_play_unrecognized_gk] = spadlconfig.actiontype_id["non_action"]
```

And update the synth call site to pass the supplementary mask:

```python
    synth = _synthesize_gk_distribution_actions(
        rows,
        play_gk_throwout=play_gk_throwout,
        play_gk_punt=play_gk_punt,
        play_gk_supplementary=play_gk_supplementary,
        preserve_native=preserve_native,
    )
```

(The supplementary path was already passed as zeros in Task 4.4; now it carries the real mask.)

- [ ] **Step 5.5: Run supplementary tests, verify pass**

Run:
```bash
uv run pytest tests/spadl/test_sportec.py::TestSportecGoalkeeperIdsSupplementary -v --tb=short
```

Expected: all ~6 tests pass.

- [ ] **Step 5.6: Run full sportec test suite, confirm no regressions**

Run:
```bash
uv run pytest tests/spadl/test_sportec.py -v --tb=short -q 2>&1 | tail -20
```

Expected: all sportec tests pass.

- [ ] **Step 5.7: Stage (DO NOT commit)**

```bash
git add silly_kicks/spadl/sportec.py tests/spadl/test_sportec.py
```

---

## Task 6: metrica.py Bug 3 — `goalkeeper_ids` primary mechanism (TDD)

**Files:**
- Modify: `silly_kicks/spadl/metrica.py` (add `goalkeeper_ids: set[str] | None = None`; conservative GK routing)
- Modify: `tests/spadl/test_metrica.py` (add `TestMetricaGoalkeeperIdsRouting`)

**Behaviour:** Metrica's source format genuinely lacks GK markers anywhere in its event taxonomy. Without `goalkeeper_ids`, the converter emits zero `keeper_*` actions (preserves 1.9.0 default behaviour — no breaking change). With `goalkeeper_ids`:

| Source event | When player_id ∈ goalkeeper_ids | Without goalkeeper_ids |
|---|---|---|
| `PASS` (any subtype) | synthesize **keeper_pick_up + pass** (similar to sportec throwOut: GK had ball, distributed) | unchanged (pass / cross / goalkick / head bodypart per existing dispatch) |
| `RECOVERY` (any subtype) | **keeper_pick_up** | unchanged (interception) |
| `CHALLENGE` with subtype `"AERIAL-WON"` | **keeper_claim** | unchanged (tackle if WON) |
| `CHALLENGE` with subtype `"AERIAL-LOST"` (or other AERIAL variants) | unchanged (default routing — GK lost the duel) | unchanged |
| All other event types by GK | unchanged | unchanged |

- [ ] **Step 6.1: Write failing tests**

Append to `tests/spadl/test_metrica.py`:

```python
# ---------------------------------------------------------------------------
# Bug 3 — Metrica lacks native GK markers; goalkeeper_ids enables coverage
# (1.10.0; supersedes pre-1.10.0 where Metrica emitted zero keeper_* actions
# and the lakehouse adapter had no way to surface GK actions)
# ---------------------------------------------------------------------------


def _df_metrica_pass_by_gk() -> pd.DataFrame:
    df = _df_minimal_pass()
    df["player"] = ["GK_HOME"]
    df["team"] = ["Home"]
    return df


def _df_metrica_recovery_by_gk() -> pd.DataFrame:
    df = _df_metrica("RECOVERY", None)
    df["player"] = ["GK_HOME"]
    return df


def _df_metrica_aerial_won_by_gk() -> pd.DataFrame:
    df = _df_metrica("CHALLENGE", "AERIAL-WON")
    df["player"] = ["GK_HOME"]
    return df


def _df_metrica_aerial_lost_by_gk() -> pd.DataFrame:
    df = _df_metrica("CHALLENGE", "AERIAL-LOST")
    df["player"] = ["GK_HOME"]
    return df


class TestMetricaGoalkeeperIdsRouting:
    """Without goalkeeper_ids: zero keeper_* actions (1.9.0 default preserved).
    With goalkeeper_ids: PASS by GK → synth, RECOVERY by GK → keeper_pick_up,
    CHALLENGE-AERIAL-WON by GK → keeper_claim. Other events unchanged.
    """

    def test_no_goalkeeper_ids_pass_remains_pass(self):
        actions, _ = metrica_mod.convert_to_actions(_df_metrica_pass_by_gk(), home_team_id="Home")
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["pass"]

    def test_no_goalkeeper_ids_recovery_remains_interception(self):
        actions, _ = metrica_mod.convert_to_actions(_df_metrica_recovery_by_gk(), home_team_id="Home")
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["interception"]

    def test_no_goalkeeper_ids_emits_zero_keeper_actions(self):
        # Bundle 4 events, all by GK_HOME. Without goalkeeper_ids,
        # output has zero keeper_* actions — preserves 1.9.0 behavior.
        events = pd.concat(
            [
                _df_metrica_pass_by_gk(),
                _df_metrica_recovery_by_gk(),
                _df_metrica_aerial_won_by_gk(),
                _df_metrica_aerial_lost_by_gk(),
            ],
            ignore_index=True,
        )
        # Make event_ids unique
        events["event_id"] = list(range(len(events)))
        events["start_time_s"] = [10.0, 20.0, 30.0, 40.0]
        events["end_time_s"] = [10.5, 20.5, 30.5, 40.5]
        actions, _ = metrica_mod.convert_to_actions(events, home_team_id="Home")
        keeper_ids = {
            spadlconfig.actiontype_id[t]
            for t in ("keeper_save", "keeper_claim", "keeper_punch", "keeper_pick_up")
        }
        keeper_count = actions["type_id"].isin(keeper_ids).sum()
        assert keeper_count == 0

    def test_pass_by_gk_synthesizes_two_actions(self):
        actions, _ = metrica_mod.convert_to_actions(
            _df_metrica_pass_by_gk(), home_team_id="Home", goalkeeper_ids={"GK_HOME"}
        )
        assert len(actions) == 2
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["keeper_pick_up"]
        assert actions["type_id"].iloc[1] == spadlconfig.actiontype_id["pass"]

    def test_recovery_by_gk_maps_to_keeper_pick_up(self):
        actions, _ = metrica_mod.convert_to_actions(
            _df_metrica_recovery_by_gk(), home_team_id="Home", goalkeeper_ids={"GK_HOME"}
        )
        assert len(actions) == 1
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["keeper_pick_up"]

    def test_aerial_won_by_gk_maps_to_keeper_claim(self):
        actions, _ = metrica_mod.convert_to_actions(
            _df_metrica_aerial_won_by_gk(), home_team_id="Home", goalkeeper_ids={"GK_HOME"}
        )
        assert len(actions) == 1
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["keeper_claim"]

    def test_aerial_lost_by_gk_unchanged(self):
        # AERIAL-LOST → CHALLENGE not WON → dropped by default Metrica dispatch.
        # Adding goalkeeper_ids does NOT promote it to keeper_claim.
        actions, _ = metrica_mod.convert_to_actions(
            _df_metrica_aerial_lost_by_gk(), home_team_id="Home", goalkeeper_ids={"GK_HOME"}
        )
        assert len(actions) == 0

    def test_pass_by_non_gk_player_unchanged_with_goalkeeper_ids(self):
        df = _df_minimal_pass()
        df["player"] = ["NOT_GK"]
        actions, _ = metrica_mod.convert_to_actions(
            df, home_team_id="Home", goalkeeper_ids={"GK_HOME"}
        )
        assert len(actions) == 1
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["pass"]

    def test_empty_goalkeeper_ids_set_equivalent_to_none(self):
        df = _df_metrica_pass_by_gk()
        actions_empty, _ = metrica_mod.convert_to_actions(df, home_team_id="Home", goalkeeper_ids=set())
        actions_none, _ = metrica_mod.convert_to_actions(df, home_team_id="Home", goalkeeper_ids=None)
        assert len(actions_empty) == len(actions_none) == 1

    def test_set_piece_freekick_by_gk_unchanged(self):
        # Set pieces are not affected by the goalkeeper_ids routing — a
        # GK taking a free kick is dispatched as freekick_short by default.
        df = _df_metrica("SET PIECE", "FREE KICK")
        df["player"] = ["GK_HOME"]
        actions, _ = metrica_mod.convert_to_actions(
            df, home_team_id="Home", goalkeeper_ids={"GK_HOME"}
        )
        assert actions["type_id"].iloc[0] == spadlconfig.actiontype_id["freekick_short"]

    def test_pass_by_gk_synthetic_pass_has_other_bodypart(self):
        # The synthesized pass uses bodypart=other (matching sportec throwOut shape).
        actions, _ = metrica_mod.convert_to_actions(
            _df_metrica_pass_by_gk(), home_team_id="Home", goalkeeper_ids={"GK_HOME"}
        )
        # Action 1 (the synth pass) — bodypart = other
        assert actions["bodypart_id"].iloc[1] == spadlconfig.bodypart_id["other"]

    def test_preserve_native_propagates_to_synthesized_actions(self):
        df = _df_metrica_pass_by_gk()
        df["my_extra"] = ["custom"]
        actions, _ = metrica_mod.convert_to_actions(
            df, home_team_id="Home",
            goalkeeper_ids={"GK_HOME"}, preserve_native=["my_extra"],
        )
        assert actions["my_extra"].iloc[0] == "custom"
        assert actions["my_extra"].iloc[1] == "custom"
```

- [ ] **Step 6.2: Run failing tests**

Run:
```bash
uv run pytest tests/spadl/test_metrica.py::TestMetricaGoalkeeperIdsRouting -v --tb=short
```

Expected: tests fail (TypeError on unexpected `goalkeeper_ids` keyword argument; tests using the parameter all error).

- [ ] **Step 6.3: Add `goalkeeper_ids` parameter + routing to metrica.py**

Edit `silly_kicks/spadl/metrica.py`. Update the signature:

```python
def convert_to_actions(
    events: pd.DataFrame,
    home_team_id: str,
    *,
    preserve_native: list[str] | None = None,
    goalkeeper_ids: set[str] | None = None,
) -> tuple[pd.DataFrame, ConversionReport]:
```

Update the docstring:

```python
    """Convert normalized Metrica event DataFrame to SPADL actions.

    Parameters
    ----------
    events : pd.DataFrame
        Normalized Metrica event data.
    home_team_id : str
        Home-team identifier (used to flip away-team coords).
    preserve_native : list[str] or None, default ``None``
        Caller-attached columns to preserve through to the output.
    goalkeeper_ids : set[str] or None, default ``None``
        Optional set of player_ids known to be goalkeepers in this match.
        Metrica's source format does not natively mark GK actions in any
        event subtype; without this parameter the output contains zero
        ``keeper_*`` actions (preserved as 1.9.0 default behaviour, no
        breaking change). When provided, applies conservative routing:

        - ``PASS`` (any subtype) by GK → synth ``keeper_pick_up + pass``
          (GK had ball, then distributed).
        - ``RECOVERY`` (any subtype) by GK → ``keeper_pick_up``.
        - ``CHALLENGE`` with subtype ``"AERIAL-WON"`` by GK → ``keeper_claim``.

        Other event types and other CHALLENGE subtypes (including
        ``"AERIAL-LOST"``) are unchanged. An empty set is equivalent to
        ``None``. Pass goalkeeper_ids from match metadata (squad records
        / dim_players join) to enable GK coverage on Metrica data.
    """
```

Thread `goalkeeper_ids` to `_build_raw_actions`:

```python
    raw_actions = _build_raw_actions(events, preserve_native, goalkeeper_ids=goalkeeper_ids)
```

Update `_build_raw_actions` signature:

```python
def _build_raw_actions(
    events: pd.DataFrame,
    preserve_native: list[str] | None,
    *,
    goalkeeper_ids: set[str] | None = None,
) -> pd.DataFrame:
```

Inside `_build_raw_actions`, BEFORE the existing dispatch (right after `sub_raw = ...`):

```python
    # Goalkeeper routing (Bug 3 fix, 1.10.0). Metrica has no native GK
    # markers; this is the only mechanism that surfaces GK actions.
    is_gk_player = np.zeros(n, dtype=bool)
    if goalkeeper_ids:
        gk_str_set = {str(g) for g in goalkeeper_ids}
        is_gk_player = rows["player"].astype(str).isin(gk_str_set).to_numpy()
```

Then update each affected dispatch:

For PASS: find the existing PASS block. After the existing assignments, add:

```python
    # GK distribution: PASS by GK → keeper_pick_up + synthesized pass.
    is_pass_gk_distribution = is_pass & is_gk_player
    type_ids[is_pass_gk_distribution] = spadlconfig.actiontype_id["keeper_pick_up"]
    bodypart_ids[is_pass_gk_distribution] = spadlconfig.bodypart_id["other"]
```

For RECOVERY: replace the existing RECOVERY block:

```python
    # --- RECOVERY -> interception (or keeper_pick_up if by known GK) ---
    is_recovery = typ == "RECOVERY"
    is_recovery_gk = is_recovery & is_gk_player
    type_ids[is_recovery & ~is_gk_player] = spadlconfig.actiontype_id["interception"]
    type_ids[is_recovery_gk] = spadlconfig.actiontype_id["keeper_pick_up"]
    bodypart_ids[is_recovery_gk] = spadlconfig.bodypart_id["other"]
```

For CHALLENGE (existing):

```python
    # --- CHALLENGE: WON -> tackle; AERIAL-WON-by-GK -> keeper_claim;
    # other LOST/AERIAL-LOST/etc -> drop ---
    is_challenge = typ == "CHALLENGE"
    is_challenge_won = is_challenge & (sub_raw == "WON")
    is_challenge_aerial_won = is_challenge & (sub_raw == "AERIAL-WON")
    is_challenge_aerial_won_gk = is_challenge_aerial_won & is_gk_player
    type_ids[is_challenge_won] = spadlconfig.actiontype_id["tackle"]
    type_ids[is_challenge_aerial_won_gk] = spadlconfig.actiontype_id["keeper_claim"]
    bodypart_ids[is_challenge_aerial_won_gk] = spadlconfig.bodypart_id["other"]
    is_challenge_dropped = is_challenge & ~is_challenge_won & ~is_challenge_aerial_won_gk
    type_ids[is_challenge_dropped] = spadlconfig.actiontype_id["non_action"]
```

Now add the synthesis logic for PASS-by-GK. Add a new helper `_synthesize_metrica_gk_pass` near the bottom of `metrica.py`:

```python
def _synthesize_metrica_gk_pass(
    rows: pd.DataFrame,
    is_pass_gk_distribution: np.ndarray,
    preserve_native: list[str] | None,
) -> pd.DataFrame:
    """Build synthetic pass rows for PASS-by-GK events (Bug 3 fix).

    Each source row matching ``is_pass_gk_distribution`` already had its
    type_id set to keeper_pick_up by the caller. This helper emits the
    second half (a synthesized SPADL pass) immediately after.

    Returned DataFrame includes a ``_row_order`` column for interleaving.
    """
    if not is_pass_gk_distribution.any():
        return pd.DataFrame()

    src_indices = np.where(is_pass_gk_distribution)[0]
    src = rows.iloc[src_indices].copy().reset_index(drop=True)
    n_synth = len(src)

    synth = pd.DataFrame(
        {
            "game_id": src["match_id"].astype("object"),
            "original_event_id": (src["event_id"].astype(str) + "_synth_pass").astype("object"),
            "period_id": src["period"].astype(np.int64),
            "time_seconds": src["start_time_s"].astype(np.float64),
            "team_id": src["team"].astype("object"),
            "player_id": src["player"].astype("object"),
            "start_x": src["start_x"].astype(np.float64),
            "start_y": src["start_y"].astype(np.float64),
            "end_x": src["end_x"].astype(np.float64),
            "end_y": src["end_y"].astype(np.float64),
            "type_id": np.full(n_synth, spadlconfig.actiontype_id["pass"], dtype=np.int64),
            "result_id": np.full(n_synth, spadlconfig.result_id["success"], dtype=np.int64),
            "bodypart_id": np.full(n_synth, spadlconfig.bodypart_id["other"], dtype=np.int64),
        }
    )
    synth["_row_order"] = src_indices.astype(np.int64) * 2 + 1

    if preserve_native:
        for col in preserve_native:
            if col in src.columns:
                synth[col] = src[col].to_numpy()
            else:
                synth[col] = np.nan

    return synth
```

In `_build_raw_actions`, after the main `actions = pd.DataFrame(...)` assembly and BEFORE the `actions = actions.loc[~drop_mask]` filter, add the synthesis interleave (parallel to sportec's pattern):

Find:

```python
    actions = pd.DataFrame(
        {
            "game_id": rows["match_id"].astype("object"),
            ...
        }
    )

    actions = actions.loc[~drop_mask]
    actions = actions[actions["type_id"] != spadlconfig.actiontype_id["non_action"]].reset_index(drop=True)
```

Replace with:

```python
    actions = pd.DataFrame(
        {
            "game_id": rows["match_id"].astype("object"),
            "original_event_id": rows["event_id"].astype("object"),
            "period_id": rows["period"].astype(np.int64),
            "time_seconds": rows["start_time_s"].astype(np.float64),
            "team_id": rows["team"].astype("object"),
            "player_id": rows["player"].astype("object"),
            "start_x": rows["start_x"].astype(np.float64),
            "start_y": rows["start_y"].astype(np.float64),
            "end_x": rows["end_x"].astype(np.float64),
            "end_y": rows["end_y"].astype(np.float64),
            "type_id": type_ids,
            "result_id": result_ids,
            "bodypart_id": bodypart_ids,
        }
    )

    # Carry preserve_native onto main actions BEFORE synthesis.
    if preserve_native:
        for col in preserve_native:
            if col in rows.columns:
                actions[col] = rows[col].to_numpy()
            else:
                actions[col] = np.nan

    # Synthesize keeper-distribution pass rows (Bug 3 fix).
    is_pass_gk_distribution_mask = is_pass & is_gk_player
    synth = _synthesize_metrica_gk_pass(rows, is_pass_gk_distribution_mask, preserve_native)
    if len(synth) > 0:
        actions["_row_order"] = np.arange(len(actions), dtype=np.int64) * 2
        combined = pd.concat([actions, synth], ignore_index=True, sort=False)
        # Honour drop_mask via _row_order alignment: drop_mask is aligned
        # with the original 1:1 actions; map via _row_order back to the
        # source index. drop_mask[i] = True means drop the source row at i,
        # which corresponds to _row_order == i*2 (and any synth at i*2+1
        # would also be dropped — but synth rows by construction don't
        # come from drop_mask rows, since drop_mask only fires on
        # SET PIECE FREE KICK rows, not PASS-by-GK rows).
        combined = combined.sort_values("_row_order", kind="mergesort").drop(columns=["_row_order"]).reset_index(drop=True)
        actions = combined
        # Re-apply drop_mask via type_id non_action (drop_mask sources are
        # already non_action-marked elsewhere).

    actions = actions.loc[~np.isin(np.arange(len(actions)), np.where(drop_mask)[0])] if len(synth) == 0 else actions
    actions = actions[actions["type_id"] != spadlconfig.actiontype_id["non_action"]].reset_index(drop=True)
```

Hmm, the `drop_mask` integration is getting complex due to row interleaving. Simplification: since `drop_mask` only fires on SET PIECE FREE KICK rows that get composed into a following SHOT, and PASS-by-GK rows are not SET PIECE FREE KICK rows, we can apply `drop_mask` BEFORE the synthesis interleave. Cleaner approach:

Replace the above with:

```python
    actions = pd.DataFrame(
        {
            "game_id": rows["match_id"].astype("object"),
            "original_event_id": rows["event_id"].astype("object"),
            "period_id": rows["period"].astype(np.int64),
            "time_seconds": rows["start_time_s"].astype(np.float64),
            "team_id": rows["team"].astype("object"),
            "player_id": rows["player"].astype("object"),
            "start_x": rows["start_x"].astype(np.float64),
            "start_y": rows["start_y"].astype(np.float64),
            "end_x": rows["end_x"].astype(np.float64),
            "end_y": rows["end_y"].astype(np.float64),
            "type_id": type_ids,
            "result_id": result_ids,
            "bodypart_id": bodypart_ids,
        }
    )

    # Carry preserve_native onto main actions BEFORE synthesis.
    if preserve_native:
        for col in preserve_native:
            if col in rows.columns:
                actions[col] = rows[col].to_numpy()
            else:
                actions[col] = np.nan

    # Apply drop_mask FIRST (composed FREE KICK rows), then synthesize.
    # _row_order aligns synth rows with their post-drop position by
    # using the original index — but since drop_mask only drops SET
    # PIECE rows (never PASS-by-GK rows), the synth row_orders remain
    # valid relative to the source index. Track via a parallel "kept"
    # array on actions to assign row_order before dropping.
    actions["_row_order"] = np.arange(len(actions), dtype=np.int64) * 2
    actions = actions.loc[~drop_mask].reset_index(drop=True)

    # Synthesize keeper-distribution pass rows.
    is_pass_gk_distribution_mask = is_pass & is_gk_player & ~drop_mask
    synth = _synthesize_metrica_gk_pass(rows, is_pass_gk_distribution_mask, preserve_native)
    if len(synth) > 0:
        combined = pd.concat([actions, synth], ignore_index=True, sort=False)
        combined = combined.sort_values("_row_order", kind="mergesort").drop(columns=["_row_order"]).reset_index(drop=True)
        actions = combined
    else:
        actions = actions.drop(columns=["_row_order"])

    actions = actions[actions["type_id"] != spadlconfig.actiontype_id["non_action"]].reset_index(drop=True)
```

- [ ] **Step 6.4: Run goalkeeper_ids tests, verify pass**

Run:
```bash
uv run pytest tests/spadl/test_metrica.py::TestMetricaGoalkeeperIdsRouting -v --tb=short
```

Expected: all ~12 tests pass.

- [ ] **Step 6.5: Run full metrica test suite**

Run:
```bash
uv run pytest tests/spadl/test_metrica.py -v --tb=short -q 2>&1 | tail -20
```

Expected: all metrica tests pass, including pre-existing ones.

- [ ] **Step 6.6: Stage**

```bash
git add silly_kicks/spadl/metrica.py tests/spadl/test_metrica.py
```

---

## Task 7: statsbomb.py + opta.py — `goalkeeper_ids` API symmetry no-op (TDD)

**Files:**
- Modify: `silly_kicks/spadl/statsbomb.py` (signature + docstring + no-op acceptance)
- Modify: `silly_kicks/spadl/opta.py` (signature + docstring + no-op acceptance)
- Modify: `tests/spadl/test_statsbomb.py` (add `TestStatsBombGoalkeeperIdsNoOp`)
- Modify: `tests/spadl/test_opta.py` (add `TestOptaGoalkeeperIdsNoOp`)

**Behaviour:** Parameter silently accepted; output is byte-for-byte identical with and without the parameter (StatsBomb / Opta source events natively mark GK actions, so `goalkeeper_ids` is supplementary and currently has no effect on output).

- [ ] **Step 7.1: Write failing no-op tests**

Append to `tests/spadl/test_statsbomb.py`:

```python
# ---------------------------------------------------------------------------
# goalkeeper_ids — accepted as no-op for cross-provider API symmetry (1.10.0)
# StatsBomb's source events natively mark GK actions, so the parameter has
# no effect on output. Asserting byte-for-byte equivalence catches drift.
# ---------------------------------------------------------------------------


class TestStatsBombGoalkeeperIdsNoOp:
    def test_goalkeeper_ids_parameter_is_accepted(self):
        # Smoke: passing the parameter should not raise.
        from silly_kicks.spadl import statsbomb as sb

        # Use the smallest possible valid input via the fixtures.
        # If a minimal _df builder isn't present, build a tiny one inline.
        events = pd.DataFrame(
            {
                "game_id": [1],
                "event_id": ["e1"],
                "period_id": [1],
                "timestamp": ["00:00:01.000"],
                "team_id": [100],
                "player_id": [200],
                "type_name": ["Pass"],
                "location": [[60.0, 40.0]],
                "extra": [{}],
            }
        )
        actions, _ = sb.convert_to_actions(
            events, home_team_id=100, goalkeeper_ids={200, 300}
        )
        assert isinstance(actions, pd.DataFrame)

    def test_goalkeeper_ids_output_identical_with_and_without(self):
        from silly_kicks.spadl import statsbomb as sb

        events = pd.DataFrame(
            {
                "game_id": [1, 1],
                "event_id": ["e1", "e2"],
                "period_id": [1, 1],
                "timestamp": ["00:00:01.000", "00:00:02.000"],
                "team_id": [100, 100],
                "player_id": [200, 201],
                "type_name": ["Pass", "Pass"],
                "location": [[60.0, 40.0], [70.0, 40.0]],
                "extra": [{}, {}],
            }
        )
        a_none, _ = sb.convert_to_actions(events, home_team_id=100)
        a_set, _ = sb.convert_to_actions(events, home_team_id=100, goalkeeper_ids={200})
        # Byte-for-byte equality on every column.
        pd.testing.assert_frame_equal(a_none, a_set, check_dtype=True)

    def test_empty_goalkeeper_ids_set_equivalent_to_none(self):
        from silly_kicks.spadl import statsbomb as sb

        events = pd.DataFrame(
            {
                "game_id": [1],
                "event_id": ["e1"],
                "period_id": [1],
                "timestamp": ["00:00:01.000"],
                "team_id": [100],
                "player_id": [200],
                "type_name": ["Pass"],
                "location": [[60.0, 40.0]],
                "extra": [{}],
            }
        )
        a_empty, _ = sb.convert_to_actions(events, home_team_id=100, goalkeeper_ids=set())
        a_none, _ = sb.convert_to_actions(events, home_team_id=100, goalkeeper_ids=None)
        pd.testing.assert_frame_equal(a_empty, a_none, check_dtype=True)
```

Append a parallel test class to `tests/spadl/test_opta.py`:

```python
# ---------------------------------------------------------------------------
# goalkeeper_ids — accepted as no-op for cross-provider API symmetry (1.10.0)
# ---------------------------------------------------------------------------


class TestOptaGoalkeeperIdsNoOp:
    def test_goalkeeper_ids_parameter_is_accepted(self):
        from silly_kicks.spadl import opta

        events = pd.DataFrame(
            {
                "game_id": [1],
                "event_id": ["e1"],
                "period_id": [1],
                "minute": [0],
                "second": [1],
                "team_id": [100],
                "player_id": [200],
                "type_name": ["pass"],
                "outcome": [True],
                "start_x": [50.0],
                "start_y": [50.0],
                "end_x": [60.0],
                "end_y": [50.0],
                "qualifiers": [{}],
            }
        )
        actions, _ = opta.convert_to_actions(events, home_team_id=100, goalkeeper_ids={200})
        assert isinstance(actions, pd.DataFrame)

    def test_goalkeeper_ids_output_identical_with_and_without(self):
        from silly_kicks.spadl import opta

        events = pd.DataFrame(
            {
                "game_id": [1, 1],
                "event_id": ["e1", "e2"],
                "period_id": [1, 1],
                "minute": [0, 0],
                "second": [1, 2],
                "team_id": [100, 100],
                "player_id": [200, 201],
                "type_name": ["pass", "pass"],
                "outcome": [True, True],
                "start_x": [50.0, 60.0],
                "start_y": [50.0, 50.0],
                "end_x": [60.0, 70.0],
                "end_y": [50.0, 50.0],
                "qualifiers": [{}, {}],
            }
        )
        a_none, _ = opta.convert_to_actions(events, home_team_id=100)
        a_set, _ = opta.convert_to_actions(events, home_team_id=100, goalkeeper_ids={200})
        pd.testing.assert_frame_equal(a_none, a_set, check_dtype=True)
```

(Both test files already import pandas as pd at the top.)

- [ ] **Step 7.2: Run no-op tests, verify they fail**

Run:
```bash
uv run pytest tests/spadl/test_statsbomb.py::TestStatsBombGoalkeeperIdsNoOp tests/spadl/test_opta.py::TestOptaGoalkeeperIdsNoOp -v --tb=short
```

Expected: TypeError on unexpected `goalkeeper_ids` keyword.

- [ ] **Step 7.3: Add no-op `goalkeeper_ids` parameter to statsbomb.py**

Edit `silly_kicks/spadl/statsbomb.py`. Update the signature:

```python
def convert_to_actions(
    events: pd.DataFrame,
    home_team_id: int,
    xy_fidelity_version: int | None = None,
    shot_fidelity_version: int | None = None,
    *,
    preserve_native: list[str] | None = None,
    goalkeeper_ids: set[int] | None = None,
) -> tuple[pd.DataFrame, ConversionReport]:
```

Update the docstring (Parameters section), adding after `preserve_native`:

```python
    goalkeeper_ids : set[int] or None, default ``None``
        Accepted for cross-provider API symmetry with the
        ``sportec`` / ``metrica`` converters (silly-kicks 1.10.0+); has
        no effect on StatsBomb output because StatsBomb's source events
        natively mark GK actions via the ``Goal Keeper`` event type. The
        parameter is silently accepted; the output is byte-for-byte
        identical with and without it. Empty set ≡ ``None``.
```

Inside the function body, no logic changes — the parameter is consumed and ignored. Add a noqa-style docstring acknowledgment by referencing the parameter name in the body to avoid pyright "unused parameter" complaints (pyright doesn't flag this by default for `basic` mode, but explicit is safer):

Right after `_validate_input_columns(events, EXPECTED_INPUT_COLUMNS, provider="StatsBomb")` add:

```python
    # goalkeeper_ids: accepted for cross-provider API symmetry; no-op for
    # StatsBomb because source events natively mark GK actions.
    _ = goalkeeper_ids
```

- [ ] **Step 7.4: Add no-op `goalkeeper_ids` parameter to opta.py**

Edit `silly_kicks/spadl/opta.py`. Update the signature:

```python
def convert_to_actions(
    events: pd.DataFrame,
    home_team_id: int,
    *,
    preserve_native: list[str] | None = None,
    goalkeeper_ids: set[int] | None = None,
) -> tuple[pd.DataFrame, ConversionReport]:
```

Update the docstring (Parameters section), adding after `preserve_native`:

```python
    goalkeeper_ids : set[int] or None, default ``None``
        Accepted for cross-provider API symmetry with the
        ``sportec`` / ``metrica`` converters (silly-kicks 1.10.0+); has
        no effect on Opta output because Opta's source events natively
        mark GK actions via the dedicated ``save`` / ``claim`` /
        ``punch`` / ``keeper pick-up`` event types. The parameter is
        silently accepted; the output is byte-for-byte identical with
        and without it. Empty set ≡ ``None``.
```

Add the no-op acknowledgment after `_validate_input_columns(...)`:

```python
    # goalkeeper_ids: accepted for cross-provider API symmetry; no-op for
    # Opta because source events natively mark GK actions.
    _ = goalkeeper_ids
```

- [ ] **Step 7.5: Run no-op tests, verify all PASS**

Run:
```bash
uv run pytest tests/spadl/test_statsbomb.py::TestStatsBombGoalkeeperIdsNoOp tests/spadl/test_opta.py::TestOptaGoalkeeperIdsNoOp -v --tb=short
```

Expected: 6 tests pass.

- [ ] **Step 7.6: Stage**

```bash
git add silly_kicks/spadl/statsbomb.py silly_kicks/spadl/opta.py tests/spadl/test_statsbomb.py tests/spadl/test_opta.py
```

---

## Task 8: Cross-provider parity meta-test (TDD)

**Files:**
- Create: `tests/spadl/test_cross_provider_parity.py`

This is the regression gate that would have caught Bugs 1-3 in 1.7.0 if it had existed. Parametrized over the 5 DataFrame converters; each must emit at least one `keeper_*` action when given a fixture exercising GK paths (with appropriate `goalkeeper_ids` where the source format requires it).

- [ ] **Step 8.1: Create `tests/spadl/test_cross_provider_parity.py`**

Create:

```python
"""Cross-provider parity meta-test (added in silly-kicks 1.10.0).

For every DataFrame SPADL converter, asserts that, when given a fixture
that exercises GK paths (with ``goalkeeper_ids`` where the source format
requires it), the output contains at least one ``keeper_*`` action.

Pre-1.10.0 this test would have failed for sportec (Pass→Play bug + missing
throwOut/punt vocabulary) and metrica (no native GK markers + no
goalkeeper_ids parameter). Post-1.10.0 all 5 DataFrame converters pass —
ensuring future converter regressions in the keeper-action emission class
surface immediately.

The Wyscout DataFrame converter is exercised via a synthetic fixture (an
aerial duel by a known goalkeeper, since vendoring a Wyscout production
fixture is out of scope for PR-S10).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl import coverage_metrics

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_KEEPER_TYPE_NAMES = frozenset({"keeper_save", "keeper_claim", "keeper_punch", "keeper_pick_up"})


# ---------------------------------------------------------------------------
# Adapters: load each provider's vendored fixture and run convert_to_actions
# ---------------------------------------------------------------------------


def _load_idsse_fixture():
    from silly_kicks.spadl import sportec

    parquet_path = _REPO_ROOT / "tests" / "datasets" / "idsse" / "sample_match.parquet"
    if not parquet_path.exists():
        pytest.fail(
            f"IDSSE fixture not found at {parquet_path}. Regenerate via "
            f"scripts/extract_provider_fixtures.py --provider idsse."
        )
    events = pd.read_parquet(parquet_path)
    home_team = events["team"].dropna().iloc[0]
    # The brief noted a known GK player_id pattern on bronze.idsse_events;
    # the fixture's GK ids are recoverable as the player_ids on rows where
    # play_goal_keeper_action is non-null.
    if "play_goal_keeper_action" in events.columns:
        gk_ids = set(events.loc[events["play_goal_keeper_action"].notna(), "player_id"].dropna().astype(str).tolist())
    else:
        gk_ids = None
    actions, _ = sportec.convert_to_actions(events, home_team_id=str(home_team), goalkeeper_ids=gk_ids)
    return actions


def _load_metrica_fixture():
    from silly_kicks.spadl import metrica

    parquet_path = _REPO_ROOT / "tests" / "datasets" / "metrica" / "sample_match.parquet"
    if not parquet_path.exists():
        pytest.fail(
            f"Metrica fixture not found at {parquet_path}. Regenerate via "
            f"scripts/extract_provider_fixtures.py --provider metrica."
        )
    events = pd.read_parquet(parquet_path)
    home_team = events["team"].dropna().iloc[0]
    # For Metrica we can't derive GK ids from the source format (no GK
    # markers exist). Use the convention "first PASS player from the home
    # team" as the assumed GK to surface at least one synth path. For real
    # production use, callers MUST supply goalkeeper_ids from squad metadata.
    home_passes = events[(events["type"] == "PASS") & (events["team"] == home_team)]
    if home_passes.empty:
        pytest.skip("Metrica fixture lacks any PASS-by-home-team event; cannot exercise GK path")
    gk_id = str(home_passes["player"].iloc[0])
    actions, _ = metrica.convert_to_actions(events, home_team_id=str(home_team), goalkeeper_ids={gk_id})
    return actions


def _load_statsbomb_fixture():
    from silly_kicks.spadl import statsbomb

    # Use the existing PR-S8 fixtures.
    fixture_path = _REPO_ROOT / "tests" / "datasets" / "statsbomb" / "raw" / "events" / "7298.json"
    if not fixture_path.exists():
        pytest.fail(
            f"StatsBomb fixture not found at {fixture_path}. See "
            f"tests/datasets/statsbomb/README.md for vendoring details."
        )

    with open(fixture_path, encoding="utf-8") as f:
        events_raw = json.load(f)

    _top_level_keys = {"id", "period", "timestamp", "team", "player", "type", "location"}
    adapted = pd.DataFrame(
        [
            {
                "game_id": 7298,
                "event_id": e.get("id"),
                "period_id": e.get("period"),
                "timestamp": e.get("timestamp"),
                "team_id": (e.get("team") or {}).get("id"),
                "player_id": (e.get("player") or {}).get("id"),
                "type_name": (e.get("type") or {}).get("name"),
                "location": e.get("location"),
                "extra": {k: v for k, v in e.items() if k not in _top_level_keys},
            }
            for e in events_raw
        ]
    )
    home_team_id = int(adapted["team_id"].dropna().iloc[0])
    actions, _ = statsbomb.convert_to_actions(adapted, home_team_id=home_team_id)
    return actions


def _load_opta_fixture():
    from silly_kicks.spadl import opta

    # Synthetic Opta-shape fixture exercising GK actions (no production
    # Opta sample is currently vendored; this synthetic one verifies the
    # converter emits keeper_save when given an appropriate "save" event).
    events = pd.DataFrame(
        {
            "game_id": [1, 1],
            "event_id": ["e1", "e2"],
            "period_id": [1, 1],
            "minute": [0, 0],
            "second": [1, 2],
            "team_id": [100, 200],
            "player_id": [201, 100],
            "type_name": ["pass", "save"],
            "outcome": [True, True],
            "start_x": [50.0, 5.0],
            "start_y": [50.0, 50.0],
            "end_x": [60.0, 5.0],
            "end_y": [50.0, 50.0],
            "qualifiers": [{}, {}],
        }
    )
    actions, _ = opta.convert_to_actions(events, home_team_id=100)
    return actions


def _load_wyscout_fixture():
    from silly_kicks.spadl import wyscout

    # Synthetic Wyscout-shape fixture exercising the goalkeeper_ids
    # aerial-duel reclassification path (live since 1.0.0).
    events = pd.DataFrame(
        {
            "game_id": [1],
            "event_id": [1],
            "period_id": [1],
            "milliseconds": [1000],
            "team_id": [100],
            "player_id": [200],
            "type_id": [1],  # _WS_TYPE_DUEL
            "subtype_id": [10],  # _WS_SUBTYPE_AIR_DUEL
            "positions": [[{"x": 10, "y": 50}, {"x": 10, "y": 50}]],
            "tags": [[{"id": 703}]],  # tag 703 = "won"
        }
    )
    actions, _ = wyscout.convert_to_actions(events, home_team_id=100, goalkeeper_ids={200})
    return actions


_PROVIDER_LOADERS = {
    "sportec": _load_idsse_fixture,
    "metrica": _load_metrica_fixture,
    "statsbomb": _load_statsbomb_fixture,
    "opta": _load_opta_fixture,
    "wyscout": _load_wyscout_fixture,
}


# ---------------------------------------------------------------------------
# The parity gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider", list(_PROVIDER_LOADERS.keys()))
def test_converter_emits_at_least_one_keeper_action(provider: str):
    """Each DataFrame SPADL converter, given a fixture exercising GK paths
    (with appropriate goalkeeper_ids where the source format requires it),
    must emit at least one keeper_* action.

    This test would have caught Bugs 1-3 in silly-kicks 1.7.0 if it had
    existed.
    """
    actions = _PROVIDER_LOADERS[provider]()
    m = coverage_metrics(actions=actions, expected_action_types=set(_KEEPER_TYPE_NAMES))
    keeper_count_total = sum(m["counts"].get(t, 0) for t in _KEEPER_TYPE_NAMES)
    assert keeper_count_total > 0, (
        f"Provider {provider!r} emitted zero keeper_* actions on its fixture. "
        f"Coverage breakdown: counts={m['counts']}, missing={m['missing']}."
    )


@pytest.mark.parametrize("provider", list(_PROVIDER_LOADERS.keys()))
def test_converter_returns_canonical_spadl_columns(provider: str):
    """Sanity check: every converter returns a DataFrame with type_id present."""
    actions = _PROVIDER_LOADERS[provider]()
    assert "type_id" in actions.columns
    assert len(actions) > 0
```

- [ ] **Step 8.2: Run the cross-provider parity tests**

Run:
```bash
uv run pytest tests/spadl/test_cross_provider_parity.py -v --tb=short
```

Expected: all 10 tests pass (5 providers × 2 test methods). The IDSSE parity test specifically validates Bug 1 + Bug 2 fixes. The Metrica test validates Bug 3.

If the IDSSE fixture's GK player_ids don't include any with throwOut/punt qualifiers (depends on the random sample), the test may need an adjustment. In that case, add an inline check that the fixture contains at least one Play row with `play_goal_keeper_action` non-null.

- [ ] **Step 8.3: Stage**

```bash
git add tests/spadl/test_cross_provider_parity.py
```

---

## Task 9: Documentation — vocabulary tables, docstring updates

**Files:**
- Modify: `silly_kicks/spadl/sportec.py` (module docstring + DFL vocabulary tables)
- Modify: `silly_kicks/spadl/metrica.py` (module docstring + goalkeeper_ids contract section)

- [ ] **Step 9.1: Add DFL vocabulary tables to sportec.py module docstring**

Edit the top of `silly_kicks/spadl/sportec.py`. Replace the current module docstring:

```python
"""Sportec (DFL) DataFrame SPADL converter.

Converts already-normalized DFL event DataFrames (e.g., luxury-lakehouse
``bronze.idsse_events`` shape, Bassek 2025 DFL parse output) to SPADL actions.

Consumers with raw DFL XML files should use ``silly_kicks.spadl.kloppy`` after
``kloppy.sportec.load_event(...)``.
"""
```

With:

```python
"""Sportec (DFL) DataFrame SPADL converter.

Converts already-normalized DFL event DataFrames (e.g., luxury-lakehouse
``bronze.idsse_events`` shape, Bassek 2025 DFL parse output) to SPADL actions.

Consumers with raw DFL XML files should use ``silly_kicks.spadl.kloppy`` after
``kloppy.sportec.load_event(...)``.

DFL ``event_type`` vocabulary (recognized as input)
---------------------------------------------------

The dispatch consumes the following DFL event_type values (case-sensitive):

================== ======================================================
``event_type``     SPADL mapping
================== ======================================================
``Play``           Pass-class by default; refined by qualifier (see below)
``ShotAtGoal``     ``shot`` / ``shot_freekick`` / ``shot_penalty``
``TacklingGame``   ``tackle``
``Foul``           ``foul`` (with optional ``Caution`` pairing for cards)
``FreeKick``       ``freekick_short`` / ``freekick_crossed``
``Corner``         ``corner_short`` / ``corner_crossed``
``ThrowIn``        ``throw_in``
``GoalKick``       ``goalkick``
================== ======================================================

DFL ``play_goal_keeper_action`` qualifier vocabulary
----------------------------------------------------

For ``Play`` events, the ``play_goal_keeper_action`` qualifier disambiguates
between pass-class and GK-class semantics:

================== ============================== ============================
Qualifier value    SPADL mapping                  Notes
================== ============================== ============================
``""`` (empty)     ``pass`` / ``cross``           Pass-class default
``save``           ``keeper_save``
``claim``          ``keeper_claim``
``punch``          ``keeper_punch``
``pickUp``         ``keeper_pick_up``
``throwOut``       ``keeper_pick_up`` + ``pass``  GK distribution by hand
``punt``           ``keeper_pick_up`` + ``goalkick``  GK distribution by foot
unrecognized       ``non_action`` (filtered)      Defensive
================== ============================== ============================

The ``throwOut`` and ``punt`` rows synthesize TWO SPADL actions per source
event: a ``keeper_pick_up`` representing the GK's reception of the ball,
followed by a ``pass`` (for ``throwOut``) or ``goalkick`` (for ``punt``)
representing the GK's distribution. Action_ids are renumbered dense after
synthesis. ``preserve_native`` columns propagate to both rows.

Bug history
-----------

Pre-1.10.0:

- ``is_pass = et == "Pass"`` silently dropped ALL DFL ``Play`` events
  (the actual pass-class event_type) to ``non_action``. Net effect: all
  IDSSE matches in production lost ALL pass-class events for ~3 release
  cycles. Fixed in 1.10.0 by restructuring the dispatch around
  ``et == "Play"``.
- ``play_goal_keeper_action`` qualifier vocabulary was incomplete (only
  ``save`` / ``claim`` / ``punch`` / ``pickUp``); ``throwOut`` and ``punt``
  silently dropped to ``non_action``. Fixed in 1.10.0 by adding the
  2-action synthesis.

See ``docs/superpowers/specs/2026-04-29-gk-converter-coverage-parity-design.md``
for the full design rationale.
"""
```

- [ ] **Step 9.2: Add Metrica goalkeeper_ids contract section to metrica.py module docstring**

Edit the top of `silly_kicks/spadl/metrica.py`. Replace the current module docstring:

```python
"""Metrica DataFrame SPADL converter.

Converts already-normalized Metrica event DataFrames (e.g., parsed from
Metrica's open-data CSV / EPTS-JSON formats) to SPADL actions.

Consumers with raw Metrica files should use ``silly_kicks.spadl.kloppy`` after
``kloppy.metrica.load_event(...)``.
"""
```

With:

```python
"""Metrica DataFrame SPADL converter.

Converts already-normalized Metrica event DataFrames (e.g., parsed from
Metrica's open-data CSV / EPTS-JSON formats) to SPADL actions.

Consumers with raw Metrica files should use ``silly_kicks.spadl.kloppy`` after
``kloppy.metrica.load_event(...)``.

Metrica ``type`` / ``subtype`` vocabulary
-----------------------------------------

Recognized event types (case-sensitive ``UPPER``):

==================== =====================================================
``type``             SPADL mapping
==================== =====================================================
``PASS``             ``pass`` (default) / ``cross`` / ``goalkick`` /
                     ``keeper_pick_up + pass`` (when by GK and goalkeeper_ids)
``SHOT``             ``shot`` (with set-piece composition for FREE KICK)
``RECOVERY``         ``interception`` (default) / ``keeper_pick_up`` (when by GK)
``CHALLENGE``        ``tackle`` (when WON) / ``keeper_claim`` (AERIAL-WON by GK) /
                     dropped (LOST / other AERIAL variants)
``BALL LOST``        ``bad_touch`` (fail)
``FAULT``            ``foul`` (with CARD pairing for cards)
``SET PIECE``        ``freekick_short`` / ``corner_short`` / ``throw_in`` / ``goalkick``
==================== =====================================================

Goalkeeper coverage contract
----------------------------

Metrica's source format does NOT natively mark GK actions in any event
subtype — the taxonomy is purely positional/contextual (PASS / CARRY /
CHALLENGE / etc.). To surface ``keeper_*`` SPADL actions, callers must
pass ``goalkeeper_ids`` from match metadata (squad records or
``dim_players`` join on the registered position group).

Without ``goalkeeper_ids``, the output contains zero ``keeper_*`` actions
(preserves 1.9.0 default behaviour).

With ``goalkeeper_ids``, conservative routing applies:

- ``PASS`` (any subtype) by GK → synthesize ``keeper_pick_up + pass``
- ``RECOVERY`` (any subtype) by GK → ``keeper_pick_up``
- ``CHALLENGE`` ``AERIAL-WON`` by GK → ``keeper_claim``
- All other event types unchanged (a GK taking a free kick is still
  ``freekick_short``, not a keeper action — set pieces are positional
  acts, not GK acts in the SPADL vocabulary)

Bug history
-----------

Pre-1.10.0: silly-kicks Metrica converter emitted zero ``keeper_*`` actions
on every match (no source GK markers, no parameter to disambiguate).
``add_gk_role`` and ``add_pre_shot_gk_context`` correctly emitted NULL on
the resulting SPADL — but the upstream gap meant lakehouse production had
zero GK coverage on Metrica. Fixed in 1.10.0 by adding the
``goalkeeper_ids: set[str] | None`` parameter.

See ``docs/superpowers/specs/2026-04-29-gk-converter-coverage-parity-design.md``
for the full design rationale.
"""
```

- [ ] **Step 9.3: Stage**

```bash
git add silly_kicks/spadl/sportec.py silly_kicks/spadl/metrica.py
```

---

## Task 10: Release artifacts — version, CHANGELOG, TODO, C4

**Files:**
- Modify: `pyproject.toml` (version bump)
- Modify: `CHANGELOG.md` (add `## [1.10.0]` entry)
- Modify: `TODO.md` (close PR-S10, bump PR-S11, add atomic coverage_metrics tech debt)
- Modify: `docs/c4/architecture.dsl` (add `coverage_metrics` to spadl container helper enumeration)
- Regen: `docs/c4/architecture.html` (via `/final-review` skill or manual structurizr export)

- [ ] **Step 10.1: Bump version in pyproject.toml**

Edit `pyproject.toml:7`:

```diff
-version = "1.9.0"
+version = "1.10.0"
```

- [ ] **Step 10.2: Add CHANGELOG entry**

Insert at the top of `CHANGELOG.md` (after the header lines and before the existing `## [1.9.0]` entry):

```markdown
## [1.10.0] — 2026-04-29

### Added
- **Public `silly_kicks.spadl.coverage_metrics(*, actions, expected_action_types)` utility**
  for computing per-action-type coverage on a SPADL action stream. Returns
  a `CoverageMetrics` TypedDict (also re-exported from `silly_kicks.spadl`).
  Keyword-only arguments. Resolves `type_id` to action-type name via
  `spadlconfig.actiontypes_df`; reports any expected action types that
  produced zero rows under `missing`. Out-of-vocab `type_id` values are
  reported as `"unknown"` rather than raising. Mirrors the PR-S8
  `boundary_metrics` shape and discipline.
- **`goalkeeper_ids: set[str] | None = None` parameter on
  `silly_kicks.spadl.sportec.convert_to_actions`** as a supplementary
  signal: when provided, Play events whose `player_id` is in the set
  AND which have NO explicit `play_goal_keeper_action` qualifier are
  routed to the keeper_pick_up + pass 2-action synthesis. The
  qualifier-driven mapping remains the primary contract.
- **`goalkeeper_ids: set[str] | None = None` parameter on
  `silly_kicks.spadl.metrica.convert_to_actions`** as the PRIMARY
  mechanism for surfacing GK actions. Metrica's source format lacks
  native GK markers; with `goalkeeper_ids`, conservative routing applies
  (PASS by GK → synth, RECOVERY by GK → keeper_pick_up, CHALLENGE
  AERIAL-WON by GK → keeper_claim). Without it: 0 keeper_* actions
  (1.9.0 default behaviour preserved — no breaking change).
- **`goalkeeper_ids` no-op acceptance on `statsbomb.convert_to_actions`
  and `opta.convert_to_actions`** for cross-provider API symmetry. Both
  source formats natively mark GK actions; the parameter is silently
  accepted with byte-for-byte identical output.
- **DFL distribution qualifiers `throwOut` and `punt` now produce SPADL
  actions** (sportec converter). Each source row synthesizes TWO
  actions: `keeper_pick_up + pass` (bodypart=other) for `throwOut`,
  `keeper_pick_up + goalkick` (bodypart=foot) for `punt`. Both rows
  inherit the source's `(player_id, team, period, time, x, y)`.
  `preserve_native` columns propagate to both. Action_ids renumbered
  dense after synthesis.
- **Production-shape vendored fixtures** under
  `tests/datasets/idsse/sample_match.parquet` (~25-35 KB; subset of
  bronze.idsse_events match J03WMX, includes throwOut + punt rows) and
  `tests/datasets/metrica/sample_match.parquet` (~25-30 KB; subset of
  Metrica Sample Game 2). Build script at
  `scripts/extract_provider_fixtures.py` (Databricks pull for IDSSE,
  offline kloppy-fixture subset for Metrica). Attribution READMEs
  alongside.
- **Cross-provider parity meta-test** at
  `tests/spadl/test_cross_provider_parity.py`. Parametrized over all 5
  DataFrame converters (statsbomb, opta, wyscout, sportec, metrica);
  asserts each emits at least one `keeper_*` action when given a
  fixture exercising GK paths. This is the regression gate that would
  have caught Bugs 1-3 in 1.7.0 if it had existed.

### Fixed
- **Sportec converter no longer drops all DFL `Play` events to
  non_action.** The pre-1.10.0 dispatch checked `et == "Pass"` for
  pass-class events, but DFL bronze never emits `"Pass"` — the actual
  event_type is `"Play"`. Net effect since 1.7.0: all IDSSE matches in
  production lost ~60-80% of their actions (every pass, cross, and head
  pass) to silent non_action drop. Fix restructures the dispatch so
  `Play` events with no GK qualifier route to `pass` / `cross` (with
  optional head bodypart) and `Play` events with a recognized GK
  qualifier route to `keeper_*` actions. Defensive: `Play` events with
  an unrecognized non-empty qualifier still drop to `non_action`.
- **Sportec converter no longer drops `throwOut` and `punt` GK
  distribution events to non_action.** These DFL qualifier values
  represent GK distribution actions (throwing or kicking the ball to
  a teammate); pre-1.10.0 they were unmapped. Fix synthesizes 2
  SPADL actions per source event (see Added section).
- **Metrica converter now produces non-zero GK coverage when
  `goalkeeper_ids` is supplied.** Pre-1.10.0 the converter had no
  mechanism to surface GK actions, leaving downstream `add_gk_role` /
  `add_pre_shot_gk_context` enrichments at 100% NULL on every Metrica
  match in production.

### Notes
- This release closes the upstream gap that surfaced during
  luxury-lakehouse PR-LL2 production deploy (2026-04-29): post-deploy
  validation found 100% NULL `gk_role` and `defending_gk_player_id` on
  IDSSE (2,522 rows) and Metrica (5,839 rows) sources. With silly-kicks
  1.10.0, downstream lakehouse can re-run `apply_spadl_enrichments`
  against IDSSE + Metrica with non-NULL GK coverage (handled by
  separate lakehouse PR-LL3).
- Behaviour change for IDSSE consumers: bronze.spadl_actions row count
  per IDSSE match will increase materially (every Play event now
  surfaces, plus throwOut/punt rows now produce 2 actions each). This
  is the intended fix; downstream aggregation may need to re-baseline.
- Wyscout converter unchanged — `goalkeeper_ids` was already present
  from 1.0.0.
- Atomic-SPADL `coverage_metrics` parity is queued as tech debt
  (atomic uses 33 action types vs standard's 23; deferred until a
  consumer asks). Tracked in `TODO.md ## Tech Debt`.

```

- [ ] **Step 10.3: Update TODO.md**

Edit `TODO.md`. Find the `## Open PRs` table row for PR-S10 and DELETE it (it's shipped now). Find the existing PR-S10 row:

```markdown
| PR-S10 | Medium-Large | `add_possessions` algorithmic precision improvement | ... |
```

Replace with PR-S11:

```markdown
| PR-S11 | Medium-Large | `add_possessions` algorithmic precision improvement | Close the precision gap from ~42% toward 60-70% via brief-opposing-action merge rule, defensive-action class, and/or spatial continuity check. Plus re-measure `max_gap_seconds` parameter sweep using the `boundary_metrics` utility before changing the default. New parameters likely: `merge_brief_opposing_actions`, `brief_action_window_seconds`. Atomic-SPADL counterpart must mirror any semantic change. Re-numbered from PR-S10 (which became the GK converter coverage parity work, shipped in 1.10.0). |
```

In the `## Tech Debt` section, append a new row:

```markdown
| C-1 | Low | Atomic-SPADL `coverage_metrics` parity | Atomic-SPADL has its own action vocabulary (33 types vs standard's 23). `silly_kicks.atomic.spadl.coverage_metrics` would mirror the standard utility added in 1.10.0. Defer until a concrete consumer ask — same disposition as `add_possessions` atomic parity that took 4 cycles to materialize. |
```

- [ ] **Step 10.4: Update the C4 architecture diagram**

Edit `docs/c4/architecture.dsl:15`. Find the `spadl = container` description string and add `coverage_metrics` to the public-helper enumeration:

```diff
-            spadl = container "silly_kicks.spadl" "SPADL conversion + post-conversion enrichments: 23 action types, 6 dedicated DataFrame converters with preserve_native passthrough (StatsBomb, Opta, Wyscout, Sportec, Metrica) plus a kloppy gateway covering the same providers, output coords clamped to [0, 105] x [0, 68] AND unified to canonical 'all-actions-LTR' SPADL orientation across all paths, ConversionReport audit; public enrichment helpers (add_names, add_possessions, GK analytics suite — gk_role / distribution_metrics / pre_shot_gk_context); boundary_metrics utility for validating add_possessions output against provider-native possession_id" "Python" "Library"
+            spadl = container "silly_kicks.spadl" "SPADL conversion + post-conversion enrichments: 23 action types, 6 dedicated DataFrame converters with preserve_native passthrough (StatsBomb, Opta, Wyscout, Sportec, Metrica) plus a kloppy gateway covering the same providers, output coords clamped to [0, 105] x [0, 68] AND unified to canonical 'all-actions-LTR' SPADL orientation across all paths, ConversionReport audit; public enrichment helpers (add_names, add_possessions, GK analytics suite — gk_role / distribution_metrics / pre_shot_gk_context); boundary_metrics + coverage_metrics utilities for validating add_possessions output and per-action-type coverage" "Python" "Library"
```

- [ ] **Step 10.5: Regenerate the C4 HTML diagram (skip if /final-review handles it)**

Note: per the `final-review` skill, C4 regeneration is part of the pre-commit ritual. Skip this step manually here — it'll be done in Task 12 by `/final-review`.

- [ ] **Step 10.6: Stage release artifacts**

```bash
git add pyproject.toml CHANGELOG.md TODO.md docs/c4/architecture.dsl
git status
```

---

## Task 11: Verification gates

**Files:** none (run-only)

Run the same gates that CI will run, against the staged + working-tree state, before the single commit.

- [ ] **Step 11.1: Re-pin CI tools**

```bash
uv pip install --upgrade ruff==0.15.7 pyright==1.1.395 pandas-stubs==2.3.3.260113
```

Expected: succeeds (already installed in Task 0; this is a defensive re-check).

- [ ] **Step 11.2: Lint clean**

```bash
uv run ruff check silly_kicks/ tests/ scripts/
```

Expected: zero errors.

If any errors surface (e.g., S105 hardcoded-password false positives on string literals like "throwOut" — though these should be filtered out by the existing per-file-ignores), fix inline.

- [ ] **Step 11.3: Format check**

```bash
uv run ruff format --check silly_kicks/ tests/ scripts/
```

If anything fails, run `uv run ruff format silly_kicks/ tests/ scripts/` to auto-fix, then re-stage the affected files.

- [ ] **Step 11.4: Pyright type check**

```bash
uv run pyright silly_kicks/ 2>&1 | tail -10
```

Expected: zero errors. If any surface (likely candidates: numpy mask broadcasting, set type comprehension narrowing), fix inline.

- [ ] **Step 11.5: Full test suite**

```bash
uv run pytest tests/ -m "not e2e" -v --tb=short -q 2>&1 | tail -10
```

Expected: ~620 passed (~560 baseline + ~60 new). Zero failures, ~4 skipped (pre-existing).

If any tests fail at this stage, STOP and investigate.

- [ ] **Step 11.6: Optional — run e2e tests too (the WorldCup-2018 and PR-S8 fixture-based tests)**

```bash
uv run pytest tests/ -v --tb=short -q 2>&1 | tail -10
```

Expected: same as Step 11.5 with no skips on the e2e markers (1.9.0 dropped most). If anything in the prediction-pipeline or boundary-metrics e2e tests regresses, STOP.

- [ ] **Step 11.7: Stage any auto-fixed format changes**

```bash
git status
```

If `ruff format` modified any files in Step 11.3, re-stage them:

```bash
git add silly_kicks/ tests/ scripts/
```

---

## Task 12: /final-review + 5 user-gated steps (commit, push, PR, merge, tag)

**Files:** the spec + plan + all preceding changes — bundled into ONE commit.

Per `feedback_commit_policy` memory and the narrowed hook from PR-S8 (only `git commit` is sentinel-gated; push/PR/merge/tag are chat-only). User must explicitly approve at each gate.

- [ ] **Step 12.1: Stage the spec + this plan into the same commit**

```bash
git add docs/superpowers/specs/2026-04-29-gk-converter-coverage-parity-design.md docs/superpowers/plans/2026-04-29-gk-converter-coverage-parity.md
git status
```

Expected: 22+ files staged (per the file structure at the top of this plan). Verify the list looks complete:

- `silly_kicks/spadl/utils.py`
- `silly_kicks/spadl/__init__.py`
- `silly_kicks/spadl/sportec.py`
- `silly_kicks/spadl/metrica.py`
- `silly_kicks/spadl/statsbomb.py`
- `silly_kicks/spadl/opta.py`
- `tests/spadl/test_coverage_metrics.py` (new)
- `tests/spadl/test_sportec.py`
- `tests/spadl/test_metrica.py`
- `tests/spadl/test_statsbomb.py`
- `tests/spadl/test_opta.py`
- `tests/spadl/test_cross_provider_parity.py` (new)
- `tests/datasets/idsse/sample_match.parquet` (new)
- `tests/datasets/idsse/README.md` (new)
- `tests/datasets/metrica/sample_match.parquet` (new)
- `tests/datasets/metrica/README.md` (new)
- `scripts/extract_provider_fixtures.py` (new)
- `pyproject.toml`
- `CHANGELOG.md`
- `TODO.md`
- `docs/c4/architecture.dsl`
- `docs/c4/architecture.html` (regen during /final-review)
- `docs/superpowers/specs/2026-04-29-gk-converter-coverage-parity-design.md` (existing untracked)
- `docs/superpowers/plans/2026-04-29-gk-converter-coverage-parity.md` (this file)

- [ ] **Step 12.2: Run `/final-review` skill (regenerates C4 HTML, runs all gates)**

Invoke the `mad-scientist-skills:final-review` skill via the Skill tool. The skill regenerates the C4 architecture HTML from the .dsl, runs ruff / pyright / pytest, and reviews documentation consistency. It will surface any remaining gaps.

If `/final-review` modifies any files (likely just `docs/c4/architecture.html`), re-stage them:

```bash
git add docs/c4/architecture.html
```

Address any other findings from `/final-review` inline.

- [ ] **Step 12.3: Present summary to user; await explicit "approved" + sentinel touch**

Present to the user (in chat):

```
✅ All 12 task gates green:
- coverage_metrics utility shipped + tested
- sportec Bug 1 (Play recognition) fixed + 7 tests
- sportec Bug 2 (throwOut/punt synthesis) fixed + 10 tests
- sportec goalkeeper_ids supplementary signal + 6 tests
- metrica Bug 3 (goalkeeper_ids primary mechanism) fixed + 12 tests
- statsbomb + opta goalkeeper_ids no-op + 6 tests
- cross-provider parity meta-test + 10 tests
- production-shape fixtures vendored (idsse + metrica)
- /final-review pass: ruff + format + pyright + pytest all green
- C4 architecture diagram regenerated

Ready for the single commit. Per silly-kicks policy:
1. Run `!touch ~/.claude-git-approval` to release the sentinel
2. Reply "approved" to authorize the commit
```

WAIT for explicit user approval + sentinel touch.

- [ ] **Step 12.4: Single commit on user approval (sentinel-gated)**

After user approval + sentinel:

```bash
git commit -m "$(cat <<'EOF'
feat(spadl): GK converter coverage parity — sportec Pass→Play + throwOut/punt synthesis + Metrica goalkeeper_ids + coverage_metrics utility — silly-kicks 1.10.0

Closes 3 distinct production bugs surfaced by luxury-lakehouse PR-LL2's
post-deploy validation:

1. sportec Pass→Play — DFL bronze never emits "Pass"; the actual
   event_type for pass-class events is "Play". Pre-1.10.0 ALL IDSSE
   pass-class events silently dropped to non_action. Fix restructures
   the dispatch so Play events with no GK qualifier are pass-class.

2. sportec throwOut/punt — GK distribution qualifiers were unmapped.
   Each source row now synthesizes 2 SPADL actions: keeper_pick_up + pass
   for throwOut, keeper_pick_up + goalkick for punt.

3. Metrica goalkeeper_ids — Metrica's source format lacks native GK
   markers anywhere. New goalkeeper_ids: set[str] | None parameter
   enables conservative GK routing (PASS by GK → synth, RECOVERY by GK
   → keeper_pick_up, CHALLENGE-AERIAL-WON by GK → keeper_claim).

Plus institutionalises the test discipline that would have caught these
bugs:

- silly_kicks.spadl.coverage_metrics(*, actions, expected_action_types)
  public utility (mirrors PR-S8 boundary_metrics shape)
- Production-shape fixtures: tests/datasets/{idsse,metrica}/sample_match.parquet
- scripts/extract_provider_fixtures.py — Databricks pull (idsse) +
  kloppy-fixture subset (metrica)
- Cross-provider parity meta-test parametrized over all 5 DataFrame
  converters

API symmetry: goalkeeper_ids accepted on all 5 DataFrame converters
(statsbomb / opta = no-op; wyscout = unchanged from 1.0.0; sportec =
supplementary; metrica = primary).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Run `git status` after to verify the commit succeeded.

- [ ] **Step 12.5: User approval → push -u**

WAIT for chat approval. Then:

```bash
git push -u origin feat/gk-converter-coverage-parity
```

- [ ] **Step 12.6: User approval → gh pr create**

WAIT for chat approval. Then:

```bash
gh pr create --title "feat(spadl): GK converter coverage parity — silly-kicks 1.10.0" --body "$(cat <<'EOF'
## Summary

Closes 3 distinct production bugs in silly-kicks's sportec + metrica converters that left lakehouse production with 100% NULL `gk_role` + `defending_gk_player_id` coverage on IDSSE (2,522 rows) and Metrica (5,839 rows). Surfaced by luxury-lakehouse PR-LL2's post-deploy validation (2026-04-29).

- **Bug 1 (sportec):** `Pass` → `Play` event_type recognition — pre-1.10.0 all DFL pass-class events on IDSSE silently dropped to non_action.
- **Bug 2 (sportec):** `throwOut` / `punt` qualifier synthesis — pre-1.10.0 these GK distribution qualifiers were unmapped; now synthesize keeper_pick_up + pass / goalkick.
- **Bug 3 (metrica):** New `goalkeeper_ids: set[str] | None` parameter — Metrica's source format lacks native GK markers; this is the only mechanism that can surface keeper_* actions on Metrica.

Plus institutionalises the test discipline:
- Public `coverage_metrics(*, actions, expected_action_types)` utility
- Production-shape fixtures vendored under `tests/datasets/{idsse,metrica}/`
- Cross-provider parity meta-test parametrized over all 5 DataFrame converters

API symmetry: `goalkeeper_ids` accepted on every DataFrame converter (no-op on statsbomb / opta; unchanged on wyscout from 1.0.0).

Spec: `docs/superpowers/specs/2026-04-29-gk-converter-coverage-parity-design.md`
Plan: `docs/superpowers/plans/2026-04-29-gk-converter-coverage-parity.md`

## Test plan

- [x] coverage_metrics — 14 new tests (Contract, Correctness, Degenerate)
- [x] sportec Play recognition — 7 new tests
- [x] sportec throwOut/punt synthesis — 10 new tests
- [x] sportec goalkeeper_ids supplementary — 6 new tests
- [x] metrica goalkeeper_ids routing — 12 new tests
- [x] statsbomb + opta goalkeeper_ids no-op — 6 new tests
- [x] cross-provider parity meta-test — 10 new tests
- [x] All pre-existing tests pass
- [x] CI matrix (ubuntu 3.10/3.11/3.12 + windows 3.12) all green before merge

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Capture the PR URL from the output and present it to the user.

- [ ] **Step 12.7: Wait for CI green → user approval → merge**

Watch CI status:

```bash
gh pr checks --watch
```

When green, WAIT for user approval. Then:

```bash
gh pr merge --admin --squash --delete-branch
```

- [ ] **Step 12.8: User approval → tag v1.10.0 → PyPI auto-publish**

WAIT for user approval. Then on `main`:

```bash
git checkout main
git pull origin main
git tag v1.10.0
git push origin v1.10.0
```

The PyPI auto-publish workflow fires on tag push.

- [ ] **Step 12.9: Verify PyPI publish + final state**

After ~2-5 min:

```bash
gh run list --workflow=publish.yml --limit 3
```

Or visit `https://pypi.org/project/silly-kicks/1.10.0/` to confirm.

Update memories:

- `project_release_state.md` → 1.10.0 current
- `project_followup_prs.md` → PR-S10 SHIPPED in 1.10.0; PR-S11 (was-PR-S10 add_possessions improvement) is now the next-in-queue

Done.

---

## Self-review checklist (run before presenting plan to user)

1. **Spec coverage:** every section of the spec maps to at least one task above. ✓
   - § 2 Goal 1 (Bug 1) → Task 3
   - § 2 Goal 2 (Bug 2) → Task 4
   - § 2 Goal 3 (Bug 3) → Task 6
   - § 2 Goal 4 (API symmetry) → Tasks 5, 6, 7
   - § 2 Goal 5 (coverage_metrics) → Task 1
   - § 2 Goal 6 (production-shape fixtures) → Task 2
   - § 2 Goal 7 (cross-provider parity gate) → Task 8
   - § 2 Goal 8 (vocabulary documentation) → Task 9
   - § 5 Fixture sourcing → Task 2 (Option A for IDSSE, Option B for Metrica — verified by probe)
   - § 7 Verification gates → Task 11
   - § 8 Commit cycle → Task 12

2. **Spec discrepancy noted:** Spec § 4.3 says Bug 1 is a "single-line replacement"; plan corrects this to a multi-line restructure (Task 3) with rationale at the top of this plan. ✓

3. **Type / signature consistency:** every reference to `goalkeeper_ids` uses the same shape `set[str] | None = None` (sportec/metrica) or `set[int] | None = None` (statsbomb/opta), keyword-only after `*,`. Wyscout's pre-existing positional-or-keyword shape is intentionally unchanged. ✓

4. **No placeholders:** every step contains actual code, exact paths, exact commands. ✓

5. **Single-commit ritual:** every task ends with `git add` (stage), no intermediate `git commit`. The single commit happens in Task 12. ✓

6. **Hook scope honoured:** sentinel-gated only on `git commit` (Task 12.4). Push / PR / merge / tag (Tasks 12.5-12.8) are chat-only approval. ✓

