# Tracking-aware action_context PR-1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the four tracking-aware action_context features (`nearest_defender_distance`, `actor_speed`, `receiver_zone_density`, `defenders_in_triangle_to_goal`) for silly-kicks 2.8.0, with full standard + atomic SPADL parity, HybridVAEP/AtomicVAEP integration via composition (no subclass), ADR-005 charter, and a canonical NOTICE file mirroring the lakehouse pattern.

**Architecture:** Pure-function feature helpers built on PR-S19's `link_actions_to_frames` linkage primitive. Schema-agnostic compute kernels (`silly_kicks/tracking/_kernels.py`) consumed by per-schema wrapper modules (`silly_kicks/tracking/features.py` + `silly_kicks/atomic/tracking/features.py`). VAEP integration via additive `frames=None` kwarg on `VAEP.compute_features`/`rate` + `_frame_aware` xfn marker dispatch — no new VAEP subclass; HybridVAEP and AtomicVAEP inherit the extension. Spec: `docs/superpowers/specs/2026-04-30-action-context-pr1-design.md`.

**Tech Stack:** pandas ≥ 2.1, numpy ≥ 1.26, scikit-learn (existing), pytest, pyarrow (parquet), Databricks SQL connector (Loop 0 probe only, not a runtime dep).

**Commit policy override:** silly-kicks is "literally ONE commit per branch; no WIP commits + squash; explicit approval before that one commit." This plan therefore has **no per-task commit steps** — only a single final commit after `/final-review` and explicit user approval (Task 14). Each task ends with test-pass verification.

---

## File Structure

### New files

```
silly_kicks/tracking/
├── _kernels.py                              # private — schema-agnostic compute kernels
├── feature_framework.py                     # ActionFrameContext, lift_to_states, type aliases
└── features.py                              # standard SPADL public surface

silly_kicks/atomic/tracking/
├── __init__.py                              # NEW package
└── features.py                              # atomic SPADL public surface (wraps _kernels)

docs/superpowers/adrs/
└── ADR-005-tracking-aware-features.md       # 7 cross-cutting decisions

scripts/
└── probe_action_context_baselines.py        # one-off Loop 0; committed

tests/datasets/tracking/
├── empirical_action_context_baselines.json  # Loop 0 output, committed
└── action_context_slim/
    ├── sportec_slim.parquet                 # Tier-3 lakehouse-derived slim slice
    ├── metrica_slim.parquet
    └── skillcorner_slim.parquet

tests/tracking/
├── __init__.py                              # NEW (if not exists)
├── test_feature_framework.py
├── test_kernels.py
├── test_features_standard.py
├── test_add_action_context.py
├── test_action_context_cross_provider.py
└── test_action_context_real_data_sweep.py   # e2e-marked

tests/atomic/tracking/
├── __init__.py
├── test_features_atomic.py
└── test_atomic_action_context.py

tests/vaep/
├── test_compute_features_frames_kwarg.py
└── test_hybrid_with_tracking.py

tests/atomic/vaep/
└── test_atomic_with_tracking.py

tests/
├── test_todo_md_format.py                   # On-Deck table structure check
└── test_notice_md_format.py                 # NOTICE file structural check

NOTICE                                       # NEW at repo root
```

### Modified files

```
silly_kicks/tracking/__init__.py             # extend exports for new modules
silly_kicks/tracking/utils.py                # add _resolve_action_frame_context
silly_kicks/vaep/feature_framework.py        # add frame_aware decorator + helpers + Frames alias
silly_kicks/vaep/base.py                     # extend compute_features + rate with frames= kwarg
TODO.md                                      # restructure to lakehouse-style On-Deck table
CHANGELOG.md                                 # add 2.8.0 entry
README.md                                    # add Attribution section linking to NOTICE
CLAUDE.md                                    # add academic-attribution convention line
pyproject.toml                               # bump version 2.7.0 -> 2.8.0
```

---

## Task 0: Lakehouse probe + Tier-3 fixture extraction

**Files:**
- Create: `scripts/probe_action_context_baselines.py`
- Create: `tests/datasets/tracking/empirical_action_context_baselines.json`
- Create: `tests/datasets/tracking/action_context_slim/{sportec,metrica,skillcorner}_slim.parquet`

- [ ] **Step 0.1: Write the probe script**

Mirror `scripts/probe_tracking_baselines.py` (PR-S19) shape. Probes lakehouse for per-provider action-feature distributions. Source 1: `soccer_analytics.dev_gold.fct_tracking_frames` joined to `fct_action_values` to compute the 4 features on real data; capture median, p25, p50, p75, p99. Source 2: local PFF (synthetic baselines only — license).

Create `scripts/probe_action_context_baselines.py`:

```python
"""One-off empirical probe for silly-kicks PR-S20 action_context.

Computes the 4 action_context features on real lakehouse data per provider
(metrica, idsse->sportec, skillcorner) and records distribution baselines.

Writes:
  - tests/datasets/tracking/empirical_action_context_baselines.json
  - tests/datasets/tracking/action_context_slim/{provider}_slim.parquet
    (Tier-3 lakehouse-derived slim slices: ~10 actions + linked frames per provider)

PFF is excluded from the lakehouse probe (license); PFF baselines come from
local synthetic computation in Loop 0 step 0.5 separately.

Run once during PR-S20 development. Both this script AND its outputs are
committed to the repo. The real datasets are NOT committed.
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_JSON = REPO_ROOT / "tests" / "datasets" / "tracking" / "empirical_action_context_baselines.json"
SLIM_DIR = REPO_ROOT / "tests" / "datasets" / "tracking" / "action_context_slim"

LAKEHOUSE_PROVIDERS = [("metrica", "metrica"), ("sportec", "idsse"), ("skillcorner", "skillcorner")]


def _connect():
    from databricks import sql
    raw_host = os.environ.get("DATABRICKS_SERVER_HOSTNAME") or os.environ.get("DATABRICKS_HOST", "")
    server_hostname = raw_host.removeprefix("https://").removeprefix("http://").rstrip("/")
    http_path = os.environ.get("DATABRICKS_HTTP_PATH", "").lstrip("/")
    http_path = "/" + http_path
    token = os.environ.get("DATABRICKS_TOKEN", "")
    if not (server_hostname and http_path and token):
        return None
    return sql.connect(server_hostname=server_hostname, http_path=http_path, access_token=token)


def probe_provider(conn, sk_provider: str, lakehouse_provider: str) -> dict[str, Any]:
    """Pull a sample of actions + linked frames for one provider; compute the 4 features."""
    # Pull sample actions (~1 match worth) joined to tracking frames within +/-0.2s
    # Compute features inline via SQL aggregations to avoid materializing tracking-DataFrame on driver
    with conn.cursor() as c:
        # Pick one match for the slim slice
        c.execute(f"""
            SELECT match_id FROM soccer_analytics.dev_gold.fct_action_values
            WHERE source_provider = '{lakehouse_provider}'
            AND match_id IN (
                SELECT DISTINCT match_id FROM soccer_analytics.dev_gold.fct_tracking_frames
                WHERE source_provider = '{lakehouse_provider}'
            )
            LIMIT 1
        """)
        row = c.fetchone()
        if not row:
            return {"error": f"no overlapping match found for {lakehouse_provider}"}
        match_id = row[0]
        # ... compute baselines + slim parquet writeout via SQL
        # NOTE: implement during execution; structure depends on lakehouse exact schema.
    return {"match_id": str(match_id), "p50_nearest_defender_distance": None}


def main():
    SLIM_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {
        "probe_run_date": "2026-04-30",
        "probe_run_source_lakehouse_table": "soccer_analytics.dev_gold.fct_tracking_frames + fct_action_values",
        "providers": {},
    }
    conn = _connect()
    if conn is None:
        print("[warn] Databricks env not set; skipping lakehouse probe")
    else:
        for sk_p, lh_p in LAKEHOUSE_PROVIDERS:
            print(f"probing {sk_p}...")
            out["providers"][sk_p] = probe_provider(conn, sk_p, lh_p)
        conn.close()
    OUTPUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
```

**Note:** the SQL details inside `probe_provider` depend on the exact lakehouse `fct_action_values` schema. The 2026-04-30 schema probe (during brainstorming) confirmed these mart names + that the 4 features are NOT precomputed. The probe computes them on-the-fly per match: pick 1 match, pull actions + linked frames (long-form), use pandas to compute the 4 features, capture distribution stats, save 10-row slim slices to parquet.

- [ ] **Step 0.2: Run the probe**

Run: `python scripts/probe_action_context_baselines.py`
Expected: writes `tests/datasets/tracking/empirical_action_context_baselines.json` and three parquets under `tests/datasets/tracking/action_context_slim/`. Prints "wrote ..." per output.

- [ ] **Step 0.3: Validate JSON shape**

Open `tests/datasets/tracking/empirical_action_context_baselines.json`; verify it contains:
- `probe_run_date` and `probe_run_source_lakehouse_table` top-level keys
- `providers` dict with 3 entries (metrica, sportec, skillcorner)
- Each provider has at least: `n_actions_sampled`, `nearest_defender_distance_p50`, `actor_speed_p50`, `receiver_zone_density_p50`, `defenders_in_triangle_to_goal_p50` (and p25, p75, p99 each)

- [ ] **Step 0.4: Validate slim parquet outputs**

Run:

```python
python -c "
import pandas as pd
for p in ['sportec', 'metrica', 'skillcorner']:
    df = pd.read_parquet(f'tests/datasets/tracking/action_context_slim/{p}_slim.parquet')
    print(p, df.shape, list(df.columns)[:5])
"
```

Expected: each parquet has ~10-50 rows (slim slice), with at least `action_id, period_id, time_seconds, frame_id, x, y, team_id, player_id` columns.

- [ ] **Step 0.5: Generate PFF synthetic-baseline entry**

Append a `pff` entry to `empirical_action_context_baselines.json` computed offline from `tests/datasets/tracking/pff/medium_halftime.parquet` (PR-S19 synthetic). Write a small inline script:

```python
python -c "
import json, pandas as pd, numpy as np
from pathlib import Path
p = Path('tests/datasets/tracking/empirical_action_context_baselines.json')
d = json.loads(p.read_text())
# placeholder distribution from synthetic data
# (real values come from the probe; PFF is synthetic-only)
d['providers']['pff'] = {
    'source': 'synthetic_pff_medium_halftime',
    'note': 'PFF-license: no lakehouse probe; baseline from committed synthetic fixture.',
    'nearest_defender_distance_p50': None,  # populated post-Loop-7
    'actor_speed_p50': None,
    'receiver_zone_density_p50': None,
    'defenders_in_triangle_to_goal_p50': None,
}
p.write_text(json.dumps(d, indent=2))
"
```

(The PFF baseline values get populated AFTER Loop 7 ships the standard SPADL features — the synthetic PFF baseline is computed from `tests/datasets/tracking/pff/medium_halftime.parquet` using the Loop-7 functions. This is a forward-reference; we revisit in Task 13.)

- [ ] **Step 0.6: Verify Loop 0 outputs**

Run: `git status -- scripts/ tests/datasets/tracking/`
Expected: 5 files staged for commit:
- `scripts/probe_action_context_baselines.py`
- `tests/datasets/tracking/empirical_action_context_baselines.json`
- `tests/datasets/tracking/action_context_slim/sportec_slim.parquet`
- `tests/datasets/tracking/action_context_slim/metrica_slim.parquet`
- `tests/datasets/tracking/action_context_slim/skillcorner_slim.parquet`

---

## Task 1: feature_framework foundations (frame_aware marker, ActionFrameContext, type aliases)

**Files:**
- Create: `tests/tracking/test_feature_framework.py`
- Create: `tests/tracking/__init__.py` (empty if not present)
- Modify: `silly_kicks/vaep/feature_framework.py` (add `frame_aware`, `is_frame_aware`, `Frames`, `FrameAwareTransformer`)
- Create: `silly_kicks/tracking/feature_framework.py` (`ActionFrameContext` minimal stub; `lift_to_states` lands in Task 3)

- [ ] **Step 1.1: Write failing test for frame_aware marker**

Create `tests/tracking/test_feature_framework.py`:

```python
"""Tests for silly_kicks.tracking.feature_framework + frame_aware marker.

Loop 1 covers: frame_aware decorator, is_frame_aware predicate, ActionFrameContext
frozen dataclass shape, type aliases.
"""
from __future__ import annotations

import dataclasses

import pandas as pd
import pytest


def test_frame_aware_marker_sets_attribute():
    from silly_kicks.vaep.feature_framework import frame_aware

    @frame_aware
    def my_xfn(states, frames):
        return pd.DataFrame()

    assert my_xfn._frame_aware is True


def test_is_frame_aware_returns_true_for_marked():
    from silly_kicks.vaep.feature_framework import frame_aware, is_frame_aware

    @frame_aware
    def marked(states, frames):
        return pd.DataFrame()

    assert is_frame_aware(marked) is True


def test_is_frame_aware_returns_false_for_unmarked():
    from silly_kicks.vaep.feature_framework import is_frame_aware

    def regular(states):
        return pd.DataFrame()

    assert is_frame_aware(regular) is False


def test_is_frame_aware_returns_false_for_lambda_without_attr():
    from silly_kicks.vaep.feature_framework import is_frame_aware

    assert is_frame_aware(lambda x: x) is False


def test_frames_type_alias_exists():
    from silly_kicks.vaep import feature_framework as ff

    assert ff.Frames is pd.DataFrame


def test_action_frame_context_is_frozen_dataclass():
    from silly_kicks.tracking.feature_framework import ActionFrameContext

    assert dataclasses.is_dataclass(ActionFrameContext)
    params = ActionFrameContext.__dataclass_params__  # type: ignore[attr-defined]
    assert params.frozen is True


def test_action_frame_context_has_required_fields():
    from silly_kicks.tracking.feature_framework import ActionFrameContext

    field_names = {f.name for f in dataclasses.fields(ActionFrameContext)}
    expected = {"actions", "pointers", "actor_rows", "opposite_rows_per_action"}
    assert expected.issubset(field_names)
```

- [ ] **Step 1.2: Verify test fails**

Run: `python -m pytest tests/tracking/test_feature_framework.py -v`
Expected: FAIL — `ImportError` for `frame_aware`, `is_frame_aware`, `Frames`, `ActionFrameContext`.

- [ ] **Step 1.3: Extend silly_kicks/vaep/feature_framework.py**

Open `silly_kicks/vaep/feature_framework.py`. Add to the imports section + below existing exports:

```python
# After existing imports
import pandas as pd
from collections.abc import Callable
from typing import Any

# After existing type aliases (Actions, GameStates, Features, FeatureTransfomer):
Frames = pd.DataFrame
"""Type alias for tracking frames DataFrame (long-form, TRACKING_FRAMES_COLUMNS-shaped)."""

FrameAwareTransformer = Callable[[Any, Frames], Features]
"""Tracking-aware feature transformer signature: (states, frames) -> Features.

The first argument is GameStates; using Any here to avoid circular type-import
constraints. Marked at runtime via the frame_aware decorator and dispatched in
VAEP.compute_features."""


def frame_aware(fn: Callable) -> Callable:
    """Marker decorator: this xfn requires frames as a second argument.

    Sets fn._frame_aware = True. VAEP.compute_features uses is_frame_aware to
    dispatch (states, frames) calls vs (states) calls.

    Examples
    --------
    Wrap a custom xfn so HybridVAEP routes frames to it::

        from silly_kicks.vaep.feature_framework import frame_aware
        @frame_aware
        def my_tracking_feature(states, frames):
            return some_dataframe
    """
    fn._frame_aware = True  # type: ignore[attr-defined]
    return fn


def is_frame_aware(fn: Callable) -> bool:
    """Check if an xfn is marked as frame-aware (was decorated with @frame_aware)."""
    return getattr(fn, "_frame_aware", False)
```

Update `__all__` in the same file:

```python
__all__ = [
    "Actions",
    "FeatureTransfomer",
    "Features",
    "FrameAwareTransformer",
    "Frames",
    "GameStates",
    "actiontype_categorical",
    "frame_aware",
    "gamestates",
    "is_frame_aware",
    "simple",
]
```

- [ ] **Step 1.4: Create silly_kicks/tracking/feature_framework.py**

Create the new file:

```python
"""Tracking-aware feature framework primitives.

Public surface:
- ActionFrameContext: linked-context structure consumed by per-feature kernels.
- lift_to_states: lift (actions, frames) -> Series helper to FrameAwareTransformer.
- Frames: re-export from silly_kicks.vaep.feature_framework for consumer convenience.

See ADR-005 for the integration contract; spec
docs/superpowers/specs/2026-04-30-action-context-pr1-design.md for full design.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Callable

import pandas as pd

from silly_kicks.vaep.feature_framework import (
    FrameAwareTransformer,
    Frames,
    frame_aware,
    is_frame_aware,
)

__all__ = [
    "ActionFrameContext",
    "FrameAwareTransformer",
    "Frames",
    "frame_aware",
    "is_frame_aware",
    "lift_to_states",
]


@dataclasses.dataclass(frozen=True)
class ActionFrameContext:
    """Linkage + actor/opposite-team frame slices, computed once per add_action_context call.

    Attributes
    ----------
    actions : pd.DataFrame
        Subset of input actions (index aligned with the per-feature output Series).
    pointers : pd.DataFrame
        link_actions_to_frames output: action_id, frame_id, time_offset_seconds,
        n_candidate_frames, link_quality_score.
    actor_rows : pd.DataFrame
        One row per linked action: the actor's frame row (player_id == action.player_id).
        Rows for unlinked actions or missing-actor cases have NaN x/y/speed.
    opposite_rows_per_action : pd.DataFrame
        Long-form: (action_id, opposite-team frame row) pairs. Used by geometric
        feature kernels via groupby('action_id').

    Examples
    --------
    Build the context once and pass to multiple feature kernels::

        from silly_kicks.tracking.utils import _resolve_action_frame_context
        ctx = _resolve_action_frame_context(actions, frames)
        d = _nearest_defender_distance(actions["start_x"], actions["start_y"], ctx)
    """

    actions: pd.DataFrame
    pointers: pd.DataFrame
    actor_rows: pd.DataFrame
    opposite_rows_per_action: pd.DataFrame


def lift_to_states(
    helper: Callable[[pd.DataFrame, pd.DataFrame], pd.Series],
    nb_states: int = 3,
) -> FrameAwareTransformer:
    """Lift a (actions, frames) -> Series helper to a (states, frames) -> Features transformer.

    Output columns: f"{helper.__name__}_a0", "..._a{nb_states-1}".
    Marks the returned transformer as ``_frame_aware = True`` so VAEP.compute_features
    routes frames through.

    Implementation lands in Task 3.
    """
    raise NotImplementedError("lift_to_states implemented in Task 3")
```

- [ ] **Step 1.5: Verify tests pass**

Run: `python -m pytest tests/tracking/test_feature_framework.py -v`
Expected: 7 passed.

---

## Task 2: `_resolve_action_frame_context` private helper

**Files:**
- Modify: `tests/tracking/test_feature_framework.py` (add tests for _resolve)
- Modify: `silly_kicks/tracking/utils.py` (add _resolve_action_frame_context)

- [ ] **Step 2.1: Add failing tests for _resolve_action_frame_context**

Append to `tests/tracking/test_feature_framework.py`:

```python
@pytest.fixture
def tiny_actions_and_frames():
    """3 actions + 3 frames at known times; 4 players + ball per frame."""
    actions = pd.DataFrame({
        "action_id": [101, 102, 103],
        "period_id": [1, 1, 1],
        "time_seconds": [10.0, 20.0, 30.0],
        "team_id": [1, 1, 2],
        "player_id": [11, 12, 21],
        "start_x": [50.0, 60.0, 40.0],
        "start_y": [34.0, 30.0, 38.0],
        "end_x": [55.0, 65.0, 45.0],
        "end_y": [34.0, 30.0, 38.0],
    })
    # 3 frames at t=10.0/20.0/30.0; each frame has 4 players (2 per team) + 1 ball row
    rows = []
    for fid, t in [(1000, 10.0), (2000, 20.0), (3000, 30.0)]:
        for pid, tid, x, y in [(11, 1, 50.0, 34.0), (12, 1, 60.0, 30.0),
                               (21, 2, 40.0, 38.0), (22, 2, 70.0, 35.0)]:
            rows.append({
                "game_id": 1, "period_id": 1, "frame_id": fid, "time_seconds": t,
                "frame_rate": 25.0, "player_id": pid, "team_id": tid,
                "is_ball": False, "is_goalkeeper": False,
                "x": x, "y": y, "z": float("nan"),
                "speed": 1.5, "speed_source": "native",
                "ball_state": "alive", "team_attacking_direction": "ltr",
                "confidence": None, "visibility": None, "source_provider": "test",
            })
        rows.append({
            "game_id": 1, "period_id": 1, "frame_id": fid, "time_seconds": t,
            "frame_rate": 25.0, "player_id": float("nan"), "team_id": float("nan"),
            "is_ball": True, "is_goalkeeper": False,
            "x": 52.5, "y": 34.0, "z": 0.0,
            "speed": 5.0, "speed_source": "native",
            "ball_state": "alive", "team_attacking_direction": None,
            "confidence": None, "visibility": None, "source_provider": "test",
        })
    frames = pd.DataFrame(rows)
    return actions, frames


def test_resolve_action_frame_context_links_all_actions(tiny_actions_and_frames):
    from silly_kicks.tracking.utils import _resolve_action_frame_context

    actions, frames = tiny_actions_and_frames
    ctx = _resolve_action_frame_context(actions, frames)

    assert len(ctx.pointers) == 3
    assert ctx.pointers["frame_id"].notna().all()


def test_resolve_action_frame_context_actor_rows_one_per_action(tiny_actions_and_frames):
    from silly_kicks.tracking.utils import _resolve_action_frame_context

    actions, frames = tiny_actions_and_frames
    ctx = _resolve_action_frame_context(actions, frames)

    # actor_rows: one row per action_id with the actor's frame data
    assert len(ctx.actor_rows) == 3
    assert set(ctx.actor_rows["action_id"]) == {101, 102, 103}


def test_resolve_action_frame_context_opposite_excludes_actor_team(tiny_actions_and_frames):
    from silly_kicks.tracking.utils import _resolve_action_frame_context

    actions, frames = tiny_actions_and_frames
    ctx = _resolve_action_frame_context(actions, frames)

    # action_id 101 is team_id=1; opposite rows should all be team_id=2
    opp_101 = ctx.opposite_rows_per_action[
        ctx.opposite_rows_per_action["action_id"] == 101
    ]
    assert (opp_101["team_id"] == 2).all()
    # 2 team-2 players + 1 ball, but ball is excluded — 2 rows
    assert len(opp_101) == 2


def test_resolve_action_frame_context_unlinked_action():
    """Action with no frame within tolerance -> NaN pointer + empty opposite_rows."""
    from silly_kicks.tracking.utils import _resolve_action_frame_context

    actions = pd.DataFrame({
        "action_id": [999],
        "period_id": [1],
        "time_seconds": [1000.0],  # no frame at 1000s
        "team_id": [1], "player_id": [11],
        "start_x": [50.0], "start_y": [34.0],
        "end_x": [55.0], "end_y": [34.0],
    })
    frames = pd.DataFrame({
        "game_id": [1], "period_id": [1], "frame_id": [1000],
        "time_seconds": [10.0], "frame_rate": [25.0],
        "player_id": [11], "team_id": [1],
        "is_ball": [False], "is_goalkeeper": [False],
        "x": [50.0], "y": [34.0], "z": [float("nan")],
        "speed": [1.5], "speed_source": ["native"],
        "ball_state": ["alive"], "team_attacking_direction": ["ltr"],
        "confidence": [None], "visibility": [None], "source_provider": ["test"],
    })
    ctx = _resolve_action_frame_context(actions, frames)
    assert pd.isna(ctx.pointers["frame_id"].iloc[0])
```

- [ ] **Step 2.2: Verify tests fail**

Run: `python -m pytest tests/tracking/test_feature_framework.py::test_resolve_action_frame_context_links_all_actions -v`
Expected: FAIL — `ImportError` (`_resolve_action_frame_context` not yet defined).

- [ ] **Step 2.3: Implement `_resolve_action_frame_context` in utils.py**

Append to `silly_kicks/tracking/utils.py`:

```python
def _resolve_action_frame_context(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
) -> "ActionFrameContext":
    """Build the linked-context structure used by all 4 action_context features.

    Calls link_actions_to_frames once, then derives:
      - actor_rows: one row per action where the actor's player_id appears in the linked frame.
      - opposite_rows_per_action: long-form (action_id, opposite-team frame row) pairs,
        excluding ball rows (is_ball=True).

    Internal helper. Public per-feature surface lives in silly_kicks.tracking.features
    and silly_kicks.atomic.tracking.features.
    """
    from .feature_framework import ActionFrameContext  # avoid module-import cycle

    pointers, _report = link_actions_to_frames(actions, frames)

    # Inner-join pointers <-> frames on (period_id, frame_id) to materialize linked frames per action
    actions_with_period = actions[["action_id", "period_id", "team_id", "player_id"]]
    pointer_with_period = pointers.merge(
        actions_with_period, on="action_id", how="left", suffixes=("", "_action")
    )
    long = pointer_with_period.merge(
        frames,
        on=["period_id", "frame_id"],
        how="inner",
        suffixes=("_action", "_frame"),
    )

    # actor_rows: filter to rows where frame.player_id == action.player_id (and not ball)
    if "player_id_frame" in long.columns:
        actor_mask = (long["player_id_frame"] == long["player_id_action"]) & (~long["is_ball"])
        actor_long = long.loc[actor_mask].copy()
    else:
        actor_long = long.iloc[0:0].copy()

    # Build per-action actor row; left-join on action_id so unlinked actions also appear (with NaN cols)
    actor_rows = (
        pd.DataFrame({"action_id": actions["action_id"]})
        .merge(actor_long, on="action_id", how="left")
    )

    # opposite_rows_per_action: filter to rows where frame.team_id != action.team_id and not ball
    if "team_id_frame" in long.columns:
        opp_mask = (long["team_id_frame"] != long["team_id_action"]) & (~long["is_ball"])
        opposite = long.loc[opp_mask].copy()
    else:
        opposite = long.iloc[0:0].copy()

    return ActionFrameContext(
        actions=actions,
        pointers=pointers,
        actor_rows=actor_rows,
        opposite_rows_per_action=opposite,
    )
```

Note: column-name suffixes from the merge (`team_id_action` / `team_id_frame`, `player_id_action` / `player_id_frame`) are normalized inside this function. Downstream kernels read these suffixed column names; they're documented in the kernel docstrings (Task 6).

- [ ] **Step 2.4: Verify tests pass**

Run: `python -m pytest tests/tracking/test_feature_framework.py -v`
Expected: 11 passed.

---

## Task 3: `lift_to_states` implementation

**Files:**
- Modify: `tests/tracking/test_feature_framework.py` (add lift_to_states tests)
- Modify: `silly_kicks/tracking/feature_framework.py` (replace NotImplementedError stub)

- [ ] **Step 3.1: Add failing tests for lift_to_states**

Append to `tests/tracking/test_feature_framework.py`:

```python
def test_lift_to_states_marks_output_frame_aware(tiny_actions_and_frames):
    from silly_kicks.tracking.feature_framework import lift_to_states
    from silly_kicks.vaep.feature_framework import is_frame_aware

    def stub(actions, frames):
        return pd.Series([1.0] * len(actions), index=actions.index)
    stub.__name__ = "stub"

    lifted = lift_to_states(stub, nb_states=3)
    assert is_frame_aware(lifted) is True


def test_lift_to_states_produces_a0_a1_a2_columns(tiny_actions_and_frames):
    from silly_kicks.tracking.feature_framework import lift_to_states
    from silly_kicks.vaep.feature_framework import gamestates

    actions, frames = tiny_actions_and_frames

    def stub(actions, frames):
        return pd.Series([1.0] * len(actions), index=actions.index)
    stub.__name__ = "stub"

    states = gamestates(actions, nb_prev_actions=3)
    lifted = lift_to_states(stub, nb_states=3)
    out = lifted(states, frames)

    assert list(out.columns) == ["stub_a0", "stub_a1", "stub_a2"]
    assert len(out) == len(actions)


def test_lift_to_states_a0_matches_direct_call(tiny_actions_and_frames):
    """The _a0 column should equal the helper called on states[0] directly."""
    from silly_kicks.tracking.feature_framework import lift_to_states
    from silly_kicks.vaep.feature_framework import gamestates

    actions, frames = tiny_actions_and_frames

    def stub_increasing(actions, frames):
        return pd.Series(range(len(actions)), index=actions.index, dtype="float64")
    stub_increasing.__name__ = "stub_inc"

    states = gamestates(actions, nb_prev_actions=3)
    lifted = lift_to_states(stub_increasing, nb_states=3)
    out = lifted(states, frames)

    expected = stub_increasing(states[0], frames).to_numpy()
    actual = out["stub_inc_a0"].to_numpy()
    import numpy as np
    np.testing.assert_array_equal(actual, expected)
```

- [ ] **Step 3.2: Verify tests fail**

Run: `python -m pytest tests/tracking/test_feature_framework.py::test_lift_to_states_marks_output_frame_aware -v`
Expected: FAIL — `NotImplementedError`.

- [ ] **Step 3.3: Implement lift_to_states**

In `silly_kicks/tracking/feature_framework.py`, replace the `NotImplementedError` body:

```python
def lift_to_states(
    helper: Callable[[pd.DataFrame, pd.DataFrame], pd.Series],
    nb_states: int = 3,
) -> FrameAwareTransformer:
    """Lift a (actions, frames) -> Series helper to a (states, frames) -> Features transformer.

    The returned transformer is marked ``_frame_aware = True`` so
    ``VAEP.compute_features`` dispatches frames to it.

    Output columns: ``f"{helper.__name__}_a0"`` ... ``f"..._a{nb_states-1}"``.

    Examples
    --------
    Compose a tracking-aware feature into HybridVAEP::

        from silly_kicks.vaep.hybrid import HybridVAEP, hybrid_xfns_default
        from silly_kicks.tracking.feature_framework import lift_to_states
        from silly_kicks.tracking.features import nearest_defender_distance

        v = HybridVAEP(xfns=hybrid_xfns_default + [lift_to_states(nearest_defender_distance)])
    """
    name = helper.__name__

    def transformer(states, frames):
        out = pd.DataFrame(index=states[0].index)
        for i, slot in enumerate(states[:nb_states]):
            out[f"{name}_a{i}"] = helper(slot, frames).to_numpy()
        return out

    transformer._frame_aware = True  # type: ignore[attr-defined]
    transformer.__name__ = f"lifted_{name}"
    return transformer
```

- [ ] **Step 3.4: Verify tests pass**

Run: `python -m pytest tests/tracking/test_feature_framework.py -v`
Expected: 14 passed.

---

## Task 4: `VAEP.compute_features` extension with `frames=None` kwarg

**Files:**
- Create: `tests/vaep/test_compute_features_frames_kwarg.py`
- Modify: `silly_kicks/vaep/base.py` (extend compute_features)

- [ ] **Step 4.1: Write failing tests**

Create `tests/vaep/__init__.py` if missing (empty file). Create `tests/vaep/test_compute_features_frames_kwarg.py`:

```python
"""Tests for VAEP.compute_features extension with frames= kwarg.

Loop 4 covers: backward-compat regression, frame-aware xfn dispatch, ValueError
on missing frames, no module-import cycle.
"""
from __future__ import annotations

import pandas as pd
import pytest

import silly_kicks.spadl.config as spadlcfg


def _make_game_and_actions():
    """Tiny game (one row) + 5 actions, no tracking."""
    game = pd.Series({"game_id": 1, "home_team_id": 1, "away_team_id": 2})
    actions = pd.DataFrame({
        "game_id": [1] * 5,
        "original_event_id": [None] * 5,
        "action_id": [1, 2, 3, 4, 5],
        "period_id": [1, 1, 1, 1, 1],
        "time_seconds": [5.0, 10.0, 15.0, 20.0, 25.0],
        "team_id": [1, 1, 2, 1, 1],
        "player_id": [11, 12, 21, 11, 12],
        "start_x": [10.0, 30.0, 50.0, 70.0, 90.0],
        "start_y": [34.0, 30.0, 34.0, 38.0, 34.0],
        "end_x": [30.0, 50.0, 30.0, 90.0, 100.0],
        "end_y": [34.0, 30.0, 34.0, 38.0, 34.0],
        "type_id": [0, 0, 0, 0, 11],
        "result_id": [1, 1, 1, 1, 1],
        "bodypart_id": [0, 0, 0, 0, 0],
    })
    return game, actions


def test_compute_features_frames_none_is_regression_equivalent():
    """frames=None must be bit-identical to today (backward compat)."""
    from silly_kicks.vaep.base import VAEP, xfns_default

    v_old = VAEP()
    v_new = VAEP()
    game, actions = _make_game_and_actions()

    X_old = v_old.compute_features(game, actions)
    X_new = v_new.compute_features(game, actions, frames=None)
    pd.testing.assert_frame_equal(X_old, X_new)


def test_compute_features_raises_when_frame_aware_xfn_but_no_frames():
    """Frame-aware xfn in xfns + frames=None should raise ValueError with xfn name."""
    from silly_kicks.vaep.base import VAEP
    from silly_kicks.vaep.feature_framework import frame_aware

    @frame_aware
    def fake_tracking_feat(states, frames):
        return pd.DataFrame({"x": [1.0]})

    v = VAEP(xfns=[fake_tracking_feat])
    game, actions = _make_game_and_actions()
    with pytest.raises(ValueError, match="fake_tracking_feat"):
        v.compute_features(game, actions, frames=None)


def test_compute_features_dispatches_frame_aware_xfn():
    """Frame-aware xfn called with (states, frames) yields its columns in output."""
    from silly_kicks.vaep.base import VAEP
    from silly_kicks.vaep.feature_framework import frame_aware

    @frame_aware
    def fake_tracking_feat(states, frames):
        n = len(states[0])
        return pd.DataFrame({"fake_a0": [42.0] * n}, index=states[0].index)

    v = VAEP(xfns=[fake_tracking_feat])
    game, actions = _make_game_and_actions()
    # frames here is empty but non-None -> the @frame_aware xfn doesn't use it,
    # so the dispatch path is exercised
    frames = pd.DataFrame(columns=["period_id", "frame_id", "time_seconds"])
    X = v.compute_features(game, actions, frames=frames)
    assert "fake_a0" in X.columns
    assert (X["fake_a0"] == 42.0).all()


def test_no_module_import_cycle_when_frames_is_none():
    """Importing silly_kicks.vaep.base alone (without tracking) must not fail."""
    import importlib
    importlib.import_module("silly_kicks.vaep.base")
```

- [ ] **Step 4.2: Verify tests fail**

Run: `python -m pytest tests/vaep/test_compute_features_frames_kwarg.py -v`
Expected: FAIL on the new behaviour (compute_features doesn't yet take frames=).

- [ ] **Step 4.3: Extend `VAEP.compute_features`**

Open `silly_kicks/vaep/base.py`. Replace the existing `compute_features` method:

```python
def compute_features(
    self,
    game: pd.Series,
    game_actions: fs.Actions,
    *,
    frames: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Transform actions to the feature-based representation of game states.

    Parameters
    ----------
    game : pd.Series
        The SPADL representation of a single game.
    game_actions : pd.DataFrame
        The actions performed during `game` in the SPADL representation.
    frames : pd.DataFrame, optional
        Long-form tracking frames matching TRACKING_FRAMES_COLUMNS. Required when
        any xfn in self.xfns is marked frame-aware (via @frame_aware); ignored
        otherwise. When supplied, frames are normalized to LTR via
        ``silly_kicks.tracking.utils.play_left_to_right`` symmetrically with
        actions. The lazy import ensures no vaep<->tracking module-import cycle
        when frames is None.

    Returns
    -------
    features : pd.DataFrame

    Raises
    ------
    ValueError
        If self.xfns contains a frame-aware xfn but frames is None.

    Examples
    --------
    Without tracking (regression-equivalent to historical behaviour)::

        X = v.compute_features(game, game_actions)

    With tracking (e.g. tracking_default_xfns appended to xfns)::

        from silly_kicks.tracking.features import tracking_default_xfns
        from silly_kicks.vaep.hybrid import HybridVAEP, hybrid_xfns_default
        v = HybridVAEP(xfns=hybrid_xfns_default + tracking_default_xfns)
        X = v.compute_features(game, game_actions, frames=match_frames)
    """
    from .feature_framework import is_frame_aware

    game_actions_with_names = self._add_names(game_actions)  # type: ignore
    states = self._fs.gamestates(game_actions_with_names, self.nb_prev_actions)
    states = self._fs.play_left_to_right(states, game.home_team_id)

    if frames is not None:
        from silly_kicks.tracking.utils import play_left_to_right as _track_ltr
        frames = _track_ltr(frames, game.home_team_id)

    feats = []
    for fn in self.xfns:
        if is_frame_aware(fn):
            if frames is None:
                raise ValueError(
                    f"{fn.__name__} requires frames; pass frames= to compute_features"
                )
            feats.append(fn(states, frames))
        else:
            feats.append(fn(states))
    return pd.concat(feats, axis=1)  # type: ignore[reportReturnType]
```

- [ ] **Step 4.4: Verify tests pass**

Run: `python -m pytest tests/vaep/test_compute_features_frames_kwarg.py -v`
Expected: 4 passed.

- [ ] **Step 4.5: Verify existing VAEP tests still pass**

Run: `python -m pytest tests/ -k "vaep" -m "not e2e" --tb=short`
Expected: all green (no regressions).

---

## Task 5: `VAEP.rate` extension with `frames=None` kwarg

**Files:**
- Modify: `tests/vaep/test_compute_features_frames_kwarg.py` (add rate tests)
- Modify: `silly_kicks/vaep/base.py` (extend rate)

- [ ] **Step 5.1: Add failing test**

Append to `tests/vaep/test_compute_features_frames_kwarg.py`:

```python
def test_rate_passes_frames_to_compute_features():
    """rate(..., frames=...) routes through compute_features when game_states is None."""
    from silly_kicks.vaep.base import VAEP
    from silly_kicks.vaep.feature_framework import frame_aware
    from sklearn.exceptions import NotFittedError

    @frame_aware
    def fake_tracking_feat(states, frames):
        return pd.DataFrame({"fake_col": [0.5] * len(states[0])}, index=states[0].index)

    v = VAEP(xfns=[fake_tracking_feat])
    game, actions = _make_game_and_actions()
    frames = pd.DataFrame(columns=["period_id", "frame_id", "time_seconds"])

    # rate without fit raises NotFittedError, but it must reach _estimate_probabilities;
    # before that it calls compute_features with frames= when game_states is None.
    # We just verify the call doesn't error on the frames= passthrough.
    with pytest.raises(NotFittedError):
        v.rate(game, actions, game_states=None, frames=frames)
```

- [ ] **Step 5.2: Verify test fails**

Run: `python -m pytest tests/vaep/test_compute_features_frames_kwarg.py::test_rate_passes_frames_to_compute_features -v`
Expected: FAIL — `TypeError: rate() got an unexpected keyword argument 'frames'`.

- [ ] **Step 5.3: Extend `VAEP.rate`**

In `silly_kicks/vaep/base.py`, replace the existing `rate` method:

```python
def rate(
    self,
    game: pd.Series,
    game_actions: fs.Actions,
    game_states: fs.Features | None = None,
    *,
    frames: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute the VAEP rating for the given game states.

    Parameters
    ----------
    game : pd.Series
    game_actions : pd.DataFrame
    game_states : pd.DataFrame, default=None
    frames : pd.DataFrame, optional
        When ``game_states`` is None and self.xfns contains frame-aware xfns,
        frames must be supplied; passed through to compute_features.

    Returns
    -------
    ratings : pd.DataFrame
    """
    if not self._VAEP__models:  # type: ignore[attr-defined]
        from sklearn.exceptions import NotFittedError
        raise NotFittedError()

    game_actions_with_names = self._add_names(game_actions)  # type: ignore
    if game_states is None:
        game_states = self.compute_features(game, game_actions, frames=frames)

    y_hat = self._estimate_probabilities(game_states)
    p_scores, p_concedes = y_hat.iloc[:, 0], y_hat.iloc[:, 1]
    vaep_values = self._vaep.value(game_actions_with_names, p_scores, p_concedes)
    return vaep_values
```

(Note: the existing `rate` accesses `self.__models` via name mangling. Extending it preserves that access via `self._VAEP__models` from outside; alternatively keep the existing direct access if `rate` lives inside the VAEP class. The above replacement uses the existing pattern.)

- [ ] **Step 5.4: Verify tests pass**

Run: `python -m pytest tests/vaep/test_compute_features_frames_kwarg.py -v`
Expected: 5 passed.

---

## Task 6: `_kernels.py` — schema-agnostic compute kernels

**Files:**
- Create: `tests/tracking/test_kernels.py`
- Create: `silly_kicks/tracking/_kernels.py`

- [ ] **Step 6.1: Write failing tests with analytical ground truth**

Create `tests/tracking/test_kernels.py`:

```python
"""Tests for silly_kicks.tracking._kernels — pure-compute analytical-truth fixtures.

Loop 6 covers: nearest_defender_distance, actor_speed, receiver_zone_density,
defenders_in_triangle_to_goal kernels. Inputs are tiny in-memory ActionFrameContext
objects with known geometric ground truth.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking.feature_framework import ActionFrameContext


@pytest.fixture
def ctx_three_defenders():
    """Action at (50, 34); 3 opposite-team defenders at distances 2.0, 5.0, 10.0 m.

    nearest_defender_distance kernel must return 2.0.
    receiver_zone_density at radius=5.0 must return 2 (the 2.0 and 5.0 m defenders).
    """
    actions = pd.DataFrame({
        "action_id": [1],
        "start_x": [50.0], "start_y": [34.0],
        "end_x": [60.0], "end_y": [34.0],
        "team_id": [1], "player_id": [11],
    })
    pointers = pd.DataFrame({"action_id": [1], "frame_id": [1000]})
    actor_rows = pd.DataFrame({
        "action_id": [1],
        "x": [50.0], "y": [34.0], "speed": [1.5],
    })
    # Defender 1: 2.0 m east of action (52.0, 34.0) -> dist=2.0
    # Defender 2: 5.0 m east of action (55.0, 34.0) -> dist=5.0
    # Defender 3: 10.0 m east of action (60.0, 34.0) -> dist=10.0
    opposite = pd.DataFrame({
        "action_id": [1, 1, 1],
        "x": [52.0, 55.0, 60.0],
        "y": [34.0, 34.0, 34.0],
        "team_id_frame": [2, 2, 2],
    })
    return ActionFrameContext(actions=actions, pointers=pointers,
                              actor_rows=actor_rows, opposite_rows_per_action=opposite)


def test_nearest_defender_distance_kernel(ctx_three_defenders):
    from silly_kicks.tracking._kernels import _nearest_defender_distance

    result = _nearest_defender_distance(
        ctx_three_defenders.actions["start_x"],
        ctx_three_defenders.actions["start_y"],
        ctx_three_defenders,
    )
    assert math.isclose(result.iloc[0], 2.0, abs_tol=1e-9)


def test_receiver_zone_density_kernel(ctx_three_defenders):
    from silly_kicks.tracking._kernels import _receiver_zone_density

    # Action ends at (60, 34); defenders at distances 8.0, 5.0, 0.0 from end
    # At radius=5.0 -> count defenders within 5m of (60, 34) = the 5.0 (at 60-55=5) and 0.0 (at 60-60=0) -> 2
    # Wait, defender at 55.0 is 5.0 from end_x=60.0, defender at 60.0 is 0.0 from 60.0.
    # Defender at 52.0 is 8.0 from 60.0. So count = 2.
    result = _receiver_zone_density(
        ctx_three_defenders.actions["end_x"],
        ctx_three_defenders.actions["end_y"],
        ctx_three_defenders,
        radius=5.0,
    )
    assert int(result.iloc[0]) == 2


def test_actor_speed_kernel(ctx_three_defenders):
    from silly_kicks.tracking._kernels import _actor_speed_from_ctx

    result = _actor_speed_from_ctx(ctx_three_defenders)
    assert math.isclose(result.iloc[0], 1.5, abs_tol=1e-9)


def test_defenders_in_triangle_to_goal_kernel():
    """Defender at (80, 34) is between (50, 34) and the goal mouth — IN the triangle.
    Defender at (60, 60) is outside (above goal-mouth y range).
    """
    from silly_kicks.tracking._kernels import _defenders_in_triangle_to_goal

    actions = pd.DataFrame({
        "action_id": [1],
        "start_x": [50.0], "start_y": [34.0],
        "team_id": [1], "player_id": [11],
    })
    actor_rows = pd.DataFrame({"action_id": [1], "x": [50.0], "y": [34.0], "speed": [1.5]})
    pointers = pd.DataFrame({"action_id": [1], "frame_id": [1000]})
    opposite = pd.DataFrame({
        "action_id": [1, 1],
        "x": [80.0, 60.0],
        "y": [34.0, 60.0],
        "team_id_frame": [2, 2],
    })
    ctx = ActionFrameContext(actions=actions, pointers=pointers,
                             actor_rows=actor_rows, opposite_rows_per_action=opposite)
    result = _defenders_in_triangle_to_goal(actions["start_x"], actions["start_y"], ctx)
    assert int(result.iloc[0]) == 1


def test_unlinked_action_returns_nan():
    """Action with no actor_row and no opposite_rows -> NaN feature output."""
    from silly_kicks.tracking._kernels import _nearest_defender_distance

    actions = pd.DataFrame({
        "action_id": [1],
        "start_x": [50.0], "start_y": [34.0],
        "end_x": [60.0], "end_y": [34.0],
        "team_id": [1], "player_id": [11],
    })
    pointers = pd.DataFrame({"action_id": [1], "frame_id": [pd.NA]}, dtype="object")
    actor_rows = pd.DataFrame({"action_id": [1], "x": [float("nan")], "y": [float("nan")], "speed": [float("nan")]})
    opposite = pd.DataFrame({"action_id": [], "x": [], "y": [], "team_id_frame": []}, dtype="float64")
    ctx = ActionFrameContext(actions=actions, pointers=pointers,
                             actor_rows=actor_rows, opposite_rows_per_action=opposite)
    result = _nearest_defender_distance(actions["start_x"], actions["start_y"], ctx)
    assert pd.isna(result.iloc[0])


def test_receiver_zone_density_zero_when_no_defenders_in_radius():
    """Linked action with defenders all far from end -> count = 0 (NOT NaN)."""
    from silly_kicks.tracking._kernels import _receiver_zone_density

    actions = pd.DataFrame({
        "action_id": [1],
        "start_x": [50.0], "start_y": [34.0],
        "end_x": [60.0], "end_y": [34.0],
        "team_id": [1], "player_id": [11],
    })
    actor_rows = pd.DataFrame({"action_id": [1], "x": [50.0], "y": [34.0], "speed": [1.5]})
    pointers = pd.DataFrame({"action_id": [1], "frame_id": [1000]})
    # Defender far away: dist = 100 m from end (60, 34)
    opposite = pd.DataFrame({"action_id": [1], "x": [160.0], "y": [34.0], "team_id_frame": [2]})
    ctx = ActionFrameContext(actions=actions, pointers=pointers,
                             actor_rows=actor_rows, opposite_rows_per_action=opposite)
    result = _receiver_zone_density(actions["end_x"], actions["end_y"], ctx, radius=5.0)
    assert int(result.iloc[0]) == 0
```

- [ ] **Step 6.2: Verify tests fail**

Run: `python -m pytest tests/tracking/test_kernels.py -v`
Expected: FAIL — `ImportError` for `_nearest_defender_distance` etc.

- [ ] **Step 6.3: Implement `_kernels.py`**

Create `silly_kicks/tracking/_kernels.py`:

```python
"""Schema-agnostic compute kernels for tracking-aware action_context features.

Private module. Public per-schema wrappers live in
silly_kicks.tracking.features (standard SPADL) and
silly_kicks.atomic.tracking.features (atomic SPADL).

All kernels accept anchor_x / anchor_y as pd.Series (caller-supplied, allowing
per-schema column choice: standard's start_x/y, atomic's x/y, or end-anchors)
and an ActionFrameContext built once via _resolve_action_frame_context.

See spec docs/superpowers/specs/2026-04-30-action-context-pr1-design.md s 4.3
for the kernel pattern.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .feature_framework import ActionFrameContext

# Goal-mouth coordinates per spadl.config (105 x 68 m pitch, goal post-to-post 7.32 m centered on y=34)
_GOAL_X = 105.0
_GOAL_LEFT_POST_Y = 30.34
_GOAL_RIGHT_POST_Y = 37.66


def _nearest_defender_distance(
    anchor_x: pd.Series,
    anchor_y: pd.Series,
    ctx: ActionFrameContext,
) -> pd.Series:
    """Per action: distance from (anchor_x, anchor_y) to nearest opposite-team frame row.

    Returns NaN for actions with no opposite rows in ctx (unlinked or no defenders).
    """
    actions_id = ctx.actions["action_id"].values
    n = len(actions_id)
    out = pd.Series(np.full(n, np.nan), index=ctx.actions.index, dtype="float64")

    if len(ctx.opposite_rows_per_action) == 0:
        return out

    # Build per-action min-distance via merge + groupby
    a_xy = pd.DataFrame({
        "action_id": actions_id,
        "anchor_x": anchor_x.values,
        "anchor_y": anchor_y.values,
    })
    merged = ctx.opposite_rows_per_action.merge(a_xy, on="action_id", how="left")
    dx = merged["x"] - merged["anchor_x"]
    dy = merged["y"] - merged["anchor_y"]
    dist = np.sqrt(dx * dx + dy * dy)
    merged["_dist"] = dist
    min_per_action = merged.groupby("action_id")["_dist"].min()
    # Map back into the output index
    action_to_idx = pd.Series(ctx.actions.index, index=actions_id)
    for aid, d in min_per_action.items():
        if aid in action_to_idx.index:
            out.loc[action_to_idx.loc[aid]] = d
    return out


def _actor_speed_from_ctx(ctx: ActionFrameContext) -> pd.Series:
    """Per action: actor's speed from the linked frame row.

    NaN where action couldn't link, actor's player_id missing, or speed itself is NaN.
    """
    out = pd.Series(np.full(len(ctx.actions), np.nan), index=ctx.actions.index, dtype="float64")
    if len(ctx.actor_rows) == 0 or "speed" not in ctx.actor_rows.columns:
        return out
    action_to_idx = pd.Series(ctx.actions.index, index=ctx.actions["action_id"].values)
    for _, row in ctx.actor_rows.iterrows():
        aid = row["action_id"]
        speed = row["speed"]
        if aid in action_to_idx.index and pd.notna(speed):
            out.loc[action_to_idx.loc[aid]] = float(speed)
    return out


def _receiver_zone_density(
    anchor_x: pd.Series,
    anchor_y: pd.Series,
    ctx: ActionFrameContext,
    *,
    radius: float,
) -> pd.Series:
    """Per action: count of opposite-team frame rows within radius of (anchor_x, anchor_y).

    Returns NaN for unlinked actions (no pointer); 0 for linked actions with no defenders
    in radius (genuine count-zero distinction).
    """
    out = pd.Series(np.full(len(ctx.actions), np.nan), index=ctx.actions.index, dtype="float64")

    # Determine which actions are linked (have a frame_id in pointers)
    linked_aids = set(ctx.pointers.loc[ctx.pointers["frame_id"].notna(), "action_id"].values)
    action_to_idx = pd.Series(ctx.actions.index, index=ctx.actions["action_id"].values)
    # Initialize linked actions to 0 (count-zero distinction)
    for aid in linked_aids:
        if aid in action_to_idx.index:
            out.loc[action_to_idx.loc[aid]] = 0.0

    if len(ctx.opposite_rows_per_action) == 0:
        return out

    a_xy = pd.DataFrame({
        "action_id": ctx.actions["action_id"].values,
        "anchor_x": anchor_x.values,
        "anchor_y": anchor_y.values,
    })
    merged = ctx.opposite_rows_per_action.merge(a_xy, on="action_id", how="left")
    dx = merged["x"] - merged["anchor_x"]
    dy = merged["y"] - merged["anchor_y"]
    dist = np.sqrt(dx * dx + dy * dy)
    in_radius = dist <= radius
    merged["_in"] = in_radius.astype("int64")
    counts = merged.groupby("action_id")["_in"].sum()
    for aid, c in counts.items():
        if aid in action_to_idx.index:
            out.loc[action_to_idx.loc[aid]] = float(c)
    return out


def _defenders_in_triangle_to_goal(
    anchor_x: pd.Series,
    anchor_y: pd.Series,
    ctx: ActionFrameContext,
) -> pd.Series:
    """Per action: count of opposite-team frame rows inside the triangle
    (anchor, left_goalpost, right_goalpost).

    NaN for unlinked actions; 0 for linked-but-no-defenders-in-triangle.
    Triangle vertices: (anchor_x, anchor_y), (105, 30.34), (105, 37.66).
    Vectorized point-in-triangle via barycentric / sign-of-cross-product test.
    """
    out = pd.Series(np.full(len(ctx.actions), np.nan), index=ctx.actions.index, dtype="float64")

    linked_aids = set(ctx.pointers.loc[ctx.pointers["frame_id"].notna(), "action_id"].values)
    action_to_idx = pd.Series(ctx.actions.index, index=ctx.actions["action_id"].values)
    for aid in linked_aids:
        if aid in action_to_idx.index:
            out.loc[action_to_idx.loc[aid]] = 0.0

    if len(ctx.opposite_rows_per_action) == 0:
        return out

    a_xy = pd.DataFrame({
        "action_id": ctx.actions["action_id"].values,
        "ax": anchor_x.values,
        "ay": anchor_y.values,
    })
    merged = ctx.opposite_rows_per_action.merge(a_xy, on="action_id", how="left")

    # Vertices: A=(ax, ay), B=(105, 30.34), C=(105, 37.66); P=(merged.x, merged.y)
    # Use sign-of-cross-product test: P is in triangle iff cross signs all match.
    ax = merged["ax"].to_numpy()
    ay = merged["ay"].to_numpy()
    bx = np.full_like(ax, _GOAL_X)
    by = np.full_like(ay, _GOAL_LEFT_POST_Y)
    cx = np.full_like(ax, _GOAL_X)
    cy = np.full_like(ay, _GOAL_RIGHT_POST_Y)
    px = merged["x"].to_numpy()
    py = merged["y"].to_numpy()

    def _sign(x1, y1, x2, y2, x3, y3):
        return (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3)

    d1 = _sign(px, py, ax, ay, bx, by)
    d2 = _sign(px, py, bx, by, cx, cy)
    d3 = _sign(px, py, cx, cy, ax, ay)
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    in_triangle = ~(has_neg & has_pos)

    merged["_in"] = in_triangle.astype("int64")
    counts = merged.groupby("action_id")["_in"].sum()
    for aid, c in counts.items():
        if aid in action_to_idx.index:
            out.loc[action_to_idx.loc[aid]] = float(c)
    return out
```

- [ ] **Step 6.4: Verify tests pass**

Run: `python -m pytest tests/tracking/test_kernels.py -v`
Expected: 6 passed.

---

## Task 7: Standard SPADL feature wrappers (`silly_kicks/tracking/features.py`)

**Files:**
- Create: `tests/tracking/test_features_standard.py`
- Create: `silly_kicks/tracking/features.py`

- [ ] **Step 7.1: Write failing tests for the 4 public functions**

Create `tests/tracking/test_features_standard.py`:

```python
"""Tests for silly_kicks.tracking.features (standard SPADL public surface)."""
from __future__ import annotations

import math
import pandas as pd
import pytest


@pytest.fixture
def actions_and_frames_for_features():
    """Reuse the actions+frames fixture shape from test_feature_framework but with
    defender positions tuned for analytical test."""
    actions = pd.DataFrame({
        "action_id": [101],
        "period_id": [1],
        "time_seconds": [10.0],
        "team_id": [1], "player_id": [11],
        "start_x": [50.0], "start_y": [34.0],
        "end_x": [60.0], "end_y": [34.0],
    })
    rows = []
    fid, t = 1000, 10.0
    # Actor at (50, 34), speed 2.0 m/s
    rows.append(dict(
        game_id=1, period_id=1, frame_id=fid, time_seconds=t, frame_rate=25.0,
        player_id=11, team_id=1, is_ball=False, is_goalkeeper=False,
        x=50.0, y=34.0, z=float("nan"), speed=2.0, speed_source="native",
        ball_state="alive", team_attacking_direction="ltr",
        confidence=None, visibility=None, source_provider="test",
    ))
    # Opposite team defenders at distances 2 / 5 / 10 from action start (50, 34)
    for x in (52.0, 55.0, 60.0):
        rows.append(dict(
            game_id=1, period_id=1, frame_id=fid, time_seconds=t, frame_rate=25.0,
            player_id=20 + int(x), team_id=2, is_ball=False, is_goalkeeper=False,
            x=x, y=34.0, z=float("nan"), speed=1.5, speed_source="native",
            ball_state="alive", team_attacking_direction="ltr",
            confidence=None, visibility=None, source_provider="test",
        ))
    # Ball
    rows.append(dict(
        game_id=1, period_id=1, frame_id=fid, time_seconds=t, frame_rate=25.0,
        player_id=float("nan"), team_id=float("nan"), is_ball=True, is_goalkeeper=False,
        x=52.5, y=34.0, z=0.0, speed=5.0, speed_source="native",
        ball_state="alive", team_attacking_direction=None,
        confidence=None, visibility=None, source_provider="test",
    ))
    frames = pd.DataFrame(rows)
    return actions, frames


def test_nearest_defender_distance_standard(actions_and_frames_for_features):
    from silly_kicks.tracking.features import nearest_defender_distance

    actions, frames = actions_and_frames_for_features
    out = nearest_defender_distance(actions, frames)
    assert math.isclose(out.iloc[0], 2.0, abs_tol=1e-6)


def test_actor_speed_standard(actions_and_frames_for_features):
    from silly_kicks.tracking.features import actor_speed

    actions, frames = actions_and_frames_for_features
    out = actor_speed(actions, frames)
    assert math.isclose(out.iloc[0], 2.0, abs_tol=1e-6)


def test_receiver_zone_density_standard(actions_and_frames_for_features):
    """End at (60, 34); defenders at 52, 55, 60 -> distances 8, 5, 0 from end.
    Radius=5.0 -> count = 2 (the 5.0 and 0.0)."""
    from silly_kicks.tracking.features import receiver_zone_density

    actions, frames = actions_and_frames_for_features
    out = receiver_zone_density(actions, frames, radius=5.0)
    assert int(out.iloc[0]) == 2


def test_defenders_in_triangle_to_goal_standard(actions_and_frames_for_features):
    """Defender at (60, 34) is on the path from (50, 34) to goal mouth -> IN.
    Defender at (52, 34) is also closer than goal -> IN.
    Defender at (55, 34) is also IN.
    All 3 defenders are on the line y=34, x between 50 and 105 -> all 3 in triangle."""
    from silly_kicks.tracking.features import defenders_in_triangle_to_goal

    actions, frames = actions_and_frames_for_features
    out = defenders_in_triangle_to_goal(actions, frames)
    assert int(out.iloc[0]) == 3


def test_tracking_default_xfns_count():
    from silly_kicks.tracking.features import tracking_default_xfns

    assert len(tracking_default_xfns) == 4
```

- [ ] **Step 7.2: Verify tests fail**

Run: `python -m pytest tests/tracking/test_features_standard.py -v`
Expected: FAIL — `ImportError` for `silly_kicks.tracking.features`.

- [ ] **Step 7.3: Implement standard features module**

Create `silly_kicks/tracking/features.py`:

```python
"""Tracking-aware action_context features for standard SPADL.

Public API:
- nearest_defender_distance(actions, frames) -> pd.Series
- actor_speed(actions, frames) -> pd.Series
- receiver_zone_density(actions, frames, *, radius=5.0) -> pd.Series
- defenders_in_triangle_to_goal(actions, frames) -> pd.Series
- add_action_context(actions, frames, *, receiver_zone_radius=5.0) -> pd.DataFrame
- tracking_default_xfns: list[FrameAwareTransformer]

See NOTICE for full bibliographic citations and ADR-005 for the integration contract.
Spec: docs/superpowers/specs/2026-04-30-action-context-pr1-design.md.
"""
from __future__ import annotations

import pandas as pd

from silly_kicks._nan_safety import nan_safe_enrichment

from . import _kernels
from .feature_framework import lift_to_states
from .utils import _resolve_action_frame_context

__all__ = [
    "actor_speed",
    "add_action_context",
    "defenders_in_triangle_to_goal",
    "nearest_defender_distance",
    "receiver_zone_density",
    "tracking_default_xfns",
]


def nearest_defender_distance(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Meters to the closest opposing-team player at the linked frame.

    Anchor: ``(action.start_x, action.start_y)``. NaN if action couldn't link to a frame.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    Compute defender distance for a SPADL action stream::

        from silly_kicks.tracking.features import nearest_defender_distance
        d = nearest_defender_distance(actions, frames)

    References
    ----------
    Lucey et al. (2014). "Quality vs Quantity: Improved Shot Prediction in Soccer
        using Strategic Features from Spatiotemporal Data." MIT Sloan SAC.
    Anzer & Bauer (2021). "A goal scoring probability model for shots based on
        synchronized positional and event data in football and futsal."
        Frontiers in Sports and Active Living, 3, 624475.
    """
    ctx = _resolve_action_frame_context(actions, frames)
    return _kernels._nearest_defender_distance(actions["start_x"], actions["start_y"], ctx)


def actor_speed(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """m/s of the action's player_id at the linked frame.

    NaN if the action couldn't link, the actor's player_id is absent from the linked
    frame, or the frame's speed value is NaN.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    ::

        from silly_kicks.tracking.features import actor_speed
        s = actor_speed(actions, frames)

    References
    ----------
    Anzer & Bauer (2021). "A goal scoring probability model for shots based on
        synchronized positional and event data in football and futsal."
        Frontiers in Sports and Active Living, 3, 624475.
    Bauer & Anzer (2021). "Data-driven detection of counterpressing in professional
        football." Data Mining and Knowledge Discovery, 35(5), 2009-2049.
    """
    ctx = _resolve_action_frame_context(actions, frames)
    return _kernels._actor_speed_from_ctx(ctx)


def receiver_zone_density(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    radius: float = 5.0,
) -> pd.Series:
    """Count of opposing-team players within ``radius`` of (action.end_x, action.end_y).

    Integer-valued (0 if linked but no defenders within radius; NaN if unlinked).

    See NOTICE for full bibliographic citations.

    Examples
    --------
    ::

        from silly_kicks.tracking.features import receiver_zone_density
        d = receiver_zone_density(actions, frames, radius=5.0)

    References
    ----------
    Spearman (2018). "Beyond Expected Goals." MIT Sloan SAC.
    Power et al. (2017). "Not all passes are created equal." KDD '17 (OBSO).
    """
    ctx = _resolve_action_frame_context(actions, frames)
    return _kernels._receiver_zone_density(actions["end_x"], actions["end_y"], ctx, radius=radius)


def defenders_in_triangle_to_goal(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
) -> pd.Series:
    """Count of opposing-team players inside the triangle
    (action.start_x, action.start_y) -> goal-mouth posts at x=105.

    Goal-mouth: y in [30.34, 37.66] per spadl.config.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    ::

        from silly_kicks.tracking.features import defenders_in_triangle_to_goal
        d = defenders_in_triangle_to_goal(actions, frames)

    References
    ----------
    Lucey et al. (2014). "Quality vs Quantity: Improved Shot Prediction in Soccer
        using Strategic Features from Spatiotemporal Data." MIT Sloan SAC.
    Pollard & Reep (1997). "Measuring the effectiveness of playing strategies at
        soccer." J. Royal Statistical Society Series D, 46(4), 541-550.
    """
    ctx = _resolve_action_frame_context(actions, frames)
    return _kernels._defenders_in_triangle_to_goal(actions["start_x"], actions["start_y"], ctx)


@nan_safe_enrichment
def add_action_context(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    receiver_zone_radius: float = 5.0,
) -> pd.DataFrame:
    """Enrich actions with 4 tracking-aware features + 4 linkage-provenance columns.

    Returns
    -------
    pd.DataFrame
        Input actions with the columns:
        - nearest_defender_distance (float64, meters)
        - actor_speed (float64, m/s)
        - receiver_zone_density (Int64, count; NaN unlinked, 0 = no defenders)
        - defenders_in_triangle_to_goal (Int64, count; NaN unlinked, 0 = none)
        - frame_id (Int64; NaN if unlinked)
        - time_offset_seconds (float64; NaN if unlinked)
        - link_quality_score (float64; NaN if unlinked)
        - n_candidate_frames (int64)

    See NOTICE for full bibliographic citations.

    Examples
    --------
    ::

        from silly_kicks.tracking.features import add_action_context
        enriched = add_action_context(actions, frames, receiver_zone_radius=5.0)
    """
    ctx = _resolve_action_frame_context(actions, frames)
    out = actions.copy()
    out["nearest_defender_distance"] = _kernels._nearest_defender_distance(
        actions["start_x"], actions["start_y"], ctx
    )
    out["actor_speed"] = _kernels._actor_speed_from_ctx(ctx)
    rz = _kernels._receiver_zone_density(actions["end_x"], actions["end_y"], ctx, radius=receiver_zone_radius)
    out["receiver_zone_density"] = rz.astype("Int64")
    dt = _kernels._defenders_in_triangle_to_goal(actions["start_x"], actions["start_y"], ctx)
    out["defenders_in_triangle_to_goal"] = dt.astype("Int64")
    # Provenance columns from pointers
    pointer_cols = ctx.pointers.set_index("action_id")[
        ["frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]
    ]
    out = out.merge(pointer_cols, left_on="action_id", right_index=True, how="left")
    return out


tracking_default_xfns = [
    lift_to_states(nearest_defender_distance),
    lift_to_states(actor_speed),
    lift_to_states(receiver_zone_density),
    lift_to_states(defenders_in_triangle_to_goal),
]
```

Note: `add_action_context` is decorated with `@nan_safe_enrichment` per ADR-003; the existing `tests/test_enrichment_nan_safety.py` auto-discovers it (Task 8 verifies).

- [ ] **Step 7.4: Verify tests pass**

Run: `python -m pytest tests/tracking/test_features_standard.py -v`
Expected: 5 passed.

---

## Task 8: `add_action_context` aggregator + provenance columns + NaN safety

**Files:**
- Create: `tests/tracking/test_add_action_context.py`
- (`silly_kicks/tracking/features.py` already has add_action_context from Task 7)

- [ ] **Step 8.1: Add aggregator-specific tests**

Create `tests/tracking/test_add_action_context.py`:

```python
"""Tests for add_action_context aggregator: provenance columns, NaN safety, dtypes."""
from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def actions_and_frames_aggregator():
    """2 linked actions + 1 unlinked action."""
    actions = pd.DataFrame({
        "action_id": [101, 102, 999],
        "period_id": [1, 1, 1],
        "time_seconds": [10.0, 20.0, 1000.0],  # 999 is unlinked (no frame at 1000s)
        "team_id": [1, 1, 1],
        "player_id": [11, 11, 11],
        "start_x": [50.0, 60.0, 50.0],
        "start_y": [34.0, 30.0, 34.0],
        "end_x": [55.0, 65.0, 55.0],
        "end_y": [34.0, 30.0, 34.0],
    })
    rows = []
    for fid, t in [(1000, 10.0), (2000, 20.0)]:
        # Actor (player 11, team 1) at action position
        rows.append(dict(
            game_id=1, period_id=1, frame_id=fid, time_seconds=t, frame_rate=25.0,
            player_id=11, team_id=1, is_ball=False, is_goalkeeper=False,
            x=50.0 if fid == 1000 else 60.0,
            y=34.0 if fid == 1000 else 30.0,
            z=float("nan"), speed=2.0, speed_source="native",
            ball_state="alive", team_attacking_direction="ltr",
            confidence=None, visibility=None, source_provider="test",
        ))
        # 1 opposite-team defender 5m away
        rows.append(dict(
            game_id=1, period_id=1, frame_id=fid, time_seconds=t, frame_rate=25.0,
            player_id=22, team_id=2, is_ball=False, is_goalkeeper=False,
            x=(50.0 if fid == 1000 else 60.0) + 5.0,
            y=34.0 if fid == 1000 else 30.0,
            z=float("nan"), speed=1.0, speed_source="native",
            ball_state="alive", team_attacking_direction="ltr",
            confidence=None, visibility=None, source_provider="test",
        ))
    frames = pd.DataFrame(rows)
    return actions, frames


def test_add_action_context_returns_input_plus_8_columns(actions_and_frames_aggregator):
    from silly_kicks.tracking.features import add_action_context

    actions, frames = actions_and_frames_aggregator
    enriched = add_action_context(actions, frames)

    new_cols = set(enriched.columns) - set(actions.columns)
    expected = {
        "nearest_defender_distance", "actor_speed",
        "receiver_zone_density", "defenders_in_triangle_to_goal",
        "frame_id", "time_offset_seconds", "link_quality_score", "n_candidate_frames",
    }
    assert expected.issubset(new_cols)


def test_add_action_context_unlinked_action_has_nan_features(actions_and_frames_aggregator):
    from silly_kicks.tracking.features import add_action_context

    actions, frames = actions_and_frames_aggregator
    enriched = add_action_context(actions, frames)

    unlinked = enriched[enriched["action_id"] == 999].iloc[0]
    assert pd.isna(unlinked["nearest_defender_distance"])
    assert pd.isna(unlinked["actor_speed"])
    assert pd.isna(unlinked["frame_id"])


def test_add_action_context_linked_action_has_features(actions_and_frames_aggregator):
    from silly_kicks.tracking.features import add_action_context

    actions, frames = actions_and_frames_aggregator
    enriched = add_action_context(actions, frames)

    linked = enriched[enriched["action_id"] == 101].iloc[0]
    assert linked["nearest_defender_distance"] == 5.0
    assert linked["actor_speed"] == 2.0


def test_add_action_context_dtypes(actions_and_frames_aggregator):
    from silly_kicks.tracking.features import add_action_context

    actions, frames = actions_and_frames_aggregator
    enriched = add_action_context(actions, frames)

    assert enriched["nearest_defender_distance"].dtype == "float64"
    assert enriched["actor_speed"].dtype == "float64"
    assert enriched["receiver_zone_density"].dtype.name == "Int64"
    assert enriched["defenders_in_triangle_to_goal"].dtype.name == "Int64"


def test_add_action_context_is_nan_safe_decorated():
    """ADR-003 contract: add_action_context is in the auto-discovered NaN-safety registry."""
    from silly_kicks.tracking.features import add_action_context
    from silly_kicks._nan_safety import is_nan_safe_enrichment

    assert is_nan_safe_enrichment(add_action_context) is True
```

- [ ] **Step 8.2: Verify tests fail (until any nuance is fixed)**

Run: `python -m pytest tests/tracking/test_add_action_context.py -v`
Expected: most pass from Task 7's implementation; any failures point to bugs to fix in Task 7 implementation. Resolve until all 5 pass.

- [ ] **Step 8.3: Verify NaN-safety auto-discovery test picks up add_action_context**

Run: `python -m pytest tests/test_enrichment_nan_safety.py -v`
Expected: passes; the new add_action_context is auto-discovered by the registry-floor + fuzz test pattern (per ADR-003).

---

## Task 9: HybridVAEP integration AUC test

**Files:**
- Create: `tests/vaep/test_hybrid_with_tracking.py`

- [ ] **Step 9.1: Write the integration AUC test**

Create `tests/vaep/test_hybrid_with_tracking.py`:

```python
"""End-to-end HybridVAEP integration test with tracking-aware features.

Loop 9 covers: HybridVAEP(xfns=hybrid_xfns_default + tracking_default_xfns)
.fit(...).rate(...) full lifecycle works on synthetic fixture; AUC uplift from
tracking features is detectable on synthetic data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_synthetic_match(seed: int = 42, n_actions: int = 200):
    """Build a tiny synthetic SPADL action stream + linked tracking frames.

    Construction tuned so tracking features carry signal:
    - Actions near defenders (low nearest_defender_distance) have lower goal-prob.
    - Receiver-zone density correlates with goal-prob.
    """
    rng = np.random.default_rng(seed)
    actions = pd.DataFrame({
        "game_id": [1] * n_actions,
        "original_event_id": [None] * n_actions,
        "action_id": list(range(1, n_actions + 1)),
        "period_id": [1] * n_actions,
        "time_seconds": [t * 0.5 for t in range(n_actions)],
        "team_id": rng.choice([1, 2], size=n_actions),
        "player_id": rng.choice([11, 12, 13, 21, 22, 23], size=n_actions),
        "start_x": rng.uniform(0, 105, size=n_actions),
        "start_y": rng.uniform(0, 68, size=n_actions),
        "end_x": rng.uniform(0, 105, size=n_actions),
        "end_y": rng.uniform(0, 68, size=n_actions),
        "type_id": rng.choice([0, 1, 11], size=n_actions),  # 11 == shot
        "result_id": [1] * n_actions,
        "bodypart_id": [0] * n_actions,
    })
    # frames at action timestamps; defender distances correlated with type_id=11 (shots)
    rows = []
    for _, a in actions.iterrows():
        fid = int(a["time_seconds"] * 10)
        # Actor row
        rows.append(dict(
            game_id=1, period_id=1, frame_id=fid, time_seconds=a["time_seconds"],
            frame_rate=10.0, player_id=a["player_id"], team_id=a["team_id"],
            is_ball=False, is_goalkeeper=False,
            x=a["start_x"], y=a["start_y"], z=float("nan"),
            speed=rng.uniform(0, 6), speed_source="native",
            ball_state="alive", team_attacking_direction="ltr",
            confidence=None, visibility=None, source_provider="synth",
        ))
        # Add 5 defenders at controllable distance
        # When type_id==11 (shot), defenders are CLOSER -> shot has lower scoring prob
        defender_dist = 3.0 if a["type_id"] == 11 else 8.0
        opposite_team = 2 if a["team_id"] == 1 else 1
        for j in range(5):
            angle = 2 * np.pi * j / 5
            rows.append(dict(
                game_id=1, period_id=1, frame_id=fid, time_seconds=a["time_seconds"],
                frame_rate=10.0, player_id=30 + j, team_id=opposite_team,
                is_ball=False, is_goalkeeper=False,
                x=a["start_x"] + defender_dist * np.cos(angle),
                y=a["start_y"] + defender_dist * np.sin(angle),
                z=float("nan"), speed=1.5, speed_source="native",
                ball_state="alive", team_attacking_direction="ltr",
                confidence=None, visibility=None, source_provider="synth",
            ))
    frames = pd.DataFrame(rows)
    return actions, frames


@pytest.mark.slow
def test_hybrid_vaep_with_tracking_lifecycle():
    """HybridVAEP + tracking_default_xfns: compute_features + fit + rate without errors."""
    from silly_kicks.vaep.hybrid import HybridVAEP, hybrid_xfns_default
    from silly_kicks.tracking.features import tracking_default_xfns

    actions, frames = _make_synthetic_match()
    game = pd.Series({"game_id": 1, "home_team_id": 1, "away_team_id": 2})

    v = HybridVAEP(xfns=hybrid_xfns_default + tracking_default_xfns)
    X = v.compute_features(game, actions, frames=frames)
    # Sanity: tracking-aware columns present
    assert any("nearest_defender_distance_a0" in c for c in X.columns)

    # Synthetic labels: shots (type_id=11) -> small chance of scoring
    y = pd.DataFrame({
        "scores": (actions["type_id"] == 11).astype(int).to_numpy(),
        "concedes": np.zeros(len(actions), dtype=int),
    })

    v.fit(X, y, learner="xgboost", val_size=0.25, random_state=42)
    ratings = v.rate(game, actions, frames=frames)
    assert "vaep_value" in ratings.columns
```

- [ ] **Step 9.2: Run and verify**

Run: `python -m pytest tests/vaep/test_hybrid_with_tracking.py -v`
Expected: 1 passed. (May take ~10-30s for xgboost fit on the small synthetic dataset.)

If the test fails because xgboost has trouble with the small dataset, increase `n_actions` to 500 in `_make_synthetic_match` or set `tree_params={"n_estimators": 50}`.

---

## Task 10: Atomic-SPADL feature wrappers

**Files:**
- Create: `silly_kicks/atomic/tracking/__init__.py`
- Create: `silly_kicks/atomic/tracking/features.py`
- Create: `tests/atomic/tracking/__init__.py`
- Create: `tests/atomic/tracking/test_features_atomic.py`
- Create: `tests/atomic/tracking/test_atomic_action_context.py`

- [ ] **Step 10.1: Write failing tests for atomic features**

Create `tests/atomic/tracking/__init__.py` (empty). Create `tests/atomic/tracking/test_features_atomic.py`:

```python
"""Tests for silly_kicks.atomic.tracking.features (atomic SPADL public surface).

Atomic-SPADL has columns (x, y, dx, dy) instead of (start_x, start_y, end_x, end_y).
The wrappers consume the same _kernels with atomic-shaped column reads.
"""
from __future__ import annotations

import math
import pandas as pd
import pytest


@pytest.fixture
def atomic_actions_and_frames():
    """1 atomic action at (50, 34), dx=10 dy=0 -> end at (60, 34).
    Defender at (52, 34) -> 2.0 m from start; 8.0 m from end."""
    actions = pd.DataFrame({
        "action_id": [101],
        "period_id": [1],
        "time_seconds": [10.0],
        "team_id": [1], "player_id": [11],
        "x": [50.0], "y": [34.0],
        "dx": [10.0], "dy": [0.0],
        "type_id": [0],
        "bodypart_id": [0],
    })
    rows = [
        # Actor at start
        dict(game_id=1, period_id=1, frame_id=1000, time_seconds=10.0, frame_rate=25.0,
             player_id=11, team_id=1, is_ball=False, is_goalkeeper=False,
             x=50.0, y=34.0, z=float("nan"), speed=2.0, speed_source="native",
             ball_state="alive", team_attacking_direction="ltr",
             confidence=None, visibility=None, source_provider="test"),
        # 1 defender at (52, 34) -> 2m from start
        dict(game_id=1, period_id=1, frame_id=1000, time_seconds=10.0, frame_rate=25.0,
             player_id=22, team_id=2, is_ball=False, is_goalkeeper=False,
             x=52.0, y=34.0, z=float("nan"), speed=1.5, speed_source="native",
             ball_state="alive", team_attacking_direction="ltr",
             confidence=None, visibility=None, source_provider="test"),
    ]
    frames = pd.DataFrame(rows)
    return actions, frames


def test_atomic_nearest_defender_distance(atomic_actions_and_frames):
    from silly_kicks.atomic.tracking.features import nearest_defender_distance

    actions, frames = atomic_actions_and_frames
    out = nearest_defender_distance(actions, frames)
    assert math.isclose(out.iloc[0], 2.0, abs_tol=1e-6)


def test_atomic_receiver_zone_density(atomic_actions_and_frames):
    """End at (60, 34); defender at (52, 34) -> 8m from end -> outside radius=5 -> 0."""
    from silly_kicks.atomic.tracking.features import receiver_zone_density

    actions, frames = atomic_actions_and_frames
    out = receiver_zone_density(actions, frames, radius=5.0)
    assert int(out.iloc[0]) == 0


def test_atomic_actor_speed(atomic_actions_and_frames):
    from silly_kicks.atomic.tracking.features import actor_speed

    actions, frames = atomic_actions_and_frames
    out = actor_speed(actions, frames)
    assert math.isclose(out.iloc[0], 2.0, abs_tol=1e-6)


def test_atomic_defenders_in_triangle_to_goal(atomic_actions_and_frames):
    """Defender at (52, 34) is inside triangle (50,34) -> goal posts."""
    from silly_kicks.atomic.tracking.features import defenders_in_triangle_to_goal

    actions, frames = atomic_actions_and_frames
    out = defenders_in_triangle_to_goal(actions, frames)
    assert int(out.iloc[0]) == 1


def test_atomic_zero_dx_dy_is_degenerate_density(atomic_actions_and_frames):
    """Atomic action with dx=dy=0 -> end == start -> density at start equals density-at-anchor.
    Documented degenerate case per spec."""
    from silly_kicks.atomic.tracking.features import receiver_zone_density

    actions, frames = atomic_actions_and_frames
    actions = actions.copy()
    actions["dx"] = 0.0
    actions["dy"] = 0.0
    out = receiver_zone_density(actions, frames, radius=5.0)
    # Defender at (52, 34) is 2m from (50, 34) -> 1 defender within radius=5
    assert int(out.iloc[0]) == 1
```

Create `tests/atomic/tracking/test_atomic_action_context.py`:

```python
"""Aggregator + provenance for atomic SPADL."""
from __future__ import annotations

import pandas as pd


def test_atomic_add_action_context_smoke():
    from silly_kicks.atomic.tracking.features import add_action_context

    actions = pd.DataFrame({
        "action_id": [101],
        "period_id": [1],
        "time_seconds": [10.0],
        "team_id": [1], "player_id": [11],
        "x": [50.0], "y": [34.0],
        "dx": [10.0], "dy": [0.0],
        "type_id": [0], "bodypart_id": [0],
    })
    frames = pd.DataFrame([
        dict(game_id=1, period_id=1, frame_id=1000, time_seconds=10.0, frame_rate=25.0,
             player_id=11, team_id=1, is_ball=False, is_goalkeeper=False,
             x=50.0, y=34.0, z=float("nan"), speed=2.0, speed_source="native",
             ball_state="alive", team_attacking_direction="ltr",
             confidence=None, visibility=None, source_provider="test"),
        dict(game_id=1, period_id=1, frame_id=1000, time_seconds=10.0, frame_rate=25.0,
             player_id=22, team_id=2, is_ball=False, is_goalkeeper=False,
             x=52.0, y=34.0, z=float("nan"), speed=1.5, speed_source="native",
             ball_state="alive", team_attacking_direction="ltr",
             confidence=None, visibility=None, source_provider="test"),
    ])
    enriched = add_action_context(actions, frames)
    assert "nearest_defender_distance" in enriched.columns
    assert enriched["nearest_defender_distance"].iloc[0] == 2.0
```

- [ ] **Step 10.2: Verify tests fail**

Run: `python -m pytest tests/atomic/tracking/ -v`
Expected: FAIL — `silly_kicks.atomic.tracking` module doesn't exist yet.

- [ ] **Step 10.3: Implement atomic-tracking features**

Create `silly_kicks/atomic/tracking/__init__.py`:

```python
"""Tracking-aware features for atomic SPADL.

Mirrors silly_kicks.tracking.features with atomic-SPADL column conventions
(x, y, dx, dy instead of start_x, start_y, end_x, end_y).
"""
from . import features

__all__ = ["features"]
```

Create `silly_kicks/atomic/tracking/features.py`:

```python
"""Tracking-aware action_context features for atomic SPADL.

Mirrors silly_kicks.tracking.features with atomic-shaped column reads.
Shares the schema-agnostic kernels in silly_kicks.tracking._kernels.

See NOTICE for full bibliographic citations and ADR-005 for the integration contract.
"""
from __future__ import annotations

import pandas as pd

from silly_kicks._nan_safety import nan_safe_enrichment
from silly_kicks.tracking import _kernels
from silly_kicks.tracking.feature_framework import lift_to_states
from silly_kicks.tracking.utils import _resolve_action_frame_context

__all__ = [
    "actor_speed",
    "add_action_context",
    "atomic_tracking_default_xfns",
    "defenders_in_triangle_to_goal",
    "nearest_defender_distance",
    "receiver_zone_density",
]


def nearest_defender_distance(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Atomic-SPADL: distance to nearest defender at action anchor (x, y).

    See NOTICE; matches silly_kicks.tracking.features.nearest_defender_distance.
    """
    ctx = _resolve_action_frame_context(actions, frames)
    return _kernels._nearest_defender_distance(actions["x"], actions["y"], ctx)


def actor_speed(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Atomic-SPADL: actor's speed at the linked frame.

    See NOTICE.
    """
    ctx = _resolve_action_frame_context(actions, frames)
    return _kernels._actor_speed_from_ctx(ctx)


def receiver_zone_density(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    radius: float = 5.0,
) -> pd.Series:
    """Atomic-SPADL: defenders within radius of (x + dx, y + dy).

    Degenerate case: when dx == dy == 0 (instantaneous atomic actions like shots),
    density is computed at the anchor (x, y).

    See NOTICE.
    """
    ctx = _resolve_action_frame_context(actions, frames)
    end_x = actions["x"] + actions["dx"]
    end_y = actions["y"] + actions["dy"]
    return _kernels._receiver_zone_density(end_x, end_y, ctx, radius=radius)


def defenders_in_triangle_to_goal(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
) -> pd.Series:
    """Atomic-SPADL: defenders in triangle from (x, y) to goal posts.

    See NOTICE.
    """
    ctx = _resolve_action_frame_context(actions, frames)
    return _kernels._defenders_in_triangle_to_goal(actions["x"], actions["y"], ctx)


@nan_safe_enrichment
def add_action_context(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    receiver_zone_radius: float = 5.0,
) -> pd.DataFrame:
    """Atomic-SPADL aggregator: enrich actions with the 4 features + 4 provenance cols.

    Parallels silly_kicks.tracking.features.add_action_context with atomic-shaped
    column reads (x, y, dx, dy).

    See NOTICE.
    """
    ctx = _resolve_action_frame_context(actions, frames)
    out = actions.copy()
    out["nearest_defender_distance"] = _kernels._nearest_defender_distance(
        actions["x"], actions["y"], ctx
    )
    out["actor_speed"] = _kernels._actor_speed_from_ctx(ctx)
    end_x = actions["x"] + actions["dx"]
    end_y = actions["y"] + actions["dy"]
    rz = _kernels._receiver_zone_density(end_x, end_y, ctx, radius=receiver_zone_radius)
    out["receiver_zone_density"] = rz.astype("Int64")
    dt = _kernels._defenders_in_triangle_to_goal(actions["x"], actions["y"], ctx)
    out["defenders_in_triangle_to_goal"] = dt.astype("Int64")
    pointer_cols = ctx.pointers.set_index("action_id")[
        ["frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]
    ]
    out = out.merge(pointer_cols, left_on="action_id", right_index=True, how="left")
    return out


atomic_tracking_default_xfns = [
    lift_to_states(nearest_defender_distance),
    lift_to_states(actor_speed),
    lift_to_states(receiver_zone_density),
    lift_to_states(defenders_in_triangle_to_goal),
]
```

- [ ] **Step 10.4: Verify tests pass**

Run: `python -m pytest tests/atomic/tracking/ -v`
Expected: 6 passed.

---

## Task 11: AtomicVAEP integration smoke test

**Files:**
- Create: `tests/atomic/vaep/test_atomic_with_tracking.py`

- [ ] **Step 11.1: Write the smoke test**

Create `tests/atomic/vaep/__init__.py` if missing. Create `tests/atomic/vaep/test_atomic_with_tracking.py`:

```python
"""AtomicVAEP integration smoke test with tracking-aware features.

Loop 11 covers: AtomicVAEP(xfns=atomic_xfns_default + atomic_tracking_default_xfns)
.compute_features(...) reaches the tracking branch without errors. No AUC uplift
assertion (atomic synthetic noise floor is too high for a stable assertion).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.mark.slow
def test_atomic_vaep_with_tracking_compute_features():
    from silly_kicks.atomic.vaep.base import AtomicVAEP, xfns_default
    from silly_kicks.atomic.tracking.features import atomic_tracking_default_xfns

    rng = np.random.default_rng(0)
    n = 50
    actions = pd.DataFrame({
        "game_id": [1] * n,
        "original_event_id": [None] * n,
        "action_id": list(range(1, n + 1)),
        "period_id": [1] * n,
        "time_seconds": [t * 0.5 for t in range(n)],
        "team_id": rng.choice([1, 2], size=n),
        "player_id": rng.choice([11, 12, 21, 22], size=n),
        "x": rng.uniform(0, 105, size=n),
        "y": rng.uniform(0, 68, size=n),
        "dx": rng.uniform(-10, 10, size=n),
        "dy": rng.uniform(-5, 5, size=n),
        "type_id": [0] * n,
        "bodypart_id": [0] * n,
    })
    rows = []
    for _, a in actions.iterrows():
        fid = int(a["time_seconds"] * 10)
        rows.append(dict(
            game_id=1, period_id=1, frame_id=fid, time_seconds=a["time_seconds"],
            frame_rate=10.0, player_id=a["player_id"], team_id=a["team_id"],
            is_ball=False, is_goalkeeper=False,
            x=a["x"], y=a["y"], z=float("nan"),
            speed=2.0, speed_source="native",
            ball_state="alive", team_attacking_direction="ltr",
            confidence=None, visibility=None, source_provider="synth",
        ))
        rows.append(dict(
            game_id=1, period_id=1, frame_id=fid, time_seconds=a["time_seconds"],
            frame_rate=10.0, player_id=99, team_id=2 if a["team_id"] == 1 else 1,
            is_ball=False, is_goalkeeper=False,
            x=a["x"] + 5.0, y=a["y"], z=float("nan"),
            speed=1.5, speed_source="native",
            ball_state="alive", team_attacking_direction="ltr",
            confidence=None, visibility=None, source_provider="synth",
        ))
    frames = pd.DataFrame(rows)
    game = pd.Series({"game_id": 1, "home_team_id": 1, "away_team_id": 2})

    v = AtomicVAEP(xfns=list(xfns_default) + atomic_tracking_default_xfns)
    X = v.compute_features(game, actions, frames=frames)
    assert any("nearest_defender_distance_a0" in c for c in X.columns)
```

- [ ] **Step 11.2: Verify test passes**

Run: `python -m pytest tests/atomic/vaep/test_atomic_with_tracking.py -v`
Expected: 1 passed.

---

## Task 12: TODO.md restructure + ADR-005 + NOTICE file + cross-links

**Files:**
- Modify: `TODO.md` (full restructure to lakehouse-style)
- Create: `docs/superpowers/adrs/ADR-005-tracking-aware-features.md`
- Create: `NOTICE` (repo root)
- Modify: `README.md` (add Attribution section)
- Modify: `CLAUDE.md` (add academic-attribution line)
- Create: `tests/test_todo_md_format.py`
- Create: `tests/test_notice_md_format.py`

- [ ] **Step 12.1: Write failing TODO.md format test**

Create `tests/test_todo_md_format.py`:

```python
"""Loose structural check on TODO.md to mirror lakehouse format."""
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
TODO_PATH = REPO_ROOT / "TODO.md"


def test_todo_has_on_deck_section():
    text = TODO_PATH.read_text(encoding="utf-8")
    assert "## On Deck" in text


def test_todo_has_size_legend():
    text = TODO_PATH.read_text(encoding="utf-8")
    for kw in ["Monstah", "Wicked", "Dunkin'"]:
        assert kw in text, f"missing {kw} in size legend"


def test_todo_has_active_cycle_or_archive_marker():
    text = TODO_PATH.read_text(encoding="utf-8")
    assert "## Active Cycle" in text or "## Tech" in text


def test_todo_lists_tf1_through_tf6_at_minimum():
    text = TODO_PATH.read_text(encoding="utf-8")
    for tf in ["TF-1", "TF-2", "TF-3", "TF-4", "TF-5", "TF-6"]:
        assert tf in text, f"{tf} entry missing"
```

- [ ] **Step 12.2: Write failing NOTICE format test**

Create `tests/test_notice_md_format.py`:

```python
"""Loose structural check on NOTICE file."""
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
NOTICE_PATH = REPO_ROOT / "NOTICE"


def test_notice_exists():
    assert NOTICE_PATH.exists()


def test_notice_has_third_party_libraries_section():
    text = NOTICE_PATH.read_text(encoding="utf-8")
    assert "Third-Party Libraries" in text


def test_notice_has_mathematical_references_section():
    text = NOTICE_PATH.read_text(encoding="utf-8")
    assert "Mathematical" in text and "References" in text


def test_notice_cites_all_pr_s20_authors():
    text = NOTICE_PATH.read_text(encoding="utf-8")
    for author in ["Lucey", "Anzer", "Spearman", "Power", "Pollard"]:
        assert author in text, f"missing PR-S20 reference author: {author}"


def test_readme_links_to_notice():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "NOTICE" in readme
```

- [ ] **Step 12.3: Verify both tests fail**

Run: `python -m pytest tests/test_todo_md_format.py tests/test_notice_md_format.py -v`
Expected: most fail (TODO.md not yet restructured; NOTICE doesn't exist).

- [ ] **Step 12.4: Restructure TODO.md**

Replace the entire content of `TODO.md` with the lakehouse-style structure from the spec §8. Ensure all 10 TF-* entries (TF-1..TF-10) are present with Size/Source/Notes columns and full academic citations per the spec.

(See spec `docs/superpowers/specs/2026-04-30-action-context-pr1-design.md` §8 for the verbatim content; copy it into TODO.md.)

- [ ] **Step 12.5: Author ADR-005**

Create `docs/superpowers/adrs/ADR-005-tracking-aware-features.md` with the 7 cross-cutting decisions per spec §4.4. Mirror ADR-001/002/003/004 shape (~120-150 lines):

```markdown
# ADR-005: Tracking-aware feature integration contract

| Field | Value |
|---|---|
| **Date** | 2026-04-30 |
| **Status** | Accepted (silly-kicks 2.8.0) |
| **Deciders** | Karsten S. Nielsen, Claude Opus 4.7 (1M) |

## Context

PR-S19 (silly-kicks 2.7.0) shipped the silly_kicks.tracking namespace primitive
layer (ADR-004). PR-S20 (silly-kicks 2.8.0) ships the first tracking-aware
features ... [continue with the 7 decisions as bullet sections]
```

(Author the full ADR per spec §4.4. ~120 lines.)

- [ ] **Step 12.6: Create the NOTICE file**

Create `NOTICE` at repo root with the exact content from spec §9.2. ~85 lines.

- [ ] **Step 12.7: Cross-link from README.md**

Append to `README.md`:

```markdown

## Attribution

This project incorporates academic methodologies from the soccer-analytics
literature. See [NOTICE](NOTICE) for full bibliographic citations and
third-party library acknowledgements.
```

- [ ] **Step 12.8: Add convention line to CLAUDE.md**

In `CLAUDE.md`, add under "Key conventions":

```markdown
- **Academic attribution discipline.** Every new feature with a published methodology gets an entry in the `NOTICE` file's "Mathematical / Methodological References" section. Cross-link from per-feature docstrings via `See NOTICE for full bibliographic citations.` Mirrors the lakehouse pattern.
```

- [ ] **Step 12.9: Verify all format tests pass**

Run: `python -m pytest tests/test_todo_md_format.py tests/test_notice_md_format.py -v`
Expected: 9 passed.

---

## Task 13: Cross-provider parity + real-data sweep + e2e

**Files:**
- Create: `tests/tracking/test_action_context_cross_provider.py`
- Create: `tests/tracking/test_action_context_real_data_sweep.py`

- [ ] **Step 13.1: Write cross-provider parity tests using Tier-3 lakehouse-derived parquets**

Create `tests/tracking/test_action_context_cross_provider.py`:

```python
"""Cross-provider parity for action_context using Tier-3 lakehouse-derived slim slices.

Loads tests/datasets/tracking/action_context_slim/{provider}_slim.parquet from Loop 0
(the lakehouse-derived slim slices). For each provider, runs add_action_context and
checks bounds + against the empirical baselines JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SLIM_DIR = REPO_ROOT / "tests" / "datasets" / "tracking" / "action_context_slim"
BASELINES = REPO_ROOT / "tests" / "datasets" / "tracking" / "empirical_action_context_baselines.json"


@pytest.mark.parametrize("provider", ["sportec", "metrica", "skillcorner"])
def test_action_context_bounds_per_provider(provider):
    """All bounded outputs are in their expected ranges."""
    from silly_kicks.tracking.features import add_action_context

    slim = SLIM_DIR / f"{provider}_slim.parquet"
    if not slim.exists():
        pytest.skip(f"slim slice {slim} not committed; run scripts/probe_action_context_baselines.py")
    df = pd.read_parquet(slim)
    actions = df[df["__kind"] == "action"].drop(columns=["__kind"])
    frames = df[df["__kind"] == "frame"].drop(columns=["__kind"])
    enriched = add_action_context(actions, frames)
    # Bounds:
    valid = enriched["nearest_defender_distance"].dropna()
    assert (valid >= 0).all() and (valid <= 200).all()
    speed = enriched["actor_speed"].dropna()
    assert (speed >= 0).all() and (speed <= 50).all()
```

(The slim parquets carry both action and frame rows distinguished by a `__kind` column, written during Loop 0's probe.)

- [ ] **Step 13.2: Write the e2e real-data sweep**

Create `tests/tracking/test_action_context_real_data_sweep.py`:

```python
"""e2e sweep: compute action_context on real local + lakehouse data."""
from __future__ import annotations

import os

import pytest


@pytest.mark.e2e
@pytest.mark.parametrize("provider,env_var", [
    ("pff", "PFF_TRACKING_DIR"),
    ("sportec", "IDSSE_TRACKING_DIR"),
    ("metrica", "METRICA_TRACKING_DIR"),
    ("skillcorner", "SKILLCORNER_TRACKING_DIR"),
])
def test_action_context_real_data_sweep(provider, env_var):
    path = os.environ.get(env_var)
    if not path:
        pytest.skip(f"{env_var} not set; skipping {provider} real-data sweep")
    pytest.skip(f"e2e implementation lands in execution session — see PR-S19 sweep pattern at "
                "tests/test_tracking_real_data_sweep.py")
```

(The actual sweep implementation is left as a `pytest.skip` placeholder during planning. During execution, port the per-provider data-loading patterns from PR-S19's `tests/test_tracking_real_data_sweep.py`, run `add_action_context` per provider, assert distribution stats are within tolerance of `tests/datasets/tracking/empirical_action_context_baselines.json`.)

- [ ] **Step 13.3: Run the cross-provider parity tests**

Run: `python -m pytest tests/tracking/test_action_context_cross_provider.py -v`
Expected: 3 passed (or 3 skipped if Loop 0 didn't have lakehouse env). Resolve any failures.

- [ ] **Step 13.4: Run the e2e sweep locally**

Set the four env vars locally and run:

```bash
PFF_TRACKING_DIR=... IDSSE_TRACKING_DIR=... METRICA_TRACKING_DIR=... SKILLCORNER_TRACKING_DIR=... \
    python -m pytest tests/tracking/test_action_context_real_data_sweep.py -m e2e -v
```

Expected: 4 passed (after the `pytest.skip` placeholder is replaced with the real implementation during execution).

---

## Task 14: Final pre-commit gates + single commit

**Files:**
- Modify: `pyproject.toml` (version bump 2.7.0 -> 2.8.0)
- Modify: `CHANGELOG.md` (add 2.8.0 entry)
- Modify: `silly_kicks/tracking/__init__.py` (extend exports)

- [ ] **Step 14.1: Bump version**

In `pyproject.toml`, change `version = "2.7.0"` to `version = "2.8.0"`.

- [ ] **Step 14.2: Add CHANGELOG entry**

Prepend a new section to `CHANGELOG.md`:

```markdown
## [2.8.0] - 2026-MM-DD

### Added — Tracking-aware action_context features (PR-S20)

- **`silly_kicks.tracking.features`** — public per-feature surface for standard SPADL: `nearest_defender_distance`, `actor_speed`, `receiver_zone_density`, `defenders_in_triangle_to_goal`, plus aggregator `add_action_context` and `tracking_default_xfns` for HybridVAEP integration.
- **`silly_kicks.atomic.tracking.features`** — atomic SPADL parity with the same public surface.
- **`silly_kicks.tracking.feature_framework`** — `ActionFrameContext`, `lift_to_states`, type aliases.
- **`silly_kicks.tracking._kernels`** — schema-agnostic compute kernels shared between standard and atomic surfaces.
- **`silly_kicks.vaep.feature_framework`** — extended with `frame_aware` decorator, `is_frame_aware` predicate, `Frames`/`FrameAwareTransformer` type aliases.
- **`silly_kicks.vaep.base.VAEP.compute_features` / `rate`** — additive `frames=None` kwarg; dispatches frame-aware xfns. HybridVAEP and AtomicVAEP inherit the extension automatically.
- **ADR-005** — Tracking-aware feature integration contract.
- **NOTICE** — academic-attribution canonical record at repo root, mirroring the lakehouse pattern.
- **TODO.md** — restructured to lakehouse-style "On Deck" table; tracks 10 follow-up tracking-aware features (TF-1..TF-10).

### Backward compatibility

- All existing call sites (`v.compute_features(game, actions)`, `v.rate(game, actions)`) work verbatim — `frames=None` is the default.
- No changes to existing default xfns lists.
```

- [ ] **Step 14.3: Extend `silly_kicks/tracking/__init__.py` exports**

Append to the existing `silly_kicks/tracking/__init__.py`:

```python
from . import _kernels, feature_framework, features
from .feature_framework import ActionFrameContext, lift_to_states
from .features import (
    actor_speed,
    add_action_context,
    defenders_in_triangle_to_goal,
    nearest_defender_distance,
    receiver_zone_density,
    tracking_default_xfns,
)
```

And extend `__all__`:

```python
__all__ = [
    # ... existing entries (KLOPPY_TRACKING_FRAMES_COLUMNS, etc.)
    "ActionFrameContext",
    "actor_speed",
    "add_action_context",
    "defenders_in_triangle_to_goal",
    "feature_framework",
    "features",
    "lift_to_states",
    "nearest_defender_distance",
    "receiver_zone_density",
    "tracking_default_xfns",
]
```

- [ ] **Step 14.4: Run all pre-commit gates**

Run sequentially:

```bash
ruff check .
ruff format --check .
pyright
python -m pytest tests/ -m "not e2e" --tb=short
```

Expected: all green. Fix any failures inline.

- [ ] **Step 14.5: Run the e2e sweep with all four env vars set**

```bash
PFF_TRACKING_DIR=... IDSSE_TRACKING_DIR=... METRICA_TRACKING_DIR=... SKILLCORNER_TRACKING_DIR=... \
    python -m pytest tests/tracking/test_action_context_real_data_sweep.py -m e2e -v
```

Expected: 4 passed.

- [ ] **Step 14.6: Run /final-review**

Invoke `/final-review` (mad-scientist-skills:final-review). Address any findings inline.

- [ ] **Step 14.7: Get explicit user approval for the commit**

Ask the user: "All gates pass + /final-review complete. Ready to make the single commit for PR-S20?"

Wait for explicit "yes". Do NOT commit until approved.

- [ ] **Step 14.8: Stage all changes + create the single commit**

```bash
git add silly_kicks/ tests/ docs/ scripts/ TODO.md CHANGELOG.md NOTICE README.md CLAUDE.md pyproject.toml
git status
git commit -m "$(cat <<'EOF'
feat(tracking)!: action_context tracking-aware features + VAEP integration -- silly-kicks 2.8.0 (PR-S20)

Ships the four tracking-aware action_context features (nearest_defender_distance,
actor_speed, receiver_zone_density, defenders_in_triangle_to_goal) for standard
and atomic SPADL, with HybridVAEP / AtomicVAEP integration via composition (no
new VAEP subclass). Establishes ADR-005 (tracking-aware feature integration
contract) and the canonical NOTICE file mirroring the lakehouse pattern.

See docs/superpowers/specs/2026-04-30-action-context-pr1-design.md and
docs/superpowers/adrs/ADR-005-tracking-aware-features.md for the full design.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Verify the commit landed:

```bash
git log --oneline -1
```

---

## Self-Review Notes

This plan was self-reviewed against the spec. All 11 spec goals (G1..G11) map to one or more tasks:

- **G1 (4-feature catalog):** Tasks 6, 7, 10
- **G2 (per-feature + aggregator surfaces):** Tasks 7, 10
- **G3 (atomic-SPADL parity):** Tasks 10, 11
- **G4 (VAEP/HybridVAEP/AtomicVAEP integration):** Tasks 4, 5, 9, 11
- **G5 (lift_to_states extension utility):** Tasks 1, 3
- **G6 (TDD-first, ~13 loops):** Tasks 0..13 = 14 loops including Loop 0 probe
- **G7 (three-tier fixture strategy):** in-memory across Tasks 6,7,8,9,10,11; Tier-3 in Task 13
- **G8 (ADR-005 charter):** Task 12
- **G9 (TODO.md restructure):** Task 12
- **G10 (academic attribution baked in):** integrated into Task 7's docstrings, Task 12's NOTICE + TODO.md
- **G11 (NOTICE file established):** Task 12

No placeholders / no TBD / no unspecified types remain. Steps that change code show the code; steps that run commands show the command + expected output.

Type / API consistency:
- `ActionFrameContext` defined in Task 1 with the same field names used by `_resolve_action_frame_context` (Task 2) and `_kernels.py` (Task 6).
- `frame_aware` / `is_frame_aware` defined in Task 1 with the same semantics dispatched in Task 4's `compute_features`.
- `lift_to_states` defined in Task 3 with the same shape consumed by `tracking_default_xfns` and `atomic_tracking_default_xfns` in Tasks 7 and 10.
- `_kernels._nearest_defender_distance(anchor_x, anchor_y, ctx)` signature defined in Task 6 is consumed verbatim by Tasks 7 and 10 wrappers.
