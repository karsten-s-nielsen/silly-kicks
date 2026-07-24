# TF-51 — Per-event Defensive Credit/Debit Family — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a per-event defensive credit/debit family — proximity-gated signed credit attributed to individual defenders, sized by shot xG or the attacker's xT at a turnover — as a new `silly_kicks/tracking/defensive_credit/` sub-package with 10 named rules + a per-team "bravery" rollup.

**Architecture:** A pure sub-package (`_params`, `_sizing`, `_resolution`, `_chaining`, `_rules`, `_bravery`, `_orchestration`) with three public entry points re-exported through `tracking/__init__.py`: `compute_defensive_credits` (long-form, one row per (action, credited player, rule)), `add_defensive_credit` (the per-action aggregate — the C4 +1 action-coupled aggregator, a thin `@nan_safe_enrichment` wrapper in `tracking/features.py`), and `compute_bravery` (event-only, per-team). Rules are pure functions over a shared per-invocation context (ONE `link_actions_to_frames`, ONE action-LTR reprojection, ONE `physical_grid` build). No `*_xfns` factory (F4 result-leakage — ADR-039/042), guarded by an executable absence test. No atomic mirror in v1.

**Tech Stack:** pandas / numpy; the injected fitted `ExpectedThreat` surface (`xthreat._physical`: `values_at_points` / `require_fitted_xt`), the injected per-shot `xg_column` port (fail-loud, `xtgk/_xg_reward` idiom), `link_actions_to_frames`, `reproject_to_action_ltr` / `acting_team_attacks_rtl` (ADR-028), `id_compat.ids_differ` (ADR-019), `add_possessions` / `resolve_next_touch_receiver`, the shipped `shot_blocked` / `cross_blocked` columns (4.56.0 / ADR-046).

**Spec:** `docs/superpowers/specs/2026-07-22-tf51-defensive-credit-design.md` (v5, plan-ready after two review rounds).

**Version/PR/ADR:** silly-kicks **4.57.0 / PR-S128 / ADR-047** (assigned at commit-prep; verified next-free against `origin/main` at 4.56.0 / PR-S127 / ADR-046).

---

## Conventions locked for this plan (read once)

**Rule vocabulary (closed set, §5 of the spec) — 10 names, used verbatim everywhere:**
`pressure_on_missed_shot`, `failed_pressure_shot_on_target`, `shot_block`, `pressure_pass_fail`, `recovery_double_credit`, `synchronized_final_third_pressure`, `forced_bad_touch`, `failed_cross_block`, `failed_marking_through_ball`, `beaten_1v1`.

**Long-form schema (`compute_defensive_credits` return), exact column names + dtypes:**
`game_id` (object), `period_id` (int64), `action_id` (object), `player_id` (source-dtype, Int64 when NaN-coded), `team_id` (source-dtype), `rule` (object, ∈ `DEFENSIVE_CREDIT_RULES`), `signed_value` (float64, NaN when fired-but-unsizable), `anchor_type` (object), `frame_id` (Int64), `sizing` (object, ∈ `{"xg", "xt"}`).

**Aggregate columns (`add_defensive_credit` adds):**
`defensive_credit_net` (float64), `defensive_credit_plus` (float64), `defensive_credit_minus` (float64), `n_defensive_credits` (Int64).

**Bravery columns (`compute_bravery` return, per (game_id, team_id)):**
`game_id`, `team_id`, `bravery_shots` (float64), `bravery_open_play_crosses` (float64), `bravery_set_piece_crosses` (float64 = NaN in v1), `bravery_pct_known_domain` (float64), `n_shots_faced` (Int64), `n_open_play_crosses_faced` (Int64), `n_set_piece_crosses_faced` (Int64), `n_blocks_known` (Int64).

**Sign convention:** `+` = defender removed danger; `−` = defender conceded it. The `−passer` rows belong to the *acting* team and are EXCLUDED from `add_defensive_credit`'s aggregate (defending-team scoping, R2-1) but kept in the long-form.

**Repo commands:**
- Tests: `python -m pytest tests/ -m "not e2e and not slow" -q`
- Lint: `python -m ruff check silly_kicks/ tests/` and `python -m ruff format --check <files>`
- Types: `python -m pyright` (whole repo — CI gates `silly_kicks` + `tests` + `scripts`)
- All `warnings.warn(...)` calls MUST pass `stacklevel=2`.

**Commit discipline:** this repo ships ONE commit per branch after explicit owner approval + `/final-review`. The `git commit` steps below are written per-task for a subagent-driven executor that squashes; if executing inline under this repo's policy, DO NOT run intermediate `git commit` — hold all changes uncommitted and make the single commit only at the end after `/final-review` + explicit approval. (See the repo commit policy.)

---

## File structure

```
silly_kicks/tracking/defensive_credit/
  __init__.py        # thin declarative re-export of the public names ONLY
  _params.py         # DefensiveCreditParams frozen dataclass; DEFENSIVE_CREDIT_RULES + per-rule constants; box geometry
  _sizing.py         # xg_of_shot(actions, xg_column) + extinguished_xt(points, xt) ports
  _resolution.py     # resolve_responsible_defenders(...) — nearest opponent(s), box-aware, ADR-028 + ADR-019
  _chaining.py       # resulting_shot_in_possession(...) + recovery_after_pass(...) — possession-scoped
  _rules.py          # RuleContext + one pure function per rule + RULE_REGISTRY
  _bravery.py        # compute_bravery(...) event-only, per-team, per-type breakdown
  _orchestration.py  # compute_defensive_credits(...) long-form + _aggregate_defensive_credit(...) bodies
```

Modified:
- `silly_kicks/tracking/features.py` — add the decorated `add_defensive_credit` thin wrapper (so the nan-safety/purity/liveness gates that scan `tracking.features` see it).
- `silly_kicks/tracking/__init__.py` — export the public surface.
- `docs/c4/architecture.dsl` + `docs/c4/architecture.html` — bump aggregator count 30 → 31.
- `NOTICE`, `CLAUDE.md`, `TODO.md`, `CHANGELOG.md`, `pyproject.toml`, `silly_kicks/__init__.py`, `uv.lock` — attribution + version bump at the end.

New tests:
```
tests/tracking/test_defensive_credit_params.py
tests/tracking/test_defensive_credit_sizing.py
tests/tracking/test_defensive_credit_resolution.py
tests/tracking/test_defensive_credit_chaining.py
tests/tracking/test_defensive_credit_rules.py            # all 10 rules, both-sides fixtures
tests/tracking/test_defensive_credit_orchestration.py    # long-form assembly + closed vocab + mirror-invariance + sizing regression + rules-gating
tests/tracking/test_defensive_credit_aggregate.py        # add_defensive_credit defending-scoped, always-finite
tests/tracking/test_bravery.py                           # per-type breakdown + set-piece exposure + worked example
tests/tracking/test_defensive_credit_xfns_absence_guard.py
tests/tracking/test_defensive_credit_perf_budget.py
tests/spadl/test_block_detection_contract.py             # EXTEND: cross_blocked ⊆ cross-type invariant
tests/tracking/test_defensive_credit_e2e.py              # @e2e owner-gated
```
Registry edits (existing files): `tests/test_add_star_purity.py`, `tests/test_enrichment_nan_safety.py`, `tests/tracking/test_aggregator_column_liveness.py`, `tests/tracking/conftest_id_dtype.py`.

---

## Shared test helpers (build once, reuse across rule tests)

Several tests need a tiny synthetic (actions, frames) pair with a controllable defender at a known distance. Task 3 creates `tests/tracking/_defensive_credit_fixtures.py` with builders the later tasks import. Do NOT duplicate fixture code — import from there.

---

### Task 1: Package scaffold + `DefensiveCreditParams` + `DEFENSIVE_CREDIT_RULES`

**Files:**
- Create: `silly_kicks/tracking/defensive_credit/__init__.py`
- Create: `silly_kicks/tracking/defensive_credit/_params.py`
- Test: `tests/tracking/test_defensive_credit_params.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/tracking/test_defensive_credit_params.py
import pytest

from silly_kicks.tracking.defensive_credit._params import (
    DEFENSIVE_CREDIT_RULES,
    DefensiveCreditParams,
    is_inside_attacked_box,
)


def test_rules_is_closed_tuple_of_ten():
    assert isinstance(DEFENSIVE_CREDIT_RULES, tuple)
    assert len(DEFENSIVE_CREDIT_RULES) == 10
    assert len(set(DEFENSIVE_CREDIT_RULES)) == 10  # no dupes
    assert "shot_block" in DEFENSIVE_CREDIT_RULES
    assert "pressure_pass_fail" in DEFENSIVE_CREDIT_RULES


def test_defaults_match_spec():
    p = DefensiveCreditParams()
    assert p.proximity_outside_box_m == 4.5
    assert p.proximity_inside_box_m == 3.0
    assert p.resulting_shot_max_actions == 10
    assert p.recovery_max_actions == 3
    assert p.through_ball_delta_xt_min == 0.02
    assert p.beaten_1v1_min_shot_xg == 0.05
    # synchronized boundary derived from the pitch third (105 / 3)
    assert p.synchronized_zone_boundary_x == pytest.approx(35.0)
    assert set(p.rules) == set(DEFENSIVE_CREDIT_RULES)  # all enabled by default


def test_rules_subset_validation():
    with pytest.raises(ValueError, match="unknown rule"):
        DefensiveCreditParams(rules=frozenset({"not_a_rule"}))


def test_negative_proximity_rejected():
    with pytest.raises(ValueError, match="proximity"):
        DefensiveCreditParams(proximity_outside_box_m=-1.0)


def test_box_membership_action_ltr():
    # attacked goal at x=105; box is x >= 105-16.5 = 88.5, |y-34| <= 20.16
    assert is_inside_attacked_box(100.0, 34.0) is True
    assert is_inside_attacked_box(100.0, 10.0) is False  # wide of the box
    assert is_inside_attacked_box(80.0, 34.0) is False  # short of the box
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/tracking/test_defensive_credit_params.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'silly_kicks.tracking.defensive_credit'`.

- [ ] **Step 3: Create the package `__init__.py` (thin re-export — fill in as later tasks land)**

```python
# silly_kicks/tracking/defensive_credit/__init__.py
"""TF-51 — per-event defensive credit/debit family.

See NOTICE for full bibliographic citations (Sumpter, Soccermatics Pro module 16.3;
Bischofberger/Bauer/Baca, arXiv:2606.19931).
"""

from ._params import DEFENSIVE_CREDIT_RULES, DefensiveCreditParams

__all__ = [
    "DEFENSIVE_CREDIT_RULES",
    "DefensiveCreditParams",
]
```

- [ ] **Step 4: Implement `_params.py`**

```python
# silly_kicks/tracking/defensive_credit/_params.py
"""Frozen params + the closed rule vocabulary + box geometry for TF-51."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from silly_kicks.spadl import config as spadlconfig

# --- pitch + box geometry (action-LTR: acting team attacks x=105) ---
_FIELD_LENGTH: float = spadlconfig.field_length  # 105.0
_FIELD_WIDTH: float = spadlconfig.field_width  # 68.0
# Penalty-area geometry. spadlconfig ships no canonical box constant (see CLAUDE.md /
# ADR-019 discussion); the repo duplicates it. We adopt _xcross_attempt.py's values
# (16.5 depth, 20.16 half-width = 40.32/2). NOTE the cross-module discrepancy: _ghost_gk.py
# uses 40.3 (half 20.15). 0.01 m apart; neither cites the other. We pick 40.32 (the FIFA
# Laws figure) and flag it here rather than silently choosing.
_BOX_DEPTH_M: float = 16.5
_BOX_HALF_WIDTH_M: float = 20.16
_GOAL_Y_C: float = _FIELD_WIDTH / 2.0  # 34.0

# --- closed rule vocabulary (DAS_SOURCE_VALUES pattern) ---
RULE_PRESSURE_ON_MISSED_SHOT = "pressure_on_missed_shot"
RULE_FAILED_PRESSURE_SHOT_ON_TARGET = "failed_pressure_shot_on_target"
RULE_SHOT_BLOCK = "shot_block"
RULE_PRESSURE_PASS_FAIL = "pressure_pass_fail"
RULE_RECOVERY_DOUBLE_CREDIT = "recovery_double_credit"
RULE_SYNCHRONIZED_FINAL_THIRD_PRESSURE = "synchronized_final_third_pressure"
RULE_FORCED_BAD_TOUCH = "forced_bad_touch"
RULE_FAILED_CROSS_BLOCK = "failed_cross_block"
RULE_FAILED_MARKING_THROUGH_BALL = "failed_marking_through_ball"
RULE_BEATEN_1V1 = "beaten_1v1"

DEFENSIVE_CREDIT_RULES: tuple[str, ...] = (
    RULE_PRESSURE_ON_MISSED_SHOT,
    RULE_FAILED_PRESSURE_SHOT_ON_TARGET,
    RULE_SHOT_BLOCK,
    RULE_PRESSURE_PASS_FAIL,
    RULE_RECOVERY_DOUBLE_CREDIT,
    RULE_SYNCHRONIZED_FINAL_THIRD_PRESSURE,
    RULE_FORCED_BAD_TOUCH,
    RULE_FAILED_CROSS_BLOCK,
    RULE_FAILED_MARKING_THROUGH_BALL,
    RULE_BEATEN_1V1,
)

SIZING_XG = "xg"
SIZING_XT = "xt"
SIZING_VALUES: tuple[str, ...] = (SIZING_XG, SIZING_XT)


def is_inside_attacked_box(x: float, y: float) -> bool:
    """True iff (x, y) in action-LTR coords is inside the attacked penalty area (goal at x=105)."""
    return bool((x >= _FIELD_LENGTH - _BOX_DEPTH_M) and (abs(y - _GOAL_Y_C) <= _BOX_HALF_WIDTH_M))


@dataclass(frozen=True)
class DefensiveCreditParams:
    """All fields spec-frozen / intent-set — never calibrated (see spec §4.2, §14)."""

    proximity_outside_box_m: float = 4.5
    proximity_inside_box_m: float = 3.0
    synchronized_zone_boundary_x: float = field(default_factory=lambda: _FIELD_LENGTH / 3.0)
    resulting_shot_max_actions: int = 10
    recovery_max_actions: int = 3
    through_ball_delta_xt_min: float = 0.02  # provisional
    beaten_1v1_min_shot_xg: float = 0.05  # provisional
    rules: frozenset[str] = field(default_factory=lambda: frozenset(DEFENSIVE_CREDIT_RULES))

    def __post_init__(self) -> None:
        for name, val in (
            ("proximity_outside_box_m", self.proximity_outside_box_m),
            ("proximity_inside_box_m", self.proximity_inside_box_m),
        ):
            if not val > 0:
                raise ValueError(f"{name} must be > 0, got {val}")
        if self.resulting_shot_max_actions < 1 or self.recovery_max_actions < 1:
            raise ValueError("resulting_shot_max_actions and recovery_max_actions must be >= 1")
        unknown = set(self.rules) - set(DEFENSIVE_CREDIT_RULES)
        if unknown:
            raise ValueError(f"unknown rule(s): {sorted(unknown)}; allowed: {DEFENSIVE_CREDIT_RULES}")

    def proximity_threshold(self, x: float, y: float) -> float:
        """Box-aware marking/pressure radius at an action-LTR anchor location."""
        return self.proximity_inside_box_m if is_inside_attacked_box(x, y) else self.proximity_outside_box_m
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `python -m pytest tests/tracking/test_defensive_credit_params.py -q`
Expected: PASS (5 tests).

- [ ] **Step 6: Lint + types**

Run: `python -m ruff check silly_kicks/tracking/defensive_credit/ tests/tracking/test_defensive_credit_params.py` — Expected: `All checks passed!`
Run: `python -m ruff format silly_kicks/tracking/defensive_credit/ tests/tracking/test_defensive_credit_params.py`

- [ ] **Step 7: Commit** (subagent-driven executors only — see commit discipline note)

```bash
git add silly_kicks/tracking/defensive_credit/__init__.py silly_kicks/tracking/defensive_credit/_params.py tests/tracking/test_defensive_credit_params.py
git commit -m "feat(tracking): TF-51 params + closed rule vocabulary"
```

---

### Task 2: Sizing ports (`_sizing.py`) — xG + extinguished xT

**Files:**
- Create: `silly_kicks/tracking/defensive_credit/_sizing.py`
- Test: `tests/tracking/test_defensive_credit_sizing.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/tracking/test_defensive_credit_sizing.py
import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking.defensive_credit._sizing import extinguished_xt, xg_of_shot


def test_xg_of_shot_reads_injected_column():
    actions = pd.DataFrame({"action_id": [7], "xg": [0.23]})
    assert xg_of_shot(actions.iloc[0], xg_column="xg") == pytest.approx(0.23)


def test_xg_of_shot_nan_passes_through():
    actions = pd.DataFrame({"action_id": [7], "xg": [np.nan]})
    assert np.isnan(xg_of_shot(actions.iloc[0], xg_column="xg"))


def test_xg_of_shot_fails_loud_when_column_absent():
    actions = pd.DataFrame({"action_id": [7]})
    with pytest.raises(ValueError, match="xg_column"):
        xg_of_shot(actions.iloc[0], xg_column="xg")


def test_extinguished_xt_reads_fitted_surface(fitted_xt):
    # a deep-in-attack point should have higher xT than a deep-own-half point
    vals = extinguished_xt([(95.0, 34.0), (10.0, 34.0)], fitted_xt)
    assert vals[0] > vals[1]


def test_extinguished_xt_requires_fitted(unfitted_xt):
    with pytest.raises((ValueError, RuntimeError)):  # require_fitted_xt raises NotFittedError family
        extinguished_xt([(50.0, 34.0)], unfitted_xt)
```

Add fixtures `fitted_xt` / `unfitted_xt` to `tests/tracking/conftest.py` if not already present (reuse the existing xT fixture factory used by `tests/tracking/test_aggregator_column_liveness.py::_fresh_xt` — import and expose it):

```python
# tests/tracking/conftest.py  (append; skip if a fitted_xt fixture already exists)
import pytest
from tests.tracking.test_aggregator_column_liveness import _fresh_xt  # a small fitted ExpectedThreat


@pytest.fixture
def fitted_xt():
    return _fresh_xt()


@pytest.fixture
def unfitted_xt():
    from silly_kicks.xthreat import ExpectedThreat
    return ExpectedThreat()  # constructed, never .fit()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/tracking/test_defensive_credit_sizing.py -q`
Expected: FAIL — `ModuleNotFoundError: ... _sizing`.

- [ ] **Step 3: Implement `_sizing.py`**

```python
# silly_kicks/tracking/defensive_credit/_sizing.py
"""Sizing ports: xG per shot (injected column) + extinguished xT (injected fitted surface)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.xthreat import require_fitted_xt, values_at_points


def xg_of_shot(shot_action: pd.Series, *, xg_column: str) -> float:
    """Return the injected per-shot xG. Fail-loud if the column is absent (xtgk/_xg_reward idiom).

    xG is a *pre-block* quantity, so a blocked shot still carries a value (shot_block sizing needs it).
    A present-but-NaN xG passes through as NaN -> a fired-but-unsizable long-form row.
    """
    if xg_column not in shot_action.index:
        raise ValueError(
            f"xg_column {xg_column!r} not found on the shot action. Supply a calibrated per-shot xG "
            f"column (silly-kicks ships no xG model; see spec §7)."
        )
    return float(shot_action[xg_column])


def extinguished_xt(points, xt) -> np.ndarray:
    """xT at each action-LTR (x, y) point on the injected fitted surface (the threat extinguished).

    ``points``: iterable of (x, y) in action-LTR metres (attacked goal at x=105).
    NaN coords -> NaN value (values_at_points is NaN-tolerant).
    """
    require_fitted_xt(xt, caller="defensive_credit.extinguished_xt")
    if len(points) == 0:
        return np.array([], dtype="float64")
    xs = np.asarray([p[0] for p in points], dtype="float64")
    ys = np.asarray([p[1] for p in points], dtype="float64")
    return values_at_points(xt, xs, ys)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/tracking/test_defensive_credit_sizing.py -q` — Expected: PASS (5).

- [ ] **Step 5: Lint + commit**

Run: `python -m ruff check silly_kicks/tracking/defensive_credit/_sizing.py tests/tracking/test_defensive_credit_sizing.py && python -m ruff format silly_kicks/tracking/defensive_credit/_sizing.py`
```bash
git add silly_kicks/tracking/defensive_credit/_sizing.py tests/tracking/test_defensive_credit_sizing.py tests/tracking/conftest.py
git commit -m "feat(tracking): TF-51 sizing ports (xG + extinguished xT)"
```

---

### Task 3: Responsible-defender resolution (`_resolution.py`) + shared fixtures

**Files:**
- Create: `silly_kicks/tracking/defensive_credit/_resolution.py`
- Create: `tests/tracking/_defensive_credit_fixtures.py`
- Test: `tests/tracking/test_defensive_credit_resolution.py`

**What it does (spec §6):** given a triggering action, an anchor reference point in action-LTR, and the linked frame, return the nearest opponent(s) (team_id ≠ acting team, compared with `ids_differ`) within the box-aware threshold — with three modes: `nearest` (single), `all_within`, `all_within_beyond_nearest`. Frame player positions are reprojected to action-LTR (ADR-028). Empty within-threshold set → no defender (the rule will emit no row).

- [ ] **Step 1: Build the shared fixtures file**

```python
# tests/tracking/_defensive_credit_fixtures.py
"""Synthetic (actions, frames) builders for TF-51 rule tests. Import from here; never duplicate."""

from __future__ import annotations

import numpy as np
import pandas as pd


def one_action(
    *,
    action_id=1,
    type_name="shot",
    result_name="fail",
    team_id=10,
    player_id=100,
    start_x=95.0,
    start_y=34.0,
    end_x=105.0,
    end_y=34.0,
    period_id=1,
    time_seconds=50.0,
    game_id="g1",
    **extra,
) -> pd.DataFrame:
    from silly_kicks.spadl import config as spadlconfig

    row = {
        "game_id": game_id,
        "period_id": period_id,
        "action_id": action_id,
        "time_seconds": time_seconds,
        "team_id": team_id,
        "player_id": player_id,
        "type_id": spadlconfig.actiontype_id[type_name],
        "result_id": spadlconfig.result_id[result_name],
        "bodypart_id": spadlconfig.bodypart_id["foot"],
        "start_x": start_x,
        "start_y": start_y,
        "end_x": end_x,
        "end_y": end_y,
    }
    row.update(extra)
    return pd.DataFrame([row])


def frame_with_defender(
    *,
    action_time=50.0,
    period_id=1,
    game_id="g1",
    acting_team_id=10,
    defender_team_id=20,
    defender_x=96.0,
    defender_y=34.0,
    frame_id=500,
    home_team_id=10,
) -> pd.DataFrame:
    """One frame at the action time with a single defender at (defender_x, defender_y).

    Home team attacks x=105 (convention of convert_to_frames); team_attacking_direction encodes it.
    For acting_team == home, action-LTR == frame coords (no reprojection). For the away-action
    mirror test, pass acting_team_id != home_team_id and place the defender in FRAME coords.
    """
    rows = []
    for team, x, y in ((defender_team_id, defender_x, defender_y),):
        rows.append(
            {
                "game_id": game_id,
                "period_id": period_id,
                "frame_id": frame_id,
                "time_seconds": action_time,
                "team_id": team,
                "player_id": 900,
                "x": x,
                "y": y,
                "vx": 0.0,
                "vy": 0.0,
                "is_ball": False,
                "is_goalkeeper": False,
                "team_attacking_direction": 1 if home_team_id == acting_team_id else -1,
                "home_team_id": home_team_id,
            }
        )
    # a ball row (linkage/geometry helpers tolerate its presence)
    rows.append(
        {
            "game_id": game_id, "period_id": period_id, "frame_id": frame_id, "time_seconds": action_time,
            "team_id": np.nan, "player_id": np.nan, "x": 100.0, "y": 34.0, "vx": 0.0, "vy": 0.0,
            "is_ball": True, "is_goalkeeper": False, "team_attacking_direction": 1, "home_team_id": home_team_id,
        }
    )
    return pd.DataFrame(rows)
```

- [ ] **Step 2: Write the failing resolution test**

```python
# tests/tracking/test_defensive_credit_resolution.py
import pandas as pd

from silly_kicks.tracking.defensive_credit._params import DefensiveCreditParams
from silly_kicks.tracking.defensive_credit._resolution import resolve_responsible_defenders
from tests.tracking._defensive_credit_fixtures import frame_with_defender, one_action


def _ctx(defender_x):
    actions = one_action(start_x=95.0, start_y=34.0)
    frames = frame_with_defender(defender_x=defender_x, defender_y=34.0)
    return actions, frames


def test_defender_within_threshold_is_returned():
    actions, frames = _ctx(defender_x=96.0)  # 1.0 m from anchor (95,34), inside outside-box radius 4.5
    res = resolve_responsible_defenders(
        actions, frames, anchor_x=95.0, anchor_y=34.0, acting_team_id=10,
        mode="nearest", params=DefensiveCreditParams(),
    )
    assert list(res["player_id"]) == [900]
    assert list(res["team_id"]) == [20]


def test_defender_outside_threshold_returns_empty():
    actions, frames = _ctx(defender_x=90.0)  # 5.0 m from (95,34) > 4.5
    res = resolve_responsible_defenders(
        actions, frames, anchor_x=95.0, anchor_y=34.0, acting_team_id=10,
        mode="nearest", params=DefensiveCreditParams(),
    )
    assert res.empty


def test_all_within_beyond_nearest_drops_the_closest():
    actions = one_action(start_x=95.0, start_y=34.0)
    # two defenders within 4.5 m: one at 1 m, one at 2 m
    import numpy as np
    frames = pd.concat([
        frame_with_defender(defender_x=96.0),  # 1 m -> player 900
    ], ignore_index=True)
    # add a second defender row manually
    extra = frames.iloc[[0]].copy()
    extra["player_id"] = 901
    extra["x"] = 97.0  # 2 m
    frames = pd.concat([frames, extra], ignore_index=True)
    res = resolve_responsible_defenders(
        actions, frames, anchor_x=95.0, anchor_y=34.0, acting_team_id=10,
        mode="all_within_beyond_nearest", params=DefensiveCreditParams(),
    )
    assert set(res["player_id"]) == {901}  # nearest (900) dropped
```

- [ ] **Step 3: Run to verify it fails**

Run: `python -m pytest tests/tracking/test_defensive_credit_resolution.py -q`
Expected: FAIL — `ModuleNotFoundError: ... _resolution`.

- [ ] **Step 4: Implement `_resolution.py`**

```python
# silly_kicks/tracking/defensive_credit/_resolution.py
"""Nearest-opponent(s) resolution within a box-aware threshold, in action-LTR coords."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from silly_kicks.id_compat import ids_match
from silly_kicks.tracking._action_orientation import acting_team_attacks_rtl

from ._params import DefensiveCreditParams

Mode = Literal["nearest", "all_within", "all_within_beyond_nearest"]

_FIELD_LENGTH = 105.0
_FIELD_WIDTH = 68.0


def resolve_responsible_defenders(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    anchor_x: float,
    anchor_y: float,
    acting_team_id,
    mode: Mode,
    params: DefensiveCreditParams,
    frame_id: int | None = None,
) -> pd.DataFrame:
    """Return opponents within the box-aware threshold of the (anchor_x, anchor_y) action-LTR point.

    Columns: player_id, team_id, distance_m (ascending). Empty when none are within threshold.
    ``frame_id``: the linked frame for the triggering action; if None, uses the single frame present
    (the per-invocation orchestrator passes the resolved frame_id).
    """
    if frame_id is not None:
        fr = frames[frames["frame_id"] == frame_id]
    else:
        fr = frames
    # opponents only (team_id != acting team) — dtype-safe (ADR-019)
    is_opponent = ~ids_match(fr["team_id"], acting_team_id) & fr["team_id"].notna() & ~fr["is_ball"]
    opp = fr[is_opponent].copy()
    if opp.empty:
        return _empty()

    # reproject opponent positions to action-LTR for THIS action (ADR-028). The single action here
    # shares one flip decision; the orchestrator computes acting_team_attacks_rtl once for the batch.
    flip = bool(acting_team_attacks_rtl(actions, frames).iloc[0])
    px = _FIELD_LENGTH - opp["x"].to_numpy() if flip else opp["x"].to_numpy()
    py = _FIELD_WIDTH - opp["y"].to_numpy() if flip else opp["y"].to_numpy()

    dist = np.hypot(px - anchor_x, py - anchor_y)
    thr = params.proximity_threshold(anchor_x, anchor_y)
    within = dist <= thr
    if not within.any():
        return _empty()

    out = pd.DataFrame(
        {"player_id": opp["player_id"].to_numpy()[within], "team_id": opp["team_id"].to_numpy()[within],
         "distance_m": dist[within]}
    ).sort_values("distance_m", kind="stable").reset_index(drop=True)

    if mode == "nearest":
        return out.iloc[[0]].reset_index(drop=True)
    if mode == "all_within_beyond_nearest":
        return out.iloc[1:].reset_index(drop=True)
    return out  # all_within


def _empty() -> pd.DataFrame:
    return pd.DataFrame({"player_id": [], "team_id": [], "distance_m": pd.Series([], dtype="float64")})
```

> **Perf note (M3):** this per-action form is used only inside the batched orchestrator (Task 8), which computes `acting_team_attacks_rtl(actions, frames)` and the link ONCE and passes the resolved `frame_id`. The `acting_team_attacks_rtl(...).iloc[0]` call inside is a single-action convenience for the unit test; Task 8 passes the precomputed flip via the RuleContext so the orchestrator does not recompute per rule.

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/tracking/test_defensive_credit_resolution.py -q` — Expected: PASS (3).

- [ ] **Step 6: Lint + commit**

```bash
git add silly_kicks/tracking/defensive_credit/_resolution.py tests/tracking/_defensive_credit_fixtures.py tests/tracking/test_defensive_credit_resolution.py
git commit -m "feat(tracking): TF-51 responsible-defender resolution + shared fixtures"
```

---

### Task 4: Chaining (`_chaining.py`) — resulting shot + recovery

**Files:**
- Create: `silly_kicks/tracking/defensive_credit/_chaining.py`
- Test: `tests/tracking/test_defensive_credit_chaining.py`

**What it does (spec §8):** both resolutions are possession-scoped via `add_possessions` (adds `possession_id`, int64), boundary-guarded on `(game_id, period_id, possession_id)`.
- `resulting_shot_in_possession(actions, anchor_idx, *, max_actions)`: the first shot by the **attacking** team in the anchor's possession within `max_actions` forward rows; returns the shot row (or None).
- `recovery_after_pass(actions, pass_idx, *, max_actions)`: the first defending-team ball-regain within `max_actions` rows of the failed pass; returns the recovery row (or None). NaN-team rows skipped (ADR-027).

- [ ] **Step 1: Write the failing test**

```python
# tests/tracking/test_defensive_credit_chaining.py
import pandas as pd

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking.defensive_credit._chaining import (
    recovery_after_pass,
    resulting_shot_in_possession,
    with_possessions,
)


def _stream(rows):
    base = dict(game_id="g1", period_id=1, bodypart_id=spadlconfig.bodypart_id["foot"],
                start_x=60.0, start_y=34.0, end_x=70.0, end_y=34.0)
    out = []
    for i, r in enumerate(rows):
        d = dict(base)
        d.update(action_id=i, time_seconds=float(i), **r)
        d["type_id"] = spadlconfig.actiontype_id[d.pop("type_name")]
        d["result_id"] = spadlconfig.result_id[d.pop("result_name")]
        out.append(d)
    return with_possessions(pd.DataFrame(out))


def test_resulting_shot_found_in_same_possession():
    actions = _stream([
        {"type_name": "pass", "result_name": "success", "team_id": 10, "player_id": 1},
        {"type_name": "dribble", "result_name": "success", "team_id": 10, "player_id": 2},
        {"type_name": "shot", "result_name": "fail", "team_id": 10, "player_id": 3},
    ])
    shot = resulting_shot_in_possession(actions, 0, attacking_team_id=10, max_actions=10)
    assert shot is not None
    assert shot["action_id"] == 2


def test_resulting_shot_none_when_no_shot():
    actions = _stream([
        {"type_name": "pass", "result_name": "success", "team_id": 10, "player_id": 1},
        {"type_name": "pass", "result_name": "fail", "team_id": 10, "player_id": 2},
    ])
    assert resulting_shot_in_possession(actions, 0, attacking_team_id=10, max_actions=10) is None


def test_recovery_after_failed_pass():
    actions = _stream([
        {"type_name": "pass", "result_name": "fail", "team_id": 10, "player_id": 1},
        {"type_name": "interception", "result_name": "success", "team_id": 20, "player_id": 99},
    ])
    rec = recovery_after_pass(actions, 0, max_actions=3)
    assert rec is not None
    assert rec["player_id"] == 99


def test_recovery_none_beyond_cap():
    actions = _stream([
        {"type_name": "pass", "result_name": "fail", "team_id": 10, "player_id": 1},
        {"type_name": "dribble", "result_name": "success", "team_id": 10, "player_id": 2},
        {"type_name": "dribble", "result_name": "success", "team_id": 10, "player_id": 2},
        {"type_name": "dribble", "result_name": "success", "team_id": 10, "player_id": 2},
        {"type_name": "interception", "result_name": "success", "team_id": 20, "player_id": 99},
    ])
    assert recovery_after_pass(actions, 0, max_actions=3) is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/tracking/test_defensive_credit_chaining.py -q` — Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement `_chaining.py`**

```python
# silly_kicks/tracking/defensive_credit/_chaining.py
"""Possession-scoped resulting-shot + recovery resolvers."""

from __future__ import annotations

import pandas as pd

from silly_kicks.id_compat import same_id
from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl.utils import add_possessions

_SHOT_TYPE_IDS = frozenset(
    spadlconfig.actiontype_id[t] for t in ("shot", "shot_penalty", "shot_freekick")
)


def with_possessions(actions: pd.DataFrame) -> pd.DataFrame:
    """Attach possession_id (int64), sorted (game_id, period_id, action_id). Pure — returns a copy."""
    return add_possessions(actions)


def resulting_shot_in_possession(actions, anchor_idx, *, attacking_team_id, max_actions):
    """First shot by attacking_team_id in the anchor's possession, within max_actions forward rows."""
    anchor = actions.iloc[anchor_idx]
    same_poss = (
        (actions["game_id"] == anchor["game_id"])
        & (actions["period_id"] == anchor["period_id"])
        & (actions["possession_id"] == anchor["possession_id"])
    )
    fwd = actions[same_poss & (actions.index > anchor_idx)].head(max_actions)
    for _, r in fwd.iterrows():
        if r["type_id"] in _SHOT_TYPE_IDS and same_id(r["team_id"], attacking_team_id):
            return r
    return None


def recovery_after_pass(actions, pass_idx, *, max_actions):
    """First OPPONENT ball-regain within max_actions rows of the failed pass. NaN-team skipped.

    The defending team is inferred as the first team != the pass's acting team (two-team match) —
    the SINGLE recovery resolver (P-3: no duplicate in _rules). Returns the recovery row or None.
    """
    passer_team = actions.iloc[pass_idx]["team_id"]
    fwd = actions.iloc[pass_idx + 1 : pass_idx + 1 + max_actions]
    for _, r in fwd.iterrows():
        if pd.isna(r["team_id"]):
            continue  # ADR-027: NaN-team rows never decide
        if not same_id(r["team_id"], passer_team):  # first opponent regain
            return r
    return None
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/tracking/test_defensive_credit_chaining.py -q` — Expected: PASS (4).

- [ ] **Step 5: Lint + commit**

```bash
git add silly_kicks/tracking/defensive_credit/_chaining.py tests/tracking/test_defensive_credit_chaining.py
git commit -m "feat(tracking): TF-51 possession-scoped chaining (resulting shot + recovery)"
```

---

### Task 5: Rule engine skeleton + the three shot rules (`_rules.py`)

**Files:**
- Create: `silly_kicks/tracking/defensive_credit/_rules.py`
- Test: `tests/tracking/test_defensive_credit_rules.py` (create; extended in Tasks 6–7)

**Design:** `RuleContext` bundles everything a rule needs (the action row, its integer position `idx`, the acting team, the resolved `frame_id`, the full `actions`/`frames`, the sizing ports bound to `xg_column`/`xt`, the chaining resolvers, `params`). Each rule is a pure function `(ctx) -> list[CreditRow]`; `CreditRow` is a dataclass matching the long-form schema. `RULE_REGISTRY: dict[str, Callable]` maps rule name → function.

The three shot rules are a **mutually-exclusive partition** of shot outcomes (spec §5): blocked → `shot_block`; on-target/goal → `failed_pressure_shot_on_target`; off-target → `pressure_on_missed_shot`.

- [ ] **Step 1: Write the failing shot-rule tests**

```python
# tests/tracking/test_defensive_credit_rules.py
import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking.defensive_credit._params import DefensiveCreditParams
from silly_kicks.tracking.defensive_credit._rules import RULE_REGISTRY, RuleContext
from tests.tracking._defensive_credit_fixtures import frame_with_defender, one_action


def _shot_ctx(*, result_name, shot_blocked=pd.NA, xg=0.2, defender_x=96.0, fitted_xt=None, on_target=pd.NA):
    """on_target: nullable-boolean the orchestrator normally computes (goal / provider / TF-48).
    Set it explicitly in unit tests since synthetic frames can't derive shot_on_target_derived."""
    actions = one_action(type_name="shot", result_name=result_name, start_x=95.0, start_y=34.0)
    actions["shot_blocked"] = pd.array([shot_blocked], dtype="boolean")
    actions["_on_target"] = pd.array([on_target], dtype="boolean")
    actions["xg"] = [xg]
    frames = frame_with_defender(defender_x=defender_x, defender_y=34.0)
    return RuleContext.build_single(
        actions, frames, idx=0, xg_column="xg", xt=fitted_xt,
        blocked_column="shot_blocked", params=DefensiveCreditParams(),
    )


def test_pressure_on_missed_shot_credits_nearest_defender(fitted_xt):
    # OFF-target (a genuine miss): result fail, not blocked, on_target definitively False -> +credit
    ctx = _shot_ctx(result_name="fail", shot_blocked=False, xg=0.2, on_target=False, fitted_xt=fitted_xt)
    rows = RULE_REGISTRY["pressure_on_missed_shot"](ctx)
    assert len(rows) == 1
    assert rows[0].rule == "pressure_on_missed_shot"
    assert rows[0].signed_value == pytest.approx(0.2)  # +xG, off-target
    assert rows[0].player_id == 900
    assert rows[0].sizing == "xg"


def test_failed_pressure_shot_on_target_debits_defender_on_goal(fitted_xt):
    # a GOAL is on-target (result success -> on_target True by construction)
    ctx = _shot_ctx(result_name="success", shot_blocked=False, xg=0.3, on_target=True, fitted_xt=fitted_xt)
    rows = RULE_REGISTRY["failed_pressure_shot_on_target"](ctx)
    assert len(rows) == 1
    assert rows[0].signed_value == pytest.approx(-0.3)  # -xG, on-target


def test_pressured_saved_shot_is_negative(fitted_xt):
    # THE P-1 regression: a SAVED shot is result=fail but ON-target -> failed_pressure (NEGATIVE),
    # NOT pressure_on_missed_shot (+credit). SPADL result can't tell saved from off-target; _on_target does.
    ctx = _shot_ctx(result_name="fail", shot_blocked=False, xg=0.3, on_target=True, fitted_xt=fitted_xt)
    assert RULE_REGISTRY["failed_pressure_shot_on_target"](ctx)[0].signed_value == pytest.approx(-0.3)
    assert RULE_REGISTRY["pressure_on_missed_shot"](ctx) == []  # must NOT +credit a saved shot


def test_unknown_on_target_fires_neither_pressure_rule(fitted_xt):
    # on_target unknown (NA) -> we do NOT fabricate a sign; neither pressure rule fires (no row).
    ctx = _shot_ctx(result_name="fail", shot_blocked=False, xg=0.2, on_target=pd.NA, fitted_xt=fitted_xt)
    assert RULE_REGISTRY["pressure_on_missed_shot"](ctx) == []
    assert RULE_REGISTRY["failed_pressure_shot_on_target"](ctx) == []


def test_shot_block_credits_the_blocker(fitted_xt):
    ctx = _shot_ctx(result_name="fail", shot_blocked=True, xg=0.25, on_target=pd.NA, fitted_xt=fitted_xt)
    rows = RULE_REGISTRY["shot_block"](ctx)
    assert len(rows) == 1
    assert rows[0].rule == "shot_block"
    assert rows[0].signed_value == pytest.approx(0.25)  # +xG to the blocker


def test_shot_rules_are_mutually_exclusive(fitted_xt):
    # a blocked shot fires ONLY shot_block, not the two pressure rules (blocked precedence, on_target moot)
    ctx = _shot_ctx(result_name="fail", shot_blocked=True, xg=0.25, on_target=True, fitted_xt=fitted_xt)
    assert RULE_REGISTRY["shot_block"](ctx)  # fires
    assert RULE_REGISTRY["pressure_on_missed_shot"](ctx) == []  # blocked precedence
    assert RULE_REGISTRY["failed_pressure_shot_on_target"](ctx) == []


def test_shot_rule_no_defender_no_row(fitted_xt):
    ctx = _shot_ctx(result_name="fail", shot_blocked=False, xg=0.2, defender_x=80.0, on_target=False, fitted_xt=fitted_xt)
    assert RULE_REGISTRY["pressure_on_missed_shot"](ctx) == []  # defender too far


def test_shot_rule_nan_xg_fires_but_unsizable(fitted_xt):
    ctx = _shot_ctx(result_name="fail", shot_blocked=False, xg=np.nan, on_target=False, fitted_xt=fitted_xt)
    rows = RULE_REGISTRY["pressure_on_missed_shot"](ctx)
    assert len(rows) == 1 and np.isnan(rows[0].signed_value)  # fired-but-unsizable
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/tracking/test_defensive_credit_rules.py -q` — Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement `_rules.py` — context, CreditRow, shot-outcome helpers, and the three shot rules**

```python
# silly_kicks/tracking/defensive_credit/_rules.py
"""RuleContext + one pure function per rule + RULE_REGISTRY."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from silly_kicks.spadl import config as spadlconfig

from ._chaining import recovery_after_pass, resulting_shot_in_possession, with_possessions
from ._params import (
    RULE_FAILED_PRESSURE_SHOT_ON_TARGET,
    RULE_PRESSURE_ON_MISSED_SHOT,
    RULE_SHOT_BLOCK,
    SIZING_XG,
    SIZING_XT,
    DefensiveCreditParams,
)
from ._resolution import resolve_responsible_defenders
from ._sizing import extinguished_xt, xg_of_shot

_SHOT_TYPE = spadlconfig.actiontype_id["shot"]
_GOAL_RESULT = spadlconfig.result_id["success"]  # a scored shot is on-target by construction


@dataclass
class CreditRow:
    game_id: object
    action_id: object
    player_id: object
    team_id: object
    rule: str
    signed_value: float
    anchor_type: str
    frame_id: object
    sizing: str


@dataclass
class RuleContext:
    actions: pd.DataFrame
    frames: pd.DataFrame
    idx: int
    xg_column: str
    xt: object
    blocked_column: str
    params: DefensiveCreditParams
    frame_id: object  # resolved linked frame for this action (Int64/NaN)
    acting_team_id: object

    @property
    def action(self) -> pd.Series:
        return self.actions.iloc[self.idx]

    def defenders(self, *, anchor_x, anchor_y, mode):
        return resolve_responsible_defenders(
            self.actions, self.frames, anchor_x=anchor_x, anchor_y=anchor_y,
            acting_team_id=self.acting_team_id, mode=mode, params=self.params,
            frame_id=self.frame_id,
        )

    @classmethod
    def build_single(cls, actions, frames, *, idx, xg_column, xt, blocked_column, params):
        """Convenience builder for unit tests: single action, single frame."""
        act = with_possessions(actions)
        fid = int(frames["frame_id"].iloc[0]) if "frame_id" in frames.columns and len(frames) else None
        return cls(
            actions=act, frames=frames, idx=idx, xg_column=xg_column, xt=xt,
            blocked_column=blocked_column, params=params, frame_id=fid,
            acting_team_id=act.iloc[idx]["team_id"],
        )


def _is_blocked(ctx: RuleContext) -> bool:
    val = ctx.action.get(ctx.blocked_column, pd.NA)
    return val is True or val == True  # noqa: E712 — nullable-boolean True only (NA/False -> False)


def _on_target_state(ctx: RuleContext):
    """Tri-state on-target: True / False / None (unknown).

    A goal (result success) is on-target. Otherwise read the precomputed nullable-boolean
    ``_on_target`` column the ORCHESTRATOR attaches (Task 8): provider outcome -> TF-48
    ``shot_on_target_derived`` fallback. Unknown (NA) -> None: the pressure rules DO NOT fire
    (we never fabricate a sign; a saved shot must not be mistaken for a miss -- P-1).
    """
    a = ctx.action
    if a["result_id"] == _GOAL_RESULT:
        return True
    val = a.get("_on_target", pd.NA)
    if pd.isna(val):
        return None
    return bool(val)


def _shot_credit(ctx: RuleContext, *, rule: str, sign: float, mode: str) -> list[CreditRow]:
    a = ctx.action
    defs = ctx.defenders(anchor_x=a["start_x"], anchor_y=a["start_y"], mode=mode)
    if defs.empty:
        return []
    xg = xg_of_shot(a, xg_column=ctx.xg_column)
    rows = []
    for _, d in defs.iterrows():
        rows.append(CreditRow(
            game_id=a["game_id"], action_id=a["action_id"], player_id=d["player_id"],
            team_id=d["team_id"], rule=rule, signed_value=sign * xg,
            anchor_type="shot", frame_id=ctx.frame_id, sizing=SIZING_XG,
        ))
    return rows


def rule_pressure_on_missed_shot(ctx: RuleContext) -> list[CreditRow]:
    if _is_blocked(ctx):
        return []
    if _on_target_state(ctx) is not False:  # fires ONLY when definitively OFF-target
        return []
    return _shot_credit(ctx, rule=RULE_PRESSURE_ON_MISSED_SHOT, sign=+1.0, mode="nearest")


def rule_failed_pressure_shot_on_target(ctx: RuleContext) -> list[CreditRow]:
    if _is_blocked(ctx):
        return []
    if _on_target_state(ctx) is not True:  # fires ONLY when definitively ON-target
        return []
    return _shot_credit(ctx, rule=RULE_FAILED_PRESSURE_SHOT_ON_TARGET, sign=-1.0, mode="nearest")


def rule_shot_block(ctx: RuleContext) -> list[CreditRow]:
    if not _is_blocked(ctx):
        return []
    return _shot_credit(ctx, rule=RULE_SHOT_BLOCK, sign=+1.0, mode="nearest")


RULE_REGISTRY: dict[str, Callable[[RuleContext], list[CreditRow]]] = {
    RULE_PRESSURE_ON_MISSED_SHOT: rule_pressure_on_missed_shot,
    RULE_FAILED_PRESSURE_SHOT_ON_TARGET: rule_failed_pressure_shot_on_target,
    RULE_SHOT_BLOCK: rule_shot_block,
}
```

> Note: `_shot_credit` calls `xg_of_shot` which returns NaN when the injected xG is NaN → `sign * NaN = NaN` → a fired-but-unsizable row (satisfies `test_shot_rule_nan_xg_fires_but_unsizable`). The rule is only invoked on shot rows by the orchestrator; the unit test builds a shot context directly.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/tracking/test_defensive_credit_rules.py -q` — Expected: PASS (6).

- [ ] **Step 5: Lint + commit**

```bash
git add silly_kicks/tracking/defensive_credit/_rules.py tests/tracking/test_defensive_credit_rules.py
git commit -m "feat(tracking): TF-51 rule engine + three shot rules (mutually-exclusive partition)"
```

---

### Task 6: The four turnover rules (`_rules.py` extension)

**Rules (spec §5, all sized `xT(origin)`):** `pressure_pass_fail` (+presser / −passer at the passer origin), `recovery_double_credit` (+recoverer at recovery location / −passer at passer origin), `synchronized_final_third_pressure` (+ each within-threshold defender beyond the nearest, when the failed pass is in the carrier's own defensive third), `forced_bad_touch` (+presser at the bad-touch location).

- [ ] **Step 1: Add the failing turnover-rule tests** (append to `tests/tracking/test_defensive_credit_rules.py`)

```python
def _pass_ctx(*, result_name="fail", start_x=40.0, start_y=34.0, defender_x=41.0,
              team_id=10, player_id=5, fitted_xt=None, action_id=1):
    from silly_kicks.tracking.defensive_credit._params import DefensiveCreditParams
    actions = one_action(type_name="pass", result_name=result_name, start_x=start_x, start_y=start_y,
                         end_x=55.0, end_y=34.0, team_id=team_id, player_id=player_id, action_id=action_id)
    frames = frame_with_defender(defender_x=defender_x, defender_y=start_y)
    return RuleContext.build_single(actions, frames, idx=0, xg_column="xg", xt=fitted_xt,
                                    blocked_column="shot_blocked", params=DefensiveCreditParams())


def test_pressure_pass_fail_emits_plus_presser_minus_passer(fitted_xt):
    ctx = _pass_ctx(start_x=40.0, defender_x=41.0, fitted_xt=fitted_xt)
    rows = RULE_REGISTRY["pressure_pass_fail"](ctx)
    assert len(rows) == 2
    plus = [r for r in rows if r.signed_value > 0][0]
    minus = [r for r in rows if r.signed_value < 0][0]
    assert plus.player_id == 900 and plus.team_id == 20  # presser (defender)
    assert minus.player_id == 5 and minus.team_id == 10   # passer (acting team)
    assert plus.signed_value == pytest.approx(-minus.signed_value)  # same origin -> equal magnitude
    assert plus.sizing == "xt"


def test_pressure_pass_fail_no_defender_no_rows(fitted_xt):
    ctx = _pass_ctx(start_x=40.0, defender_x=60.0, fitted_xt=fitted_xt)  # far
    assert RULE_REGISTRY["pressure_pass_fail"](ctx) == []


def test_forced_bad_touch_credits_presser(fitted_xt):
    from silly_kicks.tracking.defensive_credit._params import DefensiveCreditParams
    actions = one_action(type_name="bad_touch", result_name="fail", start_x=45.0, start_y=34.0,
                         team_id=10, player_id=5)
    frames = frame_with_defender(defender_x=46.0, defender_y=34.0)
    ctx = RuleContext.build_single(actions, frames, idx=0, xg_column="xg", xt=fitted_xt,
                                   blocked_column="shot_blocked", params=DefensiveCreditParams())
    rows = RULE_REGISTRY["forced_bad_touch"](ctx)
    assert len(rows) == 1 and rows[0].signed_value > 0 and rows[0].sizing == "xt"


def test_synchronized_fires_only_in_own_defensive_third(fitted_xt):
    from silly_kicks.tracking.defensive_credit._params import DefensiveCreditParams
    # carrier's own defensive third = action-LTR x <= 35 (pressing team's high press).
    # place TWO defenders within threshold; synchronized credits the one BEYOND nearest.
    actions = one_action(type_name="pass", result_name="fail", start_x=20.0, start_y=34.0, team_id=10, player_id=5)
    frames = frame_with_defender(defender_x=21.0, defender_y=34.0)  # 1 m -> 900
    extra = frames.iloc[[0]].copy(); extra["player_id"] = 901; extra["x"] = 22.0
    frames = pd.concat([frames, extra], ignore_index=True)
    ctx = RuleContext.build_single(actions, frames, idx=0, xg_column="xg", xt=fitted_xt,
                                   blocked_column="shot_blocked", params=DefensiveCreditParams())
    rows = RULE_REGISTRY["synchronized_final_third_pressure"](ctx)
    assert {r.player_id for r in rows} == {901}  # beyond-nearest


def test_synchronized_silent_outside_defensive_third(fitted_xt):
    from silly_kicks.tracking.defensive_credit._params import DefensiveCreditParams
    actions = one_action(type_name="pass", result_name="fail", start_x=70.0, start_y=34.0, team_id=10, player_id=5)
    frames = frame_with_defender(defender_x=71.0, defender_y=34.0)
    extra = frames.iloc[[0]].copy(); extra["player_id"] = 901; extra["x"] = 72.0
    frames = pd.concat([frames, extra], ignore_index=True)
    ctx = RuleContext.build_single(actions, frames, idx=0, xg_column="xg", xt=fitted_xt,
                                   blocked_column="shot_blocked", params=DefensiveCreditParams())
    assert RULE_REGISTRY["synchronized_final_third_pressure"](ctx) == []
```

`recovery_double_credit` needs a 2-action stream (a fail pass then a recovery); its test lives in the orchestration test (Task 8) where the full stream + chaining is exercised. The rule function is still implemented here.

- [ ] **Step 2: Run to verify the new tests fail**

Run: `python -m pytest tests/tracking/test_defensive_credit_rules.py -q -k "turnover or pressure_pass or forced_bad or synchronized"`
Expected: FAIL — `KeyError: 'pressure_pass_fail'` (registry lacks the rules).

- [ ] **Step 3: Implement the four turnover rules (append to `_rules.py`, then extend `RULE_REGISTRY`)**

```python
from ._params import (  # extend the existing import block
    RULE_FORCED_BAD_TOUCH,
    RULE_PRESSURE_PASS_FAIL,
    RULE_RECOVERY_DOUBLE_CREDIT,
    RULE_SYNCHRONIZED_FINAL_THIRD_PRESSURE,
)


def _xt_at(ctx: RuleContext, x: float, y: float) -> float:
    return float(extinguished_xt([(x, y)], ctx.xt)[0])


def rule_pressure_pass_fail(ctx: RuleContext) -> list[CreditRow]:
    a = ctx.action
    defs = ctx.defenders(anchor_x=a["start_x"], anchor_y=a["start_y"], mode="nearest")
    if defs.empty:
        return []
    val = _xt_at(ctx, a["start_x"], a["start_y"])  # xT(origin) — same point for both rows
    d = defs.iloc[0]
    return [
        CreditRow(a["game_id"], a["action_id"], d["player_id"], d["team_id"],
                  RULE_PRESSURE_PASS_FAIL, +val, "pass", ctx.frame_id, SIZING_XT),
        CreditRow(a["game_id"], a["action_id"], a["player_id"], a["team_id"],
                  RULE_PRESSURE_PASS_FAIL, -val, "pass", ctx.frame_id, SIZING_XT),
    ]


def rule_forced_bad_touch(ctx: RuleContext) -> list[CreditRow]:
    a = ctx.action
    defs = ctx.defenders(anchor_x=a["start_x"], anchor_y=a["start_y"], mode="nearest")
    if defs.empty:
        return []
    val = _xt_at(ctx, a["start_x"], a["start_y"])
    d = defs.iloc[0]
    return [CreditRow(a["game_id"], a["action_id"], d["player_id"], d["team_id"],
                      RULE_FORCED_BAD_TOUCH, +val, "bad_touch", ctx.frame_id, SIZING_XT)]


def rule_synchronized_final_third_pressure(ctx: RuleContext) -> list[CreditRow]:
    a = ctx.action
    if a["start_x"] > ctx.params.synchronized_zone_boundary_x:  # not in carrier's own defensive third
        return []
    defs = ctx.defenders(anchor_x=a["start_x"], anchor_y=a["start_y"], mode="all_within_beyond_nearest")
    if defs.empty:
        return []
    val = _xt_at(ctx, a["start_x"], a["start_y"])
    return [CreditRow(a["game_id"], a["action_id"], d["player_id"], d["team_id"],
                      RULE_SYNCHRONIZED_FINAL_THIRD_PRESSURE, +val, "pass", ctx.frame_id, SIZING_XT)
            for _, d in defs.iterrows()]


def rule_recovery_double_credit(ctx: RuleContext) -> list[CreditRow]:
    a = ctx.action
    rec = recovery_after_pass(ctx.actions, ctx.idx, max_actions=ctx.params.recovery_max_actions)  # single resolver (P-3)
    if rec is None:
        return []
    passer_val = _xt_at(ctx, a["start_x"], a["start_y"])                 # -passer at the passer origin
    rec_val = _xt_at(ctx, float(rec["start_x"]), float(rec["start_y"]))  # +recoverer at the recovery location
    return [
        CreditRow(a["game_id"], a["action_id"], rec["player_id"], rec["team_id"],
                  RULE_RECOVERY_DOUBLE_CREDIT, +rec_val, "pass", ctx.frame_id, SIZING_XT),
        CreditRow(a["game_id"], a["action_id"], a["player_id"], a["team_id"],
                  RULE_RECOVERY_DOUBLE_CREDIT, -passer_val, "pass", ctx.frame_id, SIZING_XT),
    ]


RULE_REGISTRY.update({
    RULE_PRESSURE_PASS_FAIL: rule_pressure_pass_fail,
    RULE_RECOVERY_DOUBLE_CREDIT: rule_recovery_double_credit,
    RULE_SYNCHRONIZED_FINAL_THIRD_PRESSURE: rule_synchronized_final_third_pressure,
    RULE_FORCED_BAD_TOUCH: rule_forced_bad_touch,
})
```

> The recovery logic lives once in `_chaining.recovery_after_pass` (ADR-019-safe via `same_id`, from Task 4). `_rules` calls it directly — no duplicate resolver, no raw `!=`, no `if False` branch (P-3).

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/tracking/test_defensive_credit_rules.py -q` — Expected: PASS (all shot + turnover tests).

- [ ] **Step 5: Lint + commit**

```bash
git add silly_kicks/tracking/defensive_credit/_rules.py tests/tracking/test_defensive_credit_rules.py
git commit -m "feat(tracking): TF-51 four turnover rules (xT(origin)-sized)"
```

---

### Task 7: The three chained rules (`_rules.py` extension)

**Rules (spec §5, all sized resulting-shot xG):** `failed_cross_block` (cross → receipt → shot: −nearest def at receipt / +shot-blocker), `failed_marking_through_ball` (high-ΔxT completed pass → shot: −responsible def at pass moment), `beaten_1v1` (successful take-on → quality shot: −beaten def).

- [ ] **Step 1: Add the failing chained-rule tests** (these need multi-action streams; add to `test_defensive_credit_rules.py`)

```python
def _stream_ctx(rows, *, idx, fitted_xt, frames):
    from silly_kicks.spadl import config as spadlconfig
    from silly_kicks.tracking.defensive_credit._params import DefensiveCreditParams
    base = dict(game_id="g1", period_id=1, bodypart_id=spadlconfig.bodypart_id["foot"])
    recs = []
    for i, r in enumerate(rows):
        d = dict(base); d.update(action_id=i, time_seconds=float(i), **r)
        d["type_id"] = spadlconfig.actiontype_id[d.pop("type_name")]
        d["result_id"] = spadlconfig.result_id[d.pop("result_name")]
        d.setdefault("shot_blocked", pd.NA); d.setdefault("xg", np.nan)
        recs.append(d)
    actions = pd.DataFrame(recs)
    actions["shot_blocked"] = pd.array(actions["shot_blocked"].tolist(), dtype="boolean")
    return RuleContext.build_single(actions, frames, idx=idx, xg_column="xg", xt=fitted_xt,
                                    blocked_column="shot_blocked", params=DefensiveCreditParams())


def test_beaten_1v1_debits_defender_on_quality_resulting_shot(fitted_xt):
    frames = frame_with_defender(defender_x=51.0, defender_y=34.0, action_time=0.0)
    ctx = _stream_ctx([
        dict(type_name="take_on", result_name="success", team_id=10, player_id=5,
             start_x=50.0, start_y=34.0, end_x=55.0, end_y=34.0),
        dict(type_name="shot", result_name="fail", team_id=10, player_id=6,
             start_x=95.0, start_y=34.0, end_x=105.0, end_y=34.0, xg=0.2),
    ], idx=0, fitted_xt=fitted_xt, frames=frames)
    rows = RULE_REGISTRY["beaten_1v1"](ctx)
    assert len(rows) == 1 and rows[0].signed_value == pytest.approx(-0.2) and rows[0].team_id == 20


def test_beaten_1v1_no_quality_shot_no_row(fitted_xt):
    frames = frame_with_defender(defender_x=51.0, defender_y=34.0, action_time=0.0)
    ctx = _stream_ctx([
        dict(type_name="take_on", result_name="success", team_id=10, player_id=5,
             start_x=50.0, start_y=34.0, end_x=55.0, end_y=34.0),
        dict(type_name="shot", result_name="fail", team_id=10, player_id=6,
             start_x=95.0, start_y=34.0, end_x=105.0, end_y=34.0, xg=0.01),  # below 0.05 floor
    ], idx=0, fitted_xt=fitted_xt, frames=frames)
    assert RULE_REGISTRY["beaten_1v1"](ctx) == []


def test_failed_cross_block_pair(fitted_xt):
    frames = frame_with_defender(defender_x=101.0, defender_y=40.0, action_time=0.0)
    ctx = _stream_ctx([
        dict(type_name="cross", result_name="success", team_id=10, player_id=5,
             start_x=90.0, start_y=5.0, end_x=100.0, end_y=40.0),
        dict(type_name="shot", result_name="fail", team_id=10, player_id=6,
             start_x=100.0, start_y=40.0, end_x=105.0, end_y=34.0, xg=0.3, shot_blocked=True),
    ], idx=0, fitted_xt=fitted_xt, frames=frames)
    rows = RULE_REGISTRY["failed_cross_block"](ctx)
    signs = sorted(r.signed_value for r in rows)
    assert signs == pytest.approx([-0.3, 0.3])  # -def at receipt, +blocker


def test_failed_marking_through_ball(fitted_xt):
    frames = frame_with_defender(defender_x=61.0, defender_y=34.0, action_time=0.0)
    ctx = _stream_ctx([
        dict(type_name="pass", result_name="success", team_id=10, player_id=5,
             start_x=60.0, start_y=34.0, end_x=95.0, end_y=34.0),  # big forward ΔxT
        dict(type_name="shot", result_name="fail", team_id=10, player_id=6,
             start_x=95.0, start_y=34.0, end_x=105.0, end_y=34.0, xg=0.2),
    ], idx=0, fitted_xt=fitted_xt, frames=frames)
    rows = RULE_REGISTRY["failed_marking_through_ball"](ctx)
    assert len(rows) == 1 and rows[0].signed_value == pytest.approx(-0.2) and rows[0].team_id == 20
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/tracking/test_defensive_credit_rules.py -q -k "beaten or cross_block or through_ball"`
Expected: FAIL — `KeyError` (rules not registered).

- [ ] **Step 3: Implement the three chained rules (append to `_rules.py`)**

```python
from ._params import (  # extend imports
    RULE_BEATEN_1V1,
    RULE_FAILED_CROSS_BLOCK,
    RULE_FAILED_MARKING_THROUGH_BALL,
)

_TAKE_ON = spadlconfig.actiontype_id["take_on"]
_CROSS = spadlconfig.actiontype_id["cross"]
_PASS = spadlconfig.actiontype_id["pass"]
_SUCCESS = spadlconfig.result_id["success"]


def _resulting_shot(ctx: RuleContext):
    return resulting_shot_in_possession(
        ctx.actions, ctx.idx, attacking_team_id=ctx.action["team_id"],
        max_actions=ctx.params.resulting_shot_max_actions,
    )


def rule_beaten_1v1(ctx: RuleContext) -> list[CreditRow]:
    a = ctx.action
    if a["type_id"] != _TAKE_ON or a["result_id"] != _SUCCESS:
        return []
    shot = _resulting_shot(ctx)
    if shot is None:
        return []
    xg = xg_of_shot(shot, xg_column=ctx.xg_column)
    if not (xg >= ctx.params.beaten_1v1_min_shot_xg):  # NaN-safe: NaN fails the floor -> no row
        return []
    # beaten defender = nearest opponent to the take-on start within the OUTSIDE-box radius
    defs = ctx.defenders(anchor_x=a["start_x"], anchor_y=a["start_y"], mode="nearest")
    if defs.empty:
        return []
    d = defs.iloc[0]
    return [CreditRow(a["game_id"], a["action_id"], d["player_id"], d["team_id"],
                      RULE_BEATEN_1V1, -xg, "take_on", ctx.frame_id, SIZING_XG)]


def rule_failed_cross_block(ctx: RuleContext) -> list[CreditRow]:
    a = ctx.action
    if a["type_id"] != _CROSS or a["result_id"] != _SUCCESS:
        return []
    shot = _resulting_shot(ctx)
    if shot is None:
        return []
    xg = xg_of_shot(shot, xg_column=ctx.xg_column)
    rows: list[CreditRow] = []
    # -def at the receipt point (cross end)
    defs = ctx.defenders(anchor_x=a["end_x"], anchor_y=a["end_y"], mode="nearest")
    if not defs.empty:
        d = defs.iloc[0]
        rows.append(CreditRow(a["game_id"], a["action_id"], d["player_id"], d["team_id"],
                              RULE_FAILED_CROSS_BLOCK, -xg, "cross", ctx.frame_id, SIZING_XG))
    # +blocker if the resulting shot was blocked (nearest opp to the shot origin)
    blocked_val = shot.get(ctx.blocked_column, pd.NA)
    if blocked_val is True or blocked_val == True:  # noqa: E712
        bdefs = ctx.defenders(anchor_x=shot["start_x"], anchor_y=shot["start_y"], mode="nearest")
        if not bdefs.empty:
            b = bdefs.iloc[0]
            rows.append(CreditRow(a["game_id"], a["action_id"], b["player_id"], b["team_id"],
                                  RULE_FAILED_CROSS_BLOCK, +xg, "cross", ctx.frame_id, SIZING_XG))
    return rows


def rule_failed_marking_through_ball(ctx: RuleContext) -> list[CreditRow]:
    a = ctx.action
    if a["type_id"] != _PASS or a["result_id"] != _SUCCESS:
        return []
    dxt = _xt_at(ctx, a["end_x"], a["end_y"]) - _xt_at(ctx, a["start_x"], a["start_y"])
    if not (dxt >= ctx.params.through_ball_delta_xt_min):  # NaN-safe floor
        return []
    shot = _resulting_shot(ctx)
    if shot is None:
        return []
    xg = xg_of_shot(shot, xg_column=ctx.xg_column)
    # responsible def = nearest opponent to the pass origin at the pass moment
    defs = ctx.defenders(anchor_x=a["start_x"], anchor_y=a["start_y"], mode="nearest")
    if defs.empty:
        return []
    d = defs.iloc[0]
    return [CreditRow(a["game_id"], a["action_id"], d["player_id"], d["team_id"],
                      RULE_FAILED_MARKING_THROUGH_BALL, -xg, "pass", ctx.frame_id, SIZING_XG)]


RULE_REGISTRY.update({
    RULE_FAILED_CROSS_BLOCK: rule_failed_cross_block,
    RULE_FAILED_MARKING_THROUGH_BALL: rule_failed_marking_through_ball,
    RULE_BEATEN_1V1: rule_beaten_1v1,
})
```

- [ ] **Step 4: Run to verify all rule tests pass**

Run: `python -m pytest tests/tracking/test_defensive_credit_rules.py -q` — Expected: PASS.

Also assert the registry is complete:

```python
def test_registry_covers_every_rule():
    from silly_kicks.tracking.defensive_credit._params import DEFENSIVE_CREDIT_RULES
    from silly_kicks.tracking.defensive_credit._rules import RULE_REGISTRY
    assert set(RULE_REGISTRY) == set(DEFENSIVE_CREDIT_RULES)
```

- [ ] **Step 5: Lint + commit**

```bash
git add silly_kicks/tracking/defensive_credit/_rules.py tests/tracking/test_defensive_credit_rules.py
git commit -m "feat(tracking): TF-51 three chained rules + registry completeness"
```

---

### Task 8: Orchestration — `compute_defensive_credits` (long-form) + closed-vocab + mirror-invariance + sizing-regression + rules-gating

**Files:**
- Create: `silly_kicks/tracking/defensive_credit/_orchestration.py`
- Test: `tests/tracking/test_defensive_credit_orchestration.py`

**What it does (spec §6, §9.1, §12):** one pass over `actions`: ONE `link_actions_to_frames` (or caller `links`), ONE `acting_team_attacks_rtl`, resolve `frame_id` per action positionally, then for each action build a `RuleContext` and run every enabled rule; collect `CreditRow`s into the long-form DataFrame with the exact schema + dtypes. `physical_grid` is built implicitly once inside `values_at_points` per call — acceptable for v1 (the perf budget pins the LINK to one call; if profiling shows repeated grid builds, cache the grid in the context, a noted refinement).

- [ ] **Step 1: Write the failing orchestration tests**

```python
# tests/tracking/test_defensive_credit_orchestration.py
import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking.defensive_credit import (
    DEFENSIVE_CREDIT_RULES,
    DefensiveCreditParams,
    compute_defensive_credits,
)
from tests.tracking._defensive_credit_fixtures import frame_with_defender, one_action

_LONG_COLS = ["game_id", "period_id", "action_id", "player_id", "team_id", "rule",
              "signed_value", "anchor_type", "frame_id", "sizing"]


def _shot_scene(fitted_xt):
    actions = one_action(type_name="shot", result_name="fail", start_x=95.0, start_y=34.0)
    actions["shot_blocked"] = pd.array([False], dtype="boolean")
    actions["cross_blocked"] = pd.array([pd.NA], dtype="boolean")
    # deterministic OFF-target so the orchestrator does not need the TF-48 frame fallback on
    # synthetic frames; the saved-shot test overrides this to True. (on_target_column default.)
    actions["shot_on_target_derived"] = pd.array([False], dtype="boolean")
    actions["xg"] = [0.2]
    frames = frame_with_defender(defender_x=96.0, defender_y=34.0)
    return actions, frames


def test_long_form_schema_and_values(fitted_xt):
    actions, frames = _shot_scene(fitted_xt)
    out = compute_defensive_credits(actions, frames, xg_column="xg", xt=fitted_xt)
    assert list(out.columns) == _LONG_COLS
    assert (out["rule"] == "pressure_on_missed_shot").all()
    assert out["signed_value"].iloc[0] == pytest.approx(0.2)
    assert set(out["sizing"]) <= {"xg", "xt"}


def test_pressured_saved_shot_is_negative_end_to_end(fitted_xt):
    # THE P-1 regression at the orchestrator level: a SAVED shot (result fail, on-target) must yield
    # failed_pressure_shot_on_target (NEGATIVE), never pressure_on_missed_shot (+credit).
    actions, frames = _shot_scene(fitted_xt)
    actions["shot_on_target_derived"] = pd.array([True], dtype="boolean")  # a save = on-target
    out = compute_defensive_credits(actions, frames, xg_column="xg", xt=fitted_xt)
    assert (out["rule"] == "failed_pressure_shot_on_target").all()
    assert out["signed_value"].iloc[0] == pytest.approx(-0.2)


def test_closed_vocabulary(fitted_xt):
    actions, frames = _shot_scene(fitted_xt)
    out = compute_defensive_credits(actions, frames, xg_column="xg", xt=fitted_xt)
    assert set(out["rule"]) <= set(DEFENSIVE_CREDIT_RULES)


def test_rules_gating_disables_a_rule(fitted_xt):
    actions, frames = _shot_scene(fitted_xt)
    params = DefensiveCreditParams(rules=frozenset(set(DEFENSIVE_CREDIT_RULES) - {"pressure_on_missed_shot"}))
    out = compute_defensive_credits(actions, frames, xg_column="xg", xt=fitted_xt, params=params)
    assert "pressure_on_missed_shot" not in set(out["rule"])
    # re-enable -> the row reappears
    out2 = compute_defensive_credits(actions, frames, xg_column="xg", xt=fitted_xt)
    assert "pressure_on_missed_shot" in set(out2["rule"])


def test_fired_but_unsizable_is_nan_row(fitted_xt):
    actions, frames = _shot_scene(fitted_xt)
    actions["xg"] = [np.nan]
    out = compute_defensive_credits(actions, frames, xg_column="xg", xt=fitted_xt)
    assert len(out) == 1 and np.isnan(out["signed_value"].iloc[0])


def test_no_defender_no_row(fitted_xt):
    actions, frames = _shot_scene(fitted_xt)
    frames.loc[frames["player_id"] == 900, "x"] = 80.0  # move defender far
    out = compute_defensive_credits(actions, frames, xg_column="xg", xt=fitted_xt)
    assert out.empty


def test_xg_column_missing_fails_loud(fitted_xt):
    actions, frames = _shot_scene(fitted_xt)
    actions = actions.drop(columns=["xg"])
    with pytest.raises(ValueError, match="xg_column"):
        compute_defensive_credits(actions, frames, xg_column="xg", xt=fitted_xt)


def test_mirror_invariance_home_vs_away(fitted_xt):
    """Same physical situation as a home action and an away action -> identical action-LTR credit
    (ADR-028). Asymmetric + extreme fixture so a y-symmetric one can't pass vacuously."""
    # HOME action: acting team 10 == home, attacks x=105. Defender at frame (96, 20).
    home_actions = one_action(type_name="shot", result_name="fail", start_x=95.0, start_y=20.0, team_id=10)
    home_actions["shot_blocked"] = pd.array([False], dtype="boolean"); home_actions["xg"] = [0.2]
    home_frames = frame_with_defender(defender_x=96.0, defender_y=20.0, acting_team_id=10, home_team_id=10)
    home = compute_defensive_credits(home_actions, home_frames, xg_column="xg", xt=fitted_xt)

    # AWAY action: acting team 20 != home(10). In action-LTR the same situation: anchor (95,20),
    # defender 1 m away. In FRAME coords (home attacks x=105), the away action is point-reflected:
    # action-LTR (95,20) -> frame (10, 48); defender frame (9, 48).
    away_actions = one_action(type_name="shot", result_name="fail", start_x=95.0, start_y=20.0, team_id=20)
    away_actions["shot_blocked"] = pd.array([False], dtype="boolean"); away_actions["xg"] = [0.2]
    away_frames = frame_with_defender(defender_x=9.0, defender_y=48.0, acting_team_id=20, home_team_id=10,
                                      defender_team_id=10)
    away = compute_defensive_credits(away_actions, away_frames, xg_column="xg", xt=fitted_xt)

    assert not home.empty and not away.empty
    assert home["signed_value"].iloc[0] == pytest.approx(away["signed_value"].iloc[0])
    assert home["rule"].iloc[0] == away["rule"].iloc[0]


def test_sizing_regression_dangerous_turnover_scores_higher(fitted_xt):
    """A turnover forced near the defending goal (high xT(origin)) >> the same turnover deep in own half."""
    def _pass_scene(sx):
        a = one_action(type_name="pass", result_name="fail", start_x=sx, start_y=34.0, team_id=10, player_id=5)
        a["shot_blocked"] = pd.array([pd.NA], dtype="boolean"); a["xg"] = [np.nan]
        f = frame_with_defender(defender_x=sx + 1.0, defender_y=34.0)
        return compute_defensive_credits(a, f, xg_column="xg", xt=fitted_xt,
                                         params=DefensiveCreditParams(rules=frozenset({"pressure_pass_fail"})))
    near = _pass_scene(95.0)   # near attacked goal -> high xT(origin)
    deep = _pass_scene(10.0)   # own half -> low xT(origin)
    near_plus = near[near["signed_value"] > 0]["signed_value"].iloc[0]
    deep_plus = deep[deep["signed_value"] > 0]["signed_value"].iloc[0]
    assert near_plus > deep_plus
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/tracking/test_defensive_credit_orchestration.py -q`
Expected: FAIL — `ImportError: cannot import name 'compute_defensive_credits'`.

- [ ] **Step 3: Implement `_orchestration.py`**

```python
# silly_kicks/tracking/defensive_credit/_orchestration.py
"""Batch orchestration: actions -> long-form defensive-credit rows."""

from __future__ import annotations

import pandas as pd

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking.utils import link_actions_to_frames

from ._chaining import with_possessions
from ._params import DEFENSIVE_CREDIT_RULES, DefensiveCreditParams
from ._rules import RULE_REGISTRY, CreditRow, RuleContext

_LONG_COLS = ["game_id", "period_id", "action_id", "player_id", "team_id", "rule",
              "signed_value", "anchor_type", "frame_id", "sizing"]


def _ensure_on_target(act: pd.DataFrame, frames: pd.DataFrame, pointers: pd.DataFrame, on_target_column: str) -> pd.DataFrame:
    """Attach a nullable-boolean ``_on_target`` per action (P-1). Shots only; others NA.

    Goal (result success) -> True. Else the injected ``on_target_column`` if present, else the
    frame-based TF-48 ``shot_on_target_derived`` fallback (reuses the ONE link via ``links=pointers``,
    so the perf budget's single-link contract holds). ``pd.NA`` stays NA (unknown) -> the pressure
    rules do not fire, so a saved shot is never mis-signed as a miss.
    """
    act = act.copy()
    if on_target_column in act.columns:
        base = pd.array(act[on_target_column], dtype="boolean")
    else:
        from silly_kicks.tracking.features import add_shot_goalmouth

        gm = add_shot_goalmouth(act, frames, links=pointers)
        col = gm["shot_on_target_derived"] if "shot_on_target_derived" in gm.columns else pd.Series(pd.NA, index=act.index)
        base = pd.array(col, dtype="boolean")
    is_shot = (act["type_id"] == _SHOT_TYPE).to_numpy()
    is_goal = ((act["type_id"] == _SHOT_TYPE) & (act["result_id"] == _GOAL_RESULT)).to_numpy()
    base[~is_shot] = pd.NA  # only meaningful for shots
    base[is_goal] = True  # a scored shot is on-target
    act["_on_target"] = base
    return act


_SHOT_TYPE = spadlconfig.actiontype_id["shot"]
_GOAL_RESULT = spadlconfig.result_id["success"]


def compute_defensive_credits(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    xg_column: str,
    xt,
    blocked_column: str = "shot_blocked",
    on_target_column: str = "shot_on_target_derived",
    links: pd.DataFrame | None = None,
    params: DefensiveCreditParams | None = None,
) -> pd.DataFrame:
    """Long-form: one row per (triggering action, credited player, rule). Pure."""
    params = params or DefensiveCreditParams()
    act = with_possessions(actions).reset_index(drop=True)

    pointers = links if links is not None else link_actions_to_frames(act, frames)[0]
    act = _ensure_on_target(act, frames, pointers, on_target_column)  # P-1: attach _on_target (tri-state)
    fid_by_pos = (
        pointers.drop_duplicates("action_id").set_index("action_id")["frame_id"]
        .reindex(act["action_id"].to_numpy()).to_numpy()
    )

    enabled = [r for r in DEFENSIVE_CREDIT_RULES if r in params.rules]
    rows: list[CreditRow] = []
    for idx in range(len(act)):
        fid = fid_by_pos[idx]
        fid = None if pd.isna(fid) else int(fid)
        ctx = RuleContext(
            actions=act, frames=frames, idx=idx, xg_column=xg_column, xt=xt,
            blocked_column=blocked_column, params=params, frame_id=fid,
            acting_team_id=act.iloc[idx]["team_id"],
        )
        for rule_name in enabled:
            rows.extend(RULE_REGISTRY[rule_name](ctx))

    return _to_long_form(rows, act)


def _to_long_form(rows: list[CreditRow], act: pd.DataFrame) -> pd.DataFrame:
    if not rows:
        return _empty_long_form(act)
    df = pd.DataFrame([r.__dict__ for r in rows])  # 9 fields; period_id is NOT per-credit -> merge it
    pid = act[["action_id", "period_id"]].drop_duplicates("action_id")
    df = df.merge(pid, on="action_id", how="left")
    df = df[_LONG_COLS]  # reorder to the canonical schema (period_id now present via the merge)
    df["signed_value"] = df["signed_value"].astype("float64")
    df["frame_id"] = df["frame_id"].astype("Int64")
    df["period_id"] = df["period_id"].astype("int64")
    return df.reset_index(drop=True)


def _empty_long_form(act: pd.DataFrame) -> pd.DataFrame:
    empty = {c: pd.Series([], dtype="object") for c in _LONG_COLS}
    empty["signed_value"] = pd.Series([], dtype="float64")
    empty["frame_id"] = pd.Series([], dtype="Int64")
    empty["period_id"] = pd.Series([], dtype="int64")
    return pd.DataFrame(empty)
```

> `CreditRow` deliberately carries **no** `period_id` (it is not per-credit); `_to_long_form` merges it from `act` on `action_id`. The merge above is the code, not a footnote — it runs as written.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/tracking/test_defensive_credit_orchestration.py -q` — Expected: PASS (8).

- [ ] **Step 5: Wire `compute_defensive_credits` into the package `__init__.py`**

```python
# silly_kicks/tracking/defensive_credit/__init__.py  (extend)
from ._orchestration import compute_defensive_credits
from ._params import DEFENSIVE_CREDIT_RULES, DefensiveCreditParams

__all__ = ["DEFENSIVE_CREDIT_RULES", "DefensiveCreditParams", "compute_defensive_credits"]
```

- [ ] **Step 6: Lint + commit**

```bash
git add silly_kicks/tracking/defensive_credit/_orchestration.py silly_kicks/tracking/defensive_credit/__init__.py tests/tracking/test_defensive_credit_orchestration.py
git commit -m "feat(tracking): TF-51 compute_defensive_credits orchestration + mirror/sizing gates"
```

---

### Task 9: Per-action aggregate `add_defensive_credit` (defending-scoped, always-finite)

**Files:**
- Modify: `silly_kicks/tracking/defensive_credit/_orchestration.py` (add `aggregate_defensive_credit` body)
- Modify: `silly_kicks/tracking/features.py` (add the decorated `add_defensive_credit` wrapper)
- Test: `tests/tracking/test_defensive_credit_aggregate.py`

**Spec §9.2:** returns `actions` + `defensive_credit_net/_plus/_minus` (skipna finite sums), `n_defensive_credits` (Int64), rolled up to the triggering action, **scoped to defending-team rows only** (the `−passer` acting-team rows excluded). Always finite — a genuine 0 for no-credit actions.

- [ ] **Step 1: Write the failing aggregate test**

```python
# tests/tracking/test_defensive_credit_aggregate.py
import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import add_defensive_credit
from silly_kicks.tracking.defensive_credit._params import DefensiveCreditParams
from tests.tracking._defensive_credit_fixtures import frame_with_defender, one_action


def _pass_scene(fitted_xt):
    a = one_action(type_name="pass", result_name="fail", start_x=95.0, start_y=34.0, team_id=10, player_id=5)
    a["shot_blocked"] = pd.array([pd.NA], dtype="boolean"); a["cross_blocked"] = pd.array([pd.NA], dtype="boolean")
    a["shot_on_target_derived"] = pd.array([pd.NA], dtype="boolean")  # present -> no TF-48 fallback on synthetic frames
    a["xg"] = [np.nan]
    f = frame_with_defender(defender_x=96.0, defender_y=34.0)
    return a, f


def test_aggregate_excludes_acting_team_passer_debit(fitted_xt):
    a, f = _pass_scene(fitted_xt)
    out = add_defensive_credit(a, f, xg_column="xg", xt=fitted_xt)
    # pressure_pass_fail emits +presser(team 20) and -passer(team 10). Aggregate is defending-scoped
    # to the acting action's row: net should reflect the DEFENDING credit only, not cancel to 0.
    assert out["defensive_credit_plus"].iloc[0] > 0
    assert out["defensive_credit_minus"].iloc[0] == 0.0  # -passer excluded (acting team)
    assert out["defensive_credit_net"].iloc[0] == pytest.approx(out["defensive_credit_plus"].iloc[0])
    assert out["n_defensive_credits"].iloc[0] >= 1


def test_aggregate_always_finite_no_credit_action(fitted_xt):
    a, f = _pass_scene(fitted_xt)
    f.loc[f["player_id"] == 900, "x"] = 60.0  # defender far -> no credit
    out = add_defensive_credit(a, f, xg_column="xg", xt=fitted_xt)
    assert out["defensive_credit_net"].iloc[0] == 0.0
    assert out["n_defensive_credits"].iloc[0] == 0


def test_aggregate_is_pure(fitted_xt):
    a, f = _pass_scene(fitted_xt)
    before = a.copy()
    add_defensive_credit(a, f, xg_column="xg", xt=fitted_xt)
    pd.testing.assert_frame_equal(a, before)  # input unmutated
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/tracking/test_defensive_credit_aggregate.py -q`
Expected: FAIL — `ImportError: cannot import name 'add_defensive_credit'`.

- [ ] **Step 3: Add `aggregate_defensive_credit` to `_orchestration.py`**

```python
def aggregate_defensive_credit(
    actions, frames, *, xg_column, xt, blocked_column="shot_blocked",
    on_target_column="shot_on_target_derived", links=None, params=None,
) -> pd.DataFrame:
    """actions + per-action aggregate columns (defending-team-scoped). Pure — returns a NEW frame.

    No ``home_team_id`` (P-2): the defending/attacking split derives from ``team_id != acting-team``
    and reprojection uses ``acting_team_attacks_rtl``, so a home_team_id would be a dead required param.
    """
    long = compute_defensive_credits(
        actions, frames, xg_column=xg_column, xt=xt, blocked_column=blocked_column,
        on_target_column=on_target_column, links=links, params=params,
    )
    out = actions.copy()
    # defending-team rows only: exclude credits whose team_id == the acting action's team_id.
    # Join each credit to its action's acting team, keep credits where credited team != acting team.
    act_team = actions.set_index("action_id")["team_id"]
    if long.empty:
        defending = long
    else:
        long = long.copy()
        long["_acting_team"] = long["action_id"].map(act_team)
        # credited row is a DEFENDER iff its team_id differs from the acting team
        from silly_kicks.id_compat import ids_differ
        keep = ids_differ(long["team_id"], long["_acting_team"])
        defending = long[keep.to_numpy()]

    grp = defending.groupby("action_id")["signed_value"] if not defending.empty else None
    net = grp.sum(min_count=0) if grp is not None else pd.Series(dtype="float64")
    plus = defending[defending["signed_value"] > 0].groupby("action_id")["signed_value"].sum() if not defending.empty else pd.Series(dtype="float64")
    minus = defending[defending["signed_value"] < 0].groupby("action_id")["signed_value"].sum() if not defending.empty else pd.Series(dtype="float64")
    cnt = defending.groupby("action_id").size() if not defending.empty else pd.Series(dtype="int64")

    aid = out["action_id"]
    out["defensive_credit_net"] = aid.map(net).fillna(0.0).astype("float64")
    out["defensive_credit_plus"] = aid.map(plus).fillna(0.0).astype("float64")
    out["defensive_credit_minus"] = aid.map(minus).fillna(0.0).astype("float64")
    out["n_defensive_credits"] = aid.map(cnt).fillna(0).astype("Int64")
    return out
```

> NaN handling: a fired-but-unsizable credit has `signed_value = NaN`; `groupby.sum(min_count=0)` skips NaN by default (`skipna=True`), so the aggregate stays finite while the long-form keeps the NaN. `n_defensive_credits` counts ALL defending rows incl. unsizable ones — so `(net=0, n>0)` distinguishes fired-but-unsizable from no-credit `(0, 0)` (spec §9.2). Confirm with an explicit test if needed.

- [ ] **Step 4: Add the decorated wrapper to `features.py`**

```python
# silly_kicks/tracking/features.py  (add near the other add_* aggregators; _nan_safety already imported)
from .defensive_credit._orchestration import aggregate_defensive_credit as _aggregate_defensive_credit


@nan_safe_enrichment
def add_defensive_credit(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    xg_column: str,
    xt,
    blocked_column: str = "shot_blocked",
    on_target_column: str = "shot_on_target_derived",
    links: pd.DataFrame | None = None,
    params=None,
) -> pd.DataFrame:
    """Per-action defending-team defensive-credit aggregate (TF-51). See NOTICE for citations.

    No ``home_team_id`` (P-2 — the split derives from ``team_id != acting-team``). Links ONCE and
    threads the pointers through the aggregate + the provenance merge (single-link perf budget).
    """
    pointers = links if links is not None else link_actions_to_frames(actions, frames)[0]  # the ONE link
    out = _aggregate_defensive_credit(
        actions, frames, xg_column=xg_column, xt=xt, blocked_column=blocked_column,
        on_target_column=on_target_column, links=pointers, params=params,
    )
    provenance_cols = ["frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]
    if not any(c in out.columns for c in provenance_cols) and len(pointers) > 0:
        ptr = pointers.drop_duplicates("action_id").set_index("action_id")[provenance_cols]
        out = out.merge(ptr, left_on="action_id", right_index=True, how="left")
    return out
```

> No `home_team_id` (P-2): the earlier draft carried it "for gate parity", but it is genuinely unused (the defending/attacking split derives from `team_id != acting-team`; reprojection uses `acting_team_attacks_rtl`). A dead required kwarg is a Hyrum liability, so it is dropped and the id-dtype gate registers with the `_a` adapter (Task 12), not `_ah`.

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/tracking/test_defensive_credit_aggregate.py -q` — Expected: PASS (3).

- [ ] **Step 6: Lint + commit**

```bash
git add silly_kicks/tracking/defensive_credit/_orchestration.py silly_kicks/tracking/features.py tests/tracking/test_defensive_credit_aggregate.py
git commit -m "feat(tracking): TF-51 add_defensive_credit per-action aggregate (defending-scoped)"
```

---

### Task 10: Bravery (`_bravery.py`) — per-type breakdown + set-piece exposure

**Files:**
- Create: `silly_kicks/tracking/defensive_credit/_bravery.py`
- Test: `tests/tracking/test_bravery.py`

**Spec §9.3:** event-only, per-team. `bravery_shots`, `bravery_open_play_crosses`, `bravery_set_piece_crosses = NaN`, headline `bravery_pct_known_domain` (known domain = shots + open-play crosses), plus faced-counts. Cross-type identified from SPADL action type (`cross` = open-play; `corner_crossed`/`freekick_crossed` = set-piece), NOT from `cross_blocked` non-NA. R2-2 per-type unknown → NaN.

- [ ] **Step 1: Write the failing bravery tests**

```python
# tests/tracking/test_bravery.py
import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking import compute_bravery


def _final_actions(rows):
    base = dict(game_id="g1", period_id=1, bodypart_id=spadlconfig.bodypart_id["foot"],
                start_x=90.0, start_y=34.0, end_x=100.0, end_y=40.0)
    out = []
    for i, r in enumerate(rows):
        d = dict(base); d.update(action_id=i, time_seconds=float(i), **r)
        d["type_id"] = spadlconfig.actiontype_id[d.pop("type_name")]
        d["result_id"] = spadlconfig.result_id[d.pop("result_name", "fail")]
        d.setdefault("shot_blocked", pd.NA); d.setdefault("cross_blocked", pd.NA)
        out.append(d)
    df = pd.DataFrame(out)
    df["shot_blocked"] = pd.array(df["shot_blocked"].tolist(), dtype="boolean")
    df["cross_blocked"] = pd.array(df["cross_blocked"].tolist(), dtype="boolean")
    return df


def test_bravery_worked_example_shots_only():
    # team 20 faces 5 shots by team 10, 4 blocked -> bravery_shots = 0.8
    rows = [dict(type_name="shot", team_id=10, player_id=1, shot_blocked=(i < 4)) for i in range(5)]
    out = compute_bravery(_final_actions(rows))
    row = out[out["team_id"] == 20].iloc[0]  # the DEFENDING team
    assert row["bravery_shots"] == pytest.approx(0.8)
    assert row["n_shots_faced"] == 5


def test_set_piece_crosses_are_exposed_not_dropped():
    rows = [
        dict(type_name="shot", team_id=10, player_id=1, shot_blocked=True),
        dict(type_name="cross", team_id=10, player_id=2, cross_blocked=True),          # open-play
        dict(type_name="corner_crossed", team_id=10, player_id=3, cross_blocked=pd.NA),  # set-piece
        dict(type_name="freekick_crossed", team_id=10, player_id=4, cross_blocked=pd.NA),
    ]
    out = compute_bravery(_final_actions(rows))
    row = out[out["team_id"] == 20].iloc[0]
    assert np.isnan(row["bravery_set_piece_crosses"])   # NaN, never 0
    assert row["n_set_piece_crosses_faced"] == 2         # the gap is exposed
    # headline is over the KNOWN domain (1 shot + 1 open-play cross, both blocked = 1.0),
    # UNCHANGED by the set-piece crosses:
    assert row["bravery_pct_known_domain"] == pytest.approx(1.0)


def test_all_na_cross_column_yields_nan_open_play_component():
    rows = [dict(type_name="cross", team_id=10, player_id=2, cross_blocked=pd.NA) for _ in range(3)]
    out = compute_bravery(_final_actions(rows))
    row = out[out["team_id"] == 20].iloc[0]
    assert np.isnan(row["bravery_open_play_crosses"])  # unknown -> NaN, not 0


def test_both_columns_na_yields_nan_headline_and_warns():
    rows = [dict(type_name="shot", team_id=10, player_id=1, shot_blocked=pd.NA)]
    with pytest.warns(UserWarning, match="bravery"):
        out = compute_bravery(_final_actions(rows))
    row = out[out["team_id"] == 20].iloc[0]
    assert np.isnan(row["bravery_pct_known_domain"])
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/tracking/test_bravery.py -q`
Expected: FAIL — `ImportError: cannot import name 'compute_bravery'`.

- [ ] **Step 3: Implement `_bravery.py`**

```python
# silly_kicks/tracking/defensive_credit/_bravery.py
"""Event-only, per-team bravery — % of opponent final actions blocked, per-type breakdown (R2-2)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from silly_kicks.spadl import config as spadlconfig

_SHOT = spadlconfig.actiontype_id["shot"]
_CROSS = spadlconfig.actiontype_id["cross"]  # open-play cross
_CORNER_CROSSED = spadlconfig.actiontype_id["corner_crossed"]
_FREEKICK_CROSSED = spadlconfig.actiontype_id["freekick_crossed"]
_SET_PIECE_CROSSES = frozenset({_CORNER_CROSSED, _FREEKICK_CROSSED})

_COLS = ["game_id", "team_id", "bravery_shots", "bravery_open_play_crosses",
         "bravery_set_piece_crosses", "bravery_pct_known_domain",
         "n_shots_faced", "n_open_play_crosses_faced", "n_set_piece_crosses_faced", "n_blocks_known"]


def compute_bravery(
    actions: pd.DataFrame,
    *,
    shot_blocked_column: str = "shot_blocked",
    cross_blocked_column: str = "cross_blocked",
) -> pd.DataFrame:
    """Per (game_id, defending team) bravery. The defending team is the OPPONENT of the actor."""
    a = actions
    is_shot = a["type_id"] == _SHOT
    is_open_cross = a["type_id"] == _CROSS
    is_set_cross = a["type_id"].isin(_SET_PIECE_CROSSES)
    final = a[is_shot | is_open_cross | is_set_cross]
    if final.empty:
        return pd.DataFrame({c: pd.Series([], dtype="float64" if "bravery" in c else "object") for c in _COLS})

    out_rows = []
    for (game_id, actor_team), g in final.groupby(["game_id", "team_id"], dropna=True):
        # the DEFENDING team faced these actor_team final actions
        # (single-opponent assumption: the other team on the pitch; per-game bravery is opponent-facing)
        defending_team = _opponent_team(a, game_id, actor_team)
        n_shots = int((g["type_id"] == _SHOT).sum())
        n_open = int((g["type_id"] == _CROSS).sum())
        n_set = int(g["type_id"].isin(_SET_PIECE_CROSSES).sum())

        b_shots = _rate(g, _SHOT, shot_blocked_column)
        b_open = _rate(g, _CROSS, cross_blocked_column)
        b_set = np.nan  # v1 column limitation — always unknown

        known_blocked, known_faced = _known_domain(g, shot_blocked_column, cross_blocked_column)
        headline = (known_blocked / known_faced) if known_faced > 0 and not np.isnan(known_blocked) else np.nan
        if np.isnan(headline):
            warnings.warn(
                f"bravery: game {game_id} team {defending_team} has no known-domain block signal "
                f"(both shot and cross blocked columns unknown) -> bravery_pct_known_domain=NaN.",
                stacklevel=2,
            )
        out_rows.append({
            "game_id": game_id, "team_id": defending_team,
            "bravery_shots": b_shots, "bravery_open_play_crosses": b_open,
            "bravery_set_piece_crosses": b_set, "bravery_pct_known_domain": headline,
            "n_shots_faced": n_shots, "n_open_play_crosses_faced": n_open,
            "n_set_piece_crosses_faced": n_set,
            "n_blocks_known": int(known_blocked) if not np.isnan(known_blocked) else pd.NA,
        })
    df = pd.DataFrame(out_rows, columns=_COLS)
    for c in ("n_shots_faced", "n_open_play_crosses_faced", "n_set_piece_crosses_faced", "n_blocks_known"):
        df[c] = df[c].astype("Int64")
    return df


def _rate(g, type_id, blocked_col):
    sub = g[g["type_id"] == type_id]
    if len(sub) == 0:
        return np.nan
    if blocked_col not in sub.columns or sub[blocked_col].isna().all():
        return np.nan  # R2-2: unknown -> NaN, never 0
    return float((sub[blocked_col] == True).sum()) / float(len(sub))  # noqa: E712


def _known_domain(g, shot_col, cross_col):
    """Blocked-count + faced-count over shots + open-play crosses whose block-status is known."""
    known_blocked = 0.0
    known_faced = 0
    any_known = False
    for type_id, col in ((_SHOT, shot_col), (_CROSS, cross_col)):
        sub = g[g["type_id"] == type_id]
        if len(sub) == 0:
            continue
        if col in sub.columns and not sub[col].isna().all():
            any_known = True
            known_blocked += float((sub[col] == True).sum())  # noqa: E712
            known_faced += len(sub)
    return (known_blocked if any_known else np.nan), known_faced


def _opponent_team(actions, game_id, actor_team):
    """The single opponent team id in this game (two-team assumption)."""
    from silly_kicks.id_compat import same_id
    teams = [t for t in actions[actions["game_id"] == game_id]["team_id"].dropna().unique()
             if not same_id(t, actor_team)]
    return teams[0] if teams else pd.NA
```

> The two-team opponent resolution is a documented v1 assumption (a match has exactly two teams). Add a guard test if a game has ≠2 teams (raise or NaN) — a plan-level hardening if real data ever violates it; for v1 the fixture is two-team.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/tracking/test_bravery.py -q` — Expected: PASS (5).

- [ ] **Step 5: Wire into the package `__init__.py`**

```python
from ._bravery import compute_bravery  # add to imports + __all__
```

- [ ] **Step 6: Lint + commit**

```bash
git add silly_kicks/tracking/defensive_credit/_bravery.py silly_kicks/tracking/defensive_credit/__init__.py tests/tracking/test_bravery.py
git commit -m "feat(tracking): TF-51 bravery per-type breakdown + set-piece exposure"
```

---

### Task 11: Public surface wiring (`tracking/__init__.py`)

**Files:**
- Modify: `silly_kicks/tracking/__init__.py`
- Test: `tests/tracking/test_defensive_credit_public_api.py`

- [ ] **Step 1: Write the failing public-API test**

```python
# tests/tracking/test_defensive_credit_public_api.py
import silly_kicks.tracking as T


def test_public_names_exported():
    for name in ("compute_defensive_credits", "add_defensive_credit", "compute_bravery",
                 "DefensiveCreditParams", "DEFENSIVE_CREDIT_RULES"):
        assert name in T.__all__, f"{name} missing from tracking.__all__"
        assert hasattr(T, name)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/tracking/test_defensive_credit_public_api.py -q` — Expected: FAIL (names not in `__all__`).

- [ ] **Step 3: Wire `tracking/__init__.py`** — add the imports + `__all__` entries (alphabetical among the `add_*`/`compute_*`/params blocks):

```python
# imports (params + primitives from the sub-package; add_* from features)
from .defensive_credit import (
    DEFENSIVE_CREDIT_RULES,
    DefensiveCreditParams,
    compute_bravery,
    compute_defensive_credits,
)
from .features import (
    ...,
    add_defensive_credit,   # add to the existing `from .features import (...)` block
    ...,
)

# __all__ additions:
    "DEFENSIVE_CREDIT_RULES",
    "DefensiveCreditParams",
    "add_defensive_credit",
    "compute_bravery",
    "compute_defensive_credits",
```

- [ ] **Step 4: Run to verify it passes + full sub-package suite green**

Run: `python -m pytest tests/tracking/test_defensive_credit_public_api.py tests/tracking/test_defensive_credit_*.py tests/tracking/test_bravery.py -q`
Expected: PASS.

- [ ] **Step 5: Lint + commit**

```bash
git add silly_kicks/tracking/__init__.py tests/tracking/test_defensive_credit_public_api.py
git commit -m "feat(tracking): TF-51 public surface exports"
```

---

### Task 12: Register with the four auto-enumerating CI gates

**Files:**
- Modify: `tests/test_add_star_purity.py` (PURITY_ENTRIES)
- Modify: `tests/test_enrichment_nan_safety.py` (`_TRACKING_NEEDS_EXTRA` + a dispatch branch)
- Modify: `tests/tracking/test_aggregator_column_liveness.py` (ENTRIES + possibly extend the fixture)
- Modify: `tests/tracking/conftest_id_dtype.py` (AGGREGATORS)

These gates have meta-assertions that FAIL if a new `add_*` export is unregistered — so this task is mandatory, not optional.

- [ ] **Step 1: Run the meta-assertions to see them fail RED first**

Run: `python -m pytest tests/test_add_star_purity.py::test_meta_registration_complete_per_package tests/tracking/test_aggregator_column_liveness.py::test_meta_surface_complete tests/tracking/test_id_dtype_invariance.py::test_enumerated_surface_equals_registered -q`
Expected: FAIL — each reports `add_defensive_credit` missing from its registry.

- [ ] **Step 2: Register in `PURITY_ENTRIES` (`tests/test_add_star_purity.py`)**

Add an entry keyed `"tracking:add_defensive_credit"`. It needs owned `(actions, frames, xt)` inputs plus `xg_column`. Build a small local inputs factory (the shared `make_actions`/`make_frames` do not carry an `xg`/`shot_blocked` column):

```python
def _dc_inputs():
    a = make_actions()
    a = a.copy()
    a["xg"] = 0.2
    a["shot_blocked"] = pd.array([pd.NA] * len(a), dtype="boolean")
    a["cross_blocked"] = pd.array([pd.NA] * len(a), dtype="boolean")
    a["shot_on_target_derived"] = pd.array([pd.NA] * len(a), dtype="boolean")  # present -> no TF-48 fallback
    return [a, make_frames(), _fresh_xt()]


def _dc_invoke(inputs):
    return F.add_defensive_credit(inputs[0], inputs[1], xg_column="xg", xt=inputs[2])


# in PURITY_ENTRIES:
"tracking:add_defensive_credit": _one(_dc_inputs, _dc_invoke),
```

- [ ] **Step 3: Register NaN-safety (`tests/test_enrichment_nan_safety.py`)**

Add `"add_defensive_credit"` to the `_TRACKING_NEEDS_EXTRA` set, then add a dispatch branch in `test_tracking_helper_extra_kwargs_nan_safe` supplying the required kwargs + a `xg`/`shot_blocked`/`cross_blocked` column on the NaN-laced fixture:

```python
    elif name == "add_defensive_credit":
        actions = actions.copy()
        actions["xg"] = 0.1
        actions["shot_blocked"] = pd.array([pd.NA] * len(actions), dtype="boolean")
        actions["cross_blocked"] = pd.array([pd.NA] * len(actions), dtype="boolean")
        actions["shot_on_target_derived"] = pd.array([pd.NA] * len(actions), dtype="boolean")
        out = helper(actions, frames, xg_column="xg", xt=_fresh_xt())  # P-2: no home_team_id
```

(Use the module's existing `_fresh_xt`/`_HOME` helpers, matching the `add_off_ball_run_values`/`add_xt_gk` branches.)

- [ ] **Step 4: Register liveness (`tests/tracking/test_aggregator_column_liveness.py`)**

Add to `ENTRIES` (add `from silly_kicks.spadl import config as spadlconfig` to the test file if absent):

```python
def _dc_runner():
    a = _actions().copy()
    a["xg"] = 0.15
    a["shot_blocked"] = pd.array([pd.NA] * len(a), dtype="boolean")
    a["cross_blocked"] = pd.array([pd.NA] * len(a), dtype="boolean")
    # P-6: defensive_credit_MINUS is non-constant ONLY if the fixture yields a defending-team `-`
    # credit — one of failed_pressure_shot_on_target / beaten_1v1 / failed_marking / failed_cross_block.
    # So set >=1 pressured shot ON-target (-> - debit) AND >=1 OFF-target (-> + credit): shot_on_target_derived
    # True for one shot row, False for another. (An all-False on_target gives + credits only, so _minus
    # is all-0.0 and the liveness non-constant check FAILS on _minus.)
    ot = pd.array([pd.NA] * len(a), dtype="boolean")
    shots = list(a.index[(a["type_id"] == spadlconfig.actiontype_id["shot"]).to_numpy()])
    if shots:
        ot[shots[0]] = False       # off-target -> pressure_on_missed_shot (+)
    if len(shots) > 1:
        ot[shots[1]] = True        # on-target -> failed_pressure_shot_on_target (-) -> drives _minus
    a["shot_on_target_derived"] = ot
    return (a, F.add_defensive_credit(a, _frames(), xg_column="xg", xt=_fresh_xt()))  # P-2: no home_team_id


# in ENTRIES:
"add_defensive_credit": _dc_runner,
```

**Verify the assertion the gate makes** (`tests/tracking/test_aggregator_column_liveness.py:455-500`): the added columns must be non-100%-null, AND every float metric with ≥2 observed values must be non-constant. So `defensive_credit_net`, `_plus`, **and `_minus`** each need ≥2 distinct values across the fixture. `_plus` needs an off-target pressured shot; **`_minus` specifically needs a defending-team `-` credit** (the P-6 point — an all-positive fixture makes `_minus` a constant 0.0 and FAILS the non-constant check). `n_defensive_credits` is `Int64` (dtype-exempt from the non-constant check, but must be non-100%-null). **If the two existing shot windows don't put a defender within threshold**, extend `make_actions`/`make_frames` with a 6th window: an on-target pressured shot with a defender ~1 m away (the `-` scene) plus ensure an off-target one (the `+` scene) — mirroring how TF-35 added action 5. Document the addition in a comment.

- [ ] **Step 5: Register id-dtype invariance (`tests/tracking/conftest_id_dtype.py`)**

`add_defensive_credit` takes no `home_team_id` (P-2) and needs `xg`/`shot_blocked`/`shot_on_target_derived` columns the standard `_a`/`_ah`/`_axh` adapters don't add — so use a custom adapter that augments actions and accepts-but-ignores the harness's `home_team_id` (the `_named` harness always passes one positionally):

```python
def _dc(name):
    def _run(a, f, home_team_id):  # home_team_id accepted for harness parity, NOT forwarded (P-2)
        a = a.copy()
        a["xg"] = 0.1
        a["shot_blocked"] = pd.array([pd.NA] * len(a), dtype="boolean")
        a["cross_blocked"] = pd.array([pd.NA] * len(a), dtype="boolean")
        a["shot_on_target_derived"] = pd.array([pd.NA] * len(a), dtype="boolean")
        return F.add_defensive_credit(a, f, xg_column="xg", xt=_xt())
    return _named(_run, name)


# in AGGREGATORS:
_dc("add_defensive_credit"),
```

- [ ] **Step 6: Run all four gates GREEN**

Run: `python -m pytest tests/test_add_star_purity.py tests/test_enrichment_nan_safety.py tests/tracking/test_aggregator_column_liveness.py tests/tracking/test_id_dtype_invariance.py -q`
Expected: PASS (incl. the meta-assertions).

- [ ] **Step 7: Lint + commit**

```bash
git add tests/test_add_star_purity.py tests/test_enrichment_nan_safety.py tests/tracking/test_aggregator_column_liveness.py tests/tracking/conftest_id_dtype.py
git commit -m "test(tracking): TF-51 register add_defensive_credit with the four auto-gates"
```

---

### Task 13: The no-xfns absence guard (ADR-039/042)

**Files:**
- Create: `tests/tracking/test_defensive_credit_xfns_absence_guard.py`

**Spec §4.1:** an executable, auto-discovering guard enforcing that no `*defensive_credit*` / `*bravery*` transformer appears in any default xfn list (TF-48 no-xfns precedent). There is no xfns factory to anchor, so this is the pure-absence form.

- [ ] **Step 1: Write the guard (copy the TF-48 `test_shot_goalmouth_no_xfns_guard.py` shape)**

```python
# tests/tracking/test_defensive_credit_xfns_absence_guard.py
"""ADR-039/042: defensive credit gates on the action's own result + downstream shot outcome
(F4 result-leakage), so it ships NO xfns factory and MUST NOT appear in any default xfn list."""

import importlib

_MODULES = (
    "silly_kicks.tracking.features",
    "silly_kicks.atomic.tracking.features",
    "silly_kicks.vaep",
    "silly_kicks.vaep.base",
    "silly_kicks.atomic.vaep",
    "silly_kicks.atomic.vaep.base",
)
_FORBIDDEN = ("defensive_credit", "bravery")


def _default_lists():
    found = {}
    for modname in _MODULES:
        try:
            mod = importlib.import_module(modname)
        except ImportError:
            continue
        for attr in dir(mod):
            if "default_xfns" in attr or attr.startswith("xfns_default") or attr.startswith("hybrid_xfns_default"):
                obj = getattr(mod, attr)
                if isinstance(obj, list):
                    found[f"{modname}.{attr}"] = obj
    return found


def test_default_lists_discovered():
    """Floor sanity — the discovery finds SOME default lists (guard isn't vacuous)."""
    assert _default_lists(), "no default xfn lists discovered — the absence guard would be vacuous"


def test_no_defensive_credit_transformer_in_any_default_list():
    for name, lst in _default_lists().items():
        for fn in lst:
            fn_name = getattr(fn, "__name__", str(fn))
            for forbidden in _FORBIDDEN:
                assert forbidden not in fn_name, (
                    f"{name} contains a TF-51 transformer ({fn_name}) — defensive credit gates on "
                    f"result + downstream shot outcome (F4 leakage, ADR-039/042); it ships no xfns "
                    f"and MUST NOT enter a default/union xfn list feeding HybridVAEP."
                )


def test_no_defensive_credit_xfns_factory_exists():
    """TF-51 v1 ships NO xfns factory (spec §4.1). This pins that decision (delete if v2 adds one)."""
    import silly_kicks.tracking as T
    assert not hasattr(T, "defensive_credit_xfns")
    assert not hasattr(T, "bravery_xfns")
```

- [ ] **Step 2: Run — Expected: PASS (green, since no such xfns exist).**

Run: `python -m pytest tests/tracking/test_defensive_credit_xfns_absence_guard.py -q`

- [ ] **Step 3: Commit**

```bash
git add tests/tracking/test_defensive_credit_xfns_absence_guard.py
git commit -m "test(tracking): TF-51 no-xfns absence guard (F4 leakage, ADR-039/042)"
```

---

### Task 14: Cross-provider `cross_blocked` ⊆ cross-type invariant (extend the shipped contract)

**Files:**
- Modify: `tests/spadl/test_block_detection_contract.py`

**Spec §12 (CS-1, owner decision: ships with TF-51):** for every converter, on every emitted actions frame, `cross_blocked` non-`NA` ⊆ (`type == cross`) — i.e. `pd.NA` on every `corner_crossed`/`freekick_crossed`. This is a converter-only property (no frames/xt/TF-51 code); it lives in the already-shipped block-detection contract file but ships in the TF-51 PR because TF-51's bravery denominator is the consumer that motivates it.

- [ ] **Step 1: Add the failing invariant test**

```python
# tests/spadl/test_block_detection_contract.py  (append)
def test_cross_blocked_is_subset_of_open_play_cross_type():
    """CS-1 (TF-51): cross_blocked non-NA => SPADL type == 'cross' (open-play). It must be pd.NA on
    every corner_crossed / freekick_crossed, so the bravery open-play denominator is well-defined.
    Runs on the same converter fixtures the rest of this contract test uses."""
    from silly_kicks.spadl import config as spadlconfig
    from silly_kicks.spadl.utils import add_names

    open_cross = spadlconfig.actiontype_id["cross"]
    for loaded in (
        load_statsbomb(7298),
        load_sportec_native_per_period(),
        load_metrica_native_per_period(),
    ):
        actions = loaded[0] if isinstance(loaded, tuple) else loaded
        if "cross_blocked" not in actions.columns:
            continue
        non_na = actions["cross_blocked"].notna()
        offending = actions[non_na & (actions["type_id"] != open_cross)]
        assert offending.empty, (
            f"cross_blocked is non-NA on {len(offending)} non-open-play-cross rows "
            f"(types {sorted(offending['type_id'].unique())}); it must be pd.NA on set-piece crosses "
            f"(corner_crossed/freekick_crossed). Bravery's open-play denominator depends on this."
        )
```

> If the existing `test_block_detection_contract.py` iterates a known set of provider loaders, reuse that exact iteration (import the same loaders it already imports). The assertion is provider-agnostic — it runs on whatever converters the contract file already exercises. For GS + Wyscout (the two real `cross_blocked` emitters) the invariant holds by construction (verified in the spec review); this test locks it against future drift.

- [ ] **Step 2: Run — Expected: PASS** (the shipped converters already satisfy it).

Run: `python -m pytest tests/spadl/test_block_detection_contract.py -q -k "cross_blocked_is_subset"`

- [ ] **Step 3: Commit**

```bash
git add tests/spadl/test_block_detection_contract.py
git commit -m "test(spadl): TF-51 cross_blocked ⊆ cross-type cross-provider invariant"
```

---

### Task 15: Structural perf budget (call-count spy)

**Files:**
- Create: `tests/tracking/test_defensive_credit_perf_budget.py`

**Spec §12 (M3):** pin `link_actions_to_frames` to exactly ONE call per `add_defensive_credit` invocation over a 100-action batch (the batched-resolution contract).

- [ ] **Step 1: Write the budget test (copy the `test_pre_window_perf_budget.py` shape)**

```python
# tests/tracking/test_defensive_credit_perf_budget.py
import numpy as np
import pandas as pd

from silly_kicks.tracking import features as _features
from tests._perf_structural import call_counter
from tests.tracking.test_pressure_perf_budget import fixture_100  # 100-action (actions, frames)

_ = fixture_100


def test_add_defensive_credit_links_once_per_100_actions(fixture_100, monkeypatch, fitted_xt):
    actions, frames = fixture_100
    actions = actions.copy()
    actions["xg"] = 0.1
    actions["shot_blocked"] = pd.array([pd.NA] * len(actions), dtype="boolean")
    actions["cross_blocked"] = pd.array([pd.NA] * len(actions), dtype="boolean")
    actions["shot_on_target_derived"] = pd.array([pd.NA] * len(actions), dtype="boolean")  # skip TF-48 fallback

    calls = call_counter(monkeypatch, _features, "link_actions_to_frames")
    result = _features.add_defensive_credit(actions, frames, xg_column="xg", xt=fitted_xt)  # P-2: no home_team_id

    assert "defensive_credit_net" in result.columns
    assert calls["n"] == 1, (
        f"add_defensive_credit linked {calls['n']} times for 100 actions (expected 1). "
        "Per-action re-linking is the O(actions) regression the structural budget proxies."
    )
```

> The Task-9 wrapper **already** does the single link: it computes `pointers` once and threads `links=pointers` down the whole chain (`aggregate_defensive_credit` → `compute_defensive_credits` → `_ensure_on_target`'s `add_shot_goalmouth`), and reuses `pointers` for its own provenance merge. So `link_actions_to_frames` fires exactly once — from the wrapper in `features.py`. `call_counter` patches `silly_kicks.tracking.features` (`_features`), the module where the wrapper resolves the name; the orchestrator's own guarded call (in `_orchestration`) never fires because `links` is always supplied down the chain. Providing `shot_on_target_derived` on the fixture skips the TF-48 fallback; even without it, `add_shot_goalmouth(links=pointers)` reuses the pointers and adds no link call. No refactor needed — this test simply pins the contract. **The budget now transitively depends on TF-48 honoring `links=`** (via the on-target fallback); this assertion is the self-check — if `add_shot_goalmouth` ever re-links internally despite `links=pointers`, `calls == 1` fails loud here. To exercise the fallback path in the budget too, add a second variant that omits `shot_on_target_derived` (forcing the TF-48 call) and asserts `calls == 1` still holds.

- [ ] **Step 2: Run — Expected: PASS (calls["n"] == 1).**

Run: `python -m pytest tests/tracking/test_defensive_credit_perf_budget.py -q`

- [ ] **Step 3: Commit**

```bash
git add tests/tracking/test_defensive_credit_perf_budget.py silly_kicks/tracking/features.py
git commit -m "test(tracking): TF-51 structural perf budget (link once per batch)"
```

---

### Task 16: Owner-gated e2e (real match)

**Files:**
- Create: `tests/tracking/test_defensive_credit_e2e.py`

**Spec §12:** on a real GS match (fitted xT + injected xG + the shipped `shot_blocked`/`cross_blocked` columns), assert the family runs end-to-end and produces sane sign/magnitude distributions. Follows the GS goal-capture / sportec-playeval owner-gated-e2e precedent (`@pytest.mark.e2e`, skip without the pining token). Model it on the shipped `tests/spadl/test_gradientsports_block_e2e.py`.

- [ ] **Step 1: Write the e2e (skipped in public CI)**

```python
# tests/tracking/test_defensive_credit_e2e.py
"""Owner-gated e2e: TF-51 defensive-credit family on a real GS match with a fitted xT + injected xG.

SCOPE (P-8): this is a PLUMBING / SANITY smoke — it asserts the family runs end-to-end on real data
and produces sane sign/magnitude distributions. It is NOT xG or xT accuracy validation: the xG is a
crude distance heuristic and xT is fit on the single match. Accuracy validation is the owner-run
SkillCorner cross-check (spec §12), not this test.
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

_MATCH = "10502"
_TOKEN = os.environ.get("PINING_FOR_THE_DATA_TOKEN")
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not _TOKEN, reason="owner-tier Gradient Sports data (PINING_FOR_THE_DATA_TOKEN)"),
]


def _load_loader():
    spec = importlib.util.spec_from_file_location(
        "_loader_pining", str(Path(__file__).parents[2] / "scripts" / "_loader_pining.py")
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _fit_xt(actions):
    from silly_kicks.xthreat import ExpectedThreat
    return ExpectedThreat().fit(actions)


def test_defensive_credit_family_on_real_gs_match():
    from silly_kicks.tracking import add_defensive_credit, compute_bravery, compute_defensive_credits

    L = _load_loader()
    _prov, _m, actions, frames, _home = next(
        iter(L.load_matches(providers=["gradientsports"], match_ids={"gradientsports": [_MATCH]}))
    )
    # injected xG proxy for the e2e (no lakehouse xG here): a crude shot-distance heuristic column.
    from silly_kicks.spadl import config as spadlconfig
    shot = actions["type_id"] == spadlconfig.actiontype_id["shot"]
    dist = np.hypot(105.0 - actions["start_x"], 34.0 - actions["start_y"])
    actions = actions.copy()
    actions["xg"] = np.where(shot, np.clip(0.4 * np.exp(-dist / 12.0), 0.0, 1.0), np.nan)
    xt = _fit_xt(actions)

    long = compute_defensive_credits(actions, frames, xg_column="xg", xt=xt)
    assert not long.empty, "expected some defensive credit rows on a real match"
    assert set(long["rule"]).issubset(set(__import__("silly_kicks.tracking", fromlist=["DEFENSIVE_CREDIT_RULES"]).DEFENSIVE_CREDIT_RULES))
    # sane magnitudes: |signed_value| bounded (xG in [0,1], xT small)
    assert long["signed_value"].abs().max() <= 1.0

    agg = add_defensive_credit(actions, frames, xg_column="xg", xt=xt)  # P-2: no home_team_id; on-target via TF-48 on real frames
    assert (agg["n_defensive_credits"].fillna(0) >= 0).all()
    assert np.isfinite(agg["defensive_credit_net"].to_numpy()).all()  # always finite

    brav = compute_bravery(actions)
    assert (brav["bravery_pct_known_domain"].dropna().between(0.0, 1.0)).all()
    assert brav["n_set_piece_crosses_faced"].sum() >= 0  # set-piece gap exposed
```

- [ ] **Step 2: Run locally with the token (owner), and confirm it SKIPS in public CI**

Run (public): `python -m pytest tests/tracking/test_defensive_credit_e2e.py -q` — Expected: `1 skipped` (no token).
Run (owner, token set): Expected: PASS on real match 10502.

- [ ] **Step 3: Commit**

```bash
git add tests/tracking/test_defensive_credit_e2e.py
git commit -m "test(tracking): TF-51 owner-gated GS e2e"
```

> The **SkillCorner-native construct-validity cross-check** (`scripts/validate_defensive_credit_vs_skillcorner.py`, spec §12) is owner-run, reported-not-gated, and NOT required for the library PR. Add it as a follow-up script if the owner wants the cross-check before publication; it is out of the v1 test-gated scope. (Note it in TODO.md.)

---

### Task 17: Attribution, C4, docs, version bump

**Files:**
- Modify: `NOTICE`, `CLAUDE.md`, `docs/c4/architecture.dsl`, `docs/c4/architecture.html`, `TODO.md`, `CHANGELOG.md`, `pyproject.toml`, `silly_kicks/__init__.py`, `uv.lock`

- [ ] **Step 1: NOTICE — add the attribution block**

Append to the "Mathematical / Methodological References" section:
```
- TF-51 defensive credit/debit family: Sumpter, Soccermatics Pro module 16.3 (coach-consulted credit
  rules); Bischofberger, Bauer, Baca, "Blame is easier than praise" (arXiv:2606.19931) — the published,
  validated precedent for the xT(origin) turnover sizing (xDT = -ΔxT), consistent with the existing
  TF-28 accessible-space/DAS citation. MSC-corpus vocabulary: Tigres Femenil "bravery" metric,
  RB Salzburg 4-action pressing taxonomy.
```
Add `See NOTICE for full bibliographic citations.` to the module docstrings of `_params.py`, `_rules.py`, `_bravery.py`.

- [ ] **Step 2: C4 — bump the aggregator count 30 → 31**

Edit `docs/c4/architecture.dsl` line ~23 (the `tracking` container description): change `30 action-coupled aggregators` → `31 action-coupled aggregators` and (if it fits under the 200-char cap — verify) add `+ defensive credit (TF-51)`. **Verify the description stays ≤ 200 chars** (`python -m pytest tests/test_c4_dsl_description_cap.py -q`), then regenerate `architecture.html` via the c4 skill:
```bash
# export DSL -> puml -> svg -> assemble (see the c4 skill; tools in ~/.claude/tools)
java -jar ~/.claude/tools/structurizr.war export -workspace docs/c4/architecture.dsl -format plantuml/c4plantuml -output <render-dir>
java -jar ~/.claude/tools/plantuml.jar <render-dir>/*.puml -tsvg
python <c4-skill-dir>/c4_assemble.py docs/c4 --svg-dir <render-dir>
```
Run `python -m pytest tests/test_c4_dsl_description_cap.py -q` — Expected: PASS. (There is NO test asserting the count 31 — it is prose; bump it by hand.)

- [ ] **Step 3: CLAUDE.md — add the sub-package bullet**

Add a bullet to the `tracking/` architecture section summarizing TF-51 (new `defensive_credit/` sub-package, 10 rules + bravery, xT(origin) turnover sizing validated by arXiv:2606.19931, no xfns → no retrain, C4 30→31, block-detection prerequisite already shipped 4.56.0).

- [ ] **Step 4: TODO.md + CHANGELOG.md + version bump**

- Update TODO.md TF-51 row → shipped; note the deferred v2 items (atomic mirror, individual cross_block rule, reverse-xT pressing lens, DPA/role model, lane-geometry blocker, line-break-gated through-ball) + the owner-run SkillCorner cross-check script.
- **Add a cross-cutting follow-up TODO line** (reviewer note, not a TF-51 blocker): add a **canonical penalty-area constant to `spadlconfig`** (per ADR-021 "pitch dims live in spadlconfig") and source it from `_ghost_gk.py` (currently 40.3/2 = 20.15), `_xcross_attempt.py` (40.32/2 = 20.16), and TF-51 `_params.py` — closing the 0.01 m box-half-width drift at the root. TF-51 tactically picks the FIFA-correct 40.32 + flags the discrepancy (Chesterton's Fence); the strategic single-source fix is its own small refactor.
- Add a CHANGELOG entry.
- Bump the version in `pyproject.toml`, `silly_kicks/__init__.py` (`__version__`), `uv.lock` (the `silly-kicks` package version line), CHANGELOG header, TODO header — **all 5 spots** — to the owner-confirmed next-free version (assign at commit-prep).

- [ ] **Step 5: Full suite + lint + types green**

Run: `python -m pytest tests/ -m "not e2e and not slow" -q` — Expected: all pass.
Run: `python -m ruff check silly_kicks/ tests/` — Expected: `All checks passed!`
Run: `python -m ruff format --check silly_kicks/tracking/defensive_credit/ tests/tracking/test_defensive_credit_*.py tests/tracking/test_bravery.py`
Run: `python -m pyright` — Expected: 0 errors (whole repo).

- [ ] **Step 6: /final-review, then the single commit (owner-approved)**

Run `/final-review` (mandatory before the single commit — code quality + ADR review + doc drift + C4). Draft the ADR (`docs/superpowers/adrs/ADR-0XX-tf51-defensive-credit.md`) during final-review. Then present `git status --short` + `git diff --stat` + the proposed commit message and get explicit owner approval before `git commit`.

---

## Self-Review

**1. Spec coverage** — every spec section maps to a task:
- §2.1 prerequisite consumption → Tasks 5/10 (shot_blocked/cross_blocked) + Task 14 (invariant).
- §3 D1 xT(origin) → Task 6 (turnover rules) + Task 8 (sizing-regression gate).
- §3 D2 blocked_column port → Task 5/8 (`blocked_column="shot_blocked"` param).
- §3 D3 no atomic mirror → not built (explicitly out; Task 13 pins no xfns; no atomic task).
- §3 D4 bravery event-only per-type → Task 10.
- §4 module structure → Tasks 1–11 (one module per task, `add_*` in features.py per repo convention — a deliberate adaptation of §4's `_orchestration.py`-hosts-`add_*` note, so the nan-safety/purity/liveness gates that scan `tracking.features` see the decorated wrapper).
- §4.2 params → Task 1. §5 rule catalog (10 rules) → Tasks 5/6/7. §5.1 double-debit → Task 6 (pressure_pass_fail + recovery emit −passer). §6 resolution → Task 3. §7 sizing → Task 2. §8 chaining → Task 4. §9.1 long-form → Task 8. §9.2 aggregate → Task 9. §9.3 bravery → Task 10. §10 safety/purity → Task 12. §11 deferred → Task 17 (TODO). §12 validation → Tasks 5–16 (each gate). §13 attribution/C4 → Task 17. §14 open questions → params defaults (Task 1) + noted.

**2. Placeholder scan** — no "TBD"/"handle edge cases"/"similar to Task N". Every code step has real code. The two refactor notes (period_id join in Task 8; single-link threading in Task 9/15) are explicit instructions with the exact change, not placeholders.

**3. Type consistency** — `DEFENSIVE_CREDIT_RULES` (10 names) is used identically in `_params`, `_rules` (RULE_REGISTRY keys), `_orchestration` (enabled filter), and the tests. `CreditRow` fields match `_LONG_COLS` (after the Task-8 period_id-join refactor removes per-row period_id). `compute_defensive_credits` / `add_defensive_credit` / `compute_bravery` signatures are identical across Tasks 8/9/10/11 and the spec §4.1. `blocked_column` default `"shot_blocked"` consistent. Sizing tokens `"xg"`/`"xt"` (`SIZING_XG`/`SIZING_XT`) consistent.

**Resolved during review round 1 (P-1..P-8):**
- **On-target / saved-shot detection (P-1) — RESOLVED (was the one correctness bug).** The orchestrator's `_ensure_on_target` (Task 8) computes a tri-state `_on_target` per shot: goal → True; else the injected `on_target_column` if present; else the frame-based TF-48 `shot_on_target_derived` fallback (reusing the shared link); unknown → NA → the pressure rules DO NOT fire (no fabricated sign). A pressured **saved** shot now correctly yields `failed_pressure_shot_on_target` (−), regression-tested at both the rule and orchestrator level.
- **`home_team_id` dropped (P-2)** — `add_defensive_credit` takes none; the defending split derives from `team_id ≠ acting-team`. id-dtype registers with a custom accept-and-ignore adapter.
- **Single recovery resolver (P-3)** — `_chaining.recovery_after_pass` (ADR-019-safe via `same_id`) is the only one; the `_rules._recovery` duplicate + `if False` branch are gone.
- **`_to_long_form` `period_id` merge (P-4)** is concrete code (merged from `act` on `action_id`), not a footnote — the example runs.
- **Liveness `_minus` (P-6)** — the liveness runner sets ≥1 on-target + ≥1 off-target pressured shot so `defensive_credit_minus` is non-constant; extend the fixture with a negative-defence window if the existing shots don't put a defender in range.
- **Dead code removed (P-7):** `_bravery`'s unused `all_na_shot`.
- **e2e labeled a plumbing/sanity smoke (P-8),** not xG/xT accuracy validation.

**Remaining deliberate decisions (not gaps):**
- Box half-width 40.32/2 = 20.16 chosen over `_ghost_gk`'s 40.3/2 = 20.15 — flagged in `_params.py` (Chesterton's Fence: neither module cites the other).
- Bravery's two-team opponent resolution is a documented v1 assumption (a match has exactly two teams).
- P-5 (spec reconcile) — the spec's §7 `values_at_points(physical_grid(...))` wording + the §12 "physical_grid once" perf item are moot (the real API is `values_at_points(model, x, y)`, no grid build); the spec is updated to match. The per-point `_xt_at` (one `values_at_points` call per rule/action) could batch into one call — optional, low priority, NOT a grid build.
