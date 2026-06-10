# General Restart-Coordinate Enrichment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote the goal-kick-scoped `resolve_gk_geometry` into a general, provider-agnostic restart-coordinate enrichment that imputes NaN coordinates for all Law-fixed-spot restart types (goal-kick / penalty / corner / throw-in) and emits them as new, provenance-tagged columns — without mutating canonical coordinates and without triggering any model retrain.

**Architecture:** A single general engine `resolve_restart_geometry` (the source of truth) lives in `silly_kicks/tracking/_gk_geometry.py`. The existing public `resolve_gk_geometry` becomes a thin delegation shim (rename columns + `restart_prior→goalkick_prior` label map + **revert all non-goalkick rows to native-or-unresolved** + drop dest-confidence) so its 4 call sites stay byte-identical (no retrain). A public `add_restart_coordinates` enrichment in `spadl/utils.py` (frames-optional, lazy-imports tracking) is the consumer entry point. Phase 1 is additive only; canonical `start_x/end_x` are never touched (Phase 2, a future PR, does the canonical promotion + coordinated retrain).

**Tech Stack:** Python, numpy, pandas. No new dependency, no training. Patterns: `@nan_safe_enrichment` (ADR-003), `_id_compat` (ADR-019), frames-optional lazy import (ADR-005 §5), geometry tripwire (ADR-018 style), `XtGkReport`-style frozen dataclass report.

**Spec:** `docs/superpowers/specs/2026-06-10-general-restart-coordinate-enrichment-design.md`.

**Rules honored:** feature branch in the main checkout (NOT a worktree); single commit at the end (docs bundle into the feature commit); no commit/PR without explicit approval.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `silly_kicks/tracking/_gk_geometry.py` | Modify | Add `resolve_restart_geometry` (general engine) + `_tracking_ball_xy` + per-type rule-point/side helpers + tripwire; convert `resolve_gk_geometry` to the delegation shim. |
| `silly_kicks/tracking/_restart_report.py` | Create | `RestartCoordinateReport` frozen dataclass (counts per source × type + tripwire reversions). |
| `silly_kicks/spadl/utils.py` | Modify | Add public `@nan_safe_enrichment add_restart_coordinates` (frames-optional; lazy-imports the engine). |
| `silly_kicks/tracking/__init__.py` | Modify | Export `resolve_restart_geometry`, `RestartCoordinateReport`. |
| `silly_kicks/spadl/__init__.py` | Modify | Export `add_restart_coordinates`. |
| `tests/tracking/test_gk_geometry.py` | Modify | Parity-baseline hardening (exact column set, throw-in revert, edge cases). |
| `tests/tracking/test_restart_geometry.py` | Create | General engine: per-type tiers, side precedence, tripwire, no-mutation, events-only. |
| `tests/spadl/test_add_restart_coordinates.py` | Create | Public helper: columns, frames-optional, sort, NaN-safety. |
| `tests/tracking/test_restart_xtgk_parity.py` | Create | Full xT-GK + completion parity across all 4 call sites (red-first capture). |

---

## Constants & shared design (referenced by multiple tasks)

Add to `_gk_geometry.py` (alongside the existing `_GOALKICK`, `_GOAL_AREA_DEPTH`, `_RULE_POINT`, `_CONF`):

```python
_PENALTY = spadlconfig.actiontype_id["shot_penalty"]
_THROW_IN = spadlconfig.actiontype_id["throw_in"]
_CORNER_TYPES = (spadlconfig.actiontype_id["corner_crossed"], spadlconfig.actiontype_id["corner_short"])
_RESTART_PRIOR_TYPES = (_GOALKICK, _PENALTY, _THROW_IN, *_CORNER_TYPES)  # types that get a rule-point

_PENALTY_SPOT = (spadlconfig.field_length - 11.0, spadlconfig.field_width / 2.0)  # (94.0, 34.0) LTR
_CORNER_X = spadlconfig.field_length          # 105.0 (opponent goal line)
_TOUCHLINE_LO, _TOUCHLINE_HI = 0.0, spadlconfig.field_width  # 0.0 / 68.0
_MID_Y = spadlconfig.field_width / 2.0        # 34.0 (side split)

# Per-type restart-prior confidence (generic source label is always "restart_prior";
# confidence varies by type). goalkick 0.2 is FROZEN (parity). Others provisional (spec §4.4).
_PRIOR_CONF = {_GOALKICK: 0.2, _PENALTY: 0.5, _THROW_IN: 0.3,
               _CORNER_TYPES[0]: 0.4, _CORNER_TYPES[1]: 0.4}
_CONF_TRACKING_BALL = 0.8       # origin
_CONF_TRACKING_BALL_DEST = 0.5  # dest (provisional; spec §9 may drop)
_CONF_NEXT_EVENT = 0.6
_CONF_TRACKING_GK = 0.7         # FROZEN (goalkick parity)

# Tripwire regions (LTR; imputed coords only). Tolerances provisional (spec §6).
_TRIPWIRE = {
    "goalkick": lambda x, y: x <= _GOAL_AREA_DEPTH,
    "penalty": lambda x, y: abs(x - _PENALTY_SPOT[0]) <= 3.0 and abs(y - _PENALTY_SPOT[1]) <= 3.0,
    "corner": lambda x, y: x >= 100.0 and (y <= 5.0 or y >= 63.0),
    "throw_in": lambda x, y: y <= 3.0 or y >= 65.0,
}
```

New general source enum (the `*_coord_source` values): `native`, `tracking_ball`, `tracking_gk`, `restart_prior`, `next_event`, `unresolved`, and `tripwire_reverted` (origin only — set by `apply_restart_tripwire` at the edge, Task 4/7).

---

## Task 1: Parity baseline — pin current `resolve_gk_geometry` behavior (GREEN guard)

This task adds the regression guard BEFORE any refactor (TDD: capture current behavior). These tests must pass against the **current, unmodified** code.

**Files:**
- Test: `tests/tracking/test_gk_geometry.py` (extend)

- [ ] **Step 1: Add the exact-column-set + throw-in-passthrough + edge-case tests**

Append to `tests/tracking/test_gk_geometry.py`:

```python
_THROW = 2  # throw_in type_id


class TestResolveGkGeometryFrozenContract:
    """Pins the pre-promotion contract so the Task-5 delegation shim stays byte-identical."""

    def test_exact_output_columns_no_dest_confidence(self):
        g = resolve_gk_geometry(_actions(), frames=None)
        assert set(g.columns) == {
            "origin_x", "origin_y", "origin_source", "origin_confidence",
            "dest_x", "dest_y", "dest_source",
        }  # note: NO dest_confidence column in the frozen contract

    def test_nongoalkick_throwin_not_imputed(self):
        # A throw_in with NaN origin must stay native-or-unresolved (goalkick-only imputation).
        a = _actions(type_id=[_THROW, _GK], start_x=[np.nan, 5.0], start_y=[np.nan, 34.0])
        g = resolve_gk_geometry(a, frames=None)
        assert g.loc[0, "origin_source"] == "unresolved"
        assert np.isnan(g.loc[0, "origin_x"])

    def test_offposition_gk_goalkick_falls_to_rule_point(self):
        # (Major-2a) off-position GK must NOT be used; falls to goalkick_prior.
        frames = pd.DataFrame({
            "game_id": [9], "period_id": [1], "frame_id": [1250], "time_seconds": [50.0],
            "team_id": [1], "player_id": [10], "is_goalkeeper": [True], "is_ball": [False],
            "x": [40.0], "y": [33.0], "source_provider": ["sportec"],
        })
        g = resolve_gk_geometry(_actions(), frames=frames)
        assert g.loc[1, "origin_source"] == "goalkick_prior"
        assert g.loc[1, "origin_x"] == pytest.approx(5.5)

    def test_goalkick_no_native_end_no_next_event_unresolved(self):
        # (Major-2b) NaN end + last row -> dest unresolved (must STAY unresolved post-refactor).
        g = resolve_gk_geometry(_actions(end_x=[55.0, np.nan], end_y=[34.0, np.nan]), frames=None)
        assert g.loc[1, "dest_source"] == "unresolved"
        assert np.isnan(g.loc[1, "dest_x"])
```

- [ ] **Step 2: Run — expect PASS against current code (this is the baseline guard)**

Run: `python -m pytest tests/tracking/test_gk_geometry.py -v`
Expected: PASS (all, including the 4 new tests). If `test_nongoalkick_throwin_not_imputed` or the column-set test fails, STOP — the assumed current contract is wrong; re-read `_gk_geometry.py` and correct the baseline before proceeding.

- [ ] **Step 3: Capture the committed GOLDEN snapshot of the pre-refactor output (Medium-3 guard)**

This is the real byte-identical guard for the Task-5 refactor — a full-frame snapshot pins column
set + order + dtypes + every cell, which per-cell asserts miss. Generate it on the **current,
unmodified** code, then commit it.

Run (writes two golden parquet files under a committed fixtures dir):

```bash
python -c "
import numpy as np, pandas as pd
from silly_kicks.tracking._gk_geometry import resolve_gk_geometry
a = pd.DataFrame(dict(
    game_id=[9,9,9,9], period_id=[1,1,1,1], action_id=[0,1,2,3], team_id=[1,1,1,1],
    player_id=[10,11,12,10], type_id=[22,5,2,0], time_seconds=[5.,6.,7.,70.],
    start_x=[np.nan,np.nan,np.nan,50.], start_y=[np.nan,np.nan,np.nan,30.],
    end_x=[60.,95.,40.,np.nan], end_y=[30.,10.,20.,np.nan]))
fr = pd.DataFrame(dict(game_id=[9],period_id=[1],frame_id=[1250],time_seconds=[5.],team_id=[1],
    player_id=[10],is_goalkeeper=[True],is_ball=[False],x=[4.0],y=[33.0],source_provider=['sportec']))
import os; os.makedirs('tests/tracking/_fixtures', exist_ok=True)
resolve_gk_geometry(a, frames=None).to_parquet('tests/tracking/_fixtures/gk_geometry_golden_noframes.parquet')
resolve_gk_geometry(a, frames=fr).to_parquet('tests/tracking/_fixtures/gk_geometry_golden_frames.parquet')
print('golden written')
"
```

Then add the golden assertion test to `tests/tracking/test_gk_geometry.py` (it PASSES now and must
keep passing after Task 5):

```python
import pathlib  # noqa: E402

_FIX = pathlib.Path(__file__).parent / "_fixtures"


class TestGoldenSnapshot:
    def _multi(self):
        return pd.DataFrame(dict(
            game_id=[9, 9, 9, 9], period_id=[1, 1, 1, 1], action_id=[0, 1, 2, 3], team_id=[1, 1, 1, 1],
            player_id=[10, 11, 12, 10], type_id=[22, 5, 2, 0], time_seconds=[5., 6., 7., 70.],
            start_x=[np.nan, np.nan, np.nan, 50.], start_y=[np.nan, np.nan, np.nan, 30.],
            end_x=[60., 95., 40., np.nan], end_y=[30., 10., 20., np.nan]))

    def _frames(self):
        return pd.DataFrame(dict(game_id=[9], period_id=[1], frame_id=[1250], time_seconds=[5.],
            team_id=[1], player_id=[10], is_goalkeeper=[True], is_ball=[False], x=[4.0], y=[33.0],
            source_provider=["sportec"]))

    def test_golden_noframes(self):
        got = resolve_gk_geometry(self._multi(), frames=None)
        pd.testing.assert_frame_equal(got, pd.read_parquet(_FIX / "gk_geometry_golden_noframes.parquet"))

    def test_golden_frames(self):
        got = resolve_gk_geometry(self._multi(), frames=self._frames())
        pd.testing.assert_frame_equal(got, pd.read_parquet(_FIX / "gk_geometry_golden_frames.parquet"))
```

Run: `python -m pytest tests/tracking/test_gk_geometry.py::TestGoldenSnapshot -v`
Expected: PASS against current code.

- [ ] **Step 4: Commit checkpoint** (intermediate; squashed into the final feature commit — do NOT push)

```bash
git add tests/tracking/test_gk_geometry.py tests/tracking/_fixtures/gk_geometry_golden_noframes.parquet tests/tracking/_fixtures/gk_geometry_golden_frames.parquet
git commit -m "test(tracking): pin resolve_gk_geometry frozen contract + golden snapshot pre-promotion"
```
(Commit requires the sentinel/approval per house rules — if blocked, keep changes staged and continue; the final single commit covers everything.)

---

## Task 2: `_tracking_ball_xy` primitive (RED→GREEN)

**Files:**
- Modify: `silly_kicks/tracking/_gk_geometry.py`
- Test: `tests/tracking/test_restart_geometry.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/tracking/test_restart_geometry.py`:

```python
"""Tests for resolve_restart_geometry (general restart-coordinate enrichment).

Spec: docs/superpowers/specs/2026-06-10-general-restart-coordinate-enrichment-design.md
"""
import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._gk_geometry import _tracking_ball_xy, resolve_restart_geometry

_GK, _CORNER_C, _THROW, _PEN, _PASS = 22, 5, 2, 12, 0


def _frame(**over):
    base = dict(
        game_id=[9, 9], period_id=[1, 1], frame_id=[1250, 1250], time_seconds=[50.0, 50.0],
        team_id=[1, 0], player_id=[10, -1], is_goalkeeper=[False, False],
        is_ball=[False, True], x=[50.0, 104.5], y=[20.0, 0.5], source_provider=["gradientsports", "gradientsports"],
    )
    base.update(over)
    return pd.DataFrame(base)


def test_tracking_ball_xy_selects_ball_row():
    from silly_kicks.tracking._kernels import resolve_frame_ids_by_position

    a = pd.DataFrame(dict(game_id=[9], period_id=[1], action_id=[7], team_id=[1],
                          type_id=[_CORNER_C], time_seconds=[50.0],
                          start_x=[np.nan], start_y=[np.nan], end_x=[95.0], end_y=[10.0]))
    # Minor-9: assert the fixture actually links first, so a linkage-fixture failure is
    # distinguishable from a ball-selection bug.
    assert np.isfinite(resolve_frame_ids_by_position(a, _frame(), links=None)[0]), "fixture linkage failed"
    xy = _tracking_ball_xy(a, _frame(), links=None)
    assert xy[0, 0] == pytest.approx(104.5)
    assert xy[0, 1] == pytest.approx(0.5)


def test_tracking_ball_xy_coerces_string_is_ball():
    # ADR-019: object/string is_ball must be coerced, not assumed bool (~is_ball no-op bug).
    a = pd.DataFrame(dict(game_id=[9], period_id=[1], action_id=[7], team_id=[1],
                          type_id=[_CORNER_C], time_seconds=[50.0],
                          start_x=[np.nan], start_y=[np.nan], end_x=[95.0], end_y=[10.0]))
    fr = _frame(is_ball=["False", "True"])
    xy = _tracking_ball_xy(a, fr, links=None)
    assert xy[0, 0] == pytest.approx(104.5)
```

- [ ] **Step 2: Run — expect FAIL (`_tracking_ball_xy` undefined)**

Run: `python -m pytest tests/tracking/test_restart_geometry.py::test_tracking_ball_xy_selects_ball_row -v`
Expected: FAIL — `ImportError: cannot import name '_tracking_ball_xy'`.

- [ ] **Step 3: Implement `_tracking_ball_xy`**

Add to `_gk_geometry.py` (sibling of `_tracking_gk_xy`):

```python
def _tracking_ball_xy(
    actions: pd.DataFrame, frames: pd.DataFrame, links: pd.DataFrame | None
) -> np.ndarray:
    """Ball position at each action's linked frame; NaN where unavailable. The ball IS the
    dead-ball restart spot. ADR-019: ``is_ball`` is coerced (object/string ``"True"``/``"False"``
    columns would make a bare ``~is_ball`` a no-op)."""
    from ._kernels import resolve_frame_ids_by_position

    n = len(actions)
    res = np.full((n, 2), np.nan, dtype=float)
    fid = resolve_frame_ids_by_position(actions, frames, links=links)
    is_ball = frames["is_ball"].astype(bool)  # coerce (ADR-019)
    ball_frames = frames[is_ball]
    fg = ball_frames.groupby("frame_id")
    for i in range(n):
        if not np.isfinite(fid[i]):
            continue
        try:
            fr = fg.get_group(int(fid[i]))
        except KeyError:
            continue
        if fr.empty:
            continue
        res[i] = (float(fr.iloc[0]["x"]), float(fr.iloc[0]["y"]))
    return res
```

- [ ] **Step 4: Run — expect PASS (both ball tests)**

Run: `python -m pytest tests/tracking/test_restart_geometry.py -k tracking_ball -v`
Expected: PASS.

- [ ] **Step 5: Commit checkpoint** (staged; final commit covers it)

```bash
git add silly_kicks/tracking/_gk_geometry.py tests/tracking/test_restart_geometry.py
git commit -m "feat(tracking): _tracking_ball_xy ball-position primitive (ADR-019 coerce)"
```

---

## Task 3: `resolve_restart_geometry` general engine (RED→GREEN)

**Files:**
- Modify: `silly_kicks/tracking/_gk_geometry.py`
- Test: `tests/tracking/test_restart_geometry.py`

- [ ] **Step 1: Write the failing tests (per-type tiers + side precedence + no-mutation + events-only)**

Append to `tests/tracking/test_restart_geometry.py`:

```python
def _restart(type_id, **over):
    base = dict(
        game_id=[9, 9], period_id=[1, 1], action_id=[0, 1], team_id=[1, 1], player_id=[10, 11],
        time_seconds=[5.0, 6.0], type_id=[type_id, _PASS],
        start_x=[np.nan, 50.0], start_y=[np.nan, 30.0], end_x=[np.nan, 60.0], end_y=[np.nan, 30.0],
    )
    base.update(over)
    return pd.DataFrame(base)


class TestResolveRestartGeometryEventsOnly:
    def test_goalkick_origin_rule_point(self):
        g = resolve_restart_geometry(_restart(_GK), frames=None)
        assert g.loc[0, "enriched_start_x"] == pytest.approx(5.5)
        assert g.loc[0, "start_coord_source"] == "restart_prior"
        assert g.loc[0, "start_coord_confidence"] == pytest.approx(0.2)

    def test_penalty_origin_rule_point(self):
        g = resolve_restart_geometry(_restart(_PEN), frames=None)
        assert g.loc[0, "enriched_start_x"] == pytest.approx(94.0)
        assert g.loc[0, "enriched_start_y"] == pytest.approx(34.0)
        assert g.loc[0, "start_coord_confidence"] == pytest.approx(0.5)

    def test_corner_side_from_native_end_y(self):
        # native end_y=10 (<34) -> near corner (105, 0)
        g = resolve_restart_geometry(_restart(_CORNER_C, end_x=[95.0, 60.0], end_y=[10.0, 30.0]), frames=None)
        assert g.loc[0, "enriched_start_x"] == pytest.approx(105.0)
        assert g.loc[0, "enriched_start_y"] == pytest.approx(0.0)
        assert g.loc[0, "start_coord_source"] == "restart_prior"

    def test_corner_side_unresolvable_stays_unresolved(self):
        # no native end, no next-event y, no frames -> cannot determine side -> unresolved
        g = resolve_restart_geometry(_restart(_CORNER_C, end_x=[np.nan, np.nan], end_y=[np.nan, np.nan],
                                              start_x=[np.nan, np.nan], start_y=[np.nan, np.nan]), frames=None)
        assert g.loc[0, "start_coord_source"] == "unresolved"

    def test_openplay_pass_no_rule_point(self):
        # a pass with NaN origin gets NO rule-point (events-only -> unresolved)
        g = resolve_restart_geometry(_restart(_PASS), frames=None)
        assert g.loc[0, "start_coord_source"] == "unresolved"

    def test_dest_next_event_full_frame(self):
        g = resolve_restart_geometry(_restart(_GK), frames=None)
        # row0 NaN end -> next row (action 1) start (50,30)
        assert g.loc[0, "enriched_end_x"] == pytest.approx(50.0)
        assert g.loc[0, "end_coord_source"] == "next_event"

    def test_does_not_mutate_input(self):
        a = _restart(_GK)
        before_sx, before_ex = a["start_x"].copy(), a["end_x"].copy()
        resolve_restart_geometry(a, frames=None)
        pd.testing.assert_series_equal(a["start_x"], before_sx)
        pd.testing.assert_series_equal(a["end_x"], before_ex)

    def test_emits_new_column_contract(self):
        g = resolve_restart_geometry(_restart(_GK), frames=None)
        assert {"enriched_start_x", "enriched_start_y", "start_coord_source", "start_coord_confidence",
                "enriched_end_x", "enriched_end_y", "end_coord_source", "end_coord_confidence"} <= set(g.columns)
```

- [ ] **Step 2: Run — expect FAIL (`resolve_restart_geometry` undefined)**

Run: `python -m pytest tests/tracking/test_restart_geometry.py::TestResolveRestartGeometryEventsOnly -v`
Expected: FAIL — import error.

- [ ] **Step 3: Implement `resolve_restart_geometry` + helpers**

Add to `_gk_geometry.py`:

```python
def _side_y(actions: pd.DataFrame, frames, links) -> np.ndarray:
    """Y used to pick a corner/throw-in side: native end_y -> next-event start_y -> tracking-ball y.
    NaN where none resolves (caller leaves the row unresolved -- never guess a side)."""
    side = actions["end_y"].to_numpy(float).copy()
    _, ny = _next_event_start(actions)
    side = np.where(np.isfinite(side), side, ny)
    if frames is not None:
        ball = _tracking_ball_xy(actions, frames, links)
        side = np.where(np.isfinite(side), side, ball[:, 1])
    return side


def _throwin_x(actions: pd.DataFrame, frames, links) -> np.ndarray:
    """X along the touchline for a throw-in: next-event start_x -> tracking-ball x. NaN if none."""
    nx, _ = _next_event_start(actions)
    x = nx.copy()
    if frames is not None:
        ball = _tracking_ball_xy(actions, frames, links)
        x = np.where(np.isfinite(x), x, ball[:, 0])
    return x


def resolve_restart_geometry(
    actions: pd.DataFrame,
    *,
    frames: pd.DataFrame | None = None,
    links: pd.DataFrame | None = None,
    impute_types: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """General restart-coordinate enrichment. Returns an index-aligned frame with
    enriched_start_x/_y/enriched_end_x/_y + start_coord_source/start_coord_confidence +
    end_coord_source/end_coord_confidence. NEVER mutates ``actions``. PURE: emits no warnings and
    applies no tripwire (the tripwire is a feature-policy step applied at the
    ``add_restart_coordinates`` edge; spec §6) -- so the ``resolve_gk_geometry`` shim that delegates
    here can never leak a warning onto the frozen ``compute_xt_gk`` path.

    PRECONDITION: ``actions`` is in chronological ``(game_id, period_id, action_id)`` order
    (the ``next_event`` ``shift(-1)`` is positional). The public ``add_restart_coordinates``
    sorts first; callers passing pre-sorted SPADL streams (e.g. ``compute_xt_gk``) satisfy this.

    ``impute_types``: action-type ids eligible for imputation past ``native``. ``None`` = all types
    (the general default). The ``resolve_gk_geometry`` shim passes ``(goalkick,)`` so non-goalkick
    rows are NEVER imputed (parity: matches the frozen goalkick-only contract; perf: zero
    ``_tracking_ball_xy`` work on the frozen hot path).

    Origin tiers (confidence order): native -> [goalkick: in-area tracking_gk; non-goalkick:
    tracking_ball] -> restart_prior (goalkick/penalty/corner/throw_in only) -> unresolved.
    Destination tiers: native -> next_event (full-frame positional) -> tracking_ball (non-goalkick
    only) -> unresolved. tracking_ball is gated OFF for goal-kicks (origin AND dest) so
    ``resolve_gk_geometry`` stays byte-identical (spec §4.1 invariant).

    See NOTICE; spec 2026-06-10-general-restart-coordinate-enrichment-design.md.
    """
    n = len(actions)
    out = pd.DataFrame(index=actions.index)
    sx = actions["start_x"].to_numpy(float)
    sy = actions["start_y"].to_numpy(float)
    ex = actions["end_x"].to_numpy(float)
    ey = actions["end_y"].to_numpy(float)
    tid = actions["type_id"].to_numpy()
    is_gk = tid == _GOALKICK
    is_corner = np.isin(tid, _CORNER_TYPES)
    is_throw = tid == _THROW_IN
    eligible = np.ones(n, dtype=bool) if impute_types is None else np.isin(tid, tuple(impute_types))

    # ---------- origin ----------
    ox, oy = sx.copy(), sy.copy()
    osrc = np.where(np.isfinite(sx) & np.isfinite(sy), "native", "unresolved").astype(object)
    oconf = np.where(osrc == "native", 1.0, 0.0).astype(float)

    need = (osrc == "unresolved") & eligible
    # tier 2a: goalkick in-area tracking-GK (goalkick ONLY; no tracking_ball for goalkick)
    if frames is not None and (need & is_gk).any():
        gk = _tracking_gk_xy(actions, frames, links)
        use = need & is_gk & np.isfinite(gk[:, 0])
        ox[use], oy[use] = gk[use, 0], gk[use, 1]
        osrc[use], oconf[use] = "tracking_gk", _CONF_TRACKING_GK
        need = (osrc == "unresolved") & eligible
    # tier 2b: tracking-ball (NON-goalkick eligible rows). Skipped entirely on the goalkick-only
    # (frozen) path -- (need & ~is_gk) is empty there, so _tracking_ball_xy is never called.
    if frames is not None and (need & ~is_gk).any():
        ball = _tracking_ball_xy(actions, frames, links)
        use = need & ~is_gk & np.isfinite(ball[:, 0])
        ox[use], oy[use] = ball[use, 0], ball[use, 1]
        osrc[use], oconf[use] = "tracking_ball", _CONF_TRACKING_BALL
        need = (osrc == "unresolved") & eligible
    # tier 3: restart rule-points (restart-prior types only). _side_y / _throwin_x computed ONLY
    # when a corner/throw-in actually needs them (avoids wasted _tracking_ball_xy on the frozen path).
    side = _side_y(actions, frames, links) if (need & (is_corner | is_throw)).any() else None
    twx = _throwin_x(actions, frames, links) if (need & is_throw).any() else None
    for i in np.where(need)[0]:
        t = tid[i]
        if t == _GOALKICK:
            ox[i], oy[i] = _RULE_POINT
        elif t == _PENALTY:
            ox[i], oy[i] = _PENALTY_SPOT
        elif t in _CORNER_TYPES:
            if side is None or not np.isfinite(side[i]):
                continue  # cannot determine side -> leave unresolved
            ox[i], oy[i] = _CORNER_X, (_TOUCHLINE_LO if side[i] < _MID_Y else _TOUCHLINE_HI)
        elif t == _THROW_IN:
            if side is None or twx is None or not (np.isfinite(side[i]) and np.isfinite(twx[i])):
                continue
            ox[i], oy[i] = twx[i], (_TOUCHLINE_LO if side[i] < _MID_Y else _TOUCHLINE_HI)
        else:
            continue  # open-play / freekick_short -> no rule-point
        osrc[i], oconf[i] = "restart_prior", _PRIOR_CONF[t]

    out["enriched_start_x"], out["enriched_start_y"] = ox, oy
    out["start_coord_source"], out["start_coord_confidence"] = osrc, oconf

    # ---------- destination ----------
    dx, dy = ex.copy(), ey.copy()
    dsrc = np.where(np.isfinite(ex) & np.isfinite(ey), "native", "unresolved").astype(object)
    dconf = np.where(dsrc == "native", 1.0, 0.0).astype(float)
    dneed = (dsrc == "unresolved") & eligible
    # tier 2: next_event (eligible rows; full-frame positional). On the goalkick-only path this fires
    # for goalkicks only -> matches the frozen contract's goalkick-gated next_event.
    if dneed.any():
        nx, ny = _next_event_start(actions)
        use = dneed & np.isfinite(nx) & np.isfinite(ny)
        dx[use], dy[use] = nx[use], ny[use]
        dsrc[use], dconf[use] = "next_event", _CONF_NEXT_EVENT
        dneed = (dsrc == "unresolved") & eligible
    # tier 3: tracking-ball dest (NON-goalkick eligible rows). Empty on the goalkick-only path.
    if frames is not None and (dneed & ~is_gk).any():
        ball = _tracking_ball_xy(actions, frames, links)
        use = dneed & ~is_gk & np.isfinite(ball[:, 0])
        dx[use], dy[use] = ball[use, 0], ball[use, 1]
        dsrc[use], dconf[use] = "tracking_ball", _CONF_TRACKING_BALL_DEST

    out["enriched_end_x"], out["enriched_end_y"] = dx, dy
    out["end_coord_source"], out["end_coord_confidence"] = dsrc, dconf
    return out  # NO tripwire here -- applied at the add_restart_coordinates edge (Task 7)
```

- [ ] **Step 4: Run — expect PASS (events-only engine tests)**

Run: `python -m pytest tests/tracking/test_restart_geometry.py::TestResolveRestartGeometryEventsOnly -v`
Expected: PASS (all 8).

- [ ] **Step 5: Add the frames-path tier tests + run**

Append:

```python
class TestResolveRestartGeometryFrames:
    def _frames(self, **over):
        base = dict(
            game_id=[9], period_id=[1], frame_id=[1250], time_seconds=[5.0],
            team_id=[0], player_id=[-1], is_goalkeeper=[False], is_ball=[True],
            x=[104.6], y=[0.4], source_provider=["gradientsports"],
        )
        base.update(over)
        return pd.DataFrame(base)

    def test_corner_origin_tracking_ball_beats_prior(self):
        a = _restart(_CORNER_C)
        a.loc[0, "action_id"], a.loc[1, "action_id"] = 0, 1
        g = resolve_restart_geometry(a, frames=self._frames())
        assert g.loc[0, "start_coord_source"] == "tracking_ball"
        assert g.loc[0, "enriched_start_x"] == pytest.approx(104.6)
        assert g.loc[0, "start_coord_confidence"] == pytest.approx(0.8)

    def test_goalkick_never_uses_tracking_ball(self):
        # goalkick + ball tracked at midfield: must NOT pick the ball; in-area-GK absent -> rule-point
        a = _restart(_GK)
        g = resolve_restart_geometry(a, frames=self._frames(x=[50.0], y=[20.0]))
        assert g.loc[0, "start_coord_source"] == "restart_prior"  # never tracking_ball
```

Run: `python -m pytest tests/tracking/test_restart_geometry.py::TestResolveRestartGeometryFrames -v`
Expected: PASS.

- [ ] **Step 6: Commit checkpoint** (staged)

```bash
git add silly_kicks/tracking/_gk_geometry.py tests/tracking/test_restart_geometry.py
git commit -m "feat(tracking): resolve_restart_geometry general engine (per-type tiers, side precedence)"
```

---

## Task 4: Geometry tripwire — pure edge helper (RED→GREEN)

The tripwire is a **feature-policy** step applied at the `add_restart_coordinates` edge (Task 7),
**NOT** inside the engine — so the frozen `resolve_gk_geometry` path (which delegates to the engine)
never tripwires, by construction (spec §6, Major-1 fix). This task builds the pure helper; Task 7 calls
it. Revert semantics are **locked: revert-to-`unresolved`** (no re-run loop), and reverted rows are
tagged with the distinct source **`tripwire_reverted`** (so the report can count them, Medium-4 fix).
Origin-only (destinations unguarded in Phase 1, spec §6).

**Files:**
- Modify: `silly_kicks/tracking/_gk_geometry.py`
- Test: `tests/tracking/test_restart_geometry.py`

- [ ] **Step 1: Write the failing test (test the helper directly — it's pure)**

Append to `tests/tracking/test_restart_geometry.py`:

```python
from silly_kicks.tracking._gk_geometry import apply_restart_tripwire  # noqa: E402


class TestTripwire:
    def _enriched(self, source, x, y, type_id=_CORNER_C):
        # minimal enriched frame (as resolve_restart_geometry would emit) for one origin row
        return pd.DataFrame({
            "type_id": [type_id],
            "enriched_start_x": [x], "enriched_start_y": [y],
            "start_coord_source": [source],
            "start_coord_confidence": [{"native": 1.0, "restart_prior": 0.4}.get(source, 0.8)],
            "enriched_end_x": [60.0], "enriched_end_y": [30.0],
            "end_coord_source": ["native"], "end_coord_confidence": [1.0],
        })

    def test_imputed_out_of_region_reverts_to_tripwire_reverted(self):
        # an imputed corner at midfield (x=50) violates the corner region (x>=100) -> reverted
        df = self._enriched("tracking_ball", 50.0, 20.0)
        with pytest.warns(UserWarning):
            n = apply_restart_tripwire(df)
        assert n == 1
        assert df.loc[0, "start_coord_source"] == "tripwire_reverted"
        assert df.loc[0, "start_coord_confidence"] == pytest.approx(0.0)
        assert np.isnan(df.loc[0, "enriched_start_x"])

    def test_native_out_of_region_warns_not_reverted(self):
        # native coord out of region is provider truth -> warn-only, keep native
        df = self._enriched("native", 80.0, 34.0, type_id=_GK)
        with pytest.warns(UserWarning):
            n = apply_restart_tripwire(df)
        assert n == 0
        assert df.loc[0, "start_coord_source"] == "native"
        assert df.loc[0, "enriched_start_x"] == pytest.approx(80.0)

    def test_in_region_imputed_untouched(self):
        df = self._enriched("restart_prior", 105.0, 0.0)
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("error")  # no warning expected
            n = apply_restart_tripwire(df)
        assert n == 0
        assert df.loc[0, "start_coord_source"] == "restart_prior"
```

- [ ] **Step 2: Run — expect FAIL (`apply_restart_tripwire` undefined)**

Run: `python -m pytest tests/tracking/test_restart_geometry.py::TestTripwire -v`
Expected: FAIL — ImportError.

- [ ] **Step 3: Implement the pure tripwire helper**

Add to `_gk_geometry.py` (ensure `import warnings` is present at the top):

```python
def _tripwire_key(t: int) -> str | None:
    if t == _GOALKICK:
        return "goalkick"
    if t == _PENALTY:
        return "penalty"
    if t in _CORNER_TYPES:
        return "corner"
    if t == _THROW_IN:
        return "throw_in"
    return None


def apply_restart_tripwire(out: pd.DataFrame) -> int:
    """Validate imputed restart ORIGIN coords against their Law region, IN PLACE on an enriched
    frame (as emitted by ``resolve_restart_geometry``). Imputed (non-``native``) coords that violate
    -> reverted to NaN, source ``tripwire_reverted``, confidence 0.0. Native violations warn only
    (provider truth, never reverted). Destinations are NOT guarded in Phase 1 (spec §6). Returns the
    reversion count. PURE policy step -- called by ``add_restart_coordinates``, never by the engine
    (so the frozen ``resolve_gk_geometry`` path stays silent + revert-free)."""
    tid = out["type_id"].to_numpy()
    sx = out["enriched_start_x"].to_numpy().copy()
    sy = out["enriched_start_y"].to_numpy().copy()
    ssrc = out["start_coord_source"].to_numpy().astype(object).copy()
    sconf = out["start_coord_confidence"].to_numpy().astype(float).copy()
    reverts = 0
    for i in range(len(out)):
        key = _tripwire_key(int(tid[i]))
        if key is None or not np.isfinite(sx[i]):
            continue
        if _TRIPWIRE[key](sx[i], sy[i]):
            continue  # in-region
        if ssrc[i] == "native":
            warnings.warn(
                f"add_restart_coordinates: native {key} origin ({sx[i]:.1f},{sy[i]:.1f}) outside "
                f"its Law region (data-quality signal; not reverted).",
                stacklevel=2,
            )
            continue
        warnings.warn(
            f"add_restart_coordinates: imputed {key} origin ({sx[i]:.1f},{sy[i]:.1f}) outside its "
            f"Law region; reverted to unresolved.",
            stacklevel=2,
        )
        sx[i] = sy[i] = np.nan
        ssrc[i], sconf[i] = "tripwire_reverted", 0.0
        reverts += 1
    out["enriched_start_x"], out["enriched_start_y"] = sx, sy
    out["start_coord_source"], out["start_coord_confidence"] = ssrc, sconf
    return reverts
```

- [ ] **Step 4: Run — expect PASS**

Run: `python -m pytest tests/tracking/test_restart_geometry.py::TestTripwire -v`
Expected: PASS (all 3).

- [ ] **Step 5: Commit checkpoint** (staged)

```bash
git add silly_kicks/tracking/_gk_geometry.py tests/tracking/test_restart_geometry.py
git commit -m "feat(tracking): apply_restart_tripwire pure edge helper (revert->tripwire_reverted)"
```

---

## Task 5: Convert `resolve_gk_geometry` to the delegation shim (parity-critical)

**Files:**
- Modify: `silly_kicks/tracking/_gk_geometry.py`
- Test: `tests/tracking/test_gk_geometry.py` (Task-1 tests MUST stay green)

- [ ] **Step 1: Replace the body of `resolve_gk_geometry` with the shim**

Replace the existing `resolve_gk_geometry` implementation (keep the signature + docstring intent) with:

```python
def resolve_gk_geometry(
    actions: pd.DataFrame, *, frames: pd.DataFrame | None, links: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Goal-kick coordinate derivation (the frozen pre-promotion contract). Thin delegation to
    :func:`resolve_restart_geometry` with ``impute_types=(goalkick,)`` -- so the engine imputes
    GOAL-KICKS ONLY (non-goalkick rows are never imputed -> no revert needed) and runs no tripwire
    (pure engine). The shim then renames to the legacy columns, drops the dest-confidence column,
    and maps ``restart_prior`` -> ``goalkick_prior``. ``actions`` is never mutated. Public API; do
    NOT change the output contract (4 internal callers + the xT-GK completion path depend on it
    byte-for-byte)."""
    g = resolve_restart_geometry(actions, frames=frames, links=links, impute_types=(_GOALKICK,))

    # Whole-array numpy (no .loc-mask assignment -> index-independent, matches the original style).
    osrc = g["start_coord_source"].to_numpy().astype(object).copy()
    # label map: restart_prior -> goalkick_prior (goalkick rule-point). tracking_gk / native /
    # next_event / unresolved pass through unchanged. NO tracking_ball->tracking_gk mapping.
    osrc = np.where(osrc == "restart_prior", "goalkick_prior", osrc)

    out = pd.DataFrame(index=actions.index)
    out["origin_x"] = g["enriched_start_x"].to_numpy()
    out["origin_y"] = g["enriched_start_y"].to_numpy()
    out["origin_source"] = osrc
    out["origin_confidence"] = g["start_coord_confidence"].to_numpy()
    out["dest_x"] = g["enriched_end_x"].to_numpy()
    out["dest_y"] = g["enriched_end_y"].to_numpy()
    out["dest_source"] = g["end_coord_source"].to_numpy()
    # DROP end_coord_confidence -- the frozen contract has origin_confidence only.
    return out[["origin_x", "origin_y", "origin_source", "origin_confidence",
                "dest_x", "dest_y", "dest_source"]]
```

> Why no revert step: with `impute_types=(goalkick,)` the engine leaves every non-goalkick row at
> native-or-unresolved (it never imputes them), so the shim needs no post-hoc revert. For goal-kick
> rows the engine == the frozen contract by construction (same `_tracking_gk_xy` clamp, same
> `_next_event_start`, same rule-point + `_CONF` values; `tracking_ball` gated off for goal-kicks).

- [ ] **Step 2: Add the shim-stays-silent test (Major-1 guard) + run the full frozen suite**

Append to `tests/tracking/test_gk_geometry.py`:

```python
class TestShimNoTripwireLeak:
    def test_resolve_gk_geometry_emits_no_warning_on_out_of_region_native(self):
        # The engine is pure; the shim never tripwires -> a native-out-of-region goalkick is SILENT
        # through resolve_gk_geometry (the frozen contract emitted no warnings).
        import warnings as _w
        a = _actions(start_x=[80.0, 5.0], start_y=[34.0, 34.0])  # row0 goalkick native x=80 (out of area)
        with _w.catch_warnings():
            _w.simplefilter("error")  # any warning -> test failure
            g = resolve_gk_geometry(a, frames=None)
        assert g.loc[0, "origin_source"] == "native"
        assert g.loc[0, "origin_x"] == pytest.approx(80.0)
```

Run: `python -m pytest tests/tracking/test_gk_geometry.py -v`
Expected: PASS (all original + Task-1 frozen-contract tests + the new no-warning test). If
`test_offposition_gk_goalkick_falls_to_rule_point`, `test_goalkick_no_native_end_no_next_event_unresolved`,
`test_nongoalkick_throwin_not_imputed`, the column-set test, or the no-warning test fails → the shim
diverges; debug before continuing (this is the gate that makes consolidation safe).

- [ ] **Step 3: Assert the committed GOLDEN snapshot (from Task 1) is frame-equal**

Run: `python -m pytest tests/tracking/test_gk_geometry.py::TestGoldenSnapshot -v`
Expected: PASS — `pd.testing.assert_frame_equal(resolve_gk_geometry(fixture), golden)` for both the
frames=None and frames-supplied fixtures (the golden was captured pre-refactor in Task 1).

- [ ] **Step 4: Run the completion-model + xT-GK suites — expect PASS (transitive parity)**

The completion-path guarantee is **transitive, not "hope the old suites are thorough"**: the Task-1
golden pins `resolve_gk_geometry`'s output byte-identical (Step 3), and the completion features +
`compute_xt_gk` are a **pure function of that resolver output** — so identical resolver output ⇒
identical completion/xT-GK output, necessarily. The golden's `_multi()` fixture exercises the
`tracking_gk` / `restart_prior` / `next_event` tiers the density path actually consumes, so the
coverage is real. These suites are the executable confirmation of that transitive guarantee.

Run: `python -m pytest tests/tracking/test_gk_completion.py tests/tracking/test_xt_gk.py tests/tracking/test_xt_gk_e2e.py -m "not e2e" -v`
Expected: PASS. (The e2e is owner-gated; the non-e2e portion must pass.)

- [ ] **Step 5: Commit checkpoint** (staged)

```bash
git add silly_kicks/tracking/_gk_geometry.py tests/tracking/test_gk_geometry.py
git commit -m "refactor(tracking): resolve_gk_geometry delegates to general engine (impute_types goalkick-only)"
```

---

## Task 6: `RestartCoordinateReport` (RED→GREEN)

**Files:**
- Create: `silly_kicks/tracking/_restart_report.py`
- Test: `tests/tracking/test_restart_geometry.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
from silly_kicks.tracking._restart_report import RestartCoordinateReport  # noqa: E402


class TestReport:
    def test_from_frame_counts_match_value_counts(self):
        g = resolve_restart_geometry(
            pd.concat([_restart(_GK), _restart(_PEN)], ignore_index=True).assign(
                action_id=range(4), game_id=9, period_id=1),
            frames=None)
        rep = RestartCoordinateReport.from_frame(g)
        assert rep.n_rows == 4
        assert rep.start_source_counts == dict(g["start_coord_source"].value_counts())

    def test_n_tripwire_reversions_counts_tagged_rows(self):
        # frame with one tripwire_reverted row -> report surfaces it (Medium-4)
        g = pd.DataFrame({
            "start_coord_source": ["restart_prior", "tripwire_reverted", "native"],
            "end_coord_source": ["next_event", "native", "native"],
        })
        rep = RestartCoordinateReport.from_frame(g)
        assert rep.n_tripwire_reversions == 1
```

- [ ] **Step 2: Run — expect FAIL (module missing)**

Run: `python -m pytest tests/tracking/test_restart_geometry.py::TestReport -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement the report**

Create `silly_kicks/tracking/_restart_report.py`:

```python
"""Aggregate provenance QA for add_restart_coordinates output (mirrors XtGkReport / LinkReport)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RestartCoordinateReport:
    """Counts per origin/destination source for a restart-coordinate-enriched frame. A convenience
    over a downstream ``GROUP BY start_coord_source`` (not load-bearing). By construction the counts
    equal the columns' ``value_counts``."""

    n_rows: int
    start_source_counts: dict[str, int]
    end_source_counts: dict[str, int]
    n_tripwire_reversions: int  # rows the tripwire reverted (start_coord_source == "tripwire_reverted")

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> "RestartCoordinateReport":
        """Build the report from an ``add_restart_coordinates`` / ``resolve_restart_geometry`` frame.
        ``n_tripwire_reversions`` preserves the QA distinction between never-resolvable
        (``unresolved``) and resolved-then-reverted (``tripwire_reverted``) rows (spec §6, Medium-4).

        Examples
        --------
        >>> rep = RestartCoordinateReport.from_frame(enriched)
        >>> rep.start_source_counts["restart_prior"]  # doctest: +SKIP
        12
        """
        ssc = dict(df["start_coord_source"].value_counts(dropna=True))
        return cls(
            n_rows=len(df),
            start_source_counts=ssc,
            end_source_counts=dict(df["end_coord_source"].value_counts(dropna=True)),
            n_tripwire_reversions=int(ssc.get("tripwire_reverted", 0)),
        )
```

- [ ] **Step 4: Run — expect PASS**

Run: `python -m pytest tests/tracking/test_restart_geometry.py::TestReport -v`
Expected: PASS.

- [ ] **Step 5: Commit checkpoint** (staged)

```bash
git add silly_kicks/tracking/_restart_report.py tests/tracking/test_restart_geometry.py
git commit -m "feat(tracking): RestartCoordinateReport provenance QA dataclass"
```

---

## Task 7: Public `add_restart_coordinates` (RED→GREEN)

**Files:**
- Modify: `silly_kicks/spadl/utils.py`
- Test: `tests/spadl/test_add_restart_coordinates.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/spadl/test_add_restart_coordinates.py`:

```python
import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl.utils import add_restart_coordinates

_GK = 22


def _actions():
    return pd.DataFrame(dict(
        game_id=[9, 9], period_id=[1, 1], action_id=[1, 0], team_id=[1, 1], player_id=[10, 10],
        type_id=[_GK, 0], time_seconds=[6.0, 5.0],
        start_x=[np.nan, 50.0], start_y=[np.nan, 30.0], end_x=[60.0, 55.0], end_y=[30.0, 30.0],
    ))


def test_emits_enriched_columns_and_does_not_mutate_canonical():
    a = _actions()
    before = a["start_x"].copy()
    out = add_restart_coordinates(a, frames=None)
    assert {"enriched_start_x", "start_coord_source", "start_coord_confidence",
            "enriched_end_x", "end_coord_source", "end_coord_confidence"} <= set(out.columns)
    # canonical start_x untouched on the original frame
    pd.testing.assert_series_equal(a["start_x"], before)


def test_sorts_by_game_period_action():
    out = add_restart_coordinates(_actions(), frames=None)
    assert list(out["action_id"]) == [0, 1]  # sorted


def test_goalkick_origin_imputed_events_only():
    out = add_restart_coordinates(_actions(), frames=None)
    gk = out[out["type_id"] == _GK].iloc[0]
    assert gk["start_coord_source"] == "restart_prior"
    assert gk["enriched_start_x"] == pytest.approx(5.5)


def test_nan_identifier_safe():
    a = _actions()
    a.loc[0, "player_id"] = np.nan
    out = add_restart_coordinates(a, frames=None)  # must not raise
    assert len(out) == 2
```

- [ ] **Step 2: Run — expect FAIL (function missing)**

Run: `python -m pytest tests/spadl/test_add_restart_coordinates.py -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement `add_restart_coordinates`**

Add to `silly_kicks/spadl/utils.py` (after `add_pre_shot_gk_context`):

```python
_ADD_RESTART_COORDS_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "game_id", "period_id", "action_id", "type_id", "start_x", "start_y", "end_x", "end_y",
)


@nan_safe_enrichment
def add_restart_coordinates(
    actions: pd.DataFrame, *, frames: pd.DataFrame | None = None, links: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Impute missing restart coordinates as new, provenance-tagged columns (Phase 1, additive).

    Derives origin/destination coordinates for restart actions whose native coordinate is NaN
    (e.g. ~60% of Gradient Sports goal-kicks). Law-fixed-spot restarts (goal-kick, penalty, corner,
    throw-in) get a geometric rule-point tier; all rows also get tracking-ball / next-event tiers.
    Canonical ``start_x/start_y/end_x/end_y`` are **never mutated** — the imputed values land in new
    ``enriched_*`` columns with per-row ``*_coord_source`` + ``*_coord_confidence`` provenance.

    When ``frames`` is supplied, the tracking-ball / in-area tracking-GK tiers are enabled (higher
    confidence); with ``frames=None`` (events-only) only native / rule-point / next-event tiers run.
    NaN identifiers route to the documented per-row default (ADR-003).

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL action stream. Requires ``game_id``, ``period_id``, ``action_id``, ``type_id``,
        ``start_x``, ``start_y``, ``end_x``, ``end_y``.
    frames : pd.DataFrame | None, default None
        Long-form tracking frames (``TRACKING_FRAMES_COLUMNS``). Enables tracking tiers.
    links : pd.DataFrame | None, default None
        Pre-computed action->frame pointers (skips internal linking).

    Returns
    -------
    pd.DataFrame
        Sorted copy of ``actions`` (by ``game_id, period_id, action_id``) with 8 appended columns:
        ``enriched_start_x/_y``, ``start_coord_source``, ``start_coord_confidence``,
        ``enriched_end_x/_y``, ``end_coord_source``, ``end_coord_confidence``.

    Raises
    ------
    ValueError
        If a required column is missing.

    Examples
    --------
    Impute goal-kick origins (events-only) and keep only high-confidence positions::

        actions, _ = gradientsports.convert_to_actions(events, home_team_id=100)
        enriched = add_restart_coordinates(actions)
        confident = enriched[enriched["start_coord_confidence"] >= 0.7]
    """
    missing = [c for c in _ADD_RESTART_COORDS_REQUIRED_COLUMNS if c not in actions.columns]
    if missing:
        raise ValueError(
            f"add_restart_coordinates: actions missing required columns: {sorted(missing)}. "
            f"Got: {sorted(actions.columns)}"
        )
    sorted_actions = actions.sort_values(
        ["game_id", "period_id", "action_id"], kind="mergesort"
    ).reset_index(drop=True)

    from silly_kicks.tracking._gk_geometry import apply_restart_tripwire, resolve_restart_geometry

    geom = resolve_restart_geometry(sorted_actions, frames=frames, links=links)  # PURE: no tripwire
    geom["type_id"] = sorted_actions["type_id"].to_numpy()  # apply_restart_tripwire needs type_id
    apply_restart_tripwire(geom)  # feature-policy step at the EDGE (spec §6); mutates geom in place
    for col in (
        "enriched_start_x", "enriched_start_y", "start_coord_source", "start_coord_confidence",
        "enriched_end_x", "enriched_end_y", "end_coord_source", "end_coord_confidence",
    ):
        sorted_actions[col] = geom[col].to_numpy()
    return sorted_actions
```

The tripwire runs HERE (the edge), not in the engine — so the frozen `resolve_gk_geometry` path
(which calls the pure engine) never tripwires (spec §6, Major-1). The `RestartCoordinateReport` (Task
6) is built by the caller via `RestartCoordinateReport.from_frame(out)` when they want the QA tally
(it reads the `tripwire_reverted` tag from the emitted frame).

- [ ] **Step 4: Add the edge-tripwire test + run**

Append to `tests/spadl/test_add_restart_coordinates.py`:

```python
_CORNER_C = 5


def test_tripwire_reverts_bad_imputed_corner_at_edge():
    # corner with NaN origin, native end at midfield-ish -> side picks a corner, but force an
    # out-of-region imputed coord via frames ball at midfield; tripwire (edge) reverts + tags.
    a = pd.DataFrame(dict(
        game_id=[9], period_id=[1], action_id=[0], team_id=[1], player_id=[10],
        type_id=[_CORNER_C], time_seconds=[5.0],
        start_x=[np.nan], start_y=[np.nan], end_x=[np.nan], end_y=[np.nan]))
    frames = pd.DataFrame(dict(
        game_id=[9], period_id=[1], frame_id=[1250], time_seconds=[5.0], team_id=[0],
        player_id=[-1], is_goalkeeper=[False], is_ball=[True], x=[50.0], y=[20.0],
        source_provider=["gradientsports"]))
    with pytest.warns(UserWarning):
        out = add_restart_coordinates(a, frames=frames)
    assert out.loc[0, "start_coord_source"] == "tripwire_reverted"
    assert np.isnan(out.loc[0, "enriched_start_x"])
```

- [ ] **Step 5: Run — expect PASS**

Run: `python -m pytest tests/spadl/test_add_restart_coordinates.py -v`
Expected: PASS (all 5).

- [ ] **Step 6: Commit checkpoint** (staged)

```bash
git add silly_kicks/spadl/utils.py tests/spadl/test_add_restart_coordinates.py
git commit -m "feat(spadl): add_restart_coordinates public enrichment (frames-optional, edge tripwire)"
```

---

## Task 8: Exports

**Files:**
- Modify: `silly_kicks/tracking/__init__.py`
- Modify: `silly_kicks/spadl/__init__.py`

- [ ] **Step 1: Export from tracking**

In `silly_kicks/tracking/__init__.py`:
- Add `"RestartCoordinateReport"` and `"resolve_restart_geometry"` to `__all__` (alphabetical: `RestartCoordinateReport` near the class exports; `resolve_restart_geometry` right after `"resolve_gk_geometry"` at line ~193).
- Change the import at line 234 to: `from ._gk_geometry import resolve_gk_geometry, resolve_restart_geometry`
- Add: `from ._restart_report import RestartCoordinateReport`

- [ ] **Step 2: Export from spadl**

In `silly_kicks/spadl/__init__.py`:
- Add `"add_restart_coordinates"` to `__all__` (after `"add_pre_shot_gk_context"`, line ~22).
- Add `add_restart_coordinates,` to the `from .utils import (...)` block (line ~64, after `add_pre_shot_gk_context,`).

- [ ] **Step 3: Verify imports resolve**

Run: `python -c "import silly_kicks.tracking as t; import silly_kicks.spadl as s; print(t.resolve_restart_geometry, t.RestartCoordinateReport, s.add_restart_coordinates)"`
Expected: prints three callables, no ImportError.

- [ ] **Step 4: Commit checkpoint** (staged)

```bash
git add silly_kicks/tracking/__init__.py silly_kicks/spadl/__init__.py
git commit -m "feat: export resolve_restart_geometry, RestartCoordinateReport, add_restart_coordinates"
```

---

## Task 9: Cross-cutting CI gates

**Files:** (tests only; fixes flow back into Task 3/7 code if a gate fails)

- [ ] **Step 1: NaN-safety gate (ADR-003 auto-discovery)**

Run: `python -m pytest tests/test_enrichment_nan_safety.py -v`
Expected: PASS — `add_restart_coordinates` is auto-discovered (it is `@nan_safe_enrichment`-decorated) and survives NaN identifiers. If it fails, the body must route NaN-id rows to the per-row default (the resolver already NaN-tolerates; ensure no `.astype(int)` on NaN ids).

- [ ] **Step 2: id-dtype invariance gate (ADR-019)**

Run: `python -m pytest tests/tracking/test_id_compat_lint.py -v` and `python -m pytest -k "id_dtype" -v`
Expected: PASS — the new `_tracking_ball_xy` uses `resolve_frame_ids_by_position` + `is_ball.astype(bool)` (no raw `==` on ids). If the AST lint flags `_gk_geometry.py`, route any id comparison through `_id_compat`.

- [ ] **Step 3: Public-API Examples docstring gate**

Run: `python -m pytest tests/test_public_api_examples.py -v`
Expected: PASS — `add_restart_coordinates` (in `spadl/utils.py`, an enumerated module) has an `Examples` section. `resolve_restart_geometry` (in underscore module `_gk_geometry.py`) is not gate-required.

- [ ] **Step 4: Full tracking + spadl suite (no regressions)**

Run: `python -m pytest tests/tracking/ tests/spadl/ -m "not e2e" --tb=short -q`
Expected: PASS. Read the actual summary line (`N passed`), not a piped tail.

- [ ] **Step 5: Commit checkpoint** (staged, if any test files changed)

```bash
git add -A && git commit -m "test: restart-coordinate enrichment passes nan-safety/id-dtype/examples gates"
```

---

## Task 10: Full xT-GK + completion parity (red-first capture)

**Files:**
- Create: `tests/tracking/test_restart_xtgk_parity.py`

- [ ] **Step 1: (No separate baseline needed.)**

The byte-identical guard is the **Task-1 committed golden** (captured on unmodified code). These
parity tests are **supplementary** — they assert the shim-vs-engine type-gating contract directly
(throw-in not imputed in the shim; goal-kick legacy labels/columns), complementing the golden.

- [ ] **Step 2: Write the parity test**

Create `tests/tracking/test_restart_xtgk_parity.py`:

```python
"""Goal-kick parity: the resolve_restart_geometry promotion must not change any
resolve_gk_geometry consumer's output (xT-GK + completion). Spec §7."""
import numpy as np
import pandas as pd

from silly_kicks.tracking._gk_geometry import resolve_gk_geometry, resolve_restart_geometry

_GK, _THROW = 22, 2


def _mixed():
    return pd.DataFrame(dict(
        game_id=[9, 9, 9], period_id=[1, 1, 1], action_id=[0, 1, 2], team_id=[1, 1, 1],
        player_id=[10, 11, 10], type_id=[_GK, _THROW, _GK], time_seconds=[5.0, 6.0, 70.0],
        start_x=[np.nan, np.nan, np.nan], start_y=[np.nan, np.nan, np.nan],
        end_x=[60.0, 40.0, np.nan], end_y=[30.0, 20.0, np.nan],
    ))


def test_throwin_not_imputed_in_shim_but_imputed_in_general():
    a = _mixed()
    legacy = resolve_gk_geometry(a, frames=None)  # engine runs impute_types=(goalkick,)
    general = resolve_restart_geometry(a, frames=None)  # default impute_types -> all
    # row 1 is a throw_in: the shim's goalkick-only impute_types means it is NEVER imputed
    # (NaN origin -> unresolved), so no revert step is needed.
    assert legacy.loc[1, "origin_source"] == "unresolved"
    assert np.isnan(legacy.loc[1, "origin_x"])
    # general imputes it (next_event dest at minimum; origin unresolved here w/o side -> ok)
    assert general.loc[1, "end_coord_source"] == "next_event"


def test_goalkick_legacy_labels_and_columns():
    legacy = resolve_gk_geometry(_mixed(), frames=None)
    assert set(legacy.columns) == {
        "origin_x", "origin_y", "origin_source", "origin_confidence",
        "dest_x", "dest_y", "dest_source"}
    # goalkick row 0 -> rule-point labeled goalkick_prior (not restart_prior)
    assert legacy.loc[0, "origin_source"] == "goalkick_prior"
    # goalkick row 2 -> NaN end, no in-period next event (last row) -> unresolved (Major-2b)
    assert legacy.loc[2, "dest_source"] == "unresolved"
```

- [ ] **Step 3: Run — expect PASS**

Run: `python -m pytest tests/tracking/test_restart_xtgk_parity.py -v`
Expected: PASS.

- [ ] **Step 4: Owner-gated GS e2e (manual / owner machine)**

Document in the test file (skip-marked `@pytest.mark.e2e`) that the owner runs the real WC2022 GS xT-GK + `compute_gk_completion` / `add_gk_completion` outputs pre/post and asserts unchanged values on native-origin rows, plus the new `add_restart_coordinates` source distribution sanity. Not a CI gate (GS not in CI).

- [ ] **Step 5: Commit checkpoint** (staged)

```bash
git add tests/tracking/test_restart_xtgk_parity.py
git commit -m "test(tracking): xT-GK/completion goal-kick parity for the restart-geometry promotion"
```

---

## Task 11: ADR, CLAUDE.md, version bump, final review, single commit

**Files:**
- Create: `docs/superpowers/adrs/ADR-0NN-restart-coordinate-enrichment.md` (next free number, reconciled vs `origin/main`)
- Modify: `CLAUDE.md` (add the `PR-S## ships …` line + the new spadl helper)
- Modify: `TODO.md` (mark the general-enrichment item done; note Phase 2 deferred)
- Modify: `CHANGELOG.md`, `pyproject.toml`, `silly_kicks/__init__.py`, `uv.lock` (version bump)

- [ ] **Step 1: Determine the next free version + ADR number**

Run: `git fetch origin && git tag --sort=-v:refname | head -3 && ls docs/superpowers/adrs/ | tail -3`
Pick the next free patch/minor after the latest tag (new public feature → minor bump) and the next free ADR number. Per memory: nothing is pre-reserved; reconcile now.

- [ ] **Step 2: Write the ADR**

Create the ADR documenting: the additive-now / canonical-later phasing; the per-type prior geometry + tripwire; the provenance contract — `enriched_*`, `*_coord_source` (**enumerate all 7 released values**: `native` / `tracking_ball` / `tracking_gk` / `restart_prior` / `next_event` / `unresolved` / `tripwire_reverted`, the last origin-only), `*_coord_confidence`; the single-engine consolidation + the goal-kick delegation shim (`impute_types=(goalkick,)` goalkick-only imputation, label map, dropped dest-confidence; tripwire at the edge, not the engine); and the **Phase-2 promotion recipe** (copy enriched→canonical, retrain VAEP/xT/calibration, re-baseline goldens, promote tripwire to a hard gate; **note `enriched_*` is NaN for `unresolved`/`tripwire_reverted` rows — those had NaN native anyway, so a future apply-PR must not assume `enriched_*` is always finite**). Reference the live-probe scope evidence.

- [ ] **Step 3: Update CLAUDE.md + TODO.md**

CLAUDE.md: add a `PR-S## ships …` line under the spadl section describing `add_restart_coordinates`. Mention it is Phase 1 (additive, no retrain) and `resolve_gk_geometry` is now a shim over `resolve_restart_geometry`.
TODO.md: mark the "Goal-kick / event coordinate enrichment — GENERAL" item as Phase-1 shipped; log Phase 2 (canonical promotion + coordinated retrain) as the remaining follow-up.

- [ ] **Step 4: Version bump (5 sites + lock)**

Per `reference_version_bump_checklist`: `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, `TODO.md` header, then `uv lock`.

- [ ] **Step 5: C4 check (expect C4-free)**

Confirm: no new KDE backend / trained model / tracking aggregator. `add_restart_coordinates` is a spadl helper (NOT in `tracking.__all__`'s `add_*` set), so the tracking aggregator count is unchanged. Verify `len([n for n in tracking.__all__ if n.startswith('add_')]) - 1` is unchanged → skip C4 regen.

Run: `python -c "import silly_kicks.tracking as t; print(len([n for n in t.__all__ if n.startswith('add_')]) - 1)"`
Expected: the same count as before this PR (no new tracking `add_*`).

- [ ] **Step 6: Replicate the full CI lint + type + test locally**

Run (whole tree, per house rule):
```
ruff check silly_kicks/ tests/ scripts/
ruff format --check silly_kicks/ tests/ scripts/
pyright silly_kicks/ tests/
python -m pytest tests/ -m "not e2e" -q
```
Expected: all clean. Fix any N806/format/type issues inline (e.g. `# type: ignore[arg-type]` idioms for `float(df.loc[...])`).

- [ ] **Step 7: Final review + single commit (REQUIRES EXPLICIT APPROVAL)**

Squash the staged checkpoints into ONE feature commit on the feature branch (spec + plan + ADR + code + tests + version bump bundled). HOLD for the user's explicit per-commit approval + sentinel before committing; then open the PR on request. Do NOT push or create the PR without approval.

---

## Self-Review

**Spec coverage:** §2 scope → Task 3 (rule-point types) + Task 7 (events-only). §3 placement/shim/`impute_types` → Tasks 5, 7. §4 tier model + goal-kick invariant → Task 3 (type-gating) + Task 1/5. §5 output contract (incl. `tripwire_reverted`) → Task 3/7 columns + Task 6 report. §6 tripwire-at-the-edge → Task 4 (pure helper) + Task 7 (applied in `add_restart_coordinates`). §7 testing → Tasks 1,3,4,9,10. §8 ADR/C4/docs → Task 11. All sections covered.

**Round-3 review fixes incorporated:** Major-1 (tripwire leaked onto frozen path) → tripwire moved to the `add_restart_coordinates` edge; the engine is pure; Task-5 adds a shim-emits-no-warning test. Major-2 (wasted `_tracking_ball_xy` on the frozen hot path) + Medium-5 (fragile `.loc` revert) → engine gains `impute_types`; the shim passes `(goalkick,)` so the engine does zero non-goalkick work and needs no revert. Medium-3 (baseline) → Task 1 captures a committed golden snapshot asserted frame-equal in Task 5. Medium-4 (lost reversion tally) → reverted rows tagged `tripwire_reverted`; report gains `n_tripwire_reversions`. Minor-6 (dest unguarded) stated in §6. Minor-8 (revert semantics) locked to revert-to-unresolved in spec §6. Minor-9 (fixture linkage) → fid-finite assert in Task 2.

**Placeholder scan:** no TBD/TODO in code steps; every code step shows complete code. The revert-semantics decision is now LOCKED (revert-to-unresolved, spec §6) — no open decision points remain.

**Type consistency:** `resolve_restart_geometry` emits `enriched_*` / `*_coord_source` / `*_coord_confidence` consistently across Tasks 3/5/7; the shim maps to `origin_*` / `dest_*` in Task 5; `_tracking_ball_xy(actions, frames, links)` matches `_tracking_gk_xy` and is called identically in Tasks 2/3; `apply_restart_tripwire(out)` (Task 4) is called only by `add_restart_coordinates` (Task 7); `impute_types` is `tuple[int,...]|None` in Tasks 3/5.

**Sequencing:** the byte-identical guard is the **Task-1 committed golden snapshot** (captured on unmodified code) + the pre-existing `test_xt_gk.py`/`test_gk_completion.py` suites staying green at Task 5 Step 4. Task 10's tests are supplementary (post-refactor), not the baseline — the golden is the real guard (Medium-3).
