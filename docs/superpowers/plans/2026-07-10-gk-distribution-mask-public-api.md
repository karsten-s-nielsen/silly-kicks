# Public `gk_distribution_mask` API + ρ loader cleanup — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export the canonical GK-distribution domain logic as a public, frame-optional `tracking.gk_distribution_mask` (with a `native`/`robust` lever), reduce the frozen v1 `_gk_distribution_mask` to a byte-identical shim over it, and correct the ρ retention loader's dropped-column reference (`gk_was_distributing` → self-adapting `is_gk_distribution`).

**Architecture:** New public resolver lives in the non-frozen `silly_kicks/tracking/_gk_resolve.py` beside its sibling `acting_gk_from_frames`. All boolean logic runs on positional numpy arrays in `actions` row-order, wrapped with `actions.index` once (sidesteps `ids_equal`'s POSITIONAL RangeIndex return). The frozen `_xt_gk._gk_distribution_mask` delegates to it in `native` mode (golden-gated byte-identity). The ρ loader/trainer swap the dropped-column reference for a self-adapting `is_gk_distribution` read with explicit NULL→False coalescing.

**Tech Stack:** Python, pandas/numpy, pytest. No new dependencies. Governing: ADR-036 (xT-GK v2), ADR-007 (GK id), ADR-019 (dtype-safe ids), M5 (v1 freeze).

**Spec:** `docs/superpowers/specs/2026-07-10-gk-distribution-mask-public-api-design.md`

**Commit policy (this repo):** ONE squashed commit per branch, created only after explicit user approval (Task 8). Do NOT commit per-task — the per-task "verify" steps stand in for commits. Branch `pr-s110-gk-distribution-mask-public-api` already exists and holds the spec + this plan (uncommitted).

---

## File Structure

- **Create** `tests/tracking/test_gk_distribution_mask.py` — all public-function tests + the golden byte-identity shim gate.
- **Create** `tests/xtgk/test_retention_loader_domain.py` — pure `should_select_is_gk_distribution` + trainer domain (present-nonnull / present-null / absent) + loader-drop-safety guard.
- **Modify** `silly_kicks/tracking/_gk_resolve.py` — add public `gk_distribution_mask` + private `_native_actor_is_gk`; add module constants + imports.
- **Modify** `silly_kicks/tracking/__init__.py` — export `gk_distribution_mask` (`__all__` + import block).
- **Modify** `silly_kicks/tracking/_xt_gk.py` (FROZEN) — `_gk_distribution_mask` becomes the shim.
- **Modify** `scripts/_loader_databricks.py` — pure `should_select_is_gk_distribution`, catalog-qualified probe, `_build_retention_sql(include_is_gk_distribution)`, drop `gk_was_distributing`.
- **Modify** `scripts/train_gk_retention.py` — `prepare_retention_training_data` domain uses `is_gk_distribution`.
- **Modify** docs: `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, `TODO.md`, `CLAUDE.md`, `docs/superpowers/adrs/ADR-036-*.md`, `uv.lock`, `docs/c4/architecture.{dsl,html}` (regen — count stays 28).

---

## Task 1: Public `gk_distribution_mask` + `_native_actor_is_gk`

**Files:**
- Modify: `silly_kicks/tracking/_gk_resolve.py`
- Test: `tests/tracking/test_gk_distribution_mask.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/tracking/test_gk_distribution_mask.py`:

```python
"""Public gk_distribution_mask: semantics, native/robust lever, dtype/NaN/index safety."""

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import gk_distribution_mask

_GOALKICK, _PASS, _THROW_IN, _SHOT = 22, 0, 2, 11  # spadlconfig.actiontype_id


def _frow(pid, team, gk, t, *, is_ball=False, x=50.0, y=34.0):
    return dict(
        game_id=1, period_id=1, frame_id=round(t * 25), time_seconds=t, frame_rate=25.0,
        player_id=pid, team_id=team, is_ball=is_ball, is_goalkeeper=gk, x=float(x), y=float(y),
        z=0.0, speed=1.0, vx=0.0, vy=0.0, speed_source="native", ball_state="alive",
        team_attacking_direction="ltr", confidence=None, visibility=None,
        source_provider="gradientsports", is_goalkeeper_source="native",
    )


def _frames(rows):
    f = pd.DataFrame(rows)
    f["player_id"] = f["player_id"].astype("Int64")
    f["team_id"] = f["team_id"].astype("Int64")
    return f


def _action(aid, type_id, player_id, team_id, t):
    return dict(game_id=1, action_id=aid, period_id=1, time_seconds=t,
                team_id=team_id, player_id=player_id, type_id=type_id,
                start_x=50.0, start_y=34.0, end_x=60.0, end_y=34.0, result_id=1)


# GK = player 1 (team 5) detected across the whole window; outfielder = player 10.
def _one_keeper_frames():
    rows = []
    for t in (9.9, 10.0, 19.9, 20.0):
        rows += [_frow(1, 5, True, t), _frow(10, 5, False, t, x=60.0),
                 _frow(2, 6, True, t, x=100.0), _frow(pd.NA, pd.NA, False, t, is_ball=True)]
    return _frames(rows)


def test_frames_none_is_goalkicks_only():
    actions = pd.DataFrame([
        _action(0, _GOALKICK, 1, 5, 10.0),
        _action(1, _PASS, 1, 5, 20.0),  # GK pass, but no frames -> cannot detect
    ])
    out = gk_distribution_mask(actions, None)
    assert isinstance(out, pd.Series) and out.dtype == bool
    assert out.tolist() == [True, False]


def test_full_mask_detects_gk_pass_and_goalkick():
    actions = pd.DataFrame([
        _action(0, _GOALKICK, 10, 5, 10.0),  # goalkick by NON-GK -> True (actor-independent)
        _action(1, _PASS, 1, 5, 20.0),        # GK pass -> True
        _action(2, _PASS, 10, 5, 20.0),       # outfielder pass -> False
    ])
    out = gk_distribution_mask(actions, _one_keeper_frames(), resolve_gk="robust")
    assert out.tolist() == [True, True, False]


def test_throw_in_by_gk_is_true_and_gk_shot_is_false():
    actions = pd.DataFrame([
        _action(0, _THROW_IN, 1, 5, 10.0),  # GK throw-in -> True
        _action(1, _SHOT, 1, 5, 20.0),      # GK shot -> False (not pass/throw_in)
    ])
    out = gk_distribution_mask(actions, _one_keeper_frames(), resolve_gk="robust")
    assert out.tolist() == [True, False]


def test_native_equals_robust_for_single_detected_keeper():
    actions = pd.DataFrame([
        _action(0, _PASS, 1, 5, 10.0), _action(1, _PASS, 10, 5, 20.0),
        _action(2, _GOALKICK, 10, 5, 10.0),
    ])
    f = _one_keeper_frames()
    nat = gk_distribution_mask(actions, f, resolve_gk="native")
    rob = gk_distribution_mask(actions, f, resolve_gk="robust")
    assert nat.tolist() == rob.tolist() == [True, False, True]


def test_substitution_native_over_includes_robust_tightens():
    # playerA(1) keeper early (t~10), playerB(9) keeper late (t~50); A off after the sub.
    rows = []
    for t in (9.9, 10.0):
        rows += [_frow(1, 5, True, t), _frow(9, 5, False, t, x=55.0), _frow(2, 6, True, t, x=100.0)]
    for t in (49.9, 50.0):
        rows += [_frow(9, 5, True, t), _frow(1, 5, False, t, x=55.0), _frow(2, 6, True, t, x=100.0)]
    f = _frames(rows)
    # A pass by the SUBSTITUTED-OFF keeper (player 1) AFTER the sub (t=50).
    actions = pd.DataFrame([_action(0, _PASS, 1, 5, 50.0)])
    nat = gk_distribution_mask(actions, f, resolve_gk="native")
    rob = gk_distribution_mask(actions, f, resolve_gk="robust")
    assert nat.tolist() == [True]   # global set still contains player 1 as a keeper
    assert rob.tolist() == [False]  # time-resolved acting keeper at t=50 is player 9
    # robust is the strict subset (tightens, never broadens):
    assert (rob.to_numpy() <= nat.to_numpy()).all()


def test_non_range_index_is_aligned():
    actions = pd.DataFrame([
        _action(0, _GOALKICK, 10, 5, 10.0), _action(1, _PASS, 1, 5, 20.0),
    ], index=[7, 42])  # non-default index (filtered per-match path)
    out = gk_distribution_mask(actions, _one_keeper_frames(), resolve_gk="robust")
    assert list(out.index) == [7, 42]
    assert out.loc[7] and out.loc[42]


def test_string_vs_numeric_ids_match_native():
    # actions carry string ids; frames Int64 -> the NATIVE path is dtype-safe throughout
    # (canonical_id_series on game/team/player, ADR-019) and must still match.
    # NB: robust's GK *resolution* depends on acting_gk_from_frames's team join (raw ==, same-provider
    # dtypes in practice) -- that's its contract, not gk_distribution_mask's, so cross-dtype is asserted
    # on native here; robust's player match via ids_equal is separately dtype-safe.
    actions = pd.DataFrame([_action(0, _PASS, "1", "5", 20.0)])
    out = gk_distribution_mask(actions, _one_keeper_frames(), resolve_gk="native")
    assert out.tolist() == [True]


def test_nan_player_or_team_not_in_scope():
    actions = pd.DataFrame([_action(0, _PASS, np.nan, 5, 20.0)])
    out = gk_distribution_mask(actions, _one_keeper_frames(), resolve_gk="robust")
    assert out.tolist() == [False]


def test_missing_required_column_raises():
    actions = pd.DataFrame([_action(0, _PASS, 1, 5, 20.0)]).drop(columns=["player_id"])
    with pytest.raises(ValueError, match="player_id"):
        gk_distribution_mask(actions, _one_keeper_frames())


def test_game_id_absent_key_shape_both_modes():
    f = _one_keeper_frames()
    actions = pd.DataFrame([_action(0, _PASS, 1, 5, 20.0)]).drop(columns=["game_id"])
    for mode in ("native", "robust"):
        out = gk_distribution_mask(actions, f, resolve_gk=mode)
        assert out.tolist() == [True], mode


def test_invalid_resolve_gk_raises():
    actions = pd.DataFrame([_action(0, _PASS, 1, 5, 20.0)])
    with pytest.raises(ValueError, match="resolve_gk"):
        gk_distribution_mask(actions, _one_keeper_frames(), resolve_gk="bogus")  # type: ignore[arg-type]
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/tracking/test_gk_distribution_mask.py -q`
Expected: FAIL — `ImportError: cannot import name 'gk_distribution_mask'`.

- [ ] **Step 3: Implement `gk_distribution_mask` + `_native_actor_is_gk`**

In `silly_kicks/tracking/_gk_resolve.py`, add to the imports (after the existing `from ._id_compat import ids_match`):

```python
import numpy.typing as npt

from silly_kicks.spadl import config as spadlconfig

from ._id_compat import canonical_id_series, ids_equal, ids_match

_GOALKICK = spadlconfig.actiontype_id["goalkick"]
_PASS = spadlconfig.actiontype_id["pass"]
_THROW_IN = spadlconfig.actiontype_id["throw_in"]
```

(Add `Literal` to the typing import at the top of the file: `from typing import Literal`.)

Then add, directly above `def _gk_from_frames_linked`:

```python
def gk_distribution_mask(
    actions: pd.DataFrame,
    frames: pd.DataFrame | None = None,
    *,
    resolve_gk: Literal["native", "robust"] = "robust",
    tolerance_seconds: float = 0.2,
) -> pd.Series:
    """Per-action boolean: is this a GK distribution? (goal-kick OR pass/throw-in by the acting GK).

    True for any ``goalkick`` (actor-independent), OR a ``pass``/``throw_in`` whose actor is the
    acting team's goalkeeper. Returns a bool ``pd.Series`` aligned to ``actions.index``.

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL actions. Required columns: ``type_id``, ``player_id``, ``team_id`` (and
        ``period_id``/``time_seconds``/``game_id`` used by the frame link when ``frames`` is given).
    frames : pd.DataFrame | None, default None
        Long-form tracking frames (TRACKING_FRAMES_COLUMNS). ``None`` -> goal-kicks-only (the GK
        open-play-pass term is undetectable without frames; both modes degrade to goal-kicks).
    resolve_gk : {"native", "robust"}, default "robust"
        ``"robust"`` resolves the acting GK per action via :func:`acting_gk_from_frames` (time-accurate,
        roster-identity fallback) -- the default and the resolver the lakehouse pins for its goal-kick
        taker override. ``"native"`` uses a global ``frames[is_goalkeeper]`` (game,team,player)
        set-membership (reproduces the frozen v1 mask byte-for-byte; used by the v1 shim). For the GK-pass
        term ``robust`` is a subset of ``native`` (it tightens stale/substituted keepers, never broadens).
    tolerance_seconds : float, default 0.2
        Frame-link tolerance passed to :func:`acting_gk_from_frames` (robust only).

    Notes
    -----
    Pure (never mutates ``actions``); dtype-safe id matching (ADR-019); NaN actor -> not in scope.
    See NOTICE for full bibliographic citations.
    """
    missing = [c for c in ("type_id", "player_id", "team_id") if c not in actions.columns]
    if missing:
        raise ValueError(f"gk_distribution_mask: actions missing required column(s) {missing}")

    type_id = actions["type_id"].to_numpy()
    is_goalkick = type_id == _GOALKICK
    is_open = np.isin(type_id, (_PASS, _THROW_IN))

    if frames is None:
        return pd.Series(is_goalkick, index=actions.index)

    if resolve_gk == "native":
        actor_is_gk = _native_actor_is_gk(actions, frames)
    elif resolve_gk == "robust":
        acting_gk = acting_gk_from_frames(actions, frames, tolerance_seconds=tolerance_seconds)
        actor_is_gk = ids_equal(actions["player_id"], acting_gk).to_numpy()
    else:
        raise ValueError(f"resolve_gk must be 'native' or 'robust', got {resolve_gk!r}")

    mask = is_goalkick | (is_open & actor_is_gk)
    return pd.Series(mask, index=actions.index)


def _native_actor_is_gk(actions: pd.DataFrame, frames: pd.DataFrame) -> npt.NDArray[np.bool_]:
    """Positional bool array: is each action's (game,team,player) in the frames' is_goalkeeper set?

    Byte-identical to the frozen v1 ``_gk_distribution_mask`` set-membership block (global over all
    frames, NOT the linked frame). dtype-safe via ``canonical_id_series`` (ADR-019).
    """
    gk = frames[frames["is_goalkeeper"].astype(bool) & (~frames["is_ball"].astype(bool))]
    keyed_by_game = "game_id" in actions.columns and "game_id" in frames.columns

    gk_team = canonical_id_series(gk["team_id"]).to_numpy()
    gk_player = canonical_id_series(gk["player_id"]).to_numpy()
    act_team = canonical_id_series(actions["team_id"]).to_numpy()
    act_player = canonical_id_series(actions["player_id"]).to_numpy()
    if keyed_by_game:
        gk_game = canonical_id_series(gk["game_id"]).to_numpy()
        act_game = canonical_id_series(actions["game_id"]).to_numpy()
        gk_set = set(zip(gk_game, gk_team, gk_player, strict=True))
        return np.array([(g, t, p) in gk_set for g, t, p in zip(act_game, act_team, act_player, strict=True)])
    gk_set = set(zip(gk_team, gk_player, strict=True))
    return np.array([(t, p) in gk_set for t, p in zip(act_team, act_player, strict=True)])
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/tracking/test_gk_distribution_mask.py -q`
Expected: PASS (all ~11 tests). If `test_string_vs_numeric_ids_match` fails, confirm `_native`/`ids_equal` route through `canonical_id_series` (they do) — a real fail means the fixture ids weren't canonicalized.

---

## Task 2: Export `gk_distribution_mask` from the tracking package

**Files:**
- Modify: `silly_kicks/tracking/__init__.py:69` (`__all__`) and `:298` (import block)
- Test: `tests/tracking/test_gk_distribution_mask.py` (import already exercises it)

- [ ] **Step 1: Write the failing test**

Append to `tests/tracking/test_gk_distribution_mask.py`:

```python
def test_public_export_and_docstring_example():
    import silly_kicks.tracking as T

    assert "gk_distribution_mask" in T.__all__
    assert T.gk_distribution_mask is gk_distribution_mask
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/tracking/test_gk_distribution_mask.py::test_public_export_and_docstring_example -q`
Expected: FAIL — `gk_distribution_mask` not in `__all__` (the top-level import in the test file works because it resolves the name transitively; this asserts the curated export).

- [ ] **Step 3: Add the export**

In `silly_kicks/tracking/__init__.py`, in the `__all__` list next to `"acting_gk_from_frames"` (line ~69) add:

```python
    "gk_distribution_mask",
```

In the import block next to `acting_gk_from_frames` (line ~298):

```python
from ._gk_resolve import (
    acting_gk_from_frames,
    gk_distribution_mask,
    ...
)
```

(Match the existing import shape — if `acting_gk_from_frames` is imported on its own line, add `gk_distribution_mask` on its own line in the same alphabetical block.)

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/tracking/test_gk_distribution_mask.py -q`
Expected: PASS (all tests).

---

## Task 3: v1 `_gk_distribution_mask` byte-identical shim + golden gate

**Files:**
- Modify: `silly_kicks/tracking/_xt_gk.py:303-328`
- Test: `tests/tracking/test_gk_distribution_mask.py`

- [ ] **Step 1: Write the failing golden test**

Append to `tests/tracking/test_gk_distribution_mask.py`:

```python
def test_v1_shim_byte_identical_golden():
    """The frozen v1 _gk_distribution_mask must equal a pinned golden that INCLUDES a native GK
    open-play pass (exercises is_open & actor_is_gk, not just the goal-kick term)."""
    from silly_kicks.tracking._xt_gk import _gk_distribution_mask

    actions = pd.DataFrame([
        _action(0, _GOALKICK, 10, 5, 10.0),  # goalkick by non-GK -> True
        _action(1, _PASS, 1, 5, 20.0),        # GK open-play pass -> True (the risky branch)
        _action(2, _PASS, 10, 5, 20.0),       # outfielder pass -> False
        _action(3, _SHOT, 1, 5, 20.0),        # GK shot -> False
        _action(4, _THROW_IN, 1, 5, 20.0),    # GK throw-in -> True
    ])
    out = _gk_distribution_mask(actions, _one_keeper_frames())
    golden = np.array([True, True, False, False, True])
    assert isinstance(out, np.ndarray) and out.dtype == np.bool_
    assert out.tolist() == golden.tolist()
    # The golden is non-trivial on the GK-pass branch (rows 1 & 4 are non-goalkick True):
    non_goalkick_true = out[[1, 2, 3, 4]].sum()
    assert non_goalkick_true == 2  # rows 1 (pass) and 4 (throw_in)
    # And equals the public native path exactly:
    assert out.tolist() == gk_distribution_mask(actions, _one_keeper_frames(), resolve_gk="native").tolist()
```

- [ ] **Step 2: Run to verify it passes with the CURRENT frozen body**

Run: `python -m pytest tests/tracking/test_gk_distribution_mask.py::test_v1_shim_byte_identical_golden -q`
Expected: PASS against the *current* (pre-shim) frozen body — this pins the golden BEFORE the refactor, proving the golden matches existing behavior. (If it fails now, the golden is wrong — fix the golden, not the code.)

- [ ] **Step 3: Replace the frozen body with the shim**

In `silly_kicks/tracking/_xt_gk.py`, replace the entire body of `_gk_distribution_mask` (lines 303-328) with:

```python
def _gk_distribution_mask(actions: pd.DataFrame, frames: pd.DataFrame) -> npt.NDArray[np.bool_]:
    """True for in-scope GK distributions: any goalkick, OR a pass/throw_in whose actor is the acting
    team's goalkeeper (global frames[is_goalkeeper] set-membership; dtype-safe, ADR-019). Non-GK-distribution
    rows -> False. Byte-identical shim over the public tracking.gk_distribution_mask (resolve_gk='native')."""
    from silly_kicks.tracking._gk_resolve import gk_distribution_mask

    return gk_distribution_mask(actions, frames, resolve_gk="native").to_numpy()
```

Then remove the now-unused `canonical_id_series` import from `_xt_gk.py` **only if** no other site in the file uses it. Verify first:

Run: `grep -n "canonical_id_series" silly_kicks/tracking/_xt_gk.py`
- If the shim is the only former user (no other matches) → change line 28 `from ._id_compat import canonical_id_series, ids_equal` to `from ._id_compat import ids_equal`.
- If `ids_equal` is also now unused, drop it too. (Check: `grep -n "ids_equal" silly_kicks/tracking/_xt_gk.py`.)

- [ ] **Step 4: Run the golden + the full v1 byte-stability + xtgk regression gates**

Run: `python -m pytest tests/tracking/test_gk_distribution_mask.py tests/xtgk/test_regression_boundary.py -q`
Expected: PASS. The golden is unchanged across the refactor (byte-identity), and the xtgk boundary gates confirm no v1 output-column / import-cycle regression.

Run: `python -m pytest tests/tracking/ -k "xt_gk or gk_completion" -q`
Expected: PASS — the three `_gk_distribution_mask` consumers (v1 compute, completion, features) are unaffected by the shim.

---

## Task 4: ρ loader — self-adapting `is_gk_distribution` (drop `gk_was_distributing`)

**Files:**
- Modify: `scripts/_loader_databricks.py:264-303`
- Test: `tests/xtgk/test_retention_loader_domain.py`

- [ ] **Step 1: Write the failing pure-decision test**

Create `tests/xtgk/test_retention_loader_domain.py`:

```python
"""ρ loader/trainer GK-distribution domain: self-adapting is_gk_distribution + NULL coalescing."""

import numpy as np
import pandas as pd


def test_should_select_is_gk_distribution_present_absent():
    from scripts._loader_databricks import should_select_is_gk_distribution

    assert should_select_is_gk_distribution({"action_id", "is_gk_distribution", "pressure"}) is True
    assert should_select_is_gk_distribution({"action_id", "pressure"}) is False
    assert should_select_is_gk_distribution(set()) is False


def test_build_retention_sql_conditionally_includes_column():
    from scripts._loader_databricks import _build_retention_sql

    with_col = _build_retention_sql(include_is_gk_distribution=True)
    without = _build_retention_sql(include_is_gk_distribution=False)
    assert "c.is_gk_distribution" in with_col
    assert "is_gk_distribution" not in without
    assert "gk_was_distributing" not in with_col and "gk_was_distributing" not in without
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/xtgk/test_retention_loader_domain.py -q`
Expected: FAIL — `should_select_is_gk_distribution` / `_build_retention_sql` not defined.

- [ ] **Step 3: Refactor the loader**

In `scripts/_loader_databricks.py`, replace the `_RETENTION_SQL` module constant (lines 264-275) and the top of `load_retention_cohort` with:

```python
# --- xT-GK v2 retention (rho) cohort loader (ADR-036 §Part 3, marts-native) -----------------------
# Tracking-frames deprecated: features come from the gold action marts. fct_action_values supplies
# the base SPADL (geometry/type/result/possession); fct_action_context supplies pressure AND the
# GK-distribution domain flag (is_gk_distribution = tracking.gk_distribution_mask, resolve_gk="robust").
# Keyed on (match_key, action_id). The is_gk_distribution SELECT is self-adapting on schema presence
# (TRANSITIONAL: collapse to an unconditional read once the column is permanently materialized -- see
# the retrain follow-up in the spec's Deferred section).
_RETENTION_SQL_TEMPLATE = """
WITH v AS (SELECT * FROM soccer_analytics.dev_gold.fct_action_values WHERE data_source = %(ds)s)
SELECT
  v.match_key AS game_id, v.period AS period_id, v.action_id, v.time_seconds,
  v.team_id, v.player_id, v.start_x, v.start_y, v.end_x, v.end_y,
  v.action_type, v.action_result, v.possession_id, v.data_source,
  c.pressure_on_actor__bekkers_pi AS pressure{is_gk_distribution_select}
FROM v
LEFT JOIN soccer_analytics.dev_gold.fct_action_context c
  ON c.match_key = v.match_key AND c.action_id = v.action_id
ORDER BY v.match_key, v.period, v.time_seconds, v.action_id
"""

# Catalog-qualified so it resolves against soccer_analytics, NEVER the session default catalog
# (an unqualified information_schema returned false-negatives -> the column would silently never load).
_IS_GK_DISTRIBUTION_PROBE = """
SELECT column_name FROM soccer_analytics.information_schema.columns
WHERE table_schema = 'dev_gold' AND table_name = 'fct_action_context'
"""


def should_select_is_gk_distribution(existing_columns: set[str]) -> bool:
    """Pure decision: include c.is_gk_distribution in the SELECT iff the column exists (transitional)."""
    return "is_gk_distribution" in existing_columns


def _build_retention_sql(*, include_is_gk_distribution: bool) -> str:
    frag = ",\n  c.is_gk_distribution" if include_is_gk_distribution else ""
    return _RETENTION_SQL_TEMPLATE.format(is_gk_distribution_select=frag)
```

Then update `load_retention_cohort` body (lines 278-302) to probe + build + conditionally coerce:

```python
def load_retention_cohort(data_source: str) -> pd.DataFrame:
    """Full attack-LTR action stream for the rho retention trainer (marts-native; NO tracking frames).

    Maps the gold string ``action_type``/``action_result`` to SPADL ``type_id``/``result_id``
    (unmapped -> -1, harmless: not a shot/move), carries ``pressure`` + (when materialized) the
    ``is_gk_distribution`` GK-distribution domain flag. Sorted by (game_id, period_id, time_seconds, action_id).
    """
    import silly_kicks.spadl.config as spadlconfig

    if data_source not in _ALLOWED_PROVIDERS:
        raise ValueError(f"data_source {data_source!r} not in allowlist {sorted(_ALLOWED_PROVIDERS)}")
    conn = _connect()
    try:
        cols = {r["column_name"] for r in _query_param(conn.cursor(), _IS_GK_DISTRIBUTION_PROBE, {}).to_dict("records")}
        include = should_select_is_gk_distribution(cols)
        df = _query_param(conn.cursor(), _build_retention_sql(include_is_gk_distribution=include), {"ds": data_source})
    finally:
        conn.close()
    df["type_id"] = df["action_type"].map(spadlconfig.actiontype_id).fillna(-1).astype("int64")
    df["result_id"] = df["action_result"].map(spadlconfig.result_id).fillna(-1).astype("int64")
    for col in ("start_x", "start_y", "end_x", "end_y", "pressure", "time_seconds"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "is_gk_distribution" in df.columns:
        df["is_gk_distribution"] = df["is_gk_distribution"].fillna(False).astype(bool)
    df = df[df["time_seconds"].notna()].copy()
    return df.sort_values(["game_id", "period_id", "time_seconds", "action_id"], kind="stable").reset_index(drop=True)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/xtgk/test_retention_loader_domain.py::test_should_select_is_gk_distribution_present_absent tests/xtgk/test_retention_loader_domain.py::test_build_retention_sql_conditionally_includes_column -q`
Expected: PASS. (The live probe/connection path is owner-run — not in CI.)

---

## Task 5: Trainer domain uses `is_gk_distribution` + loader-drop-safety guard

**Files:**
- Modify: `scripts/train_gk_retention.py:29-38`
- Test: `tests/xtgk/test_retention_loader_domain.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/xtgk/test_retention_loader_domain.py`:

```python
def _domain_actions(is_gk_col=None):
    # 3 goalkicks + 3 GK-passes + 3 outfield passes; geometry/pressure present so features are finite.
    n = 9
    df = pd.DataFrame({
        "game_id": [1] * n, "action_id": range(n), "period_id": [1] * n,
        "time_seconds": np.arange(n, dtype=float),
        "team_id": [5] * n, "player_id": [1, 1, 1, 1, 1, 1, 10, 10, 10],
        "type_id": [22, 22, 22, 0, 0, 0, 0, 0, 0],  # 3 goalkick, 6 pass
        "result_id": [1] * n,
        "start_x": np.linspace(5, 50, n), "start_y": [34.0] * n,
        "end_x": np.linspace(20, 70, n), "end_y": [34.0] * n,
        "pressure": [0.3] * n,
    })
    if is_gk_col is not None:
        df["is_gk_distribution"] = is_gk_col
    return df


def test_domain_present_nonnull_broadens_beyond_goalkicks():
    from scripts.train_gk_retention import prepare_retention_training_data

    # is_gk_distribution True for the 3 GK-passes (rows 3-5), so domain = 3 goalkicks + 3 GK-passes.
    col = pd.Series([False, False, False, True, True, True, False, False, False])
    X, y, groups = prepare_retention_training_data(_domain_actions(col))
    assert len(X) >= 4  # more than the 3 goalkicks alone (some GK-passes survive the finite-label filter)


def test_domain_present_null_coalesces_to_false_goalkicks_only():
    from scripts.train_gk_retention import prepare_retention_training_data

    # The rollout population: column exists but is NULL everywhere -> coalesced to False -> goalkicks-only.
    col = pd.Series([pd.NA] * 9, dtype="object")
    X_null, *_ = prepare_retention_training_data(_domain_actions(col))
    X_absent, *_ = prepare_retention_training_data(_domain_actions(None))
    assert len(X_null) == len(X_absent)  # NULL == absent (goalkicks-only), NOT dropped/corrupted


def test_dropped_column_is_not_a_feature():
    # Loader-drop safety: removing gk_was_distributing can't break inference (it's a domain input,
    # never a model feature). The invariant is asserted UNCONDITIONALLY against the always-available
    # feature-name list (so a weights-less CI still exercises the guarantee), with the bundled default
    # model as an ADDITIONAL layer when present.
    from silly_kicks.xtgk._retention_features import RETENTION_FEATURE_NAMES

    assert "gk_was_distributing" not in RETENTION_FEATURE_NAMES  # unconditional core guarantee
    assert "is_gk_distribution" not in RETENTION_FEATURE_NAMES

    from silly_kicks.xtgk._retention import GkRetentionModel

    try:
        m = GkRetentionModel.from_variant("default")
    except FileNotFoundError:
        return  # weights not bundled -> the unconditional assertions above still ran
    assert "gk_was_distributing" not in m.feature_names
    assert list(m.feature_names) == RETENTION_FEATURE_NAMES
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/xtgk/test_retention_loader_domain.py -k domain -q`
Expected: FAIL — the trainer still keys on `gk_was_distributing`, so a `is_gk_distribution` column is ignored (`test_domain_present_nonnull_broadens_beyond_goalkicks` fails: domain stays 3 goalkicks).

- [ ] **Step 3: Update the trainer domain**

In `scripts/train_gk_retention.py`, update the docstring + the mask (lines 29-38):

```python
    """Build (features, labels, groups) from a full attack-LTR action stream (marts-native).

    Domain = **goal-kicks** (the mart-reliable GK-distribution subset) UNION the materialized
    ``is_gk_distribution`` flag (= tracking.gk_distribution_mask, resolve_gk="robust"; NULLs coalesced
    to False -- the rollout population is out of scope, not dropped). ``retains`` is computed on the FULL
    stream then masked. Drops geometry-unscoreable + truncated-window (NaN-label) rows.
    """
    import silly_kicks.spadl.config as spadlconfig

    mask = actions["type_id"].to_numpy() == spadlconfig.actiontype_id["goalkick"]
    if "is_gk_distribution" in actions.columns:
        mask = mask | actions["is_gk_distribution"].fillna(False).to_numpy(dtype=bool)
```

(The rest of the function — `extract_retention_features`, `retains`, the finite-label filter — is unchanged.)

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/xtgk/test_retention_loader_domain.py -q`
Expected: PASS (all 5 tests; the guard test skips only if weights aren't bundled — they are).

- [ ] **Step 5: Grep for any remaining `gk_was_distributing`**

Run: `grep -rn "gk_was_distributing" scripts/ silly_kicks/ tests/`
Expected: NO matches (the reference is fully retired). If any remain, they are stragglers — remove/repoint them.

---

## Task 6: Public-surface auto-enumerating gates

**Files:**
- Test: existing gates (no new files) — `tests/test_public_api_examples.py`, `tests/tracking/test_id_compat_*`, `tests/test_enrichment_nan_safety.py`

- [ ] **Step 1: Run the Examples gate**

Run: `python -m pytest tests/ -k "public_api_examples or examples" -q`
Expected: PASS. `gk_distribution_mask` is a resolver (returns a Series), not an `add_*` aggregator. If the Examples gate requires a docstring `Examples` block for every public export, add one to `gk_distribution_mask`:

```python
    Examples
    --------
    >>> import pandas as pd
    >>> from silly_kicks.tracking import gk_distribution_mask
    >>> actions = pd.DataFrame(
    ...     {"type_id": [22, 0], "player_id": [10, 1], "team_id": [5, 5]}
    ... )
    >>> gk_distribution_mask(actions, frames=None).tolist()  # goal-kicks-only without frames
    [True, False]
```

Re-run the gate; expected PASS.

- [ ] **Step 2: Run the id-dtype + nan-safety gates**

Run: `python -m pytest tests/tracking/ -k "id_compat or id_dtype" -q && python -m pytest tests/test_enrichment_nan_safety.py -q`
Expected: PASS. `gk_distribution_mask` is not an `add_*` aggregator, so it is not auto-enrolled in the `add_*` id-dtype/nan-safety/liveness gates (those enumerate `add_*` names). Its own dtype/NaN safety is covered by Task 1's `test_string_vs_numeric_ids_match` + `test_nan_player_or_team_not_in_scope`. No new wiring required. If any gate DOES pick it up and fails, that's a real signal — investigate before proceeding.

- [ ] **Step 3: Full tracking + xtgk suite**

Run: `python -m pytest tests/tracking/ tests/xtgk/ -m "not e2e" -q`
Expected: PASS.

---

## Task 7: Version bump, docs, ADR amendment, final review

**Files:**
- Modify: `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, `TODO.md`, `CLAUDE.md`, `docs/superpowers/adrs/ADR-036-*.md`, `uv.lock`, `docs/c4/architecture.{dsl,html}`

- [ ] **Step 1: Bump the version 4.42.0 → 4.43.0 in lockstep**

Edit `pyproject.toml` (`version = "4.43.0"`) and `silly_kicks/__init__.py` (`__version__ = "4.43.0"`). Then:

Run: `grep -rn "4.42.0" pyproject.toml silly_kicks/__init__.py`
Expected: NO matches (both now 4.43.0).

- [ ] **Step 2: Update the lock file**

Run: `python -m pip install -e ".[test]" >/dev/null && (uv lock 2>/dev/null || true)`
Then confirm `uv.lock` shows `4.43.0` for `silly-kicks` (if `uv` is unavailable, note it and skip — CI regenerates).

- [ ] **Step 3: CHANGELOG + TODO**

Prepend a `## 4.43.0` entry to `CHANGELOG.md`:

```markdown
## 4.43.0

- **feat(tracking): public `gk_distribution_mask` (PR-S110, ADR-036 amendment).** Exports the GK-distribution
  domain logic as a stable, frame-optional API. `resolve_gk="robust"` (default) resolves the acting GK per
  action via `acting_gk_from_frames` — **time-accurate**: for the GK-pass term it is a strict **subset** of
  `"native"` (the frozen global-`is_goalkeeper` set-membership), *tightening* stale/substituted keepers, NOT
  broadening (do not switch to `native` "for more rows" — those extra rows are stale-keeper noise).
  `frames=None` → goal-kicks-only. The frozen v1 `_gk_distribution_mask` is now a byte-identical shim over it
  (golden-gated). ρ retention loader/trainer drop the shot-scoped `gk_was_distributing` for a self-adapting,
  NULL-coalesced `is_gk_distribution` read (lakehouse materializes
  `fct_action_context.is_gk_distribution = gk_distribution_mask(..., "robust")`); the loader's `pressure`
  column is unchanged (`pressure_on_actor__bekkers_pi`, pinned in PR-S109). Additive public API; no
  `xt_gk`/VAEP value change, no retrain.
```

In `TODO.md`, remove the lakehouse-export request line if present, and add a deferred follow-up under the tracked section: "ρ retrain on the broadened `is_gk_distribution` domain + collapse the transitional self-adapting loader probe to an unconditional read (blocked on the lakehouse materializing the column)."

- [ ] **Step 4: CLAUDE.md convention line**

Add one line to the tracking/`_gk_resolve` narrative in `CLAUDE.md` (near the `acting_gk_from_frames` mention, PR-S106): note PR-S110 ships the public `gk_distribution_mask` (native/robust lever, frame-optional; v1 `_gk_distribution_mask` is now a byte-identical shim over it) + the ρ loader's `gk_was_distributing → is_gk_distribution` self-adapting correction. Keep it terse, matching the surrounding density.

- [ ] **Step 5: ADR-036 amendment**

Append an amendment section to the ADR-036 file (`ls docs/superpowers/adrs/ADR-036-*.md` to find the exact filename) documenting: the public `gk_distribution_mask` + `resolve_gk` lever; the `robust ⊆ native` tightening semantics; the `fct_action_context.is_gk_distribution` lakehouse contract; the v1 byte-identical shim; the ρ loader's dropped-column correction with NULL-coalescing; and the deferred retrain + loader simplification.

- [ ] **Step 6: /final-review (C4 Phase 4 mandatory)**

Invoke the `final-review` skill. Regenerate `docs/c4/architecture.{dsl,html}` via the `c4` skill (structurizr.war + plantuml.jar, Java 21). The C4 action-coupled-aggregator count **stays 28** — `gk_distribution_mask` is a resolver, not an `add_*` aggregator (same rationale as `acting_gk_from_frames`). Confirm no drift in README / CLAUDE.md / version.

- [ ] **Step 7: Full suite, ruff, pyright**

Run: `python -m pytest tests/ -m "not e2e" -q`
Run: `ruff check . && ruff format --check .`
Run: `pyright`
Expected: all green. Fix any issue before Task 8 (Shift Left — no pre-existing errors left behind).

---

## Task 8: Single commit (explicit approval) + push + PR

- [ ] **Step 1: Present the diff summary and REQUEST APPROVAL**

Show `git status` + `git diff --stat`. **STOP and ask the user for explicit approval to commit** (repo policy: ONE commit/branch, only on explicit approval). Do not proceed without it.

- [ ] **Step 2: Commit (after approval)**

```bash
git add -A
git commit -m "$(cat <<'EOF'
feat(tracking): public gk_distribution_mask + rho loader is_gk_distribution -- silly-kicks 4.43.0 (ADR-036, PR-S110)

Export the GK-distribution domain as a stable, frame-optional tracking.gk_distribution_mask
(native/robust lever); frozen v1 _gk_distribution_mask becomes a byte-identical shim over it.
rho retention loader/trainer drop the shot-scoped gk_was_distributing for a self-adapting,
NULL-coalesced is_gk_distribution read. Additive; no xt_gk/VAEP value change, no retrain.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011yincSJTWHYjAXZAQpBQRy
EOF
)"
```

- [ ] **Step 3: Push + open PR**

```bash
git push -u origin pr-s110-gk-distribution-mask-public-api
gh pr create --title "feat(tracking): public gk_distribution_mask -- silly-kicks 4.43.0 (ADR-036, PR-S110)" --body "<summary + spec/plan links>"
```

Then watch CI (only if the user says "watch ci"); otherwise report the PR URL and stop.

---

## Self-Review Notes

- **Spec coverage:** §3 public fn (Task 1), §4 shim + golden (Task 3), §5 loader/domain + NULL coalescing (Tasks 4-5), §6 lakehouse contract (documented in ADR/CHANGELOG, Task 7), §8 all test cases (Tasks 1/3/5), §9 release (Task 7-8), §10 open items (import-cycle verified, probe catalog-qualified, robust default) all mapped.
- **`ids_equal` positional:** handled by computing on positional numpy arrays and wrapping with `actions.index` once (Task 1, Step 3) — the non-RangeIndex test (Task 1) guards it.
- **Golden covers the GK-pass branch:** Task 3's fixture has 2 non-goal-kick True rows (pass + throw-in) with an explicit `non_goalkick_true == 2` assertion.
- **No per-task commits:** repo policy — single commit in Task 8 behind an approval gate.
- **Discovered (out of scope, low-risk):** `acting_gk_from_frames` compares action-team vs frame-team ids
  with a raw `==` (`_gk_resolve.py:67`), so its GK resolution is dtype-fragile if those dtypes ever differ
  (same-provider dtypes match in practice, so unobserved). NOT fixed here (Chesterton's Fence — not this
  PR's surface); note it in `TODO.md` as a follow-up (route line 67 through `ids_equal`/`same_id`). The
  cross-dtype test targets `native` accordingly.
