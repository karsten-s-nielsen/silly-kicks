# TF-23b — Geometric frame-LTR backstop on the native tracking adapters — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `silly_kicks.tracking.{gradientsports,sportec}.convert_to_frames` self-correct a wrong/absent ET direction flag from goalkeeper geometry, via a shared `direction.finalize_orientation` tail that layers the idempotent geometric net on top of the flag-flip.

**Architecture:** Extract the duplicated orientation tail (ET guard → per-period flag flip → period-gated `team_attacking_direction` label → geometric backstop) of the two native adapters into one shared `direction.finalize_orientation(...)`. Give the geometric net (`orient_frames_to_ltr_by_geometry`) an `on_missing_home` policy parameter (default `"raise"`, so direct/lakehouse callers are byte-identical; adapters pass `"warn"`) and restrict its flip loop to known periods so it never orients penalty shootouts. The backstop is a byte-identical no-op on the correct-flag path.

**Tech Stack:** Python 3.10–3.14, pandas, numpy, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-06-18-tf23b-geometric-backstop-native-adapters-design.md` (rev-4). **Version:** 4.34.0. **Branch:** `pr-s99-tf23b-geometric-backstop`. **ADR:** ADR-035.

**Workflow constraints (project policy — DO NOT VIOLATE):**
- **ONE commit per branch**, made only after explicit owner approval + `/final-review`. **Tasks below contain NO per-task `git commit` steps** — run tests green at each checkpoint instead.
- `/final-review` is mandatory before the single commit. Never tag before main CI is green.
- Version-bump hard gate: `pyproject.toml` + `silly_kicks/__init__.py` + `TODO.md` + `CHANGELOG.md` must ALL show `4.34.0`.
- `ruff format --check` AND `ruff check` AND `pyright silly_kicks/` (full package) must pass.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `silly_kicks/tracking/direction.py` | Modify | Add `on_missing_home` to `orient_frames_to_ltr_by_geometry`; restrict its flip loop to `_LTR_KNOWN_PERIODS`; add the new `finalize_orientation` shared tail |
| `silly_kicks/tracking/gradientsports.py` | Modify | Replace the inline ET-guard/flip/label block with a `direction.finalize_orientation(...)` call; remove the now-unused `ids_match` import |
| `silly_kicks/tracking/sportec.py` | Modify | Same collapse as gradientsports |
| `tests/tracking/test_orient_by_geometry.py` | Modify | Add `on_missing_home` (`warn`) + period-5-PSO unit tests |
| `tests/tracking/test_finalize_orientation.py` | Create | Direct unit test of the shared helper (flip + label + no-op + self-correct + copy-at-entry) |
| `tests/tracking/test_adapter_extra_time_orientation.py` | Modify | Invert the wrong-flag test → self-corrects; add `absolute_frame` variant; add a multi-outfield-player feature-level assertion |
| `tests/regressions/extratime/test_et_guard_parity.py` | Modify | Split the orientation test: events reflect, tracking self-correct |
| `tests/regressions/extratime/test_real_et_roundtrip.py` | Modify | Rewrite the GS test onto the regenerated native-GK fixture + geometric ground truth |
| `tests/regressions/extratime/gs_et/frames.parquet` | Regenerate (DGX) | Match 10517 P3 with native `is_goalkeeper`, from pining |
| `tests/regressions/extratime/README.md` | Modify | Correct the "A-League" competition typo → "WC2022 knockout ET"; document native GK |
| `scripts/regenerate_gs_et_native_gk.py` | Create (DGX) | Reproducible fixture regeneration from pining |
| `docs/superpowers/adrs/ADR-035-geometric-backstop-native-adapters.md` | Create | Decision record |
| `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, `TODO.md` | Modify | Version 4.34.0 + changelog/TODO |

---

## Task 0: Branch + environment + baseline green

**Files:** none (setup)

- [ ] **Step 1: Create the feature branch off main**

Run:
```bash
git checkout main && git pull && git checkout -b pr-s99-tf23b-geometric-backstop
```

- [ ] **Step 2: Sync the dev environment**

Run: `pip install -e ".[test]"`
Expected: installs cleanly.

- [ ] **Step 3: Confirm the relevant baseline suite is green BEFORE any change**

Run:
```bash
python -m pytest tests/tracking/test_orient_by_geometry.py tests/tracking/test_adapter_extra_time_orientation.py tests/regressions/extratime/ -m "not e2e" -v --tb=short
```
Expected: all PASS (this is the pre-change baseline; note the counts).

---

## Task 1: Add `on_missing_home` policy to the geometric net

**Files:**
- Modify: `silly_kicks/tracking/direction.py` (import line 17; signature ~164-170; zero-home branch ~232-236)
- Test: `tests/tracking/test_orient_by_geometry.py`

- [ ] **Step 1: Write the failing tests for the `"warn"` policy AND the `copy=False` knob**

Add to `tests/tracking/test_orient_by_geometry.py`:

```python
def test_on_missing_home_warn_returns_unoriented():
    # home GK on the attacking half in P1 (would normally flip); with a non-matching
    # home_team_id and on_missing_home="warn", the net must warn and return UN-oriented.
    frames = _two_period_match(home_gk_x_p1=100.0, home_gk_x_p2=5.0)
    with pytest.warns(UserWarning, match="matched ZERO"):
        out = orient_frames_to_ltr_by_geometry(frames, home_team_id="NOPE", on_missing_home="warn")
    p1 = out[(out.period_id == 1) & (out.player_id == "hgk")].iloc[0]
    assert p1.x == pytest.approx(100.0)  # un-oriented: no flip applied


def test_copy_false_mutates_in_place_and_returns_same_object():
    # review R1: pin the copy=False optimization as a contract (finalize's 2-copy behavior
    # depends on it) so a future defensive `frames.copy()` re-add can't silently rot the perf win.
    df = _two_period_match(home_gk_x_p1=100.0, home_gk_x_p2=5.0)  # P1 home GK on attacking half => flips
    out = orient_frames_to_ltr_by_geometry(df, home_team_id="H", copy=False)
    assert out is df  # no defensive copy taken
    # mutated in place: P1 home GK reflected from x=100 to x=5
    assert df[(df.period_id == 1) & (df.player_id == "hgk")].iloc[0].x == pytest.approx(5.0)
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `python -m pytest tests/tracking/test_orient_by_geometry.py::test_on_missing_home_warn_returns_unoriented tests/tracking/test_orient_by_geometry.py::test_copy_false_mutates_in_place_and_returns_same_object -v`
Expected: BOTH FAIL — `TypeError: ... unexpected keyword argument 'on_missing_home'` / `'copy'`.

- [ ] **Step 3: Add `Literal` to the typing import**

In `silly_kicks/tracking/direction.py`, change:
```python
from typing import Any
```
to:
```python
from typing import Any, Literal
```

- [ ] **Step 4: Add the parameters to the signature + docstring**

In `orient_frames_to_ltr_by_geometry`, add `on_missing_home` AND a `copy` knob (after `game_id`).
The `copy` knob (review #1) lets `finalize_orientation` — which already owns a fresh copy — avoid a
redundant third full-frame copy on tracking-scale data, while direct/lakehouse callers keep the safe
default:
```python
def orient_frames_to_ltr_by_geometry(
    frames: pd.DataFrame,
    *,
    home_team_id: Any,
    source: str = "",
    game_id: Any = None,
    on_missing_home: Literal["raise", "warn"] = "raise",
    copy: bool = True,
) -> pd.DataFrame:
```

Add to the docstring's Parameters section (after the `source, game_id` entry):
```
    on_missing_home : {"raise", "warn"}, default "raise"
        Policy when ``home_team_id`` matches zero player rows (cannot anchor):
        ``"raise"`` (ADR-019 default — mis-orienting is worse than failing; used by
        direct/lakehouse callers); ``"warn"`` emits a ``UserWarning`` and returns the
        frame un-oriented (the native adapters pass this so their established
        warn-don't-raise contract holds — the flag-flip result stands).
    copy : bool, default True
        When True, operate on a defensive copy (input never mutated). Callers that
        already own a fresh frame (e.g. :func:`finalize_orientation`) pass ``False`` to
        avoid a redundant full-frame copy on tracking-scale data.
```

- [ ] **Step 4b: Honor the `copy` knob at the net's entry**

In `orient_frames_to_ltr_by_geometry`, change the defensive copy line:
```python
    out = frames.copy()
```
to:
```python
    out = frames.copy() if copy else frames
```
(With `copy=False` the caller guarantees ``frames`` is already a private copy, so in-place
``.loc`` writes are safe and produce no ``SettingWithCopyWarning``.)

- [ ] **Step 5: Branch the zero-home guard on the policy**

Replace the existing zero-home raise block:
```python
    if not bool((is_player & is_home).any()):
        raise ValueError(
            f"orient_frames_to_ltr_by_geometry: home_team_id={home_team_id!r} matched ZERO "
            f"player rows ({source} game={game_id}) --- refusing to guess orientation."
        )
```
with:
```python
    if not bool((is_player & is_home).any()):
        msg = (
            f"orient_frames_to_ltr_by_geometry: home_team_id={home_team_id!r} matched ZERO "
            f"player rows ({source} game={game_id})"
        )
        if on_missing_home == "raise":
            raise ValueError(msg + " --- refusing to guess orientation.")
        warnings.warn(msg + " --- orientation left as-is.", stacklevel=2)
        return out
```

- [ ] **Step 6: Run the new tests + the existing raise test together**

Run: `python -m pytest tests/tracking/test_orient_by_geometry.py::test_on_missing_home_warn_returns_unoriented tests/tracking/test_orient_by_geometry.py::test_copy_false_mutates_in_place_and_returns_same_object tests/tracking/test_orient_by_geometry.py::test_zero_home_match_raises -v`
Expected: ALL PASS (warn returns un-oriented; `copy=False` mutates in place + returns the same object; default still raises and `match="matched ZERO"` still matches).

---

## Task 2: Restrict the net's flip loop to known periods (never orient PSO)

**Files:**
- Modify: `silly_kicks/tracking/direction.py` (flip loop ~250)
- Test: `tests/tracking/test_orient_by_geometry.py`

- [ ] **Step 1: Write the failing period-5 test**

Add to `tests/tracking/test_orient_by_geometry.py`:

```python
def test_period5_pso_not_flipped():
    # Penalty shootout: both teams attack one end, so home-GK x is a meaningless anchor.
    # The net must NOT flip period 5 even when the home GK sits at the attacking end (x=100).
    rows = [
        _frame(5, "H", "hgk", 100.0, 5.0, is_gk=True),
        _frame(5, "A", "agk", 100.0, 34.0, is_gk=True),
    ]
    out = orient_frames_to_ltr_by_geometry(pd.DataFrame(rows), home_team_id="H")
    assert out[(out.period_id == 5) & (out.player_id == "hgk")].iloc[0].x == pytest.approx(100.0)
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `python -m pytest tests/tracking/test_orient_by_geometry.py::test_period5_pso_not_flipped -v`
Expected: FAIL — the net currently flips period 5 (home GK x becomes 5.0, not 100.0).

- [ ] **Step 3: Skip non-known periods in the flip loop**

In `orient_frames_to_ltr_by_geometry`, change the loop header:
```python
    for period in pd.Series(period_arr[player_arr]).dropna().unique():
```
to:
```python
    for period in pd.Series(period_arr[player_arr]).dropna().unique():
        if period not in _LTR_KNOWN_PERIODS:  # period 5 (PSO): orientation undefined --- never flip
            continue
```

- [ ] **Step 4: Run the period-5 test + the full net suite**

Run: `python -m pytest tests/tracking/test_orient_by_geometry.py -v`
Expected: ALL PASS (new period-5 test green; periods 1–4 idempotency/orientation cases unchanged — byte-identical for no-PSO data).

---

## Task 3: Add the shared `direction.finalize_orientation` helper

**Files:**
- Modify: `silly_kicks/tracking/direction.py` (new function, place it directly AFTER `orient_frames_to_ltr_by_geometry`)
- Test: `tests/tracking/test_finalize_orientation.py` (create)

- [ ] **Step 1: Write the failing unit tests**

Create `tests/tracking/test_finalize_orientation.py`:

```python
"""Direct unit tests for the shared adapter orientation tail (TF-23b, ADR-035)."""

import pandas as pd
import pytest

from silly_kicks.tracking import direction


def _raw(period, team, player, isgk, isball, x, y):
    return {
        "game_id": "g1",
        "period_id": period,
        "frame_id": period * 10,
        "is_ball": isball,
        "is_goalkeeper": isgk,
        "team_id": team,
        "player_id": player,
        "x": x,
        "y": y,
    }


def _p1_frame():
    # home GK deep at low x (=20); home attacks right under home_team_start_left=True.
    return pd.DataFrame([
        _raw(1, "H", "hgk", True, False, 20.0, 34.0),
        _raw(1, "A", "agk", True, False, 85.0, 34.0),
        _raw(1, None, None, False, True, 50.0, 34.0),
    ])


def test_finalize_correct_flag_labels_and_noop():
    out = direction.finalize_orientation(
        _p1_frame(), home_team_id="H", home_team_start_left=True,
        home_team_start_left_extratime=None, source="test",
    )
    hgk = out[(out.period_id == 1) & (out.player_id == "hgk")].iloc[0]
    assert hgk.x == pytest.approx(20.0)               # correct flag => no flip, no backstop
    assert hgk.team_attacking_direction == "ltr"      # period-gated label


def test_finalize_does_not_mutate_input():
    df = _p1_frame()
    before = df.copy(deep=True)
    direction.finalize_orientation(
        df, home_team_id="H", home_team_start_left=True,
        home_team_start_left_extratime=None, source="test",
    )
    pd.testing.assert_frame_equal(df, before)          # copy-at-entry: input untouched


def test_finalize_wrong_et_flag_self_corrects():
    df = pd.DataFrame([
        _raw(3, "H", "hgk", True, False, 20.0, 34.0),  # raw: home GK deep at low x
        _raw(3, "A", "agk", True, False, 85.0, 34.0),
    ])
    # extratime=False flips P3 (home GK -> x=85); the geometric backstop restores it to low x.
    out = direction.finalize_orientation(
        df, home_team_id="H", home_team_start_left=True,
        home_team_start_left_extratime=False, source="test",
    )
    assert out[(out.period_id == 3) & (out.player_id == "hgk")].iloc[0].x == pytest.approx(20.0)
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `python -m pytest tests/tracking/test_finalize_orientation.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'finalize_orientation'`.

- [ ] **Step 3: Implement `finalize_orientation`**

In `silly_kicks/tracking/direction.py`, add directly after `orient_frames_to_ltr_by_geometry`:

```python
def finalize_orientation(
    out: pd.DataFrame,
    *,
    home_team_id: Any,
    home_team_start_left: bool,
    home_team_start_left_extratime: bool | None,
    source: str,
    game_id: Any = None,
    on_missing_home: Literal["raise", "warn"] = "warn",
) -> pd.DataFrame:
    """Shared orientation tail for the sportec + gradientsports native tracking adapters.

    Single source of truth for the ET guard, the per-period flag flip, the post-flip
    period-gated ``team_attacking_direction`` label, and the TF-23b geometric backstop.
    Expects ``out`` to already carry canonical ``x``/``y`` (105x68 m) plus ``team_id``,
    ``period_id``, ``is_ball``, ``is_goalkeeper``. **Returns a NEW frame and does not mutate
    the input** (copy-at-entry). The output is in home-attacks-right (absolute) convention;
    the caller applies :func:`play_left_to_right` afterward for ``output_convention="ltr"``.

    The geometric backstop (:func:`orient_frames_to_ltr_by_geometry`) self-corrects any period
    whose home GK sits on the attacking half --- e.g. a wrong ``home_team_start_left_extratime``
    placeholder. It is idempotent, so on a correct-flag match it is a byte-identical no-op.
    ``on_missing_home="warn"`` (the adapter default) preserves the adapters' warn-don't-raise
    contract without re-implementing the net's zero-home condition.

    Parameters
    ----------
    out : pd.DataFrame
        Frames with canonical ``x``/``y`` already constructed.
    home_team_id : Any
        Home-team id matching ``out["team_id"]`` (ADR-019 ``ids_match``).
    home_team_start_left, home_team_start_left_extratime : bool, bool | None
        Per-period flip flags (see :func:`home_attacks_right_per_period`).
    source : str
        Converter identity for guard/warning messages, e.g. ``"sportec convert_to_frames"``.
    game_id : Any
        Diagnostic context for the backstop's warnings.
    on_missing_home : {"raise", "warn"}, default "warn"
        Backstop zero-home policy (see :func:`orient_frames_to_ltr_by_geometry`).

    Returns
    -------
    pd.DataFrame
        New frame in home-attacks-right convention.

    Examples
    --------
    Collapse a native adapter's orientation tail to one call::

        from silly_kicks.tracking import direction
        out = direction.finalize_orientation(
            out, home_team_id=home_team_id, home_team_start_left=True,
            home_team_start_left_extratime=None, source="sportec convert_to_frames",
        )
    """
    out = out.copy()  # clean value semantics --- never mutate the caller's frame
    require_et_direction(out["period_id"], home_team_start_left_extratime, source=source)

    flips = home_attacks_right_per_period(home_team_start_left, home_team_start_left_extratime)
    home_rtl_periods = {p for p, attacks_right in flips.items() if not attacks_right}
    flip_mask = out["period_id"].isin(home_rtl_periods).to_numpy()
    out.loc[flip_mask, "x"] = _PITCH_LENGTH_M - out.loc[flip_mask, "x"]
    out.loc[flip_mask, "y"] = _PITCH_WIDTH_M - out.loc[flip_mask, "y"]

    out["team_attacking_direction"] = None
    is_player = (~out["is_ball"].astype(bool)).to_numpy(dtype=bool)
    # ADR-019 dtype-safe is_home: a raw `==` silently matched ZERO players when home_team_id was
    # int and team_id object-string -> every player mislabeled -> play_left_to_right double-flip
    # -> mis-oriented frames (2026-06-09 fix). Do NOT "simplify" back to ==.
    is_home = ids_match(out["team_id"], home_team_id).fillna(False).to_numpy(dtype=bool)
    is_known = out["period_id"].isin(_LTR_KNOWN_PERIODS).to_numpy(dtype=bool)
    out.loc[is_player & is_home & is_known, "team_attacking_direction"] = "ltr"
    out.loc[is_player & ~is_home & is_known, "team_attacking_direction"] = "rtl"

    if is_player.any():  # all-ball frame: nothing to anchor; skip the net entirely
        # copy=False: `out` is already this function's private copy (copy-at-entry), so the
        # net can mutate in place -- avoids a redundant third full-frame copy (review #1).
        out = orient_frames_to_ltr_by_geometry(
            out,
            home_team_id=home_team_id,
            source=source,
            game_id=game_id,
            on_missing_home=on_missing_home,
            copy=False,
        )
    return out
```

- [ ] **Step 4: Run the helper tests + the full net suite**

Run: `python -m pytest tests/tracking/test_finalize_orientation.py tests/tracking/test_orient_by_geometry.py -v`
Expected: ALL PASS.

---

## Task 4: Wire both native adapters onto the shared helper + update ET/parity tests

**Files:**
- Modify: `silly_kicks/tracking/gradientsports.py` (import line 26; block lines 122–153)
- Modify: `silly_kicks/tracking/sportec.py` (import line 31; block lines 138–164)
- Modify: `tests/tracking/test_adapter_extra_time_orientation.py`
- Modify: `tests/regressions/extratime/test_et_guard_parity.py`

- [ ] **Step 1: Invert the wrong-flag ET test (red) + add an absolute_frame variant**

In `tests/tracking/test_adapter_extra_time_orientation.py`, REPLACE `test_wrong_extra_time_flag_reverses_p3_p4` with:

```python
@pytest.mark.parametrize("adapter, home_id, away_id, home_gk, away_gk", _ADAPTERS)
def test_wrong_extra_time_flag_self_corrects_via_geometry(adapter, home_id, away_id, home_gk, away_gk):
    """TF-23b: a WRONG ET flag is self-corrected by the geometric backstop ->
    home GK at x=5 (away GK x=100) in ALL FOUR periods (ltr convention)."""
    raw = _raw_4period(home_id, away_id, home_gk, away_gk)
    out, _ = adapter.convert_to_frames(
        raw,
        home_team_id=home_id,
        home_team_start_left=True,
        home_team_start_left_extratime=True,  # WRONG for this physical setup (P3/P4 reversed)
        output_convention="ltr",
    )
    for p in (1, 2, 3, 4):
        hg = out[(out["period_id"] == p) & (out["player_id"] == home_gk)].iloc[0]
        ag = out[(out["period_id"] == p) & (out["player_id"] == away_gk)].iloc[0]
        assert abs(hg["x"] - 5.0) < 0.01, f"{adapter.__name__} P{p} home GK x={hg['x']}"
        assert abs(ag["x"] - 100.0) < 0.01, f"{adapter.__name__} P{p} away GK x={ag['x']}"


@pytest.mark.parametrize("adapter, home_id, away_id, home_gk, away_gk", _ADAPTERS)
def test_wrong_extra_time_flag_self_corrects_absolute_frame(adapter, home_id, away_id, home_gk, away_gk):
    """The backstop runs before the convention branch, so absolute_frame is corrected too:
    home GK at low x=5 in all periods despite the wrong ET flag."""
    raw = _raw_4period(home_id, away_id, home_gk, away_gk)
    out, _ = adapter.convert_to_frames(
        raw,
        home_team_id=home_id,
        home_team_start_left=True,
        home_team_start_left_extratime=True,  # WRONG
        output_convention="absolute_frame",
    )
    for p in (1, 2, 3, 4):
        hg = out[(out["period_id"] == p) & (out["player_id"] == home_gk)].iloc[0]
        assert abs(hg["x"] - 5.0) < 0.01, f"{adapter.__name__} P{p} home GK abs x={hg['x']}"
```

- [ ] **Step 2: Split the cross-provider parity orientation test (red for tracking)**

In `tests/regressions/extratime/test_et_guard_parity.py`, REPLACE `test_all_converters_et_orientation_reflects_with_flag` with:

```python
_EVENTS = ("sportec_events", "gs_events", "metrica_events")
_TRACKING = ("sportec_tracking", "gs_tracking")


@pytest.mark.parametrize("case", _EVENTS)
def test_events_et_orientation_reflects_with_flag(case):
    out_left = run_converter(case, et=True, flag=True)
    out_right = run_converter(case, et=True, flag=False)
    coord = _COORD[case]
    et_l = out_left[out_left["period_id"].isin([3, 4])].reset_index(drop=True)
    et_r = out_right[out_right["period_id"].isin([3, 4])].reset_index(drop=True)
    assert len(et_l) > 0 and len(et_l) == len(et_r)
    xl = et_l[coord].to_numpy(dtype="float64")
    xr = et_r[coord].to_numpy(dtype="float64")
    finite = np.isfinite(xl) & np.isfinite(xr)
    assert finite.any(), f"{case}: no finite ET coordinates to compare"
    assert np.allclose(xl[finite] + xr[finite], 105.0, atol=1e-6), (
        f"{case}: ET not reflected by flag: {xl[finite]} vs {xr[finite]}"
    )


@pytest.mark.parametrize("case", _TRACKING)
def test_tracking_et_self_corrects_regardless_of_flag(case):
    # TF-23b: the geometric backstop self-corrects a wrong ET flag, so flag=True and
    # flag=False converge to the SAME orientation (NOT reflected) -- net is tracking-only.
    out_left = run_converter(case, et=True, flag=True)
    out_right = run_converter(case, et=True, flag=False)
    et_l = out_left[out_left["period_id"].isin([3, 4])].reset_index(drop=True)
    et_r = out_right[out_right["period_id"].isin([3, 4])].reset_index(drop=True)
    assert len(et_l) > 0 and len(et_l) == len(et_r)
    xl = et_l["x"].to_numpy(dtype="float64")
    xr = et_r["x"].to_numpy(dtype="float64")
    finite = np.isfinite(xl) & np.isfinite(xr)
    assert finite.any(), f"{case}: no finite ET coordinates to compare"
    assert np.allclose(xl[finite], xr[finite], atol=1e-6), (
        f"{case}: ET not self-corrected: {xl[finite]} vs {xr[finite]}"
    )
```

- [ ] **Step 3: Run the new tests to confirm they fail (adapters not yet wired)**

Run:
```bash
python -m pytest tests/tracking/test_adapter_extra_time_orientation.py tests/regressions/extratime/test_et_guard_parity.py -v
```
Expected: `test_wrong_extra_time_flag_self_corrects_*` FAIL (P3/P4 still reversed) and `test_tracking_et_self_corrects_regardless_of_flag` FAIL (still reflected). The events parity + raise-shape + positive ET tests still PASS.

- [ ] **Step 4: Wire the gradientsports adapter onto `finalize_orientation`**

In `silly_kicks/tracking/gradientsports.py`, DELETE the `from ._id_compat import ids_match` import (line 26 — it becomes unused). Then REPLACE the block (the `require_et_direction` call through the `team_attacking_direction` label assignment, current lines 122–153):

```python
    direction.require_et_direction(
        raw_frames["period_id"], home_team_start_left_extratime, source="gradientsports convert_to_frames"
    )

    out = raw_frames.copy()
    out["x"] = out["x_centered"] + 52.5
    out["y"] = out["y_centered"] + 34.0

    home_attacks_right = direction.home_attacks_right_per_period(
        home_team_start_left=home_team_start_left,
        home_team_start_left_extratime=home_team_start_left_extratime,
    )
    home_rtl_periods = {p for p, attacks_right in home_attacks_right.items() if not attacks_right}
    flip_mask = out["period_id"].isin(home_rtl_periods).to_numpy()
    out.loc[flip_mask, "x"] = 105.0 - out.loc[flip_mask, "x"]
    out.loc[flip_mask, "y"] = 68.0 - out.loc[flip_mask, "y"]

    out["team_attacking_direction"] = None
    is_player = (~out["is_ball"].astype(bool)).to_numpy(dtype=bool)
    # ADR-019: dtype-safe is_home. A raw `==` silently matched zero players when home_team_id was
    # int and the frame team_id was object-string -> every player mislabeled "rtl" -> downstream
    # play_left_to_right double-flip -> mis-oriented frames (2026-06-09 fix).
    is_home = ids_match(out["team_id"], home_team_id).fillna(False).to_numpy(dtype=bool)
    if is_player.any() and not (is_player & is_home).any():
        warnings.warn(
            f"gradientsports.convert_to_frames: home_team_id={home_team_id!r} matched ZERO player "
            "rows (id dtype vs frame team_id mismatch?) -- frame orientation would be wrong.",
            stacklevel=2,
        )
    is_known_period = out["period_id"].isin([1, 2, 3, 4]).to_numpy(dtype=bool)
    out.loc[is_player & is_home & is_known_period, "team_attacking_direction"] = "ltr"
    out.loc[is_player & ~is_home & is_known_period, "team_attacking_direction"] = "rtl"
```

with:

```python
    out = raw_frames.copy()
    out["x"] = out["x_centered"] + 52.5
    out["y"] = out["y_centered"] + 34.0

    # Shared orientation tail (ADR-035): ET guard -> per-period flag flip -> period-gated
    # team_attacking_direction label -> idempotent geometric backstop (self-corrects a wrong
    # home_team_start_left_extratime from GK geometry; byte-identical no-op on the correct
    # flag). on_missing_home="warn" preserves this adapter's warn-don't-raise contract.
    out = direction.finalize_orientation(
        out,
        home_team_id=home_team_id,
        home_team_start_left=home_team_start_left,
        home_team_start_left_extratime=home_team_start_left_extratime,
        source=f"{_PROVIDER_NAME} convert_to_frames",
        game_id=(out["game_id"].iloc[0] if len(out) else None),
        on_missing_home="warn",
    )
```

- [ ] **Step 5: Wire the sportec adapter onto `finalize_orientation`**

In `silly_kicks/tracking/sportec.py`, DELETE the `from ._id_compat import ids_match` import (line 31 — unused). Then REPLACE the block (the `require_et_direction` call through the `team_attacking_direction` label assignment, current lines 134–164):

```python
    # Periods in which the home team attacks RTL in raw input --- in those
    # periods ALL rows (player + ball) flip so the output frame is
    # home-team-attacks-LTR. Ball carries NaN direction; flip decisions
    # therefore key on the period rather than the per-row direction column.
    direction.require_et_direction(out["period_id"], home_team_start_left_extratime, source="sportec convert_to_frames")
    home_attacks_right = direction.home_attacks_right_per_period(
        home_team_start_left=home_team_start_left,
        home_team_start_left_extratime=home_team_start_left_extratime,
    )
    home_rtl_periods = {p for p, attacks_right in home_attacks_right.items() if not attacks_right}
    flip_mask = out["period_id"].isin(home_rtl_periods).to_numpy()
    out.loc[flip_mask, "x"] = 105.0 - out.loc[flip_mask, "x"]
    out.loc[flip_mask, "y"] = 68.0 - out.loc[flip_mask, "y"]

    # Post-flip direction column: home-team rows -> "ltr"; away-team rows ->
    # "rtl"; ball rows stay NaN. Period 5 (PSO) has undefined direction and
    # so retains NaN even on player rows.
    out["team_attacking_direction"] = None
    is_player = (~out["is_ball"].astype(bool)).to_numpy(dtype=bool)
    # ADR-019: dtype-safe is_home (raw `==` silently matched zero players for an int home_team_id
    # vs object-string team_id -> mis-oriented frames; 2026-06-09 fix, mirrors gradientsports).
    is_home = ids_match(out["team_id"], home_team_id).fillna(False).to_numpy(dtype=bool)
    if is_player.any() and not (is_player & is_home).any():
        warnings.warn(
            f"sportec.convert_to_frames: home_team_id={home_team_id!r} matched ZERO player rows "
            "(id dtype vs frame team_id mismatch?) -- frame orientation would be wrong.",
            stacklevel=2,
        )
    is_known_period = out["period_id"].isin([1, 2, 3, 4]).to_numpy(dtype=bool)
    out.loc[is_player & is_home & is_known_period, "team_attacking_direction"] = "ltr"
    out.loc[is_player & ~is_home & is_known_period, "team_attacking_direction"] = "rtl"
```

with:

```python
    # Shared orientation tail (ADR-035): ET guard -> per-period flag flip -> period-gated
    # team_attacking_direction label -> idempotent geometric backstop (self-corrects a wrong
    # home_team_start_left_extratime from GK geometry; byte-identical no-op on the correct
    # flag). on_missing_home="warn" preserves this adapter's warn-don't-raise contract.
    out = direction.finalize_orientation(
        out,
        home_team_id=home_team_id,
        home_team_start_left=home_team_start_left,
        home_team_start_left_extratime=home_team_start_left_extratime,
        source=f"{_PROVIDER_NAME} convert_to_frames",
        game_id=(out["game_id"].iloc[0] if len(out) else None),
        on_missing_home="warn",
    )
```

- [ ] **Step 6: Run the ET self-correction + parity + RT no-regress (byte-identity) tests**

Run:
```bash
python -m pytest tests/tracking/test_adapter_extra_time_orientation.py tests/regressions/extratime/test_et_guard_parity.py tests/regressions/extratime/test_rt_no_regress.py -v
```
Expected: ALL PASS — the inverted + absolute_frame self-correction tests, the events-reflect + tracking-self-correct parity split, the raise-shape parity, the positive ET test, AND `test_rt_only_output_value_identical_to_3_30_golden` for all 5 cases (the backstop is a no-op on the correct-flag RT goldens → byte-identical).

---

## Task 5: Feature-level closure (concern 5)

**Files:**
- Modify: `tests/tracking/test_adapter_extra_time_orientation.py` (add one feature-level test)

- [ ] **Step 1: Write the feature-level self-correction test**

Add to `tests/tracking/test_adapter_extra_time_orientation.py`. This pins a feature derived from **multiple non-GK outfield players** (the back-line centroid x) — NOT a GK-dominated quantity — so it catches non-uniform/partial reflection and symmetric-projection false-passes (`feedback_symmetry_test_insufficient_pin_ground_truth`), rather than restating the GK-x check:

```python
def _outfield_backline_centroid_x(out, team_id, period):
    """Mean x of a team's three deepest (lowest-x) OUTFIELD players in a period.

    A multi-player, non-GK-dominated orientation-sensitive quantity. Used to confirm the
    feature layer (not just GK-x) is correctly oriented after self-correction.
    """
    sub = out[(out["period_id"] == period) & (out["team_id"] == team_id) & (~out["is_ball"])]
    sub = sub[sub["is_goalkeeper"] == False]  # noqa: E712 -- explicit non-GK
    return sub.nsmallest(3, "x")["x"].mean()


def test_wrong_et_flag_self_corrects_outfield_feature():
    """The geometry-derived back-line centroid (non-GK, multi-player) matches the correct-flag
    conversion on the ET periods after the backstop self-corrects a wrong ET flag (gradientsports
    fixture carries multiple outfielders per team)."""
    raw = _raw_4period_outfield(home_id=57, away_id=99, home_gk=1, away_gk=2)
    correct, _ = gradientsports.convert_to_frames(
        raw, home_team_id=57, home_team_start_left=True,
        home_team_start_left_extratime=False, output_convention="absolute_frame",
    )
    wrong, _ = gradientsports.convert_to_frames(
        raw, home_team_id=57, home_team_start_left=True,
        home_team_start_left_extratime=True, output_convention="absolute_frame",  # WRONG
    )
    for p in (3, 4):
        c = _outfield_backline_centroid_x(correct, 57, p)
        w = _outfield_backline_centroid_x(wrong, 57, p)
        assert abs(c - w) < 0.5, f"P{p} outfield centroid mismatch: correct={c} wrong={w}"
```

Add the asymmetric multi-outfielder fixture builder near `_raw_4period` (deep home GK + three home outfielders at distinct, asymmetric x so a partial/mirror error is detectable):

```python
def _raw_4period_outfield(home_id, away_id, home_gk, away_gk):
    """Like _raw_4period but with three home + three away OUTFIELD players at asymmetric x."""
    home_gk_xc = {1: -47.5, 2: 47.5, 3: 47.5, 4: -47.5}
    away_gk_xc = {1: 47.5, 2: -47.5, 3: -47.5, 4: 47.5}
    # home outfielders sit ahead of their GK (own half -> middle): asymmetric offsets.
    home_of_xc = {1: [-30.0, -20.0, -5.0], 2: [30.0, 20.0, 5.0], 3: [30.0, 20.0, 5.0], 4: [-30.0, -20.0, -5.0]}
    away_of_xc = {1: [30.0, 20.0, 5.0], 2: [-30.0, -20.0, -5.0], 3: [-30.0, -20.0, -5.0], 4: [30.0, 20.0, 5.0]}
    rows = []
    for p in (1, 2, 3, 4):
        f = p * 100
        rows.append(_raw_row(p, f, home_gk, home_id, False, True, home_gk_xc[p], 0.0))
        rows.append(_raw_row(p, f, away_gk, away_id, False, True, away_gk_xc[p], 0.0))
        for i, xc in enumerate(home_of_xc[p]):
            rows.append(_raw_row(p, f, 10 + i, home_id, False, False, xc, 5.0 * i))
        for i, xc in enumerate(away_of_xc[p]):
            rows.append(_raw_row(p, f, 20 + i, away_id, False, False, xc, 5.0 * i))
        rows.append(_raw_row(p, f, None, None, True, False, 0.0, 0.0))
    return pd.DataFrame(rows)
```

- [ ] **Step 2: Run the feature-level test**

Run: `python -m pytest tests/tracking/test_adapter_extra_time_orientation.py::test_wrong_et_flag_self_corrects_outfield_feature -v`
Expected: PASS (the wrong-flag back-line centroid matches the correct-flag centroid on P3/P4 after self-correction).

---

## Task 6: Create ADR-035 + version bump + CHANGELOG + TODO

**Files:**
- Create: `docs/superpowers/adrs/ADR-035-geometric-backstop-native-adapters.md`
- Modify: `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, `TODO.md`

- [ ] **Step 1: Write ADR-035**

Create `docs/superpowers/adrs/ADR-035-geometric-backstop-native-adapters.md` covering (from spec §6): the decision (shared `finalize_orientation` + geometric backstop, backstop-not-replacement, GS + sportec); the **ADR-029 no-refactor fence supersession scoped to the native adapters only** (kloppy stays intentionally un-routed); `on_missing_home` policy injection (default `"raise"`) **and the additive `copy: bool=True` knob** (default-preserving; `finalize` passes `copy=False` to avoid the redundant third full-frame copy — both are public-net API additions, review R2); the **period-5 public-net behavior change** (never orient PSO, for all callers incl. TF-23 SC/Metrica builders; PSO excluded from geometric analysis); closes ADR-031 Gate D (gated on G3); enumerated retrain scope (≤3 GS WC2022 ET-tracking matches — events=5/tracking=3 — plus any wrong-flag IDSSE-ET; the G1 non-no-op list is authoritative; period-5 frames across providers); anchor-context asymmetry; lazy idempotent cross-repo migration; Chesterton's-Fence note that the all-period loop was an unscoped `cf7b29b` side effect. The G1/G2/G3 results (Task 8) are recorded here before ship. References ADR-034, ADR-031, ADR-029, ADR-019, ADR-010. No NOTICE change (reuses TF-23's ADR-053 credit).

**Write the "enumerated retrain scope" and "period-5 match list" subsections with an explicit
`> PENDING G1 (Task 8) — do not ship until filled` placeholder (review #5).** Those lists are only
known after the Task 8 G1 run; the placeholder prevents an interleaved commit from shipping the ADR
with a guessed/empty scope.

- [ ] **Step 2: Bump the version in `pyproject.toml` and `silly_kicks/__init__.py`**

Run to confirm the current strings, then edit both `4.33.0` → `4.34.0`:
```bash
grep -nE '^version|__version__' pyproject.toml silly_kicks/__init__.py
```
Edit `pyproject.toml` `version = "4.33.0"` → `version = "4.34.0"` and `silly_kicks/__init__.py` `__version__ = "4.33.0"` → `__version__ = "4.34.0"`.

- [ ] **Step 3: Add the CHANGELOG entry**

Add a `## [4.34.0]` `### Changed` block to `CHANGELOG.md` (newest at top, matching the file's existing format): the two native tracking adapters (`gradientsports`, `sportec`) self-correct a wrong/absent ET direction flag from GK geometry via the shared `direction.finalize_orientation` backstop (closes ADR-031 Gate D; **VAEP/tracking retrain trigger** for the ≤3 GS WC2022 ET-tracking matches + any wrong-flag IDSSE-ET whose ET flag was wrong — see ADR-035 for the exact G1 list). Public-net change: `orient_frames_to_ltr_by_geometry` gains `on_missing_home` and `copy` parameters (both additive, default-preserving — direct callers byte-identical) and **no longer orients period-5 / penalty-shootout frames for any caller** (incl. the TF-23 SkillCorner/Metrica builders; PSO frames are excluded from geometric analysis; the lakehouse self-assesses any SC/Metrica PSO re-materialization). The backstop's zero-home warning text changed. ADR-035.

- [ ] **Step 4: Update `TODO.md`**

Update the Current-release header to 4.34.0 (PR-S99); move TF-23b out of "Research & Future Work"; note the lakehouse `correct_frames_to_home_ltr` deletion is now unblocked (cross-repo, not a silly-kicks TODO).

- [ ] **Step 5: Verify the version-bump hard gate**

Run: `grep -rnE "4\.34\.0" pyproject.toml silly_kicks/__init__.py CHANGELOG.md TODO.md`
Expected: a match in all four files.

---

## Task 7 (DGX): Regenerate `gs_et` with native GK + geometric ground-truth test (concern 4)

> **Runs on the DGX** (`ssh karsten@192.168.68.73`) — canonical compute, pining is data source #1. The fixture artifact is committed with the PR. If the executing engineer lacks DGX access, the owner runs Steps 2–3.

**Files:**
- Create: `scripts/regenerate_gs_et_native_gk.py`
- Regenerate: `tests/regressions/extratime/gs_et/frames.parquet`
- Modify: `tests/regressions/extratime/README.md`, `tests/regressions/extratime/test_real_et_roundtrip.py`

- [ ] **Step 1: Write the regeneration script**

Create `scripts/regenerate_gs_et_native_gk.py` that:
- (a) loads GS match **10517** via the pining loader (`scripts/_loader_pining._build_gradientsports`).
  **Fail LOUD (raise) if 10517 is absent from the pining corpus — NO automatic fallback.** The
  committed fixture's identity (10517) is referenced by the README and the test, so any substitution
  must be a deliberate, documented owner decision, not a silent switch. (If 10517 is genuinely absent,
  stop and surface to the owner.)
- (b) slices period 3;
- (c) projects exactly the GS tracking adapter input columns (`EXPECTED_INPUT_COLUMNS` from
  `silly_kicks/tracking/gradientsports.py`: `game_id, period_id, frame_id, time_seconds, frame_rate,
  player_id, team_id, is_ball, is_goalkeeper, x_centered, y_centered, z, speed_native, ball_state`)
  carrying **native** `is_goalkeeper` (`is_goalkeeper_source="native"`) — **no extra restricted fields**;
- (d) **derives `home_team_id` from the loaded match metadata** (never hardcoded) and asserts it equals
  the documented `364` as a tripwire (raise if it differs — that would mean the loaded match isn't the
  documented one). Writes `tests/regressions/extratime/gs_et/frames.parquet` + `meta.parquet`
  (`home_team_id`=derived, `home_start_left`, `home_team_start_left_extratime`). The README and test
  read `home_team_id` from `meta.parquet` — no hardcoded id downstream.

- [ ] **Step 2: Run it on the DGX + confirm native GK**

Run (on DGX):
```bash
ssh karsten@192.168.68.73 'cd ~/Development/silly-kicks && python scripts/regenerate_gs_et_native_gk.py'
```
Then verify the regenerated fixture carries a native home GK at low x in P3 (the geometric truth). Copy the artifact back into the repo working tree.

- [ ] **Step 3: Correct the README + rewrite the GS round-trip test on geometric ground truth**

In `tests/regressions/extratime/README.md`, correct the `gs_et/` heading "Gradient Sports A-League ET match" → "Gradient Sports **WC2022 knockout** ET match" with a one-line note (10517 is WC2022; A-League tracking here is SkillCorner — prior label was a typo), and document that `frames.parquet` now carries native `is_goalkeeper`.

In `tests/regressions/extratime/test_real_et_roundtrip.py`, REPLACE the GS section (the `_gs_frames_from_bronze` roster-synthesis helper + `test_gs_real_et_roundtrip_correct_orientation` + `test_gs_real_et_raises_without_flag`) with a native-GK + **geometric-ground-truth** test that: (a) reads the regenerated adapter-input `frames.parquet` + `meta.parquet`; (b) establishes 10517's correct P3 orientation geometrically (home-GK median x on the low-x half) by converting with the true flag; (c) converts with the **negated** ET flag and asserts the net recovers that geometric truth (home GK back on the low-x half; ET coords in SPADL bounds) — NOT anchored on the `home_team_start_left_extratime` placeholder value (`reference_gs_et_flag_placeholder_unreliable`); (d) keeps a no-flag raise assertion.

- [ ] **Step 4: Run the real-data ET round-trip test**

Run: `python -m pytest tests/regressions/extratime/test_real_et_roundtrip.py -v`
Expected: PASS (native-GK geometric self-correction + raise-without-flag).

---

## Task 8 (DGX): Empirical validation — HARD SHIP GATE (release blocker)

> **Runs on the DGX.** Results recorded in ADR-035 BEFORE the single commit / tag. Do not ship if G1 finds any correct-flag non-no-op.

**Files:** none (validation; results → ADR-035)

- [ ] **Step 1: G1 — no-op proof across all correct-flag matches**

On the DGX, for every GS×64 + IDSSE×7 pining match, compare **pre-backstop** vs **post-backstop**
`convert_to_frames` output with the correct flag and assert byte-identical.

**Mechanism (review #4 — there is no runtime toggle to "disable the backstop"):** compare two builds of
the library on the same pining inputs — the **pre-change baseline** (the installed `4.33.0` wheel, or a
`git worktree`/`git stash` of the pre-TF-23b commit) vs the **`4.34.0` working tree** — per match, via
`pandas.testing.assert_frame_equal(..., check_dtype=True)`. **Strict dtypes (review R3):** both builds
construct `final` through the identical schema-typed `pd.DataFrame({col: out[col]…})` + dtype-coercion
loop, so dtypes must match — strict makes this no-op proof strongest and won't mask a subtle dtype
regression (unlike `test_rt_no_regress`, which compares a parquet-roundtripped golden and must relax
dtypes). If a genuinely benign dtype diff appears, document that single exception rather than relaxing
the whole gate. State the exact baseline ref used in the run log so the gate is reproducible on re-run.

Record the changed-match list (expected: only wrong-flag ET matches) + max coordinate deltas + which
matches carry period-5 frames. **Any correct-flag non-no-op blocks ship — investigate (Chesterton's
Fence).**

- [ ] **Step 2: G2 — GS self-correction on real native GK**

Convert a sample GS ET match with the deliberately-negated ET flag; assert the net restores the correct-flag output.

- [ ] **Step 3: G3 — IDSSE/sportec self-correction on real native GK (Gate D closure)**

Same negated-flag self-correction on a real IDSSE ET match. If no IDSSE match has ET frames, record that explicitly and re-scope Gate D closure (do not silently claim it).

- [ ] **Step 4: Record G1/G2/G3 results in ADR-035**

Write the G1 non-no-op match list (the enumerated retrain scope), period-5-match list, and G2/G3 outcomes into ADR-035.

---

## Final: verification + single commit + release

- [ ] **Step 1: Full local quality gate**

Run:
```bash
ruff format --check .
ruff check .
pyright silly_kicks/
python -m pytest tests/ -m "not e2e" -v --tb=short
```
Expected: all clean / all pass. (Fix any unused-import / type findings — notably the removed `ids_match` imports.)

- [ ] **Step 2: `/final-review`**

Run the mandatory `/final-review` gate; address findings.

- [ ] **Step 3: Single commit (after explicit owner approval)**

ONE commit for the whole branch (code + tests + regenerated fixture + ADR-035 + version artifacts). Use the project's commit-message convention.

- [ ] **Step 4: PR → wait for main CI green → tag**

Open the PR; the owner monitors CI. Only after main CI is green: tag `v4.34.0` and push.

---

## Self-Review (against spec rev-4)

- **§3.3 shared helper** → Task 3 (+ wiring Task 4). **§3.4 on_missing_home** → Task 1. **§3.5 period-5** → Task 2. ✔
- **§4 blast radius** (invert ET test, parity split, RT no-regress byte-identity, warn-text) → Task 4. ✔
- **§5.1 behavioral** → Task 4 Step 1. **§5.2 net on_missing_home + PSO** → Tasks 1–2. **§5.2b direct helper test** → Task 3. **§5.3 parity split** → Task 4 Step 2. **§5.4 contract preservation** (zero-home warns) → Task 1 (`test_on_missing_home_warn_returns_unoriented`). **§5.5 native-GK real fixture + geometric ground truth** → Task 7. **§5.6 feature-level (non-GK)** → Task 5. **§5.7 G1/G2/G3** → Task 8. ✔
- **§6 ADR-035** → Task 6 Step 1 + Task 8 Step 4. **§7 version artifacts** → Task 6. ✔
- **P1 drop `"skip"`** → Task 1 (`Literal["raise","warn"]`). **P2 copy-at-entry** → Task 3. **P3 partial supersession** → Task 6 ADR. **P4 non-GK feature** → Task 5. **P5 licensing** → owner-cleared; Task 7 commits only required columns. **P6 direct unit test** → Task 3. ✔
- **Unused-import cleanup** (`ids_match` in both adapters) → Task 4 Steps 4–5 + Final Step 1. ✔

### Plan review resolutions (cross-session, 2026-06-19 — approved, polish folded)
1. **Three full-frame copies → `copy: bool=True` net knob** (Task 1 Step 4/4b; `finalize` passes
   `copy=False`, Task 3) — restores steady state to 2 copies on tracking-scale data; additive,
   default-preserving, P2 contract intact.
2. **Regen no-silent-fallback + derived id** (Task 7 Step 1) — fail loud if 10517 absent; derive
   `home_team_id` from match metadata (tripwire-assert ==364); README/test read it from `meta.parquet`.
3. **Preserve the ADR-019 `ids_match` rationale comment** (Task 3) — carried onto `finalize`'s
   `is_home` line so a future maintainer doesn't "simplify" back to `==` (Chesterton's Fence).
4. **G1 mechanism specified** (Task 8 Step 1) — compare the 4.33.0 baseline (installed wheel / prior
   commit worktree) vs the 4.34.0 working tree per match; record the baseline ref for reproducibility.
5. **ADR retrain-scope subsections marked `PENDING G1`** (Task 6 Step 1) — placeholder blocks a commit
   shipping a guessed/empty scope.

### Plan review round 2 (cross-session, 2026-06-19 — approved; R1–R3 folded)
- **R1 — test the `copy` knob** (Task 1 Step 1/2/6): red-first `test_copy_false_mutates_in_place_and_returns_same_object` (`out is df` + in-place flip) pins the optimization as a contract so a future defensive `frames.copy()` re-add fails loudly instead of silently rotting the perf win.
- **R2 — `copy` is public API → traceability**: added to the CHANGELOG public-net line (Task 6 Step 3), ADR-035 (Task 6 Step 1), and the spec (rev-5, §3.4 + §9 round 5) so spec↔plan don't drift.
- **R3 — G1 strict dtypes** (Task 8 Step 1): `check_dtype=True` (both sides build `final` identically; in-memory vs in-memory, no parquet roundtrip) strengthens the no-op proof; document any single benign dtype exception rather than relaxing the gate.
