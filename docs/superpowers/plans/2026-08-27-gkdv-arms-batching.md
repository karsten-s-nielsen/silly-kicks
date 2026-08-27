# gkdv arms batching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add batched `delta_das_batch` / `delta_threat_suppression_batch` entry points to the GKDV arms so the lakehouse makes **one** accessible-space call per leg per unit instead of one per scored frame — turning a >45 min/unit (watchdog-killed) pass into ~30 s/unit — while adopting once-per-unit direction pinning and preserving single-frame back-compat.

**Architecture:** `delta_das_batch` pins attacking direction ONCE over the whole unit (`_das_port.pin_direction` on the full stack), attaches it to both legs, calls `get_individual_das` ONCE per leg, and reduces per `(game_id, period_id, frame_id)` with `min_count=1` (honest NaN, never a fictional 0.0). `delta_threat_suppression_batch` zips two per-frame `groupby(KEY)` iterators (no rescan) sharing one per-unit `goal_map`. The single-frame arms become thin wrappers over the batch. No vectorized spearman kernel (measured YAGNI — the threat arm is 0.16 % of the cost).

**Tech Stack:** pandas, numpy, `accessible-space` (the `[das]` extra), `silly_kicks.id_compat` (ADR-019), `silly_kicks.tracking` public seams.

**Spec:** `docs/superpowers/specs/2026-08-27-gkdv-arms-batching-design.md` (APPROVED, two lakehouse reviews). Read it first — this plan implements it verbatim.

## Global Constraints

- **No commit without explicit owner approval.** No `git commit` in any step. Version / PR-S / ADR numbers are assigned at commit-prep AFTER `git fetch && git merge origin/main` (NEXT-FREE at authoring: 4.97.0 / PR-S168 / ADR-075 — do NOT reserve).
- **No retrain, no re-materialize.** gkdv has never produced output (`GKDV_ENABLED=False`); the once-per-unit pin and the 0.0→NaN fix change no persisted value.
- **Once-per-unit direction pin** (owner decision) — the batch pins direction once over the unit; the single-frame wrapper (a one-frame unit) is unchanged.
- **`_das_port` stays the ONLY seam onto accessible-space** — no new `tracking`-internal import in `_arms.py` (`tests/gkdv/test_import_allowlist.py`).
- **ADR-019 id_compat** for every id compare (`ids_match` scalar / `ids_equal` column-vs-column); never raw `==` on ids.
- **NaN, never 0.0** for an unscoreable frame — the per-frame reduce uses `sum(min_count=1)`.
- **Lint at CI scope:** `python -m ruff check silly_kicks/ tests/`, `python -m ruff format --check silly_kicks/ tests/`, bare `python -m pyright`. Tools are NOT on PATH — use `python -m`.
- **Tests:** `python -m pytest tests/ -m "not e2e" -v --tb=short`. accessible-space real-scoring tests use `pytest.importorskip("accessible_space")`; structural (stubbed) tests run on every leg.

## File Structure

- **Modify** `silly_kicks/gkdv/_das_port.py` — add `team_das_by_frame` (batch reduce) + `_resolve_attacking_team_per_row` (scalar/Series broadcast, fail-loud on missing key).
- **Modify** `silly_kicks/gkdv/_arms.py` — add `delta_das_batch`, `delta_threat_suppression_batch`; rewrite `delta_das` / `delta_threat_suppression` as thin wrappers.
- **Modify** `silly_kicks/gkdv/__init__.py` — export the two batch functions.
- **Create** `tests/gkdv/test_arms_batch.py` — all batch-specific tests (amortization oracle, direction semantics, NaN-not-0.0, alignment guard, fail-loud key, threat oracle, back-compat, purity, call-count).
- **Modify** `tests/tracking/test_das.py` — the mixed-batch contract test (pins `get_individual_das`'s NaN-not-crash behaviour; benefits `add_das` too).
- **Modify** `CHANGELOG.md`, `CLAUDE.md` (gkdv bullet). NOTICE unchanged (no new methodology). C4 unchanged (no new aggregator) — verify only.

---

## Task 1: Mixed-batch contract test (pin `get_individual_das`'s NaN-not-crash / NaN-not-0.0 behaviour)

Lands FIRST (spec §7 step 1): the design depends on accessible-space NaN-ing a non-simulatable frame inside a batch. Verified at 2.0.15; this is its durable CI form.

**Files:**
- Test: `tests/tracking/test_das.py` (append)

**Interfaces:**
- Consumes: `silly_kicks.tracking.get_individual_das`.
- Produces: nothing (test-only).

- [ ] **Step 1: Write the failing test**

```python
# tests/tracking/test_das.py (append)
def test_get_individual_das_mixed_batch_nans_bad_frame_not_crash():
    """A batch of [simulatable, non-simulatable] frames must NaN the bad frame, never raise
    and never fabricate a score. get_individual_das is already called this way by add_das
    (features.py single-pass), so this pins a production-relied-upon third-party contract.
    Version-noted: accessible_space==2.0.15."""
    pytest.importorskip("accessible_space")
    from silly_kicks.tracking import get_individual_das

    _CARRIER = "ball_carrier_player_id"

    def good(fid):
        rows = [
            dict(player_id="gk1", team_id="1", is_ball=False, is_goalkeeper=True, x=10.0, y=34.0, vx=0.0, vy=0.0),
            dict(player_id="d1", team_id="1", is_ball=False, is_goalkeeper=False, x=20.0, y=30.0, vx=0.3, vy=0.1),
            dict(player_id="a1", team_id="2", is_ball=False, is_goalkeeper=False, x=30.0, y=34.0, vx=1.0, vy=0.0),
            dict(player_id="a2", team_id="2", is_ball=False, is_goalkeeper=False, x=40.0, y=38.0, vx=1.0, vy=0.2),
            dict(player_id="ball", team_id=None, is_ball=True, is_goalkeeper=False, x=40.0, y=34.0, vx=0.0, vy=0.0),
        ]
        for r in rows:
            r.update(game_id=1, period_id=1, frame_id=fid, team_in_possession="2")
        df = pd.DataFrame(rows)
        df[_CARRIER] = pd.Series(["a2"] * len(df), dtype="string", index=df.index)
        return df

    bad = good(2)
    bad = bad[~bad["is_ball"].astype(bool)].reset_index(drop=True)  # NO ball row -> non-simulatable

    out = get_individual_das(pd.concat([good(1), bad], ignore_index=True))  # must NOT raise
    players = out[~out["is_ball"].astype(bool)]
    das_by_frame = {fid: sub["DAS"] for fid, sub in players.groupby("frame_id")}

    assert das_by_frame[1].notna().any(), "the simulatable frame must score"
    assert float(das_by_frame[1].dropna().sum()) > 0.0
    assert not das_by_frame[2].notna().any(), "the non-simulatable frame's DAS must be all-NaN, not a fabricated score"
```

- [ ] **Step 2: Run it — expect PASS immediately** (this pins existing behaviour, not new code)

Run: `python -m pytest tests/tracking/test_das.py::test_get_individual_das_mixed_batch_nans_bad_frame_not_crash -v`
Expected: PASS on `accessible_space==2.0.15`. If it FAILS (raise, or a non-NaN frame-2), STOP — the whole design's mixed-batch premise is false and must be re-evaluated (the spec's §8 fallback: a per-frame filter in `get_individual_das`). This is the one test we *want* green from the start; it is a tripwire, not a red-first TDD step.

---

## Task 2: `_das_port.team_das_by_frame` + `_resolve_attacking_team_per_row` (batch reduce seam)

**Files:**
- Modify: `silly_kicks/gkdv/_das_port.py`
- Test: `tests/gkdv/test_arms_batch.py` (create)

**Interfaces:**
- Consumes: `get_individual_das`, `silly_kicks.id_compat.ids_equal`.
- Produces:
  - `_attacking_team_by_frame(frames, attacking_team_id_by_frame) -> dict[tuple, team]` — one attacking team per distinct `(game_id, period_id, frame_id)`; scalar broadcast or Series lookup; **RAISES on a Series missing a scored-frame key** (spec §4.2). The SINGLE shared resolver used by BOTH the DAS reduce (mapped to per-row) and the threat loop (Task 4) — resolving round-3 finding 4 (no duplicate resolvers).
  - `team_das_by_frame(frames, attacking_team_id_by_frame, *, direction_col) -> pd.Series` — index `MultiIndex(game_id, period_id, frame_id)`, value = attacking-team DAS sum per frame (NaN via `min_count=1` when no finite attacking DAS).

- [ ] **Step 1: Write the failing test** (per-frame reduce, scalar + Series, min_count=1, fail-loud)

```python
# tests/gkdv/test_arms_batch.py (create)
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

_KEY = ["game_id", "period_id", "frame_id"]
_CARRIER = "ball_carrier_player_id"


def _good_frame(fid: int, gk_x: float = 10.0) -> pd.DataFrame:
    rows = [
        dict(player_id="gk1", team_id="1", is_ball=False, is_goalkeeper=True, x=gk_x, y=34.0, vx=0.0, vy=0.0),
        dict(player_id="d1", team_id="1", is_ball=False, is_goalkeeper=False, x=20.0, y=30.0, vx=0.3, vy=0.1),
        dict(player_id="a1", team_id="2", is_ball=False, is_goalkeeper=False, x=30.0, y=34.0, vx=1.0, vy=0.0),
        dict(player_id="a2", team_id="2", is_ball=False, is_goalkeeper=False, x=40.0, y=38.0, vx=1.0, vy=0.2),
        dict(player_id="ball", team_id=None, is_ball=True, is_goalkeeper=False, x=40.0, y=34.0, vx=0.0, vy=0.0),
    ]
    for r in rows:
        r.update(game_id=1, period_id=1, frame_id=fid, team_in_possession="2")
    df = pd.DataFrame(rows)
    df[_CARRIER] = pd.Series(["a2"] * len(df), dtype="string", index=df.index)
    return df


def _unit(n: int) -> pd.DataFrame:
    return pd.concat([_good_frame(fid) for fid in range(1, n + 1)], ignore_index=True)


def test_team_das_by_frame_reduces_per_frame_over_attacking_team():
    pytest.importorskip("accessible_space")
    from silly_kicks.gkdv import _das_port

    unit = _unit(3)
    unit["attacking_direction"] = _das_port.pin_direction(unit).to_numpy()
    out = _das_port.team_das_by_frame(unit, "2", direction_col="attacking_direction")

    assert isinstance(out, pd.Series)
    assert list(out.index.names) == _KEY
    assert len(out) == 3
    # NON-VACUITY (round-4 defect 2): every frame is scoreable, so the reduce must be all-finite.
    # Without this, an all-NaN result (e.g. a tuple-dtype miss in the per-row `MultiIndex.map`)
    # makes `out.dropna()` empty and `(empty > 0).all()` trivially True -- a guard that cannot fail.
    assert out.notna().all(), "every scored frame must reduce to a finite DAS (not silently all-NaN)"
    assert (out > 0.0).all(), "attacking team has positive dangerous space on every scored frame"


def test_team_das_by_frame_series_is_looked_up_per_frame_and_missing_key_raises():
    pytest.importorskip("accessible_space")
    from silly_kicks.gkdv import _das_port

    unit = _unit(2)
    unit["attacking_direction"] = _das_port.pin_direction(unit).to_numpy()

    # Complete Series: OK.
    att = pd.Series({(1, 1, 1): "2", (1, 1, 2): "2"})
    att.index.names = _KEY
    out = _das_port.team_das_by_frame(unit, att, direction_col="attacking_direction")
    assert len(out) == 2

    # Missing key for frame 2: fail-loud, NOT a silent NaN.
    partial = pd.Series({(1, 1, 1): "2"})
    partial.index.names = _KEY
    with pytest.raises((KeyError, ValueError)):
        _das_port.team_das_by_frame(unit, partial, direction_col="attacking_direction")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/gkdv/test_arms_batch.py -k team_das_by_frame -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'team_das_by_frame'`.

- [ ] **Step 3: Implement `team_das_by_frame` + `_resolve_attacking_team_per_row`**

```python
# silly_kicks/gkdv/_das_port.py (append; keep the existing pin_direction / team_das)
_FRAME_KEY = ["game_id", "period_id", "frame_id"]


def _attacking_team_by_frame(frames, attacking_team_id_by_frame):
    """{(game_id, period_id, frame_id): attacking_team} for every DISTINCT scored frame.

    Scalar -> broadcast to every key; Series -> looked up per key, RAISING (fail-loud) if any
    scored frame's key is absent (an incomplete caller mapping is a bug, and a silent NaN would
    hide it -- spec §4.2). ONE resolver shared by the DAS reduce and the threat loop (round-3
    finding 4).
    """
    keys = [tuple(k) for k in frames[_FRAME_KEY].drop_duplicates().to_numpy()]
    if isinstance(attacking_team_id_by_frame, pd.Series):
        missing = [k for k in keys if k not in attacking_team_id_by_frame.index]
        if missing:
            raise KeyError(
                f"attacking_team_id_by_frame is missing {len(missing)} scored-frame key(s), e.g. "
                f"{missing[:3]}. Supply one entry per scored frame; gkdv fails loud rather than "
                "silently NaN-ing a frame (spec §4.2)."
            )
        return {k: attacking_team_id_by_frame.loc[k] for k in keys}
    return {k: attacking_team_id_by_frame for k in keys}


def team_das_by_frame(frames, attacking_team_id_by_frame, *, direction_col):
    """Per-frame attacking-team DAS sum under a PINNED direction column, over a multi-frame stack.

    ONE accessible-space call over the whole stack (the amortization), then a per-`(game_id,
    period_id, frame_id)` reduce over the attacking team's players with ``min_count=1`` so a frame
    with no finite attacking DAS is NaN, never the fictional 0.0 that ``DAS.dropna().sum()`` yields
    (spec §4.3).
    """
    from silly_kicks.id_compat import ids_equal
    from silly_kicks.tracking import get_individual_das

    out = get_individual_das(frames, attacking_direction_col=direction_col)
    att_map = _attacking_team_by_frame(out, attacking_team_id_by_frame)
    att_per_row = pd.Series(pd.MultiIndex.from_frame(out[_FRAME_KEY]).map(att_map), index=out.index)
    is_att_player = (~out["is_ball"].astype(bool)) & ids_equal(out["team_id"], att_per_row)
    das = out["DAS"].where(is_att_player)  # NaN outside the attacking team's players
    result = das.groupby([out[k] for k in _FRAME_KEY]).sum(min_count=1)
    result.index.names = _FRAME_KEY
    return result
```

- [ ] **Step 4: Run to verify PASS**

Run: `python -m pytest tests/gkdv/test_arms_batch.py -k team_das_by_frame -v`
Expected: PASS. (If `ids_equal` is column-vs-column NA-safe as ADR-019 requires, the ball row's `None` team_id yields False and is excluded regardless of `~is_ball`.)

---

## Task 3: `delta_das_batch` — the DAS batch entry point

**Files:**
- Modify: `silly_kicks/gkdv/_arms.py`
- Test: `tests/gkdv/test_arms_batch.py`

**Interfaces:**
- Consumes: `_das_port.pin_direction`, `_das_port.team_das_by_frame`, `silly_kicks.tracking.DasUnscoreableError`.
- Produces: `delta_das_batch(actual_frames, ghost_frames, *, attacking_team_id_by_frame, params=_DEFAULT_PARAMS) -> pd.Series`.

- [ ] **Step 1: Write the failing tests** — amortization oracle (bit-exact), direction-semantics, NaN-not-0.0, alignment guard, whole-batch DasUnscoreableError

```python
# tests/gkdv/test_arms_batch.py (append)
_GHOST_GK_X = 100.0


def _looped_reference(actual, ghost, *, attacking_team_id, direction):
    """Amortization reference: the SAME once-per-unit direction, but get_individual_das called
    PER FRAME. Isolates batching (batch vs loop of identical math) from the direction change."""
    from silly_kicks.gkdv import _das_port

    a = actual.copy(); a["attacking_direction"] = direction.to_numpy()
    g = ghost.copy(); g["attacking_direction"] = direction.to_numpy()
    out = {}
    for (ka, a_sub), (kg, g_sub) in zip(a.groupby(_KEY), g.groupby(_KEY)):
        assert ka == kg
        a_das = _das_port.team_das(a_sub, attacking_team_id=attacking_team_id, direction_col="attacking_direction")
        g_das = _das_port.team_das(g_sub, attacking_team_id=attacking_team_id, direction_col="attacking_direction")
        out[ka] = a_das - g_das
    s = pd.Series(out); s.index.names = _KEY
    return s


def test_delta_das_batch_is_bit_exact_amortization_of_the_per_frame_loop():
    pytest.importorskip("accessible_space")
    from silly_kicks.gkdv import _das_port, delta_das_batch

    actual = _unit(4)
    ghost = _unit(4)  # scoreable both legs; a real ghost differs, but the ORACLE tests amortization
    ghost.loc[ghost["player_id"] == "gk1", "x"] = 12.0  # a small keeper move so legs are not identical

    direction = _das_port.pin_direction(actual)
    ref = _looped_reference(actual, ghost, attacking_team_id="2", direction=direction)
    got = delta_das_batch(actual, ghost, attacking_team_id_by_frame="2")

    # Bit-exact if accessible-space's per-frame result is call-count-invariant; else pin a
    # measured, version-noted atol here (spec §5.1) and document it.
    pd.testing.assert_series_equal(got, ref, check_names=False, rtol=0, atol=0)


def test_delta_das_batch_nans_unscoreable_frame_not_zero():
    pytest.importorskip("accessible_space")
    from silly_kicks.gkdv import delta_das_batch

    good_a, good_g = _good_frame(1), _good_frame(1, gk_x=12.0)
    bad_a = _good_frame(2)[lambda d: ~d["is_ball"].astype(bool)].reset_index(drop=True)  # no ball
    bad_g = bad_a.copy()
    actual = pd.concat([good_a, bad_a], ignore_index=True)
    ghost = pd.concat([good_g, bad_g], ignore_index=True)

    out = delta_das_batch(actual, ghost, attacking_team_id_by_frame="2")
    assert np.isfinite(out.loc[(1, 1, 1)]), "the scoreable frame must have a finite delta"
    assert pd.isna(out.loc[(1, 1, 2)]), "the unscoreable frame must be NaN, never a fabricated 0.0"


def test_delta_das_batch_raises_on_misaligned_legs():
    pytest.importorskip("accessible_space")
    from silly_kicks.gkdv import delta_das_batch

    actual = _unit(2)
    ghost = _unit(2).iloc[::-1].reset_index(drop=True)  # reversed row order, same index 0..n-1
    with pytest.raises(ValueError, match="align"):
        delta_das_batch(actual, ghost, attacking_team_id_by_frame="2")


def test_delta_das_batch_whole_batch_unscoreable_returns_all_nan_over_keys():
    pytest.importorskip("accessible_space")
    from silly_kicks.gkdv import delta_das_batch

    actual = _unit(2).copy(); actual["team_in_possession"] = pd.NA  # dead-ball whole batch
    ghost = actual.copy()
    out = delta_das_batch(actual, ghost, attacking_team_id_by_frame="2")
    assert len(out) == 2 and out.isna().all(), "a wholly-unscoreable unit is all-NaN over its frame keys"
```

The direction-semantics has TWO tests. First, the CONSEQUENCE (spec §5.2, round-3 finding 2) — the once-per-unit pin is stable where a single frame would flip. This is what uses `_GHOST_GK_X` (verified numerically: per-unit `-1.0` stable vs per-frame-alone `+1.0` flip). **It is a real-scoring test** (`importorskip`, round-4 defect 1): `pin_direction` → `_pin_attacking_direction` calls `_import_accessible_space()` (`_das.py:533`, fail-fast) and uses `accessible_space.interface.infer_playing_direction` (`:551/:557`), so it is NOT structural and would `ImportError` on a no-`[das]` leg without the guard:

```python
def test_once_per_unit_pin_is_stable_where_a_single_frame_would_flip():
    """Spec §5.2 CONSEQUENCE: pinning direction ONCE over the unit gives the flip frame the SAME
    (stable) direction as the majority, whereas pinning that frame ALONE (the OLD per-frame
    behaviour) flips it. `pin_direction` uses accessible-space's `infer_playing_direction`, so this
    is a real-scoring test (skipped without [das]). The LOAD-BEARING assertion is
    `d_flip_unit != d_flip_alone`; `d_flip_unit == d_normal` is a near-by-construction sanity check
    (infer_playing_direction is constant per (period, team_in_possession))."""
    pytest.importorskip("accessible_space")
    from silly_kicks.gkdv import _das_port

    # 3 frames with team-1 keeper low (x=10) + 1 flip frame with it at _GHOST_GK_X (=100). The unit
    # mean keeps team-1 the argmin (26.25 < team-2's 35), so the flip frame stays stable under the
    # per-unit pin; ALONE it crosses team-2's mean and flips.
    unit = pd.concat(
        [_good_frame(1), _good_frame(2), _good_frame(3), _good_frame(4, gk_x=_GHOST_GK_X)],
        ignore_index=True,
    )
    per_unit = _das_port.pin_direction(unit)
    d_flip_unit = per_unit[unit["frame_id"] == 4].iloc[0]
    d_normal = per_unit[unit["frame_id"] == 1].iloc[0]
    d_flip_alone = _das_port.pin_direction(_good_frame(4, gk_x=_GHOST_GK_X)).iloc[0]

    assert d_flip_unit == d_normal, "once-per-unit pin is stable on the flip frame (majority-dominated)"
    assert d_flip_unit != d_flip_alone, "the OLD per-frame pin flips this frame -- exactly what the batch changes"
```

Second, the MECHANISM — `delta_das_batch` actually uses that once-per-unit pin (calls `pin_direction` ONCE, on the FULL factual stack, feeding both legs). STRUCTURAL (stub `_das_port`), so it runs on every leg with no `[das]`:

```python
def test_delta_das_batch_pins_ONE_direction_over_the_unit_where_per_frame_would_flip(monkeypatch):
    """Once-per-unit pin: both legs get the SAME per-unit direction, and it is stable across a
    frame whose single-frame mean-x argmin would flip. Structural -- stub the DAS scorer."""
    from silly_kicks.gkdv import _arms, delta_das_batch

    seen = {"pin_frames": []}

    def spy_pin(frames):
        # SYNTHETIC direction -- do NOT call the real pin_direction (it uses accessible-space);
        # this test only checks that pin is called ONCE on the full stack, so a stub keeps it
        # structural / every-leg (round-4 defect 1, sibling of the consequence test).
        seen["pin_frames"].append(frames.copy())
        return pd.Series(1.0, index=frames.index)

    def stub_team_das_by_frame(frames, attacking_team_id_by_frame, *, direction_col):
        # return the per-frame MEAN of the pinned direction so the test can read what was applied
        s = frames.groupby(_KEY)[direction_col].mean()
        s.index.names = _KEY
        return s

    monkeypatch.setattr(_arms._das_port, "pin_direction", spy_pin)
    monkeypatch.setattr(_arms._das_port, "team_das_by_frame", stub_team_das_by_frame)

    actual = _unit(3)
    ghost = _unit(3)
    delta_das_batch(actual, ghost, attacking_team_id_by_frame="2")

    # pin_direction was called ONCE, on the FULL factual stack (not per frame, not on the ghost).
    assert len(seen["pin_frames"]) == 1
    assert len(seen["pin_frames"][0]) == len(actual)
    # (delta = actual_mean - ghost_mean == 0 here because both legs get the identical pinned column.)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/gkdv/test_arms_batch.py -k delta_das_batch -v`
Expected: FAIL — `ImportError: cannot import name 'delta_das_batch'`.

- [ ] **Step 3: Implement `delta_das_batch`**

```python
# silly_kicks/gkdv/_arms.py (append)
import numpy as np

_FRAME_KEY = ["game_id", "period_id", "frame_id"]


def _assert_legs_aligned(actual_frames, ghost_frames, *, fn):
    cols = _FRAME_KEY + ["player_id"]
    a = actual_frames[cols].reset_index(drop=True)
    g = ghost_frames[cols].reset_index(drop=True)
    if not a.equals(g):
        raise ValueError(
            f"{fn}: the factual and ghost frames are not aligned on {cols} order. The pinned "
            "direction is applied positionally, so a misaligned ghost would be scored against "
            "another row's attacking direction. Pass the frames as build_ghost_frames returned "
            "them, restricted identically on both legs."
        )


def _frame_key_index(frames):
    keys = frames[_FRAME_KEY].drop_duplicates()
    return pd.MultiIndex.from_frame(keys)


def delta_das_batch(actual_frames, ghost_frames, *, attacking_team_id_by_frame, params=_DEFAULT_PARAMS):
    """Batched Delta-DAS: one accessible-space call per leg over all a unit's scored frames.

    See ``delta_das`` for the per-frame semantics; this is its amortized batch form. Direction is
    pinned ONCE over the unit (spec §3.1/§4.2). Returns a ``pd.Series`` indexed by
    ``(game_id, period_id, frame_id)``, value ``das(actual) - das(ghost)`` (attacker-value units;
    negative = deterrent). An unscoreable frame is NaN (``min_count=1``), never 0.0; a wholly
    unscoreable unit is all-NaN over its keys.

    Examples
    --------
    Both legs MUST come from the SAME ``build_ghost_frames`` call, restricted identically to the
    engine's scored set::

        from silly_kicks.gkdv import build_ghost_frames, delta_das_batch

        ghost_frames, provenance, report = build_ghost_frames(frames, home_team_id=1)
        scored = provenance.loc[provenance["drop_reason"].isna(), ["game_id", "period_id", "frame_id"]]
        actual = frames.merge(scored, on=["game_id", "period_id", "frame_id"])
        ghost = ghost_frames.merge(scored, on=["game_id", "period_id", "frame_id"])

        deltas = delta_das_batch(actual, ghost, attacking_team_id_by_frame=2)
    """
    from silly_kicks.tracking import DasUnscoreableError

    from . import _das_port

    _assert_legs_aligned(actual_frames, ghost_frames, fn="delta_das_batch")
    keys = _frame_key_index(actual_frames)
    try:
        direction = _das_port.pin_direction(actual_frames)  # ONCE over the unit
        actual_pinned = actual_frames.copy()
        actual_pinned["attacking_direction"] = direction.to_numpy()
        ghost_pinned = ghost_frames.copy()
        ghost_pinned["attacking_direction"] = direction.to_numpy()
        actual = _das_port.team_das_by_frame(actual_pinned, attacking_team_id_by_frame, direction_col="attacking_direction")
        ghost = _das_port.team_das_by_frame(ghost_pinned, attacking_team_id_by_frame, direction_col="attacking_direction")
    except DasUnscoreableError:
        return pd.Series(np.nan, index=keys, name="delta_das")
    delta = (actual - ghost).reindex(keys)  # NaN propagates; reindex pins order/coverage to the keys
    delta.name = "delta_das"
    return delta
```

- [ ] **Step 4: Run to verify PASS**

Run: `python -m pytest tests/gkdv/test_arms_batch.py -k delta_das_batch -v`
Expected: PASS. If `test_delta_das_batch_is_bit_exact...` fails on a tiny fp delta, measure it, replace `atol=0` with the measured value + a comment pinning `accessible_space==2.0.15`, and note it in the spec §5.1 (do NOT loosen silently).

---

## Task 4: `delta_threat_suppression_batch` — batch-first threat API (thin loop)

**Files:**
- Modify: `silly_kicks/gkdv/_arms.py`
- Test: `tests/gkdv/test_arms_batch.py`

**Interfaces:**
- Consumes: `silly_kicks.tracking.compute_threat_pc`, `SpearmanParams`.
- Produces: `delta_threat_suppression_batch(actual_frames, ghost_frames, *, attacking_team_id_by_frame, xt, goal_map, params=_DEFAULT_PARAMS) -> pd.Series`.

- [ ] **Step 1: Write the failing test** — batched == looped single-frame (bit-exact), shares one goal_map

```python
# tests/gkdv/test_arms_batch.py (append)
def test_delta_threat_suppression_batch_equals_looping_the_single_frame_arm():
    pytest.importorskip("accessible_space")
    from silly_kicks.gkdv import delta_threat_suppression, delta_threat_suppression_batch
    from silly_kicks.tracking import resolve_defended_goals
    from tests.gkdv._xt_helpers import fitted_xt_for_tests  # a tiny fitted ExpectedThreat helper

    actual = _threat_unit(3)  # lifted from test_deterrent_keeper_gives_a_NEGATIVE_delta, stacked (Task 4 note)
    ghost = actual.copy(); ghost.loc[ghost["is_goalkeeper"].astype(bool), "x"] += 2.0
    xt = fitted_xt_for_tests()
    goal_map = resolve_defended_goals(actual)

    batched = delta_threat_suppression_batch(
        actual, ghost, attacking_team_id_by_frame="2", xt=xt, goal_map=goal_map
    )
    for (k, a_sub), (_, g_sub) in zip(actual.groupby(_KEY), ghost.groupby(_KEY)):
        one = delta_threat_suppression(a_sub, g_sub, attacking_team_id="2", xt=xt, goal_map=goal_map)
        assert batched.loc[k] == pytest.approx(one, rel=0, abs=0), f"frame {k} batched != looped"


def test_delta_threat_suppression_batch_scores_a_dead_ball_unit_without_crashing():
    """Round-3 finding 3: the threat arm is possession-INDEPENDENT (compute_threat_pc takes
    attacking_team_id explicitly and reads NO team_in_possession), so a dead-ball unit SCORES
    rather than raising -- the inherent, correct asymmetry with the DAS arm's
    DasUnscoreableError->NaN. Reuse the threat fixture (see Task 4 note); set possession NA."""
    pytest.importorskip("accessible_space")
    from silly_kicks.gkdv import delta_threat_suppression_batch
    from silly_kicks.tracking import resolve_defended_goals
    from tests.gkdv._xt_helpers import fitted_xt_for_tests

    actual = _threat_unit(3)  # lifted from test_deterrent_keeper_gives_a_NEGATIVE_delta, stacked
    ghost = actual.copy(); ghost.loc[ghost["is_goalkeeper"].astype(bool), "x"] += 2.0
    actual["team_in_possession"] = pd.NA
    ghost["team_in_possession"] = pd.NA
    xt = fitted_xt_for_tests()
    goal_map = resolve_defended_goals(actual)  # GK-position based, not possession -- still resolves

    out = delta_threat_suppression_batch(actual, ghost, attacking_team_id_by_frame="2", xt=xt, goal_map=goal_map)
    assert out.notna().all(), "the threat arm scores a dead-ball unit (no DasUnscoreableError equivalent)"
```

> **Note for the implementer — reuse the EXISTING working threat fixture, do not hand-roll one.**
> `compute_threat_pc` + `resolve_defended_goals` have preconditions the minimal `_unit` fixture above
> does NOT guarantee (a GK resolvable per team for the `goal_map`; a fitted `xt`). The existing
> single-frame threat test — `tests/gkdv/test_arms.py::test_deterrent_keeper_gives_a_NEGATIVE_delta`
> (the polarity gate referenced at `_arms.py:48`) — already builds frames + a fitted `xt` +
> `goal_map` that pass `compute_threat_pc` end to end. Build this batch test by lifting THAT fixture
> and stacking it to N distinct `frame_id`s (jitter positions slightly per frame), rather than reusing
> `_unit`/`_good_frame` (which is DAS-shaped, GK only on team 1). If the xt-fitting is inline there,
> extract it to `tests/gkdv/_xt_helpers.py::fitted_xt_for_tests()` (a helper, not a fixture) so both
> the single-frame and batch tests call it; grep `require_fitted_xt` / `ExpectedThreat(` in tests for
> the existing idiom. The batched==looped assertion holds regardless of absolute orientation, since
> both legs read the same frames. Name the lifted+stacked factual-unit helper `_threat_unit(n)` (used
> by both threat tests above); each test builds its own ghost via `actual.copy()` + a keeper move.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/gkdv/test_arms_batch.py -k threat_suppression_batch -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement `delta_threat_suppression_batch`** (zip two `groupby(KEY)` iterators — no rescan)

```python
# silly_kicks/gkdv/_arms.py (append)
def delta_threat_suppression_batch(
    actual_frames, ghost_frames, *, attacking_team_id_by_frame, xt, goal_map, params=_DEFAULT_PARAMS
):
    """Batched Delta-GK-threat-suppression. A thin per-frame loop (the threat arm is ~1 ms/frame;
    spec §1.1 measured it at 0.16 % of the DAS cost, so no vectorized kernel). Returns a
    ``pd.Series`` indexed by ``(game_id, period_id, frame_id)`` matching ``delta_das_batch``.

    Both legs are iterated as two aligned ``groupby(KEY)`` streams (identical key order after the
    alignment guard), so there is no per-frame rescan of the ghost stack.
    """
    from silly_kicks.tracking import SpearmanParams

    from . import _das_port
    from silly_kicks import tracking

    _assert_legs_aligned(actual_frames, ghost_frames, fn="delta_threat_suppression_batch")
    att_per_frame = _das_port._attacking_team_by_frame(actual_frames, attacking_team_id_by_frame)  # shared resolver
    base = dict(
        xt=xt,
        goal_map=goal_map,
        method=params.pitch_control_method,
        params=SpearmanParams(lambda_gk=params.lambda_gk),
    )
    out = {}
    for (ka, a_sub), (kg, g_sub) in zip(actual_frames.groupby(_FRAME_KEY), ghost_frames.groupby(_FRAME_KEY)):
        assert ka == kg  # guaranteed by _assert_legs_aligned
        atk = att_per_frame[ka]
        a = tracking.compute_threat_pc(a_sub, attacking_team_id=atk, **base)
        g = tracking.compute_threat_pc(g_sub, attacking_team_id=atk, **base)
        out[ka] = float(a - g)
    s = pd.Series(out, name="delta_threat_suppression")
    s.index = pd.MultiIndex.from_tuples(s.index, names=_FRAME_KEY) if len(s) else _frame_key_index(actual_frames)
    return s.reindex(_frame_key_index(actual_frames))
```

The threat loop reuses `_das_port._attacking_team_by_frame` (Task 2) — no second resolver (round-3 finding 4).

> **Round-3 finding 3 — the DAS/threat NaN asymmetry is INHERENT and correct; verify + document, do NOT add a symmetric guard.** `delta_das_batch` catches `DasUnscoreableError` (velocity-less / dead-ball direction inference — DAS structurally needs velocity AND a possession-inferred direction). `compute_threat_pc` takes `attacking_team_id` **explicitly** and reads NO `team_in_possession` (verified: zero references in `_cover_shadows.py`/`pitch_control/*.py`), so it never raises on a dead-ball frame — it always scores from positions + the fitted `xt`. The two arms are therefore *independently scoreable*: a velocity-less frame is DAS-NaN but threat-valued, which is correct (they measure different things; the lakehouse §10 pooling treats each arm's NaN as its own exclude). **Task 4 verification step:** add a quick test that `delta_threat_suppression_batch` returns a finite value on a `team_in_possession=pd.NA` (dead-ball) unit — proving no crash — and add a one-line note to the arm docstring + the ADR (Task 7) that the arms are independently scoreable and need no symmetric `DasUnscoreableError` catch.

- [ ] **Step 4: Run to verify PASS**

Run: `python -m pytest tests/gkdv/test_arms_batch.py -k threat_suppression_batch -v`
Expected: PASS.

---

## Task 5: Single-frame wrappers delegate to the batch (back-compat)

**Files:**
- Modify: `silly_kicks/gkdv/_arms.py`
- Test: `tests/gkdv/test_arms_batch.py`; run the full `tests/gkdv/test_arms.py` unchanged.

**Interfaces:**
- Produces: `delta_das` / `delta_threat_suppression` rewritten as one-frame-stack wrappers over the batch; signatures and return types unchanged.

- [ ] **Step 1: Write the failing test** — a one-frame batch equals the wrapper scalar, and both equal the pre-existing single-frame result on a scoreable frame

```python
# tests/gkdv/test_arms_batch.py (append)
def test_single_frame_delta_das_equals_one_frame_batch():
    pytest.importorskip("accessible_space")
    from silly_kicks.gkdv import delta_das, delta_das_batch

    actual, ghost = _good_frame(1), _good_frame(1, gk_x=100.0)
    scalar = delta_das(actual, ghost, attacking_team_id="2")
    batched = delta_das_batch(actual, ghost, attacking_team_id_by_frame="2")
    assert batched.iloc[0] == pytest.approx(scalar, rel=0, abs=0)
    assert np.isfinite(scalar) and scalar != 0.0
```

- [ ] **Step 2: Run to verify** — should PASS if `delta_das` already returns the same math; this test EXISTS to catch a regression when Step 3 rewrites `delta_das`. Run it now (pre-rewrite) and confirm PASS, then keep it green through Step 3.

Run: `python -m pytest tests/gkdv/test_arms_batch.py -k one_frame_batch -v`

- [ ] **Step 3: Rewrite the single-frame arms as thin wrappers**

```python
# silly_kicks/gkdv/_arms.py -- REPLACE the bodies of delta_das and delta_threat_suppression
def delta_das(actual_frame, ghost_frame, *, attacking_team_id, params=_DEFAULT_PARAMS):
    """<keep the existing docstring>"""
    return float(
        delta_das_batch(
            actual_frame, ghost_frame, attacking_team_id_by_frame=attacking_team_id, params=params
        ).iloc[0]
    )


def delta_threat_suppression(actual_frame, ghost_frame, *, attacking_team_id, xt, goal_map, params=_DEFAULT_PARAMS):
    """<keep the existing docstring>"""
    return float(
        delta_threat_suppression_batch(
            actual_frame,
            ghost_frame,
            attacking_team_id_by_frame=attacking_team_id,
            xt=xt,
            goal_map=goal_map,
            params=params,
        ).iloc[0]
    )
```

> Keep the existing single-frame docstrings (Examples, polarity, the direction-pin gate reference). Move the DasUnscoreableError→NaN behaviour into `delta_das_batch` (Task 3) — a one-frame unscoreable stack now returns an all-NaN length-1 Series, and `.iloc[0]` is NaN, so `float(NaN)` preserves the old NaN return.

- [ ] **Step 4: Run the FULL existing gkdv suite + the batch suite**

Run: `python -m pytest tests/gkdv/ -v`
Expected: all PASS. In particular `tests/gkdv/test_arms.py::test_das_arm_passes_ONE_pinned_direction_to_BOTH_legs` (structural, stubs `_das_port`) and `test_das_arm_returns_a_LIVE_FINITE_delta...` must still pass — the wrapper routes through the same `_das_port` seam.

- [ ] **Step 5: Confirm no test pins the old all-NaN→0.0 single-frame edge**

Run: `grep -rn "delta_das" tests/gkdv/ | grep -i "0.0\|== 0"` and read each hit. The 0.0→NaN change (spec §4.3) only affects the previously-latent "no simulatable frame" edge; if any test asserts `delta_das(...) == 0.0` for an all-NaN input, STOP and raise it with the owner (it was asserting a bug). Expected: no such test (the live guard asserts `delta != 0.0`).

---

## Task 6: Public exports, purity, and the structural call-count guard

**Files:**
- Modify: `silly_kicks/gkdv/__init__.py`
- Test: `tests/gkdv/test_arms_batch.py`

- [ ] **Step 1: Export the batch functions** — TWO REAL edits (round-3 finding 1: the module has a real `__all__` list at `__init__.py:29`; a comment would leave the functions importable but absent from the public surface, which the public-API gate enforces).

Edit the import at `silly_kicks/gkdv/__init__.py:19`:

```python
# BEFORE
from ._arms import delta_das, delta_threat_suppression
# AFTER
from ._arms import (
    delta_das,
    delta_das_batch,
    delta_threat_suppression,
    delta_threat_suppression_batch,
)
```

Add the two names to the existing `__all__` list (the block starting at `:29`, beside `"delta_das"` / `"delta_threat_suppression"` at `:38`-`:39`):

```python
__all__ = [
    ...
    "delta_das",
    "delta_das_batch",
    "delta_threat_suppression",
    "delta_threat_suppression_batch",
    ...
]
```

- [ ] **Step 2: Write + run the purity test** (batch must not mutate caller frames)

```python
# tests/gkdv/test_arms_batch.py (append)
def test_delta_das_batch_does_not_mutate_inputs():
    pytest.importorskip("accessible_space")
    from silly_kicks.gkdv import delta_das_batch

    actual, ghost = _unit(2), _unit(2)
    ghost.loc[ghost["player_id"] == "gk1", "x"] = 12.0
    a_before, g_before = actual.copy(), ghost.copy()
    delta_das_batch(actual, ghost, attacking_team_id_by_frame="2")
    pd.testing.assert_frame_equal(actual, a_before)
    pd.testing.assert_frame_equal(ghost, g_before)
```

Run: `python -m pytest tests/gkdv/test_arms_batch.py -k mutate -v` → PASS (the impl `.copy()`s before attaching `attacking_direction`).

- [ ] **Step 3: Structural call-count guard (spec §5.7)** — the batch makes O(1) accessible-space calls, not O(frames)

```python
# tests/gkdv/test_arms_batch.py (append)
def test_delta_das_batch_calls_accessible_space_once_per_leg_regardless_of_frame_count():
    """The amortization, proven structurally (no wall-clock): 2 legs -> exactly 2
    get_individual_das calls whether the unit has 2 frames or 20."""
    from silly_kicks.gkdv import _arms

    calls = {"n": 0}

    def counting_team_das_by_frame(frames, attacking_team_id_by_frame, *, direction_col):
        calls["n"] += 1
        s = frames.groupby(_KEY)[direction_col].mean(); s.index.names = _KEY
        return s

    import unittest.mock as mock
    with mock.patch.object(_arms._das_port, "team_das_by_frame", counting_team_das_by_frame), \
         mock.patch.object(_arms._das_port, "pin_direction", lambda f: pd.Series(1.0, index=f.index)):
        for n in (2, 20):
            calls["n"] = 0
            _arms.delta_das_batch(_unit(n), _unit(n), attacking_team_id_by_frame="2")
            assert calls["n"] == 2, f"expected 2 reduce calls (one per leg) for n={n}, got {calls['n']}"
```

> This counts the `_das_port` reduce calls (each of which is exactly one `get_individual_das` call). It is scale-invariant by construction, complementing the §1.1 wall-clock. (A full ADR-073 `assert_subquadratic_growth` adopter is optional and not required here — the call-count is O(1), not O(n), so the growth-exponent harness is not the right tool; this exact-count assertion is.)

Run: `python -m pytest tests/gkdv/test_arms_batch.py -k call -v` → PASS.

---

## Task 7: Docs, C4 verification, and the CI-faithful gate + /final-review

**Files:**
- Modify: `CHANGELOG.md`, `CLAUDE.md`.
- Verify: C4 (`docs/c4/`), NOTICE.

- [ ] **Step 1: Docstrings** — ensure `delta_das_batch` / `delta_threat_suppression_batch` carry an `Examples` block (the public-API examples gate, `tests/test_public_api_examples.py`). The literal-block example in Task 3 Step 3 satisfies it (a ≥4-space-indented non-`>>>` block is accepted). Run:

`python -m pytest tests/test_public_api_examples.py -v`

- [ ] **Step 2: CHANGELOG entry** — add above the current top entry (version assigned at commit-prep):

```markdown
## [4.97.0] - gkdv arms batching (PR-Snnn, ADR-nnn)   <!-- numbers at commit-prep -->
### Added
- `silly_kicks.gkdv.delta_das_batch` / `delta_threat_suppression_batch`: batched arm entry points
  that make ONE accessible-space call per leg over a unit's scored frames (measured ~90× at unit
  scale; §1.1). Single-frame `delta_das` / `delta_threat_suppression` now delegate to the batch.
### Changed
- gkdv DAS batch pins attacking direction ONCE per unit (more robust; free — gkdv has no persisted
  output). An unscoreable frame reduces to NaN (`min_count=1`), never a fictional 0.0.
- No retrain, no re-materialize. accessible-space `[das]` extra. C4-free.
```

- [ ] **Step 3: CLAUDE.md gkdv bullet** — append to the GKDV entry: the batch arms + once-per-unit pin + the "mixed-batch NaN is a version-pinned accessible-space contract, guarded by `test_get_individual_das_mixed_batch...`, add_das shares the exposure" note. Keep it a durable contract, not narrative.

- [ ] **Step 4: ADR** — this cycle IS ADR-worthy (a public API addition + the once-per-unit direction-pin decision + the declined-pre-filter reasoning). Draft `docs/superpowers/adrs/ADR-0nn-gkdv-arms-batching.md` (number at commit-prep) from the template, recording: the batch-native amortization, once-per-unit pinning (and why it's free), no vectorized kernel (measured YAGNI), and the mixed-batch-contract guard decision.

- [ ] **Step 5: C4 verify** — no new action-coupled aggregator (the gkdv arms are already modelled). Confirm the C4 completeness gate is green; re-render only if it flags drift:

`python -m pytest tests/ -k c4 -v`

- [ ] **Step 6: Full CI-faithful gate**

```bash
python -m ruff check silly_kicks/ tests/ scripts/
python -m ruff format --check silly_kicks/ tests/ scripts/
python -m pyright
python -m pytest tests/ -m "not e2e" -v --tb=short
```
All must be clean. (accessible-space is installed here, so the `[das]` tests run; on a leg without it they `importorskip`.)

- [ ] **Step 7: /final-review** — run the final-review skill (code quality, doc drift, C4, ADR inventory). Then STOP and report; do NOT commit — the owner commits on explicit approval after assigning version/PR-S/ADR numbers from merged `origin/main`.

---

## Self-review notes (author)

- **Spec coverage:** §4.1 API → Tasks 3/4/6; §4.2 fail-loud key → Tasks 2/4; §4.3 min_count=1 + DasUnscoreableError + mixed-batch → Tasks 1/2/3; §4.4 threat loop → Task 4; §4.5 wrappers → Task 5; §4.6 alignment guard → Task 3; §5.1–5.8 oracles → Tasks 1–6; §7 ordering ← task order; §10 handoff-back is lakehouse-side (not implemented here, recorded in the spec).
- **Type consistency:** `_attacking_team_by_frame` (the ONE shared resolver, `_das_port`), `team_das_by_frame`, `_assert_legs_aligned`, `_frame_key_index` are named identically across tasks; `_FRAME_KEY` is the one key list. (Tests use `_KEY` for the same list — both are the same three columns.)
- **No placeholders** except the deliberately-deferred commit-prep numbers (version/PR-S/ADR) and the one measured-atol-if-needed in Task 3 Step 4 (a measurement instruction, not a TBD).
- **Not a commit boundary anywhere** — task order is review tractability only; the owner commits once at the end.
- **Open implementer choice flagged in Task 4:** the `fitted_xt_for_tests` helper + `_threat_unit(n)` fixture — lift both from `test_deterrent_keeper_gives_a_NEGATIVE_delta`; create the helpers only if none exists.

## Review log

- **Round-3 (plan) review — incorporated (2026-08-27).** Finding 1: Task 6 Step 1 is now a real `__all__` edit (import at `:19`, list at `:29`), not a comment. Finding 2: added the consequence test `test_once_per_unit_pin_is_stable_where_a_single_frame_would_flip` (verified numerically per-unit `-1.0` vs per-frame `+1.0`), retiring the dead `_GHOST_GK_X`; the mechanism test is kept alongside. Finding 3: verified `compute_threat_pc` reads no `team_in_possession` (possession-independent → no dead-ball raise); documented the inherent DAS/threat scoreability asymmetry + added `test_..._scores_a_dead_ball_unit_without_crashing` rather than a needless symmetric guard. Finding 4: unified to ONE `_das_port._attacking_team_by_frame` dict resolver consumed by both the DAS reduce and the threat loop.
- **Execution note (2026-08-27) — SB360 boundary-audit caught a label-alignment defect the plan's tests missed.** The full `-m "not e2e"` suite (not the plan's own test list) failed 4 ADR-053 SB360 tests for `gkdv.delta_das`: on the velocity-FULL leg it returned all-NaN. Root cause in `team_das_by_frame`: `ids_equal` returns a POSITIONAL fresh-RangeIndex mask (ADR-019), and `&`-ing it with `~out["is_ball"]` — which carries `out`'s NON-CONTIGUOUS index on a filtered frame slice — label-aligned to zero overlap → all-False → every attacking player dropped. The plan's Task-2/3 fixtures used a contiguous `pd.concat(ignore_index=True)` index and could not see it (the `feedback_test_fixtures_must_exercise_real_dtypes` class, extended to index shape). Fixed by combining the masks in numpy (`.to_numpy()` on both), and added `test_team_das_by_frame_survives_a_noncontiguous_index`. Lesson for future batch-reduce work: a filtered frame slice is the realistic shape; test with a non-contiguous index, not just `ignore_index=True`.
- **Round-4 (revised plan) review — incorporated (2026-08-27).** Both defects were introduced BY the round-3 test fixes. Defect 1 [would break CI]: the consequence test calls the real `pin_direction`, which requires accessible-space (`_das.py:533` `_import_accessible_space()`, `:551` `infer_playing_direction`) — my "pure / no accessible-space" claim was FALSE. Added `pytest.importorskip("accessible_space")` + corrected the docstring/lead-in to "real-scoring, skipped without [das]", and noted the load-bearing assertion is `d_flip_unit != d_flip_alone`. **Sibling fix the reviewer's defect implied:** the MECHANISM test's spy also called `real_pin` (same hazard); its spy now returns a SYNTHETIC direction so it stays genuinely structural/every-leg. Defect 2 [vacuous guard]: `test_team_das_by_frame_reduces_per_frame` asserted only `(out.dropna() > 0.0).all()`, which is trivially True on an all-NaN result (e.g. a `MultiIndex.map` tuple-dtype miss); added `assert out.notna().all()` first so the reduce test can actually fail.
