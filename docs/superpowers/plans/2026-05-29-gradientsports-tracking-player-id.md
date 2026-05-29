# Gradient Sports tracking player-id resolution (TF-24 PR-A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `silly_kicks.tracking.gradientsports.add_gradientsports_player_ids`, a pure helper that resolves GS tracking jersey numbers to the events SPADL `player_id`/`team_id` int space (via the roster), so GS tracking frames become joinable to GS events — fixing the silent "GS carrier accuracy = 0.0" bug.

**Architecture:** A standalone, additive, pandas-in/pandas-out helper run *before* `convert_to_frames` (the events dual-API "caller normalizes, converter converts" boundary). It maps `team_side`→`team_id` from kwargs, normalizes + dedupes the roster join key, vectorized-left-joins `(team_id, jersey)`→roster `player_id` (`Int64`, unmatched→`pd.NA` never `0`), derives `is_goalkeeper` from `positionGroupType=="GK"`, and returns a frozen `GradientsportsRosterReport`. Loud warnings (never raises) on degenerate match rate, duplicate roster keys, and zero-GK.

**Tech Stack:** Python 3.10+ (CI 3.10–3.12), pandas, numpy; pytest. Spec: `docs/superpowers/specs/2026-05-29-gradientsports-tracking-player-id-design.md`.

---

## Project conventions (read before starting)

- **ONE commit per branch, explicit approval first.** Per-task steps below do NOT commit; all staging + the single commit happen in the final task after `/final-review` and explicit user approval. Create the branch `pr-aXX-gradientsports-player-id` (or similar) off `main` as the first action.
- **Shift-left gates** (Task 10): `python -m ruff format --check .`, `python -m ruff check .`, `python -m pyright silly_kicks/` (full package), `python -m pytest tests/ -m "not e2e" -v --tb=short`.
- **Fact-check / e2e:** the env-gated real-data e2e (Task 9) is the ultimate proof; run it locally against the GS data before shipping if available. GS licence gate: **commit nothing GS-derived**.
- **No new ADR** (within ADR-004). No `@nan_safe_enrichment` (frame helper, not action-enrichment).

## File structure

- Modify: `silly_kicks/tracking/gradientsports.py` — add `GradientsportsRosterReport` (frozen dataclass) + `add_gradientsports_player_ids`.
- Modify: `silly_kicks/tracking/__init__.py` — re-export the helper + report; add to `__all__`.
- Modify: `tests/test_public_api_examples.py` — add `silly_kicks/tracking/gradientsports.py` to `_PUBLIC_MODULE_FILES`.
- Create: `tests/tracking/test_gradientsports_player_ids.py` — unit + synthetic-e2e (CI) tests.
- Create: `tests/tracking/test_gradientsports_player_ids_e2e.py` — env-gated real-data e2e.
- Modify (Task 11): `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, `TODO.md`.

---

### Task 1: `GradientsportsRosterReport` + happy-path helper

**Files:**
- Modify: `silly_kicks/tracking/gradientsports.py`
- Test: `tests/tracking/test_gradientsports_player_ids.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/tracking/test_gradientsports_player_ids.py`:

```python
"""TF-24 PR-A — GS tracking jersey->roster player-id resolution."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking.gradientsports import (
    GradientsportsRosterReport,
    add_gradientsports_player_ids,
)

HOME, AWAY = 366, 51


def _roster() -> pd.DataFrame:
    # team.id is string in raw GS; player.id is string; positionGroupType literal "GK".
    return pd.DataFrame(
        {
            "team_id": ["366", "366", "51", "51"],
            "shirt_number": ["8", "1", "10", "1"],
            "player_id": [8342, 8326, 940, 12],
            "position_group_type": ["AM", "GK", "FW", "GK"],
        }
    )


def _jersey_frames() -> pd.DataFrame:
    # 1 frame: home #8 (outfield), home #1 (GK), away #10 (outfield), away #1 (GK), ball.
    base = dict(game_id=1, period_id=1, frame_id=1, time_seconds=0.0, frame_rate=30.0,
                z=0.0, speed_native=1.0, ball_state="alive")
    rows = [
        {**base, "team_side": "home", "jersey_number": "8", "is_ball": False, "x_centered": 0.0, "y_centered": 0.0},
        {**base, "team_side": "home", "jersey_number": "1", "is_ball": False, "x_centered": -40.0, "y_centered": 0.0},
        {**base, "team_side": "away", "jersey_number": "10", "is_ball": False, "x_centered": 5.0, "y_centered": 2.0},
        {**base, "team_side": "away", "jersey_number": "1", "is_ball": False, "x_centered": 40.0, "y_centered": 0.0},
        {**base, "team_side": None, "jersey_number": None, "is_ball": True, "x_centered": 1.0, "y_centered": 1.0},
    ]
    return pd.DataFrame(rows)


class TestHappyPath:
    def test_join_and_dtypes(self):
        frames = _jersey_frames()
        out, report = add_gradientsports_player_ids(frames, _roster(), home_team_id=HOME, away_team_id=AWAY)

        assert str(out["player_id"].dtype) == "Int64"
        assert str(out["team_id"].dtype) == "Int64"
        assert out["is_goalkeeper"].dtype == bool
        # home #8 -> 8342, home #1 -> 8326, away #10 -> 940, away #1 -> 12, ball -> NA
        assert out.loc[0, "player_id"] == 8342 and out.loc[0, "team_id"] == HOME
        assert out.loc[1, "player_id"] == 8326 and bool(out.loc[1, "is_goalkeeper"]) is True
        assert out.loc[2, "player_id"] == 940 and out.loc[2, "team_id"] == AWAY
        assert pd.isna(out.loc[4, "player_id"]) and pd.isna(out.loc[4, "team_id"])
        assert bool(out.loc[0, "is_goalkeeper"]) is False
        assert isinstance(report, GradientsportsRosterReport)
        assert report.n_player_rows == 4 and report.n_matched == 4 and report.n_unmatched == 0
        assert report.n_duplicate_roster_keys == 0

    def test_does_not_mutate_input(self):
        frames = _jersey_frames()
        before = frames.copy()
        add_gradientsports_player_ids(frames, _roster(), home_team_id=HOME, away_team_id=AWAY)
        pd.testing.assert_frame_equal(frames, before)

    def test_row_count_preserved(self):
        frames = _jersey_frames()
        out, _ = add_gradientsports_player_ids(frames, _roster(), home_team_id=HOME, away_team_id=AWAY)
        assert len(out) == len(frames)

    def test_missing_required_columns_raise(self):
        with pytest.raises(ValueError, match="jersey_frames"):
            add_gradientsports_player_ids(pd.DataFrame({"is_ball": [True]}), _roster(), home_team_id=HOME, away_team_id=AWAY)
        with pytest.raises(ValueError, match="roster"):
            add_gradientsports_player_ids(_jersey_frames(), pd.DataFrame({"team_id": [1]}), home_team_id=HOME, away_team_id=AWAY)


class TestRowAlignment:
    """C2 guard: resolution must be positionally exact on shuffled, multi-frame input
    (the order-safe .map contract — a reorder would misalign every player_id)."""

    def test_per_row_correct_on_shuffled_input(self):
        frames = pd.concat([_jersey_frames(), _jersey_frames().assign(frame_id=2)], ignore_index=True)
        frames = frames.sample(frac=1.0, random_state=7).reset_index(drop=True)  # shuffle rows
        out, _ = add_gradientsports_player_ids(frames, _roster(), home_team_id=HOME, away_team_id=AWAY)
        # expected per-row id from (team_side, jersey) on the SHUFFLED frame, computed independently
        expected = {("home", "8"): 8342, ("home", "1"): 8326, ("away", "10"): 940, ("away", "1"): 12}
        for i in range(len(out)):
            if bool(out.loc[i, "is_ball"]):
                assert pd.isna(out.loc[i, "player_id"])
                continue
            want = expected[(out.loc[i, "team_side"], out.loc[i, "jersey_number"])]
            assert out.loc[i, "player_id"] == want, f"row {i} misaligned"
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/tracking/test_gradientsports_player_ids.py -v`
Expected: FAIL — `ImportError: cannot import name 'add_gradientsports_player_ids'`.

- [ ] **Step 3: Implement the dataclass + helper**

In `silly_kicks/tracking/gradientsports.py`, add near the top imports:

```python
import dataclasses
import warnings
```

Append after `convert_to_frames`:

```python
_FRAMES_REQUIRED: frozenset[str] = frozenset({"team_side", "jersey_number", "is_ball"})
_ROSTER_REQUIRED: frozenset[str] = frozenset({"team_id", "shirt_number", "player_id"})


@dataclasses.dataclass(frozen=True)
class GradientsportsRosterReport:
    """Audit of a :func:`add_gradientsports_player_ids` resolution.

    Attributes
    ----------
    n_player_rows : int
        Non-ball rows seen.
    n_matched : int
        Player rows whose ``(team_id, jersey_number)`` matched a roster entry.
    n_unmatched : int
        ``n_player_rows - n_matched``.
    unmatched_jerseys : frozenset[tuple[int, str]]
        Distinct ``(team_id, jersey_number)`` keys that did not match.
    roster_size : int
        Roster rows used for the join (after de-duplication).
    n_duplicate_roster_keys : int
        Duplicate ``(team_id, shirt_number)`` roster keys dropped (``keep="first"``).

    Examples
    --------
    >>> _, report = add_gradientsports_player_ids(frames, roster, home_team_id=366, away_team_id=51)
    >>> report.n_matched
    4
    """

    n_player_rows: int
    n_matched: int
    n_unmatched: int
    unmatched_jerseys: frozenset[tuple[int, str]]
    roster_size: int
    n_duplicate_roster_keys: int


def add_gradientsports_player_ids(
    jersey_frames: pd.DataFrame,
    roster: pd.DataFrame,
    *,
    home_team_id: int,
    away_team_id: int,
) -> tuple[pd.DataFrame, GradientsportsRosterReport]:
    """Resolve GS tracking jersey numbers to the events SPADL ``player_id`` space.

    Gradient Sports tracking frames carry only ``jerseyNum`` (+ a home/away split);
    GS events SPADL ``player_id`` is the integer roster ``player.id``. This helper
    joins ``(team_id, jersey_number)`` -> roster ``player_id`` so a tracking carrier
    is joinable to events. Run it BEFORE
    :func:`silly_kicks.tracking.gradientsports.convert_to_frames`.

    Parameters
    ----------
    jersey_frames : pd.DataFrame
        Long-form GS tracking rows. Required columns: ``team_side`` ("home"/"away";
        ``None`` for ball), ``jersey_number`` (object/string; ``None`` for ball),
        ``is_ball`` (bool). Other tracking columns are passed through untouched.
    roster : pd.DataFrame
        Required columns: ``team_id`` (coercible to int), ``shirt_number``
        (object/string), ``player_id`` (int). Optional ``position_group_type``
        (literal ``"GK"`` flags the goalkeeper).
    home_team_id, away_team_id : int
        The events SPADL ``int64`` team ids. ``team_side`` maps to these.

    Returns
    -------
    frames : pd.DataFrame
        Copy of ``jersey_frames`` with ``player_id`` (``Int64``; ``pd.NA`` for
        ball/unmatched), ``team_id`` (``Int64``), ``is_goalkeeper`` (bool) added.
    report : GradientsportsRosterReport

    Examples
    --------
    >>> frames, report = add_gradientsports_player_ids(
    ...     jersey_frames, roster, home_team_id=366, away_team_id=51
    ... )
    >>> frames, _conv_report = convert_to_frames(frames, home_team_id=366, home_team_start_left=True, output_convention="ltr")  # doctest: +SKIP
    """
    miss_f = _FRAMES_REQUIRED - set(jersey_frames.columns)
    if miss_f:
        raise ValueError(f"add_gradientsports_player_ids: jersey_frames missing columns: {sorted(miss_f)}")
    miss_r = _ROSTER_REQUIRED - set(roster.columns)
    if miss_r:
        raise ValueError(f"add_gradientsports_player_ids: roster missing columns: {sorted(miss_r)}")

    out = jersey_frames.copy()
    is_ball = out["is_ball"].astype(bool)
    is_player = ~is_ball

    # team_side -> team_id (Int64; ball / unknown side -> NA)
    side = out["team_side"].astype("string")
    team_id = pd.Series(pd.NA, index=out.index, dtype="Int64")
    team_id = team_id.mask(is_player & (side == "home"), home_team_id)
    team_id = team_id.mask(is_player & (side == "away"), away_team_id)
    out["team_id"] = team_id

    # roster lookup as a "team|shirt" -> value dict (raw jersey; Task 2 adds strip()).
    has_pos = "position_group_type" in roster.columns
    r = roster.copy()
    r["_team"] = pd.to_numeric(r["team_id"], errors="coerce").astype("Int64")
    r["_shirt"] = r["shirt_number"].astype("string")
    r["_pid"] = pd.to_numeric(r["player_id"], errors="coerce").astype("Int64")
    r["_is_gk"] = (r["position_group_type"].astype("string") == "GK") if has_pos else False
    r["_key"] = r["_team"].astype("string").str.cat(r["_shirt"], sep="|")
    # NOTE: dict(zip(...)) keeps the LAST value on a duplicate key; Task 3 dedupes r with
    # keep="first" BEFORE this so the first roster entry wins (and rows can't explode).
    pid_map = dict(zip(r["_key"].to_list(), r["_pid"].to_list()))
    gk_map = dict(zip(r["_key"].to_list(), r["_is_gk"].to_list()))

    # ORDER-SAFE resolution (C2): elementwise .map on a same-index key Series — positionally
    # exact by construction (no merge / no index reassignment / no reorder risk). A frame row
    # with NA team or NA jersey (ball rows) yields an NA key -> map miss -> pd.NA.
    frame_key = out["team_id"].astype("string").str.cat(out["jersey_number"].astype("string"), sep="|")
    out["player_id"] = frame_key.map(pid_map).astype("Int64")
    out["is_goalkeeper"] = frame_key.map(gk_map).fillna(False).astype("bool")

    matched = is_player & out["player_id"].notna()
    n_player = int(is_player.sum())
    n_matched = int(matched.sum())
    unmatched_mask = is_player & out["player_id"].isna() & out["team_id"].notna() & out["jersey_number"].notna()
    unmatched = {
        (int(t), str(j))
        for t, j in zip(out.loc[unmatched_mask, "team_id"], out.loc[unmatched_mask, "jersey_number"])
    }

    return out, GradientsportsRosterReport(
        n_player_rows=n_player,
        n_matched=n_matched,
        n_unmatched=n_player - n_matched,
        unmatched_jerseys=frozenset(unmatched),
        roster_size=len(r),
        n_duplicate_roster_keys=0,
    )

**Final statement order (state once — implementers must converge to this; minor note):** the
finished helper executes in this order: (1) validate required columns; (2) `team_side`→`team_id`;
(3) build deduped roster (Task 3 dedupe) + `pid_map`/`gk_map`; (4) `frame_key` + `.map` →
`player_id`/`is_goalkeeper`; (5) compute `n_player`/`n_matched`/`unmatched`; (6) emit warnings
(absent-`position_group_type` at step 3; **zero-GK** and **degenerate-match-rate** after step 5,
once counts exist); (7) return frames + report. Tasks 2–5 insert into this order; do not reorder
blind.
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/tracking/test_gradientsports_player_ids.py::TestHappyPath -v`
Expected: PASS (4 tests).

---

### Task 2: Join-key normalization (M1)

**Files:** Modify `silly_kicks/tracking/gradientsports.py`; Test `tests/tracking/test_gradientsports_player_ids.py`

- [ ] **Step 1: Write failing test** — append:

```python
class TestKeyNormalization:
    def test_format_drift_still_matches(self):
        frames = _jersey_frames()
        roster = _roster()
        # roster shirt as zero-padded string + int-typed; frames jersey with whitespace
        roster["shirt_number"] = ["08", "1", 10, " 1 "]  # mixed str/int/padded/space
        frames.loc[0, "jersey_number"] = " 8 "
        out, report = add_gradientsports_player_ids(frames, roster, home_team_id=HOME, away_team_id=AWAY)
        # "08" != "8" by raw string, but normalization is strip()-only (NOT zero-strip),
        # so "08" stays unmatched; " 8 " strips to "8" but roster has "08" -> unmatched.
        # The match we assert: away #10 (int 10 -> "10") and away #1 (" 1 " -> "1").
        assert out.loc[2, "player_id"] == 940   # away "10" matches int 10 stripped->"10"
        assert out.loc[3, "player_id"] == 12     # away " 1 " matches "1"
```

Note for the implementer: GS jerseys are not zero-padded in practice (`"8"`, `"10"`); normalization is `str().strip()` only — it reconciles whitespace and int-vs-string typing, NOT zero-padding. The test pins exactly that contract (whitespace + int coercion match; zero-pad does not). Adjust the asserted ids if the fixture differs; the load-bearing assertion is that whitespace/`int` forms join.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/tracking/test_gradientsports_player_ids.py::TestKeyNormalization -v`
Expected: FAIL — `" 8 "`/`10`(int)/`" 1 "` don't match without normalization.

- [ ] **Step 3: Implement** — add `.str.strip()` to the jersey key on both sides (the roster
key build and the per-frame key build). Replace:

```python
    r["_shirt"] = r["shirt_number"].astype("string")
```
with
```python
    r["_shirt"] = r["shirt_number"].astype("string").str.strip()
```
and replace:
```python
    frame_key = out["team_id"].astype("string").str.cat(out["jersey_number"].astype("string"), sep="|")
```
with
```python
    frame_key = out["team_id"].astype("string").str.cat(out["jersey_number"].astype("string").str.strip(), sep="|")
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/tracking/test_gradientsports_player_ids.py::TestKeyNormalization -v`
Expected: PASS.

---

### Task 3: Roster key uniqueness — no row explosion (N1)

**Files:** Modify `silly_kicks/tracking/gradientsports.py`; Test file.

- [ ] **Step 1: Write failing test** — append:

```python
class TestRosterUniqueness:
    def test_duplicate_key_no_explosion_and_warns(self):
        frames = _jersey_frames()
        roster = _roster()
        # inject a duplicate (team_id, shirt_number) for home #8
        dup = pd.DataFrame({"team_id": ["366"], "shirt_number": ["8"], "player_id": [9999], "position_group_type": ["AM"]})
        roster = pd.concat([roster, dup], ignore_index=True)
        with pytest.warns(UserWarning, match="duplicate"):
            out, report = add_gradientsports_player_ids(frames, roster, home_team_id=HOME, away_team_id=AWAY)
        assert len(out) == len(frames)            # no left-join explosion
        assert report.n_duplicate_roster_keys == 1
        assert out.loc[0, "player_id"] == 8342    # keep="first" -> original, not 9999
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/tracking/test_gradientsports_player_ids.py::TestRosterUniqueness -v`
Expected: FAIL — row count doubles for home #8 (explosion) and `n_duplicate_roster_keys == 0`.

- [ ] **Step 3: Implement** — after building `r["_is_gk"]` and **before** `r["_key"] = ...`
(so `dict(zip(...))`, which otherwise keeps the LAST dup, is built from the deduped frame and
thus honours `keep="first"`), insert:

```python
    _dup = r.duplicated(subset=["_team", "_shirt"], keep="first")
    n_duplicate_roster_keys = int(_dup.sum())
    if n_duplicate_roster_keys:
        warnings.warn(
            f"gradientsports roster has {n_duplicate_roster_keys} duplicate "
            "(team_id, shirt_number) key(s); keeping first",
            UserWarning,
            stacklevel=2,
        )
        r = r[~_dup]
```
Then in the returned report replace `n_duplicate_roster_keys=0` with
`n_duplicate_roster_keys=n_duplicate_roster_keys`. (`roster_size=len(r)` already reflects the
deduped roster.)

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/tracking/test_gradientsports_player_ids.py::TestRosterUniqueness -v`
Expected: PASS.

---

### Task 4: GK vocabulary pin + zero-GK warning + absent-column (M3, N2)

**Files:** Modify helper; Test file.

- [ ] **Step 1: Write failing tests** — append:

```python
class TestGoalkeeper:
    def test_gk_vocab_drift_all_false_and_warns(self):
        frames = _jersey_frames()
        roster = _roster()
        roster["position_group_type"] = ["AM", "Goalkeeper", "FW", "Goalkeeper"]  # not literal "GK"
        with pytest.warns(UserWarning, match="no GK found"):
            out, _ = add_gradientsports_player_ids(frames, roster, home_team_id=HOME, away_team_id=AWAY)
        assert not out["is_goalkeeper"].any()  # pins the exact-"GK" literal

    def test_position_column_absent_all_false_and_warns(self):
        frames = _jersey_frames()
        roster = _roster().drop(columns=["position_group_type"])
        with pytest.warns(UserWarning, match="position_group_type"):
            out, _ = add_gradientsports_player_ids(frames, roster, home_team_id=HOME, away_team_id=AWAY)
        assert not out["is_goalkeeper"].any()
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/tracking/test_gradientsports_player_ids.py::TestGoalkeeper -v`
Expected: FAIL — no warnings emitted.

- [ ] **Step 3: Implement** — replace the `_is_gk` / `has_pos` lines with:

```python
    has_pos = "position_group_type" in roster.columns
    if not has_pos:
        warnings.warn(
            "gradientsports roster has no 'position_group_type' column; is_goalkeeper will be all-False",
            UserWarning,
            stacklevel=2,
        )
```
and after the merge + before computing counts, add the zero-GK warning:

```python
    if has_pos and n_player and not bool(out["is_goalkeeper"].any()):
        observed = sorted({str(v) for v in roster["position_group_type"].dropna().unique()})
        warnings.warn(
            f"gradientsports: no GK found (positionGroupType values were {observed}); expected literal 'GK'",
            UserWarning,
            stacklevel=2,
        )
```
(Compute `n_player`/the `out["is_goalkeeper"]` assignment before this block — reorder so counts precede the warning.)

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/tracking/test_gradientsports_player_ids.py::TestGoalkeeper -v`
Expected: PASS (2 tests).

---

### Task 5: Degenerate match-rate warning + unmatched→NA (M2)

**Files:** Modify helper; Test file.

- [ ] **Step 1: Write failing tests** — append:

```python
class TestUnmatchedAndDegenerate:
    def test_unmatched_jersey_is_na_not_zero(self):
        frames = _jersey_frames()
        frames.loc[0, "jersey_number"] = "99"  # no roster entry
        out, report = add_gradientsports_player_ids(frames, _roster(), home_team_id=HOME, away_team_id=AWAY)
        assert pd.isna(out.loc[0, "player_id"])  # NA, never 0
        assert (HOME, "99") in report.unmatched_jerseys
        assert report.n_unmatched == 1

    def test_degenerate_match_rate_warns(self):
        frames = _jersey_frames()
        # wrong team-id space: pass team ids that match nothing in the roster
        with pytest.warns(UserWarning, match="unmatched"):
            out, report = add_gradientsports_player_ids(frames, _roster(), home_team_id=999, away_team_id=888)
        assert report.n_matched == 0

    def test_healthy_rate_no_degenerate_warning(self):
        frames = _jersey_frames()
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any UserWarning -> failure
            add_gradientsports_player_ids(frames, _roster(), home_team_id=HOME, away_team_id=AWAY)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/tracking/test_gradientsports_player_ids.py::TestUnmatchedAndDegenerate -v`
Expected: FAIL — `test_degenerate_match_rate_warns` emits no warning (and the unmatched/NA assertions already pass from Task 1, which is fine).

- [ ] **Step 3: Implement** — after computing `n_player`/`n_matched`, add:

```python
    if n_player and (n_matched == 0 or (n_player - n_matched) / n_player >= 0.5):
        warnings.warn(
            f"gradientsports player-id resolution matched {n_matched}/{n_player} player rows "
            "(>=50% unmatched); check team-id space / roster alignment",
            UserWarning,
            stacklevel=2,
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/tracking/test_gradientsports_player_ids.py::TestUnmatchedAndDegenerate -v`
Expected: PASS (3 tests).

---

### Task 6: Re-export + public-API Examples gate

**Files:** Modify `silly_kicks/tracking/__init__.py`, `tests/test_public_api_examples.py`; Test file.

- [ ] **Step 1: Write failing tests** — append to `tests/tracking/test_gradientsports_player_ids.py`:

```python
def test_reexported_from_tracking():
    import silly_kicks.tracking as tk
    assert hasattr(tk, "add_gradientsports_player_ids")
    assert hasattr(tk, "GradientsportsRosterReport")
    assert "add_gradientsports_player_ids" in tk.__all__
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/tracking/test_gradientsports_player_ids.py::test_reexported_from_tracking -v`
Expected: FAIL — attribute / `__all__` missing.

- [ ] **Step 3: Implement**

In `silly_kicks/tracking/__init__.py`, add an import (next to the other `from .X import` lines, alphabetical-ish):

```python
from .gradientsports import GradientsportsRosterReport, add_gradientsports_player_ids
```
and add both names to `__all__` (keep it sorted with the surrounding entries):

```python
    "GradientsportsRosterReport",
    "add_gradientsports_player_ids",
```

In `tests/test_public_api_examples.py`, add to `_PUBLIC_MODULE_FILES`:

```python
    "silly_kicks/tracking/gradientsports.py",
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/tracking/test_gradientsports_player_ids.py::test_reexported_from_tracking tests/test_public_api_examples.py -v`
Expected: PASS. The pre-existing `convert_to_frames` already has an `Examples` block (verified, `gradientsports.py:92`), so adding the file to the gate is a no-op for it; the gate then only requires `Examples` on the new helper + `GradientsportsRosterReport` (both written in Task 1). If the gate still flags a symbol, add an `Examples` block to it.

---

### Task 7: Synthetic end-to-end join (H1) + team non-degeneracy (H2b)

**Files:** Test `tests/tracking/test_gradientsports_player_ids.py`.

- [ ] **Step 1: Write the test** — append:

```python
class TestSyntheticEndToEndJoin:
    """In-CI proxy (H1): the resolved Int64 player_id must join BY VALUE to an
    INDEPENDENTLY-constructed int64 events SPADL id and yield nonzero carrier accuracy.
    NON-CIRCULAR (C1): action player_id/team_id come from the KNOWN fixture geometry +
    roster constants below, NOT from infer_ball_carrier's output — so a wrong resolution
    (e.g. string ids, wrong int) breaks the join and fails the test. Limit (N4): both ids
    still trace to the same synthetic roster, so this proves join MECHANICS only; real
    id-derivation alignment is the env-gated real-data e2e (Task 8)."""

    # Independent ground-truth roster ids (match `_roster()`):
    HOME8, HOME1, AWAY10, AWAY1 = 8342, 8326, 940, 12

    def _fixture(self, n_frames=6):
        # Carrier alternates by geometry: even frames -> ball ON home #8 (HOME carries);
        # odd frames -> ball ON away #10 (AWAY carries). Others are far. Unambiguous.
        roster = _roster()
        rows, carrier_pid, carrier_team = [], {}, {}
        for fid in range(1, n_frames + 1):
            home_carries = fid % 2 == 0
            carrier_pid[fid] = self.HOME8 if home_carries else self.AWAY10
            carrier_team[fid] = HOME if home_carries else AWAY
            ball_x = -10.0 if home_carries else 10.0
            base = dict(game_id=1, period_id=1, frame_id=fid, time_seconds=fid * 0.1,
                        frame_rate=30.0, z=0.0, speed_native=0.0, ball_state="alive")
            rows += [
                {**base, "team_side": "home", "jersey_number": "8", "is_ball": False, "x_centered": -10.0, "y_centered": 0.0},
                {**base, "team_side": "home", "jersey_number": "1", "is_ball": False, "x_centered": -45.0, "y_centered": 0.0},
                {**base, "team_side": "away", "jersey_number": "10", "is_ball": False, "x_centered": 10.0, "y_centered": 0.0},
                {**base, "team_side": "away", "jersey_number": "1", "is_ball": False, "x_centered": 45.0, "y_centered": 0.0},
                {**base, "team_side": None, "jersey_number": None, "is_ball": True, "x_centered": ball_x, "y_centered": 0.0},
            ]
        return pd.DataFrame(rows), roster, carrier_pid, carrier_team

    def test_independent_join_nonzero_accuracy_and_team_space(self):
        from silly_kicks.tracking import infer_ball_carrier, link_actions_to_frames
        from silly_kicks.tracking.gradientsports import convert_to_frames

        jersey_frames, roster, carrier_pid, carrier_team = self._fixture()
        resolved, _ = add_gradientsports_player_ids(jersey_frames, roster, home_team_id=HOME, away_team_id=AWAY)
        frames, _ = convert_to_frames(resolved, home_team_id=HOME, home_team_start_left=True, output_convention="ltr")
        carrier = infer_ball_carrier(frames)

        # INDEPENDENT events actions per frame: one by the known carrier (sameteam=True) and
        # one by a known opponent (sameteam=False). ids are roster CONSTANTS, not read from
        # `carrier` -> the join is a genuine cross-check, and sameteam is genuinely mixed.
        acts = []
        for fid, pid in carrier_pid.items():
            tid = carrier_team[fid]
            opp_pid, opp_tid = (self.AWAY1, AWAY) if tid == HOME else (self.HOME1, HOME)
            acts.append(dict(game_id=1, period_id=1, time_seconds=fid * 0.1, team_id=tid, player_id=pid, type_name="pass", is_carrier=True))
            acts.append(dict(game_id=1, period_id=1, time_seconds=fid * 0.1, team_id=opp_tid, player_id=opp_pid, type_name="pass", is_carrier=False))
        actions = pd.DataFrame(acts)
        actions["action_id"] = np.arange(len(actions))
        actions = actions.astype({"team_id": "int64", "player_id": "int64", "period_id": "int64"})

        pointers, _ = link_actions_to_frames(actions, frames)
        linked = (
            actions.merge(pointers[["action_id", "frame_id"]], on="action_id")
            .merge(
                carrier[["game_id", "period_id", "frame_id", "ball_carrier_player_id", "ball_carrier_team_id"]],
                on=["game_id", "period_id", "frame_id"], how="left",
            )
            .dropna(subset=["ball_carrier_player_id"])
        )

        # H1: carrier-actor actions match the inferred carrier by VALUE (independent int ids).
        carr_rows = linked[linked["is_carrier"]]
        acc = (carr_rows["player_id"] == carr_rows["ball_carrier_player_id"]).mean()
        assert acc > 0  # clean fixture -> expect 1.0; > 0 is the regression guard

        # H2b: team-id space correct + non-degenerate. Resolved carrier team ids are a
        # non-empty SUBSET of the events space {HOME, AWAY} (non-flaky -- does NOT require
        # carrier alternation), and the carrier/opponent action mix makes sameteam structurally
        # ~0.5 -- the robust non-degeneracy proof, independent of infer_ball_carrier (N2).
        carrier_teams = {int(t) for t in carr_rows["ball_carrier_team_id"].dropna().unique()}
        assert carrier_teams and carrier_teams <= {HOME, AWAY}
        sameteam = linked["team_id"] == linked["ball_carrier_team_id"]
        assert 0 < sameteam.mean() < 1
```

Implementer note: this exercises the real `infer_ball_carrier`/`link_actions_to_frames` API
(`ball_carrier_player_id`/`ball_carrier_team_id` are the documented `infer_ball_carrier`
outputs). Align column names to the live signature if they differ; the load-bearing,
**non-circular** asserts are `acc > 0` (independent ids), `{…} == {HOME, AWAY}`, and
`0 < sameteam.mean() < 1`. If `infer_ball_carrier`'s hysteresis resists the alternation,
increase the per-frame ball separation or `n_frames` so each frame's carrier is unambiguous.

- [ ] **Step 2: Run**

Run: `python -m pytest tests/tracking/test_gradientsports_player_ids.py::TestSyntheticEndToEndJoin -v`
Expected: PASS. (If `acc == 0`, the resolved-id↔events-id value join is broken — that is the exact bug this PR fixes; debug the dtype/value path before proceeding.)

---

### Task 8: Real-data e2e (env-gated)

**Files:** Create `tests/tracking/test_gradientsports_player_ids_e2e.py`.

- [ ] **Step 1: Write the e2e test**

**Data source — gated mock API, NO local paths (per the data-access correction).** GS WC2022
is **owner-tier (private)** on the pining-for-the-data mock provider API (Bearer token → 302 →
presigned S3). The e2e fetches artifacts at runtime via two env vars and **skips** if either is
unset — no hardcoded local paths, nothing GS-derived committed:
- `PINING_FOR_THE_DATA_TOKEN` — owner-tier bearer (GS is private-tier; the public token returns 0). **Set on the dev machine** — the e2e runs for real, not skipped.
- `PINING_API_URL` — optional override of the mock API base URL (defaults to the deployed instance).

(Confirm the env-var names match your shell's existing values; align the artifact path shape —
`/gradientsports/matches/{id}/{artifact}` per `scripts/verify_gradient_load.py` — if your API
base includes a stage prefix like `/v1`.)

```python
"""TF-24 PR-A e2e: real GS data proves carrier accuracy 0.0 -> nonzero (the bug fix).

Sources GS WC2022 (owner-tier / private) from the pining-for-the-data mock provider API.
Env-gated (skips in CI and whenever creds are absent). NO local paths; commits nothing
GS-derived (licence gate)."""

from __future__ import annotations

import bz2
import json
import os
import urllib.error
import urllib.request

import pandas as pd
import pytest

# API base is the deployed mock-provider URL (NOT a secret; includes the /v1 stage); the OWNER
# token is read from the env (never hardcoded -- it is the gated secret). Override the URL via
# PINING_API_URL if the deployment moves.
_API = os.environ.get("PINING_API_URL", "https://ozqgk9a3ji.execute-api.us-east-1.amazonaws.com/v1")
_TOKEN = os.environ.get("PINING_FOR_THE_DATA_TOKEN")  # owner-tier bearer; GS is private
_PROVIDER = "gradientsports"


def _get_json(path: str) -> object:
    req = urllib.request.Request(f"{_API}{path}", headers={"Authorization": f"Bearer {_TOKEN}"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _fetch_artifact(match_id: str, artifact: str) -> bytes:
    """Two-step gated download (mirrors scripts/verify_gradient_load.py): GET the API with the
    bearer -> 302 -> GET the presigned S3 URL WITHOUT the bearer (S3 rejects double-auth)."""
    path = f"/{_PROVIDER}/matches/{match_id}/{artifact}"

    class _NoFollow(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    api_req = urllib.request.Request(f"{_API}{path}", headers={"Authorization": f"Bearer {_TOKEN}"})
    try:
        with urllib.request.build_opener(_NoFollow).open(api_req, timeout=30) as resp:
            return resp.read()  # direct 200 (unlikely for the 302 path)
    except urllib.error.HTTPError as e:
        if e.code != 302:
            raise
        location = e.headers["Location"]
    with urllib.request.urlopen(urllib.request.Request(location), timeout=120) as resp:
        return resp.read()


def _read_jsonl(raw: bytes) -> list[dict]:
    text = bz2.decompress(raw).decode("utf-8") if raw[:2] == b"BZ" else raw.decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


@pytest.mark.e2e
@pytest.mark.skipif(not _TOKEN, reason="PINING_FOR_THE_DATA_TOKEN not set")
def test_real_gs_carrier_accuracy_nonzero():
    from silly_kicks.tracking import infer_ball_carrier, link_actions_to_frames
    from silly_kicks.tracking.gradientsports import add_gradientsports_player_ids, convert_to_frames

    matches = _get_json(f"/{_PROVIDER}/matches")["matches"]  # type: ignore[index]
    match_id = str(matches[0]["id"])

    meta = json.loads(_fetch_artifact(match_id, "metadata"))
    roster_raw = json.loads(_fetch_artifact(match_id, "roster"))
    events_raw = json.loads(_fetch_artifact(match_id, "events"))
    frames_raw = _read_jsonl(_fetch_artifact(match_id, "tracking"))[:3000]  # slice for runtime

    home_team_id = int(meta["homeTeam"]["id"])
    away_team_id = int(meta["awayTeam"]["id"])
    roster = pd.DataFrame(
        {
            "team_id": [int(r["team"]["id"]) for r in roster_raw],
            "shirt_number": [str(r["shirtNumber"]) for r in roster_raw],
            "player_id": [int(r["player"]["id"]) for r in roster_raw],
            "position_group_type": [r.get("positionGroupType") for r in roster_raw],
        }
    )

    # flatten tracking JSONL -> jersey-keyed long form (homePlayers/awayPlayers/balls)
    rows = []
    for fr in frames_raw:
        base = dict(game_id=int(match_id), period_id=int(fr["period"]), frame_id=int(fr["frameNum"]),
                    time_seconds=float(fr.get("periodGameClockTime", 0.0)), frame_rate=29.97,
                    z=0.0, speed_native=float("nan"), ball_state="alive")
        for side, key in (("home", "homePlayers"), ("away", "awayPlayers")):
            for p in fr.get(key, []):
                rows.append({**base, "team_side": side, "jersey_number": str(p["jerseyNum"]),
                             "is_ball": False, "x_centered": float(p["x"]), "y_centered": float(p["y"])})
        for b in fr.get("balls", []):
            rows.append({**base, "team_side": None, "jersey_number": None, "is_ball": True,
                         "x_centered": float(b["x"]), "y_centered": float(b["y"])})
    jersey_frames = pd.DataFrame(rows)

    resolved, report = add_gradientsports_player_ids(jersey_frames, roster, home_team_id=home_team_id, away_team_id=away_team_id)
    assert report.n_matched > 0
    frames, _ = convert_to_frames(resolved, home_team_id=home_team_id,
                                  home_team_start_left=bool(meta.get("homeTeamStartLeft", True)),
                                  output_convention="ltr")
    carrier = infer_ball_carrier(frames)
    assert len(carrier.dropna(subset=["ball_carrier_player_id"])) > 0
    # supporting: resolved carrier ids live in the events int player.id space
    assert carrier["ball_carrier_player_id"].dropna().isin(roster["player_id"]).mean() > 0

    # --- LOAD-BEARING proof (C3): the real events<->tracking carrier join. Use the RAW on-ball
    # event actor ids (gameEvents.playerId IS the events int player_id space -- the GS events
    # converter passes it straight through, gradientsports.py:421), so no full SPADL conversion
    # is needed. Align the gameEvents/possessionEvents field paths to the raw GS schema
    # (reference_pff_data_local secs 4-5) if they differ.
    on_ball_types = {"PA", "CR", "SH", "BC"}  # actor == ball carrier for these
    acts = []
    for ev in events_raw:
        ge = ev.get("gameEvents") or {}
        pe = ev.get("possessionEvents") or {}
        pid, tid, per, t = ge.get("playerId"), ge.get("teamId"), ge.get("period"), ge.get("startGameClock")
        if ge.get("gameEventType") != "OTB" or pe.get("possessionEventType") not in on_ball_types:
            continue
        if pid is None or per is None or t is None:
            continue
        acts.append(dict(action_id=len(acts), game_id=int(match_id), period_id=int(per),
                         time_seconds=float(t), team_id=int(tid) if tid is not None else 0,
                         player_id=int(pid), type_name="pass"))
    actions = pd.DataFrame(acts)
    # keep only actions inside the loaded frame window so they link
    fmax = frames.groupby("period_id")["time_seconds"].max().to_dict()
    actions = actions[actions.apply(lambda r: r["time_seconds"] <= fmax.get(r["period_id"], -1), axis=1)]
    assert len(actions) > 0, "no on-ball events fell inside the loaded tracking window"

    pointers, _ = link_actions_to_frames(actions, frames)
    linked = (
        actions.merge(pointers[["action_id", "frame_id"]], on="action_id")
        .merge(carrier[["game_id", "period_id", "frame_id", "ball_carrier_player_id"]],
               on=["game_id", "period_id", "frame_id"], how="left")
        .dropna(subset=["ball_carrier_player_id"])
    )
    accuracy = (linked["player_id"] == linked["ball_carrier_player_id"]).mean()
    assert accuracy > 0.0, f"GS carrier accuracy {accuracy} -- the 0.0 regression is NOT fixed"
```

- [ ] **Step 2: Run for REAL (the owner token IS available — do NOT let it skip)**

`PINING_FOR_THE_DATA_TOKEN` is set on the dev machine, so this test MUST actually execute (not
skip) and pass — it is the **load-bearing empirical proof** of the fix and a **hard ship gate**.

Run: `python -m pytest tests/tracking/test_gradientsports_player_ids_e2e.py -v -m e2e`
Expected: PASS (1 test; **not** skipped). If it reports `skipped`, the token env var is not
visible to the test process — fix that before proceeding (don't ship on a skipped proof).
Implementer notes: (1) **no local paths / no hardcoded token** — URL default is the deployed
mock API, token from `PINING_FOR_THE_DATA_TOKEN`; never commit the token value. (2) The match
ids come from the live `/{provider}/matches` list (API ids differ from raw filenames). (3) Align
the artifact path shape and the `gameEvents`/`possessionEvents` field access + on-ball
`possessionEventType` set to the live API / raw GS schema if they differ (`reference_pff_data_local`).
(4) The load-bearing assertion is the events-join `accuracy > 0`.

---

### Task 9: Shift-left gate sweep

**Files:** none (verification).

- [ ] **Step 1: Format** — `python -m ruff format --check silly_kicks/tracking/gradientsports.py silly_kicks/tracking/__init__.py tests/tracking/test_gradientsports_player_ids.py tests/tracking/test_gradientsports_player_ids_e2e.py tests/test_public_api_examples.py` → no reformat (else `ruff format` then recheck).
- [ ] **Step 2: Lint** — `python -m ruff check silly_kicks/ tests/tracking/test_gradientsports_player_ids.py tests/tracking/test_gradientsports_player_ids_e2e.py` → `All checks passed!`
- [ ] **Step 3: Types** — `python -m pyright silly_kicks/` → 0 errors.
- [ ] **Step 4: Full non-e2e suite** — `python -m pytest tests/ -m "not e2e" -v --tb=short` → all pass.
- [ ] **Step 5: e2e for REAL (required gate — token available)** — `python -m pytest tests/tracking/test_gradientsports_player_ids_e2e.py -m e2e -v` → **must PASS, not skip** (`PINING_FOR_THE_DATA_TOKEN` is set). A `skipped` result means the token isn't reaching the test process — resolve before ship. This passing run is the empirical proof the GS carrier=0.0 bug is fixed on real data.

---

### Task 10: Version bump + CHANGELOG + TODO + docs

**Files:** `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, `TODO.md`.

- [ ] **Step 1: Version (collision check first)** — re-read the live `version =` in `pyproject.toml`. This PR is a **minor** (additive public helper). Assuming current is `3.26.0`, bump both `pyproject.toml` and `silly_kicks/__init__.py` to `3.27.0`. If `main`/an open PR already took `3.27.0`, use the next free minor.
- [ ] **Step 2: CHANGELOG** — insert above the current top entry:

```markdown
## [3.27.0] — <ship date>

### Added
- **`silly_kicks.tracking.gradientsports.add_gradientsports_player_ids`** — resolves GS
  tracking jersey numbers to the events SPADL `player_id`/`team_id` int space via the roster
  (`(team_id, jersey) → roster player.id`, `Int64`, unmatched → `pd.NA`), with
  `is_goalkeeper` from `positionGroupType == "GK"` and a `GradientsportsRosterReport`. Run it
  before `convert_to_frames`. Fixes a silent failure where GS tracking carriers (string/
  jersey-derived ids) could not join GS events SPADL (`int64` player_id) — GS ball-carrier /
  DAS / team-in-possession features were silently broken. Loud `UserWarning`s on a degenerate
  match rate, duplicate roster keys, or zero GK; never raises (ADR-003). (TF-24 PR-A)
```

- [ ] **Step 3: TODO grooming** — if TODO.md has a TF-24 row, leave the harness (PR-B) portion and note PR-A (GS player-id helper) shipped; do not delete the TF-24 row (PR-B still pending). Add a one-line CHANGELOG-is-the-record note only if a stale PR-A row exists.
- [ ] **Step 4: Verify** — `grep -n "3.27.0" pyproject.toml silly_kicks/__init__.py CHANGELOG.md` all present.

---

### Task 11: Final review + single commit

**Files:** all changed.

- [ ] **Step 1: `/final-review`** (mandatory) — address findings. C4: this is a converter-adjacent helper; verify the diagram is still accurate (no new container/component) — likely no change.
- [ ] **Step 2: Confirm tree** — `git status` / `git diff --stat`. Expected: `silly_kicks/tracking/gradientsports.py`, `silly_kicks/tracking/__init__.py`, `tests/test_public_api_examples.py`, `tests/tracking/test_gradientsports_player_ids.py`, `tests/tracking/test_gradientsports_player_ids_e2e.py`, `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, (`TODO.md` if touched), + the spec + this plan under `docs/superpowers/`. No `.hypothesis/`, no GS-derived data.
- [ ] **Step 3: Explicit user approval, then ONE commit:**

```bash
git add silly_kicks/tracking/gradientsports.py silly_kicks/tracking/__init__.py \
  tests/test_public_api_examples.py tests/tracking/test_gradientsports_player_ids.py \
  tests/tracking/test_gradientsports_player_ids_e2e.py pyproject.toml silly_kicks/__init__.py \
  CHANGELOG.md docs/superpowers/specs/2026-05-29-gradientsports-tracking-player-id-design.md \
  docs/superpowers/plans/2026-05-29-gradientsports-tracking-player-id.md
# (add TODO.md if changed)
git commit -m "$(cat <<'EOF'
feat(tracking): GS jersey->roster player-id resolution -- silly-kicks 3.27.0 (TF-24 PR-A)

add_gradientsports_player_ids resolves GS tracking jerseyNum -> events SPADL int
player_id/team_id via the roster (Int64, unmatched->pd.NA never 0), + is_goalkeeper
from positionGroupType=="GK" + a graceful GradientsportsRosterReport. Fixes the silent
GS carrier-accuracy=0.0 bug (string/jersey carrier id couldn't join int events id).
Precursor to the TF-24 Optuna calibration harness (PR-B).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```
- [ ] **Step 4: Push + PR + wait CI green** only when asked; never tag before main CI green.

---

## Self-review (completed by plan author)

**Spec coverage:** §3 API (helper + report + contract) → Tasks 1,3,4; §3 behaviour steps 1-2 (team_side, normalize) → Tasks 1-2; step 3 (uniqueness N1) → Task 3; step 4 (Int64/NA-not-0) → Tasks 1,5; step 5 (GK pin + zero-GK N2) → Task 4; step 6 (report fields) → Tasks 1,3; step 7 (degenerate warn M2) → Task 5; §4 unit tests → Tasks 1-6; §4 synthetic E2E (H1) + H2b → Task 7; §4 real e2e → Task 8; §5 housekeeping/version → Tasks 9-10; re-export + examples gate → Task 6. All covered.

**Placeholder scan:** the two "align field/column names to the live signature" notes (Tasks 7,8) are deliberate — they point at real APIs (`infer_ball_carrier` output columns, GS raw-JSON field names per `reference_pff_data_local`) whose exact spelling is verified at implementation time; the load-bearing assertions are concrete. All shipped code is complete.

**Type/name consistency:** `add_gradientsports_player_ids(jersey_frames, roster, *, home_team_id, away_team_id) -> (frames, GradientsportsRosterReport)` and the report's six fields (`n_player_rows`, `n_matched`, `n_unmatched`, `unmatched_jerseys`, `roster_size`, `n_duplicate_roster_keys`) are identical across the dataclass, the helper, and every test. `player_id`/`team_id` `Int64`, `is_goalkeeper` bool throughout.
