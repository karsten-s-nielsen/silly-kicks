# SkillCorner Corpus + Visibility Implementation Plan (PR-A)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the 98 owner-tier SkillCorner matches reachable and *safely* classified, surface the `is_detected` flag the pipeline has been discarding, fix the coordinate defects that block routing SkillCorner onto the native builder, and land the registered-protocol machinery (nested-HPO three-arm paired test, detected-keeper ghost-GK admission) — **code and tests only, no weights.**

**Architecture:** Bottom-up in dependency order: coordinate primitives (`spadl/skillcorner.py`) → the native tracking builder (`tracking/skillcorner.py`) → the loader that feeds it → the trainers that consume it. Every registered rule from the spec becomes a **pure function with a table test**; every gate must name the mutation that kills it. The owner runs follow this PR; PR-B bundles weights.

**Tech Stack:** Python 3.10 (`.venv`), pandas/numpy, pytest, ruff 0.15.7, pyright 1.1.409. No new runtime dependencies (`pyarrow>=14` is already declared).

**Spec:** `docs/superpowers/specs/2026-07-14-skillcorner-corpus-and-visibility-design.md` (**rev 5** — the plan review found the interpolator-tell refusal was dead code, which changed a registered rule).

**Plan status:** rev 4 — executable. C1 (the nested-HPO `_fit_score` self-contradiction) rewritten as a real module-level extraction; 3 minors cleared.

---

## Context an implementer must have before touching anything

Read these four facts. Each was the subject of a review-round blocker; getting one wrong reintroduces a defect that was caught the hard way.

1. **`spadl/skillcorner.py::_transform_coords` CLAMPS.** It scales, then `.clip(0,105)/.clip(0,68)`. Harmless for events (an action is on-pitch by construction), **destructive for tracking**: measured on one real match, it snaps **11.31% of ball rows** and 0.71% of player rows, up to **9.00 m**, and it turns *a ball nine metres behind the goal* into *a ball on the goal line*. Tracking must call the **affine part only**.
2. **Null `visibility` is ambiguous.** It means "this provider observes everyone" (Gradient Sports, IDSSE) **or** "the kloppy gateway threw the flag away" (SkillCorner-via-kloppy). Never read it as a boolean. Use the provider allowlist (Task 6) and **raise** on the ambiguous case.
3. **Public-vs-owner must never be keyed on provider name.** `_PUBLIC_PROVIDERS = {"skillcorner","idsse"}` has **six sites**, and two of them (`train_xshot_occurrence.py:313`, `train_xcross_attempt.py:398`) set the *shipped artifact's label*. The 98 restricted matches carry provider `skillcorner`.
4. **The S1 within-pitch gate cannot see a pitch-dimension error.** Measured: a 4 m pitch-length error moves the statistic from 0.00047 to 0.00095, *inside* the clean public range (worst 0.00086). It catches catastrophic sign/origin breaks (0.34139). Do not let Task 5 claim otherwise — the limitation is pinned by a test **on purpose**.

---

## Guard discipline — MANDATORY, every task (rev 2)

Four review rounds have each found the same defect class: **a guard that cannot fail.** The
licensing control, the ghost-GK admission gate, the S1 exclusion, the interpolator-tell refusal —
each looked like protection and was decoration. The habit that prevents it is cheap and is now a
required step in every task:

> **For every guard you write, name the single line of PRODUCTION code whose deletion makes the
> test fail. Then delete that line, run the test, and watch it fail. Restore it.**
> A guard whose kill-line you cannot name is not a guard. Do not proceed past it.

Record the result in the task's checklist. If the test still passes with the line deleted, **stop
and report** — the guard is vacuous and the task is not done, however green the suite is.

### The companion rule: quote the definition (rev 2)

A guard can be perfectly designed and still name a symbol that does not exist. Round 6 of review
found four such: `all_feats` (the real name is `all_features`), `parts_m` (never initialised), a
`home_team_id` argument omitted from a required keyword, and — worst — a `match_id` **parsed out of
a temp filename**, which silently collapsed all ten public matches into one CV group.

> **For every symbol this plan names — function, variable, keyword, dict key — the source line that
> defines it is quoted beside it, with its file and line number.** If you cannot quote it, open the
> file. If it is not there, the plan is wrong, not the code.

Symbols verified against source for this revision:

| Symbol | Definition | Verified |
|---|---|---|
| `_transform_coords(x, y, pitch_length, pitch_width)` | `spadl/skillcorner.py:35` | ✓ clamps at `:60-61` |
| `convert_to_frames(bronze, *, home_team_id, ...)` | `tracking/skillcorner.py:91-93` | ✓ `home_team_id` is **required kw-only** |
| `_TOL_BALL = 30.0` | `tracking/skillcorner.py:70` | ✓ "provisional — re-calibrate" |
| `_resolve_token(token)` / `_base_url()` | `_loader_pining.py:66` / `:62` | ✓ |
| `_build_skillcorner(paths, match_id, tracking_limit)` | `_loader_pining.py:~284` | ✓ **already receives `match_id`** |
| `all_features`, `all_labels`, `all_game_ids`, `all_providers` | `train_ghost_gk.py:224-227` | ✓ **not** `all_feats` |
| `parts_x, parts_y, parts_g, parts_p` | `train_xshot_occurrence.py:52` | ✓ **four**, no `parts_m` |
| `_fit_score(Xtr, ytr, te_idx)` | `train_xshot_occurrence.py:185` (closure) | ✓ already drops NaN folds |
| `_pinned_params(overrides)` | `tracking/_xshot_occurrence.py:328` | ✓ in the **model** module |
| `metrics["shipped_variant"]` | `train_xshot_occurrence.py:333` | ✓ |
| `prepare_ghost_gk_training_data(...) -> (features, labels)` | `tracking/_ghost_gk.py:820` | ✓ no keeper id, no visibility |

| Guard | Kill-line (delete this; the test MUST fail) | Task |
|---|---|---|
| Off-pitch positions survive | `players["x"], players["y"] = _scale_to_spadl(...)` → restore the `+ 52.5` offset | 3 |
| Pitch scaling | the `105.0 / L` factor inside `_scale_to_spadl` | 2, 3 |
| Missing dims raise | the `raise ValueError` in the dims resolution | 3 |
| S1 rate-gate excludes | the `if report.excluded: continue` in the loader | 4 |
| `_TOL_BALL` can fire | `_TOL_BALL = 15.0` → restore `30.0` | 4 |
| Licensing / label | `shipped = artifact_label(...)` → restore `provset <= _PUBLIC_PROVIDERS` | 9 |
| Detection fail-closed | the `raise ValueError` in `keeper_detection_mask`'s all-null branch | 6 |
| Stale cache is a miss | the `cache_is_valid(...)` call → restore `features.parquet.exists()` | 11 |
| Detected-only targets | the `keep = keeper_detection_mask(...)` filter in the ghost extractor | 12 |
| Keeper-domain exclusion | the `domain = ...` exclusion of expansion keepers | 12 |

**Commands** (from repo root, `D:\Development\karstenskyt__silly-kicks_part-deux`):

```bash
.venv/Scripts/python.exe -m pytest tests/ -m "not e2e" -q --benchmark-skip
.venv/Scripts/python.exe -m ruff check silly_kicks tests scripts
.venv/Scripts/python.exe -m ruff format --check silly_kicks tests scripts
.venv/Scripts/python.exe -m pyright
```

A Bash hook blocks foreground commands over 30 s — use `run_in_background` and poll for anything longer.

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `silly_kicks/spadl/skillcorner.py` | **Modify.** Split the affine scale (`_scale_to_spadl`) out of the clamping `_transform_coords`. | 2 |
| `silly_kicks/tracking/skillcorner.py` | **Modify.** Take pitch dims; call `_scale_to_spadl` (never the clamp); recalibrate `_TOL_BALL`; implement the deferred rate-gate. | 3, 4 |
| `silly_kicks/tracking/_ghost_gk.py` | **Modify.** Surface keeper identity + detection in the extractor's `meta`; add `keeper_detection_mask`. | 6 |
| `scripts/_loader_pining.py` | **Modify.** Role-aware artifact keys; extension-preserving temp names; parquet dispatch; SkillCorner bronze shaping → native builder; `match_visibility()`; `--match-ids-json` threading (already written, uncommitted). | 5, 7, 8 |
| `scripts/_corpus.py` | **Create.** The registered corpus taxonomy: `PUBLIC_CORPUS`, `is_public_row`, artifact-label derivation. One home, two trainers import it. | 9 |
| `scripts/_paired.py` | **Create.** The registered selection rules as pure functions: `clears_rule`, `fixed_sequence_ship`, `ghost_admission`. Table-tested; no I/O. | 10 |
| `scripts/train_xshot_occurrence.py` | **Modify.** Visibility-keyed arms; three candidates; nested HPO; cache schema. | 9, 10, 11 |
| `scripts/train_xcross_attempt.py` | **Modify.** Same, byte-mirrored. | 9, 10, 11 |
| `scripts/train_ghost_gk.py` | **Modify.** Detected-only targets; keeper-grouped CV; paired admission; size-gate fix. | 12, 13 |
| `scripts/_cache.py` | **Create.** Feature-cache validity: schema version + corpus fingerprint. A stale cache must be a MISS. | 11 |
| `tests/scripts/conftest.py` | **Create.** Puts `scripts/` on `sys.path` for this directory. **Required** — `pyproject.toml` sets `pythonpath = [".", "tests"]`, so `scripts` is NOT importable, and every new test below imports a script module at module level. | 1 |
| `tests/spadl/test_skillcorner_coords.py` | **Create.** Clamp split + mutation. | 2 |
| `tests/scripts/test_loader_artifacts.py` | **Create.** Role/suffix resolution, extension preservation, `match_visibility`, cross-provider regression. | 5, 8 |
| `tests/scripts/test_loader_skillcorner_native.py` | **Create.** Bronze shaping; the action↔frame co-location e2e. | 7 |
| `tests/scripts/test_cache_schema.py` | **Create.** Stale-cache miss. | 11 |
| `tests/tracking/test_skillcorner_pitch_dims.py` | **Create.** Scaling, off-pitch survival, byte-identity on 105×68. | 3 |
| `tests/tracking/test_skillcorner_s1_gate.py` | **Create.** Rate-gate power **and** its pinned limitation. | 4 |
| `tests/tracking/test_ghost_gk_keeper_meta.py` | **Create.** Keeper identity/detection surfacing; allowlist raises. | 6 |
| `tests/scripts/test_corpus_taxonomy.py` | **Create.** Red-first compliance gate. | 9 |
| `tests/scripts/test_paired_rules.py` | **Create.** Table tests for every registered rule. | 10 |

---

## Task 1: Branch and baseline

**Files:** none (verification only)

- [ ] **Step 1: Confirm the working tree carries only the corpus-pin patch**

```bash
git status --short
```

Expected: exactly three modified files — `scripts/_loader_pining_to_cache.py`, `scripts/train_xcross_attempt.py`, `scripts/train_xshot_occurrence.py` (the `--match-ids-json` work written during the TF-19 owner runs), plus the untracked spec/plan/review docs. **Do not revert them** — they are folded in at Task 14.

- [ ] **Step 2: Branch**

```bash
git switch -c pr-s115-skillcorner-corpus-visibility
```

- [ ] **Step 3: Green baseline**

Run (background; ~11 min):

```bash
.venv/Scripts/python.exe -m pytest tests/ -m "not e2e" -q --benchmark-skip
```

Expected: `4690 passed` (the count from the 4.47.0 release gate). **One known pre-existing failure is owner-gated and excluded by `-m "not e2e"`** (`test_xshot_gradientsports_e2e`, TODO.md). If anything else fails, stop and report — do not build on a red baseline.

- [ ] **Step 4: Make `scripts/` importable from the new test directory**

`pyproject.toml` sets `pythonpath = [".", "tests"]` — **`scripts` is not on it.** Existing tests that need a script module call `sys.path.insert(0, "scripts")` *inside the test function*. The new tests in this plan import at module level, so they need this or they fail at collection.

Create `tests/scripts/conftest.py`:

```python
"""Put `scripts/` on sys.path for this directory.

pyproject sets pythonpath = [".", "tests"]; the script modules (_corpus, _paired, _cache,
_loader_pining) are not importable without this. Scoped to tests/scripts/ so the global config
is untouched.
"""

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
```

Verify it works before writing any test that depends on it:

```bash
.venv/Scripts/python.exe -m pytest tests/scripts/ -q --collect-only
```

Expected: `no tests ran` (the directory is empty) — **not** an ImportError.

```bash
git add tests/scripts/conftest.py
```

---

## Task 2: Split the affine scale from the clamp

**The most consequential change in this PR.** Rev 2 of the spec would have single-sourced the clamping function into tracking; a review round caught it.

**Files:**
- Modify: `silly_kicks/spadl/skillcorner.py:35-62`
- Test: `tests/spadl/test_skillcorner_coords.py` (create)

- [ ] **Step 1: Write the failing test**

```python
"""The events transform CLAMPS; tracking must never inherit that (spec 3.4).

Measured on real data: routing tracking through the clamping transform snaps 11.31% of ball
rows and 0.71% of player rows, by up to 9.00 m -- and turns a ball nine metres behind the goal
into a ball on the goal line, erasing goal-vs-save.
"""

import numpy as np
import pandas as pd

from silly_kicks.spadl.skillcorner import _scale_to_spadl, _transform_coords


def test_scale_is_affine_and_never_clamps():
    # raw centre-origin metres on a 104 x 68 pitch, including legitimately OFF-PITCH points:
    # a ball 9 m behind the goal line, a keeper behind his line, a ball past the touchline.
    x = pd.Series([-52.0, 0.0, 52.0, 61.0, -55.0])
    y = pd.Series([-34.0, 0.0, 34.0, 10.0, 40.0])
    sx, sy = _scale_to_spadl(x, y, 104.0, 68.0)

    assert sx.iloc[0] == 0.0        # goal line
    assert sx.iloc[1] == 52.5       # centre spot
    assert sx.iloc[2] == 105.0      # far goal line
    assert sx.iloc[3] > 105.0       # 9 m BEYOND the goal -- must survive
    assert sx.iloc[4] < 0.0         # behind the other goal -- must survive
    assert sy.iloc[4] > 68.0        # past the touchline -- must survive


def test_transform_coords_still_clamps_for_events():
    """The events converter's behaviour is UNCHANGED -- an action is on-pitch by construction."""
    x = pd.Series([61.0, -55.0])
    y = pd.Series([10.0, 40.0])
    cx, cy = _transform_coords(x, y, 104.0, 68.0)
    assert cx.iloc[0] == 105.0      # clamped
    assert cx.iloc[1] == 0.0        # clamped
    assert cy.iloc[1] == 68.0       # clamped


def test_transform_coords_equals_scale_then_clamp():
    """_transform_coords must be exactly _scale_to_spadl + clamp -- one truth, not two."""
    rng = np.random.default_rng(0)
    x = pd.Series(rng.uniform(-60, 60, 500))
    y = pd.Series(rng.uniform(-40, 40, 500))
    sx, sy = _scale_to_spadl(x, y, 103.0, 67.0)
    cx, cy = _transform_coords(x, y, 103.0, 67.0)
    pd.testing.assert_series_equal(cx, sx.clip(0.0, 105.0))
    pd.testing.assert_series_equal(cy, sy.clip(0.0, 68.0))
```

- [ ] **Step 2: Run it and watch it fail**

```bash
.venv/Scripts/python.exe -m pytest tests/spadl/test_skillcorner_coords.py -q
```

Expected: `ImportError: cannot import name '_scale_to_spadl'`.

- [ ] **Step 3: Implement the split**

In `silly_kicks/spadl/skillcorner.py`, replace the body of `_transform_coords` (currently lines 35-62) with two functions. Keep `_transform_coords`'s signature and docstring intent; the clamp comment stays with the clamp.

```python
def _scale_to_spadl(
    x: pd.Series,
    y: pd.Series,
    pitch_length: int | float,
    pitch_width: int | float,
) -> tuple[pd.Series, pd.Series]:
    """Affine map from centred metres to the SPADL 105x68 frame. NO clamping.

    This is the single-sourced coordinate truth for SkillCorner: the EVENTS converter calls it
    via ``_transform_coords`` (which clamps afterwards, safe because an action's location is
    on-pitch by construction); the TRACKING builder (``tracking/skillcorner.py``) calls it
    DIRECTLY, because tracking is full of legitimately off-pitch positions -- an out-of-play
    ball, a keeper behind his line, and decisively a ball that has crossed the goal line, which
    is what a goal IS. Clamping tracking would erase goal-vs-save. See spec 3.4.
    """
    half_length = pitch_length / 2
    half_width = pitch_width / 2
    return (x / half_length) * 52.5 + 52.5, (y / half_width) * 34.0 + 34.0


def _transform_coords(
    x: pd.Series,
    y: pd.Series,
    pitch_length: int | float,
    pitch_width: int | float,
) -> tuple[pd.Series, pd.Series]:
    """Rescale centred metres to the SPADL frame, clamped to the pitch (EVENTS only).

    Parameters
    ----------
    x, y : pd.Series
        Coordinates in centred metres (origin at the centre spot).
    pitch_length, pitch_width : int or float
        Actual pitch dimensions from ``match_metadata``.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        ``(x_spadl, y_spadl)`` in the SPADL [0, 105] x [0, 68] frame.
    """
    x_out, y_out = _scale_to_spadl(x, y, pitch_length, pitch_width)
    # Clamp to SPADL pitch boundaries (raw data can slightly exceed pitch dims).
    # NEVER reuse this for tracking -- see _scale_to_spadl's docstring.
    return x_out.clip(lower=0.0, upper=105.0), y_out.clip(lower=0.0, upper=68.0)
```

- [ ] **Step 4: Run the tests**

```bash
.venv/Scripts/python.exe -m pytest tests/spadl/test_skillcorner_coords.py tests/spadl/ -q -m "not e2e"
```

Expected: the three new tests pass, and **every existing spadl/skillcorner test still passes** — the events path is byte-unchanged.

- [ ] **Step 5: Lint, type, stage**

```bash
.venv/Scripts/python.exe -m ruff check silly_kicks/spadl/skillcorner.py tests/spadl/test_skillcorner_coords.py
.venv/Scripts/python.exe -m ruff format silly_kicks/spadl/skillcorner.py tests/spadl/test_skillcorner_coords.py
.venv/Scripts/python.exe -m pyright silly_kicks/spadl/skillcorner.py
git add silly_kicks/spadl/skillcorner.py tests/spadl/test_skillcorner_coords.py
```

---

## Task 3: Pitch dimensions in the native tracking builder

**Files:**
- Modify: `silly_kicks/tracking/skillcorner.py` (`EXPECTED_INPUT_COLUMNS` ~:40-55; the transform at ~:146-147 and ~:170-171)
- Test: `tests/tracking/test_skillcorner_pitch_dims.py` (create)

Today the builder does `x + 52.5`, `y + 34.0` with **no pitch input** — so on a 101 m pitch the goal line lands 2 m from where it belongs. 4 of the 10 public matches are 104/106 m; the lakehouse consumes this.

- [ ] **Step 1: Write the failing test**

```python
"""Pitch-dimension normalisation (spec 3.4). The builder must scale, not offset -- and must
NOT clamp (that is the events transform's job; see tests/spadl/test_skillcorner_coords.py)."""

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking.skillcorner import convert_to_frames


def _bronze(pitch_length: float, pitch_width: float, *, x: float, y: float) -> pd.DataFrame:
    """One frame, one player, one ball, on a pitch of the given dimensions."""
    return pd.DataFrame(
        [
            {
                "match_id": "m1", "period": 1, "frame": 1, "timestamp": 0.0,
                "player_id": "p1", "team_id": "A", "is_goalkeeper": False,
                "x": x, "y": y,
                "ball_x": 0.0, "ball_y": 0.0, "ball_z": 0.0,
                "is_visible": True, "frame_rate": 10.0,
                "pitch_length": pitch_length, "pitch_width": pitch_width,
            }
        ]
    )


def test_goal_line_lands_on_the_goal_line_for_a_short_pitch():
    # On a 101 m pitch the goal line is at raw x = 50.5. It must map to SPADL x = 105.
    frames, _ = convert_to_frames(
        _bronze(101.0, 67.0, x=50.5, y=0.0), home_team_id="A", output_convention="absolute_frame"
    )
    p = frames[~frames["is_ball"].astype(bool)].iloc[0]
    assert p["x"] == pytest.approx(105.0, abs=1e-9)
    assert p["y"] == pytest.approx(34.0, abs=1e-9)


def test_standard_pitch_is_unchanged():
    # 105 x 68: the new scale must be a NO-OP versus the old +52.5/+34 offset.
    frames, _ = convert_to_frames(
        _bronze(105.0, 68.0, x=10.0, y=-5.0), home_team_id="A", output_convention="absolute_frame"
    )
    p = frames[~frames["is_ball"].astype(bool)].iloc[0]
    assert p["x"] == pytest.approx(62.5, abs=1e-9)
    assert p["y"] == pytest.approx(29.0, abs=1e-9)


def test_off_pitch_positions_survive():
    """The clamp regression. A ball beyond the goal line keeps x > 105 -- goal vs save.

    KILL-LINE: restore the `+ 52.5` offset (or route through the clamping _transform_coords)
    and this test MUST fail. Verify that before moving on.
    """
    b = _bronze(105.0, 68.0, x=0.0, y=0.0)
    b.loc[0, "ball_x"] = 57.0   # 4.5 m beyond the goal line
    b.loc[0, "ball_y"] = 40.0   # 6 m past the touchline
    frames, _ = convert_to_frames(b, home_team_id="A", output_convention="absolute_frame")
    ball = frames[frames["is_ball"].astype(bool)].iloc[0]
    assert ball["x"] > 105.0
    assert ball["y"] > 68.0


def test_missing_pitch_dims_raise():
    """Fail-CLOSED (spec 3.4 / reviewer m1): a silent 105x68 default would reproduce the very
    defect being fixed, and a warning is invisible in a DGX batch log."""
    b = _bronze(105.0, 68.0, x=0.0, y=0.0).drop(columns=["pitch_length", "pitch_width"])
    with pytest.raises(ValueError, match="pitch_length"):
        convert_to_frames(b, home_team_id="A", output_convention="absolute_frame")


def test_assume_standard_pitch_is_the_explicit_opt_in():
    b = _bronze(105.0, 68.0, x=10.0, y=0.0).drop(columns=["pitch_length", "pitch_width"])
    frames, _ = convert_to_frames(
        b, home_team_id="A", output_convention="absolute_frame", assume_standard_pitch=True
    )
    p = frames[~frames["is_ball"].astype(bool)].iloc[0]
    assert p["x"] == pytest.approx(62.5, abs=1e-9)
```

- [ ] **Step 2: Run it and watch it fail**

```bash
.venv/Scripts/python.exe -m pytest tests/tracking/test_skillcorner_pitch_dims.py -q
```

Expected: `test_goal_line_lands_on_the_goal_line_for_a_short_pitch` FAILS (`105.0 != 103.0` — the fixed offset), and `test_missing_pitch_dims_raise` FAILS (no raise today).

- [ ] **Step 3: Implement**

In `silly_kicks/tracking/skillcorner.py`:

(a) Add the two columns to `EXPECTED_INPUT_COLUMNS` (after `"frame_rate"`):

```python
    "pitch_length",
    "pitch_width",
```

(b) Import the single-sourced affine map alongside the existing clock constant:

```python
from silly_kicks.spadl.skillcorner import _PERIOD_START_SECONDS, _scale_to_spadl
```

(c) Add the `assume_standard_pitch` keyword to `convert_to_frames`'s signature (default `False`), and resolve the dimensions immediately after the existing `missing = [...]` check:

```python
    if assume_standard_pitch:
        pitch_length, pitch_width = 105.0, 68.0
    else:
        missing_dims = [c for c in ("pitch_length", "pitch_width") if c not in bronze.columns]
        if missing_dims:
            raise ValueError(
                f"skillcorner.convert_to_frames: bronze missing {missing_dims}. Pitch dimensions are "
                "REQUIRED -- defaulting to 105x68 silently reproduces the goal-line defect this "
                "fixes (spec 3.4). Pass assume_standard_pitch=True only if you know the pitch is "
                "105x68."
            )
        pitch_length = float(bronze["pitch_length"].iloc[0])
        pitch_width = float(bronze["pitch_width"].iloc[0])
```

Amend the `missing` check that precedes it so `pitch_length`/`pitch_width` are not double-reported when `assume_standard_pitch=True`:

```python
    required = [c for c in EXPECTED_INPUT_COLUMNS if not (assume_standard_pitch and c.startswith("pitch_"))]
    missing = [c for c in required if c not in bronze.columns]
    if missing:
        raise ValueError(f"skillcorner.convert_to_frames: bronze missing column(s): {missing}")
```

(d) Replace the two offset blocks. **Players** (was `players["x"] = players["x"] + 52.5` / `players["y"] = players["y"] + 34.0`):

```python
    players["x"], players["y"] = _scale_to_spadl(players["x"], players["y"], pitch_length, pitch_width)
```

**Ball** (was `ball["x"] = ball["x"] + 52.5` / `ball["y"] = ball["y"] + 34.0`):

```python
    ball["x"], ball["y"] = _scale_to_spadl(ball["x"], ball["y"], pitch_length, pitch_width)
```

Note `_scale_to_spadl` — **never** `_transform_coords`. The clamp would snap 11.31% of ball rows.

- [ ] **Step 4: Run**

```bash
.venv/Scripts/python.exe -m pytest tests/tracking/test_skillcorner_pitch_dims.py tests/tracking/test_skillcorner_builder.py tests/tracking/test_skillcorner_within_pitch_invariant.py tests/tracking/test_skillcorner_gk_roster_trust.py -q -m "not e2e"
```

(There is **no** `tests/tracking/test_skillcorner.py` — the real files are the four above, plus `test_gk_skillcorner_roster.py`.)

Expected: the new tests pass. Existing fixtures that build bronze without pitch columns **will now fail** — update them to include `pitch_length=105.0, pitch_width=68.0`. That is the intended blast radius; **do not** add a silent default to avoid it, or you have reintroduced the defect.

- [ ] **Step 5: Lint, type, stage**

```bash
.venv/Scripts/python.exe -m ruff check silly_kicks/tracking/skillcorner.py tests/tracking/test_skillcorner_pitch_dims.py
.venv/Scripts/python.exe -m ruff format silly_kicks/tracking/skillcorner.py tests/tracking/test_skillcorner_pitch_dims.py
.venv/Scripts/python.exe -m pyright silly_kicks/tracking/skillcorner.py
git add silly_kicks/tracking/skillcorner.py tests/tracking/ 
```

---

## Task 4: Recalibrate `_TOL_BALL` and implement the deferred rate-gate

The in-code comment asks for exactly this: *"TOL_BALL provisional (re-calibrate from the measured bronze on the pining corpus)"* and *"The deferred CI rate-gate is the SYSTEMATIC backstop"* — deferred, i.e. never implemented. Today `_TOL_BALL = 30.0 m` against a **measured maximum ball excursion of 9.00 m**: it cannot fire.

**Calibration (measured on the known-good public 10, 10.0 M rows; calibrating on the 98 would be circular):**

| statistic | public-10 worst | 4 m pitch-dim error | catastrophic break |
|---|---|---|---|
| player rows > 3 m off | 0.00086 | 0.00095 | **0.34139** |
| ball rows > 10 m off | 0.00000 | 0.00000 | — |
| max excursion | player 11.01 m / ball **9.00 m** | — | player 63.34 m |

**Files:**
- Modify: `silly_kicks/tracking/skillcorner.py:70` (`_TOL_BALL`), plus a new pure gate function and its call in `convert_to_frames`
- Test: `tests/tracking/test_skillcorner_s1_gate.py` (create)

- [ ] **Step 1: Write the failing test**

```python
"""The S1 geometry gate (spec 4.4). Registered thresholds are calibrated on the public 10.

CRITICAL: the last test pins a LIMITATION on purpose. This gate catches catastrophic sign/origin
breaks; it CANNOT see a pitch-dimension error (measured: 0.00095 vs a clean 0.00086). Nor can
action-frame co-location, since events and tracking read the same metadata and move together.
If someone later 'fixes' that test, they have misunderstood the gate.
"""

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking.skillcorner import (
    _PLAYER_OFF_PITCH_RATE_MAX,
    _BALL_OFF_PITCH_RATE_MAX,
    _TOL_BALL,
    geometry_rate_gate,
)


def test_tol_ball_is_calibrated_below_the_measured_headroom():
    """Calibration-PROVENANCE check (not a firing check -- the gate's firing is covered by
    test_a_wild_ball_is_excluded). The largest real ball excursion measured on the public 10 is
    9.00 m; the tolerance must sit above that (so clean data never trips) yet far below 30.0 m
    (which could not trip on any real break). This pins the calibrated VALUE, not the behaviour."""
    assert 9.0 < _TOL_BALL < 30.0
    assert _TOL_BALL == 15.0   # the specific calibrated value (spec 4.4)


def _rows(player_exc: np.ndarray, ball_exc: np.ndarray) -> pd.DataFrame:
    """Frames whose SPADL x sits `exc` metres beyond the goal line (0 = on-pitch)."""
    p = pd.DataFrame({"x": 105.0 + player_exc, "y": 34.0, "is_ball": False})
    b = pd.DataFrame({"x": 105.0 + ball_exc, "y": 34.0, "is_ball": True})
    return pd.concat([p, b], ignore_index=True)


def test_clean_match_passes():
    # public-10 worst: 0.086% of players beyond 3 m, no ball beyond 10 m
    player = np.zeros(100_000)
    player[:86] = 5.0
    ball = np.zeros(5_000)
    report = geometry_rate_gate(_rows(player, ball))
    assert report.excluded is False


def test_catastrophic_break_is_excluded():
    # measured: an origin/sign break puts 34% of players beyond 3 m
    player = np.zeros(100_000)
    player[:34_000] = 20.0
    report = geometry_rate_gate(_rows(player, np.zeros(5_000)))
    assert report.excluded is True
    assert "player" in report.reason


def test_a_wild_ball_is_excluded():
    ball = np.zeros(5_000)
    ball[:50] = 25.0          # 1% of ball rows 25 m off-pitch; real worst is 9.00 m at 0.000%
    report = geometry_rate_gate(_rows(np.zeros(100_000), ball))
    assert report.excluded is True
    assert "ball" in report.reason


def test_a_pitch_dimension_error_is_INVISIBLE_to_this_gate():
    """PINNED LIMITATION -- do not 'fix' this test.

    A 4 m pitch-length error produces player_frac(>3m) = 0.00095 against a clean worst of
    0.00086. It does not, and cannot, trip. The only instruments for pitch dims are provenance
    (spec 1.6.2) and asking SkillCorner. A gate that appeared to cover this would be worse than
    no gate at all.
    """
    player = np.zeros(100_000)
    player[:95] = 3.5         # the measured 0.00095
    report = geometry_rate_gate(_rows(player, np.zeros(5_000)))
    assert report.excluded is False
```

- [ ] **Step 2: Run it and watch it fail**

```bash
.venv/Scripts/python.exe -m pytest tests/tracking/test_skillcorner_s1_gate.py -q
```

Expected: `ImportError: cannot import name 'geometry_rate_gate'`.

- [ ] **Step 3: Implement**

In `silly_kicks/tracking/skillcorner.py`, change the tolerance and add the gate. Replace line 70:

```python
# Calibrated 2026-07-14 on the 10 PUBLIC pining matches (10.0M rows, correct transform):
# the largest real ball excursion is 9.00 m. The previous 30.0 m sat above every real value,
# so the ball tolerance could never fire. 15.0 m keeps 67% headroom over the worst observed
# and zero public rows exceed it. (Calibrating on the 98 private matches would be circular --
# they are the data under validation.) See spec 4.4.
_TOL_BALL = 15.0
```

Add, next to `_count_gross_off_pitch`:

```python
# Per-match rate-gate thresholds (spec 4.4), calibrated on the public 10:
#   worst clean player_frac(>3 m) = 0.00086  ->  0.005 leaves a 5.8x margin
#   worst clean ball_frac(>10 m)  = 0.00000  ->  0.0005 is the noise floor
# A catastrophic sign/origin break measures 0.34139 -- it exceeds the player threshold by 68x.
# A 4 m PITCH-DIMENSION error measures 0.00095 and does NOT trip: this gate cannot see one, and
# neither can action-frame co-location (events and tracking read the same metadata and move
# together). That limitation is deliberate, documented, and pinned by a test.
_PLAYER_OFF_PITCH_RATE_MAX = 0.005
_BALL_OFF_PITCH_RATE_MAX = 0.0005
_PLAYER_RATE_TOL = 3.0   # metres beyond the pitch rectangle
_BALL_RATE_TOL = 10.0


@dataclass(frozen=True)
class GeometryGateReport:
    """Outcome of the per-match geometry admission gate (spec 4.4)."""

    excluded: bool
    reason: str
    player_off_pitch_rate: float
    ball_off_pitch_rate: float


def geometry_rate_gate(frames: pd.DataFrame) -> GeometryGateReport:
    """Per-match geometry admission (spec 4.4). Pure; no I/O, no mutation.

    EXCLUDES a match whose off-pitch RATE exceeds the public-10-calibrated thresholds. This is
    the systematic backstop the S1 comment called 'deferred' -- the per-row invariant only warns,
    which is invisible in a batch log.
    """
    x = frames["x"].to_numpy(float)
    y = frames["y"].to_numpy(float)
    is_ball = frames["is_ball"].to_numpy(bool)
    exc = np.maximum(
        np.maximum(np.maximum(-x, x - 105.0), 0.0),
        np.maximum(np.maximum(-y, y - 68.0), 0.0),
    )
    players, balls = exc[~is_ball], exc[is_ball]
    p_rate = float((players > _PLAYER_RATE_TOL).mean()) if len(players) else 0.0
    b_rate = float((balls > _BALL_RATE_TOL).mean()) if len(balls) else 0.0
    reasons = []
    if p_rate > _PLAYER_OFF_PITCH_RATE_MAX:
        reasons.append(f"player off-pitch rate {p_rate:.5f} > {_PLAYER_OFF_PITCH_RATE_MAX}")
    if b_rate > _BALL_OFF_PITCH_RATE_MAX:
        reasons.append(f"ball off-pitch rate {b_rate:.5f} > {_BALL_OFF_PITCH_RATE_MAX}")
    return GeometryGateReport(
        excluded=bool(reasons),
        reason="; ".join(reasons),
        player_off_pitch_rate=p_rate,
        ball_off_pitch_rate=b_rate,
    )
```

Add `from dataclasses import dataclass` to the imports if absent.

- [ ] **Step 4: WIRE IT — a pure function nobody calls excludes nothing**

Rev 1 of this plan defined `geometry_rate_gate` and never called it, so §4.4's exclusion mechanism
still did not exist. Two changes make it real.

(a) In `tracking/skillcorner.py`, run the gate at the end of `convert_to_frames` and surface it on
the report, next to the existing `n_gross_off_pitch` count:

```python
    gate = geometry_rate_gate(df)
    report.geometry_excluded = gate.excluded
    report.geometry_reason = gate.reason
    report.player_off_pitch_rate = gate.player_off_pitch_rate
    report.ball_off_pitch_rate = gate.ball_off_pitch_rate
```

Add those four fields to `TrackingConversionReport` (`silly_kicks/tracking/schema.py`) with defaults
`False` / `""` / `0.0` / `0.0`, so every other provider's report is unaffected.

(b) In `scripts/_loader_pining.py::load_matches`, **drop an excluded match** — this is the line
whose deletion must break the test:

```python
        if provider == "skillcorner" and getattr(report, "geometry_excluded", False):
            print(f"  EXCLUDED {provider}/{match_id}: {report.geometry_reason}", file=sys.stderr)
            n_excluded += 1
            continue          # <-- the kill-line for the S1 exclusion guard
```

`_build_skillcorner` must therefore return the report alongside its three values; thread it through
`_build_match` and count exclusions so the run prints `excluded N/M matches` at the end. **A silent
exclusion is as bad as no exclusion.**

- [ ] **Step 5: The gate must actually drop a match — behavioural test**

Append to `tests/tracking/test_skillcorner_s1_gate.py`:

```python
def test_convert_to_frames_reports_the_exclusion():
    """The gate must reach a CONSUMER. A pure function nobody calls excludes nothing."""
    import numpy as np
    import pandas as pd

    from silly_kicks.tracking.skillcorner import convert_to_frames

    # 34% of players 20 m off-pitch = the measured catastrophic-break signature
    n = 1000
    rows = []
    for i in range(n):
        off = 20.0 if i < 340 else 0.0
        rows.append(
            {
                "match_id": "m1", "period": 1, "frame": i, "timestamp": float(i),
                "player_id": f"p{i % 22}", "team_id": "A" if i % 2 else "B",
                "is_goalkeeper": False,
                "x": 52.0 + off, "y": 0.0,          # raw centred metres; +52 is the goal line
                "ball_x": 0.0, "ball_y": 0.0, "ball_z": 0.0,
                "is_visible": True, "frame_rate": 10.0,
                "pitch_length": 105.0, "pitch_width": 68.0,
            }
        )
    frames, report = convert_to_frames(
        pd.DataFrame(rows), home_team_id="A", output_convention="absolute_frame"
    )
    assert report.geometry_excluded is True
    assert "player" in report.geometry_reason
```

Also extend the **existing** `tests/tracking/test_skillcorner_within_pitch_invariant.py` (it already
pins the warn-and-count behaviour) with one line asserting the new report fields exist and are
`False`/`0.0` on its clean fixture — so the two guards cannot drift apart.

- [ ] **Step 6: Kill-line check (MANDATORY)**

Delete the `continue` in the loader's exclusion branch, run
`pytest tests/scripts/test_loader_artifacts.py -k exclusion -q`, and **watch it fail**. Restore it.
Then set `_TOL_BALL = 30.0`, run `test_tol_ball_can_actually_fire`, watch it fail, restore.

If either still passes, the guard is decoration — stop and report.

- [ ] **Step 7: Run**

```bash
.venv/Scripts/python.exe -m pytest tests/tracking/test_skillcorner_s1_gate.py tests/tracking/test_skillcorner_within_pitch_invariant.py tests/tracking/test_skillcorner_builder.py -q -m "not e2e"
```

Expected: all pass. (Note the real filenames — there is no `tests/tracking/test_skillcorner.py`.)

- [ ] **Step 8: Lint, type, stage**

```bash
.venv/Scripts/python.exe -m ruff check silly_kicks/tracking/ scripts/_loader_pining.py tests/tracking/test_skillcorner_s1_gate.py
.venv/Scripts/python.exe -m ruff format silly_kicks/tracking/ scripts/_loader_pining.py tests/tracking/test_skillcorner_s1_gate.py
.venv/Scripts/python.exe -m pyright silly_kicks/tracking/skillcorner.py silly_kicks/tracking/schema.py
git add silly_kicks/tracking/ scripts/_loader_pining.py tests/tracking/test_skillcorner_s1_gate.py tests/tracking/test_skillcorner_within_pitch_invariant.py
```

---

## Task 5: Loader — role-aware artifacts, extension preservation, parquet dispatch

The 98 expose role-keyed artifacts (`events`, `metadata`, `tracking`, …) instead of filename-suffixed ones. Three failure points, all reproduced live.

**Files:**
- Modify: `scripts/_loader_pining.py` — `_artifact_key` (:125-130), `_download_to_temp` (:98), `_download_artifacts` (:220-240), `_build_skillcorner` (:291)
- Test: `tests/scripts/test_loader_artifacts.py` (create)

- [ ] **Step 1: Write the failing test**

```python
"""Loader must resolve BOTH SkillCorner artifact schemas (spec 3.1)."""

import pytest

from _loader_pining import _artifact_key, _dest_name


CANONICAL = {
    "1886347_dynamic_events": "1886347_dynamic_events.csv",
    "1886347_match": "1886347_match.json",
    "1886347_tracking_extrapolated": "1886347_tracking_extrapolated.jsonl",
}
ROLE_KEYED = {
    "events": "events.parquet",
    "freeze_frames": "freeze_frames.parquet",
    "metadata": "metadata.json",
    "physical": "physical.parquet",
    "tracking": "tracking.json.gz",
}


def test_suffix_resolution_still_works():
    assert _artifact_key(CANONICAL, suffix="_match.json", role="metadata") == "1886347_match"


def test_role_fallback_resolves_the_new_schema():
    assert _artifact_key(ROLE_KEYED, suffix="_match.json", role="metadata") == "metadata"


def test_unknown_role_and_suffix_raises():
    with pytest.raises(KeyError):
        _artifact_key(ROLE_KEYED, suffix="_nope.json", role="nonexistent")


def test_dest_name_preserves_the_extension():
    """kloppy sniffs the first byte: a gzip magic 0x1f under an extensionless name raises
    DeserializationError. The manifest's FILENAME must reach the temp file."""
    assert _dest_name("skillcorner", "1021404", "tracking", "tracking.json.gz").endswith(".json.gz")
    assert _dest_name("skillcorner", "1886347", "1886347_match", "1886347_match.json").endswith(".json")
```

- [ ] **Step 2: Run it and watch it fail**

```bash
.venv/Scripts/python.exe -m pytest tests/scripts/test_loader_artifacts.py -q
```

Expected: `TypeError: _artifact_key() got an unexpected keyword argument 'role'` and `ImportError` for `_dest_name`.

Note: `tests/scripts/` may not exist — create it with an empty `__init__.py` if the repo's other script tests need one (check `tests/` layout; `pyproject.toml` sets `pythonpath = [".", "tests"]`, and script modules are imported via `sys.path.insert(0, "scripts")` in existing tests — follow `tests/tracking/test_xcross_attempt_integration.py`'s pattern).

- [ ] **Step 3: Implement**

In `scripts/_loader_pining.py`:

```python
def _artifact_key(artifacts: dict, *, suffix: str, role: str) -> str:
    """Resolve an artifact KEY by filename suffix (canonical schema) or by ROLE (2026-07 schema).

    The canonical SkillCorner open-data matches key artifacts by match-id-prefixed filename
    (``1886347_match.json``); the owner-tier matches added in 2026-07 key them by role
    (``metadata`` -> ``metadata.json``). Try the suffix first, then the role.
    """
    for key, filename in artifacts.items():
        if str(filename).endswith(suffix):
            return key
    if role in artifacts:
        return role
    raise KeyError(f"no artifact ending with {suffix!r} and no role {role!r} in {sorted(artifacts)}")


def _dest_name(provider: str, match_id: str, artifact_key: str, filename: str) -> str:
    """Temp-file name that PRESERVES the artifact's extension.

    kloppy's ``identify_data_version`` sniffs the first byte: a gzipped tracking file under an
    extensionless name is seen as binary garbage and raises DeserializationError. The manifest's
    filename carries the extension, so use it. (Safe for IDSSE/GS, which magic-sniff -- but it
    CHANGES cache keys, so a pre-existing artifact cache is re-downloaded once.)
    """
    ext = "".join(Path(str(filename)).suffixes)
    return f"{provider}_{match_id}_{artifact_key}{ext}"
```

Thread the filename into `_download_to_temp`: add a `filename: str | None = None` parameter and build `dest` with it:

```python
    dest = dest_dir / (
        _dest_name(provider, match_id, artifact_key, filename)
        if filename is not None
        else f"{provider}_{match_id}_{artifact_key}"
    )
```

In `_download_artifacts`, pass both the role and the filename:

```python
    elif provider == "skillcorner":
        roles = {
            "events": _artifact_key(artifacts, suffix="_dynamic_events.csv", role="events"),
            "metadata": _artifact_key(artifacts, suffix="_match.json", role="metadata"),
            "tracking": _artifact_key(artifacts, suffix="_tracking_extrapolated.jsonl", role="tracking"),
        }
    ...
    for role, key in roles.items():
        artifact_key = key if key in artifacts else role
        out[role] = _download_to_temp(
            provider, match_id, artifact_key, token, base_url, tmp_dir,
            use_cache=use_cache, filename=artifacts.get(artifact_key),
        )
```

In `_build_skillcorner`, dispatch the events reader on the extension:

```python
    ev_path = paths["events"]
    raw_events = (
        pd.read_parquet(ev_path)
        if str(ev_path).endswith(".parquet")
        else pd.read_csv(ev_path, low_memory=False)
    )
```

- [ ] **Step 4: Run**

```bash
.venv/Scripts/python.exe -m pytest tests/scripts/test_loader_artifacts.py -q
```

Expected: 4 passed.

- [ ] **Step 5: Cross-provider regression (reviewer m2 — the rename touches EVERY provider)**

Append to the same test file:

```python
def test_dest_name_is_stable_for_idsse_and_gs():
    """The rename must not break the providers that were already working."""
    assert _dest_name("idsse", "DFL-MAT-J03WMX", "tracking", "tracking.xml").endswith(".xml")
    assert _dest_name("gradientsports", "10502", "tracking", "tracking.jsonl.bz2").endswith(".jsonl.bz2")
```

Then run one real IDSSE and one real Gradient Sports match end-to-end (owner box, token in env; ~5 min, background):

```bash
.venv/Scripts/python.exe -c "
import sys; sys.path.insert(0, 'scripts')
from _loader_pining import load_matches
for prov, mid, actions, frames, home in load_matches(providers=['idsse'], max_per_provider=1):
    print('IDSSE OK', mid, len(actions), len(frames))
for prov, mid, actions, frames, home in load_matches(providers=['gradientsports'], max_per_provider=1):
    print('GS OK', mid, len(actions), len(frames))
"
```

Expected: both print `OK` with non-zero counts.

- [ ] **Step 6: Lint, stage**

```bash
.venv/Scripts/python.exe -m ruff check scripts/_loader_pining.py tests/scripts/test_loader_artifacts.py
.venv/Scripts/python.exe -m ruff format scripts/_loader_pining.py tests/scripts/test_loader_artifacts.py
git add scripts/_loader_pining.py tests/scripts/test_loader_artifacts.py
```

---

## Task 6: Ghost-GK — surface keeper identity and detection

**Blocked discovery:** `prepare_ghost_gk_training_data` returns `(features, labels)` and its internal `meta` carries only `game_id, period_id, frame_id, gk_team_id, gk_x_gr, gk_y_gr`. **Neither the keeper's `player_id` nor its `visibility` survives.** Both §4.3 registrations (detected-only targets, keeper-grouped CV) are impossible without this task.

**Files:**
- Modify: `silly_kicks/tracking/_ghost_gk.py` — `meta_rows.append` (~:794-803), the empty-frame column list (~:812), `prepare_ghost_gk_training_data` (~:820, ~:896-924)
- Test: `tests/tracking/test_ghost_gk_keeper_meta.py` (create)

- [ ] **Step 1: Write the failing test**

```python
"""Ghost-GK must expose WHICH keeper each label belongs to, and whether he was SEEN (spec 4.3).

Without this, keeper-grouped CV cannot be built and 'detected-keeper targets only' cannot be
enforced -- and ~80% of SkillCorner keeper positions are interpolator output.
"""

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._ghost_gk import keeper_detection_mask


def test_detection_aware_provider_with_null_visibility_RAISES():
    """FAIL-CLOSED (spec 4.3). A null here means the kloppy gateway discarded the flag -- NOT
    that the keeper was observed. Reading it as 'keep' is the licensing landmine's failure shape."""
    with pytest.raises(ValueError, match="skillcorner"):
        keeper_detection_mask(pd.Series([None, None]), provider="skillcorner")


def test_detection_aware_provider_keeps_only_detected_keepers():
    mask = keeper_detection_mask(pd.Series([True, False, True]), provider="skillcorner")
    assert list(mask) == [True, False, True]


def test_fully_observed_provider_keeps_everything():
    """GS/IDSSE are full-pitch products: every player is observed, and no flag exists."""
    mask = keeper_detection_mask(pd.Series([None, None]), provider="gradientsports")
    assert list(mask) == [True, True]


def test_unknown_provider_RAISES():
    """Unknown providers are not assumed observed."""
    with pytest.raises(ValueError, match="unknown"):
        keeper_detection_mask(pd.Series([None]), provider="mystery_vendor")
```

- [ ] **Step 2: Run it and watch it fail**

```bash
.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_gk_keeper_meta.py -q
```

Expected: `ImportError: cannot import name 'keeper_detection_mask'`.

- [ ] **Step 3: Implement the mask**

In `silly_kicks/tracking/_ghost_gk.py`, near the top-level constants:

```python
# Which providers' feeds carry a per-player detection flag (spec 4.3).
# A null `visibility` is AMBIGUOUS: for a fully-observed provider it means "no flag exists and
# none is needed"; for a detection-aware provider it means "the pipeline DISCARDED the flag"
# (the kloppy gateway hard-codes visibility=None). Reading the second as the first would train
# ghost-GK on interpolator output -- ~80% of SkillCorner keeper positions are extrapolated.
_DETECTION_AWARE_PROVIDERS = frozenset({"skillcorner"})
_FULLY_OBSERVED_PROVIDERS = frozenset({"gradientsports", "sportec", "idsse"})


def keeper_detection_mask(visibility: pd.Series, *, provider: str) -> np.ndarray:
    """Rows whose keeper was ACTUALLY DETECTED. Fail-closed on the ambiguous null (spec 4.3)."""
    if provider in _FULLY_OBSERVED_PROVIDERS:
        return np.ones(len(visibility), dtype=bool)
    if provider not in _DETECTION_AWARE_PROVIDERS:
        raise ValueError(
            f"keeper_detection_mask: unknown provider {provider!r}. Add it to "
            "_DETECTION_AWARE_PROVIDERS or _FULLY_OBSERVED_PROVIDERS -- an unknown provider is "
            "NOT assumed observed."
        )
    if visibility.isna().all():
        raise ValueError(
            f"keeper_detection_mask: provider {provider!r} carries a detection flag, but "
            "`visibility` is entirely null -- the pipeline discarded it (the kloppy gateway "
            "hard-codes visibility=None). Build these frames with tracking.skillcorner instead; "
            "training on undetected keepers means training on the interpolator (spec 4.3)."
        )
    return visibility.fillna(False).astype(bool).to_numpy()
```

- [ ] **Step 4: Surface the identity in `meta`**

Extend the `meta_rows.append({...})` dict (~:794) with two keys read off the same `gk_row` the label comes from:

```python
                        "gk_player_id": gk_row["player_id"],
                        "gk_visibility": gk_row.get("visibility"),
```

Extend the empty-frame column list (~:812) to match:

```python
            pd.DataFrame(
                columns=[
                    "game_id", "period_id", "frame_id", "gk_team_id",
                    "gk_x_gr", "gk_y_gr", "gk_player_id", "gk_visibility",
                ]
            ),
```

- [ ] **Step 5: Return `meta` on request (backcompat)**

`prepare_ghost_gk_training_data` currently returns `(features, labels)` and **four call sites depend on that**. Add an opt-in rather than changing the shape:

```python
def prepare_ghost_gk_training_data(
    frames: pd.DataFrame,
    *,
    home_team_id: str | int,
    actions: pd.DataFrame | None = None,
    subsample_fps: float | None = 1.0,
    carrier_params: dict | None = None,
    return_meta: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
```

Every filter already applied to `features`/`labels` (the NaN-label drop at ~:899 and the domain filter at ~:907) must be applied to `meta` **identically**, or the keeper identity will not line up with its row. At the existing `valid` mask:

```python
    features = features[valid.values].reset_index(drop=True)
    labels = labels[valid.values].reset_index(drop=True)
    meta = meta[valid.values].reset_index(drop=True)
```

and at the `in_domain` mask, the same. Then:

```python
    if return_meta:
        return features, labels, meta
    return features, labels
```

- [ ] **Step 6: Run**

```bash
.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_gk_keeper_meta.py tests/tracking/test_ghost_gk.py -q -m "not e2e"
```

Expected: the 4 new tests pass; **every existing ghost-GK test still passes** (the default `return_meta=False` keeps the 2-tuple).

- [ ] **Step 7: Lint, type, stage**

```bash
.venv/Scripts/python.exe -m ruff check silly_kicks/tracking/_ghost_gk.py tests/tracking/test_ghost_gk_keeper_meta.py
.venv/Scripts/python.exe -m ruff format silly_kicks/tracking/_ghost_gk.py tests/tracking/test_ghost_gk_keeper_meta.py
.venv/Scripts/python.exe -m pyright silly_kicks/tracking/_ghost_gk.py
git add silly_kicks/tracking/_ghost_gk.py tests/tracking/test_ghost_gk_keeper_meta.py
```

---

## Task 7: Route the pining SkillCorner path onto the native builder

This is what surfaces `visibility` (and recovers `ball_z`, and fixes the ~0.26 m action↔frame inconsistency). The loader currently builds SkillCorner frames through the **kloppy gateway**, which hard-codes `visibility: None` and discards `ball_z`.

**Files:**
- Modify: `scripts/_loader_pining.py` — `build_skillcorner_frames` (~:260-280)
- Test: `tests/scripts/test_loader_skillcorner_native.py` (create)

- [ ] **Step 1: Write the failing test**

```python
"""The pining SkillCorner path must produce frames with REAL visibility and ball_z (spec 3.3).

The kloppy gateway hard-codes visibility=None and drops ball_z; the native builder carries both.
"""

import json

import numpy as np
import pandas as pd

from _loader_pining import _skillcorner_bronze


def test_bronze_carries_detection_and_ball_z_and_pitch_dims():
    meta = {
        "pitch_length": 104.0,
        "pitch_width": 68.0,
        "home_team": {"id": 1},
        "players": [
            {"id": 10, "team_id": 1, "player_role": {"acronym": "GK"}},
            {"id": 11, "team_id": 2, "player_role": {"acronym": "CB"}},
        ],
    }
    raw = [
        {
            "period": 1, "frame": 1, "timestamp": 0.0,
            "player_data": [
                {"player_id": 10, "x": -50.0, "y": 0.0, "is_detected": True},
                {"player_id": 11, "x": 5.0, "y": 3.0, "is_detected": False},
            ],
            "ball_data": {"x": 0.0, "y": 0.0, "z": 1.5},
        }
    ]
    bronze = _skillcorner_bronze(raw, meta, match_id="m1")

    assert set(bronze.columns) >= {
        "match_id", "period", "frame", "timestamp", "player_id", "team_id",
        "is_goalkeeper", "x", "y", "ball_x", "ball_y", "ball_z",
        "is_visible", "frame_rate", "pitch_length", "pitch_width",
    }
    assert bronze["is_visible"].tolist() == [True, False]      # is_detected survives
    assert bronze["ball_z"].iloc[0] == 1.5                     # ball_z survives
    assert bronze["pitch_length"].iloc[0] == 104.0             # real dims reach the builder
    assert bronze.loc[bronze["player_id"] == 10, "is_goalkeeper"].iloc[0]  # roster GK
```

- [ ] **Step 2: Run it and watch it fail**

```bash
.venv/Scripts/python.exe -m pytest tests/scripts/test_loader_skillcorner_native.py -q
```

Expected: `ImportError: cannot import name '_skillcorner_bronze'`.

- [ ] **Step 3: Implement the bronze shaper**

In `scripts/_loader_pining.py` (near `build_skillcorner_frames`):

```python
def _skillcorner_bronze(raw_frames: list[dict], meta: dict, *, match_id: str) -> pd.DataFrame:
    """Shape SkillCorner V3 tracking into the native builder's EXPECTED_INPUT_COLUMNS bronze.

    Replaces the kloppy gateway on the pining path (spec 3.3): kloppy hard-codes
    visibility=None (so `is_detected` -- which exists in the feed -- is lost), discards
    `ball_z`, and scales x on a pitch length that disagrees with our own events converter.
    """
    roster = {
        int(p["id"]): (
            str(p.get("team_id")),
            str((p.get("player_role") or {}).get("acronym", "")).upper() == "GK",
        )
        for p in meta.get("players", [])
    }
    rows = []
    for rec in raw_frames:
        bd = rec.get("ball_data") or {}
        for p in rec.get("player_data") or []:
            if p.get("x") is None:
                continue
            team_id, is_gk = roster.get(int(p["player_id"]), (None, False))
            rows.append(
                {
                    "match_id": match_id,
                    "period": rec["period"],
                    "frame": rec["frame"],
                    "timestamp": rec["timestamp"],
                    "player_id": str(p["player_id"]),
                    "team_id": team_id,
                    "is_goalkeeper": is_gk,
                    "x": float(p["x"]),
                    "y": float(p["y"]),
                    "ball_x": bd.get("x"),
                    "ball_y": bd.get("y"),
                    "ball_z": bd.get("z"),
                    "is_visible": p.get("is_detected"),
                    "frame_rate": 10.0,  # SkillCorner V3 is exactly 10.000 fps (measured)
                    "pitch_length": float(meta["pitch_length"]),
                    "pitch_width": float(meta["pitch_width"]),
                }
            )
    return pd.DataFrame(rows)
```

Rewrite `build_skillcorner_frames` to use it. **Two identity bugs must not be reintroduced here** —
they are the reason this step is written out in full:

1. **NEVER derive `match_id` from the temp filename.** Executed against both schemas:

   | corpus | temp filename | `split("_")[-2]` |
   |---|---|---|
   | canonical 10 | `skillcorner_1886347_1886347_tracking_extrapolated.jsonl` | **`"tracking"`** |
   | private 98 | `skillcorner_1021404_tracking.json.gz` | `"1021404"` |

   `match_id` becomes `game_id`, and `game_id` is the **`StratifiedGroupKFold` grouping key**. Deriving
   it from a path would collapse all ten public matches into one CV group called `"tracking"` —
   silently cutting the public arm from 17 groups to 8, in the arm that decides what ships.
   `_build_skillcorner(paths, match_id, tracking_limit)` (`_loader_pining.py:~284`) **already receives
   the real `match_id`**. Thread it.

2. **`convert_to_frames` requires `home_team_id`** — it is keyword-only with no default
   (`tracking/skillcorner.py:91-93`). Read it from `meta` *before* the call, not after.

```python
def build_skillcorner_frames(paths, match_id, tracking_limit):
    """Preprocessed silly-kicks frames from SkillCorner artifacts via the NATIVE builder.

    Was: the kloppy gateway (visibility=None, no ball_z, and a pitch scale that disagrees with our
    own events converter by ~0.26 m on a 104 m pitch). Now: the TF-23/ADR-034 native builder, which
    single-sources the coordinate transform with spadl.skillcorner (spec 3.3/3.4).

    `match_id` is PASSED IN, never parsed from a path -- it becomes game_id, the CV grouping key.
    """
    import gzip

    from silly_kicks.tracking import skillcorner as tracking_sk

    with open(paths["metadata"], encoding="utf-8") as fh:
        meta = json.load(fh)
    home_team_id = str(meta["home_team"]["id"])   # required kw-only arg below

    tpath = str(paths["tracking"])
    opener = gzip.open if tpath.endswith(".gz") else open
    with opener(tpath, "rt", encoding="utf-8") as fh:  # type: ignore[operator]
        first = fh.read(1)
        fh.seek(0)
        raw = json.load(fh) if first == "[" else [json.loads(line) for line in fh if line.strip()]

    if tracking_limit:
        raw = raw[:tracking_limit]

    bronze = _skillcorner_bronze(raw, meta, match_id=str(match_id))
    frames, report = tracking_sk.convert_to_frames(
        bronze, home_team_id=home_team_id, output_convention="absolute_frame"
    )
    return _preprocess(frames), report
```

Update `_build_skillcorner` to pass `match_id` and receive the report (Task 4's exclusion needs it):

```python
def _build_skillcorner(paths, match_id, tracking_limit):
    frames, report = build_skillcorner_frames(paths, match_id, tracking_limit)
    ...
    return actions, frames, home_team_id, report
```

`build_skillcorner_frames` has one other caller — the TF-27 GK-roster e2e (per its docstring). Update
that call site too; `grep -rn "build_skillcorner_frames" scripts/ tests/` finds them all.

- [ ] **Step 3b: The guard that catches the identity bug**

Append to `tests/scripts/test_loader_skillcorner_native.py`:

```python
def test_game_id_is_the_REAL_match_id_not_a_path_fragment():
    """The silent killer. game_id is the StratifiedGroupKFold grouping key: if it is derived from
    a temp filename, all ten public matches collapse into ONE group called "tracking" and the
    public arm -- the arm that decides what ships -- drops from 17 groups to 8.

    KILL-LINE: replace `match_id=str(match_id)` with `match_id=str(paths["tracking"]).split("_")[-2]`
    and this MUST fail.
    """
    meta = {
        "pitch_length": 105.0, "pitch_width": 68.0, "home_team": {"id": 1},
        "players": [{"id": 10, "team_id": 1, "player_role": {"acronym": "GK"}}],
    }
    raw = [
        {
            "period": 1, "frame": 1, "timestamp": 0.0,
            "player_data": [{"player_id": 10, "x": 0.0, "y": 0.0, "is_detected": True}],
            "ball_data": {"x": 0.0, "y": 0.0, "z": 0.0},
        }
    ]
    bronze = _skillcorner_bronze(raw, meta, match_id="1886347")
    assert bronze["match_id"].unique().tolist() == ["1886347"]
    assert "tracking" not in bronze["match_id"].tolist()
```

- [ ] **Step 4: The behavioural gate that actually matters — action↔frame co-location**

Append to `tests/scripts/test_loader_skillcorner_native.py`:

```python
import pytest


@pytest.mark.e2e
def test_action_frame_colocation_on_a_non_105_pitch():
    """On a 104 m match, a same-player event and its linked tracking frame must agree.

    This FAILS today (~0.26 m via kloppy, ~0.50 m via the old native offset) and passes only
    after the native route + the pitch fix. The gate and the fix are inseparable.

    Owner-gated: needs PINING_FOR_THE_DATA_TOKEN and network.
    """
    import sys

    sys.path.insert(0, "scripts")
    from _loader_pining import load_matches

    from silly_kicks.tracking import link_actions_to_frames

    prov, mid, actions, frames, home = next(
        iter(load_matches(providers=["skillcorner"], match_ids={"skillcorner": ["1886347"]}))
    )
    assert frames["visibility"].notna().any(), "the whole point: detection must survive"

    links, _report = link_actions_to_frames(actions, frames)
    merged = links.merge(actions[["action_id", "start_x", "start_y"]], on="action_id")
    ball = frames[frames["is_ball"].astype(bool)][["frame_id", "period_id", "x", "y"]]
    j = merged.merge(ball, on=["frame_id", "period_id"], how="inner")
    d = ((j["start_x"] - j["x"]) ** 2 + (j["start_y"] - j["y"]) ** 2) ** 0.5
    assert d.median() < 2.0, f"action-frame co-location median {d.median():.2f} m"
```

- [ ] **Step 5: Run**

```bash
.venv/Scripts/python.exe -m pytest tests/scripts/test_loader_skillcorner_native.py -q -m "not e2e"
```

Expected: the bronze test passes. Then run the e2e once with the token set (background, ~3 min):

```bash
.venv/Scripts/python.exe -m pytest tests/scripts/test_loader_skillcorner_native.py -q -m e2e
```

Expected: PASS. **If it fails, stop** — it means the native route moved geometry the wrong way, and nothing downstream is trustworthy.

- [ ] **Step 6: Lint, stage**

```bash
.venv/Scripts/python.exe -m ruff check scripts/_loader_pining.py tests/scripts/test_loader_skillcorner_native.py
.venv/Scripts/python.exe -m ruff format scripts/_loader_pining.py tests/scripts/test_loader_skillcorner_native.py
git add scripts/_loader_pining.py tests/scripts/test_loader_skillcorner_native.py
```

---

## Task 8: Loader — `match_visibility()` accessor

**Files:**
- Modify: `scripts/_loader_pining.py`
- Test: `tests/scripts/test_loader_artifacts.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
def test_match_visibility_reads_the_manifest_field(monkeypatch):
    """The manifest already carries visibility: public | private (spec 3.2)."""
    import _loader_pining as lp

    monkeypatch.setattr(
        lp,
        "_list_matches",
        lambda provider, token, base_url: [
            {"id": "1886347", "visibility": "public", "artifacts": {}},
            {"id": "1021404", "visibility": "private", "artifacts": {}},
            {"id": "9999999", "artifacts": {}},  # field ABSENT
        ],
    )
    vis = lp.match_visibility(["skillcorner"], token="t", base_url="b")
    assert vis[("skillcorner", "1886347")] == "public"
    assert vis[("skillcorner", "1021404")] == "private"
    assert vis[("skillcorner", "9999999")] == "private"  # FAIL-CLOSED on an absent field
```

- [ ] **Step 2: Run it and watch it fail** — `ImportError: cannot import name 'match_visibility'`.

- [ ] **Step 3: Implement**

```python
def match_visibility(
    providers: list[str], *, token: str | None = None, base_url: str | None = None
) -> dict[tuple[str, str], str]:
    """Map (provider, match_id) -> "public" | "private" from the pining manifest (spec 3.2).

    FAIL-CLOSED: a match whose manifest omits `visibility` is treated as **private**. A new match
    can never silently enter the public training arm.
    """
    tok = _resolve_token(token)           # real helper names, verified: _loader_pining.py:66 and :62
    base = base_url or _base_url()
    out: dict[tuple[str, str], str] = {}
    for provider in providers:
        for m in _list_matches(provider, tok, base):
            out[(provider, str(m["id"]))] = str(m.get("visibility", "private"))
    return out
```

(Reuse whatever the module already uses to resolve the token and base URL — grep for `PINING_FOR_THE_DATA_TOKEN` and `PINING_API_URL` and match the existing helpers' names.)

- [ ] **Step 4: Run, lint, stage**

```bash
.venv/Scripts/python.exe -m pytest tests/scripts/test_loader_artifacts.py -q
.venv/Scripts/python.exe -m ruff check scripts/_loader_pining.py
git add scripts/_loader_pining.py tests/scripts/test_loader_artifacts.py
```

---

## Task 9: The corpus taxonomy — delete `_PUBLIC_PROVIDERS`

**The compliance control.** `provset <= _PUBLIC_PROVIDERS` at `train_xshot_occurrence.py:313` / `train_xcross_attempt.py:398` sets the *shipped artifact's label*: a SkillCorner+IDSSE-only run containing 98 restricted matches ships labelled **`"public"`**. Verified in code.

**Files:**
- Create: `scripts/_corpus.py`
- Modify: `scripts/train_xshot_occurrence.py` (:30, :276, :313), `scripts/train_xcross_attempt.py` (:30, :344, :398)
- Test: `tests/scripts/test_corpus_taxonomy.py` (create)

- [ ] **Step 1: Write the failing test (red-first — today's code FAILS it)**

```python
"""The licensing control (spec 3.2). A model trained on restricted data must NEVER be labelled
`public`. Today's provider-name rule labels an sc_extended-shaped run "public" -- verified."""

import numpy as np
import pytest

from _corpus import PUBLIC_CORPUS, artifact_label, is_public_row


def test_absent_visibility_is_restricted():
    """FAIL-CLOSED: unknown provenance is never public."""
    vis = {("skillcorner", "1886347"): "public"}
    got = is_public_row(
        providers=np.array(["skillcorner", "skillcorner"]),
        match_ids=np.array(["1886347", "9999999"]),   # second is absent from the map
        visibility=vis,
    )
    assert list(got) == [True, False]


def test_a_restricted_skillcorner_match_is_not_public():
    vis = {("skillcorner", "1021404"): "private"}
    got = is_public_row(
        providers=np.array(["skillcorner"]), match_ids=np.array(["1021404"]), visibility=vis
    )
    assert list(got) == [False]


def test_label_is_never_public_when_the_ship_mask_contains_restricted_rows():
    """The bug that shipped: providers={skillcorner, idsse}, no GS -> old code said "public"."""
    assert artifact_label(providers={"skillcorner", "idsse"}, all_public=False) == "sc_extended"
    assert artifact_label(providers={"skillcorner", "idsse"}, all_public=True) == "public"
    assert artifact_label(providers={"skillcorner", "gradientsports"}, all_public=False) == "full"


def test_public_corpus_is_the_known_17():
    assert len(PUBLIC_CORPUS["skillcorner"]) == 10
    assert len(PUBLIC_CORPUS["idsse"]) == 7
```

- [ ] **Step 2: The RED-FIRST test must fail against TODAY's code, at the LABEL path**

`ModuleNotFoundError: No module named '_corpus'` is **not** red-first — that is a new file not
existing yet, and it pins nothing about the bug. The spec registered this gate as *"asserted against
the label path… driven red-first against today's code (which fails it)."* The actual defect is at
`train_xshot_occurrence.py:313` / `train_xcross_attempt.py:398`, and it must be caught **there**.

Add to `tests/scripts/test_corpus_taxonomy.py`:

```python
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.mark.slow
def test_a_restricted_corpus_NEVER_ships_a_public_label(tmp_path, monkeypatch):
    """RED-FIRST against today's code (which FAILS this).

    Today: providers = {skillcorner, idsse}, no Gradient Sports -> `two_candidate` is False ->
    the else branch runs -> `provset <= _PUBLIC_PROVIDERS` is True -> a model trained on 98
    RESTRICTED matches ships labelled "public". Verified at train_xshot_occurrence.py:313.

    KILL-LINE: restore `if provset <= _PUBLIC_PROVIDERS: shipped = "public"` and this MUST fail.
    """
    sys.path.insert(0, "scripts")
    import train_xshot_occurrence as tr

    # A synthetic corpus: one PUBLIC skillcorner match + one RESTRICTED skillcorner match.
    # No Gradient Sports -> the single-candidate `else` branch is the one under test.
    monkeypatch.setattr(
        tr, "_iter_matches_from_pining", lambda *a, **k: iter(_synthetic_two_match_corpus())
    )
    monkeypatch.setattr(
        "_loader_pining.match_visibility",
        lambda providers, **k: {
            ("skillcorner", "1886347"): "public",     # the real public one
            ("skillcorner", "1021404"): "private",    # a restricted one
        },
    )
    out = tmp_path / "run"
    tr.main(["--providers", "skillcorner", "--output-dir", str(out), "--n-trials", "1"])

    metrics = json.loads((out / "xshot_occurrence_v1" / "metrics.json").read_text())
    assert metrics["shipped_variant"] != "public", (
        "a model trained on restricted data was labelled public -- the licensing landmine"
    )
    assert metrics["shipped_variant"] == "sc_extended"
```

Write `_synthetic_two_match_corpus()` as a module-level helper in the test file, yielding two
`(provider, match_id, actions, frames, home)` tuples built from the shared synthetic fixtures the
existing `tests/tracking/test_xshot_occurrence_integration.py` already uses (import them; do not
re-derive). `main()` must accept an argv list — if it does not, add `def main(argv=None)` and pass
`argv` to `parse_args`; that is a one-line change and it makes the trainer testable at all.

**Run it now, before implementing `_corpus.py`. It MUST fail with `shipped_variant == "public"`.**
If it fails for any other reason (an import, a fixture), fix the test — a red-first test that goes
red for the wrong reason proves nothing.

- [ ] **Step 3: Implement `scripts/_corpus.py`**

```python
"""The registered corpus taxonomy (spec 3.2).

Public-vs-owner is keyed on the manifest's `visibility` field, NEVER on the provider name. The
98 owner-tier SkillCorner matches added in 2026-07 carry provider `skillcorner`; the old rule
(`_PUBLIC_PROVIDERS = {"skillcorner", "idsse"}`) would absorb them into the PUBLIC arm and ship
a model trained on non-redistributable data under a `public` label. That rule is deleted.
"""

from __future__ import annotations

import numpy as np

# The 17 matches we may redistribute. Drift here fails the run loudly (spec 3.2).
PUBLIC_CORPUS: dict[str, frozenset[str]] = {
    "skillcorner": frozenset(
        {"1886347", "1899585", "1925299", "1953632", "1996435",
         "2006229", "2011166", "2013725", "2015213", "2017461"}
    ),
    "idsse": frozenset(
        {"DFL-MAT-J03WMX", "DFL-MAT-J03WN1", "DFL-MAT-J03WOH", "DFL-MAT-J03WOY",
         "DFL-MAT-J03WPY", "DFL-MAT-J03WQQ", "DFL-MAT-J03WR9"}
    ),
}


def is_public_row(
    *, providers: np.ndarray, match_ids: np.ndarray, visibility: dict[tuple[str, str], str]
) -> np.ndarray:
    """Per-row public mask. FAIL-CLOSED: an absent (provider, match) is RESTRICTED."""
    return np.array(
        [visibility.get((str(p), str(m)), "private") == "public"
         for p, m in zip(providers, match_ids, strict=True)],
        dtype=bool,
    )


def artifact_label(*, providers: set[str], all_public: bool) -> str:
    """The shipped artifact's label, derived from the SHIP MASK's composition -- not from names."""
    if all_public:
        return "public"
    if "gradientsports" in providers:
        return "full"
    return "sc_extended"


def assert_public_corpus(
    visibility: dict[tuple[str, str], str], *, expect_full_public_arm: bool = False
) -> None:
    """No match may claim `public` unless it is one of the registered 17 (spec 3.2, reviewer m4).

    SCOPE (rev 2): the check is a SUBSET assertion by default, not equality. An equality check run
    unconditionally SystemExits on every legitimate partial run -- a two-match test corpus, a
    `--providers gradientsports` run, a `--max-per-provider 1` smoke -- and it would have killed
    this PR's own flagship licensing test before its assertion was ever reached.

    `expect_full_public_arm=True` (the maintainer run, which loads every public provider) adds the
    equality check: all 17 must be present. That is the direction that catches pining DRIFT --
    a public match silently disappearing or being added.

    Both directions matter, and they are different questions:
      * subset   -> "nothing unregistered is calling itself public"   (a LICENSING failure)
      * equality -> "the registered public set is all still there"    (a DRIFT failure)
    """
    seen = {(p, m) for (p, m), v in visibility.items() if v == "public"}
    registered = {(prov, mid) for prov, ids in PUBLIC_CORPUS.items() for mid in ids}

    unregistered = seen - registered
    if unregistered:
        raise SystemExit(
            f"UNREGISTERED public match(es): {sorted(unregistered)}. A match claiming `public` that "
            "is not in PUBLIC_CORPUS would enter the redistributable training arm. Refusing to run."
        )
    if expect_full_public_arm and seen != registered:
        raise SystemExit(
            f"PUBLIC_CORPUS drift: missing {sorted(registered - seen)}. The registered public set "
            "must be fully present in a maintainer run -- a change here alters what 'public' means."
        )
```

- [ ] **Step 4: Rewire both trainers**

In **both** `train_xshot_occurrence.py` and `train_xcross_attempt.py`:

Delete the constant (line 30 in each):

```python
_PUBLIC_PROVIDERS = {"skillcorner", "idsse"}   # DELETE -- do not leave it importable
```

`_extract` must return a per-row `match_ids` array. Today it initialises **four** accumulators
(`train_xshot_occurrence.py:52`: `parts_x, parts_y, parts_g, parts_p = [], [], [], []`) — so a fifth
must be **added to that line**, not merely appended to:

```python
    parts_x, parts_y, parts_g, parts_p, parts_m = [], [], [], [], []      # :52 -- five now
    ...
            parts_p.append(np.array([prov] * len(X)))                      # :67 -- existing
            parts_m.append(np.array([str(mid)] * len(X)))                  # NEW, beside it
    ...
    return (                                                               # :72 -- was a 4-tuple
        X,
        np.concatenate(parts_y),
        np.concatenate(parts_g),
        np.concatenate(parts_p),
        np.concatenate(parts_m),
    )
```

Update the unpack site in `main()` and the feature-cache save/load (Task 11's cache-schema bump
covers the new `match_ids.npy`).

Replace the arm split (xS `:276`, xCross `:344`):

```python
    from _corpus import artifact_label, assert_public_corpus, is_public_row

    vis = match_visibility(sorted(set(providers.tolist())))
    # Subset check ALWAYS (nothing unregistered may call itself public). Equality only on the
    # maintainer run that loads every public provider -- otherwise a two-match test corpus, a
    # GS-only run, or a --max-per-provider smoke would SystemExit before doing anything.
    loads_full_public_arm = (
        {"skillcorner", "idsse"} <= set(providers.tolist()) and args.max_per_provider is None
    )
    assert_public_corpus(vis, expect_full_public_arm=loads_full_public_arm)
    is_public = is_public_row(providers=providers, match_ids=match_ids, visibility=vis)
```

Replace the label branch (xS `:313`, xCross `:398`):

```python
        ship_provs = set(providers[ship_mask].tolist())
        shipped = artifact_label(providers=ship_provs, all_public=bool(is_public[ship_mask].all()))
```

- [ ] **Step 5: Run**

```bash
.venv/Scripts/python.exe -m pytest tests/scripts/test_corpus_taxonomy.py -q
.venv/Scripts/python.exe -c "import sys; sys.path.insert(0,'scripts'); import train_xshot_occurrence, train_xcross_attempt; print('import OK')"
grep -rn "_PUBLIC_PROVIDERS" scripts/ && echo "STILL PRESENT -- delete it" || echo "deleted"
```

Expected: tests pass; imports OK; the grep prints `deleted`.

- [ ] **Step 6: Lint, type, stage**

```bash
.venv/Scripts/python.exe -m ruff check scripts/ tests/scripts/
.venv/Scripts/python.exe -m ruff format scripts/ tests/scripts/
.venv/Scripts/python.exe -m pyright scripts/_corpus.py scripts/train_xshot_occurrence.py scripts/train_xcross_attempt.py
git add scripts/_corpus.py scripts/train_xshot_occurrence.py scripts/train_xcross_attempt.py tests/scripts/test_corpus_taxonomy.py
```

---

## Task 10: The registered selection rules as pure functions

Every rule the spec registers must be a pure, table-tested function — not prose an implementer re-derives inside a training loop.

**Files:**
- Create: `scripts/_paired.py`
- Test: `tests/scripts/test_paired_rules.py` (create)

- [ ] **Step 1: Write the failing test**

```python
"""The registered decision rules (spec 4.1, 4.3). Pure functions; no I/O."""

import pytest

from _paired import clears_rule, fixed_sequence_ship, ghost_admission


def test_clears_rule_needs_k_minus_1_positive_folds_AND_a_positive_mean():
    assert clears_rule([0.01, 0.01, 0.01, 0.01, -0.001]) is True    # 4/5 positive, mean > 0
    assert clears_rule([0.01, 0.01, 0.01, -0.01, -0.01]) is False   # only 3/5
    assert clears_rule([0.001, 0.001, 0.001, 0.001, -0.02]) is False  # 4/5 but mean < 0


def test_fixed_sequence_stops_when_sc_extended_fails():
    """Registered cost: `full` cannot ship if sc_extended fails, even if its own deltas clear."""
    ship, why = fixed_sequence_ship(
        sc_extended=[-0.01] * 5,                       # fails
        full=[0.02] * 5,                               # would have cleared
        full_vs_sc=[0.02] * 5,
    )
    assert ship == "public"
    assert "sc_extended failed" in why


def test_full_displaces_sc_extended_only_by_sign_consistency_not_a_mean():
    """The rev-1 bare-mean tie-break is GONE (spec 4.1). A higher mean is not enough."""
    ship, _ = fixed_sequence_ship(
        sc_extended=[0.02] * 5,
        full=[0.03] * 5,
        full_vs_sc=[0.05, -0.01, -0.01, -0.01, -0.01],  # higher MEAN, but 1/5 folds -> fails
    )
    assert ship == "sc_extended", "ties go to less data, not to noise"


def test_full_ships_when_it_dominates_fold_by_fold():
    ship, _ = fixed_sequence_ship(
        sc_extended=[0.02] * 5, full=[0.03] * 5, full_vs_sc=[0.01] * 5
    )
    assert ship == "full"


def test_ghost_admission_requires_demonstrated_improvement_not_a_wash():
    # delta = MAE_expanded - MAE_baseline; negative = better
    assert ghost_admission([-0.1, -0.1, -0.1, -0.1, 0.01]) is True
    assert ghost_admission([0.0, 0.0, 0.0, 0.0, 0.0]) is False       # a wash keeps the status quo
    assert ghost_admission([-0.1, -0.1, 0.1, 0.1, 0.1]) is False


def test_admission_ignores_nan_folds_rather_than_failing_on_them():
    """M5: a degenerate (single-class) fold must DROP OUT, not flip the verdict to 'don't ship'.

    The pre-existing _paired_data_effect drops NaN folds explicitly. A rewrite that appends them
    unconditionally makes one bad fold veto the whole run -- the mirror image of a vacuous pass,
    and equally wrong.
    """
    assert ghost_admission([-0.1, -0.1, -0.1, float("nan"), -0.1]) is True


def test_the_interpolator_tell_is_a_DIAGNOSTIC_not_a_gate():
    """Spec rev 5: the refusal was RETIRED because it could never change a verdict.

    Admission already requires detected-only improvement, so `improves_all and not
    improves_detected` was reachable only when the fall-through returned False anyway. And under
    rev 3's detected-only TRAINING rule the mechanism it guarded no longer exists -- the model
    never sees an interpolated target.

    What remains is a reason string, so the record can distinguish 'no improvement' from 'improved
    only where the keeper was invented'. It decides NOTHING, and this test says so out loud.
    """
    verdict, reason = ghost_admission_report(
        detected_only_deltas=[0.1] * 5,     # no improvement on SEEN keepers
        all_frames_deltas=[-0.2] * 5,       # 'improves' on interpolated ones
    )
    assert verdict is False
    assert "interpolated" in reason         # the DIAGNOSTIC fires...
    # ...and the verdict is identical without it -- which is precisely why it is not a gate.
    assert ghost_admission([0.1] * 5) is False
```

- [ ] **Step 2: Run it and watch it fail** — `ModuleNotFoundError: No module named '_paired'`.

- [ ] **Step 3: Implement `scripts/_paired.py`**

```python
"""The registered decision rules (spec 4.1 and 4.3). Pure; table-tested; no I/O.

These live in one place BECAUSE they decide what ships. A rule re-derived inside a training loop
is a rule nobody can review.
"""

from __future__ import annotations

from collections.abc import Sequence


def clears_rule(deltas: Sequence[float]) -> bool:
    """The unchanged 4.9.0/4.18.0 rule: positive in >= K-1 of K folds AND a positive mean."""
    k = len(deltas)
    if k < 2:
        return False
    n_pos = sum(1 for d in deltas if d > 0.0)
    return n_pos >= k - 1 and (sum(deltas) / k) > 0.0


def fixed_sequence_ship(
    *, sc_extended: Sequence[float], full: Sequence[float], full_vs_sc: Sequence[float]
) -> tuple[str, str]:
    """Fixed-sequence selection (spec 4.1). Order is pre-registered; stop at the first failure.

    Testing two shipping candidates independently would roughly double the noise-win rate (a
    single candidate clears the sign rule ~19% of the time under a symmetric null). A fixed
    sequence holds the error rate at the single-test level with no alpha correction.

    Registered cost: if `sc_extended` fails, `full` CANNOT ship on this registration, even if its
    own deltas clear. That outcome is recorded as a finding that triggers a NEW registration.
    """
    if not clears_rule(sc_extended):
        return "public", "sc_extended failed the rule; the sequence stops (full cannot ship here)"
    if clears_rule(full) and clears_rule(full_vs_sc):
        return "full", "full clears vs public AND dominates sc_extended fold-by-fold"
    return "sc_extended", "sc_extended clears; full does not dominate it -- ties go to less data"


def ghost_admission(detected_only_deltas: Sequence[float]) -> bool:
    """Ghost-GK admission (spec 4.3). Deltas are MAE_expanded - MAE_baseline; NEGATIVE is better.

    Admit only on a DEMONSTRATED improvement under sign-consistency, measured on frames where the
    keeper was actually SEEN -- a wash leaves the 81-match status quo in place. (The rev-1 fixed
    0.05 m band was never costed: the gate's own tolerated fold noise is ~10x that band.)

    NaN folds (single-class, no usable score) DROP OUT -- they must not veto the run.
    """
    usable = [d for d in detected_only_deltas if d == d]   # NaN != NaN
    return clears_rule([-d for d in usable])


def ghost_admission_report(
    detected_only_deltas: Sequence[float], all_frames_deltas: Sequence[float] | None = None
) -> tuple[bool, str]:
    """The verdict PLUS a reason string. The reason is a DIAGNOSTIC and decides nothing (spec rev 5).

    The rev-2 'interpolator tell' REFUSAL was retired: admission already requires detected-only
    improvement, so the refusal branch was reachable only when the fall-through returned False
    anyway -- it could never change a verdict. And under rev 3's detected-keeper TRAINING rule the
    mechanism is gone: the model never sees an interpolated target.

    What survives is the ability to say WHY a candidate failed -- 'no improvement anywhere' reads
    very differently from 'improved only where the keeper was invented' -- and that distinction
    belongs in the record, not in the gate.
    """
    verdict = ghost_admission(detected_only_deltas)
    if verdict:
        return True, "improved on detected keepers under sign-consistency"
    if all_frames_deltas is not None and ghost_admission(all_frames_deltas):
        return False, (
            "no improvement on DETECTED keepers, but improved on all frames -- the gain sits on "
            "interpolated (invented) keeper positions. Diagnostic only; the verdict is unchanged."
        )
    return False, "no improvement on detected keepers"
```

- [ ] **Step 4: Run**

```bash
.venv/Scripts/python.exe -m pytest tests/scripts/test_paired_rules.py -q
```

Expected: 6 passed.

- [ ] **Step 5: Lint, type, stage**

```bash
.venv/Scripts/python.exe -m ruff check scripts/_paired.py tests/scripts/test_paired_rules.py
.venv/Scripts/python.exe -m ruff format scripts/_paired.py tests/scripts/test_paired_rules.py
.venv/Scripts/python.exe -m pyright scripts/_paired.py
git add scripts/_paired.py tests/scripts/test_paired_rules.py
```

---

## Task 11: Three candidates, nested HPO, and the cache schema

**Files:**
- Modify: `scripts/train_xshot_occurrence.py` (`_hpo_once` ~:73, `_paired_data_effect` ~:166-210, the candidate block ~:270-317, the feature cache ~:245-251)
- Modify: `scripts/train_xcross_attempt.py` (the byte-mirrored twin: `_paired_data_effect` ~:216-261, candidates ~:387-400)

- [ ] **Step 1: Cache schema — write the failing test**

`tests/scripts/test_cache_schema.py` (create):

```python
"""A stale _feature_cache/ must be a MISS (spec 3.2). The 2026-07-13/14 owner runs already
populated one on the DGX, and it has no visibility column -- reusing it would silently
re-introduce the provider-name arm split."""

import json

from _cache import cache_is_valid, write_cache_meta


def test_absent_meta_is_a_miss(tmp_path):
    (tmp_path / "features.parquet").write_bytes(b"x")   # the OLD predicate would say "hit"
    assert cache_is_valid(tmp_path, fingerprint="abc") is False


def test_schema_version_mismatch_is_a_miss(tmp_path):
    write_cache_meta(tmp_path, fingerprint="abc")
    meta = json.loads((tmp_path / "cache_meta.json").read_text())
    meta["schema_version"] = 0
    (tmp_path / "cache_meta.json").write_text(json.dumps(meta))
    assert cache_is_valid(tmp_path, fingerprint="abc") is False


def test_corpus_fingerprint_mismatch_is_a_miss(tmp_path):
    write_cache_meta(tmp_path, fingerprint="abc")
    assert cache_is_valid(tmp_path, fingerprint="DIFFERENT") is False


def test_matching_meta_is_a_hit(tmp_path):
    write_cache_meta(tmp_path, fingerprint="abc")
    assert cache_is_valid(tmp_path, fingerprint="abc") is True
```

- [ ] **Step 2: Implement `scripts/_cache.py`**

```python
"""Feature-cache validity (spec 3.2).

The trainers gated on `(cache / "features.parquet").exists()`. That predicate cannot see a
schema change -- and the cache now carries a per-row `visibility` array that decides the public
arm. An absent or mismatched cache_meta.json is a MISS.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

CACHE_SCHEMA_VERSION = 2   # 1 -> 2: adds visibility.npy + match_ids.npy


def corpus_fingerprint(rows: list[tuple[str, str, str]]) -> str:
    """Stable hash of the (provider, match_id, visibility) triples the cache was built from."""
    payload = json.dumps(sorted(rows), separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def write_cache_meta(cache_dir: Path, *, fingerprint: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "cache_meta.json").write_text(
        json.dumps({"schema_version": CACHE_SCHEMA_VERSION, "corpus_fingerprint": fingerprint}, indent=2)
    )


def cache_is_valid(cache_dir: Path, *, fingerprint: str) -> bool:
    cache_dir = Path(cache_dir)
    # The payload must exist AND the metadata must match. Keeping the features.parquet check means
    # a half-written cache (meta present, payload missing -- an interrupted extraction) is a MISS,
    # not a crash on load.
    if not (cache_dir / "features.parquet").exists():
        return False
    meta_path = cache_dir / "cache_meta.json"
    if not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text())
    return (
        meta.get("schema_version") == CACHE_SCHEMA_VERSION
        and meta.get("corpus_fingerprint") == fingerprint
    )
```

Wire it into both trainers' cache branch, replacing the bare `.exists()` predicate, and save/load `visibility.npy` + `match_ids.npy` alongside the existing arrays.

- [ ] **Step 3: Nested HPO — replace `_paired_data_effect` in both trainers**

The historical `_paired_data_effect` tuned once, outside the outer CV, on the public arm — letting
`public` tune on exactly the 17 matches that *are* the evaluation universe (differential leakage,
favouring `public`, deciding what ships; spec §4.1 / reviewer M4). The replacement tunes **inside**
each outer fold. It is written out in full in Steps **3a** (a module-level `_fit_score`) and **3b**
(the new `_paired_data_effect` body) immediately below — one authoritative copy, not a sketch.

**Provenance of the pieces this rewrite touches** (quoted from source — the signature-match column
the last review asked for):

| Symbol | Definition line | Signature / fact |
|---|---|---|
| `_hpo_once` | `train_xshot_occurrence.py:75` | `(X, y, groups, out_dir, tag, n_trials, *, negative_subsample=None, seed=42)`. Builds a sqlite store at `str(out_dir / f"study_{tag}.db")` (`:102`) — so **`out_dir` must be a real Path**; `None / "..."` raises. |
| `_fit_score` closure | `train_xshot_occurrence.py:185` | `(Xtr, ytr, te_idx)` — 3 args, lives **inside** `_paired_data_effect` (`:166`), and hardcodes `_pinned_params(shared_params)` (`:194`). It is **deleted** by the wholesale replacement below and cannot fit a candidate at its own params. |
| `_pinned_params` | `tracking/_xshot_occurrence.py:328` | coerces float HPO ranges to XGBoost's int types. |
| `subsample_negatives` | `tracking/_xshot_occurrence.py` | train-fold negative thinning. |

The M4 fix requires scoring each candidate at **its own** tuned params (nested) *and* at the public
params (shared). The 3-arg closure cannot express "at its own params" — so it is not reusable, and
the rev-2 note that said "generalise it, do not replace it" was **wrong**. Extract a real
module-level helper instead. This is the task the 45–60 DGX-hour budget exists for; write it out.

**Step 3a — add a module-level `_fit_score`** (a faithful extraction of the closure's body, with
`params` and the eval arrays made explicit):

```python
def _fit_score(X_tr, y_tr, X_te, y_te, params, *, negative_subsample=None, seed=42) -> float:
    """Fit XGBoost at `params` on (X_tr, y_tr); return PR-AUC on (X_te, y_te).

    A module-level extraction of the old _paired_data_effect closure (train_xshot_occurrence.py:185),
    with two things made EXPLICIT that the closure hardcoded:
      * `params`  -- so one code path serves both protocols: the candidate's OWN tuned params
                     (nested, PRIMARY, decides the ship) and the public params (shared, reported).
      * the eval slice (X_te, y_te) -- the closure captured Xp/yp from its enclosing scope.

    Degenerate (single-class) folds return NaN; the caller drops them. Preserves the closure's
    base_score override and train-only negative subsampling (PR-S80 M3).
    """
    import numpy as np
    import xgboost as xgb
    from sklearn.metrics import average_precision_score

    from silly_kicks.tracking._xshot_occurrence import _pinned_params, subsample_negatives

    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return float("nan")
    if negative_subsample:
        X_tr, y_tr, _ = subsample_negatives(X_tr, y_tr, y_tr, fraction=negative_subsample, seed=seed)
        if len(np.unique(y_tr)) < 2:
            return float("nan")
    p_ = dict(_pinned_params(params))
    p_["base_score"] = float(y_tr.mean())   # XGBoost's default base_score is wrong for this balance
    clf = xgb.XGBClassifier(**p_)
    clf.fit(X_tr.to_numpy(float), y_tr)
    return float(average_precision_score(y_te, clf.predict_proba(X_te.to_numpy(float))[:, 1]))
```

**Step 3b — replace `_paired_data_effect`'s body** with the nested loop. Note `out_dir` is a **real
Path**, and the per-fold, per-candidate `tag` makes every study file unique (no collision, no `None`):

```python
def _paired_data_effect(
    X, y, groups, is_public, match_ids, *, candidates, n_trials, out_dir,
    negative_subsample=None, seed=42,
) -> dict:
    """Nested-HPO paired comparison on the common public held-out folds (spec 4.1).

    For each outer fold k, EVERY candidate is tuned on its own training data with fold k's public
    games EXCLUDED, then fitted at those params and scored on fold k. No candidate's params ever
    see the fold they are scored on. `candidates` maps name -> row mask. Returns, per candidate,
    per-fold PR-AUC deltas vs `public` under both protocols:
      * "nested"        -- PRIMARY, decides the ship (each candidate at ITS OWN params)
      * "shared_params" -- REPORTED for comparability with 4.9.0/4.18.0 (candidate at PUBLIC params)
    """
    from sklearn.model_selection import StratifiedGroupKFold

    Xp, yp, gp = X[is_public], y[is_public], groups[is_public]
    k = max(2, min(5, len(np.unique(gp))))
    skf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=42)
    out = {name: {"nested": [], "shared_params": []} for name in candidates}

    for fold, (_tr, te) in enumerate(skf.split(Xp, yp, gp)):
        te_games = set(np.asarray(gp)[te].tolist())
        trainable = ~(is_public & np.isin(groups, list(te_games)))
        X_te, y_te = Xp.iloc[te], yp[te]                 # the PUBLIC held-out fold (positional)

        fold_params = {
            name: _hpo_once(
                X[mask & trainable], y[mask & trainable], groups[mask & trainable],
                out_dir, f"{name}_f{fold}", n_trials,   # real dir + unique tag -> no collision
                negative_subsample=negative_subsample, seed=seed,
            )
            for name, mask in candidates.items()
        }
        d_pub = _fit_score(
            X[candidates["public"] & trainable], y[candidates["public"] & trainable],
            X_te, y_te, fold_params["public"], negative_subsample=negative_subsample, seed=seed,
        )
        for name, mask in candidates.items():
            if name == "public":
                continue
            m = mask & trainable
            d_nested = _fit_score(X[m], y[m], X_te, y_te, fold_params[name],
                                  negative_subsample=negative_subsample, seed=seed)
            d_shared = _fit_score(X[m], y[m], X_te, y_te, fold_params["public"],
                                  negative_subsample=negative_subsample, seed=seed)
            if not (np.isnan(d_pub) or np.isnan(d_nested)):
                out[name]["nested"].append(float(d_nested - d_pub))
            if not (np.isnan(d_pub) or np.isnan(d_shared)):
                out[name]["shared_params"].append(float(d_shared - d_pub))
    return out
```

`X[m]` boolean-masks the DataFrame; `y[m]` masks the ndarray (`y` is `np.concatenate(parts_y)`);
`X_te`/`y_te` index the **public** frame positionally, exactly as the old closure did. Keep those two
index spaces separate — crossing them is the easiest way to produce a silently wrong delta.

`out_dir` is passed from `main()` (the run's `--output-dir / "xshot_occurrence_v1"`); the caller in
Task 11 Step 4 supplies it. `metrics.json` records `n_usable_folds` per candidate, so a corpus that
cannot produce ≥2 usable folds **fails** the rule rather than passing vacuously.

**Kill-line check:** in `_fit_score`, replace `params` with `fold_params["public"]`-equivalent (i.e.
score every candidate at public params) and the `nested` and `shared_params` columns become
identical — the M4 fix collapses. A unit test on `_paired_data_effect` over a synthetic corpus where
one candidate genuinely wants more capacity must show `nested != shared_params` for that candidate;
that difference is the whole point of the task.

- [ ] **Step 4: Wire the three candidates + the fixed sequence**

Replace the `two_candidate` block in both trainers:

```python
    from _paired import clears_rule, fixed_sequence_ship

    is_sc_private = (providers == "skillcorner") & ~is_public
    is_gs = providers == "gradientsports"
    cand_masks = {
        "public": is_public,
        "sc_extended": is_public | is_sc_private,
        "full": np.ones(len(X), bool),
    }
    paired = _paired_data_effect(
        X, y, groups, is_public, match_ids,
        candidates=cand_masks, n_trials=args.n_trials,
        negative_subsample=ns, seed=seed,
    )
    full_vs_sc = [
        f - s for f, s in zip(paired["full"]["nested"], paired["sc_extended"]["nested"], strict=True)
    ]
    shipped, why = fixed_sequence_ship(
        sc_extended=paired["sc_extended"]["nested"],
        full=paired["full"]["nested"],
        full_vs_sc=full_vs_sc,
    )
    print(f"Fixed-sequence verdict: ship {shipped} -- {why}")
    ship_mask = cand_masks[shipped]
```

The label then comes from Task 9's `artifact_label(...)`, and `metrics["candidates"]["paired"]` records **all three arms under both protocols**, plus `full_vs_sc` and `why`.

- [ ] **Step 5: Run**

```bash
.venv/Scripts/python.exe -m pytest tests/scripts/ -q
.venv/Scripts/python.exe -m pytest tests/tracking/test_xshot_occurrence_integration.py tests/tracking/test_xcross_attempt_integration.py -q -m "not e2e and not slow"
```

Expected: green. The integration tests exercise the trainers on synthetic data — if they were pinning the 2-candidate shape, update them to the 3-arm shape (that is the intended change, not a regression).

- [ ] **Step 6: Lint, type, stage**

```bash
.venv/Scripts/python.exe -m ruff check scripts/ tests/scripts/
.venv/Scripts/python.exe -m ruff format scripts/ tests/scripts/
.venv/Scripts/python.exe -m pyright scripts/
git add scripts/ tests/scripts/
```

---

## Task 12: Ghost-GK — detected targets, keeper-grouped CV, paired admission

**Files:**
- Modify: `scripts/train_ghost_gk.py` (extraction ~:222-290, CV ~:306-320, metrics ~:500-545)

- [ ] **Step 1: Detected-keeper targets**

The extraction loop initialises four accumulators at `train_ghost_gk.py:224-227` — the names are
**`all_features`, `all_labels`, `all_game_ids`, `all_providers`** (quoted from source; it is *not*
`all_feats`, and the loop appends via `all_features.append(feats)` at `:259`). Add a fifth,
`all_keepers: list[str] = []`, beside them, and rewrite the per-match body:

```python
from silly_kicks.tracking._ghost_gk import keeper_detection_mask

                feats, labs, meta = prepare_ghost_gk_training_data(
                    frames, home_team_id=home, actions=acts,
                    subsample_fps=args.subsample_fps, carrier_params=cp,
                    return_meta=True,
                )
                # Detected-keeper targets ONLY (spec 4.3). ~80% of SkillCorner keeper positions
                # are interpolator output; training on them teaches the interpolator. RAISES if
                # the provider carries a detection flag and the pipeline discarded it.
                keep = keeper_detection_mask(meta["gk_visibility"], provider=prov)
                feats = feats[keep].reset_index(drop=True)
                labs = labs[keep].reset_index(drop=True)
                meta = meta[keep].reset_index(drop=True)
                if not len(feats):
                    print(f"  SKIP {prov}/{mid}: no detected-keeper frames")
                    continue
                all_features.append(feats)          # :224 -- NOT all_feats
                all_labels.append(labs)
                all_game_ids.extend([gid] * len(feats))
                all_providers.extend([prov] * len(feats))
                all_keepers.extend(meta["gk_player_id"].astype(str).tolist())
```

After the loop, alongside `groups = np.array(all_game_ids)`:

```python
    keepers = np.array(all_keepers, dtype=object)
```

Persist it in the feature cache next to `groups.npy` (`np.save(cache_dir / "keepers.npy", keepers)`) and load it on the cache-hit branch — the cache-schema bump in Task 11 already invalidates any cache written without it.

- [ ] **Step 2: Write the tests FIRST — Task 12 had none, for four registered rules**

Rev 1 of this plan implemented §4.3's four rules with **zero tests**, and its one assertion was a
tautology: `domain` was *defined* as "keepers not in `expansion_keepers`", then asserted not to
intersect `expansion_keepers`. Empty by construction — the very leakage check the spec insisted be
*"asserted, not assumed."*

Create `tests/scripts/test_ghost_admission_domain.py`:

```python
"""The four registered ghost-GK rules (spec 4.3). Each test names its kill-line."""

import numpy as np
import pytest

from _ghost_domain import common_keeper_domain, keeper_folds


def test_expansion_keepers_are_excluded_from_the_evaluation_domain():
    """KILL-LINE: delete the `k not in expansion` filter -> this MUST fail."""
    keepers = np.array(["alisson", "courtois", "neuer", "courtois"])
    domain, report = common_keeper_domain(keepers, expansion_keepers={"courtois"})
    assert list(domain) == [True, False, True, False]
    assert report.n_excluded_keepers == 1


def test_the_exclusion_is_NON_VACUOUS_on_the_real_overlap():
    """META-ASSERTION. The exclusion only matters if it actually removes someone.

    Courtois is in the WC2022 Gradient Sports corpus AND in 45 of the 98. If a future refactor
    made `expansion_keepers` empty (a silently-missing --expansion-keepers file, say), every test
    above would still pass while the guard did nothing. This is the test that notices.
    """
    keepers = np.array(["alisson", "courtois", "neuer"])
    _, report = common_keeper_domain(keepers, expansion_keepers={"courtois"})
    assert report.n_excluded_keepers > 0, "the domain exclusion removed nobody -- it is inert"

    with pytest.raises(ValueError, match="empty"):
        common_keeper_domain(keepers, expansion_keepers=set())


def test_no_keeper_appears_in_both_train_and_test_folds():
    """KILL-LINE: swap GroupKFold for KFold -> this MUST fail."""
    keepers = np.array([f"gk{i // 20}" for i in range(200)])   # 10 keepers, 20 rows each
    domain = np.ones(len(keepers), bool)
    for tr, te in keeper_folds(keepers, domain, n_splits=5):
        assert not (set(keepers[tr]) & set(keepers[te]))


def test_underpowered_domain_is_reported_not_interpreted():
    keepers = np.array(["a", "b", "c"])         # 3 keepers, 5 folds -> underpowered
    _, report = common_keeper_domain(keepers, expansion_keepers={"z"}, n_splits=5)
    assert report.underpowered is True
```

Add `tests/scripts/test_detected_targets.py`:

```python
"""Detected-keeper training targets (spec 4.3)."""

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._ghost_gk import keeper_detection_mask


def test_undetected_keeper_rows_are_dropped_for_skillcorner():
    """KILL-LINE: delete the `keep = keeper_detection_mask(...)` filter in the extractor loop
    -> this MUST fail (the undetected row survives)."""
    vis = pd.Series([True, False, True])
    keep = keeper_detection_mask(vis, provider="skillcorner")
    assert keep.sum() == 2


def test_a_provider_whose_flag_was_discarded_RAISES_rather_than_training_on_the_interpolator():
    with pytest.raises(ValueError, match="discarded|null"):
        keeper_detection_mask(pd.Series([None, None]), provider="skillcorner")
```

- [ ] **Step 3: Implement `scripts/_ghost_domain.py`** (extracted so it is testable without a fit)

```python
"""The ghost-GK common keeper domain and its folds (spec 4.3). Pure; no I/O."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DomainReport:
    n_domain_keepers: int
    n_excluded_keepers: int
    underpowered: bool


def common_keeper_domain(
    keepers: np.ndarray, *, expansion_keepers: set[str], n_splits: int = 5
) -> tuple[np.ndarray, DomainReport]:
    """Baseline keepers MINUS anyone appearing in the 98 (spec 4.3).

    The two corpora have DIFFERENT keeper populations, so there is no shared domain to hold fixed
    unless we construct one. Courtois is in the WC2022 Gradient Sports corpus AND in 45 of the 98
    -- so a keeper the expanded model trained on could otherwise land in the baseline's TEST fold.

    Raises when `expansion_keepers` is empty: an inert exclusion is worse than none, because every
    downstream assertion still passes while the guard does nothing.
    """
    if not expansion_keepers:
        raise ValueError(
            "common_keeper_domain: expansion_keepers is empty -- the exclusion would be inert. "
            "Pass --expansion-keepers from the Stage-B run, or state explicitly that the corpora "
            "share no keepers (they do share at least Courtois)."
        )
    domain = np.array([str(k) not in expansion_keepers for k in keepers], dtype=bool)
    n_dom = len(set(keepers[domain].tolist()))
    n_exc = len(set(keepers.tolist())) - n_dom
    return domain, DomainReport(
        n_domain_keepers=n_dom,
        n_excluded_keepers=n_exc,
        underpowered=n_dom < n_splits * 2,
    )


def keeper_folds(keepers: np.ndarray, domain: np.ndarray, *, n_splits: int = 5):
    """GroupKFold by KEEPER -- not by match. The target IS keeper positioning, and half the new
    cohort's keeper-slots are three Real Madrid keepers, so match folds would let Courtois appear
    in both train and test."""
    from sklearn.model_selection import GroupKFold

    idx = np.flatnonzero(domain)
    cv = GroupKFold(n_splits=n_splits)
    for tr, te in cv.split(idx, groups=keepers[idx]):
        yield idx[tr], idx[te]
```

Wire it into `train_ghost_gk.py` (replacing the `StratifiedGroupKFold` split when `--keeper-grouped`
is passed), and add `--expansion-keepers PATH` (an `.npy` of keeper ids present in the 98; the
Stage-B run writes it, the baseline run consumes it) and `--keeper-grouped` (default `False`, so the
shipped artifact's headline metrics stay match-grouped and comparable with 4.14.0).

- [ ] **Step 4: Characterise the selection bias (spec rev 5) — a measurement, not a gate**

Detection correlates with the camera seeing the keeper, which correlates with the ball being near
him. So detected frames over-represent the *engaged* keeper and under-represent the deep, off-ball,
sweeper-line regime GKDV cares about. The spec registers this as a **stated limitation**, measured
before Stage B is interpreted. In the trainer, alongside the metrics:

```python
        "detection_selection_bias": {
            "ball_to_keeper_distance_detected_mean": float(d_det.mean()),
            "ball_to_keeper_distance_undetected_mean": float(d_undet.mean()),
            "keeper_depth_detected_mean": float(x_det.mean()),
            "keeper_depth_undetected_mean": float(x_undet.mean()),
            "note": (
                "Detected keeper frames are a SELECTION-BIASED sample (the camera sees the keeper "
                "when the ball is near him). This is a stated limitation, not a gate -- no rule in "
                "this cycle detects it. See spec 4.3 rev 5."
            ),
        },
```

- [ ] **Step 5: Report both scoring schemes**

`metrics.json` gains, alongside the existing match-grouped block:

```python
        "keeper_grouped": {
            "n_domain_keepers": n_domain_keepers,
            "underpowered": bool(n_domain_keepers < args.cv_folds * 2),
            "detected_only_mae_euclidean_per_fold": detected_fold_maes,
            "all_frames_mae_euclidean_per_fold": all_frames_fold_maes,
        },
```

The **admission verdict itself** is computed by the owner-run comparison script from the two runs' `metrics.json` via `_paired.ghost_admission(...)` — the trainer reports, it does not decide.

- [ ] **Step 6: Fix the size gate (reviewer m3)**

`artifact_bytes = sum(artifact_dir.rglob("*"))` sweeps in `_feature_cache/` (~220 MB) — which is why both variants reported `artifact_size_lt_15mb: FAIL` on 2026-07-13 while the true payload was 14.64 MB. Replace it:

```python
    # Measure the SHIPPED file set, not a directory walk. The feature cache lives inside
    # ghost_gk_v1/ and is ~220 MB; including it made the gate meaningless. The bundled payload
    # is exactly these files (compare silly_kicks/tracking/_ghost_gk_weights/default/).
    _SHIPPED = ("rfcde_weights.npz", "metadata.json", "SHA256SUMS")
    artifact_bytes = sum((artifact_dir / f).stat().st_size for f in _SHIPPED if (artifact_dir / f).exists())
```

- [ ] **Step 7: Run**

```bash
.venv/Scripts/python.exe -m pytest tests/scripts/test_ghost_admission_domain.py tests/scripts/test_detected_targets.py tests/tracking/test_ghost_gk.py tests/tracking/test_ghost_gk_r3.py -q -m "not e2e and not slow"
.venv/Scripts/python.exe scripts/train_ghost_gk.py --help
```

Expected: green; `--help` shows `--expansion-keepers` and `--keeper-grouped`.

- [ ] **Step 8: Lint, type, stage**

```bash
.venv/Scripts/python.exe -m ruff check scripts/train_ghost_gk.py
.venv/Scripts/python.exe -m ruff format scripts/train_ghost_gk.py
.venv/Scripts/python.exe -m pyright scripts/train_ghost_gk.py
git add scripts/train_ghost_gk.py
```

---

## Task 13: Fold in the corpus-pin patch

The `--match-ids-json` work already exists in the working tree (written during the TF-19 owner runs, when the pining SkillCorner listing grew mid-run and killed three trainers). It is uncommitted.

**Files:** `scripts/_loader_pining_to_cache.py`, `scripts/train_xshot_occurrence.py`, `scripts/train_xcross_attempt.py` (already modified)

- [ ] **Step 1: Verify the flag still works after Tasks 5–11 reshaped the loader**

```bash
.venv/Scripts/python.exe scripts/train_xshot_occurrence.py --help | grep match-ids-json
.venv/Scripts/python.exe scripts/_loader_pining_to_cache.py --help | grep match-ids-json
```

Expected: the flag is present in both.

- [ ] **Step 2: Confirm it composes with `match_visibility`**

The pin file selects *which* matches load; `visibility` decides *which arm* they land in. They are orthogonal — but assert it once, in `tests/scripts/test_corpus_taxonomy.py`:

```python
def test_the_corpus_pin_and_the_visibility_arm_are_orthogonal():
    """The pin says WHICH matches load; visibility says WHICH ARM they join. A pinned private
    match is still private."""
    vis = {("skillcorner", "1021404"): "private"}
    got = is_public_row(
        providers=np.array(["skillcorner"]), match_ids=np.array(["1021404"]), visibility=vis
    )
    assert list(got) == [False]
```

- [ ] **Step 3: Stage**

```bash
git add scripts/_loader_pining_to_cache.py scripts/train_xshot_occurrence.py scripts/train_xcross_attempt.py tests/scripts/test_corpus_taxonomy.py
```

---

## Task 14: Docs

**Files:** `docs/superpowers/adrs/ADR-038-skillcorner-corpus-and-visibility.md` (create), `CLAUDE.md`, `TODO.md`, `CHANGELOG.md`, `docs/research/skillcorner_corpus/` (create), the spec + plan + the three review documents.

- [ ] **Step 1: ADR-038.** One subsection each: (1) the 98 are owner-tier, so the public arm and the prior paired verdicts are unaffected; (2) the licensing control (visibility-keyed, fail-closed, `_PUBLIC_PROVIDERS` deleted) — **including the verified trace** that `provset <= _PUBLIC_PROVIDERS` labelled an sc_extended-shaped run `"public"`; (3) the clamp split, with the measured damage (11.31% of ball rows, 9.00 m, goal-vs-save erased) and the rule *tracking never calls `_transform_coords`*; (4) pitch-dimension scaling + the provenance basis for 104 m + the open SkillCorner question; (5) the detection finding (19.6% GK) and what it vindicates (GS-only measurement, ADR-024's SkillCorner keeper-origin distrust); (6) the S1 recalibration + rate-gate **and its pinned limitation** (blind to pitch-dim errors — say so, or someone will assume coverage); (7) the registered protocol (fixed sequence, nested HPO, detected-keeper ghost admission) with a pointer to `scripts/_paired.py` as the executable statement of it; (8) Hyrum: the lakehouse re-materializes SkillCorner frames (geometry moves up to 2.0 m on non-standard pitches — a correctness fix).

- [ ] **Step 2: Research records.** Copy the measurement artifacts out of the session scratchpad into `docs/research/skillcorner_corpus/`: the manifest dump, the pitch-dimension census, the detection rates, the clamp measurement, the kloppy-divergence measurement, and the S1 calibration + power tables. These are the evidence behind every registered number; a reader must be able to check them.

- [ ] **Step 3: CLAUDE.md** — a PR-S115 bullet in the tracking/providers section: the native SkillCorner route (visibility + ball_z + single-sourced coordinates), the clamp rule, the visibility-keyed corpus taxonomy, the registered-protocol modules.

- [ ] **Step 4: TODO.md** — record the two open items this cycle *creates*: **ask SkillCorner** which pitch length their coordinates are normalised against (104 m metadata vs kloppy's ~103.5 m, non-affine — nobody has characterised it), and the kloppy gateway's `visibility: None` limitation for external users (the native builder is the supported path).

- [ ] **Step 5: CHANGELOG** — a new section at the next free version (verify with `git ls-remote --tags origin`; expect `4.48.0`). Flag the two Hyrum events: lakehouse SkillCorner geometry moves; the research corpus's SkillCorner frames change (which is why the owner runs re-baseline in Stage A).

- [ ] **Step 6: Stage**

```bash
git add docs/ CLAUDE.md TODO.md CHANGELOG.md
```

---

## Task 15: Full gate and version bump

- [ ] **Step 1:** `.venv/Scripts/python.exe -m ruff check silly_kicks tests scripts` and `ruff format --check silly_kicks tests scripts` → both clean.
- [ ] **Step 2:** `.venv/Scripts/python.exe -m pyright` (whole repo) → 0 errors.
- [ ] **Step 3:** `.venv/Scripts/python.exe -m pytest tests/ -m "not e2e" -q --benchmark-skip` (background, ~11 min) → all pass.
- [ ] **Step 4:** Bump the version at all five sites (`pyproject.toml`, `silly_kicks/__init__.py`, `TODO.md` header, `CHANGELOG.md` header, then `uv lock`), and stage them.

---

## Task 16: Final review and the gated commit

- [ ] **Step 1:** Run `/final-review`.
- [ ] **Step 2:** Re-run the Task 15 gate.
- [ ] **Step 3:** Write the commit message to a temp file and **present it, with the full diff, to the user. STOP. Do not commit without explicit approval.**

Draft:

```
feat(tracking,scripts)!: SkillCorner owner-tier corpus + visibility surfacing + registered retrain protocol -- silly-kicks 4.48.0 (ADR-038, PR-S115)

Code only; no weights (they land in PR-B after the registered owner runs).

- Licensing control: public-vs-owner is keyed on the manifest's `visibility`, never on the
  provider name. `_PUBLIC_PROVIDERS` is deleted (6 sites). Two of them set the shipped
  artifact's LABEL -- a SkillCorner+IDSSE run containing the 98 restricted matches shipped
  labelled "public". Red-first CI gate.
- Coordinates: `_scale_to_spadl` (affine) is split out of the clamping `_transform_coords`.
  Tracking calls the affine map ONLY -- the clamp snapped 11.31% of ball rows by up to 9.00 m
  and turned a ball nine metres behind the goal into a ball on the goal line.
- Pitch dimensions: the native SkillCorner builder scales instead of offsetting (4 of the 10
  public matches are not 105x68; the goal line was off by up to 2.0 m). Missing dims now RAISE.
- Detection: the pining SkillCorner path routes through the native builder, surfacing
  `is_detected` -> `visibility` (goalkeepers are detected in only 19.6% of frames) and
  recovering `ball_z`.
- S1 geometry gate: `_TOL_BALL` 30.0 -> 15.0 m (the largest real ball excursion is 9.00 m, so
  the gate could not fire), and the deferred rate-gate is implemented -- with its limitation
  pinned by a test: it CANNOT see a pitch-dimension error, and neither can action-frame
  co-location.
- Registered protocol (spec 4): three-arm fixed-sequence paired test with tuning NESTED inside
  the outer CV; ghost-GK trains on detected-keeper targets only, over a common keeper domain,
  with a paired sign-consistency admission rule and an interpolator-tell refusal.

Hyrum: lakehouse re-materializes SkillCorner frames (geometry moves up to 2.0 m on
non-standard pitches -- a correctness fix). Research-corpus SkillCorner frames change, which is
why the owner runs re-baseline in Stage A.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
```

- [ ] **Step 4:** After approval: commit, push, open the PR, squash-merge on green CI, tag, confirm publish. Then the owner runs (spec §4) begin.

---

## Self-review notes

**Spec coverage:** §3.1 → Task 5; §3.2 → Tasks 8, 9, 11(cache); §3.3 → Task 7; §3.4 → Tasks 2, 3; §4.1 → Tasks 10, 11; §4.2 → owner-run (the plan's Stage-A/B split is executed after this PR); §4.3 → Tasks 6, 10, 12; §4.4 → Task 4; §4.5 → registered, no code (measurement is GS-only and this PR adds no measurement path); §5 Hyrum → Task 14; §6 tests → distributed across every task, each naming its mutation.

**Not in this PR by design:** weights (PR-B); the owner runs themselves; the `freeze_frames` / `physical` artifact roles (no consumer); upstreaming `is_detected` into kloppy (documented as a gateway limitation instead).

**Known risks the implementer must not smooth over:**
1. Task 3 will break existing `tracking/skillcorner` fixtures that build bronze without pitch columns. **Update the fixtures** — do not add a silent default.
2. Task 6 changes `prepare_ghost_gk_training_data`'s internals; the `meta` frame must be filtered by the *same* masks as `features`/`labels`, or keeper identity silently misaligns with its row. The tests do not catch a misalignment that preserves length — check it by eye.
3. Task 11's nested HPO multiplies training cost by K. That is the point (spec §4.1), and it is why the owner budget is 45–60 DGX-hours. Do not "optimise" it back to a single outer tune.
