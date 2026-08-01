# ADR-028 Orientation Defect Class — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the ADR-028 orientation defect class — four measured root causes where a value in
one coordinate convention is combined with a value in another — behind a durable gate that is
observed failing before any fix lands.

**Architecture:** Detection first. PR 1 repairs the test fixtures (which currently cannot express the
defect), adds a fail-loud orientation seam, and lands a 33-aggregator registry with two gates: Gate A
(ADR-028 physical mirror) and Gate B (`home_team_id` invariance). The four root causes are registered
`xfail(strict=True)`, so the gate is red-by-marker on day one and each later PR deletes exactly one
marker. PRs 2-4 then correct one root cause each.

**Tech Stack:** Python 3.10-3.14, pandas, numpy, pytest. No new runtime dependencies.

**Spec:** `docs/superpowers/specs/2026-07-29-adr028-orientation-defect-class-design.md`. Read §2
(evidence) and §4 (decisions D1-D5) before starting — every task below implements a numbered decision.

---

## Conventions for every task

**Run the suite:** `python -m pytest tests/ -m "not e2e" -q --benchmark-skip`
**Run one test:** `python -m pytest tests/path/test_x.py::test_name -v`
**Lint before every commit:** `ruff check silly_kicks tests scripts && ruff format --check silly_kicks tests scripts && pyright`

**`pyright` BARE, never `pyright silly_kicks`.** CI runs it over the whole repo **including `tests/`**,
so a scoped local run passes while CI fails. An earlier revision of this line said `pyright silly_kicks`
and that is exactly what happened on PR 1: one `list[FrameAwareTransformer]` vs
`list[FeatureTransfomer]` invariance error in a new test file, invisible to the scoped run, red on CI.
The repo has hit this before (ADR-036 cycle).

**Branch:** work on a branch, never `main`. One branch per PR group (`pr1-detection`,
`pr2-cover-shadows`, `pr3-value-corrections`, `pr4-loader-and-weights`).

**Do NOT assign a version number until commit-prep**, and merge `origin/main` first — this repo has
had four version collisions in three days from parallel sessions.

**The xfail discipline (load-bearing).** Registered defects use `pytest.mark.xfail(strict=True)`.
Strict means a passing xfail is a FAILURE, so the engineer who fixes a defect is forced to delete its
marker. Never use bare `xfail`, never `xfail(reason=...)` without `strict=True`.

**Non-vacuity is mandatory.** Every gate added here must be shown to FAIL before it is shown to pass.
Each task's "verify it fails" step is not optional bookkeeping — it is the only evidence the gate can
do its job. A gate that has never been observed red is the exact defect TF-30 (a) existed to remove.

---

## File structure

| File | Responsibility | PR |
|---|---|---|
| `tests/tracking/_provider_inputs.py` | fixture builders — convention fix (D4), `balance_teams`, GS labels (D5) | 1 |
| `silly_kicks/tracking/_warnings.py` | new `OrientationUnresolvedWarning` category | 1 |
| `silly_kicks/tracking/_action_orientation.py` | fail-loud seam (D2) | 1 |
| `tests/tracking/_mirror_registry.py` | **new** — `MirrorEntry` + `MIRROR_ENTRIES` (33 entries) | 1 |
| `tests/tracking/test_mirror_registry.py` | **new** — Gate A, Gate B, two meta-assertions | 1 |
| `tests/datasets/tracking/action_context_slim/metrica_expected.parquet` | regenerated baseline | 1 |
| `CLAUDE.md` | ADR-028 bullet correction (§3.3) | 1 |
| `silly_kicks/tracking/features.py` | RC1 passer reprojection (2 sites); RC3 EPV reflection | 2, 3 |
| `silly_kicks/tracking/_gk_geometry.py` | RC2 reprojection (2 functions) | 3 |
| `scripts/_loader_pining.py` | RC4 SkillCorner orientation | 4 |
| `silly_kicks/tracking/_gk_completion_weights/default/` | retrained weights | 4 |

`tests/tracking/test_action_ltr_mirror_invariance.py` (the existing hand-listed 5-aggregator gate)
is **kept, not replaced** — it carries hand-tuned probes (the off-centre ghost fixture, the OBSO
tolerance history) that the registry does not reproduce. The registry is additive.

---

# PR 1 — Detection

## Task 1: Fixture actions emit action-LTR (D4 convention)

**Files:**
- Modify: `tests/tracking/_provider_inputs.py` (add helper; call it at the end of `synthesize_actions`)
- Test: `tests/tracking/test_provider_inputs_convention.py` (create)

- [ ] **Step 1: Write the failing test**

```python
"""The shared fixture builder must emit ACTION-LTR actions (ADR-028, spec D4)."""

from __future__ import annotations

import pytest

from silly_kicks.tracking._action_orientation import FIELD_LENGTH, FIELD_WIDTH
from tests.tracking._provider_inputs import load_provider_frames, synthesize_actions

PROVIDERS = ["sportec", "metrica", "skillcorner", "gradientsports"]


def _direction_map(frames):
    players = frames[~frames["is_ball"].astype(bool)]
    players = players[players["team_attacking_direction"].notna()]
    out = {}
    for (period, team), grp in players.groupby(["period_id", players["team_id"].astype(str)]):
        out[(period, str(team))] = grp["team_attacking_direction"].iloc[0]
    return out


@pytest.mark.parametrize("provider", PROVIDERS)
def test_synthesized_actions_are_action_ltr(provider):
    """An RTL-attacking team's action coords must be the POINT REFLECTION of its frame position.

    Non-vacuity: the assertion below is meaningless unless at least one RTL action exists, so the
    test asserts that first. Before D5 lands, gradientsports has none -- that is why D5 is a
    prerequisite and why this test is parameterized over all four providers.
    """
    frames = load_provider_frames(provider)
    actions = synthesize_actions(frames)
    directions = _direction_map(frames)

    rtl_rows = [
        row
        for _, row in actions.iterrows()
        if directions.get((row["period_id"], str(row["team_id"]))) == "rtl"
    ]
    assert rtl_rows, f"{provider}: no RTL action in the fixture -- the check would be vacuous"

    for row in rtl_rows:
        own = frames[
            (frames["period_id"] == row["period_id"])
            & (frames["player_id"].astype(str) == str(row["player_id"]))
        ]
        if own.empty:
            continue
        nearest = own.iloc[(own["time_seconds"] - row["time_seconds"]).abs().argsort()[:1]].iloc[0]
        assert row["start_x"] == pytest.approx(FIELD_LENGTH - float(nearest["x"]), abs=1e-6)
        assert row["start_y"] == pytest.approx(FIELD_WIDTH - float(nearest["y"]), abs=1e-6)
```

- [ ] **Step 2: Run it and verify it FAILS**

Run: `python -m pytest tests/tracking/test_provider_inputs_convention.py -v`
Expected: FAIL. `sportec`, `metrica`, `skillcorner` fail the reflection assertion (actions are in
frame convention); `gradientsports` fails the non-vacuity assertion (no RTL action — D5 fixes that in
Task 3). All four failing is the correct starting state.

- [ ] **Step 3: Implement the reflection helper**

Add to `tests/tracking/_provider_inputs.py`, immediately above `synthesize_actions`:

```python
def _to_action_ltr(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """Point-reflect action coords for teams attacking RTL in ``frames`` (ADR-028, spec D4).

    ``synthesize_actions`` stamps coordinates from raw frame rows, which are frame-LTR (home
    attacks +x). SPADL actions are action-LTR (the ACTING team attacks +x), so an away-team
    action must be point-reflected: x -> 105-x AND y -> 68-y. Both axes -- an x-only mirror is
    exact only for a y-symmetric scene, which is how ADR-041's incomplete repair survived.

    Frames with no direction label are left UNCHANGED: this helper cannot invent an orientation,
    and silently guessing one is the RC4 failure mode.
    """
    from silly_kicks.tracking._action_orientation import FIELD_LENGTH, FIELD_WIDTH

    players = frames[~frames["is_ball"].astype(bool)]
    players = players[players["team_attacking_direction"].notna()]
    if players.empty:
        return actions

    rtl = {
        (period, str(team))
        for (period, team), grp in players.groupby(["period_id", players["team_id"].astype(str)])
        if grp["team_attacking_direction"].iloc[0] == "rtl"
    }
    mask = pd.Series(
        [(p, str(t)) in rtl for p, t in zip(actions["period_id"], actions["team_id"], strict=True)],
        index=actions.index,
    )
    if not mask.any():
        return actions

    out = actions.copy()
    for col, extent in (("start_x", FIELD_LENGTH), ("end_x", FIELD_LENGTH),
                        ("start_y", FIELD_WIDTH), ("end_y", FIELD_WIDTH)):
        out.loc[mask, col] = extent - out.loc[mask, col]
    return out
```

- [ ] **Step 4: Call it from `synthesize_actions`**

`synthesize_actions` currently ends with `return pd.DataFrame({...})`. Bind that to a name and
reflect it:

```python
    built = pd.DataFrame(
        {
            # ... existing dict body, UNCHANGED ...
        }
    )
    return _to_action_ltr(built, frames)
```

- [ ] **Step 5: Run the test again**

Run: `python -m pytest tests/tracking/test_provider_inputs_convention.py -v`
Expected: `sportec`, `metrica`, `skillcorner` PASS. `gradientsports` still FAILS on non-vacuity
(no RTL action until Task 3). That is expected and is fixed by Task 3, not here.

- [ ] **Step 6: Commit**

```bash
git add tests/tracking/_provider_inputs.py tests/tracking/test_provider_inputs_convention.py
git commit -m "test(fixtures): synthesize_actions emits action-LTR (ADR-028 D4)"
```

---

## Task 2: `balance_teams` opt-in parameter (D4 coverage)

**Files:**
- Modify: `tests/tracking/_provider_inputs.py` (`synthesize_actions` signature + sampling)
- Test: `tests/tracking/test_provider_inputs_convention.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/tracking/test_provider_inputs_convention.py`:

```python
def test_balance_teams_defaults_off_and_is_byte_identical():
    """The default MUST NOT move any existing baseline (spec D4)."""
    frames = load_provider_frames("sportec")
    from pandas.testing import assert_frame_equal

    assert_frame_equal(synthesize_actions(frames), synthesize_actions(frames, balance_teams=False))


def test_balance_teams_true_produces_both_teams():
    """Opt-in gives a usable away population; the 9:1 default cannot gate orientation."""
    frames = load_provider_frames("sportec")
    balanced = synthesize_actions(frames, balance_teams=True)
    counts = balanced["team_id"].astype(str).value_counts()
    assert len(counts) >= 2, f"expected both teams, got {counts.to_dict()}"
    assert counts.min() >= 3, f"minority team too small to gate on: {counts.to_dict()}"
```

- [ ] **Step 2: Run and verify it fails**

Run: `python -m pytest tests/tracking/test_provider_inputs_convention.py -k balance -v`
Expected: FAIL with `TypeError: synthesize_actions() got an unexpected keyword argument 'balance_teams'`.

- [ ] **Step 3: Implement**

Change the signature:

```python
def synthesize_actions(
    frames: pd.DataFrame,
    n_actions: int = N_ACTIONS_PER_PROVIDER,
    *,
    balance_teams: bool = False,
) -> pd.DataFrame:
```

Replace the `sample = ...` line:

```python
    pool = candidates.drop_duplicates(["period_id", "frame_id"]).reset_index(drop=True)
    if balance_teams:
        # Round-robin across teams. The default path picks the first-listed player per frame,
        # which is team-blind and lands ~9:1 on whichever team sorts first -- an artifact of frame
        # row order, not of the data. An orientation gate needs a real away population.
        groups = [g.reset_index(drop=True) for _, g in pool.groupby(pool["team_id"].astype(str))]
        rows = []
        for i in range(max((len(g) for g in groups), default=0)):
            for g in groups:
                if i < len(g):
                    rows.append(g.iloc[i])
        pool = pd.DataFrame(rows).reset_index(drop=True)
    sample = pool.head(n_actions).reset_index(drop=True)
```

- [ ] **Step 4: Run the balance tests, then the whole tracking suite**

Run: `python -m pytest tests/tracking/test_provider_inputs_convention.py -k balance -v`
Expected: PASS.

Run: `python -m pytest tests/tracking tests/invariants -m "not e2e" -q --benchmark-skip`
Expected: PASS except the four `*_expected.parquet` baseline comparisons, which Task 4 regenerates.
Record exactly which tests fail — Task 4 asserts that only metrica's values move.

- [ ] **Step 5: Commit**

```bash
git add tests/tracking/_provider_inputs.py tests/tracking/test_provider_inputs_convention.py
git commit -m "test(fixtures): balance_teams opt-in for orientation coverage (ADR-028 D4)"
```

---

## Task 3: GS direction labels derived, not hardcoded (D5)

**Files:**
- Modify: `tests/tracking/_provider_inputs.py:71`
- Test: `tests/tracking/test_provider_inputs_convention.py` (append)

- [ ] **Step 1: Write the failing test**

```python
def test_gradientsports_labels_both_directions():
    """`_provider_inputs.py:71` hardcoded "ltr" for BOTH teams, so no GS fixture could
    exercise any orientation path. Derived from geometry: team 100's keeper sits at x~20.5,
    team 200's at x~60.5 in both periods, so 100 attacks ltr and 200 attacks rtl."""
    frames = load_provider_frames("gradientsports")
    players = frames[~frames["is_ball"].astype(bool)]
    by_team = {
        str(team): sorted(grp["team_attacking_direction"].dropna().unique())
        for team, grp in players.groupby(players["team_id"].astype(str))
    }
    assert by_team == {"100": ["ltr"], "200": ["rtl"]}, by_team
```

- [ ] **Step 2: Run and verify it fails**

Run: `python -m pytest tests/tracking/test_provider_inputs_convention.py -k gradientsports_labels -v`
Expected: FAIL — `{'100': ['ltr'], '200': ['ltr']}`.

- [ ] **Step 3: Implement**

In `load_provider_frames`, replace the hardcoded scalar at `:71`:

```python
                # Derived, NOT hardcoded (spec D5). The old scalar "ltr" labelled BOTH teams the
                # same way, so acting_team_attacks_rtl returned all-False and no GS fixture could
                # exercise ADR-028 at all. Team 100's keeper sits at x~20.5 and team 200's at
                # x~60.5 in both periods (outfield medians 32.5 / 72.5), so 100 defends the low
                # end and attacks ltr; 200 is the mirror. This synthetic fixture does NOT swap
                # ends at half-time.
                "team_attacking_direction": df["team_id"].map({100: "ltr", 200: "rtl"}),
```

- [ ] **Step 4: Run both GS tests**

Run: `python -m pytest tests/tracking/test_provider_inputs_convention.py -k "gradientsports" -v`
Expected: both PASS. The Task 1 non-vacuity assertion for `gradientsports` now passes too, because
the fixture finally has an RTL action.

- [ ] **Step 5: Commit**

```bash
git add tests/tracking/_provider_inputs.py tests/tracking/test_provider_inputs_convention.py
git commit -m "test(fixtures): derive gradientsports attacking direction (ADR-028 D5)"
```

---

## Task 4: Regenerate the metrica baseline; prove the other three do not move

**Files:**
- Modify: `tests/datasets/tracking/action_context_slim/metrica_expected.parquet`
- Test: `tests/tracking/test_provider_inputs_convention.py` (append)

- [ ] **Step 1: Record the pre-regeneration diff**

```bash
python -m pytest tests/tracking/test_action_context_expected_output.py -v
```

Expected: metrica FAILS (its 9 away rows moved). Record whether sportec / skillcorner / pff fail.
Per the spec's measurement they must NOT: their single away action is the NaN keeper_save.
**If any of those three fails, STOP** — the convention fix has a wider blast radius than measured
and the plan's premise needs re-checking before regenerating anything.

- [ ] **Step 2: Write the guard test that pins the expected magnitude**

```python
def test_metrica_baseline_moved_and_others_did_not():
    """The convention fix (D4) must move metrica ONLY, and by a physically meaningful amount.

    Pre-fix, metrica action 1 emitted nearest_defender_distance 1.029355 m against a true
    18.328738 m -- the wrong value looked MORE plausible than the right one, which is why it
    survived. This test pins the direction of the correction, not just that something changed.
    """
    import pathlib

    import pandas as pd
    from silly_kicks.tracking.features import nearest_defender_distance

    # __file__-anchored, matching the repo idiom (e.g. tests/scripts/test_build_gkdv_arm_values.py).
    # A CWD-relative path works only when pytest runs from the repo root and breaks silently
    # otherwise.
    repo = pathlib.Path(__file__).resolve().parents[2]
    frames = load_provider_frames("metrica")
    actions = synthesize_actions(frames)
    live = nearest_defender_distance(actions, frames)
    committed = pd.read_parquet(
        repo / "tests" / "datasets" / "tracking" / "action_context_slim" / "metrica_expected.parquet"
    ).set_index("action_id")["nearest_defender_distance"]

    merged = pd.DataFrame({"live": live.to_numpy()}, index=actions["action_id"]).join(committed)
    both = merged.dropna()
    assert len(both) >= 8, f"too few comparable rows: {len(both)}"
    assert (both["live"] - both["committed"]).abs().max() < 1e-9, "baseline is stale"
    assert both["live"].max() > 5.0, (
        "post-fix distances still look like the pre-fix collapse (max was 1.17 m); "
        "the reflection may not be applied"
    )
```

- [ ] **Step 3: Regenerate**

```bash
python scripts/regenerate_action_context_baselines.py
git diff --stat tests/datasets/tracking/action_context_slim/
```

Expected: only `metrica_expected.parquet` changes among the tracked baselines.

**Two things WILL happen that the first draft of this plan did not anticipate, both verified:**

1. **A new `gradientsports_expected.parquet` appears.** The regenerator writes
   `{provider}_expected.parquet` for `provider="gradientsports"`, and that file has never existed —
   the repo carried `pff_expected.parquet` from before the PFF -> Gradient Sports rename, read by
   NOTHING (`grep pff_expected` finds only documentation). Creating it turns ON two tests that were
   silently `pytest.skip`-ing at `test_action_context_expected_output.py:58`, so the GS baseline is
   pinned for the first time. **Keep the new file and `git rm` the orphan** — git records this as a
   rename, which is what it is.
2. **`empirical_action_context_baselines.json` changes.** Metrica's distribution slots move with its
   parquet: `nearest_defender_distance` p50 `1.044 -> 18.349`, `receiver_zone_density` `0 -> 1`,
   `pre_shot_gk_distance_to_shot` `1.169 -> 55.436`. All metrica, all in the corrected direction.

- [ ] **Step 4: Run the baseline tests + the new guard**

Run: `python -m pytest tests/tracking/test_action_context_expected_output.py tests/tracking/test_provider_inputs_convention.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/datasets/tracking/action_context_slim/ tests/tracking/test_provider_inputs_convention.py
git commit -m "test(fixtures): regenerate metrica baseline on action-LTR actions (ADR-028 D4)"
```

---

## Task 5: `OrientationUnresolvedWarning` category

**Files:**
- Modify: `silly_kicks/tracking/_warnings.py`, `silly_kicks/tracking/__init__.py`
- Test: `tests/tracking/test_orientation_fail_loud.py` (create)

- [ ] **Step 1: Write the failing test**

```python
"""D2: the orientation seam must not fail silent (spec §4.2)."""

from __future__ import annotations

import pytest


def test_category_is_public_and_importable():
    from silly_kicks.tracking import OrientationUnresolvedWarning

    assert issubclass(OrientationUnresolvedWarning, UserWarning)


def test_category_is_not_a_subclass_of_the_other_categories():
    """Separate categories on purpose: silencing one signal must not silence another."""
    from silly_kicks.tracking import (
        OrientationUnresolvedWarning,
        SyntheticEPVWarning,
    )

    assert not issubclass(OrientationUnresolvedWarning, SyntheticEPVWarning)
    assert not issubclass(SyntheticEPVWarning, OrientationUnresolvedWarning)
```

- [ ] **Step 2: Run and verify it fails**

Run: `python -m pytest tests/tracking/test_orientation_fail_loud.py -v`
Expected: FAIL with `ImportError: cannot import name 'OrientationUnresolvedWarning'`.

- [ ] **Step 3: Implement**

In `silly_kicks/tracking/_warnings.py`, add to `__all__` (keep it alphabetically sorted:
`OrientationUnresolvedWarning` goes after `MissingFeatureContractWarning`) and append the class:

```python
class OrientationUnresolvedWarning(UserWarning):
    """``acting_team_attacks_rtl`` could not resolve a direction and returned an all-False flip.

    An all-False flip means NO ADR-028 re-projection is applied, so every away-team action's
    geometry silently mixes coordinate conventions. This is not hypothetical: the pining loader
    shipped SkillCorner frames with ``team_attacking_direction`` null on 100% of rows, and the
    resulting features were wrong on every away action with no signal of any kind.

    Warn rather than raise, deliberately: consumers legitimately hold absolute/unlabelled frames
    (ADR-029), and a raise has no reachable remedy inside a converter. Fail-closed belongs in CI.

    Examples
    --------
    Treat an unresolved orientation as fatal in a production pipeline::

        import warnings
        from silly_kicks.tracking import OrientationUnresolvedWarning

        warnings.filterwarnings("error", category=OrientationUnresolvedWarning)
    """
```

In `silly_kicks/tracking/__init__.py`, add `"OrientationUnresolvedWarning"` to `__all__` and to the
existing `from ._warnings import (...)` block, both alphabetically.

- [ ] **Step 4: Run**

Run: `python -m pytest tests/tracking/test_orientation_fail_loud.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add silly_kicks/tracking/_warnings.py silly_kicks/tracking/__init__.py tests/tracking/test_orientation_fail_loud.py
git commit -m "feat(tracking): OrientationUnresolvedWarning category (ADR-028 D2)"
```

---

## Task 6: The fail-loud seam itself (D2)

**Files:**
- Modify: `silly_kicks/tracking/_action_orientation.py:143-160`
- Test: `tests/tracking/test_orientation_fail_loud.py` (append)

- [ ] **Step 1: Write the failing test — all three silent branches**

```python
import numpy as np
import pandas as pd

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking._action_orientation import acting_team_attacks_rtl

PASS_ID = spadlconfig.actiontype_id["pass"]


def _frames():
    base = dict(
        game_id=1, period_id=1, frame_id=10, time_seconds=1.0, frame_rate=25.0, z=0.0,
        speed=0.0, speed_source="native", ball_state="alive", confidence=None, visibility=None,
        source_provider="synthetic", is_goalkeeper_source="native", is_goalkeeper=False,
    )
    return pd.DataFrame([
        {**base, "player_id": 1, "team_id": 1, "is_ball": False, "x": 30.0, "y": 20.0,
         "team_attacking_direction": "ltr"},
        {**base, "player_id": 2, "team_id": 2, "is_ball": False, "x": 70.0, "y": 50.0,
         "team_attacking_direction": "rtl"},
    ])


def _actions():
    return pd.DataFrame([dict(
        game_id=1, period_id=1, action_id=1, team_id=2, player_id=2.0, type_id=PASS_ID,
        result_id=1, start_x=35.0, start_y=18.0, end_x=50.0, end_y=30.0, time_seconds=1.0,
    )])


def test_warns_when_direction_column_is_absent():
    from silly_kicks.tracking import OrientationUnresolvedWarning

    f = _frames().drop(columns=["team_attacking_direction"])
    with pytest.warns(OrientationUnresolvedWarning):
        flip = acting_team_attacks_rtl(_actions(), f)
    assert not flip.any()


def test_warns_when_direction_column_is_all_null():
    """RC4's shape: SkillCorner sets the column to None, so it EXISTS and is entirely NA."""
    from silly_kicks.tracking import OrientationUnresolvedWarning

    f = _frames()
    f["team_attacking_direction"] = None
    with pytest.warns(OrientationUnresolvedWarning):
        flip = acting_team_attacks_rtl(_actions(), f)
    assert not flip.any()


def test_warns_when_join_keys_do_not_align():
    """The branch the first draft of this spec MISSED. Reachable, and silent before D2."""
    from silly_kicks.tracking import OrientationUnresolvedWarning

    f = _frames().drop(columns=["team_id"])
    with pytest.warns(OrientationUnresolvedWarning):
        flip = acting_team_attacks_rtl(_actions(), f)
    assert not flip.any()


def test_warns_when_frames_are_empty_but_actions_are_not():
    """N-S2: 'no frames but plenty of actions' is a caller error, not a no-op."""
    from silly_kicks.tracking import OrientationUnresolvedWarning

    with pytest.warns(OrientationUnresolvedWarning):
        acting_team_attacks_rtl(_actions(), _frames().iloc[0:0])


def test_silent_when_there_are_no_actions_to_flip():
    """The ONE carve-out. Narrower than the original disjunction on purpose."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        flip = acting_team_attacks_rtl(_actions().iloc[0:0], _frames())
    assert flip.empty


def test_healthy_frames_do_not_warn():
    """Non-vacuity for the whole group: the warning must not fire on the normal path."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        flip = acting_team_attacks_rtl(_actions(), _frames())
    assert flip.tolist() == [True]
```

- [ ] **Step 2: Run and verify the four warn-tests FAIL**

Run: `python -m pytest tests/tracking/test_orientation_fail_loud.py -v`
Expected: the four `test_warns_*` FAIL with `DID NOT WARN`. `test_silent_when_there_are_no_actions_to_flip`
and `test_healthy_frames_do_not_warn` PASS already — they pin behaviour that must not change.

- [ ] **Step 3: Implement**

At the top of `silly_kicks/tracking/_action_orientation.py`, add `import warnings` and:

```python
def _warn_unresolved(reason: str) -> None:
    """One message for every silent-failure exit (spec D2).

    Specified by OUTCOME, not by enumerated condition: any all-False return that is not
    "there were no actions to flip" warns. An enumerated fix rots the next time a branch
    is added -- which is exactly how the `:155` join-key branch was missed.
    """
    import warnings as _w

    from ._warnings import OrientationUnresolvedWarning

    _w.warn(
        f"acting_team_attacks_rtl: returning an all-False flip ({reason}). No ADR-028 "
        "re-projection will be applied, so away-team geometry will silently mix coordinate "
        "conventions. Orient the frames first -- convert_to_frames(output_convention='ltr') "
        "or tracking.orient_frames_to_ltr().",
        OrientationUnresolvedWarning,
        stacklevel=3,
    )
```

Then rewrite the guard block (currently `:143-160`):

```python
    flip = pd.Series(False, index=actions.index)
    if len(actions) == 0:
        return flip  # nothing to flip -- the ONE legitimate silent no-op
    if len(frames) == 0:
        _warn_unresolved("frames is empty")
        return flip
    if "team_attacking_direction" not in frames.columns:
        _warn_unresolved("frames has no team_attacking_direction column")
        return flip

    keys = [k for k in ("game_id", "period_id", "team_id") if k in actions.columns and k in frames.columns]
    if "team_id" not in keys or "period_id" not in keys:
        _warn_unresolved("actions and frames share neither team_id nor period_id join keys")
        return flip

    players = frames[~frames["is_ball"].astype(bool)]
    players = players[players["team_attacking_direction"].notna()]
    if players.empty:
        _warn_unresolved("team_attacking_direction is present but entirely null")
        return flip
```

- [ ] **Step 4: Run**

Run: `python -m pytest tests/tracking/test_orientation_fail_loud.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the full suite and triage new warnings**

Run: `python -m pytest tests/ -m "not e2e" -q --benchmark-skip`
Expected: PASS. Any test that now emits `OrientationUnresolvedWarning` is telling you it was
running on unoriented frames — investigate each rather than adding a blanket filter. Record the list
in the commit message; it is the in-repo inventory of unoriented-frame callers.

- [ ] **Step 6: Commit**

```bash
git add silly_kicks/tracking/_action_orientation.py tests/tracking/test_orientation_fail_loud.py
git commit -m "feat(tracking): fail loud when orientation is unresolved (ADR-028 D2)"
```

---

## Task 7: Mirror registry data structure + meta-assertions

**Files:**
- Create: `tests/tracking/_mirror_registry.py`
- Create: `tests/tracking/test_mirror_registry.py`

- [ ] **Step 1: Write the failing meta-assertions**

`tests/tracking/test_mirror_registry.py`:

```python
"""Gate A / Gate B over every registered tracking add_* (spec §6)."""

from __future__ import annotations

import pytest

import silly_kicks.tracking as tracking
from tests.tracking._mirror_registry import MIRROR_ENTRIES


def _public_add_names() -> set[str]:
    return {n for n in tracking.__all__ if n.startswith("add_")}


def test_every_public_add_is_registered():
    """Anti-rot, direction 1: a new aggregator must be classified or CI fails."""
    missing = _public_add_names() - set(MIRROR_ENTRIES)
    assert not missing, f"unregistered add_* (add a MirrorEntry): {sorted(missing)}"


def test_registry_has_no_stale_entries():
    """Anti-rot, direction 2: a removed aggregator must not linger."""
    stale = set(MIRROR_ENTRIES) - _public_add_names()
    assert not stale, f"registry names a non-exported add_*: {sorted(stale)}"


def test_registry_surface_is_the_expected_size():
    """Pins the count so a silent export change is visible in the diff."""
    assert len(_public_add_names()) == 33, sorted(_public_add_names())


@pytest.mark.parametrize("name", sorted(MIRROR_ENTRIES))
def test_entry_declares_a_tolerance_basis(name):
    """A tolerance without a recorded basis is a number nobody can revisit on evidence."""
    entry = MIRROR_ENTRIES[name]
    assert entry.tolerance_basis.strip(), f"{name}: tolerance {entry.tolerance} has no basis"


@pytest.mark.parametrize("name", sorted(MIRROR_ENTRIES))
def test_exempt_columns_carry_a_reason(name):
    entry = MIRROR_ENTRIES[name]
    for col, cls in entry.columns.items():
        if cls == "exempt":
            assert entry.exempt_reasons.get(col, "").strip(), f"{name}.{col}: exempt with no reason"


@pytest.mark.parametrize("name", sorted(MIRROR_ENTRIES))
def test_mirrored_pitch_absolute_columns_declare_a_reflection(name):
    """Without a declared reflection the class is an unjustified `exempt` by another name.

    The vocabulary states a testable contract -- "equals its own reflection" -- so a column in
    this class MUST say what its reflection is, and Gate A then enforces it. Otherwise a future
    contributor silences any awkward column by re-classing it.
    """
    entry = MIRROR_ENTRIES[name]
    for col, cls in entry.columns.items():
        if cls == "mirrored_pitch_absolute":
            spec = entry.reflections.get(col)
            assert spec in ("x", "y") or isinstance(spec, dict), (
                f"{name}.{col}: mirrored_pitch_absolute with no usable reflection spec ({spec!r})"
            )
            assert entry.exempt_reasons.get(col, "").strip(), (
                f"{name}.{col}: mirrored_pitch_absolute with no recorded reason"
            )
```

- [ ] **Step 2: Run and verify it fails**

Run: `python -m pytest tests/tracking/test_mirror_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: tests.tracking._mirror_registry`.

- [ ] **Step 3: Create the registry module**

`tests/tracking/_mirror_registry.py`:

```python
"""Registry backing the ADR-028 mirror gates (spec §6).

Two gates, because one instrument cannot see both defect classes:

* **Gate A** -- physical mirror. Detects CONVENTION MIXING (an action-LTR value combined with a
  frame-LTR one). ``home_team_id`` is swapped, because after a physical mirror the team attacking
  +x really is the other one.
* **Gate B** -- ``home_team_id`` invariance on FIXED canonical frames. Detects IDENTITY-KEYED
  direction inference, which Gate A is structurally blind to: swapping ``home_team_id`` restores
  the very invariant identity-keying assumes, so an identity-keyed aggregator is invariant under
  Gate A whether it is safe or not.

Mirror classes:
  ``invariant``               action-LTR geometry; base and mirror identical. The default.
  ``mirrored_pitch_absolute`` deliberately pitch-absolute; equals its own reflection.
  ``exempt``                  undefined/non-deterministic under mirror -- REQUIRES a reason.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

MirrorClass = str  # "invariant" | "mirrored_pitch_absolute" | "exempt"
HomeRole = str  # "direction_only" | "attribution" | "unused"


@dataclass(frozen=True)
class MirrorEntry:
    """One aggregator's declaration. See the module docstring for the vocabularies."""

    name: str
    call: Callable  # (actions, frames, home_team_id) -> pd.DataFrame
    columns: dict[str, MirrorClass]
    tolerance: float
    tolerance_basis: str
    home_team_id_role: HomeRole
    #: Columns that must be NON-NULL on away rows, or the comparison is vacuous.
    non_vacuity: tuple[str, ...]
    exempt_reasons: dict[str, str] = field(default_factory=dict)
    #: REQUIRED for every ``mirrored_pitch_absolute`` column: how to reflect the mirror-leg value
    #: before comparing it to the base. ``"x"`` -> FIELD_LENGTH - v, ``"y"`` -> FIELD_WIDTH - v,
    #: or a dict for a label swap (e.g. ``{"left": "right", "right": "left"}``).
    #: Without this the class is a reason-free ``exempt`` by another name.
    reflections: dict[str, str | dict] = field(default_factory=dict)
    #: Set for a KNOWN-BROKEN aggregator. Gate A xfails strictly until the fix lands.
    known_defect: str | None = None


MIRROR_ENTRIES: dict[str, MirrorEntry] = {}
```

- [ ] **Step 4: Run the meta-assertions**

Run: `python -m pytest tests/tracking/test_mirror_registry.py -v`
Expected: `test_every_public_add_is_registered` FAILS listing all 33 names. The other meta-assertions
PASS vacuously (empty registry). **That failure is the gate working** — it is the anti-rot property,
observed red before any entry exists.

- [ ] **Step 5: Commit**

```bash
git add tests/tracking/_mirror_registry.py tests/tracking/test_mirror_registry.py
git commit -m "test(tracking): mirror registry scaffold + anti-rot meta-assertions (ADR-028)"
```

---

## Task 8: Gate A — the ADR-028 mirror

**Files:**
- Modify: `tests/tracking/_mirror_registry.py` (fixture builders)
- Modify: `tests/tracking/test_mirror_registry.py` (Gate A)

- [ ] **Step 1: Add the fixture + mirror helpers to `_mirror_registry.py`**

```python
import functools

import numpy as np
import pandas as pd

FIELD_LENGTH, FIELD_WIDTH = 105.0, 68.0
HOME, AWAY = 1, 2


@functools.cache
def canonical_scene():
    """(actions, frames) -- canonical converter shape, DELIBERATELY y-ASYMMETRIC.

    y-asymmetry is not decoration: an x-only reprojection is exact on a y-symmetric scene, and
    ADR-041 shipped precisely that incomplete repair. Only a y-asymmetric oracle catches it.

    Home attacks +x ("ltr"); away attacks -x ("rtl"). Both teams act, so the away rows -- the
    only rows an ADR-028 defect touches -- are a real population, not a single token action.
    """
    from silly_kicks.spadl import config as spadlconfig

    base = dict(
        game_id=1, period_id=1, time_seconds=8.0, frame_rate=25.0, z=0.0, speed=1.0,
        speed_source="native", ball_state="alive", confidence=None, visibility=True,
        source_provider="synthetic", is_goalkeeper_source="native", vx=0.7, vy=-0.4,
    )
    rows = [
        dict(player_id=1, team_id=HOME, is_goalkeeper=True, x=5.0, y=27.0, d="ltr"),
        dict(player_id=50, team_id=AWAY, is_goalkeeper=True, x=100.0, y=41.0, d="rtl"),
    ]
    # y values are spread asymmetrically about y=34 on purpose.
    for i, (x, y) in enumerate([(28, 12), (36, 21), (44, 9), (52, 30), (60, 17),
                                (33, 44), (47, 55), (58, 38), (25, 50), (41, 62)]):
        rows.append(dict(player_id=10 + i, team_id=HOME, is_goalkeeper=False,
                         x=float(x), y=float(y), d="ltr"))
    for i, (x, y) in enumerate([(70, 14), (63, 25), (77, 8), (55, 33), (68, 19),
                                (74, 47), (61, 58), (80, 36), (66, 52), (50, 60)]):
        rows.append(dict(player_id=60 + i, team_id=AWAY, is_goalkeeper=False,
                         x=float(x), y=float(y), d="rtl"))
    rows.append(dict(player_id=np.nan, team_id=np.nan, is_goalkeeper=False,
                     x=52.0, y=23.0, d=None))

    recs = []
    for frame_id, t in ((100, 7.6), (101, 7.8), (102, 8.0)):  # 3 frames: pre-window features need history
        for r in rows:
            rec = {**base, **r, "frame_id": frame_id, "time_seconds": t}
            rec["team_attacking_direction"] = rec.pop("d")
            rec["is_ball"] = pd.isna(rec["team_id"])
            recs.append(rec)
    frames = pd.DataFrame(recs)

    PASS = spadlconfig.actiontype_id["pass"]
    SHOT = spadlconfig.actiontype_id["shot"]
    acts = []
    # Home actions: action-LTR == frame coords.
    acts.append(dict(action_id=1, team_id=HOME, player_id=10.0, type_id=PASS,
                     start_x=28.0, start_y=12.0, end_x=36.0, end_y=21.0))
    acts.append(dict(action_id=2, team_id=HOME, player_id=13.0, type_id=SHOT,
                     start_x=52.0, start_y=30.0, end_x=105.0, end_y=34.0))
    # Away actions: action-LTR == the POINT REFLECTION of the frame position.
    acts.append(dict(action_id=3, team_id=AWAY, player_id=60.0, type_id=PASS,
                     start_x=FIELD_LENGTH - 70.0, start_y=FIELD_WIDTH - 14.0,
                     end_x=FIELD_LENGTH - 63.0, end_y=FIELD_WIDTH - 25.0))
    acts.append(dict(action_id=4, team_id=AWAY, player_id=63.0, type_id=SHOT,
                     start_x=FIELD_LENGTH - 55.0, start_y=FIELD_WIDTH - 33.0,
                     end_x=105.0, end_y=34.0))
    actions = pd.DataFrame([
        {**a, "game_id": 1, "period_id": 1, "result_id": 1, "time_seconds": 8.0,
         "bodypart_id": spadlconfig.bodypart_id["foot"]}
        for a in acts
    ])
    return actions, frames


def mirror_frames(frames: pd.DataFrame) -> pd.DataFrame:
    """Physical mirror: point-reflect positions, NEGATE velocities, swap direction labels.

    Velocities negate rather than reflect (ADR-045): a point reflection maps a vector to its
    negation. Omitting this was live defect D1 in ADR-045.
    """
    f = frames.copy()
    f["x"] = FIELD_LENGTH - f["x"]
    f["y"] = FIELD_WIDTH - f["y"]
    for vcol in ("vx", "vy"):
        if vcol in f.columns:
            f[vcol] = -f[vcol]
    f["team_attacking_direction"] = f["team_attacking_direction"].map({"ltr": "rtl", "rtl": "ltr"})
    return f


def away_mask(actions: pd.DataFrame, home_team_id) -> np.ndarray:
    from silly_kicks.id_compat import ids_match

    return (~ids_match(actions["team_id"], home_team_id)).to_numpy(dtype=bool)
```

- [ ] **Step 2: Write Gate A**

Append to `tests/tracking/test_mirror_registry.py`:

```python
from tests.tracking._mirror_registry import (
    AWAY,
    HOME,
    away_mask,
    canonical_scene,
    mirror_frames,
)


def _gate_a_params():
    for name in sorted(MIRROR_ENTRIES):
        entry = MIRROR_ENTRIES[name]
        marks = []
        if entry.known_defect:
            marks.append(pytest.mark.xfail(strict=True, reason=entry.known_defect))
        yield pytest.param(name, marks=marks)


@pytest.mark.parametrize("name", _gate_a_params())
def test_gate_a_mirror_invariance(name):
    """Same physical scene, mirrored frames -> identical action-LTR output.

    NEVER share a PitchControlCache between the two legs: it keys on frame IDENTITY and excludes
    player positions, so a mirrored frame carrying its twin's identity is served the base leg's
    surface and every pitch-control family passes at exactly zero difference (ADR-043).
    Each call below builds its own.
    """
    entry = MIRROR_ENTRIES[name]
    actions, frames = canonical_scene()
    a, f = actions.copy(), frames.copy()
    am, fm = actions.copy(), mirror_frames(frames)

    base = entry.call(a, f, HOME)
    mir = entry.call(am, fm, AWAY if entry.home_team_id_role != "unused" else HOME)

    # ALL rows, not just away-in-base. The mirrored leg passes home_team_id=AWAY, so the rows that
    # are "away" THERE are the home team's actions -- which carry the defect in the opposite leg.
    # Both halves are informative; restricting to away-in-base halves sensitivity for no saving.
    mask = np.ones(len(actions), dtype=bool)
    assert away_mask(actions, HOME).any(), "fixture has no away actions -- the gate would be vacuous"

    for col in entry.non_vacuity:
        away_vals = base[col].to_numpy()[away_mask(actions, HOME)]
        assert pd.notna(away_vals).any(), (
            f"{name}.{col}: all-null on AWAY rows -- comparison is vacuous exactly where the "
            "defect lives"
        )

    for col, cls in entry.columns.items():
        if cls == "exempt":
            continue
        b = pd.to_numeric(base[col], errors="coerce").to_numpy(float)[mask]
        m = pd.to_numeric(mir[col], errors="coerce").to_numpy(float)[mask]
        if cls == "mirrored_pitch_absolute":
            # Contract: the column is deliberately PITCH-ABSOLUTE, so it must equal its own
            # REFLECTION -- compare base against the reflected mirror value, not against the raw
            # mirror value. Enforced here rather than deferred, so the class cannot become an
            # unjustified exemption.
            spec = entry.reflections[col]
            if isinstance(spec, dict):
                # CATEGORICAL -- compare labels DIRECTLY, never through to_numeric. Coercing a
                # label yields NaN, `both.any()` is then False, and the `continue` below would
                # SILENTLY PASS: an assertion that cannot fire. Same shape as the vacuity traps
                # this whole gate exists to avoid.
                reflected = mir[col].map(spec).to_numpy()[mask]
                actual = base[col].to_numpy()[mask]
                comparable = pd.notna(actual) & pd.notna(reflected)
                assert comparable.any(), (
                    f"{name}.{col}: no comparable labels -- the reflection check is vacuous"
                )
                assert (actual[comparable] == reflected[comparable]).all(), (
                    f"{name}.{col}: pitch-absolute label does not equal its own reflection"
                )
                continue
            m = (FIELD_LENGTH - m) if spec == "x" else (FIELD_WIDTH - m)
        both = np.isfinite(b) & np.isfinite(m)
        if not both.any():
            continue
        delta = np.abs(b[both] - m[both]).max()
        assert delta <= entry.tolerance, (
            f"{name}.{col}: base-vs-mirror {delta:.6g} > tol {entry.tolerance} "
            f"({entry.tolerance_basis})"
        )
```

- [ ] **Step 3: Add the `mirrored_pitch_absolute` witness — the class has NO real member**

Measured at HEAD: `add_shape_graph` emits **6 numeric columns** (`shape_graph_{density,n_edges,
mean_stability}_{attacking,defending}`). The pitch-absolute lateral label lives in `infer_positions`,
which `_shape_graph.py:877-880` records as having *"no in-library consumer"* — it is not surfaced by
any `add_*`, so it cannot be a registry column.

So without this witness the whole `mirrored_pitch_absolute` mechanism — `reflections`, the Gate A
branch, the declaration test — ships **correct but never exercised**, which is precisely the failure
this plan's detection-first design exists to prevent. Same idiom as Gate B's planted witness.

```python
def test_gate_a_enforces_the_mirrored_pitch_absolute_contract():
    """Witness: the class has no real member today, so only a plant can exercise it.

    Asserts BOTH directions -- that a correctly pitch-absolute column passes, and that omitting
    the reflection FAILS. Without the second half this is a test that cannot fail.
    """
    from tests.tracking._mirror_registry import FIELD_LENGTH, MirrorEntry

    actions, frames = canonical_scene()

    def numeric_abs(a, f, _home):
        """A genuinely pitch-absolute x: the same physical spot in BOTH legs, so under a mirror
        the emitted value becomes 105 - v."""
        out = a.copy()
        ball = f[f["is_ball"].astype(bool)].iloc[0]
        out["abs_x"] = float(ball["x"])
        return out

    base = numeric_abs(actions.copy(), frames.copy(), HOME)
    mir = numeric_abs(actions.copy(), mirror_frames(frames), AWAY)

    # Correct: base == reflection of mirror.
    assert np.allclose(base["abs_x"], FIELD_LENGTH - mir["abs_x"], atol=1e-9)
    # Discriminating: the NAIVE comparison (base vs raw mirror) must FAIL, or the reflection
    # branch is doing nothing.
    assert not np.allclose(base["abs_x"], mir["abs_x"], atol=1e-9), (
        "the plant is not discriminating -- pick a ball position off the halfway line"
    )


def test_gate_a_categorical_reflection_cannot_silently_pass():
    """The label branch must NOT route through to_numeric.

    Coercing a categorical label yields NaN, which makes `both.any()` False and lets the numeric
    path `continue` -- a silent pass. This pins the label comparison as a real assertion.
    """
    swap = {"left": "right", "right": "left"}
    base_labels = pd.Series(["left", "right", "left", "right"])
    good_mirror = pd.Series(["right", "left", "right", "left"])
    bad_mirror = pd.Series(["left", "right", "left", "right"])

    assert (base_labels == good_mirror.map(swap)).all()
    assert not (base_labels == bad_mirror.map(swap)).all(), (
        "a wrong label mapping must be detectable, or the branch asserts nothing"
    )
```

- [ ] **Step 4: Run — expect a clean collection with zero registry entries**

Run: `python -m pytest tests/tracking/test_mirror_registry.py -v`
Expected: Gate A collects 0 parameterized cases (registry still empty); both witnesses PASS; the
`test_every_public_add_is_registered` meta-assertion still fails. Correct.

- [ ] **Step 5: Commit**

```bash
git add tests/tracking/_mirror_registry.py tests/tracking/test_mirror_registry.py
git commit -m "test(tracking): Gate A mirror-invariance harness + pitch-absolute witness (ADR-028 §6)"
```

---

## Task 9: Gate B — `home_team_id` invariance

**Files:**
- Modify: `tests/tracking/test_mirror_registry.py`

- [ ] **Step 1: Write Gate B**

```python
NONSENSE_HOME_ID = 999_999


@pytest.mark.parametrize("name", sorted(MIRROR_ENTRIES))
def test_gate_b_home_team_id_invariance(name):
    """D1: direction must come from the frames, never from team identity.

    Gate A is structurally blind to identity-keying -- swapping home_team_id restores the exact
    invariant identity-keying assumes, so an identity-keyed aggregator is invariant there whether
    it is safe or not. This gate holds the frames FIXED and varies home_team_id instead, so it
    never runs an aggregator outside the convert_to_frames contract.

    The nonsense id is what makes this strictly stronger than a two-team swap: it catches
    `same_id(x, home) else ...` branches that a swap can leave looking correct.
    """
    entry = MIRROR_ENTRIES[name]
    if entry.home_team_id_role == "unused":
        pytest.skip(f"{name} does not take home_team_id")

    actions, frames = canonical_scene()
    ref = entry.call(actions.copy(), frames.copy(), HOME)
    variants = {
        "away": entry.call(actions.copy(), frames.copy(), AWAY),
        "nonsense": entry.call(actions.copy(), frames.copy(), NONSENSE_HOME_ID),
    }

    checked = 0
    for col, cls in entry.columns.items():
        if cls != "invariant":
            continue  # only pure action-LTR geometry is required to be identity-independent
        r = pd.to_numeric(ref[col], errors="coerce").to_numpy(float)
        for label, out in variants.items():
            v = pd.to_numeric(out[col], errors="coerce").to_numpy(float)
            both = np.isfinite(r) & np.isfinite(v)
            if not both.any():
                continue
            checked += 1
            delta = np.abs(r[both] - v[both]).max()
            assert delta <= entry.tolerance, (
                f"{name}.{col}: moved {delta:.6g} when home_team_id -> {label}. "
                "A mirror-invariant column is action-LTR geometry and cannot depend on which "
                "team is home -- this is identity-keyed direction inference (D1)."
            )
    assert checked > 0, f"{name}: Gate B compared nothing -- the check is vacuous"
```

- [ ] **Step 2: Prove Gate B can actually fail (non-vacuity witness)**

Add this alongside it. It plants a known identity-keyed aggregator and asserts the gate catches it:

```python
def test_gate_b_catches_a_planted_identity_keyed_aggregator():
    """Witness: without this, a green Gate B is indistinguishable from a gate that checks nothing."""
    from silly_kicks.id_compat import same_id
    from tests.tracking._mirror_registry import MirrorEntry

    def planted(actions, frames, home_team_id):
        # Deliberately keys direction on IDENTITY -- the D1 defect, in miniature.
        out = actions.copy()
        out["planted_x"] = [
            row["start_x"] if same_id(row["team_id"], home_team_id) else 105.0 - row["start_x"]
            for _, row in actions.iterrows()
        ]
        return out

    entry = MirrorEntry(
        name="planted", call=planted, columns={"planted_x": "invariant"},
        tolerance=1e-9, tolerance_basis="exact", home_team_id_role="direction_only",
        non_vacuity=("planted_x",),
    )
    actions, frames = canonical_scene()
    ref = entry.call(actions.copy(), frames.copy(), HOME)
    alt = entry.call(actions.copy(), frames.copy(), AWAY)
    delta = (ref["planted_x"] - alt["planted_x"]).abs().max()
    assert delta > 1.0, "the plant did not move -- this witness is not discriminating"
```

- [ ] **Step 3: Run**

Run: `python -m pytest tests/tracking/test_mirror_registry.py -v`
Expected: the witness PASSES (proving the gate's mechanism discriminates); Gate B collects 0 real
cases so far.

- [ ] **Step 4: Commit**

```bash
git add tests/tracking/test_mirror_registry.py
git commit -m "test(tracking): Gate B home_team_id invariance + non-vacuity witness (ADR-028 D1)"
```

---

## Task 10: Populate the registry — tier 1 (17 entries, no extra inputs)

**Files:**
- Modify: `tests/tracking/_mirror_registry.py`

These 17 take only `(actions, frames[, home_team_id])`:

`add_action_context`, `add_actor_pre_window`, `add_defensive_line`, `add_elastic_sync`,
`add_line_break`, `add_off_ball_context`, `add_off_ball_runs`, `add_packing`, `add_pitch_control`,
`add_pre_shot_gk_angle`, `add_pre_shot_gk_position`, `add_press_commitment`, `add_pressure_on_actor`,
`add_shape_graph`, `add_shot_goalmouth`, `add_structural_pass`, `add_team_shape`

**Signature outliers to respect** (verify each with `inspect.signature` before writing its entry):
`add_pre_shot_gk_angle` / `add_pre_shot_gk_position` take `frames` **keyword-only**;
`add_action_context`, `add_elastic_sync`, `add_pitch_control` take **no** `home_team_id`
(`home_team_id_role="unused"`); `add_off_ball_runs` has **no** `links` kwarg.

- [ ] **Step 1: Add one worked entry, then follow the pattern**

```python
def _entry(name, call, columns, *, tol, basis, role, non_vacuity, exempt=None, defect=None):
    MIRROR_ENTRIES[name] = MirrorEntry(
        name=name, call=call, columns=columns, tolerance=tol, tolerance_basis=basis,
        home_team_id_role=role, non_vacuity=tuple(non_vacuity),
        exempt_reasons=exempt or {}, known_defect=defect,
    )


def _register_tier1() -> None:
    from silly_kicks.tracking.features import add_defensive_line, add_pitch_control

    _entry(
        "add_defensive_line",
        lambda a, f, h: add_defensive_line(a, f, home_team_id=h),
        {
            "defensive_line_x": "invariant",
            "defensive_line_compactness_x": "invariant",
            "max_lateral_gap": "invariant",
            "lateral_width": "invariant",
            "back_n_count": "invariant",
            "defensive_line_spread_y": "invariant",
        },
        tol=1e-9, basis="pure geometry; exact under a point reflection",
        role="direction_only",
        non_vacuity=("defensive_line_x",),
    )
    _entry(
        "add_pitch_control",
        lambda a, f, _h: add_pitch_control(a, f),
        {"pitch_control_at_target__spearman": "invariant"},
        tol=2e-2,
        basis=(
            "pitch control is NOT exactly mirror-symmetric: its degenerate/no-information "
            "fallback returns exactly 0.5 at one point and 1.0 at its mirror. Measured 1.3e-2 on "
            "the OBSO fixture (_OBSO_MIRROR_TOL comment); 2e-2 leaves headroom. On the DENSE "
            "canonical scene the fallback should not fire -- if this tolerance is ever needed, "
            "the fixture has gone sparse."
        ),
        role="unused",
        non_vacuity=("pitch_control_at_target__spearman",),
    )
    # ... one _entry(...) call per remaining tier-1 aggregator, same shape.


_register_tier1()
```

- [ ] **Step 2: Add the remaining 15 tier-1 entries**

For each: read its `def` to get the exact signature, run it once on `canonical_scene()` to list the
columns it ADDS (`set(out.columns) - set(actions.columns)`), and classify each column. Default to
`invariant`; use `exempt` with a written reason for any non-numeric or provenance column
(`frame_id`, `time_offset_seconds`, `n_candidate_frames`, `link_quality_score`, `*_source`).

**`add_shape_graph` has NO pitch-absolute column — do not go looking for one.** It emits exactly six
NUMERIC columns (`shape_graph_{density,n_edges,mean_stability}_{attacking,defending}`); all six are
`invariant`. The pitch-absolute lateral label that ADR-045 D5 settled lives in `infer_positions`
(`_shape_graph.py:877-880`), which that code records as having *"no in-library consumer"* — it is not
surfaced by any `add_*`, so it cannot be a registry column.

This corrects an earlier draft of this plan, which instructed classing a shape-graph lateral label
`mirrored_pitch_absolute`. **That instruction was unfollowable** — an engineer would have had to
invent a column or mis-class a numeric one. The `mirrored_pitch_absolute` class therefore has **zero
real members in v1**; it is kept because the vocabulary needs it the moment a genuine member appears,
and it is exercised by the two planted witnesses in Task 8 Step 3 rather than by a live entry.

- [ ] **Step 3: Run both gates over tier 1**

Run: `python -m pytest tests/tracking/test_mirror_registry.py -v`
Expected: `test_every_public_add_is_registered` still fails (16 to go). Gate A and Gate B now run 17
cases each.

**All SEVEN D3 re-key targets, with the expected Gate B verdict for each.** The spec (§4.3) names
seven; five are tier-1 and land here, two are tier-2 and land in Task 11. Writing only the tier-1
five without saying so is how a coverage hole hides:

| Aggregator | Tier | Expected Gate B | Expected Gate A |
|---|---|---|---|
| `add_defensive_line` | 1 (here) | **FAIL** — identity-keyed goal end (`_defensive_line.py:210`) | pass |
| `add_line_break` | 1 (here) | **FAIL** — identity-keyed in both branches | pass |
| `add_packing` | 1 (here) | **FAIL** — x-only flip keyed on identity (`_packing.py:145-147`) | pass |
| `add_off_ball_context` | 1 (here) | **FAIL** — line-break half is identity-keyed | pass |
| `add_structural_pass` | 1 (here) | **FAIL** — identity-keyed (`_structural_pass.py:145-148`) | pass |
| `add_player_influence` | 2 (Task 11) | **FAIL** — identity-keyed xT grid reflection | pass |
| `add_cover_shadows` | 2 (Task 11) | **FAIL** — `_cover_shadows.py:1030` identity key | xfail (RC1) |

Gate A passing while Gate B fails is the two-gate design proving itself: Gate A is structurally blind
to identity-keying (spec §2.1), which is the entire reason Gate B exists.

> **STOP if an aggregator NOT in this table fails Gate B.** That is a FINDING — an identity-keyed
> site the sweep missed — not a fixture problem. The temptation at that moment is to adjust
> `canonical_scene()` until it passes; do not. Record it, and treat it as an eighth D3 target.
> Equally, if one of the seven PASSES Gate B, do not shrug it off: an identity-keyed aggregator
> invisible to the gate is exactly the coverage hole Gate B was added to close, and the fixture may
> need extending to express it.

- [ ] **Step 4: Mark the Gate B failures**

Add `known_defect="D3 re-key pending: identity-keyed direction (spec §4.3)"` to each entry that fails
Gate B, and extend `_gate_a_params`'s xfail logic to a matching `_gate_b_params`. Re-run; expected:
all xfail (not xpass).

- [ ] **Step 5: Commit**

```bash
git add tests/tracking/_mirror_registry.py tests/tracking/test_mirror_registry.py
git commit -m "test(tracking): register tier-1 aggregators in the mirror registry (ADR-028)"
```

---

## Task 11: Populate the registry — the remaining 16 entries

**Files:**
- Modify: `tests/tracking/_mirror_registry.py`

Grouped by what each needs beyond `(actions, frames, home_team_id)`:

| Group | Aggregators | Extra input |
|---|---|---|
| Fitted xT | `add_obso`, `add_pausa`, `add_space_creation`, `add_cover_shadows`, `add_gk_influence`, `add_player_influence`, `add_off_ball_run_values`, `add_xt_gk` | module-level `ExpectedThreat` built once |
| Trained model | `add_ghost_gk`, `add_xshot_occurrence`, `add_xcross_attempt`, `add_gk_completion` | `from_variant("default")`, loaded once at module scope |
| Optional dep | `add_das` | `[das]` extra (installed on every CI leg) |
| Other | `add_defensive_credit`, `add_gradientsports_player_ids`, `add_sync_score` | see notes below |

- [ ] **Step 1: Add the shared fitted-xT helper**

```python
@functools.cache
def gate_xt():
    """A NON-degenerate xT for the gate. Deliberately y-ASYMMETRIC.

    A y-symmetric grid cannot distinguish a correct point reflection from an x-only mirror --
    exactly the blind spot that let ADR-041's incomplete repair through. Do NOT 'simplify' this
    to a pure x-ramp.
    """
    import numpy as np

    from silly_kicks.xthreat import ExpectedThreat

    xt = ExpectedThreat(l=16, w=12)
    x_ramp = np.linspace(0.02, 0.9, 16)[None, :]
    y_tilt = np.linspace(0.6, 1.4, 12)[:, None]
    xt.xT = x_ramp * y_tilt
    return xt
```

- [ ] **Step 2: Register the fitted-xT group**

Worked example (repeat the shape for the other seven, adjusting columns):

```python
    from silly_kicks.tracking.features import add_cover_shadows

    _entry(
        "add_cover_shadows",
        lambda a, f, h: add_cover_shadows(a, f, gate_xt(), home_team_id=h),
        {
            "n_blocked_receivers": "invariant",
            "n_potential_receivers": "invariant",
            "blocking_score": "invariant",
            "blocked_threat_fraction": "invariant",
            "max_single_defender_blocking_score": "invariant",
            "max_single_defender_player_id": "exempt",
        },
        tol=1e-6, basis="threat integrals; measured 4.4e-16 on real GS/IDSSE away rows",
        role="direction_only",
        non_vacuity=("n_blocked_receivers", "blocking_score"),
        exempt={"max_single_defender_player_id": "player id, not a numeric quantity; None on the cheap path by design"},
        defect="RC1: raw action-LTR passer at features.py:3698 / :3861 (spec §3.1)",
    )
```

**`add_xt_gk` and `add_gk_completion` carry `defect="RC2: ..."`; `add_space_creation` carries
`defect="RC3: ..."`.** Those three plus `add_cover_shadows` are the four xfail markers PRs 2-4 delete.

- [ ] **Step 3: Register the trained-model group**

Load each model ONCE at module scope (`@functools.cache`) — `add_ghost_gk` otherwise pays the load
twice per case. If a model load raises, the entry must `pytest.skip` with the reason rather than
silently pass.

- [ ] **Step 4: Register the `other` group**

- `add_sync_score` — takes `links`, not `frames`; emits sync-quality aggregations only. All columns
  `exempt` with reason "no geometry emitted; instrument does not apply", `home_team_id_role="unused"`.
- `add_gradientsports_player_ids` — roster string join, zero coordinate reads. Same treatment.
- `add_defensive_credit` — full geometry; `role="direction_only"`; its aggregate columns are
  `invariant`.

- [ ] **Step 5: Run the whole registry**

Run: `python -m pytest tests/tracking/test_mirror_registry.py -v`
Expected: `test_every_public_add_is_registered` PASSES (33/33). Gate A shows the four RC xfails.
Gate B shows the D3 xfails. **Zero XPASS** — an xpass means the defect is not what you think.

- [ ] **Step 6: Measure the budget**

```bash
python -m pytest tests/tracking/test_mirror_registry.py -q --benchmark-skip --durations=10
```

Record total wall-clock in the commit message. The existing 5-aggregator gate is 7 tests in 1.63s.
ADR-023 forbids marking a behavioural guard `slow`, so if this exceeds ~60s, raise it rather than
adding a marker.

- [ ] **Step 7: Commit**

```bash
git add tests/tracking/_mirror_registry.py
git commit -m "test(tracking): complete the 33-aggregator mirror registry (ADR-028 §6)"
```

---

## Task 12: `_defensive_line.py:73` D3 unit + CLAUDE.md correction

**Files:**
- Modify: `tests/tracking/test_mirror_registry.py`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add the D3-unit enumeration test**

```python
def test_defensive_line_d3_unit_is_enumerated():
    """`select_back_line_players` is a PUBLIC export with three consumers. A partial re-key is
    the incomplete-repair pattern this repo has already shipped, so the gate names the whole unit.
    """
    import ast
    import pathlib

    # __file__-anchored, matching the repo idiom -- a CWD-relative path silently reads nothing
    # when pytest is invoked from anywhere but the repo root, and an empty read makes the
    # assertions below vacuous rather than red.
    repo = pathlib.Path(__file__).resolve().parents[2]
    unit = {
        "silly_kicks/tracking/_defensive_line.py",
        "silly_kicks/tracking/_packing.py",
        "silly_kicks/tracking/_gk_influence.py",
    }
    reads = set()
    for rel in sorted(unit):
        path = repo / rel
        assert path.exists(), f"D3 unit member missing: {rel}"
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "home_team_id":
                reads.add(rel)
    assert reads, "no member of the D3 unit reads home_team_id -- has the unit moved?"
    # Pin the CURRENT state so a partial re-key changes this set and fails loudly.
    assert reads == unit, (
        f"D3 unit membership changed: {sorted(reads)}. Re-key all three together or update "
        "this pin with the reason."
    )
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/tracking/test_mirror_registry.py -k d3_unit -v`
Expected: PASS (documents the current state). If it fails, one of the three files has already been
partially re-keyed — investigate before proceeding.

- [ ] **Step 3: Correct CLAUDE.md**

Find the ADR-028 bullet's sentence listing six aggregators as "now reprojected AT THEIR OWN QUERY
SEAM". Replace with:

```
They are now reprojected AT THEIR OWN QUERY SEAM -- with TWO exceptions found 2026-07-29 and
recorded in `docs/superpowers/specs/2026-07-29-adr028-orientation-defect-class-design.md`:
`space_creation` was NEVER reprojected (its `home_team_id` is a dead parameter; ADR-041 gave it
only an unconditional `axis=(0,1)` opponent mirror), and `cover_shadows` is only HALF reprojected
(the RECEIVER at `_cover_shadows.py:1164`, never the PASSER at `features.py:3698`/`:3861`).
```

- [ ] **Step 4: Commit**

```bash
git add tests/tracking/test_mirror_registry.py CLAUDE.md
git commit -m "docs: correct the ADR-028 repair claim for space_creation + cover_shadows"
```

- [ ] **Step 5: Open PR 1**

```bash
python -m pytest tests/ -m "not e2e" -q --benchmark-skip
ruff check silly_kicks tests scripts && ruff format --check silly_kicks tests scripts && pyright
git push -u origin pr1-detection
gh pr create --title "test(tracking): ADR-028 detection -- fixtures, fail-loud seam, 33-aggregator mirror registry" --body "See docs/superpowers/specs/2026-07-29-adr028-orientation-defect-class-design.md PR 1. Test-only + one warning category; no shipped value changes."
```

---

# PR 2 — RC1 cover shadows

## Task 13: Reproject the passer at both seams

**Files:**
- Modify: `silly_kicks/tracking/features.py:3698` and `:3861-3864`
- Test: `tests/tracking/test_mirror_registry.py` (delete one xfail)

- [ ] **Step 1: Confirm the gate is red without the fix**

```bash
git checkout -b pr2-cover-shadows
python -m pytest tests/tracking/test_mirror_registry.py -k "gate_a and cover_shadows" -v
```

Expected: XFAIL. That is the pre-fix state the gate recorded in PR 1.

- [ ] **Step 2: Fix the aggregator seam**

In `add_cover_shadows`, before the per-action loop:

```python
    # ADR-028: `passer_xy` is an ACTION-LTR coordinate; every position it is compared against
    # below (defenders, receivers, the ball) is FRAME-LTR. Reproject the passer INTO frame coords
    # -- not the frame into action-LTR -- because the one place that steps out to action-LTR
    # (_cover_shadows.py:1164's xT lookup) already reprojects itself.
    _flip = acting_team_attacks_rtl(actions, frames).to_numpy(dtype=bool)
```

Then replace `:3698`:

```python
        passer_xy = (float(row["start_x"]), float(row["start_y"]))
        if _flip[j]:
            passer_xy = (FIELD_LENGTH - passer_xy[0], FIELD_WIDTH - passer_xy[1])
```

`acting_team_attacks_rtl`, `FIELD_LENGTH` and `FIELD_WIDTH` are already imported at
`features.py:57-61`; confirm before adding an import.

- [ ] **Step 3: Fix the xfns seam**

In `_cover_shadow_transformer`, compute the flip per slot before the row loop, then replace
`:3861-3864`:

```python
            slot_flip = acting_team_attacks_rtl(slot, frames).to_numpy(dtype=bool)
            # ... inside the row loop:
                passer_xy = (float(row["start_x"]), float(row["start_y"]))
                if slot_flip[j]:
                    passer_xy = (FIELD_LENGTH - passer_xy[0], FIELD_WIDTH - passer_xy[1])
```

**The cache key must be built from the REPROJECTED passer** — `_get_cs` rounds `passer_xy` into its
key, so reprojecting after the key is built would serve a home-oriented surface to an away action.
Verify the `_get_cs(pid, fid, tid, passer_xy)` call receives the reprojected tuple.

- [ ] **Step 4: Delete the xfail marker**

Remove `defect="RC1: ..."` from the `add_cover_shadows` entry. PR 1 split the entries out of
`_mirror_registry.py` into the `tests/tracking/_mirror_entries/` package, so the edit lands in
`tests/tracking/_mirror_entries/influence_family.py`. Leave `defect_b=` in place — that is D3.

- [ ] **Step 5: Run**

Run: `python -m pytest tests/tracking/test_mirror_registry.py -k cover_shadows -v`
Expected: PASS (not XPASS — the marker is gone).

Run: `python -m pytest tests/ -m "not e2e" -q --benchmark-skip`
Expected: PASS. Cover-shadow value tests that pinned pre-fix numbers will fail — each is a real
value change on away rows; update the expected values and note the delta in the commit.

- [ ] **Step 6: Commit + PR**

```bash
git add silly_kicks/tracking/features.py tests/tracking/_mirror_registry.py
git commit -m "fix(tracking): reproject the cover-shadow passer into frame coords (ADR-028 RC1)"
git push -u origin pr2-cover-shadows
```

PR body must state: **re-materialize trigger** — `n_blocked_receivers` and
`max_single_defender_blocking_score` change on ~40-45% of all actions (measured 77.8-85% of away
rows). No forced VAEP retrain: `cover_shadow_xfns` is a factory in no default xfn list.

---

# PR 3 — RC2 + RC3 value corrections

> **RELEASE CONSTRAINT: PR 3 MUST NOT SHIP IN A RELEASE WITHOUT PR 4.** Today
> `GkCompletionModel` is train/serve *consistent* (fabricated away origins on both sides). PR 3
> corrects serving geometry while the bundled weights are still the old ones, which INTRODUCES a
> skew that does not exist today. Do not tag between these two PRs.

## Version numbering for PR 3 + PR 4 (DECIDED 2026-07-30 — see the 2026-08-01 correction below)

> **⚠ CORRECTED 2026-08-01 — PR 4 is 4.73.0, and "Tag ONLY `v4.72.0`" is now WRONG.**
> A concurrent session took **4.72.0** for ADR-052 and shipped it **without** the retrain, so the
> numbers below are off by one release and the tagging instruction, followed literally, would publish
> the very skew this section exists to prevent. Read every "4.72.0" below as **4.73.0**.
>
> The invariant was never the number. It is: **no tag between 4.71.0 and the retrain.** That has
> held — `v4.71.0` and `v4.72.0` were both left untagged — so 4.73.0 is the first tag since
> `v4.70.0` and is what publishes the pairing. A version number in a plan is a prediction; the
> *ordering constraint* is the decision, which is why the heading no longer says
> "do not re-litigate" about a figure that a parallel branch can invalidate.

**PR 3 bumps to 4.71.0. PR 4 bumps to 4.73.0. Tag ONLY `v4.73.0`.** 4.71.0 and 4.72.0 are real,
committed, traceable versions that are deliberately **never published to PyPI**.

PR 3's CHANGELOG heading must carry the marker inline, because that is the exact line someone reads
immediately before tagging:

```markdown
## [4.71.0] — NOT RELEASED (ships within 4.73.0 alongside PR 4)
```

PR 4 then adds its own `## [4.73.0] — YYYY-MM-DD` section above it and leaves PR 3's heading intact
(the marker is the historical record of why a version was skipped on PyPI). Use an **em dash** in the
heading, matching every existing entry.

**Merging is not releasing.** The publish workflow triggers on `tags: ["v*"]` ONLY
(`.github/workflows/publish.yml:3-5`) and builds from `pyproject.toml`; merging PR 3 publishes
nothing. So both PRs merge normally as ordinary squash commits — **no merge-setting override is
needed, and none should be made.** *(STALE as of 2026-08-01: the repo now allows merge commits —
`allow_merge_commit: true`. That does not change PR 3's reasoning, which was about tag/release
coupling and not commit shape; but PR 4 DOES need `--merge`, see its exit criteria and spec §11.8.)*
The repo was squash-only at the time (`allow_merge_commit: false`,
`allow_rebase_merge: false`); commit shape was never the constraint.

**Why not `## [Unreleased]` with no bump in PR 3.** It was considered and REJECTED. It is technically
safe — an audit confirmed nothing in CI, packaging, tests or scripts reads CHANGELOG.md, the version
is static in `pyproject.toml`, and `[Unreleased]` is canonical Keep a Changelog — but it violates the
owner's standing rule that **every PR bumps the version**, which exists as a per-PR traceability
signal independent of consumer impact. Bumping both and tagging once satisfies that rule *and* the
release constraint.

**Do not repeat the causal error found during that review.** The only prior `## [Unreleased]` on main
(PR-S51 → 3.18.0, `6afc398`) is recorded as a release incident, but `[Unreleased]` did **not** cause
it: the first incident (2.9.0, `fc62ebe`) contained zero occurrences of "Unreleased" and failed
identically. The invariant across both is *(un-bumped `pyproject.toml`) AND (tag push)*. That is the
thing to avoid.

**Residual risk is human and unguarded.** Nothing in this repo validates that a pushed tag matches
`pyproject.toml`'s version, so an erroneous `v4.71.0` push during the PR 3 → PR 4 window would build
4.70.0-or-4.71.0 artifacts and reproduce the incident. Keep the window short; do not tag from main
between these two merges for any reason.

## Task 14: RC2 — reproject in `_gk_geometry`

**Files:**
- Modify: `silly_kicks/tracking/_gk_geometry.py:130-152` and `:240-262`

- [ ] **Step 1: Confirm both gates are red**

```bash
git checkout -b pr3-value-corrections
python -m pytest tests/tracking/test_mirror_registry.py -k "xt_gk or gk_completion" -v
```
Expected: XFAIL.

- [ ] **Step 2: Fix `_tracking_gk_xy`**

Mirror the pattern its own sibling already uses at `:220-221`:

```python
    from ._action_orientation import FIELD_LENGTH, FIELD_WIDTH, acting_team_attacks_rtl

    ...
    flip = acting_team_attacks_rtl(actions, frames).to_numpy(dtype=bool)
    ...
        gx, gy = float(gk.iloc[0]["x"]), float(gk.iloc[0]["y"])
        if flip[i]:  # ADR-028: frame home-LTR -> action-LTR for away-team actions
            gx, gy = FIELD_LENGTH - gx, FIELD_WIDTH - gy
        if gx <= _GOAL_AREA_DEPTH:  # clamp AFTER the reprojection, as _tracking_gk_xy_detected does
            res[i] = (gx, gy)
```

The clamp must stay **after** the flip. Before the fix, a mis-oriented away keeper landed at high x
and fell through the clamp to the rule point — which is why RC2's error was a systematic loss of the
tracking tier rather than a wild coordinate.

- [ ] **Step 3: Fix `_tracking_ball_xy` the same way**

```python
        bx, by = float(fr.iloc[0]["x"]), float(fr.iloc[0]["y"])
        if flip[i]:
            bx, by = FIELD_LENGTH - bx, FIELD_WIDTH - by
        res[i] = (bx, by)
```

- [ ] **Step 4: Delete the RC2 xfail markers; run**

Run: `python -m pytest tests/tracking/test_mirror_registry.py -k "xt_gk or gk_completion" -v`
Expected: PASS.

Run: `python -m pytest tests/ -m "not e2e" -q --benchmark-skip`
Expected: PASS after updating restart-coordinate and xT-GK expected values.

- [ ] **Step 5: Commit**

```bash
git add silly_kicks/tracking/_gk_geometry.py tests/tracking/_mirror_registry.py
git commit -m "fix(tracking): reproject GK + ball frame coords into action-LTR (ADR-028 RC2)"
```

## Task 15: RC3 — reflect the EPV grid

**Files:**
- Modify: `silly_kicks/tracking/_space_creation.py:216`, `silly_kicks/tracking/features.py:5557`

- [ ] **Step 1: Add the parameter to `_compute_space_creation_for_action`**

The threading IS the correctness content of this task: D1 forbids deriving direction from the
`home_team_id` this function already receives, so the flip must arrive as a computed input.
`home_team_id` stays in the signature (D3 retires it by disuse, it is not removed) and stays unread.

`silly_kicks/tracking/features.py:5557` — add one keyword-only parameter:

```python
def _compute_space_creation_for_action(
    action_row: pd.Series,
    frame: pd.DataFrame,
    *,
    home_team_id: int | str,
    attacks_rtl: bool = False,
    transition_grid: np.ndarray | None = None,
    epv_grid: np.ndarray | None = None,
    pitch_control_method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
    pitch_control_cache: PitchControlCache | None = None,
) -> dict[str, float]:
```

- [ ] **Step 2: Reflect the multiplier inside it**

`silly_kicks/tracking/_space_creation.py:216` — the multiplier is built in attack-LTR and applied to
frame-LTR pitch control:

```python
    obso_multiplier = effective_transition * epv_interp  # (ny, nx) -- attack-LTR
    if attacks_rtl:
        # ADR-028: BOTH axes. An x-only mirror is exact only for a y-SYMMETRIC grid -- and the
        # synthetic ramp IS y-symmetric, which is precisely how ADR-041's first repair passed its
        # own tests while being wrong on a fitted xT.
        obso_multiplier = obso_multiplier[::-1, ::-1]
```

Thread `attacks_rtl` from `features.py` into whichever `_space_creation` helper builds the
multiplier; keep the parameter name identical at every hop so the grep is trivial.

- [ ] **Step 3: Compute the flip once and pass it at the call site**

`silly_kicks/tracking/features.py` — before the `for i, (_idx, action_row) in enumerate(...)` loop
(just after `frame_groups = frames.groupby([...])` at `:5709`):

```python
    # ADR-028 per-action re-projection flag. Computed ONCE for the whole call, from the FRAMES --
    # never from home_team_id (D1).
    _flip = acting_team_attacks_rtl(actions, frames).to_numpy(dtype=bool)
```

Then at the call site (`:5730`):

```python
        result = _compute_space_creation_for_action(
            action_row,
            frame,
            home_team_id=home_team_id,
            attacks_rtl=bool(_flip[i]),
            transition_grid=transition_grid,
            epv_grid=epv_grid,
            pitch_control_method=pitch_control_method,
            pitch_control_cache=cache,
        )
```

`i` is already the loop's positional index, so it aligns with `_flip` without an index lookup.

- [ ] **Step 4: Delete the RC3 xfail; run**

Run: `python -m pytest tests/tracking/test_mirror_registry.py -k space_creation -v`
Expected: PASS.

- [ ] **Step 5: Commit + PR**

```bash
git add silly_kicks/tracking/_space_creation.py silly_kicks/tracking/features.py tests/tracking/_mirror_registry.py
git commit -m "fix(tracking): reflect the space-creation EPV multiplier per acting team (ADR-028 RC3)"
git push -u origin pr3-value-corrections
```

PR body: **re-materialize trigger** for `xt_gk*`, `gk_completion`, `enriched_*` restart coords and
both space-creation columns. **Must not be released without PR 4.**

---

## Task 15b: Two carried doc fixes (owner-assigned 2026-07-30)

Both surfaced by the 4.70.0 `/final-review` audit, both **pre-existing on main** and unrelated to any
RC. Assigned here because they are cheap, local, and need no DGX — deliberately NOT PR 5, which is an
atomic code+retrain+re-stamp unit that unrelated doc churn would make harder to review.

**Files:**
- Modify: `SECURITY.md:5-7`
- Modify: `TODO.md` (the `TF-19` On-Deck row)

- [ ] **Step 1: SECURITY.md supported-versions table**

It still advertises `3.x`, stale since 4.0.0 (2026-05-30) and last touched in the 3.x-era PR-S29.
Replace the table body:

```markdown
| Version | Supported          |
|---------|--------------------|
| 4.x     | :white_check_mark: |
| < 4.0   | :x:                |
```

- [ ] **Step 2: De-duplicate the TF-19 On-Deck row**

The row's Notes cell contains a **424**-character span duplicated verbatim. The seam is a **single
unescaped `|`**, which adds one extra table cell (the other **two** pipes in that cell are
`\|`-escaped and render literally — do not "fix" those). The row currently ends:

```
... **Depends on: TF-15, TF-16, TF-17, TF-18.** | actual_GK) − P(action \| ghost_GK)` for action ∈ {shot, cross, key_pass}, weighted by realized-or-expected outcome value and summed across the build-up window (negative ⇒ deterrent); the counterfactual MUST substitute `predict_mean()` (the deterministic boosted HGBR in `ghost_gk_x/y`), **not** the `predict_density` KDE mode, so no train/serve backend pin is needed (ADR-016). **Depends on: TF-15, TF-16, TF-17, TF-18.** |
```

Delete the duplicate so it ends at the FIRST occurrence:

```
... **Depends on: TF-15, TF-16, TF-17, TF-18.** |
```

- [ ] **Step 3: Verify the table parses as 5 columns**

```bash
python -c "
line=[l for l in open('TODO.md',encoding='utf-8').read().split('\n') if l.startswith('| TF-19')][0]
import re
# count only UNESCAPED pipes -- \| is a literal, not a separator
print('unescaped pipes:', len(re.findall(r'(?<!\\\\)\|', line)))
"
```

Expected: `6` — matching every other row (5 columns). Before the fix it is 7.

---

# PR 4 — RC4 loader + all weights work

## Task 16: Orient SkillCorner frames in the loader

**Files:**
- Modify: `scripts/_loader_pining.py:477-479`
- Test: `tests/scripts/test_loader_orientation.py` (create)

- [ ] **Step 1: Write the failing test**

```python
"""RC4: the pining loader must not hand out unoriented frames (spec §3.4)."""

import pytest

pytestmark = pytest.mark.e2e  # needs PINING_FOR_THE_DATA_TOKEN


def test_skillcorner_frames_carry_an_orientation_label():
    from scripts._loader_pining import load_matches

    _p, _mid, actions, frames, _home = next(iter(
        load_matches(providers=["skillcorner"], max_per_provider=1)
    ))
    labelled = frames[~frames["is_ball"].astype(bool)]["team_attacking_direction"]
    assert labelled.notna().all(), (
        f"{labelled.isna().mean():.1%} of player rows unlabelled -- "
        "acting_team_attacks_rtl will return all-False and ADR-028 silently no-ops"
    )
```

- [ ] **Step 2: Run and verify it fails**

Run: `python -m pytest tests/scripts/test_loader_orientation.py -v -m e2e`
Expected: FAIL — 100% unlabelled.

- [ ] **Step 3: Fix the loader**

Change `output_convention="absolute_frame"` to `"ltr"` at `:478`. Verify `convert_to_frames` labels
in that branch (`skillcorner.py:367`).

- [ ] **Step 4: Run; confirm no `OrientationUnresolvedWarning` remains**

Run: `python -m pytest tests/scripts/test_loader_orientation.py -v -m e2e`
Expected: PASS, with no `OrientationUnresolvedWarning` emitted during the load.

- [ ] **Step 5: Commit**

```bash
git add scripts/_loader_pining.py tests/scripts/test_loader_orientation.py
git commit -m "fix(scripts): orient SkillCorner frames in the pining loader (ADR-028 RC4)"
```

## Task 17: Retrain `GkCompletionModel` and re-assess the corpora

- [ ] **Step 1: Retrain the `default` variant on the corrected seam**

```bash
python scripts/train_gk_completion.py --providers gradientsports --max-per-provider 64 --variant default
```

Record OOF AUC, ECE and reliability slope. **The recorded gate is `ece <= 0.10` and
`|slope - 1| <= 0.25`.** If it fails, do NOT ship — PR 3 stays unreleased (see the release
constraint) rather than shipping with skew.

- [ ] **Step 2: Re-assess the `skillcorner` variant**

It is compromised by BOTH root causes — RC2's seam and RC4's unoriented frames. Confirm its recorded
corpus provenance in its `metrics.json` before retraining, then retrain with `--variant skillcorner`
and apply the same gate. If it now fails the gate, drop the variant rather than shipping it: the
precedent is 4.42.0, where the SkillCorner variant was withheld for exactly this reason.

- [ ] **Step 3: Re-assess TF-24 SkillCorner calibration corpora**

Re-run the Stage 1 / Stage 2 objectives on the corrected frames and diff the recommended defaults.
Report; do not auto-apply — TF-24 recommends, it does not change library constants.

- [ ] **Step 4: Commit + PR**

```bash
# DO NOT COPY THIS -- bare directory adds. See the re-planned Task 18 Step 6.
git add silly_kicks/tracking/_gk_completion_weights/ docs/research/
git commit -m "chore(weights): retrain GkCompletionModel on ADR-028-corrected geometry"
git push -u origin pr4-loader-and-weights
```

PR body must state that xS / xCross / ghost are **verified unaffected** (§3.4 — each resolves the
defended goal geometrically from GK mean-x) and must NOT be retrained on this basis.

---

## Task 17b: Reproject the passer in the cover-shadow measurement driver (owner-assigned 2026-07-30)

Surfaced by the 4.70.0 audit. `scripts/measure_cover_shadow_argmax_agreement.py:116` builds
`passer_xy = (float(row["start_x"]), float(row["start_y"]))` and **never reprojects** — verified: the
file contains no `acting_team_attacks_rtl` call at all. It is the same defect PR 2 fixed in
`features.py`, surviving in a research driver.

**Assigned to PR 4, not PR 2 or 3, for a substantive reason:** this is the RC4 research-harness class
(§5 already routes RC4 here), and — more importantly — fixing the code raises the question of whether
to **re-run** the measurement. Re-running needs owner-tier Gradient Sports data plus a pining token,
i.e. exactly the DGX/owner session PR 4 already requires. Fixing it in a local PR would correct the
code but strand the measurement question in a PR that cannot answer it.

**Files:**
- Modify: `scripts/measure_cover_shadow_argmax_agreement.py:116`

- [ ] **Step 1: Reproject, mirroring the PR 2 fix**

Before the per-action loop:

```python
    _flip = acting_team_attacks_rtl(actions, frames).to_numpy(dtype=bool)
```

Then at `:116`:

```python
        passer_xy = (float(row["start_x"]), float(row["start_y"]))
        if _flip[j]:
            passer_xy = (FIELD_LENGTH - passer_xy[0], FIELD_WIDTH - passer_xy[1])
```

Import `acting_team_attacks_rtl`, `FIELD_LENGTH`, `FIELD_WIDTH` from
`silly_kicks.tracking._action_orientation`. Confirm the loop variable is the positional `j` from
`enumerate(...)`, as in `features.py`.

- [ ] **Step 2: Decide on a re-run — and do NOT silently invalidate the 4.67.0 gate**

The 0.157 cheap-vs-exact argmax agreement that justified gating `max_single_defender_player_id` in
4.67.0 was measured with RC1 live. The cheap path consumes the passer; the exact path does not — so
RC1 degraded exactly one arm of that comparison.

**The verdict survives, and the bound is worth recording rather than re-deriving.** Home rows are
byte-identical under RC1, so even assuming all 152 of the 970 observed agreements sit on home rows
AND every away row flips from disagree to agree, post-fix agreement is at most
`(152 + 549) / 970 = 0.723`, still below the pre-registered 0.90 floor. Reaching 0.90 would require
an away share ≥74.3% — 18pp above the measured not-home share of 801/1414 — *simultaneously* with a
100% away flip.

So a re-run is **optional, not required to preserve the gate**. If it is run, record the new figure
in `docs/research/cover_shadow_identity/` alongside the 0.157, and state explicitly that the original
was RC1-contaminated. Do **not** overwrite the old number without that note — it is the figure cited
in CLAUDE.md and the 4.67.0 CHANGELOG.

> **RESOLVED — the re-run happened, and `0.723` is now superseded by a measurement.** The post-fix
> agreement is **0.0443** (`docs/research/cover_shadow_identity/agreement.json`,
> `agreement_rate: 0.0443298969072165`), against ~0.10 by chance — so the corrected cheap path agrees
> with the exact argmax **less than random**, and RC1 had been *inflating* the figure. The gating
> verdict from 4.67.0 is unchanged and now better supported: `0.0443 ≪ 0.723 ≪ 0.90`.
>
> The bound above is left in place deliberately: it was **sound reasoning that reached the right
> verdict without the measurement**, and the measurement landing an order of magnitude inside it is
> evidence the reasoning was conservative in the correct direction. Keeping both makes that legible;
> deleting the bound would leave only the number and none of the argument. What is *not* left standing
> is its conclusion that a re-run is optional — it was run, so the live figure is 0.0443 and any
> forward citation should use it.
>
> **Two different ceilings were published for this one argument, and neither is live.** This plan
> derives **0.723** = `(152+549)/970` from the *measured* not-home share `801/1414 = 56.6%`, while
> main's shipped 4.72.0 `CHANGELOG.md:61` and `docs/research/cover_shadow_identity/README.md:22` both
> publish **0.657** = `(152+485)/970`, i.e. a flat 50% away assumption. So a reader who searches for
> the *argument* rather than the string `0.723` finds a different number and no cross-reference. The
> shipped entries are **left unedited** — the house rule is that rewriting a released note is worse
> than letting it stand — and both are superseded by the same measurement: **0.0443**, which is below
> either ceiling and below chance. Cite 0.0443; treat 0.723 and 0.657 as two drafts of a bound that
> the measurement retired.

- [ ] **Step 3: Verify the driver still runs**

```bash
python scripts/measure_cover_shadow_argmax_agreement.py --help
```

Expected: usage text, no `UnicodeEncodeError` (the cp1252 `--help` gate covers `scripts/*.py`).

---

## Self-review

**Spec coverage.** D1 → Task 9 (Gate B) + Task 12. D2 → Tasks 5-6. D3 → Task 9 + Task 12 (unit
enumeration); the re-keys themselves are xfail-registered in Task 10 Step 4 and deliberately NOT
fixed in this cycle (spec §9 lists them out of scope as byte-identical on converter output).
D4 → Tasks 1-2, 4. D5 → Task 3. RC1 → Task 13. RC2 → Task 14. RC3 → Task 15. RC4 → Task 16.
Weights → Task 17. §6 registry → Tasks 7-11. §7 Hyrum → PR body requirements in Tasks 13, 15, 17.

**Known gap 1, stated rather than hidden:** the D3 re-keys are registered as xfails but never fixed
here. That is deliberate (§9), but it means PR 1 ships a repository with standing xfail markers for
D3 in addition to the four root causes. If you want them closed, that is a fifth PR and its own
decision.

**Known gap 2:** the `mirrored_pitch_absolute` mirror class has **zero real members** in v1 —
`add_shape_graph` emits only numeric columns, and the pitch-absolute lateral label ADR-045 D5 settled
lives in `infer_positions`, outside the `add_*` surface the registry is keyed on. The class is kept
because the vocabulary needs it the moment a genuine member appears, and it is exercised by the two
planted witnesses in Task 8 Step 3. Without those witnesses the enforcement machinery would ship
correct but never observed — the exact failure this plan's detection-first design exists to prevent,
and one that would have been introduced by the fix to a review finding rather than by the original
draft.

**Placeholders:** none. Every code step carries the code; every "follow the pattern" step names the
exact aggregator list and the worked example it follows.

**Type consistency:** `MirrorEntry` fields (`name`, `call`, `columns`, `tolerance`,
`tolerance_basis`, `home_team_id_role`, `non_vacuity`, `exempt_reasons`, `known_defect`) are used
identically in Tasks 7-11. `_entry(...)` keyword names match the dataclass. `canonical_scene()`,
`mirror_frames()`, `away_mask()` and `gate_xt()` are defined once and referenced consistently.

---

# PR 4 — RE-PLANNED 2026-08-01 (supersedes Tasks 16, 17, 17b above)

> **Everything under "PR 4 — RC4 loader + all weights work" above is SUPERSEDED.** Tasks 16/17/17b
> were implemented on branch `pr-s140-rc4-and-weights`, reviewed, and **parked**; four of the five
> work items were then independently shipped by ADR-052 / 4.72.0, three of them better. That branch
> is **abandoned** — kept on origin as a record. Do not rebase it. Start fresh from `main`.
>
> **The stashed weights are DROPPED, deliberately.** A stash is one `git stash clear` or one
> fresh clone from gone, and is invisible to everyone but one machine (part-deux review: "either
> tag it or drop it — either is better than a stash"). Dropping is the right half: those weights
> must be regenerated under `--feature-space moved` regardless (§11.3), so they are not merely
> superseded but **unusable**, and keeping them invites resurrection. The branch on origin already
> records what was built.
>
> Rationale and evidence: **spec §11**. Read §11.1 before writing any RC4 prose — a fabricated IDSSE
> measurement propagated into eight files last time.

**Version:** 4.73.0 / PR-S141. No new ADR (ADR-051 already scopes "PR 4 of 5").
**Base:** `main` @ `5a67212` (4.72.0 + PR #190's docs correction).

**What survives from the parked branch:** the RC4 **SkillCorner** fix only. Everything else reverts
to main.

---

## Task 16 (re-planned): RC4 — orient SkillCorner frames in the pining loader

**Files:**
- Modify: `scripts/_loader_pining.py` (`build_skillcorner_frames`, the PUBLIC frame builder at
  :446 — **not** `_build_skillcorner` at :498, which merely delegates and carries no
  `output_convention`; the two names differ by one underscore, which is exactly the shape of a silent
  path pin)
- Create: `tests/scripts/test_loader_orientation.py`

- [ ] **Step 1: Branch from main**

```bash
git checkout main && git pull --ff-only
git checkout -b pr-s141-rc4-and-retrain
```

- [ ] **Step 2: Measure the PRE-FIX state and record it**

This is not optional bookkeeping — it is the step whose absence produced the fabrication in §11.1.
Run the loader **before** editing it, on one SkillCorner match, and record the unlabelled fraction
and the `acting_team_attacks_rtl` flip fraction. Expect `1.0000` unlabelled and a `0.0000` flip.

**Do NOT extend the claim to any other provider without measuring that provider pre-fix.** IDSSE is
NOT affected: `sportec.py:137` calls `finalize_orientation` unconditionally before the
`output_convention` branch, and spec §2.2 independently records "On IDSSE flip is True on exactly
718/718".

- [ ] **Step 3: Fix the SkillCorner builder ONLY**

Change that one call's `output_convention="absolute_frame"` to `"ltr"`. The `"ltr"` branch falls back
to `orient_frames_to_ltr_by_geometry` (ADR-035) when `home_team_start_left` is absent, which is the
case on this path — verify by reading `silly_kicks/tracking/skillcorner.py` before relying on it.

Leave `_build_idsse` and `_build_gradientsports` alone.

- [ ] **Step 4: Re-measure post-fix; both numbers now exist**

Expect unlabelled `0.0000`, both direction labels present, and a **non-zero** flip fraction (the
re-projection firing for the first time). Only now may a `X → Y` claim be written, and only for
SkillCorner.

- [ ] **Step 5: The guard — assert the RESOLVED CONVENTION, not a keyword**

Per spec §11.9, the parked guard had two holes. Requirements:

1. It must detect a builder that **omits** `output_convention` as well as one that passes
   `absolute_frame` — `_build_gradientsports` omits it entirely and was invisible to a keyword
   matcher.
2. Its non-vacuity partner must **call the guard's own body**, not re-implement a weaker matcher
   inline.
3. `scripts/_loader_pining.py` is a **library, not a corpus driver** — settled by
   `test_corpus_driver_resilience.py`'s own population logic, which skips underscore-prefixed files.
   No `_driver` migration or provenance obligation attaches to this edit.

- [ ] **Step 6: Prove the guard is red on the pre-fix source, then commit**

Restore the pre-fix line, confirm the guard fails, restore the fix. Commit RC4 **alone** — it must
land before any trainer run (Task 17 Step 1 explains why).

---

## Task 17 (re-planned): regenerate the probe, then retrain under main's CLI

- [ ] **Step 1: Understand the ordering constraint before running anything**

ADR-052 shards `_extract` per match on a token that captures **neither geometry nor library
version**. If a trainer run happens on rebased main *before* RC4 lands, it mints a generation; RC4
then changes the SkillCorner frames while the token does not, and the retrain silently reads stale
features. The only signal is a per-item `skip (shard exists)`.

**RC4 must already be committed before the first extraction.** That makes the failure unreachable
rather than guarded. Also: **do not pass `--cache-features`** — it is a bare `Path.exists()` check
that bypasses the generation directory entirely.

Optional defence in depth: add a derived orientation entry to `token_inputs`. Note `cache_token()`
would have missed both RC2 and RC4, so copy ghost's *discipline*, not its body.

> ## ⚠ STEP 2 BELOW IS SUPERSEDED — read this box first (spec §11.3.1)
>
> **The whole of Step 2 aims at the wrong artifact.** `predictions_moved` ends in an ELEMENT-WISE
> `np.allclose`, so `probe_old` and `probe_new` must be **row-aligned**. The 4.21.0 training matrix
> (1666 rows) against the current corpus (~3491) does not answer wrongly — it raises
> `ValueError: operands could not be broadcast together with shapes (1666,) (3491,)`, at the guard,
> after the whole corpus pass is paid for.
>
> **`probe_old` = the SAME corpus as the fresh fit, extracted under PRE-CHANGE geometry** — i.e.
> commit **`641dadf`** (4.70.0), immediately before RC2/RC3/RC5 landed in `89dd9af` (4.71.0). The
> guard asks a *serving* question ("does what production emits change?"), not a fitting-provenance
> one. The docstring says as much in its next clause — *"they are the SAME array whenever the feature
> space did not move"* — and the registered test builds the moved case as `X_new = X_old + 5.0`,
> same rows, shifted geometry.
>
> So **"pre-RC2", which Step 2 opens by calling wrong, is right** — for the row-alignment reason,
> not the one originally offered. Keep Step 2's *discipline* (validate, do not infer; a fail is
> ambiguous; check `N` before reading a verdict) and discard its target. The vintage archaeology it
> prescribes cost ~50 min of DGX compute and could have been pre-empted by one line: calling
> `predictions_moved` with two mismatched shapes, no corpus required.
>
> Alignment is verified in spec §11.3.1: both vintages concat in `load_matches` order (HEAD combines
> from `res.keys`, explicitly **not** `reconcile`, whose filename sort would have re-ordered), and
> the entire `641dadf..4b15365` loader diff is the SkillCorner-only RC4 change, so the GS order is
> unchanged. The remaining check is the row COUNT, which is loud.

- [ ] **Step 2 (SUPERSEDED — see the box above): Produce `--probe-old` from a VINTAGE-VALIDATED commit (not merely pre-RC2)**

Main's `--mode retrain` requires `--feature-space {unchanged,moved}`, and this case is **`moved`**
(the flag means feature *values* changed, which a geometry correction does). `moved` requires
`--probe-old`: the design matrix the **committed** model was fit on. It cannot be reconstructed from
the artifact — `to_dict()` persists no design matrix.

**"Pre-RC2" is the WRONG target.** The probe must be the matrix the **committed weights** were fit
under. Those coefficients were fit at **`e3d5e92`** (2026-06-09, **4.21.0**) — ~50 releases before
RC2, with several geometry changes in between (ADR-024 amendments, PR-S104). Get the vintage wrong
and `predictions_moved` compares the committed model against coordinates *it never saw either* —
meaningless, and nothing reports it.

**Do NOT identify the vintage with `git log -1 -- model.json`.** That finds the last **touch**, not
the last **fit**, and it is how revision 1 of this plan got it wrong: `7e875c8` (4.21.4) touched
`model.json` additively only — version bump plus the gate fields, **no `coef` change**. `7e875c8` is
nonetheless an admissible vintage, but *by validation* (its corpus-identity probe was run and passed,
CLAUDE.md PR-S91) rather than by recency. Verify, do not infer.

**Validate a candidate vintage by RUNNING THE TRAINER there** — not by passing `--mode rebundle`,
which does not exist at 4.21.x (measured at `7e875c8`: 0 occurrences of `--mode`). At that vintage
the corpus-identity assertion runs **unconditionally** (`_CORPUS_IDENTITY_ATOL`, 9 occurrences),
because rebundle is the only behaviour there.

**The check is ONE-SIDED — read a failure carefully.** A **pass** is strong: code and corpus both
reproduce the committed weights. A **fail is ambiguous** — wrong vintage *or* corpus drift (4.49.0
and 4.50.0 each shipped GS-only retrain triggers), and `_CORPUS_IDENTITY_ATOL = 0.05` is loose enough
that a pass means something while a fail does not localise. Do not start a vintage hunt on a single
failure.

**And there IS a concrete discriminator — check it BEFORE interpreting the verdict, because the run
can manufacture the ambiguity by itself.** The committed `default/metrics.json` records
`n_rows: 1666`, `n_native: 1395`, consistent with CLAUDE.md's "30 WC2022 matches" (~55 rows/match).
The pining GS manifest **now offers 64**, and `--max-per-provider` defaults to 64 at *both* the
vintage and HEAD — so an unqualified run loads roughly **twice** the original corpus and the fresh
fit differs for a reason that has nothing to do with vintage. The trainer prints `N=` on startup, so
read it first:

| observed `N` | what the assertion result means |
|---|---|
| ≈ **1666** | corpus matches — the verdict is a genuine vintage check, both ways |
| ≫ 1666 (≈3550) | corpus differs — the verdict is **uninterpretable as a vintage check**, whichever way it lands |

A **pass** at N≫1666 would be the more dangerous outcome of the two: it would look like a clean
validation while proving only that the tolerance is loose. Note also that re-running with
`--max-per-provider 30` recovers the *count* but not necessarily the *same thirty matches*, since
nothing pins selection order across releases — so treat a corpus mismatch as a reason to widen the
evidence (compare `n_native`, `base_rate`, `label_split` against `metrics.json`), not as something a
single flag reliably fixes.

Then run the extraction at the validated commit and dump `X_all` to parquet. `--cache-features` is
wired only into `_train_skillcorner`, not `main()`, so the `default` variant needs its own small dump.

**Do NOT pass `--feature-space unchanged` to get past the guard.** That declares something false and
makes the guard compare the committed model against coordinates it never saw.

- [ ] **Step 3: Persist the probe OUT-OF-BAND (not inside the artifact)**

ADR-052 records this as an ADR-011 follow-up, and this PR regenerates the artifact anyway, so it is
the natural moment. **But not in the weights directory:** `ADR-044:25` states that distributed model
artifacts carry *"learned parameters only, not per-sample training data"*, and a design matrix is
per-sample training data (part-deux review 2 — the ADR-052 follow-up was written without checking
ADR-044).

**Decision: the probe lives under `docs/research/`, referenced from `metrics.json` by path AND
SHA256.** Wheel stays parameters-only; the citation stays resolvable. A fixed sample, not a corpus.

Note regardless: adding any file to the weights directory moves `SHA256SUMS`, so a checksum-pinning
consumer sees a diff with no value change (the ADR-050 precedent).

- [ ] **Step 4: Retrain `default` from a CLEAN tree**

`--mode retrain --feature-space moved --probe-old <parquet> --reason "<what changed>"`, no
`--allow-dirty`. The reason string must name **RC2/RC5** — *not* RC4: the `default` variant is
gradientsports-only and RC4 touches only the SkillCorner builder. The parked run's reason string got
this wrong.

Gate: `ece <= 0.10` and `|slope - 1| <= 0.25`. If it fails, do not ship.

- [ ] **Step 5: Re-assess the SkillCorner variant**

It IS in RC4's blast radius (unlike `default`). Expect main's `predictions_moved` guard to be the
arbiter. The 4.42.0 precedent stands: if it fails its gate, drop the variant rather than ship it.

**Use `--max-per-provider 10`, and the reason is COMPLIANCE, not corpus-matching.** ADR-038 grew the
pining SkillCorner listing 10 → 108, and the 98 additions are owner-tier. **Measured** — these are
the manifest's own `visibility` strings via `match_visibility`, not a judgement about the data:

```
gradientsports :  64 matches -- manifest says "private"   (see the caveat below)
skillcorner    :  10 "public" + 98 "private"
select_match_ids(providers=["skillcorner"], max_per_provider=10)
    -> ['1886347','1899585','1925299','1953632','1996435','2006229','2011166','2013725','2015213','2017461']
    -> visibility: ['public']   (all ten)
```

So the cap selects exactly the public arm — verified, not assumed, and it also happens to match the
committed variant's `n_matches: 10`.

**The two `"private"` labels do NOT mean the same thing, and only one of them is an access
restriction.** The SkillCorner 98 are genuinely restricted — ADR-038 records them as *"owner-tier,
restricted, all Real Madrid LaLiga+UCL"* — so a defaulted cap-64 run would pull 54 access-restricted
matches into a **distributed wheel artifact**, and that is the reason for the cap.

**Gradient Sports is a situation the manifest simply cannot express, and the label reads as far more
restrictive than the reality.** The data is **publicly obtainable by anyone** — it sits in a public
Google Drive folder behind a free signup — and the provider's own site *encourages people to share
the work*. What is unresolved is the precise open-source **licence wording**, not who may read it,
and not whether the provider wants it used. The project's practice of not republishing the raw data
is a **deliberate conservative choice by the owner**, not a constraint imposed by the provider, and
it costs nothing because anyone can fetch the same files directly.

So `is_public_row` being fail-closed remains the right default for an automated redistribution
decision — a machine should not infer licence terms — but **do not read the GS `"private"` label as
evidence of secrecy or of a provider restriction, and do not cite it as one.** Treating the two
labels as equivalent would over-restrict GS work for no benefit while under-describing why the
SkillCorner cap actually matters.

**`train_gk_completion.py` has NO ADR-038 enforcement** — `is_public_row` / `artifact_label` /
`assert_public_corpus` appear only in `train_xcross_attempt.py` and `train_xshot_occurrence.py`.
So nothing in this trainer would have refused that run, or labelled its output restricted. Wiring
the taxonomy in is a **follow-up, not this PR**: it would dirty the tree that the retrain requires
clean, and it belongs with the other trainers rather than bolted onto one.

**Not a concern, and worth saying so plainly:** the `default` variant is trained entirely on Gradient
Sports data and has been bundled since 4.21.0 under ADR-024. Given that the data is publicly
obtainable and the provider encourages the work being shared, bundling *learned parameters* fit on it
raises no real issue — and ADR-044's parameters-only rule bounds it further by keeping per-sample
training data out of the artifact entirely. This retrain changes the corpus size but not its
character (30 GS matches → 64, same provider, same terms).

It sits oddly beside xS/xCross, which shipped their `public` arms and rejected `full` — but those
rejections were driven by **measured held-out degradation**, not by any compliance rule, so they set
no precedent that constrains this variant.

- [ ] **Step 6: DGX prerequisite**

`~/Development/silly-kicks/.venv` has `ruthless-efficiency 0.2.1`; `pyproject.toml` now requires
`>=0.4.0` (`scripts/_driver.py` keys generations on `ruthless.fingerprint`, public only from 0.4.0).
Symptom if skipped: `ImportError: cannot import name 'fingerprint'`, which surfaces as dozens of test
failures from one cause. Install before anything else.

---

## Task 18 (new): correct the stale records this PR is responsible for

- [ ] **Step 1: The 4.71.0 note, in two places**

`CHANGELOG.md`'s 4.71.0 heading and `ADR-051`'s status header both still say "ships within 4.72.0
alongside PR 4". 4.72.0 shipped **without** the retrain. Correct both.

**THREE sites, not two.** This plan's own "Version numbering for PR 3 + PR 4" section carries the
same staleness — including a literal **"Tag ONLY `v4.72.0`"** instruction, which is worse than a
stale note because following it publishes the skew. It was missed on the first pass because the
section is headed "do not re-litigate", and because an edit six lines below it (the merge-settings
STALE note) left the version numbers untouched. Enumerate by search, not by memory:
`grep -rn '4\.72\.0' CHANGELOG.md TODO.md docs/superpowers/ | grep -v 'ADR-052\|4\.72\.0\] —'`.

- [ ] **Step 2: Withdraw the IDSSE retraction**

Spec §2.2's IDSSE column was retracted as "provisional" on a fabricated premise (§11.1). That
retraction propagated into CHANGELOG, CLAUDE.md, TODO.md and ADR-051 on the abandoned branch — none
of which reached main, so **verify** rather than assume, and remove any that did.

- [ ] **Step 3: Correct `0.723` in THIS plan — the main-side fix is already taken**

**PR #190 (part-deux) corrects the 4.72.0 CHANGELOG and CLAUDE.md.** Do not duplicate it; that is
their release's error and taking it there removes a conflict from this diff. Copy their 4.67.0
handling as precedent: leave the shipped entry **unedited** with a superseded-by note, since
rewriting a shipped release note is worse than leaving it while leaving it unmarked invites a forward
citation.

**What IS ours:** `0.723` at line ~2005 of this plan — Task 17b's worst-case bound, written in PR 3
and merged to main. The reviewer searched CHANGELOG and CLAUDE.md and did not find it; it lives here.
It is derived from the pre-fix `0.157` and is stale in the same direction (the post-fix agreement is
`0.0443`, i.e. **0.44× chance, worse than random** — the defect was *inflating* it). The gating
verdict is unchanged and better supported.

- [ ] **Step 4: Record the TF-24 disposition explicitly (do not leave it silent)**

§3.4 names TF-24 as RC4's second affected consumer, so a **No** verdict must be stated, not implied.
Per spec §11.7 the evidence is: Stage 1's shipped params are TF-24-calibrated but carrier inference is
orientation-INVARIANT (measured 40/40, delta 1.01e-14), and Stage 2's params ship as **engineering**
defaults TF-24 never set. **No re-sweep trigger; no shipped value moves.** One line in the CHANGELOG
and the ADR is enough — but it must be there.

- [ ] **Step 5: Optional — carry the four additive trainer improvements**

From spec §11.2: served-vs-probe coefficients in `metrics.json`; an actionable `SystemExit` on
rebundle drift; `weight_deltas`/`coefficients_changed` in metrics; real `save`/`load` test coverage
(main's trainer tests import no model). Small, independent, and each stands alone — drop any that
complicate the diff.

> **RESOLVED — the list was written against the ABANDONED branch's trainer, and main has moved.**
> Checked against `scripts/train_gk_completion.py` on main before building anything:
>
> - **#1 and #3 are already there in substance.** `metrics.json` records `run_commit`,
>   `run_tree_dirty`, `mode`, `reason` and **`superseded_coef`** — the full replaced coefficient
>   vector, on **both** the default and SkillCorner paths. Any `weight_delta` is derivable from it,
>   so adding one would persist a computed quantity beside its own inputs. **Not built.**
> - **#2 built.** The two rebundle sites were byte-identical four-line `np.testing.assert_allclose`
>   blocks raising a float diff with no remedy, at the end of a half-hour corpus pass — and
>   `assert_allclose` short-circuits on `coef`, so it never reports the `mean`/`std` drift that is
>   precisely the signature of a feature-space move. Extracted to `_assert_rebundle_reproduces`,
>   which reports all four and names the retrain command. Three tests, both sides, one asserting the
>   **message** (asserting only "it raised" would pass on the old behaviour).
> - **#4 built.** Every existing test used plain dicts, so `_as_weights` — which reads four *private*
>   attributes off a real `GkCompletionModel` — was never run against one. Three tests: the real
>   artifact contract with shapes tied to the feature list, a save/load round trip, and a
>   non-vacuity partner. Teeth proven by planting a 0.01 coefficient corruption into a round-tripped
>   model: caught by both prongs.
>
> **Sequencing note, which turned out not to bite.** Under the two-commit structure the retrain runs
> against *already-committed* code, so anything that changes `metrics.json` fields could not reach
> this artifact without a third commit. That would have forced a real choice — but since #1 and #3
> were the only artifact-changing items and both were already satisfied, #2 and #4 land cleanly:
> one touches a failure path not taken by a successful retrain, the other is tests only. The shipped
> `run_commit` therefore records the RC4 commit, which is exactly right — it names the code that ran.

---

- [ ] **Step 6: Stage by PATH, never by bare directory**

`docs/research/` holds **15** entries as of `4b15365` — 13 directories plus 2 standalone `.md` files, so
count with `find docs/research -maxdepth 1 -mindepth 1 | wc -l`, not `ls -d */`, which sees only 13.
This cycle's own `adr028_rc4_orientation/` was the 15th. So a bare
`git add docs/research/` sweeps whatever else happens to be dirty. The part-deux session nearly swept
an unrelated `_partition.py` into a commit exactly this way, and the superseded Task 17 Step 4 above
still shows the bad form.

Name the paths, and check before committing:

```bash
git add --dry-run <each path>     # read the list before it becomes a commit
git status --porcelain            # nothing unexpected staged
```

The weights artifact is the one place this matters most: `model.json`, `metrics.json` and
`SHA256SUMS` must move **together**, and the out-of-band probe under `docs/research/` must move with
them or `metrics.json`'s SHA256 reference dangles.

---

## PR 4 exit criteria

**COMMIT STRUCTURE — exactly two commits, and the second is the last thing that happens.**

1. **RC4 alone**, already committed. Separate *only* because the sequencing constraint requires it:
   the trainer must execute against RC4-corrected code, or ADR-052's shard token silently serves
   stale SkillCorner features (§11.4).
2. **Everything else in ONE commit** — weights, research artifacts, doc corrections, and the
   five-site version bump — made **after** `/final-review` has run and every finding is fixed in the
   working tree. Do not commit and then review: the fixes become extra commits, and this PR merges
   with `--merge`, so they persist on main rather than being squashed away.

Then ONE merge. Never merge RC4 to main on its own — that would put corrected serving geometry on
main without the paired retrained weights, a smaller instance of the very skew that left 4.71.0
untagged (§11.8).

- RC4 SkillCorner fixed, with **both** sides of the claim measured and the guard proven red pre-fix.
- `default` retrained under `--feature-space moved` with a persisted probe, from a clean tree, gate green.
- SkillCorner variant assessed; dropped rather than shipped if it fails.
- 4.73.0 / PR-S141 in all five version sites (`uv lock`, do not hand-edit).
- Stale 4.71.0 notes and the superseded cover-shadow figure corrected.
- **`/final-review` runs BEFORE the commit, not after.** Everything it finds is fixed in the working
  tree, and only then is the commit made. Reviewing a commit and then fixing it produces
  fix-the-fix commits — which a `--merge` PR then preserves in history permanently, unlike a squash.
  Given this cycle's history, treat any `X → Y` claim in the review output as unverified until both
  sides are shown.
- **Merge with `--merge`, NOT a squash.** `metrics.json` records `run_commit`; a squash mints a new
  SHA and the citation stops resolving. The repo now allows merge commits (`allow_merge_commit: true`
  as of 2026-08-01 — it was squash-only, so the habit is stale). Verify afterwards:
  `git merge-base --is-ancestor <the SHA metrics.json records> origin/main`.
- Then **tag** — this PR is what makes the first tag since `v4.70.0` safe.
