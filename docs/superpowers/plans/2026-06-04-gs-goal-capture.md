# Gradient Sports Goal-Capture Correctness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Gradient Sports SPADL converter capture own goals (`RE`+`G`) and cross-goals (`CR`+`G`) correctly, exclude voided (`nonEvent`) events, and make own goals count in VAEP labels for all providers.

**Architecture:** Four changes. (1) Converter excludes `possessionEvents.nonEvent==True` voided events in the exclusion stage (runs first). (2) `RE`+`G` → `bad_touch`+`owngoal` in dispatch, validated by a post-LTR own-half geometry tripwire. (3) `CR`+`G` keeps the cross and synthesizes a `shot`+`success` (foul-synthesis pattern). (4) `vaep/labels.py` detects goals/own-goals via extracted `_is_goal`/`_is_owngoal` helpers, dropping the `"shot"` type-gate on own goals. Spec: `docs/superpowers/specs/2026-06-04-gs-goal-capture-design.md`.

**Tech Stack:** Python, pandas, numpy. Tests: pytest. Lint: ruff. Types: pyright. Run via `.venv/Scripts/python.exe`.

**Canonical pipeline order (invariant):** exclusion (incl. `nonEvent`) → dispatch (incl. provisional `RE`+`G` owngoal) → build frame → time impute → tackle passthrough → `_derive_end_coordinates` → synthesis (foul + cross-goal shot) + dense renumber → `to_spadl_ltr` → owngoal geometry tripwire → clip → finalize.

---

## File Structure

- `silly_kicks/spadl/gradientsports.py` — converter. Add `import warnings`; `nonEvent` exclusion in `convert_to_actions` exclusion block; `RE`+`G` owngoal refinement in `_dispatch_actiontype_resultid`; cross-goal synthesis in the synthesis block; post-LTR tripwire.
- `silly_kicks/vaep/labels.py` — extract `_is_goal`/`_is_owngoal`; route all ~6 sites through them.
- `scripts/_loader_pining.py` — `_gs_flatten_events`: map `nonEvent`.
- `tests/spadl/test_gradientsports.py` — unit + realistic + composition tests.
- `tests/datasets/gradientsports/synthetic_match.json` — add `nonEvent` to events; add `RE`+`G` OG, `CR`+`G` cross-goal, and a `nonEvent==True` disallowed `SH`+`G`.
- `tests/vaep/test_labels*.py` — owngoal-now-counts test.
- `tests/spadl/test_gradientsports_scoreline_e2e.py` — owner-gated scoreline guard (`@pytest.mark.e2e`).
- `docs/superpowers/adrs/ADR-0NN-owngoals-counted-in-vaep-labels.md` — new ADR.
- `CHANGELOG.md`, `TODO.md`, `pyproject.toml`, `silly_kicks/__init__.py`, `uv.lock` — release.

---

## Task 1: Component 4 — exclude voided (`nonEvent==True`) events + observability

**Files:**
- Modify: `silly_kicks/spadl/gradientsports.py` (add `import warnings`; exclusion block ~`:341-385`)
- Test: `tests/spadl/test_gradientsports.py`

- [ ] **Step 1: Write failing tests**

```python
class TestGradientsportsNonEventExclusion:
    def test_nonevent_true_excluded_and_tallied(self):
        df = pd.concat([_df_minimal_pass(), _df_minimal_pass()], ignore_index=True)
        df.loc[1, "event_id"] = 2
        df.loc[1, "possession_event_id"] = 2
        df["nonEvent"] = [False, True]
        actions, report = gs_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True, home_team_start_left_extratime=True
        )
        assert len(actions) == 1                      # the nonEvent=True row dropped
        assert report.excluded_counts.get("nonEvent") == 1

    def test_nonevent_column_absent_warns_and_noops(self):
        df = _df_minimal_pass()                        # no nonEvent column
        with pytest.warns(UserWarning, match="nonEvent"):
            actions, report = gs_mod.convert_to_actions(
                df, home_team_id=100, home_team_start_left=True, home_team_start_left_extratime=True
            )
        assert len(actions) == 1                       # no-op: row kept
        assert "nonEvent" not in report.excluded_counts  # "not checked" signal (key omitted)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/spadl/test_gradientsports.py::TestGradientsportsNonEventExclusion -q`
Expected: FAIL (`nonEvent` not excluded; no warning).

- [ ] **Step 3: Implement**

Add at top of `gradientsports.py` imports (after `from collections import Counter`):
```python
import warnings
```

In `convert_to_actions`, in the exclusion block, immediately AFTER `is_excluded = is_excluded_ge | is_excluded_pair` and the `excluded_counts` tallies, BEFORE `events = events.loc[~is_excluded].reset_index(drop=True)`:
```python
    # Component 4: exclude voided ("annulled") events — possessionEvents.nonEvent == True
    # (play called back for a foul/advantage/offside, disallowed goals). Optional column:
    # absent -> observable no-op (warn + omit the report key) so an under-equipped caller is
    # not silently left emitting voided events. See ADR + spec Component 4.
    if "nonEvent" in events.columns:
        # Robust bool coercion (NOT .astype(bool) — that maps the string "false" to True and would
        # INVERT the exclusion, dropping real events and keeping voided ones). Only true-ish counts.
        _ne = events["nonEvent"]
        if _ne.dtype == bool:
            is_nonevent = _ne.fillna(False).to_numpy()
        else:
            def _truthy(v):
                # Handles Python bool AND numpy bool (np.True_), strings, None/NaN. Avoids the
                # `v is True` trap (False for np.True_) and the `.astype(bool)` trap ("false" -> True).
                if isinstance(v, str):
                    return v.strip().lower() == "true"
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    return False
                return bool(v)
            is_nonevent = _ne.map(_truthy).fillna(False).astype(bool).to_numpy()
        excluded_counts["nonEvent"] = int((is_nonevent & ~is_excluded).sum())
        is_excluded = is_excluded | is_nonevent
    else:
        warnings.warn(
            "gradientsports: 'nonEvent' column not supplied — voided events (annulled plays, "
            "including disallowed goals) are NOT excluded. Map possessionEvents.nonEvent into the "
            "input to enable Component-4 exclusion.",
            UserWarning,
            stacklevel=2,
        )
        # excluded_counts intentionally has NO 'nonEvent' key here: "not checked" != "0 voided".
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/spadl/test_gradientsports.py::TestGradientsportsNonEventExclusion -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add silly_kicks/spadl/gradientsports.py tests/spadl/test_gradientsports.py
git commit -F .git/COMMIT_T1.txt   # subject: "feat(spadl): GS excludes nonEvent voided events (+observable no-op)"
```

---

## Task 2: Component 1 — `RE`+`G` → `bad_touch` + `owngoal` (dispatch)

**Files:**
- Modify: `silly_kicks/spadl/gradientsports.py` `_dispatch_actiontype_resultid` (~`:214-246`)
- Test: `tests/spadl/test_gradientsports.py`

- [ ] **Step 1: Write failing test**

```python
class TestGradientsportsOwnGoalCapture:
    def test_re_g_is_bad_touch_owngoal_conceding_team(self):
        df = _df_minimal_pass()
        df.loc[0, "possession_event_type"] = "RE"
        df.loc[0, "shot_outcome_type"] = "G"
        df.loc[0, "team_id"] = 100          # conceding team (acting)
        df.loc[0, "player_id"] = 7          # OG scorer (rebounder)
        # ball in conceding team's own half so the tripwire (Task 3) passes:
        df.loc[0, "ball_x"] = -45.0         # centered; near own goal after LTR
        actions, _ = gs_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True, home_team_start_left_extratime=True
        )
        assert actions.iloc[0]["type_id"] == spadlconfig.actiontype_id["bad_touch"]
        assert actions.iloc[0]["result_id"] == spadlconfig.result_id["owngoal"]
        assert actions.iloc[0]["team_id"] == 100
        assert actions.iloc[0]["player_id"] == 7

    def test_re_without_g_still_keeper_save(self):
        df = _df_minimal_pass()
        df.loc[0, "possession_event_type"] = "RE"
        df.loc[0, "shot_outcome_type"] = None
        actions, _ = gs_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True, home_team_start_left_extratime=True
        )
        assert actions.iloc[0]["type_id"] == spadlconfig.actiontype_id["keeper_save"]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/spadl/test_gradientsports.py::TestGradientsportsOwnGoalCapture -q`
Expected: FAIL (RE+G → keeper_save/fail, not bad_touch/owngoal).

- [ ] **Step 3: Implement**

In `_dispatch_actiontype_resultid`, AFTER the `keeper_pick_up` refinement (`type_id_arr = np.where(is_catch, ...)`, ~`:218`) and BEFORE the `# result_id dispatch` comment, add:
```python
    # Component 1: RE + shotOutcome "G" is an OWN GOAL (bad_touch + owngoal). Provisional here;
    # the post-LTR geometry tripwire in convert_to_actions validates/reverts. Takes priority over
    # the RE -> keeper_save/keeper_pick_up handling. Scorer = gameEvents.playerId (= rebounderPlayerId),
    # team = conceding team — both kept unchanged (ADR-001). See spec Component 1.
    is_owngoal = (pe == "RE") & (shot_outcome == "G")
    type_id_arr = np.where(is_owngoal, at_ids["bad_touch"], type_id_arr).astype("int64")
```

Then change the result dispatch to include `is_owngoal` (between `shot_goal` and `is_yellow`):
```python
    result_conds = [pass_success, shot_goal, is_owngoal, is_yellow, is_red]
    result_choices = [
        rs_ids["success"],
        rs_ids["success"],
        rs_ids["owngoal"],
        rs_ids["yellow_card"],
        rs_ids["red_card"],
    ]
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/spadl/test_gradientsports.py::TestGradientsportsOwnGoalCapture -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add silly_kicks/spadl/gradientsports.py tests/spadl/test_gradientsports.py
git commit -F .git/COMMIT_T2.txt   # "feat(spadl): GS RE+G -> bad_touch+owngoal (own-goal capture)"
```

---

## Task 3: Component 1 — post-LTR own-goal geometry tripwire (validate + WARN + revert)

**Files:**
- Modify: `silly_kicks/spadl/gradientsports.py` `convert_to_actions` (after `to_spadl_ltr`, ~`:563`)
- Test: `tests/spadl/test_gradientsports.py`

- [ ] **Step 1: Write failing test**

```python
class TestGradientsportsOwnGoalTripwire:
    def _re_g(self, ball_x):
        df = _df_minimal_pass()
        df.loc[0, "possession_event_type"] = "RE"
        df.loc[0, "shot_outcome_type"] = "G"
        df.loc[0, "team_id"] = 100
        df.loc[0, "ball_x"] = ball_x        # centered coords
        return df

    def test_re_g_in_attacking_half_reverts_with_warning(self):
        # ball at attacking end for the acting team -> implausible OG -> revert + WARN.
        df = self._re_g(ball_x=45.0)        # see note: pick a value that maps to start_x >= 52.5 post-LTR
        with pytest.warns(UserWarning, match="own-goal"):
            actions, _ = gs_mod.convert_to_actions(
                df, home_team_id=100, home_team_start_left=True, home_team_start_left_extratime=True
            )
        assert actions.iloc[0]["result_id"] == spadlconfig.result_id["fail"]
        assert actions.iloc[0]["type_id"] == spadlconfig.actiontype_id["keeper_save"]

    def test_re_g_in_own_half_kept_as_owngoal_no_warning(self):
        df = self._re_g(ball_x=-45.0)       # maps to start_x < 52.5 post-LTR
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("error")        # any warning fails the test
            actions, _ = gs_mod.convert_to_actions(
                df, home_team_id=100, home_team_start_left=True, home_team_start_left_extratime=True
            )
        assert actions.iloc[0]["result_id"] == spadlconfig.result_id["owngoal"]
```

> **Verification note for implementer:** the LTR convention is "acting team attacks toward high-x"
> (`orientation.py` docstring: "all teams attack left-to-right … shots cluster at high-x for both teams"),
> so an OG ball sits in the **own half, `start_x < field_length/2`** → revert when `start_x >= field_length/2`.
> **These synthetic tests are NOT ground truth for the inequality direction** — the `ball_x` values are
> hand-placed, so the test passes for whichever direction the implementer assumes (circular). The ONLY
> real-coordinate validation of the inequality is the **owner-gated e2e in Task 7**
> (`test_real_own_goals_captured_through_converter_no_tripwire_warn`), which runs the converter on the 3
> actual WC2022 own goals and fails if the tripwire reverts them. Treat that e2e as the gate on this
> inequality; if it fails, the inequality is backwards, not the fixture.

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/spadl/test_gradientsports.py::TestGradientsportsOwnGoalTripwire -q`
Expected: FAIL (no tripwire; attacking-half OG stays owngoal, no warning).

- [ ] **Step 3: Implement**

In `convert_to_actions`, immediately AFTER the `actions = to_spadl_ltr(...)` call (~`:563`) and BEFORE the coordinate-clip block:
```python
    # Component 1 tripwire: an RE+G own goal must sit in the conceding (acting) team's OWN half.
    # SPADL-LTR puts the acting team attacking toward high-x, so its own goal is at x=0 and a true
    # own goal's ball is at start_x < field_length/2. A row failing this is a likely rebound-GOAL or
    # a feed anomaly -> WARN + revert to the default RE handling. Converts the n=3 rule into a
    # self-policing one (see spec Component 1).
    _og = (actions["result_id"] == spadlconfig.result_id["owngoal"]).to_numpy()
    if _og.any():
        _bad = _og & (actions["start_x"].to_numpy() >= spadlconfig.field_length / 2.0)
        if _bad.any():
            warnings.warn(
                f"gradientsports: {int(_bad.sum())} RE+G own-goal(s) with ball in the acting team's "
                "attacking half (start_x >= field_length/2) — reverting to keeper_save/fail (possible "
                "rebound-goal or feed anomaly).",
                UserWarning,
                stacklevel=2,
            )
            actions.loc[_bad, "type_id"] = spadlconfig.actiontype_id["keeper_save"]
            actions.loc[_bad, "result_id"] = spadlconfig.result_id["fail"]
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/spadl/test_gradientsports.py::TestGradientsportsOwnGoalTripwire -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add silly_kicks/spadl/gradientsports.py tests/spadl/test_gradientsports.py
git commit -F .git/COMMIT_T3.txt   # "feat(spadl): GS own-goal post-LTR geometry tripwire (warn+revert)"
```

---

## Task 4: Component 2 — `CR`+`G` cross-goal: keep cross + synthesize a shot

**Files:**
- Modify: `silly_kicks/spadl/gradientsports.py` synthesis block (`convert_to_actions`, ~`:507-549`)
- Test: `tests/spadl/test_gradientsports.py`

This reworks the synthesis block to emit BOTH foul rows (existing) and cross-goal shot rows (new) from the single 1:1 `actions`↔`events` alignment, with one combined `0.5`-offset insert + dense renumber.

- [ ] **Step 1: Write failing test**

```python
class TestGradientsportsCrossGoal:
    def test_cr_g_keeps_cross_and_synthesizes_shot(self):
        df = _df_minimal_pass()
        df.loc[0, "possession_event_type"] = "CR"
        df.loc[0, "set_piece_type"] = "F"          # free-kick cross -> shot_freekick
        df.loc[0, "shot_outcome_type"] = "G"
        df.loc[0, "cross_outcome_type"] = "I"      # cross-as-pass incomplete
        df.loc[0, "team_id"] = 100
        df.loc[0, "player_id"] = 9                 # crosser = scorer
        actions, _ = gs_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True, home_team_start_left_extratime=True
        )
        assert len(actions) == 2
        # row 0: the cross (freekick_crossed), result unchanged (fail for cross_outcome != C)
        assert actions.iloc[0]["type_id"] == spadlconfig.actiontype_id["freekick_crossed"]
        assert actions.iloc[0]["result_id"] == spadlconfig.result_id["fail"]
        # row 1: synthesized shot_freekick + success by the crosser
        assert actions.iloc[1]["type_id"] == spadlconfig.actiontype_id["shot_freekick"]
        assert actions.iloc[1]["result_id"] == spadlconfig.result_id["success"]
        assert actions.iloc[1]["player_id"] == 9
        assert actions.iloc[1]["team_id"] == 100
        # dense, contiguous action_id
        assert list(actions["action_id"]) == [0, 1]

    def test_cross_goal_with_foul_orders_shot_before_foul(self):
        # round-2 LOW edge: a parent that is BOTH a cross-goal AND a foul -> a .4 synthetic shot
        # AND a .5 synthetic foul. Proves the combined synthesis block composes both offsets.
        df = _df_minimal_pass()
        df.loc[0, "possession_event_type"] = "CR"
        df.loc[0, "set_piece_type"] = "O"
        df.loc[0, "shot_outcome_type"] = "G"
        df.loc[0, "team_id"] = 100
        df.loc[0, "player_id"] = 9
        df.loc[0, "foul_type"] = "I"
        df.loc[0, "final_foul_outcome_type"] = "Y"
        actions, _ = gs_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True, home_team_start_left_extratime=True
        )
        assert [actions.iloc[i]["type_id"] for i in range(len(actions))] == [
            spadlconfig.actiontype_id["cross"],   # parent (order 0)
            spadlconfig.actiontype_id["shot"],    # synthetic shot (.4)
            spadlconfig.actiontype_id["foul"],    # synthetic foul (.5)
        ]
        assert list(actions["action_id"]) == [0, 1, 2]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/spadl/test_gradientsports.py::TestGradientsportsCrossGoal -q`
Expected: FAIL (only 1 action; no synthetic shot).

- [ ] **Step 3: Implement**

Replace the foul-synthesis block (the `# Synthesize additional row` part, `if synth_mask.any(): ...`) with a combined builder that appends both foul and cross-goal synthetic rows before a single sort+renumber. After the in-place foul conversion (`if in_place_mask.any(): ...`), use:

```python
    # ---- Combined synthesis: foul rows + cross-goal shot rows -> one 0.5-offset insert ----
    # Load-bearing invariant (round-2 #4): synthesis masks (`synth_mask`, `cg_mask`) are computed on
    # `events` and applied to `actions` rows positionally — they MUST still be 1:1 and index-aligned
    # here (exclusion reset_index'd events; actions built 1:1; _derive_end_coordinates is in-place).
    # A future row-op inserted in this span would silently mis-target synthesis; fail loud instead.
    assert len(actions) == len(events), (
        f"synthesis precondition violated: {len(actions)} actions != {len(events)} events"
    )
    actions["__order__"] = np.arange(len(actions), dtype="float64")
    synth_parts: list[pd.DataFrame] = []

    # Foul rows (parent already a real action).
    if synth_mask.any():
        foul_rows = actions.loc[synth_mask].copy()
        foul_rows["type_id"] = foul_id
        foul_rows["result_id"] = foul_result_full[synth_mask]
        foul_rows["bodypart_id"] = spadlconfig.bodypart_id["foot"]
        foul_rows["__order__"] = np.arange(len(actions))[synth_mask] + 0.5
        synth_parts.append(foul_rows)

    # Component 2: cross-goal -> keep the cross, synthesize a shot by the crosser. `events` is
    # still 1:1 with `actions` here (synthesis hasn't reordered yet).
    cg_mask = (
        (events["possession_event_type"].fillna("").to_numpy() == "CR")
        & (events["shot_outcome_type"].fillna("").to_numpy() == "G")
    )
    if cg_mask.any():
        sp_cg = events["set_piece_type"].fillna("").to_numpy()[cg_mask]
        cg_type = np.select(
            [sp_cg == "F", sp_cg == "P"],
            [at_ids2["shot_freekick"], at_ids2["shot_penalty"]],
            default=at_ids2["shot"],
        ).astype("int64")
        shot_rows = actions.loc[cg_mask].copy()
        shot_rows["type_id"] = cg_type
        shot_rows["result_id"] = spadlconfig.result_id["success"]
        # parent (cross) keeps player_id/team_id/coords/bodypart = crosser; shot inherits them.
        shot_rows["__order__"] = np.arange(len(actions))[cg_mask] + 0.4  # before a same-parent foul (.5)
        synth_parts.append(shot_rows)

    if synth_parts:
        actions = pd.concat([actions, *synth_parts], ignore_index=True)
        actions = actions.sort_values("__order__").reset_index(drop=True)
        actions["action_id"] = np.arange(len(actions), dtype="int64")
    actions = actions.drop(columns="__order__")
```

Add, near the top of `convert_to_actions` where `type_id_arr`/`result_id_arr` are produced (after `at_ids`/`rs_ids` are in scope inside the dispatch helper they're local; in `convert_to_actions` get them once):
```python
    at_ids2 = spadlconfig.actiontype_id  # local alias for synthesis (mirror of dispatch's at_ids)
```
(Place this alias just before the synthesis block.)

> **Note:** `_derive_end_coordinates` (line ~490) still runs BEFORE this block — unchanged — so end-coords are derived on the 1:1 frame; synthesized rows inherit the parent's `end_x/end_y` via `.copy()`, which is correct (shot end = cross origin; documented xG-origin caveat in spec).

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/spadl/test_gradientsports.py::TestGradientsportsCrossGoal "tests/spadl/test_gradientsports.py::TestGradientsportsFoul" -q`
Expected: PASS (cross-goal + existing foul-synthesis still green).

- [ ] **Step 5: Commit**

```bash
git add silly_kicks/spadl/gradientsports.py tests/spadl/test_gradientsports.py
git commit -F .git/COMMIT_T4.txt   # "feat(spadl): GS CR+G cross-goal -> cross + synthetic shot"
```

---

## Task 5: Component 3 — VAEP labels count own goals (extract helpers, drop "shot" gate)

**Files:**
- Modify: `silly_kicks/vaep/labels.py` (all ~6 sites with the `goal`/`owngoal` predicate)
- Test: `tests/vaep/test_labels_owngoal.py` (new) or extend existing labels test

- [ ] **Step 1: Write failing test**

```python
import pandas as pd
from silly_kicks.vaep import labels
from silly_kicks.spadl import config as spadl

def _two_team_actions():
    # team 1 acts; a bad_touch+owngoal by team 1 => team 2 scores. window picks it up.
    return pd.DataFrame({
        "game_id": [1, 1],
        "period_id": [1, 1],
        "team_id": [1, 1],
        "type_name": ["pass", "bad_touch"],
        "result_id": [spadl.result_id["success"], spadl.result_id["owngoal"]],
    })

def test_owngoal_bad_touch_counts_as_concede_for_acting_team():
    a = _two_team_actions()
    conceded = labels.concedes(a, nr_actions=2)
    assert bool(conceded.iloc[0]) is True   # the pass's team concedes (own goal next)

def test_cross_fail_then_shot_success_credits_goal():
    # round-1 #7 adjacency: a failed cross immediately followed by a synthetic shot+success
    # (the cross-goal shape) must still credit the goal to the acting team — the preceding
    # fail must not cancel/pervert the goal's VAEP credit.
    a = pd.DataFrame({
        "game_id": [1, 1],
        "period_id": [1, 1],
        "team_id": [1, 1],
        "type_name": ["freekick_crossed", "shot_freekick"],
        "type_id": [spadl.actiontype_id["freekick_crossed"], spadl.actiontype_id["shot_freekick"]],
        "result_id": [spadl.result_id["fail"], spadl.result_id["success"]],
    })
    scored = labels.scores(a, nr_actions=2)
    assert bool(scored.iloc[0]) is True     # the failed cross's team scores (synthetic shot next)

def test_no_shot_gated_owngoal_predicate_survives():
    # The bug being fixed IS a copy-pasted shot-gated owngoal predicate. Prove no copy survives:
    # zero lines combine str.contains("shot") with the owngoal result. (Meta-test — survives
    # future refactors; catches a missed site like the originally-overlooked 339-340 / 416-417.)
    import re
    from pathlib import Path
    src = Path(labels.__file__).read_text(encoding="utf-8")
    offenders = [
        ln for ln in src.splitlines()
        if 'str.contains("shot")' in ln and "owngoal" in ln
    ]
    assert offenders == [], f"shot-gated owngoal predicate(s) still present: {offenders}"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/vaep/test_labels_owngoal.py -q`
Expected: FAIL (bad_touch owngoal not detected — `type_name.contains("shot")` gate).

- [ ] **Step 3: Implement**

At module scope in `silly_kicks/vaep/labels.py` (after imports). The helpers stay on **`type_name`** to
preserve the existing input contract (every site reads `actions["type_name"]` today; switching to
`type_id` would `KeyError` a valid `type_name`-only caller — round-2 #3), but use an **explicit name
set** instead of the fragile `str.contains("shot")` substring (round-2 #F):
```python
_SHOT_TYPE_NAMES = frozenset({"shot", "shot_penalty", "shot_freekick"})

def _is_goal(actions: pd.DataFrame) -> pd.Series:
    return actions["type_name"].isin(_SHOT_TYPE_NAMES) & (actions["result_id"] == spadl.result_id["success"])

def _is_owngoal(actions: pd.DataFrame) -> pd.Series:
    # owngoal result is unambiguous — no action-type gate (own goals are bad_touch, not shots).
    return actions["result_id"] == spadl.result_id["owngoal"]
```
(`spadl` is the imported alias for `silly_kicks.spadl.config` — the current code references `spadl.result_id[...]`.)

Route **EVERY** occurrence through the helpers — do NOT enumerate by line number. There are **8** owngoal
pairs (verified: 111-112, 188-189, 205-206, 225-226, 245-246, 290-291, **339-340**, **416-417**) plus a
goals-only predicate at **510** (and unrelated keeper predicates at 542/573 — leave those). Find them by
grep, not by the stale "~6 sites" list (missing 339-340 / 416-417 keeps the bug in two functions):
```bash
grep -nE 'result_id\["(owngoal|success)"\]' silly_kicks/vaep/labels.py
```
Replace each `goal = actions["type_name"].str.contains("shot") & (...success)` with `goal = _is_goal(actions)`
(incl. the goals-only line 510, renaming its local to match), and each
`owngoal = actions["type_name"].str.contains("shot") & (...owngoal)` with `owngoal = _is_owngoal(actions)`.

- [ ] **Step 4: Run to verify pass + no regression**

Run: `.venv/Scripts/python.exe -m pytest tests/vaep/ -q`
Expected: the new test PASSES; pre-existing labels tests still pass (goal detection unchanged for shots).

- [ ] **Step 5: Commit**

```bash
git add silly_kicks/vaep/labels.py tests/vaep/test_labels_owngoal.py
git commit -F .git/COMMIT_T5.txt   # "fix(vaep): count own goals in scores/concedes/xG by result (helper extraction)"
```

---

## Task 6: Realistic + composition fixtures (loader `nonEvent` + synthetic_match.json)

**Files:**
- Modify: `scripts/_loader_pining.py` `_GS_EVENT_FIELD_MAP` / `_gs_flatten_events`
- Modify: `tests/datasets/gradientsports/synthetic_match.json`
- Test: `tests/spadl/test_gradientsports.py`

- [ ] **Step 1: Add `nonEvent` to the GS flatten map**

In `scripts/_loader_pining.py`, add to `_GS_EVENT_FIELD_MAP`:
```python
    "nonEvent": "nonEvent",
```
(So the converter input carries `nonEvent` and Component 4 fires through the loader path.)

- [ ] **Step 2: Extend the committed fixture (production-shaped, synthetic ids)**

Edit `tests/datasets/gradientsports/synthetic_match.json` — add `nonEvent: false` to all existing events' `possessionEvents`, and append four events (synthetic ids, modeled on real shapes):
1. `RE`+`G` own goal: `possessionEvents.possessionEventType="RE"`, `shotOutcomeType="G"`, `nonEvent=false`, `rebounderPlayerId=<conceding player>`, `shooterPlayerId=null`; `gameEvents.playerId=<same>`, `teamId=<conceding>`, ball in that team's own half.
2. `CR`+`G` cross-goal: `possessionEventType="CR"`, `setpieceType="F"`, `shotOutcomeType="G"`, `crossOutcomeType="I"`, `nonEvent=false`, `crosserPlayerId=<crosser>`, `gameEvents.playerId=<crosser>`.
3. Disallowed goal: `possessionEventType="SH"`, `shotOutcomeType="G"`, `nonEvent=true`.
4. A foul event (existing pattern) — for the composition test.

- [ ] **Step 3: Write the realistic + composition tests**

```python
class TestGradientsportsGoalCaptureRealistic:
    def test_owngoal_crossgoal_captured_disallowed_excluded(self):
        events = _load_synthetic_events()
        actions, report = gs_mod.convert_to_actions(
            events, home_team_id=100, home_team_start_left=True, home_team_start_left_extratime=True
        )
        og = (actions["result_id"] == spadlconfig.result_id["owngoal"]).sum()
        assert og == 1                                   # the RE+G own goal
        shots = actions[actions["type_id"].isin(
            [spadlconfig.actiontype_id[n] for n in ("shot", "shot_freekick", "shot_penalty")])]
        assert (shots["result_id"] == spadlconfig.result_id["success"]).sum() >= 1   # synth cross-goal shot
        assert report.excluded_counts.get("nonEvent") == 1   # disallowed SH+G dropped

    def test_composition_dense_action_ids_and_order(self):
        events = _load_synthetic_events()
        actions, _ = gs_mod.convert_to_actions(
            events, home_team_id=100, home_team_start_left=True, home_team_start_left_extratime=True
        )
        assert list(actions["action_id"]) == list(range(len(actions)))   # dense + contiguous
        # round-2 LOW: the cross-goal's synthetic shot sorts immediately AFTER its cross. Find the
        # cross-goal cross row (the freekick_crossed/cross whose source CR carried shotOutcome G) and
        # assert the next row is a shot-class action by the same player (locks the .4 offset).
        shot_ids = {spadlconfig.actiontype_id[n] for n in ("shot", "shot_freekick", "shot_penalty")}
        cross_ids = {spadlconfig.actiontype_id[n] for n in ("cross", "freekick_crossed", "corner_crossed")}
        idxs = [i for i in range(len(actions) - 1)
                if actions.iloc[i]["type_id"] in cross_ids
                and actions.iloc[i + 1]["type_id"] in shot_ids
                and actions.iloc[i + 1]["player_id"] == actions.iloc[i]["player_id"]]
        assert idxs, "expected a cross immediately followed by its synthetic shot (same player)"
```

- [ ] **Step 4: Run + RED-prove**

Run: `.venv/Scripts/python.exe -m pytest tests/spadl/test_gradientsports.py -q`
Expected: PASS. To RED-prove the realistic test catches the bug, temporarily `git stash push -- silly_kicks/spadl/gradientsports.py`, run the realistic test (expect FAIL), then `git stash pop`.

- [ ] **Step 5: Commit**

```bash
git add scripts/_loader_pining.py tests/datasets/gradientsports/synthetic_match.json tests/spadl/test_gradientsports.py
git commit -F .git/COMMIT_T6.txt   # "test(spadl): GS goal-capture realistic + composition fixtures (nonEvent/RE+G/CR+G)"
```

---

## Task 7: Golden re-baseline + owner-gated catalog scoreline e2e

**Files:**
- Modify: any committed VAEP-label goldens that move (enumerate first)
- Create: `tests/spadl/test_gradientsports_scoreline_e2e.py`

- [ ] **Step 1: Enumerate moving goldens**

Run: `.venv/Scripts/python.exe -m pytest tests/ -m "not e2e" -q` and capture failures. Separately, grep committed fixtures for own goals that now count:
Run: `.venv/Scripts/python.exe -m pytest tests/vaep tests/spadl tests/atomic -q -m "not e2e"`
List every failing golden assertion that shifted due to own goals now counting (expected: providers whose committed fixtures contain `result_id==owngoal`).

- [ ] **Step 2: Regenerate the goldens + document deltas**

For each moving golden, regenerate the expected values and record the delta (`+N scores / +M concedes`) in the PR body. Do NOT blanket-accept — confirm each delta equals the own-goal count in that fixture (a delta larger than the own-goal count is a real regression to investigate).

- [ ] **Step 3: Owner-gated catalog scoreline guard**

```python
import os
import pytest

import importlib.util, json, tempfile, warnings
from pathlib import Path

# Real-goal counts (nonEvent=False G) per match — the over-count cases + their scorelines.
EXPECTED_REAL_GOALS = {"3853": 3, "10503": 3, "3855": 3}   # extend with a few normal scorelines too
# The 3 confirmed own goals: match -> (conceding teamId, OG scorer playerId = rebounderPlayerId).
KNOWN_OWN_GOALS = {"10503": (364, 11856), "3853": (374, 4002), "3855": (368, 4602)}

def _load_loader():
    spec = importlib.util.spec_from_file_location(
        "_loader_pining", str(Path(__file__).parents[2] / "scripts" / "_loader_pining.py")
    )
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

@pytest.mark.e2e
@pytest.mark.skipif(not os.environ.get("PINING_FOR_THE_DATA_TOKEN"), reason="owner-tier GS data")
def test_real_goal_population_matches_scorelines():
    L = _load_loader(); tok, base = L._resolve_token(None), L._base_url()
    for mid, expected in EXPECTED_REAL_GOALS.items():
        with tempfile.TemporaryDirectory() as tmp:
            p = L._download_to_temp("gradientsports", mid, "events", tok, base, Path(tmp))
            events = json.load(open(p, encoding="utf-8"))
        real = sum(1 for ev in events
                   if (ev.get("possessionEvents") or {}).get("shotOutcomeType") == "G"
                   and not (ev.get("possessionEvents") or {}).get("nonEvent", False))
        assert real == expected, f"g{mid}: {real} real goals, expected {expected}"

@pytest.mark.e2e
@pytest.mark.skipif(not os.environ.get("PINING_FOR_THE_DATA_TOKEN"), reason="owner-tier GS data")
def test_real_own_goals_captured_through_converter_no_tripwire_warn():
    # round-2 #2: the ONLY validation of the tripwire inequality on REAL own-goal coordinates.
    # Run the full converter (via load_matches) on each OG match; the OG must be bad_touch+owngoal on
    # the conceding team/scorer, and NO tripwire WARN may fire (a backwards inequality would revert all
    # 3 real OGs to keeper_save/fail + warn). Filter to the tripwire message so unrelated UserWarnings
    # (e.g. ET filtering) don't false-fail.
    from silly_kicks.spadl import config as spadlconfig
    L = _load_loader()
    for mid, (team, scorer) in KNOWN_OWN_GOALS.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _prov, _m, actions, _frames, _home = next(iter(L.load_matches(
                providers=["gradientsports"], match_ids={"gradientsports": [mid]}, tracking_limit=1)))
        tripped = [w for w in caught if "own-goal" in str(w.message) and "attacking half" in str(w.message)]
        assert not tripped, f"g{mid}: tripwire reverted a real own goal — inequality likely backwards: " \
                            f"{[str(w.message) for w in tripped]}"
        og = actions[(actions["result_id"] == spadlconfig.result_id["owngoal"])
                     & (actions["team_id"] == team)]
        assert len(og) == 1, f"g{mid}: expected exactly 1 owngoal for team {team}, got {len(og)}"
        assert (og["type_id"] == spadlconfig.actiontype_id["bad_touch"]).all()
        assert (og["player_id"] == scorer).all()
```

(Both run only where the owner token + data are present; public CI skips them.)

- [ ] **Step 4: Run (non-e2e)**

Run: `.venv/Scripts/python.exe -m pytest tests/ -m "not e2e" -q` → all green.

- [ ] **Step 5: Run the validating e2e — REQUIRED GATE (round-2 #1)**

This is the **only** real-coordinate validation of the Task-3 tripwire inequality; the non-e2e suite
cannot catch a backwards inequality (synthetic tripwire tests pass either way). Do NOT skip it.

```bash
.venv/Scripts/python.exe -m pytest tests/spadl/test_gradientsports_scoreline_e2e.py -m e2e -q
```
- If `PINING_FOR_THE_DATA_TOKEN` is set: **require green** before proceeding. A failure of
  `test_real_own_goals_captured_through_converter_no_tripwire_warn` means the tripwire reverted real own
  goals → the inequality in Task 3 is backwards; fix it (flip to `start_x < field_length/2`) and re-run.
  Also confirm the three `KNOWN_OWN_GOALS` scorer ids (11856/4002/4602) against the data (round-2 #3) —
  a wrong id false-fails here.
- If the token is **NOT** available to the implementing agent: **HALT and flag the maintainer** — the
  tripwire inequality is unvalidated against real data and the maintainer MUST run `pytest -m e2e` with
  the token before merge. Do not mark this task complete on a silent skip.

- [ ] **Step 6: Commit**

```bash
git add tests/ silly_kicks/
git commit -F .git/COMMIT_T7.txt   # "test(vaep): re-baseline own-goal-affected goldens + owner-gated GS scoreline e2e"
```

---

## Task 8: ADR + CHANGELOG + version + final verification

**Files:**
- Create: `docs/superpowers/adrs/ADR-0NN-owngoals-counted-in-vaep-labels.md`
- Modify: `CHANGELOG.md`, `TODO.md`, `pyproject.toml`, `silly_kicks/__init__.py`, `uv.lock`

- [ ] **Step 1: Write the ADR** — "Own goals counted in VAEP scores/concedes/xG labels by result, independent of action type." Record: context (the `"shot"` gate undercount, all providers), decision (result-based owngoal detection via `_is_owngoal`), consequences (label distributions shift; VAEP models retrain consideration; golden re-baseline). Use `docs/superpowers/adrs/ADR-TEMPLATE.md`; next free ADR number.

- [ ] **Step 2: Version bump** — reconcile against `origin/main` (per the version-bump checklist; no reserved numbers). This is a cross-cutting behavior change → **minor**. If `main` is at 4.12.2, use **4.13.0** unless taken. Bump pyproject + `__init__` + CHANGELOG (with the Hyrum + nonEvent-contract flags) + TODO; then `uv lock`.

- [ ] **Step 3: Full local CI + e2e gate** — `ruff check silly_kicks/ tests/ scripts/`; `ruff format --check`; `pyright silly_kicks/`; `pytest tests/ -m "not e2e"`. All clean. **AND** confirm the Task-7 Step-5 e2e gate was satisfied: either `pytest -m e2e` ran green with the owner token, or the maintainer was explicitly flagged to run it before merge (round-2 #1). Do not release on an unvalidated tripwire.

- [ ] **Step 4: Commit** (bundle the spec + plan + ADR into the feature commit per the no-standalone-doc-commits rule).

```bash
git add -A
git commit -F .git/COMMIT_T8.txt   # "feat(spadl,vaep): GS own/cross goal capture + nonEvent exclusion + owngoal labels -- silly-kicks 4.13.0 (ADR-0NN)"
```

---

## Plan review dispositions (lakehouse round 1)

| # | Concern | Disposition |
|---|---------|-------------|
| 1 | Task 5 missed 2 of 8 owngoal sites (339-340, 416-417) | **Fixed (verified 8 pairs + line 510).** Task 5 now routes **all** occurrences by grep (not line numbers), routes the goals-only 510 through `_is_goal`, and adds a **guard meta-test** asserting zero surviving `str.contains("shot") … owngoal` lines. |
| 2 | Tripwire inequality never validated on real OG coords (synthetic test circular; Task-7 e2e didn't run the converter) | **Fixed.** New owner-gated e2e `test_real_own_goals_captured_through_converter_no_tripwire_warn` **runs `convert_to_actions`** on g3853/g10503/g3855, asserting each OG = `bad_touch`+`owngoal` with **no tripwire WARN**. Task-3 note corrected: synthetic tests are NOT ground truth; this e2e is the inequality gate. |
| 3 | `type_name`→`type_id` silent contract change | **Fixed.** Helpers stay on `type_name` (preserves contract) using an explicit name-set `_SHOT_TYPE_NAMES` (also resolves #F substring fragility). |
| 4 | Cross-goal synthesis assumes events↔actions 1:1 with no assertion | **Fixed.** `assert len(actions) == len(events)` immediately before the combined synthesis block. |
| 5 | `nonEvent` `.astype(bool)` footgun ("false"→True inverts exclusion) | **Fixed.** Robust coercion: real-bool fast path; else only true-ish strings count (never inverts on "false"). |
| LOW | e2e match coverage thin | Expanded `EXPECTED_REAL_GOALS` to g3853/g10503/g3855 (+note to add normal scorelines). |
| LOW | composition order not asserted | Added relative-order assertion (synthetic shot immediately after its cross, same player). |
| LOW | cross-goal + same-parent foul edge | Added Task-4 test proving `.4` shot precedes `.5` foul. |

Confirmed-correct (not re-touched): atomic path untouched; Component-4 numbers; golden re-baseline delta guard; version 4.13.0 reconcile-against-main.

**Round 2 (approve-after-one-fix):**

| # | Concern | Disposition |
|---|---------|-------------|
| 1 | Validating e2e defined but never **run** by any step (#2 only nominally closed) | **Fixed.** Task 7 **Step 5** is now a required e2e gate: run `pytest -m e2e`; green-required if token set; **HALT + flag maintainer** if no token. Task 8 Step 3 re-asserts the gate before release. |
| 2 | `nonEvent` coercion: `np.bool_` slips `v is True` | **Fixed.** `_truthy` handles Python+numpy bool, strings, None/NaN. |
| 3 | `KNOWN_OWN_GOALS` scorer ids only checked by the (now-run) e2e | **Resolved by #1** — ids (11856/4002/4602) verified from the catalog investigation; the Step-5 e2e now CI-checks them where the token exists. |
| 4 | e2e assumes `load_matches` 5-tuple | **Verified correct** — `_loader_pining.py:145` yields `(provider, match_id, actions, frames, home)`. No change. |

## Notes for the executor

- **Frequent commits**: one per task (above). Per the repo's commit-sentinel rule, the maintainer creates the sentinel before each `git commit`; present the diff and HOLD.
- **Commit messages**: write each to `.git/COMMIT_T*.txt` and `git commit -F` (multiline messages via file, per repo convention).
- **Atomic mirror**: no code change (already result-based); the atomic suite in Task 6/7 guards no regression.
- **Lakehouse handoff**: after merge, send copy/paste context — the bronze→input mapping must surface `possessionEvents.nonEvent` or the absent-column warning fires and Component 4 is silently missed.

## Execution addendum (scope pulled in during implementation)

- **`is_synthetic` provenance column (maintainer pulled into scope during execution).** Beyond the four
  components above, an `is_synthetic` (bool) column was added to `GRADIENTSPORTS_SPADL_COLUMNS`, set
  `True` on the converter-injected rows (the Task-4 cross-goal shot AND the existing synthesized foul
  rows) and `False` on real 1:1 rows. This resolves the round-2 spec concern E (a consumer de-duping on
  the shared `original_event_id` would otherwise collapse/drop a synthesized row). Tested in
  `TestGradientsportsSyntheticProvenance` (schema membership + default-False + shot/foul flagged True).
  See the spec's round-2 dispositions + ADR-018 (Neutral).
- **Single commit, not per-task.** Per maintainer workflow, the whole feature ships as one branch / one
  commit / one PR (the per-task `git commit` steps above were folded into a single 4.13.0 feature commit
  bundling code + tests + spec + plan + ADR).
