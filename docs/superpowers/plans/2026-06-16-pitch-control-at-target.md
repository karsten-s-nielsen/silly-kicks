# `pitch_control_at_target` — retire dead at-ball column, re-aim to destination (Implementation Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Retire the informationally-dead `pitch_control_at_ball__<method>` and replace it with a live
`pitch_control_at_target__<method>` sampled at the action destination, with the mandatory ADR-028
re-projection that the near-ball degeneracy had been masking.

**Architecture:** A localized change to `pitch_control_at_action` (the only producer; `add_pitch_control` +
`pitch_control_xfns` delegate to it) + its atomic mirror. The per-frame PPCF surface and `PitchControlCache`
key are UNCHANGED — only the query point moves (start→end) and is re-projected into the frame's
absolute-frame convention via the ADR-028 helpers. Other PPCF consumers
(obso/cover_shadows/gk_influence/player_influence/space_creation) are untouched.

**Tech stack:** Python 3.10–3.14, pandas/numpy, pytest, ruff, pyright. Spec:
`docs/superpowers/specs/2026-06-16-pitch-control-at-target-design.md`. **Decision:** ADR-032 (new).
**Base:** `main` @ v4.30.0 → **target 4.31.0**.

**Owner-policy adaptations (override the skill's per-task commits):** feature branch
`pr-s96-pitch-control-at-target` off `main`, **no worktree**. **NO intermediate commits** — every task ends
in a local verify checkpoint + `git add` (staging); the **single commit** is the last task, after
`/final-review` + explicit owner approval. RED-first = run + observe failure + capture output, NOT a commit.
Never tag before CI green.

---

## File structure

| Path | Responsibility | Action |
|------|----------------|--------|
| `silly_kicks/tracking/features.py` | `pitch_control_at_action` re-aim to `end_*` + ADR-028 reprojection; rename col/`__name__`/docstrings → `at_target`; `add_pitch_control`/`pitch_control_xfns`/`pitch_control_default_xfns` follow | Modify |
| `silly_kicks/atomic/tracking/features.py` | atomic `pitch_control_at_action` synth `end_x=x+dx, end_y=y+dy` (not just `x→start_x`); rename docstrings/`__name__` | Modify |
| `silly_kicks/calibration/_features.py` | rename the 3 `pitch_control_at_ball__*` feature names → `pitch_control_at_target__*` (lines ~69-71) | Modify |
| `tests/tracking/pitch_control/test_pitch_control_at_target.py` | NEW — ground-truth correctness (asymmetric+extreme, home+away) + mirror-invariance + re-aim sanity | Create |
| `tests/tracking/test_aggregator_column_liveness.py` | DELETE the `add_pitch_control` `STRUCTURAL_CONSTANTS` entry + `test_pitch_control_at_ball_near_ball_degeneracy`; ADD a hard off-ball-destination precondition guard for `add_pitch_control` | Modify |
| `tests/tracking/pitch_control/test_action_coupled.py` | column-name assertion `at_ball`→`at_target` (`:117`) | Modify |
| `tests/tracking/pitch_control/test_atomic_pitch_control.py` | column-name assertions `at_ball`→`at_target` (`:101`, `:128`) | Modify |
| `docs/superpowers/adrs/ADR-032-pitch-control-at-target.md` | NEW ADR | Create |
| `CLAUDE.md` | update PR-S31/TF-7 mention + the liveness-gate `STRUCTURAL_CONSTANTS` example (retired→live) | Modify |
| `pyproject.toml`/`__init__.py`/`CHANGELOG.md`/`TODO.md`/`uv.lock` | version 4.31.0 | Modify |

---

## Phase 1 — standard `pitch_control_at_action` re-aim + ADR-028 reprojection (TDD)

### Task 1.1: Ground-truth correctness test (THE correctness core) — author + observe RED

**Files:** Create `tests/tracking/pitch_control/test_pitch_control_at_target.py`.

- [ ] **Step 1: Write the test.** Two minimal frames + actions. ASYMMETRIC + EXTREME geometry so the cell and
      its 180° reflection differ by the full PPCF range (≈1.0 vs ≈0.0). The away case is the one that pins the
      reprojection DIRECTION.

```python
"""Ground-truth correctness for pitch_control_at_target (ADR-032).

Mirror-symmetry is necessary but NOT sufficient (a symmetric-wrong / cancelling-double-flip projection passes
a symmetry-only test). So we pin an ABSOLUTE, hand-computable value on an ASYMMETRIC frame: the action's
destination cell is acting-team-controlled (~1.0) while its 180-degree absolute-frame reflection is
opponent-controlled (~0.0). For the AWAY action the correct projection lands on the ~1.0 cell and a
wrong-direction flip lands on the ~0.0 reflection -> the test is RED on a broken projection. EXTREME
separation (acting players adjacent to the destination, opponents ~a pitch-length away) keeps the asymptote
robust against the Spearman sigmoid/reaction-time params.
"""

import numpy as np
import pandas as pd

from silly_kicks.tracking.features import pitch_control_at_action

_HOME, _AWAY = "H", "A"


def _frame(frame_id, *, controllers_xy, opponents_xy, controller_team, opp_team):
    """One canonical (home-attacks-right) frame: `controller_team` clustered adjacent to controllers_xy,
    `opp_team` clustered adjacent to opponents_xy, plus a ball off in midfield. Players static (vx=vy=0)."""
    rows = []
    for k, (cx, cy) in enumerate(controllers_xy):
        rows.append(dict(game_id=1, period_id=1, frame_id=frame_id, time_seconds=10.0,
                         player_id=f"{controller_team}{k}", team_id=controller_team, is_ball=False,
                         x=cx, y=cy, vx=0.0, vy=0.0,
                         team_attacking_direction="ltr" if controller_team == _HOME else "rtl"))
    for k, (ox, oy) in enumerate(opponents_xy):
        rows.append(dict(game_id=1, period_id=1, frame_id=frame_id, time_seconds=10.0,
                         player_id=f"{opp_team}{k}", team_id=opp_team, is_ball=False,
                         x=ox, y=oy, vx=0.0, vy=0.0,
                         team_attacking_direction="ltr" if opp_team == _HOME else "rtl"))
    rows.append(dict(game_id=1, period_id=1, frame_id=frame_id, time_seconds=10.0, player_id=None,
                     team_id=None, is_ball=True, x=52.5, y=34.0, vx=0.0, vy=0.0,
                     team_attacking_direction=None))
    return pd.DataFrame(rows)


def _action(team_id, frame_id, end_x, end_y):
    return pd.DataFrame([dict(game_id=1, period_id=1, action_id=1, frame_id=frame_id, time_seconds=10.0,
                              team_id=team_id, start_x=52.5, start_y=34.0, end_x=end_x, end_y=end_y)])


def test_home_action_destination_controlled_by_acting_team_reads_high():
    # HOME attacks right (ltr). Destination (80,20) is action-LTR == absolute (no flip). Home players adjacent
    # to (80,20); away cluster at the reflection (25,48). PPCF(home)@(80,20) ~ 1.0.
    fr = _frame(100, controllers_xy=[(80, 20), (79, 21), (81, 19)],
                opponents_xy=[(25, 48), (26, 47), (24, 49)], controller_team=_HOME, opp_team=_AWAY)
    s = pitch_control_at_action(_action(_HOME, 100, 80.0, 20.0), fr, method="spearman")
    assert s.iloc[0] > 0.9, f"home at_target should be ~1.0 (acting team controls destination), got {s.iloc[0]}"


def test_away_action_pins_reprojection_direction():
    # AWAY action: action-LTR destination (80,20) -> absolute-frame (105-80, 68-20)=(25,48). Correct projection
    # samples (25,48) where AWAY players are clustered -> PPCF(away) ~ 1.0. A wrong-direction (no-flip)
    # projection samples (80,20) where HOME is clustered -> PPCF(away) ~ 0.0 -> RED. Asymmetric+extreme: the two
    # cells differ by the full range, so this cannot pass on a symmetric-wrong projection.
    fr = _frame(200, controllers_xy=[(25, 48), (26, 47), (24, 49)],
                opponents_xy=[(80, 20), (79, 21), (81, 19)], controller_team=_AWAY, opp_team=_HOME)
    s = pitch_control_at_action(_action(_AWAY, 200, 80.0, 20.0), fr, method="spearman")
    assert s.iloc[0] > 0.9, (
        f"away at_target should be ~1.0 after the ADR-028 reprojection lands on the absolute-frame cell "
        f"(25,48); got {s.iloc[0]} (a ~0.0 value means the projection direction is wrong / not applied)"
    )
```

- [ ] **Step 2: Add the multi-action vectorization guard (review P1 — the biggest gap).** The real path
      processes a DataFrame of MANY actions in one call: `flip = acting_team_attacks_rtl(actions, frames)`
      and `q = reproject_to_action_ltr(...)` are per-action, then sampled positionally. A misalignment (a
      differently-indexed `flip` Series, applying the away-flip to the wrong row, label-vs-positional drift)
      is INVISIBLE to single-action tests — one row can't be mis-assigned. So assert a mixed batch in ONE call:

```python
def test_multi_action_mixed_home_away_each_row_correct():
    # ONE frame, ONE pitch_control_at_action call, TWO actions: a HOME action (dest (80,20), no flip) and an
    # AWAY action (action-LTR dest (80,20) -> absolute (25,48), flips). Home cluster @ (80,20), away cluster @
    # (25,48). Both rows must read ~1.0 on THEIR correct cell. A flip mis-aligned across rows reads ~0.0 for at
    # least one (home would sample (25,48)=away, or away would sample (80,20)=home) -> RED. This is the only
    # test exercising the per-action flip vectorization the production path uses.
    # The base frame ALREADY has HOME@(80,20) + AWAY@(25,48) + exactly one ball — exactly what the two actions
    # need (home samples (80,20)→~1.0; away flips (80,20)→(25,48)→~1.0). No cluster re-add / concat (that
    # duplicated the ball — review Q1).
    fr = _frame(400, controllers_xy=[(80, 20), (79, 21), (81, 19)],
                opponents_xy=[(25, 48), (26, 47), (24, 49)], controller_team=_HOME, opp_team=_AWAY)
    actions = pd.concat([_action(_HOME, 400, 80.0, 20.0), _action(_AWAY, 400, 80.0, 20.0)], ignore_index=True)
    actions["action_id"] = [1, 2]
    s = pitch_control_at_action(actions, fr, method="spearman")
    assert s.iloc[0] > 0.9, f"home row (no flip) should read ~1.0 @ (80,20); got {s.iloc[0]}"
    assert s.iloc[1] > 0.9, f"away row (flipped) should read ~1.0 @ (25,48); got {s.iloc[1]} (flip mis-aligned?)"
```

- [ ] **Step 3: Run → RED.** `python -m pytest tests/tracking/pitch_control/test_pitch_control_at_target.py -v`
      Expected: the home, away, and multi-action tests FAIL (current code samples `start_x/start_y` = the ball
      at (52.5,34) → the 0.5 fallback; and even after a bare re-aim, without the reprojection the away rows
      land on (80,20)≈0.0). Capture the output.

### Task 1.2: Mirror-invariance companion test

- [ ] **Step 1: Add** to the same file — same physical situation mirrored home↔away yields the SAME `at_target`
      (necessary-not-sufficient symmetry guard, ADR-028 `test_action_ltr_mirror_invariance` pattern):

```python
def test_mirror_invariance_home_vs_away():
    # Physically identical: acting team controls its destination in both. Home dest (80,20); away dest mirrored.
    fh = _frame(300, controllers_xy=[(80, 20)], opponents_xy=[(25, 48)], controller_team=_HOME, opp_team=_AWAY)
    fa = _frame(301, controllers_xy=[(25, 48)], opponents_xy=[(80, 20)], controller_team=_AWAY, opp_team=_HOME)
    sh = pitch_control_at_action(_action(_HOME, 300, 80.0, 20.0), fh, method="spearman").iloc[0]
    sa = pitch_control_at_action(_action(_AWAY, 301, 80.0, 20.0), fa, method="spearman").iloc[0]
    assert abs(sh - sa) < 0.05, f"mirrored home/away at_target must agree: {sh} vs {sa}"
```

- [ ] **Step 2: Run → RED** (same reasons as 1.1). Capture.

### Task 1.3: Implement the re-aim + reprojection; rename → `at_target`

**Files:** Modify `silly_kicks/tracking/features.py` (`pitch_control_at_action` ~1859-1940 + the 3 docstrings +
`pitch_control_xfns.__name__`).

- [ ] **Step 1: Rename the column** everywhere in this function group: `col_name = f"pitch_control_at_target__{method}"`
      (was `at_ball`); the `pitch_control_at_action`/`add_pitch_control` docstrings; and
      `_pc_helper.__name__ = f"pitch_control_at_target__{method}"` in `pitch_control_xfns`.
- [ ] **Step 2: Replace the sample point + add the ADR-028 reprojection.** Precompute the per-action flip +
      reprojected query BEFORE the loop; sample at the reprojected destination:

```python
    from ._action_orientation import acting_team_attacks_rtl, reproject_to_action_ltr

    # Re-aim from the ball (degenerate ~0.5) to the action DESTINATION, re-projected into the frame's
    # absolute-frame convention so away-team actions sample the correct cell (ADR-028 / ADR-032). The
    # cached per-frame surface stays absolute-frame (cache key unchanged) -- only the query point flips.
    #
    # INVOLUTION (do NOT "simplify" this): reproject_to_action_ltr is the 180-degree reflection
    # (x->105-x, y->68-y) on rtl rows; the reflection is its own inverse. The action's end point is in
    # action-LTR; applying the SAME per-action flip lands it on the absolute-frame cell the surface is keyed
    # in. Applying a "to_ltr"-named helper to ltr points to obtain ABSOLUTE reads backwards but is correct --
    # the away ground-truth + multi-action tests go RED if anyone replaces this with an identity/no-op.
    flip = acting_team_attacks_rtl(actions, frames)
    q = actions[["end_x", "end_y"]].rename(columns={"end_x": "_qx", "end_y": "_qy"}).copy()
    q = reproject_to_action_ltr(q, flip, x_cols=["_qx"], y_cols=["_qy"])
    qx = q["_qx"].to_numpy(dtype="float64")  # positional order == actions row order (== flip/loop order)
    qy = q["_qy"].to_numpy(dtype="float64")
    # ... existing loop, but replace the start_x/start_y block with: ...
        if np.isnan(qx[i]) or np.isnan(qy[i]):
            continue
        results[i] = surface.at_point(float(qx[i]), float(qy[i]))
```

- [ ] **Step 3: Run → GREEN.** Tasks 1.1 + 1.2 pass. Also run the existing
      `tests/tracking/pitch_control/` suite to confirm no collateral break. **Stage.**

---

## Phase 2 — atomic mirror

### Task 2.1: Atomic `pitch_control_at_action` synthesizes `end_*`; rename; parity

**Files:** Modify `silly_kicks/atomic/tracking/features.py` (`pitch_control_at_action` ~800-822,
`add_pitch_control` docstring ~833, `atomic_pitch_control_xfns.__name__` ~860).

- [ ] **Step 1: Synthesize end coordinates** (the standard now samples `end_x/end_y`; atomic has only
      `x,y,dx,dy`). Mirror `_structural_pass_atomic_endpoints`:

```python
    adapted = actions.copy()
    adapted["start_x"] = adapted["x"]
    adapted["start_y"] = adapted["y"]
    adapted["end_x"] = adapted["x"] + adapted["dx"]
    adapted["end_y"] = adapted["y"] + adapted["dy"]
    return _std_pc(adapted, frames, links=links, method=method)
```

- [ ] **Step 2: Rename** the `add_pitch_control` docstring + `_pc_helper.__name__` in `atomic_pitch_control_xfns`
      → `pitch_control_at_target__{method}`.
- [ ] **Step 3: Add atomic-parity assertion** to `test_pitch_control_at_target.py`: an atomic action with
      `x,y,dx,dy` matching a standard action's `start/end` yields the same `at_target` value. Run → GREEN. **Stage.**

---

## Phase 3 — retire the dead column + harden liveness + rename consumers

### Task 3.1: Liveness gate — retire the structural constant, add the hard precondition

**Files:** Modify `tests/tracking/test_aggregator_column_liveness.py`.

- [ ] **Step 1: DELETE** the `"add_pitch_control": {...}` entry from `STRUCTURAL_CONSTANTS` (the dict becomes
      `{}`; `test_meta_structural_constants_are_wired` still holds — empty ⊆ ENTRIES) and DELETE
      `test_pitch_control_at_ball_near_ball_degeneracy`.
- [ ] **Step 2: Confirm + pin the fixture has off-ball destinations.** Inspect the liveness `_frames()` /
      action fixture; verify `add_pitch_control` now emits a non-NaN, **non-constant**
      `pitch_control_at_target__spearman` across the 5 windows. Add a hard precondition test (do NOT leave to
      chance — review A):

```python
def test_pitch_control_at_target_fixture_has_offball_destinations():
    """Precondition for the add_pitch_control liveness teeth: the fixture MUST have >=K actions whose
    destination is off-ball, else at_target is ~0.5 everywhere and the non-constant check passes weakly."""
    import numpy as np
    actions, frames = _actions(), _frames()  # the gate's own fixture builders (this file; make_actions/make_frames)
    offball = 0
    for _, a in actions.iterrows():
        fr = frames[(frames.period_id == a.period_id) & (frames.frame_id == a.frame_id) & frames.is_ball]
        if fr.empty or np.isnan(a.end_x):
            continue
        if np.hypot(a.end_x - fr.x.iloc[0], a.end_y - fr.y.iloc[0]) > 10.0:  # R = 10 m
            offball += 1
    assert offball >= 2, f"liveness fixture has only {offball} off-ball-destination actions; gate would lose teeth"
```

      (`_actions()`/`_frames()` are the gate's existing fixture builders in this file, delegating to the
      file-local `make_actions()`/`make_frames()`.) The 5-window fixture's actions DO carry `end_x/end_y`
      (e.g. start (60,30) → end (80,40), ~30 m off the ball). **Confirm ≥2 destinations clear R=10 m against
      that window's ball; if not, extend the fixture with an explicit off-ball-destination action** (document
      the addition). Then the standard liveness non-constant check applies to `pitch_control_at_target__spearman`.

### Task 3.2: Rename the calibration feature names

**Files:** Modify `silly_kicks/calibration/_features.py` (~69-71).

- [ ] **Step 1:** Rename the 3 entries `pitch_control_at_ball__{spearman,fernandez_bornn,voronoi}` →
      `pitch_control_at_target__...`. `git grep pitch_control_at_ball silly_kicks/ scripts/` → confirm zero
      remaining references in non-test code. Run `tests/calibration/` (non-e2e) → green. **Stage.**

### Task 3.3: Update column-name assertions in the pitch-control tests

**Files:** `git grep -n pitch_control_at_ball tests/` is the AUTHORITY (review P3) — update EVERY hard-coded
reference it returns, not just the two known files (`test_action_coupled.py:117`,
`test_atomic_pitch_control.py:101,128`).

- [ ] **Step 1:** Run `git grep -n pitch_control_at_ball tests/`. Replace every hard-coded
      `pitch_control_at_ball__*` literal with `pitch_control_at_target__*`. Then re-run the grep → **zero
      remaining** (the rename is complete here, NOT discovered later in Task 4.1's suite run). Note: the
      auto-enumerating gates (dup-action_id, id-dtype, nan-safety, public-API examples) enumerate by
      *function name* (`add_pitch_control` / `pitch_control_xfns`) and call the function — they do NOT
      hard-code the column string, so they pick up the rename automatically and need no edit. Only a
      literal-string reference needs editing; the grep finds all of those. Run the touched files → green. **Stage.**

---

## Phase 4 — full local gate

### Task 4.1: Full verification

- [ ] **Step 1:** `ruff format --check . && ruff check .` → clean.
- [ ] **Step 2:** `python -m pyright` (BARE — full scope incl `tests/`; the PR-S95 CI-miss lesson) → 0 errors.
- [ ] **Step 3:** `python -m pytest tests/ -m "not e2e and not slow" --benchmark-skip -q` (whole `tests/`,
      background per the >30s rule) → all pass. This is a **confirmation** that Task 3.3's grep caught every
      hard-coded `at_ball` reference and the auto-enumerating gates (which key on the function name, not the
      column string) pick up the rename — NOT the place to discover a missed literal (that's Task 3.3's job).

---

## Phase 5 — ADR + docs + version + handoff + commit

### Task 5.1: ADR-032

**Files:** Create `docs/superpowers/adrs/ADR-032-pitch-control-at-target.md`.

- [ ] Record: retire-and-replace; Option A (re-aim to destination) over the rejected potential-control model
      variant; the ball-travel-time rationale; the ADR-028 reprojection the degeneracy was masking + the
      ground-truth (asymmetric/extreme) test that pins the involution direction (not just symmetry); the
      in-place + shots (target-cell contestation, kept, per-type) semantics; retrain trigger framed as a
      lakehouse column-lifecycle migration (AC + DEFCON), atomic with the pin bump, batched with the Metrica
      recompute, A/B-validated.

### Task 5.2: CLAUDE.md

- [ ] Update the TF-7/PR-S31 line (the column is now `at_target`) and the liveness-gate convention paragraph
      (the `pitch_control_at_ball__spearman` `STRUCTURAL_CONSTANTS` example was RETIRED — the column is live;
      cite ADR-032). Note C4 count stays 28.

### Task 5.3: Version bump 4.31.0 (5-file gate)

**Files:** `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, `TODO.md`, `uv.lock`.

- [ ] Bump `4.30.0 → 4.31.0`; CHANGELOG entry (retire `at_ball` → `at_target`, ADR-028 reprojection fix,
      retrain trigger, breaking column rename); **TODO** — remove the now-shipped
      `pitch_control_at_ball__spearman` redesign bullet from "### Blocked or Deferred" (completed items are
      deleted, not annotated); `uv lock` (or hand-edit the silly-kicks version line — empty-extra-free, so
      the version line is the only change). Verify all five agree.

### Task 5.4: `/final-review`

- [ ] Run `/final-review` (code + docs + C4 drift). C4 is expected no-op (no new aggregator/container; count
      28) — regenerate `docs/c4/architecture.*` and confirm no diff beyond any wording. Address findings.

### Task 5.5: Lakehouse-migration handoff (copy/paste — NOT a silly-kicks TODO)

- [ ] Draft the copy/paste handoff: the `pitch_control_at_ball__*` → `pitch_control_at_target__*` rename is a
      full ADR-013-style column-lifecycle MIGRATION across AC (`schema.py`, `enrich.py`, `action_context.py`,
      `oracle_map.py`, `tracking_context.py`) **and DEFCON** (`defcon_lite*.py` + the schema-parity tests):
      bronze drop+add migration SQL + runner, dbt rename, Lakebase reshape, HF republish, DEFCON-parity tests,
      a forced AC recompute (dead-0.5 → live + away-team correction). **4.31.0 is BREAKING — atomic with the
      migration, NOT a currency pin-bump** (running AC against 4.31.0 before the migration → KeyError). Batch
      the recompute with the parked Metrica y-fix; A/B-validate the new feature (Brier improves / no regress),
      don't blind-adopt.

### Task 5.6: Single commit (ONLY after explicit owner approval)

- [ ] `git add -A`; one commit (subject:
      `feat(tracking)!: pitch_control_at_target replaces the dead at-ball column + ADR-028 reprojection -- silly-kicks 4.31.0 (ADR-032, PR-S96)`)
      ending with the `Co-Authored-By` trailer. Do NOT tag. Wait for CI green (owner monitors), then tag `v4.31.0`.

---

## Self-review

- **Spec coverage:** retire-and-replace (1.3/3.1) · re-aim to destination (1.3) · ADR-028 reprojection (1.3 +
  the ground-truth test 1.1) · ground-truth asymmetric+extreme + home/away direction pin (1.1) · mirror
  companion (1.2) · hard liveness precondition (3.1) · shots kept/per-type (ADR 5.1 + docstring in 1.3) ·
  atomic mirror end-synth (2.1) · calibration rename (3.2) · migration handoff incl DEFCON + atomic-pin (5.5).
- **Placeholder scan:** test code + the implementation diff are concrete; the only "confirm/inspect" step
  (3.1 fixture) is a genuine verification with a hard fallback (extend the fixture), not a vague TODO.
- **Type consistency:** `acting_team_attacks_rtl(actions, frames) -> Series`, `reproject_to_action_ltr(df,
  flip_mask, *, x_cols, y_cols) -> df` used per their real signatures (`tracking/_action_orientation.py`);
  column string `pitch_control_at_target__<method>` consistent across standard + atomic + calibration + tests.
- **RED-first under single-commit:** Tasks 1.1/1.2 are authored + run-RED before 1.3 (evidence, not a commit);
  the away ground-truth case is RED specifically until the reprojection lands.
