# TF-54b Per-action Goal Resolution — Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax for tracking. This repo forbids
> micro-commits — do **not** commit per step. Build the whole change to a fully-tested, coherent state,
> then stop at the single **human-approval commit gate** (Task 8). Nothing is committed without Karsten's
> explicit approval for that specific commit.

**Goal:** Make `compute_territorial_defense` resolve the attacked goal correctly on per-action-LTR SB360
freeze-frames (it currently scores ~0 on its only target data), via per-frame convention-based resolution,
dual-mode by an explicit `frame_convention` parameter.

**Architecture:** The compute owns the frame convention through a single `goal_map_for(game, period,
acting, opponent) → GoalMap` factory. `per_action_ltr` (default, SB360) builds a per-frame `GoalMap` from
the ADR-028 convention (the acting team attacks x=105 → defends 0; opponent defends 105). `match_ltr`
(continuous-tracking-derived, home-attacks-right) returns a single per-match `resolve_defended_goals` map —
today's behavior. `classify_arm_a_domain` and the arms stay convention-agnostic; they call the factory.

**Tech Stack:** Python, pandas, numpy; `silly_kicks.tracking.GoalMap` / `resolve_defended_goals`;
`silly_kicks.spadl.config`; the existing `compute_threat_pc` seam (unchanged).

**Spec:** `docs/superpowers/specs/2026-09-10-tf54b-per-action-goal-resolution-design.md` (Approved — read it
first; this plan argues from it).

## Global Constraints

- **No micro-commits; one commit for the whole coherent change.** The commit also carries the already-green
  review-round (IMPL-01/02/04, L7/L8/L9) and the SkillCorner corpus registration — this fix is what unblocks
  committing all of it. **Stop at Task 8's human-approval gate; do not `git add`/commit/push before it.**
- **Preserve existing contracts:** ADR-055 (`GoalMap` type + `attacked_goal` real opponent-lookup, canonical
  string keys), ADR-042 (dropped-and-counted conservation identities), ADR-027 (honest-NaN, never fabricated
  0), ADR-063 (threat suppression is a Tier-1 zero-velocity lift), ADR-019 (`id_compat` for every id compare).
- **`match_ltr` output MUST be byte-identical to today's per-match behavior** (Task 4 golden).
- **`resolve_defended_goals` itself is NOT modified** — reused verbatim inside the `match_ltr` factory only.
- **Arm signatures (`arm_a_threat_suppressed`, `arm_b_threat_suppressed`) do NOT change** — they already take
  `goal_map`; the compute passes a per-frame map.
- **CI-faithful gates:** `python -m pytest tests/ -m "not e2e" -q -p no:randomly`; ruff check + format on
  `silly_kicks/ tests/ scripts/`; CI-faithful pyright (py3.12 venv) adds 0 errors.

---

## File Structure

- **`silly_kicks/territorial_defense/_engine.py`** — add `action_ltr_goal_map`; change
  `classify_arm_a_domain` to take a `goal_map_for` factory (replacing `goal_map`).
- **`silly_kicks/territorial_defense/_compute.py`** — add `frame_convention` param + the `goal_map_for`
  factory; thread it into classify + `_score_arm_a` + `_score_arm_b`; drop the per-match
  `resolve_defended_goals` call on the `per_action_ltr` branch.
- **`scripts/validate_territorial_defense.py`** — the dose battery builds per-frame `action_ltr_goal_map`s
  (SB360-only; hard-codes the `per_action_ltr` factory).
- **`tests/territorial_defense/_fixtures.py`** — add `make_per_action_ltr_fixture`.
- **`tests/territorial_defense/test_compute.py` / `test_arms.py` / `test_engine.py`** — new fixture; the
  two-sided regression; the `match_ltr` byte-identity golden; convention arguments on existing tests.
- **`tests/invariants/glossary_emitted_columns.py`, `tests/test_scale_guards.py`** — thread `match_ltr` on
  the match-oriented fixtures.
- **`tests/sb360/_registry.py` + `_entries/_boundary.py`** — re-record the `compute_territorial_defense`
  verdict (now scores).
- **`docs/superpowers/adrs/ADR-091-tf54b-per-action-goal-resolution.md`** (new), **`CLAUDE.md`**,
  **`CHANGELOG.md`**.

## Interfaces

- Produces: `action_ltr_goal_map(game_id, period_id, *, acting_team_id, opponent_team_id) -> GoalMap`
  (in `_engine.py`, public-in-package).
- Changed: `classify_arm_a_domain(domain, frames, *, params=_DEFAULT_PARAMS, visible_area=None,
  goal_map_for, groups=None) -> pd.DataFrame` (`goal_map` → `goal_map_for`).
- Changed: `compute_territorial_defense(actions, frames, *, xt, links=None, visible_area=None,
  frame_convention: Literal["per_action_ltr","match_ltr"]="per_action_ltr", params=_DEFAULT_PARAMS)`.
- Changed (internal): `_score_arm_a(classified, frames, *, xt, goal_map_for, params, groups=None)`,
  `_score_arm_b(actions, frames, domain, *, xt, goal_map_for, params, groups=None)`.
- Unchanged: `arm_a_threat_suppressed(..., goal_map, ...)`, `arm_b_threat_suppressed(..., goal_map, ...)`,
  `resolve_defended_goals`, `compute_threat_pc`, `TerritorialDefenseReport`, `TD_SAMPLE_COLUMNS`.

---

### Task 1: `action_ltr_goal_map` helper (engine)

**Files:**
- Modify: `silly_kicks/territorial_defense/_engine.py`
- Test: `tests/territorial_defense/test_engine.py`

**Interfaces:**
- Produces: `action_ltr_goal_map(game_id, period_id, *, acting_team_id, opponent_team_id) -> GoalMap`.
- Consumes: `silly_kicks.tracking.GoalMap`, `silly_kicks.id_compat.canonical_id`,
  `silly_kicks.spadl.config.field_length`, `types.MappingProxyType`.

- [ ] **Step 1: Write the failing test**

```python
# tests/territorial_defense/test_engine.py
from types import MappingProxyType  # noqa: F401  (only if needed)

def test_action_ltr_goal_map_resolves_both_ends():
    from silly_kicks.territorial_defense._engine import action_ltr_goal_map

    gm = action_ltr_goal_map(7, 1, acting_team_id=1, opponent_team_id=2)
    # acting team attacks x=105 (defends 0); opponent defends 105
    assert gm.attacked_goal(7, 1, 1, allow_guess=True) == 105.0   # acting attacks opponent's end
    assert gm.attacked_goal(7, 1, 2, allow_guess=True) == 0.0     # opponent attacks acting's end
    # dtype-agnostic keys (ADR-019/ADR-055 rule 2)
    assert gm.attacked_goal(7, 1, "1", allow_guess=True) == 105.0
```

- [ ] **Step 2: Run → FAIL** `pytest tests/territorial_defense/test_engine.py::test_action_ltr_goal_map_resolves_both_ends -q` — Expected: `ImportError`/`AttributeError`.

- [ ] **Step 3: Implement** — add to `_engine.py` (imports: `from types import MappingProxyType`, `from silly_kicks.spadl import config as spadlconfig`, and `GoalMap` from `silly_kicks.tracking` — `resolve_defended_goals` is already imported there):

```python
def action_ltr_goal_map(game_id, period_id, *, acting_team_id, opponent_team_id) -> GoalMap:
    """Per-action-LTR goal map for ONE frame (ADR-028): the acting team of the frame's action attacks
    x=105 (defends 0); the opponent defends x=105. Correct by convention -- no GK, so it is FOV-proof
    where a per-frame ``resolve_defended_goals`` would fail on a keeperless freeze-frame. Keys canonical
    (ADR-055 rule 2). This resolves a DIFFERENT frame convention from ground truth than
    ``resolve_defended_goals`` (which estimates the match-oriented end from GK positions); it is not a
    fork -- both route through ``GoalMap`` and its real opponent-lookup ``attacked_goal``.
    """
    g, p = canonical_id(game_id), canonical_id(period_id)
    fl = float(spadlconfig.field_length)
    resolved = {
        (g, p, canonical_id(acting_team_id)): 0.0,
        (g, p, canonical_id(opponent_team_id)): fl,
    }
    return GoalMap(MappingProxyType(resolved), MappingProxyType({}), frozenset())
```

- [ ] **Step 4: Run → PASS.**

- [ ] **Step 5: Guard the not-a-fork consumption seam** — add a test that `action_ltr_goal_map` returns the
  same `GoalMap` type `resolve_defended_goals` returns, so downstream `attacked_goal`/`get` behave identically:

```python
def test_action_ltr_goal_map_is_a_goalmap():
    from silly_kicks.tracking import GoalMap
    from silly_kicks.territorial_defense._engine import action_ltr_goal_map
    assert isinstance(action_ltr_goal_map(1, 1, acting_team_id=1, opponent_team_id=2), GoalMap)
```

---

### Task 2: `goal_map_for` factory + convention-agnostic classify + dispatch (compute/engine)

**Files:**
- Modify: `silly_kicks/territorial_defense/_engine.py` (`classify_arm_a_domain`, `_classify_one`)
- Modify: `silly_kicks/territorial_defense/_compute.py` (`compute_territorial_defense`, `_score_arm_a`, `_score_arm_b`)
- Test: `tests/territorial_defense/test_compute.py`

**Interfaces:**
- Consumes: `action_ltr_goal_map` (Task 1); `resolve_defended_goals`; `GoalEndUnresolvedError`.
- Produces: `frame_convention` param on `compute_territorial_defense`; `goal_map_for` factory threaded to
  classify + arms.

- [ ] **Step 1: Write the two-sided regression test (RED)** — the guard that was missing (spec §8.3). This
  imports `make_per_action_ltr_fixture` from **Task 5** (build Task 5 first, per the recommended order
  5→1→2→…):

```python
# tests/territorial_defense/test_compute.py
def test_per_action_frames_score_under_convention():
    """Mixed-acting-team per-action frames: match_ltr (per-match GK mean) is bimodal -> 0 scored;
    per_action_ltr resolves per-frame -> scores. Two-sided: fails on the bug, passes on the fix."""
    from tests.territorial_defense._fixtures import make_per_action_ltr_fixture, make_fitted_xt
    actions, frames = make_per_action_ltr_fixture()
    xt = make_fitted_xt()

    s_match, _ = compute_territorial_defense(actions, frames, xt=xt, frame_convention="match_ltr")
    s_pa, _ = compute_territorial_defense(actions, frames, xt=xt, frame_convention="per_action_ltr")

    a_match = pd.to_numeric(s_match["a_threat_suppressed"], errors="coerce")
    a_pa = pd.to_numeric(s_pa["a_threat_suppressed"], errors="coerce")
    assert a_match.notna().sum() == 0        # per-match resolution: all unresolved_geometry
    assert a_pa.notna().sum() >= 1           # per-frame convention: scores
```

- [ ] **Step 2: Run → FAIL** (`frame_convention` unknown kwarg). If the fixture isn't built yet, do **Task
  5** first, then return here.

- [ ] **Step 3: Change `classify_arm_a_domain` to take `goal_map_for`** (`_engine.py`): replace the
  `goal_map=None` param with `goal_map_for` (required). In `_classify_one`, replace the `goal_map` argument
  with `goal_map_for` and build the per-row map:

```python
def _classify_one(row, groups, polys, params, goal_map_for) -> tuple[str, int]:
    fr = groups.get(row.game_id, row.period_id, row.frame_id)
    if len(fr) == 0:
        return NO_ACTOR, -1
    ...  # is_actor handling unchanged
    gm = goal_map_for(row.game_id, row.period_id, row.defending_team_id, row.attacking_team_id)
    if gm.attacked_goal(row.game_id, row.period_id, row.attacking_team_id, allow_guess=True) is None:
        return UNRESOLVED_GEOMETRY, -1
    ...  # rest unchanged (no_defenders / removal_undersupported / fov gates)
```

Update `classify_arm_a_domain`'s signature (`goal_map=None` → `goal_map_for`), drop the internal
`resolve_defended_goals(frames)` fallback (the factory owns resolution), and pass `goal_map_for` into
`_classify_one`.

- [ ] **Step 4: Add `frame_convention` + the factory in `compute_territorial_defense`** (`_compute.py`):

```python
from typing import Literal
FRAME_CONVENTIONS = ("per_action_ltr", "match_ltr")

def compute_territorial_defense(actions, frames, *, xt, links=None, visible_area=None,
                                frame_convention: Literal["per_action_ltr", "match_ltr"] = "per_action_ltr",
                                params=_DEFAULT_PARAMS):
    require_fitted_xt(xt, caller="compute_territorial_defense")
    if frame_convention not in FRAME_CONVENTIONS:
        raise ValueError(f"frame_convention={frame_convention!r} not in {FRAME_CONVENTIONS}.")
    groups = group_rows(frames, _FRAME_KEYS)
    if frame_convention == "per_action_ltr":
        def goal_map_for(game_id, period_id, acting_team_id, opponent_team_id):
            return action_ltr_goal_map(game_id, period_id,
                                       acting_team_id=acting_team_id, opponent_team_id=opponent_team_id)
    else:
        _match_map = resolve_defended_goals(frames)
        def goal_map_for(game_id, period_id, acting_team_id, opponent_team_id):
            return _match_map
    domain = select_arm_a_domain(actions, params=params)
    classified = classify_arm_a_domain(domain, frames, params=params, visible_area=visible_area,
                                       goal_map_for=goal_map_for, groups=groups)
    arm_a, classified = _score_arm_a(classified, frames, xt=xt, goal_map_for=goal_map_for, params=params, groups=groups)
    arm_b, arm_b_census = _score_arm_b(actions, frames, domain, xt=xt, goal_map_for=goal_map_for, params=params, groups=groups)
    samples = _assemble_samples(classified, arm_a, arm_b)
    report = build_report(classified["td_source"], params=params, n_frames_in=len(domain), arm_b=arm_b_census)
    return samples, report
```

Remove the old `goal_map = resolve_defended_goals(frames)` line and the `import` of it if now unused on the
per-action path (it stays imported for the `match_ltr` branch).

- [ ] **Step 5: Thread the factory into the arms** (`_compute.py`): `_score_arm_a`/`_score_arm_b` take
  `goal_map_for` (replacing `goal_map`); in each per-frame loop build `gm = goal_map_for(game, period,
  acting, opponent)` and pass `goal_map=gm` to the arm. For `_score_arm_a`: `acting=cand.defending_team_id,
  opponent=cand.attacking_team_id`. For `_score_arm_b`: `acting=opp_row.team_id,
  opponent=defending_team_id`, and use the same `gm` for the `attacked = gm.attacked_goal(opp_row.game_id,
  opp_row.period_id, opp_row.team_id, allow_guess=True)` target-reflection check (logic unchanged).

- [ ] **Step 5b: Migrate the 5 DIRECT callers of the renamed seams (GOAL-PLAN-01).** Renaming
  `goal_map` → `goal_map_for` (required) on `classify_arm_a_domain`/`_score_arm_a`/`_score_arm_b` breaks 5
  direct call sites that the compute-level dispatch does NOT cover (they need a *factory*, not
  `frame_convention`). Migrate each — wrap the existing per-match map as a factory `goal_map_for=lambda *a:
  <map>` (or, where a per-action test is intended, `action_ltr_goal_map`). **`groups=` is the existing L7
  param (default `None`), NOT a new arg** (GOAL-PLAN-04) — these callers keep the default, so only the
  `goal_map` kwarg changes:

  | Site | Current | Migrate to |
  |---|---|---|
  | `tests/territorial_defense/test_engine.py:124` `classify_arm_a_domain(dom, frames, visible_area=...)` (relied on the removed internal `resolve_defended_goals` default) | *(no goal_map passed)* | add `goal_map_for=lambda *a: resolve_defended_goals(frames)` (import `resolve_defended_goals` from `silly_kicks.tracking`) — preserves the conservation test on its match-oriented frames |
  | `tests/test_scale_guards.py:634` `classify_arm_a_domain(domain, frames, goal_map=gm)` | `goal_map=gm` | `goal_map_for=lambda *a: gm` |
  | `tests/test_scale_guards.py:651` `classify_arm_a_domain(domain, frames, goal_map=gm)` | `goal_map=gm` | `goal_map_for=lambda *a: gm` |
  | `tests/test_scale_guards.py:653` `C._score_arm_a(classified, frames, xt=xt, goal_map=gm, ...)` | `goal_map=gm` | `goal_map_for=lambda *a: gm` |
  | `tests/test_scale_guards.py:720` `C._score_arm_b(actions, frames, domain, xt=xt, goal_map=gm, ...)` | `goal_map=gm` | `goal_map_for=lambda *a: gm` |

  The scale guards stub the arms (`arm_a_threat_suppressed → 0.0`) and count `rows_scanned` — convention-
  invariant, so wrapping the per-match map preserves their intent. Run `pytest tests/territorial_defense/test_engine.py
  tests/test_scale_guards.py -k "arm_a or arm_b or classify or territorial" -q` → green.

- [ ] **Step 6: Run → PASS** the two-sided regression + the existing `test_compute` shape test (the latter
  under `per_action_ltr` once its fixture is per-action-LTR (Task 5), or with `frame_convention="match_ltr"`
  while it still uses the match-oriented fixture).

- [ ] **Step 7: Verify conservation still holds** — assert in the regression test that
  `report.n_frames_scored + sum(report.drop_reasons.values()) == report.n_frames_in` under both conventions.

---

### Task 3: Driver dose-battery per-frame maps

**Files:**
- Modify: `scripts/validate_territorial_defense.py`
- Test: `tests/scripts/test_validate_territorial_defense.py`

- [ ] **Step 1: Update `_measure_match`** — replace `gm = resolve_defended_goals(frames)` with a per-frame
  factory (SB360-only, hard-coded `per_action_ltr`): import `action_ltr_goal_map` from
  `silly_kicks.territorial_defense._engine`. Call `classify_arm_a_domain(..., goal_map_for=goal_map_for)`.
  In the per-scored-frame loop, build `gm = action_ltr_goal_map(cand.game_id, cand.period_id,
  acting_team_id=cand.defending_team_id, opponent_team_id=cand.attacking_team_id)` and use it as the
  `goal_map` in `resp_kw` (feeding `_threat`, `_dose_unit_vector`'s fallback, and the inline
  `arm_a_threat_suppressed`).

- [ ] **Step 2: Run** `pytest tests/scripts/test_validate_territorial_defense.py -q` — the existing driver
  tests use `make_e2e_fixture` (match-oriented). Since the driver is now `per_action_ltr`, either switch the
  driver test's `_sb_item()` to `make_per_action_ltr_fixture`, or assert on the battery mechanics that are
  convention-invariant. Update the test to the per-action-LTR fixture so `_measure_match` scores (mirrors
  real SB360). Expected: green.

---

### Task 4: `match_ltr` byte-identity golden + convention arguments on existing tests

**Files:**
- Modify: `tests/territorial_defense/test_compute.py`, `tests/invariants/glossary_emitted_columns.py`,
  `tests/test_scale_guards.py`

- [ ] **Step 1: Byte-identity golden** — add `test_match_ltr_matches_per_match_resolution`: on the
  match-oriented fixture, `compute_territorial_defense(..., frame_convention="match_ltr")` equals a
  reference computation that threads `resolve_defended_goals(frames)` directly (the pre-fix behavior). Assert
  `pd.testing.assert_frame_equal` on the two `samples`. This proves `match_ltr` reproduces today's output.

- [ ] **Step 2: Thread `match_ltr`** where a match-oriented fixture is used and the convention matters:
  `glossary_emitted_columns.py` (column set is convention-invariant, but pass `match_ltr` so it scores),
  `test_scale_guards.py::test_compute_territorial_defense_is_subquadratic` (its `make_td_scaling_fixture` is
  match-oriented → pass `frame_convention="match_ltr"`; the growth counter is convention-invariant). Run each
  → green.

- [ ] **Step 3:** Run the full `tests/territorial_defense/` + the two touched invariants/scale tests → green.

---

### Task 5: Per-action-LTR fixture + regression fixture

**Files:**
- Modify: `tests/territorial_defense/_fixtures.py`

**Interfaces:**
- Produces: `make_per_action_ltr_fixture() -> tuple[pd.DataFrame, pd.DataFrame]`.

- [ ] **Step 1: Build `make_per_action_ltr_fixture`** — each frame in *its acting team's* LTR. Reuse
  `_rich_rows()` for D's interception frames (team-1's LTR: team-1 keeper low x). For the team-2 pass frame,
  **point-reflect** the rich geometry (`x → 105 - x`, `y → 68 - y`) so it is in team-2's LTR (team-2 keeper
  low x, D at high x). Keep the actions table identical to `make_e2e_fixture` (3 team-1 interceptions + 1
  team-2 pass; `frame_id == action_id`). Helper:

```python
def _reflect_rows(rows):
    out = []
    for r in rows:
        r = dict(r)
        r["x"] = 105.0 - r["x"]
        r["y"] = 68.0 - r["y"]
        out.append(r)
    return out

def make_per_action_ltr_fixture():
    """Per-action-LTR (SB360-shaped): D's 3 interception frames in team-1's LTR; the team-2 pass frame
    point-reflected into team-2's LTR. Exercises the per_action_ltr default (the convention real SB360 uses)."""
    parts = []
    for fid in range(3):                       # team-1 (D) actions -> team-1 LTR
        fr = pd.DataFrame(_rich_rows()); fr["frame_id"] = fid
        parts.append(_typed(fr))
    fr3 = pd.DataFrame(_reflect_rows(_rich_rows())); fr3["frame_id"] = 3   # team-2 pass -> team-2 LTR
    parts.append(_typed(fr3))
    frames = pd.concat(parts, ignore_index=True)
    actions = <same actions table as make_e2e_fixture>   # 3 interceptions (team 1) + 1 pass (team 2)
    return actions, frames
```

- [ ] **Step 2: Non-vacuity assertions** (spec R1) — add
  `test_per_action_ltr_fixture_scores_both_arms`: under `per_action_ltr`, `a_frames_scored >= 1` with a
  positive `a_threat_suppressed`, AND `b_frames_scored >= 1` with a positive `b_threat_suppressed` (removing
  the contesting defender raises threat). If Arm B does not score with the naive reflection, adjust the pass
  `end`/frame geometry so the reflected pass lands in D's hull and D is the contesting defender — mirror the
  discipline in `make_e2e_fixture`'s docstring. Run → green.

- [ ] **Step 3:** Re-point the `test_compute.py` e2e shape/conservation test and the glossary/`_measure_match`
  driver test to this fixture under the default `per_action_ltr` where they should reflect the real target;
  keep the `match_ltr` golden on the match-oriented fixture. Run `tests/territorial_defense/` → green.

---

### Task 6: SB360 audit re-adjudication

**Files:**
- Modify: `tests/sb360/_entries/_boundary.py`, `tests/sb360/_registry.py`, and re-record via
  `tests/sb360/_regenerate.py` + `_adjudicate.py`.

- [ ] **Step 1: Re-derive the observation** — with the fix, `compute_territorial_defense` on the SB360 anchor
  frames now **scores** instead of all-NaN. Run the SB360 audit to see the new machine observation
  (`all_nan` → `differs`/`partial_nan` for the Tier-1 lift columns).

- [ ] **Step 2: Re-adjudicate the human verdict** — the threat arms are ADR-063 Tier-1 zero-velocity lifts, so
  the verdict is `differs_by_design` (not `silent_degrade`); `verdict_provenance="substantive"`. Re-record.

- [ ] **Step 3: Confirm invariants** — `NOT_EXERCISED_BUDGET` and the `_EXPECTED_DARK_COLUMNS`
  `b_attribution_slippage` entry still hold (slippage stays honest-NaN on anonymous SB360 even though the
  threat arms now score — the defenders remain anonymous). Run `tests/sb360/` → green.

---

### Task 7: ADR-091 + CLAUDE.md + CHANGELOG

**Files:**
- Create: `docs/superpowers/adrs/ADR-091-tf54b-per-action-goal-resolution.md`
- Modify: `CLAUDE.md`, `CHANGELOG.md`

- [ ] **Step 1: ADR-091** — decision record: the per-action-LTR goal-resolution defect, the `frame_convention`
  dual-mode contract (`per_action_ltr` default, `match_ltr` for continuous-tracking-derived frames), the
  `action_ltr_goal_map`-is-not-an-ADR-055-fork argument, and the fixture-masking testing lesson. Use
  `docs/superpowers/adrs/ADR-TEMPLATE.md`.

- [ ] **Step 2: CLAUDE.md** — update the `territorial_defense` bullet: the SB360 default is `per_action_ltr`
  (per-frame convention resolution), `match_ltr` reuses `resolve_defended_goals`; add the fixture-masking
  lesson to the testing-discipline section ("a fixture in the wrong coordinate convention passes tests while
  the real path is broken").

- [ ] **Step 3: CHANGELOG** — extend the 4.112.0 entry (or a new sub-bullet): the SB360 goal-resolution fix
  (0 → scores on real SB360), `frame_convention` param, no re-materialize / no VAEP retrain.

---

### Task 8: Full validation + real-data re-validation + COMMIT GATE (STOP)

**Files:** none (validation only).

- [ ] **Step 1: Full CI-faithful suite** — `python -m pytest tests/ -m "not e2e" -q -p no:randomly` → 0
  failures. `python -m ruff check silly_kicks/ tests/ scripts/` + `--format --check` → clean. CI-faithful
  pyright (py3.12 venv) → 0 new errors.

- [ ] **Step 2: Real-WC2022 re-validation (scratch, NOT committed)** — run the actual
  `compute_territorial_defense` (default `per_action_ltr`) on a real public WC2022 match via statsbombpy
  (the smoke path): confirm the arms **score** end-to-end (not 0) and the report conserves. Report the
  scored counts + delta distribution.

- [ ] **Step 3: STOP at the human-approval commit gate.** Present the full diff — this fix **plus** the
  already-green review-round (IMPL-01/02/04, L7/L8/L9) and the SkillCorner corpus registration — as ONE
  coherent commit. Do **not** `git add`/commit/push. Wait for Karsten's explicit approval for that specific
  commit. Only then commit, and only then close draft PR #235 / open the new PR.

---

## Self-Review

- **Spec coverage:** §5.1 (dual-mode) → Task 2 Step 4; §5.2 (`action_ltr_goal_map`) → Task 1; §5.3
  (`goal_map_for` factory, convention-agnostic classify/arms) → Task 2 Steps 3–5; §5.4 (driver) → Task 3;
  §8.1 (per-action fixture) → Task 5; §8.2 (`match_ltr` golden) → Task 4 Step 1; §8.3 (two-sided regression)
  → Task 2 Step 1; §9 (SB360 audit) → Task 6; §12 (lesson) + ADR-091 → Task 7; §14 (validation) → Task 8.
- **Type consistency:** `goal_map_for(game, period, acting, opponent)` signature identical at every call
  site (classify, `_score_arm_a`, `_score_arm_b`, driver); `action_ltr_goal_map` keyword-only `acting_team_id`
  / `opponent_team_id` consistent between Task 1 and its callers.
- **No placeholders** except the two explicitly-marked "same actions table as `make_e2e_fixture`" and "ADR
  body per template" — both point to a concrete existing artifact.
- **Ordering note:** Task 5 (fixture) is a dependency of Task 2 Step 1 and Task 3 Step 2 — build the fixture
  (Task 5 Step 1) before running those RED tests, or inline a minimal fixture in Task 2 Step 1 and replace it.
  Recommended execution order: **5 → 1 → 2 → 3 → 4 → 6 → 7 → 8.**
