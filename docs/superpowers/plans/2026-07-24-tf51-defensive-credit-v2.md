# TF-51 v2 — Defensive-credit refinements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Ship four refinements to the shipped v1 defensive-credit family (`silly_kicks/tracking/defensive_credit/`)
plus one bundled v1 bug fix, in one branch/PR.

**Architecture:** Refine in place; preserve the family's hexagonal port split (`_sizing`=value / `_resolution`=who /
`_chaining`=sequence / `_rules`=policy / `_orchestration`=wiring). One new descriptive feature
(`tracking/_press_commitment.py`) lives outside the family. No new `*_xfns`, no VAEP retrain.

**Tech Stack:** Python 3.10+, pandas, numpy, scipy; pytest; ruff + pyright (CI `lint`).

**Design source:** `docs/superpowers/specs/2026-07-24-tf51-defensive-credit-v2-design.md` (rev 3). Section refs
below (`spec §N`) point there for rationale — this plan is the *how*, the spec is the *why*. Read the spec's
Decisions log (§13) and Open Questions (§11) before starting.

**Plan-review status:** review-1 close-out applied (P1–P15, all verified against code) — P1 `_is_true` not
`is True`; P2 `._params` constants (no import cycle); P3 hand-built asymmetric xT grid + literal expectations
(no `rate_at`); P4 the `_cover_shadows` cone formula; P5 the resolver core returns a sorted frame; P6
`CreditRow` gains `resolution` + the ~14-site sweep + `anchor_actor`; P7 accurate gate coverage (atomic mirror
= purity-only); P8 pass-through mirror; P9 red-first perf guard; P10 candidate-gated precompute.
**Review-2 (final) close-out applied (Q1–Q12):** Q1 the exhaustive-`xt_pressing` test moved to Task 3b Step 6a
(RED before emit); Q2 `lane_blocker` explicitly BYPASSES `threshold_m` + after-refactor guard; Q3 build the
real short-circuit/`<NA>` conditions (not a forced signal); Q4 floor the shot-lane corridor (the pinch-to-0
would fall back to nearest on close blocks); Q5 scalar 2-D cross (np.cross 2-D is deprecated); Q6 name drift
→ `opponents_within`; Q7 vectorised lane geometry; Q8 skip precompute when the rule is disabled; Q9 the
grid-array-read fallback; Q10 rationale = single-source (underscore import is fine, ruff has no PLC/TID);
Q11 pass-through-mirror rationale; Q12 already closed. **Reviewer: "CLEAR TO EXECUTE … No further review round
needed." The plan is execution-ready.**

**Implementation order (pinned, spec §1):** B2 fix → Item 3 → Item 1 → Item 2 → Item 5. Item 3 first because it
removes `rule_failed_marking_through_ball`'s `_xt_at` calls, cleaning Item 1's seam.

**Commit policy (repo-specific — overrides the skill's per-task `git commit`):** work on a feature branch off
`main`; WIP commits on the branch are fine (the repo **squash-merges to one commit**). Do **NOT** push, open a
PR, tag, or bump the version without explicit owner approval. The plan **STOPS before the final commit**
(Task 7). Branch name: `tf51-defensive-credit-v2` (no version number — [[feedback_no_version_number_until_commit_prep]]).

**Standing test discipline:** run `python -m pytest tests/ -m "not e2e" -q` (the e2e in Task-6 is owner-gated on
`PINING_FOR_THE_DATA_TOKEN`). After **any** lint fix re-run the full trio `ruff check` + `ruff format --check` +
`pyright` ([[feedback_run_full_lint_after_any_lint_fix]]). Every mutate-→-RED step must be run and seen RED
before implementing ([[feedback_invariance_test_needs_discriminating_power]]).

---

## File Structure

| File | Change | Task |
|------|--------|------|
| `silly_kicks/tracking/defensive_credit/_chaining.py` | scope `recovery_after_pass` to `(game_id, period_id)` | 1 |
| `silly_kicks/tracking/_line_breaking.py` | extract `home_team_id`-free straddle core; re-point internals | 2 |
| `silly_kicks/tracking/defensive_credit/_orchestration.py` | precompute line-break signal on `RuleContext`; new `resolution` col | 2, 4 |
| `silly_kicks/tracking/defensive_credit/_rules.py` | swap through-ball gate; `sized_xt` call-sites; `lane_blocker` ask; `ANCHOR_TYPE_VALUES` | 2, 3, 4 |
| `silly_kicks/tracking/defensive_credit/_params.py` | remove `through_ball_delta_xt_min`; add `pressing_lens`, `shot_lane_cone_width_factor`; `SIZING_VALUES`, `RESOLUTION_VALUES` | 2, 3, 4 |
| `silly_kicks/tracking/defensive_credit/_sizing.py` | primitives-only `sized_xt(x, y, xt, *, pressing_lens)` | 3 |
| `silly_kicks/tracking/defensive_credit/_resolution.py` | `lane_blocker` mode; thin adapter over the lifted shared resolver | 2, 4, 5 |
| `silly_kicks/tracking/_opponent_resolution.py` (NEW) | shared `opponents_within(..., threshold_m, ...)` core (sorted within-threshold frame) | 5 |
| `silly_kicks/tracking/_velocity_availability.py` (NEW) | extracted `velocity_unavailable_by_design` | 5 |
| `silly_kicks/tracking/_das.py` | re-point to the extracted velocity-availability helper | 5 |
| `silly_kicks/tracking/_press_commitment.py` (NEW) | commitment primitive | 5 |
| `silly_kicks/tracking/features.py` | `add_press_commitment` aggregator | 5 |
| `silly_kicks/atomic/tracking/features.py` | thin `add_press_commitment` atomic mirror | 5 |
| `silly_kicks/feature_glossary.py` | 3 `press_commitment*` entries | 5 |
| `NOTICE` | practitioner-concept attribution string | 5 |
| `tests/tracking/…` | new/updated tests per task | all |
| `docs/c4/architecture.dsl` + `architecture.html` | re-derive count (+1 tracking, +1 atomic.tracking) | 7 |
| `CHANGELOG.md`, `CLAUDE.md`, `TODO.md`, `pyproject.toml`, `uv.lock`, `silly_kicks/__init__.py` | version/registers | 7 |

---

## Task 0: Branch

- [ ] **Step 1:** `git fetch origin && git checkout main && git reset --hard origin/main` (sync — a parallel
  session may have advanced `main`; [[feedback_session_start_env_sync]]).
- [ ] **Step 2:** `git checkout -b tf51-defensive-credit-v2`.
- [ ] **Step 3:** confirm baseline green: `python -m pytest tests/tracking/test_defensive_credit_orchestration.py tests/tracking/test_defensive_credit_chaining.py -q`. Expected: PASS.

---

## Task 1: B2 — `recovery_after_pass` game/period-boundary fix (spec §7)

**Files:** `silly_kicks/tracking/defensive_credit/_chaining.py`; `tests/tracking/test_defensive_credit_chaining.py`

- [ ] **Step 1 — Write two failing tests.** In `test_defensive_credit_chaining.py` add:

```python
def test_recovery_does_not_cross_game_boundary():
    # Game A: failed pass at the last action; Game B opens with an opponent action within max_actions.
    # The opponent action belongs to a DIFFERENT game -> must NOT count as a recovery.
    actions = pd.DataFrame({
        "game_id":   [1, 1, 2, 2],
        "period_id": [1, 1, 1, 1],
        "action_id": [0, 1, 0, 1],
        "team_id":   [10, 10, 20, 20],   # game 2 is a different match; team 20 is foreign
        # ... start_x/y, end_x/y, type_id (pass, fail), result_id per the fixture helper ...
    })
    # pass_idx = 1 (game 1's failed pass); rows 2,3 are game 2.
    assert recovery_after_pass(_as_actions(actions), pass_idx=1, max_actions=3) is None

def test_recovery_still_fires_within_same_game():   # NON-VACUITY (N1): the fix must not kill the rule
    actions = _single_game_failed_pass_then_opponent_regain()   # opponent regain at pass_idx+1, same game/period
    assert recovery_after_pass(actions, pass_idx=..., max_actions=3) is not None
```

  (Use the file's existing fixture builder; if none exposes multi-game rows, add a `_two_game_actions()` helper.)
- [ ] **Step 2 — Run, verify RED.** `python -m pytest tests/tracking/test_defensive_credit_chaining.py -k "boundary or still_fires" -q`. Expected: `test_recovery_does_not_cross_game_boundary` FAILS (v1 crosses the boundary); the still-fires test PASSES (guards against over-fixing).
- [ ] **Step 3 — Implement.** In `_chaining.py::recovery_after_pass`, scope the forward slice to the passer's game+period BEFORE the opponent search:

```python
def recovery_after_pass(actions, pass_idx, *, max_actions):
    anchor = actions.iloc[pass_idx]
    passer_team = anchor["team_id"]
    # why NOT possession-scoped (N1): a recovery is by definition a possession CHANGE
    # (add_possessions makes every team-change a boundary), so clamping to the passer's
    # possession_id would make the opponent-search vacuous and this rule fire never.
    fwd = actions.iloc[pass_idx + 1 : pass_idx + 1 + max_actions]
    fwd = fwd[(fwd["game_id"] == anchor["game_id"]) & (fwd["period_id"] == anchor["period_id"])]
    for _, r in fwd.iterrows():
        if pd.isna(r["team_id"]):
            continue
        if not same_id(r["team_id"], passer_team):
            return r
    return None
```

  (v1 returns the **row** — `return r`, `_chaining.py:46` — P14; keep that shape.)
- [ ] **Step 4 — Run, verify GREEN.** Same command as Step 2. Both PASS.
- [ ] **Step 5 — Regression.** `python -m pytest tests/tracking/test_defensive_credit_chaining.py tests/tracking/test_defensive_credit_orchestration.py -q`. Expected: PASS.

---

## Task 2: Item 3 — Line-break-gated through-ball (spec §5)

### Task 2a: Extract a `home_team_id`-free straddle core (N3)

**Files:** `silly_kicks/tracking/_line_breaking.py`; `tests/invariants/test_invariant_line_breaking.py`,
`tests/tracking/test_line_breaking*.py`

- [ ] **Step 1 — Characterize the one functional `home_team_id` site.** Read `_line_breaking.py:250-260` (the
  coordinate flip) and `:158-161,225` (opponent selection from `action_team`). Confirm `home_team_id` is used
  *only* to flip coords, nowhere to pick the opponent (spec §5 / review N3).
- [ ] **Step 2 — Write a characterization test (GREEN now, guards the refactor).** In
  `tests/tracking/test_line_breaking_core.py` (new): build a small linked action+frame, call the public
  `detect_line_breaking(actions, frames, home_team_id=H)`, snapshot the `line_break__ward` /
  `line_breaking_type__ward` / `lines_broken__ward` outputs. Run: PASS (pins current behaviour).
- [ ] **Step 3 — Extract the core.** Add a private `_straddle_core(action_ltr_pass_xy, opp_positions_ltr, params)`
  that takes **already-action-LTR** coordinates (pass start/end + opponent points) and returns
  `(is_break: bool|NA, break_type: str|None, n_lines: int|NA)` — the Ward-cluster + `_segments_intersect`
  straddle logic lifted verbatim, minus the coordinate flip. Re-implement `detect_line_breaking` to (i) select
  opponents from `action_team`, (ii) flip to action-LTR via the existing branch, (iii) delegate to
  `_straddle_core`. No output change.
- [ ] **Step 4 — Run the characterization + invariant tests.** `python -m pytest tests/tracking/test_line_breaking_core.py tests/invariants/test_invariant_line_breaking.py tests/tracking/test_line_breaking*.py -q`. Expected: PASS (byte-identical outputs; the 5 pinning tests still hold).
- [ ] **Step 5 — Re-point TF-32's tests at the core (N3).** Where the existing line-breaking tests assert the
  straddle logic, add at least one that calls `_straddle_core` directly on action-LTR inputs, so the core and
  `detect_line_breaking` are pinned to one implementation. Run: PASS.

### Task 2b: Precompute the line-break signal on `RuleContext` (N3/B3b, spec §5)

**Files:** `_orchestration.py`, `_rules.py`

- [ ] **Step 1 — Red-first perf guard (P9 — apply the GUARD RULE, don't "imagine" the failure).** In
  `tests/tracking/test_defensive_credit_perf_budget.py`, spy `link_actions_to_frames` (and `linkage`/the Ward
  clustering) and assert one link call for a **100-action fixture that actually contains successful passes
  reaching the line-break path** (else the guard is vacuous — P9). To make it genuinely red-first: temporarily
  wire the naive per-action `detect_line_breaking` call inside the rule loop, **run the perf test and SEE IT
  RED** (many link/cluster calls), then revert to the precompute. Record that you saw it red.
- [ ] **Step 2 — Implement precompute, gated to candidate rows (P10).** In `_orchestration.py`, after the
  single `link_actions_to_frames` call, compute the per-action boolean `line_break_between_lines` **only for
  candidate rows** — successful passes (`rule_failed_marking_through_ball` fires only on those, `_rules.py:403`)
  with a linked frame; **every other action (shots, fouls, clearances, take-ons) stays `<NA>`** so no Ward
  clustering runs for them (the perf gate counts link calls, not clustering — P10). Feed `_straddle_core`
  action-LTR opponent points via the scalar-flip idiom (reuse the orchestrator's once-computed `flip_series`,
  scalar `flip` per action, N4). Store the result on `RuleContext`. Thread the existing `links=` so no second
  link happens. **Skip the precompute entirely when the rule is disabled (Q8):** gate it on
  `RULE_FAILED_MARKING_THROUGH_BALL in enabled` (the `enabled` list already exists, `_orchestration.py:96`) —
  one line, drops all Ward-clustering cost for callers who disable the rule.
- [ ] **Step 3 — Run perf + orchestration tests.** Expected: one link call; clustering only on candidate rows;
  PASS. (Add a rough per-match runtime note to Task 6 so a clustering regression is visible.)

### Task 2c: Swap the through-ball gate; remove `through_ball_delta_xt_min` (B4)

**Files:** `_rules.py`, `_params.py`; `tests/tracking/test_defensive_credit_rules*.py`

- [ ] **Step 1 — Write failing behavioural tests (both directions, spec §9).**

```python
def test_through_ball_fires_on_between_lines_break():
    # a pass that straddles two adjacent same-line defenders -> fires
    ...
    assert any(r.rule == "failed_marking_through_ball" for r in rows)

def test_through_ball_does_not_fire_on_progressive_non_linebreak():
    # a high-ΔxT progressive pass that does NOT break the line (v1 WOULD have fired) -> no fire
    ...
    assert not any(r.rule == "failed_marking_through_ball" for r in rows)

def test_through_ball_no_fire_on_genuine_short_circuit():
    # Q3: BUILD the real condition, don't force the signal. A frame with fewer than min_opponents
    # (default 3, _line_breaking.py:50) -> genuine short-circuit-0 -> no fire. Also exercises Task 2b's
    # candidate gating (a real precompute must NOT write True onto this row).
    rows = _run(_pass_with_only_two_opponents_in_frame())
    assert not any(r.rule == "failed_marking_through_ball" for r in rows)

def test_through_ball_no_fire_on_unlinked_action():
    # Q3: an action with no linkable frame -> genuine <NA> -> no fire.
    rows = _run(_through_ball_pass_with_no_linkable_frame())
    assert not any(r.rule == "failed_marking_through_ball" for r in rows)
```

  (These build the actual conditions — a forced-signal loop would only re-test `_is_true`'s collapse, which is
  already covered, and could not catch a precompute bug that writes `True` onto an unlinked row —
  [[feedback_auto_enumerating_gates_cannot_see_an_all_na_surface]] / vacuous-fixture discipline.)

- [ ] **Step 2 — Run, verify RED.** Expected: the second test FAILS under the v1 ΔxT gate.
- [ ] **Step 3 — Implement the gate swap.** In `rule_failed_marking_through_ball`, replace the
  `dxt >= ctx.params.through_ball_delta_xt_min` condition with the family's nullable-boolean helper (P1 — do
  **NOT** write `is True`: `np.bool_(True) is True` and `pd.NA is True` are both `False`, so `is True` would
  make the rule fire **never**):

```python
if not _is_true(ctx.line_break_between_lines[idx]):   # _rules.py:113; True fires, False/short-circuit-0/<NA> don't
    return []
```

  This gives §5's four-state collapse for free. Remove the two `_xt_at` ΔxT calls. Add a call-site comment
  pinning the `between_lines` meaning (MINOR).
- [ ] **Step 4 — Remove the param.** Delete `through_ball_delta_xt_min` from `_params.py`
  `DefensiveCreditParams`. Grep the repo for the name (`rg through_ball_delta_xt_min`) and update every
  reference (tests, fixtures). A caller passing it now gets a standard `TypeError` (frozen dataclass) — add a
  test asserting `pytest.raises(TypeError)`.
- [ ] **Step 5 — Run.** `python -m pytest tests/tracking/test_defensive_credit_rules*.py tests/tracking/test_defensive_credit_orchestration.py -q`. Expected: PASS.

---

## Task 3: Item 1 — Reverse-xT pressing lens (spec §3)

### Task 3a: Closed-vocabulary constants (B5)

**Files:** `_params.py`, `_rules.py`; `tests/tracking/test_defensive_credit_orchestration.py`

- [ ] **Step 1 — Add constants.** In `_params.py`:
  `SIZING_XT_PRESSING = "xt_pressing"`; `SIZING_VALUES = (SIZING_XG, SIZING_XT, SIZING_XT_PRESSING)`;
  `ANCHOR_TYPE_VALUES = ("shot", "pass", "bad_touch", "cross", "take_on")`.
  Replace the five inline `anchor_type` string literals in `_rules.py` with the constants.
  (`RESOLUTION_VALUES` is defined in **Task 4 Step 6**, where the `resolution` column lands — P11; defining its
  closed-set test here would be a vacuous pass.)
- [ ] **Step 2 — Repoint the guard test (only what passes NOW; Q1).** Change
  `test_defensive_credit_orchestration.py:43` from the hardcoded `<= {"xg","xt"}` to
  `set(out["sizing"]) <= set(SIZING_VALUES)`; add an `anchor_type` closed-set test. Run: PASS (constants match
  current behaviour). **The exhaustive-when-on `xt_pressing` assertion does NOT belong here** — `pressing_lens`
  and `xt_pressing` don't exist until Task 3b, so it moves to **Task 3b Step 6a** (RED before emit, GREEN
  after).

### Task 3b: The lens (B6/N2)

**Files:** `_sizing.py`, `_rules.py`, `_params.py`; tests

- [ ] **Step 1 — Add the param.** `pressing_lens: bool = False` on `DefensiveCreditParams` (validated bool).
- [ ] **Step 2 — Write the T1 ground-truth test (spec §9, P3).** Pin **literal** expected values from a
  **hand-built, doubly-asymmetric** xT grid — **NOT** `xt.rate_at(...)` (which does not exist; the real API is
  `values_at_points`/`ExpectedThreat.rate`) and **NOT** by re-invoking the production lookup on both sides
  (that would pass green through a lookup bug — [[feedback_identity_gate_expectation_by_construction]]; note
  `values_at_points` carries a documented `(n_rows-1)-yj` row inversion, so an implementation-by-construction
  test would hide exactly the class of bug T1 exists to catch). Build an `ExpectedThreat` whose grid cells are
  **hand-set and asymmetric in BOTH x and y** (so a row-inversion or axis-swap changes the answer), then assert
  the numbers you *put in the grid*:

```python
def test_pressing_lens_reflects_to_the_mirror_point():
    # Hand-build a grid where the cell covering (20,20) holds a KNOWN small value and the cell covering
    # (85,48) holds a KNOWN large value; asymmetric in x AND y. Expected numbers = the cell values set here,
    # independent of the lookup implementation.
    xt = _hand_built_asymmetric_xt(cell_at_20_20=0.013, cell_at_85_48=0.207)   # helper sets the grid explicitly
    assert sized_xt(20.0, 20.0, xt, pressing_lens=False) == pytest.approx(0.013)   # literal
    assert sized_xt(20.0, 20.0, xt, pressing_lens=True)  == pytest.approx(0.207)   # literal, = the (105-20, 68-20)=(85,48) cell
    assert sized_xt(20.0, 20.0, xt, pressing_lens=True) > sized_xt(20.0, 20.0, xt, pressing_lens=False)  # semantic
```

  Write `_hand_built_asymmetric_xt` to construct an `ExpectedThreat` with an explicit grid (see how the
  xthreat tests build a fitted model with a known grid); the two literal expectations are the cell values it
  sets, the third assertion pins the deep-regain semantic. **Fallback if a fitted `ExpectedThreat` won't accept
  an explicit grid (Q9)** — `require_fitted_xt` may reject a hand-set grid: fit on a tiny synthetic set, then
  read the **grid array element directly** with hand-computed cell indices and assert `sized_xt(...)` against
  *that array value*. Reading the array bypasses `values_at_points`, so a row-inversion is still caught — do
  **not** drift back to calling the production lookup on both sides.
- [ ] **Step 3 — Run, verify RED** (`sized_xt` doesn't exist yet).
- [ ] **Step 4 — Implement the primitives-only helper (N2, P2).** In `_sizing.py`:

```python
from ._params import _FIELD_LENGTH, _FIELD_WIDTH   # acyclic: _params imports only spadlconfig, no back-edge to _sizing

def sized_xt(x: float, y: float, xt, *, pressing_lens: bool) -> float:
    """Value port: xT at (x,y), or at its 180deg reflection under the pressing lens.
    Primitives only -- takes no RuleContext (avoids the _sizing -> _rules import cycle)."""
    from silly_kicks.xthreat import values_at_points   # function-local: keeps the xthreat import lazy (the cycle _sizing.py:29 guards)
    if pressing_lens:
        x, y = _FIELD_LENGTH - x, _FIELD_WIDTH - y
    return float(values_at_points(xt, [x], [y])[0])   # the SAME per-point lookup extinguished_xt/_xt_at use
```

  **Correcting the plan's earlier note:** the constraint is that the **xthreat** import stays function-local
  (cycle-closing, `_sizing.py:29-32`, gated by `tests/test_no_import_cycles.py`) — a **sibling `._params`
  import is fine** (`_params` → `spadlconfig` only, no back-edge). Prefer `._params`'s `_FIELD_LENGTH`/
  `_FIELD_WIDTH` over `spadlconfig.field_length` for **single-source** reasons — both are first-party, so
  "first-party" does not distinguish them (Q10); the family already exposes the constants, so reuse them.
  Importing the underscore-prefixed names across modules is fine: ruff's `select` has no `PLC`/`TID`
  (private-name-import is not enabled), and the repo already does it (`pitch_control/_cache.py:26`). Verify the
  exact `values_at_points` signature against `xthreat/_physical.py:159` at implementation (it is
  `values_at_points(model, x, y, *, require_fitted=True)`).
- [ ] **Step 5 — Gate per-call-site (B6).** In the four turnover rules (`pressure_pass_fail`,
  `forced_bad_touch`, `synchronized_final_third_pressure`, `recovery_double_credit`), replace the
  `_xt_at(ctx, x, y)` sizing calls with `sized_xt(x, y, ctx.xt, pressing_lens=ctx.params.pressing_lens)`. Each
  row reflects the point it is sized at (spec §3 two-anchor note: `recovery_double_credit`'s `+recoverer` uses
  the recovery point). Do **not** touch `_xt_at` (the through-ball's ex-caller — already removed in Task 2c,
  so `_xt_at` now has only the turnover callers, but leave it intact as the raw lookup).
- [ ] **Step 6a — Write the exhaustive-when-on test, RED-first (Q1, moved from Task 3a).** With
  `pressing_lens=True`, assert the turnover-row `sizing` set **equals** `{"xt_pressing"}` (exhaustive, not just
  subset) and that the shot rows stay `"xg"`. Run: **RED** (`xt_pressing` not emitted yet) — this makes it a
  real gate, not a test written against already-working code.
- [ ] **Step 6 — Emit the `xt_pressing` token.** When `pressing_lens` is on, the turnover rows' `sizing` value
  is `SIZING_XT_PRESSING`; else `SIZING_XT`. Wire it in the `CreditRow` construction. Run Step 6a: **GREEN**.
- [ ] **Step 7 — Run.** T1 test + the exhaustive-when-on guard + full orchestration tests. Assert **default
  (`pressing_lens=False`) is byte-identical to pre-change** (snapshot a cohort's long-form before/after). PASS.
- [ ] **Step 8 — Docstring/NOTICE caveat + worked example (MINOR, P13).** Add to the `pressing_lens`
  docstring + NOTICE: the lens "diverges from the validated `xT(origin)` standard **and under-values last-ditch
  defending**." Include the **worked numeric example** the spec §3 requires (the reflection is easy to invert):
  e.g. "a deep regain at `(20, 20)` reflects to `(85, 48)` — high `xT` (high press rewarded); a last-ditch
  clearance at `(100, 34)` reflects to `(5, 34)` — near-zero `xT` (last-ditch under-valued)."

---

## Task 4: Item 2 — Lane-geometry `shot_block` blocker (spec §4)

**Files:** `_resolution.py`, `_params.py`, `_orchestration.py`, `_rules.py`; tests

- [ ] **Step 1 — Add the corridor params.** On `DefensiveCreditParams`: `shot_lane_cone_width_factor: float =
  0.2` (match `_cover_shadows`); `shot_lane_max_t: float = <intent-set, e.g. 0.9>` (distance-along-lane cap,
  normalised `t` units — the flag-independent GK backstop, §4); `shot_lane_min_half_width_m: float =
  <intent-set, ~1.0>` (corridor floor so the cone doesn't pinch to 0 at the shooter, Q4). Add positive-float
  validation for all three in `__post_init__` (`_params.py:78-89`). Mark `shot_lane_max_t` and
  `shot_lane_min_half_width_m` intent-set/never-calibrated (Open Questions §11).
- [ ] **Step 2 — Write the three-fixture discriminator (T5, spec §9), all RED-first.**

```python
def test_lane_blocker_credits_on_lane_not_nearest_origin():
    # defender A near origin but off-lane; defender B on the lane but farther from origin
    # v1 credits A; v2 must credit B
    ...
def test_lane_blocker_excludes_goalkeeper():
    frames = _frame_with_gk_on_shot_line()
    assert _is_goalkeeper_set(frames)              # N5: assert the flag is genuinely populated first
    rows = compute_defensive_credits(...); block = _shot_block_row(rows)
    assert block.player_id != _gk_id(frames)
def test_lane_blocker_credits_far_but_on_lane_defender():
    # defender 10 m from origin, dead on the lane -> credited (proves origin threshold dropped)
    ...
```

- [ ] **Step 3 — Run, verify RED.**
- [ ] **Step 4 — Widen the `Mode` Literal + closed-set test.** In `_resolution.py:15` add `"lane_blocker"` to
  the `Mode` Literal; add a test asserting `set(get_args(Mode)) == {...}` (mirrors B5).
- [ ] **Step 5 — Implement `lane_blocker` resolution in `_resolution.py`.** The lane is the segment from the
  shot origin to the attacked goal `(105, 34)`; let `û` = its unit vector and `shot_dist` its length. For each
  defending **outfielder** (`~is_goalkeeper`, N5), reproject its frame coords to action-LTR via the scalar-flip
  idiom (`_resolution.py:53-54`, `_FIELD_LENGTH`/`_FIELD_WIDTH` — P2/P4 no fourth hardcoded `105`/`68`), then:

Write it **vectorised over all defenders** (Q7 — `resolve_responsible_defenders` is vectorised at
  `_resolution.py:53-58`; `d_xy` below is an `(n,2)` array, everything else broadcasts):

```python
d = d_xy - origin_xy                                                         # (n,2)
t = np.clip((d @ u_hat) / shot_dist, 0.0, 1.0)                               # fraction along lane; 0<=t<=1 IS the in-front constraint
half_width_at_t = np.maximum(                                               # Q4: FLOOR the corridor -- see below
    params.shot_lane_cone_width_factor * shot_dist / 2.0 * t,               # MATCHES _cover_shadows:556 (full-dist half-width * normalised t)
    params.shot_lane_min_half_width_m,
)
perp = np.abs(u_hat[0] * d[:, 1] - u_hat[1] * d[:, 0])                       # Q5: 2-D cross as an explicit scalar (np.cross on 2-D vectors is DeprecationWarning in numpy>=2.0)
in_corridor = (perp <= half_width_at_t) & (t <= params.shot_lane_max_t)     # max_t = the distance-along-lane cap (§4 flag-independent GK backstop)
```

  **Why the floor (Q4):** the raw cone pinches to **0 at `t=0`** (the shot origin) — correct for a *pass* lane
  (ball starts at the passer's foot) but **backwards for a shot block**, whose most common real blocker is a
  defender who has closed the shooter down (**small `t`**), exactly where the corridor is ~0 wide. Without the
  floor `lane_blocker` would fall back to `nearest` on most real blocks (the §9 fallback-rate number would
  eventually surface it, but only after the whole item is built). `shot_lane_min_half_width_m` ≈ a body's reach
  (~1.0 m), intent-set (Open Questions §11). Blocker = the in-corridor defender minimising `perp`.
  **`shot_lane_max_t`** is the distance-along-lane cap in the same normalised-`t` units (a keeper on the
  goal-line sits at `t≈1`; cap below that). Fallback → `mode="nearest"` when no defender is in the corridor or
  no frame links; record `"lane"` vs `"nearest_fallback"` in the `resolution` value.
- [ ] **Step 6 — Wire `rule_shot_block` + the mode-valued `resolution` column (P6 — 4 edit sites, not one).**
  `rule_shot_block` asks for `mode="lane_blocker"` (thin policy line — geometry stays in `_resolution`). Then:
  1. **`CreditRow` gains a 10th field** `resolution: str` (`_rules.py:50-60`) — REQUIRED, because
     `_to_long_form` does `pd.DataFrame([r.__dict__ ...])[_LONG_COLS]` (`_orchestration.py:122-125`), so a
     `_LONG_COLS` entry with no `CreditRow` field is a **`KeyError`**.
  2. **Define `RESOLUTION_VALUES`** in `_params.py`:
     `("nearest", "all_within", "all_within_beyond_nearest", "lane", "nearest_fallback", "anchor_actor")`.
     The `−passer` rows in `pressure_pass_fail` / `recovery_double_credit` credit the **acting-team passer**,
     not a resolved defender → their `resolution = "anchor_actor"` (P6; a dedicated member, not a null hole).
  3. **Sweep ~14 `CreditRow(...)` construction sites across all 10 rules** — each passes the mode that actually
     resolved it (`resolve_responsible_defenders`'s `mode`, or `"lane"`/`"nearest_fallback"` for shot_block,
     or `"anchor_actor"` for the passer debits). This is a sweep, not a one-liner.
  4. **`_empty_long_form`** (`_orchestration.py:132-137`): add `resolution` with `dtype="object"` so the empty
     path matches the populated path.
  Add the `resolution` column to `_LONG_COLS` (10→11); repoint `test_defensive_credit_orchestration.py:40`'s
  exact-equality list to the 11-col schema; add the `RESOLUTION_VALUES` closed-set test (asserting the emitted
  `resolution` set ⊆ `RESOLUTION_VALUES`, and that `anchor_actor` appears on a `−passer` row).
- [ ] **Step 7 — Run.** The three discriminators + orchestration schema tests. PASS.

---

## Task 5: Item 5 — Pressure-commitment cue (spec §6)

### Task 5a: Lift the nearest-opponent resolver (N6)

**Files:** `silly_kicks/tracking/_opponent_resolution.py` (NEW), `_resolution.py`; tests

- [ ] **Step 1 — Write a test for the shared core (P5 — sorted frame, NOT "nearest or None").** The core must
  serve all three of `resolve_responsible_defenders`'s modes (`nearest` / `all_within` /
  `all_within_beyond_nearest`, `_resolution.py:74-78`), so a "nearest-or-None" return is unbuildable as an
  adapter base. `test_opponent_resolution.py`: `opponents_within(frame_slice, *, anchor_x, anchor_y,
  acting_team_id, threshold_m, flip) -> pd.DataFrame` returns the within-threshold opponents **sorted ascending
  by distance** (`player_id`, `team_id`, `distance_m`) — empty frame if none. Assert on a small frame. RED
  (module doesn't exist).
- [ ] **Step 2 — Implement the shared core** taking `threshold_m: float` (not a params object). It must
  preserve, **verbatim from `_resolution.py:44`**, the opponent mask
  `~ids_match(fr["team_id"], acting_team_id) & fr["team_id"].notna() & ~fr["is_ball"].astype(bool)` (ball-row +
  NaN-team exclusions, ADR-027/ADR-019) and the `_resolution.py:53-54` scalar-flip reprojection +
  `_FIELD_LENGTH`/`_FIELD_WIDTH`. **`~is_goalkeeper` does NOT live in the core** — Item 2's `lane_blocker` needs
  it, Item 5's press resolution must NOT (a keeper can press), so the caller applies it (or pass it as an
  explicit `exclude_goalkeeper: bool` kwarg). GREEN.
- [ ] **Step 3 — Adapt `_resolution.py`.** Make `resolve_responsible_defenders` a thin adapter: for the
  `nearest`/`all_within`/`all_within_beyond_nearest` modes, call the core with
  `threshold_m = params._proximity_threshold(anchor_x, anchor_y)` (dependency inverted), then apply the mode
  selection (`iloc[[0]]` / `iloc[1:]` / all) on the returned sorted frame. **`mode="lane_blocker"` MUST bypass
  `threshold_m` entirely (Q2)** — spec §4 forbids the origin proximity filter (it would make Item 2
  near-vacuous), so `lane_blocker` branches *before* the threshold call (or passes `math.inf`); its only
  filters are the corridor + the `t`-cap (Task 4 Step 5). Since Task 4 runs **before** this refactor,
  guard it: `lane_blocker` must remain threshold-free after the refactor — **Task 4's
  `test_lane_blocker_credits_far_but_on_lane_defender` (defender 10 m from origin, on the lane) is the guard;
  re-run it here and confirm still GREEN.** Then run the full defensive-credit suite — assert **byte-identical**
  behaviour for the three original modes (box-aware threshold + mode selection preserved).

### Task 5b: Extract the velocity-availability helper (N/MINOR speed_source)

**Files:** `silly_kicks/tracking/_velocity_availability.py` (NEW), `_das.py`; tests

- [ ] **Step 1 — Extract** `velocity_unavailable_by_design(frames) -> bool` from `_das.py:249-265` into the new
  module (preserve the "ALL rows marked, not any" rule + its docstring). Re-point `_das.py` to import it.
- [ ] **Step 2 — Run DAS tests** (`tests/tracking/test_das*.py`). PASS (behaviour unchanged).

### Task 5c: The commitment primitive (N7, spec §6)

**Files:** `silly_kicks/tracking/_press_commitment.py` (NEW); `tests/tracking/test_press_commitment.py`

- [ ] **Step 1 — Params + source vocab.** A `PressCommitmentParams` frozen dataclass:
  `commitment_window_seconds: float = 0.5`, `press_max_distance_m: float = <intent-set>`,
  `min_separation_m: float = <intent-set>` (mark all three intent-set/never-calibrated). A closed
  `PRESS_COMMITMENT_SOURCE_VALUES` tuple `("computed", "no_pressing_defender", "velocity_unavailable",
  "window_too_short", "degenerate_axis", "unlinked")` with a structural raise on unknown (the
  `DasUnscoreableError` pattern, `_das.py:42-69`).
- [ ] **Step 2 — Write RED tests (spec §9 / N7):** committed press (defender accelerating along the axis) →
  positive `press_commitment`; containing (braking) → negative; no opponent within `press_max_distance_m` →
  `no_pressing_defender` NaN; window unspanned → `window_too_short` NaN; defender within `min_separation_m` →
  `degenerate_axis` NaN; all rows `speed_source="unavailable"` → `velocity_unavailable` NaN; partial-marked or
  missing `vx/vy` → `raise`.
- [ ] **Step 3 — Implement.** `compute_press_commitment(actions, frames, *, links=None, params=None)`. **State
  the columns it reads off `actions` (P8, for the atomic-mirror decision):** ids + `time_seconds` (for linking)
  + `team_id` (acting team, to resolve the actor's frame row) — **NOT** action `start_x`/`start_y` (the actor
  position and the pressing defender both come from the linked *frame*, not the action point). Resolve the
  pressing defender via the Task-5a core `opponents_within(..., threshold_m=press_max_distance_m,
  exclude_goalkeeper=False)` and take the nearest (first row; a keeper may press). Fix `axis` = unit(defender→
  actor) at the action frame; per window frame `v_close = (vx,vy)·axis`; `press_commitment =
  leastsquares_slope(v_close over the window frames)` (NOT a two-point diff, N7); require a fixed ≥0.1 s
  baseline (skip-at-edges, no sub-baseline fallback). Apply the degenerate-axis (`min_separation_m`), distance
  (`press_max_distance_m`), and velocity guards → the matching `_source` value. Emit `press_commitment`,
  `press_commitment_closing_speed`, `press_commitment_source`.
- [ ] **Step 4 — Run.** All Step-2 tests GREEN.

### Task 5d: Aggregator + atomic mirror + glossary/NOTICE + gates (spec §6/§8)

**Files:** `tracking/features.py`, `atomic/tracking/features.py`, `feature_glossary.py`, `NOTICE`; test registries

- [ ] **Step 1 — `add_press_commitment`** in `tracking/features.py`: `@nan_safe_enrichment`, pure (returns a
  new frame), idempotent provenance-merge, accepts `links=`. Delegates to `compute_press_commitment`.
- [ ] **Step 2 — Thin atomic mirror** in `atomic/tracking/features.py` (N11/P8). Because
  `compute_press_commitment` reads only ids + `time_seconds` + `team_id` off `actions` (Step 3 — **no**
  `start_x`/`start_y`), the mirror needs **no coordinate rename bridge** — it is a **pass-through** (verify: if
  the delegate never reads a std-only coordinate column, no `x`→`start_x` rename is needed, unlike
  `add_cover_shadows` which renames because it reads `start_x`). Confirm the exact columns before writing any
  bridge; a rename that does nothing is worse than none. **Record why it exists (Q11)** — a one-line comment on
  the mirror: it is a pure alias for **API symmetry + discoverability** (every `tracking.features` aggregator has
  an `atomic.tracking.features` twin), and the C4 `atomic.tracking` +1 is **symmetry, not new capability** —
  otherwise the next reader deletes it as dead weight.
- [ ] **Step 3 — Register in the auto-gates (P7 — the four gates do NOT all cover `atomic.tracking`).** The
  gate surfaces differ; register accordingly:
  - `tests/test_add_star_purity.py` (`PURITY_ENTRIES`): **two** keys — `"tracking:add_press_commitment"` AND
    `"atomic.tracking:add_press_commitment"` (the meta-assertion iterates all four packages, `:572-577`).
  - `tests/test_enrichment_nan_safety.py`: **tracking only** (its surface is `tracking.features`, not
    `atomic.tracking.features`, `:20-24`).
  - `tests/tracking/test_id_dtype_invariance.py`: **tracking only** (`dir(F)` over tracking features).
  - `tests/tracking/test_aggregator_column_liveness.py`: **tracking only** (`tracking.__all__`, `:527-532`).
  So the **atomic mirror is gate-covered by purity alone** — an honest limitation (record it in a one-line
  comment on the mirror + note it in spec §8), not a silent assumption. The tracking meta-assertions require
  registration or CI fails.
- [ ] **Step 4 — xfns absence guard (T4).** Add `not hasattr(T, "press_commitment_xfns")` to
  `tests/tracking/test_defensive_credit_xfns_absence_guard.py` (or the nearest xfns-absence guard).
- [ ] **Step 5 — Glossary + NOTICE (B11).** Add three `FeatureColumn` entries (`press_commitment`,
  `press_commitment_closing_speed`, `press_commitment_source`) to `feature_glossary.py` with `unit`,
  `emitting_module="…_press_commitment"`, `higher_is_better` direction, and a practitioner-concept
  `attribution` string; add that exact string to `NOTICE`. Run `tests/test_feature_glossary_coverage.py`,
  `tests/test_feature_glossary_notice_linkage.py`, `tests/invariants/test_glossary_emitted_columns.py`. PASS.
- [ ] **Step 6 — Run the tracking suite.** `python -m pytest tests/tracking/ tests/test_add_star_purity.py tests/test_enrichment_nan_safety.py tests/test_feature_glossary_*.py -q`. PASS.

---

## Task 6: Owner-gated e2e acceptance numbers (spec §9)

**Files:** `tests/tracking/test_defensive_credit_e2e.py`

- [ ] **Step 1 — Extend the e2e to REPORT (print/assert-bounds) the acceptance table** on real GS match 10502:
  Item 2 `resolution` breakdown + total `shot_block` row count vs v1 + % attribution changed + GK-credit count
  + **per-team `is_goalkeeper` coverage** (N5); Item 3 line-break state distribution (`True`/`False`/
  short-circuit-0/`<NA>`) + v1↔v2 firing delta; Item 5 `press_commitment_source` distribution + sign split.
- [ ] **Step 2 — Run (owner, with token):** `PINING_FOR_THE_DATA_TOKEN=… python -m pytest
  tests/tracking/test_defensive_credit_e2e.py -q -s`. **Report the numbers to the owner**; if GK-flag coverage
  is 0, or fallback dominates, or the firing delta ≈0, STOP and reconsider (spec §9 fail conditions).

---

## Task 7: Commit-prep (STOP for owner approval — do NOT commit without it)

- [ ] **Step 1 — Full lint trio** ([[feedback_run_full_lint_after_any_lint_fix]]):
  `python -m ruff check silly_kicks/ tests/ scripts/ && python -m ruff format --check silly_kicks/ tests/ scripts/ && python -m pyright`. Fix + re-run the whole trio on any failure.
- [ ] **Step 2 — Full non-e2e suite:** `python -m pytest tests/ -m "not e2e" -q`. Expected: all PASS.
- [ ] **Step 3 — Merge from main + reconcile version (the recurring parallel-session collision).** `git fetch
  origin` then merge/rebase `origin/main`; re-check the register next-free. **As of 2026-07-24 it is
  `4.61.0 / PR-S132 / ADR-049`** — the parallel session already took `4.60.0/PR-S131` (#175) that TF-51 v2 had
  reserved (the 2nd such collision this cycle after 4.58.0), so **re-confirm AGAIN at this step** — it may have
  advanced further. Resolve any CHANGELOG/TODO conflicts.
- [ ] **Step 4 — Version bump ×5 + docs** ([[feedback_version_bump_hard_gate]]): `pyproject.toml`,
  `silly_kicks/__init__.py`, `uv.lock`, `CHANGELOG.md`, `TODO.md` release line — all to the confirmed next
  version; write the ADR (`ADR-<next>`, supersedes ADR-046/047 Opta wording); update `CLAUDE.md`
  defensive-credit bullet + `docs/PRIVATE_CONSUMERS.md` if any private module moved.
- [ ] **Step 5 — Re-derive the C4 count** (ADR-043 — do NOT copy a number): regenerate `architecture.html` via
  the c4 skill's Java pipeline; expected +1 in `silly_kicks.tracking` and +1 in `silly_kicks.atomic.tracking`
  (`add_press_commitment` + its mirror). Note the atomic increment is a **C4-DSL/documentation** count only —
  the aggregator-liveness surface is tracking-only (P15/P7), so no liveness entry accompanies the mirror. Run
  `tests/test_c4_dsl_description_cap` if the DSL text changed.
- [ ] **Step 6 — `/final-review`** (includes C4 — [[feedback_final_review_gate]]).
- [ ] **Step 7 — STOP.** Report completion + the e2e acceptance numbers to the owner. **Do NOT `git commit`,
  push, open a PR, or tag without explicit approval** ("stop when ready to commit").

---

## Self-Review (author checklist — completed)

**Spec coverage:** every spec §3–§7 item + §8 gates + §9 acceptance numbers + §11 open params maps to a task
(1=§7, 2=§5, 3=§3, 4=§4, 5=§6, 6=§9, 7=§8/C4). Item 4 + Opta correctly ABSENT (split/dropped).
**Placeholder scan:** provisional param *values* (`press_max_distance_m`, `min_separation_m`, distance-cap) are
flagged intent-set in Task 5c/4 and Open Questions — a spec-time decision, not a plan placeholder. No TBD steps.
**Type/name consistency:** `sized_xt(x, y, xt, *, pressing_lens)`, `opponents_within(..., threshold_m, ...)`,
`_straddle_core`, `resolution`/`RESOLUTION_VALUES`, `SIZING_VALUES`, `press_commitment_source` used identically
across tasks. **Order:** B2 → 3 → 1 → 2 → 5, matching spec §1 (Task 2c removes the `_xt_at` through-ball caller
before Task 3 gates the seam).
