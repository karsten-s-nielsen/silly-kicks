# TF-54b — Per-action goal resolution for SB360 freeze-frames (design)

**Date:** 2026-09-10
**Status:** Approved — spec review passed (0 blocking, 0 should-fix); GOAL-SPEC-01 ruled **(i)** and
GOAL-SPEC-02/03 folded in (2026-09-10). Ready for implementation planning.
**Feature:** `silly_kicks.territorial_defense` (TF-54b)
**Corrects:** ADR-090 (an implicit, undocumented assumption in the compute)
**Proposed decision record:** ADR-091

---

## 1. Summary

`compute_territorial_defense` scores **~0 on real StatsBomb-360 freeze-frames — its primary target
data.** The compute resolves the attacked goal once per match via `resolve_defended_goals` (a mean-GK-x
estimator), but SB360 frames are **per-acting-team-LTR** (each freeze-frame is aligned to its action's
LTR convention, so the acting team attacks x=105). A team's keeper is therefore **bimodal** in x across
the match (x≈0 in its own actions, x≈105 in the opponent's), the per-match mean lands near midfield, and
`resolve_defended_goals` collapses both teams to the same end → `attacked_goal` returns `None` for every
team → every frame drops `unresolved_geometry`.

The fix is **per-frame goal resolution from the action-LTR convention** (deterministic, needs no GK), made
**dual-mode** by an explicit `frame_convention` parameter so the existing per-match path — which is correct
for match-oriented (continuous-tracking-derived) frames — is preserved. Prototype on real WC2022 confirms
the direction: Arm A goes from **0 → 103 scored** on one match, with sensible non-negative suppression
values.

This is additive and behind a new default that only affects SB360 (per-action) frames; the metric has
never produced committed output, so there is **no re-materialize and no VAEP retrain**.

---

## 2. The finding (problem statement)

A pre-commit sanity smoke ran the full library path on a real **public WC2022** SB360 match — **match
`3857254`, Denmark 0-0 Tunisia** (the same match referenced in §3 and §10) — via statsbombpy open data →
`shape_snapshots` → `snapshot_to_tracking_frames` → the actor bridge → `compute_territorial_defense`. It
executed cleanly and the report **conserved exactly** (ADR-042), but:

| Arm | scored | drops |
|---|---|---|
| A | **0 / 122** | `unresolved_geometry` 103, `no_actor` 19 |
| B | **0 / 187** | `unresolved_geometry` 169, `removal_undersupported` 2, `missing_frame` 16 |

`goal_map.attacked_goal(...)` returned `None` for **every** `(game, period, team)` despite **508 GK rows**
being present in the frames.

**Why the existing tests did not catch it.** The e2e fixture (`tests/territorial_defense/_fixtures.py`)
is **match-oriented** — team-1's keeper is pinned at x=2 in every frame — so `resolve_defended_goals`
resolves and the metric scores (`a_frames_scored == 3`). That fixture exercises the *match-oriented*
path, which SB360 **never uses**. The real per-action-LTR case had no test. This is the durable lesson
in §12.

---

## 3. Root cause

- `silly_kicks.spadl.statsbomb.convert_to_actions` emits **per-acting-team-LTR** actions (ADR-028: the
  acting team attacks x=105).
- `silly_kicks.providers.statsbomb.shape_snapshots` aligns each freeze-frame to *its* action, and
  `snapshot_to_tracking_frames` does **not** re-orient ("coordinates must be in the current SPADL
  coordinate system"). So SB360 frames are per-action-LTR: `frame_id == action_id`, one frame per action,
  each in that action's acting-team-LTR orientation.
- `resolve_defended_goals` (ADR-055) is a **per-`(game, period, team)`** estimator: it takes the *mean*
  GK x and picks the defended end via `_end_from_mean_x(mean_x < 52.5)`. It assumes a **consistent**
  orientation across a match's frames (correct for continuous tracking, home-attacks-right).
- On per-action-LTR frames a team's keeper is **bimodal**: measured on WC2022 match 3857254, team 776's GK
  x has **std ≈ 46.9** (min −4.4, max +108.3; 103 rows below center, 61 above); team 777 std ≈ 48.5. Both
  means land near center and resolve to the *same* end, so `attacked_goal` hits its "opponent end == own
  end → None" guard for every team.

The per-match estimator is structurally the wrong tool for per-action frames; no per-`(game, period, team)`
map can represent per-action orientation, because the *same* team attacks x=105 in its own actions and
x=0 in the opponent's.

---

## 4. Approaches considered

**(1) Per-frame goal map from the action-LTR convention — CHOSEN.** For a per-action frame, the acting
team of that action attacks x=105 (defends 0) and the opponent defends 105, *by the ADR-028 definition*.
Build a trivial per-frame `GoalMap` from the (known) acting team + opponent. Deterministic, needs no GK,
**FOV-proof** (a keeperless freeze-frame still resolves). Prototype-confirmed (§10).

**(2) Orient frames to home-attacks-right upfront, then reuse `resolve_defended_goals`.** Flip each
per-action frame whose acting team is the away team, producing a consistent match orientation. Rejected:
needs `home_team_id`, a per-frame point reflection, and rework of Arm B's target reflection (which is
defined against the per-action frame); more moving parts than (1) for the same result, and it re-derives
via GK estimation what (1) knows exactly.

**(3) Per-frame `resolve_defended_goals(fr)` (GK-based, one frame at a time).** A single per-action frame
*is* consistently oriented, so per-frame GK estimation would resolve it. Rejected: still GK-dependent →
fails on a keeperless freeze-frame where (1) succeeds; and it re-runs an estimator to recover a quantity
the convention already fixes.

---

## 5. Design

### 5.1 Dual-mode dispatch (explicit convention)

`compute_territorial_defense` gains a keyword-only parameter:

```python
def compute_territorial_defense(
    actions, frames, *, xt, links=None, visible_area=None,
    frame_convention: Literal["per_action_ltr", "match_ltr"] = "per_action_ltr",
    params=_DEFAULT_PARAMS,
) -> tuple[pd.DataFrame, TerritorialDefenseReport]:
```

- **`per_action_ltr` (default — SB360):** build a **per-frame** goal map from each frame's action acting
  team; do **not** call `resolve_defended_goals`.
- **`match_ltr` (continuous-tracking-derived, home-attacks-right):** the **current behavior** — one
  per-match `resolve_defended_goals(frames)` map, threaded through classify + both arms.

The default is `per_action_ltr` because the metric's declared target (ADR-090) is SB360 freeze-frames.
`match_ltr` is a **first-class, committed mode** (owner ruling, 2026-09-10), not speculative surface: it is
the correct resolution for continuous-tracking-derived per-action frames (home-attacks-right), a use the
owner intends to adopt (§11). It is therefore **genuinely exercised**, not merely fixture-preserving: it
gets its own match-oriented fixture that tests `match_ltr` *as that mode* (not as a stand-in for the SB360
default) plus a byte-identity golden against today's per-match output (§8). This keeps it clear of the §12
masking pattern — each fixture tests the convention it actually represents.

**No auto-detection.** The convention is an explicit caller declaration. GK-bimodality or
`frame_id == action_id` heuristics can misfire (ADR-059: a detector must have discriminating evidence, and
a match-oriented caller can also build 1:1 per-action frames), so the caller states it.

`frame_convention` is validated against the two-member `Literal` (a bad value raises), consistent with the
`GkdvParams`/`TerritorialDefenseParams` fail-at-construction discipline.

### 5.2 The per-frame goal map

New helper in `_engine.py`:

```python
def action_ltr_goal_map(game_id, period_id, *, acting_team_id, opponent_team_id) -> GoalMap:
    """A per-action-LTR frame's goal map: the acting team attacks x=105 (defends 0); the opponent
    defends x=105. Correct by the ADR-028 convention -- no GK, so it is FOV-proof where a per-frame
    resolve_defended_goals would fail on a keeperless freeze-frame. Keys canonical (ADR-055 rule 2)."""
    g, p = canonical_id(game_id), canonical_id(period_id)
    resolved = {
        (g, p, canonical_id(acting_team_id)): 0.0,
        (g, p, canonical_id(opponent_team_id)): float(spadlconfig.field_length),
    }
    return GoalMap(MappingProxyType(resolved), MappingProxyType({}), frozenset())
```

Both team entries are present, so `attacked_goal(acting) = 105` and `attacked_goal(opponent) = 0` resolve
via `GoalMap`'s real opponent-lookup (never `105 - get(...)`; ADR-055 rule 2 preserved).

**ADR-055 relationship (not a fork).** ADR-055 collapsed ten hand-rolled *re-estimations of the same
quantity* (mean-GK-x for match-oriented frames). `action_ltr_goal_map` resolves a **different frame
convention** from **ground truth** (the action-LTR definition), not a re-estimate of the GK-based one. It
routes through the same `GoalMap` type and the same `attacked_goal` accessor, so there is one *consumption*
seam. The spec makes this distinction explicit; the reviewer should confirm it holds.

### 5.3 Where the map is built — the compute owns the mode (GOAL-SPEC-02)

**One place owns the convention: `compute_territorial_defense`.** It builds a mode-encapsulating factory and
threads *that* down; `classify_arm_a_domain` and the arms stay **convention-agnostic** — they never see
`frame_convention`, they just call the factory (or receive a `GoalMap`). This is the spec-review's
GOAL-SPEC-02 (and answers OQ1): the mode lives in exactly one function.

```python
# inside compute_territorial_defense, after resolving `frame_convention`:
if frame_convention == "per_action_ltr":
    def goal_map_for(game_id, period_id, acting_team_id, opponent_team_id) -> GoalMap:
        return action_ltr_goal_map(game_id, period_id,
                                   acting_team_id=acting_team_id, opponent_team_id=opponent_team_id)
else:  # match_ltr
    _match_map = resolve_defended_goals(frames)
    def goal_map_for(game_id, period_id, acting_team_id, opponent_team_id) -> GoalMap:
        return _match_map            # per-match; acting/opponent unused
```

- The arms (`arm_a_threat_suppressed`, `arm_b_threat_suppressed`) already take a `goal_map` and forward it
  to `compute_threat_pc` — **no arm signature change.** They are called once per frame in the compute
  loops, so the compute calls `goal_map_for(...)` in the loop and passes the result as `goal_map=`.
  - **`_score_arm_a`:** per candidate, `goal_map_for(game, period, acting=cand.defending_team_id,
    opponent=cand.attacking_team_id)`.
  - **`_score_arm_b`:** per (defender, in-hull pass), `goal_map_for(game, period, acting=opp_row.team_id,
    opponent=defending_team_id)`. The existing target logic (`target = (ex, ey) if attacked == 105 else
    (fl-ex, fw-ey)`) is **unchanged** and takes the `else` branch correctly under `per_action_ltr` (the
    passer, as acting team, attacks 105 → `attacked=105` → no reflection, because the pass end and the
    freeze-frame are already in the same per-action frame). Under `match_ltr` the shared map drives it as
    today.
- **`classify_arm_a_domain` / `_classify_one`:** takes `goal_map_for` (replacing its `goal_map` param) and
  calls it per row — `goal_map_for(row.game_id, row.period_id, acting=row.defending_team_id,
  opponent=row.attacking_team_id)` — then runs the *identical* `attacked_goal(attacking_team_id) is None`
  drop check on the returned map. It is convention-agnostic; under `per_action_ltr` the check is
  unresolvable only when `attacking_team_id` is NA (the game lacks exactly two teams).
- **`compute_territorial_defense`:** resolves `frame_convention`, builds `goal_map_for` (the only place the
  branch exists), and threads it into classify + both arms. Under `per_action_ltr` no `resolve_defended_goals`
  call is made; under `match_ltr` it is built **once** and closed over.

`resolve_defended_goals` remains imported and used **only** inside the `match_ltr` factory.
`action_ltr_goal_map` is a public-in-package helper (no leading underscore, matching `classify_arm_a_domain`
/ `select_arm_a_domain` / `remove_player_row`) so the driver imports it from `_engine` and reuses it (§5.4).
The driver, being SB360-only, hard-codes the `per_action_ltr` factory rather than exposing `frame_convention`.

### 5.4 Driver (`scripts/validate_territorial_defense.py`)

`_measure_match` builds `gm = resolve_defended_goals(frames)` for its dose battery (`_threat`,
`_dose_unit_vector`, the inline `arm_a_threat_suppressed`). Under the SB360 default it must mirror the
compute: build a **per-frame** `action_ltr_goal_map` per scored Arm-A frame (from `cand.defending_team_id`
+ `cand.attacking_team_id`) and use it for the factual/dosed threat and the removal delta. `_dose_unit_vector`'s
fallback goal end reads the same per-frame map. The driver targets SB360 → `per_action_ltr`; it does not
expose `frame_convention` (owner-run, single-purpose).

---

## 6. Semantic changes

- **`unresolved_geometry` meaning (per_action_ltr):** now only "the game lacks exactly two teams" (rare),
  not "GK mean ambiguous" (systematic). So the vast majority of SB360 frames now **score** rather than
  drop. The **conservation identities are unchanged** (`n_frames_scored + Σ drops == n_frames_in`, and the
  Arm-B pair identity); only the *distribution* across `scored`/drop-reasons changes.
- **No re-materialize, no VAEP retrain.** The metric is `compute_*`, in no default xfn list, and has never
  produced committed output; the fix changes only SB360 (per-action) behavior. `match_ltr` output is
  byte-identical to today's behavior.
- **The metric now scores on SB360, which makes its construct validity a live question.** That is
  explicitly **out of scope** here (§13) — it is the owner-run commit-2 battery's job. This spec's success
  criterion is *the arms score and conserve on real SB360*, not *the numbers are construct-valid*.

---

## 7. Caller enumeration (CLAUDE.md rule: every caller of every changed function, classified)

**Changed public/near-public seams:** `compute_territorial_defense` (new param), `classify_arm_a_domain`
(goal resolution), `_score_arm_a`/`_score_arm_b` (internal), driver `_measure_match`, and the new
`action_ltr_goal_map` (added).

| Caller | Of | Convention | Affected? | Evidence |
|---|---|---|---|---|
| `scripts/validate_territorial_defense.py:_measure_match` | `compute_territorial_defense`, `classify_arm_a_domain` | `per_action_ltr` (default) | **YES** — now scores; dose battery re-plumbed to per-frame maps | Its corpus is SB360; the smoke showed 0-scored under the old path |
| `tests/sb360/_entries/_boundary.py:56` (SB360 audit) | `compute_territorial_defense` | `per_action_ltr` (default) | **YES** — verdict flips from all-NaN toward `differs_by_design`/scores; re-adjudicate | SB360-shaped anchor frames; this is the boundary entry |
| `tests/territorial_defense/test_compute.py` (unit) | `compute_territorial_defense` | uses match-oriented fixture → run under `match_ltr` **or** switch to the new per-action-LTR fixture | **YES** — fixture/param update | Fixture is match-oriented (§8) |
| `tests/invariants/glossary_emitted_columns.py:315` | `compute_territorial_defense` | match-oriented fixture → `match_ltr` (column set is convention-invariant) | **column-neutral**; pass `match_ltr` to keep it scoring | Only asserts emitted column *names* |
| `tests/test_scale_guards.py:739` | `compute_territorial_defense` | match-oriented `make_td_scaling_fixture` → `match_ltr` | growth guard is convention-invariant | Counts `rows_scanned`, not goal ends |
| `_compute.py` (internal) | `_score_arm_a`, `_score_arm_b` | both | **YES** — build per-frame maps | This spec |
| `tests/territorial_defense/test_engine.py:124` | `classify_arm_a_domain` | direct | **YES** — relies on the removed internal `resolve_defended_goals` default → pass `goal_map_for=lambda *a: resolve_defended_goals(frames)` | conservation test, match-oriented frames |
| `tests/test_scale_guards.py:634,651` | `classify_arm_a_domain` | direct | **YES** — `goal_map=gm` → `goal_map_for=lambda *a: gm` | ADR-073 scale guards |
| `tests/test_scale_guards.py:653` | `_score_arm_a` | direct | **YES** — `goal_map=gm` → `goal_map_for=lambda *a: gm` | ADR-073 scale guard |
| `tests/test_scale_guards.py:720` | `_score_arm_b` | direct | **YES** — `goal_map=gm` → `goal_map_for=lambda *a: gm` | ADR-073 scale guard |

**GOAL-PLAN-01 (plan review):** these 5 **direct** callers of the renamed `goal_map` → `goal_map_for` seams
are migrated in the plan's Task 2 Step 5b — the compute-level `frame_convention` dispatch does not reach
them (they take a factory, not a convention). `groups=` on these calls is the existing L7 param (default
`None`), unchanged.

**Not-a-caller but downstream:** the four unit-test fixtures (`make_e2e_fixture`, `make_rich_frame`,
`make_arm_a_fixture`, `make_td_scaling_fixture`) are match-oriented; the SB360 *audit* fixture (anchor
frames in `_boundary.py`) is per-action-LTR. No committed research artifact exists (the construct-validity
battery has never run), so nothing downstream goes stale. The prototype scratch scripts are not committed.

---

## 8. Fixtures and the missing regression test

1. **`per_action_ltr` (SB360 default) gets a genuinely per-action-LTR fixture** (`make_per_action_ltr_fixture`):
   each frame is in *its acting team's* LTR. Concretely, D's interception frames stay in team-1's LTR
   (team-1 keeper low x), but the team-2 pass frame is the **point reflection** (x→105−x, y→68−y) into
   team-2's LTR (team-2 keeper low x). Arm A + Arm B both score under the default with this fixture. This is
   the fixture the metric's real target actually uses.
2. **`match_ltr` is tested AS A MODE, not as fixture-preservation** (ruling GOAL-SPEC-01(i)). The existing
   match-oriented fixture is retained and its tests pass `frame_convention="match_ltr"` — but framed as
   "this is the continuous-tracking-derived (home-attacks-right) input a tracking provider supplies," which
   is the convention that fixture genuinely represents (not a stand-in for SB360). Add an explicit
   **`test_match_ltr_matches_per_match_resolution` byte-identity golden**: `match_ltr` output equals a
   direct `resolve_defended_goals`-threaded computation, proving the mode reproduces today's per-match
   behavior exactly (§16 R2).
3. **Add the two-sided regression test that was missing** (`test_per_action_frames_score_under_convention`):
   a frame set with **mixed acting teams** (so a per-match `resolve_defended_goals` is bimodal → 0 scored)
   asserts that (a) under `match_ltr` the metric yields all-`unresolved_geometry` on it (the pre-fix
   behavior — a per-match resolution can't orient mixed-acting-team per-action frames), and (b) under
   `per_action_ltr` it scores. This is the two-sided guard (CLAUDE.md "every band needs a test from both
   sides"): it fails on the bug and passes on the fix, and it doubles as a discriminating test that the two
   modes are genuinely different (not the same object).
4. **Scale guards / liveness** update to the fixture's convention (the growth/rows-scanned semantics are
   convention-invariant, so only the `frame_convention` argument threads through).

---

## 9. SB360 audit re-adjudication

`compute_territorial_defense` is a registered SB360 boundary entry (`tests/sb360/_entries/_boundary.py`,
`tests/sb360/_registry.py`). Under the fix it **scores** on the anchor (SB360-shaped) frames instead of
dropping everything, so its machine observation and human verdict move (e.g. `all_nan` →
`differs_by_design` for the Tier-1 lift columns, per ADR-053/ADR-063). Re-run
`tests/sb360/_regenerate.py` + `_adjudicate.py`, re-record, and confirm the `NOT_EXERCISED_BUDGET` and
`_EXPECTED_DARK_COLUMNS` (the IMPL-01 `b_attribution_slippage` entry) still hold — slippage stays honest-NaN
on anonymous SB360 even though the *threat* arms now score, because the defenders remain anonymous.

---

## 10. Prototype evidence (already run)

A scratch prototype (not committed) built `action_ltr_goal_map` per frame and ran Arm A over the real
WC2022 match's defensive-action frames:

```
MATCH 3857254 | Arm-A candidates attempted (>=2 defenders, has actor): 103
OLD (per-match resolve_defended_goals) scored: 0
NEW (per-frame action-LTR)              scored: 103
NEW delta_a: n=103 mean=0.01547 min=0.0 max=0.572 | positive(suppressed): 11, negative: 0
```

Non-negative deltas (11 frames where removing D opens a dangerous zone; 92 where D was not the sole
coverer → exactly 0) — the expected shape for a removal counterfactual on real data.

---

## 11. Value for continuous-tracking providers (`match_ltr`)

A tracking provider (Sportec/GS/SkillCorner) can derive one freeze-frame per on-ball action from continuous
tracking. Those frames are **match-oriented** (home-attacks-right, via `convert_to_frames`), for which
`resolve_defended_goals` — i.e. `match_ltr` — is already correct. Keeping that path:

- gives them a removal counterfactual at the **on-ball action grid** with **no FOV attrition** (all players
  present → no `unresolved_geometry`/`fov_cropped`/`no_actor`), i.e. cleaner than SB360;
- complements (does not duplicate) gkdv/DAS, which are continuous-frame metrics.

Requirement they must meet: **Arm A needs `is_actor`** on the derived frame (identity-exact). They stamp it
from the action's acting player when deriving the frame — a documented input, not a silent assumption.

---

## 12. Testing-discipline lesson (durable)

**A fixture in the wrong coordinate convention passes tests while the real path is broken.** The e2e
fixture was match-oriented, but the metric's only real input is per-action-LTR SB360 — so every unit test
exercised a convention the target never uses, and the metric scored 0 on real data undetected across the
whole build (Tasks 6–9) and its reviews. The guard is: **a fixture must reproduce the target provider's
coordinate convention**, and a metric whose value depends on frame orientation needs a real-data (or
convention-faithful) scoring check, not only a synthetic one. This belongs in CLAUDE.md alongside the
existing "synthetic fixture masked X" precedents.

---

## 13. Out of scope

- **Construct validity of the now-scoring metric** — the owner-run commit-2 battery.
- **Any change to `resolve_defended_goals`** — it stays exactly as-is; `match_ltr` reuses it verbatim.
- **The `is_actor` requirement for Arm A** — retained; documented as a `match_ltr` caller input.
- **Auto-detection of frame convention** — explicitly rejected (§5.1).

---

## 14. Validation plan

1. Unit: the new `per_action_ltr` fixture + the two-sided regression test score; `match_ltr` tests remain
   byte-identical.
2. SB360 audit re-adjudicated + green; conservation, `NOT_EXERCISED_BUDGET`, `_EXPECTED_DARK_COLUMNS` hold.
3. Full CI-faithful suite green; ruff + format clean; pyright adds 0 errors.
4. **Real-data re-validation:** re-run the WC2022 smoke through the *actual* `compute_territorial_defense`
   (default `per_action_ltr`) and confirm the arms score end-to-end and the report conserves. (Scratch,
   not committed.)

---

## 15. Files changed

- `silly_kicks/territorial_defense/_engine.py` — add `action_ltr_goal_map`; branch `classify_arm_a_domain`
  on `frame_convention`.
- `silly_kicks/territorial_defense/_compute.py` — `frame_convention` param + dispatch; per-frame maps in
  `_score_arm_a`/`_score_arm_b`; drop per-match resolve on the default branch.
- `scripts/validate_territorial_defense.py` — per-frame maps in the dose battery.
- `tests/territorial_defense/_fixtures.py` — add `make_per_action_ltr_fixture`.
- `tests/territorial_defense/test_compute.py`, `test_arms.py`, `test_engine.py` — new fixture + the
  two-sided regression test; existing tests pass `match_ltr` where they use match-oriented fixtures.
- `tests/invariants/glossary_emitted_columns.py`, `tests/test_scale_guards.py` — thread `match_ltr`.
- `tests/sb360/` — re-record the `compute_territorial_defense` verdict.
- `docs/superpowers/adrs/ADR-091-*.md` (new), `CLAUDE.md`, `CHANGELOG.md` — the contract + the lesson.

---

## 16. Risks and open questions

- **R1 — Arm B geometry under the new fixture.** Point-reflecting the team-2 pass frame into team-2's LTR
  must keep the pass landing in D's hull and keep the contesting defender removal non-vacuous. Mitigation:
  build the fixture by reflecting the existing rich geometry and assert Arm B scores > 0 with a positive
  delta (non-vacuity), mirroring the current fixture's discipline.
- **R2 — `match_ltr` byte-identity.** Claim: `match_ltr` reproduces today's output exactly. Mitigation: the
  existing match-oriented tests run under `match_ltr` and must stay byte-identical (a golden check).
- **OQ1 — RESOLVED (GOAL-SPEC-02):** `classify_arm_a_domain` and the arms are **convention-agnostic**; the
  compute owns the mode via a `goal_map_for(game, period, acting, opponent)` factory that classify + arms
  call (§5.3). One place holds the `frame_convention` branch.
- **OQ2 — RESOLVED:** `action_ltr_goal_map` accepted (states the convention).
- **GOAL-SPEC-01 — RULED (i), 2026-09-10:** `match_ltr` is a first-class, committed mode with its own
  mode-testing fixture + a byte-identity golden (§8.2), not fixture-preservation. The owner intends to adopt
  it for continuous-tracking-derived frames (§11), so it is genuinely exercised.
