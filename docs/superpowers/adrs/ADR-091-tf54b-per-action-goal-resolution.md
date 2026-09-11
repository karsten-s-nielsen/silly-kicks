# ADR-091: TF-54b per-action goal resolution for SB360 freeze-frames (`frame_convention`)

| Field | Value |
|---|---|
| **Date** | 2026-09-10 |
| **Status** | Accepted |
| **Deciders** | Karsten Nielsen |
| **Corrects** | ADR-090 (an implicit, undocumented frame-convention assumption in the compute) |
| **Spec** | `docs/superpowers/specs/2026-09-10-tf54b-per-action-goal-resolution-design.md` |

> **Note (4.112.0 commit 2):** `compute_territorial_defense` and `action_ltr_goal_map` were subsequently
> **DEMOTED to experimental** (ADR-090's construct-validity battery found the removal arm
> `instrument_void`) — both are now PRIVATE (`territorial_defense._compute` / `._engine`), no public
> surface. The per-frame goal-resolution decision below still STANDS and is retained in that private code
> for the replacement-ghost redesign; only the import path changed (public → `._compute`/`._engine`).

## Context

ADR-090 shipped `compute_territorial_defense` — a removal-counterfactual valuation of a defender's
positioning on StatsBomb-360 freeze-frames. A pre-commit sanity smoke on a real public WC2022 match
(match `3857254`, Denmark 0-0 Tunisia) via statsbombpy open data found it scores **~0 on real SB360, its
only target data**: every Arm-A/Arm-B frame dropped `unresolved_geometry` (0 / 122 Arm-A, 0 / 187 Arm-B),
`attacked_goal(...)` returned `None` for every team despite 508 GK rows.

**Root cause.** SB360 actions are per-acting-team-LTR (ADR-028: the acting team attacks x=105), and
`shape_snapshots` aligns each freeze-frame to *its* action, so the frames are **per-action-LTR**
(`frame_id == action_id`, one frame per action). `resolve_defended_goals` (ADR-055) is a
per-`(game, period, team)` estimator of the defended end from the **mean GK x** — it assumes a *consistent*
orientation across a match's frames (true for continuous tracking, home-attacks-right). On per-action
frames a team's keeper is **bimodal** (measured std ≈ 47 m on match 3857254: x≈0 in its own actions, x≈105
in the opponent's); the per-match mean lands at midfield and both teams collapse to the same defended end,
so `attacked_goal` hits its "opponent end == own end → None" guard for every team.

The unit tests all passed because their fixtures (`make_e2e_fixture` and the SB360 audit anchor) are
**match-oriented or imbalanced** — a convention/composition the per-match resolver happens to resolve — so
they exercised a path the SB360 target never uses. This is the fixture-masking defect class (§ below).

A per-`(game, period, team)` map **cannot** represent per-action orientation: the same team attacks x=105
in its own actions and x=0 in the opponent's.

## Decision

**1. Resolve the attacked goal PER FRAME from the action-LTR convention.** New public-in-package helper
`silly_kicks.territorial_defense._engine.action_ltr_goal_map(game, period, *, acting_team_id,
opponent_team_id) -> GoalMap`: the acting team of the frame's action attacks x=105 (defends 0); the
opponent defends x=105. This is deterministic, needs no GK, and is **FOV-proof** (a keeperless freeze-frame
still resolves). Prototype on real WC2022 (Arm A): **0 → 103 scored**, non-negative suppression (11
positive, 92 zero, mean 0.015).

**2. It is NOT an ADR-055 fork.** ADR-055 collapsed ten hand-rolled *re-estimations of the same quantity*
(mean-GK-x for match-oriented frames). `action_ltr_goal_map` resolves a **different frame convention** from
**ground truth** (the action-LTR definition), not a re-estimate; it returns a `GoalMap` and routes through
the same real opponent-lookup `attacked_goal`. `resolve_defended_goals` is unchanged and reused verbatim
for the match-oriented mode.

**3. Dual-mode via an explicit `frame_convention` parameter.** `compute_territorial_defense(...,
frame_convention: Literal["per_action_ltr", "match_ltr"] = "per_action_ltr")`. `per_action_ltr` (default,
SB360) builds the per-frame map; `match_ltr` (continuous-tracking-derived, home-attacks-right) uses a
single per-match `resolve_defended_goals` map — byte-identical to the historical behaviour. **No
auto-detection** (ADR-059: a detector must have discriminating evidence, and a match-oriented caller can
also build 1:1 per-action frames): the caller declares the convention. `match_ltr` is a first-class,
committed mode (owner intends to adopt it for tracking providers deriving per-action frames — full
observability, no FOV attrition), tested as its own mode plus a byte-identity golden.

**4. The compute owns the convention; classify + the arms are convention-agnostic.** A single
`goal_map_for(game, period, acting, opponent) -> GoalMap` factory (built once from `frame_convention`) is
threaded into `classify_arm_a_domain` (now takes `goal_map_for`, not `goal_map`), `_score_arm_a`, and
`_score_arm_b`; the arm functions (`arm_a_threat_suppressed`, `arm_b_threat_suppressed`) are unchanged
(they already take a `goal_map`). The owner-run driver `validate_territorial_defense` mirrors this,
hard-coding the `per_action_ltr` factory (SB360-only).

## Consequences

### Positive
- **The metric SCORES on real SB360** (its target data). `unresolved_geometry` now means only "the game
  lacks exactly two teams" (rare), not "GK mean ambiguous" (systematic).
- **GK-independent / FOV-proof.** The SB360 audit's `gk_absent` roster (both keepers removed) now **scores**
  the threat arms (`differs_by_design`, an ADR-063 Tier-1 positional lift) instead of the pre-ADR-091
  `no_signal` (which relied on the GK-based resolver failing). `b_attribution_slippage` stays honest-NaN
  there (anonymous defenders). `NOT_EXERCISED_BUDGET` 52 → 50.
- **The tracking-provider door is genuinely open** (`match_ltr`), preserving the currently-correct per-match
  path rather than regressing it.

### Neutral
- **Conservation identities unchanged** (ADR-042); only the distribution across `scored`/drop-reasons
  changes. **No re-materialize, no VAEP retrain** — the metric is `compute_*`, in no default xfn list, and
  never produced committed output; `match_ltr` output is byte-identical to today.
- Construct validity of the now-scoring metric is **out of scope** here — it is the owner-run commit-2
  battery's job (reported-not-gated).

### Negative / limits
- **Arm A requires `is_actor`** on the frame (identity-exact); a `match_ltr` tracking caller must stamp it
  from the action's acting player. Documented, not a silent assumption.
- The per-action convention is correct only for genuinely per-action frames; a caller that mislabels
  match-oriented frames as `per_action_ltr` gets a wrong orientation. The default matches ADR-090's SB360
  scope; the parameter is the explicit escape.

## Testing-discipline lesson (durable)

**A fixture in the wrong coordinate convention passes tests while the real path is broken.** The e2e and
audit fixtures were match-oriented / imbalanced, but the metric's only real input is per-action-LTR SB360,
so every unit test exercised a convention the target never uses and the metric scored 0 on real data
undetected across the whole build (Tasks 6–9) and its reviews. Guard: a metric whose value depends on
frame orientation needs a **convention-faithful fixture** (here `make_per_action_ltr_fixture`, with the
opponent-action frame point-reflected into its acting team's LTR) and a **two-sided regression** (per-match
resolution → 0 scored on mixed-acting-team frames; per-frame convention → scores), plus a real-data (or
convention-faithful) scoring check — not only a synthetic one. Joins the ADR-032 / ADR-053 fixture-validity
precedents.
