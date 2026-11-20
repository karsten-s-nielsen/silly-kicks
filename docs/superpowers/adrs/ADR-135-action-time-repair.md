# ADR-135: Each action time is the action's moment, or the row says it is not

| Field | Value |
|---|---|
| **Date** | 2026-11-18 |
| **Status** | Accepted |
| **Deciders** | Karsten Nielsen (owner): rulings R29-R64 of the action-time repair directive (`2026-11-18-action-time-repair-supervisor-directive-C.md` in the owner's shared reviews folder); scope and design delegated to the supervisor |
| **Review record** | independent reviews A and B of the spec (six rounds), the plan, and this implementation (`2026-11-18-action-time-repair-*` in the owner's shared reviews folder) |
| **Amends** | ADR-133 (decision 6, the SkillCorner restart rows), ADR-132 (item 6, the wheel ship rule), ADR-031 (the DFL parser port), ADR-050 (the declared-constant registries), ADR-130 (the SkillCorner stoppage spans) |

## Context

The 4.144.0 release refit (PR-S209) trains xShot, xCross, ghost-GK, ghost-outfield and the corner models on the
staged GS, SkillCorner and idsse actions. Its supervisor found that many staged action times were not the moment of
the action, and that nothing in a row said so:

- A SkillCorner restart was typed onto the first possession after it (51,874 rows in 879 matches); 4.144.0 (ADR-133
  decision 6) moved that row to the end of the possession. A restart whose first possession is the other side's gave
  no row at all (42 % of the corners). The time, the place and the team of these rows were the receiver's.
- idsse event times are the annotators' clock, not the tracking frame; ELASTIC alignment was available and unused.
- The score and phase inputs of the ghost and xCross features keyed on `time_seconds` only, which is
  period-relative, so a first-half goal or set piece reached a second-half query at a smaller clock time.
- Each reader of the score kept its own copy of the goal rule, and the copies differed.

## Decision

1. **`time_source`** is a SPADL column (`object`, never missing) with the closed vocabulary `TIME_SOURCE_VALUES`:
   `"event"` (the provider's event time), `"tracking"` (a frame time: SkillCorner, Gradient Sports), `"elastic"`
   (an idsse event time aligned by ELASTIC), `"kick"` (a SkillCorner restart or penalty with the kick measured in the
   tracking), `"placeholder"` (a derived restart whose time is the receiver's reception minus 0.02 s), `"order_only"`
   (a derived restart with no first possession: its time keeps the row order only) and `"derived"` (a dribble
   between two rows). `is_order_only` and `is_time_unmeasured` (`placeholder` or `order_only`) are the predicates.
   `_finalize_output` refuses a missing or unknown value.
2. **SkillCorner restart and penalty rows.** Each restart of the four types (corner, free kick, goal kick,
   throw-in) and each penalty is a derived row of its own, for the side the marker names, with no player (R40); the
   first possession keeps the type of its own action. The row's place is the rule point of its type, its end the
   first possession's start, its result whether that possession is the taker's side. The stable key of a SkillCorner
   row is `(game_id, original_event_id, action_provenance, restart_kind(type_id))` (spec section 4.18).
3. **`sync_action_times`** is the time step that the loader runs once per match with frames: ELASTIC on the
   `"event"` rows; the corner and penalty kick inside its restart gap (`restart_gaps`) from the ball's rest and
   release; the dribble midpoint; a chronological sort and renumbering. The events-only build is the full build
   before the step (spec section 4.10).
4. **The score and phase lookups take the period** (R51): their callback is `(game_id, period_id, time_seconds)`.
   An `"order_only"` row opens no phase window and ends none (R50).
5. **One goal rule** (R52), `spadl._goals.goal_mask` / `owngoal_mask`: a successful shot, penalty or direct free
   kick, and an own goal on any type (for the other team). The VAEP labels, the win-probability state and the score
   lookup read it; a shoot-out row (period 5) is no goal of the score. The amended commit 2 (R83, R88, R92, R95) extends the readers:
   `add_game_state`, the `goalscore` feature and its atomic twin, the VAEP formula, the coordination event windows, the ELASTIC goal termination (R98),
   the atomic conversion and the match-outcome census. The Gradient Sports converter puts the shoot-out in period 5,
   with no location (R89, R93). Commit 3 (R99-R107) moves five scripts onto the shared rule and makes a period-5 row
   no part of the match in `compute_match_outcome` (shots, goals, team count) and in the win-probability fit and
   state (score, cards, clock); a shoot-out row holds the end state of the match with a leverage of 0.
6. **The SkillCorner running score** (R56): where the provider's `team_score` / `opponent_team_score` are on the
   native rows (the converter's `preserve_native` list), the score lookup reads them, with the new score in effect
   from the native row after the first higher one; a decrease or a missing value falls back to the goal rows.
7. **The penalty rows** (R57, R58): the idsse `Penalty` event is a `shot_penalty` row typed from its nested shot;
   the DFL parser reads the outcome under `Penalty > ShotAtGoal` (R63, an owner-ruled exception to the no-touch rule
   of the event block; the parser now differs from the luxury-lakehouse parser at `0efac60` here). A `Penalty` with a
   nested shot and no outcome (bronze from a parser of 4.144.0 or earlier) is excluded with a warning.
8. **Readers of an action's time** take no window at an `"order_only"` row (R54: the VAEP `"time"` labels, `retains`,
   the turnover anchor, packing, the team-metrics spells, the possession carve-out, the coordination windows) and no
   frame at an unmeasured shot (R58). The producer `run_tracking_features` drops the frame pointer of an unmeasured
   row and blanks every family column on it (R45, R50).
9. **A pre-repair artifact warns** (R55): the ghost-GK, ghost-outfield and xCross contracts declare
   `match_context_revision = 1.0` from a third registry, `REVISION_CONTRACT_KEYS`; an artifact without it loads with
   `PredatesDeclaredConstantWarning`. The tag is bumped by hand. The gate `tests/tracking/test_match_context_revision_gate.py` pins the lookups' outputs on a fixed probe per revision (R67), so a change of those rules fails CI until the tag and its pin move.
10. **The wheel xShot and xCross ship the public arm** (R38, R49): the campaign's `ship_variant` forces it; the
    paired rule's own verdict is recorded as `rule_verdict` / `rule_why` and reported at adoption (bar A17).

## Alternatives considered

| Option | Why not |
|---|---|
| Keep the restart on the first possession, with its time moved to the reception | The place and the team stay the receiver's; 42 % of corners still give no row |
| Infer the taker from the tracking (R40) | The tracking gives no reliable taker in this cycle; a wrong taker is worse than none |
| A per-row confidence instead of a closed vocabulary | A reader needs a rule it can test (`is_order_only`), not a threshold it must choose |
| Keep each reader's goal rule | The copies disagreed on penalties, direct free kicks and own goals |

## Consequences

- Not byte-identical: `score_differential`, `score_diff` and `phase` change on every provider (R51, R52); the
  SkillCorner and idsse action tables change rows and times; the 15 ghost-GK, ghost-outfield and xCross artifacts and
  the corner models are refit (spec section 6).
- The bronze of an idsse match with a penalty gains the outcome columns, which changes its column order; every caller
  of `parse_dfl_events`, the lakehouse bronze among them, sees it at its next upgrade. Stored bronze keeps its old
  rows and warns at conversion.
- A pre-repair ghost-GK, ghost-outfield or xCross bundle warns at load until it is refit.
- The `ragnarok` working tree rewrites the parser's nested-child loop; its rebase must carry the R63 rule.

## Limits

The limits of spec section 9: SkillCorner kicks are measured for corners and penalties only; the place of an
unmeasured restart is a rule point; restart rows have no taker; a restart that a goal follows with no possession
between has no shot row; a direct free-kick shot has no row; the gradientsports throw-in time is not measured; the
running score lags the kick-off reception (median 1.5 s); the revision tag is bumped by hand.
