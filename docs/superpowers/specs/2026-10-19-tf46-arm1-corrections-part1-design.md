# TF-46 Arm 1 corrections — Part 1 (converters, events-only) design

> **Status in this release (TF-46 Commit 1; owner decisions 2026-10-31 and 2026-11-03).** This is the approved
> `fcb91c0` text, vendored so the shipped code's section citations resolve. The shipped Part 1 code implements
> it. The recut omits some Part 1 tooling (for example `scripts/regen_tf46_fixtures.py`, which generated the
> committed fixtures at `fcb91c0`). The decision ADRs it names are deferred to the joint release (`TODO.md`).
> The restage plan (`docs/superpowers/plans/2026-10-30-tf46-stoppage-restage-xshot-xcross-refit.md`) holds the
> scope decisions.

**Status:** draft, uncommitted. Revised after review rounds 1–4 (reviewers A and B). Round 4: A APPROVE; B
APPROVE WITH CHANGES — TF46C-SPEC-11 resolved by decision E: the shared helper tolerates an absent signal
column (no raise); the base-schema contract is enforced and tested at `_finalize_output`, so atomic /
hand-built / pre-re-convert frames are not wrongly rejected and the release stays additive (minor).
Awaiting owner-sequenced re-review (round 5).
**Cycle:** TF-46 Arm 1 corrections. Branch `tf-46-arm1-corrections`.
**Part:** 1 of 2. Part 1 is events-only converter work. Part 2 is tracking work (separate spec).
**Release:** Part 1 and Part 2 ship together in one release, one version bump (decision C8).

## 1. Purpose

The shipped corner layer (release 4.139.0) rests on SkillCorner restart rows that are mislabelled at the
converter. This spec corrects the converters, events-only, for every restart type and three providers. It
does not touch tracking. Part 2 adds the tracking kick resolver, the per-corner orientation fix, the
validation registry and the refit.

## 2. Root defect and the key correction

SkillCorner Game Intelligence (GI) records possessions. `X_reception` is the **receiver's** possession,
never the taker's touch. GI does not record the taker's touch for corners, free kicks, throw-ins or goal
kicks. The kick-off is the only exception: GI records the kick-off taker.

Consequence in the shipped converter (`spadl/skillcorner.py`): each restart row is the receiver's
possession relabelled as the set piece. Effects:

- defence-won restarts (`X_against`) are dropped, because the converter keys on `*_for` only
  (`skillcorner.py:353-356`). Missing shares on opendata v3: corner 0.25, free kick 0.21, throw-in 0.16,
  goal kick 0.31;
- a receiver's shot deletes the restart (the `end_type != "shot"` guards on corner and free kick);
- goal kicks are credited to the receiving centre-back, not the goalkeeper;
- the "taker", the kick location and the kick time are the receiver's.

This also explains the ADR-024 (release 4.37.0) misdiagnosis: the SkillCorner goal-kick native origin read
as a "broadcast ball 14–20 m downfield" was the receiver, not a broadcast artefact. `distrust_native_origin`
has no live caller that passes `True` (ADR-121 kept the parameter). This spec amends ADR-024; the code
change for the goal-kick origin is in Part 2.

## 3. Scope

In scope (Part 1):

1. SkillCorner marker-keyed derived restarts for every restart type and side (§5).
2. SkillCorner events-only goals: `match.json` reconciliation, own-goal, missing-goal and penalty-direct
   handling (§6).
3. Sportec F4 (`BallClaiming`) and F7 (free-kick crossed) (§7).
4. StatsBomb K2 (set-piece passes with `Unknown` outcome) and the cross-provider chain convention (§8).
5. The `add_restart_coordinates` derived-row guard and `identify_corners` coherence (§9).
6. Real-data gates and fixtures (§10).
7. Consumers, migration, ADRs, retrains, lakehouse handoff (§11).

Out of scope (Part 2, listed in §13): the tracking kick resolver and `ball_onset`; F3 TF-79 z-gate; F0
per-corner orientation; F5 kick-relative timing; F6 GI falsifier join; R* and `taker_player_id`; the
validation registry and withhold states; `refine_restart_types` (`type_source="tracking"`); the goal-kick
origin code change; the refit and the §12 memo; `attribution_margin`; the census for SkillCorner video
overlap.

## 4. Data tiers and citation rule

Governing rule: ADR-116 reversibility-not-provenance. The only bar for a committed document is
reversibility.

- Citable without qualification: opendata v3 (MIT, 20 matches) and IDSSE (CC-BY-4.0, 7 matches).
- Allowed as non-reversible aggregates, named by data tier: owner-tier SkillCorner corpus (889 matches),
  Gradient Sports, PL. Name the tier, never the source course, module, chapter or instructor.
- Not allowed in committed documents: per-row provider fields that rebuild raw data (coordinates,
  timestamps, per-event native labels, tracking slices).
- Licence is a separate check from reversibility. StatsBomb open-data numbers need StatsBomb attribution
  and owner approval. A `NOTICE` data-licence entry is required only when data is redistributed, not for an
  aggregate.

## 5. SkillCorner marker-keyed derived restarts

### 5.1 Derivation

Use `player_possession` rows only. `on_ball_engagement` rows copy the same GI values and must not seed a
restart.

Emit one derived restart row from each of:

- (a) an interruption marker: `game_interruption_after` in {`corner`, `free_kick`, `goal_kick`,
  `throw_in`} × {`_for`, `_against`}, on the last possession before the stoppage. Exceptions:
  - a marker on the last possession of a period (a half-time carry-over; no restart happened);
  - a `penalty_*` marker (GI penalty precision was 0 of 1, recall 0 of 2; never type a penalty from the
    marker).
- (b) a first-possession tag that the previous possession's marker does not explain. These are chain
  second restarts (including the same type with a different team) and the restarts with no marker.

Tags that never create a restart:

- `goal_*` tags;
- a tag on the first possession of a period;
- a tag on a post-goal kick-off. A post-goal kick-off is the first possession of the conceding team that
  carries the new running score, with `start_type == "unknown"` (§6.3).

### 5.2 Restarting team and opponent (decision C6)

- The restarting team is the marker row's team for `_for`, and the opponent for `_against`.
- The opponent is the other distinct `team_id` among the filtered possessions. Compare ids with
  `silly_kicks.id_compat` (`canonical_id` / `same_id`), never `str()` (a float id gives `"866.0"`, which
  does not equal `"866"`).
- Integrity anchor: `home_team.id` (already required) must be among the event team ids, else raise
  `ValueError` naming `match_metadata["id"]`. Today the mismatch passes silently; the raise is a new
  fail-loud behaviour recorded in the CHANGELOG.
- Optional authority: when `away_team.id` is present, require the event team set to equal
  `{home_team.id, away_team.id}`; when absent, continue. `away_team.id` is not a required key.
- Team-count guard: more than two distinct team ids raises; fewer than two does not derive `_against`
  restarts and counts them unresolved in the report (single-team slices convert today).
- An `_against` row's `team_id` is always a real team id in the same form as the other rows, never NaN.

### 5.3 Receiver rows

Delete the priority-1 restart relabel (`skillcorner.py:351-356`) and the native `is_short` rule
(`skillcorner.py:344-348`). Every possession keeps its own type from `end_type`. A receiver's shot or cross
becomes visible.

### 5.4 Derived-row fields

| Field | Value |
|---|---|
| `type_id` | goal kick: `goalkick`; throw-in: `throw_in`; corner: `corner_crossed` (see `type_source`); free kick: `freekick_crossed` when the first-possession point lies in the restarting team's attacking penalty area (margin m = 0), else `freekick_short`. |
| `type_source` (base column, §11.2) | `native` for goal kicks and throw-ins; `inferred` for the free-kick subtype and for a lost corner (opponent first touch); `default` for a won corner (restarting-team first touch). `<NA>` where no subtype judgement applies. |
| `restart_evidence` (base column, §11.2) | `tagged` / `untagged` / `chain` / `chain_ambiguous` / `tag_only`. `<NA>` on non-derived rows. (`scored_tag` is a goals-reconciliation class, §6.2, not a restart-evidence value; it describes a receiver's scoring shot, which is not a derived restart row.) |
| `team_id` | restarting team (§5.2) |
| `player_id` | NaN (ADR-003) |
| `start_x`, `start_y` | NaN. §9 sets the `add_restart_coordinates` tiers for derived rows. |
| `end_x`, `end_y` | the next possession's start point in the SPADL frame, reflected 180° through `reflection.py` (ADR-045) when the next possession's team differs from the restarting team; NaN for a chain. |
| `time_seconds` | an ordering placeholder: next-possession start − 0.02·(n − k + 1) for position k of n in a chain, guarded so it never crosses the marker possession's end. Not a kick time. |
| `result_id` | for a non-chain restart: success when the next possession belongs to the restarting team, else fail. For a chain restart: per the §5.5 table (not a blanket fail). |
| `result_source` | `native` when the next possession's `start_type` is `*_reception` or `*_interception`; `inferred` when it is `unknown`, untagged, or chained. The closed set stays {`native`, `inferred`, `stopgap`}. |
| `bodypart_id` | `other` for throw-ins (the SPADL convention); `foot` otherwise. |
| `action_provenance` | `derived`. |
| `original_event_id` | the next possession's `event_id`, or the marker's for a chain. |

Coordinate note: `_transform_coords` (`skillcorner.py:55-78`) only rescales; it does not flip. The
converter treats GI as `POSSESSION_PERSPECTIVE` and `to_spadl_ltr` is a no-op for it, so each possession is
already in its own attacking-left-to-right frame. The 180° reflection, applied when the next possession's
team differs from the restarting team, is therefore the whole of the frame change. The attacking-side
sign-flip against tracking is a Part 2 concern.

### 5.5 Chain result rule

A chain is a marker-derived restart whose next possession is itself a restart: it carries a restart tag,
with no possession between. The first restart is the marker row; it takes the result below and
`restart_evidence` `chain` or `chain_ambiguous`. The second restart is a separate derived row (§5.1(b))
with `restart_evidence="tag_only"`, resolved on its own next possession. T is the restarting team of the
first restart.

| Second restart | Awarded to | `result_id` | `restart_evidence` |
|---|---|---|---|
| corner, goal kick or throw-in | T | fail | chain |
| corner, goal kick or throw-in | opponent | fail | chain_ambiguous |
| free kick or penalty | T | success | chain_ambiguous |
| free kick or penalty | opponent | fail | chain |

"success" for the free-kick-to-T case means T kept possession through a foul. It does not mean the delivery
reached its target. The cross-provider convention (§8.2) excludes `chain_ambiguous` rows from delivery and
completion metrics and from the GK-completion training set (G1).

### 5.6 Corner subtype

Every derived corner is `corner_crossed`. The events-only 15 m short rule is dropped: it failed the κ bar on
opendata (κ 0.45, point below 0.50), because the SkillCorner first touch is the next possession, which
skips header touches that are not possessions.

- A lost corner (opponent first touch) is `type_source="inferred"`. The constant-`crossed` call is
  validated: opendata Wilson lower bound 0.86, DFL 0.89, Gradient Sports 0.982 (PFF label).
- A won corner (restarting-team first touch) is `type_source="default"`, because a constant `crossed`
  fails the κ bar for won corners. The measured short rate (opendata 9 %, DFL 17 %, Gradient Sports kept
  corners 0.57 crossed) is stated in the docs. `default` rows are excluded from short/crossed metrics until
  Part 2 sets `type_source="tracking"`.

### 5.7 Free-kick subtype

The derived free-kick "crossed" rule labels a free kick crossed when its first-possession point (the
derived row's end point, §5.4) lies in the restarting team's attacking penalty area, margin m = 0. It ships
with `type_source="inferred"`.

Validation (pre-registered, D-FKbox and D2a): opendata κ 0.68 [0.55–0.81], precision 1.00, recall 0.65 at
m = 0; Gradient Sports κ 0.82 at the selected margin, flat −2 m to +1 m. m = 0 passes on both providers and
needs no tuned constant. A provider-specific margin would need its own pre-registered held-out calibration.
Coverage caveat for the docs: the opendata truth-covered subset is crossed at 0.42, above the uncovered
rate, so the κ may carry selection bias.

### 5.8 Post-processors

- `_add_dribbles` cannot bridge a derived row: a NaN start blocks the inbound side, and the team change
  blocks the outbound side.
- `_derive_end_coordinates` skips the row, because its end is set or NaN, never a placeholder.
- The final combined sort moves to `sort_actions_chronologically` (`skillcorner.py:545`), the ADR-065
  order-insensitive convention with the `_assert_chronological_action_id` guard. The current
  `sort_values(["period_id","time_seconds"])` is already stable (a multi-key lexsort), so this is a
  convention and guard alignment, not a stability fix; the placeholder times make co-timestamped rows more
  likely, which is why the guarded path matters.
- Tests pin each behaviour.

## 6. SkillCorner events-only goals

### 6.1 Truth

Ground truth is `match.json`: per-player `goal` and `own_goal`, plus the team scores. Precondition: per
team, goals plus the opponent's own goals equal the score (fails in 0 of 889 owner matches, 0 of 20
opendata). The GI running score (`team_score` / `opponent_team_score`) places each goal in time and names
the scoring team.

### 6.2 Repair algorithm (G3)

Two passes. Each assignment consumes that player's metadata count.

- Pass 1 over all score changes: `normal`, `scored_tag`, `own_goal_inverted`.
- Pass 2 in time order: `own_goal_surplus`, `swap`, `missing_goal`, `own_goal_unsignalled`.

Classes:

- `normal`: a shot row with `end_type=="shot"` and `game_interruption_after=="goal_for"`, correct scorer.
- `scored_tag`: the scorer's shot carries only `game_interruption_before=="goal_for"`. A normal restart,
  not a chain.
- `own_goal_inverted`: the benefiting team's shot carries `goal_against`; the conceding kick-off carries
  `goal_for`. Kept, expected rare (0 owner, 1 opendata).
- `own_goal_surplus`: recorded as the shooter's goal; `match.json` gives that shooter fewer goals.
- `missing_goal`: no goal row; exactly one T player has a goal deficit and no own-goal candidate exists.
- `own_goal_unsignalled` (new class): fires for a score change favouring T only when all three hold:
  (1) no goal row (no T possession after the previous score change has `end_type=="shot"` +
  `goal_for`, and no tag-only `goal_for`); (2) no goal deficit (no T player has an unassigned metadata
  goal); (3) a unique own-goal candidate (exactly one opponent player has an unassigned metadata
  `own_goal`, counted per player).
- `swap`: one teammate over-credited, another under-credited.
- `ambiguous` (`og`, `og-or-swap`, `ambiguous_missing`): more than one candidate fits. Not repaired.
- `coverage_gap`: the GI final score differs from `match.json`.
- `unresolved`: a GI goal with no metadata goal, or an unlocated goal.

### 6.3 Repaired-row fields and the output marker (decision A2)

Every repaired or inferred row carries a visible marker: `result_source="inferred"` and a `repair_kind`
value equal to the class name. A spec note alone is not sufficient.

- `own_goal_unsignalled` / `own_goal_inverted` / `own_goal_surplus`: emit a `bad_touch` row with
  `result_id="owngoal"` (ADR-018) for the `match.json` own-goal player (conceding team). End = the
  conceding team's goal-mouth centre; start = NaN (decision C2). Time = the end of the last possession in
  T's goal episode carrying `game_interruption_after` in {`goal_for`, `goal_against`}; else the end of the
  last possession before the score change, flagged inferred. Not the kick-off event.
- `missing_goal`: set the unique deficit player's last native shot in the episode to `success`; if there is
  none, emit a derived `shot` row (start NaN, time inside the episode). When the episode follows a
  `penalty_*` marker (144 of 214 owner cases), type the derived row `shot_penalty` and record the penalty
  context on the row.
- `swap`: not repaired at row level; reported.

### 6.4 Goal episode and kick-off anchor

- No fixed time window. Bound each goal's candidate search to the possessions after the previous score
  change (the episode bound). The owner-corpus shot-to-kick-off window reaches a p99 of 233 s and a max of
  632 s, so a fixed window loses goals.
- The post-goal kick-off is the first conceding-team possession that carries the new running score, gated
  on `start_type == "unknown"` only. Drop the distance gate: owner post-goal kick-offs start a median 21 m
  from the centre spot, and only 0.25 pass a 3 m gate, because the broadcast location is noisy. When the
  kick-off is not found, anchor on the score-change event.

### 6.5 Penalties (events-only)

- No penalty is typed from a `penalty_*` marker, and a penalty creates no derived restart row (§5.1
  excludes `penalty_*` markers).
- A direct scored penalty is recovered by the goals reconciliation (§6.2) as a `missing_goal` whose
  episode carries a `penalty_*` marker or a `penalty_against` post-goal kick-off tag. It emits one
  `shot_penalty` row (§6.3), typed by the rule in §6.6. There is exactly one path; no derived restart is
  involved.
- A missed penalty has no events-only signal; Part 2's detector is the only route. This gap is a stated
  known limit.

### 6.6 Restart then goal with no possession between: reading (i), one row

Reading (i) is the typing rule for a goal recovered with no possession between a restart (or penalty
episode) and the score change. It emits one row, never two.

- When a derived restart of type free kick, corner or goal kick is followed by a score change with no
  possession between, retype that single restart row: free kick → `shot_freekick`; corner or goal kick →
  `shot` (SPADL has no corner-shot type).
- A penalty has no derived restart; its `shot_penalty` row comes from the `missing_goal` path (§6.3, §6.5).
- player: the reconciled scorer when exactly one T player is under-credited, else NaN with an unresolved
  gate row;
- start: the constant penalty spot `(94.0, 34.0)` for a penalty (§9.3), else NaN;
- result and source: `success`, `result_source="inferred"`, `repair_kind` set;
- if `match.json` shows an own goal, keep the delivery with `result_id="fail"` and add the ADR-018 own-goal
  row instead.

Reading (ii) (a delivery plus a separate derived shot) needs tracking evidence and is Part 2.

## 7. Sportec F4 and F7

### 7.1 F4 `BallClaiming` (option B)

`BallClaiming` is an annotation, not a touch: 205 of 206 IDSSE events share a timestamp (±0.1 s) with the
same player's event. `claim_type` counts: `InterceptedBall` 146, `BallClaimed` 42, `BallHeld` 17, `Block`
1. The event itself produces no SPADL row today (0 of 206).

> **Amended 2026-10-20 (owner decision, TF46C-IMPL-16 / A1-IMPL-10).** `BallClaimed` now follows the
> `InterceptedBall` rule: it is an outfield ball win, mapped to a derived `interception` (the option that
> keeps the most information). This matches house precedent — Metrica `RECOVERY` → interception, SkillCorner
> `start_type=="recovery"`. Every claim now lands: 206 of 206, `ball_claim_unattached` = 0. The counts below
> are measured on the committed IDSSE fixtures and pinned from the independent oracle
> (`scripts/measure_tf46_census.py`); the earlier "17 / About 129" estimates do not reproduce on the
> fixtures (measured: 12 `InterceptedBall` skip, 134 + 42 = 176 interceptions).

- Add `ball_claim_type` as a dedicated qualifier column (`InterceptedBall` / `BallClaimed` / `BallHeld` /
  `Block`), following the ADR-001 dedicated-column pattern. It is recorded on the same player's same-time
  partner row for `InterceptedBall` / `BallHeld` / `Block`; for `BallClaimed` it is recorded on the derived
  interception row (below), so a consumer keeps ball wins separable from pass interceptions. `BallClaimed`
  is never mapped to `keeper_claim` (0 of 42 are by a goalkeeper). `Block` is a qualifier only (1 event; its
  partner is a clearance, and `shot_blocked` already holds the block signal).
- Emit a derived `interception` row from `InterceptedBall` AND `BallClaimed`, except where the partner row
  is already `tackle` or `keeper_*` (12 of 146 `InterceptedBall`; 0 of 42 `BallClaimed`) — 134 + 42 = 176
  interceptions. Place it just before the partner row, at the partner's timestamp, same player, partner's
  start coordinates (the partner's time, not the claim moment). With no surviving partner (23
  `InterceptedBall` + 7 `BallClaimed`, all from the `OtherBallAction` → `non_action` drop — see the plan's
  open-item note), emit a standalone row at the claim's own time and coordinates.
- Emit a derived `keeper_pick_up` from a `BallHeld` only where no keeper row exists (3 of 17 with
  `goalkeeper_ids`, 4 without). The other 14 already have one; a guard test forbids a duplicate. All 17
  `BallHeld` are by a goalkeeper, so `BallHeld` itself identifies the keeper.
- The existing `Play` `claim` branch (`sportec.py:838`, `play_gk_claim`) stays. No IDSSE data exercises it,
  but other DFL feeds may (Chesterton's Fence).

### 7.2 Sportec schema change (decision B2)

Add `action_provenance` (`native` / `derived`) to `SPORTEC_SPADL_COLUMNS`. Mark the derived F4 rows
`derived`, and mark the `keeper_pick_up` rows already synthesized from `throwOut` / `punt` `derived` for
consistency. The column is additive. §11 lists the consumers.

### 7.3 F7 free-kick crossed

Replace the `freekick_execution_mode` crossed mapping (`sportec.py:880-883`), whose values never occur, with
the DFL `play_penalty_box` qualifier: `freekick_crossed` when `play_penalty_box == true`, else
`freekick_short`. IDSSE value counts: true 48, false 120, missing 4. A missing qualifier (4 cases) stays
`freekick_short`, `type_source="inferred"`, counted in the report. F7 adds no column; it only changes the
`type_id` mapping to read the DFL `play_penalty_box` qualifier. The subtype is the provider flag, not a
derived "delivered into box" judgement; against a box-delivery label the flag scores precision 0.95, recall
0.56. This is the ADR-127 fix class.

## 8. StatsBomb K2 and the cross-provider chain convention

### 8.1 K2 set-piece passes with `Unknown` outcome

The `Unknown` / `Injury Clearance` → `non_action` branch is `conditions[0]` (`statsbomb.py:510-512`) and
wins first in `np.select`. A set-piece pass (`Corner`, `Free Kick`, `Goal Kick`, `Throw-in`) with outcome
`Unknown` therefore becomes `non_action` before the set-piece branches at `conditions[1]-[6]`
(`conditions[7]` is the open-play cross, not a set piece). On the StatsBomb open-data corpus (StatsBomb
open data, attributed) this drops about 1.43 set-piece passes per match.

Fix: evaluate the set-piece type before the `Unknown` guard, so a set-piece pass keeps its set-piece type
regardless of outcome. Set its `result_id` from the next event, and mark the row `result_source="inferred"`
so the shared helper (§11.3) excludes it from the training sets. StatsBomb therefore gains the
`result_source` column (§11.2, decision B). Keep the `non_action` mapping for non-set-piece passes with
`Unknown` / `Injury Clearance` outcome (Chesterton's Fence: the rule removes non-play passes). This affects
`identify_corners`, the restart counts and the `xsuccess` training set on StatsBomb (§11.5).

### 8.2 Cross-provider chain convention (decision B4)

Audit every converter (Opta, Wyscout, Sportec, Gradient Sports, kloppy, Metrica, StatsBomb) for the chain
case "a corner or free kick, then a free kick or penalty to the restarting team". Apply one written
convention, recorded in one ADR: `result_id="success"`, `restart_evidence="chain_ambiguous"`,
`result_source="inferred"`. "success" means the team kept possession through a foul, not that the delivery
reached its target. `chain_ambiguous` rows are excluded from delivery and completion metrics and from the
GK-completion training set (G1) through the shared helper (§11.3). `restart_evidence` and `result_source`
are base-schema columns on every provider (§11.2, decision D), so a chain-case converter sets the values on
its chain rows with no schema change; a converter with no chain data leaves them `<NA>`.

## 9. `add_restart_coordinates` guard and `identify_corners` coherence

### 9.1 Derived-row guard in `resolve_restart_geometry`

The guard keys on `action_provenance == "derived"` on a **restart type**. SkillCorner is the only provider
with derived restart rows in Part 1. Sportec also gains `action_provenance` (§7.2), but its derived rows
are `interception` / `keeper_pick_up`, not restart types, so the restart-geometry guard never reaches them.
A provider whose restart rows are all native is unchanged.

Derived rows skip every tier that samples frames or the next event: origin tier 2a (`tracking_gk`), tier 2b
(`tracking_ball`), the `_side_y` / `_throwin_x` fallbacks, and destination `next_event` / `tracking_ball`.
The row's `time_seconds` is an ordering placeholder, not a kick time, so a frame sample at that time returns
the receiver's position. Part 2's kick resolver adds a validated tier. The enrichment is additive
(`enriched_start_x` / `_y`, ADR-025); it never fills the canonical `start_x` / `start_y`, which stay NaN on
a derived corner, throw-in or free kick.

### 9.2 Enriched origin for derived restart rows (events only)

| Type | Enriched origin | `start_coord_source` |
|---|---|---|
| goal kick | the rule-point (`_RULE_POINT`) | `restart_prior` |
| free kick | unresolved | `unresolved` |
| corner, throw-in | unresolved; never guess the side | `unresolved` |

The `shot_penalty` row from §6.6 is a shot, not a restart type, so it is not an enrichment tier here. Its
canonical start is the constant penalty spot, set in the converter (§9.3).

The `foul_point` tier is not added. It failed its pre-registered bar (opendata median 3.46–3.94 m, p90
10–18 m; the foul-to-kick drift alone exceeds a 5 m p90 on IDSSE). A measured-failing tier does not ship,
not even flagged, because a consumer uses any coordinate it is handed (Hyrum's Law). A derived free kick
with no origin is `unresolved` and counted in the report. Part 2 supplies a validated free-kick origin.

### 9.3 Penalty spot (decision: constant, overrides rider C1)

A derived `shot_penalty` start is the constant penalty spot `_PENALTY_SPOT = (94.0, 34.0)`
(`field_length − 11`, `field_width / 2`). SkillCorner output is `POSSESSION_PERSPECTIVE` and `to_spadl_ltr`
is a no-op, so the acting (penalty-taking) team always attacks +x; the spot is frame-invariant across
period, team and a shoot-out. `GoalMap` is not used: it answers "which goal does a team attack?" for an
absolute frame, a question possession-perspective already removes, and it needs full tracking frames that
the events-only converter does not have. This overrides rider C1 (per-period flip, half-time and shoot-out
flip tests), whose premise holds only for an absolute-frame converter; none emits `shot_penalty` in Part 1.
Test: the `shot_penalty` start equals `(94.0, 34.0)`, including a shoot-out case, proving frame-invariance
rather than a flip that cannot occur.

### 9.4 `identify_corners` coherence

- `corner_side`: a NaN `start_y` currently yields `"right"` without warning
  (`corners/_identify.py:91`). Change the rule so a NaN `start_y` yields a NaN side on every provider.
- A derived corner has a NaN start, so `t_kick`, `kick_x`, `kick_y` and `corner_side` are NaN. `t_kick` is
  NaN, not the ordering placeholder.
- `corner_kind` comes from `type_id`; `type_source` travels from the actions.
- `corner_source`: rename the SkillCorner token `skillcorner_gi` → `skillcorner_marker` in
  `CORNER_SOURCE_VALUES` (decision C7). The repo has no consumer that branches on the value and
  `docs/PRIVATE_CONSUMERS.md` lists none, so this is a minor-version change with a CHANGELOG note. The plan
  runs a caller-sweep to confirm no committed artifact or model-card metadata embeds `skillcorner_gi`
  before the rename.
- Downstream (`add_corner_windows`, encoders) returns NaN outputs with a reason (ADR-042) for a NaN
  `t_kick`, never a crash.
- `identify_corners(gi=)` is deprecated (decision B1): accept it, emit `DeprecationWarning`, ignore it.
  The GI refinement assumed GI's `start_type` was the taker's, which is false. Remove the parameter at the
  next major version.
- SkillCorner corner outputs are NaN until Part 2 resolves kick time, origin and subtype. Because Part 1
  and Part 2 ship together (C8), no release exposes the NaN window.

## 10. Gates, fixtures and tests

### 10.1 Fixtures (decision B3)

Commit slim real fixtures: 20 opendata matches (MIT) and 7 IDSSE matches (CC-BY-4.0, `NOTICE` attribution).
Only the columns the converters read. A generator script rebuilds each fixture byte for byte (the
`_generate_synthetic_match` precedent; the existing `tests/datasets/skillcorner/raw/1886347_events_slim.csv`
is the slim-fixture precedent). No course, PL or Gradient Sports data in a committed fixture (those are
reversible). The StatsBomb G8 gate uses a committed synthetic StatsBomb-shaped fixture plus a thin
`importorskip` e2e on real open data (decision A5(1), the `test_gradientsports_scoreline_e2e` precedent); no
StatsBomb data is committed.

Required fixture cases: own-goal kick-off `2016236`; wrong `throw_in_for` tag `2011166`; penalty kick-off
`2006229`; a half-time carry-over; the 3 `BallHeld` cases with no keeper row; a synthetic reading-(i) case.

### 10.2 Gates

Expected values are pinned constants from the pre-freeze runs (§12). The gate asserts; it does not compute.

| Gate | Asserts |
|---|---|
| G-census restart census | per-type and per-`restart_evidence` derived-row counts match the census. Half-time carry-overs and kick-off tags: 0 rows. (Named G-census, not G1, because "G1" is the GK-completion training set, §5.5/§8.2/§11.3.) |
| G2 no double count | no receiver row carries a restart type; each restart has one derived row. |
| G3 goals | after the two-pass repair, team and player goals equal `match.json`, except for the pinned allow-list (the ambiguous classes and the 3 score-defect matches `2053105`, `2050969`, `2052202`). Zero wrong assignments. |
| G4 chronology | derived times stay inside the marker bracket; `_assert_chronological_action_id` passes. |
| G5 kick-off anchor | the post-goal kick-off is found on `start_type=="unknown"`; the fallback anchors on the score-change event. |
| G6 Sportec F4 | each of the 206 `BallClaiming` events lands on a partner qualifier or a derived row (`ball_claim_unattached` = 0, amended 2026-10-20); the interception (176) and keeper-row (4) counts are pinned from the independent oracle; no duplicate keeper row. |
| G7 Sportec F7 | `freekick_crossed` count equals the `play_penalty_box` true count; the 4 missing-qualifier rows are `freekick_short`, `type_source="inferred"`. |
| G8 StatsBomb K2 | set-piece passes with `Unknown` outcome keep their type; non-set-piece `Unknown` passes stay `non_action`; chain rows follow the convention. |
| G-train | the shared exclusion helper (§11.3), which the G1 GK-completion training filter and the xsuccess training filter call, excludes every row with `result_source` in {`inferred`, `stopgap`}, every `restart_evidence=="chain_ambiguous"` row, and every `repair_kind` row — including the K2 StatsBomb set-piece rows. |

### 10.3 Unit tests

Synthetic tests check each rule from both sides and assert non-vacuity (the rule fired at least once):
marker exclusions; the tag-explained-by-marker logic; each chain-table row; the 180° reflection fired when
the next possession's team differs from the restarting team; the free-kick box rule at m = 0; the
`shot_penalty` constant spot incl. a shoot-out; each repair path with uniqueness met and not met; the C6
opponent resolution, the `home_team.id`-not-in-events raise, the `away_team.id` mismatch raise, a
single-team slice, and a float-typed `team_id`; the derived-row geometry guard with frames present and
absent; the NaN `corner_side` fix on every provider (non-vacuous: it was `"right"` before); a NaN `t_kick`
passing through `add_corner_windows` as a NaN row with a reason; the F4 duplicate-keeper guard; the C2
own-goal row surviving schema validation, VAEP feature computation and xT scoring with a NaN start (xT drop
does not touch `bad_touch`, which is not a move action).

Two tests pin the base-column contract (decision E): (1) every converter emits `result_source`,
`restart_evidence` and `type_source` — the real anti-silent-leak guarantee, enforced by `_finalize_output`
strict projection, asserted per converter; (2) the shared helper tolerates a frame missing the columns
(absence ⇒ no exclusion, no raise), exercised on an atomic-SPADL frame and a hand-built frame.

## 11. Consumers, migration, ADRs

### 11.1 Blast radius

- Base schema (every provider): three new nullable columns `result_source`, `restart_evidence`,
  `type_source` (§11.2, decision D). `<NA>` where a provider or row does not carry the signal. Readers: the
  shared exclusion helper (§11.3) and its callers (`xsuccess`, `expected_passing` (pass only), `gk_decision`,
  the xT-GK completion model, the G1 GK-completion training filter), plus the corner layer (`type_source`)
  and VAEP / xT.
- SkillCorner actions: `_against` restart rows now exist; receiver rows keep their own type; `result_source`
  moves to base; new SkillCorner-specific `repair_kind`; own-goal and missing-goal rows. Readers:
  `identify_corners`, `add_restart_coordinates`, VAEP, xT, xT-GK G1, ghost-GK, ghost-outfield, lakehouse
  marts.
- Sportec actions: new provider-native `action_provenance`, `ball_claim_type`; F4 `interception` /
  `keeper_pick_up` rows; F7 `freekick_crossed` (`result_source` / `restart_evidence` / `type_source` come
  from base). Readers: VAEP, xT, ghost, lakehouse.
- StatsBomb actions: set-piece passes with `Unknown` outcome now kept, typed, and marked
  `result_source="inferred"` (K2; the column is base). Readers: `identify_corners`, restart counts, VAEP,
  `xsuccess`.
- Every converter the §8.2 audit confirms has the chain case: chain rows gain `result_source="inferred"`
  and `restart_evidence="chain_ambiguous"` (base columns; no schema change needed). Readers: `xsuccess`,
  `gk_decision`, the xT-GK completion model, the corner layer, VAEP.

Each changed reader is enumerated both ways in the plan, under the convention that a spec changing a public
seam enumerates every caller (the caller-sweep floor). ADR-050 is the geometry-constant / predicate-shape
contract and is not that rule.

### 11.2 Schema and contract updates

`result_source`, `restart_evidence` and `type_source` are cross-provider signal columns: several
converters emit them with one closed vocabulary, and one shared helper (§11.3) and the corner layer read
them. They go on base `SPADL_COLUMNS`, with `<NA>` where a provider or a row does not carry the signal.
This is the ADR-046 pattern (`shot_blocked` / `cross_blocked` are base nullable columns, `schema.py:28-29`,
`<NA>` where a provider cannot encode). Decision C (round 2, per-converter placement) is reopened and
reversed: the owner waived the re-convert cost that was its only remaining basis, and the "matches existing
layering" premise was false for the base-finalizing converters (StatsBomb, Opta, Wyscout). This is decision
D (round 3). The contrast is deliberate: `action_provenance` and the tackle qualifiers stay provider-native
passthrough on their own constants; a signal a shared consumer reads goes on base.

- Base `SPADL_COLUMNS`: add `result_source`, `restart_evidence`, `type_source`, all nullable string
  (`object` / `<NA>`). Count 16 → 19. Every provider schema spreads `SPADL_COLUMNS`
  (`KLOPPY_SPADL_COLUMNS = {**SPADL_COLUMNS, …}`, `SPORTEC = {**KLOPPY, …}`, `GRADIENTSPORTS` and
  `SKILLCORNER` likewise), so the three columns propagate to every provider with no edit to
  `KLOPPY_SPADL_COLUMNS` (which is an id-dtype family base, not a signal schema).
- `SKILLCORNER_SPADL_COLUMNS`: **remove** `result_source` (it now comes from base; keep the name, dtype and
  vocabulary identical — Hyrum). Add `repair_kind` (SkillCorner-specific goals-reconciliation provenance).
  Keep `action_provenance`.
- `SPORTEC_SPADL_COLUMNS`: add `action_provenance`, `ball_claim_type` only. `type_source`, `result_source`
  and `restart_evidence` come from base.
- No new per-provider schema constant is created. StatsBomb, Opta and Wyscout gain the three columns
  through the base schema they already finalize on.
- Value domains, each nullable with `<NA>` = "not carried":
  - `result_source`: {`native`, `inferred`, `stopgap`}. Populated where the spec classifies the row
    (SkillCorner, Sportec, StatsBomb K2, chain rows). `<NA>` elsewhere. The cycle does **not** back-fill
    `native` across all providers — that per-provider result audit is deferred (owner-approved, §13).
  - `restart_evidence`: {`tagged`, `untagged`, `chain`, `chain_ambiguous`, `tag_only`}. On derived restart
    rows; `<NA>` on all other rows and providers.
  - `type_source`: {`native`, `inferred`, `default`, `tracking`}. `tracking` is produced only in Part 2 but
    is admitted now. `<NA>` where no subtype judgement applies.
- `reflection.py` registry: move `result_source` from the provider-variant union into the canonical block;
  add `restart_evidence` and `type_source` as invariant (none is geometric). Update the hardcoded counts in
  `test_schema.py` (base 16 → 19) and `test_reflection.py`, and the reflection docstring counts.
- Atomic SPADL drops all three columns (as ADR-046 drops the block columns). The plan names every atomic
  consumer of the shared helper and gives it `preserve_native`, or states that none exists.
- `CORNER_SOURCE_VALUES`: swap `skillcorner_gi` → `skillcorner_marker`; add `type_source` to
  `CORNER_IDENTIFY_COLUMNS`.
- `metric_contracts` and the schema and tracking-schema gates must accept the new columns (a tracking-frame
  schema add trips full-suite-only gates; the plan budgets a CI round).

### 11.3 Shared exclusion helper (decision B5)

One helper decides whether a row is excluded from delivery and completion metrics and from training. It
excludes `result_source` in {`inferred`, `stopgap`}, `restart_evidence=="chain_ambiguous"`, and any
`repair_kind` row; it keeps `<NA>` and tolerates a missing column (absence ⇒ no exclusion). It does **not**
raise on an absent column (decision E, round 4).

The anti-silent-leak property is structural, enforced at the producer, not in the helper. Because the three
signal columns are on base `SPADL_COLUMNS`, strict `_finalize_output` projection (`df[cols]` → `KeyError`,
`utils.py:1727-1728`) already raises if a converter omits one, so any converter output always carries them.
A frame that lacks a column therefore has no marked rows to leak — a row cannot be marked `inferred` or
`chain_ambiguous` in a column that does not exist. A helper raise would duplicate the producer-side contract
in a second place, and would wrongly reject the frames that legitimately lack the columns: atomic SPADL
(§11.2 drops all three), hand-built frames, and pre-re-convert frames. Enforcement stays at the boundary
that owns the schema; the helper reads. The converter emit-contract is tested (§10.3): every converter
emits the three base columns, which is the real guarantee.

`repair_kind` is SkillCorner-specific; the helper tolerates its absence the same way. Every consumer that
reads a result or a completion calls the helper: `xsuccess`, `expected_passing`, `gk_decision`, the xT-GK
completion model, the G1 GK-completion training filter and the corner layer. The plan enumerates every
caller both ways.

### 11.4 Atomic re-convert (ADR-065)

Re-materialize every provider. The three base-schema columns (§11.2, decision D) change every converter's
output schema, so every provider's stored tables must be re-materialized to match — the ADR-046 base-column
consequence the owner accepted when reopening decision C. The `action_id` renumber is an atomic cross-table
migration (ADR-065). The spec and the plan carry an ADR-065 re-convert notice.

### 11.5 Retrains (decision B6)

Run a per-model staleness pre-flight, then retrain the models that read converter output: xshot and xcross
(SkillCorner and IDSSE), the ghost-GK variants (SkillCorner and Sportec), ghost-outfield, and `xsuccess`
(its StatsBomb training set changes under K2; `_model.py:167` trains on every non-`non_action` row, so the
kept set-piece passes enter it, now excluded by the shared helper where their result is inferred). The
corner models are refit in Part 2. Re-run the library's SkillCorner and Sportec validation artifacts. The
retrains, the refit, the §12 memo and the lakehouse handoff run once, after both parts are built (C8).

### 11.6 CHANGELOG and lakehouse

- CHANGELOG entries: the restart semantics correction; the new fail-loud `home_team.id` anchor; the
  `corner_source` rename; the base-schema `result_source` / `restart_evidence` / `type_source` additions
  (16 → 19) and the move of `result_source` off `SKILLCORNER_SPADL_COLUMNS` — note the resulting SkillCorner
  output column-order shift (`result_source` moves from the SkillCorner tail into the base block; same name,
  dtype and vocabulary, Hyrum); the Sportec F4 and F7 changes;
  the StatsBomb K2 change; the retrain triggers; the all-provider ADR-065 re-convert notice.
- A lakehouse handoff for the consumer models (VAEP and the marts): all provider tables change (the base
  schema gained three columns).

### 11.7 ADRs (named, not numbered until commit-prep)

- SkillCorner restart semantics (marker-keyed derivation, receiver-not-taker, every restart type and side).
- ADR-024 amendment (the goal-kick native origin was the receiver, not a broadcast artefact).
- Sportec F7 (the `play_penalty_box` crossed mapping, ADR-127 class).
- Sportec F4 (`BallClaiming` qualifier plus derived interception / keeper rows).
- The cross-provider chain convention (§8.2, all converters).
- The base-schema `result_source` / `restart_evidence` / `type_source` signal columns, citing ADR-046 as
  the precedent (cross-provider signal read by a shared consumer → base nullable, `<NA>` where not carried;
  decisions B and D). It records the reopening of decision C and why.

### 11.8 Documentation fix

Correct the `corner_reception` description that states GI records the taker; it records the receiver. The
text lives in the `identify_corners` / `_apply_gi_refinement` docstring in `corners/_identify.py` and in
the TF-46 corner-offence spec, not in a standalone document.

## 12. Pre-registered bars and measured constants

Bars were fixed before the runs; the runs reported the constants. All three runs passed (correction 0
applied throughout).

| Item | Bar | Result | Decision |
|---|---|---|---|
| Free-kick box rule, SkillCorner (D-FKbox) | κ lower 95 % > 0 and point ≥ 0.50 | κ 0.68 [0.55–0.81] at m = 0, precision 1.00 | ship `inferred`, m = 0 |
| Free-kick box margin, Gradient Sports (D2a) | max κ with point ≥ 0.50, lower 95 % > 0 | κ 0.82, flat −2 m to +1 m | ship m = 0 universal |
| Lost-corner crossed (D2b) | Wilson lower 95 % ≥ 0.85 | PFF 0.982, box 0.928 | lost corner `crossed`, `inferred` |
| Won-corner crossed (D2b) | same | 0.57 crossed | `default` |
| Goal census (D1 / K1) | every goal in exactly one class or reported; zero wrong assignments | 0 wrong; per-player 0.979 owner, 0.984 opendata | ship repair; allow-list |

## 13. Known limitations

- A repeated restart with the same type and team (a corner deflected for a second corner) gives one marker
  and one tag, so one derived row covers two restarts. Part 2's kick resolver reports brackets with two or
  more dead-ball kicks as a diagnostic only.
- When GI records the penalty taker's own possession under a free-kick tag (1 of 20 opendata matches),
  Part 1 emits a derived free kick plus the taker's shot. Part 2's penalty detector merges the two
  (decision B8); events alone cannot.
- A missed penalty has no events-only signal (§6.5). Part 2's detector is the only route.
- The repaired own-goal row's time is not independently validated: `match.json` carries no goal times.
- The free-kick box-rule κ may carry selection bias (§5.7).
- **K2 chain detection is successor-adjacent and best-effort (owner-ratified 2026-10-20, A1-IMPL-13 /
  TF46C-IMPL-21).** The §8.2 `chain_ambiguous` label is set when the next MEANINGFUL event after a
  set-piece-`Unknown` restart — the next event that is not StatsBomb bookkeeping or a `Foul Committed` /
  `Foul Won` — is a Free Kick taken as a pass or a Penalty taken as a shot by the same team. It does not fire
  when the awarded free kick is recorded as a direct shot, nor across an unexpected non-skip event. This is
  label-only: every K2 row is `result_source="inferred"`, so the shared helper excludes it regardless of the
  chain label, and the inferred `result_id` (success when the successor is the same team, else fail) is
  unaffected. This is a documented best-effort limitation; no fuller chain resolver is scheduled (§14 lists
  the Part 2 work and a chain resolver is not among it).
- **Deferred (owner-approved 2026-10-19):** the per-provider `result_source` native back-fill audit. This
  cycle populates `result_source` only where the spec classifies the row (SkillCorner, Sportec, StatsBomb
  K2, chain rows) and leaves `<NA>` elsewhere. Back-filling `native` across all providers needs a
  per-provider result-tier audit and a shared "what counts as native" rule (convention-success rows such
  as SPADL dribble have no provider outcome behind them); it changes no exclusion behaviour this cycle
  (the helper keeps both `native` and `<NA>`), so it is deferred to its own spec → plan → A/B cycle rather
  than guessed here.

## 14. Out of scope — Part 2 (tracking)

F3 TF-79 z-gate (`z_source=="measured"`); the shared kick resolver and `ball_onset`; R* and
`taker_player_id`; F0 per-corner orientation via `GoalMap`; F5 kick-relative timing; F6 GI falsifier join
with a loader read timeout and resume; the validation registry and the withhold states (D2);
`refine_restart_types(actions, frames)` with `type_source="tracking"`, including the penalty row-merge
(B8); the goal-kick origin code change (the ADR-024 amendment's code); the GS extension with GS
`first_contact_source="event"`; the single refit (IDSSE + GS + course-tier, non-reversible, recorded by
tier) and the §12 construct-validity memo; `attribution_margin`; the census for SkillCorner video overlap.

## 15. Decision provenance

Every lettered decision below was ruled by the owner on 2026-10-19 during the brainstorm, except where
marked "author design call" (an author choice the owner did not object to) or "author, overrides C1". The
owner relayed each brainstorm answer from the Deep Learning session, which holds the evidence. This section
exists so the plan and the reviewers can tell an owner-approved choice from an author choice (Hard Rule 8);
nothing here is a deferral or a scope cut.

| ID | Decision | Provenance |
|---|---|---|
| D0 | Two sequenced parts on one branch, each with its own spec, plan, A/B review and PR; events-only first | owner |
| D1(c) | A pure `refine_restart_types(actions, frames)` step rewrites `type_id`, `type_source="tracking"` (Part 2) | owner |
| D2 | Withhold until a truth gate passes; the κ and kick-resolver bars (Part 2) | owner |
| D3 | No video access; no Part 3; SkillCorner tracking outputs withheld; refit IDSSE + GS (+ course tier) | owner |
| D4 | GS `first_contact_source="event"` (Part 2) | owner |
| D5 | Include the bundled retrains and the validation re-runs in the cycle | owner |
| A1–A5 | Section approvals, with the A1 kick-off, A2 penalty-limit / synthetic / visible-marker, A3 F4 / F7, A4 foul-point-below-observed, A5 fixture-set riders | owner |
| B1 | Deprecate `identify_corners(gi=)` now, minor version | owner |
| B2 | Add `action_provenance` to the Sportec schema; mark synthesized keeper rows derived | owner |
| B3 | Commit slim real fixtures (opendata + IDSSE) | owner |
| B4 | Audit the chain convention across all converters, one ADR | owner |
| B5 | One shared metric-exclusion helper | owner |
| B6 | Per-model staleness pre-flight, then retrain the affected models in the release | owner |
| B7 | Commits may split, each a fully-tested coherent state, no trailing fix stream | owner |
| B8 | `refine_restart_types` may merge the derived free-kick row with the taker's shot (Part 2) | owner |
| C1 | Penalty spot by attack direction per period | owner — **overridden** for this converter: SkillCorner output is possession-perspective, so the spot is the constant `(94.0, 34.0)` and `GoalMap` is neither available nor needed (§9.3, review round 1, owner-approved 2026-10-19) |
| C2 | Own-goal end = conceding goal-mouth centre, start NaN | owner |
| C3 | No retention column (retention is derivable once F2 is fixed, which is this release) | owner |
| C4 | Penalty-retake and repeated-restart diagnostics get report counts | owner |
| C5 | SkillCorner `ConversionReport` subclass | author design call, owner did not object |
| C6 | Opponent = the other event `team_id`, with the integrity anchor | owner (DL code-investigated spec) |
| C7 | `corner_source` rename as a minor + CHANGELOG change | owner |
| C8 | Ship Part 1 and Part 2 together, one release | owner |
| C9 | The 4 missing `play_penalty_box` free kicks stay `freekick_short`, `type_source="inferred"` | owner |
| K1 | Owner-corpus goal-signature census before the freeze | owner |
| K2 | Include the StatsBomb `Unknown`-outcome fix and the chain convention | owner |
| K3 | Committed docs may cite StatsBomb numbers with attribution | owner |
| Course (i) | Course-derived aggregates and ids are allowed in committed docs, named by tier (ADR-116) | owner |
| B (round 1) | The exclusion marker mechanism: `restart_evidence` + `result_source`; add `xsuccess` to the retrains | owner-approved 2026-10-19 (review round 1) |
| C (round 2) | Marker placement on per-converter extended schemas, helper tolerates absence | owner-approved 2026-10-19 (round 2) — **reopened and reversed by decision D**, because its "matches existing layering" premise was false (StatsBomb/Opta/Wyscout finalize on base) and the owner waived the re-convert cost that was its only remaining basis |
| D (round 3) | `result_source`, `restart_evidence`, `type_source` go on base `SPADL_COLUMNS` (nullable, `<NA>` where not carried), per the ADR-046 pattern; `type_source` also base; re-convert all providers. (The initial "helper requires them / raises if absent" sub-detail was superseded by decision E.) | owner-approved 2026-10-19 (review round 3; resolves TF46C-SPEC-09) |
| E (round 4) | The shared helper tolerates an absent signal column (no raise); the base-schema contract is enforced and tested at `_finalize_output`, not re-checked in the helper, so atomic / hand-built / pre-re-convert frames are not wrongly rejected and the release stays additive (minor) | owner-approved 2026-10-19 (review round 4; resolves TF46C-SPEC-11) |
| Native-fill audit | the per-provider `result_source` native back-fill is deferred to its own cycle; this cycle leaves `<NA>` where unclassified | owner-approved deferral 2026-10-19 (§13) |
| Drop the 15 m corner rule; `default` for won corners | owner saw D4 and approved the drop; the constant-`crossed` call passes the Wilson bar (§5.6, §12) |
| Drop the `foul_point` tier | pre-registered: the tier failed its bar (D3); the owner saw D3 (§9.2, §12) |
| Pre-registered bars and measured constants | the bars were owner-set; the constants are the D1/D2/D-FKbox run results (§12) |
| BallClaimed → interception (impl round r2) | `BallClaimed` is an outfield ball win, mapped to a derived `interception` carrying `ball_claim_type="BallClaimed"` on the row (precedent: Metrica RECOVERY, SkillCorner `recovery`, kloppy `BallClaiming`→recovery); every claim lands, `ball_claim_unattached` = 0, `interceptions_added` = 176 (134 + 42) | owner-decided 2026-10-20 (TF46C-IMPL-16 / A1-IMPL-10) |
| Skipped-BallClaimed lands as a qualifier (impl round r2) | a `BallClaimed` whose partner is already `tackle`/`keeper_*` records the qualifier on that partner (no derived row), so the claim still lands (0 cases on IDSSE) | owner-approved 2026-10-20 (TF46C-IMPL-19 / A1-IMPL-12) |
| `OtherBallAction` stays `non_action` (P1, impl round r2) | DFL `OtherBallAction` has no sub-type beyond `DefensiveClearance` and no published definition (IDSSE paper); kloppy keeps it GENERIC; non-clearance ones stay `non_action`; only the false "preserved for VAEP context" comment is corrected | owner-decided 2026-10-20 (in-cycle census) |
| ADR-019 `canonical_id` in the F4 helper + oracle (impl round r2) | replace the `player_id.astype(str)` dict-key with `canonical_id` / `canonical_id_series` | owner-approved 2026-10-20 |
| xT-GK helper wiring (Commit 4, impl round r2) | `MarkovPossessionValue.fit` + `EmpiricalTurnoverValue.fit` call the shared exclusion helper (turnover: exclude as example, keep as scan context); `GkRetentionModel` needs none (possession-based label) | owner-approved 2026-10-20 (TF46C-IMPL-20 / -24 / -25) |
| K2 chain-detection scope (Commit 4, impl round r2) | the §8.2 chain label is successor-adjacent and best-effort — next meaningful event (bookkeeping / `Foul Committed`-`Won` skipped), FK-as-pass / penalty-as-shot, within-period; FK recorded as a direct shot is not detected; label-only (exclusion holds via `result_source="inferred"`). Documented in §13. | owner-ratified 2026-10-20 (A1-IMPL-13 / TF46C-IMPL-26 / -27) |
