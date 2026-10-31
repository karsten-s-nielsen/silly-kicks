# TF-46 Arm 1 corrections — Part 2 (tracking) design

> **Status in this release (TF-46 Commit 1; owner decisions 2026-10-31 and 2026-11-03).** This is the approved
> `fcb91c0` text, vendored so the shipped code's section citations resolve. Only part of it ships now; the rest
> is deferred to the joint release (`TODO.md`). The restage plan
> (`docs/superpowers/plans/2026-10-30-tf46-stoppage-restage-xshot-xcross-refit.md`) holds the scope decisions.
>
> - **Ships:** §5.1 `DataProfile` and the §15 gate table, which ships EMPTY with its fail-closed lookup; §8 and
>   §17.4 `attribution_margin`; §14.1 F3 (inert while the gate table is empty); the §14.3 F6 library change;
>   §17.2, §17.5 and §17.7 (the SkillCorner converter, Sportec converter and SkillCorner tracking adapter
>   columns; the §17.7 report readers `frame_flips_from_report` / `period_frame_ranges_from_report` are deferred); §25 (Sportec DFL penalties); §26 (SkillCorner GI period-label repair); §27.15 (the GS detection
>   flag) and §27.19 (the stoppage column); and the §18.2 xShot/xCross refit, provenance-stamped only.
> - **Deferred to the joint release:** §6 kick resolver; §7 taker; §9 first contact; §10 gated restart
>   origin; §11 `refine_restart_types`; §12 missed-penalty detector; §13 `identify_corners` wiring and F0;
>   §14.2 F5 and the §14.3 drivers; §15 gate-table derivation; §16 pre-registered constants and pre-flight runs,
>   with the run (9) degradation store and fidelity method of §27; §17.1 (the refinement action columns) and
>   §17.3; §18 except the xShot/xCross refit, including the §18.2.1 censoring (off by default for xShot/xCross;
>   the ghost-GK restart-phase censor is not in the tree); §20 ADRs.
> - **Not vendored:** §23 (the review-round finding log). It stays in `fcb91c0`.

> **Revision 9 (2026-10-21), after review round 8 (A APPROVE; B APPROVE WITH CHANGES; C APPROVE; r8 sha
> 229b5541; no blocker).** Round-8 findings are resolved in §23.8: the rule-(d) goalkeeper exclusion no longer
> swallows a parry (B-53); §11.1 names both retype branches (B-54); `penalty_outcome` has a closed vocabulary
> (B-55, C-60); the one-kick match tolerates a 1–2 frame offset (C-59); one citation corrected.
>
> **Revision 8 (2026-10-21), after review round 7 (A APPROVE; B APPROVE WITH CHANGES; C APPROVE WITH CHANGES;
> r7 sha 9f23cafd; no blocker).** Round-7 findings are resolved in §23.7: the guard pairs on penalty context
> only and lets the gated attribution decide, so a missed penalty followed by a later goal or a rebound is
> handled without depending on a GI taker possession (B-50, C-57); rule (b) is anchored on the release; the
> one-kick match uses time and team, not noisy event locations (B-51); annotation never overwrites a native
> result or type (B-52); rule (d) ignores the goalkeeper's net retrieval (C-58).
>
> **Revision 7 (2026-10-21), after review round 6 (A APPROVE; B APPROVE WITH CHANGES; C REQUEST CHANGES;
> r6 sha 549b9600).** Round-6 findings are resolved in §23.6: the GI sequence rule is bounded to a rebound
> window `W_rb` with a kinematic live-ball test, and a pooled goal / missed / rebound ATTRIBUTION gate now
> guards every derived missed row and the rebound branch (C-55 / B-49, DL fix); one kick is never two rows
> (C-56 / B-46); the guard pairs only a penalty-context goal with a kick-off-followed detection (B-47); the
> rebound retype is an explicit, gated extension with the fabricated spot start removed (B-48).
>
> **Revision 6 (2026-10-21), after review round 5 (A APPROVE; B APPROVE WITH CHANGES; C APPROVE WITH CHANGES;
> r5 sha 441eec29; no blocker).** Round-5 findings are resolved in §23.5: the SkillCorner GI sequence rule now
> covers the whole penalty possession and late in-play balls (C-52 / B-43, DL fix), the consistency guard has a
> pairing key and a rebound branch (C-53 / B-45), and the merge population matches Part 1 (B-44).
>
> **Revision 5 (2026-10-21), after review round 4 (A APPROVE; B APPROVE WITH CHANGES; C REQUEST CHANGES; r4
> sha 3ffae951).** Round-4 findings are resolved in §23.4: the per-profile penalty goal attribution (C-49, DL
> fix), adequacy-based base precedence with GS admitted for `penalty` and `merge` (B-39, C-50), and the Law 14
> rebound correction (B-40).
>
> **Revision 4 (2026-10-21), after review round 3 (A APPROVE; B REQUEST CHANGES; C REQUEST CHANGES;
> r3 sha 1a478e59).** Round-3 findings are resolved or answered in §23.3. Owner rulings (relayed DL): the
> provenance contract for the joint-release retrains (per-dimension masks; frame censoring for occurrence
> labels, §18.2) and the Sportec penalty result marking (`native` when fully determined, new
> `penalty_result_method`, §25.2). DL retracted its "_MAPPED" `BallDeflection` text; `_EXCLUDED_EVENT_TYPES`
> is agreed by A, B and C.
>
> **Revision 3 (2026-10-21), after review round 2 (A APPROVE; B APPROVE WITH CHANGES; C REQUEST CHANGES;
> r2 sha 251b0b6f).** Round-2 findings are resolved or answered in §23.2. The owner ruled design call 6
> (narrowed, sourced from a GI fact), the home of the Sportec penalty fix (this spec, first commit, §25), the
> ratified constants (§16.1) and the second contact base (Gradient Sports per-signal admission; no Metrica).
> The DL answered the eight round-2 follow-ups with measured evidence on all 7 IDSSE matches and the GS World
> Cup 2022 cache. Section numbers are stable from revision 2; §25 is new.
>
> Revision 2 (after round 1, r1 sha 10df5eb1): owner restored five dropped items and relayed the DL designs
> B1–B9. Owner position throughout: scope and breaking changes are not a concern; maximum value, gold standard.

## 1. Purpose

Part 2 adds the tracking-refinement layer over Part 1's events-only SPADL. It resolves kick times, takers,
origins and first contacts from tracking; refines restart subtypes; detects penalties that events cannot see;
fixes corner orientation; and gates every tracking-derived output behind a truth-calibrated validation table.
It also carries one events-only converter fix found in review: the Sportec converter drops the DFL `Penalty`
event (§25). Part 1 and Part 2 ship together in one release, one version bump (C8). The re-convert, the
retrains, the corner-model refit, the §12 memo and the lakehouse handoff run once at that joint release, after
the other session's HF-fanout fit-job refactor merges.

## 2. Scope

In scope:

1. The Sportec DFL `Penalty` mapping, the direct-free-kick pairing, the `BallDeflection` routing (woodwork
   qualifier `shot_woodwork_type`), and the two real-fixture guards (goal reconciliation on every provider with
   committed real fixtures; exact unrecognized-type allow-list) — events-only, the FIRST Part-2 commit (§25).
2. The kick resolver: search bracket, `tf79_contact`, per-type `ball_onset`, the kinematic release reference
   `K`, arbitration (§6).
3. `taker_player_id` with a real taker-identity route (§7).
4. `attribution_margin` on the TF-79 contacts table (§8).
5. First contact R* (F1) with attribution and time, the geographic rule when the taker is unknown, the
   native event-coded first contact (D4), and `first_possession_time` from a GI fact (§9).
6. The validated restart origin, including the goal-kick keeper origin (the ADR-024 amendment's code) (§10).
7. `refine_restart_types`: subtype rewrite, the B8 merge, penalty rows, one renumber point (§11).
8. The missed-penalty detector, signal `penalty` (§12).
9. `identify_corners` wiring, the tracking↔action frame transform, the GI direction check; F0 (§4.3, §13).
10. F3 validated z-quality overhead gate; F5 kick-relative timing; F6 GI joins (drivers and library) (§14).
11. The truth model (§5), the validation gate table and the pooling ladder (§15).
12. The ratified pre-registered constants and the pre-flight runs (1)–(16) as in-repo drivers (§16).
13. The SkillCorner video-overlap census, an ADR-116-safe count per data profile (§16 run (15)).
14. The refit of every bundled corner model, corpus recorded by tier, leave-one-tier-out stability (§18).
15. The joint-release corrections to Part-1 §6.5, §9.2 and §13 text (§18.4).

Out of scope: a video truth GATE (video stays never-a-gate; a video sample is a separate owner decision after
the census); the Metrica sample games as a contact base (withdrawn, §5.3); Arm 2 (parked outside the repo).

## 3. Governing constraints

- **Honest NaN.** A cell that fails its gate ships NaN with a stated reason (`withheld`), never a fabricated
  value. The row is kept (ADR-042 kept-row-with-reason; ADR-003 NaN actor ids).
- **Native labels are never overwritten.** On providers whose events code a field, the native value wins; a
  tracking rule may only flag a disagreement (§11.1). This holds for types, takers, first contacts and merges.
- **Purity and idempotence.** `resolve_kicks`, `detect_penalties`, `refine_restart_types` and every edited
  enricher are pure (ADR-033 `PURITY_ENTRIES`) and idempotent: a second run changes nothing (tested).
- **Reversibility.** Every rewritten row keeps `type_id_events`; every merged row keeps its pre-merge identities
  AND the dropped row's content (§11.4). The pre-refine frame is reconstructible from the post-refine frame
  (round-trip tested, §19). The `action_id` renumber itself is not reversed (downstream tables
  re-materialize, ADR-065).
- **A reported time is never a search parameter.** No output time is read from a bracket bound or a pad
  (§9.4).
- **Fail-closed.** A gate-table key that is absent resolves to `withheld` and is counted.
- **Statistics, one convention everywhere.** Wilson two-sided 95 % (z = 1.96) lower bound. Folds are matches
  (leave-one-match-out, LOMO). A gate passes only when BOTH the Wilson lower bound AND the match-cluster
  bootstrap 2.5th percentile (B = 2000 match resamples) clear the bar, the cell has n ≥ 35 rows and
  m ≥ `m_min` = 5 matches. A Wilson lower bound ≥ 0.90 needs at least 35 rows with zero errors (0.9011 at
  n = 35).
- **Tolerances.** Every tolerance is calibrated on the calibration folds and scored LOMO on held-out
  matches, so a share-within-τ is never scored on the matches that set τ. A calibrated value above its
  ratified ceiling (§16.1) fails that signal for that data profile (withheld). A calibrated τ below the
  measured truth-jitter floor is raised to the floor.
- **ADR-116.** Committed artifacts (the gate-table constant, research aggregates, model cards) carry
  aggregates and data-profile keys only. A data-profile KEY is named by provider, tracking / GI release and
  competition-season, never by an access tier (closes round-1 O1). Corpus records (the refit model card,
  §18.1) name data tiers as Part-1 §14 does (IDSSE, GS, opendata, owner, course-tier aggregate), never a
  course, module or instructor.
- **No local paths in committed docs.** Pre-flight drivers live in-repo; the DL probe scripts are ported,
  not referenced by path.
- **No version or ADR numbers until commit-prep** (joint release).
- **Commit gate (B-28).** Every commit this spec names — the §25 Sportec commit included — is a coherent,
  fully tested state, and lands only after the session shows the owner the exact file list and diff and the
  owner explicitly approves THAT commit. No plan step, green test run or earlier approval substitutes for it.

## 4. Architecture, module map, pipeline

### 4.1 Pipeline

```
[first commit, events-only] Sportec converter: DFL Penalty -> shot_penalty; direct-FK pairing (§25)

SPADL actions (Part 1 + §25) + tracking frames + GoalMap
  -> resolve_kicks            (tracking/_kick_resolver.py)   bracket, t_kick, taker, kick evidence
  -> detect_penalties         (tracking/_penalty_detector.py) penalty detections (no row mutation)
  -> refine_restart_types     (spadl/_refine_restart_types.py) subtype, merges, penalty rows; ONE renumber
  -> add_restart_coordinates  (existing, ADR-025)             gated origin -> enriched_start_x/_y
  -> identify_corners         (existing)                      t_kick, kick_x/_y, corner_side from gated values
  -> add_corner_windows       (existing)                      R* first contact, winner, t_target
```

Every stage is pure and idempotent and reads the static gate table (§15). `resolve_kicks` and
`detect_penalties` add columns only; `refine_restart_types` is the ONLY stage that adds, removes or
renumbers rows.

### 4.2 Modules

New:

- `tracking/_kick_resolver.py` — `resolve_kicks(actions, frames, *, provider, goal_map)`.
- `tracking/_penalty_detector.py` — `detect_penalties(actions, frames, *, provider, goal_map)`.
- `tracking/_restart_validation.py` — `RESTART_GATE_TABLE`, `gate(...)`, `DataProfile`, the column
  constants of §17, the ratified constants of §16.1, and the ADR-056 derivation assertion against the
  committed pre-flight aggregate.
- `spadl/_refine_restart_types.py` (private module, amended 2026-10-22; public as `silly_kicks.spadl.refine_restart_types`) — `refine_restart_types(actions, frames, *, provider, penalties)`.

Edited:

- `spadl/sportec.py` — §25 (first commit).
- `spadl/skillcorner.py` — derived restart rows carry the GI facts `gi_marker_possession_frame_end`,
  `gi_next_possession_frame_start`, `gi_attacking_side` (§6.1 CL-1 amendment, §9.4).
- `tracking/_ball_contact.py` — F3 (§14.1); `attribution_margin` columns on the contacts table (§8).
- `spadl/_training_exclusion.py` — `provenance_masks` (per-dimension) beside the existing
  `training_exclusion_mask`; every training entry point declares `reads=` (§18.2.1).
- The training entry points (xsuccess, expected_passing, xtgk, xshot, xcross, ghost-GK / ghost-outfield where
  action-anchored, the corner models) — exclusion or frame censoring per §18.2.1.
- `tracking/_gk_geometry.py` — `resolve_restart_geometry` (the live ADR-025 engine, :339; the xT-GK v1
  `resolve_gk_geometry` shim was removed by ADR-121) gains the gated origin tiers (§10).
- `corners/_identify.py` — reads the gated `t_kick`, origin and `corner_side` (§13.2).
- `corners/_windows.py` — R* (§9), taker exclusion, `winner_source`, `first_possession_time` (§9.4).
- `corners/_zones.py`, `corners/_encode.py` — F0 (§13.3).
- The §12 battery drivers — F5, F6 (§14.2, §14.3).

### 4.3 The tracking↔action frame transform (B8) and the GI direction check

One transform maps tracking-frame coordinates to the action frame, reusing the existing ADR-028 seam
(`acting_team_attacks_rtl` / `reproject_to_action_ltr`) with the direction from `GoalMap` (ADR-055). For a row
whose relevant team (the restarting team; the taker; the contact team) attacks −x in that period, the
transform is the ADR-045 180° point reflection `(x′, y′) = (105 − x, 68 − y)`; otherwise the identity. Every
tracking read and write in Part 2 goes through it: region tests, origin, corner side, F0. SkillCorner's action
frame is possession perspective (the acting team attacks +x). A direction `GoalMap` cannot resolve (`<NA>`,
ADR-051) withholds every geometric output of that row.

**GI direction check (SkillCorner; B-26 ruling).** Compared per period, not per possession:

- For each team and period, the majority GI `attacking_side` over that team's possessions (at least 5
  possessions) is compared with `GoalMap`. A team-period with fewer than 5 possessions has no majority: its
  geometric outputs are withheld and counted (fail-closed, B-37).
- A possession that disagrees with its period majority is GI noise: withhold the geometric outputs of that
  possession's rows and count them.
- A period whose majority disagrees with `GoalMap`: withhold every geometric output of that period, count it
  and warn. The library does not raise.
- Corpus guard (drivers and validation runs only): raise when mismatched periods exceed 1 % of periods, or when
  any match has both periods mismatched — both patterns indicate a systematic frame or flip bug, not noise.

**Amended 2026-10-23 (owner, DL ruling on the R.2 stop).** R.2 found exactly one mismatched period in every
SkillCorner match (897 of 1794): GI `attacking_side` is in the DELIVERED tracking frame, but the frames (and so
`GoalMap`) are oriented left-to-right by the adapter, and the check compared the two without the flip. The corpus
guard caught it as designed.
- The check maps the GI side through the frames' recorded per-period flip before the comparison: oriented rtl =
  delivered rtl XOR flip. The flip is a declared keyword (`flips` = the adapter's `report.ltr_flip_by_period`,
  keyed `(game_id, period_id)`), the same decide-once / apply pattern as the camera footprint (§17.7). The
  adapter also stamps the APPLIED flip on `frames.attrs` (`"absolute_frame"`: False for every period); the stamp is
  a fallback only, under the `pitch_length` precedence (keyword first), because attrs drop through most pandas
  operations.
- A period with no flip information (no keyword, no stamp, or a recorded `None`) is direction-UNRESOLVED:
  withheld and counted, never compared. Raw GI is never compared against oriented frames.
- The corpus guard also raises when unresolved periods exceed 1 % (a lost flip record would otherwise pass it).
- Audit: `gi_attacking_side` is the only GI value in the delivered frame, and this check is its only reader.
  Every other GI column is a frame id, a time or a label (orientation-free); the SkillCorner SPADL coordinates are
  in the action frame by construction.
- Staging records the flip per match (the `_profile` record), and the drivers pass it explicitly.
- The documented route from the report (C2 delta review, C2-IMPL-28 / B-17): `report.ltr_flip_by_period` is keyed
  by period only, so `skillcorner.frame_flips_from_report(report, output_convention=...)` adds the report's
  `game_id` (`absolute_frame`: False per converted period). A period-keyed mapping passed as `flips` raises a
  `TypeError` that names the helper. Frames written to disk lose the attrs stamp: store the record with them.

## 5. Truth model

### 5.1 Data profile

`DataProfile = (provider, tracking_release, gi_release, competition_season)`. GI labels drift between releases
(corner first-possession κ 0.42), so the GI release is part of the key. Every calibration, gate and
degradation profile is per data profile. Nothing is ever pooled across data profiles. GI / tracking pairing is not
a property of the data profile: it is decided per match on measured alignment (§6.1; amended 2026-10-21, C-11).

### 5.2 Tier 1 — reference truth: WHICH and WHEN

**WHICH kick (the native restart event):**

- IDSSE: `CornerKick` / `FreeKick` / `GoalKick` / `ThrowIn` / `Penalty` at `timestamp_seconds` (`Penalty`
  now reaches SPADL, §25). The DFL `calculated_frame` is NOT used: it is null on 655 of 659 restarts in the
  committed IDSSE fixtures.
- GS (PFF): the OTB game event carrying `gameEvents.setpieceType`. `C` (corner) and `F` (free kick) are
  verified; run (1) confirms the goal-kick, throw-in and penalty codes from the field's enum. The anchor frame
  is the tracking frame whose `game_event_id` equals the event's `gameEventId` (PFF's manual event-to-frame
  sync; the event `startTime` is the same video clock). This anchors WHICH; `K` still gives WHEN.

**Event-time semantics:** run (1) measures, per type, the median and dispersion of `K − t_event`. A median
offset above 1.0 s flags the event time as decision time; such an event is not used as the WHEN anchor.

**WHEN — the kinematic release `K` (C-32; physical, never fitted to event times).** `K` applies the §16.1
release thresholds unchanged to the undegraded 25 fps optical track:

- dead-ball kicks (corner, free kick, goal kick, penalty): rest = the ball within 0.5 m of its rest centroid
  for ≥ 0.5 s; release = the first frame with displacement > max(0.5 m, 3 × the rest-window position SD) AND
  speed > 3 m/s on 2 consecutive frames;
- throw-in (no ground rest): release = the first frame of free flight, ball speed > 4 m/s AND distance to the
  thrower increasing by > 0.3 m over 0.2 s, starting at or behind the touchline.

Run (1) reports a ±50 % sensitivity check on each threshold. The thresholds are never tuned to the event times.

**Amended 2026-10-22 (owner approval of the DL strike rule, C2 re-review; Addendum 2 of the C2 C report).**
WHICH rest window `K` uses:

- Every maximal rest window in the observation window is found. Only windows at the §6.4 rest location for the
  type (action frame, GoalMap direction) qualify, and each gets its own release (own centroid and SD, the thresholds
  above, the §6.4 direction).
- A **strike** is a release whose peak ball speed, from the release to the next rest window (or the window end),
  is ≥ `v_kick` (8 m/s, §16.1). Hand placements and rolling the ball into place stay well below it.
  **Amended again 2026-10-22 (owner, C2 r2 A-07):** the peak must be SUSTAINED on `k_release_frames` (2)
  consecutive samples -- one jitter frame above `v_kick` is not a strike. The strike and the release tests read
  ONE deduplicated speed series (TF-79 `trajectory_dedup_mask`; consecutive samples counted after dedup; rest
  frames are never deduplicated), so a duplicate-frame track still detects its strike. Run (1) varies `v_kick`
  ±50 % and the frame count through `k_release_frames` (1 / 3).
- `K` = the release of the last qualifying rest at or before the FIRST strike in the observation window. A rest
  after the first strike is a post-kick rest and never anchors `K`, whatever the outcome (blocked, saved, cleared).
- No strike in the window: one qualifying rest gives `K` from that rest; two or more give `none`, counted by
  restart type and subtype (fail closed, the §6.3 pattern).
- The rest location is the median over the last 0.5 s of the chosen rest.
- Run (1) guards (pre-registered): `v_kick` ±50 % (4 and 12 m/s); the no-strike `none` rate by subtype under the
  §6.3 B-41 rule (short coverage below crossed coverage by more than 0.10 returns `v_kick` to the owner); the
  `none` rate by outcome class (blocked / not blocked, from the reference events), where a difference above 0.10
  returns the rule to the owner.

**K gate:** on held-out folds, the share of rows with `|K − t_event − median_offset| ≤ τ_event` clears the §3
bar (the dispersion around the per-type median offset is gated; the offset itself is reported). The held-out
residual distribution is the measured τ floor. On a reference profile where the K gate passes,
`t_kick = K` with evidence `reference`.

**Dependency statement:** `K`, `ball_onset` and `tf79_contact` read the same ball track (TF-79 contacts
include `velocity_break` candidates). `ball_onset` is never scored against `K` on the same undegraded track.
Every detector report also gives the COMPOSED error, detector vs the native event time (B-27).

**Location fields (C-30; DFL).** The DFL position fields mean different things and run (1) characterises each
per event type before any location truth is used:

- `X-Position` / `Y-Position`: snapped to the Law point for set pieces (corner flag p90 0.00 m; penalties
  94.0). Used as location truth where run (1) shows it is not a tracking copy (ShotAtGoal `X-Position` lies a
  median 1.1 m from `X-PositionFromTracking`, so it is not a copy there).
- `X-Source-Position`: the INCIDENT location, not the execution spot (penalties: source x 81.9–87.4 while
  `X-Position` = 94.0; free kicks: a median 3.7 m apart). Not used as execution-location truth.
- `X-PositionFromTracking`: tracking-derived; never truth.
- `TacklingGame` has no source position; run (1) tests whether its `X-Position` equals the tracked player
  position. If it does, duels are excluded from location truth and the excluded share is reported.

A location is called "independent" truth only where run (1) shows it. **Fallback (B-29):** if run (1) shows that
a field is a tracking copy for an event type, that event type has NO location truth: its origin and subtype
cells are withheld (never scored against a tracking-derived value). C's round-3 fixture query supports the
characterisation: DFL `X-Position` snaps only on Law-fixed coordinates (corners: 4 distinct values, the flags;
throw-in y: 6 distinct, x: 255; goal kicks: 114 of 119 distinct; free kicks: 172 of 172 distinct).

### 5.3 Tier 2 — transfer truth (SkillCorner and any provider without reference events)

The detectors are tested here: run on tracking degraded to the target profile, compared with `K` and the
native events on the undegraded track.

- **Degradation profile** (one per target data profile, run (9)): frame rate; ball-detection gaps by region
  (corner-flag zone, touchline, goal-kick box) and their length distribution; ball-position noise; the
  provider's ball fill / extrapolation behaviour (SkillCorner ships extrapolated broadcast ball positions,
  `tracking/skillcorner.py:357`, and the library may interpolate, `:383-389` — the degradation reproduces the
  production configuration); the `z_source` mix and estimated-z behaviour; player visibility; identity swaps.
  **Amended 2026-10-21 (DL ruling C-PLAN-19, §17.7):** on a profile whose camera footprint passes the §16.1
  footprint gate, player visibility is reproduced by a camera-follow model (the footprint pose relative to the
  ball, fitted on SkillCorner calibration matches) that masks off-camera objects in the IDSSE / GS tracking;
  fidelity is scored on held-out matches. A profile that fails the gate keeps the current approximation, marked.
  Ball detection, gaps and fill are measured on the carried source ball detection flag (§17.7), never on
  extrapolated positions.

  **Camera-follow model (amended 2026-10-21, DL ruling R3; plan review r4 C-PLAN-20).** Not a median: the model
  stores, per ball-position bin × phase, the EMPIRICAL set of footprint SEQUENCES relative to the ball, one per
  calibration event window. The degradation samples one sequence per event window (the windows are defined by
  "Sequence windows and stitching" below), drawn whole from a single calibration event (seeded) — never per frame, since per-frame draws make the camera flicker and create
  spurious short visibility gaps, which breaks the gap-length fidelity. Phase (shared by fit and scoring):
  restart window = [`kick_bracket_lo`, `t_kick` + 6 s] (SUPERSEDED: the lower bound is now the stoppage s0, see
  "Restart-window boundary" below), with `kick_bracket_hi` in place of `t_kick` when `t_kick`
  is withheld; open play = everything else. A ball-bin × phase cell with fewer than 30 calibration events pools
  neighbouring bins within the same phase (never across phases); the pooling is recorded in the run (9)
  aggregate. Scoring reuses the "player visibility near the ball" tolerance (≤ 0.03), scored held-out separately
  for restart windows; no new constant.

  **Sequence windows and stitching (amended 2026-10-21, DL ruling 1; plan review r5 C-PLAN-22).**
  - Open play: fixed segments of `cam_seg_s` = 10 s. A segment keeps its draw when the ball changes bin.
  - Restart windows: the CORE [`t_kick` − `cam_seg_s`, `t_kick` + 6 s] uses one restart-phase sequence ALIGNED on
    `t_kick` (`kick_bracket_hi` when `t_kick` is withheld), cropped to fit. The part of the window before the core
    (the dead-ball setup, back to the stoppage at `kick_bracket_lo`) is filled with stitched `cam_seg_s` segments
    drawn from the RESTART phase. When a core sequence is shorter than the core, its edge footprint is held for the
    remainder, and the hold is recorded with its duration; a hold over 2 s is reported per profile.
  - Stitching: each new segment, and each restart core, joins the previous segment by taking, from the draws in
    its cell (bin × phase, after pooling), the `cam_stitch_k` = 5 whose first-frame footprint is nearest to the
    previous segment's last-frame footprint, then choosing one of them at random (seeded). Footprint distance =
    the mean distance in metres between matching corners, in ball-relative coordinates.
  - Constants `cam_seg_s` and `cam_stitch_k` join §16.1 with ±50 % sensitivity in run (9).

  **Restart-window boundary = the stoppage s0 (amended 2026-10-21, DL ruling on C-PLAN-24; plan review r7).
  Replaces the R3 phase rule above and the core / setup bounds of ruling 1.** Broadcast camera behaviour changes
  at the stoppage itself (wider shots, cuts, replays), so the boundary is the stoppage time s0 — not the event
  bracket (which is at most ±2 s on native anchors and would discard real pre-kick restart behaviour) and not a
  fixed 10 s before the kick (which labels live play before a quick restart as "restart" and contaminates the
  restart-phase calibration).
  - **s0 per provider (source recorded per row):** (i) SkillCorner derived rows: `kick_bracket_lo` (the marker
    possession `frame_end`); (ii) frames with a ball status (e.g. IDSSE `ball_state`): the first `"dead"` frame of
    the dead run that ends at the kick; (iii) otherwise: the time of the last live-play event before the restart.
  - **Restart phase** = [s0, `t_kick` + 6 s], with `kick_bracket_hi` in place of `t_kick` when `t_kick` is
    withheld. Every other frame is open play. One definition, shared by fit and scoring.
  - **Core** = [max(s0, `t_kick` − `cam_seg_s`), `t_kick` + 6 s]: one restart-phase sequence aligned on `t_kick`,
    chosen by the k-nearest seeded rule, cropped; an edge hold is recorded and a hold over 2 s is reported
    (unchanged).
  - **Setup segments** fill [s0, core start) when s0 < `t_kick` − `cam_seg_s`, drawn from the restart phase and
    stitched (unchanged).
  - **Seam kinds:** `open-setup`, `open-core` (the open → restart join at s0, depending on whether setup segments
    exist), `setup-setup`, `setup-core`, `core-open`, `open-open`. The seam test asserts each kind on fixtures
    that contain it.
  - **Replays:** calibration sequences keep null-footprint spans (e.g. broadcast replays during stoppages), so
    stoppage visibility gaps are reproduced; a null-footprint span makes every object `visibility = False` for
    those frames.
  - **Tests:** a throw-in taken 3 s after s0 → the core starts at s0, no setup segment, an `open-core` seam, and
    the frames before s0 stay open play; a corner with a ±2 s anchor and s0 25 s before the kick → setup segments
    over [s0, `t_kick` − 10 s), a full 16 s core and an `open-setup` seam; the s0 source is recorded per row for
    each of the three sources; a calibration sequence containing a null-footprint span reproduces that gap in the
    applied output.

  **Off-camera position replay (amended 2026-10-21, DL ruling R2; plan review r4 C-PLAN-21).** SkillCorner ships
  EXTRAPOLATED positions for undetected players, and TF-79 attribution, `attribution_margin` and R* read every
  finite position. So the degradation does not only flip visibility: for an off-camera spell of length L in the
  IDSSE / GS frames it samples an end error e from SkillCorner's re-detection jump distribution in the bin of L,
  and displaces the position along a path that grows from 0 at the spell start to e at the spell end; the jump
  direction relative to the player's last velocity is sampled from SkillCorner's empirical angle distribution
  (fitted, not gated). Seeded. Stated limit: SkillCorner reveals the extrapolation error only at re-detection;
  the error path inside a spell is modelled, not observed.

  **Amended 2026-10-26 (CL-42, DL rulings on the degradation defects and the GS role; §27.12, §27.13).** The
  degradation runs in this order: the exact frame-rate grid → the camera chain and mask in the camera frame (one
  physical camera side per match; one whole restart window per restart; the open-play continuation) → the in-view
  misses → the ball gaps (a gap-start probability per detected live frame) → white ball noise matched on the second
  difference → the identity swaps → the off-camera replay. IDSSE is the only base (GS tracking is broadcast-derived).
  Stated limits: (1) SkillCorner in-view player positions carry broadcast error (homography, detection jitter,
  smoothing lag); the degradation does not add it to IDSSE and it cannot be identified from SkillCorner alone, so the
  transfer results for player-position signals (`taker`, `first_contact`, the attribution margin) are optimistic by
  this error (the player d2 report gives a lower bound). (2) The slow part of the SkillCorner ball error (calibration
  drift, smoothing lag) is not identifiable from SkillCorner alone, except at positions the law fixes, where it is
  measured (report; the slow-error study, §27.13).
- **Anchor simulation:** for each base restart, build a SkillCorner-like bracket from the native events
  (lo = the last event before the stoppage; hi = the first touch of the next possession) and run the resolver
  inside it.
- **Fidelity check:** per data profile and per region, each degraded-profile statistic must match the target
  within the §16.3 tolerances. A failed statistic withholds only the signals that depend on it (§16.3
  dependency map); it does not disqualify the whole profile.
- **Bases (Q4 ruling; AMENDED 2026-10-26, §27.13: IDSSE is the only transfer base; GS admits nothing and keeps its
  PFF-label checks as a report).** IDSSE is the contact base. Gradient Sports is admitted as a transfer base PER SIGNAL
  only when GS passes that signal's own tier-1 check on GS truth at the §3 bar:
  - `taker`: the TF-79 kick-contact player against PFF `gameEvents.playerId` on the set-piece event;
  - `kick_time` via `tf79_contact`: `t_contact` within τ of `K`, WHICH anchored at the tracking frame whose
    `game_event_id` equals the event's `gameEventId`;
  - `penalty` (B-39): the §12 detector on GS tracking against PFF penalty events (in-play and shoot-outs),
    pooled per §12;
  - `merge` (B-39, C-50): the §11.4 merge rule against PFF set pieces whose own event is the shot
    (`setpieceType` + shot) and PFF penalties.
  `z_overhead` stays off for GS (§14.1). `first_contact` stays IDSSE-only: the GS failure there is measured
  (team agreement 0.48 with the overhead gate, 0.58 without, against ≈ 0.60 by chance, after the ADR-125
  trajectory dedup). Non-contact signals may use GS only where GS has independent truth: `ball_onset` and
  `kick_time` (K on GS tracking; SUPERSEDED 2026-10-26, §27.13: GS tracking is broadcast-derived, not optical, so
  GS leaves the transfer tier and tier 1 for K / onset / origin geometry), subtype (the `possessionEventType` label,
  §11.2), and origin GEOMETRY ONLY — the corner arc and the throw-in y (the touchline). GS has NO origin truth for the free-kick
  spot, the goal-kick spot or the throw-in x (C-43). The PFF event ball / player arrays are a tracking snapshot
  and are never truth for any signal.
  **Base precedence (B-30, B-39; pre-registered per signal, decided on SAMPLE ADEQUACY before any outcome is
  scored):** (1) IDSSE alone, if its cell meets n ≥ 35 and m ≥ 5; else (2) GS alone, if GS is admitted for that
  signal and its cell meets n and m; else (3) IDSSE and GS combined, only if each base passes the §15.3
  non-inferiority test against the pooled result; else the cell is withheld. The choice depends only on n, m
  and admission — never on which base passes the gate — so there is no best-of-two selection (which roughly
  doubles the false-pass rate), and GS can cover a cell that IDSSE alone cannot reach. For signals whose GS
  truth has the SAME meaning and independence as IDSSE's — `kick_time` via `K`, `penalty`, `merge`, and origin
  geometry — the pre-registered order is "the larger adequate base first" (by n, decided before scoring; C-54),
  which raises power without a best-of-two choice. Contact signals keep IDSSE first; subtype keeps IDSSE first
  (its GS truth is a label, not the library's location meaning). Run (2) reports the GS admission per signal (n, m, Wilson, bootstrap). A signal whose GS
  admission fails stays IDSSE-only; its thin cells stay withheld and are reported.
- **No Metrica base** (withdrawn): 3 matches cannot reach `m_min` = 5 alone and cannot pass the base
  non-inferiority test; the sample games are anonymised, of unknown competition and internally inconsistent.
- `both` is gated here: on a degraded ball track the two detectors can share one error source.

### 5.4 GI is a bracket, never a vote

GI contributes the §6.1 bracket on every SkillCorner row as a hard constraint. GI never votes and never
originates a time.

### 5.5 Resolution order and evidence

A cell ships values from tier 1 if its data profile has reference events and the gate passes; otherwise from
tier 2 if the transfer gate passes; otherwise `withheld`. Every gated output records
`<signal>_evidence ∈ {reference, transfer}`, `<signal>_gate_version` and `<signal>_cell_resolution`
(§15.3). Video is not an evidence tier.

### 5.6 Truth extraction rules (C-01r, C-36)

**GS (PFF) fields:**

| Use | Field |
|---|---|
| WHICH | the OTB game event with `gameEvents.setpieceType` |
| WHICH anchor frame | the tracking frame whose `game_event_id` = the event's `gameEventId` |
| Taker | `gameEvents.playerId` on the set-piece event; cross-check `possessionEvents.passerPlayerId` / `crosserPlayerId` / `shooterPlayerId` |
| First touch | the next event by a different player: actor `gameEvents.playerId`; on a challenge event the actor is `possessionEvents.challengeWinnerPlayerId`; cross-check `touchPlayerId` / `clearerPlayerId` / `keeperPlayerId` / `receiverPlayerId`; time = that event's `game_event_id` frame |
| Subtype | the manual label `possessionEventType` (`CR` crossed vs `PA` short), `crossZoneType` as support. The PFF event ball / player arrays are a tracking snapshot (event ball (−1.43, −0.29) vs tracking (−1.39, −0.29) at a checked frame), so GS has NO independent next-touch location truth. |

**DFL first-touch extraction:**

- Touch event types: `Play`, `OtherBallAction`, `TacklingGame`, `ShotAtGoal`, and a standalone
  `BallClaiming`. Not touches: `Foul`, `Caution`, `Substitution`, `FairPlay`, `Offside`, whistles.
- A corner whose first event after the kick is a `Foul` gets first-contact truth class `foul`: excluded from
  the agreement gate and counted.
- `TacklingGame`: the winner. The DFL event `player_id` equals `tackle_winner` on 1,412 of 1,412 duels. Team
  truth = the winner's team ("who won the first ball", the defence-rate meaning). Player-level agreement is
  not scored on duel rows (a contact attributed to the losing duelist is a physical touch the truth does not
  name); duel rows are reported separately. PFF challenge events use the same rule.
- `Delete`: dropped (536 in the fixtures; annotation records of deleted events, even though they carry player
  ids).
- A `BallClaiming` co-timed (±0.1 s) with another event of the same player: the partner event is the touch.
  A `BallClaiming` with no co-timed partner (1 of 206) counts as the touch.
- Time: `timestamp_seconds`. Location (subtype, IDSSE only): `X-Position` / `Y-Position` per §5.2.

## 6. The kick resolver (`resolve_kicks`)

### 6.1 Search bracket

Every restart row gets `[kick_bracket_lo, kick_bracket_hi]`:

- **Native anchor:** `[t_event − a, t_event + b]`, with `a`, `b` = the p0.5 / p99.5 quantiles of
  `t_event − K`, each clipped to 2.0 s. A row whose kick lies outside the clipped bracket is `none` and is
  counted. The signal fails for that data profile only if more than 2 % of reference rows lie outside.
- **SkillCorner derived row:** `lo` = the marker possession `frame_end` (the stoppage); `hi` = the receiver
  possession `frame_start` (GI `frame_start` maps exactly to the tracking `frame_id`).

  > **Amended 2026-10-21 (owner ruling, relayed DL; plan review r1, CL-1).** The SkillCorner converter parses
  > only the GI `time_start`, truncated to whole seconds (`spadl/skillcorner.py:318-321`) — the truncation behind
  > the F5 offset (a GI possession ≈ 1.1 s early) — and it has no tracking frames. So: the converter emits
  > `gi_marker_possession_frame_end`, `gi_next_possession_frame_start` (nullable `Int64`) on derived restart rows,
  > joined by possession identity; `<NA>` elsewhere, on chain rows with no possession between, and when the input
  > lacks the GI frame columns. `gi_attacking_side` (string) is carried on EVERY row backed by a GI possession,
  > native and derived (amended 2026-10-21, owner ruling, plan review r2 B-11: the §4.3 direction check takes a
  > per-team, per-period majority over ≥ 5 possessions, so it needs every possession). `resolve_kicks` maps
  > the frame ids through `frames` to period-relative times and outputs `kick_bracket_lo`, `kick_bracket_hi` and
  > `gi_next_possession_time`. Missing frame ids → bracket NaN → `kick_time_source="none"`, counted. A frame id
  > absent from `frames` inside the tracking frame range (a frame gap; an out-of-range id fails the pairing Range
  > test below) → bracket NaN → `none`, counted separately as
  > `gi_frame_unmapped`. No whole-second fallback: the GI `time_start` clock never builds a bracket or a reported
  > time. Evidence: GI `frame_start` equals the tracking `frame_id` (corner-validation handoff F5: frame 16691 = the
  > SPADL `t_kick`, 1668.1 s).
  >
  > **Per-match GI / tracking pairing (amended 2026-10-21, owner ruling, relayed DL; plan review r2 C-11; replaces
  > the earlier profile-level release binding).** Frame-id existence cannot fail — tracking frame ids are dense over
  > the match, so GI ids from a different snapshot still map at ≈ 100 % — and mispairing is per match (e.g. one
  > match's GI refreshed). So pairing is decided PER MATCH on measured alignment: `gi_pairing ∈ {paired,
  > unpaired}`. A SkillCorner match is `paired` only when all three tests pass:
  > 1. **Range (precondition):** every GI `frame_start` / `frame_end` lies inside the match's tracking frame range,
  >    in the same period.
  > 2. **Alignment:** (a) proximity — at GI `player_possession` `frame_start` frames where both the possessing
  >    player and the ball are detected, at least `pair_prox_share` = 0.90 have the player within `pair_prox_m` =
  >    2.0 m of the ball; (b) lag — the alignment offset lies within `pair_lag_tol_frames` = 1 frame of 0.
  >
  >    **Amended 2026-10-21 (owner ruling Decision 1 + DL ruling Decision 2; plan review r3 C-PLAN-14 / -15 / -17,
  >    B-15). Replaces the "lag peak" reading of (b):** the share-versus-offset curve is flat after `frame_start`
  >    (the possessor stays near the ball during the carry), so its argmax is noise-placed. The alignment offset is
  >    the offset o ∈ [−`pair_lag_scan_frames`, +`pair_lag_scan_frames`] (= ±20) that maximises the frame-level
  >    agreement between the GI possession intervals [`frame_start` + o, `frame_end` + o] and the frames where the
  >    SkillCorner frame-level possession `sc_possession_player_id` (§17.7) equals the GI possessor. Ties: the
  >    smallest |o|, then the negative o; the number of tied offsets is recorded. A profile whose tracking lacks
  >    the frame-level possession field uses, as its marked fallback, the frames where the GI possessor is within
  >    `pair_prox_m` of a detected ball. The proximity share (a) stays a REQUIRED second test — the physical check
  >    that the two SkillCorner-derived signals cannot fake together. "Detected" is the source ball detection flag
  >    carried through the adapter (§17.7), read through `detected_mask` (ADR-109); an extrapolated ball never
  >    counts. Each GI possession counts once (de-duplicated by its GI `event_id`). Frame-level possession is a
  >    SkillCorner-processed output: it serves this internal-consistency test only and is never the truth for
  >    `taker`, `first_contact`, `kick_time` or any gated signal.
  > 3. **Direction:** the §4.3 GI direction check passes for the match.
  >
  > **Amended 2026-10-23 (owner, DL rulings on the R.2 stop, OD-6).** Replaces the ratified `pair_prox_share` =
  > 0.90, which had no measurement behind it. R.2 found 0 of 895 SkillCorner matches paired; the 20 known-paired
  > opendata matches scored 0.77–0.90 at 0.90.
  > - **Bar rule (pre-registered in the DL ruling before the diagnostic ran):** `pair_prox_share` = the
  >   `pair_neg_quantile` = 0.99 quantile (numpy `linear` method) of the negative-control proximity shares, plus
  >   `pair_prox_margin` = 0.10. `pair_prox_m` stays 2.0 m.
  > - **Negative controls:** every GI possession start shifted by ±5, ±10 and ±30 s (clock-offset failures), and GI
  >   cross-paired with another match (wrong-file failure). The proximity test itself is unchanged: one frame, the
  >   GI `frame_start`, possessor and ball both detected.
  > - **Derived value:** 0.233. Negatives n = 125 (120 time shifts + 5 evaluable cross-pairs; cross-pairing is
  >   mostly not evaluable because the player ids differ), maximum 0.141, p99 0.133. The 20 known-paired matches are
  >   an INDEPENDENT check, not the calibration set: all 20 pass, minimum share 0.773, minimum margin 0.54.
  >   Proximity peaks at offset 0 (median 0.848; offset −3 / +3 frames: 0.285 / 0.716), so GI `frame_start` does
  >   not lead the touch, consistent with `pair_lag_tol_frames` = 1. By GI start type the share within 2 m is 0.89
  >   for `pass_reception` and 0.42–0.58 for interceptions and set-piece receptions (the ball is still arriving).
  > - **Pinned data:** the owner-tier pining staging of the 20 public SkillCorner opendata matches (data profile
  >   `skillcorner|sc_jsonl|gi|2024/2025`), staged dataset generation `7350d9bd4ff7e8ab`, code at `96fb7ea`; a
  >   scratch diagnostic (not an artifact; DL ruling).
  > - **Per-profile guard (run (2)):** every data profile re-derives the bar from its own time-shift negatives
  >   (they need no truth). A profile whose derived bar exceeds the adopted 0.233, or a profile with an evaluable
  >   match share but no negative control, stops the run (`GiPairingBarError`) and returns the bar to the owner with
  >   that profile's data. A profile with no evaluable GI possession start has nothing to pair or guard.
  > - **All verdicts recorded:** `GiPairing` and the run (2) report carry `pair_range_ok`, `pair_proximity_ok`,
  >   `pair_lag_ok` and `pair_direction_ok` (nullable; `<NA>` = not evaluable), evaluated even after an earlier
  >   test fails. The decision order and `failed_test` are unchanged: paired only when all four are true.
  > - **Direction fails closed:** the Direction test fails when the match has no period that was actually compared
  >   (a recorded frame flip AND at least 5 possessions). Zero compared periods is the absence of evidence, not
  >   agreement. A match with at least one compared, agreeing period still pairs; its unresolved periods are
  >   withheld (§4.3) and counted (`n_unresolved_periods`, `n_insufficient_team_periods`).
  >
  > **Amended again 2026-10-23 (owner, DL rulings on the second R.2 stop: lag, range, bar).** After the direction
  > fix, R.2 on `6d998a5` resolved and agreed 1794 of 1794 periods, proximity passed in 895 of 895, and 754 of 895
  > matches paired; 4 known-paired opendata matches failed (3 lag, 1 range).
  > - **Lag:** `pair_lag_tol_frames` = 2, symmetric about 0; the test is never centred on a measured offset (a
  >   profile-wide desync would then pass). The measured alignment offset is systematic: −1 in 760 matches, −2 in
  >   133, 0 in 2 (all profiles alike) — the SkillCorner possession field leads GI by 1–2 frames (100–200 ms at
  >   10 fps), while the physical proximity peaks at offset 0, so it is a definition difference between two
  >   SkillCorner labels, not a sync failure. The ratified ±1 had no measurement behind it; 2 is chosen after
  >   seeing the data, and its protection is independent (proximity peaks at 0). Sensitivity 1 / 3 frames. Run (2)
  >   reports the offset distribution per data profile; a profile median outside [−2, 2] stops the run and returns
  >   the lag rule to the owner (`GiPairingLagError`). Zero interval agreement at every offset is not evaluable
  >   (the all-tie break would otherwise pick offset 0), so a GI file shifted past the scan fails the lag test.
  > - **Range (provider period range; amended 2026-10-25, DL ruling
  >   `D:\Development\_reviews\2026-10-25-tf46-part2-gi-period-ruling-C.md`; supersedes the 2026-10-23 structural edge
  >   rule):** each GI possession's `[frame_start, frame_end]` must lie inside the PROVIDER range of its `period_id`:
  >   the first and last raw tracking frame the provider labels with that period, empty frames included. No frame
  >   margin and no clipping. A possession inside another period's range, outside every range, or partly over its
  >   range's edge fails the match; each violation is recorded with its class (`range_violations`). A GI frame inside
  >   the provider range but absent from the retained frames is not a Range failure; the rows that read it are
  >   withheld through `gi_frame_unmapped`. Without a supplied provider range the test uses the retained range
  >   (stricter) and records `range_basis = "retained"`; R.2 requires `"provider"`. Observed on the
  >   `7350d9bd4ff7e8ab` staging (876,024 possessions): 876,016 inside both ranges; 7 owner-tier possessions labelled
  >   period 1 whose frames are the first frames of period 2 (a GI period-label defect, repaired in the converter,
  >   §26); 1 opendata possession (2007448, frame 28557) inside the raw period-2 range and absent from the retained
  >   frames; 0 partly over any edge. The 2026-10-23 description ("GI coverage running past the tracking at a
  >   whistle") was not verified against a diagnostic and is withdrawn.
  >   The interval agreement of the lag test drops a frame absent from the tracking from its denominator (missing
  >   data, not disagreement); a present frame with no possessor stays a miss (DL ruling 2026-10-25, Q4).
  > - **Bar:** the pre-registered part is the RULE, not the number. Per-profile derived bars on the R.2 staging
  >   (generation `7350d9bd4ff7e8ab`, staged at `6d998a5`): `sc_parquet` 2024/25 0.24407 (n = 417), 2025/26 0.23527
  >   (n = 408), 2023/24 0.23002 (n = 50), opendata 0.23333 (n = 20). Adopted: the maximum, rounded UP to 3 places,
  >   `pair_prox_share` = 0.245 (the DL ruling named 0.244; 0.244 is below the derived 0.24407, so the guard would
  >   fire on the same staging). **Confirmed 2026-10-23 by the DL** (C2 delta-2 review, reviewer C: "the deviation
  >   is correct and the error was mine"; rounding up keeps the ruling's intent, adopt the maximum). One constant, not per-profile bars: no outcome changes (the smallest match share is
  >   0.77) and p99 is unstable on small profiles; a future profile deriving above it returns to the owner.
  > - **Provenance:** every run sidecar records the staging generation (`staging_generation`).
  >
  > An `unpaired` match withholds every bracket-dependent output for THAT match only, counted as `gi_unpaired` with
  > the failed test named. Provenance (the source and snapshot or hash of the GI file and of the tracking file) is
  > recorded per match and reported, but it is not the deciding test: a provenance mismatch with a passing alignment
  > ships and is flagged; a provenance match with a failing alignment is withheld. The constants join §16.1 with
  > ±50 % sensitivity in run (2). Calibration: opendata ships GI and tracking together, so its matches are the
  > known-paired reference; run (2) reports the proximity share and the alignment offset for every match, with
  > match ids on every tier (ADR-116: ids are non-reversible references; C-PLAN-18), and if any
  > known-paired opendata match fails, the thresholds return to the owner as a decision point before any gate is
  > scored.
- **Chain rows:** the bracket lies between the chain members' brackets.

The detectors OBSERVE `[lo − τ, hi + τ]` so a candidate near a boundary is seen, but the strict bracket
`[lo, hi]` is a hard constraint for EVERY arbitration state (C-33): a candidate outside `[lo, hi]` is kept as
evidence in its candidate column but excluded from arbitration, so no state can ship a pad candidate. A derived-row fixture whose kick precedes
the placeholder `time_seconds` is found inside the bracket (test).

### 6.2 Restart regions

Corner: within 10 m of the flag. Goal kick: the penalty area. Throw-in: within 3 m of the touchline. Free
kick, optical profiles: within 3 m of the stationary-window centroid (§6.4). Free kick, SkillCorner: a 5 m
disk around the ball position at the launch contact frame (§6.3). Penalty: within `r` of the pitch-scaled
spot (§12). All region tests run in the action frame (§4.3).

### 6.3 `tf79_contact` — the kick contact (C-31; B-20)

- **Dead-ball kicks, default:** `t_contact` = the LAST contact of the FIRST restarting-team player among the
  IN-REGION contacts after the ball's last entry into the restart region, before any other player's in-region
  contact. The taker's set-up touches (taker = first contact in 84 % of IDSSE restarts) are absorbed into that
  player's sequence; a short-corner or short-throw RECEIVER is a different player and is never taken as the kick
  (the round-2 "last in-region contact" rule took the receiver — a DL relay error, corrected). A SkillCorner
  bracket starts at the stoppage, so the first toucher inside the bracket can be the ball retriever; counting
  only in-region contacts after the ball's last entry into the region removes the retriever (C-41). The
  residual error is guarded by the `taker` and `kick_time` gates: a cell below the §3 bar is withheld (B-32).
- **SkillCorner free kicks — the launch rule (no rest window needed):** `t_contact` = the restarting-team
  TF-79 contact that launches the peak detected-ball speed inside the strict bracket, with launch speed
  ≥ `v_kick` (8 m/s; hand placements and rolling the ball into place stay well below it). If a second launch
  inside the bracket is within 15 % of the peak speed, the launch is ambiguous: `none`, counted.
- Run (2) reports first-contact vs last-contact-of-first-player as a number, the taker split by subtype, the
  mismatch on simulated long brackets, and the launch rule's `none` rate and coverage BY SUBTYPE (B-38, C-44).
  If the short free-kick launch coverage is below the crossed coverage by more than 0.10 (absolute; B-41),
  `v_kick` returns to the owner before the subtype gate is scored (the gate must not run on an outcome-selected
  subset).

### 6.4 `ball_onset` per restart type

`ball_onset` is the detector that applies the §5.2 `K` definitions to the target profile's track (on a degraded
or broadcast track it is a detector, scored in tier 2; on the undegraded optical track it is `K` itself and is
never scored against itself).

| Type | Rest condition | Release |
|---|---|---|
| Corner | within 0.5 m of the rest centroid for ≥ 0.5 s, rest centroid within 1.5 m of the flag arc | the §5.2 dead-ball release, moving into the pitch |
| Goal kick | as above, rest inside the goal area | the dead-ball release, away from the own goal line |
| Free kick | as above, rest anywhere | the dead-ball release, any direction |
| Penalty | as above, rest at the pitch-scaled spot | the dead-ball release, toward the goal |
| Throw-in | no ground rest | the §5.2 throw-in release |

**Amended 2026-10-22 (owner approval of the DL ruling on C2 r2 C-25 / C-22).** The rest locations above are
Law geometry in REAL metres, mapped into the scaled frame with the provider pitch dimensions: the penalty spot,
the goal-area depth and half-width and the corner-arc centre scale with `pitch_length` / `pitch_width` (one shared
helper, `tracking._law_geometry`, used by the resolver, the detector and the run (1) K truth). The goal-kick rest
centroid may lie within one ball radius (Law 2, 0.11 m, canonical `spadlconfig.ball_radius`) of the goal area,
lines included (Law 1). The converters stamp `pitch_width` beside `pitch_length`; the stages take both as declared
keywords, with the attrs as fallback, and a provider already on 105 × 68 needs neither. The resolver report
counts `no_episode` by restart type and subtype. Run (1) reports the signed distance of each reference rest
centroid to its Law boundary (goal area, corner arc, spot); if legal rests lie outside because of tracking noise,
the noise margin returns to the owner with that distribution. No tuned margin now.

On estimated-z broadcast tracking the held or resting ball is often not tracked; there `ball_onset` is not
applicable unless its cell passes its gate, and `t_kick` comes from `tf79_contact` alone. Run (3) reports the
onset rate by subtype (the check against selection on crossed deliveries).

### 6.5 Arbitration

`kick_time_delta = t_contact − t_onset` (signed), computed whenever both candidates exist in the observation
window. Only candidates inside the strict bracket take part in arbitration (§6.1).

| `kick_time_source` | Condition | `t_kick` | `kick_time_resolution` |
|---|---|---|---|
| `kinematic_release` | reference profile, K gate passes | `K` | `single` |
| `tf79_contact` | only `t_contact` was observed, and it lies inside the strict bracket | `t_contact` | `single` |
| `ball_onset` | only `t_onset` was observed, and it lies inside the strict bracket | `t_onset` | `single` |
| `both` | both inside the strict bracket, `|Δ| ≤ τ` | the pre-registered agreement winner (smaller median abs error vs truth, run (2)); amended 2026-10-22: read from the gated `kick_time` entry (`GateEntry.agreement_winner`), never hard-coded | `agreement` |
| the inside candidate's detector (`tf79_contact` / `ball_onset`) | both observed, `|Δ| ≤ τ`, exactly one inside the strict bracket (C-42, B-31) | the inside candidate | `single` |
| `conflict_resolved_<source>` | both observed, `|Δ| > τ`, exactly one inside the strict bracket (B3) | that candidate | `bracket` |
| `conflict_resolved_<source>` | both inside the strict bracket, `|Δ| > τ`, the `(restart_type, sign(Δ))` cell's calibrated rule passes (§15) | the rule's source | `calibrated_rule` |
| `conflict` | both inside the strict bracket, `|Δ| > τ`, unresolved | NaN | `<NA>` |
| `none` | no candidate inside the strict bracket, or an ambiguous launch | NaN | `<NA>` |
| `withheld` | the cell failed every gate | NaN | `<NA>` |

`bracket` resolution is gated per cell in tier 2, where the bracket is simulated (run (5)).

### 6.6 Columns (on actions, restart rows only; `<NA>` elsewhere)

`kick_bracket_lo`, `kick_bracket_hi`, `t_contact`, `t_onset`, `kick_time_delta`, `t_kick`, `kick_time_source`,
`kick_time_resolution`, `kick_time_evidence`, `kick_time_gate_version`, `kick_time_cell_resolution`,
`kick_retake`. The candidate columns `t_contact` / `t_onset` are NaN only when that detector found no candidate
in the observation window; they are kept (as evidence) on `conflict`, `bracket`-resolved and `withheld` rows,
including a candidate outside the strict bracket. `t_kick` is NaN when `conflict`,
`none` or `withheld`.

### 6.7 Retakes

A retake (C-45) requires, all inside the strict bracket: (1) a release; (2) the ball returns to rest (the
§16.1 rest condition, ≥ 0.5 s) with its new rest centroid within `ρ_retake` of the first rest centroid
(penalty 1.0 m; free kick 1.0 m; corner 1.5 m, the same flag's arc; goal kick: the same goal area); (3) a
second release. A second release without a return to rest is not a retake. Throw-ins: not applicable (no
ground rest). A retake sets `kick_retake = true` and are counted in the report;
the resolver never silently takes the first onset. A retake row's `t_kick` follows the arbitration on the LAST
onset, gated by its own `cell_state = retake` (never pooled, §15.3).

**Amended 2026-10-22 (owner approval of the DL strike rule):** (1) and (3) are STRIKES (§5.2 amendment: peak
speed ≥ `v_kick` up to the next rest). A placement nudge is below `v_kick` and never sets `kick_retake`.
`t_kick` follows the last strike of the retake chain. The goal kick's "same goal area" is the goal area itself:
every qualifying goal-kick rest lies in it.

## 7. Taker (`taker_player_id`)

- **Event providers** (IDSSE, GS, StatsBomb, other native rows): `taker_player_id` = the native restart row's
  `player_id` (for a Sportec penalty, `shot_player` per §25), `taker_source="native"`. No tracking, not gated.
- **SkillCorner derived rows:** `taker_player_id` = the attributed player of the §6.3 kick contact;
  `taker_source="tracking"`.
- Gate signal `taker`: player agreement with tier-1 truth (the DFL restart-event player; PFF
  `gameEvents.playerId`), §3 bar, tier 2 with the player-visibility and identity-swap degradation.
  `cell_state` = restart type × subtype (the short-corner taker is the hard case; run (2) splits it). A failed
  cell ships NaN, `taker_source="withheld"`.
- `taker_known = taker_player_id.notna()` enters the `first_contact` cell state (§9).

## 8. `attribution_margin` (contacts table)

On the TF-79 contacts table (`detect_ball_contacts`), per resolved contact, distances to the ball at the
contact frame:

- `attribution_margin` = d(nearest OTHER-team player) − d(attributed player), metres.
- `attribution_margin_player` = d(nearest other player, any team) − d(attributed player), metres.
- `attribution_other_visible` (boolean): an other-team player is visible in the frame.
- NaN when the contact is unresolved or no other-team player is visible; never `+inf`.
- `contested = attribution_margin ≤` the contested threshold (starts at 0.75 m; recalibration bounded to
  [0.5, 1.0] m, §16.1).
- The corner table carries `first_contact_attribution_margin` for the R* contact.

## 9. First contact R* (B4, F1, D4) and `first_possession_time`

### 9.1 Rule

- **Known taker:** the earliest resolved contact in `(t_kick, t_kick + 6]`, excluding the taker, more than
  10 m from every flag. The half-open TF-79 `first_contact_in_window` `[t0, t1)` is called with `t0` strictly
  after `t_kick` and `t1 = t_kick + 6` included by an explicit boundary check.
- **Unknown taker (geographic rule):** exclude every contact inside the restart region; the window starts after
  the last in-region contact.

### 9.2 Gate

- **Truth (tier 1):** the native first touch extracted by §5.6 — DFL including `OtherBallAction` and
  `BallClaiming` (not Sportec SPADL, which drops non-clearance `OtherBallAction`: defence rate 0.43 from SPADL
  vs 0.61 native); `foul` truth class excluded and counted; duel rows team-only.
- **Metric:** team agreement (and player agreement where the truth codes the player and the row is not a duel)
  AND `|t − t_truth| ≤ τ_fc` (ceiling 0.40 s, §16.1), §3 bar. Touch identity comes from the match rule (the
  first native touch after the kick), not from the tolerance.
- **cell_state:** `contested` / `uncontested` (§8) × `taker_known`. Contested and uncontested are never
  pooled (measured 0.67 vs 0.89). A failed cell ships NaN `winner_team_id`, `winner_player_id`,
  `first_contact_time` with `winner_source="withheld"`.
- Base: IDSSE only (§5.3).

### 9.3 Native event-coded first contact (D4; B-25)

For a data profile whose events code a first touch (GS via the §5.6 PFF fields), the event-coded first touch
is NATIVE PASSTHROUGH: `winner_source="event_coded"`, `first_contact_evidence="reference"`. It is not gated
against itself; its coverage is reported. `winner_source ∈ {tf79_contact, event_coded, withheld, none}` on the
corner table; the name avoids the existing action-level `first_contact_source` (TF-79 detection method,
`CONTACT_SOURCE_VALUES`, `_ball_contact.py:254`).

### 9.4 `first_possession_time` (design call 6, Q1 ruling)

- `gi_next_possession_time` is the `frame_start` time of the GI possession that follows the restart, a GI fact
  independent of how the bracket is built. **Amended 2026-10-21 (owner ruling, CL-1; §6.1):** the converter emits
  the frame id `gi_next_possession_frame_start`, and `resolve_kicks` outputs `gi_next_possession_time` by mapping
  that id through the tracking frames; `first_possession_time` reads that output.
- `add_corner_windows`: `first_possession_time = gi_next_possession_time`.
- Chain rows: `<NA>`, `first_possession_status="chain"` (no possession between the restarts;
  `kick_bracket_hi` there is a bracket bound, not a possession time).
- Native-anchor rows: `<NA>`, `first_possession_status="native_anchor"` (no GI).
- Simple derived rows: `first_possession_status="gi"`.
- `first_possession_time` is never sourced from `kick_bracket_hi`: a change to the τ pad or the chain bracket
  must never change a reported time.
- `add_corner_windows(gi=)` is deprecated (warn and ignore); its exact-float `t_kick` join
  (`_windows.py:74-77, 90-93`) is removed.
- R* moves `t_target` (`t_target = min(t_fc + lead, t_kick + cap)`, `_windows.py:105`); the corner-model refit
  consumes the corrected `t_target` (§18).

## 10. Restart origin (B7, the ADR-024 amendment's code)

### 10.1 Tiers

Implemented inside `resolve_restart_geometry` (`_gk_geometry.py:339`), written to `enriched_start_x/_y` only
(ADR-025); the canonical `start_x/_y` are never written.

1. `stationary_window` — the median ball xy over the §6.4 rest window, where the origin cell passes.
2. `t_kick_ball` — the ball xy at the kick contact frame (at contact the ball is at the foot; the ≈ 2 m per
   frame error appears only after the kick). Used where no rest window exists: throw-ins, and SkillCorner free
   kicks via the §6.3 launch contact frame.
3. Goal kick only: the existing `tracking_gk` tier (keeper detected), re-anchored to the §6.1 bracket instead
   of ±1 s around the placeholder time. The token's meaning widens (C-40); the CHANGELOG and the origin ADR
   record it.
4. The existing `restart_prior` rule point.

`start_coord_source` gains `stationary_window` and `t_kick_ball`; its other values are unchanged (`native`,
`tracking_gk`, `tracking_ball`, `restart_prior`, `tripwire_reverted`, `unresolved`). New `origin_evidence`,
`origin_gate_version`, `origin_cell_resolution`. There is no parallel `restart_origin_source` column. All
coordinates are in the action frame (§4.3).

### 10.2 Origin gate (signal `origin`)

| Type | Tier-1 truth (§5.2 field rules) |
|---|---|
| Corner | the corner arc (both coordinates known within ≈ 1 m) |
| Throw-in | the touchline gives y exactly; the native event `X-Position` gives x |
| Goal kick | inside the goal area (constraint) and the native event `X-Position` / `Y-Position` |
| Free kick | the native event `X-Position` / `Y-Position` (the execution spot; NOT `X-Source-Position`, the incident location) |

Metric per restart type: median error ≤ 2 m AND the share within 5 m clears the §3 bar (the Part-1 foul-point
bar; fixed, never calibrated). Tier 2 applies on degraded tracking. This gates the Part-1 §9.2 promise of a
validated free-kick origin. No pooling (§15.3).

### 10.3 Retiring `distrust_native_origin` (Chesterton's Fence)

The `distrust_native_origin` ladder (:345, :422) exists because the native SkillCorner goal-kick origin sits
14–20 m from the keeper. The 4.37.0 amendment attributed this to a broadcast artefact; the TF-46 re-diagnosis
(the ADR-024 amendment) attributes it to the receiver's possession. Under either diagnosis the native origin is
not the keeper position, so the fence's reason holds. Retire the ladder only after run (10) measures the §10.1
replacement on the same metric — origin error against the detected keeper — and the replacement passes. Until
then the ladder stays.

## 11. `refine_restart_types` — the single row-mutating stage

`refine_restart_types(actions, frames, *, provider, penalties) -> actions'`, pure, idempotent.

### 11.1 Native labels

Rewrite only rows with `type_source ∈ {default, inferred}`. On a native row, set
`type_tracking_disagrees = true` where the tracking rule disagrees; never rewrite.

**Rebound extension (B-48).** A Part-1 `repair_kind` goal row is not native — it rebuilds an event the provider
did not record — but its `type_source` is `<NA>` (Part-1 §5.4), so the rule above does not cover it. Refine may
rewrite such a row in exactly the §12 guard's two `missed` branches — `rebound` and `later_goal` — gated by
the attribution gate, retyping `shot_penalty → shot` with the start handling stated there (B-54). No other
`repair_kind` rewrite is allowed. Sportec's four
`play_penalty_box`-missing free kicks (`type_source="inferred"`) are excluded from tier-1 subtype truth.

### 11.2 Subtype (B5, corrected)

- **Library definition:** `crossed` = the delivery's next touch lies inside the penalty area; `short` otherwise.
- **Tier-1 truth, IDSSE:** the event-coded location of the next touch (DFL `X-Position` / `Y-Position` per
  §5.2, with the run (1) duel rule), inside or outside the penalty area.
- **Tier-1 truth, GS:** the manual label `possessionEventType` (`CR` / `PA`) with `crossZoneType` support
  (the PFF event location is a tracking snapshot, §5.6). This is a LABEL truth, not the library's location
  meaning.
- **Reported separately:** agreement with the native labels (DFL `corner_target_area`, `play_penalty_box`; PFF
  codes) and, on IDSSE, the label-vs-location meaning mismatch.
- **Rule under test:** "ball enters the penalty area inside the window", scored on tier 1 and on tier-2
  degraded tracking against the profile's truth.
- **Gate (per data profile × direction):** per-direction precision (`crossed→short`, `short→crossed`) clears
  the §3 bar AND beats the per-cell majority-class baseline (paired match-cluster bootstrap CI > 0); report κ.
  (A constant `crossed` on WON corners already reaches a Wilson lower bound ≈ 0.81 opendata / 0.63 DFL.) No
  pooling (§15.3).
- A passing cell rewrites `type_id` (`corner_short ↔ corner_crossed`, `freekick_short ↔ freekick_crossed`),
  `type_source="tracking"`, `subtype_evidence`, `subtype_gate_version`; `type_id_events` and
  `type_source_events` hold the prior `type_id` and `type_source` (B-19).

### 11.3 Penalty rows

`refine_restart_types` consumes `detect_penalties` output (§12). On a SkillCorner profile:

- a detected penalty with no score change (missed: saved or off target) whose pooled detection cell passes
  adds a derived `shot_penalty` row: `result_id="fail"`, `action_provenance="derived"`,
  `result_source="inferred"`, `type_source="tracking"`, `penalty_evidence`, `penalty_gate_version`;
- a gated save (behind the pooled saved / off-target split gate, §12) also adds a derived `keeper_save` row
  for the attributed keeper: `result_source="inferred"`, `type_source="tracking"`,
  `action_provenance="derived"` (the SPADL shot + keeper_save convention, so shot-stopping consumers see it);
- a scored penalty adds no row: the Part-1 reconciliation `shot_penalty` already exists, and the two paths must
  agree (`penalty_crosscheck`, §12). The one exception is the §12 rebound branch: the penalty was missed and the
  episode's goal was a rebound, so a derived missed row ships beside the (retyped) goal row (B-48).
- **One kick, one row (C-56, B-46):** before any derived penalty row is emitted — here or in the §12 `rebound` /
  `later_goal` branches (B-56) — refine checks for an existing row that already represents the kick, matched by TIME and TEAM, never by
  event location (SkillCorner event locations are noisy: post-goal kick-offs start a median 21 m from the spot,
  Part-1 §6.4; B-51): a merged B8 row (§11.4); or an attacking-team shot row whose GI possession contains the
  detected release, with a tolerance of τ (0.30 s, the §16.1 ceiling) on both bounds
  (`frame_start − τ ≤ release ≤ frame_end + τ`; a GI set-piece possession can start at the kick touch, and the
  kinematic release and GI `frame_start` can differ by 1–2 frames; C-59) — the mis-tagged taker possession (Part-1
  §13) or GI recording the taker's possession under a penalty tag; or, on a native provider, the native penalty
  shot within the `τ_event` ceiling of the release. The penalty's own window for this match is [rest-window start,
  release + `W`]. Run (12) reports the match accuracy inside the attribution gate. If a matching row exists,
  refine ANNOTATES it instead of adding a row, writing ONLY provenance and outcome columns — `penalty_evidence`,
  `penalty_gate_version`, `penalty_crosscheck`, `penalty_outcome` (closed vocabulary `goal` / `saved` /
  `off_target` / `missed`, the last when the attribution passes but the pooled saved / off-target split gate does
  not; `<NA>` when unattributed; B-55, C-60) — and never its `result_id` or `type_id`
  (a native row is never rewritten, §3; B-52). A derived `keeper_save` row is added only if no keeper action
  already represents the save.

Because every refine-created row carries `result_source="inferred"`, the shared training-exclusion helper
excludes them (B-23). On native providers the detector never adds rows: their events code penalties natively
(Sportec since §25).

### 11.4 B8 merge

- **Population (B-44):** a SkillCorner derived FREE-KICK restart row followed, with no possession between, by a
  restarting-team shot possession. A penalty never creates a derived restart row (Part-1 §6.5, §6.6); the
  penalty case here is the Part-1 §13 mis-tag, where GI records the penalty taker's own possession under a
  free-kick tag and Part 1 emits a derived free kick plus the taker's shot. The events-first spot form below
  catches that mis-tag; the tracking form also covers a genuine direct free-kick shot. Native providers never merge: StatsBomb, Opta, kloppy and GS
  type the shot by qualifier (one row); Metrica merges in-converter (`metrica.py:494-511`); Sportec gains the
  same in-converter pairing in §25.
- **Events-first rule:** the shot's start lies within `r` of the constant penalty spot `(94.0, 34.0)` and
  within `Δt` of the bracket (§16.1: r ≤ 2.0 m, Δt ≤ 2.0 s; the merge radius tolerates the constant's offset
  from the pitch-scaled spot) → one `shot_penalty` row. Events carry the shot's player and start.
- **Tracking rule:** for the remaining population, merge when the gated `taker_player_id` equals the shooter
  and the gated origin lies within `r` of the shot start; if either input is withheld, no merge.
- **Truth:** tier-2 simulation from IDSSE / GS native events where the restart event was itself the shot (GS
  direct free-kick shots and penalties). The simulation rebuilds the SkillCorner SHAPE — a derived free-kick row
  (marker-keyed, placeholder time, NaN player and start) followed by the shot — so the merge rule is scored on
  the population SkillCorner actually has (B-44).
- **Gate (signal `merge`):** precision clears the §3 bar; report recall. A cell below n = 35 is withheld
  (expected on the small SkillCorner population; reported). No pooling (§15.3).
- **Merged row (C-34):** `type_id` = `shot_penalty` for the events-first spot form, else the shot's `type_id`;
  `result_id`, `player_id` and the canonical `start_x/_y` = the shot's (canonical coordinates are never
  written from tracking; the gated origin goes to `enriched_start_x/_y` only, §10); `result_source` = the
  shot's; `restart_evidence` = the derived row's; `type_source="inferred"` for the events-first form (no
  tracking is involved, B-36) and `"tracking"` for the tracking rule. `type_id_events` and `type_source_events`
  hold the SHOT's pre-merge `type_id` and `type_source` (C-48), so a `shot → shot_penalty` retype round-trips.
- **Dropped-row content (C-19r, B-19):** the merged row keeps `merged_from_derived_action_id`,
  `merged_from_shot_action_id` (pre-renumber `action_id`s, nullable `Int64`; not `original_event_id`, which
  collides) AND the dropped derived row's content: `merged_derived_type_id`, `merged_derived_time_seconds`,
  `merged_derived_end_x`, `merged_derived_end_y`, `merged_derived_result_id`,
  `merged_derived_result_source`, `merged_derived_restart_evidence`, `merged_derived_type_source`,
  `merged_derived_original_event_id`, `merged_derived_repair_kind`,
  `merged_derived_gi_marker_possession_frame_end`, `merged_derived_gi_next_possession_frame_start`,
  `merged_derived_gi_attacking_side` (B-19; frame-id names per the CL-1 amendment, §6.1). All
  `merged_derived_*` values are the PRE-refine values: refine performs the subtype rewrite and the merge in one
  stage, and the dropped row is captured before any rewrite (B-42). Every other field of a derived row is fixed by the Part-1 derivation
  rules (player NaN, start NaN, `action_provenance="derived"`, the restarting team), so the round trip holds.

### 11.5 One renumber point

All row additions (penalty rows) and removals (merges) happen in this stage, followed by ONE atomic
`action_id` renumber via `sort_actions_chronologically` (ADR-065). Kick, taker and origin columns travel with
their rows; a merged row keeps the derived row's kick columns. Merges and penalty rows are C4 report counts.

## 12. Missed-penalty detector (signal `penalty`)

`detect_penalties(actions, frames, *, provider, goal_map)`, pure, no row mutation.

- **Spot:** 11 m from the goal line in real metres (from the provider's pitch length), converted to the action
  frame. The constant `(94.0, 34.0)` differs from the true scaled spot by up to 0.55 m on 100–110 m pitches, so
  the detector does not use it (the Part-1 `shot_penalty` start constant is unchanged).
- **Detection (C-45; ratified, relayed DL):** every position constant is evaluated on the MEDIAN position over
  the rest window — the last 0.5 s of ball rest before the release (`K` / `t_kick`) — never at the release
  frame, where keepers step forward and outfield players encroach (both tolerated in practice).
  - Ball: within `r` (≤ 1.0 m) of the pitch-scaled spot, stationary ≥ `s` (≥ 1.0 s); onset toward the goal
    (§6.4).
  - Goalkeeper: the defending goalkeeper (identified by the ADR-078 keeper resolver) lies within `d_gk` = 2.0 m
    of the goal line and laterally within the posts ± 1.0 m (`|y − 34| ≤ 4.66 m`). Law 14 requires part of one
    foot on or in line with the line; a tracked body centroid sits 0.3–0.5 m in front of the feet; tracking
    noise adds 0.2–0.5 m.
  - Encroachment: restricted region R = the penalty area ∪ the disk of radius 9.15 m around the pitch-scaled
    spot (the arc). A player encroaches when their median rest-window position lies inside R by MORE than
    `e_enc` = 1.0 m (distance from the point to R's boundary). `k` = 2 = the maximum count of encroaching
    players, counting outfield players of BOTH teams and excluding the taker (the attacking player within 3 m
    of the ball at the release) and the defending goalkeeper. Detection requires count ≤ `k`. Players not
    visible in the rest window are not counted; the visible share is reported per detection.
  - Run (12) reports a ±50 % sensitivity check for every constant.
- **Outcome (B-34; ratified, relayed DL):** `W` = 3.0 s from the release (a driven penalty crosses the line
  0.4–0.6 s after release, a chip ≈ 1.3 s; a parry or woodwork deflection ending in the goal can take 2–3 s;
  after 3 s any goal belongs to the rebound phase), ±50 % sensitivity in run (12).
  - The penalty's outcome is `goal` when the ball crosses the goal line, between the posts and under the bar,
    within `W` of the release, before any contact by a player other than the defending goalkeeper. A goal after
    a goalkeeper touch or a woodwork deflection is still the penalty's goal (the taker is credited).
  - A goal after another player's contact is a separate rebound event and the penalty stays `missed`
    (`saved` / `off_target`). Law 14 (B-40): the taker may play the ball again once ANOTHER player — the
    goalkeeper included — has touched it. So a taker goal after a SAVE is a legal rebound goal, with a score
    change, and the penalty stays `saved`. A taker second touch after the woodwork, with no other player's touch,
    is an offence: no goal is awarded and the events show no score change.
  - **The score change decides:** a `goal` outcome requires a score change in the events for the attacking
    team. Crossing observed: the tracking crossing time must satisfy the rule above. Crossing NOT observed (the
    ball untracked at the line, common on broadcast tracking where estimated z cannot check the bar) — goal
    attribution is per data profile, because profiles record the goal time differently (C-49):
    - **Profiles with a precise event goal time (IDSSE, GS):** `goal` when no non-goalkeeper contact is detected
      between the release and the event goal time, and that goal time lies within `W` + the `τ_event` ceiling
      (0.40 s) of the release; otherwise the goal is a rebound event and the penalty stays `missed`.
    - **SkillCorner (GI):** GI writes the new score at the conceding team's post-goal kick-off (Part-1 §6.4),
      not at the goal (median 81 s late in the DL census), and the reconciliation `shot_penalty` time is only
      episode-bounded, so no event goal time exists. The GI SEQUENCE rule applies: `goal` when (a) the score
      changes for the attacking team; (b) the NEXT GI possession that starts after the release is the
      conceding team's post-goal kick-off, with no intervening possession (a SkillCorner penalty normally has
      no taker GI possession of its own — DL's round-6 opendata check: the possession before the penalty is the
      fouled one, and the next is the conceding kick-off; GI does sometimes record the taker's possession, the
      Part-1 §13 mis-tag — so the rule is anchored on the release, never on a taker possession; C-57); (c) no non-goalkeeper contact with the ball INSIDE
      the field of play is detected over [release, release + `W_rb`] — GI skips non-possession touches (Part-1
      §5.6), so a saved penalty plus a one-touch rebound finish leaves no intervening possession and only this
      check can see it (C-52, B-43); and (d) the ball is not LIVE at any time in [`W`, `W_rb`] after the
      release, where live = detected, inside the field of play and moving at ≥ 3 m/s — EXCLUDING a live segment
      that starts AFTER the ball has entered the goal (crossed the goal line between the posts and under the
      bar): the keeper kicking or throwing the ball out of the net after a goal (C-58). Because the crossing is
      often untracked on SkillCorner, "entered the goal" is tested operationally: the segment's first detected
      ball position, or the last ball detection before the segment, lies beyond the goal line between the
      posts (C-61). The attribution gate bounds the residual error. A segment that starts
      with the goalkeeper's PARRY is NOT excluded — the ball never entered the goal, and a post-save live ball
      is exactly the rebound evidence (d) must see (B-53). `W_rb` = 8.0 s (§16.1). Run (12) reports the
      frequency of these excluded segments.
      The window is bounded on purpose (C-55, B-49): after a real goal, players carry or kick the ball back to
      the centre spot (≈ 1–2 min on SkillCorner before the post-goal kick-off), and those in-field contacts and
      the resting ball must not turn a scored penalty into a rebound. SkillCorner has no dead-ball signal
      (`ball_state="alive"` on every frame, `tracking/skillcorner.py:275-281`), so liveness is kinematic.
      Otherwise the goal is a rebound event and the penalty stays `missed`. The tier-2 SkillCorner simulation
      includes post-save rebound goals AND post-goal ball handling; run (12) reports both and the post-goal
      handling timing relative to the release (the evidence that pins `W_rb`).
    - Tier 2 reproduces each profile's goal-time semantics: the SkillCorner simulation builds the GI-sequence
      evidence from the native events, never a precise goal time (the round-1 C-02 anchor pattern). Run (12)
      reports the GI score-change timing for penalty goals.
  - **Attribution gate (C-55):** the goal / missed / rebound ATTRIBUTION is gated pooled, in tier 2, with the
    target profile's goal-time semantics and a base that includes post-goal ball handling. A derived missed
    `shot_penalty` row ships only when BOTH the pooled detection gate and the attribution gate pass; the rebound
    branch below ships or retypes only when the attribution gate passes, and otherwise resolves as `disagree`.
  - **Consistency guard, pairing on penalty context only (C-53, B-45, B-47, B-50, C-57):** a Part-1
    reconciliation goal row that is `shot_penalty`-typed (Part-1 penalty context, §6.3) pairs with the LAST
    penalty detected in its goal episode. Pairing does not depend on any GI possession (a SkillCorner penalty
    normally has no taker possession). Earlier detections in the episode are retakes (§6.7) or penalties judged
    on their own gates. The gated ATTRIBUTION (rules (a)–(d) above, or the event-goal-time rule on IDSSE / GS)
    then decides whether the episode's goal is the penalty's:
    - attribution `goal`: `penalty_crosscheck="agree"`; no derived row, no retype;
    - attribution gate does not pass: `penalty_crosscheck="disagree"`; no derived row, no retype (counted);
    - attribution `missed` with the gate passing — the penalty was saved or went off target, and the episode's
      goal came later: either a rebound (a non-goalkeeper in-field contact within [release, release + `W_rb`])
      or a later open-play goal (another possession intervened before the goal, B-50). The penalty's missed row
      ships (or an existing kick row is annotated, §11.3) with `penalty_crosscheck="rebound"` or
      `"later_goal"`, and the reconciliation goal row — the goal is not the penalty's, so its Part-1
      `shot_penalty` typing is wrong — Part-1 §6.3 types it `shot_penalty`
      after a penalty marker, with the constant penalty-spot start (Part-1 §6.6) — is retyped
      `shot_penalty → shot` by `refine_restart_types` under the §11.1 rebound extension. On that retype the
      canonical `start_x/_y` is set to NaN (the rebound shot's location is unknown in events; the spot constant
      would be fabricated, B-48), the prior start is kept in `start_x_events` / `start_y_events`, and any
      tracking location goes to `enriched_start_x/_y` only. `type_id_events` / `type_source_events` keep the
      prior type. The rebound and later-goal cases are their own reported counts.
  - No score change: `missed`; within `missed`, `saved` (a defending-goalkeeper TF-79 contact in flight,
    attributed to the keeper) vs `off_target`, behind the pooled split gate.
  - The pooled outcome gate scores this rule against tier-1 truth: the DFL penalty result (§25.2) and the PFF
    shot outcome.
- **Shoot-outs:** in-play and shoot-out kicks are a REPORTED stratum, not a gating split (C-35r): detection is
  gated pooled over both under the §15.3 licence (each stratum n ≥ 20, non-inferiority margin 0.10). A
  shoot-out kick never feeds the match-goal reconciliation cross-check (shoot-out goals are not match goals).
- **Truth:** native penalty events — DFL `Penalty` (now in Sportec SPADL, §25; 3 in the IDSSE fixtures) and
  PFF penalties including the GS World Cup 2022 shoot-outs (≈ 41 kicks). Run (12) first verifies that the
  shoot-out tracking frames exist.
- **Gates (C-35, C-55):** detection is gated POOLED over outcomes (precision clears the §3 bar; report recall).
  `missed` rows ship only when the pooled detection gate AND the pooled attribution gate (above) pass (the
  outcome comes from the score change and the attribution rule, not from a per-outcome cell). The `saved` / `off_target` split is gated pooled; it alone admits a derived `keeper_save`
  row (§11.3).
- **Part-1 closure is conditional (B-24):** this closes the Part-1 §6.5 / §13 missed-penalty limitation only
  for data profiles whose pooled detection gate passes; elsewhere the limitation stands and is reported. The
  §18.4 text correction states the outcome per data profile.

## 13. `identify_corners` wiring and F0

### 13.1 Frame

All corner geometry goes through §4.3.

### 13.2 `identify_corners` inputs

`identify_corners` consumes `t_kick` from the resolver, `kick_x/kick_y` from the gated origin in the action
frame (`enriched_start_x/_y` where `origin_evidence` is set), and `corner_side` from the origin y (NaN when the
origin is withheld). Without this wiring, SkillCorner derived corners ship NaN (the C8 premise).

### 13.3 F0

`corners/_zones.py:27` hard-codes the attacking goal at x = 105 (frame-LTR) and `_encode` never orients per
corner, while frames are home-attacks-right (ADR-029). Each corner is oriented through §4.3 per provider and
per period. Verify the convention per provider and per period through `GoalMap`. Tests cover both attack
directions × both periods and a shoot-out period.

## 14. F3, F5, F6

### 14.1 F3 — validated z quality

`_ball_contact.py:188` sets `playable = bool(np.isnan(z) or z <= params.contact_max_z_m)`, ignoring `z_source`
(the ADR-125 provenance column). The overhead gate applies on a data profile only when that profile's entry
`(profile, z_overhead)` passes: the declared `z_source == "measured"` is necessary, not sufficient; the gate
must not reduce first-contact agreement against tier-1 truth (paired match-cluster bootstrap CI ≥ 0). GS
declares `measured` but the gate hurts there (0.48 vs 0.58), so the overhead gate stays off for GS.

### 14.2 F5

The §12.6 convergent-timing driver computes Spearman on KICK-RELATIVE times (`t − t_kick`), per provider, on
non-`<NA>` rows, handling the GI clock's whole-second truncation. On native providers it uses
`first_contact_time` from rows with `winner_source="event_coded"` (§9.3); no new column is needed.

### 14.3 F6 — GI joins, drivers and library

- Drivers: the §12.7 GI-falsifier join keys on the restart identity, not an exact float; the drivers pass the
  GI data; the loader read timeout and resume are fixed.
- Library: §9.4 (the exact-float `t_kick` join in `add_corner_windows` is removed; `gi=` deprecated).

## 15. The validation gate table (`RESTART_GATE_TABLE`)

### 15.1 Key

`(data_profile, restart_type, signal, cell_state)`. Absent key → `withheld`, counted.

### 15.2 Signals

| Signal | Tier-1 truth | Metric | Bar | `cell_state` |
|---|---|---|---|---|
| `kick_time` | native event (WHICH) + `K` (WHEN) | detectors: share with `|t − K| ≤ τ` (tier 2); K gate: dispersion `|K − t_event − median| ≤ τ_event` | §3 | resolver state × sign(Δ); `retake` |
| `taker` | DFL restart-event player; PFF `gameEvents.playerId` | player agreement | §3 | restart type × subtype |
| `first_contact` | §5.6 native first touch | team (+ player, non-duel) agreement AND `|t − t_truth| ≤ τ_fc` | §3 | contested × taker_known |
| `subtype` | IDSSE event-coded next-touch location; GS `possessionEventType` label | per-direction precision AND beats majority baseline | §3 + CI > 0 | direction |
| `merge` | native direct-shot restarts (simulated) | precision (report recall) | §3 | population form (events-first / tracking) |
| `penalty` | DFL `Penalty`; PFF penalties incl. shoot-outs; the native penalty result (§25.2) / PFF shot outcome for attribution | detection precision, pooled over outcomes (report recall); goal / missed / rebound attribution accuracy, pooled; saved / off-target split pooled | §3 | — (pooled; in-play / shoot-out reported strata under the §15.3 licence) |
| `origin` | §10.2 per type | median ≤ 2 m AND share within 5 m | §3 | restart type × origin tier |
| `z_overhead` | first-contact truth | gate on vs off, paired | CI ≥ 0 | — |

### 15.3 Placement, derivation, pooling

A declared constant in `tracking/_restart_validation.py`. Entries record the passing tier, the metric, the
Wilson and cluster-bootstrap bounds, n, m and the cell resolution. The constants are pinned from the committed
pre-flight aggregate (§16); an ADR-056 gate asserts the table derives from that artifact and that its population
is enumerated exactly. The table states per cell whether its n and m can reach the bar.

**Pooling ladder (C-37; pre-registered).** A child cell may take its parent cell's result only when all three
hold: (1) the parent passes the §3 gate; (2) every child has n ≥ 20; (3) non-inferiority for every child — the
match-cluster bootstrap 90 % CI of (child accuracy − parent accuracy) has its lower bound above −0.10. A child
with n < 20, or failing (3), stays withheld and is reported. A non-significant difference test is NOT used (with
small n it passes by construction). A pooled output carries `<signal>_cell_resolution = pooled_<level>`;
otherwise `cell`. Never pool across data profiles.

| Signal | Ladder (fine → coarse) | Never pooled |
|---|---|---|
| `kick_time` | sign(Δ) inside the conflict-resolved states; then {corner, free kick, goal kick} (foot-kicked dead balls) | conflict with non-conflict states; `retake`; throw-ins |
| `taker` | subtype (short / crossed); then {corner, free kick, goal kick} | throw-ins |
| `first_contact` | `taker_known` | contested with uncontested (0.67 vs 0.89) |
| `subtype`, `origin`, `merge` | none | (truth and meaning differ by direction, type and tier) |
| `penalty` | outcomes pooled (§12); then in-play + shoot-out under the licence | — |

**Coverage forecast (IDSSE, 7 matches: 57 corners, 172 free kicks, 119 goal kicks, 308 throw-ins, 3
penalties).** AMENDED 2026-10-26 (§27.13): GS is no longer a base, so every "GS admission" route below is closed; the
IDSSE-only forecast with the shortfall per cell is in §27.13, and every withheld cell is reported "withheld (no
base)". Corners: `first_contact` uncontested ≈ 28–39 (borderline), contested ≈ 10–21 (withheld by
construction); `taker` short-corner ≈ 10 (withheld); `kick_time` `both` ≈ 35–45 (passes n), `conflict` ≈ 5–10
(withheld). Free kicks, goal kicks and throw-ins: n is ample for every signal. GS per-signal admission (§5.3)
is the route to more contact coverage. `penalty` (B-24): in-play n ≈ 3 IDSSE + ≈ 20–25 GS (< 35 alone); the GS
World Cup 2022 shoot-outs add ≈ 41 kicks over 5 shoot-outs (m = 5), so the pooled cell can reach the bar only
under the licence, or with GS as the admitted base (§5.3); otherwise missed penalties stay withheld and the
Part-1 limitation stands (§12). `merge` (C-50): IDSSE has 0 direct free-kick shots and 3 penalties, but GS merge
truth is an event fact (`setpieceType` + shot): 44 direct free-kick shots over 64 WC22 matches (DL D2a), plus
the in-play penalties, plus ≈ 41 shoot-out kicks. With GS admitted for `merge`, the tier-2 simulation (which
rebuilds the SkillCorner derived-free-kick-plus-shot shape, §11.4) can reach n ≥ 35 for both forms; without
it the merge is withheld (reported). The SkillCorner population itself is small (the §13 mis-tag: 1 of 20
opendata matches).

### 15.4 Minimum n and m

n ≥ 35 rows and m ≥ 5 matches per cell (or per pooled parent). A cell below either stays `withheld` (or
`conflict` for `kick_time`).

### 15.5 Calibration stamp

Every gated output carries `<signal>_gate_version` (the gate-table artifact hash) beside `<signal>_evidence`
and `<signal>_cell_resolution`: `kick_time`, `taker`, `first_contact`, `subtype`, `merge`, `penalty`, `origin`.

## 16. Pre-registered constants and pre-flight runs

### 16.1 Ratified constants (owner, 2026-10-21; Q3)

Every constant below is pre-registered before any gate is scored. A constant not in this table is not
pre-registered. The table merges the round-2 Q3 ratified values with the region and window constants ruled in
earlier rounds (B6, B9, F1). A calibrated value above its ceiling fails that signal for that data profile (withheld); a
calibrated τ below the measured truth-jitter floor is raised to the floor.

| Constant | Value |
|---|---|
| `τ_event` | ceiling 0.40 s; a median offset > 1.0 s flags decision time, and the event is not used as the WHEN anchor |
| `τ` (detector vs K) | ceiling 0.30 s |
| `τ_fc` | ceiling 0.40 s |
| `a`, `b` | p0.5 / p99.5 of `t_event − K`, clipped to 2.0 s; the signal fails if > 2 % of reference rows lie outside |
| Origin | median ≤ 2 m AND share within 5 m clears the §3 bar; fixed, never calibrated |
| Contested threshold | starts at 0.75 m; recalibration bounded to [0.5, 1.0] m |
| Penalty | `r` ≤ 1.0 m around the pitch-scaled spot; `s` ≥ 1.0 s |
| Penalty position evaluation | the median position over the last 0.5 s of ball rest before the release; never the release frame |
| Penalty goalkeeper `d_gk` | 2.0 m from the goal line; laterally within the posts ± 1.0 m (`|y − 34| ≤ 4.66 m`); keeper from the ADR-078 resolver |
| Penalty encroachment `e_enc` | 1.0 m inside R (penalty area ∪ the 9.15 m disk around the pitch-scaled spot), measured to R's boundary |
| Penalty encroachment count `k` | 2 (outfield players of both teams; excludes the taker — attacking player within 3 m of the ball at release — and the defending goalkeeper; unseen players not counted) |
| Retake radius `ρ_retake` | penalty 1.0 m; free kick 1.0 m; corner 1.5 m (same flag's arc); goal kick: the same goal area; throw-in n/a |
| Penalty outcome window `W` | 3.0 s from the release; crossing-not-observed path allows `W` + 0.40 s (`τ_event` ceiling) |
| Penalty rebound window `W_rb` | 8.0 s from the release (ceiling 10 s); live ball = detected, in the field of play, ≥ 3 m/s |
| Penalty sensitivity | each penalty constant (incl. `W_rb`) ±50 %, reported in run (12) |
| Merge | `r` ≤ 2.0 m (around the constant spot); `Δt` ≤ 2.0 s |
| `m_min` | 5 matches per cell (an absolute count) |
| K rest (dead-ball kicks) | within 0.5 m of the rest centroid for ≥ 0.5 s |
| K release (dead-ball kicks) | displacement > max(0.5 m, 3 × rest SD) AND speed > 3 m/s on 2 consecutive frames |
| K release (throw-in) | ball speed > 4 m/s AND distance to the thrower increasing by > 0.3 m over 0.2 s |
| K sensitivity | each threshold ±50 %, reported in run (1); thresholds are never fitted to event times |
| Agreement winner (amended 2026-10-22, owner) | per data profile × restart type: the candidate (`tf79_contact` / `ball_onset`) with the smaller median (over matches) of the per-match median absolute error against K, on run (2) rows where both were observed; an exact tie or no paired error gives `tf79_contact` (a documented default, not a result); carried on the `kick_time` gate entry |
| K strike (amended 2026-10-22) | a release with peak speed ≥ `v_kick` (8 m/s) up to the next rest; K = the last qualifying rest at or before the first strike; the peak sustained on `k_release_frames` consecutive deduplicated samples (amended again 2026-10-22); no strike + ≥ 2 qualifying rests = `none`, counted; `v_kick` ±50 % in run (1); B-41 short-vs-crossed and blocked-vs-not (> 0.10) guards return the rule to the owner |
| SkillCorner free-kick launch | `v_kick` ≥ 8 m/s; ambiguous if a second launch is within 15 % of peak speed; region = a 5 m disk; `v_kick` returns to the owner if short launch coverage < crossed coverage − 0.10 (absolute) |
| Corner region | 10 m from the flag |
| Throw-in region | 3 m from the touchline |
| Free-kick region (optical) | 3 m from the stationary-window centroid |
| Corner rest | rest centroid within 1.5 m of the flag arc |
| §6.4 Law locations (amended 2026-10-22) | real metres via the provider pitch (`pitch_length` / `pitch_width`); goal-kick rest within one ball radius (0.11 m, Law 2) of the goal area, lines included; no tuned margin (run (1) boundary distances return any noise margin to the owner) |
| R* window | `(t_kick, t_kick + 6 s]`, > 10 m from every flag |
| Pooling step | child n ≥ 20; non-inferiority margin 0.10 on a 90 % match-cluster bootstrap CI |
| GI direction check (amended 2026-10-23) | period majority over ≥ 5 possessions, the GI side mapped through the frames' recorded per-period flip (§4.3); a period without a recorded flip is unresolved (withheld, counted, never compared); corpus guard raises at > 1 % mismatched periods, at > 1 % unresolved periods, or when any match has both periods mismatched |
| GI / tracking pairing (amended 2026-10-21, C-11; r3 C-PLAN-14 / -16; amended 2026-10-23, OD-6 and the second R.2 stop) | `pair_prox_m` = 2.0 m; `pair_prox_share` = 0.245, DERIVED = the `pair_neg_quantile` = 0.99 quantile of the time-shift / cross-pair negative-control shares + `pair_prox_margin` = 0.10, adopted as the maximum per-profile derived bar rounded up to 3 places (§6.1 amendments); `pair_lag_tol_frames` = 2 frames (symmetric; a profile median offset outside ±2 returns the rule); `pair_lag_scan_frames` = 20 (±20); Range = the provider period range, no margin (§6.1, amended 2026-10-25; the 2026-10-23 structural edge rule is retired). Sensitivity in run (2): the lag constants ±50 %, integers rounded (tolerance 1 / 3, scan 10 / 30); the derived bar is never varied directly — its inputs are: `pair_prox_margin` 0.05 / 0.15 and `pair_prox_m` 1.0 / 3.0 m with the bar re-derived from the negatives at that distance, each variant reporting the known-paired pass count and minimum margin. A known-paired opendata match failing, or a profile whose own negatives derive a bar above the adopted `pair_prox_share` (0.245), returns the thresholds to the owner |
| Sensitivity of a share constant (amended 2026-10-23, DL ruling) | a constant valued as a share or probability in [0, 1] moves ±50 % of its distance to the nearer bound, `min(s, 1 − s)` (0.90 → 0.85 / 0.95; 0.99 → 0.985 / 0.995); every sensitivity value is tested valid for its constant (shares in [0, 1], frame counts non-negative integers, distances > 0) |
| GI period-label repair (§26, DL ruling 2026-10-25) | no tuned constant: the period offsets are the existing `_PERIOD_START_SECONDS`; the identity tolerance (G2) is one decisecond, the GI clock's resolution; the frame rate (10 fps) is a provider fact that G2 verifies per match |
| Camera-follow segments (amended 2026-10-21, DL ruling 1, C-PLAN-22; §5.3) | `cam_seg_s` = 10 s; `cam_stitch_k` = 5; each ±50 % sensitivity in run (9) (`cam_stitch_k` rounded to 2 / 8; `cam_seg_s` 5 / 15 s); a restart-core hold over 2 s is reported per profile |
| The c-matched camera draw (added 2026-10-27; DL rulings 2026-10-26 item 4 and 2026-10-27 item 2; §27.20) | `cam_k_c` = 25: the calibration windows nearest by \|c_source - c_reference\| kept before the stitching rule (open play) or after the D pool (restart); sensitivity 10 / 50 in run (9) (the DL's explicit pair, `EXPLICIT_SENSITIVITY`, not the ±50 % rule) |
| Camera footprint validation (amended 2026-10-21, DL ruling C-PLAN-19; §17.7) | per SkillCorner data profile, ≥ `footprint_inside_share` = 0.99 of detected objects lie inside their frame's footprint within `footprint_tol_m` = 1.0 m; a failing profile does not use the footprint and falls back to the current approximation, marked; ±50 % sensitivity on the tolerance in run (9) |
| Ball-noise rest window (RETIRED) | `noise_rest_lo_s` = 1.5 s / `noise_rest_hi_s` = 0.5 s and their 4 sensitivity values were ratified 2026-10-26 and retired the same day (owner decision on a DL error, §27.13): the noise statistic is the second difference; no value |
| Tier-2 grid source rate (added 2026-10-26, DL ruling Q5 (a); owner decision 2026-10-26; §27.13) | `grid_source_rate_tol_share` = 0.01: a reference period whose median positive source step differs from 1 / `frame_rate` by more than 1 % fails closed (`SourceRateError`); no sensitivity |
| Time gap (a DEFINITION, not a ratified constant; owner decision 2026-10-26; §27.14) | a time step longer than `GAP_FACTOR` = 1.5 x the stream's frame interval (1 / `frame_rate`) is a gap (rows are missing). One function (`is_time_gap`) for the chain's missing rows, the steps diagnostic (column name `ball_steps_gt_0_15s_share` kept), E's in-view runs and the d2 triples. Reason: on an exact grid every factor in (1, 2) splits the steps the same way; 1.5 is the midpoint, the most margin against timestamp jitter. No variants (a test pins 1.1 and 1.9) |
| Ball smoothing (added 2026-10-26, DL ruling v3 Q2; owner-approved; §27.14) | sigma_k is a FITTED parameter per profile, not ratified: the grid 0, 0.25, ..., 3.0 frames is a report parameter; a sigma_k on the grid edge (0 or 3.0) is reported and returns to the DL. The kick rise-time windows (base [t_kick - 0.5 s, t_kick], peak [t_kick, t_kick + 1.0 s]) are report parameters; a degraded median more than one WHOLE frame from the SkillCorner median (round(abs difference / frame interval) > 1; DL ruling v4) returns to the DL (owner). The 0.01 m position lattice is a verified SkillCorner fact (2026-10-26), not a tuned value |
| Ball jitter and the still window (added 2026-10-26, DL ruling v4 Q1; owner record; §27.17) | sigma_n is a FITTED parameter per profile, not ratified: the grid 0, 0.01, ..., 1.50 m is a report parameter; a sigma_n on the grid edge is reported and returns to the DL. The jitter goes BEFORE the smoother (the measured SkillCorner still lag-1 is +0.47, interval 0.39 to 0.55). The still window [t_kick - 1.0 s, t_kick - 0.3 s] before a resolved placed-ball kick (never a throw-in) and its minimum of 3 detected ball frames are report parameters (the slow-error study window). The outside-footprint return condition is a degraded share above 0.5 %; the still-spread return band is the existing rate-ratio band [0.67, 1.5] |
| Statistics | Wilson two-sided 95 % (z = 1.96) AND cluster-bootstrap 2.5th percentile (B = 2000); n ≥ 35; LOMO |
| Fidelity tolerances (run (9)) | §16.3 |

### 16.2 Pre-flight runs

The spec freezes after these report (the Part-1 D1 / D2 / D-FKbox pattern). Each run is an in-repo `scripts/`
driver on the `scripts/_driver.py` seam (ADR-052), refusing a dirty tree and stamping provenance (ADR-037,
`require_clean_tree`), writing a committed tier-named aggregate under `docs/research/tf46_part2/` from which the
gate table derives (ADR-056). The DL probe scripts are ported into these drivers. Compute runs on the DGX
through the owner.

1. Tier-1 event-time semantics per type (median and dispersion of `K − t_event`), the K gate, the τ_event floor,
   native bracket quantiles `a`, `b`, the ±50 % K sensitivity, the GS set-piece codes, the DFL location-field
   characterisation (§5.2).
2. `tf79_contact` (first contact vs last contact of the first player), the agreement winner, the contested
   threshold recalibration, the `kick_time` and `taker` cells, the taker split by subtype, and the GS base
   admission per signal (§5.3).
3. `ball_onset` per type, onset rate by subtype.
4. Tier-2 transfer scoring per signal.
5. The bracket-filter gate (tier 2, simulated bracket).
6. Subtype gate per data profile × direction against the majority baseline; native-label agreement; the IDSSE
   label-vs-location mismatch.
7. Merge: events-first rule, then the tracking rule.
8. LOMO fold definition and bootstrap B — pre-registered before any gate is scored.
9. One degradation profile per SkillCorner data profile, plus the §16.3 fidelity check.
10. Goal-kick origin replacement vs detected keeper (the `distrust_native_origin` retirement condition).
11. Taker gate.
12. Penalty detector incl. shoot-outs (first: verify the shoot-out tracking frames exist).
13. Origin gate per type.
14. K-vs-event gate and event-time semantics (part of run (1)).
15. SkillCorner video-overlap census: per data profile, the count of matches with accessible full-match
    broadcast video (opendata first); an ADR-116-safe count. If overlap exists, a video truth sample is a
    separate owner decision; video stays never-a-gate.
16. Corner-model refit stability: leave one tier out, zone-assignment agreement vs the full fit.

### 16.3 Tier-2 fidelity tolerances (run (9); ratified, relayed DL)

**General rules (pre-registered):**

- Every statistic is computed per data profile and per region — the corner-flag zone (≤ 10 m from a flag),
  the touchline band (≤ 3 m), the goal-kick box, and the rest of the pitch — on the degraded IDSSE / GS
  tracking and on the target SkillCorner profile.
- The degradation is FIT on the calibration target matches and the fidelity statistics are SCORED on held-out
  matches (amended 2026-10-25, CL-42, §27.3: for run (9), one seeded held-out split per data profile, not LOMO),
  so a statistic the degradation was tuned to cannot pass by construction (C-46). The
  emergent statistics — kick-window miss rate and nearest player visible at contact — are the primary
  fidelity tests.
- Margin = max(the fixed value below, 2 × the match-cluster bootstrap SE of the target statistic), capped at
  2 × the fixed value. If 2 × SE exceeds the cap, the statistic fails characterisation for that profile.
  (Amended 2026-10-25, CL-42, §27.6: the form per rule type, the bootstrap, the region `all` for every global
  statistic, the required-row inventory and `not_evaluable`; §27.5 for identity swaps.)
- A failed statistic withholds only the signals that depend on it (dependency map below). It does not
  disqualify the whole profile.

| Statistic | How measured | Pass when |
|---|---|---|
| Frame rate | resample to the target rate | exact |
| `z_source` mix | replace z with the estimated z | exact, by construction |
| Ball detection rate | share of frames with a detected ball, per region | absolute difference ≤ 0.03 |
| Ball-gap length distribution | gap lengths in seconds, per region | KS distance ≤ 0.10 AND p90 within ±20 % (relative) |
| Kick-window miss rate | share of restart kicks with no detected ball frame within ±0.3 s of `K` | absolute difference ≤ 0.03 |
| Ball second-difference distribution (replaces "Ball-position noise SD", amended 2026-10-26 v3, §27.14) | per region of the middle frame, the |d2| of the ball, d2 = p(t+1) - 2 p(t) + p(t-1) on consecutive detected triples of the exact grid; s (1.4826 x MAD) and the d2 lag-1 are reported, never gated | KS distance ≤ 0.10 under the margin rule (the gap-statistic tolerance; no new constant) |
| Fill share | share of ball frames that are filled or extrapolated (not detected) | absolute difference ≤ 0.03 overall, ≤ 0.05 per region |
| Player visibility near the ball | share of player-frames detected within 10 m of the ball | absolute difference ≤ 0.03 (≤ 0.05 for all players) |
| Nearest player visible at contact | share of contact frames where the nearest player to the ball is detected | absolute difference ≤ 0.03 |
| Identity-swap rate | proxy: same-id jumps implying > 12 m/s between consecutive frames, per player-hour; the same proxy after injecting swaps into IDSSE / GS | 95 % CI of the rate ratio within [0.67, 1.5] (rare-event Poisson) |
| Off-camera position error at re-detection (amended 2026-10-21, DL ruling R2; C-PLAN-21) | per player spell of `visibility == False` frames followed by `visibility == True`: the distance (m) between the last extrapolated position and the first detected position; binned by time off camera (< 1 s, 1–3 s, 3–10 s, > 10 s); spells that end at a period end are excluded and counted | per bin: KS distance ≤ 0.10 AND p90 within ±20 % (relative), under the margin rule above; scored held-out |

Identity swaps have no ground truth on SkillCorner, so both sides use the same physical-implausibility proxy.
Its known limit is reported: it misses swaps between players who are close together.

Amended 2026-10-25 (CL-42, DL rulings; details in §27.5–§27.7): the identity-swap proxy counts a jump only
between two consecutive detected frames of one id (exposure = detected player-time), and its rule is the exact
conditional rate-ratio interval with an overdispersion correction; fill share is ≤ 0.05 per region and ≤ 0.03 in
region `all`; player visibility near the ball is scored separately on open-play frames and on restart-phase frames
(both per region, ≤ 0.03), and the visibility of all players (`player_visibility_all`, ≤ 0.05) in region `all`;
the nearest player visible at contact and the off-camera position error are measured as §27.6 defines; a `subtype`
row whose window starts at a `tf79_contact` `t_kick` depends on the contact statistics (B-33, gap 7m); withholding
is per score row and its `source_path` (§27.7).

Amended 2026-10-26 (CL-42, DL ruling on the degradation defects; owner decisions; details in §27.12): the frame
rate is measured on an exact grid (linear interpolation; "exact" = |rate − target| ≤ 1e-6 × target); the ball
detection rate and the fill share give an undetected frame the region of the last detected frame; a ball gap is a
maximal undetected run of the period's stream, by the region of the last detected frame before it, and a run that
touches a period start or end is censored (excluded, counted); the ball-position noise SD is the RMS over the rest
windows [t_kick − 1.5 s, t_kick − 0.5 s] of corners, goal kicks and free kicks of the 2-D SD about the window
mean, by the region of the window mean (fewer than 30 windows pools every region). The "stationary windows (≥ 0.5 s
at rest)" rule above is replaced: a speed threshold on noisy positions keeps the quiet stretches only.

Amended 2026-10-26 (v3; DL ruling and owner decision, §27.14): the noise SD row (the rest window, then s of d2 with
the ratio band) is REPLACED by the ball second-difference distribution: the KS distance between the |d2| histograms of
detected triples, degraded against SkillCorner held out, per region, ≤ 0.10 under the margin rule. Reason: SkillCorner
smooths its ball track (d2 lag-1 +0.66 / +0.64; white noise gives -2/3) and its positions lie on a 0.01 m lattice, so
s moves in steps of 25-33 %, wider than the ±15 % band. s and the lag-1 are reported only.

**Dependency map (a statistic gates only these signals):**

| Signal | Statistics it depends on |
|---|---|
| `kick_time` via `ball_onset` / `K`; `origin`; `subtype` | frame rate, `z_source` mix, ball detection rate, gap distribution, kick-window miss rate, the ball d2 distribution (was: noise SD), fill share |
| `kick_time` via `tf79_contact`; `taker`; `first_contact` | all of the above, plus player visibility near the ball, nearest player visible at contact, identity-swap rate, off-camera position error at re-detection (R2) |
| `merge` | the statistics of `taker` and `origin` |
| `penalty` | the ball statistics plus player visibility (all players); the `saved` / `off_target` split reads a TF-79 goalkeeper contact, so it also depends on the contact statistics (C-51), including the off-camera position error at re-detection (R2) |

Dependencies follow the source path a row ACTUALLY uses (B-33): a SkillCorner origin read at the launch
contact, a subtype whose window starts at a `tf79_contact` `t_kick`, and any signal whose `t_kick` came from
`tf79_contact` depend on the contact statistics (player visibility near the ball, nearest player visible at
contact, identity-swap rate, off-camera position error at re-detection) as well as the ball statistics.

## 17. Schema and registries

### 17.1 Actions (additive enrichment, not base `SPADL_COLUMNS`)

Declared in `RESTART_REFINEMENT_COLUMNS` (`tracking/_restart_validation.py`):

| Column | dtype | Stage |
|---|---|---|
| `kick_bracket_lo`, `kick_bracket_hi`, `t_contact`, `t_onset`, `kick_time_delta`, `t_kick` | float64 | resolve_kicks |
| `kick_time_source`, `kick_time_resolution`, `kick_time_evidence`, `kick_time_gate_version`, `kick_time_cell_resolution` | string | resolve_kicks |
| `kick_retake` | boolean | resolve_kicks |
| `kick_rest_x`, `kick_rest_y` | float64 | resolve_kicks (amended 2026-10-21, owner ack B-13: the public rest-window median in the action frame, consumed by the origin tiers) |
| `gi_pairing`, `gi_pairing_failed_test` | string | resolve_kicks (amended 2026-10-21, owner ruling C-11: per-match GI / tracking pairing, §6.1; `<NA>` off SkillCorner) |
| `refine_created` | boolean | refine (amended 2026-10-21, owner ack B-13: True only on rows refine adds — the round-trip marker) |
| `taker_player_id` | provider id dtype (ADR-019) | resolve_kicks |
| `taker_source`, `taker_evidence`, `taker_gate_version`, `taker_cell_resolution` | string | resolve_kicks |
| `type_id_events` | Int64 | refine |
| `type_source_events` | string | refine |
| `start_x_events`, `start_y_events` | float64 | refine (rebound retype only, §12) |
| `type_tracking_disagrees` | boolean | refine |
| `subtype_evidence`, `subtype_gate_version`, `subtype_cell_resolution` | string | refine |
| `merged_from_derived_action_id`, `merged_from_shot_action_id` | Int64 | refine |
| `merged_derived_type_id`, `merged_derived_result_id` | Int64 | refine |
| `merged_derived_time_seconds`, `merged_derived_end_x`, `merged_derived_end_y` | float64 | refine |
| `merged_derived_gi_marker_possession_frame_end`, `merged_derived_gi_next_possession_frame_start` | Int64 | refine |
| `merged_derived_gi_attacking_side` | string | refine |
| `gi_next_possession_time` | float64 | resolve_kicks (CL-1 amendment) |
| `merged_derived_result_source`, `merged_derived_restart_evidence`, `merged_derived_type_source`, `merged_derived_original_event_id`, `merged_derived_repair_kind` | string | refine |
| `merge_evidence`, `merge_gate_version`, `merge_cell_resolution` | string | refine |
| `penalty_evidence`, `penalty_gate_version`, `penalty_cell_resolution`, `penalty_crosscheck`, `penalty_outcome` | string | refine |
| `origin_evidence`, `origin_gate_version`, `origin_cell_resolution` | string | add_restart_coordinates |

`start_coord_source` gains two values (§10.1). `type_source` already admits `tracking` (Part 1). Refine-created
rows carry the base signal columns explicitly (§11.3, §11.4).

### 17.2 SkillCorner converter (`SKILLCORNER_SPADL_COLUMNS`)

**Amended 2026-10-21 (owner rulings, CL-1 and B-11; §6.1):** adds `gi_marker_possession_frame_end`,
`gi_next_possession_frame_start` (nullable `Int64`; derived restart rows only — `<NA>` on non-derived rows, on chain
rows where no possession lies between, and when the input lacks the GI frame columns) and `gi_attacking_side`
(string; every row backed by a GI possession, native and derived). The times (`kick_bracket_lo`,
`kick_bracket_hi`, `gi_next_possession_time`) and the per-match `gi_pairing` are `resolve_kicks` outputs (§17.1).
**Amended 2026-10-21 (owner ack, plan CL-13; plan review r3):** adds `gi_possession_frame_start`,
`gi_possession_frame_end` (nullable `Int64`) and `gi_event_id` (the GI possession `event_id`) on every row backed
by a GI possession, native and derived — the input of the §6.1 per-match pairing test, which de-duplicates by
`gi_event_id` so each possession counts once. `<NA>` when the input lacks the GI columns.
**Amended 2026-10-25 (DL ruling, §26):** adds `period_source` (string, never `<NA>`): `"native"` = `period_id`
is the GI label; `"frame"` = the period and the time come from the GI frame through the §26 repair. Reflection
kind `invariant`; not read by `training_exclusion_mask` or `provenance_masks`.

### 17.3 Corner table (`corners/_columns.py`)

Adds `taker_player_id`, `taker_known`, `winner_source`, `first_contact_evidence`, `first_contact_gate_version`,
`first_contact_cell_resolution`, `first_contact_attribution_margin`, `first_possession_status`, and carries
`kick_time_source` / `kick_time_evidence`. `t_kick` and `kick_x/_y` already exist (`_columns.py:36, :39`) and
now hold gated values.

### 17.4 Contacts table (`tracking/_ball_contact.py`)

Adds `attribution_margin`, `attribution_margin_player`, `attribution_other_visible`.
**Amended 2026-10-21 (DL ruling C-PLAN-19, §17.7):** adds `attribution_other_unseen_reason` (string:
`off_camera` / `missed_detection`; `<NA>` when an other-team player is visible, or when the profile has no
validated camera footprint), so a non-visible opponent is split into "outside the camera footprint" and
"inside it but not detected".

### 17.5 Sportec converter (`SPORTEC_SPADL_COLUMNS`)

Adds `shot_woodwork_type` (string: `post_left` / `post_right` / `crossbar`; `<NA>` on all other rows) and
`penalty_result_method` (string: `shot_outcome` / `score_reconciliation` / `kickoff_order` / `continued_play` /
`score_elimination` / `unresolved`, `continued_play` added by the §25.2 step-3m amendment;
`<NA>` on non-penalty rows), both the ADR-001 dedicated-column pattern (§25.2, §25.4), both reflection-invariant.
**`repair_kind` (amended 2026-10-21, owner-approved decision D1, DL-recommended):** promoted to base
`SPADL_COLUMNS` (19 → 20), NOT sportec-specific — two emitters (SkillCorner goals-reconciliation, Sportec §25
penalties: `penalty_outcome_unresolved` / `penalty_score_inconsistent`) and shared readers
(`training_exclusion_mask`, `xtgk/_markov`, `xtgk/_turnover`) meet the Part-1 decision-D base-promotion rule.
Closed vocabulary `REPAIR_KIND_VALUES`; joins `_BASE_SIGNAL_COLUMNS` (Atomic SPADL drops it). Name / dtype /
vocabulary unchanged (Hyrum); the SkillCorner column order shifts.

### 17.7 SkillCorner tracking adapter fields (amended 2026-10-21; plan review r3 C-PLAN-15 + owner Decision 2, ruled by the DL session)

**Governing constraint (DL ruling).** Frame-level possession and the camera footprint are SkillCorner-processed
outputs, not independent truth, and the IDSSE / GS bases do not have them. They may serve SkillCorner internal
consistency (the §6.1 pairing) and SkillCorner profile measurement (§16.3, the camera model). They are never the
truth a transfer cell is scored against, and never the truth for `taker`, `first_contact`, `kick_time` or any
gated signal.

The SkillCorner tracking adapter (`tracking/skillcorner.py` and the pining bronze builder) carries three source
fields that it discarded before:

1. **Ball detection (C-PLAN-15, blocking).** Ball-row `visibility` = the source `ball_data.is_detected`, with
   True / False / null preserved (False = extrapolated; null = no position). A bronze without the field keeps
   today's null. Every consumer that needs a "detected ball" (the §6.1 proximity share; the §16.3 ball statistics
   — detection rate, gap lengths, kick-window miss, fill share; §6.3 / §6.4 SkillCorner launch and onset rules)
   reads it through `detected_mask` (ADR-109), so an extrapolated ball never counts as observed.
2. **Frame-level possession.** `sc_possession_player_id` and `sc_possession_group`, frame-level columns
   following the existing frame-level column convention (as `ball_state`); declared in the tracking schema;
   reflection kind `invariant`. Use: the §6.1 pairing alignment only. **Storage (amended 2026-10-21, DL ruling
   R1, plan review r4 B-17):** `sc_possession_player_id` is `category` over the provider's RAW object ids — the
   `team_id` pattern (ADR-106; written once at build time, ADR-103); comparisons stay value-neutral through
   `id_compat._decat` (ADR-019). "Provider id dtype" means id identity, not storage. A null possession stays
   `<NA>`, never a sentinel category; the categories are the raw provider ids, unchanged. **Id form (amended
   2026-10-21, DL ruling 2; plan review r5 B-24):** "unchanged" means identity-preserving, not "keep the JSON
   integer type": the categories use the same canonical string form as the frames' `player_id` (ADR-019), built
   with `id_compat.canonical_id_series` or by converting the JSON values to strings before any DataFrame exists —
   never `astype(str)` on a column that may already be `float64` because of null possessions (that gives
   `"366.0"`).
3. **Camera footprint (`image_corners_projection`).** The four footprint corners as `cam_<corner>_x` / `_y`
   floats, rescaled with `_scale_to_spadl` exactly like the tracking, declared geometric in the ADR-045
   reflection registry (point pairs); a null footprint stays null. Representation: frame-level columns with a
   memory-bounded dtype, or a companion frame table if the frame-level-column cost breaches the TF-58 memory
   budget (the plan decides, under A/B review). Derived: `in_camera_view` (nullable boolean per object row: the
   point lies inside the footprint polygon; `<NA>` when the footprint is null). Uses:
   - §16.3 player visibility near the ball, measured on the footprint;
   - a camera-follow model for tier 2 (§5.3): the footprint pose relative to the ball, fitted on SkillCorner
     calibration matches and applied to IDSSE / GS tracking to mask off-camera objects; fidelity scored on
     held-out matches (C-46);
   - "off camera" versus "missed detection" for ADR-109 consumers and for `attribution_other_unseen_reason`
     (§17.4).

   **Existing ADR-109 consumers (amended 2026-10-21, owner + DL ruling R4):** no existing `detected_mask`
   consumer changes behaviour this cycle; an invariance test proves it, and a failure stops execution as an owner
   decision point, never a consumer patch. `in_camera_view` is NOT wired into the GK detection gate this cycle
   (it would change a trained model's input with no evidence of benefit, inside a release retrain). Measured
   instead: run (9) and the ghost-GK retrain report give the share of ghost-GK training frames with the keeper
   undetected-in-view versus undetected-off-camera — the evidence a later owner decision on wiring needs.

   **Validation gate (pre-registered, §16.1).** On each SkillCorner data profile, at least 0.99 of detected
   objects lie inside their frame's footprint, within 1.0 m. A profile that fails does not use the footprint and
   falls back to the current approximation, marked.

**Prerequisite before C2.** Check the pining owner-tier SkillCorner tracking source for `ball_data.is_detected`,
`possession` and `image_corners_projection`. A field that is absent is not used on that profile (the per-profile
fallback) and is reported. Null rule: frames with null fields (e.g. frame 0 in opendata) carry `<NA>`.

**Tests (DL ruling).** The adapter preserves all three fields from the opendata JSONL shape, including nulls;
the footprint reflects with the frames; `in_camera_view` is correct on a hand-built polygon; the pairing
alignment is offset 0 on aligned data, flags a 5-frame shift and is robust to a 10-frame carry plateau; the
footprint validation gate is tested both sides (0.99 / 0.98); the camera model is fitted on calibration matches
and scored on held-out matches.

**Amended 2026-10-22 (DL ruling, owner-relayed — orientation-flip recording; plan CL-15 amendment).** The
SkillCorner path orients via the GEOMETRIC `orient_frames_to_ltr_by_geometry` (per-period flip when the home-GK
median x > 52.5, read AFTER GK resolution incl. the `derive_goalkeepers` fallback), not the flag-based
`compute_attacking_direction`. The per-period flip is decided ONCE and recorded, never decided twice:
- `direction.py` splits deciding from applying: a private pure function returns `{period_id: True | False |
  None}` (True/False = a GK anchor decided it; `None` = no anchor, left as-is + warned; period 5 always False).
  `orient_frames_to_ltr_by_geometry` keeps its signature and byte-identical output (float32 store-back,
  zero-home raise, `team_attacking_direction` fill, every warning's stacklevel — Hyrum, lakehouse callers). The
  flag path computes `flip = not home_attacks_right`.
- `TrackingConversionReport.ltr_flip_by_period: Mapping[int, bool | None]` (default `None`) records it;
  `absolute_frame` → empty / all-False; other providers stay `None` (populating theirs is an owner scope
  question). It also makes the silent normal flips auditable.
- CL-15 "one helper, no copy" = shared pieces: `_resolve_pitch_dims`, `_scale_footprint` (same `_scale_to_spadl`
  affine), `_apply_flips` (via `reflect_columns`, float32-safe). `convert_camera_footprint(bronze, *, report,
  assume_standard_pitch=False)` scales + APPLIES `report.ltr_flip_by_period` (never re-decides; RAISES on a
  non-SkillCorner report or a report for another match). `orient_camera_footprint(footprint, flips)` applies an
  explicit flip mapping keyed `(game_id, period_id)`.
- `in_camera_view` is computed after scaling, before orientation, in float64, before the float32 cast
  (orientation-invariant: positive-diagonal affine + 180° rotation both preserve containment and corner order).
  Corner-order-independent point-in-quad (all cross products non-negative OR all non-positive); a non-convex /
  zero-area quad → `<NA>`, counted on the report; a NaN own-position → `<NA>`. `preprocess` does not recompute
  it (it uses the positions the provider reported).
- Tests add, on top of the above: `test_orient_by_geometry.py` passes unchanged (byte-identical); an asymmetric
  flip fixture; a no-native-GK fixture; a no-anchor `None` period; `absolute_frame` no flips; period 5 never; a
  wrong-match report raises; `in_camera_view` identical `ltr` vs `absolute_frame`; reversed-corner-order mask
  equal; non-convex → `<NA>` + counter; a tripwire unflipped footprint fails the §16.1 gate.

**Amended 2026-10-25 (DL ruling, Q4; §6.1 Range).** The adapter records the PROVIDER period range: per period,
the first and last raw frame the provider labels with that period (empty frames count; period-`None` frames do
not). The bronze carries only frames with objects, so the reader of the raw feed computes the range with the pure
helper `provider_period_frame_ranges(records)` and passes it to `convert_to_frames(..., period_frame_ranges=)`,
which checks that every retained frame lies inside it (a raise otherwise: a range from another match), records it
on `TrackingConversionReport.provider_period_frame_ranges` and stamps it on `frames.attrs`. The consumer keys it
`(game_id, period_id)` through `period_frame_ranges_from_report(report)`; a period-only key raises `TypeError`
(the C2-IMPL-28 lesson). Resolution at the consumer: the explicit argument, then the `attrs` stamp, then the
retained range (fails closed); the basis is recorded per match (`GiPairing.range_basis`). Staging records the range
in `_profile` and `Part2Match` passes it explicitly.

### 17.6 Registries touched

Reflection kinds (ADR-045; times and ids invariant, `shot_woodwork_type` invariant, origin through the existing
enriched columns), purity entries (ADR-033), the frame-consuming call convention (ADR-078), `feature_glossary`
for the derived columns (ADR-048), the tracking-schema and full-suite-only gates, the provider SPADL column
additions (SkillCorner §17.2, Sportec §17.5; the 3.30 ET-regression golden carve-out), and the C4 model. `metric_contracts` is not touched (no new
metric family). The plan enumerates each gate both ways.

## 18. Refit, retrains, memo, lakehouse, Part-1 text (joint release)

### 18.1 Corner-model refit

Every bundled corner model: `_run_zone_weights` and `_routine_weights`, after the B6 per-model staleness
pre-flight. Inputs: F0 orientation, the R* first contact (it moves `t_target`), the Part-1 recovered lost
corners. Corpus: IDSSE + GS + SkillCorner, each recorded by tier (IDSSE, GS, opendata, owner, and the course
tier as an aggregate named by tier); the refit is non-reversible and the model card records per-tier match and
corner counts, the GI release and a feature hash (ADR-116: weights and aggregates only). Stability: run (16).
The re-fit is recorded in the model cards, the CHANGELOG and the F0 ADR.

> **Amended 2026-10-21 (owner ruling, relayed DL; plan review C-04) — the course tier.** ADR-116's publish bar is
> reversibility; weights and aggregates are non-reversible, so ADR-116 does not block weights refitted on
> course-tier data. The licence is a separate axis: the course datasets are CC-BY-NC-4.0, and whether model
> weights are "adapted material" under it is legally unsettled (not a legal opinion); attribution is honoured in
> every case. Procedure:
> 1. **Deduplicate first:** match course-tier matches against the owner-tier corpus by match id (competition, date,
>    teams); only matches NOT in the owner tier form the course tier; report the overlap count.
> 2. **Data path:** the DL session fetches the non-overlapping subset from the gated HF datasets with
>    `snapshot_download(allow_patterns=...)` after the owner's go-ahead for that download; it is converted with the
>    same SkillCorner converter (GI release pinned) under its own data profile.
> 3. **Measure the value:** run (16) reports zone-assignment agreement of the full fit against the fit without the
>    course tier, for `_run_zone_weights` and `_routine_weights`.
> 4. **Pre-registered decision rule:** agreement ≥ 0.95 for both models → the shipped and published weights are
>    refitted WITHOUT the course tier; otherwise the owner decides with the measured gain — (a) ship with the
>    course tier, with CC-BY-NC attribution in the model cards and `NOTICE`, or (b) ship without it and keep it for
>    research, or (c) ship the default weights without it and publish a course-tier-inclusive NON-DEFAULT variant to
>    a private, owner-only HF repo (owner note 2026-10-21: established precedent for non-default weights), with
>    CC-BY-NC attribution in that variant's card.
> 5. In every case the course tier enters the run (16) stability check and the tier-named research aggregates
>    (ADR-116), and the model card records the training corpus by tier. Records: the overlap count, per-tier match
>    and corner counts, the run (16) agreement table, and the decision with its rationale in the CHANGELOG and the
>    F0 / refit ADR.

### 18.2 Retrains

The Part-1 retrain set (xshot, xcross, ghost-GK, ghost-outfield, xsuccess) runs once here, after the HF-fanout
refactor merges. The §25 commit lands before it, so the Sportec-fed models train on the corrected goals.

### 18.2.1 Provenance contract for every training entry point (B-23; owner ruling, relayed DL)

One helper, per-dimension masks — no hand-picked helper per model. `spadl/_training_exclusion.py` gains
`provenance_masks(actions) -> dict[str, pd.Series]`; `training_exclusion_mask` stays as the result mask and its
four callers (xsuccess, expected_passing, xtgk markov, xtgk turnover) keep working.

| Dimension | A row is unvalidated when |
|---|---|
| `result` | the existing rule: `result_source ∈ {inferred, stopgap}`, `restart_evidence == "chain_ambiguous"`, or `repair_kind` set |
| `time` | a Part-1 derived restart row (placeholder `time_seconds`) with no gated `t_kick`, or a `repair_kind` row (time only bounded by its goal episode). A derived row WITH a gated `t_kick` uses `t_kick` as its event time. Part-2 penalty rows (gated detection) and Sportec native penalties are valid. |
| `type` | `type_source == "default"`, for a model whose type set includes that subtype |
| `location` | canonical start is NaN and no gated `enriched_start` exists (`origin_evidence` unset) |
| `existence` | none today (every derived row records a real event); the dimension exists so a future uncertain-existence row has a home |

**Treatment by example type:**

- **Action-level examples** (xsuccess, expected_passing, xtgk): exclude a row whose READ dimensions are
  unvalidated.
- **Frame-level occurrence labels** (xshot, xcross; `_build_occurrence_labels`, `tracking/_occurrence_labels.py`
  labels FRAMES: 1 when a same-team event selected by `type_id` lies in `[t, t + horizon]`): an event whose
  time or type is unvalidated is NOT dropped — dropping it turns the frames before it into false negatives.
  Instead, CENSOR: remove from training and evaluation every frame of the event's team in
  `[t_lo − horizon, t_hi]`, where `[t_lo, t_hi]` is the event's uncertainty interval (the goal episode for a
  repair row; the kick bracket for a derived restart without a gated `t_kick`). Count the censored frames.
  Events with a validated time are used at that time (`t_kick` for gated derived rows).

**Per-retrain table (pinned):**

| Model | Reads | Treatment |
|---|---|---|
| xsuccess, expected_passing | result (label), location (start features) | exclude rows whose result OR location is unvalidated |
| xtgk markov / turnover | result, possession order | exclude per the existing result mask; order uses the guarded chronological sort, valid for placeholder times |
| xshot | shot `type_id` + time (label); frame features | gated events at their validated time; an unvalidated-time shot (a `missing_goal` derived shot) censors its episode window |
| xcross (default, open-play cross) | cross `type_id` + time | the same rule; set-piece crosses are not in the default type set |
| xcross (sensitivity, + set-piece crosses) | also type | a `default` subtype or an ungated derived time censors its window |
| ghost-GK, ghost-outfield | frames | unaffected ONLY if no action anchors the frame selection; if action rows anchor the frames (e.g. shot frames), the time dimension applies to the anchor rows. The plan states which applies, from the code. |
| corner models (run-zone, routine) | §15 gated outputs | governed by `withheld` / `type_source` / gated `t_kick` |

**A contract that cannot drift:** each training entry point declares `reads=` (its dimensions); an ADR-056-style
test asserts every caller's declaration matches the columns its label and feature builders actually read; a
test per caller covers the exclusion or censoring on a fixture with each kind of unvalidated row; the retrain
reports carry excluded-row and censored-frame counts per model.

### 18.3 Memo and lakehouse

The §12 construct-validity memo is regenerated. The lakehouse handoff covers the consumer models and marts.

### 18.4 Part-1 text corrections at the joint release

Part-1 §6.5 and §13 (missed penalties: detected where the pooled gate passes, per data profile, §12); Part-1
§9.2 (the free-kick origin is now gated, §10.2).

## 19. Tests and gates

- Gate-table derivation (ADR-056), exact population, fail-closed absent key.
- Purity and idempotence of every stage (second run changes nothing), including a merged row and a penalty row.
- Per signal, both sides: a passing cell ships the value with evidence, stamp and cell resolution; a failing
  cell ships NaN with `withheld`.
- Pooling: a child with n < 20 stays withheld; a child failing non-inferiority stays withheld; a pooled child
  carries `pooled_<level>`; contested and uncontested never pool.
- Native-never-rewritten: a native row with a disagreeing tracking rule keeps its `type_id` and sets
  `type_tracking_disagrees`.
- Reversibility as a ROUND TRIP: reconstruct the full pre-refine frame from `type_id_events`, the
  `merged_from_*` and `merged_derived_*` columns, and compare every column.
- Refine-created rows carry `result_source="inferred"` and are excluded by the training-exclusion helper.
- Bracket: a derived-row kick before the placeholder time is found; a candidate outside the strict bracket is
  kept as evidence but never ships, for every state; a conflict with exactly one candidate inside resolves
  with `kick_time_resolution="bracket"`; the native bracket clip sends an out-of-bracket kick to `none`.
- Kick contact: a short-corner receiver inside the region is not taken as the kick; the SkillCorner launch rule
  finds the launch and marks an ambiguous double launch `none`.
- Retake classification and count.
- Origin tiers: each tier fires on its fixture; canonical `start_x/_y` never written (including merged rows).
- Taker: native passthrough (incl. a Sportec penalty); SkillCorner contact route; withheld path.
- First contact: the §5.6 extraction (duel winner, `Delete` dropped, co-timed `BallClaiming`, `foul` class);
  the event-coded passthrough; the geographic rule.
- `first_possession_time`: a simple derived row equals the GI receiver `frame_start`; a chain row is `<NA>` with
  status `chain`; a native row is `<NA>`; a change to the τ pad leaves `first_possession_time` unchanged.
- Penalty detector: pitch-scaled spot, goal cross-check, missed (no score change), saved split with keeper
  attribution, shoot-out cell.
- Renumber: one atomic renumber, `action_id` chronological (ADR-065 guard).
- Frame transform: both attack directions × both periods × a shoot-out period; GI direction check — an isolated
  disagreeing possession is withheld and counted, a disagreeing period is withheld, counted and warned, the
  corpus guard raises at its thresholds.
- F3: estimated z never overhead-gates; a passing `z_overhead` profile does; GS stays off.
- F5 kick-relative Spearman per provider; F6 identity-keyed GI joins; `add_corner_windows(gi=)` deprecation.
- Fidelity: a failed fidelity statistic withholds exactly the signals in its §16.3 dependency row (and, by the
  source-path rule, the signals that used a contact path) and no others; a statistic whose 2 × SE exceeds its
  cap fails characterisation; fidelity is scored on held-out matches, never on the matches that fit the
  degradation.
- Arbitration exhaustiveness: every combination of (observed, inside-bracket, |Δ| vs τ) maps to exactly one
  §6.5 row; both-agree-one-inside resolves `single` with the inside candidate.
- Kick contact: the ball retriever at the start of a long SkillCorner bracket is not taken as the kick (only
  in-region contacts after the ball's last entry count).
- Penalty: constants evaluated on the rest-window median, not the release frame (a keeper stepping forward at
  the kick still detects); up to `k` = 2 encroachers within `e_enc` still detects, a third does not; a goal
  after a goalkeeper touch or the woodwork is the penalty's goal; a rebound or a taker double touch leaves the
  penalty `missed`; no score change → `missed`; the crossing-not-observed path uses the event goal time within
  `W` + 0.40 s on IDSSE / GS; a SkillCorner-shaped scored penalty with an untracked crossing (score written at
  the post-goal kick-off) is classified `goal` by the GI sequence rule and no derived missed row ships beside
  the reconciliation goal row (C-49); a taker goal after a save is a legal rebound and the penalty stays
  `saved`, while a taker second touch after the woodwork yields no score change (B-40); in-play and shoot-out
  pool only under the licence.
- Base precedence: IDSSE alone when adequate; GS alone when IDSSE is thin and GS is admitted; combined only under
  non-inferiority; the larger adequate base first for the same-meaning signals; the choice never depends on
  which base passes.
- SkillCorner penalty rebounds: a saved penalty plus a one-touch rebound finish with no intervening GI possession
  is NOT the penalty's goal (rule (c)); a ball tracked in play after `W` is NOT the penalty's goal (rule (d)).
- Consistency guard: the reconciliation goal pairs with the last detection in its goal episode; agree → no row;
  disagree without rebound evidence → derived row withheld; rebound → derived missed row ships with
  `penalty_crosscheck="rebound"` and the reconciliation row is retyped `shot_penalty → shot` with its prior
  type kept in `type_id_events`.
- Merge population: the simulated SkillCorner shape (derived free-kick row + shot) is what the merge gate scores.
- Post-goal handling (C-55, B-49): a scored SkillCorner penalty followed by players returning the ball to the
  centre spot → `agree`, no derived row, no retype; a contact after `W_rb` is ignored; a resting ball is not live.
- Pairing (B-47, B-50, C-57): a SkillCorner penalty with NO taker GI possession still pairs (penalty context
  only); a saved penalty → defending possession → later attacking goal ships the missed row with
  `penalty_crosscheck="later_goal"` and retypes the `shot_penalty` goal row to `shot`; an attribution gate that
  does not pass yields `disagree`, no row, no retype.
- Rule (d) (C-58, B-53): the goalkeeper kicking or throwing the ball out of the net after a goal is not a live
  ball; a live ball after the goalkeeper's PARRY is (rebound evidence).
- `penalty_outcome` takes only `goal` / `saved` / `off_target` / `missed` / `<NA>`; `missed` when the split gate
  fails. The one-kick match accepts a release 1–2 frames before the GI possession `frame_start` (C-59).
- Annotation (B-52): annotating an existing native penalty shot writes `penalty_outcome` and provenance only; its
  `result_id` and `type_id` are unchanged. The one-kick match uses time and team; a shot far from the spot in
  noisy event coordinates still matches by its possession containing the release frame (B-51).
- One kick, one row (C-56, B-46): a missed mis-tagged penalty yields one row (the merged / existing shot,
  annotated), never a second derived `shot_penalty`; an existing keeper action suppresses a derived
  `keeper_save`.
- Rebound retype (B-48): the retyped `shot` has NaN canonical start, the spot constant kept in
  `start_x_events` / `start_y_events`; the round trip restores it; no retype without the attribution gate.
- Retake: release → return to rest within `ρ_retake` → second release is a retake; a second release without a
  return to rest is not.
- Provenance contract (§18.2.1): the `reads=` declaration test; per caller, exclusion (action-level) or frame
  censoring (occurrence labels) on a fixture with each kind of unvalidated row; censored-frame counts reported.
- Round trip covers a subtype rewrite (`type_source_events`), a merged `shot → shot_penalty` retype
  (`type_id_events` = the shot's), and the dropped derived row's `gi_*` and `repair_kind`.
- GS origin: no GS cell is scored for the free-kick spot, the goal-kick spot or the throw-in x.
- Coverage report: share of non-NaN output per data profile × cell × evidence × cell resolution.
- §25 tests and guards (§25.5).
- Fixtures: IDSSE (CC-BY-4.0 + NOTICE) and opendata (MIT) slices only; GS / owner-tier numbers appear as
  tier-named aggregates (ADR-116).

## 20. ADRs (named, not numbered until commit-prep)

- Sportec DFL `Penalty` mapping, direct-free-kick pairing and `BallDeflection` routing (woodwork qualifier)
  (§25; follows the Sportec F4 / F7 draft ADRs).
- The kick resolver, the truth model (tiers, `K`, bracket) and the validation gate table with the pooling
  ladder.
- `refine_restart_types`: native-never-rewritten, subtype, merge, penalty rows, the single renumber point.
- The missed-penalty detector.
- F0 per-corner orientation and the corner-model re-fit.
- The restart origin, including the goal-kick keeper origin (the ADR-024 amendment's code) and the widened
  `tracking_gk` meaning; the `distrust_native_origin` retirement is CONDITIONAL on run (10).
- `attribution_margin` and the taker identity.
- The SkillCorner tracking adapter (§17.7): ball detection `visibility`, frame-level possession, the camera
  footprint companion table and `in_camera_view`; the orientation-flip recording on the conversion report
  (`ltr_flip_by_period`) with the `direction.py` decide/apply split, so frames and footprint cannot disagree
  (amended 2026-10-22, DL ruling; plan CL-15 amendment).
- The training provenance contract: per-dimension masks, frame censoring for occurrence labels, the `reads=`
  declaration gate (§18.2.1).

## 21. Known limitations

- SkillCorner outputs ship per cell wherever a tier passes; cells that fail stay `withheld` / `conflict`.
  Coverage is reported as a number.
- The IDSSE contact base is thin for corners (57 in 7 matches): contested first contact and the short-corner
  taker are withheld by construction unless GS per-signal admission adds coverage.
- Tier-2 transfer truth is only as good as the degradation profile's fidelity; a failed fidelity statistic
  withholds the signals that depend on it (§16.3). The identity-swap proxy misses swaps between players who are
  close together.
- GS subtype truth is a manual LABEL, not the library's location meaning; the mismatch is measured on IDSSE
  only.
- The merge population on SkillCorner is small; the merge may stay withheld (reported).
- The missed-penalty closure holds only where the pooled detection gate passes.
- A same-type same-team repeat restart remains one Part-1 marker row; retakes are reported, not split.
- GI timing is tracking-derived; GI is used only as a bracket and as the `first_possession_time` fact.

## 22. Decision provenance

| Decision | Ruling | Source |
|---|---|---|
| Spec now, constants TBD-pending-run | Q1 = (a) | owner 2026-10-21 |
| `ball_onset` a full validated second source (§6.4) | Q2 = (i) | owner 2026-10-21 |
| One declared static gate table (§15) | Q3 = (A) | owner 2026-10-21 |
| Architecture (§4) | approved | owner 2026-10-21 |
| Conflict arbitration keeps the evidence (§6.5) | Q4 = (z) | owner relays DL 2026-10-21 |
| Truth tiers; video never a gate; D3 blanket SkillCorner withhold superseded (§5) | Q5 | owner relays DL 2026-10-21 |
| Tiers apply to every signal (§15.2) | Q6 = (I) | owner 2026-10-21 |
| Cell key, per-signal metrics, held-out, min-n, stamps; native-never-rewritten, `type_id_events`, subtype, events-first merge, pipeline | round-0 approve-with-changes | owner relays DL 2026-10-21 |
| Restore taker (§7), attribution_margin (§8), video census (§16 run 15), missed-penalty detector (§12), refit by tier (§18.1) | round-1 Bucket 1 | owner relays DL 2026-10-21 |
| B1–B9 corrected designs (§5.2, §6.1, §5.4, §9, §11.2, §6.3, §10, §4.3/§13, §6.4); runs (11)–(16) | round-1 Bucket 2 | owner relays DL 2026-10-21 (DL owns the round-1 relay errors) |
| `first_possession_time` = `gi_next_possession_time` on SkillCorner derived rows; `<NA>` on chain / native; never from `kick_bracket_hi`; `gi=` deprecated (§9.4) | round-2 Q1 (design call 6) | owner relays DL 2026-10-21 |
| Sportec `Penalty` fix in this spec (§25), FIRST Part-2 commit, no separate re-convert, named ADR, Guards 1 and 2 | round-2 Q2 | owner relays DL 2026-10-21 |
| Ratified constants (§16.1), incl. τ_fc 0.40 s, clipped a/b, pitch-scaled penalty spot | round-2 Q3 | owner relays DL 2026-10-21 |
| No Metrica base; GS per-signal transfer-base admission; `first_contact` IDSSE-only (§5.3) | round-2 Q4 | owner relays DL 2026-10-21 |
| Guard 1 (goal reconciliation) on every provider with committed real fixtures (§25.5) | round-3 pre-review | owner 2026-10-21 |
| Run (9) fidelity tolerances, per-region computation, SE-scaled capped margin, per-signal dependency map (a failed statistic withholds only its dependent signals) (§16.3, §5.3) | round-3 pre-review | owner relays DL 2026-10-21 |
| Training provenance contract: per-dimension `provenance_masks`, exclusion for action-level examples, frame censoring for occurrence labels, per-retrain table, `reads=` declaration gate (§18.2.1) | round-3 Q1 (B-23) | owner relays DL 2026-10-21 |
| Sportec penalty result: steps 1–3 `native` with `penalty_result_method`, only step 4 `inferred`; rule 2 concludes "all missed" at 0 (§25.2) | round-3 Q2 (B-35 / C-47) | owner relays DL 2026-10-21 |
| Penalty detector constants evaluated on the rest-window median (`d_gk`, `e_enc`, `k`, `ρ_retake`), the retake definition (§6.7), `W` = 3.0 s and the outcome rule incl. "the score change decides" (§12, §16.1) | round-4 pre-review (C-45, B-34) | owner relays DL 2026-10-21 |
| GS origin truth geometry-only; PFF arrays never truth (§5.3); C-41 in-region first-player kick contact (§6.3); C-35r penalty strata under the licence (§12); C-46 fidelity fit/score split (§16.3); `BallDeflection` in `_EXCLUDED_EVENT_TYPES` (DL retracted "_MAPPED") | round-3 DL review (C) | DL as Reviewer C 2026-10-21 |
| `BallDeflection` routed by `deflection_type`: woodwork → `shot_woodwork_type` on the paired shot (+ deflection-point end), referee → documented exclusion, other → counted unpaired; never fed to `shot_blocked` / `cross_blocked` (§25.4) | round-3 pre-review | owner relays DL 2026-10-21 |
| GS fields (§5.6); DFL first-touch extraction (§5.6); pooling ladder + coverage forecast (§15.3); SkillCorner free-kick launch rule (§6.3); GI direction check withhold-and-count + corpus guard (§4.3); K physical thresholds (§5.2); C-31 kick contact; C-32 physical K | round-2 DL answers | owner relays DL 2026-10-21 (DL owns the C-31 / C-32 relay errors) |
| Scope and breaking changes not a concern; maximum value | owner position | owner relays DL 2026-10-21 |

Author design calls for round-3 review (not owner rulings):

- carried from round 2, accepted by all three reviewers (some with the changes now applied): `winner_source`
  (§9.3); `tracking_gk` reuse with the widened meaning recorded (§10.1); the single renumber point (§11.5);
  the derived `keeper_save` row behind the pooled split gate (§11.3); the observation pad around the strict
  bracket (§6.1); `merged_from_*` as two `Int64` columns (§11.4); the enrichment-constant placement (§17.1);
- new in round 3: the SkillCorner converter emits the two GI bracket facts as converter columns (§17.2);
  `first_possession_status` (§9.4); the `merged_derived_*` content set (§11.4); the signal values on
  refine-created rows (§11.3); `<signal>_cell_resolution` (§15.3); out-of-bracket candidates kept as evidence
  but excluded from arbitration, with `bracket` resolution retained (§6.1, §6.5); the §25 section placement and
  the allow-list reasons (§25.4). (`BallDeflection` in `_EXCLUDED_EVENT_TYPES` with a qualifier pass, the
  `BallClaiming` precedent `sportec.py:190-195`, was accepted by A, B and C in round 3; C retracted its "_MAPPED"
  ruling text. No longer an open call.)
- new in round 4 (accepted by A, B and C): `type_source_events` (§11.2, §11.4); the both-agree-one-inside
  arbitration row resolving `single` (§6.5); the in-region / last-entry restriction combined with the
  restarting-team restriction for the kick contact (§6.3, C-41 + B-32).
- new in round 5 (accepted by A, B and C): adequacy-based base precedence (§5.3, B-39); the `v_kick`
  escalation threshold of 0.10 absolute (§6.3, §16.1, B-41); the penalty consistency guard (§12).
- new in round 6 (accepted by A, B and C; the retype conditionally): the guard's rebound branch retypes the
  Part-1 reconciliation goal row `shot_penalty → shot` (§12, B-45); "the larger adequate base first" for the
  same-meaning signals (§5.3, C-54).
- new in round 7: the explicit §11.1 rebound extension for `repair_kind` rows, with NaN canonical start and
  `start_x_events` / `start_y_events` (B-48); annotate-not-add for an existing kick row (§11.3, C-56); the
  penalty-context + kick-off pairing restriction (§12, B-47; superseded in round 8). `W_rb` = 8.0 s and the
  attribution gate are DL's C-55 specification.
- new in round 8: pairing on penalty context only, with the gated attribution deciding goal / rebound /
  later goal (§12). B-50 and C-57 proposed different fixes to the same clause: C-57 rewords the possession
  condition inside the pairing; B-50 removes the possession condition from the pairing. B-50's form is adopted
  because it also resolves C-57's case (no pairing step depends on a possession) and B's later-goal case, which
  C-57's wording alone leaves open; C-57's release-anchored wording is adopted in rule (b) of the attribution.
  Also: the time-and-team one-kick match (§11.3, B-51); the `penalty_outcome` column (§11.3, §17.1, B-52);
  rule (d) net-retrieval exclusion (§12, C-58). All four accepted by A, B and C in round 8 (B-50 vs C-57
  confirmed by C as DL).

No open owner or DL item remains in the spec text; the freeze waits only on the §16.2 runs.

## 23. Finding resolution

Not vendored (owner decision 2026-11-03): the review-round finding log stays in `fcb91c0`.

## 24. Out of scope

Arm 2 (DEFENCE coupling + corner-GK), parked outside the repo. A video truth gate. The Metrica sample games as
a contact base.

## 25. Sportec DFL `Penalty` mapping and direct-free-kick pairing (events-only; first Part-2 commit)

### 25.1 Defect (measured on all 7 IDSSE matches, committed fixtures)

`Penalty` is in neither `_MAPPED_EVENT_TYPES` (`sportec.py:150-166`) nor `_EXCLUDED_EVENT_TYPES`
(`:176-197`), so the converter drops it as unrecognized. The DFL penalty is ONE event with no paired
`ShotAtGoal`. `final score = ShotAtGoal goals + own goals + DFL Penalty events` holds on 7 of 7 matches, but
Sportec SPADL goal totals match the final score on only 4 of 7: J03WMX (1:2), J03WOH (4:0) and J03WR9 (1:2) each
lose one penalty goal. Sportec SPADL is missing 3 goals across the 7 matches. The one `GoalDisallowed`
(handball) follows a shot recorded as wide, so no disallowed goal counts as a goal.

### 25.2 Mapping

- Add `Penalty` to `_MAPPED_EVENT_TYPES` → SPADL `shot_penalty`.
- `player_id` = `shot_player`, else `penalty_prospective_taker`, else `<NA>` (`player_id` is empty on 3 of 3;
  `shot_player` and `penalty_prospective_taker` are present and equal on 3 of 3).
- `team_id` = `shot_team` / `penalty_team`.
- Start = `X-Position` / `Y-Position` (the rule-snapped spot; not `X-Source-Position`, the incident location).
- `bodypart_id` from `shot_type_of_shot` (`rightLeg` / `leftLeg` → foot).
- Time = `timestamp_seconds`.
- Result (owner ruling, relayed DL; B-35 / C-47). The DFL `Penalty` is a provider-recorded event (taker, time
  and spot present); only its result is computed, from the provider's own final score and shot outcomes. That
  is the Part-1 §5.4 class (result determined from provider-coded fields → `native`), not the G3 class
  (rebuilt events → `inferred`). The reconciliation identity (final = `ShotAtGoal` goals + opponent own goals +
  penalties) holds on 7 of 7 IDSSE matches.

  | Step | Rule | `result_source` | `penalty_result_method` |
  |---|---|---|---|
  | 1 | `shot_outcome_type` present (the legacy format puts penalty fields on `ShotAtGoal`; that path is kept and still types `shot_penalty`) | `native` | `shot_outcome` |
  | 2 | reconciled penalty goals (`whistle_final_result` − `ShotAtGoal` goals − opponent own goals) = 0 → all `fail`; = the team's penalty count → all `success` | `native` | `score_reconciliation` |
  | 3 | an opponent `KickOff` after the penalty (same period) before any on-ball event → `success` (a period-start kick-off never counts) | `native` | `kickoff_order` |
  | 3m | play continued (amended 2026-10-21, DL ruling Flag 3): the first on-ball event after the penalty comes before any kick-off → the penalty did not score. On-ball = a DFL event type that maps to an on-ball SPADL action (NOT a card / substitution / VAR action / `FinalWhistle` / annotation; a VAR-disallowed goal restarts with a free kick, so the rule gives `fail`). Runs with step 3, before step 3b | `native` | `continued_play` |
  | 3b | elimination (amended 2026-10-21, owner ruling, relayed DL; plan review B-09 / C-08), after steps 2 / 3 / 3m, per team: R = reconciled penalty goals − penalties already resolved `success`; U = the team's unresolved penalties; R = 0 → every unresolved penalty `fail`; R = U → every unresolved penalty `success`; repeat until nothing changes | `native` | `score_elimination` |

  **Counting set (amended 2026-10-21, owner approval Decision 4; plan review r3 B-14).** Steps 2, 3 and 3b count
  and resolve ONLY the team's new-format `Penalty` rows. A legacy `ShotAtGoal` penalty is resolved by step 1, and
  its goal is already inside "`ShotAtGoal` goals" in the step-2 subtraction, so it never enters "the team's
  penalty count", R or U. This also handles a match that mixes both formats. Consequence: the 3b branch R = U → `success` is unreachable at match level (R = U implies reconciled goals = the
  penalty count, which step 2 resolves first); it stays implemented and is unit-tested on the pure helper.
  | 4 | still unresolved (amended 2026-10-21, DL ruling Flag 3). With a final score, R′ = clamp(R, 0, U): the earliest R′ penalties `success`, the rest `fail`, so the team total still equals the final score (the Part-1 `missing_goal` precedent). No final score → all `fail` | `inferred` + `repair_kind="penalty_outcome_unresolved"`, or `"penalty_score_inconsistent"` + a report warn when R ∉ [0, U] | `unresolved` |

  **Step 4 split (amended 2026-10-21, DL ruling Flag 3).** An unconditional `fail` at step 4 corrupts the
  pre-penalty VAEP labels, the later score state and Guard 1 when the penalty actually scored (the
  training-exclusion removes only the penalty row). R′ = clamp(R, 0, U) keeps the team total equal to the final
  score; an R outside [0, U] is a match-level inconsistency (`penalty_score_inconsistent`, warned), so Guard 1
  fails by design. `continued_play` and `penalty_score_inconsistent` join the `penalty_result_method` /
  `repair_kind` vocabularies.

  Step 3b is `native` because the result is fully determined by provider-coded fields (the final score, the shot
  outcomes, and a kick-off proven by provider events). The ABSENCE of a kick-off is never evidence of a miss — a
  goal can be followed by the end of the period with no kick-off (IDSSE J03WR9 is a scored penalty followed
  directly by `FinalWhistle`); only the arithmetic concludes `fail`.

  The shared training-exclusion helper therefore excludes only step 4. A consumer wanting a stricter policy
  filters on `penalty_result_method != "shot_outcome"`. `ConversionReport` counts penalties per method.
- `penalty_retaken_penalty == true`: only the final attempt is a shot; earlier attempts are counted as retakes
  (C4 report).

### 25.2b Native DFL own goal (decision 3; DL ruling 2026-10-21, owner-confirmed)

DFL codes an own goal as a top-level `OwnGoal` event, not a `ShotAtGoal` outcome. The step-2 identity (`final
= ShotAtGoal goals + opponent own goals + penalties`) needs it, or a missed penalty can be mislabelled native
`success`. The parse port dropped it (no `_EVENT_TYPE_PREFIX` entry — an upstream lakehouse omission). Fix
(ruled option (c)):

- Parse: add the `OwnGoal` → `own_goal` prefix (adaptation (f)); a `LOCAL HARDENING` warn catches any
  attr-bearing first-child tag with no prefix. The golden-parity slice has no `OwnGoal`, so it is unchanged.
- Converter: `OwnGoal` → ADR-018 `bad_touch` / `owngoal`, added to `_MAPPED_EVENT_TYPES`, body part from
  `own_goal_type_of_shot`. The credit is field-authoritative: `own_goal_team_own_goal_counts_for` (a DFL-CLU
  id) names the credited team, resolved to a role via `team_id_native` ↔ `team`; the owngoal row's `team_id`
  is the conceder; the field RESOLVES it when the native role is `unknown`; a known-role disagreement is a
  counted conflict (`own_goal_attribution_conflicts`). Step 2 counts own goals by RESULT against the
  field-resolved team. (`TeamOwnGoalCountsFor` is a CLU id, not a role label, so the earlier `== team` check
  could never fire; the synthetic tests now use CLU ids.)
- No committed real DFL fixture has an own goal (IDSSE = 0), so coverage is the CLU-id synthetic tests + the
  kloppy spliced-excerpt two-path parse-contract test (gateway and native parse agree).

### 25.3 Direct free kick, two rows

The case does not occur in IDSSE (0 of 164 shots have `shot_after_free_kick`). Add an in-converter pairing, as
Metrica does (`metrica.py:494-511`): a `FreeKick` followed by a same-player `ShotAtGoal` with
`shot_after_free_kick == true` within 5 s gives one `shot_freekick` row, and the `FreeKick` row is dropped.
Unit-tested on a synthetic fixture.

### 25.4 Guard 2 — exact unrecognized-type allow-list (real fixtures)

Converting the 7 IDSSE matches must give `report.unrecognized_counts` keys equal to a declared allow-list, both
directions (the ADR-056 pattern). Today 14 types (112 events) are neither mapped nor excluded: `PenaltyNotAwarded`
18, `Nutmeg` 16, `FinalWhistle` 14, `SpectacularPlay` 12, `VideoAssistantAction` 9, `ChanceWithoutShot` 9,
`PossessionLossBeforeGoal` 8, `Run` 8, `PlayerNotSentOff` 6, `BallDeflection` 4, `Penalty` 3, `SitterPrevented`
2, `CautionTeamofficial` 2, `GoalDisallowed` 1. Each type moves to `_MAPPED_EVENT_TYPES` (`Penalty`) or to
`_EXCLUDED_EVENT_TYPES` with a stated reason; annotation-only types (`Nutmeg`, `SpectacularPlay`, `Run`,
`ChanceWithoutShot`, …) are excluded. After the change the allow-list is empty, and any new unlisted type fails
the gate (the same class of silent drop as the 3.10.1 `CornerKick` alias).

**`BallDeflection` (owner ruling, relayed DL; measured on all 7 IDSSE matches).** The 4 events contain no block
by a player: 3 are woodwork (`postRight` ×2, `crossBar` ×1, at x = 0 or 105, y 30.3–37.5, each 0.75–2.6 s
after a `ShotAtGoal` with `shot_outcome_type="woodwork"`, `deflection_player` = the shooter) and 1 is
`referee` (a `RefereeBall` follows 23 s later). Feeding them into `shot_blocked` would mark 3 woodwork shots
blocked and corrupt the column that TF-51's shot-block rule and the bravery metric read (ADR-046); DFL already
codes blocks on the shot (`sportec.py:1071-1077`). Route by `deflection_type`:

- **Woodwork** (`postLeft` / `postRight` / `crossBar`): pair with the same player's preceding `ShotAtGoal`
  with `shot_outcome_type == "woodwork"`, within 3 s, same period. Set the new Sportec qualifier column
  `shot_woodwork_type` (`post_left` / `post_right` / `crossbar`; §17.5) on the shot row. Where the shot's end
  is the placeholder (end = start; Sportec shots are not in `_DERIVE_END_TYPE_IDS`), set `end_x/_y` to the
  deflection point — where the shot ended, StatsBomb's shot-end meaning. No new row.
- **Referee:** a documented exclusion (no SPADL action meaning; the restart is the following `RefereeBall`),
  counted as `deflection_referee`.
- **Any other `deflection_type`, or woodwork with no pairable shot:** counted as `deflection_unpaired` and
  warned. Never mapped to `shot_blocked` / `cross_blocked` without its own measured evidence.
- **Set membership:** `BallDeflection` is recognized and consumed as a qualifier, not mapped 1:1 to a row. It is
  listed in `_EXCLUDED_EVENT_TYPES` with that reason, exactly as `BallClaiming` is (`sportec.py:190-195`), and a
  qualifier pass consumes it (§22 design call).

### 25.5 Guard 1 — goal reconciliation (real fixtures) and tests

- Guard 1 (owner ruling: every provider): for each match in a provider's committed real fixtures, SPADL goals
  per team (successful `shot` / `shot_freekick` / `shot_penalty` plus `owngoal` results) = the provider's final
  score (IDSSE `whistle_final_result`; required on 7 of 7). Every provider with committed real fixtures gets the
  gate. A provider whose fixtures carry no final-score field is listed in the gate with a stated reason, never
  silently skipped. SkillCorner opendata is covered by the Part-1 G3 reconciliation; the gate asserts that
  coverage rather than duplicating it. The plan enumerates the providers and their final-score fields.
- **Guard 1b (decision D3): kick-off reconciliation on IDSSE ×7** — every non-period-start kick-off by T is
  preceded by exactly one goal for T's opponent since the previous kick-off, and every goal is followed by an
  opponent kick-off (or the period end) with no on-ball event between. Metrica Guard 1b and the Sample Game 2
  bronze fixture are dropped (owner ruling: no value / CI cost).
- **Guard 1c (decision 3, owner-approved): running score on IDSSE ×7** — at every native goal event carrying
  `shot_outcome_current_result`, the cumulative SPADL goals per team (match order) equal it. The per-team /
  per-goal check the totals gate cannot make; it validates penalty + own-goal attribution. Not StatsBomb
  (incomplete kick-off typing); the own-goal element is covered by the CLU-id synthetic + parse-contract tests.
- The registry (`EVENT_FIXTURES`) classifies every committed event fixture and a `GUARD1_DECISION` records,
  per real / excerpt / g3 / spliced fixture, a gated method or an `exempt: <reason>` authority (a test asserts
  the gated sets match the live gates).
- Tests: the 3 IDSSE penalties become `shot_penalty`, `result_id="success"`, `player_id = shot_player`,
  step 2, `result_source="native"`, `penalty_result_method="score_reconciliation"`; a synthetic all-missed team
  (step 2, all `fail`, `native`); a synthetic split case resolved by kick-off order (step 3, `native`,
  `kickoff_order`), and a second-half period-start kick-off that must NOT count; a synthetic inconsistent score
  (step 4, `inferred`, `repair_kind` set, `unresolved`); the legacy `ShotAtGoal` penalty path still types
  `shot_penalty`; a retaken penalty yields one shot; the synthetic direct-free-kick pairing gives one
  `shot_freekick` row; the 3 IDSSE woodwork shots carry `shot_woodwork_type` and the deflection-point end; the
  referee deflection is counted and excluded; a synthetic unpaired deflection is counted and warned;
  `shot_blocked` is unchanged on all 4 deflections; Guards 1 and 2 on the real fixtures.

### 25.6 Release

One coherent events-only commit with its tests green, placed FIRST in the Part-2 sequence, landed only after
the owner sees its file list and diff and explicitly approves that commit (§3 commit gate, B-28) (§12's tier-1
penalty truth and §11.3's "native providers never add rows" depend on it). `docs/context/providers.md`
(Sportec section) is updated in the same commit. No separate re-convert: Part-1 decision D re-converts all
providers at the joint release; the single ADR-065 notice and the CHANGELOG list the change (Sportec gains
`shot_penalty` rows; Sportec goal totals change, +3 across the 7 IDSSE matches; new sportec columns
`shot_woodwork_type` and `penalty_result_method`; `repair_kind` promoted to base `SPADL_COLUMNS` 19 → 20 by
decision D1; Metrica gains an own-goal row for a `BALL OUT`/`…GOAL` event by decision D2; woodwork shots gain
a real end point). Any golden that pins Sportec goal totals is updated in the same commit (the §17.6
carve-out; Guard 1 audits it). The commit lands before the joint retrain.

## 26. SkillCorner GI period-label repair (added 2026-10-25; DL ruling, owner-delegated)

Source: the R.2 #3 range stop. Evidence: `D:\Development\_handoffs\2026-10-25-tf46-part2-r2-gi-period-dl-handoff.md`
§3 (diagnostics `diag_period_census.py`, `diag_period_detail.py`, `diag_raw_periods.py`, `diag_clock_raw.py`,
`diag_implied_starts.py`, `diag_poss_range.py`, `diag_ab_repair.py`). Ruling:
`D:\Development\_reviews\2026-10-25-tf46-part2-gi-period-ruling-C.md` (Q1–Q9). Owner rulings 2026-10-25: the
converter fix belongs in this cycle; the owner delegated the open questions to the DL session ("For the outstanding
questions, I will delegate to the DL session") and relayed its ruling; it lands in the C2 amend (C1 is not rewritten);
the opendata slim fixtures gain `match_periods` (the §26.4 diagnostic reads it) and the GI columns the repair and
`parse_passing_options` read. Amended by the DL in the C2 delta-3 review (2026-10-25,
`D:\Development\_reviews\2026-10-25-tf46-part2-c2-delta3-impl-C.md`, "Rulings on the author's questions" 1, 2, 7): no
feature-glossary entry (§26.5), the symmetric `boundary_block` (§26.3), the new `GiPeriodRepairWarning` category.

### 26.1 Defect

In 7 of 895 SkillCorner matches (owner tier) GI labels the second-half kick-off possession, and its 5–11
`passing_option` events, as period 1. Its frames are inside period 2 by every independent source: the raw tracking
period labels, `match.json` `match_periods` (6 of 7; one has none), the frame order (the first possession of the
second half), and the physical check where evaluable (possessor 0.4–2.3 m from the detected ball). GI computed the
clock of those events on the period-1 basis (residual 0.0 s under the label; +80 to +377 s under period 2), so the
clock is label-derived and not a second witness. Two isolated `on_ball_engagement` events conflict too (2048846,
2052202). In the SPADL output the kick-off pass becomes the last action of period 1, and in 3 matches a synthetic
dribble bridges half-time; `parse_passing_options` carries period 1 on 48 options.

### 26.2 The rule (`silly_kicks/spadl/_skillcorner_periods.py`, one pure helper; numpy, pandas and the numpy + pandas `silly_kicks.id_compat` only — never `silly_kicks.tracking`)

1. **Clock identity.** For `player_possession`, `passing_option` and `off_ball_run`, the GI clock equals
   `offset(period) + (frame_start − S_period) / 10`, offsets = `_PERIOD_START_SECONDS`; exact within one decisecond
   for 100 % of these events in 895 matches. `on_ball_engagement` is not on the frame clock (68–71 %).
2. `S_p` = the mode of `frame_start − round(10·(clock − offset_p))` over the `player_possession` events labelled `p`.
   It equals the raw tracking period start in 895 of 895 matches.
3. `period(f)` = `max{q : S_q ≤ f}`. An event conflicts when `period(frame_start) ≠ label`.
4. **Re-base (delta form, integer deciseconds):** `clock' = clock + (offset_q − offset_p) − (S_q − S_p)/10`, computed
   as `round(10·clock) + 10·(offset_q − offset_p) − (S_q − S_p)` and divided by 10 once. An unrepaired event keeps
   the existing `_parse_time_start` value (byte-identical).
5. The repair runs BEFORE `_to_period_relative` and the chronological sort. The raw `time_start` string is never
   rewritten: the helper returns the repaired period, the repaired continuous clock (seconds) and `period_source`.
6. **Event types.** `player_possession`, `passing_option`, `off_ball_run`: repaired. `on_ball_engagement`: never
   repaired; an OBE whose label conflicts with its frame period is withheld from tackle enrichment and counted
   (`obe_period_conflict`).

### 26.3 Guards (match-level, all-or-nothing; never raise)

A failed precondition repairs NOTHING in the match (output byte-identical to the unrepaired conversion), records the
reason in `period_repair_status` and warns (`GiPeriodRepairWarning`, a separate category):

| Guard | Condition | On failure |
|---|---|---|
| G0 | `frame_start`, `period` and `time_start` are present | `not_evaluable`, no warning |
| G1 | every labelled period that has GI events has at least one `player_possession` | `skipped_g1` + warn |
| G2 | the identity of §26.2.1 holds for EVERY frame-clock event under its own label, within one decisecond (this also verifies the 10 fps assumption per match); and no frame-clock event whose frame lies in another period lacks a readable clock (it could not be re-based; delta-3 review A C2-DELTA-09) | `skipped_g2` + warn |
| G3 | `S_p` strictly increases with `p` | `skipped_g3` + warn |
| G4 (post-condition) | after the repair, every `passing_option` and `off_ball_run` has the period of its associated possession (`associated_player_possession_event_id`), the ids joined through `id_compat` (AGENTS.md id rule; delta-3 review A C2-DELTA-06); without the association or the `event_id` column, a repair that would move a linked event is unverifiable | `skipped_g4` + warn |

No numeric ceiling. The repair shape is classified and reported, not gated: `boundary_block` = for every old → new
period move, the repaired possessions are the possessions of the new period NEAREST the old one in frame order (the
first ones when the new period follows, the last ones when it precedes) — a block at the period boundary, in either
direction (DL ruling in the delta-3 review, 2026-10-25: the symmetric meaning is the intended one);
otherwise `interior`, which warns. Statuses: `clean` (evaluated, no conflict), `repaired`, `skipped_g1` …
`skipped_g4`, `not_evaluable`. When a guard skips the match, OBE conflicts are not withheld either (the output stays
byte-identical to the unrepaired conversion).

### 26.4 `match_periods` diagnostic (not a gate)

The repair decision uses GI facts only (single-sourcing: `parse_passing_options` has no `match_metadata`). When
`match_metadata` carries `match_periods`, each `S_p` is compared with `match_periods[p].start_frame` (start frames
only; the declared end frame is not a reliable bound). A mismatch warns, sets `period_start_mismatch`, and the
repair still runs.

### 26.5 Provenance and single-sourcing

- `period_source` on `SKILLCORNER_SPADL_COLUMNS` (§17.2): `"native"` / `"frame"`. Derived rows take the value of
  the GI possession their period comes from; a synthetic dribble takes the value of the action it precedes (its
  period source). Never `<NA>`. Reflection `invariant`. NOT a feature-glossary entry: the glossary documents
  derived `add_*` / `*_xfns` outputs only (no provider schema column is in it; its `test_no_stale_entries` gate
  rejects one) — DL ruling 2026-10-25 in the delta-3 review (reviewer C, item 1: "Keep the deviation; do not widen the
  glossary contract in this cycle"); the column is documented in the `SKILLCORNER_SPADL_COLUMNS` docstring, §17.2 and
  `docs/context/providers.md`. NOT `repair_kind` (that marks the result and the time unvalidated and drops the rows
  from training); not in `training_exclusion_mask` / `provenance_masks`.
- `SkillCornerConversionReport` gains `period_repair_status`, `period_relabelled` (count per event type),
  `period_repair_shape` (`boundary_block` / `interior` / `None`), `obe_period_conflict`, `period_start_mismatch`;
  `dataclasses.asdict` round-trips.
- `providers/skillcorner/gi.parse_passing_options` calls the same helper: its `period_id` is the repaired period and
  it gains `period_source` (an additive output change, CHANGELOG). An import-allowlist test pins the helper's imports
  (no `silly_kicks.tracking`).

### 26.6 Accepted consequences

For the 7 matches the kick-off pass moves to period 2 at 0.0–0.6 s; in 3 of them the `action_id` order changes and
one synthetic dribble disappears; 48 GK-decision option rows move to period 2. A value change for downstream tables
(an `action_id` renumber is an atomic cross-table migration): CHANGELOG + the C4 lakehouse re-materialize note.

### 26.7 Tests (DL ruling Q8)

Every repair / new-Range test fails at `1007ef6` (recorded in the delta-3 handoff); no-op tests pass on both and
have a failing twin; pandas 2 and 3. (1) Real-data no-op on all 20 opendata slim fixtures (equal to the `1007ef6`
conversion except `period_source`: per-column digests of the dtype kind and the exact values, plus the column order;
status `clean`; same for `parse_passing_options`).
(2) Mislabel-and-restore on all 20, both directions (P2 → P1, the observed defect; P1 → P2, the reverse boundary):
byte-identical restore, `period_source == "frame"` on exactly the restored rows, `repaired`, `boundary_block`.
(3) The re-base is pinned: a relabel without it gives a different conversion. (4) G0–G4 one-sided and two-sided.
(5) Shape `interior` warns; `boundary_block` does not. (6) OBE withheld (both shapes) and the non-conflicting
`direct_regain` still upgrades to a tackle — amended in the delta-3 review (B C2-DELTA-05, C C2-IMPL-31): the
conflicting OBE keeps the LABEL and clock window of the derived interception it would upgrade, only its frame lies in
the other period, and its twin with a frame in the label's period upgrades; deleting the withholding fails the test. (7) `match_periods` diagnostic both sides. (8) Single-sourcing on every
fixture. (9) Provenance registries. (10) Range on the provider basis: the 2007448 shape passes with
`gi_frame_unmapped`; `other_period` / `outside` / `partial_edge` / `no_period_range` each fail with the class;
retained basis fails the 2007448 shape; a period-only key raises `TypeError`. (11) The adapter range (empty
labelled frames in, period-`None` frames out). (12) `interval_agreement` drops absent frames; a present frame with no
possessor stays a miss. (13) The import allowlist. (14) R.2 aggregates per data profile.

### 26.8 R.2 pre-registered expectations (on the new staging; any deviation stops the run and returns to the owner)

- `period_repair_status == "repaired"` in exactly the 7 matches of handoff §3.1 (1766992, 1800188, 1874585,
  1904770, 1949363, 2008992, 2037362) and `"clean"` in the other 888; 0 `skipped_*`; 0 `period_start_mismatch`.
- Relabelled: `player_possession` 7, `passing_option` 48, `off_ball_run` 0; all 7 shapes `boundary_block`.
- `obe_period_conflict` = 2 (2048846, 2052202).
- `range_basis == "provider"` in 895 of 895; `range_violations` empty in 895 of 895.
- 895 of 895 matches pair, including all 20 known-paired opendata matches (2007448 included).
- The lag-profile median guard and the bar guard keep their rules; the bar is re-derived on the new staging.
- A stale staging fails mechanically: R.2 requires `range_basis == "provider"` and the `period_source` column on the
  staged actions.

## 27. Run (9) degradation store and fidelity corrections (added 2026-10-25; CL-42)

Source: the fourth R.2 stop. At `6741710` the `kick_cells` `score` stage was killed by the kernel OOM killer
(`kern.log` 2026-10-26T00:30:56Z, 16 workers at about 14 GB anon-rss each, 4 deaths each, 0 units done). The
calibrate output was 23.8 GB of JSON, and every score unit re-parsed it (`RunContext.upstream` has no cache).
Probe (aggregates only): `D:\Development\_handoffs\tf46p2-r2-diag\camera_probe_6741710.json` (895 SkillCorner matches,
25.0 GB of camera JSON, 1,511,386 carried sequences of 365,203 unique, 38.2 M unique frames). Rulings: the DL ruling
`D:\Development\_handoffs\tf46p2-r2-diag\dl_ruling_cl42_2026-10-25.md` (findings 1–6, store items a–k) and its addendum
`D:\Development\_handoffs\tf46p2-r2-diag\dl_ruling_cl42_addendum_2026-10-25.md` (gaps 7a–7n, Q1–Q11, the corrected
held-out form). Owner rulings 2026-10-25: findings 4, 5 and 6 are in CL-42; 7a–7k are in CL-42 ("7a-k fold in");
7l, 7m and 7n are in CL-42; the owner confirms the DL reading of ADR-116 (the camera sequences and the redetection
samples are row-level derived data and never go into `docs/research/`; ADR-116 needs no change). CL-42 lands in the
C2 amend. R.2 then reruns from scratch (`tf46p2_truth` first, one `run_commit`, C-03); the §26.8 census is unchanged.

### 27.1 Camera-follow windows: raw cells, one pooling, kick alignment (findings 1, 4)

- **One raw-window function, one pooling function.** `camera_follow_windows` returns the RAW windows of the
  calibration matches (one per event window, never pooled), each with its match id, period, start frame, ball bin
  and phase. `_pool_cells` / `_pooling_record` are the only pooling functions. `fit_camera_follow` = the windows,
  then the pooling. The run (9) store builder (§27.4) pools ONCE per data profile over the raw windows of the
  profile's calibration matches. "Fewer than 30 calibration events" counts the profile's calibration set, never one
  match. A profile unit emits raw windows only.
- **Kick alignment.** One function, `restart_frame_alignment(frames_index, restarts)`, gives every frame its phase,
  its restart (the restart whose window [s0, `t_kick` + 6 s] contains the frame and whose s0 is the latest; on a
  tie, the later kick) and `kick_relative_s` = t − `t_kick` of that restart (`kick_bracket_hi` when `t_kick` is
  withheld; NaN in open play). `event_phase` reads the same function, so fit and scoring share one rule. The profile
  unit computes it from `_sc_restart_table(run.actions)`, the table `event_phase` reads. A restart-phase run of
  frames splits where its restart changes (one window per restart event; author choice under finding 4).
- The camera-follow numbers change (a correction toward §5.3). No constant changes.

### 27.2 The restart table rule for s0 source (iii) (gap 7k)

- s0 source (iii) = the time of the last live-play event strictly before the restart EVENT: the events with
  `time_seconds` < the restart row's own `time_seconds`, the restart row excluded by id, `is_live` applied where a
  provider supplies it. One function (`stoppage_time`) serves fit and scoring; sources (i) and (ii) are unchanged.
- The restart tables carry `restart_id` and `t_event` (the restart row's own time). Run (9) reports, per data
  profile, the count of rows per s0 source and the count where the pre-CL-42 rule picked the restart row itself
  (`kick_cells_s0_diagnostic`).

### 27.3 One held-out split per data profile (finding 5, corrected; replaces "(LOMO)" in §16.3 for run (9))

- `heldout_split(match_ids, seed=46, share=0.20)`: sort the profile's unique SkillCorner match ids, permute them with
  `numpy.random.default_rng(46)`, hold out the first `max(1, round(0.20 × n))` (Python `round`; no tie occurs for
  an integer n), and keep the rest as the calibration set. The split does not depend on the input order.
- Fit on the calibration set only: the degradation parameters (`degradation_params`), the camera store, the
  redetection store and the identity-swap rate. Score fidelity on the held-out matches only. Every transfer run
  (run (9) and the walkers) uses that one model. The footprint gate (§16.1) stays a gate over all the profile's
  matches; it is not a fit.
- Asserted: no held-out id enters any fit (`HeldOutLeakError`), in calibrate and in each store unit. The camera store
  keeps `seq_match` (the match of each sequence), which proves the exclusion.

### 27.4 The degradation store (findings 2, 3; ruling items a–k)

- **Stage.** `tf46p2_kick_cells` gains the stage `models` between `calibrate` and `score`. Its units are
  `cam_<pp>_<ss>` (one camera store per data profile `pp` × `cam_seg_s` ∈ {10, 5, 15} s, `ss` = the seconds,
  zero-padded) for each profile whose footprint gate passes, and `redet_<pp>` (one redetection store per profile).
  `pp` = the profile's position in the sorted profile keys of the calibrate output. Each unit result is a `FileSet`
  (ADR-128), so it lives in the `models` generation and follows the stage token.
- **Files** (`.npy`, written with `allow_pickle=False`, float64 / int64 only; `.npz` is not used because it does not
  memory-map):
  - camera: `frames` (N, 9) float64 (8 ball-relative corner coordinates + the kick-relative or window time),
    `seq_offsets` (S + 1), `seq_match`, `seq_period`, `seq_start_frame`, `seq_cell` (cell id = 2 × ball bin + phase,
    open = 0, restart = 1), `first_corners` (S, 4, 2), and the pooled membership `pool_cells`, `pool_offsets`,
    `pool_members` (index arrays into the sequences).
  - redetection: per off-camera bin `jump_<k>`, `angle_<k>`, `match_<k>` (k = the bin position).
  - `manifest.json`: canonical JSON (sorted keys, no whitespace) with the schema (`tf46p2-camera-store-1` /
    `tf46p2-redetection-store-1`), the data profile, `seg_s`, `min_events`, the calibration ids, the per-cell
    aggregates and pooling record, the totals, and per file its dtype, shape and sha256.
- **Order.** Sequences sort by (match id, period, window start frame); redetection samples by match id, keeping the
  within-match order. So the bytes do not depend on unit order or worker count.
- **Handle.** The model sha256 is the sha256 of the manifest bytes. The `models` output carries, per profile, the
  handles (sha256 + the store-relative unit path) and the aggregates only.
- **Float64.** JSON floats round-trip float64 exactly, so the storage change alone changes no number.
- **Reader** (`scripts/_tf46p2_store.py`, `open_degradation_store`): a per-process cache keyed by sha256; a host cache
  (`SILLY_KICKS_TF46P2_STORE_CACHE`, default `~/.cache/silly-kicks/tf46p2-stores`) verified once per host under a
  lock: a local store is hashed in place and recorded; a remote store is materialised into `<cache>/<sha256>/` and
  verified there. Arrays open with `numpy.load(mmap_mode="r", allow_pickle=False)`, and their dtype and shape must
  equal the manifest. Fail closed with named errors: `DegradationStoreMissingError` (absent), and
  `DegradationStoreIntegrityError` (a hash, dtype or shape mismatch, or a pickled array).
- **Vectorised draw.** `_choose_sequence` reads the stored first-frame corners (S, 4, 2): the same distances, the same
  `argsort`, the same rng call order. The chain is bit-identical to the list path (tested).
- **Upstream cache.** `RunContext.upstream` caches per process, keyed by the store and the generation path; a cached
  value is reused only while the result's `.sha256` sidecar is unchanged, and the caller gets a copy.
- **Typed unit rows.** A unit frame may carry typed rows (null `payload`) next to the JSON records: the profile unit
  emits its raw camera windows and its redetection samples as parquet `list<double>` columns, never as JSON text.
  `split_records` reads the JSON rows; `typed_records` reads the typed rows.
- **ADR-116.** The calibrate output, the `models` output and every research table carry handles and aggregates only:
  per cell the raw n, pooled yes / no, the pooled-in bins and n after pooling; the sequence and frame totals; the
  calibration and held-out ids; `seg_s`, `min_events`, the schema and the sha256; for redetection the n per bin and
  the sha256. The gap-length samples in `params` (lengths only, no time, no position) and the per-match statistics
  stay (admissible).
- **Walkers.** `origin`, `penalty` and `subtype_merge` resolve the stores by the sha256 in the upstream
  `kick_cells_profiles` table (through their own `--store`) and fail closed when a store is absent or its hash
  differs.

### 27.5 Identity swaps (gaps 7b, 7c; DL amendment to the §16.3 identity-swap row)

- **Proxy (amended).** A jump counts only between two consecutive DETECTED frames of one id (speed > 12 m/s). The
  exposure is detected player-time (the clipped dt of those frame pairs). A swap inside an off-camera spell shows
  under R2 (the off-camera position error), so it is not counted twice.
- **Rate.** The SkillCorner detected→detected proxy jumps per detected player-hour, fitted on the calibration set.
- **Injection** (`inject_identity_swaps`; order: camera mask → swap injection → off-camera replay; its own seed
  stream `default_rng([seed, 2])`): per reference period, a Poisson count with mean (rate / 2) × the period's
  detected player-hours (one event gives two proxy jumps, one per id). Each event: a frame chosen uniformly over the
  period's frames and a detected player chosen uniformly, paired with its nearest detected same-team player beyond
  the proxy distance (12 m/s × the target frame interval). With no partner, redraw up to 20 times (author choice
  under the ruling) and count the misses. The ids (`player_id`, `is_goalkeeper`) stay exchanged until the first frame
  where either player is not detected, or the period end; the revert adds no detected→detected jump. Reported limits:
  swaps closer than the proxy distance are not injected; the duration rule is an assumption (SkillCorner has no swap
  ground truth).
- **Rule.** The exact conditional method with an overdispersion correction: on each side φ = max(1, Pearson χ² /
  (m − 1)) over the per-match counts against the exposure-expected counts (φ = 1 when m < 2); x / φ and E / φ enter
  the method; given N = x_d + x_t, the Clopper–Pearson 95 % interval of p = x_d / N maps to RR = p / (1 − p) ·
  E_t / E_d. Pass when the whole interval lies in [0.67, 1.5]. N = 0 on both sides is `not_evaluable`. No margin.
  φ, RR and the interval are reported.

### 27.6 The fidelity method (gaps 7a, 7d–7h, 7l; Q1, Q4–Q7)

- **Inventory** (one list, `FIDELITY_INVENTORY`): per region (`REGIONS`): `ball_detection_rate`, `fill_share`,
  `gap_distribution`, `noise_sd`, `player_visibility_near_ball` (OPEN-PLAY frames only), and
  `player_visibility_near_ball_restart` (restart-phase frames only, phase from `restart_frame_alignment`); region
  `all`: `frame_rate`, `z_source_mix`, `kick_window_miss_rate`, `nearest_player_visible_at_contact`,
  `identity_swap_rate`, `fill_share` (overall), `player_visibility_all` (the share of ALL player-frames detected, at
  any distance); region `all`, per off-camera bin: `off_camera_position_error`. Every global statistic is in region
  `all`, never in `rest` (one of the four regions).
- **Definitions.** `nearest_player_visible_at_contact`: per TF-79 contact of the side's own pipeline run, the nearest
  player row of the contact frame (either team, detected or not) to the contact position; the statistic is the share
  of those players that are detected (the SkillCorner detection flag; the degraded `visibility` after the mask). A
  contact frame with no player row is excluded and counted. `off_camera_position_error`: `redetection_error_stats` on
  each side, per bin. The camera mask also writes `in_camera_view` on the degraded frames (inside True, outside
  False, a null footprint `<NA>`), the field SkillCorner ships (§17.7).
- **Rules.** Fixed values: |Δ| ≤ 0.03 for the ball detection rate, the kick-window miss rate, both near-ball
  visibilities and the nearest player at contact; |Δ| ≤ 0.05 for `fill_share` per region and `player_visibility_all`;
  |Δ| ≤ 0.03 for `fill_share` in `all`; the noise ratio within [0.85, 1.15]; KS ≤ 0.10 AND p90 within ±20 % for the
  gap distribution and each R2 bin; exact for the frame rate and the `z_source` mix; §27.5 for identity swaps.
- **Margin rule.** B = 2000 match-cluster bootstrap draws over the held-out SkillCorner matches; seed = the first 8 hex
  digits of sha256("46|profile|region|statistic|bin"); each draw recomputes the target with its own pooling rule
  (`_pooled_stats`); SE = the SD (ddof = 1) of the draws. Absolute difference with tolerance t: margin = max(t,
  2 SE), cap 2t. Noise ratio: half-width = max(0.15, 2 SE / T), cap 0.30. p90: relative half-width = max(0.20,
  2 SE(p90) / p90), cap 0.40. KS: bound = max(0.10, the p95 of KS(T*_b, T)), cap 0.20. No margin for the exact
  statistics or identity swaps. An uncapped margin above its cap FAILS CHARACTERISATION (`passed` False). Fewer than
  2 held-out matches with the statistic: `not_evaluable` (the SE is not defined; author choice under the ruling).
- **No silent pass.** A required row that is missing or non-finite on either side is `not_evaluable`; a KS with an
  empty side and an R2 bin with n = 0 on either side are `not_evaluable`. Each row has one `status`: `passed`,
  `failed`, `failed_characterisation` or `not_evaluable`, with its reason. Every status except `passed` withholds the
  statistic's dependents and is reported.
- **Keeper split (7i).** `keeper_undetected_split` per profile, on the SkillCorner matches and on the degraded frames
  (counts only, not gated).

### 27.7 Row-level withholding (gaps 7j, 7m, 7n; Q9)

- Every score row carries `source_path`, written when it is scored. `kick_time`: the `kick_time_source` (the contact
  path when the `t_kick` value came from `tf79_contact`: `tf79_contact`, `conflict_resolved_tf79_contact`, or
  `both` with the `tf79_contact` agreement winner); `taker` and `first_contact`: `tf79_contact`; `subtype`:
  `event_time` (the window starts at the event time); `origin`: the origin tier, with `@<kick_time_source>` for
  `t_kick_ball` (a launch-contact read when that source is a contact path); `merge`: the union of its taker and
  origin paths; `penalty`: `detection`, `attribution` or `split`.
- `row_fidelity_dependencies(signal, source_path)` = the ball statistics, plus the contact statistics when the row
  reads a contact (`kick_time` / `origin` on a contact path, `taker`, `first_contact`, `merge`, the penalty `split`),
  plus `player_visibility_all` for every penalty row. It is asserted to be a subset of `FIDELITY_DEPENDENCIES`
  [signal]. `FIDELITY_DEPENDENCIES["subtype"]` gains the contact statistics (7m: the B-33 subtype row);
  `player_visibility_near_ball_restart` joins the contact statistics; `FIDELITY_TOLERANCES` gains the two new
  statistics.
- One shared function (`withhold_transfer_rows`) drops a TRANSFER row whose dependencies meet a non-passed statistic
  of its profile, and reports the counts per profile × signal × source path × statistic. Every walker that scores
  transfer rows (`kick_cells`, `subtype_merge`, `penalty`, `origin`) applies it before its aggregate, from the
  `kick_cells_fidelity` table (the walkers read it upstream; a missing table refuses). The aggregate gains
  `fidelity_dependencies` (the union over its rows; empty for reference rows). `tf46p2_gate_table.py` fails closed
  (`FidelityWithholdingError`) when a surviving transfer aggregate row depends on a non-passed statistic.

### 27.8 Camera sensitivity (finding 6)

- Run (9) evaluates `cam_seg_s` at 5 / 15 s (each with its own store) and `cam_stitch_k` at 2 / 8 (on the base
  store). A variant reports its fidelity statistics and verdicts only (`kick_cells_sensitivity`, which also holds
  the noise rest-window variants of §27.12; renamed 2026-10-26 from `kick_cells_camera_sensitivity`); no variant
  score row enters any aggregate. Gating uses the ratified values.

### 27.9 Cost

- `apply_camera_follow` and `_player_visibility_near_ball` are vectorised; each is tested bit-identical to its loop.
- `replay_offcamera_error` writes its displacements once (author, 2026-10-26, from the measurement: 386 of 398 s
  of one transfer run were one `.loc` write per off-camera spell, about 331,000 spells once 7o makes the camera
  spells real). The same spells, rng draws and arithmetic; tested bit-identical to the per-spell writes.
- Before W = 16 on the DGX: one score unit with its peak RSS and wall time recorded, then W = 2, then the full run.

### 27.10 Tests (each fails under the named mutant)

- Per-match raw windows + the profile merge equal one fit on the union (mutant: per-match pooling restored). A cell
  with ≥ 30 events at profile level but < 30 in each match does not pool; a cell under 30 takes raw neighbour
  sequences only.
- The store round trip gives bit-identical sequences and a bit-identical chain for a fixed seed; the sha256 is
  unchanged under a permuted unit order; a one-byte change, an absent store and a pickled array fail closed with a
  named error; a second open in the same process reads no file again; the walkers fail closed on an absent or wrong
  sha256.
- No per-frame array in the calibrate output, the `models` output or the profiles table; the ADR-116 gate passes.
- `kick_relative_s` is 0 at a planted kick frame and the chain maps `t_kick` to that frame (mutant: the column
  dropped). The s0 (iii) rule excludes the restart row (mutant: the pre-CL-42 rule).
- No held-out id enters a fit, and the split is the same for a permuted input order.
- A noisy target fails characterisation (mutant: `failed_characterisation` always False). Textbook counts give the
  textbook interval, and φ > 1 widens it. A re-detection jump is not a swap; a detected→detected jump is. The
  injection is seeded, and its expected event count = rate / 2 × exposure. No global row is in region `rest`. The
  inventory is complete on a real fixture (mutant: one statistic dropped).
- 7m and 7n: one mutant each (the subtype map without the contact statistics; a walker that skips the withholding;
  the gate-table guard deleted). The four sensitivity values are recorded.
- The vectorised `apply_camera_follow` and `_player_visibility_near_ball` equal their loops.

### 27.11 Defects found by the DGX cost measurement (author, 2026-10-26; for the delta-4 review)

- **7o — the degraded player detection flag was discarded.** IDSSE and GS frames ship `visibility` = None on every
  row. `degrade_to` copied the carried flag (`detected`) to ball rows only and then dropped it, so every degraded
  player inside the camera was `<NA>`, read as undetected by every visibility statistic, and `keeper_undetected_split`
  refused the all-null flag (the measurement's score unit failed on it). Fix: a reference row without a source flag
  takes the carried flag; the camera mask alone makes a player undetected. Tested.
- **7p — float32 frames and the position writes.** `_inject_ball_noise` and `replay_offcamera_error` wrote float64
  displacements into float32 columns (ADR-106 storage): pandas 2 upcasts the column with a FutureWarning, pandas 3
  raises. Fix: an explicit float64 upcast before the writes (the values pandas 2 stored). Tested under
  `FutureWarning` as an error.

### 27.12 Degradation defects A–F (added 2026-10-26; DL ruling and owner decisions; for the delta-4 review)

Source: the first honest fidelity table, on the DGX staged subset `cl42sub` (40 calibration and 10 held-out
SkillCorner matches; one GS and one IDSSE unit): 3 passed, 1 failed characterisation, 31 failed. The causes are
defects in the degradation itself (§5.3), which CL-42's honest scoring exposed. Evidence:
`D:\Development\_handoffs\2026-10-26-tf46-part2-cl42-degradation-defects-dl-handoff.md`, the probes in
`D:\Development\_handoffs\tf46p2-r2-diag\` (`camera_side_probe.json`, `defects_c_f_probe_results_2026-10-26.md`).
Rulings: `D:\Development\_handoffs\tf46p2-r2-diag\dl_ruling_cl42_defects_2026-10-26.md` (saved verbatim).
Owner decisions, 2026-10-26: A–D join CL-42; one combined delta-4 review (CL-42 + A–F); E joins CL-42; F joins CL-42,
and its two rest-window bounds join §16.1 as ratified constants with a ±50 % sensitivity in run (9); the GS role
waits for a DL ruling (`D:\Development\_handoffs\2026-10-26-tf46-part2-cl42-gs-role-dl-handoff.md`), and A, B, D,
E and F do not depend on it.

**Stage order (one entry point, `_tf46p2_pipeline.degrade_reference` → `degrade_to`).** B → C (pending, below) →
D (camera chain and mask in the camera frame) → E (in-view misses) → A (ball gaps on live frames) → F (ball noise)
→ identity swaps → the off-camera replay (R2). Each stochastic stage has its own seed stream: the chain
`default_rng(seed)`, swaps `[seed, 2]`, A `[seed, 3]`, E `[seed, 4]`, F `[seed, 5]`. kick_cells and the walkers
build the same per-unit inputs (`reference_context`: the restart table, the native restarts, K, the noise anchors),
so every transfer run of a unit is the same degradation.

- **A — ball gaps** (`scripts/_tf46p2_degrade.py`).
  - Statistic, both sides: a gap is a maximal run of undetected ball frames in the time-ordered ball stream of one
    period. Its length is the time from the last detected frame before the run to the first detected frame after
    it. Its region is the region of the last detected frame before the run. A run that touches a period start or
    end is censored: it is excluded and counted (`n_excluded`). Detection rate and fill share per region: an
    undetected frame takes the region of the last detected frame of the period; frames before the first detected
    frame of a period have no region and are counted (`ball_unregioned_frames`). The position of an undetected
    frame is never read.
  - Fit, calibration matches only: per cell (region, phase; the phase from `restart_frame_alignment`), h = gap
    starts / the detected live frames with a next frame in the period, summed over the matches; L = the uncensored
    live lengths. A gap that holds a null-footprint frame is a camera gap: the chain makes it, so it is left out of
    h and L and counted (the SkillCorner camera / live split, `kick_cells_gap_split`). A cell with fewer than 30
    gaps pools its region's two phases, then every region of its phase (recorded). A profile whose footprint gate
    fails has no chain, so its fit counts every frame live.
  - Injection, per reference period, after the camera chain, on live-footprint frames only: at each detected frame
    in cell c with a next frame in the period, a gap starts with probability h_c; L is drawn from L_c with
    replacement; the frames with t_i < t < t_i + L become undetected and the next frame stays detected; a gap that
    reaches the period end is cut (censored). The kick-window miss rate is not fitted; it stays a held-out check.
  - Author choices: an end-censored live gap counts as a start in h (its start is observed), not in L; the gap
    count of the pooling rule is the uncensored L count.
- **B — the exact frame-rate grid** (`resample_exact_grid`). Per period and per object, sorted by time, duplicate
  timestamps collapse to the first row by source `frame_id`; the duplicates are counted, and counted again when
  their positions differ by more than 0.01 m. The period grid is t_k = k / rate from the first source time of the
  period rounded up to the grid. `x`, `y` and the ball `z` interpolate linearly between the two bracketing source
  frames of the same object, only when they are at most 2 source intervals apart; else the object has no row at
  that grid point. Every other column comes from the nearer source frame (a tie takes the earlier). The new
  `frame_id` values are consecutive per period and unique in the match; `source_frame_id` carries the nearer
  source frame; `frame_rate` = the target. The staged derived columns of the UNDEGRADED track are removed
  (`x_smoothed`, `y_smoothed`, `_preprocessed_with`, `vx`, `vy`, `accel_x`, `accel_y`, `accel`, `pos_var`, `vel_var`,
  `accel_var`); the base-schema `speed` and `speed_source` are kept and set to null (author choice: the 19-column
  schema stays complete). The frame-rate statistic stays "exact": |1 / median positive dt − target| ≤ 1e-6 ×
  target. Join audit: the transfer run joins the reference events to the degraded frames by time only
  (`simulated_bracket_ids` maps each anchor time to the degraded frames' own ids); tested. Author choices pending
  the DL (GS-role handoff Q5): the source interval = 1 / the frames' nominal `frame_rate`; one grid per period for
  every object; a ball record missing at the source is a hole (bridged within 2 intervals, else no row), not an
  undetected frame.
- **C — identity swaps.** No change in this section. The DL rejected a smoothed-copy proxy and a report-only rule.
  Diagnostic C (i): the null rate at 10 Hz is 232–258 per hour on GS 10502 and 0.028 per hour on IDSSE
  DFL-MAT-J03WMX (the target 0.53 per hour). Diagnostic C (ii): the raw GS records carry `visibility` (`VISIBLE` /
  `ESTIMATED`) and `confidence` per object; 0.31 of the GS 10502 player positions and 0.63 of its ball positions are
  `VISIBLE`. GS is broadcast-derived tracking, not full observation. The player-position stage the DL ruled "if (ii)
  shows full observation" is not built; the GS role waits for the DL. Until then the §27.5 rule fails closed on a
  null rate above the target. The run reports the null rate per profile × provider before the injection
  (`kick_cells_swap_injection`).
- **D — one camera side per match.** The probe found the camera at raw low y in both periods of all 895
  SkillCorner units, and on opposite sides in the staged frame. Fit: a flipped period is mapped into the camera
  frame through the library (`orient_camera_footprint` for the eight `cam_*` point pairs, `reflect_columns` for the
  ball; the default SPADL dims); an unresolved flip fails closed. The store schema is `tf46p2-camera-store-2`.
  `calibrate` asserts that every SkillCorner period has its camera at low y in the camera frame
  (`CameraSideError`; author choice: the DL premise, made a stop). Draw: per reference period σ_p = +1 when the
  camera is at staged low y, else −1 (staged low y when side = low and the period is not flipped, or side = high and
  it is flipped); the chain is built on the ball timeline in the camera frame and mapped back with the same library
  reflection. Side assignment: within each reference provider, the units ranked by sha256("46|provider|match id");
  the first ⌈n/2⌉ get low y; the sides are recorded (`kick_cells_camera_sides`) and the walkers read them upstream
  (a unit without a side fails closed). Reference flips (author default pending the DL, GS-role handoff Q4): period
  2 is reflected against period 1 (the teams change ends); an extra-time period fails closed. Re-measured per run:
  the side probe on the degraded chain (`kick_cells_camera_side_check`: the camera-frame value must be negative in
  every period), and the chain area and in-view share against SkillCorner (`kick_cells_coverage`).
- **E — in-view misses** (DL defect E). After the mask and before the swaps, a seeded 2-state chain (detected /
  missed) per player over its in-view frames: the first in-view frame of a run draws P(missed), each next in-view
  frame moves by the probabilities of the previous frame's cell. Never per-frame draws. Fit on the SkillCorner
  calibration rows with `in_camera_view` True: P(miss | detected), P(detected | missed) and P(missed) on the first
  in-view frame, per cell (the ball's carried region, the phase, ≤ 10 m from the ball, ≤ 2 m from the footprint
  edge). A cell with fewer than 30 miss spells pools within its phase: over the edge band first, then over the
  region (recorded, `kick_cells_inview_model`). The R2 replay then displaces the in-view miss spells too. A profile
  whose footprint gate fails has no chain and no E.
- **F — ball noise** (DL defect F). SUPERSEDED 2026-10-26 by §27.13 item 3 (the rest window is retired; the noise
  statistic is the second difference). Kept as the record of the retired rule. Measured before: the fitted SD was 28.5–54.0 m per region (the old statistic
  measured where the ball rests, not noise). Statistic, both sides: the rest window [t_kick − `noise_rest_lo_s`,
  t_kick − `noise_rest_hi_s`] of each corner, goal kick and free kick (`kick_bracket_hi` when `t_kick` is withheld;
  no throw-in), on the detected frames, at least 5; the window's 2-D SD about its mean; per region of the window
  mean, the RMS over the windows; a region with fewer than 30 windows takes the all-region value (recorded). The
  fidelity pooling and its bootstrap apply the same rule per draw. Injected SD per region = sqrt(max(0, sd_SC² −
  sd_ref²)); sd_ref is measured the same way on the reference grid track before any injection, with the windows at
  the reference kick truth K (author choice: K is the reference kick the scorer reads). The injection is AR(1) per
  axis along each period's ball stream, with the lag-1 correlation fitted in the same SkillCorner windows; the
  lag-1 correlation is reported on both sides and never gated.
  - Sensitivity (owner, 2026-10-26): `noise_rest_lo_s` 0.75 / 2.25 s and `noise_rest_hi_s` 0.25 / 0.75 s, each with
    the other at its ratified value, each a full transfer run reported in `kick_cells_sensitivity` (statistics and
    verdicts only). Note: the variant [t_kick − 0.75 s, t_kick − 0.5 s] holds 2–3 frames at 10 Hz, below the 5-frame
    minimum, so it is `not_evaluable` by construction.

**Tests (each fails under its mutant)** — `tests/scripts/test_tf46p2_cl42_defects.py`:
- A: a 1 s gap after a 60 s excursion measures 1 s (mutant: the region-masked stream); an undetected frame with a
  NaN position takes the carried region (mutant: `rest`); fit then inject on a synthetic stream reproduces the gaps
  per detected frame and the lengths (mutant: the whole pooled sample in one match); a gap with a null-footprint
  frame stays out of the fit (mutant: included).
- B: 25 Hz and 29.97 Hz tracks give dt = 0.1 within 1e-9 with linear positions (mutant: the nearest frame, 12.5 Hz);
  a hole wider than 2 intervals is not bridged; duplicates are counted; the derived columns are removed; the
  simulated bracket ids exist in the degraded frames (the join audit).
- D: a flipped window equals the mirrored unflipped window, with the same signed-area sign; the side probe holds
  on the chain (mutant: no reflection); the side split is balanced and order-independent.
- E: a known detected share and mean spell length are reproduced (mutants: per-frame draws; no stage).
- F: a known placement jitter is recovered while the ball rests at the 4 corner flags (mutant: the old formula);
  the injected noise has the fitted lag-1 correlation.

### 27.13 The GS role, the chain fixes and F on the second difference (added 2026-10-26; DL ruling and owner decisions)

Source: the DL ruling 2026-10-26 on `D:\Development\_handoffs\2026-10-26-tf46-part2-cl42-gs-role-dl-handoff.md` and
`D:\Development\_handoffs\2026-10-26-tf46-part2-cl42-af-results-dl-handoff.md`, saved verbatim with the owner decisions
at `D:\Development\_handoffs\tf46p2-r2-diag\dl_ruling_cl42_gs_role_af_results_2026-10-26.md`. The DL owns three errors:
the round-2 Q4 premise "K on GS optical tracking"; the E premise "about 84 %" (an author inference over two
denominators; the measured SkillCorner in-view detected share is 0.977); and the F fallback to `kick_bracket_hi`
(after the ball flight on a SkillCorner derived row). This section supersedes §27.12 F and the parts of §5.3, §15.3
and §16.1 named below. Superseded in turn by §27.14 (DL ruling v3, 2026-10-26): item 3 (F: the white noise on d2
and its ratio gate), the restart-window "holds the edge of the s0-aligned part" rule of item 2, and the stage order.

**GS role (Q1, Q2; owner decision 2026-10-26: the DL route, no new base).** The raw GS records are broadcast-derived
tracking: every object carries `visibility` (`VISIBLE` / `ESTIMATED`) and `confidence`; in match 10502, 0.31 of the
player positions and 0.63 of the ball records are `VISIBLE`, and 26 % of the records have no ball. GS is not
"optical tracking" anywhere in this spec; every sentence that says so is corrected by this section.
- GS leaves the transfer tier for every signal (`TRANSFER_BASE_PROVIDERS` = IDSSE only; `degrade_reference` refuses
  another provider; `transfer_runs` yields nothing for GS). §5.3 base precedence reduces to IDSSE; `admitted` is
  True for IDSSE rows only (`aggregate_scores`). `choose_base` is unchanged: with GS never admitted it selects IDSSE
  or withholds.
- GS leaves tier 1 for every signal whose truth reads GS tracking: `kick_time` (K), `ball_onset` (K) and origin
  geometry (the corner arc, the throw-in y). The truth walker keeps only the GS set-piece code counts for a GS unit.
- GS keeps the checks whose truth is a PFF event label, reported as "the library on GS broadcast-derived input", and
  admitting nothing (`gs_admission_table`: `gs_tier1_passes`, `gs_admitted` always False):

  | GS check | Truth | Scored instance (anchor) | K used? |
  |---|---|---|---|
  | `taker` (kick_cells, the reference arm) | PFF `gameEvents.playerId` on the set-piece event | the native restart row (`restart_id`); the kick-contact player of the SkillCorner rules | no |
  | `subtype` (subtype_merge) | PFF `possessionEventType` | the native restart, aligned by its event time (`t_event`) | no |
  | `penalty` (penalty) | PFF penalty events | a detection matched within W of the detector's own release | no |
  | `merge` (subtype_merge) | PFF set piece + shot | the native event pairs of `merge_simulation` | no |

- Owner additions: (1) the shortfall per withheld cell: n, m and the extra IDSSE matches needed to reach the bar; (2)
  the corner-project impact stated on its own: `first_contact` was IDSSE-only before (no change); the corner cells
  whose GS coverage drops are the `taker` short-corner cell and the `kick_time` corner state cells; penalty and
  merge do not affect the corner study; (3) every withheld cell appears as "withheld (no base)" in the gate table
  (`tf46p2_gate_table.withheld_cells` -> `gate_withheld.parquet`; a transfer signal with no row at all appears as a
  `("*", signal, "*")` placeholder) and in the release notes; never silently missing.
- Coverage forecast, IDSSE only (7 matches; `D:\Development\_handoffs\tf46p2-r2-diag\coverage_forecast_idsse_only.json`;
  restart-type level, a cell state is at least this thin):

  | Restart type | IDSSE n / m | GS n / m (no longer a base) | Bar n >= 35 and m >= 5 | Extra IDSSE matches |
  |---|---|---|---|---|
  | corner, crossed | 53 / 7 | 460 / 64 | met | 0 |
  | corner, short | 4 / 3 | 112 / 51 | not met | 55 |
  | free kick, crossed | 48 / 7 | 209 / 62 | met | 0 |
  | free kick, short | 124 / 7 | 1,562 / 64 | met | 0 |
  | goal kick | 144 / 7 | 1,001 / 64 | met | 0 |
  | throw-in | 308 / 7 | 2,600 / 64 | met | 0 |
  | penalty | 3 / 3 | 64 / 23 | not met | 75 |
  | merge (direct free-kick shot) | 0 / 0 | 46 / 31 | not met | no IDSSE instance |

- Finding G (owner decision 2026-10-26: in this C2 amend, in full, including the retrains, the republished model cards
  and the research re-runs before the joint release): the library treats GS as fully observed
  (`_FULLY_OBSERVED_PROVIDERS`, `Z_PROVENANCE`, the GS adapter's null `visibility`) and as dead-ball observed (the
  only GS loader writes a constant `"alive"`). Its design waits for the DL
  (`D:\Development\_handoffs\2026-10-26-tf46-part2-cl42-g-design-slow-error-dl-handoff.md`); it is not built yet. R.2
  waits for G.

**Q3 — no player-position stage; a stated limit (§5.3).** IDSSE's null swap rate at 10 Hz is 0.028 per hour, under the
0.53 per hour target; the injection supplies the swaps (§27.5 applies on IDSSE). Stated limit: SkillCorner in-view
player positions carry broadcast error (homography, detection jitter, smoothing lag); the degradation does not add it
to IDSSE, and it cannot be identified from SkillCorner alone; the transfer results for player-position signals
(`taker`, `first_contact`, the attribution margin) are optimistic by this error. Report (owner): the in-view player
d2 per axis on detected triples, s = 1.4826 x MAD and the p90 of |d2|, SkillCorner against degraded IDSSE
(`kick_cells_d2_report`); caveat: d2 sees fast jitter only, not the broadcast offset, so the number is a lower bound on
the optimism and does not show that the limit is small. Never gated.

**Q4 — reference flips: the author default stands.** Period 2 is reflected against period 1 (Law 8); an extra-time
period fails closed (none in IDSSE).

**Q5 — B.** (a) The nominal source rate is kept, with a guard: a period whose median positive source step differs
from 1 / `frame_rate` by more than `grid_source_rate_tol_share` = 0.01 fails closed (`SourceRateError`; owner: never
replaced by "use the measured dt", which would hide a loader defect). (b) One grid per period for every object.
(c) The ball grid-hole share is reported per reference unit (`grid_ball_grid_missing` / `grid_ball_grid_points`;
SkillCorner: 0.136 % of within-period ball steps > 0.15 s; a larger IDSSE share returns to the DL); a frame without a
chain entry (a player row on a grid frame with no ball row) is a null-footprint frame (every object undetected),
counted (`rows_without_chain_entry`).

**Item 1 — the chain fixes (coverage 0.53 against 0.73).** (i) For the relative pose only, a NaN SkillCorner ball is
interpolated linearly between the nearest finite positions of the period (the ends carried), counted; a live
footprint never becomes a null frame in the store. (ii) A window takes the ball bin of its first frame with a finite
ball; a window without one is dropped and counted. (iii) The stitch reference is the last LIVE relative footprint of
the period; with none, the draw is uniform (seeded), counted (`uniform_draws`). Diagnostics (both sides): the player
visibility decomposition (`kick_cells_coverage`: null or missing view, in view among the known, detected among the in
view), the attack-direction lead per phase (`kick_cells_attack_lead`: the footprint centroid minus the ball along x,
signed by the possessing team's attack direction; the DL rules on an attack-direction cell key after these numbers),
the frame-weighted live-footprint area of the stored windows and of the SkillCorner frames.

**Item 2 — camera gaps no longer capped by `cam_seg_s`.**
- Restart phase: ONE SkillCorner restart window per reference restart, used whole, drawn from the restart cell of the
  ball at s0 (camera frame) among the `cam_stitch_k` windows whose dead-ball duration D = t_kick - s0 is nearest the
  reference D (seeded; no footprint stitching at the open -> restart seam). Frames at t >= t_kick - `cam_seg_s` align
  on kick-relative time (the core); earlier frames align on the time since s0; a reference D longer than the window's
  holds the edge of the s0-aligned part (a recorded hold); a shorter D lets the core win the overlap. `kick_bracket_hi`
  stands in for a withheld `t_kick` on both sides. A later restart's s0 cuts an earlier window. The setup segments are
  gone; `SEAM_KINDS` = open-open, open-core, core-open, core-core.
- Open play: continuation. A stored window whose last frame is null stores its source continuation up to the first
  live frame after the span (censored at the period end, recorded); the drawn segment runs through it and the next
  segment stitches to that live frame. Restart windows store their continuation past t_kick + 6 s. A window whose first
  frame is null starts inside a span another window holds: it is never drawn at an open seam (the period start is not
  a seam), counted. Store: `seq_cont_start`, `seq_cont_censored`, `first_times` (schema `tf46p2-camera-store-2`).
- No new constant (`cam_stitch_k` is reused). Diagnostics: the SkillCorner camera-gap split by phase
  (`kick_cells_gap_split`), the null-footprint ball-frame share on both sides, the hold durations.

**Item 3 — F on the second difference (supersedes §27.12 F).** The rest window is retired: owner decision 2026-10-26
removes `noise_rest_lo_s` / `noise_rest_hi_s` (ratified 2026-10-26 and retired the same day, a DL error) and their 4
sensitivity variants. Statistic, both sides, no anchor: triples of consecutive detected ball frames of one period on
the exact 10 Hz grid; per axis d2 = p(t+1) - 2 p(t) + p(t-1); s = 1.4826 x MAD(d2) per region of the middle frame
(author choice: the x and y d2 values pooled, one isotropic s per region); a region with fewer than 30 triples takes
the all-region value (recorded). The per-match statistic carries its d2 histogram (fixed 0.2 mm bins over +/-0.2 m plus
an underflow and an overflow bin; author choice) so matches pool exactly and the bootstrap draw is a matrix product.
Fit and inject: white Gaussian noise per axis and per region, sigma^2 = max(0, s_SC^2 - s_ref^2) / 6, s_ref measured on
the reference grid track before any injection. Score: s on the held-out matches, ratio within [0.85, 1.15]
(unchanged). Reported, never gated: the d2 lag-1 autocorrelation on both sides (-2/3 for white noise) and s_ref per
region (a large s_ref returns to the DL). Stated limit (§5.3): the slow part of the SkillCorner ball error
(calibration drift, smoothing lag) cannot be identified from SkillCorner alone, except at positions the law fixes,
where it is measured (report): the slow-error study (owner decision 2026-10-26; report only, never gated, a degraded-
IDSSE control; the corner arc within 1 m of the flag point, the penalty spot, the throw-in touchline y) waits for its
DL design. Speed jitter is matched within a bound: white noise gives sqrt(s^2 / 3) per frame step; an error close to a
random walk would give sqrt(s^2 / 2), up to 18 % more.

**Item 4 — E is kept; the overshoot is recorded.** Reported on both sides, never gated: the in-view run count, mean,
p50 and p90 length, the share of degraded in-view runs that start at a seam frame, and the in-view detected share
(SkillCorner 0.977). After the chain fixes, if the run lengths match and E still overshoots, it returns to the DL.

**Stage order:** B -> D (the chain: the restart draw, the continuation, the three fixes) -> E -> A -> F (white,
matched on d2) -> swaps -> R2. C is not built. The seed streams are unchanged.

**Tests (each fails under its mutant)** -- `tests/scripts/test_tf46p2_cl42_defects.py`: a player row on a frame without a
ball row is undetected; a live-footprint frame with a NaN SkillCorner ball stays live in the store; a window takes the
bin of its first finite ball; after a null-ended segment the draw covers the whole cell; a 30 s replay starting 2 s
before an open-play window end stays 30 s in the chain; a window starting inside a span is never drawn at a seam; a 25 s
setup maps the time since s0 from one window; white noise of known sigma is recovered within 5 % under 2 m/s^2
deceleration and 1 % kick impulses; the injected noise gives the fitted s (and the d2 lag-1 of -2/3); the GS tier-1
report admits nothing; every withheld transfer cell is reported with its reason.

### 27.14 The v3 rulings: SkillCorner missing rows, the ball smoother, the attack direction (added 2026-10-26)

Source: the DL ruling 2026-10-26 on `D:\Development\_handoffs\2026-10-26-tf46-part2-cl42-v3-results-dl-handoff.md`,
saved verbatim with the owner decisions at
`D:\Development\_handoffs\tf46p2-r2-diag\dl_ruling_cl42_v3_g_slow_error_2026-10-26.md`. The DL owns error 4: the d2
ruling (§27.13 item 3) assumed that the fast part of the SkillCorner ball error is white jitter; the measured d2 lag-1
(+0.66 / +0.64) shows smoothing, and the white injection made the degraded side worse. This section supersedes §27.13
item 3, the restart-window hold of §27.13 item 2 and the §27.13 stage order.

**The v3 measurement (cl42sub, IDSSE only; `tf46p2-r2-diag\cl42_v3_summary.txt`).** Fidelity 6 passed / 28 failed / 1
failed characterisation. Root cause of the coverage regression: SkillCorner shows a stoppage replay as MISSING ROWS --
a null row, no rows, a null row. In the calibration store 36.6 % of the restart-window time has no rows (39 % of the
restart windows hold a time gap; p50 17 s, p90 38 s, max 148 s; both edge rows null in 1,422 / 1,415 of 1,424 gaps);
open play 0.16 %. The restart rule maps by time, so a reference frame inside a gap took the null edge row: the degraded
null share was 0.27 against 0.13 by rows, while by time both sides are about 0.54 in the restart phase.

**Q1 -- SkillCorner missing rows (option (a)).** The SkillCorner bronze carries only frames with objects; the
degradation reproduces that production shape.
- The chain carries one of three states per reference frame: live, null or MISSING. A frame is missing when its
  mapped time lies inside a time gap of the drawn window (two window rows more than 1.5 frame intervals apart, the
  frame more than half an interval from both). One mapping rule for every span (`_map_rows`): open-play windows map
  by the time since the window start (no longer by index), the restart core by kick-relative time, the earlier
  restart frames by the time since s0, the continuations by their own time.
- `apply_camera_follow` drops EVERY object's row at a missing frame; the gap edges stay null rows; the remaining frame
  ids are kept (never renumbered: SkillCorner frame ids also have holes). E, A and F run on the remaining rows.
- A long stoppage: when the cell is empty or its nearest D is shorter than the reference D by more than `cam_seg_s`,
  the `cam_stitch_k` candidates nearest in D come from the whole restart phase (every bin; the same direction),
  counted (`restart_phase_fallbacks`). A middle that no window covers is missing, with one null row at each border
  (the SkillCorner pattern), counted with its duration (`uncovered_spans`); never an edge hold across the middle. The
  core-end hold (a window cut before t_kick + 6 s) stays a recorded hold.
- Addition 1: a simulated bracket time inside a dropped span has no frame: `simulated_bracket_ids` gives a frame id
  strictly between the bordering ids (absent from the frames), so the resolver counts `gi_frame_unmapped` and gives a
  NaN bracket and `kick_time_source` "none" (§6.1); never the nearest remaining frame. Counted (`brackets_in_gap`).
- Addition 2: `ball_steps_gt_0_15s_share` is a reproduced check; the share of restart TIME without rows is reported on
  both sides (`restart_time_without_rows_share`; SkillCorner v3: 0.366).
- Re-measure: the coverage decomposition, the kick-window miss rate, the restart near-ball visibility, the E runs.

**Q2 -- the ball smoother replaces F's noise (owner-approved 2026-10-26).**
- Precondition, verified 2026-10-26 (`tf46p2-r2-diag\lattice_probe.json`): every SkillCorner x / y (10 matches, ball
  and players, 10.6 M values) lies within 1e-5 m of a 0.01 m multiple. The last degradation step rounds every degraded
  x / y to 0.01 m, players and ball, after R2 (`round_positions`; float32 storage kept).
- The white-noise injection is removed (the s_ref problems go with it). F is a zero-phase Gaussian of width sigma_k
  frames on the ball x / y along each period's stream (`smooth_ball`), segment by segment (never across a time gap:
  SkillCorner has no data there), every ball row including the ones A made undetected. Position: after A.
- sigma_k per profile (`fit_sigma_k`): the grid sigma whose pooled d2 lag-1 (the mean of x and y) on the reference
  grid tracks -- smoothed, then rounded, as the output is -- is nearest the profile's SkillCorner calibration lag-1.
  The reference curve is the summed `smoothing_lag_sums` of every transfer-base unit on the 10 Hz grid (the profile
  stage emits it; a profile whose frame rate is not 10 Hz gets no sigma_k, recorded). Author choice: the curve pools
  the transfer-base units (one sigma_k per profile), and the fit target mirrors the output (smooth, then round).
- The §16.3 row: the |d2| KS ≤ 0.10 under the margin rule, per region (no pooling of a thin region: the margin
  carries it; a region with no triple on a side is not evaluable). The fit target (lag-1) and the gate (|d2|
  distribution) are different statistics, so the gate cannot pass by construction.
- The d2 histogram bins are centred on multiples of 0.2 mm (author choice): a 0.01 m lattice value lands on a bin
  centre, so +v and -v fold into one |d2| bin.
- Reports, never gated: s (1.4826 x MAD of d2) and the d2 lag-1 on both sides; owner additions: (1) the 10-90 %
  ball-speed rise time around a RESOLVED t_kick on both sides (the same resolver; SkillCorner held out;
  `kick_cells_kick_rise`); a degraded median more than 0.1 s from the SkillCorner median returns to the DL; (2) P(d2 =
  0) on both sides and the KS on the non-zero part (`kick_cells_d2_regions`; a still ball dominates the restart
  regions); (3) a sigma_k on the grid edge is reported and returns to the DL (`kick_cells_smoothing_model`).
- Stated limits (§5.3): the phase of SkillCorner's smoother (a causal lag) cannot be identified without truth; the
  stage is zero-phase, so a SkillCorner smoothing lag in kick times is not reproduced. The slow error belongs to the
  slow-error study (§27.16). No player stage (the player d2 p90 differs by about 1 cm); the rounding applies to players.

**Q3 -- the attack-direction cell key.** SkillCorner leads the play (median 6.5 m open, 2.5 m restart); the degraded
side showed 0.5 / -0.5 m.
- d = +1 when the possessing team attacks +x in the camera frame, else -1; d = the staged attack direction x sigma_p
  (the D reflection also reverses x). Possession: the team of the latest event at or before the frame (SkillCorner: the
  GI events of the run; the reference: the native events, carried forward); in the restart phase the RESTARTING team
  (`possession_direction`; the attack-lead report reads the same rule). Author choice: the SkillCorner possession team
  is the team of the latest GI event of the pipeline run (the GI possession team).
- Cell = (ball bin, phase, d); a window is stored under d at its first frame with a resolved d (none: dropped,
  counted `windows_no_direction`); a segment draws with the reference d at its start; pooling stays within the phase
  AND the direction. Windows and segments are never split at a possession change. Reported: the frames whose
  reference d differs from the drawn window's d (`direction_mismatch_frames` / `direction_frames`).
- Store schema `tf46p2-camera-store-3`; kick_cells schema `tf46p2-kick-cells-7`.

**Return condition (DL ruling v3):** the share of DETECTED ball frames outside the live footprint, both sides
(`kick_cells_ball_outside_footprint`). On SkillCorner it must be near 0; a degraded share that is not small against it
returns A to the DL.

**Stage order:** B -> D (the chain: three states; one whole restart window per restart, the long-stoppage fallback,
the continuation, every span mapped by time; the direction key; missing rows dropped) -> E -> A -> F (the smoother) ->
swaps -> R2 -> the 0.01 m rounding. The seed streams are unchanged (the F stream is no longer drawn).

**Tests (each fails under its mutant)** -- `tests/scripts/test_tf46p2_cl42_defects.py`: the gap definition splits
alike for the factors 1.1 and 1.9 (mutant: a factor below one frame step); the smoothed d2 lag-1 rises with sigma and
the fit takes the nearest (mutant: no smoothing); a sigma_k on the grid edge is flagged; the smoother never reaches
across a time gap (mutant: one joined signal); the rounding puts every position on the lattice and keeps the dtype; the
d2 gate is a KS on |d2| under the margin rule; the still share splits out of the d2 report; the smoother spreads the
kick speed rise and an unresolved kick is never used; detected balls outside the live footprint are counted; the
restart time without rows counts the gap time; a reference frame inside a window gap is missing and its rows dropped
(mutant: no gap rule); open play maps by time (mutant: index order); a long stoppage draws from the whole phase and an
uncovered middle is missing with null borders (mutant: no fallback); a bracket time inside a dropped span has no frame
(mutant: the nearest frame); a window stored with one direction is never drawn for the other (mutant: no key); the
lead diagnostic recovers a planted lead; the restart-phase direction is the restarting team.

### 27.15 Finding G: GS is detection-aware, estimated in z, not dead-ball observed (added 2026-10-26; DL design)

Owner decision 2026-10-26: in this C2 amend, in full (the adapter / loader flag, the detection-aware set, the z
provenance, every consumer, the retrains, the republished model cards and the research re-runs, before the joint
release). DL design (G1-G8, `dl_ruling_cl42_v3_g_slow_error_2026-10-26.md`):
- G1: the GS adapter accepts an optional input column `visibility_native` (closed vocabulary: `VISIBLE` -> True,
  `ESTIMATED` -> False; any other value raises) and writes the boolean `visibility`; without the column `visibility`
  is None and the detection consumers fail loud (as the SkillCorner adapter).
- G2: the loader reads the raw arrays with their own flags, never the `*Smoothed` copies; `confidence` is not
  carried (its distribution is reported in the staging census).
- G3: GS `z_source = "estimated"` for the whole match (static, ADR-125), so `is_aerial` is NaN on GS. Hyrum's law:
  `is_aerial` / `aerial_source` on GS change from booleans to NaN / `estimated`; G7 classifies every GS reader.
- G4: GS leaves `_DEAD_BALL_OBSERVED_PROVIDERS`; the constant `"alive"` stays (the schema has no null).
- G5: only the GS units are re-staged, with a per-unit census incl. the 27.5 m keeper clamp
  (`validate_gk_position_clamp`) split by VISIBLE / ESTIMATED.
- G6: no consumer that reads detection is exempt on GS; position-only consumers are unchanged; "detection-aware" never
  implies "has a camera footprint" (a footprint consumer tests for the footprint itself), recorded per consumer.
- G7: a classification table of every model and research artifact (detection mask -> retrain / re-run; GS z or
  `is_aerial` -> re-run; the GS dead-ball state -> re-run; GS positions only -> unchanged numbers, the text corrected;
  GS events only -> unchanged) goes to the DL before any run; each model publish needs the owner's approval (ADR-088).
- G8: dated in-place amendments of ADR-069 and ADR-125; the `Z_PROVENANCE` comment and `docs/context/` corrected.

G6 sweep (2026-10-26; every production call site of the taxonomy; "footprint?" = does it infer a camera footprint
from detection-awareness):

| Consumer | Reads | On GS after G | Footprint? |
|---|---|---|---|
| `_provider_visibility` keeper wrappers (`resolve_detected_keeper_xy`, `keeper_detected`) and their callers: `utils._resolve_action_frame_context` (action context, pre-shot GK), `features` GK influence / closing time / `gk_influence_xfns`, `_gk_geometry` goal-kick origin, `restdefense` structure / danger, `gk_decision._reconstruct`, `gkdv._engine._apply_domain` | the keeper's detection flag | gated: an ESTIMATED keeper is honest-NaN; frames without the flag fail loud | no |
| `detected_mask` / `keeper_detection_mask`: `_ghost_gk`, `coordination._signals` (`det_mode`), `derive_coordination_params`, `train_ghost_gk` | the per-row detection flag | gated | no |
| `_ball_contact` attribution (`detect_ball_contacts`, provider declared) | the candidates' detection flag | gated; `footprint_active` reads the data profile's gate entry, never membership | no (tests the gate entry) |
| `_kick_resolver._detected_ball_speed` | the ball flag when it is non-null (SkillCorner semantics) | ESTIMATED balls no longer count in the launch rule | no |
| `coordination._windows` | `dead_ball_observed` | GS stoppage split no longer reads `ball_state` | no |
| `_camera_footprint` (`footprint_inside_share`, `keeper_undetected_split`), `_restart_frame._FrameIndex`, `_penalty_detector` | SkillCorner flag semantics on frames that carry a footprint / `in_camera_view` | footprint inputs are explicit arguments; GS has none | tests the footprint itself |
| pre-flights: `materialize_tc3_frames._guard_provider_frames`, `_corpus_visibility`, `_coordination_corpus` visibility pass, `train_ghost_gk` shard choice | set membership | GS shards are now checked (fail loud without the flag) | no |
| `build_corner_routines`, `validate_corners` (`classified_provider`) | the classification | GS stays classified | no |

G7 ruling (DL, 2026-10-26; `tf46p2-r2-diag\dl_ruling_g7_2026-10-26.md`). The principle: a GS position used as TRUTH (a
training target, a validation reference, a measured quantity) must be a detected row; a GS position used as an INPUT
(a feature or context, as production data looks) stays as it is (class 4). Ruled rows: ghost-GK (all three variants)
RETRAIN; xcross / xshot class 4 with a recorded consumer defect (below); the retention weights class 5; the receiver
weights unchanged; `ball_contact_detection` class 1 + 2 RE-RUN; tf58 class 1 + 3 RE-RUN; tf19 instrument / signoff, the
GKDV arm values, tf60 layer 3, tf19 pr2 / pr3b RE-RUN after the retrain; the Part 2 resolvers in-cycle. Open: the
ghost-outfield option (owner, after the author's target-row numbers), the goalmouth keeper reference (the ESTIMATED
share first), and the rows that need a transitive trace. Each retrain and each publish needs the owner's approval
(ADR-088).

G7 trace ruling and owner decisions (2026-10-26; `tf46p2-r2-diag\dl_ruling_g7_trace_2026-10-26.md`):
- The principle is refined: INPUT -> class 4 holds only when the estimate cannot carry the outcome. A class 4 row whose
  claim pairs GS or SkillCorner positions at t with an outcome after t also reports the claim statistic on the
  all-detected stratum and the rest (a look-ahead check).
- tf56 positioning is criterion 1 (the defender positions are the measured side).
- The goalmouth validator re-runs with the keeper mask; its z claim is corrected (GS z is not measured).
- Owner decisions:
  - C: accept R-C1 to R-C3 (measure the event-derived stoppage intervals; route the `ball_state` readers through
    `resolve_stoppages`; retrain xcross / xshot only on a material change).
  - A: accept the ghost-outfield T0 / T1 / T2 held-out study.
  - ADR-038 §5: accept a dated amendment (GKDV measurement registers to the detected keeper rows on any provider,
    plus IDSSE).
  - TF-48: add a z-source provenance column to the shot-goalmouth outputs.

G6 sweep addition: the `ball_state` readers (DL ruling G7 item C). A `ball_state` reader must test
`dead_ball_observed`, as a detection reader tests detection. Outside the set (GS, SkillCorner, Metrica, the kloppy
gateway) the column is the constant "alive" (G4), so a reader that filters on it keeps the dead-ball frames. Status
(2026-10-26, static trace `tf46p2-r2-diag\g7_call_trace.py`); each "no" is a recorded consumer defect, not fixed here:
an event-derived dead-ball mask is new scope (G4), the owner decides.

| Reader | Use of `ball_state` | Tests the set? |
|---|---|---|
| `coordination._windows.resolve_stoppages` / `_ball_state_intervals` | the stoppage split | yes (else the event evidence) |
| `tracking._xshot_occurrence.prepare_xshot_training_data` | keeps `alive` frames for the xS training rows | no |
| `tracking._xcross_attempt` (the possession scan) | drops a possession whose ball row is `dead` | no |
| `causal.opportunities._frame_domain_state` | the opportunity domain state | no |
| `gkdv._engine._apply_domain` | drops frames that are not `alive` | no |
| `positioning._compute._classify_and_score` | the frame's ball state in the classification | no |
| `restdefense._counterfactual`, `restdefense._windows.select_rest_defense_samples` | drop dead-ball samples | no |
| `tracking._elastic_sync` | selects `alive` frames | no |
| `tracking._off_ball_runs`, `tracking._run_values` | drop dead-ball teammates / actions | no |
| the adapters `gradientsports`, `kloppy`, `sportec` | the per-period ball-out time (a producer-side summary) | n/a: GS and the kloppy gateway report 0 s |

### 27.16 The slow-error study (report only; added 2026-10-26; DL design)

Owner decision 2026-10-26 (report only, never gated, a degraded-IDSSE control). Events: corners, penalties and throw-ins
with a RESOLVED t_kick (never `kick_bracket_hi`, never `none`); SkillCorner: every match of the profile (nothing is
fitted); control: the degraded IDSSE transfer run with the same resolver. Window: the detected ball frames in [t_kick -
1.0 s, t_kick - 0.3 s], at least 3 frames (report parameters; sensitivity [-2.0, -0.3] s and [-0.6, -0.2] s). Per
event the window median per axis; the offset e to the fixed point: corner -- the nearest corner flag point, along the
goal line and along the touchline in the camera frame (Law 17: within 1 m); penalty -- the nearest penalty spot (11,
34) or (94, 34); throw-in -- e_y, the signed distance to the nearer touchline, positive outward (never x). Never
excluded by |e|; |e| > 5 m flagged and counted; excluded only for no resolved kick or fewer than 3 frames (counted
per reason). Split near / far camera side. Report per (profile, type, near / far, side): n, p50 / p90 of |e|, the
mean signed e per axis, the excess SkillCorner minus control for p50 and p90, sqrt(max(0, RMS_SC^2 - RMS_ctrl^2)),
each with a match-clustered bootstrap CI (B = 2000); aggregates only. The penalty control (IDSSE has 3) reports its n
and wide CI; never another type's control.

Implementation (2026-10-26; `scripts/_tf46p2_slow_error.py`): `slow_error_events` per match and side (the run's own
resolver output; the restart type from the SPADL `type_id`), `slow_error_table` per (profile, window, type, camera
side, side) as `kick_cells_slow_error`. Windows: `main` [-1.0, -0.3] s, `wide` [-2.0, -0.3] s, `late` [-0.6, -0.2] s.
Author choices: (1) the corner and penalty offsets are taken in the camera frame (a period with sigma -1 is
point-reflected), e1 along the touchline (camera-frame x) and e2 along the goal line (camera-frame y, + = away from the
camera), so a mean signed e2 shows a pull toward or away from the camera; the throw-in e2 is the signed distance to the
nearer touchline (+ = outward) and e1 is NaN; (2) near / far = the window median camera-frame y below / above half the
pitch width; (3) the RMS is of |e| (the 2-D offset; the throw-in |e2|); (4) the bootstrap resamples each side's matches
independently (B = 2000) for each side's p50 / p90 and the three excess values; (5) the exclusion counts are reported
on the `all` camera-side row. Tests (`tests/scripts/test_tf46p2_slow_error.py`): a planted 1.5 m far-side offset is
recovered and the near side shows none (mutant: no side split); equal anchor jitter on both sides gives zero excess
(mutant: no control); a `kick_bracket_hi` anchor is refused (mutant: accepted); a throw-in uses y only (mutant: a 2-D
distance); a flipped period is reflected into the camera frame and a thin window is counted.

### 27.17 The v4 rulings: the still ball, the outside-footprint condition, the kick rise in frames (added 2026-10-26)

DL ruling 2026-10-26 on the v4 measurement (verbatim with the owner record:
`D:\Development\_handoffs\tf46p2-r2-diag\dl_ruling_cl42_v4_2026-10-26.md`). v4 found that the |d2| gate fails in every
region only because of the still ball: SkillCorner P(d2 = 0) is 0.08-0.10 in every region; the degraded side was 0.19
(rest) to 0.62 (corner flag). The DL owned an error: the v3 Q2 ruling said "the fast jitter is below the lattice", which
was wrong. This section extends §27.14 (Q2); it supersedes no row of §16.3.

**Q1-v4 -- white jitter before the smoother.**
- Precondition (rule 0), measured before the build (`still_ball_probe.py`, `still_ball_probe.json`; 40 SkillCorner
  calibration matches of the cl42sub profile). Still window: [t_kick - 1.0 s, t_kick - 0.3 s] before a resolved
  placed-ball kick (corner, goal kick, free kick, penalty, kick-off; never a throw-in: the ball is in the hands), the
  slow-error study window. Result: d2 lag-1 +0.47 (95 % match-clustered interval 0.39 to 0.55; near +0.41, far +0.53),
  P(d2 = 0) 0.100 (0.092 to 0.109), position spread (1.4826 x MAD about the window median) 0.19 m (0.17 to 0.22).
  731 of 1,173 windows were used; 442 had fewer than 3 detected ball frames (excluded, counted). The windows are
  corners, free kicks and goal kicks only: the resolver has no kick-off rows, and no penalty gave a window.
- Owner record 2026-10-26: "Record: jitter before the smoother, as measured" with four additions (below). The lag-1 is
  positive and its interval excludes 0, so the jitter goes BEFORE the smoother (a negative lag-1 would put it after
  the smoother, before the rounding).
- The stage (`jitter_ball`): N(0, sigma_n^2) added to the ball x and y of every ball row, detected and undetected, then
  the smoother (`smooth_ball`), then the rest of the chain; the 0.01 m rounding stays last, after R2. The unit normals
  are drawn ONCE from the seed stream [seed, 5] (F's stream) and scaled by sigma_n (common random numbers). sigma_n 0
  or NaN draws nothing (the v4 output exactly).
- The fit (`smoothing_fit_sums`, `fit_jitter_smoothing`; per profile, on the calibration matches). The reference grid
  tracks (10 Hz) are jittered, smoothed and rounded as the output is, per grid pair: sigma_n 0, 0.01, ..., 1.50 m and
  sigma_k 0, 0.25, ..., 3.0 frames (report parameters, not ratified). For each sigma_n, sigma_k is the grid value whose
  pooled reference lag-1 is nearest the SkillCorner lag-1 (as §27.14; ties: the smaller). sigma_n is the grid value
  whose still-window P(d2 = 0) at its sigma_k is nearest the SkillCorner calibration value (ties: the smaller). The
  degraded anchor is the reference kick time K (`kick_truth`; NaN = no window): the fit cannot rerun the resolver per
  grid point, and the 0.3 s margin covers the resolver error on SkillCorner. Owner addition 2: the reference per-type
  P(d2 = 0) is weighted by the SkillCorner triple share of each type. Author choices: a type with SkillCorner triples
  but no reference triples is left out and the other weights are renormalised (`bases_missing`, reported); no
  SkillCorner still triples gives sigma_n and sigma_k NaN (no jitter, no smoothing; `fit_status`). The fit seed per
  reference unit is `unit_seed(match_id, "jitter_fit")`.
- The gate (§16.3) is unchanged. The fit target (the still zero share) overlaps the gate's zero mass, so the report also
  gives the |d2| KS without the still-window triples on both sides (`ball_d2_distribution_nonstill`, the run's own
  resolver; `kick_cells_d2_regions.ks_full_nonstill`).
- Reports, never gated: sigma_n, sigma_k, the fit status and weights (`kick_cells_smoothing_model`); the 2-D fit curve
  (`kick_cells_smoothing_fit_curve`); the still-window table (`kick_cells_still_windows`) per side -- SkillCorner
  calibration (the fit target), SkillCorner held out (the check), degraded -- and per group (all, near, far, each
  restart type): the windows, the excluded share, P(d2 = 0), the lag-1 and the spread with 95 % match-clustered
  intervals, the detrended spread and the ball-speed p50 / p90. Near / far: the window's median camera-frame ball y
  below / above half the pitch width (the camera is at camera-frame low y, DL ruling D). Each side uses its own
  resolver for the windows (SkillCorner: the reference run; degraded: the transfer run).
- Owner additions (2026-10-26; none is a new constant or gate): (1) symmetry -- the degraded still windows keep only
  the restart types that have SkillCorner windows, with the same exclusion (fewer than 3 detected ball frames); the
  excluded share is reported on both sides; (2) the type weighting above, and the statistics per type on both sides;
  (3) the spread about a per-window straight line in time (detrended) and the ball-speed p50 / p90 on both sides, so a
  rule-5 return shows slow drift against a wrong kernel; (4) rule 5 stays as ruled; nothing is tuned outside the
  ruled grids; the v4 coverage decomposition runs in the same v5 DGX run.
- Return to the DL when: the precondition interval contains 0; sigma_n or sigma_k is on a grid edge; the degraded /
  SkillCorner held-out still spread ratio lies outside [0.67, 1.5] (the existing rate-ratio band, `RATE_RATIO_BAND`,
  not a new constant). Author choice: the spread check uses the held-out SkillCorner side, as the gate does.

**Outside the footprint (the DL judges v4 small; no A mask).** The return condition is now a number: a degraded
share of detected balls outside the live footprint above 0.5 % (`OUTSIDE_FOOTPRINT_RETURN_SHARE`, about 6 x
SkillCorner's 0.08 %) returns A to the DL. Also reported on both sides: the resolved kicks with a detected ball
outside the footprint in [t_kick - 1.0 s, t_kick + 1.0 s] (`kicks_with_ball_outside`), and the detected ball frames
with a degenerate footprint quad (not a live frame; `ball_detected_degenerate`).

**The kick rise time is compared in whole frames.** A degraded median returns to the DL when round(|degraded -
SkillCorner| / frame interval) > 1 (`rise_returns_to_dl`), so a float residue never decides.

**The degraded gap split.** The camera / live gap split is reported on the degraded side too (`kick_cells_gap_split`
gains a `side` column).

**Schema:** kick_cells `tf46p2-kick-cells-8` (the fit-curve rows, the still-window records, the degraded gap split).

**Stage order:** B -> D -> E -> A -> F (the jitter, then the smoother) -> swaps -> R2 -> the 0.01 m rounding.

**Tests (each fails under its mutant)** -- `tests/scripts/test_tf46p2_cl42_defects.py` and
`tests/scripts/test_tf46p2_cl42.py`: sigma_n 0 reproduces the v4 output (mutant: the noise is drawn); on a constant
ball the jitter before the smoother gives a positive still lag-1 (mutant: after the smoother, the sign flips); the
still P(d2 = 0) does not rise with sigma_n (common random numbers); the still-window selection excludes throw-ins and
applies the 0.3 s margin; the fit kernel equals the stage path; the fit weights each type by the SkillCorner triple
share (mutant: the unweighted pool picks another sigma_n) and flags the grid edge; the detrended spread removes a slow
drift; the still table keeps the SkillCorner types and flags rules 0 and 5; the still triples leave the non-still d2
report; the histogram KS equals the exact two-sample KS on lattice fixtures with float32 residuals; degenerate quads
are counted; a kick counts once when a ball is outside the footprint within 1 s; the outside table flags only a
degraded share above 0.5 %; the kick-rise return counts whole frames (0.4 vs 0.3: no; 0.5 vs 0.3: yes;
0.30000000000000004 included); the degraded gap split is counted and summed.

### 27.18 The v5 rulings: the still set, the lag-1 estimator, the corner census (added 2026-10-26)

DL ruling 2026-10-26 on the v5 measurement (verbatim: `D:\Development\_handoffs\tf46p2-r2-diag\dl_ruling_cl42_v5_2026-10-26.md`).
DL error 6 (owned): the throw-in y is not a clean fixed point; the ball is held about 2 m up, so a monocular system
projects it away from the camera.

- **Q1 (built):**
  - The still set is corners and free kicks, plus penalties and kick-offs when they give windows. Goal kicks leave it
    on both sides (`STILL_BALL_BASES`). The rule-0 check reruns on the new set as a report (the calibration row of
    `kick_cells_still_windows`).
  - No per-type precondition (Q3).
  - The degraded anchor stays K; corners join after the Q2 ruling.
  - The fit uses every IDSSE unit of the run (all 7 in the next measurement).
  - The d2 lag-1 is the bounded r = sum ab / sqrt(sum a^2 sum b^2) over the same pair set in every use
    (`d2_lag1` returns the three sums; `lag1_corr`; `pooled_lag1`; the profile statistic carries sum b^2 in its
    `samples`).
  - Rule 5 has two ratios, both in [0.67, 1.5]: the still spread and the still speed p50 (`speed_ratio`). The
    detrended spread stays a report.
  - The still-window variogram gamma(l) = 1/2 mean (p(t + l) - p(t))^2, x and y pooled, l = 1..7 frames, per side
    and group with match-clustered intervals (`kick_cells_still_variogram`).
  - If rule 5 still fires, the result returns to the DL with the variogram; no slow-component stage is built first.
- **Q2 (census returned, no fix):** IDSSE does not track the ball's position while the ball is dead. Before every
  corner, the ball is frozen at its last live position, and `ball_state` is dead. So the §6.4 corner location cannot
  pass. Reference K: corners 1 / 57, penalties 0 / 3, goal kicks 43 / 144, the same at 6741710. Not a library defect.
  The census and the cause: `D:\Development\_handoffs\2026-10-26-tf46-part2-cl42-v5-q2-census-dl-handoff.md`. R.2 waits
  for the ruling.
- **Q4:** the |d2| gate is unchanged. Re-measure on 7 units after Q1 and Q2.
- **Q5 (to do):** two checks before any change. (1) A round-trip test: placing the drawn window on its own
  SkillCorner ball must give the source in-view share exactly (also as a fixture test). (2) A configuration mismatch:
  the in-view shares by delta-c band (< 5, 5–15, > 15 m) per phase.
- **Q6 (built):**
  - The throw-in rows carry a `reading`: "ground-projection bias + slow error" when the SkillCorner signed e2 is
    negative on the near touchline and positive on the far one. The corner rows read "no reading until the control
    has corners".
  - `flagged_kick_sources` counts the kick-time sources of the rows above 5 m.
  - The median and p90 excess with their intervals were already reported (`excess_p50_m`, `excess_p90_m`).
- **Tests:**
  - the lag-1 correlation is bounded on adversarial fixtures (the old ratio reads 2.0);
  - goal kicks left the still set;
  - the variogram rises for a drift and is 0 for a still ball;
  - the still table flags both rule-5 ratios;
  - the throw-in reading follows the e2 sign by camera side;
  - the flagged rows report their kick-time sources.

### 27.19 R-C1 rules, R-C2 (the stoppage column) and the v5 Q5 draw (added 2026-10-26)

DL ruling 2026-10-26 on R-C1, the v5 Q5 checks and the look-ahead list (verbatim:
`D:\Development\_handoffs\tf46p2-r2-diag\dl_ruling_g7_rc1_q5_2026-10-26.md`). Owner decisions 2026-10-26 on R-C2
(verbatim: `D:\Development\_handoffs\tf46p2-r2-diag\owner_decisions_rc2_2026-10-26.md`).

**The event-derived stoppage spans (one core, `silly_kicks.tracking._stoppages.event_stoppage_intervals`).**
- Spans: every restart closes the span that the previous action opened; every goal opens a span that the next action
  closes.
- Start (DL item 1). The span is half-open at the previous action: that action's instant is live. With frames, the
  start is the first ball frame of a run of at least `MIN_OUT_RUN_FRAMES = 2` frames with the ball centre beyond a
  line by at least `spadlconfig.ball_radius` (0.11 m) in real metres (`resolve_pitch_dims`; Law 9: wholly crossed). A
  goal span reads the goal lines only. Else the start stays at the previous action. `start_source`: `ball_out` /
  `after_previous_action`; `start_ball_detected` is the detection flag of the ball frame that set a `ball_out` start.
- End (DL item 2). A native restart event carries its time (IDSSE; Gradient Sports confirmed: its restarts are native
  PFF events, timed by the event clock): `end_source = "event"`. A SkillCorner restart is a derived row with an
  ordering-only placeholder time, so its span ends at the Part 2 resolver's `t_kick` when it resolves
  (`resolved_kick`; rule version 2: only when span start < `t_kick` <= the placeholder), else at the placeholder as
  an UNKNOWN span (`placeholder`; `unknown` in version 1). A SkillCorner restart row that is not
  `action_provenance == "derived"` raises (an older converter: reconvert).
- The frame state (`frame_stoppage_state`): `dead` inside a span with a known end, `unknown` inside an unknown span only
  (a known dead span wins an overlap), `live` elsewhere. Every reader treats dead and unknown as not live.
- `STOPPAGE_RULE_VERSION = 2` (version 1: the rules above, 2026-10-26; version 2: the resolved-kick rule, 2026-10-27,
  below). Bump it whenever a span rule changes; a column of another version is refused.

**The stoppage column (owner decision R-C2, the reader API).** `derive_stoppage_state(frames, *, actions=None,
pitch_length=None, pitch_width=None)` returns a copy with `stoppage_state` (`live` / `dead` / `unknown`) and
`stoppage_source` (`ball_state` / `events_v2` / `event_instant`) on every row (the rows of `ball_state`); `ball_state` is
never written.
- IDSSE / Sportec: from `ball_state` (alive -> live, dead -> dead, any other value -> unknown).
- Other providers: from the core over `actions` (required). The provider is the game's `source_provider` (one per game,
  else `ValueError`; an unclassified provider raises). Full-rate frames only: a period whose ball rows step by more
  than `THIN_FRAME_FACTOR = 1.5` frame intervals raises (build the column before any subsampling or filtering).
- `out.attrs["stoppage_report"]`: the rule version, per game the provider, the source and the span counts by
  `start_source` / `end_source`, the rejected resolved kicks by restart type (`resolved_kick_rejected`), and the
  spans (plain tuples).
- The reader gate `has_stoppage_state(frames, *, where)`: True with a current column; False when the column is absent
  and every provider observes `ball_state` (the reader keeps its `ball_state` logic, unchanged); else `ValueError`
  (no column on SkillCorner, Gradient Sports or Metrica; another rule version; no `source_provider`; an unclassified
  provider). `stoppage_not_live(frames)` reads the column.
- The 10 readers route through the gate: `prepare_xshot_training_data`, `prepare_xcross_training_data`,
  `causal.build_opportunities` (`_frame_domain_state`), `gkdv._engine._apply_domain`, `compute_positioning_gap`
  (`_classify_and_score`; the gate runs before the down-sample), `restdefense._counterfactual._apply_domain`,
  `select_rest_defense_samples`, `align_events_to_frames` (`_inplay_segments`; the gate runs after the freeze-frame
  early return), the TF-4 kernel `_off_ball_runs_kernel` and the shared candidacy `_prepare_run_candidates`, and
  `detect_off_ball_runs`.
- The training preparers never train on an UNKNOWN frame; they count the would-be rows and their positive labels in
  `features.attrs["n_stoppage_unknown_rows"]` / `["n_stoppage_unknown_labels"]` (DL item 2).
- The TF-58 coordination split adopts the same core only after the R-C1 rules are final (owner decision, conditions
  1-9 in the decision file); it is not changed yet.

**Author choices (surfaced):**
1. Decided 2026-10-27 (owner): keep `derive_stoppage_state`. The function is named `derive_stoppage_state`, not
   `add_stoppage_state` (the owner's example name): it enriches
   FRAMES, as `derive_velocities` / `derive_goalkeepers` do; the `add_*` registries (purity, SB360 verdicts, mirror,
   glossary) are built for action enrichers. A rename is cheap if the owner prefers `add_`.
2. `THIN_FRAME_FACTOR = 1.5` reuses the CL-42 time-gap factor (`GAP_FACTOR`); it is a refusal tolerance, not a
   measured constant.
3. A missing `stoppage_state` value (for example a frame without a ball row after a merge) reads as not live.

**Measurement (`rc1_v2_probe.py` / `rc1_v2.json`, `rc1_v2_noint_probe.py` / `rc1_v2_noint.json`).** The shipped xS
stage `xshot115` was built by the converter at 4095251 (the shipped models' `training_commit`); its SkillCorner
restarts are native rows (the old converter). The spans are rebuilt from the current converter's actions for the same
115 matches (the R.2 staging, converted at 6741710) and applied to the same training frames: every xS ball frame has
the same time and position in both stagings (4,801,391 SkillCorner and 1,002,644 IDSSE ball frames; 0 differences),
and all 2,596 old SkillCorner shot times are in the new actions. Two arms for the SkillCorner end: `shipped` (the gate
table is empty until R.2: every SkillCorner restart is unknown) and `pass_all` (a diagnostic: the resolver's kick times
with the pass-all gate table).

**Fixtures (owner decision, R-C2 fixtures).** Builders with an unclassified label were re-labelled to `sportec` (an
optical feed with an observed `ball_state`, no detection flag); builders without `source_provider` got `sportec`. A
scratch recorder wrapped every provider-keyed lookup and compared each test under the old label (gate off) and the new
label: 303 tests, 0 changed lookups other than the dead-ball gate. Sparse Gradient Sports / SkillCorner fixtures use
the test-only helper `tests/_stoppage_fixtures.mark_all_live`; full-rate provider fixtures call
`derive_stoppage_state`. The G sweep missed one fixture: `scripts/_toy_corpora.ghost_gk_frames` (Gradient Sports, no
`visibility`), so 49 GKDV tests failed with `KeyError: 'visibility'`; it now carries `visibility = True` (the scene it
always modelled).

**Returned to the owner (decided 2026-10-26: see the freeze-frame and leg-B paragraphs below):** four readers are reached in production with SB360 freeze frames
(`source_provider = "snapshot"`, unclassified): the TF-4 kernel (`add_off_ball_runs`, `add_off_ball_context`),
`detect_off_ball_runs` (`add_off_ball_run_values`), the GKDV domain (`build_ghost_frames`, `delta_das`,
`delta_threat_suppression`) and the rest-defence domain; they now raise there. The SB360 leg-B fixture (`synthetic`)
is not re-labelled: a classified label changes its visibility classification. The production drivers that pass
SkillCorner / Gradient Sports / Metrica frames to a reader must build the column.

**SB360 freeze frames (owner decision 2026-10-26, "yes" to the gold-standard proposal).** A freeze frame sits at an
action instant, and under the span rules no action instant lies inside a span, so `derive_stoppage_state` gives
`source_provider = "snapshot"` frames `stoppage_state = "live"` with `stoppage_source = "event_instant"` (no actions
needed). The readers stay fail closed: freeze frames without the column raise. A property test proves the invariant
over random action sequences. The SB360 leg-B fixture is labelled `sportec` (the feed it models); the lookup check and
the full slow SB360 battery show no changed lookup and no changed verdict, so nothing was regenerated.

**The load seams (owner decision 2026-10-26, conditions 1-8).** One shared function, `scripts/_stoppage_seam.py`
`stoppage_seam(frames, actions, *, cap=None)`: it drops `t_kick` from the actions (a build with resolved kicks needs
its own `stoppage_source` value, an owner decision), builds the column on the full frames, then applies the loader's
cap. The seams: the pining builders (`_build_idsse`, `_build_gradientsports`, `build_skillcorner_frames` with
`actions`, `build_statsbomb_match`; each now converts the FULL feed, then the seam, then the former `tracking_limit`
selection, then `_preprocess`), `_loader_databricks.load_matches` (the seam, then `head`), and the staged-unit reader
`scripts/_driver_sources.read_staged_unit` (frames + actions under the read gate, an optional raw-frame `validate`
hook, then the seam), used by the xS / xCross trainers' per-unit extract, `hf_jobs/headtohead`,
`measure_box_constant_delta` and `measure_f1b_feature_delta`. `run_tracking_features`, `enrich_invariant` and the
readers never build the column. Consequences: a capped pining load now converts the whole feed (orientation and the S1
gate see the full match); the xS stage `hf-stage-xshot115` (old-converter SkillCorner actions) is refused until it is
restaged at the current sha (owner decision: restage).

**Saved products (condition 6).** Every reader-built shard / stage product records `stoppage_token()`
(`STOPPAGE_RULE_VERSION`, `STOPPAGE_SOURCES`, the seam rule) in its `for_each` `token_inputs` or ADR-128 `declare`, so a
product built before the column lands in another generation and is never resumed or read: the xS / xCross extract
shards, the GKDV / TF-19 / TF-60 / xS-probe / sign-off / layer-2 / causal / tf56 / openGoal / box-constant / f1b shards
and the calibration stage stores. The committed research artifacts of the `declare_inputs` drivers record it as
`stoppage` in their input contract, so the staleness detector flags them. The trained models and the committed
artifacts: see the next paragraphs (owner decisions 2026-10-26).

**Toy and fixture changes (reported, no expected value changed):** the toy xS / xCross games declare the rate they are
really sampled at (`_toy_frame_rate`, 1 / the median step) and write SPADL-complete actions (`action_id`; the toy shots
`fail`); `_game_frames` declares `frame_rate = 1 / dt`; the GS loader fixture writes 29.97 Hz clocks; two f1b / box
fixtures declare 1 Hz. The thinness refusal is not relaxed.

**Tests:** `tests/tracking/test_stoppages.py` (the core: the half-open start, the two-frame run beyond the ball
radius, the real-metre margin, the goal lines only, the SkillCorner end sources, the old-converter raise, the overlap
precedence, the extrapolated start flag; the column: observed providers, equality with the core, thin frames, mixed
providers; the gate), `tests/tracking/test_stoppage_readers.py` (each reader raises for SkillCorner without the column,
an unclassified provider, no `source_provider` and an older rule version; mutant: the gate fails open),
`tests/tracking/test_stoppage_modules.py` (GKDV, rest defence and off-ball runs drop a dead and an unknown span built by
`derive_stoppage_state`; mutant: the module ignores the column).

**Trained models (owner decision 2026-10-26: refuse, retrain first; `legacy_override` allowed).** A reader-built model
records the stoppage provenance of its training frames in `metadata.json`: `stoppage_rule_version` and
`stoppage_sources` (`silly_kicks.tracking.stoppage_provenance`; `combine_stoppage_provenance` over matches).
`XShotOccurrenceModel` / `XCrossAttemptModel.save()` refuses a model without a current provenance (`RuntimeError`);
`load()` refuses an artifact without one (`IntegrityError`, `verify_stoppage_provenance`). It is the LAST guard, after
SHA, pitch, chirality and the feature contract, so a tampered artifact still fails on its own guard.
`legacy_override=True` loads it with a `StoppageProvenanceWarning`, for audit use only (the R-C3 head-to-head);
`from_variant` and `from_hub` never pass it. The trainers record the provenance per match (the extract side column
`_stoppage`, `stoppage_side_value`), combine it per provider in the study config (`stoppage_by_provider`) and over the
shipped rows at fit (`shipped_stoppage`); `metrics.json` records it as `stoppage_provenance`. Consequence: the shipped
bundles (`_xshot_weights`, `_xcross_weights`, default and position_only) and the HF `*-v1` repos refuse to load, so
`add_xshot_occurrence` / `add_xcross_attempt` / `compute_*` with the default model, and every caller of them, raise
until the R-C3 retrain ships in this release. The retrain covers every xS / xCross variant: the bundled default and
position_only, the Hub-only `sc_extended`, and the HF mirrors (each publish owner-approved, ADR-088).

**Interim tests (owner decision 2026-10-26).** Until the retrain commit, a test that pins a shipped bundle asserts its
refusal instead of its body (`tests/_stoppage_fixtures.bundle_pinned`, `assert_bundle_refused`); a property gate
(purity, mirror, NaN safety, liveness, duplicate action ids, goal-map consumers, SB360 axis locks, glossary, geometry
constants, the velocity guards, the Hub score functions) loads the shipped weights under `legacy_override=True`
(`legacy_bundles`, the fixture `legacy_xs_xc_bundles`). All three FAIL once a bundle loads without the override, so the
retrain commit removes them and the tests run their bodies again. A toy model fitted on a hand-built matrix states a
toy provenance (`with_toy_stoppage_provenance`). Tests: `tests/tracking/test_stoppage_models.py` (save refuses; load
refuses an absent, half, stale-version, unknown-source or empty provenance; the override warns and never back-fills;
the structural guard fires first; the frames helper; combine refuses a stale part).

**`resample_frames` (owner decision 2026-10-26: carry the ball row's state).** Every row of a resampled (game, period,
grid frame) takes the `stoppage_state` / `stoppage_source` of that frame's ball row (the ball's held source frame); a
grid frame without a ball row gets a missing value (not live), never a default; two ball rows in one frame raise.
`interpolate_frames` creates no rows; the readers read the ball row's state.

**The artifact gate (owner decisions 2026-10-26: a CI gate until regenerated; all 18 reader-built artifacts and the 2
older stale ones).** `tests/scripts/test_input_contracts.py` FAILS, never warns, while a strict committed artifact is
stale. The strict drivers are derived: every script that records `stoppage_token()` (less the two trainers, whose
product is the refused model), every script that reads `bundled_model_stoppage()` or `require_stoppage_provenance()`
(model-derived), and `derive_coordination_params` / `validate_match_outcome_calibration` (stale before this cycle).
Each strict driver has an `input_contract()` with the stoppage token (a model-derived one also the bundled models'
provenance) and writes it into its artifact; the tf60 harness copies the arm table's contract (`upstream_contract`).
The 12 committed artifacts without a contract are listed in `_OWED` with their producer (the artifacts name none); the
list must reach empty. A model-derived driver refuses a model metrics file without `stoppage_provenance`. At this
commit: 8 contract-carrying files are stale (tf19_entanglement, xcross_causal, tf19_instrument_responsiveness,
tf56_positioning, the 2 trained_artifact_integrity verdicts, tf58 `derivation.json`, tf53 `metrics.json`) and 12 are
owed (box_constant_delta; f1b_float32 metrics and hub_smoke; position_only_variants x2; tf19_pr3b; tf19_pr3b_xs_v2 x2;
tf19_signoff_power; tf24_stage2_refresh calibration_report; tf60_layer3 x2). Each is an owner-run regeneration, shown to
the owner before it starts; the model-dependent ones run after the retrain. Excluded, with the reason given to the
owner: tf57 (IDSSE only: the column equals the `ball_state` mapping), pr5_platform_atol (extractor numerics, no model
weights), gs_input_convention (events only).

**Restage (owner decision 2026-10-26: all three).** `xshot115`, `xshot17` and `xcross_probe` (recorded 2026-10-18 with
old-converter SkillCorner actions) are restaged at the pushed sha, owner-run with the HF write token:
`scripts/materialize_tc3_frames.py`, then `scripts/hf_jobs/stage.py`, then `python -m scripts.hf_jobs datasets record
<name>`. Until then the trainers refuse them at the load seam (a SkillCorner restart that is not a derived row raises).

**Older defects (owner decision 2026-10-26: fix all three in this cycle).** The `exact_prewarped_cutoff` doctest
expected 0.462389; the function and its formula give 0.462466, now the expected value (CI does not run private-module
doctests). The tf58 `derivation.json` and the tf53 `metrics.json` are gated (above) and regenerated owner-run.

**Fixtures added after the first report.** The glossary harness positioning leg (`_positioning_columns`) is
re-labelled `sportec` (4 tests compared, 0 changed lookups). The xCross trainer smoke's toy probe is re-labelled
`synthetic` -> `sportec` (`--probe-providers sportec`), and the dead `PROVIDER_ID_KIND["synthetic"]` entry is removed:
the trainer runs as a CLI subprocess, so the lookup check used a subprocess recorder (positive control first). Before
(gate off): 0 provider-keyed lookups in 54 processes; after: only the gate's own `dead_ball_observed` /
`validate_provider`; 0 changed. Two stub tests set the seam to the identity (the Databricks query-count scale guard,
the head-to-head vintage test).

**R-C1 final: stoppage rule version 2 (DL ruling 2026-10-27; verbatim:
`D:\Development\_handoffs\tf46p2-r2-diag\dl_ruling_rc1v2_q5_class5_2026-10-27.md`).**
- The added rule. A resolved `t_kick` is a span end only when span start < `t_kick` <= the restart row's own time (the
  placeholder). A later kick is after the receiver's possession began; a kick at or before the start is before the
  stoppage began. Both are impossible, so both are resolver errors. The span then ends at the placeholder as an UNKNOWN
  span, with `resolved_kick_rejected = True` (a new column of the span table).
- Superseded 2026-10-27 by the Q1 ruling (below): the flag became `resolved_kick_reject_reason`.
- The count. The R.2 run (9) gives the count to the resolver diagnostics: `tf46p2_kick_cells` records per match and
  type the resolved kicks and the rejected kicks by reason (`resolved_kick_rejections`, record
  `resolved_kick_reject_reason`, table `kick_cells_resolved_kick_reject_reason`). Its schema is
  `tf46p2-kick-cells-9`, and its profile stage key carries the rule version. The column report carries no reason
  count: the column refuses actions with `t_kick` (owner decision 2026-10-27, below).
- The property test (`test_no_action_instant_is_ever_inside_a_span`) has no exclusion: a resolved kick is drawn
  anywhere (before the previous action, at either action instant, between them, after the placeholder, after the next
  action). The band is tested from both sides (`t_kick` at the placeholder is accepted, 0.1 s later is rejected; at the
  open start is rejected, 0.05 s later is accepted). Mutant check: the version 1 end (any finite kick) fails all three
  new tests.
- `STOPPAGE_RULE_VERSION = 2`; `stoppage_source` is `events_v2`. The tests import the constant. A version 1 column, a
  version 1 model provenance and a version 1 shard token are refused, or are another generation.
- Author reading (surfaced, then ruled 2026-10-27 Q1, below): "span start" is the start rule read up to the restart
  row's own time, so the start never depends on the kick.
- Author naming (surfaced): the end source of every placeholder end is `placeholder` (`unknown` in version 1), as the
  ruling names it. The frame state of such a span stays `unknown`.
- With this rule the R-C1 rules are final: no further tuning to IDSSE. The TF-58 adoption (owner condition 2) can
  start. Accepted (IDSSE only; not transferable to SkillCorner): precision 0.950, recall 0.869; start offset p50
  -0.08 s; end offset p50 -0.39 s. On IDSSE (no resolved kicks) versions 1 and 2 give the same spans.
- Known limits. They go on the model cards (done: the four xS / xCross cards) and in the tf58 report; they do not
  reopen the rules: drop balls have no SPADL restart action, so they have no span (14 of the 19 uncovered IDSSE runs,
  26,236 frames), and on SkillCorner, Gradient Sports and Metrica these dead frames count as live; 24.1 % of the
  SkillCorner `ball_out` starts come from an extrapolated ball (shipped arm); the unknown-span cost is 22.9 % of the
  SkillCorner xS rows and 698 positives (shipped), 19.4 % and 296 with resolved kicks.
- Defect found by the throw-in probe and fixed: the goals reconciliation re-types a native SkillCorner shot to
  `shot_penalty` and keeps `action_provenance = "native"`; the core refused it as an old-converter row (3 of the 895
  staged SkillCorner matches). Such a row ends its span at its own event time (`_SC_OWN_TIME_RESTARTS`; a derived
  `shot_penalty` too, since the DL ruling of 2026-10-27, item 3).
- Measured (`tf58_spans_probe.py`, the pass-all resolver on the 108 SkillCorner xS matches): the built reading rejects
  812 of 5,334 resolved kicks (737 at or before the start, 75 after the placeholder, all at the receiver's possession
  time, placeholder + 0.02 s); the start read up to the kick (the version 1 window) would reject 119. Returned to the
  DL: which reading is the rule, and are the 75 rejected. The throw-in report (`rc1_v2_throwin_probe.py`): the
  uncovered detected touchline ball-outs are 1.17 per 90 on SkillCorner against 1.36 on IDSSE, so they do not explain
  the gap (29.0 against 41.5); the "missing markers" inference is not supported for the ball-outs the tracking sees.
  Accepted by the DL (2026-10-27): no rule change; the limit (a ball-out inside the dropped replay frames is not
  visible) is on the four cards.

**The Q1 / Q2 ruling and the owner decisions of 2026-10-27 (verbatim: `dl_ruling_rcv2_q1q2_tf58_2026-10-27.md`,
`owner_decisions_dl_recs_2026-10-27.md`).**
- Q1, the kick rule (version 2 kept: not committed, no seam passes `t_kick`, only builds with `t_kick` change). With
  P the previous action's instant, R the restart row's own time, K the kick and B the first ball-out of start rule (b)
  in (P, R): K <= P or K > R is rejected; P < K <= R with no B gives (P, K), dead; with B < K gives [B, K), dead; with
  B >= K the kick is rejected and the span is (P, R), UNKNOWN, `after_previous_action`. So no dead frame is marked
  live; only live frames are lost, and the count shows it. Built: `resolved_kick_reject_reason`
  (`KICK_REJECT_REASONS`: `none` / `before_previous_action` / `ball_out_after_kick` / `after_restart_row`) replaces
  the bool; `rejected_ball_out_detected` keeps the flag of the ball-out after the kick. B is searched in (P, R), open
  at R as in version 1, so spans without a kick do not move (the DL's "(P, R]" was returned to it; confirmed (P, R) on
  2026-10-27, below).
  Tests: each case from both sides of each bound; mutant: the built start (B) on a `ball_out_after_kick` reject fails
  (verified).
- Q2: the 75 kicks at the receiver's possession time stay rejected. Owner-approved resolver change: the SkillCorner
  derived-row bracket is `[lo, hi)` (`_arbitrate(hi_open=)`, `_bracket` returns the flag); a native anchor's bracket
  stays closed. Test: a contact or onset on `hi` is not a kick; mutant: `hi_open` ignored (verified).
- Measured on the 108 SkillCorner xS matches, the pass-all resolver (`kick75_probe.py`, before the bracket change):
  of the 75, 74 sit exactly at R + 0.02 s and 1 at R + 2.72 s; kick_time_source tf79_contact 43,
  conflict_resolved_tf79_contact 14, ball_onset 15, both 3; the next possession is a reception in 65 (free kick 22,
  goal kick 20, pass 10, throw-in 9, corner 4) and not a reception in 10 (start_type unknown 7, interception 3; 6 of
  the 10 by the restarting team). After both changes (`rule_v2_reasons_probe.py`): 5,305 resolved kicks; rejected:
  before the previous action 46, ball-out after the kick 699 (483 detected, 216 extrapolated), after the restart row
  0. The accepted kick-to-reception p50 / p90 (s): corner 2.2 / 19.5, throw-in 3.2 / 18.4, free kick short 2.7 /
  32.5, free kick crossed 2.2 / 32.0, goal kick 14.4 / 35.7.
- Owner 1: the name stays `derive_stoppage_state` (settled).
- Owner 2: `derive_stoppage_state` raises `ValueError` when the actions carry a `t_kick` column (even all-NaN): its
  label `events_v<N>` does not record the kicks. The core keeps `t_kick` for the R.2 probes. Test; mutant: the guard
  removed (verified). With R.2 (owner-approved timing): a labelled source naming the rule version and the gate-table
  identity, and a corpus driver asserting one `stoppage_source` across its games. The module fixture of
  `tests/tracking/test_stoppage_modules.py` takes its dead span from a goal instead of a resolved kick (same bounds).
- Owner 3a / 3b / 3c and the DL item 4: see the CHANGELOG (sigma note; the baseline's per-field provenance; the GS
  shot driver's guard verdicts; the dated corrections). The committed D20 precision 0.089 is cited only by the
  committed tf58 `metrics.json` (regenerated in the tf58 re-run) and in this section; the context doc's "909 of ~980"
  line carries a dated correction.

**The DL answers of 2026-10-27 (verbatim: `dl_answers_q2_bwindow_penalty_2026-10-27.md`), the Q2 census R1 / R2, and
the defects found by the full targeted run.** Results: `2026-10-27-tf46-part2-q2-r1-r2-items23-dl-handoff.md`.
- Item 2 (the B window): (P, R) is right ("(P, R]" was the DL's notation error). No code change. Test: a 2-frame
  ball-out run that starts at 24.8 s (R = 25.0 s, K = 24.0 s) is B and rejects the kick; a run that starts at 24.9 s
  (its second frame at R) or at R is not B, so the kick is accepted. Mutants: the window (P, R] (fails the 24.9 s case);
  a search that reads past R (fails the 24.9 s and 25.0 s cases). Both verified.
- Item 3 (a derived SkillCorner `shot_penalty`): it ends at R, `end_source = "event"`, dead up to R, with or without a
  kick (`_SC_OWN_TIME_RESTARTS`, native or derived). Test: no K, K just before R and K just after R. The property test
  draws derived `shot_penalty` rows. Mutants: the DL's (the span ends at any finite K) fails the specific test and the
  property test; routing the row through the rule-v2 kick branch fails the specific test (the property holds under it,
  as rule v2 rejects K > R). Both verified. The |K - R| report has no population: the 13 derived `shot_penalty` rows of
  the 108 SkillCorner xS matches carry no resolved `t_kick`, because the goals reconciliation
  (`spadl/_skillcorner_goals.py`) emits them without the GI bracket frame columns, which only the restart emitter sets.
- R1 (`q2_r1_raw_probe.py`): the IDSSE dead-ball freeze is in the raw DFL feed in all 3 windows (J03WMX periods 1 and 2,
  J03WN1 period 1). No frame is missing; a dead run repeats the last alive X / Y and writes Z = 0.0; the lakehouse
  fixture and the silly-kicks parse keep the raw counts, and `ball_state` equals the raw `BallStatus` on every frame.
  The resume is a relocation jump (4.69 m and 4.08 m in one frame).
- R2 (`q2_r2_kresume_probe.py`; fired counts equal the census at 6741710): |K - t_resume| <= 1 frame in 118 / 137 free
  kicks (0.861), 40 / 43 goal kicks (0.930), 243 / 279 throw-ins (0.871) and 1 / 1 corner. K is t_resume + 1 frame at
  p10 / p50 / p90 for the dead-ball types and t_resume at p50 for throw-ins; the K rest episode is the frozen hold at
  p50. R3 is the DL's.
- Defects found by the full targeted run and fixed (each test fails before the fix):
  - The part-2 toy built its SkillCorner units with native restart rows, a shape the staged SkillCorner actions never
    have, so the `kick_cells` profile unit raised in the stoppage core (`test_dataset_read_audit`; the slow parity test
    would fail the same way). The toy now builds them with `simulated_skillcorner_actions` (derived rows with GI
    bracket frame ids), the R.2 transfer-run model of a SkillCorner match.
  - Three new coordination `groupby` calls lacked `observed=True` (ADR-112; `test_reduce_categorical_safety`).
  - The D2 calibration golden (`tests/scripts/_fixtures/d2_calibration_golden.json`) re-baselined by its regenerator:
    only `input_contract.stoppage` and its digest moved (the stoppage token); every reliability value is unchanged.
  - The HF head-to-head (from the #288 merge) loaded the committed xS / xCross incumbent strictly, so the R-C3 public
    adoption (`cycle1.toml`) would fail. Owner decision 2026-10-27 (verbatim: `owner_decision_headtohead_override_2026-
    10-27.md`), the narrow auto-override: `_load_incumbent` loads strictly and reloads under `legacy_override=True`
    only when the refusal names the artifact's own stoppage provenance problem; the record goes into the head-to-head
    JSON; the fresh operand always loads strictly. Interim test: the shipped bundle as the fresh operand is refused, and
    the fresh operand is the same weights re-saved with a toy provenance. Mutants: an override on any refusal; an
    override without the strict try; an override on the fresh load. All verified.

**Rule version 3, the penalty fixes, R3 / R4 and the restart proxy (owner decisions 2026-10-27; DL rulings 2026-10-27;
verbatim in `D:\Development\_handoffs\tf46p2-r2-diag\`: `dl_ruling_q2_r3_item3`, `dl_answers_item3_q1q3`,
`dl_ruling_pen_marker`, `dl_ruling_pen_measure`, `dl_ruling_marker_end_time`, `dl_picks_owner_passed`,
`owner_decisions_builds_pens`, `owner_decision_pen_bracket`, `owner_decisions_pen_revised_fixA`, `dl_ruling_proxy_probe`,
`owner_decision_proxy_bar`, `dl_ruling_proxy_match`, `owner_decision_proxy_revised_binds`, `dl_ruling_proxy_match2`).**
- DL error 9 (owned by the DL): the derived SkillCorner `shot_penalty` sits at the START of the possession that won the
  penalty (`game_interruption_after = "penalty_for"`; 13 / 13 rows in the 108 xS matches, `pen_marker_probe.py`), not
  at the kick; GI has no possession for the kick. The item 3 rule ("ends at R, event") is retired.
- Converter (owner-approved, the revised design): a derived `shot_penalty` sits at the END of that possession
  (`time_seconds + (frame_end - frame_start) / GI_FRAMES_PER_SECOND`; the GI clock is a function of the frame); a marker
  matched through `game_interruption_after` never has its own shot re-typed (that shot drew the penalty), and the FIRST
  taker shot after it is the kick (a rebound is live play). No bracket and no kick search for penalties (the kick source
  stays the section 12 detector). Mutants: the start anchor, the re-typed marker shot, the last shot (each verified).
- Fix (A) (owner-approved): a new SkillCorner column `gi_interruption_after` (the raw GI tag on every GI-possession row,
  `<NA>` elsewhere; no existing column carried it).
- Stoppage rule version 3 (`STOPPAGE_RULE_VERSION = 3`; readers refuse version-2 columns; rebuild):
  - a derived SkillCorner `shot_penalty` ends no restart span; the goal rule covers it (dead from the row, open, to the
    next action); a native one still ends its span at its own time;
  - the penalty-marker span (`kind = "penalty_marker"`): from the marker possession's end -- `gi_possession_frame_end`
    mapped through the frames by the resolver's exact lookup (`_kick_resolver._frame_time`; `start_source =
    "gi_possession_end"`) -- or, when it does not map (no frames, a dropped frame, or a time before the row's own: the
    one fallback of start rule (b)), from the row's own time (`"after_previous_action"`); open; to the first action row
    after that start: DEAD (`"event"`) when it is a native shot by the same team, else UNKNOWN (`"next_action"`, a new
    end source); a known dead span wins an overlap;
  - one open-start rule for both readers (`start_is_open`: every start but `ball_out`) and one set of UNKNOWN end
    sources (`UNKNOWN_END_SOURCES`) for `frame_stoppage_state`;
  - SkillCorner actions without `gi_interruption_after` raise (an older converter: reconvert). The R.2 staging and the
    measurement subsets predate the column: they are restaged before the version-3 core reads them (owner-run).
  - Tests and mutants (verified): the restart-span rule kept for a derived penalty; the fallback making the shot branch
    UNKNOWN; the fallback giving no span; a nearest-frame mapping; a closed `gi_possession_end` start; a silent missing
    column. The property test (no action instant inside a span) draws random markers and frame ends.
- Measured (`pen_marker_probe.py`, the version-2 build, 30 marker groups in 108 SkillCorner matches): over (marker end,
  next possession start) the derived-row groups read live 0.059 (the goal span); the 15 row-less groups read live 0.93
  to 1.00 over a gap of p50 170 s -- the reason for fix (A). Seven row-less groups SCORED by the GI running score with no
  awarded-team success shot row within [start - 60 s, next + 600 s] (their matches report `swap` / `ambiguous`): the
  per-episode trace is report-only and open.
- R4 (owner-approved): `dead_ball_position_tracked` (IDSSE verified by R1; Sportec fail-closed); on such a provider
  `rest_episodes(provider=)` refuses a rest window with a `ball_state` "dead" row (no `ball_state`: every window), and
  `kick_verdict` reads `position_not_tracked` (counted `k_position_not_tracked`). Author reading, flagged to the DL: a
  repeat run in ALIVE play is a still ball at 1 cm resolution (the hold test), so the guard refuses dead rows only.
- R3 (owner-approved): `kick_reference_withheld` (one gate, `R3_WITHHELD_REASON`): the R.2 kick_time rows against an
  IDSSE / Sportec reference K are not scored; `kick_window_miss_rate` is `not_evaluable` with the reason; the kick-rise
  and onset tables carry it; the gate table's kick_time cells read `withheld (<reason>)`. Open to the DL: whether the run
  (1) K gate (`tf46p2_truth`) is withheld too.
- The restart proxy (owner bar: ESS >= 100 and <= 10 % of the C mass off support; the revised definitions bind):
  free kick PASSES (revised ESS 369.3, off 4.51 %); goal kick (ESS 50.7) and corner (off 70.3 %) withheld; throw-in and
  penalty withheld. The own-kick rule (the kicker's own SPADL kick action within +/- 2.0 s) and the pre-registered loft
  sensitivity are built with the free-kick proxy cells and return to the DL before the cells are used.

**TF-58 adoption (owner decision 2026-10-26, conditions 1-9; started after "R-C1 final").**
- `resolve_stoppages(mode="events" / "auto")` reads `event_stoppage_intervals(actions, frames, provider=...)`. An
  UNKNOWN span counts as dead (condition 4). `max_stoppage_s = 25 s` stays (condition 3) and filters after the core,
  in `resolve_stoppages` only (condition 1).
- `StoppageEvidence` keeps `source` and gains `spans` (`STOPPAGE_EVIDENCE_SPAN_COLUMNS`: kind, start, end,
  start_source, end_source; `ball_state` for both on the observed path) and `open_start` (condition 5). The split uses
  the frame column's bounds (`_PeriodStoppages`, `_in_intervals`): open at an event-span start, closed at a ball-out
  or `ball_state` start, open at every end.
- Condition 8: `test_the_coordination_split_and_the_frame_column_read_one_core` (the same spans and the same per-row
  mask; mutant: the old closed span). The old-rule fixture `make_coordination_actions` writes SkillCorner restarts as
  `derived` rows (the current converter does; the core refuses a native one); no expected value changed. The D1
  bootstrap-SE goldens are re-baselined: their seed is D1's contract digest, which now records the stoppage token.
- Condition 9: the GS restarts are native PFF events timed by the event clock (confirmed in code), so `event`.
- D3 counts the splits per provider by source and kind (table `stoppage_splits`, `metrics.json` `stoppage_splits`,
  `report.md`; `tf58-d3-2`, `tf58-metrics-3`). Its D20 leg's `STOPPAGE_PROVIDERS` is derived from the observed
  dead-ball set: IDSSE only. Finding: the committed D20 agreement (precision 0.089 over 71 matches) scored 64 GS
  matches against a constant `"alive"`.
- The four TF-58 drivers record `stoppage_token()` in their input contracts and shard tokens; their committed
  artifacts (`derivation.json`, `calibration.json`, `metrics.json`, `numerics_noflip.json`) are stale under the gate:
  the tf58 re-run, owner-run at a clean tree. The ship verdicts (H1-H7, `gated_pass`, condition 6) are compared then.
- Measured (966 staged matches; old -> new): SkillCorner 41,711 -> 36,049 splits (2,166,584 -> 1,833,896 dead
  seconds); Gradient Sports 2,636 -> 2,353 (126,574 -> 112,860 s); IDSSE 268 -> 268, all 7 matches identical. The D20
  leg on IDSSE: precision 0.904 -> 0.950, recall 0.896 -> 0.873, IoU 0.918 -> 0.934. The splits at risk (condition 4)
  are most SkillCorner unknown splits, because the p90 kick-to-reception times are long (18-36 s).

### 27.20 The c-matched camera draw (added 2026-10-27; DL rulings 2026-10-26 item 4 and 2026-10-27 item 2)

Verbatim: `D:\Development\_handoffs\tf46p2-r2-diag\dl_ruling_g7_rc1_q5_2026-10-26.md` (item 4) and
`dl_ruling_rc1v2_q5_class5_2026-10-27.md` (item 2). The v5 coverage check 2 showed the mechanism: the in-view loss
falls from 0.25 at delta_c > 15 m to 0.04 at < 5 m, and 65 % of the player rows sit at delta_c > 15 m. The drawn
window was framed for another player configuration. The placement stays ball-relative; the DRAW changes.

- **Key.** c = (outfield centroid - ball) in the camera frame, 2-D (the outfield players: every non-ball,
  non-goalkeeper row, extrapolated SkillCorner players included, as delta_c read them). A point reflection negates a
  difference vector, so c in the camera frame = sigma x c in the staged frame. A window's c is the mean of c over the
  frames of the window proper (its continuation excluded); the reference c is the mean over the frames the window will
  cover (open play: [the segment start, + `cam_seg_s`), cut at the next restart; restart: the restart window's frames).
- **Open play (the stitching rule stays, spec section 5.3):** the eligible cell members (`skip_first_null` as now) ->
  the `cam_k_c` = 25 nearest by |c_source - c_reference| -> among those, the `cam_stitch_k` = 5 nearest by the seam
  distance -> uniform, seeded; ties by window id (the store order). At a chain start (no live footprint): uniform among
  the 25.
- **Restart (no seam rule):** the phase fallback only under its ruled condition (an empty cell, or the nearest D
  shorter than the reference D by more than `cam_seg_s`), then on the chosen view: (1) the members with
  |D - D_ref| <= `cam_seg_s`; (2) if none, the `cam_stitch_k` nearest D; (3) the `cam_k_c` nearest by |delta_c| in that
  pool (all when fewer), then uniform, seeded.
- **Author choices (surfaced):** a reference span with no outfield player gives no c; that draw skips the c stage and is
  counted (`c_ref_missing_draws`, in the chain counts and the report). A member with no c sorts last. The camera store
  fails closed without c (schema 4 requires `frame_shape`; `build_camera_store` refuses a window without it). An
  in-memory `CameraFollowModel` (test fixtures) may carry no c: its members then all sort last, and a timeline
  without `cx` / `cy` skips the c stage, counted as above.
- **Constants:** `cam_k_c = 25` joins `RATIFIED` (DL-ruled; this section records it, dated 2026-10-27). Its
  sensitivity is the DL's explicit pair 10 / 50 (`EXPLICIT_SENSITIVITY`), not the +/-50 % rule (12 / 38).
  `cam_stitch_k` keeps 2 / 8.
- **Store (schema `tf46p2-camera-store-4`):** per stored frame `frame_shape` = (c_x, c_y, the outfield spread (the
  RMS distance of the outfield players from their centroid), the source players in the frame's footprint, the source
  players with a known in-view state); per window `seq_c` and `seq_spread`. The camera window records carry
  `cam_shape`. The stores are rebuilt.
- **Report (statistics only for the arms, never a score row):** per profile, on the base camera variant, for the draw
  as built and two report arms, the v5 draw (`seam`) and c alone (`c_only`): delta_c per player row by band (< 5,
  5-15, > 15 m); the degraded and the source in-view share by band; the reuse (the most draws of one source window,
  the distinct windows drawn); the spread second key (the in-view gap by |spread_source - spread_reference| band:
  < 2.5, 2.5-5, >= 5 m, author-chosen bands); the open-play seam jump (the mean corner distance at each seam); per
  restart draw the pool size after steps (1)-(3) and the share below `cam_k_c`. Sensitivity: `cam_k_c` 10 / 50 and
  `cam_stitch_k` 2 / 8, one at a time.
- **Return conditions (unchanged):** more than 20 % of the player rows stay at delta_c > 15 m; a fidelity row that
  passed in v5 fails after the change. Choose by the mechanism, never by the fidelity gate. Then the full fidelity
  table is re-measured on the measurement units.

**The trajectory key (DL ruling 2026-10-27, cdraw; verbatim `dl_ruling_cdraw_2026-10-27.md`).** The first run (the
window-mean key) met both return conditions (gt15 share 0.373; 6 rows passed in v5 and failed). The key changes to the
TRAJECTORY distance: the mean over the frames both sides have of |c_src(t) - c_ref(t)|, aligned from the window start
(a reference frame maps to the nearest window row, never into a time gap or past the window proper); a member sharing
fewer than half of its window-proper frames is not eligible (`c_key_ineligible_draws` when none is). `C_KEYS`
(`trajectory` ruled, `window_mean` the report arm). The chain records the draw index per frame and the chosen member's
key distance (`src_draw`, `draw_keys`). Report arms: `draw_arm=seam` (the v5 draw on the same tree) carries its OWN
fidelity rows, so the draw is the only difference against `base`; `draw_arm=c_only` and `draw_arm=c_window_mean` are
statistics only. Report keys added: the delta_c > 15 m rows by the drawn window's own key distance (<= 5 m drift / > 5 m
pool) and the spread bands inside delta_c < 5 m. `cam_seg_s` stays 10 s (a ratified constant).
- Measured (cl42sub-g; `ctraj_summary.txt`): the gt15 share 0.369 (window-mean arm 0.373; seam arm 0.650) -- return
  condition 1 still met; under the trajectory key 98.4 % of the gt15 rows come from draws whose own key distance is above
  5 m (the pool), against 72.4 % under the window-mean key. Against the seam arm on the same tree one row passed and
  fails (goalkick_box player_visibility_near_ball 0.820 -> 0.872, target 0.804) and three failed and pass -- return
  condition 2 still met; most of the first run's v5 -> new losses were tree changes since v5, not the draw. The spread
  gap inside delta_c < 5 m does not fall with the spread mismatch: no second key. Returned to the DL; v6 waits.
