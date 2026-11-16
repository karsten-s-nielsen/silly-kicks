# TF-79 ball-contact detector v2 — design

- **Status:** design approved by the owner and by independent reviews A and B (2026-11-15). This text restates that design in its final state (owner ruling 2026-11-15); the decision trail is in ADR-134 and the review reports `D:\Development\_reviews\2026-11-15-cross-definition-*`.
- **Branch:** `feat/cross-definition` (the cross-definition cycle).
- **ADR:** ADR-134 (the v2 design); an amendment to ADR-124 (the validation rule, section 6; the `min_velocity_change_ms` default, section 4).
- **Parent:** `docs/superpowers/specs/2026-11-13-cross-definition-design.md` (decision 12 there).

## 1. Intent

`detect_ball_contacts` (TF-79, ADR-124/125) finds ball touches in tracking data and credits each to a player. Two consumers ask one question: after an action, who touched the ball next? The cross enricher (TF-86) ends a cross's ball path at the first contact by another player. TF-46 corners take the first contact after the kick as the first-contact winner. On Gradient Sports the v1 answer is a false touch in flight on about half of all releases (section 3), so a cross label, a corner winner or a block call built on it cannot be better than it.

The goal: after a release, the first touch by another player is the right player at the right time, measured against independent ground truth. The detector stays provider-neutral.

Owner rulings (2026-11-15):
1. Redesign the detector in this cycle.
2. Physics rules first. A pre-declared bar on held-out matches decides. If the rules miss the bar, a learned scorer on top of the rule candidates follows in this cycle, under its own spec round.
3. ADR-124's validation is amended: thresholds are chosen on a development half of the GS matches and scored once on the held-out half (section 6).
4. Every "next contact" consumer skips the acting player's own contacts (section 5).
5. The next-touch measure and its bar (section 6), with the timing row at ±0.10 s; the design stays minimal: two tuned knobs.

## 2. Ground truth (PFF on-ball sequences)

Every raw GS `OTB` game event is one player's on-ball sequence: a player, a team, and `[startTime, endTime]` on the event clock. `startTime` is the first touch (the reception); `endTime` is the last touch (the release). Between two sequences the ball is in flight and nobody touches it. The clock is made period-relative with the period's first kickoff `startTime`, as the GS converter does (cross spec 6.6 B); the staged frames carry `periodElapsedTime`.

**What the truth can and cannot judge.**
- *The sequence edges.* On isolated edges of the development half (no other PFF edge within 0.5 s, the sequence at least 0.5 s long; about 115 edges each), the largest ball velocity-vector change within ±0.3 s of the edge lies a median −0.066 s from the edge; 10th–90th percentile about −0.23 to +0.23 s. The proxy is itself noisy, so the true agreement is better, but the truth supports a tolerance of 0.2 s, not 0.1 s.
- *A filtered release check* (a speed rise of more than 3 m/s over two frames; passes, crosses, shots and clearances; all 62 matches): the change minus `endTime` is a median 0.000 s, interquartile range −0.067 to +0.066 s (n 54,534).
- *Inside a sequence* the truth has only the two edges. A carry holds several touches that PFF does not time, so a contact inside a sequence, away from its edges, cannot be judged; the measures leave such contacts out.
- *Bias in both directions.* A touch that PFF does not record counts against the detector; an edge timed within the tolerance counts for it. Only GS has this truth in the corpus: IDSSE events are hand-timed (event-to-contact offset median −0.10 s, interquartile range −0.84 to +0.42 s, `docs/research/ball_contact_detection/metrics.json`), and SkillCorner events come from broadcast tracking.

## 3. Evidence (read-only, DGX, 2026-11-15)

The scripts are in `D:\Temp\cross-def\`; the outputs are in `~/cross-def/` on the DGX. The data is the commit-3 re-stage (`~/cross-def/xshot187-r8/out`, generation `3836f5993b9c5a9f`), with the v1 defaults.

**What the held-out half influenced.** Section 3.1 ran on all 62 matches before the split existed; its numbers motivated the design (the reach gate, the removal of RDP, the knee time). Section 3.2, section 3.3 and the tolerance read the development half only. The held-out run therefore guards the two tuned thresholds, not the design choices.

### 3.1 The v1 baseline (all 62 matches)

375,305 contacts, 131,967 sequences, tolerance 0.1 s: right player inside the right sequence 30.7 %, in flight between sequences 32.4 %, inside a sequence but another player 7.2 %, unresolved 29.6 %. The nearest contact to a release by the releasing player is a median 0.20 s early. Three causes:
- *No reach gate.* v1 credited the nearest player within 2.0 m and, when nobody was there, the nearest within 5.0 m. The false contacts' nearest player is a median 2.35 m away; the right contacts' 0.97 m.
- *Path-shape candidates.* RDP vertices (0.3 m) fire on the curve of a ball in flight; the false contacts have the ball higher (median z 0.16 against 0.00 m) and faster (incoming 5.98 against 3.81 m/s).
- *Early timing.* An RDP-only cluster took its earliest vertex, so a release was placed early.

Separating signals (rank AUC, right against false): nearest-player distance 0.81, ball height 0.68, incoming speed 0.67, gap to the second-nearest player 0.64; v1's `velocity_change_ms` points the wrong way (0.39).

**Crosses** (847 GS open-play crosses with PFF code B, C or D): an opponent contact before PFF's next event on 62 % of the completed and 63 % of the defended crosses. PFF times the block a median 0.10 s after the cross (10th–90th percentile 0.03–0.25 s).

### 3.2 Gates alone do not suffice (development half)

A reach gate and a height gate on the v1 candidates give at best precision 0.74 and keep only 36–65 % of the right contacts at radius 0.8–1.2 m. The candidates and their timing must change too.

### 3.3 The next-touch measure on v1 (development half, 24 matches)

30,189 releases followed by another player's sequence that starts after the release. The first contact credited to a player other than the releaser, at or after the release:

| Class | τ 0.15 s | τ 0.20 s | τ 0.30 s |
|---|---|---|---|
| right (the next player, within τ of the reception) | 21.9 % | 26.3 % | 33.8 % |
| false first (any player, before the reception − τ) | 55.7 % | 51.7 % | 44.1 % |
| wrong player (within τ of the reception) | 4.1 % | 5.1 % | 7.0 % |
| missed (nothing until the reception + τ) | 18.3 % | 16.9 % | 15.1 % |

For blocks (the next sequence is the blocker's `RE`, n 870), 54.4 % are missed at 0.2 s. Release timing (27,435 releases of the kick types): no releaser contact within ±0.5 s on 44.2 %; otherwise a median −0.13 s (10th–90th percentile −0.33 to +0.10 s).

### 3.4 Duplicate runs (all three providers)

Exact-duplicate ball runs lasting longer than the velocity window (item 1 of section 4) are a ball at rest, almost never a stale repeat of a moving ball: on 8 GS development matches, 15 of 452,552 moving-ball runs (speed in and out at least 2 m/s) last longer than the window. IDSSE freezes the dead ball (about 150 long runs per match; SkillCorner about 4). Collapsing a rest leaves a kick from it without "before" samples. On 7 IDSSE matches, 70 kicks leave a rest of at least 0.3 s (first step 2–40 m/s; a faster step is a jump of the frozen ball, not a kick): with rest runs kept, 70 have a contact within 0.12 s; with every run collapsed, 32 (loosest grid point) (`D:\Temp\cross-def\rest_runs.py`, `static_kicks.py`).

## 4. Design

The detector keeps its public entry points (`detect_ball_contacts`, `add_ball_contacts`, `first_contact_in_window`), its output columns and its row contract (an unresolved candidate is kept with `player_id` `<NA>`, ADR-124 E7). What changes is how a candidate is found, timed and credited, and how the consumers skip the actor (section 5).

1. **Ball kinematics.** The window is `max(kinematic_window_s, 2 × frame period)`, the same floor as the merge window: a mean velocity needs at least two frame intervals to average out one noisy sample (0.067 s at 29.97 Hz, 0.08 s at 25 Hz, 0.2 s at 10 Hz). The trajectory keeps the ADR-125 duplicate-frame dedup, with one refinement: a run of identical positions that lasts longer than the window is a ball at rest, not a stale repeat, and all its samples are kept (section 3.4). At each sample, the horizontal ball velocity before it is the mean over the window that ends at the sample, and the velocity after it is the mean over the window that starts at the sample; windows are in seconds on the deduplicated samples.
2. **Candidates.** The velocity change at a sample is |v_after − v_before|. A candidate is a local maximum of the change that is at least `min_velocity_change_ms`, where the incoming or the outgoing speed is at least `min_candidate_speed_ms` (a ball at rest that is kicked has no incoming speed). Candidates within `max(merge_window_s, 2 × frame period)` of each other keep the larger maximum. The candidate time is the time of that maximum (the knee). RDP is removed. Only the horizontal velocity is used, so a ground bounce is not a candidate; a header changes the horizontal velocity and stays one.
3. **Reach and credit.** At the candidate's frame, the nearest player whose horizontal distance to the ball is at most `contact_radius_m` gets the credit, while the ball height is at most `contact_max_z_m` (2.6 m). There is no fallback radius. Where the ball z is NaN or not `"measured"` (ADR-125), the height check is skipped (`aerial_source` records it). A candidate with no player in reach, or a frame without player rows, is an unresolved row.
4. **Output values.**
   - `velocity_change_ms`: the candidate's own change, the value that admitted it.
   - `confidence`: `_confidence` keeps its form. Its proximity term is `1 − dist / contact_radius_m`; the source weight is the constant 1.0. Unresolved rows keep NaN.
   - `contact_source`: the closed vocabulary `CONTACT_SOURCE_VALUES` keeps its four values, because the category dtype is built from it and a consumer may compare the category set (Hyrum). v2 emits `"velocity_break"` (credited) and `"unresolved"` only.

The code splits into `_candidate_peaks` (items 1–2 without the threshold: it reads only `kinematic_window_s` and `min_candidate_speed_ms`) and `_finalize_contacts` (the threshold, the merge, the reach and the outputs), so the tuning harness scores exactly the shipped code.

**`BallContactParams` (public, `tracking.__all__`).** The class is keyword-only (`@dataclass(frozen=True, kw_only=True)`), so a positional call from v1 code raises `TypeError` instead of binding silently to shifted fields.

| Field | v2 |
|---|---|
| `contact_radius_m` | the reach gate (tuned, section 6) |
| `contact_max_z_m` | 2.6 m (physics prior, not tuned) |
| `min_velocity_change_ms` | the candidate threshold (tuned); its v1 meaning, an opt-in floor with default 0.0, ends |
| `min_candidate_speed_ms` | 2.0 m/s; the incoming or the outgoing speed |
| `merge_window_s` | 0.12 s (not tuned) |
| `aerial_z_m`, `dv_scale_ms` | unchanged |
| `kinematic_window_s` | **new**, 0.067 s: two frame intervals at 29.97 Hz (not tuned) |
| `rdp_epsilon_m`, `speed_drop_frac`, `heading_turn_deg`, `contact_exist_radius_m` | **removed**; passing one raises `TypeError` |

`__post_init__` checks: `kinematic_window_s > 0`, `contact_radius_m > 0`, `min_velocity_change_ms >= 0`, `merge_window_s > 0`.

## 5. The consumers skip the actor

v2 times a release at the kick, so a window that starts at the action time holds the actor's own release contact on about half of the GS and SkillCorner rows. Without a rule, TF-46 would name the corner taker as the first-contact winner, and `add_ball_contacts` would report the actor's own touch.

The rule, one for every "next contact" consumer (the cross enricher already applies it to the passer): **the next contact is the first resolved contact by a player other than the acting player.**
- `first_contact_in_window(contacts, period_id, t0, t1, *, exclude_player_id=None)`: a new keyword-only argument (additive); contacts credited to that player are skipped.
- TF-46 (`corners/_windows.py`) passes the corner taker's id: `identify_corners` carries it as the new column `taker_player_id` (the `player_id` of the corner action; `<NA>` when the actions carry none). `add_corner_windows` refuses a corners table without the column (`ValueError` naming `identify_corners`), because skipping nobody would let the taker be named the winner without notice; a `<NA>` taker is allowed.
- `add_ball_contacts` skips the action's own `player_id` in its `[action time, next action time)` window.
- Both compares are dtype-safe through `id_compat` (`ids_match`, `canonical_id`; ADR-019): an action `player_id` of `Int64` 366 and a frame `player_id` of `"366"` or `366.0` are the same player. Both ADR-019 gates cover them: `first_contact_in_window` has an id-scalar invariance entry for `exclude_player_id`, and `add_ball_contacts` is a linked aggregator in the id-dtype gate.
- Cost: a real second touch by the same player before anyone else is skipped. At a set piece that is a double-touch offence; in open play it is the next action, which has its own window.

Otherwise unchanged: the output schema `_CONTACT_COLUMNS`, the dtypes, `is_aerial` and `aerial_source` (ADR-125), purity, NaN safety and provider neutrality.

## 6. Tuning and validation (pre-declared; amends ADR-124)

ADR-124 chose its defaults from physics priors and a synthetic fixture. v2 chooses two thresholds against PFF truth, with a development and held-out split: the reach (`contact_radius_m`) and the size of a real change (`min_velocity_change_ms`). Every other parameter is a physics prior or keeps its v1 value.

**The split.** The id is the PFF game id as a decimal string (`str(int(game_id))`, UTF-8), never the staged key `gradientsports__<id>`. A match is in the development half when `int(sha256(id).hexdigest(), 16) % 2 == 0`. The harness scores every GS match the pining loader lists (64): 24 development, 40 held-out. The held-out half includes the two TF-19 probe matches (10502, 10503, `HELD_OUT_PROBE`): the probe rule bars training, and the detector is tuned on the development half only (owner ruling 2026-11-15, fixed before any held-out look). The harness test pins the development list on the 62 staged matches.

**The measures** (tolerance τ = 0.2 s, from section 2).
- *Population:* every pair of consecutive `OTB` sequences in a period, S1 then S2, by different players (P1, P2), with S2's `startTime` strictly after S1's `endTime`.
- *Next touch:* the first resolved contact credited to a player other than P1, at or after S1's `endTime`. Its class: **right** (credited to P2, within τ of S2's `startTime`, edges included); **false first** (any player, earlier than S2's `startTime` − τ); **wrong player** (another player, within τ); **missed** (none until S2's `startTime` + τ). The four shares sum to 1.
- *Release timing:* over the S1 of the kick types (`PA`, `CR`, `SH`, `CL`), the contact credited to P1 nearest to S1's `endTime`, within ±0.5 s; the median of that contact's time minus `endTime`. A release with no such contact is counted as missing and reported.
- *Reported, not gated:* the block class shares (S2 of type `RE`) and the unresolved rate.

**The bar** (held-out half; v1 on the development half in brackets):

| Measure, τ 0.2 s | Bar (inclusive) | v1 (development half) |
|---|---|---|
| next touch right | ≥ 0.60 | 0.263 |
| next touch false first | ≤ 0.10 | 0.517 |
| release timing median | within ±0.10 s | −0.13 s |

The timing row is ±0.10 s, three frames at 29.97 Hz, because the ball's own change sits about two frames before `endTime` on isolated releases (section 2): a tighter row would fail an exact detector on the truth's clock offset. It still catches a bias like v1's.

**The grid** (development half only): `contact_radius_m` ∈ {0.8, 1.0, 1.2, 1.5} × `min_velocity_change_ms` ∈ {1, 2, 3, 4, 5}; 20 points.

**The selection** (a total order).
1. A point is *feasible* when it meets all three bar rows on the development half.
2. Among feasible points, keep those whose "right" share is at least the best feasible "right" share − 0.005.
3. Among them, take the larger `min_velocity_change_ms` first, then the smaller `contact_radius_m` (the stricter detector).
4. **No feasible point:** the cycle stops before the held-out run, and the result goes to the owner (the learned-scorer round). The held-out half stays unseen.
5. **A chosen point on the grid's edge** is accepted and reported; extending the grid on the development half is the owner's call, before the held-out run.

**The order of the runs.**
1. The development run reads the clean code commit (commit 4) and writes the chosen point to its report. The author copies it to the ledger. This freezes the point.
2. The held-out run reads the same commit and takes the point from that report (`--dev-report`); it accepts no match list and refuses a report from another commit, so it cannot score a different point or population.
3. Commit 5 sets the defaults to the chosen point and cites both runs.
- **A run that fails before it prints a score** has not looked; it is rerun after the fix, and the ledger says so.
- **A bug found after the held-out look:** the owner decides. A fixed rerun counts as a second look, and the memo states it.

**Other providers** (not tuned). With the chosen point, the harness reports for IDSSE and SkillCorner the share of event actions that have a contact credited to the same player within 0.5 s, and the unresolved rate. The reference is the same report from the same harness code at commit 3 (`dd41724d`, the v1 detector), on a clean checkout, over the same matches loaded through the pining loader; commit 4 changes no converter, so only the detector differs. **Owner trigger:** the agreement share falls by more than 0.10 (absolute) from the reference for either provider, or the SkillCorner unresolved rate rises by more than 0.10. It is read with the held-out run; when it fires, the owner decides before commit 5.

## 7. The harness (in the repo)

The measurement comes from a committed driver (ADR-037), `scripts/measure_ball_contact_validation.py` (extended, not duplicated):
- a GS truth mode (`--gs-truth`) that reads the raw GS events through the pining loader (the accessor `gradientsports_events_json` returns the cached events JSON) and reports the section 6 measures;
- `--split dev` (scores the 20-point grid and writes the chosen point) and `--split holdout --dev-report <file>` (scores that point only);
- the existing IDSSE and SkillCorner reports, with the section 6 trigger (`--reference`).

It keeps the driver contract (`for_each`, ADR-052), the compute-contract gate (ADR-128) and the provenance wiring (`tests/scripts/test_provenance_wiring.py`).

## 8. Consumers

| Consumer | What it reads | Check |
|---|---|---|
| `tracking/_cross_delivery.py` (TF-86) | the first non-passer contact (the path end), the passer's own contact (`passer_contact_dt`) | the cross calibration runs read the tuned detector (cross spec 7.4) |
| `corners/_windows.py` (TF-46) | `first_contact_in_window` after the kick | the actor skip, tested both sides |
| `add_ball_contacts` (public; re-exported by `tracking/features.py`, called by no in-repo aggregator) | per-action `first_contact_*` | the actor skip, tested both sides |
| `scripts/build_corner_routines.py`, `corners/_encode.py` | contact-timed corner windows feeding `RunZoneModel`, `CornerRoutineModel` and the run encoding | owed refits listed in `docs/research/cross-definition/caller-sweep.md` |
| `scripts/validate_corners.py` | the same windows | its tests |
| `scripts/measure_ball_contact_validation.py` | the detector | extended (section 7) |

## 9. Tests

On synthetic trajectories, each band from both sides:
- **Candidate:** a change just above and just below `min_velocity_change_ms`, and the threshold itself (inclusive); a straight ball at constant speed gives none; a curved flight gives none.
- **Speed gate:** a kick from an exactly static ball (incoming 0, outgoing above `min_candidate_speed_ms`) is a candidate; a change with both speeds below it is not.
- **Dedup:** a rest run longer than the window is kept whole and a run of exactly the window or less is collapsed (helper level); a one-frame stale repeat in flight gives no candidate.
- **Timing:** a kick at a known time gives a contact within one frame of it.
- **Reach:** a player at `contact_radius_m` ± 0.05 m; the ball above `contact_max_z_m`; NaN z; a player only within 5 m; a frame without player rows.
- **Bounce:** a measured-z ground bounce with an unchanged horizontal velocity gives no candidate.
- **Merge:** two changes closer than the merge window give one contact, farther apart two; the floor is two frame intervals.
- **Frame rate:** the same kick at 29.97, 25 and 10 Hz gives one contact at each rate; one noisy sample in flight is not a candidate over two frame intervals.
- **Outputs, params, contract:** `velocity_change_ms` is the admitting change; `confidence` at distance 0 and at the reach; only `"velocity_break"` and `"unresolved"`; removed fields and positional calls raise `TypeError`; each `__post_init__` check; columns, dtypes, purity, NaN safety, an empty and a single-frame input.
- **Actor skip:** `first_contact_in_window` with `exclude_player_id`; TF-46 with the taker's contact just after the kick; `add_ball_contacts` the same; the ADR-019 dtype gates.
- **Harness:** the split lists; the four classes, their τ edges and the strict S2-after-S1 population; the release timing takes the nearest contact; the first kickoff anchors the clock; the selection on keys that agree and that disagree, the near-best window, the inclusive bar rows and the no-feasible-point stop; the held-out run scores exactly the point in the given development report; the trigger on both sides, the unresolved arm for SkillCorner only.

## 10. Non-goals and anti-goals

- **Non-goal:** a learned scorer, unless the bar is missed.
- **Non-goal:** touch truth for providers other than GS.
- **Anti-goal:** a provider branch inside the detector.
- **Anti-goal:** tuning on the held-out half, or scoring it more than once except under section 6's bug rule.

## 11. The author's calls

- Horizontal velocity only, so bounces are not candidates.
- The candidate time is the velocity-change maximum (the knee).
- A duplicate run longer than the velocity window is a ball at rest (section 3.4), so the dedup threshold is the window itself, not a new parameter.
- `"rdp"` and `"both"` stay in `CONTACT_SOURCE_VALUES` and are no longer emitted.
- `contact_max_z_m` stays a physics prior (2.6 m); v2 does not special-case the keeper's hands.
- `kinematic_window_s` (0.067 s) is a physics prior and `merge_window_s` keeps v1's 0.12 s; only the reach and the change threshold are tuned.
- The tolerance 0.2 s, the 20-point grid, the selection order and the other-provider trigger are the author's; the bar's numbers are the owner's.
