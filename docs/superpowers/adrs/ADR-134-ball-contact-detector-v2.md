# ADR-134: Ball-contact detector v2 — kinematic candidates, a reach gate, and a held-out truth bar

| Field | Value |
|---|---|
| **Date** | 2026-11-15 |
| **Status** | Accepted |
| **Deciders** | Karsten Nielsen (owner rulings of 2026-11-15) |
| **Review record** | independent reviews A and B (`D:\Development\_reviews\2026-11-15-cross-definition-*`) |
| **Supersedes / amends** | supersedes the candidate and attribution design of ADR-124 (RDP UNION velocity breaks; the 5 m fallback); amends ADR-124's validation rule (Notes 3) and its no-hard-drop rule (Notes 5); refines ADR-125's dedup (rest runs kept) |
| **Source spec** | `docs/superpowers/specs/2026-11-15-tf79-contact-detector-v2-design.md` |

## Context

Two consumers ask one question of `detect_ball_contacts`: after an action, who touched the ball next? The cross
enricher (TF-86) ends a cross's path at that contact, and TF-46 takes it as the corner's first-contact winner.
Scored against PFF's own on-ball sequences on Gradient Sports (detector spec section 3), the v1 answer was a false
touch in flight on about half of all releases, and the right player within 0.2 s on about a quarter. Three
measured causes: no reach gate (the nearest player within 5 m was credited when nobody was within 2 m),
path-shape candidates (RDP vertices fire on a ball curving in flight), and early timing (an RDP-only cluster took
its earliest vertex, so releases came about 0.2 s early).

## Decision

1. **A candidate is a kinematic change.** At each sample, the change between the mean horizontal ball velocity over
   the window before it and over the window after it; a candidate is a local maximum of that change, at least
   `min_velocity_change_ms`, where the incoming or the outgoing speed reaches `min_candidate_speed_ms`. The window
   is `max(kinematic_window_s, 2 × frame period)`. RDP is removed. The candidate time is the maximum (the knee).
   The ADR-125 dedup keeps a run of identical positions that lasts longer than the window: that is a ball at rest
   (a provider that freezes the dead ball, such as IDSSE), not a stale repeat, and collapsing it would leave a kick
   from rest with no "before" samples. Stale repeats of a moving ball are shorter than the window (GS: 15 of 452,552
   moving-ball runs are longer).
2. **A contact needs a player in reach.** The nearest player within `contact_radius_m` is credited while a measured
   ball is no higher than `contact_max_z_m`; there is no fallback radius. Nobody in reach -> an `"unresolved"` row.
3. **The detector is two steps** (`_candidate_peaks`, threshold-free; `_finalize_contacts`, threshold, merge and
   reach), so the tuning harness scores exactly the shipped code.
4. **Every "next contact" consumer skips the acting player** (`first_contact_in_window(..., exclude_player_id=)`,
   TF-46 via the new `identify_corners` column `taker_player_id`, `add_ball_contacts`), dtype-safe (ADR-019).
5. **`BallContactParams` is keyword-only.** `rdp_epsilon_m`, `speed_drop_frac`, `heading_turn_deg` and
   `contact_exist_radius_m` are removed; `kinematic_window_s` is new; `min_velocity_change_ms` becomes the
   candidate threshold. `contact_source` keeps its four-value vocabulary and emits `"velocity_break"` and
   `"unresolved"` only.
6. **Validation (amends ADR-124 Notes 3).** Two thresholds (`contact_radius_m`, `min_velocity_change_ms`) are chosen
   on a 20-point grid over a pre-declared development half of the GS matches (sha256 of the PFF game id, even) and
   scored once on the held-out half against the owner's bar: next touch right ≥ 0.60, false first ≤ 0.10, release
   timing median within ±0.10 s (τ 0.2 s). The design itself used all 62 matches (phase 1), so the held-out half
   guards the two thresholds, not the design. A missed bar sends the cycle to a learned-scorer spec round.

## Amendment (2026-11-15): the contact scorer is the credit step

The rules of decisions 2 and 6 missed the bar on the development half (no feasible point: best right 0.345, false
first 0.200; the held-out half was not scored). A development-half diagnostic located the failure: a candidate lies
within 0.2 s of 90 % of PFF receptions, but a fixed reach radius is the wrong credit rule (the receiver is within
1.5 m of the ball on 50 %, yet the nearest player on 69 %). Per owner ruling 2 of the detector spec, a learned scorer
on the rule candidates follows (scorer spec `docs/superpowers/specs/2026-11-15-tf79-contact-scorer-design.md`):

7. **The scorer credits a candidate.** The candidates stay threshold-free; the K = 3 nearest players are the
   options; a logistic regression (fitted with scikit-learn, served with numpy, parameters only) scores each option
   on the candidate's kinematics, the option's distance, rank, gap and approach, and the measured ball height; the
   best option is credited when its probability reaches `min_contact_probability`. `contact_scorer=None` keeps the
   reach gate as the model-free option. `confidence` becomes the credited probability.
8. **Validation.** One threshold is tuned by 6-fold grouped cross-validation on the development half; the refit on
   all development matches is frozen and scored once on the held-out half against the unchanged bar. The scorer is
   fitted by the `DRIVERS` pipeline `build_contact_scorer` on the staged `gs_contact_truth` dataset (the loader's
   frames plus PFF's sequences; ADR-123: no pining in the job), proved equal to the live loader before training.
9. **Artifact.** `silly_kicks/tracking/_contact_scorer_weights/default/` with the ADR-050 feature contract on a
   committed trajectory probe; non-reversible parameters on a publicly obtainable corpus, so it ships public
   (ADR-116, ADR-132).

## Amendment 2 (2026-11-16): scorer v2 and the track-supported pairs

The logistic scorer of decision 7 missed the bar out of fold on the development half (run on commit 4 `cb456270`: no
feasible point; best right 0.335, false first 0.278; the held-out half was not scored). A development-half
diagnosis (scorer spec r7 section 1.3) found three causes. A stronger model on the same features gained only 0.05.
Real receptions are soft and look like flight jitter at the peak frame, and the right receiver is often credited
0.3–0.4 s before PFF marks control. And the truth and the tracks disagree on many pairs: within τ of PFF's
reception, the tracks put PFF's receiver within 1.5 m of the ball on only 58.6 % of pairs, so "right ≥ 0.60" over
all pairs asks for about what the tracks can show at all. Owner rulings of 2026-11-16 replace decisions 7 and 8 in
part (scorer spec r7):

7a. **Scorer v2.** The options and the candidates are unchanged. Seven control features from positions describe
   what the ball and the option do in the 0.4 s before and after the peak (path speeds, the ball-option distance,
   their relative displacement). A gradient-boosted tree ensemble (scikit-learn `HistGradientBoostingClassifier`,
   fixed settings, exported to plain tree arrays and served with numpy; parameters only, pickle-free) scores the 16
   features. A first-touch decoding then keeps a peak only when no peak within ±0.3 s has a higher best-option
   probability (temporal non-maximum suppression; the window is stored in `model.json`). `decide_options` stays the
   one credit gate for the detector and the scorer driver.
8a. **The measured population.** The bar's "right" and "false first" rows are measured on the track-supported pairs:
   PFF's receiver is within 1.5 m of the ball at a ball frame within ±τ of the reception. The thresholds and the
   release-timing row are unchanged, and every report states the all-pairs measures and the supported share beside
   them. The population was chosen after the development scores were known and is fixed before the single held-out
   look. The supported "right" is conditional on the truth and the tracks agreeing; it is not an accuracy for
   receptions in general.

## Why the no-hard-drop rule changes (ADR-124 Notes 5)

ADR-124 kept every candidate (ranked, never dropped) because a dropped real touch would be unrecoverable by the
deferred Needleman-Wunsch event-alignment stage. That stage is still deferred. The shipped consumers read the
first contact after an action, where a false touch is exactly the harm the evidence measured. v2 therefore applies
the tuned threshold by default; a caller that wants every candidate, such as a future alignment stage, sets
`min_velocity_change_ms=0`, which stays valid.

## Alternatives considered

| Option | Why not |
|---|---|
| Keep v1, drop only the 5 m fallback | Removes about two thirds of the false touches and about half of the real ones (detector spec 3.2) |
| A learned scorer first | Trained on one provider, unvalidated on SkillCorner and IDSSE; kept as the escalation if the rules missed the bar (owner ruling), which they did (amendment) |
| A tree model as the scorer (first fit) | A second runtime path for a first fit; the logistic scorer came first. It missed the bar, and a tree model on the same features gained only 0.05 "right", so Amendment 2 adds the control features and the decoding with it |
| The bar on all pairs (Amendment 2) | The tracks support PFF's receiver at the ball on 58.6 % of pairs, so 0.60 over all pairs sits at the track ceiling; the all-pairs figures stay in every report |
| A wider velocity window (Amendment 2) | 0.067–0.267 s gave all-pairs "right" 0.417–0.392 at false first ≤ 0.10: not the lever |
| Tune on all 62 matches | No unbiased estimate of the chosen thresholds; the owner chose the development / held-out split |

## Evidence

Commit 5 adds the numbers: the rule baseline, the scorer's out-of-fold development scores and chosen threshold, the
single held-out score against the bar, and the IDSSE and SkillCorner reports against the commit-3 v1 reference
(`docs/research/contact_scorer/`).

## Consequences

- Every contact changes for every provider; TF-46 first-contact winners, the cross path end and `first_contact_*`
  move. A positional or removed-field `BallContactParams` call raises `TypeError` (named in the release notice).
- The corner models fitted on contact-timed targets (`RunZoneModel`, `CornerRoutineModel`) are classified owed or
  unaffected in `docs/research/cross-definition/caller-sweep.md`.
- `identify_corners` emits `taker_player_id`, and `add_corner_windows` refuses a corners table without it (a table
  saved before this release is rebuilt with `identify_corners`).
- ADR-125's accepted edge (a terminal dead-stop is not detected) narrows: a rest that outlasts the window keeps its
  samples, so the stop shows as a fast-in / slow-out change.
