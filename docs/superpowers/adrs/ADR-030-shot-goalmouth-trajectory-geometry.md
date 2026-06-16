# ADR-030: TF-48 post-shot goalmouth crossing geometry (`add_shot_goalmouth`)

| Field | Value |
|---|---|
| **Date** | 2026-06-10 |
| **Status** | Accepted |
| **Deciders** | Karsten (with Claude); lakehouse d32 session (4 cross-session review rounds) |

## Context

The lakehouse GK Analytics redesign needs **PSxG (post-shot xG)** for the tracking providers
(Gradient Sports WC2022 x64, SkillCorner x10, IDSSE x7). PSxG requires the goalmouth crossing
coordinates (y, z) of on-target shots, which the tracking providers' event feeds do not carry —
but the ball trajectory in the tracking frames can recover them, and adds kinematics (true ball
speed, time-to-goal-line) event data can never provide. The lakehouse scores the output with its
existing StatsBomb-trained PSxG model (Butcher et al. 2025 xGOT logistic); silly-kicks ships pure
geometry, no model, no ADR-011 lifecycle.

Spec: `docs/superpowers/specs/2026-06-10-shot-goalmouth-psxg-design.md` (converged across both
sessions). Plan: `docs/superpowers/plans/2026-06-10-tf48-shot-goalmouth.md`.

## Decision

New `tracking/_shot_goalmouth.py` kernel + `add_shot_goalmouth` edge in `features.py`
(ADR-025-style engine/edge split; the engine is PURE — no warnings, never mutates inputs) +
atomic mirror. Key sub-decisions, each load-bearing:

1. **Orientation-agnostic engine.** Goal ends come from the GK map (`defended_goal_x`, extracted
   byte-identically from `_xshot_occurrence` into `_gk_resolve.py`; xS re-imports via shim);
   output is canonicalized to attacked-goal-at-x=105 (full point reflection `x->105-x, y->68-y`
   to preserve handedness). The engine never reads `team_attacking_direction` — the same
   synthetic match in opposite global conventions produces byte-identical canonical output
   (test-pinned). Three-state goal-end resolution: resolved / degenerate (PSO: both teams'
   GKs classified to one end -> fallback to the end nearer the window's mean ball x) /
   unresolved (NaN action team etc.). Probe-backed: the GS WC2022 feed carries NO period-5
   events or frames (all four shootout matches probed 2026-06-10), so the PSO path ships
   synthetic-fixture-pinned only.
2. **Fit-window policy.** Segment ends at the FIRST of observed plane straddle / trajectory
   break (residual jump, consecutive speed drop, reversal vs the shot's OWN initial direction —
   not vs the goal, so own-goals/mishits fall through to `no_crossing`) / window cap
   (`data_end` when the slice itself ran out). Recorded per row in `shot_fit_end_reason`.
3. **t0 anchoring** (pilot-evolved; see "Pilot-driven kernel hardening" below): PRIMARY = the
   plane-approach flight run ending nearest the plane, re-anchored at the LAST run sample within
   `_CONTACT_RADIUS_M` (2 m) of the shooter's own action coordinates (the GS event x/y is an
   exact ball-track point — median shooter-to-nearest-sample distance 0.0 m; a cross + header is
   ONE continuous approach run, so run selection alone fits the assist). FALLBACK = FIRST
   shot-consistent discontinuity (speed increase toward the attacked goal) within an asymmetric
   window. Largest-discontinuity selection rejected: a close-range save inside the window would
   win and the fit would lock onto the post-save track (fixture-pinned).
4. **z-profile taxonomy** (`shot_z_profile`): `rolling` (all z <= `rolling_z_max_m`; crossing z =
   segment mean), `airborne` (fixed-g ballistic, 2 free params — robust at SkillCorner's 2-4
   samples where a free quadratic overfits), `bounced` (vz sign flip near ground with
   drop-AND-rise >= `bounce_min_dz_m` hysteresis; a noisy airborne vz flip at height stays
   airborne). The crossing comes from the LATEST sub-segment before the plane — the trajectory
   the GK actually faces. **Branch-3 unreachability proof:** a detected vz flip at k requires
   z[k-1], z[k], z[k+1] all finite (np.diff over NaN yields NaN; NaN comparisons are False), so
   a detected bounce always leaves >= 2 finite sub-segment samples; the `f.sum() >= 2` guard is
   retained defensively (it also serves the airborne sparse-z path).
5. **Per-column segment provenance** (cross-session M-1): `shot_speed` ALWAYS from the EARLIEST
   (contact) sub-segment — it answers "how hard was it hit" and is never superseded;
   `shot_crossing_y/z` + `shot_fit_n_frames` + `shot_fit_rmse` from the segment that PRODUCED
   the crossing (post-bounce when superseded); `shot_time_to_goal_line` is elapsed real time
   from refined contact to the crossing (spans the bounce). `no_crossing` rows still carry the
   attempted fit's diagnostics (NA is reserved for truly-unfitted sources — R4b).
6. **NO VAEP xfns factory** [owner-decided 2026-06-10]: every output is determined by what
   happened after ball contact; as a gamestate feature it encodes the shot's outcome — result
   leakage of the HybridVAEP class. Guard test auto-discovers every default xfn list and
   asserts absence (`tests/tracking/test_shot_goalmouth_no_xfns_guard.py`).
7. **Atomic mirror included** [owner-decided]: thin delegation with
   `shot_type_ids=_ATOMIC_SHOT_TYPE_IDS` ({shot, shot_penalty}; `shot_freekick` is a `freekick`
   atom — existing pre-shot-GK precedent). NO coordinate synthesis (no end=x+dx): the engine
   consumes action_id/game_id/period_id/time_seconds/team_id/type_id plus the action's OWN
   location (`start_x`/`start_y`; atomic `x`/`y`) as the OPTIONAL contact anchor, reflected into
   frame coords via goal_x (orientation-agnosticism preserved; NaN -> un-anchored, ADR-003).
8. **Pointer resolution = the `add_xt_gk`-verbatim path**, NOT `_resolve_action_frame_context`
   (rejected cross-session review point M1/R2: the context helper additionally builds
   actor/opponent row joins TF-48 never consumes). The engine never reads `links` (the window
   is time-sliced via `slice_around_event`); the edge uses it only for the provenance merge.
9. **`shot_on_target_derived`**: ball-center within posts/bar expanded by
   `on_target_tolerance_m` (ball radius). Post/bar physical width is INTENTIONALLY folded into
   the tolerance, not modeled. NA when crossing z is unavailable or source is not
   observed/extrapolated. Provider `result_id` is a validation cross-check, never an input.
10. **Own goals intentionally excluded** (trajectory points away from the attacked plane ->
    `no_crossing`): PSxG-faced-by-the-opposing-GK semantics do not apply; fixture-pinned.
11. **Confidence map** (`shot_crossing_confidence`): PROVISIONAL except one choice — observed
    (1.0) STRICTLY dominates any extrapolated score (capped 0.9). Inputs include z_profile +
    producing-segment size (an exactly-determined 2-sample z refit has RMSE == 0 and would
    otherwise out-score an honest 5-point fit). Calibrated at the SB pilot.
12. **Edge warning policy**: `add_shot_goalmouth` warns when > 50% of shot rows are
    `no_ball_frames`/`unresolved` (data-quality signal, mirrors `on_low_coverage`).

**Units contract:** SPADL meters, goal centre y=34, mouth 7.32 m, bar 2.44 m. The
meters->StatsBomb-normalized mapping is owned by the lakehouse at scoring time; StatsBomb y is
INVERTED vs SPADL y (`spadl_y = 68 - sb_y*68/80`, `silly_kicks/spadl/statsbomb.py`), so the
indicative mapping is `y_sb = 40 - (y_m - 34)/0.9144`, `z_sb = z_m/0.9144` — and the validation
harness settles handedness empirically on goals (agreement >= 0.8 gate; relaxed from 0.9 at floor
registration — see Validation) BEFORE any floor.

## Validation (spec section 10)

GS WC2022 = the same 64 matches as StatsBomb open data. Owner-run harness
`scripts/validate_shot_goalmouth_sb.py` (pining loaders, DGX): outcome-literal runtime assert;
documented tie-breaker matching with ambiguous->unmatched; handedness settlement before floors;
**GOALS-vs-SAVES stratification** (SB save end_locations are save-points, not plane crossings —
floors run on GOALS only; the Delta(saves)-Delta(goals) split quantifies the lakehouse PSxG
train/serve shift for free); per-frame-rate sensitivity sweep incl. the module constants outside
the params surface; raw-z vs smoothed-z (`ballsSmoothed`) comparison whose conclusion is relayed
cross-session (the lakehouse AC adapter currently feeds raw GS z).

**Pre-registered accept floors (registered 2026-06-11 at pilot-v5 review, BEFORE the held-out
run; owner-approved):**

| Floor | Value | Pilot-v5 measured |
|---|---|---|
| goals median \|dy\| (m) | <= 2.5 | 1.21 |
| goals median \|dz\| (m) | <= 1.25 | 0.52 |
| on-target agreement | >= 0.45 | 0.61 |
| resolution coverage (observed+extrapolated among matched on-target) | >= 0.60 | 0.795 |

Floors are regression TRIPWIRES with ~2x headroom over the pilot-measured values (the
default-stability discipline: corpus drift and provider noise move the numbers; every structural
failure mode found during the pilot loop — phantom breaks, assist-fit, orientation bugs — blows
through these floors by multiples). Pilot-measured values are recorded alongside so silent drift
toward a floor is visible before it fails. The held-out e2e
(`tests/tracking/test_shot_goalmouth_sb_e2e.py`) skips LOUDLY until the floors are registered
here; a floor failure is a hard STOP (no silent threshold adjustment).

**One-shot holdout protocol:** the pilot absorbed five tuning cycles, so every kernel constant is
pilot-fitted; the holdout (the 48 non-pilot GS matches) is the single uncontaminated measurement
of generalization. It runs ONCE, against the kernel as registered; the outcome is either "ship"
or "documented failure analysis + a NEW holdout protocol" — never tune-and-re-run. Future kernel
changes re-validate on the FULL corpus with these floors as the gate, acknowledging the original
holdout is spent as a held-out set.

**ROUND 2 (2026-06-11, owner-approved "A + B and round 2"):** two changes registered BEFORE the
round-2 holdout, both motivated by the round-1 failure analysis below:

- **(A) Handedness instrument REBUILT on GK geometry.** The round-1 gate compared near-centre
  goal-mouth ball tags (derived crossing vs SB hand-coded `end_location`) — measured too noisy to
  settle a transform: in-mouth-filtered agreement 0.75–0.79 on BOTH pilot and holdout-r1 (16-5 /
  17-5 majorities; a wrong sign reads ~0.2), with several dissenters agreeing with SB EXACTLY
  under the opposite mapping. The round-2 gate votes per MATCHED SHOT (not per goal) on the SB
  shot freeze-frame DEFENDING GK's y vs the GS-tracked GK's y (canonicalized via the engine's
  goal_x reflection): one well-identified object, fit-independent, ~10× the voter pool. The
  ball-tag vote is demoted to the informational `handedness_ball_diag` with an in-mouth
  plausibility split (a GOAL whose derived crossing is outside the mouth is a self-evident
  measurement failure — 7/21 pilot, 10/22 holdout-r1 ball votes). Floor stays 0.8.
- **(B) Kernel: contact EXISTENCE + playable-height contactability** (the worst measured P2
  class, a 12.6 m "observed" goal: a cross passing 6 m OVERHEAD of the shooter then crossing the
  extended plane 9.5 m wide — behind the goal — was fitted as the shot). A contactable sample =
  2-D near the stamped shot location AND at playable height (z <= `_CONTACT_MAX_Z_M` 2.6 m). No
  contactable sample within `_CONTACT_EXIST_RADIUS_M` (5 m) anywhere in the window -> the window
  provably does not contain the shot -> `insufficient_frames` (honest no-fit, never a wrong
  crossing); the fit start is clamped to the LAST contactable sample within `_CONTACT_RADIUS_M`
  (2 m; generalizes the v3 run-vicinity anchor); NaN stamp -> un-anchored fit (ADR-003).

The registered floors are UNCHANGED. Honest context recorded: they were calibrated on the
P1-only pilot (the round-1 instrument bug below); the full-sample v6 re-baseline measured goals
dy median 2.10 m vs the 2.5 floor (1.19× headroom, down from the 2× design) — (B) targets exactly
the P2 junk that widened it. **Round-2 pilot (v7) reference values:** GK-instrument handedness
0.883 on 205 voters (181–24 flip); goals dy median 2.08 / p90 4.66 / dz median 0.53; saves dy
median 2.65; coverage 0.663 (the contact-existence bar costs ~0.09 coverage in exchange for
never fitting a window that provably lacks the shot — the thinnest floor margin, 1.10×);
on-target agreement 0.635. The measured 12.6 m overhead-arc goal resolves `insufficient_frames`;
the chip-curl extrapolation (5.4 m, `on_target_derived=False`) remains the documented residual
kernel limitation.

**HOLDOUT ROUND 2 (2026-06-11): PASS — ALL FOUR FLOORS + THE HANDEDNESS GATE.** One run, kernel
and floors as registered: GK-instrument handedness **0.882** on 646 voters (570–76 flip; pilot
0.883 — instrument-stable); goals dy median **2.166 m** (floor 2.5; p90 5.75), dz median
**0.482 m** (floor 1.25); on-target agreement **0.600** (floor 0.45); coverage **0.620** (floor
0.60 — the thin margin predicted at registration; the contact-existence bar trades coverage for
never fitting a window that provably lacks the shot). 999/1125 GS shots matched (126 unmatched);
goals n=78, saves n=142 (saves dy median 2.04 — the goals/saves Δ collapsed once save-point
semantics were measured on the full corpus). Verdict per the one-shot protocol: **SHIP**.

**HOLDOUT ROUND 1 (2026-06-11): ABORTED at the handedness gate — DOCUMENTED FAILURE ANALYSIS.**
The run completed all 49 matches and aborted at agreement 0.773 (17–5 flip on 22 clearly-sided
goals) < 0.8, before floors were evaluated. Analysis (per-goal votes + debug capture):
(a) the SIGN is re-confirmed (a wrong transform reads ~0.23, the mirror); (b) the 5 dissenters
scatter across 5 different matches, all `observed`-source airborne crossings — no match/period
clustering; (c) **the analysis exposed that ALL 22 voters were period 1 — the harness's GS↔SB
shot matching had silently dropped period 2 since the FIRST pilot** (417/421 matched shots P1;
681/704 unmatched `no_candidate`). Root cause: GS SPADL `time_seconds` is the CUMULATIVE match
clock (P2 = 2700+rel; the known "GS time-base guarded lakehouse-side" convention — actions and
frames agree, so the KERNEL was never affected), while the harness converted SB to
period-relative — every GS P2 shot sat exactly 2700 s from its SB counterpart (probe: 5647 ↔
2947+2700, to the second). The 4 "matched" P2 shots were SPURIOUS cross-matches (late-SB-P2
stoppage ↔ early-GS-P2). Consequence: every pilot metric (v1–v5) and the registered floors were
measured on a HALF-BLIND instrument (P1 goals only). Two harness fixes (instrument repair, kernel
untouched): the strict abort now writes the full report BEFORE raising ("abort with a report" was
always the contract — round 1 destroyed its own failure-analysis capture), and the SB clock is
used CUMULATIVE, matching the GS base directly. **New protocol: re-baseline the pilot with the
fixed instrument (the pilot is the tuning set — re-running it is legitimate), re-confirm or
re-register the floors with the owner on full-sample data, then ONE new holdout run. Holdout
round 1 is spent; its abort is this record.**

**Handedness gate relaxed 0.9 -> 0.8 (registered with the floors).** The gate exists to settle
the y-axis SIGN of the meters->SB transform, and its two hypotheses are cleanly separable: a
wrong sign produces agreement near 0.1 (the mirror image), frontier noise produces 0.8-0.95. The
sign is settled (pilot 8-1 flip, agreeing with the in-repo converter-derived prior); the one
dissenter is a MEASURED GS-vs-SB hand-tag disagreement class (sides flipped on near-centre
crossings where GS's own in-net ball samples corroborate GS), not transform ambiguity. A 0.9
floor conflates the two hypotheses and would abort the one-shot holdout on a predictable false
positive; 0.8 still discriminates perfectly. The per-goal vote breakdown (`handedness_diag`)
stays in every report so a future genuinely-ambiguous corpus (0.5-0.7 band) remains visible. Do
NOT relax below 0.8 — the gate stops separating its hypotheses.

**Fixed en route:** the pining GS loader (`scripts/_loader_pining.py`) hardcoded `z=0.0` for ALL
frame rows and never read the raw ball records' `z` — every GS frame downstream of the loader had
flat zero ball z (probe 2026-06-10: z is present on 100% of raw GS ball records, range
-0.76..12.42 m). Fixed for ball rows (players keep 0.0 — no z in GS player records). Any prior
loader-fed analysis that consumed GS ball z (e.g. xS ball-z features at GS inference) saw zeros;
flagged to the owner.

### Pilot-driven kernel hardening (2026-06-11, five evidence-first cycles v1->v5)

Each fix was diagnosed from the harness's `--debug-shots` artifact (the tracer WRAPS the real
kernel functions during the enrichment call — recorded internals are byte-identical to what
produced the reported numbers), reproduced as a red fixture first, then validated on a fresh
pilot run. Pilot trajectory: goals median |dy| 1.99 m (plateau) -> 1.21 m; handedness 6-2 (0.75)
-> 8-1 (0.889); coverage 0.769 -> 0.795; on-target agreement 0.50 -> 0.61.

1. **Sample-and-hold collapse** (`_collapse_held_samples`): GS's raw `balls` channel delivers
   ~15 Hz positions duplicated at 29.97 Hz stamps (50% exact consecutive-duplicate x/y/z on ALL
   127 pilot windows; raw-artifact-confirmed channel property — `ballsSmoothed` is true ~30 Hz
   but x/y-divergent, agree@5cm ~0.002). A held duplicate is a phantom zero-velocity sample:
   0.1 s baselines phase-oscillate (a 5.5 m/s carry reads ~3.7/~7.3 m/s — leaking the 7 m/s
   flight gate) and LS fits pick up a ~0.3-0.4 m sawtooth (phantom trajectory-breaks at 3-6 real
   positions — the plateau's worst-goals signature). Exact-equality collapse to the FIRST stamp.
2. **Contact anchor** (Decision point 3): assist-cross/dribble + shot form one approach run;
   re-anchor at the last sample within 2 m of the shooter. Measured wins: header goal 7.33 ->
   3.52 m (observed), two goals to 0.03/0.15 m.
3. **z-aware flight + LOCAL residual break**: a real chip goal decelerated 13->3 m/s horizontally
   while climbing to z=2.7 m — x-only logic speed-drop-broke it and trimmed the core to 2
   samples. Airborne (finite z above the rolling band) counts as flight in the trim; the
   speed-drop break is skipped at airborne checkpoints; the residual break compares vs a fit on
   the LAST `_LOCAL_FIT_WINDOW_S` (0.4 s), not the segment-anchored fit (which phantom-breaks ANY
   smoothly curving chip/curl ~1 s in). A deflection violates even the local fit -> save
   semantics preserved; segments < 0.4 s byte-identical to the anchored check.
4. **Extrapolation-leverage cap** (`max_extrapolation_leverage = 3.0`, params surface): t* beyond
   3x the producing segment's evidence span is a guess, not a fit (v4-measured: leverage > 3 had
   dy median 6.22 m, max 41 m — all saves, zero goals; below, 2.35 m). Joins the t* bounds family
   -> `no_crossing`.
5. **Post-trim bookkeeping re-anchor**: `n_fit_frames` reported the PRE-trim grow end (artifact
   showed n=19 on a 12-sample core); the bounce-supersession arithmetic shared the offset.

**Measured non-fixable frontier (bounds any kernel vs SB):** observed-straddle crossings
(interpolated directly from tracking data) disagreeing with SB's hand-tagged `end_location` by
2-5 m with SIDES flipped on near-centre crossings — in the clearest case GS tracks a curling shot
crossing near-centre AND follows the ball into the net (x < 0 samples corroborate GS). The goals
|dy| tail (p90 ~4.4 m) is dominated by this class; crossing-z disagreement is bidirectional
(GS z-channel onset lag).

### Completion cycle (4.28.0) — two refinements + one investigated-and-rejected guard

Re-validated on the FULL 64-match GS corpus (post-holdout protocol; floors are the regression
gate). **(1) Curve-aware y extrapolation IMPLEMENTED** (`_extrapolate_crossing_y`): the
constant-velocity fit extrapolated a curling/dipping flight's crossing LINEARLY (the residual curl
limitation noted in earlier cycles); now a span-gated quadratic-y is used when the producing
segment supports a curvature estimate AND the quadratic markedly out-fits the line (real curl, not
jitter), capped tighter than the linear leverage. **(2) Earliest-reaching flight-run tie-break**
(`_find_flight_run`): among plane-approach runs that REACH the goal line, the SHOT is the EARLIEST
— the bare nearest-plane rule had anchored t0 PAST a real in-mouth crossing on a measured holdout
goal. Full-corpus result: goals |dy| median 2.17 -> **2.08**, all floors pass, no regression.
**(3) Over-bar wide-straddle re-selection guard — INVESTIGATED OFFLINE AND REJECTED.** A guard
that, for an observed crossing ABOVE the crossbar, prefers a later low in-mouth crossing was
re-scored on the full corpus: it fixes 2 over-bar GOALS (dy 8.7->4.4, 5.6->1.5) but BREAKS 3 SAVES
(one clean save 0.34->3.55 m), because the "above-bar then later low crossing" signature is
inference-indistinguishable between a goal dipping in and a save's REBOUND — the kernel has no SB
tag at serve. Net aggregate is a wash (goals median 2.085->1.990; saves 2.360->2.420; on-target
agreement unchanged), so it trades fixed goals for new wrong outputs on currently-correct saves.
These ~2 over-bar goals stay documented provider frontier (floors pass; lakehouse gates on
`shot_crossing_confidence`). Do NOT re-add the guard without an inference-time goal/save
discriminator.

## Consequences

- Additive only: no canonical column changes, no retrain trigger (NOT in any default xfn list;
  guard-tested). C4 action-coupled-aggregator count 27 -> 28.
- Lakehouse items (tracked their side): SkillCorner z plumbing fix (`convert.py:463`), the
  meters->SB transform + scoring, AC columns/migration, conditional GS z-source switch pending
  the pilot's raw-vs-smoothed conclusion.
- Attribution: Anzer & Bauer (2021) cited contextually in docstrings (already in NOTICE via
  `pre_shot_gk_*`); no new NOTICE entry (pure geometry).
- Deferred: drag-aware horizontal fit (Spearman 2017 constants already in-house) — only if the
  SB validation shows speed-dependent bias; ELASTIC-based contact alignment upgrade — only if
  the pilot shows residual sync bias.
