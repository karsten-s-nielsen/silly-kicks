# Soccermatics Pro + Modern Soccer Coach courses → silly-kicks future-work plan (2026-07-16)

> **What this is.** A full, independent ingestion of TWO course corpora, cross-referenced against
> **silly-kicks 4.48.1 (main @ 3930f03)**:
> 1. **Soccermatics Pro** (`D:\[Karsten]\Dropbox\[Coaching]\Twelve Football\Soccermatics Pro`,
>    all 17 modules — every `.srt`, `.rtf`, PDF, notebook, script), superseding the prior
>    analysis-session handoff (`2026-07-16-handoff-soccermatics-tracking-metrics.md`), whose
>    capability map was checked claim by claim — corrections in §2.
> 2. **Modern Soccer Coach** (`D:\[Karsten]\Dropbox\[Coaching]\Modern Soccer Coach` — the
>    Advanced Tactical Analysis & Coaching Methodology course, the 4-webinar Football Analysis
>    Bootcamp, standalone webinars, and all PDFs/PPTX/DOCX/XLSX; transcribed 2026-07-16). Mostly
>    coaching content, but with three high-value practitioner sources: the Packing Data lesson
>    (folded into W5), the Bootcamp Webinar 3 Tigres Femenil report system + Clemson pressing
>    study (folded into T1/W4), and rotation/reception taxonomies (folded into W1). Still
>    untranscribed: the `Webinars/` folder (5 videos — **Michele Aragona, "Scientific and
>    Holistic Approach to Set-Pieces" is TF-46-relevant, recommend transcribing**) and the 20
>    scanning drill videos (training content, low value).
>
> **No code was changed; this is a planning/TODO-capture document.** Every §4 entry is written
> to TODO-ready depth: definition with exact constants, verified repo deltas, design decisions
> to settle at spec time, validation plan, and flags (C4 / retrain / coordination / attribution).

Method: 9 + 4 module-cluster extraction agents (exhaustive per-lesson metric/method/threshold
extraction) + 5 repo-verification agents (each handoff claim checked against real source with
file:line evidence) + targeted web research (packing canon) + first-hand reads of the two
packing transcripts, synthesized by the session with full silly-kicks context. Raw extractions:
session scratchpad `soccermatics/` (C1–C9, verify_1–5) + `msc/` (M1, M2, B1, R1).

---

## 1. Headline

The prior handoff's core finding **holds**: silly-kicks already implements most of the tracking
curriculum (modules 14–15 fully, much of 16), frequently beyond what the course teaches. But the
full-course sweep (modules 1–11, which the handoff did not prioritize) plus the repo verification
surfaced **more than the two gaps it named** — including one defect-shaped finding (OBSO's EPV
factor is a synthetic placeholder, never real xT) and a direct unlock for a tracked TODO item
(TF-35 now has a practitioner anchor). Ranked plan in §4.

## 2. Corrections to the prior handoff (verified against source)

1. **"Packing ≈ the existing line-breaking detector" is WRONG.** `_line_breaking.py` counts
   *lines/clusters* broken (`lines_broken__ward`, Int64 0–3), and the threshold kernel's
   `n_attackers_behind_line` counts the acting team's own players — the inverse subject from
   packing. The genuine near-match is **`structural_lbs`** (`_structural_pass.py:78`): a per-pass
   count of individual opponents with `start_x < d_x ≤ end_x` — structurally the Impect packing
   definition, with a self-documented 1-D caveat (no lateral/lane restriction; a far-touchline
   defender still counts). Verdict: packing is **partially present via LBS**, absent as a
   named, attributed, canon-faithful metric. See W5 (elevated from Tier 3 on owner interest).
2. **The off-ball-runs docstring quote was misattributed.** The "no sprint-intensity filtering"
   disclaimer lives in `features.py:799` (TF-3 `actor_arc_length_pre_window`), not in
   `_off_ball_runs.py`. The substantive claim stands: the detector is displacement-only
   (≥3.0 m net displacement over a 1.5 s pre-window, sign-test direction), no typology, no
   valuation, and its `mean_off_ball_run_speed_pre_window` is displacement/window, not true speed.
3. **"OBSO is the module-15 fusion" needs an asterisk.** `_obso.py` implements Spearman (2018)
   verbatim (`PPCF × Transition × EPV`) — but the **EPV factor defaults to a synthetic linear
   x-ramp** (`np.linspace(0.01, 0.3, nx)` tiled; `_obso.py::_make_synthetic_epv_grid`) and **no
   call path anywhere wires a real `ExpectedThreat` grid into it** (`epv_grid` is a raw
   `np.ndarray | None` through `add_obso`/`obso_xfns`/`_space_creation`). The genuine PC×xT
   fusion exists only in `_player_influence.py` (required injected fitted `ExpectedThreat`,
   `Σ PC_player(z)·xT(z)·cell_area`). See W3.
4. **Physical-metrics absence: confirmed exactly, plus one new fact.** Nothing landed in
   4.47.0–4.48.1. `derive_velocities` is literally one `deriv=2` short (`_velocity.py:119` uses
   `savgol_filter(..., deriv=1)` only). New since the handoff: ADR-038's corpus pass documented
   SkillCorner's **`physical.parquet`** product (29×36 per-player-per-match: total/running/HSR/
   sprint/hi distance + counts, med/high accel/decel event counts, explosive-accel-to-HSR/sprint
   + time-to-threshold, `psv99`, `physical_check_passed`) in
   `docs/research/skillcorner_corpus/schema_new_physical.txt` — **available but consumed nowhere**.
5. **Module 14 (formations/compactness) is fully covered** — including the one open question:
   `compute_team_shape` does emit `convex_hull_area` (`_team_shape.py:158`, scipy `ConvexHull`).
6. **The Spearman-parameter diff the course invites is already satisfied.**
   `SpearmanParams` defaults (`pitch_control/_params.py`): reaction 0.7 s, max_acceleration
   7.0 m/s², sigma 0.45 s, average_ball_speed 15 m/s, lambda_gk 3.0 — matching the canonical
   Spearman/Shaw constants taught in lessons 15.2a/2b (the course additionally names λ=4.3 s⁻¹
   control rate and player max speed 5 m/s for Shaw's constant-velocity variant; silly-kicks
   deliberately uses acceleration-based TTI per the documented lineage note). No action needed
   beyond noting the external confirmation.
7. **The handoff's ranking inverted the fit.** Its #1 (physical) is the *biggest* gap but a
   *genre decision*; its #2 (off-ball runs) is now the better-grounded lead because TF-35 is an
   already-tracked TODO item whose blocking objection ("no published academic anchor") the course
   materially softens.

## 3. Repo state relevant to scope (verified @ 4.48.1)

- **TF-35** (TODO, Research & Future Work): "Off-ball run type classification — overlap /
  underlap / far side / advance / support taxonomy. Priority-deferred; no published academic
  anchor." SkillCorner's 10-type run taxonomy (`associated_off_ball_run_subtype`) is present in
  `dynamic_events.csv` and **read-and-discarded** by `spadl/skillcorner.py` (tallied in
  `excluded_counts`, never mapped).
- **TF-46** (set-piece corner CDHMM) and **TF-47** (conditional/KNN xT) are the tracked items
  adjacent to the course's set-piece and tracking-context-xT themes. Deep-learning trajectory
  models are explicitly out of scope (TF-46 rejects the TacticAI/GNN line).
- **No tracked items** exist for: physical/locomotor, packing, possession-retention/regain team
  KPIs, player aging curves, season repeatability, match-outcome simulation.
- **ADR-038 registered protocol** governs anything touching SkillCorner training data or
  xS/xCross/ghost-GK retrains. **Part-deux session currently owns** `spadl/skillcorner.py`,
  `tracking/{skillcorner,metrica,_ghost_gk}.py`, `scripts/_loader_pining.py` + corpus helpers,
  and the three trainers (DGX Stage B in flight). Coordination flags are noted per item below.

---

## 4. Candidate work — TODO-ready entries

### Tier 1 — recommended builds (each still needs its own brainstorm/spec cycle before code)

#### W1 · TF-35 unlock: typed + valued off-ball runs (module 16.2)

**Definition (Sumpter/Twelve, production club method, module 16.2 — with worked numbers):**
- **Target run** — credit the receiving runner with **MAX(pitch_control × xT)** over the space
  the run opened, *not* the realized value of the received pass (the receiver decelerating on
  the ball lowers realized value; the potential of the opened space is the skill). Worked
  example: Rashford, max value of opened space 0.07.
- **Disruptive run** — a **sprint coincident with a pass event** where the runner does **not**
  receive; credited by the PC×xT value of the space the run opened (worked example: Luke Shaw,
  0.11). Crediting by measured defender displacement was considered and **explicitly rejected**
  by the course — defenders must react because the pass *could* go there.
- **Space-creation credit** — when the phase's pass to a *different* (target-run) player
  succeeds, the disruptive runners in that phase are additionally credited with the enabled
  pass's value. One phase can carry multiple simultaneous credits.
- Composite player radar over the three (Firmino = standout on all three despite modest
  minutes; Ibrahimović = mostly self-serving target runs — the course's face-validity anecdote).

**Repo state (verified):** TF-4 detector (`_off_ball_runs.py`) is displacement-only
(`pre_seconds=1.5`, `min_displacement_m=3.0`, sign-test direction, dead-ball → NaN); no typing,
no valuation, no run-level identity. Primitives for the valuation all exist: `PitchControlCache`
(+ `decompose=True` per-player surfaces, already consumed by `player_influence`), injected
fitted `ExpectedThreat` (the `player_influence`/SK-xT-2 pattern), `link_actions_to_frames`
anchoring. Note `_space_creation.py` is a DIFFERENT construct (LOO counterfactual m²) — the
course's method is surface-max valuation, simpler and complementary.

**Design decisions for the spec:**
1. **Run detection upgrade** — add a sprint/intensity gate to the detector (per-frame `speed`
   ≥ threshold sustained ≥ T; thresholds ideally shared with W2's band constants — v1 can use a
   single threshold on the existing `speed` column without W2).
2. **"Space opened" operationalization** — the course never formalizes it. Candidates:
   (a) region where the runner's per-player PC surface exceeds a floor (reuse
   `decompose=True`); (b) disc of radius r around the run endpoint; (c) cells where team PC
   increased over the run window. Pick ONE in spec, parameterize.
3. **Three candidate taxonomies** — the course's *value-role* taxonomy (target/disruptive/
   space-creation) vs TF-35's original *geometric* taxonomy (overlap/underlap/far-side/advance/
   support) vs the MSC rotations lesson's *rotation-archetype* taxonomy (LOW: #6-drops-to-back-3
   variants, FB-inverts-into-midfield; MID: FB-high/winger-drops interchange, wide three-player
   relay; LAST-LINE: 5-attacker pin + a coordinated drop-in-front/dart-behind two-player relay).
   They compose (a disruptive run can be an underlap inside a last-line rotation). Spec should
   decide whether v1 ships value-roles only (recommended — it is what the valuation needs) with
   geometry/rotations later.
4. **Output shape** — house pattern suggests: pure primitive returning a long-form per-run
   table (`game_id, action_id, player_id, run_type, run_value, ...`) + an `add_*` aggregator
   emitting per-action columns (n_runs by type, max/sum run value) + optional `*_xfns`.
5. **Receiver resolution** — SPADL has no `receiver_player_id`; use the established
   next-same-team-touch pattern (same as W5).
6. **SkillCorner's native 10-type taxonomy** — use as **external validation labels**
   (ADR-007-style tiering), not as the source. Ingesting it requires `spadl/skillcorner.py`
   changes that are **part-deux-owned right now** — defer that sub-part or hand it off. Also
   note: the kloppy gateway hardcodes `visibility: None` (memory), so any run-detection QA on
   SkillCorner should use the native 4.48.0 route.
7. **Reception-quality labels (MSC Tigres definitions — candidate companion features):**
   "pockets" = receiving between the opponent's back and midfield lines **facing forward with
   time and space** ("not just a player with back to goal — then they're not able to progress");
   "spacing behind" = any player beyond the opponent back line (deliberately relaxed — beyond
   the fullback with covering CBs still counts). The facing-forward criterion needs body
   orientation, which the tracking schema does not carry — velocity direction at reception is
   the honest proxy (document the limitation). Same "front-foot" idea as W5's
   `secured_reception`; one shared operationalization should serve both.

**Validation plan:** mirror-invariance (ADR-028 gate pattern) + a ground-truth asymmetric
fixture (runner opens space at a known cell → known max value; per
`feedback_symmetry_test_insufficient_pin_ground_truth`); liveness + purity + dup-action-id +
id-dtype auto-gates (registering the new `add_*` wires them); SkillCorner label agreement as an
owner-gated e2e once ingestion lands.

**Attribution / anchors:** Sumpter/Twelve (course module 16.2) — a practitioner anchor, weaker
than a paper; say so in the ADR. **Esposito et al. 2026** (IJSSC, DOI 10.1177/17479541261427153,
on disk) was RE-VERIFIED 2026-07-16 against the full text: a 4-axis conceptual taxonomy with
**no computable run-type geometry** — consistent with the 2026-06-05 audit already recorded in
TODO's TF-35 entry. It anchors the *framing*, not the build; the valuation arm's anchor is the
course itself. NOTICE entries for both.

**Flags:** tracking-required; new action-coupled aggregator → **C4 +1**; in no default xfn list
(additive, **no retrain**). Effort: medium-large (detection upgrade + valuation + typing).

#### W2 · Physical/locomotor metrics (module 13) — biggest clean gap, scope decision needed

**Definition (SkillCorner/sports-science vocabulary as taught, exact constants):**
- Speed bands: **running 15–20 km/h, HSR 20–25 km/h, sprint >25 km/h**; high-intensity =
  HSR + sprint (reported both ways because practitioners disagree on "high intensity").
- Distances per band + **effort counts** (an effort counts once sustained **≥1 s** above the
  band threshold). Total distance; **m/min** = distance / minutes played.
- **Acceleration/deceleration events**: 1.5–3.0 m/s² sustained **≥0.7 s** (sports-science
  grounded); mirrored for decelerations. SkillCorner additionally counts explosive-accel-to-HSR
  / -to-sprint events + time-to-threshold.
- **PSV-99**: 99th percentile of the pooled per-sprint top speeds — a deliberate robustness
  construct against tracking-error spikes (raw max speed is unreliable). "Top 5 PSV-99" for
  repeatable top-end. **>12 m/s ⇒ treat as tracking error** (course's hard cap).
- Units: prefer m/s internally (course argues m/s; SkillCorner native is km/h; ÷3.6).

**Repo state (verified):** absent entirely. `derive_velocities` (SG, `deriv=1`) emits vx/vy/
speed only — `deriv=2` gives ax/ay for free. `tracking/utils.py::_derive_speed` is the cruder
fallback (unsmoothed finite difference). `TRACKING_CONSTRAINTS["speed"]=(0,50)`. SkillCorner
`physical.parquet` documented but unconsumed (§2.4).

**Design decisions for the spec:**
1. **Scope ADR first** — this is athletic/load profiling, a genre expansion beyond on-ball
   valuation. Owner has ruled everything in-scope for TODO capture; the ADR still needs to say
   the mission expands (or that this ships as a clearly-labeled sidecar).
2. **Acceleration in preprocess** — extend `derive_velocities` (or a sibling) with `deriv=2`:
   emit `ax`, `ay`, `accel_magnitude` = **vector norm** `hypot(ax, ay)` — explicitly NOT a
   scalar speed-diff (the exact bug the course flags in widely-circulated reference code; the
   repo's only speed-diff is `_elastic_sync.py`'s private ball-sync signal, which is fine for
   its purpose but must never be surfaced as acceleration).
3. **Aggregation module** — new pure `tracking/physical.py` (or `_physical.py` + public
   wrapper): frames → per-(game, player) summary (optionally per period). NOT an action-coupled
   aggregator → **C4 aggregator count unchanged** (likely a new container in the diagram).
4. **Visibility gating** — compute only over frames where the player is genuinely detected:
   the native SkillCorner route's `is_detected` (ADR-038) is the gate; full-tracking providers
   (sportec/GS, 25 Hz) need none; **the kloppy gateway path is unusable for this** (hardcodes
   `visibility: None` — memory). Emit a coverage fraction per player so consumers can judge
   (total distance is the weakest metric under broadcast tracking — off-camera estimates are
   5–10 m off; PSV-99 is the most robust — course's own hedging, worth echoing in docstrings).
5. **Params dataclass** — band edges, bout minimums, accel thresholds, the 12 m/s cap, and the
   percentile (99) as a frozen dataclass with SkillCorner-glossary defaults; `for_provider`
   auto-promotion mirrors `PreprocessConfig` (mind the known `is_default()` gap —
   `project_preprocess_api_robustness_gaps`).
6. **Bout detection** — run-length encoding over thresholded speed/accel series; period
   boundaries break bouts; NaN (undetected) frames break bouts.

**Validation plan:** Tier-1 external ground truth = recompute the SkillCorner corpus and
compare against `physical.parquet` per player-match (respect `physical_check_passed`); agree
tolerances in spec (expect distance to disagree most, PSV-99 least — that ordering is itself
the sanity check). Unit fixtures: synthetic constant-speed / ramp trajectories with known
distances, bout counts, PSV. The PR-S86 roster-validation pattern applies.

**Flags:** additive, **no retrain** (not in any xfn list). Coordination: pure frames-consumer —
does not touch part-deux files; the corpus validation should wait for / coordinate with Stage B.
Effort: medium. Attribution: SkillCorner physical-data glossary
(skillcorner.crunch.help) + course module 13 in NOTICE.

#### W3 · Wire real xT into OBSO's EPV (and transition) factors — defect-shaped, cheap

**Finding (verified):** every OBSO / pass-OBSO / space-creation value ever served used the
synthetic `linspace(0.01, 0.3)` EPV ramp (and a Gaussian dummy transition) unless a caller
hand-built grids — none do, anywhere. `player_influence` already shows the correct pattern:
required injected fitted `ExpectedThreat`, `xt.interpolator(kind="linear")(pc.grid_x,
pc.grid_y)`, ADR-028-aware mirroring.

**Proposal (additive opt-in — NOT a default flip):**
1. Public pure adapter `epv_grid_from_xt(model, grid_x, grid_y)` (placement: `tracking/_obso.py`
   or a small `tracking/_xt_adapters.py`) — interpolate `ExpectedThreat` onto a PC grid;
   fail-loud on unfitted (`TYPE_CHECKING`-only import, duck-typed at runtime — the SK-xT-2
   precedent, no new runtime edge).
2. Optional `transition_grid_from_xt(model, ...)` — the fitted per-source-zone transition row
   of the ball's current zone, interpolated 16×12 → PC grid. More design care needed (OBSO's
   existing ball-conditioned Gaussian weighting overlaps this semantically); spec may keep
   transition as-is for v1 and ship EPV only.
3. Thread an `xt=` convenience kwarg through `add_obso`/`obso_xfns`/`compute_pass_obso`/
   `add_space_creation` (mutually exclusive with raw `epv_grid`).
4. **Do not change the defaults**: `space_creation` shares `_get_default_grids` and 4.24.0
   deliberately x-mirrors the transition/EPV artifacts for the opponent surface — an injected
   xt-derived EPV must flow through that same mirroring seam (implementation note: mirror the
   *interpolated grid*, exactly as the synthetic one is mirrored today). Consider a one-time
   `warnings.warn` when the synthetic defaults are used (observable but additive).
5. Documentation: state plainly that the synthetic defaults are placeholders for unit-scale
   demos, not production surfaces.

**Validation plan:** golden test — adapter output equals manual interpolation on a hand-built
grid; discriminating behavioral test — OBSO with real xT ≠ OBSO with synthetic ramp on the same
frames (mutate → red); auto-gates (purity/liveness/dup-action-id/id-dtype) already enumerate
`add_obso`/`obso_xfns`.

**Flags:** C4-free; serve-output changes only for callers who opt in (obso/space-creation xfns
are in no default list → **no forced retrain**; opting in is the caller's self-triggered
retrain, same as xt_xfns). Effort: small. Attribution: Spearman 2018 already in NOTICE; no new
entry needed.

#### W4 · Per-event defensive credit/debit family (module 16.3)

**Definition (Sumpter, coach-consulted, module 16.3 — exact rules as taught):**
- **Proximity/marking thresholds:** a defender "applies pressure" / holds marking
  responsibility within **4.5 m outside the penalty box, 3.0 m inside** (explicitly
  coach-negotiated and somewhat arbitrary → frozen params dataclass).
- **Signed scoring rules** (credit sized by the danger value — shot rules by that shot's xG,
  pass/recovery rules by xT):
  - pressure on a shot that misses → **+** for the presser (sized by shot xG);
  - failed pressure, shot on target → **−** for the nearest defender;
  - pressure on a pass that fails → **+** presser / **−** passer;
  - pressure → pass fails → own team recovers → **double credit** (pressure point + recovery
    point), + for the recoverer, − for the original passer;
  - failed cross-block → **−** nearest defender at the receipt point, **+** for the eventual
    shot-blocker;
  - failed marking on a high-xT through ball → **−** sized by the resulting shot's xG, assigned
    to the nearest/responsible defender *at the moment the pass was played*;
  - beaten 1v1 leading to a quality shot → **−** for the nearest defender (4.5 m);
  - forced bad touch → **+** for the presser.
- **Synchronized final-third pressure** (team metric): when a pass fails in the final third,
  credit **ALL** teammates simultaneously within 4.5 m of the carrier (≥3 in the worked
  example) — collective option-reduction, validated qualitatively against a Liverpool pressing
  season.
- Named-but-unbuilt extension: **passing-lane blocking credit** (geometric lane obstruction
  without an event) — silly-kicks' cover-shadow/lane-control machinery is a head start.

**MSC corpus additions (2026-07-16):**
- **"Bravery" (Tigres Femenil)** — % of the opposition's total "final actions" (shots + crosses)
  that were blocked (worked example: 32/40 = 80%) — a concrete, event-only defensive-credit
  team metric to include in the family (or T1).
- **Pressure-commitment signal (PSG/Luis Enrique webinar)** — Curneen argues pressure intensity
  is objectifiable as *touching-distance proximity + no deceleration of the approach run*
  (committed full-speed closing vs the traditional slow-down-and-show cue). The
  no-deceleration component is computable from frame velocities and would be a novel
  *commitment* dimension on top of the existing distance-based `pressure_on_actor` flavors —
  capture as an optional feature cue, not v1.
- **RB Salzburg 4-action taxonomy** (win / intercept / block a pass option / cover a teammate)
  — a clean labeling vocabulary for the credit rules' event anchors; the 10 Salzburg pressing
  principles (cover shadow as ball-as-light-source, net-not-chain, luring/trap angles,
  overload-on-ball, lurking, press-not-cover) qualitatively corroborate the existing
  `cover_shadows`/pressure design and supply docstring-grade vocabulary.

**Repo state:** no counterpart exists (VAEP values the on-ball event stream; `pressure_on_actor`
measures pressure but assigns no signed credit; nothing does proximity-gated defensive
attribution). All required primitives exist: linked frames, nearest-opponent geometry
(cover-shadows / xS feature kernels), injected xG (`xg_column` port precedent — silly-kicks
ships no xG), fitted `ExpectedThreat` (opponent-perspective mirroring precedent in 4.24.0
space-creation).

**Design decisions for the spec:** (1) exact SPADL anchor per rule (shot, pass+result,
bad_touch, tackle, interception, clearance, cross); (2) "responsible defender" resolution — at
the *linked frame of the triggering action* with ADR-028 re-projection and ADR-019 id-compat
throughout; (3) "reverse xT" formalization for pass/recovery sizing (opponent-perspective
mirrored surface); (4) output shape — long-form per-credit table (`action_id, player_id,
rule, signed_value`) primitive + per-action `add_*` aggregate + per-player rollup left to
consumers; (5) NaN-safe per ADR-003; (6) which rules are v1 vs deferred (recommend: the six
pressure/marking rules v1; lane-blocking deferred).

**Validation plan:** rule-by-rule ground-truth fixtures (synthetic frame + action → known
signed credit); mirror-invariance gate; liveness/purity/dup-action-id/id-dtype auto-gates via
registration; qualitative sanity vs the course's worked examples.

**Flags:** tracking-required; new action-coupled aggregator → **C4 +1**; no default xfn list →
**no retrain**. Effort: medium. Attribution: course module 16.3 (practitioner anchor) in NOTICE.

#### W5 · Packing — Impect-faithful surface over the existing LBS kernel *(elevated from Tier 3; owner interest, recognition value)*

**Canonical definition (Impect — Stefan Reinartz & Jens Hegeler, ~2015, post-WC2014):** packing
counts the opponents removed from the defensive phase — left behind the ball — by a single
**completed pass or dribble/carry**. Core question: "are there fewer defenders between the ball
and the goal than before the action?" Two canon details most re-implementations miss:
1. **Every bypassed opponent generates TWO credits — one to the passer, one to the receiver**
   ("completed **and secured**" reception; packing deliberately values the off-ball movement
   that created the receiving space).
2. The headline sub-metric is **defenders outplayed** (bypassing the last line is weighted above
   midfield bypasses — the "Impect value"). Sources vary on the line: RWTH/Impect framing says
   "last six players" (GK-inclusive); the owner's Modern Soccer Coach webinar note says
   "Goal threat packing — **last 4 defenders only**" → make line size + GK-inclusion parameters,
   default back-4 to match `compute_defensive_line(n=4)`.

**Published academic anchor (satisfies the house attribution discipline):** Goes, Kempe,
Meerhoff & Lemmink (2019), "Not Every Pass Can Be an Assist" (Big Data,
doi:10.1089/big.2018.0067) — formalizes outplayed defenders **longitudinally (goal-to-goal
axis) only**, i.e. the published formalization IS the 1-D geometry; also defines the negative
counterpart on failed passes (own teammates outplayed / distance conceded toward own goal).
Vendor definitions genuinely vary (Soccermatics module 17: SkillCorner uses the literal
last/second-last defender; others cluster) — pick canon + parameterize, don't chase consensus.

**The "subtract on immediate return / bounce pass" question (owner, 2026-07-16 — RESOLVED
against the now-transcribed Modern Soccer Coach source):** no source documents a retroactive
time-window *subtraction*. The MSC packing lesson (`6_Packing Data.srt`, identical segment
embedded in Bootcamp Webinar 2) addresses the bounce pass directly as a **coding decision at
count time**: a Declan Rice line-breaker played straight back → "you've got to decide whether
that's a packing moment," vs a ball received "on the front foot" with space → packing. That is
an *exclusion* rule (count or don't count), aligning with Impect's "completed and **secured**"
reception condition. Separately, **net-packing implementations** apply directional multipliers
(open-source `football-packing` library: **forward ×1, sideways ×0.5, backward ×−1**) under
which a wall pass A→C→A nets ≈0 over the sequence — the closest thing to "subtraction," expressed
per-action. Capture BOTH as spec parameters: `secured_reception` (bounce-back exclusion — MSC/
Impect-aligned; operationalizable as receiver's next action not an immediate return, or ball
stays beyond the bypassed line ≥T s, reusing `retains()`-style windowing) and `net_packing`
(directional multipliers, default off = Impect per-action canon).

**Additional rules from the MSC lesson (practitioner "packing expert", reconciled 2026-07-16):**
- **"Still in a position to defend" eligibility** — the expert awards FEWER points when bypassed
  defenders can still defend (box defenders → 2 not more; "eliminates three defenders, there's
  still one recovering, so not four points" — a *recovering* defender is not packed). Human-coded
  there; with tracking data this maps to a **TTI/recovery-aware eligibility criterion**
  (defender behind the ball but able to recover before the receiver's next action ≠ eliminated)
  — implementable with the existing pitch-control TTI machinery, and a more principled
  refinement than corridor restriction. Capture as an optional `eligibility="tti"` variant
  (default remains pure-geometry canon).
- **Goal-threat packing = the last 4 defenders** ("back four, or back three + a defensive
  midfielder — whatever the last four defenders are"), with **receiving packing** explicitly
  added and **three scoring channels: pass, reception, dribble** — corroborates the
  defenders-outplayed parameterization (default n=4 per this source; RWTH says last 6).
- **One-touch dribbles count** (a one-touch elimination of 3 players scored 3 — no "traditional
  dribble" requirement).
- **GK distributions score packing** (Ederson worked example, 6 points) — the packing domain
  includes goal-kicks/keeper passes → natural tie-in with `gk_distribution_mask` and a cheap
  companion read on keeper line-breaking distribution alongside xT-GK.
- **Practitioner composite + reference constants** (downstream/validation material, not core):
  "packing score" = players-packed × forward-pass-% (Romero 53 × 79% = 41.9 — a
  retention-adjusted composite); ~**8 packing points per shot** as a working ratio; **67.4% of
  PL open-play goals involve ≥1 packing action**; ~2 players bypassed per packing action;
  % of packing passes followed by forward progression (69% worked example — matches the owner's
  original webinar note). Useful as external sanity anchors for validation, and as
  lakehouse-side derived metrics.

**Repo state (verified):** `structural_lbs` (`_structural_pass.py`) already computes the 1-D
forward bypass count — canon-faithful geometry (its documented far-touchline caveat is faithful
to the published formalization, not a defect). Deltas to canonical packing (the actual work):
1. **Completion gate** — `_kernels.py::_structural_pass_at_actions` computes for every
   pass/cross regardless of `result_id` (verified: no result check at line 953); packing counts
   only completed actions.
2. **Action set** — currently `type_id ∈ (0, 1)` (pass, cross) only; packing includes
   dribbles/carries (SPADL dribble has start/end — same kernel applies).
3. **Receiver credit** — Impect's two-entry convention; new columns (e.g. `packing_made` /
   `packing_received`); receiver = next same-team touch (no `receiver_player_id` in SPADL).
4. **Defenders-outplayed sub-metric** — gate the count to the opposing back line via existing
   `compute_defensive_line`/`select_back_line_players` (n parameterized, GK-inclusion flag).
5. **Naming/attribution** — emit as a `packing_*` surface with Impect + Goes-et-al. NOTICE
   entries; **`structural_lbs` stays frozen byte-identical** (Hyrum) — packing is a sibling
   surface sharing the kernel, not a rename.
6. **Optional non-canon variants (flagged, never default):** corridor/lane restriction (a
   silly-kicks extension — published canon is deliberately 1-D); `net_packing`;
   `sustained_packing` (above). ADR-005 §8 naming if shipped.
7. **Downstream (lakehouse, not library):** the owner's webinar-note aggregations — packing
   rate (% of passes that pack), % of packing passes followed by forward progression,
   player-combination packing matrices (e.g. RB→CAM).

**Validation plan:** hand-built fixtures with known bypass counts per rule (completion gate,
dribble, receiver credit, back-line gate); golden identity — `packing_made` with all extras off
== `structural_lbs` on completed passes (discriminating: flip the completion gate → red);
auto-gates via registration.

**Flags:** tracking-required (opponent positions); new action-coupled aggregator → **C4 +1**;
no default xfn list → **no retrain**. Effort: small-to-medium (wrapper + attribution over an
existing kernel). MSC reconciliation DONE (2026-07-16, transcripts ingested — see the two
sub-sections above). Sources: Impect (impect.com; Reinartz/Hegeler), Goes et al. 2019
doi:10.1089/big.2018.0067, RWTH Aachen "Packing Rate – the Art of Outplaying", ESPN Reinartz
profile, kharasportsdaily/wyfutbol explainers (receiver-credit detail), `football-packing`
library (github.com/samirak93/football-packing — directional multipliers), Modern Soccer Coach
`6_Packing Data.srt` + Bootcamp Webinar 2 (practitioner variant rules), owner note
`Modern Soccer Coach/Packing passes.txt`.

### Tier 2 — validation/QA wins (small, bundle-able into hygiene PRs)

#### V-a · xT solver cross-check gates
The Markov system has three provably-equivalent solutions: exact `np.linalg.solve(I−A, g)`,
fixed-point iteration (= `xthreat.value_iteration`), Monte-Carlo absorption sampling (course
worked all three to agreement on a toy 3-zone chain: A=[[.25,.20,.10],[.10,.25,.20],
[.10,.10,.25]], g=[.05,.15,.05] → xT≈[0.150, 0.252, 0.120]). **Shape:** test-side oracle first
(per `feedback_speculative_api_surface_is_debt` — no public API until a consumer appears): a
`tests/` helper that solves (I−A)x=g exactly and asserts `value_iteration` output ≈ exact on
both the toy chain and a real fitted grid; optional MC agreement test (seeded, loose tol).
Discriminating requirement: perturb the transition matrix → gate goes red. Respect the house
rule: `value_iteration` convergence is monotone-from-below raw-diff — do NOT "fix" to abs.
Effort: small. No API change.

#### V-b · External xT magnitude anchors
Sumpter's real multi-league fitted values: in-front-of-goal ≈37.7%, intermediate ≈18% / 7.5%,
generic mid-pitch ≈1%, **GK zone ≈2.5%**, own-half non-GK ≈0.5%, deep own corner ≈0. Corpus-
dependent → encode as ORDER/ratio plausibility gates, not absolute equality (goalmouth ≫ deep
corner; GK zone between own-half and mid-pitch; monotone toward goal along the central column)
on the committed fixture grids, + cite the absolute anchors in docs as external reference.
Directly on-theme after PR-S113's fabricated-zone lesson (a fitted V(z,p)/xT surface should be
plausibility-checkable against an independent published fit). Effort: tiny.

#### V-c · Pass-risk completion diagnostic
Lesson 15.2b's workflow: score every completed pass at its own destination/moment (no grid
render), histogram the model-implied success probability, inspect the low-p-but-completed tail
and what happened next (majority of the 18 sub-50% completed passes in the demo match were
followed by an immediate challenge/loss — "technically complete, functionally lost").
**Repo shortcut (verified):** `pitch_control_at_target` already samples PC at the action
destination — the diagnostic is that column + `result_id`, fed through the extra-free
`_calibration_metrics.py` (`ece`, `reliability_slope`) + AUC. **Shape:** owner-run script (or
`@e2e` test) producing a calibration report of the Spearman surface against real completion
outcomes per provider — a QA tool for pitch-control quality in the same spirit as the
GkCompletionModel gates, no new library surface needed for v1. Effort: small.

#### V-d · Retention-label cross-validation vs the course reference
Diff `xtgk/_retention_labels.py::retains()` against the course's worked reference
(lesson 3.4 + the shipped `Possessions retained after 5s` KPI): window 5 s (ours: 10 s default);
shot within window → retained (ours: same); ball out for goal-kick/corner/throw-in/kick-off →
lost (ours: opponent-boundary semantics — check set-piece handling); ignore-list
Pressure/Block/Shield/Foul-Committed/Duel as non-possession-changing (StatsBomb vocabulary;
ours operates on SPADL where these mostly don't exist as actions — document the mapping);
Twelve's house rule requiring 2 touches to count a recovery at all. **Deliverable:**
verification note documenting each divergence as deliberate-or-not; optional network-gated
`@e2e` fixture reproducing the course's worked numbers (England WEuro-2025 QF: 76 recoveries →
70 analysable → 80.0% retained; by third 81.5/77.3/81.0%) via statsbombpy open data (the
`test_xthreat_statsbomb_e2e.py` pattern). Effort: tiny-small. Verification only — no behavior
change without its own decision.

#### V-e · Season-repeatability harness (longitudinal construct validity)
Type II / major-axis regression (error on both axes — NOT OLS) + Pearson r of the same player's
metric in season N vs N+1, per position group — the *longitudinal* complement to the house ICC
discipline (cross-sectional). Course findings to encode as documented cautions: xT-from-passes
r≈0.82 and touches/90 r≈0.79 (skill-like, repeatable); **G−xG has one of the lowest
year-to-year correlations of any tracked metric** — never surface it as "finishing skill";
repeatability is position-dependent (rank per position, not globally). **Shape:** pure-numpy
major-axis regression (~20 LOC — no `pylr2` dep) + a `season_repeatability(df, metric,
player_key, season_key)` helper in `calibration/` or scripts. **Blocked on multi-season data**
(WC tournaments are single-cohort; check whether the 98-match owner-tier SkillCorner corpus
spans seasons). Effort: small, data-gated.

#### V-f · Possession-chain convention check
The course's chain-break rule: a chain breaks only on **TWO consecutive opposition touches**
(+2 for shot/foul/offside/ball-out; half-end always breaks), and a chain ended by a FOUL has
the following chain's shot xG stitched backward onto it. Verify VAEP `window="possession"`
against this convention — divergence may be deliberate (document) or an oversight (decide).
Also confirm whether StatsBomb-style **xGChain** (credit every touch in a chain with the
terminal shot's xG; "Messi test") is materially covered by `window="possession"`; if not, it is
a near-zero-cost extra baseline for the xT-GK v2 construct-validity harness (more honest naive
baselines strengthen the §3-style verdict machinery). Effort: tiny (investigation first).

### Tier 3 — self-contained optional features (owner scope calls, rough value order)

#### T1 · Event-only team-KPI module (Twelve match-report glossary)
Pure-pandas-over-SPADL team-match metrics, none currently in the library (all with the report's
exact definitions + worked values as reference):
- **PPDA** — opponent passes in their defensive 60% ÷ our defensive actions in the same region
  (worked: 4.17).
- **Defensive intensity** — defensive actions (duels, interceptions, tackles, fouls) per minute
  out of possession (7.16).
- **Field tilt %** — share of final-third possession vs opponent (63%).
- **Pass tempo** — passes per minute of possession (19.97 / opp 15.83).
- **Defensive action height / recovery line height / turnover line height** — mean x (m) of
  defensive actions / open-play recoveries / open-play losses (41.15 / — / 66.68).
- **Time to defensive action / time to recovery** — mean seconds from losing the ball to first
  defensive action / to regaining possession (7.30 s / 7.69 s).
- **Recoveries + recoveries-within-5s %** (14%) — the gegenpressing execution metric (the
  Hammarby 10%→20% coaching-intervention anecdote runs on exactly this).
- **Possessions retained after 5 s** (18 / 62%) — the course's coded-from-scratch metric
  (cross-links V-d).
- **Conversion chain** — possessions-to-final-third %, final-third-to-box %, box touches,
  box-to-shot %.
- **Long-ball %** — own-defensive-half passes traveling >32 m.
- **High-opportunity shots** — non-penalty shots with xG > 0.15 (needs injected xG).
- **"…within 10 s after recovery" windowed family** — every offensive metric recomputed in a
  post-recovery window (parameterized; Hammarby's KPI used 5 s).

**MSC corpus additions (2026-07-16 — the Bootcamp Webinar 3 Tigres/Clemson/Coventry material is
a second full practitioner KPI system; fold these in as first-class entries):**
- **Counter-press windows must support BOTH time-based AND pass-count-based definitions.**
  Named practitioner presets found across the corpora: Barcelona ~6 s to press (Guardiola rule),
  Coventry academy "**compact after 7 seconds, regain the ball in 5 seconds**", RB Leipzig
  "score within 10 s of winning the ball", Hammarby 5 s regain, and **Tigres' "The Hunt" =
  regain within 3 PASSES or less** ("some people do it within seconds, other people will do it
  in passes — for us it's three passes or less"), with a stated 60%+ season success standard.
  Ship the window as a parameter (`seconds=` xor `passes=`) with these presets documented.
- **Interception-height taxonomy (Tigres)** — High/Medium/Low keyed on which opponent lines
  remain in front of the ball at the regain: High = only the opponent's back line remains
  (regain beyond their 6/8); a 6-or-8 still behind the ball → Medium/Low. A *structural*
  (line-relative) alternative to the coordinate-based recovery-line-height metric — ship both.
- **Post-regain security metrics (Tigres)** — second-pass completion rate after a regain; count
  of failed first-passes after regain ("when we win the ball we want to secure it"); forward
  vs backward/sideways first-option-taken split (individual coaching cue).
- **Build-up outcome taxonomy (Tigres, 6 states)** — per build-up: progressed to final quarter /
  progressed to next phase / opposition interception in own half / stayed in phase one /
  opposition won ball in own half / led to an opposition shot; reported as counts + success
  rate (worked: 19 build-ups, 7 successful, 36%).
- **Breakout success by channel (Clemson)** — "progress the ball past the halfway line in
  possession", tallied per 3 vertical channels (left/center/right).
- **Switch-of-play-conditioned press success (Clemson)** — per short goal kick: was a switch of
  play prevented × was the ball regained (worked, small-N: 9/9 regain when switch prevented,
  0 when allowed).
- **Aerial first-ball → second-ball chaining (Clemson)** — aerial duel outcome linked to the
  subsequent loose-ball (second-ball) outcome, profiled per team and per player-matchup
  ("wins first balls, loses second balls" as a scouting lever).
- **"Bravery" blocked-final-actions %** (see W4 — could live in either family).
- **Compactness-recovery time (tracking-flavored, from the Coventry "compact after 7 s" KPI)** —
  seconds from possession loss until team compactness (existing `compute_team_shape` metrics)
  returns under a threshold; a small, novel `team_shape`-consumer metric rather than an
  event-only KPI — note it here to keep the KPI set coherent.
- **Block-classification anchors (MSC module 1)** — high press ⇔ defensive line pushed toward
  the halfway line; low block ⇔ 18-yard box as the line reference — usable as documented
  default thresholds for classifying block height from `defensive_line_height`.
- **Guardiola counter-press gate (MSC module 1)** — Barcelona rule: aggressive counter-press
  only if ≥3 completed passes in the current zone (4-zone split) else drop back a zone — a
  possession-depth-gated turnover-response *context* feature candidate (capture only).

**Design:** `spadl/team_metrics.py` (or a `team/` namespace); grain (game_id, team_id);
possession attribution + possession-minutes need a definitional section in the spec (reuse the
VAEP possession-window machinery or a simple consecutive-touch run — decide once, use for all).
**Boundary call for the owner:** these are team-match aggregates (downstream-flavored), but
single-sourcing definitions in the library is what delete-and-depend argues for. Effort: medium
(many small definitions; test volume — larger now with the MSC additions; consider a v1/v2
split: the Twelve glossary set v1, the Tigres/Clemson set v2). No tracking (except the
compactness-recovery-time note), no retrain, C4: no action-coupled aggregator.

#### T2 · Match-outcome simulation (win probability / xPoints)
Injected per-shot xG (port pattern; silly-kicks ships no xG) → team goal distribution =
**Poisson-binomial** over that team's shots (exact via DP convolution — cheap; MC optional for
parity) → joint over both teams (independence assumption, same as the course's model — document
it) → P(win/draw/loss) + **xPoints** = 3·P(win) + 1·P(draw). Course worked example for face
validity: Arsenal 33% / draw 25% / West Ham 41%, xPoints 1.25, in a match Arsenal won 1-0 on
lower xG (1.57 vs 1.72). Validation: analytic fixtures (sum to 1; symmetric shot lists →
symmetric probabilities; single-shot edge cases). Shape: small pure module; event-only.
Effort: small. C4: new small container, no aggregator.

#### T3 · "Van Dijk" territorial-dominance metric (module 10.2, Earpiece-productionized)
Event-only defensive-value construct, no tracking needed:
1. **Defensive area** — convex hull of the **70% of a player's own-half defensive-action
   locations nearest their centroid** (trimmed hull, robust to outliers), per player per
   match/season window.
2. **xT into the area** — classify every opposition pass targeted into the hull
   success/failure (forward flagged separately); weight by xT: **xT-conceded** vs
   **xT-prevented** (worked example: 0.93 in, 0.11 prevented, 0.35 net — the lesson does NOT
   fully formalize "prevented"; spec must pin it — proposal: prevented = xT of failed
   opposition passes into the hull, conceded = xT of completed ones).
3. Distinguishes "territory the opposition cannot pass into" (Van Dijk profile) from "excellent
   once the ball arrives" — two orthogonal defensive qualities.
Implementation: scipy `ConvexHull` + point-in-hull; injected fitted `ExpectedThreat`; SPADL
defensive-action set (tackle/interception/clearance — pin in spec). Effort: small-medium.
Attribution: Sumpter/Twelve (course module 10.2) in NOTICE.

#### T4 · Glicko-2 duel-rating module (StatsBomb HOPS pattern)
Pairwise-contest rating over duel outcomes: every duel is a rating update between winner and
loser (Glicko-2: rating + RD + volatility; pure-python ~100 LOC, no new dep). **Data already
exists:** sportec emits `tackle_winner_*`/`tackle_loser_*` (ADR-001 — qualifier-derived facts
as dedicated columns). Extension per provider coverage: SPADL tackle/take_on adjacency
(tackler vs dribbler, result decides winner) as a derivation helper; aerial duels are
provider-specific (no SPADL type) — start with ground duels, document coverage honestly.
Design: rating period = match; provider-coverage table in the spec. Effort: small-medium.
C4: new module, not an aggregator. Attribution: Glickman (Glicko-2), StatsBomb HOPS precedent.

#### T6 · `describe_level` helper + machine-readable feature glossary
1. **`describe_level(z)`** — the wordalisation staircase (z≥1.5 outstanding / ≥1 excellent /
   ≥0.5 good / ≥−0.5 average / ≥−1 below average / else poor), vectorized + NaN-safe (ADR-003).
   Tiny, pure, lets any downstream report layer share canonical thresholds.
2. **Machine-readable glossary** of every emitted feature column (name, definition, unit,
   emitting module, attribution) as a data file + registry, with an auto-enumerating CI gate
   (every registered `add_*`/`*_xfns` output column must have an entry — the house gate
   pattern). Large one-time authoring cost (~100+ columns), incremental afterward; directly
   serves downstream wordalisation/Earpiece-style consumers and doubles as documentation.
Effort: helper tiny; glossary medium (authoring-heavy).

#### T7 · Modern Soccer Coach corpus — remaining notes (capture-only, no build implied)
- **7-zone final-third taxonomy** (`MSC Final Third Attacking Zones.pptx`, geometry extracted
  from shape coordinates): Goal Zone (central, near goal) + 2× Zone 1 (penalty-area inside
  channels) + 2× Zone 4 (wide byline/flank) + 2× Zone 2 (wide edge-of-box) + 1× Zone 3 (central
  edge-of-box / cutback). A pure spatial tagging schema — possible companion to the existing
  Gelade cross-zone feature if a final-third zone tag is ever wanted; downstream otherwise.
- **Pattern-confirmation heuristic** (Bootcamp W1): a behavior needs ~3 recurrences across games
  to be treated as an opponent tendency — consumer-side scouting discipline, not library code.
- **Formation-conditional stat splitting** (Clemson): opponent-facing KPIs segmented by the
  opponent formation encountered, never pooled — a consumer-side reporting convention worth
  documenting wherever team KPIs ship.
- **Tigres momentum chart** (shot-on-target/goal = 3 pts, off-target = 1 pt, cumulative
  timeline) — consumer-side visualization; noted only because it is a named practitioner
  alternative to xT-flow momentum.
- The rest of the MSC corpus (coach-philosophy modules, zone scouting checklists, coding-window
  workflow, tactical-system webinars, drill eBooks, AI-prompting webinar, IDP templates) is
  honestly **coaching/process content with no library-relevant metrics** — ingested, verdict
  recorded, nothing further to capture. The two `docx` files are dead WeTransfer link letters
  (Sarri throw-ins + periodization chart content NOT on disk).

## 5. Explicitly rejected (with reasons)

- **In-library shot-xG model** (module 7). Conflicts with a standing architectural decision:
  silly-kicks deliberately ships NO xG and injects the lakehouse's `fct_shot_xg`
  (canonical-SPADL xG v3, SB-360 context-aware) everywhere one is needed (ADR-036). Building
  one would duplicate the lakehouse model with a worse training corpus and fork the xG
  definition across repos. The course's tracking-xG *features* also substantially overlap
  xS + `pre_shot_gk_*` + cover-shadow geometry. Any future reversal argues on its own merits.
- **Deep learning (module 17: GNN/CNN-LSTM/offline RL).** Off-architecture (no-NN scipy/sklearn
  idiom; TODO already rejects the TacticAI line at TF-46). Two reusable takeaways only: the
  vendor-inconsistent line-breaking definitions independently validate the Ward-clustering
  choice, and "offline RL on real data over online simulation" matches the house real-data-gate
  bias. Policy optimization ("what should the player have done") is consumer-side territory.
- **Wordalisation LLM plumbing (module 11).** I/O-heavy application layer; violates hexagonal
  zero-I/O. (T6 captures the two pure spinoffs.)
- **Reporting/dashboard conventions** (radars, percentile/z-score displays, match/season report
  structures, rolling-window cadences, bench-perspective plots, Streamlit). Downstream consumer
  concerns; documented here only as ecosystem context.
- **Player aging curves / development trajectories (module 10.3).** Application-layer modeling
  on top of a feature library. (The season-repeatability *method* from the same module IS
  captured, as V-e, because it validates library metrics.)
- **Module 14 formations + module 15 pitch control as build targets** — already implemented at
  or beyond course level (verified, including convex hull area and canonical Spearman
  constants).

## 6. Coordination constraints (as of 2026-07-16)

- Part-deux owns `spadl/skillcorner.py`, `tracking/{skillcorner,metrica,_ghost_gk}.py`, the
  pining loader + corpus helpers, and the xS/xCross/ghost-GK trainers; DGX Stage B (~28–30 h)
  is running. W1's SkillCorner-taxonomy sub-part and any `physical.parquet` ingestion touching
  their files must wait or be handed off. W1's core, W3, W4, W5, and all of Tier 2/3 are
  collision-free.
- ADR-038's registered retrain protocol: none of the above retrains xS/xCross/ghost-GK; any
  future work that does must route through it.
- Next-free version 4.49.0, next-free ADR 039 — reserve at release time.

## 7. Suggested sequencing

1. **Owner decisions:** (a) W2 genre call (in scope as mission expansion or labeled sidecar);
   (b) first Tier-1 build (W1 / W4 / W5 all have complete definitions now; W5 is the smallest);
   (c) T1's library-vs-lakehouse boundary call.
2. **W3 + Tier 2 (V-a/V-b/V-c/V-d/V-f)** bundle into one or two hygiene PRs regardless of the
   Tier-1 choice.
3. Whichever Tier-1 item wins gets the standard brainstorm → spec → plan-review cycle before
   any code.
4. ~~When the Modern Soccer Coach `6_Packing Data.mp4` is transcribed, reconcile W5~~ — DONE
   2026-07-16 (transcribed + reconciled; no subtraction rule exists in the source; W5 updated).
5. **Remaining transcription candidates** (owner's whisper pipeline): the MSC `Webinars/` folder —
   **Michele Aragona "Scientific and Holistic Approach to Set-Pieces"** (TF-46-relevant, worth
   doing before any TF-46 spec), Gary Curneen "College Tactical Analysis", Ian McCall
   "Creativity in Football", Lloyd Yaxley "Goalkeeper Periodization" (GK-adjacent, likely
   load-management not analytics). The 20 scanning drill videos are training content — skip.
6. ~~Esposito et al. 2026 PDF unread~~ — READ 2026-07-16 (docling full text): 4-axis conceptual
   taxonomy, no computable run-type geometry; matches the 2026-06-05 TODO audit. Framing anchor
   only.
7. **MIGRATED 2026-07-16:** all actionable entries moved into `TODO.md` → On Deck →
   "Course-derived candidates" (flat table, no tier — the old Tiers 1–6 encoded dependencies;
   these have none). Numbers assigned: TF-49 packing, TF-50 physical, TF-51 defensive credit,
   TF-52 team KPIs, TF-53 win probability, TF-54 territorial dominance, TF-55 duel ratings;
   TF-35 moved from Research & Future Work into the same table (valuation arm unblocked);
   un-numbered rows: OBSO xT-wiring (TF-40 follow-up), validation/QA bundle, describe_level +
   glossary. This doc remains the detail reference the TODO rows point at.
