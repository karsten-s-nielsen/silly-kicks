# TF-58 — Team-coordination dynamics: relative phase, cross-correlation, vector coding, spectral, cluster-phase (design)

**Status:** Approved (owner, 2026-09-26; independent reviews r1–r3).
**Decision ADR:** one new ADR, numbered at commit-prep as the next free number on `main` (F1b holds ADR-106, native
DAS holds ADR-107/108). Amendment notes on ADR-048 (glossary `Unit` vocabulary gains `cycles/min`) and ADR-098
(metric-contract completeness keyed per exported constant).
**Version / PR:** assigned at commit-prep only (single-sourced `silly_kicks/_version.py`, ADR-079). Base `__version__`
is `4.127.0`; F1b and native DAS claim the next numbers first.
**Branch / base:** `feat/tf58-team-coordination` @ `05cfa56` (`main`). One rebase onto `main` after F1b (ADR-106) and
native DAS (ADR-107/108) merge is expected; §13.1 lists every touch point.
**Prerequisite:** ruthless-efficiency **0.7.0** with `GridSearchStrategy` (§8.4, handoff
`D:\Development\_handoffs\ruthless-grid-strategy-handoff.md`), designed and built by a separate session; this
session reviews it.
**Model routing (owner rule):** Opus 5.5 authors this spec and the plan. Implementation does not run on Opus 5.5; it
runs on the owner's chosen implementation model, inline, without subagents. Every review is independent and external.
**Owner decisions recorded (2026-09-26):** D1–D21, §4.

---

## 0. Executive summary (for reviewers)

silly-kicks already computes the *collective variables* of a football team from tracking data: where its centre is
(`compute_team_shape`: centroid, stretch, length, width, hull area) and where its back line stands
(`compute_defensive_line`, TF-14). It has nothing that describes how those signals **move together over time**:
whether two teams expand and contract in step, whether a back line's retreat trails the attack's push, how
synchronised a whole team is, or how fast a team's shape oscillates. The ecological-dynamics literature measures
exactly this, and a coach reads it as "are we moving as a unit, and who is leading whom?".

TF-58 adds that temporal-coordination layer as a new, descriptive, tracking-only metric package,
`silly_kicks.coordination`. It is not part of the VAEP path, adds no action-coupled feature, and needs no model
retrain. It implements, from the primary papers:

- **relative phase** (Hilbert instantaneous phase; Bourbousson, Sève & McGarry 2010; Folgado et al. 2014),
- **lagged cross-correlation** with lead/lag (Moura et al. 2016),
- **vector coding** coordination patterns (Moura et al. 2016, after Sparrow et al. 1987 and Chang et al. 2008),
- **spectral** median frequency and coherence (Moura et al. 2013; Welch 1967),
- **cluster-phase** whole-team synchrony, with sample entropy (Frank & Richardson 2010; Richardson et al. 2012;
  Duarte et al. 2013; Richman & Moorman 2000),

at four levels (team vs team, cross-variable and sub-unit pairs, player dyads, whole team), over caller-chosen
windows (whole periods, sliding windows, or possession sequences built from events or tracking), with a
**surrogate chance baseline** on every coupling number so a coach can tell coupling from coincidence.

Reading the papers surfaced defects and gaps that this design fixes and records: Moura 2016's printed vector-coding
equation takes an absolute value that makes its own 0–360° bins unreachable (§7.8.3); Bourbousson 2010 applies linear
statistics to wrapped angles (§7.8.1); Moura 2013 leaves the spectrum's DC removal implicit (§7.8.4); Duarte 2013
splices a substitute's trajectory onto the replaced player's (§7.6). Most of the corpus is SkillCorner broadcast
tracking, where a third of outfield positions are extrapolated and no dead-ball signal exists, so detection is treated
as data, stoppages fall back to event evidence, and both effects are measured in-cycle (§7.6, §7.11).

The unspecified numbers are set in three tiers (§8): **paper-fixed** constants; **derived** per provider by
deterministic procedures on the ~980-match corpus (residual analysis for the filter cutoff, noise floors, rhythm
band, a synthetic broadcast-occlusion experiment); and a **gated sensitivity sweep** on ruthless-efficiency's new
`GridSearchStrategy`. An in-cycle validation artifact replicates each paper's published findings on the full corpus
and reports every metric's reliability.

Performance was designed in (§7.15): measured on a real 25 Hz fixture, the existing per-frame `compute_team_shape`
loop costs 1.26 ms per frame (37 h per corpus pass at 10 Hz), Qhull costs 0.51 ms per call, and naive surrogate loops
cost 160 ms per dyad series. Vectorised collective-variable and defensive-line kernels, a phasor representation,
FFT-based surrogates and optional `numba` inner loops bring the budget to about 12–45 s per match single-core.

Two seams outside the package change: `tracking.preprocess` gains a zero-phase Butterworth filter, resampling and
residual analysis, and a vectorised collective-variable kernel becomes the single definition behind
`compute_team_shape` and `compute_defensive_line`. `compute_defensive_line` stays byte-identical;
`compute_team_shape`'s `convex_hull_area` changes by at most 1e-9 relative (§12).

---

## 1. Context

### 1.1 What exists

- `silly_kicks/tracking/_team_shape.py::compute_team_shape(frames, team_id, *, n_defensive_lines=3)` — per-frame
  centroid, convex hull area, length, width, Euclidean stretch index, Ward-clustered line height and gaps for ONE team;
  a Python loop over `groupby(["game_id","period_id","frame_id"])` with `ConvexHull` and `linkage` per frame. Called by
  `tracking/features.py:2438` and `:2572` (`add_team_shape`) and `restdefense/_compute.py:83`.
- `silly_kicks/tracking/_defensive_line.py::compute_defensive_line(frames, *, goal_map, n=4, adaptive_max_n=5)` — per
  (frame, team) back-line geometry, a Python loop over groups, direction from the `GoalMap` (ADR-055). Called by
  `causal/_confounders.py:157`, `restdefense/_compute.py:81`, `tracking/_kernels.py:863`,
  `tracking/_off_ball_runs.py:316`.
- The coach-facing descriptive cluster: TF-50 (planned), TF-52 `team_metrics`, TF-54 `territory`, TF-55 `duels`.
  Packages follow one shape: frozen `Params` with `default()` / `for_provider()` / `is_default()`, a closed
  `*_source` vocabulary, a conserving `Report`, `metric_contracts` registration, a two-way import allowlist.

### 1.2 The gap

No silly-kicks function measures coupling between two collective signals over time, the synchrony of a team as a
whole, or the rhythm of a team's expansion and contraction. `TODO.md` row TF-58 records the gap and its guardrails:
pure per-(game, window[, unit]) module; not action-coupled (the C4 aggregator count stays 33); tracking-required;
frozen params with `for_provider`; additive, no retrain; numpy/scipy; NOTICE attribution (ADR-005).

### 1.3 What TF-58 is not

- Not a VAEP feature family. It adds no `add_*` and no `*_xfns`, and nothing in the valuation path reads it.
- Not a composite or ranking. It ships raw primitives and per-window summaries; team ratings, archetypes and
  statistical tests across matches stay consumer-side (ADR-009).
- Not a freeze-frame metric. StatsBomb 360 has no time series and is refused (§7.3).

---

## 2. Primary sources and what each fixes

Implemented from the primary papers, never ported from a toolbox. DataGoal (Bedo et al. 2026, MATLAB, Apache-2.0) is a
cross-check only: not a dependency, not ported.

| Source | Read | What TF-58 takes from it |
|---|---|---|
| Bourbousson, Sève & McGarry 2010, Part 2, *JSS* 28(3):349–358 | full text (author copy) | team centroid and per-axis stretch index; Hilbert relative phase per axis; wrapped to ±180°; 12 × 30° histogram; relative stretch index SI_A − SI_B and its sign |
| Bourbousson, Sève & McGarry 2010, Part 1, *JSS* 28(3):339–347 | abstract | player-dyad level; cited only |
| Moura et al. 2016, *JSS* 34(24):2224–2232 | full text | spread (Frobenius norm, outfield only); 3rd-order Butterworth low-pass at 0.4 Hz on trajectories; cross-correlation over ±15 s, max \|r\| and its lag; vector coding with attacking team on the x-axis over offensive sequences split into thirds; Table 1 bins |
| Moura et al. 2013, *JSS* 31(14):1568–1577 | full text | surface area (convex hull, outfield) and spread; same filter, validated by distance and residual analysis; 7.5 Hz; FFT power spectrum; median frequency in cycles·min⁻¹; binary possession series spectrum |
| Folgado et al. 2014, *PLoS One* 9(5):e97145 | methods section | 3 Hz Butterworth on positions; Hilbert relative phase per axis; % time in −30°..30° ("near-in-phase") for outfield dyads |
| Duarte et al. 2013, *Hum. Mov. Sci.* 32:555–566 | full text | cluster-phase on football; 10 Hz; 11 players incl. goalkeeper; ρ_group mean/SD/SampEn; player–team relative phase; team–team Pearson r and Cross-SampEn; SampEn m = 1, r = 0.2·SD; stoppages > 25 s excluded |
| Richardson et al. 2012, *Front. Physiol.* 3:405 | full text (open access) | cluster-phase algorithm (q, φ_k, φ̄_k, ρ_k, ρ_group,i, ρ_group); centring and low-pass before Hilbert; surrogate data for chance-level synchrony |
| Frank & Richardson 2010, *Physica D* 239:2084–2092 | cited via Richardson 2012 | cluster-phase test statistic (Kuramoto order parameter) |
| Lamb & Stöckl 2014, *Clin. Biomech.* 29:484–493 | abstract | continuous relative phase recipe: centre → analytic signal → phase |
| Sparrow et al. 1987, *J. Mot. Behav.* 19:115–129; Chang, Van Emmerik & Hamill 2008, *J. Biomech.* 41:3101–3105 | cited via Moura 2016 | four-quadrant coupling angle; the four coordination patterns |
| Richman & Moorman 2000, *AJP Heart* 278:H2039–H2049 | cited via Duarte 2013 | sample entropy / cross-sample entropy definitions |
| Welch 1967; Carter 1987 | standard references | Welch spectral averaging; coherence estimator bias |
| Schreiber & Schmitz 2000, *Physica D* 142:346–382 | standard reference | IAAFT surrogates |
| Winter 2009, *Biomechanics and Motor Control of Human Movement*, 4th ed. | standard reference | residual analysis for the Butterworth cutoff; dual-pass cutoff correction |
| Pfister et al. 2013, *Front. Psychol.* 4:700 | standard reference | bimodality coefficient (H7) |
| Mardia & Jupp 2000, *Directional Statistics* | standard reference | circular mean, mean resultant length, circular SD |

The papers read in full are in the owner's uploads for this session; their text was read end to end (docling with a
pypdf cross-check for equations). Every constant this spec calls "Tier A" (§8.1) is quoted from a paper read in full;
the few fixed values no paper supplies are listed separately as conventions, each with its rationale (§8.1).

---

## 3. Investigation findings (evidence)

### 3.1 Findings from reading the papers

1. **Moura 2016 Eq. 2 is internally inconsistent.** It is printed as θ_vc(i) = arctan |Δθ₂ / Δθ₁|. The absolute value
   restricts θ to [0°, 90°], but the paper's own Table 1 bins span 0–360° and its text names angles of 135°, 225° and
   315°. The four-quadrant form `atan2(Δθ₂, Δθ₁)` of Sparrow 1987 / Chang 2008 is the only reading consistent with
   Table 1. TF-58 implements `atan2` and pins the divergence with a regression test showing the printed form
   misclassifies (§9.1).
2. **Moura 2013 does not state mean removal before the FFT.** Without it, the DC component of a positive-valued signal
   (spread, area) dominates the spectrum and the median frequency collapses toward 0; the reported 0.22–0.83
   cycles·min⁻¹ implies it was removed. TF-58 removes the mean and records this as an interpretation.
3. **Bourbousson 2010 reports linear mean and SD on angles wrapped to ±180°**, which is biased near ±180°. TF-58
   reports circular statistics (§7.8.1).
4. **Duarte 2013 splices** a substitute's trajectory onto the replaced player's column, creating a phase
   discontinuity at the join. TF-58 computes each player's phase on that player's own on-pitch segments (§7.6).
5. **Duarte excludes only stoppages longer than 25 s**, and Moura 2016 names including injury and substitution
   stoppages as a limitation. TF-58 splits continuous segments at dead-ball stoppages longer than 25 s (§7.6).
6. **Richardson 2012 recommends surrogate data** for chance-level synchrony. TF-58 ships a surrogate baseline on
   every coupling metric (§7.9).

### 3.2 Measured costs (throwaway probe, scratchpad, this laptop, py3.14, pandas 3)

On the committed 25 Hz fixture `tests/datasets/elastic_sync/j03wmx_slice/frames.parquet`:

| Operation | Measured | Per match at 10 Hz (~54k frames × 2 teams) |
|---|---|---|
| `compute_team_shape` (existing loop) | 1,256 µs per frame per team | ~136 s (37 h per 980-match pass; ~92 h at 25 Hz) |
| `scipy.spatial.ConvexHull` on 10 points | 513 µs per call | ~55 s |
| `compute_defensive_line` (existing loop) | 152 µs per frame-team | ~16 s |
| `scipy.signal.hilbert`, N = 27k | 1.3 ms | negligible |
| FFT cross-correlation, N = 27k | 1.2 ms | negligible |
| SampEn m+1 count, `cKDTree.count_neighbors`, Chebyshev, N = 27k | 260 ms per series | ~26 s (~100 series) |
| SampEn m count, sorted 1-D | 1.1 ms | negligible |
| 199 relative-phase surrogates, direct `angle(exp(...))` | 160 ms per pair-series | ~93 s (580 dyad pair-series) |

These drove the performance design (§7.15).

### 3.3 Detection in SkillCorner broadcast tracking

`docs/research/skillcorner_corpus/e2e_result.txt` measures, on private match 1021404, goalkeeper detection in 19.6%
of frames and outfield detection in 66.6%. SkillCorner is 909 of the ~980 corpus matches. The native
`tracking.skillcorner` builder keeps the per-player flag as `visibility`; the kloppy gateway discards it (ADR-069).

### 3.4 Dead-ball signal availability

`tracking/skillcorner.py:275–281` sets `ball_state = "alive"` on every frame ("SkillCorner's native feed carries no
reliable dead-ball signal"), and `tracking/metrica.py:179` does the same. Only the Sportec/IDSSE and Gradient Sports
native builders and the kloppy gateway set real states. A stoppage split that reads only `ball_state` would therefore
do nothing, silently, on 909 of the ~980 corpus matches. §7.6 defines the evidence precedence that replaces it (D20).

### 3.5 In-flight work in the sibling clones (read-only)

- **F1b** (`karstenskyt__silly-kicks`, `feat/f1b-float32-frames-id-category` @ `24ef308`, commit 1 frozen): frame
  coordinates become float32 storage with float64 compute and `team_id` becomes `category` (ADR-106).
  `tests/tracking/test_frame_coord_upcast_gate.py` pins `.to_numpy(dtype=float64)` on coordinate reads, scanning
  `silly_kicks/tracking/*.py` and `silly_kicks/tracking/pitch_control/*.py` only (not recursive: it misses
  `tracking/preprocess/`). F1b's `smooth_frames` writes `x_smoothed`/`y_smoothed` as float32.
- **Native DAS** (`karstenskyt__silly-kicks_part-deux`, `feat/das-native`, implementation in progress, uncommitted):
  touches `tracking/_das*.py`, `gkdv/`, `tracking/features.py`, `positioning/_objectives.py`. It removes the
  `player_id = "ball"` sentinel (unblocking `player_id → category` later), keeps `numba` as an optional accelerator
  with a numpy fallback, and packs frames through `_das_pack.pack_frames` (duplicate-row raise, canonical-id player
  order, `GoalMap` direction, reason codes). No TF-58 code path overlaps.

### 3.6 ruthless-efficiency 0.6.0 (read-only)

`OptunaConfig.sampler` is `Literal["tpe", "random"]`: there is no grid sampler. `StrategyConfig` (importable as
`ruthless.config.StrategyConfig`, defined in `ruthless/config/strategies.py`; not exported from top-level `ruthless`;
TF-58 never imports it) is a discriminated
union designed to grow without a breaking change. `RandomSearchStrategy` is the zero-dependency strategy template.
The calibration selection helpers in `silly_kicks.calibration` (`match_cv_splits`, `cv_standard_error`,
`exceeds_noise_floor`, `select_recommended_point`, `build_selection_artifact`) are numpy/sklearn only.

---

## 4. Owner decisions (2026-09-26)

- **D1.** Three levels of analysis — team–team, sub-unit and cross-variable, player dyads — through one signal-agnostic
  engine.
- **D2.** Cluster-phase is in (Richardson 2012; Duarte 2013).
- **D3.** Windows come from a `windows` port with three builders (period, possession from events, possession from
  tracking); events and tracking are never mixed in one call; `n_phases` subdivides windows (default 3 on possession
  windows).
- **D4.** Continuous segments split only at dead-ball stoppages longer than `max_stoppage_s = 25` (Duarte 2013).
  Amended by D20 for providers without a real dead-ball signal.
- **D5.** Three parameter tiers (A paper-fixed, B derived, C gated sensitivity); AlphaEvolve is not used; the
  validation artifact is in-cycle; two commits.
- **D6.** Surrogate baselines on every coupling metric: circular time-shift by default (K = 199), IAAFT opt-in;
  surrogate mean, percentile and excess columns; seeded and deterministic.
- **D7.** SampEn and Cross-SampEn are in (m = 1, r = 0.2·SD), with O(N log N) counting.
- **D8.** Goalkeeper policy per method, paper-faithful by default; no trajectory splicing.
- **D9.** Detection-aware handling for SkillCorner, including a synthetic broadcast-occlusion leg in the artifact.
- **D10.** Architecture approach A: a new `silly_kicks.coordination` package plus seam improvements in
  `tracking.preprocess`, a vectorised collective-variable kernel, and a `metric_contracts` gate re-key.
- **D11.** Design sections 1–5 as presented.
- **D12 (R1).** Vectorised convex-hull area replaces Qhull everywhere; `compute_team_shape.convex_hull_area` changes by
  at most 1e-9 relative; collinear frames still return exactly 0.0; fewer than 3 players still return NaN.
- **D13 (R2).** `compute_defensive_line` is vectorised and stays byte-identical.
- **D14 (R4 v2).** The sensitivity sweep runs as cached per-level corpus passes plus a ruthless-efficiency
  `GridSearchStrategy` (one-at-a-time design, then a joint confirmation point). ruthless 0.7.0 is built by a separate
  session; this session reviews it.
- **D15 (R5).** Phase is computed on maximal continuous segments; windows only select samples.
- **D16 (R6).** Optional `numba` kernels for three inner loops, numpy as the reference and fallback, parity-gated.
- **D17 (R7).** Dyads default to period windows; dyads on possession windows are opt-in.
- **D18 (R3, R8; session calls).** Phasor representation for relative-phase statistics and surrogates; structural cost
  guards and per-stage timing counters.
- **D19.** Forward-compatibility with in-flight F1b and native DAS work (§3.5, §13.1).
- **D20.** Dead-ball evidence (§3.4): a declared, fail-closed provider taxonomy `_DEAD_BALL_OBSERVED_PROVIDERS` in the
  neutral `tracking/_provider_visibility.py`; stoppage evidence in precedence frames `ball_state` (observed providers)
  → events (when `actions` are passed) → `unavailable` (no split); a per-row `coord_stoppage_source`; report counts;
  and a D3 leg measuring event-derived stoppages and the metric deltas on the providers with true `ball_state`.
- **D21.** ruthless 0.7.0 (spec approved round 4, `D:\Development\_reviews\2026-09-26-grid-search-strategy-spec-r4.md`)
  makes `StoreConfig.objective_id` required and adds a full Optuna resume-identity guard (a single
  `ruthless_identity` study attribute holding `objective_id` and a config fingerprint; legacy studies fail closed with
  `ruthless.strategies.optuna_.adopt_legacy_store(config)` as the remedy). Owner-approved on `StoreConfig`, honoured by
  Grid and Optuna. Consequences carried by this cycle (§7.16, §8.4, §12): D2 sets `objective_id` to its
  shard-generation token; every silly-kicks `StoreConfig` construction is migrated with one identity rule.

---

## 5. Goals and success criteria

1. **Fidelity.** Each method reproduces its primary-source definition; every divergence is named, justified and tested
   (§7.8.8).
2. **Correctness by construction.** Every kernel passes analytic ground-truth tests (§9.1); every threshold is tested
   from both sides; every counterfactual (surrogate) asserts non-vacuity.
3. **Honest degradation.** Every NaN carries a `*_source` reason from a closed vocabulary; windows are dropped and
   counted, never silently removed; the `Report` conserves.
4. **Interpretability.** Every metric column has a glossary entry that states its scale and direction, plus the
   published reference value where the cited literature reports one (the §8.5 figures), so a first-time reader can
   judge a value. No range is invented where no paper reports one (owner ruling, 2026-10-02, review minor 10).
5. **Validated in-cycle.** The D3 artifact replicates H1–H7 (§8.5) and reports reliability and poolability for every
   metric **construct** — a metric column × the keys that define *what* it measures (§8.5 grain) — on the full corpus.
   Each construct cell carries an explicit **power verdict**: an underpowered or hopelessly imprecise cell is terminal
   **"unmeasurable"** and is never pooled up a level to rescue power. Circular-mean constructs are scored with
   rotation-invariant **circular** reliability (§8.5), not an origin-dependent linear ICC. Units are keyed per measured
   entity and per (match, entity); the artifact states that across-halves reliability is an upper bound on true
   match-to-match reliability and that cross-match player reliability is unmeasurable on anonymised corpora. (Amended
   2026-10-04 per the owner's batch-3 ruling and reviews P-1/P-2; was "reports reliability and poolability for every
   metric".)
6. **Performance** (restated per the owner's ruling C on the performance measurements, ratified 2026-09-29; the
   original goal was "≤ 45 s per match single-core with numpy only (≤ 15 s with `numba`) on 10 Hz analysis; a full
   D3 pass fits in ≤ 1 h on 16 disjoint-slice workers"). The binding bound is the corpus: a full D3 pass (k = 199)
   fits in ≤ 1 h on 16 disjoint-slice workers with `numba`. Per match, cost scales with the provider's tracking:
   broadcast tracking (SkillCorner, 10 Hz, fragmented detections) ~18 s with `numba` / ~25 s numpy; native full
   tracking (IDSSE 25 Hz, GradientSports 29.97 Hz) ~115-130 s with `numba` / ~440-455 s numpy, of which ~32-35 s is
   the k = 0 floor (signal preparation + observed metrics) and the rest the surrogate nulls (199 draws x every window
   x every pair, through the §7.9 identities). No method, draw count, rate or dyad is traded for speed. The per-stage
   timers in every D3 manifest record these numbers; structural guards (not wall-clock) pin the algorithmic bounds.
7. **No regressions.** `compute_defensive_line` byte-identical; `add_team_shape` identical except
   `team_shape_convex_hull_area_*` within 1e-9 relative; the full suite green on the pandas-2 and pandas-3 legs.

---

## 6. Scope

### In scope

The `silly_kicks.coordination` package (§7.1–§7.14); the `tracking.preprocess` additions; the vectorised
collective-variable and defensive-line kernels with `compute_team_shape` and `compute_defensive_line` delegating; the
`metric_contracts` gate re-key; extraction of shared driver helpers (`scripts/_reliability.py`,
`scripts/_corpus_visibility.py`); drivers D1, D2, D3; the validation artifact; all CI gates and documentation (§9,
§11); the ruthless floor bump and its migration (D21: every `StoreConfig` construction, the three calibration study
builders, and the shared `scripts/_provenance.objective_id` helper).

### Explicitly not included (owner-approved decisions)

| Item | Decision |
|---|---|
| AlphaEvolve | Not used: the functional forms are fixed by the primary sources (D5). |
| Optuna TPE search in D2 | Replaced by an exhaustive grid on ruthless `GridSearchStrategy` (D14). |
| Dyads on possession windows by default | Opt-in via `dyad_windows="all"` (D17). |
| Normalising mixed-unit pairs for vector coding | Not invented: no primary source defines it; mixed-unit pairs get the scale-free methods only (§7.7). |

### Considered and rejected

| Option | Why rejected |
|---|---|
| TF-58 inside `tracking/` | `tracking` is the per-frame and action-coupled namespace; its registry gates would try to claim a per-window family. |
| A separate top-level DSP package | No second consumer; speculative public surface. `coordination/_kernels/` gives the same isolation. |
| Keep Qhull inside `compute_team_shape` | Two hull implementations for one quantity; Qhull costs 0.51 ms per call. |
| Hilbert per possession-window slice | Edge-dominated on ~15 s slices; a 2-minute rhythm cannot appear (D15). |
| Linear statistics on wrapped angles | Biased near ±180° (§3.1 item 3). |
| Trajectory splicing for substitutes | Artificial phase discontinuity (§3.1 item 4). |
| Splitting at every dead ball | Fragments halves into dozens of short segments (D4). |

---

## 7. Design

### 7.1 Architecture (hexagonal)

```
silly_kicks/coordination/
  __init__.py            public surface (§7.2)
  _config.py             CoordinationParams (frozen) + for_provider map import
  _provider_params_generated.py   GENERATED by D1/D2 (commit 2); empty map in commit 1
  _columns.py            every output schema (plain dicts), key lists, vocabularies
  _report.py             CoordinationReport (conserving)
  _windows.py            window builders + segmenting
  _signals.py            frames -> CoordinationSignals (the numeric port)
  _catalog.py            default pair catalog (L1-L4)
  _compute.py            compute_* families + compute_team_coordination orchestrator
  _series.py             compute_coordination_series (raw instantaneous series)
  _kernels/
    __init__.py
    _phase.py            centring, reflect-padding, analytic signal, phasors, phase validity
    _circular.py         circular mean / resultant length / circular SD / histogram
    _xcorr.py            lagged Pearson cross-correlation, Fisher-z pooling
    _vector_coding.py    coupling angle + Table 1 classification
    _spectral.py         periodogram median frequency, Welch coherence pooling
    _cluster.py          cluster phase, player-team relative phase, rho statistics
    _entropy.py          SampEn / Cross-SampEn (sorted 1-D + 2-D dominance counting)
    _surrogates.py       time-shift + IAAFT generators, seeded; FFT/phasor accelerated statistics
    _numba.py            optional @njit inner loops (parity-gated against the numpy reference)
```

**Import allowlists** (pinned by `tests/coordination/test_import_allowlist.py`, mirroring
`tests/restdefense/test_import_allowlist.py`, with planted-violation meta-tests):
- `_kernels/*` imports only `numpy`, `scipy`, the standard library, and optionally `numba`. No pandas, no silly-kicks.
- `coordination` imports `silly_kicks.tracking` public seams (`resolve_defended_goals`, `GoalMap`, `infer_ball_carrier`,
  `derive_team_in_possession`, `link_actions_to_frames`, the preprocess array kernels, the collective-variable kernel),
  `silly_kicks.spadl` (`add_possessions`, action-type config), `silly_kicks.id_compat`, `silly_kicks._frame_index`,
  `silly_kicks.reflection` / `tracking._geometry` transforms, `silly_kicks.tracking._provider_visibility`.
- Nothing in `silly_kicks` imports `coordination`; `tracking` must never import it.

**Seam changes outside the package** (every caller enumerated in §7.16):
1. `tracking/preprocess/_butterworth.py` (new): array kernels `butterworth_lowpass(values, fs, cutoff_hz, order)`,
   `resample_uniform(t, values, fs_out, run_bounds)`, `residual_analysis_cutoff(values, fs, grid)`; frame-level
   wrappers `smooth_frames(method="butterworth")` and `resample_frames(frames, target_hz)`. `PreprocessConfig` gains
   `butterworth_cutoff_hz` and `butterworth_order`; `SmoothingMethod` gains `"butterworth"`.
2. `tracking/_collective.py` (new): `compute_collective_variables(frames, *, include_goalkeeper=False)` public, plus the
   array kernels (`collective_from_positions`, `hull_area_batch`, `back_line_batch`). `compute_team_shape` delegates
   centroid, length, width, stretch index and hull area; `compute_defensive_line` delegates its six columns.
3. `tests/test_metric_contracts.py`: completeness keyed per exported `*_METRIC_COLUMNS` constant (ADR-098 amendment).
4. `scripts/_reliability.py` (new): `icc1`, `split_half_reliability`, `type_ii_slope`, `compare_providers`, extracted
   byte-for-byte from `scripts/validate_team_kpi_reliability.py`; `scripts/validate_gk_decision.py` drops its
   identical copy of `icc1`.
5. `scripts/_corpus_visibility.py` (new): `validate_corpus_visibility` moved from `scripts/train_ghost_gk.py`, which
   re-imports it under the same name. The re-import is load-bearing: the tests load `train_ghost_gk.py` by file path
   and call `t.validate_corpus_visibility(...)` directly as a module attribute
   (`tests/scripts/test_trainer_cache_and_providers.py:185–188, 194–195`), so dropping it breaks those calls.
6. `tracking/_provider_visibility.py` (D20): `_DEAD_BALL_OBSERVED_PROVIDERS = frozenset({"sportec", "idsse",
   "gradientsports"})` and `dead_ball_observed(provider) -> bool`, which raises on a provider absent from the existing
   classification (`validate_provider`), so an unclassified provider is never assumed observed. `skillcorner` and
   `metrica` are unobserved by construction (§3.4), whichever builder produced the frames.

### 7.2 Public surface (`silly_kicks.coordination.__all__`)

```python
@dataclass(frozen=True)
class CoordinationParams: ...          # §7.14, §8.1; default() / for_provider() / is_default()

def period_windows(frames, *, length_s: float | None = None, step_s: float | None = None) -> pd.DataFrame
def possession_windows_from_actions(actions, frames, *, links=None, n_phases: int = 3,
                                    possession_kwargs: Mapping[str, Any] | None = None) -> pd.DataFrame
def possession_windows_from_frames(frames, *, carrier=None, n_phases: int = 3,
                                   params: CoordinationParams | None = None) -> pd.DataFrame

def build_coordination_signals(frames, *, windows, params: CoordinationParams | None = None, actions=None,
                               goal_map: GoalMap | None = None, links=None) -> CoordinationSignals

def compute_relative_phase(signals, *, levels=DEFAULT_LEVELS, pairs=None,
                           dyad_windows: Literal["period", "all"] = "period"
                           ) -> tuple[pd.DataFrame, pd.DataFrame, CoordinationReport]   # (pair, pair_phase, report)
def compute_cross_correlation(signals, *, levels=..., pairs=None, dyad_windows=...) -> tuple[pd.DataFrame, CoordinationReport]
def compute_vector_coding(signals, *, levels=..., pairs=None) -> tuple[pd.DataFrame, pd.DataFrame, CoordinationReport]
def compute_coherence(signals, *, levels=..., pairs=None) -> tuple[pd.DataFrame, CoordinationReport]
def compute_spectral(signals) -> tuple[pd.DataFrame, CoordinationReport]
def compute_cluster_phase(signals) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, CoordinationReport]
def compute_relative_stretch(signals) -> tuple[pd.DataFrame, CoordinationReport]
def compute_coordination_series(signals, *, kind: Literal["relative_phase", "coupling_angle", "cluster_amplitude"],
                                pairs=None) -> pd.DataFrame

def compute_team_coordination(frames, *, windows=None, actions=None, params=None, goal_map=None, links=None,
                              levels=DEFAULT_LEVELS, dyad_windows="period") -> CoordinationResult

class CoordinationSignals: ...         # frozen; §7.4
class CoordinationResult: ...          # frozen; one attribute per table (§7.12) + report
class CoordinationReport: ...          # frozen; §7.13
class CoordinationCoverageWarning(UserWarning): ...   # subclasses no other category
# plus every *_COLUMNS / *_KEYS / *_METRIC_COLUMNS constant and COORD_SOURCE_VALUES / COORD_LEVELS / COORD_SIGNALS
```

The metric-family computes emit rows for the pair tables; `compute_team_coordination` concatenates the method
columns of one pair row across families (outer-joined on the pair keys, §7.12). `windows=None` in the orchestrator means
`period_windows(frames)` plus `possession_windows_from_actions` when `actions` is given, else
`possession_windows_from_frames`. `params=None` resolves `CoordinationParams.for_provider(<frames source_provider>)`
(the ADR-009 auto-promotion idiom); a frame set with more than one `source_provider` raises.

### 7.3 Input contract and refusals

`frames` follow `TRACKING_FRAMES_COLUMNS`. Required: `game_id`, `period_id`, `frame_id`, `time_seconds`, `frame_rate`,
`player_id`, `team_id`, `is_ball`, `is_goalkeeper`, `x`, `y`, `ball_state`, `team_attacking_direction`,
`source_provider`; `visibility` for detection-aware providers.

Raised before any computation (fail loud, each with a remedy in the message):

| Condition | Error | Remedy named |
|---|---|---|
| a required column is missing | `ValueError` | the column list |
| freeze-frame source: `source_provider == "snapshot"` (the `snapshot_to_tracking_frames` output, `tracking/_snapshot.py:141`, e.g. StatsBomb 360) | `ValueError` | "TF-58 needs a tracking time series" |
| `team_attacking_direction` absent or all-null (unoriented) | `ValueError` | `orient_frames_to_ltr` / `play_left_to_right` (ADR-029) |
| duplicate `(game_id, period_id, frame_id, player_id)` rows (ADR-004) | `ValueError` | loader de-duplication (the GS duplicate-frame case) |
| more than one ball row per frame | `ValueError` | same |
| detection-aware provider with `visibility` all-null | via `assert_detection_aware_visibility` (ADR-069) | native rebuild |
| more than one `source_provider` in one call | `ValueError` | split by provider |
| windows from mixed sources in one call | `ValueError` | one source per call (D3) |

Not refused; degraded per window with a `*_source` token (§7.13): an unresolvable goal end
(`GoalEndUnresolvedError` caught by name at the edge, token `goal_end_unresolved`), insufficient players, insufficient
detection, too-short segments.

### 7.4 Signal preparation (the numeric port)

`build_coordination_signals` runs once per call and produces a frozen `CoordinationSignals`. Per `(game_id,
period_id)`, grouped once via `silly_kicks._frame_index.group_rows` (ADR-068; the caller is registered in
`tests/_scale_guarded.SCALE_GUARDED` with a group-dimension growth guard, ADR-073):

1. **Dense scatter.** Long-form rows are scattered into preallocated float64 arrays `(T_native, P_slots, 2)` per team
   by `(frame index, player slot)`, where the frame index comes from `np.searchsorted` on the sorted unique frame ids
   and the player slot from the canonical-id order (`id_compat.canonical_id`, ADR-019; the DAS `pack_frames`
   convention). Coordinates are read with `.to_numpy(dtype="float64")` (ADR-106 boundary). A detection mask of the
   same shape comes from `visibility` (all-true for fully observed providers).
2. **Continuous segments.** A player's on-pitch time is split into runs at: dead-ball stoppages longer than
   `max_stoppage_s` (evidence per §7.6 "Stoppage evidence", measured on `time_seconds`, ADR-017 period-relative), detection gaps
   longer than `max_detection_gap_s` (detection-aware providers only), and absence (substitution, red card). Detection
   gaps up to `max_detection_gap_s` are bridged by linear interpolation. Team-level signals use the team's
   **segments** = maximal time intervals free of long stoppages and of changes in the team's on-pitch outfield count
   (a red card steps `spread` and the other extensive signals, which would otherwise leak into spectra and
   correlations; a substitution keeps the count and does not split). An inexact handover -- the incoming player
   appears a little before or after the outgoing one leaves -- moves the count off its level and back: a count
   excursion that returns to its prior count within `max_stoppage_s` is a handover and does not split; a longer or
   non-returning change does (owner ruling 2026-10-03, review M-13; measured on 50 corpus matches: GradientSports
   handovers all return within 22.5 s, most within 2 s, the next change lasts 42.5 s; SkillCorner and IDSSE count
   changes all sit at a detection gap or inside a long stoppage). The samples of an absorbed excursion keep their
   values.
3. **Filter.** Each player run is low-pass filtered with a zero-phase Butterworth (`scipy.signal.butter(..., output=
   "sos")` + `sosfiltfilt`) at the native rate, before resampling, so the filter is also the anti-aliasing filter.
   `butterworth_cutoff_hz` is the −3 dB cutoff of the **combined** dual-pass response: the design frequency is
   `cutoff / C` with `C = (2^(1/2) − 1)^(1/(2·order))` (Winter 2009). A run shorter than `sosfiltfilt`'s pad length is
   dropped and counted (`too_short`).
4. **Resample** on a uniform period-relative `time_seconds` grid by linear interpolation within runs only (never across
   a split). The effective rate is `min(native_hz, max(analysis_hz, 10 × butterworth_cutoff_hz))`: 10 Hz by default
   (Duarte 2013), raised when a derived cutoff exceeds 1 Hz so the rate stays at least 10 × the cutoff (the rule
   approved with the tiers), never above the native rate. When the native cap binds, the effective rate is recorded in
   the report. A cutoff at or above the native Nyquist raises.
5. **Team signals** via the collective-variable array kernel (§7.5) on the filtered, resampled positions: centroid
   x/y, length, width, stretch (Euclidean), stretch x/y, spread, hull area, `defensive_line_x`, `compactness_x`,
   `back_line_high_x`, with an `observed_fraction` per sample (share of the team's on-pitch players detected).
6. **Player series:** filtered, resampled x/y per player run (observed samples only for detection-aware providers).
7. **Possession series:** per sample, the team in possession from the same source as the windows (§7.6), as a
   nullable team id plus the binary series for Moura 2013's spectrum.
8. **Orientation** (§7.10): the reference-team direction per `(period, team)` from the `GoalMap` built once per match
   (`resolve_defended_goals(frames)`, ADR-055) unless `goal_map` is passed.
9. **Phases and phasors** (§7.8.1): computed once per continuous segment for every series a method needs, never per
   window (D15).

`CoordinationSignals` holds the arrays, masks, segment index, window index (window → sample ranges), possession
series, phasors, provenance (`provider`, params used, `window_source`) and nothing mutable.

### 7.5 Collective-variable kernel (single definition)

For team `T` at sample `t`, with `n` = players with valid coordinates (outfield unless `include_goalkeeper`):

| Variable | Definition | Minimum n | Source |
|---|---|---|---|
| `centroid_x`, `centroid_y` | mean of x, y | 1 | Bourbousson 2010 |
| `team_length`, `team_width` | max − min of x, y | 1 | existing `compute_team_shape` |
| `stretch_index` | mean Euclidean distance to the centroid | 1 | Clemente et al. 2013 (existing) |
| `stretch_x`, `stretch_y` | mean \|x − c_x\|, mean \|y − c_y\| | 1 | Bourbousson 2010 (per-axis stretch index) |
| `spread` | √(Σ_{i>j} d_ij²) | 2 | Moura 2012/2013/2016 (Frobenius norm of the lower-triangular distance matrix) |
| `convex_hull_area` | area of the convex hull | 3 | Moura 2013; existing |
| `defensive_line_x`, `compactness_x`, `back_line_high_x`, `lateral_width`, `max_lateral_gap`, `back_n_count` | TF-14 definitions | 3 | existing `compute_defensive_line` |

Kernel mechanics:
- Padded arrays `(F, P, 2)` with a validity mask; all reductions are masked. Sums use the same order as the existing
  per-frame code wherever byte-identity is required (below).
- **Spread** uses the exact identity Σ_{i<j} ‖p_i − p_j‖² = n · Σ_i ‖p_i − c‖², which makes it O(n) instead of O(n²);
  a parity test pins it against the naive double sum within 1e-9 relative.
- **Hull area (D12).** A point is a hull vertex iff the largest angular gap between the directions to the other points
  is ≥ π (`atan2` angles, `(F, n, n−1)` sorted per row). A point whose gap is within rounding of π lies on, or within
  rounding of, the hull boundary, so misclassifying it moves the area only at rounding level. Area is the shoelace sum
  over hull vertices ordered by angle around their mean (which lies inside the hull). All-collinear frames (every orientation determinant
  exactly zero) return exactly 0.0, preserving `compute_team_shape`'s `QhullError → 0.0` contract; `n < 3` returns NaN.
  Parity against `scipy.spatial.ConvexHull` within 1e-9 relative on random and adversarial point sets (duplicates,
  near-collinear, 3 and 11 points).
- **Back line (D13).** Per row, sort x toward the defended goal (from the `GoalMap`, never team identity), apply
  `_select_n`'s rule vectorised (the fixed-N path and the adaptive dominant-gap path), then the six TF-14 reductions.
  The selected back line has at most 5 values, and numpy sums fewer than 8 values sequentially, so masked sums with
  trailing exact zeros reproduce the loop's floating-point order: **byte-identical** parity is required and gated.
- `compute_team_shape` keeps its public signature and output schema. It delegates centroid, length, width, stretch
  index and hull area to the kernel and keeps its Ward line clustering. `add_team_shape` is unchanged in code; its
  `team_shape_convex_hull_area_*` columns inherit the ≤ 1e-9 relative change (§12).

### 7.6 Windows and segments

**Window contract** (`COORD_WINDOW_COLUMNS`): `game_id`, `period_id`, `window_kind` ∈ {`period`, `sliding`,
`possession`}, `window_id` (Int64, sequential per `(game_id, period_id, window_kind)` by start time), `window_source` ∈
{`period`, `possession_events`, `possession_tracking`, `caller`}, `start_time_s`, `end_time_s` (period-relative,
inclusive start, exclusive end), `attacking_team_id` (nullable), `terminal_action` (nullable), `terminal_team_id`
(nullable), `n_phases` (Int64). A caller-built table must satisfy the contract (validated; `window_source="caller"`).

**Builders.**
- `period_windows`: one window per period; with `length_s` and `step_s`, sliding windows inside each period.
- `possession_windows_from_actions`: `spadl.add_possessions` defines possessions; each window runs from the first
  action's time to the time of the event that **ends** the possession: the attacking team's last action when it is a
  shot type, otherwise the opponent's first action of the next possession (the regaining tackle, interception, etc.).
  `terminal_action` is that event's SPADL type name and `terminal_team_id` its team; Moura 2016's shot/tackle subset is
  a filter on these columns. Times come from the action clock (period-relative); frames are selected by
  `time_seconds`. `links` (from `link_actions_to_frames`) may be passed to reuse a linkage.
- `possession_windows_from_frames`: `infer_ball_carrier` (or a passed `carrier`) → `derive_team_in_possession`;
  possession spells are maximal runs of one team in possession, bridging NA gaps up to `possession_gap_s` (Tier B,
  §8.2); `terminal_action` and `terminal_team_id` are `<NA>` (unknowable from tracking, never guessed).

**Stoppage evidence (D20).** Resolved once per call, in precedence order, and recorded per row as
`coord_stoppage_source`:
1. `ball_state`: when `dead_ball_observed(source_provider)`, dead intervals are the maximal `ball_state == "dead"` runs.
2. `events`: otherwise, when `actions` are passed. A dead interval runs from the end of the action before each
   set-piece restart (`throw_in`, `freekick_crossed`, `freekick_short`, `corner_crossed`, `corner_short`, `goalkick`,
   `shot_freekick`, `shot_penalty`) to that restart's start, and from a goal (a shot type with result `success`, or any
   action with result `owngoal`) to the next action. Action times are period-relative (ADR-017).
3. `unavailable`: otherwise. No stoppage split is made (Moura 2016's behaviour), and every row says so.

Only intervals longer than `max_stoppage_s` split a segment. The report counts calls, windows and seconds per source.

**Segments vs windows (D15).** Instantaneous quantities (phase, relative phase, cluster phase, coupling angle) are
computed once per continuous segment (§7.4 step 2). A window selects the samples inside `[start, end)`; per-window
summaries reduce those samples (`np.add.reduceat` over window boundaries), so all windows cost O(N) per series.
Windowed quantities (cross-correlation, spectra, coherence) are computed on each window ∩ segment slice that meets the
method's minimum length and are pooled across slices (§7.8). A window's `n_phases` subdivision assigns each of its
samples to phase `k` when its fractional position lies in `((k−1)/n, k/n]` (Moura's thirds for `n = 3`).

### 7.7 Signal and pair catalog

Signals carry a declared unit (`metres`, `m^2`) and a kind (`positional` for x/y-type, `magnitude` otherwise).
Vector coding runs only on pairs with the same unit (a coupling angle depends on the relative scale; no primary source
normalises mixed units).

| Level | Pairs (per axis where applicable) | Methods | Windows |
|---|---|---|---|
| **L1 team–team** (`team_team`) | same variable, team A vs team B: `centroid_x`, `centroid_y`, `stretch_x`, `stretch_y`, `stretch_index`, `spread`, `convex_hull_area`, `team_length`, `team_width` | relative phase, cross-correlation, coherence, vector coding | all |
| **L1 relative stretch** | SI_A − SI_B for `stretch_x`, `stretch_y` | RSI summary (§7.8.7) | all |
| **L2 cross-variable** (`cross_variable`) | defending `defensive_line_x` ↔ attacking `centroid_x`; defending `compactness_x` ↔ attacking `team_length`; defending ↔ attacking `stretch_x` | relative phase, cross-correlation, coherence, vector coding (all same-unit) | possession windows (role needed; `no_possession_role` elsewhere) |
| **L2 intra-team** (`intra_team`) | a team's `defensive_line_x` ↔ its own `centroid_x` | same | all |
| **L3 dyads** (`dyad`) | 45 intra-team outfield pairs per team, 100 inter-team outfield pairs; x and y | relative phase (incl. % near-in-phase), cross-correlation | period windows (default); all with `dyad_windows="all"` |
| **L4 cluster** | whole team (incl. GK), per axis; player–team; team–team | cluster-phase statistics, SampEn, Cross-SampEn, Pearson | all |
| **Spectral** | every team signal; the possession series | median frequency | all |

Pair order: on possession windows A = attacking team (Moura's x-axis); otherwise A = the team first in
`canonical_id` order (deterministic, not a direction claim). `team_a_id` / `team_b_id` are emitted.

A caller's `pairs` argument is a sequence of frozen `PairSpec(level, signal_a, signal_b, role)` values, `role` ∈
{`canonical`, `attacking_defending`, `same_team`}, validated against `COORD_SIGNALS` and the unit rule; `pairs=None`
means the default catalog above.

### 7.8 Method definitions

#### 7.8.1 Phase and relative phase

- Per continuous segment: subtract the segment mean (Lamb & Stöckl 2014); even-reflection pad each end by
  `min(segment length − 1, round(60 × fs / band_low_cpm))` samples (one period of the band's lower edge, at the
  effective rate fs, capped at `len − 1` because numpy's `reflect` mode needs at least two samples; amended
  2026-10-04, review A-47); `scipy.signal.hilbert`; discard the pad. θ ∈ (−π, π].
  Store the phasor z = e^{iθ} once (D18).
- Phase validity: the share of samples whose instantaneous frequency (d unwrap(θ)/dt) is positive, per series
  (`coord_rp_phase_valid_fraction_a/_b`). Reported, never used to filter.
- φ(t) = wrap(θ_A − θ_B) to (−180°, 180°], evaluated through `z_A · conj(z_B)`.
- Outputs per window: mean resultant length R = |mean e^{iφ}|; circular mean = arg(mean e^{iφ}); circular SD =
  √(−2 ln R) (Mardia & Jupp); 12 histogram fractions with bin centres −180°, −150°, …, 150° and half-width 15°, the
  −180° bin covering [−180°, −165°) ∪ [165°, 180°] (Bourbousson 2010); % near-in-phase = share of |φ| ≤ 30° (Folgado
  2014), computed as `Re(z_A · conj(z_B)) ≥ cos 30°`.

#### 7.8.2 Cross-correlation

- r(ℓ) = Pearson correlation of A[t] and B[t+ℓ] over the overlap t ∈ [max(0, −ℓ), min(N, N − ℓ)), for ℓ ∈ [−L, L],
  L = `xcorr_max_lag_s` × fs (±15 s, Moura 2016; fs the effective rate).
- Pooling across window ∩ segment slices: Fisher z per lag, weighted by (n_ℓ − 3), back-transformed.
- Outputs: max |r| over ℓ, its lag in seconds, the signed r at that lag, r at ℓ = 0. **A positive lag means A leads B.**
  Ties resolve to the smallest |ℓ|, then the negative lag.
- Minimum slice length: `4 × L` samples (the overlap at the maximum lag is at least three quarters of the slice).

#### 7.8.3 Vector coding

- Per window, on consecutive samples inside one segment (a difference never spans a split): Δ_A, Δ_B.
- A sample with |Δ_A| < ε_A **and** |Δ_B| < ε_B is stationary: dropped and counted (`coord_vc_n_stationary`).
- θ = `atan2(Δ_B, Δ_A)` mapped to [0°, 360°) (the four-quadrant form; §3.1 item 1). Positional signals are oriented first
  (§7.10).
- Classification, half-open bins from Moura 2016 Table 1: in-phase [22.5°, 67.5°) ∪ [202.5°, 247.5°); anti-phase
  [112.5°, 157.5°) ∪ [292.5°, 337.5°); A-phase [0°, 22.5°) ∪ [157.5°, 202.5°) ∪ [337.5°, 360°]; B-phase
  [67.5°, 112.5°) ∪ [247.5°, 292.5°). On possession windows A-phase is the attacking-team phase and B-phase the
  defending-team phase.
- Outputs per window and per phase subdivision: the four fractions, circular mean coupling angle, coupling-angle
  variability (circular SD).
- Minimum: 3 non-stationary samples per phase subdivision.

#### 7.8.4 Spectral median frequency

- Per window ∩ segment slice: `scipy.signal.periodogram(detrend="constant", window="boxcar")` (Moura's FFT spectrum,
  mean removed, §3.1 item 2); DC excluded; median frequency = the frequency at which cumulative power reaches half the
  total (linear interpolation between bin edges: each bin's power is spread uniformly over its Δf-wide band centred on
  the bin, so a pure tone on a bin returns exactly that bin's frequency, §9.1 — amended 2026-10-04, review A-06: the
  first implementation placed each bin's cumulative power at the bin centre and read half a bin low); reported in
  cycles·min⁻¹. Slices pool duration-weighted.
- Minimum slice length: 2 periods of `band_low` (§8.2).
- The possession series (1 = team A in possession, 0 = team B) gets the same treatment (Moura 2013 Table II), in a row
  with `team_id` NA and `signal = "possession"`. Between possession windows it holds the last possession, so it is a
  step function that changes only at exchanges (Moura 2013 Eq. 1); samples before the first possession of a period are
  excluded.

#### 7.8.5 Coherence

- Magnitude-squared coherence from Welch cross- and auto-spectra (`scipy.signal.csd`, `scipy.signal.welch`; Hann
  window, `nperseg = welch_segment_s × fs`, 50% overlap). Across slices the **spectra** are pooled (weighted by
  their Welch segment counts) before forming |ΣP_xy|² / (ΣP_xx ΣP_yy), never the coherences.
- Outputs: mean coherence over [`band_low`, `band_high`], the peak frequency (cycles·min⁻¹), and the total segment
  count K. Coherence is biased upward at small K (Carter 1987): K is reported and the surrogate baseline gives the
  chance level. Minimum K: 4.

#### 7.8.6 Cluster phase and entropy

- Per team and axis, over players with a valid phase at t (detection-aware: observed samples only; goalkeeper
  included by default, Duarte 2013): q(t) = arg( mean_k e^{iθ_k(t)} ); φ_k(t) = θ_k(t) − q(t); per window φ̄_k =
  arg(mean_t e^{iφ_k}), ρ_k = |mean_t e^{iφ_k}|; ρ_group,i = | mean_k e^{i(φ_k(t) − φ̄_k)} |; ρ_group = mean_t
  ρ_group,i (Richardson 2012). Samples with n(t) < `min_players` (default 6, the smallest validated group in Richardson
  2012) are excluded and counted (`insufficient_players` when a window has none). A window with fewer than 2 usable
  samples is `too_short` (amended 2026-10-02, owner ruling): with one sample each φ̄_k is that sample's φ_k, so
  ρ_group ≡ 1 for the data and every surrogate draw -- no information, and a percentile decided by float rounding.
- SampEn (Richman & Moorman 2000): templates of length m and m + 1, Chebyshev tolerance r = 0.2 · SD (one global
  scale over the window's usable samples), self-matches excluded; SampEn = −ln(A/B); A = 0 or B = 0 gives NaN with
  `entropy_undefined`. Applied to ρ_group,i per window and to each player's φ_k per window (Duarte). **Templates are
  built WITHIN a continuous run only** — never spanning a detection/stoppage gap (a template that joined non-adjacent
  samples across a gap would count a spurious match); pairs are then counted over the union of all runs' templates
  (amended 2026-10-04, review A-25). **φ_k is a wrapped angle**, so each run is UNWRAPPED to a continuous phase
  before the (linear, Chebyshev) SampEn — consistent with the instantaneous-frequency unwrap in §7.8.1, and because
  SampEn measures the regularity of consecutive increments (unwrap preserves them, circular distance on wrapped values
  would match dynamically-distinct turns of a winding phase). Precondition (checked): a run's wrapped consecutive step
  stays below π; a step at/over π means the analysis rate is too low to unwrap reliably and raises. ρ_group,i is a
  magnitude in [0, 1] (no wrap).
- Cross-SampEn: both series z-scored (one global scale), templates from one matched against the other, r = 0.2
  (Richman & Moorman); templates WITHIN a run only, as above. Applied to the two teams' ρ_group,i series, with their
  Pearson r (Duarte).
- Counting is exact in integers: m = 1 by a sorted 1-D range count; m + 1 by a 2-D dominance count (sort on one
  coordinate, Fenwick tree on the other), O(N log² N) in `numba` with `cKDTree.count_neighbors(p=inf)` as the numpy
  reference. Parity is exact (integers).

#### 7.8.7 Relative stretch index

SI_A − SI_B per axis (Bourbousson 2010). Outputs per window: mean (m), fraction of samples > 0, sign-switch rate per
minute, bimodality coefficient BC = (γ² + 1) / (κ + 3(n−1)²/((n−2)(n−3))) with γ skewness and κ excess kurtosis
(Pfister et al. 2013; BC > 5/9 indicates bimodality).

#### 7.8.8 Divergences from the primary sources

| Paper | Printed or implied | TF-58 | Why |
|---|---|---|---|
| Moura 2016 Eq. 2 | arctan \|Δθ₂/Δθ₁\| | `atan2(Δθ₂, Δθ₁)` | the printed form cannot reach Table 1's bins |
| Moura 2013 | FFT, mean removal unstated | mean removed, DC excluded | otherwise the median collapses to ~0 |
| Bourbousson 2010 | linear mean/SD of wrapped angles | circular statistics | bias near ±180° |
| Duarte 2013 | raw positions | the common zero-phase filtered positions | one preprocessing definition; noise-induced phase slips |
| Duarte 2013 | substitute spliced onto the replaced player | per-player segments | no artificial discontinuity |
| Moura 2016 | cross-correlation over whole halves incl. stoppages | segments split at stoppages > 25 s, Fisher-z pooled | the paper's own named limitation |
| Moura 2013/2016 | 30 Hz video, 7.5 Hz sampling, 0.4 Hz cutoff | 10 Hz analysis; cutoff by residual analysis per provider (default 0.4 Hz) | the paper's procedure applied to our providers |

### 7.9 Surrogates (D6)

- **Time-shift (default).** Within each continuous segment, series B (for pairs), every player's phase series
  independently (cluster phase), or one team's ρ_group,i (team sync) is circularly shifted by s drawn uniformly from
  [τ, N − τ], where τ = `min_shift_s` × fs (≥ the measured decorrelation time, §8.2). Windows then select
  samples from the shifted segment. K = `n_surrogates` = 199. Observed statistic and null alike are computed per
  window ∩ A-segment ∩ B-segment slice (§7.4 step 2, D15), so neither spans an on-pitch-count change (touching
  segments split; review M-13, 2026-10-02). Only the segments holding a slice the statistic reads are shifted; one of
  those shorter than 2τ + 1 leaves the null undefined (`segment_too_short`, plan Task 17), while a segment that merely
  overlaps the window -- holding no slice the statistic reads -- takes no part (2026-10-03).
- **IAAFT (opt-in, `surrogate_method="iaaft"`).** Iterated amplitude-adjusted Fourier transform (Schreiber & Schmitz
  2000), `iaaft_max_iter = 100`, per segment. **Implemented for the PAIR families only** (relative phase,
  cross-correlation, vector coding, coherence); the cluster-phase and team-sync nulls implement the time shift only
  and REFUSE `iaaft` fail-loud (`NotImplementedError`) rather than silently substitute the shift and report
  `computed` — a surrogate method is an inferential choice (amended 2026-10-04, review A-17). The per-family domain is
  single-sourced (`SURROGATE_METHODS_BY_FAMILY`), whose union is exactly the accepted `surrogate_method` values.
- **Estimator identity.** Every surrogate statistic is computed by exactly the estimator used for the observed value.
  The accelerations are algebraic identities, each parity-tested against the direct computation:
  - R and circular mean for all K shifts from one circular cross-correlation of the phasors;
  - % near-in-phase through phasor products (direct O(N·K), vectorised, `numba` optional);
  - cross-correlation surrogates from one circular FFT cross-correlation plus an exact correction for the |ℓ|
    wrapped terms per lag, with means and variances from prefix sums (`numba` optional);
  - cluster-phase surrogates by per-player shifted phasors.

  An identity declines -- the direct null scores the slice -- where a B segment it touches holds a non-finite value
  (the FFT would spread it to every shift); relative phase runs its identity only where a fixed size rule says it is
  cheaper (ruling A, `RP_IDENTITY_CROSSOVER`); both branches are parity-equal, so the rule chooses speed, never a
  result.
- **Determinism.** Draws come from `np.random.SeedSequence(entropy=surrogate_seed, spawn_key=<digest of the canonical
  key (game, period, segment, pair or team, method)>)`: identical regardless of processing order, worker slicing or
  resume.
- **Outputs** per surrogated metric: `*_surrogate_mean`; `*_percentile` = (#{s < obs} + ½ #{s = obs}) / K;
  `*_excess` = obs − surrogate mean. Surrogated metrics: R, % near-in-phase, max |r|, vector-coding in-phase and
  anti-phase fractions, band-mean coherence, ρ_group mean, team-sync Pearson r. `n_surrogates = 0` disables them (the
  D2 passes).

### 7.10 Orientation and identity

- Direction comes only from the `GoalMap` (ADR-055, ADR-051 D3), never from team identity.
- Scale-free outputs (relative phase statistics, cross-correlation, coherence, cluster phase, vector-coding pattern
  fractions) are invariant to a common reflection of both signals. Positional signals are nevertheless oriented so
  every direction-bearing output (the vector-coding mean coupling angle) has a fixed meaning: positional x and y signals
  of a pair are expressed in the **reference team's** attacking direction through the ADR-051 goal-relative point
  reflection (`tracking._geometry.to_goal_relative_x/_y`, array-applied), both signals of a pair through the same
  transform. Reference team = attacking team on possession windows, team A otherwise.
- Pinned by a physical mirror test (reflect the frames and swap direction labels → every output identical, including
  the mean coupling angle) and an identity test (relabel team ids → outputs identical up to the relabel).

### 7.11 Detection-aware providers (D9)

- Player-level methods (dyads, player–team, the phases feeding cluster phase) use observed samples only (§7.4).
- Team-level signals use every on-pitch player's best-estimate position; dropping off-camera players would bias every
  team signal toward the ball side. Every sample carries `observed_fraction`; every output row carries
  `coord_observed_fraction_a/_b` (window means) and `coord_detection_source` ∈ {`fully_observed`, `detection_aware`}.
- A window whose mean observed fraction is below `min_observed_fraction` (Tier B, per metric family) emits NaN with
  `insufficient_detection`, never a fabricated value (ADR-055).
- The in-cycle occlusion leg quantifies the effect (§8.3).

### 7.12 Output tables

All schemas are plain dicts in `_columns.py`, the single source every gate iterates. Metric columns are prefixed
`coord_`. Keys and tokens are `object`; counts `Int64`; metrics `float64`. Ids go through `id_compat`
(`canonical_id` in pair keys, `align_join_keys` for merges, `restore_id_dtype` on output).

| Table | Grain (keys) | Metric groups |
|---|---|---|
| `COORD_WINDOW` | the window contract (§7.6) | — |
| `COORD_PAIR` | `game_id, period_id, window_kind, window_id, level, signal_a, signal_b, axis, team_a_id, team_b_id, player_a_id, player_b_id` | relative phase (`rp_*`: mean_deg, circ_sd_deg, resultant_length, pct_near_in_phase, 12 `hist_bin_*`, phase_valid_fraction_a/_b), cross-correlation (`xc_*`: max_abs_r, lag_s, r_at_max, r_lag0), vector coding (`vc_*`: pct_in_phase, pct_anti_phase, pct_a_phase, pct_b_phase, mean_angle_deg, angle_variability_deg, n_stationary), coherence (`coh_*`: band_mean, peak_freq_cpm, n_segments), surrogate triples, coverage (duration_s, n_samples, n_segments, observed_fraction_a/_b), one `*_source` per method group, `detection_source`, `stoppage_source` |
| `COORD_PAIR_PHASE` | `COORD_PAIR` keys + `phase_index` | relative phase and vector coding per `n_phases` subdivision (no surrogates) |
| `COORD_SPECTRAL` | `game_id, period_id, window_kind, window_id, team_id (nullable), signal` | `median_freq_cpm`, duration_s, n_segments, source |
| `COORD_CLUSTER_TEAM` | `game_id, period_id, window_kind, window_id, team_id, axis` | `rho_group_mean`, `rho_group_sd`, `rho_group_sampen`, `n_players_mean`, surrogate triple, coverage, source |
| `COORD_CLUSTER_PLAYER` | `… team_id, player_id, axis` | `phi_mean_deg`, `rho_k`, `phi_sd_deg`, `phi_sampen`, `on_pitch_s`, source |
| `COORD_TEAM_SYNC` | `game_id, period_id, window_kind, window_id, axis` | `team_sync_pearson_r`, `team_sync_cross_sampen`, surrogate triple, source |
| `COORD_RSI` | `game_id, period_id, window_kind, window_id, axis` | `rsi_mean_m`, `rsi_fraction_positive`, `rsi_switch_rate_per_min`, `rsi_bimodality_coefficient`, source |

`COORD_RSI` is a session call: RSI was approved as an L1 output (D11) without a named table; its grain differs from the
pair table's. Every metric table carries `coord_detection_source` and `coord_stoppage_source` (D20) as provenance
columns (tokens, not metrics; excluded from the glossary, the `TR_HULL_SOURCE` precedent). Each metric table exports `*_KEYS`, `*_METRIC_COLUMNS` and `*_COLUMNS` and registers its own
`METRIC_CONTRACTS` family (`coordination_pair`, `coordination_pair_phase`, `coordination_spectral`,
`coordination_cluster_team`, `coordination_cluster_player`, `coordination_team_sync`, `coordination_rsi`).
`compute_coordination_series` returns a long table: `game_id, period_id, segment_id, time_s, kind, <pair or team
keys>, value` (no contract: a raw primitive, not a mart).

### 7.13 Degradation taxonomy, report and warnings

`COORD_SOURCE_VALUES` = {`scored`, `too_short`, `insufficient_detection`, `insufficient_players`,
`goal_end_unresolved`, `not_commensurate`, `no_possession_role`, `degenerate_constant`, `entropy_undefined`}.
`entropy_undefined` completes the set for SampEn (A or B = 0); it is a session call. Every metric group has its own
`*_source` column, so one row can be `scored` for relative phase and `too_short` for coherence. Degraded rows are kept
(ADR-042).

`CoordinationReport` (frozen): the params; windows in, scored and dropped per reason; segments; samples dropped as
stationary, unobserved or below `min_players`; per method, rows per source token; the stoppage source used and the
dead seconds and split count it produced (D20). Conservation (windows in = scored +
Σ reasons; per method, rows = Σ tokens) is asserted in tests.

`CoordinationCoverageWarning` (a new category subclassing none of the existing ones, test-enforced) fires with
`stacklevel=2` when the dropped share of windows in a call exceeds `coverage_warn_fraction` (default 0.25): once per
public family compute, and once per orchestrator call (which suppresses its families'), attributed to the caller.
The same category names any (game, period) whose actions `possession_windows_from_actions` skipped because the frames
do not cover that period (de-risk finding, 2026-10-01).

### 7.14 Numeric contract and parameters

- float64 everywhere inside kernels; coordinate reads upcast at the boundary (ADR-106). TF-58 never writes coordinates
  back into frame storage: it uses the array kernels, so F1b's float32 storage rounding never enters a coupling result.
- `CoordinationParams` fields, grouped by tier (§8.1): Tier A — `xcorr_max_lag_s=15.0`, `n_phases=3`,
  `near_in_phase_deg=30.0`, `max_stoppage_s=25.0`, `sampen_m=1`, `sampen_r_sd=0.2`, `butterworth_order=3`,
  `analysis_hz=10.0`; conventions — `min_players=6`, `n_surrogates=199`, `iaaft_max_iter=100`,
  `coverage_warn_fraction=0.25`; Tier B (per provider, generated) —
  `butterworth_cutoff_hz` (default 0.4), `max_detection_gap_s`, `min_observed_fraction` (per metric family),
  `vc_epsilon` (per signal), `min_shift_s` (per signal), `band_low_cpm`, `band_high_cpm`, `welch_segment_s`,
  `possession_gap_s`; other — `surrogate_method="time_shift"`, `surrogate_seed=0`, `include_goalkeeper` (per method).
  `__post_init__` rejects impossible combinations. `for_provider` merges
  `_provider_params_generated.PROVIDER_COORDINATION_PARAMS` (empty in commit 1). Commit 1 is never released alone: the
  release carries commit 2's derived values.

### 7.15 Performance design

| # | Measure | Effect (per match at 10 Hz) |
|---|---|---|
| R1 | vectorised collective kernel + angular-gap hull | ~190 s → ~2 s |
| R2 | vectorised back line | ~16 s → < 1 s |
| R3 | phasor representation; surrogate statistics by circular cross-correlation | ~93 s → ~9 s (numpy) |
| R4 | D2 cached per-level passes, surrogates off; selection over cached shards | sweep ≈ 11 cached passes |
| R5 | phases once per segment; windows reduce by `reduceat` | O(N) per series across all windows |
| R6 | optional `numba` for the near-in-phase surrogate count, the exact cross-correlation wrap correction, and 2-D dominance counting | SampEn ~26 s → < 1 s; surrogates ~9 s → ~2 s |
| R7 | dyads on period windows by default | output ÷ ~100 |
| R8 | structural guards; per-stage timing counters in the driver manifest | regressions fail CI |

**`numba` conventions** (shared with native DAS, ADR-076/008): explicit float64/int64 signatures, serial by default,
no `prange` unless parity-gated; the numpy path is the reference; results never depend on whether `numba` is
installed (integer counts are exact; float sums follow the reference order). The file is
`coordination/_kernels/_numba.py`; the CI numba cache key (`.github/workflows/ci.yml:91`, pattern
`silly_kicks/tracking/**/*_numba*.py`) is extended to `silly_kicks/coordination/**/_numba*.py`, which
`tests/test_ci_shard_wiring.py::test_numba_cache_key_covers_all_njit_files` enforces.

**Cost model.** Preparation ~5 s; dyads ~2–9 s; SampEn < 1–26 s; the rest ~5 s → ≈ 12–45 s per match single-core.
D3 full pass ≈ 3–12 h single-core, ≈ 15–45 min on 16 workers (disjoint slices, `scripts/_partition.providers_for_slice`),
≈ 2 GB peak per worker. The plan measures one real match per provider (SkillCorner 10 Hz, IDSSE 25 Hz, Gradient
Sports) before any corpus launch and records it.

**Measured (2026-09-29, the official D3 metrics pass, one match per provider, k = 199; goal 6 restated on it).**
Per match with `numba` / numpy: SkillCorner 18.1 / 25.2 s, IDSSE 113.8 / 437.2 s, GradientSports 131.3 / 453.5 s
(k = 0 floor 13.7 / 32.0 / 34.5 s). Corpus (909 SkillCorner + 64 GradientSports + 7 IDSSE matches): 7.1 h
single-core, 27 min on 16 workers with `numba`; 15.3 h single-core, 57 min numpy-only. Memory: a worker loading one
full-tracking match held ~7-8 GB resident (the local no-flip trial, 2026-09-29), not ~2 GB, so a corpus run sizes its
worker count by memory as well as by cores.

### 7.16 Seam changes: every caller, with evidence

| Changed symbol | Callers (`grep` at `05cfa56`) | Effect on each |
|---|---|---|
| `compute_team_shape` (delegation) | `tracking/features.py:2438`, `:2572` (`add_team_shape`); `restdefense/_compute.py:83` | identical output except `convex_hull_area` ≤ 1e-9 relative; restdefense reads only `team_length` from it (`_TS_COLS`, `restdefense/_compute.py:55`), so restdefense is byte-identical |
| `compute_defensive_line` (delegation) | `causal/_confounders.py:157`; `restdefense/_compute.py:81`; `tracking/_kernels.py:863`; `tracking/_off_ball_runs.py:316` | byte-identical (gated) |
| `smooth_frames` (new method) | `tracking/gradientsports.py:201`, `kloppy.py:265`, `metrica.py:248`, `skillcorner.py:389`, `sportec.py:208` | none: they pass a `config` whose default method stays `"savgol"` |
| `PreprocessConfig` (two new fields) | every `PreprocessConfig(...)` construction; `_provider_defaults_generated.py` via `scripts/regenerate_provider_defaults.py` | additive fields with defaults; the generator and `tests/test_preprocess_baseline_integrity.py` re-verified |
| `icc1` / `split_half_reliability` / `type_ii_slope` / `compare_providers` (moved) | `scripts/validate_team_kpi_reliability.py`; `tests/scripts/test_team_kpi_reliability.py:14–19`; `scripts/validate_gk_decision.py` (`icc1` copy); `tests/scripts/test_gk_decision_battery_kernels.py:24–35` | re-imported under the same names; identical code |
| `validate_corpus_visibility` (moved) | `scripts/train_ghost_gk.py:503`; `tests/scripts/test_trainer_cache_and_providers.py:185–188, 194–195` (load `train_ghost_gk.py` by file path and call `t.validate_corpus_visibility(...)` directly; the monkeypatch at `:184` targets `GhostGkModel.fit`, not this function) | `train_ghost_gk` re-imports the name, so the direct module-attribute calls still resolve; a test pins that the re-import exists |
| `tests/test_metric_contracts.py` completeness | the test itself | re-keyed per exported constant; existing families unchanged |
| `silly_kicks.calibration.stage1_config` / `stage2_config` / `xt_bandwidth_config` (ruthless 0.7.0 migration, D21) | `scripts/calibrate_tracking_defaults.py:66, :69`; `scripts/calibrate_xt_bandwidth.py:41`; `scripts/check_stage1_argmax.py:91`; `tests/calibration/test_spaces.py:8`; the doctests at `calibration/_spaces.py:10, :34, :58, :101` | each builder gains a required keyword `objective_id: str`, forwarded to `StoreConfig(objective_id=...)`; every caller passes the D21 identity; doctests and the test pass a literal |
| `StoreConfig(...)` in the trainers (D21) | `scripts/train_xshot_occurrence.py:202`; `scripts/train_xcross_attempt.py:270`; `tests/tracking/test_xshot_occurrence_integration.py:69` | pass the D21 identity (the test passes a literal) |
| `tracking/_provider_visibility.py` (two additions, D20) | existing names `validate_provider`, `assert_detection_aware_visibility`, `_DETECTION_AWARE_PROVIDERS`, `_FULLY_OBSERVED_PROVIDERS` and their callers | additive; no existing name changes; the module-alias rule for `_ghost_gk` (ADR-069) is untouched |

The plan re-runs this enumeration at implementation time and at the rebase (the sweep is the floor).

---

## 8. Calibration and validation

### 8.1 Tiers

- **Tier A — fixed by a primary source; never tuned:** `xcorr_max_lag_s` (Moura 2016), `n_phases` (Moura 2016),
  `near_in_phase_deg` (Folgado 2014), `max_stoppage_s` (Duarte 2013), `sampen_m` and `sampen_r_sd` (Duarte 2013),
  `butterworth_order` (Moura 2013/2016), the default `analysis_hz = 10` (Duarte 2013; the effective rate follows the
  §7.4 step 4 rule), and the module constants (Moura 2016 Table 1 bins; Bourbousson 2010's 12 × 30° histogram).
- **Conventions — fixed values no paper supplies, each with its rationale; not tuned:** `n_surrogates = 199` (the
  §7.9 percentile divides by K = `n_surrogates`, so K = 199 gives a resolution of 1/K ≈ 0.005 — fine enough to
  separate a coupled pair's percentile from chance at the reported 0.01 and 0.05 levels; the earlier 1/(K+1)
  "exact ranks" rationale assumed a (K+1)-divisor the code does not use; amended 2026-10-04, review A-56);
  `iaaft_max_iter = 100` (a cap on Schreiber & Schmitz's iterate-to-convergence loop; non-convergence is reported);
  `min_players = 6` (the smallest group size Richardson 2012 validated cluster-phase on); `coverage_warn_fraction =
  0.25`; the dual-pass cutoff correction (Winter 2009).
- **Tier B — derived per provider by deterministic procedures (D1 driver):** §8.2.
- **Tier C — gated sensitivity (D2 driver):** §8.4. A Tier-B value is replaced only when an alternative beats it on
  held-out reliability by more than the noise floor **and** replication still passes.

### 8.2 D1 `scripts/derive_coordination_params.py` (Tier B)

Pass A (raw positions):
- **`butterworth_cutoff_hz`:** residual analysis (Winter 2009) per player-axis run: R(f) = RMS(x − x̂_f) over
  f ∈ [0.1, 5.0] Hz in 0.05 Hz steps; fit a line to the linear high-frequency tail (the noise region); the cutoff is
  where R(f) meets that line's intercept. Per provider: the median over runs.

Pass B (with the pass-A cutoffs):
- **`vc_epsilon`** per signal: 1.4826 · MAD of the first differences of (resampled raw − filtered) signal, the
  noise-induced change per sample.
- **`min_shift_s`** per signal: the 95th percentile, over match-halves, of the first zero crossing of the
  autocorrelation.
- **`band_low_cpm`, `band_high_cpm`:** the 5th and 95th percentiles of team-signal median frequencies.
- **`welch_segment_s`:** the SHORTEST segment giving frequency resolution ≤ (band width)/4, rounded up to the next
  10 s; the ≥ 8-Welch-segments-per-half (50% overlap) check is RECORDED, not binding (amended 2026-10-04, review
  A-52: the shortest length that meets the resolution requirement maximises the segment count K, which lowers
  coherence's small-K upward bias ≈ 1/K and its variance, and keeps coverage — a longer segment drops every
  continuous stretch shorter than itself; the plan and code already compute the shortest, so this corrects the
  spec to match). Resolution is the requirement; the recorded flag is the shortfall diagnostic.
- **`possession_gap_s`:** on event + tracking matches, the value in {0.2, 0.4, …, 3.0} s maximising the boundary F1 of
  frames-only possession spells against event possessions (the TF-52 possession-ground-truth idiom).
- **Occlusion leg (D9)** on GS 64 + IDSSE 7 (fully observed): a camera field of view centred on the ball x (clamped
  to the pitch), longitudinal width W calibrated so mean outfield detection equals SkillCorner's measured 66.6%; the
  resulting goalkeeper rate is reported against SkillCorner's 19.6% as a check. Masked positions are filled by linear
  interpolation between detections (an approximation of SkillCorner's own model, labelled as one). Every metric is
  compared with its full-observation truth.
  - `min_observed_fraction` is **consumed per family** (amended 2026-10-04, owner ruling on the C.8.6 grain + review
    P-8 / batch-3): the library's detection gate is inherently per-family (`params.min_observed_fraction[family]`, one
    float per method family), so the consumed threshold is the **MAX over the family's constructs** of each governed
    construct's smallest qualifying share (**fail-closed** — MAX ≥ every construct's own share, so each construct is
    gated at least as strictly as its own curve requires). A construct's qualifying share is the smallest
    observed-fraction bin whose median absolute error is at most 0.5 × that construct's between-match SD under full
    observation (the **circular** SD and wrapped angular error for the circular-mean constructs, §8.5). A construct's
    curve is **estimable** iff every occlusion decile bin it needs carries at least `OCCLUSION_MIN_MATCHES_PER_BIN`
    matches (a named constant in `scripts/_coordination_thresholds.py`, pre-registered as `= 5` from the 71-match
    GS+IDSSE occlusion corpus and recorded in `derivation.json`) **and** the curve has a finite, unique crossing of its
    0.5·SD bar. A construct whose curve never clears the bar below 1.0 falls its family back to **1.0 (full-observation
    only)**, recorded as a **FINDING** (such a family yields ~no SkillCorner rows at broadcast detection — stated, never
    lowered to manufacture rows). `derivation.json` records, per construct, the curve + n/CI per bin + the estimable
    verdict + the binding construct; and, as an **over-restriction diagnostic**, how much more strictly the family MAX
    gates than each construct's own share (and any family driven to 1.0 by a single construct), which is the owner's
    decision input for the per-construct-consumption follow-up below. The **public** `CoordinationParams.min_observed_
    fraction` therefore stays `Mapping[family → float]`; per-construct *consumption* (a strictly-less-conservative gate
    behind a private per-construct table) is an owner-gated, ADR-costed follow-up, taken only if the over-restriction
    diagnostic shows the retention loss is material — it changes no correctness (MAX is already fail-closed).
  - `max_detection_gap_s`: the longest gap whose bridged-position RMSE is at most 2 × the residual-analysis noise RMS.

D1 **generates** `coordination/_provider_params_generated.py`; `tests/coordination/test_provider_params_generated.py`
proves the generator reproduces the committed file byte-for-byte (the `regenerate_provider_defaults.py` precedent).
Output: `<--out>/derivation.json` and `<--out>/_provider_params_generated.py` (the artifact handoff, below); commit
2 copies both into `docs/research/tf58_team_coordination/` and the package.

### 8.3 Shared driver discipline (D1, D2, D3)

`scripts/_driver.for_each(load=)` over `list_match_refs` (resume before load, ADR-052), `assert_conservation`,
`_require_injective`, `require_clean_tree(git_provenance())`, `declare_inputs`, `--allow-dirty` for development only;
enrolled in `ARTIFACT_DRIVERS` (`tests/scripts/test_provenance_wiring.py`); pre-flight
`scripts/_corpus_visibility.validate_corpus_visibility` (ADR-069 Layer 2); aggregate-only outputs with ADR-038
visibility labelling (NDA-tier aggregates allowed); disjoint worker slices via `providers_for_slice`; per-stage timing
counters in the manifest (R8). The full corpus is used; it is never shrunk.

- **Partitioned passes (review B-1, 2026-10-02).** Each worker writes only its share of a pass (`<pass>.<tag>.parquet`
  + a manifest naming every key it was handed and what became of it). Every corpus-wide step combines ALL shares and
  refuses a missing worker, a failed key, a key handed to two workers, mixed generations or commits, workers that
  disagree on a declared field, or a population other than `--corpus-json` (the full, unsplit listing).
- **Artifact handoff (owner ruling M-5, 2026-10-02).** No driver writes the repo on the DGX. D1's reduce writes
  `derivation.json` and the generated module into its `--out`; D2 and D3 read the artifact explicitly (`--derivation`;
  D3 also `--calibration`), compute with exactly the maps the codegen renders, and carry its sha256 in every shard
  token, manifest and artifact; D2's confirm writes `calibration.json` and the final module into its `--out`. Commit
  2 copies the artifacts into the repo.
- **Layer-2 pre-flight for a raw-artifact corpus (owner ruling 2026-10-03).** `validate_corpus_visibility` reads
  parquet shard metadata, and the TF-58 drivers read no parquet shards (they load each match from raw provider
  files through the pining loader), so the ADR-069 Layer 2 takes this form: every corpus pass, BEFORE any compute,
  loads one match of each detection-aware provider in its slice through its own loader and refuses with the
  ADR-069 remedy if that match's `visibility` flag was discarded (`_coordination_corpus.visibility_preflight`; a
  match that fails to load for another reason is skipped for the next). A single match's hole is still refused per
  match by `detected_mask`'s all-null trap, recorded as failed, and refused by every combine.

### 8.4 D2 `scripts/calibrate_coordination.py` (Tier C, D14)

- **Parameters swept, one at a time, around each Tier-B value:** `butterworth_cutoff_hz` (7 levels),
  `max_detection_gap_s` (5), `min_observed_fraction` (6), `welch_segment_s` (5), `vc_epsilon` (5, as multiples of the
  derived value). Rule-derived quantities (resample rate, decorrelation times, minimum lengths) are not swept.
- **Layer (a), heavy:** one `for_each` corpus pass per preparation level (cutoff, detection gap: ~11 passes), with
  `n_surrogates=0`, writing per-match metric shards under a generation token that includes the level. Post-preparation
  levels reuse the cached prepared signals of the Tier-B baseline level.
- **Layer (b), selection on ruthless:** `GridSearchStrategy(GridConfig(design="one_at_a_time", baseline=<Tier-B
  values>, param_space=<levels as Choice>))`. The objective reads the cached shards and returns **held-out
  team-discrimination reliability** — D2's calibration purpose: how well a candidate's parameters make a metric
  separate teams — computed over the **constructs** the parameter affects (§8.5 grain) with the **same per-construct
  estimator D3 reports**: linear ICC(1) (`scripts/_reliability.icc1`) for a linear construct and rotation-invariant
  **circular** reliability (§8.5) for a circular-mean construct, so D2 never selects on a linear ICC of a circular mean
  (amended 2026-10-04, reviews P-2/R-3; aligns to the approved plan's A-35). A construct whose cell is `unmeasurable`
  (§8.5 power verdict) is excluded from the mean, never pooled to rescue power; the cross-validation folds are keyed by
  index, and a fold a construct cannot score contributes NaN (kept, not dropped — A-34). D2's objective GRAIN
  (team discrimination across matches) differs from D3's binding-reliability GRAIN (within-match split-half internal
  consistency); that difference is a **ruled reconciliation**, stated verbatim in `metrics.json` / `report.md` (not an
  optional disclosure, review P-2). D2's swept parameters affect only linear magnitude / fraction constructs, so no
  circular-mean construct enters D2's objective today — the circular estimator is wired so one could without a
  definition mismatch. Cross-validated by match (`match_cv_splits`, SE by `cv_standard_error`). A missing shard raises
  `FatalEvaluationError` (never a partial corpus). Per parameter: `select_recommended_point` + `exceeds_noise_floor` +
  `MIN_EFFECT_SIZE`.
- **Joint confirmation:** `GridConfig(design="points", points=[<joint selection>])`; the joint setting must beat the
  Tier-B baseline beyond the noise floor and pass H1–H7; otherwise the Tier-B values stand (gated fallback, recorded).
- **Resume identity (D21):** each grid run uses `store=StoreConfig(kind="sqlite", path=<per-run path under the D2
  output>, objective_id=<the shard-generation token>)`, where the token is the name `scripts/_driver.generation_dir`
  derives from the pass's `token_inputs`. A regenerated shard set therefore changes the id and ruthless refuses to
  resume from stale scores. The OAT run and the confirmation run use different store paths (their configs differ and
  ruthless guards config identity).
- **Level types:** levels, baseline and points are native Python `float`s (converted with `float(...)` from the D1
  derivation), because ruthless 0.7.0 rejects numpy scalars at construction (fingerprintability) and compares levels
  type-strictly.
- Output: `<--out>/calibration.json` and `<--out>/_provider_params_generated.py` (§8.3 artifact handoff). The
  selections are serialised by a generic `_selection_dict`, faithful to ADR-060: `build_selection_artifact` is
  xt-bandwidth-specific (it hard-codes `beta`/`gamma`), so D2 does not reuse it (TF58-IMPL-06, amended 2026-10-03).
  The module is the calibrated one when the gate clears, else byte-identical to D1's (the Tier-B values stand). The
  OAT selection is written to `<--out>/oat.json`; a joint moving ≥ 2 parameters is prepared by its own partitioned
  layer (`--layer joint`).
- **Prerequisite:** ruthless-efficiency 0.7.0, interface pinned in the handoff's section 4. `pyproject.toml`
  `[calibration]` and `[train]` extras raise `ruthless-efficiency[optuna]>=0.7.0`.

### 8.5 D3 `scripts/validate_team_coordination.py` (the in-cycle artifact; reported, not gated)

Runs the final parameters on the full corpus. Hypotheses and thresholds are pre-registered in
`scripts/_coordination_thresholds.py` and referenced, never inlined:

| # | Source | Hypothesis | Test and threshold |
|---|---|---|---|
| H1 | Bourbousson 2010 (basketball: longitudinal 6.9° ± 10.5°, lateral −4.9° ± 39.9°) | team centroid relative phase is more stable longitudinally than laterally, and near in-phase | paired sign test on match-halves, R_x > R_y, one-sided p < 0.01; pooled longitudinal circular mean within ±30° of 0° |
| H2 | Moura 2016 (r 0.41 ± 0.09 / 0.36 ± 0.13; lag 0.33 ± 0.30 / 0.21 ± 0.36 s) | spread cross-correlation is positive with a short lag | signed r at max \|r\| positive in ≥ 70% of team-pair halves; median \|lag\| ≤ 1.0 s |
| H3 | Moura 2016 | early-third anti-phase and attacking-team-phase fractions are higher when the possession ends in a shot than in a tackle | one-sided Wilcoxon rank-sum, p < 0.05 each (event providers) |
| H4 | Moura 2013 (0.63 ± 0.10 → 0.47 ± 0.14 area; 0.60 ± 0.14 → 0.46 ± 0.16 spread) | median frequency < 1 cycle·min⁻¹; first half > second half | < 1 in ≥ 95% of team-halves; paired one-sided Wilcoxon signed-rank p < 0.01 |
| H5 | Duarte 2013 (0.89 ± 0.12 longitudinal, 0.73 ± 0.16 lateral) | ρ_group longitudinal > lateral; possession has no effect | paired sign test p < 0.01; TOST equivalence in vs out of possession within ±0.05, α = 0.05 |
| H6 | Folgado 2014 | dyad near-in-phase distribution | descriptive (quantiles per axis) |
| H7 | Bourbousson 2010 | RSI is bimodal; RSI sign switches follow possession changes | BC > 5/9 in ≥ 50% of team-halves; share of switches within 10 s after a possession change above its time-shift surrogate 95th percentile |

Also reported — **per construct and provider** (amended 2026-10-04 per the owner's batch-3 ruling and review P-1; the
author's "one representative metric per family" was the defect this replaces). A **construct** is a metric column × the
keys that define *what* it measures: pair families `level` + `signal_a` + `signal_b` + `axis`; spectral `signal`;
cluster / team-sync / RSI `axis`; phase-row metrics also `phase_index`. This is ~2000 cells.

- **Reliability** — each construct's **binding** reliability with a 95% CI and an explicit **power verdict**. The
  binding definition is team-discrimination ICC(1) (`scripts/_reliability.icc1`) for a **linear** construct, and
  **rotation-invariant circular reliability** `1 − (within-group circular variance / total circular variance)` via mean
  resultant lengths for the three **circular-mean** constructs (`coord_rp_mean_deg`, `coord_vc_mean_angle_deg`,
  `coord_phi_mean_deg`) — a plain ICC on a circular mean is origin-dependent, so the linear estimator is simply wrong
  for them; correcting it is a strengthening this spec now carries (a latent §8.5 defect), not a plan-level override.
  The cos/sin-component ICCs are kept only as **diagnostics**, with the origin pinned in `derivation.json`. The
  dispersion columns (`coord_rp_circ_sd_deg`, `coord_vc_angle_variability_deg`, `coord_phi_sd_deg`) are magnitudes and
  stay **linear**. The circular columns are declared in a **gated registry** with an anti-rot meta-test (§9.4). A
  construct's cell is terminal **"unmeasurable"** — never pooled up a level to rescue power — when
  `n_groups < RELIABILITY_MIN_N_GROUPS` (`unmeasurable_reason="n<min"`), its reliability CI half-width exceeds
  `RELIABILITY_MAX_CI_HALFWIDTH` (`"ci_too_wide"`, so a nominally-powered but hopelessly imprecise cell is still
  terminal), or — for a circular-mean construct — its mean resultant length is below `CIRCULAR_RELIABILITY_MIN_RBAR`
  (`"Rbar->0"`, circular reliability being undefined as concentration → 0). These three thresholds are **named,
  pre-registered constants** in `scripts/_coordination_thresholds.py`, fixed here before the corpus is scored so `min`
  is never a post-hoc free parameter across the ~2000 cells (review PLAN-02). They are pre-registered as
  `RELIABILITY_MIN_N_GROUPS = 30`, `RELIABILITY_MAX_CI_HALFWIDTH = 0.25`, `CIRCULAR_RELIABILITY_MIN_RBAR = 0.10`
  (owner-ratifiable here), and the `unmeasurable_reason` vocabulary is exactly `{n<min, ci_too_wide, Rbar->0}`.
- **Split-mode reliability** — within-match split-half with the **Spearman–Brown** prophecy correction and the
  **half-length labelled** (A-53). This is internal consistency and an **upper bound** on true match-to-match
  reliability (honesty line 1).
- **Association slope** — reported as **SMA/RMA** (standardised / reduced major axis), not an OLS "Type-II slope", with
  the value pinned (A-36).
- **Cross-provider poolability** (`compare_providers`) — same per-construct shape, CI and power verdict.
- **Units.** Each construct is grouped on its measured **entity**: cluster-player metrics on the player, dyad metrics
  on the unordered player pair (canonical via `id_compat`, gated), team-level metrics on the team; the unit is declared
  per construct. Keys are **per-(match, entity)** — player ids are match-local on anonymised corpora — so the retest
  axis is match-halves and the group is `(match, player)` / `(match, unordered pair)` / `(match, team)`, two
  observations each. A global player id is **never** keyed across matches on anonymised data; an underpowered unit is
  unmeasurable, never pooled to team. Two **honesty lines** appear verbatim in both the `metrics.json` provenance and
  `report.md`: (1) across-halves = within-match split-half (shared opponent / setup) = internal consistency = an
  **upper bound** on true match-to-match reliability; (2) cross-match player reliability is **unmeasurable** on
  anonymised corpora (no roster linkage) — stable-roster providers are noted as a future extension.
- **Decile stratification** — each construct's distribution by SkillCorner observed-fraction decile (circular median +
  circular spread for the circular-mean constructs).
- **Occlusion** — the final per-construct occlusion error curves (§8.2 grain and estimability criterion).
- **Stoppage leg (D20)** — on GS and IDSSE, the precision, recall and interval overlap of event-derived stoppages
  longer than 25 s against true `ball_state`, and each metric's change between ball-state splitting, event splitting and
  no splitting.

A failed hypothesis is a finding recorded in the ADR for the owner to rule on; nothing is auto-dropped. The D2-gate
population dependence of H1–H7 is stated in `metrics.json` / `report.md` (review A-35). Outputs (into `<--out>`, copied
in by commit 2, §8.3): `docs/research/tf58_team_coordination/metrics.json` — top-level provenance (`schema_version`,
`run_commit` / `run_tree_dirty` / `run_tree_state` / `run_tree_hash`, `corpus_visibility`, `input_contract`, `params`
with the D1/D2 sha256, `population`, `honesty` as the two lines above) over a `constructs` **list** of per-construct
cells (`column`, `construct_key`, `unit`, `kind` ∈ {linear, circular}, `reliability` {value, ci, estimator, n_groups,
n_obs, power, unmeasurable_reason}, `diagnostics` {icc_cos, icc_sin, origin_deg} for circular else {}, `poolability`,
`split_mode`, `deciles`) — and `report.md`, which carries a **descriptive** per-column summary (median + range of
`reliability.value` across that column's constructs, with the count of `unmeasurable` cells), explicitly **not**
presented as "the column's ICC", plus the two honesty lines verbatim. Any new emitted column registers in
`metric_contracts` or the ADR-098 completeness gate fails.

---

## 9. Tests and CI

TDD throughout; detection lands before the fix (ADR-051); every band tested from both sides; every counterfactual
asserts non-vacuity.

### 9.1 Kernel ground truth (`tests/coordination/kernels/`)

- Phase and relative phase: sinusoids with a known offset → circular mean = offset, R ≈ 1; anti-phase → 180°; added
  noise lowers R monotonically. Pre-Hilbert non-vacuity: mean-centring is load-bearing (an un-centred Hilbert on a
  DC-offset signal is badly phase-biased; the kernel's centring fixes it), and reflect-padding reduces edge error on
  a representative slow (~1-minute rhythm, non-integer-cycle) window — its edge benefit is regime-dependent, not
  universal (measured ~2× on a 1.35-cycle window; near-neutral for integer-cycle sinusoids, which are artificially
  FFT-periodic). [TF-58 Task 8: corrected from "reflect-padding reduces edge error versus no padding" after
  measurement; owner-approved.]
- Circular statistics: parity with `scipy.stats.circmean` / `circstd`; histogram bin edges tested on both sides,
  including the ±180° wrap.
- Rotation-invariant circular reliability (§8.5, the new kernel): on a hand/analytic case `1 − (within-group circ var /
  total circ var)` via mean resultant lengths equals the worked value; the **rotation-invariance property** — adding a
  constant phase to every value leaves the reliability unchanged (the whole point, where a cos/sin-component ICC would
  move); perfectly concentrated within-group → 1, within = total dispersion → 0; concentration → 0 returns the
  `Rbar->0` unmeasurable signal, not a spurious number; the cos/sin diagnostic ICCs are reported but never bind.
- Cross-correlation: B = A shifted by +k samples → lag = +k/fs, r = 1 (A leads); inverted B → negative r; Fisher-z
  pooling equals the single-slice result on one slice.
- Vector coding: every Table 1 edge from both sides; stationary dropping with ε; **the printed abs-value Eq. 2
  misclassifies a 135°/315° coupling** — reading anti-phase as in-phase (a 225° coupling gets the wrong angle, 45°,
  but coincidentally the same in-phase class, so it does not expose the class error; amended 2026-10-04, review A-44)
  — (regression pin for §3.1 item 1); differences never span a segment split.
- Spectral: a pure tone at f₀ → median frequency f₀ (cycles·min⁻¹); a positive offset does not move it (mean removal);
  coherence ≈ 1 for a linearly filtered pair and ≈ 1/K for independent noise.
- Cluster phase: identical phases → ρ = 1; constant per-player lags → ρ_group = 1 (Frank & Richardson); uniform random
  phases → ρ small; `min_players` from both sides.
- SampEn / Cross-SampEn: reference values on published example series; exact integer parity of the numba, KD-tree and
  naive O(N²) counters; `entropy_undefined` reachable.
- Surrogates: time-shift preserves the autocorrelation exactly; a coupled pair's percentile separates from an uncoupled
  pair's (non-vacuity); seeded draws identical across processing orders; every accelerated statistic equals its direct
  computation on small N.
- Butterworth: zero phase lag on an in-band sinusoid; the stated −3 dB at `cutoff_hz` for the dual pass; residual
  analysis recovers a planted cutoff; resampling exact on linear signals; never interpolates across a split.

### 9.2 Seam parity

- `compute_defensive_line`: byte-identical to the pre-change loop on every committed tracking fixture, both
  directions, `n ∈ {3, 4, 5, "adaptive"}`, fewer than 3 players.
- `compute_team_shape`: all columns except `convex_hull_area` byte-identical; `convex_hull_area` within 1e-9 relative;
  collinear → exactly 0.0; n < 3 → NaN; `rtl` teams.
- Hull kernel vs `ConvexHull` on adversarial sets; spread identity vs the naive double sum.
- `add_team_shape` golden: only `team_shape_convex_hull_area_*` may differ, within 1e-9 relative.

### 9.3 Behaviour and contracts

Stoppage and detection-gap splits from both sides of each threshold; the stoppage-evidence precedence (each of
`ball_state`, `events`, `unavailable` reachable; `dead_ball_observed` raises on an unclassified provider; a constant
`"alive"` SkillCorner frame set never reports `ball_state`; event-derived intervals for every restart type and for
goals); substitution without splicing; red-card player
counts; each builder, including possession terminal events; mixed window sources refused; every §7.3 refusal; every
`COORD_SOURCE_VALUES` token reachable; report conservation; the warning category is distinct; purity (no input
mutated); id-dtype invariance over int, string and `category` ids; mirror and identity invariance (§7.10).

### 9.4 New-sibling-package gates (fail only in the full suite)

`tests/coordination/__init__.py` (empty); `feature_glossary` entries for every metric column with a package-specific
attribution token matched verbatim in `NOTICE`, and the `Unit` vocabulary gaining `"cycles/min"` (ADR-048 amendment);
run-and-diff legs in `tests/invariants/glossary_emitted_columns.py` for each `compute_*` plus non-vacuity anchors in
`tests/invariants/test_glossary_emitted_columns.py`; `_PUBLIC_MODULE_FILES` in `tests/test_public_api_examples.py`
and an Examples section (doctest for pure functions, RST literal block otherwise) on every public function, class and
method; the C4 container (`tests/test_c4_dsl_description_cap.py::test_every_shipped_subpackage_has_a_c4_container`) and
the glossary column count in `docs/c4/architecture.dsl` (459 + TF-58's columns), rendered with Graphviz `dot`;
`ARTIFACT_DRIVERS` enrolment of D1–D3; the import allowlist with planted-violation meta-tests; `metric_contracts`
registration of the seven families plus the re-keyed completeness test; `SCALE_GUARDED` entries with
`assert_subquadratic_growth` over the group dimension (games × periods, pairs) and the proof that a regressed rescan goes
quadratic; a structural test that FFT and phasor call counts do not grow with K; the numba cache-key pattern; the
F1b upcast gate's `_KERNEL_DIRS` extended to `silly_kicks/coordination` (and `tracking/preprocess`) at the rebase.
A **gated circular-column registry** (the three circular-mean constructs) with an **anti-rot meta-test** that derives
the circular set from the kernel/glossary and asserts the registry equals it exactly, so a new circular column cannot be
added without declaring it; and a `metrics.json` **per-construct schema** gate — every cell carries
`construct_key` / `unit` / `kind` / the `reliability` block, the `power` verdict is one of {measured, unmeasurable} with
`unmeasurable_reason` in `{null, n<min, ci_too_wide, Rbar->0}`, circular cells carry the cos/sin diagnostics and a
pinned origin, and the two honesty lines are present verbatim — with `metric_contracts` registering any newly emitted
reliability columns (ADR-098).

### 9.5 Real data and CI placement

Liveness on the committed provider fixtures: every metric column non-NaN and non-constant somewhere, with a precondition
test on the fixture itself (ADR-032). Corpus-scale and multi-match tests are `@pytest.mark.slow` (ADR-023); `.test_durations`
is regenerated. Driver tests cover the pure reduce kernels, codegen reproduction, resume, exclusion, pre-flight
visibility and every side-effect path under mocks. Both pandas legs run locally before any commit is proposed; lint at CI
scope (`ruff check silly_kicks/ tests/ scripts/`, `ruff format --check`, bare `pyright`).

---

## 10. Dependencies and packaging

- Runtime: no new dependency (numpy, scipy — `scipy.signal`, `scipy.spatial.cKDTree`, `scipy.stats` — pandas).
- Optional: `numba` via the existing `[numba]` extra (also in `[test]`), as an accelerator only.
- `[calibration]` and `[train]`: `ruthless-efficiency[optuna]>=0.7.0` (was `>=0.4.0`); `uv lock` regenerated.
- `.github/workflows/ci.yml`: the numba cache-key pattern and its comment (§7.15).

---

## 11. Documentation and full artifact set

Commit 1: this spec and the plan; the ADR (design section); `NOTICE` (a TF-58 methodology paragraph citing every §2
source); `feature_glossary` entries and the `Unit` addition; `metric_contracts`; C4 `.dsl` + `.html` (the
`coordination` container; the glossary count); `AGENTS.md` Architecture bullet (within the `test_agents_md_budget.py`
byte budget); `docs/context/tracking-metrics.md` narrative; `docs/PRIVATE_CONSUMERS.md` checked (no private module is
renamed). Commit 2: `docs/research/tf58_team_coordination/` (`derivation.json`, `calibration.json`, `metrics.json`,
`report.md`), the generated params file, the ADR results section, `CHANGELOG.md`, `TODO.md` (TF-58 row removed), the
version bump at commit-prep.

---

## 12. Breaking changes and downstream notice

- **`compute_team_shape.convex_hull_area` and `add_team_shape`'s `team_shape_convex_hull_area_{attacking,defending}`**
  change by at most 1e-9 relative (a different but exact algorithm). Not a retrain trigger; the lakehouse sees no
  material change. Recorded in the ADR and CHANGELOG.
- `smooth_frames`, `PreprocessConfig`, `tracking.__all__` (new `compute_collective_variables`, the preprocess array
  kernels and `resample_frames`): additive.
- New public package `silly_kicks.coordination` with seven metric contracts: additive. The lakehouse may materialise
  them; the notice lists the tables, grains and the SkillCorner coverage caveat.
- `tests/test_metric_contracts.py` re-keyed: internal.
- **ruthless 0.7.0 migration (D21).** `silly_kicks.calibration.stage1_config`, `stage2_config` and
  `xt_bandwidth_config` gain a required keyword `objective_id` (breaking for any external caller). Every in-repo
  caller derives it with one shared helper, `scripts/_provenance.objective_id(objective, inputs)` →
  `"<objective qualified name>@<git commit>:<declare_inputs digest>"`: fail-closed, because any code or declared-input
  change starts a fresh study (resume exists for crash recovery within one clean-tree run, which the drivers already
  enforce with `require_clean_tree`).
- **Owner-held Optuna studies become legacy.** Existing calibration and trainer SQLite studies (e.g. on the DGX)
  carry no `ruthless_identity`. Under 0.7.0 resuming them raises. The remedy is
  `adopt_legacy_store(config)` when the objective is genuinely unchanged, else a new store path. The CHANGELOG
  names both.

---

## 13. Commit structure and sequencing

1. **Prerequisite:** ruthless-efficiency 0.7.0 released (separate session; this session reviews its spec, plan and
   implementation). Commit-1 implementation may proceed in parallel against a local editable install of the ruthless
   feature branch; commit 1 is proposed for approval only after 0.7.0 is on PyPI, because its `[calibration]` /
   `[train]` floor and the D2 driver tests must resolve against the published release in CI.
2. **Commit 1 — code**, on `feat/tf58-team-coordination`: everything in §6 "In scope" except the corpus artifacts and
   generated values; spec and plan included (untracked files count as a dirty tree for the drivers). Full suite green on
   both pandas legs; `/final-review`; owner approval.
3. **Owner-run on DGX** against the clean commit-1 tree: D1 → D2 → D3 (DGX is the canonical compute; data via pining).
4. **Commit 2 — artifacts:** §11 commit-2 items. Full suite green; `/final-review`; owner approval. Version assigned here.
5. One PR, squash policy per the owner. Never merged until CI is green.

### 13.1 Rebase onto F1b and native DAS

- F1b: `team_id`-keyed `groupby` calls use `observed=True`; the upcast gate's `_KERNEL_DIRS` gains
  `silly_kicks/coordination` and `silly_kicks/tracking/preprocess`; `smooth_frames(method="butterworth")` writes
  `x_smoothed`/`y_smoothed` at the storage dtype (F1b's cast); the collective kernel reads coordinates as float64.
- Native DAS: textual conflicts only in shared registries (`NOTICE`, C4, `AGENTS.md`, `CHANGELOG`, `ARTIFACT_DRIVERS`,
  `.test_durations`, `pyproject.toml`, the numba cache-key line). If DAS's frame-contract validation lands on `main` as a
  reusable helper, `_signals.py` reuses it instead of its own duplicate-row and ball-row checks. Player ids are handled
  category-safely (`id_compat`, `observed=True`) ahead of a later `player_id → category`.
- The §7.16 caller sweep is re-run after the rebase, and the F1b upcast gate's scan scope (§3.5: non-recursive
  `silly_kicks/tracking/*.py` + `tracking/pitch_control/*.py`, read from the F1b clone, absent on this base) is
  re-verified on the merged `main` before `_KERNEL_DIRS` is extended.

---

## 14. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Hilbert phase is meaningless on drifting, non-oscillatory positional signals | phase-validity column reported; vector coding (no Hilbert) alongside; H1/H5 replication on real data |
| SkillCorner extrapolation biases coupling | detection-aware policy; occlusion leg; coverage stratification; `min_observed_fraction` gate |
| No dead-ball signal for SkillCorner and Metrica | event-derived stoppages when actions are passed; `coord_stoppage_source` on every row; the D3 stoppage leg measures the error |
| Reliability sweep selects an over-smoothed cutoff | Tier C can only move a value when replication still passes |
| Surrogate estimator drift from the observed estimator | algebraic-identity accelerations parity-tested against direct computation |
| Output volume | dyads on period windows by default; `COORD_PAIR_PHASE` without surrogates |
| ruthless 0.7.0 slips | TF-58 commit 1 depends on it; the handoff pins the contract so both proceed in parallel |
| CI time on the binding windows leg | heavy tests `@slow`; numba cache key covers the new file; `.test_durations` regenerated |
| A paper interpretation is wrong | every divergence is named, tested and recorded in the ADR |

---

## 15. Consequences

silly-kicks gains the temporal-coordination layer of the ecological-dynamics literature at four levels, with chance
baselines, honest degradation and a validation artifact on ~980 matches. Two long-standing hot loops
(`compute_team_shape`, `compute_defensive_line`) become vectorised single definitions, benefiting restdefense, causal,
off-ball runs and the team-shape features. ruthless-efficiency gains a general grid strategy with one-at-a-time and
explicit-point designs.

---

## 16. Numbering and provenance

- ADR number, PR-S number and version: assigned at commit-prep from `main` at that time.
- Spec written 2026-09-26 on `feat/tf58-team-coordination` @ `05cfa56`.
- Papers read in full: Bourbousson 2010 Part 2, Moura 2016, Moura 2013, Duarte 2013, Richardson 2012; Folgado 2014
  methods section; others by abstract or as standard references (§2).
- Probe evidence (§3.2): a throwaway scratchpad script, not committed.

---

## Appendix A — Sources

- Bourbousson, J., Sève, C., & McGarry, T. (2010). Space–time coordination dynamics in basketball: Part 1.
  *Journal of Sports Sciences*, 28(3), 339–347. https://doi.org/10.1080/02640410903503632
- Bourbousson, J., Sève, C., & McGarry, T. (2010). Space–time coordination dynamics in basketball: Part 2.
  *Journal of Sports Sciences*, 28(3), 349–358. https://doi.org/10.1080/02640410903503640
- Carter, G. C. (1987). Coherence and time delay estimation. *Proceedings of the IEEE*, 75(2), 236–255.
- Chang, R., Van Emmerik, R., & Hamill, J. (2008). Quantifying rearfoot–forefoot coordination in human walking.
  *Journal of Biomechanics*, 41(14), 3101–3105. https://doi.org/10.1016/j.jbiomech.2008.07.024
- Clemente, F. M., Couceiro, M. S., Martins, F. M. L., & Mendes, R. (2013). Measuring tactical behaviour using
  technological metrics: case study of a football game. *International Journal of Sports Science & Coaching*, 8(4).
  (Existing `NOTICE` entry; the Euclidean stretch index.)
- Duarte, R., Araújo, D., Correia, V., Davids, K., Marques, P., & Richardson, M. J. (2013). Competing together.
  *Human Movement Science*, 32(4), 555–566. https://doi.org/10.1016/j.humov.2013.01.011
- Folgado, H., Duarte, R., Fernandes, O., & Sampaio, J. (2014). Competing with lower level opponents decreases
  intra-team movement synchronization. *PLoS One*, 9(5), e97145. https://doi.org/10.1371/journal.pone.0097145
- Frank, T. D., & Richardson, M. J. (2010). On a test statistic for the Kuramoto order parameter of synchronization.
  *Physica D*, 239, 2084–2092.
- Lamb, P. F., & Stöckl, M. (2014). On the use of continuous relative phase. *Clinical Biomechanics*, 29(5), 484–493.
  https://doi.org/10.1016/j.clinbiomech.2014.03.008
- Mardia, K. V., & Jupp, P. E. (2000). *Directional Statistics*. Wiley.
- Moura, F. A., Martins, L. E. B., Anido, R. O., Barros, R. M. L., & Cunha, S. A. (2012). Quantitative analysis of
  Brazilian football players' organisation on the pitch. *Sports Biomechanics*, 11(1), 85–96.
- Moura, F. A., Martins, L. E. B., Anido, R. O., Ruffino, P. R. C., Barros, R. M. L., & Cunha, S. A. (2013). A spectral
  analysis of team dynamics and tactics in Brazilian football. *Journal of Sports Sciences*, 31(14), 1568–1577.
  https://doi.org/10.1080/02640414.2013.789920
- Moura, F. A., van Emmerik, R. E. A., Santana, J. E., Martins, L. E. B., Barros, R. M. L., & Cunha, S. A. (2016).
  Coordination analysis of players' distribution in football using cross-correlation and vector coding techniques.
  *Journal of Sports Sciences*, 34(24), 2224–2232. https://doi.org/10.1080/02640414.2016.1173222
- Pfister, R., Schwarz, K. A., Janczyk, M., Dale, R., & Freeman, J. B. (2013). Good things peak in pairs.
  *Frontiers in Psychology*, 4, 700.
- Richardson, M. J., Garcia, R. L., Frank, T. D., Gergor, M., & Marsh, K. L. (2012). Measuring group synchrony: a
  cluster-phase method. *Frontiers in Physiology*, 3, 405. https://doi.org/10.3389/fphys.2012.00405
- Richman, J. S., & Moorman, J. R. (2000). Physiological time-series analysis using approximate entropy and sample
  entropy. *AJP Heart and Circulatory Physiology*, 278, H2039–H2049.
- Schreiber, T., & Schmitz, A. (2000). Surrogate time series. *Physica D*, 142, 346–382.
- Sparrow, W. A., Donovan, E., van Emmerik, R., & Barry, E. B. (1987). Using relative motion plots to measure changes
  in intra-limb and inter-limb coordination. *Journal of Motor Behavior*, 19(1), 115–129.
- Welch, P. D. (1967). The use of fast Fourier transform for the estimation of power spectra. *IEEE Transactions on
  Audio and Electroacoustics*, 15(2), 70–73.
- Winter, D. A. (2009). *Biomechanics and Motor Control of Human Movement* (4th ed.). Wiley.
- Bedo, B. L. S., et al. (2026). DataGoal (MATLAB toolbox, Apache-2.0). Cross-check only.

## Appendix B — Review history (internal)

- **Round 1** (`D:\Development\_reviews\2026-09-26-tf58-team-coordination-spec.md`): APPROVE WITH FOLLOW-UPS.
  TF58-SPEC-01 (§7.16 misstated the `validate_corpus_visibility` test mechanism) — fixed in §7.1 item 5 and §7.16.
  TF58-SPEC-02 (`StrategyConfig` import path) — pinned in §3.6. Cross-clone note (F1b upcast-gate scope) — re-verify
  step added to §13.1. Managed dependency (ruthless 0.7.0) — sequencing clarified in §13 item 1.
- **Round 2** (`D:\Development\_reviews\2026-09-26-tf58-team-coordination-spec-r2.md`): APPROVE; both round-1 findings
  resolved; nothing new.
- **Round 3** (`D:\Development\_reviews\2026-09-26-tf58-team-coordination-spec-r3.md`): APPROVE; the D21 delta verified
  (caller sweep exact); nothing new. Carried: re-confirm the D21 sweep at the rebase and against the ruthless 0.7.0
  implementation review.
- **Delta reviewed in round 3:** D21. ruthless 0.7.0's final
  contract (required `StoreConfig.objective_id`, full Optuna identity guard) → §4 D21, §7.16 two new rows, §8.4 D2
  resume identity and native-float levels, §12 migration and legacy-study notice.
