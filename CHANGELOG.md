# Changelog

All notable changes to silly-kicks will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.35.1] — 2026-06-29

### Fixed — exclude pandas 3.0.4 (C-layer segfault on py3.11+)

- **pandas 3.0.4 segfaults (SIGSEGV / exit 139)** in its C `take_nd` → `maybe_promote` path when a
  whole-DataFrame boolean mask carries a `datetime64` column — reproduced deterministically on
  Python 3.11+ via `spadl.orientation.detect_input_convention` (the sportec actions carry a datetime
  column), which crashed the CI test suite. Bisected in an isolated py3.12 env: **3.0.2 ✓, 3.0.3 ✓,
  3.0.4 ✗** (same numpy 2.4.6 / scipy 1.18.0), so it is purely a pandas-3.0.4 regression.
- Dependency constraint tightened to `pandas>=2.1.1,!=3.0.4` — excludes **only** the broken release
  (pip resolves the safe 3.0.3) so a fixed 3.0.5+ is adopted automatically. `uv.lock` regenerated.
- No library code change; no behaviour change on a non-broken pandas. To be reported upstream
  (pandas-dev/pandas).

## [4.35.0] — 2026-06-27

### Changed — xT-GK PEV/DZV fidelity fix (Eyestone Q1–Q3, ADR-024 amendment, PR-S100)

- **PEV now measures its forward gain on the GK-revalued surface** `V_GK = xT · φ(z,d)`, not raw
  `xT` (CHANGE 1, Eyestone Q1+Q2). On the raw grid the keeper-zone forward gain is ~0 — the measured
  PEV inertia — because keepers live in the flat part of the xT surface; revaluing the surface is the
  point. `progress = V_GK*(z′) − V_GK*(z)`; the pressure-gated rectified form `PEV = ρ·max(0, progress)`
  is **unchanged**. RAV remains the sole owner of the destination value, so Option B is untouched (no
  double-count).
- **DZV is now the published defensive-zone revaluation multiplier** `M(z) = φ(z,d)·[1 − V_GK(z)/max V_GK]`
  applied as the **revaluation increment** on the origin possession value, `(M−1)·V_GK(z)`, gated to
  the defensive third (CHANGE 2, Eyestone Q3; Option A). This replaces the old additive `v_def − xT_raw(z)`
  back-pass floor. The increment (not the revalued total) keeps base — which surrenders the origin's raw
  threat — orthogonal to DZV. Per-action DZV lands O(0.01) (Jeff's ~0.009 La Liga anchor), not the raw
  multiplier's O(2.5).
- **φ(z,d)** `= α·(1 − d/D_max)^(−β)` for `d < D_threshold`, else 1, with `d` = LTR origin x: `α=2.1`,
  `β=0.8` are **canonical** (Eyestone 2026-06-27); `D_max=105`, `D_threshold=35` (= `defensive_third_boundary`)
  are provisional. `XtGkParams` gains `dzv_alpha`/`dzv_beta`/`dzv_d_max`; the now-dead `v_def` is retired.
  The scalar `phi` param stays the preset-modulated overall DZV weight (the canonical shape lives in the
  φ grid).
- **Invariant (Eyestone constraint):** φ enters value via PEV and DZV **only** — base keeps `−xT*(origin)`
  and RAV keeps `xT*(z′)`/`xT*_counter` on the raw `xT*` surface. Guarded behaviorally
  (`test_phi_shape_changes_only_pev_and_dzv_not_base_or_rav`).
- **Not a forced VAEP retrain** (xt_gk is opt-in, in no default xfn list) — but an `xt_gk` serve-output
  change: the lakehouse re-materializes `fct_action_context` and re-runs the WC2022 cohort/report. C4
  count unchanged (no new aggregator/model/backend; stays 28). Atomic mirror inherits.

## [4.34.0] — 2026-06-19

### Changed — TF-23b geometric frame-LTR backstop on the native tracking adapters (ADR-035, PR-S99)

- The native tracking adapters `tracking.gradientsports.convert_to_frames` and
  `tracking.sportec.convert_to_frames` now **self-correct a wrong/absent extra-time direction
  flag from goalkeeper geometry**, via a shared `direction.finalize_orientation` tail that layers
  the idempotent geometric backstop (`orient_frames_to_ltr_by_geometry`) on top of the per-period
  flag-flip. Byte-identical no-op on the correct-flag path. Closes ADR-031 **Gate D** (IDSSE-ET
  handedness). **VAEP/tracking retrain trigger** for the ≤3 GS WC2022 ET-tracking matches + any
  wrong-flag IDSSE-ET whose ET flag was wrong — see ADR-035 for the exact (G1) changed-match list.
- Public-net change: `orient_frames_to_ltr_by_geometry` gains `on_missing_home` and `copy`
  parameters (both additive, default-preserving — direct/lakehouse callers byte-identical), and
  **no longer orients period-5 / penalty-shootout frames for any caller** (including the TF-23
  SkillCorner/Metrica builders). PSO frames are excluded from geometric analysis (practical impact
  nil); the lakehouse self-assesses any SkillCorner/Metrica PSO re-materialization.
- The backstop's zero-home warning text changed (now emitted by the net via `on_missing_home="warn"`).

## [4.33.0] — 2026-06-18

### Added — TF-23 SkillCorner + Metrica bronze→frame builders (ADR-034, PR-S98)

- Two pure, bronze-consuming converters — `tracking.skillcorner.convert_to_frames` and
  `tracking.metrica.convert_to_frames` — parallel to `tracking.sportec` /
  `tracking.gradientsports`. They single-source the SkillCorner/Metrica coordinate
  rescale, period-relative clock, id-namespacing, GK derivation, speed, and LTR
  orientation that the luxury-lakehouse previously duplicated three ways (the kloppy
  gateway oracle + two lakehouse builders). Emit the kloppy-variant schema
  (`SKILLCORNER_TRACKING_FRAMES_COLUMNS` / `METRICA_TRACKING_FRAMES_COLUMNS`).
- `tracking.orient_frames_to_ltr_by_geometry` — flag-free geometric frame-LTR
  orientation (per-period home-GK-median-x anchor, point-reflect mis-oriented periods,
  idempotent), a schema-adapted port of the luxury-lakehouse ADR-053
  `correct_frames_to_home_ltr`. Retained alongside the flag-based `orient_frames_to_ltr`.
- SkillCorner `ball_z` recovery — the builder maps the (previously discarded) real ball
  height into the `z` column, unblocking SkillCorner post-shot height features (TF-48
  PSxG) that were silently null in production.

### Input contract (Metrica)

- `tracking.metrica.convert_to_frames` requires bronze `y` in **SPADL bottom-to-top**
  convention (the lakehouse bronze landing already provides this). kloppy's metrica NATIVE
  coordinate system is top-to-bottom, so a consumer landing bronze straight from a kloppy
  `TrackingDataset` must flip `y` (`1 − y`) first. Fed contract-honoring bronze, the
  builder matches `tracking.kloppy.convert_to_frames` byte-for-byte (validated dx=dy=0 on
  Metrica open-data game 1, incl. LTR orientation).

### Notes

- Additive; no silly-kicks model retrain (new modules + new public orienter; existing
  converters/gateway untouched; in no default xfn list). The luxury-lakehouse adopts the
  builders + orienter and retires its two builder copies + its orientation net (its
  re-materialize trigger, not silly-kicks'); ADR-031 Gate C closes on the shipping path
  via the event-anchored y-identity gate. GS native-adapter ET orientation is tracked as
  the TF-23b follow-on.

## [4.32.0] — 2026-06-16

### Added — `add_*` input-purity CI gate (ADR-033, PR-S97)

Every public `add_*` enricher must be PURE: it must not mutate any caller-supplied DataFrame/Series/ndarray
and must return a NEW object. New auto-enumerating gate `tests/test_add_star_purity.py` — one canonical
`PURITY_ENTRIES` registry covering the full public surface (`spadl`/`atomic.spadl`/`tracking`/
`atomic.tracking`, including the 15 `atomic.tracking.features` mirrors), build-fresh-owned-inputs-once +
snapshot-every-array-arg + value-equality + `out is not input`. Two meta-assertions pin the surface to the
public export (`__all__` UNION `.features.__all__`), so a new `add_*` cannot land unregistered. A best-effort
AST heuristic nudges toward per-branch coverage; the contributor contract (CLAUDE.md: any column-conditional
`add_*` registers ≥2 variants) is the real backstop. Joins the auto-enumerating-gate family alongside
nan-safety / liveness / dup-`action_id` / id-dtype.

### Fixed — `add_gk_distribution_metrics` mutated the caller's frame when `gk_role` was present (ADR-033, PR-S97)

When `gk_role` was already present, both `add_gk_distribution_metrics` implementations (standard +
atomic-SPADL) assigned their four columns straight onto the caller's input DataFrame and returned it,
contradicting the documented "Sorted copy" contract (the `gk_role`-absent path always copied, so the old
column-list mutation guard never caught it). Now hoists `out = actions.sort_values([...]).reset_index(drop=True)`
to the top and operates on `out`. **Identity + order only, no value miscompute, no recompute:** the sort key
matches `add_gk_role`'s internal sort (so the `require_gk_role` path is value/order-identical) and derivation
is per-row vectorized — the lakehouse need not re-materialize. The repo-wide audit found the mutation class is
otherwise clean (only this helper).

### Changed — `pitch_control_at_action` → `pitch_control_at_target` (BREAKING rename; ADR-033, PR-S97)

The action-coupled function is renamed to match its emitted `pitch_control_at_target__<method>` column base
(unchanged since 4.31.0) — standard + atomic, plus `__all__`, imports, and callers. Window-justified: no
released consumer of the 4.31.0 column rename exists yet. The lakehouse keeps its own DEFCON
`pitch_control_at_action` mart column (different semantics; not silly-kicks').

### Changed — docstring tightening (Part C; ADR-033, PR-S97)

Enumerated emitted columns + dtypes for `add_off_ball_runs` / `add_off_ball_context` / `add_shot_goalmouth`;
added the `gk_pass_length_class` Categorical/Spark-`StringType` note and the `gk_xt_delta`
caller-supplied-`(12,8)`-SPADL-grid (never self-fit) note to `add_gk_distribution_metrics` (standard +
atomic). A doc-accuracy test pins each exhaustive-claim helper's emitted feature set to an explicit
`frozenset` and asserts the docstring names every column.

## [4.31.0] — 2026-06-16

### Changed — pitch control re-aimed to the action destination; dead at-ball column retired (ADR-032, PR-S96)

**BREAKING column rename.** The informationally-dead `pitch_control_at_ball__<method>` (the Spearman PPCF
at the ball is the degenerate ~0.5 reaction-time fallback, so the column was ~0.5 for every well-linked
action) is **retired** and replaced by a live `pitch_control_at_target__<method>` sampled at the action
**destination** `(end_x, end_y)`, where ball-travel-time is positive so players can contest it.

- **Mandatory ADR-028 re-projection (fixes a latent away-team bug the degeneracy had masked).** The old code
  sampled the action-LTR `(start_x, start_y)` against an absolute-frame (home-attacks-right) surface with no
  per-action flip — wrong for away-team actions, harmless only because near-ball is 0.5 in both conventions.
  The new code re-projects the query via `acting_team_attacks_rtl` + `reproject_to_action_ltr` (the cached
  per-frame surface + `PitchControlCache` key are unchanged — only the query point flips). Applies to all
  three methods (`spearman`/`fernandez_bornn`/`voronoi`) and the atomic mirror (synthesizes `end=x+dx,y+dy`).
- **Per-type semantics (kept uniform):** open-play destination control for passes/crosses/carries;
  target-cell contestation (GK/defender-dominated) for shots; ~0.5 for in-place actions (no destination —
  honest). A model conditions on this via `type_id`.
- **Localized:** other PPCF consumers (`obso`/`cover_shadows`/`gk_influence`/`player_influence`/
  `space_creation`) sample their own points and are untouched. The dead column's `STRUCTURAL_CONSTANTS`
  liveness exemption + its near-ball-degeneracy invariant test are removed (the column is now live); a hard
  off-ball-destination precondition guards the liveness gate's teeth.
- **VAEP/tracking + calibration retrain trigger** (dead constant → live signal + away-team correction). The
  silly-kicks calibration feature set + lakehouse consumers re-materialize. **Lakehouse adoption is a
  breaking column-lifecycle migration (AC + DEFCON), atomic with the pin bump — not a currency bump.** C4-free
  (aggregator count stays 28).

## [4.30.0] — 2026-06-16

### Added — DFL / Sportec parse+shape port (ADR-031, PR-S95 / T3)

A new `silly_kicks/providers/sportec/` package (behind a `[parse-dfl]` optional extra) **single-sources
the IDSSE/Sportec DFL parser**, eliminating the dev/prod parser drift and retiring the y-inverting
loader-local kloppy `_kloppy_tracking_to_frames` from the calibration/pining harness in favour of the
native `spadl.sportec` / `tracking.sportec` converters.

- **Public surface:** `parse_dfl_match_info` / `parse_dfl_tracking` / `parse_dfl_events` (DFL XML →
  RAW provider-canonical bronze) + `shape_tracking_to_native` / `shape_events_to_native` (bronze →
  converter input) + `derive_idsse_home_team_start_left{,_extratime}`; typed returns `MatchInfo` /
  `SportecTrackingBronze` / `SportecEventBronze` (silly-kicks' own domain names — a versioned cross-repo
  bronze contract, ADR-031 N1).
- **Verbatim lift.** Parse/shape function bodies are upstreamed byte-for-byte from luxury-lakehouse @
  `0efac60`; the only adaptations are the `logger`-arg defaulting, two inlined cross-module helpers
  (`idsse_native_match_id`, `finalize_bronze_df`), and a materialised 246-column events bronze-column
  set (the lakehouse derives it from a schema module). Pinned by `tests/datasets/sportec/idsse_slice/SOURCE_SHA`.
- **Data-quality is consumer-side.** The port emits RAW bronze (no Savitzky-Golay smoothing / velocity
  derivation); the harness applies `_preprocess` after shaping, and a delete-and-depend lakehouse keeps
  its own smoothing after the parse.
- **Golden parity test** (`tests/providers/sportec/test_parse_port_parity.py`) asserts the port
  reproduces goldens captured by running the **real** lakehouse functions on a reduced real-WC2022 IDSSE
  slice — a genuine "port reproduces production" guard (sensitivity-proven).
- **No new tracking aggregator** → the action-coupled aggregator count is unchanged (28). New C4
  container (`providers.sportec`) feeding both sportec converters.

### Changed — IDSSE harness re-route (N6 retrain trigger)

`scripts/_loader_pining.py::_build_idsse` now parses via the port → native converters. The action↔frame
y-axis now agrees (acting-player frame-y matches the action `start_y` to ~0.2 m after the ADR-028
re-projection, vs ~11.8 m on the retired kloppy path), and the action `team_id` is remapped from the
`"home"/"away"` label to the DFL CLU id so the ADR-028 join aligns with the CLU-keyed frames. **IDSSE
calibration/pining feature values change → those consumers re-materialize.** Documented by
`tests/calibration/test_calibration_invariance_e2e.py`. (Gate D: the native sportec converter was
already y-correct; IDSSE's old misalignment was partial, not the clean SkillCorner-style inversion T1
fixed.)

## [4.29.0] — 2026-06-16

### Fixed — kloppy tracking-gateway y-axis inversion (CS-pin; ADR-031, PR-S94 / T1)

The kloppy **tracking** gateway (`silly_kicks.tracking.kloppy.convert_to_frames`) produced frames with
a y-axis **inverted** relative to the SPADL action y-axis (`action_y == 68 − frame_y`) for every
kloppy-based provider (SkillCorner, Metrica, and the IDSSE dev-harness path). The **event** gateway
pinned the canonical `_SoccerActionCoordinateSystem` (origin `BOTTOM_LEFT`, vertical `BOTTOM_TO_TOP`);
the tracking gateway never did, retaining each provider's kloppy-native vertical. It is a single-axis
y mirror (NOT orientation — orthogonal to ADR-028/029); error `|68 − 2y|` is 0 at centre and ~full
pitch width at the touchlines (why it hid).

- `_SoccerActionCoordinateSystem` extracted to `silly_kicks/spadl/_kloppy_coordinates.py` (with a
  `socceraction_coordinate_system(metadata)` helper); both gateways import it (DRY). The event-path
  output is **byte-identical** (the helper reads the same metadata the inline construction did).
- `tracking/kloppy.py` now pins the coordinate system. **Signature is CS-only** — it drops
  `to_pitch_dimensions` and relies on the CS's own standardized 0–105/0–68 dimensions, matching the
  event gateway. (Keeping `to_pitch_dimensions` while adding the CS silently overrides the CS's
  vertical and leaves y inverted — verified on SkillCorner.) **NOT** a blanket `y = 68 − y` flip
  (which would double-invert an already-canonical provider; guarded by a no-op test).

**Scope (Gate C):** this fixes the **calibration/pining path + external kloppy-gateway consumers** —
the lakehouse builds SkillCorner/Metrica frames via its own bronze builders, not this gateway.
**Retrain trigger:** VAEP + tracking **calibration** consumers for **SkillCorner and Metrica** (both
were inverted; Gate A). The native sportec/IDSSE path is unaffected (Gate D: y-correct). Gradient
Sports native and event-only providers (StatsBomb/Wyscout/Opta) unaffected. Decision: ADR-031.
First of a sequence; the IDSSE/Sportec DFL parse-port single-sourcing (T3) follows in PR-S95.

## [4.28.0] — 2026-06-15

### Added — TF-48 post-shot goalmouth crossing geometry (`add_shot_goalmouth`; ADR-030)

New `silly_kicks.tracking.add_shot_goalmouth(actions, frames, *, links=None, params=None)` derives,
for each shot action (`shot`/`shot_freekick`/`shot_penalty`), the goal-plane crossing from the
post-contact ball trajectory in tracking frames: `shot_crossing_y`/`shot_crossing_z` (SPADL meters,
canonical attacked-goal-at-x=105), `shot_speed` (fitted initial speed at contact — ALWAYS the
contact sub-segment, never a post-bounce refit), `shot_time_to_goal_line`,
`shot_on_target_derived` (posts/bar expanded by the ball-radius tolerance), plus full provenance
(`shot_crossing_source` ∈ {observed, extrapolated, insufficient_frames, no_crossing,
no_ball_frames, unresolved}, `shot_crossing_confidence`, `shot_fit_n_frames`, `shot_fit_rmse`,
`shot_fit_end_reason`, `shot_z_profile` ∈ {airborne, rolling, bounced}). Pure geometry, no model —
the lakehouse scores the output with its existing StatsBomb-trained PSxG model (Goals Prevented for
the tracking providers). Engine (`compute_shot_goalmouth`) is pure + orientation-agnostic (goal
ends from the GK map; `defended_goal_x` extracted byte-identically from xS into `_gk_resolve.py`);
the per-shot kernel is pilot-hardened on real WC2022 data: a sample-and-hold collapse (GS's raw
`balls` channel delivers ~15 Hz positions duplicated at 29.97 Hz stamps — 50% exact
consecutive-duplicate x/y/z, raw-artifact-confirmed; held duplicates are phantom zero-velocity
samples that phase-modulated every speed gate and saw-toothed the fits into phantom
trajectory-breaks), flight-run anchoring for t0 (GS stamps shots up to ~2.6 s before contact) with
a contact anchor (the shooter's own action coordinates — measured exact ball-track points on GS —
split a continuous assist-cross/dribble + shot approach run at the contact; orientation-agnostic
via the goal_x reflection), 0.1 s-baseline velocities (per-frame finite differences amplify
29.97 fps jitter ~30×), LOCAL residual break checks (a segment-anchored linear residual
phantom-breaks any smoothly curving chip/curl ~1 s in; a deflection violates even the local fit),
z-aware flight classification (an airborne decelerating chip is flight; carries/frozen tails are
on the ground), a flight-core trim (slow ground heads/tails removed; away-flying balls stay honest
`no_crossing`; sub-flight balls `insufficient_frames`), an extrapolation-leverage cap
(`max_extrapolation_leverage`: t\* beyond 3× the fitted span is a guess, not a fit — pilot-measured
dy median 6.2 m vs 2.4 m below the cap), and a contact-EXISTENCE bar (a window whose ball never
comes contactably near the stamped shot location — 2-D within 5 m at playable height z ≤ 2.6 m —
provably does not contain the shot → honest `insufficient_frames`; kills the measured worst class,
a 12.6 m "observed" goal crossing fitted from a pre-contact assist arc passing 6 m overhead).
`ShotGoalmouthParams` (pilot-calibrated defaults) + `ShotGoalmouthReport`
QA aggregate + per-Series wrappers + atomic mirror ({shot, shot_penalty}; `shot_freekick` is a
`freekick` atom). **NO VAEP xfns factory** — post-contact outcome descriptors are
HybridVAEP-class result leakage; a guard test auto-discovers every default xfn list and asserts
absence. NOT in any default xfn list → **no retrain trigger**. C4 action-coupled-aggregator count
27 → 28. Owner-gated GS↔StatsBomb WC2022 acceptance harness
(`scripts/validate_shot_goalmouth_sb.py` + held-out e2e with ADR-pre-registered floors; goals/saves
stratified — SB save end_locations are save-points, not plane crossings). **Holdout-validated
accuracy (one-shot protocol, ADR-030 pre-registered floors, 48 held-out WC2022 matches, 999
matched shots): goals |Δy| median 2.17 m (floor ≤ 2.5; p90 5.7 — tail dominated by
observed-straddle GS-vs-SB hand-tag disagreements where GS's own in-net samples corroborate GS),
|Δz| median 0.48 m (floor ≤ 1.25), on-target resolution coverage 0.620 (floor ≥ 0.60), on-target
agreement 0.60 (floor ≥ 0.45) — ALL FLOORS PASS.** The meters→SB y-handedness is settled on GK
GEOMETRY (SB shot freeze-frame defending GK vs the GS-tracked GK: 0.882 flip agreement on 646
voters, pilot-vs-holdout instrument-stable at 0.883/0.882; the round-1 ball-tag gate was measured
too noisy to settle a transform and is demoted to an informational diag). Holdout round 1 aborted
at that ball-tag gate, and its documented failure analysis exposed a harness clock-base bug that
had silently excluded ALL period-2 shots from every pilot metric (GS SPADL `time_seconds` is the
CUMULATIVE match clock — the known lakehouse-guarded GS convention — while the harness converted
SB to period-relative; fixed, matching now covers both halves; full record in ADR-030).
Crossing-z is GS-z-channel-limited (onset lag). **Completion cycle (4.28.0):** two kernel
refinements, re-validated on the FULL 64-match GS corpus (the post-holdout protocol): (1) a
span-gated curve-aware y extrapolation — the constant-velocity fit extrapolated a curling/dipping
flight's crossing LINEARLY (measured 5.4 m on a real chip-curl goal); when the producing segment
supports a curvature estimate AND a quadratic markedly out-fits the line (real curl, not jitter),
the crossing y is taken from the quadratic, capped tighter than the linear leverage; (2) an
earliest-reaching flight-run tie-break — when >1 plane-approach run reaches the goal line, the SHOT
is the EARLIEST (the bare nearest-plane rule had anchored t0 PAST a real in-mouth crossing on a
measured holdout goal). Full-corpus re-validation: goals |Δy| median **2.08 m** (improved from
holdout 2.17), |Δz| 0.49, coverage 0.63, on-target agreement 0.61, GK-handedness 0.882 on 851
voters — ALL FLOORS PASS, no regression. A final-kernel sensitivity sweep (`--sweep`, extended to
the contact/flight module constants on a 10 fps-downsampled copy) confirms the kernel is robust (no
cliffs) and the new constants are inert on resolution. **Provider coverage:** GS is validated;
SkillCorner/IDSSE currently return `insufficient_frames` due to a SEPARATELY-TRACKED upstream bug
(kloppy-derived tracking frames have an inverted y-axis vs SPADL actions —
`docs/research/bug_kloppy_tracking_y_inverted.md`, tracked in TODO.md); TF-48's kernel is
provider-agnostic and resolves SkillCorner to the GS baseline once that input is corrected (proven:
a coordinate-fix smoke-test lifts SC resolution 0.12→0.60). Decision: ADR-030.

### Fixed

- **pining GS loader dropped ball z** (`scripts/_loader_pining.py`): every frame row was hardcoded
  `z=0.0` and the raw ball records' `z` (present on 100% of GS ball records; probe 2026-06-10) was
  never read → all loader-fed GS frames had flat zero ball z. Ball rows now carry the real z
  (players keep 0.0 — no z in GS player records). Affects any loader-fed analysis that consumed GS
  ball z (e.g. xS ball-z features at GS inference saw zeros). **Audited:** re-ran the xS PR-S80
  public-vs-full data-effect test with real GS z (controlled A/B at the shipped public params, GS z
  real vs forced-0) — all 5 folds stay negative (mean Δ -0.058 with real z vs -0.026 with z=0;
  ship_two=false either way), so the shipped xS conclusion (ship the public-only model) is UNCHANGED;
  the loader z bug did not flip it.

## [4.27.1] — 2026-06-15

### Documentation
- **ADR-code reconciliation sweep** — verified all 29 ADRs (ADR-001…029) against the current tree; 25 were clean, no behavioral drift found. Corrected stale prose in 5 living ADRs: ADR-004/ADR-005 (`TRACKING_FRAMES_COLUMNS` is 20 columns, not 19 — the `is_goalkeeper_source` provenance column added in PR-S26 was undocumented; now listed), ADR-004/ADR-006 (the `tracking._direction` module path was renamed to the public `tracking.direction` in 4.0.0 per ADR-010), ADR-010 (Status `pending implementation` → `implemented in 4.0.0`), ADR-017 (de-pinned a drifted `gradientsports.py:416` line-number citation). Historical specs/plans/CHANGELOG were intentionally left untouched (immutable point-in-time records; PR-S19 genuinely shipped 19 columns).
- **TODO.md** — collapsed the bloated multi-paragraph "Last updated" header (which had accreted per-version historical notes) back to a single current-release summary line; relocated the parked `pitch_control_at_ball__spearman`-redesign item from the header into Technical Debt → Blocked or Deferred (and fixed a `>`-at-line-start Markdown blockquote render bug in it).

### Notes
- **Documentation-only — no library/package code, schema, or behavior change; no model retrain.** The `silly_kicks` package is byte-identical to 4.27.0.

## [4.27.0] — 2026-06-13

### Added
- `silly_kicks.tracking.orient_frames_to_ltr(frames, *, home_team_id, home_team_start_left, home_team_start_left_extratime=None)` — orients *unlabeled* absolute tracking frames into the canonical home-attacks-right (LTR) frame, single-sourcing the orientation contract for consumers that build frames from a non-kloppy source (bronze DataFrames). Pure composition of existing primitives (`compute_attacking_direction` + `play_left_to_right`) with fail-loud guards (missing-schema, already-labeled → use `play_left_to_right`, zero home-match, ET-without-flag). Companion to ADR-028: ADR-028's per-action reprojection no-ops on absolute frames (`team_attacking_direction = None`), so consumers must orient first. Decision: ADR-029.

### Notes
- **Additive — no model retrain.** Existing providers (sportec/gradientsports/kloppy) are byte-unchanged; the helper is new and not called internally. **Consumer impact:** adopting `orient_frames_to_ltr` in the lakehouse metrica/skillcorner bronze builders fixes their previously-bimodal tracking action geometry (`pre_shot_gk_x`, `defensive_line_x`, `nearest_defender_distance`, `pressure_on_actor__*`, etc.); those providers must be re-materialized lakehouse-side. The helper is only as correct as the caller-derived `home_team_start_left` — validate it per game.
- Added a positive extra-time orientation regression guard for the native `gradientsports`/`sportec.convert_to_frames` ET path (`tests/tracking/test_adapter_extra_time_orientation.py`), prompted by a live GS-ET flip that was a consumer-side `home_team_start_left_extratime` placeholder bug, not a silly-kicks bug.

## [4.26.0] — 2026-06-12

### Fixed — tracking geometry now emitted in the per-action SPADL LTR frame (systemic orientation bug; ADR-028)

**Breaking value change. VAEP/tracking-retrain trigger — re-materialize all tracking action-context.**

SPADL actions are per-acting-team LTR (the acting team attacks x=105); `convert_to_frames`
output is home-attacks-right (the home team attacks x=105 every period). The two are a 180°
point reflection apart for away-team actions, and the tracking-geometry layer sampled frame
positions **without re-projecting** them into the per-action LTR frame. On ~50% of
tracking-provider action rows (away-team actions) this produced wrong values:

- **Absolute positions at the wrong end** (visibly bimodal): `pre_shot_gk_x/y`,
  `pre_shot_gk_distance_to_goal` (reached 106 m), `defensive_line_x`, `back_line_high_x`,
  `team_shape_centroid_x/y_*`, `team_shape_defensive_line_height_*`.
- **Mixed-frame scalars** (action anchor combined with frame positions → numerically wrong,
  not just mis-oriented, and not visibly bimodal): `nearest_defender_distance`,
  `receiver_zone_density`, `defenders_in_triangle_to_goal`, all `pressure_on_actor__*`,
  `pre_shot_gk_distance_to_shot`, `pre_shot_gk_angle_*`.
- **`ghost_gk_x/y`** were goal-relative (defended goal at x=0) while the actual-GK features
  intended action-LTR → cross-frame "ghost deviation ≈ 90 m" downstream.

Fixed by one canonical re-projection (`tracking/_action_orientation.py`, driven by the
frame's `team_attacking_direction`) applied at three seams: the shared `ActionFrameContext`
(fixes all 8 context kernels at once and makes their hardcoded goal-at-105 correct),
`_defensive_line_at_actions`, and `add_team_shape`/`_team_shape_at_actions`. `add_ghost_gk`
now emits action-LTR (`x → 105 − gr_x`; `y` mirrored for away actions); the model stays
goal-relative. `compute_team_shape` is additionally made orientation-aware so
`defensive_line_height`/`inter_line_gap_*` are each team's *true* defensive line (was the
min-x cluster for everyone → the away team's advanced line). Self-reconciling features
(`structural_pass`, `gk_influence`, `player_influence`, `cover_shadows`, `shape_graph`,
`obso`, `space_creation`, `das`, `pitch_control`, `pausa`, `xt_gk`) are unchanged. A
mirror-symmetry property test (`tests/tracking/test_action_ltr_mirror_invariance.py`) is the
durable guard. Home-team values are byte-identical; only away-team values change.

Also fixed a latent pandas-3.0 compatibility bug surfaced en route: the frame-fallback GK
resolver in `add_pre_shot_gk_context` filled `defending_gk_player_id` via `.fillna()` with an
object Series; pandas 3.0 stopped silently downcasting the result, leaving the column `object`
(float64 on pandas 2.x), which made the downstream float-vs-object GK id match find zero rows →
NaN GK position. The fill now restores the contractual float64 dtype. Affected real data on
pandas 3.0 whenever the GK resolves via the frame fallback (the common path — DFL/Sportec rarely
emit `keeper_save`).

## [4.25.0] — 2026-06-11

### Fixed — GS null-actor duel/foul events emit NaN team_id/player_id (was sentinel 0); nullable Int64 (lakehouse production outage; ADR-001)

The Gradient Sports converter emitted the integer sentinel `0` as `team_id`/`player_id`
on null-actor events. Because `0` is non-NaN, it masqueraded as a real id, bypassed every
downstream `pd.isna` NaN-route, and crashed the strict opponent-resolution guard in
`tracking._space_creation._resolve_opponent_team_id`
(`ValueError: attacking_team_id '0' does not uniquely match the frame team ids [...]`),
taking down every Gradient Sports unit in the lakehouse action-context pipeline (2026-06-11).
Under ≤4.22.1 the same rows produced silent NaN space values; 4.23.0's loud two-team guard
turned the latent corruption into a hard failure. Good guard, bad input — the fix is upstream,
in the converter.

**Root cause.** `spadl/gradientsports.py` did `events["team_id"].astype("Int64").fillna(0)
.astype("int64")` (and the same on `player_id`) because `SPADL_COLUMNS` types both as
non-nullable `int64`, which cannot hold NaN. Gradient Sports is the only int-id provider;
the kloppy-family providers carry object-string ids where the absent actor is naturally
`None` (pd.isna-routable), which is why no other provider hit this.

**Ground truth (canonical PFF WC2022 feed, 64 matches).** The null-team events are the
two-sided duels and dedicated fouls — **594 `OTB`+`CH` challenges + 28 `FOUL`+`FO` fouls** —
and on **every one of them `gameEvents.playerId` is ALSO null** (a challenge is a 50/50 duel
with `homeDuelPlayerId` *and* `awayDuelPlayerId` and no single owning team; a dedicated foul
has no on-the-ball actor). The only team-resolving ids that exist (challenger / winner /
culprit) are possession-event *qualifiers* — synthesizing `team_id` from them is exactly the
ADR-001 violation that silly-kicks 2.0.0 removed (the sportec tackle-winner override the
lakehouse reported in PR-LL2; ADR-001 itself classifies team-less fouls as *legitimate NULL*).
So NaN is the architecturally-correct value, confirmed with the lakehouse, which withdrew its
original "resolve from the acting player's roster" prescription (that acting player does not
exist on the feed).

**Fix.**

- `GRADIENTSPORTS_SPADL_COLUMNS` types `team_id` / `player_id` as nullable **`Int64`** (was
  the inherited `int64`); they mirror the canonical `gameEvents` actor verbatim, **NaN where
  the actor is absent — never a sentinel 0**.
- **ADR-001-legal self-heal** (`_resolve_team_ids`): where a row has a real canonical
  `player_id` but a null `team_id`, derive the team from that player's other same-match rows
  (a player belongs to one team per match). Keys ONLY on the canonical `player_id` column,
  NEVER on a duel/foul qualifier; an ambiguous mapping raises rather than guesses. On the
  canonical feed this resolves nothing (player_id is null wherever team_id is), so all
  null-actor rows are NaN; it self-heals only genuine player-present/team-absent rows.
- Orientation `_mirror_per_period` is NA-safe (`na_value=False`): a null `team_id` keeps the
  EXACT coordinate orientation the pre-fix sentinel 0 produced (`0 != home_team_id` and
  `NA == home_team_id` both collapse to "not home"), so only `team_id`/`player_id` change —
  coordinates are byte-identical.
- `atomic.spadl.convert_to_atomic` preserves the source `team_id`/`player_id` dtype instead
  of force-casting to the atomic schema's `int64` (which crashed on the new GS NaN — and
  would also have crashed on sportec/skillcorner object-string ids; latent bug fixed).
- **`tracking._line_breaking` (Ward line-breaking, `add_line_break(method="ward")`) — two
  fixes a downstream NaN-safety audit surfaced** (the early opponent-resolution crash had been
  masking them): (1) a NaN-team action now NaN-routes instead of raising
  `TypeError: boolean value of NA is ambiguous` at the opponent-set list-comp (`t != <NA>`);
  (2) the opponent set now uses the ADR-019 `same_id` instead of a raw `!=`, which on a
  mixed-dtype pairing (Int64 action team vs object-string frame team — exactly GS actions on
  tracking frames) was always True and silently kept the actor's OWN team as the "opponent",
  mis-computing every GS Ward line-break. The ADR-019 AST lint missed this because the
  operands are named `t`/`action_team`, not `*_id`. **All 14 frame-aware AC consumers
  (space_creation, obso, pitch_control, shape_graph, team_shape, structural_pass,
  line_break[ward+threshold], das, gk_influence, player_influence, cover_shadows, pausa,
  pressure) verified to NaN-route a NaN-team action on a healthy two-team frame** — no crash,
  real rows still compute; the few non-NaN values on the NaN-team row are team-INDEPENDENT
  frame properties (`pitch_control_at_ball`, `pressure_on_actor`), not miscomputes.

**Impact / re-conversion delta (acceptance #5).** ~**622 SPADL actions per WC2022 corpus**
(≈594 tackle + 28 foul + the 1 null-actor touch→bad_touch) flip `team_id`/`player_id` from
the sentinel `0` to NaN. Downstream these now route to the NaN-row default (e.g.
`space_created` returns the NaN row) instead of crashing — they carry NO enrichment, which is
honest for a contested duel / stoppage. **Hyrum / retrain trigger:** GS `team_id`/`player_id`
dtype `int64`→`Int64` is an observable schema change, and the value flip shifts any
team/player-keyed GS feature for these rows — VAEP/tracking consumers re-materialize GS.
Decision: ADR-027 (GS null-actor NaN identifiers), grounded in ADR-001 (no qualifier→identifier
override) + ADR-003 (NaN-safe enrichment) + ADR-019 (id-dtype contract). C4-free (no new
aggregator/model/backend; count stays 27).

## [4.24.0] — 2026-06-11

### Fixed/Changed — opponent OBSO orientation MIRRORED + LEAN 2-column contract (TF-41 round-2; ADR-026 amended; owner-approved breaking)

The lakehouse rejected 4.23.0's opponent triplet: under a complementary pitch-control model
with a SHARED, UNMIRRORED multiplier, `opp_obso = (1−pc)·M` is fully determined by `pc·M`, so
the opponent leave-one-out was the exact pointwise negation of the team LOO
(`opponent_space_destroyed_m2 ≡ space_created_m2` bit-for-bit; reproduced and
algebraically confirmed — informationally empty). The owner additionally directed a contract
reshape in the same release (no consumer has adopted any 4.23.x surface):

- **Semantic fix — the opponent surface is weighed by the opponent's OWN attacking
  geometry**: the same transition/EPV grid ARTIFACTS mirrored along x to the goal the
  opponent attacks; the ball-anchored distance weight is unchanged. Grid resolution, sigmas,
  and PC method stay shared (magnitudes comparable). Both the analytical
  (complement-decomposition) and naive (explicit recompute) paths consume the same mirrored
  multiplier — one metric, two estimators (round-2 acceptance #4 method-consistency test:
  spearman vs voronoi agree in sign and order of magnitude). Anti-mirror gate (round-2
  acceptance #2) red-first then green; a geography pin makes silent un-mirroring untestable.
- **LEAN CONTRACT (breaking, owner decision): `add_space_creation` now emits exactly TWO
  columns** — **`space_created_m2`** (>= 0; the actor's LOO on their own team's OBSO
  surface; attacking value) and **`space_denied_m2_opponent`** (>= 0; the same LOO on the
  mirrored opponent surface; rest-defense value). The structurally-zero columns are RETIRED
  rather than shipped: the LOO is pointwise-MONOTONE — removing a player can only decrease
  his own team's control and increase the opponent's, everywhere, for every shipped PC
  method — so a team-side "destroyed" (zero since TF-41 shipped) and an opponent-side
  "created" are always 0, and net columns are exact redundancies of the live pair.
  `compute_space_created` is leaned identically (per-player `space_created_m2` +
  `space_denied_m2_opponent`); `space_creation_xfns` is 2 features × 3 gamestates = **6 VAEP
  columns**. A retired-columns guard test blocks any resurrection. This answers the round-2
  question "is team destroyed expected to acquire real values?": NO — the column no longer
  exists (round-2 acceptance #1's non-zero-opponent-created clause is mathematically
  unsatisfiable under removal-based LOO; producing it would need a repositioning-counterfactual
  estimand, out of scope).
- **Liveness gate gains the round-2 non-constant check**: every float metric column added by
  any of the 28 aggregators with >= 2 observed values must carry > 1 distinct value, with a
  declared, justified, invariant-tested `STRUCTURAL_CONSTANTS` registry (never silent
  exclusions). The multi-domain fixture gained real per-window variation (velocities,
  y-layout, GK drift, kick power/timing, event-clock jitter off the frame grid, an isolated
  sprinting carrier, receivers ahead of the block, one-man-down windows).
- **New finding flagged by the gate — `pitch_control_at_ball__spearman` is near-ball
  degenerate**: the Spearman PPCF deviates from the 0.5 fallback only ~18 m+ from the ball
  (the ball reaches nearer cells before any player's reaction time), and the column samples
  linked-action START points, which are always near the ball — so it is **~0.5 for every
  well-linked action in production**. Declared + invariant-tested as a structural constant;
  lakehouse should treat the column as informationally dead pending redesign (tracked in
  TODO). This is the third dead-metric instance the gate's bug class covers.

Hyrum: BREAKING schema change vs 4.23.0 (columns dropped + renamed; opponent values change),
accepted by owner/lakehouse agreement — no consumer adopted the 4.23.x surface and the 4.23.x
line is superseded. **Final adoption column list: `space_created_m2`,
`space_denied_m2_opponent`.** Not a VAEP retrain trigger (xfns opt-in, in no default list).

## [4.23.0] — 2026-06-11

### Added — the space-creation `*_opponent` triplet is IMPLEMENTED (TF-41; lakehouse-mandated; ADR-026)

The lakehouse rejected 4.22.2's contract-removal resolution and mandated implementation
(option 1 of the original report). `add_space_creation` now emits a live
`space_created_m2_opponent` / `space_destroyed_m2_opponent` / `net_space_m2_opponent`:

- **Semantics**: the actor's leave-one-out differential OBSO evaluated on the **opposing
  team's OBSO surface** (actor as defender of that surface), per Fernandez & Bornn (2018).
  `*_created_m2` >= 0 is opponent space existing because of the actor's presence;
  `*_destroyed_m2` >= 0 is opponent space the actor's presence denies (the defensive-value
  reading); `net_*` = created − destroyed (signed).
- **Identical inputs by construction**: same linked frame, evaluation grid, OBSO sigmas,
  transition/EPV grids, and pitch-control method as the `_team` triplet — magnitudes are
  directly comparable. Analytical path (Spearman/F&B) derives the opponent surface from the
  complement of the SAME decomposed baseline (zero extra pitch-control computations);
  the Voronoi naive fallback recomputes the opponent surface explicitly per removal.
  Verified by an analytical-vs-naive opponent parity oracle on both decomposable methods.
- **Opponent resolution** is dtype-robust (`ids_match`, ADR-019). A linked frame without
  exactly two team ids **raises `ValueError`** carrying the game/period/frame/action key —
  corrupt input fails loud, never silent NaN. NaN actor identifiers still route to the
  ADR-003 NaN-row default.
- **NaN-mask parity**: the `_opponent` triplet is NaN exactly where the `_team` triplet is
  NaN (single-call design — no new degradation paths). Gated by a coverage-parity test.
- **Contract lockstep**: `_SPACE_CREATION_COLUMNS` (6), both return paths, the docstring,
  and `space_creation_xfns` (now 6 features × 3 gamestates = **18 VAEP columns**, was 9).
- **Meta-gate (recurrence guarantee), repo-wide**: `tests/tracking/test_aggregator_column_liveness.py`
  runs EVERY registered tracking `add_*` (all 28, including the jersey-frames helper) on a
  multi-domain fixture (pass / shot / GK goalkick / attacking-third ball / wide-area cross
  windows with the actor carrying the ball) and asserts every column an aggregator ADDS is
  non-null somewhere — a documented contract column that is 100%-NaN now fails CI for ANY
  aggregator, with NO exception set (conditional columns get domain-exercising fixtures, not
  exclusions) and a meta-assertion pinning the gate surface to `tracking.__all__` so a new
  aggregator cannot land unwired. Plus the space-creation-specific lakehouse acceptance
  tests: coverage parity, symmetry sanity, sign/range oracle, two-team guard.
- `compute_space_created` gains `include_opponent_perspective: bool = False` (additive;
  default output schema unchanged).

Hyrum: `space_creation_xfns` length changes 3 → 6 (opt-in factory, in no default xfn list —
opting in remains a self-triggered VAEP retrain per ADR-005). `add_space_creation` output
gains 3 columns; existing 3 are byte-identical. Lakehouse re-adds the bronze column
(`ADD COLUMNS` adoption PR) and extends its value-audit oracles. Minor bump per the
lakehouse release-mechanics requirement (contract re-expansion must not ship as a patch
on top of 4.22.2's removal).

### Changed — pyright now gates `tests/` + `scripts/` in CI (infra-only, no wheel change)

- **CI type-gate widened from `pyright silly_kicks/` to the full tree** (config-driven: pyproject
  `[tool.pyright] include = ["silly_kicks", "tests", "scripts"]` + `extraPaths = ["scripts"]` so
  the tests' runtime `sys.path` import of scripts-modules resolves statically). 301 pre-existing,
  never-gated diagnostics across 73 files fixed to zero.
- **`scripts/` fixes carry real hardening** (behavior-neutral at every site): explicit
  `RuntimeError("HPO produced no best candidate")` narrowing in `train_xcross_attempt.py` /
  `train_xshot_occurrence.py` (previously a latent end-of-sweep `AttributeError`); post-`fit()`
  Optional-narrowing asserts in `train_gk_completion.py` / `train_ghost_gk.py`; honest return
  annotations on the `_extract` helpers (pyright NoReturn mis-inference).
- **`tests/` fixes are type-only**: trailing `# type: ignore[...]` per the codebase idiom for
  pandas-stubs/numpy-stubs strictness, a handful of precise annotations (e.g. `MockSurface`
  attribute declarations in `test_obso.py`), and `[import-not-found]` suppressions on the two
  importorskip-guarded optional deps (`statsbombpy`, `xarray`). Every edited test file verified
  byte-identical pass/skip outcomes against its pre-edit baseline.
- Known suppressed-not-fixed class: `ruthless` `IntRange`/`Choice`/`FloatRange` `.log` and
  `StoreConfig` Optional annotations are stub gaps in the ruthless package itself
  (runtime-verified present); fix belongs upstream in ruthless, after which the
  `tests/calibration/test_spaces.py` suppressions can drop.

No library code changed (`silly_kicks/` untouched); not a retrain trigger; nothing re-materializes.

## [4.22.2] — 2026-06-11

### Removed — dead `*_opponent` triplet dropped from the `add_space_creation` contract (TF-41)

- **Breaking (column removal): `add_space_creation` no longer emits
  `space_created_m2_opponent`, `space_destroyed_m2_opponent`, `net_space_m2_opponent`.**
  The triplet had been hard-coded `np.nan` on every code path since its introduction
  (3.21.0, PR-S57) — a schema-only dead contract confirmed 100%-NULL across all four
  tracking providers by the lakehouse action-context pipeline (bug report 2026-06-11).
  The TF-41 spec never defined opponent-side semantics: `compute_space_created` is the
  attacking-team leave-one-out differential OBSO (Fernandez & Bornn 2018), and
  `space_creation_xfns` was always deliberately team-side only. An opponent-side metric
  (the actor's leave-one-out effect on a counterfactual opponent-attacking OBSO surface)
  would be a new research feature with its own sign/EPV-mirroring design — not a fill-in
  of these columns. The team triplet (`space_created_m2_team`, `space_destroyed_m2_team`,
  `net_space_m2_team`) is unchanged, byte-identical.
- The contract gate (`tests/tracking/test_space_creation.py`) now asserts the emitted
  columns are exactly the team triplet **and that each populates** (no dead column can
  silently re-enter the contract).

Hyrum note: consumers that mirror the documented column list (lakehouse
`bronze.spadl_action_context`) drop the dead `*_opponent` columns on adoption; no values
change anywhere, so nothing re-materializes. Not a VAEP retrain trigger
(`space_creation_xfns` output is unchanged).

## [4.22.1] — 2026-06-11

### Fixed — lakehouse bug-report 2026-06-11 hardening (ghost-GK clamp, completion-variant alias)

Four small fixes from the lakehouse 4.22.0 production report (items confirmed against source; the
two suspected value bugs — `xt_gk_pev` ≈ 0 and `obso_peak > obso_optimal` — were verified
**by-design** and are documented below rather than changed):

- **Ghost-GK served position clamped to the physical pitch** (`compute_ghost_gk`): garbage input
  (e.g. an upstream mis-flagged `is_goalkeeper`, which can wrong-foot the per-period goal-side flip)
  can push the boosted regressor far outside its trained label domain — a served keeper 5.7 m
  *behind the goal line* is never physically meaningful. Served `ghost_gk_x/y` (goal-relative) are
  now clamped to x ∈ [0, 105], y ∈ [0, 68] with a warning. Clamp target is the **physical pitch,
  not the trained grid domain** — healthy slight extrapolation past the 30 m label filter (sweeper
  rushes) stays **byte-unchanged**, so this only ever fires on corrupt input. The clamp lives at the
  serving seam; `GhostGkModel.predict_mean` keeps its exact-boosted parity contract (ADR-016).
- **`GkCompletionModel.from_variant("gs")` no longer raises `FileNotFoundError`**: variant KEYS
  (the `variant_key_for_provider` vocabulary, where `"gs"` names the GS-construct model) now alias
  onto the bundled weight DIRS (`"gs"` → `"default"`), so the two public APIs compose. Same shared
  cached instance; no behavior change for `compute_xt_gk` (its private resolver already fell back).
- **`tracking.gradientsports.convert_to_frames` `home_team_id` annotation fixed to `int | str`**
  (runtime has been dtype-safe + fail-loud-on-zero-match since 4.15.0/ADR-019; the annotation and
  docstring now say so).
- **`compute_pass_obso` docstring**: `peak_obso` (max over *time* at the fixed target) and
  `optimal_obso` (max over *teammate positions* at the event frame) maximize different axes and are
  **not mutually ordered** — `peak > optimal` is legitimate; both dominate `actual_obso`.

By-design confirmations for the report: `xt_gk_pev = rho × max(0, progress)` is exactly 0 whenever
no opponent is inside the Andrienko pressure oval (~9 m) — structurally true for every goal kick
(law: opponents outside the box) — or the move is non-forward; the emitted `xt_gk_pressure` column
is `rho` and discriminates the two. `LinkReport.per_period_link_rate` (requested as new) has shipped
since 4.12.0/ADR-017.

Hyrum note: the ghost-GK clamp is a serve-output change **only on physically-impossible rows**
(observed: metrica with a corrupted upstream GK flag). No retrain trigger; lakehouse re-materializes
ghost-GK only if it wants the clamped values for already-ingested corrupt matches.

## [4.22.0] — 2026-06-10

### Added — general restart-coordinate enrichment (Phase 1, additive; ADR-025)

New public `silly_kicks.spadl.add_restart_coordinates(actions, *, frames=None, links=None)` imputes
missing coordinates for Law-fixed-spot restart types — goal-kick (6-yard box), penalty (spot), corner
(arc), throw-in (touchline) — and emits them as **new** provenance-tagged columns
(`enriched_start_x/_y`, `enriched_end_x/_y`, `start_coord_source` / `end_coord_source`,
`start_coord_confidence` / `end_coord_confidence`), **never mutating** the canonical
`start_x/start_y/end_x/end_y`. Frames-optional: with `frames` supplied the tracking-ball / in-area
tracking-GK tiers raise confidence; events-only uses native / rule-point / next-event tiers. A
geometry tripwire (à la ADR-018) reverts an imputed origin outside its Law region to
`tripwire_reverted` (warns); native out-of-region coords warn only. Optional aggregate
`silly_kicks.tracking.RestartCoordinateReport` (counts per source + `n_tripwire_reversions`).

This promotes the goal-kick-scoped `resolve_gk_geometry` (ADR-024) into a single general engine
`silly_kicks.tracking.resolve_restart_geometry` (parameterised by `impute_types`); `resolve_gk_geometry`
is now a thin, **byte-identical** shim over it (`impute_types=(goalkick,)`), so xT-GK / completion and
all 4 internal callers are unchanged — **no model retrain**. Scope grounded by a live lakehouse probe:
NaN coordinates are a Gradient Sports set-piece phenomenon (StatsBomb/Wyscout/SkillCorner are 0%), so
the Law-geometry prior is defensible. The canonical-coordinate promotion (which WOULD retrain
VAEP/xT/calibration) is a deferred Phase 2 (separate PR). Additive only — no existing behavior or
output changes.

## [4.21.4] — 2026-06-10

### Changed — xT-GK per-type base-rate serve switch (goal-kick completion honesty)

`compute_xt_gk` now serves the **per-type calibrated base rate** (tagged `xt_gk_completion_source =
"base_rate"`) instead of the geometric model for any completion-variant sub-domain whose held-out AUC
lower-confidence-bound ≤ 0.5 (or degenerate / below a minimum sample) — the gate is a single
`serve_mode_from_lcb(lcb, n)` decision baked into the `GkCompletionModel` artifact (`_type_serve_mode`
\+ `_type_gate_metrics`, version 1.1.0). `load()` **fail-opens**: a pre-gate (4.21.0) artifact serves
all types `"model"` = prior behavior. The switch is **data-driven per variant, not a blanket
goal-kick rule**: the bundled **SkillCorner** gate routes **goal-kicks → `base_rate`** (held-out AUC
0.433, near-chance from tracking geometry) while keeping GK-passes model-scored (AUC 0.737); the
bundled **GS `default`** keeps **goal-kicks `model`-scored** (AUC 0.836, LCB 0.798 — GS goal-kick
completion *is* predictable from geometry). Near-empty throw-in sub-domains (degenerate AUC) base-rate
by construction in both.

Coefficients are **byte-unchanged** — the re-bundle attaches the gate onto the committed model
(corpus-identity-guarded; the guard tolerates the unrecorded-`tracking_limit` density float noise but
aborts on a real retrain). **Not a VAEP retrain** (xt_gk is opt-in, in no default xfn list) — but an
`xt_gk` serve-output change for the flipped types: the lakehouse re-materializes `xt_gk` for the
**SkillCorner goal-kick rows (~15% of its GK-distribution actions) plus degenerate throw-ins (both
variants)**; GS goal-kicks are unaffected. ADR-024 amendment. (4.21.0 §2.3/m3 follow-up.)

## [4.21.3] — 2026-06-09

### Changed — sportec DFL `play_evaluation` success-allowlist (completion robustness)

Native sportec pass/set-piece completion now uses a **success-allowlist** (`fail` iff the DFL
`Evaluation` is non-empty and ∉ `{successfullyCompleted, successful}`) instead of an exact
`== "unsuccessful"` match — so any unseen reason-coded failure token (e.g. `unsuccessfulBecauseOfFoul`)
is failed by construction, and a missing/empty `play_evaluation` still maps to success (no mass-fail
on non-DFL data). Single-sourced across the main and synth-distribution sites (`_extract_play_eval` +
`_play_evaluation_is_fail` + `_warn_unexpected_play_eval`); an unexpected token is warned, not silently
classified. **Aligns the native converter with the kloppy gateway** (same success set) and is
**byte-identical on observed DFL data** — verified on all 7 IDSSE matches, whose only non-success
`play_evaluation` token is `unsuccessful` (robustness hardening, not a re-mapping). Hyrum surface: a
DFL stream carrying failure tokens beyond `unsuccessful` would shift its fail distribution. Adds a
CI-everywhere native-shape distribution regression test and an owner-gated Databricks-bronze e2e over
the 7 IDSSE matches (`fetch_idsse_events`). No shipped-API change. (TODO 4.20.1 follow-up; refines BUG-2.)

## [4.21.2] — 2026-06-09

### Added — owner-gated lakehouse-mart xT held-out-NLL cross-check

A permanent `@pytest.mark.e2e`, owner-gated regression tripwire
(`tests/test_xthreat_nll_lakehouse_e2e.py`) triangulating KDE-vs-Singh held-out transition-NLL on
**passes** against `soccer_analytics.dev_gold.fct_action_values` (the 4.17.0 work ran this as a
non-committed one-off; ~4% relative KDE win on ~8.9M actions). Fits on the full train, scores a
passes-only holdout (parity with the StatsBomb sibling + the published "Held-out NLL (passes)"
3.789→3.748), and **on the full corpus only** hard-asserts at 16×12 that the tuned KDE(4.0) clears a
conservative 1.5% relative-win floor AND the shipped-default KDE(1.0) strictly beats Singh
(no floor — the default's margin erodes with corpus growth); logs 12×8. Adds the
`fetch_action_values` + pure `shape_action_values` mart helpers to `scripts/_loader_databricks.py`
(unit-tested) and pure `nll_relative_win` / `kde_clears_tripwire` verdict helpers (unit-tested).
Skips wherever the owner Databricks credentials + `databricks-sql-connector` are absent (public CI).
**No shipped-library change** — every artifact is in `scripts/` + `tests/`; the `silly_kicks/` wheel
is unchanged except `__version__`. Additive — no behavior change, no retrain trigger. (TODO SK-xT-1
follow-up; ADR-021.)

## [4.21.1] — 2026-06-09

### Changed — ADR-019 AST lint extended to the converter-adapter orientation seam

The boundary lint (`tests/tracking/test_id_compat_lint.py`) no longer blanket-skips the tracking
converter adapters. `ALLOW_MODULES` is narrowed from
`{_id_compat.py, sportec.py, gradientsports.py, kloppy.py}` to **`{_id_compat.py}`** — the helper
module that defines the primitives is now the *sole* exemption, so every tracking module (converter
adapters included) has its id comparisons under the lint. This closes the gap that let **BUG-4** —
the 4.20.1 frame-orientation dtype bug, a raw `team_id == home_team_id` that silently matched zero
players for an int arg vs object-string frames (the `structural_sgm` away-team blow-up root cause) —
reach production: it was a fourth ADR-019 id-dtype instance, and the over-broad file-skip hid it.

- `gradientsports.py` / `sportec.py` `convert_to_frames` already use `ids_match` (the 4.20.1 fix);
  un-skipping them puts the orientation seam under the lint.
- `kloppy.py`'s orientation comparison is routed through `same_id` (it was `str()`-vs-`str()`
  internal — no caller-dtype boundary — so this is **behavior-identical**, chosen for one consistent
  rule and zero per-module exemptions).
- Two guards lock the narrowing: a discriminating proof that the detector actually fires on the BUG-4
  shape (distinguishing a genuinely-clean adapter from a detector that never fires for the shape), and
  an anti-regression assertion pinning `ALLOW_MODULES == {_id_compat.py}`.

ADR-019 amendment. The single library-code change (kloppy's `==` → `same_id`) is behavior-identical
(str-vs-str): no behavior change, no retrain trigger — the BUG-4 *fix* shipped in 4.20.1; this guards
the *class*.

## [4.21.0] — 2026-06-09

### Added — xT-GK (Eyestone): Expected Threat for Goalkeepers (ADR-024)

A new **pure parametric compute feature** (not a trained model) that re-values goalkeeper
distribution actions (goal-kicks, keeper passes/throws), implementing Jeffrey Eyestone's
**xT-GK** (winner, Pitch to the Pros 1) publicly with his attribution. Tracking-required
(the pressure-escape component needs a pressure signal, which no provider preserves through
SPADL). Lives in `silly_kicks/tracking/_xt_gk.py` with the standard ADR-005 surfaces:

- `compute_xt_gk` / `add_xt_gk` (`@nan_safe_enrichment`) / `xt_gk_xfns` (VAEP factory) + atomic mirror.
- `XtGkParams` frozen dataclass + `XtGkParams.for_philosophy(...)` (possession / counter / direct /
  high_press / low_block presets, provisional in-range values).
- Emits raw components `xt_gk_base` / `xt_gk_pev` / `xt_gk_rav` / `xt_gk_dzv` / `xt_gk_pressure`
  plus the composite `xt_gk`, per GK-distribution action.

Design (all confirmed with Jeffrey, 2026-06-08): the destination value is counted **once**
(owned by the risk-adjusted term; the composite base is origin-only — **Option B**); RAV's
pass-completion probability comes from a fitted **`GkCompletionModel`** (see goal-kick coverage
below); the baseline xT grid is a **required caller-injected, pre-fitted `ExpectedThreat`**
(no self-fit, no leakage); the interpretive parameters are intent-set and never calibrated.

In **no** default xfn list — opting xT-GK into a VAEP model is a deliberate, self-triggered
retrain. No change to any existing feature (no retrain trigger). Phase 2 (opt-in team/dataset
parameter estimation) is deferred. Attribution + consent trail in `NOTICE` and ADR-024.

#### Goal-kick coverage — coordinate derivation + RAV completion model (ADR-024 amendment)

The owner-gated OOD smoke escalated: accessible-space's open-play xC resolved for only ~31%
of real goal-kicks (long aerials are out of its validated regime), and ~67% of real GS
goal-kicks carry a NaN origin — together capping real goal-kick coverage at a small fraction.
Both are closed **honestly tagged**, so the composite is defined for ~all in-scope goal-kicks
*with a resolvable destination* and every value carries machine-readable provenance:

- **Coordinate derivation** (`resolve_gk_geometry`, `silly_kicks/tracking/_gk_geometry.py`):
  a **scoped, conditional** origin (native → in-area tracking-GK clamped to `x ≤ 16.5 m` →
  goal-area rule point `(5.5, 34)`) + destination (native → in-period next-event start, guarded
  at `(game_id, period_id)` boundaries) that **feeds the valuation internally and NEVER mutates
  the shared `actions` frame** (a converter-level coordinate change would be a Hyrum/retrain
  trigger for every downstream consumer). Per-row provenance + a continuous confidence are
  emitted: new output columns `xt_gk_origin_source`, `xt_gk_dest_source`,
  `xt_gk_origin_confidence`, plus an optional aggregate `XtGkReport` for pipeline QA.
- **RAV completion model** (`GkCompletionModel`, `silly_kicks/tracking/_gk_completion.py`):
  a **logistic** GK-distribution pass-completion model (sklearn at fit, pure-numpy
  `sigmoid(Xβ)` at serve — **no new runtime dependency**), trained on the observed SPADL
  `result_id == success` label. Bundled GS `default` (30 WC2022 matches, native-origin pooled
  out-of-fold gate: AUC 0.838, CI95 [0.81, 0.86], n_native 1395, Brier 0.122 < base 0.171);
  pickle-free JSON + SHA256 envelope; `from_variant("default")` with a caller `completion=`
  override. Missing-value policy: per-feature density NaN → training-mean impute (neutral after
  standardization); whole-row geometry-unscoreable → per-type base rate (standalone
  `compute_gk_completion` only — the RAV path NaNs unresolvable-destination rows honestly).
- **`[das]` is no longer required** for xT-GK; `compute_xt_gk` / `add_xt_gk` gain a
  `completion: GkCompletionModel | None = None` kwarg. `compute_gk_completion` and
  `add_gk_completion` are exported -- the latter is the lakehouse wide-table aggregator,
  emitting a `gk_completion` column per in-scope GK distribution (NaN out-of-scope) by reusing
  RAV's exact scoring path (geometry on the full action list, then masked), so the column
  equals the P(success) RAV consumes. Train==serve parity is enforced at every producer (shared
  domain predicate, shared geometry resolution on the full action list before masking, shared
  density producer, shared feature extract).

#### SkillCorner completion: native-`result_id` fix + provider-aware variant family (ADR-024 amendment)

Makes SkillCorner `xt_gk` construct-correct and poolable with Gradient Sports.

- **SkillCorner `result_id` → native completion (`silly_kicks/spadl/skillcorner.py`).** The converter
  previously labelled pass/set-piece completion with a `same_team_next` possession proxy, which
  agrees with the native outcome only ~0.72–0.79 and **overstated goal-kick success by ~16 pp**
  (0.86 vs the true 0.70). It now routes `result_id` through the **single native construct** —
  `pass_outcome` (SPADL "reached a teammate") → `received==True` (success-only) → residual
  `same_team_next` — with a new dedicated **`result_source`** column (`native` / `inferred` /
  `stopgap`) recording the per-row label tier. **VAEP-retrain trigger** (SkillCorner scores/concedes
  label distribution shifts; the lakehouse re-materializes SkillCorner VAEP). `received==False` is
  never treated as a failure (it can be a completion to a non-targeted teammate).
- **Provider-aware completion variant** (`GkCompletionModel`): pure `variant_key_for_provider`
  (`skillcorner` → its own weights; everything else → the native-completion `gs` default) + auto-
  selection in `compute_xt_gk`/`add_xt_gk` from `frames["source_provider"]` (caller `completion=`
  override wins; >1 real provider raises; `snapshot` excluded). The GK-completion model trains on the
  **`native` tier only** (`pass_outcome`) — `inferred`/`stopgap` are positive-only / proxy and would
  bias the multiplicatively-consumed calibration.
- **Bundled `skillcorner` variant** (10 SkillCorner matches; GS-transfer re-measured on the corrected
  native label was **0.412** GK-pass AUC, worse than chance → distinct weights required). SkillCorner
  GK-pass **AUC 0.739, ECE 0.036**; goal-kicks are **chance (0.433)** from geometry — model-scored but
  a documented low-discrimination limitation (base-rate-equivalent in practice, on-scale per the
  comparability gate). `from_variant("skillcorner")`.
- **Pooling safety:** new provenance columns `xt_gk_completion_variant` / `xt_gk_completion_source` +
  `XtGkReport.spans_multiple_variants`; a cross-provider comparability gate
  (`scripts/_xtgk_comparability.py`, owner-run) found SC-vs-GS `xt_gk` **within tolerance** on matched
  distance bands → pool directly, no re-scale. The "do not pool across variants without a validated
  comparability" contract is documented (ADR-024).

## [4.20.1] — 2026-06-09

### Fixed — provider data-quality bugs (SkillCorner time-base + goalkick; sportec pass completion; SGM bound + frame-orientation dtype)

Four data-quality defects surfaced while validating GK-distribution completion cross-provider
(corroborated + root-caused with the lakehouse bronze). **The SkillCorner, sportec, and
frame-orientation fixes change VAEP/tracking label/feature distributions for those providers —
retrain triggers.**

- **SkillCorner `time_seconds` is now period-relative (BUG 1, ADR-017).**
  `silly_kicks/spadl/skillcorner.py::_parse_time_start` parsed SkillCorner's `"MM:SS"`
  *continuous broadcast clock* literally, so 2nd-half/ET events landed at ~2700–5800 s while the
  period-relative tracking frames reset to 0 — collapsing action↔frame linkage for the entire
  2nd half + ET (every frame-linked tracking feature silently degraded there). New
  `_to_period_relative` subtracts the period-start offsets `{1:0, 2:2700, 3:5400, 4:6300, 5:7200}`.
  Regression-guarded by a unit test + a strengthened owner-gated e2e (the old check only asserted
  intra-period monotonicity, which a continuous clock also satisfies).
- **SkillCorner goalkick result no longer hard-wired to success (BUG 2).** It was unconditionally
  `success`, bypassing the `same_team_next` possession check used for every other pass; now routed
  through it (lost-to-opponent → `fail`).
- **sportec pass/set-piece completion from native DFL `play_evaluation` (BUG 2).**
  `silly_kicks/spadl/sportec.py` marked *every* pass/cross/freekick/corner/throw-in/goalkick
  `success`, ignoring the `play_evaluation` attribute it already parsed. Now: `unsuccessful` →
  fail; `successfullyCompleted`/`successful`/NULL → success (conservative). Applies to Play,
  set-piece events (which carry it via their nested Play), and the punt-synthesised goalkick
  (inherits its parent Play's evaluation). DFL goalkicks are ~71% complete, not 100%.
- **`metrica.py` left unchanged (measured-correct).** Metrica represents pass loss as a separate
  `BALL LOST` event; a `PASS` is a *completed* pass (98% same-team-next in the fixture, losses
  never attached to a `PASS`), so `result=success` is correct by design.
- **`structural_sgm` numeric blow-up bounded (BUG 3, symptom).**
  `silly_kicks/tracking/_structural_pass.py`: `sgm = 1/rho_r − 1/rho_p` exploded to ~±1e8 when the
  passer/receiver was far from all defenders (the σ=15 "intrinsically bounded, no eps-floor"
  claim was falsified on real byline-cross / fast-break frames). `rho` is now floored at a
  defender's 3σ contribution (`exp(-4.5)≈0.0111`), capping `1/rho`≈90; normal-geometry values are
  unchanged. The falsified docstring is corrected. Defense-in-depth for BUG 4 below.
- **Frame-orientation `home_team_id` dtype bug (BUG 4, ADR-019) — the SGM root cause.**
  `gradientsports.py` and `sportec.py` (tracking adapters) set `team_attacking_direction` via a
  raw `team_id == home_team_id`, which silently matched **zero** players when `home_team_id` was
  passed as `int` and the frame `team_id` was object-string (`"366"`) — every player mislabeled,
  then `play_left_to_right` double-flipped, producing **mis-oriented frames** (the ~4× away-team
  SGM blow-up, and a latent corruption of *every* frame-linked tracking feature whenever the
  caller's `home_team_id` dtype mismatched). Both now use the dtype-safe `_id_compat.ids_match`
  and **fail loud** if `home_team_id` matches no player. Regression-guarded by an int-vs-str
  orientation-invariance test. (The kloppy gateway is unaffected — it derives `home_team_id`
  internally as a string.)

## [4.20.0] — 2026-06-08

### Added — SK-xT-3 calibration-integrated xT bandwidth/resolution sweep (ADR-009, ADR-021)

`silly_kicks.calibration.xt_bandwidth_config` + `XtBandwidthObjective` — a `ruthless`/Optuna sweep
over xT `KDEParams.bandwidth` × `GridSpec` resolution × `adaptive` minimizing K-fold held-out
transition-NLL, with the Singh no-smoothing baseline reported alongside. Recommends a
`KDEParams`+`GridSpec` via an auditable manifest (`scripts/calibrate_xt_bandwidth.py`); **changes no
library default** (ADR-009). The recommendation is scoped to held-out *destination likelihood*
(xT-quality impact reported, not asserted) and a downstream Spearman cross-check vs realised goals
is emitted. The CLI supports download/parse caching for repeated runs: `_loader_pining.load_matches`
gains an opt-in `cache_dir` (persistent, atomic-write artifact cache), and the CLI adds `--cache-dir`,
`--corpus-cache` (assembled-corpus parquet — skips download+parse on re-runs), and `--subsample-games`
(corpus-size contrast off the cache). The corpus is canonicalised to the standard SPADL columns +
string-cast ids so the multi-provider parquet is serialisable.

### Changed — vectorized gaussian xT KDE core (internal; no public-API change)

`kde_smoothed_transition_matrix` now factors a shared, vectorized gaussian seam
(`_gaussian_transition_from_grouped`) — softmax-stabilized, much faster per call, sklearn retained
only for non-gaussian kernels. The gaussian numerics are re-pinned (Chesterton-verified: one caller,
`singh_counts` default) and now stay finite/correct in the small-bandwidth regime where the previous
sklearn-wrapper underflowed to the mean-row fallback.

## [4.19.2] — 2026-06-08

### Changed — CI slow-test gating: invariant heavy tests on a single primary leg (ADR-023)

CI-/test-infra only — **no runtime change** (the wheel is byte-identical; `silly_kicks/` is untouched).
The `test` matrix previously ran the full non-e2e suite on all 4 legs (ubuntu 3.10/3.11/3.12 +
windows 3.12), making the slow Windows runner a ~16–20 min long pole. The expensive
**platform-/interpreter-invariant** tests (train-script smokes, same-run internal-consistency / KDE
parity, calibration cache-equivalence) now carry `@pytest.mark.slow` and run **once on a primary leg**
(`ubuntu-latest` 3.12, identified by a matrix `primary: true` flag); every other leg runs
`-m "not e2e and not slow"`. The `--benchmark-only` step is likewise primary-leg-only.

The `slow` set was chosen from **real Windows-leg CI durations** (local profiling is not a faithful
proxy). **Version-sensitive tests** (golden-hash / snapshot / absolute-numeric) and cheap
behavioral-contract guards (dup-`action_id`, id-dtype-invariance, orientation/roster) are deliberately
**not** marked `slow` — they stay on all legs (OS + interpreter axes). The matrix partition is guarded
structurally by `tests/test_ci_slow_gating_wired.py`; `pyyaml` is now a direct `[test]` dep (the
tripwire's parser). No xdist (it OOM-killed the runners before). Decision: ADR-023.

## [4.19.1] — 2026-06-08

### Added — TF-27 SkillCorner derived-GK Tier-1 roster validation (PR-S86, ADR-007)

Upgrades `_gk_identification.derive_goalkeepers` validation for SkillCorner from Tier-2
(algorithm self-consistency) to **Tier-1** (external ground truth). A new owner-runnable
e2e (`tests/tracking/test_gk_skillcorner_roster_e2e.py`) anchors `derived_gk_picks` against
the pining `match.json` roster GK (`player_role.acronym == "GK"`) per team, with an
exact-set-equality gate (catches over-identification) + a fail-loud join-key guard. Verified
**20/20 team-GKs across all 10 public A-League matches** — no algorithm change required.
A CI-runnable synthetic guard (`tests/tracking/test_gk_skillcorner_roster.py`) shares the same
pure comparator (`tests/_skillcorner_sample.py`), and `scripts/download_skillcorner_sample.py`
populates the sample dir (also unblocks the existing SkillCorner SPADL e2e). Metrica external
verification remains impossible on public anonymized data (no roster) — a documented permanent
limitation (ADR-007).

### Changed

- Refactored `scripts/_loader_pining._build_skillcorner` to delegate frame construction to a
  new `build_skillcorner_frames` seam (single frame path; verbatim relocation, no behaviour
  change — calibration unaffected). Breadcrumb for future calibration work.
- ADR-007 / CLAUDE.md: SkillCorner derived-GK identification recorded as Tier-1 external-roster
  validated.

## [4.19.0] — 2026-06-08

### Added — xT as a VAEP feature (`xt__<method>` xfn factory, ADR-022)

`silly_kicks.vaep.features.xt_xfns(*, model)` (and its atomic mirror
`silly_kicks.atomic.vaep.features.xt_xfns`) wire a fitted `ExpectedThreat` into the VAEP
feature framework as a **frame-free**, opt-in feature transformer. It emits one
`xt__<model.method>` column per gamestate slot (`xt__singh_counts_a0/_a1/_a2`,
`xt__kde_smoothed_*`), following the ADR-005 §8 `<feature>__<method>` naming convention, and
preserves `ExpectedThreat.rate`'s NaN contract for non-move / failed-move actions.

- **Caller-supplies-the-model.** The factory closes over a *fitted* `ExpectedThreat` and fails
  closed otherwise (`None` → `ValueError`, an unfitted model → `NotFittedError`, a `str` →
  `NotImplementedError` — a reserved door for a future bundled-grid variant). Train/serve
  consistency is the caller's responsibility: fit + freeze the grid once and reuse the identical
  object at serve time (mirrors the `FrozenXt` / ADR-009 discipline). `ExpectedThreat` is imported
  only under `TYPE_CHECKING` (duck-typed at runtime) — **no new runtime dependency edge**; bare
  `import silly_kicks` is unaffected.
- **Opt-in — no forced retrain.** `xt_xfns` is in **none** of the default/union xfn lists; opt in
  with `VAEP(xfns=xfns_default + xt_xfns(model=frozen_xt))`. A guard test enforces its absence from
  the defaults. **Opting it into your own xfns is a self-triggered VAEP retrain.**
- **Atomic mirror reuses `model.rate()`** (unchanged) via a synthesized standard-SPADL frame with a
  **type-aware** `result_id` — dribbles are intrinsically successful (never followed by a
  `receival`); pass/cross success iff the next atom is `receival`. A blanket next-atom test would
  NaN every dribble; the type-aware predicate keeps `xt__<method>` column-symmetric across both
  SPADL flavours (verified by a geometry-keyed cross-representation oracle on the committed WC2018
  fixture, plus a dribble keystone gate). Slots map by the composite
  `(game_id, period_id, action_id)` key. A pass/cross that is the last action of a period has no
  following atom and yields NaN (inherent atomic-representation edge; documented).

`ExpectedThreat.rate()` is left **byte-identical** (no `_rate_cells` extraction) — the SK-xT-1
parity gate and golden snapshots are untouched. Decision: ADR-022; attribution Singh (2018).

## [4.18.0] — 2026-06-07

### Added — TF-17 xCrossAttempt (xCross) TRAINED weights + GK validation + TF-19 wiring (PR-S85)

The weights follow-up to PR-A's untrained code (4.11.0). Bundled the **`public`** xCrossAttempt
model (skillcorner + idsse), trained on the clean-4.13.0-GS pining corpus (81 matches, 701,210
wide-area frames / 11,930 cross-positives) against the 4.7.0 carrier defaults, on DGX Spark.
A pre-registered `public`-vs-`full` two-candidate paired test (common public held-out, shared
params) found owner-tier Gradient Sports data **degraded** public generalization in **all 5 folds**
(Δ PR-AUC −0.009…−0.067) → shipped the reproducible public-only model (no Hub repo, mirrors xS).
public CV: PR-AUC 0.0606 > base 0.0177; Brier 0.0172 < 0.0173; log-loss 0.0841 < ln2.
`from_variant("default")` + `from_hub` live; `xcross_attempt_xfns` wired into
`pre_shot_gk_full_default_xfns` (+ atomic mirror) **only**, not the general default.

### Validation (reported in the bundled metrics.json; the GK-extension headline)

- **`tf19_ready = False`** (pre-registered inert-GK contingency): the GK substitution-sensitivity
  probe moves P(cross) by a median **0.00107** on a realistic GK shift — **2.6× the nearest-defender
  control** (0.00041) and ∞× the random-outfielder band (0.0), i.e. GK position carries *relative*
  signal, **but below the pre-registered absolute floor (0.01)** — too small to drive a meaningful
  TF-19 `Δ_cross`. The surface ships regardless (a weak signal is not a build break); TF-19 (GKDV
  Layer 3) consumption is gated on GK feature-engineering first, never shipped silently as novelty.
- GK-block ablation: Δ PR-AUC +0.0011 (≈0 marginal CV lift) — yet `gk_theta` is the #4 feature by
  permutation importance (0.0125): informative-but-collinear (the gate is the probe, not ablation).
- **`score_differential` is the #2 feature by CV-held-out permutation importance (0.0216)** at 1.0
  coverage — *material* for xCross (unlike Ghost-GK). Measured on the clean GS stream: range
  [−5, +6], 0 impossible values (the old ±18 cache would have corrupted this — the clean rebuild was
  load-bearing).

### Added — TF-17 xCross causal validation harness (PR-C, ADR-015)

The paper-faithful causal arm closing TF-17. Private `silly_kicks/_causal/` port (pure numpy/sklearn,
no R, no new dependency): propensity-score matching (ATT/ATNT, 1:1 nearest-neighbor **with
replacement**, no caliper, logistic propensity on standardized covariates, **Abadie–Imbens (2006)
matching SEs**) + a spell-based crosser-anchored opportunity builder. `scripts/validate_xcross_causal.py`
ablates the GK confounder block against a **row-permuted-GK placebo null band**, with a positivity
guard, a PS-overlap + SMD-improvement claim gate, and a GK missing-indicator. The treatment window is
`(entry, min(entry+T, spell_end)]` (fixed-`T` cap → no spell-length confounding; `spell_end` clamp →
no cross-phase misattribution); the outcome is measured strictly post-treatment. The causal finding is
a **reported** research artifact (`docs/research/xcross_causal/`), never a ship/CI gate — only the
known-truth method tests (`tests/causal/`) gate CI. Reconstructs the paper's sender-level unit;
tracking-only-opportunity-detection + league/era divergence reported, not hidden. Decision: ADR-015;
attribution arXiv:2505.11841.

### Causal result (reported in `docs/research/xcross_causal/`; clean all-provider corpus)

Run on the full 3-provider pining corpus (skillcorner + idsse + gradientsports), seed 0:
**23,966 opportunities / 669 treated (base outcome rate 4.3%)**.

- **The cross effect is real and significant.** ATT (with GK block) **+0.0927 (SE 0.0156)**; ATT
  without the GK block +0.0747 (SE 0.0167); ATNT +0.0551 (SE 0.0133) — ≈5σ. Crossing causally raises
  the ~6-second scoring-opportunity outcome by **+7–9 percentage points** over the 4.3% base.
- **The matching is valid:** propensity overlap 1.0 (no density trimming), max SMD 0.51 → **0.078**
  post-match (< 0.1) → `causal_claim_supported = True`.
- **The novel GK-position block does NOT clear the placebo band** (`gk_clears_placebo_band = False`,
  **reported, not a gate**): adding the GK block shifts the ATT by **0.0179**, below the
  row-permuted-GK placebo p95 of **0.0239** — i.e. not distinguishable from a shuffled-GK column on
  this corpus. **This independently corroborates `tf19_ready = False`:** two methods (the PR-B
  predictive substitution probe and this PR-C causal placebo ablation) now agree the GK block carries
  relative-but-not-distinguishable signal → TF-19 stays gated.
- **The GS feature fix was load-bearing.** With the `canonical_id` fix, GK/base NaN fractions are
  ~0 (8.3e-5 / 0.0) and all three providers reach carrier-coverage 1.0 (GS contributes 19,833 of the
  23,966 opportunities). An earlier run on the un-fixed extractor was a **false positive** —
  `gk_clears_placebo_band = True` driven entirely by an 82.8%-NaN GS missingness confound; the fix
  flipped it to the correct negative. (See the GS bug entry below.)

### Fixed — GradientSports xCross feature extraction returned all-NaN (silent)

`extract_xcross_features` matched the ball-carrier / goalkeeper by stringifying the frame's
`player_id` / `team_id` via `.to_numpy().astype(str)`. GradientSports tracking frames carry
**nullable `Int64`** ids, and `Int64.to_numpy()` **upcasts to float64** → `"11094.0"`, which never
equals the clean-int carrier key `"1336"` → the carrier mask matched 0 rows → **every
carrier-anchored confounder and the entire GK block came back NaN for all GradientSports frames**
(≈83% of the real corpus, and the whole shipped GS xCross-inference path). Numeric team comparisons
survived (`366.0 == 366`), so only the string player match broke; kloppy/string-id providers were
unaffected, which is why it stayed latent. Fixed by routing the id match through the ADR-019
`_id_compat.canonical_id` / `canonical_id_series` contract (collapses `366` / `366.0` / `Int64(366)`
/ `"366"` → `"366"`). The existing tests only asserted column *existence*; added
`test_int64_id_frames_resolve_carrier_and_gk_features` to assert feature *values* resolve (notna) on
Int64-id frames. The shipped public model trains on kloppy/string providers so its weights are
unaffected (no retrain); the fix repairs GS xCross *inference* (was silently NaN → xgboost-missing).

- `prepare_xcross_training_data` raised `TypeError: boolean value of NA is ambiguous` on real frames
  whose `team_id` column carries `pd.NA` (ball row / unresolved GS jersey); the defending-team
  computation now filters by `is_ball` + `dropna()` (mirrors `compute_xcross_attempt`). Surfaced by
  the maintainer-run training pilot.

### Note

- A future TF-24 carrier-default change is an xCross retrain trigger (carrier params recorded in
  metadata + consumed at inference).

## [4.17.0] — 2026-06-07

### Added — SK-xT-1: pluggable, evaluatable xT (`silly_kicks.xthreat`)

`silly_kicks/xthreat.py` is now the `silly_kicks/xthreat/` package with a pluggable transition
family in silly-kicks house style (string-dispatch + frozen-dataclass params, no ABCs; ADR-021):

- **`ExpectedThreat(method="singh_counts" | "kde_smoothed", params=..., l=, w=)`** — the
  `singh_counts` default is **byte-identical** to the prior implementation (proven by an
  in-process frozen-oracle parity gate over the WC2018 fixture + `spadl_actions`). KDE-smoothed
  transitions (`kde_smoothed_transition_matrix`, Silverman-1986 bandwidth, optional adaptive
  per-source-zone) are a new flavor; `KDEParams.bandwidth` defaults to 1.0 (pure Silverman — a
  conservative, corpus-agnostic baseline). KDE strictly beats Singh at every scale tested; the
  held-out-NLL-optimal multiplier is corpus-size-dependent (~1 on a 64-match sample, ≥4 on an
  8.9M-action mart) — tune via `compute_holdout_nll`. `singh_transition_matrix` is vectorized
  (`np.add.at`), byte-identical to the legacy per-zone loop (exact-equality parity gate).
- **`GridSpec`** — first-class variable resolution (pitch dims stay in `spadlconfig`; SSOT).
- **Standalone `value_iteration`** (extracted byte-identically from the legacy solver; optional
  `max_iter` guard, default unbounded) + **`singh_transition_matrix`** / `silverman_2d`.
- **Held-out transition-model NLL evaluator** — `holdout_split` (`game_id`-keyed),
  `compute_holdout_nll` (pure: matrix + holdout + grid), `compute_holdout_nll_per_group`. The
  first held-out xT evaluation primitive in silly-kicks. (NOT an xT-quality metric — it scores
  destination likelihood under the transition matrix.)

KNN/conditional xT (pre-publication; tracking-join-dependent) is deferred. The lakehouse `XTGrid`
typed wrapper is NOT adopted (xthreat keeps its raw `.xT` ndarray). **Additive — no behavior
change on the default Singh path, so no retrain trigger for existing consumers** (incl. the TF-24
calibration `FrozenXt`). Promotion proposed by the luxury-lakehouse session; attribution:
Singh (2018), Silverman (1986), Salimi et al. (2026, LISS poster, pre-publication). Decision: ADR-021.

## [4.16.1] — 2026-06-07

### Fixed — Sportec/DFL converter mislabelled ~99% of passes as crosses

`convert_to_actions` flagged a pass as a cross via `_opt("play_flat_cross", False).fillna(False).astype(bool)`.
DFL bronze emits the `play_flat_cross` qualifier as the native string `"true"`/`"false"`, and
`pd.Series(["false"]).astype(bool)` is `True` (any non-empty string is truthy) — so every pass whose
`play_flat_cross` was non-null, **including the literal `"false"`, became a cross**. On real Bundesliga
data this inverted the pass/cross split (e.g. match J03WMX: 875 cross / 7 pass, where cross should be
~2–4% of passes). It was the only `.astype(bool)` on a string qualifier in `sportec.py`; the sibling
qualifiers (`shot_after_free_kick`, the two `*_defensive_clearance`) already parse the string correctly.

Fixed by parsing the string explicitly:
`_opt("play_flat_cross", "").fillna("").astype(str).str.lower().eq("true")`, matching the in-file sibling
convention. Because `str(True).lower() == "true"`, this also handles a native-bool column correctly, so the
existing bool-flag behaviour is preserved. **Hyrum:** Sportec/IDSSE (DFL/Bundesliga) pass-vs-cross labels
change for all event data — a SPADL re-conversion + downstream VAEP retrain trigger for Sportec consumers.

## [4.16.0] — 2026-06-07

### Added — TF-45 structural-pass primitives (LBS / SGM / SDI)

Per-pass structural primitives quantifying how a pass deforms the opponent's defensive structure
(Karakuş & Arkadaş 2026, arXiv:2603.28916): **Line Bypass Score** (`structural_lbs`), **Space Gain
Metric** (`structural_sgm`), **Structural Disruption Index** (`structural_sdi`). New module
`silly_kicks/tracking/_structural_pass.py`: a pure pandas-free core `_structural_pass_core`, the
per-frame `compute_structural_pass_metrics`, the `@nan_safe_enrichment add_structural_pass`
aggregator, and the `structural_pass_xfns` VAEP factory (both via the shared
`_kernels._structural_pass_at_actions` batch kernel — 3×-not-9× call-count budget). Atomic mirror
synthesizes `end = x+dx`. `StructuralPassParams.sigma = 15.0` is empirically tuned on 2,466 real
WC2022 passes (smallest σ at which the inverse-density SGM is intrinsically pitch-bounded; see
`scripts/tune_structural_pass_sigma.py`). **Library ships RAW primitives only** — the TIV z-norm
composite, K-means archetypes, and passer/receiver rankings are corpus-level (consumer-side). Decision:
ADR-005. Owner-gated e2e validates against real WC2022 Gradient Sports tracking.

### Fixed — Systemic dup-`action_id` crash across frame-aware xfns (ADR-020)

The per-slot `pointers.set_index("action_id").at[aid, "frame_id"]` pattern crashed when a `*_xfns`
factory was composed into a VAEP model: shifted gamestate slots repeat the period-boundary action, so
`action_id` is non-unique and `.at` returns a Series (`ValueError: truth value of a Series is
ambiguous`), and provenance merges fan out (`Length mismatch`). Empirically confirmed across **8
families**: `pitch_control`, `obso`, `pausa`, `space_creation`, `pressure`, `cover_shadow`,
`gk_influence`, `player_influence`. Fixed via a shared `_kernels.resolve_frame_ids_by_position`
(positional, dup-safe), a red-first behavioral gate that auto-enumerates every `*_xfns`
(`tests/tracking/test_frame_aware_xfns_dup_action_id.py`), and a per-family retrofit. **Behavior change
(Hyrum):** these `*_xfns` previously raised in the gamestate path and now produce values — a VAEP
feature-matrix change / retrain trigger for any consumer using the xfns path. (The production/lakehouse
`add_*` aggregator path on full action streams was unaffected — unique `action_id`.) Decision: ADR-020.

### Fixed — Ghost-GK public-API export gap

`silly_kicks.tracking` now exports the full ghost-GK feature surface — `add_ghost_gk`, `ghost_gk_xfns`,
`compute_ghost_gk`, `GhostGkModel`, `GhostGkDensity` — from the package root (previously reachable only
via the `silly_kicks.tracking.features` / `._ghost_gk` submodules). `add_ghost_gk` was the only feature
`add_*` aggregator missing from `tracking.__all__`; this aligns ghost-GK with every other tracking
feature (e.g. space-creation, xS, xCross) and corrects the C4 action-coupled-aggregator count.

## [4.15.0] — 2026-06-06

### Added — Dtype-safe id contract at tracking-feature seams (ADR-019)

Tracking-feature consumers compared SPADL-action identifiers against tracking-frame identifiers (and
the scalar `home_team_id` argument), and merged action↔frame frames on id-valued keys, with raw
`==`/`!=`. These silently mis-resolve when the two sides have different dtypes (`Int64(366) == "366"`
→ `False`), or **raise** on a mixed-dtype merge key — so any caller whose id dtype differs from the
library's (e.g. the lakehouse, which persists frame ids as **string** while actions stay **bigint**)
got silently-wrong actor / opponent / defending-GK / defensive-line / possession / attacking-team
resolution. ADR-019 introduces a **dtype-safe id contract** at the consumer seams:

- **New `silly_kicks.tracking._id_compat`** — one definition of "id identity": a single `_canonical`
  truth (scalar `canonical_id` + vectorized `canonical_id_series`, integral-float collapse so
  `366`/`366.0`/`Int64(366)`/`"366"` → `"366"`; genuine strings pass through), comparison helpers
  (`ids_equal`/`ids_differ`/`ids_match`/`same_id`, NA-safe, non-nullable `np.bool_`), and a pre-merge
  `align_join_keys` (numeric-vs-object only; numeric-vs-numeric and object-vs-object merge fine). A
  same-kind/both-object fast path means **zero overhead** for matched-dtype pipelines and
  genuine-string providers (sportec/kloppy).
- **New public `validate_id_dtypes(actions, frames, *, home_team_id=None, on_mismatch="raise")`** +
  `IdDtypeDiagnosis` (exported from `silly_kicks.tracking`) — an opt-in loud pre-flight guard mirroring
  ADR-017's `validate_time_base`. Not threaded through the ~30 aggregators; the seam coercion already
  makes them correct.
- The seams are fixed comprehensively (every registered `add_*` aggregator) and guarded by a red-first
  **asymmetric** dtype-invariance gate (numeric actions × string frames, and the reverse, with
  `home_team_id` an independent axis) + a meta-assertion (gate surface == registered surface) + a
  boundary-focused AST lint + a structural de-dup perf guard.

### Fixed — three latent correctness bugs the contract exposed (Hyrum: feature values change)

The contract corrects pre-existing silently-wrong behavior, so some feature values change for
**numeric (pure-library) callers too**, not only string-id callers. **VAEP models consuming these
features should be re-fit.**

- **`_resolve` opponent mask counted the ball as an opponent** for object-`is_ball` providers
  (kloppy/sportec/metrica/skillcorner): the old `~long["is_ball"]` on an **object**-dtype bool column
  is a no-op (`~True → -2`, truthy), so the ball leaked into opponents. Fixed via `.astype(bool)` +
  `ids_differ`'s both-present rule. Affects any opponent-aggregating feature; notably it inflated
  `bekkers_pi` pressure (the ball was a phantom presser). The
  `test_per_method_cross_provider_median_within_2x` calibration drops `bekkers_pi` (a kinematic model,
  not geometry-comparable across providers; its prior agreement was the ball artifact).
- **`add_player_influence` / `add_cover_shadows` team/opponent mislabel:** `str(action_team) ==
  str(frame_team)` broke because `DataFrame.iterrows()` upcasts a numpy-`int64` action `team_id` to
  `float64` (`str(5.0) != "5"`) while the nullable `Int64` frame side stays `"5"`. Fixed via `same_id`.
- **Object-path opponent join-miss:** an unmatched `how="left"` row satisfied raw `NaN != "5"` → True
  → wrongly "opponent". `ids_differ`'s both-present rule excludes it (the numeric path already did).

### Lakehouse handshake

The lakehouse may drop its string-coercion workaround and rely on the seam coercion, or call
`validate_id_dtypes(..., on_mismatch="raise")` at work-unit entry. ADR-001 (converter identifier
conventions) is preserved — the fix lives entirely at the consumer seams. No new **runtime**
dependencies; `import silly_kicks` stays dependency-light.

### Internal — deterministic perf guards + CI runtime

Library runtime is unaffected (test-infra only). The wall-clock perf budgets (`assert mean_ms <
budget`) flaked on shared CI runners (`compute_team_shape` 6.2ms > 5ms, `compute_gk_influence`
10.4ms > 10ms) — a recurring red-CI source. Every such budget is replaced with a **deterministic
structural guard** that asserts the invariant the budget actually protected, via a call-count spy on
the dominant primitive (`tests/_perf_structural.py`):

- pitch-control consumers (`compute_player_influence` / `compute_gk_influence`) build the per-frame
  surface ONCE (the ADR-008 cache contract), not per player/zone;
- the Ward line decompositions (`compute_team_shape` / `detect_line_breaking`) cluster once per
  frame, not per player/segment;
- `pressure_on_actor` (×3) / `add_actor_pre_window` link actions→frames ONCE per batch, not per
  action;
- the pitch-control kernels (Spearman / Fernandez-Bornn) run one vectorised grid pass per team, not
  per cell;
- the SPADL/atomic throughput converters stay vectorised (zero `apply(axis=1)`/`iterrows`/`itertuples`).

The benchmark *measurements* are retained (no hard timing asserts) and run single-threaded for clean
trend data. The dominant ghost-GK cost is cut at the source: the golden gates run the exact
`cpu-numba` KDE backend (matches `vectorized`/scipy at 1e-9 on the kernel; ~7.8× faster) and the
bundled-model golden slices to 4 frozen samples (`vectorized` ↔ scipy parity stays locked by the
kernel + model-traveling tests). A `pytest-xdist` parallelization was evaluated and reverted — on the
4-core/7GB CI runners it regressed py3.12 from pass to a memory/JIT-pressure kill (the opposite of the
16-core local speedup); the bulk suite stays serial.

## [4.14.0] — 2026-06-06

### Changed — Ghost-GK serves the exact boosted HGBR mean (integrity fix), pickle-free (ADR-016, PR-S83)

`compute_ghost_gk` served the KDE **mode** (~4.65 m held-out MAE) while the model card reported ~1.1 m
for the sklearn `predict_mean` that `save()`/`load()` discarded (it raised after `load()` — **never
served**). 4.14.0 closes the gap: `predict_mean` / `predict()` / `compute_ghost_gk` now serve the
**exact sklearn `HistGradientBoostingRegressor` boosted prediction** — held-out euclidean MAE **1.07 m**
(5-fold, vs the served mode's 4.65 m) — reconstructed **pickle-free** from serialized tree node arrays +
baselines (`baseline + Σ_trees leaf_value`; new `_vectorized_leaf_values` kernel, an
independent-parity-tested sibling of the KDE traversal). Inference stays sklearn-free + numpy-only +
deterministic, and is **sklearn-version-independent** (sklearn couples only at fit/extract).

An earlier attempt to serve the leaf-weighted **conditional mean** (no re-publish) was built and
**empirically rejected** — it measured 7.0 m, *worse* than the 4.65 m mode (the conditional GK-position
density is broad + multimodal, so central tendencies sit in low-density valleys). The boosted mean is a
structurally stronger estimator (squared-error boosting on the full 26-feature interaction). See ADR-016
for the rejection table + the stratified ship gate.

- **`fit()` trains `phase` numerically** (`categorical_features=None`) — removes 24 categorical split
  nodes whose routing bitsets aren't serialized, making the numeric reconstruction match sklearn exactly
  **and** closing a latent KDE categorical-routing capability gap. The density/spread shifts slightly as
  a result (expected; the served value is now the boosted mean, not the mode).
- **Artifact format change (version 1.2.0):** the npz now carries the gk_y tree ensemble + both
  baselines; `metadata.serve_estimator = "boosted_mean"`. **Both bundled `default` (wheel) and Hub
  `full` weights are re-fit + re-published.** `load()` **fails closed** on a conflicting `serve_estimator`
  (R3) and on pre-Option-A artifacts (missing gk_y trees → clear "re-fit required" error).

> **BREAKING — column rename:** the emitted spread column `ghost_gk_spread` is renamed
> **`ghost_gk_density_spread`** (in `compute_ghost_gk`, `add_ghost_gk`, `ghost_gk_xfns`, and the atomic
> mirror). The served position is now the boosted mean while the spread is the conditional-**density**
> dispersion (a different read-out — NOT the standard error of the served point); the rename makes that
> structural. **Lakehouse consumers must rename the column on consume and re-materialize `ghost_gk_*`.**

> **Hyrum's Law / behavior change:** every served `ghost_gk_x/y` value changes (deliberate value change,
> not an API break); `model.predict()` is a public-API **semantic** change (returns the boosted mean, not
> the KDE mode — the mode remains reachable via `predict_density(...).mode_x/mode_y`); old-format weights
> no longer load (re-fit required). The lakehouse must re-materialize the ghost-GK columns.

## [4.13.0] — 2026-06-04

### Added / Fixed — Gradient Sports goal-capture correctness + VAEP own-goal labeling (ADR-018)

Completes the Gradient Sports / PFF FC goal-capture work begun in 4.12.2 (which removed the false
`shot_outcome_type == "O" → owngoal` mapping). Empirically grounded in the full WC2022 catalog (64
matches, 144,541 events).

- **Own goals captured (`silly_kicks.spadl.gradientsports`).** `possession_event_type == "RE"` (rebound)
  with `shot_outcome_type == "G"` is an own goal → `bad_touch` + `owngoal`, attributed to the conceding
  team and the rebounder/scorer (`gameEvents.playerId`), per the StatsBomb/opta/sportec precedent. A
  post-LTR **geometry tripwire** validates each own goal sits in the conceding team's own half
  (`start_x < field_length/2`); a row failing it emits a `UserWarning` and reverts to `keeper_save`/`fail`
  (guards the n=3 rule against rebound-goals/feed anomalies). The 3 real WC2022 own goals
  (Enzo Fernández, Aguerd, Neuer) are captured correctly (owner-gated e2e).
- **Cross-goals captured.** `possession_event_type == "CR"` with `shot_outcome_type == "G"` keeps the
  cross/`freekick_crossed` action and **synthesizes a `shot`/`shot_freekick` + `success`** by the crosser
  (foul-synthesis pattern), so a direct cross-goal registers as a goal (SPADL records goals only as
  shots).
- **Synthesized-row provenance.** A new `is_synthetic` (bool) column on `GRADIENTSPORTS_SPADL_COLUMNS` is
  `True` on converter-injected rows (the cross-goal shot **and** the synthesized foul rows, which share
  their parent's `original_event_id`) and `False` on real 1:1 rows — so a consumer de-duping on
  `original_event_id` can keep the synthesized row instead of silently collapsing/dropping it.
- **Voided events excluded.** `possessionEvents.nonEvent == True` (annulled plays — fouls/advantage called
  back, offside, disallowed goals; 1081 across WC2022, incl. 21 disallowed goals) are now dropped in the
  exclusion stage with a `ConversionReport.excluded_counts["nonEvent"]` tally. The `nonEvent` input column
  is **optional**: absent → an observable no-op (one-time `UserWarning` + the report key omitted, so
  "not checked" ≠ "0 voided"), so existing callers keep working but get a loud nudge to supply it.
- **Own goals counted in VAEP labels (all providers) — ADR-018.** `vaep/labels.py` now detects own goals
  by **result** (`result_id == owngoal`) via a single-source `_is_owngoal` helper, dropping the
  `type_name.str.contains("shot")` gate that silently zeroed out every provider's own goals (they are all
  `bad_touch`). Goal detection uses a sibling `_is_goal` with an explicit `{shot, shot_penalty,
  shot_freekick}` name-set. A guard test forbids the old shot-gated owngoal pattern from reappearing.

> **Hyrum's Law / behavior change:** (1) VAEP `scores`/`concedes`/xG label distributions shift for every
> provider whose data contains own goals (~3–5% of goals previously invisible now count) — VAEP models
> retrained on these labels will shift. (2) Gradient Sports action counts change: voided events dropped,
> own goals now `bad_touch`+`owngoal`, cross-goals gain a synthetic shot row (flagged `is_synthetic=True`,
> sharing the cross's `original_event_id`), and the GS output gains the `is_synthetic` column. (3) The
> `nonEvent` soft input-contract: GS callers must surface `possessionEvents.nonEvent` to exclude voided
> events, else the warning fires. The atomic-SPADL surface inherits all converter changes; the atomic
> label path already counted own goals.

## [4.12.2] — 2026-06-04

### Fixed — Gradient Sports / PFF FC shot `shot_outcome_type == "O"` mis-mapped to `owngoal`

The Gradient Sports converter mapped shot `shot_outcome_type == "O"` to the SPADL `owngoal` result.
`"O"` is in fact the **off-target** shot bucket (alongside `S`=saved, `B`=blocked); the four main
shot outcomes are `G`=goal / `S`=saved / `O`=off-target / `B`=blocked, and only `G` is a success.
The mapping was an unsourced assumption inherited from the original PFF FC converter (2.6.0), never
checked against the PFF FC codebook.

Verified against the full PFF FC / Gradient Sports WC2022 feed (all 64 matches): `"G"` counts
reproduce every final scoreline **and** the exact penalty-shootout arithmetic (e.g. ARG–FRA final
3–3, pens 4–2 → G=12 = 6 regulation/ET + 6 shootout), confirming own goals already surface under
`"G"`; meanwhile MAR–ESP finished **0–0** yet carries `O=10`, and `"O"` recurs 4–17× in *every*
match — impossible for own goals.

- `silly_kicks.spadl.gradientsports`: dropped the `shot_outcome_type == "O" → owngoal` branch. `"O"`
  (and every non-`"G"` shot outcome) now falls through to `fail`, like `S`/`B`.
- The converter now maps **no** shot outcome to `owngoal`. Own goals are encoded as `"G"` and
  `shot_outcome_type` alone cannot distinguish them, so correct own-goal attribution remains an open
  item pending the PFF FC codebook.

> **Hyrum's Law / behavior change:** SPADL stores built from this converter previously contained
> phantom `owngoal` results (~563 across the 64 WC2022 GS matches, up to 17/match); these are now
> `fail`. Consumers that counted or filtered on `owngoal` from Gradient Sports data will see those
> rows reclassified — lakehouse SPADL stores should be re-baselined. The atomic-SPADL surface
> inherits the change via the shared converter.

## [4.12.1] — 2026-06-04

### Fixed — `compute_ghost_gk` crash when a team has ≥2 goalkeepers in one frame

`compute_ghost_gk` (hence `add_ghost_gk` / `ghost_gk_xfns`, and their atomic mirror) raised
`ValueError: Must have equal len keys and value when setting with an iterable` for any frame
containing two or more `is_goalkeeper=True` rows with the same `team_id`. Reported by
luxury-lakehouse: a provider match rostered a backup keeper carried on-pitch alongside the starter
in 100% of frames, so the very first batch of each half crashed and the match produced zero output.
GK-substitution overlap frames trigger the same fault intermittently.

- Root cause: `_extract_all_ghost_gk_features` emits one inference sample **per GK row**, all keyed on
  `(game_id, period_id, frame_id, gk_team_id)`. A second same-team GK in a frame produced duplicate
  keys; the downstream `how="left"` merge onto the GK rows then inflated past `gk_mask.sum()` and the
  positional assignment back into `out.loc[gk_mask, ...]` length-mismatched.
- Fix: `compute_ghost_gk` now collapses duplicate `(frame, gk_team)` inference samples (keeping the
  first) **before** `predict_density`. The features are byte-identical per `(frame, gk_team)` — only
  the per-GK-row label differs, and labels are unused at inference — so both GK rows receive the same
  ghost-GK prediction, and the KDE runs once per `(frame, gk_team)` rather than once per GK row. The
  training-data builder (`prepare_ghost_gk_training_data`) keeps its per-GK-row sampling (distinct
  labels) untouched. Single-GK frames are unaffected (the de-dup is a no-op) — the frame-restriction
  byte-identical golden still holds.

## [4.12.0] — 2026-06-04

### Added — period-relative `time_seconds` contract + loud per-period link-coverage guard (ADR-017)

Documents and enforces silly-kicks' canonical **period-relative** `time_seconds` convention
(seconds since the start of each period, resetting to 0 — NOT absolute match-clock), and makes a
low action↔frame link-coverage outcome loud. Resolves the GradientSports period-2 silent-data-loss
class reported by luxury-lakehouse (a period-relative-vs-absolute time-base mismatch dropped ~81% of
GS period-2 actions with no signal).

- `silly_kicks.tracking.utils.link_actions_to_frames` gains `min_link_rate: float = 0.5` and
  `on_low_coverage: Literal["warn", "raise", "ignore"] = "warn"`. The guard is evaluated **per
  period** (worst period), never the match aggregate — a match-aggregate floor would launder a
  catastrophically-unlinked period behind a healthy one. The warning/error message carries the
  per-period rate, unlinked count, and — when a period's action/frame ranges are near-disjoint — a
  suspected time-base-mismatch hint.
- `LinkReport` gains `per_period_link_rate: dict[int, float]`, computed from the internal per-period
  merge (not the returned pointers, which drop `period_id`).
- New public `silly_kicks.tracking.validate_time_base(actions, frames, *, on_mismatch="raise")` +
  `TimeBaseDiagnosis` — the primary guard for consumers that pre-filter / window / batch actions by
  time before linking (the linker guard cannot see actions a pre-filter already dropped). Call it on
  the **unfiltered** inputs at work-unit entry.
- `MISMATCH_OVERLAP_FLOOR = 0.2` time-base-mismatch diagnostic, decoupled from `min_link_rate` (the
  *cause hypothesis* vs the *symptom*).
- The period-relative convention is documented on the tracking + events converter docstrings,
  `link_actions_to_frames` / `slice_around_event`, and the SPADL + tracking schemas, and pinned by
  convention lock tests for the converters whose `time_seconds` arithmetic the library owns (Opta,
  StatsBomb). GradientSports `time_seconds` is a verbatim pass-through originating upstream and is
  guarded lakehouse-side.

> **Hyrum's Law / behavior change:** `link_actions_to_frames` now emits a `UserWarning` by default
> on low per-period coverage. Consumers running `-W error` / `filterwarnings=error` will start
> failing on genuinely-degraded matches — the intended shift-left. Pass `on_low_coverage="ignore"`
> for a known-partial match, or `"raise"` to escalate. The atomic-SPADL surface inherits the change
> via the shared linker.

## [4.11.0] — 2026-06-03

### Added — xCrossAttempt (xCross) cross-attempt-propensity model (TF-17, GKDV Layer 2)

A per-frame, STATE-anchored surface — `P(the in-possession team attempts a cross within ~1 s of a
frame)` — the cross analogue of xShotOccurrence (TF-16) and the next decision-probability surface in
the GKDV program. Inspired by Cao et al. (2025, arXiv:2505.11841); realizes 7 of the paper's 8
confounders (crosser position #7 omitted — no faithful tracking-only proxy) and **extends the
propensity model with a novel goalkeeper-position confounder block** (the paper's confounder set
excluded all GK variables).

- New `silly_kicks.tracking._xcross_attempt`: `extract_xcross_features`, `build_xcross_labels`,
  `prepare_xcross_training_data`, `XCrossAttemptModel` (pinned-deterministic XGBoost; pickle-free
  booster-JSON + metadata + SHA256SUMS), and the ADR-005 surfaces `compute_xcross_attempt` /
  `add_xcross_attempt` / `xcross_attempt_xfns` (+ atomic mirror).
- Shared `silly_kicks.tracking._occurrence_labels._build_occurrence_labels`, extracted from
  `build_xshot_labels` (now a thin, bit-identical wrapper — xS labels unchanged).
- HPO objective (`_xcross_attempt_objective`) + training CLI (`scripts/train_xcross_attempt.py`),
  gated behind the existing `[train]` extra; inference gates on `[xgboost]` (lazy — `import
  silly_kicks` stays dependency-light).
- **Ships UNTRAINED** (code + synthetic CI fixture + real-provider extraction tests):
  `from_variant`/`from_hub` raise `FileNotFoundError` until the weights follow-up (PR-B), and
  `xcross_attempt_xfns` is NOT wired into any default xfn list yet. The causal ATT/ATNT validation
  harness is a separate follow-up (PR-C).
- `score_differential` (confounder #1) requires match-context `actions`; `compute`/`add` accept an
  optional `actions=` kwarg (NaN-tolerant when omitted). A future `infer_ball_carrier` carrier-default
  change is an xCross retrain trigger (carrier params recorded + consumed from model metadata, R3).
- **Released on top of** the ghost-GK re-fit (PR-S81, 4.10.0); TF-17 ships as 4.11.0.

## [4.10.0] — 2026-06-03

### Fixed — Ghost-GK serve-carrier consistency (PR-S81)

`compute_ghost_gk` now computes the ball-carrier on the **full** frames and threads it into
feature extraction, so the `team_in_possession` feature matches training. Previously the serve
path passed no carrier, leaving `team_in_possession` hardcoded to `0.0` at inference while
training computed the real carrier — a latent train/serve skew (contradicting the TF-18 spec §5).

This **changes served `ghost_gk_x` / `ghost_gk_y`** on the small fraction of frames where the
defending GK's team is in possession. Measured on a real SkillCorner match (3000 GK-samples):
**0.4 % of frames change, max 4.03 m, median 0 m, mean 0.004 m** — a long-tail effect, but a
Hyrum-observable change for consumers (incl. the lakehouse). Driven by the bug fix, so it applies
to every variant, not only the re-fit.

### Changed — Ghost-GK R3 carrier-param record/consume + 4.7.0 re-fit (PR-S81)

- **R3.** `GhostGkModel` now records the ball-carrier scoring params (`tolerance_m`/`beta`/`gamma`)
  it was trained under in `metadata.json` (model `version` 1.0.0 → 1.1.0), plus
  `sklearn_version` / `training_commit` / `training_platform`, and **consumes** them at serve
  (`compute_ghost_gk` resolves possession with `model.carrier_params`, not the live library
  default). Mirrors the xShotOccurrence R3 pattern. Back-compatible: a v1.0.0 artifact without the
  field loads with the library default.
- **Bundled weights re-fit** against the 4.7.0 carrier defaults (`beta=0.0, gamma=0.25`,
  PR-S79) on 81 pining matches (887k samples, DGX Spark): `default` (wheel) + `full` (Hub). The
  re-fit is quality-equivalent to the incumbent (held-out KDE-mode MAE 4.47 m vs 4.41 m;
  `predict_mean` CV 1.12 m vs 1.14 m) and aligns the served carrier regime with the library
  default + adds R3 provenance.
- `compute_ghost_gk` / `add_ghost_gk` / `ghost_gk_xfns` gain an optional `carrier=` passthrough
  (cache convention, mirrors `links`) so pipeline callers compute the carrier once.
- `prepare_ghost_gk_training_data` gains an additive `carrier_params=` kwarg (return type
  unchanged); the shared `_build_occurrence`-style time-windowed extraction is unchanged.

### Internal

- Shared `DEFAULT_CARRIER_PARAMS` consumed by Ghost-GK (anti-drift). New maintainer scripts:
  `validate_ghost_gk_refit.py` (apples-to-apples gate), `measure_ghost_gk_serve_delta.py`,
  `_loader_pining_to_cache.py`. `train_ghost_gk.py` records the carrier params + provenance.

### Packaging

- The `full` Ghost-GK weights (~91 MB) are now **removed from the repository** — they are
  Hub-distributed (`silly-kicks/ghost-gk-v1`) and `from_variant("full")` falls back to
  `from_hub`. A `[tool.hatch.build.targets.sdist]` exclude is added alongside the existing wheel
  exclude (each hatch target has its own include/exclude set): the larger re-fit `default` had
  pushed the sdist — which still bundled `full/` — past PyPI's 100 MB per-file limit.

## [4.9.1] — 2026-06-03

### Fixed — DAS crash on a degenerate (zero-frame) frame subset

`add_das` / `das_at_action` / `get_das` / `get_individual_das` could crash with
`AttributeError: 'NoneType' object has no attribute 'x_grid'` when handed a frame subset in which
**no single frame contains both the ball and players** (after resolving `team_in_possession`).
accessible-space restricts its simulation to frames present in *both* its ball-row set and its
player-row set (`transform_into_arrays`: `frames_to_consider = ball_frames & player_frames`), but its
own emptiness guard runs *before* that intersection — so a non-empty subset whose ball and player
frames are **disjoint** collapses to a zero-frame `PLAYER_POS` (`F == 0`), `simulate_passes_chunked`
returns `None`, and `get_dangerous_accessible_space` dereferences `None.x_grid`. The resulting
`AttributeError` was not in silly-kicks' DAS-degradation `except` tuple, so it propagated as a hard
crash instead of degrading to NaN.

This surfaced in a downstream lakehouse run (Gradient Sports match 10502, one action batch whose
per-action **link-restricted** frames lost their ball or player rows) on silly-kicks 4.9.0 with
accessible-space 2.1.0. The unguarded `None` dereference exists across accessible-space 2.x.

**Fix:** a new `_has_simulatable_frame()` precondition in the silly-kicks DAS boundary detects the
disjoint-frame case *before* calling accessible-space and returns **NaN DAS** (with a `UserWarning`),
consistent with silly-kicks' existing "undefined case → NaN DAS" contract. This makes the whole
`add_das` family robust to the accessible-space fragility for *all* consumers, not just the one that
hit it. Valid frames are unaffected (the guard fires only when `ball_frames ∩ player_frames` is empty).
No public API change; no behavior change for inputs that already produced DAS.

`get_xc` (expected pass completion) shares the same accessible-space boundary and the same degenerate
collapse (`get_expected_pass_completion` runs the identical `transform_into_arrays`, simulating one
frame per pass). When no pass references a frame containing both the ball and players, that path
also reaches `F == 0` — surfacing as `AssertionError: Dimension F is 0` rather than the DAS path's
`AttributeError`, but the same root, and `get_xc` had no NaN degradation of its own. The same
precondition (shared `_frames_with_ball_and_players` helper) now guards `get_xc`, returning **NaN xC**
for the affected passes instead of crashing.

`get_xc` is now also hardened against the accessible-space × pyarrow-strings incompatibility: it used a
lighter frame prep that coerced only `player_id` to numpy object, leaving pyarrow-backed `StringDtype`
team columns in place — and accessible-space's offside path 2-D-indexes the team arrays
(`passer_teams[:, np.newaxis]`), which pyarrow strings reject with `IndexError: too many indices for
array` (the default string dtype on newer pandas / Python 3.11+, so it bit only those CI legs).
`get_xc` now uses the canonical `_prepare_frames` (which coerces `team_id` / `team_in_possession` /
`player_id`) and coerces the pass `team_id` / `player_id` too. This mirrors the DAS path's existing
coercion.

## [4.9.0] — 2026-06-02

### Added — TF-16 xShotOccurrence (xS) trained weights (GKDV Layer 2)

The xS shot-occurrence model now ships **trained** (PR-S75 shipped it untrained). A bundled
`default` variant (~1.2 MB XGBoost booster) loads via `XShotOccurrenceModel.from_variant("default")`
/ `from_hub`, and `model=None` on `compute_xshot_occurrence` / `add_xshot_occurrence` now resolves to
it. `xshot_occurrence_xfns` is wired into `pre_shot_gk_full_default_xfns` (and its atomic mirror) —
**not** the general `tracking_default_xfns`, which stays model-free (adding a frame-time bundled-weights
+ `[xgboost]` dependency to the broad default would be a Hyrum break). New `scripts/publish_xshot_occurrence.py`.

**Training (DGX Spark, 81 matches / 1,194,849 rows / ~18% positive; against the 4.7.0 carrier
defaults).** A pre-registered two-candidate comparison — `public` (skillcorner + idsse) vs `full`
(+ gradientsports), evaluated on a common public held-out set at shared hyperparameters — found that
**adding owner-tier gradientsports data degraded generalization to public-provider matches in all 5
folds** (PR-AUC Δ ≈ −0.037), so the **reproducible `public` model shipped** (CV PR-AUC 0.307 > base
rate 0.202; Brier 0.151 < base-rate 0.161). Model metadata records `shipped_variant` + `provider_list`,
`carrier_params`, `pitch_length`/`pitch_width`/`geometry_version` (TF-38 coordinate-change template),
and `xgboost_version`/`training_platform`. `pyarrow` added to the `[train]` extra (feature-cache).

### Changed

- **xS carrier defaults sourced from a single shared constant**
  `silly_kicks.tracking._ball_carrier.DEFAULT_CARRIER_PARAMS` (the 4.7.0 calibrated `tolerance_m=3.0,
  beta=0.0, gamma=0.25`) — removes the prior stale hardcoded copy and any future drift.
- **xS HPO objective** now uses `StratifiedGroupKFold` (stable per-fold positives under the ~0.02 base
  rate) and **drops `scale_pos_weight`** (xS is a calibrated `P(shot)`; the trainer gates on PR-AUC
  **and** Brier vs base rate and is fail-closed — it refuses to write a sub-bar artifact).
- `home_team_id` is now optional on the xS serve surface (it was unused — goal is resolved GK-based).
- `XShotOccurrenceModel.load` fails closed on a pitch-dimension/unit metadata mismatch (warns on a
  translation-only `geometry_version` change).
- **`prepare_xshot_training_data` no longer subsamples** — the `negative_subsample`/`seed` parameters
  are removed and it always returns the faithful class distribution (it is the train/serve-parity
  entry point; subsampling it pre-split silently contaminated downstream CV eval folds + base-rate
  baselines). Negative subsampling now lives in a standalone **`subsample_negatives(features, labels,
  groups, *, fraction, seed)`** helper with a **train-only** contract, applied by the trainer to
  **train folds only** (HPO + gate CV + paired test + final fit); held-out folds always keep the true
  balance. (Surfaced as review M3.)

## [4.8.0] — 2026-06-02

### Added — opt-in `kde_backend="fft-cic"` ghost-GK KDE backend (CIC / bilinear binning)

A fourth opt-in `kde_backend="fft-cic"` for the ghost-GK KDE, adding **CIC (cloud-in-cell / bilinear)
binning** on the existing FFT-convolution path (`predict_density` / `compute_ghost_gk` /
`add_ghost_gk` / `ghost_gk_xfns`, and the atomic mirror — flat string, no signature change). Binning
is the only seam: `_kde_density_fft` (NGP) and `_kde_density_fft_cic` share the extracted
`_kde_setup` + `_fft_convolve_field` verbatim and differ only in `_bin_ngp` vs `_bin_cic`. On near-tie
**multimodal** grids CIC reduces NGP's spurious mode flips ~76% (real data: NGP shifts the emitted GK
mode up to ~6 m on ~22% of actions → CIC ~5%) and tightens the raw grid (~5.7e-3 vs 1.5e-2 median
rel-err), at ~2× the NGP bin cost (still ~1000×+ over brute force). No new dependency (core scipy).
**Prefer `fft-cic` over `fft` for new FFT consumers** unless you need NGP's extra speed on
known-unimodal data; `vectorized`/`cpu-numba` remain the only exact-raw-grid backends. Decision:
ADR-014 (amended).

### Changed — ADR-014 mode-fidelity correction (`fft` docstring; no runtime change to `fft`)

`"fft"` (NGP) is **unchanged** and stays the fft-default — existing `"fft"` callers are unaffected.
But its documented fidelity contract is corrected: 4.6.0 claimed the emitted scalars (incl. the mode)
are "robust to per-cell binning noise"; that holds for mean/spread always and the mode on *unimodal*
grids, but **on near-tie multimodal grids NGP can flip the emitted mode by several metres** — a claim
4.6.0's *unimodal* parity bench structurally could not surface. The `_kde_density_fft` /
`predict_density` docstrings and ADR-014 are amended accordingly.

**Hyrum heads-up:** any trained-model consumer of the ghost-GK *mode* should pin one `kde_backend` for
train and serve (and persist it in metadata) — under `fft` the GK mode can differ by ~6 m on
multimodal frames. (TF-16 xShotOccurrence is unaffected — it uses the resolved/defending GK, not the
ghost-GK mode.)

## [4.7.0] — 2026-06-02

### Changed — TF-24 apply: Optuna-calibrated `infer_ball_carrier` defaults (`beta` 0.5→0.0, `gamma` 1.0→0.25)

The TF-24 calibration is applied. `infer_ball_carrier` and `ball_carrier_at_action` defaults change:
**`beta` 0.5 → 0.0** (velocity-toward-ball weighting did not help carrier-actor accuracy → selection is
now purely distance-based) and **`gamma` 1.0 → 0.25** (near-stateless hysteresis). These are Optuna-calibrated
at the held `tolerance_m=3.0` against a 3-provider fold (SkillCorner + IDSSE/DFL + Gradient Sports); the
Balanced (25-match) and Gold-max (45-match) folds **independently agreed** (`beta`≈0.0002/0.0009,
`gamma`≈0.221/0.259). Gain is modest — ~+2pp carrier accuracy at the default radius. This closes TF-24.

**`tolerance_m` is deliberately left at 3.0.** The carrier-actor-action calibration objective is
**under-determined on the radius**: its labels are on-ball moments only (no loose-ball negatives), so a wider
radius monotonically improves recall and the objective presses `tolerance_m` to the upper search bound on both
folds — a label-set artifact, not a validated optimum. Calibrating the radius would need loose-ball negatives.
(The earlier `tolerance_m≈1.0` from the pre-4.4.0 sweep was the *opposite* artifact of a since-fixed
precision-only objective; see 4.4.0.)

**Heads-up (Hyrum's Law):** `infer_ball_carrier` is called across the tracking layer (DAS, ghost-GK,
defensive line, team shape, possession), so this shifts carrier attribution for **every** tracking consumer
(including lakehouse) — calibrated and modest, but a behavior change. It is also a retrain input for the
(currently untrained) TF-16 xShotOccurrence model, which records + consumes the carrier params. **TF-25**
(provider-specific pressure-aggregation form) is **not triggered** — the cross-provider dispersion its trigger
requires did not appear (the only dispersion was the `tolerance_m` label-set artifact + carrier data-quality
differences, neither indicating provider-dependent aggregation form).

## [4.6.0] — 2026-06-02

### Added — `kde_backend="fft"` ghost-GK KDE backend (binned-convolution, ~2000× on the full-k regime)

`GhostGkModel.predict_density` / `compute_ghost_gk` / `add_ghost_gk` / `ghost_gk_xfns` accept
`kde_backend="fft"` (default stays `"vectorized"`). The three existing backends are brute-force
point×grid (O(k·m)); `fft` bins the weighted training points onto the fixed grid (nearest-grid-point)
and runs one `scipy.signal.fftconvolve` against the analytic anisotropic Gaussian — **O(k + m·log m)**,
independent of k. On the production regime (`_leaf_match_weights` returns all ~35 816 training points
on every prediction → 137 M Gaussian evals brute-force), measured **~2355×** (4247 → 1.80 ms/prediction).
Reuses the exact `_kde_setup` kernel + `cho_factor` PD-branch, so the singular-covariance uniform
fallback is unchanged. **No new dependency** (scipy is already core). Decision: ADR-014.

**Faithful on the emitted scalars, NOT on the raw grid (opt-in for this reason).** `fft` matches the
scipy oracle on the three values `predict_density` emits — `mode_x`/`mode_y` (39/40 exact, ≤1 grid
cell), `mean_x`/`mean_y` (≤5.5 mm), `spread` (≤0.16% rel) — because those are grid integrals / entropy
/ argmax-peak, robust to per-cell binning noise. It is **NOT bit-faithful on the raw
`GhostGkDensity.probabilities` grid** (NGP binning quantizes per-cell mass: ~1.5% typical, up to ~65%
on near-zero tail cells). **Hyrum's Law:** consumers that read the raw `probabilities` grid (not just
the 3 scalars) should keep `"vectorized"`; consumers that froze a golden on `ghost_gk_x/y` must
re-baseline when adopting `fft` (~2.5% of predictions flip the discrete mode by ≤1 cell — a genuine
flat-ridge near-tie). Default unchanged, so this is non-breaking.

## [4.5.0] — 2026-06-02

### Added — cacheable carrier inference (`pre` / `links` kwargs) for the calibration sweep

`infer_ball_carrier` gains an optional `pre` kwarg (a precomputed `_pre_index_frames(frames)`),
and `ball_carrier_at_action` gains optional `pre` + `links` kwargs (mirroring the `links` convention
on the `add_*` aggregators). The pre-index step (long-form frames → dense per-frame numpy arrays)
**dominates carrier-inference cost — ~99%, measured — yet is a pure function of `frames`**, fully
independent of the swept `tolerance_m`/`beta`/`gamma`; likewise the action→frame linking depends only
on the fixed link tolerance. Callers that re-resolve carriers on the *same* frames with *different*
params can now compute these once and pass them back, skipping the re-marshalling. Both default to
`None` (compute internally) and are **bit-identical** to recomputing — gated by
`tests/tracking/test_ball_carrier.py::TestCachedPreLinks` (`assert_frame_equal` / `assert_series_equal`
across the defaults, the recall-aware optimum region, and a tight radius).

### Changed — TF-24 Stage-1 carrier objective uses an invariant-prepare cache (~50–100× faster sweep)

`CarrierAccuracyObjective` previously re-ran the full per-match pre-index + linking on **every Optuna
trial**, even though both are param-invariant — making the Stage-1 sweep pandas-bound (numba accelerates
only the ~1% kernel, so it gave no real speedup). `_match_accuracy` is now split into a cached
`_prepare_match` (the param-invariant pre-index + link pointers + linked mask + actor ids, computed once
per match and reused across all trials) and a cheap `_accuracy` (kernel + lookup only). Measured **137×**
per-trial speedup on a 20k-frame synthetic match; a full gold-max Stage-1 sweep drops from ~days to
~tens of minutes. The result is bit-identical to the uncached path — `_match_accuracy` is retained as the
one-shot reference oracle, gated by `test_prepare_cached_once_and_matches_uncached` (prepare runs exactly
once per match; cached evaluate == uncached). No public objective API change; zero global state.

## [4.4.1] — 2026-06-01

### Fixed (documentation/test) — correcting the 4.2.0 DAS "value-neutral" claim

The 4.2.0 changelog stated the ball-carrier offside forwarding was *"value-neutral (zero AS/DAS
change) on real data"*. **That was wrong.** Its validating A/B test placed the carrier clearly
onside, so it never exercised the offside path. On real matches the on-ball carrier is frequently
tracked just ahead of the ball/offside line, where accessible-space (with `respect_offside`, the
default) would **delete the carrier as offside** ("treats offside players like air") unless the
passer is exempted. Forwarding `player_in_possession_col` (4.2.0+) exempts the passer, so **DAS
(`das_team`/`das_opponent`/`das_diff`) did change in 4.2.0 — a correctness fix, not a regression**:
the ball carrier is no longer mis-flagged offside. The shift is large but rare (≈1% of frames, only
where the carrier crosses the offside line; tens–hundreds of m² when it hits), because deleting a
central on-ball player materially perturbs the accessible-space tessellation.

No runtime behaviour changes in this release. Changes: (a) the misleading
`test_forwarding_is_value_neutral_and_silences_warning` is renamed/scoped to the onside-only case,
and a new `test_offside_carrier_forwarding_changes_das` locks the correct behaviour (DAS *must*
change when the carrier would be offside); (b) ADR-012 amended to record the corrected finding.

**Downstream (Hyrum's Law):** consumers who froze DAS goldens under ≤4.1.1 must re-baseline — the
≤4.1.1 values encode the pre-fix bug (carrier mis-flagged offside). The ≥4.2.0 values are correct.

## [4.4.0] — 2026-06-01

### Fixed — TF-24 Stage-1 carrier objective was precision-only (no recall term)

`CarrierAccuracyObjective` (`silly_kicks.calibration._carrier_objective`) averaged accuracy **only over
carrier-actor actions where a carrier was inferred** — `matched[valid].mean()` with `valid = inferred.notna()`.
Actions whose actor ended up beyond `tolerance_m` of the ball (→ NaN inference) were dropped from the
denominator instead of counted as misses, so there was **no recall penalty**: accuracy rose monotonically as
the candidacy radius shrank, and the optimum collapsed onto the search lower bound. The objective was
structurally blind to the very parameter it calibrates. `_match_accuracy` now uses the set of carrier-actor
actions that successfully **link** to a frame as the denominator; a linked action with a NaN inferred carrier
is a **miss**, while genuine link failures (independent of the swept params) stay excluded. This makes the
objective sensitive to `tolerance_m` (an over-tight radius is penalized through lost recall). Calibration-only
— no public runtime API change. Regression-gated by `tests/calibration/test_carrier_objective.py`
(`test_unreachable_actor_counts_as_miss` = 0.5, `test_link_failure_excluded_not_penalized` = 1.0).

**Consequence for the TF-24 apply-PR:** the completed maintainer sweep's headline `tolerance_m ≈ 1.0` (both
folds, pressed to the search lower bound) is now understood to be a **degenerate boundary artifact of the old
precision-only objective, not a validated optimum** — the two folds reproduced the same artifact, not an
independent optimum. The `infer_ball_carrier` defaults are therefore **left unchanged** (`tolerance_m=3.0`,
`beta=0.5`, `gamma=1.0`) pending a Stage-1 **re-sweep on the fixed objective**, which will produce a real
interior optimum to apply in a follow-up. Stage-2 (augmented-VAEP Brier; `k3`, `pre_seconds`,
`min_displacement_m`) is a separate held-out-Brier objective and is unaffected by this fix.

**TF-25 (provider-specific defaults) disposition:** not triggered. TF-25 fires only if `tolerance_m`/`k3`
disperse meaningfully across providers; the only Stage-1 signal so far (the boundary collapse) is an artifact,
and Stage-2 was flat. Re-evaluate after the fixed-objective re-sweep.

### Changed — kloppy `convert_to_actions` auto-derives `game_id` from dataset metadata

`silly_kicks.spadl.kloppy.convert_to_actions(dataset, game_id=None)` now falls back to the dataset's own
`metadata.game_id` (stringified to match the tracking gateway `silly_kicks.tracking.kloppy`, which uses
`str(metadata.game_id)`) when the caller omits `game_id`. Previously the column was left unset (`None`).
This is the **library-side fix for the IDSSE/Sportec join failure** that the TF-24 harness worked around at
the loader layer: SPADL actions carried `game_id=None` while the frames carried the real id, so the
`(game_id, period_id, frame_id)` joins in every tracking `add_*` enrichment missed every row. Now any
kloppy-gateway consumer (Sportec/IDSSE, plus Metrica/SkillCorner via kloppy) gets join-compatible event and
frame `game_id`s out of the box. **Heads-up (Hyrum's Law):** a caller that omitted `game_id` and relied on
the column staying `None` will now see the dataset's id; pass `game_id` explicitly to override (caller values
are always respected verbatim, ADR-001). Datasets with no `metadata.game_id` (e.g. the Metrica fixture) keep
the unset/NaN column. Gated by `tests/spadl/test_kloppy.py::TestKloppyGameIdAutoDerive`.

### Build — `numba` added to the `[calibration]` extra

The `[calibration]` extra now installs `numba>=0.59.0`. The Stage-1/Stage-2 objectives call
`infer_ball_carrier` + the pitch-control kernels once per trial; without numba those run as pure-Python loops,
the dominant cost of a full sweep. Calibration venvs now get the compiled fast path by default (the TF-24
maintainer sweep ran without it and was needlessly slow). `import silly_kicks` stays numba-free (lazy `@njit`).

## [4.3.0] — 2026-06-01

### Added — `cpu-numba` ghost-GK KDE backend (~10× the closed-form hot loop, single-thread)

`GhostGkModel.predict_density` / `compute_ghost_gk` / `add_ghost_gk` / `ghost_gk_xfns` accept
`kde_backend="cpu-numba"` (default stays `"vectorized"` = cpu-numpy). It runs a serial `@njit` fully-fused closed-form KDE loop
(no per-block temporaries), validated parity-exact (rtol 1e-9, incl. the production-scale k≈36000 case
and the near-singular zone) against the numpy kernel. The headline **~10× on the hot loop was measured
numba-serial vs numpy with all thread env vars pinned to 1** (`OMP/OPENBLAS/MKL/NUMEXPR/NUMBA_NUM_THREADS=1`)
— single-thread-vs-single-thread, the Spark-`applyInPandas` in-venue reality. The numpy setup keeps
`cho_factor` for the PD/singular branch + `log_det`, so the singular→uniform fallback boundary is
byte-identical to the numpy path. Requires the `[numba]` extra (lazily imported; `import silly_kicks`
stays numba-free). Opt-in — value-equivalent to the numpy default within golden tolerance.

### Changed — default ghost-GK KDE whitening is now closed-form (removes `cho_solve`)

**Heads-up for pinned consumers (Hyrum's Law): this shifts the DEFAULT `vectorized` backend's output, not
just the opt-in `cpu-numba` path.** `_kde_density_vectorized` now computes the 2×2 Mahalanobis energy in
closed form (`0.5/det·(h₂₂·dx² − 2·h₁₂·dx·dy + h₁₁·dy²)`) instead of `cho_solve`, sharing a new `_kde_setup`
with the numba backend. Every consumer's `ghost_gk_x`/`ghost_gk_y`/`ghost_gk_spread` move by `~1e-12..1e-9`
on a plain `4.2.0 → 4.3.0` upgrade, even without selecting a new backend. `cho_factor` is retained for the
PD-branch + `log_det` (singular→uniform boundary unchanged from 4.2.0); value-equivalent within the frozen
golden's `rtol≈1e-7` (golden NOT regenerated). The closed form alone is ~1.0× single-thread (the win is the
numba loop above); it lands as the shared foundation.

## [4.2.0] — 2026-06-01

### Changed — ghost-GK density now uses a vectorized scipy-faithful KDE (default)

`GhostGkModel.predict_density` replaces the per-sample `scipy.stats.gaussian_kde` with a
vectorized weighted-Gaussian KDE kernel that reuses scipy's exact Scott bandwidth +
weighted-covariance + Cholesky whitening (`cho_factor`/`cho_solve`), so outputs match the
scipy reference within float64 tolerance (golden-master gated: continuous grid at
`rtol≈1e-7`+atol+NaN-mask, discrete mode at exact argmax). The scipy path is retained as a
selectable reference via a new `kde_backend="scipy" | "vectorized"` argument (default
`"vectorized"`); the per-sample leaf-match is vectorized and the training set is streamed in
blocks (`train_block`, default 1024) to bound memory under the serverless 1 GB UDF cap.
Motivation: full-chain profiling identified `add_ghost_gk` as the dominant action-context
cost. Output columns (`ghost_gk_x/y/spread`) and the public API are unchanged.

### Added — DAS forwards the ball carrier to accessible-space (correct offside, no log flood)

`derive_team_in_possession` now also preserves `ball_carrier_player_id` on the returned
frames. `get_das` / `get_individual_das` accept `player_in_possession_col`
(default `"ball_carrier_player_id"`): when present it is forwarded to accessible-space so
`respect_offside` (the DAS default) excludes the passer from the offside mask. This silences
accessible-space's per-call `player_in_possession_col` warning that previously flooded logs.
A/B + unit tests confirm the forwarding is value-neutral (zero AS/DAS change) on real data;
any future change would be a documented accuracy improvement. When no carrier column is
available, silly-kicks emits its own one-time guidance instead of the per-call library
warning. An explicitly-named missing column raises `ValueError`.

### Fixed — clearer dead-ball message on link-restricted DAS subsets

When `add_das(..., links=...)` restricts to an all-dead-ball frame subset, silly-kicks now
raises its own clear "dead-ball window" `ValueError` (degraded to NaN as before) instead of
letting accessible-space's generic "empty / no non-NaN team in possession" error surface.

### Changed — elastic-sync distance lookup vectorized

`_build_player_ball_distance_lookup` builds its key/value dict vectorially instead of per-row
`.iloc` access (behaviour-preserving; golden-checked).

## [4.1.1] — 2026-06-01

### Fixed — numba on-disk cache no longer hard-fails import on read-only installs

`@njit(cache=True)` makes numba persist compiled code to disk, which requires a writable
cache *locator* to be resolved **at decoration time** (module import): a writable
`__pycache__` beside the source, a writable user-wide cache dir, or `NUMBA_CACHE_DIR` set.
On read-only / ephemeral installs — e.g. Databricks serverless, where the wheel lands on a
read-only ephemeral NFS path — all three locators fail and numba raises
`RuntimeError: cannot cache function ... no locator available` from *inside* a successful
import. Because the failing decoration runs when `silly_kicks.tracking` is imported, it took
down **all** tracking functionality (`infer_ball_carrier`, pitch control, and everything
that transitively imports them), not just the cached kernel. The existing
`try/except ImportError → _HAS_NUMBA = False` fallbacks did not catch it — the exception is a
`RuntimeError`, not an `ImportError`.

- The four `@njit` kernels (`_carrier_loop_numba` in `tracking/_ball_carrier_numba.py`;
  `tti_numba` / `influence_numba` / `gaussian_influence_numba` in
  `tracking/pitch_control/_numba_kernels.py`) now gate `cache` on a module-level
  `_NUMBA_CACHE` flag that **defaults OFF**, so import never resolves a cache locator and
  cannot hard-fail on a read-only/ephemeral filesystem. `cache=False` keeps full native JIT
  speed; it only drops cross-process cache persistence (a one-time ~1–5 s recompile per fresh
  worker process — which an ephemeral worker discards on teardown anyway).
- Opt back in to on-disk caching in stable environments (persistent cluster, local dev with a
  writable install) via `SILLY_KICKS_NUMBA_CACHE=1`, **or** by pointing numba's own
  `NUMBA_CACHE_DIR` at a writable directory (a consumer that sets it gets caching for free,
  with no second silly-kicks-specific variable to remember).
- Regression coverage: `tests/tracking/test_numba_cache_gating.py` asserts the default env
  disables the cache (the decorated dispatchers keep numba's `NullCache`) and that either
  opt-in env var re-enables it.

Caught 2026-06-01 running tracking enrichment on Databricks serverless.

## [4.1.0] — 2026-05-31

### Added — xShotOccurrence (xS) model (TF-16, GKDV Layer 2)

Per-frame shot-occurrence probability — `xS = P(a shot is attempted by the in-possession
team within ~1 second of a tracking frame)` — implementing the xS sub-model of Pipping,
Feng & Sabin (2026), arXiv:2512.00203 ("Beyond Expected Goals: A Probabilistic Framework
for Shot Occurrences in Soccer"). Distinct from xG: xS models shot *taking*, not shot
*quality*. This is GKDV Layer 2 — TF-19 will decompose `P(shot | actual_GK) −
P(shot | ghost_GK)`. The paper's xG and xG+ composition are deliberately out of scope
(silly-kicks values goals/threat via VAEP and xthreat).

- New `silly_kicks.tracking._xshot_occurrence`: the paper-faithful 27-feature extractor
  (`extract_xshot_features`; ball r/θ/z/speed, `openGoal` goal-mouth obstruction, GK
  distance/bearing, 5 nearest defenders + 5 nearest attackers) in goal-relative
  coordinates via a new shared `silly_kicks.tracking._geometry` helper; a time-windowed
  label builder (`build_xshot_labels`, robust to non-contiguous `frame_id`); the
  `XShotOccurrenceModel` (deterministic XGBoost, pickle-free booster-JSON + SHA256SUMS
  serialization); and the ADR-005 surfaces `compute_xshot_occurrence` /
  `add_xshot_occurrence` (`@nan_safe_enrichment`) / `xshot_occurrence_xfns`.
- `prepare_xshot_training_data` — the shared train/serve feature/label entry point with
  the paper's data-curation domain filter (alive-ball + attacking-third) and an optional
  seeded negative-subsample.
- HPO via the `ruthless` `CachedObjective` substrate (new `silly_kicks.tracking
  ._xshot_occurrence_objective`) + a `scripts/train_xshot_occurrence.py` CLI. New generic
  `[train]` extra (`ruthless-efficiency[optuna]` + xgboost); inference gates on the
  existing `[xgboost]` extra and keeps `import silly_kicks` dependency-light.
- `XShotFeatureSet` Literal with the `"extended"` variant reserved (raises
  `NotImplementedError` this release). Atomic mirror in `atomic.tracking.features`.
- Decision: **ADR-011** (trained-model feature lifecycle: code → training → bundled/Hub
  weights). Attribution: NOTICE entry for arXiv:2512.00203.

**Ships untrained.** This release is code + a synthetic CI fixture + real-provider
extraction tests only; no model weights are bundled (`from_variant`/`from_hub` raise until
the follow-up). The maintainer training run, bundled/Hub weights, the empirical PR-AUC
acceptance gates, and wiring `xshot_occurrence_xfns` into the default xfn lists are
deferred to a follow-up PR (it needs the gated multi-provider corpus the live TF-24 sweep
is using). Note: a future TF-24 apply-PR change to `infer_ball_carrier` defaults is an xS
retrain trigger — the carrier params used are recorded in model metadata and consumed at
inference to keep train/serve consistent until then.

## [4.0.3] — 2026-05-30

### Fixed — TF-24 calibration loader download resilience (maintainer tooling only)

The pining match loader (`scripts/_loader_pining.load_matches`) had no retry: a single transient
download/read blip (an empty/partial S3 fetch surfacing as kloppy `InputNotFoundError`, or a
`urllib`/`OSError` network hiccup) crashed the entire fold load. Across the TF-24 sweep's ~140
match-downloads (two phases × Stage 1 + Stage 2, each re-downloading its matches), a crash during
Stage-2 `prepare()` would discard hours of DAS enrichment.

- New `_build_match_with_retry` wraps each match's download+build in a 3-attempt loop with a fresh
  temp dir and linear backoff, then fails loud if the match is genuinely unfetchable.

**Consumer impact: none.** Confined to `scripts/` + `tests/`; the importable `silly_kicks` package is
byte-identical to 4.0.2 apart from the version string. Released for traceability.

## [4.0.2] — 2026-05-30

### Fixed — TF-24 IDSSE calibration provider exclusion (maintainer tooling only)

The TF-24 calibration loader silently calibrated on two providers instead of three. The Sportec
kloppy-gateway converter (`spadl_kloppy.convert_to_actions`) leaves `game_id` as `None`, while the
loader's frames carry the DFL match id from kloppy metadata. Every tracking-feature join (ball
carrier, DAS, defensive line, team shape) keys on `(game_id, period_id, frame_id)`, so the
`None`-vs-id mismatch dropped every IDSSE row → zero carrier signal → `signal_sanity` excluded IDSSE.

- `scripts/_loader_pining._build_idsse` now stamps `actions.game_id` from the frames' `game_id`
  (verified: 0 → 772/1090 valid carrier inferences on a real IDSSE match).
- `scripts/calibrate_tracking_defaults._load_fold` gains a fail-loud `game_id`-consistency guard
  (`_assert_match_game_id_consistent`) so a silent provider drop can never recur, with unit tests.

**Consumer impact: none.** Changes are confined to `scripts/` + `tests/` (the maintainer calibration
harness, not shipped in the wheel); the importable `silly_kicks` package is byte-identical to 4.0.1
apart from the version string. Released for traceability. The lakehouse stamps `game_id` from its
bronze tables and is unaffected.

## [4.0.1] — 2026-05-30

### Fixed — TF-24 calibration sweep runnable on all three providers

Two latent bugs blocked the TF-24 maintainer calibration sweep. Both lived in code
paths the calibration tests never exercised — the Stage-2 **CLI** wiring and the
**Gradient Sports** Stage-2 path (the e2e + unit fixtures cover SkillCorner only).

- **Stage-2 xT wiring in `scripts/calibrate_tracking_defaults.py`.** `main()` passed the
  `FrozenXt` *artifact* straight into `AugmentedVaepBrierObjective`, but the objective needs
  the inner `ExpectedThreat` (gk-influence / cover-shadows call `xt.interpolator(...)`). Stage 2
  via the CLI crashed at `prepare()` with `AttributeError: 'FrozenXt' object has no attribute
  'interpolator'`. The objective now accepts the `FrozenXt` and unwraps `.xt` internally, so the
  CLI passes one artifact to both the objective and the report manifest. Annotations tightened
  (`Any → FrozenXt` / `ExpectedThreat`) so the type checker rejects the mistake; the e2e + smoke
  tests now drive the same wiring the CLI uses, with a new
  `run_stage(stage=2, xt=<FrozenXt>)` regression guard.

- **`bekkers_pi` pressure crashed on duplicate frame records.** Some Gradient Sports tracking
  exports ship the same `(period, frameNum)` record up to 16× (content-divergent copies).
  `_pressure_bekkers` deduped the actor row but not the ball row, so a multi-row ball context
  built a 3-D `ball_pos` and crashed `_bekkers_tti` with a numpy broadcast error. The ball path
  now dedups keep-first (mirrors the actor path). The calibration loader also dedups the upstream
  duplicate frame records (root cause — restores the ADR-004 one-row-per-`(period, frame, player)`
  contract; otherwise pitch-control / DAS / team-shape silently compute on inflated rows too).

## [4.0.0] — 2026-05-30

### Changed (BREAKING) — symmetric fail-loud extra-time direction

Per-period-absolute converters (Sportec/IDSSE, Metrica, Gradient Sports) flip
coordinates **per period** by the home team's start direction. Extra time
(periods 3/4) requires a separate `home_team_start_left_extratime` flag. The
native converters previously handled a **missing** ET flag inconsistently — some
raised, but **Sportec tracking silently defaulted**, shipping geometrically wrong
ET coordinates with no signal. This release makes the behaviour **symmetric and
fail-loud** across all five converters. Decision: **ADR-010**.

- **`silly_kicks.tracking.sportec.convert_to_frames` now RAISES** on extra-time
  (period 3/4) without `home_team_start_left_extratime` (previously it silently
  defaulted to wrong ET geometry). **This is the breaking behaviour change.**
- **Standardized ET error message across all five converters.** Sportec tracking
  (new), Sportec events, Metrica events, Gradient Sports tracking + events all now
  raise the **same `ValueError` message shape** via the shared guard:
  `"<source>: data contains ET periods (period_id in {3, 4}) but
  home_team_start_left_extratime was not provided. ..."`. Sportec/Metrica **events**
  and Gradient Sports already raised on ET-without-flag (since 3.0.1 / earlier);
  their message **text** is now standardized — the exception **type stays
  `ValueError`** and the trigger condition is unchanged. **Consumers parsing the
  old message text must update** (Hyrum's Law: 4 messages re-worded).
- **New public guard `silly_kicks.tracking.require_et_direction(period_ids,
  home_team_start_left_extratime, *, source)`** — re-exported from
  `silly_kicks.spadl` for the events side. Lets consumers pre-flight-validate a
  batch before converting (and a CI sentinel detect a pin/metadata mismatch).
- **New public helper `silly_kicks.tracking.filter_extratime_frames(frames, *,
  label)`** — drops ET periods for **calibration/sampling only** (with a
  `UserWarning`); production must source the real ET flag, not drop ET.
- **Module rename `silly_kicks.tracking._direction` → `silly_kicks.tracking.direction`**
  (now a public module; single home for the direction helpers + the guard). The
  `home_attacks_right_per_period` function keeps its name.

### Migration

- **Pass `home_team_start_left_extratime`** to `convert_to_frames` /
  `convert_to_actions` for any match with extra time (sourced from provider
  metadata, e.g. DFL `HomeTeamStartLeftSideExtraTime` / Gradient Sports
  `homeTeamStartLeftExtraTime`). Without it, ET matches now raise.
- **Lakehouse / consumers with ET matches: upgrade to the lakehouse Phase-A PR
  first** (adds `MatchMeta.home_team_start_left_extratime` and plumbs it to
  `convert_to_frames`/`convert_to_actions`) **BEFORE** bumping the silly-kicks pin
  to 4.0.0. A pin bump without that plumbing will raise on any in-scope ET match.
- Importers of the old `tracking._direction` module path must update to
  `tracking.direction`.

### Added

- **TF-24 calibration sweep memory bounds.** `scripts/calibrate_tracking_defaults.py`
  gains `--match-ids PROVIDER:id1,id2` (repeatable), `--max-matches-per-provider`,
  and `--tracking-limit`, threaded through `_load_fold` into the loaders;
  `_loader_pining.load_matches` gains `max_per_provider`. Defaults are unchanged
  (load everything); set the flags to bound memory and run the sweep locally
  (previously the fold load hardcoded "all matches at full depth" and could OOM).

## [3.30.0] — 2026-05-30

### Changed
- **`add_das` no longer crashes on all-dead-ball batches.** When `team_in_possession` is
  all-NaN within the frames (a dead-ball window — e.g. the ball is out of play and
  `infer_ball_carrier` found no carrier), `_pin_attacking_direction` now raises the
  canonical `ValueError` that `add_das` already catches and degrades to NaN, instead of
  letting accessible-space's `infer_playing_direction` raise an **uncaught
  `AssertionError`** (which previously escaped `add_das`'s `except` and crashed the caller).
  Attacking direction is genuinely undefined without a possessing team, so DAS is NaN
  there — an honest "not applicable", not a crash. silly-kicks does **not** fabricate
  possession (the PR-S67 invariant: *"DAS is only valid when a team has possession"*);
  supply `attacking_direction_col=...` to bypass inference when the direction is known.
  Happy path (possession present) is bit-identical.
- **`pressure_on_actor(method="bekkers_pi", use_ball_carrier_max=True)` degrades
  per-action on missing ball rows instead of raising / NaN-ing.** When an action's linked
  frame has no ball position (e.g. Metrica windows where kloppy returned no
  `ball_coordinates`), that action falls back to the Bekkers **base model**
  (pressure-on-player only) — a documented variant (Bekkers 2024 §2.4), never NaN, never a
  raise. Actions whose frames *do* have a ball still use the ball-carrier-max improvement.
  Both the whole-batch `ValueError` (no ball rows anywhere) and the pre-3.30.0 per-action
  NaN are removed; genuine data-shape errors (missing `vx`/`vy`) still raise loudly. Happy
  path bit-identical (golden-master + snapshot unchanged). Atomic mirror included.
  Surfaced by the luxury-lakehouse AC-1 (`bronze.spadl_action_context`) production run on
  IDSSE dead-ball batches.

## [3.29.1] — 2026-05-30

### Changed
- **`ruthless-efficiency[optuna]` floor raised to `>=0.2.1`** in the `[calibration]` extra (and
  the dev/test deps). 0.2.1 fixes a `warm_start` off-by-one in `OptunaStrategy`: a fresh
  warm-started study ran `n_trials - 1` trials (at `n_trials=2`, only the warm-start baseline,
  with zero exploration trials). The TF-24 calibration stage configs seed a warm-start (the
  current library defaults), so the maintainer sweep must run against `>=0.2.1` for `n_trials`
  to be honored and the calibration manifest's trial count to be accurate. Calibration-tooling
  only (the `[calibration]` extra is lazy/optional, not imported by `silly_kicks/__init__`);
  no runtime library change.

### Fixed
- **Calibration manifest `silly_kicks_version`** now records `silly_kicks.__version__` (the
  source version that actually ran) instead of `importlib.metadata.version("silly-kicks")`
  (installed-dist metadata, which is stale on an editable install bumped post-install — the
  typical maintainer dev-sweep environment).

## [3.29.0] — 2026-05-29

### Added
- **`attacking_direction_col` passthrough on `add_das` / `_precompute_das_lookup`**
  (`silly_kicks.tracking.features`). When supplied, it names a column on `frames`
  holding a caller-precomputed **per-frame numeric (+1/-1)** attacking direction —
  one value per `(game_id, period_id, frame_id)`, the in-possession team's
  direction. silly-kicks validates it (exists / numeric / fully covered per group,
  restricted to the action-linked frames), **skips `_pin_attacking_direction`**,
  and threads it straight to `get_individual_das` (the 3.25.0 lower-level
  passthrough propagated up one layer). This lets callers bypass per-frame
  direction inference when the direction is already known and inference would
  assert or mis-infer — notably a dead-ball window with no non-NaN
  `team_in_possession`, where `_pin`'s `infer_playing_direction` raises an
  `AssertionError` that escaped `add_das`'s `except`. A misconfigured column fails
  loud (`ValueError`/`TypeError`, e.g. rejecting a raw string `"ltr"`/`"rtl"`
  column); it is **not** degraded to NaN. The contract is purely additive and
  carries no convention coupling: silly-kicks does not interpret
  `team_in_possession`, map string labels, or touch the library's possession gate
  (frames with NaN possession still yield NaN DAS — invariant preserved). The
  per-team→per-frame reduction and possession modeling remain the caller's
  responsibility. `attacking_direction_col=None` is bit-identical to prior
  behavior (direction inferred via `_pin`). Uncovered by the luxury-lakehouse AC-1
  production run on IDSSE dead-ball batches.

## [3.28.0] — 2026-05-29

### Added
- **TF-24 calibration harness** (`silly_kicks.calibration`, optional `[calibration]` extra):
  Optuna-TPE calibration of three tracking defaults — `infer_ball_carrier`
  (`tolerance_m`/`beta`/`gamma`), `LinkParams.k3`, and off-ball-run
  `pre_seconds`/`min_displacement_m` — against real multi-provider tracking data via
  `ruthless-efficiency[optuna]`. Pure, provider-agnostic objectives/CV/gates in the library
  (`CarrierAccuracyObjective`; `AugmentedVaepBrierObjective` as a ruthless `CachedObjective` with
  invariant-prepare + per-trial-patch and a deterministic-XGBoost cache-equivalence guarantee);
  match-stratified CV (GroupKFold-5 / leave-one-match-out); a **frozen exogenous xT artifact**
  (fit on a disjoint corpus, sha256-checksummed, fail-closed exclusion) for train–serve-consistent,
  leak-free feature extraction; H1 degenerate-feature penalty (stateless, default-Brier-anchored);
  per-provider signal-sanity + DAS-degradation surfacing; TF-25 provider-specific-defaults gate.
  Plus a `scripts/calibrate_tracking_defaults.py` CLI with pining-for-the-data + Databricks-bronze
  loaders (SkillCorner/IDSSE public, Gradient Sports owner-tier) and a data + version + xT-identity
  manifest. The harness **recommends** values + produces an auditable report; it does NOT change
  the library default constants (that is a separate "apply" PR after the maintainer's real sweep).

## [3.27.0] — 2026-05-29

### Added
- **`silly_kicks.tracking.gradientsports.add_gradientsports_player_ids`** — resolves Gradient
  Sports tracking jersey numbers to the events SPADL `player_id`/`team_id` int space via the
  roster (`(team_id, jersey_number) → roster player.id`, output `Int64`, unmatched → `pd.NA`
  never `0`), with `is_goalkeeper` from `positionGroupType == "GK"`, `team_id` from a
  caller-supplied home/away split, and a `GradientsportsRosterReport` audit. Run it before
  `convert_to_frames`. Fixes a silent failure where GS tracking carriers (jersey-derived /
  string ids) could not join GS events SPADL (`int64` player_id) — GS ball-carrier /
  DAS / team-in-possession features were silently broken. Order-safe (elementwise map, no
  row explosion); loud `UserWarning`s on a degenerate match rate, duplicate roster keys, a
  missing/zero-GK `positionGroupType`; never raises (ADR-003). Verified end-to-end on real GS
  WC2022 data (carrier accuracy 0.0 → nonzero). (TF-24 PR-A)

## [3.26.0] — 2026-05-29

### Performance
- **Ghost-GK linked-frame restriction (`add_ghost_gk`, `ghost_gk_xfns`, TF-18).**
  `compute_ghost_gk` gains an optional `link_frame_ids` kwarg that restricts both
  the heavy per-frame feature extraction and the per-sample density KDE
  (`predict_density`) to action-linked frames. The extractor still walks every
  frame to maintain the cross-period one-step velocity state and computes the
  per-period defending-goal mean-x over the full frames, so the two cross-frame
  dependencies are preserved exactly and the per-sample KDE has no cross-sample
  coupling — the output is **byte-identical** to the unrestricted compute (golden
  tests cover the goal-flip and velocity edge cases, plus a discrimination test
  proving a naive frame pre-filter would NOT be bit-identical). `add_ghost_gk`
  derives the set from its link pointers (supplied or internally computed);
  `ghost_gk_xfns` restricts to the union of its three gamestate slots. Measured
  with the bundled model: the per-250-frame batch (the lakehouse fan-out unit)
  drops from ~47.5 min to ~27 s (~100×); the dominant residual is the irreducible
  per-linked-frame KDE (~4.4 s/eval), not extraction (~4.7% of the restricted
  cost). No new columns, no API break (additive kwarg). (PR-S66)

## [3.25.1] — 2026-05-28

### Performance
- **cover_shadows `max_single_defender_blocking_score` (`detailed=False`)** is now
  computed via a single vectorized leave-one-out instead of an `O(blockers × receivers)`
  `lane_control` re-run (~4× faster on a dense 10v10 frame). The per-defender man-marking
  re-classification was hoisted out of the loop — it is provably a no-op for lane-blocker
  removals (removing a non-winner cannot change a greedy nearest-first matching; see the
  `TestManMarkerInvariantUnderLaneBlockerRemoval` property test). **Bit-identical within
  `rtol 1e-10`** (validated against an independent frozen oracle) — **no value or API change,
  and no downstream golden/model regeneration required.** The exact `detailed=True` path is
  unchanged. (PR-S65)

## [3.25.0] — 2026-05-28

### Fixed
- **ELASTIC alignment for native-frame-numbered providers** (IDSSE/Sportec):
  `align_events_to_frames` assumed `frame_id == time_seconds * frame_rate`
  (0-based), producing all-NaN alignments for providers whose `frame_id` has a
  non-zero origin (e.g. period 1 from 10000). Now derives a per-`(game_id,
  period_id)` linear `frame_id ↔ time_seconds` fit (`_fit_frame_time_relationship`)
  used for both the candidate-frame window and the `aligned_ts` / `error_seconds`
  conversion; falls back to `time * frame_rate` when frames lack `time_seconds`.
  0-based providers (Metrica/StatsBomb) are unaffected (bit-identical).

### Added
- **Shared per-frame pitch-control surface** (`PitchControlCache`, TF-7): memoizes
  canonical per-frame surfaces keyed on `(game_id, period_id, frame_id, team,
  method, params, ball_position, decompose)`, so the enrichment families that use
  pitch control compute each surface once instead of once per family. Threaded via
  an optional `pitch_control_cache` kwarg (mirrors the `links` pattern) on
  `add_obso`, `add_cover_shadows`, `add_gk_influence`, `add_player_influence`,
  `add_space_creation`, `add_pitch_control` (+ `pitch_control_at_action`). Each
  aggregator uses a fresh local cache by default (within-pass reuse); a
  caller-supplied cache extends reuse across families in one pass. Only
  canonical-frame surfaces are cached — counterfactual (player-removed) surfaces
  stay uncached. Zero global state. Output is bit-identical.
- `attacking_direction_col` passthrough on `get_individual_das` — supply a
  precomputed per-frame direction column instead of inferring it.

### Changed
- **DAS + shape_graph linked-frame restriction** (perf, bit-identical): when
  `links` is supplied, `add_das` / `add_shape_graph` restrict the expensive
  per-frame computation to the action-linked frames. For DAS, attacking direction
  is pinned on the *full* frames first (`_pin_attacking_direction`, reusing
  accessible-space's own `infer_playing_direction`) before restriction, so the
  per-period direction inference cannot flip on the restricted subset — making the
  result provably bit-identical. shape_graph is a pure per-frame snapshot, so its
  restriction is trivially identical.
- **OBSO**: hoisted the per-period `(frame_id, time_seconds)` window table out of
  the per-pass loop (was `O(passes × frames)`); pitch control now flows through the
  shared cache, reusing surfaces across overlapping pass windows. Narrowed the
  per-pass `except Exception` to `(ValueError, KeyError, IndexError)` so unexpected
  errors propagate instead of being masked as NaN (ADR-002 no-silent-swallow).
- **cover_shadows**: hoisted the receiver position / `xT` / baseline lane-control
  out of the per-blocker loop (bit-identical).

## [3.24.0] — 2026-05-28

### Added
- **Bundled Ghost-GK model weights**: `"default"` variant (~9 MB, 36 k training
  samples) ships inside the wheel — zero-config inference out of the box.
  `"full"` variant (~91 MB, 537 k training samples) lazy-downloads from
  HuggingFace Hub on first use (requires `pip install silly-kicks[ghost-gk]`).
- `GhostGkVariant` type alias (`Literal["default", "full"]`) exported from
  `silly_kicks.tracking`.
- `GhostGkModel.from_variant("full")` class method for explicit variant loading.
- `model="default" | "full"` parameter on `compute_ghost_gk` and `add_ghost_gk`
  (backward-compatible: `None` still selects the default model).

### Changed
- `_resolve_model` cascade: caller > env var > bundled variant (for `"default"`)
  or HuggingFace Hub download (for `"full"`).
- Training script round-trip verification compares serialized weights instead
  of running intractable KDE predictions.
- Training script caches extracted features to disk (`_feature_cache/`) and
  uses `predict_mean()` for permutation importance.
- SHA-256 integrity check normalizes CRLF → LF before hashing `.json` files,
  fixing cross-platform (Windows → Linux CI) hash mismatches.

## [3.23.0] — 2026-05-27

### Added
- `snapshot_to_tracking_frames` public API in `silly_kicks.tracking` — converts
  per-event player-position snapshots (e.g. StatsBomb 360 freeze-frames) into
  the 20-column `TRACKING_FRAMES_COLUMNS` schema + pre-built linkage pointers.
  Enables all single-frame `add_*` enrichment functions on freeze-frame data
  without modification. (PR-S61)
- `"snapshot"` added to `TRACKING_CATEGORICAL_DOMAINS["source_provider"]` domain
  set.

### Fixed
- **Ghost-GK goal_x period-flip**: `extract_ghost_gk_features` hardcoded
  `goal_x` by team identity, which is wrong for SkillCorner LTR-normalized
  data where teams swap ends at halftime. Now infers defending goal per
  (game_id, period_id, team_id) from mean GK x position with team-identity
  fallback. Previously dropped ~50% of SkillCorner training data via
  domain filter.

## [3.22.2] — 2026-05-27

### Fixed
- **DAS exception handling**: Widen `add_das()` / `das_at_action()` / VAEP
  transformer exception tuple from `(ValueError, RuntimeError, ImportError)` to
  also include `IndexError` and `TypeError`. Both occur in production on
  degenerate Voronoi tessellations (collinear players) and NaN tracking
  coordinates respectively. Graceful degradation to NaN columns instead of
  pipeline crash.

## [3.22.1] — 2026-05-27

### Added
- **DAS `chunk_size` passthrough**: `add_das()`, `das_at_action()`, and
  `_precompute_das_lookup()` accept optional `chunk_size: int | None` kwarg,
  threaded through to `accessible-space`. Enables memory-constrained
  environments (e.g. Databricks `applyInPandas` with 1 GB group memory cap)
  to process large matches without OOM.

### Fixed
- **Ghost-GK training script**: `pd.NA` boolean ambiguity crash when
  `ball_carrier_team_id` is `pd.NA` (`extract_ghost_gk_features` line 511).
- **Ghost-GK training script**: Glob priority swap — prefer tc3 cache layout
  (`**/frames.parquet`) over flat (`*.parquet`) to avoid stale non-tracking
  parquets in cache root.
- **CI perf budget**: Bump Andrienko pressure budget from 100ms to 120ms to
  accommodate Windows CI runner timing variance.

## [3.22.0] — 2026-05-26

### Added
- **Game state enrichment** (`add_game_state`): Derives running scoreline from
  successful shots and classifies each action as `"winning"`, `"losing"`, or
  `"drawing"` from the acting team's perspective. Pure SPADL enrichment — no
  tracking data required. `@nan_safe_enrichment` decorated; NaN `team_id` rows
  default to `"drawing"` (ADR-003). Exported from `silly_kicks.spadl`.

## [3.21.0] — 2026-05-26

### Added
- **Library extraction — 5 new tracking primitives + 1 enhancement (TF-39..TF-44, PR-S57):**
  - **TF-39 Shape Graph** (`_shape_graph.py`): Sotudeh 2026 iterative
    Delaunay edge-removal + face-center 5×5 position decomposition.
    `compute_shape_graph`, `ShapeGraph`, `add_shape_graph` aggregator,
    `shape_graph_xfns` 36-column VAEP factory.
  - **TF-40 OBSO** (`_obso.py`): Spearman 2018 Off-Ball Scoring Opportunity
    surface. `compute_obso_surface`, `ObsoSurface`/`ObsoParams` frozen
    dataclasses, `add_obso` aggregator with frame-precomputation cache,
    `obso_xfns` 9-column VAEP factory.
  - **TF-41 Space Creation** (`_space_creation.py`): Fernandez & Bornn 2018
    OBSO-weighted leave-one-out counterfactual. `compute_space_created`,
    `SpaceCreationParams`, `add_space_creation` aggregator,
    `space_creation_xfns` 9-column VAEP factory.
  - **TF-42 PAUSA** (`_pausa.py`): Lee 2026 pass utility via temporal-spatial
    OBSO decomposition. `compute_pausa`/`compute_pausa_batch`,
    `add_pausa` aggregator, `pausa_xfns` 9-column VAEP factory.
  - **TF-43 ELASTIC Sync** (`_elastic_sync.py`): Kim et al. 2025 event-tracking
    synchronization via ball acceleration + proximity scoring.
    `extract_ball_features`, `align_events_to_frames`, `ElasticSyncParams`,
    `add_elastic_sync` aggregator, `elastic_sync_xfns` 6-column VAEP factory.
  - **TF-44 Ward inter-line gaps** (`_team_shape.py` enhancement): Ward
    hierarchical clustering for defensive line identification + inter-line
    gap metrics. `n_defensive_lines` parameter; 3 new columns
    (`defensive_line_height`, `inter_line_gap_1`, `inter_line_gap_2`).
- Atomic mirror re-exports for all new VAEP xfn factories.

## [3.20.1] — 2026-05-26

### Fixed
- **Ghost-GK training script OOM prevention:** Replaced bulk `pd.concat` of all
  tracking parquets with per-file on-demand loading following lakehouse TC-3
  pattern. Raw frames are loaded one parquet at a time, features extracted
  per-game, then frames released immediately via explicit `del`. Peak memory
  drops from ~2x total frame data to one parquet file + accumulated feature
  matrix. Schema validation uses zero-data `pyarrow.parquet.read_schema`.

## [3.20.0] — 2026-05-26

### Added
- **Ghost-GK training data assembly + HuggingFace Hub publish pipeline (TF-18):**
  - `prepare_ghost_gk_training_data`: public API for extracting training
    features + labels from tracking frames with match context resolution
    (score state, set-piece phase), label domain filtering, and subsample support
  - `_build_score_lookup`: home-perspective cumulative score from SPADL goal
    actions with own-goal attribution flip
  - `_build_phase_lookup`: set-piece phase with 10s exponential decay
    (throw-in excluded per restart semantics)
  - `_extract_all_ghost_gk_features`: shared batch helper used by both
    `compute_ghost_gk` (inference) and `prepare_ghost_gk_training_data`
    (training), eliminating duplicated iteration logic
  - `compute_ghost_gk` now accepts optional `actions` parameter for
    match context enrichment (score + phase features)
  - `add_ghost_gk` now accepts optional `actions_for_context` parameter,
    threaded through to `compute_ghost_gk`
  - `scripts/train_ghost_gk.py`: full training CLI with StratifiedGroupKFold
    CV, permutation importance, metrics.json acceptance criteria, round-trip
    verification
  - `scripts/publish_ghost_gk.py`: HuggingFace Hub publish CLI with
    `--verify-only` dry-run mode and download round-trip verification

### Fixed
- **`compute_ghost_gk` timestamp key:** Fixed `"timestamp"` → `"time_seconds"`
  key in velocity state tracking, matching the tracking schema column name

## [3.19.0] — 2026-05-25

### Added
- **Ghost-GK positioning model (TF-18, GKDV Layer 2):**
  Per-frame ghost-GK density prediction using RFCDE (leaf co-occurrence
  weighted 2D KDE over HistGradientBoostingRegressor partitions).
  Predicts where a league-average GK would position given game state.
  - `GhostGkModel`: fit/predict/predict_density/save/load/from_hub
  - `GhostGkDensity`: frozen dataclass (60x64 grid, joint 2D mode)
  - `extract_ghost_gk_features`: 26-feature goal-relative extractor
  - `compute_ghost_gk`: batched per-frame primitive
  - `add_ghost_gk`: action-coupled aggregator (no provenance leak)
  - `ghost_gk_xfns`: 9-column VAEP factory (3 cols x 3 states)
  - Vectorized numpy tree traversal (no sklearn at inference)
  - Serialization: npz + metadata.json + SHA256SUMS (no pickle)
  - Lazy download from HuggingFace Hub via `[ghost-gk]` extra
  - New extras: `[ghost-gk]` (huggingface_hub), `[ghost-gk-train]` (skl2onnx)
  - Training script: `scripts/train_ghost_gk.py`
  - Atomic mirror in `silly_kicks.atomic.tracking.features`

## [3.18.2] — 2026-05-25

### Fixed
- **game_id dtype mismatch between actions (int64) and frames (str):**
  Lakehouse SPADL pipelines produce `actions.game_id` as int64 (via
  `hash_native_id_to_bigint`) while `frames.game_id` retains native
  string values. Fixed 5 unguarded merge/lookup sites across
  `_defensive_line_at_actions`, `ball_carrier_at_action`,
  `add_team_shape`, and `_team_shape_at_actions` by casting both sides
  to `str` when dtypes differ. Same pattern as the PR-S44 fix in
  `_off_ball_runs` and `_line_breaking`.

## [3.18.1] — 2026-05-24

### Fixed
- **`slice_around_event` OOM on high-framerate tracking data:** Replaced
  O(A*F) cartesian merge on `period_id` with O(A*log F) per-period
  `np.searchsorted` on sorted frame times. At 25fps (Gradient Sports
  WC2022), the old implementation produced ~1.6 billion intermediate rows
  (12+ GiB allocation) and crashed; the new implementation materializes
  only the windowed subset. Affects `add_actor_pre_window` and
  `add_off_ball_runs` callers.

## [3.18.0] — 2026-05-23

### Added
- `compute_player_influence`: per-frame primitive computing off-ball xT and uniquely reachable area for all outfield players (TF-36 + TF-33)
- `add_player_influence`: action-coupled aggregator emitting 7 columns (`actor_reachable_area_m2`, `off_ball_xt_team`, `off_ball_xt_opponent`, `off_ball_xt_diff`, `reachable_area_team`, `reachable_area_opponent`, `reachable_area_diff`)
- `player_influence_xfns`: VAEP factory (21 columns across 3 gamestate slots)
- 5 per-Series helpers: `actor_reachable_area_m2`, `off_ball_xt_team`, `off_ball_xt_opponent`, `reachable_area_team`, `reachable_area_opponent`
- `PlayerInfluence` frozen dataclass return type

## [3.17.0] — 2026-05-23

### Changed
- **`infer_ball_carrier` ~30-50x faster via numba vectorization:** Replaced
  Python `iterrows()` inner loop with dense numpy pre-indexing + numba `@njit`
  kernel. A full GS WC2022 match (~200K frames) now completes in ~112ms
  (was ~31s). Python fallback when numba unavailable (~10-20x faster than
  iterrows). Public API unchanged; output bit-identical to previous
  implementation.

### Added
- `silly_kicks/tracking/_ball_carrier_numba.py` — optional `@njit(cache=True)`
  kernel for ball-carrier inference.
- Tests: 16 new tests in `test_ball_carrier_numba_parity.py` — Python kernel
  correctness (6), pre-index round-trip (2), numba parity (5), fallback path
  (3), plus 2 e2e tests (benchmark + real-data numba-vs-numpy parity).

## [3.16.2] — 2026-05-23

### Fixed
- **`derive_velocities` crashes on single-frame player-period groups:** `np.gradient`
  requires ≥2 points; a player-period with exactly 1 frame (real-world: GS WC2022
  match 3851, away #10, period 2) triggered `ValueError`. Guard now sets vx/vy/speed
  to NaN for ≤1-frame groups.

### Added
- Tests: `test_single_frame_group_no_crash`, `test_two_frame_group_produces_finite_velocity`,
  `test_mixed_group_sizes_single_and_normal` — 3 edge-case tests for short player-period groups.

## [3.16.1] — 2026-05-22

### Fixed
- **Gradient Sports out-of-bounds coordinates not clipped to pitch:** The GS
  converter was the only provider missing coordinate clipping to SPADL pitch
  bounds [0, 105] x [0, 68]. Lakehouse WC2022 evidence: 1,108/91,931 actions
  (1.2%) had OOB values (max ~5m x, ~8m y from throw-ins, GK overruns,
  tracking noise). Added `.clip()` after LTR normalization, matching all other
  converters.

### Added
- Tests: `TestGradientsportsCoordinateClipping` — 6 tests covering high/low OOB
  start coords, end coords after derive, away-team OOB after LTR flip, in-bounds
  guard, and full synthetic fixture zero-OOB integration test.
- Synthetic fixture: 4 OOB events added (pass high-x, cross high-y, clearance
  low-x/y, away-team pass high-y) with realistic values from lakehouse evidence.

## [3.16.0] — 2026-05-21

### Fixed
- **Gradient Sports NaN `time_seconds` on dedicated FOUL events:** Real GS data
  has NULL `startGameClock` on all 28 dedicated FOUL events (gameEventType=FOUL,
  possessionEventType=FO) across 13/64 WC2022 matches. The converter now imputes
  NaN `time_seconds` via forward-fill + back-fill within each period.

### Added
- Tests: `TestGradientsportsNanTimeSeconds` — 3 tests covering ffill imputation,
  bfill fallback for period-leading NaN, and full synthetic fixture smoke test.
- Synthetic fixture: dedicated FOUL event now has `startGameClock: null` matching
  real GS data; null-actor events (OTB+CH + FOUL+FO) now generated by the script
  rather than manually appended.

## [3.15.4] — 2026-05-21

### Fixed
- **Gradient Sports null-actor `team_id` crash:** Events with null `teamId`
  (OTB+CH challenges and FOUL+FO fouls with no actor, ~17 per WC 2022 match)
  caused `IntCastingNaNError` at `gradientsports.py:420`. Fixed by applying the
  same `Int64 → fillna(0) → int64` pattern already used for `player_id`.

### Added
- Tests: `TestGradientsportsNullActorEvents` — 3 unit tests covering OTB+CH
  and FOUL+FO null-actor events plus mixed-batch conversion.
- Tests: `test_synthetic_match_null_actor_events_convert` — E2E assertion on
  the synthetic fixture with two new null-actor events (gameEventId 46, 47).

## [3.15.3] — 2026-05-18

### Fixed
- **`play_left_to_right` ball-flip bug:** Ball rows were not flipped because
  they have `team_attacking_direction = None` (set by converters). Changed from
  per-team to per-period normalization: identify periods where home team has
  "rtl" direction, then flip ALL rows (players + ball) in those periods. This
  preserves all pairwise Euclidean distances between entities.
- **Downstream `_validate_ltr` validators:** Updated validators in
  `_cover_shadows.py`, `_defensive_line.py`, and `_off_ball_runs.py` to accept
  period-normalized frames (`{"ltr", "rtl"}` after `play_left_to_right`) instead
  of rejecting any "rtl" values. Validators now reject unexpected values or
  all-rtl-only frames.

### Added
- Tests: `test_play_left_to_right_ball_flip.py` — 16 regression tests covering
  ball-player spatial consistency, per-period normalization, edge cases (NaN,
  PSO, ball-only, string team IDs), and downstream validator compatibility.
- Tests: `test_invariant_spatial_consistency.py` — 9 physical-invariant tests
  (3 scenarios × 3 invariants) verifying `play_left_to_right` preserves all
  pairwise distances, normalizes home direction to "ltr", and keeps ball
  direction as None.

## [3.15.2] — 2026-05-17

### Fixed
- **Sportec/IDSSE shot goal detection:** DFL/IDSSE events use
  `shot_outcome_type = "successful"` for goals, but the converter only matched
  `"goal"` (legacy format). Real IDSSE data: all goals had `"successful"`, zero
  had `"goal"`. Now accepts both `"goal"` and `"successful"`.
- **Metrica SHOT compound-subtype goal detection:** SG1 compound subtypes like
  `"ON TARGET-GOAL"` and `"HEAD-ON TARGET-GOAL"` were not matched by
  `sub_raw == "GOAL"`. Replaced with `endswith("GOAL")` pattern (same approach
  as PR-S43's CHALLENGE fix).
- **Ward line-breaking game_id type mismatch:** `detect_line_breaking` dict-based
  frame lookup silently returned empty results when actions carried string
  game_ids and frames carried int game_ids (or vice versa). Now aligns types
  before lookup.
- **Off-ball runs line-break game_id type mismatch:** `_line_break_kernel` had
  the same dict-based lookup vulnerability plus a merge crash on mixed
  `game_id` dtypes. Now aligns types before both the merge and the lookup.

### Added
- Tests: `test_shot_outcome_type_mapping` (6 parametrized cases) in
  `test_sportec.py` covering all real IDSSE `shot_outcome_type` values.
- Tests: `TestMetricaShotCompoundSubtypes` (10 parametrized tests) in
  `test_metrica.py` covering all real SG1 compound SHOT subtypes.
- Tests: `TestGameIdTypeMismatch` (2 tests) in `test_line_breaking.py` covering
  matching and mismatched game_id types.
- Tests: `TestLineBreakKernelGameIdTypeMismatch` (1 test) in
  `test_off_ball_runs.py` covering the off-ball-runs variant.

## [3.15.1] — 2026-05-15

### Fixed
- **`_derive_end_coordinates` NaN guard:** The source-data guard only checked
  `end_x == start_x` (placeholder pattern). When end coordinates are NaN (Metrica
  SG1 set pieces: freekick_short, corner_short, throw_in, goalkick), the guard
  silently skipped derivation because `NaN != NaN` in pandas. Now also triggers on
  `end_x.isna()`.
- **Metrica CHALLENGE compound-subtype parsing:** The old exact-match
  `sub_raw == "WON"` caught 0/233 challenges on SG1 (all real subtypes are compound
  dash-separated: "TACKLE-WON", "GROUND-WON", "AERIAL-WON", etc.). Replaced with
  `endswith("WON")` / `endswith("LOST")` + interior-token decomposition for AERIAL
  and FAULT. Tackles, keeper claims, and fouls now surface correctly on SG1 data.
- **Metrica foul extraction from CHALLENGE-FAULT-LOST:** SG1 has no
  `type == "FAULT"` events; fouls are encoded as CHALLENGE subtypes containing
  "FAULT" + ending in "LOST" (e.g., "TACKLE-FAULT-LOST"). These now map to
  `foul` (fail) with card pairing working via the existing `_apply_card_pairs`.

### Added
- Tests: `TestNaNEndCoordinates` (5 tests) in `test_derive_end_coordinates.py`.
- Tests: `TestMetricaChallengeCompoundSubtypes` (19 parametrized tests) in
  `test_metrica.py` covering compound WON, FAULT-LOST, bare LOST, bare subtypes,
  priority edge cases, GK routing, and card pairing.

## [3.15.0] — 2026-05-15

### Fixed
- **End coordinates for single-position providers (Bug #7):** DFL/Sportec and
  Gradient Sports events carry only one `(x, y)` per event, so all SPADL actions
  had `end_x == start_x`. Replaced `_fix_clearances()` with shared
  `_derive_end_coordinates()` that overwrites `end_x`/`end_y` with the next
  action's `start_x`/`start_y` for 9 pass-class action types (pass, cross,
  throw_in, freekick_crossed, freekick_short, corner_crossed, corner_short,
  clearance, goalkick). Source-data guard preserves providers that already supply
  explicit end coordinates (StatsBomb, Opta, Wyscout, Metrica, SkillCorner).
  Period-boundary safe via `groupby("period_id").shift(-1)`. Eliminates ~90% of
  spurious dribble insertions on single-position providers.
- **GK features NULL for IDSSE/Sportec (Bug #2):**
  `add_pre_shot_gk_context(frames=...)` now uses `defending_gk_from_frames()` as
  a `.fillna()` fallback when events-based lookback finds no `keeper_save` within
  the search window. Shots within tracking coverage now reliably populate
  `defending_gk_player_id` and all downstream GK position/angle features.

### Added
- `_derive_end_coordinates()` in `silly_kicks.spadl.base` — shared end-coordinate
  derivation for all 8 converters (Sportec, StatsBomb, Opta, Wyscout, Metrica,
  SkillCorner, kloppy, Gradient Sports).
- `_DERIVE_END_TYPE_IDS` frozenset — canonical set of 9 action type IDs eligible
  for end-coordinate derivation.
- Unit tests: `tests/spadl/test_derive_end_coordinates.py` (15 tests).
- Integration tests: `tests/spadl/test_end_coord_integration.py` (6 tests across
  Sportec, StatsBomb, and Gradient Sports converters).
- Integration tests: `tests/spadl/test_gk_fallback_integration.py` (2 tests using
  paired IDSSE events + tracking fixture).
- Test fixture: `tests/datasets/idsse/paired_tracking.parquet` — real paired
  tracking data for match J03WMX (2 time windows, 2 GKs, ~37 KB).

### Removed
- `_fix_clearances()` from `silly_kicks.spadl.base` — superseded by
  `_derive_end_coordinates()` which covers all pass-class types, not just
  clearances.

## [3.14.1] — 2026-05-15

### Fixed
- **DAS team symmetry bug:** `_precompute_das_lookup` used `get_das()` which
  returns a single per-frame scalar, producing identical DAS values for both
  teams and `das_diff` always zero. Switched to `get_individual_das()` with
  per-team aggregation — `das_team` and `das_opponent` now correctly differ
  between attacking and defending teams.
- **Cover shadow man-marking over-absorption:** `_classify_man_markers` used
  greedy union — any defender within `man_mark_radius` (3.0m) of *any*
  attacker's behind-point was excluded from lane analysis. In compact
  formations, overlapping exclusion zones from 10 attackers absorbed most/all
  defenders, producing `blocking_score = 0`. Replaced with greedy
  nearest-first 1:1 assignment — each defender marks at most one attacker.

### Added
- `test_precompute_das_lookup_asymmetric` — CI test asserting per-team DAS
  asymmetry with realistic 11v11 spatial setup (was previously untested).
- `test_mutual_exclusion_shared_behind_points` — CI test asserting man-marking
  mutual exclusion with overlapping attacker behind-point zones.
- `test_zero_length_pass_returns_false` — CI test documenting expected Ward
  line-breaking behavior on zero-length trajectories (IDSSE/Sportec root
  cause: single event position produces `start == end`, `pass_len = 0 <
  min_pass_length = 3.0`).

## [3.13.0] — 2026-05-13

### Added
- **Pre-linking optimization (`links` kwarg):** All tracking `add_*` aggregators now accept
  an optional `links: pd.DataFrame | None = None` keyword argument. When provided, the
  function skips its internal `link_actions_to_frames` call and uses the caller-supplied
  pointers. Pipeline callers (e.g. lakehouse) pre-link once and pass `links` to all
  enrichment steps, reducing N × 2-5s to 1 × 2-5s per match (~25-65s saved per match
  at 14 enrichment steps). Fully backwards-compatible — existing callers are unchanged.
  Functions updated: `add_action_context`, `add_pre_shot_gk_position`,
  `add_pre_shot_gk_angle`, `add_actor_pre_window`, `add_pressure_on_actor`,
  `add_defensive_line`, `add_line_break`, `add_off_ball_context`, `add_team_shape`,
  `add_pitch_control`, `pitch_control_at_action`, `add_das`, `add_gk_influence`,
  `add_cover_shadows`, `add_pre_shot_gk_context` (spadl/utils), `pressure_on_actor`.
  Internal helpers also accept `links` for full thread-through.

## [3.12.0] — 2026-05-13

### Changed (BREAKING)
- **PFF → Gradient Sports rename:** All public API symbols, module paths, and runtime
  provider identifiers renamed from `pff` to `gradientsports` to reflect the PFF FC →
  Gradient Sports corporate rebrand.
  - `silly_kicks.spadl.pff` → `silly_kicks.spadl.gradientsports`
  - `silly_kicks.tracking.pff` → `silly_kicks.tracking.gradientsports`
  - `PFF_SPADL_COLUMNS` → `GRADIENTSPORTS_SPADL_COLUMNS`
  - `PFF_TRACKING_FRAMES_COLUMNS` → `GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS`
  - `source_provider` column value: `"pff"` → `"gradientsports"`
  - `PreprocessConfig.for_provider("pff")` → `PreprocessConfig.for_provider("gradientsports")`
  - `GRADIENTSPORTS_TRACKING_DIR` env var replaces `PFF_TRACKING_DIR`
  - Example walkthrough: `pff_wc2022_walkthrough.py` → `gradientsports_wc2022_walkthrough.py`

  > **Note:** Historical CHANGELOG entries below this point retain the original "PFF"
  > terminology as they document the state of the codebase at the time of each release.
  > The rename applies from 3.12.0 onwards.

## [3.11.3] — 2026-05-12

### Fixed
- **xT NaN coordinate crash:** `ExpectedThreat.fit()` and `ExpectedThreat.rate()` no longer
  raise `IntCastingNaNError` when move actions (passes, dribbles, crosses) contain NaN
  coordinates. NaN-coordinate actions are silently dropped during transition matrix fitting
  and receive `NaN` ratings, consistent with the existing guard in `_count()`. Affects
  real-world data from Metrica (4 passes + 1148 other actions) and Sportec/IDSSE (160 fouls).

## [3.11.2] — 2026-05-11

### Fixed
- **Provenance column skip guard:** `add_action_context`, `add_pre_shot_gk_position`,
  `add_actor_pre_window`, and `add_pressure_on_actor` now skip merging linkage-provenance
  columns (`frame_id`, `time_offset_seconds`, `link_quality_score`, `n_candidate_frames`)
  when they already exist on the input DataFrame. Aligns with the idempotent pattern
  established by `add_defensive_line`, `add_team_shape`, `add_gk_influence`, and
  `add_cover_shadows` (PR-S27+). Without this guard, chaining multiple `add_*` enrichments
  produced `_x`/`_y` suffixed duplicate columns via `pd.merge`.

## [3.11.1] — 2026-05-11

### Fixed
- **Tracking namespace re-export gap:** `add_actor_pre_window`, `add_pressure_on_actor`,
  `pressure_on_actor`, and 13 related symbols (TF-2 + TF-3 per-Series helpers, xfn lists,
  pressure param types, pre-shot GK per-Series helpers) were exported from
  `silly_kicks.tracking.features` but never re-exported from `silly_kicks.tracking`.
  Oversight from PR-S25 (3.2.0). All 16 symbols now accessible at `silly_kicks.tracking.*`.

## [3.11.0] — 2026-05-11

### Added
- **TF-30: Cover Shadow Features — Lane Control + Blocking Score:**
  - `CoverShadowParams` frozen dataclass with all tunable physics constants
  - `LaneControlResult` frozen dataclass with per-line blocking probabilities + 3 decision flags
  - `ball_drag_time()` — Spearman 2017 quadratic air drag ball travel time
  - `player_tti()` — 3-phase react + accelerate + cruise time-to-intercept
  - `lane_control()` — per-(passer, receiver) corridor-discretized blocking probability
  - `compute_blocking_score()` — grid-based Voronoi counterfactual threat reduction
  - `add_cover_shadows()` — action-coupled aggregator (5 columns: `n_blocked_receivers`, `n_potential_receivers`, `blocking_score`, `blocked_threat_fraction`, `max_single_defender_blocking_score`)
  - `cover_shadow_xfns()` — VAEP factory (15 columns = 5 x 3 game states)
  - Atomic SPADL mirror
  - Ref: Cascioli, Wang, Stradiotti, Van Roy, Robberechts, Wouters, Jaspers & Davis 2025 (Hudl/DTAI, KU Leuven)

## [3.10.1] — 2026-05-10

### Fixed
- **Sportec CornerKick alias:** DFL XML uses `CornerKick` as the event tag but
  the DataFrame converter only accepted `Corner`. Callers passing raw XML tag
  names (e.g., lakehouse ingestion) had corner events silently dropped to
  `unrecognized_counts` (~16% of events in 7-match Bundesliga figshare
  collection). Both `Corner` and `CornerKick` are now accepted.
- **Sportec OtherBallAction handling:** DFL `OtherBallAction` events were
  silently dropped. Now mapped: `DefensiveClearance=true` produces a SPADL
  `clearance` action; other `OtherBallAction` events are mapped internally
  (appear in `mapped_counts`) but filtered as `non_action`.

## [3.10.0] — 2026-05-10

### Added
- **TF-15: GK influence primitives** (GKDV Layer 1):
  - `compute_gk_influence()` per-frame entry point with 3 primitives:
    threat-weighted pitch control share, uniquely reachable area, zone closing time
  - `Zone` dataclass with `six_yard_box()`, `near_post()`, `far_post()` factories
  - `GkInfluence` + `ZoneClosingTime` frozen return dataclasses
  - GK-specific kinematic parameters (`gk_reaction_time`, `gk_max_acceleration`)
  - Action-coupled: `add_gk_influence`, `gk_influence_xfns`, 4 per-Series helpers
  - Atomic SPADL mirror
  - Frame-precomputation cache in xfns factory
- **Prerequisite: `compute_tti`** exported as public API from `pitch_control`
- **Prerequisite: `select_back_line_players`** extracted from `_defensive_line.py`

### Fixed
- **TF-32 H1:** Independent dropna misalignment in `_line_breaking.py` (joint
  dropna prevents silent data corruption when opponent has valid x but NaN y)
- **TF-32 H2:** Extension-poisoning on `line_breaking_type` — `between_lines`
  now correctly dominates when both extension and through-player intersections
  occur in the same cluster
- **TF-32 M4:** Non-pass actions (shots, dribbles, etc.) now correctly produce
  pd.NA instead of being analyzed for line-breaking

## [3.9.0] — 2026-05-09

### Added
- **TF-31 Team Shape Envelope:** `compute_team_shape` per-frame primitive (7 metrics: n_outfield_players, centroid_x, centroid_y, convex_hull_area, team_length, team_width, stretch_index) + `add_team_shape` aggregator (14 action-coupled columns) + `team_shape_xfns` VAEP factory (36 columns). Ref: Clemente et al. 2013.
- **TF-32 Ward Line-Breaking:** `detect_line_breaking` per-action Ward-clustering line-breaking detection (3 columns: line_break__ward, lines_broken__ward, line_breaking_type__ward) + `LineBreakingParams` frozen dataclass + `line_breaking_ward_xfns` VAEP factory (9 columns). Extends `add_line_break` with `method="ward"` dispatch. Ref: Karakus & Arkadas 2025.

### Changed
- `add_line_break` gains `method` kwarg (`"threshold"` default, `"ward"` new) and `params` kwarg for Ward-specific parameters. Default behavior unchanged.
- `synthesize_actions` in test fixtures now gives pass actions a +20m forward trajectory offset (was zero-length).

## [3.8.0] — 2026-05-06

### Added

#### TF-28: DAS adapter — Dangerous Accessible Space

- `silly_kicks.tracking._das` module — thin adapter over `accessible-space` PyPI package (MIT)
- `get_das(frames)` → team-level AS/DAS per frame
- `get_individual_das(frames)` → per-player AS/DAS per frame
- `get_xc(passes, frames)` → expected pass completion per pass
- `derive_team_in_possession(frames, carrier)` → general tracking helper (in `_ball_carrier.py`)
- `das_at_action(actions, frames)` → action-coupled DAS
- `add_das(actions, frames)` → enrichment aggregator (`das_team`, `das_opponent`, `das_diff`)
- `das_xfns` — VAEP-compatible xfn list (single-pass precomputation, 9 columns)
- `[das]` optional extra in pyproject.toml (`accessible-space>=2.0,<3`)

#### TF-29: VAEP design-space variants — windowing + goalscore bias control

- `window` parameter on `scores()` / `concedes()`: `"action"` (default), `"possession"`, `"time"`
- `window_seconds` parameter for time-based windowing (default 15.0s)
- `xfns_default_no_goalscore` in `vaep/base.py`
- `hybrid_xfns_default_no_goalscore` in `vaep/hybrid.py`

#### Academic references (NOTICE)

- Bischofberger & Baca 2026 (Dangerous Accessible Space)
- Cascioli, Robberechts, Van Tente & Davis 2024-2025 (DTAI VAEP design-space blog series)

## [3.7.0] — 2026-05-05

### Added

#### TF-7: Pitch control models (Spearman / Fernandez-Bornn / Voronoi)

- `silly_kicks.tracking.pitch_control` subpackage — three-flavor spatial control computation
- `compute_pitch_control(frame, attacking_team_id, *, method, params, decompose, ball_position)` → `PitchControlSurface`
- `compute_pitch_control_at_points(frame, targets, attacking_team_id, *, method, params, ball_position)` → `np.ndarray`
- `PitchControlSurface` frozen dataclass with `at_point`, `at_points`, `control_in_region`, `player_share`, `player_surface`, `to_xarray` methods
- `SpearmanParams` / `FernandezBornnParams` / `VoronoiParams` frozen parameter dataclasses
- Optional numba acceleration via `_numba_kernels.py` (`@njit(cache=True)` mirrors of numpy kernels; 5-10x speedup)
- `pitch_control_at_action(actions, frames, *, method)` — action-coupled VAEP integration (NaN-safe, introspection-mode compatible)
- `add_pitch_control(actions, frames, *, method)` — enrichment aggregator
- `pitch_control_xfns(method)` / `pitch_control_default_xfns` — VAEP factory + default list
- Atomic-SPADL mirrors: `atomic.tracking.features.pitch_control_at_action`, `add_pitch_control`, `atomic_pitch_control_xfns`

#### Academic references (NOTICE)

- Spearman et al. 2017 (kinematic TTI pitch control)
- Fernandez & Bornn 2018 (bivariate-normal pitch control)
- Shaw & Sudarshan 2020 (ball-travel-time filter)

#### Architecture

- ADR-008: Pitch Control Subpackage Architecture

## [3.6.0] — 2026-05-05

### Added

#### TF-4: Off-ball runs + line-break detection

- `add_off_ball_runs(actions, frames, *, home_team_id)` — 4 off-ball-run columns: `n_off_ball_runners_pre_window`, `max_off_ball_run_displacement_pre_window`, `mean_off_ball_run_speed_pre_window`, `n_off_ball_runners_toward_goal_pre_window`
- `add_line_break(actions, frames, *, home_team_id)` — 2 line-break columns: `line_break` (nullable boolean), `n_attackers_behind_line` (Int64)
- `add_off_ball_context(actions, frames, *, home_team_id)` — umbrella aggregator adding all 6 columns
- `off_ball_context_xfns(home_team_id)` — VAEP factory (6 features x 3 states = 18 columns)

#### Academic references (NOTICE)

- Spearman 2018 (OBSO framework — off-ball-runs and line-break concepts)
- Power et al. 2017 (contextual passing risk/reward; line-breaking passes)

## [3.5.0] — 2026-05-05

### Added

#### TF-5: Per-frame ball-carrier inference

- `silly_kicks.tracking._ball_carrier.infer_ball_carrier(frames, *, tolerance_m, beta, gamma)` — per-frame ball-carrier identification via composite distance + velocity-toward-ball scoring with hysteresis. Returns one row per (game_id, period_id, frame_id) with carrier player_id, distance, and team_id. Distance-only fallback when vx/vy columns absent.
- `silly_kicks.tracking.features.ball_carrier_at_action(actions, frames, ...)` — action-coupled wrapper resolving ball carrier at each linked frame.

#### Consistency: `compute_defensive_line` game_id groupby

- `compute_defensive_line` now includes `game_id` in groupby + return schema, preventing cross-game collisions when processing multi-game batches.

#### Academic references (NOTICE)

- Bauer & Anzer 2021 (Data Mining and Knowledge Discovery) — velocity-toward-ball carrier identification heuristic.
- Vidal-Codina et al. 2022 (Sports Engineering) — hysteresis recommendation for ball-possession algorithms.

## [3.4.0] — 2026-05-05

### Added

#### TF-13: Frame-based defending-GK resolution

- `silly_kicks.tracking.features.defending_gk_from_frames(actions, frames)` — resolves defending GK `player_id` from tracking frames for all actions (not just shots). Fallback for events-based `defending_gk_player_id` NaN rows.

#### TF-14: Defensive-line geometry

- `silly_kicks.tracking._defensive_line.compute_defensive_line(frames, *, home_team_id, n=4)` — per-frame 6-column back-line geometry for both teams. Columns: `defensive_line_x`, `back_line_high_x`, `compactness_x`, `lateral_width`, `max_lateral_gap`, `back_n_count`. Supports fixed N ∈ {3, 4, 5} or `"adaptive"` via x-gap clustering (1.5× dominance rule).
- 6 per-Series action-coupled features: `defensive_line_x`, `back_line_high_x`, `compactness_x`, `lateral_width`, `max_lateral_gap`, `back_n_count`.
- `silly_kicks.tracking.features.add_defensive_line(actions, frames, *, home_team_id, n=4)` — aggregator enriching actions with 6 defensive-line columns + 4 linkage-provenance columns (skip-if-exists).
- `silly_kicks.tracking.features.defensive_line_xfns(home_team_id, *, n=4)` — VAEP xfn factory returning one multi-column transformer (6 cols × 3 states = 18 output columns).

#### NaN-safety CI

- `tests/test_enrichment_nan_safety.py` extended: auto-discovers `@nan_safe_enrichment` helpers in `silly_kicks.tracking.features` (≥6 registry floor); parametrized fuzz tests for all tracking helpers.

#### Academic references (NOTICE)

- Herold et al. 2022 (arXiv:2511.06191) — defensive-line height/compactness as match-outcome discriminators.
- Forcher et al. 2022 (arXiv:2511.00121) — back-line shape for pass-into-box models.
- FIFA EFI 2022 — practitioner 4-back defensive-line metrics.

## [3.3.0] — 2026-05-04

silly-kicks 3.3.0: Kloppy gateway `is_goalkeeper` hardening (PR-S26).

### Added

#### GK identification

- `silly_kicks.tracking._gk_identification.derive_goalkeepers` — B+ filtered algorithm for positional GK identification. Always-run design with agreement-based `is_goalkeeper_source` provenance. Handles: standard GKs (strict criteria: pa_dwell ≥ 0.40 AND dist < 20m), sweeper-keepers (rank-sum fallback), GK substitutions (multi-GK detection), brief outfielders (n_frames filter).
- `is_goalkeeper_source` column added to `TRACKING_FRAMES_COLUMNS` schema — values `"native"` (algorithm agrees with kloppy) or `"derived"` (algorithm overrode kloppy).
- `TrackingConversionReport.n_teams_gk_derived` — count of (game_id, team_id) pairs where source="derived".
- `TrackingConversionReport.derived_gk_picks` — audit trail: `dict[(game_id, team_id), list[player_id]]` of algorithm picks.

#### Kloppy gateway integration

- `silly_kicks.tracking.kloppy.convert_to_frames` now runs the GK identification algorithm on all Metrica/SkillCorner matches, fixing 21-50% → 100% GK detection rate.

#### Native path updates

- `silly_kicks.tracking.sportec.convert_to_frames` emits `is_goalkeeper_source="native"`.
- `silly_kicks.tracking.pff.convert_to_frames` emits `is_goalkeeper_source="native"`.

#### Architectural decision

- ADR-007: GK identification algorithm — documents thresholds, alternatives considered, and agreement-based source resolution design.

#### Test fixtures

- `tests/datasets/tracking/synthetic/gk_substitution.parquet` — multi-GK substitution scenario (2 teams × 2 GKs each).
- `tests/datasets/tracking/synthetic/sweeper_keeper.parquet` — sweeper-keeper fallback case (pa_dwell < 0.40).
- `tests/datasets/tracking/synthetic/brief_outfielder.parquet` — brief substitute exclusion case (n_frames filter).

## [3.2.0] — 2026-05-04

silly-kicks 3.2.0: TF-3 actor pre-window features + TF-2 multi-flavor pressure feature (PR-S25).

### Added

#### Tracking-aware features

- `silly_kicks.tracking.features.actor_arc_length_pre_window` — geometric arc-length of actor's path over the pre-action window (TF-3, default xfn). NOT Bauer & Anzer's filtered/threshold covered-distance feature; pure geometry, no sprint-intensity filtering.
- `silly_kicks.tracking.features.actor_displacement_pre_window` — net Euclidean displacement variant of TF-3 (window-first to window-last valid position).
- `silly_kicks.tracking.features.add_actor_pre_window` — aggregator emitting both columns + 4 provenance columns.
- `silly_kicks.tracking.features.actor_pre_window_default_xfns` — default xfn list (arc-length only).
- `silly_kicks.tracking.features.pressure_on_actor` — multi-flavor pressure feature (TF-2); methods: `andrienko_oval` (default; Andrienko 2017), `link_zones` (Link 2016), `bekkers_pi` (Bekkers 2024).
- `silly_kicks.tracking.features.add_pressure_on_actor` — aggregator emitting one `pressure_on_actor__<method>` per requested method.
- `silly_kicks.tracking.features.pressure_default_xfns` — default xfn list (Andrienko only, single default flavor).
- Atomic-SPADL parallel surface for all of the above (`silly_kicks.atomic.tracking.features.*`).

#### New module

- `silly_kicks.tracking.pressure` — multi-flavor pressure dispatch + per-method parameter dataclasses (`AndrienkoParams`, `LinkParams`, `BekkersParams`, `Method` Literal, `validate_params_for_method`).

#### Architectural decision

- ADR-005 §8 amendment: multi-flavor xfn column-naming convention (`<feature>__<method>` suffixes; default xfn list ships single default-method xfn; per-method params via flavor-specific frozen dataclass).

#### Attribution

- NOTICE entries: Andrienko 2017, Link 2016, Bekkers 2024 + BSD-3-Clause attribution to UnravelSports for the Bekkers TTI port.
- Vendored 30-line BSD-3-Clause excerpt at `tests/_vendored/unravelsports_tti.py` (test-only) so the Bekkers golden-master parity test runs unconditionally on Python 3.10+ without requiring the live `unravelsports` package (which targets Python 3.11+).

#### Test-only optional dependencies

- `unravelsports>=1.2` (extra `golden-master`) — preferred canonical source for the Bekkers golden-master parity test on Python 3.11+; the test falls back to `tests/_vendored/unravelsports_tti.py` when the live package isn't installed (e.g., Python 3.10).

#### Test infrastructure

- `tests/datasets/metrica/sample_match.parquet` regenerated with the 0–1 → 0–105/0–68 SPADL-frame rescale (matches `per_period_match.parquet` and the lakehouse `adapt_metrica_events_for_silly_kicks` adapter); previous fixture leaked Metrica's normalized 0–1 frame into bronze rows. `scripts/extract_provider_fixtures.py --provider metrica` now applies the rescale at extract time.
- Invariant tests (`tests/invariants/test_direction_of_play.py`, `test_gk_position.py`, `test_vaep_geometric_sanity.py`) hardened: `pytest.skip` paths replaced with explicit assertions or parametrize-list exclusions; shot counts now span all SPADL shot variants (`shot` / `shot_penalty` / `shot_freekick`) so converters' set-piece-composition rules don't mask the invariant; GK position invariant now also covers `keeper_pick_up`. Skipping count on the invariant suite went from 11 to 0.

## [3.1.0] — 2026-05-02

### Added

- **TF-6 — `sync_score`** (`silly_kicks.tracking.utils.sync_score`,
  `add_sync_score`, `LinkReport.sync_scores()`): per-action tracking↔events
  sync-quality scores. New columns when used via `add_sync_score`:
  - `sync_score_min`
  - `sync_score_mean`
  - `sync_score_high_quality_frac`
- **TF-8 — smoothing primitives** (`silly_kicks.tracking.preprocess.smooth_frames`,
  `derive_velocities`): Savitzky-Golay (canonical) and EMA smoothing of
  positional columns. Schema-additive output columns:
  - `x_smoothed`, `y_smoothed`
  - `vx`, `vy`, `speed`
  - `_preprocessed_with` (per-row provenance tag — load-bearing because
    `pandas.DataFrame.attrs` does not propagate through merge/concat/applyInPandas)
- **TF-9 — interpolation / gap-filling** (`silly_kicks.tracking.preprocess.interpolate_frames`):
  linear NaN gap-filling up to `max_gap_seconds` (cubic deferred to TF-9-cubic).
  Same schema as input — no new columns, just NaN cells replaced where the
  gap is short enough.
- **TF-12 — `pre_shot_gk_angle_*`** (`silly_kicks.tracking.features.add_pre_shot_gk_angle`,
  `pre_shot_gk_angle_to_shot_trajectory`, `pre_shot_gk_angle_off_goal_line`,
  `pre_shot_gk_angle_default_xfns`, `pre_shot_gk_full_default_xfns` + atomic
  mirror). New columns:
  - `pre_shot_gk_angle_to_shot_trajectory` (float64, radians, signed)
  - `pre_shot_gk_angle_off_goal_line` (float64, radians, signed)
- **`PreprocessConfig`** (`silly_kicks.tracking.preprocess.PreprocessConfig`):
  shared preprocessing config dataclass with `default()` / `for_provider(name)`
  factories and flag-based `is_default()`. Construction-time validator rejects
  `derive_velocity=True` + `smoothing_method=None`.
- **Tracking-converter optional `preprocess` kwarg** on
  `silly_kicks.tracking.sportec.convert_to_frames`,
  `tracking.pff.convert_to_frames`, and `tracking.kloppy.convert_to_frames`.
  Default `None` ⇒ zero behavior change. When set, applies interpolation /
  smoothing / velocity-derivation per the config; auto-promotes
  `PreprocessConfig.default()` to `PreprocessConfig.for_provider(<this_provider>)`,
  with `force_universal=True` + `UserWarning` fallback for unsupported providers.
- **Umbrella facade extension**: `silly_kicks.spadl.utils.add_pre_shot_gk_context`
  (and atomic mirror) now emits 6 GK-tracking columns when called with
  `frames=...` (the existing 4 from PR-S21 plus the 2 new TF-12 angles).
  The `frames=None` path is bit-identical to silly-kicks 2.9.0 — 4 columns.
  Lakehouse boundary tests asserting on the `frames=...` column-set need
  `expected_columns` extended by `pre_shot_gk_angle_to_shot_trajectory` and
  `pre_shot_gk_angle_off_goal_line`.
- **Empirical baselines**: `tests/fixtures/baselines/preprocess_baseline.json`
  + `preprocess_sweep_log.json` (per-provider stats across all 4 supported
  tracking providers including SkillCorner) +
  `scripts/probe_preprocess_baseline.py` +
  `scripts/regenerate_provider_defaults.py` (codegen pipeline replaces
  manual sync hand-edit).

### Changed

- **scipy is now a hard runtime dependency** (`scipy>=1.10.0`) — required by
  `tracking.preprocess` for Savitzky-Golay smoothing + derivative. Previously
  optional for `silly_kicks.xthreat` only.

### Notes

- ADR-005 amendment formalising the multi-flavor convention asymmetry
  (suffixed columns for VAEP xfns; canonical-single columns for preprocessing
  utilities) lands alongside the TF-2 `pressure_on_actor` PR (scheduled
  within 24-48 hours of PR-S24 merge — bounded deferral).
- Lakehouse pin bump: `silly-kicks>=3.1.0,<4`. No 3.0.x → 3.1.0 migration
  needed beyond the boundary-test column-set update above and (when adopting
  preprocessing inside Spark UDFs) declaring `_preprocessed_with` +
  smoothed/velocity fields explicitly in the `applyInPandas` `StructType`
  schema.

## [3.0.1] — 2026-05-02

### Breaking-correctness fix (PR-S23) — Sportec + Metrica per-period direction-of-play

`silly_kicks.spadl.sportec.convert_to_actions` and
`silly_kicks.spadl.metrica.convert_to_actions` now correctly handle
per-period-absolute bronze events (teams switching ends after halftime).
silly-kicks 3.0.0 declared these converters as `ABSOLUTE_FRAME_HOME_RIGHT`,
producing wrong-end SPADL output for half of every match. ADR-006 erratum
documents the corrected per-converter declaration table.

Callers must now pass per-period direction info via one of two paths
(otherwise `ValueError` with migration guidance):

```python
# Path A -- bool pair (preferred; matches PFF events + tracking-Sportec API)
actions, report = sportec.convert_to_actions(
    events,
    home_team_id="DFL-CLU-XXXXX",
    home_team_start_left=True,                     # from DFL MatchInformation.xml
    home_team_start_left_extratime=False,          # only when ET periods present
)

# Path B -- explicit mapping (escape hatch for arbitrary periods)
actions, report = metrica.convert_to_actions(
    events,
    home_team_id="Home",
    home_attacks_right_per_period={1: True, 2: False},
)
```

Trained VAEP / HybridVAEP / xT models on Sportec or Metrica data from
silly-kicks 3.0.0 must be re-trained on 3.0.1 output.

### Test infrastructure

- New per-period orientation fixtures committed at
  `tests/datasets/idsse/per_period_match.parquet` (Bassek et al. CC-BY 4.0)
  and `tests/datasets/metrica/per_period_match.parquet` (CC-BY-NC-4.0;
  same precedent as existing Metrica Sample Game 2 fixture). Both are
  excluded from the published wheel.
- New `test_per_team_per_period_shots_attack_high_x` parametrized over
  both new fixtures in `tests/invariants/test_direction_of_play.py`.
  Closes the invariant-density gap that let PR-S22's bug ship.
- 5 new `TestSportecPerPeriodKwargContract` + 5 new
  `TestMetricaPerPeriodKwargContract` negative-path tests for kwarg
  resolution policy.

### Detector hardening (TF-22)

`silly_kicks.spadl.orientation.detect_input_convention` no longer
false-positives `ABSOLUTE_FRAME_HOME_RIGHT` on sparse-shot
per-period-absolute matches. New guard: when no team has reliable shots
in ≥ 2 distinct periods, returns `convention=None, confidence="low"`.
Validator re-enabled at sportec / metrica / pff converter call sites
declaring `PER_PERIOD_ABSOLUTE`.

### Atomic-SPADL pathway

Smoke test added at `tests/atomic/test_atomic_orientation.py` verifying
the SPADL → atomic-SPADL composition preserves canonical-LTR. No
converter changes (atomic has no native sportec/metrica converter).

### Other

- `silly_kicks/__init__.py` `__version__` bumped from "1.0.2" (stale
  since at least 2.0.0) to "3.0.1" so it now matches `pyproject.toml`.
- `scripts/extract_provider_fixtures.py` gains `--variant {default, per_period}`
  flag for regenerating either fixture variant. Per-period extraction
  pulls from `bronze.idsse_events` / `bronze.metrica_events` on Databricks
  (env-var auth).
- `NOTICE` "Test Data Sources" section attributes the new IDSSE +
  Metrica Sample Game 1 fixtures.

## [3.0.0] — 2026-05-02

### Breaking — Correctness (PR-S22)

**Direction-of-play handling refactor.** The dual-mirror inversion that has
been present since v0.1.0 is fixed. SPADL canonical convention is "all teams
attack left-to-right" -- every team's actions at high-x in their own frame.
Every silly-kicks SPADL converter now produces this convention directly via
the new :func:`silly_kicks.spadl.to_spadl_ltr` dispatcher. Decision: ADR-006.

**Code-side regression window.** The bug was present in the native StatsBomb,
Wyscout, and Opta converters AND in `vaep.base.VAEP.compute_features` since
the v0.1.0 fork (verified `git show 0b29178`). The kloppy gateway acquired
the same code path in 1.7.0 but routed correctly because kloppy's transform
already normalised to absolute-frame-home-right.

**Consumer-artifact impact depends on which converter path each artifact's
data went through.** Categorically affected:

- Cached SPADL action tables derived from native ``silly_kicks.spadl.statsbomb``
  / ``wyscout`` / ``opta`` -- away-team ``(x, y)`` were mirrored to the wrong
  end of the pitch.
- Trained VAEP / HybridVAEP models built on Sportec / Metrica / kloppy-gateway
  / PFF SPADL -- VAEP feature engineering (now correctly free of the second
  mirror) inverted away-team rows in gamestates.
- Trained xG / xT models that consume polar / spatial features.
- Pre-computed xT grids derived from broken SPADL inputs (U-shaped instead of
  goal-monotonic).
- Tracking-aware features: ``add_action_context`` (PR-S20),
  ``add_pre_shot_gk_context`` (PR-S21).
- Any downstream model trained on action-coord features.
- Any test baseline / golden value calibrated on the prior pipeline.
- Any dataset published from silly-kicks output that mirrors SPADL or VAEP.

Per-consumer migration is the consumer's responsibility; this CHANGELOG enumerates
the categorical impact rather than specific consumer artifacts.

### Added

- **`silly_kicks.spadl.orientation`** (NEW module) — canonical direction-of-play
  primitives:
  - ``InputConvention`` enum: ``POSSESSION_PERSPECTIVE`` (StatsBomb, Wyscout),
    ``ABSOLUTE_FRAME_HOME_RIGHT`` (Sportec, Metrica, Opta, kloppy gateway),
    ``PER_PERIOD_ABSOLUTE`` (PFF).
  - ``to_spadl_ltr(actions, *, input_convention, home_team_id, ...)`` —
    single canonical normalizer; each converter calls it exactly once.
  - ``detect_input_convention(events, *, match_col, x_max, ...)`` — heuristic
    detector; tiered confidence (≥10 shots/group = high, 5-9 = medium, <5 =
    ambiguous defer).
  - ``validate_input_convention(events, declared, *, on_mismatch)`` — wired
    into every converter; warn by default, raise under
    ``SILLY_KICKS_ASSERT_INVARIANTS=1``. Surfaces upstream loader regressions.
- **`silly_kicks.vaep.base.VAEP.compute_features(..., frames_convention="absolute_frame")`**
  — explicit kwarg controlling tracking-frame normalisation.
- **`silly_kicks.tracking.{sportec,pff,kloppy}.convert_to_frames(..., output_convention=…)`**
  — opt-in ``"ltr"`` mode for callers wanting SPADL LTR tracking output
  directly. Default behaviour preserved (absolute_frame); ``None`` (legacy
  unspecified) emits ``DeprecationWarning`` recommending callers be explicit.
- **`tests/invariants/`** (NEW directory) — physical-invariant test layer
  parametrised across providers with real fixtures:
  - ``test_direction_of_play.py`` — per-team shots cluster at high-x,
    parametrised × ``xy_fidelity_version ∈ {1, 2}`` for StatsBomb.
  - ``test_vaep_geometric_sanity.py`` — VAEP shot dist < 50m AND xT
    goal-monotonic.
  - ``test_gk_position.py`` — GK actions cluster at defended (low-x) goal.
  - ``test_input_convention_detector.py`` — detector + validator semantics
    against real fixtures.

### Changed

- **`silly_kicks.spadl.statsbomb`, `wyscout`, `opta`, `sportec`, `metrica`,
  `kloppy`, `pff`** — every ``convert_to_actions`` now routes the
  direction-of-play step through ``to_spadl_ltr(input_convention=…)`` and
  emits canonical SPADL LTR. The ``input_convention`` declared by each
  converter is the load-bearing contract; ``validate_input_convention``
  surfaces violations.
- **`silly_kicks.spadl.opta.convert_to_actions`** — docstring contract
  added: the converter expects loader-pre-normalised absolute-frame data
  with NO per-period switching. Raw Opta f24 ships per-period switching;
  callers must pre-normalise upstream.
- **`silly_kicks.vaep.base.VAEP.compute_features`** — removed the inline
  ``play_left_to_right`` call (the dual-mirror that this CHANGELOG fixes).
  Converter output is already canonical SPADL LTR.
- **`silly_kicks.spadl.utils._finalize_output`** — debug-mode invariant
  assertion gated on ``SILLY_KICKS_ASSERT_INVARIANTS=1``: per-team shot mean
  start_x must be > field_length/2.
- **`silly_kicks.spadl.play_left_to_right`** + atomic-SPADL,
  ``silly_kicks.vaep.features.play_left_to_right`` + atomic-VAEP equivalents
  — docstrings updated. Functions are retained as public boundary helpers
  (absolute-frame → SPADL LTR) but no longer called by silly-kicks itself.

### Removed

- **`silly_kicks.spadl.base._fix_direction_of_play`** (private symbol) —
  replaced by ``silly_kicks.spadl.to_spadl_ltr``. Was only ever called by
  the converters themselves; no public API impact.

### Migration

Re-derive any cached artifact whose path went through an affected converter.
Specifically: re-derive SPADL action tables from raw events; re-train VAEP /
HybridVAEP models; re-compute xT grids; re-baseline empirical golden values;
re-publish any silly-kicks-derived datasets. The new validator surfaces input
convention mismatches as warnings; set ``SILLY_KICKS_ASSERT_INVARIANTS=1`` in
CI to promote them to failures.

## [2.9.0] — 2026-05-01

### Added — Pre-shot GK position + baselines backfill (PR-S21)

- **`silly_kicks.tracking.features`** — 4 GK-position helpers: `pre_shot_gk_x`,
  `pre_shot_gk_y`, `pre_shot_gk_distance_to_goal`, `pre_shot_gk_distance_to_shot`.
  Plus aggregator `add_pre_shot_gk_position(actions, frames) -> pd.DataFrame`
  that emits the 4 GK columns + 4 linkage-provenance columns. Decorated with
  `@nan_safe_enrichment` per ADR-003. Plus `pre_shot_gk_default_xfns` (4
  `lift_to_states` wrappers) for HybridVAEP integration.
- **`silly_kicks.atomic.tracking.features`** — atomic-SPADL parity with the
  same public surface (`atomic_pre_shot_gk_default_xfns`). Mirrors the standard
  surface with atomic-shaped column reads (`x, y`) and atomic shot type ids
  (`{shot, shot_penalty}` — atomic does not recognize `shot_freekick`).
- **`silly_kicks.spadl.utils.add_pre_shot_gk_context(*, frames=None)`** — additive
  optional `frames` kwarg. When supplied, emits 4 GK-position columns + 4
  provenance columns by lazy-importing the canonical compute (preserves
  ADR-005 §5 no-cycle invariant). When `frames=None` (default), behavior is
  bit-identical to silly-kicks 2.8.0 — no frames-related columns appear.
  Backward-compat pinned by golden-fixture test.
- **`silly_kicks.atomic.spadl.utils.add_pre_shot_gk_context`** — atomic mirror
  of the same `frames=None` extension.
- **`silly_kicks.tracking._kernels._pre_shot_gk_position`** (private) —
  schema-agnostic compute kernel shared between standard and atomic surfaces.
- **`silly_kicks.tracking.feature_framework.ActionFrameContext`** gains
  `defending_gk_rows: pd.DataFrame` field (default-factory empty DataFrame —
  preserves direct construction backward-compat).
- **`scripts/regenerate_action_context_baselines.py`** — one-shot regenerator
  for `*_expected.parquet` files + `empirical_action_context_baselines.json`.
- **`tests/datasets/tracking/action_context_slim/{provider}_expected.parquet`**
  — per-provider expected output committed for the bit-exact per-row
  regression gate (4 providers).
- **`tests/tracking/_provider_inputs.py`** — shared loader/synthesizer for the
  regenerator and CI gate; keeps both in sync.
- **`tests/tracking/test_action_context_expected_output.py`** — bit-exact
  per-row regression gate (4 providers).
- **`tests/tracking/test_empirical_action_context_baselines.py`** — JSON shape
  gate + JSON-vs-parquet consistency gate.

### Changed

- **`silly_kicks.spadl.utils.add_pre_shot_gk_context`** + atomic mirror —
  bug-fix: `defending_gk_player_id` output column now preserves the input
  `player_id` dtype. Numeric `player_id` (canonical SPADL_COLUMNS:
  PFF / StatsBomb / Opta / Wyscout / Metrica) → `float64` NaN-coded (unchanged).
  Object/string `player_id` (`KLOPPY_SPADL_COLUMNS` / `SPORTEC_SPADL_COLUMNS`
  schema) → `object` dtype with `None` for unidentified rows. Previous
  unconditional `int(gk_id_raw)` cast crashed on string Sportec player_ids;
  surfaced by PR-S21's TF-11 regression-gate exercising real-shot rows on
  Sportec data.
- **`tests/datasets/tracking/empirical_action_context_baselines.json`** —
  all 256 percentile slots backfilled (4 percentiles × 8 features × 4 providers).
  Per-row gate exercises real GK-position computation on at least one shot
  per provider (synthesizer in `tests/tracking/_provider_inputs.py` stamps a
  synthetic keeper_save → shot pair anchored on real frame goalkeeper data
  so the events-side helper populates `defending_gk_player_id` and the
  tracking aggregator emits non-NaN GK position).
- **`NOTICE`** — Anzer & Bauer (2021) entry description expanded to enumerate
  defending-GK-position alongside player_speed and distance-to-defender.
- **`TODO.md`** — TF-1 + TF-11 marked SHIPPED. PR-S21 active-cycle entry.
  Bundled National Park additions: TF-12 (`pre_shot_gk_angle_*`), TF-13
  (frame-based GK identification fallback), TF-14 (defensive-line features).

### Removed

- **4 vestigial `test_placeholder` stubs** (National Park cleanup): the
  `TestKloppyE2E.test_placeholder` (`test_kloppy.py`),
  `TestSpadlConvertorE2E.test_placeholder` (`test_opta.py`, `test_wyscout.py`),
  and `TestSpadlConvertor.test_placeholder` (`test_statsbomb.py`) classes
  were inert `pytest.skip()` calls inherited from the v0.1.0 socceraction
  fork (the original DataLoader classes — `OptaLoader` / `StatsBombLoader` /
  `PublicWyscoutLoader` / `KloppyLoader` — were removed at fork time but the
  e2e test scaffolds were left behind as no-op skip stubs). Plus the
  unreferenced `pytestmark_e2e` module attribute in `test_opta.py`. Net
  effect: `pytest -m e2e` now runs 12 PASSED / 0 SKIPPED instead of
  12 PASSED / 4 SKIPPED — the SKIPPED column is no longer a hiding place
  for genuine missing-fixture failures.

### Notes

- No breaking changes. PR-S21 ships entirely within ADR-005's locked
  architecture; no new ADR.
- Per-Series GK helpers (`pre_shot_gk_x` etc.) silently emit all-NaN when
  `defending_gk_player_id` is absent from `actions` — required by VAEP's
  `feature_column_names` introspection path. The aggregator
  `add_pre_shot_gk_position` raises `ValueError` (user-direct boundary).
  Documented in helper docstrings + `pre_shot_gk_default_xfns`.

## [2.8.0] — 2026-05-01

### Added — Tracking-aware action_context features (PR-S20)

- **`silly_kicks.tracking.features`** --- public per-feature surface for
  standard SPADL: `nearest_defender_distance`, `actor_speed`,
  `receiver_zone_density`, `defenders_in_triangle_to_goal`. Plus aggregator
  `add_action_context(actions, frames, *, receiver_zone_radius=5.0) -> pd.DataFrame`
  that enriches input actions with the 4 features + 4 linkage-provenance
  columns (`frame_id`, `time_offset_seconds`, `link_quality_score`,
  `n_candidate_frames`). Decorated with `@nan_safe_enrichment` per ADR-003.
  Plus `tracking_default_xfns` (4 `lift_to_states` wrappers) for
  HybridVAEP integration.
- **`silly_kicks.atomic.tracking.features`** --- atomic-SPADL parity with
  the same public surface (`atomic_tracking_default_xfns`). Mirrors the
  standard surface with atomic-shaped column reads (`x, y, dx, dy`).
- **`silly_kicks.tracking.feature_framework`** --- `ActionFrameContext`
  frozen dataclass + `lift_to_states` (lifts an `(actions, frames) -> pd.Series`
  helper to a `(states, frames) -> Features` transformer). Re-exports
  `frame_aware`, `is_frame_aware`, `Frames`, `FrameAwareTransformer`.
- **`silly_kicks.tracking._kernels`** (private) --- schema-agnostic compute
  kernels shared between standard and atomic public surfaces. Per
  ADR-005 §3 (kernel-extraction pattern).
- **`silly_kicks.tracking.utils._resolve_action_frame_context`** (private)
  --- builds the linked-context structure (linkage pointers + per-action
  actor row + opposite-team frame rows) once per `add_action_context()` call.
- **`silly_kicks.vaep.feature_framework`** --- extended with `frame_aware`
  decorator, `is_frame_aware` predicate, and `Frames` / `FrameAwareTransformer`
  type aliases. Marker-decorator pattern parallels the existing
  `@nan_safe_enrichment` contract (ADR-003).
- **`silly_kicks.vaep.base.VAEP.compute_features` / `rate`** --- additive
  `frames=None` keyword-only parameter. Frame-aware xfn dispatch via
  `is_frame_aware`. `HybridVAEP` and `AtomicVAEP` inherit the extension
  automatically (no code changes in their files). Symmetric LTR-normalization
  via lazy import of `tracking.utils.play_left_to_right` only when
  `frames is not None` (no module-import-time vaep <-> tracking cycle).
- **`silly_kicks._nan_safety`** --- new `is_nan_safe_enrichment(fn)` peer
  predicate to the existing `nan_safe_enrichment` decorator. Mirrors the
  new `is_frame_aware` introspection API.
- **ADR-005** ([docs/superpowers/adrs/ADR-005-tracking-aware-features.md](docs/superpowers/adrs/ADR-005-tracking-aware-features.md))
  --- tracking-aware feature integration contract. Captures the seven
  cross-cutting decisions PR-S20 introduces so PR-S21+ tracking-aware
  features inherit them without re-litigation.
- **`NOTICE`** --- canonical academic-attribution record at repo root,
  mirroring the lakehouse pattern. Cross-linked from `README.md` and
  `CLAUDE.md`. Cites Lucey et al. (2014), Anzer & Bauer (2021),
  Spearman (2018), Power et al. (2017), Pollard & Reep (1997) for the 4
  PR-S20 features, plus the foundational SPADL / VAEP / Atomic-SPADL / xT
  literature.
- **`TODO.md` restructured** to the lakehouse-style "On Deck" table.
  Eleven follow-up tracking-aware features (TF-1..TF-10) tracked with
  Size / Source / Notes columns and academic citations; TF-11 tracks the
  baselines-JSON backfill.
- **Loop 0 lakehouse probe** --- `scripts/probe_action_context_baselines.py`
  pulls slim-slice action+frame parquets per provider into
  `tests/datasets/tracking/action_context_slim/` (sportec / metrica /
  skillcorner; ~10 actions + linked frames each). Probe + outputs
  committed; real datasets are not. Backbone for the cross-provider
  parity test.
- **Tier-3 cross-provider parity test** ---
  `tests/tracking/test_action_context_cross_provider.py` runs
  `add_action_context` against the lakehouse-derived slim parquets per
  provider; asserts bounds + linkage rate >= 95% + actor_speed populated
  >= 80%.
- **e2e real-data sweep** ---
  `tests/tracking/test_action_context_real_data_sweep.py` (4
  e2e-marked tests, env-gated). Mirrors PR-S19's sweep shape: PFF via
  `PFF_TRACKING_DIR`; IDSSE / Metrica / SkillCorner via Databricks SQL.
  Skips with explicit reason on missing env.

### Backward compatibility

- All existing call sites (`v.compute_features(game, actions)`,
  `v.rate(game, actions)`) work verbatim --- `frames=None` is the
  default and walks the same code path. Regression-tested in
  `test_compute_features_frames_none_is_regression_equivalent`.
- No changes to `xfns_default`, `hybrid_xfns_default`, or atomic
  `xfns_default`. Tracking-aware features must be opted in by appending
  `tracking_default_xfns` (or `atomic_tracking_default_xfns`) to the
  caller's xfns list.

## [2.7.0] — 2026-04-30

### Added

- **`silly_kicks.tracking` namespace** --- first-class tracking-data
  support, parallel to `silly_kicks.spadl`. Hexagonal pure-function
  contract: `convert_to_frames(...) -> tuple[pd.DataFrame,
  TrackingConversionReport]`, zero I/O, zero global-state mutation.
  Nineteen-column long-form canonical schema
  (`TRACKING_FRAMES_COLUMNS`), per-provider dtype variants
  (`KLOPPY_TRACKING_FRAMES_COLUMNS`, `SPORTEC_TRACKING_FRAMES_COLUMNS`,
  `PFF_TRACKING_FRAMES_COLUMNS`), 105 x 68 m SPADL coordinates,
  long-form ball-row encoding (`is_ball=True`), `team_attacking_direction` /
  `ball_state` / `speed_source` provenance columns.
- **Four-provider adapter coverage** --- Sportec/IDSSE
  (`silly_kicks.tracking.sportec`, native), PFF
  (`silly_kicks.tracking.pff`, native), Metrica + SkillCorner
  (`silly_kicks.tracking.kloppy`, gateway via `kloppy.TrackingDataset`).
  PFF native is preferred over kloppy's PFF tracking parser for
  symmetry with `silly_kicks.spadl.pff` (PR-S18) and shared use of the
  `_direction.home_attacks_right_per_period` helper.
- **Linkage primitive**
  (`silly_kicks.tracking.utils.link_actions_to_frames` +
  `slice_around_event`) --- the load-bearing cross-pipeline operation
  that PR-S20+ tracking-aware features will build on. Returns pointer
  DataFrame plus `LinkReport` audit. Default tolerance 0.2 s, pinned
  by an explicit default-stability test.
- **Hybrid speed policy** --- adapters trust native speed where
  provided (PFF, Sportec); derive via `_derive_speed` (per-player
  groupby + diff) where missing (Metrica, SkillCorner). The
  `speed_source` column records provenance.
- **Empirical-probe-driven synthetic fixtures** ---
  `scripts/probe_tracking_baselines.py` measures real-data statistics
  (frame rates, NaN-rate-per-column, off-pitch tail rates,
  ball-visibility rates, distance-to-ball percentiles) from the
  lakehouse mart + local PFF; the committed JSON baseline at
  `tests/datasets/tracking/empirical_probe_baselines.json` parameterizes
  the per-provider synthetic generators. `realistic.parquet` fixtures
  inject baseline-calibrated edge cases (off-pitch tail, ball-out
  interval, ball-x throw-in tail) for CI; deterministic
  `tiny.parquet` / `medium_halftime.parquet` remain available for
  exact-answer unit tests.
- **`tests/test_tracking_real_data_sweep.py`** --- e2e-marked sweep
  exercising all four adapters against real data (local PFF JSONL.bz2 +
  lakehouse-derived Sportec / Metrica / SkillCorner samples). Skipped
  in CI; run locally before each tracking PR's single commit.
- **ADR-004**
  (`docs/superpowers/adrs/ADR-004-tracking-namespace-charter.md`) ---
  silly_kicks.tracking namespace charter; nine invariants locking the
  schema + adapter taxonomy + linkage contract for PR-S20+ to inherit.
- **`pyproject.toml`** --- `kloppy` optional minimum bumped to >= 3.18.0
  (kloppy 3.18 ships Metrica + SkillCorner tracking parsers used by the
  gateway). Pytest `pythonpath` config now includes `["", "tests"]` so
  per-provider synthetic-fixture generators are importable in test code
  via `datasets.tracking.<provider>.generate_synthetic`.

### Changed

- **`silly_kicks/spadl/pff.py`** --- the per-period direction lookup
  (`home_attacks_right_per_period`) is extracted into
  `silly_kicks/tracking/_direction.py` so events PFF, tracking PFF,
  and tracking Sportec adapters share one implementation. Pure
  refactor; the events test suite (127 tests) passes unchanged.

### Deferred

Tracking-aware features deferred to follow-up scoping cycles, in
priority order (per ADR-004 invariant 9): `action_context()` (PR-S20,
target 2.8.0), `pressure_on_carrier()`, `infer_ball_carrier()`,
`sync_score()`, pitch-control models (Spearman / Voronoi), smoothing
primitives (Savitzky-Golay, EMA), multi-frame interpolation /
gap filling, ReSpo.Vision adapter (licensing-gated).

## [2.6.0] — 2026-04-30

### Added

- **`silly_kicks.spadl.pff`** — first-class PFF FC / Gradient Sports
  events-data converter. Hexagonal pure-function contract (events
  DataFrame in, SPADL DataFrame + ConversionReport out, zero I/O).
  Mirrors the sportec / metrica converter shape. Dispatch table covers
  PFF's hierarchical event vocabulary (`gameEvents` × `possessionEvents`
  + `set_piece_type`): pass / cross / shot / clearance / dribble (BC) /
  tackle (CH) / keeper_save+keeper_pick_up (RE) / bad_touch (TC) +
  set-piece compositions (kickoff / open play / corner / free kick /
  throw-in / goal kick / penalty) + foul row synthesis with card
  result mapping. Excludes `OUT` / `SUB` / period-boundary / `OTB+IT`
  rows with full ConversionReport audit trail.
- **`silly_kicks.spadl.PFF_SPADL_COLUMNS`** — extended output schema:
  `SPADL_COLUMNS` + four nullable `Int64` tackle-passthrough columns
  (`tackle_winner_player_id`, `tackle_winner_team_id`,
  `tackle_loser_player_id`, `tackle_loser_team_id`) per ADR-001.
  `Int64` (pandas nullable) is a deliberate dtype departure from
  `SPORTEC_SPADL_COLUMNS`'s `object` dtype: PFF identifiers are integers
  whereas kloppy hands sportec strings.
- **Per-period direction-of-play normalization** — first silly-kicks
  converter requiring perspective-real coordinate handling. Two new
  parameters (`home_team_start_left`, `home_team_start_left_extratime`)
  carry the metadata-derived flip information per period.
- **`tests/datasets/pff/`** — synthetic match fixture
  (`synthetic_match.json`) plus deterministic generator
  (`_generate_synthetic_match.py`). Synthetic-only test policy until
  PFF licensing for redistributable real-data slices is confirmed.
- **`docs/examples/pff_wc2022_walkthrough.py`** — end-to-end pipeline
  demonstration (documentation, not test). Reads from a user-supplied
  PFF directory and walks events → SPADL → Atomic-SPADL → coverage /
  boundary metrics → VAEP labels.
- **`TODO.md` Tracking namespace entry** — captures the deferred
  `silly_kicks.tracking.*` design with verified luxury-lakehouse prior
  art (3 providers / 20 matches / ~38M player-frames in
  `soccer_analytics.dev_gold.fct_tracking_frames` as of 2026-04-30) and
  library-native architectural rules.

### Changed

- **`silly_kicks.spadl._finalize_output`** recognizes pandas extension
  dtypes (`Int64`, `Float64`, `boolean`, `string`, etc.) on schema
  entries — small surface-area generalization, fully backwards-
  compatible with existing object/int64 dtype handling. Required for
  `PFF_SPADL_COLUMNS` `Int64` tackle columns.
- **`tests/spadl/test_cross_provider_parity.py`** — PFF added as a
  parametrize entry; participates in the keeper-action emission gate,
  schema-shape gate, and ADR-001 team_id-mirror gate alongside the five
  pre-existing converters.
- **Pre-release empirical validation** — converter validated against the
  full WC 2022 dataset (64 matches, 144,541 events → 91,931 SPADL actions,
  zero conversion failures, zero unrecognized vocabulary). The sweep
  surfaced 6 vocabulary patterns the hand-authored synthetic-fixture suite
  missed (OFF / ON / G / THIRDKICKOFF / FOURTHKICKOFF game_event_types and
  OTB+empty initialNonEvent markers); all are now in the converter's
  excluded vocabulary, exercised by the synthetic fixture, and asserted by
  test_pff.py. Also surfaced a real-data schema detail: PFF stores
  ``fouls`` as a single dict per event (not a JSON array, contrary to
  initial fixture authoring); fixture + loaders updated. Standalone
  ``FOUL`` gameEventType events with ``possessionEventType="FO"`` now
  convert in-place to the canonical foul SPADL row (no phantom non_action
  parent).

## [2.5.0] — 2026-04-30

### Added

- **`silly_kicks._nan_safety.nan_safe_enrichment`** — marker decorator
  declaring an enrichment helper satisfies the NaN-safety contract
  (ADR-003). Sets `fn._nan_safe = True`; CI gates auto-discover decorated
  helpers via this attribute.
- **`goalkeeper_ids: set | None = None`** keyword-only parameter on
  `silly_kicks.spadl.utils.add_gk_role` and
  `silly_kicks.atomic.spadl.utils.add_gk_role`. When provided,
  distribution-detection extends with two additional matching rules:
  (a) `current player_id ∈ goalkeeper_ids` AND prev keeper-type same-team;
  (b) NaN-team fallback — both player_ids NaN AND same team_id AND prev
  keeper-type. Closes the lakehouse coverage gap on IDSSE/Metrica data
  with sparse player attribution. When `None` (default), behavior is
  byte-for-byte unchanged.
- **`tests/test_enrichment_nan_safety.py`** — auto-discovered NaN-fuzz
  test (15 cases). Parametrizes over every `@nan_safe_enrichment` helper
  × synthetic NaN-laced SPADL fixture; asserts no crash + sensible
  defaults. Includes registry-floor sanity assertions that catch silent
  discovery breakage.
- **`tests/test_enrichment_provider_e2e.py`** — auto-discovered
  cross-provider e2e regression (21 cases). Parametrizes over every
  `@nan_safe_enrichment` standard helper × vendored fixtures from
  StatsBomb / IDSSE / Metrica; atomic helpers run on the
  StatsBomb-derived atomic-SPADL fixture.
- **`tests/test_gk_role_goalkeeper_ids.py`** — feature tests for the new
  `goalkeeper_ids` parameter (8 cases): backward-compat, rule (a)
  known-GK match, rule (b) NaN-team fallback, edge cases (atomic, empty
  set, team-boundary respect).
- **`docs/superpowers/adrs/ADR-003-nan-safety-enrichment-helpers.md`** —
  formalizes the NaN-safety contract for public enrichment helpers,
  alternatives considered, and the registry-floor sanity assertion as
  the bulletproof for the auto-discovery mechanism.
- **CLAUDE.md "Key conventions" amendment** pointing to ADR-003.

### Fixed

- **`silly_kicks.spadl.utils.add_pre_shot_gk_context`** —
  `ValueError: cannot convert float NaN to integer` at line 543 when
  the most-recent defending-keeper-action's `player_id` is NaN
  (e.g. IDSSE bronze data with sparse player attribution). Surfaced
  2026-04-30 by the luxury-lakehouse `compute_spadl_vaep` task. Fix:
  detect NaN before the `int(...)` cast; `continue` to next shot
  (defending_gk_player_id stays NaN per the function's documented
  contract). Symmetric fix at
  `silly_kicks.atomic.spadl.utils.add_pre_shot_gk_context` line 826.
- **`silly_kicks.spadl.utils.add_gk_distribution_metrics`** — latent
  `ValueError: cannot convert float NaN to integer` at lines 374-377
  on `.astype(int)` zone-binning when a distribution-eligible row has
  NaN coordinates. Fix: filter `eligible` mask by `np.isfinite(...)`
  on all four coords. Symmetric fix at
  `silly_kicks.atomic.spadl.utils.add_gk_distribution_metrics`
  lines 665-668.
- **`silly_kicks.spadl.utils.coverage_metrics`** (defensive) — same
  `int(NaN)` crash class on `int(tid)` at line 1074 if input has NaN
  `type_id`. Fix: NaN guard before the cast; NaN type_ids tally as
  "unknown". Symmetric fix at
  `silly_kicks.atomic.spadl.utils.coverage_metrics` line 1036. Not
  under ADR-003 (TypedDict-returning, not enrichment helper) — fixed
  while we're here.

### Changed

- 10 public enrichment helpers (5 standard + 5 atomic) decorated with
  `@nan_safe_enrichment`: `add_possessions`, `add_names`, `add_gk_role`,
  `add_gk_distribution_metrics`, `add_pre_shot_gk_context` × 2 packages.

### Notes

- **Hyrum's Law surface:** `add_gk_role.__signature__` gains the new
  `goalkeeper_ids` keyword-only parameter. Consumers using
  `inspect.signature(add_gk_role)` would see the addition. Documented
  in ADR-003 as accepted exposure.
- **Test count:** 884 → 928 passing, 4 deselected (+44 net delta:
  15 fuzz + 21 e2e + 8 goalkeeper_ids feature tests). Pyright clean
  (0 errors / 0 warnings / 0 informations).
- Future direction: nullable-Int64 dtype migration for `player_id` /
  `team_id` columns is the long-term answer to type-level NaN-safety;
  out of scope for this PR (ADR-003 § Notes / Future direction).

## [2.4.0] — 2026-04-30

### Added

- **`silly_kicks.vaep.feature_framework`** — new public module holding the 7
  framework primitives both standard and atomic VAEP feature stacks build on:
  4 type aliases (`Actions`, `Features`, `FeatureTransfomer`, `GameStates`),
  `gamestates`, `simple`, and the promoted helper
  `actiontype_categorical(actions, spadl_cfg)`. Cross-package framework
  boundary now has a name; atomic-VAEP no longer reaches into
  `vaep.features.core` for framework primitives.
- **`actiontype_categorical(actions, spadl_cfg)`** — promoted from the
  previously-private `_actiontype` helper in `vaep.features.core` to a public,
  SPADL-config-parameterized framework helper. Both standard-VAEP and
  atomic-VAEP wrap it with `@simple` to produce their respective `actiontype`
  feature transformers. Drops the implicit-None config fallback (the function
  is meaningless without a config); positional `spadl_cfg` parameter.
  Examples-section docstring per the public-API discipline.
- **`tests/vaep/test_feature_framework_layout.py`** — 7-case framework-layout
  lock (T-D). Asserts each framework primitive's canonical home is
  `silly_kicks.vaep.feature_framework`.
- **`docs/superpowers/adrs/ADR-002-shared-vaep-feature-framework-boundary.md`** —
  captures the framework-extraction decision, the 4 alternatives considered,
  and the `_actiontype → actiontype_categorical` rename rationale.

### Changed

- **`silly_kicks.vaep.features.core` slimmed** to its standard-SPADL-specific
  helpers (`play_left_to_right`, `feature_column_names`); re-exports the
  framework primitives from `silly_kicks.vaep.feature_framework` so existing
  `from silly_kicks.vaep.features.core import gamestates` paths continue to
  resolve (Hyrum's-Law preservation).
- **`silly_kicks.atomic.vaep.features` imports framework directly from
  `vaep.feature_framework`** (no longer reaches into `vaep.features.core`);
  per-concern feature reuse from `bodypart` / `context` / `temporal` is
  preserved (intentional verbatim code-share, not framework leak).
- **`silly_kicks.vaep.features.actiontype` body updated** to call
  `actiontype_categorical(actions, spadlcfg)` instead of the private
  `_actiontype(actions)` (the latter relied on an implicit-None spadlcfg
  fallback; the new call passes spadlcfg explicitly — same resolved
  behaviour).
- **T-A backcompat (`tests/vaep/test_features_backcompat.py`)** gains one row
  for `actiontype_categorical`. 33 → 34 cases.
- **T-B layout (`tests/vaep/test_features_submodule_layout.py`)** drops the 6
  framework rows now living outside the features package. 33 → 27 cases.
- **T-C atomic-coupling (`tests/atomic/test_features_per_concern_import.py`)
  rewritten** to forbid `vaep.features.core` import for framework primitives
  and require import from `vaep.feature_framework`. Retains the existing
  package-root-import forbid + 3 per-concern-import requirements.
- **Examples-gate file list** (`tests/test_public_api_examples.py`) adds
  `silly_kicks/vaep/feature_framework.py`. 26 → 27 cases.

### Removed

- **`silly_kicks.vaep.features.core._actiontype`** — promoted to public
  `actiontype_categorical(actions, spadl_cfg)` in the new framework module.
  Was a leading-underscore-private symbol; never in `__all__`; never
  documented as public surface.

### Closed

- **TODO A9** — `atomic/vaep/features.py` per-concern coupling — closed via
  framework extraction (the trigger-condition resolution from PR-S15's
  deferral). The `## Architecture` section of `TODO.md` is now empty.
  See ADR-002.

### Notes

- **Hyrum's Law surface:** `gamestates.__module__` (and `simple.__module__`)
  flips from `silly_kicks.vaep.features.core` to
  `silly_kicks.vaep.feature_framework`. Consumers introspecting via
  `inspect.getmodule(gamestates)` would see the new value. Documented in
  ADR-002 as accepted exposure.
- **Test count:** 881 → 884 passing, 4 deselected (+3 net delta: +1 T-A row,
  -6 T-B rows, +7 T-D cases, +1 Examples-gate parametrize). Pyright clean
  (0 errors / 0 warnings / 0 informations).

## [2.3.0] — 2026-04-30

### Changed

- **`silly_kicks.vaep.features` decomposed from a 1170-line monolith into a
  package** of 8 concern-focused submodules (`core`, `actiontype`, `result`,
  `bodypart`, `spatial`, `temporal`, `context`, `specialty`). Hybrid visibility:
  every previously-public symbol remains importable via the package path
  (`from silly_kicks.vaep.features import startlocation` keeps working
  unchanged); submodule paths are also importable for advanced/atomic-internal
  use. Closes the long-standing TODO architecture entry. **Pure structural
  refactor — zero behavior change; every existing test passes through every
  step.**
- **`silly_kicks.atomic.vaep.features` updated to import per-concern.** 12
  symbols imported across 4 grouped statements against
  `vaep.features.{core,bodypart,context,temporal}` (was: single 12-symbol
  monolith import). TODO A9 partially addressed (severity Medium → Low) —
  full decoupling deferred until atomic features need to diverge independently.
  Local type alias duplicates (`Actions = pd.DataFrame` etc.) replaced by a
  single import from `vaep.features.core` (DRY cleanup).

### Added

- **8 new public-API submodule paths** (`silly_kicks.vaep.features.core`,
  `.actiontype`, `.result`, `.bodypart`, `.spatial`, `.temporal`, `.context`,
  `.specialty`). Documented as implementation detail of where each symbol
  lives — the canonical entry point remains the package itself.
- **3 new test files locking the structure:** T-A backcompat (33 parametrized
  cases asserting every public symbol stays importable from the package path),
  T-B submodule layout (33 parametrized cases asserting each symbol's
  `__module__` matches the design contract), T-C atomic-per-concern (1 test
  asserting atomic imports from per-concern submodules, not the package root).
- **CI gate (`tests/test_public_api_examples.py::_PUBLIC_MODULE_FILES`)
  widened from 19 → 26 entries** to cover all 8 new submodule paths. Net +7
  parametrize cases.

### Closed

- **TODO A19** (default hyperparameters scattered across 3 learner functions):
  reviewed and closed without code change. Already centralized as
  `_XGBOOST_DEFAULTS` / `_CATBOOST_DEFAULTS` / `_LIGHTGBM_DEFAULTS` module-level
  constants since 1.9.0; the audit description ("scattered across 3 functions")
  predates that extraction.
- **TODO O-M1** (full `events.copy()` at top of StatsBomb `convert_to_actions`):
  reviewed and closed without code change. The defensive copy is correct by
  design — `_flatten_extra` mutates the DataFrame by adding ~22 underscore
  columns; without the copy, caller's events would be mutated in place.
- **TODO O-M6** (temporary n×3 DataFrame for StatsBomb fidelity version check):
  reviewed and closed without code change. ~50 KB peak per match; could be
  numpy-fied for marginal gain (~25 KB savings); no measurable impact.

No API breakage. 881 tests passing (807 baseline + 33 T-A + 33 T-B + 1 T-C +
7 net gate delta), 4 deselected.

## [2.2.0] — 2026-04-30

### Added

- **`silly_kicks.atomic.spadl.coverage_metrics`** — Atomic-SPADL counterpart to
  the standard `silly_kicks.spadl.coverage_metrics` utility (added in 1.10.0).
  Resolves `type_id` against the atomic 33-type vocabulary
  (`silly_kicks.atomic.spadl.config.actiontypes`) including atomic-only types
  (`receival`, `interception`, `out`, etc.) and post-collapse names (`corner`,
  `freekick`). Reuses the standard `CoverageMetrics` TypedDict from
  `silly_kicks.spadl.utils` as the single source of truth — both standard and
  atomic surfaces import the same type. Closes TODO C-1 (deferred from 1.10.0).
- **Examples sections on 25 previously-uncovered public-API surfaces** across
  `silly_kicks/vaep/labels.py` (5), `silly_kicks/vaep/formula.py` (3),
  `silly_kicks/atomic/vaep/features.py` (9), `silly_kicks/atomic/vaep/labels.py` (5),
  and `silly_kicks/atomic/vaep/formula.py` (3). Closes the PR-S13 documentation
  coverage gap.

### Changed

- **CI guardrail (`tests/test_public_api_examples.py`) widened from 14 → 19
  module files.** The gate now mechanically enforces Examples coverage across
  the entire public API surface; future PRs that add a public function
  without an Example fail CI.

No API breakage. New public symbols (`coverage_metrics`, `CoverageMetrics`
re-export) are additive only.

## [2.1.1] — 2026-04-30

### Added

- **Examples sections on all public API surfaces.** Closes the long-standing D-8
  documentation gap. Every public function / class / method in
  `silly_kicks.spadl`, `silly_kicks.atomic.spadl`, `silly_kicks.vaep`,
  `silly_kicks.atomic.vaep`, and `silly_kicks.xthreat` now has a 3-7 line
  illustrative example showing typical usage. ~50 surfaces newly documented.
- **CI guardrail at `tests/test_public_api_examples.py`.** AST-based parametrized
  test asserts every public symbol has an `Examples` section in its docstring.
  Future PRs that add a public function without an Example fail CI; the failure
  message points to canonical-style references (`add_possessions`,
  `boundary_metrics`).

### Changed

- **D-9 entry removed from `TODO.md`.** Tech-debt entry was stale — all 9
  module-level helpers in `silly_kicks/xthreat.py` are already underscore-
  prefixed; the entry tracked work that was completed prior to silly-kicks 2.0.0.

No API or behavior changes.

## [2.1.0] — 2026-04-29

### ⚠️ Breaking

- **`add_possessions` default for `max_gap_seconds` changed from 5.0 to 7.0**
  in both `silly_kicks.spadl.add_possessions` and
  `silly_kicks.atomic.spadl.add_possessions`. Empirically Pareto-optimal at
  the per-match recall floor on 64 StatsBomb WorldCup-2018 matches (full
  campaign data:
  `docs/superpowers/specs/2026-04-29-add-possessions-precision-improvement-design.md`).
  Same input DataFrame produces different `possession_id` values for any
  pair of actions where the time gap is in `[5, 7)` seconds AND the team
  did not change.

  **Opt-out:** explicit `add_possessions(actions, max_gap_seconds=5.0)`.

  This default change is shipped as a minor bump under pragmatic semver
  (luxury-lakehouse is the only known consumer; one-line opt-out preserves
  prior behavior). Strict semver would call this 3.0.0.

### Added

- **`silly_kicks.spadl.add_possessions` (and atomic counterpart)** new
  opt-in keyword-only parameters for precision-improvement rules:

  - `merge_brief_opposing_actions: int = 0` + `brief_window_seconds: float = 0.0`
    (paired) — brief-opposing-action merge rule. Suppresses team-change
    boundaries when team B has 1..N consecutive actions sandwiched between
    team A actions within the time window. Both must be > 0 to enable;
    both 0 to disable; exactly one > 0 raises `ValueError`.
  - `defensive_transition_types: tuple[str, ...] = ()` — defensive-transition
    rule. Listed action types do not trigger team-change boundaries on
    their own. Recommended: `("interception", "clearance")`.

  All defaults disable the rules, preserving 2.0.x algorithmic behavior
  except for the `max_gap_seconds` default change above.

- **`tests/datasets/statsbomb/spadl-WorldCup-2018.h5`** regenerated with
  `preserve_native=["possession"]` — the 64-match HDF5 fixture is now a
  reusable regression corpus for `add_possessions`. New file size ~6 MB
  (one extra `possession` column on ~128K rows under zlib compression).

- **`tests/spadl/test_add_possessions.py::TestBoundaryAgainstStatsBomb64Match`**
  64-match parametrized regression gate complementing the existing 3-fixture
  cross-competition gate. Each match independently gated at
  `recall >= 0.83 AND precision >= 0.30`.

### Changed

- **`silly_kicks/spadl/utils.py`** boundary-detection logic refactored
  into a private `_compute_possession_boundaries` helper, mirroring the
  atomic-side `_compute_possessions` factoring. Public API unchanged;
  internal seam for the new opt-in rules.

- **`tests/spadl/test_add_possessions.py::TestBoundaryAgainstStatsBombNative`**
  per-match recall threshold lowered from 0.85 to 0.83. Absorbs the
  slightly reduced recall margin at the new `max_gap_seconds=7.0` default
  (worst observed across 64 matches: R_min=0.854) plus pandas/numpy
  version-drift safety margin.

### Behavior baselines

`add_possessions` empirical performance at the new default (no opt-in
rules, 64 WC-2018 matches):

| Metric | Mean | sd | Min |
|---|---|---|---|
| Precision | 0.439 | 0.035 | 0.350 |
| Recall | 0.939 | 0.023 | 0.854 |
| F1 | 0.597 | — | — |

(Compare to 2.0.x at `max_gap_seconds=5.0`: P=0.412, R=0.950, F1=0.574.)

Recommended opt-in settings: see `add_possessions` docstring and
`docs/superpowers/specs/2026-04-29-add-possessions-precision-improvement-design.md`.

## [2.0.0] — 2026-04-29

### ⚠️ Breaking

- **`silly_kicks.spadl.sportec.convert_to_actions` no longer overrides
  `team_id` / `player_id` from DFL `tackle_winner` / `tackle_winner_team`
  qualifiers.** Per ADR-001
  (`docs/superpowers/adrs/ADR-001-converter-identifier-conventions.md`),
  the SPADL converter contract is "caller's identifier conventions are
  sacred — never overridden from qualifiers." Caller-supplied `team` /
  `player_id` values mirror verbatim into the output. Pre-2.0.0 behavior
  silently rewrote ~56% of tackle rows on consumers using a
  caller-normalized `team` convention (see luxury-lakehouse PR-LL2
  close-out report).
- **Sportec output schema changes from `KLOPPY_SPADL_COLUMNS` to
  `SPORTEC_SPADL_COLUMNS`** — 14 + 4 = 18 columns. The 4 new columns
  surface DFL qualifier values: `tackle_winner_player_id`,
  `tackle_winner_team_id`, `tackle_loser_player_id`,
  `tackle_loser_team_id`. NaN on non-tackle rows; NaN when the qualifier
  is absent. Sportec consumers asserting against `KLOPPY_SPADL_COLUMNS`
  must switch to `SPORTEC_SPADL_COLUMNS`.

### Migration

If your pre-2.0.0 sportec consumer relied on the tackle-winner override
AND your upstream `team` / `player_id` columns are in the same
identifier convention as DFL's `tackle_winner_team` / `tackle_winner`
qualifiers (raw `DFL-CLU-...` / `DFL-OBJ-...`), call the new helper
post-conversion:

```python
from silly_kicks.spadl import sportec, use_tackle_winner_as_actor
actions, _ = sportec.convert_to_actions(events, home_team_id="DFL-CLU-XXX")
actions = use_tackle_winner_as_actor(actions)
```

If your `team` / `player_id` columns use any other convention, the
post-1.10.0 behavior already preserved your conventions correctly — no
migration needed; the bug fix is automatic on upgrade.

### Added

- **First silly-kicks ADR.** `docs/superpowers/adrs/ADR-001-converter-identifier-conventions.md`
  + `docs/superpowers/adrs/ADR-TEMPLATE.md` (vendored verbatim from
  luxury-lakehouse) establish the silly-kicks ADR pattern. Future
  decisions that add an exception to project-wide conventions, change
  schema ownership, or hardcode a workaround for a platform constraint
  get an ADR.
- **`silly_kicks.spadl.SPORTEC_SPADL_COLUMNS`** schema constant (18-key
  dict) — extends `KLOPPY_SPADL_COLUMNS` with the 4 tackle qualifier
  passthrough columns. Re-exported from `silly_kicks.spadl`.
- **`silly_kicks.spadl.use_tackle_winner_as_actor(actions) -> pd.DataFrame`**
  — pure post-conversion enrichment that restores pre-2.0.0 sportec
  SPADL "actor = winner" semantic for callers whose upstream identifier
  convention matches DFL's qualifier format. Raises `ValueError` early
  on missing required columns. Mirrors the `add_*` helper family pattern.
- **Cross-provider parity regression gate**
  (`tests/spadl/test_cross_provider_parity.py::test_team_id_mirrors_input_team`).
  Parametrized over all 5 DataFrame converters; asserts each output's
  `team_id` values are a subset of the input `team` values. Locks the
  ADR-001 contract per-provider going forward; would have caught the
  1.7.0 sportec bug.
- **e2e on the IDSSE production fixture**
  (`TestSportecAdrContractOnProductionFixture`, 5 tests). Verifies the
  contract works on production-shape data: caller's labels survive
  through the converter; the 4 new columns are populated for qualifier
  rows; the migration helper round-trips correctly; 1.10.0 keeper
  coverage is preserved.

### Changed

- **CLAUDE.md "Key conventions" section** gains one rule citing ADR-001:
  "Converter identifier conventions are sacred. SPADL DataFrame
  converters never override the caller's `team_id` / `player_id`
  columns from provider-specific qualifiers..."
- **Sportec module docstring** documents the 4 tackle qualifier
  passthrough columns + the `SPORTEC_SPADL_COLUMNS` schema + the
  migration helper. References ADR-001.

### Removed

- **`silly_kicks.spadl.sportec` tackle override block** at the previous
  `sportec.py:559-565`. The 6-line override that silently rewrote
  `team_id` / `player_id` from raw DFL qualifier values is gone.
- **`tests/spadl/test_sportec.py::TestSportecActionMappingShotsTacklesFoulsGK::test_tackle_uses_winner_as_actor`**
  — was asserting the now-removed override. Covered by the new
  `TestSportecTackleNoOverride` + `TestSportecTackleWinnerColumns`
  classes.

### Audit findings

Manual cross-converter review (this cycle) confirmed sportec.tackle
was the unique violator of the ADR-001 contract:

| Converter | Override `player_id` / `team_id`? | Notes |
|---|---|---|
| `silly_kicks.spadl.sportec` | YES (removed) | The bug. |
| `silly_kicks.spadl.metrica` | NO | 1.10.0 GK routing only changes `type_id` / `bodypart_id`. |
| `silly_kicks.spadl.wyscout` | NO | 1.0.0 aerial-duel reclassification only changes `type_id` / `subtype_id`. |
| `silly_kicks.spadl.statsbomb` | NO | No qualifier-driven overrides. |
| `silly_kicks.spadl.opta` | NO | No qualifier-driven overrides. |
| `silly_kicks.spadl.kloppy` | NO | Gateway path. |

The 2.0.0 change is surgical (one converter), but the parity gate locks
the contract for all future converter additions.

### Notes

- silly-kicks 2.0.0 is the project's first semver-major release. The
  library is ~3 weeks old (0.1.0 shipped 2026-04-06); major versions
  aren't precious — bumping locks the contract before more downstream
  consumers pin against pre-2.0.0 behavior.
- luxury-lakehouse can bump `silly-kicks>=2.0.0,<3.0` and (optionally)
  drop their `_team_label_to_dfl_id` shim from PR-LL2 close-out, OR
  keep it as a documented winner-attribution post-conversion pattern.

## [1.10.0] — 2026-04-29

### Added
- **Public `silly_kicks.spadl.coverage_metrics(*, actions, expected_action_types)` utility**
  for computing per-action-type coverage on a SPADL action stream. Returns
  a `CoverageMetrics` TypedDict (also re-exported from `silly_kicks.spadl`).
  Keyword-only arguments. Resolves `type_id` to action-type name via
  `spadlconfig.actiontypes_df`; reports any expected action types that
  produced zero rows under `missing`. Out-of-vocab `type_id` values are
  reported as `"unknown"` rather than raising. Mirrors the PR-S8
  `boundary_metrics` shape and discipline.
- **`goalkeeper_ids: set[str] | None = None` parameter on
  `silly_kicks.spadl.sportec.convert_to_actions`** as a supplementary
  signal: when provided, Play events whose `player_id` is in the set
  AND which have NO explicit `play_goal_keeper_action` qualifier are
  routed to the keeper_pick_up + pass 2-action synthesis. The
  qualifier-driven mapping remains the primary contract.
- **`goalkeeper_ids: set[str] | None = None` parameter on
  `silly_kicks.spadl.metrica.convert_to_actions`** as the PRIMARY
  mechanism for surfacing GK actions. Metrica's source format lacks
  native GK markers; with `goalkeeper_ids`, conservative routing applies
  (PASS by GK → synth, RECOVERY by GK → keeper_pick_up, CHALLENGE
  AERIAL-WON by GK → keeper_claim). Without it: 0 keeper_* actions
  (1.9.0 default behaviour preserved — no breaking change).
- **`goalkeeper_ids` no-op acceptance on `statsbomb.convert_to_actions`
  and `opta.convert_to_actions`** for cross-provider API symmetry. Both
  source formats natively mark GK actions; the parameter is silently
  accepted with byte-for-byte identical output.
- **DFL distribution qualifiers `throwOut` and `punt` now produce SPADL
  actions** (sportec converter). Each source row synthesizes TWO
  actions: `keeper_pick_up + pass` (bodypart=other) for `throwOut`,
  `keeper_pick_up + goalkick` (bodypart=foot) for `punt`. Both rows
  inherit the source's `(player_id, team, period, time, x, y)`.
  `preserve_native` columns propagate to both. Action_ids renumbered
  dense after synthesis.
- **Production-shape vendored fixtures** under
  `tests/datasets/idsse/sample_match.parquet` (~166 KB; 308-row subset
  of `soccer_analytics.bronze.idsse_events` match `idsse_J03WMX`,
  includes throwOut + punt rows) and
  `tests/datasets/metrica/sample_match.parquet` (~20 KB; 300-event
  subset of Metrica Sample Game 2). Build script at
  `scripts/extract_provider_fixtures.py` (Databricks pull for IDSSE,
  offline kloppy-fixture subset for Metrica). Attribution READMEs
  alongside.
- **Cross-provider parity meta-test** at
  `tests/spadl/test_cross_provider_parity.py`. Parametrized over all 5
  DataFrame converters (statsbomb, opta, wyscout, sportec, metrica);
  asserts each emits at least one `keeper_*` action when given a
  fixture exercising GK paths. This is the regression gate that would
  have caught Bugs 1-3 in 1.7.0 if it had existed.
- **`pyarrow>=14.0.0` added to `[test]` extras** to back parquet I/O
  for the new fixtures (`pd.read_parquet` / `pd.DataFrame.to_parquet`).

### Fixed
- **Sportec converter no longer drops all DFL `Play` events to
  non_action.** The pre-1.10.0 dispatch checked `et == "Pass"` for
  pass-class events, but DFL bronze never emits `"Pass"` — the actual
  event_type is `"Play"`. Net effect since 1.7.0: all IDSSE matches in
  production lost ~60-80% of their actions (every pass, cross, and head
  pass) to silent non_action drop. Fix restructures the dispatch so
  `Play` events with no GK qualifier route to `pass` / `cross` (with
  optional head bodypart) and `Play` events with a recognized GK
  qualifier route to `keeper_*` actions. Defensive: `Play` events with
  an unrecognized non-empty qualifier still drop to `non_action`.
  ``"Pass"`` is removed from the recognized event-type vocabulary so
  legacy callers (if any) surface in `unrecognized_counts` (loud)
  rather than silently mapping to non_action.
- **Sportec converter no longer drops `throwOut` and `punt` GK
  distribution events to non_action.** These DFL qualifier values
  represent GK distribution actions (throwing or kicking the ball to
  a teammate); pre-1.10.0 they were unmapped. Fix synthesizes 2
  SPADL actions per source event (see Added section).
- **Metrica converter now produces non-zero GK coverage when
  `goalkeeper_ids` is supplied.** Pre-1.10.0 the converter had no
  mechanism to surface GK actions, leaving downstream `add_gk_role` /
  `add_pre_shot_gk_context` enrichments at 100% NULL on every Metrica
  match in production.

### Notes
- This release closes the upstream gap that surfaced during
  luxury-lakehouse PR-LL2 production deploy (2026-04-29): post-deploy
  validation found 100% NULL `gk_role` and `defending_gk_player_id` on
  IDSSE (2,522 rows) and Metrica (5,839 rows) sources. With silly-kicks
  1.10.0, downstream lakehouse can re-run `apply_spadl_enrichments`
  against IDSSE + Metrica with non-NULL GK coverage (handled by
  separate lakehouse PR-LL3).
- Behaviour change for IDSSE consumers: bronze.spadl_actions row count
  per IDSSE match will increase materially (every Play event now
  surfaces as a SPADL pass, plus throwOut/punt rows now produce 2
  actions each). This is the intended fix; downstream aggregation may
  need to re-baseline.
- Wyscout converter unchanged — `goalkeeper_ids` was already present
  from 1.0.0.
- Atomic-SPADL `coverage_metrics` parity is queued as tech debt
  (atomic uses 33 action types vs standard's 23; deferred until a
  consumer asks). Tracked in `TODO.md ## Tech Debt`.

## [1.9.0] — 2026-04-29

### Added
- **Vendored `tests/datasets/statsbomb/spadl-WorldCup-2018.h5`** — committed
  HDF5 fixture for the FIFA World Cup 2018 (64 matches, 128,484 SPADL
  actions, 5.9 MB on disk with zlib compression). All 5 prediction
  pipeline tests in `tests/vaep/`, `tests/test_xthreat.py`, and
  `tests/atomic/` now run on every PR + push. Pre-1.9.0 these tests
  silently skipped in CI and locally because the fixture was never
  committed. Net: ~9 release cycles of zero coverage on the prediction
  pipeline (VAEP fit/rate, xT fit/rate, atomic VAEP fit/rate) is now
  closed.
- **`scripts/build_worldcup_fixture.py`** — reproducible HDF5 generator.
  Downloads StatsBomb open-data WorldCup-2018 raw events (cached at
  `tests/datasets/statsbomb/raw/.cache/`, gitignored), converts each via
  `silly_kicks.spadl.statsbomb.convert_to_actions`, writes the multi-key
  HDFStore. CLI: `--output`, `--cache-dir`, `--no-cache`, `--verbose`,
  `--quiet`. Cold-cache run on broadband: ~30-60 sec. Warm-cache re-run:
  ~5 sec. No new dependencies (stdlib + pandas + already-present
  pytables).
- **`scripts/` is now linted in CI** — `.github/workflows/ci.yml` runs
  `ruff check` and `ruff format --check` on `silly_kicks/`, `tests/`,
  AND `scripts/`. Pyright include stays `silly_kicks/` only — build
  scripts aren't worth full type-checking.

### Changed
- **`tests/conftest.py::sb_worldcup_data` calls `pytest.fail` instead of
  `pytest.skip` when the HDF5 is absent.** Matches the PR-S8 pattern for
  committed fixtures: once a fixture is committed, "missing" is a
  packaging error worth surfacing prominently — not a silent skip that
  lets CI quietly regress. Failure message points at the build script
  for regeneration.
- The 5 `test_predict*` cases (`tests/vaep/test_vaep.py::test_predict`,
  `tests/vaep/test_vaep.py::test_predict_with_missing_features`,
  `tests/test_xthreat.py::test_predict`,
  `tests/test_xthreat.py::test_predict_with_interpolation`,
  `tests/atomic/test_atomic_vaep.py::test_predict`) no longer carry the
  `@pytest.mark.e2e` marker. They run in the regular suite on every CI
  matrix slot (4 slots, ~5-15 sec overhead per slot — negligible).

### Fixed
- **`silly_kicks.xthreat.ExpectedThreat.interpolator()` is no longer
  broken on SciPy 1.14+.** The wrapper used `scipy.interpolate.interp2d`
  which was removed in SciPy 1.14.0 (the import succeeds but the call
  raises `NotImplementedError`). The bug was latent since 1.0.0 because
  `tests/test_xthreat.py::test_predict_with_interpolation` was the only
  consumer and it was `@pytest.mark.e2e`-marked + skipping silently.
  Surfaced precisely when this PR dropped the marker. Replaced with
  `scipy.interpolate.RectBivariateSpline` — the SciPy-recommended
  bug-for-bug compatible replacement for regular grids — wrapped to
  preserve the legacy `interp(xs, ys) -> (W, L)` calling convention so
  callers downstream of `interpolator()` need no changes. Output shape
  and indexing semantics unchanged.
- The `test_interpolate_xt_grid_no_scipy` regression test that mocks
  the missing-scipy path now mocks `RectBivariateSpline` instead of the
  removed `interp2d`.

### Documentation
- **`docs/DEFERRED.md` deleted; live items migrated to a new `## Tech
  Debt` section in `TODO.md`.** Per the National Park Principle —
  bundle the cleanup of the rotting parallel doc into this cycle since
  we're already touching `TODO.md` anyway. Audit history preserved in
  `git log -- docs/DEFERRED.md`. Migrated items: A19 (default
  hyperparameters scattered), D-9 (5 xthreat module-level functions
  naming), O-M1 (StatsBomb `events.copy()`), O-M6 (StatsBomb fidelity
  version check temporary DataFrame). Items judged "by design / accept"
  and not migrated: A15 (kloppy LSP differs by design), A16 (no plugin
  registry — YAGNI for 4 converters), A17 (`_fit_*` coupling — partial
  refactor done, diminishing returns), S5 (optional ML deps no upper
  bounds — librarian convention).
- `CLAUDE.md` no longer references `docs/DEFERRED.md` (file removed).

### Notes
- WorldCup HDF5 file size: 5.9 MB on disk (well under GitHub's 50 MB soft
  warn / 100 MB hard reject thresholds — no Git LFS needed). Total wheel
  size unchanged (test fixtures live under `tests/`, excluded from
  `[tool.hatch.build.targets.wheel] packages = ["silly_kicks"]`).
- The `tests/datasets/statsbomb/raw/.cache/` directory is gitignored —
  raw event JSONs (~192 MB total) are downloaded on demand by the build
  script and never committed.

## [1.8.0] — 2026-04-29

### Added
- **Public `silly_kicks.spadl.boundary_metrics(*, heuristic, native)` utility**
  for computing precision / recall / F1 between two possession-id sequences.
  Returns a `BoundaryMetrics` TypedDict (also re-exported from
  `silly_kicks.spadl`). Keyword-only arguments — the metric is asymmetric
  (precision and recall swap when inputs swap), so positional usage is a
  silent footgun the API surface eliminates. Returns `0.0` for any metric
  whose denominator is zero (empty / single-row / constant sequences).
  Length-mismatched inputs raise `ValueError`.
- 3 vendored StatsBomb open-data fixtures under
  `tests/datasets/statsbomb/raw/events/` (matches 7298, 7584, 3754058 —
  Women's World Cup, Champions League, Premier League; ~9 MB total).
  License attribution in `tests/datasets/statsbomb/README.md`. Used by
  the new parametrized regression gate.

### Changed
- **`add_possessions` docstring is now honest about empirical performance.**
  The previous "boundary-F1 ~0.90" claim was 30+ percentage points above
  the actual measurement on StatsBomb open-data. New text reports
  recall ~0.93, precision ~0.42, F1 ~0.58 (peak ~0.605 at
  `max_gap_seconds=10.0`) and explains why precision is the way it is
  (intrinsic to the team-change-with-carve-outs algorithm class, not a
  defect — StatsBomb's proprietary annotation merges brief opposing-
  team actions back into the containing possession; the heuristic
  cannot replicate that structurally).
- **e2e validation gate replaces F1 ≥ 0.80 with recall ≥ 0.85 AND
  precision ≥ 0.30 per match.** Recall enforces the helper's primary
  contract (catching every real boundary). Precision floor catches the
  "boundary cardinality halved or doubled" regression class that affects
  per-possession aggregation downstream. F1 stays in the assert message
  for diagnostics only — gating on F1 would re-introduce the
  misrepresentation problem this PR is fixing.
- **Test class renamed** `TestBoundaryF1AgainstStatsBombNative` →
  `TestBoundaryAgainstStatsBombNative`. Parametrized over the 3 vendored
  fixtures with per-match independent gates.

### Fixed
- **e2e regression coverage now actually runs in CI.** The previous
  `TestBoundaryF1AgainstStatsBombNative::test_boundary_f1_against_native_possession_id`
  was `@pytest.mark.e2e` and silently skipped on every CI run since
  1.2.0 because the fixture wasn't committed. It was also skipping
  locally (the fixture was never on the user's only development
  machine). Net: ~6 release cycles of zero coverage on this test. PR-S8
  vendors the fixtures and drops the marker so the test runs on every
  PR + push.

### Notes
- Empirical baselines verified locally on the committed fixtures:
  recall {0.9425, 0.9268, 0.9259}, precision {0.4484, 0.4306, 0.3855},
  F1 {0.6077, 0.5880, 0.5443} for matches 7298 / 7584 / 3754058
  respectively. All comfortably above the gate thresholds; tightest
  margin is precision on 3754058 (8.55pp above floor).
- The 5 `test_predict*` cases in `tests/vaep/`, `tests/test_xthreat.py`,
  and `tests/atomic/` continue to skip in CI (and locally) because they
  depend on the un-committed `tests/datasets/statsbomb/spadl-WorldCup-2018.h5`
  fixture. Closing that gap is queued as PR-S9 (generate the HDF5 from
  open-data raw events; commit + drop e2e markers). Tracked in
  `TODO.md`.
- Algorithmic precision improvement for `add_possessions` is queued as
  PR-S10 (look-ahead merge rules for brief opposing-team actions;
  re-measure `max_gap_seconds` defaults using the new
  `boundary_metrics` utility).

## [1.7.0] — 2026-04-29

### Added
- **Dedicated DataFrame SPADL converters for Sportec and Metrica.** New
  modules `silly_kicks.spadl.sportec` and `silly_kicks.spadl.metrica`
  expose `convert_to_actions(events_df, home_team_id, *,
  preserve_native=None) -> tuple[pd.DataFrame, ConversionReport]`,
  matching the established `statsbomb` / `wyscout` / `opta` shape.
  Designed for consumers who already have normalized event data in
  pandas form (lakehouse bronze layers, ETL pipelines, research
  notebooks) and don't want to reconstruct a kloppy `EventDataset` from
  flat rows. Existing kloppy-path consumers continue to use
  `silly_kicks.spadl.kloppy` — both paths produce equivalent SPADL output
  (empirically verified by cross-path consistency tests under
  `tests/spadl/test_sportec.py::TestSportecCrossPathConsistency` and
  `tests/spadl/test_metrica.py::TestMetricaCrossPathConsistency`).
- ~120 recognized DFL qualifier columns surfaced via Sportec converter,
  covering pass / shot / tackle / foul / set-piece / play / cross /
  cards / substitution / penalty / VAR / chance / specialised /
  tracking-derived qualifier groups.
- Metrica set-piece-then-shot composition rule: `SET PIECE` (FREE KICK)
  immediately followed (≤ 5s, same player, same period) by `SHOT`
  upgrades the shot to SPADL `shot_freekick` and drops the SET PIECE
  row.

### Changed
- **`silly_kicks.spadl.kloppy.convert_to_actions` now applies
  `_fix_direction_of_play` automatically** (extracting home team from
  `dataset.metadata.teams[0].team_id`). Pre-1.7.0 the kloppy converter
  was the lone outlier among silly-kicks SPADL converters — it stayed
  in kloppy's `Orientation.HOME_AWAY` (home plays LTR, away plays RTL)
  while StatsBomb / Wyscout / Opta all flipped away-team coords for
  canonical "all-actions-LTR" SPADL convention. 1.7.0 unifies the
  convention across all 6 converters
  (`statsbomb` / `wyscout` / `opta` / `kloppy` / new `sportec` / new
  `metrica`) so all converters emit semantically equivalent SPADL output
  for the same source event stream. Hyrum's Law disclaimer: zero current
  consumers built against 1.6.0's HOME_AWAY-oriented kloppy output (per
  user confirmation during brainstorming).

### Notes
- Cross-path consistency proof: dedicated DataFrame converters and the
  kloppy gateway path produce equivalent SPADL DataFrames when given
  the same source data bridged through test helpers.
- New shared pytest conftest at `tests/spadl/conftest.py` provides
  module-scoped `sportec_dataset` and `metrica_dataset` fixtures
  reusable across `test_kloppy.py`, `test_sportec.py`, and
  `test_metrica.py`.

## [1.6.0] — 2026-04-28

### Added
- **Kloppy converter: Sportec and Metrica support.** `Provider.SPORTEC`
  (Sportec Solutions / IDSSE Bundesliga event format) and `Provider.METRICA`
  (Metrica Sports) are now first-class allowlisted providers in
  `silly_kicks.spadl.kloppy.convert_to_actions`. Empirical verification on
  real fixture data confirms zero new event-type mappings are required —
  both providers' kloppy serializers emit only event types already covered
  by the existing `_MAPPED_EVENT_TYPES` ∪ `_EXCLUDED_EVENT_TYPES` sets.
  `preserve_native` works transparently for both (their `raw_event` is a
  `dict`).
- Real-fixture end-to-end test suites for Sportec and Metrica under
  `tests/spadl/test_kloppy.py`, plus a parametrized coordinate-clamping
  test and a per-provider `ConversionReport` shape test. Test fixtures
  vendored from kloppy's BSD-3-Clause-licensed test files into
  `tests/datasets/kloppy/`.

### Fixed
- **`_SoccerActionCoordinateSystem` was unusable on real datasets.** The
  class definition omitted `__init__`, but `convert_to_actions()`
  instantiated it with `pitch_length=` / `pitch_width=` kwargs. On any
  kloppy version with the current `CoordinateSystem` ABC signature
  (kloppy 3.15+), this raised `TypeError` the moment a real
  `EventDataset` reached `dataset.transform()`. Latent since 1.0.0
  because pre-existing `tests/spadl/test_kloppy.py` was pure mocks
  that never reached the transform call. Affected **all** kloppy-based
  conversion including the previously-allowlisted StatsBomb path.
- 2 pyright errors in `silly_kicks/xthreat.py:402` surfaced by newer
  pandas-stubs / numpy-stubs versions: explicit `dtype=np.float64` added
  to two `np.linspace` calls so the inferred `NDArray[float64]` matches
  the `interp(...)` callable signature.

### Changed
- **Kloppy converter now clamps output coordinates to
  `[0, field_length] × [0, field_width]` (105 × 68 m).** This aligns the
  kloppy converter with the established silly-kicks convention — StatsBomb
  / Wyscout / Opta converters all clamp; kloppy was the lone outlier.
  Empirically Metrica events emit slight off-pitch coords (observed
  `x ∈ [-1.62, 104.63]` on the sample game) within source-recording-noise
  tolerance. Downstream consumers depending on raw off-pitch coordinates
  from the kloppy path specifically should re-verify (no such consumer
  documented).

## [1.5.0] — 2026-04-27

### Added
- **Atomic-SPADL parity for the 1.1.0 → 1.4.0 helper family.** The five
  helpers shipped on standard SPADL (`preserve_native` primitive,
  `add_possessions`, `add_gk_role`, `add_gk_distribution_metrics`,
  `add_pre_shot_gk_context`) plus a new defensive `validate_atomic_spadl`
  helper now have first-class atomic counterparts under
  `silly_kicks.atomic.spadl`:
  - `convert_to_atomic(actions, *, preserve_native=...)` — surfaces
    caller-attached columns from the input SPADL dataframe alongside the
    canonical 13 atomic columns. Synthetic atomic rows generated by the
    conversion (`receival` / `interception` / `out` / `offside` / `goal`
    / `owngoal` / `yellow_card` / `red_card`) receive `NaN` in the
    preserved columns — same behaviour as the standard converters'
    `preserve_native` for synthetic dribble rows.
  - `add_possessions(actions)` — atomic counterpart with two atomic-
    specific adaptations: (a) set-piece restart names match the post-
    collapse atomic types (`corner` / `freekick` / `throw_in` /
    `goalkick`); (b) `yellow_card` / `red_card` synthetic rows are
    transparent to boundary detection — they never trigger a possession
    boundary on their own and inherit the surrounding state via
    forward-fill within `game_id`.
  - `add_gk_role(actions)` — atomic counterpart; reads `x` (NOT
    `start_x`) for the penalty-area threshold check. Same five
    categories.
  - `add_gk_distribution_metrics(actions, xt_grid=None)` — atomic
    counterpart with three atomic-specific adaptations: (a) length is
    `sqrt(dx² + dy²)` from atomic's `(dx, dy)` columns; (b) xT delta is
    from `(x, y)` to `(x + dx, y + dy)`; (c) pass success is detected
    from the FOLLOWING atomic action by row index (`receival` =
    success; `interception` / `out` / `offside` = failure; no following
    action = conservative failure with `gk_xt_delta = NaN`). Atomic
    launch types collapse `{pass, goalkick, freekick_short,
    freekick_crossed}` into `{pass, goalkick, freekick}` (where
    `freekick` is the post-collapse name).
  - `add_pre_shot_gk_context(actions)` — atomic counterpart; recognises
    only `shot` and `shot_penalty` as shot rows. (Standard SPADL's
    `shot_freekick` is collapsed into atomic's `freekick`, mixing
    pass-class and shot-class freekicks; the helper does not attempt to
    disambiguate.)
  - `validate_atomic_spadl(df)` — defensive schema validator. Returns
    input unchanged for chaining; warns on dtype mismatches; raises on
    missing columns.

  All five helpers are vectorised on numpy/pandas; sub-50ms per 1500-
  action match (CI hard bound 200ms; benchmark assertions in
  `tests/test_benchmark.py`). 174 new atomic tests including a
  cross-validation suite asserting algorithmic equivalence between the
  standard and atomic helpers when applied to a SPADL stream and its
  atomic projection.

### Fixed
- Test infra: `tables>=3.9.0` (pytables) added to the `[test]` extras —
  required by `pd.HDFStore` for the `sb_worldcup_data` fixture in
  `tests/conftest.py`. Without it, the 5 `test_predict*` cases (vaep /
  xthreat / atomic vaep) errored at collection time with
  `ImportError("Missing optional dependency 'pytables'")`.
- Test infra: the `sb_worldcup_data` fixture now `pytest.skip(...)`s
  when the `spadl-WorldCup-2018.h5` dataset is not present locally,
  rather than erroring with `FileNotFoundError`. Aligns with the
  `@pytest.mark.e2e` semantics ("requires downloaded datasets") for the
  5 affected tests.

### Notes
- Atomic-SPADL parity TODO is now closed.

## [1.4.0] — 2026-04-27

### Added
- **GK analytics suite v1** — three composable post-conversion enrichments
  for SPADL action streams, mirroring the public-helper shape of
  `add_names()` and `add_possessions()`:
  - `add_gk_role(actions)` — tags each action with the goalkeeper's role
    context: `shot_stopping` / `cross_collection` / `sweeping` / `pick_up` /
    `distribution` (or `None` for non-GK actions). Sweeping is a
    position-based override for `keeper_*` actions taken outside the
    penalty area; in clean event data only `keeper_save` realistically
    appears outside the box (sweeper-style rush-out save). The other
    three keeper types outside the box are illegal handball offences and
    effectively non-existent in regulation play.
  - `add_gk_distribution_metrics(actions, xt_grid=None)` — adds
    `gk_pass_length_m`, `gk_pass_length_class` (short/medium/long),
    `is_launch`, and `gk_xt_delta` to GK distribution actions. Auto-calls
    `add_gk_role` when `gk_role` column is absent. xT delta only computed
    for successful distributions when an xT grid is provided. `is_launch`
    requires both length > `long_threshold` and a deliberate-distribution
    pass type (`pass`, `goalkick`, `freekick_short`, `freekick_crossed`).
  - `add_pre_shot_gk_context(actions)` — for every shot, looks back up to
    `lookback_actions` rows or `lookback_seconds` seconds (smaller wins)
    in the same `(game_id, period_id)` and tags the defending GK's recent
    activity: `gk_was_distributing`, `gk_was_engaged`,
    `gk_actions_in_possession`, `defending_gk_player_id`. Genuinely novel
    — no published OSS / academic equivalent surfaces a goalkeeper's
    pre-shot activity context as explicit per-shot features.

  All three are vectorised on numpy/pandas; sub-50ms per 1500-action match.
  References cited in docstrings: Yam (MIT Sloan), Lamberts GVM (2025),
  Butcher et al. xGOT (2025).

### Notes
- Atomic-SPADL parity for the GK analytics suite is deferred (TODO under
  `## Architecture`). Same disposition as `add_possessions`.

## [1.3.0] — 2026-04-27

### Added
- `pandas-stubs>=2.2.0` pinned in the `[dev]` extras and the CI lint job.
  Without `pandas-stubs`, pyright's bundled pandas typings under-report
  Series / DataFrame types (e.g. arithmetic on ``.values`` collapses to
  the union ``np_1darray | ExtensionArray | Categorical``), masking real
  type issues in CI while spuriously failing locally on certain method
  chains. With `pandas-stubs` in the dev path, pyright reports a
  consistent set of issues across all environments.

### Fixed
- 15 type errors that surfaced once `pandas-stubs` was installed:
  - `vaep/features.py` and `atomic/vaep/features.py` — replaced
    `Series.values` with `Series.to_numpy()` in polar-coordinate
    arithmetic so the return type is `np.ndarray` instead of the
    ``np_1darray | ExtensionArray | Categorical`` union (which doesn't
    support `**` / `/` / `-`).
  - `spadl/opta.py` — same `.values` → `.to_numpy()` swap in
    ``_fix_owngoals`` arithmetic.
  - `spadl/statsbomb.py` — synthetic interception-event `extra` payload
    now built as an explicit ``pd.Series([..], dtype=object)`` instead
    of `[dict] * n`, matching pandas-stubs's accepted setitem value types.
  - `spadl/utils.py` `_finalize_output()` — schema dtype string passed
    through `np.dtype(...)` so it narrows to ``DtypeObj`` for the
    `astype` overload set.
- Removed two `cast(pd.DataFrame, ...)` workarounds in
  `add_possessions` (introduced in 1.2.0). With `pandas-stubs`,
  non-inplace ``sort_values()`` / ``drop()`` correctly return
  `DataFrame`, making the casts redundant.

## [1.2.0] — 2026-04-27

### Added
- `silly_kicks.spadl.utils.add_possessions(actions, *, max_gap_seconds=5.0,
  retain_on_set_pieces=True)` — provider-agnostic possession-sequence
  reconstruction for any SPADL action stream. Adds a `possession_id: int64`
  column via a team-change-with-carve-outs heuristic: boundaries on team
  change, period change (within a game), or time gap >= `max_gap_seconds`,
  with a foul→opposing-team-set-piece carve-out that retains the previous
  possession (the team that won the foul resumes its sequence). Counter
  resets to 0 at each new `game_id`. Mirrors the public-enrichment shape
  of `add_names()` (post-conversion, returns a copy with the new column).
  Vectorised on numpy/pandas; ~1ms per 1500-action match, sub-3ms on 10k.
- Performance benchmarks for `add_possessions` (1500-action and 10k-action
  scenarios) added to `tests/test_benchmark.py` with hard CI bounds
  (200ms / 2s respectively) catching accidental quadratic regressions.
- e2e-marked boundary-F1 validation test against StatsBomb's native
  `possession` field (using `preserve_native=['possession']` from 1.1.0
  to surface the native truth alongside the heuristic). Skips when the
  raw StatsBomb fixture is absent; documents the validation procedure
  for downstream consumers wanting to re-measure the agreement rate
  against their own data.

### Notes
- Atomic-SPADL parity for `add_possessions` is deferred (TODO under
  `## Architecture`). Apply the same passthrough mechanism when there's
  a concrete consumer asking for it.

## [1.1.0] — 2026-04-27

### Added
- `preserve_native` parameter on `convert_to_actions` for all four SPADL
  converters (`statsbomb`, `wyscout`, `opta`, `kloppy`). Surfaces provider-
  native event fields alongside the canonical SPADL output as extra columns
  on the returned DataFrame — useful for surfacing fields that the canonical
  SPADL schema doesn't carry (e.g. StatsBomb's native `possession` sequence
  number, `possession_team`, `play_pattern`; Wyscout bronze passthroughs;
  Opta competition metadata). Each `preserve_native` field must be present
  on the input and must not overlap with the SPADL schema; both conditions
  raise `ValueError` early. Synthetic actions inserted by `_add_dribbles`
  get NaN in preserved columns (no source event to inherit from).
- `extra_columns` parameter on internal `silly_kicks.spadl.utils._finalize_output()`
  that powers the public `preserve_native` feature.
- `_validate_preserve_native()` helper in `silly_kicks.spadl.utils` for
  shared upfront validation across providers (input-column presence +
  schema-overlap check).
- Kloppy `preserve_native` requires kloppy >= 3.15 with raw-event
  preservation. Each preserved field is read from `event.raw_event[field]`.

## [1.0.0] — 2026-04-07

### Added
- DEBUG logging for kloppy silent event drops (aerial duels, unrecognized GK subtypes)
- `.github/CODEOWNERS` for code owner review enforcement

### Fixed
- StatsBomb converter now accepts both `"goalkeeper"` and `"goal_keeper"` keys in the
  extra dict — adapters that snake-case the event type name no longer silently lose all
  keeper actions

### Improved
- `ConversionReport` docstring: full Attributes section, usage example, provider-specific
  key type note
- `add_names()` docstring: explicit guarantee that caller-added columns are preserved
- `_finalize_output()` docstring: guarantee that all SPADL_COLUMNS are present
- `config.py` docstring: `actiontype_id`, `result_id`, `bodypart_id` reverse dicts documented
- Wyscout `convert_to_actions()`: Returns section now documents `ConversionReport`;
  `goalkeeper_ids` notes `None` ≡ empty set equivalence

### Removed
- `docs/plans/` and `docs/specs/` — internal development artifacts with local paths

### Changed
- Version bump: 0.1.0 → 1.0.0 (Production/Stable)
- C4 diagram genericized (removed project-specific references)

## [0.1.0] — 2026-04-06

### Added
- Initial release as maintained successor to socceraction v1.5.3
- SPADL converters: StatsBomb, Opta, Wyscout, Kloppy
- VAEP and Atomic-VAEP frameworks
- HybridVAEP — result-leakage-free action valuation
- xG-targeted labels via `xg_column` parameter
- Expected Saves (xS) label via `save_from_shot()`
- Expected Claims (xC) label via `claim_from_cross()`
- Cross zone feature (Gelade 2017 four-zone classification)
- Assist type feature (through ball, cutback, cross, set piece, progressive pass)
- Wyscout `goalkeeper_ids` parameter for GK aerial duel routing (#37)
- `ConversionReport` audit trail for every conversion
- `validate_spadl()` utility for DataFrame validation
- Input validation with clear error messages per provider
- "Nothing Left Behind" mapping registries (mapped/excluded/unrecognized events)
- Reproducible training via `random_state` parameter

### Changed (from socceraction v1.5.3)
- Dropped pandera dependency — schemas are plain Python constants
- Dropped multimethod dependency
- Removed numpy<2.0 upper bound
- All converters return `tuple[pd.DataFrame, ConversionReport]`
- All `apply(axis=1)` hot paths replaced with `np.select` vectorization
- Wyscout module decomposed into 3 files
- Gamestates uses vectorized shift instead of `groupby().apply()`
- Config DataFrame factories cached with `@functools.cache`
- Labels vectorized (shift-based accumulation replaces 27-column loop)
- `actiontype_result_onehot` uses numpy broadcasting

### Fixed
- Bug #507: Empty game crash in `gamestates()`
- Bug #950: `actiontype` feature wrong for Atomic-SPADL
- Bug #784: Opta converter silently drops card events
- Bug #831: Atomic-SPADL missing "out" for blocked/saved shots
- Bug #37/D44: Wyscout keeper_claim/punch differentiation
- Bug #946: pandas 3.0 `fillna(inplace=True)` deprecation
- pandas 3.0 `groupby().apply(as_index=False)` key column drop
