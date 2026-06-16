# TF-48 — Post-shot goalmouth crossing geometry (`add_shot_goalmouth`) — design spec

## Executive summary (for review)

The lakehouse GK Analytics redesign needs **PSxG (post-shot xG)** for the tracking providers
(Gradient Sports WC2022 ×64, SkillCorner ×10, IDSSE ×7). PSxG requires the goalmouth crossing
coordinates (y, z) of on-target shots — which the tracking providers' *event* feeds do not carry, but
their *ball trajectory* can recover. This spec defines a new silly-kicks tracking feature,
**TF-48 `add_shot_goalmouth`**: for each shot action, fit the post-contact ball trajectory from
tracking frames and emit the goal-plane crossing point (y, z), shot kinematics (speed,
time-to-goal-line), a derived on-target classification, and provenance/confidence columns in the
established ADR-025 style.

**Pure geometry, no model.** silly-kicks emits SPADL meters; the lakehouse maps them into its
StatsBomb-trained PSxG feature space (Butcher et al. 2025 xGOT logistic) and scores with the existing
model. No new trained artifact, no ADR-011 lifecycle, no weights.

**The hard part is the fit window**, not the fit: the trajectory must be fit only on the segment
between shot contact and the first subsequent contact (save / deflection / block changes the
trajectory — a deflected ball's observed crossing is not the shot's crossing). §6 specifies a
residual/velocity-break detector with an explicit end-reason column so nothing is laundered.

**Validation gold mine (the acceptance test):** the GS WC2022 corpus is the *same 64 matches* as
StatsBomb open data. Derived (y, z) from GS tracking is validated directly against StatsBomb's
hand-coded `end_location` on matched on-target shots (owner-gated e2e, DGX; `statsbombpy`
optional-dep precedent already exists in `tests/test_xthreat_statsbomb_e2e.py`).

**Owner-decided scope (2026-06-10):** NO VAEP `*_xfns` factory — crossing coordinates and kinematics
are **post-contact outcome descriptors**; feeding them to VAEP as features of the shot action leaks
the shot's result (the exact leakage class HybridVAEP exists to remove). Atomic mirror IS in scope.

---

**Date:** 2026-06-10 · **Author:** Karsten (with Claude) · **Status:** Draft — awaiting cross-session
review (lakehouse d32 session). **Origin:** lakehouse handoff "PSxG for tracking providers"
(GK Analytics page redesign investigation, 2026-06-10). **Not blocking** lakehouse GK page v1
(which ships geometry-based shot stopping from existing `pre_shot_gk_*` AC columns); this is the
fast-follow that unlocks Goals Prevented.
**ADR:** a new ADR (number reconciled against `origin/main` at PR time; expected ADR-030).

---

## 1. Verified facts (codebase + data probes, 2026-06-10)

Every anchor below was verified against source this session.

| Fact | Evidence |
|---|---|
| Ball `z` is in the frames schema | `TRACKING_FRAMES_COLUMNS["z"]`, documented constraint (0, 10.0 m) — `silly_kicks/tracking/schema.py:22,61` |
| z reaches frames: GS + sportec native adapters | both `EXPECTED_INPUT_COLUMNS` include `"z"` (`tracking/gradientsports.py`, `tracking/sportec.py`) |
| z reaches frames: SkillCorner via kloppy gateway | gateway plumbs `frame.ball_coordinates.z` → ball rows' `z` (`tracking/kloppy.py:160-175`); player rows z=NaN. kloppy 3.18's SkillCorner deserializer builds `Point3D(z=float(z))` from raw `frame_record.get("z")`, **preserving z=0.0** (`kloppy/infra/serializers/tracking/skillcorner.py:122-128`). Caveat: kloppy's separate `_raw_coordinates_to_point` helper uses a falsy `x and y and z` chain that drops `z==0.0` — the tracking-frame path does NOT go through it, but the implementation must include a DGX probe on the 10 pining matches confirming (a) SkillCorner raw frames carry `z` at all and (b) z survives to silly-kicks frames |
| Real GS ball z quality | probe, match 10502, first 20k frames: 14,109 ball records, **z present 100%** (keys `visibility/x/y/z`). Range **−0.76…12.42 m**, median 0.12 — z is noisy: small NEGATIVE values occur and the max EXCEEDS the schema's documented (0,10) constraint. The fit must tolerate out-of-constraint z (§8) |
| GS raw feed ALSO ships a smoothed ball track | probe: frame-level `ballsSmoothed` dict `{visibility, x, y, z}` (plus `*PlayersSmoothed`) — the lakehouse bronze `z_smoothed` derives from it. The silly-kicks adapter consumes whatever ball rows the caller flattens in (raw vs smoothed is the caller's ingestion choice); the §10 pilot compares raw-z vs smoothed-z fits (free lever, no new feature) |
| GS WC2022 carries NO shootout data | WC2022 had FOUR shootouts (10506 Japan–Croatia, 10508 Morocco–Spain, 10510 Croatia–Brazil, 10517 Argentina–France). Probed all four event files: periods 1–4 only, zero period-5 events. Probed 10517 full tracking: frame periods {1: 94832, 2: 96811, 3: 28991, 4: 34548}, zero period-5 frames. The FEED omits shootouts — the earlier "0 matches reached PSO" sweep claim was about the feed, not the tournament (cross-session review H1) |
| StatsBomb y-handedness is inverted vs SPADL | our own converter: `spadl_y = 68 − sb_y·68/80` (`silly_kicks/spadl/statsbomb.py:418`) — drives the §7 transform sign + the §10 in-test handedness settlement (cross-session review H5) |
| `@nan_safe_enrichment` is marker-only | `_nan_safety.py:18` sets `fn._nan_safe = True` and nothing else — the decorator registers the helper for the ADR-003 fuzz gate; the NaN-identifier behavior itself must be IMPLEMENTED in `add_shot_goalmouth` (cross-session review M1) |
| Linkage + windowing primitives exist | `link_actions_to_frames` (`tracking/utils.py:183`) + `slice_around_event(actions, frames, pre_seconds, post_seconds)` (`tracking/utils.py:745`) — period-bounded, long-form, carries all frame columns incl. `z`, joins `time_offset_seconds`. Action + frame `time_seconds` are both period-relative (ADR-017) |
| Orientation | The handoff's "frames keep per-period orientation" is **wrong, favorably**: adapters per-period flip so frames are **home-attacks-LTR** (`tracking/direction.py` `compute_attacking_direction` docstring). SPADL actions are per-action LTR. Reconciliation precedent: xS resolves each (game, period, team)'s **defended goal end from mean GK x** with outfield-mean fallback (`_defended_goal_x`, `tracking/_xshot_occurrence.py:553`) |
| Frame rates | per-row `frame_rate` schema column (GS 29.97, IDSSE 25, SkillCorner 10 fps); never hardcoded |
| Shot domain | standard: `_STANDARD_SHOT_TYPE_IDS = {shot, shot_freekick, shot_penalty}` (`tracking/features.py:73`). Atomic: `shot_freekick` is remapped to the atomic `freekick` type at conversion (`atomic/spadl/base.py:274-278`), so the atomic domain is `_ATOMIC_SHOT_TYPE_IDS = {shot, shot_penalty}` (`atomic/tracking/features.py:52`) — existing pre-shot-GK precedent |
| Provenance precedent | ADR-025 `resolve_restart_geometry` source/confidence columns + `RestartCoordinateReport` (`tracking/_restart_report.py`); `XtGkReport` aggregate pattern |
| Attribution | Anzer & Bauer (2021) (xGOT lineage) already in NOTICE via `pre_shot_gk_*`; Spearman et al. (2017) drag model already in NOTICE + implemented (`_cover_shadows.ball_drag_time`) if a drag-aware fit is ever needed |

NaN-coordinate prevalence is irrelevant here (we read frames, not event coords); the relevant
denominator is **shots with sufficient post-contact ball frames** (§10 reports it per provider).

## 2. Scope & division of labor

**silly-kicks (this TF):** pure trajectory geometry over frames, provider-agnostic, no model, no
provider event semantics beyond SPADL shot action types. New `tracking/_shot_goalmouth.py` +
public surfaces (§4), atomic mirror, tests, owner-gated SB validation e2e, ADR.

**lakehouse (tracked there, NOT in this repo):**
- SkillCorner `z` plumbing fix in `src/analytics/action_context/convert.py:463`
  (`_bronze_skillcorner_to_frames` currently sets `frames["z"] = np.nan` despite bronze carrying
  `ball_z`) — prerequisite for SkillCorner coverage, one-line, flagged in the handoff.
- The meters→StatsBomb-normalized feature transform (§7) + scoring with the existing PSxG model
  (their ADR-013 inference-mart pattern); AC columns + migration; GK page consumption.
- Policy for `shot_crossing_z` NaN rows (z-less provider data): score-with-imputed-z vs exclude —
  consumer decision, not ours.
- **Conditional on the §10.4 raw-vs-smoothed pilot outcome:** the lakehouse AC GS adapter currently
  feeds RAW `z` (`z_smoothed` is in bronze but consumed nowhere in the AC path — verified
  lakehouse-side, review round 2). If the pilot concludes smoothed-z fits better, the adapter must
  switch its `z` source (tracked lakehouse-side) — otherwise validation would run on smoothed-z
  while production runs on raw-z, a silent data-layer train/serve-style mismatch. The pilot
  conclusion is explicitly communicated cross-session either way.
- Verify SB WC2022 is ingested lakehouse-side if they want their own cross-check; the silly-kicks
  acceptance e2e pulls StatsBomb open data directly via `statsbombpy` and does NOT depend on
  lakehouse ingestion.

**Out of scope:** GK offset relative to the trajectory (already covered by `pre_shot_gk_*`); any
trained tracking-native PSxG model (possible later upgrade following the `gk_completion`
bundled-weights pattern); Phase-2-style canonical mutation (nothing canonical is touched — all
columns are new and additive); VAEP xfns (owner-decided, §4).

## 3. Naming & accounting

- **TF-48** (TF-46 corner-roles and TF-47 conditional-xT are taken in TODO.md).
- New action-coupled aggregator → **C4 count 27→28** (final-review C4 diagram update).
- Version: next minor at PR time (4.23.0 if nothing ships first).

## 4. Public API surface

New kernel module `silly_kicks/tracking/_shot_goalmouth.py`; public exports re-exported per house
pattern (`tracking/__init__.py` + `tracking/features.py` namespace conventions):

```python
@dataclass(frozen=True)
class ShotGoalmouthParams:
    post_window_seconds: float = 2.0      # max post-contact window (period-bounded by the slice)
    min_fit_frames: int = 3               # below -> insufficient_frames
    break_residual_m: float = 0.75        # horizontal residual jump that ends the fit segment
    break_speed_drop_frac: float = 0.5    # fractional horizontal-speed drop that ends the segment
    max_time_to_plane_seconds: float = 3.0  # fitted crossing later than this -> no_crossing
    rolling_z_max_m: float = 0.3          # ground band: "rolling" classification AND the bounce
                                          # detector's z-at-flip ceiling (§6 — ONE band, deliberately
                                          # shared; a bounce is by definition a near-ground event)
    bounce_min_dz_m: float = 0.25         # hysteresis: min |Δz| swing around a vz sign flip for it
                                          # to count as a bounce (§6 — rejects z-noise flips)
    on_target_tolerance_m: float = 0.11   # ball radius; widens the mouth/bar for the derived flag
    contact_refinement: bool = True       # refine t0 to the first shot-consistent discontinuity (§6)
    # __post_init__ validates positivity / sanity (house __post_init__-validator pattern)

def compute_shot_goalmouth(
    actions, frames, *, links=None, params: ShotGoalmouthParams | None = None
) -> pd.DataFrame:
    """PURE engine: index-aligned output frame, never mutates actions, no warnings
    (ADR-025 engine/edge split)."""

@nan_safe_enrichment
def add_shot_goalmouth(
    actions, frames, *, links=None, params: ShotGoalmouthParams | None = None
) -> pd.DataFrame:
    """actions + new columns (NaN out-of-scope rows); idempotent linkage-provenance
    merge skip; edge policy (warnings on anomalies) lives here, not in the engine."""

@dataclass(frozen=True)
class ShotGoalmouthReport:
    n_shots: int
    source_counts: dict[str, int]
    end_reason_counts: dict[str, int]
    z_profile_counts: dict[str, int]  # observability parity with the other two taxonomies;
                                      # the cheapest corpus-scale detector of bounce misclassification
    n_on_target_derived: int
    # .from_frame(df) classmethod mirroring RestartCoordinateReport
```

Per-Series feature wrappers (`shot_crossing_y(actions, frames)` etc.) follow the
`pre_shot_gk_*` per-Series pattern so the columns are individually consumable; the aggregator is
the primary surface.

**No VAEP `*_xfns` factory** [owner-decided 2026-06-10]. Rationale recorded in the ADR: every
output here is determined by what happened *after* ball contact — as a VAEP gamestate feature it
encodes the shot's outcome (a hard-hit upper-corner crossing implies on-target implies high
P(goal)), which is result leakage of the HybridVAEP class. A guard test asserts none of these
columns/functions appear in any default xfn list (`xfns_default*`, `tracking_default_xfns`,
`pre_shot_gk_full_default_xfns`, atomic mirrors).

**Atomic mirror** [owner-decided 2026-06-10]: `atomic.tracking.features` mirror of
`add_shot_goalmouth` + per-Series wrappers. NO coordinate synthesis is needed (unlike the xt_gk
mirror): the engine consumes only `action_id`, `game_id`, `period_id`, `time_seconds`, `team_id`,
`type_id` — never the action's coordinates (the trajectory comes from frames, the goal end from the
GK map). The mirror is a thin delegation with `shot_type_ids=_ATOMIC_SHOT_TYPE_IDS`. Documented
caveat: direct-freekick shots are `freekick` atoms in atomic space and are intentionally absent
from the atomic domain (existing pre-shot-GK precedent).

`links=` kwarg per convention; note the linked anchor frame is only the window *anchor* — the
post-contact samples come from `slice_around_event`, which is link-independent. Pre-linking still
saves the anchor-resolution cost and keeps signature parity.

## 5. Orientation & attacked-goal resolution

Everything is computed in **frame coordinates** (home-attacks-LTR), then canonicalized at output:

1. Build the (game_id, period_id, team_id) → defended-goal-x map from frames exactly as xS does
   (mean GK x per group, outfield-mean fallback) — **extract `_defended_goal_x` from
   `_xshot_occurrence.py` into `tracking/_gk_resolve.py`** (the existing frame-based GK-resolution
   module — the natural home) rather than duplicating; xS re-imports it (pure refactor,
   byte-identical, National-Park). The extraction KEEPS the helper's `.astype(bool)` on
   `is_ball`/`is_goalkeeper` as-is — those are schema-real bools (ADR-019's `_truthy_bool` concern
   targets object columns); "fixing" it would break the byte-identical claim. Implementer + reviewer
   note, on purpose.
2. Attacked goal for a shot by team T in (game, period) = the goal defended by the OTHER team in
   that group (the map has exactly two team entries per (game, period); a malformed group →
   `unresolved`). Team-id comparisons route through `_id_compat` (ADR-019).
3. The goal plane is `x = goal_x ∈ {0, 105}` in frame coords. Fit + intersect there (§6).
4. **Canonical output orientation: attacked goal at x=105.** When the attacked goal is at x=0,
   apply the full point reflection `x→105−x, y→68−y` (z, speeds, times invariant; the y-flip is
   required to preserve handedness — documented once here, never re-derived). Output
   `shot_crossing_y` is therefore directly comparable across shots, goal mouth at y ∈
   [30.34, 37.66], centre 34.
5. PSO (period 5): `team_attacking_direction` is undefined there, but the GK-position map still
   resolves the defended end (only the defending GK stands on a line). If the map is degenerate
   (both teams' GKs at the same end — likely in PSO since all kicks target one goal and the idle GK
   waits nearby), fall back to: the end nearer the mean ball position of the period's shot windows;
   if still ambiguous → `unresolved`. **Probe-backed status (2026-06-10):** WC2022 had four
   shootouts, but the GS feed carries NEITHER period-5 events NOR period-5 frames (all four
   matches' events + the final's full tracking probed — §1), and no other current corpus has PSO
   either. The PSO path therefore cannot be exercised on real data today: it ships as the
   documented fallback above, pinned by a synthetic fixture, and is promoted to real-data
   validation if a PSO-carrying corpus ever lands. Correctness over coverage.

## 6. Trajectory fit & window-end detection

**Window assembly.** Ball rows only (`is_ball` coerced via the `_truthy_bool` pattern — never
`.astype(bool)` an object column, ADR-019), `slice_around_event(shot_rows, ball_frames,
pre_seconds=0.3, post_seconds=params.post_window_seconds)`. The small pre-window exists only for
contact refinement.

**Contact refinement (`contact_refinement=True`).** Event clocks lag/lead frames; the first frames
after the nominal `time_seconds` may be pre-contact. Refine t₀ to the **FIRST qualifying**
ball-kinematic discontinuity within [t_event − 0.3 s, t_event + 0.3 s], where *qualifying* means
shot-consistent: a speed INCREASE whose post-discontinuity horizontal velocity points toward the
attacked goal. (Largest-discontinuity selection is explicitly rejected: for a close-range shot
saved within the window, the save IS the largest discontinuity, and refining onto it would fit the
post-save trajectory — cross-session review H3; fixture mandatory.) If no qualifying discontinuity
clears the noise floor, keep t_event. This is deliberately local and cheap — full event-to-frame
alignment (TF-43 ELASTIC, `align_events_to_frames`) is the documented upgrade path if validation
shows residual sync bias. Frames at t < t₀ are excluded from the fit.

**Fit-segment end = FIRST of:**
- **(a) plane crossing observed** — consecutive samples straddle the goal plane → crossing
  interpolated directly between them (source `observed`); the fit still runs for kinematics.
- **(b) trajectory break** — incremental refit: grow the segment frame by frame; end it when the
  newest sample's horizontal residual vs the current fit exceeds `break_residual_m`, OR horizontal
  speed drops by more than `break_speed_drop_frac` between consecutive samples, OR the horizontal
  velocity direction reverses. This catches saves, deflections, blocks, and bounces off post.
- **(c) window cap / data end** — `post_window_seconds` reached, period slice exhausted, or ball
  samples gap out (occlusion).

Recorded per row in `shot_fit_end_reason ∈ {plane_crossed, trajectory_break, window_cap,
data_end}`. A blocked/deflected shot (reason `trajectory_break`) still gets an `extrapolated`
crossing from its pre-break segment when that segment has ≥ `min_fit_frames` — that *is* the shot's
intended crossing; the lakehouse filters by on-target/source as it sees fit. Nothing is silently
dropped: every shot row gets a source.

**Fit model (Phase-1, deliberately minimal):**
- x(t), y(t): constant-velocity least squares on the segment. Drag is real at 30 m/s but flight
  times here are short (typically ≤ 0.6 s); the SB validation (§10) arbitrates whether a drag-aware
  upgrade (Spearman 2017 constants, already in-house) is needed — do not pre-build it.
- z(t): governed by a **z-profile classification** of the fit segment (cross-session review H2 —
  the three horizontal break detectors cannot see a ground bounce, and a fixed-g ballistic fit
  across a z-kink produces a nonsense crossing z). Recorded in `shot_z_profile`:
  - **`rolling`** — all segment z below `rolling_z_max_m`: crossing z = the segment's mean z (≈0,
    ground-level daisy-cutter). No ballistic fit; avoids fragile bounce-fitting for multi-bounce
    rollers.
  - **`airborne`** — no bounce detected: ballistic fit with FIXED gravity curvature,
    `z(t) = z₀ + v_z·t − ½·9.81·t²`, two free parameters — robust at the 2–4 usable samples
    SkillCorner's 10 fps yields close-range, where a free quadratic overfits.
  - **`bounced`** — detector (M-2, noise-hardened against the measured GS z noise §1): a
    finite-difference vz sign flip counts as a bounce ONLY when (i) z at the flip ≤
    `rolling_z_max_m` (a bounce is a near-ground event; the band is deliberately shared with the
    rolling classification — one ground concept, one param) AND (ii) the |Δz| swing around the flip
    ≥ `bounce_min_dz_m` (hysteresis). A noisy airborne trajectory whose finite-difference vz flips
    sign at height MUST stay `airborne` (fixture-pinned). The crossing comes from the trajectory
    the GK actually faces, i.e. the LATEST sub-segment before the plane — three branches:
    post-bounce sub-segment ≥ `min_fit_frames` samples → refit x/y AND z on it (it produces the
    crossing); ≥ 2 samples → refit z only (ballistic is 2-parameter) and keep the full-segment x/y
    fit; < 2 → `shot_crossing_z` NaN while the horizontal crossing y stands. **Per-column segment
    provenance (M-1)** — which segment feeds which output is fixed, not "all": see §7. Multiple
    bounces recurse to the latest sub-segment; a segment that has degenerated to rolling is
    classified `rolling`.
  - Negative fitted z at the plane clamps to 0 (ground).
- Minimum `min_fit_frames` samples (default 3) → else source `insufficient_frames`, all outputs NaN.
- z-less samples (provider z all-NaN): the 2D fit still runs; `shot_z_profile` is NA and
  `shot_crossing_z` / `shot_speed` (3D) degrade as documented in §7.

**Plane intersection.** Solve `x(t*) = goal_plane_x` from the x fit. `t* ≤ 0`, `vx` pointing away
from the plane, or `t* > max_time_to_plane_seconds` → source `no_crossing` (mishits, dribbles
mis-typed as shots), outputs NaN. Else evaluate y(t*), z(t*) → crossing; source `extrapolated`
(or `observed` per (a), where the interpolated crossing wins and the fit only supplies kinematics).

## 7. Output contract

All columns NEW + additive; canonical `start_*/end_*` never touched; SPADL meters; canonical
attacked-goal-at-x=105 orientation (§5.4). Non-shot rows: all NaN (xfns-discipline). dtypes float64
unless noted.

| Column | Meaning |
|---|---|
| `shot_crossing_y` | y at the goal plane, m (mouth = 30.34–37.66) |
| `shot_crossing_z` | z at the goal plane, m (bar = 2.44). NaN when input z is unavailable/insufficient |
| `shot_speed` | **fitted initial speed at t₀, ALWAYS from the EARLIEST (contact) sub-segment** — it answers "how hard was the shot hit" and is never superseded by a post-bounce refit (M-1). 3D `√(vx²+vy²+vz(t₀)²)`, m/s (horizontal components are that sub-segment's constant-velocity fit, i.e. segment-average; under drag this UNDERESTIMATES true contact speed — documented in the docstring, the column does not promise more than the fit delivers). Falls back to 2D horizontal speed when z unavailable (and `shot_crossing_z` is NaN, so the degradation is visible) |
| `shot_time_to_goal_line` | **elapsed real time** from (refined) contact t₀ to the crossing time of the segment that produced the crossing — for a bounced shot this SPANS the bounce (M-1) |
| `shot_on_target_derived` | nullable boolean: crossing within posts+bar expanded by `on_target_tolerance_m`. NA unless source ∈ {observed, extrapolated}. Post/bar physical width is INTENTIONALLY folded into the tolerance rather than modeled — a deliberate decision, not an omission; do not "improve" it without a new decision. Provider `result_id` is a *sanity cross-check* in validation (§10), never an input |
| `shot_z_profile` (object) | `airborne` / `rolling` / `bounced` (§6); NA when no z data or no fit |
| `shot_crossing_source` (object) | `observed` / `extrapolated` / `insufficient_frames` / `no_crossing` / `no_ball_frames` (no ball samples in the window — the window is time-sliced, link-independent, so there is no separate "link failed" case) / `unresolved` (goal-end resolution failed) |
| `shot_crossing_confidence` | continuous [0,1]; provisional map (ADR-025 style): observed 1.0; extrapolated = f(n_fit_frames, fit RMSE, extrapolated-flight fraction, **z_profile + z-sub-segment sample count**) — the z inputs exist because a 2-sample z-only ballistic refit is exactly determined (zero residual df, RMSE ≡ 0) and would otherwise out-score an honest 5-point fit (L-1); exact formula calibrated at the validation pilot; all other sources 0.0 |
| `shot_fit_n_frames` (Int64) | samples in **the segment that PRODUCED the crossing** (post-bounce when superseded; M-1). For `no_crossing` rows a fit DID run: populated from the attempted segment (NA is reserved for truly-unfitted sources — plan-review R4b) |
| `shot_fit_rmse` | horizontal residual RMSE of **the segment that produced the crossing**, m (M-1; `no_crossing` rows carry the attempted segment's RMSE per the same convention) |
| `shot_fit_end_reason` (object) | `plane_crossed` / `trajectory_break` / `window_cap` / `data_end`; NA when no fit ran |
| + the 4 standard linkage-provenance columns | `frame_id`, `time_offset_seconds`, `n_candidate_frames`, `link_quality_score` — idempotent skip when present |

**Units contract / SB transform (documented once, owned by the lakehouse at scoring time):**
StatsBomb goalmouth space is 120×80 **yards**; the lakehouse-confirmed PSxG normalization is
`_GOAL_Y_MIN=36.0`, `_GOAL_Y_MAX=44.0`, `_GOAL_Z_MAX=8.0`, `y_norm=(y−36)/8`, `z_norm=z/8`
(lakehouse `analytics/goalkeeper.py:277-293`, confirmed cross-session 2026-06-10). The SB goal
mouth (y 36–44) spans 8 yards = 7.3152 m ≈ the physical 7.32 m mouth, so the mapping is a plain
meters→yards conversion about the goal centre — **with a y-handedness FLIP**: StatsBomb y increases
in the OPPOSITE sense to SPADL y (verified in-repo: our own SB converter inverts,
`spadl_y = 68 − sb_y·68/80`, `silly_kicks/spadl/statsbomb.py:418`). Indicative:
`y_sb = 40 − (shot_crossing_y − 34) / 0.9144` and `z_sb = shot_crossing_z / 0.9144`
(SB bar ≈ 2.44 / 0.9144 ≈ 2.67 yd). The §10 validation e2e instruments this transform and therefore
cannot defer handedness: it must settle it empirically inside the test (goals' crossing side vs SB
far/near-post coding) before any floor is evaluated — a mirrored transform would inflate Δy and
fail floors mysteriously. The lakehouse owns the scoring-time mapping; silly-kicks guarantees only
physical SPADL meters with the §5.4 orientation (normative statement: "TF-48 emits physical meters,
goal centre y=34, mouth 7.32 m, bar 2.44 m").

## 8. Robustness & edge cases (spec'd, each gets a fixture)

- **Noisy/out-of-constraint z** (measured: −0.76…12.42 m on real GS): fit is least-squares (no
  pre-clamping of inputs); fitted crossing z < 0 clamps to 0; z > 10 at the plane is reported as-is
  (a real lob can cross high) — the schema constraint documents typical range, it is not an
  invariant the feature may assume.
- **Duplicate (period, frame_id) records** (GS ships up to 16×): dedup keep-first on ball samples
  before fitting (memory: GS duplicate-frames trap).
- **Immediate deflection** (break within < `min_fit_frames`): `insufficient_frames`, NaN — honest
  over heroic.
- **Looping shot over the bar**: ballistic z handles it; `shot_on_target_derived=False` with valid
  crossing y/z (crossing z above bar).
- **Away-team shot / period-2 flip**: covered by §5 (orientation fixtures mandatory).
- **Ball samples missing entirely in the window** (occlusion / dead feed): `no_ball_frames`.
- **Mis-typed shots that never travel goalward**: `no_crossing`.
- **Own goals**: the trajectory points at the actor's OWN goal → vx points away from the attacked
  plane → `no_crossing`. Correct, but by construction rather than by accident: the exclusion is
  INTENTIONAL (PSxG-faced-by-the-opposing-GK semantics do not apply to own goals), recorded in the
  ADR and pinned by a fixture.
- **Caller orientation convention**: home-attacks-LTR is what silly-kicks adapters emit, but
  external callers (e.g. the lakehouse builds SkillCorner/Metrica frames itself, unflipped,
  `team_attacking_direction=None`) may pass other global conventions. **Documented invariant: the
  engine assumes NOTHING about global orientation** — goal ends come from the GK map (§5.1), the
  output is canonicalized (§5.4), and the engine never reads `team_attacking_direction`. The
  consumed-frame-columns contract is exactly: `game_id`, `period_id`, `frame_id`, `time_seconds`,
  `team_id`, `is_ball`, `is_goalkeeper`, `x`, `y`, `z`. Hardened by an orientation-invariance test:
  the same synthetic match in both conventions → byte-identical canonical outputs.
- **Mixed id dtypes** (Int64 vs object): all comparisons via `_id_compat`; the auto-enumerating
  id-dtype-invariance gate covers the new `add_*` automatically.
- **NaN identifiers** (`team_id` NaN on a shot row): NaN outputs, never a crash — IMPLEMENTED in
  `add_shot_goalmouth`'s row handling (the `@nan_safe_enrichment` decorator is marker-only, §1; it
  registers the helper for the ADR-003 fuzz gate, it does not provide the behavior).

## 9. Performance

~25–30 shots/match × a ≤ 2.3 s ball-only slice × a tiny least-squares — negligible next to linkage.
No numba, no cache object. If a structural perf guard is added, the spy target is
`slice_around_event` call count (≤ 1 per `add_shot_goalmouth` call — the slice is batched for all
shots at once, NOT per-shot).

## 10. Validation & acceptance (owner-gated, DGX)

**Acceptance test = GS↔StatsBomb WC2022 cross-validation.** Same 64 matches in both sources.

Protocol:
1. Match shots GS↔SB per match by (period, game-clock proximity, team, scoreline context); the
   tie-breaker ORDERING is spelled out in the matching script's docstring; ambiguous matches land
   in the unmatched report, never best-effort matched; unmatched shots reported, never silently
   dropped.
2. For matched **on-target** shots (SB on-target outcome vocabulary — {Goal, Saved, Saved to
   Post}, the last LIVE-VERIFIED at the 2026-06-11 pilot where the spec-text guess "Saved To
   Post" tripped the runtime guard exactly as designed; the matching script asserts its outcome
   literals against the actual statsbombpy vocabulary at runtime and FAILS LOUD on a
   casing/zero-match drift, never silently matching nothing) with 3D `end_location`, compare TF-48 (y, z) — converted to SB units by the test,
   instrumenting the §7 transform — against SB's hand-coded `end_location`. The test settles the
   y-handedness empirically FIRST (§7; goals' crossing side vs SB post coding) before any floor is
   evaluated.
3. **Stratify GOALS vs SAVES (cross-session review H4).** SB `end_location` semantics differ by
   outcome: a goal's end_location is a true plane crossing; a save's is (per SB semantics, to be
   confirmed by the pilot itself) the SAVE point — 1–3 m off the line for an off-line GK — which
   differs from TF-48's plane extrapolation BY CONSTRUCTION. Calibration + primary accept floors
   run on GOALS ONLY (unambiguous ground truth); saves are reported separately. If Δ(goals) ≪
   Δ(saves), the gap is definitional — evidence FOR the fit, not against it. The Δ(saves)−Δ(goals)
   split also quantifies, for free, the lakehouse's train/serve shift (their PSxG model trained on
   SB save-point semantics, served plane crossings) — recorded in the ADR for the lakehouse to
   consume.
4. **Two-stage per the validation-rigor policy:** a pilot subset (e.g. 16 matches) measures the
   error distribution and calibrates the confidence map (§7) + params defaults (§4) — including an
   explicit per-frame-rate sensitivity row (GS 29.97 / IDSSE 25 / SkillCorner 10 fps; esp.
   `break_residual_m` at 10 fps, ~3 m/frame at 30 m/s, where one noisy sample can end a minimum
   segment) and a raw-z vs smoothed-z (`ballsSmoothed`) fit comparison on GS (§1). The accept
   floors are then PRE-REGISTERED and evaluated on the held-out remaining matches. Provisional
   floor shape (numbers set at pilot review, recorded in the ADR before the held-out run): median
   |Δy| and median |Δz| ≤ X m on GOALS with source ∈ {observed, extrapolated}; on-target agreement
   (derived vs SB outcome) ≥ Y%; resolution coverage (source ∈ {observed, extrapolated} among
   matched on-target shots) ≥ Z%. SB hand-coded locations are themselves noisy — floors judge
   *usability for PSxG*, not frame-truth.
5. Per-provider coverage report (fraction of shots by source) for GS/IDSSE/SkillCorner via the
   pining corpora; SkillCorner kloppy-z empirical probe (§1 caveat) runs here.
6. Sanity cross-checks: goals (SPADL `result_id` success) must be overwhelmingly
   `shot_on_target_derived=True`; `shot_speed` distribution physically plausible (~10–35 m/s).

All heavy validation runs on the DGX against pining data (canonical-compute policy); nothing
downloaded locally; CI sees only synthetic fixtures + the committed-fixture tests.

## 11. Testing plan (TDD)

Red-first unit surface (synthetic trajectories with known closed-form crossings):
- straight drive, dipping ballistic, lob over bar, wide miss (`no_crossing`), observed crossing
  (straddling samples), deflection mid-flight (break detection), block (segment + extrapolation),
  **bounced shot — all THREE post-bounce branches** (≥ `min_fit_frames` → full x/y+z supersession;
  ≥ 2 → z-only refit; < 2 → crossing-z NaN, y stands), **bounced-shot column provenance**
  (`shot_speed` from the contact sub-segment, crossing/fit-quality from the producing segment,
  `shot_time_to_goal_line` spanning the bounce — §7 M-1 contract pinned), **noisy airborne
  trajectory** (finite-difference vz sign flips at height MUST stay `airborne` — M-2 hysteresis),
  **rolling daisy-cutter** (multi-bounce → `rolling`, crossing z ≈ 0), **close-range save inside
  the refinement window** (refinement must lock the shot, not the save), **own goal** (`no_crossing`,
  intentional), **orientation invariance** (same match, both global conventions → byte-identical
  canonical outputs), 2-frame close-range (`insufficient_frames`), occlusion gap, dup frames,
  away-team + period-2 orientation, PSO degenerate map (synthetic — no real PSO corpus exists, §5.5),
  z-all-NaN provider, NaN team_id, mixed dtypes.
- Spatial-fixture-coverage policy: fixtures enumerate the pathology × branch matrix above, not
  happy-path only; fixtures use realistic mixed dtypes (memory: fixture-dtype policy).
- Param `__post_init__` validation tests.
- Auto-gates that pick the surface up automatically (verify, don't assume): id-dtype invariance,
  nan-safety fuzz, public-API Examples, provenance-skip guard, no-default-xfns guard (new, §4).
- Atomic mirror parity test (standard vs atomic on the same synthetic match, shot/shot_penalty rows
  byte-identical; freekick-shot absence documented in the test).
- Owner-gated e2e: §10 (marked `e2e`; SB network test follows the `statsbombpy` importorskip
  pattern).

## 12. Documentation & bookkeeping

- ADR (expected ADR-030): post-shot trajectory geometry — window-end detection policy, z-profile
  (rolling/airborne/bounced) policy + latest-sub-segment-faced-by-the-GK rationale, fixed-g
  ballistic choice, contact-refinement first-qualifying rule, no-xfns leakage rationale,
  canonical-orientation contract + the engine's orientation-agnostic invariant, the intentional
  own-goal exclusion, the SB goals-vs-saves end_location semantics split + the lakehouse
  train/serve shift note (H4b), acceptance-floor pre-registration record.
- NOTICE: no new entry required (pure geometry; Anzer & Bauer 2021 cited contextually in
  docstrings; Spearman 2017 only if the drag upgrade ever lands).
- TODO.md: TF-48 row removed on ship (grooming policy); lakehouse-side items live in the lakehouse
  repo's tracker, referenced from the handoff.
- CLAUDE.md tracking section: one-line TF-48 summary on ship.
- C4: aggregator count 27→28.

## 13. Open items (post cross-session review, 2026-06-10)

Resolved by the lakehouse review round (recorded in §7/§10): SB normalization constants confirmed
(`y_norm=(y−36)/8`, `z_norm=z/8`); y-handedness settled in-repo (flip; empirically re-confirmed
inside the e2e before floors); lakehouse `convert.py:463` z fix tracked their side, ships in their
GK-page cycle. The lakehouse AC path does NOT go through kloppy (frames built from bronze in
`convert.py`) — our kloppy probe validates the silly-kicks gateway path, theirs validates the AC
path; both needed, neither substitutes.

Still open:
1. **Params defaults** (§4) provisional pending the pilot, with an explicit per-frame-rate
   sensitivity row (§10.4; esp. `break_residual_m` at SkillCorner 10 fps).
2. **Confidence map** (§7) provisional, ADR-025 style; calibrated at the pilot.
3. **SkillCorner raw-z existence** — unverified until the DGX pining probe (§1); if SkillCorner z
   turns out absent/unusable, SkillCorner degrades to crossing-y-only (PSxG then lakehouse-policy,
   §2).
4. **Contact refinement** (§6): first-qualifying-discontinuity default vs ELASTIC alignment
   upgrade — pilot decides whether the upgrade is needed.
5. **SB saved-shot end_location semantics** (§10.3): the goals-vs-saves stratification is in the
   protocol regardless; the pilot's Δ(saves)−Δ(goals) split confirms or refutes the save-point
   premise empirically.
