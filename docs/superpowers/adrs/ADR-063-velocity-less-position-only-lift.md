# ADR-063: Velocity-less-provider position-only lift — the zero-velocity model behind ONE edge seam

| Field | Value |
|---|---|
| **Date** | 2026-08-18 |
| **Status** | Accepted |
| **Deciders** | Karsten S. Nielsen (owner); drafted with Claude (Opus 4.8) |

## Context

On the real 30-match SB360 licensed corpus (enabled in 4.84.0, ADR-062), **40 of ~230 tracking
battery columns come out fully-NaN**, and the residual after the 4.84.0 team-id fix is
velocity-related. SB360 is a single freeze-frame per action: it DECLARES `speed_source ==
SPEED_SOURCE_UNAVAILABLE` on every row, because a lone snapshot has no per-player temporal history,
so `vx`/`vy` *cannot* exist (not "not yet"). Four velocity-requiring pitch-control aggregators —
`add_gk_influence`, `add_cover_shadows`, `add_player_influence`, `add_space_creation` — refused or
degraded on those frames, while `add_pitch_control` already computed a zero-velocity Spearman surface
(it zero-filled `vx`/`vy`). This ADR makes the four consistent with `add_pitch_control` — and does so
through a *principled* `speed_source`-aware seam, not `add_pitch_control`'s too-loose unconditional
zero-fill. It follows 4.84.0, is its own release, and is **additive on every velocity-bearing frame
(no retrain)**.

The organising principle:

> **Surface a value iff the zero-velocity computation is the honest MODEL output, not a
> systematically BIASED estimate of a physical fact wearing the same column name.**

## Decision framework — model-relative vs physical-quantity estimate (MEASURED, not asserted)

Classify what the zero-velocity computation PRODUCES:

- **Model-relative** — a dimensionless share or a model-integral with no velocity-independent
  physical referent (pitch-control share, off-ball xT, OBSO differential, blocking score). The
  zero-velocity value IS the honest model output — the classic Spearman reaction-time model, *weaker,
  not invented* (ADR-053) → **Tier 1, LIFT.**
- **Physical-quantity estimate** — an estimate of a physical fact (m², seconds) that exists
  regardless of observation (reachable area, closing time). `compute_tti` is strictly monotone in the
  along-target velocity component, so zeroing a moving player's velocity **systematically** understates
  area / shifts closing time → **Tier 2, SUPPRESS to NaN** (PREFERRED D1).
- **Velocity- or multi-frame-constitutive** — the quantity cannot exist without velocity or a time
  window → **Tier 3, honest-NaN.** A value would be fabrication.

**The boundary is measured (D2 below), not placed by armchair reasoning** (ADR-055 "measured, not
inferred"). The distinguishing axis is the DIMENSIONAL nature — not delta magnitude: a model-relative
column can be quite velocity-sensitive and still be Tier 1, because there is no "true" value it is
biased against.

## Decisions

**1. ONE method-aware, fail-fast edge helper.** `tracking._velocity_availability.zero_velocity_if_unavailable(frames, *, method="spearman")`:
- `vx`/`vy` PRESENT → returns `frames` unchanged (same object, no copy).
- ABSENT and DECLARED (`velocity_unavailable_by_design` — the marker on EVERY row) → a COPY with
  `vx`=`vy`=0.0 (the zero-velocity positional model).
- ABSENT, marker NOT set, `method` requires velocity → **RAISES `ValueError`** (a forgotten
  `derive_velocities()` is a caller BUG). It leaves the marker on the zero-fill copy (internally
  inconsistent if forwarded — a one-line caveat at the seam; fine for the immediate pitch-control
  call). The velocity-requiring method set is single-sourced from the dispatch.
- ABSENT, marker not set, velocity-FREE method (`voronoi`) → unchanged (voronoi needs no velocity).

**2. Policy at the edge; the dispatch stays a pure engine.** The helper is called at the compute_*
seams (`compute_gk_influence`, `compute_player_influence`, `compute_threat_pc`,
`compute_blocking_score`, `compute_space_created`) for direct callers, AND at the `_*_at_actions`
kernel EDGE (before the per-frame loop) for `gk_influence`/`player_influence` so the forgotten-velocity
raise precedes the loop's degrade-to-NaN handler and the declared zero-fill happens ONCE. It REPLACES
two ad-hoc unconditional zero-fill blocks (`pitch_control_at_target` at `features.py:2632-2638`;
`compute_space_created` at `_space_creation.py:179-184`) and one silent `return None`
(`_compute_cover_shadow_dict:1036`). `compute_pitch_control` is UNCHANGED — it stays the pure
computation that raises when it cannot compute (the same rule that puts the ghost clamp at the serving
seam and the `xt_gk` base-rate switch in `compute_xt_gk`, not in `predict_*`).

**3. FAIL-FAST on the caller bug — the ADR-043 discipline (best practice; scope/breaking accepted by
the owner).** A forgotten `derive_velocities()` RAISES uniformly across every public entry
(`pitch_control_at_target` + the four `add_*`), rather than degrading to a warn+NaN column that is
"indistinguishable downstream from legitimately-absent" (ADR-043's own words about DAS). All FOUR
aggregators call the helper at a kernel-EDGE BEFORE their per-frame/per-action loop, so the raise is
uniform (a fully-unlinkable match, where no action reaches compute, still raises rather than returning
all-NaN silently) and the declared zero-fill happens once: `add_gk_influence`/`add_player_influence`
NEED the edge because their per-frame loops catch `ValueError` (the edge precedes the catch), while
`add_cover_shadows`/`add_space_creation` gain it for uniformity + efficiency (their compute paths -- the
`_compute_cover_shadow_dict` `return None` replacement and the `compute_space_created` block replacement
-- also fail-fast for direct callers). `add_obso`/`add_pausa` fail-fast via the single
`_precompute_obso_lookup` seam (see the Scope note). The rejected alternative (the spec's "helper returns
unchanged on forgotten, dispatch raises, aggregators catch → warn+NaN") preserved the existing
graceful degradation but LEFT the silent-wrong-number path in `add_space_creation`'s ad-hoc zero-fill
and the all-NaN-hides-a-bug shape everywhere.

**4b. The `*_xfns` opt-in VAEP transformers honour the SAME contract.** `gk_influence_xfns`,
`cover_shadow_xfns`, `player_influence_xfns` and `space_creation_xfns` route through the same seam'd
`compute_*`, so Tier-1 already lifts for them — but their per-frame `except (ValueError, KeyError)`
would swallow the forgotten-velocity raise into warn+NaN, and the player transformer's own assembly
re-introduces the `actor_reachable_area_m2 = 0.0` GK-actor leak. Both are fixed to match the `add_*`
path: the edge helper is called at each transformer's top (after the `frames is None` introspection
guard) so a forgotten frame raises before the per-frame catch, and the player transformer replicates
the Tier-2 assembly suppression. `space_creation_xfns` needed no change — it delegates to
`add_space_creation` directly. The helper also returns an EMPTY frame set unchanged (never raises),
because VAEP `feature_column_names` introspects by calling frame-aware transformers with an empty
frames DataFrame. Still **no retrain**: on a velocity-bearing frame the transformer is byte-identical.

**4. Tier-2 SUPPRESSION emits NO per-row token (PREFERRED D1).** `VelocityRegimeDiagnosis`
(`schema.py:310-312`) states the velocity regime "is a property of the whole frame set rather than of
any row, which is why this is a diagnostic rather than a per-row provenance column that would carry a
constant." A per-family `velocity_unavailable` token that only ever says one thing is exactly that
rejected shape — `das_source` is a legitimate per-row column ONLY because it VARIES
(`computed`/`unlinked`/`unscoreable_frame`/`team_unresolved`). So the Tier-2 columns are set to NaN
and the signal is the existing frame-level `validate_velocity_regime`. **Because `compute_tti`
auto-fills `vx`/`vy`=0, the Tier-2 columns compute FOR FREE once the share path stops raising —
keeping them NaN is an ACTIVE suppression, not the absence of a lift.** For `player_influence` the
suppression is *also* applied at the aggregator assembly: `actor_reachable_area_m2` initialises to 0.0
and stays 0.0 for a GK actor (excluded from the per-player influence dict), so the per-player NaN does
not reach it.

## Tiers (exact columns)

**Tier 1 — LIFT (11 spatial-control columns + 2 ride-alongs):**

| Aggregator | Columns |
|---|---|
| `add_cover_shadows` | `blocked_threat_fraction`, `blocking_score`, `max_single_defender_blocking_score`, `max_single_defender_player_id`, `n_blocked_receivers`, `n_potential_receivers` |
| `add_space_creation` | `space_created_m2`, `space_denied_m2_opponent`, `obso_epv_source` |
| `add_gk_influence` | `gk_pitch_control_share_weighted` |
| `add_player_influence` | `off_ball_xt_team`, `off_ball_xt_opponent`, `off_ball_xt_diff` |

**Tier 2 — SUPPRESS to NaN (7 kinematic estimates):** `gk_reachable_area_m2`,
`gk_closing_time_mean_s__six_yard_box`, `gk_closing_time_min_s__six_yard_box`,
`actor_reachable_area_m2`, `reachable_area_team`, `reachable_area_opponent`, `reachable_area_diff`.

**Tier 3 — keep honest-NaN (untouched):** DAS (`das_*`), `add_actor_pre_window`/`add_off_ball_*`
windows, `actor_speed`, `press_commitment*`, `add_ghost_gk`, `add_xcross_attempt`,
`add_shot_goalmouth` — none is one of the four aggregators.

### Scope — the four lifted aggregators + `add_obso`/`add_pausa` (fail-fast only)

The four aggregators above own the 40 fully-NaN columns and get the full lift. **`add_obso`/`add_pausa`
were NEVER in that set** — they already zero-filled on declared frames — but they kept the SAME loose
unconditional zero-fill that silently accepted a forgotten `derive_velocities()`. They are extended to
fail-fast at TWO seams: (1) `_precompute_obso_lookup`, the shared aggregator/xfns seam that
`add_obso`, the per-Series `obso_actual`/`obso_peak`/`obso_optimal`, `obso_xfns`, and `add_pausa` →
`add_obso` → `pausa_xfns` all route through; and (2) the public low-level engine `compute_pass_obso`,
whose own per-frame velocity seam `_ensure_velocity_columns` is now marker-aware (it delegates to the
same `zero_velocity_if_unavailable`), so a DIRECT caller of that engine fails fast too instead of
silently zero-filling. Declared behaviour is **byte-identical** (both seams still zero-fill, so the
ADR-053 obso verdict is unchanged and there is no re-adjudication); a forgotten frame now RAISES. `add_pressure_on_actor(method="bekkers_pi")` is **deliberately
unmodified** — an opt-in pressure method (the default `andrienko_oval` is velocity-free, so it is not in
the default battery), and whether it should lift or stay honest-NaN on a declared frame is its own tier
decision, out of scope here.

## D2 — the measured boundary

On the velocity-BEARING audit fixture (`tests/sb360/_fixture.build_leg_b`), each candidate column was
computed with real `vx`/`vy` and with `vx`=`vy`=0; `rel Δ = max|zero − aware| / max|aware|`:

| Column | rel Δ | signed Δ (zero − aware) | tier |
|---|---|---|---|
| `gk_pitch_control_share_weighted` | 0.004 | +0.0003 | Tier 1 (near-invariant) |
| `off_ball_xt_team` / `_opponent` / `_diff` | 0.19 / 0.10 / 0.27 | −271 / +11 / −282 | Tier 1 (model-relative) |
| `blocking_score` / `blocked_threat_fraction` / `max_single_defender_blocking_score` | 0.17 / 0.10 / 0.78 | −5.4 / −0.005 / −0.14 | Tier 1 (model-relative) |
| `space_created_m2` / `space_denied_m2_opponent` | 0.17 / 0.12 | +0.70 / −0.08 | Tier 1 (model-relative) |
| `reachable_area_team` / `_opponent` / `_diff` | 0.50 / 1.0 / 0.33 | −5.5 / −5.5 / +0 | **Tier 2 (large, directional understatement)** |
| `gk_closing_time_min` / `_mean` | 0.08 / 0.06 | −0.05 / −0.03 | **Tier 2 (directional)** |
| `gk_reachable_area_m2` / `actor_reachable_area_m2` | 0 / 0 | +0 / +0 | Tier 2 by m²-reasoning (single-keeper/GK-actor fixture does not cross τ; the multi-player `reachable_area_team` exhibits the bias) |

**Conclusion: no column moves.** The physical-quantity columns (m²/s) show the predicted directional
bias; the model-relative columns vary without a physical referent.

## Retrain / breaking-change analysis

- **NO retrain.** The change touches ONLY frames missing `vx`/`vy`; the helper returns the SAME object
  on a velocity-bearing frame, so every trained-model input is byte-identical. Declared `"invariant on
  velocity-bearing frames"` in the ADR-045 idiom.
- **Breaking (intended), and the spec's "in-repo blast radius is ZERO" was WRONG.** Two in-repo tests
  passed velocity-less-UNDECLARED frames (raw provider frames carry `speed` but no `vx`/`vy`) and were
  passing *vacuously* on all-NaN output: `test_player_influence_aggregator.py` (9 tests;
  `test_diff_identity` filtered to an empty `notna()` mask) and
  `test_provenance_skip_guard.py::test_chained_enrichments...`. Both are fixed by deriving velocities
  in the fixture — the exact `smooth_frames` + `derive_velocities` step `test_cover_shadows.py`
  already uses — which makes them exercise a real computation. The lesson: verify blast-radius claims
  by EXECUTING the full suite, not by reasoning.
- **Cross-repo exposure.** `add_pitch_control` / `pitch_control_at_target` are PUBLIC and consumed by
  the lakehouse d32 repo; a downstream caller passing a velocity-less-but-undeclared frame now raises
  (intended — it catches the mistake). A Hyrum's-Law change worth stating.

## ADR-053 SB360 audit re-adjudication

The four aggregators' velocity-axis verdicts change **per column**: the model-relative Tier-1 columns
move to **`differs → differs_by_design`** (the `positional_pc` rationale — the three new members are
added to `PITCH_CONTROL_DERIVED`); the velocity-INVARIANT Tier-1 columns (`n_blocked_receivers`,
`n_potential_receivers`, `obso_epv_source`) read **`identical → works`**; the suppressed Tier-2 columns
read **`all_nan → honest_nan`** (the AUTO rule, deliberately withheld). `_regenerate.py` is not
idempotent (it genericised `add_packing`'s hand-written rationale in `_shape.py`, which was restored
from HEAD), so only `_gk.py` and `_space.py` — the four aggregators — change; every other entry
round-trips byte-identically.

## Rejected alternatives

- **Per-aggregator zero-fill** (copy `add_pitch_control`'s block into each) — duplicates the loose
  unconditional zero-fill five times and silences the exact mistake the dispatch guard catches.
- **Degrade-vs-raise policy in the `compute_pitch_control` dispatch** — the dispatch stays pure; the
  decision is a serving-seam policy.
- **`method="voronoi"`** — a *different* model (hard nearest-player assignment), not the zero-velocity
  limit of the reaction-time model; would not match `add_pitch_control` and shifts the burden onto every
  caller. (Voronoi remains an explicit caller choice, unchanged.)
- **A per-row `velocity_unavailable`-style token** — the rejected constant-column shape (Decision 4).

## Consequences

C4-free (no new aggregator/backend/model — the count is unchanged; behaviour changes only on
velocity-less frames). The atomic mirror inherits. The SB360 coverage artifact is refreshed on the real
licensed corpus in a separate driver pass (the fully-NaN battery-column count drops 40 → ~27:
Tier-1's 13 populated, Tier-2's 7 kept NaN, Tier-3's 20 kept NaN). **The lift is COVERAGE-GATED, so
~27 is a best case:** an actor/keeper-specific Tier-1 column (`gk_pitch_control_share_weighted`, the
cover-shadow columns, `space_created_m2`) populates only where that player is captured in the
freeze-frame (SB360 partial visibility + the ADR-055 keeper-dependence), while team-level columns
(`off_ball_xt_*`, `obso`, `pausa`) populate more readily. The committed open-360 fixture-only preview
confirmed exactly that split — team-level lifted, Tier-2 suppressed, actor/keeper-specific NaN where the
player was absent — so the true count is what the licensed-corpus pass measures, and may exceed ~27.

**Known limit (voronoi + gk/player).** The edge helper's fail-fast is keyed on the pitch-control
`method`, but `add_gk_influence`/`add_player_influence` primitives (b)/(c) always use the Spearman
kinematic TTI regardless of that method. So a `method="voronoi"` (non-default) call on a
forgotten-velocity frame slips past the helper and degrades to warn+NaN at the per-frame TTI rather
than raising — a triple-rare corner (non-default method + caller bug), and it warns rather than
failing silently. Left as documented behaviour; the default (`spearman`) path fails fast as designed.

**The silent-fabrication class is now EMPTY.** `scripts/audit_velocity_fixtures.py` (the ADR-053/4.76.0
velocity-fixture discriminator) measured seven pitch-control aggregators as `SENSITIVE` — silently
producing a DIFFERENT value on declared-but-absent velocity, the shape a defective fixture asserts on.
This change moves all seven to `REFUSES`, so the `SENSITIVE` set is now **empty**: no velocity consumer
silently fabricates (4.76.0 had closed only the ghost path). Its positive control is reframed around
refusal (`surfaced_refusing` — the engine still SURFACES a planted bad fixture, now loud not silent), a
new test pins `SENSITIVE` empty, and the standing gate catches `convicted OR surfaced_refusing` so it
does not go vacuous.
