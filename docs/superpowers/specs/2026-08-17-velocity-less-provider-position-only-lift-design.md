# Velocity-less-provider position-only lift — design

> **For the reviewing session:** this spec is a design proposal, not an approved plan. Two
> decisions are explicitly flagged **FOR REVIEW** (§7). Everything else is a reasoned default that
> the review may challenge. No code has been written.

## Executive summary

On the real 30-match SB360 licensed corpus (enabled in 4.84.0), **40 of ~230 tracking battery
columns come out fully-NaN**. The 4.84.0 team-id fix already restored ~35% of columns; the residual
40 are velocity-related. This spec proposes to **lift the 13 model-relative columns** (verdict
`differs_by_design`) — the ones a *zero-velocity positional model* answers honestly — keep the 20
velocity/multi-frame-constitutive columns as honest-NaN, and route the **7 biased physical-estimate
columns through the D1 decision** (§7 — the review's PREFERRED resolution keeps them NaN too, so the
likely lift is **13, up to 20** if the owner opts the biased set in behind a frame-level signal). One
principle:

> **Surface a value iff the zero-velocity computation is the honest MODEL output, not a systematically
> BIASED estimate of a physical fact wearing the same column name.**

> **Review status (2026-08-17):** another session reviewed this spec + its plan and verified the premise
> sound; its findings are folded in — the decision axis is reframed (model-relative vs physical-estimate),
> the boundary is now MEASURED (D2), the D1 per-row-token lean is REJECTED, the "one seam" claim is
> corrected, and the cross-repo exposure is stated. See the inline `[Review correction: …]` markers.

The change is small in code but is a **behaviour change for every velocity-less provider** (SB360
today, any future freeze-frame or broadcast-snapshot provider), so it wants its own ADR (**ADR-063**),
its own release, and a re-run of the SB360 coverage artifact. It follows 4.84.0 (which provides the
loader + the coverage artifact that motivates it).

**It fixes an existing inconsistency rather than adding a special case:** `add_pitch_control` already
computes a zero-velocity spearman surface (it zero-fills `vx`/`vy`), while `gk_influence`,
`cover_shadows`, `player_influence` and `space_creation` refuse. This proposal makes them consistent
— and does so through the *principled* seam DAS already uses, not `add_pitch_control`'s too-loose
blanket zero-fill.

## Context — what was found (4.84.0 SB360 driver)

- SB360 frames declare `speed_source == SPEED_SOURCE_UNAVAILABLE`: a single freeze-frame per action
  has no per-player temporal history, so `vx`/`vy` **cannot** exist (not "not yet").
- `pitch_control/_dispatch.py:75-82` raises unconditionally when a velocity-requiring method
  (`spearman`/`fernandez_bornn`) is asked for on a frame with no `vx`/`vy` **columns**.
- `pitch_control_at_target` (`features.py:2632-2638`) **zero-fills `vx`/`vy` = 0.0 if the columns are
  missing**, so its spearman runs as a valid zero-velocity model (0.902 populated on the corpus).
- The four aggregators below call `compute_pitch_control` (or `compute_tti`) **without** that
  preparation, so they hit the raise → NaN (or, for `space_creation`, an actual raise).
- A principled seam already exists: `tracking/_velocity_availability.velocity_unavailable_by_design(frames)`
  (`:15-30`) returns True iff **every** row carries `speed_source == SPEED_SOURCE_UNAVAILABLE`, and
  **False when the marker is absent** — so a frame merely *missing* `vx`/`vy` (a forgotten
  `derive_velocities()`) is correctly NOT treated as declared-unavailable. **DAS already uses it**
  (`_das.py:254-259`) to degrade to `das_source=unscoreable_frame` on declared-unavailable frames
  while staying loud on a mistake. This proposal extends that same discipline to pitch control.

Attribution: Spearman 2018 (reaction-time pitch control — the zero-velocity baseline); Fernández &
Bornn 2018 (velocity-aware influence — the refinement). See NOTICE.

## The decision framework

> **Reframed after review (2026-08-17).** The original "unbiased vs biased" framing was imprecise and
> led to the wrong D1 answer. The correct axis is **model-relative vs physical-quantity estimate**, and
> the boundary is **measured, not asserted**.

Classify what the zero-velocity computation PRODUCES:

- **Model-relative** — a dimensionless share or a model-integral with no velocity-independent physical
  referent (pitch-control share, off-ball xT). The zero-velocity value IS the honest model output;
  there is no "true" value it is biased against. This is the classic Spearman reaction-time model —
  **weaker, not invented** (ADR-053) → **Tier 1, lift.** Verdict is **`differs_by_design`**, NOT
  "works" (`VelocityRegimeDiagnosis`, `schema.py:308` "a well-defined positional model, not a fabrication";
  ADR-053:45).
- **Physical-quantity estimate** — an estimate of a physical fact (m², seconds) that exists regardless
  of observation (reachable area, closing time). `compute_tti` is strictly monotonic in the
  along-target velocity component (`d(tti)/d(v_proj) < 0`), so zeroing a moving player's velocity
  **systematically** understates area / overstates closing time (~5× reach-radius swing at 5 m/s;
  confirmed, directional) → **Tier 2, biased.** Its fate is D1.
- **Velocity- or multi-frame-constitutive** — the quantity cannot exist without velocity or a time
  window → **Tier 3, honest-NaN.** A value would be fabrication.

**The boundary is MEASURED (§7 D2):** the velocity-bearing fixture built for the Task 8 invariance proof
ALSO measures each column's zero-velocity-vs-velocity-aware delta; small/none → Tier 1, large/directional
→ Tier 2. Do not place the boundary by armchair reasoning (ADR-055 "measured, not inferred").

## Tiers (exact columns)

### Tier 1 — LIFT; verdict `differs_by_design` (13 columns, subject to the D2 measurement)

Model-relative quantities. Zero velocity is the **classic** Spearman reaction-time model — weaker, not
invented (ADR-053) — so the value is honest but its ADR-053 verdict is **`differs_by_design`**, NOT
"works" (this corrects the original table and aligns with the Testing section, which already said the
verdicts flip toward `works`/`differs_by_design`).

**Two of the 13 are ride-along columns, not spatial-control quantities:** `obso_epv_source` (a
provenance token) and `max_single_defender_player_id` (an id) are simply emitted once the aggregator
stops raising — the "zero velocity is the classic model" rationale does not apply to them. Harmless, but
do not over-interpret the count as "13 honest positional metrics" (it is 11 + 2 ride-alongs).

| Aggregator | Columns |
|---|---|
| `add_cover_shadows` (6) | `blocked_threat_fraction`, `blocking_score`, `max_single_defender_blocking_score`, `max_single_defender_player_id`, `n_blocked_receivers`, `n_potential_receivers` |
| `add_space_creation` (3) | `space_created_m2`, `space_denied_m2_opponent`, `obso_epv_source` |
| `add_gk_influence` (1) | `gk_pitch_control_share_weighted` |
| `add_player_influence` (3) | `off_ball_xt_team`, `off_ball_xt_opponent`, `off_ball_xt_diff` |

### Tier 2 — physical-quantity estimates, systematically biased at zero velocity; fate decided by D1 (7 columns)

Kinematic quantities. The zero-velocity value is the **from-a-standstill** limit — well-defined, but a
`compute_tti`-monotonic *systematically biased* estimator of the motion-aware physical truth (understates
reachable area, overstates closing time). This puts them **closer to the velocity-constitutive Tier-3
family than to the model-relative Tier-1** — they estimate a physical fact that exists independent of
observation, and the zero-velocity estimate is directionally wrong. **Their fate is the D1 decision
(§7):** the review REJECTS lifting them behind a per-row token, and the two live options are (preferred)
keep them NaN + a `velocity_unavailable`-style token like DAS/`press_commitment`, or (acceptable) lift
them behind the existing frame-level `VelocityRegimeDiagnosis`. Do NOT ship a biased physical number
with no signal at all.

| Aggregator | Columns |
|---|---|
| `add_gk_influence` (3) | `gk_reachable_area_m2`, `gk_closing_time_mean_s__six_yard_box`, `gk_closing_time_min_s__six_yard_box` |
| `add_player_influence` (4) | `actor_reachable_area_m2`, `reachable_area_team`, `reachable_area_opponent`, `reachable_area_diff` |

### Tier 3 — DO NOT LIFT; keep honest-NaN (20 columns)

Velocity- or multi-frame-**constitutive**; a value would be fabrication.

| Reason | Columns |
|---|---|
| Accessible-space is constitutively velocity (external dep; already degrades correctly) | `add_das`: `das_diff`, `das_opponent`, `das_team` (3) |
| Need a multi-frame pre-window; SB360 is one freeze-frame per action | `add_actor_pre_window` (2), `add_off_ball_context` (2), `add_off_ball_runs` (2) |
| The column IS a velocity | `add_action_context.actor_speed` (1), `add_press_commitment.press_commitment{,_closing_speed}` (2) |
| Fitted model refuses velocity-less input by design (ADR-054) | `add_ghost_gk.ghost_gk_{x,y}` (2), `add_xcross_attempt.xcross_attempt` (1) |
| Needs multi-frame ball trajectory | `add_shot_goalmouth` (5): `shot_crossing_y`, `shot_crossing_z`, `shot_on_target_derived`, `shot_time_to_goal_line`, `shot_z_profile` |

## Architecture

**One shared edge helper, policy at the edge.** Add to `tracking/_velocity_availability.py` a helper
that PREPARES frames for a velocity-requiring pitch-control call:

```
def zero_velocity_if_unavailable(frames: pd.DataFrame) -> pd.DataFrame:
    """Return frames unchanged if vx/vy present; a zero-velocity copy if velocity is DECLARED
    unavailable (speed_source == SPEED_SOURCE_UNAVAILABLE on every row); unchanged otherwise (so the
    dispatch's loud raise still fires on a forgotten derive_velocities())."""
```

- The four aggregators (`_gk_influence`, `_cover_shadows`, `_player_influence`, `_space_creation`)
  call it once at their pitch-control seam, before `compute_pitch_control` / `compute_tti`.
- `pitch_control_at_target` **replaces its ad-hoc `vx/vy=0` block** (`features.py:2632-2638`) with the
  same helper — so its current *unconditional* zero-fill (which silently accepts a forgotten-velocity
  frame) becomes `speed_source`-aware too. **This is a deliberate tightening** (a caller who today
  gets a silent zero-velocity value on a non-declared-unavailable frame will now raise). Breaking, and
  intended.
- The `compute_pitch_control` dispatch is **UNCHANGED** — it stays a pure computation that raises when
  it cannot compute (policy stays at the edge; the engine stays pure, per the codebase rule that put
  the ghost clamp at the serving seam and the xt_gk base-rate switch in `compute_xt_gk`, not in
  `predict_*`).
- **The "one seam" claim does NOT fully hold — a second latent zero-fill path exists (review finding).**
  `gk_influence` uses `compute_tti` directly (closing-time / reachable-area), and `compute_tti` has its
  OWN loose `vx`/`vy`→0.0 defaults (`_gk_influence.py:206-207` and `335-337`) that are NOT routed through
  the helper and are NOT `speed_source`-aware — the exact loose pattern this proposal tightens elsewhere,
  currently masked only because the pitch-control-share dispatch raises first. For "single principled
  seam" to be TRUE the fix must either remove those defaults (rely on the helper) or make them
  `speed_source`-aware; at minimum the plan must exercise them, not claim one seam.
- **Helper caveat (a one-line comment at the seam):** the zero-velocity copy adds `vx`/`vy`=0 while
  LEAVING `speed_source=unavailable`, so it is internally inconsistent if that frame is ever passed
  ONWARD (a consumer reading `speed_source` sees "unavailable" on a frame that now carries velocity
  columns). Fine for the immediate pitch-control call it is built for; note it so a later reader is not
  surprised.

## Retrain / breaking-change analysis

- **No retrain.** No bundled model trains on a velocity-less provider (SB360 is not in any training
  corpus), and the change touches ONLY frames missing `vx`/`vy` — a velocity-bearing frame's output is
  byte-identical, so every existing trained-model input is unchanged. State this in the ADR-045
  reflection-registry idiom ("invariant on velocity-bearing frames").
- **Breaking (intended), WITH cross-repo exposure — state it explicitly.** `pitch_control_at_target`
  (and the tightened `add_pitch_control`) on a frame missing `vx`/`vy` that is NOT declared-unavailable
  now RAISES instead of silently returning a zero-velocity value. **In-repo blast radius is ZERO**
  (reviewer-verified): no currently-green test passes such a frame, and the calibration path always
  `derive_velocities` first (`_loader_pining.py:492`). **BUT `add_pitch_control` is PUBLIC and consumed
  downstream** (the lakehouse d32 repo). A downstream caller passing a velocity-less-but-undeclared frame
  would now raise — intended (it catches the mistake), but it is a cross-repo Hyrum's-Law change the ADR
  must state. The in-repo caller enumeration belongs IN the plan (fold it in — do NOT defer it to an
  implementation-time grep).
- **VAEP:** the four aggregators are not in a default xfn list change; opting them in is unchanged. No
  default-config VAEP feature moves on velocity-bearing data.

## §7 — Decisions FOR REVIEW

**D1 — Tier-2 guard mechanism.** *[Review correction: the original author-lean — a per-row
`{measured, zero_velocity}` token per family — is REJECTED. It contradicts an explicit design decision
in the same module.]* `VelocityRegimeDiagnosis` (`schema.py:310-312`) states the velocity regime "is a
property of the whole frame set rather than of any row, **which is why this is a diagnostic rather than a
per-row provenance column that would carry a constant**." The proposed per-family token IS exactly that
per-row constant (all `zero_velocity` on SB360, all `measured` elsewhere) — it duplicates a frame-level
signal for no new information and adds a schema contradiction.

**A `velocity_unavailable`-style token that only ever says one thing is the SAME rejected shape (re-review
R3).** `das_source` is a legitimate per-row column ONLY because it VARIES — `computed` / `unlinked` /
`unscoreable_frame` / `team_unresolved` (`_das.py`, `DAS_SOURCE_VALUES`); the variation is what carries
information. A Tier-2 token that is constant `velocity_unavailable` on every SB360 row carries nothing the
frame-level diagnostic does not. The two live options:
- **Preferred — treat Tier-2 like Tier-3: keep it NaN, NO per-row token.** The signal is the existing
  frame-level `VelocityRegimeDiagnosis` (`validate_velocity_regime`), which already declares the whole
  frame set velocity-unavailable. Schema-consistent, simplest, no new vocabulary. **Implementation subtlety
  (do NOT miss):** because `compute_tti` auto-fills `vx`/`vy`=0, once the share path stops raising the
  Tier-2 columns compute FOR FREE — so keeping them NaN needs an EXPLICIT suppression, not just "don't lift."
  *(A per-row source column is defensible only if it is a REAL multi-reason `…_source` in the `das_source`
  idiom — `velocity_unavailable` vs `unlinked` vs `no_gk_in_frame` — and then it must VARY and be tested as
  such. A constant is rejected. That is a strictly larger change; add it only on an explicit ask.)*
- **Acceptable — lift Tier-2 behind the existing frame-level `VelocityRegimeDiagnosis`** (schema-consistent,
  no new per-row column). Only if the owner values the single-provider freeze-frame-research use case
  enough to ship a biased physical number with an opt-in frame-level signal.

The owner picks. The spec's original fallback ("no per-row guard → keep Tier-2 NaN") is now the
**PREFERRED** option — the review supplied the evidence that it is the right answer, not the fallback.

**D2 — Tier boundaries, MEASURED not asserted.** The Tier 1/2/3 split must be placed EMPIRICALLY (ADR-055
"measured, not inferred"), not by the armchair reasoning the table currently encodes. **Reuse the
velocity-bearing fixture built for the Task 8 invariance proof — ONE fixture, two purposes** — to measure
each candidate column's zero-velocity-vs-velocity-aware delta: small/none → model-relative (Tier 1);
large/directional → biased physical estimate (Tier 2). This settles the boundary cases the table asserts
(is `gk_pitch_control_share_weighted` truly ~unbiased while `gk_reachable_area_m2` is biased? does any of
`press_commitment` ride pitch control rather than being pure velocity?). Do not trust this table over the
measurement.

## Rejected alternatives

- **Per-aggregator zero-fill (copy `add_pitch_control`'s block into each).** Rejected: duplicates the
  logic five times, and propagates `add_pitch_control`'s *loose* unconditional zero-fill — it would
  NOT distinguish declared-unavailable from a forgotten `derive_velocities()`, silencing the exact
  mistake the dispatch guard exists to catch.
- **Put the degrade-vs-raise policy in the `compute_pitch_control` dispatch.** Rejected on the "policy
  at the edge, engine stays pure" rule. The dispatch must stay a pure computation; the decision to
  degrade is a serving-seam policy.
- **Tell callers to pass `method="voronoi"`.** Rejected as the primary: Voronoi is a *different* model
  (hard nearest-player assignment), not the zero-velocity limit of the reaction-time model, so it
  would not match `add_pitch_control` or answer the same question; and it puts the burden on every
  caller rather than honouring the provider's `speed_source` declaration. (Voronoi remains available as
  an explicit caller choice for anyone who wants that model — unchanged.)

## Testing

- **Both-sides, per aggregator:** on a declared-unavailable fixture, each Tier-1/2 aggregator produces
  finite values (not NaN, not a fabricated constant); on a frame merely *missing* `vx`/`vy` with no
  `speed_source=unavailable` marker, it still **raises** (the mistake path). A liveness-style
  non-vacuity assertion: a mutation that should move the value out of range does.
- **Tier 3 stays NaN:** assert the constitutively-velocity columns remain NaN on the same fixture
  (guards against an over-broad fix).
- **ADR-053 SB360 audit re-adjudication:** the four aggregators' recorded verdicts flip from
  `raises`/`honest_nan` toward `works` / `differs_by_design`; regenerate + re-adjudicate with a written
  rationale (`tests/sb360/_regenerate.py` + `_adjudicate.py`), round-trip byte-identical.
- **Velocity-bearing invariance:** a velocity-bearing fixture is byte-identical before/after (the
  no-retrain claim's proof).
- **Purity / id-dtype / liveness / mirror gates:** re-verify the four aggregators still pass (the fix
  is additive to their output on velocity-less frames).
- **Driver re-run:** re-run `scripts/validate_sb360_licensed_corpus.py` from a clean commit to refresh
  `docs/research/sb360_licensed_coverage/` (fully-NaN column count drops **40 → ~27 under the PREFERRED D1**,
  the 13 Tier-1 columns populated and Tier-2 kept NaN; → ~20 only under the ACCEPTABLE branch).

## Sequencing

Lands **after** 4.84.0 (which ships the SB360 loader + the coverage artifact this measures against).
Its own release, its own ADR-063, C4-free (no new aggregator/backend/model — the aggregator count is
unchanged; behaviour changes only on velocity-less frames).
