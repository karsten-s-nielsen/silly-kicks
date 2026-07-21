# ADR-042: TF-35 off-ball run valuation

| Field | Value |
|---|---|
| **Date** | 2026-07-18 |
| **Status** | Accepted |
| **Deciders** | Karsten S. Nielsen; lakehouse review session (spec rounds 1–2, plan rounds 1–2) |
| **Supersedes / amends** | Extends ADR-005 (feature surface + `<feature>__<method>` naming), ADR-008 (shared pitch-control cache), ADR-019 (id-dtype seams), ADR-028 (action-LTR emission), ADR-033 (`add_*` purity); inherits the ADR-039 F4 result-leakage decision |
| **Source plan** | `docs/superpowers/plans/2026-07-18-xt-epv-wiring-and-tf35-run-valuation.md` |

## Context

TF-4 (PR-S30) already detects off-ball runs, but only COUNTS them. It cannot say whether a
run mattered: a 6 m jog into a crowded zone and a 6 m sprint into the space behind a
defensive line score identically. The Soccermatics Pro course frames a run's worth as the
threat of the space the runner comes to control, which is exactly what the shipped
pitch-control and xT surfaces can already express — the pieces existed, nothing joined them.

The lakehouse has an outstanding full recompute of `fct_action_context`, which made this the
right moment: new tracking-derived columns cost nothing extra if they land before that run.

## Decision

Two pure primitives in a new `silly_kicks/tracking/_run_values.py`, deliberately split so the
expensive half is optional:

**1. `detect_off_ball_runs(actions, frames, *, home_team_id, params=None)`** — geometry only.
One row per qualifying `(action, runner)` pair. Candidacy is the SHARED TF-4/TF-35 predicate
`_prepare_run_candidates` (extracted at the third consumer, per the house rule: the TF-4
kernel, TF-35's detector, and the gate that compares them). A candidate becomes a run when
its first-to-last displacement clears `min_displacement_m` AND its peak speed clears
`min_peak_speed_ms` (default 5.56 m/s = 20 km/h). Positions are emitted in SPADL
**action-LTR** (ADR-028), so `run_end_x > run_start_x` means "toward the attacked goal" for
both teams.

**2. `value_off_ball_runs(runs, actions, frames, xt, ...)`** — role + value. Domain is a
completed `pass`/`cross` whose next same-team touch resolves to a receiver. The receiver's
run is the `"target"` run; every other qualifying run is `"disruptive"`. `run_value` is the
MAXIMUM of `pitch_control * threat` over the cells the runner controls (per-player influence
at or above the resolved region floor).

**3. `add_off_ball_run_values`** emits five wide columns; `off_ball_run_value_xfns` exposes
four of them × 3 slots. Atomic mirrors both.

### The decisions worth recording

**Max, not influence-weighted mean.** `run_value` answers "how much dangerous space did this
runner come to own", so it takes the peak of the controlled region. The consequence is the
whole `region_influence_floor` apparatus: a max over a thresholded region NEEDS a threshold.
An influence-weighted mean would delete the knob entirely — every cell would contribute in
proportion to influence, with nothing to threshold. **That trade (peak opportunity vs average
ownership) is a recorded v2 fork, not an oversight**, and it is stated in the params
docstring rather than buried here.

**The 0.1 default floor is a spec-time starting value, NOT calibrated.** Its calibration is
the sensitivity probe (0.05 / 0.1 / 0.2). `resolved_region_floor()` FAILS LOUD for any
`pitch_control_method` without a recorded floor, and the message names the specific hazard:
`voronoi` per-player influence is binary `{0, 1}`, so every floor in `(0, 1]` selects the
same cells and the knob silently stops meaning anything.

**`peak_speed_source` is emitted as DATA.** When a provider's `speed` column is entirely NaN
across a runner's window, the gate falls back to `displacement / duration`. That fallback
systematically UNDER-states peak speed (an average cannot exceed a maximum), so a
sparse-speed provider LOSES runs rather than gaining them. Recording which path was taken
per run makes the bias auditable instead of a docstring footnote.

**Mean disruptive value must divide by `n_valued_disruptive_runs`, never
`n_disruptive_runs`.** The sum skips unvalued runs, so the wrong denominator deflates the
mean in exact proportion to how poor the provider's tracking coverage was — converting a
data-quality gradient into an apparent tactical one. Carried as a `.. warning::` on the
aggregator, not just here.

**A visibility gap is not a zero.** A runner absent from the linked frame's pitch control
keeps `run_value` NaN, KEEPS ITS ROW, and keeps its event-derived role; the call warns once
with `RunValueCoverageWarning`. Fabricating a 0 would silently reward bad tracking.

**`n_valued_disruptive_runs` is excluded from the VAEP surface.** It is a coverage
denominator, not a football quantity; as a model feature it would let tracking visibility
stand in for play quality.

**Result leakage (inheriting ADR-039 F4).** The domain gates on the action's OWN `result_id`,
so as an a0-slot feature this is the leakage class `HybridVAEP` exists to strip.
`off_ball_run_value_xfns` is in NO default xfn list, and that is enforced by an
auto-discovering executable guard (`test_run_value_xfns_leakage_guard.py`) with a
transformer-name anchor so a rename fails loudly rather than neutering the check.

**`_safe_index_of` resolves the player index ONCE with a canonical-id compare.**
`PitchControlSurface.player_surface` / `.player_share` use a raw `==` (`_surface.py:140,167`)
and RAISE on a miss, so a check-then-call would blow up mid-loop on exactly the mixed-dtype
ids those helpers exist to serve.

**TF-4's `toward_goal` is re-keyed onto `acting_team_attacks_rtl` — a BEHAVIOUR CHANGE, not a
no-op.** TF-4 was re-keyed onto `acting_team_attacks_rtl`; this did NOT eliminate identity-keying from the action-coupled geometry layer — other action-coupled aggregators still take `home_team_id` by design (the earlier "last module keyed on home/away identity" phrasing was wrong; ADR-045 D6). `_line_break_kernel`'s coordinate resolution and the per-frame influence families likewise take `home_team_id` by design, while the direction authority
already had 7 production call sites. The two disagree exactly where the acting team has no
direction-carrying frame row in that period: identity-keying always resolves and flips the
away team, the direction authority conservatively does not flip. Rows like that exist in real
data (a team briefly absent from a broadcast window). Honest framing, carried in the code
comment: **this buys CONSISTENCY, not correctness** — on correctly-labelled frames the two
already agree, and on unoriented frames both are arbitrary. Verified NOT a retrain trigger:
`off_ball_context_xfns` is absent from `tracking_default_xfns`.

**Role assignment on shifted VAEP slots is approximate, and says so.** `gamestates` fills
the leading `i` rows of slot `i` with a REPEAT of the first action, so on those fill rows the
"next same-team touch" resolves to the action's own actor — who is excluded from run
candidacy, making every run there read as `disruptive`. Away from the boundary the shifted
rows still hold consecutive actions, so resolution is correct; the effect is bounded by `i`
rows per game per slot. `packing_xfns` refuses `require_secured` outright for the same
shifted-slot reason (ADR-039); TF-35 degrades instead of refusing, because the numeric
columns stay meaningful and the alternative (emitting NaN for every slot) is strictly worse.
Recorded in the factory docstring as a `.. note::`, not left implicit.

## Consequences

- Five new candidate `fct_action_context` columns (`run_value_target`, `n_disruptive_runs`,
  `run_value_disruptive_sum`, `n_valued_disruptive_runs`, `run_value_enabled_pass`) plus the
  per-run `peak_speed_source` provenance.
- TF-4's `n_off_ball_runners_toward_goal_pre_window` changes on unresolvable-direction rows.
- No VAEP retrain from TF-35 itself (opt-in, in no default list).
- **Liveness-fixture extension (execution finding).** The repo's shared liveness fixture
  topped out at ~4.95 m/s of off-ball movement, BELOW the 5.56 m/s sprint gate, so every
  run-value column would have been born dead. Two windows gained genuine sprinters plus a
  follow-up touch that makes the cross window's receiver resolvable. A first pass gave each
  window one disruptive sprinter and `run_value_disruptive_sum` came out constant at 1.0
  across the fixture — live but informationally dead, exactly the failure the non-constant
  check exists to catch; window 4 therefore carries a second disruptive sprinter.
- **Recorded latent gap (NOT fixed here — scope).** `PitchControlSurface.player_surface` /
  `.player_share` compare ids with a raw `==`, an ADR-019 gap that also makes
  `_player_influence.py`'s `except ValueError: -> 0.0` **silently zero players** on
  dtype-mismatched frames. Tracked in TODO Technical Debt.
- **A Spearman property worth knowing before writing fixtures** (cost a debugging cycle):
  `compute_pitch_control` zeroes a player's influence at every cell the BALL reaches before
  that player can. A runner standing next to the ball therefore has zero influence
  EVERYWHERE, and any value oracle built on such a fixture degenerates to 0. Model behaviour,
  not a bug — but a fixture trap.
