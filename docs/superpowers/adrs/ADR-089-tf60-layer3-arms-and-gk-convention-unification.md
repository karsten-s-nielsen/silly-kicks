# ADR-089: TF-60 Layer-3 counterfactual arms + ghost-GK both-axes convention unification

| Field | Value |
|---|---|
| **Date** | 2026-09-06 |
| **Status** | Accepted |
| **Deciders** | Karsten Nielsen |

## Context

Two things were inconsistent and one thing was incomplete after TF-60 PR5:

1. **The ghost-GK feature extractor used an x-ONLY goal-relative convention.** `extract_ghost_gk_features`
   flipped only the x axis (`105 - x`) to make features goal-relative, leaving signed-y features
   (`ball_y`, `ball_vy`, `atk_cy`, `ball_to_goal_angle`) and the target `gk_y` in absolute frame
   coordinates. The correct goal-relative transform is a **180-degree point reflection** — both axes,
   `(x, y) -> (105 - x, 68 - y)` and `(vx, vy) -> (-vx, -vy)` — exactly as `_geometry.to_goal_relative_*`
   already does for the other models (ADR-051 §8b). The x-only form is not orientation-invariant: one
   physical scene scores differently at the two goal ends. The ghost-OUTFIELD model (PR5) already used
   the both-axes convention via its module-level `_gr_x`/`_gr_y`.

2. **The two ghost serves returned goal-relative coordinates, not frame coordinates.** Every consumer
   (gkdv `_engine`, the new restdefense counterfactual) had to re-derive orientation to write the ghost
   back into a frame. gkdv's `_engine` did an **x-only** goal-relative -> frame conversion, which was
   correct only while the ghost-GK model was itself x-only. Unifying the model to both-axes would break
   that consumer silently.

3. **TF-60 Layer 3 (the counterfactual deterrent arms) was unbuilt.** The rest-defense cycle had shipped
   Layer 1 (structure KPIs, ADR-080), Layer 2 (danger valuation, ADR-081), the ghost-GK sweeper re-fit
   (ADR-083), and the trained ghost-outfield model (ADR-087), but not the arms that price how much the
   in-possession team's actual rearguard / keeper suppresses the opponent's counter-danger versus a
   league-average ghost.

The owner directed these be folded into ONE "gold-standard" cycle (scope/breaking not a concern) rather
than the originally-planned separate PR4 (GK arms) + PR6 (outfield arm).

## Decision

Ship as one cycle (two provenance-mandated commits — code, then re-fit weights + applied artifacts):

1. **Ghost-GK both-axes convention (retrain all 5 variants).** The extractor routes every signed-y
   feature AND the target through module-level self-inverse transforms `_to_gr_x`/`_to_gr_y`/`_to_gr_vx`/
   `_to_gr_vy` (hoisted from the per-call closures so the extractor and the serve share one function).
   The chirality gate is strengthened to catch a y-only regression. All 5 variants
   (`default`/`position_only`/`sweeper`/`sweeper_position_only`/`full`) are re-fit from the Commit-1 SHA.

2. **Both serves return frame-ready `ghost_x`/`ghost_y`** via each model's own goal-relative inverse
   (`serve_ghost_gk_positions`, `serve_ghost_outfield_positions`), keeping `ghost_gr_x`/`ghost_gr_y` as
   audit. No consumer re-derives orientation — the transform is the single source of truth.

3. **gkdv `_engine` consumes the serve's frame coords** (the x-only goal-relative->frame math is
   deleted). This is the load-bearing consequence of (1): once the model is both-axes, the old x-only
   engine conversion mislocated the away-team keeper in y.

4. **Complete Layer-3 arms (`restdefense/`), a gkdv sibling.** `build_restdefense_ghost_frames` ghosts
   the in-possession team A's OWN rearguard (`which="rearguard"`) or keeper (`which="keeper"`) near A's
   OWN goal when A is committed-forward; `rest_defense_outfield_deterrent` / `rest_defense_gk_deterrent`
   price the opponent B's counter-danger via gkdv's generic `delta_threat_suppression_batch` /
   `delta_das_batch` seams (attacker-value units, `negative = deterrent`). A `_probe.py` mirrors gkdv's
   TF-19 A+2 instrument-validity probe, reusing gkdv's PUBLIC verdict functions.

5. **The gkdv re-materialize + TF-19 sign-off re-run are IN this cycle** (Phase B, from the new weights).

6. **A corpus driver + applied construct-validity report** (`scripts/build_tf60_layer3_arm_values.py`;
   `docs/research/tf60_layer3_construct_validity/`). xT is fit once on the loaded corpus and injected —
   the established convention for a reported-not-gated corpus measurement driver that needs an
   `ExpectedThreat` (`measure_cover_shadow_argmax_agreement.py`; silly-kicks ships no xT model).

## The "keeper-agnostic" property is STRUCTURAL isolation, not value invariance (correction)

The design spec (§6.2) originally asserted the outfield arm is *"keeper-agnostic in the delta (a
property, not a coincidence) — the keeper's control/TTI contribution cancels."* **That is an over-claim,
proven false by execution and corrected here.** `delta_threat_suppression` carries `lambda_gk` (A's
keeper as a TTI control agent) and pitch control is NONLINEAR, so A's fixed keeper interacts differently
with the DIFFERING rearguard across the two legs; it does not cancel. Measured on the toy fixture (after
the opponent-selection fix below): moving A's keeper shifts the outfield THREAT arm ~3.6% (the space/DAS
arm is keeper-blind-generic and near-invariant, ~1%). The exact-equality test the spec mandated passed
ONLY under a since-fixed opponent-selection bug.

The **honest, exact** property is STRUCTURAL: the outfield counterfactual repositions only A's rearguard
and NEVER moves A's keeper, so the arm attributes no keeper repositioning to the rearguard; the value is
rearguard-DOMINATED, not keeper-free. This is what the shipped code + tests assert.

## Two correctness bugs found and fixed during implementation (both `ids_match` misuse)

Both traced to `ids_match(SeriesA, SeriesB)` — `ids_match` is Series-vs-SCALAR, so passing a Series as
the scalar compares every row against the whole Series object and returns all-False:

- **The keeper counterfactual scored nothing** (`no_ghost_served` on every in-domain keeper frame): the
  GK serve is keyed `(frame, team)` with no `player_id`, but the provenance/write-back match on
  `player_id`; and the possession restriction used the Series-vs-Series `ids_match`. Fixed by attaching
  A's keeper `player_id` to the served rows and using `ids_equal` (column-vs-column).
- **The outfield arm priced team A's OWN accessible space, not the opponent B's**: `_opponent_by_frame`
  used `~ids_match(SeriesA, SeriesB)` (all-False -> `~` all-True), so B silently resolved to the FIRST
  team per frame (= A). Fixed with `ids_differ`. The prior arm tests passed because they never asserted
  the arm VALUE.

Both are guarded red-first. A third pre-existing robustness bug (`compute_rest_defense` crashed with
`KeyError('rd_num_superiority')` on a match with zero committed-forward samples — fatal to a corpus pass)
is fixed to return an empty declared-columns table.

## A fourth bug: `add_ghost_gk` double-flipped away-team y (Commit-1 defect, masked by transient-red)

Switching the ghost-GK model to serve BOTH-AXES goal-relative `gr_y` (Decision 1) left `add_ghost_gk`'s
ADR-028 action-LTR reprojection (`features.py`) unchanged, and it was written for the x-only model:
`ghost_gk_y = 68 - gr_y` **only for away-team actions** (flip-gated), because under the old convention
`gr_y` was absolute-frame y. Once `gr_y` became goal-relative, that away-only flip DOUBLE-flipped: the
keeper's goal-relative flip is the COMPLEMENT of the acting team's action flip, so the per-action
reflection cancels the model's own and action-LTR y must be `68 - gr_y` **UNIFORMLY**. The buggy code
produced `mir.y = 68 - base.y` (a physical y point-reflection, ~6.67 m off) for the flip=False leg.

The fix is one line (`features.py`: flip-gated → uniform `FIELD_WIDTH - gy`; `ghost_gk_xfns` inherits it
by delegation). Its correctness is verified by `test_ghost_gk_mirror_invariant` (base.y ≈ mir.y to 0.02 m;
non-vacuity flip_dy 6.67 m). **Why it shipped in Commit 1:** that test loads the bundled `default`, so it
was in the Phase-A chirality transient-red set and its assertion never ran — a transient-red test masking
a real value bug. The velocity-path output golden was re-captured against the corrected reprojection
(measured move recorded in its docstring). This affects only the VAEP feature path (`add_ghost_gk` /
`ghost_gk_xfns`, in no default xfn list); the gkdv/restdefense/TF-19 corpus passes use
`serve_ghost_gk_positions`, which was correctly updated, so their artifacts are unaffected.

## Consequences

- **Retrain / Hyrum triggers:** the ghost-GK both-axes change invalidates all 5 bundled variants'
  feature-contract + chirality fingerprints (regenerated Phase B); gkdv arm values change for away-team
  keepers (the x-only engine conversion was wrong in y); the gkdv re-materialize + TF-19 sign-off are
  re-run from the new weights.
- **Additive to VAEP** (the arms are in no default xfn list; reported-not-gated) — no VAEP retrain.
- **C4-free** — no new action-coupled aggregator, backend, or model container.
- **Phase-A transient-red:** between Commit 1 and Commit 2 the bundled ghost-GK tests (golden / chirality
  / feature-contract / any `model=None` bundled load) are red on load; the branch is green at its tip
  (Commit 2, the `--merge` gate) once the re-fit weights are bundled.
- **SB360 boundary-audit registration for the two arms — DONE.** Both `rest_defense_outfield_deterrent`
  and `rest_defense_gk_deterrent` are registered in `BOUNDARY_ENTRY_POINTS` with per-column verdicts
  *observed by execution* (both arms: threat `differs_by_design` — the ADR-063 zero-velocity
  pitch-control lift; space `honest_nan` — the velocity-required DAS degrade; `gk_absent`
  `not_exercised` — no keeper/orientation), and `NOT_EXERCISED_BUDGET` 48→52. The verdicts that load the
  bundled ghost-GK are green at the Commit-2 tip (re-fit weights), the same sequencing the gkdv entries
  followed.
- **A fifth bug (FINDING-5): `rest_defense_outfield_deterrent` crashed on keeperless frames.** Observing
  the SB360 `gk_absent` verdict surfaced it: on a both-keepers-removed frame the goal orientation is
  unresolvable, so `compute_threat_pc` raises `GoalEndUnresolvedError`, and the outfield arm — which
  ghosts the rearguard (present without keepers) so it reaches the threat computation — did not catch it
  and crashed, unlike its siblings (`compute_rest_defense`, the gkdv arms, the keeper arm) which all
  degrade to honest-NaN per ADR-055. Fixed (`_arms.py::_frame_deltas` catches it → NaN threat) + a
  red-first test on the `gk_absent` roster. Only the synthetic audit roster / a no-visible-keeper frame
  hits it; real full-tracking always resolves the goal.
- **The pining SkillCorner loader was fixed to read the new artifact schema (rode this cycle to run the
  full-corpus validation).** SkillCorner's newer artifacts ship tracking as a PARQUET table
  (`_tracking.parquet`, nested `ball_data`/`player_data` struct+list) instead of
  `_tracking_extrapolated.jsonl`; `build_skillcorner_frames` opened it as UTF-8 text and crashed, so the
  DGX validation drivers (`load_matches`) could not load any SkillCorner match. The fix reads the parquet
  into the same per-frame record shape (coercing the numpy `player_data` to a list); verified against
  real corpus matches (frame counts match the vetted tc3 cache exactly) + a parquet-vs-jsonl equivalence
  test. The re-fit *weights* are unaffected (training reads the tc3 cache, never the raw loader).
- **The 4 restdefense arm columns are documented in `feature_glossary` (`emitting_module=_arms`, with an
  emitted-columns leg so the entries are non-stale) and carry a spec-§10 liveness gate** (non-NaN +
  non-constant on the `computed` rows of the multi-domain fixture; both arms measured live). The arms
  are dtype-safe via `id_compat`; the id-scalar completeness registry is genuinely **N/A** here — its
  population is `spadl`/`atomic`/`vaep`/`causal`/`tracking` (restdefense, like gkdv, is not in it), which
  spec §10's "id-scalar registry *if a new public id-scalar function is added*" already accommodates.

Attribution: Le et al. 2017 (ghosting); Kim 2026 (DEFCON-GNN, comparator); Bischofberger & Baca 2026
(rest-defense framing). See `NOTICE`.
