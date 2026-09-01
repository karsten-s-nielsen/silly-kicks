# TF-60 §9 — Ghost-GK in-possession validity (the finding that reshaped the arc)

- **Date:** 2026-08-30
- **Cycle:** TF-60 PR3 brainstorm (before any PR3 code)
- **Decision it drove:** insert a **rest-defense GK-ghost model re-fit** cycle *before* the Layer-3 GK
  arms (spec §9/§20.1 "this is a gate, not an assumption"; owner ruling 2026-08-30). See
  `docs/superpowers/specs/2026-08-30-tf60-restdefense-gk-ghost-refit-design.md`.
- **Status of the number:** the QUALITATIVE conclusion is settled (structural + controlled probe); the
  authoritative *fraction* of real committed-forward frames above the ceiling is deferred to the DGX
  corpus, measured inside the re-fit cycle (owner: "qualitative finding is enough to proceed"). The DGX
  was unavailable (other session) so this is a **local, directional** read — decisive on the mechanism,
  not on the population fraction.

## The question

Rest defense (TF-60) ghosts the **in-possession** team's keeper — a high sweeper standing well off its
own goal while its team is committed forward — and prices how much the actual keeper's position
suppresses the opponent's counter-danger versus a league-average "ghost" keeper (spec §7.3). The
counterfactual is served by the shipped `GhostGkModel` (TF-18). §9 asks whether that model — exercised
by GKDV only in the **defending-near-attacked-goal** regime — is valid for the **opposite** geometry.

## The structural finding (settles it before any measurement)

`prepare_ghost_gk_training_data` (`silly_kicks/tracking/_ghost_gk.py:1204-1220`) **filters the training
labels to a goal-relative box capped at `GRID_X_MAX = 30 m`**, with the comment *"Filter label domain
(sweeper-keeper rushes, off-pitch artifacts)."* The bundled `default` model's `metadata.json`
`grid_spec.x_max` is `30.0`, and `serve_ghost_gk_positions` flags `ghost_out_of_box` for exactly
`gr_x > 30 m`.

Consequently the model was **never trained on the keeper labels above 30 m** it would need to predict for
rest defense. `predict_mean` (the boosted HGBR mean) can only interpolate within its trained label
support, so it saturates toward the 30 m ceiling and **cannot place a keeper at 35–45 m** — precisely the
high-sweeper regime rest defense is about. The 26-feature set *can* condition on the regime
(`team_in_possession`, `ball_x`, `ball_distance_to_goal` are all present, and the extractor is
keeper-team-centric), so the failure is not "the model can't see the regime" — it is "the model was
trained never to answer in that range."

## The controlled extrapolation probe (direct confirmation)

Because the small local slices don't contain aggressive high-line moments (see distribution read below),
the decisive test is a **controlled counterfactual**: take a clean full-tracking slice
(`tests/datasets/tracking/action_context_slim/sportec_slim.parquet`), **rigidly translate the whole scene
upfield** by Δ metres toward the attacking goal (a physically-coherent progressively-higher line;
velocities and relative geometry preserved), and serve the shipped `default` model. The home keeper
defends x=0, so its goal-relative x rises by ~Δ. Does the model's prediction **track** the real keeper up
the pitch, or **saturate** at the trained ceiling?

| Δ (m) | actual home-GK gr_x (mean / max) | **predicted** ghost gr_x (mean / max) | `ghost_out_of_box` | verdict |
|---:|---|---|---:|---|
| 0  | 9.4 / 20.7  | 9.7 / 20.7  | 0 % | tracks (+0.3) |
| 5  | 14.4 / 25.7 | 14.4 / 25.8 | 0 % | tracks (−0.0) |
| 10 | 19.4 / **30.7** | 19.3 / **27.8** | 0 % | tail begins clipping |
| 15 | 24.4 / 35.7 | 23.3 / **28.5** | 0 % | mean tracks, tail capped |
| 20 | 29.4 / 40.7 | 26.7 / **29.4** | 0 % | gap −2.7 |
| 25 | **34.4** / 45.7 | **28.6** / **29.8** | 0 % | **SATURATES (gap −5.8)** |

Reading:

1. **The prediction hard-caps at ~29.8 m** regardless of how advanced the scene is. When the actual keeper
   mean is 34.4 m (Δ=25) the model predicts 28.6 m; the predicted **max never exceeds ~29.8 m** while the
   actual max reaches 45.7 m. That ceiling is `GRID_X_MAX = 30`.
2. **The aggressive-sweeper tail clips first** (predicted max 27.8 vs actual 30.7 already at Δ=10), which is
   exactly the signal rest defense most wants to preserve (spec §16.1: "known sweeper-keepers should score
   more negative").
3. **`ghost_out_of_box` is blind to this failure** — it stays 0 % throughout, because the model *clips its
   own output* to ≤30 m and then honestly reports "in box." The regime being out-of-domain is a property of
   the **input**, which no output-based flag can detect. This is why §9 called it *"a gate, not an
   assumption"*: no runtime signal catches it.

## The distribution read (context; why the slices can't be the whole story)

Measuring the **actual** in-possession keeper's goal-relative depth on the committed real-tracking slices
(orientation via `resolve_defended_goals`, ADR-055; possession via `infer_ball_carrier`; committed-forward
= ball past halfway toward the attacked goal):

| Fixture | trust | committed-forward keeper gr_x (n; median; p95; max; %>30 m) |
|---|---|---|
| `sportec_slim` | **full-tracking, native GK — trustworthy** | n=172; med 17.5; p95 27.7; max 28.1; 0 % |
| `skillcorner_slim` | FOV/detection-biased | n=14; med 14.5; max 14.7; 0 % |
| `metrica_slim` | **discarded** (derived-GK mislabel) | keeper at y≈60 m (touchline) — not a keeper |

- **sportec** (the one clean source) already brushes the ceiling in a *tame* slice — committed-forward p95
  ≈ 28 m, overall max 30.8 m. A full 90-minute corpus with genuine high-press moments will push over far
  more often; the slice simply lacks those moments (which is *why* the controlled probe above, not the raw
  slice, is the decisive instrument).
- **skillcorner's 0 %** is the **documented FOV/detection bias** — the broadcast camera only sees the keeper
  when the ball is near him, and the trainer's own `detection_selection_bias` note records that detected
  frames "under-sample the deep sweeper regime GKDV cares about." SkillCorner structurally cannot observe a
  high sweeper, so its 0 % is not evidence keepers stay low.
- **metrica discarded:** `metrica_slim` flags only one team's keeper (Tier-2 derived, single player_id) at
  x≈69, **y≈60 m (near the touchline)** — a mislabeled wide player, not a keeper (ADR-007 Tier-2 fragility).
  Its 87 %>30 m was a pure artifact.

## Conclusion

The shipped `default` `GhostGkModel` is **structurally inadequate** for the rest-defense in-possession
high-sweeper regime. It is not "coverage is thin" — it is a **hard ceiling at ~30 m baked into the
training label filter**, invisible to `ghost_out_of_box`. Serving it for the Layer-3 GK arms would
systematically compress the deterrent for exactly the aggressive sweeper-keepers the metric exists to
reward. The fix the spec anticipated (§9/§20.1) — a **rest-defense-appropriate GK-ghost variant** re-fit
with the label cap lifted and the grid extended — is required, and is scoped as its own trained-model
cycle (the re-fit sub-spec).

## Reproduction

Probe scripts (local, committed fixtures only) are archived with this finding
(`probe_extrapolation.py`, `probe_depth_distribution.py`). The re-fit cycle promotes the extrapolation
probe into a committed regression gate that asserts **both** sides: the old `default` variant saturates
(the defect), and the new extended-grid variant tracks past 30 m (the fix) — the two-sided,
non-vacuity discipline (CLAUDE.md "every band needs a test from both sides").
