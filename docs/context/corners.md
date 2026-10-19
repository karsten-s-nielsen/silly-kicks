# Corners (TF-46 Arm 1)

The `silly_kicks/corners/` family implements the Shaw & Gopaladesikan corner-kick playbook ("Routine
Inspection", 42 Analytics) as **raw primitives** (ADR-110): tactical blends, archetype naming and
which-routine-wins rankings stay consumer-side. Arm 1 is offence (unsupervised, label-free); Arms 2
(defence coupling) and 3 (role-conditioned valuation) are later cycles.

## Pipeline

1. `identify_corners(actions, provider=)` — the direct/short/unclear corner table (per-provider; the
   SkillCorner GI `start_type` / `game_interruption_before` tokens, the sportec/IDSSE DFL corner, the
   SB360 freeze-frame corner). `corner_id` is a within-table per-match chronological `int64` grain key —
   join cross-table on the stable natural key `(game_id, period_id, t_kick)`, never on `corner_id`.
2. `add_corner_windows(corners, frames, contacts)` — per-corner timing: `t_init` (`t_kick - 2 s`),
   `t_target` (`min(first_contact + lead, t_kick + cap)`), the first-contact winner from the shipped
   TF-79 `first_contact_in_window` (NaN when unresolved — NO GI fallback); SB360 freeze corners take the
   kick frame. `first_possession_time` is a GI diagnostic.
3. Reflect every corner to the canonical side (Y-flip for right-flag corners), so one orientation.
4. `compute_corner_run_encoding(windowed, frames, model=RunZoneModel)` — each attacker's run encoded as
   (initial zone × target zone) via Gaussian responsibilities → the per-corner run vector (the NMF term
   matrix). Occupancy rule: only players whose target lands in the goal-centre-radius region
   (`goal_radius_m = 25.0`: box + top-of-box + second-ball edge) contribute (spec §16; a stationary
   second-ball holder counts, no displacement threshold).
5. `compute_corner_routine_features(encoding, model=CornerRoutineModel)` — NMF routine-feature
   coefficients; `cluster_corner_routines` groups corners (agglomerative, raw assignments only).
6. `compute_corner_chance_quality(windowed, actions, xg_column=)` — descriptive per-corner `n_shots` /
   `n_goals` (model-free, always) + provider-native `sum_xg` (honest-NaN where absent — silly-kicks
   ships no xG model, ADR-110).

## Fitted models (ADR-011 params-only, bundled in the wheel)

- `RunZoneModel` (GMM): `_run_zone_weights/default` — init (6) + target (7, BIC-selected over the
  goal-radius region) Gaussian mixtures; `encode_runs` serves pure-numpy. Fail-closed load: SHA256 +
  chirality fingerprint + feature contract (`goal_radius_m`).
- `CornerRoutineModel` (NMF): `_routine_weights/default` — rank-elbow basis (`alpha = 0.0`);
  orientation-NEUTRAL (no chirality — it factors already-oriented run vectors). Fail-closed load: SHA256.
- Both carry a `training_commit` stamp (fit on a clean tree, `require_clean_tree`, ADR-037) and a
  wheel `MODEL_CARD.md` (receiver pattern — wheel-bundled, not HF-published; the xy-encoder modality
  does not fit the frame-scorer Hub harness). Classified `frame_geometry` in `WEIGHTS_CLASSIFICATION`
  (ADR-106 float32 policy) and carried by the corner-scoped `tests/corners/test_bundled_weights.py`.

## sportec `corner_crossed` (ADR-127)

`sportec`/IDSSE `corner_crossed` discriminates via the DFL `play_height` qualifier (`high` → crossed),
replacing a fictional target/placing vocabulary that matched 0/57 corners. VALUE-CHANGING (~93% of DFL
corners flip `corner_short` → `corner_crossed`); SemVer minor; measured consumer blast radius (the
`type_corner_crossed`/`_short` one-hots, `defensive_credit` bravery) shifts toward correctness, no
retrain.

## Construct validity (§12)

`docs/research/corner_routines/` (README + `metrics.json`): the per-routine chance-quality metric
**discriminates teams on `n_shots` (p = 0) and survives net-of-team** (opponent leave-one-out +
routine fixed-effect) — a corner shot-generation instrument, not a team-quality ranking. `n_goals` /
`sum_xg` are underpowered; season-to-season transfer is not established. Reported, not gated; §12.6
convergent timing (TF-79 vs GI) and §12.7 SB360 init-divergence bound the serve error.

## References

- Spec: `docs/superpowers/specs/2026-10-16-tf46-corner-offence-design.md`.
- Plan: `docs/superpowers/plans/2026-10-16-tf46-corner-offence.md`.
- ADRs: ADR-110 (raw primitives), ADR-011 (trained-model lifecycle), ADR-037 (clean-tree provenance),
  ADR-127 (sportec corner-crossed). `NOTICE`: Shaw & Gopaladesikan attribution (ADR-005).
