# ADR-028: Tracking geometry is emitted in the per-action SPADL LTR frame (centralized re-projection)

| Field | Value |
|---|---|
| **Date** | 2026-06-12 |
| **Status** | Accepted |
| **Deciders** | Karsten Nielsen, Claude (luxury-lakehouse production report `silly_kicks_tracking_geometry_ltr_frame_20260612`) |

## Context

SPADL actions and tracking frames live in two **different** coordinate conventions:

- **Actions** (`spadl.orientation.to_spadl_ltr`): per-acting-team LTR. Each action is
  normalized so the *acting* team attacks x=105 (their own goal at x=0). The physical
  orientation differs per action depending on which team is acting — a 180° point
  reflection (`x→105−x, y→68−y`, per `_mirror_absolute_frame`) separates a home-team
  action's frame from an away-team action's frame.
- **Frames** (`tracking.convert_to_frames`, default `output_convention="absolute_frame"`):
  home-attacks-right. The *home* team attacks x=105 in **every** period; the away team
  attacks x=0. One consistent frame for the whole match. Every adapter populates a per-row
  `team_attacking_direction ∈ {"ltr","rtl"}` (home="ltr", away="rtl", None for ball).

The two agree for home-team actions and are 180°-mirrored for away-team actions. The
tracking-geometry layer sampled frame positions but **never re-projected** them into the
per-action LTR frame, so on every away-team action (~50% of rows):

1. **Absolute-position outputs** landed at the wrong end (visibly bimodal): `pre_shot_gk_x`
   showed a 50/50 split at x≈10 vs x≈100 with an empty middle; `pre_shot_gk_distance_to_goal`
   reached 106 m (a physically impossible GK→goal distance).
2. **Mixed-frame outputs** — features that combine an action-LTR anchor (`start_x/end_x`)
   with frame-coord positions — produced numerically wrong distances/counts that were NOT
   visibly bimodal and so escaped the lakehouse's notice: `nearest_defender_distance`
   (18 m "nearest" defender), `receiver_zone_density`, `defenders_in_triangle_to_goal`, all
   `pressure_on_actor__*` flavors, `pre_shot_gk_distance_to_shot`, `pre_shot_gk_angle_*`.

A single frame is shared by **both** teams' actions within a possession window; because the
two teams need opposite orientations, **the caller cannot fix this by globally orienting the
frames**. The re-projection must be per-action and library-owned.

## Decision

**Contract:** every emitted per-action tracking-geometry POSITION column is expressed in
the action-LTR frame of the action it annotates (acting team attacks x=105, defended goal
at x=105 for the defending team).

Implemented by **one** canonical helper (`tracking/_action_orientation.py`,
`acting_team_attacks_rtl` + `reproject_to_action_ltr`) applied at exactly three seams, plus
an emit-time transform for ghost-GK:

1. **`_resolve_action_frame_context`** (the shared `ActionFrameContext`): re-project the
   sampled `actor_rows` / `opposite_rows_per_action` / `defending_gk_rows` x/y per the
   acting team's direction. This fixes all 8 context kernels at once
   (`nearest_defender_distance`, `receiver_zone_density`, `defenders_in_triangle_to_goal`,
   `pre_shot_gk_position`, `pre_shot_gk_angle`, the 3 `pressure` flavors) — and makes their
   hardcoded goal at (105, 34) *correct*, because after re-projection the acting team
   genuinely attacks x=105.
2. **`_defensive_line_at_actions`**: re-project `defensive_line_x` / `back_line_high_x`
   (spans/counts are invariant).
3. **`add_team_shape` / `_team_shape_at_actions`**: re-project `centroid_x/y` and
   `defensive_line_height` (both teams). **`compute_team_shape` is additionally made
   orientation-aware** (deepest line nearest the defended goal via `team_attacking_direction`)
   so `defensive_line_height` / `inter_line_gap_*` are the team's *true* defensive line for
   both teams (was the min-x cluster for everyone → the away team's *advanced* line) and are
   mirror-invariant. The "ltr"/home path is byte-identical; only "rtl"/away teams change.
4. **`add_ghost_gk` emit**: the model stays goal-relative (defended goal at x=0; y in
   absolute-frame terms). At the action-coupling seam: `ghost_gk_x → 105 − gr_x` (uniform —
   `gr_x` already measures from the defended goal) and `ghost_gk_y → 68 − gr_y` for away
   actions / `gr_y` for home actions (the per-action flip; verified against the training
   target at `_ghost_gk.py`, which keeps `gk_y_gr = gk_y_raw`).

**Direction source.** The per-action flip is derived from the frame's
`team_attacking_direction`, not from `home_team_id` — it is the ground truth of "which way
does this team attack in these coordinates," is robust to any frame orientation, and needs
no new signature arguments on the older `add_*` functions. The id-valued join uses the
ADR-019 `align_join_keys` so a string-id provider does not silently mis-match.

**Why centralize at the frame-sampling seam, not the outputs.** The lakehouse report
proposed re-projecting the emitted *outputs*. That fixes only the visible absolute positions
and misses every mixed-frame scalar (the Type-2 distances/counts), which are numerically
wrong, not merely mis-oriented. Re-projecting the sampled frame rows *once* fixes both
classes uniformly and keeps the kernels' goal constants correct by construction.

**Left untouched (verified not to double-flip).** Features that already self-reconcile
orientation via their own `home_team_id`/`goal_x`/xT-grid flips, or are direction-invariant,
are not in the buggy set and do **not** consume `ActionFrameContext`: `structural_pass`,
`gk_influence`, `player_influence`, `cover_shadows`, `shape_graph`, `obso`, `space_creation`,
`das`, `pitch_control`, `pausa`, `xt_gk`.

## Consequences

- **VAEP/tracking-retrain trigger.** Emitted values change for ~50% of tracking-provider
  action rows (away-team actions) across the affected columns. The lakehouse re-materializes
  all tracking action-context and re-runs any model that consumes these features. `ghost_gk_x/y`
  flip frame (goal-relative → action-LTR), so the ghost-vs-actual-GK comparison becomes a
  same-frame subtraction. Away-team `team_shape_defensive_line_height` / `inter_line_gap_*`
  additionally change selection (true defensive line). Home-team values for the
  orientation-aware compute are byte-identical.
- **Durable guard.** `tests/tracking/test_action_ltr_mirror_invariance.py` asserts that the
  same physical situation yields identical action-LTR geometry whether the acting team
  attacks left or right (a physical mirror of the frame + swapped `team_attacking_direction`),
  across pre_shot_gk / defensive_line / team_shape / ghost_gk.
- **No C4 change.** No new aggregator, KDE backend, or trained model; the aggregator count
  is unchanged → C4-free.
- **Atomic mirror** inherits automatically (the atomic surface reuses the same context
  kernels and the tracking `add_ghost_gk`).

Attribution: none (a coordinate-frame correctness fix, no new published methodology).
