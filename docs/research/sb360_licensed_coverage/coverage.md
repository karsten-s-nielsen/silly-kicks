# SB360 licensed-corpus coverage

What the library produces on the **licensed** StatsBomb 360 corpus (30 matches). The companion to the open-data `../sb360_coverage/coverage.md`.

## Provenance

| | |
|---|---|
| Driver | `scripts/validate_sb360_licensed_corpus.py` |
| Generation | `100c80a1b37b40a6` |
| Matches | 30 attempted, 0 failed |
| Commit | `cf2f155011df60f22dd659cbc516f534e884a595` |
| Tree | clean |

Rendered from the committed `coverage.parquet`; licensed data is never committed.

## Frame-existence coverage (per GK-domain type)

| Type | matches | actions | frame-existence |
|---|---|---|---|
| `all` | 30 | 53106 | 0.907 |
| `shot_freekick` | 10 | 17 | 1.000 |
| `shot` | 30 | 877 | 0.987 |
| `keeper_save` | 30 | 199 | 0.975 |
| `cross` | 30 | 656 | 0.968 |
| `shot_penalty` | 6 | 16 | 0.750 |
| `goalkick` | 30 | 509 | 0.444 |

## Battery aggregator coverage

_Battery numbers are STRUCTURAL coverage (did the aggregator run + fraction populated on real freeze-frames), NOT tactical values -- they are synthetic-input hybrids; a coverage fraction is a denominator, never a signal (ADR-042)._

**31 of 230** battery columns are fully-NaN across the corpus (mean populated fraction 0) -- the velocity-derived, ADR-063 Tier-2-suppressed, constitutively-tracking, and SB360-anonymity (no persistent freeze-frame player identity, ADR-054) columns. `add_visible_area_coverage`-style coverage fractions are denominators, not signals.

<details><summary>The fully-NaN columns</summary>

- `add_action_context.actor_speed`
- `add_actor_pre_window.actor_arc_length_pre_window`
- `add_actor_pre_window.actor_displacement_pre_window`
- `add_cover_shadows.max_single_defender_player_id`
- `add_das.das_diff`
- `add_das.das_opponent`
- `add_das.das_team`
- `add_ghost_gk.ghost_gk_x`
- `add_ghost_gk.ghost_gk_y`
- `add_gk_influence.gk_closing_time_mean_s__six_yard_box`
- `add_gk_influence.gk_closing_time_min_s__six_yard_box`
- `add_gk_influence.gk_reachable_area_m2`
- `add_off_ball_context.max_off_ball_run_displacement_pre_window`
- `add_off_ball_context.mean_off_ball_run_speed_pre_window`
- `add_off_ball_runs.max_off_ball_run_displacement_pre_window`
- `add_off_ball_runs.mean_off_ball_run_speed_pre_window`
- `add_player_influence.actor_reachable_area_m2`
- `add_player_influence.reachable_area_diff`
- `add_player_influence.reachable_area_opponent`
- `add_player_influence.reachable_area_team`
- `add_press_commitment.press_commitment`
- `add_press_commitment.press_commitment_closing_speed`
- `add_shot_goalmouth.shot_crossing_y`
- `add_shot_goalmouth.shot_crossing_z`
- `add_shot_goalmouth.shot_on_target_derived`
- `add_shot_goalmouth.shot_time_to_goal_line`
- `add_shot_goalmouth.shot_z_profile`
- `add_space_creation.obso_epv_source`
- `add_space_creation.space_created_m2`
- `add_space_creation.space_denied_m2_opponent`
- `add_xcross_attempt.xcross_attempt`

</details>

## ADR-062 visibility companions

Per count feature: the source-token breakdown (row counts) and the mean observed fraction. _An observed fraction is a coverage denominator, not a signal (ADR-042)._ Fractions are UNWEIGHTED per-match means; the frame-existence table above is denominator-weighted -- do not cross-compare them as the same statistic.

| Feature | source | total rows |
|---|---|---|
| `defenders_in_triangle_to_goal` | `degenerate_polygon` | 0 |
| `defenders_in_triangle_to_goal` | `degenerate_region` | 0 |
| `defenders_in_triangle_to_goal` | `no_polygon` | 192 |
| `defenders_in_triangle_to_goal` | `observed` | 47715 |
| `defenders_in_triangle_to_goal` | `unlinked` | 5199 |
| `nearest_defender_distance` | `degenerate_polygon` | 0 |
| `nearest_defender_distance` | `degenerate_region` | 844 |
| `nearest_defender_distance` | `no_polygon` | 192 |
| `nearest_defender_distance` | `observed` | 46871 |
| `nearest_defender_distance` | `unlinked` | 5199 |
| `receiver_zone_density` | `degenerate_polygon` | 0 |
| `receiver_zone_density` | `degenerate_region` | 0 |
| `receiver_zone_density` | `no_polygon` | 192 |
| `receiver_zone_density` | `observed` | 47715 |
| `receiver_zone_density` | `unlinked` | 5199 |

| Feature | mean observed fraction |
|---|---|
| `defenders_in_triangle_to_goal` | 0.298 |
| `nearest_defender_distance` | 0.922 |
| `receiver_zone_density` | 0.782 |

## Pitch coverage, roster, raises

- **Observed pitch fraction** (real `visible_area`): mean 0.273, min 0.200, max 0.366 over 30 matches. _A coverage denominator, not a signal._
- **Pitch-coverage source tokens** (summed rows -- coverage counts, not signals): `degenerate_polygon` 0, `no_polygon` 192, `observed` 47715, `unlinked` 5199.
- **Roster keeper-resolution rate** (a coverage rate, not a signal): mean 1.000 over 30 matches.
- **Aggregators that raised** (an honest refusal, not a defect):
  - `add_space_creation`: 23 matches (freeze-frame carried only one team's players near the action).

## The 40 -> 31 fully-NaN lift

The 4.85.0 velocity-less lift (ADR-063) moved the fully-NaN battery count from **40** (prior state) to the **31** this parquet records: velocity-requiring pitch-control aggregators now serve the zero-velocity positional model on declared freeze-frames.

## Reading limits / reproducing

- The battery per-column numbers are structural coverage, not tactics (see the caveat above).
- Licensed data is never committed. Refresh the parquet with `python scripts/validate_sb360_licensed_corpus.py` (owner token required), then re-render with `python scripts/render_sb360_licensed_coverage.py`.

