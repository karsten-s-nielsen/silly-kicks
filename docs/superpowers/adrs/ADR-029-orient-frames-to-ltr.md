# ADR-029: Frame-LTR orientation is single-sourced via `orient_frames_to_ltr`

| Field | Value |
|---|---|
| **Date** | 2026-06-13 |
| **Status** | Accepted |
| **Deciders** | Karsten Nielsen, Claude (luxury-lakehouse report `silly_kicks_metrica_skillcorner_ltr_frame_20260613`) |

## Context

ADR-028 (4.26.0) re-projects per-action tracking geometry into the per-acting-team
LTR frame, but only for frames already in the canonical home-attacks-right
convention. silly-kicks owns that convention for providers it converts via
`convert_to_frames(output_convention="ltr")` (sportec, gradientsports) and the kloppy
gateway (metrica, skillcorner). But a consumer holding **bronze DataFrames** (not a
kloppy `TrackingDataset`) cannot use the gateway and must build frames itself; the
lakehouse does exactly this for metrica/skillcorner, in **absolute** orientation
(`team_attacking_direction = None`, no per-period flip). ADR-028's reprojection
filters to `team_attacking_direction.notna()` rows, so on all-null frames it no-ops
and ~50% of action rows carry mirror-wrong geometry. Empirically (post-4.26.0 local
recompute): idsse/GS `pre_shot_gk_x` ~101 (clean); metrica 60.6, skillcorner 53.5
(bimodal).

The orientation logic already existed as two shared primitives
(`compute_attacking_direction` + `play_left_to_right`) but only `play_left_to_right`
(labeled-input) was public; nothing served the unlabeled-absolute case, so the
consumer re-implemented orientation incompletely.

## Decision

Add one public `silly_kicks.tracking.orient_frames_to_ltr(frames, *, home_team_id,
home_team_start_left, home_team_start_left_extratime=None)` that composes the existing
primitives (no new orientation math) with fail-loud preconditions:

- required-schema guard (raises on missing columns);
- already-labeled guard (raises if `team_attacking_direction` non-null -> use
  `play_left_to_right`), which also makes the helper non-idempotent-but-guarded;
- zero-match guard (raises if `home_team_id` matches no player row -- ADR-019);
- ET guard (raises on ET periods without the ET flag).

Two public entry points, by input state: **labeled** absolute frames ->
`play_left_to_right`; **unlabeled** absolute frames -> `orient_frames_to_ltr`. The
lower-level `compute_attacking_direction` stays private.

**Consumer contract:** any consumer building tracking frames from a non-kloppy source
MUST orient them (via `orient_frames_to_ltr` for unlabeled frames) into the
home-attacks-right convention before the per-action geometry layer (ADR-028). The
helper is only as correct as the caller-derived `home_team_start_left`; consumers MUST
validate that flag per game (e.g. assert each game's defending GK lands near the
attacked goal post-orient).

## Consequences

- Additive: no existing provider behaviour changes; no silly-kicks model retrain. The
  sportec/GS adapters and kloppy gateway are NOT refactored through the helper
  (primitives already shared; refactor risks goldens/retrain for zero gain).
- The lakehouse adopts the helper in its metrica/skillcorner bronze builders and
  re-materializes those providers (its consequence, not a bundled-model retrain).
- Decided against a native `metrica`/`skillcorner.convert_to_frames` (option a): it
  would duplicate the kloppy gateway and still cannot consume bronze DataFrames; TF-23
  already retired the metrica native loader.
- Extra-time orientation is regression-guarded on both the helper and the native
  adapters (`tests/tracking/test_adapter_extra_time_orientation.py`), prompted by a
  live GS-ET flip that was a consumer-side `home_team_start_left_extratime` placeholder
  bug, not a silly-kicks bug (the adapter ET path is correct given a correct flag).
