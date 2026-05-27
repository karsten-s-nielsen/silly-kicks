# Snapshot-to-Tracking-Frames Converter — Design Spec

**Date:** 2026-05-27
**Status:** Approved
**PR:** TBD (PR-S61 candidate)
**Scope:** New `snapshot_to_tracking_frames` public API in `silly_kicks.tracking`

## Problem

The luxury-lakehouse action-context table planned a 3-tier enrichment model:

1. **Event-only** (StatsBomb, Wyscout) → game_state + GK resolution
2. **SB 360** (StatsBomb w/ freeze-frames) → above + subset of tracking chain
3. **Full tracking** (IDSSE, Metrica, SkillCorner, Gradient Sports) → full `add_*` chain

Tier 2 is blocked because all tracking `add_*` functions require the 20-column
`TRACKING_FRAMES_COLUMNS` schema from continuous tracking data. StatsBomb 360
freeze-frames are per-event snapshots (all visible player positions at event
time) that contain the positional data needed for single-frame features but are
not in the tracking schema format.

## Solution

A provider-agnostic converter that maps per-event player-position snapshots into
the standard 20-column tracking frame schema, plus pre-built linkage pointers.
Once converted, any single-frame `add_*` enrichment function works on
freeze-frame data without modification.

## Schema registration

The value `"snapshot"` must be added to
`TRACKING_CATEGORICAL_DOMAINS["source_provider"]` in `schema.py` so that
any downstream validation accepting the domain set recognises
snapshot-derived frames. This is part of the implementation, not a
caller responsibility.

## Module location

`silly_kicks/tracking/_snapshot.py` — new private module.
Public API re-exported from `silly_kicks.tracking`.

## Function signature

```python
def snapshot_to_tracking_frames(
    snapshots: pd.DataFrame,
    actions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert per-event player-position snapshots to tracking frame schema.

    Parameters
    ----------
    snapshots : pd.DataFrame
        One row per player per event. Required columns: action_id, team_id,
        is_goalkeeper, x, y. Optional: player_id (synthetic sequential int
        if absent). Coordinates must be in the current SPADL coordinate system.
    actions : pd.DataFrame
        SPADL actions DataFrame. Used to derive game_id, period_id,
        time_seconds, and ball position (start_x, start_y) per frame.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (frames, links) where:
        - frames: 20-column TRACKING_FRAMES_COLUMNS schema, one synthetic
          frame per action that has snapshot data.
        - links: Pre-built pointer DataFrame matching the
          link_actions_to_frames output contract (action_id, frame_id,
          time_offset_seconds=0.0, n_candidate_frames=1,
          link_quality_score=1.0).
    """
```

## Input contract — `snapshots` DataFrame

| Column          | Type         | Required | Notes                                        |
|-----------------|--------------|----------|----------------------------------------------|
| `action_id`     | int64        | yes      | Join key to SPADL actions                    |
| `team_id`       | int64/object | yes      | Player's team (caller resolves from provider-specific flags) |
| `player_id`     | int64/object | no       | Synthetic sequential int if absent           |
| `is_goalkeeper`  | bool         | yes      | Caller resolves from provider-specific flags |
| `x`             | float64      | yes      | SPADL coordinates (current system)           |
| `y`             | float64      | yes      | SPADL coordinates (current system)           |

**Not accepted as input:** `teammate`, `actor`, `keeper`, `visible_area`,
`location` — these are provider-specific. The caller maps them to the
schema above before calling.

## Output — `frames` DataFrame

Standard 20-column `TRACKING_FRAMES_COLUMNS`. Column mapping:

| Output column              | Source                                                     |
|----------------------------|------------------------------------------------------------|
| `game_id`                  | From `actions` join on `action_id`                         |
| `period_id`                | From `actions` join on `action_id`                         |
| `frame_id`                 | `action_id` (1:1 mapping)                                  |
| `time_seconds`             | From `actions` join on `action_id`                         |
| `frame_rate`               | `NaN` (snapshot, not continuous)                           |
| `player_id`                | From snapshots, or synthetic sequential int (see dtype note) |
| `team_id`                  | From snapshots (dtype preserved — see dtype note)          |
| `is_ball`                  | `False` for player rows; one `True` ball row per frame     |
| `is_goalkeeper`            | From snapshots                                             |
| `x`, `y`                   | Passthrough from snapshots                                 |
| `z`                        | `NaN`                                                      |
| `speed`                    | `NaN` (single snapshot, no velocity)                       |
| `speed_source`             | `NaN`                                                      |
| `ball_state`               | `"alive"` (see ball_state note)                            |
| `team_attacking_direction` | `"ltr"` (SPADL actions are LTR-normalized)                 |
| `confidence`               | `NaN`                                                      |
| `visibility`               | `NaN`                                                      |
| `source_provider`          | `"snapshot"`                                               |
| `is_goalkeeper_source`     | `"native"`                                                 |

### Dtype handling

The converter preserves input dtypes for `game_id`, `player_id`, and
`team_id`. This matches the existing converter pattern where
`KLOPPY_TRACKING_FRAMES_COLUMNS` overrides these columns to `"object"`
dtype. When `snapshots` supplies string-typed identifiers, the output
frames will have string-typed identifiers; when int64, int64. No cast
is applied.

When `player_id` is absent and synthetic sequential IDs are generated,
they are emitted as `int64`.

### Ball row

One synthetic ball row per frame, with `x`/`y` from the action's
`start_x`/`start_y`, `is_ball=True`, `player_id=NaN`, `team_id=NaN`.
This ensures functions that filter `~is_ball` work correctly.

## Output — `links` DataFrame

Pre-built pointer DataFrame matching the `link_actions_to_frames` output
contract. Since each frame is generated *from* a specific action, the link
is exact by construction — no temporal nearest-neighbor matching needed.

| Column                | Value                    |
|-----------------------|--------------------------|
| `action_id`           | From actions              |
| `frame_id`            | Same as `action_id`       |
| `time_offset_seconds` | `0.0` (exact match)       |
| `n_candidate_frames`  | `1`                       |
| `link_quality_score`  | `1.0` (perfect link)      |

Only actions present in `snapshots` appear in `links`. Actions without
snapshot data are excluded from both `frames` and `links`.

## Zero hardcoded coordinate constants

The converter contains no hardcoded pitch dimensions (`105.0`, `68.0`,
`52.5`, etc.). Coordinates pass through untouched from input to output.
This makes the converter forward-compatible with TF-38 (CDF center-origin
coordinate system, 4.0.0 breaking change) by construction.

## Downstream compatibility

### Works (single-frame features)

These functions consume player positions at a single moment and work
correctly on snapshot-derived frames:

- `add_line_break(method="ward")` — opponent positions at pass moment
- `add_team_shape` — formation snapshot
- `add_defensive_line` — back line at action moment
- `add_action_context` (mostly) — 3 of 4 columns work fully:
  `nearest_defender_distance`, `defenders_in_triangle_to_goal`,
  `receiver_zone_density` are purely positional. Only `actor_speed`
  degrades to NaN because it reads the `speed` column from the linked
  frame, which is NaN on snapshots.

### Gracefully degrades (temporal / velocity features)

These functions require multi-frame temporal windows, derived velocities
(`vx`/`vy`), or auxiliary columns not present on snapshot-derived
frames. On snapshot-derived frames they produce NaN/empty output, which
is the correct behavior for instantaneous data:

- `add_actor_pre_window` — needs frames before the action
- `add_off_ball_context` / off-ball runs — needs frame sequences
- `add_elastic_sync` — needs continuous frame stream
- `add_cover_shadows` — checks for `vx`/`vy` columns; returns `None`
  (→ NaN) when absent. The lane-control time-to-intercept model
  requires player velocities.
- `add_das` — requires `vx`/`vy` and `team_in_possession` columns;
  raises `ValueError` (caught by the wrapper → NaN + warning) when
  they are absent. Not Voronoi-based — uses accessible-space velocity
  projection.
- `pitch_control_at_action` — needs player velocities from frame deltas

## Known limitations

### Incomplete player data on freeze-frames

StatsBomb 360 freeze-frames only contain players visible to the broadcast
camera. A freeze-frame with 18 players does not mean 4 players are off the
pitch — it means 4 are outside the camera's field of view. The
`visible_area` polygon (available in SB360 data) describes the camera
coverage boundary.

The converter does not process or propagate `visible_area`. Downstream
functions assume complete player data (all 22 players) because that is what
full tracking provides. On freeze-frame-derived frames, features are
computed on the visible subset. This is acceptable for most single-frame
features:

- **Line-breaking:** If the defensive line is visible, intersection
  detection works correctly even with missing far-side players.
- **Cover shadows / blocking score:** Missing players undercount blocking
  lanes, but visible-area players produce valid lane geometry.
- **Team shape:** ConvexHull computed on visible players only — area and
  spread will be underestimates when players are off-camera.

A future enhancement could add a `n_visible_players` column to the
`links` DataFrame (alongside the existing `link_quality_score`,
`n_candidate_frames`, etc.). This would let downstream consumers gate
on player completeness without schema changes to the tracking frames
themselves. This is not in scope for this spec.

### `frame_id` collision with continuous tracking

The converter sets `frame_id = action_id` (1:1 mapping). If a caller
later concatenates snapshot-derived frames with continuous tracking
frames from the same game, `frame_id` values will collide (continuous
tracking typically uses provider-assigned integer frame numbers that
may overlap with SPADL `action_id` values).

This is acceptable because the two frame sources serve different
pipeline stages and are not designed to be concatenated into a single
DataFrame. The lakehouse enrichment model routes Tier 2 (freeze-frame)
and Tier 3 (full tracking) through separate branches. If a future use
case requires merging, the caller can namespace `frame_id` values
(e.g. offset or prefix) before concatenation.

### `ball_state` hardcoded to `"alive"`

All snapshot-derived frames set `ball_state="alive"`. StatsBomb 360
freeze-frames also exist for dead-ball events (free kicks before the
ball is in play, throw-ins, etc.), so this is technically inaccurate
for those events. This is acceptable because no current `add_*`
function gates on `ball_state` — it is informational metadata carried
through from continuous tracking providers. If a future feature needs
dead-ball awareness, the caller can derive it from the SPADL action
type (`type_name in {"freekick", "throw_in", "corner", "goalkick"}`)
rather than from the tracking frame.

### `home_team_id` resolution

Several downstream `add_*` functions require `home_team_id` as a
keyword-only parameter. The converter does not resolve or propagate
`home_team_id`. The caller (lakehouse, notebook, etc.) must provide it to
each `add_*` call from StatsBomb event metadata. This matches the existing
pattern — tracking converters also do not embed `home_team_id` in their
output.

## Testing

- **Schema validation:** output `frames` matches `TRACKING_FRAMES_COLUMNS`
  dtypes; output `links` matches `link_actions_to_frames` contract.
- **Ball rows:** one ball row per frame, position matches action's `start_x`/`start_y`.
- **Synthetic `player_id`:** when `player_id` absent from snapshots, synthetic
  sequential IDs are generated, unique per frame.
- **Empty input:** 0 snapshots → empty frames + empty links (both valid DataFrames
  with correct columns).
- **Partial coverage:** actions without snapshot data excluded from both outputs;
  actions with snapshot data are complete.
- **Links contract:** `time_offset_seconds=0.0`, `n_candidate_frames=1`,
  `link_quality_score=1.0` on all rows.
- **Downstream degradation:** `snapshot_to_tracking_frames` →
  `add_cover_shadows(links=links)` and `add_das(links=links)` both
  return all-NaN feature columns (not raise). Verifies that
  velocity-dependent features degrade gracefully on snapshot-derived
  frames.
- **Downstream works:** `snapshot_to_tracking_frames` →
  `add_line_break(method="ward", links=links)` → verify line-break
  detection matches equivalent hand-constructed tracking frames (moved
  from round-trip test for clarity).
- **actor_speed degradation:** `snapshot_to_tracking_frames` →
  `add_action_context(links=links)` → verify `actor_speed` is NaN
  while the other 3 columns (`nearest_defender_distance`,
  `defenders_in_triangle_to_goal`, `receiver_zone_density`) have
  valid values.
- **home_team_id omission:** calling `add_line_break(method="ward")`
  on snapshot-derived frames WITHOUT `home_team_id` raises `TypeError`.
  This locks the contract — `home_team_id` is the caller's
  responsibility, and nobody should "fix" this by making it optional
  with a wrong default.

### Synthetic fixture shape

Tests use a 3-action fixture: action 0 with 6 snapshot players (3 v 3,
one GK per side), action 1 with 0 snapshot players (partial coverage
test), action 2 with 4 snapshot players (2 v 2, `player_id` column
absent — synthetic ID test). This covers the three key branches
(normal, missing, no-player-id) in a single minimal fixture.

## Not in scope

- StatsBomb coordinate transform (120×80 → SPADL) — caller's responsibility.
- `teammate`/`actor`/`keeper` → `team_id`/`is_goalkeeper` resolution — caller's
  responsibility.
- `visible_area` polygon processing — see Known Limitations above.
- VAEP xfn factory — snapshot features route through existing
  `line_breaking_ward_xfns()`, `cover_shadow_xfns()`, etc. unchanged.
