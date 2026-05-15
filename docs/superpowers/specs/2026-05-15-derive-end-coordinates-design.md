# Derive End Coordinates + GK Fallback for Single-Position Providers

## Goal

Two correctness fixes for single-position providers (DFL/Sportec, Gradient Sports):

1. **Bug #7 (end coordinates):** Fix `end_x == start_x` for pass-class SPADL actions. Replace the existing `_fix_clearances()` with a shared, period-safe `_derive_end_coordinates()` that benefits all 7 converters.
2. **Bug #2 (GK features NULL):** Wire `defending_gk_from_frames()` as a fallback in `add_pre_shot_gk_context()` so shots on providers lacking keeper_save events still get `defending_gk_player_id` populated from tracking data.

## Background

DFL XML events provide one position per event. The Sportec converter (`sportec.py:958-961`) sets `end_x = rows["x"]` and `end_y = rows["y"]`, producing identical start and end coordinates for every action. Two existing functions already solve this for specific types using the "next-event" pattern:

- `_fix_clearances()` in `base.py` — sets clearance `end_x/end_y` from next action's start position
- `_add_dribbles()` in `base.py` — synthesizes dribbles with `end = next.start`

The Gradient Sports converter has a similar data limitation (single `ball_x/ball_y` per event) and already applies next-event end derivation at `gradientsports.py:546-559`, but indiscriminately to ALL action types including shots, tackles, and keeper saves where `end = start` is semantically correct.

### Empirical validation (IDSSE fixture, match J03WMX Databricks probe)

| Action type | Count | % with start==end |
|---|---|---|
| pass | 49 | 100% |
| cross | 5,307 | 100% |
| throw_in | 308 | 100% |
| shot | 164 | 100% |
| tackle | 1,412 | 100% |
| clearance | 159 | 0% (fixed by `_fix_clearances`) |
| dribble | 4,620 | 0% (synthesized by `_add_dribbles`) |

### Downstream features broken

| Feature | Module | Mechanism |
|---|---|---|
| `line_break__ward` | `_line_breaking.py:189` | `pass_len = 0` -> filtered by `min_pass_length` -> always FALSE |
| `receiver_zone_density` | `features.py:307` | Counts defenders at `end_x/end_y` which equals origin |
| `line_break` (threshold) | `_off_ball_runs.py:306` | `end_x > spadl_def_line_x` uses wrong position |
| VAEP delta features | downstream | `end_x - start_x` always 0 |
| xT delta | downstream | `xt(end) - xt(start)` always 0 |

### SkillCorner status

SkillCorner's source data (`dynamic_events.csv`) provides explicit `x_end/y_end` and `player_targeted_x_reception` columns. Primary actions already have correct end coordinates (0% start==end on passes). Defensive actions and keeper saves correctly set `end = start` (in-place actions). No end-coordinate derivation needed. SkillCorner benefits only from the `_fix_clearances` period-safety fix (shared via `base.py`).

## Design

### New shared function: `_derive_end_coordinates()` in `base.py`

Replaces `_fix_clearances()`. Derives `end_x/end_y` from the next action's `start_x/start_y` for pass-class action types where the ball physically travels to a different location.

#### Source-data guard (lakehouse review issue #1)

The function MUST NOT overwrite end coordinates already provided by the source data. Providers like StatsBomb (pass.end_location), Opta (native end_x/end_y), Wyscout (_make_new_positions), Metrica (native end_x/end_y), and SkillCorner (x_end/y_end) all supply explicit end coordinates.

**Guard condition:** Only derive end coordinates for rows where `end_x == start_x AND end_y == start_y`. This indicates the source did not provide a separate end coordinate (single-position provider pattern). Rows where the source already provided different end coordinates are left untouched.

```python
def _derive_end_coordinates(actions: pd.DataFrame) -> pd.DataFrame:
    if len(actions) == 0:
        return actions
    actions = actions.copy()

    # Only derive for pass-class types where source didn't provide end coords.
    needs_derivation = (
        actions["type_id"].isin(_DERIVE_END_TYPE_IDS)
        & (actions["end_x"] == actions["start_x"])
        & (actions["end_y"] == actions["start_y"])
    )

    # Period-safe next-action lookup.
    next_start_x = actions.groupby("period_id")["start_x"].shift(-1)
    next_start_y = actions.groupby("period_id")["start_y"].shift(-1)

    # Apply: rows matching the guard get next-event start as their end.
    # NaN from shift (last action per period) -> no overwrite (needs_derivation
    # is True but next_start_x is NaN -> loc assignment writes NaN, so we
    # must only assign where next values are available).
    mask = needs_derivation & next_start_x.notna()
    actions.loc[mask, "end_x"] = next_start_x[mask].values
    actions.loc[mask, "end_y"] = next_start_y[mask].values
    return actions
```

This makes `_derive_end_coordinates` safe to call from ALL converters. Providers with explicit end coordinates are untouched; single-position providers get the derivation.

#### Action type classification

**Derive end from next event** (`_DERIVE_END_TYPE_IDS`):

| Type | ID | Rationale |
|---|---|---|
| pass | 0 | Ball travels to receiver/interceptor |
| cross | 1 | Ball travels to target area |
| throw_in | 2 | Ball travels to receiver |
| freekick_crossed | 3 | Ball travels to target area |
| freekick_short | 4 | Ball travels to receiver |
| corner_crossed | 5 | Ball travels to target area |
| corner_short | 6 | Ball travels to receiver |
| clearance | 18 | Ball travels away from goal |
| goalkick | 22 | Ball travels to receiver |

**Keep end = start** (excluded from derivation):

| Type | Rationale |
|---|---|
| shot, shot_penalty, shot_freekick | Shooting position is the coordinate; next event is GK save or restart, not ball destination |
| take_on | In-place action |
| foul | In-place action |
| tackle | In-place action |
| interception | In-place action |
| keeper_save, keeper_claim, keeper_punch, keeper_pick_up | In-place keeper action |
| bad_touch | In-place action |
| dribble | Synthesized separately with correct coordinates |
| non_action | Filtered out |

#### Period boundary safety

Uses `groupby("period_id")` + `shift(-1)` instead of bare `shift(-1)`. Last action per period keeps `end = start` (NaN from shift -> no overwrite). This fixes the cross-period contamination bug in the current `_fix_clearances()` which uses bare `shift(-1)` without period guard.

Note on `_add_dribbles`: it also uses bare `shift(-1)` at `base.py:29`, but has an explicit `same_period = actions.period_id == next_actions.period_id` filter at line 45 that catches cross-period dribbles. The unguarded shift computes dx/dy across period boundaries, but the filter prevents spurious dribble synthesis. This is benign and does not need changing.

#### Coordinate frame

Must run pre-LTR (same coordinate frame as the existing `_fix_clearances`). All events in the same period share DFL-centered absolute coordinates, so within-period shift is safe.

Exception: Gradient Sports converter currently runs its derivation post-LTR. The fix will move it to use the shared function called pre-LTR (before `to_spadl_ltr`), matching the Sportec pattern. See Gradient Sports section below for foul-synthesis ordering concern.

### Per-converter changes

#### Sportec (`sportec.py`)

Insert `_derive_end_coordinates()` call at line 656, replacing `_fix_clearances()`:

```
raw_actions = _build_raw_actions(events, ...)
actions = _derive_end_coordinates(raw_actions)   # was: _fix_clearances(raw_actions)
actions = to_spadl_ltr(actions, ...)
actions = _add_dribbles(actions)
```

#### Gradient Sports (`gradientsports.py`)

Replace the indiscriminate post-LTR next-event block (lines 542-559) with a pre-LTR call to `_derive_end_coordinates()`.

**Foul synthesis ordering (lakehouse review issue #3):** The GS converter synthesizes foul rows at lines 484-526 by interleaving them immediately after their parent action (0.5-offset sort key). If `_derive_end_coordinates` runs after foul synthesis, a synthesized foul row between pass A and action B means A's `shift(-1)` sees the foul's `start_x` (which equals A's `start_x`, since fouls are in-place at the parent's position). A would get `end_x = foul.start_x = A.start_x` -- defeating the derivation.

**Fix:** Call `_derive_end_coordinates` BEFORE foul synthesis (before line 484). At that point, all rows have their final `type_id` from the dispatch table (except in-place foul conversions at line 508 which change non_action -> foul; non_action is not in `_DERIVE_END_TYPE_IDS` so this is harmless). The synthesized foul rows don't exist yet, so the shift chain sees the real next action.

```
# ... dispatch table produces actions with type_id set ...
actions = _derive_end_coordinates(actions)   # BEFORE foul synthesis
# ... foul synthesis block (lines 484-526) ...
actions = to_spadl_ltr(actions, ...)
# DELETE: the old indiscriminate post-LTR block at lines 546-559
```

Note: Gradient Sports does not call `_fix_clearances` or `_add_dribbles` currently. The new `_derive_end_coordinates` handles clearances. Dribbles are mapped natively from source data.

#### SkillCorner (`skillcorner.py`)

No change to end coordinate logic (source data provides explicit end coords). The source-data guard (`end_x == start_x`) ensures `_derive_end_coordinates` is a no-op for passes/crosses/etc. that already have correct end coords. Benefits from `_fix_clearances` -> `_derive_end_coordinates` replacement at line 415 for clearance period-safety.

#### All other converters (StatsBomb, Opta, Wyscout, Metrica, Kloppy gateway)

Replace `_fix_clearances(actions)` import and call with `_derive_end_coordinates(actions)`. The source-data guard (`end_x == start_x AND end_y == start_y`) ensures source-provided end coordinates are never overwritten. Only clearances (which have `end = start` in source data and are in the type set) benefit from the derivation. The period-safety fix is the main benefit for these providers.

### Impact on `_add_dribbles`

After the fix, for pass-class actions on single-position providers: `actions.end_x = next_actions.start_x`, so `dx = actions.end_x - next_actions.start_x` is approximately 0 -> `far_enough = False` -> no spurious dribble synthesized between a pass and its reception.

On the IDSSE `per_period_match.parquet` fixture (verified): 639 of 708 current dribbles are spurious (follow same-team pass-class action where `end_x == start_x`). These will be correctly eliminated. No code change needed in `_add_dribbles`.

### `_fix_clearances` removal

`_fix_clearances` is deleted entirely from `base.py`. Its functionality is subsumed by `_derive_end_coordinates` with clearance in the type set, plus the period-safety fix and the source-data guard. All 7 converters that import it switch to the new function.

### Backward compatibility

This is a **correctness fix**, not a breaking API change. The function signature for the replacement is identical (`DataFrame -> DataFrame`). Output changes:

- Sportec/Gradient Sports: pass-class actions get correct `end_x/end_y` (previously identical to start)
- All providers: clearance end derivation gains period-boundary safety
- All providers with source end coords: source values preserved (guard prevents overwrite)
- Sportec: spurious dribble count decreases (correct behavior)
- Hyrum's Law note: any consumer depending on `end_x == start_x` for Sportec pass actions was depending on broken behavior
- **Guard false positives (0.02-0.05%):** StatsBomb (718 / 2.9M = 0.02%) and Wyscout (832 / 1.6M = 0.05%) have a small number of passes with genuine source-provided `end == start` (zero-length passes: intercepted at passer's feet or coordinate-precision artifacts). The guard matches these and overwrites with `next.start`. This is arguably more informative than the original zero-length annotation, but is a subtle behavioral change.
- **Clearance guard effect:** For providers with source-provided clearance end coordinates (StatsBomb: 99.6% of clearances have `end != start`; Wyscout: 99.7%), the guard preserves source values instead of unconditionally overwriting with next-event position as `_fix_clearances` did. This is more correct than the prior behavior — source-annotated clearance destinations are more accurate than next-event start positions.

### Version

silly-kicks 3.15.0 (new feature: derived end coordinates for pass-class actions on single-position providers + GK fallback wiring).

## Bug #2: GK Fallback in `add_pre_shot_gk_context`

### Root cause

DFL event data does not produce `keeper_save` or `keeper_claim` SPADL actions in the normal sense. DFL's `BallClaiming` event type exists but never appears within the lookback window (5 actions / 10 seconds) of any shot. Empirically, 0 of 20 shots in match J03WMX have a BallClaiming event within +/-10 seconds.

The events-based backward lookback loop in `add_pre_shot_gk_context()` (`utils.py:656-700`) searches for `_GK_KEEPER_TYPE_NAMES` (keeper_save, keeper_claim, keeper_punch, keeper_pick_up) in the preceding window. When none are found, `defending_gk_player_id` stays NaN. This cascades:

1. `defending_gk_player_id` = NaN for all shots
2. `_resolve_action_frame_context` in `tracking/utils.py:310-322` matches `frame.player_id == action.defending_gk_player_id` -- empty when NaN
3. `_pre_shot_gk_position()` in `_kernels.py:194-253` receives empty `ctx.defending_gk_rows` -- all 6 GK positional features NaN

### Fix

Insert a `defending_gk_from_frames()` fallback between the events-based resolution (line 705) and the tracking feature import (line 710) in `add_pre_shot_gk_context()`:

```python
sorted_actions["defending_gk_player_id"] = defending_gk_player_id

# Fallback: fill NaN defending_gk_player_id from tracking frames.
# DFL/Sportec events rarely produce keeper_save actions, so the
# events-based lookback above leaves most shots with NaN.  The
# frame-based resolver finds the opposing team's is_goalkeeper=True
# row in the nearest tracking frame.
if frames is not None:
    from silly_kicks.tracking._gk_resolve import defending_gk_from_frames

    gk_series = defending_gk_from_frames(sorted_actions, frames)
    sorted_actions["defending_gk_player_id"] = (
        sorted_actions["defending_gk_player_id"].fillna(gk_series)
    )

# PR-S21: when tracking frames supplied, lazy-import + merge GK-position columns.
if frames is not None:
    ...
```

Note: `defending_gk_from_frames()` does not accept a `links` parameter (its signature is `(actions, frames, *, tolerance_seconds=0.2)`). It calls `link_actions_to_frames` internally. The extra linking call costs ~5ms on typical data sizes and is acceptable. The links computed later by `add_pre_shot_gk_position` are separate.

The `defending_gk_from_frames()` function already exists (TF-13, PR-S27, `tracking/_gk_resolve.py`). It links each action to the nearest tracking frame, finds the opposing team's `is_goalkeeper=True` player, and returns a Series of player_ids. The `.fillna()` pattern is documented in its own docstring.

### Impact

- Sportec/IDSSE: shots go from 0% to ~100% `defending_gk_player_id` populated (when tracking frames are supplied)
- All 6 GK positional/angle features (`pre_shot_gk_x`, `pre_shot_gk_y`, `pre_shot_gk_distance_to_goal`, `pre_shot_gk_distance_to_shot`, `pre_shot_gk_angle_to_shot_trajectory`, `pre_shot_gk_angle_off_goal_line`) become non-NaN for shots with nearby tracking frames
- Providers with good events-based coverage (StatsBomb, Opta, Wyscout) are unaffected -- their `defending_gk_player_id` is already populated, so `.fillna()` is a no-op
- SkillCorner and Metrica may see marginal improvement for edge-case shots where events-based lookback missed

### Backward compatibility

Additive only. Shots that previously had NaN now get a value; shots that already had a value are unchanged (`.fillna()` preserves existing non-NaN).

## Test Fixtures

### Paired IDSSE events + tracking fixture

The existing IDSSE events fixture (`per_period_match.parquet`, 1,715 events from match J03WMX) is already committed. A new paired tracking fixture (`paired_tracking.parquet`, ~18K rows, ~518 KB) has been extracted from the lakehouse covering two time windows:

| Window | Period | Timestamp range | Events covered | Purpose |
|--------|--------|-----------------|----------------|---------|
| 1 | P1 | 90.0 - 107.0s | 3 Play + 1 ShotAtGoal | GK fallback test: shot has no nearby BallClaiming, tracking has is_goalkeeper=True for both GKs |
| 2 | P2 | 624.0 - 640.0s | 3 Play + 1 ThrowIn + 2 OtherBallAction + 1 TacklingGame + 1 ShotAtGoal | End-coordinate test: consecutive pass-class events with tracking coverage |

Both windows have 22 players with 2 identified GKs (`DFL-OBJ-0002DR` away, `DFL-OBJ-0002HE` home). The match_id is preserved as `J03WMX` (DFL DataHub free-sample license permits non-commercial redistribution, same as the events fixture).

Extraction script: `scripts/extract_paired_idsse_fixture.py`.

## Testing

### Unit tests for `_derive_end_coordinates`

1. **Pass-class types get next-event end**: synthetic 5-action sequence (pass, cross, throw_in, tackle, goalkick) -> pass/cross/throw_in/goalkick get next.start; tackle keeps end=start
2. **Period boundary safety**: last action of period 1 keeps end=start (not contaminated by period 2's first action)
3. **Shot exclusion**: shot action keeps end=start even when next event exists
4. **Clearance inclusion**: clearance gets next-event end (subsumes old `_fix_clearances`)
5. **Empty DataFrame**: no crash on empty input
6. **Single action**: single action keeps end=start (shift produces NaN)
7. **Source-data guard**: action with `end_x != start_x` (source provided end coords) is NOT overwritten even if type_id is in the derive set

### Integration tests per converter (Bug #7)

1. **Sportec/IDSSE**: convert IDSSE `per_period_match.parquet` fixture, assert pass-class actions have `end_x != start_x` for the majority (period-boundary last actions excepted)
2. **Gradient Sports**: convert GS fixture, assert shots have `end_x == start_x`, passes have `end_x != start_x`
3. **SkillCorner**: convert SC fixture, assert passes still have correct end coords from source data (regression guard)

### Source-data preservation regression tests (lakehouse review issue #5)

4. **StatsBomb**: convert WC2018 fixture, assert pass end coordinates match source `pass.end_location` (not overwritten by next-event start)
5. **Opta/Wyscout/Metrica**: convert existing provider fixtures, assert pass `end_x != start_x` actions are unchanged (source values preserved)
6. **StatsBomb clearance guard**: convert WC2018 fixture, find clearances with `end_x != start_x` (source-provided), assert end coordinates match source (not overwritten by next-event). Validates the more-conservative behavior vs old `_fix_clearances`.

### Dribble count regression test (Bug #7)

Assert Sportec IDSSE dribble count decreases after fix. Verified baseline: 708 dribbles pre-fix, 639 are spurious (follow same-team pass-class action where `end_x == start_x`). Post-fix: expect ~69 legitimate dribbles.

### GK fallback integration test (Bug #2)

Using paired IDSSE events (`per_period_match.parquet`) + tracking (`paired_tracking.parquet`):

1. Convert events to SPADL actions
2. Call `add_pre_shot_gk_context(actions, frames=tracking_frames)`
3. Assert shots within the tracking time windows have non-NaN `defending_gk_player_id`
4. Assert the resolved GK is from the opposing team (home shot -> away GK `DFL-OBJ-0002DR`, away shot -> home GK `DFL-OBJ-0002HE`)
5. Assert GK positional features (`pre_shot_gk_x`, `pre_shot_gk_y`, `pre_shot_gk_distance_to_goal`, `pre_shot_gk_distance_to_shot`) are non-NaN for those shots

### Existing test suite

- `test_line_breaking.py::test_zero_length_pass_returns_false` remains valid (documents the min_pass_length guard)
- Run full `pytest tests/ -m "not e2e"` to verify no regressions
