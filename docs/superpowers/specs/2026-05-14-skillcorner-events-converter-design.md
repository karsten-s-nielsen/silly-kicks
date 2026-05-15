# SkillCorner Events SPADL Converter

**Date:** 2026-05-14
**Status:** Approved
**Scope:** New dedicated SPADL converter for SkillCorner event data (`dynamic_events.csv`)

---

## 1. Context

SkillCorner distributes event data alongside tracking data for every match. The event format is `dynamic_events.csv` — a 294-column CSV with 4 event types (`player_possession`, `on_ball_engagement`, `passing_option`, `off_ball_run`). This is fundamentally different from traditional action-per-row event models (StatsBomb, Opta, etc.): SkillCorner is possession-centric, and defensive actions must be inferred from cross-referencing possession changes with `on_ball_engagement` (OBE) rows.

silly-kicks already supports SkillCorner tracking data via the kloppy gateway (`silly_kicks.tracking.kloppy`), but has no SkillCorner event/SPADL converter. kloppy itself has no SkillCorner event deserializer — only tracking parsers.

The data source is [pining-for-the-data](https://github.com/karsten-s-nielsen/pining-for-the-data), which redistributes 10 A-League matches from SkillCorner open data (MIT license) via a mock REST API. All 10 matches have 4 artifacts each: `match.json`, `dynamic_events.csv`, `phases_of_play.csv`, `tracking.jsonl`.

## 2. Public API

Single entry point following the Gradient Sports dedicated DataFrame-input converter pattern:

```python
def convert_to_actions(
    events: pd.DataFrame,
    match_metadata: dict,
    *,
    preserve_native: bool = False,
) -> tuple[pd.DataFrame, ConversionReport]:
```

### Parameters

- **`events`** — full `dynamic_events.csv` as a DataFrame (all 294 columns; the converter selects what it needs). Caller loads the CSV; the converter does not do I/O.
- **`match_metadata`** — parsed `match.json` dict. Required keys: `pitch_length`, `pitch_width`, team information, period metadata.
- **`preserve_native`** — when `True`, attaches `original_event_id` (the SC `event_id`) as an extra column via `_finalize_output(..., extra_columns=["original_event_id"])`.

### Returns

`tuple[pd.DataFrame, ConversionReport]` — SPADL actions with `SKILLCORNER_SPADL_COLUMNS` schema + audit report.

## 3. Action Dispatch Table

Only `player_possession` rows are converted to native SPADL actions. The other 3 event types (`on_ball_engagement`, `passing_option`, `off_ball_run`) are either used as inference inputs or excluded.

### 3.1 Native Actions (from `player_possession` rows)

Dispatch uses two columns: `game_interruption_before` (with `_for` suffix identifying the set piece taker) and `end_type`. Set piece detection takes priority over `end_type` dispatch.

**Empirical `end_type` taxonomy** (all 10 A-League matches): `pass` (8585), `possession_loss` (491), `shot` (226), `clearance` (141), `foul_suffered` (104), `unknown` (19).

**Empirical `game_interruption_before` taxonomy** (`_for` variants only): `throw_in_for` (359), `free_kick_for` (175), `goal_kick_for` (86), `corner_for` (45), `goal_against` (28 — kickoff restart after conceding).

| Priority | SC Field Combination | SPADL Action Type | Result Logic |
|---|---|---|---|
| 1 | `game_interruption_before == "goal_kick_for"` | `goalkick` | success (always) |
| 1 | `game_interruption_before == "corner_for"` | `corner_crossed` / `corner_short` | heuristic: short if next action same team within 15m |
| 1 | `game_interruption_before == "throw_in_for"` | `throw_in` | success if next possession is same team |
| 1 | `game_interruption_before == "free_kick_for"` | `freekick_crossed` / `freekick_short` | same short-detection heuristic |
| 2 | `end_type == "shot"` | `shot` | success if `game_interruption_after == "goal_for"` |
| 2 | `end_type == "pass"` + cross detection | `cross` | success if next possession is same team |
| 2 | `end_type == "pass"` | `pass` | success if next possession is same team |
| 2 | `end_type == "clearance"` | `clearance` | success (always) |
| 2 | `end_type == "foul_suffered"` | `foul` | success (always; the fouled player's possession) |
| 3 | `end_type == "possession_loss"` | `non_action` | fail |
| 3 | `end_type == "unknown"` | `non_action` | fail |

### 3.2 Cross Detection

A pass is reclassified as a cross using native SC columns when available:

```
cross = (player_targeted_third_pass == "attacking_third") AND
        (player_targeted_channel_pass IN ("wide_left", "wide_right"))
```

Coverage: `player_targeted_third_pass` / `player_targeted_channel_pass` are non-null on ~98% of passes (empirical: 887/902 in match 1886347). For the ~2% of passes missing these columns, fall back to the spatial heuristic: `start_x > 70.0` (attacking third in SPADL frame) AND wide channel (`start_y < 15.0` or `start_y > 53.0`).

### 3.3 Excluded Event Types

| Event Type | Reason |
|---|---|
| `on_ball_engagement` | Used as inference input for derived actions, not directly converted |
| `passing_option` | Off-ball positioning data, no SPADL analog |
| `off_ball_run` | Off-ball movement data, no SPADL analog |

Within `player_possession`, rows with null `player_id` or `team_id` are excluded as a defensive guard (empirically 0/999 in match 1886347, but retained for robustness against future data variations).

**`_against` variants in `game_interruption_before`** (`throw_in_against`, `free_kick_against`, `goal_kick_against`, `corner_against`, `goal_against`): these mark the first possession after the opponent's set piece or restart. They do NOT identify a set piece taker — the row is just a normal possession. Dispatch falls through to `end_type`-based mapping.

## 4. Derived Actions

**Dual-action production:** A single `player_possession` row may produce both a derived action (from `start_type`, this section) and a native action (from `end_type`, Section 3). When both fire, the derived action is ordered immediately before the native action. Example: a `pass_interception` row with `end_type == "pass"` produces `interception` (derived) followed by `pass` (native). Empirical: 78/89 interception rows also produce a native pass.

Three categories of actions are synthesized. All derived actions are flagged with `action_provenance = "derived"`.

### 4.1 Defensive Actions — Two-Source Strategy

Defensive actions use the `start_type` column on `player_possession` as the **primary** source, with OBE cross-referencing as a **secondary** enrichment for tackles.

**Primary source: `start_type` column (on `player_possession`)**

The `start_type` column directly identifies how each possession began. Empirical taxonomy (match 1886347, representative):

| `start_type` | Count | SPADL Mapping |
|---|---|---|
| `pass_reception` | 707 | Not a defensive action (ball receipt) |
| `pass_interception` | 89 | `interception` (success) |
| `recovery` | 61 | `interception` (success) — ball recovery from loose play; 100% involve a team change |
| `keep_possession` | 37 | Not a defensive action (dribble continuation) |
| `throw_in_reception` | 37 | Not a defensive action |
| `unknown` | 24 | Skip |
| `free_kick_reception` | 20 | Not a defensive action |
| `goal_kick_reception` | 13 | Not a defensive action |
| `throw_in_interception` | 4 | `interception` (success) |
| `corner_interception` | 2 | `interception` (success) |
| `goal_kick_interception` | 1 | `interception` (success) |
| `free_kick_interception` | 1 | `interception` (success) |

Any `start_type` ending in `_interception` OR equal to `recovery` maps to a derived `interception` action. The interception is attributed to the player/team on the `player_possession` row (the intercepting/recovering player). Coordinates from `x_start`/`y_start` of that row. Rationale: both interceptions and recoveries involve a team change (100% empirically) and represent genuine possession wins — mapping to `non_action` would make them VAEP-invisible.

**Secondary source: OBE cross-referencing (for tackles)**

`start_type` does not distinguish tackles from other possession wins. To identify tackles, the converter cross-references `on_ball_engagement` rows. The OBE `end_type` column (NOT `engagement_type`, which does not exist) provides the signal:

| OBE `end_type` | Count (match 1886347) | Meaning |
|---|---|---|
| `direct_regain` | 24 | Physical ball-winning — maps to `tackle` |
| `indirect_regain` | 120 | Ball-winning without direct contact — already captured by `start_type` interception |
| `direct_disruption` | 18 | Disrupted play but didn't win possession — skip |
| `indirect_disruption` | 44 | Indirect disruption — skip |
| `foul_committed` | 13 | Defender fouled — already captured by `end_type == "foul_suffered"` on PP |
| NaN | 718 | No outcome — skip |

When a `player_possession` row has `start_type` indicating a possession win (`*_interception` or `recovery`) AND a temporally adjacent OBE row (within 2 seconds) has `end_type == "direct_regain"`, the action is upgraded from `interception` to `tackle`. The OBE player/coordinates are used for the tackle action (the defender who made the physical challenge). Empirical: 9/89 interceptions and 11/61 recoveries have OBE `direct_regain` overlap in match 1886347.

If no matching OBE is found, the `start_type`-based mapping stands. The `ConversionReport` tracks counts of OBE-enriched vs start_type-only defensive actions.

### 4.2 Keeper Saves

When a `player_possession` follows a shot (`end_type == "shot"`) and the possession player is on the defending team:
- A `keeper_save` action is synthesized
- Player/team from the post-shot possession row
- Coordinates from the post-shot possession row
- Result: `success` (save made)

### 4.3 Synthetic Dribbles

Uses the existing `_add_dribbles()` utility from `silly_kicks.spadl.utils`. Dribbles are synthesized between consecutive same-team actions with spatial displacement. These also get `action_provenance = "derived"`.

## 5. Coordinate Transform

### 5.1 SC Coordinate Convention

SC uses **attacking-direction-normalized centered meters**: each row's `(x_start, y_start)` is in the attacking team's frame with origin at center spot. Positive x = toward the goal being attacked. This is `POSSESSION_PERSPECTIVE` — the same convention as StatsBomb/Wyscout. No direction flip is needed.

### 5.2 Pitch Dimension Variability

Empirical findings from all 10 A-League matches:
- Pitch lengths vary: **104m** (2 matches), **105m** (6 matches), **106m** (2 matches)
- Pitch width is consistently **68m**
- Coordinate ranges track `[-pitch_length/2, +pitch_length/2]` x `[-34, +34]`

### 5.3 Transform Formula

```python
half_length = pitch_length / 2    # from match_metadata
half_width  = pitch_width / 2     # from match_metadata

start_x = (x_start / half_length) * 52.5 + 52.5   # -> [0, 105]
start_y = (y_start / half_width)  * 34.0 + 34.0    # -> [0, 68]
```

This rescales to the canonical 105x68 SPADL frame regardless of actual pitch dimensions. Same approach as the kloppy tracking converter (which uses `MetricPitchDimensions` to map to 105x68).

A simple `+ 52.5` offset (as Gradient Sports uses) would be incorrect for non-standard pitches — coordinates would land outside `[0, 105]` or compress the scale.

### 5.4 End Coordinates

SC events provide multiple end-position columns with different semantics:

| Column | Semantics | Coverage |
|---|---|---|
| `x_end` / `y_end` | Ball carrier's position at end of possession | 100% non-null |
| `player_targeted_x_reception` / `player_targeted_y_reception` | Pass receiver's position (ball arrival point) | ~75-85% of passes |

**Key distinction:** `x_end` is where the PASSER ends up, not where the BALL goes. `player_targeted_x_reception` is where the ball arrives. Mean absolute difference: ~9m (max 63m). These are semantically different.

**End coordinate strategy by action type:**

| Action Type | `end_x` / `end_y` Source | Rationale |
|---|---|---|
| `pass` / `cross` | `player_targeted_x_reception` / `y_reception` (fallback: `x_end` / `y_end`) | Ball destination is the semantically correct SPADL end coordinate |
| `shot` | `x_end` / `y_end` | Shooter's end position (imperfect but better than duplicating start) |
| `clearance` | `x_end` / `y_end`, then `_fix_clearances` patches from next action | Consistent with other converters |
| `goalkick`, `throw_in`, `freekick_*`, `corner_*` | `player_targeted_x_reception` / `y_reception` (fallback: `x_end` / `y_end`) | Set piece delivery destination |
| `foul`, `non_action` | `x_end` / `y_end` | Player's end-of-possession position |

All end coordinates go through the same rescaling transform as start coordinates (Section 5.3).

### 5.5 Time Mapping

SC provides three time columns:

| Column | Type | Example | Precision |
|---|---|---|---|
| `time_start` | string | `"00:01.8"` | Sub-second (0.1s) |
| `minute_start` | int | `0` | Integer minutes |
| `second_start` | int | `1` | Integer seconds (loses the 0.8s fraction) |

**Use `time_start` (sub-second).** Parse `MM:SS.d` string as `time_seconds = minutes * 60 + seconds + fraction`. Integer-second fallback (`minute_start * 60 + second_start`) creates 19 same-second collisions per match (pairs of possessions appearing simultaneous), breaking the monotonic `time_seconds` invariant that Tier 2 tests assert. Sub-second parsing eliminates all collisions (0 in match 1886347, verified).

### 5.6 No Clipping

Coordinates slightly outside `[0, 105]` / `[0, 68]` (ball off-pitch during throw-ins, etc.) are preserved — consistent with all other silly-kicks converters.

### 5.7 LTR Normalization

The converter calls `to_spadl_ltr(actions, input_convention=POSSESSION_PERSPECTIVE, home_team_id=...)` which is a no-op copy (possession-perspective is already SPADL LTR after the coordinate shift).

## 6. Output Schema

### 6.1 Schema Constant

New constant in `schema.py`:

```python
SKILLCORNER_SPADL_COLUMNS: dict[str, str] = {
    **KLOPPY_SPADL_COLUMNS,           # object IDs (game_id, team_id, player_id)
    "action_provenance": "object",    # "native" | "derived"
}
```

Extends `KLOPPY_SPADL_COLUMNS` (object-dtype IDs) with one extra column.

### 6.2 Provenance Semantics

| Value | Meaning |
|---|---|
| `"native"` | Direct mapping from a `player_possession` row |
| `"derived"` | Synthesized action: `start_type`-based interception/recovery, OBE-enriched tackle, keeper save, or dribble |

### 6.3 ConversionReport

Standard `ConversionReport` return:

```python
ConversionReport(
    provider="skillcorner",
    total_events=<raw CSV rows (player_possession only)>,
    total_actions=<output action count>,
    mapped_events=<native action count>,
    excluded_events=<Counter of excluded event types>,
    unrecognized_events=<Counter of unmapped combos>,
)
```

## 7. Edge Cases & Known Limitations

### 7.1 Known Limitations

1. **Penalty kicks are invisible.** SC's `dynamic_events.csv` records the foul causing the PK and the subsequent kickoff, but the penalty kick itself has no `player_possession` row. The converter does NOT synthesize a PK action. Downstream xG models should be aware that PK shots are absent.

2. **No own goals.** SC events don't flag own goals. Goals map as `result_id = success` on shots.

### 7.1.1 Body Part Mapping

SC provides two boolean columns for body part detection:

| Column | True Count (match 1886347) | SPADL Mapping |
|---|---|---|
| `is_header` | 37/999 (3.7%) | `head` |
| `hand_pass` | 6/999 (0.6%) | `other` (GK throws/distributions) |
| neither | 956/999 | `foot` (default) |

Dispatch priority: `is_header` > `hand_pass` > default `foot`. Materially improves downstream xG/VAEP (headed shots have fundamentally different xG profiles).

### 7.2 Edge Cases Handled

1. **NaN coordinates** — rows with missing `x_start`/`y_start` propagate NaN through the transform (not 0.0).

2. **NaN `player_id`/`team_id`** — defensive guard excludes rows with null actor IDs (empirically 0% in the 10-match dataset, but retained for robustness).

3. **Extra-time periods** — SC `period = 3, 4` map directly to SPADL `period_id`. No special handling needed (possession-perspective convention, no per-period direction flip).

4. **Multiple possessions at same timestamp** — preserved as separate actions in CSV row order (SC guarantees chronological ordering).

5. **OBE with no matching possession change** — OBE enrichment silently skips when no temporally adjacent possession change is found; `start_type`-based mapping stands as the primary signal.

6. **`player_targeted_x_reception` missing on passes** — ~15-25% of passes lack the targeted reception coordinates. Falls back to `x_end`/`y_end` (passer's end position — less accurate but non-null).

## 8. Module Layout

### 8.1 File Structure

```
silly_kicks/spadl/
    skillcorner.py              # Public API + dispatch + coordinate transform
    _skillcorner_inference.py   # Derived action logic (OBE inference, keeper saves)
```

Two files keeps each under ~400 lines, makes inference logic independently testable, and is extensible to a subpackage later (file move, not rewrite).

### 8.2 Integration Points (changes to existing files)

| File | Change |
|---|---|
| `schema.py` | Add `SKILLCORNER_SPADL_COLUMNS` |
| `__init__.py` | Add `skillcorner` to public re-exports (lazy import) |

No changes to `orientation.py`, `utils.py`, or `config.py` — uses existing infrastructure as-is.

### 8.3 Dependencies on Existing Infrastructure

- `orientation.to_spadl_ltr` with `POSSESSION_PERSPECTIVE`
- `utils._add_dribbles`, `utils._fix_clearances`, `utils._finalize_output`
- `config.field_length`, `config.field_width`, action type / body part / result lookups
- `schema.ConversionReport`

## 9. Testing Strategy

### Tier 1 — Deterministic Synthetic Fixtures (CI)

Slim CSV fixtures in `tests/datasets/skillcorner/` with realistic values from A-League data:

- `basic_possessions.csv` — pass, shot, cross, clearance, foul, goal kick, corner, throw-in, free kick
- `obe_derived.csv` — OBE rows producing derived tackles/interceptions
- `keeper_saves.csv` — shot-to-GK sequence for keeper save inference
- `penalty_gap.csv` — foul with PK aftermath and no PK row
- `match_metadata.json` — minimal match JSON

Assertions: correct action type/result dispatch, provenance tagging, coordinate transform correctness (including 104m and 106m pitches), `ConversionReport` tallies, schema compliance, body part dispatch (`is_header` -> head, `hand_pass` -> other), end coordinate strategy (pass uses `player_targeted_x_reception` when available, fallback to `x_end`), `start_type`-based interception detection, OBE-enriched tackle upgrade, cross detection via native channel/third columns.

### Tier 2 — Full-Match Integration (CI, committed slim Parquet)

One real A-League match converted end-to-end (~50 KB Parquet). Assertions: no unrecognized events, expected SPADL action type coverage, monotonic `time_seconds` per period, no NaN in required columns.

### Tier 3 — e2e Against All 10 Matches (marked `e2e`)

All 10 matches from local path. Assertions: cross-match schema consistency, physical invariants (shot x > 52.5, GK x < 52.5, clearance x < 52.5), empty unrecognized counters, no crashes.

### Physical-Invariant Tests

In `tests/invariants/`: shot high-x, keeper save low-x, clearance low-x, coordinates within `[0, 105]` x `[0, 68]` (with tolerance).
