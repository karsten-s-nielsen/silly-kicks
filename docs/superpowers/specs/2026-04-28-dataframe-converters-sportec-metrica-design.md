# Dedicated DataFrame converters for Sportec + Metrica + kloppy direction-of-play unification

**Status:** Approved (design)
**Target release:** silly-kicks 1.7.0
**Author:** Claude Opus 4.7 (1M)
**Date:** 2026-04-28
**Predecessor:** 1.6.0 (`docs/superpowers/specs/2026-04-28-kloppy-sportec-metrica-design.md`)

---

## 1. Problem

silly-kicks 1.6.0 added Sportec / Metrica support **only via the kloppy path** (`silly_kicks.spadl.kloppy.convert_to_actions(EventDataset, ...)`). That covers consumers who start from raw provider XML/JSON. It does **not** cover consumers who already have normalized event data in pandas DataFrames — which is the typical lakehouse / ETL / research-notebook shape (e.g., luxury-lakehouse's `bronze.idsse_events` and `bronze.metrica_events`).

Reconstructing a kloppy `EventDataset` from already-flat rows is non-trivial domain-model reconstruction (Periods, Teams, Players, per-event-type Event subclasses with qualifiers). That work belongs once in silly-kicks, not in every consumer.

**Additionally**, an empirical probe during brainstorming revealed an inconsistency in silly-kicks itself:
- `silly_kicks/spadl/statsbomb.py:251`, `silly_kicks/spadl/opta.py:173`, `silly_kicks/spadl/wyscout.py:285` all call `_fix_direction_of_play(home_team_id)` to flip away-team coordinates so all actions are emitted in canonical SPADL "all-actions-LTR" orientation.
- `silly_kicks/spadl/kloppy.py` does NOT — it stays in kloppy's `Orientation.HOME_AWAY` (home plays LTR, away plays RTL).

This means the same Sportec match converted via the kloppy path versus a dedicated DataFrame path would produce coordinates in different orientations. The brief's claim "both paths produce identical SPADL output" is currently false. With zero consumers of 1.6.0 yet (luxury-lakehouse is waiting for 1.7.0), now is the time to fix it.

## 2. Goals

1. New `silly_kicks/spadl/sportec.py` exposing
   `convert_to_actions(events: pd.DataFrame, home_team_id: str, *, preserve_native: list[str] | None = None) -> tuple[pd.DataFrame, ConversionReport]`
   for normalized DFL bronze-shaped DataFrames.
2. New `silly_kicks/spadl/metrica.py` with the same shape for normalized Metrica events.
3. Modify `silly_kicks/spadl/kloppy.py` to apply `_fix_direction_of_play` (extracting home team from `dataset.metadata.teams[0].team_id`), aligning all 4+ silly-kicks converters on the canonical "all-actions-LTR" SPADL convention.
4. Empirically prove "both paths produce identical SPADL output" via cross-path consistency tests that run the same source data through both the kloppy path and the new dedicated path and assert the SPADL outputs match.

## 3. Non-goals

1. No multi-match input support — the new converters take a single match's events, matching the established statsbomb/wyscout pattern. Multi-match handling is the consumer's responsibility (`groupby("match_id").apply(...)`).
2. No Opta dedicated-path consolidation. Opta has its own dedicated `silly_kicks/spadl/opta.py` already; that's untouched here.
3. No new event types added to the SPADL action vocabulary. Provider events that don't map cleanly are dropped (recorded in `ConversionReport.excluded_counts` or `unrecognized_counts`).
4. No removal of `silly_kicks/spadl/kloppy.py`. Both paths (kloppy + dedicated) coexist in 1.7.0 — different consumer profiles need different entry points.
5. No new aerial-duel (`CHALLENGE` `AERIAL-WON`/`LOST`) handling for Metrica beyond drop. A future PR could route to `keeper_claim` for known GK player_ids (similar to wyscout's `goalkeeper_ids` kwarg) — out of scope here.
6. No DataFactory / StatsPerform / PFF / other providers in this PR. Strictly Sportec + Metrica.
7. No breaking changes to public APIs of `statsbomb`, `wyscout`, `opta`, or kloppy `convert_to_actions` signatures (kloppy's behavior changes — semver-pragmatic, documented in CHANGELOG `### Changed`).
8. No new `silly_kicks.atomic.spadl.sportec` / `metrica` atomic converters in this PR. The new dedicated converters emit canonical SPADL that can be projected to atomic via the existing `silly_kicks.atomic.spadl.convert_to_atomic(actions, ...)` helper (shipped in 1.5.0). Atomic-side parity for these providers, if needed, is a follow-up PR.

## 4. Architecture

### 4.1 File structure

```
silly_kicks/spadl/sportec.py       NEW  ~600-800 LOC
silly_kicks/spadl/metrica.py       NEW  ~400-500 LOC
silly_kicks/spadl/kloppy.py        MOD  +3-5 lines (extract home team, call _fix_direction_of_play)
CHANGELOG.md                       MOD  ## [1.7.0] entry
README.md                          MOD  Provider coverage list refresh
pyproject.toml                     MOD  version 1.6.0 -> 1.7.0
docs/c4/architecture.{dsl,html}    MOD  spadl container description refresh + regen
tests/datasets/sportec/            NEW  hand-crafted small fixtures (CSV or .py with literal DataFrames)
tests/datasets/metrica/            NEW  hand-crafted small fixtures (if needed; primary fixture is kloppy-derived)
tests/spadl/test_sportec.py        NEW  ~700-900 LOC
tests/spadl/test_metrica.py        NEW  ~400-600 LOC
tests/spadl/test_kloppy.py         MOD  +1 test (TestKloppyDirectionOfPlay::test_away_team_coords_flipped)
```

Single-file modules per provider (no `_sportec_events.py` / `_sportec_mappings.py` decomposition). Wyscout's split is historical complexity, not a precedent worth copying.

### 4.2 Public API (both new modules)

```python
def convert_to_actions(
    events: pd.DataFrame,
    home_team_id: str,
    *,
    preserve_native: list[str] | None = None,
) -> tuple[pd.DataFrame, ConversionReport]:
```

`home_team_id` type is `str` for both providers (DFL uses `"DFL-CLU-000017"`; Metrica uses `"Home"` / `"Away"` or numeric IDs cast to string). Output schema uses `KLOPPY_SPADL_COLUMNS` (string `team_id` / `player_id` / `game_id`) — matches the kloppy path's existing schema choice for these providers.

### 4.3 Pipeline (both new modules)

```
1. _validate_input_columns(events, REQUIRED_COLUMNS, provider="...")
2. _validate_preserve_native(events, preserve_native, provider="...", schema=KLOPPY_SPADL_COLUMNS)
3. counters = Counter(events[event_type_col])
4. raw_actions = _build_raw_actions(events)              # provider-specific dispatch
5. actions = _fix_clearances(raw_actions)
6. actions = _fix_direction_of_play(actions, home_team_id)
7. actions["action_id"] = range(len(actions))
8. actions = _add_dribbles(actions)
9. clamp coords to [0, 105] x [0, 68]                    # consistent with 1.6.0 kloppy
10. actions = _finalize_output(actions, schema=KLOPPY_SPADL_COLUMNS, extra_columns=preserve_native)
11. emit ConversionReport (mapped/excluded/unrecognized counts)
```

Coordinate clamping happens **after** `_fix_direction_of_play` — direction-flipping a value slightly outside the pitch (e.g., `x=-1.6`) and then clamping handles both off-pitch source data AND post-flip out-of-bounds.

### 4.4 kloppy.py modification (Option C — direction-of-play unification)

Insert extraction + flip immediately before the existing clamping block:

```python
# Extract home team from kloppy metadata. Orientation.HOME_AWAY puts home team first.
home_team_id = dataset.metadata.teams[0].team_id

# ... existing pipeline through _add_dribbles ...

# Flip away-team coords for canonical "all-actions-LTR" SPADL convention.
# Aligns this converter with the established statsbomb/wyscout/opta/sportec/metrica behavior.
df_actions = _fix_direction_of_play(df_actions, home_team_id)

# ... existing clamping and _finalize_output ...
```

**No public API change** to `silly_kicks.spadl.kloppy.convert_to_actions`. Behavior change is documented in CHANGELOG `### Changed` with explicit Hyrum's Law disclaimer (zero current consumers, validated with user during brainstorming).

## 5. Sportec event-type mapping (DFL → SPADL)

### 5.1 Required input columns

| Column | Dtype | Description |
|---|---|---|
| `match_id` | string | DFL match identifier (e.g., `"J03WMX"`) |
| `event_id` | string | Unique event identifier |
| `event_type` | string | DFL event-type name (`"Pass"`, `"ShotAtGoal"`, `"TacklingGame"`, `"Foul"`, `"FreeKick"`, `"Corner"`, `"ThrowIn"`, `"GoalKick"`, `"Substitution"`, `"Caution"`, `"Offside"`, `"Whistle"`, etc.) |
| `period` | int | Match period (1=first half, 2=second half, 3=ET first, 4=ET second) |
| `timestamp_seconds` | float | Seconds within the current period (period-local) |
| `player_id` | string | Player identifier (NULL for non-actor events) |
| `team` | string | Team identifier |
| `x` | float | X coordinate in meters in [0, 105] (clamped on output) |
| `y` | float | Y coordinate in meters in [0, 68] (clamped on output) |

### 5.2 Recognized qualifier columns

A module-level constant `_RECOGNIZED_QUALIFIER_COLUMNS: frozenset[str]` enumerates the optional qualifier columns the converter consults. Validation policy: required columns must be present; any subset of recognized qualifiers may be present; unknown columns are ignored unless listed in `preserve_native`. The full list is the union of qualifier columns in luxury-lakehouse's `bronze.idsse_events` schema (~120 columns covering pass / shot / tackle / foul / freekick / corner / throwin / goalkick / play / play context / cross / cards / substitution / penalty / VAR / chance / specialised / tracking-derived). Categorized in code with a comment per group.

### 5.3 Event-type → SPADL action dispatch

| DFL `event_type` | SPADL action(s) | Driving qualifiers |
|---|---|---|
| `Pass` | `pass` (default), `cross` (if `play_height='cross'` or `play_flat_cross='true'`) | `play_height`, `play_flat_cross`, body-part inference via `play_height='head'` |
| `ShotAtGoal` | `shot` (default); `shot_freekick` if `shot_after_free_kick='true'`; `shot_penalty` if any `penalty_*` qualifier present | `shot_outcome_type`, `shot_outcome_current_result`, `shot_after_free_kick`, `penalty_*` |
| `TacklingGame` | `tackle` (actor = `tackle_winner`/`tackle_winner_team` per existing precedence) | `tackle_winner_action`, `tackle_winner_result`, `tackle_dribble_evaluation` |
| `Foul` | `foul` (`fail`); upgraded to `result_id="yellow_card"` / `"red_card"` if a paired `Caution` event with same fouler exists in the next ≤ 3s | `foul_team_fouler`, `foul_fouler`; pair to `caution_card_color` |
| `FreeKick` | `freekick_short` (default), `freekick_crossed` (if `freekick_execution_mode` indicates long-pass / cross) | `freekick_execution_mode` |
| `Corner` | `corner_short` (default), `corner_crossed` (if `corner_target_area` indicates box / `corner_placing` indicates aerial) | `corner_target_area`, `corner_placing`, `corner_rotation` |
| `ThrowIn` | `throw_in` (bodypart=`other`) | — |
| `GoalKick` | `goalkick` | — |
| `Play` with `play_goal_keeper_action` set | `keeper_save` / `keeper_claim` / `keeper_punch` / `keeper_pick_up` per qualifier value | `play_goal_keeper_action` |
| `Substitution`, `Caution`, `Whistle`, `Offside`, `KickOff`, others | excluded (administrative or non-on-ball) | — |

**Body part inference order (first match wins):**
1. `play_height='head'` → `head`
2. (future: tracking-derived position-based head detection — out of scope here)
3. default → `foot`

**Owngoal handling:** `ShotAtGoal` with `shot_outcome_type` indicating own-goal is emitted as SPADL `bad_touch` with `result_id="owngoal"` — matches existing converter precedent for owngoals.

**Tackle actor resolution:** DFL `TacklingGame` events have BOTH a generic actor column (`player_id`/`team` from §5.1) AND winner/loser qualifier pairs (`tackle_winner`/`tackle_winner_team` / `tackle_loser`/`tackle_loser_team` from §5.2). The SPADL row uses `tackle_winner` / `tackle_winner_team` as the actor when present (the team that won the duel — semantically correct for SPADL `tackle`). If both are NULL on a given row, fall back to the generic `player_id`/`team` columns. Implementation: vectorized `np.where(tackle_winner.notna(), tackle_winner, player_id)` etc. The precedence is locally implemented in `sportec.py` — silly-kicks does not have a pre-existing `_PLAYER_ATTR_ORDER` helper for this (the brief referenced a luxury-lakehouse-side pattern).

## 6. Metrica event-type mapping

### 6.1 Required input columns

| Column | Dtype | Description |
|---|---|---|
| `match_id` | string | Match identifier (e.g., `"Sample_Game_1"`) |
| `event_id` | int or string | Unique event identifier |
| `type` | string | Metrica event type (uppercase: `"PASS"`, `"SHOT"`, `"RECOVERY"`, `"BALL OUT"`, `"BALL LOST"`, `"CHALLENGE"`, `"FAULT RECEIVED"`, `"FAULT"`, `"CARD"`, `"SUBSTITUTION"`, `"SET PIECE"`) |
| `subtype` | string (nullable) | Metrica subtype (varies by type — see §6.3 dispatch table) |
| `period` | int | Match period |
| `start_time_s` | float | Event start time in seconds (period-local) |
| `end_time_s` | float | Event end time in seconds (period-local) |
| `player` | string (nullable) | Primary actor player identifier |
| `team` | string | Team identifier (`"Home"` / `"Away"` for sample games) |
| `start_x` | float | Start X (meters) |
| `start_y` | float | Start Y (meters) |
| `end_x` | float | End X (meters) |
| `end_y` | float | End Y (meters) |

### 6.2 Optional columns recognized

`to`, `start_frame`, `end_frame`, `pitch_length_m` (default 105.0 if absent), `pitch_width_m` (default 68.0 if absent), `subtypes_all_json` (EPTS multi-subtype list — primary `subtype` is consulted first).

### 6.3 Event-type → SPADL action dispatch

| Metrica `type` | `subtype` (matched first by case-insensitive equality) | SPADL action(s) |
|---|---|---|
| `PASS` | (none / `HEAD` / `DEEP BALL` / `THROUGH BALL`) | `pass` (head bodypart if `HEAD`) |
| `PASS` | `CROSS` | `cross` |
| `PASS` | `GOAL KICK` | `goalkick` |
| `SHOT` | `ON TARGET` / `OFF TARGET` / `BLOCKED` / `WOODWORK` / `GOAL` | `shot` (default) — composition rule §6.4 may upgrade |
| `RECOVERY` | (none / various) | `interception` |
| `CHALLENGE` | `WON` | `tackle` (`success`) |
| `CHALLENGE` | `LOST` | excluded (winner's row carries the action) |
| `CHALLENGE` | `AERIAL-WON` / `AERIAL-LOST` | excluded (no SPADL aerial duel; v1) |
| `BALL LOST` | various | `bad_touch` (`fail`) |
| `BALL OUT` | various | excluded |
| `FAULT RECEIVED` | — | excluded (receiver perspective; foul is in the matching `FAULT` row) |
| `FAULT` | — | `foul` (`fail`); upgraded with paired `CARD` row's color if present in next ≤ 3s |
| `SET PIECE` | `FREE KICK` | `freekick_short` (default) — composition rule §6.4 may upgrade |
| `SET PIECE` | `CORNER KICK` | `corner_short` |
| `SET PIECE` | `THROW IN` | `throw_in` |
| `SET PIECE` | `GOAL KICK` | `goalkick` |
| `SET PIECE` | `KICK OFF` | excluded |
| `CARD` | — | excluded (paired to corresponding `FAULT` row's `result_id`) |
| `SUBSTITUTION` | — | excluded |

### 6.4 Set-piece-then-shot composition rule

If a `SET PIECE` row at index `i` is **immediately followed** at row `i+1` by a `SHOT` row in the **same period**, by the **same player**, within **≤ 5 seconds** (`start_time_s` delta), the shot row's SPADL action is upgraded:

| Preceding `SET PIECE` subtype | Upgraded `SHOT` action | SET PIECE row disposition |
|---|---|---|
| `FREE KICK` | `shot_freekick` | dropped (action carried by SHOT row) |
| `CORNER KICK` | `shot` (no upgrade — corners-then-shots stay as `shot`) | retained as `corner_short` action |

If no upgrade applies, both rows produce their own SPADL actions independently. Implementation: vectorized 1-row lookback. Penalty handling: Metrica's penalties show up as a `SHOT` immediately preceded by a synthetic context (no Metrica `PENALTY` event-type exists in the public sample); for v1 we don't synthesize `shot_penalty` from Metrica data — caller must opt-in via the data pipeline if needed.

## 7. Test plan

### 7.1 Hybrid fixture strategy

Per user-approved Option C:
- **Sportec** — both kloppy-derived (cross-path consistency, end-to-end signal) AND hand-crafted (qualifier coverage). The 29-event Sportec kloppy fixture is too sparse to exercise the rich qualifier set; hand-crafted fixtures fill the gaps surgically.
- **Metrica** — primarily kloppy-derived (3,594-event fixture is rich enough). Hand-crafted only for the §6.4 set-piece-then-shot composition rule edge cases.

### 7.2 Test-class structure

```
tests/spadl/test_sportec.py
├── TestSportecContract                     output schema, dtypes, empty input, no input mutation
├── TestSportecRequiredColumns              missing column raises ValueError with column names
├── TestSportecActionMapping                12 hand-crafted mini-DataFrames covering each row of §5.3
├── TestSportecCoordinateClamping           off-pitch coords clamped not dropped
├── TestSportecActionId                     range(len(actions)) per match
├── TestSportecPreserveNative               kwarg works, schema overlap raises ValueError
├── TestSportecAddDribbles                  synthetic dribble insertion correctness
├── TestSportecDirectionOfPlay              away-team coords flipped (x -> 105-x, y -> 68-y)
└── TestSportecCrossPathConsistency         load kloppy sportec fixture -> bridge to bronze DataFrame ->
                                            run sportec.convert_to_actions -> compare to kloppy.convert_to_actions
                                            output on same dataset; assert equal SPADL DataFrames

tests/spadl/test_metrica.py
├── TestMetricaContract                     same shape as Sportec
├── TestMetricaActionMapping                covers each row of §6.3
├── TestMetricaSetPieceShotComposition      §6.4 rule unit-tested (4 hand-crafted scenarios)
├── TestMetricaCoordinateClamping           off-pitch coords clamped (specifically Metrica fixture's negative x)
├── TestMetricaPreserveNative               same as Sportec
├── TestMetricaDirectionOfPlay              away-team coords flipped
└── TestMetricaCrossPathConsistency         same cross-path proof as Sportec

tests/spadl/test_kloppy.py                  MODIFIED
└── TestKloppyDirectionOfPlay               new — assert kloppy converter now flips away-team coords
                                            (RED on main, GREEN after Option C fix)
```

### 7.3 Cross-path consistency test (the architectural promise)

```python
def test_cross_path_consistency_sportec(sportec_dataset):
    # Path A: kloppy
    actions_kloppy, _ = silly_kicks.spadl.kloppy.convert_to_actions(sportec_dataset, game_id="X")

    # Path B: dedicated DataFrame converter
    bronze_df = _kloppy_dataset_to_sportec_bronze(sportec_dataset)  # test helper
    home_team_id = sportec_dataset.metadata.teams[0].team_id
    actions_dedicated, _ = silly_kicks.spadl.sportec.convert_to_actions(bronze_df, home_team_id=home_team_id)

    # Both paths must produce the same SPADL output (modulo sort order; sort then compare)
    assert_frame_equal(
        actions_kloppy.sort_values(["period_id", "time_seconds", "action_id"]).reset_index(drop=True),
        actions_dedicated.sort_values(["period_id", "time_seconds", "action_id"]).reset_index(drop=True),
        check_like=True,
    )
```

The `_kloppy_dataset_to_sportec_bronze(...)` test helper bridges kloppy `EventDataset` → bronze-shaped DataFrame matching the §5.1 schema. Walks events; for each event:
1. Map `event.event_type.name` → DFL XML event-type string (`PASS` → `"Pass"`, `SHOT` → `"ShotAtGoal"`, etc.)
2. Extract `event.coordinates`, `event.player.player_id`, `event.team.team_id`, `event.timestamp.total_seconds()`, `event.period.id`, `event.event_id`, `event.dataset.metadata.match_id` (or fixture-known constant)
3. Surface raw_event dict keys as bronze qualifier columns (snake_case lowercase the keys)

Lives in `tests/spadl/test_sportec.py` as a module-level helper (not exported from silly_kicks itself — test infrastructure).

### 7.4 Hand-crafted Sportec fixtures

Stored as Python literal-DataFrame factories in `tests/spadl/test_sportec.py` (no separate JSON/CSV files needed for small fixtures). Example:

```python
def _df_sportec_pass_with_cross():
    return pd.DataFrame({
        "match_id":           ["J03WMX"] * 2,
        "event_id":           ["e1", "e2"],
        "event_type":         ["Pass", "Pass"],
        "period":             [1, 1],
        "timestamp_seconds":  [10.5, 12.8],
        "player_id":          ["DFL-OBJ-0001", "DFL-OBJ-0002"],
        "team":               ["DFL-CLU-A", "DFL-CLU-A"],
        "x":                  [50.0, 95.0],
        "y":                  [34.0, 50.0],
        # qualifiers (only these two events test cross detection)
        "play_flat_cross":    [False, True],
        "play_height":        ["flat", "cross"],
    })
```

12 such factories cover the §5.3 dispatch table.

### 7.5 Pre-existing Sportec/Metrica fixtures

Already vendored at `tests/datasets/kloppy/` (from 1.6.0):
- `sportec_events.xml` (15 KB), `sportec_meta.xml` (12 KB)
- `metrica_events.json` (1.7 MB), `epts_metrica_metadata.xml` (34 KB)

These are **kept where they are** (under `tests/datasets/kloppy/`). The cross-path consistency tests load them via `kloppy.sportec.load_event` / `kloppy.metrica.load_event` then bridge to bronze via the test helper. No new committed fixtures required for the cross-path tests.

The hand-crafted Sportec fixtures live inline in test code (no new files under `tests/datasets/sportec/` — that subdirectory is created only if a future PR needs vendored Sportec bronze data).

## 8. Implementation order

1. Branch: `feat/dataframe-converters-sportec-metrica` from main at `0cff18e`.
2. **Spec doc commit (deferred — folded into single squash commit per user policy):** spec doc + plan doc stage but do not commit independently.
3. **TDD round 1: Sportec.** Write Sportec test file (Contract + ActionMapping + DirectionOfPlay + AddDribbles + Clamping + Preserve + ActionId tests). All RED. Implement `silly_kicks/spadl/sportec.py` minimally to pass each test in order. Skip CrossPathConsistency until the kloppy fix lands.
4. **TDD round 2: Metrica.** Same flow for metrica.py.
5. **TDD round 3: kloppy direction-of-play fix.** Write `TestKloppyDirectionOfPlay::test_away_team_coords_flipped` (RED). Add the 3-5 line fix to `kloppy.py`. GREEN.
6. **TDD round 4: Cross-path consistency.** Write `TestSportecCrossPathConsistency` and `TestMetricaCrossPathConsistency`. Initially RED if any divergence exists; iterate fixes on the dedicated converter side until both paths produce identical SPADL.
7. Pre-PR gates: `uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest tests/ -m "not e2e"` all green.
8. CHANGELOG entry, README provider list, version bump 1.6.0 → 1.7.0.
9. Final-review skill: code quality review, ADR check, C4 architecture refresh.
10. Single commit with full Co-Authored-By trailer (per `feedback_commit_policy` — literally one commit, no WIP commits).
11. User-approval-gated push, PR open, merge, tag.

## 9. CHANGELOG entry

```markdown
## [1.7.0] — 2026-04-XX

### Added
- **Dedicated DataFrame SPADL converters for Sportec and Metrica.** New
  modules `silly_kicks.spadl.sportec` and `silly_kicks.spadl.metrica`
  expose `convert_to_actions(events_df, home_team_id, *,
  preserve_native=None) -> tuple[pd.DataFrame, ConversionReport]`,
  matching the established `statsbomb`/`wyscout`/`opta` shape. Designed
  for consumers who already have normalized event data in pandas form
  (lakehouse bronze layers, ETL pipelines, research notebooks) and don't
  want to reconstruct a kloppy `EventDataset` from flat rows. Existing
  kloppy-path consumers continue to use `silly_kicks.spadl.kloppy` —
  both paths produce identical SPADL output (empirically verified by
  cross-path consistency tests).
- ~120 recognized DFL qualifier columns surfaced via Sportec
  `_RECOGNIZED_QUALIFIER_COLUMNS`, covering pass / shot / tackle / foul
  / set-piece / play / cross / cards / substitution / penalty / VAR /
  chance / specialised / tracking-derived qualifier groups.
- Metrica set-piece-then-shot composition rule: `SET PIECE` (FREE KICK)
  immediately followed (≤ 5s, same player, same period) by `SHOT`
  upgrades the shot to SPADL `shot_freekick` and drops the SET PIECE
  row.

### Changed
- **`silly_kicks.spadl.kloppy.convert_to_actions` now applies
  `_fix_direction_of_play` automatically** (extracting home team from
  `dataset.metadata.teams[0].team_id`). Pre-1.7.0, the kloppy converter
  was the lone outlier among silly-kicks SPADL converters — it stayed
  in kloppy's `Orientation.HOME_AWAY` (home-LTR, away-RTL) while
  StatsBomb / Wyscout / Opta all flipped away-team coords for canonical
  "all-actions-LTR" SPADL convention. 1.7.0 unifies the convention
  across all 5 (kloppy + statsbomb + wyscout + opta + new dedicated
  sportec/metrica) so `silly_kicks.spadl.{statsbomb,wyscout,opta,kloppy,
  sportec,metrica}` all emit semantically equivalent SPADL output for
  the same source event stream. Hyrum's Law disclaimer: zero current
  consumers built against 1.6.0's HOME_AWAY-oriented kloppy output (per
  user confirmation during brainstorming).

### Notes
- Cross-path consistency proof: `TestSportecCrossPathConsistency` and
  `TestMetricaCrossPathConsistency` empirically verify that
  `silly_kicks.spadl.sportec.convert_to_actions(bronze_df, ...)` and
  `silly_kicks.spadl.kloppy.convert_to_actions(EventDataset, ...)`
  produce identical SPADL DataFrames when given the same source data
  bridged through a test helper.
```

## 10. Release ritual (per validated 1.6.0 ritual)

1. Pre-PR gates green (lint / format / type / full test suite).
2. Single squash commit on branch.
3. User-approval-gated push: `git push -u origin feat/dataframe-converters-sportec-metrica`.
4. User-approval-gated PR open: `gh pr create ...`.
5. Wait CI green.
6. User-approval-gated merge: `gh pr merge --admin --squash --delete-branch`.
7. Local `git pull` on main.
8. User-approval-gated tag push: `git tag v1.7.0 && git push origin v1.7.0` → triggers `.github/workflows/publish.yml` → PyPI auto-publish.
9. Verify via PyPI simple-API endpoint (JSON endpoint is CDN-cached and lags by minutes).

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Cross-path consistency tests fail because the two converters compute action_id slightly differently (e.g., synthetic-dribble insertion order varies) | Sort both outputs by `(game_id, period_id, time_seconds, action_id)` before comparison; `assert_frame_equal(check_like=True)` is order-insensitive on columns. If genuine semantic divergence is found, that's the test catching a real bug — fix in the dedicated converter, not in the test. |
| Hand-crafted Sportec fixtures don't reflect real DFL data shape (false negatives in tests) | Cross-path consistency test uses real kloppy-derived data, providing a real-data check that complements the hand-crafted unit tests. The two together catch both schema/contract bugs and real-data mapping bugs. |
| `dataset.metadata.teams[0].team_id` doesn't reliably resolve to the home team across kloppy versions | Verified empirically with kloppy 3.18 in 1.6.0 work. Add defensive: if `dataset.metadata.orientation` is not `Orientation.HOME_AWAY`, raise a clear error explaining the precondition. |
| Sportec qualifier list (~120 columns) drifts as luxury-lakehouse evolves its bronze schema | `_RECOGNIZED_QUALIFIER_COLUMNS` is a frozenset; unknown columns are ignored unless in `preserve_native`. Adding new qualifiers is a non-breaking schema extension. |
| Metrica `subtype` casing varies (`"FREE KICK"` vs `"Free Kick"`) | Implement subtype matching with case-insensitive comparison: normalize via `.str.upper()` at validation time. |
| The kloppy direction-of-play fix breaks one of the 21 existing tests in `tests/spadl/test_kloppy.py` (any test that asserts specific x/y values) | Re-run full kloppy suite after the fix; update any test that asserted home-LTR coords to expect all-actions-LTR. Document affected tests in commit message. |
| Cross-path consistency test reveals a divergence in something like body-part inference (kloppy reads BodyPart qualifier; dedicated reads `play_height`) | The brief intent is "both paths produce identical SPADL." If divergence is unfixable (genuinely different source signals), document in CHANGELOG which fields are NOT identical and why. Acceptable for v1; tighten in a future PR. |

## 12. Verification checklist (post-implementation)

- [ ] All new test classes GREEN (estimated 50+ test functions across both new test files + 1 added to test_kloppy.py)
- [ ] `TestSportecCrossPathConsistency` and `TestMetricaCrossPathConsistency` GREEN (empirical proof of "both paths produce identical SPADL output")
- [ ] All existing `tests/` tests still GREEN (no regression in StatsBomb / Wyscout / Opta / kloppy-path / VAEP / atomic / utils)
- [ ] `uv run ruff check .` → 0 violations
- [ ] `uv run ruff format --check .` → no diff
- [ ] `uv run pyright` → 0 errors
- [ ] `pyproject.toml` version is `1.7.0`
- [ ] `CHANGELOG.md` has the `## [1.7.0]` entry per §9
- [ ] `README.md` provider-coverage list mentions Sportec and Metrica dedicated converters
- [ ] C4 architecture (`docs/c4/architecture.{dsl,html}`) updated and regenerated
- [ ] Single commit on the branch with message
  `feat(spadl): dedicated DataFrame converters for Sportec + Metrica + kloppy direction-of-play unification — silly-kicks 1.7.0`
- [ ] PR-S7 from 1.6.0 was 1 commit; this PR is also 1 commit (per `feedback_commit_policy`)
