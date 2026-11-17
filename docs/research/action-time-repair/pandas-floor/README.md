# The declared pandas floor (R75, R79, R80)

The library declares `pandas>=2.1.1` and `numpy>=1.26.0`. The first full run of the `pandas-floor` CI job (pandas
2.1.1, numpy 1.26.4, pyarrow 25.0.1; run 44524339397 on `2901fa5c`) failed 57 tests in 16 files; 13,270 passed.
The same 16 files on pandas 2.3.3 with numpy 1.26.4 (the lakehouse pair) failed 21, all in one script (cause A).
So numpy 1.x broke only that script; every library failure needed pandas 2.1.

| Cause | Tests | Raising line, cause | Package | Repair |
|---|---|---|---|---|
| A | 21 | `scripts/derive_coordination_params.py` `np.quantile(..., weights=)`: numpy >= 2.0 only | numpy 1.x | `weighted_quantile` writes out numpy 2's weighted `inverted_cdf` |
| B | 14 + 1 | `Int64` with NA `.to_numpy(dtype=float)` with no `na_value` (3 test files, `scripts/validate_team_coordination.py`) | pandas 2.1 | `na_value=np.nan` at the proven reads and their paired reads |
| C | 7 | `id_compat.ids_match` on `PitchControlSurface.player_ids` (read-only object array): "buffer source array is read-only" | pandas < 2.2.1 | `id_compat._id_input` copies a read-only object input before 2.2.1 only |
| D | 5 | `add_possessions` (SPADL, atomic): `team_id != prev_team` on `pd.NA` objects into `&` | pandas < 2.2 | `_pandas_compat.masked_to_numpy` at the `team_id` read |
| E | 10 | = C, caught per frame: `add_gk_influence` all-NaN, `off_ball_xt_team` 0.0 | pandas < 2.2.1 | fixed by C |

C covers E: with `ids_match` patched in memory to copy a read-only object input, the 7 C/E files gave 220 passed,
0 failed at the floor. `readonly_compare_probe.py` measured the gate (`PANDAS_COMPARES_READ_ONLY_OBJECT` = pandas
>= 2.2.1).

## The searches (R79 bar A31, point 1)

At `2901fa5c` (tree `527bf49c`), one search per form, in `silly_kicks/` and `scripts/`:

- `search-c.md`: the read-only array origins (18; two are object arrays, both in `PitchControlSurface`) and their
  consumers.
- `search-d.md`: 85 compare sites on a dtype-less `.to_numpy()` / `.values` feeding `&` / `|` / `~`
  (`scripts/search_d.py`): 34 with a literal subject, 51 with a non-literal subject.
- `search-b.md`: 336 `.to_numpy(dtype=float)` reads with no `na_value`, classified by `scripts/classify_b.py`: 160 safe
  by column, 54 on `x` / `y` / `ball_*` / `frame_rate` (masked only in the Sportec bronze dtype maps,
  `providers/sportec/parse.py`; the canonical tracking schema declares `float32`), 122 with a variable subject.

## The 34 literal-subject compare sites, read (R80)

Line numbers are at `2901fa5c`. Contract sources: `SPADL_COLUMNS` / `ATOMIC_SPADL_COLUMNS` declare `game_id`,
`period_id`, `team_id`, `player_id`, `type_id`, `result_id` as `int64` and `time_seconds` as `float64`;
`_finalize_output` casts to them (`spadl/utils.py:1749-1756`). `GRADIENTSPORTS_SPADL_COLUMNS` overrides
`team_id` / `player_id` to nullable `Int64`, which holds NA on the null-actor duel and foul events
(`spadl/schema.py`, the GS block). The ADR-003 NaN-safety tests feed the same. `KLOPPY_SPADL_COLUMNS` /
`SPORTEC_SPADL_COLUMNS` declare `game_id` / `team_id` / `player_id` as `object` (strings, not a nullable
integer). `TRACKING_FRAMES_COLUMNS` declares `game_id` / `period_id` as `int64` (`tracking/schema.py:11-12`).

| # | Site (2901fa5c) | Column | Nullable integer with NA at that point? | Decision |
|---|---|---|---|---|
| 1 | `scripts/_coordination_hypotheses.py:54` | frames `period_id` | No: `int64` (`TRACKING_FRAMES_COLUMNS`) | Safe |
| 2 | `scripts/derive_coordination_params.py:831` | frames `period_id` (and `game_id` read as object) | No: `int64` | Safe |
| 3 | `silly_kicks/atomic/spadl/utils.py:484` | `period_id` | No: `int64`, cast non-null | Safe |
| 4 | `silly_kicks/atomic/spadl/utils.py:485` | `time_seconds` | No: `float64` (NaN is a numpy float; no `pd.NA`) | Safe |
| 5 | `silly_kicks/atomic/spadl/utils.py:486` | `team_id` | **Yes: GS `Int64` with NA** | Proven by the floor job; repaired (`masked_to_numpy`) |
| 6 | `silly_kicks/atomic/spadl/utils.py:527` | `game_id` | No: `int64`, or object strings (kloppy, sportec); never NA | Safe |
| 7 | `silly_kicks/atomic/spadl/utils.py:531` | `time_seconds` | No: `float64` | Safe |
| 8 | `silly_kicks/atomic/spadl/utils.py:744` | `game_id` | No | Safe |
| 9 | `silly_kicks/atomic/spadl/utils.py:746` | `type_id` (via `next_type` dtype copy) | No: `int64`, cast non-null | Safe |
| 10 | `silly_kicks/atomic/spadl/utils.py:953` | `game_id` | No | Safe |
| 11 | `silly_kicks/atomic/spadl/utils.py:956` | `team_id` (`add_pre_shot_gk_context`) | **Yes: GS `Int64` with NA** | **Fails at the floor**: red test written |
| 12 | `silly_kicks/atomic/spadl/utils.py:978` | `player_id` (`add_pre_shot_gk_context`) | **Yes: GS `Int64` with NA** | **Fails at the floor** (the same function; raises next once #11 is fixed) |
| 13 | `silly_kicks/atomic/tracking/features.py:247` | `game_id` | No | Safe |
| 14 | `silly_kicks/atomic/tracking/features.py:295` | `game_id` | No | Safe |
| 15 | `silly_kicks/atomic/vaep/features.py:421` | `game_id` | No | Safe |
| 16 | `silly_kicks/atomic/vaep/features.py:427` | `type_id` (via `next_type`) | No | Safe |
| 17 | `silly_kicks/spadl/base.py:200` | `period_id` in `classify_deliveries` | No: every converter builds it as an integer before the call (GS `astype("int64")` at `gradientsports.py:679`); public use takes SPADL (`int64`) | Safe |
| 18 | `silly_kicks/spadl/gradientsports.py:383` | raw event `period_id` | An NA period fails the converter at `:679` (`astype("int64")`) on EVERY pandas, so no floor-only divergence | Safe (fails the same on every version) |
| 19 | `silly_kicks/spadl/gradientsports.py:386` | raw event `period_id` | Same as #18 | Safe (same reason) |
| 20 | `silly_kicks/spadl/sportec.py:993` | `event_type` (text) | Not an integer. Never NA: the XML element's tag (`providers/sportec/parse.py:1048`) | Safe |
| 21 | `silly_kicks/spadl/utils.py:480` | `result_id` | No: `int64`, cast non-null | Safe |
| 22 | `silly_kicks/spadl/utils.py:684` | `game_id` | No | Safe |
| 23 | `silly_kicks/spadl/utils.py:687` | `team_id` (`add_pre_shot_gk_context`) | **Yes: GS `Int64` with NA** | **Fails at the floor**: red test written |
| 24 | `silly_kicks/spadl/utils.py:712` | `player_id` (`add_pre_shot_gk_context`) | **Yes: GS `Int64` with NA** | **Fails at the floor** (the same function) |
| 25 | `silly_kicks/spadl/utils.py:917` | `period_id` | No | Safe |
| 26 | `silly_kicks/spadl/utils.py:918` | `time_seconds` | No: `float64` | Safe |
| 27 | `silly_kicks/spadl/utils.py:919` | `team_id` | **Yes: GS `Int64` with NA** | Proven by the floor job; repaired (`masked_to_numpy`) |
| 28 | `silly_kicks/spadl/utils.py:967` | `game_id` | No | Safe |
| 29 | `silly_kicks/spadl/utils.py:971` | `time_seconds` | No | Safe |
| 30 | `silly_kicks/tracking/_gk_geometry.py:474` | `time_source` (text) | Not an integer. Never NA: `_finalize_output` refuses an NA `time_source` (`spadl/utils.py:1740`) | Safe |
| 31 | `silly_kicks/tracking/_stoppages.py:407` | `stoppage_state` (text) | Not an integer. Every frame is labelled (`np.where` / `STOPPAGE_STATES`, `_stoppages.py:226/283`) | Safe |
| 32 | `silly_kicks/tracking/direction.py:280` | frames `period_id` | No: `int64` | Safe |
| 33 | `silly_kicks/tracking/features.py:1726` | `result_id` | No | Safe |
| 34 | `silly_kicks/tracking/features.py:6604` | `visible_area_source` (text) | Not an integer. Every branch appends a constant (`_visibility.py:315-332`) | Safe |

**Result:** 30 safe, 2 proven and repaired (#5, #27), and 2 functions (4 sites: #11, #12, #23, #24) that fail
at the floor. For `add_pre_shot_gk_context` (SPADL and atomic) with a GS null-actor row inside a shot's
lookback window, `team_id[win] != shooter_team` on `pd.NA` objects feeds `&` and then `.any()`, which raises
"boolean value of NA is ambiguous".

**Red proof** (pandas 2.1.1, numpy 1.26.4, before the repair at those sites):
`test_add_pre_shot_gk_context_with_a_null_actor_row_in_the_window_at_pandas_floor_2_1_1` and
`test_atomic_add_pre_shot_gk_context_with_a_null_actor_row_in_the_window_at_pandas_floor_2_1_1` fail with
`TypeError: boolean value of NA is ambiguous`. Both pass on pandas 2.3.3 with numpy 1.26.4 and with numpy 2.2.6.

**Repair (applied, R80):** `team_id = masked_to_numpy(...)` and `player_id = masked_to_numpy(...)` at
the reads in both functions (`spadl/utils.py:663-664`, `atomic/spadl/utils.py:932-933` at `2901fa5c`), the same helper as
the proven sites. `player_id_is_object` is read from the column dtype (`:650` / `:919`), not from the array,
so its value is unchanged. On pandas 2.2 and later the arrays are identical, so no output changes.

The four sites in `add_pre_shot_gk_context` are repaired with `masked_to_numpy` (R80), with the red tests
`test_add_pre_shot_gk_context_with_a_null_actor_row_in_the_window_at_pandas_floor_2_1_1` and its atomic twin.

## The named limit (R80)

The floor holds for every path the tests run and for the 34 compare sites that were read. By the owner's ruling,
two searched lists were **not read**: the 51 compare sites with a non-literal subject (provider columns inside the
converters, `search-d.md`) and the 122 float reads with a variable subject (`search-b.md`, `UNKNOWN`). A floor
failure on one of those paths is possible and is not excluded by this work.

## The two raw id compares (R80, checked, no change)

`tracking/pitch_control/_surface.py` `player_share` (`self.player_team_ids == team_id`) and `tracking/_gk_influence.py`
(`surface.player_team_ids == defending_team_id`) are raw numpy compares, not the pandas path, so they do not fail
at the floor. Each compares inside one source (the scalar comes from the same array, or from the keeper's row of
the same frame); the one input that breaks them is a player row with a missing team id.
