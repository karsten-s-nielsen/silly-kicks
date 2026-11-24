# ADR-137: Tracking batch seams, multi-game linking, team-relative shape-graph labels, DFL event derivation

| Field | Value |
|---|---|
| **Date** | 2026-11-12 |
| **Status** | Accepted |
| **Deciders** | Karsten |

## Context

The lakehouse handoff of 2026-11-08 named six costs and defects in the tracking batch path. A second handoff named
the DFL event derivation that the lakehouse keeps as private copies. A third handoff scored the shape-graph position
labels against SkillCorner `player_role` and found the lateral ladder inverted for a team that attacks +x.

The verification for this cycle found a larger defect. `link_actions_to_frames` was game-blind. On two games with the
same period and frame ids, it kept 100 of 200 distinct frame rows and linked game 8's actions to game 7's frames. Its
pointers carried no game key, so a consumer that merged them on `action_id` doubled the rows. The repo already treated
multi-game frames as supported (`smooth_frames`, `derive_velocities`, `interpolate_frames`), so the linker was the gap
(owner ruling R5).

The speed work had one hard constraint: every change is bit-identical to the pre-change code ("Bit-identity protocol"
below). The only exceptions are the value changes that the owner approved one by one ("Approved value changes" below).

## Decision

Add public batch seams for pitch control, shape graphs, and the numba backend query. Make the linker and every pointer
consumer game-aware through a positional, fingerprint-bound `action_row`. Correct `infer_positions` to team-relative
labels. Single-source the DFL period, time, and actor rules behind a public contract.

### The public seams

- **`silly_kicks.backends`** (W1): `active_backend(family)` has the semantics of `use_numba`: it raises
  `NumbaUnavailableError` or `ValueError`. `backend_report()` never raises; it gives `"numba"`, `"numpy"`, or
  `"unavailable"` per family, plus `"numba_version"`. The module re-exports `FAMILIES` and `NumbaUnavailableError`.
  `_numba_dispatch` stays the implementation.
- **Pitch control** (W2, W3): `compute_pitch_control_batch(..., max_frames_per_chunk=2048)` (owner ruling F2; a
  positive `int` or `None`), `PitchControlFrameIndex`, `build_pitch_control_index(frames)`, and
  `compute_pitch_control_batch_from_index(index, requests, *, method, params, max_frames_per_chunk)`. The spearman
  batch extracts the players, resolves the ball, and scores TTI once per distinct frame slice, from one columnar index.
  The combine runs per request. The docstring states the measured `tracemalloc` peak per backend, taken on the whole
  gradientsports 10517 match (`docs/research/tracking-compute-perf/`).
- **Shape graphs** (W5): `compute_shape_graphs(frames, *, stability_threshold=45.0, with_positions=True,
  retain_graphs=False) -> ShapeGraphBatch` (`frames_summary`, `players`), and `SHAPE_GRAPH_SOURCE_VALUES`.
  `add_shape_graph` and `shape_graph_xfns` take their per-team metrics from the same selection. Chunks on whole-frame
  boundaries concatenate to one call's result (owner ruling R16). An outfield row with a missing key value is in no
  team-frame, and one `UserWarning` counts such rows (AG4, review A-03). The docstrings of this cycle's modules use public
  import paths only (owner ruling F3, `test_examples_use_public_paths`).
- **DFL events** (W8): `derive_dfl_event_periods`, `dfl_event_column`, `dfl_actor_columns`,
  `resolve_dfl_event_actors`, `complete_dfl_event_bronze`, `DflPeriodReport`, `DFL_EVENT_TYPE_PREFIX`,
  `DFL_NESTED_PREFIX`, and the two warnings `DflEventTimeWarning` and `DflGameSectionWarning`, all from
  `silly_kicks.providers.sportec`. `parse_dfl_events` reads the XML in one pass and applies the same rules, so the
  parser and a consumer bronze cannot drift. A per-row time defect becomes a counted `<NA>` period with one warning
  (owner ruling R3). The parser keeps dropping such rows.

### Multi-game linking (W9)

- `link_actions_to_frames` and `link_actions_to_frames_elastic` index frames on (game_id, period_id, frame_id) and
  link within each (game_id, period_id). `id_compat.align_join_keys` aligns the game keys. Every game-blind helper of
  the linker is now game-aware, and `LinkReport` gains `per_game_period_link_rate`.
- **`action_row`** (int64, W9.2, owner ruling R7) is appended to the pointers. It is the 0-based row of the action in
  the `actions` frame passed to the linker. There is no pointer `game_id` column, so no `game_id_x`/`game_id_y` suffix
  break occurs. The pointer order is a stable sort on (period_id, time_seconds, game rank). For a single-game input
  that is the previous order.
- **`action_fingerprint`** (uint64, owner ruling R9) follows `action_row`. It is the BLAKE2b 8-byte digest of the UTF-8
  record `"{game}|{period}|{action}|{time}"`. `game`, `period`, and `action` are `id_compat.canonical_id` strings.
  `time` is `float(time_seconds).hex()` after an upcast to float64. A missing value is the token `"<NA>"`. A committed
  test vector freezes the algorithm, so a change to it is a contract change.
- **Duplicates are legal** (W9.3). VAEP gamestate slots repeat `action_id` by design. The positional join handles them
  and multi-game input in the same way.
- **Binding** (W9.5). One shared helper serves every `links=` acceptor:
  - A `links` frame with `action_row` is verified at every position: the row is in range, and its fingerprint equals
    the fingerprint of `actions.iloc[action_row]`. Any mismatch raises `ValueError`.
  - A legacy `links` frame without `action_row` is joined on `action_id` only for a single-game input with unique ids
    on both sides. Otherwise it raises `ValueError` and asks for a re-link.
  - Every function that re-sorts, filters, or subsets `actions` remaps or re-links the pointers. It never forwards
    them unchanged.

  The contract text is: "`links` must be built from exactly this `actions` frame, in this row order".
- Every consumer in `silly_kicks/` and `scripts/` (owner ruling R8) joins pointer to action on `action_row`, and
  pointer to frame on (game_id, period_id, frame_id). The W9.6 gate derives the consumer population from the call
  graph and asserts it exactly (ADR-056). The fingerprint is computed once per `actions` object per top-level call. Its
  cache holds a strong reference to the frame and is never keyed on a bare `id()`.

### Per-action independence (W6)

`detect_line_breaking` gives each action row its own result. It reads the frames through one `group_rows` grouping
with canonical keys and loops only over candidate rows. A result depends only on its own row and its own frame. So a
two-game input equals the two single-game calls (the game-split property), and a duplicate row gets its own result.

### Team-relative position labels (W5.6, owner ruling R14)

`infer_positions` is corrected in place, with the same signature. Every label is in the team's own frame with SPADL
handedness (ADR-004). "B" is the team's own goal end. "L" is the left of a player who faces the goal that the team
attacks. All label inputs go through one attack-frame map, the ADR-051 point reflection: the identity for +x, and
`(105 - x, 68 - y)` for -x. A non-finite or zero `attacking_direction` raises `ValueError`. This decision supersedes
ADR-045 D5, which settled pitch-absolute lateral labels by default when no consumer existed.

### Speed work

- W5.1: the shape-graph core keeps an incremental face index and caches the opposite angles. It is bit-identical.
- W7: `smooth_frames` and `derive_velocities` memoise the Kalman covariance per call (W7.1) and walk contiguous sorted
  slices (W7.3). Both are bit-identical. The numba Kalman (W7.2) was conditional on the W0 measurement.
- The new `_influence_rows` numba kernel lives in its own module (`pitch_control/_batch_numba_kernels.py`) in the
  `PITCH_CONTROL` family, with its own parity tests.

## Bit-identity protocol

"Bit-identical" means that these are all equal: every float's bits (including the sign of zero), the dtypes, the
NaN/`<NA>` placement, the index, the column order, and the element order of returned containers.
`scripts/_bitwise.py::assert_bits_equal` canonicalises NaN payloads and compares floats as unsigned-integer views,
because `np.array_equal` and `assert_frame_equal(check_exact=True)` accept `-0.0 == 0.0`. During the cycle the
reference was a frozen copy of the pre-change code under `tests/oracles/`, which the package never imported, and
`FROZEN_ORACLE_CLOSURE_DIGEST` pinned the source of every live function that the oracles reached, computed at the
branch base. After the before/after record's bit compare, the cycle deleted both (owner ruling 2026-11-18, C5). From
then on each batch entry point keeps an exact guard against the per-frame public call in the same test run
(`compute_pitch_control_batch`, `compute_threat_pc_batch`, `compute_shape_graphs`), and the role gate keeps one
standalone 4.139 copy, `scripts/_shape_graph_v4139_baseline.py`, as its baseline arm.

Some candidates are not bit-identical by construction: a non-correctly-rounded scalar replacement, a new numba kernel,
or a BLAS/LAPACK replacement. Such a candidate ships only if its parity tests pass on x86 (GitHub CI) and on arm64
(the DGX), and the bit compare of the before/after record passes (owner rulings R13 and R22; 2026-11-18, C1 and C2).
A candidate that fails is replaced by the bit-identical variant, and its measured gain is recorded. The gate is never
loosened (AG1).

## Approved value changes

Each class below changes values only for the named inputs. Every other input stays bit-identical.

| Ruling | Input class that changes | Detection test (red at the base) |
|---|---|---|
| R1 | `detect_line_breaking` on multi-game input or duplicate `action_id`: each row gets its own result | `tests/tracking/test_line_breaking_independence.py::test_gamestate_duplicates_each_row_own_result`, `::test_chunk_property_game_split` |
| R2 | the pitch-control dedup key: the requests `(k, 1)` and `(k, "1")` no longer merge (an exact-type key) | `tests/tracking/pitch_control/test_spearman_index.py::test_dispatch_key_exact_type` |
| R2 | the `astype(str)` game re-keys of `add_shape_graph`, `shape_graph_xfns`, and `detect_line_breaking`: a float `game_id` such as `7.0` silently missed `"7"` | `tests/tracking/test_shape_graph_rekey.py::test_shape_graph_game_id_rekey_detection`, `::test_shape_graph_xfn_game_id_rekey_detection` |
| R2 | the DFL qualifier fill `_resolve_idsse_team_from_qualifiers`: an int-vs-str CLU id now matches (`id_compat.ids_equal`) | `tests/providers/sportec/test_event_derivation.py::test_qualifier_team_dtype_mismatch_detection` |
| R5/R7/R9 | every linker and pointer consumer on multi-game input; `on_low_coverage="raise"` can now raise on a badly linked second game | `tests/tracking/test_tracking_utils_link_multigame.py::test_two_game_reproduction_links_own_game`, `::test_multigame_floor_names_the_bad_game`; the W9.6 gate `tests/test_multigame_linking.py` |
| R5 (W9.5) | `links=` without `action_row` on multi-game or duplicate-id input, an out-of-range `action_row`, or a fingerprint mismatch: now `ValueError` | `tests/tracking/test_link_identity.py::test_legacy_links_single_game_unique_ok_multigame_raises`, `::test_legacy_links_duplicate_ids_raise` |
| R6 | `add_shape_graph`'s action path and `_shape_graph_at_actions` on multi-game or duplicate-id input | `tests/tracking/test_shape_graph_rekey.py` |
| R14 | the `infer_positions` labels (below) | `tests/tracking/test_shape_graph_labels.py` |
| W9.6 (owner 2026-11-12) | `defensive_line_xfns` and `off_ball_context_xfns`: every column is float64 (NaN missing, `line_break_a*` 1.0 / 0.0 / NaN); before, an `Int64` / `boolean` kernel column gave int64 / bool or float64 / object by the data. Dtypes only, on every input | `tests/tracking/test_xfn_float_columns.py` |
| W9.6 (owner 2026-11-12) | the visible-area join: one seam `silly_kicks._visible_area` for its five consumers; (game_id, action_id) when `visible_area` has `game_id`, else an unambiguous one-game `action_id` join, else `ValueError`; `add_visible_area_coverage` reads the link status by action row. Multi-game input only | `tests/test_visible_area_join.py`, `tests/test_visible_area_game_split.py`, `tests/territorial_defense/test_engine.py::test_engine_reads_each_games_own_polygon`, `tests/tracking/test_visibility.py::test_link_status_is_read_by_action_row_not_by_action_id_across_games` |
| A-01 (owner 2026-11-23) | `sync_score` rows follow the action rows that the links were built on (the index stays `action_id`); before, they were sorted by `action_id`. Single-game input whose action rows are not in `action_id` order | `tests/test_sync_score.py::test_sync_score_rows_follow_the_action_rows_in_one_game` |
| A-01 (owner 2026-11-23) | `add_ball_contacts` on an actions frame with a non-default index: the player-id columns are no longer `<NA>` (a bug fix) | `tests/tracking/test_ball_contact.py::test_add_ball_contacts_keeps_player_ids_on_a_non_default_index` |
| A-01 (owner 2026-11-23) | the restart-geometry tracking tiers (`resolve_restart_geometry`, `add_restart_coordinates`) read the linked frame on (period_id, frame_id): a provider whose `frame_id` repeats across periods no longer reads another period's frame | `tests/tracking/test_restart_geometry.py::test_tracking_ball_xy_reads_the_frame_of_its_own_period` |
| A-02 (owner 2026-11-23) | defensive credit (`compute_defensive_credits`, `add_defensive_credit`, the through-ball line-break signal) reads the linked frame on (period_id, frame_id): the same input class | `tests/tracking/test_defensive_credit_orchestration.py::test_frame_lookup_is_period_aware_within_one_game`, `tests/tracking/test_defensive_credit_through_ball.py::test_through_ball_reads_the_frame_of_its_own_period` |
| A-02 (owner 2026-11-23) | `slice_around_event` with `game_id` on both frames: a window reads only its own game's frames. Multi-game input only | `tests/tracking/test_tracking_utils_link_multigame.py::test_slice_around_event_game_split` |

### R14 in detail

- **D-lat.** For a team that attacks +x, the lateral ladder was inverted ("R" for high y). Every caller's lateral
  labels for such a team change: L and R swap, and LC and RC swap. Detection: `test_dlat_plus_x_high_y_full_back_is_L`.
- **D-eqf.** The equal-frequency fallback had the same inversion. Detection: `test_deqf_fallback_team_relative`.
- **D-tree.** The pitch-thirds branch flipped x twice for a team that attacks -x, so a deep middle group got "AM", not
  "DM". Detection: `test_dtree_minus_x_deep_middle_is_DM`.
- **The new `ValueError`** for a non-finite or zero `attacking_direction`, which was read as +1. Detection:
  `test_direction_must_be_finite_nonzero`.
- **Relations to 4.139** (seeded continuous fixtures; tested against the frozen oracle until C5 deleted it, and on
  the whole gradientsports 10517 in the C1 record, where every changed label is the lateral mirror):
  - Attacks +x: `vertical` equals the oracle's. `horizontal` equals the oracle's with L↔R and LC↔RC.
  - Attacks -x: `vertical` equals the oracle's, except where the pitch-thirds branch fires (D-tree). `horizontal`
    equals the oracle's, except on the equal-frequency fallback for n ∉ {5, 10} players.
  - **The -x lateral fallback class.** The new binning ranks `68 - y` onto `R, RC, C, LC, L`, the mirror of the +x
    binning. It agrees with the oracle only for n = 5 and n = 10. For n = 3, 4, 6, 7, 8, 9 (a red card, or a player
    missing from the tracking data), four players move one lateral level. For n = 3, all three players move, by up to
    two levels. `test_minus_x_lateral_fallback_n9` pins one case.
  - **The -x lateral exact-tie class.** When the lateral middle group is exactly two players with equal y, the index
    tie-break gives the lower index "RC" (the oracle gave "LC"), so the pair swaps.
    `test_minus_x_two_player_middle_tie` pins it.
  - **Mirror invariance (exact).** `infer_positions(compute_shape_graph(P), P, -1.0)` equals
    `infer_positions(compute_shape_graph(R(P)), R(P), +1.0)` for float32 positions `P` and their float64 reflection.
- **Reachability (owner ruling 2026-11-11).** On graphs that `compute_shape_graph` builds, the pitch-thirds branch and
  the -x lateral fallback never fire. So the D-tree fix and the -x fallback fix are correctness-only: only a hand-built
  `ShapeGraph` reaches them. The real-data change of R14 is the +x lateral swap. The W5.7 harness still counts both
  classes (expected 0) and reports the counts.
- **W5.7 real-data gate (owner ruling R15).** `scripts/validate_shape_graph_roles.py` scores the labels against
  SkillCorner `player_role` on the 20 public SkillCorner matches, in four cells (vertical and lateral, home and away).
  Each cell must beat a role-mix-preserving block-permutation null (above `null_p95`). It must not fall below the
  4.139 baseline (`scripts/_shape_graph_v4139_baseline.py`). It must show exact mirror invariance on every scored
  team-frame. Result (DGX, 2026-11-23, code identity `FREEZE_ID`): the gate passed in all four cells.

  | Cell | Correct / n | Agreement (Wilson 95 %) | `null_p95` | `p` | 4.139 baseline |
  |---|---|---|---|---|---|
  | Vertical, home | 255 / 292 | 0.873 (0.830-0.907) | 0.397 | 1.0e-4 | 0.873 |
  | Vertical, away | 255 / 291 | 0.876 (0.833-0.909) | 0.385 | 1.0e-4 | 0.876 |
  | Lateral, home | 203 / 217 | 0.935 (0.895-0.961) | 0.553 | 1.0e-4 | 0.041 |
  | Lateral, away | 204 / 217 | 0.940 (0.900-0.965) | 0.562 | 1.0e-4 | 0.940 |

  No labelled team-frame differs from its mirror. The pitch-thirds branch and the lateral fallback fired on 0
  team-frames. The R14 fix moves only the home lateral cell. The handoff's away vertical gap does not persist over the
  20 matches, and no defect was found. Memo: `docs/research/shape-graph-positions/`.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. A pointer `game_id` column instead of `action_row` | A readable key | A `game_id_x`/`game_id_y` suffix break for every consumer that merges on `action_id` against a frame with `game_id`; still ambiguous on duplicate ids | Breaks existing merges (reviewer B) and does not handle gamestate duplicates |
| B. Raise on a duplicate `action_id` | Simple | Breaks `compute_features` with any frame-aware xfn on every two-period match (gamestate slots) | Measured by both reviewers (owner ruling R7) |
| C. `action_row` without a fingerprint | One column | A re-sorted frame of the same length passes a range check, so a pointer can silently index the wrong action | Owner ruling R9 |
| D. Keep pitch-absolute lateral labels (ADR-045 D5) | No value change | The lakehouse v2 is now a consumer; the measured home lateral agreement was 0/7, 2/11, and 1/12 | Owner ruling R14 |
| E. A tolerance-based speed-up | More speed | Values change without an owner decision | AG1: a tolerance comes back only as a measured proposal |
| F. Overload the first parameter of `compute_pitch_control_batch` with an index | One function | A Hyrum surface: the type of the first argument changes the behaviour | A separate `compute_pitch_control_batch_from_index` |
| H. A `ValueError` for input with more than one game | The smallest change: no pointer columns, no fingerprint | Every multi-game caller must split by game itself, and the defect stays in each consumer that joins on `action_id` | Owner ruling R5 (fix it at the root); confirmed 2026-11-18 (reviewer C, TBC-SCOPE-08: W9 stays as built and is closed) |
| G. (chosen) The seams above, a positional fingerprint-bound `action_row`, team-relative labels | Correct multi-game support; one source for each rule | A minor version and a CHANGELOG value-change list | — |

## Consequences

### Positive

- A caller can score many frames and many request waves without private imports, with a bounded memory peak.
- Multi-game frames link correctly in the linker and in every consumer. A stale or re-sorted `links` frame fails loud
  instead of mis-joining.
- Shape-graph labels mean the same thing for both teams, so labels compare across teams and directions.
- The lakehouse can delete its DFL copies, its `_shape_graph`/`_line_breaking` private imports, and its `_HAS_NUMBA`
  guard (handback note).

### Negative

- Every pointer frame gains two columns. Legacy `links=` frames work only for single-game unique-id input.
- Lateral labels change for every team that attacks +x. A consumer that stored labels must re-materialize them.
- The fingerprint costs one BLAKE2b pass per `actions` object per call, linear in the number of actions.
- The oracle copies under `tests/oracles/` were deleted in this cycle, after the before/after record's bit compare
  (owner ruling 2026-11-18, C5). Without them, a change in the last bits of a rewritten core passes every test.
  Larger value changes fail: the DFL parser has its lakehouse goldens, and the preprocess, the linkers, the
  shape-graph core and `detect_line_breaking` have value snapshots at `rtol=1e-12`. The float32-stored preprocess
  outputs compare within one float32 ulp, and their finite-difference columns within twice the effect of one position
  step (5e-4 m/s, 1e-2 m/s²) (`tests/regressions/tracking_batch/`, owner rulings 2026-11-23).

### Neutral

- The HF leg and the W0 driver left this cycle (owner rulings 2026-11-18, C2). The two defects that the work found on
  `main` (orphaned executor workers; the `datasets fetch` argument) went to the HF tooling's owner session as
  findings. ADR-128 is not amended.

## Related

- **Specs:** `docs/superpowers/specs/2026-11-08-tracking-batch-and-dfl-events-design.md`
- **Plans:** `docs/superpowers/plans/2026-11-08-tracking-batch-and-dfl-events.md`
- **Research:** `docs/research/tracking-compute-perf/`, `docs/research/shape-graph-positions/`
- **ADRs:** supersedes ADR-045 D5; amends ADR-008 and ADR-105 (pitch-control batch), ADR-068 (frame index), and
  ADR-031 (DFL parse port).
