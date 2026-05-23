# Ball Carrier Numba Vectorization — Design Spec

**Date:** 2026-05-23
**Status:** Draft
**Scope:** Performance refactor of `infer_ball_carrier` in `silly_kicks/tracking/_ball_carrier.py`

## Problem

`infer_ball_carrier` uses a Python-level `iterrows()` loop over every tracking
frame. At ~0.16ms/frame Python overhead, a single GS WC2022 match (~200K
unique frames) takes ~31s. This makes Optuna hyperparameter calibration
(TC-3: 77 matches × 100 trials) impractical (~67 hours).

**Benchmark** (silly-kicks 3.16.2, Windows 11, Ryzen 9 7950X):

| Metric | Current | Target (kernel) | Target (end-to-end) |
|---|---|---|---|
| Per-frame cost | 0.16ms | <5μs | — |
| Single GS match (200K frames) | 31.4s | <1.0s (kernel) | <2.0s (incl. pre-index + post-process) |
| 77-match calibration trial | ~40 min | — | <2 min |
| 100 Optuna trials | ~67 hours | — | <3.5 hours |

The <1s target is for the numba kernel only. Pre-indexing (pandas groupby +
dense array fill) adds overhead that scales with input size. The end-to-end
target of <2s per match accounts for this. If pre-indexing proves to be the
bottleneck (>1s for 4.5M rows), it can be optimized separately (e.g.
`DataFrame.to_numpy()` bulk conversion instead of per-group iteration) without
changing the kernel interface.

## Root cause

The inner loop (lines ~129-226) iterates Python-level via `group_sorted.iterrows()`.
Per frame it pays: dict lookup, Series extraction, `.to_numpy()` calls, type
checking, dict-append to results list. Steps 3-7 (distance, filter, velocity
score, hysteresis, argmin) are already vectorized per-frame — the bottleneck
is the Python/pandas overhead wrapping each frame iteration.

## Approach

**Pre-index to dense numpy arrays + numba @njit inner loop.** The hysteresis
(incumbent_pid) makes the algorithm inherently sequential per-period, so
cross-frame vectorization is not possible. But all Python/pandas overhead can
be eliminated by converting to dense arrays once and running a tight numba
loop over float64/int64 arrays.

When numba is unavailable, the same dense arrays are processed by an identical
Python loop — still ~10-20× faster than iterrows (no pandas overhead in the
hot path).

## Architecture

Three-phase pipeline replacing the current monolithic function body:

### Phase 1: Pre-index (pandas → numpy, runs once, no numba)

**Player-to-slot mapping:**
- `unique_pids = sorted(frames[~ball_mask]["player_id"].unique())` — sorted
  gives deterministic tiebreak (lexicographic for strings, numeric for ints;
  matches Python `<` semantics used by current `_select_best`).
- `pid_to_slot: dict[Any, int]` maps player_id → integer index.
- Same for `team_id → team_slot`.
- Inverse arrays `slot_to_pid` and `slot_to_tid` for post-processing.
- **Assumption:** `player_id` column must be homogeneous type (all numeric or
  all string). Mixed-type player_ids are not supported by the current
  implementation (`sorted()` raises `TypeError` on mixed int/str) and remain
  unsupported. This matches real-world data: each provider uses a single
  player_id dtype (int64 for StatsBomb/Opta/GS, object for Sportec DFL-OBJ-*).

**Dense array construction:**
- Group player_rows by `(game_id, period_id, frame_id)`.
- `max_players = max(group sizes)` — typically 22-23, sets array width.
- Build arrays shaped `(n_frames, max_players)`:
  - `px`, `py`: player positions (NaN-padded for absent slots). Players with
    `x.notna()` pass the initial filter, so valid-slot positions are non-NaN.
  - `pvx`, `pvy`: player velocities (NaN-padded for absent slots). **NaN
    velocity values are possible for valid players** (e.g. single-frame groups
    post-3.16.2 fix). The kernel treats NaN velocity as 0.0
    velocity-toward-ball (same as current `np.where(np.isnan(v_toward), 0.0, ...)`
    behavior). When no vx/vy columns exist on the input DataFrame, all velocity
    arrays are zero-filled.
  - `player_slots`: integer player indices (-1 for empty slots)
  - `team_slots`: integer team indices (-1 for empty slots)
  - `n_valid`: actual player count per frame `(n_frames,)`
- Build per-frame arrays `(n_frames,)`: `bx`, `by`, `ball_dead` (bool).
- `ball_dead[f] = True` when `ball_state == "dead"` OR `bx`/`by` is NaN.
  "No candidates within tolerance" is computed inside the kernel, not pre-computed.
- Iterate groups once to fill arrays — O(n_player_rows), same asymptotic cost
  as the current dict-building step. Actual wall-clock for 4.5M rows TBD at
  implementation time; if >1s, optimize via bulk `to_numpy()` conversion.

**Segment boundaries:**
- Group by `(game_id, period_id)` — same as current outer loop.
- Record `seg_starts[s]` and `seg_ends[s]` as contiguous index ranges into
  the frame-sorted arrays. Each segment is a half-open range
  `[seg_starts[s], seg_ends[s])`.
- Frames sorted by `(game_id, period_id, frame_id)` as a stable sort during
  pre-indexing, producing equivalent ordering to the current per-group
  `group.sort_values("frame_id")`.

### Phase 2: Inner loop (numba @njit kernel or Python fallback)

**Numba kernel** (`_ball_carrier_numba.py`):

`_ball_carrier_numba.py` imports numba at module level and raises `ImportError`
if absent (same as `pitch_control/_numba_kernels.py`). The `try/except` in
`_ball_carrier.py` catches this and sets `_HAS_NUMBA = False`.

```python
@njit(cache=True)
def _carrier_loop_numba(
    bx, by, ball_dead,           # (n_frames,) float64, float64, bool
    px, py, pvx, pvy,            # (n_frames, max_players) float64, NaN-padded
    player_slots,                # (n_frames, max_players) int64, -1 for empty
    n_valid,                     # (n_frames,) int64
    seg_starts, seg_ends,        # (n_segments,) int64
    tolerance_m, beta, gamma,    # float64 scalars
    has_velocity,                # bool
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (winner_slot, winner_dist), both shape (n_frames,)."""
```

Per-frame logic (identical in both kernel and fallback):
1. If `ball_dead[f]`: set `winner_slot[f] = -1`, reset incumbent to -1, continue.
2. Compute distances for `n_valid[f]` players: `sqrt((px[f,i]-bx[f])^2 + (py[f,i]-by[f])^2)`.
3. Filter to `<= tolerance_m`. If none within tolerance: winner=-1, reset incumbent, continue.
4. If `has_velocity`: compute velocity-toward-ball score per candidate. **NaN
   velocity values are treated as 0.0 velocity-toward-ball** (clamped via
   explicit NaN check before dot product, matching current behavior). Apply
   beta weighting: `score = dist - beta * max(v_toward, 0)`.
5. If incumbent >= 0 and incumbent is among candidates: subtract gamma from incumbent's score.
6. Select winner: lowest score; tiebreak by lowest `player_slots[f,i]` value
   (equivalent to lowest player_id via sorted mapping). Uses same `1e-12`
   tolerance as current `_select_best`.
7. Write `winner_slot[f]`, `winner_dist[f]`, update incumbent.

Incumbent resets to -1 at each segment boundary (new game/period).

**Python fallback** (`_carrier_loop_numpy`): identical logic, same signature,
plain Python loop over the dense arrays. Lives in `_ball_carrier.py`. No
separate file. Still ~10-20× faster than iterrows since no pandas overhead.

**Import pattern** (matches pitch_control):
```python
# _ball_carrier_numba.py raises ImportError if numba absent.
# Consumer catches it and falls back to Python loop.
try:
    from ._ball_carrier_numba import _carrier_loop_numba
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
```

**Thread safety:** The numba `@njit` kernel releases the GIL during execution,
enabling true multi-threaded parallelism via `ThreadPoolExecutor` for
multi-match workloads. Each match can be pre-indexed and kerneled
independently. This is a significant capability unlock for the TC-3
calibration use case (77 matches per trial can be parallelized across threads).

### Phase 3: Post-process (numpy → pandas, runs once)

- Map `winner_slot[f]` back to `player_id` / `team_id` via `slot_to_pid` /
  `slot_to_tid` inverse arrays. `winner_slot == -1` maps to NaN.
- Build output DataFrame with same schema: `game_id, period_id, frame_id,
  ball_carrier_player_id, ball_carrier_distance_m, ball_carrier_team_id`.
- Preserve input `player_id`/`team_id` dtype (Int64 casting, same as current code).

## Public API

**No changes.** `infer_ball_carrier(frames, *, tolerance_m, beta, gamma)` has
identical signature, identical return schema, identical output values.
`derive_team_in_possession` is untouched. `ball_carrier_at_action` is untouched.

## Tiebreak determinism

Current `_select_best` uses Python `<` on player_ids (works for int and string).
The pre-indexing maps player_id → slot_index via sorted unique player_ids:
- Numeric player_ids: sorted numerically → lowest slot = lowest pid
- String player_ids (Sportec `DFL-OBJ-*`): sorted lexicographically → lowest slot = lowest pid

Tiebreak-by-lowest-slot-index in the kernel is semantically equivalent to
tiebreak-by-lowest-player_id. Output is bit-identical to current implementation.

**Assumption:** `player_id` column is homogeneous type. `sorted()` raises
`TypeError` on mixed int/string. This is not a new limitation — the current
`_select_best` uses Python `<` which also fails on mixed types. All real-world
providers use a single dtype per column.

## File changes

**Modified:**
- `silly_kicks/tracking/_ball_carrier.py` — rewrite `infer_ball_carrier` body
  to three-phase pipeline. Add `_pre_index_frames()`, `_carrier_loop_numpy()`,
  `_post_process()` helpers. Delete `_nan_row()` and `_select_best()` (logic
  inlined in kernels; no external consumers — verified via grep).
  `derive_team_in_possession` unchanged.

**New:**
- `silly_kicks/tracking/_ball_carrier_numba.py` — `_carrier_loop_numba`
  `@njit(cache=True)` kernel (~60-80 LOC). Imports numba at module level,
  raises `ImportError` if absent.
- `tests/tracking/test_ball_carrier_numba_parity.py` — parity, fallback,
  and round-trip mapping tests.

**Unchanged:**
- `tests/tracking/test_ball_carrier.py` — existing 25 unit tests, zero modifications.
- `tests/invariants/test_invariant_ball_carrier.py` — existing 3 invariant tests.
- `silly_kicks/tracking/__init__.py` — no public API changes.
- `silly_kicks/tracking/features.py` — `ball_carrier_at_action` unchanged.

## Testing strategy

**Existing tests (28 tests, must all pass unchanged):**
- 25 unit tests in `test_ball_carrier.py`: velocity scoring, hysteresis
  (retain/override/reset on dead ball/reset on no-candidate/first frame),
  distance-only fallback (with warning/with hysteresis), edge cases (dead ball,
  NaN ball state as alive, no ball row, NaN ball coords, no candidates, GK
  carrier, tiebreak, empty frames, set-piece transition, multiple ball rows,
  multi-game batch), return schema (columns/distance bounded/team matches/
  fresh index), action-coupled wrapper (linked/unlinked).
- 3 invariant tests in `test_invariant_ball_carrier.py`: distance bounded by
  tolerance, carrier is never ball row, team_id matches carrier player.

**New tests:**

1. **Numba parity** — run both `_carrier_loop_numba` and `_carrier_loop_numpy`
   on identical pre-indexed arrays, assert bit-identical `winner_slot` and
   `winner_dist`. Parametrized over scenarios: basic, hysteresis retention,
   dead ball gaps, multi-segment, velocity scoring, NaN velocity.

2. **Pre-index round-trip** — verify `pid_to_slot → slot_to_pid` is identity
   for both int and string player_ids (Sportec DFL-OBJ-* coverage).

3. **e2e benchmark** (`pytest.mark.e2e`) — full GS match via existing e2e
   fixture path (real match parquet, not synthetic). Uses `pytest-benchmark`
   fixture for statistical rigor (mean, stddev, IQR). Performance assertion
   on `benchmark.stats.stats.mean` rather than raw wall-clock, matching all
   existing perf budget tests in the codebase. Not a CI gate.

4. **Fallback path** — patch `_HAS_NUMBA = False`, run subset of existing
   tests, confirm numpy fallback produces correct results.

## Dependencies

No new dependencies. numba already declared in `[numba]` and `[test]` extras
(`numba>=0.59.0`). Import pattern matches `pitch_control/_numba_kernels.py`.

## Not in scope

- No version bump (happens at commit time).
- No ADR — performance refactor with identical API/behavior.
- No NOTICE entry — no new published methodology.
- No changes to `derive_team_in_possession` or action-coupled wrappers.

## Risk assessment

**Low risk:** Public API unchanged, output bit-identical, 28 existing tests
(25 unit + 3 invariant) as correctness gate plus new parity tests.

**Main risk:** Pre-indexing edge cases (empty frames, single-player frames,
NaN positions, NaN velocities). All covered by existing edge case tests in
`test_ball_carrier.py`.

**Pre-indexing performance:** If pandas groupby + dense array fill exceeds 1s
for 4.5M rows, the pre-indexing can be optimized (bulk `to_numpy()`, avoid
per-group iteration) without changing the kernel interface. The kernel
interface is the stable contract; pre-indexing is an implementation detail.
