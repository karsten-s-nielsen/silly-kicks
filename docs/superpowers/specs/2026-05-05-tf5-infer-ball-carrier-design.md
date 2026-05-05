# TF-5: `infer_ball_carrier` -- per-frame ball-carrier inference

| Field | Value |
|---|---|
| **Date** | 2026-05-05 |
| **Status** | Draft |
| **PR** | PR-S28 |
| **Release target** | silly-kicks 3.5.0 |
| **ADR dependencies** | ADR-004 (tracking charter, item #3), ADR-005 (tracking-aware features) |

## 1. Motivation

Most tracking providers do not supply an explicit "ball carrier" label per frame.
Downstream features -- `pressure_on_carrier`, off-ball runs (TF-4), pitch-control
possession attribution (TF-7), and the GKDV stack (TF-15..TF-19) -- all need to
know *who has the ball* at each frame. Today the only carrier signal is the SPADL
`player_id` at linked event timestamps, which covers ~1-2% of frames. TF-5 fills
every alive-ball frame with a heuristic carrier inference.

`ball_carrier_team_id` is the implicit possession signal for the entire downstream
stack. TF-4 (off-ball runs) uses it to split attackers vs defenders. TF-7 (pitch
control) uses it for the attacking/defending team partition. TF-15..TF-19 (GKDV)
use it for possession context. Shipping `ball_carrier_team_id` now prevents every
downstream PR from reinventing possession attribution.

Academic anchor: Bauer, P., & Anzer, G. (2021). "Data-driven detection of
counterpressing in professional football." Data Mining and Knowledge Discovery,
35(5), 2009-2049. Section 3 uses a closest-player-with-velocity-toward-ball
heuristic as input to their counterpressing classifier. No single canonical
reference proposes ball-carrier identification as a primary contribution; most
papers assume the carrier is given.

Vidal-Codina, F., et al. (2022). "Automatic Event Detection in Football Using
Tracking Data." Sports Engineering, 25, 18. Notes that "the algorithm for ball
possession should have some inertia" -- motivates the hysteresis parameter (s3.2).

## 2. Scope

Two primary deliverables in one PR, plus one consistency fix:

1. **Per-frame primitive** (`_ball_carrier.py`): `infer_ball_carrier(frames, *, tolerance_m, beta, gamma) -> pd.DataFrame`.
   Operates on tracking frames. Returns one row per `(game_id, period_id, frame_id)`.

2. **Action-coupled wrapper** (`features.py`): `ball_carrier_at_action(actions, frames, ...) -> pd.Series`.
   Thin wrapper following `defending_gk_from_frames` pattern. Returns per-action
   `ball_carrier_player_id` aligned with `actions.index`.

3. **Consistency fix:** add `game_id` to `compute_defensive_line` groupby and
   return schema (see s11).

**Not in scope:** aggregator, xfn, VAEP wiring, `pressure_on_carrier`. Those
consume the primitive when they land.

## 3. Algorithm

### 3.1 Scoring formula

For each frame where `ball_state` is not `"dead"` (including NaN/None -- see
s3.3 C1) and a ball row with valid coordinates exists:

1. Compute Euclidean distance `d` from every non-ball player to ball position
   `(bx, by)`.
2. Filter to candidates within `tolerance_m`. If none -> NaN row.
3. **Velocity-aware scoring (when `vx`/`vy` columns present):**
   - Compute velocity-toward-ball: `v_toward = dot((vx, vy), unit(ball - player))`.
     Negative values (moving away) are clamped to 0.
   - Composite score: `score = d - beta * max(v_toward, 0)`.
   - **Hysteresis bonus:** if the candidate was the carrier in the previous frame
     of the same `(game_id, period_id)`, subtract `gamma` from their score:
     `score_incumbent = score - gamma`.
   - Select candidate with lowest (possibly hysteresis-adjusted) score.
4. **Distance-only fallback (when `vx`/`vy` absent or all-NaN for a frame):**
   - Select candidate with smallest `d`, with hysteresis bonus `gamma` applied
     to the incumbent (if any).
   - Emit `UserWarning` (once per call) when columns are absent.
5. **Tiebreak:** lowest `player_id` (deterministic, matching `defending_gk_from_frames`).
   For string player_ids (sportec/kloppy DFL-OBJ-* identifiers), `min()` on
   strings is deterministic but ordering is alphabetic rather than semantically
   meaningful. Acceptable for reproducibility; no semantic intent.

### 3.2 Parameter semantics

| Parameter | Type | Default | Units | Meaning |
|-----------|------|---------|-------|---------|
| `tolerance_m` | float | 3.0 | meters | Max ball-to-player distance for candidacy |
| `beta` | float | 0.5 | seconds | Distance advantage per m/s of velocity toward ball |
| `gamma` | float | 1.0 | meters | Hysteresis bonus for incumbent carrier |

**`tolerance_m` is a carrier-attribution radius, not a possession/dribbling-contact
threshold.** It answers "which player is responsible for this ball" during
receptions, first touches, aerials, and loose-ball situations -- not "is the ball
at the player's feet." Vidal-Codina et al. (2022) uses 0.5-1.0 m for close-contact
possession; our 3.0 m default is intentionally broader to cover the full attribution
envelope. TF-24's Optuna sweep will calibrate from this starting point.

`beta` has a physical interpretation: a player moving 2 m/s toward the ball
gets `2 * 0.5 = 1.0 m` of effective distance reduction. This means a player
3.0 m away moving at 2 m/s toward the ball scores equivalently to a stationary
player at 2.0 m.

`gamma` provides temporal inertia: the incumbent carrier's score is reduced by
`gamma` meters, so a new player must be `gamma` meters better in composite score
to take over possession. This eliminates flickering in contested 50/50 situations
without introducing lag -- genuine possession changes (tackle, interception) produce
score differences >> `gamma` and switch immediately. `gamma=0` gives stateless
per-frame behaviour (no inertia). Default 1.0 m means: "a new candidate must be
at least 1 m better in composite score than the current carrier to take over."

### 3.3 Edge cases

| Condition | Behaviour |
|-----------|-----------|
| `ball_state="dead"` | NaN -- no carrier during dead ball. Incumbent resets (no hysteresis carry-over across dead-ball gaps). |
| `ball_state` is NaN/None | **Treated as alive** -- produce carrier inference. Metrica and SkillCorner providers often do not populate `ball_state` reliably (kloppy adapter sets `None` when frame.ball_state is `None`). Treating NaN as dead would yield zero coverage for these providers. Documented in docstring. |
| No ball row in frame | NaN. Incumbent resets. |
| Ball coords NaN | NaN. Incumbent resets. |
| No candidate within `tolerance_m` | NaN. Incumbent resets. |
| `vx`/`vy` columns absent | Distance-only fallback (+ hysteresis) + `UserWarning` (once) |
| `vx`/`vy` present but all-NaN for candidates in a frame | Distance-only fallback (+ hysteresis) for that frame (no warning) |
| Multiple ball rows per frame | Mean of non-NaN ball x/y positions across ball rows (if only one ball row, mean == that row). No row-order dependency. |
| GK near ball | GKs are eligible candidates (goal kicks, back-passes) |
| Empty frames | Empty DataFrame with correct columns |
| First frame of a `(game_id, period_id)` group | No incumbent -- gamma bonus not applied. Pure scoring. |

### 3.4 Dead-ball / set-piece transition

During set piece setup `ball_state="dead"` -> NaN, and the incumbent carrier
resets. The instant the ball is struck and `ball_state` transitions to `"alive"`,
the kicker is identified via pure scoring (no hysteresis, since incumbent was
reset). Players standing behind the ball but not kicking have near-zero
velocity-toward-ball, so the scoring formula discriminates correctly.

### 3.5 Sequential processing requirement

Hysteresis makes the algorithm sequential within each `(game_id, period_id)`
group: frames must be processed in ascending `frame_id` order, and each frame's
result depends on the previous frame's carrier. The function is still pure
(no global state, no side effects), just order-dependent within each group.

Implementation: sort frames by `(game_id, period_id, frame_id)`, iterate groups
sequentially, track incumbent carrier per group. The per-frame scoring is still
vectorized (distance/velocity computation for all candidates at once); only the
carrier selection step is sequential. Performance budget: see s9.6.

## 4. Return schema

### 4.1 Per-frame primitive

`infer_ball_carrier(frames) -> pd.DataFrame`:

| Column | Dtype | Description |
|--------|-------|-------------|
| `game_id` | matches input | Frame key |
| `period_id` | matches input | Frame key |
| `frame_id` | matches input | Frame key |
| `ball_carrier_player_id` | matches frames' `player_id` dtype | Inferred carrier. NaN per s3.3. |
| `ball_carrier_distance_m` | float64 | Euclidean distance from carrier to ball. NaN when carrier is NaN. |
| `ball_carrier_team_id` | matches frames' `team_id` dtype | Carrier's team. NaN when carrier is NaN. Doubles as implicit possession signal for downstream consumers (TF-4, TF-7, TF-15..TF-19). |

One row per unique `(game_id, period_id, frame_id)` in input. Output has a
fresh `RangeIndex` (no relationship to input index).

The groupby includes `game_id` so the primitive handles multi-game frame
batches correctly. Single-game callers see no difference. This matches the
`game_id`-in-groupby convention established by the `compute_defensive_line`
consistency fix (s11).

### 4.2 Action-coupled wrapper

`ball_carrier_at_action(actions, frames) -> pd.Series`:

Returns `ball_carrier_player_id` per action, aligned with `actions.index`.
Dtype matches frames' `player_id` dtype. NaN where unlinked or no carrier found.

Pattern: `link_actions_to_frames` -> join with `infer_ball_carrier` output on
`(period_id, frame_id)` -> map back to action index. The join uses
`(period_id, frame_id)` -- not `(game_id, period_id, frame_id)` -- because
`link_actions_to_frames` does not return `game_id`. This is consistent with
the `defending_gk_from_frames` and `_defensive_line_at_actions` patterns, and
is correct under the single-game context that `link_actions_to_frames`
implicitly assumes.

## 5. Module placement

| File | Content |
|------|---------|
| `silly_kicks/tracking/_ball_carrier.py` | `infer_ball_carrier` primitive (~120-150 LOC). Per-frame standalone primitive pattern (like `_defensive_line.py`), NOT the `_kernels.py` kernel+wrapper pattern -- this operates on raw frames, not an `ActionFrameContext`. |
| `silly_kicks/tracking/features.py` | `ball_carrier_at_action` wrapper (~25 LOC) + `__all__` entry |
| `silly_kicks/atomic/tracking/features.py` | Re-export `ball_carrier_at_action` from `silly_kicks.tracking.features` (not a mirror). The function reads no action-geometry columns (`start_x`/`end_x`/`dx`/`dy`) -- only `action_id`, `period_id`, `time_seconds` for linkage -- so standard and atomic versions are identical. Re-export avoids drift. |
| `silly_kicks/tracking/__init__.py` | Re-export `infer_ball_carrier`, `ball_carrier_at_action` |
| `NOTICE` | Bauer & Anzer 2021 s3 + Vidal-Codina et al. 2022 citations |

## 6. Candidate filtering

- **Team constraint: none.** All players are eligible. The scoring formula
  naturally picks the correct carrier. No possession input required -- avoids
  circular dependency (ball carrier *infers* possession, not the other way
  around).
- **GK exclusion: none.** GKs genuinely carry the ball (goal kicks, back-passes,
  sweeper-keeper plays).

## 7. `vx`/`vy` dependency

The primitive does NOT hard-require preprocessed frames. Behaviour:

- **`vx`/`vy` columns present:** full velocity-aware scoring.
- **`vx`/`vy` columns absent:** distance-only fallback (hysteresis still applies)
  with `UserWarning`:
  `"vx/vy columns not found; falling back to distance-only carrier inference. Call derive_velocities() first for velocity-aware scoring."`
  Warning emitted once per call via `warnings.warn(..., stacklevel=2)`.

This follows the "hybrid speed policy" spirit from ADR-004 s7 -- usable at any
pipeline stage, best results after preprocessing.

## 8. Calibration path (TF-24 amendment)

Ship with `tolerance_m=3.0`, `beta=0.5`, `gamma=1.0` as best-engineering-guess
defaults. TF-24's Optuna sweep expands scope to include `(tolerance_m, beta,
gamma)` alongside `LinkParams.k3` (and optionally the full 6-scalar Link
parameter set), validated against linked-event carrier accuracy across providers.

Validation metric: `(inferred_carrier == action.player_id).mean()` at linked
event timestamps. Thousands of labels per match across Sportec + Metrica + PFF.

**TF-24 diagnostic note:** The calibration script may want a private
`ball_carrier_retained_by_hysteresis` boolean signal to understand when `gamma`
is binding vs irrelevant. This can be added as a private diagnostic output of
the per-frame function at TF-24 implementation time -- not needed in this PR's
public API.

## 9. Testing

### 9.1 TDD unit tests (`tests/tracking/test_ball_carrier.py`)

All tests written before implementation. Coverage targets:

| Test | Asserts |
|------|---------|
| Velocity-aware scoring | Velocity breaks distance-only tie (synthetic frame) |
| Hysteresis retains incumbent | Slightly-further incumbent kept when within gamma margin |
| Hysteresis overridden | New carrier wins when score difference exceeds gamma |
| Hysteresis resets on dead ball | Dead-ball gap clears incumbent; next alive frame is pure scoring |
| Hysteresis resets on NaN carrier | No-candidate frame clears incumbent |
| Distance-only fallback | Correct carrier + `UserWarning` when vx/vy absent |
| Distance-only + hysteresis | Hysteresis applies even in distance-only mode |
| Dead-ball -> NaN | `ball_state="dead"` rows produce NaN carrier |
| `ball_state` NaN -> carrier inferred | NaN ball_state treated as alive (C1) |
| No ball row -> NaN | Frame without `is_ball=True` row |
| Ball coords NaN -> NaN | Ball row with NaN x/y |
| No candidates within tolerance | All players > `tolerance_m` from ball |
| GK as carrier | Goal kick scenario: GK is closest to ball |
| Tiebreak determinism | Two equidistant players: lowest player_id wins |
| Empty frames | Returns empty DataFrame with correct columns |
| Set-piece transition | Dead -> alive picks kicker correctly (no hysteresis, fresh) |
| Multiple ball rows | Mean of non-NaN ball positions used |
| Multi-game batch | `game_id` in groupby prevents cross-game collisions |
| First frame of period | No incumbent -- pure scoring, no gamma applied |
| Action-coupled wrapper | Linked carrier matches expected player_id |
| Action-coupled NaN | Unlinked action -> NaN |

### 9.2 Invariant tests (`tests/invariants/`)

- `ball_carrier_distance_m <= tolerance_m` for all non-NaN rows.
- `ball_carrier_player_id` is never a ball row's player_id.
- `ball_carrier_team_id` matches the team_id of the carrier's player row in frames.

### 9.3 Provider-coverage e2e

Per-provider synthetic fixtures must include:
- At least 1 alive-ball frame with a clear carrier (closest + velocity-toward).
- At least 1 dead-ball frame.
- At least 1 frame where velocity breaks a distance tie.
- At least 1 frame with NaN `ball_state` (Metrica/SkillCorner pattern).
- At least 2 consecutive alive-ball frames to exercise hysteresis.

Regenerate fixture data as needed to ensure coverage both locally and in CI.

### 9.4 NaN-safety

`ball_carrier_at_action` is a per-Series helper returning `pd.Series`, matching
the `defending_gk_from_frames` precedent. Per-Series helpers are NOT decorated
with `@nan_safe_enrichment` -- that decorator is exclusively for `add_*`
aggregators returning `pd.DataFrame`. No `@nan_safe_enrichment` on this function.

### 9.5 Public-API examples

Docstring `Examples` sections on both `infer_ball_carrier` and
`ball_carrier_at_action`, per CI-enforced `tests/test_public_api_examples.py`.

### 9.6 Performance benchmark

Add a `pytest-benchmark` test for `infer_ball_carrier` with an approximate
budget. Target: **<=2 ms per 1000 frames** on a single-match synthetic
fixture (~135k frames for 90 min at 25 fps). Use `sys.platform == "win32"`
guard with 1.5x ceiling for Windows CI runners (per feedback pattern).

This prevents regression if future scoring logic grows more complex.

## 10. Academic attribution

NOTICE entries:

```
- Bauer, P., & Anzer, G. (2021). "Data-driven detection of counterpressing in
  professional football." Data Mining and Knowledge Discovery, 35(5), 2009-2049.
  (Section 3 describes a velocity-toward-ball heuristic for carrier
  identification, used as input to their counterpressing classifier.
  Adapted for infer_ball_carrier primitive in silly_kicks.tracking.)

- Vidal-Codina, F., Evans, N., El Fakir, B., & Billingham, J. (2022).
  "Automatic Event Detection in Football Using Tracking Data."
  Sports Engineering, 25, 18.
  (Inertia/hysteresis recommendation for ball-possession algorithms;
  motivates the gamma hysteresis parameter in infer_ball_carrier.)
```

**Implementation-time note:** verify Vidal-Codina author list against the
Springer publication -- MIT Sports Lab papers sometimes have additional
institutional co-authors beyond the MIT DSpace listing.

Cross-link from `infer_ball_carrier` docstring:
`See NOTICE for full bibliographic citations.`

## 11. Consistency fix: `compute_defensive_line` `game_id`

### 11.1 Problem

`compute_defensive_line` (TF-14, PR-S27) groups by `(period_id, frame_id,
team_id)` without `game_id`. `infer_ball_carrier` groups by `(game_id,
period_id, frame_id)` -- the `game_id` is required because hysteresis state
must not bleed across games. This creates an inconsistency between the two
per-frame primitives.

While `compute_defensive_line` is stateless (no hysteresis), omitting `game_id`
is incorrect for multi-game batches: `(period_id=1, frame_id=100)` could
collide across games, producing incorrect merged rows in the output.

### 11.2 Fix (bundled with TF-5)

Add `game_id` to `compute_defensive_line`:

1. **`_defensive_line.py`**: add `"game_id"` to `required_cols`, `result_cols`,
   and the `groupby` key list. Update `_make_row` dicts to include `game_id`.
2. **`_kernels.py:_defensive_line_at_actions`**: update the merge from
   `left_on=["period_id", "frame_id_int"]` to
   `left_on=["period_id", "frame_id_int", "game_id"]` (with corresponding
   `right_on`). Actions already carry `game_id` from the SPADL schema.
3. **Tests**: existing test fixtures already include `game_id=1` in frame rows.
   Update assertions to expect `game_id` in output columns. Add one test case
   with two games sharing `(period_id=1, frame_id=1)` to confirm no collision.

### 11.3 Consumer impact

- `add_defensive_line` and per-Series defensive-line features in `features.py`:
  consume `_defensive_line_at_actions` output, which already projects to the 6
  feature columns. The `game_id` addition to the intermediate `dl` DataFrame is
  transparent -- no signature or column-set change on the public feature surface.
- `defensive_line_xfns`: same -- the xfn transformer calls
  `_defensive_line_at_actions` internally.
- Lakehouse consumer (TF-10): additive column in the per-frame primitive output.
  `game_id` is already present in all lakehouse tracking tables; this closes a
  latent multi-game-batch correctness gap.

National Park Principle: fixing this now prevents the inconsistency from
propagating to future per-frame primitives and avoids a dedicated follow-up PR.
