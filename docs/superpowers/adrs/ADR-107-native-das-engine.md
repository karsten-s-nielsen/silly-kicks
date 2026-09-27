# ADR-107: Native DAS engine — drop `accessible-space`

| Field | Value |
|---|---|
| **Date** | 2026-09-26 |
| **Status** | Accepted |
| **Deciders** | Karsten |

## Context

TF-28 DAS (Dangerous Accessible Space) and `get_xc` were a thin adapter over the external
`accessible-space` 2.0.15 package. That seam cost the repo three ways: (1) a hard optional runtime
dependency on a single-maintainer library, installed on every CI leg; (2) a `player_id = "ball"`
sentinel write that blocked `player_id → category` (ADR-106); (3) proven **defects** in the library
we could not fix at the seam — direction-biased angular quadrature (ADR-108), frame conflation across
periods/games (keyed by `frame_id` alone), row-order-dependent passer alignment, and fictional `0.0`
outputs where the answer is undefined. Probes established every DAS value would change under a correct
implementation, so a native reimplementation is the moment to also fix the defects as **documented
divergences** rather than reproduce them.

Owner set: gold-standard; scope/breaking are not constraints. Fix at the correct seam, no workarounds.

## Decision

Reimplement DAS + xC as a **native, parity-proven engine** and drop `accessible-space` from every
runtime and CI path. The physics is a faithful float64 reproduction of Bischofberger & Baca (2026) (see
`NOTICE`); the reference remains a **dev-only** parity oracle (`das-reference` extra, pinned
`accessible-space==2.0.15`; the committed golden fixture under `tests/tracking/_fixtures/das_golden/`).

**Architecture (hexagonal):**

- `_das_pack` (input port): pandas → ragged contiguous **float64** arrays. Upcasts `x`/`y`/`vx`/`vy`
  with `np.asarray(..., dtype=np.float64)` at the read boundary (ADR-106 float32 storage), sorts frames
  by `(game_id, period_id, frame_id)` and players by canonical id, resolves direction, and computes
  per-frame `Reason` codes. The ball is found by the truthy `is_ball` mask — **no `player_id` write**
  (removes the ADR-106 sentinel blocker; `player_id → category` is now unblocked, next cycle).
- `_das_engine` (reference-order numpy engine + dispatch) / `_das_numba` (fused `@njit` adapter, all DAS
  `@njit` code confined here, covered by the CI numba cache key). `compute_das(..., engine="auto")`
  selects numba when importable and `SILLY_KICKS_DAS_FORCE_NUMPY != "1"`, else numpy.
- `_das` (output port / public facade): `get_das` / `get_individual_das` / `get_xc` /
  `estimate_das_cost`, the degradation taxonomy (`DasUnscoreableError` / `DAS_SOURCE_*`), and the
  confined private `individual_das_paired` (the gkdv counterfactual seam).

**Direction** comes from the `GoalMap` (ADR-055), never from team mean-x; unresolved direction is a
`<NA>` value → the frame degrades to NaN (ADR-051). Callers may instead supply a per-frame numeric
`attacking_direction_col` (mutually exclusive with `goal_map`).

**Paired kernel (SC-1).** `pack_paired` / `compute_das_paired` / `individual_das_paired` score a factual
frame and a counterfactual (e.g. a gkdv ghost) together under ONE `GoalMap`. The "moved" rows are
**derived** (a row is moved iff any of `x`/`y`/`vx`/`vy` differs bitwise, NaN-aware, between the legs);
every other column and the whole ball row must agree exactly, else `ValueError`. This makes the
frame-keyed-cache landmine structurally impossible rather than guarded. This cycle the paired path is a
**two-call amortization** (each leg scored independently, bit-identical to `get_individual_das`);
offside-aware buffer sharing was dropped (the offside cross-dependency between legs is what SC-1
corrects). gkdv reaches native DAS through exactly this one confined seam.

**Carrier constant per frame (SC-2).** The ball-carrier column (`ball_carrier_player_id`) must be
constant within a frame (NaN-aware) or `ValueError` — the reference silently took the first row in input
order (the D-PASSER defect class).

**Dispatch / chunking.** The numpy engine vectorises over a block of frames (default `chunk_size = 32`,
memory-bounded — `tests/tracking/test_das_scale_memory.py` proves the working peak is bounded by
`chunk_size`, not the total frame count). The numba engine loops per frame with O(P·T) scratch allocated
once per frame (serial by default; `prange` over frames when `n_threads > 1`, byte-identical to serial).
Frames whose `Reason` is not `OK` get NaN without being simulated.

`estimate_das_cost` keeps the ADR-105 advisory frame-count guardrail (`DasCostWarning` when the distinct
frame count crosses a threshold; reported, never gating; DAS values are unchanged by it).

### Unknown-extra behaviour

Deleting the `das` extra means `pip install "silly-kicks[das]"` names an extra that no longer exists.
Verified in a scratch venv against a `uv build` wheel: `uv pip install --dry-run "silly-kicks[das] @
<wheel>"` **warns** (`WARNING: silly-kicks X does not provide the extra 'das'`) and installs the base
package — it is not an error. Downstream pins on `[das]` therefore degrade to a warning, not a break.

### Reference divergences (each has a named test in `test_das_divergences.py`)

| Code | Reference (2.0.15) | Native |
|---|---|---|
| D-KEY | keyed by `frame_id` only (conflates periods/games) | keyed by `(game_id, period_id, frame_id)` |
| D-BALLNAN | NaN ball → AS/DAS `0.0` | NaN, `unscoreable_frame` |
| D-OFF | < 2 finite defenders → arbitrary offside line | offside not applied |
| D-PASSER | passer aligned by input row order | aligned by frame key (row-order invariant) |
| D-POSSABSENT | possession team absent → AS `0.0` | NaN |
| D-DUP | duplicate `(game,period,frame,player)` rows silently dropped | `ValueError` |
| D-MULTIBALL | several ball rows misalign | `ValueError` |
| D-DIRVAL | non-±1 direction scales the coordinates | `ValueError` |
| D-POSSVAR | first row's possession after a team sort | `ValueError` |
| D-XC-FRAME | missing pass frame → `ValueError` | per-pass NaN + one aggregated `UserWarning` |
| D-XC-TEAM | pass team absent from tracking → `ValueError` | per-pass NaN + aggregated warning |

## Alternatives considered

| Option | Why rejected |
|---|---|
| Keep the `accessible-space` seam, fix at the boundary | the proven defects (quadrature bias, frame conflation, passer row-order) live inside the library; a boundary wrapper cannot reach them, and the `player_id="ball"` sentinel blocks ADR-106 |
| Native engine but reproduce the reference quadrature | ships a direction-biased metric (ADR-108); reference quadrature survives only as an internal parity-gate mode |
| Offside-aware buffer sharing in the paired kernel this cycle | the offside line is a per-leg cross-dependency; sharing it is what SC-1 exists to prevent — dropped, two-call amortization instead |
| **(chosen) native float64 engine (numpy + fused numba), GoalMap direction, defects fixed as documented divergences, drop the dependency** | parity-proven (Δ=0 vs reference at rtol=atol=1e-12 under the reference quadrature), materially faster, no external runtime dep, unblocks `player_id→category` |

## Consequences

### Positive

- No external DAS runtime/CI dependency; the reference is a dev-only oracle.
- The `player_id="ball"` sentinel is gone → `player_id→category` is unblocked (ADR-106 follow-up).
- Every reference defect is fixed and pinned by a named divergence test with the golden-recorded
  reference behaviour beside it.
- Parity is proven byte-for-byte against the committed oracle; the numba path is bit-identical to numpy
  within `rtol=atol=1e-10`.

### Negative

- **Every DAS value changes** (periodic quadrature + GoalMap direction). Consumers that persist
  `das_team`/`das_opponent`/`das_diff` (the calibration features) and the gkdv ΔDAS arm require a
  re-materialize; the corpus shift is quantified at commit 2 (ADR-108).
- Breaking: the `das` extra is deleted (unknown-extra warning, above); `get_das`/`get_individual_das`
  drop `use_progress_bar` and take keyword-only args (no `**kwargs`).

### Neutral

- `DasUnscoreableError` stays the ONLY degradable DAS exception; `DAS_SOURCE_VALUES` stays five tokens
  (ADR-043, unchanged). Velocity-unavailable-by-design frames degrade with `unscoreable_frame` — the
  `_das_pack` validation checks that BEFORE the other required columns, so an SB360 freeze-frame
  (velocity-less AND lacking `team_in_possession`) self-degrades rather than raising (ADR-063).
- gkdv imports exactly one private tracking symbol, `_das.individual_das_paired`, confined to
  `gkdv/_das_port.py` (allowlisted). The dead `_pin_attacking_direction` exemption is gone.
- Cost constants (`_DAS_SECONDS_PER_FRAME`, and any `prange` efficiency factor) are re-measured from the
  owner-corpus artifact at commit 2 (Task 17).

## References

Spec: `docs/superpowers/specs/2026-09-26-das-native-design.md`. Plan:
`docs/superpowers/plans/2026-09-26-das-native.md`. Quadrature: ADR-108. Builds on ADR-055 (GoalMap
direction), ADR-051 (unresolved direction is `<NA>`), ADR-043 (degradation taxonomy), ADR-106 (float32
frames; unblocks `player_id→category`), ADR-063 (velocity-availability tiers), ADR-076 (numba
bit-identity). `NOTICE`: Bischofberger & Baca (2026) attribution + the reproduced MIT licence.
