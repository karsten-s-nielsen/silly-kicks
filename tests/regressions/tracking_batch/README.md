# Tracking-batch value snapshots

**Why:** the tracking-batch cycle (spec `docs/superpowers/specs/2026-11-08-tracking-batch-and-dfl-events-design.md`)
rewrote five cores for speed and proved them bit-identical to 4.139 with frozen oracles. When the oracles ended
(ruling C5), four of those cores kept only tests that check behaviour. The owner ruled on 2026-11-23 ("Tolerance
snapshots (Recommended)") that each gets a committed value snapshot.

## Audit (2026-11-23)

| Core | Value pins before this folder | Snapshot here |
|---|---|---|
| Preprocess (`smooth_frames`, `derive_velocities`) | none | `preprocess_<method>` for savgol, butterworth, ema and kalman |
| Linkers (`link_actions_to_frames`, `link_actions_to_frames_elastic`) | small hand-built cases only | `link_tolerance`, `link_tolerance_two_games`, `link_elastic_two_games` |
| Shape-graph core (`compute_shape_graph`, `infer_positions`) | edge counts only; no stability | `shape_graph_edges`, `shape_graph_faces`, `shape_graph_labels` |
| `detect_line_breaking` | hand-built cases and one two-row golden | `line_breaking` (the seeded fixture that reaches every route) |
| DFL parser (`parse_dfl_events`) | the lakehouse goldens in `tests/providers/sportec/test_parse_port_parity.py` | none needed |

## Files

- `_builders.py`: the seeded inputs and the public call of each case.
- `capture_goldens.py`: writes `golden_<case>.parquet` for every case.
- `test_value_snapshots.py`: compares each case with its golden. Float64 columns compare at `rtol=1e-12` (with
  `atol=1e-12` near zero). The preprocess outputs are float32 storage (ADR-106), and they compare within one float32
  ulp: the float64 compute differs in its last bits between CPUs, and the float32 rounding can then land on the
  neighbouring float32 (measured on arm64: one position in 2,132). The velocity and acceleration columns are finite
  differences of that position, so they compare within twice the effect of one position step: 5e-4 m/s for `vx`,
  `vy` and `speed`, 1e-2 m/s² for `accel_x`, `accel_y` and `accel` (owner rulings 2026-11-23). So a real value
  change fails on every CPU, and last-bit differences between CPUs pass. All other columns compare exactly.

## When a snapshot fails

1. Find the cause. A failure means the output values changed.
2. If the change is not approved, fix the code.
3. If the owner approved the value change, run `python tests/regressions/tracking_batch/capture_goldens.py`, and
   record the change in the CHANGELOG entry that ships the new goldens.

A change to a fixture generator in `tests/_prod_fixtures.py` or to `_inputs()` in
`tests/tracking/test_line_breaking_independence.py` also moves these snapshots. Then recapture, and say in the
commit that only the inputs changed.

Captured from a tree whose code identity differs from the round-0 freeze `ba77a3b6` only in comments and
docstrings. The fixes of implementation review round 1 (spec section 7) change no value in these cases: the goldens
pass unchanged on the measured freeze `c64baa66`, on x86 and on arm64, and on the cycle commit in CI. The C1 record
(`docs/research/tracking-compute-perf/README.md`) names the identities and the one line between the freeze and the
cycle commit.
