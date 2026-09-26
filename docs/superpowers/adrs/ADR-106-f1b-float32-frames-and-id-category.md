# ADR-106: F1b — float32 tracking-frame storage + team_id→category

| Field | Value |
|---|---|
| **Date** | 2026-09-25 |
| **Status** | Accepted |
| **Deciders** | Karsten |

## Context

ADR-103 (frame `category` for static columns + bounded cache + batched pitch-control) and ADR-105
(vectorized spearman kernel + scorer batching) shipped the *no-retrain* tracking-memory/CPU work and
both deferred the retrain-carrying "F1b" piece. F1b is the last silly-kicks-internal optimization that
forces a bundled-model retrain, so this cycle clears it.

Tracking frames are ~0.95–3.2 M rows/match; their coordinate + identifier columns dominate frame RAM.
Storing coordinates as `float32` halves that (float32 gives ~1e-5 m resolution, orders below
tracking-sensor precision), and `category` collapses a very-low-cardinality id. But float32 STORAGE
rounds the coordinate ~1e-5 m, which exceeds the trained-model feature-contract `atol=1e-6` — a real
feature-value change, hence a retrain. Owner set: gold-standard, scope/breaking not a constraint.

## Decision

Store tracking-frame coordinate + kinematic columns (`x`/`y`/`z`/`vx`/`vy`/`speed`/`x_smoothed`/
`y_smoothed`) as **`float32`, computed as `float64`** (every kernel upcasts the coord slice to float64 at
its boundary, so the only numeric drift is the deterministic storage-rounding and the numba/ADR-076
bit-identity is preserved). Store frame **`team_id` as `category`** (static, very low cardinality);
**`player_id` stays Int64/object** (dynamic post-build — the ADR-103 rule). Retrain every bundled model
that trains on tracking-frame geometry, on float32 frames, with clean provenance. SPADL action
coordinates stay `float64` (frames only). Two commits: schema migration (no weights), then DGX-trained
weights (the version + CHANGELOG are claimed at the weights commit, not the schema commit).

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| Compute natively in float32 | max in-loop win | numba float32 specialization (breaks ADR-076 bit-identity), compounded non-deterministic drift | float64-compute-from-float32-storage gives the at-rest win with a single deterministic drift source |
| float32 on SPADL action coords too | uniform | degrades the canonical event-coordinate contract every event metric depends on, for ~no memory (events are thousands of rows) + retrains all event-only models | frames only |
| **Part B — off-frame `FrameProvenance`** | purest schema | its memory target (constant provenance strings) was ALREADY captured by ADR-103's `category`; the off-frame move adds a companion-object API ripple (`pandas.attrs` doesn't propagate) + the xt-gk per-row variant-read risk | **DROPPED** — YAGNI/Chesterton |
| `player_id → category` too | more memory | `player_id` is MUTATED post-build (`_das.py` `.loc[ball_mask,"player_id"]="ball"` masked setitem, `_run_values` Int64-reassign, the keeper/actor identity bridges) → category is not setitem-transparent to a new category (ADR-103 rule) → crashes | **team_id only** (option A); player_id reconsiderable once DAS is native (removes the `_das` sentinel blocker) |
| **(chosen) float32 storage / float64 compute + team_id→category + retrain-all-frame-geometry** | halves the dominant columns, deterministic, train/serve-consistent, fully re-provenanced | a DGX retrain of the frame-geometry bundle | — |

## Consequences

### Positive

- Halves the dominant tracking-frame column memory (coords + team_id) at rest.
- `id_compat._decat` unwraps `category` to its underlying dtype at every comparison → id logic is
  value-neutral; the ADR-019 category-axis gate proves all 34 aggregators invariant.
- ADR-076 numba bit-identity preserved (measured: parity gates green on the float32-storage tree,
  because the numba boundaries were already `to_numpy(dtype="float64")`).
- The read boundary widens float32→float64; the **reverse boundary** (a float64 result written back onto a
  coord column via a masked/scalar setitem) narrows float64→storage. Every such write-back casts the RHS to
  the column dtype (`reflection.py`, `tracking/direction.py`, `tracking/utils.py`, `positioning/_optimizer.py`,
  `gkdv/_engine.py`, `gkdv/_probe.py`, `restdefense/_counterfactual.py`, `restdefense/_probe.py`,
  `tracking/_model_eval.py`). Under pandas 2 an un-cast float64→float32 setitem silently upcast; under pandas 3
  it raises `LossySetitemError` (ADR-057 pandas-major difference).

### Negative

- A DGX retrain of the frame-geometry bundled models (xshot/xcross/ghost_gk/ghost_outfield/receiver, and
  gk_completion pending the T10 classification), with regenerated chirality/feature-contract probes.
- Breaking dtype change on `TRACKING_FRAMES_COLUMNS` (coords float32, team_id category); a consumer
  asserting the old frame dtypes breaks (intended). Downstream VAEP/xT fit on tracking features is a
  consumer retrain trigger (out of this cycle).

### Neutral

- Atomic-`interception`-dedup residue: NIL in-repo (already discharged at ADR-096; config `len==32`,
  fixtures consistent, lookups symbolic). The AtomicVAEP retrain + lakehouse re-materialize is
  consumer-side.
- `observed=True` added to every frame `team_id`-keyed groupby (silences the pandas `observed=False`
  deprecation on the new category key; behaviour unchanged when all categories are observed).
- **Write-back-cast forward-guard = the pandas-3 CI leg, not a bespoke AST gate** (F1B-PLAN-06). The read
  boundary is guarded by an AST gate (`test_frame_coord_upcast_gate.py`) because "did this read upcast?" is a
  static property. The write-back boundary is guarded by the ADR-057 pandas-major span instead: CI runs pandas
  3 on 3.11/3.12/windows-3.12, and pandas 3 raises `LossySetitemError` on any NEW un-cast float64→float32
  write-back at the site of the offence — a stronger, dynamic guard than an AST heuristic, and a bespoke
  write-back AST gate would be a brittle narrow duplicate of it. A red-green storage-dtype test on
  `reflect_columns` documents the class; the pandas-3 leg is the sanctioned recurrence guard.
- Reviewer follow-ups (spec r1): F1B-SPEC-01 (commit-1 leaves probe builders float64 → un-retrained
  models load; commit-1 model-load gate), F1B-SPEC-02 (weights-dir classification gate), F1B-SPEC-03
  (gk_completion classify-not-skip).

## References

Spec: `docs/superpowers/specs/2026-09-25-f1b-float32-frames-id-category-design.md`. Plan:
`docs/superpowers/plans/2026-09-25-f1b-float32-frames-id-category.md`. Builds on ADR-103 (category rule),
ADR-105 (vectorized kernel), ADR-058 (nullable frame ids), ADR-019 (`id_compat`), ADR-076 (numba
bit-identity), ADR-096 (atomic interception dedup). Next cycle: native DAS reimplementation.
