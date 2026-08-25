# ADR-068: Rescan-in-loop remediation + the `group_rows` frame-index seam

| Field | Value |
|---|---|
| **Date** | 2026-08-25 |
| **Status** | Accepted |
| **Deciders** | Karsten S. Nielsen |

## Context

The 2026-08-24 optimization audit found one anti-pattern accounting for every Critical finding and ~⅓ of the total: a **full-table filter inside a per-item Python loop** — `df[df["frame_id"] == fid]`, `df[(df["a"]==x) & (df["b"]==y)]`, or a positional forward scan — where the same rows could be located in O(1) from a grouping built once. It is O(n·m) (often O(n²)) where O(n) is available, invisible at test scale (≤1.5 K actions / ≤2.5 K frames) and only biting at real-match scale (43 K–146 K frames, 0.95–3.2 M rows) — which is why a downstream lakehouse consumer, not CI, reported the first instance (`EmpiricalTurnoverValue`, ~800 K rows / 505 matches). The class had **nine confirmed members** across `causal/`, `tracking/`, `spadl/`, and `scripts/`, plus the turnover O(n²).

Constraints: every **perf** fix must be **output-preserving** (byte-identical; no VAEP/model-retrain trigger), follow the house **fail-loud, never silent-degrade** discipline (ADR-043/ADR-051), respect **ADR-019** dtype-safe id handling, and add **no new runtime dependency** (except the turnover kernel's optional `numba`, which already ships as an extra with a pure-Python fallback). The one deliberate exception is the **ADR-065 turnover-fit row-order fix (item A)** bundled here: a correctness fix that is value-changing on a non-chronological corpus (see Consequences).

## Decision

Consolidate the rescan-in-loop fix behind **one seam, `silly_kicks/_frame_index.py::group_rows`** (an O(1) dtype-safe row-group lookup backed by `groupby().indices`), applied across the nine sites; hoist the remaining loop-invariant recompute (interpolator/grid/`np.unique`/`ids_isin`/man-marker-classification); and rewrite the turnover scan as **`numba`-@njit on equality-preserving integer codes with a pure-Python fallback**. Everything is byte-identical, gated per-fix by a parity test **plus** a structural call-count guard.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Nine inline `groupby().get_group()` fixes | no new module | duplicates the ADR-019 key-canonicalisation + collision hazard 9×; nine independent guard surfaces; pattern has no documented home | not DRY; re-enters easily |
| B. Turnover: pure-numpy possession pre-aggregation | no numba dep | byte-identity is fragile — the per-row window break × consecutive-opponent-possessions × ball-back interplay is hard to reproduce exactly | correctness risk on a shipped label |
| C. **Chosen:** `group_rows` seam + invariant hoists + `numba`-on-codes turnover | one guard, one home, one ADR; byte-identical by construction; dtype-safe (factorize handles object/Int64 ids); numba fallback keeps no-numba installs working | one new private leaf module; the turnover path gains an optional numba kernel | — |

`group_rows` is **private** (`_frame_index.py`) not public: it is an internal perf primitive with no cross-repo consumer, following the `_geometry`/`_polygon`/`lane_control` precedent (recorded in the in-repo table of `docs/PRIVATE_CONSUMERS.md`), not the mandatory-seam `id_compat` precedent.

## Consequences

### Positive
- The dominant O(n²) class is removed at real-match scale; the turnover fit's dominant cost drops from an ~800 K-row Python double loop to a compiled scan.
- One tested seam (`group_rows`) with a construction-time collision guard (refuses a mixed-dtype key column rather than silently losing rows) is the single home for the pattern.
- `calibrate_tracking_defaults._load_fold` now **fails loud** (RAM-aware, before OOM) instead of risking a silent subset calibration; the script drivers `_xtgk_comparability` / `_loader_databricks` / `_loader_pining_to_cache` gain cache-reuse / IN-batching / resume.

### Negative
- One new private module; a future reader must know `group_rows` is the canonical rescan fix.
- The turnover path depends on `numba` for the fast kernel (optional extra; pure-Python fallback preserves correctness without it).

### Neutral
- The rescan-in-loop remediations + the turnover **scan** (item B) are byte-identical (parity-gated), so **no VAEP/model retrain trigger** from them.

### Value-changing (deliberate, called out against the byte-identical majority)
- **Turnover fit row-order robustness (item A, ADR-065).** Sorting `EmpiricalTurnoverValue.fit` to the robust `(game_id, period_id, time_seconds, action_id)` key before the positional scan changes the fitted `V_opp` on a **non-chronological** corpus. The production mart read (Databricks EXTERNAL_LINKS + Arrow chunks) carries **no order guarantee** — the reversed-input `0.20`-vs-`0.30` defect the lakehouse reported came from that path — so the old fit was order-dependent and item A corrects it. silly-kicks bundles **no** turnover artifact (no silly-kicks retrain), but this is a **downstream `V_opp` re-fit trigger**: a consumer that fit on unordered marts must re-fit and diff. No-op only where the fit corpus was already sorted. The byte-identity oracle proves item B (the scan) under consistent input; it does **not** prove item A is inert on real data, and the two must not be conflated.
- `_load_fold`'s new RAM fail-fast is the one operator-facing behaviour change (opt out with `--allow-large`).
- `providers/sportec/parse.py` single-pass (ADR-031 lift) is byte-identical (lakehouse DFL golden), and silly-kicks now owns that parser (the lakehouse's delete-and-depend removed its copy).

## Attribution

Internal (optimization-audit remediation). Builds on ADR-019 (`id_compat`), ADR-052 (`_driver` resume), ADR-065 (order-insensitive converters).
