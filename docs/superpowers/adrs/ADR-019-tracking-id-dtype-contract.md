# ADR-019: Dtype-safe id contract at tracking-feature seams

| Field | Value |
|---|---|
| **Date** | 2026-06-06 |
| **Status** | Accepted (silly-kicks 4.15.0) |
| **Deciders** | Karsten Nielsen (maintainer), silly-kicks part-deux review sessions |

(ADR-015 is reserved by TF-17 PR-C. ADR-016 = ghost-GK served estimator; 017 = time-base contract;
018 = own-goal VAEP labels. This ADR is 019.)

## Context

Tracking-feature consumers compare SPADL-action identifiers against tracking-frame identifiers
(and against the scalar `home_team_id` argument) with raw `==` / `!=`, and merge action↔frame
frames on id-valued keys. Both assume the two sides share a dtype. When they do not, pandas
silently returns the wrong boolean (`Int64(366) == "366"` → `False`; `obj("366") != Int64(366)` →
`True`), or **raises** on a mixed-dtype merge key (`ValueError: You are trying to merge on int64
and object columns`). The result is silently-wrong actor / opponent / defending-GK /
defensive-line / possession / attacking-team resolution for any caller whose id dtype differs from
the library's.

The original framing ("GS frames force `Int64`") was a misdiagnosis. Verified facts:

- The library's own GS pipeline is **not** broken — GS SPADL actions are `int64`, GS frames are
  `Int64`, and `int64 == Int64` compares correctly (`'i'` kind both).
- The silent failure requires a **string** on one side. Confirmed against the Databricks lakehouse:
  `bronze.spadl_actions` ids are **bigint**, but `dev_gold.fct_tracking_frames` persists them as
  **string** — so the lakehouse runs string frames against numeric actions, the exact mismatch, and
  papered over it with a per-table coercion workaround.
- Therefore "align GS frames to object" (the original proposal) is the wrong lever — it would break
  pure-library numeric callers, not help the lakehouse, and violate ADR-001 (the GS `Int64` frame
  dtype is a deliberate PR-S18 convention).

## Decision

Introduce a **dtype-safe id contract at the tracking-feature consumer seams**, covering both
**comparisons** and id-valued **merge keys**, independent of any provider's id dtype.

### The shared primitive (`silly_kicks/tracking/_id_compat.py`)

One definition of "id identity": a single `_canonical` truth (scalar `canonical_id` + vectorized
`canonical_id_series` both delegate to it) that maps `366` / `366.0` / `np.int64(366)` /
`Int64(366)` / `"366"` → `"366"` (integral-float collapse), passes genuine strings through
(`"DFL-CLU-A"`), and maps NA → `pd.NA`. Built on it:

- `ids_equal` / `ids_differ` — element-wise, NA-safe (`ids_differ` requires **both present**, so an
  unmatched `how="left"` row is never mis-classified as "opponent"); return non-nullable `np.bool_`.
- `ids_match(series_or_array, scalar)` — vectorized `== scalar`, accepts `pd.Series` or `np.ndarray`.
- `same_id(a, b)` — scalar↔scalar (groupby-loop / per-row comparisons), `False` if either is NA.
- `align_join_keys(left, right, keys)` — **pre-merge** key alignment; `keys` accept `(left, right)`
  name pairs (e.g. `frame_id_int`↔`frame_id`). Prevents the mixed-dtype-merge `ValueError`.

**Performance:** a `_directly_comparable` predicate (same numpy kind **or** both object) gives
matched-dtype pipelines and genuine-string providers (sportec/kloppy) a raw fast path — **zero
canonicalization**. Coercion is paid only at an actual dtype boundary, and the hot
`_resolve_action_frame_context` masks canonicalize each id column **at most once** per call.

### The opt-in guard (mirrors ADR-017's `validate_time_base`)

`validate_id_dtypes(actions, frames, *, home_team_id=None, on_mismatch="raise") -> IdDtypeDiagnosis`
— a standalone public pre-flight a dtype-sensitive consumer (the lakehouse) calls to fail loud. It
is **not** threaded through the ~30 aggregators (the seam coercion already makes them correct), so
the public surface grows by exactly one function + one frozen dataclass.

### Why coercion is silent at the seams but the validator raises

String-canonical equality is always correct for like-to-like id comparison (same-match /
same-provider by construction), so a cross-dtype seam is not a per-match error to warn about (that
would spam the lakehouse every match). The diagnosis is where a caller opts into strictness.

### Completeness mechanism

A **red-first, asymmetric** behavioral gate (`test_id_dtype_invariance.py`) runs every registered
aggregator on an all-numeric baseline and on asymmetric permutations (numeric actions × string
frames, and the reverse, with `home_team_id` dtype as an independent axis), asserting feature
outputs are invariant. A meta-assertion forces the gate's enumerated surface to equal the registered
`add_*` surface, so a new aggregator cannot escape. A boundary-focused AST lint
(`test_id_compat_lint.py`) flags raw `== home_team_id` / cross-suffix comparisons in new primitives.
A structural perf guard asserts the de-dup holds (no id column canonicalized twice).

### Scope: boundary-focused, not total-rewrite

The package has ~50 id comparisons, but most are **frame-vs-frame** (a frame `team_id` compared to a
scalar derived from those same frames) — dtype-consistent by construction and safe. Only the
**boundary-crossing** seams (`home_team_id` scalar arg; action-derived id vs frame id; cross-source
merge keys) are routed through the helpers. The behavioral gate is the arbiter: a safe seam keeps
the gate green and is left untouched; a broken one fails and is fixed.

### Rejected alternatives

- **Numeric coercion** at the seams — destroys genuine-string ids (sportec `"DFL-CLU-A"` → `NaN`).
- **Force one dtype at converter output** — an ADR-001 break + a Hyrum break for every frame-id
  consumer, and it still does not fix the `home_team_id` scalar arg or string action ids.

## Consequences

- **New public surface:** `validate_id_dtypes` + `IdDtypeDiagnosis` (exported from
  `silly_kicks.tracking`). No new dependencies; `import silly_kicks` stays dependency-light.
- **ADR-001 preserved** — converter identifier conventions untouched; the fix lives at the consumer
  seams.
- **Latent bugs fixed (Hyrum — feature values change):** the contract corrects pre-existing
  silently-wrong behavior, so some feature values change for **numeric (pure-library) callers too**,
  not just string-id callers:
  - **`add_player_influence` / `add_cover_shadows`** team-vs-opponent labeling used
    `str(action_team) == str(frame_team)`, which broke because `DataFrame.iterrows()` upcasts a
    numpy-`int64` action `team_id` to `float64` (`str(5.0) != "5"`) while the nullable `Int64` frame
    side does not. Off-ball-xT and lane features were mis-split between `_team` / `_opponent`. Now
    correct via `same_id` (integral-float collapse).
  - The opponent-mask `ids_differ` both-present rule fixes a latent `how="left"` join-miss on the
    object path (an unmatched row was wrongly counted as "opponent").
  - `compute_das` opponent lookup, `_space_creation` actor match, and the ghost-GK / DL / off-ball
    merges shed ad-hoc `astype(str)` patches in favor of the shared primitive.
- **Lakehouse handshake:** may drop its string-coercion workaround and instead rely on the seam
  coercion, or call `validate_id_dtypes(..., on_mismatch="raise")` at work-unit entry. Flagged in the
  CHANGELOG as the cross-repo handshake.
- **C4-free** — no new KDE backend, trained model, or `add_*` aggregator → DSL tokens/count unchanged.

See ADR-001 (identifier conventions), ADR-017 (the `validate_time_base` guard this mirrors), ADR-005
(the atomic mirror, which composes the fixed aggregators).
