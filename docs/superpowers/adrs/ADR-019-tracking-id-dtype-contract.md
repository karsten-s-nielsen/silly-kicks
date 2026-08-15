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

## Amendment (4.21.1, 2026-06-09): the converter-adapter orientation seam is an ADR-019 boundary

**Context.** The 4.20.1 provider data-quality batch fixed BUG-4: the tracking adapters
`gradientsports.py` and `sportec.py` derived per-frame `team_attacking_direction` in
`convert_to_frames` via a raw `team_id == home_team_id`. When the caller passed an int
`home_team_id` against an object-string frame `team_id` (the lakehouse shape), the comparison
silently matched **zero** players → every player mislabeled → `play_left_to_right` double-flipped
the frame. This was the root cause of the `structural_sgm` away-team blow-up and latent corruption
of every frame-linked tracking feature on a dtype mismatch. The fix routed both through `ids_match`
and made a zero-match **fail loud**, with a per-adapter int-vs-str invariance test.

**Why it went uncaught.** This is a fourth ADR-019 id-dtype instance, and it slipped past the static
backstop because the AST lint (`tests/tracking/test_id_compat_lint.py`) **blanket-skipped** the
adapter modules. The original `ALLOW_MODULES` rationale — "a raw `== home_team_id` here is the
converter's OWN arg in the provider id space (ADR-001)" — conflated two different comparisons in the
same file: the genuinely provider-space jersey→roster mapping (ADR-001, safe) and the
**orientation seam**, where `home_team_id` is a caller-supplied argument of uncontrolled dtype
compared against the frame `team_id` Series — the *exact* ADR-019 boundary, not an ADR-001
exemption. A whole-file skip is the same unexamined-fence failure mode the contract exists to
prevent.

**Decision.** The converter-orientation seam is in scope of the boundary lint. `ALLOW_MODULES` is
narrowed from `{_id_compat.py, sportec.py, gradientsports.py, kloppy.py}` to **`{_id_compat.py}`** —
the helper module that defines and tests the primitives is the *sole* exemption; every tracking
module, converter adapters included, routes its id comparisons through the helpers.

- `gradientsports.py` / `sportec.py` — `convert_to_frames` already uses `ids_match` (the 4.20.1
  BUG-4 fix); un-skipping them puts the seam under the lint.
- `kloppy.py` — its orientation comparison is `str()`-vs-`str()` internal (`home_team_id` derived as
  `str(home_team.team_id)`, no caller-dtype boundary), so a raw `==` was already correct. It is now
  routed through `same_id` anyway, for **consistency** (one rule — adapters never compare ids raw)
  and so the whole adapter family stays under the lint with no per-module exemption to reason about.
  The change is behavior-identical (both sides already strings; `same_id`'s both-object fast path
  adds negligible per-player overhead); the earlier concern that this would "pessimize provably-correct
  code" was judged immaterial against the hot loop's existing pure-Python per-player dict-building.

Two guards are added so the narrowing cannot silently regress: a **discriminating proof** that the
detector actually fires on the BUG-4 shape (`out["team_id"] == home_team_id`) — distinguishing a
genuinely-clean adapter from a detector that never fires for this shape — and an **anti-regression
lock** pinning `ALLOW_MODULES == {_id_compat.py}` (no adapter can be re-exempted).

**Scope of the lint, unchanged.** The two flagged shapes (`== home_team_id`; cross-source `_action`
vs `_frame` suffix) are unchanged. The orientation seam is already shape 1, so no new detection
logic was needed — only the over-broad exemption was removed. The single library-code change
(kloppy's `==` → `same_id`) is **behavior-identical** (str-vs-str): no behavior change, no retrain
trigger (the BUG-4 *fix* shipped in 4.20.1; this amendment guards the *class*).
## Amendment (4.81.0, PR-S151) — the VAEP label seam, and a shape `~ids_equal` gets wrong

`vaep/labels.py` and `atomic/vaep/labels.py` compared ids with a raw `==`/`!=` at **sixteen** sites
(counted from the diff, not by eye: 8 + 4 + 4). ADR-027 guarantees Gradient Sports emits null-actor
rows whose `team_id` is NULL — as nullable `Int64` carrying NA, precisely so a sentinel cannot
bypass `pd.isna` routing — and the labels are a consumer seam this ADR governs. The sixteen sites
took **three** shapes, and only two of them crash:

| shape | sites | where | failure |
|---|---:|---|---|
| Series vs Series (`team_id == shifted_team`) | 8 | action + xG windows | nullable-boolean → a **`pd.NA` LABEL**; `.to_numpy()` gives object, and the harness's `np.unique(y[train_idx])` raises `TypeError: boolean value of NA is ambiguous` |
| scalar vs scalar (`team_id.loc[i] == team_id.loc[j]`) | 4 | possession windows | `if same_team:` on `pd.NA` raises immediately — the ADR-027 `_line_breaking.py` defect, different module |
| numpy element (`tid[i] != tid[j]`) | 4 | time windows | `tid = np.asarray(actions["team_id"].values)` turns nullable `Int64` into `float64` + `nan`, so `nan != nan` is **True**: the row is read as an OPPONENT and `_concedes_time` charges it with the goal. **No exception.** |

The third is why this is an amendment and not a routine fix. **`~ids_equal(...)` is the wrong
repair**: it satisfies the two crashing shapes while preserving the silent one, because negating
"same" promotes every NULL-team row to "opponent". A row with no team is **neither** the same team
nor an opponent, and only `ids_differ` — which requires BOTH ids present — can say so. The scalar
loop needed the same reasoning spelled out, since `not same_id(a, b)` has exactly the `~ids_equal`
defect:

```python
def _other_team_scalar(a, b) -> bool:
    return not (pd.isna(a) or pd.isna(b)) and not same_id(a, b)
```

**Found three hours into a TF-24 Stage-2 run**, by the crash — while the silent shape had been
mislabelling GS null-actor rows the whole time.

**A regression the fix itself introduced, now pinned.** `ids_equal`/`ids_differ` are **POSITIONAL**
and return a fresh `RangeIndex`. Combining that with a label-indexed Series LABEL-ALIGNS, producing
a UNION: a 400-row frame yielded **410** rows. Any filtered or sliced caller has a non-`0..n-1`
index, which is the normal case downstream. The idiom is to take `.to_numpy()` and reattach the
source index explicitly. **This is the third time this repo has hit it**, so the guard is behavioural
(`test_a_non_rangeindex_frame_keeps_its_length_and_index`), not prose.

**Retrain trigger, bounded.** `scores()`/`concedes()` return corrected values on any corpus carrying
null-actor rows — Gradient Sports today. Providers with no NULL `team_id` are byte-identical, which
is asserted rather than argued (`test_clean_int64_labels_are_UNCHANGED_by_the_na_routing`), so this
is a re-materialize for GS-derived label columns and a no-op elsewhere. No bundled artifact moves.

**Where this sits against ADR-043's enumeration registry.** It does not extend it. The registry is
complete over *id-scalar arguments of public functions*; these are internal comparisons between two
columns of one frame, which that surface does not reach — the same bound the 4.79.0
`_elastic_sync` dict-key defect established. The lesson repeats rather than extends: a
completeness-by-enumeration gate bounds **the surface it enumerates**, never the defect class.
