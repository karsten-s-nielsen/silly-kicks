# Tracking id-dtype safety contract — design

| Field | Value |
|---|---|
| **Date** | 2026-06-06 |
| **Status** | Draft (design approved; pending spec review) |
| **Target** | silly-kicks **4.15.0** (next free minor; reconcile against `origin/main` at release) |
| **ADR** | **ADR-019** (next free; reconcile at release) |
| **Author** | part-deux session |

## Problem

Tracking-feature consumers compare SPADL-action identifiers against tracking-frame
identifiers (and against the scalar `home_team_id` argument) with raw `==` / `!=`.
Those comparisons silently assume both sides share a dtype. When they do not, pandas
returns element-wise `False` (or `True` for `!=`) with **no error**, silently breaking
actor / opponent / defending-GK / defensive-line / possession resolution.

### What the original TODO claimed vs. what is actually true (verified 2026-06-06)

The TODO framed this as a **GS-specific schema bug**: `GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS`
forces `player_id`/`team_id` to `Int64` "whereas KLOPPY/SPORTEC use object (native strings)",
so `Int64(366) == "366" → False`, and proposed *"align GS frames to object"* as one fix.

Empirical fact-check **reframes it** — it is a cross-dtype comparison gap, **not** a GS schema bug:

1. **The library's own GS pipeline is not broken.** GS SPADL actions inherit the base
   `SPADL_COLUMNS` → `player_id`/`team_id` are **`int64`** (only the tackle-passthrough columns
   are `Int64`). GS frames are `Int64`. Empirically `pd.Series(int64) == pd.Series(Int64)` →
   `[True, True]` (pandas 2.3.3). A pure-library single-provider GS pipeline compares correctly.
2. **The silent failure requires a string on one side:**
   - `Int64 == "366"` → `[False, False]` — the `home_team_id="366"` scalar-arg case.
   - `Int64 == object(str)` → `[False, False]` — string action-id columns vs numeric frames.
   - `object(str) != Int64` → `[True, True]` — the opponent mask inverts: every row looks like
     an opponent.
3. **Backend confirmation (Databricks `soccer_analytics`):** `bronze.spadl_actions` has
   `player_id`/`team_id` = **bigint**, but `dev_gold.fct_tracking_frames` has them = **string**
   (the lakehouse persists frame ids as strings, platform-wide; `match_id` is even string in
   frames yet bigint in actions). So the lakehouse runs **string frames against bigint actions** —
   the exact silent mismatch — and papers over it with a string-coercion workaround after
   `convert_to_frames`.
4. **"Align GS frames to object" is the wrong lever.** The lakehouse already has string frames +
   bigint actions, so emitting object frames would (a) break pure-library GS users (`int64`
   actions vs `object` frames) and (b) not help the lakehouse. It also violates ADR-001 /
   Chesterton's Fence — the GS `Int64` frame dtype is a deliberate PR-S18 convention (NaN on ball
   rows; mirrors `GRADIENTSPORTS_SPADL_COLUMNS`).

**Reframed problem statement:** the library lacks an enforced **dtype-safe id contract at the
tracking-feature consumer seams** — covering both id *comparisons* and id-valued *merge keys*
(M1). The fix is dtype-safe comparison + join-key alignment + an opt-in loud validator —
independent of any single provider's id dtype.

## Decision (approved)

- **Contract intent:** *coerce + guard (both).* Internally normalize both sides of every id
  comparison **and id-valued merge key** so any caller dtype "just works", **and** expose a public
  dtype-mismatch validator that raises loudly when asked.
- **Implementation shape:** *canonical-string coercion* at the seams via a shared helper, with a
  **same-dtype fast path** (zero overhead when dtypes already match) and a **standalone public
  validator** (no parameter threaded through the ~30 aggregators).
- **Scope:** *comprehensive + CI gate.* Apply the helper at **every** id-comparison **and id
  merge-key** seam, and add a CI gate that prevents new raw id comparisons / unaligned merges from
  regressing the contract.

### Rejected alternatives

- **Numeric coercion (`pd.to_numeric`) at the seams** — destroys sportec/kloppy genuine-string
  ids (`"DFL-CLU-A"` → `NaN`). Not universal.
- **Force one dtype at converter output (`_finalize_output` casts frame ids to string)** — an
  ADR-001 converter-contract change + a Hyrum break for every frame-id consumer, and it still
  does not fix the `home_team_id` scalar arg or string action ids. More invasive, less complete.

## §1 — Shared primitive: `silly_kicks/tracking/_id_compat.py` (private)

The single anti-corruption seam for id identity. Pure functions, no I/O, no global state
(hexagonal charter). Lives in `tracking/` because the seam is tracking-feature consumption.

```python
def _canonical(x) -> str | pd.NA   # SINGLE SOURCE OF TRUTH (scalar element)
    # NA/None/NaN → pd.NA (never matches)
    # integral floats/ints collapse: 366, 366.0, np.int64(366), Int64(366), "366" → "366"
    # genuine strings pass through: "DFL-CLU-A" → "DFL-CLU-A"
    # non-integral floats (data corruption, e.g. 366.5) → "366.5" as-is (won't match — correct)

def canonical_id(x) -> str | pd.NA                   # scalar entry → delegates to _canonical
def canonical_id_series(s: pd.Series) -> pd.Series   # vectorized, NaN-safe, object dtype

def ids_equal(a: pd.Series, b: pd.Series) -> pd.Series   # np.bool_; NA never equals anything → False
def ids_differ(a: pd.Series, b: pd.Series) -> pd.Series   # np.bool_; present(a)&present(b)&(a!=b)
def ids_match(series: pd.Series, scalar) -> pd.Series     # np.bool_; series == scalar, vectorized
def same_id(a, b) -> bool                                 # scalar↔scalar (groupby-loop comparisons)
                                                          # returns False if either is NA (NA never matches)

def align_join_keys(left: pd.DataFrame, right: pd.DataFrame,
                    keys: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]
    # PRE-MERGE: canonicalize the named key cols on both sides to a common dtype before merge.
    # Per key: if both sides already share a numpy kind → no-op (fast path, zero cost);
    # else coerce both to canonical string via canonical_id_series. Returns (left, right)
    # ready for pd.merge(on=keys). Covers id-valued join keys (game_id/period_id/frame_id).
```

**Two distinct layers — comparison AND merge-key alignment (M1).** id dtypes interact at the
seams in *two* places, and they fail differently:
- **Post-merge comparison** (`ids_equal`/`ids_differ`/`ids_match`/`same_id`) — a cross-dtype
  comparison silently returns the wrong boolean.
- **Pre-merge join keys** (`align_join_keys`) — `pd.merge` on a key where one side is `int64` and
  the other is `object` **raises `ValueError: You are trying to merge on int64 and object
  columns`** (verified). It does *not* silently no-match. The comparison helpers operate on the
  *already-merged* frame and are structurally incapable of fixing this — the merge raises first.

Both layers route through the same `_canonical` truth, so "id identity" is defined once.

**One canonicalization truth (B2).** `canonical_id` (scalar) and `canonical_id_series`
(vectorized) **both delegate to the same `_canonical` definition** and are unit-tested against the
same table at both entry points. A naive `.astype(str)` is **forbidden** — verified
`pd.Series([366.0]).astype(str) → ["366.0"]`, which would *not* collapse to `"366"` and would break
equality against an `int64`/`Int64` `366`. The vectorized integral-collapse uses an `Int64`
round-trip for float-integral values (verified: `[366.0, 366.5, NaN] → ["366", "366.5", <NA>]`)
rather than per-element Python `.map`, so the hot long-frame columns coerce in one vectorized pass.

**Same-dtype fast path:** `ids_equal`/`ids_differ`/`ids_match` first check whether both sides
already share a numpy kind; if so they perform the raw comparison with **no stringify pass** —
preserving today's performance exactly for matched-dtype pipelines (verified `Int64.kind ==
int64.kind == 'i'`, so nullable/non-nullable ints raw-compare). Coercion cost is paid only when
dtypes actually differ. **Note (C3):** kind-based logic treats `float64` (`'f'`) vs `int64`
(`'i'`) as *not* same-kind, so a float-id provider takes the coercion path (and the validator
flags it) even though `366.0 == 366`; this is conservative and still correct (string-collapse
matches), just documented so a caller isn't surprised.

**Return dtype pinned (C1):** `ids_equal`/`ids_differ`/`ids_match` return a **non-nullable
`np.bool_` Series with NA already resolved to `False`**. In pandas 2.3.3 a nullable-boolean mask
with NA does not raise (it silently drops the NA row), but silent is the exact failure class this
contract exists to kill, and a nullable bool can propagate NA through downstream `.sum()`/
arithmetic. Pinning `np.bool_` makes behavior identical across pandas versions.

**Explicit NA semantics:**
- `ids_equal` → `False` wherever either side is NA (NA never equals anything).
- `ids_differ` → requires **both present** (so a NaN frame / ball row is never mis-classified as
  "opponent"). This fixes the `object(str) != Int64 → True` opponent-mask inversion. **It also
  silently fixes a latent left-join-miss bug (N1, intended):** the merge-based opponent seams are
  `how="left"`; an unmatched row gets NaN `team_id_dl`/`team_id_gk`. On the object/string path raw
  `NaN != "5" → True` wrongly includes the unmatched row as "opposing" (the numeric path already
  excludes it because `NA != 5 → NA →` mask-drop). `ids_differ`'s both-present rule makes **both
  paths** exclude it — a real correctness improvement on the object path, documented here so the
  behavior change is a recorded fix, not a surprise.

**Same-provider assumption (C2):** string-canonical equality could in theory match an action
`team_id` int `366` against a genuine-string frame `"366"` from a *different* provider. This cannot
occur because actions + frames at any seam are same-match / same-provider by construction — which
is precisely why string-canonical is "always correct for like-to-like" comparison.

**Why a module, not inline helpers:** the comparison shapes recur across ~12 primitives; one
tested definition of "id identity" + the CI gate can assert everyone routes through it.

## §2 — Public validator + guard (mirrors ADR-017 `validate_time_base`)

**`IdDtypeDiagnosis`** — frozen dataclass in `tracking/schema.py`, sibling to `TimeBaseDiagnosis`:

```python
@dataclasses.dataclass(frozen=True)
class IdDtypeDiagnosis:
    per_column: dict[str, tuple[str, str]]        # id col -> (action_dtype, frame_dtype)
    coercion_required_columns: tuple[str, ...]    # cols whose action/frame kinds differ
    home_team_id_dtype: str | None                # dtype/kind of the scalar arg, if supplied
    home_team_id_requires_coercion: bool          # scalar kind vs frame team_id kind
    message: str

    @property
    def has_mismatch(self) -> bool: ...
```

**`validate_id_dtypes(actions, frames, *, home_team_id=None, on_mismatch="raise") -> IdDtypeDiagnosis`**
in `utils.py`, exported from `tracking/__init__.py`:

- Pure `_diagnose_id_dtypes` core compares the numpy kind of each shared id column
  (`player_id`, `team_id`, `defending_gk_player_id` when present) across actions vs frames, plus
  the `home_team_id` scalar vs frame `team_id`.
- `on_mismatch="raise"` default — the explicitly-invoked guard fails loud, exactly like
  `validate_time_base` (the asymmetry with seam coercion is intentional). `"warn"` / `"ignore"`
  available. The diagnosis is returned under all policies.
- This is the **opt-in loud guard** a dtype-sensitive consumer (e.g. the lakehouse) calls at
  work-unit entry. It is **not** threaded as a parameter through the aggregators — the seam
  coercion already makes them correct, so there is no per-call failure to surface. API addition is
  exactly one public function + one dataclass (minimal Hyrum surface), consistent with how
  ADR-017 added `validate_time_base` standalone.

**Why coercion is silent at the seams but the validator raises:** string-canonical equality is
*always correct* for like-to-like id comparison, so a cross-dtype seam is not an error to warn
about per-match (that would spam the lakehouse every match). The diagnosis is where a caller opts
into strictness.

## §3 — Seam application + CI gate

### Seam inventory (4 comparison shapes) — expanded after the 2026-06-06 audit

| Shape | Helper | Sites (line-verified 2026-06-06) |
|---|---|---|
| merged-col `==` merged-col (actor, GK masks) | `ids_equal` | `_resolve_action_frame_context` (`player_id_frame==player_id_action` utils.py:600; `defending_gk_player_id==player_id_frame` :622, already `.notna()`-guarded) |
| merged-col `!=` merged-col (opponent masks) | `ids_differ` | `_resolve_action_frame_context` (`team_id_frame!=team_id_action` utils.py:610); **+ merge-based opponent seams with custom suffixes:** ghost-GK `team_id_action!=team_id_gk` (features.py:3786), defensive-line `team_id_dl!=team_id_action` (_kernels.py:861), off-ball-runs `team_id_dl!=team_id_action` (_off_ball_runs.py:291) |
| frame-col `==` `home_team_id` scalar (vectorized) | `ids_match` | `play_left_to_right` (utils.py:156), `_defensive_line.select_back_line_players` (:62), pitch-control direction |
| groupby-key / scalar `==` `home_team_id` (Python scalar) | `same_id` | `_defensive_line` (:206), `_gk_influence` (:307/:350), `_off_ball_runs` (:331/:353), `_line_breaking` (:241), `_player_influence` (:120), `_ghost_gk` (:759) |

Plus possession: `_ball_carrier` / `derive_team_in_possession` (team-id resolution from frames →
coerce the carrier's team against the in-possession team).

**Audit findings (drove the expansion):**
- The opponent comparison is **not confined** to `_resolve_action_frame_context`. Three other
  functions build their **own** merged frame with custom suffixes (`_gk`, `_dl`) and compare
  `team_id_action` against it. A manual inventory missed these; the behavioral CI gate is what
  guarantees they're covered. The AST lint must therefore key on `team_id_*` / `player_id_*`
  prefixes generally, not just the `_frame`/`_action` pair.
- `features.py:3775-3776` **already hand-codes** a `game_id.astype(str)` coercion **on both merge
  sides** immediately *before* `linked.merge(gk_ghost, on=["game_id","period_id","frame_id"])`
  (features.py:3779). Chesterton's Fence — now fully understood: it exists because **`pd.merge`
  raises on a mixed-dtype join key** (verified `ValueError`), not because of a comparison. It is a
  **pre-merge join-key alignment, a different layer** than the `ids_*` comparison helpers.
  - **Correction (M1):** an earlier draft said this contract "subsumes that ad-hoc patch; the
    workaround is removed and replaced by the shared helper at the comparison." That was **wrong** —
    deleting the coercion and relying on `ids_differ` (post-merge) would make ghost-GK **crash at
    the merge** in the lakehouse's string-frames × numeric-actions scenario, before any comparison
    runs. The workaround's *behavior is kept*; it is replaced by the shared `align_join_keys`
    primitive (same `_canonical` truth), not deleted.

### Merge-key alignment seams (M1) — route through `align_join_keys`

Every actions↔frames / frames↔derived merge that can see a string-id caller must align its id
join keys first. Known at-risk merges (confirm full set + exact keys at implementation):

| Merge | Keys | Site |
|---|---|---|
| ghost-GK defending-GK | `game_id, period_id, frame_id` | features.py:3779 (replaces the 3775-3776 hand-patch) |
| link_actions_to_frames internal | `period_id, frame_id` (+ time) | utils.py (the linker merge) |
| `_resolve_action_frame_context` | `period_id, frame_id` | utils.py:591 |
| defensive-line / off-ball derived merges | per-function | `_kernels.py`, `_off_ball_runs.py` |

`align_join_keys`'s same-kind fast path makes this a **no-op for pure-library matched pipelines**
(zero cost) and a one-time canonicalization only when a caller mixes dtypes. The M2 asymmetric CI
gate (below) raises → output ≠ baseline if any at-risk merge is missed.

**A1 — coerce once per merge, locally (decided, not boundary-mutation).** In
`_resolve_action_frame_context` the three masks re-coerce the same two columns
(`player_id_frame` at :600 and :622; `team_id_frame` at :610). Fix: coerce each suffixed id column
**once into a local canonical array** at the top of the mask block and reuse across actor/opp/GK.
This kills the duplication with **no mutation of `long`** → no dtype change flows into
`ActionFrameContext.actor_rows`/`defending_gk_rows` (consumed by kernels) → zero Hyrum surface.

*Rejected: normalize `long`'s id columns once at construction (the reviewer's "Consider").* The
audit shows there is **no single ingress boundary** — `long`, the ghost-GK `merged`, the DL
`merged`, and the off-ball `merged` are four independent merges. "Normalize at the boundary" is
therefore the same touch-count as per-merge local coercion, while additionally mutating columns
that propagate into kernels. Per-merge local coercion **is** the boundary pattern here, at strictly
lower risk. The scalar `home_team_id` / `same_id` paths legitimately have no single seam and keep
the lightweight scalar helpers (negligible cost — one comparison against one scalar).

### CI gate — two complementary layers (both non-e2e, regular suite)

1. **Behavioral dtype-invariance test (primary) — ASYMMETRIC, red-first (M2)** —
   `tests/tracking/test_id_dtype_invariance.py`. The invariant that matters is **the output is
   unchanged when the two sides have *different* dtypes**, because the production failure is
   asymmetric (the lakehouse runs *numeric actions × string frames*, §Problem.3).
   - **Baseline:** all-numeric (numeric actions × numeric frames × numeric `home_team_id`).
   - **Variants (each must equal baseline):** frames→string while actions stay numeric; actions→
     string while frames stay numeric; **`home_team_id` dtype varied independently** of the frame
     dtype (the scalar-arg case is a *separate* failure axis). Run the full mixed permutation set.
   - **Why the earlier homogeneous design was wrong:** casting *both* sides to string gives
     `string×string`, which already compares correctly on the **unfixed** code (`object==object`
     works), identical to `numeric×numeric` → `output_A == output_B` → the test passes with **no
     helper in place**. It could never go red. This repeats the "a value-neutral test must exercise
     the path that *can* change the value" failure class — the asymmetric variants are the fix.
   - **Red-first discipline:** run this against the **current (unfixed)** code first and confirm it
     goes **red** (asymmetric comparisons mis-resolve; at-risk merges raise) *before* the helpers
     land; green after. A variant that does not go red is not guarding anything.
   - The asymmetric `string frames × numeric actions` variant is also the detector for **M1** — a
     missed `align_join_keys` makes the merge raise → output ≠ baseline.
   - Enumerates the public surface from the existing `*_default_xfns` / `add_*` lists in
     `features.py`, so coverage tracks the real API (per the "audit functionality, not just tests"
     lesson).
   - **Meta-assertion (B3):** the test also asserts its **enumerated set == the actual registered
     public aggregator surface** (the `add_*` / `*_xfns` exports). Without this, a new aggregator
     that is added but *not* registered in the `*_default_xfns` lists would be silently skipped by
     the asymmetric harness — reintroducing the same "false green" the gate exists to prevent.
2. **Static lint (belt-and-suspenders)** — `tests/tracking/test_id_compat_lint.py`. An AST scan
   over `silly_kicks/tracking/` flags raw `==`/`!=` `Compare` nodes whose operand is a known id
   name (`home_team_id`, `*_id_frame`, `*_id_action`, `defending_gk_player_id`, or
   `team_id`/`player_id` when the other operand is `home_team_id`), with a small curated allowlist
   (`.notna()`, sentinel checks). Catches a raw comparison introduced in a *new primitive* before
   it is wired into any public aggregator.

## Testing strategy (TDD / hexagonal / e2e)

- **Unit (red-first):** the canonicalization table (`366` / `366.0` / `np.int64` / `Int64` /
  `"366"` / `NA` / sportec-string) run against **both** `canonical_id` (scalar) **and**
  `canonical_id_series` (vectorized) — same expected output from both, so the two entry points
  can't diverge (B2). Plus the NA semantics + `np.bool_` return (C1) of
  `ids_equal` / `ids_differ` / `ids_match` / `same_id`.
- **Per-seam regression (red-first):** reproduce each silent break (string ids → correct
  resolution); confirm the opponent-mask NaN inversion is fixed; cover the three merge-based
  opponent seams (ghost-GK, defensive-line, off-ball) surfaced by the audit.
- **Validator:** `validate_id_dtypes` raise / warn / ignore + diagnosis-shape tests, including the
  C3 float-vs-int conservative-flag behavior.
- **Production-scale perf guard (B1) — structural, not wall-clock.** Assert each hot id column in
  `_resolve_action_frame_context` is canonicalized **at most once per call** (spy on the coercion
  helper's call count / a `_row` shape invariant) on a realistic long frame (≥100k rows). A
  **structural** guard is chosen deliberately over a wall-clock ceiling: shared-CI wall-clock is
  flaky (a prior tracking feature failed CI at 501ms against a 500ms ceiling). This proves A1's
  de-duplication held and that the lakehouse's always-cross-dtype path pays the coercion exactly
  once per column, not 2–3× per merge. The structural guard proves *no duplication*, not
  *affordable absolute cost* — so (N2) an informational `pytest-benchmark` on the cross-dtype
  `_resolve_action_frame_context` path **does run in CI** (non-gating, reported) so a 10× blowup is
  *visible* rather than invisible; "we measured it once" beats "it's probably fine." One vectorized
  `Int64` round-trip on 100k rows is cheap, so this is expected to stay well within budget.
- **e2e-style:** real GS converter output (numeric) + a string-coerced copy proving identical
  features. The synthetic-fixture version runs in the normal suite; a PINING-gated variant only if
  a real fixture is required.

## Non-functional / packaging

- **No new dependencies.** numpy>=2.0 / pandas 2.3.x compatible.
- **C4-free:** no new KDE backend, trained model, or `add_*` aggregator → no DSL token/count
  change → no regen (per the C4 enumeration rule).
- **Hyrum's Law:** purely additive + bug-fixing. Pure-library matched-dtype pipelines are
  unchanged (fast path). String-id callers previously got silently-wrong features and now get
  correct ones — a deliberate correctness change (flag in CHANGELOG). New public surface:
  `validate_id_dtypes` + `IdDtypeDiagnosis`.
- **ADR-001 preserved:** converter identifier conventions are untouched; the fix lives entirely at
  the consumer seams.
- **Release:** single feature branch, single commit, PR at the end. Version + ADR numbers
  reconciled against `origin/main` at release (target 4.15.0 / ADR-019). No commit without
  explicit approval.

## Review disposition (cross-session review, 2026-06-06)

| Item | Disposition |
|---|---|
| **A1** — anti-corruption at comparison-granularity; lakehouse never hits fast path; `_resolve_action_frame_context` re-coerces columns | **Adopted (de-dup) + partial push-back.** Coerce each suffixed column once into a local array, reuse across masks. **Declined** boundary-mutation of `long`: the audit shows ≥4 independent merges (no single boundary) and mutation propagates into kernels — per-merge local coercion is the boundary pattern at lower risk. |
| **B1** — no production-scale perf guard on the always-slow cross-dtype path | **Adopted, refined.** Structural call-count guard (coerce ≤ once per column per call) on a ≥100k-row frame, *not* a flaky wall-clock ceiling; optional informational benchmark. |
| **B2** — `str(366.0)→"366.0"` trap; pin one impl for scalar + vectorized | **Adopted.** Single `_canonical` truth; naive `.astype(str)` forbidden; vectorized `Int64` round-trip; unit table runs both entry points. |
| **B3** — behavioral test must meta-assert enumerated surface == registered xfns | **Adopted.** Meta-assertion added to the invariance test. |
| **C1** — pin return to non-nullable `np.bool_`, NA→False | **Adopted.** |
| **C2** — state same-provider assumption | **Adopted** (§1). |
| **C3** — validator over-flags `float64` vs `int64` | **Adopted as documented note** (§1, validator). |
| Audit by-product | Seam inventory expanded: 3 merge-based opponent seams (ghost-GK, DL, off-ball) + an existing `game_id.astype(str)` ad-hoc workaround. |

### Revision-2 review (2026-06-06)

| Item | Disposition |
|---|---|
| **M1** — contract fixes post-merge comparisons; the ghost-GK `game_id.astype(str)` is a *pre-merge join-key* alignment; `pd.merge` **raises** on mixed-dtype keys, so deleting it regresses to a crash | **Adopted (blocking fix).** Verified `ValueError` on `int64×object` merge. Added `align_join_keys` to §1 as a first-class primitive; route the ghost-GK/linker/`_resolve`/DL/off-ball merges through it; **keep** the coercion behavior (replace the hand-patch with the shared primitive, do not delete). Corrected the erroneous "subsumes/removed" claim in §3. |
| **M2** — primary gate casts *both* sides to string (homogeneous) → green on broken code, never red | **Adopted (blocking fix).** Verified `numeric×numeric` and `string×string` are identical pre-fix. Rewrote the gate to **asymmetric** variants (numeric actions × string frames + reverse), `home_team_id` as an independent axis, baseline = all-numeric, **red-first against current code**. Same "exercise the path that can change the value" lesson. |
| **N1** — `ids_differ` both-present also fixes a latent `how="left"` join-miss on the object path | **Adopted as documented fix** (§1). |
| **N2** — structural guard proves no duplication, not absolute cost | **Adopted.** Informational `pytest-benchmark` now runs in CI (non-gating, visible). |

## Lakehouse handshake (Hyrum)

Once shipped, the lakehouse can **drop its string-coercion workaround** and instead either (a)
rely on the seam coercion directly, or (b) call `validate_id_dtypes(..., on_mismatch="raise")` at
work-unit entry to assert its actions/frames id dtypes before running features. Recorded in the
CHANGELOG as the cross-repo handshake.
