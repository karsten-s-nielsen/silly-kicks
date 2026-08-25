# Detection-aware provider visibility guardrails — design (rev 2)

> Part of the position-only-variants cycle. Prevents the class of failure that surfaced when the ghost
> position-only trainer aborted mid-corpus on stale kloppy-built SkillCorner frames (`visibility=None`).
> Rev 2 incorporates the first spec/plan review. Spec date: 2026-08-24.

## Context

The ghost model trains keeper *detection*, so its labels are only valid where the keeper was actually
observed. SkillCorner is the one provider that carries a per-player detection flag
(`_DETECTION_AWARE_PROVIDERS = {"skillcorner"}`, `_ghost_gk.py:281`); every other bundled provider is
`_FULLY_OBSERVED_PROVIDERS = {"gradientsports", "sportec", "idsse", "metrica"}` (observed by
construction). The **native** `tracking.skillcorner` builder preserves the flag as `visibility`
(`_loader_pining.py:588`, `is_visible = is_detected`); the **kloppy** gateway hard-codes
`visibility=None` (`_loader_pining.py:551,616,673`), which the contract treats as "detection
discarded".

`keeper_detection_mask(visibility, *, provider)` (`_ghost_gk.py:312`) already encodes the rule: for a
detection-aware provider whose `visibility` is **entirely null**, RAISE ("training on undetected
keepers means training on the interpolator", spec 4.3). This is correct and must not weaken
(Chesterton's Fence — it is what *caught* the bad data).

**The incident (reconciled).** The June `clean_cache` held 10 SkillCorner games, **all kloppy-built**
(all-null `visibility`), grouped at corpus positions 72–81. The ghost run reached them last;
`keeper_detection_mask` raised on each, and `for_each`'s "3 consecutive failures → abort" fired at the
third (absolute game **74/81**). So "aborted at 74/81" and "3 games in" describe one incident: three
*consecutive* SkillCorner failures beginning at position 72. The failure was **systematic** (every
SkillCorner game was kloppy-built), not a tail — verified by the all-consecutive-SkillCorner abort
signature. The guardrails below nonetheless also cover the *mixed* case (see Layer 2), so a future
partial rebuild can't reintroduce the abort.

**The gap.** The rule only fires **per-frame, inside the corpus pass**. The pre-flight
`validate_corpus_providers` (`train_ghost_gk.py:133`, called `:346`) validates provider
**classification** (`validate_provider`) but NOT that a detection-aware provider's frames actually
*carry* visibility. So a stale/kloppy corpus passes the pre-flight, then the mask raises deep in
extraction and `for_each` aborts ~an hour in with an ordering-dependent message.

## Decision

A **visibility-usability contract for detection-aware providers**, enforced at two layers around one
shared rule, failing **loud** with the native-rebuild remedy — never silently excluding.

### 1. One shared rule (single source)

`assert_detection_aware_visibility(visibility: pd.Series, *, provider: str) -> None` — RAISES iff
`provider in _DETECTION_AWARE_PROVIDERS and visibility.isna().all()`. Empty series raises too
(`pd.Series([]).isna().all()` is `True`), **matching the original `keeper_detection_mask` semantics**
exactly (no `len()` guard). `keeper_detection_mask` is refactored to call it.

**Not "byte-equivalent" — the honest claim (fixes review B2):** the RAISE (type + trigger, including
the empty case) and the returned MASK (`fillna(False)`) are preserved; the **message is unified** —
the shared rule uses a provider-generic message rather than the old `keeper_detection_mask:`-prefixed
one, because the same message is now surfaced from the materializer and the pre-flight too, where that
prefix would be misleading. A substring regression test (`match="tracking.skillcorner"`) is therefore
correct (we are *intentionally* changing the message, not pinning the old bytes); the mask output and
the empty→raise behaviour get their own equivalence assertions.

**Module placement (review M3 — DECIDED: B).** The taxonomy is a property of *provider data*, not of
the ghost model — Layer 1 proves it, since the general `materialize_tc3_frames` builder is not
ghost-specific and coupling it to `_ghost_gk` is a layering smell. This repo already ruled on exactly
this shape: 4.53.0 promoted `_id_compat` **out of** `tracking/` because "a mandatory seam is public
API by definition — the underscore was a false signal" (`PRIVATE_CONSUMERS.md:49-61`); a rule multiple
consumers must obey should not hide inside one consumer's private module. So:

- **Move** to a new neutral `tracking/_provider_visibility.py`: `_DETECTION_AWARE_PROVIDERS`,
  `_FULLY_OBSERVED_PROVIDERS`, `validate_provider`, and the new `assert_detection_aware_visibility`.
- **Keep** `keeper_detection_mask` in `_ghost_gk.py` — it is a ghost-specific *consumer* of the rule;
  it imports the taxonomy + rule from the neutral module and delegates.
- **Private, not public, not top-level.** Every consumer today is tracking-adjacent (a tracking model
  + two scripts); `providers/` does not need it, so the top-level `silly_kicks/_polygon.py` /
  `id_compat` position would be speculative (YAGNI). Promotion to public `silly_kicks.provider_visibility`
  is a later, deliberate step *if* an external consumer appears — the id_compat path.
- **True clean break via a MODULE alias, not bare imports (fixes review-2 MEDIUM).** `keeper_detection_mask`
  stays in `_ghost_gk.py` and still consumes `validate_provider` (`_ghost_gk.py:314`) and
  `_FULLY_OBSERVED_PROVIDERS` (`:315`). A bare `from ._provider_visibility import validate_provider,
  _FULLY_OBSERVED_PROVIDERS` would make both names attributes of `_ghost_gk` — so
  `from silly_kicks.tracking._ghost_gk import validate_provider` would keep working: a **transitive
  re-export**, functionally the shim this move claims to avoid — while `_ghost_gk._DETECTION_AWARE_PROVIDERS`
  (used only by the moved `validate_provider`, so not imported back) would break. That asymmetric
  half-clean break is exactly the silent-drift this refactor exists to kill. So `_ghost_gk.py` imports
  the **module** (`from . import _provider_visibility as _pv`) and references `_pv.validate_provider` /
  `_pv._FULLY_OBSERVED_PROVIDERS` / `_pv.assert_detection_aware_visibility` — **no moved name enters
  `_ghost_gk`'s namespace**, so `_ghost_gk.validate_provider` genuinely fails (the id_compat "fails
  loudly" property, now real). A negative test pins it (`_ghost_gk` exposes none of the four moved
  names).
- **Migrated in-repo sites (enumerated).** `scripts/train_ghost_gk.py:141`
  (`from ..._ghost_gk import validate_provider` → `..._provider_visibility`); the new materializer
  importer; and `tests/scripts/test_trainer_cache_and_providers.py:122`
  (`test_validate_provider_is_shared_not_duplicated`, currently `gg.validate_provider`) → import from
  `_provider_visibility`, so the "single source" test pins the *new* home.
  `test_keeper_detection_mask_still_rejects_an_unknown_provider` (`:135`) needs **no** change —
  `keeper_detection_mask` stays in `_ghost_gk` and still validates.
- **Packaging (verified):** `[tool.hatch.build.targets.wheel] packages = ["silly_kicks"]`
  (`pyproject.toml:154-155`) auto-includes a new module under `silly_kicks/tracking/`; the only
  `exclude`s are the two `full/` weight dirs (`:160`), neither matching a `.py`. No manifest change.
- **`PRIVATE_CONSUMERS.md`:** a one-line entry in the **in-repo first-party** table (beside
  `_model_eval.py` / `_cover_shadows.py`) — the module's only consumers are in-repo and reached by
  real `import`s, so a rename fails loudly in-repo; not a lakehouse row.

Blast radius verified: the lakehouse pins the `_ghost_gk.py` module **path** for its ADR-044 drift
guard (`PRIVATE_CONSUMERS.md:18`) — a new *sibling* module does not touch that, and `_ghost_gk.py` is
not removed; no external consumer pins `validate_provider` / `_DETECTION_AWARE_PROVIDERS` as symbols
(grep of both lakehouse copies returned zero symbol imports — review-2 confirmed).

### 2. Layer 1 — build-time (shift-left), materializer

Extract a **module-level** seam `_guard_provider_frames(frames: pd.DataFrame, provider: str) -> None`
in `materialize_tc3_frames.py` (fixes review B1 — `_work` is a nested closure, not importable, so the
guard must be a testable module-level function that `_work` calls). For a detection-aware provider it
**(a) requires the `visibility` column to be present** — raising if absent (fixes review M2: a future
regression that *drops* the column, not just nulls it, is exactly a discarding regression this guard
exists to catch) — and **(b) calls the shared rule** on it. `_work` calls `_guard_provider_frames`
before returning frames (`materialize_tc3_frames.py:255`). Consumer-agnostic build-time regression
guard; the current native path already carries `visibility`, so this fires only on a future revert.

### 3. Layer 2 — consume-time pre-flight, ghost trainer

**Single provider resolution (fixes review S1).** The trainer reads a **flat `for_each` generation
directory** (`materialize_tc3_frames.py:260-261`; shard stems are `join_key((provider, match_id))`,
`:237`) — there is **no `{provider}/` path segment**, so path-based resolution silently fails on the
production layout. The existing discovery loop (`train_ghost_gk.py:337-346`) already reads
`source_provider` from every file and then discards the per-file mapping. Capture it once as
`provider_by_path: dict[Path, str]`; feed its values to `validate_corpus_providers` and the map to the
new `validate_corpus_visibility`.

**Mechanism = parquet `null_count` metadata (adopts review S3).** Verified on a real native shard:
`visibility` is arrow `bool` / pandas `object`, and its nulls **are real parquet nulls** (metadata
`null_count` 46962 of 1,080,126 = the observed 4.3%), **not** float-NaN — so the metadata path is
reliable here (the float-NaN caveat the review raised does not apply; if a future producer stores
`visibility` as float, re-check). `validate_corpus_visibility` reads, for **every** detection-aware
shard, the per-row-group `null_count` of `visibility` (metadata-only, zero data pages) and RAISES if
`null_count == num_rows` (all-null). Reading *every* detection-aware shard (not "sample one") catches a
**mixed** corpus as well as a systematic one — closing the review's load-bearing "one question" (a
tail-kloppy rebuild can't slip past). It also requires the `visibility` column present for a
detection-aware shard (M2, consume side).

**Completeness boundary (review-2 Minor 2).** `provider_by_path` is built only from shards that
declare a `source_provider` column — the discovery loop `continue`s past a file that lacks it
(`train_ghost_gk.py:340`), and that case is already handled downstream (`prov = "unknown"`, which
`validate_provider` rejects at extraction). So "reads every detection-aware shard" is precisely "every
detection-aware shard that declares `source_provider`". The materializer always writes
`source_provider`, so within a corpus this materializer produced the guarantee is complete; the
residual (a hand-built shard with no `source_provider`) falls to the existing extraction-time reject,
not to Layer 2.

### 4. Behaviour: RAISE, never silent-exclude

An all-null detection-aware shard is a data-quality defect (detection discarded). Silently excluding it
would train on a quietly-truncated corpus while reporting success — the quiet-truncation failure
`collect_home_team_map` already guards against (`test_materialize_tc3_frames.py:103,114,137`) and the
ADR-043 "an all-NaN column is indistinguishable downstream" discipline. The raise names the remedy
(rebuild via `tracking.skillcorner`). The `for_each` 3-consecutive-abort **stays** as a general
backstop; the pre-flight simply makes it not fire for the visibility class.

**Input-population note (review M1).** The shared rule is applied to **raw per-player** `visibility` at
Layers 1/2, whereas the training filter runs on **keeper-only** `meta["gk_visibility"]`
(`train_ghost_gk.py:606`). For the systematic/mixed all-null case these coincide (kloppy nulls
*everyone*, keeper included). Layer 2 passing does **not** guarantee the keeper-only gate passes (a
hypothetical outfield-detected / keeper-null shard) — that residual is the per-frame
`keeper_detection_mask`'s job at train time, unchanged. The spec claims only "same rule, different
population; the guard covers the systematic discard, not a keeper-specific null".

## Alternatives considered

- **Sample one shard per provider (rev 1's approach).** Superseded by reading every shard's
  `null_count` — same cost class (metadata-only) but complete (catches mixed).
- **Pre-flight only (no build-time layer).** Rejected: misses a future materializer regression; the
  build-time layer is one shared call.
- **Exclude-and-count.** Rejected for the systematic case (quiet truncation); the per-frame mask +
  `for_each` backstop already handle genuinely per-item failures.

## Testing (TDD, red-first)

- **Shared rule:** all-null SkillCorner → raises (`match="tracking.skillcorner"`); non-null SkillCorner
  → no-op; **empty SkillCorner → raises** (matches original semantics — pins B2's preserved behaviour);
  fully-observed provider (gradientsports) all-null → no-op.
- **`keeper_detection_mask` after refactor:** same raise on all-null + empty; **mask output unchanged**
  on a mixed series (equivalence assertion on the returned array); all-True for a fully-observed
  provider.
- **Clean-break negative test (review-2 MEDIUM):** `import silly_kicks.tracking._ghost_gk as gg`;
  assert `gg` exposes **none** of the four moved names (`validate_provider`,
  `assert_detection_aware_visibility`, `_DETECTION_AWARE_PROVIDERS`, `_FULLY_OBSERVED_PROVIDERS`) — so
  the old import paths fail, and a future bare `from ._provider_visibility import validate_provider`
  regression (re-introducing the transitive re-export) turns the test red.
- **Layer 1:** `_guard_provider_frames` on a detection-aware frame with all-null `visibility` → raises;
  with the column **absent** → raises (M2); native (visibility-bearing) → passes; fully-observed
  all-null → passes. Plus an AST/wiring assertion that `_work` calls `_guard_provider_frames`.
- **Layer 2 (fires BEFORE extraction — review B3):** drive the trainer entrypoint (`_load` idiom) on a
  data-dir whose SkillCorner shard is all-null, with `GhostGkModel.fit` (and ideally the extractor)
  monkeypatched to a spy; assert it RAISES and the spy's call-count is 0 — the
  `test_unclassified_provider_fails_BEFORE_any_fitting` idiom
  (`test_trainer_cache_and_providers.py:106-119`). Co-locate here (review S2), not in the tracking
  unit-test file. Both sides: a native / fully-observed-only data-dir passes.
- **Non-vacuity:** the raise fires only on detection-aware + all-null, never on a fully-observed
  provider carrying a null column.

## ADR & scope

ADR-068: the detection-aware visibility-usability contract + two-layer enforcement;
cross-refs ADR-038 and spec 4.3; names the M3 module-coupling decision taken. Additive — no model
change, no retrain trigger. Verification (review M4): run `.venv`/`.venv312` interpreters **directly**
(not `uv run` — the `[train]` extra breaks its resolution here), capturing **all** `FAILED` lines
(never a `tail`-truncated log).
