# Corpus-driver resilience — design

**Date:** 2026-07-29
**Status:** proposed
**Scope:** `scripts/` only — no `silly_kicks/` change, so the wheel is identical to its predecessor
**Motivating case:** `scripts/validate_xs_probe.py`

---

## 1. Executive summary

Twenty-one drivers in `scripts/` do expensive per-item corpus work. **Three** survive a crash.
**Fourteen** hold every result in memory and write once at the end, so a failure at hour 13 of a
14-hour pass loses all of it. **Four** have no per-item loop at all but re-pay an uncached Databricks
query on every invocation.

This is not a missing convention. **Four** separate partial mechanisms already exist in the repo,
covering seven drivers between them (§3) — and two of the four are near-duplicates that share no
code: `scripts/_cache.py` writes `cache_meta.json` beside a `features.parquet`, while
`train_ghost_gk` writes `cache_token.txt` beside its own `features.parquet`, with different validity
rules. **None of the four owns the loop**, which is precisely why resume and progress are the parts
that keep being omitted: there is nowhere for them to live except in each author's memory.

The cycle extracts one shared seam, `scripts/_driver.py`, whose `for_each` owns the loop and
therefore owns per-item persistence, resume, progress and failure accounting. It **consumes** the two
mechanisms that are already shared modules (`_partition.py`, `_cache.py`) rather than replacing them,
and **absorbs** the two that are driver-local. The four loop-free drivers get an opt-in cohort cache
from the same module. A rewritten CI gate checks **adoption of the helper**, not the presence
of capability keywords — the keyword version of this gate was drafted first and handed clean passes
to three drivers that are demonstrably accumulate-then-write.

Two honest limits, recorded here rather than discovered later: the staleness token cannot be checked
for completeness, and an adoption gate cannot check correctness. Both are stated in §7.

---

## 2. The problem, measured

Population and classification are derived by AST over `scripts/*.py` (excluding `_`-prefixed
modules). A driver is in-population if it calls a corpus loader (`load_matches`,
`select_match_ids`, `load_xtgk_cohort`, `load_retention_cohort`) or exposes a corpus-shaped CLI
(`--data-dir`, `--match-ids-json`, `--max-per-provider`, `--providers`) **and** contains a per-item
loop.

| Shape | Count | Meaning |
|---|---:|---|
| A — accumulate-then-write | **14** | per-item loop, no write inside it; a crash loses the whole run |
| A — resumable | **3** | write inside the loop, guarded by an existence check |
| B — no loop | **4** | one uncached cohort query, then whole-cohort analysis |

Full table:

| Driver | Shape | Existing mechanism | Progress output |
|---|---|---|---|
| `build_gkdv_arm_values` | resumable | `_partition` | no |
| `build_layer2_spells` | resumable | `_partition` | yes |
| `validate_xshot_causal` | resumable | `_partition` | yes |
| `calibrate_tracking_defaults` | accumulate | — | no |
| `calibrate_xt_bandwidth` | accumulate | own `--corpus-cache` (no token) | no |
| `derive_opengoal_range` | accumulate | — | no |
| `measure_cover_shadow_argmax_agreement` | accumulate | — | no |
| `run_signoff_power` | accumulate | — | yes |
| `train_ghost_gk` | accumulate | own `_feature_cache` + `cache_token` | yes |
| `train_gk_completion` | accumulate | — | yes |
| `train_xcross_attempt` | accumulate | `_cache` | yes |
| `train_xshot_occurrence` | accumulate | `_cache` | yes |
| `tune_structural_pass_sigma` | accumulate | — | no |
| `validate_shot_goalmouth_sb` | accumulate | — | no |
| `validate_xcross_causal` | accumulate | — | no |
| `validate_xs_probe` | accumulate | — | no |
| `validate_xtgk_possession_value` | accumulate | — | no |
| `train_gk_retention` | no loop | — | no |
| `validate_xtgk_v2` | no loop | — | no |
| `xtgk_v2_kappa_sweep` | no loop | — | no |
| `xtgk_v2_keeper_discrimination` | no loop | — | no |

### 2.1 The detector's own first version was wrong

The first classifier counted `.append` calls only. `.extend` and dict-subscript accumulation
(`out[match_id] = value`) scored as "no per-item state", which misclassified
`derive_opengoal_range` (`.extend`) and `validate_xtgk_possession_value` (subscript) as loop-free
exemption candidates. The corrected detector counts `append`/`extend`/`update`/`add` **and**
subscript assignment. The counts in §2 are the corrected ones.

This is recorded because the same correction is load-bearing for the CI gate's population
derivation in §5.

A second detector error, found during spec self-review: the "existing mechanism" column was
populated by a substring test (`"_cache" in src`), which matched `--cache-dir`, `corpus_cache` and
`_feature_cache`. It credited three drivers with using `scripts/_cache.py` when only
`train_xshot_occurrence` and `train_xcross_attempt` import it. The column above is corrected to
genuine imports. The correction is what surfaced the fourth mechanism in §3 — a substring test
reported *more* adoption than exists, exactly inverting the finding.

Both errors are the same shape as the false passes in §5, and all three argue for the same
conclusion: **substring and keyword tests over source are not evidence of behaviour.**

### 2.2 Reconciling the driver count

The figure of 16 unhardened drivers, used when this cycle was scoped, came from the first detector
and its hand-written debt list. Under the corrected detector the population is **21**: 14
accumulate-then-write and 4 loop-free need changes (18), and the 3 already-resumable drivers are
migrated behaviour-preservingly (§6 step 3). The scope decision — all of them, no exclusions — is
unchanged; only the count is.

### 2.3 The motivating case

`scripts/validate_xs_probe.py:94` iterates `load_matches(...)`, appending to `per_variant_deltas`
(`:105`) and `per_match` (`:116`). The loop body prints nothing. Analysis begins at `:132` and the
only write is `_write` at `:277-281`. It ran for roughly 14 hours over ~80 matches with no
intermediate output and no way to observe progress.

**It is already partitionable.** 4.63.0 added `--match-ids-json` to it. That is the sharpest
statement of the problem: the parallelism was added without the resumability, so N workers each lose
their entire slice on a crash instead of one process losing everything.

### 2.4 Shape B is not cheap

The four loop-free drivers each open with `load_xtgk_cohort` or `load_retention_cohort` — a
Databricks query — and cache nothing. Their entire CLI surface is `--provider` (plus `--variant` for
`train_gk_retention`). Every re-run, including a re-run that changes only an analysis constant,
re-pays the query.

---

## 3. What already exists

| # | Mechanism | Provides | Consumers |
|---|---|---|---|
| 1 | `scripts/_partition.py` | `worker_tag`, `list_match_ids`, `providers_for_slice`, `write_table_atomically`, `aggregate_manifests` | `build_gkdv_arm_values`, `build_layer2_spells`, `validate_xshot_causal` |
| 2 | `scripts/_cache.py` | `corpus_fingerprint`, `write_cache_meta`, `cache_is_valid`; `features.parquet` + `cache_meta.json`, absent meta ⇒ MISS | `train_xshot_occurrence`, `train_xcross_attempt` |
| 3 | `train_ghost_gk`'s `_feature_cache/` | `features.parquet` + `cache_token.txt`; token derived from `_PENALTY_AREA_Y_MIN/_MAX/_X` | `train_ghost_gk` |
| 4 | `calibrate_xt_bandwidth`'s `--corpus-cache` | read-if-exists / else assemble-and-write parquet; **no token** | `calibrate_xt_bandwidth` |

Mechanisms 2 and 3 are the same idea implemented twice: a validity marker written beside a
`features.parquet`, differing only in file name and validity rule, sharing no code. Mechanism 4 is
the Shape-B cohort cache of §4.3, already built once — and built without a staleness token, so a
re-materialized corpus is silently reused. That is the concrete instance of the hazard §4.3 exists
to avoid, not a hypothetical.

Distinct from all four, and out of scope: `--data-dir` / `--cache-dir` on `train_ghost_gk` and
`validate_shot_goalmouth_sb` is an *input* artifact cache (the TC3 pining layout), not a cache of
computed results.

`cache_token`'s docstring already states the principle this spec generalizes:

> Deriving rather than hand-bumping is the whole point: a literal version string goes stale inside
> the very re-fit cycle it exists to protect.

The gap is structural, and the two properties are split cleanly across the mechanisms:

| | fine-grained resume | staleness protection |
|---|---|---|
| `_partition.py` (1) | yes | **no** — a code change silently reuses stale shards |
| `_cache.py` / `cache_token` (2, 3) | **no** — all-or-nothing | yes |
| `--corpus-cache` (4) | **no** | **no** |

No mechanism has both, and none owns the loop, so none can supply resume *and* progress to a driver
that does not already have them.

The shard drivers carry the trainers' bug in mirror image. `_partition.py`-based resume keys on
shard-file existence alone, so **no** change to the code that produced a shard can invalidate it.
For `validate_xshot_causal` the genuinely exposed inputs are `causal/opportunities.py` and
`shot_arm_config` (imported at `:238`, inside `build_shards`) plus the model metadata — **not**
`causal/matching.py`, which is imported at `:127` inside `_entanglement_analysis` and therefore
re-runs from the shards on every invocation. Whether any specific stale reuse occurred is not
asserted here; the exposure is structural and provable from the code, and that is what the token in
§4.2 closes.

**Reconciliation is NOT in `_partition.py`.** That module reads only `manifest_*.json` (`:110`).
Shard-to-table reconciliation is driver-local and glob-based, at four sites:
`build_gkdv_arm_values.py:257`, `build_layer2_spells.py:136`, `validate_xshot_causal.py:288` and
`:331` (the `n_shards` provenance field), each a bare `glob("*.parquet")`. This constrains the token
encoding in §4.1 and is the reason it is a directory rather than a filename suffix.

---

## 4. Design

### 4.1 `scripts/_driver.py`

A new module beside `_partition.py`, `_cache.py` and `_provenance.py`. It **consumes** the first two
rather than replacing them: `_partition.py` keeps partitioning and *manifest* aggregation (it reads
`manifest_*.json` only — shard reconciliation is driver-local, see §3), and `_cache.py`'s fingerprint
generalizes into the token contract.

Named `_driver.py`, not `_corpus_pass.py`: `scripts/_corpus.py` already exists and holds the corpus
*taxonomy* (`is_public_row`, `assert_public_corpus`, the visibility triples that feed
`corpus_fingerprint`). Two modules whose names both begin `_corpus` would be misread, and the
distinction — taxonomy versus execution — is exactly the one that must stay legible.

```python
def for_each(
    items,                          # ITERABLE, consumed LAZILY -- never listed (see below)
    *,
    key,                            # callable(item) -> str | tuple[str, ...]; REQUIRED (see below)
    work,                           # callable(item) -> DataFrame | None
    shard_root,                     # Path; generation dirs live under this
    token_inputs,                   # REQUIRED (see 4.2)
    token_reason=None,              # REQUIRED iff token_inputs == {}
    counters=None,                  # callable(item, result) -> dict; per-item manifest counters
    tag="all",                      # partition name, from _partition.worker_tag
    label="item",
    max_consecutive_failures=3,
) -> CorpusPassResult              # carries .shard_dir — the generation dir callers glob
```

Properties:

**`items` is STREAMED, never materialised.** `for_each` iterates with `enumerate` and never calls
`list(items)`. The corpus loader is the reason: `load_matches` returns an `Iterator` that downloads
and parses each match — actions *and* a full tracking DataFrame — inside the loop before yielding,
and its own docstring says `max_per_provider` "bounds total memory ... loading all matches at full
depth can OOM". Listing it would hold ~80 matches' frames resident at once, and would also defeat
resume: nothing could be skipped until everything had been downloaded. It would invert this design's
own thesis, which indicts fourteen drivers for holding every *result* in memory — inputs are far
larger than results.

**What streaming costs, stated rather than glossed.** Two properties are given up, both deliberately:

1. **The injectivity check fires at the colliding item, not before any work.** A non-injective `key`
   costs one item's compute instead of failing in the first second. What it still prevents is the
   failure that matters — the second item is never silently counted as `skipped` — so conservation
   can never certify a run that dropped data. Drivers on the primitives path, which enumerate their
   keys into a list anyway, keep the cheaper pre-loop form (`_require_injective`).
2. **The total is unknown.** A generator has no length, so `progress` takes `n: int | None` and
   renders `[3/?]` when it is `None`; `for_each` passes `None` at every call site. A driver that
   genuinely knows its total — the primitives path, or a materialised list — passes it and gets
   `[3/64]`. Counting the corpus just to render a denominator would reintroduce exactly the
   materialisation this avoids.

**Owns the loop.** Per architecture fork C, `for_each` is the default shape. A driver whose loop
genuinely cannot invert — measured: `calibrate_xt_bandwidth` has 4 corpus loops,
`calibrate_tracking_defaults` has 3, and six have 2 (`measure_cover_shadow_argmax_agreement`,
`run_signoff_power`, `train_ghost_gk`, `validate_shot_goalmouth_sb`, `validate_xcross_causal`,
`validate_xshot_causal`) — may use the exposed primitives instead, but must record why (§5).

**Resume by construction, with the token as a DIRECTORY.** The shard path is
`shard_root / token / f"{key}.parquet"` — **not** a token suffix on the filename. A changed token
yields a different generation directory, so a stale shard can be neither read nor half-overwritten;
the failure mode is unrepresentable rather than merely guarded. Stale generations stay visible on
disk as intended, and `--prune-stale` becomes a directory removal.

The directory form is load-bearing, not cosmetic. Reconciliation is driver-local and every existing
site is a bare `glob("*.parquet")` (§3), so a **filename** suffix would make the combined table
concatenate the old-token and new-token shard for the same match — with different values — the first
time a declared input changed. `n_shards`, a provenance field at `validate_xshot_causal.py:331`,
would overcount by the same factor. That is precisely the defect class `_partition.py:8-11` was
extracted to prevent ("a false self-description in a provenance-bearing file"), reintroduced one
layer down, and it would silently corrupt the tables feeding
`run_signoff_power --spells/--arm-values` — numbers CLAUDE.md, TODO.md and CHANGELOG.md already
quote as measured. A generation directory keeps every existing glob correct once scoped to
`CorpusPassResult.shard_dir`, and makes migrating the three resumable drivers a path-prefix change,
so §8's "existing shards are reusable" stays true via a one-time `mkdir <token>` and move.

**A required, validated `key`.** Existing shards are named `f"{provider}__{match_id}.parquet"` at all
three sites, and `validate_xshot_causal.py:266-268` already warns that a provider containing `__`
"would silently mis-split" — while
`test_validate_xshot_causal_shards.py::test_cluster_key_distinguishes_providers_sharing_a_game_id`
exists because providers demonstrably share `game_id`s in this corpus. So a bare `match_id` key would
let two providers overwrite each other's shard while the resume check reports a hit. `key` is
therefore required, returns a `str` or a tuple the helper joins with a separator it **validates** —
rejecting any component containing the separator, loudly — and its components are retained as
columns in the shard. This is the same trap `_partition.py:44-57` documents for
`providers_for_slice`.

**Progress centrally.** Unbuffered stdout is forced once for every adopter (the trick
`train_ghost_gk.py` already applies locally, with the comment *"so background tasks show progress
immediately"*), and each item emits an index/total/elapsed line — with the total rendered `?`
whenever it is unknown, which for a streamed corpus is always (above).

**Per-item failure is counted, not fatal.** One bad match must not cost fourteen hours. A failing
item records its exception in the manifest and the pass continues; `max_consecutive_failures`
consecutive failures abort. The count is load-bearing for the same reason
`n_degenerate_by_size` is in `causal/power.py`: a pass that failed 60 of 64 items and returned a
short clean table is worse than one that crashed, because the table looks like an answer.

**The `work` return contract is a tidy frame plus manifest counters.** `work` returns ONE long-form
DataFrame (or `None`, meaning *zero rows* — which still writes a shard, see below), never a dict of
frames; per-item scalars go through `counters` into the manifest. This is stated now because the motivating driver cannot be expressed otherwise:
`validate_xs_probe` accumulates `per_variant_deltas`, a dict of variant to list of frames (`:105`),
*and* `per_match`, a list of counter dicts (`:116`). Under this contract variant becomes a column,
and `n_contributing` (`:133`) becomes a manifest sum rather than an in-memory scan — reusing the
int-and-dict merge `aggregate_manifests` already performs at `_partition.py:129-145`. A union return
type was rejected: every adopter would have to branch on it. If the contract does not hold for all
fourteen drivers, that is far cheaper to learn in step 1 than in step 6.

**Manifest aggregation stays `_partition.py`'s, and gains one field.** `for_each` writes a
per-worker `manifest_<tag>.json` in the shape `aggregate_manifests` already sums, so the
contribution-gated commit-consistency rule is inherited rather than re-implemented. `for_each`
additionally reconciles its own generation directory and writes the combined table via
`write_table_atomically` — the piece that was previously copy-pasted into each driver.

**The manifest's `n_attempted` counts true attempts and EXCLUDES skips.** Not the
`attempted + skipped` conservation quantity, which is `assert_conservation`'s. The distinction is
load-bearing rather than cosmetic: `_partition.py:128` strips a manifest of its vote on commit
consistency only when it *positively declares* it built nothing. A full-resume pass declares
`n_matches: 0` today and correctly abstains; writing `attempted + skipped` would make it declare
`n_attempted: 64`, regain its vote, and reproduce the §3.3 entanglement `commit_consistent: false`
false alarm that `_partition.py:91-101` was written to kill. Measured, on real manifests through
the real helper. Two quantities, two names.

**The combined table stays at `dest/`, and the generation token goes in the manifest.** Not inside
the generation directory: `run_signoff_power --spells/--arm-values` reads `dest/layer2_spells.parquet`
today, and moving it would be a Hyrum break on a documented CLI contract. But keeping it at `dest/`
means two generations write the same path and the file is whichever token ran last —
`write_table_atomically` makes that atomic, not *attributable*, and a reader cannot tell whether the
combined table matches the generation directory beside it. For a cycle whose central argument is
`_partition.py:8-11` ("a false self-description in a provenance-bearing file"), leaving that implicit
would be self-defeating. So `for_each` records the generation token as a manifest field, **and `aggregate_manifests` gains
named handling for it** — emitting `generations_seen` and `generation_consistent`, mirroring
`commits_seen`/`commit_consistent` exactly.

The helper does **not** collect arbitrary strings: its field loop special-cases `run_commit` and
`run_tree_dirty` by name, sums ints, merges dicts, and silently drops a `str`. An earlier draft of
this section asserted the opposite and a test certified it by never exercising the helper — a green
test on an absent feature, which is the exact false-green shape §5 is about. Measured: two workers
on different generations produce an aggregate with no `generation` key and `commit_consistent:
True`.

**Honest ceiling.** Even with the named handling, the combined table at `dest/` is still not
*attributable* to a generation — last-writer-wins is a race. What this buys is **detection** that a
mixed-generation corpus exists, not attribution of the table.

**And the general trap, closed.** The drop rule is a hazard for every future manifest field, and it
caught this cycle twice — `generation`, then `run_tree_state` one task later, in the same revision
that first wrote the rule down. Dropping stays the behaviour (a named case carries per-field
semantics: `run_commit` is contributor-gated, `run_tree_dirty` is OR-ed, `generation` is a set plus
a consistency flag — one generic collector would give all of them one wrong semantic). What changes
is that `aggregate_manifests` now emits **`dropped_fields`**, so a field that never reaches the
corpus artifact is visible in the artifact instead of absent from it. This belongs in the ADR's
consequences: *a manifest field needs a named case in `aggregate_manifests`, or it stops at the
per-worker file.*

**The primitives surface, enumerated.** Fork C's exception path is not "whatever is importable" — a
step whose purpose is to prove the primitives adequate cannot produce a crisp finding against an
unspecified API. `_driver.py` exports, and `for_each` is itself composed from:

| Primitive | Responsibility |
|---|---|
| `preflight(*checks)` | Refuse before spending — see below |
| `generation_dir(shard_root, token_inputs, token_reason=None)` | Resolve/create the token directory (§4.2) |
| `shard_path(generation, key)` | The validated-separator key join (above) |
| `write_shard(path, frame, *, tag)` | Atomic per-item write; a `None`/empty frame still writes |
| `already_done(generation, key)` | The resume predicate |
| `progress(label, i, n: int \| None, *, elapsed_s, note)` | The unbuffered per-item line; `n=None` renders the total as `?` |
| `assert_conservation(generation, keys, failed)` | The `keys − failed` check |
| `reconcile(generation, dest, *, tag)` | Glob the generation, combine, atomic-write to `dest/` |

**`preflight` — validate the output destination before paying for the input.** Two instances make
this a primitive rather than a per-driver habit:

1. §4.3 already carries one, inherited from `calibrate_xt_bandwidth.py:225-239`: a parquet-engine
   check run *"before the multi-minute load, because pandas only surfaces 'Unable to find a usable
   engine' at write time, after the work is done."*
2. The TF-24 sweep spends ~15 minutes loading a corpus and then dies creating its Optuna store. The
   corpus work is entirely discarded on a cheap step that could have been checked first.

The second is the case `for_each` does **not** cover, and that matters for scoping. `for_each` protects
a per-item walk by persisting each item; `calibrate_tracking_defaults` loads a corpus *into memory* and
then sweeps, so there is no per-item artifact to shard and a post-load failure loses everything anyway.
Sharding is the wrong tool; refusing early is the right one.

`prune_stale_generations` is exported alongside these but is **not** one of them: `for_each` never
calls it. A generation directory is the only evidence that a pass at a given set of declared
inputs ever ran, so pruning on the way past would make an accidental token change both
unrecoverable and unnoticeable — the opposite of the visibility the directory form buys. It is an
explicit operator action, surfaced per driver as `--prune-stale`.

`preflight` is the sibling of the clean-tree check in §4.5 — both are "refuse before spending", and
both belong in `main()` before any corpus work. A driver passes the checks it needs (writable output
dir, resolvable store URL, importable parquet engine); the primitive exists so the *habit* is shared
even though the checks differ. It deliberately does not try to enumerate every possible check.

**An exception-path driver MUST call `assert_conservation`.** Without that requirement the primitives
path escapes both gates — §5's static check accepts any registered primitive and says nothing about
depth, and the runtime invariant would only fire for `for_each` adopters. Since §6 step 4 deliberately
front-loads the hardest multi-loop driver, the first non-trivial adopter would otherwise be the one
both gates are blindest to. §7 states the residual exclusion.

**An empty result STILL writes a shard.** `None` from `work` means *no rows*, never *no shard*. This
is an existing, thrice-stated, test-pinned invariant of this codebase and `for_each` must preserve
it: `build_layer2_spells.py:131-132` ("Written even when EMPTY: an absent shard means 'not yet run',
a present empty one means 'run, produced no spell'. Conflating them would make a resume silently
recompute"), the same rule in `validate_xshot_causal.py:230-232`, and
`test_validate_xshot_causal_shards.py::test_an_EMPTY_shard_is_distinct_from_an_ABSENT_one` (`:63-69`),
which writes a zero-row shard and asserts both halves. Absent means not-yet-run; present-and-empty
means ran-and-produced-nothing. The motivating driver has barren items — `validate_xs_probe:133`
computes `n_contributing` precisely because some matches contribute zero — so conflating the two
would recompute every barren match on every resume, forever, inside the 14-hour pass this cycle
exists to fix.

**A conservation invariant, checked at runtime: `this pass's keys with shards == keys − failed`.**
Because a completed item ALWAYS writes a shard, a failed item is the only thing that can be missing,
so the manifest needs no third category.

It counts **the pass's own keys, never a directory-wide glob.** N workers share one `--out` and
therefore one generation directory — the token derives from `token_inputs`, identical across
workers, while `tag` names the manifest file rather than the directory. A glob would compare one
worker's slice against every worker's shards: firing non-deterministically, *after* the expensive
loop, and *before* the manifest is written — so the partition would vanish from
`aggregate_manifests`, which is the exact "64-match artifact reported `n_matches: 8`" defect
`_partition.py:8-11` was extracted to prevent. It would also be unrecoverable, since a resume skips
everything and reaches the same comparison. Race-free because `providers_for_slice` guarantees
disjoint slices.

`reconcile` deliberately does the opposite and globs the whole generation: every worker rebuilds the
combined table from all shards, which is today's behaviour. Conservation is per-pass, reconciliation
is corpus-wide; the asymmetry is load-bearing and is pinned by a test on each side.

The invariant must be **exactly right before it ships**: one that fires on healthy runs gets weakened
or deleted by the first person it inconveniences, so shipping it wrong is worse than shipping it
late. An earlier draft of this section stated it too loosely — tolerating a missing shard for a
legitimately empty result — which would have licensed exactly the implementation the three citations
above exist to prevent. Recorded because the error is instructive: the weakening looked like
conservatism and was the opposite.

### 4.2 The token contract

**`token_inputs` is a required `Mapping[str, object]`, and the digest is `ruthless.fingerprint`.**
Not a hand-rolled hash. `ruthless-efficiency` 0.4.0 promoted `fingerprint`/`fingerprint_model` to its
public API on a trigger it had written down — *"a second real caller"* — and `_driver.py` is that
caller (§10). The mapping shape is theirs; naming each input is also strictly better here, because a
key documents *what* a value is and §4.2's whole rule is about declaring the right things.

**An earlier draft of this spec hand-rolled the token, and it was defective in two measured ways.**
It built `sorted(f"{type(v).__name__}:{v!r}")` over a list and hashed that:

| input | hand-rolled `_token` | `ruthless.fingerprint` |
|---|---|---|
| an object with no `__repr__` | `<Cfg object at 0x21933F3BD90>` → **a new token every process** | `TypeError`, refuses |
| a numpy scalar | `repr` changed in numpy 2.0 → **new token on a dependency major** | structural, unaffected |
| an unsupported type | silently digests it | fail-closed |

Both failures are silent and expensive in the same direction: an unstable token means the driver never
matches its own generation directory, so every run is a full recompute and nothing reports it. That is
the exact failure class this cycle exists to eliminate, and it was sitting in the cycle's own
primitive. Delegating removes it rather than patching it.

*(One thing the earlier draft got wrong in the other direction, recorded so it is not re-derived: the
`f"{type}:{repr}"` form does **not** collide the way ruthless's old `f"{a}:{b}"` did, because JSON
preserves list-element boundaries. The defects are instability and the absence of fail-closed, not
separator collision.)*

**Path-valued inputs are normalised to `PurePosixPath` at the seam.** ruthless guarantees that the
same *logical* value digests identically on every platform; **constructing** that value is the
caller's responsibility, and `Path(str)` parses per-platform — a backslash is a separator on Windows
and an ordinary character on POSIX. Our topology is a Windows dev box, a Linux DGX and both OSes in
CI, so an un-normalised path input would orphan a generation on the other platform with no version
having changed. This was a preference when proposed; ruthless's 0.4.0 contract makes it load-bearing.

`token_inputs={}` is legal and means "this pass has no staleness risk", but then `token_reason` is
mandatory. The purpose of the mandatory reason is to make "I did not think about it" and "I decided
it does not apply" distinguishable in the source. A silent omission is the failure mode that
produced this cycle.

`train_ghost_gk.cache_token()`'s geometry constants become the reference declaration in the module
docstring.

#### 4.2.1 A digest is a compatibility surface — the worked example is real

ruthless 0.4.0 also states digest stability as a contract: any change moving an already-supported
payload's digest is breaking, with no carve-out. Applying that rule retroactively to its own history
found a live instance, and it is worth recording here because it is the concrete form of what this
token protects against. Verified against both published wheels:

```
0.3.0    IntEnum ONE 0c929d97a64c891f == bare int 1 0c929d97a64c891f
         strEnum A   f42c6de6de351b46 == bare str a f42c6de6de351b46
0.3.1    IntEnum ONE af44c96ed8ba8e57  (MOVED)      bare int 1 unchanged
         strEnum A   d1d82590d9af0e1f  (MOVED)      bare str a unchanged
```

0.3.1's release note said it invalidated nothing. It invalidated exactly the payloads carrying an
`IntEnum` or str-`Enum`, leaving everything adjacent byte-identical — so a consumer would have seen
*some* shards orphaned and the rest reused, which is harder to notice than a total miss.

**silly-kicks' exposure is zero**, by timing rather than by care: nothing here called ruthless's
fingerprint, the lock moved 0.2.1 → 0.3.1 → 0.4.0 without ever installing 0.3.0, and `_driver.py` does
not exist yet. Recorded so a future reader does not go looking for orphaned shards that never existed.

**The dependency consequence:** ruthless now pins `pydantic<3` (its `fingerprint_model` digests
`model_dump()` output, putting pydantic's serialisation inside the digest's blast radius). silly-kicks
has no direct pydantic dependency, so this constrains our resolution transitively and costs nothing —
the lock already sits at 2.13.4.

**The rule for choosing inputs:** declare what determines the shard's **content**, not what
**consumes** it. Anything recomputed downstream of the shard on every run is excluded by
construction. `validate_xshot_causal` is the worked example and the trap: its shards are produced by
`causal/opportunities.py` + `shot_arm_config` (`:238`) and the model metadata, so those are the
token inputs — while `causal/matching.py` (`:127`) runs inside the analysis that re-reads the shards
every invocation and must **not** be declared. Since §7 concedes that completeness cannot be checked,
a mis-declared input is the one failure the gate structurally cannot catch, which makes getting this
rule written down worth more than the example alone.

### 4.3 Shape B — cohort cache

The four loop-free drivers get `--cohort-cache PATH` from the same module, reusing `_cache.py`'s
fail-closed metadata. **Absent the flag, behaviour is byte-identical to today.**

`calibrate_xt_bandwidth`'s `--corpus-cache` (mechanism 4, `scripts/calibrate_xt_bandwidth.py:225-239`)
is the reference implementation and the cautionary one. Worth keeping from it: the fail-fast
parquet-engine check *before* the multi-minute load, because `pandas` only surfaces "Unable to find
a usable engine" at write time, after the work is done. Worth fixing: it validates nothing beyond
`Path.exists()`, so a cache built from a superseded corpus is reused in silence. The shared version
carries `_cache.py`'s fail-closed metadata, and `calibrate_xt_bandwidth` migrates onto it — which
means the flag gains a validity check it did not have.

The cache is opt-in and explicitly named rather than automatic, and this is a correctness decision,
not a convenience one. Marts in this repo re-materialize constantly — `fct_action_context` is owed a
re-materialize for 4.52.0 and 4.53.0 as of this writing — and a query result has no token the
helper can compute without running the query. An automatic cache inside `load_xtgk_cohort` would
therefore serve a pre-re-materialize cohort silently: a plausible number from a computation that did
not happen, which is the ADR-036/PR-S113 fabricated-origin failure class. A path the operator names
cannot be reused by accident.

### 4.4 Databricks auth precedence

`scripts/_loader_databricks.py:47-53` takes the PAT branch on **any** non-empty `DATABRICKS_TOKEN`.
The OAuth U2M fallback below it (`:54-66`) is correct and complete; it is simply unreachable when a
non-empty token is present in the environment.

**Measured on this machine, 2026-07-29** (`echo "len=${#DATABRICKS_TOKEN} prefix=${DATABRICKS_TOKEN:0:4}"`):

```
len=36 prefix=dapi
```

with `DATABRICKS_CONFIG_PROFILE` unset. A 36-character `dapi`-prefixed value is Databricks
**PAT format**; a minted OAuth bearer is a JWT. So all four Shape-B drivers currently fail auth, and
the error names the workspace rather than the stale variable.

**Cross-repo note.** The lakehouse reports that its PR #491 (2026-07-22) removed exactly this dead PAT
from `DATABRICKS_TOKEN` and from `~/.databrickscfg [DEFAULT]` on this machine. The measurement above
says a PAT-format value is still present in this session's environment, so that cleanup did not reach
here — a lakehouse-side follow-up, raised rather than resolved by this cycle. **The design is
unaffected either way**: the fix is correct for a stale PAT and for an expired short-lived bearer
alike, which is exactly why change 2 names both.

The existing comment shows the empty-string case was considered — *"An empty token string is NOT a
usable PAT -> OAuth branch"* — but not the invalid one.

Two changes:

1. `DATABRICKS_AUTH=oauth|pat` selects the branch explicitly. **Unset preserves today's precedence**,
   so CI and legacy setups are untouched.
2. An auth failure on the PAT branch is re-raised naming the precedence **and both of its causes** —
   a stale PAT *or* an expired short-lived bearer. The lakehouse side deliberately puts a minted
   OAuth bearer with a ~299 s lifetime into `DATABRICKS_TOKEN`
   (`scripts/mint_databricks_oauth.py`), so the same branch fails for two different reasons and a
   message naming only "stale PAT" would mis-diagnose the common case.

This is a precedence and diagnosis fix. It does not change which credentials are valid.

**This file has a four-test pinning suite the fix must extend.**
`tests/scripts/test_loader_databricks_connect.py` pins `_connect()` with
`test_pat_path_uses_access_token` (`:61`), `test_oauth_path_when_no_token` (`:72`),
`test_oauth_profile_is_overridable` (`:84`) and `test_empty_token_falls_through_to_oauth` (`:92`).
**None clears `DATABRICKS_AUTH`.** Introducing the variable without hardening them produces a
concrete, predictable defect: a developer who exports `DATABRICKS_AUTH=oauth` — exactly the person
this feature is for — sees `test_pat_path_uses_access_token` fail locally while CI stays green, and
an environment-dependent test is how a suite loses its authority. The §4.4 commit therefore adds
`monkeypatch.delenv("DATABRICKS_AUTH", raising=False)` to all four, plus two cases:
`DATABRICKS_AUTH=oauth` with a token set selects OAuth, and `DATABRICKS_AUTH=pat` with no token
raises loudly rather than silently falling through.

### 4.5 EIGHT artifact-writing drivers have no provenance wiring; one stamps a bare SHA into the wheel

Found while checking an unrelated question about the calibration report, then measured properly rather
than reported at the size it was stumbled on. 2026-07-30:

| | count | drivers |
|---|---:|---|
| detected as artifact-writers | 15 | names an output flag **and** persists |
| registered + fully wired | 7 | `build_gkdv_arm_values`, `build_layer2_spells`, `derive_opengoal_range`, `measure_cover_shadow_argmax_agreement`, `run_signoff_power`, `validate_xs_probe`, `validate_xshot_causal` |
| **no wiring at all** | **8** | `calibrate_tracking_defaults`, `calibrate_xt_bandwidth`, `train_ghost_gk`, `train_xcross_attempt`, `train_xshot_occurrence`, `validate_shot_goalmouth_sb`, `validate_xcross_causal`, `validate_xtgk_possession_value` |

None of the eight imports `scripts/_provenance.py`, offers `--allow-dirty`, or appears in
`ARTIFACT_DRIVERS`. All eight are in this cycle's driver population.

**The severe one is `train_ghost_gk`, and it reaches the published wheel.** `scripts/train_ghost_gk.py:175`:

```python
training_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()  # noqa: S607
except Exception:
    training_commit = None
```

A **bare `git rev-parse HEAD`** — which returns the same SHA from a dirty tree — inside a broad
`except Exception`, stamped at `:728` into the model metadata. Verified in the shipped artifact:

```
_ghost_gk_weights/default/metadata.json   training_commit=97c74d58…   tree_state=None
_xcross_weights/default/metadata.json     training_commit=absent      tree_state=None
_xshot_weights/default/metadata.json      training_commit=absent      tree_state=None
```

So the one bundled artifact that records a commit records it with **no tree state** — verifiable-looking
provenance on weights that ship to PyPI, which CLAUDE.md already names as "strictly worse than
recording nothing". This touches ADR-011's trained-model lifecycle, not just `scripts/`.

**`tests/scripts/test_provenance_wiring.py` has a dedicated AST test for exactly this call**
(`test_driver_never_shells_out_to_rev_parse_directly`, matching on CALLS so prose describing the defect
is not mistaken for committing it). It never runs against `train_ghost_gk` because `ARTIFACT_DRIVERS`
is a **hand-maintained tuple** whose meta-test only asserts `len(...) >= 6` and that each named file
exists. It does not derive its population, so a driver that writes artifacts stays off the list
silently — the same hand-maintained-inventory weakness §5's derived population fixes for resilience,
sitting in the neighbouring gate. The rule was moved out of memory into a gate; the gate kept a list
that goes stale the way memory did.

**Method note, recorded because it is the third instance in this cycle.** The first pass at this table
tested `"_provenance" in src` and reported two drivers partially wired. Both were false — the substring
matched the *variable* `corpus_provenance` (`train_ghost_gk.py:676`) and the *function*
`reward_provenance_summary` (`validate_xtgk_possession_value.py:67`). A substring probe over-reported
adoption, exactly as `"_cache" in src` did in §2.1 and in the ruthless handoff. The gate's own helper
docstring says why it is AST-based: *"a plain substring scan flagged this module's own explanatory
docstring … they cannot tell a described defect from a committed one."*

**SCOPE — decided.** Three separable pieces:

1. **Wire the two `calibrate_*` drivers.** §6 steps 4 and 7 already open them. **IN.** The plan carries
   `calibrate_xt_bandwidth` at Task 13 Step 3.5 and `calibrate_tracking_defaults` in Task 14's
   template — both, symmetrically. An earlier revision wired only the first because a follow-up edit
   silently failed to apply; the spec claimed both while the plan did one, which is exactly the kind
   of drift a twice-reviewed pair is supposed to be free of.
2. **Wire the remaining six**, including the three trainers. **IN** (owner decision, 2026-07-30).
   `train_gk_completion` is this cycle's even though another session has it parked on a branch — that
   branch is inert and its author offered to re-do against whatever lands here.
3. **Derive `ARTIFACT_DRIVERS`' population** so this cannot recur. **OUT.** A change to a different
   gate, needing "registered artifact" defined before it can be derived. Recorded as the reason the
   eight survived, not absorbed here.

**The trainer half carries an ADR-011 question this cycle does not answer.** Wiring
`train_ghost_gk`/`train_xshot_occurrence`/`train_xcross_attempt` fixes provenance *going forward*. It
does not decide what to do about `_ghost_gk_weights/default/metadata.json`, which already ships a
`training_commit` with no tree state. Re-stamping published weights is an artifact-lifecycle decision,
not a `scripts/` one, and it is explicitly deferred with the evidence recorded above.

**Sequencing note for the other session.** ADR-051's deferred PR-5 chirality cycle will retrain and
re-stamp xS and xCross — two of these three trainers. If that lands before the wiring, it produces a
fresh set of published weights carrying the same unreliable provenance. Cheaper to wire first.

---

## 5. The CI gate

`tests/scripts/test_corpus_driver_resilience.py`.

**Population** is derived by AST (§2, corrected detector), so a new corpus driver is enrolled the
moment it is written. There is no hand-maintained inventory to rot.

**Verdict is adoption**: does the driver import and call `for_each`, or a registered primitive from
`_driver`?

The first draft of this gate checked capability **tokens** — `"shard" in src`, `".is_file()" in src`,
`"flush=True" in src`. Run against the population it certified exactly five drivers:

```
test_corpus_driver_is_resilient[build_layer2_spells]     PASSED
test_corpus_driver_is_resilient[train_ghost_gk]          PASSED
test_corpus_driver_is_resilient[train_xcross_attempt]    PASSED
test_corpus_driver_is_resilient[train_xshot_occurrence]  PASSED
test_corpus_driver_is_resilient[validate_xshot_causal]   PASSED
9 passed, 16 skipped
```

(The other four of the nine are the gate's own non-parametrized tests; five is the whole certified
driver population.)

**Three of those five are accumulate-then-write** (the trainers), so 60% of its certifications are
false. It is also wrong in the other direction: `build_gkdv_arm_values` is genuinely resumable —
shard write plus existence guard — and is pinned as debt solely because it lacks `flush=True`.

A gate that is wrong in both directions is not a weak gate, it is a misleading one, and it is why
the token version is **deleted** rather than kept alongside the adoption check.

**Exemptions** live in a registry with a mandatory reason string, asserted **exactly both ways**: a
newly unhardened driver cannot join silently, and a driver that has been fixed must be removed. This
is the property the CLAUDE.md prose rule lacked.

**Non-vacuity**: a planted naive corpus driver must be classified in-population and unhardened, and
a trivial script must not be enrolled. Without both halves a green gate is indistinguishable from
one that detects nothing.

---

## 6. Rollout

One feature branch, one PR, a commit per script (squashed on merge), one version bump across the
five sites. Order builds against a known-good reference before applying to anything new:

1. **`_driver.py` + the rewritten gate.** The gate is run once with an EMPTY pending list to DERIVE
   the unmigrated population (red), then that list is recorded — so the COMMITTED state is
   green-by-skip, because CI must stay green on every intermediate commit and
   `test_the_pending_list_is_EXACT` supplies the pressure instead. This step builds the
   **whole** module — `for_each`, the enumerated primitives, *and* the §4.3 cohort cache (absorbing
   the fail-fast parquet-engine pre-check that currently lives in
   `calibrate_xt_bandwidth.py:225-239`). The cohort cache must exist here, not at step 8 with its
   four Shape-B consumers, because step 4 migrates `calibrate_xt_bandwidth`'s `--corpus-cache` onto
   it — a helper shipping as a side effect of a Shape-A driver's commit would invert this ordering's
   whole premise.
2. **Build the oracle that does not exist yet.** A red-first double-invocation test per resumable
   driver: call `main()` twice, assert the second run never enters `work`, and assert the output
   table is byte-identical. **Include the empty-shard round trip** — an item yielding zero rows must
   write a shard and must be skipped on re-run. That property is currently pinned for only one of
   the three drivers (`test_validate_xshot_causal_shards.py:63-69`) and is the one most likely to be
   lost in a migration, since losing it is invisible except as a slow resume. **The existing tests
   cannot serve as the migration oracle** — none of them
   exercises the resume branch. `test_build_layer2_spells.py:78-82` runs `main()` once and asserts
   `shard.is_file()`, never re-running to reach `if shard.is_file(): continue`;
   `test_build_gkdv_arm_values.py`'s 15 tests never touch the shard loop (its shard-adjacent test
   exercises `_partition.aggregate_manifests`); and
   `test_validate_xshot_causal_shards.py::test_an_EMPTY_shard_is_distinct_from_an_ABSENT_one`
   tests the resume check's *precondition*, not the skip. So the safety net covers writes and
   aggregation and is blind to resume — which, combined with the shard-path change in §4.1, would let
   step 3 go green while silently converting three working resumable drivers into full-recompute
   drivers. This is the one place the spec had skipped its own gate-before-change ordering.
3. **Migrate the 3 already-resumable drivers** — `build_gkdv_arm_values`, `build_layer2_spells`,
   `validate_xshot_causal` — against the step-2 oracle. If `for_each` cannot reproduce them, the
   design is wrong and it surfaces on the cheap ones.
4. **`calibrate_xt_bandwidth` — the multi-loop proof, deliberately EARLY.** At 4 corpus loops it is
   the single largest "`for_each` cannot express this" risk in the population, so it is the first
   exercise of fork C's primitives path rather than the last, and the first driver required to call
   `assert_conservation` itself (§4.1). Its `--corpus-cache` migrates onto the step-1 cohort cache in
   the same commit, since that cache is the Shape-B mechanism even though its loops are Shape A.
   **If the primitives path is inadequate, that is a design finding about fork C**, and it must
   surface here — while `for_each` has only three adopters to revise — not at step 7 of 9. Same
   reasoning that put the resumable drivers at step 3: meet the hard case early, when the cost of
   being wrong is still small.
5. **The 3 trainers** — generalize `_cache.py`'s fingerprint, fold `cache_token()`'s geometry
   constants in as the reference `token_inputs`.
6. **`validate_xs_probe`** — the motivating case, now a greenfield application of a proven helper
   rather than the prototype.
7. **Remaining Shape-A drivers**, including `calibrate_tracking_defaults` (3 loops — the second
   multi-loop driver, cheap once step 4 has settled the primitives path).
8. **Shape B (4) + the auth precedence fix.**
9. **ADR, CLAUDE.md, docs.**

The spec itself is **not committed standalone** — it lands with the cycle's PR.

**Conflict avoidance.** `measure_cover_shadow_argmax_agreement` is the other session's 4.67.0 file
and goes last, after a rebase check. The in-flight ADR-028 cycle is concentrated in
`silly_kicks/tracking/` and its tests; this cycle is `scripts/` plus one test file, so the surfaces
barely intersect — but the check is performed before each commit rather than assumed.

**`_loader_databricks.py` has pending hand-back work.** The lakehouse reports three open defects in
that file's `_convert` (the TF-24 calibration loader), handed back 2026-07-18. This cycle touches
`_connect` in the same file — a different function, so not a textual conflict, but a file with
inbound work from another party is exactly where the pre-commit rebase check earns its keep, and
whoever lands the `_convert` fix should know the auth change is coming. Noted as a **relayed** report:
nothing in the file records it (no `TODO`/`FIXME`/hand-back marker), so it could not be verified
locally.

**Two fences lifted explicitly** (Chesterton's Fence — stated here rather than silently stepped over):

- `_partition.py:13` and `:56` record that "`scripts/_loader_*` is READ-ONLY from here … which this
  cycle may not modify". That constraint is scoped to the **TF-19 partition cycle** that wrote it,
  where the concern was a partition naming a match the loader would not fetch. §4.4 modifies
  `_loader_databricks.py` for an unrelated auth-precedence reason, so the fence does not apply — and
  `_partition.py`'s own wording is corrected in this cycle to say which cycle it bound.
- The draft gate's pinned reasons for `calibrate_tracking_defaults` and `calibrate_xt_bandwidth` read
  "ISOLATION ZONE for this cycle -- not modifiable here". Those strings **predate** the scope
  decision that put all drivers in scope, and are corrected rather than honoured. The isolation
  convention is conflict-avoidance, not ownership; the live question for these two is only whether
  another session is editing them concurrently, which the pre-commit rebase check answers.

**Both `calibrate_*` drivers are IN scope — decided, not defaulted** (owner, 2026-07-29, on review of
the alternative). A cross-session review recommended deferring them to remove the 4-loop driver from
the critical path. That argument was about engineering risk rather than ownership and is sound on its
own terms; the response is to keep them and **front-load** the risk at step 4 instead, so a fork-C
inadequacy surfaces at three adopters rather than at fourteen. Recorded here because a later reader
finding a 4-loop driver mid-rollout should meet the reasoning, not re-derive the debate.

---

## 7. What this does not do

**The token cannot be checked for completeness.** An author can declare the wrong inputs and get a
token that never changes. The gate can require that a declaration exists; nothing cheap can verify
that it is sufficient. Requiring a reason for the empty declaration narrows the gap to
*mis-declaration* and closes *silent omission*, which is the failure mode observed.

**The adoption gate cannot check correctness *at source level*.** A driver can call `for_each` for
something trivial and still accumulate in a second loop. No lexical or AST rule catches that. It is,
however, checkable at **runtime** by the conservation invariant in §4.1 — `shards == attempted −
failed` — which a second uncounted loop breaks. So the *static* gate proves adoption only, and the
*runtime* invariant catches the specific evasion that matters.

**Correction: the runtime invariant does NOT catch that evasion.** An earlier draft of this spec
claimed it did. It does not. A driver that calls `for_each` over something trivial and separately
accumulates over the real corpus writes **no shards** for the second loop and lists **none** of its
items among the pass's keys — so `assert_conservation` never sees it and passes. The claim entered
via a review finding, was amplified here, and is retracted rather than quietly softened.

What `assert_conservation` genuinely proves is narrower and still worth having: **every item a pass
attempted either wrote a shard or is counted as failed.** That catches a completed item that
silently skipped its write, off-by-one counting, and stale-generation contamination.

So the honest ceiling: the static gate proves adoption; the runtime invariant proves per-pass
conservation; **neither proves the absence of a second loop.** Covering that needs a fan-in check —
the union of every manifest's declared key set against the generation's contents — which is a
different property from the per-pass one and is a recorded follow-up, not something this cycle
ships. A second tier applies on top: exception-path drivers get even the per-pass guarantee only
because §4.1 *requires* them to call `assert_conservation`, and that is a contract, not a mechanism.

**No `silly_kicks/` behaviour changes.** No model, no feature column, no VAEP retrain trigger. The
wheel packages `silly_kicks` only (`pyproject.toml:131`), so a `scripts/`-only cycle is
wheel-identical — but `scripts` **is** inside the pyright include set (`pyproject.toml:223`), so the
new module and every migrated driver must be clean under CI's full-tree, config-driven `pyright`,
not a scoped run.

**Not in scope:** the cp1252 `--help` crash, tracked separately. Its driver set overlaps this one to
an unmeasured degree; the two were each counted as "16" at different times and that coincidence is
not evidence of a relationship.

---

## 8. Verification

Tests passing is not the acceptance criterion for a resilience cycle. The acceptance criterion is a
**kill-and-resume** on a real corpus pass: interrupt it mid-run, restart it, and confirm it resumes
at the correct item, recomputes nothing already sharded, and reconciles a manifest whose totals
describe the whole corpus rather than one partition.

The `build_gkdv_arm_values` shards already exist, so this is cheap on a small slice — **but reuse is
no longer automatic** after the generation-directory change in §4.1, and the prerequisite has an
ordering wrinkle worth stating. The existing shards sit flat in `shard_dir`; the migrated driver
looks in `shard_dir/<token>/`. The operator must therefore **run once to learn the generated token
name** (it is derived from the declared inputs, not chosen), then `mkdir` that directory and move the
existing shards into it. Filenames are unchanged, so the move is purely a path prefix. Skipping this
does not corrupt anything — it just recomputes the corpus, which is the expensive way to discover the
step was missed.

A second, cheaper check: flip a declared token input and confirm the next run recomputes rather than
reusing shards — the `cache_token` scenario, now exercised rather than reasoned about.

---

## 9. ADR

An ADR at the next free number records the driver contract and supersedes the CLAUDE.md prose rule,
which has now failed twice (`validate_xshot_causal.py` wrote an artifact with no provenance;
`validate_xs_probe.py` stamped a bare `git rev-parse HEAD`). It must record both ceilings from §7
explicitly — an ADR that claims the guard is complete is worse than no ADR.

---

## 10. Overlap with `ruthless-efficiency` (owner-directed check)

`ruthless-efficiency` @ `a9566ee` (0.2.1) was surveyed for functionality this cycle would duplicate.
It is already a silly-kicks dependency — `ruthless-efficiency[optuna]>=0.2.1` under `[calibration]`
and `[train]` (`pyproject.toml:70, 105, 116`), imported by four drivers.

**Verdict: no functional duplication as designed.** It is a *general optimisation/search substrate*
(`Candidate` / `Objective` / `Strategy` / `ComputeBackend`); a corpus pass is a scan, not a search.
What it has, and why each does not serve:

| Surface | What it is | Why not reusable here |
|---|---|---|
| `strategies/evolve_/strategy.py:130-173` — `_eval_fingerprint` + `_load_cached_seeds` | Per-program result JSON carrying a fingerprint; mismatched fingerprint ⇒ skip | **Structurally the same idea.** But private, coupled to `EvolveConfig`, behind the `[evolve]` extra, per-*program* not per-corpus-item, hashes only `epochs:seed` (its docstring: "no dataset"), and has no shards, manifest, atomic write, progress or failure accounting |
| `parallel.py` — `map_work_units` | 42-line intra-objective parallel map | No persistence, no resume. Would be the right primitive *if* `for_each` ever grows in-process fan-out; today parallelism is N OS processes over `--match-ids-json` |
| `backends/remote_ssh.py` | scp config + ssh + parse one JSON metrics line | Genuinely DGX-adjacent, but bound to `ComputeBackend.evaluate(candidate, objective)`. A corpus pass is not a candidate evaluation |
| `config/common.py` — `StoreConfig(kind="sqlite")` | "single-process resume" | Optuna RDB storage for HPO trials |

**The observation worth recording.** With ruthless's fingerprint cache included, "cache a per-item
result and key it on a fingerprint" is now implemented **five** times across the two repos (§3's four
plus `evolve_`). That is the strongest possible argument that it belongs in one published place, and
`ruthless-efficiency` is the published, reusable package.

**`for_each` itself must never move there** — and that is an architectural fact, not a scheduling
one. `ruthless/CLAUDE.md` states two principles that exclude it: *"Each strategy owns its loop. The
core imposes no template-method driver"* and *"Persistence/resume is strategy-internal."* `for_each`
is precisely a template-method driver that owns persistence. Proposing it upstream would be asking
ruthless to reverse a deliberate design decision in order to host a batch harness that is outside its
optimisation/search charter. It stays here.

**The fingerprint primitive moved the other way, and this cycle consumes it.** An earlier draft of
this section argued for deferring adoption — build against 21 adopters first, promote later. That was
wrong, and §4.2's table is why: the hand-rolled token it was defending produced a different digest
every process for any input without a stable `repr`, and a different digest across a numpy major for
an unchanged declared value. Deferring would have shipped that.

What actually happened is cleaner than either plan. ruthless's own 0.3.0 spec had written down the
promotion trigger — *"Promotion is a later, additive decision that should wait for a second real
caller"* — and `scripts/_driver.py` is that second caller (`evolve_` was the first). A change request
naming the trigger produced **ruthless 0.4.0**: `fingerprint` and `fingerprint_model` public, a
digest-stability contract with no carve-out, a golden table of 44 pinned literals verified against the
published wheel, and a Windows CI leg because the digest's consumers span two platforms while its
producer's CI spanned one. None of that existed when this spec was first written.

The lesson worth keeping is about the shape of the reasoning, not the outcome: *"don't depend on it
yet"* and *"don't reach into another package's private module"* were both right, and the resolution to
each was to ask for the promotion the owner had already pre-authorised — not to build a fifth
implementation locally, and not to import from `_`-space.

**We deliberately decline `ruthless._provenance`, and that is not inconsistency.** 0.4.0 also ships
`code_identity()`, and we keep `scripts/_provenance.py` instead: **ours refuses** a dirty tree,
**theirs records** one. For an artifact driver the refusal is the entire point (§4.5), so the two
answer different questions and both should exist. Adopting a shared primitive where it dominates and
declining one where the contract differs is the same judgement applied twice.

---

## 11. Risks

| Risk | Mitigation |
|---|---|
| `for_each` cannot express a multi-loop driver | Fork C anticipates this (primitives stay exposed; the exception needs a recorded reason), and §6 step 4 **front-loads the measured worst case** — `calibrate_xt_bandwidth`, 4 loops — as the first primitives-path adopter, so an inadequacy is a design finding at three adopters rather than at fourteen. |
| Migrating a working driver breaks it | Order puts the 3 test-covered resumable drivers first, as behaviour-preserving migrations with an oracle. |
| Stale shard generations accumulate on disk | Kept visible by design, one directory per token; `prune_stale_generations` / `--prune-stale` removes a generation on explicit operator action, never automatically (a generation directory is the only record that a pass at those inputs ran). **Built in plan Task 1b** — this row asserted the mitigation twice before anything implemented it. Disk is the cheap failure — the expensive one (a double-counted table) is structurally excluded by the directory form, §4.1. |
| Per-item failure tolerance masks a systematic bug | Failures are counted in the manifest and `max_consecutive_failures` aborts. |
| A migration silently disables resume | Step 2 adds the double-invocation oracle **before** any driver moves; the existing suite is blind to this (§6). |
| Conflict with the in-flight ADR-028 cycle | Disjoint surfaces; rebase check before each checkpoint; the other session's file (`measure_cover_shadow_argmax_agreement`) goes last. **Found while checking this: that driver builds `passer_xy` from raw action-LTR `start_x`/`start_y` and passes it beside frame-LTR positions — the ADR-051 RC1 defect, still live, because it calls `_compute_cover_shadow_dict` directly and RC1 fixed only the `features.py` callers. Its recorded numbers are pre-RC1. Handed to the ADR-051 session; this cycle migrates it for resilience and changes no geometry.** |
