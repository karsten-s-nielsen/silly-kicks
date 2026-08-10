# ADR-052 — The shared corpus-driver seam: resume, staleness, progress

**Status:** Accepted (4.72.0, PR-S140). Amended 4.73.0 (PR-S141) — `probe_old` row-alignment + the re-bundle gate's non-finite/shape fail-closed; see the Amendment at the end.
**Spec:** `docs/superpowers/specs/2026-07-29-corpus-driver-resilience-design.md`
**Plan:** `docs/superpowers/plans/2026-07-29-corpus-driver-resilience.md`

---

## Context

Twenty-one `scripts/` drivers walk a corpus and do minutes-to-hours of work per item. **Three
survived a crash. Fourteen held every result in memory and wrote once at the end**, so a failure in
the cheap step after the expensive loop lost the whole pass — measured: a power-analysis driver
spent **8.7 hours** walking 64 matches, raised in the analysis step that followed, and lost every
one of them, while the one driver that had shards survived its own 64-match run twice.

This was never a missing convention. **Four partial mechanisms already existed**, covering seven
drivers between them, and the split is exactly **resume XOR staleness**:

| Mechanism | Resume | Staleness token | Owns the loop |
|---|---|---|---|
| `scripts/_partition.py` | yes | no | no |
| `scripts/_cache.py` | no (all-or-nothing) | yes | no |
| `train_ghost_gk`'s own `_feature_cache/` + `cache_token.txt` | no | yes | no |
| `calibrate_xt_bandwidth`'s `--corpus-cache` | no | no | no |

**None of the four owns the loop**, which is precisely why resume and progress kept being omitted:
there was nowhere for them to live except in each author's memory.

## Decision

`scripts/_driver.py` owns the loop. `for_each` is the default shape; the individual primitives are
the escape hatch for a driver whose loop genuinely cannot invert, and such a driver must still call
`assert_conservation` **and** `_require_injective`.

**Measured, and worth stating rather than implying: the primitives path has ZERO adopters.** All
21 drivers use `for_each`, so both escape-hatch gates skip on every one of them and
`_require_injective` has no caller anywhere in `scripts/`. The rollout expected the hardest
multi-loop driver to land there; it did not (D4 — `calibrate_tracking_defaults`'s unshardable loop
needed no shards at all). The rule is real and the gates work, but nothing exercises them today.
Adoption is CI-gated by
`tests/scripts/test_corpus_driver_resilience.py`, whose population is derived structurally and whose
verdict is adoption-based rather than lexical (see "What we got wrong" below).

### D1 — The staleness token is a generation DIRECTORY, never a filename suffix

`shard_root/<token>/<key>.parquet`. Reconciliation is driver-local `glob("*.parquet")` at every
existing site, so a suffix would make the combined table concatenate the old-token and new-token
shard for the SAME item, with different values, the first time a declared input changed — and
`n_shards`, a provenance field, would overcount by the same factor. The directory form makes that
unrepresentable rather than merely guarded, and stale generations stay visible on disk
(`prune_stale_generations` is explicit operator action, never automatic).

**The token cannot be checked for completeness, and this is not a theoretical ceiling — it bit
FOUR times in this cycle's own drivers.** An author who declares the wrong inputs gets a digest that
never changes when it should. Requiring `token_reason` for an empty declaration closes silent
OMISSION, not MIS-declaration.

An adversarial audit of all sixteen declarations, run before this ADR was committed, found:

- **`--tracking-limit` undeclared in three drivers** (`build_gkdv_arm_values`, `build_layer2_spells`,
  `validate_xshot_causal`). The flag truncates frames, so a capped smoke run and a real run are
  different corpora sharing one generation: the operator smoke-tests, relaunches against the same
  `--out`, every match reports `skip (shard exists)`, and the artifact is rebuilt from truncated
  shards. In `build_gkdv_arm_values` the counters sidecar makes it worse — `n_frames_in` /
  `n_frames_scored` REPLAY from the smoke run, so `conservation_holds` comes out **true**, the guard
  corroborating a corpus that was never walked.
- **`train_ghost_gk`'s home-team mapping undeclared.** Structurally different: `home_team_id` drives
  the goal-relative flip of every feature and label, but it is a PER-ITEM input while the token is
  per-PASS. Declaring the map would invalidate every shard whenever a match joined `--data-dir` —
  the over-invalidation the selector rule exists to avoid — so it belongs in the shard KEY instead.
  The asymmetry it closes: ADDING a missing mapping was always safe (no shard existed), while
  CHANGING one was silent and total.

**Two were findable from inside the repo, and were not found by their authors.**
`run_signoff_power`'s inline twin declares `tracking_limit` under a comment stating it *"Mirrors
`build_layer2_spells`' declaration"* — it did not. `validate_xshot_causal`'s own docstring calls it a
*"thin clone"* of `validate_xcross_causal`, which declares exactly that input; the clone dropped it.
Two sources contradicting each other in committed text, unnoticed through authoring and review.

The transferable rule: **a declaration is the one part of this design with no gate, so it needs a
reader — and the reader must be adversarial, not the author.** Reviewing your own `token_inputs` is
reviewing the assumption that produced it. The digest itself is `ruthless.fingerprint`, not hand-rolled:
an earlier draft hashed `repr()`, which digested a MEMORY ADDRESS for an object with no CUSTOM `__repr__`
(a different token every process, so the driver never matched its own generation and silently
full-recomputed).

### D2 — An empty result STILL writes a shard

An absent shard means "not yet run"; a present empty one means "ran, produced nothing". Conflating
them makes every barren item recompute on every resume, forever — and the 14-hour driver this seam
exists for has barren items. This was already thrice-documented in the three resumable drivers and
test-pinned in only ONE of them (two after this cycle; `build_gkdv_arm_values` still has no
empty-shard test); a proposed "legitimately empty" third category was **wrong** and would have
licensed exactly that trap.

### D3 — Conservation counts THIS PASS'S KEYS, never a directory glob

N workers share one `--out` and therefore one generation directory, so a glob would compare one
worker's slice against every worker's shards: it fires non-deterministically, after the expensive
loop and before the manifest write, so the partition vanishes from `aggregate_manifests` — the
"64-match artifact reported `n_matches: 8`" defect `_partition.py` was extracted to prevent. It is
also unrecoverable, because a resume skips everything and reaches the same comparison.

**`assert_conservation` alone is satisfiable by a lossy run.** A non-injective `key` makes two items
share one shard, so `present` counts that file once per duplicate key and the relation BALANCES on a
run that dropped an item — a guard that certifies its own failure. `_require_injective` is the other
half and the gate requires both. Measured: 2 items → 1 processed → conservation returns `(2, 2)` and
PASSES.

**What conservation does NOT prove:** that the driver has no OTHER loop. A driver that calls
`for_each` over something trivial and separately accumulates over the real corpus writes no shards
for that second loop and lists none of its items. Catching that needs a fan-in check at reconcile
time; recorded as a follow-up. An earlier draft claimed conservation covered it — it does not, and
the claim is retracted here as well as in the spec and plan.

### D4 — `for_each` resumes WORK, never the PRODUCTION of its items

This is fork C's real limit, and it is not the one the design predicted ("multi-loop drivers").
`load_matches` downloads AND parses a match inside the generator before yielding it, so a streamed
driver whose `work` is cheap re-pays the whole corpus on a resume in order to skip trivial writes.

**The rule: invert onto an id list only where `work` is unambiguously trivial next to the load.**
Applied to `calibrate_xt_bandwidth` (which slices actions) and `_load_xt_corpus_pining` (which
slices columns); the drivers whose `work` is the expensive half keep the streaming shape, because
`items` must be STREAMED and never `list()`ed — materialising ~80 matches' tracking frames would
defeat resume and can OOM.

`train_ghost_gk` is the documented exception in the other direction: its item source is a nested
file → game walk where a game id is only knowable from inside the file, so enumerating up front
would read every parquet once per game. It streams, and re-reads on resume while skipping the
extraction — which is the expensive half.

### D5 — `reconcile` requires a PARTITION SURFACE; every other driver combines from `res.keys`

Corpus *selectors* (`--providers`, `--max-per-provider`) are deliberately NOT in `token_inputs`, so
that narrowing a corpus reuses shards rather than re-downloading them. That makes the generation
directory a SUPERSET of any one run, and `reconcile` returns the superset. Measured on
`calibrate_xt_bandwidth`: a `--providers skillcorner` run following a two-provider run returned
`['idsse:M2', 'skillcorner:m1']`, and its sweep would have run on a corpus nobody requested.
`reconcile` is correct only where `--match-ids-json` plus a worker tag make every run a slice of ONE
logical corpus. `CorpusPassResult.keys` exists because a STREAMED driver cannot walk its source
twice to rebuild them, and a second copy of the key rule at the read site silently finds nothing the
moment it drifts.

**One driver inverts this rule and it is worth naming.** `measure_cover_shadow_argmax_agreement`
fits its xT surface on the whole loaded corpus and feeds it to both scored paths, so there the
selectors DO determine content: its `token_inputs` declares the match ids, and a wider run correctly
misses.

### D6 — Corpus-scoped counters survive resume via a sidecar; pass-scoped counts deliberately do not

`counters` is called only for items a pass ATTEMPTS, so a fully resumed worker wrote
`{'n_frames_in': 0, 'n_matches': 0}` into its manifest — while `build_gkdv_arm_values` states
in-source that corpus totals must come from summing per-worker manifests. A partitioned run in which
ANY worker resumed produced a corpus artifact under-reporting the corpus, silently.

Closed with a `<key>.counters.json` sidecar (invisible to every `*.parquet` glob) replayed on skip.
Deliberately NOT by recomputing `counters(item, shard)`: the contract lets `counters` close over
`work`'s per-item metadata, and on a skipped item that closure holds the PREVIOUS item's report —
confident wrong numbers, strictly worse than the zeros they replaced. A missing or truncated sidecar
is **counted** (`n_counters_unrecorded`), never zeroed.

`n_attempted` EXCLUDES skips, and that is not the conservation quantity: `aggregate_manifests`
strips a manifest of its vote on commit consistency only when it positively declares it built
nothing, so a full-resume pass must be able to declare zero.

**`manifest_fields(counters_unrecorded=...)` has NO default**, and losing that default is a bug fix.
All three drivers migrated before the sidecar existed wrote the call by hand and silently took the
0, so a resumed worker whose sidecars were missing produced a manifest reading
`n_counters_unrecorded: 0` beside `n_matches: 0` — a corpus artifact reporting a corpus of nothing
and asserting the report was complete. Verified by planting the old call back (the guard goes red).
Drivers use `CorpusPassResult.manifest()`. A parameter whose wrong value is invisible must not have
a convenient default.

### D7 — The `work → tidy frame` contract carries no per-item METADATA and no NON-TABULAR side state

Two escapes, and they are different:

- **Per-item metadata the frame cannot carry** is resolved by the documented `counters(item, frame)`
  closure over `work` (`build_gkdv_arm_values`'s `build_ghost_frames` report), which the sidecar
  then makes resume-safe. `train_ghost_gk`'s SkillCorner selection-bias diagnostic rides that
  channel as `(sum, count)` pairs precisely because they are summable — accumulating them in a list
  would have made a PARTIALLY resumed pass report means describing only the matches it happened to
  redo, while looking exactly like a corpus figure.
- **Genuinely non-tabular side state cannot be sharded at all.** `train_xcross_attempt`'s TF-19
  probe cohort holds whole tracking frames and `_write_probe_sample` no-ops on empty, so a resumed
  pass would silently never write the gate cohort; `_extract` now returns `res.skipped` and the
  caller RAISES unless the sample already exists on disk.

`validate_shot_goalmouth_sb` is the boundary case: SEVEN heterogeneous per-match outputs (`rows`, `report`,
`unmatched`, `sweep`, `zcmp`, `debug`, and the `vocab` set added so a resumed run does not abort on
a spurious L-4 violation), five of them non-summable and one deeply nested. Its shard's tidy unit is the MATCH and its payload is a JSON
bundle — with an explicit encoder, because `default=str` renders `pd.NA` as `"<NA>"` and the report
asks `pd.isna` about exactly that field, so a shot with an unknown on-target verdict would come back
from a shard reported as a miss.

### D8 — Conservation is UNDEFINED for a cohort-cache-only adopter

FIVE drivers adopt only `cohort_cache` (`train_gk_retention`,
`validate_xtgk_possession_value`, `validate_xtgk_v2`, `xtgk_v2_kappa_sweep`,
`xtgk_v2_keeper_discrimination`). Four have no items, no keys and no shards, so the
only way to satisfy a conservation demand is a call on an empty key list — which asserts nothing and
teaches the next contributor that these are boilerplate. The gate exempts them by
`_SHARD_PRIMITIVES` (a derived property), never by name, guarded both ways, plus a meta-assertion that the exemption is
NON-EMPTY and that every driver it fires on calls `cohort_cache`.

**Correction, found by fact-checking this ADR against its own gate:** an earlier draft claimed the
meta-assertion also proves "every driver it fires on genuinely has no shard pass". It does not —
that half is a TAUTOLOGY. The exempt list is built with `not _runs_a_per_item_pass(t)` and the
assertion restates the same expression, so it cannot fail. And the exemption already covers one
driver it should not: `validate_xtgk_possession_value` has an accumulate-then-write cohort loop
(`:136`–`:167`) and the spec's own census classifies it `accumulate`. **So this ceiling is worse
than D3's wording admits, in a named instance.**

### D9 — The cohort cache is opt-in via an explicitly named path, never automatic

A query result has no token this module can compute without running the query, and the marts behind
these cohorts re-materialize regularly, so an automatic cache would silently serve a stale cohort —
a plausible number from a computation that did not happen. A path the operator names cannot be
reused by accident.

### D10 — A pre-generation (FLAT) shard set is REPORTED, never silently ignored or moved

Every driver that had shards before this seam wrote them flat: `shard_root/<key>.parquet`. The
generation directory is a path-prefix change, so those files stay perfectly good and the resume
check simply stops looking at them — the pass reports no skips and recomputes the whole corpus.
Silent, expensive exactly where this module is meant to save time (the live case is the TF-19
arm-values shards, a 64-match multi-hour pass), and indistinguishable from a healthy first run.

`generation_dir` warns, naming the count, the offending files and the exact destination. A WARNING
rather than a raise or an automatic move: `shard_root` is caller-supplied, so a flat `.parquet` is
only PROBABLY a stale shard, and relocating data on a guess is worse than the recompute it avoids.
The operator moves them.

### D11 — Provenance is THREE-state (`clean`/`dirty`/`unknown`), and the boolean stays beside it

`git_provenance` collapsed "git is unavailable" into `dirty: True`. The fail-closed BEHAVIOUR is
right and is unchanged. The RECORD was not: `dirty: true` is a positive claim that uncommitted
modifications exist, and on a tarball checkout or a box without git that claim is false — an
artifact asserting something untrue about its own provenance is the exact failure this module
exists to prevent, one level down.

`tree_state` is added **beside** `dirty`, never replacing it, and that is a correctness decision
rather than caution: `run_tree_dirty` is already published in every artifact on disk and is OR-ed
across workers by `aggregate_manifests`, and **`bool("clean")` is truthy** — a tri-state string in
the boolean's place would make EVERY aggregate report `dirty`, so a genuinely clean corpus would be
falsely marked dirty (the already-dirty case is unaffected). The two refusal messages are now
distinct, so the unknown case no longer borrows a dirty-tree sentence and lists
`(git unavailable)` where the changed files belong.

**`tree_state` is stamped at the 14 sites (across 13 drivers) that record THIS run's provenance, and deliberately not
at the three that AGGREGATE across workers.** OR-ing `clean`/`dirty`/`unknown` has no defined
meaning, which is also why `aggregate_manifests` drops a bare `str` — correct behaviour, now
visible through `dropped_fields` rather than silent.

The two-way split is not exhaustive, and the exception is instructive: `build_gkdv_arm_values` also
re-stamped this run's `run_commit`/`run_tree_dirty` onto the AGGREGATED corpus manifest — neither a
per-run stamp nor an aggregate. It OVERWROTE the cross-worker OR that `aggregate_manifests` had just
derived, so a corpus whose last-finishing worker was clean recorded `run_tree_dirty: false` while
another worker's slice came from a dirty tree. `build_layer2_spells` never did this: the two
producers `_partition.py` exists to keep identical had diverged on precisely the field its OR is
for. Removed.

**One policy fork, surfaced not decided:** with `unknown` still folded into `dirty`, a run on a
git-less box is refused unless `--allow-dirty`. Letting `unknown` pass unaided would loosen a
fail-closed control and is the owner's call, not this cycle's.

### D12 — `for_each` renders a real `[i/n]` when the corpus is `Sized`

An unflushed detached run is indistinguishable from a hung one, which is how a 14-hour pass became
unobservable. `[37/?]` says the pass is alive; `[37/80]` says when it ends, which is the question a
maintainer actually has. A generator keeps the `?` — counting an unsized corpus to render a
denominator would reintroduce the materialisation `for_each` exists to avoid.

## Consequences

- **No feature values change.** This is a `scripts/` resilience cycle: no library behaviour, no
  weights, no retrain. Two exceptions, both corrections, both named below.
- **`train_ghost_gk`'s feature cache invalidates once.** Its recorded token widens from the
  penalty-area geometry alone to the full declared input set. That is a bug fix: a re-run at a
  different `--subsample-fps` or `--carrier-*` silently reused the previous run's feature matrix
  while `metadata.json` recorded the NEW carrier parameters — the recorded==used invariant PR-S81
  exists to hold, broken by the cache underneath it. The next run re-extracts once.
- **All FIVE weight trainers join the provenance gate, in one go.** `train_ghost_gk` stamped
  `training_commit` into the SHIPPED artifact's `metadata.json` from a bare `git rev-parse HEAD`,
  which reads the same on a dirty tree — a verifiable-looking claim about code that may never have
  existed at that commit, in the highest-stakes artifact the repo publishes. The other four
  (`train_gk_completion`, `train_gk_retention`, `train_xshot_occurrence`, `train_xcross_attempt`)
  made no false claim because they recorded **nothing** — a different failure and not a lesser one,
  since an artifact nobody can trace to a commit cannot be reproduced or audited. All five now
  refuse a dirty tree by default (so `training_commit` is true by construction, needing no
  artifact-schema change) and record `run_commit`/`run_tree_dirty` in their metrics
  UNCONDITIONALLY — `--allow-dirty` only permits the value to be `true`. **Enrolled together deliberately: a partial roll-out is exactly how the prose
  version of this rule failed twice, which is why `test_provenance_wiring.py` exists at all.**

  **Enrolling a driver means sweeping its TEST invocations too** — a test run is by definition a
  dev run, so every one needs `--allow-dirty`. NINE such call sites exist across eight test files, and the full suite
  is what found the ones I missed by hand. No static gate was added for this, deliberately: a scan for
  `"scripts/<driver>.py"` in an argv list would miss the invocations built in a helper (the
  ghost-GK CLI test does exactly that), and a scan clever enough to follow helpers is the
  "keyword tests over source are not evidence of behaviour" trap this ADR indicts three times
  below. The suite is the gate; it fails loudly and names the driver.
- **`git_provenance` chopped the first character off the first dirty filename.** Porcelain v1
  encodes the status in the first two COLUMNS, so an unstaged modification begins with a SPACE
  (`" M CHANGELOG.md"`); `_git` applied `.strip()` to the whole output, removing that space from
  the FIRST line only, and the `line[3:]` slice then ate a character, so a refusal read
  `HANGELOG.md`. Only the first entry was affected, which is precisely why it survived: the rest of
  the list looked correct. **Scope, not inflated:** `dirty_files` is read only by
  `require_clean_tree`'s message and is persisted by no driver, so no committed artifact carries a
  mangled path — the cost was a diagnostic pointing at the wrong file at the exact moment its
  reader is trying to find out what made the tree dirty. `.rstrip()` fixes it; `rev-parse` is
  unaffected. Found by READING a refusal that this cycle's own new guard emitted — the guard
  catching a bug in its own reporting.
- **`measure_cover_shadow_argmax_agreement` carried a live ADR-028 RC1 defect** — a raw action-LTR
  `passer_xy` beside frame-LTR positions, with no home-only filter. RC1 (4.70.0) fixed the
  `features.py` callers; this driver imports `_compute_cover_shadow_dict` directly, so it was never
  a registered site. It does NOT cancel between the two arms it compares (only the CHEAP path
  consumes the passer), so `docs/research/cover_shadow_identity/`'s **0.1992 is a pre-RC1 number**
  needing an owner re-run. **The gating verdict survives without one, by arithmetic:** 0.157 × 970 =
  152 agreements against a 0.90 floor needing 873; even if every away row flipped to agreeing, the
  ceiling is **≤** 637/970 = 0.657 — 637 assumes the away half is ~485 rows, which the research
  README itself hedges as "roughly"; the conclusion holds for any away share below 74.3%.
- **`calibrate_tracking_defaults --source databricks` could not run at all.** It calls whichever
  loader `--source` picks with one kwarg set, and the bronze loader accepted neither
  `tracking_limit` nor `max_per_provider`, so every such invocation died on `TypeError` before
  reading a row. The bronze loader now implements both. Dropping the kwargs at the call site would
  have made two memory bounds silently inert on the loader that most needs them.
- **`train_gk_completion` bundling now declares its question.** `--mode {rebundle,retrain}` and
  `--reason` are REQUIRED with no default, and `metrics.json` records both plus the superseded
  coefficients. A retrain asserts the **served predictions** moved, not that some parameter array
  moved: `mean`/`std` are raw-feature statistics in metres, so a coordinate correction moves them
  while standardisation absorbs it exactly and every served probability is identical. The signature
  takes TWO probes, measured: a single shared probe asks whether two functions agree on one input,
  when the question is whether each model behaves the same on the coordinates IT sees.
  `--feature-space moved` refuses when `--probe-old` is ABSENT (see the 4.73.0 amendment: it was
  written here as "ALWAYS refuses", which was already wrong), because the weights directory stores
  `coef/intercept/mean/std` but no design matrix — a loud refusal naming why beats silently
  answering the wrong question. **Follow-up for ADR-011's artifact format: persist a fixed probe
  sample beside the weights.**
- **The re-bundle's "mirror defect" was investigated, MEASURED, and is NOT a defect. The check
  stays keyed on parameters.** The plan specified re-keying it onto served predictions, on the
  grounds that a geometry correction moves `_mean` by metres while leaving "every served
  probability identical", making the abort spurious. That comparison is *committed-model on OLD
  features vs fresh-model on NEW features* — measured at **1.7e-16**, genuinely identical, and
  **irrelevant**. A re-bundle **ships the COMMITTED weights**, and production then serves them on
  the **NEW** features; that comparison moves by **0.72 in probability**. So after a feature-space
  move the abort is correct and the right action is `--mode retrain`, which ships the fresh fit and
  whose two-probe guard handles the moved case properly. Chesterton's Fence, honoured by
  execution: the reason for the control still applies. Pinned by
  `test_a_rebundle_across_a_MOVED_feature_space_must_still_abort`, which records both numbers so
  the reversal cannot be quietly re-reversed. `_CORPUS_IDENTITY_ATOL` is untouched.
- **`aggregate_manifests` silently DROPS a bare `str` field** — it handles `partition`,
  `run_commit`, `run_tree_dirty` and `generation` by name, sums ints and merges dicts, and anything
  else matches no branch. This produced TWO findings in one cycle (`generation`, then `run_tree_state`). Kept as-is (a named case
  carries per-field semantics — contributor-gated vs OR-ed vs set-plus-flag — and one generic
  collector would give all of them one wrong semantic), but it now emits `dropped_fields` so the
  drop is visible in the artifact.

## What we got wrong, and the rule it produced

**Keyword tests over source are not evidence of behaviour.** Three separate errors in this cycle,
all the same shape:

- The first adoption gate scored capability TOKENS (`"shard" in src`, `"flush=True" in src`) and
  **certified three accumulate-then-write trainers** while pinning a genuinely resumable driver as
  debt for lacking `flush=True`. Wrong in both directions.
- The population detector counted `.append` only, so `.extend` and `out[k] = v` accumulation scored
  as "no per-item state" and misclassified two drivers as exemption candidates.
- `calibrate_xt_bandwidth` was scoped as "4 corpus loops, THE primitives-path proof". **Zero of the
  four were corpus passes**: the loop predicate matched an `ast.For` whose iterator source text
  contained `match|game|provider|cohort`, so it counted a per-game label loop, a `.unique()` string
  parse, a CV-**fold** loop, and `for col in ("game_id", "team_id", "player_id")` — a loop over
  three COLUMN NAMES. The one real walk is a comprehension, invisible to a predicate that only walks
  `ast.For`.

Derive the population structurally; make the VERDICT behavioural or adoption-based, never lexical.

**And a fix's own new parameter is where the next defect lives.** Every round-4 finding was a
consequence of a round-3 fix, sitting on the parameter that fix introduced. After changing a
signature, grep its call sites and ask which argument is now unsatisfiable.

## Amendment (4.73.0, PR-S141) — `probe_old` is ROW-ALIGNED, and the re-bundle gate fails closed on NaN

The first real use of the `--mode retrain --feature-space moved` path (the ADR-051 RC2/RC5 geometry
correction) found two things this ADR's decisions did not say.

**1. `probe_old` must be ROW-ALIGNED with `probe_new`, so it is the SAME corpus under pre-change
geometry — not the matrix the committed weights were historically fit on.** The Consequences above
describe the two-probe design correctly but leave its arity implicit, and the natural reading of
"the design matrix the committed model was fit on" sent this cycle after a 4.21.0-vintage matrix.
`predictions_moved` ends in an **element-wise** `np.allclose`, so a 1666-row historical matrix
against a 3491-row current corpus raises `ValueError: operands could not be broadcast together`, at
the guard, after the whole corpus pass is paid for.

**Do NOT rely on that raise — it is an accident of those particular numbers.** Measured: a **1-row**
probe BROADCASTS against any corpus and returns a verdict *silently*. That shape is not exotic; the
follow-up recorded above proposes persisting a fixed probe **sample**, and a one-row sample is
exactly what slips through. `_assert_retrain_moved_predictions` therefore compares row counts
explicitly (4.73.0) rather than depending on numpy to object. Two things in the code already implied the right answer and were read past: the docstring's
own next clause ("they are the SAME array whenever the feature space did not move" — a historical
matrix can never be that array), and `test_a_rebundle_across_a_MOVED_feature_space_must_still_abort`,
which constructs the moved case as `X_new = X_old + 5.0`, i.e. same rows, shifted geometry.

The guard asks a **serving** question — does what production emits change — not a fitting-provenance
one. So the probe vintage is the commit immediately before the change under test (here `641dadf`,
4.70.0), extracted over the same corpus the fresh fit uses. Row **order** agrees by construction
(both vintages concat in `load_matches` order; HEAD combines from `res.keys`, explicitly not
`reconcile`, whose filename sort would re-order). Row **count** does not agree by construction and
must be observed: `prepare_gk_completion_training_data` filters on `isfinite(length) & isfinite(dest_x)`,
both derived from the geometry a correction rewrites, so membership can shift. Compare `probe_old`'s
`n_rows` against the trainer's printed `N=`. Cost of learning this the other way: ~30 minutes of
corpus compute, where one call to `predictions_moved` with two mismatched shapes would have settled it
in seconds.

The ADR-011 follow-up above — *persist a fixed probe sample beside the weights* — stands, and is now
better specified: what must be persisted is a row-aligned design matrix under a **declared** geometry
vintage, not merely "a sample".

**2. The re-bundle parameter check now fails closed on non-finite drift and on a changed feature
count.** D-level decision unchanged — the check stays keyed on parameters, for the measured reason
recorded above. But extracting the two byte-identical `np.testing.assert_allclose` blocks into one
`_assert_rebundle_reproduces` introduced a regression the extraction made easy to miss: selecting the
worst-drifting parameter with `max(drift, key=...)` is **order-dependent under NaN**, because every
comparison against NaN is `False`, so `max` keeps whichever key it was already holding. Measured: a
**one-sided** NaN — a fresh-fit NaN against finite committed weights — in `intercept`, `mean` or
`std` was **accepted**, while the `assert_allclose` calls it replaced rejected it in all four
positions; only `coef` aborted, and only because it happens to be first.

The new form is also **stricter** than what it replaced, in exactly the case its abort message names.
`assert_allclose` defaults to `equal_nan=True` and treats matched same-signed infs as equal, so a
NaN at the **same index on both sides** — a committed artifact carrying NaN met by a fresh fit that
reproduces it — was accepted by the original in all four positions too (measured), and
`GkCompletionModel.load` has no finiteness check to rule it out upstream. An earlier draft of this
paragraph claimed the predecessor "rejected all four" without that qualifier; it is true one-sided
and false both-sided.

NaN weights are a degenerate fit, and accepting them ships the committed model while reporting that
the fresh fit reproduced it — the worst available direction for a bundling gate. Non-finite drift now
aborts unconditionally, a changed feature **count** aborts with its own message instead of a raw numpy
broadcast error mid-pass, and coverage is parameterized over all four served parameters **and over
both sides**, so the strengthening is pinned by a test rather than asserted in a comment (the
pre-existing tests moved only `coef`/`mean`, so a guard that stopped checking `std` stayed green).

## Amendment (4.77.1, PR-S146) — the SCHEMA token is the fifth token-completeness defect, and the first of its shape

This ADR concedes that `token_inputs` "cannot be checked for completeness", and an adversarial audit
before it merged found four **undeclared inputs**. This is the mirror case: the declaration was
complete, and the **shard SCHEMA changed underneath it**.

4.77.0 renamed `measure_match`'s `mean_visible_pitch_fraction` to `mean_observed_pitch_fraction` and
added `n_with_polygon` in `build_sb360_coverage.py`, while leaving `token_inputs["schema"]` at
`"sb360-coverage-2"`. The fingerprint digests `token_inputs` only, never the source, so the
un-bumped token resolved to the SAME generation directory — where **22 shards carrying the old
column** were waiting. A re-run would have skipped all 22 as already-done, printed a clean
`[i/n]`, asserted conservation successfully over its own keys, and combined the OLD schema. The
ADR-042 denominator fix would simply not have taken effect, with no signal anywhere.

**Nothing in the suite could see it.** The driver's only assertion on the renamed column lives in
its `e2e` test, which CI does not run (ADR-023), so CI was green on a broken driver.

**The pattern, now in CLAUDE.md:** a declared `_EMITTED_SHARD_COLUMNS` + `_SHARD_SCHEMA_VERSION`
pair, gated three ways — the declaration matches the dict the work function ACTUALLY builds (matched
by AST, not a hand-copied list); `token_inputs["schema"]` references the constant rather than a
literal; and the pair itself is pinned — plus a run-time assertion that fails at the FIRST shard
rather than at combine time. Each of the four defect reintroductions was observed RED.

Three details that cost something to learn:

- **Never write `pd.DataFrame(rows, columns=DECLARED)` as the drift check.** It SELECTS to the
  declaration, so a dropped key vanishes and a missing one arrives as NaN — the guard certifies
  exactly the two failures it exists to catch. Compare the keys the rows actually carry.
- **Name it `_VERSION`, not `_TOKEN`.** ruff `S105` flags a `*_TOKEN` string constant as a hardcoded
  password; a `# noqa` there would suppress a real check for a naming accident.
- **An empty result must still return the DECLARED columns**, so D10's "ran, produced nothing" stays
  distinguishable from "not yet run" even on the drift-checked path.

**Corollary for hand-rolled drivers.** A one-off measurement written the same day reproduced this
ADR's original motivating defect: it hand-rolled per-match shards (to avoid coupling to another
session's checkout) but iterated `load_matches(...)` directly with try/except around only the WORK.
`load_matches` is a GENERATOR — it raises out of the loop — so one bad match killed the pass at 81
of 179 items. Sharding bought back the 81 and nothing more. **If you hand-roll around `for_each`,
you must re-implement failure ISOLATION too, not just sharding:** invert onto the id list and load
one item per call inside try/except, writing a FAILURE shard so "absent" keeps meaning "not yet
run".
