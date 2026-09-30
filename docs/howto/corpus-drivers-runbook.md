# How-to: corpus-drivers runbook

> Class-2 procedural runbook (`docs/howto`, sibling to `docs/context`). `scripts/README_calibration.md`
> is a complete runbook for the TF-24 calibration harness ONLY; this is the missing generic sibling for
> the other ~35 `build_*` / `validate_*` / `measure_*` drivers. It says how to build a corpus driver on
> the shared `scripts/_driver.py` seam instead of re-solving resume / cache / provenance — a problem
> independently half-solved four times before unification. It POINTS at `_driver.py`'s docstring and
> `docs/context/corpus-drivers.md` (the WHY/measurement); it does not duplicate them.

## The rule: adopt `scripts/_driver.py`; never hand-roll the loop

Every corpus driver adopts the shared seam — CI-gated, not prose (ADR-052). `_driver.py` OWNS the
loop; you supply the per-item work. This exists because, of 21 in-population drivers, only three
survived a crash and fourteen held every result in memory and wrote once at the end (measured: an 8.7h
pass that lost all 64 matches when the cheap step after it raised). Read `scripts/_driver.py`'s
module docstring for the full WHY before building.

## `for_each` — the default shape

```
for_each(items, *, key=, work=, shard_root=, token_inputs=, load=..., token_reason=...)
```

- **Streams** the corpus (never `list()`s it) and writes **one shard per item** into a GENERATION
  DIRECTORY named by a `ruthless.fingerprint` digest of the DECLARED inputs (`token_inputs`). A changed
  declared input yields a different directory, so a stale shard can be neither read nor half-overwritten.
- **Skips** an item whose shard already exists (that is resume); prints a FLUSHED `[i/n]` line;
  records a failure rather than losing the pass.
- **An empty result STILL writes a shard** — absent means "not yet run", present-and-empty means "ran,
  produced nothing". Conflating them recomputes every barren item forever.
- `token_inputs={}` means "this pass has no staleness risk" and REQUIRES `token_reason` — a silent
  omission and a considered decision must not look identical in source.

## Resume BEFORE load (ADR-052 D13/D14/D15) — the part that keeps being missed

`for_each` resumes WORK, not the PRODUCTION of items. Make `items` cheap REFS and pass `load=`:

- Refs come from `list_match_refs` / `list_open_data_refs` / `list_gi_refs`; the `pining_source` /
  `open_data_source` factories return `(refs, load)`. `key(ref)` and the resume/exclusion checks run
  BEFORE `item = load(ref)`, and ONE `try` spans `load`+`work` — so a finished/excluded item is never
  loaded and an unloadable one is a recorded FAILURE, not a generator blow-up.
- The streaming `load_matches` / `load_statsbomb_matches` / `load_open_data_matches` wrappers are thin
  byte-identical shims for e2e / ad-hoc callers and are **banned from drivers** (Rule A). Do not stream
  a full tracking parse inside the loop.
- Every `load_match` states `events_only=` (Rule B); no un-sharded loading loop (Rule C); admitting an
  S1-excluded match into an events-only fit lives ONLY in
  `scripts._events_admission.events_only_loader` (Rule D). The four AST rules live in
  `tests/scripts/_corpus_load_rules.py`.

## Exclusions, conservation, combination

- **`.excluded.json`:** a deterministic exclusion (a geometry gate, a velocity-less frame set) raises
  `scripts._item_outcome.ItemExcluded` / `MatchExcluded` → a `.excluded.json` MARKER, counted in
  `CorpusPassResult.excluded`, replayed on resume. It is distinct from an empty shard and from a failure.
- **Combine from `res.shard_keys`** (keys minus failures minus exclusions), **never `res.keys`** — an
  excluded key has a marker and no parquet, so `res.keys` gives you a `FileNotFoundError`.
- **Escape-hatch primitives** (a loop that genuinely cannot invert) MUST call `assert_conservation`
  **AND** `_require_injective`. Conservation ALONE is satisfiable by a lossy run: a colliding key makes
  two items share one shard and `present` counts it once per duplicate (measured: 2 items → 1 processed
  → conservation returns `(2,2)` and PASSES). `reconcile` requires a partition surface; selectors stay
  OUTSIDE `token_inputs` so a narrowed run reuses shards.

## Shard-schema token (change the columns → bump the token)

A driver that changes its SHARD SCHEMA must bump its schema token, and the two are pinned TOGETHER: a
declared `_EMITTED_SHARD_COLUMNS` + `_SHARD_SCHEMA_VERSION` pair, with `token_inputs["schema"]`
referencing the constant (not a literal). Otherwise an un-bumped token resolves to the SAME generation
directory where stale shards wait, and a re-run skips them all and reports a clean CONSERVED pass over
the OLD schema (measured: 22 stale shards, 4.77.0). Add a runtime assertion that fails at the FIRST
shard. **Never** write `pd.DataFrame(rows, columns=DECLARED)` as the check — it SELECTS to the
declaration, hiding a dropped key and NaN-filling a missing one. Compare the keys the rows ACTUALLY carry.

## Provenance — refuse a dirty tree, stamp the run

Any driver that writes a registered artifact calls `require_clean_tree(git_provenance())` FIRST — in
`main()`, before paying for corpus work — and stamps `run_commit` + `run_tree_dirty` into its output
(ADR-037, `scripts/_provenance.py`). `git rev-parse HEAD` returns the same SHA dirty or clean, so a
bare-SHA stamp records a commit that does not describe the code that ran. Absent git counts as dirty.
`--allow-dirty` permits a dev run but the artifact still records `dirty: true` — the escape hatch never
launders the fact. Enforcement is in `main()`, never the work function (a `run()` that refuses on a
dirty tree can't be tested without mocking git; the CLI refuses, `run()` records the truth). CI-gated:
`tests/scripts/test_provenance_wiring.py`.

`declare_inputs()` (`scripts/_input_contract.py`, ADR-056) digests WHICH SYMBOLS the numbers depend on
(covariate tuples, extractor module identity, `GEOMETRY_VERSION`) into an `input_contract.digest`, so a
new covariate column or a `GEOMETRY_VERSION` bump moves the digest with nobody editing the driver.

## Memo-landing convention

A driver's result lands as `docs/research/<topic>/README.md` + `metrics.json`, each carrying
`run_commit` and `run_tree_dirty: false` plus the `input_contract` block. See
`docs/howto/construct-validity.md` for the validity-gate memo shape. A driver is enrolled in the
`ARTIFACT_DRIVERS` registry, whose population is DERIVED by enumeration and asserted EXACTLY
(`_UNDERIVABLE` empty; a floor cannot detect an omission — ADR-056). Content predicates read CODE
(`string_literals`), never docstrings.

## Environment

- **`PINING_FOR_THE_DATA_TOKEN`** — owner token; enables Gradient Sports. SkillCorner + IDSSE are
  public via the built-in public token, so no env var is needed for those. (A missing owner token does
  NOT fail loud for gradientsports — it silently yields nothing; verify your corpus size.)
- **`PINING_API_URL`** — override the pining base URL (defaults to the deployed instance).
- **`DATABRICKS_HOST` / `DATABRICKS_HTTP_PATH` / `DATABRICKS_TOKEN`** — only for a `--source databricks`
  (or `--xt-corpus-source databricks`) run; the databricks id space must MATCH pining match_ids or the
  fit fails closed rather than leaking held-out matches.

## Build a new `build_*` / `validate_*` / `measure_*`

1. Define cheap refs + a `load` (use the `pining_source` / `open_data_source` factory).
2. Write the pure per-item `work(item) -> tidy frame` (no per-item metadata in the frame — use the
   documented `counters(item, frame)` closure; no non-tabular side state).
3. Declare `_EMITTED_SHARD_COLUMNS` + `_SHARD_SCHEMA_VERSION`; set `token_inputs` (reference the schema
   constant + the real staleness inputs).
4. `main()`: `require_clean_tree` first, offer `--allow-dirty` + `--match-ids-json`/`--list-matches`
   for N-process partitioning, run `for_each`, then `reconcile` + land the memo with provenance.
5. Route every id through `id_compat`; add a `tests/scripts/test_*` guard.

Full field-level API and the measured failure histories are in `scripts/_driver.py`'s docstring and
`docs/context/corpus-drivers.md` — read them, do not copy them here.
