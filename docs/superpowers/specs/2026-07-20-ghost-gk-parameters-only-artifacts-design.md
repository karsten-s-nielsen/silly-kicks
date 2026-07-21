# Ghost-GK parameters-only artifacts, bundled-weights allowlist, and the `from_variant("public")` alias fix

**Status:** rev 3 — incorporates two external review passes + production-scale measurement; ready for the plan
**Date:** 2026-07-20
**ADR:** to be assigned at PR time. **The ADR's Decision section carries the disposition for
already-published artifacts** (§10) — that disposition is a decision, not an omission, and this
spec is not complete without it.
**Supersedes nothing. Amends:** ADR-016 (served estimator), ADR-038 (corpus taxonomy reach)

---

## Executive summary (read this first)

`GhostGkModel.save()` persists three per-sample arrays — `training_gk_x`, `training_gk_y`,
`training_leaves`. RFCDE evaluates a conditional density by running a weighted KDE over the
*actual responses* of training samples sharing leaves, so retaining training targets is inherent
to the method. The arrays were correct when the density was the served read-out.

**They have not been the served read-out since 4.14.0.** ADR-016 moved the served position to the
exact boosted HGBR mean (`predict_mean`), reconstructed from tree nodes and baselines alone. The
arrays now back exactly one emitted column, `ghost_gk_density_spread`, and nothing consumes it.

Four measurements define the change:

1. **`predict_mean` is byte-identical without the arrays.** Nulling all three leaves its output
   unchanged (max abs diff 0.0), and leaves `serve_ghost_gk_positions` and the `gkdv/` engine
   unchanged. Only the density pass breaks.

2. **`ghost_gk_density_spread` has no numeric consumer.** Inside `silly_kicks/` the value is read
   once, at its own emission site (`_ghost_gk.py:2143`). It is in no default xfn list. Downstream
   marts carry it as a passthrough; the one derived goalkeeper metric reads `ghost_gk_x`/`_y`,
   which come from `predict_mean` and are unaffected.

3. **The arrays are ~90% of the artifact.** Bundled `default` goes **7,376,181 → ~764,418 bytes**
   on a pure re-save — the three arrays are 6,611,763 bytes (89.6%), of which `training_leaves`
   alone is 86.7%.

4. **A per-leaf-aggregate replacement for the column works at production scale — and the route is
   still rejected, on other grounds.** This is stated carefully because the measurement is
   **scale-dependent** and an earlier revision of this spec got it wrong.

   At the bundled 36k `default`, no aggregate arm beat a no-model constant out-of-sample. At
   production scale (Stage-B, n_train=1,039,502 / 500 trees, same harness and same n=480 queries),
   the one-free-parameter arm **does**:

   | arm | 36k | production | production vs constant |
   |---|---|---|---|
   | CONSTANT (zero information) | 2.211% | 1.309% | — |
   | b=0.5, `c` fitted leave-one-source-out | 2.425% | **0.844%** | −0.465 pp, CI [−1.267, −0.348] → **beats** |
   | ORACLE exact-neff (0 free params) | 2.210% | 1.493% | +0.184 pp, CI [−0.881, +0.364] → ties |
   | FREE-b power law | 2.681% | 2.485% | +1.176 pp → loses |

   **The rejection therefore does not rest on this measurement.** It rests on items 1–3: aggregates
   must drop `training_leaves` too (96.0% of the artifact, and the feature-inversion channel), so
   they reduce nothing that stripping does not already reduce — they only preserve a column with no
   numeric consumer, at the cost of a fitted constant to version per artifact and a mode that
   collapses onto the mean (production: 100% of queries multimodal, `|mode−mean|` median 9.222 m).

   Method, seeds, query set and both runs are banked at
   `docs/research/ghost_gk_spread_aggregates/`, whose §3.4 leads with the reversal so a future
   reader who wants the column back finds a working method rather than a closed door.

   Robustness: at production scale 7 of 480 queries collide exactly with a training row in
   leaf-space (0 of 480 at 36k), all seven in one source. Rescoring on the 473 unseen queries
   leaves every verdict unchanged (constant 1.309% → 1.297%, b=0.5 0.657% → 0.667%).

   The structural mechanism strengthens with scale: `neff/n_train` median 0.9127 at 36k →
   **0.9501** at production. The density is barely conditional. Note the associated universal is
   itself scale-dependent and must be stated per scale — "every query draws on all training rows"
   is **false at 36k** (min 35,989 of 36,000) and **true at production scale**
   (min = median = max = 1,039,502).

This spec makes distributed artifacts **parameters-only**, retires the emitted column, and keeps
`predict_density` working for a locally-fit model. It adds a CI allowlist so no future bundled
artifact ships per-sample data unnoticed, aggregate corpus provenance, corrections to two false
model-card claims, and an ordering constraint on the queued Hub upload.

It also fixes an unrelated live defect in the same area: `from_variant("public")` on
`XShotOccurrenceModel` and `XCrossAttemptModel` returns the Hub-hosted `sc_extended` artifact.

**General principle established:** a distributed model artifact should contain learned parameters,
not per-sample training data. Where a method needs training data at inference, that capability
belongs to a locally-fit model, not a shipped one.

**What this document does and does not do.** It is forward-only: §2 changes `save()`, §4 migrates
via `load(old).save(new)`. It removes nothing already distributed. The disposition for
already-published wheels and Hub revisions is a separate, deliberate decision recorded in the
ADR (§10). Retraining is likewise out of scope by decision, not oversight (§10, decision 2).

---

## 1. What is true today

| | current |
|---|---|
| `save()` fitted-guard | requires all 7 attrs incl. the 3 arrays (`_ghost_gk.py:1781-1791`) |
| `load()` | reads the 3 arrays unconditionally (`:1935-1937`) — a stripped npz raises a bare `KeyError` |
| artifact `version` | `"1.2.0"` |
| `predict_density` guard | raises `"Model not fitted. Call .fit() or .load() first."` — misleading on a loaded model |
| emitting surfaces | **four**: `compute_ghost_gk`, `add_ghost_gk` (`features.py:4353`), `ghost_gk_xfns` (`features.py:4519`), and the atomic package's re-export of the latter two |
| corpus provenance | none — `train_ghost_gk.py` imports nothing from `scripts/_corpus.py`; metadata carries no corpus keys; the only bundled model with no `metrics.json` |
| bundled-weights control | none — a weights dir ships by existing on disk; no allowlist, no `MANIFEST.in` |
| wheel/sdist excludes | only `_ghost_gk_weights/full` and `_xcross_weights/full` (`pyproject.toml:136,143`) — `default` ships in both |

Existing fail-closed precedent to mirror: the pre-Option-A guard at `:1922-1928` raises a
"re-fit required" sentence rather than surfacing a `KeyError`.

## 2. Artifact format

### 2.1 `save()` never persists the arrays

Not a keyword argument, not a default — structurally absent. A flag reintroduces the failure mode
it exists to remove, and under the parameters-only principle there is no reason to persist them.

The 7-way fitted-guard drops to 4: `_tree_nodes`, `_tree_nodes_y`, `_baseline_x`, `_baseline_y`.

### 2.2 `load()` tolerates absence

The three attributes become `None` when the keys are missing, and are still read when present, so
existing artifacts load unchanged.

### 2.3 Version and metadata

`version` `"1.2.0"` → `"1.3.0"`, plus `"stores_training_data": false`.

`1.3.0` rather than `2.0.0` deliberately: served behaviour is identical (`predict_mean`
byte-identical, chirality fingerprint unchanged), and this artifact version has tracked format
increments rather than semver.

**Forward-incompatibility is real.** No released version can read a 1.3.0 artifact — `load()` at
`:1935-1937` reads the arrays unconditionally in every shipped version, so `<= 4.53.0` meets a
bare `KeyError`. Nothing can be done for already-released code. The *new* `load()` gains a
fail-closed guard shaped like `:1922-1928`. This is a version-pin consideration for anything
Hub-hosted.

### 2.4 `predict_density` error message

Current text is wrong under this change: the model loaded fine. New message states the artifact
carries no density support and that the remedy is a local fit, not a reload.

## 3. Public surface

### 3.1 Retired

- `ghost_gk_density_spread` from **all four** emitting surfaces (§1). Note `ghost_gk_xfns` at
  `features.py:4519` hard-codes `col_names = ["ghost_gk_x", "ghost_gk_y",
  "ghost_gk_density_spread"]` and iterates `range(3)`; the atomic package re-exports rather than
  reimplementing, so it needs no column edit but **is** separately test-covered (§8.3).
- `kde_backend` from the signatures where it becomes dead.

Per the repo rule that provably-dead columns need consumer sign-off *and* proof: the proof is
executive item 2; **owner sign-off was given in the 2026-07-20 design session and is to be
recorded in the ADR's Decision section at PR time.**

### 3.2 Kept

`predict_density`, `GhostGkDensity` (all eight fields), the KDE backends, and the ADR-013 /
ADR-014 acceleration ladder — all operating on a `fit()`-ed model.

This is the deliberate line: **the artifact stops carrying a corpus; the library keeps the
capability.** Deleting the density path would conflate artifact with library and retire two ADRs'
work as a side effect of dropping a column.

### 3.3 Model card corrections

Two claims in `docs/huggingface/model-cards/ghost-gk-v1-model-card.md` are false as written and
are corrected in this PR, landing with the stripped artifact:

- `:84` — *"Only the learned model parameters (tree structure, leaf-aggregated GK positions, KDE
  weights) are published — no raw provider tracking data is redistributed."* `_ghost_gk.py:1588`
  is `self._training_gk_x = np.array(y_x, copy=True)` — per-sample targets, not leaf aggregates.
  Arithmetically it cannot be an aggregate: no tree exceeds 31 leaves, yet 20,139 distinct x
  values are stored.
- `:82` — the Gradient Sports row's *"the underlying raw tracking data is **not** redistributed."*

**`:82` becomes true of a 1.3.0 artifact and needs no rewording. `:84` does NOT** — of its four
clauses, only two are repaired by the strip. `save()` persists (`:1797-1812`) the three arrays plus
`n_trees`, `n_trees_y`, `baseline_x/y`, `tree_nodes_*` and `tree_dtype_*`. **KDE weights are not
among them and never were** — `_leaf_match_weights` recomputes them per query at `:1697`. So "KDE
weights" was false before this change and is *more* false after, its only charitable referent (the
arrays they are derived from) having been removed. A minimal edit that deletes one clause and
ships the other would leave a false sentence standing.

`:84` is therefore replaced wholesale, not patched:

> Only the learned model parameters are published: the two gradient-boosted tree ensembles (split
> thresholds, feature indices, leaf values) and their additive baselines. No per-sample training
> data and no raw provider tracking data is redistributed.

**Sequencing — the card is ONE file scoped to BOTH variants**, and it carries HF YAML frontmatter,
so it *is* the Hub README. Correcting it makes it accurate for the bundled `default` and false for
the Hub `full` until that artifact is stripped. The repo file is therefore corrected in this PR;
**the Hub README push is gated on the stripped Hub artifact**, the same gate as §7. Precedent for
the mechanics — and the reason the pathway is live — is commit `1b56ad8`, a card-only Hub push.

The card also documents `predict_density` with a runnable example, which stays valid for a
locally-fit model but not for `from_variant("default")`; it gains a precondition line.

## 4. Migration

No retrain. No bespoke strip script. Old artifacts load (arrays present); the new `save()` drops
them:

```python
GhostGkModel.load(old_dir).save(new_dir)   # SHA256SUMS regenerates
```

Two properties make this cheap, both measured: `predict_mean` is byte-identical without the
arrays, and `_chirality_block` calls only `predict_mean`, so the ADR-040 fingerprint is unchanged
and `verify_chirality` passes without re-fingerprinting.

**Provenance laundering — must be handled, not inherited.** `save()` stamps
`sklearn.__version__` at save time (`:1834`), so a naive `load(old).save(new)` rewrites a recorded
training-time version to the migrating machine's. `training_commit` and `training_platform`
survive (restored onto `self` by `load`), but `sklearn_version` does not. In a change whose §6
adds provenance, silently rewriting existing provenance is not acceptable: the migration path must
preserve the loaded `sklearn_version` rather than re-stamp it.

## 5. Bundled-weights allowlist (a CI gate)

Named accurately: this is a **CI test**, not a build-time hook. Adjacent real gap worth recording
— `publish.yml` has no `needs: ci`, so a tag can publish without the suite having passed.

**Rejected form:** "no array dimension may equal a declared corpus size." The bundled artifact's
dimensions are `{1, 243, 2968, 3416, 36000}` and the corpus sizes appear among them not at all,
because the training subsample cap decouples stored rows from corpus rows by design. The only
declaration that *would* fire is one read off the array shape — circular.

**Adopted form — a NAME allowlist.** Each bundled artifact declares the array names it may contain;
anything else fails. `training_gk_x`, `training_gk_y` and `training_leaves` are caught by name.

**The shape half is NOT specifiable as an earlier revision claimed, and the claim is withdrawn.**
Measured on the bundled artifact:

```
tree_nodes_* shapes: {(3416,): 242, (2968,): 1}   n_trees: 243   n_trees_y: 82
n_estimators: 500   max_depth: 8   max_leaf_nodes in metadata: False
```

Neither ensemble count equals `n_estimators` (early stopping), node-array size is governed by
`max_leaf_nodes` which is not recorded in metadata, and the tree ensemble is not even
shape-uniform (tree 237 has 27 leaves, not 31). The only bound derivable from the hyperparameters
that *are* recorded is `56*(2*2^8−1) = 28,616` elements — roughly 8× loose, which a
`training_gk_x` from a 20k-sample fit would pass.

Consequence, stated rather than glossed: **the gate is name-based, and a name-based allowlist is
defeated by renaming an array.** It is a guard against inadvertence, not against a determined
author. Recording `max_leaf_nodes` in metadata (§6) would make a tight shape bound writable later;
that is not attempted here.

**Honest scope: it fires on 1 of 7 bundled weights directories.** The other six pass because they
contain no per-sample array to inspect — vacuous passage, not detection. Stated so the gate is not
read as broader than it is.

**It cannot fire on a Hub-only push** — see §7.

## 6. Corpus provenance

`metadata.json` gains an aggregate provenance block: provider list, match count, row count.
**Aggregate only — match identifiers are never recorded, and no public/restricted split is
stated.**

**The join trap, which fails silently.** Do not derive `match_id` from `groups` — that is the
parquet `game_id`, and the trainer's comment at `:223-224` notes SkillCorner's is a kloppy hash
while the directory name is the match id. Derive from `pq_path.parent.name`.

A botched join produces no error: `is_public_row` is fail-closed, so a failed match yields an
all-restricted classification and an empty public arm, silently. The gate must assert the join
**lives** — `is_public.sum() > 0`, matched ids equal the registered subset — and that a
deliberately corrupted map **fails**. A one-sided assertion passes identically whether the join
works or does nothing.

## 7. Hub upload ordering constraint

`TODO.md:167-181` queues an upload of the 179-match Stage-B artifact (~208 MB), leaning overwrite
in place. Produced by today's `save()`, that artifact carries the same three arrays at roughly 29×
the bundled volume — and §5's allowlist is scoped to bundled weights directories, so it
structurally cannot fire on a Hub-only push.

**Hard constraint: the Hub `full` upload must be produced by the post-strip `save()`, and must not
proceed before this PR lands.** This is an ordering edge, not a deferral: without it, the deferral
in §10 enlarges what is distributed rather than holding it constant.

**The runbook that performs the upload must be corrected in this PR too.**
`docs/research/tf19_pr2/hf_upload_instructions.md:174-186` is a "Licensing note" that certifies the
upload as *"learned parameters only — tree structure / leaf values / calibration"* and quotes the
ghost-GK card claim **verbatim** as its precedent. It stages `rfcde_weights.npz` with an executable
`scp`. Correcting the card (§3.3) while leaving its duplicate in the document that executes the
upload would fix the claim everywhere except where it is acted on. Three edits:

1. the `:174-186` licensing note updated to match the corrected `:84` wording;
2. the ordering constraint above stated in the runbook itself, not only here;
3. a **pre-upload assertion** in the runbook: the staged `rfcde_weights.npz` must contain none of
   the three per-sample array names. A one-line `np.load(...).files` check, run against the exact
   file about to be uploaded — the same allowlist rule as §5, applied at the one seam §5's
   bundled-directory scope cannot reach.

`TODO.md:167-181` is updated in the same pass; it currently classifies the upload as "a live
breakage, not a nice-to-have", which is an instruction to proceed.

## 8. `from_variant("public")` alias fix

Independent of the artifact work; shares a root-cause class (declared identity diverging from
actual contents) and the same `from_variant` / `metadata.json` seams.

### 8.1 The defect

`_xshot_occurrence.py:549` and, byte-equivalently, `_xcross_attempt.py:576`:

```python
elif variant in ("public", "sc_extended"):   # the Hub-hosted variants
    model = cls.from_hub(_HF_REPO_ID)
```

No bundled `public/` directory exists, so `from_variant("public")` falls through to
`snapshot_download` and returns the Hub artifact, whose metadata records
`shipped_variant: 'sc_extended'`. The result is memoised under the key `"public"`.

Root cause is a stale alias: 4.9.0 reserved the name for a public Hub artifact never created;
PR-S118 added `sc_extended` to the tuple without re-auditing the left-hand side.

`GhostGkModel` is unaffected (`variant == "full"`, no variant cache, no `"public"` member).

### 8.2 The fix

```python
_VARIANT_ALIASES = {"public": "default"}   # the bundled default IS the public arm
_HUB_VARIANTS = frozenset({"sc_extended"})
```

The alias resolves **before** the cache lookup — memoising under the requested rather than the
resolved name is half the defect.

Mapping rather than raising is the literal truth: the bundled `default` metadata for both models
already declares `shipped_variant: 'public'`. Precedent at
`tests/tracking/test_gk_completion_variants.py:289-296` (the 4.22.1 `gs`→`default` alias, pinned
by object identity).

**No variant `Literal`.** To stay honest it would enumerate `sc_extended`, promoting a Hub-only
artifact to a typed first-class option — widening exactly the surface this work narrows.

### 8.3 Serve-time gate — restated

An earlier draft specified "requested name equals the resolved artifact's recorded
`shipped_variant`". **That is unsatisfiable and contradicted §8.2**: the bundled `default` records
`shipped_variant: "public"`, so the equality is `False` on the primary path.

The gate asserts instead:

- the **alias-resolved** name maps to an artifact whose recorded `shipped_variant` is in the
  allowed set for that name — i.e. `"public"` and `"default"` both resolve to the artifact
  recording `"public"`, and that is the passing case;
- `_HUB_VARIANTS.isdisjoint({"public"})` — a name presented as reproducible must resolve inside
  the wheel;
- no requested name resolves to an artifact recording a Hub-only `shipped_variant` unless that
  name is itself in `_HUB_VARIANTS`.

ADR-038's existing guard drives the *trainer* and asserts on a `metrics.json` key; it cannot
observe a loader serving a mislabelled artifact.

### 8.4 Release note

The alias fix gets its own CHANGELOG entry rather than being folded into the artifact work, so a
user-actionable correction keeps its own visibility.

## 9. Test surface

**The inventory below is grep-derived, not hand-listed.** An earlier draft listed 5 items across
4 modules; the actual surface is 7 test modules, a generator, and two committed binaries. These
are goldens, which per repo policy cannot be marked `slow`, so they run on every matrix leg — an
incomplete inventory converts one pass into a red-CI discovery loop.

### 9.1 Carrying `ghost_gk_density_spread` (from `grep -rn`)

- `tests/tracking/test_ghost_gk.py` (6 hits)
- `tests/tracking/test_ghost_gk_integration.py` (4)
- `tests/tracking/test_action_ltr_mirror_invariance.py` (3)
- `tests/tracking/test_ghost_gk_frame_restriction.py` (2, incl. `_GHOST_COLS`)
- `tests/tracking/test_ghost_gk_serve_mean.py` (2)
- `tests/tracking/test_ghost_gk_refactor_equivalence.py` (1)
- `tests/test_add_star_purity.py` (1; both registered variants — the `precomputed` fixture sets
  the column explicitly)
- `scripts/make_ghost_gk_golden.py` (1)
- binaries: `tests/tracking/data/ghost_gk_refactor_golden.npz`,
  `tests/tracking/fixtures/ghost_gk_backward_compat.parquet`, and
  `tests/tracking/fixtures/ghost_gk_kde_golden.npz` (714,048 B) — all three need regenerating.
  The third carries no hit for the column string, which is why a string-derived inventory missed
  it; it is reached through `default_model_features` (§9.2)
- `tests/tracking/test_aggregator_column_liveness.py:362` — the `add_ghost_gk` entry (matched by
  aggregator name, not column string)

**Two generators, not one.** `make_ghost_gk_golden.py` carries the column name;
`gen_ghost_gk_kde_golden.py` uses the `GhostGkDensity.spread` *field* and has no hit for the
column string. Both are in scope, differently.

### 9.2 Failures at the fixture, not the assertion

Six tests bind `default_model_features`, which calls `from_variant("default")` then
`predict_density`. Post-strip that raises, so tests **with no spread assertion at all** — e.g.
`test_golden_discrete_mode` — break entirely. This is a fixture-level change, not a two-line
assertion edit.

### 9.3 `kde_backend` removal

Nine breaking keyword call sites across six files, two of them in meta-pinned registries
(`conftest_id_scalar.py:877,1083`, `conftest_id_dtype.py:175`).

Most pointedly, `tests/tracking/test_ghost_gk_kde_vectorized.py:970` asserts
`"kde_backend" in inspect.signature(add_ghost_gk).parameters` — on the *atomic* mirror. That is a
guard whose purpose is to forbid this removal. It requires deliberate retirement with a recorded
reason in the ADR, never silent deletion.

### 9.4 Backend parity: what survives, and what is retired with a reason

**Kept — backend-vs-backend parity on a fitted fixture.** ADR-013 / ADR-014 gate the scipy oracle
against `vectorized`, `cpu-numba`, `fft` and `fft-cic`. The backends are kept, so this gate is kept
and changes fixture source to a locally-fit model. It genuinely catches kernel-implementation
divergence. Sizing is measured: the existing `small_model` (400 samples) fails the `1e-2` fft bound
on 2 of 4 samples; a 4000-sample fit passes at ~2.3× margin. The bound stays unchanged — loosening
it to accommodate a too-small fixture would silently weaken the guarantee.

**Retired, deliberately, with the reason recorded here and in the ADR — real-model fft fidelity.**
`test_golden_fft_scalars` states in its own docstring that it *"Locks the real-model fidelity in
CI -- the synthetic kernel-parity test uses broad clouds"*. An earlier revision of this spec
proposed swapping its fixture to a fitted model and called that a fixture change. It is not: it
makes the property unmeasurable, because **kernel width is a function of `n_train`**.
`_kde_setup` sets `factor = neff**(-1/6)`, so:

```
REAL       n_train=36000  neff=27322.4  bandwidth factor=0.1822
SYNTHETIC  n_train=  400            factor=0.4113  = 2.26x broader
SYNTHETIC  n_train= 4000            factor=0.2554  = 1.40x broader
SYNTHETIC  n_train= 4000, 100 est   factor=0.2530  = 1.39x broader
```

No practical fitted fixture reaches the real regime, and more estimators do not help. The
docstring's caveat is **correct, not stale** — the measurement was run specifically to test whether
it was stale, and it is not.

There is no third option: reproducing the regime would require committing per-sample density
inputs, which is the thing being removed. **Coverage lost: fft scalar fidelity against a
36k-sample model is no longer verified in CI.** Recorded rather than absorbed, per the standard
§9.3 sets.

### 9.5 Structural perf guards — re-anchored, not deleted

`test_ghost_gk_frame_restriction.py:126-145` and `:367-390` are structural perf guards (the second
so labelled in its own docstring). Both monkeypatch `predict_density`, append to a `captured` list
and read `captured[-1]`. Post-strip the call never happens, `captured` is empty, and both die with
`IndexError` — this is not the string edit §9.1's hit-count implies.

Re-anchoring to `predict_mean` would be near-vacuous: `_ghost_gk.py:2063` documents it as
*"~free (leaf-value traversal)"*, so a spy on it would guard an operation the library itself calls
free. Wall-clock perf tests are forbidden in this repo, so the spy is the sanctioned mechanism and
the question is what to spy on.

**Measured answer: the feature extractor.** On the `_make_dense_match` fixture (2,750 rows,
500 GK rows):

```
extract_all_ghost_gk_features :  1547.9 ms  (n=500 rows)
predict_mean                  :    87.0 ms
RATIO                         :    17.78x
```

`_extract_all_ghost_gk_features` is the dominant remaining cost by ~18×, so spying it preserves a
guard with real discriminating power. Both tests re-anchor there, asserting the extracted row count
rather than the density row count. The behavioural assertion `restricted_n == 2 * len(linked)` is
unchanged in meaning.

### 9.6 Version-pin surface

`§2.3`'s `1.2.0` → `1.3.0` bump is a **third breaking axis**, independent of the column string and
`kde_backend`, and both earlier inventories missed it because both were derived by grepping those
two. Three call sites assert the literal:

- `tests/tracking/test_ghost_gk_r3.py:50`
- `tests/tracking/test_ghost_gk_serve_mean.py:219`
- `tests/tracking/test_train_ghost_gk_cli.py:65`

(`tests/spadl/test_add_possessions.py` also matches `1.2.0` but refers to the library version in a
docstring — not in scope.)

### 9.7 New

- Allowlist gate over every bundled weights dir (§5), with a negative case: an artifact carrying a
  per-sample array must fail.
- Provenance join-liveness gate (§6), with the corrupted-map failure case.
- Serve-time variant-identity gate (§8.3).
- Round-trip: `load(1.2.0).save()` yields a 1.3.0 artifact whose `predict_mean` is byte-identical
  to the source **and whose `sklearn_version` is preserved, not re-stamped** (§4).

### 9.8 Scripts retired

`scripts/measure_ghost_gk_estimators.py`, `scripts/validate_ghost_gk_refit.py`,
`scripts/measure_ghost_gk_serve_delta.py`.

All three score on `mode_x`/`mode_y` — the KDE mode, not served since ADR-016 (4.14.0).
`validate_ghost_gk_refit.py:38` carries the dead reason in a comment (*"lack predict_mean, so both
models are scored the SAME (mode) way"*, true in 4.10.0 when the incumbent predated Option A). Two
are one-shot instruments whose answers are banked: the ADR-016 estimator choice and the PR-S81
carrier serve-delta.

A future re-fit gate should be written fresh against `predict_mean`.

## 10. Decisions recorded in the ADR, not deferred silently

Three judgement calls shape this document's scope. Each is defensible; none is an oversight, and a
reader should be able to tell. The ADR's Decision section records each with its reasoning:

1. **Forward-only.** This change prevents recurrence; it removes nothing already distributed. A
   disposition for already-published wheels (`default` ships in wheel *and* sdist per
   `pyproject.toml:136,143`; PyPI files are immutable and a yanked file still resolves for a
   pinned `==`) and for existing Hub revisions is recorded separately in the ADR.
2. **No retraining.** The strip transforms existing artifacts; corpus composition is unchanged.
3. **Framing.** The document is written as artifact hygiene. That is a decision about a public
   repository, taken with the disposition above, not an absence of one.

## 11. Out of scope

- **Alternatives retaining the arrays in transformed form** (quantisation, resampling, summary
  substitution). All assessed; none beats removal, which is simpler than every one of them.
- **Per-leaf aggregate replacement for the column.** Rejected on product grounds, **not** on the
  measurement — at production scale the one-parameter arm reproduces the column (executive item 4).
  This is an **open fork with a working method banked**, deliberately not a closed one.

## 12. Companion documents updated in this PR

The spec's changes falsify assertions in four documents outside `silly_kicks/`. All four are
corrected in the same commit:

- `docs/huggingface/model-cards/ghost-gk-v1-model-card.md` — `:84` replaced, `:82` becomes true
  (§3.3). Hub push gated per §3.3.
- `docs/research/tf19_pr2/hf_upload_instructions.md` — `:174-186` plus the ordering constraint and
  pre-upload assertion (§7).
- `TODO.md:167-181` — the upload entry, which currently reads as an instruction to proceed (§7).
- `pyproject.toml:135` — the comment *"The `default` weights (~12 MB) ship bundled"*. Already wrong
  (the artifact is 7,376,181 bytes ≈ 7.4 MB) and would become ~15.7× wrong at ~764 KB. It sits one
  line above the `exclude` this spec cites twice.
- `CLAUDE.md` — `:13` records the bundled variants as *"`default` (9 MB, 36k samples) / `full`
  (91 MB, 537k samples)"*, all three figures stale; `:19` states *"`predict_density` retained for
  `ghost_gk_density_spread` + the mode"*, false for a loaded artifact post-strip.

C4 is unaffected (count stays 30 — no aggregator, backend or model is added or removed) and
`NOTICE` needs no change; both were checked rather than assumed.

## 13. Open items

- Whether any external consumer reads the spread column specifically. The search covered
  `silly_kicks/` and the downstream marts, not every possible caller.
- Version-pin guidance for Hub-hosted 1.3.0 artifacts, given §2.3.
- **Feature-vector inversion from `training_leaves` — SETTLED (was an apparent conflict).** The two
  earlier measurements reproduced to the digit and measure orthogonal axes, so they never actually
  conflicted. Reconstructed per-row interval bounds are **tight on geometry** (`defensive_line_x`
  median 0.245 m, `defensive_line_depth` 0.162 m; attacker/defender x two-side-bounded on >99% of
  rows) and **loose on every corpus-join key** (`time_seconds` ~24-min window, `score_diff` its full
  5-goal range with 0 rows pinned below one goal, `phase`/`period_id`/`possession` never
  two-side-bounded; only 4 of 36,000 rows jointly tight on time and ball position). The "internally
  inconsistent" (lo > hi) bounds were diagnosed to NaN routing through `missing_go_to_left` — 100%
  attributable, since intersecting only NaN-excluding edges gives exactly 0 inconsistencies — and
  that same artifact had inflated the "tight" headline (corrected 18.53% → 14.51%, median 17 → 15).
  **Conclusion:** `training_leaves` recovers keeper-frame geometry with high fidelity but is a weak
  instrument for re-identifying which match/second/scoreline a row came from.

  **Scope boundary that must not be lost:** this covers the leaf-interval channel ONLY. The artifact
  also stores `training_gk_x/y` **verbatim** — a separate, more direct exposure that this analysis
  does not touch. "Inversion is weak" is true of corpus join keys, NOT of the stored coordinates.
  The strip removes all three arrays regardless. Banked at `docs/research/ghost_gk_spread_aggregates/`
  is the aggregate study; the inversion measurement itself was read-only and is not banked (it
  requires the arrays this change removes — re-run before the strip lands if a durable record is
  wanted).
