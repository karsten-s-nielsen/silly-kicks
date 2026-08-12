# Ghost-GK re-fit onto the canonical box constant, and the TF-24 recommendation refresh — design

**Status:** approved 2026-08-11; **revised after external review** (same date, findings verified
against `21d5ef4` — see §9). One threshold remains to be pinned inside the plan (D5's sensitivity
number); the gate it used to sit on was replaced, not re-tuned.

**The ADR number and version are assigned at COMMIT-PREP, read off `main` at that moment — never
pre-claimed** ([[no-version-number-until-commit-prep]]). Registers at writing time: `main` @
`21d5ef4`, 4.79.0 / PR-S149 / ADR-059. **Confirm; do not assume** — this cycle lost three numbers to
a concurrent session by assuming.

**Closes:** ADR-050 §6 (the deferred `_ghost_gk` unification) and the "`from_hub` stays broken until a
stripped artifact is uploaded" follow-up.
**Refreshes:** TF-24 (ADR-009), whose every recommendation on record was computed on geometry the
ADR-028 cycle has since corrected.

---

## 1. Context

Two On-Deck items, both ready, both blocked only on compute, and **sequenced**: the ghost re-fit
lands first so one corpus download serves both, and so any downstream recompute picks up the new
weights rather than forcing a second pass.

**Ghost-GK.** `_ghost_gk` still derives its penalty area from **40.3** (half-width 20.15) while
`spadlconfig` carries the Law's **40.32** (20.16). `attackers_in_box` is a trained feature, so
flipping the constant without a re-fit is train/serve skew. ADR-050 made that a mechanism: the
artifact records the constant, so an unaccompanied flip makes `load()` **raise**.

**TF-24.** Every sweep on record consumed at least one mixed-convention input. What remains is a
*recommendation refresh*, not a defect repair — ADR-009's standing rule is that TF-24 **recommends
and never changes library constants**.

### 1.1 The delta has exactly TWO contributors, and the third is provably nil

```
ghost now:   (atk_x  <  16.5) & (atk_y >= (68-40.3)/2) & (atk_y <= (68+40.3)/2)
canonical:   (gr_x  <=  16.5) & (abs(y - 34.0) <= 20.16)
```

1. **Band** — half-width 20.15 → 20.16, i.e. a **1 cm** strip at each edge.
2. **Boundary** — depth `<` → `<=`, flipping a player sitting exactly on `x = 16.5`.
3. **Representation** — min/max band → abs-distance. **Measured nil at the canonical value: the two
   forms agree on every double straddling both bounds (0 / 1602), not merely on the bounds
   themselves.**

   **Bound equality alone would NOT have established this, and the legacy constant is the
   counterexample.** At 40.3 the bounds are *also* bit-identical (`(68-40.3)/2 == 34-20.15 ==
   13.850000000000001`) and yet the predicates **disagree at exactly `y = 13.85`**: the double
   `13.85` sits fractionally below the rounded bound, so `y >= lo` is `False`, while
   `abs(13.85 - 34.0)` lands on exactly `20.15` and `<= half` is `True`. The band form compares `y`
   to a *rounded bound*; the abs form *rounds a subtraction*. Different operations, and only an
   exhaustive sweep near the bounds distinguishes them.

   The float error lives in the form being **deleted**, so the migration removes noise rather than
   introducing a profile. **`y = 13.85` under the legacy constant is a real, reproducible
   disagreement between the two forms and is therefore the single most valuable case for §6's
   property-test grid.**

**D2 must therefore attribute `n_flipped` to (1) vs (2), not merely report it.** A flip count that
cannot be decomposed is a number that cannot be reasoned about next cycle.

---

## 2. Scope

**In:** a vectorized canonical box predicate + migration of both vectorized sites; the ghost constant
unification, re-extraction, re-fit of **both** variants, and re-stamp; HF upload of both variants plus
model-card corrections; the TF-24 Stage-1 argmax check and Stage-2 sweep.

**Out:** adopting any TF-24 recommendation as a library default (ADR-009 — a separate PR); the
owner-held ghost-GK **remediation actions** (privacy request, HF token downscope, historical PyPI
wheel disposition, the off-repo audit bundle). Publishing a clean artifact is NOT one of those.

**One feature branch, minimal non-squashed commits, merged with `--merge`.**

---

## 3. Decisions

### D1 — One vectorized predicate, both sites migrated

`_geometry.py` gains `in_penalty_area_goal_relative_array(gr_x, y)`, the array sibling of the scalar
predicate. `_ghost_gk` and `_xcross_attempt` both migrate onto it.

**ADR-050 §6 never evaluated this.** It chose between *"call the scalar helper"* and *"rebind the
constants"*, rejecting the former **because the site is vectorized** — an argument against the
**scalar** helper, not against creating a vectorized one. Rebinding leaves the *expression*
duplicated across two modules, single-sourced only on the constants.

**The invariant this establishes, stated so it can be defended:** after this cycle the domain rule has
**one expression and three call sites** (scalar `_geometry`, ghost, xCross).
`tests/tracking/test_geometry_constant_enumeration.py` is what keeps the constants from re-scattering;
the new property test in §6 is what keeps the two *forms* agreeing.

**Not a straight substitution.** xCross's predicate is `… & ~is_ball`; the helper has no ball concept,
so the call site composes (`helper(gr_x, y) & ~is_ball`). Trivial, but an implementer attempting a
one-line replace gets a shape error.

**xCross is byte-identical and needs no re-stamp.** `feature_contract()` hashes `probe_sha256` + the
extracted **values** + the declared **constants** — **not the source expression**
(`_feature_contract.py:126-146`, verified). Same constants and operators give the same values, hence
the same fingerprint.

**Proof obligation:** byte-identity by **grid sweep** over the xCross site, plus its existing suites
unchanged. **If the sweep shows any delta, the premise is wrong and we stop.**

### D2 — Measure the feature delta before fitting; the result decides the ship claim

Re-extract `attackers_in_box` under both constants on the corpus, as a provenanced artifact under
`docs/research/`. Let `n_flipped` be the number of extracted rows that differ, out of `n_rows`:

- **`n_flipped == 0`** → ships as **"unification, measured no-op"**, citing `0 / n_rows`. The re-fit
  still happens; only the claim changes.
- **`n_flipped > 0`** → report `n_flipped`, the fraction, **and the split between band and boundary
  contributors** (§1.1), plus a before/after weights comparison in the PR.

No threshold: zero-versus-nonzero is the whole question.

**Two things ride in the same extraction, because the rows are already in hand and a second pass is
expensive:**

- **The behind-the-line population.** ADR-050 parks *"should a point with `gr_x < 0` count as
  in-box?"* If that is later answered "add a lower bound", `attackers_in_box` changes **again** and
  these freshly-fit weights go stale — a second corpus pass for a question whose data we are already
  holding. Count the `gr_x < 0` population now; the decision stays deferred, the measurement does not.
- **A frozen flip-band fixture.** D2 otherwise produces a number and no regression. Freeze a small set
  of rows sitting inside the flip band so a future constant change fails a test instead of demanding
  another corpus pass.

### D3 — Both variants, one extraction, two fits

`--variant` is only a **label** recorded in metadata; the real axis is `--subsample-cap`, applied
**after** extraction (`train_ghost_gk.py:648-656`, verified). So `default` (~36k samples) and `full`
(uncapped) come from **one extraction feeding two fits** — the expensive part is shared.
`--training-platform` is recorded in metadata (e.g. `dgx-spark-aarch64`).

### D4 — The declaration must derive from the same source as the predicate

**This closes a hole that would otherwise make the whole cycle self-defeating.** Ghost declares its
contract constants *independently of its predicate* (`_ghost_gk.py:1577-1578`):

```python
constants={
    "penalty_area_half_width": (_PENALTY_AREA_Y_MAX - _PENALTY_AREA_Y_MIN) / 2.0,   # -> 20.15
    "penalty_area_depth": float(_PENALTY_AREA_X),
}
```

Migrate the *predicate* only, and the re-stamp records **20.15 for a 20.16 extractor** — an artifact
that lies about the geometry it was fit on, which is exactly what ADR-050's contract exists to
prevent.

**Nothing today would catch it.** Both enumeration guards assert on **key names, never values**:
`_module_level_geometry_constants` collects `set(meta["feature_contract"]["constants"])` (`:120`) and
the pins are key-set equalities (`:158-160`). Every gate stays green.

Therefore: the declaration derives from the **same single source** the predicate consumes, and a new
test asserts the declared **value** equals the value the predicate uses.

**The value test must be GENERAL, keyed on the canonical name — not ghost-specific — because D4
relocates the constants out from under the existing gate's mechanism.** After D4, ghost has **no
module-level geometry constants at all**, and `test_geometry_constant_enumeration.py` works by
enumerating module-level constants and requiring each to be declared-or-exempt. So it would enforce
**nothing for ghost**, and the loss is silent: `test_the_enumerator_is_not_vacuous` (`:86`) only
asserts `len(found) >= 4` **across all modules**, so it stays green on xCross's remaining aliases.
That is precisely the "reads as complete, enforces nothing" failure its sibling was written against.

**D4's value test therefore TAKES OVER that responsibility rather than supplementing it:**

```python
# for every model, for every declared key: the stamped value IS the canonical value
for model, keys in built.items():
    for key in keys:
        assert stamped[model][key] == getattr(spadlconfig, key)
```

Keyed on the canonical name rather than on where a constant happens to live, it survives the
relocation, covers all three extractors, and would have caught the 20.15/20.16 divergence from the
day it was introduced.

**`_feature_contract.py` is an unlisted file in this change.** `DECLARED_CONSTANT_SOURCES`
(`:53-59`) registers ghost's `_PENALTY_AREA_X`, `_PENALTY_AREA_Y_MIN`, `_PENALTY_AREA_Y_MAX`. Once
nothing reads them there is no free branch: **delete them and the registry entries go stale**, so
`test_no_dead_entries_in_either_list` (`:96-100`) fails unless `:53-59` is pruned in the same commit;
**keep them unread** and they are dead constants, against the gate's own "exists iff read" premise.
Prune the registry with the constants.

The pinned test's docstring ("ghost's pair is the 40.3-derived one its weights were fit on") goes
stale in this cycle — update it in the same change.

### D5 — TF-24: a local argmax check on corrected data, then Stage 2

Stage 2 consumes Stage 1's optimum via `--carrier-best`, so a stale Stage 1 silently corrupts the
Stage-2 recommendation.

**Prong 1 — orientation invariance, pre-registered.** Run `infer_ball_carrier` at the recorded
optimum (`beta=0.0, gamma=0.25`, held `tolerance_m=3.0`) over **real corpus frames**, then over an
exact point reflection, and report the fraction assigning the same carrier, plus N. This promotes
`test_carrier_inference_is_orientation_invariant` from a 40-row synthetic fixture to real data at
scale. **>= 99.9% → Stage 1 stands. Below → full Stage-1 sweep.**

**Prong 2 — a LOCAL ARGMAX CHECK, not a cross-era comparison.** An earlier draft re-scored the
recorded optimum on corrected frames and compared it to *the value Stage 1 recorded*. That is
mis-specified: §1 states every prior sweep consumed mixed-convention input, so the recorded value **is
the contaminated baseline**. The difference measures how much the geometry correction moved
*accuracy* — not whether the *argmax moved*, which is the question the prong exists to answer. No
threshold on the first quantity answers the second.

Instead: re-score the recorded optimum **and its immediate neighbours** — already present in the prior
Optuna store — **on corrected frames only**, and require the argmax to remain at the recorded point.
No cross-era comparison, no invented threshold, far cheaper than a re-sweep. If it moves, sweep.

Any scalar drift number is retained as a **reported sensitivity measurement, not a gate**. If a scalar
gate is wanted anyway, pin it to the trial-to-trial spread near the optimum **on corrected data** —
not the old study's spread, for the same contamination reason.

**Land the result as a provenanced artifact.** Otherwise the next TF-24 cycle re-derives the same
invariance number from scratch.

### D6 — DGX trains; x86 stamps; validation is local

| stage | where | why |
|---|---|---|
| extraction + both fits + TF-24 sweeps | **DGX** (aarch64) | tuning/training only |
| `scripts/stamp_feature_contracts.py` | **local x86** | ADR-050 §6 step 4 |
| validation, probes, artifact inspection | **local** | data access + validation are local |

Platform-delta baseline is **inherited** (`max_abs_delta = 0.0` across 69 features,
`docs/research/pr5_platform_atol/`, 4.74.0), so `atol=1e-6 / rtol=0` stands. Re-measure **only** if
the ghost extractor's own probe changes. Its legs confound architecture with interpreter, and `atol`
cannot transfer to the quantized xCross features.

### D7 — The corpus cache materializes `frames.parquet`

**The naive prerequisite fails, verified.** `_loader_pining.load_matches(cache_dir=...)` persists
**raw downloaded provider artifacts** under `cache_dir/{provider}/{match_id}/` (`:292-294`) — there is
no `to_parquet` anywhere in the module; frames are parsed in memory and yielded. The trainer globs
`**/frames.parquet`, falls back to flat `*.parquet`, and `sys.exit(1)`s otherwise
(`train_ghost_gk.py:289-293`). The directory *shape* coincides; the contents do not. Pointing
`--data-dir` at the pining cache finds nothing.

**Decision: the corpus pass materializes the frames as a by-product.** This makes the shared-download
rationale **true rather than asserted**, hands the trainer its input, and leaves a reusable cache so
the next cycle does not pay the same download. It is new code in the critical path and therefore a
real scope addition, taken deliberately.

> **AMENDED DURING EXECUTION (2026-08-11): shards, not a TC3 tree.** This decision originally said
> "into the TC3 layout (`{provider}/{id}/frames.parquet`)". ADR-052's gate refused a naive loop --
> *"a driver delegated to a remote box must persist each item so a crash resumes"* -- and losing 179
> matches to a crash at match 150 is exactly the 8.7-hour failure that seam exists to prevent. The
> materializer therefore adopts `for_each`, which writes `shard_root/<token>/<key>.parquet` and a
> provenance manifest.
>
> **Both consumers still work, but the paths change:** `train_ghost_gk.py:291` falls back to a flat
> `*.parquet` glob, so the trainer (and the TF-24 checker) read the **generation directory**; the
> delta driver searches `**/frames.parquet` then `**/*.parquet`, so it reads `--out`. Getting this
> wrong is silent -- a `**/frames.parquet`-only glob over a `for_each` output finds nothing and
> reports an empty corpus rather than a wrong path.

**A writer is not enough — the materialized frames need a PARITY ASSERTION against an existing TC3
parquet before the corpus run.** The trainer's established input comes from a different pipeline; if
the pining parse yields a different schema, dtype set, or frame filtering, the trainer silently fits
on different data. That is the same train/serve-skew family this entire cycle exists to close,
arriving through the fix for it — and it would land *underneath* D2, corrupting the very measurement
meant to detect trouble. Assert schema + row count + a checksum on a known match against an existing
TC3 parquet first.

**Fallback, pre-registered:** if materialization proves non-trivial in the plan — layout mismatch,
memory pressure, or a preprocessing step that cannot be reproduced faithfully — **split into two
PRs.** The shared download is the only thing binding two independent deliverables into one branch;
without it there is no coupling to defend.

---

## 4. Prerequisites

Verified: DGX reachable (aarch64, 20 cores, 119 GiB, 115 free); `PINING_FOR_THE_DATA_TOKEN`,
`HF_TOKEN`, `DATABRICKS_*`, `AWS_PROFILE` present.

Verified and **failing**, now addressed by D7: the pining cache does not present the layout the
trainer reads.

---

## 5. Commit sequence

Ordering, not commit boundaries — granularity is decided at commit-prep, on owner approval.

1. Vectorized canonical predicate + xCross migration (byte-identity proven; no re-stamp).
2. **The D7 materializer and the measurement driver** — both must be **tracked before they can
   run**: `scripts/_provenance.py` counts untracked files as dirty, so code landing alongside the
   artifact it produces would have produced that artifact from a dirty tree. The materializer is new
   code that feeds a committed artifact, so it falls under the same rule; omitting it here makes
   step 3's artifact inherit a dirty tree.
3. The delta artifact from the DGX run (after the D7 parity assertion passes).
4. Ghost: constant unification + predicate migration + **declaration migration and registry pruning
   (D4)** + re-fit (both variants) + re-stamp. **Indivisible** — the contract raises on an
   unaccompanied flip. Touches `_feature_contract.py:53-59`.
5. HF upload (both variants) + model-card corrections.
6. TF-24: Stage-1 argmax check, then Stage 2; recommendations + manifest + artifact.

---

## 6. Testing and gates

- **Durable property test:** `in_penalty_area_goal_relative_array(xs, ys)` equals
  `[in_penalty_area_goal_relative(x, y) for …]` over a boundary-dense grid. **This is the artifact
  that must outlive the cycle.** The commit-1 grid sweep is a *characterization* test against an
  expression being deleted — once xCross migrates, the thing it compared to is gone and nothing
  permanent pins the array helper to the scalar one.

  Required grid points: exactly `16.5`; `34 ± 20.16`; the ULP neighbourhoods of both bounds (§1.1
  shows bound equality is not predicate equality); `gr_x < 0`; and **`y = 13.85`, the case that
  actually separates the two forms** at the legacy constant. **NaN is a specified contract, not just
  a grid point: `NaN → False` on either argument** (verified today — the scalar returns `False` for
  both `gr_x=NaN` and `y=NaN`, since `NaN <= 16.5` is `False`). The array form must match, so the
  test pins behaviour rather than recording whatever the first implementation happens to do.
- **Declared-value test (D4):** general, keyed on the canonical name, across all three extractors —
  see D4 for why it must *replace* rather than supplement the enumeration gate's coverage of ghost.
- Existing xCross + defensive-credit suites unchanged; `test_geometry_constant_enumeration.py` green
  **after its registry is pruned** (D4).
- Ghost contract re-stamped **on x86**; `load()` verifies chirality + contract fail-closed.
- **Local CI parity — read the invocations out of `ci.yml`, never from memory.** `ci.yml` defines
  **three jobs**: `lint` (:18), `test` (:32, a matrix), `pandas-span` (:120). `--benchmark-only`
  (:92) and `--doctest-modules` (:100) are **steps inside `test`**, not jobs; `pyright` is a step
  inside `lint`. The six local commands that reproduce them, quoted from the file:

      ruff check silly_kicks/ tests/ scripts/
      ruff format --check silly_kicks/ tests/ scripts/
      pyright
      pytest tests/ -m "not e2e" --benchmark-skip --tb=short          # :88, the SUPERSET leg
      pytest tests/ -m "not e2e" --benchmark-only --tb=short          # :92
      pytest --doctest-modules silly_kicks/ --ignore-glob="*/_[!_]*.py" --tb=short   # :100

  Two details that matter and were wrong or missing once already: `ci.yml` has **two** non-benchmark
  pytest invocations — `:86` (`not e2e and not slow`, non-primary legs) and `:88` (`not e2e`, primary)
  — and the superset at `:88` is the one to run locally. And the doctest command's
  `--ignore-glob="*/_[!_]*.py"` is **load-bearing**: dropping it widens the scope from the public
  surface to every private module.
- `/final-review` before commit-prep, **including the C4 regeneration** — not optional, and not
  satisfied by the C4 gates passing (they pin the DSL; nothing reads `architecture.html`).

## 7. Risks

| Risk | Response |
|---|---|
| Delta is material and weights move | That is D2's purpose — a documented before/after, attributed to band vs boundary |
| xCross grid sweep shows a delta | Premise wrong; stop and re-derive before touching ghost |
| Materialization (D7) proves non-trivial | Pre-registered fallback: split into two PRs |
| Stage-1 argmax moves | Pre-registered: full Stage-1 sweep before Stage 2 |
| Ghost's probe changes | Re-measure the aarch64/x86 delta rather than inherit it |

## 8. Follow-ups, registered not dropped

- Answer ADR-050's parked `gr_x < 0` question — now with **one** predicate to answer it in, and with
  D2's population count already measured.
- Platform provenance in `git_provenance()` (Cycle C candidate): artifacts record commit, not machine.

## 9. Review provenance

Externally reviewed 2026-08-11 against `21d5ef4`. **F1** (prerequisite fails) → D7. **F2**
(declaration/predicate divergence invisible to key-name guards) → D4, the most consequential finding.
**F4** (mis-specified cross-era comparison) → D5 prong 2. Plus the CI-job correction in §6, the
`& ~is_ball` composition note, the durable property test, and the Status softening.

**One finding was checked and did not hold.** F3 claimed the min/max→abs representation change brings
a different float-error profile. It does not, at the canonical value. F3's decomposition point
survives and is adopted in §1.1 — with **two** contributors, not three.

**Round 2 (same day) then corrected my correction, and it was right to.** My §1.1 rebuttal argued
"provably nil" from *bound equality*, which does not establish it: at the **legacy** constant the
bounds are equally bit-identical yet the two forms disagree at exactly `y = 13.85`. The conclusion
held only because the stronger check does — the forms agree on **every double straddling both bounds
(0 / 1602)** at the canonical value. §1.1 now states the sweep, not the bounds, and the legacy
disagreement became §6's most valuable test case.

Round 2 also produced **R2-1**, the observation that D4 relocates ghost's constants out from under
the enumeration gate's mechanism, silently narrowing it to xCross — folded into D4 — plus the D7
parity assertion, the D7 materializer's place in §5, the NaN contract, and the two `ci.yml`
corrections in §6.

**Both rounds' findings were re-verified against source before adoption, and one from each round was
rejected or corrected on the evidence.** A reviewer's finding is a hypothesis, not a baseline.

## 10. Decisions still open

- **D5's sensitivity number**, if a scalar is reported alongside the argmax check — pin it in the plan
  from the neighbourhood spread on corrected data.
- **One ADR.** The ghost/predicate work is a structural decision with durable consequences (one
  predicate; the contract declaring from it; ADR-050 §6 closed) and carries the ADR. The TF-24 half
  **executes** ADR-009's standing rule rather than deciding anything — it changes no library
  behaviour, and adopting a recommendation is explicitly a separate PR. It gets a provenanced
  `docs/research/` artifact plus a dated note appended to ADR-009 recording that its recommendations
  were recomputed post-ADR-028. If D7's fallback splits the PRs, PR 1 carries the ADR and PR 2 carries
  the artifact and the note.
