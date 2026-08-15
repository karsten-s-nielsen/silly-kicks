# TF-24 recommendation honesty — the indistinguishable set, prefer-incumbent selection, and `tolerance_m` as a held constant — design

**Status:** drafted 2026-08-14; **revised after cross-review** (part-deux session) the same day — all
findings folded in, four decisions taken with the owner (§10). Pending a final owner + reviewer pass
before a plan.
Facts verified against `main @ c0eb545` and branch tip `abe9f94` on 2026-08-14; file:line citations
are to that tree.

**The ADR number and version are assigned at COMMIT-PREP, read off `main` at that moment — never
pre-claimed** ([[no-version-number-until-commit-prep]]). Next-free at writing time is
`4.81.0 / PR-S151 / ADR-060`; **confirm against `origin/main` before numbering**. This spec refers to
the new ADR as **ADR-060** as a placeholder.

**This is item C of the ghost-refit + TF-24 cycle** (branch `ghost-refit-and-tf24-refresh`). Tasks 1–9
and the Phase-1 confirmation are done and gated; this spec covers only the TF-24 harness redesign
folded in after the confirmation landed. It **supersedes** the cycle memory's "Phase 2 / item C" sketch
(which proposed a `tolerance_m` re-sweep §4 shows would measure an artifact).

**Refreshes:** TF-24 (ADR-009). Its standing rule — *TF-24 recommends and never changes library
constants* — is preserved.

---

## 1. Context

TF-24 Stage 1 calibrates ball-carrier inference (`tolerance_m`, `beta`, `gamma`) by maximizing
carrier-actor accuracy on a 3-provider fold. Two measured facts make its "report one best point"
output dishonest, and each has a distinct fix.

**`beta`/`gamma` are non-identifiable.** The Phase-1 confirmation (`scripts/check_stage1_argmax.py`,
`abe9f94`) scored the recorded optimum, the shipped default, and the nearest store neighbours across
the harness's match-stratified CV folds: the whole six-point spread is **≈ 1/40 of one CV standard
error**, three points win across five folds, `argmax_moved = False`. An argmax over indistinguishable
points recommends noise and churns run-to-run.

**`tolerance_m` is under-determined — holding it at 3.0 is load-bearing.** The shipped default's
docstring is decisive (`silly_kicks/tracking/_ball_carrier.py:358–362`): the objective's labels are
on-ball moments only, with no loose-ball negatives, so a larger radius only ever *helps* the metric
and the sweep presses `tolerance_m` to its upper bound (a re-sweep store landed `7.999`). The radius
the sweep "finds" is an artifact of the label design, not a value to apply.

**The redesign's organizing principle:** since `tolerance_m` is a held constant *by construction*, it
should not be a swept parameter, a recommendation field, or a consumed variable **anywhere** — a value
that cannot be represented cannot be wrong. `beta`/`gamma` remain the recommendation, chosen by a rule
that states what "better" means.

Three current-code problems follow:

1. **The checker softens the fence.** `check_stage1_argmax.py:29–32` frames sweeping `tolerance_m` as
   "a live question," inviting a future session to apply the artifact. The authoritative rationale is
   in `_ball_carrier.py`; the harness should state it once, correctly.
2. **The raw sweep record asserts a meaningless radius.** Stage 1 writes the winning trial's
   `tolerance_m` straight into `carrier_best.json` (`calibrate_tracking_defaults.py:344–348`), so that
   file can claim `tolerance_m ≈ 7.999` — a Hyrum's-law trap for any consumer of it.
3. **Stage 2 can consume that radius.** `calibrate_tracking_defaults.py:327–331` reads
   `{tolerance_m, beta, gamma}` from `--carrier-best` and feeds it to the held-out Brier objective.

**Forcing function.** Item C runs *before* Stage 2 (Stage 2 consumes Stage 1's selection), and every
TF-24 job is a `require_clean_tree` provenance-stamped driver, so item C's code must be committed clean
before any job runs.

---

## 2. Scope

**In:**

1. Redesign the Stage-1 confirmation to emit the **indistinguishable set** and apply a
   **prefer-incumbent** selection over `beta`/`gamma` (§3.1).
2. **`tolerance_m` becomes a held constant with zero swept, recommended, or consumed representation:**
   removed from `stage1_config`'s search space and `CarrierAccuracyObjective` (a sweep no longer
   produces it); absent from both carrier artifacts; Stage 2 sources it from `DEFAULT_CARRIER_PARAMS`
   (§3.5).
3. A **shared, direction-agnostic noise-floor primitive** (`exceeds_noise_floor`) used by the
   selection's significance test and `tf25_gate_fires` (§3.2).
4. A **standing fold-stability diagnostic** with an explicit "no discriminating evidence" verdict (§3.4).
5. **Correct the `tolerance_m` framing** in the checker to the under-determination fence.
6. **Provenance for the selection artifact** — stamped, committed, validated by Stage 2 (§3.5, §4).
7. **ADR-060.**

**Out:**

- **No `tolerance_m` re-sweep** and **no loose-ball-negatives objective work** — the only honest way to
  calibrate the radius, a separate and larger project; **not pursued, not tracked** (owner, 2026-08-14).
- **No fresh Optuna sweep.** Item C reuses the existing store's `beta`/`gamma` trials.
- **No library default-constant change.** TF-24 recommends only (ADR-009); `DEFAULT_CARRIER_PARAMS` is
  untouched.

**Note (non-blocking) on the search-space removal.** Removing `tolerance_m` from the search space
changes what a *future* sweep produces (it would optimise `beta`/`gamma` at the held radius). Item C
runs no fresh sweep, and its recommendation is robust to this regardless — prefer-incumbent keeps the
shipped default, and `beta`/`gamma` are non-identifiable — so the existing store is reused as-is and
the config change takes effect only on the next real sweep. The existing store's trials swept
`tolerance_m`; the confirmation already neutralises that by holding `3.0`, so the removal *aligns*
future stores with what the confirmation already does rather than diverging from them. Chesterton's
answer: swept since TF-24's first commit (`0a76f52`, 3.28.0) on the TF-5 "calibrate all three" intent,
which the sweep's own under-determination result invalidated.

---

## 3. Architecture

Pure decision logic in the library; I/O, streaming, and provenance in the script (ADR-009).

### 3.1 `silly_kicks/calibration/_selection.py` (new) — the pure decision rule

```python
@dataclass(frozen=True)
class PointScore:
    label: str
    params: dict            # {"beta": ..., "gamma": ...} — tolerance_m is NOT here (§3.5)
    per_fold: tuple[float, ...]   # accuracy per CV fold, aligned by fold index across points
    mean: float

@dataclass(frozen=True)
class Selection:
    selected: PointScore
    incumbent: PointScore
    moved: bool
    reason: str
    best_candidate: PointScore | None
    effect_size: float | None     # best_candidate.mean - incumbent.mean
    paired_se: float | None       # SE of the per-fold (candidate - incumbent) difference

def select_recommended_point(
    *, incumbent: PointScore, candidates: list[PointScore],
    min_effect_size: float, policy: str = "prefer_incumbent",
) -> Selection: ...
```

- **Two explicit bars, both required to move** (§10 decision 3). A candidate replaces the incumbent
  only if `gain = candidate.mean - incumbent.mean` clears **both**:
  - a **practical-significance** floor: `gain > min_effect_size` (strict — an exact tie keeps the
    incumbent, matching prefer-incumbent conservatism); and
  - a **statistical-significance** test: `exceeds_noise_floor(gain, paired_se)`, where `paired_se =
    cv_standard_error([c - i for c, i in zip(candidate.per_fold, incumbent.per_fold)])` — the SE of the
    **per-fold difference**, correct because all points are scored on the *same* folds
    (`check_stage1_argmax.py:259–268`).
  Among candidates clearing both, the best `gain` wins; otherwise the incumbent holds.
- The bars answer different questions ("is it big enough to matter?" vs "is it real?"). For a
  non-identifiable parameter the effect-size floor is what prevents churn on a statistically-significant
  but trivial gain — a paired significance test *alone* would re-introduce the churn prefer-incumbent
  exists to prevent, and the marginal SE alone conflates the two questions. See ADR-060 for the full
  rationale; both bars matter because this is a **standing** diagnostic that reruns as data/K grow.
- `policy="prefer_incumbent"` is the only implementation; the `policy=` string-dispatch door matches
  the house pattern (e.g. `ExpectedThreat(method=...)`) and **raises on an unknown policy** (tested).
- **`incumbent` is the shipped default** (`beta=0, gamma=0.25`), pulled *out* of the scored pool; the
  candidate set is `{recorded_optimum} ∪ neighbours`. Today both shipped and recorded are scored as
  points (`check_stage1_argmax.py:343`); the redesign must extract the incumbent so it is never
  compared against itself (§7 finding, made concrete).
- `min_effect_size` (δ) is a **justified, frozen** constant, not plucked: derived from the smallest
  carrier-accuracy difference that produces a detectable Stage-2 Brier change, set conservatively, then
  recorded with its corpus + rationale and never re-derived per run — δ itself must not become the
  churning knob the design removes. Derivation + a robustness check are a §7 pre-land item.
- **The pairing is defended, not trusted.** `select_recommended_point` **raises** on
  `len(candidate.per_fold) != len(incumbent.per_fold)` — the paired SE is valid only if the two
  fold-vectors align, which `score_points_by_cv_fold` guarantees by construction
  (`check_stage1_argmax.py:259–268`) but the pure function must not assume silently.

### 3.2 `silly_kicks/calibration/_diagnostics.py` — the shared noise floor

`tf25_gate_fires` (`:20–31`) and the checker's `moved_beyond_noise` (`check_stage1_argmax.py:93–108`)
are the same `gain > se` test with a NaN-SE guard. Extract the tail:

```python
def exceeds_noise_floor(gain: float, se: float) -> bool:
    """True iff `se` is finite and `gain > se`. A NaN/None/inf SE never clears the floor."""
```

`tf25_gate_fires` refactors to `exceeds_noise_floor(global_brier - provider_best_brier, provider_se)`;
the selection calls it with the **paired** SE. `_diagnostics.py` is the home (not `_gates.py`, which
holds the unrelated H1/signal-sanity objective gates — verified 2026-08-14).

**inf-handling is unified, and that is called out.** `tf25_gate_fires` currently guards with
`math.isnan` (which *permits* `inf`); `exceeds_noise_floor` uses `np.isfinite` (which *rejects* it).
The observable verdict is identical (`gain > inf` is `False`, and `isnan(inf)` is `False` so the old
code also returns `False`), but this is a behavior-adjacent change to a **gate** function, so a
RED-first test pins `tf25_gate_fires`'s verdict unchanged across `{finite, nan, inf}` SE after the
refactor (§6).

### 3.3 `scripts/check_stage1_argmax.py` — orchestration only

Keeps all I/O, `for_each` streaming, CV scoring, and provenance. Changes:

- Pull the incumbent (shipped default) out of the scored pool (§3.1).
- After scoring, call `select_recommended_point` and write the selection artifact via a pure
  `build_selection_artifact(selection, *, provenance: dict) -> dict` (extracted so the artifact's shape
  — the `{beta, gamma}` payload, the *absence* of `tolerance_m` (§3.5), and the presence of the
  provenance keys — is unit-tested off `main`, §6). The builder stays pure: the orchestrator does the
  `git_provenance()` I/O and passes the dict in, so `run_commit`/`run_tree_dirty` reach the artifact the
  structural gate (§4) requires without the builder reading git.
- Emit the fold-stability diagnostic (§3.4) into `metrics.json`.
- `argmax_moved` remains as a reported diagnostic (a distinct, honest question) but is no longer the
  output; the `Selection` is.
- **Correct the module docstring** (`:29–32`) to the under-determination fence, cross-referencing
  `_ball_carrier.py`.

### 3.4 The fold-stability diagnostic

A `metrics.json` record: per point, the per-fold ranks and the between-fold vs between-point variance
ratio, plus a verdict — `"no_discriminating_evidence"` when no candidate clears **both** bars (today's
state), and a non-that verdict when one does. The verdict derives from the same
`select_recommended_point` result, so diagnostic and recommendation cannot disagree.

### 3.5 `tolerance_m` as a held constant — three enforcement points

- **Not swept.** Removed from `stage1_config`'s `param_space` and `warm_start`, and
  `CarrierAccuracyObjective` defaults it to `DEFAULT_CARRIER_PARAMS["tolerance_m"]` when a candidate
  omits it (`_carrier_objective.py:186` → `p.get(...)`). The confirmation still passes `3.0`
  explicitly, so that path is unchanged. **The Stage-1 `carrier_best.json` writer must change with it:**
  `calibrate_tracking_defaults.py:345` iterates `("tolerance_m", "beta", "gamma")` and would `KeyError`
  on the next sweep once the winning trial has no radius — it becomes `("beta", "gamma")`, so
  `carrier_best.json` is naturally `{beta, gamma}`.
- **Not in any recommendation artifact.** `carrier_selected.json` carries `{beta, gamma}` (+ provenance,
  §4) and **no `tolerance_m`**. A field that does not exist cannot carry a wrong value — the strongest
  form of the guard (unrepresentable > refused > silently-corrected).
- **Sourced from the constant by consumers.** Stage 2's validation changes from `{tolerance_m, beta,
  gamma}` to `{beta, gamma}`, and it builds `carrier_params = {"tolerance_m":
  DEFAULT_CARRIER_PARAMS["tolerance_m"], **selected}`. `DEFAULT_CARRIER_PARAMS` is the single source of
  truth (and `infer_ball_carrier`'s own default is already `3.0`).

---

## 4. Data flow, artifacts, provenance

```
Stage-1 sweep (future runs) ──────► carrier_best.json  { "beta", "gamma" }   [tolerance_m no longer swept]
                                        │
redesigned check_stage1_argmax.py ◄─────┘  reads {beta, gamma}; tolerance_m held = 3.0
   │  scores set on CV folds; incumbent = shipped default pulled out
   │  select_recommended_point(incumbent, candidates={recorded ∪ neighbours}, min_effect_size=δ)
   ├─► docs/research/tf24_stage1_confirmation/metrics.json  (set + fold-stability + Selection + run_commit/dirty)
   └─► docs/research/tf24_stage1_confirmation/carrier_selected.json
                                     { "beta": <sel>, "gamma": <sel>, "run_commit", "run_tree_dirty" }   [NO tolerance_m]
                                        │
Stage 2 (calibrate_tracking_defaults.py --stage 2 --carrier-best carrier_selected.json)
        validates {beta, gamma} + provenance (missing == dirty → REFUSE);
        carrier_params = {"tolerance_m": DEFAULT_CARRIER_PARAMS["tolerance_m"], **selected}
```

- **`carrier_selected.json` is a committed, provenanced artifact** under
  `docs/research/tf24_stage1_confirmation/` — it is the recommendation of record (ADR-009 cites the
  TF-24 manifest in a later apply-PR, so it must trace to a clean commit). It stamps
  `run_commit`/`run_tree_dirty` (the checker is already a provenance driver,
  `check_stage1_argmax.py:442–443`).
- **Stage 2 validates the input's provenance** — a missing manifest counts as dirty and Stage 2
  refuses — consistent with the corpus-driver family's "provenance on both, or the clean downstream
  SHA launders the dirty upstream" rule (`run_signoff_power` precedent). This is a new, small Stage-2
  change alongside the `{beta, gamma}` read.
- **ADR-056 output-provenance gate is STRUCTURAL** (`test_artifact_provenance_output.py` globs
  `docs/research/**/*.json`, verified 2026-08-14), so `carrier_selected.json` **and** `metrics.json`
  auto-enrol: both MUST carry top-level `run_commit` + `run_tree_dirty: False` in the JSON itself (not a
  sibling manifest, not a nested shape), or CI fails. No manual registry edit. The `_input_contract` /
  `declare_inputs` mechanism is for covariate-based artifacts (extractor + `GEOMETRY_VERSION` digests)
  and does not apply to a `{beta, gamma}` recommendation.
- **The Phase-1 confirmation artifact folds in:** the redesigned checker produces
  `docs/research/tf24_stage1_confirmation/`; there is no separate landing of the old-shape artifact.
  The `abe9f94` `argmax_moved=False` result is preserved in git history and cited in ADR-060.

---

## 5. Provenance sequencing (the order the owner flagged)

1. Implement item C on the branch. Gate locally — ruff/pyright/pytest at CI scope.
2. **2-match smoke** (local; a 1-match smoke is invalid — `match_cv_splits` does LOMO ≤7 and raises on
   one group).
3. **Commit.** DGX syncs to that exact clean commit (fresh clone; no scp'd hot-fixes — drivers refuse a
   dirty tree, so a fix means re-commit + re-run).
4. Run the redesigned confirmation on the DGX → `metrics.json` + `carrier_selected.json`, `dirty:false`.
5. **Land** those committed artifacts (landing re-runs nothing; the stamp cites the producing commit).
6. **Stage 2** on `carrier_selected.json`, clean commit, corpus held to the prior 25 matches.
7. Gates → release.

---

## 6. Testing (RED-first, both sides, non-vacuity)

`select_recommended_point`:
- incumbent **kept** when no candidate clears both bars; incumbent **replaced** when a candidate clears
  both — both sides.
- the **tie** (candidate exactly at the effect-size floor, or exactly at the paired SE → kept, strict `>`).
- **two candidates both clear → the best `gain` wins**, and `Selection` records which.
- **`recorded_optimum` clears vs a neighbour clears** — disambiguate, since `recorded_optimum` is a
  candidate distinct from the shipped incumbent.
- **effect-size floor is load-bearing:** a candidate that clears the paired SE but *not* δ is **kept**
  (guards against churn on a statistically-significant, practically-trivial gain).
- **`paired_se == 0` (constant non-zero per-fold difference) is the airtight δ-mandatory case:** the
  statistical bar is trivially cleared (`exceeds_noise_floor(gain>0, 0.0)` is True, 0 is finite), so a
  candidate with `0 < gain < δ` is **kept**. This is the regime that actually holds on current data
  (§7 / cross-review B).
- **fold-length mismatch raises** (`len(candidate.per_fold) != len(incumbent.per_fold)`).
- **unknown policy raises.**

`exceeds_noise_floor`: NaN/None/inf SE → False; a gain just above and just below the floor.

`tf25_gate_fires` refactor: verdict **unchanged across `{finite, nan, inf}`** SE (RED-first).

`build_selection_artifact(selection, *, provenance)`: the dict carries `{beta, gamma}` and **no
`tolerance_m`** key even when the input scored points were constructed with a stray radius, and carries
the passed-in `run_commit`/`run_tree_dirty` (pure — provenance is an argument, never read internally).
**Other side:** an artifact stamped `run_tree_dirty: True` is one the §4 output-provenance gate rejects
— a non-vacuity check that the `dirty:false` sequencing (§5) rests on a live gate, not an assumption.

Stage 2: **refuses** a `carrier_selected.json` with a missing/dirty manifest; builds `tolerance_m` from
`DEFAULT_CARRIER_PARAMS`, not the file.

`carrier_best.json` writer: produces `{beta, gamma}` and does **not** `KeyError` when the winning
trial's params omit `tolerance_m` (RED-first — the current writer indexes `params["tolerance_m"]`).

Fold-stability diagnostic: a **non-vacuity** assertion that the verdict flips to
non-`"no_discriminating_evidence"` on a discriminating (wide-separation, low-SE) fold set — not only
that it reports the current verdict on current data.

Framing-only edits (the docstring correction) move **no numbers** — existing behavioural gates stay
green.

**Script-level e2e:** the selection + artifact path is pure enough to exercise on a synthetic `summary`
dict with **no frames** — one committed script-level test, so the new glue is not covered only by the
manual `--max-matches 2` smoke.

---

## 7. Verifications required before landing (named, not deferred indefinitely)

- **Store reconciliation — must be RESOLVED before ADR-060 moves from Proposed.** `stage1_config` lists
  `tolerance_m` (`_spaces.py:41`, 1–8) while the checker asserts "only beta/gamma vary in the store"
  (`:29`). The ADR's headline evidence (`argmax_moved=False`, the 1/40-SE spread) is a property of the
  store the confirmation actually read, and interpreting the reused `beta`/`gamma` neighbours depends on
  whether that store swept `tolerance_m` or held it. Confirm the store's trials on the DGX.
- **Derive AND de-risk δ (`min_effect_size`).** Derive it from the smallest carrier-accuracy difference
  that produces a detectable Stage-2 Brier change (itself corpus-dependent — the xT-bandwidth optimum
  was corpus-*size*-dependent), set conservatively, and **freeze** it with its corpus + rationale.
  Then **assert the landed keep-incumbent result is invariant to δ across a plausible range**
  `[δ_lo, δ_hi]`, so the result does not hinge on δ's exact value even though the derivation is noisy.
  **On current data δ is the binding bar** — because `beta`/`gamma` barely move the metric,
  `paired_se ≈ 0` and the statistical test is near-decorative today; it earns its keep only in the
  future large-but-noisy regime a standing diagnostic must survive. The ADR states this plainly.

---

## 8. ADR-060

Records: (a) the recommendation is a *set + prefer-incumbent* over `beta`/`gamma`, not an argmax; (b)
the selection requires **both** an effect-size floor and a paired-SE significance test, and *why*
(non-identifiability makes statistical significance alone the wrong bar); (c) `tolerance_m` is a held
constant enforced at three points (**not swept** — removed from `stage1_config` +
`CarrierAccuracyObjective` + the `carrier_best.json` writer; excluded from the artifact; sourced from
the constant by Stage 2); (d) the standing fold-stability diagnostic; (e) the new committed,
provenanced, Stage-2-validated selection artifact.
**Proposed** until the plan lands, §7 resolves, and cross-review clears.

---

## 9. Self-review

- **Placeholders:** none. ADR number and version unbound (assigned at commit-prep).
- **Scope:** one plan's worth — a new library module, one extracted primitive, one script redesign, a
  small Stage-2 change, the `carrier_best.json` writer `{beta, gamma}` edit, and the `stage1_config` +
  `CarrierAccuracyObjective` source removal, ADR-060.
- **Internal consistency:** selection, fold-stability verdict, and the significance test all route
  through `select_recommended_point` / `exceeds_noise_floor`, so they cannot disagree. The anti-soft-
  guardrail thesis (§1) is now honoured by making `tolerance_m` *unrepresentable* in the artifact
  (§3.5), not merely asserted — the inconsistency the cross-review caught is closed.
- **Ambiguity resolved:** "incumbent" = the shipped default, pulled out of the pool (§3.1); the two
  bars are named and separately tested (§6).
- **Open items are pre-land, not open-ended:** §7 gates the ADR's move to Accepted.

---

## 10. Decisions taken (owner, 2026-08-14)

1. **Selection policy = prefer-incumbent.**
2. **`tolerance_m` held at 3.0; framing corrected; no re-sweep; loose-ball-negatives not pursued.**
3. **Stage-2 guard → exclude `tolerance_m` from the handoff** (unrepresentable, not asserted).
4. **Upstream sweep → full source removal in item C** (breaking changes are not a concern, and the
   change is safe: item C's recommendation does not depend on it — it reuses the existing store and
   prefer-incumbent keeps the shipped default — so no fresh sweep is needed to land it). Supersedes the
   earlier writer-pin + tracked-follow-up split.
5. **SE → effect-size floor + paired-difference SE, both required; marginal SE dropped.**
6. **Selection artifact → stamped, committed, Stage-2-validated, population-gate-declared.**
