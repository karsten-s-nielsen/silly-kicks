# ADR-051 cycle close-out + research-artifact validity — design

**Status:** rev 3, 2026-08-04. Rev 1 was reviewed in two unmerged passes per §7.2 (boundary +
detail); rev 2 corrected §1's evidence base and was reviewed again as a single detail-and-consistency
pass. Rev 3 closes that review's seven findings: the commit table could not execute §4.9's own rule
(N1), §1.3's invariance argument was single-axis (N2), K2 rested on a false premise and hid a larger
hole (N3), the `tf19_signoff_power` rebuild had a schedule but no owner (N4), PR 7's acceptance
omitted the ghost model card (N5), nothing gated on the §8 decision so shipping would have made it
implicitly (N6), and one §1.6 marker overstated a derived figure as directly measured (N7). Supersedes the Task 14/15 sections of
`docs/superpowers/plans/2026-08-02-pr5-chiral-goal-relative-transform.md`.

**Goal.** Close the ADR-051/ADR-028 orientation defect class completely, and turn the
"is this recorded number still valid?" question from archaeology into a mechanism.

**Why this spec exists.** PR 5 was specified, planned and reviewed five times, and shipped two
commits before anyone noticed that its own change invalidated three research artifacts. The spec drew
a blast radius around *trained artifacts* — who loads the weights, which stamps break — and every
review inherited that boundary rather than reconstructing it.

**Why rev 2 exists.** Rev 1 then repeated the error one level down. It asserted a caller sweep of
**three** symbols was the check that "would have caught all of this"; commit `6e3a132` changed
**four** functions, and four of the boundary review's nine findings are invisible to any caller sweep.
Rev 1 also mis-stated the cross arm's covariate block by generalizing from a grep instead of
resolving the value. Both failures are recorded in §1.9 rather than quietly fixed, because the
pattern — adjacent evidence substituted for the executed thing — is the one this cycle is about.

**Provenance marks on every measurement below:** ✅ = executed in the authoring session;
⚪ = measured by the reviewer, not independently re-run; ⛔ = **not measured**, with an owner named.

---

## 1. Measurements this spec rests on

**1.1 The change has TWO geometry axes, not one.** Commit `6e3a132` modified
`_geometry.py` (adding `to_goal_relative_y`) **and** `_xcross_attempt.py` (re-anchoring the
`_dominant_region_area` grid). The second is not a function of `goal_x`.

*Axis A — chirality.* 804 real frames from the committed slim fixtures, emulating the pre-fix
transform by disabling `to_goal_relative_y` ✅:

| covariate | max abs delta, `goal_x=105` | `goal_x=0` |
|---|---|---|
| `theta` | **2.620** rad | 0 |
| `GK_theta` | **0.786** rad | 0 |
| `GK_r` | **exactly 0** (see §1.3) | 0 |
| `r`, `openGoal`, `DefDist_0`, `DefDist_1`, `speed` | <= 1.4e-14 | 0 |
| `gk_depth_x` = `GK_r * cos(GK_theta)` | **exactly 0** | **exactly 0** |

Only bearings move, and only at the high-x goal, because `to_goal_relative_y` is the identity at
`goal_x=0`.

*Axis B — the grid re-anchor.* Verified mechanism ✅: at `res=3.0` the old y-centres were
`arange(1.5, 68.0, 3.0)` → 23 cells spanning 1.50–67.50, **centred on 34.50**; `_grid_centres` gives
1.00–67.00, **centred on 34.00**. The x axis is byte-identical (105 divides evenly by 3). Every cell
moves 0.5 m in y, at **both** goal ends. Reviewer measurement on the xCross fixture's own 16 frames at
`goal_x=0.0` ⚪: `space_controlled` max abs **26.61** (13.33% max, 2.03% mean, 14 of 16 rows); all 15
other features exactly 0.

**⛔ The full cross-arm covariate table under BOTH axes is not measured.** Axis A's table above is
xS-column-only and was produced by disabling one function rather than by comparing against
`6e3a132~1`. Owner: §4.5's driver, whose first job is to produce the correct table — measured against
the parent commit, covering `PAPER_CONFOUNDERS` and the real `GK_BLOCK` (§1.2). Nothing in this spec
may cite the axis-A table as complete.

**1.2 Which configs carry which covariates.** Corrected — rev 1 stated all three configs use
`("GK_r","GK_theta")`. Measured by resolving the config objects ✅:

```
GK_BLOCK (opportunities.py:47, the module default)
  = ['gk_r','gk_theta','gk_lateral_offset','gk_dist_near_post','gk_dist_far_post','gk_carrier_side']
shot_arm_config.gk_block = ('GK_r','GK_theta')      # explicit override, :139
layer2_config.gk_block   = ('GK_r','GK_theta')      # explicit override, :198
```

`validate_xcross_causal.py:23` imports `GK_BLOCK` and `PAPER_CONFOUNDERS` directly, so **the cross
arm uses the six lowercase xCross names** — including `gk_lateral_offset`, one of the three xCross
features PR 5's own measurement recorded as flipping under chirality. `theta` is in
`SHOT_ARM_CONFOUNDERS` and `LAYER2_CONFOUNDERS`; `space_controlled` is in `PAPER_CONFOUNDERS`
(`validate_xcross_causal.py:80`), so the cross arm is exposed to **both** axes.

The rev-1 error is instructive: line 139 carries the comment *"xS GK names — xcross gk_* don't exist
in xS features"*, which states the correct answer, in output that was on screen. Two overrides were
generalized to three configs without resolving the default.

**1.3 The Layer-2 treatment is invariant, and that is what decides `tf19_signoff_power`.**
Treatment is `gk_depth_x >= 16.5` via `_covariate_depth` (`opportunities.py:393`). Invariance is
**structural, not measurement luck** ✅: `gk_r = hypot(gkx, gky - GOAL_Y)` and
`hypot(a, -b) == hypot(a, b)` bit-exactly, as `cos(-t) == cos(t)` does — so `GK_r` is exactly
invariant and the `1.4e-14` in §1.1 belongs to another group member. Therefore `n_spells`,
`n_treated`, prevalence 0.0041 and the degenerate-replicate counts are unchanged, and those produce
`N_MIN_MATCHED: None` (max power 0.055 against a required 0.80, 0/200 estimable at n>=4000).
**Verdict invariant; decimals not** — see §4.5, which is narrower than rev 1 implied.

*This is an **axis-A** argument, and §1.1 establishes two axes.* It is nevertheless complete: `_xshot_occurrence.py` contains **0** references to `_grid_centres` / `_dominant_region_area` ✅ — the grid lives entirely in `_xcross_attempt.py`, so no xS covariate can touch axis B, and `gk_depth_x` is built from xS columns. Every other invariance claim in this spec must likewise be two-axis or state why it need not be.

**1.4 The re-gate verdict is a pure function** ✅.
`regate_verdict(*, arm: str, probe_verdict: str, entanglement: str) -> str` (`_model_eval.py:714`)
takes three string tokens. A changed entanglement token is recomputed in seconds; it does not force a
probe re-run. This is why the entanglement re-run may safely follow the probe.

**1.5 The provenance gate cannot detect an omission** ✅. `tests/scripts/test_provenance_wiring.py:108`
asserts `len(ARTIFACT_DRIVERS) >= 6`; the tuple holds **14**. Eight drivers could be deleted with the
gate green. `validate_xcross_causal` is absent from it, has no `_provenance` or `--allow-dirty`
reference, and its artifact carries neither `run_commit` nor `run_tree_dirty`.

**1.6 The xCross liveness gate is blind, and the proof is already planted.**
`test_xcross_bundled_model_is_live_not_degenerate` asserts `roc_auc_score(...) >= 0.9`. Measured ✅:
**1.0000** on **16 rows** where **7 features are constant** — the entire GK block (`gk_r`=2.0,
`gk_theta`=0.0, `gk_lateral_offset`=0.0, `gk_carrier_side`=-0.0) plus an all-NaN `score_differential`
— and only **50%** of rows satisfy the model's own `wide_area_only` domain (**derived**, not direct: the frozen fixture carries only the 16 features plus `label`, no raw ball coordinates ✅, so `_in_wide_area` cannot be called on it; the figure reconstructs `dx`/`dy` from the `ball_r` / `ball_theta` polar pair). **Consequence for §4.6:** a precondition test asserting in-domain CANNOT be written against the frozen features alone — the regenerated fixture must carry the raw ball coordinates or an explicit `in_domain` column, or the precondition is unfalsifiable. Root cause of the blindness is construction: the generator is fully synthetic and hard-pins the keeper,
`_r(10, _DEF_TEAM_ID, 2.0, 34.0, is_gk=True)` at `make_xcross_directional_fixture.py:63`.

Executed rather than argued ⚪: overwriting the GK block with `0`, `99` and `NaN` each leaves
**AUC = 1.0000**. Combined with §1.1's axis-B result — a 13.33% `space_controlled` change producing
`max|dp| = 0.000000` — the fixture is inert in **8+ of 16** features. §4.6 requires each replacement
"proven by planting the defect it exists to catch"; this is that proof, already planted.

**1.7 Platform tolerance — xShot only** ✅. `docs/research/pr5_platform_atol/` records max abs
aarch64-vs-x86 delta **0.000e+00** across 27 features. Its own README says *"one probe, one
extractor"*: `extract_xcross_features` was never platform-compared. Rev 1's "xS/xCross extractors"
was an overstatement. The two legs also confound architecture with interpreter
(x86/Windows/3.14.2 vs aarch64/Linux/3.12.3). The directory is itself unprovenanced — see §2 item 18.

**1.8 What a caller sweep does and does not see.** Enumerating callers of the three named symbols
across `silly_kicks/` + `scripts/` returns five call sites ✅ — `causal/opportunities.py`,
`tracking/_model_eval.py`, the two fixture makers, and the `tracking/__init__` re-export.

Rev 1 called this "the caller sweep that would have caught all of this." **It would not have.** A
sweep keyed on a named symbol is blind to:

- **a fourth changed function** — the sweep was of three symbols; `_grid_centres` is the fourth (§1.1);
- **wrapper functions** — `derive_opengoal_range.py:84` imports `prepare_xshot_training_data`, which
  wraps `extract_xshot_features`; it is in `ARTIFACT_DRIVERS`, writes the TF-19 sign-off S6 headroom
  threshold, and appears in no sweep result ✅. (Its answer is "unaffected", but it was never
  *classified*, which is what §7.1 demands.)
- **committed data artifacts** — a stale fixture is a caller of nothing (§1.1 axis B);
- **numbers recorded in prose** — §2 items 17 and 18;
- **the second hop** — the sweep returns `causal/opportunities.py` as one line and says nothing about
  the three research artifacts downstream. §1.2 and §1.3's hand-tracing found those, not the sweep.

The sweep is necessary and cheap. It is not sufficient, and §7.1 is worded accordingly.

**1.9 Three probes are registered, not two** ✅. `_model_eval.py:793/794-6/815` register `xcross`,
`xs` and `xs_v2`. `_xcross_wrapper` routes to `_xcross_eval.py::gk_substitution_probe` — a fifth
module in the dependency chain. Rev 1's §1.8 classified `_model_eval.py` as "covered — both probes
re-run" and named only the xS probe.

---

## 2. Inventory

Twenty open items. Nothing outside this list is in the cycle; anything discovered later is a failure
of this spec and is recorded as such (§11.4). Items 16–20 were added in rev 2 by the boundary review.

**This grouping is topical, not an ownership assignment.** Ownership is fixed in §3 and nowhere else.

**A — geometry correction**
1. PR 5 commit 3: research dirs, docs, version
2. xS probe -> `tf19_pr3b_xs_v2`, `tf19_pr3b`
3. HF upload of the xShot `sc_extended` artifact
4. PR 6 — D3 re-key, 8 aggregators, remaining Gate B xfails
5. PR 7 — ghost-GK box constant re-fit + ghost-extractor `atol` check
6. TF-24 — calibration refresh

**B — artifact validity**
7. `tf19_entanglement` + `xcross_causal` re-runs
8. `tf19_signoff_power` invariance measurement
9. Research-artifact input contracts
10. `ARTIFACT_DRIVERS` completeness gate
11. `validate_xcross_causal` provenance wiring
12. Caller sweep as a stated rule
13. xCross liveness gate + fixture

**D — added in rev 2**
16. **xCross frozen fixture regeneration** — stale in `space_controlled` since `ec543cc` (§1.1 axis B)
17. **xCross probe re-run + `tf19_pr2` refresh** — the re-run already happened and is recorded only
    as CHANGELOG prose, with no artifact, no `run_commit`, no `run_tree_dirty`
18. **`pr5_platform_atol` provenancing** — a hand-run artifact created by this PR, load-bearing for
    ADR-050's tolerance and for PR 7's acceptance baseline
19. **ADR-051 amendment** — it still carries the open deferral PR 5 closes, and CHANGELOG:13 asserts
    "§8b is recorded in ADR-051" when ADR-051 contains zero occurrences of "8b" (§1 verified)
20. **Hub artifacts + model cards** — the published `sc_extended` variants of *both* models, and
    `docs/huggingface/model-cards/*`, which document the chirality/geometry stamps

21. **`tf19_signoff_power` rebuild** — once, after PR 7; three accumulated invalidations (§3.1)

22. **Gate the taxonomy decision at the next `train_gk_completion` run** — where the label is
    actually written (§8); nothing carries a gate there today

23. **Repair the GS input-convention guard** — its enforcement tier and its test fixture both
    point away from the only data that triggers it (§5.6)

**C — registered, out of scope (see §8)**
14. GS corpus taxonomy — **but see §8, this is no longer cleanly deferrable**
15. Fold-count ship rule with no magnitude floor

---

## 3. Ordering and boundaries

**PR 5 -> Cycle B -> PR 6 -> PR 7 -> TF-24.**

Cycle B is second because PR 6 and PR 7 both invalidate research artifacts too — PR 6 moves away-team
geometry across eight aggregators; PR 7 re-fits ghost-GK, which feeds the GKDV arm values and
therefore `tf19_signoff_power`'s ICC leg. With B second, those enumerate mechanically and become the
mechanism's first real test. With B last, the hand-tracing that produced this spec happens twice more.

| | owns | explicitly does not own |
|---|---|---|
| PR 5 | items 1, 2, **3**, 7, 8, 11, 12, 13, **16**, **17**, **18**, **19**, **20** | the contract mechanism; the registry completeness gate; any aggregator re-key |
| Cycle B | items 9, 10, **23**; the invalidation report; **the `tf19_signoff_power` rebuild policy (§3.1)** | re-running anything PR 6/7 will invalidate |
| PR 6 | item 4 | ghost weights |
| PR 7 | item 5 | calibration defaults |
| TF-24 | item 6 | library default constants (ADR-009 standing rule) |
| **Next `train_gk_completion` run** | **item 22** — enforce the §8 taxonomy answer where the label is written | deciding the answer; that is item 14's re-registration |
| **Post-PR-7 rebuild** | **item 21** — rebuild `tf19_signoff_power` once, against Cycle B's invalidation report, after PR 7's ghost re-fit lands | re-deriving the arm-values table, which PR 5 does not invalidate |

Rev 1 omitted item 3 from this table while §4.7 assigned it to PR 5 — the exact ambiguity §2's
ownership rule exists to prevent. Fixed above.

**3.1 `tf19_signoff_power` accumulates three invalidations and gets an explicit owner.** Verified
chain ✅: `causal/_confounders.py:142,147` calls `compute_defensive_line`, and
`add_defensive_line` is one of PR 6's eight Gate B targets; `LAYER2_CONFOUNDERS` includes
`defensive_line_height` and `defensive_line_compactness`. (`add_pressure_on_actor` is **not** among
the eight, so `pressure_on_actor__bekkers_pi` is safe — stated precisely because rev 1 would have
over-claimed.)

| source | mechanism |
|---|---|
| PR 5 | `theta` moves — a build-time spell column, not only an analysis input |
| PR 6 | `add_defensive_line` re-key -> `defensive_line_height` / `_compactness` |
| PR 7 | ghost re-fit -> the ICC leg |

Rev 1 left this unowned: PR 6's acceptance was "Gate B green", PR 7's was "the artifact loads", and
Cycle B explicitly disclaims re-running what PR 6/7 invalidate. **Assignment.** Cycle B owns the *policy* — the artifact declares its inputs, so the accumulated
invalidation is enumerated rather than remembered. But **a schedule names a time, not an owner**, and
"after PR 7" sits outside every row of the table above — the same ambiguity §2's ownership rule
exists to prevent. The rebuild is therefore **item 21**, a named unit of work with its own row,
executed once after PR 7 rather than three times.

**PR 5 takes the minimal provenance repair (item 11), not the mechanism.** Without wiring, re-running
`validate_xcross_causal` produces a second uncitable artifact. The two-directional completeness
assertion stays in B.

---

## 4. PR 5 — remainder

Branch `adr051-pr5-chiral-transform`, 2 commits ahead of `origin/main`.

**4.0 Commit structure — corrected.** Rev 1 said "three commits; commit 3 waits for the re-runs."
That cannot execute. `scripts/_provenance.py:73-74` counts **untracked** files as dirty and §9 forbids
`--allow-dirty` for shipped artifacts, so **new driver code must be committed before it can produce a
clean artifact**. Both §4.2's wiring and §4.5's driver are new code that gate later steps. The
structure is therefore **four commits**:

| commit | contents | why it must precede the next |
|---|---|---|
| 1 (`6e3a132`) | transform + grid, code half | landed |
| 2 (`08ce9a8`) | weights + stamps | landed |
| **3 (new)** | §4.2 wiring, §4.5 driver, **§4.10 platform-`atol` driver**, §4.6 gate + fixture, §4.8 rule + CLAUDE.md, spec, ADR-051 amendment | every driver must be committed, and the tree fully clean, before any run |
| **4** | probe + re-run outputs, **driver-produced `pr5_platform_atol/`**, research dirs, CHANGELOG, TODO, version | consumes commit 3's drivers |

**The tree must be clean at the END of commit 3, not merely committed.** Rev 2 listed the spec in
commit 3 but not `docs/research/pr5_platform_atol/`, which is untracked — so §4.3's driver would still
`SystemExit`, and §4.0's own table defeated §4.9's rule. Working-tree assignment, stated exhaustively
so nothing is left dangling: `CLAUDE.md` -> commit 3; `CHANGELOG.md`, `TODO.md`, `pyproject.toml`,
`silly_kicks/__init__.py`, `uv.lock` -> commit 4.

**`pr5_platform_atol/` is REMOVED from the repo before commit 3, not committed and then restamped.**
Item 18 is circular as rev 2 wrote it — provenancing needs a driver, the driver needs a clean tree,
and the tree is dirty *because* the directory is untracked. Committing the hand-run files and later
adding a `run_commit` is precisely the restamp §4.5 forbids. The sequence is therefore:

1. Move the hand-run `dgx.json` / `x86.json` / `README.md` **out of the repo** (scratchpad). They were
   a scratch measurement that should never have been written to `docs/research/`.
2. Commit 3 ships the driver alone. Tree clean.
3. Commit 4: run the driver on **both** platforms — it emits one self-provenanced probe JSON per
   platform, each carrying its own `run_commit`, plus a comparison step — and commit that output.
   This requires the DGX to pull commit 3 first; the round trip is real and is scheduled, not assumed.

Until step 3 completes, **§1.7's number is unbacked** and is marked accordingly.

A new `scripts/` driver must also be enrolled in `ARTIFACT_DRIVERS` and clear all four per-driver
assertions in `test_provenance_wiring.py` — commit 3, not deferred to Cycle B. This applies to both
new drivers (§4.5's and §4.10's).

**4.1 xS probe** (running) -> `tf19_pr3b_xs_v2`, `tf19_pr3b`. Verdict recorded whichever way it lands.

**4.2 `validate_xcross_causal` provenance wiring.** Import `scripts/_provenance.py`, offer
`--allow-dirty`, call `require_clean_tree(git_provenance(), ...)` **from `main()`** — per the ADR-037
rule that the CLI refuses and `run()` records the truth — stamp `run_commit` + `run_tree_dirty`, add
the driver to `ARTIFACT_DRIVERS`.

**4.3 `tf19_entanglement` re-run.** 179 matches / 98,789 opportunities. If the token moves, recompute
the probe verdict via `regate_verdict` (§1.4) rather than re-running the probe.

**4.4 `xcross_causal` re-run.** 23,966 opportunities; first provenanced version of that artifact.

**4.5 `tf19_signoff_power` — record, do not rebuild, and be precise about what is recorded.**
The stored invariance is of the **treatment** (`gk_depth_x`), which licenses "verdict unchanged"
(§1.3). It says nothing about the artifact's reported decimals, which do move. The directory must
therefore mark which fields are verdict-bearing and which are knowingly stale — an unmarked
`metrics.json` reads as wholly current.

**The stale-input argument belongs in the spec, because it strengthens the decision:** `theta` is in
`LAYER2_BUILD_CONFOUNDERS` (`opportunities.py:150`), so it is a **per-spell stored column**. The
persisted spells parquet that `run_signoff_power --spells` consumes is stale in `theta` from now on.
Re-running only the analysis leg (seconds) would not fix the decimals — it would **launder** them.
Rev 1's "the changed `theta` enters only the propensity matching" read as if the artifact on disk were
untouched; it is not.

The measurement ships as a provenanced `scripts/` driver over committed fixtures, emitting the
corrected §1.1 table — **both axes, measured against `6e3a132~1`, covering `PAPER_CONFOUNDERS` and
the real `GK_BLOCK`**. Rev 1 promised this driver as "the reusable instrument for the same question in
PR 6, PR 7 and B"; an instrument built on a half-emulated baseline and an xS-only column set would
under-report every reuse.

**Two limits on that billing, recorded here rather than discovered at reuse.** (a) Four of the nine `LAYER2_CONFOUNDERS` — `defensive_line_height`, `defensive_line_compactness`, `pressure_on_actor__bekkers_pi`, `time_remaining_s` — are **join-time** columns (`causal/_confounders.py`), present in neither extractor's feature names ✅, so an extractor-keyed driver cannot measure them. Two of those four are **PR 6's own mechanism** (§3.1), so the driver must emit them as `not-measurable-by-this-driver` rather than omit them, or the instrument silently under-reports exactly where it is next needed. (b) The design is **covariate-keyed**, not model-feature-keyed: `ball_theta` moves 2.836 rad under axis A but is a model input rather than a causal covariate, so it is correctly outside §1.1's ⛔ scope and equally outside this driver's output. Reuse for a model-feature question needs a different key. **`run_commit` is never restamped without a re-run.**

**4.6 xCross gate + fixture.** Retire-and-replace per §1.6: a **precondition** test (all rows in
`wide_area_only` domain, no constant feature among the model's own inputs, >= 40 rows), a **liveness**
test (finite, in `[0,1]`, non-constant), and a **responds-to-geometry** test sweeping GK position. The
generator is repaired to vary the keeper, vary `score_differential`, restrict to in-domain rows, and
emit enough rows that ties cannot flip the gate.

*Threshold shape, derived from measurement rather than chosen* ⚪ (bundled `default`, 16 fixture rows,
one GK feature swept at a time; base rate mean p = 0.0098):

```
gk_r 0.00280   gk_theta 0.00468   gk_lateral_offset 0.00064   gk_carrier_side 0.00757
joint far/wide keeper: 0.0098 -> 0.0068  (delta -0.00301)
```

The model does respond. Three constraints follow: the threshold must be **relative, not absolute**
(any absolute bar above ~0.008 is unreachable at a ~1% base rate); it must be **GK-block-joint, not
per-feature** (`gk_lateral_offset` alone is near-inert at ~6.5% relative); and **no monotonicity
assertion** (`gk_r` and `gk_carrier_side` are single-split step functions). Measured on the degenerate
fixture §4.6 replaces, so this bounds the assertion's *shape*, not its final constant — re-measure on
the repaired fixture before fixing the number.

*Scope note — corrected.* Rev 1 claimed this item touches "no artifact … no blast radius." **False:**
the fixture is a committed artifact, stale in `space_controlled` since `ec543cc` (§1.1 axis B), and
regenerating it is item 16 — a cycle obligation, not a by-product of gate quality. The item is still
correctly in PR 5, but on the accurate ground: leaving it means the gate keeps reporting a passing
liveness claim on evidence that cannot support one, while a committed fixture holds feature #3 on a
grid no shipped model was fit on.

**4.7 HF upload (item 3) and the Hub artifacts (item 20).** Upload the xShot `sc_extended` artifact.
Then verify the *published* `sc_extended` variants of both models: `load()`'s `geometry_version` prong
only warns, but the feature-contract prong is fail-closed and the probe frame is unchanged, so a
pre-PR-5 Hub artifact takes the fingerprint-comparison branch and raises `IntegrityError` —
`XCrossAttemptModel.from_hub()` is public API. Both model cards under `docs/huggingface/model-cards/`
document the chirality/geometry stamps and must be updated. `docs/` was outside §1.8's search scope,
which is exactly the blindness §7.1 now names.

**4.8 Caller-sweep rule** into CLAUDE.md (§7.1), worded to include what it cannot see.

**4.9 Docs, CHANGELOG, version.** Version assigned at commit-prep after
`git fetch && git merge origin/main`, written to all five sites. **CHANGELOG:13 is factually wrong**
("§8b is recorded in ADR-051") and must be fixed alongside item 19's amendment. Note that the spec and
`pr5_platform_atol/` must be **committed before** §4.3/§4.4 run, not at commit-prep — untracked files
make every driver `SystemExit`.

**4.10 Platform-`atol` driver (item 18).** A provenanced `scripts/` driver replacing the hand-run
`pr5_platform_atol/`. It emits **one probe JSON per platform**, each self-provenanced with its own
`run_commit`, plus a comparison step producing the summary. One invocation per platform, because a
single machine cannot measure a cross-platform delta — which is why §4.0 schedules a DGX round trip
after commit 3 rather than assuming one. Enrolled in `ARTIFACT_DRIVERS`; clears all four per-driver
assertions. Until it has run on both platforms, §1.7's `0.000e+00` is an unbacked number and PR 7's
acceptance baseline does not exist.

---

## 5. Cycle B — research-artifact input contracts

Own ADR (number assigned at commit-prep). Unchanged from rev 1; the review attacked §5 and it held.

**5.1 Mechanism — declared input contract per artifact.** Each driver declares what its numbers
depend on — extractor identity, covariate list, `geometry_version` — written into its own
`metrics.json` alongside a digest. CI re-derives the declaration from live code and flags any artifact
whose declared surface no longer matches. Mirrors ADR-050's `_feature_contract`.

Rejected alternatives:

- **Runtime input fingerprint only** (`ruthless.fingerprint` over `token_inputs`). Impossible to
  under-declare by hand, but a wrong declaration digests stably — `for_each`'s recorded blind spot —
  and it reports *that* something changed, never *what*. Retained as a tamper-check, not the primary.
- **Central reverse index.** Simplest, and the only one answering "what does this PR invalidate?"
  directly, but it stores knowledge away from the artifact and rots the moment a driver is added
  without touching it — exactly how `ARTIFACT_DRIVERS` reached 14-with-a-floor-of-6 (§1.5).

**5.2 Invalidation is computed, never maintained.** Derived by comparing declarations against the
diff. No stored list to fall out of date.

**5.3 Warn, do not raise.** An artifact is not a serving path. It must surface at PR time, not read
time.

**5.4 Registry completeness gate.** Replace `assert len(ARTIFACT_DRIVERS) >= 6` with a
two-directional assertion pinning the tuple to the real driver population. **Lands red**, observed
failing, *before* the tuple is corrected.

**5.5 The standing rule.** Re-run when a changed input can move the **verdict**; store a **measured**
invariance when it can only move decimals; never restamp `run_commit` without re-running. The middle
clause is safe only because the invariance is stored as a number in the artifact — and, per §4.5,
only when it is an invariance of the quantity the verdict actually rests on.

**5.6 Repair the GS input-convention guard (item 23).**

**Placement argument — sequencing, not genus.** Cycle B runs BEFORE PR 6, and PR 6 re-keys away-team
geometry across eight aggregators. A convention guard that cannot see the data it governs should be
repaired before the PR that moves geometry for precisely the team-half the convention concerns. (An
earlier revision argued from defect genus -- "a gate whose theory of coverage has a hole, like §5.4
and K9." That is true and does no work: at that abstraction nearly any test gap qualifies, and §5.1's
mechanism -- per-artifact declared input contracts with CI re-derivation -- shares no machinery with a
converter runtime guard. The genus reading was post-hoc.)

**The reason it belongs in THIS cycle specifically.** The fixture must be shaped from real GS data,
which is owner-tier, so only *statistics* can travel -- unlike the `tests/datasets/sportec/idsse_slice/`
precedent, which commits a reduced slice of real data with a `SOURCE_SHA`. That means the distribution
the fixture is shaped to **must itself be a committed, provenanced measurement**, or the fixture's
shape rests on an unrecorded number. That is exactly the failure mode this cycle exists to fix.

**Measured facts** (K6 carries the corroborating field measurement):

- `validate_input_convention(on_mismatch=None)` resolves to **`"raise"`** when
  `SILLY_KICKS_ASSERT_INVARIANTS=1` (`orientation.py:89`), otherwise `"warn"` ✅.
- `.github/workflows/ci.yml:58` sets it to `"1"` ✅ — **CI is configured to raise**.
- The committed GS fixture is `tests/datasets/gradientsports/synthetic_match.json` ✅ — synthetic.
  Real GS data is owner-tier and flows only on the pining/DGX path, where the variable is unset.
- Detector bands (`orientation.py:289-292`, `:311-321`) ✅: `>=10` shots per (match, team, period)
  group -> `high`; 5-9 -> `medium`; otherwise `ambiguous` with `convention=None`; `low` is reserved
  for **fewer than two reliable groups**.
- **The fixture defers on the GROUP-COUNT clause, not on shot count** ⚪: its shot rows fall in
  `(team 100, period 1) -> 10` and `(team 100, period 2) -> 1`. The first is AT the `high` threshold,
  so it is not in the defer band at all; **only one team has shots**, so there are fewer than two
  reliable groups. An earlier revision said the fixture "almost certainly sits in the defer-silently
  band" and prescribed raising per-group shot counts -- that claim carried no measurement marker while
  every other fact here did, and it targeted the wrong parameter. Raising shot counts will not make CI
  see the case; **a second team (or second period) needs a reliable group.**

**So the enforcement is inverted:** hard-fail where the condition cannot arise, soft-warn where it
does. True regardless of which side of the disagreement is right, so it is repaired first.

**Order of work — measure, then shape, then diagnose.**

| | |
|---|---|
| **2a** | Measure the real per-group shot **and group** distribution on owner-tier GS (DGX; only summary counts travel, never coordinates). Commit it as a provenanced artifact -- see above. |
| **1** | Shape the fixture to 2a's numbers, so CI can see the case. |
| **2b** | Diagnose, and let the diagnosis choose the side: a thin sample means the detector over-claims and the fix is detector-side, without losing power where its tiers are currently correct; genuinely all-high-x groups mean the GS *event* input really is possession-perspective and something downstream compensates to produce the output K6 measured correct -- a larger finding, to be traced rather than silenced. |
| **3** | Prove the guard can still fire: plant a genuinely mis-declared provider fixture and observe it fail. The failure mode to avoid is repairing the symptom by weakening the detector, which converts a working guard into a decorative one. |
| **4** | Any GS exemption goes in a registry with a justification and a dedicated test, per the `STRUCTURAL_CONSTANTS` idiom -- never `filterwarnings`, never a per-provider `silent`. The justification cites K6's measurement, not K6's conclusion. |

An earlier revision put step 1 before the measurement, which cannot execute without guessing -- and
the guess in the text was half wrong (see the group-count finding above).

**Scope note.** This widens Cycle B from two items to three. Coherent widening -- but widening, and
recorded as such rather than absorbed.

---

## 6. PR 6 and PR 7 — scope and acceptance only

Specified now, planned when each prerequisite validates.

**PR 6 — D3 re-key.** Re-key the identity-gated direction checks across the eight Gate B aggregators
(`add_cover_shadows`, `add_defensive_line`, `add_gk_influence`, `add_line_break`,
`add_off_ball_context`, `add_packing`, `add_player_influence`, `add_structural_pass`);
`_gk_influence.py:371-372` is the named example, keyed on `same_id(attacking_team_id, home_team_id)`
and therefore correct only while frames are home-attacks-right. **Acceptance:** Gate B green across
the registry with no xfail markers remaining; Gate A unchanged — and a clean Gate A reading is *not*
evidence about identity-keying, since it is structurally blind to it. **Plus:** report
`tf19_signoff_power`'s invalidation to §3.1's schedule.

**PR 7 — ghost-GK box constant.** Flip `_ghost_gk`'s 40.3 to the canonical `spadlconfig` value,
migrate onto `in_penalty_area_goal_relative` including the strict-to-non-strict boundary, re-extract,
re-fit, measure the ghost-extractor platform `atol` **before** re-stamping, re-stamp on x86.
**Acceptance:** the artifact loads under chirality and feature-contract enforcement; the ghost `atol`
recorded in a provenanced artifact (§1.7's directory having been fixed by item 18); no unaccompanied
constant flip; **and `docs/huggingface/model-cards/ghost-gk-v1-model-card.md` updated** — item 20 covers only PR 5's two models, and a documented-stamp artifact going stale with nothing covering it is the exact class this spec exists to address.

**TF-24.** Recommendation refresh on corrected geometry. ADR-009's standing rule holds: the harness
recommends, it never changes library constants.

---

## 7. Process changes

**7.1 Caller sweep — necessary, not sufficient (PR 5).** Any spec that changes a public seam must
mechanically enumerate every caller of **every changed function** and classify each as affected or
unaffected **with evidence on both sides**. The rule must state its own blind spots, because rev 1
did not and was wrong within a day: a symbol sweep cannot see wrapper functions, committed data
artifacts, numbers recorded in prose, or the second hop (§1.8). So the sweep is the *floor*, and the
spec must additionally enumerate: every function the diff touches; every committed fixture derived
from them; every research artifact downstream of a changed module; and every number recorded outside
a driver-produced file. Cycle B makes the first and third checkable.

**7.2 Boundary review.** At least one review round is scoped explicitly to the frame, handed the
caller sweep as the thing to **attack** rather than trust, and asked only: *what is outside the
boundary the author drew?* Not merged with a detail review. **Validated:** rev 1's two-pass review
produced nine boundary findings, four of which (a fourth changed function, a third registered probe,
an unprovenanced artifact created by this PR, and unloadable Hub artifacts) are structurally
invisible to a detail pass — a detail pass checks what the document says, and these are all things
the document did not say.

**7.3 A design that must be BUILT cannot be settled by reading it.** Recorded because it cost this
cycle two revisions. Round 4 recommended sweeping both goal ends and adding a chirality mirror-pair
gate, on the ground that the fixture was blind to the defect class PR 5 exists to close. That
recommendation survived a round of reading and was **refuted the moment someone built it**: a
committed table provably cannot carry chirality evidence at any tolerance -- a forged half is
bit-identical to a real extraction on integer coordinates -- while `test_pr5_chirality_gates.py`
already does it live, on both extractors, on both axes, with a permanent plant, in 0.21 s. The correct
resolution ran in the OPPOSITE direction from the proposed fix: not "make the fixture two-ended" but
"stop asking a committed fixture to carry evidence it cannot hold." Rule: when a review round proposes
a fixture or gate design, the round that evaluates it BUILDS it. Reading rounds can rank designs; only
a build can refute one.

---

## 8. Registered out of scope — with one correction

**Item 15 (fold-count ship rule)** is deferrable as written: `fixed_sequence_ship`'s `clears_rule`
ships on "positive in >= K-1 of K folds AND a positive mean" with no minimum effect size. Its verdict
is *recorded, not obeyed* for bundling (commit 2 shipped the public arm per the PR-S118 precedent), so
nothing in PR 5 depends on re-registering it. It does decide item 3's HF payload — noted, not blocking.

**Item 14 (GS corpus taxonomy) is NOT cleanly deferrable, and rev 1 was wrong to say it was.**
Rev 1 stated *"neither can ride inside a PR that consumes it."* Commit `6e3a132` already added
`_corpus_taxonomy()` to `scripts/train_gk_completion.py`, calling `artifact_label`, which returns
`"public"` if all-public else `"full"`. The GS matches are declared private — the very declaration
item 14 exists to re-register. The commit message states the consequence: *"The bundled GS variant
will now be labelled `full`; owner decision 2026-08-02 to ship that."*

So PR 5 ships **the code path**, and the commit message's *"will now be labelled `full`"* is
forward-looking. Measured ✅: no shipped artifact carries the label, because `train_gk_completion.py`
was not run in this PR. Rev 1's B4 correctly found the code path; rev 2 and rev 3 then escalated it
to "PR 5 ships a label", which is false — an escalation across two revisions with no measurement
under it, the same shape as §1.2's error.

**What remains true and what follows.** The rule is still un-re-registered, and the code that
consumes it is PR 5's. So the question is asked at commit-3 time (a "provisional" answer is a
commit-3 code edit) and the ENFORCEMENT is registered at the next `train_gk_completion` run — item
22 — rather than pretending PR 5 decides it. The spec does not choose the answer; it puts the gate
where the label is actually written.

*Residual, not a gate:* `train_xcross_attempt.py:623` uses `artifact_label` for VARIANT SELECTION,
so the taxonomy rule may have influenced which weights PR 5 bundles even though no label string was
written. That is item 15's territory, which this section already flags as deciding item 3's HF
payload.

---

## 9. Acceptance criteria

**PR 5.** Full suite 0 failed; ruff, `ruff format --check` and pyright clean at CI scope
(`silly_kicks/ tests/ scripts/`, never `.`); all three bundled models load under **both** chirality
and feature-contract enforcement; every re-run research directory carries a `run_commit` it earned;
no number cited in CHANGELOG or a research directory that is not backed by a driver-produced
artifact; `tf19_signoff_power` marks its stale fields; C4 count unchanged.

**The §8 item-14 question is commit-3-timed; it is NOT a tag gate.** Rev 3 made it block
commit 4 and the tag. That was an over-correction, and the measurement refutes its premise:
**zero** weight artifacts under `silly_kicks/` carry `artifact_label` or `all_public` ✅, and
`_corpus_taxonomy` exists only in `scripts/train_gk_completion.py`, which PR 5 never runs. PR 5
therefore ships the CODE that will emit the label, not the label. Putting an owner round trip on
the critical path of a PR that emits nothing would be ceremony. What is real: if the answer is
"provisional", that is an edit to `train_gk_completion.py` — **commit-3 code** — so the question
must be ASKED before commit 3 is pushed even though the answer blocks nothing. The decision goes
live at the next `train_gk_completion` run (PR 7, TF-24, or any re-materialization), and **that**
is where the gate belongs — registered as item 22.

**Cycle B.** §5.4's assertion observed failing before the repair. The mechanism exercised against a
real change, not a hypothetical.

**All gates in this cycle** land red, assert the failing side as well as the passing one, carry a
non-vacuity assertion that the computation actually happened, and pin any registry to its real
surface in both directions.

**Verdicts may flip.** If the entanglement re-run moves the token, `xcross_causal` reverses, or the xS
probe lands differently, it is recorded plainly and surfaced. A flipped verdict is a result, not a
failure of the PR, and is never quietly reconciled with existing text.

---

## 10. Known, registered, not addressed in this cycle

§11.4 says anything discovered later is a failure of this document. That rule only works if things
already known are written down. These came from the rev-1 review's "noted, not investigated" list;
each is verified here to the depth stated and assigned no owner beyond registration.

| # | Finding | Verified | Disposition |
|---|---|---|---|
| K1 | `xshot_occurrence_a0/_a1/_a2` and `xcross_attempt_a0/_a1/_a2` are undocumented in `feature_glossary.py` — the base columns `xshot_occurrence` / `xcross_attempt` ARE documented, the per-slot variants are not | ✅ measured | Pre-existing ADR-048 sweep gap, not caused by this cycle. Register; do not fix inside PR 5. |
| K2 | **The ADR-020 dup-`action_id` gate has no atomic coverage, and its meta-assertion is a tautology.** `tests/tracking/test_frame_aware_xfns_dup_action_id.py` contains **0** references to `atomic` ✅ — all 15 atomic `*_xfns` mirrors sit outside the gate. Its meta-assertion compares `set(_XFNS_NAMES)` (`:132`) against the identical expression that built it (`:128`) ✅, so it is always true — plus a `>= 21` floor. Structurally the `len(ARTIFACT_DRIVERS) >= 6` anti-pattern §5.4 exists to correct. | ✅ measured | **Rewritten in rev 3.** Rev 2's K2 hypothesised an `__all__` export omission hiding a function from a gate. That premise is FALSE: the gate enumerates with `dir()`, not `__all__`, so the omission hides nothing from it. The export asymmetry is real but cosmetic *for this gate* (`xshot_occurrence_xfns` absent from atomic's `__all__`, `xcross_attempt_xfns` present; **both** absent from `tracking.features.__all__`) and matters only for the ADR-033 surface. Registered on the corrected finding — closing it as a tidy-up would have lost the actual hole. |
| K3 | `docs/research/tf19_pr3b/metrics.json` carries neither `run_commit` nor `run_tree_dirty` — a fourth unprovenanced artifact beyond §1.5's third | ✅ measured | Self-heals: §2 item 2's probe runs `--variant both`, which rewrites this directory. Stated so the heal is intentional rather than lucky. |
| K4 | `git_provenance` treats "git unavailable" as `dirty=True` / `tree_state="unknown"` | ✅ read | Correct fail-closed behaviour. Latent, not live — the DGX checkout is a real git repo. Relevant only if §4.5's driver is ever run from a non-git copy. |
| K5 | `docs/research/tf19_pr2/hf_upload_instructions.md` states each `*-v1` repo serves exactly one Hub variant at its root | ⚪ relayed | Check against items 3 and 20's upload plan **before** uploading, not after. Folded into §4.7 as a precondition. |
| K6 | **RESOLVED 2026-08-04 — cosmetic.** The GS converter warns `declared=per_period_absolute but detector inferred possession_perspective` on the pining loader path. **Measured** on 2 GS matches / 2,742 linked actions: action-vs-reprojected-frame-ball median `|dy|` = **2.75 / 2.79 m**, against a calibrated PR-S95 scale of ~0.2 m (correct) and ~11.8 m (y-inverted). | ✅ measured | **The consequential branch is falsified by the ABSENCE of a split, not by the aggregate.** Had the events been possession-perspective while the converter applied a per-period flip, period 2 would be mirrored against period 1 — a tens-of-metres asymmetry. Measured period 1 vs 2: **2.81 vs 2.71**; `flip=True` vs `flip=False`: **2.84 vs 2.64**; home vs away: **2.52 vs 2.89**. Four independent splits agree to ~1 m with no bimodality. The ~2.5 m residual is **event-to-tracking sync noise** (annotated event coordinates vs the ball at the linked frame; at ~5.5 m/s a half-second link offset is 2.75 m), not a convention error — the 11.8 m reference was a systematic `|68-2y|` mirror, a different shape entirely. **PR 6 is not blocked.** Caveats worth keeping: n=2, one corpus (WC2022 GS); and this establishes actions and frames AGREE, which would also hold if both were mirrored identically — an argument against that is `team_attacking_direction` is 100% non-null and the native GS adapter carries the ADR-035 geometric backstop, so a shared error would have to survive both. **The guard itself is repaired in Cycle B as item 23 (§5.6)** — not by silencing the warning, but because CI is wired to RAISE on this mismatch while holding only a synthetic GS fixture, so the strongest tier runs where the condition cannot arise. |
| K7 | **`validate_xcross_causal` carries its own unmeasured train/serve platform axis.** It re-extracts xCross features on **aarch64** (`build_opportunities` -> `extract_xcross_features`) while the model was fit and stamped on x86. | ⚪ relayed | Independent of the feature-contract question — the contract guards the *stamped vector*, not a corpus re-extraction. Task 4's xCross leg will bound it; if that leg is clean the axis is closed by implication, and the artifact should say so rather than leave it unstated. |
| K8 | **Enrolling three new drivers WIDENS the `ARTIFACT_DRIVERS` hole.** `test_the_driver_list_is_not_silently_empty_or_stale:108` asserts `>= 6` against **14** entries ✅; PR 5 takes it to 17, so the floor becomes even weaker relative to the population. | ✅ measured | Cycle B §5.4 owns the fix. Recorded so the widening is a known consequence of PR 5 rather than a surprise when B measures the gap. |
| K9 | **Bundled `metadata.json` carries `training_commit: null`** for both `_xshot_weights/default` and `_xcross_weights/default`, despite their trainers being in `ARTIFACT_DRIVERS` since 4.72.0. | ⚪ relayed | The provenance gate checks a driver's **source** (does it import the helper, call it from `main()`), never its **output**. A driver can satisfy every assertion and still emit an unprovenanced artifact. That is a real gap in the gate's theory of coverage and belongs in Cycle B's contract work, not in PR 5. |

---

## 11. Risks

1. **Serialized DGX.** PR 5's re-runs queue behind the probe; entanglement is the larger corpus and
   likely exceeds it. Roughly a day of wall clock. Accepted — the alternative leaves a known-stale
   artifact on `main`.
2. **A re-run may not reproduce.** The causal drivers are `for_each`-sharded and resumable, but a
   changed covariate could surface an unrelated latent defect. A finding, not a blocker.
3. **The contract can be under-declared** — §5.1's inherited weakness. Mitigated only by a declaration
   being human-readable and reviewable, where a digest is not.
4. **This spec can be incomplete in the same way rev 1 was.** It already was, once, within a day.
   §7.1 and §7.2 make that less likely, not impossible. Anything discovered outside §2 is recorded as
   a failure of this document rather than absorbed silently. **Recorded so far:** rev 1's
   three-symbol sweep (four functions), its §1.2 generalization-from-grep, its omission of item 3
   from §3, its "no blast radius" claim, and its assertion that item 14 was deferrable.
