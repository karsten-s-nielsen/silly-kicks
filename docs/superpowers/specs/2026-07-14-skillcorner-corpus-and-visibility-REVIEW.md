# Critical review — `2026-07-14-skillcorner-corpus-and-visibility-design.md`

**Reviewer:** external session · **Date:** 2026-07-14 · **Verdict: do not start the owner runs yet.**

Everything below was checked against the code and, where possible, against real SkillCorner data
(the MIT open set, cloned locally — the same 10 public matches the spec's public arm is built on).
Claims I could not verify are labelled as such.

**What is right, and worth saying first.** The §3.2 compliance control is the right control, in the
right place, for the right reason: keying the public arm on `visibility` rather than provider name,
defaulting to `private`, deleting `_PUBLIC_PROVIDERS` rather than bypassing it, and pinning
`PUBLIC_CORPUS` to the known 17. The registration-before-fitting discipline is real. Finding #3
(`is_detected`) I independently reproduced: on three public matches, **GK detection is 18.1%,
outfield 60.9%** (spec: 19.6% / 66.6% — the small gap is the different match subset). It is a true
and load-bearing finding. §1.6's pitch defect is real: `tracking/skillcorner.py:146-147,170-171`
applies a flat `+52.5 / +34.0` with no pitch input.

The problems are concentrated in the registered protocol (§4) — which is the part that cannot be
fixed after the fact.

---

## BLOCKERS

### B1 · §4.3 + §4.5 — the ghost-GK admission gate cannot detect the failure it exists to prevent

This is the most serious finding.

- Ghost-GK's **target is the keeper's position** (`train_ghost_gk.py:340-341`, `y_test["gk_x"/"gk_y"]`).
- On SkillCorner, **~80% of keeper positions are extrapolated** (§1.5; I measured 81.9% on the
  public matches).
- The expansion takes SkillCorner from **10 of 81 matches (~12%) to 108 of 179 (~60%)** of the
  ghost-GK corpus. So the share of ghost-GK's *training target* that is interpolator output goes
  from roughly 10% to roughly **half**.
- §4.5 bans the 98 from GKDV *measurement* using exactly this argument — *"a GK-substitution
  statistic computed there would substantially measure the interpolator"* — and then permits them in
  ghost-GK *training*, waving the objection off with *"ghost-GK's exposure is controlled by §4.3"*.
- **But §4.3's control is held-out-keeper MAE computed on those same extrapolated targets.** The
  gate and the corruption share a source. A model that gets better at predicting SkillCorner's
  interpolator and worse at predicting real keepers **passes**.

There is no detection filter anywhere in `train_ghost_gk.py` — I checked.

**The fix is elegant and already in this PR.** §3.3 surfaces `visibility` on the research path for
the first time. Use it: compute the §4.3 admission MAE **on detected-keeper frames only**, and
report detected/extrapolated MAE side by side. Note the spec *already* gates xt_gk keeper **origins**
on detection (§6, `_tracking_gk_xy_detected`). Gating ghost-GK keeper **targets** the same way is the
same rule applied consistently. As written, the spec is strict about detection in one place and
blind to it in the other — for the same data, in the same PR.

### B2 · §4.3 — the 0.05 m band is not a measurement

You asked whether 0.05 m is defensible or chosen to be passable. **Neither: it was chosen without a
variance calculation, and it is unresolvable in both directions.**

The same acceptance block allows `cross_fold_std < 0.5 m` (`train_ghost_gk.py:537`). At a cross-fold
std of even 0.3–0.5 m over 5 folds, the standard error of one run's mean MAE is ~0.13–0.22 m — and
§4.3 compares **two independently-run point estimates** (81-match vs 179-match), so the standard
error of the *difference* is ~0.2–0.3 m. Judging that against a **0.05 m** band is a coin flip. Under
a true zero degradation you would fail this gate a large fraction of the time; a real 0.3 m
degradation would pass it often.

**Fix — and you already own the right method.** Pair the folds: same keeper groups, same fold
assignment, same seed, baseline and expanded. Then judge the **per-fold paired ΔMAE** with the same
sign-consistency logic §4.1 uses for the paired test. Paired differences cancel the fold-to-fold
variance that dominates here. The spec applies exactly this reasoning to xS/xCross two sections
earlier and then abandons it for ghost-GK.

Secondary: the band is justified as *"~4% of the ~1.1 m incumbent MAE"*. That 1.07 m is the
**match-grouped** figure (4.14.0 model card, served boosted mean — I confirmed it is the *current*
served number, not the retired ≈1.1 m). §4.3 runs **keeper-grouped**, a strictly harder task on a
different scale. Anchor the band to the keeper-grouped baseline once you have measured it, or make
it relative.

### B3 · §3.2 — "fail-closed" is defeated by the cache-hit predicate, and `_PUBLIC_PROVIDERS` has six sites, not two

**(a) The stale cache.** Both trainers gate the cache on the *existence of one file*:

```
train_xshot_occurrence.py:251   if (cache / "features.parquet").exists():
train_xcross_attempt.py:313     if (cache / "features.parquet").exists():
```

§3.2 says a pre-existing `_feature_cache/` "is invalid and must be treated as a miss" — but specifies
**no mechanism**, and the predicate above will happily hit. This is not hypothetical: the
**2026-07-13/14 owner runs have already populated that cache on the DGX.** Add a schema-version
sentinel to the cache directory and check it in the hit predicate.

**(b) The second seam.** `_PUBLIC_PROVIDERS` appears at **six** sites:

```
train_xshot_occurrence.py:30, 276, 313
train_xcross_attempt.py:30, 344, 398
```

§3.2's code sketch replaces only the `is_public` computation (`:276`, `:344`). It never mentions
`:313` / `:398` — `if provset <= _PUBLIC_PROVIDERS: shipped = "public"` — which sets the **shipped
artifact's label**. Now trace the `sc_extended` run: its providers are `{skillcorner, idsse}` with no
Gradient Sports, so `two_candidate` is **False** (`:277` requires `"gradientsports" in provset`), the
`else` branch runs, and `provset <= _PUBLIC_PROVIDERS` is **True** → the model trained on **98
restricted matches is labelled `"public"`.**

That is the licensing landmine of §1-finding-2, alive in the one place §3.2 does not look. Deleting
the constant does force the issue — but the spec must name this site, and §6 must have a test that
fails if the *label* is name-keyed. (This is your own second-producer / pair-rule pattern.)

### B4 · §6 — the native-vs-kloppy parity test is a guard that cannot fail

§6: *"On a 104 m match they **differ**, and the native (scaled) one is asserted correct against the
goal-line identity."*

Measured (below): with the pitch fix **deleted**, native vs kloppy differs by **0.50 m** on a 104 m
match. With the fix, they differ by **0.19 m**. A test that asserts only *"they differ"* **passes
either way** — it cannot distinguish the fixed builder from the broken one. It is the only test
standing between the re-route and a geometry regression, and it is inert.

---

## MAJOR

### M1 · §3.3 / §3.4 / §6 contradict each other — and I measured which is right

kloppy **does** scale. Its SkillCorner source coordinate system is a `NormalizedPitchDimensions`
carrying the **true** `pitch_length` (104 for match 1886347), which it normalizes to [0,1] and then
maps to the standardized 0–105 metric target. So §1.6 and the §3.3 table are correct that the
research path is currently unaffected — and **§6's expectation that native and kloppy will "differ"
on a 104 m match after the fix is simply wrong.** If the fix is right, they should *agree*.

But they do not agree — and this is the part the spec never checks:

| match | pitch | max \|kloppy − **spec §3.4**\| | max \|kloppy − native **today**\| |
|---|---|---|---|
| 1925299 | 105 × 68 | **0.000 m** | 0.000 m |
| 1886347 | 104 × 68 | **0.194 m** | 0.500 m |
| 2013725 | 106 × 68 | **0.190 m** | 0.530 m |

kloppy's effective scale on the 104 m pitch is **105/103.71**, not **105/104**. The spec's formula
`(x + L/2) · 105/L` is *not* what kloppy computes. **One of the two is wrong, and the spec asserts
kloppy is "correct" (§1.6) without ever testing it** — while proposing a formula that disagrees
with it.

Note the §3.4 gate cannot adjudicate this: *"the goal line must land at exactly 0 and 105"* is
satisfied **by construction** by both formulas. Establish ground truth against an independent
landmark (penalty spot at 11 m, penalty-area edge at 16.5 m) on a non-105 pitch **before** routing
the research path onto the native builder. This is a ~20-minute check and it gates §3.3.

Consequence for §4.2: the re-route moves research-path geometry by **≤0.19 m**, not by the 0.5–2.0 m
in §1.6's table (that table is native-vs-truth, not native-vs-kloppy). §4.2's "pitch scaling" is
therefore a real but *small* pipeline change, and §1.6's "the research corpus is currently
unaffected" is right. The two sections should be reconciled so Stage A's attribution is honest.

### M2 · §4.2 — the confound you have not isolated is **ghost-GK**

You asked what else is confounded. Stage A re-baselines **xS and xCross only**. But §3.3 changes the
SkillCorner frames, ghost-GK **trains on SkillCorner frames**, and §4.3's baseline/expanded pair
never states which pipeline each run uses. If the baseline is the archived 81-match number and the
expanded run uses the new pipeline, **the ghost-GK gate confounds precisely what Stage A exists to
prevent** — pipeline change with corpus change — for the one model whose target the pipeline change
touches most.

One sentence fixes it: *both* ghost-GK runs are fresh, under the new pipeline. As written, a careful
implementer could reasonably do the wrong thing.

Second, smaller: **`ball_z` becomes available for SkillCorner only.** If Gradient Sports and IDSSE
carry no `z`, then after this change `z`-missingness **identifies the provider** inside the feature
matrix — a leakage channel the paired test cannot see, because it evaluates on public folds. Worth
one check: is `z` provider-correlated post-change, and does either model split on it?

### M3 · §4.1 — the ordering is legitimate; the cost accounting is not

**The fixed sequence itself is correctly invoked.** A pre-specified order with stop-at-first-failure
genuinely controls FWER at the single-test level. No objection there, and the a-priori ordering
argument (same-product, no domain shift, §1.2 evidence gathered before any fit) is a real argument,
not a rationalization. Three specific problems sit around it:

1. **The accepted cost is justified with evidence about a different candidate.** *"Given Gradient
   Sports' two prior losses on unchanged grounds, this is a cheap price."* But the `full` that lost
   twice was **public + GS (81)**. The `full` in this spec is **public + 98 + GS (179)**. The
   question has changed from *"does GS help public?"* to *"does GS help public + 98?"*. If the 98
   help sub-threshold and GS helps sub-threshold, together they could clear while `sc_extended`
   alone does not — and the registered rule ships `public` and never finds out. The accepted cost is
   larger than the spec says.

2. **The tie-break is the weakest rule in the document, and it decides what ships.** *"Ship `full`
   instead of `sc_extended` iff `full` also clears **and** its mean Δ exceeds `sc_extended`'s."* That
   is a single point estimate — no sign-consistency, no error control — deciding between two models
   that have both already cleared. When they are close it is a coin flip on noise. Apply the same
   ≥ K−1 sign rule to the `full`-vs-`sc_extended` contrast, or default to the simpler arm unless
   `full` dominates.

3. **The 0.34 figure assumes independence.** `full` ⊃ `sc_extended`; the two share the entire public
   arm and most of their training data, so their fold deltas are strongly positively correlated and
   the true inflation from testing both is well below 0.34. You may be paying the accepted cost of a
   fixed sequence to solve a smaller problem than you think.

### M4 · §4.1 — the protocol is structurally biased **against** the expansion it is testing

This is the point I would raise loudest, because it may change whether the runs are worth doing.

The paired comparison fits **every candidate at the public-optimal hyperparameters**
(`_paired_data_effect(..., shared_params=params_public)`, `train_xshot_occurrence.py:305-307`; the
candidate mask is `:202`). Hyperparameters tuned by HPO on **17 matches** are then used to fit a
**179-match** corpus. Larger corpora generally want different capacity and less regularization, so
the expansion arms are evaluated **under-tuned** — a systematic handicap, in the direction of "more
data looks worse".

The spec is aware this is "the existing protocol" and declines to re-tune it, for a good reason
(only the data should differ). But it then **leans on that protocol's prior verdicts** — GS's two
losses — to justify both the ordering and the accepted cost. If those losses are partly an artifact
of this handicap, the a-priori argument is weaker than claimed, and the accepted cost is being
priced off a biased prior.

Combine that with the acknowledged structural limits — 17 public matches, K=5, **~3.4 held-out
matches per fold** — and the honest question before spending 30–40 DGX-hours is: **can this protocol
ever return "yes"?** It is 0-for-2, it is low-powered, and it handicaps the arms it is testing. At
minimum, register the hyperparameter asymmetry as a known bias against the expansion arms. Better:
report, alongside the registered verdict, each candidate at *its own* HPO parameters — not as a ship
criterion, but so you can see whether a loss is a data verdict or a capacity artifact.

---

## MINOR

- **m1 · §3.4's default is fail-OPEN, in a document whose §3.2 is fail-closed.** *"Missing dimensions
  → default 105/68 with a warning."* That silently reproduces the exact defect being fixed, on any
  bronze that lacks the columns — and a warning is invisible in a DGX batch log. This contradicts
  your own fail-safe-defaults rule: unknown signals go to the safe side. Raise, or require an
  explicit `assume_standard_pitch=True` opt-in.
- **m2 · §6 has no regression test for IDSSE/GS after the `_download_to_temp` rename (§3.1.2)**,
  which touches **every** provider's cache keys and temp filenames. *"Safe for IDSSE/GS, which sniff
  magic bytes"* is asserted, not tested.
- **m3 · The size gate.** `artifact_bytes` is summed at `train_ghost_gk.py:508`, but `metrics.json`
  is written into that same directory at `:542` — the measured payload **excludes a file that
  ships**. With only 2.4% headroom claimed against the 15 MB cap, confirm the fix measures the
  *shipped* set, not a prefix of it.
- **m4 · A strength, flagged because it is load-bearing across two repos.** The `PUBLIC_CORPUS`
  assertion (exactly the known 17) is the best single control in the document. Be aware it couples
  silly-kicks to the pining mirror: if pining's *public* SkillCorner listing ever changes, this
  fires. That is correct behaviour — just know that it will.

---

## Answers to the four questions you asked

**§4.1 — is fixed-sequence the right control, is the ordering legitimate, is the cost acceptable?**
The control is right and correctly applied. The ordering is legitimate — a real a-priori argument
from evidence gathered before any fit. **The cost is *not* correctly accounted** (M3.1): it is
priced using the prior losses of a candidate that no longer exists. And the tie-break that actually
decides what ships (M3.2) is the least rigorous rule in the spec.

**§4.2 — is anything else confounded?** Yes: **ghost-GK** (M2). Stage A covers xS and xCross; the
model whose target the pipeline change most affects has no pipeline-matched control. Also check
`ball_z` provider-correlation.

**§4.3 — is 0.05 m defensible, or chosen to be passable?** Neither — it is **unresolvable** (B2).
The gate's own tolerated fold noise is ~10× the band. It was not gamed; it was never costed. And a
deeper problem sits underneath it: the gate cannot see the failure mode it exists to prevent (B1).

**§3.4 — scale, or preserve true metres?** **Scale — the call is right**, and for a stronger reason
than the spec gives: the *events* side is already normalized to 105×68, so preserving true metres in
tracking would put the goal line and the shot coordinates in different places, by up to 2 m. That is
decisive and it is not really a close call. But the implementation does **not** reproduce kloppy
(M1), and the spec asserts kloppy is correct without checking — settle that first.

**Also asked: is §3.2 genuinely fail-closed on the feature-cache path?** **No** (B3a). **Does §4.5
draw the line in the right place?** For xS/xCross yes; for ghost-GK **no** (B1). **Any §6 test that
would pass with the feature deleted?** Yes — the native-vs-kloppy parity test (B4), and the "105×68
is byte-identical" no-op test proves nothing about the fix it guards.
