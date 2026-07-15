# Critical review 3 — rev 3 of `2026-07-14-skillcorner-corpus-and-visibility-design.md`

**Reviewer:** external session · **Date:** 2026-07-14

## Verdict: the design is sound. Approve subject to four corrections — one substantive, three textual.

Rev 3 fixes all eight rev-2 findings, and fixes them **correctly** — I checked each against the code,
not the prose. No new design defects. The document also caught something I got wrong: I guessed the
keeper-overlap exclusion in §4.3 was "probably vacuous"; they checked, and **Courtois is in the
WC2022 Gradient Sports corpus *and* in 45 of the 98**. The exclusion is load-bearing. That is the
review working in both directions.

Verified fixed:

| | fix | verified |
|---|---|---|
| **N1** clamp | `_scale_to_spadl` (affine, no clamp) split from `_transform_coords` (= scale + clamp); tracking calls the former only | ✅ Correct — and it matches the builder's *existing* design: `skillcorner.py:57-70` documents an off-pitch-tolerant invariant with players legitimate to SPADL x ≈ −7.5…116 and the code says **"Not clamped"** at `:209`. Rev 2 would have fought its own builder. |
| **N3** fail-open detection | explicit `_DETECTION_AWARE_PROVIDERS` / `_FULLY_OBSERVED_PROVIDERS`; null-on-detection-aware **raises**; unknown provider **raises** | ✅ This is the fail-safe-defaults pattern applied properly |
| **N4** HPO leakage | tuning **nested inside the outer CV**; fold *k*'s public games excluded from every candidate's tuning; cost stated (~35–45 h vs ~10 h) and the budget-cap-becomes-a-model-card-caveat rule | ✅ Correct, and honestly priced |
| **N6** consistency ≠ correctness | fallacy explicitly withdrawn; 104 m chosen on **provenance**; residual uncertainty registered; **"ask SkillCorner"** carried as an action item | ✅ Exactly right |
| **N7** keeper folds | common evaluation domain = baseline keepers **minus** any keeper appearing in the 98; folds built once; no-overlap **asserted, not assumed** | ✅ |
| **N2 / N5 / N8** | §4.4 invariant live again; `_transform_coords` correctly named; §3.3's table no longer says kloppy is "scaled correctly" | ✅ |

§6 is now a real test suite — the **off-pitch-survival** gate with "route through the clamping
`_transform_coords` → fails" as its named mutation is precisely the right shape.

---

## The one substantive item left

### R1 · §4.4 registers an exclusion that nothing can perform

> "*every match's post-transform coordinates must satisfy the builder's existing within-pitch
> invariant. **Any match failing is excluded and reported.**"*

The "existing within-pitch invariant" cannot fail a match. In `tracking/skillcorner.py`:

```python
# :198-211
n_gross_off_pitch = _count_gross_off_pitch(df["x"], df["y"], df["is_ball"])
if n_gross_off_pitch:
    warnings.warn(...)          # warn + count. NEVER clamp/crash.
```

and the source comments say so outright:

- `:200` — *"Per-row GROSS off-pitch → warn + count; **NEVER clamp/crash** (one noisy row must not fail a match)."*
- `:67` — *"The **deferred** CI rate-gate is the SYSTEMATIC backstop for both."* — **deferred, i.e. not implemented.**

So §4.4 promises an exclusion mechanism that does not exist. What actually happens on a bad match is
a `warnings.warn` into a DGX batch log — which **this very spec argues is invisible** (§3.4, on why
the missing-pitch-dims default must raise rather than warn). The same argument applies here.

This matters more than it looks, because §4.4 is the **only** gate standing between the corpus and a
geometrically broken match, and the 98 are brand-new, never-validated data — including **7 pitches at
101×67** (the largest transform error in the corpus) and **13 extra-time ties**.

It is also un-calibratable as it stands: `_TOL_BALL = 30.0` m is flagged in-code as *"provisional —
re-calibrate from the measured bronze on the pining corpus"*. The measured ball excursion on real
matches is **≈ 9 m** (true scaled range `[−5.76, 114.00]`). A 30 m tolerance therefore **cannot trip
on the ball at all**. The gate is not merely unimplemented — it is tuned to silence.

**Required before approval:** §4.4 must define the rate-gate itself, since this spec is the moment
the pining corpus finally makes calibration possible:

- a per-match threshold on `n_gross_off_pitch / n_rows` (players and ball separately — they have
  different tolerances and different meanings),
- **re-calibrate `_TOL_BALL`** from the measured corpus (30 m is ~3× the observed maximum),
- exclusion on breach, counted and reported — not a warning,
- and a §6 mutation test: **plant an off-pitch match; the gate must fire.** Otherwise this is a
  fourth-consecutive guard-that-cannot-fail, and the one protecting the newest data.

---

## Three textual corrections (all in *registered* text, so all must-fix)

### R2 · §4.1 — the numbered procedure still carries the **rev-1 tie-break**

Step 2 of the registered selection procedure:

> "ship `full` instead of `sc_extended` iff `full` also clears **and its mean Δ exceeds
> `sc_extended`'s**."

Ten lines below, under **"Tie-break, registered"**:

> "Rev 1 broke a tie on a bare mean-Δ comparison … **It is replaced by** … the per-fold
> `full`-vs-`sc_extended` contrast clears `Δ_k > 0 in ≥ K−1 of K folds AND mean Δ > 0`."

The replaced rule is still sitting inside the numbered procedure an implementer will follow. This is
a **pre-registration**; an ambiguity about which rule decides the ship is the exact failure it exists
to prevent. Delete the old clause from step 2.

### R3 · §1.6.1 — the magnitude caveat contradicts the table above it

The rev-3 table presents a *completed* mirror-invariant measurement (kloppy ≈ 103.48 m, 0.263 m at
the goal line), and §3.3, §5 and §6 all quote it as settled. But the paragraph immediately below still
says: *"The magnitude **is being re-measured** mirror-invariantly, and whatever it is…"* — present
tense, as though the work were outstanding. Delete or rewrite it. (The §1.6.1 heading also still says
"rev 2".)

### R4 · §7.5 — the cost figure predates the most expensive decision in rev 3

> "**Cost.** … roughly **30–40 DGX-hours** in total."

Unchanged from rev 1 — but §4.1 now registers **nested** tuning, which *multiplies HPO by K* and is
costed there at **~35–45 h for xS + xCross alone** (against ~10 h un-nested), before Stage A, Stage B
and the two ghost-GK runs. The owner approves the budget off §7.5. Restate it.

---

## Two notes, not blocking

**The 103.48-vs-103.71 disagreement is explainable, and the explanation strengthens your case.** §1.6.1
records our two measurements of kloppy's effective pitch length as differing and "unexplained". They
differ because **kloppy's map is not affine**: a clean 1:1 fit leaves a ~0.14 m non-affine residual,
so "effective pitch length" is a fit artefact and different estimators legitimately land in different
places. You are right that it is not decision-relevant — and it sharpens the action item: *nobody has
characterised what kloppy actually does to SkillCorner coordinates.* Worth one line when you write to
SkillCorner.

**Quantify the ghost-GK evaluation domain before the run.** §4.3's keeper exclusion is load-bearing
(your Courtois finding), but the *size* of what survives is unstated. Several WC2022 keepers also keep
in LaLiga/UCL, so the exclusion bites into the Gradient Sports keeper universe. My rough count says
the surviving domain is still comfortable (order 50–60 keepers → ~10–12 per fold), so this is probably
fine — but "probably fine" is not a registered number, and if it came out small the admission test
would inherit §4.1's power problem. Report `|domain|` before Stage B.

---

## Summary

| # | Sev | Item |
|---|---|---|
| R1 | **Must fix** | §4.4 registers an exclusion no code can perform — the invariant only warns, its systematic backstop is **deferred**, and `_TOL_BALL = 30 m` is 3× the observed maximum, so it cannot trip. Define and calibrate the rate-gate; mutation-test it. |
| R2 | **Must fix** | §4.1 step 2 still carries the rev-1 mean-Δ tie-break that the tie-break paragraph replaces. Ambiguity inside a pre-registration. |
| R3 | Must fix | §1.6.1's "is being re-measured" caveat contradicts the completed table above it. |
| R4 | Must fix | §7.5's 30–40 DGX-hour cost predates nested tuning (~35–45 h for xS + xCross alone). |
| — | Note | The 103.48/103.71 gap is explained by kloppy's map being non-affine. |
| — | Note | Report the surviving ghost-GK keeper domain size before Stage B. |

**R2–R4 are text. R1 is the only real work, and it is small.** Fix those and the registration is
sound: I would approve it and move to the plan.
