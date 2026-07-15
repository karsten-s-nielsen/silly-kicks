# Critical review 2 — rev 2 of `2026-07-14-skillcorner-corpus-and-visibility-design.md`

**Reviewer:** external session · **Date:** 2026-07-14 · **Verdict: still do not start the runs.**

Rev 2 is a genuine revision, not a defensive one. Every blocker and major was accepted, B1 was fixed
**deeper** than I proposed (detected-only *training targets*, plus the interpolator-tell refusal —
that is a better fix than the one I wrote), and rev 1's "kloppy is correct" was retracted on
evidence. The B3 licensing trace was independently verified in code. That is the right way to take a
review.

So this pass is about whether the **fixes are correct**. Three of them are not, and the most serious
one is created *by* the fix. I re-measured everything on the real feed.

---

## BLOCKERS (new — introduced by rev 2)

### N1 · §3.4 — the single-sourced events transform **clamps**, and tracking is not events

This is the headline. `spadl/skillcorner.py::_transform_coords` — the function §3.4 now registers
the tracking builder to **import and call** — ends like this:

```python
# silly_kicks/spadl/skillcorner.py:59-61
# Clamp to SPADL pitch boundaries (raw data can slightly exceed pitch dims)
x_out = x_out.clip(lower=0.0, upper=105.0)
y_out = y_out.clip(lower=0.0, upper=68.0)
```

For **events** this is harmless: an action's location is on the pitch by construction. For
**tracking** it is destructive, because tracking is full of legitimately off-pitch positions — the
ball out of play, a keeper behind his line, a throw-in taker off the touchline.

Measured on three public matches (3.0 M player rows), what the clamp would silently move:

| rows | clamped | note |
|---|---|---|
| all players | **0.70%** | 20,909 rows |
| goalkeepers | **0.61%** | ghost-GK's *target* |
| **ball** | **12.26%** | 16,720 of 136,392 |
| worst single displacement | **6.97 m** | |

**12% of ball rows.** Every out-of-play ball gets snapped onto the byline or touchline — including a
ball that has crossed the goal line, which is exactly what a goal *is*: `x > 105` becomes `x = 105`,
indistinguishable from a ball on the line. `z` is untouched, so a ball 3 m behind the goal at head
height becomes a ball *on the goal line* at head height.

Three compounding problems:

1. **It hits all 108 matches, not the 4 the spec accounts for.** The clamp is unconditional, so it
   fires on the 97 pitches that are already 105×68 — the ones §3.4 is supposed to leave alone. §5's
   Hyrum table says the research path moves "≤ 0.29 m on the four non-105 public matches". It
   actually moves by **up to 6.97 m on every match**. That is an unlisted Hyrum event.
2. **§6's new structural test cannot see it.** "`tracking.skillcorner` and `spadl.skillcorner` must
   produce **bit-identical** output" passes trivially — *both* clamp. A consistency test cannot
   detect a defect the two sides share. (Same for the action↔frame co-location gate.)
3. **§4.4's pitch gate becomes vacuous** — see N2.

**Fix:** single-source the *transform*, not the *clamp*. Split `_transform_coords` into a pure
affine `_scale_coords(x, y, L, W)` and a separate `_clamp_to_pitch(...)`; events call both, tracking
calls only the first. The seam you want to share is the geometry, and the clamp is not geometry — it
is an events-domain assumption that is false in the tracking domain.

### N2 · §4.4 — the within-pitch invariant is now a guard that cannot fail

> "After §3.4, every match's post-transform coordinates must satisfy the builder's existing
> within-pitch invariant. Any match failing is excluded and reported."

After N1, **every** match satisfies the within-pitch invariant, because the transform clamps
everything into the pitch. The gate is true by construction and can never exclude anything.

This is a fresh instance of exactly the class of defect rev 2 just fixed (B4, the parity test that
could not fail) — and it was introduced by the fix. Worth pausing on: the *reason* it slipped through
is that the fix was validated at the layer of "do the two paths agree" rather than "does the guard
still bite". Once N1 is fixed, this gate becomes meaningful again — but it needs its own mutation
test (plant an out-of-pitch match; the gate must fire).

### N3 · §4.3 — the detected-only rule is **fail-open** on the very field §3.2 makes fail-closed

§4.3 registers: *"Ghost-GK trains only on frames where the keeper was actually detected, wherever
detection is knowable (`visibility` truthy). This bites SkillCorner only: Gradient Sports and IDSSE
… carry no detection flag, so they are unrestricted."*

So the implementation must read `visibility = None` as **"keep this row"**. That is fail-open on
`visibility` — in a document whose §3.2 makes `visibility`-unknown mean *restricted*, fail-closed,
and calls that "the one change that is a compliance control".

Now recall what this entire spec exists to fix: **the kloppy gateway hard-codes `visibility = None`**
(§1-finding-3). Under the new rule, any provider routed through kloppy has null visibility and its
keepers are therefore treated as "observed by construction, unrestricted" — the null that means
*"we threw the information away"* is read as *"the information was never needed."* That is the same
failure shape as the licensing landmine, one section later: a missing signal silently interpreted as
the permissive case.

**Fix (your own fail-safe-defaults rule):** an explicit allowlist of providers whose keepers are
observed by construction — `{gradientsports, idsse}` — and **any other provider with null
`visibility` raises**. Never infer "unrestricted" from a null.

### N4 · §4.1 — the new **primary** ship rule carries a fresh, unregistered bias, pointing the same way

Rev 2 makes **best-vs-best** (each candidate at its own HPO parameters) the primary rule, precisely
to remove the shared-params handicap I raised in M4. Correct diagnosis. But look at how HPO is run:

```python
# train_xshot_occurrence.py:75-88  — _hpo_once
obj = XShotOccurrenceObjective(fold={tag: [(X, pd.Series(y), groups)]}, ...)
```

HPO is run **once, outside the outer CV**, on the candidate's *entire* dataset. It is not nested
inside the public evaluation folds. So every candidate's hyperparameters are chosen having already
seen the public games that later serve as its held-out folds — and the leakage is **asymmetric**:

- `public`: `_hpo_once(X[is_public], …)` tunes on **exactly the 17 matches that constitute the whole
  evaluation universe**. Maximal leakage → its held-out score is optimistically biased.
- `full` / `sc_extended`: those same 17 matches are diluted across 115 / 179. Much weaker leakage.

Under the old shared-params protocol this did not differentiate the arms — every arm used
`params_public`, so the leakage was common-mode. **Best-vs-best converts a common-mode bias into a
differential one, favouring `public`.** The rev-2 fix removes a bias against the expansion arms and
installs a new one in the same direction — and this rule now *decides what ships*.

**Fix, cheapest first:**
- **Register it.** At minimum this must be stated in the registration, because it decides the ship.
- **Or** tune every arm's hyperparameters on data that excludes the public evaluation games (e.g. HPO
  on owner-tier data only) — unbiased across arms, no extra folds.
- **Or** nest the HPO inside the outer folds (correct, ~5× the HPO budget).

Do not leave a decision rule whose bias you have just discovered in the *previous* revision
unaudited in the *next* one.

---

## MAJOR

### N5 · §1.6.1's measurement table does not reproduce — and the spec names a function that does not exist

**The function is `_transform_coords`, not `_rescale_coordinates`.** The spec names
`_rescale_coordinates` three times (§1.6.1, §3.4, §9) and it appears nowhere in the repository. An
implementer following §3.4 literally will not find it.

The *math* claim is **correct** — `(x / (L/2)) * 52.5 + 52.5` ≡ `(x + L/2) · (105/L)` — so §3.4 and
the events converter really are algebraically identical. Good.

But the kloppy comparison table is wrong. Re-measured with a clean 1:1 dedup'd join (n = 32,953,
period 1, match 1886347 @ 104 m); rev 1's numbers *and* mine were both contaminated by a fan-out join
until now:

| raw x | events (= §3.4) | kloppy | divergence — **measured** | divergence — **spec §1.6.1** |
|---|---|---|---|---|
| −52 (goal line) | 0.000 | **−0.151** | **−0.15 m** | 0.00 m |
| 0 (centre spot) | 52.500 | **52.496** | **0.00 m** | 0.15 m (claims 52.647) |
| +52 (goal line) | 105.000 | **105.143** | **+0.14 m** | +0.29 m |

kloppy's forward map is `1.012435·x + 52.496`. So **kloppy puts the centre spot at 52.50, correctly
— not at 52.647**, and the divergence is **symmetric, ±0.15 m at the goal lines, zero at the centre**
— not monotone from one goal line to 0.29 m. The spec's headline "**diverges by up to 0.29 m** and
puts the centre spot at 52.65" is **2× overstated and mislocated**; it appears in the executive
summary, §1.6.1, and §5's Hyrum table. The real research-path geometry move from the re-route is
**≤ 0.15 m** (before N1's clamp, which dwarfs it).

One more thing nobody has noticed: after a clean fit, **kloppy's map still carries a 0.14 m
non-affine residual**. It is not a fixed affine transform at all. Whatever kloppy is doing to
SkillCorner coordinates has not actually been characterised by anyone — including this spec, which
makes a correctness claim about it.

### N6 · §1.6.1 — "consistency" is not "correctness", and the landmark check was deleted on that fallacy

> "The reviewer's request for an independent landmark check is answered by a stronger invariant:
> after the fix, tracking and events are computed by *the same function*, so they cannot disagree."

Single-sourcing guarantees **tracking == events**. It does not guarantee **either is right**. If the
shared transform mis-places the goal line, tracking and events are now wrong *together*, and — by
construction — **no test in §6 can detect it**, because every test in §6 is a consistency test.

This matters because the two candidate transforms have the **same form** and differ in exactly one
number: the assumed pitch length.

- events / §3.4: `L = 104` (from metadata)
- kloppy: `L = 103.71` (unexplained)

Single-sourcing picks 104 and makes that choice unfalsifiable.

**I think 104 is right** — metadata is the declared truth, 103.71 is an unexplained number, and
kloppy's map is not even affine (N5). But *that* is the argument the spec should make: an argument
from **provenance**, not "we made them agree, so they cannot disagree." Write that down instead.

And keep a landmark check, because it is the only thing that can confirm it. I tried to run one for
you and the cheap routes are closed: `image_corners_projection` is **all-null** in the feed, and the
dynamic-events taxonomy carries no set-piece type (`off_ball_run`, `on_ball_engagement`,
`passing_option`, `player_possession` only). The viable route is ball raw-|x| at goal-kick / corner
restarts pooled across the 10 public matches — but 52.00 vs 51.86 is a **0.14 m** question and needs
~50+ restarts to resolve. Honestly, the cheapest reliable answer is to **ask SkillCorner** what
`pitch_length` means and what kloppy is reading.

### N7 · §4.3 — the paired admission test has no common evaluation domain

> "Both runs use the **same keeper groups, the same fold ids, and the same seed**."

This cannot be done as written. The baseline corpus (81 matches) and the expanded corpus (179) have
**different keeper populations** — the 98 bring ~56 keepers including Courtois, Lunin and Kepa, who
are (almost certainly) absent from the baseline. There is no "same keeper groups" across two
different keeper sets.

§4.1 gets this exactly right — both candidates are scored *"on the common public held-out fold"*.
§4.3 needs the same construction and does not specify it. Pin it:

- **Folds are defined over the keeper population present in *both* corpora** (i.e. the baseline's
  keepers). That is the common evaluation domain.
- **Assert that no keeper in a test fold appears anywhere in the added 98** — otherwise the expanded
  model has trained on a keeper it is being tested on, and the paired Δ is leakage, not learning.
  (This is probably vacuous — RM keepers are unlikely to appear in the A-League/Bundesliga/GS
  baseline — but "probably vacuous" is what an assertion is for.)

Without this, "paired" is not defined, and it is the rule that admits or refuses the 98.

### N8 · §3.3 — dueling text: the retraction did not reach the table

§1.6.1 states: *"Rev 1 asserted 'the kloppy path scales from metadata and is correct'. **That
assertion was wrong, and is retracted.**"*

§3.3's comparison table, unchanged from rev 1:

| | kloppy gateway | native builder |
|---|---|---|
| pitch dims | **scaled correctly** | fixed +52.5/+34 (defect, §1.6) |

The retracted claim is still asserted, in a table, in the section that makes the routing decision.
This is the patch-plus-appendix pattern: the new prose lands, the old prose survives, and a reader
who scans the table gets the retracted answer. Sweep for it — §7 item 2 likewise still quotes the
uncorrected ~0.34 framing that §4.1's own correlation note now walks back.

---

## What rev 2 got right (so it is not re-litigated)

- **B1 → §4.3.** Restricting ghost-GK's *training targets* to detected keepers, not merely the
  admission metric, is the right call and better than what I proposed. The **interpolator-tell
  refusal** (all-frames MAE improves while detected-only MAE degrades ⇒ automatic refusal) is an
  excellent, falsifiable gate — the strongest single addition in this revision.
- **B2 → §4.3.** The paired ΔMAE under sign-consistency, and requiring a *demonstrated improvement*
  rather than "no material harm", is correct. (It just needs N7's common domain to be well-defined.)
- **B3 → §3.2.** Label derived from the ship mask's visibility composition; `_PUBLIC_PROVIDERS`
  deleted at all six sites; cache fingerprint + red-first CI guard. This is now a real control.
- **M3 → §4.1.** The tie-break moved to sign-consistency, and the accepted cost is re-priced with the
  deferred-cycle consequence stated. Ties going to less data is the right default.
- **m1 → §3.4.** Missing pitch dimensions now raise. Correct.

---

## Summary

| # | Sev | Item |
|---|---|---|
| N1 | **Blocker** | The single-sourced events transform **clamps**; tracking is not events. 12.3% of ball rows, 0.7% of player rows, worst 6.97 m — on **all 108 matches**. Share the scale, not the clamp. |
| N2 | **Blocker** | §4.4's within-pitch gate becomes vacuous under N1 — a new guard that cannot fail. |
| N3 | **Blocker** | §4.3's detected-only rule is **fail-open on null `visibility`** — the same failure shape as the licensing landmine, on the same field §3.2 makes fail-closed. Use a provider allowlist; raise on unknown. |
| N4 | **Blocker** | Best-vs-best (the new **primary** ship rule) has non-nested HPO leakage that is **asymmetric across arms and favours `public`** — a new bias replacing the one it fixed. Register it, or tune off the evaluation games. |
| N5 | Major | `_rescale_coordinates` does not exist (it is `_transform_coords`). §1.6.1's kloppy table does not reproduce: divergence is **±0.15 m symmetric, centre spot correct at 52.50** — not 0→0.29 m with the centre at 52.647. kloppy's map is not even affine (0.14 m residual). |
| N6 | Major | Consistency ≠ correctness. Single-sourcing cannot validate the shared transform; the landmark check is still needed. Argue from **provenance** (metadata says 104; 103.71 is unexplained), not from "they now agree". |
| N7 | Major | §4.3's paired folds are undefined — the two corpora have different keeper populations. Pin the common evaluation domain and assert no test-fold keeper is in the 98. |
| N8 | Minor | Dueling text: §3.3's table still says kloppy is "scaled correctly", which §1.6.1 retracts. §7.2 still quotes the uncorrected 0.34. |

**N1 is the one to fix tonight.** It would ship a silent 7-metre geometry corruption into the
lakehouse and into every model in this cycle, on every match — while all three of the spec's new
geometry tests pass, because they are all consistency tests and the two sides share the flaw.
