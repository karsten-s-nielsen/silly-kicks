# xT-GK v2 construct validity — resolved-origin re-run (4.46.0, ADR-036 / PR-S113)

The 4.45.0 construct-validity run was measured on a **contaminated cohort**: ~24% of the Gradient
Sports GK-distribution domain and ~18% of SkillCorner's were scored at a **fabricated grid zone**
(`flat_zones` maps a NaN coordinate to `(0,0)` → zone 176, the own-corner cell, and the scoring seam
dropped nothing). SkillCorner's goal-kick origins were separately **present-and-wrong** — the
broadcast **ball** detection, not the keeper (ADR-024 / PR-S104).

This is the re-run on **resolved keeper origins** (`fct_action_context.xt_gk_origin_x/_y`, persisted
by PR-S101 and never read by v2). Two legs, so the delta is attributable.

**Everything else is frozen**: same metrics, same baselines, same κ=1 headline, same a-priori
parameters. No retuning. Reported as it came out (ADR-036 §3).

---

## The contamination, measured on live gold

| provider | GK-distribution actions | `native` | `resolved_origin` (was fabricated) | `unresolved` (now honest NaN) |
|---|---|---|---|---|
| gradientsports | 3874 | 2928 | **530** | **416** |
| skillcorner | 5487 | 4516 | **971** | 0 |

*(post-`prepare_cohort` denominators — frame-absent null-pressure rows already dropped.)*

**946 GS rows (24.4%) were previously scored at zone 176**; 530 had a resolved origin sitting unused
in gold, and 416 were never resolvable and are now honest NaN instead of a fabricated number.
**971 SkillCorner rows** had a *present-but-wrong* broadcast-ball origin silently replaced by the
actual keeper.

## Outcome-AUC lift (v2 − max(raw_completion, destination_xt, v1_stored))

| provider | 4.45.0 (raw origins) | leg 1 (corrected coords, **pre-fix ρ**) | leg 2 (corrected coords + **retrained ρ**) |
|---|---|---|---|
| gradientsports | −0.1387 | −0.1661 | **−0.1474** |
| skillcorner | −0.0720 | −0.0476 | **−0.0268** |

## Keeper discrimination — action-level ICC by resolved `player_key`

| provider | 4.45.0 v2 | leg 1 v2 | leg 2 v2 | v1 (reference) |
|---|---|---|---|---|
| gradientsports | **−0.0020** | 0.0270 | **0.0256** | 0.0193 |
| skillcorner | **0.0109** | 0.0218 | **0.0147** | 0.0176 |

v1's ICC is unchanged by construction — v1 **always** used the resolved origins. That asymmetry is
precisely what made the 4.45.0 v2-vs-v1 head-to-head invalid.

## κ sweep (leg 2; κ=1 is the pre-frozen headline)

| provider | κ=1.0 | κ=1.5 | κ=2.0 |
|---|---|---|---|
| gradientsports | −0.1474 | −0.1513 | −0.1535 |
| skillcorner | −0.0268 | −0.0264 | −0.0270 |

κ=1 remains the right headline; nothing in the sweep argues otherwise.

---

## Verdict — honest, and mixed

**1. The "keeper-flat" finding does not survive on Gradient Sports.** v2's keeper-discrimination ICC
went from **−0.0020 (worse than nothing) to +0.0256**, and now **exceeds v1's 0.0193 for the first
time**. SkillCorner's roughly doubled (0.0109 → 0.0218 under leg 1). This is exactly the direction
predicted before the run: a fabricated origin is **keeper-independent**, so injecting it into ~24% of
every keeper's actions compresses between-keeper variance and drags ICC toward zero. Removing it
restored the signal. **The single strongest claim of the 4.45.0 negative verdict is overturned.**

**2. v2 is still NOT construct-validated by outcome-AUC.** The lift remains negative on both
providers (GS **−0.1474**, SC **−0.0268**), i.e. v2 still loses to simple baselines at predicting
whether a possession reaches a shot. SkillCorner improved materially (−0.072 → −0.027); Gradient
Sports did not (−0.139 → −0.147). On GS both v1 (AUC 0.381) and v2 (0.475) sit below chance on this
target, while `raw_completion` scores 0.622 — so the outcome-AUC lens is dominated by completion, and
neither GK-value metric adds to it.

**3. ρ's retrain trades outcome-AUC for keeper discrimination.** The retrained ρ improves the lift on
both providers but slightly *reduces* the ICC (GS 0.0270 → 0.0256; SC 0.0218 → 0.0147). Worth
knowing; not resolved here.

**Bottom line for the interpretation-fork decision.** The 4.45.0 verdict rested on two legs —
"keeper-flat" and "no outcome-AUC lift". **The first leg was an artifact of the contaminated origins
and is gone. The second stands.** Whether the remaining outcome-AUC gap is the V-reward definition
(§2.1 remainder-of-possession vs `E[first-shot xG]`) or the dormant PEV (`p′=p`, pending
receiver-pressure `q`) is now a decision that can be taken on trustworthy numbers.

### Leg-1 attribution caveat

Leg 1 (corrected coords + **pre-fix** ρ) also shifts the pre-fix ρ's *input distribution*, because its
features derive from the now-overridden coordinates. Leg 1 therefore isolates *"origin effect
**including** ρ-input shift"*, **not** pure zone relabeling.

### ρ retrain (both variants PASS the calibration gate)

| variant | rows | AUC | ECE | slope | gate |
|---|---|---|---|---|---|
| `default` (gradientsports) | 2923 → **3451** | 0.781 → **0.798** | 0.031 | 1.00 | PASS |
| `skillcorner` | 5477 → **5477** | 0.650 → **0.662** | 0.029 | 0.92 | PASS |

GS's domain **grew by 528 rows** — the goal-kicks whose NaN origins had excluded them from ρ training
entirely. SkillCorner's row count is **identical**: nothing was added or dropped, only the goal-kick
*geometry* was corrected — and the model improved anyway. Both ship; `_PROVIDER_VARIANT` unchanged.

---

## Files

- `gradientsports.md` / `skillcorner.md` — **leg 2** (shipping: corrected coords + retrained ρ)
- `*.leg1-prefix-rho.md` — **leg 1** (corrected coords + pre-fix ρ), retained for attribution
- `*.4.45.0-raw-origins.md` — the **contaminated** 4.45.0 run, retained as the historical record
- `keeper_discrimination.md` — leg 2 ICC; `keeper_discrimination.leg1-prefix-rho.md` — leg 1
