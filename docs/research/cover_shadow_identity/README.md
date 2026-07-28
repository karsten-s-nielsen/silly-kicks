# TF-30 — `max_single_defender_player_id` agreement measurement

**Verdict: the cheap path cannot support an identity column.** Agreement between the default
(`detailed=False`) argmax and the exact (`detailed=True`) argmax is **0.157**, against a
pre-registered floor of 0.90.

**Decision taken: GATE.** `max_single_defender_player_id` ships, but is populated **only** under
`detailed=True`; the cheap path returns `None` unconditionally and never names a defender. The exact
path's identity is correct by construction — it is the argmax over the same
`compute_blocking_score(defenders_to_remove=[d])` values that produce
`max_single_defender_blocking_score`, pinned by
`test_identity_is_the_argmax_over_all_lane_blockers`. The gate itself is guarded by
`test_cheap_path_never_names_a_defender`, so removing it fails CI rather than silently restoring a
column that is wrong 84% of the time.

The cost of opting in was measured: **2.3–3.2×** (39–42 ms/action cheap vs 98–125 ms/action exact,
warmed, both orderings). Not the order of magnitude an initial unwarmed measurement suggested.

Produced by `scripts/measure_cover_shadow_argmax_agreement.py`. Raw report `agreement.json`,
per-action records `records.csv`.

## The pre-registered rule

Named **before** any number was seen, and not moved afterwards:

| Outcome | Action |
|---|---|
| agreement >= **0.9** at **n >= 100** | ship as specified |
| below either | do not ship silently — gate to `detailed=True`, or drop |

0.9 is a stated engineering threshold, not derived: *a consumer reading `..._player_id` assumes it is
usually right.*

## Result

Corpus: 3 Gradient Sports WC2022 matches, `--tracking-limit 40000`.

> **Corpus limitation, stated rather than buried.** `--tracking-limit 40000` truncates each match's
> frames. The loader warns `period 1 link_rate 0.46` (363 of 676 unlinked) and, more sharply,
> `period 2 link_rate 0.00 (756 actions, 756 unlinked) ... near-disjoint (overlap 0.00)` — the cap is
> exhausted inside the first period, so **period 2 contributes nothing**. This is a FIRST-PERIOD
> sample, not a whole-match one.
>
> It is not expected to bias the comparison: truncation removes actions from BOTH paths identically,
> and there is no mechanism by which the period an action falls in would favour the lane-based argmax
> over the pitch-control one. But the honest description of the corpus is "first period of 3 matches",
> and a reader should not infer whole matches from "3 matches". Re-running without `--tracking-limit`
> would settle it; at n=970 with a CI upper bound of 0.181 against a 0.90 floor, it would not change
> the verdict.

| Quantity | Value |
|---|---|
| Actions scored | 1039 |
| Qualifying (>= 2 lane blockers, exact `max_def > TOL_ATTRIB`) | **970** |
| Agreement | **0.1567** |
| Wilson 95% CI | **[0.135, 0.181]** |
| Agree / disagree | 152 / 818 |

`n` clears the >= 100 requirement by an order of magnitude, and the interval's **upper** bound is
0.181 — the floor is not merely missed, it is out of reach. A single-match pilot gave 0.1992, so the
result is stable across corpus size.

With roughly 10 lane blockers per action, chance agreement is ~0.10. The cheap path's argmax is about
**1.6x better than random**.

## The disagreements are not near-ties

This is the part that settles it. Harm is measured in **exact-path units** — the cheap path's nominee
scored *through the exact path*, differenced against the exact winner. (It is deliberately **not**
`max_def_exact - max_def_cheap`: the two paths compute different quantities, so their maxima differ in
magnitude whether or not the argmax agrees. Both terms must share one scale.)

| Harm at disagreements | Value |
|---|---|
| median (exact units) | 0.817 |
| p90 | 1.597 |
| max | 8.612 |
| **median, as a fraction of the exact maximum** | **0.984** |
| p90, as a fraction | **1.0** |

The median disagreement gives up **98.4%** of the true winner's blocking score, and at p90 the
fraction is **1.0** — the cheap path names a defender whose exact-path contribution is **exactly
zero**. These are not two near-equal defenders being ordered differently; the cheap path routinely
names someone who did nothing.

## Why this is not a bug in the cheap path

`_cover_shadows.py` documents the lightweight branch as bit-identical to the prior per-(defender,
receiver) `lane_control` loop within rtol 1e-10 — i.e. it is faithful to a **lane-based** definition
of "which defender blocks most", never to the pitch-control Voronoi counterfactual. The two are
legitimately different constructs.

The existing `TestDetailedVsLightweightCorrelation` asserts Spearman rho >= 0.7 between the two paths
on the **value**. That is entirely compatible with the **argmax** disagreeing 84% of the time: the
cheap path ranks *how much* blocking happened without identifying *who* did it. Both facts are true
and neither is a defect — but together they mean an identity column served from the default path
would be wrong most of the time.

## Useful by-product: `TOL_ATTRIB` is now data-backed

The full `max_def` distribution (including rows the qualification filter excludes) separates cleanly:

- **69** values exactly `0.0`
- **970** non-zero, the smallest at **3.64e-3**
- median **0.832**

So "no attribution" and "small attribution" are cleanly separable, and `TOL_ATTRIB = 1e-12` sits
safely inside a gap spanning nine orders of magnitude. The constant is no longer provisional by
reasoning — it is confirmed by the distribution it was supposed to be set from.

## Provenance

The recorded run is marked `run_tree_dirty: true` — it was produced from a working tree with the
TF-30 changes uncommitted. That is recorded rather than laundered: `git rev-parse HEAD` returns the
same SHA dirty or clean, so a bare SHA would describe code that did not run. Re-run from a clean tree
(without `--allow-dirty`) to produce a citable artifact.
