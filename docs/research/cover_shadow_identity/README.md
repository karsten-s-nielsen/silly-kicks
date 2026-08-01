# TF-30 — `max_single_defender_player_id` agreement measurement

> ## RE-MEASURED post-RC1 on 2026-07-31. Agreement FELL 0.157 → **0.0443**; the verdict is unchanged.
>
> The producing driver, `scripts/measure_cover_shadow_argmax_agreement.py`, carried the ADR-028
> **RC1** defect when the original numbers were taken — it built `passer_xy` from raw **action-LTR**
> `start_x`/`start_y` and passed it beside **frame-LTR** defenders, receivers and ball, with no
> home-only filter. 4.70.0 fixed the two `features.py` callers; this driver imports
> `_compute_cover_shadow_dict` **directly**, so it was never a registered RC1 site and the defect
> stayed live until ADR-052 (4.72.0) fixed it here. Re-run from a clean tree at commit `7475a27`,
> same corpus (GS `10502`/`10503`/`10504`, `tracking_limit=40000`, 1039 scored / 970 qualifying).
>
> | | pre-RC1 | post-RC1 |
> |---|---|---|
> | agreement | 0.1567 | **0.0443** |
> | agree / disagree | 152 / 818 | **43 / 927** |
> | Wilson 95% | [0.135, 0.181] | **[0.033, 0.059]** |
> | harm median (exact units) | 0.817 | 0.846 |
> | `max_def` zero / non-zero | 69 / 970 | **69 / 970 (identical)** |
>
> **The defect had been INFLATING agreement, not suppressing it.** The earlier note reasoned about
> a *ceiling* — "even if every away row flipped to agreeing, ≤0.657" — and framed the correction as
> something that could only help. It went the other way: correcting the passer made the cheap path
> agree *less*. The bound held, but its implied direction was wrong, and that is worth recording
> rather than quietly replacing the number.
>
> **Internal check that the fix touched only what it should:** the `max_def` distribution is
> byte-identical across the two runs (69 exactly-zero, 970 non-zero, smallest non-zero
> 0.0036426529 in both). That column comes from the EXACT path, which does not consume the passer —
> so `TOL_ATTRIB`'s supporting figures stand unchanged, and the RC1 fix is confirmed to have moved
> the cheap path alone.

**Verdict: the cheap path cannot support an identity column.** Agreement between the default
(`detailed=False`) argmax and the exact (`detailed=True`) argmax is **0.0443**, against a
pre-registered floor of 0.90.

**Decision taken: GATE.** `max_single_defender_player_id` ships, but is populated **only** under
`detailed=True`; the cheap path returns `None` unconditionally and never names a defender. The exact
path's identity is correct by construction — it is the argmax over the same
`compute_blocking_score(defenders_to_remove=[d])` values that produce
`max_single_defender_blocking_score`, pinned by
`test_identity_is_the_argmax_over_all_lane_blockers`. The gate itself is guarded by
`test_cheap_path_never_names_a_defender`, so removing it fails CI rather than silently restoring a
column that is wrong **95.6%** of the time.

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
| Agreement | **0.0443** |
| Wilson 95% CI | **[0.033, 0.059]** |
| Agree / disagree | **43 / 927** |

`n` clears the >= 100 requirement by an order of magnitude, and the interval's **upper** bound is
**0.059** — the floor is not merely missed, it is out of reach by more than a factor of fifteen.

With roughly 10 lane blockers per action, chance agreement is ~0.10, so the cheap path's argmax is
**0.44x chance — WORSE than picking a lane blocker at random.** The pre-RC1 measurement read 1.6x
BETTER than random; correcting the passer reversed that reading. The earlier single-match pilot
figure of 0.1992 was taken under the same defect and is likewise superseded.

## The disagreements are not near-ties

This is the part that settles it. Harm is measured in **exact-path units** — the cheap path's nominee
scored *through the exact path*, differenced against the exact winner. (It is deliberately **not**
`max_def_exact - max_def_cheap`: the two paths compute different quantities, so their maxima differ in
magnitude whether or not the argmax agrees. Both terms must share one scale.)

| Harm at disagreements | Value |
|---|---|
| median (exact units) | 0.846 |
| p90 | 1.607 |
| max | 8.612 |
| **median, as a fraction of the exact maximum** | **0.988** |
| p90, as a fraction | **1.0** |

The median disagreement gives up **98.8%** of the true winner's blocking score, and at p90 the
fraction is **1.0** — the cheap path names a defender whose exact-path contribution is **exactly
zero**. These are not two near-equal defenders being ordered differently; the cheap path routinely
names someone who did nothing.

## Why this is not a bug in the cheap path

`_cover_shadows.py` documents the lightweight branch as bit-identical to the prior per-(defender,
receiver) `lane_control` loop within rtol 1e-10 — i.e. it is faithful to a **lane-based** definition
of "which defender blocks most", never to the pitch-control Voronoi counterfactual. The two are
legitimately different constructs.

The existing `TestDetailedVsLightweightCorrelation` asserts Spearman rho >= 0.7 between the two paths
on the **value**. That is entirely compatible with the **argmax** disagreeing **95.6%** of the time: the
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

The recorded run is `run_commit: 7475a2781dc0`, `run_tree_dirty: false`, `run_tree_state: clean` —
produced from a clean tree, so the SHA genuinely describes the code that ran. This is a **citable**
artifact.

The previous version of this file said the opposite (`run_tree_dirty: true`, "re-run from a clean
tree to produce a citable artifact"), because the original numbers were taken with the TF-30 changes
uncommitted. That re-run is what this file now reports. The ordering was forced rather than chosen:
ADR-052 made this driver REFUSE a dirty tree, so the re-measurement could not happen until the
cycle's own commits landed — the guard doing exactly its job, on its author.
