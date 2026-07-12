# xT-GK v2 — FINISH RELEASE: faithful `V_opp` + full validation (single PR) — Handoff

**Status:** DRAFT handoff for the silly-kicks session. NOT committed by the analysis session.
**Date:** 2026-07-11 · **From:** Bounou/xT-GK collaboration session.
**Directive (owner):** *No more piecemeal.* This is the **one release that finishes v2** — implement the
single faithfulness fix, run the full analysis honestly, commit the final artifacts. Governing: Eyestone
v2 migration spec (`xT-GK_v1-to-v2_Migration_Spec.md` §2.1/§2.3/§3), ADR-036.

## 0. Ship-as-ONE-PR (mechanics, decided)
- **Base:** the existing uncommitted branch `pr-s112-xtgk-v2-construct-validity` (has the harness WIP).
  **Expand its scope** to this whole work order — do NOT land S112 separately first (that preliminary
  report would be overwritten in the next PR = the exact piecemeal churn we're ending).
- **One branch · one commit · one PR** (standing owner rule). Rename the branch if you like
  (`pr-s113-xtgk-v2-finish` or keep it — your call).
- **Version: 4.44.0 → 4.45.0 (MINOR).** The `V_opp` swap changes `compute_xt_gk_v2` serve output =
  library behaviour change (not a patch). Standard lockstep (pyproject / `__init__` / CHANGELOG / TODO /
  uv.lock). **Hyrum-flag the lakehouse:** re-materialize `xt_gk_v2_*` on the 4.45.0 pin.
- **The only construct-validity / decomposition / keeper reports committed are the FINAL ones** — computed
  on the faithful `V_opp`. Nothing labelled "preliminary" lands.

## 0a. What this release IS / IS NOT
- **IS:** (1) fix the one genuine faithfulness deviation (`V_opp` §2.3); (2) fix the loader's wrong-column
  keeper read; (3) run the three analyses the program was built to need (construct-validity, component
  decomposition, **keeper discrimination**) on the faithful metric; (4) a secondary faithfulness audit;
  (5) best-practice guards so the keeper-id trap can't recur.
- **IS NOT:** a metric redesign; a Jeff response (**held** until this lands and we can separate
  "faithful-but-behaves-this-way = his question" from "our-deviation = fixed"); and **NOT** a
  result-massaging exercise — see §7.

## 0b. Why (the confirmed evidence)
Component decomposition on the CURRENT (proxy-`V_opp`) metric — term |mean| share: **pos 6% · pev 0% ·
ret 6% · dzv 89%.** The turnover term `dzv = −(1−ρ)·κ·V_opp` **is ~the entire metric.** The useful
`ρ·ΔV` value-added signal predicts well alone (outcome-AUC≤5 **0.713 GS / 0.621 SC**) but is 6–8% of
magnitude and gets swamped; adding `dzv` collapses the full metric to **0.461 GS** (below chance). It is
**not** upstream in V. Root cause = §1. This is the confirmed lever — proceed.

---

## 1. THE deviation being fixed — production `V_opp` is not Jeff's `V_opp`
Jeff §2.3, verbatim: *"`V_opp(s,a)` — Expected **opponent** threat following a loss of possession by this
action, **estimated from observed post-turnover possessions**, indexed by **origin zone × pressure**.
Widen bins where deep-zone turnovers are sparse; report per-zone sample sizes."*

Production `V_opp = MirroredTurnoverCost = V(mirror(zone))` — a **geometric proxy**, not the observed-
post-turnover estimate. The faithful surface `EmpiricalTurnoverValue` exists but is wired only as a
cross-check. The mirror **over-states** `V_opp` on deep cells (mirror of the keeper zone = opponent
near-goal = structurally large), which is exactly the dominant `dzv` term. **Faithful implementation =
make the empirical estimate the production `V_opp`.**

---

## 2. Work items — the contents of the one PR (execute all)

### W1 — Loader: read the RESOLVED keeper (`player_key`), not raw `player_id`
The earlier "no keeper id in gold, E blocked on lakehouse" was a **wrong-column read.** `player_id` is raw
native — **null for goal-kicks by SPADL design** (the reason `acting_gk_from_frames` exists). The resolved
key `player_key` is **live-verified 99.9% populated** on `is_gk_distribution` rows
(`fct_action_context.player_key` = 10,035/10,046, 130 keepers).
- **Fix:** add `c.player_key` to `load_xtgk_cohort`'s SELECT (the `fct_action_context c` join already
  exists — same additive pattern as `is_gk_distribution`/`xt_gk`). Carry it through `prepare_cohort`
  untouched. **Never** source keeper identity from raw `bronze.spadl_actions`.
- This alone unblocks E (§W5). No lakehouse dependency.

### W2 — Faithful `V_opp` (§2.3): swap production mirror → observed-post-turnover *(the library change)*
- **Promote** the empirical estimate to the production `TurnoverCost` injected into `compute_xt_gk_v2`:
  expected opponent threat from **observed post-turnover possessions**, indexed by **origin-zone ×
  pressure**. The injection seam means **no change to `_metric.py` metric code** — only which adapter is
  wired as `turnover_cost`.
- **Sparsity handling is MANDATORY (§2.3), not optional.** Deep keeper turnovers are rare → the raw
  empirical estimate is 0/noise on the exact deep cells that dominate. **Widen bins where deep-zone
  turnovers are sparse; support-gate; carry per-zone sample sizes** — mirror the deep-zone-support
  discipline already in the gate. Without this the faithful `V_opp` is unusable deep and you'll just move
  the degeneracy, not fix it.
- **Role reversal:** `MirroredTurnoverCost` becomes the **cross-check** (was production). Report empirical
  vs mirror agreement/divergence per zone — **divergence on deep cells is itself a finding** (it's what
  the mirror was silently assuming, and the source of the 89% dominance).
- **Tests (TDD):** empirical surface fits from a post-turnover cohort; **sparse deep cells widen + report
  n** (non-vacuous fixture — the fixture MUST contain sparse deep cells so the widening branch actually
  fires); `TurnoverCost` port satisfied; `compute_xt_gk_v2` consumes it unchanged; a synthetic where a
  known post-turnover chain yields the expected `V_opp`; mirror-vs-empirical divergence surfaced.

### W3 — Best-practice guards (so the keeper-id trap and V_opp deviation can't silently recur)
- **Convention (doc it):** GK-domain consumers use `player_key` (resolved), never `player_id` (raw, null
  for goal-kicks). Analysis loaders source resolved fields from GOLD marts, not raw bronze.
- **Data-contract test:** `player_key ≥ 99%` non-null on `is_gk_distribution` rows — catches both a future
  wrong-column read AND a resolver regression at CI time. (Live current value: 99.9%.)
- **Canonical GK marts** (note for the lakehouse re-materialization, not a silly-kicks code change here):
  `fct_gk_tracking_actions` (per-action: keeper + xt_gk) + `fct_gk_tracking_stats` (per-keeper). When the
  lakehouse re-materializes on 4.45.0, `xt_gk_v2` belongs there too (`dist_xt_gk_v2_mean` per
  `gk_player_key`) as the durable per-keeper v2 home — mirroring how v1's `dist_xt_gk_mean` already lives
  there. Relay this to the lakehouse session.

### W4 — Construct-validity harness + component decomposition, RUN ON THE FAITHFUL METRIC (final reports)
- Land the S112 `construct_validity_scores` harness (parameterized, GK-domain restricted, `v1_stored`
  baseline via `c.xt_gk`) — but its committed GS/SC reports are computed on the **faithful `V_opp`** and
  labelled **final**, not preliminary. Fold in the earlier review fixes (apples-to-apples baselines,
  non-vacuous fixtures).
- Land the **component decomposition** as a committed report too (pos/pev/ret/dzv shares + `ρ·ΔV`-alone vs
  partial vs full AUC) — recomputed on the faithful metric, so the before/after of the fix is on record.
- **This is where "did the fix work?" is answered honestly** — see §7.

### W5 — Keeper discrimination + face validity (the ACTUAL program goal — never yet tested)
The program exists to fix v1's degeneracy = **keeper non-discrimination (cross-keeper CV ~6%, near
constant).** This is the real SP5 instrument and Jeff's own validity mode (his Bravo/Navas reranking).
- Compute `compute_xt_gk_v2` (production ρ + faithful `V_opp`) over the full GK-distribution cohort,
  group by **`player_key`** (W1), per-keeper mean (per-90 or per-action, ≥N-distribution filter), compute
  **cross-keeper spread / CV**, and compare to **v1's** spread — v1 per-keeper is **already materialized**
  in `fct_gk_tracking_stats.dist_xt_gk_mean` per `gk_player_key` (501 match-keeper rows / 249 keepers; or
  aggregate `c.xt_gk` by `player_key`). **Does v2 separate keepers where v1 was flat?**
- Emit the **ranking** (committed report) for the owner's coaching eye — face validity: do ball-playing /
  sweeper keepers separate sensibly from long-ball keepers?
- **Genuinely two-sided:** if the turnover term still dominates *and* is ~constant across keepers, v2 can
  reproduce v1's near-constancy → it fails the real test too. Report whatever it shows.

### W6 — Secondary faithfulness audit (committed report section; do NOT silently expand scope)
- **V reward interpretation:** we use `E[first-shot xG]`; Jeff §2.1 says *"expected threat over the
  remainder of the possession."* First-shot (our reading) vs cumulative-remainder — state the
  interpretation and its evidence (relates to V's weak realized-xG OOS Spearman 0.03–0.06). **Note it;
  do not re-implement V in this PR** (that's a separate decision if it matters — flag for owner/Jeff).
- **κ:** default 1, `≥1`, sweep `[1,2]` — faithful. **Re-examine the sweep AFTER the faithful `V_opp`**
  (a smaller/faithful `V_opp` shifts the value-added-vs-turnover balance; `κ≥1` may read very
  differently). Report the sweep; **do not retune κ to make a result pass** (§7).
- **PEV dormant** (`p'=p`, receiver-pressure `q` deferred per Jeff §8-step-7) — note the metric currently
  carries no pressure-value-added term (faithful to his sequencing).
- **Anything this audit surfaces as actionable is either done IN this PR or explicitly deferred with a
  written reason** — no silent scope creep, no silent omission.

---

## 3. Honest-reporting guardrail (§7 — the non-negotiable)
The outcomes of W4 and W5 are **not pre-decidable.** If, after the faithful `V_opp`, v2 still does not
beat baselines out-of-sample and/or still does not discriminate keepers, **that is a reportable finding,
not a bug to fix.** Do NOT:
- retune the pre-registered deep-zone gate numbers post-hoc,
- pick κ to make AUC or CV look better,
- drop baselines from the lift `max(...)`,
- filter keepers/cohorts until a spread appears.
Report the faithful metric's real behaviour. A null/negative result on the faithful implementation is
exactly the clean input we need for the Jeff conversation (his question vs our deviation).

## 4. Acceptance checklist (execute end-to-end; no round-trip to the analysis session expected)
- [ ] W1 loader reads `c.player_key`; carried through `prepare_cohort`; no raw-bronze keeper source.
- [ ] W2 empirical `V_opp` is production; deep-cell bin-widening + support-gate + per-zone n; mirror
      demoted to cross-check; injection seam unchanged; all W2 tests green (incl. the non-vacuous
      sparse-deep fixture).
- [ ] W3 `player_key ≥99%` data-contract test in CI; convention documented; canonical-marts note relayed
      to lakehouse.
- [ ] W4 construct-validity + decomposition reports committed, computed on the faithful metric, labelled
      final; baselines fair.
- [ ] W5 keeper-discrimination CV (v2 vs v1) + ranking committed; ≥N filter stated.
- [ ] W6 audit report section committed; each item done-or-deferred-with-reason.
- [ ] 4.45.0 lockstep + ADR-036 amendment (summarize the faithful-`V_opp` result, decomposition
      before/after, keeper-discrimination verdict); lakehouse Hyrum-flagged to re-materialize on 4.45.0.
- [ ] Full non-e2e suite green; ruff clean; **bare pyright 0**; `/final-review`.

## 5. Sequencing (single PR, but build in this order)
`W1 loader` → `W2 faithful V_opp (+tests)` → `W3 guards` → `W4 re-run construct-validity + decomposition`
→ `W5 keeper discrimination` → `W6 audit` → lockstep 4.45.0 → PR. Then the owner reviews the committed
reports and decides the Jeff message (with faithful numbers + the questions genuinely his: κ/turnover
weighting, the validity criterion, V-reward interpretation).
