# xT-GK pre-Jeff verification + emit resolved coords — handoff (analysis side → silly-kicks)

**Date:** 2026-06-29 · **From:** xT-GK analysis side (Karsten) · **Status:** Requested. Lakehouse work is HELD
until this lands (the persist-coords schema migration depends on item 1 here).

---

## ⚑ FOLLOW-UP (2026-06-29, after 4.36.0) — Item 1 shipped; 3 verification items still open

4.36.0 correctly delivered **Item 1** (resolved-coord columns — clean: `_COORD_COLS`, kept out of `_OUTPUT_COLS`,
tie-to-value test). But the release covered the *enhancement*, not the *audits*. **Item 5 is now closed by the
analysis side** (below); **Items 2, 4, and the guard-test half of Item 3 are still open.** No formula change is
involved — this is audit/test work + one doc correction.

**Item 5 (DZV magnitude) — RESOLVED by the analysis side; one doc fix for you.**
Reconciled against the live corrected grid: keeper-zone DZV computes to ~0.020 (deep `V_GK = xT·φ ≈ 0.019–0.02`),
matching production **+0.021**. The form is faithful; the 2× vs Jeff's 0.009 is **grid amplitude** — our corrected
global grid's deep third (raw xT ≈ 0.0085) is ~2× Jeff's implied deep value (~0.004). Not a bug; within his "sanity
band, not a gate." **Action for you:** the ADR/CHANGELOG say "DZV ~0.009 / measured O(0.01) on unit fixtures
(realistic deep `V_GK` 0.005–0.01)" — but **production deep `V_GK` ≈ 0.02 and DZV ≈ +0.021**. Correct that framing so
the docs aren't misleading (and note it's a concrete instance of Item 2).

**Item 2 (test↔production parity) — STILL OPEN.** A concrete gap is already proven: the DZV unit fixtures assume deep
`V_GK` 0.005–0.01, but the real corrected grid gives ~0.02 (2–4× higher). The fixtures understate production
amplitude — which is exactly why the ADR's "~0.009" doesn't match the live +0.021. **Do:** add a
production-realistic-amplitude fixture (deep raw xT ≈ 0.0085, φ ≈ 2.2 → `V_GK` ≈ 0.02), and audit the rest of the
input contract (action/frame schema, **id dtypes**, coord convention) against what the lakehouse actually feeds
`compute_xt_gk`.

**Item 4 (golden composite test) — STILL OPEN.** The tie-to-value test checks `base == −xT*(origin)` only. Need a
full hand-worked composite: one GK distribution, fully-specified inputs (coords, a known small grid, known ρ, known
p, params), hand-compute `base/pev/rav/dzv/T/composite`, assert exact. Item 1's persisted coords make it easy to seed
from a real row.

**Item 3 (orientation) — guard-test half STILL OPEN.** The ADR documents the LTR assumption (good), but there's no
test that a mis-oriented input fails loud. **Do:** add a guard test — a non-LTR / mirrored input is rejected or
produces an asserted-wrong result — so an orientation regression can't pass silently. (The live end-to-end
verification is the analysis side's, pending the lakehouse persist-coords migration.)

**When Items 2 + 4 + the 3-guard land, ping the analysis side.** No release strictly required for these (they're
tests/docs) unless you bundle a fixture change that affects a shipped artifact.

## Why this exists (read first)
We are about to take an xT-GK finding to **Jeffrey Eyestone** (the framework's author). Before we involve him,
Karsten wants 100% confidence that the anomalies we're seeing are **real properties of the metric/data, not
artifacts of our implementation**. This is warranted: in the last week we've been bitten twice by *our own* pipeline
(a stale, non-directional global xT grid — lakehouse ADR-063; and two wrong root-cause calls before we found it).
Orientation/seam bugs have been our recurring failure mode, so we want the silly-kicks side audited end-to-end and
one auditability enhancement made, before any external conversation.

Background facts the tasks depend on:
- v4.35.0 shipped Eyestone's Q1–Q3 fidelity fix: PEV gain on the revalued surface `V_GK = xT·φ`, DZV =
  Option-A increment `(M−1)·V_GK` with the canonical `φ(z,d)=α·(1−d/D_max)^(−β)` (α=2.1, β=0.8), `v_def` retired.
- After the lakehouse fixed the stale grid (now directional, att/def ≈ 9), the WC2022 cohort re-ran and xt_gk is
  now **near-constant across keepers** (full-cohort spread 0.0036, CV 6%, top four within 0.0001). The two big
  terms (`base`, `dzv`) are ~constant for all keepers. We need to be sure that's the metric, not a bug.
- One number does NOT reconcile with Jeff: our **DZV averages +0.021/action vs his published La Liga anchor
  ~0.009/action** (his DZV ≈ +0.27/match ÷ ~30 distributions). See item 5.

Relevant code: `silly_kicks/tracking/_xt_gk.py` (`compute_xt_gk`, `_OUTPUT_COLS`, `_PROVENANCE_COLS`),
`silly_kicks/tracking/_gk_geometry.py` (`resolve_gk_geometry`).

---

## Item 1 — Emit the RESOLVED coordinates (enhancement; gates the lakehouse work)
**Today** `compute_xt_gk` emits the value cols + provenance *source tags* (`xt_gk_origin_source`,
`xt_gk_dest_source`, `xt_gk_origin_confidence`, …) but **not the resolved coordinate values**. Inside `compute_xt_gk`
the geometry frame already has `origin_x/origin_y/dest_x/dest_y` (it's what `sx/sy/ex/ey` and every grid lookup use;
for goal-kicks ~67% of these are *imputed*, not the native `start_x/end_x`).

**Do:** surface them as four new output columns for in-scope rows (NaN off-scope):
`xt_gk_origin_x`, `xt_gk_origin_y`, `xt_gk_dest_x`, `xt_gk_dest_y` (from `geom`'s resolved coords).
- Add to the output contract (`_OUTPUT_COLS` or a parallel coords set) + `XtGkReport` if relevant.
- **Test:** assert the emitted coords are exactly the ones the lookups used — e.g. `base == −xT*(origin_x, origin_y)`
  on the convolved grid for a sample row (ties the persisted coords to the computed value).

**Why:** (a) every xt_gk row becomes externally auditable — anyone (incl. Jeff, or us post-hoc) can see exactly
which coordinates produced each value, especially the imputed goal-kick origins; (b) it's the enabler for the
deferred cheap "xt_gk projection" refresh. The lakehouse schema migration to carry these columns is HELD until this
ships.

## Item 2 — Test–production parity audit
**Question:** do the xT-GK test fixtures match the **actual shape and conventions** of what the lakehouse feeds in
production? Green tests on unrepresentative fixtures are exactly how the stale grid and the mocked-Spark guard tests
gave false confidence.
**Do:** document a comparison of the test fixtures' input contract vs the live lakehouse input contract — action
schema (columns + **id dtypes**), frame schema, `data_source`/provider values, pitch dims (105×68), coordinate units.
Flag any divergence where a fixture is shaped differently from production. (Don't assume; check against the real
`fct_action_values` / frame schemas the lakehouse passes to `compute_xt_gk`.)

## Item 3 — Orientation / contract audit end-to-end (highest-risk item)
xt_gk assumes **LTR-normalized actions** (acting team attacks +x) and an **LTR grid**. The grid-orientation bug just
cost us a week; confirm the *actions and frames* feeding xt_gk are LTR-consistent too — `base = −xT*(origin)` and the
PEV gain are only meaningful if action coords agree with the grid's orientation.
**Do:** (a) document exactly what orientation `compute_xt_gk` *assumes* for actions and for frames (and where it
relies on it — `_grid_value`, `_counter_value`'s point-reflection, `resolve_gk_geometry`); (b) confirm that matches
the lakehouse gold convention (ADR-022 LTR); (c) add/strengthen a test that a non-LTR or mis-oriented input is either
rejected or produces an asserted-wrong result, so a future orientation regression fails loud.

## Item 4 — Golden hand-worked `xt_gk`
**Do:** add a golden integration test — one (or a few) GK distribution(s) with fully-specified inputs (origin/dest
coords, a known small xT grid, known pressure ρ, known completion p, params) — hand-compute `base/pev/rav/dzv/T/
composite`, and assert the code reproduces them **exactly**. This is the only thing that proves the composite
*arithmetic* end-to-end in production (unit tests pass without guaranteeing the assembled value is right). Item 1's
persisted coords make it easy to seed this from a real row.

## Item 5 — Reconcile the DZV magnitude against Jeff's anchor
On the corrected directional grid, **DZV ≈ +0.021/action and now dominates the composite**, vs Jeff's published
**~0.009/action**. Since the metric is DZV-dominated, if DZV is mis-scaled the whole composite (and our degeneracy
read) is suspect.
**Do:** decompose the 2.3× gap and document the cause — candidates: (a) the φ params (α=2.1/β=0.8) being higher than
Jeff's effective scaling; (b) the Option-A `(M−1)·V_GK` increment vs whatever normalization produced his number;
(c) our grid amplitude (max ~0.20) vs his; (d) genuinely fine (different grid/era — his anchor isn't a hard target,
per his own "treat dzv_avg ≈ +0.01 as a sanity band"). Conclusion needed: **is our DZV scale faithful to Jeff's
intent, or is there a scaling discrepancy to fix?** Do NOT change the PEV/DZV *forms* (Jeff-confirmed) unless this
surfaces an actual bug.

---

## Out of scope / sequencing
- **Do not** alter the PEV or DZV functional forms or the Option-B base — all Eyestone-confirmed (ADR-024 / the
  2026-06 email thread). This handoff is verification + the coord-emit enhancement only.
- Lakehouse persist-coords migration is **held** until item 1 ships; the rest of the lakehouse is unaffected.
- Independent of this: SkillCorner La Liga (Real Madrid, 2023–25) tracking is being ingested for a separate external
  validation run — not part of this handoff, but it's why end-to-end correctness matters now.

## Acceptance
Item 1 columns emitted + tied-to-value test green; items 2/3 documented with any mismatch surfaced and the LTR
contract confirmed + guarded; item 4 golden test green; item 5 DZV magnitude explained (faithful, or a fix proposed).
When done, ping the analysis side — we then (a) verify orientation end-to-end against the live data using the
persisted coords, (b) green-light the lakehouse persist-coords migration, (c) decide what, if anything, goes to Jeff.
