# ADR-126: Trained-artifact integrity gate + xShot label-filter defect

**Status:** Accepted (number provisional — re-confirm next-free at the commit gate; see "ADR/version ladder")
**Date:** 2026-10-16
**Spec:** `docs/superpowers/specs/2026-10-15-trained-artifact-integrity-xshot-label-defect-design.md`
**Plan:** `docs/superpowers/plans/2026-10-15-trained-artifact-integrity-xshot-label-defect-plan.md`

## Context

A 2026-10-15 re-fit-campaign e2e found the shipped xShot-occurrence weights produced a model 4× different
from a fresh fit on an identical-row corpus (committed `public` PR-AUC 0.3418 at a **22.2%** positive rate
vs fresh 0.0824 at **1.54%**, both `n_rows=156106`). Root cause, reproduced to the integer:

- `build_xshot_labels` labelled "a shot by the in-possession team within 1 s". The shot-type filter lived
  in `prepare_xshot_training_data` behind `if "type_name" in shots.columns:`. The production (pining)
  training path feeds the **full SPADL actions table** (`_loader_pining.pining_source` → `m.actions`), which
  carries `type_id` but **no `type_name`**. The guard was therefore False, no filter applied, and
  `shots_f = ALL actions` — so the label silently became "**any** action within 1 s", inflating positives
  ~13×.
- Physical proof: for skillcorner-1925299 (21 shots, 10 fps, 1 s) the label admits ≤ ~210 positives (≤ 927
  even at a 10 s horizon); the committed shard had **1449**. Feeding the full 1283-action table (no
  `type_name`) reproduces 1449 exactly; `actions[type_id == 11]` recovers 109 (the correct `shots.parquet`
  result). The 5-month 17–28% history is the defect's footprint, not evidence it was correct.
- The defect is **isolated**: `build_xcross_labels` filters by `type_id` (`_xcross_attempt.py:365`), and
  `_ghost_gk`, `_elastic_sync`, `calibration/_carrier_objective` all have a `type_id` fallback. xShot was the
  lone silent-bypass — which is why xCross reproduced its committed metrics byte-for-byte on the same corpus.

Second finding: the campaign's "owner floor" was **vacuous** — it never compared a fresh fit to the shipped
artifact — so no re-fit had been integrity-checked. Notably the defective 22% xShot still PASSED the
existing `pr_auc_gt_base_rate` acceptance check (0.3418 > 0.2220).

## Decision

1. **Fix the filter at the single robust site.** The shot-type filter moves **into** `build_xshot_labels`
   and selects by `type_id` (`{actiontype_id[t] for t in shot_types}`), mirroring `build_xcross_labels`.
   The fragile `type_name`-presence guard in `prepare_xshot_training_data` is deleted. No caller can bypass
   the filter; a `type_name`-less actions table is handled correctly. Inference (`add_xshot_occurrence`) is
   unaffected (it scores, it does not build labels).

2. **Add a layered trained-artifact integrity gate** (`scripts/validate_trained_artifact_integrity.py`,
   `silly_kicks/tracking/_artifact_integrity.py`):
   - **Reproduction tripwire** — a fresh extraction+fit must reproduce the shipped `metrics.json` headline
     (`n_rows`/`n_positive` exact, candidate `pr_auc`/`positive_rate` within a relative δ). Mismatch =
     hard stop.
   - **Construct-validity on the absolute metrics** — a **physically-derived** base-rate plausibility band
     plus `pr_auc > positive_rate` (lift), `brier < base_rate_brier` (calibration), and enough usable folds.

   The gate is never "matches the committed `pr_auc`" alone — a defective baseline passes lift; only the
   band + tripwire catch it.

3. **Why the base-rate band is DERIVED, not a fixed constant.** The maximum possible per-row positive rate
   for an occurrence label is `n_events · (fps · horizon) / n_rows` (the most positive frames an event set
   can flag, over the extracted rows). The band upper bound is `tol ×` that physical maximum, computed per
   artifact from its own `n_events`/`fps`/`n_rows`/`horizon`. A fixed `0.05` was ~2× the xShot physical
   `~0.025`, so a 2–3× inflation would slip; the derived bound closes that. **`tol = 1.5`** is the only
   tunable: it leaves headroom for the share of pre-event frames that fall inside the domain filter while
   still rejecting any ≥1.5× inflation. For a mixed-fps corpus, pass `fps = n_rows / total_duration_s`
   (then the bound is fps-independent).

4. **Audit every campaign artifact against the gate (I4), the floor not a follow-up.** xShot NO-GO
   (pre-fix) → re-fit; xCross GO; receiver / ghost×4 / ghost_outfield / the 4 public-wave trainers re-audited
   (the vacuous floor validated none). Any further NO-GO stops and is surfaced — scope to extend is the
   owner's.

5. **Re-fit the defective xShot weights** on the correct corpus (the staged `shots.parquet` carries
   `type_id`, and post-fix the actions path is also correct). Expected corrected base rate ~1.5% with ~5×
   PR-AUC lift.

## Consequences

- **Hyrum's Law.** The shipped xShot output distribution changes materially (training positive rate 22% →
  ~1.5%; the `xshot_occurrence` column / `add_xshot_occurrence` output shifts broadly). Verified in-repo: no
  first-party module consumes the xShot *probability* — `causal/opportunities.py` reuses only
  `extract_xshot_features` (unchanged by the label fix) and `xsuccess` *copied* the XGBoost precedent; so the
  affected consumer is the external lakehouse mart, recorded in `docs/PRIVATE_CONSUMERS.md`. Lakehouse
  re-materialization tracking is explicitly out of scope (owner ruling 2026-10-15).
- **Commit sequencing (forced by ADR-037).** The audit-verdict `metrics.json` and the re-fit weights carry
  `run_tree_dirty: false` only when produced against a clean (committed) tree, so they land in a **second**
  owner-gated commit after the code commit. This is the GS-dedup precedent, not micro-commits — both are
  fully-tested coherent states.
- **ADR/version ladder (2026-10-16, provisional).** ADR-124 = `ball-contact-detection` (merged `main`,
  4.136.0); ADR-125 = `gs-ball-contact-dedup` (PR #286, 4.137.0); this is **ADR-126 / 4.138.0**, both to be
  re-confirmed next-free at the commit gate (they ride PR #286's merge order).
- A permanent trained-artifact integrity check now exists; making it a standing CI floor for all trained
  artifacts (not just campaign ones) is recommended and left to a follow-up decision.
