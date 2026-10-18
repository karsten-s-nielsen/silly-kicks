# Spec: Trained-artifact integrity gate + xShot label-filter defect (ADR-126, provisional)

**Status:** DRAFT — review round 1 returned (A APPROVE, B REQUEST-CHANGES narrow); revised r2
**Date:** 2026-10-15
**Branch:** `feat/hf-jobs-training-harness` (PR #284)
**ADR number (provisional — confirm next-free at commit):** **ADR-126.** Verified ladder 2026-10-15:
ADR-124 = `ball-contact-detection` (merged `origin/main`, 4.136.0); ADR-125 = `gs-ball-contact-dedup`
(PR #286 open, 4.137.0, also the original session's checked-out branch); **126 is the next free** (no
branch / PR / spec / original-folder reservation claims 126–127). Marked provisional per the sibling-spec
convention because 125 rides an unmerged PR; re-confirm at the commit gate.
**Relationship to prior work:** This spec is a sibling of
`docs/superpowers/specs/2026-10-15-hf-visibility-blind-harness-design.md` (ADR-123 Amendment B, already
reviewed/approved/implemented). The visibility-blind e2e that validated that work **surfaced a distinct,
pre-existing library defect** in the xShot-occurrence label and the absence of any real cross-fit
integrity gate. This spec addresses that defect and the gate. The two bodies of work land together on the
branch (see §7; commit structure is a deferred owner decision — §9.1).

---

## 1. Executive summary (for reviewers)

A re-fit campaign e2e (2026-10-15, on the DGX, current code `ac875c4`) found that a fresh xShot-occurrence
fit on a staged neutral corpus produces a model **4× different** from the shipped one (committed `public`
PR-AUC 0.3418 at a 22.2% positive rate vs fresh 0.0824 at a 1.54% positive rate, **identical `n_rows`
156106**). Root-cause (reproduced to the integer, §2) is a **silent shot-type-filter bypass** in
`prepare_xshot_training_data`:

> `_xshot_occurrence.py:917-921` filters the shot set with `if "type_name" in shots.columns:`. The
> production (pining) training path feeds the **full SPADL actions table** (`pining_source` →
> `m.actions`, `_loader_pining.py:450`), which carries `type_id` but **no `type_name`** column. The guard
> is therefore False, no type filter applies, and `shots_f = ALL actions`. The xShot label — "a shot by
> the in-possession team occurs within 1 s" — silently becomes "**any action** (pass, dribble, …) within
> 1 s", inflating the positive rate ~13×.

**Direction of the defect is proven, not argued.** For one match (skillcorner-1925299): 21 real shots, at
10 fps, with a 1 s horizon, admit **≤ ~210** positive frames (and ≤ 927 even at a 10 s horizon); the
committed shard has **1449**. 1449 is physically impossible for 21 shots. Feeding the full 1283-action
table (no `type_name`) reproduces **1449 exactly**; filtering by `type_id == 11` recovers **109** (==the
correct `shots.parquet` result). So the **committed xShot weights — and the entire 5-month pining-trained
lineage (consistently 17–28%) — are defective**; the `--data-dir` staged path (a `shots.parquet` that
*does* carry `type_name`, already filtered to shots) is correct. The campaign re-fit therefore **fixes a
latent production defect**, but the defect must also be fixed **in the library** so the pining path cannot
reproduce it.

The defect is **isolated**: every sibling tracking/calibration extractor that filters actions by type
(`_xcross_attempt.py:365`, `_ghost_gk.py:556/647`, `_elastic_sync.py:647`,
`calibration/_carrier_objective.py:68`) filters by `type_id` or has a `type_id` fallback. xShot is the lone
silent-bypass. This is *why* xCross reproduced its committed metrics byte-for-byte on the same corpus.

The second finding: the campaign's "owner floor" was **vacuous** (it did not compare a fresh fit to the
shipped artifact), so **no re-fit in this campaign has been integrity-checked**. This spec adds a
**layered trained-artifact integrity gate** (faithful-reproduction tripwire + construct-validity on the
absolute metrics, including a base-rate plausibility band that would have caught 22.2%) and **audits every
committed artifact the campaign touches** against it.

**This spec does NOT** change any metric's intended definition, the SPADL schema, the HF harness's
visibility-blind design, or any model the audit finds already correct (xCross passes). It fixes one label
filter, adds one gate, re-fits the one defective model family, and audits the rest.

---

## 2. Root cause (verified by reproduction, 2026-10-15)

All numbers below were produced by executing current code (`silly_kicks 4.135.0`, DGX scratch at `ac875c4`
+ the visibility-blind patch) on the staged `pergame-17` corpus, and by reading the committed artifacts.

### 2.1 The observation

| model | committed (shipped, `run_commit a34e992`, 4.134.0) | fresh `--data-dir` fit (`ac875c4`) | gate |
|---|---|---|---|
| xShot | `n_rows` 156106, `n_pos` **34649** (22.2%), PR-AUC **0.3418** | `n_rows` 156106, `n_pos` **2405** (1.54%), PR-AUC **0.0824** | **FAIL** |
| xCross | `n_rows` 91999, `n_pos` 2849 (3.10%), PR-AUC 0.0873 | `n_rows` 91999, `n_pos` 2849 (3.10%), PR-AUC 0.0873 | **PASS (exact)** |

Per provider (committed shards `940328826482f127` vs fresh), **`n_rows` is identical to the integer**
(skillcorner 75310 = 75310; Bundesliga 80796 = 80796) while positives differ ~12× (skillcorner 15.9% vs
1.33%; Bundesliga 28.0% vs 1.74%). Identical rows ⇒ same extraction population, same frames; the gap is
purely in the label `_y`.

### 2.2 The mechanism

- `pining_source` (`scripts/_loader_pining.py:385,450`) yields `(provider, match_id, m.actions, frames,
  home)` — the **full SPADL actions table** as the second positional (`actions_or_shots` in the trainer,
  `train_xshot_occurrence.py:111`). The `--data-dir` path (`:60-64`) yields a pre-filtered
  `shots.parquet` instead.
- The staged `actions.parquet` has columns `[…, type_id, result_id, …]` and **no `type_name`**
  (`type_id` value_counts: `11` (shot) = 21 rows out of 1283). The staged `shots.parquet` has
  `type_name` (`{'shot': 21}`) and is already 21 rows.
- `prepare_xshot_training_data` (`silly_kicks/tracking/_xshot_occurrence.py:917-921`):

  ```python
  shots_f = shots
  if "type_name" in shots.columns:            # FALSE for the actions table → bypass
      shots_f = shots[shots["type_name"].isin(types)]
  elif not types:                             # types is non-empty → dead branch
      shots_f = shots.iloc[0:0]
  labels = build_xshot_labels(fidx, shots_f, horizon_seconds=horizon_seconds)
  ```

  Fed the actions table (no `type_name`), neither branch fires → `shots_f = shots` = **all 1283 actions**.

- `build_xshot_labels` → `_build_occurrence_labels` then marks a frame positive iff a same-(game, period,
  team) **action** falls in `[t, t+horizon]` — i.e. *any* action, not just shots.

### 2.3 The proof (exact reproduction + physical bound)

On staged skillcorner-1925299 (8394 attacking-third in-possession rows, 21 shots, 10 fps):

```
shots.parquet   (21, type_name=shot)   -> 109    [--data-dir path            = CORRECT]
FULL actions    (1283, NO type_name)   -> 1449   [pining path, bypass        = committed shard EXACTLY]
actions[type_id==11] (21)              -> 109    [type_id filter             = FIX, == shots.parquet]
```

Physical bound: 21 shots × ~10 frames/s × 1 s ≈ 210 max positives; a horizon sweep on the correct label
tops out at 927 even at 10 s. 1449 is unreachable with 21 shots ⇒ the committed label counted far more
than 21 events. The reproduction shows those events are the 1262 non-shot actions.

### 2.4 Why xCross is immune (and all siblings are robust)

- `build_xcross_labels` (`_xcross_attempt.py:365-366`) filters by **`type_id`**:
  `type_ids = {_spc.actiontype_id[t] for t in cross_types}; actions[actions["type_id"].isin(type_ids)]`.
  `type_id` is always present → correct on both the actions table and `crosses.parquet` → exact reproduction.
- `_ghost_gk.py:556-560` and `:647-663`, `_elastic_sync.py:647-652`,
  `calibration/_carrier_objective.py:68-74` all have a `type_id` fallback for the `type_name`-absent case.
- **xShot is the only silent-bypass instance in the codebase.**

---

## 3. Invariants (the contracts this spec adds)

- **I1 — xShot filters shots by `type_id`, never by a `type_name`-presence guard.** The shot-type filter
  must apply identically whether the input carries `type_name` or only `type_id`. Behavioural test: feed a
  `type_name`-less actions table and assert the label equals the `type_id`-filtered label (and is < the
  all-actions label). This test must FAIL on `ac875c4` (detection before fix) and PASS after.
- **I2 — No shipped trained artifact has a physically implausible base rate.** Each occurrence-labelled
  model declares a plausible positive-rate band; a fit (fresh or committed) outside it is a hard gate
  failure. xShot "shot within 1 s" cannot sit at 22%.
- **I3 — A shipped artifact is reproducible.** A fresh extraction+fit on the artifact's declared corpus
  reproduces its shipped `metrics.json` headline (`pr_auc`, `positive_rate`, `n_rows`, `n_positive`)
  within a declared relative δ. A mismatch is a HARD STOP requiring adjudication, never an auto-pass. (This
  is the non-vacuous replacement for the campaign floor.)
- **I4 — Every campaign artifact passes I2+I3 before it is adopted/shipped.** The audit is the floor, not
  a follow-up.

---

## 4. Design

### 4.1 Fix the xShot shot-type filter (mirror xCross)

Single-source the robust pattern. Move the shot-type filter **into** `build_xshot_labels` by `type_id`,
exactly as `build_xcross_labels` does, so no caller can bypass it:

```python
# _xshot_occurrence.py
_DEFAULT_SHOT_TYPES = ("shot", "shot_penalty", "shot_freekick")

def build_xshot_labels(frames_index, shots, *, horizon_seconds=1.0,
                       frame_team_col="team_in_possession",
                       shot_types=_DEFAULT_SHOT_TYPES):
    type_ids = {_spc.actiontype_id[t] for t in shot_types}
    shots_f = shots[shots["type_id"].isin(type_ids)][["game_id", "period_id", "team_id", "time_seconds"]]
    return _build_occurrence_labels(frames_index, shots_f, horizon=horizon_seconds,
                                    frame_team_col=frame_team_col)
```

and **delete** the fragile `type_name` guard block at `prepare_xshot_training_data:917-921` (the filter now
lives in `build_xshot_labels`). A `shots.parquet` that is already shot-only still passes (its rows are
`type_id==11`). `type_name`, if present, is ignored (the `type_id` is authoritative).

Optional hardening (reviewer call): factor the `{actiontype_id[t] for t in types}` + `type_id` filter into
a shared helper in `_occurrence_labels.py` used by both `build_xshot_labels` and `build_xcross_labels`, so
the two label families cannot drift again.

#### 4.1.1 Caller sweep of `build_xshot_labels` (house floor — a public-seam change enumerates every caller)

`build_xshot_labels` gains a new input requirement (the `shots`/actions frame MUST carry `type_id`). Every
caller, with before/after evidence:

| caller | file:line | type_id present? | change |
|---|---|---|---|
| `prepare_xshot_training_data` | `_xshot_occurrence.py:922` | **yes** — real actions (`type_id`, 11=shot) and `shots.parquet` both carry it | delete the `:917-921` `type_name` guard block; pass raw `shots` straight to `build_xshot_labels` (which now filters). Before: label = any-action-within-1s (bypass); after: shot-within-1s. |
| `test_label_*` ×5 (times/robust/no-period-bleed/inclusive/turnover) | `tests/tracking/test_xshot_occurrence.py:234,254,264,273,283` | **no** — fixtures are `{game_id, period_id, team_id, time_seconds}` only | add `"type_id": [11]*n` (`actiontype_id["shot"]`) to each `shots` fixture. All fixture rows are shots → filter keeps all → the `assert list(y) == …` expectations are **unchanged**. Without this they raise `KeyError: 'type_id'` after the fix. |
| `test_build_xshot_labels_bit_identical_after_refactor` (R2-L2 golden-master) | `tests/tracking/test_xshot_occurrence.py` | **verify** — add `type_id=11` to its fixture if absent | golden labels unchanged (fixture is all-shots). The golden must still reproduce bit-identically. |
| docstring example | `_xshot_occurrence.py:317` | commented (`>>> #`, not executed) | add `type_id` to the illustrative frame for correctness; no doctest runs. |

No non-test production caller other than `prepare_xshot_training_data` exists (verified by repo grep).
`add_xshot_occurrence` / `compute_xshot_occurrence` are inference paths — they do not call
`build_xshot_labels` and are unaffected.

### 4.2 Layered trained-artifact integrity gate

A new validation module (corpus driver on `scripts/_driver.py`, per `docs/howto/corpus-drivers-runbook.md`)
that, for a model family + its declared corpus, emits a verdict `metrics.json` with the provenance block
(`run_commit`, `run_tree_dirty: false`, `input_contract.digest`; ADR-037/056) and two layers:

1. **Reproduction tripwire (I3).** Re-extract + re-fit on the declared corpus; compare the headline to the
   shipped `metrics.json` within relative δ (default δ = 0.05 on `pr_auc`; exact on `n_rows`/`n_positive`).
   Mismatch ⇒ `reproduced: false` + the diff; GO requires `true`.
2. **Construct-validity on absolute metrics (I2),** following `docs/howto/construct-validity.md` to the
   extent a trained propensity model supports:
   - **base-rate plausibility band** — `positive_rate ∈ [lo, hi]` declared per model (xShot: an upper bound
     derived from `n_shots · horizon · fps / n_rows`; a 22% rate fails). This is the check that catches the
     defect.
   - `pr_auc > positive_rate` (lift; already in `acceptance.pr_auc_gt_base_rate`).
   - calibration: `brier < base_rate_brier` (already in `acceptance.brier_lt_base_rate_brier`).
   - `enough_usable_folds`.

   Where the full three-gate triad (responsiveness / discrimination-vs-null / transfer) is feasible for the
   family, run it; where not, state the scope limit per the howto's caveat pattern.

The gate **never** certifies "matches committed `pr_auc`" as a standalone pass — a defective baseline (the
committed 22% xShot) passed `pr_auc_gt_base_rate`; only the base-rate band + reproduction tripwire catch it.

### 4.3 Audit every campaign artifact (I4)

Run §4.2 against each committed artifact the campaign touches, recording a GO/NO-GO reading per the howto:

| family | prior status | expected |
|---|---|---|
| xShot occurrence (default + position_only) | **defective** (§2) | NO-GO → fix §4.1 + re-fit |
| xCross attempt | reproduces exactly | GO (confirm under the gate) |
| receiver_model | "floor passed" (vacuous floor) | **re-audit** |
| ghost_gk ×4 (default/position_only/sweeper/sweeper_po) | unaudited | audit (type-filter robust, §2.4) |
| ghost_outfield | unaudited | audit |
| the 4 public-wave trainers | "floor passed" (vacuous) | **re-audit** |

NO-GO artifacts are fixed + re-fit; the audit result is the ship gate.

### 4.4 Re-fit the defective xShot weights

Re-fit xShot (default + position_only) on the correct corpus. Two equivalent-after-§4.1 routes — a
`shots.parquet`/`--arms` staged fit (campaign path, already correct because the file carries `type_name`
and is shot-only) **and** the pining `actions` path (correct only after §4.1). The gate (§4.2) certifies the
result. The corrected base rate is ~1.5% with ~5× PR-AUC lift (healthy); the committed 22% is retired.

---

## 5. Downstream impact (Hyrum's Law) — library consumers only

The shipped xShot output distribution changes materially (positive rate 22% → ~1.5%; the
`xshot_occurrence` column and `add_xshot_occurrence` output). Enumerate in-repo consumers
(`docs/PRIVATE_CONSUMERS.md` + `add_xshot_occurrence` / `XShotOccurrenceModel` callers, incl. GKDV
Layer-2) and flag the behavioural change. **Lakehouse re-materialization tracking is explicitly out of
scope (owner ruling 2026-10-15).** The change is a defect fix; whether it is labelled a breaking change for
library consumers is called out in the ADR/CHANGELOG.

---

## 6. Testing (TDD, CI-gated; detection before fix)

1. **I1 behavioural (detection):** a unit test building `fidx` + a synthetic actions table with `type_id`
   but no `type_name`, asserting `build_xshot_labels` counts only `type_id ∈ shot-ids` (and strictly fewer
   than all-actions). Written to FAIL on `ac875c4`, PASS after §4.1.
2. **I2 gate unit:** a CI-safe slice asserting the base-rate band rejects a 22% synthetic metrics blob and
   accepts a ~1.5% one; `pr_auc_gt_base_rate` / `brier` checks.
3. **I3 gate unit:** a reproduction-tripwire unit on a tiny fixture (fresh == shipped within δ passes; a
   perturbed shipped fails). Real-corpus audit is opt-in/`e2e` (`@pytest.mark.slow`, per ADR-023 — one
   dedicated job, not inline).
4. **Regression fixture** pinning the corrected xShot base rate within I2's band.
5. **Existing-caller fixture updates (§4.1.1):** add `type_id=11` to the five `test_label_*` fixtures
   (`test_xshot_occurrence.py:234,254,264,273,283`) and the R2-L2 golden-master fixture; assertions and the
   golden labels are unchanged (all-shots fixtures). These convert a post-fix `KeyError` into a pass and are
   part of this change, not a follow-up.
6. Full suite green under `.venv` (py3.10.19) + `ruff check silly_kicks/ tests/ scripts/` + `--format
   --check` + `pyright` (scoped to changed files locally; CI authoritative).

---

## 7. Provenance, commit, re-fit (owner-gated)

- All docs (this spec, the ADR, plan) travel with the single approval-gated commit — never a separate/early
  commit (house rule).
- **No `git commit`/`push`/`merge`/`tag` without Karsten's explicit per-commit go.** Delivery ends at an
  independent-review handoff on a frozen tree; the author does not run the review.
- One coherent, fully-tested commit for this change (no micro-commits). Version bump `_version.py` →
  **4.138.0 (provisional)** — ladder: 4.136.0 = ball-contact (merged main), 4.137.0 = gs-dedup (PR #286);
  re-confirm next-free at the commit gate; **ADR-126** (see header); CHANGELOG `fix(tracking)` + the harness
  entries.
- Re-fit runs on HF under the $25 ceiling (campaign path) and/or DGX; weights land in the same commit after
  the gate certifies them. No HF spend beyond $25 without explicit go.

---

## 8. Scope boundaries — what this spec does NOT do

- Does not change the SPADL schema, `type_id` assignments, or add `type_name` to the actions table.
- Does not alter the xShot/xCross model architecture, features, horizon, or the visibility-blind harness.
- Does not re-fit or change any artifact the audit certifies GO (e.g. xCross) beyond confirming it.
- Does not touch the ORIGINAL silly-kicks folder, pining tokens, or any owner-tier data path on HF.
- Does not pseudonymize ids or change redistributability policy (ADR-116 stands).

---

## 9. Risks / open questions (for review)

### 9.0 Round-1 review resolution (A APPROVE, B REQUEST-CHANGES — both addressed)

- **ADR number (A TAI-SPEC-01).** A was correct: ADR-124 is taken by merged `ball-contact-detection` on
  `origin/main`. Verified the full ladder across `origin/main`, every remote branch, PR #286, and the
  original session's working folder (`gs-ball-contact-dedup`, max 125). Renumbered to **ADR-126**
  (next-free), marked provisional (header). B's "124 free" read was against the part-deux branch only
  (max 123) and missed merged main — superseded.
- **Caller enumeration (B TAI-SPEC-01).** Added §4.1.1: every `build_xshot_labels` caller enumerated with
  before/after evidence, and the five `test_label_*` fixtures + the R2-L2 golden-master specified to gain
  `type_id=11` (else post-fix `KeyError`). Test task added as §6.5.
- **Version (A/B TAI-SPEC-02).** Pinned to **4.138.0 (provisional)** with the ladder; the 4.137.0 collision
  with gs-dedup (#286) is noted and avoided.

### 9.1 Open

1. **Commit structure (owner decision — DEFERRED, owner 2026-10-15 "decide structure later").** Bundle
   everything into one commit (harness CLI fixes + visibility-blind + xShot fix + gate + audit + ADR), or
   split artifact-integrity from the harness work into two commits on the branch? The combined change is
   large. Version follows the structure: one commit = one `>= 4.138.0` bump; a split = two bumps. Resolve
   before the implementation plan is finalized.
2. **Audit breadth.** Should the integrity gate become a permanent CI floor for *all* trained artifacts
   (not just campaign ones), making the vacuous-floor class unrepresentable going forward? (Recommend yes.)
3. **δ for I3** and the per-model base-rate bands (I2) — proposed values in §4.2; reviewers to confirm the
   bands are tight enough to catch a 13× inflation yet tolerant of seed/n_trials noise.
4. **receiver_model / public-wave re-audit** may surface further NO-GOs; the spec commits to auditing, not
   to a predetermined outcome. If more defects appear, scope is the owner's to extend.
5. The exact `pining_source` call that fed `m.actions` was removed from `train_xshot_occurrence.py` in
   `ac875c4` (now `--data-dir`-only). The library fix (§4.1) is still required because the defect is in
   `build_xshot_labels`/`prepare_xshot_training_data` (the **training label**), which any current or future
   training caller (a restored pining path, a downstream re-fit, `xshot_count_pass`-style label builds)
   can still hit. `add_xshot_occurrence` is an inference/scoring path (it does not build labels) and is
   unaffected by §4.1 — its only exposure is that it loads the defective weights until the re-fit lands.

---

## 10. References

- Root-cause evidence: this session's DGX probes (per-provider shards, horizon sweep, exact reproduction).
- `silly_kicks/tracking/_xshot_occurrence.py:289-326,755-926` (label + extraction);
  `_xcross_attempt.py:352-367` (robust exemplar); `_occurrence_labels.py` (shared helper);
  `scripts/_loader_pining.py:385-450` (`pining_source` → `m.actions`).
- `docs/howto/construct-validity.md`, `docs/howto/authoring-a-metric.md`,
  `docs/howto/corpus-drivers-runbook.md`.
- Sibling spec: `docs/superpowers/specs/2026-10-15-hf-visibility-blind-harness-design.md` (ADR-123 Am. B).
- ADR-037 (dirty-tree refusal), ADR-056 (input_contract digest), ADR-023 (slow gating), ADR-116
  (reversibility publish bar), ADR-019 (id_compat).
