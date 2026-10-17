# Spec: Visibility-blind HF training harness (ADR-123 Amendment B)

**Status:** DRAFT — awaiting independent review
**Date:** 2026-10-15
**Amends:** ADR-123 (`docs/superpowers/adrs/ADR-123-hf-jobs-parallel-harness.md`), which established the
dataset-agnostic HF Jobs harness. This amendment extends the agnosticism from *data* to *policy*.
**Branch:** `feat/hf-jobs-training-harness` (PR #284). The change lands as one additional commit on the
branch, bundled with the weights commit (owner ruling 2026-10-15), and does **not** amend the pushed
`6f6508a`/`ac875c4`.

---

## 1. Executive summary (for reviewers)

The ADR-123 harness made the in-job trainer code **data-agnostic**: it consumes a staged neutral parquet
dataset via a data path and knows nothing about how the dataset was built (no pining token, no manifest,
no `--providers`). A re-fit campaign (2026-10-15) surfaced that two trainers — `xshot_occurrence` and
`xcross_attempt` — are **not** policy-agnostic: they ship a *visibility-gated* model candidate (one of
`public` / `sc_extended` / `full`), and that selection depends on per-match public-vs-owner
**visibility**, which the in-job code derives from the pining manifest (`_loader_pining.match_visibility`)
and the in-repo redistributability registry (`_corpus.PUBLIC_CORPUS`). On the agnostic `--data-dir` path
there is no manifest, so the in-job code fails closed to *all-private*, the `public` candidate never
forms, and the trainer ships `sc_extended` — a **4× worse** and **different** model than committed
(measured: committed `public` PR-AUC 0.3418/0.3036 for default/position_only vs fresh `sc_extended`
0.0825/0.0770). The fresh weights are not a faithful reproduction, so they must not ship.

**Owner directive (2026-10-15), which this spec implements:** *None of the code that runs on HF may have
any concept of public/private. It receives staged neutral datasets via data paths. If a returned result
must carry a public/private distinction, that distinction must enter the job as an explicit **parameter**
the job received at launch.*

This spec removes every public/private concept from the in-job path and relocates the policy to the
launcher, which passes the arm membership as an explicit neutral job parameter. It is scoped to
`xshot_occurrence` and `xcross_attempt`; `ghost_gk`, `ghost_outfield`, `receiver_model`, and the four
public trainers ship no visibility-gated candidate and are unaffected.

---

## 2. Root cause (three layers, verified in code 2026-10-15)

1. **Proximate.** `scripts/train_xshot_occurrence.py:902` sets `vis = match_visibility(...) if
   args.providers else {}`. On `--data-dir` (no `--providers`) this is `{}`. `scripts/_corpus.py:59`
   `is_public_row` is fail-closed: `visibility.get((prov, mid), "private") == "public"` → with `{}`,
   **every row is private** → `is_public` all-False. `_build_candidate_masks`
   (`train_xshot_occurrence.py:473`) then yields an empty `public` mask, and `_corpus.artifact_label`
   (`:69`) falls through to `sc_extended`. (`train_xcross_attempt.py` mirrors this exactly:
   `:53`/`:483`/`:488`.)

2. **Design.** `xshot`/`xcross` ship one of three visibility-gated candidates via a nested-HPO paired
   comparison on a common public held-out split (`_public_folds`, `run_paired_comparison`). The split and
   the arm masks require per-match visibility — an input the ADR-123 agnostic path deliberately dropped.
   The redistributable set is the in-repo constant `PUBLIC_CORPUS` (`_corpus.py:20`: 20 skillcorner + 7
   idsse `DFL-MAT-*`) — public, committed, not a pining secret — but it is consumed *in-job*.

3. **Deepest — provider-identity conflation.** `PUBLIC_CORPUS` keys the 7 Bundesliga matches under
   provider `idsse`. The staged corpus labels them `source_provider="sportec"` (the `sportec_spadl`
   *converter*, not the licensing identity). So even an in-repo `PUBLIC_CORPUS` lookup would miss them:
   `("sportec", "DFL-MAT-*") ∉ PUBLIC_CORPUS`. The committed models, pining-trained as provider `idsse`,
   matched and formed `public`. (This spec removes the in-job dependency on provider names entirely, so
   the conflation no longer affects correctness; see §7 for the optional separate normalization.)

Note: the committed xshot/xcross corpus is **the original public 17** (`_corpus.py:18`: "10 SkillCorner +
7 IDSSE"), i.e. `is_public` is all-True for this corpus; the `public` candidate is the whole 17. The
agnostic path inverted that to all-False.

---

## 3. Invariant (the contract this spec adds)

**I1 — In-job code is visibility-blind.** No module reachable from an HF job (`scripts/train_*.py` and
everything they import at runtime) may import or call `_corpus.PUBLIC_CORPUS`, `_corpus.is_public_row`,
`_corpus.assert_public_corpus`, `_corpus.artifact_label`, or `_loader_pining.match_visibility`, nor
otherwise encode a public/private rule. The public/owner taxonomy is launcher/off-HF only.

**I2 — Policy enters as an explicit job parameter.** Any public/private distinction a job's results must
carry is supplied at launch as a neutral parameter (a path to a plain JSON file of match keys / arm
memberships). The job applies it mechanically and never recomputes it.

**I3 — Datasets stay purely neutral.** The staged parquet carries no visibility column. Visibility is a
per-run launch parameter, not dataset content (a dataset is reusable across runs that classify its
matches differently).

**I4 — Fail-loud, not fail-closed-silent.** A visibility-gated trainer invoked with no arm parameter
raises an explicit error. It must never silently treat the whole corpus as one arm (the current
fail-closed-all-private behaviour is what produced the wrong model without failing).

These are CI-gated (§8).

---

## 4. Design

### 4.1 The job parameter

The launcher writes a neutral JSON **arm-membership** file and passes its path to the job, exactly as it
already does for the xcross probe (`--probe-data-dir /tmp/probe`). Proposed flag: `--arms /tmp/arms.json`.

Schema (opaque to the job — the job never interprets the names, only the memberships):

```json
{
  "arms": {
    "public":      ["skillcorner__1874553", "...", "idsse__DFL-MAT-J03WMX", "..."],
    "sc_extended": ["...superset of public..."],
    "full":        ["...superset of sc_extended..."]
  },
  "ship_order": ["public", "sc_extended", "full"]
}
```

- Keys are the dataset's own match keys (`<source_provider>__<match_id>`, the same key the trainer's
  `for_each`/`_iter_matches_from_dir` already uses — §4.2), so membership is matched mechanically against
  `match_ids`/`groups` with no provider-name or visibility logic in-job.
- The launcher includes only the arms that are non-empty for the staged corpus (e.g. a corpus with no
  gradientsports omits `full`). The job forms exactly the arms present.
- `ship_order` is the deterministic preference order the ship decision walks (the existing semantics:
  prefer the simplest/most-public admissible arm). Encoding it in the parameter keeps the *policy*
  (which arm is preferred) with the launcher.

**Minimal equivalent for the common case.** When a corpus is entirely one arm (the public-17 case: all
17 are `public`), the file is `{"arms": {"public": [all 17 keys]}, "ship_order": ["public"]}`. The job
forms a single `public` arm over all rows — reproducing the committed `public` candidate.

The exact parameter surface (single `--arms` JSON vs. a smaller `--public-keys` list plus a convention
for the supersets) is a design sub-decision for the plan; the invariants (I1–I4) hold under either. The
recommendation is the single `--arms` JSON because it carries the whole arm structure explicitly and the
job needs zero provider knowledge to build every mask.

### 4.2 In-job changes (`train_xshot_occurrence.py`, `train_xcross_attempt.py`)

1. **Remove** from the runtime path: `from _corpus import assert_public_corpus, is_public_row` and
   `from _loader_pining import match_visibility` (xshot `:895`/`:896`/`:902`/`:904`/`:905`; xcross
   mirror). These modules must no longer be importable on the HF job (I1).
2. **Add** `--arms PATH` (required on the `--data-dir` path for visibility-gated trainers; see I4).
3. **Build the arm masks from the parameter**, mechanically: for each arm in the file, `mask = np.isin(
   match_keys, arm_members)`. `is_public` becomes simply the `public`-arm mask (a membership, not a
   policy). `_build_candidate_masks` loses its `providers`/`is_sc_private` provider-name logic and instead
   returns the provided arms' masks directly.
4. **`_public_folds`** keeps its mechanics but takes the `public` mask from the parameter.
5. **Artifact labelling + provenance.** Replace the in-job `_corpus.artifact_label(...)` and the
   `all_public`/provider-name inputs to `corpus_identity(...)` with values derived from the parameter: the
   shipped artifact's label is the name of the arm whose mask was shipped (read from the file), and
   `corpus_identity` records the arm name + the match keys — no provider-name or visibility reasoning.
6. The `--providers` (pining) code path is **deleted from these trainers** (it is the only remaining
   in-job importer of `_loader_pining`/`_corpus` visibility). Pining-based training, if ever needed, runs
   off-HF and supplies `--arms` the same way. (Confirm during plan that no non-HF caller depends on the
   in-trainer `--providers` path; if one does, that caller moves to the launcher pattern.)

### 4.3 Launcher changes (`scripts/hf_jobs/launch.py`)

- A new helper computes the arm membership for a variant's staged corpus **locally/off-HF**, using
  `_corpus.PUBLIC_CORPUS` (and, where applicable, provider-name rules / the manifest). This is the *only*
  place the public/private taxonomy lives for the HF path.
- The helper must resolve the §2.3 provider-identity conflation when matching `PUBLIC_CORPUS`: the
  staged corpus's keys (`sportec__DFL-MAT-*`) are mapped to the registry's identity (`idsse__DFL-MAT-*`)
  by the launcher, off-HF. The job receives the already-resolved key list and does no mapping.
- `build_phase_command` writes `/tmp/arms.json` and threads `--arms /tmp/arms.json` for the
  visibility-gated trainers (a `TrainerSpec` flag, e.g. `visibility_gated: bool`, mirrors `probe: bool`).
- Registry: mark `xshot_occurrence` and `xcross_attempt` `visibility_gated=True`.

### 4.4 Dataset (unchanged)

`hf-refit-xshot17` and the xcross probe stay as-is (neutral parquet). No re-stage for visibility is
required — visibility is a parameter (I3). (A re-stage is needed only if the plan also normalizes
`source_provider`; see §7, which is optional and independent.)

---

## 5. Faithfulness

With `--arms` supplying `public` = all 17 keys (the committed public-17 corpus is entirely public):
`is_public` all-True → the `public` candidate forms over all rows → the nested-HPO paired comparison runs
as committed → ships `public`. Expected fresh `public` PR-AUC ≈ committed (0.34 default / 0.30
position_only), confirmed by CV-parity (fresh vs committed `candidates.public.metrics.pr_auc`). If the
fresh `public` PR-AUC does not match within the per-trainer δ, the plan investigates before shipping (the
staged corpus may differ from the committed corpus beyond the label — e.g. the §2.3 conflation changing
feature extraction — which CV-parity will reveal).

The DGX e2e gate (ADR-123 §8) is extended to assert the **shipped candidate name equals the committed
shipped candidate** (`public`), not merely that the pipeline exits 0 — the gap that let GATE-1 pass a
`sc_extended` ship as "expected single-candidate".

---

## 6. Scope — every caller of the changed seams (the floor, per house rule)

- **Visibility-gated trainers (changed):** `scripts/train_xshot_occurrence.py`,
  `scripts/train_xcross_attempt.py` — the `is_public` derivation, `_build_candidate_masks`,
  `artifact_label`/`corpus_identity` inputs, and the `--providers` path.
- **Unaffected trainers (verified):** `train_ghost_gk`, `train_ghost_outfield` (CV is provider-agnostic
  `cv_mae`/`cv_mae_euclidean`, no visibility arm), `train_receiver_model` (`top1_cv`, no arm —
  CV-parity exact: fresh==committed `0.5096914124655281`), and the four public trainers (no arm).
- **`scripts/_corpus.py` / `scripts/_loader_pining.py`:** remain for the launcher + any off-HF/DGX use;
  the plan enumerates every remaining importer and confirms none is reachable from an HF job after the
  change (the I1 CI gate makes this a build-time guarantee).
- **`scripts/hf_jobs/launch.py`, `registry.py`:** add the arm-membership helper, `--arms` threading, the
  `visibility_gated` flag.
- **`scripts/hf_jobs/headtohead.py`:** the owner head-to-head eval is already being replaced by CV-parity
  (separate owner ruling); this spec does not re-touch it beyond noting the two land together.

The plan must list each changed function with before/after and every caller, both directions.

---

## 7. Optional, independent: `source_provider` identity normalization

The `source_provider="sportec"` label on idsse/DFL matches conflates the SPADL *converter* with the data
*provider/licensing identity*. This spec does not require fixing it (the launcher resolves keys off-HF,
and the job is provider-blind). It is recorded here as a latent trap; normalizing it (so `source_provider`
is the provider identity, with the converter recorded separately) is a candidate for its own ADR and is
**out of scope** for this change unless CV-parity in §5 shows the conflation also perturbs features.

---

## 8. Testing (TDD, CI-gated)

- **I1 import gate (new):** an AST/import test asserts that importing each `scripts/train_<visibility
  trainer>.py` and transitively the modules an HF job loads does **not** import
  `_corpus.{PUBLIC_CORPUS,is_public_row,assert_public_corpus,artifact_label}` or
  `_loader_pining.match_visibility`. (Mirror of the ADR-123 agnostic negative tests
  `test_xcross_threads_agnostic...`/`test_receiver_is_dataset_agnostic...`.)
  **Gold-standard severance (HFVB-IMPL-01, impl-review r1):** the gate is a PURE STATIC closure with **no
  infra-leaf exemption**. The one pining-reaching helper on the shared ADR-052 seam —
  `_partition.list_match_ids` — is relocated to `scripts/_partition_pining.py` (maintainer-only, the
  `--providers` path), so the trainer closure (`train_* -> _driver -> _partition`) no longer reaches
  `_loader_pining` at all. There is therefore nothing to exempt and no runtime-reachability reasoning: an
  earlier draft trusted `_driver`/`_partition` as leaves (the lazy pining branch is unexecuted on
  `--data-dir`), which reintroduced the exact reachability reasoning I1 exists to eliminate. An anti-rot test
  (`test_adr052_seam_is_traversed_not_exempted`) asserts the seam IS traversed, carries no banned symbol, and
  that `_partition_pining` is NOT in the in-job closure. Callers of `list_match_ids` (maintainer corpus
  drivers) import it from `_partition_pining`; `providers_for_slice`/`worker_tag`/`partition_keys` stay in
  `_partition`.
- **I2/arm mechanics:** given an `--arms` file, the trainer builds the exact masks (membership), ships the
  expected arm, and labels the artifact from the parameter — unit-tested on a synthetic ≥2-arm fixture.
- **I4 fail-loud:** a visibility-gated trainer on `--data-dir` with no `--arms` raises (not silent
  all-private).
- **Faithfulness:** a small fixture where `public` = all rows reproduces a single `public` ship (the
  public-17 shape).
- **Launcher:** `build_phase_command` writes `/tmp/arms.json` and threads `--arms` for
  `visibility_gated` specs only; golden-tested.
- **DGX e2e (extended):** shipped-candidate-name == committed for xshot/xcross (§5).
- Full suite green under `.venv` (py3.10.19); CI authoritative.

---

## 9. Provenance, commit, re-fit

- One additional commit on `feat/hf-jobs-training-harness`, bundled with the weights commit (owner ruling
  2026-10-15). Pushed `6f6508a`/`ac875c4` are **not** amended (amending would orphan the public/receiver
  weights' `training_commit=ac875c4` stamp).
- The in-job change moves the branch tip; the re-fit xshot/xcross jobs clone the new tip, so their
  weights' `training_commit` is the new commit — recorded per-weight in each `metrics.json`, all
  main-reachable after the non-squash merge (ADR-123 provenance model). Mixed `training_commit` across
  weights (public/receiver at `ac875c4`, xshot/xcross at the new commit, ghost wherever its run lands) is
  honest and expected.
- Re-fit xshot + xcross (HF-agnostic, with `--arms`) → CV-parity confirm `public` PR-AUC ≈ committed →
  ship. The `sc_extended` fits from the 2026-10-15 run are discarded.

---

## 10. Risks / open questions

- **R1 — arm parameter shape.** Single `--arms` JSON vs. `--public-keys` + superset convention. Spec
  recommends `--arms` (fullest, zero in-job provider knowledge); plan finalizes.
- **R2 — `--providers` path removal.** Deleting the in-trainer pining path is the cleanest I1 guarantee;
  the plan must confirm no non-HF caller depends on it (grep + the e2e). If one does, it adopts the
  launcher pattern rather than blocking the removal.
- **R3 — CV-parity residual.** If fresh `public` PR-AUC does not match committed within δ after the fix,
  the staged corpus differs beyond the label (the §7 conflation may perturb features); the plan gates
  shipping on CV-parity and escalates rather than shipping a near-miss.
- **R4 — scope creep into `_corpus`/`_loader_pining`.** These stay; only their *reachability from an HF
  job* changes. The I1 gate, not a rewrite, enforces it.
