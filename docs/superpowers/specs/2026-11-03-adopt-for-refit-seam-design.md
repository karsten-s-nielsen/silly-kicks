# Adopt-for-refit verify/adopt seam — design

Status: DRAFT r2 (2026-11-03), after A/B spec review. No code yet. Author session: TF-46 Arm-1 clean cycle.

## r1 → r2 (A/B review resolutions)

- **A-01 BLOCKING (corpus-identity digest stamp breaks rehydrate) — REJECTED, resolved in B's favor.** A's
  premise is false: `a.corpus` is a MANIFEST field (`docs/artifacts/manifest.toml:115` `corpus = "public_ids"`
  for xShot), not derived from the fit's stamp, so a digest stamp does NOT flip the tag. `rehydrate_public_corpus_ids`
  (`scripts/_corpus.py:136`) is explicitly built for the visibility-blind digest-always case — it reconstructs
  the exact ids from the canonical registry (`bundled_public_arm_pairs`) and guards on `corpus_n_matches`. B
  verified this ("GOOD, do not change"). Digest-always stands (§6.2 B1a / §6.5-i); the A/B disagreement is
  flagged for the owner, resolved with this evidence.
- **A-02 SHOULD FIX + B-1 BLOCKING (the `_rebaseline` P15 routing) — ACCEPTED.** `_rebaseline` is `DriverSpec`-only
  (`adopt.py:464`); `REBASELINE_FROM` holds only `build_territory_ranking_census` (`:49`) so a trainer KeyErrors
  at `:341`; trainers expose no `input_contract()` (`:337` would AttributeError); and it returns `ok = not commits`
  so it STOPS on the code delta every refit carries. A refit cannot route through it. §6.2 B1b now gates the refit
  on **certification alone** (`_certify`: owner cv-parity; public §6.4 floor) and SKIPS `_integrity`/`reproduces`;
  no `_rebaseline`, no P15.
- **A-03 + B-2 SHOULD FIX (wrong test gate) — ACCEPTED.** `tests/test_compute_contract.py` does not exist. The
  committed-campaign gate is `tests/scripts/hf_jobs/test_campaign_cycle1.py` (pins `cycle1.toml` by path) +
  `tests/scripts/hf_jobs/test_campaign.py`. §5/§8 corrected; `tf46_refit.toml` + the `mode` field get coverage there.
- **A-04 + B CONSIDER — ACCEPTED.** The dataclass is `Campaign` (`campaign.py:51`), not `CampaignSpec`.
- **B CONSIDER (head-to-head) — ACCEPTED.** `headtohead.py:244` loops over committed AND fresh; the crash is on
  the committed load/score (§1/§6.2 B2 wording corrected).
- **Both reviewers: §6.1 mode declaration, §6.3 assemble-isolation, §6.4/§6.5-ii floor fallback — KEEP as-is.**

r2 review: **A APPROVE, B APPROVE** (dual-approved). Non-blocking polish applied: R2-A-01 (the §5 path is
`tests/scripts/test_compute_contract.py`); REFIT-SPEC-06 (§7 names the §6.4/§6.5-ii public-floor fallback as
NEW work); REFIT-SPEC-07 (§5 no longer overstates — no existing gate enforces a committed campaign's `.toml`
surface; `tf46_refit.toml` + `mode` coverage is new builder work).

## 1. Intent and why

The silly-kicks HF-campaign `verify`/`adopt` seam was built to RE-VERIFY a model: re-run the same
trainer on the same corpus and confirm the output reproduces the committed bundle (the O-7 `_OWED`
re-materialize case, where values must be unchanged). The TF-46 a761710 run is a different thing — a
**refit**: a new clean corpus (DFL restored, stoppage domain) and new weights that **supersede** a stale
committed bundle. Every `verify` gate that compares the fresh artifact against the committed model fails,
because a refit is not meant to reproduce the model it replaces.

Three concrete failures, all the same category error (refit judged as re-materialize):

- **B1a — corpus-identity form.** The fit stamps corpus identity with `all_public=is_public.all()`
  (`scripts/train_xshot_occurrence.py:937`; mirror in `scripts/train_xcross_attempt.py`). For an all-public
  arm that writes the exact `corpus_match_ids` list. But the staging record
  (`scripts/hf_jobs/datasets.py:186-187`, `all_public=False`) and `adopt.expectation_violations`
  (`scripts/hf_jobs/adopt.py:242`, public → `"digest"`) compare the `corpus_match_ids_sha256` / `corpus_n_matches`
  digest form. The public artifact therefore shows `{sha256: None, n: None}` against the record's
  `{sha256: 7f81…, n: 17}` and the expectation check fails.
- **B1b — the reproduces gate.** `adopt._integrity` (`adopt.py:289`) shells to
  `scripts/validate_trained_artifact_integrity`; its `reproduces()` (`:43`) requires the fresh model to
  match the COMMITTED model's `n_rows` / `n_positive` exactly and `pr_auc` / `positive_rate` within 5 %.
  A refit changes the corpus and the weights, so it never matches → `go = False` on every arm. This gate
  belongs to re-materialize verification, not to a refit.
- **B2 — the public head-to-head.** The public floor scores BOTH the committed and the fresh bundle
  (`scripts/hf_jobs/headtohead.py:244` loops over committed + fresh, `XShotOccurrenceModel.load(d).predict_proba`).
  This cycle's xShot refactor drifted the committed bundle's feature-contract fingerprint (stored `d86fc4ab`
  vs library `9f17a54c`), so loading/scoring the COMMITTED bundle warns `UnverifiableFeatureContractWarning`
  then throws in `predict_proba`. The floor cannot be "no regression against a committed bundle that no
  longer loads."

The refit models themselves are sound: all 8 arms fit; the owner-tier arms certify green on cv-parity
(no regression); the public xcross arms fail only the legitimate fail-closed acceptance gate
(`brier_lt_base_rate_brier: false`, the confirmed resolution null — out of scope here, ruled accept-gate +
withdraw separately).

A fourth, mechanical item rides along: the xShot public/default arm's final `assemble` reduce never ran
(its `fit` bundle is in the bucket; the controller halted on the xcross-public acceptance failure before
reaching it). The refit flow must leave every passing arm fully assembled and retrievable.

## 2. Goals

- A campaign declares its **mode**: `refit` (new corpus/weights supersede the committed bundle) or
  `rematerialize` (same model/corpus, values expected unchanged — the current default behaviour). The mode
  is a first-class, committed declaration.
- `verify` routes the corpus-identity check, the integrity/reproduces gate, and the head-to-head/floor by
  the mode, so a refit is judged as a supersede, not a reproduction.
- The TF-46 refit-subset campaign becomes a committed `scripts/hf_jobs/campaigns/tf46_refit.toml`
  carrying `mode = "refit"`, closing the ADR-128 "committed campaign" gap (it has been running from an
  uncommitted DGX-local `~/xprobe_build/cycle1_refit.toml`).
- A refit arm whose acceptance fails (xcross-public) still lets the other arms assemble, retrieve and verify
  — one arm's legitimate gate failure must not strand a sibling arm's assemble.
- The six passing a761710 arms verify green under the refit path, ready for the (separately owner-gated)
  adopt + the xcross-public withdrawal.

## 3. Non-goals

- Not touching the acceptance gates themselves (`brier_lt_base_rate_brier` etc.). The xcross-public null is
  real and its disposition (accept-gate + withdraw the stale `_xcross_weights` with a provenance note, the
  Task-8 seam flip to NaN+warn) is already decided and lives in the TF-46 Commit-2 work, not here.
- Not re-opening the re-materialize path. `mode = "rematerialize"` must keep today's exact behaviour,
  including the `reproduces` gate for the O-7 `_OWED` regen (values unchanged).
- Not changing what a refit trains or how it fits. Only how the campaign DECLARES itself and how
  `verify`/`adopt` JUDGE the result.

## 4. Anti-goals (a failure even if it technically works)

- A refit silently adopting weights that regressed. The refit floor must still be a real floor
  (certification cv-parity / a valid public floor), never "skip all checks because it is a refit."
- A `rematerialize` run being judged by the looser refit path and hiding a real reproduction drift.
- The mode being inferrable/implicit such that a run's judgement depends on heuristics rather than an
  explicit committed declaration.

## 5. Constraints

- ADR-128: campaigns run from a committed `scripts/hf_jobs/campaigns/*.toml`. Today the only committed-campaign
  gate is `tests/scripts/hf_jobs/test_campaign_cycle1.py`, pinned to `cycle1.toml` BY PATH; `test_campaign.py`
  is the planner suite (it builds `Campaign` inline, not a `.toml`-surface gate), and NO test iterates
  `campaigns/*.toml`. So a committed `tf46_refit.toml` + the `mode` field are NEW coverage work (§8), not
  something an existing gate already enforces. (`tests/scripts/test_compute_contract.py` is the PIPELINE/job
  contract, not a campaign gate.)
- The controller is stateless/resumable; run keys are `{campaign}-{commit7}-{job}-{tier}`; a mode change
  must not silently collide with or reinterpret an existing store.
- Trained-artifact contract: parameters-only, `load()` runs `verify_chirality` + `_feature_contract`
  (ADR-011/016/040/044/050). A drifted committed fingerprint is a real signal, not something to suppress.
- id-dtype and corpus-redistribution rules unchanged (ADR-019; corpus ids stay, ADR-116).
- No new runtime deps.

## 6. The decided design

### 6.1 Mode declaration (owner decision 2026-11-03)

- A new committed `scripts/hf_jobs/campaigns/tf46_refit.toml` holds exactly the four xShot/xCross instances
  (public + sc_extended, both variants) and a campaign-level `mode = "refit"`. The other `cycle1.toml`
  jobs stay out (owner-ruled).
- **Campaign name `tf46_refit`** (owner decision 2026-11-03): a UNIQUE name, distinct from `cycle1`.
  Run keys become `tf46_refit-{commit7}-{job}-{tier}`, so the 4-job refit RE-RUNS on the implemented +
  verified seam commit under fresh stores (the existing `cycle1-a761710-*` stores are abandoned). The
  owner accepted the re-fit cost (four xgboost fits on cpu-upgrade, pennies) in exchange for a clean,
  collision-free committed campaign name. Rejected: reusing `name = "cycle1"` to read the existing
  a761710 fits — it would put two committed campaign files under one name.
- `Campaign` (`campaign.py:51`) gains a `mode` field, default `"rematerialize"` (today's behaviour),
  validated against `{"refit", "rematerialize"}`. The committed-campaign gate (§5/§8) learns the new
  campaign file + the field.

### 6.2 verify routing by mode

- **Corpus-identity (B1a).** The in-job fit is visibility-blind (ADR-123 Amendment B / the I1 import-closure
  rule) and must ALWAYS stamp the DIGEST form (`corpus_match_ids_sha256` / `corpus_n_matches`); `adopt`
  rehydrates the exact public ids off-HF at step 5 via `rehydrate_public_corpus_ids` (`scripts/_corpus.py:136`),
  which reconstructs them from the canonical registry and guards on `corpus_n_matches`. The current bug is the
  `all_public=bool(inp.is_public.all())` branch at `train_xshot_occurrence.py:937` (mirror in train_xcross):
  for an all-public arm it stamps the EXACT `corpus_match_ids` form, but `expectation_violations` forces the
  DIGEST comparison for a public artifact (`adopt.py:242`, `public_ids`→`"digest"`), so the sha256/n it reads
  are null. **Builder call (6.5-i):** remove that branch; the fit stamps digest unconditionally (`all_public=False`
  at the stamp site). `a.corpus` is the unchanged manifest tag `"public_ids"`, so rehydrate still fires. This is
  the digest-always design `rehydrate_public_corpus_ids` already documents; B verified it.
- **Integrity / reproduces (B1b).** `verify` routes by mode:
  - `rematerialize` → keep `_integrity` / `reproduces` exactly as today (same model/corpus, values unchanged —
    the O-7 `_OWED` regen).
  - `refit` → SKIP `_integrity` / `reproduces` (a refit's new corpus + new weights never reproduce the committed
    model's `n_rows`/`n_positive`/`pr_auc`). Gate on certification ALONE via the existing `_certify`
    (`adopt.py:379`): owner → cv-parity on `cv_metric` (no regression within `delta`); public → the §6.4 floor.
    Do NOT route to `_rebaseline` — it is `DriverSpec`-only (`adopt.py:464`), its `REBASELINE_FROM` has no
    trainer entry (`:49`→KeyError `:341`), trainers expose no `input_contract()` (`:337`), and it STOPS on the
    code delta every refit carries. The floor stays real: the trainer's own fail-closed acceptance gates
    (`brier`/`pr_auc`/`log_loss`) ran at fit, and certification checks no-regression at verify.
- **Head-to-head / public floor (B2).** In `refit` mode, before the head-to-head scores the committed bundle
  it checks that bundle's own feature contract. **Builder call (6.5-ii):** when the committed bundle fails its
  own `load()`/contract (a drifted fingerprint — exactly the state a refit exists to replace), there is no
  valid committed baseline, so the public floor is the fresh model's fail-closed acceptance gate
  (`acceptance.*` in the fresh `metrics.json`), not a comparison against an unloadable model. A committed
  bundle that DOES load is compared as today. Never auto-pass.

### 6.3 Assemble must not be stranded

A refit campaign where one arm's acceptance fails (xcross-public) must still assemble, retrieve and verify
every OTHER arm. **This property already holds in the committed controller** (verified at plan review,
2026-11-03): `step()` continues past a halted instance (`scripts/hf_jobs/campaign.py:542` `continue`) and the
`--watch` loop exits only when every instance is settled (`settled = all(done or failed)`, `:829`), so one
arm's `halt` never strands a sibling with a launchable `assemble`. No controller code change is needed. The
a761710 xShot public/default assemble that was observed stranded was an operational artifact of an aborted
run sequence, not a controller defect; it is moot once the campaign is re-run under the renamed `tf46_refit`
keys (all four jobs re-run fresh). (r1 wrongly asserted a controller bug here — corrected after verification.)

### 6.4 The public floor, stated

The public floor in `refit` mode, in order: (1) if a loadable committed bundle exists, the head-to-head
no-regression comparison as today; (2) else, the trainer's fail-closed acceptance gate result carried in
the fresh `metrics.json` (`acceptance.*`). Never auto-pass.

### 6.5 Builder calls (settled here, flagged for A/B)

- **6.5-i** Fit stamps the digest corpus-identity form unconditionally (visibility-blind); adopt rehydrates
  exact public ids at step 5. (Alternative rejected: teach record + verify to accept the exact form — more
  sites, and it fights the existing rehydrate design. A's "stamp both forms, keep public_ids tag" is
  unnecessary: `a.corpus` is the manifest tag, not the stamp, so rehydrate fires regardless — A-01 premise
  false, resolved in B's favor.)
- **6.5-ii** A committed bundle that fails its own contract load ⇒ "no valid baseline" ⇒ the public floor
  falls back to the fresh acceptance gate. (Alternative rejected: refresh/re-fingerprint the committed
  bundle to compare — that would validate the refit against a patched stale model, which is backwards.)

## 7. What is left to the builder

- The exact shape of the `mode` field on `Campaign` (`campaign.py:51`) and its plumbing into `verify`
  (one routing switch in the `_integrity`/`_certify` dispatch, not scattered conditionals).
- The refit integrity routing: `verify` skips `_integrity` for `mode = "refit"` and gates on `_certify`
  alone. No `_rebaseline` (driver-only, no trainer baseline — B-1).
- NEW logic in the public floor (§6.4 / §6.5-ii): the committed-contract pre-check and the fresh-acceptance
  fallback in the head-to-head / `_certify` public path are new code, not "existing `_certify`" (REFIT-SPEC-06).
  (Owner cv-parity needs no change: `_committed_metrics` reads the committed `metrics.json` as a file
  (`adopt.py:278-282`), so the owner arms never load the drifted bundle and never hit the B2 crash.)
- Red-green tests per blocker: a refit arm passes verify where today it fails on B1a/B1b/B2; a
  `rematerialize` run still fails when a real reproduction drift is injected (the re-materialize path keeps
  its teeth); a sibling arm assembles when another arm's acceptance fails; the committed `tf46_refit.toml`
  + the `mode` field are enforced by the committed-campaign gate (`test_campaign_cycle1.py` / `test_campaign.py`).
- Finishing the stranded a761710 xShot public/default assemble.
- Following the authoring/howto runbooks (`docs/howto/compute-pipelines.md`, `docs/howto/hf-jobs-training.md`)
  and the ADR for any contract change (a new ADR entry for the refit mode).

## 8. Test / verification bar

- Every blocker tested from both sides (ADR convention): refit passes, re-materialize still catches drift.
- The committed-campaign gate (`tests/scripts/hf_jobs/test_campaign_cycle1.py` + `test_campaign.py`) covers
  the committed `tf46_refit.toml` and the `mode` field.
- The CI-faithful trio (ruff + pyright + `pytest -m "not e2e"`) green before any commit.
- The six a761710 arms verify green under the refit path (the acceptance proof), re-run on the DGX.

## 9. Out of scope / downstream

- Commit-2 artifact adoption, the xcross-public withdrawal, Task-8 seam flip, ADR-130/CHANGELOG/NOTICE/version,
  O-7 `_OWED` regen — all separate, owner-gated, and unblocked once this seam lands.
- The concurrent `~/sk-hfu` full-`cycle1.toml` controller on the DGX is a different effort; untouched.
