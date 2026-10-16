# ADR-123: HF Jobs parallel, dataset-agnostic training harness

| Field | Value |
|---|---|
| **Date** | 2026-10-14 |
| **Status** | Accepted |
| **Deciders** | Karsten S. Nielsen (owner) |
| **Extends** | ADR-122 (HF Jobs training harness) |

> **ADR number — CONTENDED.** `tf-79-ball-contact-detection` also provisionally claims ADR-122/123 (its
> header). Whichever branch merges second renumbers (ADR file + `AGENTS.md` + C4 pin + memo). Owner owns
> the cross-branch ADR order; re-confirm the next free number before merge.

## Context

The ADR-122 harness re-fit the four **public** trainers cleanly but its first owner-tier campaign
(2026-10-14) was halted — four defects: (1) ghost extraction was a ~4 h serial wall that never reached
the fit inside the timeout (~$54 sunk, zero weights); (2) xcross's `assemble_studies` refused (the TF-19
`_probe_sample/` was absent — the probe needed a separate held-out pass that the staged corpus dropped);
(3) the per-variant head-to-head crashed `position_only` (`feature_names mismatch` — the eval extracted
the faithful vintage); (4) `repackage` did not emit `actions.parquet`. The owner then set the direction:
**cpu-upgrade only**, **break the big jobs into parallelizable parts**, and — decisively — the code that
runs on HF must be **dataset-agnostic** and must **never** receive pining information.

## Decision

### 1. Uniform map/reduce per heavy trainer
Each heavy trainer exposes phases that fan out across small `cpu-upgrade` jobs with a serial, **byte-identical**
reduce:
- ghost_gk: `extract (--shard-index I/N)` → `build-cache` → `fold (--fold K)` → `assemble`.
- ghost_outfield: `extract` → `fold` → `assemble` (no whole-corpus cache, so no build-cache phase).
- xshot/xcross: `extract` → `prep (--prep-only)` → `study (--study TAG, existing seam)` → `assemble`.
- receiver: `extract` → reduce (plain no-`--shard-index` run: reconcile all shards + fit).
Extraction partitions by `scripts._driver.shard_select` (disjoint cover). The **single final model fit
stays serial** (splitting it would change the model). Determinism is CI-gated: fanned == serial shipped
weights + metrics, on a toy corpus spanning ≥ 2 shards + ≥ 2 folds. **The fanned fold count is coupled to
the trainer's CV split**: the orchestrator threads `--cv-folds = n_folds` for ghost, so `assemble` never
expects more fold sidecars than were fanned (a DGX-e2e-caught `FileNotFoundError` otherwise). `ghost_gk`'s `--build-cache` is a
race-free addition over the design's three phases (so fold workers never concurrently build the cache).

### 2. The in-job code is DATASET-AGNOSTIC (the headline)
No pining information ever reaches HF — no token, no manifest, no `SILLY_KICKS_PINING_MANIFEST_DIR`, no
pining-driving match-id list — and the in-job trainers do not import `_loader_pining` or use `--providers`
pining mode. Every owner-tier trainer consumes a **staged neutral parquet dataset** through a data path
(`--data-dir`, `--probe-data-dir`). How that dataset was built — pining on the owner's DGX, statsbombpy,
kloppy, or any other user's own loader — is **off-HF and not the repo's concern**; anyone can stage their
own dataset in the documented shape and run the harness. Consequences:
- **xcross** reproduces the held-out TF-19 probe from a staged per-game probe dataset via `--probe-data-dir`
  (`_iter_matches_from_dir` + `_extract(probe_providers=…)`), not the pining `--probe-match-ids-json`.
- **receiver** gains a `_data_dir_source` (`--data-dir`, flat shards + `_actions` — the eval shape); its
  `owner_sb360` reads the staged flat corpus, no manifest/`--cache-dir`/`--match-ids-json`. The `sb360_repo`
  launch param is removed (this reworks the merged-ready Commit-1 receiver, which leaked pining info to HF).
- ghost (`owner_flat`) + xshot (`owner_pergame`) were already agnostic. public = statsbombpy **open** data
  (universal), an in-job fetch.

### 3. cpu-upgrade only; reduce-RAM is an owner-gated measurement
Every `REGISTRY` flavor is `cpu-upgrade` (8 vCPU / 32 GB). The serial reduce is the one phase fan-out does
not shrink, and 32 GB is an 8× RAM cut from cpu-performance; a 32 GB OOM is **not** `--attempts`-fixable.
The **DGX e2e gate measures peak RSS per phase vs 32 GB** before any HF spend (the DGX's ample RAM would
otherwise hide the OOM); a phase over the ceiling **blocks** the wave until the owner rules on the reduce
executor/flavor. The owner declined to pre-authorize a fallback — it is decided on the measured number.

### 4. Feature-vintage-correct floor
The head-to-head eval extracts features at the variant's `feature_set` (`VariantBuild.feature_set`;
`build_eval(…, feature_set=)`), so a 26-feature `position_only` booster is scored on a matching frame.

### 5. Orchestration + DGX gate
`scripts/hf_jobs/launch.py` grows a staged DAG (`run_stage`/`wait_all`/`run_pipeline`/`StagePhaseError`)
over an injected runner — the HF CLI or `make_local_runner` (local subprocess) — so the identical pipeline
drives the DGX e2e + fallback. `scripts/hf_jobs/dgx_e2e.py` runs the pipeline locally with per-phase
`measure_rss` (GNU `time -v`). Staging produces **only neutral parquet** (the GS `10502/10503` probe slice
for xcross; the flat SB360 corpus for receiver); pining is used on the DGX to materialize, off-HF.

## Consequences
- Published results are reproducible by anyone who stages a dataset in the neutral shape — the pining layer
  is purely the owner's off-HF staging tool, never shipped to or required on HF.
- One owner-gated harness-tune commit (not amending `6f6508a`); weights land in a subsequent commit; PR #284
  merges non-squash. No commit / HF spend without explicit owner approval; the DGX e2e (peak RSS < 32 GB or
  an owner ruling) gates the wave.
- The staged dataset contract is documented in `docs/howto/hf-jobs-training.md` so others can stage their own.

## Validation — DGX e2e (2026-10-15): PASS, not blocked
All five trainers ran the full phase DAG locally on the DGX against the real owner corpora (zero HF spend).
Peak RSS per reduce vs the 32 GB cpu-upgrade ceiling:

| trainer | reduce peak RSS | blocked |
|---|---|---|
| ghost_gk | 6.07 GB | no |
| ghost_outfield | 4.96 GB | no |
| xshot_occurrence | 6.13 GB (extract 8.07) | no |
| xcross_attempt | 7.28 GB | no |
| receiver_model | 0.59 GB | no |

Max 8.07 GB — ≥ 4× under 32 GB, so **cpu-upgrade is sufficient everywhere and the reduce-RAM hold needs no
flavor exception**. Determinism (fanned == serial, byte-identical) verified under sklearn 1.9.1; the agnostic
`--probe-data-dir` (xcross held-out GS probe; 91 999 rows = the committed corpus) and receiver `--data-dir`
(SB360 30, pining-materialized off-HF to neutral parquet) paths ran end-to-end, floor included. The gate
surfaced + fixed three harness defects (cv_folds↔n_folds coupling, `measure_rss` shell, e2e `allow_dirty`).
Named datasets for the HF waves: `silly-kicks/hf-refit-{tc3, xshot17, xcross-probe, sb360}`.

## Amendment B (2026-10-15): the in-job code is visibility-BLIND; policy enters as a job parameter

A re-fit surfaced that two trainers — `xshot_occurrence` and `xcross_attempt` — ship a *visibility-gated*
candidate (`public` / `sc_extended` / `full`). That selection depends on per-match public-vs-owner
**visibility**, which the in-job code derived from the pining manifest (`_loader_pining.match_visibility`)
and the in-repo registry (`_corpus.PUBLIC_CORPUS`). On the agnostic `--data-dir` path there is no manifest,
so it fail-closed to *all-private*, the `public` candidate never formed, and the trainer shipped
`sc_extended` — a 4× worse, non-faithful model (committed `public` PR-AUC 0.3418/0.3036 vs fresh
`sc_extended` 0.0825/0.0770). Amendment A made the in-job code **data**-agnostic; this amendment makes it
**policy**-agnostic too.

**Invariants.**
- **I1 — in-job code is visibility-blind.** No module on a trainer's in-job execution path references the
  visibility taxonomy (`PUBLIC_CORPUS` / `is_public_row` / `assert_public_corpus` / `artifact_label` /
  `match_visibility`). The neutral provenance helpers (`corpus_identity` / `reproducibility` /
  `check_shipped_variant`) were split out of `_corpus` into `_corpus_identity` so the trainers import none
  of `_corpus`. Enforced by `tests/scripts/hf_jobs/test_visibility_blind_gate.py` as a **pure static-closure**
  walk (subpackage-aware + a planted-leak traversal meta-assertion). The generic ADR-052 `_driver`/`_partition`
  corpus-driver seam is **fully traversed, not trusted as a leaf**: the one pining-reaching helper
  (`list_match_ids`) was relocated to `scripts/_partition_pining.py` (maintainer-only), so `_partition` never
  reaches `_loader_pining` and the gate needs no exemption or runtime-reachability reasoning (an earlier draft
  trusted the seam as a leaf on the grounds its lazy pining branch is unexecuted on `--data-dir` — the exact
  reachability reasoning I1 exists to remove). An anti-rot test asserts the seam IS traversed, is banned-clean,
  and that `_partition_pining` is NOT in the in-job closure.
- **I2 — policy enters as a job parameter.** The launcher computes the arm membership OFF-HF
  (`arm_policy.compute_arms` over `PUBLIC_CORPUS`, converter→identity reconciled internally, emitting the
  dataset's own keys) and passes it as `--arms` (a base64-embedded JSON the job decodes to `/tmp/arms.json`).
  The in-job trainer applies the memberships by `np.isin` (`_build_candidate_masks`) and labels the shipped
  artifact from the arm name. It never computes the policy.
- **I3 — datasets stay neutral.** Visibility is a per-run parameter, never a dataset column.
- **I4 — fail-loud.** A visibility-gated trainer invoked with no `--arms` exits with an error; it never
  falls back to all-private (the fail-closed-silent behaviour that shipped the wrong candidate).

The `--providers` pining path was removed from both trainers (its only remaining `_loader_pining` reach);
maintainer DGX re-fits adopt the launcher + `--arms`. The DGX e2e now asserts the fresh shipped candidate
equals the committed one (`dgx_e2e.check_shipped_variant`) — closing the GATE-1 exit-0-only gap — and
faithfulness is confirmed by CV-parity (fresh `public` PR-AUC ≈ committed). `ghost_gk` / `ghost_outfield`
(provider-agnostic `cv_mae`), `receiver_model` (`top1_cv`) and the four public trainers ship no
visibility-gated candidate and are unchanged.
