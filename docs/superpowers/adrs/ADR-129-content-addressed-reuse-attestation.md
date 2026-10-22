# ADR-129: Content-addressed reuse and determinism attestation

| Field | Value |
|---|---|
| **Date** | 2026-10-22 |
| **Status** | Accepted |
| **Deciders** | Karsten |

## Context

ADR-128 Cycle 1 keyed a campaign store by `{campaign}-{commit7}-{job}-{tier}`, so two campaigns, or two commits
that changed nothing a stage reads, never shared a generation and re-materialized identical bytes. Cycle 1 also
proved a stage's output deterministic by re-running the whole stage in a same-commit replica and comparing byte
for byte — a 2x compute cost on every stage, every campaign. The forcing function for Cycle 1b is throughput:
re-fit campaigns that touch only a few modules must reuse the untouched generations, and the determinism proof
must not re-pay the full replica once a generation is known to be reproducible.

The store is the only shared state (the controller is stateless, ADR-128 S5.5). Reuse therefore has to be
decided from the store alone, at a clean HEAD, without trusting that the adopting checkout equals the campaign
commit. A reused generation that was NOT actually reproducible would silently ship wrong weights, so reuse is
admitted only behind a determinism attestation and a per-adoption canary re-run.

## Decision

Name the MAIN store CONTENT-ADDRESSED (`{job}-{tier}`): a generation is named by its token, so equal tokens
share one generation across commits and campaigns and unequal tokens never meet. Prove determinism ONCE per
token with a replica byte-compare and record an ATTESTATION in the main store; reuse an attested generation at
any later clean HEAD whose stage token still equals it, gated by a one-unit canary that re-runs over the reused
upstream and must reproduce main byte for byte. A replica (`-r<N>`) and a canary (`-canary`) stay FRESH per
campaign and commit.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Keep the per-commit store + full same-commit replica (Cycle 1) | simple; no reuse logic | 2x compute every campaign; no cross-commit reuse | fails the throughput forcing function |
| B. Content-address + trust the token alone (no replica, no canary) | cheapest | a non-reproducible generation ships silently; no determinism evidence | unacceptable integrity risk |
| C. Content-address + replica attestation + per-adoption canary (chosen) | reuse across commits/campaigns; determinism proven once; a mismatch quarantines | attestation + canary machinery; a canary unit per attested stage per adoption | — |

## Decision detail

### Content-addressed main store

`run_key(c, commit, job, tier)` returns `{job}-{tier}` for the main run (no campaign or commit in the key);
`-r<N>` for a replica and `-canary` for a canary stay fresh, so neither ever reads a main generation. A
generation inside the store is named by its token (ADR-128 P9 closure identity), so the store is the single
source of what already exists. The token hashes each closure file's bytes with text files' line endings
normalized to LF (`_stages._closure_file_bytes`; binary files, detected by a NUL byte, left raw), so the token
is identical on a CRLF Windows controller checkout and the LF tree a Linux job builds from — the same
cross-platform hazard the image tag avoids (`image.inputs_sha256`).

### Determinism attestation (replica byte-compare)

`verify(..., with_replica=N)` compares, per stage, the main generation to the replica generation file by file
(`adopt.generation_diff`). Run history is excluded from the compare — `*.ledger.json`, `*.failed.json`,
`_profile.json`, the `.corrupt/` tree and `_ledger/` — while every result, sidecar, FileSet member,
`*.excluded.json`, `*.counters.json`, `_summary.json`, `_inputs.json` and the output IS compared. A
byte-identical generation gets an attestation written to the MAIN store at `_attest/{stage}/{token}/{commit}.json`
carrying both jobs' recorded environment (`phases.publish_first_unit` now records `env = env_provenance()`).

An attestation is REFUSED when either job's recorded environment is empty (`phases._env` swallowed
`env_provenance`, so no `tree_state`): unknown provenance is untrustworthy and is never recorded as clean
(`scripts/_provenance.py`). The determinism compare still counts for the S7.2 bar (the bytes were identical);
only the reuse token is withheld, so a later HEAD cannot reuse a generation on undescribable code. A recorded
`dirty` tree state is kept, because its token never equals a clean HEAD's and so is never reused cross-commit.

Each `verify` call with a replica or canary also records its byte-compares, scoped to this campaign and commit, in
`_campaign/{campaign}/{commit7}/comparisons.json` (`adopt.comparisons_rel`). The keys are
`replica|{N}|{instance}|{stage}` and `canary|{instance}|{stage}`; the values are booleans. Successive calls MERGE
into the file, so a later match never overwrites an earlier mismatch. A stage counts only when it materialized
on both sides. The S7.2 determinism bar reads this record (`adopt.read_comparisons`), not the attestation files:
the attestation files span every campaign in the shared store and omit a mismatch.

### The canary and quarantine

For a stage whose token is attested on HEAD or an ancestor and not quarantined, the campaign runs ONE canary
unit (`campaign.canary_key`, deterministic per `(token, HEAD)`) in a fresh `-canary` store, reading the reused
upstream generations through `CanaryReadThroughStore` (gated by an allow file, `_campaign/{campaign}/{commit7}/
canary/{action_id}.allow.json`). `verify(..., canary=True)` compares the canary unit's bytes to main; a mismatch
calls `BucketStore.quarantine_generation` (moves the generation and its attestations under `_quarantine/`) and
records an `unsound_reuse` finding for the owner. The replica of an attested stage is SKIPPED
(`plan_for_role(..., "replica")` marks it reduced): the attestation already proves it.

### Adoption at a clean HEAD

`adopt(..., api=...)` reuses an attested generation at a later clean HEAD. `_assert_adoptable` recomputes each
stage's token at HEAD (`read_plan`) and runs `adoption_refusals`: every adopted generation's token equals HEAD's
token, some attesting commit is an ancestor of HEAD, the token is not quarantined, and its canary passed (or HEAD
attested it this campaign). A commit-free output (ADR-128 Task 16, S5.10) is stamped from the provenance of its
OLDEST ancestor attestation (`oldest_ancestor_attester`); its stamped commit becomes the artifact's
`training_commit`. With no `api` (the pure-diff unit path) `adopt` keeps the ADR-037 rule that HEAD IS the
campaign commit.

### INV-10 / INV-11

- **INV-10 (reuse is attested):** a generation is reused only when an attestation records a byte-identical
  replica compare at an ancestor commit, and the token has not been quarantined.
- **INV-11 (reuse is canary-checked):** every adoption that reuses a generation runs a canary unit over the
  reused upstream and refuses on a byte mismatch, which also quarantines the generation.

### Importance-split outcome (S5.6, Task 19)

The ghost_gk `importance` stage is NOT split per permutation group. `train_ghost_gk._importance_work`
(`scripts/train_ghost_gk.py:352`) calls scikit-learn `permutation_importance(n_repeats=5, random_state=...)` over
all feature columns in ONE call: the RNG stream is SHARED across feature groups (a per-group seed is not
derivable from the group key, failing split condition 1), and the evaluation subsample
(`np.random.default_rng(42).choice`, `:393`) and the baseline score are whole-set quantities (failing split
condition 3). The stage stays one `inner_parallel` unit.

### The prebuilt job image (S5.12, E-1/E-2)

Cycle 1 built the environment in every job (`pip install` + `uv sync`). Cycle 1b builds the locked third-party
environment ONCE into a GHCR image and the job only CHECKS it (`image.CHECK_COMMAND` = `uv sync --frozen --check
… --group hfjobs`, never an install). The image is derived purely from `uv.lock`, the Dockerfile, the base
digest, the extras and the `uv` pin (`image.derived_tag`, line endings normalized so the tag is the same on a
Windows checkout and the LF CI/GHCR build); `image.lock.json` pins the tag, the GHCR `registry_ref`, the built
`digest` and the measured `startup_s`. `statsbombpy` is supplied by the `hfjobs` dependency GROUP, not a
`--extra` (uv rejects an undefined extra), recorded in `image.GROUP_PROVIDED`.

- **E-1 (owner-gated build).** `.github/workflows/job-image.yml` builds, pushes and attests the image on a
  `job-image/env-<tag>` tag, in the owner-approved `job-image` environment, every action pinned by full commit
  SHA — the same gate pattern as `publish.yml`. `preflight.check_image` refuses a lock whose tag does not match
  its inputs, whose `startup_s` is missing, whose ref does not resolve on GHCR, or whose build provenance does
  not verify.
- **E-2 (reuse delivers INV-11).** The canary re-runs over the reused upstream through
  `CanaryReadThroughStore`, so a reuse that is not byte-reproducible is caught and quarantined before adoption.

## Consequences

### Positive

- A re-fit campaign reuses every untouched generation; only the modules that changed re-materialize.
- Determinism is proven once per token and reused across commits and campaigns.
- A non-reproducible reuse cannot ship: the canary refuses and quarantines it.

### Negative

- Attestation, canary and quarantine machinery in `adopt`/`campaign`/`store`.
- One canary unit per attested stage per adoption (far below a full replica, but not free).
- `adopt` now needs the HF API to read attestations when reusing (the pure-diff path keeps the old rule).

### Neutral

- `adopted_at_commit` and `attested_by` are recorded in `observed.json` (not the human manifest, which INV-9
  keeps adopt-written only for `training_commit`); the two fields also exist on `manifest.Artifact`.
- `phases.publish_first_unit` records `env`, computed once per attempt by the single writer.

## Related

- **Specs:** `docs/superpowers/specs/2026-10-21-campaign-throughput-design.md` (S5.9, S5.10, S5.12, S5.13, S7.2)
- **Plans:** `docs/superpowers/plans/2026-10-21-campaign-throughput-plan.md`
- **ADRs:** amends ADR-128 (P9 closure identity replaces the commit and `cross_commit_version`; P18 attestation
  replaces the same-commit replica); relates to ADR-037 (adoption by token equality at a clean HEAD satisfies the
  clean-tree split).
