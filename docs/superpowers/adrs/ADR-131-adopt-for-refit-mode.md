# ADR-131: Adopt-for-refit campaign mode

**Status:** Accepted (2026-11-03)
**Spec:** `docs/superpowers/specs/2026-11-03-adopt-for-refit-seam-design.md` (dual-approved)
**Plan:** `docs/superpowers/plans/2026-11-03-adopt-for-refit-seam.md` (dual-approved)

## Context

The ADR-128 HF-campaign `verify`/`adopt` seam was built to RE-VERIFY a model: re-run the same trainer on
the same corpus and confirm the output reproduces the committed bundle (the O-7 `_OWED` re-materialize
case, values unchanged). The TF-46 a761710 run is a REFIT: a new clean corpus + new weights that SUPERSEDE
a stale committed bundle. Every `verify` gate that compares the fresh artifact against the committed model
failed, because a refit is not meant to reproduce the model it replaces (corpus-identity form mismatch; the
`reproduces` integrity gate always false; the public head-to-head crashing on the committed bundle's drifted
feature-contract fingerprint).

## Decision

A campaign declares a first-class `mode`, and `verify` judges by it.

- **`Campaign.mode`** (`scripts/hf_jobs/campaign.py`): `"rematerialize"` (default — today's behaviour) or
  `"refit"`. `load_campaign` reads `campaign.mode`, validates `{"refit","rematerialize"}`, refuses otherwise.
  A refit campaign is a committed `scripts/hf_jobs/campaigns/*.toml` (e.g. `tf46_refit.toml`).
- **Corpus identity — stamp digest always.** The visibility-blind in-job trainer stamps the DIGEST form
  (`corpus_match_ids_sha256` / `corpus_n_matches`) unconditionally (ADR-123 Amendment B); `adopt` rehydrates
  the exact PUBLIC ids off-HF via `rehydrate_public_corpus_ids`. (Removes the `all_public=is_public.all()`
  branch at the trainer stamp sites.)
- **Integrity routing.** `verify` runs the `_integrity`/`reproduces` gate iff
  `inst.job in INTEGRITY_FAMILIES and mode != "refit"`. A refit SKIPS it (a new corpus/weights never
  reproduce the committed model) and is judged by certification alone (`_certify`). `rematerialize` keeps
  the gate — the O-7 regen must still catch a reproduction drift.
- **Public floor fallback.** In `refit` mode, `_certify`'s public path first checks the committed bundle via
  the SUBPROCESS `load_check` (the same `.venv-hfjobs` env the head-to-head loads in). When it fails its own
  contract (a drifted fingerprint), there is no valid baseline, so the floor is the FRESH model's
  fail-closed `acceptance` gates — never auto-pass. A loadable committed bundle, and every `rematerialize`
  arm, take the existing head-to-head path unchanged.

## Consequences

- A genuine refit verifies on its own terms; the re-materialize path is unchanged and keeps its teeth.
- The public corpus ids are still redistributed (ADR-116) — rehydrated at adopt, not stamped in-job.
- The committed controller already isolates a halted instance (`step` continues past a halt; the `--watch`
  loop exits only when every instance is settled), so no halt-isolation change was needed — a sibling arm's
  `assemble` is never stranded by another arm's acceptance failure.
- Not changed here: the acceptance gates themselves (the xcross-public Brier null is real; its disposition —
  accept-gate + withdraw the stale weights — is separate TF-46 Commit-2 work under ADR-130).
