# ADR-044: Ghost-GK parameters-only artifacts (+ `from_variant("public")` alias fix)

**Status:** Accepted
**Date:** 2026-07-20
**Release:** silly-kicks 4.54.0 (PR-S121)
**Amends:** ADR-016 (served estimator), ADR-038 (corpus-taxonomy reach)
**Spec:** `docs/superpowers/specs/2026-07-20-ghost-gk-parameters-only-artifacts-design.md`
**Plan:** `docs/superpowers/plans/2026-07-20-ghost-gk-parameters-only-artifacts.md`

## Context

`GhostGkModel.save()` persisted three per-sample arrays — `training_gk_x`, `training_gk_y`,
`training_leaves` — the raw per-frame goalkeeper positions of every training sample, verbatim.
RFCDE evaluates its conditional density by running a weighted KDE over the actual responses of
training samples sharing leaves, so retaining training targets is inherent to that method. The
arrays were correct when the density was the served read-out.

They have not been the served read-out since 4.14.0 (ADR-016): the served position is the exact
boosted HGBR mean (`predict_mean`), reconstructed from tree nodes + baselines alone. The arrays
back exactly one emitted column, `ghost_gk_density_spread`, which has **no numeric consumer** in
the library or the downstream marts, and are ~90% of the artifact by bytes.

## Decision

Distributed model artifacts carry **learned parameters only, not per-sample training data.**
`GhostGkModel.save()` structurally never persists the three arrays; `load()` tolerates their
absence; `predict_density` (and the whole KDE capability) survives only on a locally `fit()` model,
never a loaded one. The emitted `ghost_gk_density_spread` column and the `kde_backend` kwarg are
retired from `compute_ghost_gk` / `add_ghost_gk` / `ghost_gk_xfns`. Artifact format 1.2.0 → 1.3.0
(`stores_training_data: false`); the bundled `default` is migrated by a pure `load(old).save(new)`
re-save (7,376,181 → ~764,418 bytes, **no retrain**; `predict_mean` byte-identical, chirality
fingerprint unchanged). A `metadata.json` corpus-provenance block (providers + counts, **never**
match ids, **never** a public/restricted split) is added, and a CI **name allowlist** over every
bundled weights directory fails on any array name that is not a recognized parameter.

Retiring a provably-dead column requires consumer sign-off + proof (repo rule): proof is the
zero-consumer measurement; **owner sign-off was given in the 2026-07-20 design session.**

### The name allowlist is the generalizable control

The durable win here is not this one strip but the anti-rot property: a new array name in **any**
bundled weights directory fails CI until a human classifies it as parameter-or-per-sample. That is
what prevents the *next* inadvertent per-sample array in any future bundled model. A bare
`max(shape) <= N` size cap was considered and rejected — it is fail-open for a small-subsample
artifact (per-sample arrays under N pass) and false-positive for a legitimately larger tree (a
`max_leaf_nodes` bump pushes `tree_nodes_*` above N). The name allowlist has neither failure mode.
It is name-scoped by necessity (the artifact records no corpus-size hyperparameter to bound
against), so it guards against inadvertence, not a determined author.

### Scope decisions (recorded, not deferred silently)

1. **Forward-only.** This change prevents recurrence; it removes nothing already distributed. The
   bundled `default` shipped in every wheel **and sdist** since 3.24.0 (`pyproject.toml` excludes
   only `full/`); PyPI files are immutable and a yanked `==` still resolves. A disposition for
   already-published wheels is **not** made here — it belongs to the ghost-GK disclosure
   remediation the owner holds.

2. **No retraining.** The strip is a mechanical transformation of the existing artifact; corpus
   composition is unchanged. Consequence for provenance: the *migrated bundled `default`* records
   the corpus provenance that is honestly available at migration — `n_rows` read from the loaded
   arrays before the strip (36,000) and `providers` / `n_games` (179) from the authoritative
   training record (the Stage-B `default` `metrics.json`). `n_games` is the training-corpus game
   count; the artifact is a row-subsample and its per-game coverage is not verifiable post-strip.
   Every *future trained* artifact records all three from live data via the trainer plumbing.

3. **The §6 join-liveness gate is intentionally absent.** The owner's providers+counts-only (no
   split) provenance decision removes the `match_id → is_public_row` classification join that gate
   would guard; a tested-but-uncalled guard is the dead-guard anti-pattern. If a future change adds
   the split, it adds the guard with it.

4. **KDE goldens: frozen oracle → runtime backend parity.** `test_golden_fft_scalars` locked fft
   scalar fidelity against the 36k-sample bundled model. Post-strip that model can no longer serve
   density, and no practical locally-fit fixture reaches the real kernel-width regime (kernel width
   scales as `neff**(-1/6)`; 4000 samples is 1.40× broader, and more estimators do not help — a
   measurement run specifically to test whether the caveat was stale confirmed it is not), so the
   real-model fft-fidelity lock is retired (coverage recorded here rather than absorbed silently).
   The other KDE goldens (`test_golden_continuous` / `_discrete_mode` / `_fft_cic_scalars`) are
   **converted from a frozen oracle to a RUNTIME backend-parity check**: they compute the reference
   with the closed-form `vectorized` backend on the same fresh fit, in the same run, and compare
   cpu-numba / fft-cic against it. A *frozen* golden was portable only for the FIXED bundled model;
   a fitted-model oracle is not byte-stable across sklearn/numpy versions, so a committed golden
   false-fails on any CI leg other than the one it was generated on (this was caught by CI on the
   first push — 3.10 passed, 3.11/3.12 failed). `vectorized` is scipy-equivalent
   (`test_vectorized_kernel_matches_scipy`, 1e-9), so the runtime check preserves the same
   end-to-end parity. The committed `ghost_gk_kde_golden.npz` is reduced to the deterministic
   **query features only** (no stored density oracle).

5. **The public Hub `silly-kicks/ghost-gk-v1` is NOT touched by this release, and stays as it is.**
   The live Hub artifact has no chirality block, so 4.51.0+ `load()` rejects it for every
   `from_hub` consumer, and it still carries the raw-position arrays across its revisions. This
   release changes only the library and the bundled `default`. Restoring `from_hub` (uploading a
   stripped, chirality-carrying `full`) and withdrawing the exposed revisions are HuggingFace
   writes on live public infrastructure, and are part of the **owner-held ghost-GK disclosure
   remediation** — a separate approved workstream, **not** scheduled by this PR. A stripped `full`
   produced by the post-strip `save()` is staged for that workstream; the upload runbook
   (`docs/research/tf19_pr2/hf_upload_instructions.md`) carries a pre-upload parameters-only
   assertion.

### `from_variant("public")` alias fix (folded in by owner decision)

`XShotOccurrenceModel.from_variant("public")` and the xCross equivalent returned the Hub-hosted,
restricted `sc_extended` artifact: no bundled `public/` dir exists, so the name fell through to
`from_hub` and was cached under `"public"`. It is a stale alias — 4.9.0 reserved the name for a
public Hub artifact never created; PR-S118 added `sc_extended` alongside it without re-auditing.
Fix: an explicit `_VARIANT_ALIASES = {"public": "default"}` resolved **before** the cache (the
bundled `default` metadata already declares `shipped_variant: "public"`, so the alias is the
literal truth, not a shim), and `_HUB_VARIANTS = frozenset({"sc_extended"})`. No variant `Literal`
— that would enumerate `sc_extended` and promote a Hub-only restricted artifact to a typed
first-class option. A serve-time identity gate pins it; ADR-038's gate operates at training time
and structurally cannot observe a loader serving a mislabelled artifact. Shipped as its own
CHANGELOG line so a user-actionable correction keeps its visibility.

## Consequences

- **Hyrum / re-materialize:** `ghost_gk_density_spread` is retired; the lakehouse re-materializes
  the (unread) passthrough column out. `predict_mean` positions (`ghost_gk_x/y`) are byte-identical
  — no VAEP retrain. Consumers pinned `<= 4.53.0` cannot read a 1.3.0 artifact (forward-incompatible
  format), a version-pin consideration for anything Hub-hosted.
- The `from_variant("public")` fix changes what an explicit `"public"` caller receives — from the
  restricted Hub artifact to the reproducible bundled model. Zero callers pass `"public"` today.
- The ghost-GK KDE capability (`predict_density`, `GhostGkDensity`, the ADR-013/014 backends) is
  unchanged for locally-fit models.
