# ADR-088: Every model publish goes through one card-required seam

| Field | Value |
|---|---|
| **Date** | 2026-09-05 |
| **Status** | Accepted |
| **Deciders** | Karsten Nielsen |

## Context

silly-kicks distributes trained model artifacts (xShot, xCross, ghost-GK, ghost-outfield) to the
Hugging Face Hub via per-model `scripts/publish_*.py` scripts, all delegating to
`scripts/_hub_publish.upload_model_only` — a fixed allowlist that fail-closes on any nested repo path
(the 4.94.0 raw-shard-leak guard, ADR-072).

Two failure modes recurred and shipped undetected:

1. **The model card was dropped from the release.** The card (a `docs/huggingface/model-cards/*.md`
   file) had to be *manually* copied to `README.md` in the artifact dir before publishing. A manual
   pre-step is exactly what gets forgotten — the TF-60 PR3 sweeper cards were dropped and had to be
   fixed by a follow-up PR, and the TF-60 PR5 ghost-outfield cards were nearly dropped again.
2. **The weights themselves were skipped.** The ghost-outfield weights file is `model.npz`, which was
   never in `MODEL_ONLY_ALLOWLIST` (which listed `model.json` for xShot/xCross and `rfcde_weights.npz`
   for ghost-GK) — so the publisher would have uploaded metadata + `SHA256SUMS` + README but **no
   weights**, and no test covered the upload path (the publisher tests only exercised `--verify-only`).
   Separately, ghost-GK and ghost-outfield lacked the `create_repo(exist_ok=True)` call that xShot and
   xCross had, so a brand-new repo could error.

These are the same class of defect: a publish that silently produces an *incomplete* repo (no card, no
weights, or a missing repo) rather than failing loudly.

## Decision

Route **every** `publish_*.py` script through a single `_hub_publish.publish_model_with_card` seam that
(a) **requires a model card** and refuses — before any network call — a publish without one, (b)
creates the repo idempotently (`create_repo(exist_ok=True)`), and (c) stages the allowlisted model
files **plus the card as `README.md`** into a temp dir and uploads model-only. A card-less, weightless,
or missing-repo publish is now unrepresentable. `model.npz` is added to the allowlist.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Keep manual card staging, just fix the allowlist | Minimal change | Leaves the card a forgettable manual pre-step — the exact recurrence | Fixes one gap, not the class |
| B. A CI/lint gate that a publish PR touches the card | Catches it in review | The publish is an operator action on the DGX, not a PR; a gate cannot see it | Wrong layer |
| C. **One card-required publish seam** (chosen) | The card/weights/repo cannot be dropped by construction; one place to get right; unit-testable with a fake api | A small API change to four scripts | — |

## Consequences

### Positive
- A model publish that omits the card, the weights, or the repo is impossible; the failure is loud and pre-network.
- One seam to test and reason about; `--model-card` is a required, discoverable input on every publisher.
- Unit-tested with a fake HfApi (create-repo + card-as-README staging + refuse-missing-card), so the upload path is finally covered.

### Negative
- `--model-card` is now a required argument for a real publish (a breaking change to the four publisher CLIs — intended).

### Neutral
- Scripts-only: `scripts/` is not in the wheel (`packages = ["silly_kicks"]`), so the shipped library is byte-identical. The version bump to 4.110.0 signals a significant change to the release/publish process, not a library-API change.

## Related
- **ADRs:** builds on ADR-072 (model-only leak guard); the card-in-release-commit principle is ADR-087 / the PR3 lesson.
- **Issues / PRs:** PR-S181.
