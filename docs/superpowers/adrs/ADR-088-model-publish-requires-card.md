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

## Amendment (combined-cycle completion, 2026-10-02): the card-only seam

The decision above covers **model** publishes. Card-only pushes, which refresh a Hub README without
re-uploading weights, had no seam: the two earlier ones (`1b56ad8`, `aae6fdb`) were ad-hoc README
uploads with no guard and no read-back. This amendment closes that gap. Status stays **Accepted**.

- **One card-only seam**, `scripts/_hub_publish.publish_card_only`, driven by `scripts/publish_model_card.py`.
- **Registered cards only.** `CARD_SOURCE` maps every repo of the `silly-kicks` org to its in-repo card
  (the nine `docs/huggingface/model-cards/*-model-card.md` files and `silly_kicks/xsuccess/weights/MODEL_CARD.md`).
  The CLI accepts only a registered `--repo-id` and never a free card path, so a card cannot reach the wrong repo.
- **Existing repos only.** The seam reads `model_info` first and never calls `create_repo`.
- **LF bytes in both seams.** `card_bytes` normalizes CRLF to LF. A Windows checkout with `core.autocrlf=true`
  holds CRLF, while every Hub README is LF (measured 2026-10-02). `publish_model_with_card` stages its card
  through the same function, so both seams publish identical bytes for the same card.
- **No-op when unchanged.** An unchanged card is not re-uploaded.
- **Read-back.** After an upload the README is re-downloaded and must be byte-identical, else the push
  fails loudly.
- `--verify-only` reports changed / unchanged and the two SHA-256s without uploading.

`scripts/validate_hub_variants.py` checks the result: per repo, whether the Hub README equals its
registered card (`--require-cards-match` gates it after a push). Tests use a fake Hub API
(`tests/scripts/test_publish_model_card.py`).

The same driver also loads every frame-geometry repo through `from_hub` (anonymously) and scores a
canonical frame. A load the library itself refuses (its fail-closed integrity checks) is recorded per
repo under `load_refused`, not raised, so one bad repo cannot hide every other repo's result:

- a refused Hub-only repo always fails the run, because no planned push would replace it;
- a refused mirror fails only under `--require-mirrors-match-wheel`, the post-push check, because the
  republish is what replaces it;
- any other exception still propagates.

Measured 2026-10-02: `ghost-gk-sweeper-v1` and `ghost-gk-sweeper-position-only-v1` (Hub
`training_commit` `adafb72`, pre-ADR-089) are refused on chirality; the post-release mirror republish
replaces them.
