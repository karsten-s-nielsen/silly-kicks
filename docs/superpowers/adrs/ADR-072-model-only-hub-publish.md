# ADR-072: HuggingFace publish uploads a MODEL-ONLY allowlist, fail-closed on any nested path

| Field | Value |
|---|---|
| **Date** | 2026-08-26 |
| **Status** | Accepted |
| **Deciders** | Karsten Nielsen |

## Context

The trainers write their internal artifacts — `_feature_cache/`, `shards/` (RAW per-match tracking
parquets from restricted owner-tier providers), and (xCross) `_probe_sample/` /
`_probe_sample_comparison/` — into the **same directory** as the distributed model files
(`model.json` / `rfcde_weights.npz`, `metadata.json`, `metrics.json`, `SHA256SUMS`). The publish
scripts called `HfApi().upload_folder(artifact_dir, ...)`, which uploads that directory
**recursively**.

During the 4.94.0 Hub re-fit this shipped the training internals to five **public** repos (file
counts 127 / 197 / 126 / 196 / **370**), briefly exposing raw restricted tracking frames — a
licensing violation and a direct contradiction of the model cards' "no raw provider tracking data"
promise. It was remediated by deleting the folders and recreating the repos model-only from clean
directories (owner decision: full purge, no provider notification). The forcing function: `--verify-only`
only checks that the *model* loads; **nothing inspected the folder's file set** before upload.

## Decision

Publishing goes through `scripts/_hub_publish.py::upload_model_only()`, which uploads a fixed
**model-only allowlist** (`allow_patterns`) and then **fail-closes (`SystemExit`) if the repo carries
any nested path** afterwards (a `foo/bar` filename == a leaked subdirectory). All three
`publish_{xshot,xcross,ghost_gk}.py` route through it.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Keep `upload_folder(dir)`, "just point it at a clean dir" | no code | relies on the operator curating the dir every time; `--verify-only` cannot catch it | the exact failure mode that caused the incident — discipline is not a control |
| B. Clean the artifact dir in-place before upload (delete caches) | simple | destroys the trainer's cache/probe (needed for re-runs / audit); mutates a dir the operator may reuse | side-effect on the source of truth |
| C. `ignore_patterns` for the known bad dirs | small | a denylist misses a NEW training-internal dir a future trainer adds | allowlist is closed; denylist is open |
| D. Allowlist upload + post-upload nested-path guard (chosen) | closed set; the guard also catches a nested path left by an earlier bad publish (which the allowlist cannot remove) | one shared helper to maintain | — |

## Consequences

### Positive

- The leak class is structurally unreachable: only the allowlisted files upload, and a nested path
  (from any source, including a prior bad publish) raises loudly instead of sitting silently on a
  public repo.
- One shared seam (`_hub_publish.py`) for all three publish scripts; the allowlist is declared once.
- Unit-tested without the network (`tests/scripts/test_hub_publish_guard.py`): allowlist shape,
  `allow_patterns` passthrough, clean-repo passes, and a nested `shards/` / `_probe_sample/` /
  `_feature_cache/` path fails closed.

### Negative

- A new model-artifact filename must be added to `MODEL_ONLY_ALLOWLIST` or it silently will not
  publish. This is the safe direction (under-publish, never over-publish) and is caught by the
  publish script's own round-trip verification (the model would fail to load from the Hub).

### Neutral

- The guard checks the *repo's* file set after upload, so it also flags a pre-existing leak on a repo
  the operator did not just create — surfacing historical mistakes rather than hiding them.

## Related

- **Issues / PRs:** `#217` (silly-kicks 4.94.0)
- **Code:** `scripts/_hub_publish.py`, `scripts/publish_{xshot,xcross,ghost_gk}.py`,
  `tests/scripts/test_hub_publish_guard.py`
- **ADRs:** ADR-038 (owner-tier corpus + visibility — the licensing basis for "no raw data"),
  ADR-011 (trained-model lifecycle)

## Notes

The incident and the durable "verify the Hub file count (~5, not ~100s) after every publish" rule are
also recorded in the session incident memory. The model cards' safety statement is now the accurate
"only parameters, no raw data" (an earlier "every leaf aggregates ≥ N samples by `min_child_weight`"
phrasing was dropped — the real re-fit `min_child_weight` values are 13/12/7/2, so the ≥N claim was
false for xCross-PO).
