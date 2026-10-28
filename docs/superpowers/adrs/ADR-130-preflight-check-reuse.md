# ADR-130: Incremental preflight — reuse the heavy `check_plans` toy train by input key

| Field | Value |
|---|---|
| **Date** | 2026-10-29 |
| **Status** | Proposed (lands with the HF full-utilization cycle's code commit) |
| **Deciders** | Karsten Nielsen; reviewers A-Deux and B-Deux (spec r1–r5, plan r1–r4) |

## Context

Preflight (ADR-128 spec 5.4) runs six no-spend checks before `launch`. Five are cheap and observe live state
(`check_ref`, `check_lock`, `check_api`, `check_datasets`). One is not: `check_plans` runs every job's `--plan`
and, once per module, a toy `--run-all` — about 20–30 minutes of real training. Re-running that train on every
preflight, when nothing that affects its result changed, is pure waste.

## Decision

Cache ONLY the `check_plans` toy train, keyed by its inputs, in a single file in the preflight report store
(`_campaign/preflight_cache.json`), owned by `preflight.py` and NOT keyed by commit. A stored PASS for the
current key is reused byte-identically (the train is skipped); any input change re-runs it. Only a passing
result is cached. A missing or unreadable cache forces a full recompute (fail-open to correctness, never a
crash).

The `check_plans` key digests exactly what changes the train's result:

- `campaign_sha256` (the campaign file);
- the `uv.lock` digest;
- the pinned image digest (`launch.IMAGE`);
- the `datasets.lock.json` digest;
- a tree digest of `scripts/**/*.py`, `silly_kicks/**/*.py` and `pyproject.toml`.

The tree digest covers the IMPORTABLE source (`*.py`) only, by design (owner-ratified 2026-10-29; reviewer B
D1-IMPL-14). The toy train validates the PLAN, which depends on the code the subprocesses import; trained
weights (`.npz`/`.json`) and other data under those trees are loaded at runtime by a fit, do not change the
plan, and the `datasets.lock.json` digest already pins the data — so including them would re-run the 20–30-minute
train on every weight/data commit for no plan-correctness gain. The paths are anchored to the repo root, never
the process cwd, so running the controller from another directory cannot collapse the digest and reuse a stale pass.

The cheap checks and `check_ref` are NEVER cached — they always re-observe. `check_ref` re-checks every run
because a force-push can un-green a commit; the cheap checks are cheap. Only `check_plans` is wrapped
(`cached_check_plans`).

## Consequences

- A re-run of preflight on an unchanged tree skips the 20–30 min toy train and reuses its pass; the cheap
  checks still re-observe, so a changed ref, lock, API or dataset is still caught.
- A change to any keyed input (the campaign file, `uv.lock`, the image, `datasets.lock.json`, or any
  `scripts`/`silly_kicks` source or `pyproject.toml`) re-runs the train.
- The cache is advisory. Deleting it, or a corrupt file, costs one full recompute, never a wrong pass.

## Alternatives considered

| Option | Why rejected |
|---|---|
| Key by commit | A dirty working tree has no commit; a re-run at the same commit after a source edit must re-run the train. A content key handles both. |
| Cache every check | The cheap checks observe LIVE state (a force-push, a moved dataset revision); caching them would serve a stale pass. |
| No cache | The 20–30 min train on every preflight is the waste this ADR removes. |
