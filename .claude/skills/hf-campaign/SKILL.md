---
name: hf-campaign
description: Use when running, resuming, inspecting or adopting an HF Jobs campaign — `python -m scripts.hf_jobs campaign <file> …`, preflight, launch/resume (--dry-run), status, cancel, retrieve, verify, compare, adopt, publish (--sandbox), prune, the run ledger, datasets record — always on cpu-upgrade, with owner gates on every commit, push and Hub publish.
---

# HF campaigns

Read `docs/howto/hf-jobs-training.md` and follow it. It is the canonical runbook; this shim owns no
substance. Three owner gates, every time:

- Commit: `adopt` stops at the commit gate; commit only after an explicit owner "yes".
- Push: a separate explicit "yes".
- Hub publish: an owner go per `publish` run (`--sandbox` first; a public repo needs `--i-have-owner-approval`).
- Bucket delete: `prune` only lists; `prune --delete --i-have-owner-approval <commit7>` needs an owner yes per run.

`preflight` runs seven checks (the seventh is the org's storage headroom). After a launch that changes no
model, `compare --staging NEW --baseline OLD` is the parity proof: adopt and publish nothing when it passes.
