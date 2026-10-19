---
name: hf-campaign
description: Use when running, resuming, inspecting or adopting an HF Jobs campaign — `python -m scripts.hf_jobs campaign <file> …`, preflight, profile, launch/resume, status, retrieve, verify, adopt, publish (--sandbox), the run ledger, datasets record — always on cpu-upgrade, with owner gates on every commit, push and Hub publish.
---

# HF campaigns

Read `docs/howto/hf-jobs-training.md` and follow it. It is the canonical runbook; this shim owns no
substance. Three owner gates, every time:

- Commit: `adopt` stops at the commit gate; commit only after an explicit owner "yes".
- Push: a separate explicit "yes".
- Hub publish: an owner go per `publish` run (`--sandbox` first; a public repo needs `--i-have-owner-approval`).
