# HFU driver — resource-sharing overhaul — implementation plan

Spec: `docs/superpowers/specs/2026-11-03-hfu-resource-sharing-design.md` (A+B APPROVE, round 2, 0 blocking).
Status: done; shipped in PR #299. One task per fix (1–5), each test-first in `tests/scripts/hf_jobs/test_hf_driver.py`
(job sizing, mount cap, coverage cache and attribution, escalation, pacing and backoff), then docs (ADR-128
amendment 2026-11-03, CHANGELOG) and the owner-gated amend. The full task text is in git history.
