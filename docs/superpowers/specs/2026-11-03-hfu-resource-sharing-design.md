# HFU driver — resource-sharing overhaul — design

Status: APPROVED (A and B, round 2, 0 blocking); shipped in PR #299. The decisions, in full, are the ADR-128
amendment of 2026-11-03; the review rounds are in git history.

## Why

The first retrain on the minimal driver launched ~1,836 small jobs and reduced 0/21 instances: mount-sidecar 401
flakes on the private dataset, HTTP 429s from the bucket API, OOM churn on `ghost_gk` extract, and per-step
coverage re-reads of every manifest.

## The five fixes (driver-only; tunable in `cycle1.toml`)

1. Bounded job-sizing: `ceil(units / units_per_job)` jobs per stage, at most `max_shards`.
2. A per-dataset mount cap: in-flight plus same-pass launches over every mount dataset of an instance.
3. Coverage set+push: a per-`run_key` read-once cache (`_partitions/_coverage/state.json`), recomputed in
   `created_at` order; failure attribution by locus (a mount fault is per dataset, never per unit).
4. Memory sizing and escalation: a per-`job/stage` peak-RSS hint caps workers; an over-cap unit runs alone on the
   box (`--escalate`, W=1).
5. Launch pacing (`launch_batch`) and a contention backoff (a 429, or rising mount faults), up to `backoff_max_s`.

Scope: `driver.py`, `campaign.py` config and `cycle1.toml`; no wheel change; the version stays 4.139.1.
