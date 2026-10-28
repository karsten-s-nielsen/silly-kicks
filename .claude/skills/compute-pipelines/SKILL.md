---
name: compute-pipelines
description: Use when writing or migrating a silly-kicks trainer (train_*.py) or a corpus driver (build_/validate_/measure_* that walks matches) — the ADR-128 stage/unit contract (PIPELINE, Stage, ItemStage), single-threaded units (opt-in `thread_safe` multi-thread fits), sorted-key reduces, ordered_values for inner pools, checksum resume, the W1N1-vs-W2N2 parity test, the profile stage, the scaffolder, or the born-ready compute gate.
---

# Compute pipelines

Read `docs/howto/compute-pipelines.md` and follow it. It is the canonical runbook; this shim owns no
substance. Start a new module with `python -m scripts.hf_jobs scaffold trainer|driver <name>`; it passes
`tests/scripts/test_compute_contract.py` on day one.
