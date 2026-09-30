---
name: corpus-drivers
description: Use when building or editing a silly-kicks corpus driver — a scripts/ build_*/validate_*/measure_* pass over matches — needing resume-before-load, the scripts/_driver.py for_each seam, shard generation-dir + schema token, .excluded.json, assert_conservation/_require_injective, require_clean_tree provenance, the docs/research memo landing, or the PINING_FOR_THE_DATA_TOKEN / databricks corpus env.
---

# Corpus drivers

Read `docs/howto/corpus-drivers-runbook.md` and follow it. It is the canonical runbook; this shim
only points at it and owns no substance.

The doc covers adopting the shared `scripts/_driver.py` seam (ADR-052), `for_each` with
resume-before-load, exclusions + conservation + combination (`res.shard_keys`), the shard-schema
token, `require_clean_tree` provenance (ADR-037), the `docs/research/<topic>/` memo-landing
convention, the corpus env vars, and a step list for a new driver. It points at `_driver.py`'s
docstring and `docs/context/corpus-drivers.md` for field-level detail.
