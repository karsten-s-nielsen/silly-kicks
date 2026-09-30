---
name: authoring-a-metric
description: Use when adding or creating a new metric in silly-kicks — a pure-fn add_* enricher or compute_* aggregator, wiring metric_contracts (ADR-098), PURITY_ENTRIES (ADR-033), the tracking liveness gate, the call-convention registry, *_xfns leak guard, feature_glossary, NOTICE, or deciding additive-vs-value-changing (version/Hyrum/lakehouse re-materialize).
---

# Authoring a metric

Read `docs/howto/authoring-a-metric.md` and follow it top-to-bottom. It is the canonical checklist;
this shim only points at it and owns no substance.

The doc covers, in order: `add_*` vs `compute_*`; package layout; test + validity-driver placement;
the `metric_contracts` registry (ADR-098); the `add_*` purity registry (ADR-033); the tracking
liveness gate + call-convention registry; the liveness-fixture precondition; the `*_xfns` leak guard;
`feature_glossary` + `NOTICE` + C4; and the additive-vs-value-changing release decision.

For the validity gate a metric must pass, see `docs/howto/construct-validity.md`. For the corpus
driver that validates it on real data, see `docs/howto/corpus-drivers-runbook.md`.
