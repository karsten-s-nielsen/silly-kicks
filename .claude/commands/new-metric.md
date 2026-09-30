---
description: Start a new silly-kicks metric — open the authoring checklist and follow it.
---

# /new-metric

You are adding a new metric to silly-kicks. Invoke the `authoring-a-metric` skill and follow
`docs/howto/authoring-a-metric.md` top-to-bottom.

Decide first whether it is an action-coupled `add_*` enricher or a standalone `compute_*` aggregator,
then wire each step the checklist names (package layout; `metric_contracts` registration, ADR-098;
`PURITY_ENTRIES`, ADR-033; tracking liveness + call-convention registries; `*_xfns` leak guard;
`feature_glossary` + `NOTICE` + C4; the additive-vs-value-changing release decision). For the
validity gate and its corpus driver, see `docs/howto/construct-validity.md` and
`docs/howto/corpus-drivers-runbook.md`.

$ARGUMENTS
