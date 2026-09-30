---
name: construct-validity
description: Use when validating that a silly-kicks metric measures the skill it claims — the three construct-validity gates (responsiveness / discrimination-vs-null ICC / predictive-transfer), the GO/NO-GO reading and metrics.json memo shape, the "state what the number does NOT measure" caveat pattern, or adding a validate_*.py harness.
---

# Construct-validity gates

Read `docs/howto/construct-validity.md` and follow it. It is the canonical reference; this shim only
points at it and owns no substance.

The doc defines the three gates once (responsiveness, discrimination-vs-permutation-null,
predictive/transfer), pins the verdict/memo shape to a real `metrics.json`
(`docs/research/gk_decision_construct_validity/`), documents the "state what it does NOT measure"
caveat pattern (`docs/research/pass_risk_calibration/`), and says how to add a `validate_*.py`.

For the corpus-driver mechanics a `validate_*.py` is built on, see
`docs/howto/corpus-drivers-runbook.md`.
