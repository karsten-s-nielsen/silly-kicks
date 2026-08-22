"""Task 10 (R5): pre-registered apply-gate thresholds for the cover-shadow sigma/lambda re-tune.

Every threshold is a NAMED constant FIXED BEFORE THE RUN. Changing one after seeing results invalidates
the gate, so the apply gate (Task 12) and the deployment gate (Task 6b) REFERENCE these names -- never
inline literals -- and a test pins that they do, so the bar cannot move silently. The values are a
judgement pinned in the spec (docs/superpowers/specs/2026-08-20-*); re-pin only BEFORE a run, with a reason.
"""

from __future__ import annotations

#: trajectory-validated subset must be >= 30% of intercepted failures, else conjunct-1 is unmeasurable -> null
MIN_COVERAGE: float = 0.30

#: model top-1 must beat the geometric proxy by >= 5 pts on the validated subset (hard: proxy near-ceiling, R1)
MIN_RECEIVER_MARGIN: float = 0.05

#: < half the sigma/lambda shift may be attributable to the lane-pressure open-target channel (H2/R3)
MAX_BIAS_SHARE: float = 0.50

#: R6 -- max fraction of failed passes whose failure mode (interception vs out) is un-classifiable before
#: the split is trusted; above it, GS next-action tagging is too noisy to route model-vs-trajectory targets
MAX_AMBIGUOUS_RATE: float = 0.20
