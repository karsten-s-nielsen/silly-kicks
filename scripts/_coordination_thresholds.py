"""Pre-registered TF-58 hypothesis thresholds (spec 8.5) -- referenced, never inlined.

One module so the D3 validation driver and its tests read the SAME frozen constants; a change here is a
pre-registration change, visible in the diff. ``tests/scripts/test_coordination_driver_modules.py`` pins every
value against the spec table.
"""

from __future__ import annotations

# H1 -- centroid longitudinal phase-locking is tighter than lateral.
H1_P = 0.01
H1_LONGITUDINAL_MEAN_WITHIN_DEG = 30.0

# H2 -- team spread cross-correlates near zero lag.
H2_POSITIVE_SHARE = 0.70
H2_MEDIAN_ABS_LAG_S = 1.0

# H3 -- early-third relative phase differs by the possession's terminal outcome.
H3_P = 0.05

# H4 -- collective oscillation is slow (median frequency below the ceiling).
H4_BELOW_CEIL_SHARE = 0.95
H4_CEIL_CPM = 1.0
H4_P = 0.01
#: Moura 2013 measured team AREA and SPREAD (spec 8.5's H4 row); H4 tests exactly these, each on its own.
H4_SIGNALS = ("convex_hull_area", "spread")

# H5 -- group synchrony is higher longitudinally, and in vs out of possession (TOST).
H5_P = 0.01
H5_TOST_MARGIN = 0.05
H5_TOST_ALPHA = 0.05

# H7 -- relative stretch alternates bimodally, switching around possession changes.
H7_BC_THRESHOLD = 5.0 / 9.0
H7_BIMODAL_SHARE = 0.50
H7_SWITCH_WINDOW_S = 10.0
H7_SURROGATE_PERCENTILE = 95.0
#: The H7 time-shift surrogate: the spec 8.1 K convention, and ONE pre-registered seed (the pre-registration
#: date) -- D2's gate and D3's report must draw the same surrogates on the same data (review A-41/A-53).
H7_N_SURROGATES = 199
H7_SEED = 20260926

#: D3's final-validation surrogate count (spec 8.5): a full baseline draw. Same K = 199 rank convention as §8.1, but
#: its own constant (the D3 family-metric surrogates, distinct from the H7 switch null); single-sourced here so the D3
#: driver references it rather than inlining the literal (test_driver_inlines_no_hypothesis_threshold_literal).
D3_N_SURROGATES = 199

# Shared: a stoppage leg splits a coordination window at a dead-ball interval longer than this.
STOPPAGE_LEG_MIN_S = 25.0

# --- A-09: per-cell reliability power verdict (spec §8.5, owner batch-3) ---------------------------------------
# Pre-registered BEFORE the corpus is scored, so ``min`` is not a post-hoc free parameter across the ~2000
# per-construct cells. A cell that trips any of these is terminal "unmeasurable" and is NEVER pooled up a level to
# rescue power. Values are owner-ratifiable at the commit-1 review.
#: Fewer than this many measured retest groups -> unmeasurable ("n<min"). An ICC(1) / circular-reliability point
#: estimate on 2 obs/group is unstable below ~30 groups (standard ICC sample-size guidance).
RELIABILITY_MIN_N_GROUPS = 30
#: A reliability 95% CI wider than +/- this (half-width) -> unmeasurable ("ci_too_wide"), even if nominally
#: powered: a CI half-width above 0.25 spans a third of the [0, 1] reliability scale.
RELIABILITY_MAX_CI_HALFWIDTH = 0.25
#: A circular-mean construct whose overall mean resultant length is below this -> unmeasurable ("Rbar->0"):
#: rotation-invariant circular reliability is undefined as concentration -> 0 (below R-bar 0.10 the distribution is
#: effectively uniform and the circular-variance ratio is noise-dominated).
CIRCULAR_RELIABILITY_MIN_RBAR = 0.10
#: An occlusion-decile bin with fewer than this many matches cannot estimate a stable median absolute error, so a
#: construct whose curve needs that bin is not estimable per-construct (spec §8.2). Set from the 71-match
#: fully-observed occlusion corpus (GS 64 + IDSSE 7); recorded in ``derivation.json``.
OCCLUSION_MIN_MATCHES_PER_BIN = 5
#: The closed ``unmeasurable_reason`` vocabulary (spec §8.5 / the metrics.json schema), in threshold order.
UNMEASURABLE_REASONS = ("n<min", "ci_too_wide", "Rbar->0")

# H6 is descriptive (no pass/fail); the gate is every other registered hypothesis.
GATED_HYPOTHESES = ("H1", "H2", "H3", "H4", "H5", "H7")
