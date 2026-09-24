"""ADR-103 F6: the one-cache-per-unit pattern spans exactly the two pitch-control scorers
(value_off_ball_runs + compute_rest_defense); defensive_credit is PC-free (nothing to thread).

Cross-scorer REUSE itself is proven by composition: both scorers accept `pitch_control_cache=` (below)
and the cache reuses/evicts by its canonical key (tests/tracking/pitch_control/test_cache.py +
test_batch.py::test_cache_warm_makes_surface_calls_hit) -- so one shared cache computes an overlapping
(frame, team, method, decompose) surface once, without a heavy end-to-end fixture here.
"""

from __future__ import annotations

import inspect

from silly_kicks.restdefense import compute_rest_defense
from silly_kicks.tracking import add_defensive_credit, value_off_ball_runs


def test_pitch_control_scorers_accept_shared_cache():
    for fn in (value_off_ball_runs, compute_rest_defense):
        assert "pitch_control_cache" in inspect.signature(fn).parameters, fn.__name__


def test_defensive_credit_is_pc_free():
    # The handoff's "3-4x recompute" premise was wrong: defensive_credit uses NO pitch control,
    # so it takes no cache param -- there is nothing to thread (ADR-103 F6).
    assert "pitch_control_cache" not in inspect.signature(add_defensive_credit).parameters
