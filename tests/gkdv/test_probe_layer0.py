"""TF-19 A+2 Task 2: the Layer-0 instrument-validity verdict.

Pure function over ALREADY-POOLED corpus statistics (never per shard). The two `arm_unscoreable`
short-circuits (thin domain OR any all-NaN leg) fire FIRST and are asserted DISTINCT from
`instrument_void` -- a velocity-less arm or a thin domain must never read as "the instrument is
broken". The `either-leg` test is what makes the void `and` load-bearing (see the module docstring's
discrimination note): both-pass and both-fail alone cannot separate `and` from `or`.
"""

from __future__ import annotations

import numpy as np

from silly_kicks.gkdv._probe import MIN_DOMAIN_FRAMES, layer0_instrument_verdict


def test_live_instrument_is_valid():
    # saturating median 0.5 >= 5x realistic 0.05 => valid (either leg suffices).
    v = layer0_instrument_verdict(
        realistic_abs=np.full(300, 0.05),
        saturating_abs=np.full(300, 0.5),
        placebo_p95=0.02,
        n_domain=300,
    )
    assert v == "instrument_valid"


def test_dead_instrument_is_void():
    # saturating flat (0.04): NOT >= 5x realistic (0.05->0.25) AND NOT > placebo p95 (0.10) => void.
    v = layer0_instrument_verdict(
        realistic_abs=np.full(300, 0.05),
        saturating_abs=np.full(300, 0.04),
        placebo_p95=0.10,
        n_domain=300,
    )
    assert v == "instrument_void"


def test_either_leg_suffices_placebo_only_is_valid():
    # Discrimination case (makes the void `and` load-bearing): saturating 0.20 clears the placebo
    # band (0.20 > 0.10) but NOT the 5x realistic multiple (0.20 < 0.25). Either leg suffices, so
    # valid. Under a buggy `void iff not-multiple OR not-placebo` this would flip to void; the
    # both-pass/both-fail tests cannot see that flip (their two legs agree).
    v = layer0_instrument_verdict(
        realistic_abs=np.full(300, 0.05),
        saturating_abs=np.full(300, 0.20),
        placebo_p95=0.10,
        n_domain=300,
    )
    assert v == "instrument_valid"


def test_velocity_less_arm_is_unscoreable_not_void():
    v = layer0_instrument_verdict(
        realistic_abs=np.full(300, np.nan),
        saturating_abs=np.full(300, np.nan),
        placebo_p95=np.nan,
        n_domain=300,
    )
    assert v == "arm_unscoreable"  # asserted DISTINCT from instrument_void
    assert v != "instrument_void"


def test_zero_realistic_baseline_does_not_vacuously_validate():
    # On the zero-dominated Delta-DAS arm real_med can be 0; then `sat_med >= 5*0 = 0` is trivially
    # TRUE and would vacuously validate a DEAD instrument. A sound verdict falls to the placebo leg:
    # sat_med (0.01) below the placebo band (0.05) -> instrument_void, not a fabricated valid.
    v = layer0_instrument_verdict(
        realistic_abs=np.zeros(300),
        saturating_abs=np.full(300, 0.01),
        placebo_p95=0.05,
        n_domain=300,
    )
    assert v == "instrument_void"  # must NOT pass vacuously via 5*0=0


def test_zero_realistic_baseline_still_valid_via_the_placebo_backstop():
    # real_med == 0 but the saturating dose clears the placebo band -> the instrument DOES respond;
    # the placebo leg is the real backstop when there is no realistic baseline.
    v = layer0_instrument_verdict(
        realistic_abs=np.zeros(300),
        saturating_abs=np.full(300, 0.5),
        placebo_p95=0.05,
        n_domain=300,
    )
    assert v == "instrument_valid"


def test_thin_domain_is_unscoreable_not_void():
    v = layer0_instrument_verdict(
        realistic_abs=np.full(3, 0.05),
        saturating_abs=np.full(3, 0.04),
        placebo_p95=0.10,
        n_domain=3,
    )
    assert v == "arm_unscoreable"  # a thin domain must NOT read as "broken"
    assert v != "instrument_void"
    assert MIN_DOMAIN_FRAMES > 3
