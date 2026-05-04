"""Scaffolding contract test for silly_kicks.tracking.pressure module.

Verifies the params-dataclass + Method-literal + validator surface BEFORE
any kernel work lands. Pin-tests the public-surface invariants.
"""

from __future__ import annotations

import pytest

from silly_kicks.tracking.pressure import (
    AndrienkoParams,
    BekkersParams,
    LinkParams,
    Method,
    validate_params_for_method,
)

# Method is a Literal alias; reference it so importers exercise the symbol.
_ = Method


def test_andrienko_defaults() -> None:
    p = AndrienkoParams()
    assert p.q == 1.75
    assert p.d_front == 9.0
    assert p.d_back == 3.0


def test_link_defaults() -> None:
    p = LinkParams()
    assert p.r_hoz == 4.0
    assert p.r_lz == 3.0
    assert p.r_hz == 2.0
    assert p.angle_hoz_lz_deg == 45.0
    assert p.angle_lz_hz_deg == 90.0
    assert p.k3 == 1.0


def test_bekkers_defaults() -> None:
    p = BekkersParams()
    assert p.reaction_time == 0.7
    assert p.sigma == 0.45
    assert p.time_threshold == 1.5
    assert p.speed_threshold == 2.0
    assert p.max_player_speed == 12.0  # canonical UnravelSports per spec section 4.1
    assert p.use_ball_carrier_max is True


def test_dataclasses_are_frozen() -> None:
    p = AndrienkoParams()
    with pytest.raises((AttributeError, Exception)):
        p.q = 2.0  # type: ignore[misc]


def test_validate_params_for_method_match() -> None:
    validate_params_for_method("andrienko_oval", AndrienkoParams())
    validate_params_for_method("link_zones", LinkParams())
    validate_params_for_method("bekkers_pi", BekkersParams())
    validate_params_for_method("andrienko_oval", None)  # None means use defaults


def test_validate_params_for_method_mismatch() -> None:
    with pytest.raises(TypeError, match=r"andrienko_oval.*expects AndrienkoParams.*got LinkParams"):
        validate_params_for_method("andrienko_oval", LinkParams())
    with pytest.raises(TypeError, match=r"bekkers_pi.*expects BekkersParams.*got AndrienkoParams"):
        validate_params_for_method("bekkers_pi", AndrienkoParams())


def test_validate_unknown_method() -> None:
    with pytest.raises(ValueError, match=r"Unknown method.*not_a_method"):
        validate_params_for_method("not_a_method", None)  # type: ignore[arg-type]
