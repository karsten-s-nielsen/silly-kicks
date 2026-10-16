"""TF-46 corners param-dataclass shape gates."""

from __future__ import annotations

import dataclasses

import pytest

from silly_kicks.corners import (
    ChanceQualityParams,
    CornerWindowParams,
    RoutineParams,
    RunEncodingParams,
)

_ALL_PARAMS = [CornerWindowParams, RunEncodingParams, RoutineParams, ChanceQualityParams]


@pytest.mark.parametrize("cls", _ALL_PARAMS)
def test_frozen(cls):
    obj = cls()
    with pytest.raises(dataclasses.FrozenInstanceError):
        obj._is_universal_default = True  # type: ignore[misc]


@pytest.mark.parametrize("cls", _ALL_PARAMS)
def test_default_flag_roundtrip(cls):
    assert cls.default().is_default() is True
    assert cls.default(force_universal=True).is_default() is False
    assert cls().is_default() is False  # a hand-built config is not a factory default


@pytest.mark.parametrize("cls", _ALL_PARAMS)
def test_for_provider_unlisted_returns_base(cls):
    assert cls.for_provider("skillcorner") == cls()


def test_window_defaults():
    p = CornerWindowParams()
    assert (p.init_lead_s, p.target_contact_lead_s, p.target_cap_s, p.outcome_window_s) == (2.0, 1.0, 2.0, 6.0)


def test_sentinel_field_excluded_from_equality():
    # _is_universal_default must not affect == (compare=False)
    assert CornerWindowParams() == CornerWindowParams.default()
