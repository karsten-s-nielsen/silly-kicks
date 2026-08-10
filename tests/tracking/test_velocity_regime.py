"""``validate_velocity_regime`` -- the third member of the ADR-017 / ADR-019 diagnostic family."""

from __future__ import annotations

import pandas as pd
import pytest

import silly_kicks.tracking as T
from tests.sb360._fixture import build_leg_a, build_leg_b


def test_positional_only_regime():
    _a, frames, _l = build_leg_a()
    d = T.validate_velocity_regime(frames, on_mismatch="ignore")
    assert d.regime == "positional_only"
    assert d.has_velocity_columns is False


def test_velocity_informed_regime():
    _a, frames, _l = build_leg_b()
    d = T.validate_velocity_regime(frames, on_mismatch="ignore")
    assert d.regime == "velocity_informed"


def test_mixed_regime_is_the_one_fail_loud_exists_for():
    _a, frames, _l = build_leg_a()
    frames = frames.copy()
    frames.loc[frames.index[: len(frames) // 2], "speed_source"] = None
    d = T.validate_velocity_regime(frames, on_mismatch="ignore")
    assert d.regime == "mixed"
    with pytest.raises(ValueError, match="mixed"):
        T.validate_velocity_regime(frames, on_mismatch="raise")


def test_forgot_derive_velocities_is_NOT_labelled_mixed():
    """Distinct from MIXED, and the case a user is most likely to hit. Labelling it 'mixed' would
    raise with 'some rows can carry velocity and others structurally cannot', which is false here."""
    _a, frames, _l = build_leg_b()
    frames = frames.drop(columns=["vx", "vy"]).copy()
    frames["speed_source"] = "native"
    d = T.validate_velocity_regime(frames, on_mismatch="ignore")
    assert d.regime == "velocity_missing"
    assert "derive_velocities()" in d.message


def test_frames_without_a_speed_source_column_do_not_CRASH():
    """Measured on the first draft: frames.get('speed_source') returns None, None == marker is a
    Python bool, and False.sum() raises AttributeError. A row-count guard does not prevent it."""
    frames = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    d = T.validate_velocity_regime(frames, on_mismatch="ignore")
    assert d.regime == "velocity_missing"
    assert d.has_velocity_columns is False


def test_an_empty_frame_set_does_not_raise():
    """Follows the siblings: measured on a schema-shaped zero-row frame, validate_time_base AND
    validate_id_dtypes both return a diagnosis rather than raising."""
    empty = pd.DataFrame({c: pd.Series(dtype="object") for c in ("speed_source", "vx", "vy")})
    d = T.validate_velocity_regime(empty)  # default on_mismatch="raise"
    assert d.regime == "empty"


def test_the_regime_vocabulary_is_exported_and_closed():
    """A regime string that can RAISE by default is a consumer-facing contract."""
    assert set(T.VELOCITY_REGIME_VALUES) == {
        "velocity_informed",
        "positional_only",
        "mixed",
        "velocity_missing",
        "empty",
    }
