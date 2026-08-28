import warnings

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import FOV_REGIME_VALUES, FovDiagnosis, validate_fov

_PITCH = np.array([[0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0]])
_LEFT_HALF = np.array([[0.0, 0.0], [52.5, 0.0], [52.5, 68.0], [0.0, 68.0]])


def _va(rows):  # rows: list[(action_id, polygon|None)]
    return pd.DataFrame({"action_id": [r[0] for r in rows], "polygon": [r[1] for r in rows]})


def test_full_coverage_regime():
    d = validate_fov(_va([(1, _PITCH), (2, _PITCH)]))
    assert d.regime == "full_coverage"
    assert d.n_actions == 2
    assert all(f >= 0.98 for f in d.observed_pitch_fraction.values())


def test_cropped_regime():
    d = validate_fov(_va([(1, _LEFT_HALF), (2, _LEFT_HALF)]))
    assert d.regime == "fov_cropped"
    assert round(d.observed_pitch_fraction[1], 3) == 0.5


def test_absent_regime():
    d = validate_fov(_va([(1, None), (2, np.zeros((2, 2)))]))  # None + degenerate
    assert d.regime == "absent"


def test_mixed_raises_by_default():
    with pytest.raises(ValueError):
        validate_fov(_va([(1, _PITCH), (2, _LEFT_HALF)]))


def test_mixed_warns_under_warn():
    with pytest.warns(UserWarning):
        d = validate_fov(_va([(1, _PITCH), (2, _LEFT_HALF)]), on_mismatch="warn")
    assert d.regime == "mixed"


def test_mixed_ignore_is_silent():
    # on_mismatch="ignore" on a `mixed` set neither raises NOR warns -- it returns the diagnosis
    # silently with regime == "mixed" (the third branch, alongside raise/warn).
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        d = validate_fov(_va([(1, _PITCH), (2, _LEFT_HALF)]), on_mismatch="ignore")
    assert caught == []
    assert d.regime == "mixed"


def test_empty_never_raises():
    d = validate_fov(_va([]))
    assert d.regime == "empty" and d.n_actions == 0


def test_regime_in_vocabulary():
    assert validate_fov(_va([(1, _PITCH)])).regime in FOV_REGIME_VALUES


def test_returns_fov_diagnosis_instance():
    # Surface check: validate_fov returns the public frozen FovDiagnosis dataclass.
    assert isinstance(validate_fov(_va([(1, _PITCH)])), FovDiagnosis)
