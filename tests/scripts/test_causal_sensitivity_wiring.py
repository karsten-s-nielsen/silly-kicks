"""TF-71 gate 11: both causal harnesses emit a JSON-safe sensitivity block (no corpus; synthetic)."""

from __future__ import annotations

import json
from importlib import import_module

import numpy as np
import pytest


def _binary_fixture(seed=0, n=400):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    Z = (rng.uniform(size=n) < 1 / (1 + np.exp(-X[:, 0]))).astype(int)
    Y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.3 * Z + X[:, 0])))).astype(float)
    return Z, X, Y


@pytest.mark.parametrize("driver", ["validate_xcross_causal", "validate_xshot_causal"])
def test_sensitivity_block_is_json_safe(driver):
    Z, X, Y = _binary_fixture()
    blk = import_module(driver)._sensitivity_block(Z, X, Y, seed=0, gammas=(0.0, 0.5), n_boot=20)
    json.dumps(blk)  # must not raise (gamma_star inf -> "inf"; tuples -> arrays)
    assert {"ipw_att", "gamma_star", "msm_go", "agreement_gap", "caveat"} <= set(blk)
