"""SK-XT-SER — ExpectedThreat serialize seam (to_dict/from_dict/save/load).

Red-green TDD for the JSON, pickle-free, fail-closed, raw-orientation-verbatim serialize
seam (spec ``docs/superpowers/specs/2026-09-20-expectedthreat-serialize-seam-design.md``).
The fitted model must cross a process boundary without re-``fit()`` (the lakehouse ExT-v2 fold).
"""

from __future__ import annotations

import json
from typing import cast

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from silly_kicks.xthreat import ExpectedThreat, KDEParams, Method, destination_profiles

_METHODS = ("singh_counts", "kde_smoothed")


def _fit(method: str, actions) -> ExpectedThreat:
    """Fit an ExpectedThreat and assert it is NON-DEGENERATE (PLAN-02).

    A degenerate (all-zero / single-nonzero) ``xT`` makes the round-trip guard vacuous, so
    every serialize test rests on this precondition rather than assuming it.
    """
    params = KDEParams() if method == "kde_smoothed" else None
    xt = ExpectedThreat(method=cast(Method, method), params=params).fit(actions)
    assert np.count_nonzero(xt.xT) > 1, f"{method}: fitted xT is degenerate on this fixture -- test would be vacuous"
    return xt


# --------------------------------------------------------------------------- #
# 1. Round-trip fidelity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", _METHODS)
def test_round_trip_fidelity(method: str, spadl_actions) -> None:
    xt = _fit(method, spadl_actions)
    xt2 = ExpectedThreat.from_dict(xt.to_dict())

    for attr in ("xT", "scoring_prob_matrix", "shot_prob_matrix", "move_prob_matrix", "transition_matrix"):
        a, b = getattr(xt, attr), getattr(xt2, attr)
        assert np.array_equal(a, b), f"{attr} not bit-identical across round-trip"
        assert a.dtype == b.dtype, f"{attr} dtype changed: {a.dtype} -> {b.dtype}"
        assert a.shape == b.shape, f"{attr} shape changed: {a.shape} -> {b.shape}"

    assert len(xt2.heatmaps) == len(xt.heatmaps)
    for h1, h2 in zip(xt.heatmaps, xt2.heatmaps, strict=True):
        assert np.array_equal(h1, h2)
        assert h1.shape == h2.shape

    assert (xt2.l, xt2.w, xt2.eps, xt2.method) == (xt.l, xt.w, xt.eps, xt.method)
    assert xt2.params == xt.params
    assert xt2.grid == xt.grid


# --------------------------------------------------------------------------- #
# 2. Functional equivalence (catches a silent orientation flip)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", _METHODS)
def test_functional_equivalence(method: str, spadl_actions) -> None:
    xt = _fit(method, spadl_actions)
    xt2 = ExpectedThreat.from_dict(xt.to_dict())

    assert np.allclose(xt.rate(spadl_actions), xt2.rate(spadl_actions), equal_nan=True)

    xs = spadl_actions["start_x"].to_numpy()
    ys = spadl_actions["start_y"].to_numpy()
    p1 = destination_profiles(xt, xs, ys)
    p2 = destination_profiles(xt2, xs, ys)
    assert np.allclose(p1.zone_centres, p2.zone_centres, equal_nan=True)
    assert np.allclose(p1.zone_values, p2.zone_values, equal_nan=True)
    assert np.allclose(p1.probabilities, p2.probabilities, equal_nan=True)


# --------------------------------------------------------------------------- #
# 3. Unfitted to_dict raises
# --------------------------------------------------------------------------- #
def test_unfitted_to_dict_raises() -> None:
    with pytest.raises(NotFittedError):
        ExpectedThreat().to_dict()


# --------------------------------------------------------------------------- #
# 5. JSON boundary (no non-JSON type leaks)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", _METHODS)
def test_json_boundary(method: str, spadl_actions) -> None:
    xt = _fit(method, spadl_actions)
    round_tripped = ExpectedThreat.from_dict(json.loads(json.dumps(xt.to_dict())))
    assert np.array_equal(xt.xT, round_tripped.xT)
    assert np.allclose(xt.rate(spadl_actions), round_tripped.rate(spadl_actions), equal_nan=True)


# --------------------------------------------------------------------------- #
# 6. Non-vacuity / fail-closed
# --------------------------------------------------------------------------- #
def test_from_dict_missing_array_raises(spadl_actions) -> None:
    d = _fit("singh_counts", spadl_actions).to_dict()
    # non-vacuity: the unmutated dict reconstructs
    assert ExpectedThreat.from_dict(dict(d)) is not None
    d.pop("transition_matrix")
    with pytest.raises((KeyError, ValueError)):
        ExpectedThreat.from_dict(d)


def test_from_dict_unknown_format_version_raises(spadl_actions) -> None:
    d = _fit("singh_counts", spadl_actions).to_dict()
    d["format_version"] = 2
    with pytest.raises(ValueError):
        ExpectedThreat.from_dict(d)
    d2 = _fit("singh_counts", spadl_actions).to_dict()
    del d2["format_version"]
    with pytest.raises(ValueError):
        ExpectedThreat.from_dict(d2)


def test_from_dict_all_zero_xt_raises(spadl_actions) -> None:
    d = _fit("singh_counts", spadl_actions).to_dict()
    d["xT"] = np.zeros_like(np.asarray(d["xT"])).tolist()
    with pytest.raises(NotFittedError):
        ExpectedThreat.from_dict(d)


# --------------------------------------------------------------------------- #
# 7. save / load file round-trip
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", _METHODS)
def test_save_load_file_round_trip(method: str, spadl_actions, tmp_path) -> None:
    xt = _fit(method, spadl_actions)
    path = tmp_path / "xt.json"
    xt.save(path)
    xt2 = ExpectedThreat.load(path)
    assert np.array_equal(xt.xT, xt2.xT)
    assert np.allclose(xt.rate(spadl_actions), xt2.rate(spadl_actions), equal_nan=True)


def test_load_bad_format_version_raises(spadl_actions, tmp_path) -> None:
    d = _fit("singh_counts", spadl_actions).to_dict()
    d["format_version"] = 99
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(ValueError):
        ExpectedThreat.load(path)
