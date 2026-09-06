"""compute_territorial_dominance is PURE -- never mutates the caller's actions (ADR-033).

Three variants (per the ADR-033 contract for a conditional path): window=None (per-game atoms), a
window set (pooled aggregation), and method="counterfactual" (TF-54b -- a THIRD, materially different
code path through ``_counterfactual_dispatch`` / ``counterfactual_rows``, injecting a completion model).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.territory import CounterfactualParams, TerritoryParams, compute_territorial_dominance
from silly_kicks.xthreat import ExpectedThreat, GridSpec

_SUCCESS = spadlconfig.result_id["success"]
_TACKLE = spadlconfig.actiontype_id["tackle"]
_PASS = spadlconfig.actiontype_id["pass"]
_KEEP_ALL = TerritoryParams(trim_fraction=1.0)


class _ConstCompletion:
    """Trivial injected completion model (mirrors tests/territory/test_counterfactual_compute.py)."""

    def predict_completion(self, ox, oy, tx, ty):
        return np.full(np.asarray(tx, dtype=float).shape, 0.6)


def _toy_xt(value: float = 0.1) -> ExpectedThreat:
    xt = ExpectedThreat()
    xt.xT = np.full(np.asarray(xt.xT).shape, value, dtype=float)
    return xt


class _UniformXt:
    """Duck-typed fitted xT carrying a real ``transition_matrix`` (mirrors
    tests/territory/test_counterfactual_compute.py's ``_ToyUniformXt``). ``method="counterfactual"``'s
    failed-pass leg calls ``xthreat.destination_profiles``, which indexes ``model.transition_matrix`` --
    a plain ``ExpectedThreat()`` (as built by ``_toy_xt`` above) carries no such matrix, so the
    counterfactual purity test below needs this rather than the completed_failed-only ``_toy_xt``."""

    def __init__(self, value: float = 0.1, nx: int = 8, ny: int = 6) -> None:
        self.l, self.w = nx, ny
        self.grid = GridSpec(n_zones_x=nx, n_zones_y=ny)
        n = nx * ny
        self.xT = np.full((ny, nx), float(value))
        self.transition_matrix = np.full((n, n), 1.0 / n)
        self.method = "singh_counts"


def _scene():
    rows = [
        {
            "game_id": 1,
            "period_id": 1,
            "team_id": 10,
            "player_id": 1,
            "type_id": _TACKLE,
            "result_id": _SUCCESS,
            "start_x": x,
            "start_y": y,
            "end_x": x,
            "end_y": y,
            "time_seconds": 10.0,
        }
        for x, y in [(5, 20), (15, 20), (15, 48), (5, 48)]
    ] + [
        {
            "game_id": 1,
            "period_id": 1,
            "team_id": 20,
            "player_id": 99,
            "type_id": _PASS,
            "result_id": _SUCCESS,
            "start_x": 80,
            "start_y": 40,
            "end_x": 95,
            "end_y": 40,
            "time_seconds": 20.0,
        }
    ]
    df = pd.DataFrame(rows)
    df["action_id"] = range(len(df))
    return df


@pytest.mark.parametrize("window", [None, [1]])
def test_compute_does_not_mutate_actions(window):
    actions = _scene()
    snapshot = actions.copy(deep=True)
    out, _ = compute_territorial_dominance(actions, xt=_toy_xt(0.1), window=window, params=_KEEP_ALL)
    pd.testing.assert_frame_equal(actions, snapshot)  # input untouched
    assert out is not actions


def _scene_with_failed_pass():
    """The v1 purity scene plus a FAILED opponent pass, so the counterfactual variant below exercises
    both the completed leg (valued at the observed end) and the failed leg (modeled over the
    death-direction cone) -- the two branches ``counterfactual_rows`` mutates/derives arrays for."""
    df = _scene()
    extra = pd.DataFrame(
        [
            {
                "game_id": 1,
                "period_id": 1,
                "team_id": 20,
                "player_id": 99,
                "type_id": _PASS,
                "result_id": spadlconfig.result_id["fail"],
                "start_x": 80,
                "start_y": 40,
                "end_x": 90,
                "end_y": 40,
                "time_seconds": 25.0,
            }
        ]
    )
    out = pd.concat([df, extra], ignore_index=True)
    out["action_id"] = range(len(out))
    return out


def test_compute_counterfactual_does_not_mutate_actions():
    """method="counterfactual" (TF-54b) is a materially different code path (``_counterfactual_dispatch``
    -> ``counterfactual_rows``, injecting a completion model) -- ADR-033 requires its own purity variant,
    not just inheritance from the default-method test above."""
    actions = _scene_with_failed_pass()
    snapshot = actions.copy(deep=True)
    out, _ = compute_territorial_dominance(
        actions,
        xt=_UniformXt(0.1),  # type: ignore[arg-type]  -- duck-typed (ADR-022), see class docstring
        method="counterfactual",
        completion_model=_ConstCompletion(),  # type: ignore[arg-type]
        params=_KEEP_ALL,
        cf_params=CounterfactualParams.default(),
    )
    pd.testing.assert_frame_equal(actions, snapshot)  # input untouched
    assert out is not actions
