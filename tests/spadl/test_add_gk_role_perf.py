"""Structural perf guard (ADR-068): add_gk_role hoists the k-invariant ADR-019 ids_isin
content-probe out of the distribution-lookback loop, so it runs ONCE regardless of
distribution_lookback_actions (K), not once per k.

The hoist is byte-identical, so the PARITY half is the (unchanged) correctness suite --
test_add_gk_role.py (+ test_atomic_add_gk_role.py); this file adds only the STRUCTURAL half."""

import silly_kicks.id_compat as _idc
from silly_kicks.spadl.utils import add_gk_role
from tests._perf_structural import call_counter
from tests.spadl._gk_test_fixtures import _df, _make_gk_action, _make_pass_action


def _gk_then_pass():
    # GK save by player 999, then a pass by the same player -> distribution detection fires.
    return _df(
        [
            _make_gk_action(action_id=0, keeper_action="keeper_save", player_id=999, team_id=100),
            _make_pass_action(action_id=1, player_id=999, team_id=100),
        ]
    )


def test_ids_isin_hoisted_runs_once_for_k_gt_1(monkeypatch):
    # ids_isin is a function-local import from id_compat, so spy it at the source.
    calls = call_counter(monkeypatch, _idc, "ids_isin")
    out = add_gk_role(_gk_then_pass(), distribution_lookback_actions=3, goalkeeper_ids={999})
    assert calls["n"] == 1  # once total; pre-ADR-068 the probe ran once per k (== 3)
    assert "gk_role" in out.columns
