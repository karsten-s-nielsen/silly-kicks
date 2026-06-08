from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

import silly_kicks.xthreat as xt
from silly_kicks.vaep import features as fs
from silly_kicks.vaep.features.expected_threat import xt_xfns
from tests._xthreat_helpers import _corpus_with_shots, _moves


@pytest.fixture(scope="module")
def fitted_xt() -> xt.ExpectedThreat:
    """A small fitted singh_counts xT model (fast, deterministic).

    Fit on a corpus WITH shots so the value-iteration grid is non-zero (a passes-only
    corpus yields an all-zero ``.xT`` -> NotFittedError on rate)."""
    return xt.ExpectedThreat().fit(_corpus_with_shots(n_per_zone=40, seed=0))


def test_factory_returns_single_transformer(fitted_xt: xt.ExpectedThreat) -> None:
    transformers = xt_xfns(model=fitted_xt)
    assert isinstance(transformers, list)
    assert len(transformers) == 1


def test_column_names_track_method(fitted_xt: xt.ExpectedThreat) -> None:
    cols = fs.feature_column_names(xt_xfns(model=fitted_xt), nb_prev_actions=3)
    assert cols == ["xt__singh_counts_a0", "xt__singh_counts_a1", "xt__singh_counts_a2"]


def test_values_equal_model_rate(fitted_xt: xt.ExpectedThreat) -> None:
    actions = _moves(n_per_zone=10, seed=1)
    states = fs.gamestates(actions, 1)
    out = xt_xfns(model=fitted_xt)[0](states)
    expected = fitted_xt.rate(states[0])
    np.testing.assert_array_equal(out["xt__singh_counts_a0"].to_numpy(), expected)


def test_fail_closed_none() -> None:
    with pytest.raises(ValueError, match="fitted ExpectedThreat"):
        xt_xfns(model=None)


def test_fail_closed_str() -> None:
    with pytest.raises(NotImplementedError, match="bundled"):
        xt_xfns(model="default")


def test_fail_closed_unfitted() -> None:
    with pytest.raises(NotFittedError):
        xt_xfns(model=xt.ExpectedThreat())


def test_exported_from_features_package() -> None:
    from silly_kicks.vaep import features as fs_pkg

    assert hasattr(fs_pkg, "xt_xfns")
    assert "xt_xfns" in fs_pkg.__all__


def test_exported_from_vaep_package() -> None:
    import silly_kicks.vaep as v

    assert hasattr(v, "xt_xfns")
    assert "xt_xfns" in v.__all__


def test_not_in_any_default_list(fitted_xt: xt.ExpectedThreat) -> None:
    """Opt-in invariant: the produced transformer is in no default/union xfn list."""
    from silly_kicks.vaep import base, hybrid

    produced = xt_xfns(model=fitted_xt)[0]
    for lst in (
        base.xfns_default,
        base.xfns_default_no_goalscore,
        hybrid.hybrid_xfns_default,
        hybrid.hybrid_xfns_default_no_goalscore,
    ):
        assert produced not in lst
        # also: no xfn already in the defaults emits an xt__ column
        names = [getattr(f, "__name__", "") for f in lst]
        assert not any(n.startswith("xt__") for n in names)


def test_vaep_integration_adds_column(fitted_xt: xt.ExpectedThreat) -> None:
    """compute_features with xt_xfns appended produces the xt__ columns, dtype float."""
    from silly_kicks.vaep import VAEP
    from silly_kicks.vaep.base import xfns_default

    actions = _moves(n_per_zone=15, seed=2)
    game = pd.Series({"game_id": 1, "home_team_id": 1})
    v = VAEP(xfns=xfns_default + xt_xfns(model=fitted_xt), nb_prev_actions=3)
    X = v.compute_features(game, actions)
    for c in ("xt__singh_counts_a0", "xt__singh_counts_a1", "xt__singh_counts_a2"):
        assert c in X.columns
        assert X[c].dtype == np.float64


@pytest.mark.filterwarnings("ignore")
def test_e2e_worldcup_finite_for_moves(sb_worldcup_data) -> None:
    """On the committed WC2018 fixture: finite xT for successful moves, NaN for shots."""
    import silly_kicks.spadl.config as cfg
    from tests._xthreat_helpers import _worldcup_ltr

    ltr = _worldcup_ltr(sb_worldcup_data)
    model = xt.ExpectedThreat().fit(ltr)
    one_game = ltr[ltr.game_id == ltr.game_id.iloc[0]].copy()
    out = xt_xfns(model=model)[0](fs.gamestates(one_game, 1))
    col = out["xt__singh_counts_a0"]
    is_succ_move = one_game.type_id.isin([cfg.actiontype_id[t] for t in ("pass", "dribble", "cross")]) & (
        one_game.result_id == cfg.result_id["success"]
    )
    assert np.isfinite(col[is_succ_move.to_numpy()]).any()
    is_shot = (one_game.type_id == cfg.actiontype_id["shot"]).to_numpy()
    assert col[is_shot].isna().all()
