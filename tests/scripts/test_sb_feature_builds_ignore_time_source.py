"""The StatsBomb-only artifacts of ``tf46_refit.toml`` see the new ``time_source`` column and no other change.

Action-time repair spec section 6: the ``sb_open`` and ``sb360`` jobs (xSuccess, pass completion, win probability,
match-outcome dependence, the territory census, the receiver model) get ``time_source == "event"`` on every row and
nothing else; their feature builds select columns by name, so each builds the same features with and without the
column. Checked here on the committed WorldCup 2018 SPADL fixture (StatsBomb) and the receiver test's tracking scene.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def sb_game(sb_worldcup_data):
    games = sb_worldcup_data["games"]
    gid = int(games["game_id"].iloc[0])
    actions = sb_worldcup_data[f"actions/game_{gid}"]
    actions = actions.drop(columns=[c for c in ("time_source",) if c in actions.columns]).reset_index(drop=True)
    home = games.loc[games["game_id"] == gid, "home_team_id"].iloc[0]
    assert len(actions) > 100  # non-vacuity
    return gid, home, actions, actions.assign(time_source="event")


def test_xsuccess_rows(sb_game):
    from scripts.train_xsuccess import onball_rows_for_match

    gid, _home, plain, with_ts = sb_game
    pd.testing.assert_frame_equal(onball_rows_for_match(with_ts, gid), onball_rows_for_match(plain, gid))


def test_pass_completion_rows(sb_game):
    from scripts.train_pass_completion import pass_rows_for_match

    gid, _home, plain, with_ts = sb_game
    pd.testing.assert_frame_equal(pass_rows_for_match(with_ts, gid), pass_rows_for_match(plain, gid))


def test_win_probability_shard(sb_game):
    from scripts.train_win_probability import match_shard

    _gid, home, plain, with_ts = sb_game
    pd.testing.assert_frame_equal(match_shard(with_ts, home, "2018-06-14"), match_shard(plain, home, "2018-06-14"))


def test_match_outcome_slice(sb_game):
    from scripts.train_match_outcome_dependence import extract_match_slice

    _gid, _home, plain, with_ts = sb_game
    rng = np.random.default_rng(0)
    xg = rng.uniform(0.0, 0.5, len(plain))
    pd.testing.assert_frame_equal(extract_match_slice(with_ts.assign(xg=xg)), extract_match_slice(plain.assign(xg=xg)))


def test_territory_counterfactual(sb_game):
    from silly_kicks.territory import CounterfactualParams, TerritoryParams, compute_territorial_dominance
    from silly_kicks.xthreat import ExpectedThreat

    _gid, _home, plain, with_ts = sb_game
    xt = ExpectedThreat().fit(plain)

    class _ConstCompletion:  # duck-typed PassCompletionModel port (ADR-022), as the glossary invariant uses
        def predict_completion(self, ox, oy, tx, ty):
            return np.full(np.asarray(tx, dtype=float).shape, 0.6)

    def run(a):
        samples, _ = compute_territorial_dominance(
            a, xt=xt, method="counterfactual", params=TerritoryParams(trim_fraction=1.0),
            completion_model=_ConstCompletion(), cf_params=CounterfactualParams.default(),  # type: ignore[arg-type]
        )  # fmt: skip
        return samples

    want = run(plain)
    assert len(want) > 0  # non-vacuity
    pd.testing.assert_frame_equal(run(with_ts), want)


@pytest.mark.parametrize("strategy", ["trajectory", "id"])
def test_receiver_candidate_rows(strategy):
    from scripts.train_receiver_model import extract_candidate_rows
    from tests.scripts.test_train_receiver_model import _one_pass_match, _traj_match

    a, fr = _traj_match(receiver_on_ray=True) if strategy == "trajectory" else _one_pass_match(receiver_in_frame=True)
    want = extract_candidate_rows(a, fr, feature_set="public", labeling_strategy=strategy)
    assert len(want) > 0  # non-vacuity
    got = extract_candidate_rows(a.assign(time_source="event"), fr, feature_set="public", labeling_strategy=strategy)
    pd.testing.assert_frame_equal(got, want)
