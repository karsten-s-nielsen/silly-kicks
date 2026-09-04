"""update_glicko -- validated against Glickman's published Glicko-2 worked example + inactivity growth."""

from __future__ import annotations

import math

import pytest

from silly_kicks.duels import DuelRatingParams, GlickoState, update_glicko


def test_glickman_worked_example():
    # Glickman, "Example of the Glicko-2 system": player (1500, 200, 0.06) vs opponents (1400,30),
    # (1550,100), (1700,300) with results 1/0/0 -> new (1464.06, 151.52, 0.05999).
    ratings = {
        "P": GlickoState(1500.0, 200.0, 0.06),
        "A": GlickoState(1400.0, 30.0, 0.06),
        "B": GlickoState(1550.0, 100.0, 0.06),
        "C": GlickoState(1700.0, 300.0, 0.06),
    }
    games = [("P", "A", 1.0), ("P", "B", 0.0), ("P", "C", 0.0)]
    new = update_glicko(ratings, games)
    p = new["P"]
    assert p.rating == pytest.approx(1464.06, abs=0.02)
    assert p.rd == pytest.approx(151.52, abs=0.02)
    assert p.volatility == pytest.approx(0.05999, abs=1e-5)


def test_inactive_player_rd_grows():
    ratings = {"P": GlickoState(1500.0, 200.0, 0.06)}
    p = update_glicko(ratings, [])["P"]  # contested no duel this period
    assert p.rating == 1500.0  # unchanged
    assert p.volatility == 0.06  # unchanged
    phi = 200.0 / 173.7178
    assert p.rd == pytest.approx(math.sqrt(phi * phi + 0.06 * 0.06) * 173.7178)  # RD grew by sigma
    assert p.rd > 200.0


def test_inactivity_growth_can_be_disabled():
    ratings = {"P": GlickoState(1500.0, 200.0, 0.06)}
    params = DuelRatingParams(apply_inactivity_rd_growth=False)
    p = update_glicko(ratings, [], params=params)["P"]
    assert (p.rating, p.rd, p.volatility) == (1500.0, 200.0, 0.06)  # fully unchanged


def test_unseen_player_seeded_from_params():
    # A player appearing only in a game (not in `ratings`) is seeded from params.initial_state().
    new = update_glicko({}, [("X", "Y", 1.0)])
    assert set(new) == {"X", "Y"}
    # winner X should end above the seed 1500, loser Y below.
    assert new["X"].rating > 1500.0 > new["Y"].rating
