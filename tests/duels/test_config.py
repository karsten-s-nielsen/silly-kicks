"""DuelRatingParams + GlickoState -- Glicko-2 defaults, flag semantics, empty per-provider override."""

from __future__ import annotations

from silly_kicks.duels import DuelRatingParams, GlickoState


def test_default_and_flag():
    assert DuelRatingParams.default().is_default() is True
    assert DuelRatingParams.default(force_universal=True).is_default() is False
    assert DuelRatingParams().is_default() is False


def test_for_provider_returns_base_for_unlisted():
    assert DuelRatingParams.for_provider("sportec") == DuelRatingParams()
    assert DuelRatingParams.for_provider("statsbomb") == DuelRatingParams()


def test_glicko2_defaults():
    p = DuelRatingParams()
    assert (p.initial_rating, p.initial_rd, p.initial_volatility, p.tau) == (1500.0, 350.0, 0.06, 0.5)
    assert p.apply_inactivity_rd_growth is True


def test_initial_state():
    s = DuelRatingParams().initial_state()
    assert isinstance(s, GlickoState)
    assert (s.rating, s.rd, s.volatility) == (1500.0, 350.0, 0.06)
