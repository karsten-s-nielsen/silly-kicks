"""MatchOutcomeParams / MatchOutcomeReport (TF-53 Task 1)."""

from __future__ import annotations

import dataclasses

import pytest

from silly_kicks.match_outcome import MatchOutcomeParams, MatchOutcomeReport


def test_default_and_for_provider():
    assert MatchOutcomeParams.default().is_default() is True
    assert MatchOutcomeParams.default(force_universal=True).is_default() is False
    assert MatchOutcomeParams().is_default() is False
    assert MatchOutcomeParams.for_provider("statsbomb") == MatchOutcomeParams()
    assert MatchOutcomeParams.for_provider("wyscout") == MatchOutcomeParams()


def test_defaults_are_both_corrections():
    # ADR-097: default is both corrections ON (collapse on correctness, dixon_coles on measured evidence).
    p = MatchOutcomeParams()
    assert p.same_possession == "collapse"
    assert p.team_dependence == "dixon_coles"


def test_post_init_rejects_invalid_enums():
    with pytest.raises(ValueError, match="same_possession"):
        MatchOutcomeParams(same_possession="collapsed")  # type: ignore[arg-type]  # typo -> raise
    with pytest.raises(ValueError, match="team_dependence"):
        MatchOutcomeParams(team_dependence="dixoncoles")  # type: ignore[arg-type]  # typo -> raise


def test_frozen():
    p = MatchOutcomeParams()
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.same_possession = "collapse"  # type: ignore[misc]


def test_report_conservation_identities():
    r = MatchOutcomeReport(MatchOutcomeParams(), 10, 9, 1, 240, 235, 5, 3)
    assert r.n_matches_scored + r.n_matches_excluded_not_two_teams == r.n_matches_in
    assert r.n_shots_with_xg + r.n_shots_null_xg == r.n_shots
