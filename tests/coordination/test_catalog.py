"""TF-58 Task 16: coordination pair catalogue."""

from __future__ import annotations

import pytest

from silly_kicks.coordination._catalog import DEFAULT_PAIRS, METHODS_BY_LEVEL, PairSpec, resolve_pairs


def test_default_catalog_matches_spec_table():
    by_level: dict[str, list[PairSpec]] = {}
    for p in DEFAULT_PAIRS:
        by_level.setdefault(p.level, []).append(p)
    assert len(by_level["team_team"]) == 9
    assert len(by_level["cross_variable"]) == 3
    assert len(by_level["intra_team"]) == 1
    assert len(by_level["dyad"]) == 4
    # exact L1 canonical set (same signal vs same signal)
    l1 = {p.signal_a for p in by_level["team_team"]}
    assert l1 == {
        "centroid_x",
        "centroid_y",
        "stretch_x",
        "stretch_y",
        "stretch_index",
        "spread",
        "convex_hull_area",
        "team_length",
        "team_width",
    }
    assert all(p.signal_a == p.signal_b and p.role == "canonical" for p in by_level["team_team"])
    cross = {(p.signal_a, p.signal_b) for p in by_level["cross_variable"]}
    assert cross == {("centroid_x", "defensive_line_x"), ("team_length", "compactness_x"), ("stretch_x", "stretch_x")}
    assert by_level["intra_team"][0] == PairSpec("intra_team", "defensive_line_x", "centroid_x", "same_team")
    assert {(p.signal_a, p.role) for p in by_level["dyad"]} == {
        ("player_x", "same_team"),
        ("player_x", "canonical"),
        ("player_y", "same_team"),
        ("player_y", "canonical"),
    }


def test_pairspec_validation_each_rule_both_sides():
    PairSpec("team_team", "centroid_x", "centroid_x", "canonical")  # valid
    with pytest.raises(ValueError, match="level must be"):
        PairSpec("bogus", "centroid_x", "centroid_x", "canonical")
    with pytest.raises(ValueError, match="not an analysable"):
        PairSpec("team_team", "possession", "possession", "canonical")
    with pytest.raises(ValueError, match=r"role .* not allowed"):
        PairSpec("cross_variable", "centroid_x", "defensive_line_x", "canonical")
    with pytest.raises(ValueError, match="dyad pairs must be"):
        PairSpec("dyad", "player_x", "player_y", "same_team")
    with pytest.raises(ValueError, match="team-scope"):
        PairSpec("team_team", "player_x", "player_x", "canonical")  # player scope at a team level


def test_axis_and_commensurate_properties():
    hull_vs_spread = PairSpec("team_team", "convex_hull_area", "spread", "canonical")
    assert hull_vs_spread.commensurate is False  # m^2 vs metres
    assert hull_vs_spread.axis == "scalar"
    centroid_mixed = PairSpec("team_team", "centroid_x", "centroid_y", "canonical")
    assert centroid_mixed.axis == "mixed"
    assert PairSpec("team_team", "centroid_x", "centroid_x", "canonical").commensurate is True


def test_resolve_pairs_filters_levels():
    dyads = resolve_pairs(None, ["dyad"])
    assert dyads and all(p.level == "dyad" for p in dyads)
    team = resolve_pairs(None, ["team_team"])
    assert len(team) == 9
    custom = [PairSpec("team_team", "spread", "spread", "canonical")]
    assert resolve_pairs(custom, ["team_team"]) == tuple(custom)
    assert resolve_pairs(custom, ["dyad"]) == ()


def test_methods_by_level():
    assert METHODS_BY_LEVEL["dyad"] == ("relative_phase", "cross_correlation")
    assert "coherence" in METHODS_BY_LEVEL["team_team"]
    assert "vector_coding" in METHODS_BY_LEVEL["cross_variable"]
