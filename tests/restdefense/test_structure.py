"""Layer-1 structure metrics (TF-60, ADR-080) -- hand-computed values + orientation symmetry."""

import math

import pandas as pd

from silly_kicks.restdefense._config import RestDefenseParams
from silly_kicks.restdefense._structure import SampleContext, layer1_metrics, rd_num_superiority
from tests.restdefense._fixtures import make_rest_defense_fixture

_PARAMS = RestDefenseParams()


def _frame_slice(frames, fid):
    return frames[frames["frame_id"] == fid].reset_index(drop=True)


# Contexts built from the fixture + the engine values verified in test_compute / by hand. Option B:
# the 8th field is the REARGUARD lateral_width (back-4 y-spread), NOT whole-team team_width (which is
# 38/36 here -- distinguished by the fixture's wide forwards); the 9th is whole-team team_length.
#   a0 (frame 100, A=team1, G_A=0, ball=70): line 24, compactness 12, lat_width 28, length 47
#   a1 (frame 101, A=team1, G_A=0, ball=75): line 30.5, compactness 10, lat_width 24, length 44
#   a2 (frame 102, A=team2, G_A=105, ball=35): line 81, compactness 12, lat_width 28, length 47 (mirror of a0)
_CTX_A0 = SampleContext(1, 2, 70.0, 0.0, 105.0, 24.0, 12.0, 28.0, 47.0)
_CTX_A1 = SampleContext(1, 2, 75.0, 0.0, 105.0, 30.5, 10.0, 24.0, 44.0)
_CTX_A2 = SampleContext(2, 1, 35.0, 105.0, 0.0, 81.0, 12.0, 28.0, 47.0)


def _metrics(frames, fid, ctx):
    return layer1_metrics(_frame_slice(frames, fid), ctx, params=_PARAMS)


def test_a0_hand_computed_values():
    _, frames = make_rest_defense_fixture()
    m = _metrics(frames, 100, _CTX_A0)
    assert m["rd_num_superiority"] == 4  # A=5 outfield behind ball, B=1 -> 4
    assert m["rd_num_superiority_gk"] == 5  # + A's keeper in the band
    assert m["rd_zone_occupancy"] == 3  # team1 in [0, 24]: GK(5), 18, 22
    assert m["rd_line_height"] == 24.0
    assert m["rd_line_height_relative"] == -46.0  # 24 - 70
    assert m["rd_compactness_x"] == 12.0
    assert m["rd_width"] == 28.0
    assert m["rd_depth"] == 47.0
    assert m["rd_shape_2_3_vs_3_2"] == "4-1"  # {18,22,26,30 | 65}
    assert m["rd_gk_line_height"] == 5.0
    assert m["rd_gk_to_line_distance"] == -19.0  # 5 - 24


def test_a1_variant_values_differ_from_a0():
    _, frames = make_rest_defense_fixture()
    m = _metrics(frames, 101, _CTX_A1)
    assert m["rd_num_superiority"] == 2  # A=5, B=3 (three team2 players in [0,75]) -> 2
    assert m["rd_num_superiority_gk"] == 3
    assert m["rd_zone_occupancy"] == 3  # team1 in [0, 30.5]: GK(9), 26, 28
    assert m["rd_line_height"] == 30.5
    assert m["rd_line_height_relative"] == -44.5
    assert m["rd_compactness_x"] == 10.0
    assert m["rd_width"] == 24.0
    assert m["rd_depth"] == 44.0
    assert m["rd_gk_line_height"] == 9.0
    assert m["rd_gk_to_line_distance"] == -21.5


def test_orientation_symmetry_home_equals_away():
    """a2 (away, own goal x=105) is the exact point-reflection of a0 (home, own goal x=0), so every
    metric must be identical -- direction comes from the goal end, never team identity (ADR-055)."""
    _, frames = make_rest_defense_fixture()
    home = _metrics(frames, 100, _CTX_A0)
    away = _metrics(frames, 102, _CTX_A2)
    assert home == away
    # And the pinned A != B value is preserved under the mirror (symmetry alone would pass for an
    # all-zero / wrong-band impl; pinning 4 = 5 - 1 rules that out).
    assert away["rd_num_superiority"] == 4


def test_num_superiority_counts_both_teams_toward_a_goal():
    """Passing B's own goal instead of A's would invert the band -- pin that A's goal is used."""
    _, frames = make_rest_defense_fixture()
    fr = _frame_slice(frames, 100)
    toward_ga = rd_num_superiority(fr, _CTX_A0)  # both toward G_A = 0
    assert toward_ga == 4
    # a context that (wrongly) used B's goal would count a different band; guard the sign is right.
    assert toward_ga > 0  # A has the rearguard superiority behind its own goal


def test_missing_keeper_yields_nan_gk_metrics():
    """No observed A keeper (FOV crop) -> honest-NaN GK metrics, never a fabricated 0 (ADR-063)."""
    _, frames = make_rest_defense_fixture()
    fr = _frame_slice(frames, 100)
    fr = fr[~((fr["team_id"] == 1) & fr["is_goalkeeper"])].reset_index(drop=True)  # drop team1 GK
    m = layer1_metrics(fr, _CTX_A0, params=_PARAMS)
    assert math.isnan(m["rd_gk_line_height"])
    assert math.isnan(m["rd_gk_to_line_distance"])
    # non-GK metrics still resolve
    assert m["rd_num_superiority"] == 4


def test_num_superiority_na_when_opponent_unresolvable():
    """IMPL-04: a non-two-team frame set (opponent_id NA) -> rd_num_superiority is pd.NA, never a
    silent A-count (the B-count would fabricate a 0). Opponent-free metrics still resolve."""
    _, frames = make_rest_defense_fixture()
    ctx = SampleContext(1, pd.NA, 70.0, 0.0, 105.0, 24.0, 12.0, 28.0, 47.0)  # opponent_id = NA
    m = layer1_metrics(_frame_slice(frames, 100), ctx, params=_PARAMS)
    assert m["rd_num_superiority"] is pd.NA
    assert m["rd_num_superiority_gk"] is pd.NA
    assert m["rd_zone_occupancy"] == 3  # opponent-free -> still computed


def test_zone_occupancy_na_when_line_unresolved():
    _, frames = make_rest_defense_fixture()
    ctx = SampleContext(1, 2, 70.0, 0.0, 105.0, float("nan"), float("nan"), float("nan"), float("nan"))
    m = layer1_metrics(_frame_slice(frames, 100), ctx, params=_PARAMS)
    assert m["rd_zone_occupancy"] is pd.NA
