from __future__ import annotations

from scripts._sb_roster import build_gk_roster_map


def test_build_gk_roster_map_filters_goalkeepers_keyed_by_team():
    roster = {
        901: {"name": "A", "jersey": 1, "team": 10, "position": "Goalkeeper"},
        102: {"name": "B", "jersey": 9, "team": 10, "position": "Center Forward"},
        902: {"name": "C", "jersey": 1, "team": 20, "position": "Goalkeeper"},
    }
    assert build_gk_roster_map(roster) == {10: 901, 20: 902}
