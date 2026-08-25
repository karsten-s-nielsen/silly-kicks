"""Structural perf guard for derive_goalkeepers (ADR-068): the per-(game,team) read and the GK
write each build their group lookup ONCE, not once per team / per GK."""

import pandas as pd

import silly_kicks.tracking._gk_identification as _gk
from silly_kicks.tracking._gk_identification import derive_goalkeepers
from tests._perf_structural import call_counter


def _multi_team_frames(n_frames: int = 20) -> pd.DataFrame:
    rows = []
    for game_id in ("g1", "g2"):
        for team_id in ("tA", "tB"):
            players = [
                {"player_id": f"{team_id}_gk", "x": 5.0, "y": 34.0},  # GK: in PA, near goal
                {"player_id": f"{team_id}_o1", "x": 50.0, "y": 34.0},
                {"player_id": f"{team_id}_o2", "x": 55.0, "y": 30.0},
            ]
            for _fid in range(n_frames):
                for p in players:
                    rows.append(
                        {
                            "game_id": game_id,
                            "team_id": team_id,
                            "player_id": p["player_id"],
                            "x": p["x"],
                            "y": p["y"],
                            "is_ball": False,
                            "is_goalkeeper": False,
                        }
                    )
    return pd.DataFrame(rows)


def test_group_lookup_built_bounded_not_per_team(monkeypatch):
    calls = call_counter(monkeypatch, _gk, "group_rows")
    n_frames = 20
    frames = _multi_team_frames(n_frames)
    frames_out, picks = derive_goalkeepers(frames)
    # 4 (game, team) groups in the fixture. group_rows is built EXACTLY twice regardless:
    # once for the team read, once for the GK write. Pre-ADR-068 the whole table was boolean-scanned
    # per team (read) AND per GK (write) -> the cost scaled with n_teams; here it does not.
    assert calls["n"] == 2
    # sanity: the fixture really has >2 (game, team) groups a per-team scan would have hit
    assert frames[["game_id", "team_id"]].drop_duplicates().shape[0] == 4
    # and the GK write actually fired (byte-identical behaviour preserved): one GK per (game, team),
    # each present in every frame -> 4 teams * n_frames rows flagged.
    assert frames_out["is_goalkeeper"].sum() == 4 * n_frames
    assert len(picks) == 4
