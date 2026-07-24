import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking import compute_bravery


def _final_actions(rows):
    base: dict[str, object] = dict(
        game_id="g1",
        period_id=1,
        bodypart_id=spadlconfig.bodypart_id["foot"],
        start_x=90.0,
        start_y=34.0,
        end_x=100.0,
        end_y=40.0,
    )
    # A benign team-20 pass so BOTH teams are present in the match (the two-team opponent
    # inference needs team 20 to appear somewhere; it is not a shot/cross so it never counts).
    rows = [*list(rows), dict(type_name="pass", result_name="success", team_id=20, player_id=99)]
    out = []
    for i, r in enumerate(rows):
        d: dict[str, object] = dict(base)
        d.update(action_id=i, time_seconds=float(i), **r)
        d["type_id"] = spadlconfig.actiontype_id[str(d.pop("type_name"))]
        d["result_id"] = spadlconfig.result_id[str(d.pop("result_name", "fail"))]
        d.setdefault("shot_blocked", pd.NA)
        d.setdefault("cross_blocked", pd.NA)
        out.append(d)
    df = pd.DataFrame(out)
    df["shot_blocked"] = pd.array(df["shot_blocked"].tolist(), dtype="boolean")
    df["cross_blocked"] = pd.array(df["cross_blocked"].tolist(), dtype="boolean")
    return df


def test_bravery_worked_example_shots_only():
    # team 20 faces 5 shots by team 10, 4 blocked -> bravery_shots = 0.8
    rows = [dict(type_name="shot", team_id=10, player_id=1, shot_blocked=(i < 4)) for i in range(5)]
    out = compute_bravery(_final_actions(rows))
    row = out[out["team_id"] == 20].iloc[0]  # the DEFENDING team
    assert row["bravery_shots"] == pytest.approx(0.8)
    assert row["n_shots_faced"] == 5


def test_set_piece_crosses_are_exposed_not_dropped():
    rows = [
        dict(type_name="shot", team_id=10, player_id=1, shot_blocked=True),
        dict(type_name="cross", team_id=10, player_id=2, cross_blocked=True),  # open-play
        dict(type_name="corner_crossed", team_id=10, player_id=3, cross_blocked=pd.NA),  # set-piece
        dict(type_name="freekick_crossed", team_id=10, player_id=4, cross_blocked=pd.NA),
    ]
    out = compute_bravery(_final_actions(rows))
    row = out[out["team_id"] == 20].iloc[0]
    assert np.isnan(row["bravery_set_piece_crosses"])  # NaN, never 0
    assert row["n_set_piece_crosses_faced"] == 2  # the gap is exposed
    # headline is over the KNOWN domain (1 shot + 1 open-play cross, both blocked = 1.0),
    # UNCHANGED by the set-piece crosses:
    assert row["bravery_pct_known_domain"] == pytest.approx(1.0)


def test_all_na_cross_column_yields_nan_open_play_component():
    rows = [dict(type_name="cross", team_id=10, player_id=2, cross_blocked=pd.NA) for _ in range(3)]
    out = compute_bravery(_final_actions(rows))
    row = out[out["team_id"] == 20].iloc[0]
    assert np.isnan(row["bravery_open_play_crosses"])  # unknown -> NaN, not 0


def test_both_columns_na_yields_nan_headline_and_warns():
    rows = [dict(type_name="shot", team_id=10, player_id=1, shot_blocked=pd.NA)]
    with pytest.warns(UserWarning, match="bravery"):
        out = compute_bravery(_final_actions(rows))
    row = out[out["team_id"] == 20].iloc[0]
    assert np.isnan(row["bravery_pct_known_domain"])
