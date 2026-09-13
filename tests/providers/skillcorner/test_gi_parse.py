import pandas as pd

from silly_kicks.providers.skillcorner import parse_passing_options


def _gi():
    # possession p1: 3 options (one targeted); a player_possession row that must be ignored
    return pd.DataFrame(
        [
            dict(
                event_type="player_possession",
                event_id="p1",
                player_id=99,
                associated_player_possession_event_id=None,
                player_in_possession_id=None,
                team_id=7,
                x_start=None,
                y_start=None,
                targeted=None,
                xpass_completion=None,
                n_opponents_bypassed=None,
                period=1,
                match_id=555,
            ),
            dict(
                event_type="passing_option",
                event_id="o1",
                associated_player_possession_event_id="p1",
                player_in_possession_id=99,
                team_id=7,
                player_id=11,
                x_start=-20.0,
                y_start=5.0,
                targeted=False,
                xpass_completion=0.98,
                n_opponents_bypassed=1.0,
                period=1,
                match_id=555,
            ),
            dict(
                event_type="passing_option",
                event_id="o2",
                associated_player_possession_event_id="p1",
                player_in_possession_id=99,
                team_id=7,
                player_id=12,
                x_start=-6.0,
                y_start=-21.0,
                targeted=True,
                xpass_completion=0.66,
                n_opponents_bypassed=8.0,
                period=1,
                match_id=555,
            ),
            dict(
                event_type="passing_option",
                event_id="o3",
                associated_player_possession_event_id="p1",
                player_in_possession_id=99,
                team_id=7,
                player_id=13,
                x_start=-40.0,
                y_start=-12.0,
                targeted=False,
                xpass_completion=0.99,
                n_opponents_bypassed=1.0,
                period=1,
                match_id=555,
            ),
        ]
    )


def test_parse_passing_options_schema_and_semantics():
    out = parse_passing_options(_gi(), game_id="555")
    assert list(out.columns) == [
        "game_id",
        "period_id",
        "decision_id",
        "possessor_id",
        "team_id",
        "target_player_id",
        "target_x",
        "target_y",
        "is_chosen",
        "completion",
        "opponents_bypassed",
    ]
    assert len(out) == 3  # only passing_option rows
    assert out["decision_id"].nunique() == 1
    assert out["is_chosen"].sum() == 1  # exactly one targeted
    assert out["is_chosen"].dtype == bool
    chosen = out[out["is_chosen"]].iloc[0]
    assert chosen["completion"] == 0.66 and chosen["opponents_bypassed"] == 8.0


def test_parse_passing_options_empty_yields_declared_columns():
    empty = pd.DataFrame({"event_type": ["player_possession"]})
    out = parse_passing_options(empty, game_id="555")
    assert list(out.columns) == [
        "game_id",
        "period_id",
        "decision_id",
        "possessor_id",
        "team_id",
        "target_player_id",
        "target_x",
        "target_y",
        "is_chosen",
        "completion",
        "opponents_bypassed",
    ]
    assert len(out) == 0
