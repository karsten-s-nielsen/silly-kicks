"""Shared per-period keeper-appearance builder (TF-59 PR1, spec §5.5, Part A).

``build_keeper_appearances_from_segments`` decomposes provider-agnostic keeper on-pitch SEGMENTS
(``KeeperSegment``) into the per-``(game, period_id, team)`` port rows every extractor emits. This
is the ONE decomposition; the StatsBomb / Sportec / GS extractors all feed it.
"""

from __future__ import annotations

import math

import numpy as np

from silly_kicks.keeper_identity import (
    KEEPER_APPEARANCE_COLUMNS,
    KeeperSegment,
    build_keeper_appearances_from_segments,
    validate_keeper_appearances,
)


def test_full_match_starter_decomposes_to_two_open_period_rows() -> None:
    # (period 1, 0.0) -> (period 2, +inf) over [1, 2] -> exactly two rows, both open to the period end.
    seg = KeeperSegment(
        team_id="T",
        player_id="P",
        source="starting_xi",
        start_period=1,
        start_time=0.0,
        end_period=2,
        end_time=math.inf,
    )
    ap = build_keeper_appearances_from_segments([seg], [1, 2], game_id="g1")
    assert list(ap.columns) == list(KEEPER_APPEARANCE_COLUMNS)
    assert list(ap["period_id"]) == [1, 2]
    assert list(ap["start_time_seconds"]) == [0.0, 0.0]
    assert bool(np.isinf(ap["end_time_seconds"]).all())


def test_starter_subbed_mid_period_two() -> None:
    # (1, 0.0) -> (2, 300.0): period-1 row open to +inf, period-2 row ends at 300.
    seg = KeeperSegment(
        team_id="T",
        player_id="P",
        source="starting_xi",
        start_period=1,
        start_time=0.0,
        end_period=2,
        end_time=300.0,
    )
    ap = build_keeper_appearances_from_segments([seg], [1, 2], game_id="g1")
    p1 = ap[ap["period_id"] == 1].iloc[0]
    p2 = ap[ap["period_id"] == 2].iloc[0]
    assert p1["start_time_seconds"] == 0.0 and np.isinf(p1["end_time_seconds"])
    assert p2["start_time_seconds"] == 0.0 and p2["end_time_seconds"] == 300.0


def test_replacement_single_period_row() -> None:
    # (2, 300.0) -> (2, +inf): a single period-2 row starting at 300.
    seg = KeeperSegment(
        team_id="T",
        player_id="R",
        source="sub_events",
        start_period=2,
        start_time=300.0,
        end_period=2,
        end_time=math.inf,
    )
    ap = build_keeper_appearances_from_segments([seg], [1, 2], game_id="g1")
    assert list(ap["period_id"]) == [2]
    assert ap.iloc[0]["start_time_seconds"] == 300.0
    assert np.isinf(ap.iloc[0]["end_time_seconds"])


def test_period_two_starter_emits_no_period_one_row() -> None:
    # (2, 0.0) -> (2, +inf) over [1, 2]: only a period-2 row; the p1 slice is out of the segment's span.
    seg = KeeperSegment(
        team_id="T",
        player_id="P",
        source="starting_xi",
        start_period=2,
        start_time=0.0,
        end_period=2,
        end_time=math.inf,
    )
    ap = build_keeper_appearances_from_segments([seg], [1, 2], game_id="g1")
    assert list(ap["period_id"]) == [2]


def test_string_ids_survive() -> None:
    seg = KeeperSegment(
        team_id="DFL-CLU-0",
        player_id="DFL-OBJ-1",
        source="starting_xi",
        start_period=1,
        start_time=0.0,
        end_period=1,
        end_time=math.inf,
    )
    ap = build_keeper_appearances_from_segments([seg], [1], game_id="g1")
    assert list(ap["player_id"]) == ["DFL-OBJ-1"]
    assert list(ap["team_id"]) == ["DFL-CLU-0"]
    validate_keeper_appearances(ap)  # a validated port, idempotent


def test_empty_segments_yield_empty_validated_port() -> None:
    ap = build_keeper_appearances_from_segments([], [1, 2], game_id="g1")
    assert list(ap.columns) == list(KEEPER_APPEARANCE_COLUMNS)
    assert len(ap) == 0
