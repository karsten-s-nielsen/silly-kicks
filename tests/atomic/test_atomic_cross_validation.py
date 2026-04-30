"""Cross-validation tests for standard ↔ atomic helper equivalence (1.5.0).

These tests verify that the atomic helpers produce results consistent with
their standard SPADL counterparts when applied to a stream and its atomic
projection. This catches algorithmic drift between the two implementations
— per the Q6 lock in the PR-S5 design, cross-validation is "the strongest
correctness anchor."

For each helper, two paths are compared:

  - Path A: ``standard_helper(spadl_actions)`` → annotation per SPADL row
  - Path B: ``atomic_helper(convert_to_atomic(spadl_actions))`` → annotation
    per atomic row (atomic stream has more rows due to synthetic atomic
    actions: receival/interception/out/offside/goal/owngoal/yellow_card/
    red_card/dribble).

We map SPADL rows to their atomic counterparts via ``original_event_id`` and
filter out atomic-only synthetic rows, then assert per-original-row
equivalence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.atomic.spadl import config as atomicspadlconfig
from silly_kicks.atomic.spadl import convert_to_atomic
from silly_kicks.atomic.spadl.utils import add_gk_role as atomic_add_gk_role
from silly_kicks.atomic.spadl.utils import add_possessions as atomic_add_possessions
from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl.schema import SPADL_COLUMNS
from silly_kicks.spadl.utils import add_gk_role as standard_add_gk_role
from silly_kicks.spadl.utils import add_possessions as standard_add_possessions

_ACT = spadlconfig.actiontype_id
_RES = spadlconfig.result_id
_BP = spadlconfig.bodypart_id

_ATOMIC_ONLY_TYPE_NAMES = (
    "receival",
    "interception",
    "out",
    "offside",
    "goal",
    "owngoal",
    "yellow_card",
    "red_card",
)


def _spadl_row(
    *,
    action_id: int,
    type_name: str = "pass",
    result_name: str = "success",
    team_id: int = 100,
    player_id: int = 200,
    time_seconds: float = 0.0,
    start_x: float = 50.0,
    start_y: float = 34.0,
    end_x: float = 60.0,
    end_y: float = 34.0,
    game_id: int = 1,
    period_id: int = 1,
    bodypart_name: str = "foot",
) -> dict[str, object]:
    return {
        "game_id": game_id,
        "original_event_id": str(action_id),
        "action_id": action_id,
        "period_id": period_id,
        "time_seconds": time_seconds,
        "team_id": team_id,
        "player_id": player_id,
        "start_x": start_x,
        "start_y": start_y,
        "end_x": end_x,
        "end_y": end_y,
        "type_id": _ACT[type_name],
        "result_id": _RES[result_name],
        "bodypart_id": _BP[bodypart_name],
    }


def _spadl_df(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col, dtype in SPADL_COLUMNS.items():
        df[col] = df[col].astype(np.dtype(dtype))
    return df


def _filter_to_originals(atomic_df: pd.DataFrame) -> pd.DataFrame:
    """Return only the atomic rows that came from a SPADL parent row.

    Excludes atomic-only synthetic types and synthetic dribbles (which have
    NaN ``original_event_id``).
    """
    atomic_only_ids = {atomicspadlconfig.actiontype_id[name] for name in _ATOMIC_ONLY_TYPE_NAMES}
    keep = (
        ~atomic_df["type_id"].isin(atomic_only_ids)
        & atomic_df["original_event_id"].notna()
        & (atomic_df["original_event_id"] != "")
    )
    return atomic_df[keep].copy()


class TestAddPossessionsCrossValidation:
    """Standard add_possessions on SPADL stream vs atomic add_possessions on
    convert_to_atomic(SPADL stream) — boundaries should agree on original rows."""

    def test_simple_alternating_team_sequence(self):
        spadl = _spadl_df(
            [
                _spadl_row(action_id=0, team_id=100, time_seconds=0.0, type_name="pass"),
                _spadl_row(action_id=1, team_id=200, time_seconds=1.0, type_name="pass"),
                _spadl_row(action_id=2, team_id=100, time_seconds=2.0, type_name="pass"),
            ]
        )
        # Path A.
        std = standard_add_possessions(spadl)
        # Path B.
        atomic_in = convert_to_atomic(spadl)
        atomic_out = atomic_add_possessions(atomic_in)
        # Filter atomic to original rows only and align by original_event_id.
        atomic_originals = _filter_to_originals(atomic_out)
        std_pid = dict(zip(std["original_event_id"], std["possession_id"], strict=True))
        atomic_pid = dict(zip(atomic_originals["original_event_id"], atomic_originals["possession_id"], strict=True))
        # Per-row possession_id should match.
        for ev_id, std_val in std_pid.items():
            assert atomic_pid[ev_id] == std_val, (
                f"event {ev_id}: std possession_id={std_val}, atomic possession_id={atomic_pid[ev_id]}"
            )

    def test_freekick_after_foul_carve_out_agrees(self):
        # Standard SPADL: pass(B) → foul(A) → freekick_short(B). Carve-out should fire.
        spadl = _spadl_df(
            [
                _spadl_row(action_id=0, team_id=200, time_seconds=0.0, type_name="pass"),
                _spadl_row(action_id=1, team_id=100, time_seconds=1.0, type_name="foul", result_name="fail"),
                _spadl_row(action_id=2, team_id=200, time_seconds=2.0, type_name="freekick_short"),
            ]
        )
        std = standard_add_possessions(spadl)
        atomic_in = convert_to_atomic(spadl)
        atomic_out = atomic_add_possessions(atomic_in)
        atomic_originals = _filter_to_originals(atomic_out)
        std_pid = dict(zip(std["original_event_id"], std["possession_id"], strict=True))
        atomic_pid = dict(zip(atomic_originals["original_event_id"], atomic_originals["possession_id"], strict=True))
        # Both pipelines should classify the freekick (event 2) as the same possession as
        # the foul (event 1) — carve-out fires identically on both sides.
        assert std_pid["1"] == std_pid["2"]
        assert atomic_pid["1"] == atomic_pid["2"]
        # And the absolute values should match.
        for ev_id in std_pid:
            assert atomic_pid[ev_id] == std_pid[ev_id]

    def test_long_gap_creates_new_possession_in_both(self):
        # Gap = 20s. ``convert_to_atomic`` interleaves a synthetic receival
        # at midpoint (t=10) and a dribble at 3/4 (t=15) between the two
        # passes. With max_gap_seconds=7.0 (silly-kicks 2.1.0 default), the
        # receival's 10s gap from the first pass triggers a boundary on
        # the atomic side; the originals at t=0 and t=20 land in distinct
        # possessions. Smaller gaps like 10s could leave intermediate
        # atomic rows below the threshold and break the cross-pipeline
        # parity invariant this test guards.
        spadl = _spadl_df(
            [
                _spadl_row(action_id=0, team_id=100, time_seconds=0.0, type_name="pass"),
                _spadl_row(action_id=1, team_id=100, time_seconds=20.0, type_name="pass"),
            ]
        )
        std = standard_add_possessions(spadl)
        atomic_in = convert_to_atomic(spadl)
        atomic_out = atomic_add_possessions(atomic_in)
        atomic_originals = _filter_to_originals(atomic_out)
        # Both should split.
        assert std["possession_id"].iloc[0] != std["possession_id"].iloc[1]
        atomic_pid_for = dict(
            zip(atomic_originals["original_event_id"], atomic_originals["possession_id"], strict=True)
        )
        assert atomic_pid_for["0"] != atomic_pid_for["1"]


class TestAddGkRoleCrossValidation:
    """Standard add_gk_role on SPADL stream vs atomic add_gk_role on the atomic
    projection — gk_role values should match on original rows."""

    @pytest.mark.parametrize(
        "keeper_action,expected_role",
        [
            ("keeper_save", "shot_stopping"),
            ("keeper_claim", "cross_collection"),
            ("keeper_punch", "cross_collection"),
            ("keeper_pick_up", "pick_up"),
        ],
    )
    def test_keeper_action_role_matches(self, keeper_action: str, expected_role: str):
        spadl = _spadl_df(
            [
                _spadl_row(
                    action_id=0,
                    type_name=keeper_action,
                    player_id=999,
                    start_x=5.0,
                    end_x=5.0,
                    bodypart_name="other",
                ),
            ]
        )
        std = standard_add_gk_role(spadl)
        atomic_in = convert_to_atomic(spadl)
        atomic_out = atomic_add_gk_role(atomic_in)
        atomic_originals = _filter_to_originals(atomic_out)
        # Both sides should agree.
        assert std["gk_role"].iloc[0] == expected_role
        assert atomic_originals["gk_role"].iloc[0] == expected_role

    def test_sweeping_outside_box_matches(self):
        spadl = _spadl_df(
            [
                _spadl_row(
                    action_id=0,
                    type_name="keeper_save",
                    player_id=999,
                    start_x=20.0,
                    end_x=20.0,
                    bodypart_name="other",
                ),
            ]
        )
        std = standard_add_gk_role(spadl)
        atomic_in = convert_to_atomic(spadl)
        atomic_out = atomic_add_gk_role(atomic_in)
        atomic_originals = _filter_to_originals(atomic_out)
        assert std["gk_role"].iloc[0] == "sweeping"
        assert atomic_originals["gk_role"].iloc[0] == "sweeping"

    def test_distribution_matches(self):
        spadl = _spadl_df(
            [
                _spadl_row(
                    action_id=0,
                    type_name="keeper_save",
                    player_id=999,
                    start_x=5.0,
                    end_x=5.0,
                    bodypart_name="other",
                ),
                _spadl_row(
                    action_id=1,
                    type_name="pass",
                    player_id=999,
                    time_seconds=1.0,
                    start_x=5.0,
                    end_x=15.0,
                ),
            ]
        )
        std = standard_add_gk_role(spadl)
        atomic_in = convert_to_atomic(spadl)
        atomic_out = atomic_add_gk_role(atomic_in)
        atomic_originals = _filter_to_originals(atomic_out)

        std_role_for = dict(zip(std["original_event_id"], std["gk_role"], strict=True))
        atomic_role_for = dict(zip(atomic_originals["original_event_id"], atomic_originals["gk_role"], strict=True))
        # Same role on both sides for both rows.
        assert std_role_for["0"] == atomic_role_for["0"] == "shot_stopping"
        assert std_role_for["1"] == atomic_role_for["1"] == "distribution"
