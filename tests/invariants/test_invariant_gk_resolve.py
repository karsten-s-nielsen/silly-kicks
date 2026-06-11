"""Physical invariants for defending_gk_from_frames (TF-13)."""

from __future__ import annotations

import pandas as pd

from tests.tracking.conftest import _make_actions, _make_frames


class TestGkResolveInvariants:
    def test_resolved_player_is_goalkeeper(self):
        from silly_kicks.tracking._gk_resolve import defending_gk_from_frames

        frames = _make_frames()
        actions = _make_actions(team_id=1)
        result = defending_gk_from_frames(actions, frames)
        resolved_pid = result.iloc[0]
        if pd.notna(resolved_pid):
            gk_pids = set(frames[frames["is_goalkeeper"] == True]["player_id"].dropna())  # noqa: E712
            assert resolved_pid in gk_pids

    def test_resolved_player_is_opposing_team(self):
        from silly_kicks.tracking._gk_resolve import defending_gk_from_frames

        frames = _make_frames()
        actions = _make_actions(team_id=1)
        result = defending_gk_from_frames(actions, frames)
        resolved_pid = result.iloc[0]
        if pd.notna(resolved_pid):
            # The resolved GK should NOT be on the acting team
            acting_team = actions["team_id"].iloc[0]
            gk_team = frames.loc[frames["player_id"] == resolved_pid, "team_id"].iloc[0]  # type: ignore[union-attr]
            assert gk_team != acting_team

    def test_empty_actions_empty_result(self):
        from silly_kicks.tracking._gk_resolve import defending_gk_from_frames

        frames = _make_frames()
        actions = _make_actions().iloc[:0]  # empty
        result = defending_gk_from_frames(actions, frames)
        assert len(result) == 0

    def test_result_length_matches_actions(self):
        from silly_kicks.tracking._gk_resolve import defending_gk_from_frames

        frames = _make_frames()
        actions = _make_actions(team_id=1)
        result = defending_gk_from_frames(actions, frames)
        assert len(result) == len(actions)

    def test_result_index_matches_actions(self):
        from silly_kicks.tracking._gk_resolve import defending_gk_from_frames

        frames = _make_frames()
        actions = _make_actions(team_id=1)
        actions.index = [42]  # non-default index
        result = defending_gk_from_frames(actions, frames)
        assert result.index.tolist() == [42]
