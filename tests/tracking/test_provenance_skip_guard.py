"""Verify that chaining multiple add_* enrichments does not produce
duplicate linkage-provenance columns (frame_id_x, frame_id_y, etc.).

PR-S37 (silly-kicks 3.11.2) added the skip guard to the 4 aggregators
that were missing it: add_action_context, add_pre_shot_gk_position,
add_actor_pre_window, add_pressure_on_actor.  This test exercises the
full chain to catch regressions.
"""

from __future__ import annotations

import pandas as pd
import pytest

from silly_kicks.spadl.utils import add_pre_shot_gk_context
from silly_kicks.tracking.features import (
    add_action_context,
    add_actor_pre_window,
    add_obso,
    add_player_influence,
    add_pre_shot_gk_position,
    add_pressure_on_actor,
    add_shape_graph,
    add_space_creation,
)
from tests.tracking._provider_inputs import load_provider_frames, synthesize_actions

_PROVENANCE_COLS = {"frame_id", "time_offset_seconds", "link_quality_score", "n_candidate_frames"}


@pytest.mark.parametrize("provider", ["sportec"])
def test_chained_enrichments_no_duplicate_provenance(provider: str) -> None:
    """Chain add_pre_shot_gk_context(frames) -> add_action_context ->
    add_actor_pre_window -> add_pressure_on_actor and verify provenance
    columns appear exactly once (no _x/_y suffixes)."""
    frames = load_provider_frames(provider)
    actions = synthesize_actions(frames, n_actions=5)

    # Step 1: add_pre_shot_gk_context WITH frames => internally merges provenance
    actions = add_pre_shot_gk_context(actions, frames=frames)
    assert _PROVENANCE_COLS.issubset(actions.columns), "Step 1 should add provenance"

    # Step 2: add_action_context => should SKIP provenance (already present)
    actions = add_action_context(actions, frames)
    _assert_no_suffix_duplicates(actions, "add_action_context")

    # Step 3: add_actor_pre_window => should SKIP provenance
    actions = add_actor_pre_window(actions, frames)
    _assert_no_suffix_duplicates(actions, "add_actor_pre_window")

    # Step 4: add_pressure_on_actor => should SKIP provenance
    actions = add_pressure_on_actor(actions, frames, methods=("andrienko_oval",))
    _assert_no_suffix_duplicates(actions, "add_pressure_on_actor")

    # Step 5: add_player_influence => should SKIP provenance
    xt = _make_xt()
    home = actions["team_id"].dropna().iloc[0]
    actions = add_player_influence(actions, frames, xt, home_team_id=home)
    _assert_no_suffix_duplicates(actions, "add_player_influence")

    # Step 6: add_shape_graph => should SKIP provenance
    actions = add_shape_graph(actions, frames, home_team_id=home)
    _assert_no_suffix_duplicates(actions, "add_shape_graph")

    # Step 7: add_obso => should SKIP provenance
    actions = add_obso(actions, frames, home_team_id=home)
    _assert_no_suffix_duplicates(actions, "add_obso")

    # Step 8: add_space_creation => should SKIP provenance
    actions = add_space_creation(actions, frames, home_team_id=home)
    _assert_no_suffix_duplicates(actions, "add_space_creation")

    # Final: provenance columns exist exactly once
    for col in _PROVENANCE_COLS:
        matches = [c for c in actions.columns if c == col or c.startswith(f"{col}_")]
        assert matches == [col], f"Expected exactly [{col}], got {matches}"


@pytest.mark.parametrize("provider", ["sportec"])
def test_first_enrichment_still_adds_provenance(provider: str) -> None:
    """When called on clean actions (no provenance yet), the aggregator
    should still add provenance columns — the skip guard must not suppress
    the first merge."""
    frames = load_provider_frames(provider)
    actions = synthesize_actions(frames, n_actions=5)

    # No provenance yet
    assert not _PROVENANCE_COLS.intersection(actions.columns)

    actions = add_action_context(actions, frames)
    assert _PROVENANCE_COLS.issubset(actions.columns), "First enrichment must add provenance"


@pytest.mark.parametrize("provider", ["sportec"])
def test_pre_shot_gk_position_skips_provenance_when_present(provider: str) -> None:
    """add_pre_shot_gk_position was the most common collision source
    because add_pre_shot_gk_context(frames=...) calls it internally."""
    frames = load_provider_frames(provider)
    actions = synthesize_actions(frames, n_actions=5)

    # Establish provenance via add_action_context
    actions = add_pre_shot_gk_context(actions)
    actions = add_action_context(actions, frames)
    assert _PROVENANCE_COLS.issubset(actions.columns)

    # add_pre_shot_gk_position should skip provenance
    actions = add_pre_shot_gk_position(actions, frames)
    _assert_no_suffix_duplicates(actions, "add_pre_shot_gk_position")


def _make_xt():
    import numpy as np

    from silly_kicks.xthreat import ExpectedThreat

    xt = ExpectedThreat(l=16, w=12)
    xt.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))
    return xt


def _assert_no_suffix_duplicates(df: pd.DataFrame, step_name: str) -> None:
    """Assert no _x/_y suffixed provenance columns exist."""
    for base in _PROVENANCE_COLS:
        for suffix in ("_x", "_y"):
            bad = f"{base}{suffix}"
            assert bad not in df.columns, f"After {step_name}: found '{bad}' — provenance skip guard failed"
