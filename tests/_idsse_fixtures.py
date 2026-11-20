"""The single reader of the committed idsse fixtures.

``tests/datasets/idsse/per_period_match.parquet`` and ``sample_match.parquet`` are BRONZE (the parser's output). The
documented path to SPADL is ``parse -> shape_events_to_native -> convert_to_actions``; the native shaping resolves the
team and the player of the set-piece and foul rows from the DFL qualifiers. A test that gave the bronze straight to
the converter saw those rows with team ``"unknown"`` (and turned a home row as an away row). Every test reads the
fixtures here, and ``tests/test_idsse_fixture_path.py`` pins that no other test file builds a path to them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

IDSSE_FIXTURE_DIR = Path(__file__).resolve().parent / "datasets" / "idsse"
IdsseFixture = Literal["sample_match", "per_period_match"]


def idsse_fixture_path(name: IdsseFixture) -> Path:
    """The committed fixture file (for a test that must check that it exists)."""
    return IDSSE_FIXTURE_DIR / f"{name}.parquet"


def idsse_fixture_native(name: IdsseFixture) -> pd.DataFrame:
    """The fixture through ``shape_events_to_native``: the converter's documented input."""
    from silly_kicks.providers.sportec import shape_events_to_native

    return shape_events_to_native(pd.read_parquet(idsse_fixture_path(name)))


def idsse_goalkeeper_ids(native: pd.DataFrame) -> set[str] | None:
    """The keeper ids of a native fixture frame: the players of the rows with a goalkeeper action."""
    if "play_goal_keeper_action" not in native.columns:
        return None
    rows = native.loc[native["play_goal_keeper_action"].notna(), "player_id"]
    return set(rows.dropna().astype(str).tolist())
