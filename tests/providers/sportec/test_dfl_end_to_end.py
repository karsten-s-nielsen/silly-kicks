"""W8.4: a DFL consumer that builds its own bronze to the public column contract gets the parser's periods,
timestamps and actors through ``complete_dfl_event_bronze`` (spec 2026-11-08 tracking-batch section 4.8; plan Task 14).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from scripts._bitwise import assert_bits_equal
from silly_kicks.providers.sportec import (
    complete_dfl_event_bronze,
    dfl_actor_columns,
    dfl_event_column,
    parse_dfl_events,
    parse_dfl_match_info,
)
from silly_kicks.providers.sportec._dfl_tables import _PLAYER_ATTR_ORDER

_FIX = Path(__file__).resolve().parents[2] / "datasets" / "sportec" / "idsse_slice"
_MATCH_ID = "J03WMX"


def _consumer_bronze(path: Path) -> pd.DataFrame:
    """A consumer's own DFL bronze: one row per Event with a first child, the raw ``event_time``, and the contract's
    actor columns (``dfl_actor_columns``), read from the XML independently of the parser."""
    rows = []
    for ev in ET.parse(path).getroot().iter("Event"):  # noqa: S314 - the committed test slice
        children = list(ev)
        if not children:
            continue
        child = children[0]
        row: dict = {"event_time": ev.get("EventTime"), "event_type": child.tag, "kickoff_game_section": None}
        if child.tag == "KickOff":
            plays = [c for c in child if c.tag == "Play"]
            row["kickoff_game_section"] = child.get("GameSection")
            row["kickoff_play_present"] = bool(plays)
            row["kickoff_first_play_player"] = plays[0].get("Player") if plays else None
            row["kickoff_player"] = child.get("Player")
        else:
            for attr, col in zip(
                _PLAYER_ATTR_ORDER.get(child.tag, ["Player"]), dfl_actor_columns(child.tag), strict=True
            ):
                row[col] = child.get(attr)
                assert col == dfl_event_column(child.tag, attr)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["kickoff_play_present"] = out["kickoff_play_present"].astype("boolean")
    return out


def _values(column: pd.Series) -> list:
    """The column's values with every missing value as None (the two paths hold object vs string dtypes)."""
    return [None if pd.isna(v) else v for v in column.tolist()]


def test_end_to_end_consumer_bronze_equals_parser_path():
    info = parse_dfl_match_info(str(_FIX / "info.xml"))
    parsed = parse_dfl_events(str(_FIX / "events.xml"), match_info=info, match_id=_MATCH_ID)
    consumer, report = complete_dfl_event_bronze(_consumer_bronze(_FIX / "events.xml"), info)
    kept = consumer[consumer["period"].notna()].reset_index(drop=True)  # the parser's drop (the docstring's line)
    assert len(kept) == len(parsed) > 0 and report.kickoff_times
    for col in ("player_id", "team", "team_id_native", "home_team_id_native", "away_team_id_native"):
        assert _values(kept[col]) == _values(parsed[col]), col
    assert kept["period"].astype("int64").tolist() == parsed["period"].astype("int64").tolist()
    assert_bits_equal(
        kept["timestamp_seconds"].to_numpy(dtype="float64"), parsed["timestamp_seconds"].to_numpy(dtype="float64")
    )
