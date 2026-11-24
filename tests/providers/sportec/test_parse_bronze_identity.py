"""W8.5: the single-sourced parser (spec 2026-11-08 tracking-batch section 4.8; plan Task 14).

Its bronze was proven bit-identical to the pre-change parser against a frozen oracle and on the 7 idsse matches (W8
corpus pass); the oracle ended with ruling C5 (spec section 7). The existing goldens (``test_parse_port_parity.py``)
pin the bronze; these tests pin the warning behaviour and the restart rule."""

from __future__ import annotations

import logging
import re
import warnings
from pathlib import Path

import pytest

from silly_kicks.providers.sportec import parse_dfl_events, parse_dfl_match_info

_FIX = Path(__file__).resolve().parents[2] / "datasets" / "sportec" / "idsse_slice"
_MATCH_ID = "J03WMX"


def test_parse_dfl_events_emits_no_warning():
    # the pre-change parser emitted none on this fixture either (checked when the oracles ended)
    info = parse_dfl_match_info(str(_FIX / "info.xml"))
    with warnings.catch_warnings(record=True) as new:
        warnings.simplefilter("always")
        events = parse_dfl_events(str(_FIX / "events.xml"), match_info=info, match_id=_MATCH_ID)
    assert len(events) > 0  # non-vacuity
    assert [(w.category, str(w.message)) for w in new] == []


def test_parse_logs_no_restart_kickoff(tmp_path, caplog):
    """A section-less KickOff after a period start is a restart: the parser no longer logs it as an unrecognised
    GameSection (owner ruling 2026-11-12). It keeps its period and its period-relative time."""
    text = (_FIX / "events.xml").read_text(encoding="utf-8")
    kickoff = re.search(r"<Event [^>]*>\s*<KickOff.*?</Event>", text, re.S)
    assert kickoff is not None
    ko = kickoff.group(0)
    restart = ko.replace(' GameSection="firstHalf"', "").replace("15:30:12.230", "15:40:00.000")
    second = ko.replace('"firstHalf"', '"secondHalf"').replace("15:30:12.230", "16:35:00.000")
    restart, second = (x.replace('EventId="18226500000006"', f'EventId="{i}"') for i, x in ((1, restart), (2, second)))
    path = tmp_path / "events.xml"
    path.write_text(text.replace("</PutDataRequest>", f"{restart}\n{second}\n</PutDataRequest>"), encoding="utf-8")
    info = parse_dfl_match_info(str(_FIX / "info.xml"))
    with caplog.at_level(logging.WARNING):
        new = parse_dfl_events(str(path), match_info=info, match_id=_MATCH_ID)
    assert not [r for r in caplog.records if "GameSection" in r.getMessage()]
    assert len(new) == len(parse_dfl_events(str(_FIX / "events.xml"), match_info=info, match_id=_MATCH_ID)) + 2
    rows = new.assign(event_id=new["event_id"].astype(str)).set_index("event_id").loc[["1", "2"]]
    assert rows["event_type"].tolist() == ["KickOff", "KickOff"]
    assert rows["period"].tolist() == [1, 2]  # the restart stays in period 1; the secondHalf KickOff starts period 2
    assert rows["timestamp_seconds"].tolist() == pytest.approx([587.77, 0.0], abs=1e-6)  # 15:40:00.000 - 15:30:12.230
