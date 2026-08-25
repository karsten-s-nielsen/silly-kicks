"""Structural perf guard (ADR-068): the DFL position XML is parsed in a SINGLE pass -- one
`ET.iterparse` over the file, not two. Byte-identity is the golden's job (test_parse_port_parity)."""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from silly_kicks.providers.sportec import parse_dfl_match_info, parse_dfl_tracking
from tests._perf_structural import call_counter

_FIX = Path(__file__).resolve().parents[2] / "datasets" / "sportec" / "idsse_slice"
_MATCH_ID = "J03WMX"


@pytest.fixture(scope="module")
def match_info():
    return parse_dfl_match_info(str(_FIX / "info.xml"))


def test_positions_parsed_in_a_single_pass(monkeypatch, match_info):
    calls = call_counter(monkeypatch, ET, "iterparse")
    parse_dfl_tracking(str(_FIX / "positions.xml"), match_info=match_info, match_id=_MATCH_ID)
    assert calls["n"] == 1  # was 2 (ball pass + player pass); the single-pass reads the file once
