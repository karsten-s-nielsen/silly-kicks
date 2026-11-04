"""Task 2.5 — SkillCornerConversionReport subclass + restart/goals census (spec section 6.3 / C4)."""

import dataclasses
import json
from pathlib import Path

import pandas as pd

from silly_kicks.spadl import skillcorner
from silly_kicks.spadl.schema import ConversionReport, SkillCornerConversionReport

_OPEN = Path(__file__).parents[1] / "datasets" / "skillcorner" / "opendata"


def _convert(mid: str):
    events = pd.read_csv(_OPEN / f"{mid}_events_slim.csv", low_memory=False)
    meta = json.load(open(_OPEN / f"{mid}_match.json", encoding="utf-8"))
    return skillcorner.convert_to_actions(events, meta)


def test_report_is_conversion_report_subclass():
    _, rep = _convert("2016236")  # own-goal kick-off match
    assert isinstance(rep, SkillCornerConversionReport)
    assert isinstance(rep, ConversionReport)  # the existing isinstance contract still holds


def test_report_carries_restart_and_goals_census():
    _, rep = _convert("2016236")
    assert isinstance(rep.restart_counts, dict)
    assert isinstance(rep.restart_evidence_counts, dict)
    assert rep.goals_repaired  # own_goal_inverted + normals
    assert rep.goals_coverage_gap is False
    assert rep.goals_ambiguous == 0
    assert rep.goals_unresolved == 0


def test_report_asdict_round_trips():
    _, rep = _convert("2016236")
    d = dataclasses.asdict(rep)
    assert d["provider"] == "skillcorner"
    assert "restart_counts" in d
    assert "restart_evidence_counts" in d
    assert "goals_repaired" in d
    assert "repeated_restart_count" in d
