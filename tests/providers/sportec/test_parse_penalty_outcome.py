"""R63 / bar A25: the parser reads the outcome tag under ``Penalty > ShotAtGoal`` with the rule it applies one level
up for a ``ShotAtGoal`` event (``shot_outcome_type`` + the outcome tag's attributes as ``shot_outcome_*``)."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from silly_kicks.providers.sportec.parse import _build_event_row


def _row(xml: str) -> dict[str, object]:
    elem = ET.fromstring(xml)  # noqa: S314  # nosemgrep: use-defused-xml-parse -- an inline test literal
    first_child = next(iter(elem))
    return _build_event_row(elem, first_child, first_child.tag, "M", 1, {"P1": "home"}, {})


_PENALTY = (
    '<Event EventId="7" EventTime="2026-01-01T15:20:00.000+01:00">'
    '<Penalty Team="T1" CausingPlayer="P9">'
    '<ShotAtGoal Player="P1" TypeOfShot="rightLeg">{outcome}</ShotAtGoal>'
    "</Penalty></Event>"
)


@pytest.mark.parametrize(
    ("outcome", "value"),
    [('<SuccessfulShot GoalZone="center"/>', "successful"), ('<SavedShot GoalKeeper="P5"/>', "saved")],
    ids=["scored", "saved"],
)
def test_penalty_nested_shot_outcome_is_read(outcome, value):
    row = _row(_PENALTY.format(outcome=outcome))
    assert row["shot_outcome_type"] == value
    assert row["shot_player"] == "P1"  # the nested shot's attributes, as today


def test_penalty_nested_outcome_attributes_take_the_shot_outcome_prefix():
    row = _row(_PENALTY.format(outcome='<SuccessfulShot GoalZone="center"/>'))
    assert row["shot_outcome_goal_zone"] == "center"


def test_penalty_without_a_nested_shot_has_no_outcome():
    row = _row('<Event EventId="8" EventTime="2026-01-01T15:20:00.000+01:00"><Penalty Team="T1"/></Event>')
    assert row.get("shot_outcome_type") in (None, "")


def test_shot_at_goal_event_is_unchanged():
    xml = (
        '<Event EventId="9" EventTime="2026-01-01T15:20:00.000+01:00">'
        '<ShotAtGoal Player="P1"><SavedShot GoalKeeper="P5"/></ShotAtGoal></Event>'
    )
    row = _row(xml)
    assert row["shot_outcome_type"] == "saved" and row["shot_outcome_goal_keeper"] == "P5"
