"""The loader's kick mapping and its gap check (action-time repair spec §4.9 step 3, §4.4 ``restart_gaps``)."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts._loader_pining import _skillcorner_kicks
from silly_kicks.spadl import config as spadlconfig

_T = spadlconfig.actiontype_id


def _actions(corner_time: float, corner_source: str) -> pd.DataFrame:
    rows = [
        ("a", 0, 20.0, "pass", "native", "tracking"),
        ("b", 1, corner_time, "corner_crossed", "derived", corner_source),
        ("b", 2, 42.0, "pass", "native", "tracking"),
    ]
    return pd.DataFrame(
        [
            {"game_id": "g", "original_event_id": oid, "action_id": aid, "period_id": 1, "time_seconds": t,
             "type_id": _T[ty], "action_provenance": prov, "time_source": src}
            for oid, aid, t, ty, prov, src in rows
        ]
    )  # fmt: skip


def _frames() -> pd.DataFrame:
    t = [round(x * 0.1, 1) for x in range(0, 500)]
    return pd.DataFrame({"game_id": "g", "period_id": 1, "frame_id": range(len(t)), "time_seconds": t, "is_ball": True})


def test_a_kick_in_its_gap_maps_to_its_frame_and_side():
    pre = _actions(39.98, "placeholder")
    stepped = _actions(31.0, "kick")
    kicks, outside = _skillcorner_kicks(pre, stepped, _frames())
    assert kicks == {("b", "corner"): (310, "before")}
    assert outside == []


def test_the_check_fires_on_a_kick_after_the_reception():
    pre = _actions(39.98, "placeholder")
    stepped = _actions(41.0, "kick")  # inside b's possession, after b.frame_start (40 s): a defect
    kicks, outside = _skillcorner_kicks(pre, stepped, _frames())
    assert kicks == {}
    assert len(outside) == 1


def test_an_order_only_kick_has_the_after_side():
    pre = _actions(39.97, "order_only")
    stepped = _actions(31.0, "kick")
    kicks, _ = _skillcorner_kicks(pre, stepped, _frames())
    assert kicks == {("b", "corner"): (310, "after")}


@pytest.mark.parametrize("src", ["placeholder", "order_only"])
def test_no_kick_rows_map_nothing(src):
    pre = _actions(39.98, src)
    kicks, outside = _skillcorner_kicks(pre, pre, _frames())
    assert kicks == {} and outside == []


def test_a_period_end_penalty_kick_has_the_period_end_side():
    # RTR-IMPL A04(a): a penalty row with no non-derived row after it in its period (spec 4.17) has the gap
    # [a.frame_end, the period's last frame] and the side "period_end" (spec 4.9 step 3)
    def table(t, src):
        return pd.DataFrame(
            [
                {"game_id": "g", "original_event_id": "a", "action_id": 0, "period_id": 1, "time_seconds": 20.0,
                 "type_id": _T["pass"], "action_provenance": "native", "time_source": "tracking"},
                {"game_id": "g", "original_event_id": "a", "action_id": 1, "period_id": 1, "time_seconds": t,
                 "type_id": _T["shot_penalty"], "action_provenance": "derived", "time_source": src},
            ]
        )  # fmt: skip

    kicks, outside = _skillcorner_kicks(table(20.03, "order_only"), table(45.0, "kick"), _frames())
    assert kicks == {("a", "penalty"): (450, "period_end")}
    assert outside == []
