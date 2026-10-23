"""The GI direction check on REAL SkillCorner opendata through the DEFAULT tracking converter (R.2 2026-10-23).

R.2 found every SkillCorner match with exactly one GI-direction-mismatched period: the frames are oriented
left-to-right (``convert_to_frames`` default ``"ltr"``) while GI ``attacking_side`` is in the delivered frame, and
the check compared the two without the recorded flip. These tests drive match 1886347 (MIT opendata; committed
slim bronze = every 100th frame, ``scripts/regen_tf46_fixtures.py``) through BOTH conventions with its GI actions.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

import pandas as pd
import pytest

from silly_kicks.spadl import skillcorner as sk_spadl
from silly_kicks.tracking import resolve_defended_goals
from silly_kicks.tracking import skillcorner as sk_tracking
from silly_kicks.tracking._restart_frame import GiDirectionMismatchWarning, gi_direction_check, resolve_frame_flips

_OPEN = Path(__file__).resolve().parents[1] / "datasets" / "skillcorner" / "opendata"
_MID = "1886347"


@lru_cache(maxsize=1)
def _actions() -> pd.DataFrame:
    meta = json.loads((_OPEN / f"{_MID}_match.json").read_text(encoding="utf-8"))
    events = pd.read_csv(_OPEN / f"{_MID}_events_slim.csv", low_memory=False)
    return sk_spadl.convert_to_actions(events, meta)[0]


def _frames(convention: Literal["ltr", "absolute_frame"]):
    meta = json.loads((_OPEN / f"{_MID}_match.json").read_text(encoding="utf-8"))
    bronze = pd.read_parquet(_OPEN / f"{_MID}_tracking_bronze_slim.parquet")
    home = str(meta["home_team"]["id"])
    return sk_tracking.convert_to_frames(bronze, home_team_id=home, output_convention=convention)


def test_fixture_precondition_gi_sides_and_both_periods_present():
    # the gate below is only meaningful if GI sides exist for both teams in both regular periods
    acts = _actions()
    backed = acts[acts["gi_attacking_side"].notna()]
    assert set(backed["period_id"].astype(int)) >= {1, 2}
    assert backed.groupby(["period_id", "team_id"]).size().min() >= 5


@pytest.mark.parametrize("convention", ["ltr", "absolute_frame"])
def test_real_match_has_no_mismatched_period_in_either_convention(convention, recwarn):
    frames, report = _frames(convention)
    # the documented route (C2 delta C2-IMPL-28): the report's flips through frame_flips_from_report
    flips = sk_tracking.frame_flips_from_report(report, output_convention=convention)
    assert flips == resolve_frame_flips(frames)  # the attrs stamp records the same applied flip
    chk = gi_direction_check(_actions(), resolve_defended_goals(frames), flips=flips)
    assert {int(p) for _g, p in chk.checked_periods} >= {1, 2}  # non-vacuity: both periods compared
    assert chk.mismatched_periods == ()
    assert chk.unresolved_periods == ()
    assert not [w for w in recwarn if issubclass(w.category, GiDirectionMismatchWarning)]


def test_the_raw_period_keyed_report_dict_is_refused_with_the_route_named():
    frames, report = _frames("ltr")
    with pytest.raises(TypeError, match="frame_flips_from_report"):
        gi_direction_check(_actions(), resolve_defended_goals(frames), flips=report.ltr_flip_by_period)  # type: ignore[arg-type]


def test_real_match_ltr_frames_compared_raw_mismatch_exactly_one_period():
    # the R.2 defect, reproduced: oriented frames + the GI side taken as delivered (no flip) -> one period off
    frames, report = _frames("ltr")
    assert any(report.ltr_flip_by_period.values())  # non-vacuity: this match has a flipped period
    raw = {(_MID, p): False for p in report.ltr_flip_by_period}
    with pytest.warns(GiDirectionMismatchWarning):
        chk = gi_direction_check(_actions(), resolve_defended_goals(frames), flips=raw)
    assert len(chk.mismatched_periods) == 1
