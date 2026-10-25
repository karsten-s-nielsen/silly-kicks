"""The SkillCorner PROVIDER period range (TF-46 Arm 1 Part 2 spec sections 6.1 / 17.7, amended 2026-10-25; DL ruling
Q4): the raw feed's per-period frame range (empty frames included, period-``None`` frames excluded), recorded by the
adapter, re-keyed per ``(game_id, period_id)`` and read by the GI pairing Range test. Real data: opendata 1886347
(the committed slim bronze + ``match.json`` ``match_periods``)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from silly_kicks.spadl import skillcorner as sk_spadl
from silly_kicks.tracking import resolve_defended_goals
from silly_kicks.tracking import skillcorner as sk_tracking
from silly_kicks.tracking._restart_frame import assess_gi_pairing, gi_direction_check, resolve_period_frame_ranges
from silly_kicks.tracking.schema import PERIOD_FRAME_RANGES_ATTR

_OPEN = Path(__file__).resolve().parents[1] / "datasets" / "skillcorner" / "opendata"
_MID = "1886347"


def _meta() -> dict:
    return json.loads((_OPEN / f"{_MID}_match.json").read_text(encoding="utf-8"))


def _bronze() -> pd.DataFrame:
    return pd.read_parquet(_OPEN / f"{_MID}_tracking_bronze_slim.parquet")


def _declared() -> dict[int, tuple[int, int]]:
    """The provider range of 1886347 from its match_periods (start exact; the end is the declared end frame)."""
    return {int(m["period"]): (int(m["start_frame"]), int(m["end_frame"])) for m in _meta()["match_periods"]}


def test_provider_range_counts_empty_labelled_frames_and_skips_period_none():
    # the 2007448 shape: P2's first labelled frames (30..39) carry no objects; they still belong to the range
    records = [
        {"frame": 0, "period": None, "player_data": []},
        {"frame": 10, "period": 1, "player_data": [{"player_id": 1}]},
        {"frame": 25, "period": 1, "player_data": []},
        {"frame": 27, "period": None, "player_data": []},
        {"frame": 30, "period": 2, "player_data": []},
        {"frame": 40, "period": 2, "player_data": [{"player_id": 1}]},
        {"frame": None, "period": 2},
    ]
    assert sk_tracking.provider_period_frame_ranges(records) == {1: (10, 25), 2: (30, 40)}
    as_frame = pd.DataFrame([{"frame": r["frame"], "period": r["period"]} for r in records])
    assert sk_tracking.provider_period_frame_ranges(as_frame) == {1: (10, 25), 2: (30, 40)}


def test_convert_to_frames_records_stamps_and_rekeys_the_provider_range():
    ranges = _declared()
    frames, report = sk_tracking.convert_to_frames(
        _bronze(), home_team_id=str(_meta()["home_team"]["id"]), period_frame_ranges=ranges
    )
    assert report.provider_period_frame_ranges == ranges
    keyed = sk_tracking.period_frame_ranges_from_report(report)
    assert keyed == {(_MID, p): lohi for p, lohi in ranges.items()}
    assert frames.attrs[PERIOD_FRAME_RANGES_ATTR] == [[_MID, p, lo, hi] for p, (lo, hi) in sorted(ranges.items())]
    assert resolve_period_frame_ranges(frames) == keyed  # the attrs fallback reads the same record


def test_convert_to_frames_without_a_range_records_none_and_the_rekey_refuses():
    frames, report = sk_tracking.convert_to_frames(_bronze(), home_team_id=str(_meta()["home_team"]["id"]))
    assert report.provider_period_frame_ranges is None and PERIOD_FRAME_RANGES_ATTR not in frames.attrs
    with pytest.raises(ValueError, match="no provider period range"):
        sk_tracking.period_frame_ranges_from_report(report)


def test_a_range_excluding_retained_frames_raises_and_a_tuple_key_is_a_type_error():
    bronze = _bronze()
    home = str(_meta()["home_team"]["id"])
    p1_last = int(bronze.loc[bronze["period"] == 1, "frame"].max())
    short = {**_declared(), 1: (_declared()[1][0], p1_last - 1)}  # cuts the last retained period-1 frame
    with pytest.raises(ValueError, match="outside the supplied provider range"):
        sk_tracking.convert_to_frames(bronze, home_team_id=home, period_frame_ranges=short)
    with pytest.raises(TypeError, match="provider_period_frame_ranges"):
        sk_tracking.convert_to_frames(
            bronze,
            home_team_id=home,
            period_frame_ranges={(_MID, 1): (0, 10)},  # pyright: ignore[reportArgumentType]  # wrong key shape, on purpose
        )


def test_real_match_pairs_on_the_provider_basis_without_a_violation():
    meta = _meta()
    events = pd.read_csv(_OPEN / f"{_MID}_events_slim.csv", low_memory=False)
    actions = sk_spadl.convert_to_actions(events, meta)[0]
    frames, report = sk_tracking.convert_to_frames(
        _bronze(), home_team_id=str(meta["home_team"]["id"]), period_frame_ranges=_declared()
    )
    direction = gi_direction_check(
        actions, resolve_defended_goals(frames), flips=sk_tracking.frame_flips_from_report(report)
    )
    gp = assess_gi_pairing(
        actions,
        frames,
        provider="skillcorner",
        direction=direction,
        period_ranges=sk_tracking.period_frame_ranges_from_report(report),
    )[_MID]
    assert gp.range_basis == "provider" and gp.range_violations == () and gp.pair_range_ok is True
    # the other side: the slim bronze keeps every 100th frame, so its RETAINED range is narrower than the provider's
    # and the stricter fallback flags the possessions over its edges (frames without the attrs stamp = no range)
    plain = frames.copy()
    plain.attrs.pop(PERIOD_FRAME_RANGES_ATTR)
    retained = assess_gi_pairing(actions, plain, provider="skillcorner", direction=direction)[_MID]
    assert retained.range_basis == "retained" and retained.pair_range_ok is False
    assert {v[3] for v in retained.range_violations} <= {"partial_edge", "outside", "other_period"}
