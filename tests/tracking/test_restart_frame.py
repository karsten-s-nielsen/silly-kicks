"""Task 2.2 frame transform + GI direction check + per-match pairing (spec sections 4.3, 6.1). Both sides
of every bar."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import resolve_defended_goals
from silly_kicks.tracking._restart_frame import (
    GiDirectionMismatchWarning,
    GiPairing,
    alignment_offset,
    assess_gi_pairing,
    gi_direction_check,
    interval_agreement,
    pairing_proximity_profile,
    resolve_frame_flips,
    team_attacks_rtl,
    to_action_frame,
)


def _goal_frames() -> pd.DataFrame:
    # team 1 GK low-x (defends 0, attacks 105); team 2 GK high-x (defends 105, attacks 0).
    rows = []
    for p in (1, 2):
        rows += [
            {
                "game_id": "g",
                "period_id": p,
                "team_id": "1",
                "is_ball": False,
                "is_goalkeeper": True,
                "x": 4.0,
                "y": 34.0,
            },
            {
                "game_id": "g",
                "period_id": p,
                "team_id": "2",
                "is_ball": False,
                "is_goalkeeper": True,
                "x": 101.0,
                "y": 34.0,
            },
        ]
    return pd.DataFrame(rows)


GOAL_MAP = resolve_defended_goals(_goal_frames())
#: the frames were not flipped in either period: the GI side compares as delivered
NO_FLIP = {("g", 1): False, ("g", 2): False}


# ---- to_action_frame ----


def test_to_action_frame_reflect_identity_and_na():
    x = np.array([10.0, 10.0, 10.0])
    y = np.array([20.0, 20.0, 20.0])
    rtl = pd.array([True, False, pd.NA], dtype="boolean")
    xo, yo = to_action_frame(x, y, rtl)
    assert (xo[0], yo[0]) == (95.0, 48.0)  # reflected
    assert (xo[1], yo[1]) == (10.0, 20.0)  # identity
    assert np.isnan(xo[2]) and np.isnan(yo[2])  # <NA> -> NaN


# ---- team_attacks_rtl ----


def test_team_attacks_rtl_from_goalmap():
    actions = pd.DataFrame({"game_id": ["g", "g", "g"], "period_id": [1, 1, 1], "team_id": ["1", "2", "99"]})
    rtl = team_attacks_rtl(actions, GOAL_MAP)
    assert rtl.iloc[0] is False or rtl.iloc[0] == False  # team 1 attacks 105 -> not rtl  # noqa: E712
    assert bool(rtl.iloc[1]) is True  # team 2 attacks 0 -> rtl
    assert rtl.iloc[2] is pd.NA  # unknown team -> unresolved


# ---- gi_direction_check ----


def _acts(side_by_team: dict[str, list[str]], *, period: int = 1) -> pd.DataFrame:
    rows = []
    eid = 0
    for team, sides in side_by_team.items():
        for s in sides:
            rows.append(
                {
                    "game_id": "g",
                    "period_id": period,
                    "team_id": team,
                    "gi_attacking_side": s,
                    "gi_event_id": f"e{eid}",
                    "player_id": team,
                }
            )
            eid += 1
    return pd.DataFrame(rows)


def test_gi_direction_majority_agrees_not_withheld():
    # team 1 attacks 105 (ltr), team 2 attacks 0 (rtl). 5 agreeing possessions each.
    acts = pd.concat([_acts({"1": ["ltr"] * 5}), _acts({"2": ["rtl"] * 5})], ignore_index=True)
    chk = gi_direction_check(acts, GOAL_MAP, flips=NO_FLIP)
    assert not chk.withheld.any()
    assert chk.mismatched_periods == ()


def test_gi_direction_period_majority_disagrees_withholds_and_warns():
    # team 1 GI says rtl (majority) but GoalMap says team 1 attacks 105 (ltr) -> whole period withheld + warn.
    acts = _acts({"1": ["rtl"] * 5})
    with pytest.warns(GiDirectionMismatchWarning):
        chk = gi_direction_check(acts, GOAL_MAP, flips=NO_FLIP)
    assert chk.withheld.all()
    assert chk.mismatched_periods == (("g", 1),)


def test_gi_direction_insufficient_possessions_withheld_and_counted():
    acts = _acts({"1": ["ltr"] * 4})  # < 5
    chk = gi_direction_check(acts, GOAL_MAP, flips=NO_FLIP)
    assert chk.withheld.all() and chk.n_insufficient_team_periods == 1


def test_gi_direction_single_noise_possession_withheld():
    acts = _acts({"1": ["ltr", "ltr", "ltr", "ltr", "rtl"]})  # one disagrees with the ltr majority
    chk = gi_direction_check(acts, GOAL_MAP, flips=NO_FLIP)
    assert chk.n_noise_possessions == 1
    assert chk.withheld.sum() == 1  # only the noise possession row


def test_gi_direction_maps_the_delivered_side_through_the_recorded_flip():
    # R.2 2026-10-23: GI attacking_side is in the DELIVERED frame. Period 2 was flipped to left-to-right, so a
    # delivered "rtl" for team 1 is "ltr" in the frames -- which GoalMap (built from the oriented frames) says.
    acts = pd.concat([_acts({"1": ["rtl"] * 5}, period=2), _acts({"2": ["ltr"] * 5}, period=2)], ignore_index=True)
    chk = gi_direction_check(acts, GOAL_MAP, flips={("g", 1): False, ("g", 2): True})
    assert chk.mismatched_periods == () and not chk.withheld.any()
    assert chk.checked_periods == (("g", 2),) and chk.unresolved_periods == ()


def test_gi_direction_a_misflipped_period_is_detected():
    # non-vacuity: the SAME GI sides with the period recorded as NOT flipped disagree with GoalMap -> mismatch
    acts = pd.concat([_acts({"1": ["rtl"] * 5}, period=2), _acts({"2": ["ltr"] * 5}, period=2)], ignore_index=True)
    with pytest.warns(GiDirectionMismatchWarning):
        chk = gi_direction_check(acts, GOAL_MAP, flips={("g", 2): False})
    assert chk.mismatched_periods == (("g", 2),) and chk.withheld.all()


@pytest.mark.parametrize("flips", [None, {}, {("g", 1): None}], ids=["no-keyword", "empty", "none-flip"])
def test_gi_direction_without_a_recorded_flip_is_unresolved_not_a_mismatch(flips, recwarn):
    # no flip information for the period -> UNRESOLVED: withheld + counted, never compared, no mismatch warning
    acts = _acts({"1": ["rtl"] * 5})  # would MISMATCH if compared raw
    chk = gi_direction_check(acts, GOAL_MAP, flips=flips)
    assert chk.unresolved_periods == (("g", 1),)
    assert chk.mismatched_periods == () and chk.checked_periods == ()
    assert chk.withheld.all()
    assert not [w for w in recwarn if issubclass(w.category, GiDirectionMismatchWarning)]


def test_resolve_frame_flips_keyword_wins_then_attrs_then_none():
    frames = pd.DataFrame({"game_id": ["7"]})
    frames.attrs["ltr_flip_by_period"] = [["7", 1, True], ["7", 2, None]]
    assert resolve_frame_flips(frames) == {("7", 1): True, ("7", 2): None}
    assert resolve_frame_flips(frames, {(7, 1): False}) == {("7", 1): False}  # keyword first; ids canonical
    assert resolve_frame_flips(pd.DataFrame()) == {}  # no stamp, no keyword -> nothing recorded


# ---- interval_agreement / alignment_offset ----


def _poss_by_frame(period, frames, possessor):
    return pd.DataFrame({"period_id": period, "frame_id": list(frames), "possessor": possessor})


def test_interval_agreement_and_offset_aligned_and_shifted():
    intervals = pd.DataFrame({"period_id": [1], "frame_start": [10], "frame_end": [14], "possessor": ["A"]})
    offsets = np.arange(-20, 21)
    # aligned: possessor A at frames 10..14 -> offset 0 agreement 1.0
    aligned = _poss_by_frame(1, range(10, 15), "A")
    off, _n = alignment_offset(interval_agreement(intervals, aligned, offsets=offsets), offsets)
    assert off == 0
    # shifted by 5: possessor A at frames 15..19, another player on the (present) frames around it -> offset 5.
    # (Absent frames are missing data since the DL ruling 2026-10-25, Q4, so the surrounding frames are present.)
    shifted = _poss_by_frame(1, range(5, 25), ["B"] * 10 + ["A"] * 5 + ["B"] * 5)
    off2, _ = alignment_offset(interval_agreement(intervals, shifted, offsets=offsets), offsets)
    assert off2 == 5


def test_alignment_offset_ties_prefer_smallest_abs_then_negative():
    offsets = np.arange(-2, 3)
    agreement = np.array([1.0, 0.0, 1.0, 0.0, 1.0])  # ties at -2, 0, +2
    off, n_tied = alignment_offset(agreement, offsets)
    assert off == 0 and n_tied == 3
    agreement2 = np.array([1.0, 0.0, 0.0, 0.0, 1.0])  # ties at -2, +2 -> negative
    off2, n2 = alignment_offset(agreement2, offsets)
    assert off2 == -2 and n2 == 2


# ---- assess_gi_pairing ----


def _pair_frames(possessor_frames: dict[int, str], *, ball_xy=(50.0, 34.0), player_xy=(50.5, 34.0)) -> pd.DataFrame:
    rows = []
    for f, poss in possessor_frames.items():
        rows.append(
            {
                "game_id": "g",
                "period_id": 1,
                "frame_id": f,
                "player_id": None,
                "team_id": None,
                "is_ball": True,
                "is_goalkeeper": False,
                "x": ball_xy[0],
                "y": ball_xy[1],
                "visibility": True,
                "sc_possession_player_id": poss,
            }
        )
        rows.append(
            {
                "game_id": "g",
                "period_id": 1,
                "frame_id": f,
                "player_id": "A",
                "team_id": "1",
                "is_ball": False,
                "is_goalkeeper": False,
                "x": player_xy[0],
                "y": player_xy[1],
                "visibility": True,
                "sc_possession_player_id": poss,
            }
        )
    return pd.DataFrame(rows)


def _pair_actions(frame_start=10, frame_end=14) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["g"],
            "period_id": [1],
            "player_id": ["A"],
            "gi_event_id": ["e0"],
            "gi_possession_frame_start": [frame_start],
            "gi_possession_frame_end": [frame_end],
            "gi_attacking_side": ["ltr"],
        }
    )


_NO_MISMATCH = gi_direction_check(pd.DataFrame(), GOAL_MAP, flips=NO_FLIP)
#: a direction check with ONE compared, agreeing period of match "g" (team 1 attacks 105 = ltr): the
#: Direction test needs a compared period (fail closed on zero, DL ruling 2026-10-23)
_AGREES = gi_direction_check(_acts({"1": ["ltr"] * 5}), GOAL_MAP, flips=NO_FLIP)


def test_pairing_non_skillcorner_is_empty():
    assert assess_gi_pairing(_pair_actions(), _pair_frames({10: "A"}), provider="idsse", direction=_NO_MISMATCH) == {}


def test_pairing_aligned_is_paired():
    frames = _pair_frames({f: "A" for f in range(10, 15)})
    out = assess_gi_pairing(_pair_actions(), frames, provider="skillcorner", direction=_AGREES)
    gp = out["g"]
    assert gp.pairing == "paired" and gp.failed_test is None
    assert gp.alignment_offset == 0 and gp.alignment_source == "sc_possession"


def test_pairing_out_of_range_fails_range():
    frames = _pair_frames({f: "A" for f in range(10, 15)})  # frames 10..14
    # wholly outside (the edge rule clips a PARTLY-outside possession instead; second R.2 stop ruling)
    out = assess_gi_pairing(
        _pair_actions(frame_start=500, frame_end=999), frames, provider="skillcorner", direction=_AGREES
    )
    assert out["g"].pairing == "unpaired" and out["g"].failed_test == "range"


def test_pairing_low_proximity_fails_proximity():
    frames = _pair_frames({f: "A" for f in range(10, 15)}, player_xy=(10.0, 10.0))  # possessor far from ball
    out = assess_gi_pairing(_pair_actions(), frames, provider="skillcorner", direction=_AGREES)
    assert out["g"].pairing == "unpaired" and out["g"].failed_test == "proximity"


def test_pairing_shifted_fails_lag():
    # possessor A only at frames 15..19 -> alignment offset 5 > 1.
    frames = _pair_frames({f: ("A" if f >= 15 else "B") for f in range(10, 20)})
    out = assess_gi_pairing(_pair_actions(), frames, provider="skillcorner", direction=_AGREES)
    assert out["g"].pairing == "unpaired" and out["g"].failed_test == "lag"


def test_pairing_direction_mismatch_fails_direction():
    frames = _pair_frames({f: "A" for f in range(10, 15)})
    with pytest.warns(GiDirectionMismatchWarning):
        bad_dir = gi_direction_check(_acts({"1": ["rtl"] * 5}), GOAL_MAP, flips=NO_FLIP)  # flags ("g", 1)
    out = assess_gi_pairing(_pair_actions(), frames, provider="skillcorner", direction=bad_dir)
    assert out["g"].pairing == "unpaired" and out["g"].failed_test == "direction"


def test_pairing_records_every_verdict_even_after_an_earlier_failure():
    # DL ruling 2026-10-23: proximity fails, yet lag and direction are still evaluated and recorded
    frames = _pair_frames({f: "A" for f in range(10, 15)}, player_xy=(10.0, 10.0))
    gp = assess_gi_pairing(_pair_actions(), frames, provider="skillcorner", direction=_AGREES)["g"]
    assert gp.failed_test == "proximity"
    assert (gp.pair_range_ok, gp.pair_proximity_ok, gp.pair_lag_ok, gp.pair_direction_ok) == (True, False, True, True)
    ok = assess_gi_pairing(
        _pair_actions(), _pair_frames({f: "A" for f in range(10, 15)}), provider="skillcorner", direction=_AGREES
    )["g"]
    assert (ok.pair_range_ok, ok.pair_proximity_ok, ok.pair_lag_ok, ok.pair_direction_ok) == (True, True, True, True)


def test_pairing_no_evaluable_start_is_not_evaluable_and_fails():
    frames = _pair_frames({f: "A" for f in range(10, 15)}).assign(visibility=False)  # nothing detected
    gp = assess_gi_pairing(_pair_actions(), frames, provider="skillcorner", direction=_AGREES)["g"]
    assert gp.pair_proximity_ok is None and gp.failed_test == "proximity" and gp.pairing == "unpaired"


def test_pairing_all_periods_unresolved_fails_direction_closed():
    # zero compared periods = absence of evidence, never agreement (fail closed)
    frames = _pair_frames({f: "A" for f in range(10, 15)})
    unresolved = gi_direction_check(_acts({"1": ["ltr"] * 5}), GOAL_MAP, flips={("g", 1): None})
    gp = assess_gi_pairing(_pair_actions(), frames, provider="skillcorner", direction=unresolved)["g"]
    assert gp.pairing == "unpaired" and gp.failed_test == "direction" and gp.pair_direction_ok is False
    assert gp.n_unresolved_periods == 1


def test_pairing_one_resolved_and_one_unresolved_period_pairs_and_withholds_the_unresolved():
    frames = _pair_frames({f: "A" for f in range(10, 15)})
    acts = pd.concat([_acts({"1": ["ltr"] * 5}), _acts({"1": ["ltr"] * 5}, period=2)], ignore_index=True)
    chk = gi_direction_check(acts, GOAL_MAP, flips={("g", 1): False, ("g", 2): None})
    assert chk.withheld[acts["period_id"] == 2].all() and not chk.withheld[acts["period_id"] == 1].any()
    gp = assess_gi_pairing(_pair_actions(), frames, provider="skillcorner", direction=chk)["g"]
    assert gp.pairing == "paired" and gp.n_unresolved_periods == 1


def test_pairing_insufficient_team_periods_are_counted_per_match():
    frames = _pair_frames({f: "A" for f in range(10, 15)})
    acts = pd.concat([_acts({"1": ["ltr"] * 5}), _acts({"2": ["rtl"] * 3})], ignore_index=True)
    chk = gi_direction_check(acts, GOAL_MAP, flips=NO_FLIP)
    gp = assess_gi_pairing(_pair_actions(), frames, provider="skillcorner", direction=chk)["g"]
    assert gp.n_insufficient_team_periods == 1 and gp.pairing == "paired"


def test_pairing_matches_frames_by_canonical_game_id_across_dtypes():
    # C2 delta C2-DELTA-01: int action game ids meet string frame game ids (ADR-019), never "not evaluable"
    frames = _pair_frames({f: "A" for f in range(10, 15)}).assign(game_id="7", time_seconds=lambda d: d.frame_id / 10)
    acts = _pair_actions().assign(game_id=7)
    gp = assess_gi_pairing(acts, frames, provider="skillcorner", direction=_AGREES)[7]
    assert gp.n_evaluable == 1 and gp.pair_proximity_ok is True
    prof = pairing_proximity_profile(acts, frames, prox_ms=(2.0,), shifts_s=())
    assert prof["n_evaluable"].tolist() == [1]


def _multi_actions(intervals: list[tuple[int, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": "g",
            "period_id": 1,
            "player_id": "A",
            "gi_event_id": [f"e{i}" for i in range(len(intervals))],
            "gi_possession_frame_start": [a for a, _ in intervals],
            "gi_possession_frame_end": [b for _, b in intervals],
            "gi_attacking_side": "ltr",
        }
    )


def _two_period_frames() -> pd.DataFrame:
    """Retained frames: P1 10..19, P2 40..49 (possessor A at the ball throughout)."""
    p1 = _pair_frames({f: "A" for f in range(10, 20)})
    p2 = _pair_frames({f: "A" for f in range(40, 50)}).assign(period_id=2)
    return pd.concat([p1, p2], ignore_index=True)


def _periods_actions(intervals: list[tuple[int, int, int]]) -> pd.DataFrame:
    """GI possessions as (period_id, frame_start, frame_end)."""
    return _multi_actions([(a, b) for _p, a, b in intervals]).assign(period_id=[p for p, _a, _b in intervals])


#: the PROVIDER period ranges of match "g" (empty frames included): P1 10..25, P2 30..60
_PROVIDER = {("g", 1): (10, 25), ("g", 2): (30, 60)}


def test_range_provider_basis_admits_a_frame_inside_the_raw_range_absent_from_the_retained_frames():
    # the 2007448 shape (DL ruling 2026-10-25, Q4): a 1-frame P2 possession at frame 35, inside the provider P2 range
    # (30..60) but before the first retained P2 frame (40) -> Range passes on the provider basis
    acts = _periods_actions([(1, 10, 14), (2, 35, 35), (2, 40, 44)])
    gp = assess_gi_pairing(
        acts, _two_period_frames(), provider="skillcorner", direction=_AGREES, period_ranges=_PROVIDER
    )["g"]
    assert gp.pair_range_ok is True and gp.range_basis == "provider" and gp.range_violations == ()
    assert gp.pairing == "paired"
    # the other side: no supplied range -> the RETAINED basis (stricter) fails the same possession, classified
    ret = assess_gi_pairing(acts, _two_period_frames(), provider="skillcorner", direction=_AGREES)["g"]
    assert ret.range_basis == "retained" and ret.pair_range_ok is False and ret.failed_test == "range"
    assert ret.range_violations == ((2, 35, 35, "outside"),)


@pytest.mark.parametrize(
    ("violation", "cls"),
    [
        ((1, 17, 40), "partial_edge"),  # the retired clipping's case: it FAILS now, nothing is clipped
        ((1, 40, 44), "other_period"),  # the GI period-label defect shape (frames of P2, labelled P1)
        ((1, 70, 80), "outside"),
        ((3, 10, 12), "no_period_range"),
    ],
)
def test_range_violation_classes_each_fail_the_match(violation, cls):
    acts = _periods_actions([(1, 10, 14), violation])
    gp = assess_gi_pairing(
        acts, _two_period_frames(), provider="skillcorner", direction=_AGREES, period_ranges=_PROVIDER
    )["g"]
    assert gp.pair_range_ok is False and gp.failed_test == "range" and gp.pairing == "unpaired"
    assert gp.range_violations == ((*violation, cls),) and gp.range_basis == "provider"
    # non-vacuity: the valid possession alone pairs on the same frames and ranges
    ok = assess_gi_pairing(
        _periods_actions([(1, 10, 14)]),
        _two_period_frames(),
        provider="skillcorner",
        direction=_AGREES,
        period_ranges=_PROVIDER,
    )["g"]
    assert ok.pair_range_ok is True and ok.range_violations == ()


def test_range_reads_the_attrs_stamp_and_rejects_a_period_only_key():
    from silly_kicks.tracking._restart_frame import resolve_period_frame_ranges
    from silly_kicks.tracking.schema import PERIOD_FRAME_RANGES_ATTR

    frames = _two_period_frames()
    frames.attrs[PERIOD_FRAME_RANGES_ATTR] = [["g", 1, 10, 25], ["g", 2, 30, 60]]
    acts = _periods_actions([(1, 10, 14), (2, 35, 35)])
    gp = assess_gi_pairing(acts, frames, provider="skillcorner", direction=_AGREES)["g"]
    assert gp.range_basis == "provider" and gp.pair_range_ok is True
    assert resolve_period_frame_ranges(None, {(7, 1): (1, 2)}) == {("7", 1): (1, 2)}  # canonical game id
    with pytest.raises(TypeError, match="period_frame_ranges_from_report"):
        assess_gi_pairing(
            acts,
            frames,
            provider="skillcorner",
            direction=_AGREES,
            period_ranges={1: (10, 25)},  # pyright: ignore[reportArgumentType]  # the wrong key shape, on purpose
        )


def test_lag_negative_control_shifted_gi_fails_and_zero_agreement_is_not_evaluable():
    # GI shifted +5 s (50 frames at 10 fps, beyond the +/-20 scan): no offset agrees -> lag not evaluable -> fails
    frames = _pair_frames({f: "A" for f in range(10, 15)})
    shifted = _pair_actions(frame_start=60, frame_end=64)
    frames = pd.concat([frames, _pair_frames({f: "B" for f in range(58, 70)})], ignore_index=True)
    gp = assess_gi_pairing(shifted, frames, provider="skillcorner", direction=_AGREES)["g"]
    assert gp.pair_lag_ok is None and gp.failed_test in ("proximity", "lag") and gp.pairing == "unpaired"
    # A-02: no offset is recorded, so the lag table and its median guard never see a fake 0
    assert gp.alignment_offset is None and gp.n_tied_offsets == 0
    from scripts._tf46p2_common import gi_pairing_report, pairing_lag_table
    from silly_kicks.tracking._restart_validation import DataProfile

    rep = gi_pairing_report({"g": gp}, DataProfile("skillcorner", "r", "g", "cs"))
    assert pairing_lag_table(rep).empty
    aligned = assess_gi_pairing(_pair_actions(), frames, provider="skillcorner", direction=_AGREES)["g"]
    assert aligned.pair_lag_ok is True  # non-vacuity: the same frames pass when GI is aligned


def test_lag_tolerance_two_frames_passes_an_offset_of_two():
    # the sc_possession field leading GI by 2 frames (the measured -2 case) passes at pair_lag_tol_frames = 2
    frames = _pair_frames({f: ("A" if 8 <= f <= 12 else "B") for f in range(5, 20)})
    gp = assess_gi_pairing(_pair_actions(), frames, provider="skillcorner", direction=_AGREES)["g"]
    assert gp.alignment_offset == -2 and gp.pair_lag_ok is True


def test_the_retired_edge_clip_surface_is_gone():
    # DL ruling 2026-10-25 (Q5): the 2026-10-23 edge rule is retired with no clipping left behind
    import silly_kicks.tracking._kick_resolver as kr
    import silly_kicks.tracking._restart_frame as rf

    fields = {f.name for f in dataclasses.fields(GiPairing)}
    assert {"n_edge_clipped", "edge_clipped"}.isdisjoint(fields) and {"range_basis", "range_violations"} <= fields
    assert not hasattr(rf, "_range_split") and not hasattr(kr, "_reads_clipped_possession")
    assert "gi_edge_clipped" not in kr._empty_report()


def test_interval_agreement_drops_absent_frames_and_keeps_a_present_unpossessed_frame_as_a_miss():
    # DL ruling 2026-10-25 (Q4): an absent frame is missing data (out of the denominator), not disagreement
    intervals = pd.DataFrame({"period_id": [1], "frame_start": [10], "frame_end": [19], "possessor": ["A"]})
    offsets = np.arange(-3, 4)
    # frames 10..12 are ABSENT from the tracking (a gap at the interval start); A possesses every present frame 13..22
    present = _poss_by_frame(1, range(13, 23), "A")
    agree = interval_agreement(intervals, present, offsets=offsets)
    assert agree[3] == pytest.approx(1.0)  # offset 0: the 7 present frames are all A; the 3 absent ones do not count
    off, _ = alignment_offset(agree, offsets)
    assert off == 0
    # the former denominator (an absent frame counted as a miss) moved the argmax to the offset that avoids the gap
    as_misses = _poss_by_frame(1, range(7, 23), [None] * 6 + ["A"] * 10)
    off_old, _ = alignment_offset(interval_agreement(intervals, as_misses, offsets=offsets), offsets)
    assert off_old == 3
    # a PRESENT frame with no possessor stays a miss
    with_gap = _poss_by_frame(1, range(10, 20), ["A"] * 5 + [None] + ["A"] * 4)
    assert interval_agreement(intervals, with_gap, offsets=np.array([0]))[0] == pytest.approx(0.9)


def test_a_ball_only_frame_is_present_on_both_possessor_paths():
    # A C2-DELTA-08 / B C2-DELTA-08: one definition of a present frame (any tracking row at (period, frame))
    from silly_kicks.tracking._restart_frame import _DetectedIndex, _possessor_by_frame

    frames = _pair_frames({f: "A" for f in range(10, 15)})
    ball_only = frames[frames["is_ball"] & (frames["frame_id"] == 12)]
    frames = pd.concat([frames[frames["frame_id"] != 12], ball_only], ignore_index=True)  # frame 12: the ball only
    poss = pd.DataFrame({"period_id": [1], "frame_start": [10], "frame_end": [14], "possessor": ["A"]})
    _src, sc = _possessor_by_frame(frames, poss, 2.0, _DetectedIndex(frames))
    _src2, fb = _possessor_by_frame(frames.drop(columns=["sc_possession_player_id"]), poss, 2.0, _DetectedIndex(frames))
    assert sorted(sc["frame_id"]) == sorted(fb["frame_id"]) == list(range(10, 15))  # 12 present on both paths
    assert sc.set_index("frame_id").loc[12, "possessor"] == "A"  # the frame-level possessor from the ball row


def test_proximity_fallback_rows_cover_present_frames_only():
    # the marked fallback (no sc_possession column): every PRESENT frame gets a row, an absent one none
    from silly_kicks.tracking._restart_frame import _DetectedIndex, _possessor_by_frame

    frames = _pair_frames({f: "A" for f in range(12, 20)}).drop(columns=["sc_possession_player_id"])
    poss = pd.DataFrame({"period_id": [1], "frame_start": [10], "frame_end": [14], "possessor": ["A"]})
    source, table = _possessor_by_frame(frames, poss, 2.0, _DetectedIndex(frames))
    assert source == "proximity_fallback"
    assert sorted(table["frame_id"]) == list(range(12, 20))  # 10, 11 absent -> no row; 15..19 present -> rows
    assert table.set_index("frame_id").loc[[12, 13, 14], "possessor"].tolist() == ["A", "A", "A"]
    assert table.set_index("frame_id").loc[[15, 19], "possessor"].isna().all()


def test_pairing_proximity_profile_shifted_controls_fall_below_the_match():
    # the possessor is at the ball only at frame 10; a 1 s shift (10 frames at 10 fps) lands where it is not
    rows = []
    for f in range(0, 41):
        near = f == 10
        for is_ball, pid, x in ((True, None, 50.0), (False, "A", 50.5 if near else 80.0)):
            rows.append(
                {
                    "game_id": "g",
                    "period_id": 1,
                    "frame_id": f,
                    "time_seconds": f / 10.0,
                    "player_id": pid,
                    "is_ball": is_ball,
                    "x": x,
                    "y": 34.0,
                    "visibility": True,
                }
            )
    prof = pairing_proximity_profile(
        _pair_actions(frame_start=10, frame_end=12), pd.DataFrame(rows), prox_ms=(2.0,), shifts_s=(-1.0, 1.0)
    )
    by = dict(zip(prof["shift_s"].astype(float), prof["prox_share"].astype(float), strict=True))
    assert by[0.0] == 1.0 and by[1.0] == 0.0 and by[-1.0] == 0.0
    assert set(prof["n_evaluable"]) == {1}


# ---- sc_possession_player_id reader allow-list (B-19 / DL ruling) ----

import pathlib  # noqa: E402

#: `sc_possession_player_id` may appear ONLY here: the single READER (the pairing in `_restart_frame`) plus
#: declaration / plumbing (the converter writer, the schema, the reflection registry, the bronze writer). It is
#: a SkillCorner-processed output that must never reach a gated signal (spec section 17.7). NOTE (minor plan
#: deviation, flagged for the A/B handoff): the plan's B-19 list also named `feature_glossary.py`, but the
#: field is a raw frame adapter column (like `visibility` / `ball_state` / `z_source`, none of which are in the
#: glossary), so it carries no glossary entry -- the glossary documents computed features, not provider fields.
_SC_POSS_ALLOWLIST = {
    "silly_kicks/tracking/_restart_frame.py",
    "silly_kicks/tracking/skillcorner.py",
    "silly_kicks/tracking/schema.py",
    "silly_kicks/reflection.py",
    "scripts/_loader_pining.py",
}


def _literal_sites(roots, literal: str) -> set[str]:
    out: set[str] = set()
    for root in roots:
        for py in pathlib.Path(root).rglob("*.py"):
            try:
                if literal in py.read_text(encoding="utf-8"):
                    out.add(py.as_posix())
            except (OSError, UnicodeDecodeError):
                continue
    return out


def test_sc_possession_player_id_reader_allowlist():
    hits = _literal_sites(("silly_kicks", "scripts"), "sc_possession_player_id")
    assert hits == _SC_POSS_ALLOWLIST, f"unexpected reader(s): {sorted(hits - _SC_POSS_ALLOWLIST)}"


def test_reader_allowlist_meta_detects_a_plant(tmp_path):
    # Meta-assertion: planting the literal in another module makes the scan find it (the gate can fail).
    (tmp_path / "planted.py").write_text('leak = "sc_possession_player_id"\n', encoding="utf-8")
    assert _literal_sites((str(tmp_path),), "sc_possession_player_id") == {(tmp_path / "planted.py").as_posix()}
