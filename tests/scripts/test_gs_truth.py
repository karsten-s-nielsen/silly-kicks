"""The shared GS truth module (scorer spec S4): the split rule, the PFF on-ball sequences, the next-touch measures
and the selection order, used by the harness and the scorer driver."""

from __future__ import annotations

import pandas as pd

_DEV = [
    "3816",
    "3819",
    "3820",
    "3822",
    "3826",
    "3829",
    "3830",
    "3832",
    "3837",
    "3838",
    "3840",
    "3846",
    "3847",
    "3848",
    "3849",
    "3851",
    "3854",
    "10504",
    "10506",
    "10507",
    "10509",
    "10513",
    "10514",
    "10516",
]


_ROSTER = [str(i) for i in [*range(3812, 3860), *range(10504, 10518)]]  # the 62 staged GS matches


def test_split_rule_is_the_declared_hash():
    import scripts._gs_truth as d

    assert [i for i in _ROSTER if d.split_of(i) == "dev"] == _DEV
    assert sum(d.split_of(i) == "holdout" for i in _ROSTER) == 38
    assert d.split_of(3816) == d.split_of("3816") == d.split_of(3816.0) == "dev"  # the id form is canonical
    import pytest

    with pytest.raises(ValueError):
        d.split_of("gradientsports__3816")  # the staged key is never hashed as is


def _seq(rows):
    """rows: (period, t0, t1, player, ptype)."""
    return pd.DataFrame(rows, columns=["period_id", "t0", "t1", "player", "ptype"])


def _contacts(rows):
    """rows: (period, t, player or None)."""
    return pd.DataFrame(
        {
            "period_id": [r[0] for r in rows],
            "time_seconds": [r[1] for r in rows],
            "player_id": pd.array([r[2] for r in rows], dtype="Int64"),
            "contact_source": ["unresolved" if r[2] is None else "velocity_break" for r in rows],
        }
    )


def test_next_touch_classes_on_a_toy():
    import scripts._gs_truth as d

    seq = _seq(
        [
            (1, 0.0, 1.0, 1, "PA"),  # 1 -> 2: right (2 at 2.05, within 0.2 s of 2.0)
            (1, 2.0, 3.0, 2, "PA"),  # 2 -> 3: false first (5 at 3.5, before 4.0 - 0.2)
            (1, 4.0, 5.0, 3, "PA"),  # 3 -> 4: wrong player (5 at 6.1, near 6.0)
            (1, 6.0, 7.0, 4, "PA"),  # 4 -> 6: missed (nothing until 8.2)
            (1, 8.0, 9.0, 6, "IT"),
        ]
    )
    c = _contacts([(1, 1.01, 1), (1, 2.05, 2), (1, 3.5, 5), (1, 4.1, 3), (1, 6.1, 5), (1, 8.5, 7)])
    s = d.score_next_touch(seq, c)
    assert (s["n"], s["right"], s["false_first"], s["wrong_player"], s["missed"]) == (4, 1, 1, 1, 1)


def test_release_timing_on_a_toy():
    import scripts._gs_truth as d

    seq = _seq([(1, 0.0, 1.0, 1, "PA"), (1, 2.0, 3.0, 2, "CR"), (1, 4.0, 5.0, 3, "IT")])
    c = _contacts([(1, 0.96, 1), (1, 4.9, 3)])  # player 1 releases 0.04 s early; player 2's release has no contact
    s = d.score_next_touch(seq, c)
    assert s["release_offsets"][0] == pytest_approx(-0.04)
    assert len(s["release_offsets"]) == 2 and pd.isna(s["release_offsets"][1])  # IT is not a kick type


def test_release_timing_takes_the_nearest_contact():
    import scripts._gs_truth as d

    seq = _seq([(1, 0.0, 1.0, 1, "PA"), (1, 2.0, 3.0, 2, "IT")])
    c = _contacts([(1, 0.7, 1), (1, 1.05, 1)])  # the first P1 contact in +-0.5 s is 0.3 s early; the nearest is late
    assert d.score_next_touch(seq, c)["release_offsets"] == [pytest_approx(0.05)]


def test_population_needs_s2_after_s1():
    import scripts._gs_truth as d

    seq = _seq([(1, 0.0, 1.0, 1, "PA"), (1, 1.0, 2.0, 2, "IT")])  # S2 starts exactly at S1's end: not in the measure
    assert d.score_next_touch(seq, _contacts([(1, 1.0, 2)]))["n"] == 0


def test_class_edges_are_inside_tau():
    import scripts._gs_truth as d

    seq = _seq([(1, 0.0, 1.0, 1, "PA"), (1, 2.0, 3.0, 2, "IT")])
    early = d.score_next_touch(seq, _contacts([(1, 1.75, 2)]), tau=0.25)  # exactly S2's start - tau
    late = d.score_next_touch(seq, _contacts([(1, 2.25, 2)]), tau=0.25)  # exactly S2's start + tau
    assert (early["right"], early["false_first"]) == (1, 0)
    assert (late["right"], late["missed"]) == (1, 0)


def pytest_approx(v):
    import pytest

    return pytest.approx(v, abs=1e-9)


#: The rule detector's order towards the stricter point: the larger change threshold, then the smaller radius.
_RULE = lambda k: (-k[1], k[0])  # noqa: E731


def _grid(points):
    """points: {(radius, dv): (right, false_first, timing)}."""
    return {k: {"right_share": r, "false_first_share": f, "timing_median_s": t} for k, (r, f, t) in points.items()}


def test_selection_order_is_total():
    import scripts._gs_truth as d

    radius_only = _grid({(1.0, 3.0): (0.70, 0.05, 0.0), (1.2, 3.0): (0.70, 0.05, 0.0)})
    assert d.select_point(radius_only, prefer=_RULE, population="all") == (
        1.0,
        3.0,
    )  # the smaller radius breaks the tie
    threshold_only = _grid({(1.2, 3.0): (0.70, 0.05, 0.0), (1.2, 4.0): (0.70, 0.05, 0.0)})
    assert d.select_point(threshold_only, prefer=_RULE, population="all") == (
        1.2,
        4.0,
    )  # the larger threshold breaks the tie
    disagree = _grid({(1.0, 3.0): (0.70, 0.05, 0.0), (1.2, 4.0): (0.70, 0.05, 0.0)})
    assert d.select_point(disagree, prefer=_RULE, population="all") == (
        1.2,
        4.0,
    )  # the keys disagree: the threshold key comes first


def test_bar_rows_are_inclusive():
    import scripts._gs_truth as d

    def m(right, false_first, timing):
        return {"right_share": right, "false_first_share": false_first, "timing_median_s": timing}

    assert d.meets_bar(m(0.60, 0.10, 0.10), population="all") and d.meets_bar(
        m(0.60, 0.10, -0.10), population="all"
    )  # every row at its edge
    assert not d.meets_bar(m(0.60, 0.10, 0.1001), population="all") and not d.meets_bar(
        m(0.60, 0.10, -0.1001), population="all"
    )
    assert not d.meets_bar(m(0.5999, 0.10, 0.0), population="all") and not d.meets_bar(
        m(0.60, 0.1001, 0.0), population="all"
    )


def test_near_best_window_both_sides():
    import scripts._gs_truth as d

    inside = _grid({(1.2, 2.0): (0.700, 0.05, 0.0), (1.0, 5.0): (0.696, 0.05, 0.0)})  # 0.004 below the best
    assert d.select_point(inside, prefer=_RULE, population="all") == (
        1.0,
        5.0,
    )  # in the window, so the stricter point wins
    outside = _grid({(1.2, 2.0): (0.700, 0.05, 0.0), (1.0, 5.0): (0.694, 0.05, 0.0)})  # 0.006 below the best
    assert d.select_point(outside, prefer=_RULE, population="all") == (1.2, 2.0)


def test_no_feasible_point_stops_before_holdout():
    import scripts._gs_truth as d

    assert (
        d.select_point(
            _grid({(1.2, 3.0): (0.55, 0.05, 0.0), (1.0, 3.0): (0.70, 0.12, 0.0)}), prefer=_RULE, population="all"
        )
        is None
    )


def test_otb_sequences_from_raw_events():
    import scripts._gs_truth as d

    ev = [
        {"startTime": 100.0, "gameEvents": {"gameEventType": "FIRSTKICKOFF", "period": 1}},
        {
            "startTime": 105.0,
            "endTime": 107.5,
            "gameEvents": {"gameEventType": "OTB", "period": 1, "playerId": 9},
            "possessionEvents": {"possessionEventType": "PA"},
        },
        {"startTime": 108.0, "endTime": 108.0, "gameEvents": {"gameEventType": "OUT", "period": 1}},
    ]
    s = d.otb_sequences(ev)
    assert s[["period_id", "t0", "t1", "player", "ptype"]].values.tolist() == [[1, 5.0, 7.5, "9", "PA"]]
    late_kickoff = [*ev, {"startTime": 300.0, "gameEvents": {"gameEventType": "FIRSTKICKOFF", "period": 1}}]
    assert d.otb_sequences(late_kickoff)["t0"].tolist() == [5.0]  # the period's first kickoff anchors, not a later one


def test_release_timing_population_includes_a_last_release():
    # Review A C46: the timing population is every S1 of a kick type, also one with no following pair.
    import scripts._gs_truth as d

    seq = _seq([(1, 0.0, 1.0, 1, "IT"), (1, 2.0, 3.0, 2, "SH")])  # the shot is the period's last sequence
    s = d.score_next_touch(seq, _contacts([(1, 3.02, 2)]))
    assert s["release_offsets"] == [pytest_approx(0.02)]


# ---------------------------------------------------------------------------------- the two populations (S r7 3)
def _frames(ball, players, *, fps=25.0, period=1):
    """ball: [(t, x, y)]; players: {pid: [(t, x, y)]} -> tracking frames with frame_id = round(t * fps)."""
    rows = [
        {
            "period_id": period,
            "frame_id": round(t * fps),
            "time_seconds": t,
            "is_ball": True,
            "player_id": pd.NA,
            "x": x,
            "y": y,
        }
        for t, x, y in ball
    ]
    for pid, track in players.items():
        rows += [
            {
                "period_id": period,
                "frame_id": round(t * fps),
                "time_seconds": t,
                "is_ball": False,
                "player_id": pid,
                "x": x,
                "y": y,
            }
            for t, x, y in track
        ]
    out = pd.DataFrame(rows)
    out["player_id"] = out["player_id"].astype("Int64")
    return out


def _still_ball(t0, t1, fps=25.0):
    import numpy as np

    return [(float(t), 50.0, 30.0) for t in np.round(np.arange(round(t0 * fps), round(t1 * fps) + 1) / fps, 6)]


def test_track_supported_radius_both_sides():
    import scripts._gs_truth as d

    seq = _seq([(1, 0.0, 1.0, 1, "PA"), (1, 2.0, 3.0, 2, "IT"), (1, 4.0, 5.0, 3, "IT")])
    ball = _still_ball(0.0, 5.0)
    near = {2: [(2.0, 51.49, 30.0)], 3: [(4.0, 51.51, 30.0)]}  # 1.49 m in, 1.51 m out
    flags = d.track_supported(seq, _frames(ball, near))
    assert flags.dtype == bool and flags.tolist() == [False, True, False]
    # Review B CD-SB-101 (T6): the radius is inclusive; exactly 1.5 m is supported.
    edge = d.track_supported(seq, _frames(ball, {2: [(2.0, 51.5, 30.0)]}))
    assert edge.tolist() == [False, True, False]


def test_the_supported_near_best_window_reads_the_supported_best():
    # Review B CD-SB-99 (T5): the best "right" of the window comes from the same population as the window.
    import scripts._gs_truth as d

    def m(right, sup_right):
        return {
            "right_share": right,
            "false_first_share": 0.05,
            "sup_right_share": sup_right,
            "sup_false_first_share": 0.05,
            "timing_median_s": 0.0,
        }

    grid = {0.4: m(0.40, 0.70), 0.5: m(0.60, 0.66)}  # the supported best is 0.4; all pairs would favour 0.5
    assert d.select_point(grid, prefer=lambda th: -th, population="supported") == 0.4


def test_track_supported_window_both_sides():
    import scripts._gs_truth as d

    seq = _seq([(1, 0.0, 1.0, 1, "PA"), (1, 2.0, 3.0, 2, "IT"), (1, 4.0, 5.0, 3, "IT")])
    ball = _still_ball(0.0, 5.0)
    at_edge = {2: [(2.2, 50.5, 30.0)], 3: [(4.24, 50.5, 30.0), (3.76, 50.5, 30.0)]}  # t2 + tau in; 0.24 s out
    assert d.track_supported(seq, _frames(ball, at_edge), tau=0.2).tolist() == [False, True, False]


def test_track_supported_needs_finite_rows():
    import numpy as np

    import scripts._gs_truth as d

    seq = _seq([(1, 0.0, 1.0, 1, "PA"), (1, 2.0, 3.0, 2, "IT"), (1, 4.0, 5.0, 3, "IT"), (1, 6.0, 7.0, 4, "IT")])
    ball = [b if abs(b[0] - 6.0) > 1e-9 else (6.0, np.nan, np.nan) for b in _still_ball(0.0, 7.0)]
    players = {
        3: [(4.0, np.nan, np.nan)],  # only a non-finite row: unsupported
        4: [(6.0, 50.5, 30.0)],  # near the ball, but the ball row of that frame is non-finite: skipped
    }  # player 2 has no row at all
    assert d.track_supported(seq, _frames(ball, players)).tolist() == [False, False, False, False]


def test_supported_counts_and_unread_first_rows():
    import scripts._gs_truth as d

    seq = _seq(
        [
            (1, 0.0, 1.0, 1, "PA"),
            (1, 2.0, 3.0, 2, "PA"),  # pair 1 -> 2: supported, right
            (1, 4.0, 5.0, 3, "PA"),  # pair 2 -> 3: unsupported, missed
            (2, 0.0, 1.0, 4, "PA"),  # first row of period 2: flagged True, the S2 of no pair
            (2, 2.0, 3.0, 5, "IT"),  # pair 4 -> 5: supported, false first
        ]
    )
    c = _contacts([(1, 2.05, 2), (2, 1.5, 7)])
    plain = d.score_next_touch(seq, c)
    assert not any(k.startswith("sup_") for k in plain)  # no column, no keys: every current caller is unchanged
    s = d.score_next_touch(seq.assign(supported=[False, True, False, True, True]), c)
    assert {k: s[k] for k in ("n", "right", "false_first", "missed")} == {
        k: plain[k] for k in ("n", "right", "false_first", "missed")
    }
    assert (s["sup_n"], s["sup_right"], s["sup_false_first"], s["sup_wrong_player"], s["sup_missed"]) == (2, 1, 1, 0, 0)


def test_shares_adds_the_supported_keys_only_with_their_counts():
    import scripts._gs_truth as d

    base = dict.fromkeys(
        (
            "n",
            "right",
            "false_first",
            "wrong_player",
            "missed",
            "n_block",
            "block_right",
            "block_missed",
            "n_contacts",
            "n_unresolved",
        ),
        0,
    ) | {"n": 10, "right": 4}
    plain = d.shares(base, [0.0])
    assert (
        plain["right_share"] == pytest_approx(0.4) and "sup_right_share" not in plain and "supported_share" not in plain
    )
    sup = d.shares(
        base | {"sup_n": 5, "sup_right": 4, "sup_false_first": 1, "sup_wrong_player": 0, "sup_missed": 0}, [0.0]
    )
    assert sup["right_share"] == pytest_approx(0.4)  # right_share keeps its all-pairs meaning
    assert (sup["sup_n"], sup["sup_right_share"], sup["sup_false_first_share"]) == (
        5,
        pytest_approx(0.8),
        pytest_approx(0.2),
    )
    assert (sup["sup_wrong_player_share"], sup["sup_missed_share"], sup["supported_share"]) == (
        0.0,
        0.0,
        pytest_approx(0.5),
    )


def test_the_population_decides_the_point():
    import pytest

    import scripts._gs_truth as d

    def m(right, ff, sup_right, sup_ff):
        return {
            "right_share": right,
            "false_first_share": ff,
            "sup_right_share": sup_right,
            "sup_false_first_share": sup_ff,
            "timing_median_s": 0.0,
        }

    grid = {0.4: m(0.62, 0.09, 0.50, 0.12), 0.5: m(0.45, 0.20, 0.66, 0.09)}
    assert d.select_point(grid, prefer=lambda th: -th, population="all") == 0.4
    assert d.select_point(grid, prefer=lambda th: -th, population="supported") == 0.5
    assert d.meets_bar(grid[0.5], population="supported") and not d.meets_bar(grid[0.5], population="all")
    with pytest.raises(KeyError):
        d.meets_bar({"right_share": 0.7, "false_first_share": 0.0, "timing_median_s": 0.0}, population="supported")
    with pytest.raises(ValueError):
        d.meets_bar(grid[0.5], population="both")
