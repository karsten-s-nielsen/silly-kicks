"""TF-49 secured_reception -- retains() skeleton + REQUIRED foul-skip (spec s3).

Keystone fixtures per the spec s7 list: fouled-receiver NOT a loss; saved shot
True (both the 3-row shot-IS-reception and the 4-row touch-then-shot shapes);
foul -> shot_penalty True; opponent possession boundary False; behind-line
same-team return False; empty-window extension (opponent restart False vs
late-foul-then-same-team True; line_x does NOT extend); truncation NaN;
possession_id present AND absent; NaN-team rows skipped (ADR-027); window
anchored at the RECEPTION row; non-RangeIndex inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl.utils import _resolve_next_touch_positions, add_possessions
from silly_kicks.tracking._packing import secured_reception

_T = spadlconfig.actiontype_id

A = 1
B = 2


def _acts(rows):
    """rows: dicts with keys team, t, and optional type ('pass'), x (60.0), period (1)."""
    return pd.DataFrame(
        {
            "game_id": 1,
            "period_id": [r.get("period", 1) for r in rows],
            "action_id": range(len(rows)),
            "time_seconds": [float(r["t"]) for r in rows],
            "team_id": [r["team"] for r in rows],
            "type_id": [_T[r.get("type", "pass")] for r in rows],
            "start_x": [float(r.get("x", 60.0)) for r in rows],
        }
    )


def _line_x(n, at0=55.0):
    lx = pd.Series(np.full(n, np.nan), dtype="float64")
    lx.iloc[0] = at0
    return lx


def _is_true(v):
    return (not pd.isna(v)) and bool(v)


def _is_false(v):
    return (not pd.isna(v)) and not bool(v)


def test_fouled_receiver_is_not_a_loss():
    """F2 keystone: heuristic possessions emit a boundary AT the foul row -- the
    foul-skip is what keeps this True (a bare possession-boundary rule flips it)."""
    a = _acts(
        [
            {"team": A, "t": 10, "x": 40},
            {"team": A, "t": 11, "x": 58},  # reception
            {"team": B, "t": 12, "type": "foul", "x": 47},  # skipped
            {"team": A, "t": 13, "type": "freekick_short", "x": 60},
            {"team": A, "t": 15, "x": 62},  # beyond window (t_r+3=14), window observed
        ]
    )
    out = secured_reception(a, _line_x(5))
    assert _is_true(out.iloc[0])


def test_saved_shot_is_true_shot_is_reception():
    """Round-2 finding 1, literal 3-row shape: the shot IS the next same-team touch.
    The subsequent opponent keeper_save (a possession boundary) must not flip it."""
    a = _acts(
        [
            {"team": A, "t": 10, "x": 40},
            {"team": A, "t": 11, "type": "shot", "x": 90},  # reception == shot
            {"team": B, "t": 12, "type": "keeper_save", "x": 5},
        ]
    )
    out = secured_reception(a, _line_x(3))
    assert _is_true(out.iloc[0])


def test_saved_shot_is_true_touch_then_shot():
    """Round-2 finding 1, 4-row shape: reception touch, then the shot decides True."""
    a = _acts(
        [
            {"team": A, "t": 10, "x": 40},
            {"team": A, "t": 11, "type": "dribble", "x": 70},  # reception
            {"team": A, "t": 12, "type": "shot", "x": 90},
            {"team": B, "t": 13, "type": "keeper_save", "x": 5},
        ]
    )
    out = secured_reception(a, _line_x(4))
    assert _is_true(out.iloc[0])


def test_foul_then_penalty_shot_is_true():
    """Skip/shot composition: foul skipped, ensuing same-team shot_penalty decides."""
    a = _acts(
        [
            {"team": A, "t": 10, "x": 40},
            {"team": A, "t": 11, "x": 58},  # reception
            {"team": B, "t": 12, "type": "foul", "x": 47},  # skipped
            {"team": A, "t": 13, "type": "shot_penalty", "x": 94},
        ]
    )
    out = secured_reception(a, _line_x(4))
    assert _is_true(out.iloc[0])


def test_opponent_possession_boundary_is_false():
    a = _acts(
        [
            {"team": A, "t": 10, "x": 40},
            {"team": A, "t": 11, "x": 58},  # reception
            {"team": B, "t": 12, "x": 47},  # team change -> possession boundary
        ]
    )
    out = secured_reception(a, _line_x(3))
    assert _is_false(out.iloc[0])


def test_behind_line_same_team_return_is_false():
    """The bounce-pass case: same-team action starting behind line_x within the window."""
    a = _acts(
        [
            {"team": A, "t": 10, "x": 40},
            {"team": A, "t": 11, "x": 58},  # reception
            {"team": A, "t": 12, "x": 45},  # 45 < line_x 55 -> False
        ]
    )
    out = secured_reception(a, _line_x(3))
    assert _is_false(out.iloc[0])


def test_empty_window_opponent_restart_is_false():
    """Round-2 finding 3: boundary test extends to the FIRST non-skipped event."""
    a = _acts(
        [
            {"team": A, "t": 10, "x": 40},
            {"team": A, "t": 11, "x": 58},  # reception; ball then out of play
            {"team": B, "t": 20, "type": "throw_in", "x": 30},  # opponent restart
        ]
    )
    out = secured_reception(a, _line_x(3))
    assert _is_false(out.iloc[0])


def test_empty_window_late_foul_then_same_team_is_true():
    """Round-2 finding 3: the late foul is skipped; the ensuing same-team free kick
    is possession-implying and undecisive -> window observed -> True."""
    a = _acts(
        [
            {"team": A, "t": 10, "x": 40},
            {"team": A, "t": 11, "x": 58},  # reception
            {"team": B, "t": 20, "type": "foul", "x": 47},  # skipped
            {"team": A, "t": 21, "type": "freekick_short", "x": 60},
        ]
    )
    out = secured_reception(a, _line_x(4))
    assert _is_true(out.iloc[0])


def test_empty_window_line_x_test_does_not_extend():
    """Spec s3: only the shot/boundary tests extend beyond an empty window --
    a late same-team behind-line action is NOT a bounce-pass loss."""
    a = _acts(
        [
            {"team": A, "t": 10, "x": 40},
            {"team": A, "t": 11, "x": 58},  # reception
            {"team": A, "t": 20, "x": 45},  # behind line but beyond window
        ]
    )
    out = secured_reception(a, _line_x(3))
    assert _is_true(out.iloc[0])


def test_truncated_window_is_na():
    a = _acts(
        [
            {"team": A, "t": 10, "x": 40},
            {"team": A, "t": 11, "x": 58},  # reception
            {"team": A, "t": 12, "type": "dribble", "x": 60},  # then data ends
        ]
    )
    out = secured_reception(a, _line_x(3))
    assert pd.isna(out.iloc[0])


def test_possession_id_present_and_absent_agree():
    """Self-heal path (add_possessions) == precomputed possession_id path."""
    a = _acts(
        [
            {"team": A, "t": 10, "x": 40},
            {"team": A, "t": 11, "x": 58},
            {"team": B, "t": 12, "x": 47},
        ]
    )
    healed = secured_reception(a, _line_x(3))
    precomputed = secured_reception(add_possessions(a), _line_x(3))
    assert _is_false(healed.iloc[0])
    pd.testing.assert_series_equal(healed, precomputed)


def test_nan_team_row_inside_window_is_skipped():
    """Review blocker 5 (ADR-027): a GS null-actor row must be skipped, never a
    raw != 'opponent' that decides a false loss."""
    a = _acts(
        [
            {"team": A, "t": 10, "x": 40},
            {"team": A, "t": 11, "x": 58},  # reception
            {"team": np.nan, "t": 12, "type": "tackle", "x": 50},  # skipped (NaN team)
            {"team": A, "t": 13, "x": 60},
            {"team": A, "t": 15, "x": 62},  # beyond window, window observed
        ]
    )
    out = secured_reception(a, _line_x(5))
    assert _is_true(out.iloc[0])


def test_window_is_anchored_at_reception_not_pass():
    """Review blocker 4: 2 s flight -> decisive event 2.5 s after RECEPTION (4.5 s
    after the pass) is still inside the 3 s window because it starts at t_r."""
    a = _acts(
        [
            {"team": A, "t": 10, "x": 40},
            {"team": A, "t": 12, "x": 58},  # reception after 2 s flight
            {"team": A, "t": 14.5, "x": 45},  # behind line, t_r+2.5 -> inside window
        ]
    )
    out = secured_reception(a, _line_x(3))
    assert _is_false(out.iloc[0])


def test_reception_row_itself_is_not_a_scannable_window_event():
    """Review blocker 4: the receiving row's own start_x must never trigger the
    behind-line rule (a receiver collecting behind the line is not a bounce-pass)."""
    a = _acts(
        [
            {"team": A, "t": 10, "x": 40},
            {"team": A, "t": 12, "x": 50},  # reception BEHIND line_x 55 -- not scanned
            {"team": A, "t": 13, "x": 60},
            {"team": A, "t": 16, "x": 62},  # beyond window, window observed
        ]
    )
    out = secured_reception(a, _line_x(4))
    assert _is_true(out.iloc[0])


def test_non_rangeindex_inputs_resolve_identically():
    """Round-2 minor 2 (the blocker-1 bug class pinned at the secured seam): a
    pre-filtered/sliced caller must resolve identically and keep its index."""
    a = _acts(
        [
            {"team": A, "t": 10, "x": 40},
            {"team": A, "t": 11, "x": 58},
            {"team": A, "t": 12, "x": 45},  # behind line -> False
        ]
    )
    lx = _line_x(3)
    a.index = pd.Index([7, 8, 9])
    lx.index = pd.Index([7, 8, 9])
    out = secured_reception(a, lx)
    assert list(out.index) == [7, 8, 9]
    assert _is_false(out.loc[7])


def test_nan_line_x_rows_are_na():
    """packing_made == 0 / no-geometry rows: NaN line_x -> <NA> (never decided)."""
    a = _acts(
        [
            {"team": A, "t": 10, "x": 40},
            {"team": A, "t": 11, "x": 58},
            {"team": A, "t": 15, "x": 62},
        ]
    )
    lx = pd.Series(np.full(3, np.nan), dtype="float64")
    out = secured_reception(a, lx)
    assert out.isna().all()
    assert out.dtype == "boolean"


def test_na_receiver_rows_are_na():
    """Opponent-next rows never resolve a receiver -> <NA>."""
    a = _acts(
        [
            {"team": A, "t": 10, "x": 40},
            {"team": B, "t": 11, "x": 58},  # opponent touch -> no receiver for row 0
            {"team": B, "t": 15, "x": 62},
        ]
    )
    out = secured_reception(a, _line_x(3))
    assert out.isna().all()


def test_precomputed_receiver_pos_matches_internal():
    """add_packing passes its precomputed positions -- must equal the None path."""
    a = _acts(
        [
            {"team": A, "t": 10, "x": 40},
            {"team": A, "t": 11, "x": 58},
            {"team": B, "t": 12, "x": 47},
        ]
    )
    internal = secured_reception(a, _line_x(3))
    explicit = secured_reception(a, _line_x(3), _resolve_next_touch_positions(a))
    pd.testing.assert_series_equal(internal, explicit)


def test_length_mismatch_raises():
    a = _acts([{"team": A, "t": 10}, {"team": A, "t": 11}])
    with pytest.raises(ValueError, match="equal-length"):
        secured_reception(a, pd.Series([55.0]))


def test_foul_between_pass_and_reception_anchors_at_true_reception():
    """Execution-review D1: the foul row must not become the anchor -- the window
    anchors at the TRUE reception, whose own start_x is never tested."""
    a = _acts(
        [
            {"team": A, "t": 10, "x": 40},
            {"team": B, "t": 10.5, "type": "foul", "x": 47},  # advantage played; skipped
            {"team": A, "t": 11, "x": 58},  # true reception (t_r = 11)
            {"team": A, "t": 13, "x": 60},
            {"team": A, "t": 14.5, "x": 62},  # beyond t_r+3 -> window observed
        ]
    )
    out = secured_reception(a, _line_x(5))
    assert _is_true(out.iloc[0])


def test_time_tied_positionally_swapped_rows_resolve_identically():
    """Execution-review D4: the scan runs in action_id order, so a stream whose
    time-tied rows arrive positionally swapped resolves identically."""
    rows = [
        {"team": A, "t": 1, "x": 30, "id": 0},
        {"team": A, "t": 2, "x": 40, "id": 1},  # the packing pass (line_x 60)
        {"team": A, "t": 2, "x": 72, "id": 2},  # time-tied reception
        {"team": A, "t": 4, "x": 72, "id": 3},
        {"team": A, "t": 6, "x": 80, "id": 4},
    ]

    def build(order):
        sel = [rows[i] for i in order]
        df = pd.DataFrame(
            {
                "game_id": 1,
                "period_id": 1,
                "action_id": [r["id"] for r in sel],
                "time_seconds": [float(r["t"]) for r in sel],
                "team_id": [r["team"] for r in sel],
                "type_id": _T["pass"],
                "start_x": [float(r["x"]) for r in sel],
            }
        )
        lx = pd.Series(np.full(5, np.nan))
        lx.iloc[order.index(1)] = 60.0
        return df, lx

    canonical, lx_c = build([0, 1, 2, 3, 4])
    swapped, lx_s = build([0, 2, 1, 3, 4])  # tie rows positionally swapped
    out_c = secured_reception(canonical, lx_c)
    out_s = secured_reception(swapped, lx_s)
    assert _is_true(out_c.iloc[1])
    assert _is_true(out_s.iloc[2])  # same physical pass, same label


def test_na_possession_never_decides_boundary():
    """Execution-review D6 (ADR-027 class): a caller-supplied possession_id with a
    missing value on the opponent row must not attest a boundary."""
    a = _acts(
        [
            {"team": A, "t": 10, "x": 40},
            {"team": A, "t": 11, "x": 58},  # reception
            {"team": B, "t": 12, "x": 47},  # opponent touch, possession UNATTESTED
        ]
    )
    a["possession_id"] = pd.array([1, 1, pd.NA], dtype="Int64")
    out = secured_reception(a, _line_x(3))
    assert pd.isna(out.iloc[0])  # undecidable, not a false loss
