import numpy as np
import pandas as pd

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.xtgk._retention_labels import retains

PASS = spadlconfig.actiontype_id["pass"]
SHOT = spadlconfig.actiontype_id["shot"]
GOALKICK = spadlconfig.actiontype_id["goalkick"]
FOUL = spadlconfig.actiontype_id["foul"]
FREEKICK = spadlconfig.actiontype_id["freekick_short"]
TACKLE = spadlconfig.actiontype_id["tackle"]
NON_ACTION = spadlconfig.actiontype_id["non_action"]
SUCCESS = spadlconfig.result_id["success"]
FAIL = spadlconfig.result_id["fail"]


def _row(aid, t, team, typ, res, pid):
    return dict(
        game_id=1,
        period_id=1,
        action_id=aid,
        time_seconds=t,
        team_id=team,
        player_id=1,
        type_id=typ,
        result_id=res,
        possession_id=pid,
        start_x=5.0,
        start_y=34.0,
        end_x=20.0,
        end_y=34.0,
    )


def test_retained_when_team_keeps_ball_through_window():
    # window 1.5s is fully covered by the 2s of data -> observed retention -> 1.0 (not NaN)
    a = pd.DataFrame(
        [
            _row(0, 0.0, 10, GOALKICK, SUCCESS, 0),
            _row(1, 1.0, 10, PASS, SUCCESS, 0),
            _row(2, 2.0, 10, PASS, SUCCESS, 0),
        ]
    )
    out = retains(a, window_seconds=1.5)
    assert out.iloc[0] == 1.0


def test_lost_when_opponent_takes_over_in_window():
    a = pd.DataFrame(
        [
            _row(0, 0.0, 10, GOALKICK, SUCCESS, 0),
            _row(1, 1.0, 20, PASS, SUCCESS, 1),  # opponent possession
        ]
    )
    out = retains(a, window_seconds=10.0)
    assert out.iloc[0] == 0.0


def test_retained_when_team_shoots_in_window():
    a = pd.DataFrame(
        [
            _row(0, 0.0, 10, GOALKICK, SUCCESS, 0),
            _row(1, 1.0, 10, SHOT, FAIL, 0),
        ]
    )
    out = retains(a, window_seconds=10.0)  # decisive shot -> 1.0 regardless of truncation
    assert out.iloc[0] == 1.0


def test_truncated_window_with_no_decisive_event_is_nan():
    # a lone goal-kick near a period end: the 10s window is truncated to 0s of observable data and
    # nothing decisive happens -> we did NOT observe retention -> NaN (excluded from training).
    a = pd.DataFrame([_row(0, 2699.0, 10, GOALKICK, SUCCESS, 0)])
    out = retains(a, window_seconds=10.0)
    assert np.isnan(out.iloc[0])


# --- PR-S117 hardening (packing-seam rules; probe-verified label NO-OP on the live rho
# --- cohorts 2026-07-17 -- the gold-mart possession ids never hit these preconditions,
# --- which is exactly why applying them required no retrain; ADR-039 relay item 1).


def test_foul_row_never_decides_a_loss():
    """The add_possessions-shaped bias (heuristic emits a boundary AT the foul row):
    winning a foul is not losing the ball -- the foul row must be skipped and the
    ensuing same-team free kick keeps the label at retained."""
    a = pd.DataFrame(
        [
            _row(0, 0.0, 10, GOALKICK, SUCCESS, 0),
            _row(1, 1.0, 20, FOUL, FAIL, 1),  # opponent fouls; heuristic boundary AT this row
            _row(2, 2.0, 10, FREEKICK, SUCCESS, 1),  # carve-out: same possession id as the foul
            _row(3, 6.0, 10, PASS, SUCCESS, 1),
        ]
    )
    out = retains(a, window_seconds=5.0)
    assert out.iloc[0] == 1.0


def test_nan_team_row_never_decides():
    """GS null-actor rows (ADR-027): NaN team must be skipped, never read as 'opponent'."""
    a = pd.DataFrame(
        [
            _row(0, 0.0, 10, GOALKICK, SUCCESS, 0),
            _row(1, 1.0, np.nan, TACKLE, FAIL, 1),  # null-actor duel row
            _row(2, 2.0, 10, PASS, SUCCESS, 1),
            _row(3, 6.0, 10, PASS, SUCCESS, 1),
        ]
    )
    out = retains(a, window_seconds=5.0)
    assert out.iloc[0] == 1.0


def test_nan_team_ANCHOR_is_undecidable_not_a_loss():
    """ADR-027 delta-review finding: a NaN-team ANCHOR (not just decider) has no
    knowable 'whose team retained' answer -- it must be NaN, never a decisive
    0.0/1.0. Pre-fix, `not _same(team[gj], NA)` was vacuously True, so the unknown
    anchor team satisfied the opponent-boundary prong and possession-diff decided a
    false loss."""
    a = pd.DataFrame(
        [
            _row(0, 0.0, 10, GOALKICK, SUCCESS, 0),
            _row(1, 1.0, np.nan, TACKLE, FAIL, 0),  # null-actor ANCHOR (unknown team)
            _row(2, 2.0, 20, PASS, SUCCESS, 1),  # attested opponent boundary in-window
            _row(3, 6.0, 20, PASS, SUCCESS, 1),
        ]
    )
    out = retains(a, window_seconds=5.0)
    assert np.isnan(out.iloc[1])  # anchor row 1 has an unknown team -> undecidable
    # and it must be undecidable regardless of what follows: a same-team-shaped shot
    # or a quiet window would otherwise flip the same unknowable row to NaN vs 1.0 vs 0.0.
    b = pd.DataFrame(
        [
            _row(0, 0.0, np.nan, TACKLE, FAIL, 0),  # lone null-actor anchor, quiet window
            _row(1, 1.0, 10, PASS, SUCCESS, 0),
            _row(2, 6.0, 10, PASS, SUCCESS, 0),
        ]
    )
    assert np.isnan(retains(b, window_seconds=5.0).iloc[0])


def test_non_action_row_never_decides():
    a = pd.DataFrame(
        [
            _row(0, 0.0, 10, GOALKICK, SUCCESS, 0),
            _row(1, 1.0, 20, NON_ACTION, FAIL, 1),  # not a possession-implying event
            _row(2, 2.0, 10, PASS, SUCCESS, 1),
            _row(3, 6.0, 10, PASS, SUCCESS, 1),
        ]
    )
    out = retains(a, window_seconds=5.0)
    assert out.iloc[0] == 1.0


def test_na_possession_never_decides_boundary():
    """An unattested possession id must not attest a boundary (the packing D6 class)."""
    a = pd.DataFrame(
        [
            _row(0, 0.0, 10, GOALKICK, SUCCESS, 0),
            _row(1, 1.0, 20, PASS, SUCCESS, 1),
            _row(2, 2.0, 10, PASS, SUCCESS, 0),
            _row(3, 6.0, 10, PASS, SUCCESS, 0),
        ]
    )
    a["possession_id"] = pd.array([0, pd.NA, 0, 0], dtype="Int64")
    out = retains(a, window_seconds=5.0)
    assert out.iloc[0] == 1.0  # the NA-possession opponent row is undecidable, not a loss


def test_time_tied_rows_label_order_insensitively():
    """PR-S117 (b): the scan runs in canonical (time_seconds, action_id) order, so a
    stream whose time-tied rows arrive positionally swapped labels identically. The
    discriminating tie is WITH THE ANCHOR: a tie-partner positionally after the
    anchor was scanned, positionally before it was silently excluded."""
    anchor = _row(2, 1.0, 10, GOALKICK, SUCCESS, 0)
    tied_opp = _row(3, 1.0, 20, PASS, SUCCESS, 1)  # decisive loss, tied with the anchor
    later = _row(4, 4.0, 10, PASS, SUCCESS, 0)
    order_a = pd.DataFrame([anchor, tied_opp, later])
    order_b = pd.DataFrame([tied_opp, anchor, later])  # same match, ties swapped
    out_a = retains(order_a, window_seconds=2.0)
    out_b = retains(order_b, window_seconds=2.0)
    # invariance across input orders...
    assert out_a.iloc[0] == out_b.iloc[1]
    # ...AND the ground-truth pin (symmetry alone is insufficient -- house discipline):
    # canonical order puts action_id 3 AFTER the anchor (2), so the loss IS scanned.
    assert out_a.iloc[0] == 0.0


def test_mart_shaped_stream_labels_unchanged():
    """The no-op leg (why no retrain was needed): mart possession ids stay continuous
    through foul rows, and a REAL attested opponent boundary still decides a loss."""
    a = pd.DataFrame(
        [
            _row(0, 0.0, 10, GOALKICK, SUCCESS, 0),
            _row(1, 1.0, 20, FOUL, FAIL, 0),  # mart shape: possession CONTINUOUS through the foul
            _row(2, 2.0, 20, PASS, SUCCESS, 1),  # real opponent possession -> loss
        ]
    )
    out = retains(a, window_seconds=5.0)
    assert out.iloc[0] == 0.0
