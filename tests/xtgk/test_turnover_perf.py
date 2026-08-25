"""Item B / ADR-068: the turnover scan rewrite (numba-on-int-codes + pure-Python fallback) is
BYTE-IDENTICAL to the prior nested Python loop, kept here verbatim as `_ref_raw`."""

import numpy as np
import pandas as pd
import pytest

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.xtgk._turnover import (
    EmpiricalTurnoverValue,
    _opp_first_shot_scan,
    _opp_first_shot_scan_fast,
)
from tests._perf_structural import call_counter

_PASS = spadlconfig.actiontype_id["pass"]
_SHOT = spadlconfig.actiontype_id["shot"]


def _row(game, team, typ, poss, t, xg=0.0, turn=False):
    return dict(game_id=game, team_id=team, type_id=typ, possession_id=poss, time_seconds=t, xg=xg, _turnover=turn)


def _churny_corpus():
    rows = [
        # G1: single opponent possession, first opp shot credited
        _row(1, "A", _PASS, 1, 10.0, turn=True),
        _row(1, "A", _PASS, 1, 11.0),  # same possession -> skipped
        _row(1, "B", _PASS, 2, 12.0),
        _row(1, "B", _SHOT, 2, 14.0, xg=0.3),  # first opp shot -> 0.3
        # G2: TWO consecutive opponent possessions before the shot (the naive-rewrite trap)
        _row(2, "A", _PASS, 1, 10.0, turn=True),
        _row(2, "B", _PASS, 2, 11.0),
        _row(2, "B", _PASS, 3, 12.0),  # different opp possession, still opponent -> keep scanning
        _row(2, "B", _SHOT, 3, 13.0, xg=0.4),  # -> 0.4
        # G3: ball returns to the loser BEFORE any shot -> no credit
        _row(3, "A", _PASS, 1, 10.0, turn=True),
        _row(3, "B", _PASS, 2, 11.0),
        _row(3, "A", _PASS, 3, 12.0),  # ball back to loser -> break
        _row(3, "B", _SHOT, 4, 13.0, xg=0.5),  # after ball-back -> NOT credited
        # G4: shot BEYOND a finite window (window-dependent)
        _row(4, "A", _PASS, 1, 10.0, turn=True),
        _row(4, "B", _PASS, 2, 15.0),
        _row(4, "B", _SHOT, 2, 25.0, xg=0.6),  # dt=15 > 10 -> excluded at window=10, credited at None
        # G5: within-window shot (credited either way)
        _row(5, "A", _PASS, 1, 10.0, turn=True),
        _row(5, "B", _SHOT, 2, 18.0, xg=0.7),  # dt=8 <= 10 -> 0.7 both windows
    ]
    return pd.DataFrame(rows)


def _ref_raw(a, xg_column, window_seconds):
    """The pre-ADR-068 nested Python loop, VERBATIM on raw values -- the byte-identity ground truth."""
    out = np.zeros(len(a), dtype=float)
    team = a["team_id"].to_numpy()
    typ = a["type_id"].to_numpy()
    xg = a[xg_column].fillna(0.0).to_numpy(dtype=float)
    game = a["game_id"].to_numpy()
    poss = a["possession_id"].to_numpy()
    t = a["time_seconds"].to_numpy(dtype=float)
    turn = a["_turnover"].to_numpy()
    n = len(a)
    for i in range(n):
        if not turn[i]:
            continue
        for j in range(i + 1, n):
            if game[j] != game[i] or (window_seconds is not None and (t[j] - t[i]) > window_seconds):
                break
            if poss[j] == poss[i]:
                continue
            if team[j] == team[i]:
                break
            if typ[j] == _SHOT:
                out[i] = xg[j]
                break
    return out


@pytest.mark.parametrize("window", [None, 10.0])
def test_rewrite_is_byte_identical_to_reference_loop(window):
    a = _churny_corpus()
    tv = EmpiricalTurnoverValue(window_seconds=window)
    got = tv._opp_first_shot_after_turnover(a, "xg", window_seconds=window)
    exp = _ref_raw(a, "xg", window)
    np.testing.assert_array_equal(got, exp)


def test_expected_credits_are_what_we_think():
    # Non-vacuity: pin the actual values so the oracle isn't "both wrong the same way".
    a = _churny_corpus()
    tv = EmpiricalTurnoverValue()
    none_out = tv._opp_first_shot_after_turnover(a, "xg", window_seconds=None)
    win_out = tv._opp_first_shot_after_turnover(a, "xg", window_seconds=10.0)
    turn_pos = a.index[a["_turnover"]].tolist()
    assert [none_out[i] for i in turn_pos] == [0.3, 0.4, 0.0, 0.6, 0.7]  # G4 shot credited at window=None
    assert [win_out[i] for i in turn_pos] == [0.3, 0.4, 0.0, 0.0, 0.7]  # G4 shot excluded at window=10


def test_scan_kernel_invoked_once_not_per_turnover(monkeypatch):
    # Structural guard (ADR-068 "parity PLUS structural call-count guard"): the O(n*k) scan is ONE
    # kernel call per fit, not the old O(n^2) per-turnover rescan. _churny_corpus has 5 turnovers; a
    # per-turnover rescan would invoke the kernel 5 times.
    import silly_kicks.xtgk._turnover as tv_mod

    calls = call_counter(monkeypatch, tv_mod, "_opp_first_shot_scan_fast")
    EmpiricalTurnoverValue()._opp_first_shot_after_turnover(_churny_corpus(), "xg", window_seconds=None)
    assert calls["n"] == 1


def test_numba_kernel_is_active_when_installed():
    # CLAUDE.md: [numba] is bundled in [test] so CI exercises the @njit path. Guard against a SILENT
    # permanent fallback to pure-Python (which passes every parity test yet loses the ~100x speedup):
    # when numba is importable, the COMPILED kernel must be the one that runs.
    import importlib.util

    from silly_kicks.xtgk._turnover import _NUMBA_TURNOVER

    if importlib.util.find_spec("numba") is None:
        pytest.skip("numba not installed -- the pure-Python fallback is the correct path here")
    assert _NUMBA_TURNOVER is True


def test_numba_and_python_kernels_agree():
    # The @njit kernel (if compiled) reproduces the pure-Python kernel exactly.
    from silly_kicks.xtgk._turnover import _equality_codes

    a = _churny_corpus()
    args = (
        a["_turnover"].to_numpy(dtype=bool),
        _equality_codes(a["game_id"]),
        _equality_codes(a["possession_id"]),
        _equality_codes(a["team_id"]),
        a["type_id"].to_numpy(dtype=np.int64),
        a["xg"].to_numpy(dtype=float),
        a["time_seconds"].to_numpy(dtype=float),
        _SHOT,
        np.inf,
    )
    np.testing.assert_array_equal(_opp_first_shot_scan_fast(*args), _opp_first_shot_scan(*args))


def test_nan_team_does_not_crash_and_never_matches():
    # GS null-actor team_id is NA; the raw `==` loop raised on it, the code path treats each NA as a
    # DISTINCT team (never ball-back), so it runs and the turnover still finds its opponent shot.
    a = pd.DataFrame(
        [
            _row(1, "A", _PASS, 1, 10.0, turn=True),
            _row(1, pd.NA, _PASS, 2, 11.0),  # null-actor row: distinct team, not the loser
            _row(1, "B", _SHOT, 3, 12.0, xg=0.9),
        ]
    )
    a["team_id"] = a["team_id"].astype("object")
    out = EmpiricalTurnoverValue()._opp_first_shot_after_turnover(a, "xg", window_seconds=None)
    assert out[0] == pytest.approx(0.9)  # NA row didn't false-trigger ball-back; opp shot credited
