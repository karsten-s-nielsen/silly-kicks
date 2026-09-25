"""ADR-105 Task 1: the vectorized cross-frame spearman kernel is BYTE-IDENTICAL to the per-frame
`compute_spearman` loop (np.array_equal, max |Δ| exactly 0), including ragged/padded/unsorted frames.

The parity is the oracle: these tests pass on the per-frame-loop baseline of `compute_pitch_control_batch`
AND must stay green after `compute_spearman_batch` replaces the loop. Byte-identity holds because real
per-frame valid-player counts stay below numpy's pairwise-summation threshold (128) -> the influence sum is
SEQUENTIAL, so a masked-to-0.0 padding row is an exact no-op (VKS-SPEC-02 / P<128 precondition)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.tracking.pitch_control import compute_pitch_control, compute_pitch_control_batch


def _frame(fid, players, ball=(50.0, 34.0)):
    rows: list[dict] = [
        dict(
            game_id=1,
            period_id=1,
            frame_id=fid,
            is_ball=False,
            is_goalkeeper=g,
            team_id=t,
            player_id=pid,
            x=x,
            y=y,
            vx=vx,
            vy=vy,
        )
        for (pid, t, g, x, y, vx, vy) in players
    ]
    rows.append(
        dict(
            game_id=1,
            period_id=1,
            frame_id=fid,
            is_ball=True,
            is_goalkeeper=False,
            team_id=np.nan,
            player_id=np.nan,
            x=ball[0],
            y=ball[1],
            vx=0.0,
            vy=0.0,
        )
    )
    return pd.DataFrame(rows)


def _ref(frames, fk, team, method, dec):
    return compute_pitch_control(frames[frames["frame_id"] == fk[2]], team, method=method, decompose=dec)


def test_batch_byte_identical_ragged_padded_unsorted():
    # Frame A: 5 players. Frame B: 3 players (padding vs A's max) in a DIFFERENT team/gk/position order.
    a = _frame(
        10,
        [
            (1, 1, True, 8, 34, 0, 0),
            (2, 1, False, 60, 20, 1, 0),
            (3, 2, True, 45, 40, 0, 1),
            (4, 2, False, 70, 34, 0, 0),
            (5, 1, False, 30, 50, -1, 0),
        ],
    )
    b = _frame(11, [(9, 2, False, 70, 34, 0, 0), (8, 1, True, 8, 34, 0, 0), (7, 2, True, 45, 40, 0, 1)])
    frames = pd.concat([a, b], ignore_index=True)
    reqs = [((1, 1, 10), 1, dec) for dec in (False, True)] + [((1, 1, 11), 1, dec) for dec in (False, True)]
    for method in ("spearman",):
        got = compute_pitch_control_batch(frames, reqs, method=method)
        assert len(got) == len(reqs)
        for (fk, team, dec), g in zip(reqs, got, strict=True):
            ref = _ref(frames, fk, team, method, dec)
            assert np.array_equal(g.surface, ref.surface), (method, fk, dec)
            if dec:
                assert g.per_player_influence is not None and ref.per_player_influence is not None
                assert g.player_ids is not None and ref.player_ids is not None
                assert np.array_equal(g.per_player_influence, ref.per_player_influence)
                assert np.array_equal(g.player_ids, ref.player_ids)


def test_batch_byte_identical_empty_and_one_sided_frames():
    # A frame with ONLY defenders (att empty -> 0.5 where both zero) + a NaN-position player (filtered).
    only_def = _frame(20, [(1, 2, True, 8, 34, 0, 0), (2, 2, False, 60, 34, 0, 0)])
    nan_player = _frame(
        21, [(1, 1, False, np.nan, np.nan, 0, 0), (2, 2, False, 60, 34, 0, 0), (3, 1, True, 8, 34, 0, 0)]
    )
    frames = pd.concat([only_def, nan_player], ignore_index=True)
    reqs = [((1, 1, 20), 1, True), ((1, 1, 21), 1, True)]
    got = compute_pitch_control_batch(frames, reqs, method="spearman")
    for (fk, team, dec), g in zip(reqs, got, strict=True):
        ref = _ref(frames, fk, team, "spearman", dec)
        assert np.array_equal(g.surface, ref.surface), fk


def test_batch_byte_identical_larger_but_subthreshold_playercount():
    # 40 players (< 128 pairwise threshold) -> sequential sum -> padded batch stays exact.
    rng = np.random.default_rng(7)
    players = [
        (
            i,
            1 if i % 2 == 0 else 2,
            i < 2,
            float(rng.uniform(0, 105)),
            float(rng.uniform(0, 68)),
            float(rng.uniform(-3, 3)),
            float(rng.uniform(-3, 3)),
        )
        for i in range(40)
    ]
    big = _frame(30, players)
    small = _frame(31, players[:6])
    frames = pd.concat([big, small], ignore_index=True)
    reqs = [((1, 1, 30), 1, False), ((1, 1, 31), 1, False)]
    got = compute_pitch_control_batch(frames, reqs, method="spearman")
    for (fk, team, dec), g in zip(reqs, got, strict=True):
        ref = _ref(frames, fk, team, "spearman", dec)
        assert np.array_equal(g.surface, ref.surface), fk


def test_row_order_alignment_is_load_bearing():
    """Non-vacuity (VKS-SPEC-02): the combine is order-sensitive, so a mis-aligned team mask MOVES the
    surface -- proving the byte-identity parity gate above is discriminating, not vacuous."""
    from silly_kicks.id_compat import ids_match
    from silly_kicks.tracking.pitch_control._grids import pitch_grid
    from silly_kicks.tracking.pitch_control._params import SpearmanParams
    from silly_kicks.tracking.pitch_control._spearman import (
        _extract_frame_players,
        _spearman_combine,
        compute_tti,
    )

    sp = SpearmanParams()
    gx, gy, targets = pitch_grid(sp.grid_cells_x, sp.grid_cells_y)
    frame = _frame(
        50,
        [
            (1, 1, True, 8, 34, 0, 0),
            (2, 1, False, 60, 20, 1, 0),
            (3, 2, True, 45, 40, 0, 1),
            (4, 2, False, 70, 34, 0, 0),
        ],
    )
    _extracted = _extract_frame_players(frame)
    assert _extracted is not None
    pos, vel, is_gk, pids, team_s = _extracted
    tti = compute_tti(pos, vel, targets, sp.reaction_time, sp.max_acceleration)
    is_att = ids_match(team_s, 1).to_numpy()
    good = _spearman_combine(tti, is_att, is_gk, pids, team_s.to_numpy(), None, sp, gx, gy, targets, 1, False)
    bad = _spearman_combine(tti, is_att[::-1], is_gk, pids, team_s.to_numpy(), None, sp, gx, gy, targets, 1, False)
    assert not np.array_equal(good.surface, bad.surface)  # a permutation bug WOULD be caught


def test_batch_calls_compute_tti_once_for_n_frames(monkeypatch):
    """The vectorization win: N frames -> ONE compute_tti call (all players concatenated), not N."""
    from silly_kicks.tracking.pitch_control import _spearman_batch
    from tests._perf_structural import call_counter

    frames = pd.concat(
        [_frame(fid, [(1, 1, True, 8, 34, 0, 0), (2, 2, False, 60, 34, 0, 0)]) for fid in range(10, 18)],
        ignore_index=True,
    )
    reqs = [((1, 1, fid), 1, False) for fid in range(10, 18)]
    calls = call_counter(monkeypatch, _spearman_batch, "compute_tti")
    out = compute_pitch_control_batch(frames, reqs, method="spearman")
    assert len(out) == 8
    assert calls["n"] == 1  # 8 frames, ONE compute_tti (vs 8 in the per-frame loop)
