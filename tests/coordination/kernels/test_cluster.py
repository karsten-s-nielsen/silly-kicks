"""TF-58 Task 12: cluster-phase synchrony (Richardson et al. 2012)."""

from __future__ import annotations

import numpy as np
import pytest

from silly_kicks.coordination._kernels import _cluster
from silly_kicks.coordination._kernels import _cluster_reference as REF
from silly_kicks.coordination._kernels._cluster import (
    ShiftRun,
    cluster_phase,
    pearson_rows,
    shifted_rho_group_means,
    window_cluster_stats,
)

_FORCE_NUMPY = "SILLY_KICKS_COORDINATION_FORCE_NUMPY"


def test_identical_phases_rho_one():
    t = np.linspace(0.0, 4 * np.pi, 60)
    z = np.exp(1j * t)[:, None] * np.ones((1, 5))
    valid = np.ones((60, 5), dtype=bool)
    _q, rel, usable = cluster_phase(z, valid, 3)
    st = window_cluster_stats(rel, valid, usable, 0, 60)
    assert np.allclose(st.rho_k, 1.0, atol=1e-12)
    assert abs(st.rho_group_mean - 1.0) < 1e-12
    assert usable.all()


def test_constant_per_player_lags_rho_group_one():
    t = np.linspace(0.0, 6 * np.pi, 300)
    phi = np.array([0.0, 0.5, 1.0, -0.3, 0.8])
    z = np.exp(1j * (t[:, None] + phi[None, :]))  # common omega, fixed per-player lags (Frank & Richardson)
    valid = np.ones((300, 5), dtype=bool)
    _q, rel, usable = cluster_phase(z, valid, 3)
    st = window_cluster_stats(rel, valid, usable, 0, 300)
    assert np.allclose(st.rho_k, 1.0, atol=1e-9)  # each player perfectly locked to the group
    assert abs(st.rho_group_mean - 1.0) < 1e-9  # constant lags -> perfect group synchrony
    # phi_bar recovers the lags relative to q (up to the group's mean offset)
    got = np.angle(np.exp(1j * ((st.phi_bar - st.phi_bar[0]) - (phi - phi[0]))))
    assert np.allclose(got, 0.0, atol=1e-9)


def test_uniform_random_phases_rho_small():
    rng = np.random.default_rng(0)
    theta = rng.uniform(-np.pi, np.pi, (200, 11))
    z = np.exp(1j * theta)
    valid = np.ones((200, 11), dtype=bool)
    _q, rel, usable = cluster_phase(z, valid, 3)
    st = window_cluster_stats(rel, valid, usable, 0, 200)
    assert st.rho_group_mean < 0.45


def test_min_players_both_sides():
    z = np.exp(1j * np.zeros((2, 5)))
    valid = np.zeros((2, 5), dtype=bool)
    valid[0, :3] = True  # exactly min_players
    valid[1, :2] = True  # one short
    _q, _rel, usable = cluster_phase(z, valid, 3)
    assert bool(usable[0]) is True
    assert bool(usable[1]) is False


def test_invalid_players_excluded_per_sample():
    z = np.exp(1j * np.array([[0.0, 0.0, 0.0, 0.0]]))  # 1 sample, 4 players
    valid = np.array([[True, True, True, False]])
    _q, rel, usable = cluster_phase(z, valid, 3)
    assert bool(usable[0]) is True
    assert rel[0, 3] == 0.0  # the invalid player contributes nothing
    assert np.isfinite(_q[0])


# --------------------------------------------------------------------------- batched surrogate null (spec 7.9)
def _per_draw_reference(z, valid, min_players, runs, start, end, n_draws):
    """The per-player-run null computed the direct way: per draw, roll each player's phasor within each of its runs
    (plan R1), then ``cluster_phase`` + ``window_cluster_stats`` -- the oracle for ``shifted_rho_group_means``."""
    out = []
    for d in range(n_draws):
        zk = z.copy()
        for r in runs:
            zk[r.lo : r.hi, r.player] = np.roll(z[r.lo : r.hi, r.player], int(r.shifts[d]))
        _q, rel, usable = cluster_phase(zk, valid, min_players)
        out.append(window_cluster_stats(rel, valid, usable, start, end).rho_group_mean)
    return np.array(out)


def _assert_bitwise_equal(got: np.ndarray, want: np.ndarray) -> None:
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    assert got.shape == want.shape
    same = (got.view(np.int64) == want.view(np.int64)) | (np.isnan(got) & np.isnan(want))
    assert same.all(), f"first mismatch at {int(np.flatnonzero(~same)[0])}: {got[~same][:3]} vs {want[~same][:3]}"


def _case(rng, k, n_draws, p_hole=0.0):
    """Players with 1-3 disjoint phase runs each (finite phasors inside, NaN outside -- as ``_cluster_inputs``
    builds them), and a window that cuts some runs, contains others and misses the rest.

    ``p_hole`` additionally marks finite samples inside runs invalid, so the kernel's masking is exercised on
    shifted-in values it must ignore.
    """
    n_rows = int(rng.integers(150, 420))
    theta = np.cumsum(rng.normal(0.0, 0.25, (n_rows, k)), axis=0) + rng.uniform(-np.pi, np.pi, k)
    z_full = np.exp(1j * theta)
    z = np.full((n_rows, k), complex(np.nan, np.nan))
    valid = np.zeros((n_rows, k), dtype=bool)
    runs = []
    for j in range(k):
        n_runs = int(rng.integers(1, 4))
        cuts = np.sort(rng.choice(np.arange(0, n_rows + 1), size=2 * n_runs, replace=False))
        for lo, hi in zip(cuts[0::2], cuts[1::2], strict=True):
            lo, hi = int(lo), int(hi)
            z[lo:hi, j] = z_full[lo:hi, j]
            valid[lo:hi, j] = True
            runs.append(ShiftRun(j, lo, hi, rng.integers(0, hi - lo, size=n_draws)))
    if p_hole:
        valid &= rng.random((n_rows, k)) >= p_hole
    start = int(rng.integers(0, n_rows // 3))
    end = int(rng.integers(2 * n_rows // 3, n_rows + 1))
    return z, valid, runs, start, end


@pytest.mark.parametrize("force_numpy", [False, True])
@pytest.mark.parametrize("k", [1, 3, 4, 7, 8, 11, 16])
@pytest.mark.parametrize("p_hole", [0.0, 0.2])
def test_shifted_rho_group_means_is_byte_identical_to_the_per_draw_loop(monkeypatch, k, p_hole, force_numpy):
    # Estimator identity (spec 7.9) on BOTH backends: every draw is exactly ``cluster_phase`` + ``window_cluster_stats``
    # on the shifted period -- the observed estimator -- bit for bit.
    if force_numpy:
        monkeypatch.setenv(_FORCE_NUMPY, "1")
    rng = np.random.default_rng(1000 * k + int(p_hole * 100))
    n_finite = 0
    for n_draws in (1, 7, 23):
        z, valid, runs, start, end = _case(rng, k, n_draws, p_hole)
        for min_players in sorted({1, min(3, k), k}):
            usable = cluster_phase(z, valid, min_players)[2]
            want = _per_draw_reference(z, valid, min_players, runs, start, end, n_draws)
            got = shifted_rho_group_means(z, valid, usable, start, end, runs, n_draws)
            _assert_bitwise_equal(got, want)
            n_finite += int(np.isfinite(want).sum())
    assert n_finite > 0  # the comparison is not NaN == NaN throughout


def test_shifted_rho_group_means_with_precomputed_parts_is_byte_identical():
    # A caller scoring many windows of one z hands the numba null its real/imag split once (`parts=`): same values.
    rng = np.random.default_rng(77)
    z, valid, runs, start, end = _case(rng, 11, 23, 0.2)
    usable = cluster_phase(z, valid, 3)[2]
    want = shifted_rho_group_means(z, valid, usable, start, end, runs, 23)
    got = shifted_rho_group_means(z, valid, usable, start, end, runs, 23, parts=_cluster.phasor_parts(z))
    _assert_bitwise_equal(got, want)
    assert np.isfinite(want).any()
    with pytest.raises(ValueError, match="parts have shapes"):
        shifted_rho_group_means(z, valid, usable, start, end, runs, 23, parts=_cluster.phasor_parts(z[:-1]))


def test_shifted_rho_group_means_keeps_every_shifted_value_finite():
    # Runs hold only finite phasors, so a shift can never carry a NaN into a valid slot (the team-segment design
    # did, NaN-ing whole draws): a substitution-style roster yields a finite null for every draw.
    rng = np.random.default_rng(11)
    theta = np.cumsum(rng.normal(0.0, 0.25, (300, 8)), axis=0)
    z = np.exp(1j * theta)
    valid = np.ones((300, 8), dtype=bool)
    valid[150:, 2] = False  # player 2 leaves...
    valid[:150, 7] = False  # ...player 7 joins
    z[~valid] = complex(np.nan, np.nan)
    runs = [ShiftRun(j, 0, 300, rng.integers(0, 300, size=9)) for j in (0, 1, 3, 4, 5, 6)]
    runs += [ShiftRun(2, 0, 150, rng.integers(0, 150, size=9)), ShiftRun(7, 150, 300, rng.integers(0, 150, size=9))]
    usable = cluster_phase(z, valid, 3)[2]
    got = shifted_rho_group_means(z, valid, usable, 40, 260, runs, 9)
    _assert_bitwise_equal(got, _per_draw_reference(z, valid, 3, runs, 40, 260, 9))
    assert np.isfinite(got).all()


def test_shifted_rho_group_means_is_chunk_invariant(monkeypatch):
    # Every draw-chunk size of the numpy reference gives the same bits (one draw per chunk, a ragged last chunk, all
    # draws at once).
    monkeypatch.setenv(_FORCE_NUMPY, "1")
    rng = np.random.default_rng(7)
    k = 11
    z = np.exp(1j * np.cumsum(rng.normal(0.0, 0.25, (300, k)), axis=0))
    valid = np.ones((300, k), dtype=bool)
    valid[np.arange(20, 280, 9), :] = False  # rows with no valid player: not usable -> the usable-row gather
    runs = [ShiftRun(j, 0, 300, rng.integers(0, 300, size=23)) for j in range(k)]
    usable = cluster_phase(z, valid, 3)[2]
    assert usable[20:280].any()
    assert not usable[20:280].all()
    want = _per_draw_reference(z, valid, 3, runs, 20, 280, 23)
    assert np.isfinite(want).all()
    for chunk in (1, 5 * 260 * k, 1 << 30):
        monkeypatch.setattr(_cluster, "SURROGATE_CHUNK_ELEMENTS", chunk)
        _assert_bitwise_equal(shifted_rho_group_means(z, valid, usable, 20, 280, runs, 23), want)


def test_shifted_rho_group_means_moves_the_null():
    # Non-vacuity: players riding one common (random-walk) phase with fixed lags are perfectly synchronised;
    # shifting each player independently decorrelates them, so the null sits well below the observation.
    rng = np.random.default_rng(3)
    common = np.cumsum(rng.normal(0.0, 0.3, 600))
    z = np.exp(1j * (common[:, None] + np.linspace(0.0, 1.0, 6)[None, :]))
    valid = np.ones(z.shape, dtype=bool)
    _q, rel, usable = cluster_phase(z, valid, 3)
    obs = window_cluster_stats(rel, valid, usable, 100, 500).rho_group_mean
    runs = [ShiftRun(j, 0, 600, rng.integers(50, 550, size=19)) for j in range(6)]
    null = shifted_rho_group_means(z, valid, usable, 100, 500, runs, 19)
    assert obs > 0.99
    assert float(null.max()) < obs - 0.2


def test_shifted_rho_group_means_no_usable_rows_is_nan():
    z = np.exp(1j * np.zeros((20, 4)))
    valid = np.zeros((20, 4), dtype=bool)
    valid[:, :2] = True  # 2 < min_players everywhere
    usable = cluster_phase(z, valid, 3)[2]
    runs = [ShiftRun(j, 0, 20, np.zeros(5, dtype=np.int64)) for j in range(2)]
    got = shifted_rho_group_means(z, valid, usable, 0, 20, runs, 5)
    assert got.shape == (5,)
    assert np.isnan(got).all()


@pytest.mark.parametrize(
    ("runs", "match"),
    [
        ([ShiftRun(0, 0, 20, np.zeros(4, dtype=np.int64))], "shape"),  # wrong draw count
        ([ShiftRun(0, 0, 20, np.full(5, -1, dtype=np.int64))], r"\[0, 20\)"),  # negative shift
        ([ShiftRun(0, 0, 20, np.full(5, 20, dtype=np.int64))], r"\[0, 20\)"),  # shift == run length
        ([ShiftRun(4, 0, 20, np.zeros(5, dtype=np.int64))], "player"),  # no such player column
        ([ShiftRun(0, 5, 5, np.zeros(5, dtype=np.int64))], "empty"),  # empty run
        (  # two runs of one player overlap: which shift owns the shared rows would be ambiguous
            [ShiftRun(1, 0, 12, np.zeros(5, dtype=np.int64)), ShiftRun(1, 10, 20, np.zeros(5, dtype=np.int64))],
            "overlap",
        ),
    ],
)
def test_shifted_rho_group_means_rejects_malformed_runs(runs, match):
    z = np.exp(1j * np.zeros((20, 4)))
    valid = np.ones((20, 4), dtype=bool)
    usable = cluster_phase(z, valid, 3)[2]
    with pytest.raises(ValueError, match=match):
        shifted_rho_group_means(z, valid, usable, 0, 20, runs, 5)


# --------------------------------------------------------------------------- D4: the redefined reference arithmetic
# ADR-111 D4: explicit real arithmetic for the complex products, sequential sums in a fixed documented order, unit
# phasors as ``s / sqrt(re^2 + im^2)`` (``1 + 0j`` where ``|s| == 0``) instead of ``exp(1j * angle(s))``. The numba
# kernel and the numpy reference then agree bit for bit, and both sit within 1e-12 of the as-built arithmetic (kept as
# ``_cluster_reference``, the parity oracle and the no-flip gate's reference leg); the measured max is reported.
@pytest.mark.parametrize("k", [1, 3, 4, 8, 11, 16])
@pytest.mark.parametrize("p_hole", [0.0, 0.2])
def test_cluster_null_backends_agree_bit_for_bit(monkeypatch, k, p_hole):
    pytest.importorskip("numba")
    rng = np.random.default_rng(500 + 10 * k + int(p_hole * 10))
    n_finite = 0
    for n_draws in (1, 9, 23):
        z, valid, runs, start, end = _case(rng, k, n_draws, p_hole)
        usable = cluster_phase(z, valid, min(3, k))[2]
        got_nb = shifted_rho_group_means(z, valid, usable, start, end, runs, n_draws)
        monkeypatch.setenv(_FORCE_NUMPY, "1")
        got_np = shifted_rho_group_means(z, valid, usable, start, end, runs, n_draws)
        monkeypatch.delenv(_FORCE_NUMPY)
        _assert_bitwise_equal(got_nb, got_np)
        n_finite += int(np.isfinite(got_np).sum())
    assert n_finite > 0


@pytest.mark.parametrize("k", [1, 3, 8, 11, 16])
def test_cluster_null_matches_the_as_built_arithmetic(k):
    rng = np.random.default_rng(700 + k)
    worst, n_cmp = 0.0, 0
    for n_draws in (7, 23):
        for p_hole in (0.0, 0.2):
            z, valid, runs, start, end = _case(rng, k, n_draws, p_hole)
            for min_players in sorted({1, min(3, k), k}):
                usable = cluster_phase(z, valid, min_players)[2]
                np.testing.assert_array_equal(usable, REF.cluster_phase(z, valid, min_players)[2])
                got = shifted_rho_group_means(z, valid, usable, start, end, runs, n_draws)
                want = REF.shifted_rho_group_means(z, valid, usable, start, end, runs, n_draws)
                np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
                fin = np.isfinite(want)
                if fin.any():
                    worst = max(worst, float(np.max(np.abs(got[fin] - want[fin]))))
                    n_cmp += int(fin.sum())
    print(f"cluster null, k={k}: max |D4 - as-built| = {worst:.3e} over {n_cmp} draws")
    assert n_cmp > 0
    assert worst <= 1e-12


@pytest.mark.parametrize("force_numpy", [False, True])
def test_cluster_null_at_zero_shift_is_the_observed_estimator(monkeypatch, force_numpy):
    # Every run shifted by 0 makes each draw the observation itself -- bit for bit on both backends, so an observation
    # and a surrogate that coincide in exact arithmetic coincide in floating point too (percentile ties stay ties).
    if force_numpy:
        monkeypatch.setenv(_FORCE_NUMPY, "1")
    rng = np.random.default_rng(31)
    z, valid, runs, start, end = _case(rng, 9, 5, 0.1)
    zero_runs = [ShiftRun(r.player, r.lo, r.hi, np.zeros(5, dtype=np.int64)) for r in runs]
    _q, rel, usable = cluster_phase(z, valid, 3)
    obs = window_cluster_stats(rel, valid, usable, start, end).rho_group_mean
    assert np.isfinite(obs)
    _assert_bitwise_equal(shifted_rho_group_means(z, valid, usable, start, end, zero_runs, 5), np.full(5, obs))


def test_zero_resultant_player_phasor_is_one_not_nan():
    # Player 0's relative phasors cancel EXACTLY over the window (+1 then -1): |sum| == 0. Its unit phasor is 1+0j --
    # the as-built exp(1j * angle(0)) -- never 0/0 = NaN, so the rows keep their synchrony values.
    z = np.array([[1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]], dtype=np.complex128)
    valid = np.ones((2, 3), dtype=bool)
    _q, rel, usable = cluster_phase(z, valid, 3)
    st = window_cluster_stats(rel, valid, usable, 0, 2)
    ref = REF.window_cluster_stats(REF.cluster_phase(z, valid, 3)[1], valid, usable, 0, 2)
    assert float(np.sum(rel[:, 0]).real) == 0.0  # the exact cancellation the case is built on
    assert np.isfinite(st.rho_group_i).all()
    _assert_bitwise_equal(st.rho_group_i, ref.rho_group_i)
    assert st.rho_group_mean == ref.rho_group_mean
    runs = [ShiftRun(j, 0, 2, np.zeros(3, dtype=np.int64)) for j in range(3)]
    _assert_bitwise_equal(shifted_rho_group_means(z, valid, usable, 0, 2, runs, 3), np.full(3, st.rho_group_mean))


def test_observed_cluster_statistics_match_the_as_built_arithmetic():
    rng = np.random.default_rng(41)
    worst: dict[str, float] = {}
    for trial in range(40):
        k = int(rng.integers(1, 14))
        n = int(rng.integers(30, 400))
        z = np.exp(1j * np.cumsum(rng.normal(0.0, 0.3, (n, k)), axis=0))
        valid = rng.random((n, k)) >= 0.15 * (trial % 3)
        z[~valid] = complex(np.nan, np.nan)
        mp = int(rng.integers(1, k + 1))
        _q, rel, usable = cluster_phase(z, valid, mp)
        _qr, rel_r, usable_r = REF.cluster_phase(z, valid, mp)
        np.testing.assert_array_equal(usable, usable_r)
        np.testing.assert_array_equal(np.isnan(rel), np.isnan(rel_r))
        s = int(rng.integers(0, n // 2))
        e = int(rng.integers(s + 1, n + 1))
        st = window_cluster_stats(rel, valid, usable, s, e)
        sr = REF.window_cluster_stats(rel_r, valid, usable_r, s, e)
        pairs = {
            "rel": (rel, rel_r),
            "phi_bar": (np.exp(1j * st.phi_bar), np.exp(1j * sr.phi_bar)),  # compare on the circle
            "rho_k": (st.rho_k, sr.rho_k),
            "rho_group_i": (st.rho_group_i, sr.rho_group_i),
            "rho_group_mean": (np.array([st.rho_group_mean]), np.array([sr.rho_group_mean])),
            "rho_group_sd": (np.array([st.rho_group_sd]), np.array([sr.rho_group_sd])),
        }
        for name, (have, want) in pairs.items():
            np.testing.assert_array_equal(np.isnan(have), np.isnan(want), err_msg=name)
            fin = ~np.isnan(want)
            if fin.any():
                worst[name] = max(worst.get(name, 0.0), float(np.max(np.abs(have[fin] - want[fin]))))
        assert st.n_players_mean == sr.n_players_mean or (np.isnan(st.n_players_mean) and np.isnan(sr.n_players_mean))
    print("observed cluster statistics, max |D4 - as-built|:", {k: f"{v:.3e}" for k, v in worst.items()})
    assert worst and max(worst.values()) <= 1e-12


# --------------------------------------------------------------------------- team-sync Pearson (D4)
def _pearson_masked(a, b, mask):
    m = mask & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1])


@pytest.mark.parametrize("force_numpy", [False, True])
def test_pearson_rows_matches_corrcoef(monkeypatch, force_numpy):
    if force_numpy:
        monkeypatch.setenv(_FORCE_NUMPY, "1")
    rng = np.random.default_rng(51)
    w = 300
    a = 0.6 + 0.2 * np.sin(np.linspace(0.0, 9.0, w)) + rng.normal(0.0, 0.05, w)
    b = np.stack([0.5 + 0.2 * np.sin(np.linspace(s, s + 9.0, w)) + rng.normal(0.0, 0.05, w) for s in range(9)])
    b[2, 40:90] = np.nan  # a draw whose shifted B carries unobserved rows
    mask = rng.random(w) > 0.1
    got = pearson_rows(a, b, mask, 4)
    want = np.array([_pearson_masked(a, row, mask) for row in b])
    dev = float(np.max(np.abs(got - want)))
    print(f"two-pass Pearson vs np.corrcoef: max |dr| = {dev:.3e}")
    assert dev <= 1e-12
    assert np.abs(got).max() > 0.5  # non-vacuous


def test_pearson_rows_backends_agree_bit_for_bit(monkeypatch):
    pytest.importorskip("numba")
    rng = np.random.default_rng(52)
    a = rng.normal(0.0, 1.0, 257)
    b = rng.normal(0.0, 1.0, (11, 257))
    b[3, ::7] = np.nan
    mask = rng.random(257) > 0.2
    got_nb = pearson_rows(a, b, mask, 4)
    monkeypatch.setenv(_FORCE_NUMPY, "1")
    _assert_bitwise_equal(got_nb, pearson_rows(a, b, mask, 4))


def test_pearson_rows_undefined_cases_are_nan():
    a = np.arange(10, dtype=np.float64)
    b = np.stack([np.arange(10, dtype=np.float64), np.full(10, 3.0), np.arange(10, dtype=np.float64)])
    mask = np.ones(10, dtype=bool)
    b[2, 3:] = np.nan  # only 3 finite rows left: below min_n
    got = pearson_rows(a, b, mask, 4)
    assert got[0] == 1.0  # a perfectly correlated row: exactly 1 (and never outside [-1, 1])
    assert np.isnan(got[1])  # zero variance
    assert np.isnan(got[2])  # too few rows
