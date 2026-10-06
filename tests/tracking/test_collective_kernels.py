"""Ground-truth tests for the collective-variable array kernels (TF-58 Task 2, spec 9.1/9.2)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import ConvexHull, QhullError

from silly_kicks.tracking._collective import (
    _CCW_ERRBOUND_A,
    BACK_LINE_VARIABLES,
    _collinear_exact,
    _exactly_collinear,
    back_line_batch,
    collective_from_positions,
    compact_rows,
    hull_area_batch,
    pack_groups,
)
from tests.tracking._legacy_collective_oracle import _legacy_select_n

_DL_VARIANTS = [(3, 5), (4, 5), (5, 5), ("adaptive", 3), ("adaptive", 4), ("adaptive", 5)]


def _qhull(p: np.ndarray) -> tuple[float, bool]:
    try:
        return float(ConvexHull(p).volume), False
    except QhullError:
        return 0.0, True


# --------------------------------------------------------------------------- hull area


def test_hull_area_matches_qhull_random():
    rng = np.random.default_rng(58)
    for n in range(3, 15):
        pts = rng.uniform((0.0, 0.0), (105.0, 68.0), size=(400, n, 2))
        got = hull_area_batch(pts, np.full(400, n, dtype=np.int64))
        want = np.array([_qhull(p)[0] for p in pts])
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=0)


def test_hull_area_exact_collinear_and_coincident_is_zero():
    line = np.array([[[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [5.0, 10.0]]])
    same = np.array([[[3.0, 3.0]] * 4])
    assert hull_area_batch(line, np.array([4]))[0] == 0.0
    assert hull_area_batch(same, np.array([4]))[0] == 0.0


def test_hull_area_precision_flat_within_abs_bound():  # owner ruling R3
    # Precision-flat points have a true area of ~0. The kernel and Qhull each round to a different
    # sub-1e-9 value (whether Qhull raises or returns a picometre^2 area), so R3's ABSOLUTE bound is the
    # right assertion here -- a relative bound is meaningless at that magnitude (that regime is why R3
    # exists). Real, non-degenerate hulls are covered relatively by test_hull_area_matches_qhull_random.
    rng = np.random.default_rng(7)
    raised = 0
    for _ in range(200):
        x = np.sort(rng.uniform(0, 100, 6))
        pts = np.column_stack([x, 0.3 * x + 1.0 + rng.normal(0, 1e-13, 6)])[None]
        _, flat = _qhull(pts[0])
        got = hull_area_batch(pts, np.array([6]))[0]
        raised += int(flat)
        assert abs(got) <= 1e-9
    assert raised > 0, "no precision-flat case reached Qhull's error path -- the R3 branch is untested"


def test_hull_area_adversarial():
    cases = [
        np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [10.0, 10.0]]),  # duplicated vertex
        np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 0.0], [5.0, 8.0]]),  # a point on an edge
        np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 3.0]]),  # exactly 3
        np.array(  # 11 points, 7 interior
            [
                [0.0, 0.0],
                [20.0, 0.0],
                [20.0, 20.0],
                [0.0, 20.0],
                [5.0, 5.0],
                [6.0, 7.0],
                [8.0, 9.0],
                [10.0, 10.0],
                [12.0, 8.0],
                [9.0, 12.0],
                [7.0, 11.0],
            ]
        ),
        np.array(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [1.0, 1.0]]
        ),  # integer grid, collinear subset
    ]
    for pts in cases:
        want, flat = _qhull(pts)
        got = hull_area_batch(pts[None], np.array([len(pts)]))[0]
        if flat:
            assert abs(got) <= 1e-9
        else:
            assert got == pytest.approx(want, rel=1e-9, abs=0.0)


def test_hull_area_fewer_than_three_is_nan():
    pos = np.full((3, 5, 2), np.nan)
    pos[0, 0] = [1.0, 2.0]
    pos[1, :2] = [[1.0, 2.0], [3.0, 4.0]]
    got = hull_area_batch(pos, np.array([1, 2, 0], dtype=np.int64))
    assert np.isnan(got).all()


# --------------------------------------------------------------------------- A2 exact predicate


def test_exact_predicate_rejects_float_false_positive():
    # 3 points whose NAIVE float orientation determinant is exactly 0.0, but which are not collinear.
    x0, y0 = 0.0, 0.0
    x1, y1 = 1.0 + 2.0**-52, 1.0
    x2, y2 = 1.0 + 2.0**-51, 1.0 + 2.0**-52
    det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    assert det == 0.0  # a `det == 0.0` test would wrongly call them collinear
    pts = np.array([[[x0, y0], [x1, y1], [x2, y2]]])
    assert not bool(_exactly_collinear(pts)[0])
    assert not _collinear_exact(pts[0])


def test_exact_predicate_accepts_exactly_collinear_dyadic_sets():
    m = np.arange(5, 12)
    x = m * 2.0**-10
    y = 3.0 * x + 7.0  # exactly representable; anchor is non-origin (x[0] = 5*2**-10)
    pts = np.column_stack([x, y])[None]
    assert bool(_exactly_collinear(pts)[0])
    assert _collinear_exact(pts[0]) is True
    coincident = np.array([[[3.5, 4.25]] * 6])
    assert bool(_exactly_collinear(coincident)[0])


def test_exact_predicate_agrees_with_rational_oracle():
    rng = np.random.default_rng(202)
    for _ in range(2000):
        n = int(rng.integers(3, 12))
        mode = rng.integers(0, 3)
        if mode == 0:  # random
            pts = rng.uniform(0, 100, (n, 2))
        elif mode == 1:  # near-collinear (+- 1 ulp jitter)
            x = np.sort(rng.uniform(0, 100, n))
            base = 0.7 * x + 2.0
            pts = np.column_stack([x, base + rng.choice([-1, 0, 1], n) * np.spacing(base)])
        else:  # exactly collinear dyadic
            xm = rng.integers(0, 500, n) * 2.0**-8
            pts = np.column_stack([xm, 2.0 * xm - 3.0])
        assert bool(_exactly_collinear(pts[None])[0]) == _collinear_exact(pts)


def test_exact_fallback_runs_only_on_unsettled_rows(monkeypatch):
    import silly_kicks.tracking._collective as c

    calls = {"n": 0}
    orig = c._collinear_exact

    def spy(pts):
        calls["n"] += 1
        return orig(pts)

    monkeypatch.setattr(c, "_collinear_exact", spy)
    rng = np.random.default_rng(1)
    well_spread = rng.uniform(0, 100, size=(1000, 6, 2))
    c._exactly_collinear(well_spread)
    assert calls["n"] == 0  # every row settled by the float filter

    calls["n"] = 0
    # Mix EXACTLY-collinear rows (det exactly 0 -> the float filter cannot certify non-zero -> unsettled ->
    # fallback runs) with well-spread random rows (settled). The fallback must run exactly on the unsettled rows.
    rows = []
    for i in range(50):
        if i < 30:
            xm = (np.arange(6) + i) * 2.0**-8
            rows.append(np.column_stack([xm, 2.0 * xm - 3.0]))  # exactly collinear
        else:
            rows.append(rng.uniform(0, 100, (6, 2)))  # well-spread
    mixed = np.stack(rows)
    c._exactly_collinear(mixed)
    rel = mixed - mixed[:, :1, :]
    far = np.argmax(np.hypot(rel[..., 0], rel[..., 1]), axis=1)
    v = rel[np.arange(mixed.shape[0]), far]
    dl = v[:, None, 0] * rel[..., 1]
    dr = v[:, None, 1] * rel[..., 0]
    cn = np.abs(dl - dr) > _CCW_ERRBOUND_A * (np.abs(dl) + np.abs(dr))
    unsettled = int((~cn.any(axis=1)).sum())
    assert calls["n"] == unsettled and unsettled > 0


# --------------------------------------------------------------------------- collective variables


def test_spread_identity_matches_naive_double_sum():
    rng = np.random.default_rng(9)
    for n in range(2, 15):
        pos = rng.uniform(0, 100, (200, n, 2))
        got = collective_from_positions(pos, np.full(200, n, dtype=np.int64))["spread"]
        naive = np.empty(200)
        for r in range(200):
            d = pos[r][:, None, :] - pos[r][None, :, :]
            sq = (d[..., 0] ** 2 + d[..., 1] ** 2)[np.triu_indices(n, 1)]
            naive[r] = np.sqrt(sq.sum())
        np.testing.assert_allclose(got, naive, rtol=1e-9, atol=0)


def test_collective_from_positions_bit_identical_to_per_row_numpy():
    rng = np.random.default_rng(11)
    g = 600
    ns = rng.integers(1, 15, g)
    pmax = int(ns.max())
    pos = np.full((g, pmax, 2), np.nan)
    for i, n in enumerate(ns):
        pos[i, :n, 0] = rng.uniform(0, 105, n)
        pos[i, :n, 1] = rng.uniform(0, 68, n)
    res = collective_from_positions(pos, ns.astype(np.int64))
    for i, n in enumerate(ns):
        xs, ys = pos[i, :n, 0], pos[i, :n, 1]
        cx, cy = np.mean(xs), np.mean(ys)
        assert res["centroid_x"][i] == cx
        assert res["centroid_y"][i] == cy
        assert res["team_length"][i] == np.max(xs) - np.min(xs)
        assert res["team_width"][i] == np.max(ys) - np.min(ys)
        assert res["stretch_index"][i] == np.mean(np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2))
        assert res["stretch_x"][i] == np.mean(np.abs(xs - cx))
        assert res["stretch_y"][i] == np.mean(np.abs(ys - cy))


# --------------------------------------------------------------------------- back line


def test_back_line_batch_matches_legacy_rows():
    rng = np.random.default_rng(0)
    g = 3000
    ps = rng.integers(1, 12, g)
    d0 = rng.integers(0, 2, g).astype(bool)
    pmax = int(ps.max())
    pos = np.full((g, pmax, 2), np.nan)
    for i, p in enumerate(ps):
        pos[i, :p, 0] = rng.integers(0, 11, p).astype(float)  # integer x -> tie-heavy
        pos[i, :p, 1] = rng.uniform(0, 68, p)
    counts = ps.astype(np.int64)
    for n, amn in _DL_VARIANTS:
        res = back_line_batch(pos, counts, d0, n=n, adaptive_max_n=amn)
        exp = {k: np.full(g, np.nan) for k in BACK_LINE_VARIABLES[:-1]}
        exp_bn = np.zeros(g, dtype=np.int64)
        for i, p in enumerate(ps):
            if p < 3:
                continue
            xs, ys = pos[i, :p, 0], pos[i, :p, 1]
            order = np.argsort(xs) if d0[i] else np.argsort(-xs)
            xss, yss = xs[order], ys[order]
            ne = _legacy_select_n(xss, n, amn, int(p))
            sx, sy = xss[:ne], yss[:ne]
            exp["defensive_line_x"][i] = np.mean(sx)
            exp["compactness_x"][i] = np.max(sx) - np.min(sx)
            exp["back_line_high_x"][i] = np.max(sx) if d0[i] else np.min(sx)
            exp["lateral_width"][i] = np.max(sy) - np.min(sy)
            ys2 = np.sort(sy)
            exp["max_lateral_gap"][i] = np.max(np.diff(ys2)) if len(ys2) >= 2 else 0.0
            exp_bn[i] = ne
        for k in BACK_LINE_VARIABLES[:-1]:
            np.testing.assert_array_equal(res[k], exp[k])
        np.testing.assert_array_equal(res["back_n_count"], exp_bn)
        np.testing.assert_array_equal(res["valid"], counts >= 3)


# --------------------------------------------------------------------------- pack / compact


def test_pack_groups_preserves_within_group_order():
    rng = np.random.default_rng(3)
    codes = rng.integers(0, 5, 40)
    x = np.arange(40, dtype=float)
    y = np.arange(40, dtype=float) + 100.0
    pos, counts, first_row = pack_groups(codes, x, y, 5)
    for gcode in range(5):
        rows = np.flatnonzero(codes == gcode)  # input order
        got = pos[gcode, : len(rows)]
        np.testing.assert_array_equal(got[:, 0], x[rows])
        np.testing.assert_array_equal(got[:, 1], y[rows])
        assert counts[gcode] == len(rows)
        if len(rows):
            assert first_row[gcode] == rows[0]
        assert np.isnan(pos[gcode, len(rows) :]).all()


def test_compact_rows_left_aligns_valid_slots():
    pos = np.array(
        [
            [[np.nan, np.nan], [1.0, 2.0], [np.nan, np.nan], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0], [np.nan, np.nan], [np.nan, np.nan]],
        ]
    )
    valid = np.array([[False, True, False, True], [True, True, False, False]])
    out, counts = compact_rows(pos, valid)
    np.testing.assert_array_equal(counts, [2, 2])
    np.testing.assert_array_equal(out[0, :2], [[1.0, 2.0], [3.0, 4.0]])  # slot order preserved
    np.testing.assert_array_equal(out[1, :2], [[5.0, 6.0], [7.0, 8.0]])
