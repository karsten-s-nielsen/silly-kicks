"""Orientation goldens for the physical-coordinate xT adapters (ADR-041).

The fixture value EQUALS the physical y-centre of each storage band, so ANY y-mirror bug
returns ``68 - y``. A y-symmetric fixture would pass under the very bug these adapters
exist to prevent (feedback_symmetry_test_insufficient_pin_ground_truth).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from silly_kicks.xthreat import (
    ExpectedThreat,
    physical_grid,
    require_fitted_xt,
    values_at_points,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _asymmetric_model() -> ExpectedThreat:
    """Grid value == the physical y-centre of the storage band it lives in.

    Storage row 0 is the TOP of the pitch (``xthreat/_grid.py``), so storage row ``i``
    holds physical y-centre ``(w - 1 - i + 0.5) * (68 / w)``.
    """
    m = ExpectedThreat()
    w, _l = m.xT.shape  # (12, 16)
    cell_w = 68.0 / w
    for i in range(w):
        m.xT[i, :] = (w - 1 - i + 0.5) * cell_w
    return m


class TestPhysicalGrid:
    def test_orientation_golden_uniform(self):
        m = _asymmetric_model()
        gx = (np.arange(104) + 0.5) * (105.0 / 104)
        gy = (np.arange(68) + 0.5) * (68.0 / 68)
        out = physical_grid(m, gx, gy)
        assert out.shape == (68, 104)
        # Interior only: the outermost half-bands extrapolate off the spline's node range.
        interior = (gy > 68.0 / 12) & (gy < 68.0 - 68.0 / 12)
        assert interior.sum() >= 40  # non-vacuity: the assertion covers real rows
        np.testing.assert_allclose(out[interior, 0], gy[interior], atol=1e-9)

    def test_orientation_golden_nonuniform_grid_y(self):
        """No symmetry precondition: the data-level flipud works on ANY ascending grid."""
        m = _asymmetric_model()
        gy = np.array([10.0, 20.0, 55.0])  # ascending, NOT mirror-symmetric about 34
        out = physical_grid(m, np.array([10.0, 52.5, 95.0]), gy)
        np.testing.assert_allclose(out[:, 0], gy, atol=1e-9)

    def test_corner_cells_pinned(self):
        m = ExpectedThreat()
        m.xT[:] = 0.0
        m.xT[0, 15] = 0.9  # storage top-right   -> physical (x~101.7, y~65.2)
        m.xT[11, 0] = 0.2  # storage bottom-left -> physical (x~3.3,  y~2.83)
        gx = np.array([105.0 / 16 * 0.5, 105.0 / 16 * 15.5])
        gy = np.array([68.0 / 12 * 0.5, 68.0 / 12 * 11.5])
        out = physical_grid(m, gx, gy)
        assert out[1, 1] == pytest.approx(0.9)  # high y, high x
        assert out[0, 0] == pytest.approx(0.2)  # low y,  low x

    def test_rejects_non_ascending_grid(self):
        m = _asymmetric_model()
        with pytest.raises(ValueError, match="ascending"):
            physical_grid(m, np.array([50.0, 10.0]), np.array([10.0, 50.0]))

    def test_rejects_degenerate_grid(self):
        m = _asymmetric_model()
        with pytest.raises(ValueError, match="cell centres"):
            physical_grid(m, np.array([50.0]), np.array([10.0, 50.0]))


class TestValuesAtPoints:
    def test_matches_rate_exactly(self):
        """Same frozen cell indexer + row inversion as ExpectedThreat.rate."""
        m = _asymmetric_model()
        actions = pd.DataFrame(
            {
                "type_id": [0, 0],
                "result_id": [1, 1],
                "start_x": [10.0, 60.0],
                "start_y": [10.0, 50.0],
                "end_x": [30.0, 80.0],
                "end_y": [40.0, 20.0],
            }
        )
        expected = values_at_points(m, actions["end_x"], actions["end_y"]) - values_at_points(
            m, actions["start_x"], actions["start_y"]
        )
        assert np.isfinite(expected).all() and np.any(expected != 0.0)  # non-vacuity
        np.testing.assert_allclose(m.rate(actions), expected, atol=1e-12)

    def test_nan_coords_are_nan(self):
        m = _asymmetric_model()
        out = values_at_points(m, np.array([np.nan, 10.0]), np.array([34.0, np.nan]))
        assert np.isnan(out).all()


class TestRequireFittedXt:
    @pytest.mark.parametrize(
        ("bad", "exc"),
        [("default", NotImplementedError), (None, ValueError), ("UNFITTED", NotFittedError)],
    )
    def test_triple(self, bad, exc):
        model = ExpectedThreat() if bad == "UNFITTED" else bad
        with pytest.raises(exc):
            require_fitted_xt(model, caller="probe")

    def test_messages_are_byte_identical_to_the_shipped_text(self):
        """caller="xt_xfns" must reproduce today's messages EXACTLY (review N4).

        Both collapsed call sites pass caller="xt_xfns", so this pins that the guard
        move is a pure refactor with no user-visible text change.
        """
        with pytest.raises(NotImplementedError) as e1:
            require_fitted_xt("default", caller="xt_xfns")
        assert str(e1.value) == ("xt_xfns: bundled xT grid variants are not shipped yet; pass a fitted ExpectedThreat.")
        with pytest.raises(ValueError) as e2:
            require_fitted_xt(None, caller="xt_xfns")
        assert str(e2.value) == "xt_xfns requires a fitted ExpectedThreat (model=...)."
        with pytest.raises(NotFittedError) as e3:
            require_fitted_xt(ExpectedThreat(), caller="xt_xfns")
        assert str(e3.value) == ("xt_xfns requires a fitted ExpectedThreat; call model.fit(actions) first.")


class TestNoSecondImplementation:
    """Property, not binding (review N4): the guard logic exists in exactly ONE module."""

    _FRAGMENT = "bundled xT grid variants are not shipped yet"

    def test_only_physical_module_contains_the_guard_text(self):
        root = _REPO_ROOT / "silly_kicks"
        owners = sorted(
            p.relative_to(root).as_posix()
            for p in root.rglob("*.py")
            if self._FRAGMENT in p.read_text(encoding="utf-8")
        )
        assert owners == ["xthreat/_physical.py"], f"guard duplicated into: {owners}"

    @pytest.mark.parametrize(
        "mod",
        [
            "silly_kicks/vaep/features/expected_threat.py",
            "silly_kicks/atomic/vaep/features.py",
        ],
    )
    def test_call_sites_delegate(self, mod):
        src = (_REPO_ROOT / mod).read_text(encoding="utf-8")
        assert 'require_fitted_xt(model, caller="xt_xfns")' in src
        assert "_require_fitted_xt" not in src, f"{mod}: private guard name survived"


class TestRequireFittedOptOut:
    """``require_fitted=False`` relaxes ONLY the all-zero-grid check (ADR-041).

    It exists so the orientation repair could be shared with ``compute_gk_influence`` and
    ``compute_blocking_score`` -- both of which have a PINNED contract of degrading to NaN
    on a degenerate surface -- without importing a fail-closed policy they never had. The
    calibration harness also legitimately fits an all-zero grid from a slim corpus.
    """

    @staticmethod
    def _grids():
        return np.linspace(0.0, 105.0, 8), np.linspace(0.0, 68.0, 6)

    def test_all_zero_grid_raises_by_default(self):
        from sklearn.exceptions import NotFittedError

        gx, gy = self._grids()
        with pytest.raises(NotFittedError):
            physical_grid(ExpectedThreat(), gx, gy)

    def test_all_zero_grid_is_sampled_when_opted_out(self):
        gx, gy = self._grids()
        out = physical_grid(ExpectedThreat(), gx, gy, require_fitted=False)
        assert out.shape == (len(gy), len(gx))
        assert np.all(out == 0.0), "an all-zero grid must sample to zeros, not to noise"

    @pytest.mark.parametrize("bad", [None, "default"])
    def test_none_and_str_still_fail_closed_under_the_opt_out(self, bad):
        """The relaxation is NOT a blanket bypass: these are misuse under every contract."""
        gx, gy = self._grids()
        with pytest.raises((ValueError, NotImplementedError)):
            physical_grid(bad, gx, gy, require_fitted=False)

    def test_opt_out_does_not_change_a_fitted_grid(self):
        """Byte-identical on the happy path -- the flag must only gate the CHECK."""
        m = ExpectedThreat()
        m.xT[:] = np.linspace(0.01, 0.4, m.xT.shape[1])[np.newaxis, :]
        gx, gy = self._grids()
        np.testing.assert_array_equal(
            physical_grid(m, gx, gy),
            physical_grid(m, gx, gy, require_fitted=False),
        )


class TestValuesAtPointsRequireFittedOptOut:
    """``values_at_points`` carries the SAME opt-out contract as ``physical_grid``.

    Added when the cover-shadow repair routed through it and immediately broke 8 calibration
    tests: that harness legitimately fits an all-zero grid from a slim corpus, and
    ``compute_blocking_score`` has a pinned degrade-to-NaN contract. The two physical
    adapters must not disagree about when they fail closed.
    """

    def test_all_zero_raises_by_default(self):
        from sklearn.exceptions import NotFittedError

        with pytest.raises(NotFittedError):
            values_at_points(ExpectedThreat(), np.array([50.0]), np.array([34.0]))

    def test_all_zero_returns_zeros_when_opted_out(self):
        out = values_at_points(ExpectedThreat(), np.array([50.0, 10.0]), np.array([34.0, 8.0]), require_fitted=False)
        assert out.shape == (2,)
        assert np.all(out == 0.0)

    @pytest.mark.parametrize("bad", [None, "default"])
    def test_none_and_str_still_fail_closed(self, bad):
        with pytest.raises((ValueError, NotImplementedError)):
            values_at_points(bad, np.array([50.0]), np.array([34.0]), require_fitted=False)

    def test_opt_out_does_not_change_a_fitted_lookup(self):
        m = ExpectedThreat()
        m.xT[:] = np.linspace(0.01, 0.4, m.xT.shape[1])[np.newaxis, :]
        x, y = np.array([10.0, 90.0]), np.array([8.0, 60.0])
        np.testing.assert_array_equal(values_at_points(m, x, y), values_at_points(m, x, y, require_fitted=False))

    def test_both_adapters_agree_on_the_contract(self):
        """Non-vacuity: neither adapter may be stricter than the other.

        Note the two take DIFFERENT inputs -- physical_grid takes grid AXES, values_at_points
        takes matched per-POINT arrays. Passing axes to the latter is a ValueError, which is
        how this test first failed.
        """
        gx, gy = np.linspace(0.0, 105.0, 8), np.linspace(0.0, 68.0, 6)
        px, py = np.array([10.0, 50.0, 90.0]), np.array([8.0, 34.0, 60.0])
        assert np.all(physical_grid(ExpectedThreat(), gx, gy, require_fitted=False) == 0.0)
        assert np.all(values_at_points(ExpectedThreat(), px, py, require_fitted=False) == 0.0)
