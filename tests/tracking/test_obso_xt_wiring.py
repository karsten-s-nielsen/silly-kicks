"""xt= injection, warning categories and EPV provenance for the OBSO family (ADR-041)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from silly_kicks.tracking import (
    IgnoredSurfaceInputsWarning,
    SyntheticEPVWarning,
)
from silly_kicks.tracking import features as F
from silly_kicks.tracking.utils import link_actions_to_frames
from silly_kicks.xthreat import ExpectedThreat
from tests.tracking._space_creation_mirror_fixture import build_mirror_fixture


def _fitted_xt() -> ExpectedThreat:
    """Fitted-looking xT with a real y gradient (so it is NOT the synthetic ramp)."""
    m = ExpectedThreat()
    w, _l = m.xT.shape
    for i in range(w):
        m.xT[i, :] = 0.01 + 0.02 * (w - 1 - i)
    return m


@pytest.fixture
def obso_actions():
    return build_mirror_fixture()[0]


@pytest.fixture
def obso_frames():
    return build_mirror_fixture()[1]


class TestMutualExclusion:
    def test_add_obso_rejects_both(self, obso_actions, obso_frames):
        with pytest.raises(ValueError, match="either xt= or epv_grid=, not both"):
            F.add_obso(obso_actions, obso_frames, xt=_fitted_xt(), epv_grid=np.ones((68, 104)))

    def test_add_space_creation_rejects_both(self, obso_actions, obso_frames):
        with pytest.raises(ValueError, match="either xt= or epv_grid=, not both"):
            F.add_space_creation(
                obso_actions, obso_frames, home_team_id=5, xt=_fitted_xt(), epv_grid=np.ones((68, 104))
            )

    def test_factories_reject_both(self):
        with pytest.raises(ValueError, match="either xt= or epv_grid=, not both"):
            F.obso_xfns(xt=_fitted_xt(), epv_grid=np.ones((68, 104)))

    def test_unfitted_xt_raises(self, obso_actions, obso_frames):
        from sklearn.exceptions import NotFittedError

        with pytest.raises(NotFittedError):
            F.add_obso(obso_actions, obso_frames, xt=ExpectedThreat())


class TestSyntheticWarning:
    def test_synthetic_default_warns_with_category(self, obso_actions, obso_frames):
        with pytest.warns(SyntheticEPVWarning, match="synthetic"):
            F.add_obso(obso_actions, obso_frames)

    def test_xt_supplied_does_not_warn(self, obso_actions, obso_frames):
        with warnings.catch_warnings():
            warnings.simplefilter("error", SyntheticEPVWarning)
            F.add_obso(obso_actions, obso_frames, xt=_fitted_xt())

    def test_explicit_grid_does_not_warn(self, obso_actions, obso_frames):
        with warnings.catch_warnings():
            warnings.simplefilter("error", SyntheticEPVWarning)
            F.add_obso(obso_actions, obso_frames, epv_grid=np.full((68, 104), 0.2))

    def test_factory_warns_at_factory_call_time(self):
        with pytest.warns(SyntheticEPVWarning):
            F.obso_xfns()

    @pytest.mark.parametrize(
        "call",
        [
            pytest.param(lambda a, f: F.add_obso(a, f), id="add_obso"),
            pytest.param(lambda a, f: F.add_space_creation(a, f, home_team_id=5), id="add_space_creation"),
            pytest.param(lambda a, f: F.obso_xfns(), id="obso_xfns"),
            pytest.param(lambda a, f: F.pausa_xfns(), id="pausa_xfns"),
            pytest.param(lambda a, f: F.space_creation_xfns(home_team_id=5), id="space_creation_xfns"),
        ],
    )
    def test_warning_blames_the_caller_not_silly_kicks_internals(self, obso_actions, obso_frames, call):
        """stacklevel must point at THIS test file, not features.py (review S4).

        A warning attributed to library internals is unactionable noise -- the caller
        cannot tell which of their call sites produced it.
        """
        with pytest.warns(SyntheticEPVWarning) as rec:
            call(obso_actions, obso_frames)
        blamed = [w for w in rec if issubclass(w.category, SyntheticEPVWarning)]
        assert blamed, "no SyntheticEPVWarning recorded"
        assert blamed[0].filename.endswith("test_obso_xt_wiring.py"), (
            f"warning blamed {blamed[0].filename}:{blamed[0].lineno} instead of the caller"
        )


class TestDiscriminatingRealXt:
    def test_real_xt_differs_from_synthetic(self, obso_actions, obso_frames):
        """Non-vacuous: the injected surface must actually move the numbers."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntheticEPVWarning)
            syn = F.add_obso(obso_actions, obso_frames)
        real = F.add_obso(obso_actions, obso_frames, xt=_fitted_xt())
        both = syn["obso_actual"].notna() & real["obso_actual"].notna()
        assert both.any(), "fixture produced no OBSO rows - the comparison would be vacuous"
        delta = (syn.loc[both, "obso_actual"] - real.loc[both, "obso_actual"]).abs()
        assert delta.max() > 1e-6


class TestEpvSourceProvenance:
    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            ({}, "synthetic"),
            ({"xt": "FITTED"}, "xt"),
            ({"epv_grid": "GRID"}, "injected"),
        ],
    )
    def test_source_label(self, obso_actions, obso_frames, kwargs, expected):
        if kwargs.get("xt") == "FITTED":
            kwargs["xt"] = _fitted_xt()
        if kwargs.get("epv_grid") == "GRID":
            kwargs["epv_grid"] = np.full((68, 104), 0.2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntheticEPVWarning)
            out = F.add_obso(obso_actions, obso_frames, **kwargs)
        valued = out["obso_actual"].notna()
        assert valued.any()
        assert (out.loc[valued, "obso_epv_source"] == expected).all()

    def test_dtype_is_pandas_string_not_object(self, obso_actions, obso_frames):
        """Folding the seed into the np.nan pre-seed loop would silently give object."""
        out = F.add_obso(obso_actions, obso_frames, xt=_fitted_xt())
        assert str(out["obso_epv_source"].dtype) == "string"

    def test_present_even_with_supplied_links(self, obso_actions, obso_frames):
        """Must NOT live inside the `links is None` provenance branch."""
        pointers, _ = link_actions_to_frames(obso_actions, obso_frames)
        out = F.add_obso(obso_actions, obso_frames, xt=_fitted_xt(), links=pointers)
        valued = out["obso_actual"].notna()
        assert valued.any()
        assert (out.loc[valued, "obso_epv_source"] == "xt").all()

    def test_space_creation_shares_the_same_name(self, obso_actions, obso_frames):
        """ONE name across the OBSO family: a consumer joining them gets no conflict."""
        out = F.add_space_creation(obso_actions, obso_frames, home_team_id=5, xt=_fitted_xt())
        valued = out["space_created_m2"].notna()
        assert valued.any()
        assert (out.loc[valued, "obso_epv_source"] == "xt").all()


class TestPausaWiring:
    def test_warns_when_surface_inputs_ignored(self, obso_actions, obso_frames):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntheticEPVWarning)
            enriched = F.add_obso(obso_actions, obso_frames)
        with pytest.warns(IgnoredSurfaceInputsWarning, match="ignored"):
            F.add_pausa(enriched, obso_frames, xt=_fitted_xt())

    def test_no_ignored_warning_when_recomputing(self, obso_actions, obso_frames):
        """Fresh actions (no obso columns) -> inputs ARE consulted, no misuse warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", IgnoredSurfaceInputsWarning)
            F.add_pausa(obso_actions, obso_frames, xt=_fitted_xt())

    def test_threads_pitch_control_cache(self, obso_actions, obso_frames):
        from silly_kicks.tracking.pitch_control import PitchControlCache

        cache = PitchControlCache()
        F.add_pausa(obso_actions, obso_frames, xt=_fitted_xt(), pitch_control_cache=cache)
        assert len(cache) > 0


class TestGridRegistrationContract:
    """Producer and consumer must agree on ONE coordinate registration (ADR-041).

    Found by cross-session review: `_resolve_epv_grid` originally built the xt= EPV grid at
    CELL CENTRES ((i + 0.5) * L / n) while every consumer of that grid is NODE-registered --
    compute_pass_obso's index map, PitchControlSurface's linspace, and _interpolate_grid's
    linspace. Because the grid is built at exactly (grid_ny, grid_nx), _interpolate_grid's
    identity shortcut returns it unresampled, so nothing corrected the drift: a systematic
    +-0.505 m error at the edges, invisible on the synthetic linear ramp but not on a
    structured fitted-xT surface.

    This test is the guard that stops the mismatch silently reappearing.
    """

    def test_built_grid_matches_the_consumer_index_map(self):
        from silly_kicks.tracking._obso import ObsoParams
        from silly_kicks.tracking.features import _resolve_epv_grid

        p = ObsoParams()
        grid, source = _resolve_epv_grid(_fitted_xt(), None, caller="contract")
        assert source == "xt"
        assert grid is not None
        assert grid.shape == (p.grid_ny, p.grid_nx)

        # The index -> x map compute_pass_obso inverts: idx = x / L * (n - 1), i.e. the
        # sample coordinate for column i is exactly linspace(0, L, n)[i].
        expected_x = np.linspace(0.0, p.pitch_length, p.grid_nx)
        expected_y = np.linspace(0.0, p.pitch_width, p.grid_ny)
        idx_map_x = np.arange(p.grid_nx) / (p.grid_nx - 1) * p.pitch_length
        idx_map_y = np.arange(p.grid_ny) / (p.grid_ny - 1) * p.pitch_width
        np.testing.assert_allclose(expected_x, idx_map_x, rtol=0, atol=1e-12)
        np.testing.assert_allclose(expected_y, idx_map_y, rtol=0, atol=1e-12)

        # And the grid we built must be the model sampled at exactly those coordinates.
        from silly_kicks.xthreat import physical_grid

        np.testing.assert_allclose(grid, physical_grid(_fitted_xt(), expected_x, expected_y), rtol=0, atol=0)

    def test_cell_centre_registration_would_be_rejected(self):
        """Non-vacuity: the contract above must actually discriminate the wrong convention.

        Needs a model with X structure -- ``_fitted_xt`` varies only in y, so an x-shift
        against it is invisible and this guard would pass vacuously.
        """
        from silly_kicks.tracking._obso import ObsoParams
        from silly_kicks.xthreat import ExpectedThreat, physical_grid

        x_varying = ExpectedThreat()
        _w, n_x = x_varying.xT.shape
        x_varying.xT[:, :] = np.linspace(0.01, 0.5, n_x)  # structure along x

        p = ObsoParams()
        nodes = np.linspace(0.0, p.pitch_length, p.grid_nx)
        centres = (np.arange(p.grid_nx) + 0.5) * (p.pitch_length / p.grid_nx)
        assert np.abs(nodes - centres).max() > 0.5  # the drift this guards against
        gy = np.linspace(0.0, p.pitch_width, p.grid_ny)
        assert not np.allclose(physical_grid(x_varying, nodes, gy), physical_grid(x_varying, centres, gy))


class TestTargetIndexRounding:
    """floor -> round: node registration's correct nearest-neighbour rule (ADR-041)."""

    def test_target_index_is_mirror_symmetric(self):
        from silly_kicks.tracking._obso import ObsoParams

        p = ObsoParams()

        def idx(x: float) -> int:
            return int(np.clip(round(x / p.pitch_length * (p.grid_nx - 1)), 0, p.grid_nx - 1))

        # x and its mirror must land on mirrored columns; with int()/floor they did not
        # (x=15 -> 14, x=90 -> 88, but the mirror of 14 is 89).
        for x in (15.0, 30.0, 47.5, 90.0):
            assert idx(x) == (p.grid_nx - 1) - idx(p.pitch_length - x), f"asymmetric at x={x}"
