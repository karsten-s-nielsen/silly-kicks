import dataclasses

import pytest

from silly_kicks.xthreat._params import (
    GridSpec,
    KDEParams,
    SinghParams,
    validate_params_for_method,
)


def test_gridspec_defaults_match_legacy_16x12():
    g = GridSpec()
    assert (g.n_zones_x, g.n_zones_y) == (16, 12)
    assert g.n_zones == 192


def test_gridspec_cell_dims_from_spadlconfig():
    import silly_kicks.spadl.config as cfg

    g = GridSpec(n_zones_x=12, n_zones_y=8)
    assert g.cell_length == pytest.approx(cfg.field_length / 12)
    assert g.cell_width == pytest.approx(cfg.field_width / 8)


def test_gridspec_rejects_nonpositive():
    with pytest.raises(ValueError):
        GridSpec(n_zones_x=0, n_zones_y=8)


def test_validate_accepts_matching_params():
    validate_params_for_method("singh_counts", None)
    validate_params_for_method("singh_counts", SinghParams())
    validate_params_for_method("kde_smoothed", KDEParams())


def test_validate_rejects_mismatched_params():
    with pytest.raises(TypeError):
        validate_params_for_method("singh_counts", KDEParams())


def test_validate_rejects_unknown_method():
    with pytest.raises(ValueError):
        validate_params_for_method("bogus", None)  # type: ignore[arg-type]


def test_params_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        KDEParams().bandwidth = 1.0  # type: ignore[misc]


def test_gridspec_default_matches_grid_module_constants():
    from silly_kicks.xthreat import _grid
    from silly_kicks.xthreat._params import GridSpec

    assert (GridSpec().n_zones_x, GridSpec().n_zones_y) == (_grid.N, _grid.M)
