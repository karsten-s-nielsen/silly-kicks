"""TF-60 PR3 Task 1: GhostGridSpec value object + byte-preserving serialization."""

from silly_kicks.tracking._ghost_gk import DEFAULT_GHOST_GRID, GhostGridSpec


def test_default_grid_matches_current_module_constants():
    from silly_kicks.tracking import _ghost_gk as g

    assert DEFAULT_GHOST_GRID.x_min == g.GRID_X_MIN == 0.0
    assert DEFAULT_GHOST_GRID.x_max == g.GRID_X_MAX == 30.0
    assert DEFAULT_GHOST_GRID.y_min == g.GRID_Y_MIN == 18.0
    assert DEFAULT_GHOST_GRID.y_max == g.GRID_Y_MAX == 50.0
    assert DEFAULT_GHOST_GRID.resolution == g.GRID_RESOLUTION == 0.5


def test_to_metadata_dict_is_the_exact_7_key_shape_with_derived_nx_ny():
    md = DEFAULT_GHOST_GRID.to_metadata_dict()
    assert list(md.keys()) == ["x_min", "x_max", "y_min", "y_max", "nx", "ny", "resolution"]
    assert md == {
        "x_min": 0.0,
        "x_max": 30.0,
        "y_min": 18.0,
        "y_max": 50.0,
        "nx": 60,
        "ny": 64,
        "resolution": 0.5,
    }


def test_derived_nx_ny_reproduce_committed_grid_nx_ny():
    from silly_kicks.tracking import _ghost_gk as g

    assert DEFAULT_GHOST_GRID.nx == g.GRID_NX == 60
    assert DEFAULT_GHOST_GRID.ny == g.GRID_NY == 64


def test_extended_sweeper_grid_derives_105_nx():
    grid = GhostGridSpec(x_min=0.0, x_max=52.5, y_min=18.0, y_max=50.0, resolution=0.5)
    assert grid.nx == 105 and grid.ny == 64
    assert grid.to_metadata_dict()["x_max"] == 52.5


def test_grid_spec_is_frozen_and_hashable():
    # frozen dataclass -> usable as a value (equality + hashable), matching GhostGkDensity's idiom.
    import dataclasses

    import pytest

    grid = GhostGridSpec(0.0, 30.0, 18.0, 50.0, 0.5)
    assert grid == DEFAULT_GHOST_GRID
    assert hash(grid) == hash(DEFAULT_GHOST_GRID)
    with pytest.raises(dataclasses.FrozenInstanceError):
        # The assignment MUST raise at runtime (that's the test); pyright statically flags it because
        # the dataclass is frozen, which is exactly the property under test -- suppress just that.
        grid.x_max = 99.0  # pyright: ignore[reportAttributeAccessIssue]
