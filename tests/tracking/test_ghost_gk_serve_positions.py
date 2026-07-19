"""serve_ghost_gk_positions: positions-only serve + per-row clamp/OOD provenance (TF-19 PR-3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import serve_ghost_gk_positions
from tests.tracking.test_ghost_gk import _fitted_model, _make_ghost_gk_frames


def _frames() -> pd.DataFrame:
    """Ghost-GK-shaped frames (ball, one GK per team, defenders, attackers)."""
    return _make_ghost_gk_frames()


# The fixture is one frame carrying exactly one goalkeeper per team.
_EXPECTED_GK_ROWS = 2


def test_returns_one_row_per_frame_and_gk_team():
    out = serve_ghost_gk_positions(_frames(), model=_fitted_model()[0], home_team_id=1)
    # NON-VACUITY meta-assertion (precedent: test_deep_zone_gate_nan_invariance). Without
    # it this test -- and the notna/isfinite/duplicated assertions in the two below, which
    # are all vacuously true on zero rows -- passes against a stub that always returns the
    # empty frame. Assert the EXACT expected count, not merely non-empty.
    assert len(out) == _EXPECTED_GK_ROWS, (
        f"expected {_EXPECTED_GK_ROWS} GK rows (one per team) for this fixture, got {len(out)}"
    )
    assert set(out.columns) >= {
        "game_id",
        "period_id",
        "frame_id",
        "gk_team_id",
        "ghost_gr_x",
        "ghost_gr_y",
        "ghost_clamped",
        "ghost_out_of_box",
    }
    assert not out.duplicated(subset=["game_id", "period_id", "frame_id", "gk_team_id"]).any()


def test_flags_are_non_null_booleans():
    out = serve_ghost_gk_positions(_frames(), model=_fitted_model()[0], home_team_id=1)
    for col in ("ghost_clamped", "ghost_out_of_box"):
        assert out[col].notna().all(), f"{col} must be non-null (bool(NaN) is True)"
        assert out[col].dtype == bool


def test_positions_are_finite():
    out = serve_ghost_gk_positions(_frames(), model=_fitted_model()[0], home_team_id=1)
    assert np.isfinite(out["ghost_gr_x"]).all()
    assert np.isfinite(out["ghost_gr_y"]).all()


# --- BOTH FLAGS MUST FIRE (spec section 7) ------------------------------------------
# These two are the load-bearing tests in this module. The flags define the probe's
# TRUSTED STRATUM (section 3.1(3)): if either is structurally always-False, the
# dose-banded gate silently evaluates on everything and PR-3b inherits a dead
# stratification. Every OTHER test here is satisfiable with both flags all-False.


def test_out_of_box_flag_FIRES_on_a_planted_beyond_hull_ghost(monkeypatch):
    """Plant a ghost past GRID_X_MAX (30 m goal-relative); the flag must fire."""
    model = _fitted_model()[0]
    monkeypatch.setattr(
        type(model),
        "predict_mean",
        lambda self, features: np.column_stack([np.full(len(features), 45.0), np.full(len(features), 34.0)]),
    )
    out = serve_ghost_gk_positions(_frames(), model=model, home_team_id=1)
    assert out["ghost_out_of_box"].any(), "out-of-box flag never fires -> stratum is dead"
    assert not out["ghost_clamped"].any(), "45 m is on-pitch; the clamp must NOT fire here"


def test_clamped_flag_FIRES_on_a_planted_off_pitch_ghost(monkeypatch):
    """Plant a ghost outside the physical pitch; the clamp flag must fire."""
    model = _fitted_model()[0]
    monkeypatch.setattr(
        type(model),
        "predict_mean",
        lambda self, features: np.column_stack([np.full(len(features), -12.0), np.full(len(features), 34.0)]),
    )
    with pytest.warns(UserWarning, match="outside the physical pitch"):
        out = serve_ghost_gk_positions(_frames(), model=model, home_team_id=1)
    assert out["ghost_clamped"].all(), "clamp flag never fires -> per-row provenance is dead"
    assert (out["ghost_gr_x"] >= 0.0).all(), "a clamped position must land back on the pitch"


# --- THE NO-GK BRANCH ---------------------------------------------------------------
# Reachable (no detected keeper in the window) and it is the branch whose dtype contract
# the downstream probe's _validate_targets depends on. A per-match loop that concatenates
# a no-GK match with a populated one degrades int64 join keys to object unless the empty
# frame declares them identically -- so the parity assertion below is what makes that fix
# stick. NOTE: a fully-empty `frames` raises IndexError on both entry points identically;
# that is pre-existing upstream behaviour and deliberately not exercised here.


def _no_gk_frames() -> pd.DataFrame:
    """Same fixture with every keeper flag cleared -> the zero-row serve path."""
    frames = _make_ghost_gk_frames()
    frames["is_goalkeeper"] = False
    return frames


def test_no_gk_returns_empty_frame_with_the_declared_columns():
    out = serve_ghost_gk_positions(_no_gk_frames(), model=_fitted_model()[0], home_team_id=1)
    assert len(out) == 0
    assert set(out.columns) >= {
        "game_id",
        "period_id",
        "frame_id",
        "gk_team_id",
        "ghost_gr_x",
        "ghost_gr_y",
        "ghost_clamped",
        "ghost_out_of_box",
    }


def test_no_gk_join_key_dtypes_match_the_populated_path():
    """The empty branch must be dtype-identical to the populated one on the join keys.

    Otherwise ``pd.concat`` across matches silently degrades ``period_id``/``frame_id``
    from int64 to object and the downstream join on those keys mis-resolves (ADR-019).
    """
    model = _fitted_model()[0]
    empty = serve_ghost_gk_positions(_no_gk_frames(), model=model, home_team_id=1)
    populated = serve_ghost_gk_positions(_frames(), model=model, home_team_id=1)
    assert len(populated) > 0, "populated fixture is vacuous -- parity check proves nothing"

    for col in ("game_id", "period_id", "frame_id", "gk_team_id"):
        assert empty[col].dtype == populated[col].dtype, (
            f"{col}: empty branch is {empty[col].dtype}, populated path is "
            f"{populated[col].dtype} -- a concat across matches would degrade this join key"
        )


def test_concat_of_no_gk_and_populated_preserves_join_key_dtypes():
    """The failure mode the dtype parity exists to prevent, driven end to end."""
    model = _fitted_model()[0]
    empty = serve_ghost_gk_positions(_no_gk_frames(), model=model, home_team_id=1)
    populated = serve_ghost_gk_positions(_frames(), model=model, home_team_id=1)

    combined = pd.concat([empty, populated], ignore_index=True)
    assert len(combined) == len(populated)
    for col in ("period_id", "frame_id"):
        assert combined[col].dtype == populated[col].dtype, (
            f"{col} degraded to {combined[col].dtype} when concatenated with a no-GK match"
        )
