"""TF-60 PR3 Tasks 5-6: sweeper variant literals + additive resolver branch + the two-sided
saturation-vs-tracking gate (the signature test), plus the P3P-04 behavioral compute_ghost_gk run."""

import numpy as np

from silly_kicks.tracking._ghost_gk import GhostGkModel, GhostGridSpec, _resolve_ghost_model_for_frames

from ._ghost_toy import (
    fit_toy,
    home_defending_x0,
    home_team_of,
    load_sportec_slim_frames,
    translated_training_set,
    two_team_frames,
)

_SWEEPER_GRID = GhostGridSpec(0.0, 52.5, 18.0, 50.0, 0.5)


# --- Task 5: additive sweeper-family resolver branch --------------------------------------------
def test_sweeper_family_resolves_faithful_on_velocity_bearing(monkeypatch):
    seen = []

    def fake_from_variant(cls, key="default"):
        seen.append(key)
        return GhostGkModel(feature_set=("position_only" if key.endswith("position_only") else "faithful"))

    monkeypatch.setattr(GhostGkModel, "from_variant", classmethod(fake_from_variant))
    _model, key = _resolve_ghost_model_for_frames(two_team_frames(velocity=True), "sweeper")
    assert key == "sweeper" and seen == ["sweeper"]


def test_sweeper_family_resolves_position_only_on_velocity_less(monkeypatch):
    def fake_from_variant(cls, key="default"):
        return GhostGkModel(feature_set="position_only")

    monkeypatch.setattr(GhostGkModel, "from_variant", classmethod(fake_from_variant))
    _model, key = _resolve_ghost_model_for_frames(two_team_frames(velocity=False), "sweeper")
    assert key == "sweeper_position_only"


def test_sweeper_position_only_missing_returns_none_never_default(monkeypatch):
    def fake_from_variant(cls, key="default"):
        if key == "sweeper_position_only":
            raise FileNotFoundError
        raise AssertionError(f"must not fall back to {key!r}")

    monkeypatch.setattr(GhostGkModel, "from_variant", classmethod(fake_from_variant))
    model, key = _resolve_ghost_model_for_frames(two_team_frames(velocity=False), "sweeper")
    assert model is None and key == "sweeper_position_only"


def test_gkdv_none_path_is_byte_identical(monkeypatch):
    # model=None must still resolve to the DEFAULT family, NEVER sweeper (the additivity linchpin).
    seen = []

    def fake_from_variant(cls, key="default"):
        seen.append(key)
        return GhostGkModel()

    monkeypatch.setattr(GhostGkModel, "from_variant", classmethod(fake_from_variant))
    _resolve_ghost_model_for_frames(two_team_frames(velocity=True), None)
    assert seen == ["default"]


def test_sweeper_is_a_valid_variant_literal():
    import typing

    from silly_kicks.tracking import _ghost_gk

    assert set(typing.get_args(_ghost_gk.GhostGkVariant)) >= {
        "default",
        "full",
        "position_only",
        "sweeper",
        "sweeper_position_only",
    }


# --- Task 4 (P3P-04): compute_ghost_gk works on the extended grid (mean path, behavioral) --------
def test_compute_ghost_gk_runs_on_extended_grid_mean_path():
    from silly_kicks.tracking import compute_ghost_gk

    frames = two_team_frames(velocity=True)
    sweeper = GhostGkModel(n_estimators=20, max_depth=3, grid_spec=_SWEEPER_GRID)
    fit_toy(sweeper)
    out = compute_ghost_gk(frames, home_team_id=home_team_of(frames), model=sweeper)  # must NOT raise
    gk = out[out["is_goalkeeper"].astype("boolean").fillna(False)]
    assert gk["ghost_gk_x"].notna().any() and gk["ghost_gk_y"].notna().any()


# --- Task 6: the two-sided saturation-vs-tracking gate ------------------------------------------
def test_default_saturates_and_sweeper_tracks_upfield():
    """Translate a clean scene upfield: the shipped default caps its predicted keeper at ~30 m while
    a toy extended-grid model tracks past it. Direction, not magnitude (survives the toy->real swap)."""
    from silly_kicks.id_compat import canonical_id
    from silly_kicks.tracking import derive_velocities, serve_ghost_gk_positions
    from silly_kicks.tracking._ghost_gk import prepare_ghost_gk_training_data
    from silly_kicks.tracking.preprocess import smooth_frames

    base = load_sportec_slim_frames()
    home = home_defending_x0(base)

    sweeper = GhostGkModel(n_estimators=40, max_depth=4, grid_spec=_SWEEPER_GRID)
    feats, labs = prepare_ghost_gk_training_data(
        translated_training_set(base), home_team_id=home, subsample_fps=None, grid_spec=_SWEEPER_GRID
    )
    sweeper.fit(feats, labs)

    def pred_max(model, delta):
        f = base.copy()
        f["x"] = np.clip(f["x"].to_numpy(dtype=float) + delta, 0.0, 105.0)
        f = derive_velocities(smooth_frames(f))
        served = serve_ghost_gk_positions(f, home_team_id=home, model=model)
        svh = served[served["gk_team_id"].map(canonical_id) == canonical_id(home)]
        return float(svh["ghost_gr_x"].max())

    d25 = pred_max(None, 25)  # shipped default: cannot exceed its 30 m ceiling
    s25 = pred_max(sweeper, 25)  # toy sweeper: places a keeper well past 30 m
    assert d25 <= 31.0, f"default should saturate <=30 m, got {d25:.1f}"
    assert s25 > 33.0, f"sweeper should track past 30 m, got {s25:.1f}"
    assert s25 - d25 > 3.0, f"legs must measurably differ (non-vacuity), got {s25 - d25:.1f}"
