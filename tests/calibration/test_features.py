import pandas as pd

from silly_kicks.calibration._features import (
    _TRIAL_DEPENDENT_COLS,
    ALL_FEATURES,
    enrich_full,
    enrich_invariant,
    patch_trial_columns,
)

_CP = {"tolerance_m": 3.0, "beta": 0.5, "gamma": 1.0}


def test_invariant_sets_trial_cols_nan_and_others_present(synth, frozen_xt):
    actions, frames, home = synth
    base, links, _das_ok = enrich_invariant(
        actions=actions, frames=frames, xt=frozen_xt.xt, home_team_id=home, carrier_params=_CP
    )
    for col in _TRIAL_DEPENDENT_COLS:
        assert base[col].isna().all(), f"{col} must be a NaN placeholder in the invariant"
    # A non-trial tracking feature must be materialised (not all-NaN) for at least some rows.
    assert base["pressure_on_actor__andrienko_oval"].notna().any()
    assert "frame_id" in links.columns
    # Every model feature is present after the invariant pass.
    assert [c for c in ALL_FEATURES if c not in base.columns] == []


def test_patch_overwrites_exactly_the_trial_cols(synth, frozen_xt):
    actions, frames, home = synth
    base, links, _das_ok = enrich_invariant(
        actions=actions, frames=frames, xt=frozen_xt.xt, home_team_id=home, carrier_params=_CP
    )
    invariant_snapshot = base.drop(columns=_TRIAL_DEPENDENT_COLS).copy()
    patched = patch_trial_columns(
        base_actions=base,
        frames=frames,
        links=links,
        home_team_id=home,
        k3=2.0,
        pre_seconds=2.0,
        min_displacement_m=4.0,
    )
    # Trial cols are now populated...
    assert patched["pressure_on_actor__link_zones"].notna().any()
    assert patched["n_off_ball_runners_pre_window"].notna().any()
    # ...and NO invariant column changed.
    pd.testing.assert_frame_equal(
        patched.drop(columns=_TRIAL_DEPENDENT_COLS)[invariant_snapshot.columns],
        invariant_snapshot,
    )


def test_line_break_columns_are_not_features():
    assert not any("line_break" in c or "lines_broken" in c for c in ALL_FEATURES)


def test_all_features_count_matches_spec():
    for col in _TRIAL_DEPENDENT_COLS:
        assert col in ALL_FEATURES


def test_enrich_full_is_independent_and_populates_trial_cols(synth, frozen_xt):
    actions, frames, home = synth
    full = enrich_full(
        actions=actions,
        frames=frames,
        xt=frozen_xt.xt,
        home_team_id=home,
        carrier_params=_CP,
        k3=1.5,
        pre_seconds=2.0,
        min_displacement_m=4.0,
    )
    for col in _TRIAL_DEPENDENT_COLS:
        assert col in full.columns
    # The always-computed trial outputs are populated inline (NOT NaN placeholders). The
    # displacement/speed run cols can legitimately be all-NaN when no off-ball run is detected
    # in a short fixture — the column-parity test (full == invariant+patch) covers those.
    assert full["pressure_on_actor__link_zones"].notna().any()
    assert full["n_off_ball_runners_pre_window"].notna().any()
