"""Tests for the GK-distribution completion model (xT-GK RAV).

Per docs/superpowers/plans/2026-06-08-xt-gk-goalkick-coverage-implementation.md (Tasks B1-B4).
"""

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._gk_completion import (
    GK_COMPLETION_FEATURE_NAMES,
    GkCompletionModel,
    _gk_completion_density,
    compute_gk_completion,
    extract_gk_completion_features,
    prepare_gk_completion_training_data,
)
from silly_kicks.tracking._gk_geometry import resolve_gk_geometry


def _geom():
    # output of resolve_gk_geometry for 2 goal-kicks (origin imputed for row 1)
    return pd.DataFrame(
        {
            "origin_x": [5.0, 5.5],
            "origin_y": [34.0, 34.0],
            "dest_x": [55.0, 60.0],
            "dest_y": [34.0, 30.0],
            "type_id": [22, 22],
        }
    )


class TestFeatures:
    def test_feature_names_and_shape(self):
        X = extract_gk_completion_features(_geom(), defender_density=pd.Series([3.0, 5.0]))
        assert list(X.columns) == GK_COMPLETION_FEATURE_NAMES
        assert len(X) == 2

    def test_length_is_origin_to_dest(self):
        X = extract_gk_completion_features(_geom(), defender_density=pd.Series([np.nan, np.nan]))
        assert X.loc[0, "length"] == pytest.approx(50.0)  # |55-5|, dy 0

    def test_missing_density_left_nan_for_model_to_impute(self):
        # P3: extract does NOT sentinel-fill; the MODEL mean-imputes density NaN (neutral).
        X = extract_gk_completion_features(_geom(), defender_density=pd.Series([np.nan, 2.0]))
        assert np.isnan(X.loc[0, "dest_defender_density"])
        assert X.loc[1, "dest_defender_density"] == 2.0


class TestModel:
    def _Xy(self, n=400):
        rng = np.random.default_rng(0)
        length = rng.uniform(5, 70, n)
        X = pd.DataFrame(
            {
                "length": length,
                "forwardness": rng.uniform(-1, 1, n),
                "dy_abs": rng.uniform(0, 30, n),
                "dest_x": rng.uniform(20, 100, n),
                "dest_y_off": rng.uniform(0, 34, n),
                "dest_defender_density": rng.uniform(0, 6, n),
                "is_goalkick": (rng.random(n) > 0.5).astype(float),
                "is_throw_in": np.zeros(n),
            }
        )
        y = (rng.random(n) < 1 / (1 + np.exp((length - 35) / 12))).astype(int)
        return X, pd.Series(y)

    def test_fit_predict_in_unit_interval(self):
        X, y = self._Xy()
        m = GkCompletionModel().fit(X, y)
        p = m.predict_proba(X)
        assert p.shape == (len(X),)
        assert (p >= 0).all() and (p <= 1).all()

    def test_pure_numpy_serve_matches_sklearn(self):
        X, y = self._Xy()
        m = GkCompletionModel().fit(X, y)
        m2 = GkCompletionModel.from_dict(m.to_dict())
        np.testing.assert_allclose(m.predict_proba(X), m2.predict_proba(X), atol=1e-9)

    def test_per_type_base_rate_fallback(self):
        # C2: a GEOMETRY-unscoreable row (NaN geometry) -> the recorded per-type base rate.
        X, y = self._Xy()
        m = GkCompletionModel().fit(X, y)
        m._base_rates["goalkick"] = 0.55
        m._base_rates["throw_in"] = 0.95
        gk = pd.DataFrame([{c: np.nan for c in X.columns}])
        gk["is_goalkick"] = 1.0
        gk["is_throw_in"] = 0.0
        ti = pd.DataFrame([{c: np.nan for c in X.columns}])
        ti["is_goalkick"] = 0.0
        ti["is_throw_in"] = 1.0
        assert m.predict_proba(gk)[0] == pytest.approx(0.55)
        assert m.predict_proba(ti)[0] == pytest.approx(0.95)
        assert m._base_rate_for_type(1.0, 0.0) != m._base_rate_for_type(0.0, 1.0)

    def test_density_nan_is_mean_imputed_not_base_rate(self):
        # P3: only density missing (geometry fine) -> scored on geometry with density mean-imputed.
        X, y = self._Xy()
        m = GkCompletionModel().fit(X, y)
        di = m.feature_names.index("dest_defender_density")
        r_nan = X.iloc[[0]].copy()
        r_nan["dest_defender_density"] = np.nan
        r_mean = X.iloc[[0]].copy()
        r_mean["dest_defender_density"] = m._mean[di]
        np.testing.assert_allclose(m.predict_proba(r_nan), m.predict_proba(r_mean), atol=1e-12)

    def test_save_load_roundtrip_sha(self, tmp_path):
        X, y = self._Xy()
        m = GkCompletionModel().fit(X, y)
        m.save(tmp_path)
        assert (tmp_path / "model.json").exists() and (tmp_path / "SHA256SUMS").exists()
        r = GkCompletionModel.load(tmp_path)
        np.testing.assert_allclose(m.predict_proba(X), r.predict_proba(X), atol=1e-9)


class TestPrepare:
    def _actions(self, results):
        n = len(results)
        return pd.DataFrame(
            {
                "game_id": [9] * n,
                "action_id": list(range(n)),
                "team_id": [1] * n,
                "player_id": [10] * n,
                "period_id": [1] * n,
                "time_seconds": np.arange(n) * 10.0,
                "type_id": [22] * n,
                "result_id": results,
                "start_x": [5.0] * n,
                "start_y": [34.0] * n,
                "end_x": np.linspace(40, 90, n),
                "end_y": [34.0] * n,
            }
        )

    def test_returns_X_y_groups(self):
        a = self._actions([1, 0, 1, 1, 0, 1])
        X, y, groups = prepare_gk_completion_training_data(a, frames=None)
        assert len(X) == len(y) == len(groups)
        assert set(np.unique(y)) <= {0, 1}
        assert "origin_source" in X.columns  # metadata for the native gate (#1)

    def test_degenerate_label_raises(self):
        a = self._actions([1, 1, 1, 1, 1, 1])
        with pytest.raises(ValueError, match="degenerate"):
            prepare_gk_completion_training_data(a, frames=None)


class TestComputeStandalone:
    def _fitted(self):
        rng = np.random.default_rng(1)
        n = 300
        length = rng.uniform(5, 70, n)
        X = pd.DataFrame(
            {
                "length": length,
                "forwardness": rng.uniform(-1, 1, n),
                "dy_abs": rng.uniform(0, 30, n),
                "dest_x": rng.uniform(20, 100, n),
                "dest_y_off": rng.uniform(0, 34, n),
                "dest_defender_density": rng.uniform(0, 6, n),
                "is_goalkick": np.ones(n),
                "is_throw_in": np.zeros(n),
            }
        )
        y = (rng.random(n) < 1 / (1 + np.exp((length - 35) / 12))).astype(int)
        return GkCompletionModel().fit(X, pd.Series(y))

    def test_compute_returns_series_in_unit_interval(self):
        m = self._fitted()
        a = pd.DataFrame(
            {
                "game_id": [9],
                "action_id": [0],
                "team_id": [1],
                "player_id": [10],
                "period_id": [1],
                "time_seconds": [5.0],
                "type_id": [22],
                "start_x": [5.0],
                "start_y": [34.0],
                "end_x": [55.0],
                "end_y": [34.0],
            }
        )
        s = compute_gk_completion(a, frames=None, model=m)
        assert s.name == "gk_completion"
        assert 0.0 <= s.iloc[0] <= 1.0

    def test_geometry_unscoreable_returns_per_type_base_rate(self):
        # #3: a goalkick with unresolvable destination (NaN end, last row -> no next-event) routes
        # to the per-type base rate THROUGH the public compute surface (the fallback's only live path).
        m = self._fitted()
        m._base_rates["goalkick"] = 0.61
        a = pd.DataFrame(
            {
                "game_id": [9],
                "action_id": [0],
                "team_id": [1],
                "player_id": [10],
                "period_id": [1],
                "time_seconds": [5.0],
                "type_id": [22],
                "start_x": [5.0],
                "start_y": [34.0],
                "end_x": [np.nan],
                "end_y": [np.nan],
            }
        )
        s = compute_gk_completion(a, frames=None, model=m)
        assert s.iloc[0] == pytest.approx(0.61)


class TestAddGkCompletion:
    """The lakehouse-facing aggregator: a gk_completion column for the wide action-context table."""

    def _frames(self, t=5.0):
        rows = [
            (10, 1, True, 5.0, 34.0),
            (11, 1, False, 30.0, 30.0),
            (20, 2, True, 100.0, 34.0),
            (21, 2, False, 55.0, 30.0),
            (-1, -1, False, 6.0, 34.0),
        ]
        return pd.DataFrame(
            [
                dict(
                    game_id=9,
                    period_id=1,
                    frame_id=125,
                    time_seconds=t,
                    frame_rate=25.0,
                    team_id=team,
                    player_id=pid,
                    is_goalkeeper=gk,
                    is_ball=(pid == -1),
                    x=x,
                    y=y,
                    source_provider="sportec",
                )
                for pid, team, gk, x, y in rows
            ]
        )

    def _actions(self):
        # row 0 = goalkick (in scope); row 1 = midfield pass by a non-GK (out of scope)
        return pd.DataFrame(
            {
                "game_id": [9, 9],
                "action_id": [0, 1],
                "team_id": [1, 1],
                "player_id": [10, 11],
                "period_id": [1, 1],
                "time_seconds": [5.0, 5.0],
                "type_id": [22, 0],
                "start_x": [5.0, 50.0],
                "start_y": [34.0, 34.0],
                "end_x": [55.0, 60.0],
                "end_y": [34.0, 34.0],
            }
        )

    def test_emits_column_provenance_and_masks_out_of_scope(self):
        from silly_kicks.tracking.features import add_gk_completion

        out = add_gk_completion(self._actions(), self._frames())
        assert "gk_completion" in out.columns
        assert "frame_id" in out.columns  # linkage provenance merged
        assert 0.0 <= out.loc[0, "gk_completion"] <= 1.0  # goalkick scored
        assert np.isnan(out.loc[1, "gk_completion"])  # out-of-scope -> NaN
        assert len(out) == len(self._actions())

    def test_links_kwarg_matches_internal_linking(self):
        # the lakehouse pre-links once and passes links= to every add_*; result must match.
        from silly_kicks.tracking.features import add_gk_completion
        from silly_kicks.tracking.utils import link_actions_to_frames

        a, f = self._actions(), self._frames()
        pointers = link_actions_to_frames(a, f)[0]
        out_internal = add_gk_completion(a, f)
        out_prelinked = add_gk_completion(a, f, links=pointers)
        np.testing.assert_allclose(
            out_internal["gk_completion"].to_numpy(),
            out_prelinked["gk_completion"].to_numpy(),
            equal_nan=True,
        )


class TestTrainServeParity:
    """The train==serve discipline (review C1/P1-residual): geometry, density, and feature
    assembly must be produced through the SAME code paths at fit and serve."""

    def test_geom_next_event_dest_resolved_on_full_not_masked_subset(self):
        # P1-residual blind spot: a NaN-destination goalkick's next_event destination must be
        # the next ACTUAL action's start (geometry resolved on the FULL action list, THEN
        # masked), NOT the next in-DOMAIN row's start (geometry resolved on the pre-masked
        # subset) -- otherwise dest_source=="next_event" rows skew between train and serve.
        # Fixture: goalkick(NaN dest) -> outfield pass (out of domain) -> goalkick. The first
        # goalkick's "next event" is the outfield pass, not the second goalkick.
        a = pd.DataFrame(
            {
                "game_id": [9, 9, 9],
                "action_id": [0, 1, 2],
                "team_id": [1, 1, 1],
                "player_id": [10, 11, 10],
                "period_id": [1, 1, 1],
                "time_seconds": [5.0, 15.0, 25.0],
                "type_id": [22, 0, 22],
                "result_id": [0, 1, 1],  # the two in-domain goalkicks (rows 0, 2) split fail/success
                "start_x": [5.0, 41.0, 5.0],
                "start_y": [34.0, 20.0, 34.0],
                "end_x": [np.nan, 60.0, 70.0],
                "end_y": [np.nan, 30.0, 34.0],
            }
        )
        X, _y, _groups = prepare_gk_completion_training_data(a, frames=None)
        # mask keeps the two goalkicks; the first kept row's destination is the OUTFIELD pass's
        # start (41), proving geometry was resolved on the full list before masking.
        assert X.loc[0, "dest_x"] == pytest.approx(41.0)
        # serve resolves identically (same resolve_gk_geometry on the full list).
        geom_serve = resolve_gk_geometry(a, frames=None)
        assert geom_serve.loc[0, "dest_x"] == pytest.approx(41.0)
        assert geom_serve.loc[0, "dest_source"] == "next_event"

    def test_density_helper_nan_without_frames_is_model_imputable(self):
        # The shared _gk_completion_density producer (used by BOTH prepare and serve) returns
        # NaN where it cannot link (here: no frames) -> the model mean-imputes it (review P3),
        # so an unlinked destination never crashes and never sentinels.
        a = pd.DataFrame(
            {
                "game_id": [9],
                "action_id": [0],
                "team_id": [1],
                "player_id": [10],
                "period_id": [1],
                "time_seconds": [5.0],
                "type_id": [22],
                "start_x": [5.0],
                "start_y": [34.0],
                "end_x": [55.0],
                "end_y": [34.0],
            }
        )
        geom = resolve_gk_geometry(a, frames=None)
        dens = _gk_completion_density(a, None, geom, None)
        assert dens.isna().all()  # no frames -> NaN (the documented producer contract)
