"""Tests for the Ghost-GK served estimator - Option A (exact boosted ``predict_mean``).

The served ``ghost_gk_x/y`` are the exact sklearn ``HistGradientBoostingRegressor``
boosted prediction, reconstructed pickle-free via leaf ``value`` traversal
(``baseline + sum_trees leaf_value``). See
docs/superpowers/specs/2026-06-04-pr-s82-ghost-gk-option-a-design.md.

Format-change window: tests that load the bundled ``default`` artifact (``from_variant``)
and the bundled-artifact smoke test stay RED until the new-format weights are re-bundled
(plan Task 12). The fresh-model tests below are the per-task gate for Tasks 1-11.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._ghost_gk import (
    GRID_X_MAX,
    GRID_X_MIN,
    GRID_Y_MAX,
    GRID_Y_MIN,
    SERVED_ESTIMATOR,
    GhostGkModel,
)

# Reuse the shared synthetic fixtures from the main ghost-gk test module.
from tests.tracking.test_ghost_gk import _fitted_model, _make_ghost_gk_frames


class TestBoostedParity:
    """The correctness spine: numpy reconstruction == sklearn .predict() exactly."""

    def test_boosted_predict_mean_matches_sklearn(self):
        """predict_mean (leaf-value reconstruction) == sklearn .predict() to 1e-6, incl. NaN routing."""
        from silly_kicks.tracking import _ghost_gk as gg

        rng = np.random.default_rng(3)
        n = 300
        X = pd.DataFrame(rng.standard_normal((n, 26)), columns=gg.GHOST_GK_FEATURE_NAMES)
        X["phase"] = rng.integers(0, 3, n).astype(float)  # exercise phase values
        # NaN in real features (derive names from the schema, never hardcode) so HGBR
        # actually learns missing_go_to_left and the reconstruction's missing branch is
        # parity-checked against sklearn, not asserted-by-construction.
        nan_cols = [c for c in gg.GHOST_GK_FEATURE_NAMES if c != "phase"][:4]
        for col in nan_cols:
            X.loc[rng.choice(n, 30, replace=False), col] = np.nan
        y = pd.DataFrame({"gk_x": rng.uniform(2, 20, n), "gk_y": rng.uniform(25, 45, n)})
        m = gg.GhostGkModel(n_estimators=30).fit(X, y)
        # No categorical split nodes after phase-numeric (guards exact reconstruction):
        assert sum(int(t["is_categorical"].sum()) for t in m._tree_nodes) == 0  # type: ignore[union-attr]
        # Parity vs the live sklearn regressors kept transiently after fit() (canonical col order):
        Xv = X[gg.GHOST_GK_FEATURE_NAMES].values
        sk = np.column_stack([m._sk_reg_x.predict(Xv), m._sk_reg_y.predict(Xv)])  # type: ignore[union-attr]
        np.testing.assert_allclose(m.predict_mean(X), sk, atol=1e-6)

    def test_predict_mean_reindexes_to_canonical_order(self):
        """A column-reordered DataFrame still predicts correctly (Hyrum positional guard)."""
        from silly_kicks.tracking import _ghost_gk as gg

        rng = np.random.default_rng(11)
        n = 60
        X = pd.DataFrame(rng.standard_normal((n, 26)), columns=gg.GHOST_GK_FEATURE_NAMES)
        X["phase"] = rng.integers(0, 3, n).astype(float)
        y = pd.DataFrame({"gk_x": rng.uniform(2, 20, n), "gk_y": rng.uniform(25, 45, n)})
        m = gg.GhostGkModel(n_estimators=10).fit(X, y)
        shuffled = X[list(reversed(gg.GHOST_GK_FEATURE_NAMES))]
        np.testing.assert_array_equal(m.predict_mean(X), m.predict_mean(shuffled))


class TestFitStoresEnsembles:
    def test_fit_stores_both_ensembles_and_baselines_no_categoricals(self):
        from silly_kicks.tracking import _ghost_gk as gg

        rng = np.random.default_rng(1)
        n = 120
        X = pd.DataFrame(rng.standard_normal((n, 26)), columns=gg.GHOST_GK_FEATURE_NAMES)
        X["phase"] = rng.integers(0, 3, n).astype(float)
        y = pd.DataFrame({"gk_x": rng.uniform(2, 20, n), "gk_y": rng.uniform(25, 45, n)})
        m = gg.GhostGkModel(n_estimators=10).fit(X, y)
        assert m._tree_nodes_y is not None and len(m._tree_nodes_y) > 0
        assert isinstance(m._baseline_x, float) and isinstance(m._baseline_y, float)
        assert sum(int(t["is_categorical"].sum()) for t in m._tree_nodes) == 0  # type: ignore[union-attr]  # phase numeric
        assert sum(int(t["is_categorical"].sum()) for t in m._tree_nodes_y) == 0

    def test_training_gk_y_is_input_labels(self):
        """Chesterton lock: training_gk_y comes from labels (KDE still reads them)."""
        model, _X, labels = _fitted_model()
        np.testing.assert_array_equal(model._training_gk_y, labels["gk_y"].values.astype(np.float64))


class TestPredictMeanLoadSafe:
    def test_predict_mean_boosted_load_safe_and_parity(self):
        """predict_mean is bit-identical before and after save()/load()."""
        from silly_kicks.tracking import _ghost_gk as gg

        rng = np.random.default_rng(2)
        n = 100
        X = pd.DataFrame(rng.standard_normal((n, 26)), columns=gg.GHOST_GK_FEATURE_NAMES)
        X["phase"] = rng.integers(0, 3, n).astype(float)
        y = pd.DataFrame({"gk_x": rng.uniform(2, 20, n), "gk_y": rng.uniform(25, 45, n)})
        m = gg.GhostGkModel(n_estimators=20).fit(X, y)
        before = m.predict_mean(X[:8])
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "m"
            m.save(p)
            after = gg.GhostGkModel.load(p).predict_mean(X[:8])
        np.testing.assert_array_equal(before, after)

    def test_predict_mean_works_after_load(self):
        """predict_mean returns finite (n,2) after load() (regressors are transient)."""
        model, X, _ = _fitted_model()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "m"
            model.save(p)
            out = GhostGkModel.load(p).predict_mean(X[:5])
        assert out.shape == (5, 2)
        assert np.all(np.isfinite(out))


class TestPredictAlignsWithServed:
    def test_predict_equals_predict_mean(self):
        """predict() now returns the served boosted mean (== predict_mean)."""
        model, X, _ = _fitted_model()
        np.testing.assert_allclose(model.predict(X[:6]), model.predict_mean(X[:6]), atol=1e-12)

    def test_mode_still_reachable_via_density(self):
        """The mode is not lost - predict_density still exposes it."""
        model, X, _ = _fitted_model()
        d = model.predict_density(X[:1])[0]
        assert GRID_X_MIN <= d.mode_x <= GRID_X_MAX


class TestComputeServesBoosted:
    def test_compute_reads_boosted_mean_not_mode(self):
        """compute_ghost_gk emits predict_mean (boosted), NOT the KDE mode_* (headline change).

        Built entirely from the CURRENT _ghost_gk module (fresh model, fresh compute,
        fresh extraction) so it is robust to the sys.modules reimport in the numba
        eager-import test (which can otherwise make compute's isinstance() resolve a
        different class and fall back to the bundled model). Compares compute's served
        ghost_gk_x against the SAME model's predict_mean (must match) and the
        predict_density mode_x (must differ - the discriminating assertion).
        """
        from silly_kicks.tracking import _ghost_gk as gg

        rng = np.random.default_rng(7)
        n = 80
        X = pd.DataFrame(rng.standard_normal((n, 26)), columns=gg.GHOST_GK_FEATURE_NAMES)
        X["phase"] = rng.integers(0, 3, n).astype(float)
        labels = pd.DataFrame({"gk_x": rng.uniform(2, 20, n), "gk_y": rng.uniform(25, 45, n)})
        model = gg.GhostGkModel(n_estimators=10)
        model.fit(X, labels)

        frames = pd.concat(
            [_make_ghost_gk_frames(frame_id=1, timestamp=1.0), _make_ghost_gk_frames(frame_id=2, timestamp=1.5)],
            ignore_index=True,
        )
        result = gg.compute_ghost_gk(frames, model=model, home_team_id=1)
        assert "ghost_gk_spread" not in result.columns
        gk = result["is_goalkeeper"].astype(bool) & ~result["is_ball"].astype(bool)
        served = sorted(np.round(result.loc[gk, "ghost_gk_x"].dropna().to_numpy(), 5).tolist())
        assert len(served) > 0

        # Reproduce compute's internal predictions with the SAME model object.
        carrier = gg.infer_ball_carrier(frames, **model.carrier_params)[
            ["game_id", "period_id", "frame_id", "ball_carrier_team_id"]
        ]
        feats, _meta = gg._extract_all_ghost_gk_features(frames, home_team_id=1, carrier=carrier)
        boosted = sorted(np.round(model.predict_mean(feats)[:, 0], 5).tolist())
        mode = sorted(np.round([d.mode_x for d in model.predict_density(feats)], 5).tolist())

        assert served == boosted  # compute serves predict_mean
        assert served != mode  # ... not the KDE mode (discriminating)


def _rewrite_sha256sums(path: Path) -> None:
    import hashlib

    lines = []
    for fname in ["rfcde_weights.npz", "metadata.json"]:
        raw = (path / fname).read_bytes()
        if fname.endswith(".json"):
            raw = raw.replace(b"\r\n", b"\n")
        lines.append(f"{hashlib.sha256(raw).hexdigest()}  {fname}\n")
    (path / "SHA256SUMS").write_text("".join(lines), newline="\n")


def _set_metadata_key_and_rehash(path: Path, key: str, value) -> None:
    meta = json.loads((path / "metadata.json").read_text())
    meta[key] = value
    (path / "metadata.json").write_text(json.dumps(meta, indent=2), newline="\n")
    _rewrite_sha256sums(path)


def _strip_metadata_key_and_rehash(path: Path, key: str) -> None:
    meta = json.loads((path / "metadata.json").read_text())
    meta.pop(key, None)
    (path / "metadata.json").write_text(json.dumps(meta, indent=2), newline="\n")
    _rewrite_sha256sums(path)


class TestServeEstimatorMetadata:
    def test_served_estimator_is_boosted_mean(self):
        assert SERVED_ESTIMATOR == "boosted_mean"

    def test_save_records_serve_estimator(self):
        model, _, _ = _fitted_model()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "m"
            model.save(p)
            meta = json.loads((p / "metadata.json").read_text())
        assert meta["serve_estimator"] == SERVED_ESTIMATOR
        assert meta["version"] == "1.3.0"

    def test_load_absent_serve_estimator_ok(self):
        """Back-compat: an artifact without serve_estimator loads (defaults)."""
        model, X, _ = _fitted_model()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "m"
            model.save(p)
            _strip_metadata_key_and_rehash(p, "serve_estimator")
            loaded = GhostGkModel.load(p)
        assert loaded.predict(X[:2]).shape == (2, 2)

    def test_load_conflicting_serve_estimator_raises(self):
        """A conflicting serve_estimator fails closed (R3)."""
        from silly_kicks.tracking._ghost_gk import IntegrityError

        model, _, _ = _fitted_model()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "m"
            model.save(p)
            _set_metadata_key_and_rehash(p, "serve_estimator", "something_else")
            with pytest.raises((IntegrityError, ValueError)):
                GhostGkModel.load(p)


class TestLoadFailsClosedOnPreOptionA:
    def test_load_raises_on_missing_gk_y_trees(self):
        """An old-format artifact (no gk_y trees / baselines) raises a clear re-fit error."""
        from silly_kicks.tracking._ghost_gk import IntegrityError

        model, _, _ = _fitted_model()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "m"
            model.save(p)
            # Strip the Option-A arrays from the npz to emulate a pre-Option-A artifact.
            _dropped = ("n_trees_y", "baseline_x", "baseline_y")
            with np.load(p / "rfcde_weights.npz", allow_pickle=False) as data:
                kept = {k: data[k] for k in data.files if not k.startswith("tree_nodes_y_") and k not in _dropped}
            np.savez_compressed(str(p / "rfcde_weights.npz"), **kept)
            _rewrite_sha256sums(p)
            with pytest.raises((IntegrityError, RuntimeError), match=r"(?i)re-fit|option a|pre-option"):
                GhostGkModel.load(p)


class TestBundledArtifactSmoke:
    def test_bundled_default_predict_mean_finite_in_bounds(self):
        """Load the SHIPPED bundled default + predict_mean -> finite coords in pitch bounds.

        The only e2e proving the re-published weights serialize + reconstruct correctly;
        synthetic-model tests prove the code, not the artifact.
        """
        from silly_kicks.tracking import _ghost_gk as gg

        m = GhostGkModel.from_variant("default")
        assert m._tree_nodes_y is not None and m._baseline_x is not None and m._baseline_y is not None
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((4, 26)), columns=gg.GHOST_GK_FEATURE_NAMES)
        X["phase"] = rng.integers(0, 3, 4).astype(float)
        out = m.predict_mean(X)
        assert np.all(np.isfinite(out))
        assert np.all((out[:, 0] >= GRID_X_MIN - 5) & (out[:, 0] <= GRID_X_MAX + 5))
        assert np.all((out[:, 1] >= GRID_Y_MIN - 5) & (out[:, 1] <= GRID_Y_MAX + 5))
