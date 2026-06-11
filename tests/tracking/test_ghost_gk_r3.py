"""R3 carrier-param record/consume + serve-carrier consistency (PR-S81)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from silly_kicks.tracking import prepare_ghost_gk_training_data
from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS, infer_ball_carrier
from silly_kicks.tracking._ghost_gk import (
    GHOST_GK_FEATURE_NAMES,
    GhostGkModel,
    _extract_all_ghost_gk_features,
)
from tests.tracking.test_ghost_gk import _fitted_model, _make_ghost_gk_frames


def _frames_with_velocities(n_frames: int = 30) -> pd.DataFrame:
    """A short multi-frame sequence so prepare has enough rows to extract."""
    parts = [_make_ghost_gk_frames(frame_id=fid, timestamp=float(fid)) for fid in range(1, n_frames + 1)]
    return pd.concat(parts, ignore_index=True)


class TestR3Metadata:
    def test_default_carrier_params_on_construction(self):
        model = GhostGkModel()
        assert model.carrier_params == DEFAULT_CARRIER_PARAMS
        assert model.carrier_params is not DEFAULT_CARRIER_PARAMS  # defensive copy

    def test_fit_records_supplied_carrier_params(self):
        _model, X, labels = _fitted_model()
        cp = {"tolerance_m": 3.0, "beta": 0.9, "gamma": 0.25}
        fresh = GhostGkModel(n_estimators=10)
        fresh.fit(X, labels, carrier_params=cp)
        assert fresh.carrier_params == cp

    def test_save_load_round_trips_carrier_params(self):
        _model, X, labels = _fitted_model()
        cp = {"tolerance_m": 3.0, "beta": 0.9, "gamma": 0.25}
        fresh = GhostGkModel(n_estimators=10)
        fresh.fit(X, labels, carrier_params=cp)
        with tempfile.TemporaryDirectory() as d:
            fresh.save(Path(d))
            meta = json.loads((Path(d) / "metadata.json").read_text())
            assert meta["carrier_params"] == cp
            assert meta["version"] == "1.2.0"  # Option A artifact format (gk_y ensemble + baselines)
            assert "sklearn_version" in meta
            reloaded = GhostGkModel.load(Path(d))
            assert reloaded.carrier_params == cp

    def test_load_backcompat_v1_0_0_falls_back_to_default(self, tmp_path):
        # A genuine pre-R3 (v1.0.0) artifact has NO carrier_params -> load() falls back to the
        # library default. Built by saving then stripping the R3/provenance fields (the bundled
        # default is now a v1.1.0 re-fit WITH carrier_params, so it can't test the fallback).
        import hashlib

        _model, X, labels = _fitted_model()
        m = GhostGkModel(n_estimators=10)
        m.fit(X, labels, carrier_params={"tolerance_m": 9.0, "beta": 9.0, "gamma": 9.0})  # non-default
        d = tmp_path / "v100"
        m.save(d)
        meta = json.loads((d / "metadata.json").read_text())
        for k in ("carrier_params", "sklearn_version", "training_commit", "training_platform"):
            meta.pop(k, None)
        meta["version"] = "1.0.0"
        (d / "metadata.json").write_text(json.dumps(meta, indent=2), newline="\n")
        with open(d / "SHA256SUMS", "w", newline="\n") as f:  # re-hash; load() verifies integrity
            for fn in ["rfcde_weights.npz", "metadata.json"]:
                raw = (d / fn).read_bytes()
                if fn.endswith(".json"):
                    raw = raw.replace(b"\r\n", b"\n")
                f.write(f"{hashlib.sha256(raw).hexdigest()}  {fn}\n")
        loaded = GhostGkModel.load(d)
        assert loaded.carrier_params == DEFAULT_CARRIER_PARAMS  # fallback, NOT the 9/9/9 it was saved with


class TestPrepareCarrierParams:
    def test_prepare_accepts_carrier_params_and_stays_2_tuple(self):
        frames = _frames_with_velocities()
        result = prepare_ghost_gk_training_data(
            frames, home_team_id=1, carrier_params={"tolerance_m": 3.0, "beta": 0.0, "gamma": 0.25}
        )
        assert isinstance(result, tuple) and len(result) == 2  # N1: no public break
        features, _labels = result
        assert list(features.columns) == list(GHOST_GK_FEATURE_NAMES)

    def test_prepare_none_is_unchanged_from_bare_default(self):
        frames = _frames_with_velocities()
        f_none, _ = prepare_ghost_gk_training_data(frames, home_team_id=1, carrier_params=None)
        f_default, _ = prepare_ghost_gk_training_data(
            frames, home_team_id=1, carrier_params=dict(DEFAULT_CARRIER_PARAMS)
        )
        pd.testing.assert_frame_equal(f_none, f_default)


def _team_in_poss_column(frames, *, carrier):
    """Extract the internal feature matrix and return the team_in_possession Series + meta."""
    feats, meta = _extract_all_ghost_gk_features(frames, home_team_id=1, carrier=carrier)
    return feats["team_in_possession"], meta


class TestServeCarrier:
    def test_serve_feature_matrix_has_real_team_in_poss(self):
        # P7: assert on the internal feature matrix, not just ghost_gk_x/y.
        # Ball at (50,34) coincides with away attacker a13 -> away team carries.
        frames = _make_ghost_gk_frames()
        model = GhostGkModel.from_variant("default")
        carrier = infer_ball_carrier(frames, **model.carrier_params)[
            ["game_id", "period_id", "frame_id", "ball_carrier_team_id"]
        ]
        tip_fixed, meta = _team_in_poss_column(frames, carrier=carrier)
        tip_bug, _ = _team_in_poss_column(frames, carrier=None)  # simulates the old serve bug

        away_gk = meta["gk_team_id"] == 2
        assert (tip_bug == 0.0).all()  # RED-equivalent: bug path is all-zero
        assert (tip_fixed[away_gk.values] == 1.0).all()  # away GK: its team is in possession  # type: ignore[index]

    def test_compute_ghost_gk_uses_carrier_by_default(self):
        # Post-fix: compute_ghost_gk internally computes the carrier (no carrier= passed).
        # Patch + call THROUGH the runtime module object so the patch target and the
        # called function share one module even if a prior test reimported _ghost_gk.
        import silly_kicks.tracking._ghost_gk as ggk

        frames = _make_ghost_gk_frames()
        with patch.object(ggk, "infer_ball_carrier", wraps=ggk.infer_ball_carrier) as spy:
            ggk.compute_ghost_gk(frames, model="default", home_team_id=1, kde_backend="cpu-numba")
        assert spy.called  # serve no longer skips carrier inference

    def test_supplied_carrier_skips_internal_inference(self):
        # N5: a caller-supplied carrier bypasses the internal infer_ball_carrier call.
        import silly_kicks.tracking._ghost_gk as ggk

        frames = _make_ghost_gk_frames()
        model = GhostGkModel.from_variant("default")
        carrier = ggk.infer_ball_carrier(frames, **model.carrier_params)[
            ["game_id", "period_id", "frame_id", "ball_carrier_team_id"]
        ]
        with patch.object(ggk, "infer_ball_carrier") as spy:
            ggk.compute_ghost_gk(frames, model=model, home_team_id=1, carrier=carrier, kde_backend="cpu-numba")
        assert not spy.called  # passthrough avoids recomputation

    def test_train_serve_feature_parity(self):
        # P7 test #2: team_in_possession extracted via prepare (train) == via
        # compute_ghost_gk's internal extraction (serve) on the same frames + params.
        frames = _frames_with_velocities()
        cp = dict(DEFAULT_CARRIER_PARAMS)
        train_feats, _ = prepare_ghost_gk_training_data(frames, home_team_id=1, carrier_params=cp)
        carrier = infer_ball_carrier(frames, **cp)[  # type: ignore[arg-type]
            ["game_id", "period_id", "frame_id", "ball_carrier_team_id"]
        ]
        serve_feats, _ = _extract_all_ghost_gk_features(frames, home_team_id=1, carrier=carrier)
        assert serve_feats["team_in_possession"].sum() > 0
        assert train_feats["team_in_possession"].sum() > 0


class TestAggregatorCarrierPassthrough:
    def test_add_ghost_gk_accepts_and_forwards_carrier(self):
        import silly_kicks.tracking._ghost_gk as ggk
        from silly_kicks.tracking.features import add_ghost_gk

        frames = _make_ghost_gk_frames()
        actions = pd.DataFrame(
            {
                "action_id": [0],
                "game_id": ["100"],
                "period_id": [1],
                "team_id": [1],  # home acts -> defending GK is away
                "start_x": [50.0],
                "start_y": [34.0],
                "time_seconds": [1.0],
            }
        )
        model = GhostGkModel.from_variant("default")
        carrier = ggk.infer_ball_carrier(frames, **model.carrier_params)[
            ["game_id", "period_id", "frame_id", "ball_carrier_team_id"]
        ]
        # add_ghost_gk's function-local `from ._ghost_gk import compute_ghost_gk` resolves
        # the current sys.modules module; patch.object on that same module keeps them aligned.
        with patch.object(ggk, "infer_ball_carrier") as spy:
            add_ghost_gk(actions, frames, model=model, home_team_id=1, carrier=carrier, kde_backend="cpu-numba")
        assert not spy.called  # carrier forwarded to compute_ghost_gk


class TestRecordedEqualsUsed:
    @staticmethod
    def _near_tie_frames() -> pd.DataFrame:
        base = {
            "game_id": "100",
            "period_id": 1,
            "frame_id": 1,
            "timestamp": 1.0,
            "ball_state": "alive",
            "time_seconds": 1.0,
            "source_provider": "test",
        }

        def row(pid, team, x, y, vx, vy, ball=False, gk=False):
            return {
                **base,
                "player_id": pid,
                "team_id": team,
                "x": x,
                "y": y,
                "vx": vx,
                "vy": vy,
                "speed": (vx**2 + vy**2) ** 0.5,
                "is_ball": ball,
                "is_goalkeeper": gk,
            }

        rows = [
            row("ball", None, 50.0, 34.0, 0.0, 0.0, ball=True),
            # home player: 1.9m left of ball, stationary -> UNAMBIGUOUSLY closer (NP4:
            # removes tiebreak dependence so beta=0.0 deterministically picks home).
            row("h1", 1, 48.1, 34.0, 0.0, 0.0),
            # away player: 2.0m right of ball (slightly farther), moving LEFT toward it
            # -> beta=0.9 flips the pick to away purely via velocity, not distance.
            row("a1", 2, 52.0, 34.0, -3.0, 0.0),
            row("h_gk", 1, 5.0, 34.0, 0.0, 0.0, gk=True),
            row("a_gk", 2, 100.0, 34.0, 0.0, 0.0, gk=True),
        ]
        return pd.DataFrame(rows)

    def test_carrier_params_flip_the_carrier_and_are_recorded(self):
        frames = self._near_tie_frames()
        c_dist = infer_ball_carrier(frames, tolerance_m=3.0, beta=0.0, gamma=0.25)
        c_vel = infer_ball_carrier(frames, tolerance_m=3.0, beta=0.9, gamma=0.25)

        def tid(c):
            return c["ball_carrier_team_id"].iloc[0]

        assert tid(c_dist) != tid(c_vel)  # the fixture genuinely makes beta bite (N2)

        # recorded == used: fit records the supplied carrier_params. Use the standard
        # non-degenerate synthetic features (a 24x-cloned near-tie sequence yields constant
        # feature columns, and newer sklearn/numpy raise in HGBR binning on a constant column).
        # The full prepare->fit recorded==used wiring is covered by the trainer CLI test.
        cp = {"tolerance_m": 3.0, "beta": 0.9, "gamma": 0.25}
        _m, X, labels = _fitted_model()
        m = GhostGkModel(n_estimators=10)
        m.fit(X, labels, carrier_params=cp)
        assert m.carrier_params == cp


class TestAtomicMirror:
    def test_atomic_add_ghost_gk_inherits_carrier_kwarg(self):
        import inspect

        from silly_kicks.atomic.tracking.features import add_ghost_gk as atomic_add
        from silly_kicks.tracking.features import add_ghost_gk as std_add

        assert atomic_add is std_add  # re-export, not a copy
        assert "carrier" in inspect.signature(atomic_add).parameters
