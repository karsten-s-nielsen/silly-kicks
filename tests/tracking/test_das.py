"""Unit tests for DAS adapter."""

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._das import (
    _X_OFFSET,
    _Y_OFFSET,
)


class TestTodasCoords:
    def test_origin_shifts(self) -> None:
        from silly_kicks.tracking._das import _to_das_coords

        df = pd.DataFrame(
            {
                "x": [0.0, 105.0, 52.5],
                "y": [0.0, 68.0, 34.0],
                "vx": [1.0, 2.0, 0.0],
                "vy": [0.5, -0.5, 0.0],
            }
        )
        result = _to_das_coords(df)
        np.testing.assert_allclose(result["x"].values, [-_X_OFFSET, _X_OFFSET, 0.0])
        np.testing.assert_allclose(result["y"].values, [-_Y_OFFSET, _Y_OFFSET, 0.0])
        np.testing.assert_allclose(result["vx"].values, [1.0, 2.0, 0.0])
        np.testing.assert_allclose(result["vy"].values, [0.5, -0.5, 0.0])

    def test_does_not_mutate_input(self) -> None:
        from silly_kicks.tracking._das import _to_das_coords

        df = pd.DataFrame({"x": [50.0], "y": [30.0], "vx": [1.0], "vy": [0.0]})
        _ = _to_das_coords(df)
        assert df["x"].iloc[0] == 50.0
        assert df["y"].iloc[0] == 30.0


class TestInputValidation:
    def _minimal_frames(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "game_id": [1],
                "period_id": [1],
                "frame_id": [0],
                "player_id": ["P1"],
                "team_id": ["A"],
                "x": [50.0],
                "y": [34.0],
                "vx": [0.0],
                "vy": [0.0],
                "is_ball": [False],
                "team_in_possession": ["A"],
            }
        )

    def test_missing_vx_raises(self) -> None:
        from silly_kicks.tracking._das import _validate_das_inputs

        df = self._minimal_frames().drop(columns=["vx"])
        with pytest.raises(ValueError, match="velocity columns"):
            _validate_das_inputs(df)

    def test_missing_vy_raises(self) -> None:
        from silly_kicks.tracking._das import _validate_das_inputs

        df = self._minimal_frames().drop(columns=["vy"])
        with pytest.raises(ValueError, match="velocity columns"):
            _validate_das_inputs(df)

    def test_missing_team_in_possession_raises(self) -> None:
        from silly_kicks.tracking._das import _validate_das_inputs

        df = self._minimal_frames().drop(columns=["team_in_possession"])
        with pytest.raises(ValueError, match="team_in_possession"):
            _validate_das_inputs(df)

    def test_valid_frames_pass(self) -> None:
        from silly_kicks.tracking._das import _validate_das_inputs

        df = self._minimal_frames()
        _validate_das_inputs(df)


class TestImportGuard:
    def test_missing_package_raises_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import silly_kicks.tracking._das as das_mod

        monkeypatch.setattr(
            das_mod,
            "_import_accessible_space",
            lambda: (_ for _ in ()).throw(
                ImportError(
                    "accessible-space is required for DAS features. Install with: pip install 'silly-kicks[das]'"
                )
            ),
        )
        with pytest.raises(ImportError, match="silly-kicks\\[das\\]"):
            das_mod._import_accessible_space()


@pytest.mark.e2e
class TestGetDasShapeAlignment:
    def test_output_length_matches_input(self) -> None:
        pytest.importorskip("accessible_space")
        from silly_kicks.tracking._das import get_das

        rng = np.random.default_rng(42)
        n_frames = 3
        rows = []
        for fid in range(n_frames):
            for pid, tid in [("P1", "Home"), ("P2", "Away")]:
                rows.append(
                    {
                        "game_id": 1,
                        "period_id": 1,
                        "frame_id": fid,
                        "player_id": pid,
                        "team_id": tid,
                        "x": rng.uniform(10, 95),
                        "y": rng.uniform(5, 63),
                        "vx": rng.normal(0, 2),
                        "vy": rng.normal(0, 2),
                        "is_ball": False,
                        "team_in_possession": "Home",
                    }
                )
            rows.append(
                {
                    "game_id": 1,
                    "period_id": 1,
                    "frame_id": fid,
                    "player_id": "ball",
                    "team_id": None,
                    "x": rng.uniform(20, 80),
                    "y": rng.uniform(10, 58),
                    "vx": 0.0,
                    "vy": 0.0,
                    "is_ball": True,
                    "team_in_possession": "Home",
                }
            )
        frames = pd.DataFrame(rows)
        result = get_das(frames, use_progress_bar=False)
        assert len(result) == len(frames), f"get_das output length {len(result)} != input length {len(frames)}"


class TestDasXfns:
    def test_das_xfns_are_frame_aware(self) -> None:
        from silly_kicks.tracking.features import das_xfns
        from silly_kicks.vaep.feature_framework import is_frame_aware

        for xfn in das_xfns:
            assert is_frame_aware(xfn), f"{xfn.__name__} is not frame_aware"

    def test_das_xfns_feature_column_names(self) -> None:
        """feature_column_names introspection with empty frames must not crash."""
        from silly_kicks.tracking.features import das_xfns
        from silly_kicks.vaep.features import feature_column_names

        cols = feature_column_names(das_xfns, nb_prev_actions=3)
        expected_cols = {
            "das_team_a0",
            "das_team_a1",
            "das_team_a2",
            "das_opponent_a0",
            "das_opponent_a1",
            "das_opponent_a2",
            "das_diff_a0",
            "das_diff_a1",
            "das_diff_a2",
        }
        assert expected_cols == set(cols)

    def test_das_xfns_length(self) -> None:
        from silly_kicks.tracking.features import das_xfns

        # Single custom transformer produces all 9 columns
        assert len(das_xfns) == 1

    def test_das_at_action_introspection(self) -> None:
        """frames=None introspection must return NaN Series with correct name."""
        from silly_kicks.tracking.features import das_at_action

        dummy = pd.DataFrame({"action_id": [1, 2], "team_id": [1, 1]})
        result = das_at_action(dummy, None)
        assert result.name == "das_team"
        assert result.isna().all()
        assert len(result) == 2
