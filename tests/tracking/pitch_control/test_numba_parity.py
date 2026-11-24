"""Golden-master tests: numba kernels produce identical output to NumPy.

numba is a test dependency (pyproject.toml [test] extra) — these always run in CI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking.pitch_control._batch_numba_kernels import influence_rows_numba
from silly_kicks.tracking.pitch_control._grids import pitch_grid
from silly_kicks.tracking.pitch_control._numba_kernels import (
    gaussian_influence_numba,
    influence_numba,
    tti_numba,
)
from silly_kicks.tracking.pitch_control._params import SpearmanParams
from silly_kicks.tracking.pitch_control._spearman import _compute_influence, compute_tti
from silly_kicks.tracking.pitch_control._spearman_batch import _influence_rows
from tests._prod_fixtures import make_production_frames


class TestTTIParity:
    def test_fixed_seed_parity(self):
        rng = np.random.default_rng(42)
        pos = rng.uniform(0, 105, (22, 2))
        vel = rng.uniform(-5, 5, (22, 2))
        targets = np.column_stack(
            [
                np.linspace(0, 105, 50).repeat(32),
                np.tile(np.linspace(0, 68, 32), 50),
            ]
        )
        numpy_out = compute_tti(pos, vel, targets, 0.7, 7.0)
        numba_out = tti_numba(pos, vel, targets, 0.7, 7.0)
        np.testing.assert_allclose(numpy_out, numba_out, rtol=1e-12)


class TestInfluenceParity:
    def test_fixed_seed_parity(self):
        rng = np.random.default_rng(123)
        team_tti = rng.uniform(0.5, 5.0, (11, 1600))
        opp_min = rng.uniform(0.5, 5.0, (1600,))
        numpy_out = _compute_influence(team_tti, opp_min, 0.45)
        numba_out = influence_numba(team_tti, opp_min, 0.45)
        np.testing.assert_allclose(numpy_out, numba_out, rtol=1e-12)


class TestGaussianInfluenceParity:
    def test_fixed_seed_parity(self):
        from silly_kicks.tracking.pitch_control._fernandez_bornn import _compute_gaussian_influence

        rng = np.random.default_rng(456)
        targets = rng.uniform(0, 105, (1600, 2))
        mu = rng.uniform(20, 80, (11, 2))
        # Build per-player inverse covariance and determinants from random PD matrices
        inv_cov = np.zeros((11, 2, 2))
        det_cov = np.zeros(11)
        for i in range(11):
            A = rng.standard_normal((2, 2))
            cov = A @ A.T + 0.1 * np.eye(2)
            inv_cov[i] = np.linalg.inv(cov)
            det_cov[i] = np.linalg.det(cov)
        numpy_out = _compute_gaussian_influence(targets, mu, inv_cov, det_cov)
        numba_out = gaussian_influence_numba(targets, mu, inv_cov, det_cov)
        np.testing.assert_allclose(numpy_out, numba_out, rtol=1e-10)


# ---- W4: the backend parity record on production-shaped frames (spec 2026-11-08 tracking-batch section 4.4) --------
#
# Each test compares the numba kernel with the NUMPY reference. The reference is forced with the PITCH_CONTROL switch:
# ``compute_tti``, ``_compute_influence``, and ``_influence_rows`` dispatch to numba themselves when it is on. The rule
# is today's per-element ``assert_allclose(numba, numpy, rtol=1e-12)``; each test prints its measured max abs diff
# (run with ``-s``) for the memo. The corpus diff per kernel and CPU class comes from the W0 ``kernel_diff.json``.

_SWITCH = "SILLY_KICKS_PITCH_CONTROL_FORCE_NUMPY"
_PARAMS = SpearmanParams()


def _production_players(n_frames: int = 40) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """``(pos, vel, team_code)`` per frame of the production fixture: 22 players, gaps, NaN detections, float32
    coordinates upcast to float64, seeded velocities with a NaN share (a NaN velocity is 0.0, as in the index)."""
    frames = make_production_frames(n_frames=n_frames, n_periods=2)
    rng = np.random.default_rng(20261111)
    vx, vy = rng.normal(0.0, 2.5, len(frames)), rng.normal(0.0, 2.5, len(frames))
    vx[rng.random(len(frames)) < 0.02] = np.nan
    frames = frames.assign(vx=vx, vy=vy)
    out = []
    for _key, frame in frames.groupby(["period_id", "frame_id"], sort=True):
        players = frame[~frame["is_ball"].to_numpy(dtype=bool)].dropna(subset=["x", "y"])
        if players.empty:
            continue
        pos = players[["x", "y"]].to_numpy(dtype=np.float64)
        vel = np.nan_to_num(players[["vx", "vy"]].to_numpy(dtype=np.float64), nan=0.0)
        team_code = pd.factorize(players["team_id"].to_numpy())[0]
        out.append((pos, vel, team_code))
    assert len(out) >= 30 and all(len(set(t.tolist())) == 2 for _p, _v, t in out)  # precondition: two teams
    return out


def _targets() -> np.ndarray:
    return pitch_grid(_PARAMS.grid_cells_x, _PARAMS.grid_cells_y)[2]


def _numpy_tti(monkeypatch, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    with monkeypatch.context() as m:
        m.setenv(_SWITCH, "1")
        return compute_tti(pos, vel, _targets(), _PARAMS.reaction_time, _PARAMS.max_acceleration)


def _team_splits(tti: np.ndarray, team_code: np.ndarray):
    """``(team_tti, opponent_min_tti)`` for each of the two teams."""
    for team in (0, 1):
        yield tti[team_code == team], tti[team_code != team].min(axis=0)


def _report(name: str, diffs: list[float], n: int) -> None:
    print(f"[W4 parity] {name}: max_abs_diff={max(diffs):.3e} over {n} production frames")


def test_tti_parity_production(monkeypatch):
    frames, diffs = _production_players(), []
    for pos, vel, _team in frames:
        numpy_out = _numpy_tti(monkeypatch, pos, vel)
        numba_out = tti_numba(pos, vel, _targets(), _PARAMS.reaction_time, _PARAMS.max_acceleration)
        np.testing.assert_allclose(numba_out, numpy_out, rtol=1e-12)
        diffs.append(float(np.max(np.abs(numba_out - numpy_out))))
    _report("tti_numba", diffs, len(frames))


def test_influence_parity_production(monkeypatch):
    frames, diffs = _production_players(), []
    for pos, vel, team_code in frames:
        for team_tti, opp_min in _team_splits(_numpy_tti(monkeypatch, pos, vel), team_code):
            with monkeypatch.context() as m:
                m.setenv(_SWITCH, "1")
                numpy_out = _compute_influence(team_tti, opp_min, _PARAMS.sigma)
            numba_out = influence_numba(team_tti, opp_min, _PARAMS.sigma)
            np.testing.assert_allclose(numba_out, numpy_out, rtol=1e-12)
            diffs.append(float(np.max(np.abs(numba_out - numpy_out))))
    _report("influence_numba", diffs, len(frames))


def test_influence_rows_parity_production(monkeypatch):
    # the batch core's elementwise kernel: every player row of a frame, each with its own team's opponent minimum
    frames, diffs = _production_players(), []
    for pos, vel, team_code in frames:
        splits = list(_team_splits(_numpy_tti(monkeypatch, pos, vel), team_code))
        tti_rows = np.ascontiguousarray(np.concatenate([t for t, _o in splits]))
        opp_rows = np.ascontiguousarray(np.concatenate([np.broadcast_to(o, t.shape) for t, o in splits]))
        with monkeypatch.context() as m:
            m.setenv(_SWITCH, "1")
            numpy_out = _influence_rows(tti_rows, opp_rows, _PARAMS.sigma)
        numba_out = influence_rows_numba(tti_rows, opp_rows, _PARAMS.sigma)
        np.testing.assert_allclose(numba_out, numpy_out, rtol=1e-12)
        diffs.append(float(np.max(np.abs(numba_out - numpy_out))))
    _report("influence_rows_numba", diffs, len(frames))


@pytest.mark.parametrize("distance", [0.0, 5e-11])
def test_tti_at_target_branch(monkeypatch, distance):
    # distance == 0 and 0 < distance < 1e-10 take the at-target branch on both backends: TTI is the reaction time
    pos = np.array([[0.0, 0.0]])  # at the origin the offset is exact (at x = 30, 5e-11 rounds to 5.00009e-11)
    vel = np.array([[3.0, -1.5]])
    target = np.array([[distance, 0.0]])
    assert target[0, 0] - pos[0, 0] == distance  # precondition: the offset is the parametrized distance
    with monkeypatch.context() as m:
        m.setenv(_SWITCH, "1")
        numpy_out = compute_tti(pos, vel, target, _PARAMS.reaction_time, _PARAMS.max_acceleration)
    numba_out = tti_numba(pos, vel, target, _PARAMS.reaction_time, _PARAMS.max_acceleration)
    assert numpy_out[0, 0] == numba_out[0, 0] == _PARAMS.reaction_time
