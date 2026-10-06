"""TF-58 Task 18: coordination is invariant to a full-pitch mirror and to team-id relabelling (C24, ADR-051).

Mirror: reflect every position (x -> 105 - x, y -> 68 - y) and swap the attacking directions. Because the
per-player series are reprojected into each team's goal-relative frame, every metric column must be unchanged
(angular means compared circularly). The non-vacuity leg monkeypatches the goal-relative transform to identity
so the neutralisation cannot happen -- then the mirror shifts ``coord_vc_mean_angle_deg`` by 180 degrees.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

import silly_kicks.coordination as C
import silly_kicks.coordination._signals as sig_mod
from silly_kicks.coordination import CoordinationParams, compute_team_coordination
from silly_kicks.id_compat import canonical_id_series
from tests.coordination._fixtures import make_coordination_match

pytestmark = [
    pytest.mark.filterwarnings("ignore:vx/vy columns not found"),
    pytest.mark.filterwarnings("ignore::silly_kicks.coordination.CoordinationCoverageWarning"),
]
_FAST = dataclasses.replace(CoordinationParams(), n_surrogates=5, welch_segment_s=20.0)
# Orientation invariance is a property of the deterministic geometry, not the stochastic (id-seeded) surrogate
# baseline. The surrogate rank columns (*_percentile) flip by a discrete step on ~1e-14 float noise from the
# y-reflection, so the mirror/non-vacuity legs disable surrogates and check the observed metrics.
_NO_SURR = dataclasses.replace(CoordinationParams(), n_surrogates=0, welch_segment_s=20.0)
_PITCH_X, _PITCH_Y = 105.0, 68.0

_TABLES = [
    ("pair", C.COORDINATION_PAIR_KEYS, C.COORDINATION_PAIR_METRIC_COLUMNS),
    ("pair_phase", C.COORDINATION_PAIR_PHASE_KEYS, C.COORDINATION_PAIR_PHASE_METRIC_COLUMNS),
    ("spectral", C.COORDINATION_SPECTRAL_KEYS, C.COORDINATION_SPECTRAL_METRIC_COLUMNS),
    ("cluster_team", C.COORDINATION_CLUSTER_TEAM_KEYS, C.COORDINATION_CLUSTER_TEAM_METRIC_COLUMNS),
    ("cluster_player", C.COORDINATION_CLUSTER_PLAYER_KEYS, C.COORDINATION_CLUSTER_PLAYER_METRIC_COLUMNS),
    ("team_sync", C.COORDINATION_TEAM_SYNC_KEYS, C.COORDINATION_TEAM_SYNC_METRIC_COLUMNS),
    ("rsi", C.COORDINATION_RSI_KEYS, C.COORDINATION_RSI_METRIC_COLUMNS),
]
_KEY_ID_COLS = {"game_id", "team_id", "player_id", "team_a_id", "team_b_id", "player_a_id", "player_b_id"}
_ANGULAR_MEAN = {"coord_rp_mean_deg", "coord_vc_mean_angle_deg", "coord_phi_mean_deg"}


def _mirror(frames: pd.DataFrame) -> pd.DataFrame:
    m = frames.copy()
    m["x"] = _PITCH_X - m["x"]
    m["y"] = _PITCH_Y - m["y"]
    swap = {"ltr": "rtl", "rtl": "ltr"}
    m["team_attacking_direction"] = m["team_attacking_direction"].map(lambda d: swap.get(d, d))
    return m


def _canon_sort(df: pd.DataFrame, keys) -> pd.DataFrame:
    g = df.copy()
    for k in keys:
        g[k] = (canonical_id_series(g[k]) if k in _KEY_ID_COLS else g[k]).astype("string")
    return g.sort_values(list(keys)).reset_index(drop=True)


# The mirror reflects y via 68 - y, so a player's reprojected position round-trips through two float
# subtractions and carries ~1e-13 noise. Most metrics stay < 1e-9, but coord_phi_sd_deg = sqrt(-2 ln rho_k)
# amplifies that near rho=1 to ~1.5e-6 deg. A real orientation defect is order 1 (metres) or ~180 deg (the
# defensive-line bug this gate caught), so 1e-5 cleanly separates float noise from any geometric error.
_MIRROR_ATOL = 1e-5


def _assert_metrics_equal(a: pd.DataFrame, b: pd.DataFrame, metrics, obj: str):
    for m in metrics:
        if m not in a.columns:
            continue
        va, vb = a[m].to_numpy(dtype=float), b[m].to_numpy(dtype=float)
        both_nan = np.isnan(va) & np.isnan(vb)
        if m in _ANGULAR_MEAN:
            diff = np.abs(((va - vb + 180.0) % 360.0) - 180.0)
            ok = both_nan | (diff <= _MIRROR_ATOL)
        else:
            ok = both_nan | np.isclose(va, vb, rtol=1e-6, atol=_MIRROR_ATOL)
        assert ok.all(), f"{obj}.{m}: mirror changed {int((~ok).sum())} value(s)"


def test_mirror_invariance():
    f = make_coordination_match(seconds=180.0, hz=10.0, provider="sportec", phase_offset_deg=40.0)
    base = compute_team_coordination(f, params=_NO_SURR)
    mir = compute_team_coordination(_mirror(f), params=_NO_SURR)
    for attr, keys, metrics in _TABLES:
        a = _canon_sort(getattr(base, attr), keys)
        b = _canon_sort(getattr(mir, attr), keys)
        assert len(a) == len(b), f"{attr}: row count changed under mirror"
        pd.testing.assert_frame_equal(a[list(keys)], b[list(keys)], obj=f"{attr}.keys")
        _assert_metrics_equal(a, b, metrics, attr)


def _is_surrogate(col: str) -> bool:
    # surrogate baselines are seeded per canonical id (deterministic), so they DEPEND on the labels by design.
    return col.endswith(("_surrogate_mean", "_percentile", "_excess"))


def test_identity_relabel_preserves_observed_metric_values():
    a = compute_team_coordination(make_coordination_match(seconds=150.0, team_ids=(1, 2)), params=_FAST)
    b = compute_team_coordination(make_coordination_match(seconds=150.0, team_ids=(7, 9)), params=_FAST)
    for attr, _keys, metrics in _TABLES:
        ta, tb = getattr(a, attr), getattr(b, attr)
        for m in metrics:
            if m not in ta.columns or _is_surrogate(m):
                continue
            va = np.sort(ta[m].to_numpy(dtype=float))
            vb = np.sort(tb[m].to_numpy(dtype=float))
            assert va.shape == vb.shape, f"{attr}.{m}: shape differs under relabel"
            both_nan = np.isnan(va) & np.isnan(vb)
            assert (both_nan | np.isclose(va, vb, rtol=1e-9, atol=1e-9, equal_nan=True)).all(), f"{attr}.{m} relabel"


def test_orientation_disabled_flips_vc_angle_by_180(monkeypatch):
    # Neuter the goal-relative reprojection: now the mirror is NOT undone, so a positional coupling angle flips.
    monkeypatch.setattr(sig_mod, "to_goal_relative_x_array", lambda arr, goal_x=None: np.asarray(arr))
    monkeypatch.setattr(sig_mod, "to_goal_relative_y_array", lambda arr, goal_x=None: np.asarray(arr))
    f = make_coordination_match(seconds=180.0, hz=10.0, provider="sportec", phase_offset_deg=40.0)
    base = compute_team_coordination(f, params=_NO_SURR)
    mir = compute_team_coordination(_mirror(f), params=_NO_SURR)

    def _vc_angle(res):
        p = res.pair
        row = p[(p.level == "team_team") & (p.signal_a == "centroid_x") & (p.window_kind == "period")]
        return float(row["coord_vc_mean_angle_deg"].iloc[0])

    diff = abs(((_vc_angle(base) - _vc_angle(mir) + 180.0) % 360.0) - 180.0)
    assert diff > 90.0, f"expected a ~180 deg flip with orientation disabled, got {diff:.1f} deg"
