"""TF-58 Task 18: coordination output is invariant to team/player/game id dtype (ADR-019).

int, ``string`` and ``category`` ids must yield identical metric columns; key columns must be equal after
``canonical_id_series`` normalisation. Guards that the package routes every id through ``id_compat`` and never
compares raw ``==`` or joins unaligned id keys (spec §7.11).
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

import silly_kicks.coordination as C
from silly_kicks.coordination import CoordinationParams, compute_team_coordination
from silly_kicks.id_compat import canonical_id_series
from tests.coordination._fixtures import make_coordination_actions, make_coordination_match

pytestmark = [
    pytest.mark.filterwarnings("ignore:vx/vy columns not found"),
    pytest.mark.filterwarnings("ignore::silly_kicks.coordination.CoordinationCoverageWarning"),
]
_FAST = dataclasses.replace(CoordinationParams(), n_surrogates=5, welch_segment_s=20.0)
_ID_COLS = ("game_id", "team_id", "player_id")

# (result attribute, keys constant, metric-columns constant)
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


def _cast_ids(df: pd.DataFrame, dtype: str) -> pd.DataFrame:
    out = df.copy()
    for c in _ID_COLS:
        if c in out.columns:
            out[c] = out[c].astype(dtype)  # type: ignore[arg-type]  # "string"/"category" are valid pandas dtypes
    return out


def _canon_sort(df: pd.DataFrame, keys) -> pd.DataFrame:
    g = df.copy()
    for k in keys:
        if k in _KEY_ID_COLS:
            g[k] = canonical_id_series(g[k]).astype("string")
        else:
            g[k] = g[k].astype("string")
    return g.sort_values(list(keys)).reset_index(drop=True)


def _run(dtype: str | None):
    f = make_coordination_match(seconds=150.0, hz=10.0, provider="sportec")
    a = make_coordination_actions(f)
    if dtype is not None:
        f, a = _cast_ids(f, dtype), _cast_ids(a, dtype)
    return compute_team_coordination(f, actions=a, params=_FAST)


@pytest.mark.parametrize("dtype", ["string", "category"])
def test_metric_columns_and_keys_invariant_to_id_dtype(dtype):
    base = _run(None)
    other = _run(dtype)
    for attr, keys, metrics in _TABLES:
        b = _canon_sort(getattr(base, attr), keys)
        o = _canon_sort(getattr(other, attr), keys)
        assert len(b) == len(o), f"{attr}: row count changed under {dtype} ids ({len(b)} vs {len(o)})"
        # keys equal after canonicalisation
        pd.testing.assert_frame_equal(b[list(keys)], o[list(keys)], obj=f"{attr}.keys")
        # metric columns identical in value (dtype of the column itself is not asserted)
        present = [m for m in metrics if m in b.columns]
        pd.testing.assert_frame_equal(
            b[present], o[present], obj=f"{attr}.metrics", check_dtype=False, rtol=1e-9, atol=1e-9
        )
