"""TF-46 corners-own liveness gate (C2, spec §10.3).

The tracking liveness registry (test_aggregator_column_liveness.py) is pinned to ``tracking.__all__`` and
corners is not in it, so this family carries its OWN liveness gate. Mandate: no emitted metric column is
100%-null on an in-domain fixture, and every float metric column with >=2 observations carries >1 distinct
value (for the fixed-width chance-quality mart); the data-driven encoder / routine outputs are checked for
non-vacuity (not all-null, >1 distinct value overall) rather than per-column non-constancy (a sparse
run/feature column is legitimately zero).

COMPLETENESS (C2): ``test_meta_every_exported_emitter_is_covered`` enforces REVERSE inclusion -- every
public corner emitter (a ``corners.__all__`` callable whose name starts with add_/compute_/cluster_/
aggregate_) must have a liveness case here, else CI reds. A new emitter cannot escape coverage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import silly_kicks.corners as corners_pkg
from silly_kicks.corners import (
    CornerRoutineModel,
    RoutineParams,
    RunZoneModel,
    add_corner_windows,
    aggregate_chance_quality_by_feature,
    cluster_corner_routines,
    compute_corner_chance_quality,
    compute_corner_routine_features,
    compute_corner_run_encoding,
    identify_corners,
)

_CORNER_CROSSED = 5
_SHOT = 11

#: A public corner EMITTER is a corners.__all__ callable whose name starts with one of these.
_EMITTER_PREFIXES = ("add_", "compute_", "cluster_", "aggregate_")

#: Every exported emitter must appear here (the per-emitter liveness case names below mirror it).
_COVERED = frozenset(
    {
        "add_corner_windows",
        "compute_corner_run_encoding",
        "compute_corner_routine_features",
        "cluster_corner_routines",
        "compute_corner_chance_quality",
        "aggregate_chance_quality_by_feature",
    }
)


def _exported_emitters() -> set[str]:
    return {n for n in corners_pkg.__all__ if n.startswith(_EMITTER_PREFIXES)}


def test_meta_every_exported_emitter_is_covered():
    """REVERSE inclusion (C2): a new exported emitter not given a liveness case reds this test."""
    uncovered = _exported_emitters() - _COVERED
    assert not uncovered, f"corner emitter(s) exported but NOT liveness-covered (spec §10.3 C2): {sorted(uncovered)}"
    stale = _COVERED - _exported_emitters()
    assert not stale, f"liveness case(s) for a non-exported emitter (rename/removed?): {sorted(stale)}"
    # non-vacuity: the discovery actually finds the known emitters (not an empty/typo'd pattern).
    assert {"compute_corner_chance_quality", "add_corner_windows"} <= _exported_emitters()


# --- toy models (fit in-test; no bundled weights -- the emitters are liveness-checkable now) --------
def _toy_zone_model() -> RunZoneModel:
    rng = np.random.default_rng(0)
    init = np.vstack([rng.normal([90.0, 25.0], 1.0, size=(60, 2)), rng.normal([90.0, 43.0], 1.0, size=(60, 2))])
    target = np.vstack(
        [
            rng.normal([100.0, 28.0], 1.0, size=(40, 2)),
            rng.normal([100.0, 40.0], 1.0, size=(40, 2)),
            rng.normal([86.0, 34.0], 1.0, size=(40, 2)),  # top-of-box / second-ball edge (in-region)
        ]
    )
    return RunZoneModel().fit(init, target, init_k=2, target_k=3)


def _two_corners_windowed():
    """Two in-domain corners (distinct player geometry) with resolving contacts + GI."""
    acts = pd.DataFrame(
        {
            "game_id": ["g1", "g1"],
            "period_id": [1, 1],
            "time_seconds": [100.0, 200.0],
            "team_id": ["tA", "tA"],
            "player_id": ["p1", "p2"],
            "start_x": [104.5, 104.5],
            "start_y": [1.0, 1.0],
            "type_id": [_CORNER_CROSSED, _CORNER_CROSSED],
            "result_id": [1, 1],
            "bodypart_id": [0, 0],
        }
    )
    corners = identify_corners(acts, provider="skillcorner")
    frames = pd.DataFrame(
        {
            "frame_id": [5, 9],
            "period_id": [1, 1],
            "time_seconds": [100.5, 200.7],
            "player_id": ["pW", "pX"],
            "team_id": ["tW", "tX"],
        }
    )
    contacts = pd.DataFrame(
        {
            "period_id": [1, 1],
            "frame_id": [5, 9],
            "time_seconds": [100.5, 200.7],
            "player_id": ["pW", "pX"],
            "contact_source": pd.Series(["velocity_break", "rdp"], dtype="object"),
        }
    )
    gi = pd.DataFrame({"game_id": ["g1", "g1"], "period_id": [1, 1], "time_seconds": [100.0, 200.0]})
    return corners, frames, contacts, gi


def _encoding_frames():
    """Two corners whose attacking players run to the two active zones (non-vacuous encoding)."""
    corners = pd.DataFrame(
        {
            "game_id": ["g1", "g1"],
            "corner_id": [0, 1],
            "period_id": [1, 1],
            "attacking_team_id": ["tA", "tA"],
            "corner_side": pd.Series(["left", "left"], dtype="category"),
            "corner_kind": pd.Series(["direct", "direct"], dtype="category"),
            "t_init": [98.0, 198.0],
            "t_target": [101.5, 201.5],
        }
    )

    def block(fid, t, period, xy):
        return pd.DataFrame(
            {
                "frame_id": [fid, fid],
                "period_id": [period, period],
                "time_seconds": [t, t],
                "team_id": ["tA", "tA"],
                "player_id": ["pa", "pb"],
                "x": [xy[0][0], xy[1][0]],
                "y": [xy[0][1], xy[1][1]],
                "is_ball": [False, False],
            }
        )

    frames = pd.concat(
        [
            block(1, 98.0, 1, [(90.0, 25.0), (90.0, 43.0)]),
            block(2, 101.5, 1, [(100.0, 28.0), (100.0, 40.0)]),
            block(3, 198.0, 1, [(90.0, 43.0), (90.0, 25.0)]),
            block(4, 201.5, 1, [(100.0, 40.0), (100.0, 28.0)]),
        ],
        ignore_index=True,
    )
    return corners, frames


def _chance_inputs():
    corners = pd.DataFrame(
        {
            "game_id": ["g1", "g1"],
            "corner_id": [0, 1],
            "period_id": [1, 1],
            "t_kick": [100.0, 200.0],
            "t_outcome": [106.0, 206.0],
        }
    )
    actions = pd.DataFrame(
        {
            "game_id": ["g1", "g1", "g1"],
            "period_id": [1, 1, 1],
            "time_seconds": [101.0, 103.0, 201.0],
            "type_id": [_SHOT, _SHOT, _SHOT],
            "result_id": [1, 0, 0],
            "xg": [0.5, 0.2, 0.1],
        }
    )
    return corners, actions


# --- fixture precondition (ADR-032 idiom) ----------------------------------------------------------
def test_fixture_precondition_in_domain():
    corners, _, _, _ = _two_corners_windowed()
    assert len(corners) >= 2
    assert corners["t_kick"].nunique() >= 2  # not a degenerate constant-time fixture


# --- per-emitter liveness --------------------------------------------------------------------------
def test_add_corner_windows_live():
    corners, frames, contacts, gi = _two_corners_windowed()
    out = add_corner_windows(corners, frames, contacts, gi=gi)
    for col in ("t_init", "t_target", "t_outcome", "first_contact_time", "first_possession_time"):
        assert out[col].notna().all(), f"{col} 100%-null on the in-domain fixture"
        assert out[col].nunique() > 1, f"{col} constant across >=2 corners (dead)"
    for col in ("winner_team_id", "winner_player_id", "target_snapshot_source"):
        assert out[col].notna().all(), f"{col} 100%-null on the in-domain fixture"


def test_run_encoding_live():
    corners, frames = _encoding_frames()
    enc = compute_corner_run_encoding(corners, frames, model=_toy_zone_model())
    run_cols = [c for c in enc.columns if c.startswith("run_")]
    vals = enc[run_cols].to_numpy(dtype=float)
    assert not np.isnan(vals).all(), "run-encoding is 100%-null"
    assert np.unique(vals[~np.isnan(vals)]).size > 1, "run-encoding is vacuous (one distinct value)"


def test_routine_features_live():
    corners, frames = _encoding_frames()
    enc = compute_corner_run_encoding(corners, frames, model=_toy_zone_model())
    run_cols = [c for c in enc.columns if c.startswith("run_")]
    term = enc[run_cols].to_numpy(dtype=float)
    model = CornerRoutineModel().fit(term, rank=2, alpha=0.0)
    feats = compute_corner_routine_features(enc, model=model)
    feat_cols = [c for c in feats.columns if c.startswith("routine_feat_")]
    vals = feats[feat_cols].to_numpy(dtype=float)
    assert not np.isnan(vals).all(), "routine features 100%-null"
    assert np.unique(vals[~np.isnan(vals)]).size > 1, "routine features vacuous"


def test_cluster_live():
    feats = pd.DataFrame(
        {
            "game_id": ["g1"] * 6,
            "corner_id": list(range(6)),
            "routine_feat_0": [5.0, 5.1, 4.9, 0.1, 0.0, 0.2],
            "routine_feat_1": [0.1, 0.0, 0.2, 5.0, 5.1, 4.9],
        }
    )
    out = cluster_corner_routines(feats, params=RoutineParams(n_clusters=2))
    assert out["cluster_id"].notna().all()
    assert out["cluster_id"].nunique() > 1  # resolves >1 routine cluster


def test_chance_quality_live():
    corners, actions = _chance_inputs()
    out = compute_corner_chance_quality(corners, actions, xg_column="xg")
    for col in ("n_shots", "n_goals", "sum_xg"):
        assert out[col].notna().all(), f"{col} 100%-null"
    assert out["sum_xg"].nunique() > 1, "sum_xg constant across >=2 corners (dead metric column)"


def test_aggregate_live():
    corners, actions = _chance_inputs()
    quality = compute_corner_chance_quality(corners, actions, xg_column="xg")
    feats = pd.DataFrame(
        {"game_id": ["g1", "g1"], "corner_id": [0, 1], "routine_feat_0": [1.0, 0.0], "routine_feat_1": [0.0, 1.0]}
    )
    out = aggregate_chance_quality_by_feature(feats, quality)
    assert len(out) == 2  # one row per routine feature
    assert out["weighted_mean_sum_xg"].notna().all(), "per-feature weighted chance quality 100%-null"
