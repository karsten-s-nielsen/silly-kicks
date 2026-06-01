"""Real-provider extraction tests for TF-16 xShotOccurrence (B3).

The most important coverage: every TF-24 sweep bug this week (IDSSE game_id=None,
Gradient Sports 16x duplicate frames, coord conventions, dtype-asymmetric
player_id) surfaced only on REAL multi-provider data. These exercise
extract_xshot_features + compute_xshot_occurrence on the committed slim
real-provider fixtures, in the REGULAR suite (not e2e), with no trained weights.

NOTE (S3): the slim _KEEP set omits vx/vy, so we set vx=vy=0 -> `speed` is
degenerate in this real-data path. Extraction shape + bounds + dtype-asymmetry
(the important parts) are validated; velocity is exercised in unit tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import _xshot_occurrence as xs

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SLIM = REPO_ROOT / "tests" / "datasets" / "tracking" / "action_context_slim"
PROVIDERS = ["sportec", "metrica", "skillcorner", "pff"]

_KEEP = {
    "game_id",
    "period_id",
    "frame_id",
    "time_seconds",
    "frame_rate",
    "player_id",
    "team_id",
    "is_ball",
    "is_goalkeeper",
    "x",
    "y",
    "z",
    "speed",
    "speed_source",
    "ball_state",
    "team_attacking_direction",
    "confidence",
    "visibility",
    "source_provider",
}


def _load(provider: str) -> pd.DataFrame:
    p = SLIM / f"{provider}_slim.parquet"
    if not p.exists():
        pytest.skip(f"{p} not committed")
    df = pd.read_parquet(p)
    frames = df[df["__kind"] == "frame"].drop(columns=["__kind"]).reset_index(drop=True)
    frames = frames[[c for c in frames.columns if c in _KEEP]].copy()
    frames["vx"] = 0.0
    frames["vy"] = 0.0
    return frames


def _first_full_frame(frames: pd.DataFrame) -> pd.DataFrame:
    key = frames.drop_duplicates(["game_id", "period_id", "frame_id"]).iloc[0]
    return frames[
        (frames["game_id"] == key["game_id"])
        & (frames["period_id"] == key["period_id"])
        & (frames["frame_id"] == key["frame_id"])
    ].copy()


@pytest.mark.parametrize("provider", PROVIDERS)
def test_extract_features_real_provider(provider):
    frames = _load(provider)
    g = _first_full_frame(frames)
    teams = list(g[~g["is_ball"].astype(bool)]["team_id"].dropna().unique())
    if len(teams) < 1:
        pytest.skip("frame lacks an identifiable team")
    row = xs.extract_xshot_features(g, gk_team_id=teams[0], goal_x=0.0)
    assert list(row.columns) == xs.XSHOT_FEATURE_NAMES_FAITHFUL
    og = row["openGoal"].iloc[0]
    assert np.isnan(og) or (0.0 <= og <= 1.0)
    r = row["r"].iloc[0]
    assert np.isnan(r) or r >= 0.0
    # bearings in radian range
    for col in ("theta", "GK_theta"):
        v = row[col].iloc[0]
        assert np.isnan(v) or (-np.pi - 1e-9 <= v <= np.pi + 1e-9)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_compute_xshot_real_provider_in_bounds(provider):
    frames = _load(provider)
    # tiny in-test model trained on random features (no bundled weights needed)
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, 27)), columns=xs.XSHOT_FEATURE_NAMES_FAITHFUL)
    y = pd.Series((rng.random(60) < 0.1).astype(int))
    model = xs.XShotOccurrenceModel().fit(X, y)
    home = frames["team_id"].dropna().iloc[0]
    out = xs.compute_xshot_occurrence(frames, model=model, home_team_id=home)
    vals = out["xshot_occurrence"].dropna()
    assert vals.between(0.0, 1.0).all()


@pytest.mark.parametrize("tid_dtype", ["Int64", "object", "int64"])
def test_dtype_asymmetric_team_id(tid_dtype):
    # B3: provider team_id/player_id conventions differ -- Gradient Sports ships
    # nullable Int64, kloppy providers ship object, native int64. extract +
    # compute must tolerate all three without crashing or silently dropping.
    # (The pff slim fixture is not committed -- GS is synthetic-only per license --
    # so we exercise the dtype path directly with a synthetic frame.)
    g = _build_synthetic_canonical_frame(tid_dtype)
    teams = list(g[~g["is_ball"].astype(bool)]["team_id"].dropna().unique())
    row = xs.extract_xshot_features(g, gk_team_id=teams[0], goal_x=0.0)
    assert list(row.columns) == xs.XSHOT_FEATURE_NAMES_FAITHFUL

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 27)), columns=xs.XSHOT_FEATURE_NAMES_FAITHFUL)
    y = pd.Series((rng.random(40) < 0.1).astype(int))
    model = xs.XShotOccurrenceModel().fit(X, y)
    home = g["team_id"].dropna().iloc[0]
    out = xs.compute_xshot_occurrence(g, model=model, home_team_id=home)
    # The possessing team must receive at least one (in-bounds) value, proving the
    # join did NOT silently miss on the dtype.
    vals = out["xshot_occurrence"].dropna()
    assert len(vals) >= 1
    assert vals.between(0.0, 1.0).all()


def _build_synthetic_canonical_frame(tid_dtype: str) -> pd.DataFrame:
    """One canonical-schema frame: ball + GK + outfielders for two teams, with a
    chosen team_id dtype (Int64 / object / int64)."""
    rows = [
        dict(player_id=-1, team_id=-1, is_ball=True, is_goalkeeper=False, x=20.0, y=34.0),
        dict(player_id=10, team_id=1, is_ball=False, is_goalkeeper=True, x=2.0, y=34.0),
        dict(player_id=11, team_id=1, is_ball=False, is_goalkeeper=False, x=10.0, y=30.0),
        dict(player_id=12, team_id=1, is_ball=False, is_goalkeeper=False, x=12.0, y=38.0),
        dict(player_id=20, team_id=2, is_ball=False, is_goalkeeper=True, x=103.0, y=34.0),
        dict(player_id=21, team_id=2, is_ball=False, is_goalkeeper=False, x=20.3, y=34.0),
        dict(player_id=22, team_id=2, is_ball=False, is_goalkeeper=False, x=25.0, y=30.0),
    ]
    df = pd.DataFrame(rows)
    df["game_id"] = 1
    df["period_id"] = 1
    df["frame_id"] = 100
    df["time_seconds"] = 0.0
    df["frame_rate"] = 25.0
    df["z"] = 0.0
    df["vx"] = 0.0
    df["vy"] = 0.0
    df["ball_state"] = "alive"
    if tid_dtype == "object":
        df["team_id"] = df["team_id"].astype(str)
    else:
        df["team_id"] = df["team_id"].astype(tid_dtype)
    return df


def test_gradientsports_via_real_converter():
    # Gradient Sports has NO committed slim fixture (synthetic-only per license),
    # so the slim-fixture parametrizations skip it. But the GS path (Int64
    # identifiers + the real convert_to_frames converter) is exactly the
    # dtype-asymmetry surface that bit the TF-24 sweep -- so exercise it here via
    # the realistic GS raw fixture run through the ACTUAL converter, then
    # extract + compute xS. No committed canonical GS data; the raw fixture is
    # realistic synthetic optical data.
    from silly_kicks.tracking.gradientsports import convert_to_frames

    gs_dir = REPO_ROOT / "tests" / "datasets" / "tracking" / "gradientsports"
    raw = pd.read_parquet(gs_dir / "realistic.parquet")
    frames, _report = convert_to_frames(
        raw, home_team_id=100, home_team_start_left=True, output_convention="absolute_frame"
    )
    # GS canonical schema: Int64 identifiers (the asymmetry the join must handle).
    assert str(frames["team_id"].dtype) == "Int64"
    frames = frames.copy()
    frames["vx"] = 0.0
    frames["vy"] = 0.0

    # Extraction on a real converted GS frame.
    g = _first_full_frame(frames)
    teams = list(g[~g["is_ball"].astype(bool)]["team_id"].dropna().unique())
    assert teams, "converted GS frame should have identifiable teams"
    row = xs.extract_xshot_features(g, gk_team_id=teams[0], goal_x=0.0)
    og = row["openGoal"].iloc[0]
    assert np.isnan(og) or (0.0 <= og <= 1.0)

    # compute_xshot_occurrence end-to-end on the full converted GS match.
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, 27)), columns=xs.XSHOT_FEATURE_NAMES_FAITHFUL)
    y = pd.Series((rng.random(60) < 0.1).astype(int))
    model = xs.XShotOccurrenceModel().fit(X, y)
    home = frames["team_id"].dropna().iloc[0]
    out = xs.compute_xshot_occurrence(frames, model=model, home_team_id=home)
    vals = out["xshot_occurrence"].dropna()
    assert vals.between(0.0, 1.0).all()
    # The Int64 join must not silently miss: at least one frame scored.
    assert len(vals) >= 1


@pytest.mark.parametrize("provider", PROVIDERS)
def test_goal_relative_symmetry_real_data(provider):
    # Mirroring a real frame to the other end yields identical features.
    frames = _load(provider)
    g = _first_full_frame(frames)
    teams = list(g[~g["is_ball"].astype(bool)]["team_id"].dropna().unique())
    if len(teams) < 1:
        pytest.skip("frame lacks an identifiable team")
    mirrored = g.copy()
    mirrored["x"] = 105.0 - mirrored["x"]
    a = xs.extract_xshot_features(g, gk_team_id=teams[0], goal_x=0.0)
    b = xs.extract_xshot_features(mirrored, gk_team_id=teams[0], goal_x=105.0)
    pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-9)
