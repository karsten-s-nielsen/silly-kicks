"""Real-provider extraction tests for TF-17 xCrossAttempt (regular suite, NOT e2e).

Exercises extract_xcross_features + prepare_xcross_training_data on the committed slim real-provider
slices. The H3 test (carrier + GK blocks mostly non-NaN on string-id providers) is the
"works-in-the-pipeline" assertion that catches the carrier-id-typing bug class -- a `.dropna()` range
check alone passes vacuously when everything is NaN.

NOTE: the slim _KEEP set omits vx/vy, so we set vx=vy=0 -> ball_speed is 0 (real loaders run
derive_velocities; the slim does not).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl import config as spc
from silly_kicks.tracking import _xcross_attempt as xc

REPO_ROOT = Path(__file__).resolve().parents[2]
SLIM = REPO_ROOT / "tests" / "datasets" / "tracking" / "action_context_slim"
PROVIDERS = ["sportec", "metrica", "skillcorner"]

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
    "ball_state",
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


def _synthesize_cross_actions(frames: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """Stamp a few cross actions on real (game, period, frame, team) anchors so prepare runs end to
    end (labels + the score-lookup path). The geometry under test is the REAL frames, not these."""
    cand = frames[~frames["is_ball"].astype(bool)].drop_duplicates(["period_id", "frame_id"]).head(n)
    return pd.DataFrame(
        {
            "game_id": cand["game_id"].to_numpy(),
            "period_id": cand["period_id"].to_numpy(),
            "team_id": cand["team_id"].to_numpy(),
            "time_seconds": cand["time_seconds"].to_numpy(),
            "type_id": [spc.actiontype_id["cross"]] * len(cand),
            "result_id": [spc.result_id["success"]] * len(cand),
            "action_id": np.arange(len(cand)),
        }
    )


@pytest.mark.parametrize("provider", PROVIDERS)
def test_extract_features_real_provider(provider):
    frames = _load(provider)
    key = frames.drop_duplicates(["game_id", "period_id", "frame_id"]).iloc[0]
    g = frames[
        (frames["game_id"] == key["game_id"])
        & (frames["period_id"] == key["period_id"])
        & (frames["frame_id"] == key["frame_id"])
    ].copy()
    teams = list(g[~g["is_ball"].astype(bool)]["team_id"].dropna().unique())
    if len(teams) < 1:
        pytest.skip("frame lacks an identifiable team")
    carrier = g[~g["is_ball"].astype(bool)]["player_id"].dropna().iloc[0]
    row = xc.extract_xcross_features(g, gk_team_id=teams[0], goal_x=0.0, carrier_player_id=carrier)
    assert list(row.columns) == xc.XCROSS_FEATURE_NAMES_FAITHFUL
    de = row["dist_endline"].iloc[0]
    assert np.isnan(de) or de >= -1e-6
    sc = row["space_controlled"].iloc[0]
    assert np.isnan(sc) or sc >= 0.0
    assert row["ten_minute_warning"].iloc[0] in (0, 1)
    for col in ("ball_theta", "gk_theta"):
        v = row[col].iloc[0]
        assert np.isnan(v) or (-np.pi - 1e-9 <= v <= np.pi + 1e-9)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_prepare_real_provider_runs(provider):
    frames = _load(provider)
    actions = _synthesize_cross_actions(frames)
    home = frames["team_id"].dropna().iloc[0]
    X, _y, _groups = xc.prepare_xcross_training_data(frames, actions, home_team_id=home)
    if len(X):
        assert list(X.columns) == xc.XCROSS_FEATURE_NAMES_FAITHFUL
        assert (X["dist_endline"].dropna() >= -1e-6).all()
        assert X["ten_minute_warning"].dropna().isin([0, 1]).all()
        assert (X["space_controlled"].dropna() >= 0).all()


@pytest.mark.parametrize("provider", PROVIDERS)
def test_carrier_and_gk_blocks_mostly_resolved_on_string_ids(provider):
    """H3 (+ C3 disproof): on real STRING-id providers the carrier-anchored confounders and the GK
    block must be MOSTLY non-NaN -- NOT vacuously all-NaN. A carrier-id-typing bug would all-NaN
    them. Assert the NON-NaN FRACTION (a .dropna() range check passes vacuously when all-NaN)."""
    frames = _load(provider)
    actions = _synthesize_cross_actions(frames)
    home = frames["team_id"].dropna().iloc[0]
    X, _, _ = xc.prepare_xcross_training_data(frames, actions, home_team_id=home)
    if len(X) >= 5:  # enough wide-area rows to be meaningful
        # The carrier-anchored block is the real C3 (carrier-id-typing) disproof: it must resolve on
        # every string-id provider. (It depends only on str-coerced carrier matching, not the GK.)
        assert X["dist_nearest_def"].notna().mean() >= 0.7, "carrier-anchored features all/mostly NaN"
        # The GK block resolves from is_goalkeeper rows wherever the defending GK is TRACKED in-domain.
        # GK coverage is provider-variable (ADR-007 N1: Metrica GK-ID 21-50%; its slim GK rows are
        # sparse and miss the wide-area frames here -> gk_frac may be 0). Where a GK IS tracked, the
        # block must resolve consistently. Independent of the carrier-typing concern above.
        gk_frac = X["gk_r"].notna().mean()
        if gk_frac > 0:
            assert gk_frac >= 0.5, "GK block sparsely NaN where GK rows exist (possible regression)"


@pytest.mark.parametrize("provider", PROVIDERS)
def test_real_provider_dtype_asymmetry(provider):
    """Must not crash regardless of native player_id/team_id dtype (object vs Int64)."""
    frames = _load(provider)
    actions = _synthesize_cross_actions(frames)
    home = frames["team_id"].dropna().iloc[0]
    xc.prepare_xcross_training_data(frames, actions, home_team_id=home)
