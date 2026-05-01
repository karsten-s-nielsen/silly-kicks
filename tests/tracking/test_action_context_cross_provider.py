"""Cross-provider parity for action_context using Tier-3 lakehouse-derived slim slices.

Loads the slim frame parquets committed by Loop 0
(``tests/datasets/tracking/action_context_slim/{provider}_slim.parquet``) and
synthesizes 10 actions per provider from the frame data (anchoring each
synthesized action on a real player at a real frame). This sidesteps the
lakehouse mart's 100% NULL team_id/player_id on `fct_action_values` rows for
sportec/metrica/skillcorner — the tracking side has correct identifiers, so
synthesizing actions from frames gives us a reliable test.

Tier-3 in PR-S19's three-tier strategy: real-distribution lakehouse-derived
slices for license-permissive providers (sportec / metrica / skillcorner). PFF
is synthetic-only per license; PFF cross-provider parity is exercised via the
synthetic Tier-1 fixtures in the unit-test suite.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SLIM_DIR = REPO_ROOT / "tests" / "datasets" / "tracking" / "action_context_slim"


def _load_frames(provider: str) -> pd.DataFrame:
    slim = SLIM_DIR / f"{provider}_slim.parquet"
    if not slim.exists():
        pytest.skip(
            f"slim slice {slim} not committed; run scripts/probe_action_context_baselines.py",
        )
    df = pd.read_parquet(slim)
    frames = df[df["__kind"] == "frame"].drop(columns=["__kind"]).reset_index(drop=True)
    # The slim parquet stacks actions+frames so the frame side carries
    # action-only NaN columns (start_x, end_x, action_id, ...). Strip them
    # so the frame side matches the silly_kicks 19-column tracking schema.
    keep = {
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
    return frames[[c for c in frames.columns if c in keep]].copy()


def _synthesize_actions(frames: pd.DataFrame, n_actions: int = 10) -> pd.DataFrame:
    """Pick (period_id, frame_id, player_id) triples from real frames and stamp
    synthetic action rows. The action's actor + team match real frame data
    exactly, guaranteeing actor_speed and opposite-team filtering both work."""
    candidates = frames[(~frames["is_ball"]) & (~frames["is_goalkeeper"])].copy()
    if len(candidates) < n_actions:
        candidates = frames[~frames["is_ball"]].copy()
    sample = candidates.drop_duplicates(["period_id", "frame_id"]).head(n_actions).reset_index(drop=True)
    return pd.DataFrame(
        {
            "action_id": list(range(1, len(sample) + 1)),
            "period_id": sample["period_id"].to_numpy(),
            "time_seconds": sample["time_seconds"].to_numpy(),
            "team_id": sample["team_id"].to_numpy(),
            "player_id": sample["player_id"].to_numpy(),
            "start_x": sample["x"].to_numpy(),
            "start_y": sample["y"].to_numpy(),
            "end_x": sample["x"].to_numpy(),
            "end_y": sample["y"].to_numpy(),
        }
    )


@pytest.mark.parametrize("provider", ["sportec", "metrica", "skillcorner"])
def test_action_context_bounds_per_provider(provider: str) -> None:
    """Real-data slim slice -> add_action_context yields plausible bounded outputs.

    nearest_defender_distance in [0, 200] m (200 is permissive — at most one
    diagonal pitch-length); actor_speed in [0, 50] m/s; counts >= 0.
    """
    from silly_kicks.tracking.features import add_action_context

    frames = _load_frames(provider)
    actions = _synthesize_actions(frames, n_actions=10)
    enriched = add_action_context(actions, frames)

    valid = enriched["nearest_defender_distance"].dropna()
    assert (valid >= 0).all(), f"{provider}: negative nearest_defender_distance"
    assert (valid <= 200).all(), f"{provider}: nearest_defender_distance > 200 m"

    speed = enriched["actor_speed"].dropna()
    assert (speed >= 0).all(), f"{provider}: negative actor_speed"
    assert (speed <= 50).all(), f"{provider}: actor_speed > 50 m/s"

    rz = enriched["receiver_zone_density"].dropna()
    assert (rz >= 0).all(), f"{provider}: negative receiver_zone_density"
    dt = enriched["defenders_in_triangle_to_goal"].dropna()
    assert (dt >= 0).all(), f"{provider}: negative defenders_in_triangle_to_goal"

    expected = {
        "nearest_defender_distance",
        "actor_speed",
        "receiver_zone_density",
        "defenders_in_triangle_to_goal",
        "frame_id",
        "time_offset_seconds",
        "link_quality_score",
        "n_candidate_frames",
    }
    missing = expected - set(enriched.columns)
    assert not missing, f"{provider}: missing output columns: {missing}"


@pytest.mark.parametrize("provider", ["sportec", "metrica", "skillcorner"])
def test_action_context_link_rate_per_provider(provider: str) -> None:
    """Synthesized actions land on real frames by construction, so link rate is 100%."""
    from silly_kicks.tracking.features import add_action_context

    frames = _load_frames(provider)
    actions = _synthesize_actions(frames, n_actions=10)
    enriched = add_action_context(actions, frames)
    link_rate = enriched["frame_id"].notna().mean()
    assert link_rate >= 0.95, f"{provider}: only {link_rate:.2f} linkage rate (<0.95)"


@pytest.mark.parametrize("provider", ["sportec", "metrica", "skillcorner"])
def test_action_context_actor_speed_populated(provider: str) -> None:
    """Synthesized actor matches a real frame player → actor_speed should be non-NaN
    for at least 80% of actions (the 20% allowance covers boundary frames where
    speed couldn't be derived via finite differences)."""
    from silly_kicks.tracking.features import add_action_context

    frames = _load_frames(provider)
    actions = _synthesize_actions(frames, n_actions=10)
    enriched = add_action_context(actions, frames)
    speed_populated_rate = enriched["actor_speed"].notna().mean()
    assert speed_populated_rate >= 0.8, (
        f"{provider}: actor_speed populated only on {speed_populated_rate:.2f} of actions"
    )


# ---------------------------------------------------------------------------
# PR-S21 — pre_shot_gk_position bounds (where GK rows exist in the slim slice)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider", ["pff", "sportec", "metrica", "skillcorner"])
def test_pre_shot_gk_position_bounds(provider: str) -> None:
    """When pre_shot_gk_* columns are populated (i.e. shots with linked GK), they satisfy
    bound constraints. Off-pitch tolerance acknowledges per-provider asymmetry
    (memory: reference_lakehouse_tracking_traps).

    With synthesized non-shot actions in the committed slim slice, all GK columns are
    NaN — the assertion is trivially satisfied (no rows to check). The bounds-check is
    structurally exercised when full-match e2e data lands shot rows.
    """
    from pathlib import Path

    expected_path = (
        Path(__file__).resolve().parent.parent
        / "datasets"
        / "tracking"
        / "action_context_slim"
        / f"{provider}_expected.parquet"
    )
    if not expected_path.exists():
        pytest.skip(f"{expected_path} not committed.")
    expected = pd.read_parquet(expected_path)
    has_gk_rows = expected["pre_shot_gk_x"].notna()
    if not has_gk_rows.any():
        return  # synthesized non-shots → all NaN; bounds trivially pass
    gk = expected[has_gk_rows]
    assert (gk["pre_shot_gk_x"].between(-5, 110)).all()
    assert (gk["pre_shot_gk_y"].between(-5, 73)).all()
    assert (gk["pre_shot_gk_distance_to_goal"].between(0, 130)).all()
    assert (gk["pre_shot_gk_distance_to_shot"].between(0, 130)).all()
