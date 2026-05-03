"""Shared loader for slim-provider inputs (PR-S21 TF-11).

Used by the per-row regression test (CI) AND the regenerator script (manual). Keeping
one source of truth ensures the *_expected.parquet files and the regression gate test
read identical inputs — no Hyrum-Law drift between manual regeneration and CI assertion.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from silly_kicks.spadl import config as spadlconfig

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SLIM_DIR = REPO_ROOT / "tests" / "datasets" / "tracking" / "action_context_slim"
PFF_DIR = REPO_ROOT / "tests" / "datasets" / "tracking" / "pff"

N_ACTIONS_PER_PROVIDER = 10

_FRAME_KEEP_COLS = {
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


def load_provider_frames(provider: str) -> pd.DataFrame:
    """Load frames for a provider, projecting to silly-kicks tracking schema."""
    if provider == "pff":
        df = pd.read_parquet(PFF_DIR / "medium_halftime.parquet")
        return pd.DataFrame(
            {
                "game_id": df.get("game_id", 1),
                "period_id": df["period_id"],
                "frame_id": df["frame_id"],
                "time_seconds": df["time_seconds"],
                "frame_rate": df["frame_rate"],
                "player_id": df["player_id"],
                "team_id": df["team_id"],
                "is_ball": df["is_ball"],
                "is_goalkeeper": df["is_goalkeeper"],
                "x": df["x_centered"] + spadlconfig.field_length / 2.0,
                "y": df["y_centered"] + spadlconfig.field_width / 2.0,
                "z": df["z"],
                "speed": df["speed_native"],
                "speed_source": "native",
                "ball_state": df["ball_state"],
                "team_attacking_direction": "ltr",
                "confidence": pd.NA,
                "visibility": pd.NA,
                "source_provider": "pff",
            }
        )
    df = pd.read_parquet(SLIM_DIR / f"{provider}_slim.parquet")
    frames = df[df["__kind"] == "frame"].drop(columns=["__kind"]).reset_index(drop=True)
    return frames[[c for c in frames.columns if c in _FRAME_KEEP_COLS]].copy()


def synthesize_actions(frames: pd.DataFrame, n_actions: int = N_ACTIONS_PER_PROVIDER) -> pd.DataFrame:
    """Pick (period_id, frame_id, player_id) triples from real frames; stamp synthetic actions.

    Mirror of the synthesize step in scripts/regenerate_action_context_baselines.py.
    Anchors actions on real outfield (non-GK, non-ball) frame rows so actor_speed +
    opposite-team filtering both work.

    Action mix (PR-S21 TF-11): 8 passes + 1 keeper_save + 1 shot. The keeper_save (action 9)
    is performed by a real defending-team GK from the frames so the events-side
    ``add_pre_shot_gk_context`` call populates ``defending_gk_player_id`` from it; the shot
    (action 10) is the chronologically last action so the lookback window from the shot
    captures the keeper_save and the GK position can be measured at the linked frame.
    The shooter belongs to the OPPOSITE team from the keeper_save GK.

    The slim parquet input is one match's frame slice — we just pick a frame that has
    BOTH a goalkeeper row AND an outfield non-actor on the OPPOSITE team to anchor the
    shot on, then pick a slightly later frame's outfield player for the shooter.
    """
    candidates = frames[(~frames["is_ball"]) & (~frames["is_goalkeeper"])].copy()
    if len(candidates) < n_actions:
        candidates = frames[~frames["is_ball"]].copy()
    sample = candidates.drop_duplicates(["period_id", "frame_id"]).head(n_actions).reset_index(drop=True)
    pass_id = spadlconfig.actiontype_id["pass"]
    keeper_save_id = spadlconfig.actiontype_id["keeper_save"]
    shot_id = spadlconfig.actiontype_id["shot"]
    success_id = spadlconfig.result_id["success"]
    fail_id = spadlconfig.result_id["fail"]
    foot_id = spadlconfig.bodypart_id["foot"]
    other_id = spadlconfig.bodypart_id["other"]

    # Slot N-1 (keeper_save) — find a goalkeeper frame row in the same period as sample[-2]
    # whose team_id differs from the chosen shooter (sample.iloc[-1]).
    last_frame = sample.iloc[-1]
    shooter_team = last_frame["team_id"]
    gk_frames = frames[frames["is_goalkeeper"] & (~frames["is_ball"])].copy()
    # Defending GK = opposite team of shooter, same period.
    defending_gk = gk_frames[
        (gk_frames["team_id"] != shooter_team) & (gk_frames["period_id"] == last_frame["period_id"])
    ].copy()
    if len(defending_gk) == 0:
        # No defending GK in same period → fall back to any GK on opposite team.
        defending_gk = gk_frames[gk_frames["team_id"] != shooter_team].copy()
    if len(defending_gk) == 0:
        # No opposite-team GK at all → fall back to any GK (shooter team).
        defending_gk = gk_frames.copy()
    if len(defending_gk) == 0:
        msg = "No goalkeeper rows in frames; cannot synthesize keeper_save action for GK validation."
        raise ValueError(msg)

    # Pick a GK frame BEFORE the shooter's frame_id so the keeper_save predates the shot.
    pre_shot_gk = defending_gk[defending_gk["frame_id"] < last_frame["frame_id"]]
    if len(pre_shot_gk) == 0:
        # Pick any GK frame — we'll fix up the time_seconds to be a few seconds before the shot.
        pre_shot_gk = defending_gk
    gk_row = pre_shot_gk.iloc[len(pre_shot_gk) // 2]

    # Build the action stream: indices 0..n-3 passes; n-2 keeper_save; n-1 shot.
    n_passes = len(sample) - 2
    pass_rows = sample.iloc[:n_passes]
    save_time = min(float(gk_row["time_seconds"]), float(last_frame["time_seconds"]) - 2.0)
    shot_time = float(last_frame["time_seconds"])

    return pd.DataFrame(
        {
            "game_id": [1] * len(sample),
            "original_event_id": [str(i + 1) for i in range(len(sample))],
            "action_id": list(range(1, len(sample) + 1)),
            "period_id": [
                *pass_rows["period_id"].to_numpy(),
                int(gk_row["period_id"]),
                int(last_frame["period_id"]),
            ],
            "time_seconds": [
                *pass_rows["time_seconds"].to_numpy(),
                save_time,
                shot_time,
            ],
            "team_id": [
                *pass_rows["team_id"].to_numpy(),
                gk_row["team_id"],
                last_frame["team_id"],
            ],
            "player_id": [
                *pass_rows["player_id"].to_numpy(),
                gk_row["player_id"],
                last_frame["player_id"],
            ],
            "start_x": [
                *pass_rows["x"].to_numpy(),
                float(gk_row["x"]),
                float(last_frame["x"]),
            ],
            "start_y": [
                *pass_rows["y"].to_numpy(),
                float(gk_row["y"]),
                float(last_frame["y"]),
            ],
            "end_x": [
                *pass_rows["x"].to_numpy(),
                float(gk_row["x"]),
                float(last_frame["x"]),
            ],
            "end_y": [
                *pass_rows["y"].to_numpy(),
                float(gk_row["y"]),
                float(last_frame["y"]),
            ],
            "type_id": [pass_id] * n_passes + [keeper_save_id, shot_id],
            "result_id": [success_id] * n_passes + [success_id, fail_id],
            "bodypart_id": [foot_id] * n_passes + [other_id, foot_id],
        }
    )


def synthesize_actions_per_period_dense(frames: pd.DataFrame, n_per_period: int = 5) -> pd.DataFrame:
    """Synthesize actions ensuring >=1 shot + >=1 keeper_save per period in ``frames``.

    PR-S24 fixture-density gate (Loop 4 Step 4.5 -- closes the silent vacuous-pass
    failure mode for TF-12 per-period DOP-symmetry invariants). For every period
    present in ``frames``, runs :func:`synthesize_actions` on the period's frame
    slice, then concatenates and renumbers action_id. Each period contributes
    ``n_per_period - 2`` passes + 1 keeper_save + 1 shot.

    Periods that lack a defending GK frame OR enough non-ball / non-GK rows are
    skipped (a period with no candidates cannot be synthesized at all). The
    density gate test in ``tests/tracking/test_synthesizer_fixture_density.py``
    explicitly asserts the surviving period set is non-empty AND every surviving
    period has the required action mix.

    Returns a DataFrame with the same column shape as ``synthesize_actions``;
    ``action_id`` and ``original_event_id`` are renumbered globally.
    """
    parts: list[pd.DataFrame] = []
    periods = sorted(frames["period_id"].dropna().unique())
    for period in periods:
        period_frames = frames[frames["period_id"] == period]
        non_ball_non_gk = period_frames[(~period_frames["is_ball"]) & (~period_frames["is_goalkeeper"])]
        if non_ball_non_gk.empty:
            continue
        gk_frames = period_frames[period_frames["is_goalkeeper"] & (~period_frames["is_ball"])]
        if gk_frames.empty:
            continue
        try:
            actions_p = synthesize_actions(period_frames, n_actions=n_per_period)
        except ValueError:
            continue
        parts.append(actions_p)
    if not parts:
        msg = "synthesize_actions_per_period_dense: no period had sufficient frame coverage to synthesize."
        raise ValueError(msg)
    out = pd.concat(parts, ignore_index=True)
    out["action_id"] = list(range(1, len(out) + 1))
    out["original_event_id"] = [str(i + 1) for i in range(len(out))]
    return out
