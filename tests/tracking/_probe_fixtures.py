"""Shared synthetic fixtures for the GK-substitution probe family (PR-1, ADR-037).

probe_frames()      -- two wide-area frames: carrier + attackers A2/A3, defenders B1/B2 + GK,
    ball (goal at x=105).
planted_model(kind) -- deterministic 'models' exposing predict_proba(feats_df) -> np.ndarray.
    All kinds carry a WEAK DENSE term over every Def/Off distance slot so ANY outfielder
    move yields a small nonzero delta -- without it, moving an attacker changes only
    OffDist_* (which a gk_r+DefDist_0-only model ignores), every placebo replicate
    median is exactly 0, placebo_p95 = 0, and the fail-closed no_valid_placebo guard
    fires for a fixture reason (self-verify + 4.46.0-session review, convergent).
    'mixed'    dense + 3.0*(30-GK_r)/30      # GK per-meter sensitivity DECISIVELY > controls
               # (1.5/30 vs 0.8/20 was only 1.25x -- a pass would have ridden a
               #  displacement-projection accident, not the planted property)
    'gk_blind' dense only                    # zero GK dependence: every GK move is a no-op
    'chiral'   dense + 0.9*GK_theta          # SIGNED term -> negates under the y-mirror.
               # ONLY this kind can detect chirality: GK_r/DefDist are MAGNITUDES,
               # y-mirror-INVARIANT (GOAL_Y=34 sits ON the mirror axis). A fingerprint
               # test built on 'mixed' would pass while proving nothing.
"""

import numpy as np
import pandas as pd


class _Planted:
    def __init__(self, kind: str):
        if kind not in ("mixed", "gk_blind", "chiral"):
            raise ValueError(f"unknown planted-model kind: {kind!r}")
        self.kind = kind
        self.carrier_params: dict = {}

    def predict_proba(self, feats):
        gk_r = feats["GK_r"].to_numpy(float) if "GK_r" in feats.columns else feats["gk_r"].to_numpy(float)
        dense_cols = [c for c in feats.columns if c.startswith(("DefDist_", "OffDist_"))] or (
            ["dist_nearest_def", "dist_nearest_teammate"] if "dist_nearest_def" in feats.columns else []
        )
        if not dense_cols:
            raise ValueError("planted model found no Def/Off distance columns in feats")
        dense = np.nansum([(20.0 - feats[c].to_numpy(float)) / 20.0 for c in dense_cols], axis=0)
        z = 0.05 + 0.1 * dense  # weak, dense: any outfielder move registers
        if self.kind == "mixed":
            z = z + 3.0 * (30.0 - gk_r) / 30.0
        elif self.kind == "chiral":
            th_col = "GK_theta" if "GK_theta" in feats.columns else "gk_theta"
            z = z + 0.9 * feats[th_col].to_numpy(float)
        return 1.0 / (1.0 + np.exp(-z))


def planted_model(kind: str) -> _Planted:
    return _Planted(kind)


def probe_frames() -> pd.DataFrame:
    """Two wide-area frames, ball near the left byline, carrier A1, attackers A2/A3,
    defenders B1/B2, a GK, ball row. Attacked goal at x=105 (GK near x~104)."""
    rows = []
    for fr, t in [(1, 40.0), (2, 40.4)]:
        rows += [
            dict(
                game_id="g",
                period_id=1,
                frame_id=fr,
                time_seconds=t,
                team_id="A",
                player_id="A1",
                x=96.0,
                y=8.0,
                vx=1.0,
                vy=0.0,
                is_ball=False,
                is_goalkeeper=False,
                ball_state="alive",
            ),
            dict(
                game_id="g",
                period_id=1,
                frame_id=fr,
                time_seconds=t,
                team_id="A",
                player_id="A2",
                x=99.0,
                y=34.0,
                vx=0.0,
                vy=0.0,
                is_ball=False,
                is_goalkeeper=False,
                ball_state="alive",
            ),
            dict(
                game_id="g",
                period_id=1,
                frame_id=fr,
                time_seconds=t,
                team_id="A",
                player_id="A3",
                x=92.0,
                y=26.0,
                vx=0.0,
                vy=0.0,
                is_ball=False,
                is_goalkeeper=False,
                ball_state="alive",
            ),
            dict(
                game_id="g",
                period_id=1,
                frame_id=fr,
                time_seconds=t,
                team_id="B",
                player_id="B1",
                x=100.0,
                y=20.0,
                vx=0.0,
                vy=0.0,
                is_ball=False,
                is_goalkeeper=False,
                ball_state="alive",
            ),
            dict(
                game_id="g",
                period_id=1,
                frame_id=fr,
                time_seconds=t,
                team_id="B",
                player_id="B2",
                x=98.0,
                y=12.0,
                vx=0.0,
                vy=0.0,
                is_ball=False,
                is_goalkeeper=False,
                ball_state="alive",
            ),
            dict(
                game_id="g",
                period_id=1,
                frame_id=fr,
                time_seconds=t,
                team_id="B",
                player_id="Bgk",
                x=104.0,
                y=34.0,
                vx=0.0,
                vy=0.0,
                is_ball=False,
                is_goalkeeper=True,
                ball_state="alive",
            ),
            dict(
                game_id="g",
                period_id=1,
                frame_id=fr,
                time_seconds=t,
                team_id="ball",
                player_id=None,
                x=96.0,
                y=8.0,
                vx=1.0,
                vy=0.0,
                is_ball=True,
                is_goalkeeper=False,
                ball_state="alive",
            ),
        ]
    return pd.DataFrame(rows)
