"""Shared toy-model + fixture helpers for the TF-60 PR3 ghost-GK re-fit tests.

Underscore-prefixed so pytest does not collect it; imported by the ghost-refit test modules via
``from ._ghost_toy import ...`` (the established tests/tracking sibling-helper pattern).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from silly_kicks.tracking._ghost_gk import (
    GHOST_GK_FEATURE_NAMES,
    GHOST_GK_FEATURE_NAMES_POSITION_ONLY,
    GhostGkModel,
)


def fit_toy(model: GhostGkModel, *, n: int = 300, seed: int = 0) -> GhostGkModel:
    """Fit a tiny HGBR so ``save``/``predict_*`` have trees. Random features + in-grid labels.

    For PLUMBING tests (round-trip, density guard) — the values carry no signal, only shape. The
    two-sided saturation gate does NOT use this; it trains on real fixture-derived data so the model
    learns the ball_x -> gk_x relationship.
    """
    names = GHOST_GK_FEATURE_NAMES_POSITION_ONLY if model.feature_set == "position_only" else GHOST_GK_FEATURE_NAMES
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(names))), columns=names)
    g = model.grid_spec
    labels = pd.DataFrame(
        {
            "gk_x": rng.uniform(g.x_min, g.x_max, size=n),
            "gk_y": rng.uniform(g.y_min, g.y_max, size=n),
        }
    )
    model.fit(X, labels)
    return model


def rewrite_sums(artifact_dir: Path) -> None:
    """Recompute SHA256SUMS after a metadata.json edit (mirrors GhostGkModel.save's hashing)."""
    sums_path = artifact_dir / "SHA256SUMS"
    lines = []
    for fname in ["rfcde_weights.npz", "metadata.json"]:
        raw = (artifact_dir / fname).read_bytes()
        if fname.endswith(".json"):
            raw = raw.replace(b"\r\n", b"\n")
        lines.append(f"{hashlib.sha256(raw).hexdigest()}  {fname}\n")
    sums_path.write_text("".join(lines), newline="\n")


# --- frame helpers -------------------------------------------------------------------------------
_SPORTEC_SLIM = "tests/datasets/tracking/action_context_slim/sportec_slim.parquet"


def load_sportec_slim_frames() -> pd.DataFrame:
    """The committed real full-tracking slice (frame rows only), as the §9 finding probe uses."""
    df = pd.read_parquet(_SPORTEC_SLIM)
    df = df[df["__kind"] == "frame"].copy()
    for c in ("is_ball", "is_goalkeeper"):
        df[c] = df[c].astype("boolean").fillna(False)
    return df.reset_index(drop=True)


def home_defending_x0(frames: pd.DataFrame):
    """The team that DEFENDS x=0 in period 1 (LTR home), via the GoalMap (ADR-055, never identity)."""
    from silly_kicks.tracking import resolve_defended_goals

    gmap = resolve_defended_goals(frames)
    return next(t for (g, p, t), e in gmap.resolved.items() if str(p) == "1" and float(e) == 0.0)


def translated_training_set(base: pd.DataFrame) -> pd.DataFrame:
    """Stack upfield translations of `base` so a toy model sees high-sweeper (gr_x>30) labels.

    Each translation is a distinct game_id (so prepare treats them independently); velocity-bearing
    (smooth + derive) so the FAITHFUL feature set is finite.
    """
    from silly_kicks.tracking import derive_velocities
    from silly_kicks.tracking.preprocess import smooth_frames

    parts = []
    for k, delta in enumerate((0, 8, 16, 24, 32)):
        f = base.copy()
        f["x"] = np.clip(f["x"].to_numpy(dtype=float) + delta, 0.0, 105.0)
        f["game_id"] = f["game_id"].astype(str) + f"_t{k}"
        parts.append(derive_velocities(smooth_frames(f)))
    return pd.concat(parts, ignore_index=True)


def two_team_frames(velocity: bool = True, *, n_frames: int = 3) -> pd.DataFrame:
    """Minimal valid 2-team frame set (H defends x=0, A defends x=105) for resolver / mean-path tests.

    ``velocity=False`` declares ``speed_source`` structurally unavailable on every row (the SB360
    freeze-frame shape) so ``variant_key_for_velocity`` -> ``position_only``; ``velocity=True`` stamps
    real ``vx``/``vy`` -> ``default``.
    """
    from silly_kicks.tracking.schema import SPEED_SOURCE_UNAVAILABLE

    rows = []
    for fi in range(n_frames):
        for team, gk_x in (("H", 5.0), ("A", 100.0)):
            for j in range(11):
                is_gk = j == 0
                x = gk_x if is_gk else (20.0 + 6.0 * j if team == "H" else 85.0 - 6.0 * j)
                rows.append(
                    {
                        "game_id": "g",
                        "period_id": 1,
                        "frame_id": 1000 + fi,
                        "time_seconds": float(fi),
                        "player_id": f"{team}{j}",
                        "team_id": team,
                        "is_ball": False,
                        "is_goalkeeper": is_gk,
                        "x": x,
                        "y": 34.0 + ((-1) ** j) * 5.0,
                    }
                )
        rows.append(
            {
                "game_id": "g",
                "period_id": 1,
                "frame_id": 1000 + fi,
                "time_seconds": float(fi),
                "player_id": "ball",
                "team_id": None,
                "is_ball": True,
                "is_goalkeeper": False,
                "x": 52.5,
                "y": 34.0,
            }
        )
    df = pd.DataFrame(rows)
    df["ball_state"] = "alive"
    if velocity:
        df["vx"] = 0.5
        df["vy"] = 0.0
        df["speed"] = 0.5
        df["speed_source"] = "native"
    else:
        df["speed_source"] = SPEED_SOURCE_UNAVAILABLE
    return df


def home_team_of(frames: pd.DataFrame):
    """The home team of a `two_team_frames` set (H, which defends x=0)."""
    return "H"
