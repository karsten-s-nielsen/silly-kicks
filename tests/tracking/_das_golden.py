"""Loader for the committed DAS golden fixture (the ``accessible-space`` 2.0.15 parity oracle).

Reads ``tests/tracking/_fixtures/das_golden/`` and re-applies per-scene dtypes so the native DAS
engine can be compared against the frozen reference without the library installed. See
``_fixtures/das_golden/_generate.py`` for how the fixture is produced. Plan Task 1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

GOLDEN_DIR = Path(__file__).resolve().parent / "_fixtures" / "das_golden"

GOLDEN_FILES: tuple[str, ...] = (
    "scenes_frames.csv",
    "scenes_passes.csv",
    "reference_das_team.csv",
    "reference_das_player.csv",
    "reference_xc.csv",
    "reference_errors.json",
    "metadata.json",
    "SHA256SUMS",
)

#: Scenes with a real (finite) reference; excludes the divergence (``V-*``) scenes.
NORMAL_SCENES: tuple[str, ...] = (
    "S01",
    "S02",
    "S03",
    "S04",
    "S05",
    "S06",
    "S07",
    "S08",
    "S09",
    "S10",
    "S11",
    "S12a",
    "S12b",
    "S12c",
    "M01",
    "M02",
    "M03",
)

#: Per-scene identifier dtype (the id-axis scenes S12a/b/c exercise Int64 / object / category).
_ID_DTYPE = {"S12b": "object", "S12c": "category"}
_ID_COLS = ("player_id", "team_id", "team_in_possession")


def _apply_frame_dtypes(df: pd.DataFrame, scene: str) -> pd.DataFrame:
    out = df.copy()
    for c in ("x", "y", "vx", "vy"):
        out[c] = out[c].astype("float32")
    out["dir"] = out["dir"].astype("float64")
    out["is_ball"] = out["is_ball"].astype(bool)
    id_dtype = _ID_DTYPE.get(scene, "Int64")
    for c in _ID_COLS:
        if id_dtype == "Int64":
            out[c] = out[c].astype("Int64")
        elif id_dtype == "object":
            s = out[c].astype("Int64")
            out[c] = pd.Series([None if pd.isna(v) else str(int(v)) for v in s], index=out.index, dtype="object")
        else:  # category
            s = out[c].astype("Int64")
            out[c] = pd.Series([None if pd.isna(v) else str(int(v)) for v in s], index=out.index).astype("category")
    return out.reset_index(drop=True)


@dataclass(frozen=True)
class GoldenFixture:
    frames: pd.DataFrame
    passes: pd.DataFrame
    das_team: pd.DataFrame
    das_player: pd.DataFrame
    xc: pd.DataFrame
    errors: dict
    metadata: dict

    def frames_for(self, scene: str) -> pd.DataFrame:
        sub = self.frames[self.frames["scene_id"] == scene]
        return _apply_frame_dtypes(sub.drop(columns=["scene_id"]), scene)

    def passes_for(self, scene: str) -> pd.DataFrame:
        return self.passes[self.passes["scene_id"] == scene].drop(columns=["scene_id"]).reset_index(drop=True)

    def reference_for(self, scene: str) -> dict[str, np.ndarray]:
        t = self.das_team[self.das_team["scene_id"] == scene].sort_values(
            ["game_id", "period_id", "frame_id"], kind="stable"
        )
        p = self.das_player[self.das_player["scene_id"] == scene].sort_values(
            ["game_id", "period_id", "frame_id", "player_id"], kind="stable"
        )
        return {
            "team_as": t["AS"].to_numpy(dtype=float),
            "team_das": t["DAS"].to_numpy(dtype=float),
            "player_as": p["AS"].to_numpy(dtype=float),
            "player_das": p["DAS"].to_numpy(dtype=float),
            "team_keys": t[["game_id", "period_id", "frame_id"]].to_numpy(),
            "player_keys": p[["game_id", "period_id", "frame_id", "player_id"]].to_numpy(),
        }


def load_golden() -> GoldenFixture:
    frames = pd.read_csv(GOLDEN_DIR / "scenes_frames.csv")
    passes = pd.read_csv(GOLDEN_DIR / "scenes_passes.csv")
    das_team = pd.read_csv(GOLDEN_DIR / "reference_das_team.csv")
    das_player = pd.read_csv(GOLDEN_DIR / "reference_das_player.csv")
    xc = pd.read_csv(GOLDEN_DIR / "reference_xc.csv")
    errors = json.loads((GOLDEN_DIR / "reference_errors.json").read_text(encoding="utf-8"))
    metadata = json.loads((GOLDEN_DIR / "metadata.json").read_text(encoding="utf-8"))
    return GoldenFixture(frames, passes, das_team, das_player, xc, errors, metadata)
