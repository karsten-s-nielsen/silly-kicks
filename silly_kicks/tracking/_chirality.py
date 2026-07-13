"""Behavioral chirality fingerprint (ADR-037; enforcement in load() lands in PR-2).

A y-mirrored model serves inverted signed features silently --- the 4.18.0-weights class
of bug. The fingerprint is the model's OUTPUTS on a fixed, deliberately y-ASYMMETRIC
synthetic frame: derived from behavior, so a mislabeled artifact cannot satisfy it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable

import numpy as np
import pandas as pd

_CHIRALITY_VERSION = "chirality-probe-1"


def canonical_probe_frame() -> pd.DataFrame:
    """One synthetic frame, goal at x=105, all rows deliberately OFF the y=34 mirror axis."""
    rows = [
        dict(
            game_id="chir",
            period_id=1,
            frame_id=1,
            time_seconds=10.0,
            team_id="A",
            player_id="A1",
            x=80.0,
            y=20.0,
            vx=1.0,
            vy=0.5,
            is_ball=False,
            is_goalkeeper=False,
        ),
        dict(
            game_id="chir",
            period_id=1,
            frame_id=1,
            time_seconds=10.0,
            team_id="A",
            player_id="A2",
            x=88.0,
            y=45.0,
            vx=0.0,
            vy=0.0,
            is_ball=False,
            is_goalkeeper=False,
        ),
        dict(
            game_id="chir",
            period_id=1,
            frame_id=1,
            time_seconds=10.0,
            team_id="B",
            player_id="B1",
            x=92.0,
            y=25.0,
            vx=0.0,
            vy=0.0,
            is_ball=False,
            is_goalkeeper=False,
        ),
        dict(
            game_id="chir",
            period_id=1,
            frame_id=1,
            time_seconds=10.0,
            team_id="B",
            player_id="B2",
            x=95.0,
            y=50.0,
            vx=0.0,
            vy=0.0,
            is_ball=False,
            is_goalkeeper=False,
        ),
        dict(
            game_id="chir",
            period_id=1,
            frame_id=1,
            time_seconds=10.0,
            team_id="B",
            player_id="BGK",
            x=103.0,
            y=30.0,
            vx=0.0,
            vy=0.0,
            is_ball=False,
            is_goalkeeper=True,
        ),
        dict(
            game_id="chir",
            period_id=1,
            frame_id=1,
            time_seconds=10.0,
            team_id="A",
            player_id="ball",
            x=82.0,
            y=21.0,
            vx=2.0,
            vy=1.0,
            is_ball=True,
            is_goalkeeper=False,
        ),
    ]
    return pd.DataFrame(rows)


def chirality_fingerprint(predict_on_frame: Callable[[pd.DataFrame], np.ndarray]) -> dict:
    """predict_on_frame: Callable[[pd.DataFrame], np.ndarray] --- the model's own feature
    extraction + predict on the canonical frame. Returns a JSON-serializable dict."""
    frame = canonical_probe_frame()
    frame_sha = hashlib.sha256(json.dumps(frame.to_dict("records"), sort_keys=True, default=str).encode()).hexdigest()
    outputs = np.asarray(predict_on_frame(frame), dtype=float).ravel()
    return {
        "version": _CHIRALITY_VERSION,
        "frame_sha256": frame_sha,
        "outputs": [round(float(v), 10) for v in outputs],
    }
