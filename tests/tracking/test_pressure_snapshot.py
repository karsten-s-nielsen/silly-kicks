"""Float-precision SHA-256 regression gate for pressure outputs.

Per spec section 8.6 (lakehouse review item 4): catches numpy/pandas minor-version
drift before it cascades. Failure means investigate -> if intentional, regenerate
expected SHAs via scripts/regenerate_pressure_snapshot_shas.py.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

from silly_kicks.tracking.features import pressure_on_actor

# Pinned hashes -- regenerate via REGENERATE_SNAPSHOTS=1 + scripts/regenerate_pressure_snapshot_shas.py
EXPECTED_SHAS = {
    "andrienko_oval": "8d49f11737ae1874aa3a65bf12c37d76f0f8d977738f2944a8be3244138e5a9c",
    "link_zones": "834889e6f2707046f0dcdbaea0805c829137b84d2ad54893a316a99282064ec5",
    "bekkers_pi": "3515a6aa716f97db256686f94b253ef17cebc0de510edce92778e24fbf2a3b28",
}


def _build_fixture():
    np.random.seed(42)
    n_actions = 50
    n_defenders_per_action = 5
    actions_rows = []
    frames_rows = []
    for action_id in range(n_actions):
        actor_x = 50.0 + np.random.uniform(-20, 20)
        actor_y = 34.0 + np.random.uniform(-15, 15)
        actions_rows.append(
            {
                "action_id": action_id,
                "period_id": 1,
                "team_id": "home",
                "player_id": 10 + action_id % 11,
                "start_x": actor_x,
                "start_y": actor_y,
                "type_id": 0,
                "time_seconds": float(action_id),
            }
        )
        # Frame for actor
        frames_rows.append(
            {
                "frame_id": action_id,
                "period_id": 1,
                "time_seconds": float(action_id),
                "team_id": "home",
                "player_id": 10 + action_id % 11,
                "is_ball": False,
                "x": actor_x,
                "y": actor_y,
                "vx": 0.0,
                "vy": 0.0,
                "speed": 0.0,
                "source_provider": "synthetic",
            }
        )
        # Ball
        frames_rows.append(
            {
                "frame_id": action_id,
                "period_id": 1,
                "time_seconds": float(action_id),
                "team_id": None,
                "player_id": None,
                "is_ball": True,
                "x": actor_x,
                "y": actor_y,
                "vx": 0.0,
                "vy": 0.0,
                "speed": 0.0,
                "source_provider": "synthetic",
            }
        )
        # Defenders
        for di in range(n_defenders_per_action):
            d_x = actor_x + np.random.uniform(-8, 8)
            d_y = actor_y + np.random.uniform(-8, 8)
            d_vx = np.random.uniform(-3, 3)
            d_vy = np.random.uniform(-3, 3)
            frames_rows.append(
                {
                    "frame_id": action_id,
                    "period_id": 1,
                    "time_seconds": float(action_id),
                    "team_id": "away",
                    "player_id": 100 + di,
                    "is_ball": False,
                    "x": d_x,
                    "y": d_y,
                    "vx": d_vx,
                    "vy": d_vy,
                    "speed": float(np.hypot(d_vx, d_vy)),
                    "source_provider": "synthetic",
                }
            )
    return pd.DataFrame(actions_rows), pd.DataFrame(frames_rows)


def _hash_series(s: pd.Series) -> str:
    arr = s.fillna(-99999.0).astype("float64").values
    return hashlib.sha256(arr.tobytes()).hexdigest()


def test_andrienko_snapshot_stable() -> None:
    actions, frames = _build_fixture()
    result = pressure_on_actor(actions, frames, method="andrienko_oval")
    actual = _hash_series(result)
    expected = EXPECTED_SHAS["andrienko_oval"]
    if expected.startswith("<"):
        if os.environ.get("REGENERATE_SNAPSHOTS"):
            print(f"andrienko_oval: {actual}")
            return
    assert actual == expected, f"Andrienko drift; was {expected}, now {actual}"


def test_link_snapshot_stable() -> None:
    actions, frames = _build_fixture()
    result = pressure_on_actor(actions, frames, method="link_zones")
    actual = _hash_series(result)
    expected = EXPECTED_SHAS["link_zones"]
    if expected.startswith("<"):
        if os.environ.get("REGENERATE_SNAPSHOTS"):
            print(f"link_zones: {actual}")
            return
    assert actual == expected, f"Link drift; was {expected}, now {actual}"


def test_bekkers_snapshot_stable() -> None:
    actions, frames = _build_fixture()
    result = pressure_on_actor(actions, frames, method="bekkers_pi")
    actual = _hash_series(result)
    expected = EXPECTED_SHAS["bekkers_pi"]
    if expected.startswith("<"):
        if os.environ.get("REGENERATE_SNAPSHOTS"):
            print(f"bekkers_pi: {actual}")
            return
    assert actual == expected, f"Bekkers drift; was {expected}, now {actual}"
