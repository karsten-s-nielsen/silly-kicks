"""Float-precision SHA-256 regression gate for pressure outputs.

Per spec section 8.6 (lakehouse review item 4): catches numpy/pandas minor-version
drift before it cascades. Failure means investigate -> if intentional, regenerate
expected SHAs via scripts/regenerate_pressure_snapshot_shas.py.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from silly_kicks.tracking.features import pressure_on_actor

# Pinned hashes -- regenerate via scripts/regenerate_pressure_snapshot_shas.py.
# ONE hash per method. `_hash_series` rounds to `_HASH_DECIMALS` before hashing: the runner CPU (its
# numpy SIMD path) and numpy/pandas micro-versions move trig/hypot chains by the last ULP (~1e-14),
# which a raw-bytes hash turned into a growing set of accepted hashes and a CPU-dependent flake.
# The test guards against algorithmic drift, not platform jitter.
EXPECTED_SHAS = {
    "andrienko_oval": "e1ab2f35e8bc7365a746d476956c3257400ad310f5c7c4f18f4122d85c2f7c76",
    "link_zones": "feb621a88c5d69cb313d17a0fb3bb74d9790938ee6d4f51811e827f340746276",
    "bekkers_pi": "0d303dd5ebfd6ced20b604d81d30ef9fd6ce4a4abc4af65ff7a8f26e35d51fc0",
}


def _build_fixture():
    # ADR-028 orientation labelling. The frames are the canonical home-attacks-right
    # convention, so the team named "home" carries "ltr" and "away" carries "rtl";
    # ball rows carry None, matching what convert_to_frames emits (acting_team_attacks_rtl
    # filters ball rows out anyway).
    #
    # pressure_on_actor takes no home_team_id, and the low-x-half tie-break does not
    # discriminate here: defenders are generated as actor +/- U(-8, 8), so both teams
    # occupy the same x distribution (mean ~50). The team NAMES are the only signal.
    #
    # Every action in this fixture is by "home", so the resolved flip is all-False and the
    # snapshot values are unchanged -- but it is now all-False because the direction
    # RESOLVED to "no flip for a home-team action", not because the lookup silently failed.
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
                "team_attacking_direction": "ltr",
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
                "team_attacking_direction": None,
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
                    "team_attacking_direction": "rtl",
                    "source_provider": "synthetic",
                }
            )
    return pd.DataFrame(actions_rows), pd.DataFrame(frames_rows)


# 1e-9 is five orders of magnitude above the observed ULP jitter and far below any algorithmic change.
_HASH_DECIMALS = 9


def _hash_series(s: pd.Series) -> str:
    arr = np.round(s.fillna(-99999.0).astype("float64").to_numpy(copy=True), _HASH_DECIMALS)
    arr = arr + 0.0  # fold a rounded -0.0 into 0.0: the two differ in bytes
    return hashlib.sha256(arr.tobytes()).hexdigest()


def test_andrienko_snapshot_stable() -> None:
    actions, frames = _build_fixture()
    result = pressure_on_actor(actions, frames, method="andrienko_oval")
    actual = _hash_series(result)
    expected = EXPECTED_SHAS["andrienko_oval"]
    assert actual == expected, f"Andrienko drift; expected {expected}, got {actual}"


def test_link_snapshot_stable() -> None:
    actions, frames = _build_fixture()
    result = pressure_on_actor(actions, frames, method="link_zones")
    actual = _hash_series(result)
    expected = EXPECTED_SHAS["link_zones"]
    assert actual == expected, f"Link drift; expected {expected}, got {actual}"


def test_bekkers_snapshot_stable() -> None:
    actions, frames = _build_fixture()
    result = pressure_on_actor(actions, frames, method="bekkers_pi")
    actual = _hash_series(result)
    expected = EXPECTED_SHAS["bekkers_pi"]
    assert actual == expected, f"Bekkers drift; expected {expected}, got {actual}"
