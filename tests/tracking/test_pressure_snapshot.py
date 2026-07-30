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

# Pinned hashes -- regenerate via REGENERATE_SNAPSHOTS=1 + scripts/regenerate_pressure_snapshot_shas.py
# Multiple valid hashes per method: numpy micro-version differences across runner
# images cause ULP-level float divergence in trig/hypot chains. Both hashes are
# correct outputs — the test guards against algorithmic drift, not platform jitter.
EXPECTED_SHAS = {
    "andrienko_oval": {
        "8d49f11737ae1874aa3a65bf12c37d76f0f8d977738f2944a8be3244138e5a9c",
        "23a32e695889bfc61cb90328c948b03787ac670748b20b4d7639bc7d59cf3294",
    },
    "link_zones": {
        "834889e6f2707046f0dcdbaea0805c829137b84d2ad54893a316a99282064ec5",
    },
    "bekkers_pi": {
        "3515a6aa716f97db256686f94b253ef17cebc0de510edce92778e24fbf2a3b28",
        "3ddcb35c67ca559e3ab19f405d45ff34872c29ba2f752b9992a08e39bc4086b2",
    },
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


def _hash_series(s: pd.Series) -> str:
    arr = s.fillna(-99999.0).astype("float64").values
    return hashlib.sha256(arr.tobytes()).hexdigest()  # type: ignore[attr-defined]


def test_andrienko_snapshot_stable() -> None:
    actions, frames = _build_fixture()
    result = pressure_on_actor(actions, frames, method="andrienko_oval")
    actual = _hash_series(result)
    expected = EXPECTED_SHAS["andrienko_oval"]
    assert actual in expected, f"Andrienko drift; expected one of {expected}, got {actual}"


def test_link_snapshot_stable() -> None:
    actions, frames = _build_fixture()
    result = pressure_on_actor(actions, frames, method="link_zones")
    actual = _hash_series(result)
    expected = EXPECTED_SHAS["link_zones"]
    assert actual in expected, f"Link drift; expected one of {expected}, got {actual}"


def test_bekkers_snapshot_stable() -> None:
    actions, frames = _build_fixture()
    result = pressure_on_actor(actions, frames, method="bekkers_pi")
    actual = _hash_series(result)
    expected = EXPECTED_SHAS["bekkers_pi"]
    assert actual in expected, f"Bekkers drift; expected one of {expected}, got {actual}"
