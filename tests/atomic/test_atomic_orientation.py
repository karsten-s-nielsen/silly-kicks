"""PR-S23 / silly-kicks 3.0.1: atomic-SPADL inherits per-period direction-of-play fix.

The atomic namespace converts SPADL -> atomic-SPADL post-hoc -- there is
no native sportec/metrica converter inside silly_kicks/atomic/spadl/.
This smoke test verifies the per-period orientation fix survives the
atomic decomposition.
"""

from __future__ import annotations

import pandas as pd
import pytest

from silly_kicks.atomic.spadl import add_names as atomic_add_names
from silly_kicks.atomic.spadl import convert_to_atomic
from silly_kicks.spadl import config as spadlconfig
from tests.invariants import _loaders


def test_atomic_inherits_sportec_per_period_orientation():
    """Convert IDSSE per-period match through Sportec -> SPADL -> atomic-SPADL.

    Both teams' atomic-shot rows must cluster at high-x (canonical LTR),
    confirming the PER_PERIOD_ABSOLUTE dispatch fix propagates through
    the atomic-SPADL conversion.
    """
    actions, _home_team_id = _loaders.load_sportec_native_per_period()
    # Atomic-SPADL enforces int64 game_id / team_id / player_id; sportec
    # native uses string ids (orthogonal limitation). Coerce all three id
    # columns via categorical codes for the orientation smoke test --
    # semantic id values don't matter; only orientation does. Preserve the
    # team_id mapping so home stays home for the per-(team, period) check.
    actions = actions.copy()
    for col in ("game_id", "team_id", "player_id"):
        if col in actions.columns:
            actions[col] = pd.Categorical(actions[col]).codes.astype("int64")
    atomic = convert_to_atomic(actions)
    atomic = atomic_add_names(atomic)

    atomic_shots = atomic[atomic["type_name"] == "shot"]
    if len(atomic_shots) == 0:
        pytest.skip("atomic conversion produced no shot rows for IDSSE per-period fixture")

    # Atomic-SPADL coordinates use `x` (not `start_x`).
    by_group = atomic_shots.groupby(["period_id", "team_id"]).agg(n=("x", "size"), mean_x=("x", "mean"))
    reliable = by_group[by_group["n"] >= 3]
    if reliable.empty:
        pytest.skip(f"no reliable per-(period, team) shot groups in atomic output: {by_group.to_dict('index')}")

    pitch_mid = spadlconfig.field_length / 2
    failing = reliable[reliable["mean_x"] <= pitch_mid]
    assert failing.empty, (
        f"atomic-SPADL: per-(period, team) orientation violated. Failing groups:\n"
        f"{failing.to_string()}\nFull breakdown:\n{by_group.to_string()}"
    )


def test_atomic_inherits_metrica_per_period_orientation():
    """Same invariant for Metrica Sample Game 1 through the atomic conversion."""
    actions, _home_team_id = _loaders.load_metrica_native_per_period()
    # Atomic-SPADL enforces int64 ids; Metrica native uses string ids.
    actions = actions.copy()
    for col in ("game_id", "team_id", "player_id"):
        if col in actions.columns:
            actions[col] = pd.Categorical(actions[col]).codes.astype("int64")
    atomic = convert_to_atomic(actions)
    atomic = atomic_add_names(atomic)

    atomic_shots = atomic[atomic["type_name"] == "shot"]
    if len(atomic_shots) == 0:
        pytest.skip("atomic conversion produced no shot rows for Metrica per-period fixture")

    by_group = atomic_shots.groupby(["period_id", "team_id"]).agg(n=("x", "size"), mean_x=("x", "mean"))
    reliable = by_group[by_group["n"] >= 3]
    if reliable.empty:
        pytest.skip(f"no reliable per-(period, team) shot groups in atomic output: {by_group.to_dict('index')}")

    pitch_mid = spadlconfig.field_length / 2
    failing = reliable[reliable["mean_x"] <= pitch_mid]
    assert failing.empty, (
        f"atomic-SPADL: per-(period, team) orientation violated. Failing groups:\n"
        f"{failing.to_string()}\nFull breakdown:\n{by_group.to_string()}"
    )
