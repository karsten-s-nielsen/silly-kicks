"""TF-49 event-seam smoke on the committed StatsBomb WC2018 fixture (ALL CI legs).

The owner-gated GS e2e is not the only guard on the event-only seams (spec s7):
receiver resolution + secured_reception run here on real-shaped data, including a
synthetic non_action injection (StatsBomb SPADL carries no native non_action rows,
so the skip path must be exercised on real data, not only unit fixtures).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl.utils import _resolve_next_touch_positions, resolve_next_touch_receiver
from silly_kicks.tracking import secured_reception

_GAME_KEYS = ("/actions/game_7525", "/actions/game_7529")

# Receiver-bearing packing domain (default action_types minus dribble).
_PASS_LIKE = (
    "pass",
    "cross",
    "throw_in",
    "freekick_crossed",
    "freekick_short",
    "corner_crossed",
    "corner_short",
    "goalkick",
)


def _two_games(sb_worldcup_data) -> pd.DataFrame:
    return pd.concat([sb_worldcup_data[k] for k in _GAME_KEYS], ignore_index=True)


def _pass_like_completed(actions: pd.DataFrame) -> np.ndarray:
    ids = [spadlconfig.actiontype_id[n] for n in _PASS_LIKE]
    return np.isin(actions["type_id"].to_numpy(), ids) & (
        actions["result_id"].to_numpy() == spadlconfig.result_id["success"]
    )


def test_receiver_resolution_rate_and_dtype(sb_worldcup_data):
    actions = _two_games(sb_worldcup_data)
    receiver = resolve_next_touch_receiver(actions)
    assert receiver.dtype == "Int64"  # int64 source passthrough; NEVER float64 (F5)
    mask = _pass_like_completed(actions)
    assert mask.sum() > 500, "fixture regression: too few completed pass-like rows"
    rate = receiver[mask].notna().mean()
    # StatsBomb probe anchor 0.9957 under the shipping non_action-skip rule; +-margin
    # for a 2-game sample vs the full-corpus probe.
    assert 0.95 <= rate <= 1.0, f"receiver resolution rate {rate:.4f} outside [0.95, 1.0]"


def test_secured_reception_tri_state_on_real_games(sb_worldcup_data):
    actions = _two_games(sb_worldcup_data)
    completed_pass = (actions["type_id"].to_numpy() == spadlconfig.actiontype_id["pass"]) & (
        actions["result_id"].to_numpy() == spadlconfig.result_id["success"]
    )
    line_x = pd.Series(np.where(completed_pass, 50.0, np.nan), index=actions.index)
    secured = secured_reception(actions, line_x)
    assert secured.dtype == "boolean"
    counts = secured.value_counts(dropna=False)
    assert counts.get(True, 0) > 0, "no True secured label across two real games"
    assert counts.get(False, 0) > 0, "no False secured label across two real games"
    assert secured.isna().any(), "no <NA> secured label across two real games"


def test_synthetic_non_action_injection_leaves_resolution_unchanged(sb_worldcup_data):
    """Review minor 11: inject one same-team non_action row between a completed pass
    and its receiver's touch; the skip rule must leave every resolution unchanged."""
    actions = _two_games(sb_worldcup_data)
    baseline = resolve_next_touch_receiver(actions)

    positions = _resolve_next_touch_positions(actions)
    mask = _pass_like_completed(actions)
    target = None
    for i in np.flatnonzero(mask):
        if positions.iloc[i] == i + 1:  # receiver is the immediately-next row
            target = int(i)
            break
    assert target is not None, "no completed pass with an immediately-next receiver found"

    row = actions.iloc[target]
    synthetic = {
        "game_id": row["game_id"],
        "original_event_id": None,
        "action_id": -1,  # relabeled below
        "period_id": row["period_id"],
        "time_seconds": (row["time_seconds"] + actions.iloc[target + 1]["time_seconds"]) / 2.0,
        "team_id": row["team_id"],
        "player_id": np.nan,  # GS-shaped null actor -> float64 upcast path exercised too
        "start_x": row["end_x"],
        "start_y": row["end_y"],
        "end_x": row["end_x"],
        "end_y": row["end_y"],
        "type_id": spadlconfig.actiontype_id["non_action"],
        "result_id": spadlconfig.result_id["success"],
        "bodypart_id": 0,
        "possession": row["possession"],
    }
    injected = pd.concat(
        [actions.iloc[: target + 1], pd.DataFrame([synthetic]), actions.iloc[target + 1 :]],
        ignore_index=True,
    )
    # Keep play order == action_id order after the insertion (the seam sorts by it).
    injected["action_id"] = np.arange(len(injected))

    out = resolve_next_touch_receiver(injected)
    # The injected row is at position target+1; original rows map around it.
    out_original = pd.concat([out.iloc[: target + 1], out.iloc[target + 2 :]], ignore_index=True)
    baseline_vals = baseline.reset_index(drop=True)
    assert out_original.iloc[target] == baseline_vals.iloc[target]
    pd.testing.assert_series_equal(out_original.astype("Int64"), baseline_vals.astype("Int64"), check_names=False)
