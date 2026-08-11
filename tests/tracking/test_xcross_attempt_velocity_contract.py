"""`add_xcross_attempt` must honour the velocity-availability contract, not crash on it.

Found by making `tests/sb360/_regenerate.py`'s swallowed probe failure LOUD: `add_xcross_attempt`
was the ONLY registry entry with ``columns=()`` -- its velocity block and all three visibility
blocks EMPTY -- so ADR-053's "every ``add_*`` carries an SB360 freeze-frame verdict" was not true of
it, and nothing said so. The regeneration probe raised a bare ``KeyError: 'vx'`` and the handler
turned that into "this aggregator emits nothing".

The input it crashed on is not malformed: it declares ``speed_source="unavailable"``, the token
ADR-054 introduced precisely so a frame builder can DECLARE that its source has no per-player
temporal history. SB360 freeze-frames are the canonical case.

The house contract for a velocity-dependent aggregator is two-pronged, and `_das`,
`_ghost_gk` and `_press_commitment` all implement it:

* velocity DECLARED unavailable -> degrade to NaN. Nothing is fabricated, so nothing is wrong.
* velocity NOT declared and vx/vy absent -> RAISE, informatively. That is the "forgot
  ``derive_velocities()``" case, and CLAUDE.md is explicit that fail-loud wins on a
  partially-marked frame set.

A bare ``KeyError`` on a column name is neither: it is indistinguishable from a bug, it names
nothing the caller can act on, and -- as this defect proves -- an upstream ``except`` can silently
reinterpret it as an absence of data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import silly_kicks.tracking as T
from silly_kicks.tracking.schema import SPEED_SOURCE_UNAVAILABLE


def _freeze_frame(*, declare: str, with_velocity: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One SB360-shaped snapshot: attacker in the wide area, defending keeper, ball.

    `declare` sets `speed_source` on every row; `with_velocity` decides whether vx/vy exist at all.
    A freeze-frame legitimately has neither, which is the whole point of the `unavailable` token.
    """
    rows = []
    for frame_id, t in enumerate([0.0, 0.4], start=1):
        rows += [
            dict(
                game_id="g",
                period_id=1,
                frame_id=frame_id,
                time_seconds=t,
                team_id="A",
                player_id="A1",
                x=95.0,
                y=10.0,
                is_ball=False,
                is_goalkeeper=False,
                ball_state="alive",
            ),
            dict(
                game_id="g",
                period_id=1,
                frame_id=frame_id,
                time_seconds=t,
                team_id="B",
                player_id="Bgk",
                x=104.0,
                y=34.0,
                is_ball=False,
                is_goalkeeper=True,
                ball_state="alive",
            ),
            dict(
                game_id="g",
                period_id=1,
                frame_id=frame_id,
                time_seconds=t,
                team_id="ball",
                player_id=None,
                x=95.0,
                y=10.0,
                is_ball=True,
                is_goalkeeper=False,
                ball_state="alive",
            ),
        ]
    frames = pd.DataFrame(rows)
    frames["source_provider"] = "test"  # required by link_actions_to_frames
    frames["speed_source"] = declare
    if with_velocity:
        frames["vx"] = 1.0
        frames["vy"] = 0.0

    from silly_kicks.spadl import config as spc

    actions = pd.DataFrame(
        {
            "game_id": ["g"],
            "period_id": [1],
            "action_id": [0],
            "team_id": ["A"],
            "time_seconds": [0.2],
            "type_id": [spc.actiontype_id["cross"]],
            "result_id": [spc.result_id["success"]],
        }
    )
    return frames, actions


def test_declared_unavailable_degrades_to_nan_rather_than_crashing() -> None:
    """The SB360 freeze-frame case: a DECLARED absence is data, not a malformed input.

    This is the exact input the sb360 audit probe supplies, and it raised `KeyError: 'vx'` -- which
    an upstream handler then read as "this aggregator emits no columns at all".
    """
    frames, actions = _freeze_frame(declare=SPEED_SOURCE_UNAVAILABLE, with_velocity=False)

    out = T.add_xcross_attempt(actions, frames, home_team_id="A")

    assert "xcross_attempt" in out.columns, (
        "the column vanished entirely -- a consumer cannot distinguish that from the aggregator "
        "never having run, which is the failure mode this test exists for"
    )
    assert len(out) == len(actions)
    assert out["xcross_attempt"].isna().all(), (
        "velocity is declared unavailable, so every score must be NaN. A NUMBER here would mean "
        "the model scored on features it could not have computed -- the ADR-053 fabrication shape."
    )


def test_undeclared_missing_velocity_raises_informatively() -> None:
    """The "forgot derive_velocities()" case must fail LOUD, and say what to do.

    Not a KeyError: the caller needs to be told that vx/vy are required and that declaring
    `speed_source` unavailable is the alternative. Mirrors `compute_press_commitment`'s message.
    """
    frames, actions = _freeze_frame(declare="native", with_velocity=False)

    with pytest.raises(ValueError, match=r"vx/vy") as excinfo:
        T.add_xcross_attempt(actions, frames, home_team_id="A")

    message = str(excinfo.value)
    assert "derive_velocities" in message, f"message names no remedy: {message!r}"
    assert SPEED_SOURCE_UNAVAILABLE in message, (
        f"message does not mention the declared-unavailable alternative, so a freeze-frame caller "
        f"is left with no legitimate route: {message!r}"
    )


def test_velocity_bearing_frames_are_unaffected() -> None:
    """Non-vacuity: the guard must not swallow the case it is NOT about.

    Without this, returning all-NaN unconditionally would satisfy every assertion above.
    """
    frames, actions = _freeze_frame(declare="native", with_velocity=True)

    out = T.add_xcross_attempt(actions, frames, home_team_id="A")

    assert "xcross_attempt" in out.columns
    assert out["xcross_attempt"].notna().any(), (
        "a fully velocity-bearing frame set scored NaN -- the guard is firing on input it should "
        "pass through, so the all-NaN assertions above prove nothing"
    )


def test_extract_xcross_features_honours_its_documented_nan_tolerance() -> None:
    """`extract_xcross_features` is PUBLIC and documents itself "NaN-tolerant".

    It pre-fills every feature with NaN and then overwrites what it can compute, so an absent
    optional input belongs in that NaN pre-fill -- not in an unguarded `f.loc[is_ball, "vx"]`.
    Guarded here as well as at the compute seam because this is its own entry point.
    """
    frames, _actions = _freeze_frame(declare=SPEED_SOURCE_UNAVAILABLE, with_velocity=False)
    snapshot = frames[frames["frame_id"] == 1]

    row = T.extract_xcross_features(snapshot, gk_team_id="B", goal_x=105.0, carrier_player_id="A1")

    assert len(row) == 1
    assert np.isnan(row["ball_speed"].iloc[0]), (
        "ball_speed must be NaN when vx/vy are absent -- an unguarded read raises KeyError and a "
        "zero-fill would be a fabricated stationary ball"
    )
    assert row["ball_r"].notna().all(), (
        "the POSITIONAL ball features must still be computed -- absent velocity must not take out "
        "the geometry that does not depend on it"
    )
