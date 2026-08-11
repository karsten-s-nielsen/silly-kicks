"""`add_elastic_sync` must resolve player-ball distance regardless of the frames' id dtype.

ADR-019 defect, found when `snapshot_to_tracking_frames` started casting its ids: the
player-ball distance lookup keyed on ``merged["player_id"].astype(str)`` while the query keyed on
``str(action_row["player_id"])``. With a FLOAT id column -- which is what a concat produces whenever
a frame set carries an NA id, i.e. every ball row -- the lookup stored ``"10.0"`` and the query
asked for ``"10"``. Measured: every lookup missed, `dist` fell to ``inf``, `proximity_score` to 0,
and `elastic_confidence` collapsed to exactly ``accel_weight / (accel_weight + proximity_weight)``
= **0.6 on every row**.

A constant 0.6 is the shape this repo keeps naming: a plausible number from a computation that did
not happen. The SB360 audit recorded it as ``identical`` -> ``works``, because BOTH legs were
equally broken -- a one-sided check cannot see a defect that degrades both arms the same way.

CLAUDE.md already records this exact trap ("`str(5.0)` iterrows-upcast player-influence/cover-shadow
mislabel"); this module was not on the surface the ADR-043 registry sweep enumerated, because that
registry covers id-SCALAR arguments of public functions and this is an internal dict key.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._elastic_sync import _build_player_ball_distance_lookup


def _frames(player_id_values) -> pd.DataFrame:
    rows = []
    for frame_id in (1, 2):
        for pid, x in zip(player_id_values, (10.0, 20.0), strict=True):
            rows.append(dict(game_id=7, period_id=1, frame_id=frame_id, player_id=pid, x=x, y=34.0, is_ball=False))
        rows.append(dict(game_id=7, period_id=1, frame_id=frame_id, player_id=None, x=15.0, y=34.0, is_ball=True))
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    ("label", "values"),
    [
        ("python_int", [10, 11]),
        ("float", [10.0, 11.0]),
        ("Int64", pd.array([10, 11], dtype="Int64")),
        ("string", ["10", "11"]),
    ],
)
def test_distance_lookup_keys_are_canonical_whatever_the_id_dtype(label, values) -> None:
    """One physical scene, four id dtypes, ONE key set. Anything else is a silent join-miss."""
    lookup = _build_player_ball_distance_lookup(_frames(values))

    player_keys = {key[3] for key in lookup}
    assert player_keys == {"10", "11"}, (
        f"id dtype {label!r} produced lookup keys {sorted(player_keys)}. The query side builds its "
        f"key from the ACTION's player_id, so any rendering other than the canonical one misses "
        f"every row -- silently, because a miss reads as 'infinitely far from the ball' rather than "
        f"as an error."
    )


def test_a_float_id_column_still_resolves_a_real_distance() -> None:
    """The behavioural consequence, not just the key shape.

    A miss returns `inf` from the caller's `.get(..., inf)` default, which becomes
    `proximity_score = 0` and a constant confidence. This asserts the distance is finite and
    correct, so the defect cannot return as a differently-shaped key.
    """
    lookup = _build_player_ball_distance_lookup(_frames([10.0, 11.0]))

    dist = lookup[(np.int64(7), np.int64(1), 1, "10")]
    assert np.isfinite(dist), "player-ball distance is infinite -- the lookup missed"
    assert dist == pytest.approx(5.0), f"player at x=10, ball at x=15, same y: expected 5.0, got {dist}"
