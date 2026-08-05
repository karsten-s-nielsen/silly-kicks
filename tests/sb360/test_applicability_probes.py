"""The applicability class is DERIVED, not declared.

A human picking one of three categories would put a declaration inside the LOCKED half of the
registry -- the exact observation/adjudication conflation this design exists to prevent.
"""

from __future__ import annotations

import numpy as np

from tests.sb360 import _vocabulary as V
from tests.sb360._probes import derive_applicability
from tests.sb360._registry import Sb360Entry


def _hull_like(actions, frames, links, home_team_id):
    """Support defined BY the visible players: sensitive to an extreme member."""
    players = frames[~frames["is_ball"].astype(bool)]
    return actions.assign(m=float(players["x"].max() - players["x"].min()))


def _region_like(actions, frames, links, home_team_id):
    """Fixed query region anchored on the ACTION: indifferent to a player never inside it.

    Anchored on the ball rather than a hardcoded pitch coordinate, because that is what a real
    region-support feature does (defenders in the triangle to goal, receiver zone density).
    """
    is_player = ~frames["is_ball"].astype(bool)
    players, balls = frames[is_player], frames[~is_player]
    ball_xy = balls.set_index("frame_id")[["x", "y"]].astype(float)
    dx = players["x"].astype(float) - players["frame_id"].map(ball_xy["x"])
    dy = players["y"].astype(float) - players["frame_id"].map(ball_xy["y"])
    within = np.hypot(dx.to_numpy(), dy.to_numpy()) < 15.0
    return actions.assign(m=float(within.sum()))


def _scalar_like(actions, frames, links, home_team_id):
    """Reads only the actor's own row: no spatial support at all."""
    return actions.assign(m=actions["start_x"].astype(float))


def _entry(call):
    return Sb360Entry(name="stub", call=call, columns=("m",))


def test_extreme_displacement_identifies_data_defined_support():
    cls, deltas = derive_applicability(_entry(_hull_like), "m")
    assert cls == "support_data_defined", f"got {cls} with deltas {deltas}"
    assert deltas["extreme"] > 0, "probe 1 must measurably move, or the classification is vacuous"


def test_region_support_is_not_moved_by_an_extreme_member():
    """The discriminator: a fixed query region is indifferent to a player never inside it."""
    cls, deltas = derive_applicability(_entry(_region_like), "m")
    assert cls == "region_support", f"got {cls} with deltas {deltas}"
    assert deltas["extreme"] == 0.0, (
        f"the extreme probe moved a FIXED-region feature ({deltas}); it would be misclassified as data-defined"
    )
    assert deltas["near"] > 0, "probe 2 must measurably move, or the classification is vacuous"


def test_scalar_feature_is_moved_by_neither_probe():
    cls, deltas = derive_applicability(_entry(_scalar_like), "m")
    assert cls == "no_support", f"got {cls} with deltas {deltas}"
    assert np.isclose(deltas["extreme"], 0.0)
    assert np.isclose(deltas["near"], 0.0)


def test_every_applicability_class_is_producible():
    """Vocabulary invariant 7, exercised rather than asserted in prose."""
    produced = {derive_applicability(_entry(c), "m")[0] for c in (_hull_like, _region_like, _scalar_like)}
    assert produced == V.APPLICABILITY, f"unreached classes: {sorted(V.APPLICABILITY - produced)}"


def test_probes_do_not_mutate_the_caller_frames():
    """A probe that mutated its input would corrupt every later measurement in the same run."""
    from tests.sb360 import _fixture as F
    from tests.sb360._probes import _shift

    _, frames, _ = F.build_leg_a()
    before = frames["x"].to_numpy().copy()
    _shift(frames, extreme=True)
    _shift(frames, extreme=False)
    np.testing.assert_array_equal(frames["x"].to_numpy(), before)


def test_probes_move_exactly_one_player():
    """A probe that moved several would confound 'support is data-defined' with 'the feature
    reads more than one player'."""
    from tests.sb360 import _fixture as F
    from tests.sb360._probes import _shift

    _, frames, _ = F.build_leg_a()
    for extreme in (True, False):
        moved = _shift(frames, extreme=extreme)
        differing = int((moved["x"].to_numpy() != frames["x"].to_numpy()).sum())
        assert differing == 1, f"extreme={extreme} moved {differing} rows, expected exactly 1"
