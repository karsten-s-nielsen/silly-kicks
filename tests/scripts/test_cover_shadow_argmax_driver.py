"""The cover-shadow argmax driver: ADR-028 RC1 passer reprojection + resume (ADR-052).

RC1 (4.70.0/PR-S138) fixed the two `features.py` callers that fed a raw ACTION-LTR `passer_xy` into
`_compute_cover_shadow_dict` beside FRAME-LTR positions. This driver imports that private function
DIRECTLY, so it was never a registered site and the same defect stayed live on `main` -- and it does
not cancel between the two arms the script compares, because only the CHEAP path consumes the
passer. That is the arm whose nominee the headline agreement rate is about.
"""

from __future__ import annotations

import pandas as pd
import pytest

import scripts.measure_cover_shadow_argmax_agreement as mod

_HOME, _AWAY = "H", "A"
_START = (20.0, 12.0)


def _actions() -> pd.DataFrame:
    """One home action and one away action from the SAME action-LTR start point.

    Identical coordinates on purpose: any difference in the passer that reaches the kernel is then
    attributable to the reprojection and to nothing else.
    """
    return pd.DataFrame(
        {
            "game_id": ["g", "g"],
            "action_id": [1, 2],
            "period_id": [1, 1],
            "team_id": [_HOME, _AWAY],
            "start_x": [_START[0], _START[0]],
            "start_y": [_START[1], _START[1]],
        }
    )


def _frames() -> pd.DataFrame:
    """Home attacks left-to-right, away right-to-left -- the canonical frame convention."""
    return pd.DataFrame(
        {
            "game_id": ["g"] * 4,
            "period_id": [1] * 4,
            "frame_id": [7, 7, 7, 7],
            "team_id": [_HOME, _HOME, _AWAY, _AWAY],
            "player_id": ["h1", "h2", "a1", "a2"],
            "team_attacking_direction": ["ltr", "ltr", "rtl", "rtl"],
            "is_ball": [False, False, False, False],
            "is_goalkeeper": [False, False, False, False],
            "x": [30.0, 40.0, 60.0, 70.0],
            "y": [20.0, 30.0, 40.0, 50.0],
        }
    )


@pytest.fixture()
def spy_passer(monkeypatch):
    """Record every `passer_xy` the kernel is handed, keyed by the acting team."""
    seen: list[tuple] = []

    def _fake_dict(frame_data, passer_xy, tid, xt, **_kw):
        seen.append((tid, passer_xy))
        # `max_single_defender_player_id=None` short-circuits the counterfactual re-score, which is
        # not what this test is about.
        return {"max_single_defender_player_id": None, "max_single_defender_blocking_score": 0.0}

    monkeypatch.setattr(mod, "_compute_cover_shadow_dict", _fake_dict)
    monkeypatch.setattr(mod, "_lane_blocker_count", lambda *_a, **_k: 3)
    monkeypatch.setattr(
        mod,
        "link_actions_to_frames",
        lambda actions, frames: (pd.DataFrame({"action_id": [1, 2], "frame_id": [7, 7]}), None),
    )
    return seen


def test_the_AWAY_passer_is_reprojected_into_frame_coordinates(spy_passer):
    mod.measure_match(_actions(), _frames(), _HOME, object(), match_id="m1")

    by_team = dict(spy_passer)
    assert by_team[_HOME] == _START, "the home passer is already in frame coords and must not move"
    assert by_team[_AWAY] == (mod.FIELD_LENGTH - _START[0], mod.FIELD_WIDTH - _START[1])


def test_the_reprojection_actually_MOVES_the_away_passer(spy_passer):
    """Non-vacuity. A start point at the pitch centre would satisfy the assertion above under both
    the fixed and the broken code, so the fixture's off-centre point is load-bearing: the away
    passer must land somewhere the raw coordinates never could."""
    mod.measure_match(_actions(), _frames(), _HOME, object(), match_id="m1")

    by_team = dict(spy_passer)
    assert by_team[_AWAY] != by_team[_HOME]
    assert by_team[_AWAY] != _START, "an unreprojected away passer is exactly the raw start point"
    dx = abs(by_team[_AWAY][0] - _START[0])
    assert dx > 60.0, f"the flip must cross the halfway line; moved only {dx:.1f} m"


def test_an_UNORIENTED_frame_set_leaves_every_passer_untouched(spy_passer, recwarn):
    """The other side of the flip. With no resolvable direction nothing is reprojected -- and the
    library warns rather than silently reporting an all-False flip as a measurement."""
    from silly_kicks.tracking import OrientationUnresolvedWarning

    frames = _frames().assign(team_attacking_direction=None)

    mod.measure_match(_actions(), frames, _HOME, object(), match_id="m1")

    assert {p for _t, p in spy_passer} == {_START}
    # The docstring has always CLAIMED the library warns here; nothing asserted it, so the claim
    # was decorative and `recwarn` went unused. Under the 4.80.0 nullable contract the driver
    # `.fillna(False)`s an all-<NA> flip, which is the right answer for an agreement RATE but is
    # indistinguishable from a resolved all-left-to-right scene -- the warning is the only thing
    # that tells them apart, so it is the part worth pinning.
    assert any(issubclass(w.category, OrientationUnresolvedWarning) for w in recwarn), (
        "no OrientationUnresolvedWarning -- an unoriented frame set would be measured as if it "
        "had been resolved, with nothing in the output saying otherwise"
    )
