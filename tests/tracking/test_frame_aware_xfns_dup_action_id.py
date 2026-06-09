"""Behavioral gate: NO frame-aware xfns may raise on the non-unique action_id that
real VAEP gamestate slots carry at period boundaries. Enumerates the registered
surface so future xfns are auto-covered; a meta-assertion proves the gate sees every
*_xfns factory.

See ADR (frame-aware xfns frame-id resolution) + _kernels.resolve_frame_ids_by_position.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import silly_kicks.tracking.features as F
from silly_kicks.vaep.feature_framework import gamestates
from tests.tracking.test_defensive_line import _make_frame_rows


def _xt():
    from silly_kicks.xthreat import ExpectedThreat

    xt = ExpectedThreat(l=16, w=12)
    xt.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))
    return xt


def _actions():
    return pd.DataFrame(
        {
            "game_id": [1] * 4,
            "period_id": [1] * 4,
            "action_id": [10, 11, 12, 13],
            "time_seconds": [1.0, 2.0, 3.0, 4.0],
            "team_id": [1] * 4,
            "player_id": [5, 6, 7, 8],
            "start_x": [40.0, 45.0, 50.0, 55.0],
            "start_y": [34.0] * 4,
            "end_x": [70.0, 75.0, 60.0, 65.0],
            "end_y": [34.0] * 4,
            "type_id": [0] * 4,
            "result_id": [1] * 4,
            "bodypart_id": [0] * 4,
        }
    )


# Complete frame fixture: enough columns that the ONLY failure mode is the dup-action_id
# bug (a missing column would be a fixture gap, not the bug -- see _run_family).
def _frame():
    fr = _make_frame_rows(
        home_outfield_xs=[20.0, 22.0, 24.0, 26.0],
        home_outfield_ys=[20.0, 30.0, 40.0, 50.0],
        away_outfield_xs=[50.0, 60.0, 30.0, 80.0],
        away_outfield_ys=[34.0, 20.0, 34.0, 34.0],
    )
    fr["vx"] = 0.0
    fr["vy"] = 0.0
    fr["z"] = 0.0
    fr["speed"] = 0.0
    fr["ball_state"] = "alive"
    return fr


# Construction MUST succeed (no silent skip) -- an unconstructable factory is a gate
# FAILURE, not a skip, so a family can never go unprobed (no-silent-caps discipline).
_CONSTRUCT_ALLOWLIST: set[str] = set()  # factories that genuinely cannot construct (none today)


def _build(name):
    fac = getattr(F, name)
    if isinstance(fac, list):
        return fac
    xt = _xt()
    # home_team_id=1 preferred; (xt, home_team_id=1) for xt-takers; bare for factories
    # that take neither (e.g. pitch_control_xfns(method=...), elastic_sync_xfns(*,...)).
    for args, kw in (((), {"home_team_id": 1}), ((xt,), {"home_team_id": 1}), ((), {})):
        try:
            return fac(*args, **kw)
        except TypeError:
            continue
    raise AssertionError(
        f"{name}: no known construction signature -- extend _build (do NOT skip; an "
        f"unprobed family re-opens the hole this gate closes)."
    )


# The dup-action_id bug has two symptoms: `.at` on a non-unique index ("truth value of
# a Series is ambiguous"), and a merge fan-out ("Length of values (N) does not match
# length of index (M)"). Both mean: resolve frame_id / merge provenance dup-safely.
_DUP_SIGNATURES = ("truth value of a Series is ambiguous", "does not match length of index")


def _is_dup_symptom(msg: str) -> bool:
    return any(sig in msg for sig in _DUP_SIGNATURES)


_XFNS_NAMES = sorted(n for n in dir(F) if n.endswith("_xfns"))


def test_meta_gate_covers_every_xfns_factory():
    assert set(_XFNS_NAMES) == {n for n in dir(F) if n.endswith("_xfns")}
    assert len(_XFNS_NAMES) >= 21  # bumped for xt_gk_xfns
    assert not _CONSTRUCT_ALLOWLIST, "no construct-skips are expected today"


def _run_family(name):
    """Run every frame-aware transformer of `name` through a dup-action_id gamestate.
    Discriminates the target bug from a fixture gap so 5C fixes the bug, not the fixture."""
    states = gamestates(_actions(), nb_prev_actions=3)
    assert states[1]["action_id"].duplicated().any()  # precondition: dup exists
    frame = _frame()
    for t in _build(name):
        if not getattr(t, "_frame_aware", False):
            continue
        try:
            t(states, frame)
        except Exception as exc:
            if _is_dup_symptom(str(exc)):
                raise AssertionError(
                    f"{name}: DUP-ACTION_ID BUG -- retrofit to resolve_frame_ids_by_position / "
                    f"dedup provenance merge (Task 5C)."
                ) from exc
            raise AssertionError(
                f"{name}: non-dup error ({type(exc).__name__}: {exc}). This is a FIXTURE GAP -- "
                f"extend _frame(), do NOT alter the family's logic."
            ) from exc


@pytest.mark.parametrize("name", _XFNS_NAMES)
def test_xfns_survives_duplicate_action_id_gamestate(name):
    _run_family(name)  # MUST NOT raise on the non-unique action_id
