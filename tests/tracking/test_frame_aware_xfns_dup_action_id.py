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

import silly_kicks.atomic.tracking.features as AF
import silly_kicks.tracking as T
import silly_kicks.tracking.features as F
from silly_kicks.vaep.feature_framework import gamestates
from tests.tracking.test_defensive_line import _make_frame_rows

# ADR-041 opt-out: auto-enumerating gate: sweeps EVERY registered aggregator on defaults, so the OBSO family's
# synthetic-EPV notice is expected and irrelevant here.
pytestmark = pytest.mark.filterwarnings("ignore::silly_kicks.tracking.SyntheticEPVWarning")


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


def _atomic_actions():
    """The ATOMIC-shaped twin of `_actions()` -- same trajectories, atomic schema.

    Cycle B: the atomic mirrors consume `x`/`y`/`dx`/`dy` (ATOMIC_SPADL_COLUMNS), not
    `start_x`/`end_x`. Handing them the SPADL fixture yields `KeyError: 'x'`, which
    `_run_family` correctly reports as a FIXTURE GAP rather than the dup-action_id bug --
    exactly the discrimination that keeps this gate from being "fixed" in the wrong place.

    Displacements mirror `_actions()`: (40,45,50,55) -> (70,75,60,65) on x, y flat at 34.
    """
    return pd.DataFrame(
        {
            "game_id": [1] * 4,
            "period_id": [1] * 4,
            "action_id": [10, 11, 12, 13],
            "time_seconds": [1.0, 2.0, 3.0, 4.0],
            "team_id": [1] * 4,
            "player_id": [5, 6, 7, 8],
            "x": [40.0, 45.0, 50.0, 55.0],
            "y": [34.0] * 4,
            "dx": [30.0, 30.0, 10.0, 10.0],
            "dy": [0.0] * 4,
            "type_id": [0] * 4,
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


#: Families needing a contract column the shared `_frame()` deliberately does NOT carry.
#:
#: `team_in_possession` is NOT added to `_frame()` itself: `derive_team_in_possession`
#: merges its result in without checking for a pre-existing column, so a frame that
#: already carries it comes back with `team_in_possession_x`/`_y` and every consumer that
#: re-derives (`_xshot_occurrence`, `_xcross_attempt`) dies on `KeyError:
#: 'team_in_possession'`. Supplying it per-family keeps this gate probing the dup-action_id
#: bug instead of that unrelated non-idempotency.
_NEEDS_TEAM_IN_POSSESSION = {"das_xfns"}


def _frame_for(name):
    """The shared frame, plus whatever contract columns `name` additionally requires.

    ADR-043: `das_xfns` previously swallowed `_validate_das_inputs`' missing-column
    ValueError and returned NaN, so this auto-enumerating gate probed it VACUOUSLY --
    the family returned before any frame_id lookup ran, and the dup-action_id bug it
    carries was therefore never detected here.
    """
    fr = _frame()
    if name in _NEEDS_TEAM_IN_POSSESSION:
        fr = fr.copy()
        fr["team_in_possession"] = 1
    return fr


# Construction MUST succeed (no silent skip) -- an unconstructable factory is a gate
# FAILURE, not a skip, so a family can never go unprobed (no-silent-caps discipline).
_CONSTRUCT_ALLOWLIST: set[str] = set()  # factories that genuinely cannot construct (none today)


def _build(name, mod=F):
    fac = getattr(mod, name)
    if isinstance(fac, list):
        return fac
    xt = _xt()
    # home_team_id=1 preferred; (xt, home_team_id=1) for xt-takers; (xt,) bare for the
    # ADR-055 pair (gk_influence_xfns / cover_shadow_xfns take an OPTIONAL goal_map and no
    # home_team_id at all); bare for factories that take neither (e.g.
    # pitch_control_xfns(method=...), elastic_sync_xfns(*,...)).
    #
    # `goal_map` is deliberately NOT supplied: the factory then builds it from whatever
    # frames the transformer is called with, which is the production path and the one this
    # gate is about -- a duplicated action_id in a gamestate slot.
    # ORDER MATTERS, and it bit this edit: `(xt,)` bare must come LAST. `pitch_control_xfns`
    # takes `method` as its first positional, so `pitch_control_xfns(xt)` CONSTRUCTS happily
    # -- binding the xT model to `method` -- and only fails later, inside the transformer,
    # as "Unknown method '<ExpectedThreat object>'". A construction probe that guesses by
    # try/except cannot tell a wrong-but-accepted binding from a right one, so the shapes
    # are ordered most- to least-specific and the ambiguous one is the fallback.
    for args, kw in (
        ((), {"home_team_id": 1}),
        ((xt,), {"home_team_id": 1}),
        ((), {}),
        ((xt,), {}),
    ):
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

#: The atomic module's full declared `_xfns` surface -- used by the META-assertion below.
_ATOMIC_XFNS_NAMES = sorted(n for n in dir(AF) if n.endswith("_xfns"))

#: The subset that is a genuine atomic MIRROR -- used by the BEHAVIOURAL parametrisation.
#:
#: `atomic.tracking.features` re-exports several factories from `silly_kicks.tracking` as the
#: IDENTICAL object (`cover_shadow_xfns`, `obso_xfns`, `pausa_xfns` -- verified with `is`). Those
#: consume SPADL-shaped actions and are already exercised by `_XFNS_NAMES` above; handing them the
#: atomic fixture raises `KeyError: 'start_x'`, which is a fixture gap by construction rather than
#: a defect. Filtering on OBJECT IDENTITY rather than a hand-listed exclusion set means a factory
#: that later acquires a real atomic mirror is picked up automatically.
_ATOMIC_MIRROR_NAMES = sorted(n for n in _ATOMIC_XFNS_NAMES if getattr(AF, n) is not getattr(F, n, None))


def test_meta_gate_covers_every_xfns_factory():
    """Two INDEPENDENT derivations must agree: the runtime namespace and the declared export.

    The previous version asserted
        set(_XFNS_NAMES) == {n for n in dir(F) if n.endswith("_xfns")}
    -- the same expression on both sides, always true. It also carried
    `assert len(_XFNS_NAMES) >= 21  # bumped for xt_gk_xfns`, a floor inside the very gate that
    exists because floors cannot detect an omission, with a comment recording it had already been
    hand-bumped once (Cycle B).

    The independent source is the PACKAGE export, not `features.__all__`. Measured: all four names
    absent from `features.__all__` ARE in `tracking.__all__`, so pairing against the module surface
    would manufacture four findings that are not defects.
    """
    exported = {n for n in T.__all__ if n.endswith("_xfns")}
    assert set(_XFNS_NAMES) == exported, (
        f"runtime namespace and package export disagree: "
        f"dir-only={sorted(set(_XFNS_NAMES) - exported)}, "
        f"export-only={sorted(exported - set(_XFNS_NAMES))}"
    )
    assert not _CONSTRUCT_ALLOWLIST, "no construct-skips are expected today"


def test_meta_gate_covers_every_ATOMIC_xfns_factory():
    """The gate enumerated `dir(F)` over tracking.features ONLY, so the atomic mirrors had never
    been covered by ADR-020's dup-action_id contract at all.

    `atomic.tracking.features` exports via the SUBMODULE, not the package, so its declared surface
    is `AF.__all__`. This caught `xshot_occurrence_xfns` missing from it while its sibling
    `xcross_attempt_xfns` was present.
    """
    exported = {n for n in AF.__all__ if n.endswith("_xfns")}
    assert set(_ATOMIC_XFNS_NAMES) == exported, (
        f"atomic runtime namespace and declared export disagree: "
        f"dir-only={sorted(set(_ATOMIC_XFNS_NAMES) - exported)}, "
        f"export-only={sorted(exported - set(_ATOMIC_XFNS_NAMES))}"
    )


def _run_family(name, mod=F):
    """Run every frame-aware transformer of `name` through a dup-action_id gamestate.
    Discriminates the target bug from a fixture gap so the fix lands on the bug, not the fixture."""
    states = gamestates(_atomic_actions() if mod is AF else _actions(), nb_prev_actions=3)
    assert states[1]["action_id"].duplicated().any()  # precondition: dup exists
    frame = _frame_for(name)
    for t in _build(name, mod):
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


@pytest.mark.parametrize("name", _ATOMIC_MIRROR_NAMES)
def test_atomic_xfns_survives_duplicate_action_id_gamestate(name):
    """Cycle B: the atomic mirrors were never covered by ADR-020's contract."""
    _run_family(name, AF)
