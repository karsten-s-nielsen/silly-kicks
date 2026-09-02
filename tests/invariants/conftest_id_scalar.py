"""Enumerated public id-SCALAR surface for the ADR-019 dtype contract (ADR-043, 4.53.0).

THE BOUNDARY
------------
A public function that takes an id-valued **scalar** (``home_team_id``, ``attacking_team_id``,
``gk_team_id``, ``team_id``, ``carrier_player_id``, ...) and compares it against an id
**column**. When the scalar's dtype differs from the column's, a raw ``==``/``!=`` silently
mis-resolves -- ``Int64(366) == "366"`` is ``False`` -- and the caller gets an all-False mask,
not an error.

That is not hypothetical. It shipped in ``spadl/utils.py::play_left_to_right``, where
``team_id != home_team_id`` was True for EVERY row on an object-string column, so the HOME rows
were mirrored 180 degrees too -- not merely the away rows missed.

WHY A REGISTRY AND NOT A LINT
-----------------------------
This gate REPLACES ``tests/tracking/test_id_compat_lint.py`` (deleted in 4.53.0). That lint was
incomplete **by construction**, and each failure was the same failure:

* It was a NAME heuristic. It missed the ADR-027 defect (``t != action_team`` in
  ``_line_breaking.py``'s Ward branch) purely because the operands are not named ``*_id``.
* The safe and unsafe cases are the **identical AST**. ``features.py``'s
  ``(frames["team_id"] == tid)`` -- where ``tid`` was drawn from that same column -- is SAFE,
  while ``_gk_influence.py``'s ``players["player_id"] == gk_player_id`` -- where the scalar is a
  public parameter -- is the BOUNDARY. Only *provenance* separates them, and no syntactic rule
  can see provenance. Widening the heuristic would flag correct code and breed ``noqa``
  exemptions; narrowing it re-opens the hole.
* Its glob missed 17 modules until this release.

So: **complete by ENUMERATION where the lint was incomplete by HEURISTIC.** Same idiom as
ADR-003's NaN-safety registry and ADR-033's ``PURITY_ENTRIES``.

HOW THE SURFACE IS ENUMERATED (``discover_public_id_scalar_functions``)
----------------------------------------------------------------------
Mechanical, from signatures -- never a hand-list, because a hand-list rots exactly like the
lint's glob did:

1. Walk every PUBLIC module of ``spadl/``, ``atomic/``, ``vaep/``, ``causal/``, ``tracking/``
   (skipping ``_private`` modules -- their surface reaches users only via a package re-export,
   which the walk sees anyway).
2. Take each module's ``__all__`` entries that are functions or classes. ``__all__`` IS the
   public surface **where a module declares one**.

   Where it does NOT, fall back to the module's public module-level callables DEFINED in that
   module (``obj.__module__ == mod.__name__``, so re-imported names do not double-count).
   This fallback is not a nicety: 35 of the walked modules declare no ``__all__``, and an
   ``__all__``-only rule contributed NOTHING from any of them -- leaving 13 public id-scalar
   callables in no bucket at all, among them EVERY provider ``convert_to_actions`` and every
   native ``convert_to_frames``. Those are the most-called public entry points in the library
   and the exact seam ADR-019 was written for, so an enumeration that skipped them reproduced
   the deleted lint's defining failure (a discovery rule that silently stops looking) inside
   the gate built to replace it.
3. ``inspect.signature`` each one, and keep those with an **id-scalar parameter**:
   a name ending in ``_id`` (singular), plus the explicitly-listed entity-id COLLECTION
   params (``_ENTITY_ID_COLLECTION_PARAMS``) -- ``add_gk_role(goalkeeper_ids=...)`` resolves
   its set against a ``player_id`` column via ``.isin()``, which carries the identical dtype
   hazard and would otherwise escape a singular-only rule.
4. Key each hit by its DEFINING ``module.qualname``, so the many re-exports
   (``silly_kicks.tracking.add_obso`` / ``.features.add_obso``) collapse to one entry.

Every discovered function must land in exactly one of three buckets, or the meta-assertion
fails CI:

``PUBLIC_ID_SCALAR_ENTRIES``
    Directly exercised here: invoked TWICE -- once with a dtype-matched scalar, once with a
    **mismatched-but-value-equal** one (``5`` vs ``"5"``) -- and the outputs must be IDENTICAL.
``COVERED_BY_AGGREGATOR_GATE``
    Delegated to ``tests/tracking/test_id_dtype_invariance.py``, whose ``(False, False, True)``
    permutation IS this exact axis (a string ``home_team_id`` against numeric inputs). The
    delegation is MACHINE-CHECKED against that gate's own registered surface, not asserted in
    prose -- see ``test_delegated_entries_are_really_covered``.
``NOT_INVARIANT``
    Genuinely cannot be invariant. Each carries a written justification.
``NOT_EXERCISABLE``
    Believed invariant, but no matched/mismatched pair EXISTS to probe it with (a fixed-string
    id space, or a writer that never compares). Kept distinct from ``NOT_INVARIANT`` so the
    reason is not misreported, and each carries a written justification naming the obstruction.

Excluded by the rule, deliberately: plural ``*_ids`` params that are not entity-id collections
resolved against a column -- SPADL type/result enums (``shot_type_ids``, ``outcome_result_ids``),
period numbers (``period_ids``), statistical cluster labels (``cluster_ids``), and output
containers (``PitchControlSurface.player_ids``, ``TrackingConversionReport.unrecognized_player_ids``).
None of these is a caller-supplied entity id resolved against a provider-dtyped id column, so
none can exhibit the defect this gate exists to catch.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from tests.tracking._goal_map_helpers import goal_map_like_home_team_id

# --------------------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------------------

#: The packages ADR-019 names as consuming the id-identity contract. ``silly_kicks.keeper_identity``
#: is a MODULE (not a package) added when the keeper-identity resolver was promoted out of
#: ``tracking/_keeper_identity.py`` (breaking move, no shim): ``KeeperIdentity`` is discovered here
#: rather than via the old ``tracking.features`` re-export. ``_public_modules`` handles a plain module
#: (no ``__path__`` -> it is walked as a single module).
PACKAGES = (
    "silly_kicks.spadl",
    "silly_kicks.atomic",
    "silly_kicks.vaep",
    "silly_kicks.causal",
    "silly_kicks.tracking",
    "silly_kicks.keeper_identity",
)

#: Entity-id COLLECTIONS resolved against an id column (``.isin``). Same hazard as a scalar:
#: a ``{"366"}`` set against an ``Int64`` player_id column matches nothing, silently.
#: Deliberately an explicit list -- a blanket "any ``*_ids`` param" rule would sweep in type
#: enums and output containers, which are not boundaries and would only breed exemptions.
_ENTITY_ID_COLLECTION_PARAMS = frozenset({"goalkeeper_ids"})


def _is_id_scalar_param(name: str) -> bool:
    """An id-valued caller-supplied argument resolved against an id column."""
    if name in _ENTITY_ID_COLLECTION_PARAMS:
        return True
    return name.endswith("_id") and not name.endswith("_ids")


def _public_modules(root: str) -> list[Any]:
    mod = importlib.import_module(root)
    mods = [mod]
    if hasattr(mod, "__path__"):
        for info in pkgutil.walk_packages(mod.__path__, root + "."):
            leaf = info.name.rsplit(".", 1)[-1]
            if leaf.startswith("_") or "._" in info.name:
                continue
            try:
                mods.append(importlib.import_module(info.name))
            except Exception as exc:  # pragma: no cover - an unimportable optional-extra module
                # Surfaced, never silent: a module that cannot import is invisible to the
                # meta-assertion, so an id-scalar function inside it would go unregistered
                # WITHOUT failing CI -- the exact silent-gap class this gate replaces.
                warnings.warn(
                    f"id-scalar discovery skipped {info.name}: {type(exc).__name__}: {exc}",
                    stacklevel=2,
                )
    return mods


def _public_names(mod: Any) -> list[str]:
    """The module's public surface: ``__all__`` when declared, else its own public callables.

    The fallback keys on ``obj.__module__ == mod.__name__`` so a name merely IMPORTED into the
    module (``pandas``, a sibling helper) is not mistaken for surface this module publishes --
    that would key entries under a foreign module and make the registry unstable.

    Without the fallback, a module that simply never wrote an ``__all__`` contributes NOTHING
    and every public id-scalar function inside it is invisible to the meta-assertion. That is
    the silent-gap class this gate exists to eliminate, so absence of ``__all__`` must mean
    "infer the surface", never "there is no surface".
    """
    declared = getattr(mod, "__all__", None)
    if declared:
        return list(declared)
    return [
        name
        for name, obj in vars(mod).items()
        if not name.startswith("_")
        and (inspect.isfunction(obj) or inspect.isclass(obj))
        and getattr(obj, "__module__", None) == mod.__name__
    ]


def discover_public_id_scalar_functions() -> dict[str, tuple[str, ...]]:
    """``{defining module.qualname: (id-scalar param names,)}`` over the public surface.

    Signature-driven, so a NEW public function taking an id scalar appears here the moment it
    is exported -- which is what forces it into one of the three buckets below.
    """
    found: dict[str, tuple[str, ...]] = {}
    for pkg in PACKAGES:
        for mod in _public_modules(pkg):
            for name in _public_names(mod):
                obj = getattr(mod, name, None)
                if not (inspect.isfunction(obj) or inspect.isclass(obj)):
                    continue
                try:
                    sig = inspect.signature(obj)
                except (TypeError, ValueError):  # pragma: no cover - builtins/C types
                    continue
                ids = tuple(sorted(p for p in sig.parameters if _is_id_scalar_param(p)))
                if ids:
                    found[f"{obj.__module__}.{obj.__qualname__}"] = ids
    return found


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------

HOME = 5  # home / acting / attacking team in every fixture below
AWAY = 6
GK_HOME, GK_AWAY = 1, 2  # goalkeeper player ids


def as_str(value: int) -> str:
    """Canonical string form of an int id.

    ``str(int(v))`` and NOT ``str(v)``: on a float-backed column ``str(5.0)`` is ``"5.0"``,
    a DIFFERENT id -- the "mismatched" leg would then fail for the wrong reason and the gate
    would be measuring value-inequality, not dtype-invariance.
    """
    return str(int(value))


def spadl_actions(dtype: str = "int64") -> pd.DataFrame:
    """Two home rows + one away row at distinguishable coordinates."""
    values = [HOME, HOME, AWAY]
    team = pd.Series([as_str(v) for v in values], dtype=object) if dtype == "object" else pd.Series(values, dtype=dtype)
    return pd.DataFrame(
        {
            "game_id": [1, 1, 1],
            "period_id": [1, 1, 1],
            "action_id": [0, 1, 2],
            "time_seconds": [10.0, 20.0, 30.0],
            "team_id": team,
            "player_id": pd.Series([10, 11, 20], dtype="int64"),
            "start_x": [10.0, 20.0, 30.0],
            "start_y": [10.0, 20.0, 30.0],
            "end_x": [40.0, 50.0, 60.0],
            "end_y": [15.0, 25.0, 35.0],
            "type_id": [0, 0, 0],
            "result_id": [1, 1, 1],
            "bodypart_id": [0, 0, 0],
        }
    )


def atomic_actions(dtype: str = "int64") -> pd.DataFrame:
    values = [HOME, HOME, AWAY]
    team = pd.Series([as_str(v) for v in values], dtype=object) if dtype == "object" else pd.Series(values, dtype=dtype)
    return pd.DataFrame(
        {
            "game_id": [1, 1, 1],
            "period_id": [1, 1, 1],
            "action_id": [0, 1, 2],
            "time_seconds": [10.0, 20.0, 30.0],
            "team_id": team,
            "player_id": pd.Series([10, 11, 20], dtype="int64"),
            "x": [10.0, 20.0, 30.0],
            "y": [10.0, 20.0, 30.0],
            "dx": [1.0, 2.0, 3.0],
            "dy": [-1.0, -2.0, -3.0],
            "type_id": [0, 0, 0],
            "bodypart_id": [0, 0, 0],
        }
    )


#: The player id on the KEEPER action row of the gk_role fixtures. Deliberately NOT the
#: keeper's own id: see :func:`gk_role_actions`.
GK_ROLE_KEEPER_ROW_PLAYER = "77"


def gk_role_actions() -> pd.DataFrame:
    """Actions whose ``player_id`` column is object-string, so an int ``goalkeeper_ids`` set
    (and vice-versa) exercises the ``.isin`` boundary.

    Row 0 is a keeper action; row 1 is the follow-up that ``goalkeeper_ids`` promotes to
    ``distribution``.

    THE KEEPER ROW IS ATTRIBUTED TO A DIFFERENT PLAYER (``GK_ROLE_KEEPER_ROW_PLAYER``), and
    that is the whole point. ``add_gk_role`` links a keeper action to the follow-up by
    ``same_player`` FIRST, and only then widens the link via ``goalkeeper_ids`` (rule (a):
    "the current actor is a declared GK on the same team"). An earlier version of this fixture
    put the SAME player on both rows -- so ``same_player`` already produced ``distribution``,
    ``goalkeeper_ids`` never decided anything, and the entry returned an identical ``gk_role``
    for ``{"1"}``, ``{1}`` and ``None`` alike. It passed the dtype-invariance assertion for a
    reason that had nothing to do with id resolution, which is the vacuity
    ``test_entity_id_collection_is_load_bearing`` now rejects outright.

    Splitting the ids makes rule (a) the ONLY route to ``distribution``, so the ``.isin``
    comparison decides the answer and a raw one changes it. This is also the shape rule (a)
    exists to serve: a provider that attributes the keeper action to an id other than the
    roster GK id the caller supplies.

    A non-null id is used rather than a null one so rule (b) (the both-players-NA team
    fallback) provably cannot fire -- the probe then isolates the ``.isin`` boundary alone.
    """
    import silly_kicks.spadl.config as spadlconfig

    acts = spadl_actions()
    acts["player_id"] = pd.Series([GK_ROLE_KEEPER_ROW_PLAYER, as_str(GK_HOME), "11"], dtype=object)
    acts["type_id"] = [spadlconfig.actiontype_id["keeper_save"], spadlconfig.actiontype_id["pass"], 0]
    acts["start_x"] = [5.0, 8.0, 60.0]
    return acts


def gk_role_atomic_actions() -> pd.DataFrame:
    """Atomic counterpart of :func:`gk_role_actions`, including its split-player-id design.

    ``atomic.spadl.utils.add_gk_role`` reads ``x`` (NOT ``start_x``) for the penalty-area
    threshold -- documented in its own docstring as the one atomic-specific adaptation -- so
    the standard-SPADL fixture crashes on it. Different schema, not different semantics.
    """
    import silly_kicks.atomic.spadl.config as atomicconfig

    acts = atomic_actions()
    acts["player_id"] = pd.Series([GK_ROLE_KEEPER_ROW_PLAYER, as_str(GK_HOME), "11"], dtype=object)
    acts["type_id"] = [atomicconfig.actiontype_id["keeper_save"], atomicconfig.actiontype_id["pass"], 0]
    acts["x"] = [5.0, 8.0, 60.0]
    return acts


def tracking_actions() -> pd.DataFrame:
    from tests.tracking.conftest_id_dtype import make_actions

    acts = make_actions()
    # `convert_to_atomic` carries `original_event_id` through to every synthesized atom, so an
    # actions frame lacking it dies in the atomic-mirror entries. Cheap to supply here; keeps
    # the standard and atomic arms on ONE fixture.
    acts["original_event_id"] = pd.Series(["e0", "e1"], dtype=object)
    return acts


def tracking_frames() -> pd.DataFrame:
    from tests.tracking.conftest_id_dtype import make_frames

    return make_frames()


#: xS scores only where the ball is in the in-possession team's ATTACKING THIRD (within 35 m of
#: the attacked goal); xCross additionally requires a WIDE ball (y < 14 or y > 54). Team 5
#: attacks x=105 in the shared fixture, so (85, 8) satisfies BOTH domains at once.
_ATTACKING_WIDE_BALL = (85.0, 8.0)

#: The two action timestamps in the shared tracking fixture (``make_actions``).
_ACTION_TIMES = (10.0, 20.0)


def frames_attacking_wide() -> pd.DataFrame:
    """Shared tracking frames with a team-5 carrier on the ball in the attacking-third wide area.

    Three things must ALL hold for xS/xCross to score a frame, and the generic fixture supplies
    none of them:

    1. The ball must be in the in-possession team's attacking third (xS) and wide (xCross).
       The generic fixture parks it at x~58/y~32 -- in NEITHER domain.
    2. A team-5 player must be ON the ball, because ``compute_*`` derives possession via
       ``infer_ball_carrier`` (default tolerance ~3 m). With the nearest player ~24 m away no
       carrier resolves, ``team_in_possession`` is NaN, and every frame is skipped.
    3. ``team_in_possession`` must NOT be pre-set. ``derive_team_in_possession`` merges its
       result in without checking for an existing column, so a frame that already carries one
       comes back with ``team_in_possession_x``/``_y`` -- the same non-idempotency documented in
       tests/tracking/test_frame_aware_xfns_dup_action_id.py.

    Turning the domain filters off instead would have tested a configuration nobody serves.

    COST: only the two ACTION-TIME frames are moved in-domain, not all ten. Every scored frame
    costs a pitch-control surface, and these entries each run twice -- putting all ten in domain
    took the family from seconds to minutes of CI. Two scored frames (one per action, so the
    action-linked ``add_*``/``xfns`` paths still resolve) is the smallest fixture that keeps
    every entry live.
    """
    frames = tracking_frames()
    in_domain = frames["time_seconds"].isin(_ACTION_TIMES)
    ball = frames["is_ball"].astype(bool) & in_domain
    frames.loc[ball, "x"] = _ATTACKING_WIDE_BALL[0]
    frames.loc[ball, "y"] = _ATTACKING_WIDE_BALL[1]
    # Player 10 (team 5) carries the ball -- just inside the carrier tolerance.
    carrier = (frames["player_id"].astype("Int64") == 10) & in_domain
    frames.loc[carrier, "x"] = _ATTACKING_WIDE_BALL[0] + 0.5
    frames.loc[carrier, "y"] = _ATTACKING_WIDE_BALL[1]
    return frames


def ghost_frames() -> pd.DataFrame:
    """Action-time frames only, for the ghost-GK family.

    ``compute_ghost_gk`` evaluates a 60x64 KDE grid PER FRAME, and every entry runs twice, so
    the full ten-timestamp fixture makes this family dominate the gate's wall clock by an order
    of magnitude. Two frames exercise the same seam (``home_team_id`` decides the goal-side
    flip) at a fifth of the cost.
    """
    frames = tracking_frames()
    return frames[frames["time_seconds"].isin(_ACTION_TIMES)].reset_index(drop=True).copy()


def single_frame() -> pd.DataFrame:
    """One timestamp slice of the shared tracking fixture (per-frame primitives)."""
    frames = tracking_frames()
    return frames[frames["time_seconds"] == frames["time_seconds"].max()].reset_index(drop=True).copy()


def xt_model():
    from silly_kicks.xthreat import ExpectedThreat

    xt = ExpectedThreat(l=16, w=12)
    xt.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))
    return xt


def gamestates_2():
    from silly_kicks.vaep.feature_framework import gamestates

    return gamestates(tracking_actions(), nb_prev_actions=2)


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class IdScalarEntry:
    """One public function that takes an id scalar.

    ``invoke(scalar)`` must call the target with that scalar wherever an id scalar is
    accepted, and return something comparable. The gate calls it twice -- ``matched`` then
    ``mismatched`` -- and requires identical output.
    """

    key: str  # defining module.qualname -- must match discovery
    invoke: Callable[[Any], Any]
    matched: Any = HOME
    mismatched: Any = field(default_factory=lambda: as_str(HOME))
    #: Columns whose LIVENESS the non-vacuity check must judge, for functions that return
    #: their input frame plus a computed column. Without this the passthrough columns keep the
    #: generic "something is non-null" check green while the ONLY column the function actually
    #: computed is entirely NaN -- i.e. exactly the vacuous comparison the check exists to
    #: prevent, wearing a passing badge.
    live_columns: tuple[str, ...] = ()


def _e(
    key: str,
    invoke: Callable[[Any], Any],
    *,
    matched: Any = HOME,
    mismatched: Any | None = None,
    live_columns: tuple[str, ...] = (),
) -> IdScalarEntry:
    return IdScalarEntry(
        key=key,
        invoke=invoke,
        matched=matched,
        mismatched=as_str(matched) if mismatched is None else mismatched,
        live_columns=live_columns,
    )


def _xfn_outputs(factory_result: list, states=None, frames=None) -> list:
    """Run every transformer a ``*_xfns`` factory produced.

    A factory that merely CLOSES OVER the scalar would compare equal on the returned list
    object regardless of correctness, so the closures must actually be executed -- that is
    where the boundary comparison lives.
    """
    states = gamestates_2() if states is None else states
    frames = tracking_frames() if frames is None else frames
    out = []
    for transformer in factory_result:
        out.append(transformer(states, frames) if getattr(transformer, "_frame_aware", False) else transformer(states))
    return out


# ---- spadl / atomic / vaep --------------------------------------------------------------


def _spadl_entries() -> list[IdScalarEntry]:
    import silly_kicks.atomic.spadl.utils as atomic_spadl_utils
    import silly_kicks.atomic.vaep.features as atomic_vaep_features
    import silly_kicks.spadl.utils as spadl_utils
    import silly_kicks.vaep.features.core as vaep_core
    from silly_kicks.spadl.orientation import (
        ABSOLUTE_FRAME_HOME_RIGHT,
        PER_PERIOD_ABSOLUTE,
        to_spadl_ltr,
    )

    def _states(frame):
        # The vaep helpers mutate in place -- hand each call its own copies.
        return [frame.copy(), frame.copy()]

    return [
        # The four `play_left_to_right` siblings: the KNOWN-LIVE shape (see module docstring).
        _e("silly_kicks.spadl.utils.play_left_to_right", lambda s: spadl_utils.play_left_to_right(spadl_actions(), s)),
        _e(
            "silly_kicks.atomic.spadl.utils.play_left_to_right",
            lambda s: atomic_spadl_utils.play_left_to_right(atomic_actions(), s),
        ),
        _e(
            "silly_kicks.vaep.features.core.play_left_to_right",
            lambda s: vaep_core.play_left_to_right(_states(spadl_actions()), s),
        ),
        _e(
            "silly_kicks.atomic.vaep.features.play_left_to_right",
            lambda s: atomic_vaep_features.play_left_to_right(_states(atomic_actions()), s),
        ),
        # Both orientation conventions of the shared entry point.
        _e(
            "silly_kicks.spadl.orientation.to_spadl_ltr",
            lambda s: (
                to_spadl_ltr(spadl_actions(), input_convention=ABSOLUTE_FRAME_HOME_RIGHT, home_team_id=s),
                to_spadl_ltr(
                    spadl_actions(),
                    input_convention=PER_PERIOD_ABSOLUTE,
                    home_team_id=s,
                    home_attacks_right_per_period={1: True, 2: False},
                ),
            ),
        ),
        # `.isin()` against a player_id column -- the entity-id COLLECTION shape.
        _e(
            "silly_kicks.spadl.utils.add_gk_role",
            lambda s: spadl_utils.add_gk_role(gk_role_actions(), goalkeeper_ids=s),
            matched={as_str(GK_HOME)},
            mismatched={GK_HOME},
        ),
        _e(
            "silly_kicks.atomic.spadl.utils.add_gk_role",
            lambda s: atomic_spadl_utils.add_gk_role(gk_role_atomic_actions(), goalkeeper_ids=s),
            matched={as_str(GK_HOME)},
            mismatched={GK_HOME},
        ),
    ]


# ---- provider converters (the no-``__all__`` surface) -------------------------------------
#
# Every one of these was invisible to the ``__all__``-only walk, yet ``convert_to_actions`` /
# ``convert_to_frames`` are the library's most-called public entry points and `home_team_id`
# is the single scalar ADR-019 was written about. Each entry reuses the provider suite's own
# vetted minimal fixture rather than inventing a second one that could drift out of the
# converter's accepted shape silently.


def _converter_entries() -> list[IdScalarEntry]:
    import silly_kicks.spadl.gradientsports as gs_events
    import silly_kicks.spadl.metrica as metrica_events
    import silly_kicks.spadl.opta as opta_events
    import silly_kicks.spadl.sportec as sportec_events
    import silly_kicks.spadl.statsbomb as sb_events
    import silly_kicks.spadl.wyscout as wyscout_events
    import silly_kicks.tracking.gradientsports as gs_frames
    import silly_kicks.tracking.skillcorner as sk_frames
    import silly_kicks.tracking.sportec as sportec_frames
    from silly_kicks.tracking import direction
    from tests.spadl.test_gradientsports import _df_minimal_pass as gs_events_fixture
    from tests.spadl.test_metrica import _df_metrica_pass_by_gk
    from tests.spadl.test_opta import TestOptaPreserveNative
    from tests.spadl.test_sportec import _df_minimal_pass as sportec_events_fixture
    from tests.spadl.test_statsbomb import _make_statsbomb_events
    from tests.spadl.test_wyscout import _make_pass_event
    from tests.tracking.test_adapter_extra_time_orientation import _raw_4period
    from tests.tracking.test_finalize_orientation import _raw
    from tests.tracking.test_skillcorner_builder import _bronze as sk_bronze

    # Several provider fixtures use GENUINELY string ids ("Home", "DFL-CLU-A", "H"). A string
    # like that has no value-equal numeric twin, so the gate's axis is inexpressible on it --
    # and a matched==mismatched pair would make the entry vacuous, which
    # `test_matched_and_mismatched_scalars_differ_in_dtype_not_value` rejects outright. Those
    # fixtures are therefore re-keyed onto a NUMERIC-VALUED id (still stored as the provider's
    # native string dtype), which is exactly the shape the defect needs: an object column of
    # digit-strings against an int scalar. Re-keying the fixture is honest -- provider ids are
    # opaque tokens, and a digit-string is as valid a DFL/Metrica id as any other.
    NUM_TEAM, NUM_GK = 5, 999

    def _wyscout_events():
        """A pass plus a GK aerial duel, so BOTH wyscout id-scalar axes are live at once.

        The duel is what makes `goalkeeper_ids` load-bearing: wyscout reclassifies an air duel
        by a declared keeper into `keeper_claim`, and drops it entirely otherwise -- so the set
        resolving or not is the difference between a keeper_claim row and no row.
        """
        duel = _make_pass_event(event_id=2, milliseconds=6000.0)
        duel.update(
            {
                "type_id": 1,  # _WS_TYPE_DUEL
                "subtype_id": 10,  # _WS_SUBTYPE_AIR_DUEL
                "type_name": "Duel",
                "subtype_name": "Air duel",
                "player_id": NUM_GK,
            }
        )
        return pd.DataFrame([_make_pass_event(event_id=1, milliseconds=5000.0), duel])

    def _metrica_events_numeric():
        df = _df_metrica_pass_by_gk()
        df["team"] = [as_str(NUM_TEAM)]
        df["player"] = [as_str(NUM_GK)]
        return df

    def _sportec_events_numeric():
        df = sportec_events_fixture()
        df["team"] = [as_str(NUM_TEAM)]
        df["player_id"] = [as_str(NUM_GK)]
        return df

    def _numeric_team_frame():
        """`_p1_frame` re-keyed onto numeric-valued team ids, for the direction primitives."""
        return pd.DataFrame(
            [
                _raw(1, as_str(NUM_TEAM), "hgk", True, False, 20.0, 34.0),
                _raw(1, as_str(NUM_TEAM + 1), "agk", True, False, 85.0, 34.0),
                _raw(1, None, None, False, True, 50.0, 34.0),
            ]
        )

    return [
        # ---- events converters: `home_team_id` reaches `to_spadl_ltr` -> `ids_match` ----
        _e(
            "silly_kicks.spadl.gradientsports.convert_to_actions",
            lambda s: gs_events.convert_to_actions(
                gs_events_fixture(), home_team_id=s, home_team_start_left=True, home_team_start_left_extratime=True
            ),
            matched=100,  # the gradientsports fixture's own home team
        ),
        _e(
            # BOTH id-scalar params at once (the `extract_xcross_features` tuple idiom), because
            # the registry keys per FUNCTION and `goalkeeper_ids` is the sharper axis here: it
            # drives a `.isin()` against the `player` column, so the fixture is a GK pass, where
            # resolving the set actually CHANGES the emitted action type.
            "silly_kicks.spadl.metrica.convert_to_actions",
            lambda s: metrica_events.convert_to_actions(
                _metrica_events_numeric(), home_team_id=s[0], goalkeeper_ids=s[1], home_team_start_left=True
            ),
            matched=(NUM_TEAM, {NUM_GK}),
            mismatched=(as_str(NUM_TEAM), {as_str(NUM_GK)}),
        ),
        _e(
            # `goalkeeper_ids` is accepted for cross-provider API symmetry and explicitly
            # discarded (`_ = goalkeeper_ids`, opta.py L145), so `home_team_id` is the only
            # live axis; it reaches `to_spadl_ltr` -> `ids_match`.
            "silly_kicks.spadl.opta.convert_to_actions",
            lambda s: opta_events.convert_to_actions(TestOptaPreserveNative._events_with_extras(), home_team_id=s),
            matched=157,  # the opta fixture's own home team
        ),
        _e(
            "silly_kicks.spadl.sportec.convert_to_actions",
            lambda s: sportec_events.convert_to_actions(
                _sportec_events_numeric(), home_team_id=s[0], goalkeeper_ids=s[1], home_team_start_left=True
            ),
            matched=(NUM_TEAM, {NUM_GK}),
            mismatched=(as_str(NUM_TEAM), {as_str(NUM_GK)}),
        ),
        _e(
            # StatsBomb is POSSESSION_PERSPECTIVE, so `to_spadl_ltr` is a documented no-op and
            # `goalkeeper_ids` is discarded -- invariance holds trivially TODAY. Registered
            # anyway: the entry is what makes a future change to either branch prove it stayed
            # invariant instead of silently acquiring the defect.
            "silly_kicks.spadl.statsbomb.convert_to_actions",
            lambda s: sb_events.convert_to_actions(_make_statsbomb_events(), home_team_id=s),
            matched=100,
        ),
        _e(
            # BOTH id-scalar params, like metrica/sportec. `goalkeeper_ids` was a LIVE ADR-019
            # defect found by this registry: `events["player_id"].isin(goalkeeper_ids)` was raw
            # and uncast, so against an int64 `player_id` column a caller's {"999"} matched
            # NOTHING and the aerial duel silently stayed a duel (dropped as a non_action)
            # instead of becoming keeper_claim -- measured, not inferred, and proven
            # pre-existing via `git show HEAD:silly_kicks/spadl/wyscout.py`. Fixed in 4.53.0 by
            # routing it through `ids_isin`; this entry is what keeps it fixed.
            "silly_kicks.spadl.wyscout.convert_to_actions",
            lambda s: wyscout_events.convert_to_actions(_wyscout_events(), s[0], goalkeeper_ids=s[1]),
            matched=(100, {NUM_GK}),  # the wyscout fixture's own home team
            mismatched=(as_str(100), {as_str(NUM_GK)}),
        ),
        # ---- tracking builders: `home_team_id` reaches finalize_orientation / play_ltr ----
        _e(
            "silly_kicks.tracking.gradientsports.convert_to_frames",
            lambda s: gs_frames.convert_to_frames(
                _raw_4period(57, 99, 1, 2),
                home_team_id=s,
                home_team_start_left=True,
                home_team_start_left_extratime=False,
                output_convention="ltr",
            ),
            matched=57,
        ),
        _e(
            "silly_kicks.tracking.sportec.convert_to_frames",
            # Numeric-valued team ids: the sportec builder is id-opaque, and a genuine
            # "DFL-CLU-A" has no dtype twin to compare against (see the NUM_TEAM note above).
            lambda s: sportec_frames.convert_to_frames(
                _raw_4period(as_str(NUM_TEAM), as_str(NUM_TEAM + 1), "HGK", "AGK"),
                home_team_id=s,
                home_team_start_left=True,
                home_team_start_left_extratime=False,
                output_convention="ltr",
            ),
            matched=NUM_TEAM,
        ),
        _e(
            "silly_kicks.tracking.skillcorner.convert_to_frames",
            # The bronze carries `team=31` as an int and the builder resolves ids by `str()`
            # casting both sides, so "31" vs 31 is the pair. DECLARED STRING-FIRST deliberately:
            # the gate's third axis (a FLOAT-valued scalar) fires only on a numeric `matched`,
            # and this builder does NOT survive it -- `str(31.0)` is "31.0", which matches zero
            # player rows and RAISES "refusing to guess orientation". That is a real limit of a
            # bare `str()` cast versus `canonical_id` (which collapses 31, "31" and 31.0 alike),
            # recorded here rather than hidden: it fails LOUD rather than mis-resolving silently,
            # which is why it is a fragility and not the silent defect this gate hunts.
            lambda s: sk_frames.convert_to_frames(sk_bronze(), home_team_id=s, assume_standard_pitch=True),
            matched=as_str(31),
            mismatched=31,
        ),
        # ---- orientation primitives ----
        _e(
            "silly_kicks.tracking.direction.compute_attacking_direction",
            lambda s: direction.compute_attacking_direction(
                team_id=_numeric_team_frame()["team_id"],
                period_id=_numeric_team_frame()["period_id"],
                is_ball=_numeric_team_frame()["is_ball"],
                home_team_id=s,
                home_team_start_left=True,
            ),
            matched=NUM_TEAM,
        ),
        _e(
            "silly_kicks.tracking.direction.finalize_orientation",
            lambda s: direction.finalize_orientation(
                _numeric_team_frame(),
                home_team_id=s,
                home_team_start_left=True,
                home_team_start_left_extratime=None,
                source="id-scalar-registry",
            ),
            matched=NUM_TEAM,
        ),
    ]


# ---- causal -----------------------------------------------------------------------------


def _causal_entries() -> list[IdScalarEntry]:
    from silly_kicks.causal.opportunities import build_opportunities
    from tests.causal._fixtures import META, WIDE, frames, simple_actions

    def _build(scalar):
        # `home_team_id` decides the SIGN of `score_differential` (opportunities.py L265,
        # via `same_id`), so the built rows must carry a live score_differential for this to
        # be a real probe -- a treated cross inside a wide-area spell does that.
        return build_opportunities(
            frames({10.0: 5, 10.2: 5}, {10.0: WIDE, 10.2: WIDE}),
            simple_actions([("cross", 10.1)]),
            home_team_id=scalar,
            model_metadata=META,
        )

    return [_e("silly_kicks.causal.opportunities.build_opportunities", _build)]


# ---- tracking: per-frame + per-window primitives -----------------------------------------


def _tracking_primitive_entries() -> list[IdScalarEntry]:
    import silly_kicks.tracking as T

    passer, receiver = (50.0, 34.0), (80.0, 40.0)

    # ADR-051 D3 (4.80.0) removed the `compute_defensive_line` entry. It took `home_team_id` as
    # its ONLY id scalar and now takes `goal_map`, so discovery no longer returns it and a
    # retained entry fails `test_registry_has_no_stale_entries` -- the identical rule that
    # removed ADR-055's six. A draft of this change kept the entry with a `goal_map` lambda,
    # arguing the id axis survives via `GoalMap`'s canonical key. The argument is TRUE and the
    # entry is still wrong here: this registry is keyed on `inspect.signature` discovery of
    # id-SCALAR parameters, and a function with none cannot be a member however real the
    # underlying hazard is. That hazard is covered where it now lives --
    # `tests/tracking/test_gk_resolve_goal_map.py` (canonical-key + any-dtype lookup) and the
    # `compute_packing_metrics` entry below, which varies `attacking_team_id` against a
    # canonically-keyed map in one call.
    return [
        _e(
            "silly_kicks.tracking._defensive_line.select_back_line_players",
            # 3rd arg was `home_team_id`; ADR-055 re-keyed it to `defends_x0: bool`. The old
            # lambda passed `s` twice, so `same_id(team_id, home_team_id)` was True for every
            # variant -- `True` here is that same value, and the axis the gate actually exercises
            # (`ids_match(frames["team_id"], team_id)`) is unchanged.
            lambda s: T.select_back_line_players(tracking_frames(), s, True),
        ),
        _e("silly_kicks.tracking._team_shape.compute_team_shape", lambda s: T.compute_team_shape(tracking_frames(), s)),
        _e(
            "silly_kicks.tracking._packing.compute_packing_metrics",
            # ADR-051 D3: `home_team_id` -> `goal_map`. `attacking_team_id` is still the id scalar
            # under test; the map carries the second half of the axis (canonical-key resolution).
            lambda s: T.compute_packing_metrics(
                single_frame(),
                attacking_team_id=s,
                goal_map=goal_map_like_home_team_id(single_frame(), s),
                passer_xy=passer,
                receiver_xy=receiver,
            ),
        ),
        _e(
            "silly_kicks.tracking._structural_pass.compute_structural_pass_metrics",
            # ADR-051 D3: `home_team_id` -> `attacks_rtl: bool`. The direction is now a plain bool
            # and carries NO id, so the axis this entry exercises is `attacking_team_id` alone --
            # which is the one that matters here (`ids_match` against the frame team column).
            lambda s: T.compute_structural_pass_metrics(
                single_frame(), attacking_team_id=s, attacks_rtl=False, passer_xy=passer, receiver_xy=receiver
            ),
        ),
        _e(
            "silly_kicks.tracking._player_influence.compute_player_influence",
            # ADR-051 D3: `home_team_id` -> `attacks_rtl: bool`; see the note above.
            lambda s: T.compute_player_influence(single_frame(), xt_model(), attacking_team_id=s, attacks_rtl=False),
        ),
        _e(
            "silly_kicks.tracking._space_creation.compute_space_created",
            lambda s: T.compute_space_created(single_frame(), s),
        ),
        _e(
            "silly_kicks.tracking._obso.compute_pass_obso",
            lambda s: T.compute_pass_obso([single_frame()], 0, (80.0, 40.0), s),
        ),
        _e(
            "silly_kicks.tracking._cover_shadows.compute_blocking_score",
            lambda s: T.compute_blocking_score(
                single_frame(), s, xt_model(), goal_map=goal_map_like_home_team_id(single_frame(), s)
            ),
        ),
        _e(
            "silly_kicks.tracking._cover_shadows.compute_threat_pc",
            lambda s: T.compute_threat_pc(
                single_frame(),
                attacking_team_id=s,
                xt=xt_model(),
                goal_map=goal_map_like_home_team_id(single_frame(), s),
            ),
        ),
        _e(
            # TF-60 PR2: newly public (exported for restdefense._danger). TWO id scalars
            # (attacking_team_id + gk_player_id), so `s` is a tuple: attacking = team 5, defending
            # keeper = team 6's GK (player 2). gk_player_id resolves via `ids_match` internally.
            "silly_kicks.tracking._gk_influence.compute_gk_influence",
            lambda s: T.compute_gk_influence(
                single_frame(),
                attacking_team_id=s[0],
                gk_player_id=s[1],
                xt=xt_model(),
                goal_map=goal_map_like_home_team_id(single_frame(), s[0]),
            ),
            matched=(HOME, 2),
            mismatched=(as_str(HOME), as_str(2)),
        ),
        _e(
            "silly_kicks.tracking._cover_shadows.lane_control",
            lambda s: T.lane_control(
                single_frame(),
                passer,
                receiver,
                goal_map=goal_map_like_home_team_id(single_frame(), s),
                attacking_team_id=s,
            ),
        ),
        _e(
            "silly_kicks.tracking.pitch_control._dispatch.compute_pitch_control",
            lambda s: T.compute_pitch_control(single_frame(), s),
        ),
        _e(
            "silly_kicks.tracking.pitch_control._dispatch.compute_pitch_control_at_points",
            lambda s: T.compute_pitch_control_at_points(single_frame(), np.array([[50.0, 34.0], [80.0, 40.0]]), s),
        ),
    ]


# ---- tracking: model feature extractors ---------------------------------------------------


def _tracking_model_entries() -> list[IdScalarEntry]:
    import silly_kicks.tracking as T

    return [
        _e(
            "silly_kicks.tracking._xshot_occurrence.extract_xshot_features",
            lambda s: T.extract_xshot_features(single_frame(), gk_team_id=s, goal_x=105.0),
            matched=AWAY,
        ),
        _e(
            "silly_kicks.tracking._xcross_attempt.extract_xcross_features",
            lambda s: T.extract_xcross_features(single_frame(), gk_team_id=s[0], goal_x=105.0, carrier_player_id=s[1]),
            matched=(AWAY, 10),
            mismatched=(as_str(AWAY), as_str(10)),
        ),
        _e(
            "silly_kicks.tracking._xshot_occurrence.compute_xshot_occurrence",
            lambda s: T.compute_xshot_occurrence(frames_attacking_wide(), home_team_id=s),
            live_columns=("xshot_occurrence",),
        ),
        _e(
            "silly_kicks.tracking._xcross_attempt.compute_xcross_attempt",
            lambda s: T.compute_xcross_attempt(frames_attacking_wide(), home_team_id=s),
            live_columns=("xcross_attempt",),
        ),
        _e(
            "silly_kicks.tracking._xshot_occurrence.prepare_xshot_training_data",
            lambda s: T.prepare_xshot_training_data(frames_attacking_wide(), tracking_actions(), home_team_id=s),
        ),
        _e(
            "silly_kicks.tracking._xcross_attempt.prepare_xcross_training_data",
            lambda s: T.prepare_xcross_training_data(frames_attacking_wide(), tracking_actions(), home_team_id=s),
        ),
        _e(
            "silly_kicks.tracking._ghost_gk.prepare_ghost_gk_training_data",
            lambda s: T.prepare_ghost_gk_training_data(ghost_frames(), home_team_id=s),
        ),
        _e(
            "silly_kicks.tracking._ghost_gk.compute_ghost_gk",
            lambda s: T.compute_ghost_gk(ghost_frames(), home_team_id=s),
            live_columns=("ghost_gk_x", "ghost_gk_y"),
        ),
        _e(
            "silly_kicks.tracking._ghost_gk.serve_ghost_gk_positions",
            lambda s: T.serve_ghost_gk_positions(ghost_frames(), home_team_id=s),
        ),
        _e(
            "silly_kicks.tracking._xshot_occurrence.add_xshot_occurrence",
            lambda s: T.add_xshot_occurrence(tracking_actions(), frames_attacking_wide(), home_team_id=s),
            live_columns=("xshot_occurrence",),
        ),
        _e(
            "silly_kicks.tracking._xcross_attempt.add_xcross_attempt",
            lambda s: T.add_xcross_attempt(tracking_actions(), frames_attacking_wide(), home_team_id=s),
            live_columns=("xcross_attempt",),
        ),
    ]


# ---- tracking: orientation / frame utilities ----------------------------------------------


def _tracking_orientation_entries() -> list[IdScalarEntry]:
    import silly_kicks.tracking as T

    def _unlabeled_frames():
        fr = tracking_frames()
        fr = fr.drop(columns=["team_attacking_direction"])
        fr["team_attacking_direction"] = None
        return fr

    return [
        _e("silly_kicks.tracking.utils.play_left_to_right", lambda s: T.play_left_to_right(tracking_frames(), s)),
        _e(
            "silly_kicks.tracking.utils.orient_frames_to_ltr",
            lambda s: T.orient_frames_to_ltr(_unlabeled_frames(), home_team_id=s, home_team_start_left=True),
        ),
        _e(
            "silly_kicks.tracking.direction.orient_frames_to_ltr_by_geometry",
            lambda s: T.orient_frames_to_ltr_by_geometry(tracking_frames(), home_team_id=s),
        ),
    ]


# ---- tracking: per-Series features + xfns factories ----------------------------------------

#: ADR-055 REMOVED six former members of these tuples -- ``gk_closing_time_{min,mean}_s``,
#: ``gk_pitch_control_share_weighted``, ``gk_reachable_area_m2``, ``cover_shadow_xfns`` and
#: ``gk_influence_xfns``. They are not exemptions: each took ``home_team_id`` as its ONLY id
#: scalar, and after the re-key onto ``goal_map`` they declare no ``*_id`` parameter at all, so
#: this gate's own ``inspect.signature`` discovery no longer returns them and a retained entry
#: fails ``test_registry_has_no_stale_entries``.
#:
#: The dtype hazard did not disappear, it MOVED: it now lives in ``GoalMap._key``, which
#: canonicalizes every lookup, and is exercised by
#: ``tests/tracking/test_gk_resolve_goal_map.py`` (canonical-key + any-dtype-lookup tests) and
#: by the ``compute_blocking_score`` / ``compute_threat_pc`` / ``lane_control`` entries above,
#: which still vary ``attacking_team_id`` against a canonically-keyed map.
#: ``features.py`` per-Series functions taking ``(actions, frames, *, home_team_id, ...)``.
#:
#: ADR-051 D3 (4.80.0) removed the SIX defensive-line members -- ``defensive_line_x``,
#: ``back_line_high_x``, ``compactness_x``, ``lateral_width``, ``max_lateral_gap``,
#: ``back_n_count`` -- by the same rule as the ADR-055 removals recorded above: they now take
#: ``goal_map`` and declare no ``*_id`` parameter, so discovery no longer returns them.
#:
#: NOW EMPTY. The obso three followed in the same release, by a different route: their
#: ``home_team_id`` was never read, only FORWARDED, and the chain terminated in
#: ``_precompute_obso_lookup``, which ignored it. Removing the sink made the whole chain dead and
#: the cleanup cascaded to the public surface. Kept rather than deleted, per ``_SERIES_XT``.
_SERIES_PLAIN: tuple[str, ...] = ()

#: ...and those additionally taking a positional ``xt``.
#:
#: EMPTY since ADR-051 D3 (4.80.0), and deliberately kept rather than deleted: all five members
#: (``actor_reachable_area_m2``, ``off_ball_xt_team``, ``off_ball_xt_opponent``,
#: ``reachable_area_team``, ``reachable_area_opponent``) were player-influence per-Series
#: helpers whose only id scalar was ``home_team_id``; the re-key routes direction through
#: ``acting_team_attacks_rtl`` instead, so they take no id scalar at all. The tuple survives as
#: the seam a future ``(actions, frames, xt, *, some_id)`` helper registers into -- deleting it
#: would make the next such helper arrive with nowhere obvious to go.
_SERIES_XT: tuple[str, ...] = ()

#: ``*_xfns`` factories by construction shape.
#:
#: ADR-051 D3 (4.80.0) removed the five D3 factories -- ``defensive_line_xfns``,
#: ``line_breaking_ward_xfns``, ``off_ball_context_xfns``, ``packing_xfns``,
#: ``structural_pass_xfns`` (plain) and ``player_influence_xfns`` (xt) -- for the reason
#: recorded on ``_SERIES_PLAIN``: their only id scalar was ``home_team_id``.
#: 4.80.0 additionally removed ``obso_xfns``, ``pausa_xfns``, ``shape_graph_xfns`` and
#: ``team_shape_xfns`` (plain) and ``off_ball_run_value_xfns`` (xt) as dead-parameter cascade.
#: ``space_creation_xfns`` REMAINS and is the last member: its chain ends at
#: ``_compute_space_creation_for_action``, whose unread ``home_team_id`` is a DELIBERATE
#: retention recorded in CLAUDE.md ("D3 retires it by disuse, not removal").
_XFNS_PLAIN = ("space_creation_xfns",)
_XFNS_XT: tuple[str, ...] = ()


def gk_distribution_case():
    """``(actions, frames)`` for a live GK-distribution, reused from the xT-GK suite.

    xT-GK scores ONLY goal-kicks / GK-actor passes / throw-ins, so the generic
    outfield-pass fixture yields an all-NaN column. ``tests/tracking/test_xt_gk.py`` already
    owns a vetted two-distribution fixture (a goal-kick + a back-pass by the keeper) whose
    frames satisfy the DAS validator; single-sourcing it beats maintaining a second copy that
    could drift out of the scoring domain silently. Home team is 1 there, not 5.
    """
    from tests.tracking.test_xt_gk import _frames_for, _gk_actions

    actions = _gk_actions()
    return actions, _frames_for(actions)


GK_CASE_HOME = 1  # the xT-GK fixture's home team id


def _tracking_feature_entries() -> list[IdScalarEntry]:
    import silly_kicks.atomic.tracking.features as AF
    import silly_kicks.tracking.features as F

    entries: list[IdScalarEntry] = []

    for name in _SERIES_PLAIN:
        fn = getattr(F, name)
        entries.append(
            _e(
                f"silly_kicks.tracking.features.{name}",
                (lambda f: lambda s: f(tracking_actions(), tracking_frames(), home_team_id=s))(fn),
            )
        )
    for name in _SERIES_XT:
        fn = getattr(F, name)
        entries.append(
            _e(
                f"silly_kicks.tracking.features.{name}",
                (lambda f: lambda s: f(tracking_actions(), tracking_frames(), xt_model(), home_team_id=s))(fn),
            )
        )
    for name in _XFNS_PLAIN:
        fac = getattr(F, name)
        entries.append(
            _e(
                f"silly_kicks.tracking.features.{name}",
                (lambda f: lambda s: _xfn_outputs(f(home_team_id=s)))(fac),
            )
        )
    for name in _XFNS_XT:
        fac = getattr(F, name)
        entries.append(
            _e(
                f"silly_kicks.tracking.features.{name}",
                (lambda f: lambda s: _xfn_outputs(f(xt_model(), home_team_id=s)))(fac),
            )
        )
    # The atomic mirrors define their OWN xfns factories (distinct qualnames), so they are
    # separate boundaries and get their own entries. ADR-051 D3 (4.80.0) emptied this set:
    # `packing_xfns` / `structural_pass_xfns` went with their tracking originals, then
    # `off_ball_run_value_xfns` followed when the dead-parameter cascade reached it. The loop
    # stays so the next atomic factory carrying an id scalar has an obvious home.
    for name in ():
        fac = getattr(AF, name)
        takes_xt = name == "off_ball_run_value_xfns"
        entries.append(
            _e(
                f"silly_kicks.atomic.tracking.features.{name}",
                (
                    lambda f, xt_first: (
                        lambda s: _xfn_outputs(
                            f(xt_model(), home_team_id=s) if xt_first else f(home_team_id=s),
                            states=_atomic_gamestates(tracking_actions()),
                        )
                    )
                )(fac, takes_xt),
            )
        )
    # The xT-GK pair (tracking + atomic `xt_gk_xfns`) lived here with its own GK-distribution
    # fixture. ADR-051 D3 (4.80.0) removed their `home_team_id`, which was dead and retained only
    # "for GK-feature-family signature parity" -- a rationale ADR-055 had already invalidated by
    # re-keying two of that family. They declare no id scalar now, so discovery drops them.
    entries.append(
        _e(
            "silly_kicks.tracking._xshot_occurrence.xshot_occurrence_xfns",
            lambda s: _xfn_outputs(F.xshot_occurrence_xfns(home_team_id=s), frames=frames_attacking_wide()),
        )
    )
    entries.append(
        _e(
            "silly_kicks.tracking._xcross_attempt.xcross_attempt_xfns",
            lambda s: _xfn_outputs(F.xcross_attempt_xfns(home_team_id=s), frames=frames_attacking_wide()),
        )
    )
    entries.append(
        _e(
            "silly_kicks.tracking.features.ghost_gk_xfns",
            lambda s: _xfn_outputs(F.ghost_gk_xfns(home_team_id=s), frames=ghost_frames()),
        )
    )
    return entries


def _gk_gamestates():
    from silly_kicks.vaep.feature_framework import gamestates

    return gamestates(gk_distribution_case()[0], nb_prev_actions=2)


def _atomic_gamestates(actions: pd.DataFrame):
    from silly_kicks.atomic.spadl import convert_to_atomic
    from silly_kicks.atomic.vaep.features import gamestates as atomic_gamestates

    acts = actions.copy()
    if "original_event_id" not in acts.columns:
        acts["original_event_id"] = [f"e{i}" for i in range(len(acts))]
    if "result_id" not in acts.columns:
        acts["result_id"] = 1
    if "bodypart_id" not in acts.columns:
        acts["bodypart_id"] = 0
    return atomic_gamestates(convert_to_atomic(acts), nb_prev_actions=2)


# --------------------------------------------------------------------------------------
# The registry
# --------------------------------------------------------------------------------------


def _build_registry() -> list[IdScalarEntry]:
    return [
        *_spadl_entries(),
        *_converter_entries(),
        *_causal_entries(),
        *_tracking_primitive_entries(),
        *_tracking_model_entries(),
        *_tracking_orientation_entries(),
        *_tracking_feature_entries(),
    ]


PUBLIC_ID_SCALAR_ENTRIES: list[IdScalarEntry] = _build_registry()


#: Delegated to `tests/tracking/test_id_dtype_invariance.py`, whose (False, False, True)
#: permutation sweeps a STRING `home_team_id` against numeric actions/frames -- byte-for-byte
#: this gate's axis, on the same shared fixture. MACHINE-CHECKED against that gate's own
#: registered surface (`test_delegated_entries_are_really_covered`), so this is a verified
#: pointer, not a prose promise.
COVERED_BY_AGGREGATOR_GATE: dict[str, str] = {
    f"silly_kicks.tracking.features.{n}": "tracking add_* aggregator; swept by test_id_dtype_invariance.py"
    for n in (
        # ADR-055: `add_cover_shadows` and `add_gk_influence` are GONE from this list, not
        # exempted. They took `home_team_id` as their only id scalar; after the re-key onto
        # `goal_map` they declare no `*_id` parameter, so discovery no longer returns them and
        # a retained pointer fails `test_registry_has_no_stale_entries`. They are still swept by
        # test_id_dtype_invariance.py for their ACTIONS/FRAMES id dtypes -- what is gone is the
        # `home_team_id` SCALAR axis, because the scalar is gone.
        #
        # ADR-051 D3 (4.80.0) removed SIX more by the identical rule -- `add_defensive_line`,
        # `add_line_break`, `add_off_ball_context`, `add_packing`, `add_player_influence`,
        # `add_structural_pass` -- plus the atomic `add_packing` / `add_structural_pass` mirrors
        # below. Same reading as ADR-055's: the ACTIONS/FRAMES axes are still swept next door;
        # only the scalar axis is gone, because the scalar is.
        #
        # Where the hazard MOVED, per mechanism, so this list is not read as coverage lost:
        # the two map sites resolve direction through `GoalMap._key`, which canonicalizes every
        # lookup (`tests/tracking/test_gk_resolve_goal_map.py`); the four bool sites resolve it
        # through `acting_team_attacks_rtl`, whose `(game_id, period_id, team_id)` merge is
        # dtype-exercised by the aggregator gate's frames/actions axes -- a mismatched spelling
        # there resolves NOTHING and now surfaces as <NA> rather than a silent False (4.80.0).
        #
        # 4.80.0 removed SIX more by a different route than the D3 re-key: `add_obso`,
        # `add_off_ball_run_values`, `add_pausa`, `add_shape_graph`, `add_team_shape` and
        # `add_xt_gk` never READ the parameter -- they forwarded it into a sink that ignored it,
        # so cleaning the sink cascaded to the public surface. `add_off_ball_runs` and
        # `add_space_creation` REMAIN: the first feeds `_off_ball_runs_kernel`, whose unread copy
        # is preserved deliberately (its Gate B green IS the measurement that it is unread), and
        # the second ends at `_compute_space_creation_for_action`, the CLAUDE.md-recorded
        # "retire by disuse, not removal" case.
        "add_ghost_gk",
        "add_off_ball_runs",
        "add_space_creation",
    )
} | {
    f"silly_kicks.atomic.tracking.features.{n}": (
        "atomic add_* mirror; delegates to the tracking aggregator swept by test_id_dtype_invariance.py"
    )
    for n in ()
}


#: Genuinely cannot be dtype-invariant. Each entry states WHY, and each is a case where
#: invariance would mean the function is BROKEN -- not a case we chose not to test.
NOT_INVARIANT: dict[str, str] = {
    "silly_kicks.tracking.utils.validate_id_dtypes": (
        "The DIAGNOSTIC itself (ADR-019's loud guard). Its entire job is to REPORT that an id "
        "scalar's dtype disagrees with the columns, so identical output across a matched and a "
        "mismatched scalar would mean it had stopped detecting the defect this whole gate exists "
        "for. Its own behaviour is pinned by tests/test_id_compat.py (moved up from "
        "tests/tracking/ in 4.53.0 alongside the module's promotion to silly_kicks.id_compat)."
    ),
    "silly_kicks.tracking.gradientsports.add_gradientsports_player_ids": (
        "WRITER, not comparator: home_team_id/away_team_id are ASSIGNED into the output team_id "
        "column via .mask(...), never compared against one (silly_kicks/tracking/gradientsports.py "
        "L306-307). A value-equal scalar of a different dtype therefore produces a legitimately "
        "different-dtype output column -- that is the function doing its job, and forcing "
        "invariance would forbid a caller from choosing their own id representation."
    ),
    "silly_kicks.tracking.pitch_control._surface.PitchControlSurface": (
        "SCOPE: the CONSTRUCTOR only -- which is what discovery keys on, since "
        "`inspect.signature(PitchControlSurface)` sees `__init__`. `attacking_team_id` is a "
        "recorded provenance field (surfaced in `.to_xarray()` attrs), never resolved against an "
        'id column: constructing with 5 vs "5" stores 5 vs "5", a container faithfully recording '
        "its input rather than a comparison. "
        "The two METHODS `.player_surface(player_id)` / `.player_share(player_id)` DO compare a "
        "caller-supplied id scalar against the `player_ids` array, but they now route it through "
        "`ids_match` (ADR-019) -- dtype-invariant, byte-identical on matched dtypes, and exercised "
        "by `tests/tracking/pitch_control/test_surface.py::TestPlayerIdDtypeInvariance`. They are "
        "not auto-discovered here because they are methods, not the constructor "
        "`inspect.signature(PitchControlSurface)` sees. The `player_team_ids == team_id` compare "
        "in `player_share` stays a raw `==` on purpose: `team_id` is drawn from `player_team_ids` "
        "itself (a same-source compare that cannot mismatch by construction, ADR-043 decision 6)."
    ),
    "silly_kicks.keeper_identity.KeeperIdentity": (
        "VALUE OBJECT, like PitchControlSurface: discovery keys on the NamedTuple constructor "
        "`inspect.signature(KeeperIdentity)` sees, whose `gk_id` field ends in `_id`. But `gk_id` "
        'is STORED, never resolved against an id column -- constructing with 920 vs "920" stores '
        '920 vs "920", a container faithfully recording its input rather than a comparison. '
        "Forcing invariance would forbid the resolver from preserving its provider's own keeper-id "
        "dtype (the roster/native paths deliberately keep the RAW id as `gk_id`). The dtype-safe "
        "comparisons that BUILD the value live in `resolve_keeper_identities` (canonical keys + "
        "id_compat, ADR-055), which takes no id scalar and so is not discovered here."
    ),
    "silly_kicks.keeper_identity.KeeperSegment": (
        "VALUE OBJECT, like KeeperIdentity and PitchControlSurface: discovery keys on the "
        "NamedTuple constructor `inspect.signature(KeeperSegment)` sees, whose `team_id` and "
        "`player_id` fields end in `_id`. Both are STORED, never resolved against an id column -- a "
        "KeeperSegment is a plain data container for one keeper's on-pitch tenure, so constructing "
        'with 10 vs "10" stores 10 vs "10", faithfully recording its input rather than comparing. '
        "Forcing invariance would forbid an extractor from preserving its provider's own id dtype "
        "(DFL/SkillCorner ids are strings). The only comparisons that CONSUME these segments live "
        "in `build_keeper_appearances_from_segments`, which compares period bounds only (no id)."
    ),
    "silly_kicks.keeper_identity.build_keeper_appearances_from_segments": (
        "WRITER, not comparator: its `game_id` scalar (plus each segment's `team_id`/`player_id`) "
        "is written VERBATIM into the emitted port rows, never resolved against an id column. Its "
        "ONLY comparisons are period bounds (`start_period <= period <= end_period`) and `start >= "
        "end` time slices -- no id is ever compared against an id, so a value-equal scalar of a "
        "different dtype yields a legitimately different-dtype output column, the function "
        "faithfully recording its caller's chosen id representation. The same rule that exempts "
        "`add_gradientsports_player_ids` and the kloppy `convert_to_actions` writer below."
    ),
}


#: A FOURTH bucket, deliberately distinct from ``NOT_INVARIANT``. These functions are believed
#: invariant, but the gate's axis -- one value in two dtypes -- is INEXPRESSIBLE against them,
#: so no honest entry can be written. Calling that "not invariant" would misreport the reason
#: and calling it "covered" would be a lie; naming the obstruction is the only truthful option.
#: Each entry states precisely WHAT blocks the pair, so a future change that removes the
#: obstruction is recognisable as one.
NOT_EXERCISABLE: dict[str, str] = {
    "silly_kicks.tracking._run_features.run_tracking_features": (
        "ORCHESTRATOR/forwarder, not a comparator. `home_team_id` is only forwarded (via the "
        "`_opt` signature-aware pass-through) to whichever sub-families still accept it; "
        "`run_tracking_features` never compares it against any column of its own. Every sub-family "
        "that CONSUMES `home_team_id` is independently covered by "
        "`tests/tracking/test_id_dtype_invariance.py` (the add_* gate), so an entry here would "
        "either be vacuous (the producer performs no id comparison itself) or a redundant re-run of "
        "that gate on a heavy full-family fixture. Its return is a `(DataFrame, "
        "TrackingFeaturesReport)` tuple, and the one-value-two-dtype axis is not expressible against "
        "a forwarding seam with no id comparison of its own -- the same rule that exempts the kloppy "
        "`convert_to_actions` forwarder below."
    ),
    "silly_kicks.tracking.metrica.convert_to_frames": (
        "No value-equal dtype pair EXISTS for this builder. Metrica's bronze carries no team ids "
        "at all: `_explode_team` writes the string LITERALS 'Home'/'Away' into team_id "
        "(tracking/metrica.py L77), and the builder `str()`-casts home_team_id before every "
        "comparison. The caller's id space is therefore the fixed set {'Home','Away'}, and "
        "'Home' has no numeric twin -- a matched/mismatched pair would have to differ in VALUE, "
        "which is precisely the vacuity `test_matched_and_mismatched_scalars_differ_in_dtype_not"
        "_value` rejects. The str-cast is itself the (ad-hoc) canonicalization. Its second "
        "discovered param, `jersey_to_player_id`, is a discovery false-positive of the "
        "name-ending rule: it is a nested lookup TABLE whose values become player ids, not a "
        "caller-supplied id resolved against a column."
    ),
    "silly_kicks.spadl.kloppy.convert_to_actions": (
        "WRITER, not comparator, AND not cheaply callable. `game_id` is only assigned into the "
        "output game_id column (kloppy.py L205) and used as a sort key (L226); it is never "
        "compared against any column value, so a different-dtype scalar legitimately yields a "
        "different-dtype output column -- the same rule that exempts "
        "add_gradientsports_player_ids. Separately, invoking it at all requires the optional "
        "`kloppy` dependency plus a real EventDataset built from on-disk XML fixtures, so an "
        "entry here would silently skip on any CI leg without that extra."
    ),
}
