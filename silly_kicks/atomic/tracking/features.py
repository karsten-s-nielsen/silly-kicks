"""Tracking-aware action_context features for atomic SPADL.

Mirrors silly_kicks.tracking.features with atomic-shaped column reads.
Shares the schema-agnostic kernels in silly_kicks.tracking._kernels.

See NOTICE for full bibliographic citations and ADR-005 for the integration contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from silly_kicks.tracking.pitch_control import PitchControlCache

from silly_kicks._nan_safety import nan_safe_enrichment
from silly_kicks.atomic.spadl import config as atomicconfig
from silly_kicks.id_compat import ids_equal
from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking import _kernels
from silly_kicks.tracking._packing import PackingParams
from silly_kicks.tracking._shot_goalmouth import ShotGoalmouthParams
from silly_kicks.tracking._structural_pass import (
    StructuralPassParams,
    compute_structural_pass_metrics,
)
from silly_kicks.tracking._xcross_attempt import add_xcross_attempt, xcross_attempt_xfns
from silly_kicks.tracking._xshot_occurrence import add_xshot_occurrence, xshot_occurrence_xfns
from silly_kicks.tracking.feature_framework import lift_to_states
from silly_kicks.tracking.features import (
    actor_reachable_area_m2,
    add_ghost_gk,
    add_gk_influence,
    add_player_influence,
    ball_carrier_at_action,
    cover_shadow_xfns,
    elastic_sync_xfns,
    ghost_gk_xfns,
    gk_closing_time_mean_s,
    gk_closing_time_min_s,
    gk_influence_xfns,
    gk_pitch_control_share_weighted,
    gk_reachable_area_m2,
    obso_xfns,
    off_ball_xt_opponent,
    off_ball_xt_team,
    pausa_xfns,
    player_influence_xfns,
    reachable_area_opponent,
    reachable_area_team,
    shape_graph_xfns,
    space_creation_xfns,
)
from silly_kicks.tracking.pressure import Method, PressureParams
from silly_kicks.tracking.utils import _resolve_action_frame_context

_ATOMIC_SHOT_TYPE_IDS = frozenset(spadlconfig.actiontype_id[n] for n in ("shot", "shot_penalty"))

__all__ = [
    "PackingParams",
    "StructuralPassParams",
    "actor_arc_length_pre_window",
    "actor_displacement_pre_window",
    "actor_reachable_area_m2",
    "actor_speed",
    "add_action_context",
    "add_actor_pre_window",
    "add_cover_shadows",
    "add_ghost_gk",
    "add_gk_influence",
    "add_off_ball_run_values",
    "add_packing",
    "add_pitch_control",
    "add_player_influence",
    "add_pre_shot_gk_angle",
    "add_pre_shot_gk_position",
    "add_press_commitment",
    "add_pressure_on_actor",
    "add_shot_goalmouth",
    "add_structural_pass",
    "add_xcross_attempt",
    "add_xshot_occurrence",
    "add_xt_gk",
    "atomic_actor_pre_window_default_xfns",
    "atomic_pitch_control_default_xfns",
    "atomic_pitch_control_xfns",
    "atomic_pre_shot_gk_angle_default_xfns",
    "atomic_pre_shot_gk_default_xfns",
    "atomic_pre_shot_gk_full_default_xfns",
    "atomic_pressure_default_xfns",
    "atomic_tracking_default_xfns",
    "ball_carrier_at_action",
    "compute_structural_pass_metrics",
    "cover_shadow_xfns",
    "defenders_in_triangle_to_goal",
    "elastic_sync_xfns",
    "ghost_gk_xfns",
    "gk_closing_time_mean_s",
    "gk_closing_time_min_s",
    "gk_influence_xfns",
    "gk_pitch_control_share_weighted",
    "gk_reachable_area_m2",
    "nearest_defender_distance",
    "obso_xfns",
    "off_ball_run_value_xfns",
    "off_ball_xt_opponent",
    "off_ball_xt_team",
    "packing_xfns",
    "pausa_xfns",
    "pitch_control_at_target",
    "player_influence_xfns",
    "pre_shot_gk_angle_off_goal_line",
    "pre_shot_gk_angle_to_shot_trajectory",
    "pre_shot_gk_distance_to_goal",
    "pre_shot_gk_distance_to_shot",
    "pre_shot_gk_x",
    "pre_shot_gk_y",
    "pressure_on_actor",
    "reachable_area_opponent",
    "reachable_area_team",
    "receiver_zone_density",
    "shape_graph_xfns",
    "shot_crossing_y",
    "shot_crossing_z",
    "shot_on_target_derived",
    "shot_speed",
    "shot_time_to_goal_line",
    "space_creation_xfns",
    "structural_pass_xfns",
    "xcross_attempt_xfns",
    # Cycle B: was imported (:31), used (:877) and re-exported by its sibling
    # `xcross_attempt_xfns` above -- but missing here, an asymmetry with no stated reason.
    # Found by K2's replacement meta-assertion, which pairs `dir()` against the declared export
    # instead of comparing an expression to itself.
    "xshot_occurrence_xfns",
    "xt_gk_xfns",
]


def _structural_pass_atomic_endpoints(actions: pd.DataFrame) -> pd.DataFrame:
    """Synthesize start_x/start_y/end_x/end_y from atomic x,y,dx,dy. structural_pass
    needs the RECEIVER (end), so a passer-only x->start_x rename is insufficient."""
    adapted = actions.copy()
    adapted["start_x"] = adapted["x"]
    adapted["start_y"] = adapted["y"]
    adapted["end_x"] = adapted["x"] + adapted["dx"]
    adapted["end_y"] = adapted["y"] + adapted["dy"]
    return adapted


def add_structural_pass(actions, frames, *, links=None, params=None):
    """Atomic-SPADL aggregator for structural-pass primitives (TF-45). Synthesizes
    end_x/end_y from x+dx / y+dy (atomic has no end_*), delegates to the standard
    aggregator, then drops the synthesized columns.

    Examples
    --------
    Enrich atomic actions with the structural-pass primitive columns::

        from silly_kicks.atomic.tracking.features import add_structural_pass
        enriched = add_structural_pass(atomic_actions, frames)
        enriched[["structural_lbs", "structural_sgm", "structural_sdi"]].head()
    """
    from silly_kicks.tracking.features import add_structural_pass as _std

    adapted = _structural_pass_atomic_endpoints(actions)
    result = _std(adapted, frames, links=links, params=params)
    return result.drop(columns=["start_x", "start_y", "end_x", "end_y"])


def structural_pass_xfns(*, params=None):
    """Atomic VAEP factory: each gamestate slot has its end_x/end_y synthesized from
    x,y,dx,dy before the shared kernel runs.

    Examples
    --------
    >>> from silly_kicks.atomic.tracking.features import structural_pass_xfns
    >>> xfns = structural_pass_xfns()
    >>> len(xfns)
    1
    """
    from silly_kicks.tracking.features import structural_pass_xfns as _std_xfns

    inner = _std_xfns(params=params)[0]

    def _atomic_transformer(states, frames):
        adapted_states = [_structural_pass_atomic_endpoints(s) for s in states]
        return inner(adapted_states, frames)

    _atomic_transformer._frame_aware = True  # type: ignore[attr-defined]
    _atomic_transformer.__name__ = "structural_pass"
    return [_atomic_transformer]


def _packing_atomic_adapter(actions: pd.DataFrame, params: PackingParams) -> pd.DataFrame:
    """Synthesize start/end + std type ids + a type-aware result_id from the atomic
    stream (the SK-xT-2 precedent): dribble success is intrinsic (dribbles are never
    followed by ``receival``); every other packing-domain type succeeds iff the NEXT
    atom (same game+period) is a ``receival`` OR a SAME-TEAM keeper reception
    (``keeper_pick_up`` / ``keeper_claim`` -- atomic never inserts a receival before
    keeper collections, so a completed back-pass would otherwise synthesize fail;
    execution-review D5). Non-domain atoms map to std ``non_action`` (off-domain for
    the kernel; atomic-only ids like receival/out never leak into the std domain).

    Collapsed-atom bridging (execution-review D2): ``convert_to_atomic``'s
    ``_simplify`` re-types corner_crossed/corner_short -> atomic ``corner`` and
    freekick_crossed/freekick_short/shot_freekick -> atomic ``freekick``, so the
    STANDARD set-piece names carry ZERO atomic rows. When a corner/freekick name is
    in ``params.action_types``, the matching collapsed atom joins the domain (mapped
    to the first requested std id -- the kernel only tests membership, and the
    synthesized frame never leaves the mirror). A collapsed ``freekick`` that was a
    shot_freekick stays off-domain honestly: no receival/keeper-reception follows a
    shot, so its synthesized result is fail. Name-mapped deliberately -- do NOT
    "simplify" to raw id passthrough (a future config renumber would silently
    break it)."""
    adapted = _structural_pass_atomic_endpoints(actions)
    n = len(actions)
    type_id = actions["type_id"].to_numpy()

    std_ids = np.full(n, spadlconfig.actiontype_id["non_action"], dtype="int64")
    is_domain = np.zeros(n, dtype=bool)
    for name in params.action_types:
        mask = type_id == atomicconfig.actiontype_id[name]  # NaN-safe (NaN != int)
        std_ids[mask] = spadlconfig.actiontype_id[name]
        is_domain |= mask
    for collapsed, members in (
        ("corner", ("corner_crossed", "corner_short")),
        ("freekick", ("freekick_crossed", "freekick_short")),
    ):
        requested = [name for name in members if name in params.action_types]
        if requested:
            mask = type_id == atomicconfig.actiontype_id[collapsed]
            std_ids[mask] = spadlconfig.actiontype_id[requested[0]]
            is_domain |= mask

    next_type = np.full(n, -1.0)
    same_gp = np.zeros(n, dtype=bool)
    if n > 1:
        next_type[:-1] = type_id[1:]
        game = actions["game_id"].to_numpy()
        period = actions["period_id"].to_numpy()
        same_gp[:-1] = (game[1:] == game[:-1]) & (period[1:] == period[:-1])
    is_dribble = type_id == atomicconfig.actiontype_id["dribble"]
    team_s = actions["team_id"].reset_index(drop=True)
    next_team_same = ids_equal(team_s, team_s.shift(-1)).to_numpy()
    keeper_reception_ids = [atomicconfig.actiontype_id["keeper_pick_up"], atomicconfig.actiontype_id["keeper_claim"]]
    is_received = same_gp & (
        (next_type == atomicconfig.actiontype_id["receival"])
        | (np.isin(next_type, keeper_reception_ids) & next_team_same)
    )
    success = is_domain & (is_dribble | (~is_dribble & is_received))

    adapted["type_id"] = std_ids
    adapted["result_id"] = np.where(success, spadlconfig.result_id["success"], spadlconfig.result_id["fail"])
    return adapted


def add_packing(actions, frames, *, goal_map=None, links=None, params=None):
    """Atomic-SPADL aggregator for TF-49 packing: the THREE numeric columns only
    (packing_made / packing_net / packing_goal_threat). Synthesizes end from x+dx /
    y+dy plus a type-aware result_id (:func:`_packing_atomic_adapter`), delegates to
    the standard aggregator, then assembles the output on a COPY OF THE CALLER'S
    frame -- the adapter's rewritten type_id and synthetic result_id never leak into
    the returned enrichment (execution-review D3). Receiver/secured are omitted --
    atomic ``receival`` atoms already carry receiver identity explicitly (spec s6).
    ``params.require_secured=True`` raises ``ValueError``: secured reception is
    defined on standard streams and its column is not emitted here, so gating counts
    on a dropped, atom-stream secured label would be a silent semantic trap
    (ADR-039).

    Examples
    --------
    Enrich atomic actions with the packing columns::

        from silly_kicks.atomic.tracking.features import add_packing
        enriched = add_packing(atomic_actions, frames)
        enriched[["packing_made", "packing_net", "packing_goal_threat"]].head()
    """
    from silly_kicks.tracking.features import add_packing as _std

    if params is None:
        params = PackingParams()
    if params.require_secured:
        raise ValueError(
            "atomic add_packing: params.require_secured=True is not supported -- the atomic "
            "mirror emits numeric columns only (packing_secured is dropped), so gating counts "
            "on a dropped, atom-stream secured label would be a silent semantic trap. See ADR-039."
        )
    adapted = _packing_atomic_adapter(actions, params)
    enriched = _std(adapted, frames, goal_map=goal_map, links=links, params=params)
    # Assemble on the CALLER's frame: the delegate ran on the adapter's synthesized
    # stream, whose type_id/result_id are mirror-internal. Row order is preserved by
    # the delegate (copy + how='left' provenance merge), so positional .array
    # assignment keeps the Int64/float dtypes intact.
    out = actions.copy()
    for col in ("packing_made", "packing_net", "packing_goal_threat"):
        out[col] = enriched[col].array
    provenance_cols = ["frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]
    if not any(c in out.columns for c in provenance_cols):
        for col in provenance_cols:
            if col in enriched.columns:
                out[col] = enriched[col].array
    return out


def packing_xfns(*, goal_map=None, params=None):
    """Atomic VAEP factory for packing: each gamestate slot runs through
    :func:`_packing_atomic_adapter` (endpoint + type-aware result synthesis) before
    the shared kernel. Numeric columns only; inherits the standard factory's
    require_secured rejection and its result-leakage warning (TF-49 review F4).

    Examples
    --------
    >>> from silly_kicks.atomic.tracking.features import packing_xfns
    >>> xfns = packing_xfns()
    >>> len(xfns)
    1
    """
    from silly_kicks.tracking.features import packing_xfns as _std_xfns

    if params is None:
        params = PackingParams()
    inner = _std_xfns(goal_map=goal_map, params=params)[0]

    def _atomic_transformer(states, frames):
        adapted_states = [_packing_atomic_adapter(s, params) for s in states]
        return inner(adapted_states, frames)

    _atomic_transformer._frame_aware = True  # type: ignore[attr-defined]
    _atomic_transformer.__name__ = "packing"
    return [_atomic_transformer]


#: Atomic reception atoms -- the rows that carry receiver identity. Kept as an explicit
#: name-mapped tuple (not raw ids) so a future config renumber cannot silently break it.
_RECEPTION_ATOM_NAMES = ("receival", "keeper_pick_up", "keeper_claim")


def _restore_reception_touches(actions: pd.DataFrame, adapted: pd.DataFrame) -> pd.DataFrame:
    """Re-type reception atoms from ``non_action`` to ``bad_touch`` on an adapted stream.

    ``_packing_atomic_adapter`` collapses every non-domain atom to standard
    ``non_action``, which ``resolve_next_touch_receiver`` skips because a non-action is
    not a ball touch. That is correct for packing (which drops receiver semantics on the
    atomic side entirely) and WRONG for any consumer that needs the receiver: it makes the
    receival atom invisible, so no receiver ever resolves.

    ``bad_touch`` is chosen because it is a real touch (visible to the resolver) that is
    off-domain for every receiver-consuming feature so far, so a reception can never be
    mistaken for a valued action in its own right.
    """
    out = adapted.copy()
    reception_ids = [atomicconfig.actiontype_id[name] for name in _RECEPTION_ATOM_NAMES]
    is_reception = np.isin(actions["type_id"].to_numpy(), reception_ids)
    if is_reception.any():
        out.loc[is_reception, "type_id"] = spadlconfig.actiontype_id["bad_touch"]
    return out


def add_off_ball_run_values(actions, frames, xt, *, links=None, pitch_control_cache=None, params=None):
    """Atomic-SPADL mirror of TF-35 run valuation: the five wide columns (ADR-042).

    Reuses the packing mirror's adapter (:func:`_packing_atomic_adapter`) to synthesize
    ``end`` from ``x+dx``/``y+dy`` and a type-aware ``result_id`` -- pass/cross succeed
    iff the NEXT atom is a ``receival`` or a same-team keeper reception -- because TF-35's
    domain is exactly "completed pass/cross", the same completion question packing asks.
    The adapter's rewritten ids never leak: the output is assembled on a COPY OF THE
    CALLER'S frame (execution-review D3 precedent).

    Receiver resolution runs on the SYNTHESIZED standard stream, and that needs one
    correction the packing mirror does not: the packing adapter maps EVERY non-domain atom
    to standard ``non_action``, which :func:`resolve_next_touch_receiver` deliberately
    SKIPS (a non-action is not a touch). Under that mapping the ``receival`` atom -- the
    very row that identifies the receiver -- becomes invisible, no receiver ever resolves,
    and all five columns come back <NA> for every action. Reception atoms are therefore
    re-typed to standard ``bad_touch``: a genuine touch (so the resolver sees it) that is
    off-domain for TF-35 (so it never becomes a valued action itself).

    Examples
    --------
    Enrich atomic actions with the off-ball run-value columns::

        from silly_kicks.atomic.tracking.features import add_off_ball_run_values
        enriched = add_off_ball_run_values(atomic_actions, frames, fitted_xt)
        enriched[["run_value_target", "n_disruptive_runs"]].head()
    """
    from silly_kicks.tracking.features import _RUN_VALUE_COLS
    from silly_kicks.tracking.features import add_off_ball_run_values as _std

    adapted = _packing_atomic_adapter(actions, PackingParams(action_types=("pass", "cross")))
    adapted = _restore_reception_touches(actions, adapted)
    enriched = _std(
        adapted,
        frames,
        xt,
        links=links,
        pitch_control_cache=pitch_control_cache,
        params=params,
    )
    out = actions.copy()
    for col in _RUN_VALUE_COLS:
        out[col] = enriched[col].array
    provenance_cols = ["frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]
    if not any(c in out.columns for c in provenance_cols):
        for col in provenance_cols:
            if col in enriched.columns:
                out[col] = enriched[col].array
    return out


def off_ball_run_value_xfns(xt, *, params=None, pitch_control_cache=None):
    """Atomic VAEP factory for TF-35: each gamestate slot runs through
    :func:`_packing_atomic_adapter` before the shared kernel. Inherits the standard
    factory's fitted-xt check and its opt-in / result-leakage contract (ADR-042) --
    it is in NO default xfn list.

    ``pitch_control_cache`` (TF-7) is forwarded to the standard factory so a caller can
    share one canonical-surface cache across feature families. ``None`` (default) is
    byte-identical to threading nothing.

    Examples
    --------
    Build the atomic off-ball run-value VAEP transformers::

        from silly_kicks.atomic.tracking.features import off_ball_run_value_xfns
        xfns = off_ball_run_value_xfns(fitted_xt)
        len(xfns)
        1
    """
    from silly_kicks.tracking.features import off_ball_run_value_xfns as _std_xfns

    inner = _std_xfns(xt, params=params, pitch_control_cache=pitch_control_cache)[0]
    adapter_params = PackingParams(action_types=("pass", "cross"))

    def _atomic_transformer(states, frames):
        # _restore_reception_touches is NOT optional here: without it the adapter's
        # non_action mapping hides every receival atom, no receiver resolves, and the
        # factory returns NaN for every slot -- the same silent-death the aggregator was
        # fixed for. The leakage guard only checks __name__, and the liveness check calls
        # the frames=None branch, so no gate covers this path (ADR-042 review finding 1).
        adapted = [_restore_reception_touches(s, _packing_atomic_adapter(s, adapter_params)) for s in states]
        return inner(adapted, frames)

    _atomic_transformer._frame_aware = True  # type: ignore[attr-defined]
    _atomic_transformer.__name__ = "off_ball_run_values"
    return [_atomic_transformer]


def add_xt_gk(actions, frames, xt, *, links=None, params=None):
    """Atomic-SPADL mirror of tracking.add_xt_gk (Eyestone). Synthesizes start/end from
    x,y,dx,dy (atomic has no end_*), delegates to the standard aggregator, then drops the
    synthesized columns. See silly_kicks.tracking.add_xt_gk + NOTICE.

    Examples
    --------
    Enrich atomic actions with the xT-GK columns::

        from silly_kicks.atomic.tracking.features import add_xt_gk
        enriched = add_xt_gk(atomic_actions, frames, fitted_xt)
        enriched[["xt_gk_base", "xt_gk_rav", "xt_gk"]].head()
    """
    from silly_kicks.tracking.features import add_xt_gk as _std

    adapted = _structural_pass_atomic_endpoints(actions)
    result = _std(adapted, frames, xt, links=links, params=params)
    return result.drop(columns=["start_x", "start_y", "end_x", "end_y"])


def xt_gk_xfns(xt, *, params=None):
    """Atomic VAEP factory for xT-GK: each gamestate slot has its start/end synthesized
    from x,y,dx,dy before the shared kernel runs.

    Examples
    --------
    Build the atomic xT-GK VAEP transformers::

        from silly_kicks.atomic.tracking.features import xt_gk_xfns
        xfns = xt_gk_xfns(fitted_xt)
        len(xfns)
        1
    """
    from silly_kicks.tracking.features import xt_gk_xfns as _std_xfns

    inner = _std_xfns(xt, params=params)[0]

    def _atomic_transformer(states, frames):
        adapted_states = [_structural_pass_atomic_endpoints(s) for s in states]
        return inner(adapted_states, frames)

    _atomic_transformer._frame_aware = True  # type: ignore[attr-defined]
    _atomic_transformer.__name__ = "xt_gk"
    return [_atomic_transformer]


def nearest_defender_distance(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Atomic-SPADL: distance to nearest defender at action anchor (x, y).

    See NOTICE; matches silly_kicks.tracking.features.nearest_defender_distance.

    Examples
    --------
    Compute defender distance for an atomic action stream::

        from silly_kicks.atomic.tracking.features import nearest_defender_distance
        d = nearest_defender_distance(atomic_actions, frames)
    """
    ctx = _resolve_action_frame_context(actions, frames)
    return _kernels._nearest_defender_distance(actions["x"], actions["y"], ctx)


def actor_speed(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Atomic-SPADL: actor's speed at the linked frame.

    See NOTICE.

    Examples
    --------
    ::

        from silly_kicks.atomic.tracking.features import actor_speed
        s = actor_speed(atomic_actions, frames)
    """
    ctx = _resolve_action_frame_context(actions, frames)
    return _kernels._actor_speed_from_ctx(ctx)


def receiver_zone_density(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    radius: float = 5.0,
) -> pd.Series:
    """Atomic-SPADL: defenders within radius of (x + dx, y + dy).

    Degenerate case: when dx == dy == 0 (instantaneous atomic actions like shots),
    density is computed at the anchor (x, y).

    See NOTICE.

    Examples
    --------
    ::

        from silly_kicks.atomic.tracking.features import receiver_zone_density
        d = receiver_zone_density(atomic_actions, frames, radius=5.0)
    """
    ctx = _resolve_action_frame_context(actions, frames)
    end_x = actions["x"] + actions["dx"]
    end_y = actions["y"] + actions["dy"]
    return _kernels._receiver_zone_density(end_x, end_y, ctx, radius=radius)


def defenders_in_triangle_to_goal(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
) -> pd.Series:
    """Atomic-SPADL: defenders in triangle from (x, y) to goal posts.

    See NOTICE.

    Examples
    --------
    ::

        from silly_kicks.atomic.tracking.features import defenders_in_triangle_to_goal
        d = defenders_in_triangle_to_goal(atomic_actions, frames)
    """
    ctx = _resolve_action_frame_context(actions, frames)
    return _kernels._defenders_in_triangle_to_goal(actions["x"], actions["y"], ctx)


@nan_safe_enrichment
def add_action_context(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    links: pd.DataFrame | None = None,
    receiver_zone_radius: float = 5.0,
) -> pd.DataFrame:
    """Atomic-SPADL aggregator: enrich actions with the 4 features + 4 provenance cols.

    Parallels silly_kicks.tracking.features.add_action_context with atomic-shaped
    column reads (x, y, dx, dy).

    See NOTICE.

    Examples
    --------
    ::

        from silly_kicks.atomic.tracking.features import add_action_context
        enriched = add_action_context(atomic_actions, frames)
    """
    ctx = _resolve_action_frame_context(actions, frames, links=links)
    out = actions.copy()
    out["nearest_defender_distance"] = _kernels._nearest_defender_distance(actions["x"], actions["y"], ctx)
    out["actor_speed"] = _kernels._actor_speed_from_ctx(ctx)
    end_x = actions["x"] + actions["dx"]
    end_y = actions["y"] + actions["dy"]
    rz = _kernels._receiver_zone_density(end_x, end_y, ctx, radius=receiver_zone_radius)
    out["receiver_zone_density"] = rz.astype("Int64")
    dt = _kernels._defenders_in_triangle_to_goal(actions["x"], actions["y"], ctx)
    out["defenders_in_triangle_to_goal"] = dt.astype("Int64")
    pointer_cols = ctx.pointers.set_index("action_id")[
        ["frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]
    ]
    out = out.merge(pointer_cols, left_on="action_id", right_index=True, how="left")
    return out


atomic_tracking_default_xfns = [
    lift_to_states(nearest_defender_distance),
    lift_to_states(actor_speed),
    lift_to_states(receiver_zone_density),
    lift_to_states(defenders_in_triangle_to_goal),
]


# ---------------------------------------------------------------------------
# PR-S21 — atomic pre_shot_gk_* mirror
# ---------------------------------------------------------------------------


def pre_shot_gk_x(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Atomic-SPADL: defending GK's x at the linked frame (m, LTR-normalized).

    Mirrors :func:`silly_kicks.tracking.features.pre_shot_gk_x` with atomic shot type ids
    (``{shot, shot_penalty}`` — atomic does NOT recognize ``shot_freekick``, which is
    collapsed into ``freekick``).

    REQUIRES ``defending_gk_player_id`` column in ``actions``
    (run ``silly_kicks.atomic.spadl.utils.add_pre_shot_gk_context`` first).

    See NOTICE for full bibliographic citations.

    Examples
    --------
    ::

        from silly_kicks.atomic.spadl.utils import add_pre_shot_gk_context
        from silly_kicks.atomic.tracking.features import pre_shot_gk_x
        atomic = add_pre_shot_gk_context(atomic)
        gk_x = pre_shot_gk_x(atomic, frames)

    References
    ----------
    Anzer, G., & Bauer, P. (2021).
    """
    ctx = _resolve_action_frame_context(actions, frames)
    df = _kernels._pre_shot_gk_position(actions["x"], actions["y"], ctx, shot_type_ids=_ATOMIC_SHOT_TYPE_IDS)
    return df["pre_shot_gk_x"].rename("pre_shot_gk_x")


def pre_shot_gk_y(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Atomic-SPADL: defending GK's y at the linked frame.

    See :func:`pre_shot_gk_x` for NaN/REQUIRES contract. See NOTICE for full citations.

    Examples
    --------
    ::

        from silly_kicks.atomic.spadl.utils import add_pre_shot_gk_context
        from silly_kicks.atomic.tracking.features import pre_shot_gk_y
        atomic = add_pre_shot_gk_context(atomic)
        gk_y = pre_shot_gk_y(atomic, frames)

    References
    ----------
    Anzer, G., & Bauer, P. (2021).
    """
    ctx = _resolve_action_frame_context(actions, frames)
    df = _kernels._pre_shot_gk_position(actions["x"], actions["y"], ctx, shot_type_ids=_ATOMIC_SHOT_TYPE_IDS)
    return df["pre_shot_gk_y"].rename("pre_shot_gk_y")


def pre_shot_gk_distance_to_goal(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Atomic-SPADL: Euclidean distance (m) from defending GK to goal-mouth center.

    See :func:`pre_shot_gk_x` for NaN/REQUIRES contract. See NOTICE for full citations.

    Examples
    --------
    ::

        from silly_kicks.atomic.spadl.utils import add_pre_shot_gk_context
        from silly_kicks.atomic.tracking.features import pre_shot_gk_distance_to_goal
        atomic = add_pre_shot_gk_context(atomic)
        d = pre_shot_gk_distance_to_goal(atomic, frames)

    References
    ----------
    Anzer, G., & Bauer, P. (2021).
    """
    ctx = _resolve_action_frame_context(actions, frames)
    df = _kernels._pre_shot_gk_position(actions["x"], actions["y"], ctx, shot_type_ids=_ATOMIC_SHOT_TYPE_IDS)
    return df["pre_shot_gk_distance_to_goal"].rename("pre_shot_gk_distance_to_goal")


def pre_shot_gk_distance_to_shot(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Atomic-SPADL: Euclidean distance (m) from defending GK to shot anchor (action.x, action.y).

    See :func:`pre_shot_gk_x` for NaN/REQUIRES contract. See NOTICE for full citations.

    Examples
    --------
    ::

        from silly_kicks.atomic.spadl.utils import add_pre_shot_gk_context
        from silly_kicks.atomic.tracking.features import pre_shot_gk_distance_to_shot
        atomic = add_pre_shot_gk_context(atomic)
        d = pre_shot_gk_distance_to_shot(atomic, frames)

    References
    ----------
    Anzer, G., & Bauer, P. (2021).
    """
    ctx = _resolve_action_frame_context(actions, frames)
    df = _kernels._pre_shot_gk_position(actions["x"], actions["y"], ctx, shot_type_ids=_ATOMIC_SHOT_TYPE_IDS)
    return df["pre_shot_gk_distance_to_shot"].rename("pre_shot_gk_distance_to_shot")


@nan_safe_enrichment
def add_pre_shot_gk_position(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    links: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Atomic-SPADL aggregator: 4 GK-position columns + 4 linkage-provenance columns.

    Mirrors :func:`silly_kicks.tracking.features.add_pre_shot_gk_position` with atomic
    column reads (``x``, ``y``) and atomic shot type ids (``{shot, shot_penalty}``).

    REQUIRES ``defending_gk_player_id`` column in ``actions``
    (run ``silly_kicks.atomic.spadl.utils.add_pre_shot_gk_context`` first).

    Returns
    -------
    pd.DataFrame
        Input atomic actions with the columns:
        - pre_shot_gk_x (float64, m)
        - pre_shot_gk_y (float64, m)
        - pre_shot_gk_distance_to_goal (float64, m)
        - pre_shot_gk_distance_to_shot (float64, m)
        - frame_id (Int64; NaN if unlinked)
        - time_offset_seconds (float64; NaN if unlinked)
        - link_quality_score (float64; NaN if unlinked)
        - n_candidate_frames (int64)

    Raises
    ------
    ValueError
        If ``defending_gk_player_id`` column is absent.

    See NOTICE.

    Examples
    --------
    ::

        from silly_kicks.atomic.spadl.utils import add_pre_shot_gk_context
        from silly_kicks.atomic.tracking.features import add_pre_shot_gk_position
        atomic = add_pre_shot_gk_context(atomic)
        enriched = add_pre_shot_gk_position(atomic, frames)
    """
    if "defending_gk_player_id" not in actions.columns:
        raise ValueError(
            "add_pre_shot_gk_position: actions missing required column "
            "'defending_gk_player_id'. Run silly_kicks.atomic.spadl.utils.add_pre_shot_gk_context "
            "first to populate it."
        )
    ctx = _resolve_action_frame_context(actions, frames, links=links)
    df = _kernels._pre_shot_gk_position(actions["x"], actions["y"], ctx, shot_type_ids=_ATOMIC_SHOT_TYPE_IDS)
    out = actions.copy()
    for col in ("pre_shot_gk_x", "pre_shot_gk_y", "pre_shot_gk_distance_to_goal", "pre_shot_gk_distance_to_shot"):
        out[col] = df[col]
    pointer_cols = ctx.pointers.set_index("action_id")[
        ["frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]
    ]
    out = out.merge(pointer_cols, left_on="action_id", right_index=True, how="left")
    return out


atomic_pre_shot_gk_default_xfns = [
    lift_to_states(pre_shot_gk_x),
    lift_to_states(pre_shot_gk_y),
    lift_to_states(pre_shot_gk_distance_to_goal),
    lift_to_states(pre_shot_gk_distance_to_shot),
]


# ---------------------------------------------------------------------------
# PR-S24 -- TF-12: atomic mirror of pre_shot_gk_angle_*
# ---------------------------------------------------------------------------


def pre_shot_gk_angle_to_shot_trajectory(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Atomic-SPADL: signed angle (rad) GK->shot vs goal-centre->shot at the linked frame.

    See :func:`silly_kicks.tracking.features.pre_shot_gk_angle_to_shot_trajectory` for full
    semantics. Atomic shot type ids are ``{shot, shot_penalty}``.

    Examples
    --------
    >>> # See tests/test_pre_shot_gk_angle.py for a runnable example.

    References
    ----------
    Anzer, G., & Bauer, P. (2021).
    """
    ctx = _resolve_action_frame_context(actions, frames)
    df = _kernels._pre_shot_gk_angle(actions["x"], actions["y"], ctx, shot_type_ids=_ATOMIC_SHOT_TYPE_IDS)
    return df["pre_shot_gk_angle_to_shot_trajectory"].rename("pre_shot_gk_angle_to_shot_trajectory")


def pre_shot_gk_angle_off_goal_line(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Atomic-SPADL: signed angle (rad) of GK relative to goal-line normal at goal-mouth centre.

    See :func:`silly_kicks.tracking.features.pre_shot_gk_angle_off_goal_line`.

    Examples
    --------
    >>> # See tests/test_pre_shot_gk_angle.py for a runnable example.

    References
    ----------
    Anzer, G., & Bauer, P. (2021).
    """
    ctx = _resolve_action_frame_context(actions, frames)
    df = _kernels._pre_shot_gk_angle(actions["x"], actions["y"], ctx, shot_type_ids=_ATOMIC_SHOT_TYPE_IDS)
    return df["pre_shot_gk_angle_off_goal_line"].rename("pre_shot_gk_angle_off_goal_line")


@nan_safe_enrichment
def add_pre_shot_gk_angle(
    actions: pd.DataFrame,
    *,
    frames: pd.DataFrame,
    links: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Atomic-SPADL aggregator: 2 GK-angle columns at the linked frame.

    Mirrors :func:`silly_kicks.tracking.features.add_pre_shot_gk_angle` with atomic
    column reads (``x``, ``y``) and atomic shot type ids (``{shot, shot_penalty}``).

    REQUIRES ``defending_gk_player_id`` column in ``actions``.

    Examples
    --------
    >>> # See tests/test_pre_shot_gk_angle.py for a runnable example.
    """
    if "defending_gk_player_id" not in actions.columns:
        raise ValueError(
            "add_pre_shot_gk_angle: actions missing required column 'defending_gk_player_id'. "
            "Run silly_kicks.atomic.spadl.utils.add_pre_shot_gk_context first."
        )
    ctx = _resolve_action_frame_context(actions, frames, links=links)
    df = _kernels._pre_shot_gk_angle(actions["x"], actions["y"], ctx, shot_type_ids=_ATOMIC_SHOT_TYPE_IDS)
    out = actions.copy()
    for col in ("pre_shot_gk_angle_to_shot_trajectory", "pre_shot_gk_angle_off_goal_line"):
        out[col] = df[col]
    return out


atomic_pre_shot_gk_angle_default_xfns = [
    lift_to_states(pre_shot_gk_angle_to_shot_trajectory),
    lift_to_states(pre_shot_gk_angle_off_goal_line),
]


# PR-S80: mirror the non-atomic union -- xS (GKDV Layer 2) joins the atomic GK/shot-context
# union too (the xshot_occurrence_xfns factory is action-decomposition-agnostic: it scores the
# possessing team's shot probability at the linked frame, independent of atomic sub-actions).
atomic_pre_shot_gk_full_default_xfns = (
    atomic_pre_shot_gk_default_xfns
    + atomic_pre_shot_gk_angle_default_xfns
    + xshot_occurrence_xfns()
    + xcross_attempt_xfns()
)


# ---------------------------------------------------------------------------
# PR-S25 -- atomic mirror: TF-3 actor_*_pre_window
# ---------------------------------------------------------------------------


def actor_arc_length_pre_window(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    pre_seconds: float = 0.5,
) -> pd.Series:
    """Atomic-SPADL: geometric arc-length of actor's path over pre-action window.

    See :func:`silly_kicks.tracking.features.actor_arc_length_pre_window`.

    Examples
    --------
    >>> from silly_kicks.atomic.tracking.features import actor_arc_length_pre_window
    >>> # See tests/atomic/tracking/test_pre_window_features_atomic.py for runnable examples.
    """
    df = _kernels._actor_pre_window_kernel(actions, frames, pre_seconds=pre_seconds)
    return df["actor_arc_length_pre_window"].rename("actor_arc_length_pre_window")


def actor_displacement_pre_window(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    pre_seconds: float = 0.5,
) -> pd.Series:
    """Atomic-SPADL: net Euclidean displacement over pre-action window.

    See :func:`silly_kicks.tracking.features.actor_displacement_pre_window`.

    Examples
    --------
    >>> from silly_kicks.atomic.tracking.features import actor_displacement_pre_window
    >>> # See tests/atomic/tracking/test_pre_window_features_atomic.py for runnable examples.
    """
    df = _kernels._actor_pre_window_kernel(actions, frames, pre_seconds=pre_seconds)
    return df["actor_displacement_pre_window"].rename("actor_displacement_pre_window")


@nan_safe_enrichment
def add_actor_pre_window(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    links: pd.DataFrame | None = None,
    pre_seconds: float = 0.5,
) -> pd.DataFrame:
    """Atomic-SPADL aggregator for TF-3 features.

    Examples
    --------
    >>> from silly_kicks.atomic.tracking.features import add_actor_pre_window
    >>> # See tests/atomic/tracking/test_pre_window_features_atomic.py for runnable examples.
    """
    df = _kernels._actor_pre_window_kernel(actions, frames, pre_seconds=pre_seconds)
    out = actions.copy()
    out["actor_arc_length_pre_window"] = df["actor_arc_length_pre_window"]
    out["actor_displacement_pre_window"] = df["actor_displacement_pre_window"]
    from silly_kicks.tracking.utils import link_actions_to_frames

    if links is not None:
        pointers = links
    else:
        pointers, _report = link_actions_to_frames(actions, frames)
    pointer_cols = pointers.set_index("action_id")[
        ["frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]
    ]
    out = out.merge(pointer_cols, left_on="action_id", right_index=True, how="left")
    return out


atomic_actor_pre_window_default_xfns = [lift_to_states(actor_arc_length_pre_window)]


# ---------------------------------------------------------------------------
# PR-S25 -- atomic mirror: TF-2 pressure_on_actor multi-flavor
# ---------------------------------------------------------------------------


def pressure_on_actor(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    method: Method = "andrienko_oval",
    params: PressureParams | None = None,
    links: pd.DataFrame | None = None,
) -> pd.Series:
    """Atomic-SPADL: multi-flavor pressure on actor at linked frame.

    Mirrors :func:`silly_kicks.tracking.features.pressure_on_actor` with
    atomic anchor (x, y) instead of (start_x, start_y).

    Examples
    --------
    >>> from silly_kicks.atomic.tracking.features import pressure_on_actor
    >>> # See tests/atomic/tracking/test_pressure_*_atomic.py for runnable examples per method.
    """
    from silly_kicks.tracking.pressure import (
        AndrienkoParams,
        BekkersParams,
        LinkParams,
        validate_params_for_method,
    )

    validate_params_for_method(method, params)
    if method == "andrienko_oval":
        ap = params if isinstance(params, AndrienkoParams) else AndrienkoParams()
        ctx = _resolve_action_frame_context(actions, frames, links=links)
        s = _kernels._pressure_andrienko(actions["x"], actions["y"], ctx, params=ap)
    elif method == "link_zones":
        lp = params if isinstance(params, LinkParams) else LinkParams()
        ctx = _resolve_action_frame_context(actions, frames, links=links)
        s = _kernels._pressure_link(actions["x"], actions["y"], ctx, params=lp)
    elif method == "bekkers_pi":
        bp = params if isinstance(params, BekkersParams) else BekkersParams()
        from silly_kicks.tracking._velocity_availability import velocity_unavailable_by_design

        if velocity_unavailable_by_design(frames):
            # DECLARED velocity-unavailable (SB360 freeze-frame): honest-NaN (ADR-063 amendment).
            # MIRRORS tracking.features.pressure_on_actor -- bekkers_pi's active-pressing
            # speed_threshold filter is a velocity-GATED discrete term, so its zero-velocity form is
            # artifact-dependent (Tier-3): SUPPRESS, do not lift or raise-impossibly on a freeze-frame.
            return pd.Series(np.nan, index=actions.index, name="pressure_on_actor__bekkers_pi")
        if "vx" not in frames.columns or "vy" not in frames.columns:
            raise ValueError(
                "pressure_on_actor(method='bekkers_pi'): frames missing velocity columns "
                "'vx'/'vy'. Run silly_kicks.tracking.preprocess.derive_velocities(frames) "
                "first, or use a provider that emits velocities natively."
            )
        # No whole-batch ball-row guard: _pressure_bekkers falls back per-action to
        # the base model (pressure-on-player only) when ball rows are missing.
        # ball-carrier-max is an improvement, not a requirement (Bekkers 2024 section 2.4). (3.30.0)
        ctx = _resolve_action_frame_context(actions, frames, links=links)
        from silly_kicks.tracking.features import _build_ball_xy_v_per_action

        ball_xy_v = _build_ball_xy_v_per_action(actions, frames, ctx)
        s = _kernels._pressure_bekkers(
            actions["x"],
            actions["y"],
            ctx,
            params=bp,
            ball_xy_v_per_action=ball_xy_v,
        )
    else:
        raise ValueError(f"Unknown method '{method}'.")
    return s.rename(f"pressure_on_actor__{method}")


@nan_safe_enrichment
def add_pressure_on_actor(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    links: pd.DataFrame | None = None,
    methods: tuple[Method, ...] = ("andrienko_oval",),
    params_per_method: dict[Method, PressureParams] | None = None,
) -> pd.DataFrame:
    """Atomic-SPADL aggregator for multi-flavor TF-2 pressure.

    Examples
    --------
    >>> from silly_kicks.atomic.tracking.features import add_pressure_on_actor
    >>> # See tests/atomic/tracking/test_pressure_*_atomic.py for runnable examples.
    """
    from silly_kicks.tracking.pressure import validate_params_for_method

    if params_per_method is None:
        params_per_method = {}
    for m in methods:
        validate_params_for_method(m, params_per_method.get(m))
    out = actions.copy()
    for m in methods:
        s = pressure_on_actor(actions, frames, method=m, params=params_per_method.get(m), links=links)
        out[f"pressure_on_actor__{m}"] = s.values
    from silly_kicks.tracking.utils import link_actions_to_frames

    if links is not None:
        pointers = links
    else:
        pointers, _report = link_actions_to_frames(actions, frames)
    pointer_cols = pointers.set_index("action_id")[
        ["frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]
    ]
    out = out.merge(pointer_cols, left_on="action_id", right_index=True, how="left")
    return out


atomic_pressure_default_xfns = [lift_to_states(pressure_on_actor)]


# ---------------------------------------------------------------------------
# PR-S31 -- TF-7: pitch control at the action destination (atomic variant; renamed at_target in ADR-033)
# ---------------------------------------------------------------------------


def pitch_control_at_target(
    actions: pd.DataFrame,
    frames: pd.DataFrame | None,
    *,
    links: pd.DataFrame | None = None,
    method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
    pitch_control_cache: PitchControlCache | None = None,
) -> pd.Series:
    """Pitch control at ball position for the acting team (atomic SPADL).

    Adapts atomic column names (``x, y``) to standard (``start_x, start_y``)
    and delegates to the standard implementation.

    ``pitch_control_cache`` (TF-7) is forwarded to the standard implementation so a
    caller can share one canonical-surface cache across feature families. ``None``
    (default) is byte-identical to threading nothing.

    Examples
    --------
    Compute pitch control at the acting team's action target (atomic SPADL)::

        from silly_kicks.atomic.tracking.features import pitch_control_at_target
        pc = pitch_control_at_target(actions, frames)
    """
    from silly_kicks.tracking.features import pitch_control_at_target as _std_pc

    if frames is None:
        return _std_pc(actions, None, method=method, pitch_control_cache=pitch_control_cache)

    # The standard kernel now samples the action DESTINATION (end_x, end_y) (ADR-032). Atomic SPADL has no
    # end_*; synthesize it from x,y,dx,dy (mirrors _structural_pass_atomic_endpoints) so the destination
    # re-aim + ADR-028 reprojection apply uniformly.
    adapted = actions.copy()
    adapted["start_x"] = adapted["x"]
    adapted["start_y"] = adapted["y"]
    adapted["end_x"] = adapted["x"] + adapted["dx"]
    adapted["end_y"] = adapted["y"] + adapted["dy"]
    return _std_pc(adapted, frames, links=links, method=method, pitch_control_cache=pitch_control_cache)


@nan_safe_enrichment
def add_pitch_control(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    links: pd.DataFrame | None = None,
    method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
) -> pd.DataFrame:
    """Enrich atomic actions with the ``pitch_control_at_target__<method>`` column (ADR-032).

    Examples
    --------
    Enrich atomic actions with the pitch-control-at-target column::

        from silly_kicks.atomic.tracking.features import add_pitch_control
        enriched = add_pitch_control(actions, frames)
    """
    out = actions.copy()
    s = pitch_control_at_target(actions, frames, links=links, method=method)
    out[s.name] = s.values
    return out


def atomic_pitch_control_xfns(
    method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
    *,
    pitch_control_cache: PitchControlCache | None = None,
) -> list:
    """Factory returning pitch control xfn list for atomic SPADL.

    ``pitch_control_cache`` (TF-7): pass one shared :class:`PitchControlCache` to
    every pitch-control-consuming ``*_xfns`` in a VAEP pass to compute each canonical
    surface once. ``None`` (default) uses a per-call cache -- byte-identical to today.

    Examples
    --------
    >>> from silly_kicks.atomic.tracking.features import atomic_pitch_control_xfns
    >>> xfns = atomic_pitch_control_xfns("spearman")
    """

    def _pc_helper(actions, frames):
        return pitch_control_at_target(actions, frames, method=method, pitch_control_cache=pitch_control_cache)

    _pc_helper.__name__ = f"pitch_control_at_target__{method}"
    return [lift_to_states(_pc_helper)]


atomic_pitch_control_default_xfns = atomic_pitch_control_xfns("spearman")


# ---------------------------------------------------------------------------
# PR-S36 -- TF-30: Cover shadows (atomic variant)
# ---------------------------------------------------------------------------


@nan_safe_enrichment
def add_cover_shadows(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    xt,
    *,
    links: pd.DataFrame | None = None,
    goal_map=None,
    decision_rule: Literal["any", "majority", "all"] = "majority",
    detailed: bool = False,
    method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
) -> pd.DataFrame:
    """Atomic-SPADL aggregator: cover shadow columns.

    Adapts atomic column names (x, y) to standard (start_x, start_y).

    .. versionchanged:: ADR-055
       ``home_team_id`` is **removed** and replaced by an optional ``goal_map``,
       mirroring :func:`silly_kicks.tracking.features.add_cover_shadows`. The atomic
       mirror inherits the change rather than adapting around it -- an atomic wrapper
       that kept an identity parameter the standard aggregator no longer honours would
       accept it and silently drop it.

    Examples
    --------
    Enrich atomic actions with the cover-shadow columns::

        from silly_kicks.atomic.tracking.features import add_cover_shadows
        enriched = add_cover_shadows(atomic_actions, frames, xt)
    """
    from silly_kicks.tracking.features import add_cover_shadows as _std_cs

    adapted = actions.rename(
        columns={"x": "start_x", "y": "start_y"},
        errors="ignore",
    )
    result = _std_cs(
        adapted,
        frames,
        xt,
        links=links,
        goal_map=goal_map,
        decision_rule=decision_rule,
        detailed=detailed,
        method=method,
    )
    # Rename back
    result = result.rename(
        columns={"start_x": "x", "start_y": "y"},
        errors="ignore",
    )
    return result


@nan_safe_enrichment
def add_press_commitment(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    links: pd.DataFrame | None = None,
    params=None,
) -> pd.DataFrame:
    """Atomic-SPADL mirror of ``tracking.features.add_press_commitment`` (TF-51 v2 Item 5).

    A pure PASS-THROUGH (N11/P8): ``compute_press_commitment`` reads only ids + ``time_seconds`` +
    ``team_id`` off the actions and resolves the actor + pressing defender from the linked FRAME --
    it never reads ``start_x``/``start_y`` -- so, unlike ``add_cover_shadows``, no ``x``->``start_x``
    rename bridge is needed (a rename that does nothing is worse than none). It exists for API
    symmetry + discoverability (every ``tracking.features`` aggregator has an ``atomic.tracking``
    twin); the C4 ``atomic.tracking`` +1 is SYMMETRY, not new capability. Gate-covered by PURITY
    alone -- the nan-safety / id-dtype / liveness gates are tracking-only (spec section 8).

    Examples
    --------
    Atomic actions carry the same ids + ``time_seconds`` + ``team_id`` the cue needs::

        from silly_kicks.atomic.tracking.features import add_press_commitment
        enriched = add_press_commitment(atomic_actions, frames)
    """
    from silly_kicks.tracking.features import add_press_commitment as _std_pc

    return _std_pc(actions, frames, links=links, params=params)


# ---------------------------------------------------------------------------
# TF-48: post-shot goalmouth crossing geometry -- atomic mirror (ADR-030)
# ---------------------------------------------------------------------------


@nan_safe_enrichment
def add_shot_goalmouth(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    links: pd.DataFrame | None = None,
    params: ShotGoalmouthParams | None = None,
) -> pd.DataFrame:
    """Atomic-SPADL mirror of tracking.features.add_shot_goalmouth (TF-48). NO coordinate
    synthesis (no end=x+dx): the engine consumes action_id/game_id/period_id/time_seconds/
    team_id/type_id (trajectory from frames, goal end from the GK map) plus the atom's own
    ``x``/``y`` as the OPTIONAL contact anchor (the standard-SPADL ``start_x``/``start_y``
    equivalent; NaN -> un-anchored fit, ADR-003). Atomic shot domain is {shot, shot_penalty}
    (shot_freekick is a `freekick` atom -- intentional, existing pre-shot-GK precedent).

    Examples
    --------
    >>> from silly_kicks.atomic.tracking.features import add_shot_goalmouth
    >>> enriched = add_shot_goalmouth(atomic_actions, frames)  # doctest: +SKIP
    >>> enriched[["shot_crossing_y", "shot_crossing_source"]]  # doctest: +SKIP

    See NOTICE for full bibliographic citations (Anzer & Bauer 2021).
    """
    from silly_kicks.tracking._shot_goalmouth import compute_shot_goalmouth
    from silly_kicks.tracking.utils import link_actions_to_frames

    out = actions.copy()
    comp = compute_shot_goalmouth(actions, frames, links=links, params=params, shot_type_ids=_ATOMIC_SHOT_TYPE_IDS)
    for c in comp.columns:
        out[c] = comp[c].to_numpy() if comp[c].dtype != "boolean" else comp[c].array
    provenance_cols = ["frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]
    if not any(c in out.columns for c in provenance_cols):
        pointers = links if links is not None else link_actions_to_frames(actions, frames)[0]
        if len(pointers) > 0:
            ptr_cols = pointers.set_index("action_id")[provenance_cols]
            out = out.merge(ptr_cols, left_on="action_id", right_index=True, how="left")
    return out


def _atomic_shot_goalmouth_series(actions: pd.DataFrame, frames: pd.DataFrame, col: str) -> pd.Series:
    from silly_kicks.tracking._shot_goalmouth import compute_shot_goalmouth

    return compute_shot_goalmouth(actions, frames, shot_type_ids=_ATOMIC_SHOT_TYPE_IDS)[col].rename(col)


def shot_crossing_y(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Goal-plane crossing y (m, canonical attacked-goal-at-x=105); atomic mirror.

    Examples
    --------
    >>> from silly_kicks.atomic.tracking.features import shot_crossing_y
    >>> shot_crossing_y(atomic_actions, frames).head()  # doctest: +SKIP

    See NOTICE for full bibliographic citations (Anzer & Bauer 2021).
    """
    return _atomic_shot_goalmouth_series(actions, frames, "shot_crossing_y")


def shot_crossing_z(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Goal-plane crossing z (m); atomic mirror. NaN when ball z unavailable.

    Examples
    --------
    >>> from silly_kicks.atomic.tracking.features import shot_crossing_z
    >>> shot_crossing_z(atomic_actions, frames).head()  # doctest: +SKIP

    See NOTICE for full bibliographic citations (Anzer & Bauer 2021).
    """
    return _atomic_shot_goalmouth_series(actions, frames, "shot_crossing_z")


def shot_speed(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Fitted initial ball speed at contact (m/s); atomic mirror (ADR-030 M-1 semantics).

    Examples
    --------
    >>> from silly_kicks.atomic.tracking.features import shot_speed
    >>> shot_speed(atomic_actions, frames).head()  # doctest: +SKIP

    See NOTICE for full bibliographic citations (Anzer & Bauer 2021).
    """
    return _atomic_shot_goalmouth_series(actions, frames, "shot_speed")


def shot_time_to_goal_line(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Elapsed seconds from contact to plane crossing; atomic mirror.

    Examples
    --------
    >>> from silly_kicks.atomic.tracking.features import shot_time_to_goal_line
    >>> shot_time_to_goal_line(atomic_actions, frames).head()  # doctest: +SKIP

    See NOTICE for full bibliographic citations (Anzer & Bauer 2021).
    """
    return _atomic_shot_goalmouth_series(actions, frames, "shot_time_to_goal_line")


def shot_on_target_derived(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """Nullable boolean on-target classification; atomic mirror (tolerance-folded posts/bar).

    Examples
    --------
    >>> from silly_kicks.atomic.tracking.features import shot_on_target_derived
    >>> shot_on_target_derived(atomic_actions, frames).head()  # doctest: +SKIP

    See NOTICE for full bibliographic citations (Anzer & Bauer 2021).
    """
    return _atomic_shot_goalmouth_series(actions, frames, "shot_on_target_derived")
