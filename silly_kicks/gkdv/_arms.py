"""The two gate-independent GKDV physics arms (spec §5) plus their silent-zero guards.

Both arms are defined in ATTACKER-VALUE units as ``actual - ghost``, so **negative =
deterrent** uniformly across arms.

Neither arm accepts a ``pitch_control_cache``, and that is a correctness constraint rather
than a performance choice: ``PitchControlCache``'s key is
``(game_id, period_id, frame_id, team, method, params, ball_position, decompose)``, which
excludes player positions. A ghost frame carries the SAME frame identity as its factual
twin, so a shared cache would serve the counterfactual leg the factual leg's surface and
every delta would collapse to exactly zero with no warning.

Both arms take the FULL factual and counterfactual frames for ONE scored frame. Callers
must restrict to the engine's scored set (``provenance["drop_reason"].isna()``) first:
``build_ghost_frames`` returns the full input with only the defending keeper substituted,
so a dropped frame is byte-identical across the two legs and would contribute exactly the
Delta = 0 the domain exists to exclude.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from silly_kicks.tracking import GoalMap

import numpy as np
import pandas as pd

from ._engine import _DEFAULT_PARAMS, GkdvParams

_FRAME_KEY = ["game_id", "period_id", "frame_id"]


def delta_threat_suppression(
    actual_frame: pd.DataFrame,
    ghost_frame: pd.DataFrame,
    *,
    attacking_team_id: int | str,
    xt,
    goal_map: GoalMap,
    params: GkdvParams = _DEFAULT_PARAMS,
) -> float:
    """Delta-GK-threat-suppression: ``threat_pc(actual) - threat_pc(ghost)``.

    Attacker-value units, so **negative = deterrent**: a keeper positioned better than the
    league-average ghost leaves the attackers less threat, hence
    ``threat_pc(actual) < threat_pc(ghost)``.

    The polarity above is prose; the gate that enforces it is
    ``tests/gkdv/test_arms.py::test_deterrent_keeper_gives_a_NEGATIVE_delta``.

    The pitch-control method is taken from ``params.pitch_control_method`` and threaded
    identically into both legs. There is deliberately NO second method check here:
    ``GkdvParams.__post_init__`` already rejects every GK-blind method at CONSTRUCTION, so
    a bad configuration is unrepresentable rather than merely unusable. A duplicated
    call-time check would be a second source of truth that could drift from the dataclass's
    allowlist -- the one place a future GK-aware method would be registered.

    Parameters
    ----------
    actual_frame, ghost_frame : pd.DataFrame
        The factual frame and its ghost-substituted counterfactual, for ONE scored frame.
    attacking_team_id : int | str
        Team whose threat is measured (the team NOT defended by the substituted keeper).
    xt : ExpectedThreat
        Fitted xT model supplying the per-cell threat weights.
    goal_map : GoalMap
        The defended-goal map for these frames, from
        ``silly_kicks.tracking.resolve_defended_goals``. Threaded into ``compute_threat_pc`` to
        orient each team's attack; ids are matched through the ADR-019 ``silly_kicks.id_compat``
        seam, so a value-equal scalar of a different dtype yields an identical result.
    params : GkdvParams
        Registered knobs. ``pitch_control_method`` and ``lambda_gk`` are consumed here, the
        latter forwarded into ``SpearmanParams`` as the arm's keeper-gain term.

    Returns
    -------
    float
        ``threat_pc(actual) - threat_pc(ghost)``.

    Examples
    --------
    Score one frame against its ghost counterfactual. Both legs must come from the SAME
    :func:`~silly_kicks.gkdv.build_ghost_frames` call, and only frames the engine actually
    scored may be differenced -- a dropped frame is byte-identical across the two legs and
    would contribute a spurious ``0.0``::

        from silly_kicks.gkdv import build_ghost_frames, delta_threat_suppression
        from silly_kicks.tracking import resolve_defended_goals

        ghost_frames, provenance, report = build_ghost_frames(frames, home_team_id=1)
        scored = provenance.loc[provenance["drop_reason"].isna()]
        goal_map = resolve_defended_goals(frames)

        key = ["game_id", "period_id", "frame_id"]
        one = scored.iloc[0][key]
        actual = frames.merge(one.to_frame().T, on=key)
        ghost = ghost_frames.merge(one.to_frame().T, on=key)

        delta = delta_threat_suppression(
            actual, ghost, attacking_team_id=2, xt=fitted_xt, goal_map=goal_map
        )
        # delta < 0  ->  the real keeper suppressed threat vs the league-average ghost.

    Read the SIGN, not the magnitude: the arm is in attacker-value units, so a keeper who
    outperforms the ghost produces a NEGATIVE number. Aggregating raw deltas across
    keepers without first restricting to scored frames is the one mistake this arm cannot
    detect for you.
    """
    # Single-frame arm: a thin wrapper over delta_threat_suppression_batch on a one-frame stack.
    # The batch forwards `lambda_gk` into SpearmanParams and pins the method identically into both
    # legs via the same `tracking.compute_threat_pc` seam (`__post_init__` guarantees "spearman"),
    # so the method-pin spy in tests/gkdv/test_arms.py intercepts both legs unchanged.
    return float(
        delta_threat_suppression_batch(
            actual_frame,
            ghost_frame,
            attacking_team_id_by_frame=attacking_team_id,
            xt=xt,
            goal_map=goal_map,
            params=params,
        ).iloc[0]
    )


def delta_das(
    actual_frame: pd.DataFrame,
    ghost_frame: pd.DataFrame,
    *,
    attacking_team_id: int | str,
    params: GkdvParams = _DEFAULT_PARAMS,
) -> float:
    """Delta-DAS: attacking team's dangerous accessible space, ``actual - ghost``.

    Attacker-value units, so **negative = deterrent**.

    Direction is pinned ONCE on the FACTUAL frames and the SAME pinned column is passed to
    BOTH legs. accessible-space otherwise infers playing direction per period from
    ``groupby(team)[x].mean().idxmin()``, and the ghost displacement perturbs that mean --
    so the two legs could infer OPPOSITE directions and the difference would not be a
    counterfactual at all. Routed through ``get_individual_das`` (summed per team) because
    ``get_das`` hardcodes ``infer_attacking_direction=True`` and cannot accept a pin.

    The direction-pinning rule above is prose; the gate that enforces it is
    ``tests/gkdv/test_arms.py::test_das_arm_passes_ONE_pinned_direction_to_BOTH_legs``.

    NOTE (interpretation limit, spec §5): accessible-space receives no keeper flag
    (``_COLUMN_MAP`` has no ``is_goalkeeper``), so this arm measures the accessible-space
    consequence of relocating a GENERIC player. Keeper-specific physics are not modelled
    here -- unlike the threat arm, where ``lambda_gk`` weights the keeper explicitly.

    Parameters
    ----------
    actual_frame, ghost_frame : pd.DataFrame
        The factual frame(s) and the ghost-substituted counterfactual.
    attacking_team_id : int | str
        Team whose DAS is summed. Compared against ``team_id`` through the ADR-019
        ``silly_kicks.id_compat`` seam inside the port, so a value-equal scalar of a different dtype
        yields an identical result.
    params : GkdvParams
        Registered knobs. Accepted for arm-signature symmetry; DAS consumes none of them
        today (accessible-space carries its own parameterization).

    Returns
    -------
    float
        ``DAS(actual) - DAS(ghost)`` summed over the attacking team's players.

    Examples
    --------
    Same call shape as :func:`delta_threat_suppression`, so the two arms can be scored
    over one pass of the engine's output::

        from silly_kicks.gkdv import build_ghost_frames, delta_das

        ghost_frames, provenance, report = build_ghost_frames(frames, home_team_id=1)
        scored = provenance.loc[provenance["drop_reason"].isna()]

        key = ["game_id", "period_id", "frame_id"]
        one = scored.iloc[0][key]
        actual = frames.merge(one.to_frame().T, on=key)
        ghost = ghost_frames.merge(one.to_frame().T, on=key)

        delta = delta_das(actual, ghost, attacking_team_id=2)
        # delta < 0  ->  the real keeper denied accessible space vs the ghost.

    Do NOT hand the two legs to ``accessible-space`` separately and subtract the results:
    it infers playing direction per period from the team's mean x, the ghost displacement
    perturbs that mean, and the two legs can come back pointing at OPPOSITE goals. This
    function pins direction once on the factual frames and passes that single pinned
    column into both legs, which is the whole reason it exists rather than being a
    two-line call at the use site.

    Requires the optional ``accessible-space`` dependency (the ``[das]`` extra); without
    it the call raises rather than silently returning ``0.0``.
    """
    # Single-frame arm: a thin wrapper over delta_das_batch on a one-frame stack. A one-frame unit
    # pins direction from that frame (identical to the historical per-frame pin), and the batch owns
    # the row-alignment guard, the DasUnscoreableError -> NaN degrade, and the min_count=1 reduce
    # (so a non-simulatable single frame is an honest NaN, not a fictional 0.0).
    return float(
        delta_das_batch(actual_frame, ghost_frame, attacking_team_id_by_frame=attacking_team_id, params=params).iloc[0]
    )


def _assert_legs_aligned(actual_frames: pd.DataFrame, ghost_frames: pd.DataFrame, *, fn: str) -> None:
    """Raise unless the factual and ghost stacks are row-for-row identical on
    ``(game_id, period_id, frame_id, player_id)`` order.

    Both batch arms apply the pinned direction / iterate the two legs POSITIONALLY, so a reordered
    or filtered ghost would be scored against another row's state -- a per-row error invisible in the
    returned scalars. ``build_ghost_frames`` preserves the input order (only the keeper's coordinates
    are rewritten), so a correct caller passes.
    """
    cols = [*_FRAME_KEY, "player_id"]
    a = actual_frames[cols].reset_index(drop=True)
    g = ghost_frames[cols].reset_index(drop=True)
    if not a.equals(g):
        raise ValueError(
            f"{fn}: the factual and ghost frames are not aligned on {cols} order. The pinned "
            "direction is applied positionally, so a misaligned ghost would be scored against "
            "another row's attacking direction. Pass the frames as build_ghost_frames returned "
            "them, restricted identically on both legs."
        )


def _frame_key_index(frames: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(frames[_FRAME_KEY].drop_duplicates())


def delta_das_batch(actual_frames, ghost_frames, *, attacking_team_id_by_frame, params=_DEFAULT_PARAMS):
    """Batched Delta-DAS: one accessible-space call per leg over all a unit's scored frames.

    See :func:`delta_das` for the per-frame semantics; this is its amortized batch form. Direction is
    pinned ONCE over the unit and the same column feeds both legs. Returns a ``pd.Series`` indexed by
    ``(game_id, period_id, frame_id)``, value ``das(actual) - das(ghost)`` (attacker-value units, so
    **negative = deterrent**). A frame with no finite attacking DAS on either leg is NaN
    (``min_count=1``), never a fictional 0.0; a wholly unscoreable unit (velocity-less / dead-ball) is
    all-NaN over its frame keys.

    ``attacking_team_id_by_frame`` is a scalar (one attacking team for the whole unit) or a
    ``pd.Series`` indexed by ``(game_id, period_id, frame_id)``; a Series missing a scored-frame key
    RAISES (fail-loud) rather than silently NaN-ing that frame.

    Examples
    --------
    Both legs MUST come from the SAME ``build_ghost_frames`` call, restricted identically to the
    engine's scored set::

        from silly_kicks.gkdv import build_ghost_frames, delta_das_batch

        ghost_frames, provenance, report = build_ghost_frames(frames, home_team_id=1)
        scored = provenance.loc[provenance["drop_reason"].isna(), ["game_id", "period_id", "frame_id"]]
        actual = frames.merge(scored, on=["game_id", "period_id", "frame_id"])
        ghost = ghost_frames.merge(scored, on=["game_id", "period_id", "frame_id"])

        deltas = delta_das_batch(actual, ghost, attacking_team_id_by_frame=2)
    """
    from silly_kicks.tracking import DasUnscoreableError

    from . import _das_port

    _assert_legs_aligned(actual_frames, ghost_frames, fn="delta_das_batch")
    keys = _frame_key_index(actual_frames)
    try:
        direction = _das_port.pin_direction(actual_frames)  # ONCE over the unit
        actual_pinned = actual_frames.copy()
        actual_pinned["attacking_direction"] = direction.to_numpy()
        ghost_pinned = ghost_frames.copy()
        ghost_pinned["attacking_direction"] = direction.to_numpy()
        actual = _das_port.team_das_by_frame(
            actual_pinned, attacking_team_id_by_frame, direction_col="attacking_direction"
        )
        ghost = _das_port.team_das_by_frame(
            ghost_pinned, attacking_team_id_by_frame, direction_col="attacking_direction"
        )
    except DasUnscoreableError:
        return pd.Series(np.nan, index=keys, name="delta_das")
    delta = (actual - ghost).reindex(keys)  # NaN propagates; reindex pins order/coverage to the keys
    delta.name = "delta_das"
    return delta


def delta_threat_suppression_batch(
    actual_frames, ghost_frames, *, attacking_team_id_by_frame, xt, goal_map, params=_DEFAULT_PARAMS
):
    """Batched Delta-GK-threat-suppression, matching :func:`delta_das_batch`'s index/shape.

    A thin per-frame loop -- the threat arm is ~1 ms/frame (0.16 % of the DAS cost), so no vectorized
    kernel. Both legs are iterated as two aligned ``groupby(KEY)`` streams (identical key order after
    the alignment guard), so there is no per-frame rescan of the ghost stack.

    Unlike :func:`delta_das_batch` there is NO ``DasUnscoreableError`` catch: ``compute_threat_pc``
    takes ``attacking_team_id`` explicitly and reads no ``team_in_possession``, so it always scores
    from positions -- the two arms are independently scoreable (a velocity-less frame is DAS-NaN but
    threat-valued), which is correct because they measure different things.

    Returns a ``pd.Series`` indexed by ``(game_id, period_id, frame_id)`` (attacker-value units, so
    **negative = deterrent**).

    Examples
    --------
    Same call shape as :func:`delta_das_batch`; both legs from one ``build_ghost_frames`` call,
    restricted identically to the scored set::

        from silly_kicks.gkdv import build_ghost_frames, delta_threat_suppression_batch
        from silly_kicks.tracking import resolve_defended_goals

        ghost_frames, provenance, report = build_ghost_frames(frames, home_team_id=1)
        goal_map = resolve_defended_goals(frames)
        deltas = delta_threat_suppression_batch(
            actual, ghost, attacking_team_id_by_frame=2, xt=fitted_xt, goal_map=goal_map
        )
    """
    from silly_kicks import tracking
    from silly_kicks.tracking import SpearmanParams

    from . import _das_port

    _assert_legs_aligned(actual_frames, ghost_frames, fn="delta_threat_suppression_batch")
    att_per_frame = _das_port._attacking_team_by_frame(actual_frames, attacking_team_id_by_frame)  # shared resolver
    base = {
        "xt": xt,
        "goal_map": goal_map,
        "method": params.pitch_control_method,
        "params": SpearmanParams(lambda_gk=params.lambda_gk),
    }
    out = {}
    for (ka, a_sub), (kg, g_sub) in zip(
        actual_frames.groupby(_FRAME_KEY), ghost_frames.groupby(_FRAME_KEY), strict=True
    ):
        assert ka == kg  # noqa: S101 -- _assert_legs_aligned guarantees identical group keys
        atk = att_per_frame[ka]
        a = tracking.compute_threat_pc(a_sub, attacking_team_id=atk, **base)
        g = tracking.compute_threat_pc(g_sub, attacking_team_id=atk, **base)
        out[ka] = float(a - g)
    idx = _frame_key_index(actual_frames)
    if not out:
        return pd.Series(np.nan, index=idx, name="delta_threat_suppression")
    s = pd.Series(out, name="delta_threat_suppression")
    s.index = pd.MultiIndex.from_tuples(list(s.index), names=_FRAME_KEY)
    return s.reindex(idx)
