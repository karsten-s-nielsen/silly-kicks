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

import pandas as pd

from ._engine import _DEFAULT_PARAMS, GkdvParams


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
    from silly_kicks import tracking
    from silly_kicks.tracking import SpearmanParams

    # `lambda_gk` is the ONLY term through which this arm sees the keeper, so it must be
    # FORWARDED rather than left to the pitch-control default -- otherwise a caller raising
    # it would silently get the default gain and the registered params echoed into
    # GkdvReport would misreport the run. `__post_init__` guarantees "spearman" here, which
    # is what makes the concrete SpearmanParams construction type-correct; a future GK-aware
    # method joining that allowlist must extend this mapping too, or it lands back on the
    # default gain. Every other field takes its SpearmanParams default, so the default
    # GkdvParams is byte-identical to passing `params=None`.
    kwargs = {
        "attacking_team_id": attacking_team_id,
        "xt": xt,
        "goal_map": goal_map,
        "method": params.pitch_control_method,
        "params": SpearmanParams(lambda_gk=params.lambda_gk),
    }
    # Module-attribute access at CALL time so a spy can intercept both legs and assert the
    # method pin -- see tests/gkdv/test_arms.py.
    actual = tracking.compute_threat_pc(actual_frame, **kwargs)
    ghost = tracking.compute_threat_pc(ghost_frame, **kwargs)
    return float(actual - ghost)


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
    from . import _das_port  # module-attribute access at CALL time -> stubbable

    # The pin is applied POSITIONALLY to both legs, so a ghost whose rows do not line up
    # with the factual frame would silently receive another row's direction -- a per-row
    # sign flip, invisible in the returned scalar. `build_ghost_frames` returns the full
    # input with only the keeper's coordinates rewritten, so the index is preserved; a
    # caller that has reordered or filtered one leg has broken the counterfactual anyway.
    if not actual_frame.index.equals(ghost_frame.index):
        raise ValueError(
            "delta_das: the factual and ghost frames are not row-aligned (differing index). "
            "The pinned direction is applied positionally, so a misaligned ghost would be "
            "scored against another row's attacking direction. Pass the frames as "
            "build_ghost_frames returned them, restricted identically on both legs."
        )

    # ONE direction, inferred from the FACTUAL frames, applied to BOTH legs. Neither leg
    # may infer for itself.
    direction = _das_port.pin_direction(actual_frame)
    actual_pinned = actual_frame.copy()
    actual_pinned["attacking_direction"] = pd.Series(direction).to_numpy()
    ghost_pinned = ghost_frame.copy()
    ghost_pinned["attacking_direction"] = pd.Series(direction).to_numpy()

    kwargs = {"attacking_team_id": attacking_team_id, "direction_col": "attacking_direction"}
    actual = _das_port.team_das(actual_pinned, **kwargs)
    ghost = _das_port.team_das(ghost_pinned, **kwargs)
    return float(actual - ghost)
