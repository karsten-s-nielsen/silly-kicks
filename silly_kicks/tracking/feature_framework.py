"""Tracking-aware feature framework primitives.

Public surface:
- ActionFrameContext: linked-context structure consumed by per-feature kernels.
- lift_to_states: lift (actions, frames) -> Series helper to FrameAwareTransformer.
- Frames: re-export from silly_kicks.vaep.feature_framework for consumer convenience.

See ADR-005 for the integration contract; spec
docs/superpowers/specs/2026-04-30-action-context-pr1-design.md for full design.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import pandas as pd

from silly_kicks.vaep.feature_framework import (
    FrameAwareTransformer,
    Frames,
    frame_aware,
    is_frame_aware,
)

__all__ = [
    "ActionFrameContext",
    "FrameAwareTransformer",
    "Frames",
    "frame_aware",
    "is_frame_aware",
    "lift_to_states",
]


@dataclasses.dataclass(frozen=True)
class ActionFrameContext:
    """Linkage + actor / opposite-team / defending-GK frame slices, computed once per call.

    Attributes
    ----------
    actions : pd.DataFrame
        Subset of input actions (index aligned with the per-feature output Series).
    pointers : pd.DataFrame
        link_actions_to_frames output: action_id, frame_id, time_offset_seconds,
        n_candidate_frames, link_quality_score.
    actor_rows : pd.DataFrame
        One row per linked action: the actor's frame row (player_id == action.player_id).
        Rows for unlinked actions or missing-actor cases have NaN x/y/speed.
    opposite_rows_per_action : pd.DataFrame
        Long-form: (action_id, opposite-team frame row) pairs. Used by geometric
        feature kernels via groupby('action_id').
    defending_gk_rows : pd.DataFrame
        Long-form: one row per (linked action, frame row) where
        ``frame.player_id == action.defending_gk_player_id`` AND ``not is_ball``.
        Empty DataFrame when ``defending_gk_player_id`` column is absent from
        ``actions``, when the ID is NaN, when the action is unlinked, or when
        the GK player is absent from the linked frame (substitution case).
        Consumed by ``silly_kicks.tracking._kernels._pre_shot_gk_position``.

    Direct construction of ``ActionFrameContext`` is not part of the public API;
    always build via ``silly_kicks.tracking.utils._resolve_action_frame_context``.

    Examples
    --------
    Build the context once and pass to multiple feature kernels::

        from silly_kicks.tracking.utils import _resolve_action_frame_context
        ctx = _resolve_action_frame_context(actions, frames)
        # ctx.actor_rows / ctx.opposite_rows_per_action / ctx.defending_gk_rows
        # consumed by per-feature kernels.
    """

    actions: pd.DataFrame
    pointers: pd.DataFrame
    actor_rows: pd.DataFrame
    opposite_rows_per_action: pd.DataFrame
    defending_gk_rows: pd.DataFrame = dataclasses.field(default_factory=pd.DataFrame)


def lift_to_states(
    helper: Callable[[pd.DataFrame, pd.DataFrame], pd.Series],
    nb_states: int = 3,
) -> FrameAwareTransformer:
    """Lift a (actions, frames) -> Series helper to a (states, frames) -> Features transformer.

    The returned transformer is marked ``_frame_aware = True`` so
    ``VAEP.compute_features`` dispatches frames to it.

    Output columns: ``f"{helper.__name__}_a0"`` ... ``f"..._a{nb_states-1}"``.

    Examples
    --------
    Compose a tracking-aware feature into HybridVAEP::

        from silly_kicks.vaep.hybrid import HybridVAEP, hybrid_xfns_default
        from silly_kicks.tracking.feature_framework import lift_to_states
        from silly_kicks.tracking.features import nearest_defender_distance

        v = HybridVAEP(xfns=hybrid_xfns_default + [lift_to_states(nearest_defender_distance)])
    """
    name = helper.__name__

    def transformer(states, frames):
        out = pd.DataFrame(index=states[0].index)
        for i, slot in enumerate(states[:nb_states]):
            out[f"{name}_a{i}"] = helper(slot, frames).to_numpy()
        return out

    transformer._frame_aware = True  # type: ignore[attr-defined]
    transformer.__name__ = f"lifted_{name}"
    return transformer
