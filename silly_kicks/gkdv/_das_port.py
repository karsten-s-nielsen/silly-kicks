"""The ONE seam through which gkdv reaches native DAS (ADR-107).

gkdv scores a counterfactual (factual keeper vs ghost), so the two legs MUST see the SAME
attacking direction: direction is resolved ONCE, on the FACTUAL frames, as a ``GoalMap``
(ADR-055) and threaded into BOTH legs. The native engine reads the goal map verbatim rather
than re-inferring per leg, so the ghost displacement cannot flip a leg's direction and turn
the delta into a non-counterfactual.

Two public-tracking calls sit behind this port so the direction guard runs on every CI leg:

* ``resolve_defended_goals`` / ``get_individual_das`` -- already PUBLIC ``silly_kicks.tracking``
  seams, consumed as such.
* ``individual_das_paired`` -- the CONFINED private seam (``silly_kicks.tracking._das``): the
  ADR-043-safe paired path that scores factual+ghost together (SC-1), keyed here by the
  allowlist exemption in ``tests/gkdv/test_import_allowlist.py``. It is the ONLY private import
  gkdv makes, and it lives in this one module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from silly_kicks.id_compat import ids_equal, ids_match

if TYPE_CHECKING:
    from silly_kicks.tracking import GoalMap

_FRAME_KEY = ["game_id", "period_id", "frame_id"]


def pin_direction(frames: pd.DataFrame) -> GoalMap:
    """The attacking direction, pinned ONCE from the FACTUAL frames as a ``GoalMap`` (ADR-055).

    Threading this same map into both counterfactual legs keeps direction fixed across the ghost
    substitution: the native DAS engine resolves each in-possession team's attacked end from the
    map instead of inferring it per leg, so a ghost that perturbs a team's mean-x cannot flip a
    leg's direction.

    Examples
    --------
    >>> gm = pin_direction(frames)                 # doctest: +SKIP
    >>> gm.attacked_goal(1, 1, 2, allow_guess=True)  # doctest: +SKIP
    105.0
    """
    from silly_kicks.tracking import resolve_defended_goals

    return resolve_defended_goals(frames)


def _attacking_team_by_frame(frames, attacking_team_id_by_frame):
    """``{(game_id, period_id, frame_id): attacking_team}`` for every DISTINCT scored frame.

    Scalar -> broadcast to every key; ``pd.Series`` -> looked up per key, RAISING (fail-loud) if any
    scored frame's key is absent (an incomplete caller mapping is a bug, and a silent NaN would hide
    it). ONE resolver shared by the single-leg and paired reduces.
    """
    keys = [tuple(k) for k in frames[_FRAME_KEY].drop_duplicates().to_numpy()]
    if isinstance(attacking_team_id_by_frame, pd.Series):
        missing = [k for k in keys if k not in attacking_team_id_by_frame.index]
        if missing:
            raise KeyError(
                f"attacking_team_id_by_frame is missing {len(missing)} scored-frame key(s), e.g. "
                f"{missing[:3]}. Supply one entry per scored frame; gkdv fails loud rather than "
                "silently NaN-ing a frame."
            )
        return {k: attacking_team_id_by_frame.loc[k] for k in keys}
    return {k: attacking_team_id_by_frame for k in keys}


def _reduce_team_das_by_frame(out: pd.DataFrame, attacking_team_id_by_frame) -> pd.Series:
    """Per-frame attacking-team DAS sum over a per-player ``get_individual_das`` output.

    ``min_count=1`` so a frame with no finite attacking DAS is NaN, never the fictional ``0.0``
    that ``DAS.dropna().sum()`` yields on an empty selection.
    """
    att_map = _attacking_team_by_frame(out, attacking_team_id_by_frame)
    att_per_row = pd.Series(pd.MultiIndex.from_frame(out[_FRAME_KEY]).map(att_map), index=out.index)
    # ``ids_equal`` returns a POSITIONAL fresh-RangeIndex result (ADR-019); ``out`` can carry a
    # NON-CONTIGUOUS index (a filtered frame slice), so combine the two masks via numpy to avoid
    # pandas LABEL alignment -- a label-aligned ``&`` silently yields all-False when the indexes do
    # not overlap (measured: the SB360 velocity-full leg zeroed every attacking player).
    is_att_player = (~out["is_ball"].astype(bool)).to_numpy() & ids_equal(out["team_id"], att_per_row).to_numpy()
    das = out["DAS"].where(is_att_player)  # NaN outside the attacking team's players
    result = das.groupby([out[k] for k in _FRAME_KEY]).sum(min_count=1)
    result.index.names = _FRAME_KEY
    return result


def team_das(frames: pd.DataFrame, *, attacking_team_id: int | str, goal_map: GoalMap) -> float:
    """Sum per-player DAS for the attacking team under a PINNED ``goal_map`` (ADR-055).

    Examples
    --------
    >>> team_das(frames, attacking_team_id=2, goal_map=gm)  # doctest: +SKIP
    41.7
    """
    from silly_kicks.tracking import get_individual_das

    out = get_individual_das(frames, goal_map=goal_map)
    # House idiom (~df["is_ball"].astype(bool)), NOT `!= True`: on a nullable BooleanDtype or
    # object column `pd.NA != True` yields pd.NA. ids_match (ADR-019) handles the id column vs the
    # caller-supplied scalar -- a raw `==` there mis-resolves silently across dtypes.
    rows = out[~out["is_ball"].astype(bool) & ids_match(out["team_id"], attacking_team_id)]
    return float(rows["DAS"].dropna().sum())


def team_das_by_frame(frames: pd.DataFrame, attacking_team_id_by_frame, *, goal_map: GoalMap) -> pd.Series:
    """Per-frame attacking-team DAS over a single stack under a PINNED ``goal_map``.

    ONE ``get_individual_das`` call over the whole stack, then a per-``(game_id, period_id,
    frame_id)`` reduce (``min_count=1``). Returns a ``pd.Series`` indexed by
    ``MultiIndex(game_id, period_id, frame_id)``.
    """
    from silly_kicks.tracking import get_individual_das

    out = get_individual_das(frames, goal_map=goal_map)
    return _reduce_team_das_by_frame(out, attacking_team_id_by_frame)


def paired_team_das_by_frame(
    actual: pd.DataFrame, counterfactual: pd.DataFrame, attacking_team_id_by_frame, *, goal_map: GoalMap
) -> tuple[pd.Series, pd.Series]:
    """Per-frame attacking-team DAS for BOTH counterfactual legs, through the ADR-043-safe seam.

    Routes through :func:`silly_kicks.tracking._das.individual_das_paired` (the CONFINED private
    seam) so the two legs are scored together under one ``goal_map`` and the ADR-043 frame-keyed
    landmine is structurally impossible (SC-1). Each leg is then reduced to a per-frame
    attacking-team sum (``min_count=1``). Returns ``(actual_by_frame, ghost_by_frame)``.
    """
    from silly_kicks.tracking._das import individual_das_paired

    a_out, c_out = individual_das_paired(actual, counterfactual, goal_map=goal_map)
    return (
        _reduce_team_das_by_frame(a_out, attacking_team_id_by_frame),
        _reduce_team_das_by_frame(c_out, attacking_team_id_by_frame),
    )
