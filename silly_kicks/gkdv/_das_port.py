"""The ONLY seam through which gkdv touches accessible-space.

Two functions, both resolved through this module so a test can stub them without the
optional ``[das]`` extra installed. Keeping the contact surface here (rather than importing
tracking internals from ``_arms.py``) means:

* the structural direction-pinning guard runs on EVERY CI leg, not only where
  ``accessible-space`` happens to be installed -- a guard for a live hazard that silently
  skips is not a guard;
* gkdv depends on a seam rather than on ``_das``'s internals, which is the correct
  hexagonal boundary regardless of what the test needs.

``_pin_attacking_direction`` calls ``_import_accessible_space()`` in its own body and
imports ``infer_playing_direction`` directly, so stubbing only the DAS scorer would NOT be
sufficient -- both functions must sit behind this port.

Only ``_pin_attacking_direction`` is a private import: ``get_individual_das`` is already a
PUBLIC ``silly_kicks.tracking`` seam and is consumed as one. (The plan's allowlist comment
asserted both were private; that is wrong for ``get_individual_das``, and importing it
publicly keeps the private-exemption surface as small as it can honestly be.)
"""

from __future__ import annotations

import pandas as pd

from silly_kicks.id_compat import ids_match


def pin_direction(frames: pd.DataFrame) -> pd.Series:
    """Per-row attacking direction inferred ONCE from the FACTUAL frames.

    Examples
    --------
    >>> pin_direction(frames)  # doctest: +SKIP
    0    1.0
    Name: attacking_direction, dtype: float64
    """
    from silly_kicks.tracking._das import _pin_attacking_direction

    return _pin_attacking_direction(frames)["attacking_direction"]


def team_das(frames: pd.DataFrame, *, attacking_team_id: int | str, direction_col: str) -> float:
    """Sum per-player DAS for the attacking team under a PINNED direction column.

    Examples
    --------
    >>> team_das(frames, attacking_team_id=2, direction_col="attacking_direction")  # doctest: +SKIP
    41.7
    """
    from silly_kicks.tracking import get_individual_das

    out = get_individual_das(frames, attacking_direction_col=direction_col)
    # House idiom (~df["is_ball"].astype(bool)), NOT `!= True`: on a nullable BooleanDtype or
    # object column `pd.NA != True` yields pd.NA, and a mask carrying pd.NA behaves differently
    # from a plain bool mask. ids_match (ADR-019) handles the id column vs the caller-supplied
    # scalar -- a raw `==` there mis-resolves silently across dtypes.
    rows = out[~out["is_ball"].astype(bool) & ids_match(out["team_id"], attacking_team_id)]
    return float(rows["DAS"].dropna().sum())


_FRAME_KEY = ["game_id", "period_id", "frame_id"]


def _attacking_team_by_frame(frames, attacking_team_id_by_frame):
    """``{(game_id, period_id, frame_id): attacking_team}`` for every DISTINCT scored frame.

    Scalar -> broadcast to every key; ``pd.Series`` -> looked up per key, RAISING (fail-loud) if any
    scored frame's key is absent (an incomplete caller mapping is a bug, and a silent NaN would hide
    it). ONE resolver shared by the DAS reduce and the threat loop.
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


def team_das_by_frame(frames, attacking_team_id_by_frame, *, direction_col):
    """Per-frame attacking-team DAS sum under a PINNED direction column, over a multi-frame stack.

    ONE accessible-space call over the whole stack (the amortization), then a per-``(game_id,
    period_id, frame_id)`` reduce over the attacking team's players with ``min_count=1`` so a frame
    with no finite attacking DAS is NaN, never the fictional ``0.0`` that ``DAS.dropna().sum()``
    yields on an empty selection.

    Returns a ``pd.Series`` indexed by ``MultiIndex(game_id, period_id, frame_id)``.
    """
    from silly_kicks.id_compat import ids_equal
    from silly_kicks.tracking import get_individual_das

    out = get_individual_das(frames, attacking_direction_col=direction_col)
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
