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
