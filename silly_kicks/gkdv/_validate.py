"""GKDV validation harness: spec §6.1-6.3 registered constants + §6.4 Layer 4.

Layers 0-3 and ``gkdv_discrimination_verdict`` are PR-3b, after owner sign-off -- they are
deliberately NOT built here. Layer 4 ships in PR-3 (spec §6.4 review round 2, N3(a))
because it GATES §6.1's ICC, which ships here: shipping the primary criterion without the
guard the spec says must precede its interpretation would hand anyone running §6.1 between
the two PRs a number the spec forbids interpreting, with nothing saying so.

Every constant below is REGISTERED -- locked in code before the owner run. Changing one
after a run has been executed invalidates the pre-registration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Pre-registered ICC anchor band (spec §1.3, measured 0.015-0.026 across cohorts).
#: A RANGE, not a point: SkillCorner's 0.0147 sits below a single 0.02 anchor, so a
#: power curve is reported at all three rather than at a midpoint.
ICC_ANCHORS: tuple[float, float, float] = (0.015, 0.020, 0.026)

#: Layer 4: minimum mean signed goal-relative depth separation between outer terciles,
#: in metres. Below this the arm is not tracking a behaviour keepers actually vary.
TERCILE_SEPARATION_M: float = 0.5

#: Per-arm expected direction (spec §5): both arms are ATTACKER-value, so a deterrent
#: keeper reads NEGATIVE on both. Registered here so the §6.2 sign panel cannot pick a
#: direction after seeing the data.
EXPECTED_DIRECTION: dict[str, str] = {"delta_das": "negative", "delta_threat": "negative"}

#: The two Layer 4 verdicts. ``uninterpretable`` is NOT a failure of the keepers -- it is a
#: statement that the ICC computed on this arm carries no information about them.
_ANCHORED = "anchored"
_UNINTERPRETABLE = "uninterpretable"


def behavioural_anchoring_verdict(per_keeper: pd.DataFrame, *, value_col: str, depth_col: str) -> str:
    """Layer 4: is the arm tracking a behaviour keepers actually VARY?

    Splits keepers into terciles by arm value; the top and bottom terciles must differ in
    mean signed goal-relative depth by at least :data:`TERCILE_SEPARATION_M`. If they do
    not, the arm's ICC is reported ``"uninterpretable"`` rather than as evidence.

    This is the guard the sibling possession-value metric's failure teaches: a metric can
    read flat because it rewards a behaviour good keepers do not perform, in which case an
    ICC near zero says nothing about keepers.

    **Rows with a missing value or depth are dropped before ranking**, and the mechanism is
    worth stating precisely because the obvious guess is wrong. A NaN does NOT poison the
    tercile mean -- pandas' ``mean`` skips it. What it does is worse, because it is silent:
    ``sort_values`` parks every NaN-VALUE row at the END, so those rows are ranked as though
    they were the highest-valued keepers and their real depths are averaged into the TOP
    tercile. They also inflate ``len(ranked)``, widening ``k`` and pulling additional
    genuine keepers into both outer terciles. Measured on a fixture whose 9 real keepers
    separate by 4.50 m: three NaN-value rows collapse it to 0.375 m -- flipping ``anchored``
    to ``uninterpretable`` with nothing in the output to say why.

    The verdict is a magnitude, not a direction: Layer 4 asks whether keepers vary, and §6.2
    (:data:`EXPECTED_DIRECTION`) is where direction is adjudicated.

    LIMITATION, stated rather than guarded: with fewer than three keepers the tercile
    reduces to the single extreme row on each side. The spec registers no minimum keeper
    count for Layer 4, so none is invented here -- the §6.1 clustering floors
    (``aggregate_by_keeper``'s ``min_nonzero``/``min_games``) are what thin the surface, and
    they run first.

    Parameters
    ----------
    per_keeper : pd.DataFrame
        One row per keeper, as returned by
        :func:`~silly_kicks.gkdv.aggregate_by_keeper` joined to a depth column.
        Never mutated.
    value_col : str
        Per-keeper arm value used for the tercile ranking.
    depth_col : str
        Per-keeper mean SIGNED goal-relative depth (delta-x), in metres.

    Returns
    -------
    str
        ``"anchored"`` or ``"uninterpretable"``.

    Examples
    --------
    >>> per_keeper = pd.DataFrame(
    ...     {"value": [-0.03, 0.0, 0.03], "signed_dx": [-2.0, 0.0, 2.0]}
    ... )
    >>> behavioural_anchoring_verdict(per_keeper, value_col="value", depth_col="signed_dx")
    'anchored'
    """
    ranked = per_keeper.dropna(subset=[value_col, depth_col]).sort_values(value_col)
    if not len(ranked):
        return _UNINTERPRETABLE
    k = max(1, len(ranked) // 3)
    lo = float(ranked.head(k)[depth_col].mean())
    hi = float(ranked.tail(k)[depth_col].mean())
    separation = abs(hi - lo)
    if not np.isfinite(separation):
        return _UNINTERPRETABLE
    return _ANCHORED if separation >= TERCILE_SEPARATION_M else _UNINTERPRETABLE
