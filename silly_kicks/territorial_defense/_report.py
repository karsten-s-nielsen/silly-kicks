"""TerritorialDefenseReport -- conservation bookkeeping for compute_territorial_defense (TF-54b).

Field names mirror ``GkdvReport`` / ``RestDefenseReport`` (``n_frames_in`` / ``n_frames_scored`` /
``drop_reasons``). Conservation (``n_frames_scored + sum(drop_reasons.values()) == n_frames_in``) is
asserted by a CI gate, not a dataclass property -- an unscoreable frame is dropped-AND-COUNTED
(ADR-042), never silently lost.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ._config import TerritorialDefenseParams


@dataclass(frozen=True)
class TerritorialDefenseReport:
    """Per-``compute_territorial_defense`` conservation report (ADR-042), for BOTH arms.

    Arm A conserves over the Arm-A domain (``n_frames_scored + sum(drop_reasons) == n_frames_in``).
    Arm B conserves over its qualifying ``(defender, in-hull opponent-pass)`` pairs
    (``arm_b_n_scored + sum(arm_b_drop_reasons) == arm_b_n_in``) -- a pass NOT landing in a defender's
    territory is out of domain (not counted), while a missing frame / depleted removal / unresolvable
    geometry / non-finite delta is dropped-AND-counted.

    Examples
    --------
    Both conservations hold exactly:

    >>> from silly_kicks.territorial_defense import TerritorialDefenseParams, TerritorialDefenseReport
    >>> r = TerritorialDefenseReport(
    ...     TerritorialDefenseParams(), 10, 6, {"fov_cropped_local": 3, "removal_undersupported": 1},
    ...     arm_b_n_in=8, arm_b_n_scored=5, arm_b_drop_reasons={"removal_undersupported": 2, "missing_frame": 1},
    ... )
    >>> r.n_frames_scored + sum(r.drop_reasons.values()) == r.n_frames_in
    True
    >>> r.arm_b_n_scored + sum(r.arm_b_drop_reasons.values()) == r.arm_b_n_in
    True
    """

    params: TerritorialDefenseParams
    n_frames_in: int
    n_frames_scored: int
    drop_reasons: dict = field(default_factory=dict)
    #: Arm-B conservation over qualifying (defender, in-hull opponent-pass) pairs (ADR-042).
    arm_b_n_in: int = 0
    arm_b_n_scored: int = 0
    arm_b_drop_reasons: dict = field(default_factory=dict)
