"""TerritorialDefenseParams -- frozen params for the TF-54b SB360 territorial-defense metric.

Mirrors ``GkdvParams`` (ADR-043) / ``RestDefenseParams`` (ADR-080): frozen, an EMPTY per-provider
override map (ADR-009 -- a per-provider tune is a separate gated apply PR, never this cycle), and a
fail-at-construction ``__post_init__`` that makes a GK-blind pitch-control method UNREPRESENTABLE
(the ``lambda_gk`` keeper control-agent term lives only on ``SpearmanParams``).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TerritorialDefenseParams:
    """Parameters for the territorial-defense removal counterfactual (spec §3, §5, §7).

    Attributes
    ----------
    defensive_action_type_ids:
        SPADL ``type_id`` set defining Arm A's domain (D's own defensive interventions):
        ``(9, 10, 18)`` = tackle / interception / clearance.
    pitch_control_method:
        MUST be ``"spearman"`` -- the only GK-aware method (``lambda_gk`` control-agent term). A
        GK-blind method is rejected at construction, not merely discouraged.
    lambda_gk:
        Keeper control-rate gain fed to ``SpearmanParams`` (spec §6). *(calibratable)*
    trim_fraction:
        Trimmed-hull retained fraction for Arm B (reuses v1 territory's 0.70). *(calibratable)*
    own_half_max_x:
        Arm-B hull own-half cut: a defensive action counts toward D's territory hull only if its
        action-LTR ``start_x`` is below this (default 52.5 = midfield). Matches the sibling
        ``territory`` package's ``own_half_max_x`` rather than a hardcoded ``fl / 2``. *(calibratable)*
    min_local_observed_fraction:
        SPEC-02 local-completeness gate: a scored frame's re-absorbing neighbourhood (a disk of
        radius ``local_radius_m`` around D) must be at least this fraction observed, else the frame
        is dropped ``fov_cropped_local`` (removal biases the delta upward on a cropped
        neighbourhood). *(calibratable)*
    local_radius_m:
        Radius of the SPEC-02 re-absorbing-neighbourhood disk (metres). *(calibratable)*
    min_defenders_after_removal:
        PLAN-01 guard: removing D must leave at least this many defending-team players, else the
        frame is dropped ``removal_undersupported`` (a 0-defender counterfactual degrades to
        attacker-controls-all -- a finite but upward-biased outlier).
    arm_b_rule:
        Arm B contesting-defender rule (``"nearest_to_target"`` in v1; ``"receiver_lane"`` is
        reserved for a validated follow-on and not yet accepted).

    Examples
    --------
    >>> from silly_kicks.territorial_defense._config import TerritorialDefenseParams  # DEMOTED (ADR-090)
    >>> p = TerritorialDefenseParams()
    >>> p.defensive_action_type_ids, p.pitch_control_method
    ((9, 10, 18), 'spearman')
    """

    defensive_action_type_ids: tuple[int, ...] = (9, 10, 18)
    pitch_control_method: Literal["spearman"] = "spearman"
    lambda_gk: float = 3.0
    trim_fraction: float = 0.70
    own_half_max_x: float = 52.5
    min_local_observed_fraction: float = 0.7
    local_radius_m: float = 10.0
    min_defenders_after_removal: int = 1
    arm_b_rule: str = "nearest_to_target"

    _GK_AWARE_METHODS = ("spearman",)
    _ARM_B_RULES = ("nearest_to_target",)

    def __post_init__(self) -> None:
        """Fail at CONSTRUCTION on a GK-blind method or an unknown Arm-B rule (mirrors GkdvParams)."""
        if self.pitch_control_method not in self._GK_AWARE_METHODS:
            raise ValueError(
                f"pitch_control_method={self.pitch_control_method!r} is GK-blind: lambda_gk exists "
                f"only on SpearmanParams, so the keeper control-agent term would be lost entirely. "
                f"Allowed: {self._GK_AWARE_METHODS}."
            )
        if self.arm_b_rule not in self._ARM_B_RULES:
            raise ValueError(f"arm_b_rule={self.arm_b_rule!r} not in {self._ARM_B_RULES}.")

    @classmethod
    def for_provider(cls, provider: str) -> TerritorialDefenseParams:
        """Per-provider params; base config for an unlisted provider (ADR-009 -- map ships EMPTY).

        Examples
        --------
        >>> TerritorialDefenseParams.for_provider("statsbomb") == TerritorialDefenseParams()
        True
        """
        return dataclasses.replace(cls(), **_PROVIDER_TD_PARAMS.get(provider, {}))


#: EMPTY until an ADR-009 calibration apply-gate clears (a per-provider tune is a separate PR).
_PROVIDER_TD_PARAMS: dict[str, dict] = {}

_DEFAULT_PARAMS = TerritorialDefenseParams()
