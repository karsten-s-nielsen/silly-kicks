"""Frozen parameters for the TF-63 in-game win-probability model.

``for_provider`` is empty (ADR-009): no per-provider tuning ships in v1.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class WinProbabilityParams:
    """Frozen thresholds for the in-game win-probability model (TF-63).

    Attributes
    ----------
    interval_minutes : int
        Width of a Markov-chain step, in minutes (default 1).
    regulation_minutes : int
        Regulation match length used for the ``time_remaining`` floor (default 90).
    lattice_pad : int
        Half-width ``K`` of the ``score_diff`` lattice ``[-K, +K]``; mass beyond the pad is
        absorbed at the edge and pinned negligible by the expected-goals / edge-mass gates.
    ece_max : float
        Calibration gate: maximum expected calibration error (default 0.10).
    slope_tol : float
        Calibration gate: maximum ``|reliability_slope - 1|`` (default 0.25).
    leverage_nonneg_atol : float
        Bundle-time coherence gate tolerance for ``leverage >= 0`` (default 1e-9).

    Examples
    --------
    >>> from silly_kicks.win_probability import WinProbabilityParams
    >>> WinProbabilityParams.default().interval_minutes
    1
    >>> WinProbabilityParams.for_provider("skillcorner") == WinProbabilityParams.default()
    True
    """

    interval_minutes: int = 1
    regulation_minutes: int = 90
    lattice_pad: int = 10
    ece_max: float = 0.10
    slope_tol: float = 0.25
    leverage_nonneg_atol: float = 1e-9

    @classmethod
    def default(cls) -> WinProbabilityParams:
        """The canonical default parameters.

        Examples
        --------
        >>> WinProbabilityParams.default().lattice_pad
        10
        """
        return cls()

    @classmethod
    def for_provider(cls, provider: str) -> WinProbabilityParams:
        """Per-provider parameters. Empty per ADR-009 (no tuning ships in v1).

        Examples
        --------
        >>> WinProbabilityParams.for_provider("sportec") == WinProbabilityParams.default()
        True
        """
        return cls.default()
