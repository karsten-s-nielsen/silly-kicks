"""Parameter surfaces for the pluggable xT model.

House-style string-dispatch + frozen-dataclass params (mirrors tracking/pressure.py and
tracking/pitch_control/_params.py). See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import silly_kicks.spadl.config as spadlconfig

Method = Literal["singh_counts", "kde_smoothed"]
KdeKernel = Literal["gaussian", "epanechnikov", "tophat", "exponential", "linear", "cosine"]


@dataclass(frozen=True)
class GridSpec:
    """Grid resolution for the xT model. Pitch dimensions live in ``spadlconfig`` (SSOT).

    Examples
    --------
    Build a higher-resolution grid::

        from silly_kicks.xthreat import GridSpec

        grid = GridSpec(n_zones_x=24, n_zones_y=16)
        grid.n_zones  # 384
    """

    n_zones_x: int = 16
    n_zones_y: int = 12

    def __post_init__(self) -> None:
        if self.n_zones_x < 1 or self.n_zones_y < 1:
            raise ValueError(f"GridSpec requires positive dimensions, got {self.n_zones_x}x{self.n_zones_y}")

    @property
    def n_zones(self) -> int:
        """Total zone count (``n_zones_x * n_zones_y``).

        Examples
        --------
        ::

            GridSpec(16, 12).n_zones  # 192
        """
        return self.n_zones_x * self.n_zones_y

    @property
    def cell_length(self) -> float:
        """Zone width along the pitch length, in SPADL metres.

        Examples
        --------
        ::

            GridSpec(12, 8).cell_length  # 105 / 12
        """
        return spadlconfig.field_length / self.n_zones_x

    @property
    def cell_width(self) -> float:
        """Zone height along the pitch width, in SPADL metres.

        Examples
        --------
        ::

            GridSpec(12, 8).cell_width  # 68 / 8
        """
        return spadlconfig.field_width / self.n_zones_y


@dataclass(frozen=True)
class SinghParams:
    """No parameters — row-normalized empirical counts (classic Singh 2018).

    Examples
    --------
    Select the classic counts transition (the default)::

        from silly_kicks.xthreat import ExpectedThreat, SinghParams

        xt = ExpectedThreat(method="singh_counts", params=SinghParams())
    """


@dataclass(frozen=True)
class KDEParams:
    """Per-source-zone 2D KDE smoothing of the transition matrix.

    bandwidth : multiplier on the Silverman rule when ``adaptive`` (else the raw sklearn
        ``KernelDensity`` bandwidth, in SPADL metres). Default 1.0 = pure Silverman — a
        conservative, corpus-agnostic baseline that robustly beats Singh counts at every scale
        tested. NOTE: the held-out-NLL-optimal multiplier is strongly corpus-size-dependent
        (adaptive Silverman shrinks per-zone h ~ n^(-1/6), so larger corpora need a larger
        multiplier): ~1.0 on a 64-match sample, but >=4 on an 8.9M-action production mart. Tune
        it for your corpus with ``compute_holdout_nll``. See ADR-021.
    adaptive : per-source-zone bandwidth from Silverman's rule on that row's destinations.
    kernel : sklearn ``KernelDensity`` kernel name.

    Examples
    --------
    Tune the KDE smoothing bandwidth::

        from silly_kicks.xthreat import ExpectedThreat, KDEParams

        xt = ExpectedThreat(method="kde_smoothed", params=KDEParams(bandwidth=4.0, adaptive=True))
    """

    bandwidth: float = 1.0
    adaptive: bool = True
    kernel: KdeKernel = "gaussian"


XtParams = SinghParams | KDEParams
_METHOD_TO_PARAMS_TYPE: dict[Method, type] = {
    "singh_counts": SinghParams,
    "kde_smoothed": KDEParams,
}


def validate_params_for_method(method: Method, params: XtParams | None) -> None:
    """Raise if ``params`` is the wrong type for ``method``. ``None`` always allowed (defaults).

    Examples
    --------
    Guard a method/params pairing::

        from silly_kicks.xthreat import KDEParams, validate_params_for_method

        validate_params_for_method("kde_smoothed", KDEParams())  # ok
        validate_params_for_method("singh_counts", KDEParams())  # raises TypeError
    """
    if method not in _METHOD_TO_PARAMS_TYPE:
        raise ValueError(f"Unknown xT method {method!r}; expected one of {list(_METHOD_TO_PARAMS_TYPE)}.")
    if params is None:
        return
    expected_type = _METHOD_TO_PARAMS_TYPE[method]
    if not isinstance(params, expected_type):
        raise TypeError(
            f"method={method!r} expects {expected_type.__name__}, got {type(params).__name__}. "
            f"Use {expected_type.__name__}() (or omit params=) for defaults."
        )
