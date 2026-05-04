"""Multi-flavor dispatch + per-method param dataclasses for pressure_on_actor.

Three published methodologies, each with a frozen parameter dataclass:
  - andrienko_oval -- Andrienko et al. 2017 directional oval (default)
  - link_zones    -- Link, Lang & Seidenschwarz 2016 piecewise zones
  - bekkers_pi    -- Bekkers 2024 Pressing Intensity (probabilistic TTI)

See:
  - docs/superpowers/specs/2026-05-03-tf3-tf2-design.md sections 4.1, 4.6
  - ADR-005 section 8 (multi-flavor xfn column-naming convention)
  - NOTICE for full bibliographic citations + BSD-3-Clause attribution
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Method = Literal["andrienko_oval", "link_zones", "bekkers_pi"]


@dataclass(frozen=True)
class AndrienkoParams:
    """Parameters for Andrienko 2017 oval pressure model.

    Defaults from Andrienko et al. (2017) section 3.1, calibrated with football
    experts: D_front=9 m, D_back=3 m, q=1.75. See NOTICE.
    """

    q: float = 1.75
    d_front: float = 9.0
    d_back: float = 3.0


@dataclass(frozen=True)
class LinkParams:
    """Parameters for Link, Lang & Seidenschwarz 2016 zone-based pressure model.

    Zone radii and angular boundaries from Figure 2 of the paper:
      - HOZ (Head-On Zone, toward goal):   r=4 m, alpha in [0, 45) degrees
      - LZ  (Lateral Zone, side):           r=3 m, alpha in [45, 90) degrees
      - HZ  (Hind Zone, behind):            r=2 m, alpha in [90, 180] degrees
    Mirrored for negative angles (axially symmetric across actor->goal axis).

    NOTE: Eq (3) of Link 2016 is honored as the formal specification; the paper's
    prose-described "High Pressure Zone (HPZ) constant high pressure" inner arc
    is qualitative and not implemented as a separate clamp (Plan A:
    equation-faithful, no discontinuity at 1 m).

    NOTE: k3 default = 1.0 is an engineering choice -- the paper explicitly
    states k1..k5 were "calibrated manually with experts" and does not
    publish numerical values. Calibration deferred post-release to Optuna
    sweep (silly-kicks TODO TF-24); see NOTICE.
    """

    r_hoz: float = 4.0
    r_lz: float = 3.0
    r_hz: float = 2.0
    angle_hoz_lz_deg: float = 45.0
    angle_lz_hz_deg: float = 90.0
    k3: float = 1.0


@dataclass(frozen=True)
class BekkersParams:
    """Parameters for Bekkers 2024 Pressing Intensity (probabilistic TTI model).

    All defaults verified against canonical UnravelSports BSD-3-Clause source
    at the SHA pinned in NOTICE:
      - reaction_time = 0.7 s
        (unravel/soccer/models/pressing_intensity.py L120: _reaction_time = 0.7)
      - sigma = 0.45
        (pressing_intensity.py L121: _sigma = 0.45)
      - time_threshold = 1.5 s
        (pressing_intensity.py L122: _time_threshold = 1.5)
      - speed_threshold = 2.0 m/s
        (active-pressing filter; paper section 3.1 + blog Fig 2 caption)
      - max_player_speed = 12.0 m/s
        (unravel/soccer/dataset/kloppy_polars.py L160: max_player_speed: float = 12.0)
      - use_ball_carrier_max = True (paper section 2.4 ball-carrier improvement)

    See NOTICE for BSD-3-Clause attribution.
    """

    reaction_time: float = 0.7
    sigma: float = 0.45
    time_threshold: float = 1.5
    speed_threshold: float = 2.0
    max_player_speed: float = 12.0
    use_ball_carrier_max: bool = True


PressureParams = AndrienkoParams | LinkParams | BekkersParams
_METHOD_TO_PARAMS_TYPE: dict[Method, type] = {
    "andrienko_oval": AndrienkoParams,
    "link_zones": LinkParams,
    "bekkers_pi": BekkersParams,
}


def validate_params_for_method(method: Method, params: PressureParams | None) -> None:
    """Raise loudly if method/params combination is invalid.

    Per ``feedback_post_init_validator_for_invalid_combinations`` and
    ``feedback_loud_raise_for_required_input_columns``: fail at the public-API
    boundary rather than silently coercing. Per spec section 4.6 ADR-005 amendment.

    Examples
    --------
    >>> validate_params_for_method("andrienko_oval", AndrienkoParams())
    >>> validate_params_for_method("link_zones", None)  # None means use defaults
    """
    if method not in _METHOD_TO_PARAMS_TYPE:
        raise ValueError(f"Unknown method '{method}'. Valid methods: {sorted(_METHOD_TO_PARAMS_TYPE)}")
    if params is None:
        return
    expected_type = _METHOD_TO_PARAMS_TYPE[method]
    if not isinstance(params, expected_type):
        raise TypeError(
            f"method='{method}' expects {expected_type.__name__}, "
            f"got {type(params).__name__}. "
            f"Use {expected_type.__name__}() (or omit params=) for defaults."
        )
