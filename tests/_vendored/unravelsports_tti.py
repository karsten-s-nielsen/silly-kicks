"""Vendored excerpt of UnravelSports's ``time_to_intercept`` + ``probability_to_intercept``.

Source: https://github.com/UnravelSports/unravelsports
Path:   unravel/soccer/models/utils.py
Fetched: 2026-05-04 (silly-kicks 3.2.0 / PR-S25)

Vendored only because ``unravelsports>=1.2`` requires Python 3.11+ and the
silly-kicks ``requires-python = ">=3.10"`` matrix would otherwise force the
Bekkers golden-master parity test to skip on the 3.10 CI job. The full
package is too heavy to vendor; only these two functions are needed for the
parity test.

License header preserved verbatim from the source per BSD-3-Clause attribution
clause (1).

----

BSD 3-Clause License

Copyright (c) 2025 [UnravelSports]

See: https://opensource.org/licenses/BSD-3-Clause

This project includes code and contributions from:
    - Joris Bekkers (UnravelSports)

Permission is hereby granted to redistribute this software under the BSD
3-Clause License, with proper attribution.
"""

from __future__ import annotations

import numpy as np


def probability_to_intercept(
    time_to_intercept: np.ndarray,
    tti_sigma: float,
    tti_time_threshold: float,
) -> np.ndarray:
    exponent = -np.pi / np.sqrt(3.0) / tti_sigma * (tti_time_threshold - time_to_intercept)
    # Avoid Overflow errors; np.exp does not like values above ~700.
    exponent = np.clip(exponent, -700, 700)
    return 1 / (1.0 + np.exp(exponent))


def time_to_intercept(
    p1: np.ndarray,
    p2: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    reaction_time: float,
    max_object_speed: float,
) -> np.ndarray:
    """Verbatim port of UnravelSports's ``time_to_intercept`` (utils.py)."""
    u = (p1 + v1) - p1  # Adjusted velocity of Pressing Players
    d2 = p2 + v2  # Destination of Players Under Pressure

    v = d2[:, None, :] - p1[None, :, :]  # Relative motion vector

    u_mag = np.linalg.norm(u, axis=-1)
    v_mag = np.linalg.norm(v, axis=-1)
    dot_product = np.sum(u * v, axis=-1)

    epsilon = 1e-10  # avoid dividing by zero
    angle = np.arccos(dot_product / (u_mag * v_mag + epsilon))

    r_reaction = p1 + v1 * reaction_time  # Adjusted pressing-player position after reaction time
    d = d2[:, None, :] - r_reaction[None, :, :]

    t = (
        u_mag * angle / np.pi  # Angular contribution
        + reaction_time
        + np.linalg.norm(d, axis=-1) / max_object_speed
    )
    return t
