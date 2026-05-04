"""Bit-equivalent parity between silly-kicks _bekkers_tti and UnravelSports canonical.

Per spec section 8.7 (lakehouse review item 2): "single highest-leverage addition".
Without this, the 'direct port' claim is aspirational. With it, drift is detected
at every CI run.

Source-of-truth: prefer the live ``unravelsports`` package if installed (CI 3.11+
job has it via the ``[golden-master]`` extra); fall back to a vendored
30-line BSD-3-Clause excerpt at ``tests/_vendored/unravelsports_tti.py`` so the
test runs unconditionally -- including on Python 3.10 (silly-kicks supports
3.10+; ``unravelsports>=1.2`` requires 3.11+; the vendored copy is the spec
section 8.7 fallback so the parity gate has zero skip conditions).
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from silly_kicks.tracking._kernels import _bekkers_tti


def _import_canonical_tti():
    """Return the canonical ``time_to_intercept`` -- live unravelsports if
    available, else the vendored BSD-3-Clause excerpt."""
    if importlib.util.find_spec("unravelsports") is not None:
        from unravel.soccer.models.utils import time_to_intercept  # type: ignore[reportMissingImports]

        return time_to_intercept, "unravelsports"
    from tests._vendored.unravelsports_tti import time_to_intercept

    return time_to_intercept, "vendored"


@pytest.mark.parametrize(
    "name,p1,p2,v1,v2",
    [
        ("stationary_at_zero", [[50, 34]], [[50, 34]], [[0, 0]], [[0, 0]]),
        ("stationary_at_d10", [[50, 34]], [[60, 34]], [[0, 0]], [[0, 0]]),
        ("moving_defender", [[50, 34]], [[55, 34]], [[3, 0]], [[0, 0]]),
        ("moving_target", [[50, 34]], [[55, 34]], [[0, 0]], [[2, 0]]),
        ("relative_motion", [[48, 32]], [[52, 36]], [[2, 1]], [[-1, -1]]),
        ("multiple_defenders", [[48, 32], [52, 36]], [[50, 34]], [[2, 0], [0, -1]], [[0, 0]]),
        ("away_from_target", [[50, 34]], [[60, 34]], [[-3, 0]], [[0, 0]]),  # angle penalty
    ],
)
def test_bekkers_tti_byte_equivalent_to_unravelsports(name: str, p1, p2, v1, v2) -> None:
    canonical_tti, source = _import_canonical_tti()

    p1_arr = np.asarray(p1, dtype=float)
    p2_arr = np.asarray(p2, dtype=float)
    v1_arr = np.asarray(v1, dtype=float)
    v2_arr = np.asarray(v2, dtype=float)
    ours = _bekkers_tti(
        p1=p1_arr,
        p2=p2_arr,
        v1=v1_arr,
        v2=v2_arr,
        reaction_time=0.7,
        max_object_speed=12.0,
    )
    theirs = canonical_tti(
        p1=p1_arr,
        p2=p2_arr,
        v1=v1_arr,
        v2=v2_arr,
        reaction_time=0.7,
        max_object_speed=12.0,
    )
    np.testing.assert_allclose(
        ours, theirs, rtol=1e-9, atol=1e-12, err_msg=f"{name} divergence (canonical source: {source})"
    )
