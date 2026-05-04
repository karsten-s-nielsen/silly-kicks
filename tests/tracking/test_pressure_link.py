"""Link, Lang & Seidenschwarz 2016 zone-based pressure kernel.

References (see NOTICE):
- Link, D., Lang, S., & Seidenschwarz, P. (2016), PLoS ONE 11(12): e0168768.
"""

from __future__ import annotations

import math

import pytest

from silly_kicks.tracking._kernels import _pressure_link
from silly_kicks.tracking.pressure import LinkParams
from tests.tracking.test_pressure_andrienko import _build_ctx  # reuse fixture builder


def test_link_pr_d2_in_hoz() -> None:
    """Defender at d=2 in HOZ (alpha<45) -> PR_Di = 1 - 2/4 = 0.5; aggregation 1-exp(-1*0.5) ~= 0.3935."""
    ax, ay, ctx = _build_ctx(actor_xy=(50.0, 34.0), defenders_xy=[(52.0, 34.0)])
    out = _pressure_link(ax, ay, ctx, params=LinkParams())
    expected = 1.0 - math.exp(-1.0 * 0.5)
    assert out.iloc[0] == pytest.approx(expected, rel=1e-9)


def test_link_pr_at_hoz_boundary() -> None:
    """Defender at d=4 in HOZ (alpha=0) -> at boundary -> PR_Di = 0 -> aggregate 0."""
    ax, ay, ctx = _build_ctx(actor_xy=(50.0, 34.0), defenders_xy=[(54.0, 34.0)])
    out = _pressure_link(ax, ay, ctx, params=LinkParams())
    assert out.iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_link_pr_in_lz() -> None:
    """Defender at alpha=60deg (LZ band [45, 90)) at d=1.5 -> PR_Di = 1 - 1.5/3 = 0.5; aggregate 1-exp(-0.5)."""
    angle = math.radians(60.0)
    d = 1.5
    defender = (50.0 + d * math.cos(angle), 34.0 + d * math.sin(angle))
    ax, ay, ctx = _build_ctx(actor_xy=(50.0, 34.0), defenders_xy=[defender])
    out = _pressure_link(ax, ay, ctx, params=LinkParams())
    expected = 1.0 - math.exp(-0.5)
    assert out.iloc[0] == pytest.approx(expected, rel=1e-9)


def test_link_pr_in_hz() -> None:
    """Defender behind actor at alpha=180deg d=1 -> in HZ (r=2) -> PR_Di = 1 - 1/2 = 0.5."""
    ax, ay, ctx = _build_ctx(actor_xy=(50.0, 34.0), defenders_xy=[(49.0, 34.0)])
    out = _pressure_link(ax, ay, ctx, params=LinkParams())
    expected = 1.0 - math.exp(-0.5)
    assert out.iloc[0] == pytest.approx(expected, rel=1e-9)


def test_link_pr_outside_hz() -> None:
    """Defender behind actor at d=3 (beyond HZ r=2) -> PR=0."""
    ax, ay, ctx = _build_ctx(actor_xy=(50.0, 34.0), defenders_xy=[(47.0, 34.0)])
    out = _pressure_link(ax, ay, ctx, params=LinkParams())
    assert out.iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_link_two_defenders_each_pr_half() -> None:
    """Two HOZ defenders each PR_Di~=0.5 -> sum~=1.0; aggregate 1-exp(-1.0) ~= 0.6321."""
    ax, ay, ctx = _build_ctx(
        actor_xy=(50.0, 34.0),
        defenders_xy=[(52.0, 34.0), (52.0, 33.999)],  # both essentially Theta=0
    )
    out = _pressure_link(ax, ay, ctx, params=LinkParams())
    expected = 1.0 - math.exp(-1.0)
    assert out.iloc[0] == pytest.approx(expected, rel=1e-3)


def test_link_aggregation_bounded_in_zero_one() -> None:
    """Saturating aggregation always gives output in [0, 1]."""
    ax, ay, ctx = _build_ctx(
        actor_xy=(50.0, 34.0),
        defenders_xy=[(50.5 + i * 0.1, 34.0 + i * 0.1) for i in range(5)],
    )
    out = _pressure_link(ax, ay, ctx, params=LinkParams())
    assert 0.0 <= out.iloc[0] <= 1.0


def test_link_zero_defenders_returns_zero() -> None:
    """Linked, no defenders -> aggregate 1-exp(-0) = 0."""
    ax, ay, ctx = _build_ctx(actor_xy=(50.0, 34.0), defenders_xy=[])
    out = _pressure_link(ax, ay, ctx, params=LinkParams())
    assert out.iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_link_axially_symmetric() -> None:
    """Defender at +y vs -y -> equal pressure."""
    ax_pos, ay_pos, ctx_pos = _build_ctx(actor_xy=(50.0, 34.0), defenders_xy=[(50.0, 34.0 + 1.5)])
    ax_neg, ay_neg, ctx_neg = _build_ctx(actor_xy=(50.0, 34.0), defenders_xy=[(50.0, 34.0 - 1.5)])
    p_pos = _pressure_link(ax_pos, ay_pos, ctx_pos, params=LinkParams()).iloc[0]
    p_neg = _pressure_link(ax_neg, ay_neg, ctx_neg, params=LinkParams()).iloc[0]
    assert p_pos == pytest.approx(p_neg, rel=1e-9)
