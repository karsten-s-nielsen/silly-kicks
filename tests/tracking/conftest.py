"""Shared fixture helpers for tracking tests (PR-S27: TF-13/TF-14, PR-S28: TF-5, PR-S30: TF-4)."""

from __future__ import annotations

# Re-export shared helpers so invariant tests can import from conftest
# without fragile cross-file test imports.
from tests.tracking.test_ball_carrier import _make_carrier_frame
from tests.tracking.test_defensive_line import _make_frame_rows
from tests.tracking.test_gk_resolve import _make_actions, _make_frames
from tests.tracking.test_off_ball_runs import _make_action_at, _make_multi_frame_fixture

__all__ = [
    "_make_action_at",
    "_make_actions",
    "_make_carrier_frame",
    "_make_frame_rows",
    "_make_frames",
    "_make_multi_frame_fixture",
]
