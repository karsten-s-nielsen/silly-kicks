"""Structural perf guard (TF-49): packing_xfns must call the shared kernel 3x (once
per gamestate slot), NOT 9x. Deterministic call-count, not a wall-clock ceiling."""

from __future__ import annotations

import silly_kicks.tracking._kernels as _kernels
from silly_kicks.tracking.features import packing_xfns
from tests.tracking.test_defensive_line import _make_frame_rows
from tests.tracking.test_packing_kernel import _acts


def test_kernel_called_once_per_slot(monkeypatch):
    calls = {"n": 0}
    orig = _kernels._packing_at_actions

    def spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(_kernels, "_packing_at_actions", spy)
    frame = _make_frame_rows(
        home_outfield_xs=[20.0, 22.0, 24.0, 26.0],
        home_outfield_ys=[20.0, 30.0, 40.0, 50.0],
        away_outfield_xs=[50.0, 60.0, 30.0, 80.0],
        away_outfield_ys=[34.0, 20.0, 34.0, 34.0],
    )
    transformer = packing_xfns(home_team_id=1)[0]
    slot = _acts([{}, {}])
    transformer([slot, slot, slot], frame)
    assert calls["n"] == 3
