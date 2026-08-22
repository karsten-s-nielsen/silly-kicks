"""Task 10 (R5): the pre-registered apply-gate thresholds are pinned constants."""

from __future__ import annotations

from scripts import _cover_shadow_thresholds as thr


def test_thresholds_are_pinned():
    assert thr.MIN_COVERAGE == 0.30
    assert thr.MIN_RECEIVER_MARGIN == 0.05
    assert thr.MAX_BIAS_SHARE == 0.50
    assert thr.MAX_AMBIGUOUS_RATE == 0.20  # R6: GS failure-mode tagging reliability ceiling
