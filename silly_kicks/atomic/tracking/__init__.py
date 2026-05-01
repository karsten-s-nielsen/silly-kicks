"""Tracking-aware features for atomic SPADL.

Mirrors silly_kicks.tracking.features with atomic-SPADL column conventions
(x, y, dx, dy instead of start_x, start_y, end_x, end_y).
"""

from . import features

__all__ = ["features"]
