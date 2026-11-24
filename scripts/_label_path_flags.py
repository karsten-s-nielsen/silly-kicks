"""Spies on the W5.6 label path of ``infer_positions`` (spec 2026-11-08 tracking-batch W5.6).

Used by the W5.7 role-gate driver (``scripts/validate_shape_graph_roles.py`` diagnostics) to count the label-path
classes of each team-frame.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator


@contextlib.contextmanager
def label_flag_spies() -> Iterator[dict]:
    """The label-path flags of the ``infer_positions`` calls inside the block.

    - ``pitch_thirds``: the pitch-thirds rule ran on the vertical axis (its ``levels`` list holds vertical labels).
    - ``lateral_fallback``: the equal-frequency fallback ran with the lateral ladder.
    - ``lateral_middle_tie``: the lateral middle group is exactly two players with equal values, as their sorted
      player indices.

    Examples
    --------
    >>> import numpy as np
    >>> from silly_kicks.tracking import compute_shape_graph, infer_positions
    >>> from scripts._label_path_flags import label_flag_spies
    >>> xy = np.array([[20.0, 10.0], [20.0, 58.0], [40.0, 25.0], [40.0, 43.0], [60.0, 34.0]])
    >>> with label_flag_spies() as flags:
    ...     _ = infer_positions(compute_shape_graph(xy), xy, 1.0)
    >>> flags
    {}
    """
    import silly_kicks.tracking._shape_graph as sgm

    flags: dict = {}
    real_thirds, real_eqf, real_middle = sgm._pitch_thirds_level, sgm._equal_frequency_assign, sgm._decompose_middle

    def thirds(values, middle_indices, levels, **kwargs):
        if levels and levels[0] in sgm._VERTICAL_LEVELS:
            flags["pitch_thirds"] = True
        return real_thirds(values, middle_indices, levels, **kwargs)

    def eqf(values, levels):
        if tuple(levels) == tuple(sgm._HORIZONTAL_LEVELS):
            flags["lateral_fallback"] = True
        return real_eqf(values, levels)

    def middle(middle_indices, values, *args, **kwargs):
        if kwargs.get("axis") == 1 and len(middle_indices) == 2:
            a, b = (int(i) for i in middle_indices)
            if values[a] == values[b]:
                flags["lateral_middle_tie"] = (min(a, b), max(a, b))
        return real_middle(middle_indices, values, *args, **kwargs)

    sgm._pitch_thirds_level, sgm._equal_frequency_assign, sgm._decompose_middle = thirds, eqf, middle
    try:
        yield flags
    finally:
        sgm._pitch_thirds_level, sgm._equal_frequency_assign, sgm._decompose_middle = real_thirds, real_eqf, real_middle
