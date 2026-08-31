"""Shared velocity-availability contract (extracted from _das.py, TF-51 v2).

The one place the "structurally unavailable vs caller-forgot-derive_velocities()" distinction lives,
so every velocity consumer (DAS, press-commitment) reads it identically instead of each raising
unconditionally or silently filling zeros.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from .schema import SPEED_SOURCE_UNAVAILABLE


def velocity_unavailable_by_design(frames: pd.DataFrame) -> bool:
    """True iff EVERY row declares kinematics structurally unavailable.

    Reads the ``speed_source == SPEED_SOURCE_UNAVAILABLE`` marker a frame builder stamps when its
    source has no per-player temporal history to differentiate (the freeze-frame shape -- see
    :data:`~silly_kicks.tracking.SPEED_SOURCE_UNAVAILABLE`). Absent the marker, missing ``vx``/``vy``
    is a caller bug ("forgot ``derive_velocities()``") and must fail loud; the whole point of the
    marker is that the two shapes are otherwise byte-identical at this seam.

    ALL rows must be marked, not any: a PARTIALLY marked frame set means some genuine velocity-bearing
    source is also missing its velocity, which is the caller bug -- so the fail-loud branch wins. An
    empty frame set is not marked (nothing declared it).
    """
    if "speed_source" not in frames.columns or len(frames) == 0:
        return False
    return bool((frames["speed_source"] == SPEED_SOURCE_UNAVAILABLE).all())


def variant_key_for_velocity(frames: pd.DataFrame) -> Literal["default", "position_only"]:
    """Pure 2-way variant key (Layer A of the velocity-keyed model auto-select).

    ``"position_only"`` iff the frame set DECLARES velocity structurally unavailable
    (:func:`velocity_unavailable_by_design`), else ``"default"``. No IO, no fallback, no raise -- the
    velocity analogue of ``variant_key_for_provider``. An EMPTY or partially-marked set is not
    all-unavailable, so it keys ``"default"``; a mixed set is caught separately by
    :func:`velocity_availability_is_mixed` at the serve seam.
    """
    return "position_only" if velocity_unavailable_by_design(frames) else "default"


def velocity_availability_is_mixed(frames: pd.DataFrame) -> bool:
    """True iff SOME-but-not-ALL rows declare velocity unavailable (a mixed frame set).

    :func:`velocity_unavailable_by_design` requires the marker on EVERY row and returns ``False`` on a
    partially-marked set -- so without this guard a mixed set (some freeze-frame rows, some
    velocity-bearing) would resolve to the DEFAULT velocity variant and the marked rows would get
    ``speed=NaN`` fabricated (the ADR-054 defect reappearing on mixed frames). The serve seam RAISES on
    a mixed set: mixed velocity-availability is a caller error. Empty / no ``speed_source`` column ->
    ``False`` (nothing declared).
    """
    if "speed_source" not in frames.columns or len(frames) == 0:
        return False
    n = int((frames["speed_source"] == SPEED_SOURCE_UNAVAILABLE).sum())
    return 0 < n < len(frames)


def zero_velocity_if_unavailable(frames: pd.DataFrame, *, method: str = "spearman") -> pd.DataFrame:
    """Prepare frames for a velocity-REQUIRING pitch-control call (ADR-063).

    The single edge seam that decides degrade-vs-raise from the ``speed_source`` marker, so the
    four velocity-requiring pitch-control aggregators (``gk_influence``/``cover_shadows``/
    ``player_influence``/``space_creation``) and ``pitch_control_at_target`` all read the same
    contract instead of each zero-filling loosely or raising unconditionally:

    - ``vx``/``vy`` PRESENT -> returns ``frames`` UNCHANGED (no copy, no mutation).
    - ``vx``/``vy`` ABSENT and the frame set DECLARES velocity structurally unavailable
      (:func:`velocity_unavailable_by_design` -- ``speed_source == SPEED_SOURCE_UNAVAILABLE`` on
      EVERY row, the freeze-frame shape) -> returns a COPY with ``vx``=``vy``=0.0 so a
      velocity-requiring method computes the zero-velocity positional model (the Spearman
      reaction-time limit -- a weaker model, not a fabrication; ADR-053).
    - ``vx``/``vy`` ABSENT, the marker NOT set, and ``method`` requires velocity -> RAISES
      ``ValueError``. A forgotten ``derive_velocities()`` is a caller BUG, not a declared-
      velocity-less provider, and the two shapes are otherwise byte-identical at this seam. We
      fail FAST at this policy edge rather than let a downstream broad ``except`` swallow the
      raise into an all-NaN column that is "indistinguishable downstream from legitimately-
      absent" (the ADR-043 discipline). The input is never mutated on the raise path.
    - ``vx``/``vy`` ABSENT, the marker NOT set, and ``method`` needs NO velocity (``voronoi``)
      -> returns ``frames`` UNCHANGED: a missing-velocity frame is not a bug for a position-only
      model.

    Policy lives HERE at the edge; the ``compute_pitch_control`` dispatch stays a pure engine
    that raises when it cannot compute (the same rule that puts the ghost clamp at the serving
    seam and the ``xt_gk`` base-rate switch in ``compute_xt_gk``, not in ``predict_*``).

    The zero-velocity COPY intentionally LEAVES ``speed_source == SPEED_SOURCE_UNAVAILABLE`` in
    place, so it is internally inconsistent if passed ONWARD (a consumer reading ``speed_source``
    sees "unavailable" on a frame that now carries velocity columns). That is fine for the
    immediate pitch-control call it is built for; do not forward it.

    Examples
    --------
    Prepare a tracking frame for a Spearman pitch-control call::

        from silly_kicks.tracking import zero_velocity_if_unavailable

        prepared = zero_velocity_if_unavailable(frame, method="spearman")
        # `prepared is frame` when vx/vy are present; a zero-velocity COPY on a declared-velocity-
        # less SB360 freeze-frame; a ValueError if vx/vy are missing and NOT declared unavailable.
    """
    if "vx" in frames.columns and "vy" in frames.columns:
        return frames
    if len(frames) == 0:
        # An empty frame set is not a forgotten derive_velocities() -- there is nothing to compute
        # and nothing to declare. Returned unchanged so introspection callers (VAEP column-name
        # discovery passes an EMPTY frames DataFrame) and genuinely empty matches don't raise.
        return frames
    if velocity_unavailable_by_design(frames):
        out = frames.copy()
        out["vx"] = 0.0
        out["vy"] = 0.0
        return out
    # Single-source the velocity-requiring method set from the dispatch (lazy import avoids any
    # import-order coupling; this branch is the rare caller-bug path).
    from .pitch_control._dispatch import _VELOCITY_REQUIRED_METHODS

    if method in _VELOCITY_REQUIRED_METHODS:
        raise ValueError(
            f"method={method!r} requires velocity columns ('vx', 'vy') in the tracking frame. "
            f"Call derive_velocities() (or smooth_frames()) to add them; a declared-velocity-less "
            f"provider must stamp speed_source='unavailable' on every row to get the zero-velocity "
            f"positional model, or use method='voronoi' for position-only pitch control."
        )
    return frames
