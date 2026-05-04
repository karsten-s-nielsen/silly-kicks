"""Tracking-frame preprocessing -- smoothing, interpolation, velocity derivation.

Column-naming convention
------------------------
Preprocessing utilities emit single canonical column names (``vx``, ``vy``,
``speed``, ``x_smoothed``, ``y_smoothed``) regardless of the smoothing/interpolation
method chosen. The method is recorded as a per-row ``_preprocessed_with`` provenance
column -- load-bearing because ``pandas.DataFrame.attrs`` does not propagate through
merge/concat/applyInPandas (per ``feedback_pandas_attrs_dont_propagate``).

This deliberately diverges from the convention used by VAEP feature xfns
(e.g. ``pressure_on_actor__defcon`` / ``pressure_on_actor__andrienko_cone`` per
TF-2), where suffixed names are required because parallel xfn registration would
silent-overwrite same-named columns inside ``VAEP.compute_features``.

Preprocessing has no equivalent constraint: downstream features (TF-7 pitch
control, TF-15 GK reachable area, etc.) depend on schema stability and consume
single canonical inputs. Method comparison is a separate research workflow --
call ``smooth_frames`` twice with different configs into different DataFrames
and diff.

ADR-005 §8 (PR-S25 / silly-kicks 3.2.0) formalises this asymmetry as the
multi-flavor xfn column-naming convention for VAEP-consumed feature outputs
(``<feature>__<method>`` suffixes); preprocessing utilities like
``smooth_frames`` and ``derive_velocities`` keep canonical-single-column names.

The original raw ``x`` / ``y`` columns are preserved unchanged (additive new
columns, not in-place mutation; per ``feedback_additive_columns_over_inplace_mutation``).
"""

from __future__ import annotations

from ._config import get_provider_defaults
from ._config_dataclass import PreprocessConfig
from ._interpolation import interpolate_frames
from ._smoothing import smooth_frames
from ._velocity import derive_velocities

__all__ = [
    "PreprocessConfig",
    "derive_velocities",
    "get_provider_defaults",
    "interpolate_frames",
    "smooth_frames",
]
