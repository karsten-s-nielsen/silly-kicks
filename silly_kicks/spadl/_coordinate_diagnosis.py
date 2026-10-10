"""Read-only coordinate-integrity diagnosis (SPADL frame). Pure: pandas in, frozen dataclass out.

Does NOT measure orientation/direction of play (see ``silly_kicks.tracking`` orientation tooling / the
MCP ``check_orientation`` tool) and does NOT flag legitimate off-pitch TRACKING positions (SkillCorner/SB
tracking is legitimately off-pitch; only ACTIONS are clipped to the pitch). A scale/units HEURISTIC +
bounds/NaN tripwire, not a proof of units. See
``docs/superpowers/specs/2026-10-09-agent-support-phase3-mcp-coords-and-howto-linkcheck-design.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as spadlconfig

_MIN_N = 20
_METERS_ASPECT_LO, _METERS_ASPECT_HI = 1.30, 1.80  # SPADL x/y span ratio ≈ 105/68 = 1.544
_ACTION_COLS = ("start_x", "start_y", "end_x", "end_y")
_FRAME_COLS = ("x", "y")
_NEUTRAL_SCALE = ("spadl_meters", "undetermined")


@dataclass(frozen=True)
class CoordinateDiagnosisParams:
    """Tolerances for the coordinate diagnosis (provider-neutral in v1)."""

    field_length: float = float(spadlconfig.field_length)
    field_width: float = float(spadlconfig.field_width)
    off_pitch_tol_m: float = 1.0
    gross_range_factor: float = 2.0

    @classmethod
    def for_provider(cls, provider: str) -> CoordinateDiagnosisParams:
        """v1 returns the NEUTRAL default for every provider; promotable later with no API break."""
        return cls()


@dataclass(frozen=True)
class CoordinateAxisStats:
    min: float
    p01: float
    p50: float
    p99: float
    max: float


@dataclass(frozen=True)
class CoordinateTableDiagnosis:
    table: str
    n_rows: int
    x: CoordinateAxisStats
    y: CoordinateAxisStats
    inferred_scale: str
    out_of_pitch_fraction: float
    gross_out_of_range_fraction: float
    coord_nan_fraction: float  # fraction of rows with ANY NaN coord (INFO)
    all_coords_nan: bool  # every diagnosed coord cell is NaN (the coords_all_nan predicate)


@dataclass(frozen=True)
class CoordinateDiagnosis:
    actions: CoordinateTableDiagnosis | None
    frames: CoordinateTableDiagnosis | None
    flags: list[str]
    notes: list[str]


_DEFAULT_PARAMS = CoordinateDiagnosisParams()  # frozen singleton -> safe as a default arg


def _axis_stats(a: np.ndarray) -> CoordinateAxisStats:
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        nan = float("nan")
        return CoordinateAxisStats(nan, nan, nan, nan, nan)
    p01, p50, p99 = (float(v) for v in np.percentile(finite, [1, 50, 99]))
    return CoordinateAxisStats(float(finite.min()), p01, p50, p99, float(finite.max()))


def _classify(xs: CoordinateAxisStats, ys: CoordinateAxisStats, n_finite: int) -> str:
    if n_finite < _MIN_N or not np.isfinite(xs.p99) or not np.isfinite(ys.p99):
        return "undetermined"
    xr, yr = xs.p99 - xs.p01, ys.p99 - ys.p01
    mag = max(xs.p99, ys.p99)
    aspect = xr / yr if yr > 1e-9 else float("inf")
    if mag <= 1.5:
        return "normalized_0_1"
    if _METERS_ASPECT_LO <= aspect <= _METERS_ASPECT_HI and 40.0 <= mag <= 150.0:
        return "spadl_meters"
    if mag <= 110.0:
        return "scale_0_100"
    return "suspect"


def _table_diag(
    df: pd.DataFrame, cols: tuple[str, ...], table: str, p: CoordinateDiagnosisParams
) -> CoordinateTableDiagnosis:
    present = [c for c in cols if c in df.columns]
    xcols = [c for c in present if c.endswith("x")]
    ycols = [c for c in present if c.endswith("y")]
    xv = df[xcols].to_numpy(dtype="float64", copy=True).ravel() if xcols else np.array([])
    yv = df[ycols].to_numpy(dtype="float64", copy=True).ravel() if ycols else np.array([])
    xs, ys = _axis_stats(xv), _axis_stats(yv)
    n_finite = int(min(np.isfinite(xv).sum(), np.isfinite(yv).sum())) if xcols and ycols else 0
    coord = df[present].to_numpy(dtype="float64", copy=True) if present else np.empty((len(df), 0))
    nan_frac = float(np.isnan(coord).any(axis=1).mean()) if coord.size and len(df) else 0.0
    all_nan = bool(coord.size and np.isnan(coord).all())  # true all-coords-NaN (NOT any-per-row)

    def _oob(only_gross: bool) -> float:
        if not (xcols and ycols) or not len(df):
            return 0.0
        xo = df[xcols].to_numpy(dtype="float64", copy=True)
        yo = df[ycols].to_numpy(dtype="float64", copy=True)
        if only_gross:
            bad = (np.abs(xo) > p.gross_range_factor * p.field_length).any(axis=1) | (
                np.abs(yo) > p.gross_range_factor * p.field_width
            ).any(axis=1)
        else:
            bad = ((xo < -p.off_pitch_tol_m) | (xo > p.field_length + p.off_pitch_tol_m)).any(axis=1) | (
                (yo < -p.off_pitch_tol_m) | (yo > p.field_width + p.off_pitch_tol_m)
            ).any(axis=1)
        return float(np.nanmean(bad.astype("float64")))

    return CoordinateTableDiagnosis(
        table=table,
        n_rows=len(df),
        x=xs,
        y=ys,
        inferred_scale=_classify(xs, ys, n_finite),
        out_of_pitch_fraction=_oob(False),
        gross_out_of_range_fraction=_oob(True),
        coord_nan_fraction=nan_frac,
        all_coords_nan=all_nan,
    )


def diagnose_coordinates(
    actions: pd.DataFrame | None,
    frames: pd.DataFrame | None,
    *,
    params: CoordinateDiagnosisParams = _DEFAULT_PARAMS,
) -> CoordinateDiagnosis:
    """Diagnose coordinate integrity of one match's SPADL actions and/or tracking frames.

    Read-only, pure. Raises if both ``actions`` and ``frames`` are ``None``.

    >>> import pandas as pd
    >>> a = pd.DataFrame(
    ...     {"start_x": [10.0, 50.0, 90.0], "start_y": [10.0, 34.0, 60.0],
    ...      "end_x": [20.0, 55.0, 95.0], "end_y": [12.0, 30.0, 64.0]}
    ... )
    >>> diagnose_coordinates(a, None).flags
    []
    """
    if actions is None and frames is None:
        raise ValueError("diagnose_coordinates needs at least one of actions/frames")
    a = _table_diag(actions, _ACTION_COLS, "actions", params) if actions is not None else None
    f = _table_diag(frames, _FRAME_COLS, "frames", params) if frames is not None else None
    flags: list[str] = []
    if (a and a.inferred_scale not in _NEUTRAL_SCALE) or (f and f.inferred_scale not in _NEUTRAL_SCALE):
        flags.append("coords_scale_suspect")
    if a and a.out_of_pitch_fraction > 0.0:
        flags.append("actions_out_of_pitch")
    if (a and a.gross_out_of_range_fraction > 0.0) or (f and f.gross_out_of_range_fraction > 0.0):
        flags.append("coords_gross_out_of_range")
    if (a and a.all_coords_nan) or (f and f.all_coords_nan):
        flags.append("coords_all_nan")
    if actions is not None and len(actions):
        _sx = actions[["start_x", "start_y"]].to_numpy(dtype="float64", copy=True)
        if float(np.isnan(_sx).any(axis=1).mean()) > 0.0:  # START coords ONLY (never end)
            flags.append("actions_start_nan")
    notes = [
        "Does NOT measure orientation/direction of play (see check_orientation).",
        "Frames out_of_pitch_fraction is INFO: tracking is legitimately off-pitch; only actions-off-pitch flags.",
        "Scale/units is a heuristic tripwire (aspect-ratio + magnitude), not a proof.",
    ]
    return CoordinateDiagnosis(actions=a, frames=f, flags=flags, notes=notes)
