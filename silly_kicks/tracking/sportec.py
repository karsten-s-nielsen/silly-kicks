"""Sportec/IDSSE tracking DataFrame converter.

Converts a caller-parsed Sportec-shaped tracking DataFrame (typically
derived from DFL Position XML via ``xmltodict`` + ``pd.json_normalize`` +
roster join) into the canonical SPORTEC_TRACKING_FRAMES_COLUMNS schema.

Input contract (EXPECTED_INPUT_COLUMNS):
  - game_id (object), period_id (int), frame_id (int), time_seconds (float)
  - frame_rate (float, Hz)
  - player_id (object --- DFL PersonId --- NaN on ball rows)
  - team_id (object --- DFL TeamId --- NaN on ball rows)
  - is_ball (bool), is_goalkeeper (bool)
  - x_centered, y_centered (float, DFL meters; 0 at pitch center)
  - z (float, NaN for non-ball rows on most matches)
  - speed_native (float, m/s; populated by DFL provider)
  - ball_state (object: "alive" | "dead", from DFL BallStatus)

Coordinate transformation: ``x = x_centered + 52.5``;
``y = y_centered + 34.0``. Per-period direction flip controlled by
``home_team_start_left`` and ``home_team_start_left_extratime``.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import pandas as pd

from . import _direction
from .schema import SPORTEC_TRACKING_FRAMES_COLUMNS, TrackingConversionReport

if TYPE_CHECKING:
    from .preprocess import PreprocessConfig

EXPECTED_INPUT_COLUMNS: frozenset[str] = frozenset(
    {
        "game_id",
        "period_id",
        "frame_id",
        "time_seconds",
        "frame_rate",
        "player_id",
        "team_id",
        "is_ball",
        "is_goalkeeper",
        "x_centered",
        "y_centered",
        "z",
        "speed_native",
        "ball_state",
    }
)


_PROVIDER_NAME = "sportec"


def convert_to_frames(
    raw_frames: pd.DataFrame,
    home_team_id: str,
    home_team_start_left: bool,
    home_team_start_left_extratime: bool | None = None,
    preserve_native: list[str] | None = None,
    *,
    output_convention: Literal["absolute_frame", "ltr"] | None = None,
    preprocess: PreprocessConfig | None = None,
) -> tuple[pd.DataFrame, TrackingConversionReport]:
    """Convert Sportec-shaped raw tracking frames to canonical schema.

    Parameters
    ----------
    raw_frames : pd.DataFrame
        Sportec input (see EXPECTED_INPUT_COLUMNS).
    home_team_id : str
        DFL TeamId of the home team. Used to compute team_attacking_direction.
    home_team_start_left : bool
        From DFL match-info XML; True if home team starts attacking left.
    home_team_start_left_extratime : bool | None
        From DFL match-info XML; only required when periods 3/4 (ET) present.
    preserve_native : list[str] | None
        Reserved for future PR --- pass through optional input columns.
    output_convention : {"absolute_frame", "ltr"} | None, default None
        Coordinate convention of the returned frames. ``"absolute_frame"`` (the
        historical default behaviour) emits frames in absolute-frame-home-right
        convention with per-row ``team_attacking_direction``. ``"ltr"`` applies
        :func:`silly_kicks.tracking.utils.play_left_to_right` internally so the
        output is in canonical SPADL "all teams attack left-to-right"
        convention. Passing ``None`` (the legacy unspecified state) emits a
        ``DeprecationWarning`` and defaults to ``"absolute_frame"`` -- callers
        should pick one explicitly. See ADR-006 (silly-kicks 3.0.0).

    Returns
    -------
    frames : pd.DataFrame
        Canonical SPORTEC_TRACKING_FRAMES_COLUMNS-shaped output, 105x68 m
        SPADL coordinates, in the convention requested by ``output_convention``.
    report : TrackingConversionReport

    Examples
    --------
    Parse DFL Position XML, join roster, convert in absolute-frame::

        from silly_kicks.tracking.sportec import convert_to_frames
        # raw_frames built from DFL XML upstream
        frames, report = convert_to_frames(
            raw_frames, home_team_id="DFL-CLU-0001A",
            home_team_start_left=True,
            output_convention="absolute_frame",
        )

    Or get SPADL LTR frames directly (downstream consumers like
    ``silly_kicks.vaep.VAEP.compute_features`` accept either via the
    ``frames_convention`` kwarg)::

        frames, _ = convert_to_frames(..., output_convention="ltr")
    """
    output_convention = _resolve_output_convention(output_convention, _adapter_name="sportec")
    _ = preserve_native  # reserved
    missing = EXPECTED_INPUT_COLUMNS - set(raw_frames.columns)
    if missing:
        raise ValueError(f"sportec convert_to_frames missing columns: {sorted(missing)}")

    out = raw_frames.copy()
    out["x"] = out["x_centered"] + 52.5
    out["y"] = out["y_centered"] + 34.0

    # Periods in which the home team attacks RTL in raw input --- in those
    # periods ALL rows (player + ball) flip so the output frame is
    # home-team-attacks-LTR. Ball carries NaN direction; flip decisions
    # therefore key on the period rather than the per-row direction column.
    home_attacks_right = _direction.home_attacks_right_per_period(
        home_team_start_left=home_team_start_left,
        home_team_start_left_extratime=home_team_start_left_extratime,
    )
    home_rtl_periods = {p for p, attacks_right in home_attacks_right.items() if not attacks_right}
    flip_mask = out["period_id"].isin(home_rtl_periods).to_numpy()
    out.loc[flip_mask, "x"] = 105.0 - out.loc[flip_mask, "x"]
    out.loc[flip_mask, "y"] = 68.0 - out.loc[flip_mask, "y"]

    # Post-flip direction column: home-team rows -> "ltr"; away-team rows ->
    # "rtl"; ball rows stay NaN. Period 5 (PSO) has undefined direction and
    # so retains NaN even on player rows.
    out["team_attacking_direction"] = None
    is_player = (~out["is_ball"].astype(bool)).to_numpy(dtype=bool)
    is_home = (out["team_id"] == home_team_id).fillna(False).to_numpy(dtype=bool)
    is_known_period = out["period_id"].isin([1, 2, 3, 4]).to_numpy(dtype=bool)
    out.loc[is_player & is_home & is_known_period, "team_attacking_direction"] = "ltr"
    out.loc[is_player & ~is_home & is_known_period, "team_attacking_direction"] = "rtl"

    out["speed"] = out["speed_native"].astype("float64")
    speed_source: list[object] = ["native" if pd.notna(v) else None for v in out["speed"]]
    out["speed_source"] = pd.Series(speed_source, index=out.index, dtype="object")

    out["confidence"] = None
    out["visibility"] = None
    out["source_provider"] = "sportec"

    final = pd.DataFrame({col: out[col] for col in SPORTEC_TRACKING_FRAMES_COLUMNS})
    for col, dtype_str in SPORTEC_TRACKING_FRAMES_COLUMNS.items():
        if dtype_str == "bool":
            final[col] = final[col].astype("bool")
        elif dtype_str in {"int64", "float64"}:
            final[col] = pd.to_numeric(final[col], errors="coerce").astype(dtype_str)  # type: ignore[arg-type]
        elif dtype_str == "object":
            final[col] = final[col].astype(object)

    n_input_frames = int(raw_frames["frame_id"].nunique())
    n_periods = int(raw_frames["period_id"].nunique())
    cov: dict[int, float] = {}
    ball_out: dict[int, float] = {}
    for p, g in final.groupby("period_id", sort=True):
        expected = int(g["frame_id"].max() - g["frame_id"].min() + 1)
        actual = int(g["frame_id"].nunique())
        cov[int(p)] = float(actual) / max(expected, 1)  # type: ignore[arg-type]
        ball_g = g[g["is_ball"]]
        if len(ball_g):
            dt = 1.0 / float(ball_g["frame_rate"].iloc[0])
            ball_out[int(p)] = float((ball_g["ball_state"] == "dead").sum() * dt)  # type: ignore[arg-type]

    nan_rate = {col: float(final[col].isna().mean()) for col in final.columns}

    report = TrackingConversionReport(
        provider="sportec",
        total_input_frames=n_input_frames,
        total_output_rows=len(final),
        n_periods=n_periods,
        frame_coverage_per_period=cov,
        ball_out_seconds_per_period=ball_out,
        nan_rate_per_column=nan_rate,
        derived_speed_rows=int((final["speed_source"] == "derived").sum()),
        unrecognized_player_ids=set(),
    )

    if output_convention == "ltr":
        from .utils import play_left_to_right

        final = play_left_to_right(final, home_team_id)

    if preprocess is not None:
        from .preprocess import derive_velocities, interpolate_frames, smooth_frames
        from .preprocess._resolve import resolve_preprocess

        cfg = resolve_preprocess(preprocess, provider=_PROVIDER_NAME)
        if cfg.interpolation_method is not None:
            final = interpolate_frames(final, config=cfg)
        if cfg.smoothing_method is not None:
            final = smooth_frames(final, config=cfg)
        if cfg.derive_velocity:
            final = derive_velocities(final, config=cfg)

    return final, report


def _resolve_output_convention(
    requested: Literal["absolute_frame", "ltr"] | None,
    *,
    _adapter_name: str,
) -> Literal["absolute_frame", "ltr"]:
    """Resolve the requested output_convention; warn when unspecified.

    Per ADR-006 (silly-kicks 3.0.0), tracking adapters retain absolute-frame
    default behaviour but require callers to opt in explicitly. ``None`` (the
    legacy unspecified state) emits a DeprecationWarning and falls back to
    "absolute_frame" so behaviour is preserved.
    """
    if requested is None:
        warnings.warn(
            f"tracking.{_adapter_name}.convert_to_frames called without explicit "
            "output_convention. Defaulting to 'absolute_frame' (current behaviour). "
            "Pass output_convention='absolute_frame' to silence this warning, or "
            "'ltr' to opt into SPADL LTR output. See ADR-006 (silly-kicks 3.0.0).",
            DeprecationWarning,
            stacklevel=3,
        )
        return "absolute_frame"
    if requested not in ("absolute_frame", "ltr"):
        raise ValueError(
            f"tracking.{_adapter_name}.convert_to_frames: output_convention must be "
            f"'absolute_frame' or 'ltr', got {requested!r}"
        )
    return requested
